import signal
import sys
import os
import time
import shutil
from typing import Union
import platform 
import gc

from psutil import virtual_memory, cpu_count
import numpy as np
from torch.utils.data import DataLoader
import torch 
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import PreTrainedTokenizerFast
from torch_optimizer import Adafactor

# import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.logger import Logger
from model.dataset import LowMemDataset
from config import TrainConfig, TrainConfigSFT, T5ModelConfig
from utils.functions import (
    get_bleu4_score, 
    save_model_config, 
    get_free_space_of_disk, 
    my_average,
    get_path_of_suffix_files,
    get_T5_config,
)

class ChatTrainerLowMem:
    """
    低内存版本的训练器，针对16G内存环境优化
    
    主要优化：
    1. 强制关闭数据集内存缓存（keep_in_memory=False）
    2. 使用更小的batch_size（默认1-2）
    3. 增加梯度累积步数来补偿小batch size
    4. 禁用DataLoader的num_workers，避免多进程内存开销
    5. 定期清理GPU和CPU缓存
    6. 减小评估batch_size
    """
    def __init__(self, train_config: Union[TrainConfig, TrainConfigSFT], model_config: T5ModelConfig, ) -> None:
        
        self.train_config = train_config
        self.model_config = model_config

        # file_name=None会自动生成以当前日期命名的log文件名
        self.logger = Logger('chat_trainer_low_mem', std_out=True, save2file=True, file_name=None)

        self.model = None
        self.accelerator = None

        signal.signal(signal.SIGINT, self.process_exit_handler)

        self.is_win_platform = True if platform.system().lower() == 'windows' else False

        torch.manual_seed(train_config.seed)
        torch.cuda.manual_seed_all(train_config.seed)
    
    def log_memory_usage(self, stage: str = "") -> None:
        """记录当前内存使用情况"""
        mem = virtual_memory()
        used_gb = (mem.total - mem.available) / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        percent = mem.percent
        
        msg = f"[内存监控{(' - ' + stage) if stage else ''}] 已用: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)"
        self.logger.info(msg, save_to_file=True)
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                self.logger.info(f"  GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB", save_to_file=True)
    
    def clear_memory(self) -> None:
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_exit_handler(self, signal_received, frame) -> None:
        '''
        进程退出时的操作，保存模型
        '''
        if self.accelerator and self.model:
            ask = "you are pressed `ctrl+c`,  do you want to save checkpoint? Yes (y) or No (n)"
            self.accelerator.print(ask)
            ins = input()
            
            if ins.lower() in ('yes', 'y'):

                suffix =  'exit_save_{}'.format(str(time.strftime('%Y%m%d%H%M%S', time.localtime())))

                self.accelerator.wait_for_everyone()
                self.accelerator.save_state(output_dir=self.train_config.train_state_dir)

                self.accelerator.print('model ckeck point has been saved in {}'.format(self.train_config.train_state_dir))
        
            sys.exit(0)
        else:
            print('process not in trainingg, exit.')
            sys.exit(0)

    def save_model(self, suffix: Union[str, int]) -> None:
        '''保存模型到文件
        注意：save_model不能放到is_main_process里面
        e.g:
        >>> self.save_model(epoch) # 在这里使用
        >>> if accelerator.is_main_process:
        >>>     do_somthing()
        '''
        if self.model and self.accelerator:

            # 先wait_for_everyone，再保存
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                save_path = self.train_config.model_file.format(suffix)
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

                # 先做一次清理，避免磁盘被 checkpoint 堆满（本项目单个 .bin ~ 0.7-0.8GB）
                self._cleanup_checkpoints(keep_latest_n=self.train_config.keep_latest_n_ckp)

                # 磁盘空间检查：至少预留 2GB，避免写到一半失败（你遇到的就是磁盘 100% 导致写失败）
                try:
                    free_gb = shutil.disk_usage(save_dir or ".").free / (1024 ** 3)
                except Exception:
                    free_gb = 0.0

                if free_gb < 2.0:
                    self.accelerator.print(
                        f"[WARN] 磁盘剩余空间不足({free_gb:.2f}GB)，跳过保存模型: {save_path}"
                    )
                    return

                unwrap_model = self.accelerator.unwrap_model(self.model)
                model_dict = self.accelerator.get_state_dict(unwrap_model)

                # 写入失败时常见报错：PytorchStreamWriter failed writing file / unexpected pos
                # 通常是磁盘满或底层文件系统 I/O 异常。这里做两层防护：
                # 1) 使用旧版序列化，避免 zip writer 在部分文件系统上更容易触发 unexpected pos
                # 2) 保存失败时删除不完整文件，避免下次继续把磁盘占满
                try:
                    torch.save(
                        model_dict,
                        save_path,
                        _use_new_zipfile_serialization=False,
                    )
                except Exception as e:
                    # 尝试删除半成品
                    try:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                    except Exception:
                        pass
                    raise e

    def _cleanup_checkpoints(self, keep_latest_n: int = 8) -> None:
        """
        清理旧 checkpoint，避免磁盘空间被写满。

        规则：
        - 保留 `chat_small_t5.best.bin`
        - 保留最近 keep_latest_n 个 `chat_small_t5.epoch_*_latest.bin`
        - 保留包含 `exit_save` 的应急保存
        """
        try:
            model_file_tpl = self.train_config.model_file.replace("\\", "/")
            model_dir = "/".join(model_file_tpl.split("/")[:-1]) or "."
            if not os.path.isdir(model_dir):
                return

            best_path = self.train_config.model_file.format("best")
            keep = set([best_path])

            candidates = []
            for fn in os.listdir(model_dir):
                if not fn.endswith(".bin"):
                    continue
                full = os.path.join(model_dir, fn)
                if "exit_save" in fn:
                    keep.add(full)
                    continue
                # 只清理 epoch latest 文件，避免误删其它产物
                if ".epoch_" in fn and fn.endswith("_latest.bin"):
                    candidates.append(full)

            # 按修改时间从新到旧排序
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            keep.update(candidates[:keep_latest_n])

            for p in candidates[keep_latest_n:]:
                if p not in keep and os.path.exists(p):
                    os.remove(p)
        except Exception:
            # 清理失败不影响训练
            return

    
    def delete_early_checkpoint(self, epoch: int, keep_latest_n: int=3,) -> None:
        '''
        删除最早的模型，最保留最近keep_latest_n个模型文件
        '''
        model_save_path = self.train_config.model_file
        model_save_path = model_save_path.replace('\\', '/')    # 针对win的路径，将\\替换为/
        model_save_path = '/'.join(model_save_path.split('/')[0: -1])   # 删除末尾文件名后缀
        
        model_files = get_path_of_suffix_files(model_save_path, suffix='.bin', with_create_time=True)
        
        # 进程异常退出保存模型文件不在删除范围
        train_save_model_fils = []
        for item in model_files:
            if 'exit_save' not in item[0]:

                # 大于当前epoch的文件不不删除
                f_epoch = int(item[0].split('.')[-2])
                if epoch >= f_epoch:
                    print(epoch, f_epoch, item)
                    train_save_model_fils.append(item)

        train_save_model_fils.sort(key=lambda x: x[1])  # 按照时间从小到大排序

        if len(train_save_model_fils) <= keep_latest_n:
            return
        
        to_delete_files = train_save_model_fils[0: -keep_latest_n]
        for item in to_delete_files:
            os.remove(item[0])

    
    def train(self, is_keep_training: bool=False, is_finetune: bool=False) -> None:
        '''
        低内存版本的训练函数
        
        is_keep_training: 是否从断点处加载状态继续训练
        is_finetune: 是否微调，微调的话可能需要冻结部分参数
        '''
        log = self.logger
        train_config = self.train_config
        save_steps = self.train_config.save_steps
        logging_steps = self.train_config.logging_steps

        # 【关键优化1】梯度累计步数：平衡内存和训练效果
        # 根据可用内存和GPU数量智能调整梯度累积步数
        unuse_mem_gb = virtual_memory().available / (1024 ** 3)
        num_gpus = torch.cuda.device_count()
        
        # 🚀 新策略：在保证GPU显存占用的前提下，优先降低内存使用
        # - 可用内存<10GB：使用大梯度累积（4），减少内存占用
        # - 可用内存>=10GB：使用配置的梯度累积（2），充分利用GPU显存
        if unuse_mem_gb < 10:
            # 低内存模式：增大梯度累积，减少内存占用
            # 虽然梯度累积大了，但batch_size也会相应增大，GPU显存占用不变
            accumulation_steps = 4
        else:
            # 内存充足（>=10GB）：使用配置的梯度累积，充分利用GPU显存
            accumulation_steps = train_config.gradient_accumulation_steps

        set_seed(train_config.seed)

        accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,       # 混合精度
            gradient_accumulation_steps=accumulation_steps,     # 梯度累积
            project_dir=train_config.train_state_dir,
        )

        # 【重要】所有进程都需要获取内存信息，不能只在主进程中定义
        unuse_mem = virtual_memory().available / (1024 ** 3)  # 单位：GB
        unuse_disk = get_free_space_of_disk('./')

        if accelerator.is_main_process:
            log.info('=' * 80, save_to_file=True)
            log.info('低内存模式训练 - 针对16G内存优化', save_to_file=True)
            log.info('=' * 80, save_to_file=True)
            log.info('cpu memory available: {:.2f} GB, disk space available: {:.2f} GB'.format(unuse_mem, unuse_disk), save_to_file=True)
            log.info('使用LowMemDataset: 支持多GPU + 低内存模式，按需从磁盘读取数据', save_to_file=True)
            log.info('gradient accumulation steps: {} (增加以补偿小batch size)'.format(accumulation_steps), save_to_file=True)
            log.info('operation: {}, keep training: {}, loading datasets ...'.format('finetune' if is_finetune else 'train', is_keep_training))
            
            self.log_memory_usage("训练开始前")
            
            # 验证数据文件是否存在
            if not os.path.exists(train_config.train_file):
                raise FileNotFoundError(
                    f'训练数据文件不存在: {train_config.train_file}\n'
                    f'请先运行: python prepare_sft_data.py 生成训练数据'
                )
            if not os.path.exists(train_config.validation_file):
                raise FileNotFoundError(
                    f'验证数据文件不存在: {train_config.validation_file}\n'
                    f'请先运行: python prepare_sft_data.py 生成训练数据'
                )
            
            # 如果是微调，验证预训练模型是否存在
            if is_finetune and not os.path.exists(train_config.finetune_from_ckp_file):
                raise FileNotFoundError(
                    f'预训练模型文件不存在: {train_config.finetune_from_ckp_file}\n'
                    f'请先完成预训练，或检查config.py中的finetune_from_ckp_file路径是否正确\n'
                    f'预训练命令: accelerate launch --multi_gpu --num_processes 2 ./train.py train'
                )

        # 【关键优化3】根据内存情况动态调整num_workers
        # 🚀 关键：num_workers=0 可以节省 2-4GB 内存（避免多进程复制数据）
        # 但会降低数据加载速度，需要通过增大 batch_size 来补偿
        if unuse_mem < 10:
            # 低内存（<10GB）：强制禁用多进程，节省 2-4GB 内存
            num_workers = 0
        else:
            # 内存充足（>=10GB）：启用少量多进程加速
            # 注意：每个 worker 会复制一份数据，占用额外内存
            gpu_cnt = torch.cuda.device_count()
            # 🚀 优化：减少 worker 数量（从 8 降到 4），节省内存
            num_workers = min(4, int(1 * gpu_cnt)) if gpu_cnt > 0 else 0

        # 使用LowMemDataset，支持多GPU + 低内存模式
        # ultra_low_mem=True: 每次读取时重新打开文件，避免PyArrow缓存累积
        # 这会稍微降低速度，但能显著减少内存占用（节省 5-8GB）
        # 🚀 关键优化：降低阈值到 10GB，更激进地启用超低内存模式
        use_ultra_low_mem = unuse_mem < 10  # 可用内存<10GB时启用超低内存模式
        
        if accelerator.is_main_process:
            log.info(f'ultra_low_mem模式: {use_ultra_low_mem} (可用内存: {unuse_mem:.2f}GB)', save_to_file=True)
            if use_ultra_low_mem:
                log.info('  ⚠️  超低内存模式会降低数据加载速度，但可节省 5-8GB 内存', save_to_file=True)
        
        train_dataset = LowMemDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            max_seq_len=train_config.max_seq_len,
            ultra_low_mem=use_ultra_low_mem,
        )
        valid_dataset = LowMemDataset(
            parquet_file=train_config.validation_file,
            tokenizer_dir=train_config.tokenizer_dir,
            max_seq_len=train_config.max_seq_len,
            ultra_low_mem=use_ultra_low_mem,
        )

        # 【关键优化4】根据可用内存动态调整batch_size
        # 🚀 新策略：在低内存情况下，增大 batch_size 来补偿 num_workers=0 的速度损失
        # 原理：num_workers=0 节省了 2-4GB 内存，可以用来增大 batch_size
        if unuse_mem < 10:
            # 低内存（<10GB）：使用配置的 batch_size，充分利用 GPU 显存
            # 虽然内存紧张，但通过 ultra_low_mem=True + num_workers=0 节省了内存
            batch_size = train_config.batch_size_per_gpu
            eval_batch_size = batch_size * 2
            if accelerator.is_main_process:
                log.info(f'⚠️  低内存模式（可用内存<10GB），使用 batch_size={batch_size}', save_to_file=True)
                log.info(f'  通过 ultra_low_mem=True + num_workers=0 节省内存，保持 GPU 显存占用', save_to_file=True)
        else:
            # 内存充足（>=10GB）：使用配置的batch_size，充分利用GPU显存
            batch_size = train_config.batch_size_per_gpu
            eval_batch_size = batch_size * 2
            if accelerator.is_main_process:
                log.info(f'✅ 内存充足（可用内存>=10GB），使用配置的batch_size={batch_size}', save_to_file=True)

        if accelerator.is_main_process:
            log.info(f'batch_size_per_gpu: {batch_size} (原配置: {train_config.batch_size_per_gpu})', save_to_file=True)
            log.info(f'eval_batch_size: {eval_batch_size}', save_to_file=True)

        # 根据内存情况决定是否启用pin_memory
        # 🚀 关键：pin_memory 会占用额外内存（约 1-2GB），低内存时禁用
        use_pin_memory = unuse_mem >= 10  # 内存充足时启用pin_memory加速GPU传输
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,  
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            pin_memory=use_pin_memory,  # 内存充足时启用
            num_workers=num_workers,
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=use_pin_memory,  # 内存充足时启用
            num_workers=num_workers,
        )

        device = accelerator.device
        log.info('using device: {} '.format(str(device)), save_to_file=True)
        

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = train_dataset.tokenizer
        decoder_start_token_id = tokenizer.pad_token_id

        # for t5, set decoder_start_token_id = pad_token_id
        t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=decoder_start_token_id, eos_token_id=tokenizer.eos_token_id)

        model = TextToTextModel(t5_config)

        # 微调加载的模型并冻结embedding和encoder
        if is_finetune:
            if accelerator.is_main_process:
                log.info(f'加载预训练模型: {train_config.finetune_from_ckp_file}', save_to_file=True)
            
            model.load_state_dict(torch.load(train_config.finetune_from_ckp_file, map_location='cpu'))
            
            # 冻结embedding和encoder，只训练decoder
            layers_to_freeze = [model.shared, model.encoder]
            
            trainable_params = 0
            total_params = 0
            
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            
            # 统计可训练参数
            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            if accelerator.is_main_process:
                log.info(f'SFT微调: 冻结embedding和encoder，只训练decoder', save_to_file=True)
                log.info(f'可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({100*trainable_params/total_params:.2f}%)', save_to_file=True)

        # 保存模型配置，方便修改配置后恢复
        save_model_config(t5_config.to_diff_dict(), train_config.model_config_file)
        
        # T5训练，论文推荐使用Adafactor
        optimizer = Adafactor(params=model.parameters(), lr=train_config.learn_rate)

        
        # 获取当前机器有多少个GPU，默认全部使用
        num_gpus_used = accelerator.state.num_processes

        # 单机多卡，每个step总共的batch_size = batch_size_per_gpu * num_gpus_used
        # total_batch_size 初始化为batch_size_per_gpu真的只有CPU的情况
        total_batch_size = batch_size
        total_eval_batch_size = eval_batch_size
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * batch_size
            total_eval_batch_size = num_gpus_used * eval_batch_size

        steps_per_epoch = int(np.ceil(len(train_dataset) // total_batch_size))
        eval_steps = int(np.ceil(len(valid_dataset) // total_eval_batch_size))

        if accelerator.is_main_process:
            log.info('train dataset size: {}, steps per epoch:{}; validation dataset size: {}, steps per validation: {}; datalodater num_workers: {}.'\
                    .format(len(train_dataset), steps_per_epoch, len(valid_dataset), eval_steps, num_workers), save_to_file=True)
            self.log_memory_usage("数据集加载后")

        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, 
                max_lr=train_config.div_factor * train_config.learn_rate, 
                epochs=train_config.epochs, 
                steps_per_epoch=int(np.ceil( len(train_dataset) / (batch_size * accumulation_steps) )),  # 梯度累积相当于增大了batch_size
                div_factor=train_config.div_factor,
                cycle_momentum=False,
            )
        
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
                model, 
                optimizer,
                lr_scheduler, 
                train_dataloader, 
                valid_dataloader,
            )
        
        if is_keep_training:
            accelerator.load_state(input_dir=train_config.train_state_dir)
            accelerator.register_for_checkpointing(lr_scheduler)
        
        self.model = model
        self.accelerator = accelerator
        
        if accelerator.is_main_process:
            self.log_memory_usage("模型加载后")
        
        best_bleu4 = 0.0
        best_epoch = 0
        epoch_loss_list = []

        # 添加进度条，只在主进程更新
        if accelerator.is_main_process:
            progress = Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[bold blue]{task.fields[show_info]}"),
                refresh_per_second=1,  # 每1秒钟更新一次，不要频繁更新
                )
            
            epoch_progress = progress.add_task(description='epoch: ', show_info='', total=train_config.epochs)
            steps_progress = progress.add_task(description='steps: ', show_info='', \
                                                total=np.ceil(steps_per_epoch / logging_steps))
            eval_progress = progress.add_task(description='evaluate: ', show_info='', total=eval_steps, visible=False)

            self.progress = progress
            self.eval_progress = eval_progress

            progress.start()

        # end if

        for epoch in range(train_config.epochs):
            
            if accelerator.is_main_process:
                epoch_show_txt = 'epoch: {}/{}, avg_loss: {:.6f}, best_epoch: {}, best_bleu: {}'.format(
                    epoch, train_config.epochs, my_average(epoch_loss_list), best_epoch, best_bleu4
                )
                progress.update(epoch_progress, show_info=epoch_show_txt)
                progress.reset(steps_progress)

            epoch_loss_list = []
            model.train()

            # 【关键优化5】每个epoch开始前清理内存
            self.clear_memory()

            for step, batch_data in enumerate(train_dataloader):

                input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                target_ids = batch_data['target_ids']
                # for t5 model, all labels set to `-100` are ignored (masked)
                target_ids[target_ids == decoder_start_token_id] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=target_ids,
                )

                loss = outputs.loss.mean() / accumulation_steps

                # attention here! loss.backward()
                accelerator.backward(loss) 

                # 梯度累计
                if (step + 1) % accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # 每隔save_steps步保存一次模型
                if (step + 1) % save_steps == 0 or step == steps_per_epoch:
                    self.save_model('epoch_{}_latest'.format(epoch))
                    accelerator.save_state(output_dir=train_config.train_state_dir)
                
                # ==================================以下记录loss到日志============================================
                # 每n步更新一次，避免频繁的cpu-gpu数据复制
                # 参考：https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization
                
                if step % logging_steps == 0 or step == steps_per_epoch:
                    
                    loss_cpu = loss.detach().item() * accumulation_steps
                    epoch_loss_list.append(loss_cpu)
                    
                    info_txt = 'training loss: epoch:{}, step:{}, loss:{}, device:{}'.\
                        format(epoch, step, loss_cpu, str(accelerator.device))
                    
                    log.info(info_txt, std_out=False, save_to_file=True) # 保存 loss 到文件

                    # 更新进度条
                    if accelerator.is_main_process:
                        step_show_txt = 'step: {}/{}, loss: {:.6f}'.format(step, steps_per_epoch, loss_cpu)
                        progress.advance(steps_progress, advance=1)
                        progress.update(steps_progress, show_info=step_show_txt)

                # ==================================以上记录loss到日志============================================
                
                # 【关键优化6】定期清理内存
                # 更频繁地清理内存，每50步清理一次
                if step % 50 == 0:
                    self.clear_memory()
                
                # 每200步强制清理并记录内存使用
                if step % 200 == 0 and accelerator.is_main_process:
                    self.clear_memory()
                    self.log_memory_usage(f"Epoch {epoch} Step {step}")
            
            #  end for batch setps

            # 等所有训练进程完成再开始评估
            accelerator.wait_for_everyone()

            # 【关键优化7】评估前清理内存
            self.clear_memory()

            model.eval()         
            
            cur_bleu4_score = self.evaluate(
                model=model,
                tokenizer=tokenizer,
                valid_dataloader=valid_dataloader,
                accelerator=accelerator,
                eval_steps=eval_steps,
                )

            # save model
            if cur_bleu4_score >= best_bleu4:

                best_bleu4 = cur_bleu4_score
                best_epoch = epoch
                self.save_model('best')
                accelerator.save_state(output_dir=train_config.train_state_dir)

            # 每个epoch打印一下日志
            if accelerator.is_main_process:

                progress.advance(epoch_progress, advance=1)
                info_txt = 'epoch log: epoch:{}, avg_loss:{}, cur_bleu4:{}, best_bleu4:{}, best_epoch:{}'.\
                            format(epoch, my_average(epoch_loss_list), cur_bleu4_score, best_bleu4, best_epoch)
                self.print_and_log(info_txt, accelerator)
                self.log_memory_usage(f"Epoch {epoch} 结束")


    def evaluate(self, 
                model: TextToTextModel, 
                tokenizer: PreTrainedTokenizerFast,
                valid_dataloader: DataLoader, 
                accelerator: Accelerator,
                eval_steps: int,
            ) -> float:
        
        '''
        评估，返回平均的bleu分数
        '''
        max_seq_len = self.train_config.max_seq_len
        batch_decode = tokenizer.batch_decode

        local_sum = 0.0
        local_cnt = 0

        if accelerator.is_main_process:
            self.progress.reset(self.eval_progress)
            self.progress.update(self.eval_progress, visible=True)

        max_eval_steps = eval_steps

        with torch.no_grad():
            for step, batch_data in enumerate(valid_dataloader):
                
                if step >= max_eval_steps: break
                
                if accelerator.is_main_process:
                    self.progress.advance(self.eval_progress, advance=1)
                    self.progress.update(self.eval_progress, show_info='step: {}/{}'.format(step, eval_steps))

                input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                target_ids = batch_data['target_ids']

                # 使用greedy search替代beam search，速度提升5倍以上
                outputs = accelerator.unwrap_model(model).my_generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                    search_type='greedy',  # 评估时使用greedy search，速度快很多
                )

                # 各 rank 只处理自己的 shard（不做 gather）
                outputs = outputs.detach().cpu().numpy()
                target_ids = target_ids.detach().cpu().numpy()

                outputs_txt = batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_txt = batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # 逐样本计算 bleu4，并在本地累加
                for i in range(len(target_txt)):
                    local_sum += float(get_bleu4_score(reference=target_txt[i], outputs=outputs_txt[i]))
                    local_cnt += 1

        # 汇总各 rank 的 sum / count
        device = accelerator.device
        local_sum_t = torch.tensor(local_sum, device=device, dtype=torch.float32)
        local_cnt_t = torch.tensor(local_cnt, device=device, dtype=torch.float32)
        global_sum_t = accelerator.reduce(local_sum_t, reduction="sum")
        global_cnt_t = accelerator.reduce(local_cnt_t, reduction="sum")
        avg_bleu4_score = (global_sum_t / torch.clamp(global_cnt_t, min=1.0)).item()

        if accelerator.is_main_process:
            self.progress.update(self.eval_progress, show_info='bleu4 score: {}'.format(avg_bleu4_score))
            self.progress.update(self.eval_progress, visible=False)

        return avg_bleu4_score

    def test(self, best_epoch: int=0) -> None:
        '''
        '''
        import os 

        train_config = self.train_config
        log = self.logger

        # args for dataloader
        num_workers = 0  # 低内存模式强制0

        test_dataset = LowMemDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            max_seq_len=train_config.max_seq_len,
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=min(train_config.batch_size_per_gpu, 2),  # 最大2
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            pin_memory=False,
            num_workers=num_workers,
        )

        log.info('test dataset size: {}.'.format(len(test_dataset)), save_to_file=True)

        set_seed(train_config.seed)
        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        log.info('using device: {} '.format(str(device)), save_to_file=True)

         # 获取当前运行使用了多少个GPU
        num_gpus_used = accelerator.state.num_processes

        # 单机多卡，每个step总共的batch_size = batch_size_per_gpu * num_gpus_used
        # total_batch_size 初始化为batch_size_per_gpu真的只有CPU的情况
        total_batch_size = min(train_config.batch_size_per_gpu, 2)
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * min(train_config.batch_size_per_gpu, 2)

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = test_dataset.tokenizer

        model_file = train_config.model_file.format(best_epoch)
        if os.path.isdir(model_file):
            # 传入文件夹则 from_pretrained
            model = TextToTextModel.from_pretrained(model_file)
        else:
            # load_state_dict
            t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            model = TextToTextModel(t5_config)
            model.load_state_dict(torch.load(model_file, map_location='cpu')) # set cpu for no exception
       
        model, test_dataloader = accelerator.prepare(
                model, 
                test_dataloader,
            )
        
        steps = int(np.ceil(len(test_dataset) // total_batch_size))

        local_sum = 0.0
        local_cnt = 0
        batch_decode = tokenizer.batch_decode
        max_seq_len = self.train_config.max_seq_len
        model.eval()

        if accelerator.is_main_process:
            progress = Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[bold blue]{task.fields[show_info]}"),
                refresh_per_second=1.0,
                )
                
            steps_progress = progress.add_task(description='steps: ', show_info='', total=steps)
            progress.start()
            
        with torch.no_grad():
            for step, batch_data in enumerate(test_dataloader):

                if accelerator.is_main_process:
                    progress.advance(steps_progress, advance=1)
                    progress.update(steps_progress, show_info='step: {}/{}'.format(step, steps))

                input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                target_ids = batch_data['target_ids']

                outputs = accelerator.unwrap_model(model).my_generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                )

                # 测试阶段同理：不做每步 gather，避免不同 rank 速度差导致 NCCL 超时
                outputs = outputs.detach().cpu().numpy()
                target_ids = target_ids.detach().cpu().numpy()
                
                outputs_txt = batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_txt = batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                for i in range(len(target_txt)):
                    local_sum += float(get_bleu4_score(reference=target_txt[i], outputs=outputs_txt[i]))
                    local_cnt += 1

        device = accelerator.device
        local_sum_t = torch.tensor(local_sum, device=device, dtype=torch.float32)
        local_cnt_t = torch.tensor(local_cnt, device=device, dtype=torch.float32)
        global_sum_t = accelerator.reduce(local_sum_t, reduction="sum")
        global_cnt_t = accelerator.reduce(local_cnt_t, reduction="sum")
        avg_bleu4_score = (global_sum_t / torch.clamp(global_cnt_t, min=1.0)).item()
        if accelerator.is_main_process:
            progress.update(steps_progress, show_info='bleu4 score: {}'.format(avg_bleu4_score))

        info_txt = 'test_dataset_size: {}, avg_bleu4_score:{}.'.format(len(test_dataset), avg_bleu4_score)
        log.info(info_txt, save_to_file=True)

        return avg_bleu4_score

    
    def print_and_log(self, info: str, accelerator: Accelerator=None) -> None:
        '''
        使用accelerator.print, 否则多进程打印会异常
        '''
        if not accelerator:
            print(info)
        else:
            accelerator.print(info)
        self.logger.info(info, std_out=False, save_to_file=True)

if __name__ == '__main__':
    
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainerLowMem(train_config=train_config, model_config=model_config)

    chat_trainer.train()
