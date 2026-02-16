import signal
import sys
import os
import time
import shutil
from typing import Union
import platform 

from psutil import virtual_memory, cpu_count
import numpy as np
from torch.utils.data import DataLoader
import torch 
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import PreTrainedTokenizerFast
from torch_optimizer import Adafactor

# import accelerate
import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs

# import 自定义类和函数
from model.chat_model import TextToTextModel
from utils.logger import Logger
from model.dataset import MyDataset
from config import TrainConfig, TrainConfigSFT, T5ModelConfig
from utils.functions import (
    get_bleu4_score, 
    save_model_config, 
    get_free_space_of_disk, 
    my_average,
    get_path_of_suffix_files,
    get_T5_config,
)

class ChatTrainer:
    def __init__(self, train_config: Union[TrainConfig, TrainConfigSFT], model_config: T5ModelConfig, ) -> None:
        
        self.train_config = train_config
        self.model_config = model_config

        # file_name=None会自动生成以当前日期命名的log文件名
        self.logger = Logger('chat_trainer', std_out=True, save2file=True, file_name=None)

        self.model = None
        self.accelerator = None

        signal.signal(signal.SIGINT, self.process_exit_handler)

        self.is_win_platform = True if platform.system().lower() == 'windows' else False

        torch.manual_seed(train_config.seed)
        torch.cuda.manual_seed_all(train_config.seed)
    
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
        model_save_path = model_save_path.replace('\\', '/')    # 针对win的路径，将\替换为/
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
        is_keep_training: 是否从断点处加载状态继续训练
        is_finetune: 是否微调，微调的话可能需要冻结部分参数
        '''
        log = self.logger
        train_config = self.train_config
        save_steps = self.train_config.save_steps
        logging_steps = self.train_config.logging_steps

        # 梯度累计的步数
        accumulation_steps = train_config.gradient_accumulation_steps

        set_seed(train_config.seed)

        # Increase NCCL timeout to 1 hour to prevent evaluate phase timeout
        nccl_timeout_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))

        accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,       # 混合精度
            gradient_accumulation_steps=accumulation_steps,     # 梯度累积
            project_dir=train_config.train_state_dir,
            kwargs_handlers=[nccl_timeout_kwargs],
        )

        # 根据剩余内存大小决定是否完全加载数据集到内存中
        unuse_mem = virtual_memory().available / (1024 ** 3)  # 单位：GB
        unuse_disk = get_free_space_of_disk('./')

        # 剩余内存≥48GB将把数据集留在内存中,因为2个显卡+全全部装载900多万的训练数据到内存需要大概43GB的CPU内存
        # 如果不放在内存中，将会使用迭代器生成数据，CPU 内存小于16GB也可以运行，但是不支持顺序打乱。
        # Force keep_in_memory=False to prevent memory from growing continuously.
        # When True, pandas DataFrame will be loaded into memory and may cause
        # slow memory leak due to DDP multi-process copy-on-write dirtying pages.
        # Arrow-based access (keep_in_memory=False) uses memory-mapped I/O with
        # stable memory footprint.
        keep_in_memory = False

        if accelerator.is_main_process:
            log.info('cpu memory available: {:.2f} GB, disk space available: {:.2f} GB, keep dataset in memory: {}.'\
                    .format(unuse_mem, unuse_disk, keep_in_memory), save_to_file=True)
            log.info('operation: {}, keep training: {}, loading datasets ...'.format('finetune' if is_finetune else 'train', is_keep_training))
            
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

        # args for dataloader
        # 启用num_workers加速数据加载，减少GPU等待时间
        num_workers = 0
        if not self.is_win_platform:
            cpu_cnt = cpu_count(logical=False)
            gpu_cnt = torch.cuda.device_count()
            # 内存紧张时减少worker数量：每个worker会fork进程，占用额外内存
            # 3个DDP进程 × num_workers 个子进程 = 大量内存开销
            if unuse_mem < 8.0:
                num_workers = 0  # 内存极度紧张（<8GB），禁用多进程加载
            else:
                num_workers = min(2, gpu_cnt) if gpu_cnt > 0 else 1

        train_dataset = MyDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            keep_in_memory=keep_in_memory,
            max_seq_len=train_config.max_seq_len,
        )
        valid_dataset = MyDataset(
            parquet_file=train_config.validation_file,
            tokenizer_dir=train_config.tokenizer_dir,
            keep_in_memory=keep_in_memory,
            max_seq_len=train_config.max_seq_len,
        )

        batch_size = train_config.batch_size_per_gpu
        # NOTE: evaluate uses model.generate() (autoregressive decoding) which is
        # much slower than a forward pass. Keep eval_batch_size = batch_size to
        # avoid NCCL timeout during the evaluate phase.
        eval_batch_size = batch_size

        # persistent_workers: keep worker processes alive across epochs (avoid fork overhead)
        # prefetch_factor: prefetch more batches per worker to reduce GPU idle time
        use_persistent = num_workers > 0
        prefetch = 4 if num_workers > 0 else None

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,  
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
        )
        valid_dataloader = DataLoader(
            valid_dataset, 
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
        )

        device = accelerator.device
        log.info('using device: {} '.format(str(device)), save_to_file=True)
        

        # T5: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        tokenizer = train_dataset.tokenizer
        decoder_start_token_id = tokenizer.pad_token_id

        # for t5, set decoder_start_token_id = pad_token_id
        t5_config = get_T5_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=decoder_start_token_id, eos_token_id=tokenizer.eos_token_id)

        model = TextToTextModel(t5_config)

        # Use torch.compile to speed up training (requires PyTorch 2.0+)
        # First compilation takes a few minutes, but subsequent steps are 10-30% faster.
        # if hasattr(torch, 'compile'):
        #     if accelerator.is_main_process:
        #         log.info('Applying torch.compile to model for faster training...', save_to_file=True)
        #     model = torch.compile(model)

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
        total_batch_size = train_config.batch_size_per_gpu
        total_eval_batch_size = eval_batch_size  # 评估的总batch size
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * train_config.batch_size_per_gpu
            total_eval_batch_size = num_gpus_used * eval_batch_size

        steps_per_epoch = int(np.ceil(len(train_dataset) // total_batch_size))
        eval_steps = int(np.ceil(len(valid_dataset) // total_eval_batch_size))  # 使用评估batch size计算

        if accelerator.is_main_process:
            log.info('train dataset size: {}, steps per epoch:{}; validation dataset size: {}, steps per validation: {}; datalodater num_workers: {}.'\
                    .format(len(train_dataset), steps_per_epoch, len(valid_dataset), eval_steps, num_workers), save_to_file=True)

        
        # CosineAnnealingWarmRestarts: restart learning rate every epoch
        # so that the model can learn sufficiently in later epochs,
        # unlike OneCycleLR which decays to ~0 in the second half.
        steps_per_epoch_for_scheduler = int(np.ceil(len(train_dataset) / (batch_size * accumulation_steps)))
        warmup_steps = min(getattr(train_config, 'warmup_steps', 1024), steps_per_epoch_for_scheduler // 2)

        # Phase 1: Linear warmup from learn_rate / div_factor to learn_rate
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / train_config.div_factor,  # start at learn_rate / div_factor
            end_factor=1.0,                               # ramp up to learn_rate
            total_iters=warmup_steps,
        )

        # Phase 2: Cosine annealing with warm restarts (restart every epoch)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch_for_scheduler,  # restart period = 1 epoch
            T_mult=1,                           # keep the same period for each restart
            eta_min=train_config.learn_rate / train_config.div_factor,  # min lr = initial lr
        )

        # Combine: warmup first, then cosine with warm restarts
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        
        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
                model, 
                optimizer,
                lr_scheduler, 
                train_dataloader, 
                valid_dataloader,
            )
        
        if is_keep_training:
            import glob, torch
            state_dir = train_config.train_state_dir
            moved = []

            # Pre-check scheduler compatibility on main process before any
            # rank calls load_state, to avoid FileNotFoundError on other ranks.
            if accelerator.is_main_process:
                sched_files = glob.glob(os.path.join(state_dir, 'scheduler*.bin'))
                need_skip_scheduler = False
                for f in sched_files:
                    try:
                        sd = torch.load(f, map_location='cpu')
                        # SequentialLR expects '_schedulers' key; if missing
                        # the checkpoint was saved with a different scheduler.
                        if '_schedulers' not in sd:
                            need_skip_scheduler = True
                            break
                    except Exception:
                        need_skip_scheduler = True
                        break

                if need_skip_scheduler:
                    log.info(
                        '[WARN] Scheduler state incompatible with current SequentialLR, '
                        'temporarily removing scheduler files (scheduler will reset)...',
                        save_to_file=True,
                    )
                    for f in sched_files:
                        bak = f + '.bak'
                        try:
                            os.rename(f, bak)
                            moved.append((bak, f))
                        except Exception:
                            pass

            # Synchronise all ranks so that file renames are visible to everyone
            accelerator.wait_for_everyone()

            try:
                accelerator.load_state(input_dir=state_dir)
            finally:
                # Restore scheduler state files on main process
                if accelerator.is_main_process:
                    for bak, orig in moved:
                        try:
                            os.rename(bak, orig)
                        except Exception:
                            pass
                accelerator.wait_for_everyone()
        
        self.model = model
        self.accelerator = accelerator
        
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

            # torch.cuda.empty_cache()

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
                
                # if step >= 20:break
            
            #  end for batch setps

            # 等所有训练进程完成再开始评估
            accelerator.wait_for_everyone()

            # Skip evaluate on early epochs to save time:
            # model.generate() is very slow (autoregressive decoding).
            # Only evaluate on last epoch or every 2 epochs.
            skip_eval = (train_config.epochs > 2) and (epoch < train_config.epochs - 1) and (epoch % 2 != 0)
            if skip_eval:
                cur_bleu4_score = 0.0
                if accelerator.is_main_process:
                    log.info(f'Skipping evaluate at epoch {epoch} (will eval at even epochs and last epoch)', save_to_file=True)
            else:
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
                # 最多保存最近keep_latest_n_ckp个模型文件
                # self.delete_early_checkpoint(epoch=epoch, keep_latest_n=train_config.keep_latest_n_ckp)
                self.save_model('best')
                accelerator.save_state(output_dir=train_config.train_state_dir)

            # 每个epoch打印一下日志
            if accelerator.is_main_process:

                progress.advance(epoch_progress, advance=1)
                info_txt = 'epoch log: epoch:{}, avg_loss:{}, cur_bleu4:{}, best_bleu4:{}, best_epoch:{}'.\
                            format(epoch, my_average(epoch_loss_list), cur_bleu4_score, best_bleu4, best_epoch)
                # log.info(info_txt, std_out=True, save_to_file=True)
                self.print_and_log(info_txt, accelerator)


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
        # 重要：评估阶段不要在每一步做 all_gather（gather_for_metrics）。
        # 原实现会在每个 batch 同步一次 GPU -> 等待最慢的 rank，
        # 一旦某个 rank 因为 decode/bleu 计算更慢，就会逐步“落后”，最终触发 NCCL allgather timeout。
        # 改为“各 rank 本地计算分数 + 最后 reduce 汇总”，只在末尾同步一次，大幅降低超时概率。
        max_seq_len = self.train_config.max_seq_len
        batch_decode = tokenizer.batch_decode

        local_sum = 0.0
        local_cnt = 0

        if accelerator.is_main_process:
            self.progress.reset(self.eval_progress)
            self.progress.update(self.eval_progress, visible=True)

        # Cap max eval steps to avoid NCCL timeout during long generate() loops.
        # 30 steps is enough to get a reasonable BLEU estimate.
        max_eval_steps = min(eval_steps, 30)

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
                # 注意：原代码 `bleu4_scores = ...; bleu4_scores.extend(bleu4_scores)` 是明显 bug（自我扩展导致翻倍）
                for i in range(len(target_txt)):
                    local_sum += float(get_bleu4_score(reference=target_txt[i], outputs=outputs_txt[i]))
                    local_cnt += 1

                # if step >= 5: break

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
        num_workers = 0 if self.is_win_platform else 4

        test_dataset = MyDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            keep_in_memory=False if self.is_win_platform else True,
            max_seq_len=train_config.max_seq_len,
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=train_config.batch_size_per_gpu,
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
        total_batch_size = train_config.batch_size_per_gpu
        if num_gpus_used >= 1:
            total_batch_size = num_gpus_used * train_config.batch_size_per_gpu

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

                # s = time.time()
                outputs = accelerator.unwrap_model(model).my_generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                )
                # accelerator.print('generate used: {}'.format(time.time() - s))

                # 测试阶段同理：不做每步 gather，避免不同 rank 速度差导致 NCCL 超时
                outputs = outputs.detach().cpu().numpy()
                target_ids = target_ids.detach().cpu().numpy()
                
                outputs_txt = batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_txt = batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # print('outputs: {}'.format(outputs[0:5]))
                # print('target_ids: {}'.format(target_ids[0:5]))
                # print()


                for i in range(len(target_txt)):
                    local_sum += float(get_bleu4_score(reference=target_txt[i], outputs=outputs_txt[i]))
                    local_cnt += 1

                # if step >= 10: break
        
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
    
    # trainer = ChatTrainer()
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    chat_trainer.train()
    # chat_trainer.test(best_epoch=0)