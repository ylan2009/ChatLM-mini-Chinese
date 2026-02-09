#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型评估脚本

功能：
    1. 评估预训练模型在验证集上的BLEU4分数
    2. 支持自定义模型路径、数据集路径
    3. 支持多GPU评估
    4. 生成详细的评估报告

使用方法：
    # 单GPU评估（使用默认配置）
    python evaluate_pretrain.py
    
    # 多GPU评估（推荐，速度更快）
    accelerate launch --multi_gpu --num_processes 2 ./evaluate_pretrain.py
    
    # 指定模型文件评估
    python evaluate_pretrain.py --model_file=/path/to/model.bin
    
    # 指定验证集评估
    python evaluate_pretrain.py --validation_file=/path/to/valid.parquet
    
    # 评估并保存详细结果
    python evaluate_pretrain.py --save_results=True --output_file=./eval_results.txt
    
    # 评估指定数量的样本（快速测试）
    python evaluate_pretrain.py --max_eval_samples=100

参数说明：
    --model_file: 模型文件路径，支持.bin或.safetensors格式（默认: 使用config中的best模型）
    --validation_file: 验证集文件路径（默认: 使用config中的验证集）
    --tokenizer_dir: tokenizer目录路径（默认: 使用config中的tokenizer）
    --batch_size: 评估batch size（默认: 32，评估时可以用更大的batch size）
    --max_eval_samples: 最大评估样本数，用于快速测试（默认: None，评估全部）
    --save_results: 是否保存详细结果（默认: False）
    --output_file: 结果保存路径（默认: ./evaluation_results.txt）
    --show_examples: 显示多少个预测示例（默认: 10）
"""

import os
import time
import fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

from config import TrainConfig, T5ModelConfig
from model.chat_model import TextToTextModel
from model.dataset import MyDataset
from utils.functions import get_bleu4_score, get_T5_config
from utils.logger import Logger


class PretrainEvaluator:
    """预训练模型评估器"""
    
    def __init__(self):
        self.logger = Logger('pretrain_evaluator', std_out=True, save2file=True, file_name=None)
        self.console = Console()
    
    def evaluate(
        self,
        model_file: str = None,
        validation_file: str = None,
        tokenizer_dir: str = None,
        batch_size: int = 32,
        max_eval_samples: int = None,
        save_results: bool = False,
        output_file: str = './evaluation_results.txt',
        show_examples: int = 10,
        seed: int = 23333,
    ):
        """
        评估预训练模型
        
        Args:
            model_file: 模型文件路径
            validation_file: 验证集文件路径
            tokenizer_dir: tokenizer目录路径
            batch_size: 评估batch size
            max_eval_samples: 最大评估样本数
            save_results: 是否保存详细结果
            output_file: 结果保存路径
            show_examples: 显示多少个预测示例
            seed: 随机种子
        """
        # 加载配置
        train_config = TrainConfig()
        model_config = T5ModelConfig()
        
        # 使用传入的参数或默认配置
        model_file = model_file or train_config.model_file.format('best')
        validation_file = validation_file or train_config.validation_file
        tokenizer_dir = tokenizer_dir or train_config.tokenizer_dir
        max_seq_len = train_config.max_seq_len
        
        # 验证文件是否存在
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f'模型文件不存在: {model_file}\n'
                f'请确认预训练已完成，或指定正确的模型路径'
            )
        if not os.path.exists(validation_file):
            raise FileNotFoundError(
                f'验证集文件不存在: {validation_file}\n'
                f'请先运行数据预处理脚本生成验证集'
            )
        
        # 初始化accelerator
        set_seed(seed)
        accelerator = Accelerator(mixed_precision=train_config.mixed_precision)
        device = accelerator.device
        
        if accelerator.is_main_process:
            self.logger.info(f'开始评估预训练模型', save_to_file=True)
            self.logger.info(f'模型文件: {model_file}', save_to_file=True)
            self.logger.info(f'验证集: {validation_file}', save_to_file=True)
            self.logger.info(f'使用设备: {device}', save_to_file=True)
            self.logger.info(f'Batch size: {batch_size}', save_to_file=True)
        
        # 加载数据集
        if accelerator.is_main_process:
            self.logger.info('加载验证集...', save_to_file=True)
        
        valid_dataset = MyDataset(
            parquet_file=validation_file,
            tokenizer_dir=tokenizer_dir,
            keep_in_memory=True,
            max_seq_len=max_seq_len,
        )
        
        # 如果指定了最大评估样本数，截取数据集
        if max_eval_samples and max_eval_samples < len(valid_dataset):
            # 正确截取数据集：同时更新data和length
            valid_dataset.data = valid_dataset.data.iloc[:max_eval_samples]
            valid_dataset.length = max_eval_samples
            if accelerator.is_main_process:
                self.logger.info(f'使用前 {max_eval_samples} 个样本进行快速评估', save_to_file=True)
        
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=True,
            num_workers=0,
        )
        
        if accelerator.is_main_process:
            self.logger.info(f'验证集大小: {len(valid_dataset)}', save_to_file=True)
        
        # 加载模型
        if accelerator.is_main_process:
            self.logger.info('加载模型...', save_to_file=True)
        
        tokenizer = valid_dataset.tokenizer
        t5_config = get_T5_config(
            model_config,
            vocab_size=len(tokenizer),
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        model = TextToTextModel(t5_config)
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        
        # Prepare model and dataloader
        model, valid_dataloader = accelerator.prepare(model, valid_dataloader)
        model.eval()
        
        # 计算评估步数
        num_gpus = accelerator.state.num_processes
        total_batch_size = batch_size * num_gpus if num_gpus >= 1 else batch_size
        eval_steps = int(np.ceil(len(valid_dataset) / total_batch_size))
        
        if accelerator.is_main_process:
            self.logger.info(f'评估步数: {eval_steps}', save_to_file=True)
            self.logger.info('开始评估...', save_to_file=True)
        
        # 评估
        start_time = time.time()
        bleu4_score, examples = self._evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataloader=valid_dataloader,
            accelerator=accelerator,
            eval_steps=eval_steps,
            max_seq_len=max_seq_len,
            show_examples=show_examples,
        )
        eval_time = time.time() - start_time
        
        # 输出结果
        if accelerator.is_main_process:
            self._print_results(
                bleu4_score=bleu4_score,
                eval_time=eval_time,
                dataset_size=len(valid_dataset),
                examples=examples,
                show_examples=show_examples,
            )
            
            # 保存结果
            if save_results:
                self._save_results(
                    output_file=output_file,
                    model_file=model_file,
                    validation_file=validation_file,
                    bleu4_score=bleu4_score,
                    eval_time=eval_time,
                    dataset_size=len(valid_dataset),
                    examples=examples,
                )
        
        return bleu4_score
    
    def _evaluate_model(
        self,
        model,
        tokenizer,
        dataloader,
        accelerator,
        eval_steps,
        max_seq_len,
        show_examples=10,
    ):
        """执行模型评估"""
        local_sum = 0.0
        local_cnt = 0
        examples = []
        batch_decode = tokenizer.batch_decode
        
        # 进度条
        if accelerator.is_main_process:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[bold blue]{task.fields[show_info]}"),
                refresh_per_second=1,
            )
            eval_progress = progress.add_task(
                description='评估进度: ',
                show_info='',
                total=eval_steps
            )
            progress.start()
        
        with torch.no_grad():
            for step, batch_data in enumerate(dataloader):
                if step >= eval_steps:
                    break
                
                if accelerator.is_main_process:
                    progress.advance(eval_progress, advance=1)
                    progress.update(
                        eval_progress,
                        show_info=f'step: {step}/{eval_steps}'
                    )
                
                input_ids = batch_data['input_ids']
                input_mask = batch_data['input_mask']
                target_ids = batch_data['target_ids']
                
                # 生成预测
                outputs = accelerator.unwrap_model(model).my_generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                    search_type='greedy',
                )
                
                # 转换为CPU numpy
                outputs = outputs.detach().cpu().numpy()
                target_ids = target_ids.detach().cpu().numpy()
                input_ids = input_ids.detach().cpu().numpy()
                
                # 解码
                outputs_txt = batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_txt = batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                input_txt = batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                # 计算BLEU分数
                for i in range(len(target_txt)):
                    score = float(get_bleu4_score(reference=target_txt[i], outputs=outputs_txt[i]))
                    local_sum += score
                    local_cnt += 1
                    
                    # 收集示例（只在主进程的前几个batch收集）
                    if accelerator.is_main_process and len(examples) < show_examples:
                        examples.append({
                            'input': input_txt[i],
                            'target': target_txt[i],
                            'prediction': outputs_txt[i],
                            'bleu4': score,
                        })
        
        if accelerator.is_main_process:
            progress.stop()
        
        # 汇总所有进程的结果
        device = accelerator.device
        local_sum_t = torch.tensor(local_sum, device=device, dtype=torch.float32)
        local_cnt_t = torch.tensor(local_cnt, device=device, dtype=torch.float32)
        global_sum_t = accelerator.reduce(local_sum_t, reduction="sum")
        global_cnt_t = accelerator.reduce(local_cnt_t, reduction="sum")
        avg_bleu4_score = (global_sum_t / torch.clamp(global_cnt_t, min=1.0)).item()
        
        return avg_bleu4_score, examples
    
    def _print_results(self, bleu4_score, eval_time, dataset_size, examples, show_examples):
        """打印评估结果"""
        self.console.print("\n" + "="*80, style="bold green")
        self.console.print("评估结果", style="bold green", justify="center")
        self.console.print("="*80 + "\n", style="bold green")
        
        # 创建结果表格
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan", width=30)
        table.add_column("值", style="yellow", width=40)
        
        table.add_row("验证集大小", f"{dataset_size:,} 样本")
        table.add_row("BLEU-4 分数", f"{bleu4_score:.4f}")
        table.add_row("评估耗时", f"{eval_time:.2f} 秒")
        table.add_row("平均速度", f"{dataset_size/eval_time:.2f} 样本/秒")
        
        self.console.print(table)
        
        # 显示预测示例
        if examples:
            self.console.print(f"\n预测示例（前{min(show_examples, len(examples))}个）:", style="bold cyan")
            for idx, example in enumerate(examples[:show_examples], 1):
                self.console.print(f"\n[bold]示例 {idx}:[/bold]")
                self.console.print(f"  [cyan]输入:[/cyan] {example['input']}")
                self.console.print(f"  [green]目标:[/green] {example['target']}")
                self.console.print(f"  [yellow]预测:[/yellow] {example['prediction']}")
                self.console.print(f"  [magenta]BLEU-4:[/magenta] {example['bleu4']:.4f}")
        
        self.console.print("\n" + "="*80 + "\n", style="bold green")
    
    def _save_results(self, output_file, model_file, validation_file, bleu4_score, eval_time, dataset_size, examples):
        """保存评估结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("预训练模型评估报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型文件: {model_file}\n")
            f.write(f"验证集: {validation_file}\n")
            f.write(f"验证集大小: {dataset_size:,} 样本\n")
            f.write(f"BLEU-4 分数: {bleu4_score:.4f}\n")
            f.write(f"评估耗时: {eval_time:.2f} 秒\n")
            f.write(f"平均速度: {dataset_size/eval_time:.2f} 样本/秒\n")
            
            if examples:
                f.write("\n" + "="*80 + "\n")
                f.write("预测示例\n")
                f.write("="*80 + "\n\n")
                
                for idx, example in enumerate(examples, 1):
                    f.write(f"示例 {idx}:\n")
                    f.write(f"  输入: {example['input']}\n")
                    f.write(f"  目标: {example['target']}\n")
                    f.write(f"  预测: {example['prediction']}\n")
                    f.write(f"  BLEU-4: {example['bleu4']:.4f}\n\n")
        
        self.logger.info(f'评估结果已保存到: {output_file}', save_to_file=True)


if __name__ == '__main__':
    evaluator = PretrainEvaluator()
    fire.Fire(evaluator.evaluate)
