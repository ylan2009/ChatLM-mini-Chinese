import sys
sys.path.extend(['.','..'])
import os
import re
import time
import ujson
from rich import progress
import pyarrow.parquet as pq
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from model.infer import ChatBot
from logger import Logger
from config import PROJECT_ROOT, InferConfig

from utils.raw_data_process import delete_file

log = Logger('data_process', save2file=True, file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/raw_data')


def _ensure_raw_data_dir() -> None:
    """确保原始数据目录存在。"""
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_alpaca_gpt4_raw() -> str:
    """
    从 HuggingFace 下载 alpaca-gpt4-data-zh 原始数据到本地 json 文件。

    数据集地址: https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
    返回值为本地 json 文件路径。
    """
    _ensure_raw_data_dir()
    save_path = os.path.join(RAW_DATA_DIR, 'alpaca_gpt4_data_zh.json')

    if os.path.exists(save_path):
        return save_path

    print('download alpaca-gpt4-data-zh from HuggingFace ...')
    ds = load_dataset('c-s-ale/alpaca-gpt4-data-zh', split='train')
    # 保存为 list[dict] 结构，方便后续 ujson.load 直接读取
    with open(save_path, 'w', encoding='utf-8') as f:
        ujson.dump(ds.to_list(), f, indent=4, ensure_ascii=False)

    print('save alpaca-gpt4-data-zh raw file to:', save_path)
    return save_path


def download_huozi_rlhf_raw() -> str:
    """
    从 HuggingFace 下载 huozi_rlhf_data_json 原始数据到本地 json 文件。

    数据集：https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json
    """
    _ensure_raw_data_dir()
    save_path = os.path.join(RAW_DATA_DIR, 'huozi_rlhf_data.json')

    if os.path.exists(save_path):
        return save_path

    print('download huozi_rlhf_data_json from HuggingFace ...')
    ds = load_dataset('Skepsun/huozi_rlhf_data_json', split='train')
    with open(save_path, 'w', encoding='utf-8') as f:
        ujson.dump(ds.to_list(), f, indent=4, ensure_ascii=False)

    print('save huozi_rlhf_data_json raw file to:', save_path)
    return save_path


def download_beyond_dpo_parquet() -> tuple[str, str]:
    """
    从 HuggingFace 下载 beyond/rlhf-reward-single-round-trans_chinese 数据集，
    并分别保存为 train/test parquet，路径与原代码保持一致。

    数据集：https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese
    """
    _ensure_raw_data_dir()
    train_path = os.path.join(
        RAW_DATA_DIR, 'train-00000-of-00001-789dc5dece0f1fc1.parquet'
    )
    test_path = os.path.join(
        RAW_DATA_DIR, 'test-00000-of-00001-8ecd46436fadcf7f.parquet'
    )

    need_download_train = not os.path.exists(train_path)
    need_download_test = not os.path.exists(test_path)

    if not (need_download_train or need_download_test):
        return train_path, test_path

    print('download beyond/rlhf-reward-single-round-trans_chinese from HuggingFace ...')
    if need_download_train:
        ds_train = load_dataset(
            'beyond/rlhf-reward-single-round-trans_chinese', split='train'
        )
        df_train = ds_train.to_pandas()
        df_train.to_parquet(train_path, index=False)
        print('save train parquet to:', train_path)

    if need_download_test:
        ds_test = load_dataset(
            'beyond/rlhf-reward-single-round-trans_chinese', split='test'
        )
        df_test = ds_test.to_pandas()
        df_test.to_parquet(test_path, index=False)
        print('save test parquet to:', test_path)

    return train_path, test_path


def process_alpaca_gpt4_data(max_len: int=512) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/c-s-ale/alpaca-gpt4-data-zh
    '''

    # 如果本地不存在原始文件，则先从 HuggingFace 下载
    read_file = PROJECT_ROOT + '/data/raw_data/alpaca_gpt4_data_zh.json'
    if not os.path.exists(read_file):
        read_file = download_alpaca_gpt4_raw()
    save_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    
    max_len += 8

    my_data = []

    with open(read_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        print('length of {} is {}'.format(read_file, len(data)))
        for item in progress.track(data):
            prompt = item['instruction']
            inputs = item['input']

            response = item['output']

            if len(response) > max_len: continue  # 超长的不要
            
            if len(inputs.strip()) > 0:
                prompt = f"{prompt}，{inputs}"
            
            if  len(prompt) > max_len: continue

            if len(prompt) == 0 or len(response) == 0: continue

            my_data.append(
                {
                    'prompt': prompt,
                    'chosen': response
                }
            )

    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)

def _cleanup_generated_text(txt: str, eos_token: str | None) -> str:
    """清理生成结果中的伪 EOS 文本，并在出现 eos_token 时截断。"""
    if txt is None:
        return ""
    # 常见残留
    txt = txt.replace("[EOS]", "").strip()
    if eos_token:
        pos = txt.find(eos_token)
        if pos != -1:
            txt = txt[:pos].strip()
    return txt


def generate_reject_by_strategy(data: list, strategy: str = 'mixed') -> list:
    """
    使用简单策略生成reject，不依赖模型。
    
    Args:
        data: 包含prompt和chosen的数据列表
        strategy: 生成策略
            - 'truncate': 截断chosen的后半部分
            - 'shuffle': 打乱chosen的句子顺序
            - 'random': 从其他样本随机选择chosen作为reject
            - 'mixed': 混合使用上述策略
    
    Returns:
        包含prompt、chosen和reject的数据列表
    """
    import random
    import re
    
    result = []
    
    for idx, item in enumerate(data):
        prompt = item['prompt']
        chosen = item['chosen']
        
        # 根据索引选择策略（如果是mixed模式）
        if strategy == 'mixed':
            current_strategy = ['truncate', 'shuffle', 'random'][idx % 3]
        else:
            current_strategy = strategy
        
        reject = ""
        
        if current_strategy == 'truncate':
            # 策略1：截断chosen的后半部分（保留60%-80%）
            truncate_ratio = random.uniform(0.6, 0.8)
            truncate_pos = int(len(chosen) * truncate_ratio)
            # 尝试在句子边界截断
            sentences = re.split(r'[。！？\n]', chosen)
            if len(sentences) > 1:
                keep_count = max(1, int(len(sentences) * truncate_ratio))
                reject = ''.join(sentences[:keep_count])
                # 如果原文有标点，加上标点
                if chosen[len(reject):len(reject)+1] in '。！？':
                    reject += chosen[len(reject)]
            else:
                reject = chosen[:truncate_pos]
        
        elif current_strategy == 'shuffle':
            # 策略2：打乱句子顺序
            sentences = re.split(r'([。！？\n])', chosen)
            # 将句子和标点配对
            paired = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    paired.append(sentences[i] + sentences[i+1])
                else:
                    paired.append(sentences[i])
            if len(paired) > 1:
                random.shuffle(paired)
                reject = ''.join(paired)
            else:
                # 如果只有一句话，使用截断策略
                reject = chosen[:int(len(chosen) * 0.7)]
        
        elif current_strategy == 'random':
            # 策略3：从其他样本随机选择chosen作为reject
            # 为了保证一定的相关性，选择长度相近的
            candidates = []
            target_len = len(chosen)
            for other_idx, other_item in enumerate(data):
                if other_idx != idx:
                    other_len = len(other_item['chosen'])
                    # 长度在50%-150%范围内
                    if 0.5 * target_len <= other_len <= 1.5 * target_len:
                        candidates.append(other_item['chosen'])
            
            if candidates:
                reject = random.choice(candidates)
            else:
                # 如果没有合适的候选，使用截断策略
                reject = chosen[:int(len(chosen) * 0.7)]
        
        # 确保reject不为空且与chosen不同
        if len(reject) == 0 or reject.strip() == chosen.strip():
            reject = chosen[:int(len(chosen) * 0.7)]
        
        result.append({
            'prompt': prompt,
            'chosen': chosen,
            'reject': reject,
        })
    
    return result


def _process_data_chunk(data_chunk, gpu_id, infer_config, max_len, batch_size, result_queue, progress_queue, completion_queue):
    """在指定GPU上处理数据块的工作函数（多进程版本）。"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"GPU {gpu_id}: 开始加载模型...")
    print(f"GPU {gpu_id}: 模型路径: {infer_config.model_dir}")
    print(f"GPU {gpu_id}: Tokenizer路径: {infer_config.tokenizer_dir}")

    chatbot = ChatBot(infer_config)
    model = chatbot.model
    tokenizer = chatbot.tokenizer

    model.to(device)
    model.eval()

    eos_token = getattr(tokenizer, "eos_token", None)
    print(f"GPU {gpu_id}: EOS token: {eos_token} (id={tokenizer.eos_token_id})")
    print(f"GPU {gpu_id}: PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"GPU {gpu_id}: Vocab size: {len(tokenizer)}")
    
    if eos_token is None:
        # 兜底：原代码用的是 [EOS]，但它可能不是 tokenizer 的真实 eos
        eos_token = None

    print(f"GPU {gpu_id}: 模型加载完成，开始处理 {len(data_chunk)} 条数据，per_gpu_batch_size={batch_size}")

    chunk_items = []
    batch_prompts = []
    batch_items = []
    processed_count = 0

    for idx, item in enumerate(data_chunk):
        prompt = item["prompt"]
        # 关键：必须添加EOS token，与SFT训练时的格式保持一致
        # SFT训练时input_ids = prompt + [eos_token_id]
        if eos_token:
            batch_prompts.append(f"{prompt}{eos_token}")
        else:
            batch_prompts.append(f"{prompt}[EOS]")
        batch_items.append(item)

        if len(batch_prompts) >= batch_size or idx == len(data_chunk) - 1:
            encoded = tokenizer.batch_encode_plus(
                batch_prompts,
                truncation=False,
                padding=True,
            )

            with torch.no_grad():
                input_ids = torch.LongTensor(encoded.input_ids).to(device)
                attention_mask = torch.LongTensor(encoded.attention_mask).to(device)

                # 计算合适的生成长度：prompt长度 + 期望的回答长度
                current_max_len = min(
                    input_ids.shape[1] + max_len,
                    infer_config.max_seq_len
                )

                # reject 不需要"最好"，但需要"相关且差一些"：用 sampling 比 greedy 更不容易坍缩到高频泛化回答
                # sampling模式内部已设置temperature=0.98, top_p=0.80
                outputs = model.my_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_seq_len=current_max_len,
                    search_type="sampling",
                )
            decoded = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(),
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )

            for src_item, reject, input_id_seq in zip(batch_items, decoded, input_ids):
                cur_prompt = src_item["prompt"]
                cur_chosen = src_item["chosen"]
                
                # 调试：打印前几个样本
                if processed_count < 3:
                    print(f"\n=== GPU {gpu_id} 样本 {processed_count} ===")
                    print(f"Prompt: {cur_prompt[:100]}...")
                    print(f"Raw output: {reject[:200]}...")
                
                # T5的generate返回的是decoder生成的内容，不包含输入
                # 但需要清理EOS token
                cur_reject = _cleanup_generated_text(reject, eos_token=eos_token)

                if len(cur_prompt) == 0 or len(cur_chosen) == 0 or len(cur_reject) == 0:
                    continue
                if len(cur_prompt) > max_len or len(cur_chosen) > max_len or len(cur_reject) > max_len:
                    continue

                chunk_items.append(
                    {
                        "prompt": cur_prompt,
                        "chosen": cur_chosen,
                        "reject": cur_reject,
                    }
                )

            processed_count += len(batch_items)
            batch_prompts = []
            batch_items = []

            progress_queue.put((gpu_id, processed_count, len(chunk_items)))

    # ====== 关键修改：发送完成信号 ======
    result_queue.put((gpu_id, chunk_items))
    print(f"GPU {gpu_id}: 处理完成，共处理 {processed_count} 条，生成 {len(chunk_items)} 条有效数据")
    # 主动发送完成信号
    completion_queue.put((gpu_id, "done"))
    print(f"GPU {gpu_id}: 已发送完成信号到主进程")
    # ====== 关键修改结束 ======

def generate_alpaca_gpt4_reject_by_strategy(groups_cnt: int=50000, max_len: int=320, strategy: str='mixed') -> None:
    """
    使用简单策略生成reject，不依赖模型。
    这种方法更快、更可靠，且生成的reject质量可控。
    
    Args:
        groups_cnt: 处理的样本数量
        max_len: 最大长度
        strategy: 生成策略 ('truncate', 'shuffle', 'random', 'mixed')
    """
    print('=' * 60)
    print(f'使用策略生成reject response (strategy={strategy})')
    print('=' * 60)
    
    finetune_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    save_rw_json_file = PROJECT_ROOT + '/data/my_dpo_alpaca_gpt4_data_zh.json'

    with open(finetune_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    # 只保留前 groups_cnt 条样本
    if groups_cnt is not None and groups_cnt > 0:
        data = data[:groups_cnt]
    
    print(f'总共需要处理 {len(data)} 条数据')
    
    # 使用策略生成reject
    all_items = generate_reject_by_strategy(data, strategy=strategy)
    
    # 过滤不符合要求的样本
    filtered_items = []
    for item in all_items:
        cur_prompt = item['prompt']
        cur_chosen = item['chosen']
        cur_reject = item['reject']
        
        if len(cur_prompt) == 0 or len(cur_chosen) == 0 or len(cur_reject) == 0:
            continue
        if len(cur_prompt) > max_len or len(cur_chosen) > max_len or len(cur_reject) > max_len:
            continue
        if cur_reject.strip() == cur_chosen.strip():
            continue
        
        filtered_items.append(item)
    
    print(f'处理完成，共生成 {len(filtered_items)} 条有效数据（过滤掉 {len(all_items) - len(filtered_items)} 条）')
    
    # 打印前3个样本
    print('\n前3个样本示例：')
    for i, item in enumerate(filtered_items[:3]):
        print(f'\n=== 样本 {i} ===')
        print(f'Prompt: {item["prompt"][:100]}...')
        print(f'Chosen: {item["chosen"][:100]}...')
        print(f'Reject: {item["reject"][:100]}...')
    
    # 保存结果
    with open(save_rw_json_file, 'w', encoding='utf-8') as f:
        ujson.dump(filtered_items, f, indent=4, ensure_ascii=False)
    print(f'\n结果已保存到: {save_rw_json_file}')
    print('=' * 60)


def generate_alpaca_gpt4_reject_response(groups_cnt: int=50000, max_len: int=320, batch_size: int=128) -> None:
    '''生成不是很满意的回答回答
    使用SFT后的模型生成reject response，而不是原始预训练模型
    支持多GPU并行处理，充分利用显存
    '''
    print('=' * 60)
    print('开始生成reject response')
    print('=' * 60)
    
    # 检查可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f'检测到 {num_gpus} 张GPU')
    
    if num_gpus == 0:
        print('警告: 未检测到GPU，将使用CPU（速度会很慢）')
        num_gpus = 1
        use_multi_gpu = False
    elif num_gpus == 1:
        print('警告: 只检测到1张GPU，将使用单GPU模式')
        use_multi_gpu = False
    else:
        print(f'将使用 {num_gpus} 张GPU进行并行处理')
        use_multi_gpu = True
        # 多GPU时，每张GPU处理更大的batch以充分利用显存
        # 如果用户没有显式指定batch_size，则根据GPU数量自动调整
        if batch_size == 128:  # 默认值
            batch_size = 128  # 每张GPU处理128个样本
            print(f'每张GPU的batch_size={batch_size}，总batch_size={batch_size * num_gpus}')

    # load config
    # 使用SFT后的模型生成reject response
    # SFT模型保存在 /model_save/sft/ 目录下，tokenizer在 /model_save/my_tokenizer_wiki/ 目录下
    infer_config = InferConfig()
    # 如果SFT目录下有.bin格式的checkpoint，使用最佳checkpoint
    sft_model_path = PROJECT_ROOT + '/model_save/sft/chat_small_t5.best.bin'
    if os.path.exists(sft_model_path):
        infer_config.model_dir = sft_model_path
        # tokenizer需要单独指定，因为.bin文件不包含tokenizer
        infer_config.tokenizer_dir = PROJECT_ROOT + '/model_save/my_tokenizer_wiki/'
        print(f'使用SFT模型: {sft_model_path}')
        print(f'使用tokenizer: {infer_config.tokenizer_dir}')
    else:
        # 如果没有找到SFT模型，使用默认配置（可能是DPO后的模型或其他）
        print(f'未找到SFT模型 {sft_model_path}，使用默认配置: {infer_config.model_dir}')
        if infer_config.tokenizer_dir is None:
            # 如果默认配置也没有指定tokenizer_dir，尝试使用my_tokenizer_wiki
            default_tokenizer = PROJECT_ROOT + '/model_save/my_tokenizer_wiki/'
            if os.path.exists(default_tokenizer):
                infer_config.tokenizer_dir = default_tokenizer
                print(f'使用默认tokenizer: {infer_config.tokenizer_dir}')

    finetune_file = PROJECT_ROOT + '/data/alpaca_gpt4_data_zh.json'
    save_rw_json_file = PROJECT_ROOT + '/data/my_dpo_alpaca_gpt4_data_zh.json'

    with open(finetune_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    # 只保留前 groups_cnt 条样本，避免一次生成过多导致显存溢出
    if groups_cnt is not None and groups_cnt > 0:
        data = data[: groups_cnt]
    
    log.info('length of {} is {}'.format(save_rw_json_file, len(data)), save_to_file=True)
    print(f'总共需要处理 {len(data)} 条数据')
    
    if use_multi_gpu:
        # 多GPU模式：将数据分割到不同GPU
        print(f'\n使用多GPU并行处理模式')
        print(f'将数据分割到 {num_gpus} 张GPU上，每张GPU处理约 {len(data) // num_gpus} 条数据')
        
        # 将数据分割成num_gpus份
        chunk_size = len(data) // num_gpus
        data_chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            if i == num_gpus - 1:
                # 最后一块包含所有剩余数据
                end_idx = len(data)
            else:
                end_idx = (i + 1) * chunk_size
            data_chunks.append(data[start_idx:end_idx])
            print(f'GPU {i}: 分配 {len(data_chunks[i])} 条数据 (索引 {start_idx} 到 {end_idx-1})')
        
        # 多GPU用多进程更稳（CUDA + threading 容易不真并行/偶发卡死）
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        progress_queue = ctx.Queue()
        completion_queue = ctx.Queue()  # 新增：完成信号队列

        procs = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=_process_data_chunk,
                args=(data_chunks[gpu_id], gpu_id, infer_config, max_len, batch_size, result_queue, progress_queue, completion_queue),  # 传递completion_queue
            )
            p.start()
            procs.append(p)
            print(f"已启动 GPU {gpu_id} 的进程 pid={p.pid}")
        
        # ======== 核心修复：使用完成队列监控所有进程完成 ========
        print('\n开始处理，实时进度:')
        total_processed = 0
        total_valid = 0
        gpu_processed = {i: 0 for i in range(num_gpus)}
        gpu_valid = {i: 0 for i in range(num_gpus)}
        completed_gpus = set()  # 跟踪已完成的GPU
        last_print_time = time.time()
        
        # 主循环：当所有GPU都发送完成信号时退出
        while len(completed_gpus) < num_gpus:
            # 检查进度更新
            while not progress_queue.empty():
                gpu_id, processed, valid = progress_queue.get()
                gpu_processed[gpu_id] = processed
                gpu_valid[gpu_id] = valid
                total_processed = sum(gpu_processed.values())
                total_valid = sum(gpu_valid.values())
            
            # 检查完成队列
            while not completion_queue.empty():
                gpu_id, status = completion_queue.get()
                completed_gpus.add(gpu_id)
                print(f"GPU {gpu_id}: 任务完成，已接收完成信号")
            
            # 打印进度（每秒刷新一次）
            if time.time() - last_print_time >= 5:
                progress_str = ' | '.join([f'GPU{i}: 处理{gpu_processed[i]}/有效{gpu_valid[i]}' for i in range(num_gpus)])
                print(f'总进度: 处理{total_processed} | 有效{total_valid} | {progress_str}')
                last_print_time = time.time()
            
            time.sleep(0.1)  # 降低CPU占用
        
        print("✅ 所有GPU进程已成功报告完成，进度监控完成")
        # ======== 修复结束 ========
        
        # 确保所有进程退出（虽然已完成，但安全起见）
        for p in procs:
            p.join(timeout=5)  # 最多等待5秒
        
        # 收集所有结果
        print('\n收集处理结果...')
        all_items = []
        gpu_results = {}
        for _ in range(num_gpus):
            gpu_id, chunk_items = result_queue.get()
            gpu_results[gpu_id] = chunk_items
            print(f'GPU {gpu_id}: 返回 {len(chunk_items)} 条有效数据')
        
        # 按GPU顺序合并结果，并在合并过程中保存检查点
        for gpu_id in range(num_gpus):
            all_items.extend(gpu_results[gpu_id])
            # 每收集一个GPU的结果就保存一次检查点（如果数量足够）
            if len(all_items) > 0 and len(all_items) % 2000 == 0:
                try:
                    with open(PROJECT_ROOT + '/data/outs.ckp.json', 'w', encoding='utf-8') as f_ckp:
                        ujson.dump(all_items, f_ckp, indent=4, ensure_ascii=False)
                    print(f'已保存检查点，当前合并了 {len(all_items)} 条有效数据')
                except Exception as e:
                    print(f'保存检查点错误: {e}')
        
        print(f'\n所有GPU处理完成，共生成 {len(all_items)} 条有效数据')
        
    else:
        # 单GPU模式：使用原来的逻辑
        print(f'\n使用单GPU模式')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        chatbot = ChatBot(infer_config)
        model = chatbot.model
        tokenizer = chatbot.tokenizer
        
        model.to(device)
        model.eval()
        
        print(f'总共需要处理 {len(data)} 条数据，batch_size={batch_size}')
        
        all_items = []
        batch_prompts = []
        batch_items = []
        
        for idx, item in progress.track(enumerate(data), total=len(data)):
            # 关键：必须添加EOS token，与SFT训练时的格式保持一致
            eos_token = getattr(tokenizer, "eos_token", None) or "[EOS]"
            batch_prompts.append(f"{item['prompt']}{eos_token}")
            batch_items.append(item)
            
            if idx % 500 == 0:
                print('process {} items.'.format(idx))
            
            if len(batch_prompts) >= batch_size or idx == len(data) - 1:
                encoded = tokenizer.batch_encode_plus(
                    batch_prompts,
                    truncation=False,
                    padding=True,
                )
                
                with torch.no_grad():
                    input_ids = torch.LongTensor(encoded.input_ids).to(device)
                    attention_mask = torch.LongTensor(encoded.attention_mask).to(device)
                    
                    # 计算合适的生成长度
                    current_max_len = min(
                        input_ids.shape[1] + max_len,
                        infer_config.max_seq_len
                    )
                    
                    # 使用sampling而不是greedy，增加多样性
                    # sampling模式内部已设置temperature=0.98, top_p=0.80
                    outputs = model.my_generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_seq_len=current_max_len,
                        search_type='sampling',
                    )
                
                decoded = tokenizer.batch_decode(
                    outputs.cpu().numpy(),
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )
                
                for src_item, reject in zip(batch_items, decoded):
                    cur_prompt = src_item['prompt']
                    cur_chosen = src_item['chosen']
                    
                    # 调试：打印前几个样本
                    if len(all_items) < 3:
                        print(f"\n=== 样本 {len(all_items)} ===")
                        print(f"Prompt: {cur_prompt[:100]}...")
                        print(f"Chosen: {cur_chosen[:100]}...")
                        print(f"Raw output: {reject[:200]}...")
                    
                    # T5的generate返回的是decoder生成的内容，不包含输入
                    cur_reject = _cleanup_generated_text(reject, eos_token=getattr(tokenizer, "eos_token", None))
                    
                    if len(cur_prompt) == 0 or len(cur_chosen) == 0 or len(cur_reject) == 0:
                        continue
                    if len(cur_prompt) > max_len or len(cur_chosen) > max_len or len(cur_reject) > max_len:
                        continue
                    
                    all_items.append({
                        'prompt': cur_prompt,
                        'chosen': cur_chosen,
                        'reject': cur_reject,
                    })
                
                batch_prompts = []
                batch_items = []
                
                if len(all_items) > 0 and len(all_items) % 2000 == 0:
                    try:
                        with open(PROJECT_ROOT + '/data/outs.ckp.json', 'w', encoding='utf-8') as f_ckp:
                            ujson.dump(all_items, f_ckp, indent=4, ensure_ascii=False)
                        print(f'已保存检查点，当前处理了 {len(all_items)} 条有效数据')
                    except Exception as e:
                        print('save checkpoint error: ', e)

    # 保存最终的带有 reject 的 DPO 数据
    print(f'\n处理完成，共生成 {len(all_items)} 条有效数据')
    with open(save_rw_json_file, 'w', encoding='utf-8') as f:
        ujson.dump(all_items, f, indent=4, ensure_ascii=False)
    print(f'结果已保存到: {save_rw_json_file}')
    print('=' * 60)
    
    # df = pd.DataFrame(data)
    # write_single_parquet_file(save_rw_parquet_file, df)

def replace_line(s: str) -> str:
    '''将双斜杠替换为单斜杠，既是 \\n 替换为 \n
    '''
    return re.sub('\\\\n', '\n', s)

def merge_rlhf_data(max_len: int=512) -> None:
    ''''
    处理RM高质量回答部分
    数据集：https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json
    https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese
    '''
    my_data = []
    # 确保外部两个 RLHF 源数据存在，不存在则先从 HuggingFace 下载
    huozi_file = PROJECT_ROOT + '/data/raw_data/huozi_rlhf_data.json'
    if not os.path.exists(huozi_file):
        huozi_file = download_huozi_rlhf_raw()

    alpaca_dpo_file = PROJECT_ROOT + '/data/my_dpo_alpaca_gpt4_data_zh.json'
    if not os.path.exists(alpaca_dpo_file):
        raise FileNotFoundError(
            f'找不到 {alpaca_dpo_file}，请先运行 process_alpaca_gpt4_data 和 generate_alpaca_gpt4_reject_response 生成该文件。'
        )

    read_files = [
        huozi_file,
        alpaca_dpo_file,
    ]
    save_file = PROJECT_ROOT + '/data/my_dpo_data.json'

    if os.path.exists(save_file): 
        assert delete_file(save_file)

    max_len += 8 # for eos token

    for read_file in read_files:
        items = []
        with open(read_file, 'r', encoding='utf-8') as f:
            items = ujson.load(f)

        for item in progress.track(items):
            prompt, chosen, reject = item['prompt'], item['chosen'], item['reject']

            if len(prompt) > max_len or len(chosen) > max_len or len(reject) > max_len:
                continue
            
            # reject.strip() == chosen.strip()，这两个相同的也不要
            if len(prompt) == 0 or len(chosen) == 0 or len(reject) == 0 or reject.strip() == chosen.strip(): 
                continue
            
            my_data.append({
                    'prompt': replace_line(prompt),
                    'chosen': replace_line(chosen),
                    'rejected': replace_line(reject),
            })
    
    # beyond DPO 数据，若本地不存在，则从 HuggingFace 下载后再读取
    train_parquet, test_parquet = download_beyond_dpo_parquet()
    read_files = [
        train_parquet,
        test_parquet,
    ]

    for read_file in read_files:
        pf = pq.read_table(read_file)
        for prompt, chosen, rejected  in progress.track(zip(pf['prompt'], pf['chosen'], pf['rejected']), total=pf.num_rows):
            
            prompt, chosen, rejected =  prompt.as_py(), chosen.as_py(), rejected.as_py()

            if len(prompt) > max_len or len(chosen) > max_len or len(rejected) > max_len:
                continue

            if len(prompt) == 0 or len(chosen) == 0 or len(rejected) == 0 or rejected.strip() == chosen.strip(): 
                continue
            
            my_data.append({
                    'prompt': replace_line(prompt),
                    'chosen': replace_line(chosen),
                    'rejected': replace_line(rejected),
            })
    print('length of {} is {}'.format(save_file, len(my_data)))

    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(my_data, f, indent=4, ensure_ascii=False)

def split_train_eval_dataset() -> None:
    '''划分数据集
    '''
    rw_json_file = PROJECT_ROOT + '/data/my_dpo_data.json'
    train_file = PROJECT_ROOT + '/data/my_dpo_train.json'
    eval_file = PROJECT_ROOT + '/data/my_dpo_eval.json'

    data = []

    with open(rw_json_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
    
    np.random.shuffle(data)
    split_idx = int(len(data) * 0.99)

    train_data = data[0: split_idx]
    eval_data = data[split_idx: ]

    log.info('train size: {}, eval size:{}'.format(len(train_data), len(eval_data)), save_to_file=True)

    with open(train_file, 'w', encoding='utf-8') as f:
        ujson.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(eval_file, 'w', encoding='utf-8') as f:
        ujson.dump(eval_data, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    # 1. 处理chosen文本
    # process_alpaca_gpt4_data()

    # 2. 生成rejected文本
    # 方法1：使用策略生成（推荐，快速且可靠）
    generate_alpaca_gpt4_reject_by_strategy(groups_cnt=500, strategy='mixed')
    
    # 方法2：使用模型生成（需要SFT模型训练良好）
    # generate_alpaca_gpt4_reject_response(groups_cnt=500, batch_size=256)

    # # 合并数据集
    # merge_rlhf_data()

    # # 3. split train and eval dataset
    # split_train_eval_dataset()
