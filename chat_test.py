#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话效果测试脚本
用于快速测试预训练模型或 SFT 微调模型的对话质量

用法:
    python chat_test.py                        # 默认加载 SFT best 模型，beam 解码
    python chat_test.py --model sft            # 加载 SFT best 模型
    python chat_test.py --model pretrain       # 加载预训练 best 模型
    python chat_test.py --model sft --search greedy    # greedy 解码
    python chat_test.py --model sft --search sampling  # sampling 解码
    python chat_test.py --batch                # 批量测试预设问题集
"""

import argparse
import time
import os
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.infer import ChatBot
from config import InferConfig, PROJECT_ROOT as CFG_ROOT

# ========================== 模型路径配置 ==========================
MODEL_PATHS = {
    # SFT 微调最优模型
    'sft': {
        'model_dir': CFG_ROOT + '/model_save/sft_ultra/chat_small_t5.best.bin',
        'tokenizer_dir': CFG_ROOT + '/model_save/my_tokenizer_sp',
        'desc': 'SFT 微调最优模型 (chat_small_t5.best.bin)',
    },
    # 预训练最优模型
    'pretrain': {
        'model_dir': CFG_ROOT + '/model_save/chat_small_t5.best.bin',
        'tokenizer_dir': CFG_ROOT + '/model_save/my_tokenizer_sp',
        'desc': '预训练最优模型 (chat_small_t5.best.bin)',
    },
    # HuggingFace 格式目录（如果存在）
    'hf': {
        'model_dir': CFG_ROOT + '/model_save/ChatLM-mini-Chinese/',
        'tokenizer_dir': None,  # 使用 model_dir 中的 tokenizer
        'desc': 'HuggingFace 格式模型目录',
    },
}

# ========================== 预设测试问题 ==========================
BATCH_QUESTIONS = [
    "你好，请介绍一下你自己",
    "中国的首都是哪里？",
    "1加1等于几？",
    "什么是人工智能？",
    "请写一首关于春天的诗",
    "苹果是什么颜色的？",
    "地球绕太阳转一圈需要多长时间？",
    "推荐几道中国传统美食",
    "如何保持身体健康？",
    "请用一句话解释什么是机器学习",
]


def load_model(model_key: str, search_type: str) -> tuple:
    """加载模型，返回 (ChatBot, search_type)"""
    if model_key not in MODEL_PATHS:
        print(f"[错误] 未知模型: {model_key}，可选: {list(MODEL_PATHS.keys())}")
        sys.exit(1)

    cfg = MODEL_PATHS[model_key]
    model_path = cfg['model_dir']
    tokenizer_path = cfg['tokenizer_dir']

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请检查路径是否正确，或先完成训练。")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  加载模型: {cfg['desc']}")
    print(f"  模型路径: {model_path}")
    print(f"  解码策略: {search_type}")
    print(f"{'='*60}\n")

    infer_config = InferConfig()
    infer_config.model_dir = model_path
    infer_config.tokenizer_dir = tokenizer_path
    infer_config.max_seq_len = 128  # 限制最大生成长度，避免回答过长

    t0 = time.time()
    bot = ChatBot(infer_config=infer_config)
    elapsed = time.time() - t0
    print(f"✅ 模型加载完成，耗时 {elapsed:.1f}s\n")

    return bot, search_type


def single_chat(bot: ChatBot, search_type: str) -> None:
    """交互式单轮对话"""
    print("💬 进入对话模式（输入 exit 退出，输入 cls 清屏，输入 switch 切换解码策略）")
    print(f"   当前解码策略: {search_type}")
    print("-" * 60)

    current_search = search_type
    history = []

    while True:
        try:
            user_input = input("\n\033[0;33m用户：\033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() == 'exit':
            print("再见！")
            break

        if user_input.lower() == 'cls':
            os.system('clear' if os.name != 'nt' else 'cls')
            history.clear()
            print("💬 对话已清空\n")
            continue

        if user_input.lower() == 'switch':
            options = ['greedy', 'beam', 'sampling', 'contrastive']
            idx = options.index(current_search) if current_search in options else 0
            current_search = options[(idx + 1) % len(options)]
            print(f"   ✅ 已切换解码策略为: \033[0;36m{current_search}\033[0m")
            continue

        # 生成回复
        t0 = time.time()
        try:
            # 使用 chat 方法（非流式，支持多种解码策略）
            response = bot.chat(user_input)
            # 如果 chat 方法不支持 search_type 参数，则直接调用底层
            if current_search != 'greedy':
                import torch
                # 使用 chat() 内部已拼接 [EOS]，这里直接用 batch_encode_plus
                encoded = bot.batch_encode_plus([user_input + '[EOS]'], padding=True, add_special_tokens=False)
                input_ids = torch.LongTensor(encoded.input_ids).to(bot.device)
                attention_mask = torch.LongTensor(encoded.attention_mask).to(bot.device)
                outputs = bot.model.my_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_seq_len=bot.infer_config.max_seq_len,
                    search_type=current_search,
                )
                response = bot.batch_decode(
                    outputs.cpu().numpy(),
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )[0]
                if not response:
                    response = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"
        except Exception as e:
            response = f"[生成出错] {e}"

        elapsed = time.time() - t0

        print(f"\033[0;32mChatBot：\033[0m{response}")
        print(f"\033[0;90m  ⏱ 耗时 {elapsed:.2f}s | 解码: {current_search}\033[0m")

        history.append((user_input, response))


def batch_test(bot: ChatBot, search_type: str, questions: list = None) -> None:
    """批量测试预设问题，输出对比结果"""
    if questions is None:
        questions = BATCH_QUESTIONS

    print(f"\n{'='*60}")
    print(f"  批量测试模式 | 解码策略: {search_type} | 共 {len(questions)} 个问题")
    print(f"{'='*60}\n")

    total_time = 0.0
    results = []

    for i, q in enumerate(questions, 1):
        t0 = time.time()
        try:
            if search_type == 'greedy':
                response = bot.chat(q)
            else:
                import torch
                encoded = bot.batch_encode_plus([q + '[EOS]'], padding=True, add_special_tokens=False)
                input_ids = torch.LongTensor(encoded.input_ids).to(bot.device)
                attention_mask = torch.LongTensor(encoded.attention_mask).to(bot.device)
                outputs = bot.model.my_generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_seq_len=bot.infer_config.max_seq_len,
                    search_type=search_type,
                )
                response = bot.batch_decode(
                    outputs.cpu().numpy(),
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )[0]
                if not response:
                    response = "（无输出）"
        except Exception as e:
            response = f"[出错] {e}"

        elapsed = time.time() - t0
        total_time += elapsed
        results.append((q, response, elapsed))

        print(f"[{i:02d}/{len(questions)}] \033[0;33mQ: {q}\033[0m")
        print(f"       \033[0;32mA: {response}\033[0m")
        print(f"       \033[0;90m⏱ {elapsed:.2f}s\033[0m\n")

    print(f"{'='*60}")
    print(f"  批量测试完成 | 总耗时: {total_time:.1f}s | 平均: {total_time/len(questions):.2f}s/条")
    print(f"{'='*60}\n")


def compare_models(search_type: str) -> None:
    """对比预训练模型和 SFT 模型的回答效果"""
    print(f"\n{'='*60}")
    print("  模型对比模式：预训练 vs SFT 微调")
    print(f"{'='*60}\n")

    # 加载两个模型
    print("正在加载预训练模型...")
    pretrain_bot, _ = load_model('pretrain', search_type)
    print("正在加载 SFT 模型...")
    sft_bot, _ = load_model('sft', search_type)

    print("\n输入问题进行对比（输入 exit 退出）：\n")

    while True:
        try:
            user_input = input("\033[0;33m用户：\033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == 'exit':
            break

        # 预训练模型回答
        t0 = time.time()
        pretrain_resp = pretrain_bot.chat(user_input)
        t1 = time.time()

        # SFT 模型回答
        sft_resp = sft_bot.chat(user_input)
        t2 = time.time()

        print(f"\n\033[0;34m[预训练模型]\033[0m {pretrain_resp}  \033[0;90m({t1-t0:.2f}s)\033[0m")
        print(f"\033[0;32m[SFT 模型]  \033[0m {sft_resp}  \033[0;90m({t2-t1:.2f}s)\033[0m\n")


def main():
    parser = argparse.ArgumentParser(
        description='ChatLM-mini-Chinese 对话效果测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python chat_test.py                          # SFT 模型交互对话（greedy）
  python chat_test.py --model pretrain         # 预训练模型交互对话
  python chat_test.py --model sft --search beam  # SFT 模型 beam search
  python chat_test.py --batch                  # SFT 模型批量测试
  python chat_test.py --compare                # 预训练 vs SFT 对比
        """
    )
    parser.add_argument(
        '--model', type=str, default='sft',
        choices=['sft', 'pretrain', 'hf'],
        help='选择模型: sft(SFT微调最优) / pretrain(预训练最优) / hf(HF格式目录)'
    )
    parser.add_argument(
        '--search', type=str, default='beam',
        choices=['greedy', 'beam', 'sampling', 'contrastive'],
        help='解码策略: greedy / beam / sampling / contrastive'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='批量测试模式：用预设问题集测试模型'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='对比模式：同时加载预训练和SFT模型进行对比'
    )

    args = parser.parse_args()

    if args.compare:
        compare_models(args.search)
    elif args.batch:
        bot, search_type = load_model(args.model, args.search)
        batch_test(bot, search_type)
    else:
        bot, search_type = load_model(args.model, args.search)
        single_chat(bot, search_type)


if __name__ == '__main__':
    main()
