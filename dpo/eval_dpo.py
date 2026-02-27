# coding=utf-8
"""
DPO 效果验证脚本
对比 SFT 原始模型 vs DPO 训练后模型 在相同问题上的回答质量
用法：
    python eval_dpo.py
    python eval_dpo.py --sft_model ../model_save/sft_ultra/chat_small_t5.best.bin
    python eval_dpo.py --dpo_model ../model_save/dpo/checkpoint-3138
    python eval_dpo.py --interactive   # 交互模式，手动输入问题
"""
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from config import DpoConfig, T5ModelConfig, InferConfig
from model.chat_model import TextToTextModel
from utils.functions import get_T5_config

# ============================================================
# 默认测试问题集（覆盖常见场景）
# ============================================================
DEFAULT_QUESTIONS = [
    "如何保持健康的生活方式？",
    "人工智能会取代人类工作吗？",
    "请介绍一下中国的四大发明。",
    "如何提高英语口语水平？",
    "什么是量子计算？",
    "如何处理工作中的压力？",
    "推荐几本值得阅读的书籍。",
    "气候变化对地球有哪些影响？",
    "如何学好编程？",
    "请写一首关于春天的短诗。",
]


def load_model(model_path: str, tokenizer_dir: str, device: str = "cuda"):
    """加载模型（支持目录格式和 .bin 格式）"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    if os.path.isdir(model_path):
        print(f"  从目录加载模型: {model_path}")
        model = TextToTextModel.from_pretrained(model_path)
    else:
        print(f"  从文件加载模型: {model_path}")
        t5_config = get_T5_config(
            T5ModelConfig(),
            vocab_size=len(tokenizer),
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = TextToTextModel(t5_config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))

    model.eval()
    model.to(device)
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_seq_len: int = 256,
                    search_type: str = "beam", device: str = "cuda") -> str:
    """生成回答"""
    prompt = f"{question}[EOS]"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.my_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_seq_len=max_seq_len,
            search_type=search_type,
        )

    # 解码，跳过特殊 token
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()


def compare_models(sft_model, dpo_model, tokenizer_sft, tokenizer_dpo,
                   questions: list, device: str = "cuda"):
    """逐题对比两个模型的回答"""
    print("\n" + "=" * 70)
    print("  SFT 模型  vs  DPO 模型  对比评测")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n【问题 {i}/{len(questions)}】{question}")
        print("-" * 70)

        sft_ans = generate_answer(sft_model, tokenizer_sft, question, device=device)
        dpo_ans = generate_answer(dpo_model, tokenizer_dpo, question, device=device)

        print(f"[SFT]  {sft_ans}")
        print(f"[DPO]  {dpo_ans}")

    print("\n" + "=" * 70)
    print("对比完成！")


def interactive_mode(dpo_model, tokenizer, device: str = "cuda"):
    """交互模式：手动输入问题，实时查看 DPO 模型回答"""
    print("\n" + "=" * 70)
    print("  DPO 模型交互测试（输入 'quit' 或 'exit' 退出）")
    print("=" * 70)

    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("退出。")
            break
        if not question:
            continue

        answer = generate_answer(dpo_model, tokenizer, question, device=device)
        print(f"[DPO]  {answer}")


def main():
    parser = argparse.ArgumentParser(description="DPO 效果验证脚本")
    parser.add_argument("--sft_model", type=str, default=None,
                        help="SFT 模型路径（目录或 .bin 文件），默认使用 DpoConfig.sft_model_file")
    parser.add_argument("--dpo_model", type=str, default=None,
                        help="DPO 模型路径（目录），默认使用最新 checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="tokenizer 目录，默认使用 DpoConfig.tokenizer_dir")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备，默认 cuda")
    parser.add_argument("--search_type", type=str, default="beam",
                        choices=["beam", "greedy", "sampling"],
                        help="生成策略，默认 beam search")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式：只加载 DPO 模型，手动输入问题")
    parser.add_argument("--questions", type=str, nargs="+", default=None,
                        help="自定义测试问题列表，不指定则使用内置问题集")
    args = parser.parse_args()

    cfg = DpoConfig()

    # 路径解析
    sft_model_path = args.sft_model or cfg.sft_model_file
    tokenizer_dir  = args.tokenizer or cfg.tokenizer_dir

    # DPO 模型路径：优先用参数，其次找最新 checkpoint，最后用 output_dir
    if args.dpo_model:
        dpo_model_path = args.dpo_model
    else:
        ckpt_dir = cfg.output_dir
        checkpoints = sorted(
            [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        ) if os.path.isdir(ckpt_dir) else []
        dpo_model_path = os.path.join(ckpt_dir, checkpoints[-1]) if checkpoints else ckpt_dir

    questions = args.questions or DEFAULT_QUESTIONS

    print(f"\n设备      : {args.device}")
    print(f"SFT 模型  : {sft_model_path}")
    print(f"DPO 模型  : {dpo_model_path}")
    print(f"Tokenizer : {tokenizer_dir}")
    print(f"生成策略  : {args.search_type}")

    # 加载 DPO 模型
    print("\n[1/2] 加载 DPO 模型...")
    dpo_model, dpo_tokenizer = load_model(dpo_model_path, tokenizer_dir, device=args.device)

    if args.interactive:
        interactive_mode(dpo_model, dpo_tokenizer, device=args.device)
        return

    # 加载 SFT 模型
    print("[2/2] 加载 SFT 模型...")
    sft_model, sft_tokenizer = load_model(sft_model_path, tokenizer_dir, device=args.device)

    # 对比评测
    compare_models(sft_model, dpo_model, sft_tokenizer, dpo_tokenizer,
                   questions=questions, device=args.device)


if __name__ == "__main__":
    main()
