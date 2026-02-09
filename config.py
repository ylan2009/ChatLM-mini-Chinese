from dataclasses import dataclass
from os.path import dirname, abspath

# replace '\' on windows to '/'
PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))

# ===================================================================================
# 以下为推断的配置
@dataclass
class InferConfig:
    max_seq_len: int = 320                          # 回答的最大长度
    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 模型文件路径，支持以下格式：
    # 1. 目录路径（包含huggingface格式的模型文件，如pytorch_model.bin或model.safetensors）
    # 2. .safetensors文件路径
    # 3. .bin文件路径（checkpoint格式）
    # 注意：生成reject response时应使用SFT后的模型，而不是原始预训练模型
    # 如果使用SFT的.bin格式checkpoint，需要指定具体文件路径，如：
    # model_dir = PROJECT_ROOT + '/model_save/sft/chat_small_t5.best.bin'
    model_dir: str = PROJECT_ROOT + '/model_save/ChatLM-mini-Chinese/'
    
    # tokenizer目录，如果为None则使用model_dir作为tokenizer路径（向后兼容）
    # 注意：如果model_dir指向.bin文件，必须单独指定tokenizer_dir
    # 例如：tokenizer_dir = PROJECT_ROOT + '/model_save/my_tokenizer_wiki/'
    tokenizer_dir: str = None

    # lora PDO 合并后的模型文件
    # model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.best.dpo.lora_merged.bin'
    
    # this confing for api demo:
    api_key: str = ""
    host: str = '127.0.0.1'
    port: int = 8812
    reload: bool = True
    workers: int = 1
    log_level: str = 'info'


#===================================================================================
# 以下为dpo训练配置
@dataclass
class DpoConfig:
    max_seq_len: int = 512 + 8                  # 8 for eos token 
    sft_model_file: str = PROJECT_ROOT + '/model_save/'

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/'   # tokenizer一般和model权重放在同一个文件夹

    dpo_train_file: str = PROJECT_ROOT + '/data/my_dpo_data.json'
    dpo_eval_file: str = PROJECT_ROOT + '/data/my_dpo_eval.json'

    adapter_file: str = PROJECT_ROOT + '/data/dpo/adapter_model.safetensors'
    log_dir: str = PROJECT_ROOT + '/logs/'

    per_device_train_batch_size: int = 4
    num_train_epochs: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    logging_first_step: bool = True
    logging_steps: int = 20                      
    save_steps: int = 2000
    output_dir: str = PROJECT_ROOT + '/model_save/dpo'
    warmup_steps: int = 1000
    fp16: bool = True
    seed: int = 23333
    beta: float = 0.1



# 以下为sft配置
@dataclass
class SFTconfig:
    max_seq_len: int = 384 + 8                # 8 for eos token 

    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/'

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/'  # tokenizer一般和model权重放在同一个文件夹
    sft_train_file: str = PROJECT_ROOT + '/data/sft_train.json'

    batch_size: int = 12
    num_train_epochs: int = 4
    save_steps: int = 5000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    logging_first_step: bool = True
    logging_steps: int = 100                      
    output_dir: str = PROJECT_ROOT + '/model_save/sft'
    warmup_steps: int = 100
    fp16: bool = True
    seed: int = 23333


# ===================================================================================
# 以下为训练的配置
@dataclass
class TrainConfig:
    epochs: int = 5                                 # 增加到5个epoch，让模型充分学习
    batch_size_per_gpu: int = 16
    
    learn_rate: float = 0.0001                      # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 8           # 累积梯度更新步数

    warmup_steps: int = 1024                        # 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = PROJECT_ROOT + '/model_save/train_latest_state'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain'

    logging_steps: int = 50
    save_steps: int = 5000
    
    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    keep_latest_n_ckp: int = 8                  # 训练过程中，最多保留多少个分数最好的模型文件

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256                      # 最大句子长度，默认：256



# ===================================================================================
# 以下为训练的配置
@dataclass
class TrainConfigSFT:
    epochs: int = 20                             # 增加到20个epoch，因为预训练不充分，需要更多训练
    batch_size_per_gpu: int = 20                 # 保持20，平衡显存和训练稳定性
    
    learn_rate: float = 5e-5                     # 提高到5e-5，因为预训练模型质量不好，需要更大的学习率
    div_factor: int = 25                         # 降低到25，让初始学习率更高

    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 6           # 增加到6，实际batch_size=20*2*6=240（更大的有效batch size）

    warmup_steps: int = 200                        # 降低到200，因为数据量小，快速进入正常学习率
                                                   # 预热样本数=warmup_steps * batch_size * gradient_accumulation_steps
    
    max_grad_norm: float = 1.0                     # 添加梯度裁剪，防止梯度爆炸导致loss波动

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/model_save/sft/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/sft_train_dataset_10k.parquet'      # 1万条训练数据
    validation_file: str = PROJECT_ROOT + '/data/sft_valid_dataset_1k.parquet'  # 1千条验证数据
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft/train_latest_state_sft'
    output_dir: str = PROJECT_ROOT + '/model_save/sft'

    # 根据训练集大小（8k条）调整：每个epoch约83步（2个GPU，batch_size=20，gradient_accumulation=6）
    # 每20步记录一次，每83步保存一次（即每个epoch保存一次）
    logging_steps: int = 20                        # 每个epoch约4次日志记录
    save_steps: int = 83                           # 每个epoch保存1次检查点
    
    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    keep_latest_n_ckp: int = 5                     # 训练过程中，最多保留多少个分数最好的模型文件（8万条数据总步数较少，降低保留数量）

    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256                      # 最大句子长度，默认：256


# ===================================================================================
# 以下为小数据集SFT训练配置 - 针对16G内存优化
@dataclass
class TrainConfigSFTSmall:
    """
    小数据集SFT训练配置 - 适用于16G内存环境
    
    推荐数据量：
    - 训练集：5,000样本
    - 验证集：500样本
    
    预期内存占用：8-10GB（双GPU）
    预期训练时长：2-4小时/epoch
    """
    epochs: int = 3                              # 小数据集训练3-5个epoch即可，避免过拟合
    batch_size_per_gpu: int = 1                  # 极致低内存：batch_size=1
    
    learn_rate: float = 5e-5                     # 学习率保持不变
    div_factor: int = 25                         # 保持不变

    mixed_precision: str = "bf16"                # 混合精度训练

    # 小batch_size通过梯度累积补偿
    # 实际有效batch_size = 1 * 2(GPU) * 8 = 16
    gradient_accumulation_steps: int = 8         # 梯度累积8步

    warmup_steps: int = 100                      # 小数据集减少warmup步数
    
    max_grad_norm: float = 1.0                   # 梯度裁剪

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'
    model_file: str = PROJECT_ROOT + '/model_save/sft_small/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft_small/model_config.json'
    
    # 使用prepare_small_sft_data.py生成的小数据集
    train_file: str = PROJECT_ROOT + '/data/sft_train_small_train.parquet'      # 小数据集训练数据
    validation_file: str = PROJECT_ROOT + '/data/sft_train_small_valid.parquet'  # 小数据集验证数据
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'

    # 从预训练模型开始微调
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft_small/train_latest_state_sft_small'
    output_dir: str = PROJECT_ROOT + '/model_save/sft_small'

    # 5000样本，batch_size=1*2*8=16，每个epoch约312步
    logging_steps: int = 50                      # 每个epoch约6次日志
    save_steps: int = 312                        # 每个epoch保存1次
    
    keep_latest_n_ckp: int = 3                   # 小数据集只保留3个最好的模型

    seed: int = 23333
    dataloader_buffer_size: int = 10000          # 减小buffer
    max_seq_len: int = 512                       # 序列长度512


# ===================================================================================
# 以下为高性能SFT训练配置 - 充分利用GPU显存
@dataclass
class TrainConfigSFTFast:
    """
    高性能SFT训练配置 - 充分利用GPU显存（20GB × 2）
    
    推荐数据量：
    - 训练集：5,000样本
    - 验证集：500样本
    
    优化策略：
    - 增大batch_size：从1提升到16（充分利用GPU显存）
    - 减少梯度累积：从8降到2（减少内存占用）
    - 实际有效batch_size = 16 * 2(GPU) * 2 = 64（比原来的16大4倍）
    - 增加num_workers：加速数据加载
    
    预期内存占用：10-14GB（双GPU）
    预期GPU显存占用：12-16GB/GPU（提升6-8倍）
    预期训练速度：提升5-6倍
    """
    epochs: int = 3                              # 小数据集训练3-5个epoch即可
    batch_size_per_gpu: int = 8                 # 🚀 从1提升到16，充分利用GPU显存
    
    learn_rate: float = 5e-5                     # 学习率保持不变
    div_factor: int = 25                         # 保持不变

    mixed_precision: str = "bf16"                # 混合精度训练

    # 减少梯度累积，因为batch_size已经增大
    # 实际有效batch_size = 16 * 2(GPU) * 2 = 64
    gradient_accumulation_steps: int = 2         # 🚀 从8降到2，减少内存占用

    warmup_steps: int = 100                      # 小数据集减少warmup步数
    
    max_grad_norm: float = 1.0                   # 梯度裁剪

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'
    model_file: str = PROJECT_ROOT + '/model_save/sft_fast/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft_fast/model_config.json'
    
    # 使用prepare_small_sft_data.py生成的小数据集
    train_file: str = PROJECT_ROOT + '/data/sft_train_small_train.parquet'      # 小数据集训练数据
    validation_file: str = PROJECT_ROOT + '/data/sft_train_small_valid.parquet'  # 小数据集验证数据
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'

    # 从预训练模型开始微调
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft_fast/train_latest_state_sft_fast'
    output_dir: str = PROJECT_ROOT + '/model_save/sft_fast'

    # 5000样本，batch_size=16*2*2=64，每个epoch约78步（比原来的312步快4倍）
    logging_steps: int = 15                      # 每个epoch约5次日志
    save_steps: int = 78                         # 每个epoch保存1次
    
    keep_latest_n_ckp: int = 3                   # 小数据集只保留3个最好的模型

    seed: int = 23333
    dataloader_buffer_size: int = 10000          # 减小buffer
    max_seq_len: int = 512                       # 序列长度512


#======================================================================================
# 以下为模型的配置
@dataclass
class T5ModelConfig:

    d_ff: int = 3072                        # 全连接层维度

    d_model: int = 768                      # 词向量维度
    num_heads: int = 12                     # 注意力头数 d_model // num_heads == d_kv
    d_kv: int = 64                          # d_model // num_heads

    num_decoder_layers: int = 10            # Transformer decoder 隐藏层层数
    num_layers: int = 10                    # Transformer encoder 隐藏层层数