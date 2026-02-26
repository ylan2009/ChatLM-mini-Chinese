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
    sft_model_file: str = PROJECT_ROOT + '/model_save/sft_ultra/'

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'   # tokenizer一般和model权重放在同一个文件夹

    dpo_train_file: str = PROJECT_ROOT + '/data/my_dpo_train.json'
    dpo_eval_file: str = PROJECT_ROOT + '/data/my_dpo_eval.json'

    adapter_file: str = PROJECT_ROOT + '/data/dpo/adapter_model.safetensors'
    log_dir: str = PROJECT_ROOT + '/logs/'

    # 8万样本DPO推荐默认值（兼顾稳定性与训练效率）
    per_device_train_batch_size: int = 4
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    logging_first_step: bool = True
    logging_steps: int = 20
    save_steps: int = 500
    output_dir: str = PROJECT_ROOT + '/model_save/dpo'
    warmup_steps: int = 200
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
    epochs: int = 6                                 # 增加到5个epoch，让模型充分学习
    batch_size_per_gpu: int = 32                    # 🚀 从24提升到32，充分利用GPU显存（GPU显存使用率57-67%，还有空间）
    
    learn_rate: float = 0.00015                     # 最大 div_factor * learn_rate
    div_factor: int = 50

    mixed_precision: str = "bf16"                   # 混合精度 ''no','fp16','bf16' or 'fp8'

    # 注意：计算梯度时相当于batch_size * gradient_accumulation_steps，说人话就是梯度累积步数>1时，等于增大n倍的batch_size
    gradient_accumulation_steps: int = 2            # 保持不变（实际有效batch=32*3*2=192）

    warmup_steps: int = 1024                        # 模型参数预热步数，预热样本数=warmup_steps * batch_size * gradient_accumulation_steps

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'  # tokenizer一般和model权重放在同一个文件夹
    model_file: str = PROJECT_ROOT + '/model_save/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset_6m.parquet'
    # train_file: str = PROJECT_ROOT + '/data/my_train_dataset_3m.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    # validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset_300k.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # 从哪个模型开始微调，仅当traing 函数 is_finetune = True时生效
    # 微调记得冻结某些层或者调低学习率
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存，中断后可以从此处继续训练
    train_state_dir: str = PROJECT_ROOT + '/model_save/train_latest_state'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain'

    logging_steps: int = 100                    # 🚀 从50提升到100，减少GPU→CPU同步频率
    save_steps: int = 500                       # every 500 steps save checkpoint
    
    # dataset_cache_dir: str = PROJECT_ROOT + '/data/.cache'
    # trainer_log_file: str = PROJECT_ROOT + '/logs/trainer.log'

    keep_latest_n_ckp: int = 8                  # 训练过程中，最多保留多少个分数最好的模型文件

    seed: int = 23333
    dataloader_buffer_size: int = 10000         # 🚀 从50000降到10000，减少内存占用，加速数据加载
    max_seq_len: int = 192                      # 🚀 保持192，预训练阶段足够（从256降低）



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
    - 训练集：50,000样本
    - 验证集：2,000样本
    
    优化策略：
    - 增大batch_size：从1提升到24（充分利用GPU显存）
    - 减少梯度累积：从8降到2（减少内存占用）
    - 实际有效batch_size = 24 * 2(GPU) * 2 = 96
    - 增加num_workers：加速数据加载
    
    预期内存占用：10-14GB（双GPU）
    预期GPU显存占用：12-16GB/GPU
    """
    epochs: int = 5                              # 50k数据集训练5个epoch
    batch_size_per_gpu: int = 24                # 充分利用GPU显存
    
    learn_rate: float = 5e-5                     # 学习率保持不变
    div_factor: int = 25                         # 保持不变

    mixed_precision: str = "bf16"                # 混合精度训练

    # 减少梯度累积，因为batch_size已经增大
    # 实际有效batch_size = 16 * 2(GPU) * 2 = 64
    gradient_accumulation_steps: int = 2         # 🚀 从8降到2，减少内存占用

    warmup_steps: int = 300                      # 50k数据集适当增加warmup步数
    
    max_grad_norm: float = 1.0                   # 梯度裁剪

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'
    model_file: str = PROJECT_ROOT + '/model_save/sft_fast/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft_fast/model_config.json'
    
    # 使用sample_data.py生成的数据集（50k训练 + 2k验证）
    train_file: str = PROJECT_ROOT + '/data/sft_train_small_train.parquet'      # 50k训练数据
    validation_file: str = PROJECT_ROOT + '/data/sft_train_small_valid.parquet'  # 2k验证数据
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'

    # 从预训练模型开始微调
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft_fast/train_latest_state_sft_fast'
    output_dir: str = PROJECT_ROOT + '/model_save/sft_fast'

    # 50000样本，batch_size=24*2*2=96，每个epoch约521步
    logging_steps: int = 100                     # 每个epoch约5次日志
    save_steps: int = 521                        # 每个epoch保存1次
    
    keep_latest_n_ckp: int = 3                   # 只保留3个最好的模型

    seed: int = 23333
    dataloader_buffer_size: int = 50000          # 扩大buffer匹配数据量
    max_seq_len: int = 512                       # 序列长度512


@dataclass
class TrainConfigSFTUltra:
    """
    超高性能SFT训练配置 - 针对56核CPU + 3×20GB GPU优化
    
    优化策略：
    - 大幅增加batch_size：充分利用3个GPU的60GB显存
    - 优化数据加载：56个CPU核心，num_workers=32
    - 减少梯度累积：batch_size足够大，不需要太多累积
    - 增大dataloader_buffer_size：充分利用内存
    - 启用torch.compile：JIT编译优化
    
    预期效果：
    - GPU利用率：90%+（当前只有24-33%）
    - 训练速度：提升3-5倍
    - 数据加载：零等待
    """
    epochs: int = 8                              # loss仍在下降，增加轮次让模型充分收敛
    batch_size_per_gpu: int = 24                # 
    
    learn_rate: float = 5e-5                     # 学习率保持不变
    div_factor: int = 25                         # 保持不变

    mixed_precision: str = "bf16"                # 混合精度训练

    # 实际有效batch_size = 24 * 3(GPU) * 2 = 144
    gradient_accumulation_steps: int = 2         # 🔧 恢复为2，保持训练稳定性（有效batch=144）

    warmup_steps: int = 100                      # 🔧 恢复为100，避免学习率上升过快导致训练不稳定
    
    max_grad_norm: float = 1.0                   # 梯度裁剪

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'
    model_file: str = PROJECT_ROOT + '/model_save/sft_ultra/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft_ultra/model_config.json'
    
    # 使用prepare_small_sft_data.py生成的小数据集
    train_file: str = PROJECT_ROOT + '/data/sft_train_small_train.parquet'      # 小数据集训练数据
    validation_file: str = PROJECT_ROOT + '/data/sft_train_small_valid.parquet'  # 小数据集验证数据
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'

    # 从预训练模型开始微调
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft_ultra/train_latest_state_sft_ultra'
    output_dir: str = PROJECT_ROOT + '/model_save/sft_ultra'

    # 5000样本，batch_size=32*3*1=96，每个epoch约52步
    logging_steps: int = 10                      # 每个epoch约5次日志
    save_steps: int = 52                         # 每个epoch保存1次
    
    keep_latest_n_ckp: int = 3                   # 小数据集只保留3个最好的模型

    seed: int = 23333
    dataloader_buffer_size: int = 50000          # 🚀 大幅增加buffer，充分利用56核CPU
    max_seq_len: int = 512                       # 序列长度512
    
    # 🚀 新增：数据加载优化
    dataloader_num_workers: int = 32             # 🚀 56核CPU，使用32个worker
    dataloader_pin_memory: bool = True           # 🚀 启用内存锁定，加速GPU传输
    
    # 🚀 新增：编译优化
    use_torch_compile: bool = True               # 🚀 启用torch.compile优化
    compile_mode: str = "default"                # 🚀 编译模式：default支持动态shape（my_generate自回归解码序列长度变化）
                                                 # 注意：reduce-overhead会启用CUDA Graphs，要求shape固定，与generate不兼容
    
    # 🔧 梯度优化（gradient_checkpointing在小数据集SFT中可能导致训练不稳定，已禁用）
    gradient_checkpointing: bool = False         # 🔧 禁用梯度检查点，避免影响训练稳定性
    gradient_clip_algo: str = "norm"              # 梯度裁剪算法


# ===================================================================================
# 以下为大数据集预训练配置 - 针对3×20G显存GPU + 12G内存优化
@dataclass
class TrainConfigPretrainLarge:
    """
    大数据集预训练配置 - 适用于3×20G显存GPU + 12G内存环境
    
    推荐数据量：
    - 训练集：1000万样本
    - 验证集：10万样本
    
    优化策略：
    1. 充分利用GPU显存：batch_size=32（每张卡）
    2. 减少梯度累积：gradient_accumulation_steps=2（降低内存占用）
    3. 实际有效batch_size = 32 * 3(GPU) * 2 = 192（大batch提升训练稳定性）
    4. 使用ultra_low_mem模式：避免PyArrow缓存累积
    5. 禁用num_workers：避免多进程内存开销
    6. 减小dataloader_buffer_size：降低内存占用
    7. 缩短序列长度：max_seq_len=192（预训练阶段足够）
    
    预期内存占用：8-10GB（3 GPU）
    预期GPU显存占用：16-18GB/GPU
    预期训练速度：约52k steps/epoch（1000万数据）
    """
    epochs: int = 6                              # 增加到6个epoch，让模型充分收敛
    batch_size_per_gpu: int = 32                 # 🚀 充分利用20G显存
    
    learn_rate: float = 0.00015                   # 标准学习率
    div_factor: int = 50                         # 标准div_factor

    mixed_precision: str = "bf16"                # 混合精度训练

    # 减少梯度累积，降低内存占用
    # 实际有效batch_size = 32 * 3(GPU) * 2 = 192
    gradient_accumulation_steps: int = 2         # 🚀 降低到2，减少内存占用

    warmup_steps: int = 1024                     # 标准warmup步数
    
    max_grad_norm: float = 1.0                   # 梯度裁剪

    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_sp/'
    model_file: str = PROJECT_ROOT + '/model_save/pretrain_large/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/pretrain_large/model_config.json'
    
    # 大数据集文件路径（需要自己准备）
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset_6m.parquet'      # 1000万训练数据
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'  # 10万验证数据
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'

    # 预训练不需要加载checkpoint
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'

    # 训练状态保存
    train_state_dir: str = PROJECT_ROOT + '/model_save/pretrain_large/train_latest_state'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain_large'

    # 1000万样本，batch_size=32*3*2=192，每个epoch约52k步
    logging_steps: int = 100                     # 每100步记录一次（每个epoch约520次日志）
    save_steps: int = 5000                       # 每5000步保存一次（每个epoch约10次）
    
    keep_latest_n_ckp: int = 5                   # 只保留5个最好的模型（节省磁盘空间）

    seed: int = 23333
    dataloader_buffer_size: int = 5000           # 🚀 减小buffer，降低内存占用
    max_seq_len: int = 192                       # 🚀 缩短序列长度，预训练阶段192足够


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