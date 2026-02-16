from typing import Union

from torch.utils.data import Dataset
from torch import LongTensor, cuda
from transformers import PreTrainedTokenizerFast, T5Tokenizer, AutoTokenizer
from fastparquet import ParquetFile
from torch.utils.data import DataLoader
from datasets import load_dataset
import datasets
import pyarrow.parquet as pq
from numpy import array, int64
from numpy.random import shuffle

# import sys 
# sys.path.extend(['.', '..'])

from config import PROJECT_ROOT

class MyDataset(Dataset):

    def __init__(self, 
                parquet_file: str,
                tokenizer_dir: str,
                keep_in_memory: bool=False,
                max_seq_len: int=512,
                buffer_size: int=40960,
            ) -> None:
        '''
        keep_in_memory: whether to load the entire parquet into a pandas DataFrame in memory.
            True:  fastest random access via pandas.iloc, but high memory usage (N DDP procs × full dataset).
            False: uses PyArrow column-level indexing for random access with minimal memory footprint.
                   Fully supports multi-GPU / DistributedSampler (index-based access).
        '''
        super().__init__()

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        # Read the parquet file via PyArrow
        parquet_table = pq.read_table(parquet_file)

        # Dataset length
        self.length = parquet_table.num_rows

        # Buffer size for the legacy generator path (single-GPU fallback)
        self.buffer_size = self.length if buffer_size > self.length else buffer_size

        if keep_in_memory:
            # Load entire dataset into pandas for fastest iloc access
            self.data = parquet_table.to_pandas()
            self._prompt_col = None
            self._response_col = None
        else:
            # Keep only the two Arrow columns needed; release the full table reference.
            # Arrow ChunkedArray supports O(1) __getitem__ by index without slice()/take() overhead.
            self._prompt_col = parquet_table.column('prompt')
            self._response_col = parquet_table.column('response')
            self.data = None  # not used in this mode
            del parquet_table

        # Initialize tokenizer – try T5Tokenizer first (SentencePiece), fallback to others
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            except:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

        # Legacy generator (only used when keep_in_memory=False on single GPU)
        self.sample_generator = None
    
    def item_generator(self,) -> tuple:
        '''
        一条数据的生成器，防止大数据集OOM
        '''
                
        parquet_table = self.data

        # 生成器是死循环，不用退出，训练结束（epoch结束）会停止调用next()
        buffer_list = []
        while True:

            for prompt, response in zip(parquet_table['prompt'], parquet_table['response']):
                
                # 缓存数据不够，添加数据
                if len(buffer_list) < self.buffer_size:
                    buffer_list.append( (prompt.as_py(), response.as_py()) )
                    continue
                
                # 执行到这里，缓存区够了，打乱数据
                shuffle(buffer_list)
                for p, r in buffer_list:
                    # 在这里迭代
                    yield  p, r

                # 迭代完成，清空缓存区
                buffer_list = []
    
    def __getitem__(self, index):
        '''
        Return a single sample by index.
        Both keep_in_memory=True and False support random index access,
        so DistributedSampler / multi-GPU training works in either mode.
        '''
        if self.keep_in_memory:
            data = self.data
            prompt, response = data.iloc[index].prompt, data.iloc[index].response
        else:
            # Arrow ChunkedArray[index] returns a pyarrow Scalar; .as_py() converts to str.
            prompt = self._prompt_col[index].as_py()
            response = self._response_col[index].as_py()

        max_seq_len = self.max_seq_len - 5 # len('[EOS]') = 5
        # add an eos token note that end of resopnse, using in generate.
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"

    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        tokenizer = self.tokenizer

        prompt = tokenizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokenizer([item[1] for item in data], padding=True, return_token_type_ids=False)

        input_ids = array(prompt.input_ids, dtype=int64)
        input_mask = array(prompt.attention_mask, dtype=int64)
        target_ids = array(response.input_ids, dtype=int64)

        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
    
    def __len__(self) -> int:
        return self.length


class LowMemDataset(Dataset):
    """
    低内存版本的Dataset，支持多GPU分布式训练
    
    关键特性：
    1. 不将整个数据集加载到内存
    2. 使用pyarrow直接按索引读取，支持多GPU的数据分片
    3. 内存占用极小，适合16G内存环境
    4. 支持ultra_low_mem模式，每次读取时重新打开文件，避免缓存
    """
    
    def __init__(self, 
                parquet_file: str,
                tokenizer_dir: str,
                max_seq_len: int=512,
                ultra_low_mem: bool=False,
            ) -> None:
        '''
        低内存版本的Dataset，专为多GPU + 低内存环境设计
        
        parquet_file: parquet数据文件路径
        tokenizer_dir: tokenizer目录
        max_seq_len: 最大序列长度
        ultra_low_mem: 超低内存模式，每次读取时重新打开文件（更慢但内存占用更小）
        '''
        super().__init__()
        
        self.parquet_file = parquet_file
        self.max_seq_len = max_seq_len
        self.ultra_low_mem = ultra_low_mem
        
        if ultra_low_mem:
            # 超低内存模式：只读取元数据获取长度，不保留table引用
            parquet_table = pq.read_table(parquet_file)
            self.length = parquet_table.num_rows
            del parquet_table  # 立即释放
            self.parquet_table = None
        else:
            # 标准低内存模式：保留table引用用于快速读取
            self.parquet_table = pq.read_table(parquet_file)
            self.length = self.parquet_table.num_rows
        
        # 初始化tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            except:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    def __getitem__(self, index):
        '''
        按索引返回一条样本
        
        关键：使用pyarrow的slice功能，只读取需要的行，不加载整个表到内存
        '''
        if self.ultra_low_mem:
            # 超低内存模式：每次都重新打开文件读取
            # 这样避免了pyarrow的内部缓存累积，但速度会慢一些
            # 注意：不能使用 read_row_group，因为小数据集可能只有1个row_group
            parquet_table = pq.read_table(self.parquet_file)
            row = parquet_table.slice(index, 1)
            prompt = row['prompt'][0].as_py()
            response = row['response'][0].as_py()
            del parquet_table  # 立即释放内存
        else:
            # 标准模式：使用已加载的table
            row = self.parquet_table.slice(index, 1)
            prompt = row['prompt'][0].as_py()
            response = row['response'][0].as_py()
        
        max_seq_len = self.max_seq_len - 5  # len('[EOS]') = 5
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"
    
    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        tokenizer = self.tokenizer
        
        prompt = tokenizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokenizer([item[1] for item in data], padding=True, return_token_type_ids=False)
        
        input_ids = array(prompt.input_ids, dtype=int64)
        input_mask = array(prompt.attention_mask, dtype=int64)
        target_ids = array(response.input_ids, dtype=int64)
        
        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
    
    def __len__(self) -> int:
        return self.length


class ParquetDataset:
 
    def __init__(self,  
                parquet_file: Union[str, dict],
                tokenizer_dir: str, 
                keep_in_memory: bool=False,
                cache_dir: str='./.cache',
                buffer_size: int=10240, 
                max_len: int=512, 
                seed: int=23333
            ) -> None:
        '''
        使用huggingface的loaddataset方法加载,
        parquet_file: 单个文件，此时只能使用dataset['train']，
                多个文件请用:parquet_file={'train': 'train.parquet', 'test': 'test.parquet', 'validation': 'validation.parquet'})
                其他用法见：https://huggingface.co/docs/datasets/loading
        keep_in_memory: 是否将parquet文件转换为pandas.DataFrame格式存放到内存
        '''
        self.keep_in_memory = keep_in_memory
        self.len_dict = self.__get_all_parquet_file_size(parquet_file=parquet_file)

        self.max_len = max_len
        # 初始化tokenizer，自动选择合适的tokenizer类型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            # 如果AutoTokenizer失败，尝试使用T5Tokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
            except:
                # 最后尝试PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        
        streaming = False if keep_in_memory else True 
        # streaming=True,否则大数据集OOM
        dataset = load_dataset('parquet', data_files=parquet_file, cache_dir=cache_dir, streaming=streaming) 

        # 这里的batch_size不是训练的batch_size，是传递给precess_batch_func批处理的batch_size
        dataset = dataset.map(self.precess_batch_func, batched=True, batch_size=buffer_size, \
                            remove_columns=['prompt', 'response'], fn_kwargs={'max_len': max_len})

        dataset = dataset.with_format(type="torch")

        if keep_in_memory:
           dataset = dataset.shuffle(seed=seed, keep_in_memory=keep_in_memory)
        else:
            # 只能打乱缓冲区内的数据，不能打乱整个数据集，因此可以将缓存区设置稍微大一些
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

        self.dataset = dataset
    
    @staticmethod
    def precess_batch_func(item: dict, max_len: int=512) -> dict:
        '''
        添加EOS
        '''
        max_len -= 5 # len('[EOS]') = 5
        for i in range(len(item['prompt'])):
            item['prompt'][i] = f"{item['prompt'][i][0: max_len]}[EOS]"
        for i in range(len(item['response'])):
            item['response'][i] = f"{item['response'][i][0: max_len]}[EOS]"

        return {
            'prompt': item['prompt'],
            'response': item['response'],
        }
    
    def collate_fn(self, data: list[list]) -> dict:
        '''
        合并一个批次数据返回
        '''
        
        tokenizer = self.tokenizer
        prompt = [item['prompt'] for item in data ]
        response = [item['response'] for item in data ]

        # 按批次pad
        prompt_encoded = tokenizer(prompt, padding=True, return_token_type_ids=False)
        response_encoded = tokenizer(response, padding=True, return_token_type_ids=False)

        input_ids = array(prompt_encoded.input_ids, dtype=int64)
        input_mask = array(prompt_encoded.attention_mask, dtype=int64)
        target_ids = array(response_encoded.input_ids, dtype=int64)

        ret = {
            'input_ids': LongTensor(input_ids),
            'input_mask': LongTensor(input_mask),
            'target_ids': LongTensor(target_ids),
        }
        return ret
    def __getitem__(self, index: str) -> datasets.Dataset:
        '''
        魔术方法，实现下标访问，如：dataset['train']、dataset['validation']、dataset['test']
        '''
        return self.dataset[index]
    
    def __get_all_parquet_file_size(self, parquet_file: Union[str, dict]) -> dict:
        '''
        获取所有parquet file的长度
        '''
        len_dict = dict()
        if type(parquet_file) is str:
            train_len = self.__get_size_of_praquet(parquet_file)
            len_dict['train'] = train_len
        
        if type(parquet_file) is dict:
            for split_type, file in parquet_file.items():
                len_dict[split_type] = self.__get_size_of_praquet(file)
        
        return len_dict
    
    def __get_size_of_praquet(self, file_name: str) -> int:
        '''
        获取一个parquet文件的行数
        '''
        parquet_data = pq.read_table(file_name)

        return parquet_data.num_rows 
    
    def __len__(self) -> int:
        '''
        魔术方法，如果只有一个数据集，返回默认数据集大小
        '''
        if len(self.len_dict) == 1:
            return self.len_dict['train']
        else:
            raise Exception("this dataset contains many splited datasets, use `get_dataset_size(split_name)` function to get length, e.g: get_dataset_size('train')")
    
    def get_dataset_size(self, split_name: str) -> int:
        '''
        获取每个切分数据集的长度
        split_name可取：train、validation、test
        '''
        return self.len_dict[split_name]
    
    def get_tokenizer(self, ) -> PreTrainedTokenizerFast:
        return self.tokenizer



if __name__ == '__main__':
    parquet_file = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    tokenizer_dir = PROJECT_ROOT + '/model_save/tokenizer'

    # example 1：
    dataset = MyDataset(parquet_file, tokenizer_dir, keep_in_memory=False, max_seq_len=128)
    print('\nexample 1, dataset size: ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    for epoch in range(2):
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            print('step:{}'.format(step), x.shape, x_mask.shape, y.shape)
            if step == 5:
                break

    
    # exit(0)
    # example 2:
    dataset = ParquetDataset(parquet_file, tokenizer_dir, keep_in_memory=True, max_len=32)
    dataloader = DataLoader(dataset['train'], batch_size=32, collate_fn=dataset.collate_fn)
    print('\nexample 2, dataset size: ', dataset.get_dataset_size('train'))

    for epoch in range(2):
        print('epoch: {}'.format(epoch))
        for step, batch in enumerate(dataloader):
            x, x_mask, y = batch['input_ids'], batch['input_mask'], batch['target_ids']
            print('step:{}'.format(step), x.shape, x_mask.shape, y.shape)
            if step == 5:
                break
        
    