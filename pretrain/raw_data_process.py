import ujson
import re
from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import time
from collections import defaultdict

from matplotlib import pyplot as plt
import codecs, csv
import pandas as pd 
import numpy as np
from rich import progress
from rich.table import Table
from rich.console import Console
from fastparquet import ParquetFile, write
import pyarrow.parquet as pq
from opencc import OpenCC

import sys
sys.path.extend(['.','..'])

from utils.logger import Logger
from config import PROJECT_ROOT
from utils.functions import get_path_of_suffix_files, DropDatasetDuplicate

log = Logger('data_process', save2file=True, file_name=PROJECT_ROOT + '/logs/raw_data_process.log')

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len  = len(set_a) + len(set_b)
    
    if total_len == 0: return 0.0

    inter_set =  set_a & set_b
    
    return ( 2 * len(inter_set)) / total_len

def write_single_parquet_file(file_name: str, data_frame: pd.DataFrame) -> None:
    '''
    将dataframe写到单独的parquet file中
    '''
    append = False
    if exists(file_name):
        append = True 

    write(file_name, data_frame, compression='GZIP',append=append)


def read_and_write_template(read_file: str, write_to_file: str, call_back: object, group_cnt: int=10000) -> None:
    '''
    处理数据读写模板，需要提供一个回调函数call_back，
    read_file: 原始数据文件
    write_to_file：处理后的要保存数据文件
    call_back：函数输入一个字符串，输出一个处理后的字典dict，如果输入的字符串为无效数据，请返回None
    group_cnt: parquet file分割行数
    如：
    >>> def call_back(inputs: str) -> dict:
    >>>     if check(inputs) not valid:
    >>>         return None
    ...    
    ...    do something for inputs
    ...
    >>>     my_dict = {
    >>>             'prompt': inputs['p'],
    >>>             'response': inputs['a1'] + inputs['a2'],
    >>>             ...
    >>>         }
    >>>     return my_dict
    '''

    log.info('process file:{}'.format(read_file), save_to_file=True)
    start = time.time()
    
    raw_line_cnt = 0
    keep_line_cnt = 0
    
    with progress.open(read_file, 'r', encoding='utf-8') as f_read:
        cur_rows = []
        append = cur_rows.append
        for line in f_read:
            try:
                raw_line_cnt += 1

                write_dict = call_back(line)

                if write_dict is None: continue

                keep_line_cnt += 1
                append(write_dict)
                # ujson.dump(write_obj, f_write, indent=4, ensure_ascii=False)
                # ujson.dump(write_obj, f_write,  ensure_ascii=False,)
                # f_write.write('\n')

                if len(cur_rows) >= group_cnt:
                    df = pd.DataFrame(cur_rows)
                    write_single_parquet_file(write_to_file, df)
                    cur_rows = []
                    append = cur_rows.append

            except Exception as e:
                # log.error('处理文件异常：{}, content:{}'.format(str(e), line))
                print(line)
                raise e
        
        # end for
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(write_to_file, df)
            cur_rows = []
    
    end = time.time()

    log.info('原始文件:{}，共{}行，处理后剩余{}行，保存到文件：{}。耗时：{:.6}s'\
                .format(read_file, raw_line_cnt, keep_line_cnt, write_to_file, end - start), save_to_file=True)



#=====================================数据集处理=================================

def process_web_text(keep_start: int=5, response_less_word: int=10) -> None:
    '''
    处理425万社区问答webtext2019zh知识类数据集
    keep_start: 只保留点赞数大于keep_start的问答
    response_less_word: 答案至少要有response_less_word个字
    '''
    file_names = [
        '/data/raw_data/web_text_zh_test.json',
        '/data/raw_data/web_text_zh_train.json',
        '/data/raw_data/web_text_zh_valid.json',
    ]

    save_file_name = PROJECT_ROOT + '/data/my_data/my_web_text_zh.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(line: str) -> dict:
        item = ujson.loads(line)

        if item['star'] < keep_start or len(item['content']) < response_less_word: 
            return None

        # 数据清洗
        # 去除重复的标点符号
        prompt = remove_duplicate_punctuation(item['title'])
        response = remove_duplicate_punctuation(item['content'])
        write_dict = {
            "prompt": prompt,
            "response": response,
        }
        return write_dict

    for file_name in file_names:
        read_file = PROJECT_ROOT + file_name

        read_and_write_template(read_file, save_file_name, process_function)
        
        # 输出当前文件处理完成后的前10行数据
        log.info('=' * 80, save_to_file=True)
        log.info('文件 {} 处理完成，查看保存文件的前10行数据'.format(file_name), save_to_file=True)
        try:
            pf = pq.read_table(save_file_name)
            sample_size = min(10, pf.num_rows)
            for idx in range(sample_size):
                prompt = pf['prompt'][idx].as_py()
                response = pf['response'][idx].as_py()
                log.info('第{}行 - prompt: {}'.format(idx + 1, prompt), save_to_file=True)
                log.info('第{}行 - response: {}'.format(idx + 1, response), save_to_file=True)
                log.info('-' * 80, save_to_file=True)
        except Exception as e:
            log.error('读取样例数据失败：{}'.format(str(e)), save_to_file=True)
        log.info('=' * 80, save_to_file=True)


def process_belle(response_less_word: int=15, prompt_less_word: int=3, group_cnt: int=10000) -> None:
    '''
    处理belle数据集（parquet格式）

    '''
    # 指定要处理的三个parquet文件
    belle_data_path = PROJECT_ROOT + '/data/raw_data/belle'
    file_names = [
        f'{belle_data_path}/train_1M_CN.parquet',
        f'{belle_data_path}/train_2M_CN.parquet',
        f'{belle_data_path}/train_3.5M_CN.parquet'
    ]
    
    log.info(f'将处理 {len(file_names)} 个parquet文件', save_to_file=True)

    save_file_name = PROJECT_ROOT + '/data/my_data/my_baike_qa.parquet'
    # 后续append写入，存在文件先删除
    if exists(save_file_name): 
        assert delete_file(save_file_name)

    def process_function(sentence: str) -> str:
        '''
        针对一个句子的数据清洗
        '''
        # 删除\\r
        sentence = sentence.replace('\\r','') 

        # 删除重复的标点符号
        sentence = remove_duplicate_punctuation(sentence)

        return sentence

    all_cnt, keep_cnt = 0, 0
    cur_rows = []
    append = cur_rows.append
    
    for file in file_names:
        log.info('process file: {}'.format(file), save_to_file=True)
        
        try:
            pf = pq.read_table(file)

            print(pf.column_names)
            
            # 尝试不同的列名组合
            prompt_col = None
            response_col = None
            is_conversations_format = False
            
            # 检查可能的列名
            if 'conversations' in pf.column_names:
                # conversations格式：包含多轮对话的列表
                is_conversations_format = True
            elif 'instruction' in pf.column_names and 'output' in pf.column_names:
                prompt_col = 'instruction'
                response_col = 'output'
            elif 'INSTRUCTION' in pf.column_names and 'RESPONSE' in pf.column_names:
                prompt_col = 'INSTRUCTION'
                response_col = 'RESPONSE'
            elif 'prompt' in pf.column_names and 'response' in pf.column_names:
                prompt_col = 'prompt'
                response_col = 'response'
            else:
                log.error('无法识别文件列名: {}, 列名为: {}'.format(file, pf.column_names), save_to_file=True)
                continue
            
            # 用于存储当前文件的前10行数据
            file_sample_rows = []
            file_row_cnt = 0
            
            if is_conversations_format:
                # 处理conversations格式
                for conversations in progress.track(pf['conversations'], total=pf.num_rows):
                    conversations = conversations.as_py()
                    
                    # conversations是一个列表，包含多轮对话
                    # 每个元素是 {'from': 'human'/'assistant', 'value': '内容'}
                    if not isinstance(conversations, list) or len(conversations) < 2:
                        continue
                    
                    # 提取所有human和assistant的对话
                    for i in range(len(conversations) - 1):
                        if conversations[i].get('from') == 'human' and conversations[i+1].get('from') == 'assistant':
                            all_cnt += 1
                            prompt = conversations[i].get('value', '')
                            response = conversations[i+1].get('value', '')
                            
                            # 剔除翻译任务
                            if '翻译' in prompt or 'translate' in prompt.lower():
                                continue
                            
                            # 删除表格类任务
                            if '表格' in prompt or '-----' in prompt or '-----' in response:
                                continue
                            
                            # 数据清洗
                            prompt = process_function(prompt)
                            response = process_function(response)

                            # 剔除问题和答案过短的数据
                            if len(prompt) < prompt_less_word or len(response) < response_less_word:
                                continue
                            
                            keep_cnt += 1
                            write_dict = {
                                "prompt": prompt,
                                "response": response,
                            }
                            append(write_dict)
                            
                            # 保存当前文件的前10行样例
                            if file_row_cnt < 10:
                                file_sample_rows.append(write_dict)
                                file_row_cnt += 1

                            if len(cur_rows) >= group_cnt:
                                df = pd.DataFrame(cur_rows)
                                write_single_parquet_file(save_file_name, df)
                                cur_rows = []
                                append = cur_rows.append
            else:
                # 处理普通格式
                for prompt, response in progress.track(zip(pf[prompt_col], pf[response_col]), total=pf.num_rows):
                    all_cnt += 1
                    prompt, response = prompt.as_py(), response.as_py()
                    
                    # 剔除翻译任务
                    if '翻译' in prompt or 'translate' in prompt.lower():
                        continue
                    
                    # 删除表格类任务
                    if '表格' in prompt or '-----' in prompt or '-----' in response:
                        continue
                    
                    # 数据清洗
                    prompt = process_function(prompt)
                    response = process_function(response)

                    # 剔除问题和答案过短的数据
                    if len(prompt) < prompt_less_word or len(response) < response_less_word:
                        continue
                    
                    keep_cnt += 1
                    write_dict = {
                        "prompt": prompt,
                        "response": response,
                    }
                    append(write_dict)
                    
                    # 保存当前文件的前10行样例
                    if file_row_cnt < 10:
                        file_sample_rows.append(write_dict)
                        file_row_cnt += 1

                    if len(cur_rows) >= group_cnt:
                        df = pd.DataFrame(cur_rows)
                        write_single_parquet_file(save_file_name, df)
                        cur_rows = []
                        append = cur_rows.append
            
            # 输出当前文件处理完成后的前10行数据
            log.info('=' * 80, save_to_file=True)
            log.info('文件 {} 处理完成，前{}行数据如下：'.format(file, len(file_sample_rows)), save_to_file=True)
            for idx, row in enumerate(file_sample_rows, 1):
                log.info('第{}行 - prompt: {}'.format(idx, row['prompt']), save_to_file=True)
                log.info('第{}行 - response: {}'.format(idx, row['response']), save_to_file=True)
                log.info('-' * 80, save_to_file=True)
            log.info('=' * 80, save_to_file=True)
                    
        except Exception as e:
            log.error('处理文件异常：{}, file:{}'.format(str(e), file), save_to_file=True)
            continue
    
    # end for 
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file_name, df)
        cur_rows = []

    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(save_file_name, all_cnt, keep_cnt), save_to_file=True)
  
def repair_line_error_csv_file(raw_csv_file: str, save_suffix: str, read_encoding: str='utf-8', ) -> None:
    '''
        修复csv文件，将文件中换行符替换为\n，字段中的英文字符替换为中文字符
    '''
    
    with codecs.open(raw_csv_file, 'r', encoding=read_encoding, errors='ignore') as f:
        reader = csv.reader(f)
        new_lines = []

        for line in reader:
            for i in range(len(line)):
                line[i] = line[i].replace('\n', '\\n') # 处理异常的换行符
                line[i] = line[i].replace(',', '，') # 英文逗号换为中文逗号
            new_lines.append(line)

        with open(raw_csv_file[: -4] + save_suffix, 'w', encoding='utf-8', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_lines)

def process_chinese_medical_datasets(response_less_word: int=15) -> None:
    '''
    处理中国医药领域问答数据集
    '''
    raw_dataset_dir = PROJECT_ROOT + '/data/raw_data/chinese_medical_dialogue_datasets'
    
    raw_data_files = get_path_of_suffix_files(raw_dataset_dir, '.csv')

    # 如果没有修复的文件，则修复csv文件换行异常
    suffix = '.repaired.csv'
    need_to_repair_files = [
        file_name for file_name in raw_data_files \
            if not file_name.endswith(suffix) and file_name[0: -4] + suffix not in raw_data_files
    ]
 
    # 修复异常换行的文件
    for file_name in need_to_repair_files:
        repair_line_error_csv_file(file_name, suffix, read_encoding='gb2312')

    # 重新获取原始文件（即修复后的文件）
    raw_data_files = get_path_of_suffix_files(raw_dataset_dir, suffix)

    # 获取要保存的文件名
    save_file = PROJECT_ROOT + '/data/my_data/my_chinese_medical_dialogue.parquet'
    # for file_name in raw_data_files:
    #     file_name = file_name.split('/')[-1][0: -(len(suffix))] + '.parquet'
    #     file_name = PROJECT_ROOT  + '/data/my_data/' + file_name
    #     save_files.append(file_name)

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)
    
    def process_function(line: str) -> dict:
        # department,title,ask,answer
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[3]) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item[1], item[2]) >= 0.90:
            # title 和ask 相似度过高，只用ask作为问题
            prompt = item[2]
        else:
            # title 和 ask 拼接形成问题
            prompt = "{}{}".format(item[1], item[2])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = ''.join(item[3: ]).replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < 3 or len(response) < response_less_word:
            return None
        
        write_dict = {
                "prompt": prompt,
                "response": response,
            }

        return write_dict

    for i, file_name in enumerate(raw_data_files):
        read_file = file_name        

        read_and_write_template(read_file, save_file, process_function)
        
        # 输出当前文件处理完成后的前10行数据
        log.info('=' * 80, save_to_file=True)
        log.info('文件 {} 处理完成，查看保存文件的前10行数据'.format(file_name), save_to_file=True)
        try:
            pf = pq.read_table(save_file)
            sample_size = min(10, pf.num_rows)
            for idx in range(sample_size):
                prompt = pf['prompt'][idx].as_py()
                response = pf['response'][idx].as_py()
                log.info('第{}行 - prompt: {}'.format(idx + 1, prompt), save_to_file=True)
                log.info('第{}行 - response: {}'.format(idx + 1, response), save_to_file=True)
                log.info('-' * 80, save_to_file=True)
        except Exception as e:
            log.error('读取样例数据失败：{}'.format(str(e)), save_to_file=True)
        log.info('=' * 80, save_to_file=True)


def process_finace_dataset(prompt_less_word: int=10, response_less_word: int=15) -> None:
    '''
    处理金融问答数据集
    '''
    finace_data_file = PROJECT_ROOT + '/data/raw_data/financezhidao_filter.csv'
    
    suffix = '.repaired.csv'
    if not exists(finace_data_file[0: -4] + suffix):
        repair_line_error_csv_file(finace_data_file, save_suffix=suffix, read_encoding='utf-8')

    
    def process_function(line: str) -> dict:
        # title,prompt,reply,is_best
        item = line.split(',') # csv文件逗号分割
        if len(item) < 4:
            print(item)
            return None

        if len(item[0]) + len(item[1]) < prompt_less_word or len(item[2]) < response_less_word: 
            return None

        # 数据清洗
        prompt = ''
        if get_sentences_dice_similarity(item[0], item[1]) >= 0.90:
            # title 和prompt 相似度过高，只用最长的作为问题
            prompt = item[0] if len(item[0]) > len(item[0]) else item[1]
        else:
            # title 和 ask 拼接形成问题
            prompt = "{}{}".format(item[0], item[1])

        # 删除\r
        prompt = prompt.replace('\r','') 

        # 删除重复的标点符号
        prompt = remove_duplicate_punctuation(prompt)

        # 去除重复的标点符号
        response = ''.join(item[2]).replace('\r','')
        response = remove_duplicate_punctuation(response)

        # 剔除问题和答案过短的数据
        if len(prompt) < prompt_less_word or len(response) < response_less_word:
            return None
        
        write_obj = {
                "prompt": prompt,
                "response": response,
            }

        return write_obj

  
    read_file = finace_data_file[0: -4] + suffix
    write_file = PROJECT_ROOT + '/data/my_data/' + read_file.split('/')[-1][0: -(len(suffix))] + '.parquet'

    # 后续append写入，存在文件先删除
    if exists(write_file): 
        assert delete_file(write_file)

    read_and_write_template(read_file, write_file, process_function)


def process_zhihu_kol_dataset(prompt_less_word: int=4, response_less_word: int=10, group_cnt: int=10000) -> None:
    '''
    处理知乎数据集
    
    '''
    raw_zhihu_data_path = abspath(dirname(dirname(__file__))) + '/data/raw_data/zhihu-kol'
    file_names = []
    suffix = '.parquet'
    for root, _, files in walk(raw_zhihu_data_path):
        for file in files:
            if file.endswith(suffix):
                file_names.append(root + '/' + file)
    
    
    def process_function(sentence: str) -> str:
        '''
        针对一个句子的数据清洗
        '''
        # 删除\r
        sentence = sentence.replace('\r','') 

        # 删除重复的标点符号
        sentence = remove_duplicate_punctuation(sentence)

        return sentence

    # row keys :['INSTRUCTION', 'RESPONSE', 'SOURCE', 'METADATA']
    save_file = PROJECT_ROOT + '/data/my_data/zhihu_kol.parquet'
    
    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    all_cnt, keep_cnt = 0, 0
    cur_rows = []
    append = cur_rows.append
    for file in file_names:
        pf = pq.read_table(file)
        log.info('process file: {}'.format(file), save_to_file=True)
        
        # 用于存储当前文件的前10行数据
        file_sample_rows = []
        file_row_cnt = 0

        for prompt, response in progress.track(zip(pf['INSTRUCTION'], pf['RESPONSE']), total=pf.num_rows):
            all_cnt += 1
            prompt, response = prompt.as_py(), response.as_py()
            
            prompt = process_function(prompt)
            response = process_function(response)

            if len(prompt) < prompt_less_word or len(response) < response_less_word:
                continue
            
            keep_cnt += 1
            write_dict = {
                'prompt': prompt,
                'response': response,
            }
            append(write_dict)
            
            # 保存当前文件的前10行样例
            if file_row_cnt < 10:
                file_sample_rows.append(write_dict)
                file_row_cnt += 1

            if len(cur_rows) >= group_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(save_file, df)
                cur_rows = []
                append = cur_rows.append
        
        # 输出当前文件处理完成后的前10行数据
        log.info('=' * 80, save_to_file=True)
        log.info('文件 {} 处理完成，前{}行数据如下：'.format(file, len(file_sample_rows)), save_to_file=True)
        for idx, row in enumerate(file_sample_rows, 1):
            log.info('第{}行 - prompt: {}'.format(idx, row['prompt']), save_to_file=True)
            log.info('第{}行 - response: {}'.format(idx, row['response']), save_to_file=True)
            log.info('-' * 80, save_to_file=True)
        log.info('=' * 80, save_to_file=True)
            
    # end for 
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info('save file to: {}, 全部数据共{}行，清洗后剩余{}行'.format(save_file, all_cnt, keep_cnt), save_to_file=True)


def process_belle_knowledge_enhanced_dataset(response_less_words: int=15, group_cnt: int=10000) -> None:
    '''
    处理belle开源的知识增强数据集
    '''
    file_names = [
    ]

    save_file = PROJECT_ROOT + '/data/my_data/my_belll_3M_cn.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    def process_function(line: str) -> dict:
        '''
        每行的处理函数
        '''
        item = ujson.loads(line)
        prompt = item['instruction']
        response = item['output']

        # 剔除翻译任务
        if '翻译' in prompt or 'translate' in prompt.lower():
            return None
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return None

        if len(response) < response_less_words:
            return None
        
        prompt = remove_duplicate_punctuation(prompt)
        response = remove_duplicate_punctuation(response)

        if len(response) < response_less_words:
            return None

        write_dict = {
            'prompt': prompt,
            'response': response
        }

        return write_dict

    for file in file_names:
        file = PROJECT_ROOT + file

        read_and_write_template(file, save_file, process_function)

def convert_wiki_to_simple_zh(buffer_size: int=10000) -> None:
    '''
    将繁体wiki转换为简体Wiki
    '''
    raw_zh_wiki_file = PROJECT_ROOT + '/data/raw_data/wiki.txt'
    save_zh_wiki_simple_file = PROJECT_ROOT + '/data/raw_data/wiki.simple.txt' 

    if exists(save_zh_wiki_simple_file): 
        assert delete_file(save_zh_wiki_simple_file)

    cc = OpenCC('t2s')
    cur_rows = []
    append = cur_rows.append
    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_f:
        with open(save_zh_wiki_simple_file, 'a', encoding='utf-8') as write_f:
            for line in read_f:
                line = procees_line(line)
                if len(line.strip()) == 0: continue

                line = '{}\n'.format(line)
                append(line)

                if len(cur_rows) >= buffer_size:
                    write_f.writelines(cur_rows)
                    cur_rows = []
                    append = cur_rows.append
            
            if len(cur_rows) > 0:
                write_f.writelines(cur_rows)
                cur_rows = []
        

def process_zh_wiki_data_to_datset(groups_cnt: int=10000, max_len: int=512, seed: int=23333) -> None:
    '''
    将Wiki中文数转换为问答数据集
    wiki 下载地址：https://dumps.wikimedia.org/zhwiki/
    将下载的bz2文件转换为wiki.txt参考：https://github.com/apertium/WikiExtractor
    '''
    raw_zh_wiki_file = PROJECT_ROOT + '/data/raw_data/wiki.txt'
    zhwiki_simple_file = PROJECT_ROOT + '/data/my_data/wiki_zh_simple.parquet'

    # 删除已经存在的数据
    if exists(zhwiki_simple_file): 
        assert delete_file(zhwiki_simple_file)

    # 将繁体转换为简体
    cc = OpenCC('t2s')
    all_cnt, keep_cnt = 0, 0
    
    # 构造问题的前缀
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def procees_line(line: str) -> str:
        '''
        处理一行文本
        '''
        # 将繁体转换为简体
        line = cc.convert(line)

        line = re.sub(r"\「|\」|\｢|\｣|\『|\』", '\"', line)  # 将「」｢｣『』这些符号替换成引号
        line = re.sub(r"\，\）|\；\）", '）', line)  # 罗德·法尼(Rod Dodji Fanni，）
        line = re.sub(r"\（\，|\(\，", '（', line)  # 阿魯拉·基馬(Alula Girma (，
        
        line = convert_en_punctuation_to_zh_punct(line) # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点

        return line
        
    np.random.seed(seed)
    choice = np.random.choice

    with progress.open(raw_zh_wiki_file, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        for line in read_file:
            all_cnt += 1

            # prompt已经保存，但是仍有多余的行，这些行使得response的长度＞max_len，故跳过，不处理
            if len(prompt) == 0 and pre_line_len > 0:
                pre_line_len = len(line.strip())
                continue
            
            # 清洗一行
            line = procees_line(line)
            

            # 确定问题，pre_line_len是0，既是上一行是空行，则当前行是新的百科词条，设置为prompt
            if prompt == '' and line.endswith('：') and pre_line_len == 0:
                prompt = choice(prompt_prefix).format(line[0: -1])
                continue

            pre_line_len = len(line_stripped)

            # 问题下来若干行为答案
            if prompt != '' and not line_stripped.endswith(':'):
                # 其实，pre_line_len已经是len(line_stripped)了，如果len(line_stripped)=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(response) + len(line) <= max_len and pre_line_len != 0: 
                    response = '{}{}'.format(response, line)
                elif len(response) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    response = '{}{}'.format(response, line)
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    prompt = ''
                    response = ''

            # =groups_cnt保存到文件
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(zhwiki_simple_file, df)
                cur_rows = []
                append = cur_rows.append

        # end for
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(zhwiki_simple_file, df)
            cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行".format(zhwiki_simple_file, all_cnt, keep_cnt), save_to_file=True)


def process_wiki_simple_to_dataset(groups_cnt: int=10000, max_len: int=512, seed: int=23333, skip_clean: bool=True) -> None:
    '''
    将wiki.simple.txt转换为问答数据集
    注意：wiki.simple.txt已经是简体中文，不需要再转换
    
    Args:
        groups_cnt: 每次写入parquet文件的行数
        max_len: 答案的最大长度
        seed: 随机种子
        skip_clean: 是否跳过数据清洗（默认True，直接使用wiki.simple.txt的原始内容）
    '''
    wiki_simple_file = PROJECT_ROOT + '/data/wiki.simple.txt'
    zhwiki_simple_parquet = PROJECT_ROOT + '/data/my_data/wiki_zh_simple.parquet'

    # 检查文件是否存在
    if not exists(wiki_simple_file):
        log.error(f"未找到文件: {wiki_simple_file}", save_to_file=True)
        log.error("请先运行 tokenize/process_zhwiki.py 生成 wiki.simple.txt", save_to_file=True)
        return

    # 删除已经存在的数据
    if exists(zhwiki_simple_parquet): 
        assert delete_file(zhwiki_simple_parquet)

    all_cnt, keep_cnt = 0, 0
    
    # 构造问题的前缀
    prompt_prefix = [
        '什么是{}？',
        '介绍一下{}',
        '介绍一下什么是{}',
        '写一篇关于{}的介绍',
        '{}是什么？',
        '你知道{}吗？',
        '生成关于{}的介绍',
        '我想知道关于{}的详细信息',
        '你了解{}吗？',
        '请解释一下{}',
        '对于{}，你有什么了解或看法吗？',
        '请告诉我关于{}的信息',
        '请简要描述一下{}',
        '请提供有关{}的一些详细信息',
        '能否解释一下{}是什么?',
        '请分享一些关于{}的背景知识',
        '请简要概括一下{}',
        '能给我一些关于{}的背景资料吗?',
        '有关{}的信息可以分享一下吗？',
        '你能告诉我{}是什么吗？',
    ]

    def process_line(line: str, skip_clean: bool) -> str:
        '''
        处理一行文本（wiki.simple.txt已经是简体中文，只需要基本清洗）
        
        Args:
            line: 输入行
            skip_clean: 是否跳过清洗（True则直接返回原始内容）
        '''
        if skip_clean:
            return line  # 直接返回原始内容，不做任何处理
        
        line = convert_en_punctuation_to_zh_punct(line)  # 英文标点转换为中文标点
        line = remove_duplicate_punctuation(line)  # 删除中文空括号和重复的标点
        return line
        
    np.random.seed(seed)
    choice = np.random.choice

    # 添加调试计数器
    title_cnt = 0  # 识别到的标题数量
    content_cnt = 0  # 处理的内容行数量
    saved_cnt = 0  # 保存的问答对数量
    
    log.info(f"开始处理wiki数据，skip_clean={skip_clean}", save_to_file=True)
    log.info(f"判断冒号类型: {'英文冒号(:)' if skip_clean else '中文冒号(：)'}", save_to_file=True)
    
    with progress.open(wiki_simple_file, 'r', encoding='utf-8') as read_file:
        prompt = '' 
        response = '' 
        pre_line_len = 0
        cur_rows = []
        append = cur_rows.append
        
        for line in read_file:
            all_cnt += 1

            # 先strip获取原始行的长度信息
            line_stripped = line.strip()
            
            # 每处理10万行打印一次进度
            if all_cnt % 100000 == 0:
                log.info(f"已处理 {all_cnt} 行，识别标题 {title_cnt} 个，内容行 {content_cnt} 行，保存 {saved_cnt} 条", save_to_file=True)
            
            # 确定问题：识别标题行
            # 1. 主标题：以英文冒号结尾，且上一行是空行（pre_line_len == 0）
            # 2. 子标题：以句号结尾，且长度较短（<= 20个字符）
            is_title = False
            title = ''
            
            # 主标题：数学:
            if prompt == '' and line_stripped.endswith(':') and pre_line_len == 0:
                title = line_stripped[0: -1]  # 去掉末尾的冒号
                is_title = True
            # 子标题：词源. 历史. 等
            elif line_stripped.endswith('.') and len(line_stripped) <= 20 and len(line_stripped) > 0:
                title = line_stripped[0: -1]  # 去掉末尾的句号
                is_title = True
            
            if is_title:
                # 如果之前有未保存的内容，先保存
                if prompt != '' and response != '':
                    keep_cnt += 1
                    saved_cnt += 1
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    
                    # 打印前5个保存的样例
                    if saved_cnt <= 5:
                        log.info(f"保存样例 #{saved_cnt}: prompt='{prompt[:50]}...', response='{response[:50]}...'", save_to_file=True)
                
                # 设置新的标题
                prompt = choice(prompt_prefix).format(title)
                response = ''
                pre_line_len = len(line_stripped)
                title_cnt += 1
                
                # 打印前10个标题作为调试信息
                if title_cnt <= 10:
                    log.info(f"识别标题 #{title_cnt}: '{title}' -> prompt: '{prompt}'", save_to_file=True)
                
                continue
            
            # 清洗一行（只对内容行进行清洗）
            line = process_line(line_stripped, skip_clean)
            
            pre_line_len = len(line_stripped)

            # 问题下来若干行为答案
            # 注意：如果skip_clean=True，使用英文冒号；否则使用中文冒号（因为清洗会转换）
            colon_to_check = ':' if skip_clean else '：'
            
            # 调试：打印前几行的判断逻辑
            if content_cnt < 20 and prompt != '':
                log.info(f"内容行 #{content_cnt}: line末尾='{line[-10:] if len(line) > 10 else line}', "
                        f"endswith('{colon_to_check}')={line.endswith(colon_to_check)}, "
                        f"len(line)={len(line)}, pre_line_len={pre_line_len}", save_to_file=True)
            
            if prompt != '' and not line.endswith(colon_to_check):
                content_cnt += 1
                
                # 其实，pre_line_len已经是len(line_stripped)了，如果len(line_stripped)=0，既是当前行是0，则不管答案长度够不够，都需要保存了
                if len(response) + len(line) <= max_len and pre_line_len != 0:
                    response = '{}{}'.format(response, line)
                elif len(response) + len(line) > max_len or pre_line_len == 0:
                    # 长度超了或者当前的百科已经结束，保存一条样例
                    keep_cnt += 1
                    saved_cnt += 1
                    response = '{}{}'.format(response, line)
                    append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
                    
                    # 打印前5个保存的样例
                    if saved_cnt <= 5:
                        log.info(f"保存样例 #{saved_cnt}: prompt='{prompt[:50]}...', response='{response[:50]}...'", save_to_file=True)
                    
                    prompt = ''
                    response = ''

            # =groups_cnt保存到文件
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(zhwiki_simple_parquet, df)
                cur_rows = []
                append = cur_rows.append

        # end for
        if len(prompt) > 0 and len(response) > 0:
            keep_cnt += 1
            append({'prompt': prompt, 'response': response})

        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(zhwiki_simple_parquet, df)
            cur_rows = []

    log.info("=" * 60, save_to_file=True)
    log.info(f"处理完成统计:", save_to_file=True)
    log.info(f"  - 总行数: {all_cnt}", save_to_file=True)
    log.info(f"  - 识别标题数: {title_cnt}", save_to_file=True)
    log.info(f"  - 处理内容行数: {content_cnt}", save_to_file=True)
    log.info(f"  - 保存问答对数: {saved_cnt}", save_to_file=True)
    log.info(f"  - keep_cnt: {keep_cnt}", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    log.info(f'merge into file: {zhwiki_simple_parquet}, 全部数据共{all_cnt}行，清洗后剩余{keep_cnt}行', save_to_file=True)



def merge_dataset_as_single_file(groups_cnt: int=50000, max_len: int=512, min_len: int=3, cut_max_len: bool=False) -> None:
    '''
    将多个数据集合并为一个数据集
    优化版本：使用分批读取，避免一次性加载整个文件到内存
    '''
    from_parquet_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.parquet')

    save_file = PROJECT_ROOT + '/data/my_dataset.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    cur_rows = []
    all_cnt, keep_cnt = 0, 0
    
    log.info(f"开始合并 {len(from_parquet_files)} 个数据文件...", save_to_file=True)
    
    for file_idx, file in enumerate(from_parquet_files, 1):
        log.info(f"处理文件 [{file_idx}/{len(from_parquet_files)}]: {file}", save_to_file=True)
        
        # 使用ParquetFile分批读取，避免一次性加载整个文件
        source_pf = ParquetFile(file)
        
        # 获取文件总行数
        file_total_rows = 0
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                file_total_rows += len(rows)
        
        # 重新打开文件进行处理
        source_pf = ParquetFile(file)
        file_keep_cnt = 0
        
        with progress.Progress() as prog:
            task = prog.add_task(f"[cyan]处理 {file}...", total=file_total_rows)
            
            for pf_chunk in source_pf:
                for rows in pf_chunk.iter_row_groups():
                    prompts = rows['prompt']
                    responses = rows['response']
                    
                    for i in range(len(prompts)):
                        all_cnt += 1
                        prog.update(task, advance=1)
                        
                        prompt = str(prompts[i])
                        response = str(responses[i])
                        
                        if len(prompt) < min_len or len(response) < min_len:
                            continue
                        
                        if cut_max_len and (len(prompt) > max_len or len(response) > max_len):
                            prompt = prompt[0: max_len]
                            response = response[0: max_len]
                        
                        keep_cnt += 1
                        file_keep_cnt += 1
                        cur_rows.append({'prompt': prompt, 'response': response})
                        
                        if len(cur_rows) >= groups_cnt:
                            df = pd.DataFrame(cur_rows)
                            write_single_parquet_file(save_file, df)
                            cur_rows = []
        
        log.info(f"文件 {file} 处理完成，保留 {file_keep_cnt}/{file_total_rows} 行", save_to_file=True)
        
    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
        cur_rows = []

    log.info("merge into file: {}, 全部数据共{}行，清洗后剩余{}行，保留率: {:.2f}%".format(
        save_file, all_cnt, keep_cnt, (keep_cnt/all_cnt * 100) if all_cnt > 0 else 0
    ), save_to_file=True)


def remove_dataset_duplicate_rows(groups_cnt: int=50000, batch_size: int=100000) -> None:
    '''
    使用mini_hash删除数据集中重复的部分
    优化版本：分批处理，减少内存占用
    
    Args:
        groups_cnt: 每次写入parquet文件的行数
        batch_size: 每批处理的数据量，用于控制内存占用
    '''
    from_parquet_files = PROJECT_ROOT + '/data/my_dataset.parquet'
    save_file = PROJECT_ROOT + '/data/my_dataset_no_dulpticates.parquet'
    temp_index_file = PROJECT_ROOT + '/data/temp_duplicate_indices.txt'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)
    
    if exists(temp_index_file):
        remove(temp_index_file)

    log.info("开始第一阶段：识别重复数据...", save_to_file=True)
    
    # 第一阶段：分批读取数据，识别重复项
    all_cnt = 0
    row_index = -1
    drop_dataset_duplicate = DropDatasetDuplicate(threshold=0.85, num_perm=256)
    
    # 使用ParquetFile进行分批读取
    source_pf = ParquetFile(from_parquet_files)
    
    # 先获取总行数用于进度条
    total_rows = 0
    for pf_chunk in source_pf:
        for rows in pf_chunk.iter_row_groups():
            total_rows += len(rows)
    
    all_cnt = total_rows
    log.info(f"数据集总行数: {all_cnt}", save_to_file=True)
    
    # 重新打开文件进行处理
    source_pf = ParquetFile(from_parquet_files)
    
    with progress.Progress() as prog:
        task = prog.add_task("[cyan]识别重复数据...", total=total_rows)
        
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                prompts = rows['prompt']
                responses = rows['response']
                
                for i in range(len(prompts)):
                    row_index += 1
                    doc = f"{prompts[i]}{responses[i]}"
                    drop_dataset_duplicate.add_doc(index=row_index, doc=doc)
                    prog.update(task, advance=1)
    
    # 获取需要删除的索引
    need_to_drop_indexs = drop_dataset_duplicate.get_duplicate_indexs()
    duplicate_count = len(need_to_drop_indexs)
    log.info(f"识别到 {duplicate_count} 条重复数据", save_to_file=True)
    
    # 释放内存
    del drop_dataset_duplicate
    
    # 将重复索引保存到临时文件（如果数量很大）
    if duplicate_count > 1000000:
        log.info("重复数据量较大，使用临时文件存储索引...", save_to_file=True)
        with open(temp_index_file, 'w') as f:
            for idx in need_to_drop_indexs:
                f.write(f"{idx}\n")
        # 清空内存中的集合
        need_to_drop_indexs.clear()
        # 重新加载为集合（这样查询更快）
        with open(temp_index_file, 'r') as f:
            need_to_drop_indexs = set(int(line.strip()) for line in f)
    
    log.info("开始第二阶段：过滤并保存数据...", save_to_file=True)
    
    # 第二阶段：分批读取并写入非重复数据
    cur_rows = []
    keep_cnt = 0
    row_index = -1
    
    source_pf = ParquetFile(from_parquet_files)
    
    with progress.Progress() as prog:
        task = prog.add_task("[green]过滤重复数据...", total=total_rows)
        
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                prompts = rows['prompt']
                responses = rows['response']
                
                for i in range(len(prompts)):
                    row_index += 1
                    prog.update(task, advance=1)
                    
                    # 重复的行跳过
                    if row_index in need_to_drop_indexs:
                        continue
                    
                    cur_rows.append({
                        'prompt': str(prompts[i]), 
                        'response': str(responses[i])
                    })
                    keep_cnt += 1
                    
                    # 达到批次大小，写入文件
                    if len(cur_rows) >= groups_cnt:
                        df = pd.DataFrame(cur_rows)
                        write_single_parquet_file(save_file, df)
                        cur_rows = []
    
    # 处理末尾部分
    if len(cur_rows) > 0:
        df = pd.DataFrame(cur_rows)
        write_single_parquet_file(save_file, df)
    
    # 清理临时文件
    if exists(temp_index_file):
        remove(temp_index_file)
    
    log.info("merge into file: {}, 全部数据共{}行，文档去重后剩余{}行，去重率: {:.2f}%".format(
        save_file, all_cnt, keep_cnt, (1 - keep_cnt/all_cnt) * 100 if all_cnt > 0 else 0
    ), save_to_file=True)

def shuffle_parquet_dataset(parquet_file: str, shuffle_file: str, seed: int=23333, groups_cnt: int=65536) -> None:
    '''
    打乱一个parquet文件数据集
    优化版本：使用索引打乱而非加载整个数据集，大幅减少内存占用
    '''
    if not exists(parquet_file):
        raise Exception('can not find parquet file: {}'.format(parquet_file))
    
    log.info('开始打乱数据集...', save_to_file=True)
    
    # 第一步：获取总行数
    log.info('第一阶段：统计数据集大小...', save_to_file=True)
    source_pf = ParquetFile(parquet_file)
    total_rows = 0
    for pf_chunk in source_pf:
        for rows in pf_chunk.iter_row_groups():
            total_rows += len(rows)
    
    log.info(f'数据集总行数: {total_rows}', save_to_file=True)
    
    # 第二步：生成打乱的索引
    log.info('第二阶段：生成随机索引...', save_to_file=True)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(total_rows)
    
    # 第三步：按照打乱的索引读取数据
    log.info('第三阶段：按随机顺序重组数据...', save_to_file=True)
    
    if exists(shuffle_file): 
        assert delete_file(shuffle_file)
    
    # 先读取所有数据到列表（按原始顺序）
    all_data = []
    source_pf = ParquetFile(parquet_file)
    
    with progress.Progress() as prog:
        task = prog.add_task("[cyan]读取数据...", total=total_rows)
        
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                prompts = rows['prompt']
                responses = rows['response']
                
                for i in range(len(prompts)):
                    all_data.append({
                        'prompt': str(prompts[i]),
                        'response': str(responses[i])
                    })
                    prog.update(task, advance=1)
    
    # 按照打乱的索引重组数据并分批写入
    log.info('重组并写入数据...', save_to_file=True)
    cur_rows = []
    
    with progress.Progress() as prog:
        task = prog.add_task("[green]写入打乱后的数据...", total=total_rows)
        
        for idx in shuffled_indices:
            cur_rows.append(all_data[idx])
            prog.update(task, advance=1)
            
            if len(cur_rows) >= groups_cnt:
                df = pd.DataFrame(cur_rows)
                write_single_parquet_file(shuffle_file, df)
                cur_rows = []
        
        # 处理末尾部分
        if len(cur_rows) > 0:
            df = pd.DataFrame(cur_rows)
            write_single_parquet_file(shuffle_file, df)
    
    # 释放内存
    del all_data
    del shuffled_indices
    
    log.info(f'数据打乱完成，已保存到: {shuffle_file}', save_to_file=True)

def count_my_json_data() -> None:
    '''
    统计目前的所有数据集数据量
    '''
    my_data_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.json')
    result = [['file_name', 'count']]
    all_cnt = 0
    for file in my_data_files:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        with progress.open(file, 'r', encoding='utf-8') as f:
            for _ in f:
                cur_cnt += 1
        
        all_cnt += cur_cnt
        result.append([file_name, cur_cnt])
    
    result.append(['汇总', all_cnt])

    log.info(str(result), save_to_file=True)

    console = Console()
    table = Table(show_header=True, show_lines=True,)

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)): # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)


def count_my_parquet_data(parquet_file: str=None) -> None:
    '''
    统计dir目录下所有parquet数据集数据量
    '''
    my_data_files = []

    if not parquet_file:
        my_data_files = get_path_of_suffix_files(PROJECT_ROOT + '/data/my_data', '.parquet')
    elif isdir(parquet_file):
        my_data_files = get_path_of_suffix_files(parquet_file, '.parquet')
    elif parquet_file.endswith('.parquet'):
        my_data_files = [parquet_file]
        

    result = [['file_name', 'count']]
    all_cnt = 0
    for file in my_data_files:
        file_name = file.split('/')[-1]
        cur_cnt = 0
        pf = ParquetFile(file)

        for pf_chunk in pf:
            cur_cnt += pf_chunk.info['rows']
        
        all_cnt += cur_cnt
        result.append([file_name, cur_cnt])
    
    result.append(['汇总', all_cnt])

    log.info(str(result), save_to_file=True)

    console = Console()
    table = Table(show_header=True, show_lines=True,)

    for col in result[0]:
        table.add_column(col)
    for i in range(1, len(result)): # 跳过表头
        table.add_row(str(result[i][0]), str(result[i][1]))

    console.print(table)    


def split_train_valid_test_datasets(source_parquet_file: str, max_len: int=320, seed: int=23333, train_ratio: float=0.91, test_ratio: float=0.0875, valid_ratio: float=0.0025, groups_cnt: int=50000) -> None:
    '''
    将原始数据拆分为训练集、测试集和验证集
    优化版本：使用分批读取，避免一次性加载整个数据集到内存
    '''
    assert train_ratio + test_ratio + valid_ratio == 1.0

    train_parquet_file = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    test_parquet_file = PROJECT_ROOT + '/data/my_test_dataset.parquet'
    valid_parquet_file = PROJECT_ROOT + '/data/my_valid_dataset.parquet'

    if exists(train_parquet_file): assert delete_file(train_parquet_file)
    if exists(test_parquet_file): assert delete_file(test_parquet_file)
    if exists(valid_parquet_file): assert delete_file(valid_parquet_file)

    np.random.seed(seed)

    train, test, valid = [], [], []
    train_cnt, test_cnt, valid_cnt = 0, 0, 0
    
    log.info('开始划分数据集...', save_to_file=True)
    
    # 获取总行数
    source_pf = ParquetFile(source_parquet_file)
    total_rows = 0
    for pf_chunk in source_pf:
        for rows in pf_chunk.iter_row_groups():
            total_rows += len(rows)
    
    log.info(f'数据集总行数: {total_rows}', save_to_file=True)
    log.info(f'划分比例 - 训练集: {train_ratio*100:.1f}%, 测试集: {test_ratio*100:.1f}%, 验证集: {valid_ratio*100:.1f}%', save_to_file=True)
    
    # 重新打开文件进行处理
    source_pf = ParquetFile(source_parquet_file)
    
    with progress.Progress() as prog:
        task = prog.add_task("[cyan]划分数据集...", total=total_rows)
        
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                prompts = rows['prompt']
                responses = rows['response']
                
                for i in range(len(prompts)):
                    prog.update(task, advance=1)
                    
                    prompt = str(prompts[i])
                    response = str(responses[i])
                    
                    rand = np.random.random()
                    cur_data = {
                        'prompt': prompt[0: max_len] if len(prompt) > max_len else prompt,
                        'response': response[0: max_len] if len(response) > max_len else response
                    }

                    if 0 <= rand < train_ratio:
                        train.append(cur_data)
                        train_cnt += 1
                    elif train_ratio <= rand < train_ratio + test_ratio:
                        test.append(cur_data)
                        test_cnt += 1
                    else:
                        valid.append(cur_data)
                        valid_cnt += 1
                    
                    if len(train) >= groups_cnt:
                        write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
                        train = []
                    
                    if len(test) >= groups_cnt:
                        write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
                        test = []
                    
                    if len(valid) >= groups_cnt:
                        write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
                        valid = []

    # 处理末尾部分
    if len(train) > 0:
        write_single_parquet_file(train_parquet_file, pd.DataFrame(train))
        train = []
    
    if len(test) > 0:
        write_single_parquet_file(test_parquet_file, pd.DataFrame(test))
        test = []
    
    if len(valid) > 0:
        write_single_parquet_file(valid_parquet_file, pd.DataFrame(valid))
        valid = []
    
    log.info('数据集划分完成！', save_to_file=True)
    log.info(f'训练集: {train_cnt} 行 ({train_cnt/total_rows*100:.2f}%)', save_to_file=True)
    log.info(f'测试集: {test_cnt} 行 ({test_cnt/total_rows*100:.2f}%)', save_to_file=True)
    log.info(f'验证集: {valid_cnt} 行 ({valid_cnt/total_rows*100:.2f}%)', save_to_file=True)

def parquet_to_text(sep='[SEP]', buffer_size: int=50000) -> None:
    '''
    将parquet文件转换为txt预料，句子之间用sep隔开
    txt文件用于训练tokenizer，使用huggingface的BPE训练会导致OOM
    '''
    parquet_file = PROJECT_ROOT + '/data/my_dataset.parquet'
    txt_file = PROJECT_ROOT + '/data/my_corpus.txt'

    if exists(txt_file): 
        assert delete_file(txt_file)

    source_pf = ParquetFile(parquet_file)
    cur_rows = []
    append = cur_rows.append
    with open(txt_file, 'a', encoding='utf-8') as f_write:
        for pf_chunk in progress.track(source_pf):
            for rows in pf_chunk.iter_row_groups():
                for prompt, response in zip(rows['prompt'], rows['response']):
                    append(prompt + sep + response + sep + '\n')

                    if len(cur_rows) >= buffer_size:
                        f_write.writelines(cur_rows)
                        cur_rows = []
                        append = cur_rows.append
                       
        # end for
        if len(cur_rows) > 0:
            f_write.writelines(cur_rows)
            cur_rows = []

def parquet_to_json(buffer_size: int=10000) -> None:
    '''
    将parquet文件转换为json
    
    优化说明：
    1. 使用流式写入，避免将所有数据加载到内存
    2. 分批收集数据，达到 buffer_size 时写入文件
    3. 手动构建 JSON 格式，避免 ujson.dump 的内存开销
    '''
    parquet_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'
    json_file = PROJECT_ROOT + '/data/sft_train.json'

    if exists(json_file): 
        assert delete_file(json_file)

    source_pf = ParquetFile(parquet_file)
    
    log.info(f'开始转换 parquet 到 json: {parquet_file} -> {json_file}', save_to_file=True)
    
    total_count = 0
    is_first_batch = True
    
    with open(json_file, 'w', encoding='utf-8') as f:
        # 写入 JSON 数组开始符号
        f.write('[\n')
        
        cur_rows = []
        
        for pf_chunk in progress.track(source_pf, description="转换中..."):
            for rows in pf_chunk.iter_row_groups():
                # 使用向量化操作（pandas DataFrame 使用 tolist()）
                prompts = rows['prompt'].tolist()
                responses = rows['response'].tolist()
                
                for prompt, response in zip(prompts, responses):
                    # 过滤空数据
                    if len(response) == 0 or len(prompt) == 0:
                        continue
                    
                    cur_rows.append({
                        'prompt': str(prompt),
                        'response': str(response),
                    })
                    
                    # 达到缓冲区大小时写入
                    if len(cur_rows) >= buffer_size:
                        for i, row in enumerate(cur_rows):
                            # 如果不是第一条数据，前面加逗号
                            if not is_first_batch or i > 0:
                                f.write(',\n')
                            # 写入 JSON 对象（缩进格式）
                            f.write('    ')
                            ujson.dump(row, f, ensure_ascii=False)
                            is_first_batch = False
                        
                        total_count += len(cur_rows)
                        cur_rows = []
        
        # 写入剩余数据
        if cur_rows:
            for i, row in enumerate(cur_rows):
                if not is_first_batch or i > 0:
                    f.write(',\n')
                f.write('    ')
                ujson.dump(row, f, ensure_ascii=False)
                is_first_batch = False
            total_count += len(cur_rows)
        
        # 写入 JSON 数组结束符号
        f.write('\n]')
    
    log.info(f'转换完成！共转换 {total_count} 条数据', save_to_file=True)
    log.info(f'JSON 文件已保存到: {json_file}', save_to_file=True)

def dataset_length_cnt() -> None:
    '''
    统计数据集中 prompt 和 response 的长度分布
    
    优化说明：
    1. 使用 ParquetFile 迭代器分批读取，避免一次性加载整个文件
    2. 流式统计，内存占用稳定
    3. 添加进度显示和详细日志
    '''
    dataset_file = PROJECT_ROOT + '/data/my_dataset.shuffle.parquet'
    
    if not exists(dataset_file):
        log.error(f'数据集文件不存在: {dataset_file}', save_to_file=True)
        return
    
    log.info(f'开始统计数据集长度分布: {dataset_file}', save_to_file=True)
    
    # 使用 ParquetFile 迭代器分批读取
    source_pf = ParquetFile(dataset_file)
    
    que_len_dict, ans_len_dict = defaultdict(int), defaultdict(int)
    total_rows = 0
    
    # 分批处理
    with progress.Progress() as prog:
        task = prog.add_task("[cyan]统计长度分布...", total=None)
        
        for pf_chunk in source_pf:
            for rows in pf_chunk.iter_row_groups():
                # 使用向量化操作（pandas DataFrame 使用 tolist()）
                prompts = rows['prompt'].tolist()
                responses = rows['response'].tolist()
                
                for prompt, response in zip(prompts, responses):
                    prompt = str(prompt)
                    response = str(response)
                    
                    que_len_dict[len(prompt)] += 1
                    ans_len_dict[len(response)] += 1
                    total_rows += 1
                
                prog.update(task, advance=len(prompts))
    
    log.info(f'统计完成！共处理 {total_rows} 条数据', save_to_file=True)
    
    # 转换为列表格式
    que_len, ans_len = [], []
    for k, v in que_len_dict.items():
        que_len.append([k, v])
    for k, v in ans_len_dict.items():
        ans_len.append([k, v])

    def gather_gt_x(array: list[tuple], x: int=512) -> list:
        '''
        长度大于x的合并在一起
        '''
        new_array = []
        gt_x_cnt = 0
        for item in array:
            if item[0] < x:
                new_array.append([item[0], item[1]])
            else:
                gt_x_cnt += item[1]
        new_array.append([x, gt_x_cnt])

        return new_array
    
    max_len = 512
    ans_list = gather_gt_x(ans_len, max_len)
    ans_list.sort(key=lambda x: x[0])
    que_list = gather_gt_x(que_len, max_len)
    que_list.sort(key=lambda x: x[0])
    
    ans_pd = pd.DataFrame(ans_list, columns=['length', 'count'])
    que_pd = pd.DataFrame(que_list, columns=['length', 'count'])

    def plot_sub_bar(plt, x, y, title: str, color: str='g') ->None:
        plt.bar(x, y, color=color, label='sample count')
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
        plt.legend()
        plt.xlabel('length')
        plt.ylabel('count')
        plt.title(title)

    plt.figure(figsize=(10, 10),dpi=200)
    plt.subplot(2, 2, 1)
    plot_sub_bar(plt, que_pd['length'], que_pd['count'], title='prompt length', color='c')

    plt.subplot(2, 2, 2)
    plot_sub_bar(plt, ans_pd['length'], ans_pd['count'], title='response length', color='g')

    le512_pd = ans_pd[ans_pd['length'] < 512]
    plt.subplot(2, 2, 3)
    plot_sub_bar(plt, le512_pd['length'], le512_pd['count'], title='response length < 512', color='limegreen')

    le320_pd = ans_pd[ans_pd['length'] < 320]
    plt.subplot(2, 2, 4)
    plot_sub_bar(plt, le320_pd['length'], le320_pd['count'], title='response length < 320', color='limegreen')

    output_file = PROJECT_ROOT + '/img/sentence_length.png'
    plt.savefig(output_file)
    log.info(f'长度分布图已保存到: {output_file}', save_to_file=True)
    plt.show()

def process_belle_knowledge_enhanced_dataset_for_finetune(max_len: int=320, group_cnt: int=50000) -> None:
    '''
    处理belle开源的知识增强数据集
    
    优化说明：
    1. 使用 ParquetFile 迭代器分批读取，避免一次性加载整个文件到内存
    2. 使用向量化操作替代 iterrows()，提升处理速度
    3. 提取公共过滤函数，减少重复代码
    4. 添加进度显示
    '''
    # 使用data/raw_data/belle目录下的指定parquet文件
    raw_data_dir = PROJECT_ROOT + '/data/raw_data/belle'
    save_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'

    # 后续append写入，存在文件先删除
    if exists(save_file): 
        assert delete_file(save_file)

    # 指定要处理的三个parquet文件
    parquet_files = [
        f'{raw_data_dir}/generated_chat_0.4M.parquet',
        f'{raw_data_dir}/train_0.5M_CN.parquet',
        f'{raw_data_dir}/train_2M_CN.parquet'
    ]
    
    # 翻译相关关键词（预编译，避免重复创建）
    translate_keywords = ('翻译', '英译', '译英', '中译', '译中', '汉译', '译汉')
    
    def should_filter_data(prompt: str, response: str) -> bool:
        """
        判断数据是否应该被过滤掉
        返回 True 表示应该过滤（不保留），False 表示保留
        """
        # 过滤空值（最重要的检查，放在最前面）
        if not prompt or not response:
            return True
        
        prompt_stripped = prompt.strip()
        response_stripped = response.strip()
        
        if len(prompt_stripped) == 0 or len(response_stripped) == 0:
            return True
        
        # 剔除翻译任务
        if 'translate' in prompt.lower():
            return True
        for word in translate_keywords:
            if word in prompt:
                return True
        
        # 删除表格类任务
        if '表格' in prompt or '-----' in prompt or '-----' in response:
            return True
        
        # 长度过滤
        if len(prompt) > max_len or len(response) > max_len:
            return True
        
        return False
    
    log.info(f'将处理 {len(parquet_files)} 个parquet文件', save_to_file=True)
    
    all_cnt = 0
    keep_cnt = 0
    
    for file_idx, file_path in enumerate(parquet_files, 1):
        log.info(f'[{file_idx}/{len(parquet_files)}] 处理文件: {file_path}', save_to_file=True)
        
        if not exists(file_path):
            log.warning(f'文件不存在，跳过: {file_path}', save_to_file=True)
            continue
        
        try:
            # 使用 ParquetFile 迭代器分批读取
            source_pf = ParquetFile(file_path)
            
            # 先读取一小部分数据来识别列名
            first_chunk = next(source_pf.iter_row_groups())
            columns = first_chunk.columns.tolist()
            log.info(f'文件列名: {columns}', save_to_file=True)
            
            # 重新创建迭代器（因为已经消耗了第一个chunk）
            source_pf = ParquetFile(file_path)
            
            file_sample_rows = []
            batch_data = []
            file_all_cnt = 0
            file_keep_cnt = 0
            
            # 检查是否是conversations格式
            if 'conversations' in columns:
                log.info('检测到 conversations 格式', save_to_file=True)
                
                # 分批处理
                with progress.Progress() as prog:
                    task = prog.add_task(f"[cyan]处理 {file_path.split('/')[-1]}...", total=None)
                    
                    for pf_chunk in source_pf:
                        for rows in pf_chunk.iter_row_groups():
                            # 获取 conversations 列（pandas DataFrame 使用 tolist()）
                            conversations_list = rows['conversations'].tolist()
                            
                            for conversations in conversations_list:
                                if not isinstance(conversations, list):
                                    continue
                                
                                # 提取所有human和assistant的对话
                                for i in range(len(conversations) - 1):
                                    if conversations[i].get('from') == 'human' and conversations[i+1].get('from') == 'assistant':
                                        file_all_cnt += 1
                                        prompt = conversations[i].get('value', '')
                                        response = conversations[i+1].get('value', '')
                                        
                                        # 过滤数据
                                        if should_filter_data(prompt, response):
                                            continue
                                        
                                        file_keep_cnt += 1
                                        
                                        # 收集前10行样例
                                        if len(file_sample_rows) < 10:
                                            file_sample_rows.append({'prompt': prompt, 'response': response})
                                        
                                        batch_data.append({'prompt': prompt, 'response': response})
                                        
                                        # 批量写入
                                        if len(batch_data) >= group_cnt:
                                            df = pd.DataFrame(batch_data)
                                            write(save_file, df, append=exists(save_file), compression='GZIP')
                                            batch_data = []
                                            prog.update(task, advance=group_cnt)
                
                # 写入剩余数据
                if batch_data:
                    df = pd.DataFrame(batch_data)
                    write(save_file, df, append=exists(save_file), compression='GZIP')
                
            else:
                # 识别普通格式的列名（优先级顺序很重要！）
                prompt_col = None
                response_col = None
                
                # 定义列名优先级（从高到低）
                prompt_priority = ['instruction', 'prompt', 'question', 'input']  # instruction 优先级最高
                response_priority = ['output', 'response', 'answer', 'target']
                
                # 按优先级查找 prompt 列
                for candidate in prompt_priority:
                    for col in columns:
                        if col.lower() == candidate:
                            prompt_col = col
                            break
                    if prompt_col:
                        break
                
                # 按优先级查找 response 列
                for candidate in response_priority:
                    for col in columns:
                        if col.lower() == candidate:
                            response_col = col
                            break
                    if response_col:
                        break
                
                if not prompt_col or not response_col:
                    log.error(f'无法识别文件列名: {file_path}, 列名为: {columns}', save_to_file=True)
                    continue
                
                log.info(f'使用列: prompt={prompt_col}, response={response_col}', save_to_file=True)
                
                # 分批处理
                with progress.Progress() as prog:
                    task = prog.add_task(f"[cyan]处理 {file_path.split('/')[-1]}...", total=None)
                    
                    for pf_chunk in source_pf:
                        for rows in pf_chunk.iter_row_groups():
                            # 使用向量化操作，避免 iterrows()（pandas DataFrame 使用 tolist()）
                            prompts = rows[prompt_col].tolist()
                            responses = rows[response_col].tolist()
                            
                            for prompt, response in zip(prompts, responses):
                                file_all_cnt += 1
                                prompt = str(prompt)
                                response = str(response)
                                
                                # 过滤数据
                                if should_filter_data(prompt, response):
                                    continue
                                
                                file_keep_cnt += 1
                                
                                # 收集前10行样例
                                if len(file_sample_rows) < 10:
                                    file_sample_rows.append({'prompt': prompt, 'response': response})
                                
                                batch_data.append({'prompt': prompt, 'response': response})
                                
                                # 批量写入
                                if len(batch_data) >= group_cnt:
                                    df = pd.DataFrame(batch_data)
                                    write(save_file, df, append=exists(save_file), compression='GZIP')
                                    batch_data = []
                                    prog.update(task, advance=group_cnt)
                
                # 写入剩余数据
                if batch_data:
                    df = pd.DataFrame(batch_data)
                    write(save_file, df, append=exists(save_file), compression='GZIP')
            
            # 更新总计数
            all_cnt += file_all_cnt
            keep_cnt += file_keep_cnt
            
            # 输出当前文件处理完成后的统计和样例
            log.info('=' * 80, save_to_file=True)
            log.info(f'文件 {file_path} 处理完成', save_to_file=True)
            log.info(f'该文件: 处理 {file_all_cnt} 条，保留 {file_keep_cnt} 条，过滤率: {(1 - file_keep_cnt/file_all_cnt)*100:.2f}%', save_to_file=True)
            log.info(f'前{len(file_sample_rows)}行数据样例：', save_to_file=True)
            for idx, row in enumerate(file_sample_rows, 1):
                log.info(f'第{idx}行 - prompt: {row["prompt"]}', save_to_file=True)
                log.info(f'第{idx}行 - response: {row["response"]}', save_to_file=True)
                log.info('-' * 80, save_to_file=True)
            log.info('=' * 80, save_to_file=True)
            
        except Exception as e:
            log.error(f'处理文件 {file_path} 时出错: {str(e)}', save_to_file=True)
            import traceback
            log.error(traceback.format_exc(), save_to_file=True)
            continue
    
    log.info('=' * 80, save_to_file=True)
    log.info(f'全部处理完成！', save_to_file=True)
    log.info(f'总共处理 {all_cnt} 条数据，保留 {keep_cnt} 条数据', save_to_file=True)
    if all_cnt > 0:
        log.info(f'总体过滤率: {(1 - keep_cnt/all_cnt)*100:.2f}%', save_to_file=True)
    log.info(f'数据已保存到: {save_file}', save_to_file=True)
    log.info('=' * 80, save_to_file=True)


if __name__ == '__main__':

    processed_file_dir = PROJECT_ROOT + '/data/my_data'
    if not exists(processed_file_dir):
        mkdir(processed_file_dir)
    
    # 注释了，不重复处理
    # 1.
    process_web_text(keep_start=5, response_less_word=15)

    # 2.
    process_belle(response_less_word=15)

    # 3.
    process_chinese_medical_datasets(response_less_word=15)

    # 4. 金融问答数据集质量太差了
    # process_finace_dataset(prompt_less_word=10, response_less_word=15)

    # 5.
    process_zhihu_kol_dataset(prompt_less_word=4, response_less_word=10)

    # 6.
    process_belle_knowledge_enhanced_dataset(response_less_words=5)

    convert_wiki_to_simple_zh()

    # 7.
    process_zh_wiki_data_to_datset(groups_cnt=10000, max_len=512)

    #=================================================================

    # merge
    merge_dataset_as_single_file(groups_cnt=50000, min_len=3, max_len=512, cut_max_len=True)
        
    
    remove_dataset_duplicate_rows(groups_cnt=50000)

    # # shuffle
    shuffle_parquet_dataset(
        parquet_file=PROJECT_ROOT + '/data/my_dataset.parquet', 
        shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',  
        seed=23333
    )

    # split train validated and test
    split_train_valid_test_datasets(
            source_parquet_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
            max_len=320, 
            groups_cnt=50000
        )

    parquet_to_text()

    count_my_parquet_data(PROJECT_ROOT + '/data/my_dataset.parquet')

    dataset_length_cnt()

    process_belle_knowledge_enhanced_dataset_for_finetune(max_len=320, group_cnt=50000)

    count_my_parquet_data(PROJECT_ROOT + '/data/')

    parquet_to_json()
    count_my_json_data()


