import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(FILE_DIR)+"/src")

from config.roberta_config import RoBERTa_HEADS, MODEL_DIR
from data_process.data_process import process_text, pre_process


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from tokenizers import ByteLevelBPETokenizer


DEVICE = torch.device("cuda")
TOKENIZER = ByteLevelBPETokenizer(
    vocab_file       = f"{FILE_DIR}/../{MODEL_DIR}/vocab.json", 
    merges_file      = f"{FILE_DIR}/../{MODEL_DIR}/merges.txt", 
    lowercase        = True,
    add_prefix_space = True
    )
SENTIMENT_ID = {
    'positive': 1313,
    'negative': 2430,
    'neutral' : 7974
    }


class TweetDataset:
    def __init__(self, df, CONFIG):
        self.df               = df
        self.CONFIG           = CONFIG
        self.textID           = "textID"
        self.text             = "text"
        self.selected_text    = "selected_text"
        self.sentiment        = "sentiment"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        ## 処理前のレコードデータを取得
        textID           = self.df.loc[item, self.textID]
        text             = self.df.loc[item, self.text]
        selected_text    = self.df.loc[item, self.selected_text]
        sentiment        = self.df.loc[item, self.sentiment]
        weight           = 1

        ## データクレンジング
        if self.CONFIG['PRE_PROCESS']:
            pre_selected_text = pre_process(selected_text, text)
            pt = process_text(text, pre_selected_text)
        else:
            pt = process_text(text, selected_text)
        cl_text          = pt['cl_text']
        cl_selected_text = pt['cl_selected_text']
        text_idx         = pt['text_idx']


        ## cl_textのトークナイズ
        output         = TOKENIZER.encode(cl_text)
        tokenized_text = output.ids
        offsets_text   = output.offsets
        split_text     = output.tokens
        
        tokenized_text   = np.array(tokenized_text)
        attention_mask   = np.ones(len(tokenized_text))
        token_type_ids   = np.zeros(len(tokenized_text)) # RoBERTaの場合は全て0にする
        mask_out         = np.ones(len(tokenized_text))
        # mask_out[[0,-1]] = 0 # RoBERTaの場合はコメントアウト
        
        ## cl_selected_textが対応するトークンをtargetの0/1とする
        select_range = []
        _s, _e = 0, 0
        target_len = len(cl_selected_text)
        for i in range(len(cl_text)):
            if i+target_len>len(cl_text):
                break
            if cl_text[i:i+target_len]==cl_selected_text:
                _s, _e = i+1, i+target_len+1
                select_range.append(set(np.arange(_s,_e).tolist()))
        if _e==0:
            raise ValueError("Could not find the target!!!")

        # select_range = random.choice(select_range)
        select_range = select_range[0]

        target = np.zeros(len(tokenized_text), dtype=float)
        darty_target = 0
        target_first = 0
        for i in range(len(target)):
            _s, _e = offsets_text[i][0], offsets_text[i][1]
            off_range = set(np.arange(_s, _e).tolist())
            if len(off_range)==0:
                target[i] = 0.0 # CLS/SEPといったトークンは元々の文章に含まれないためlen==0となる
                
            elif len(select_range.intersection(off_range)): #>=len(off_range):
                target[i] = 1.0 # selected_textが対応するトークンについてはtarget=1.0とする
            
            else:
                target[i] = 0.0

        if target.sum()==0:
            print(f'cl_text: {cl_text} \
            \ncl_selected_text: {cl_selected_text}')

            print(darty_target)
            raise ValueError("stop")

        ## RoBERTaは先頭末尾に[0]/[2]のSpecial tokenを手動で追加する必要がある
        offsets_text   = [(0,0)] + offsets_text + [(0,0)]
        tokenized_text = np.append(np.append([0], tokenized_text), [2])
        attention_mask = np.append(np.append([1], attention_mask), [1])
        token_type_ids = np.append(np.append([0], token_type_ids), [0])
        mask_out       = np.append(np.append([0], mask_out), [0])
        target         = np.append(np.append([0], target), [0])

        ## トークンの先頭にセンチメントを追加する
        offsets_text   = [(0,0)]*RoBERTa_HEADS + offsets_text[1:]
        tokenized_text = np.append([0]+[SENTIMENT_ID[sentiment]]+[2]+[2], tokenized_text[1:])
        attention_mask = np.append([1]*RoBERTa_HEADS, attention_mask[1:])
        mask_out       = np.append([0]*RoBERTa_HEADS, mask_out[1:])
        target         = np.append([0]*RoBERTa_HEADS, target[1:])
        token_type_ids = np.append([0]*RoBERTa_HEADS, token_type_ids[1:])

        if self.CONFIG['QA_TASK']:
            target_id = np.arange(len(target))
            target_id = target_id[target==1]
            target    = np.array([target_id.min(), target_id.max()])
            
        return {
            'textID'            : textID,
            'text'              : text,
            'sentiment'         : sentiment,
            'cl_text'           : cl_text,
            'selected_text'     : selected_text,
            'cl_selected_text'  : cl_selected_text,
            'text_idx'          : text_idx,
            'offsets'           : offsets_text,
            'tokenized_text'    : tokenized_text.tolist(),
            'mask'              : attention_mask.tolist(),
            'mask_out'          : mask_out.tolist(),
            'target'            : target.tolist(),
            'token_type_ids'    : token_type_ids.tolist(),
            'weight'            : weight,
            'split_text'        : split_text,
        }

class TweetCollate:
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG

    def __call__(self, batch):
        out = {
                'textID'            : [],
                'text'              : [],
                'sentiment'         : [],
                'cl_text'           : [],
                'selected_text'     : [],
                'cl_selected_text'  : [],
                'text_idx'          : [],
                'offsets'           : [],
                'tokenized_text'    : [], # torch.longに型変換
                'mask'              : [], # torch.longに型変換
                'mask_out'          : [], # torch.longに型変換
                'target'            : [], # torch.float32に型変換/torch.longに型変換
                'token_type_ids'    : [], # torch.longに型変換
                'weight'            : [], # torch.float32に型変換
                'split_text'        : [],
            }

        ## batchを一度listにappendする
        for i in range(len(batch)):
            for k, v in batch[i].items():
                out[k].append(v)

        ## paddingサイズを決める
        if self.CONFIG['BUCKET']:
            max_pad = 0
            for p in out['tokenized_text']:
                if len(p)>max_pad:
                    max_pad = len(p)
        else:
            max_pad = self.CONFIG['MAX_LEN']
            
        ## padding処理を行う
        for i in range(len(batch)):
            tokenized_text = out['tokenized_text'][i]
            token_type_ids = out['token_type_ids'][i]
            offsets        = out['offsets'][i]
            target         = out['target'][i]
            mask           = out['mask'][i]
            mask_out       = out['mask_out'][i]
            text_len       = len(tokenized_text)

            out['tokenized_text'][i] = (tokenized_text + [1]    *(max_pad - text_len))[:max_pad] # RoBERTaはpaddingのトークン番号が1
            out['token_type_ids'][i] = (token_type_ids + [0]    *(max_pad - text_len))[:max_pad]
            out['offsets'][i]        = (offsets        + [(0,0)]*(max_pad - text_len))[:max_pad]
            out['mask'][i]           = (mask           + [0]    *(max_pad - text_len))[:max_pad]
            out['mask_out'][i]       = (mask_out       + [0]    *(max_pad - text_len))[:max_pad]

        ## 型変換
        # torch.long
        out['tokenized_text'] = torch.tensor(out['tokenized_text'], dtype=torch.long)
        out['mask']           = torch.tensor(out['mask'], dtype=torch.long)
        out['mask_out']       = torch.tensor(out['mask_out'], dtype=torch.long)
        out['token_type_ids'] = torch.tensor(out['token_type_ids'], dtype=torch.long)
        # torch.float32
        out['weight']         = torch.tensor(out['weight'], dtype=torch.float32)
        # torch.float32/torch.long
        out['target'] = torch.tensor(out['target'], dtype=torch.long)

        return out


class SentimentBalanceSampler(Sampler):

    def __init__(self, subset, CONFIG):
        self.indices    = subset.indices.copy()
        self.df         = subset.dataset.df.loc[self.indices, ['sentiment', 'text', 'selected_text']].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.length     = len(subset)
        self.batch_size = CONFIG["TRAIN_BATCH_SIZE"]

    def shuffle_idx(self):
        self.df = self.df.sample(frac=1)
        self.df = self.df.sort_values('sentiment')

        full_batch_length = self.length - self.length % self.batch_size
        last_batch_index  = self.df.index.tolist()[full_batch_length:]

        index = np.array(self.df.index.tolist()[:full_batch_length])
        index = index.reshape(self.batch_size, -1).T
        np.random.shuffle(index)
        index = index.reshape(-1).tolist()
        index = index + last_batch_index
        
        return index

    def __iter__(self):
        count = 0
        index = self.shuffle_idx()

        while count<self.length:
            yield index[count]
            count += 1

    def __len__(self):
        return len(self.dataset)
