import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from config.roberta_config import INPUT_DIR, OUTPUT_DIR, ORI_DATA_DIR
from utils.helper import to_pickle, read_pickle, Logger, seed_everything
from data_process.data_process import preprocess_df, pp
from train_functions.loss_functions import define_criterion
from train_functions.roberta_models import build_model
from datasets.datasets import TweetDataset, TweetCollate, SentimentBalanceSampler


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import KFold, StratifiedKFold
import time
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
import copy
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torchcontrib.optim import SWA

import mlflow

DEVICE = torch.device("cuda")



def main():
    # preprocess
    train_df, _ = preprocess_df(os.path.join(FILE_DIR, ORI_DATA_DIR), show_head=False)

    # import config
    filepath = os.path.join(FILE_DIR, "./config/config.yml")
    with open(filepath) as file:
        config = yaml.safe_load(file)

    # train
    with mlflow.start_run():
        train(train_df, config)


def train(train_df, CONFIG):
    # set-up
    seed_everything(CONFIG['SEED'])
    torch.manual_seed(CONFIG['TORCH_SEED'])
    mlflow.log_params(CONFIG)

    TRAIN_LEN     = len(train_df)
    train_dataset = TweetDataset(train_df, CONFIG)
    CRITERION     = define_criterion(CONFIG)
    

    folds = StratifiedKFold(n_splits=CONFIG["FOLD"], shuffle=True, random_state=CONFIG["SEED"])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_dataset.df['textID'], train_dataset.df['sentiment'])):
        if n_fold!=CONFIG["FOLD_NUM"]:
            continue

        ## DataLoaderの定義
        train = torch.utils.data.Subset(train_dataset, train_idx)
        valid = torch.utils.data.Subset(train_dataset, valid_idx)

        DATA_IN_EPOCH = len(train)
        TOTAL_DATA    = DATA_IN_EPOCH * CONFIG["EPOCHS"]
        T_TOTAL = int(CONFIG["EPOCHS"] * DATA_IN_EPOCH / CONFIG["TRAIN_BATCH_SIZE"])

        ## modelとoptimizerの初期化
        model = build_model(CONFIG)
        model.to(DEVICE)
        model.train()

        ## From 20/05/17
        param_optimizer = list(model.named_parameters())
        bert_params  = [n for n, p in param_optimizer if "bert" in n]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            ## BERT param
            {
                'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (n in bert_params)],
                'weight_decay': CONFIG["WEIGHT_DECAY"],
                'lr':CONFIG['LR'] * 1,
                },
            {
                'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay)) and (n in bert_params)], 
                'weight_decay': 0.0, 
                'lr':CONFIG['LR'] * 1,
                },
            ## Other param
            {
                'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (n not in bert_params)],
                'weight_decay': CONFIG["WEIGHT_DECAY"],
                'lr':CONFIG['LR'] * 1,
                },
            {
                'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay)) and (n not in bert_params)], 
                'weight_decay': 0.0,
                'lr':CONFIG['LR'] * 1,
                },
            ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=4e-5)
        if CONFIG['SWA']:
            optimizer = SWA(optimizer)
            
            
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps   = int(CONFIG["WARMUP"]*T_TOTAL),
            num_training_steps = T_TOTAL
            )

        train_sampler = SentimentBalanceSampler(train, CONFIG)

        train_loader = DataLoader(
            train,
            batch_size  = CONFIG["TRAIN_BATCH_SIZE"],
            shuffle     = False,
            sampler     = train_sampler,
            collate_fn  = TweetCollate(CONFIG),
            num_workers = 1
        )

        valid_loader = DataLoader(
            valid,
            batch_size  = CONFIG["VALID_BATCH_SIZE"],
            shuffle     = False,
            # sampler     = valid_sampler,
            collate_fn  = TweetCollate(CONFIG),
            num_workers = 1
        )

        n_data   = 0
        n_e_data = 0
        best_val = 0.0
        best_val_neu = 0.0
        best_val_pos = 0.0
        best_val_neg = 0.0

        t_batch   = 0
        while n_data<TOTAL_DATA:
            print(f"Epoch : {int(n_data/DATA_IN_EPOCH)}")

            n_batch           = 0
            loss_list         = []
            jac_token_list    = []
            jac_text_list     = []
            jac_sentiment_list= []
            jac_cl_text_list  = []

            output_list       = []
            target_list       = []

            for batch in tqdm(train_loader):
                textID           = batch['textID']
                text             = batch['text']
                sentiment        = batch['sentiment']
                cl_text          = batch['cl_text']
                selected_text    = batch['selected_text']
                cl_selected_text = batch['cl_selected_text']
                text_idx         = batch['text_idx']
                offsets          = batch['offsets']

                tokenized_text   = batch['tokenized_text'].to(DEVICE)
                mask             = batch['mask'].to(DEVICE)
                mask_out         = batch['mask_out'].to(DEVICE)
                token_type_ids   = batch['token_type_ids'].to(DEVICE)
                weight           = batch['weight'].to(DEVICE)
                target           = batch['target'].to(DEVICE)

                
                ep        = int(n_data/DATA_IN_EPOCH)
                n_data   += len(textID)
                n_e_data += len(textID)
                n_batch  += 1
                t_batch  += 1
                
                
                model.zero_grad()
                # optimizer.zero_grad()
                output = model(
                    input_ids      = tokenized_text, 
                    attention_mask = mask, 
                    token_type_ids = token_type_ids,
                    mask_out       = mask_out
                    )

                loss = CRITERION(output, target)

                loss = loss * weight
                loss.mean().backward()
                loss = loss.detach().cpu().numpy().tolist()

                optimizer.step()
                
                if t_batch<T_TOTAL*0.50:
                    scheduler.step()
                    
                    
                loss_list.extend(loss)

                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()

                jac = calc_jaccard(output, batch, CONFIG)

                jac_token_list.extend(jac['jaccard_token'].tolist())
                jac_cl_text_list.extend(jac['jaccard_cl_text'].tolist())
                jac_text_list.extend(jac['jaccard_text'].tolist())
                jac_sentiment_list.extend(sentiment)

                if ((
                    ((ep>0)&(n_batch%(int(5*32/CONFIG["TRAIN_BATCH_SIZE"]))==0)) |
                    ((n_batch%(int(50*32/CONFIG["TRAIN_BATCH_SIZE"]))==0)) |
                    (n_data>=TOTAL_DATA)
                    ) and
                    (CONFIG['SWA'] and ep>0 and t_batch>=T_TOTAL*0.50)):
                    optimizer.update_swa()

                if (
                    ((ep>0)&(n_batch%(int(50*32/CONFIG["TRAIN_BATCH_SIZE"]))==0)) |
                    ((n_batch%(int(50*32/CONFIG["TRAIN_BATCH_SIZE"]))==0)) |
                    (n_data>=TOTAL_DATA)
                    ): # ((n_data>=0)&(n_data<=1600)|(n_data>=21000)&(n_data<=23000))&
                    
                    if CONFIG['SWA'] and ep>0 and t_batch>=T_TOTAL*0.50:
                        # optimizer.update_swa()
                        optimizer.swap_swa_sgd()

                    val = create_valid(model, valid_loader, CONFIG)

                    trn_loss         = np.array(loss_list).mean()
                    trn_jac_token    = np.array(jac_token_list).mean()
                    trn_jac_cl_text  = np.array(jac_cl_text_list).mean()
                    
                    trn_jac_text     = np.array(jac_text_list).mean()
                    trn_jac_text_neu = np.array(jac_text_list)[np.array(jac_sentiment_list)=='neutral'].mean()
                    trn_jac_text_pos = np.array(jac_text_list)[np.array(jac_sentiment_list)=='positive'].mean()
                    trn_jac_text_neg = np.array(jac_text_list)[np.array(jac_sentiment_list)=='negative'].mean()

                    val_loss         = val['loss'].mean()
                    val_jac_token    = val['jaccard_token'].mean()
                    val_jac_cl_text  = val['jaccard_cl_text'].mean()
                    
                    val_jac_text     = val['jaccard_text'].mean()
                    val_jac_text_neu = val['jaccard_text'][val['sentiment']=='neutral'].mean()
                    val_jac_text_pos = val['jaccard_text'][val['sentiment']=='positive'].mean()
                    val_jac_text_neg = val['jaccard_text'][val['sentiment']=='negative'].mean()

                    loss_list         = []
                    jac_token_list    = []
                    jac_cl_text_list  = []
                    jac_text_list     = []
                    jac_sentiment_list= []


                    # mlflow
                    metrics = {
                        "lr": optimizer.param_groups[0]['lr'],
                        "trn_loss": trn_loss,
                        "trn_jac_text_neu": trn_jac_text_neu,
                        "trn_jac_text_pos": trn_jac_text_pos,
                        "trn_jac_text_neg": trn_jac_text_neg,
                        "trn_jac_text": trn_jac_text,
                        "val_loss": val_loss,
                        "val_jac_text_neu": val_jac_text_neu,
                        "val_jac_text_pos": val_jac_text_pos,
                        "val_jac_text_neg": val_jac_text_neg,
                        "val_jac_text": val_jac_text,
                    }
                    mlflow.log_metrics(metrics, step=n_data)

                    if CONFIG['SWA'] and t_batch<T_TOTAL*0.50:
                        pass
                    else:
                        if best_val<val_jac_text:
                            best_val      = val_jac_text
                            best_model    = copy.deepcopy(model)

                    if CONFIG['SWA'] and ep>0 and t_batch>=T_TOTAL*0.50:
                        optimizer.swap_swa_sgd()

                if n_e_data>=DATA_IN_EPOCH:
                    n_e_data -= DATA_IN_EPOCH

                if n_data>=TOTAL_DATA:
                    filepath = os.path.join(FILE_DIR, OUTPUT_DIR, "model.pth")
                    torch.save(best_model.state_dict(), filepath)

                    # mlflow
                    mlflow.log_artifact(filepath)
                    break

def word_base_jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jaccard_apply(x): 
    return word_base_jaccard(x.iloc[0], x.iloc[1])
    

def calc_jaccard(pred, batch, CONFIG):
    """
    トークンレベルでのjaccardと、textレベルでのjaccardの計算を行う
    """
    #### トークンレベルでのjaccard計算
    bi_pred = np.zeros(pred.shape[:2])
    bi_targ = np.zeros(pred.shape[:2])
    ## 確率重みづけ方式
    tmp = 0
    for i in range(len(bi_pred)):
        ## 先にstartを決めてからendを探す方式
        _s = pred[i,:,0].argmax()
        _e = pred[i,_s:,1].argmax()+_s

        bi_pred[i, _s:_e+1] = 1
        bi_targ[i, batch['target'][i,0]:batch['target'][i,1]+1] = 1

    jaccard_token = (bi_pred*bi_targ==1.0).sum(axis=1) / (bi_pred+bi_targ>=1.0).sum(axis=1)


    #### textレベルでのjaccard計算
    ## 1と予測されたトークンのoffsetを取り出す
    jaccard_text          = []
    jaccard_cl_text       = []
    pred_selected_text    = []
    pred_cl_selected_text = []
    for i in range(bi_pred.shape[0]):
        text             = batch['text'][i]
        cl_text          = batch['cl_text'][i]
        selected_text    = batch['selected_text'][i]
        cl_selected_text = batch['cl_selected_text'][i]
        text_idx         = batch['text_idx'][i]
        offsets          = batch['offsets'][i]
        split_text       = batch['split_text'][i]

        ## 1と予測されたoffsetsをlistにする
        tmp = (np.array(offsets)[bi_pred[i]==1]).tolist()

        s_cl_text, e_cl_text = tmp[0][0]-1, tmp[-1][1]-1
        if s_cl_text<0:
            s_cl_text = 0
            
        s_text,    e_text    = np.array(text_idx)[s_cl_text], np.array(text_idx)[e_cl_text-1]+1
        

        if CONFIG['PRE_PROCESS']:
            pp_selected_text = pp(text[s_text:e_text], text)
            pp_cl_selected_text = pp(cl_text[s_cl_text:e_cl_text], cl_text)

            jaccard_text.append(word_base_jaccard(selected_text, pp_selected_text))
            jaccard_cl_text.append(word_base_jaccard(cl_selected_text, pp_cl_selected_text))
            pred_selected_text.append(text[s_text:e_text])
            pred_cl_selected_text.append(cl_text[s_cl_text:e_cl_text])

        else:
            jaccard_text.append(word_base_jaccard(selected_text, text[s_text:e_text]))
            jaccard_cl_text.append(word_base_jaccard(cl_selected_text, cl_text[s_cl_text:e_cl_text]))
            pred_selected_text.append(text[s_text:e_text])
            pred_cl_selected_text.append(cl_text[s_cl_text:e_cl_text])

    jaccard_text    = np.array(jaccard_text)
    jaccard_cl_text = np.array(jaccard_cl_text)

    return {
        'jaccard_token'         : jaccard_token,
        'jaccard_cl_text'       : jaccard_cl_text,
        'jaccard_text'          : jaccard_text,
        'pred'                  : bi_pred,
        'pred_selected_text'    : pred_selected_text,
        'pred_cl_selected_text' : pred_cl_selected_text,
        }


def create_valid(model, valid_loader, CONFIG):
    model.eval()
    valid_ids                   = []
    valid_text                  = []
    valid_sentiment             = []
    valid_pred_selected_text    = []
    valid_pred_cl_selected_text = []
    output_list                 = []
    target_list                 = []
    CRITERION = define_criterion(CONFIG)

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            textID         = batch['textID']
            text           = batch['text']
            sentiment      = batch['sentiment']
            tokenized_text = batch['tokenized_text'].to(DEVICE)
            mask           = batch['mask'].to(DEVICE)
            mask_out       = batch['mask_out'].to(DEVICE)
            target         = batch['target'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            target         = batch['target'].to(DEVICE)
            
            output = model(
                input_ids      = tokenized_text, 
                attention_mask = mask, 
                token_type_ids = token_type_ids,
                mask_out       = mask_out
                )

            loss = CRITERION(output, target).detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            loss = loss.tolist()
            
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            
            jac = calc_jaccard(output, batch, CONFIG)

            # if CONFIG["QA_TASK"]:
            #     output = jac['pred']

            if CONFIG["BUCKET"]:
                pad = np.zeros((output.shape[0], CONFIG["MAX_LEN"]-output.shape[1], 2))
                pad[:,:,:] = -1e10
                output = np.append(output, pad, axis=1)
            

            valid_ids.extend(textID)
            valid_text.extend(text)
            valid_sentiment.extend(sentiment)
            valid_pred_selected_text.extend(jac['pred_selected_text'])
            valid_pred_cl_selected_text.extend(jac['pred_cl_selected_text'])

            if i==0:
                valid_preds           = output
                valid_loss            = loss
                valid_jaccard_token   = jac['jaccard_token']
                valid_jaccard_cl_text = jac['jaccard_cl_text']
                valid_jaccard_text    = jac['jaccard_text']
            else:
                valid_preds           = np.append(valid_preds, output, axis=0)
                valid_loss            = np.append(valid_loss, loss, axis=0)
                valid_jaccard_token   = np.append(valid_jaccard_token, jac['jaccard_token'], axis=0)
                valid_jaccard_cl_text = np.append(valid_jaccard_cl_text, jac['jaccard_cl_text'], axis=0)
                valid_jaccard_text    = np.append(valid_jaccard_text, jac['jaccard_text'], axis=0)
            
    valid_ids                   = np.array(valid_ids)
    valid_text                  = np.array(valid_text)
    valid_sentiment             = np.array(valid_sentiment)
    valid_pred_selected_text    = np.array(valid_pred_selected_text)
    valid_pred_cl_selected_text = np.array(valid_pred_cl_selected_text)

    return {
        'textID'                : valid_ids,
        'text'                  : valid_text,
        'sentiment'             : valid_sentiment,
        'pred_ori'              : valid_preds,
        'pred_selected_text'    : valid_pred_selected_text,
        'pred_cl_selected_text' : valid_pred_cl_selected_text,
        'loss'                  : valid_loss, 
        'jaccard_token'         : valid_jaccard_token,
        'jaccard_cl_text'       : valid_jaccard_cl_text,
        'jaccard_text'          : valid_jaccard_text,
        }


if __name__ == '__main__':
    start_time = time.time()
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print(f"\nsuccess! [{(time.time()-start_time)/60:.1f} min]")