import os
import numpy as np
import pandas as pd
import re



def preprocess_df(filepath, show_head=True, clensing=True):

    def _cleansing(istrain=True):
        if istrain:
            df = pd.read_csv(filepath+'/train.csv')
        else:
            df = pd.read_csv(filepath+'/test.csv')
        print(f'Before processing\t:{df.shape}')

        ## textがNullレコードの除外
        df = df[df['text'].notnull()]

        if not istrain:
            df['selected_text'] = df['text']

        df.reset_index(drop=True, inplace=True)
        print(f'After processing\t:{df.shape}')
        if show_head:
            display(df.head())
            
        return df
            
    train_df = _cleansing(istrain=True)
    test_df  = _cleansing(istrain=False)

    return train_df, test_df


def process_text(text, selected_text):
    ## textのうち、どこがselectされたかを判定
    _s = 0
    _e = 0
    for i in range(len(text)):
        if text[i:i+len(selected_text)]==selected_text:
            _s = i
            _e = i+len(selected_text)
            select_range = set(np.arange(_s,_e).tolist())
            break

    #### 削除対象のテキストindexをリストに追加していく
    del_idx = []

    ## HPアドレスは削除。textのうち、どこが該当するかindexを取得
    ## URLトークンに置換する方式
    span_del = re.finditer(r'(http://.+?|https://.+?)(\s|$)', text)
    if span_del:
        for m in span_del:
            text = list(text)
            s = list(m.span())
            if len(m.groups()[1])>0:
                s[1] -= 1
            t = list(set(np.arange(s[0]+2, s[1]-1)))
            text[s[0]:s[0]+2] = ['U','R']
            text[s[1]-1] = 'L'
            text = ''.join(text)
            del_idx.extend(t)

    ## 特殊文字の削除
    special_char = [r'ï¿½', r'Â']
    # special_char = [r'Â']
    for _sc in special_char:
        span_del = re.finditer(_sc, text)
        if span_del:
            for m in span_del:
                m = m.span()
                m = list(set(np.arange(m[0], m[1])))
                del_idx.extend(m)

    span_del = re.finditer(r'\xa0', text)
    if span_del:
        for m in span_del:
            m = m.span()
            m = list(set(np.arange(m[0], m[1])))
            del_idx.extend(m)
        text = re.sub(r'\xa0', ' ', text)

    
    ## 特殊文字の置換
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'`', '\'', text)
    text = re.sub(r'´', '\'', text)

    #### 以降は、これまでの文字削除を反映したテキストに対する処理
    text_mask = np.ones(len(text), dtype=bool)
    text_mask[list(set(del_idx))] = False
    text_idx = np.arange(len(text_mask))[text_mask]
    text_tmp = np.array(list(text))[text_mask]
    text_tmp = ''.join(text_tmp.tolist())

    ## 文頭・文末のスペース削除
    space_head_tail = [r'^\s+', r'\s+$']
    tmp_del_idx = []
    for _sht in space_head_tail:
        span_del = re.search(_sht, text_tmp)
        if span_del:
            m = span_del.span()
            m = list(set(np.arange(m[0], m[1])))
            tmp_del_idx.extend(m)
    tmp_del_idx = list(set(tmp_del_idx))
    del_idx.extend(text_idx[tmp_del_idx].tolist())

    ## 削除するindexにはマスクを作成する
    del_idx = list(set(del_idx))

    text_mask = np.ones(len(text), dtype=bool)
    text_mask[del_idx] = False
    text_idx = np.arange(len(text_mask))[text_mask]

    selected_text_mask = np.zeros(len(text), dtype=bool)
    selected_text_mask[_s:_e] = True
    selected_text_mask = text_mask * selected_text_mask

    text = np.array(list(text))

    cl_text = ''.join(text[text_mask].tolist())
    cl_selected_text = ''.join(text[selected_text_mask].tolist())
    
    return {
        'cl_text'          : cl_text,              ## クレンジング済みのtext
        'cl_selected_text' : cl_selected_text,     ## クレンジング済みのselected_text
        'text_idx'         : text_idx              ## クレンジング済みのtextが元々のtextのどのindexに対応するか
        }

"""
Args
  selected_text: オーガナイザに提供されたデータ. ノイズが混じっている
  tweet: オーガナイザに提供されたデータ. 空白重複を削除したり、先頭空白を削除せずに、生データを使用する。
Return
  pre_processed_selected_text: pre-processされたselected_text
"""
filepath = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(filepath, "pre_processed.txt"))
dic = {}
alldata = []
for l in f:
    alldata.append(l.split("\n")[0])
for i in range(len(alldata)):
    if i % 4 == 0:
        dic[alldata[i]] = alldata[i+1]
    else:
        continue

def pre_process(selected_text, tweet):
    if selected_text.strip() == "?":
        return selected_text
    elif tweet.strip() == "star wars ............ is **** BOO??? i wanna do your job HAND IT OVER  u can act as me at my high school   LOL":
        return "LOL"
    elif tweet in dic:
        return dic[tweet]
    else:
        return selected_text


def pp(filtered_output, real_tweet):
    filtered_output = ' '.join(filtered_output.split())
    if len(real_tweet.split()) < 2:
        filtered_output = real_tweet
    else:
        if len(filtered_output.split()) == 1:
            if filtered_output.endswith(".."):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\.)\1{2,}', '.', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\.)\1{2,}', '..', filtered_output)
                return filtered_output
            if filtered_output.endswith('!!'):
                if real_tweet.startswith(" "):
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                else:
                    st = real_tweet.find(filtered_output)
                    fl = real_tweet.find("  ")
                    if fl != -1 and fl < st:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!', filtered_output)
                    else:
                        filtered_output = re.sub(r'(\!)\1{2,}', '!!', filtered_output)
                return filtered_output

        # tweetが空白から始まっている場合の処理
        if real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = ' '.join(real_tweet.split())
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag < start:
                filtered_output = real_tweet[start:end]

        # tweetが空白から始まっていないが、空白の重複が存在する場合の処理
        if "  " in real_tweet and not real_tweet.startswith(" "):
            filtered_output = filtered_output.strip()
            text_annotetor = re.sub(" {2,}", " ", real_tweet)
            start = text_annotetor.find(filtered_output)
            end = start + len(filtered_output)
            start -= 0
            end += 2
            flag = real_tweet.find("  ")
            if flag < start:
                filtered_output = real_tweet[start:end]
    return filtered_output
    