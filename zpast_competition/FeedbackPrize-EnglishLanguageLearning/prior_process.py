import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

stemmer = SnowballStemmer("english")

def preprocessing(df_train, df_test):
  # 1つの1つのデータを代入しているので、結果としてリストの要素1個1個が1つのデータに対応するようになる。
  processed_essays_train=[]
  for ess in list(df_train['full_text']):
    processed_essays_train.append(tokenization_and_stemming(ess))

  # tag document
  # 単にタグ付けをしているだけ、前処理した文書データに他カラムの数値を対応させているだけ。
  processed_essays_train_tagged = [TaggedDocument(processed_essays_train[i], list(df_train.iloc[i,2:8])) for i in range(0,len(processed_essays_train))]
  # print(processed_essays_train_tagged[0])
  return processed_essays_train_tagged
 

def tokenization_and_stemming(text):
    tokens = []
    filtered_tokens = []

    pattern = r'''(?x)          
        (?:[A-Z]\.)+        
      | \w+(?:-\w+)*       
      | \$?\d+(?:\.\d+)?%?  
      | \.\.\.              
      | [][.,;"'?():_`-]   
    '''

    # NLTK+正規表現を使ったトークン化
    for word in nltk.regexp_tokenize(text, pattern):
        tokens.append(word.lower())

    # 文字列中のすべての文字が英字で、かつ1文字以上ある場合のみの単語だけにする。
    for token in tokens:
        if token.isalpha(): 
            filtered_tokens.append(token)

    # 語幹の抽出
    # 例：torched ⇨　tourch,
    # https://yottagin.com/?p=3218
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

if __name__ =='__main__':
  df_train = pd.read_csv("data/train.csv")
  df_test =pd.read_csv('data/test.csv')
  preprocessing(df_train, df_test)
