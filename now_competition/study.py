from ctypes import sizeof
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import os
import numpy as np

stemmer = SnowballStemmer("english")

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
    for word in tqdm(nltk.regexp_tokenize(text, pattern)):
        tokens.append(word.lower())

    # 文字列中のすべての文字が英字で、かつ1文字以上ある場合のみの単語だけにする。
    for token in tokens:
        if token.isalpha(): 
            filtered_tokens.append(token)

    # 語幹の抽出
    # 例：torched ⇨　tourch,
    # https://yottagin.com/?p=3218
    stems = [stemmer.stem(t) for t in tqdm(filtered_tokens)]
    return stems

if __name__ =='__main__':
  df_train = pd.read_csv("data/train.csv")
  df_test =pd.read_csv('data/test.csv')
  
  # 1つの1つのデータを代入しているので、結果としてリストの要素1個1個が1つのデータに対応するようになる。
  processed_essays_train=[]
  for ess in list(df_train['full_text'][0:2]):
    processed_essays_train.append(tokenization_and_stemming(ess))

  # tag document
  # 単にタグ付けをしているだけ、前処理した文書データに他カラムの数値を対応させているだけ。
  processed_essays_train_tagged = [TaggedDocument(processed_essays_train[i], list(df_train.iloc[i,2:8])) for i in range(0,len(processed_essays_train))]
  # print(processed_essays_train_tagged[0])

  
  if not os.path.exists('model/doc2vec.model'):
  # doc2vec model
    doc2vecModel = Doc2Vec(vector_size=2000, window=2, dm=0, min_count=1, workers=8, epochs=40)
  # モデルのセーブ
    doc2vecModel.save("model/doc2vec.model")

  # モデルのロード(モデルが用意してあれば、ここからで良い)
  doc2vecModel = Doc2Vec.load('model/doc2vec.model')
  # 単語の登録
  doc2vecModel.build_vocab(processed_essays_train_tagged)

  # 学習
  doc2vecModel.train(processed_essays_train_tagged, total_examples=doc2vecModel.corpus_count, epochs=doc2vecModel.epochs)
  # 文章のベクトル化
  xx=[doc2vecModel.infer_vector(processed_essays_train[i]) for i in tqdm(range(0,len(processed_essays_train)))]

  X = np.array(xx)

# 配列として出力させる。
  cohesion_train = df_train.cohesion.to_numpy() 
  syntax_train = df_train.syntax.to_numpy() 
  vocabulary_train = df_train.vocabulary.to_numpy() 
  phraseology_train = df_train.phraseology.to_numpy() 
  grammar_train = df_train.grammar.to_numpy() 
  conventions_train = df_train.conventions.to_numpy() 




