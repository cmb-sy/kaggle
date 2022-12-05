from ctypes import sizeof
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import os
import numpy as np
import optuna
import sys
# sys.path.append("../LightGBM")
import lightgbm
# from lightgbm.callback import log_evaluation, early_stopping

# stemmer = SnowballStemmer("english")


# def tokenization_and_stemming(text):
#     tokens = []
#     filtered_tokens = []

#     pattern = r'''(?x)          
#         (?:[A-Z]\.)+        
#       | \w+(?:-\w+)*       
#       | \$?\d+(?:\.\d+)?%?  
#       | \.\.\.              
#       | [][.,;"'?():_`-]   
#     '''

#     # NLTK+正規表現を使ったトークン化
#     for word in tqdm(nltk.regexp_tokenize(text, pattern)):
#       tokens.append(word.lower())

#     # 文字列中のすべての文字が英字で、かつ1文字以上ある場合のみの単語だけにする。
#     for token in tokens:
#         if token.isalpha(): 
#             filtered_tokens.append(token)

#     # 語幹の抽出
#     # 例：torched ⇨　tourch,
#     # https://yottagin.com/?p=3218
#     stems = [stemmer.stem(t) for t in tqdm(filtered_tokens)]
#     return stems

# if __name__ =='__main__':
#   df_train = pd.read_csv("data/train.csv")
#   df_test =pd.read_csv('data/test.csv')
  
#   # 1つの1つのデータを代入しているので、結果としてリストの要素1個1個が1つのデータに対応するようになる。
#   # essにはリストが返ってくる。そのリストをデータ分だけリストにいれる。
#   processed_essays_train=[]
#   for ess in list(df_train['full_text']):
#     processed_essays_train.append(tokenization_and_stemming(ess))

#   # tag document
#   # 単にタグ付けをしているだけ、前処理した文書データに他カラムの数値を対応させているだけ。
#   processed_essays_train_tagged = [TaggedDocument(processed_essays_train[i], list(df_train.iloc[i,2:8])) for i in range(0,len(processed_essays_train))]
  
#   if not os.path.exists('model/doc2vec.model'):
#   # Doc2Vecmodelの作成
#     doc2vecModel = Doc2Vec(vector_size=2000, window=2, dm=0, min_count=1, workers=8, epochs=40)
#   # モデルのセーブ
#     doc2vecModel.save("model/doc2vec.model")

#   # モデルのロード(モデルが用意してあれば、ここからで良い)
#   doc2vecModel = Doc2Vec.load('model/doc2vec.model')

#   # 単語のリストを作成し、単語に番号を振る。（絶対必要）
#   doc2vecModel.build_vocab(processed_essays_train_tagged)

#   # 学習
#   # total_exsampleはドキュメントの数
#   doc2vecModel.train(processed_essays_train_tagged, total_examples=doc2vecModel.corpus_count, epochs=doc2vecModel.epochs)
#   # 文章のベクトル化
#   xx=[doc2vecModel.infer_vector(processed_essays_train[i]) for i in tqdm(range(0,len(processed_essays_train)))]

#   X = np.array(xx)
#   print(X.shape)

# # # 配列として出力させる。
#   cohesion_train = df_train.cohesion.to_numpy() 
#   syntax_train = df_train.syntax.to_numpy() 
#   vocabulary_train = df_train.vocabulary.to_numpy() 
#   phraseology_train = df_train.phraseology.to_numpy() 
#   grammar_train = df_train.grammar.to_numpy() 
#   conventions_train = df_train.conventions.to_numpy() 

# def objective(trial, data=X, target=optuna_train_values):
#     print(optuna_train_values)
#     train_x, test_x, train_y, test_y = model_selection.train_test_split(data, target, test_size=0.3, random_state=42)
#     param = {
#         'metric': 'rmse', 
#         'random_state': 42,
#         'n_estimators': trial.suggest_int('n_estimators', 10, 500),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.001, 0.01, 0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.2, 0.4,0.5,0.6,0.7,0.8,1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.004, 0.008, 0.01, 0.02, 0.05, .1, 0.2, 0.5]),
#         'max_depth': trial.suggest_categorical('max_depth', [10, 20,100, 150]),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
#     }
#     model = LGBMRegressor(**param)  
    
#     model.fit(train_x, train_y, eval_set=[(test_x, test_y)], callbacks=[log_evaluation(period=0)])
    
#     preds = model.predict(test_x)
    
#     rmse = np.sqrt(metrics.mean_squared_error(test_y, preds))
  
#     return rmse


