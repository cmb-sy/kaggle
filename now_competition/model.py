from ctypes import sizeof
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
# d2v import
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def doc2Vec_model(text_data):
    # doc2vec model
  doc2vecModel = Doc2Vec(vector_size=2000, window=2, dm=0, min_count=1, workers=8, epochs=40)
  # モデルのセーブ
  doc2vecModel.save("doc2vec.model")
  # モデルのロード(モデルが用意してあれば、ここからで良い)
  doc2vecModel = Doc2Vec.load('doc2vec2.model')
  # 単語の登録
  doc2vecModel.build_vocab(processed_essays_train_tagged)

  # 学習
  doc2vecModel.train(processed_essays_train_tagged, total_examples=doc2vecModel.corpus_count, epochs=doc2vecModel.epochs)
  # 文章のベクトル化
  xx=[doc2vecModel.infer_vector(processed_essays_train[i]) for i in range(0,len(processed_essays_train))]

# xxはベクトルである必要があるので、tsneを使用してその分離可能性を確認する。
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  X_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(xx)

  plt.scatter(X_embedded[:,0], X_embedded[:,1])
  plt.show()

if __name__ =='__main__':
  df_train = pd.read_csv("data/train.csv")
  df_test =pd.read_csv('data/test.csv')
  
  # doc2vec model
  doc2vecModel = Doc2Vec(vector_size=2000, window=2, dm=0, min_count=1, workers=8, epochs=40)
  # モデルのセーブ
  doc2vecModel.save("doc2vec.model")
  # モデルのロード(モデルが用意してあれば、ここからで良い)
  doc2vecModel = Doc2Vec.load('doc2vec2.model')
  # 単語の登録
  doc2vecModel.build_vocab(processed_essays_train_tagged)

  # 学習
  doc2vecModel.train(processed_essays_train_tagged, total_examples=doc2vecModel.corpus_count, epochs=doc2vecModel.epochs)
  # 文章のベクトル化
  xx=[doc2vecModel.infer_vector(processed_essays_train[i]) for i in range(0,len(processed_essays_train))]

# xxはベクトルである必要があるので、tsneを使用してその分離可能性を確認する。
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  X_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(xx)

  plt.scatter(X_embedded[:,0], X_embedded[:,1])
  plt.show()
