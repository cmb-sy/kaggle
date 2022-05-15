from random import random
import sklern.ensemble import RandomForestClassifier as rfc
import pandas as pd
from prior_process import prior_process as pp

def random_forest(train_data, test_data):
  # n_estimator : 決定木数の設定。大きい方が精度が良いが（多くの多数決が可能）時間とメモリとのトレードオフ関係,     defalut=100
  # random_state : 決定木へサンプリングするときのランダムseed値

  #modelを構築
  rc = rfc(random_state=100)
  #modelを学習
  rc.fit()
  


if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  de_test =pd.read_csv('train.csv')
  data = pp(df_train)
  


