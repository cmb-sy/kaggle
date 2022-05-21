from cgi import test
from random import random
from sklearn.ensemble import RandomForestClassifier as rfc
import pandas as pd
from prior_process import prior_process as pp
import sys
sys.path.append("../")
from library.change_submit_file.change_submit_file import change_submit_file

def random_forest(train_data, test_data):
  # n_estimator : 決定木数の設定。大きい方が精度が良いが（多くの多数決が可能）時間とメモリとのトレードオフ関係,defalut=100
  # random_state : 決定木へサンプリングするときのランダムseed値

  #modelを構築
  rc = rfc(random_state=100)
  
  target = train_data["Survived"].values
  # 「train」の以下の目的変数と説明変数の値を取得
  train_features = train_data[["Pclass", "Sex"]].values

  #modelを学習
  rc.fit(train_features, target)


  # 「test」の以下の説明変数の値を取得
  test_features = test_data[["Pclass", "Sex"]].values
  # 学習したモデルで予測
  predict = rc.predict(test_features)

  change_submit_file("PassengerId", "Survived", df_test, predict)
  
if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv('test.csv')
  pp_train, pp_test = pp(df_train, df_test)
  submit = pd.read_csv("submit.csv")
  print(submit.shape)
  random_forest(pp_train, pp_test)
