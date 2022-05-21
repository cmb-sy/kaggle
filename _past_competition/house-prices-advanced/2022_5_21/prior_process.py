""""
予測用データは全ての行を残す必要があるので除去作用はしない
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../")
from library.visualizer.visualizer import visualizer

def prior_process(train, test, chg_s_list):
  #  # 意味のない数字を文字列へ変換
  # for column in chg_s_list:
  #   train[column] = train[column].astype(str)

  #  カテゴリカル変数の欠損には欠損を示す文字列’NA’を補完
  #  数値型変数の欠損には0.0を補完
  nan_col_list = train.isnull().sum()[train.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化
  nan_float_cols = (train[nan_col_list].dtypes=='float64').index.tolist() #float64
  nan_obj_cols = (train[nan_col_list].dtypes=='object').index.tolist() #object
  # float64型で欠損している場合は0を代入
  for nan_float_col in nan_float_cols:
  #locで欠損部分の位置を特定し、0.0を代入(行名, 列名)
    train.loc[train[nan_float_col].isnull(), nan_float_col] = 0.0
  # object型で欠損している場合は'NA'を代入
  for nan_obj_col in nan_obj_cols:
    train.loc[train[nan_obj_col].isnull(), nan_obj_col] = 'NA'
    
  return train, test

if __name__ == '__main__':
  df_train = pd.read_csv("../train.csv")
  df_test =pd.read_csv("../test.csv")
  chg_s_list = ['MSSubClass','YrSold','MoSold']
  p_train, p_test = prior_process(df_train, df_test, chg_s_list)
  visualizer("v_dir", p_train, p_test)




