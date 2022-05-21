from operator import index
from pydoc import describe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

def visualizer(directory_path, train, test, purpose_value=None, chg_log=False):
  directory_path = str(directory_path)
  if os.path.exists(directory_path):
    shutil.rmtree(directory_path)
    os.mkdir(directory_path)

  if purpose_value != None:
    #目的変数のヒストグラ
    if chg_log:
      purpose_value_hist = sns.histplot(np.log(train[purpose_value]))
    else:
      purpose_value_hist = sns.histplot(train[purpose_value]) 

    purpose_value_hist.set_title("histgram of purpose value")
    tmp = purpose_value_hist.get_figure()
    tmp.savefig('v_dir/_purpose_calue_histgram.png',  orientation="landscape")

  # 目的変数の概要
    file = open(directory_path + '/purpose.txt', 'w')
    file.write("< 目的変数の概要 >\n")
    file.write(str(train[purpose_value].describe()))
    file.write("\n\n尖度と歪度（どれくらい正規分布から離れているか）\n")
    if chg_log:
      file.write(f"歪度: {round(np.log(train[purpose_value]).skew(),4)}\n")
      file.write(f"尖度: {round(np.log(train[purpose_value]).kurt(),4)}")
    else:
      file.write(f"歪度: {round(train[purpose_value].skew(),4)}\n")
      file.write(f"尖度: {round(train[purpose_value].kurt(),4)}")
    file.close()

  # 説明変数
  file = open(directory_path + '/_train_explain_value.txt', 'w')
  N, D = train.shape
  pd.set_option('display.max_rows', N)
  pd.set_option('display.max_columns', D)
  file.write("< 説明変数の欠損割合 >\n")
  file.write(str(train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)))

  # 欠損を含むカラムのデータ型を確認
  col_list = train.isnull().sum()[train.isnull().sum()>0].index.tolist() 
  file.write("\n\n< 説明変数の欠損データのデータ型 >\n")
  file.write(str(train[col_list].dtypes.sort_values() ))
  file.close()


  file = open(directory_path + '/train_describe.txt', 'w')
  N, D = train.shape
  file.write("< 説明変数の概要 >\n")
  pd.set_option('display.max_columns', D)
  file.write(str(train.head()))
  file.close()


if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv("test.csv")
  visualizer(df_train, df_test, "SalePrice")

