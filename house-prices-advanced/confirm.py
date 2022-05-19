from operator import index
from pydoc import describe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualizer(train, test, purpose_value):
  new_dir_path = 'data/temp/new-dir'
  os.mkdir(new_dir_path)


#目的変数のヒストグラム
  purpose_value_hist = sns.histplot(train[purpose_value]) 
  purpose_value_hist.set_title("histgram of purpose value")
  tmp = purpose_value_hist.get_figure()
  tmp.savefig('_purpose_calue_histgram.png',  orientation="landscape")

# 目的変数の概要
  file = open('_purpose.txt', 'w')
  file.write("目的変数の概要\n")
  file.write(str(train[purpose_value].describe()))
  file.write("\n\n尖度と歪度（どれくらい正規分布から離れているか）\n")
  file.write(f"歪度: {round(train[purpose_value].skew(),4)}\n")
  file.write(f"尖度: {round(train[purpose_value].kurt(),4)}")
  file.close()

  # 説明変数
  file = open('_train_explain_value.txt', 'w')
  N, D = train.shape
  pd.set_option('display.max_rows', N)
  pd.set_option('display.max_columns', D)
  file.write("説明変数の欠損割合\n")
  file.write(str(train.isnull().sum()))
  file.close()

if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv("test.csv")
  visualizer(df_train, df_test, "SalePrice")

