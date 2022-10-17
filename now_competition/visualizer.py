import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil

def visualizer(directory_path, df_train, df_test):
  if os.path.exists(directory_path): # ディレクトリが存在するなら新規に作成
    shutil.rmtree(directory_path) # ディレクトリを中身ごと削除
    os.mkdir(directory_path) 
  
  file = open(directory_path + '/train_describe.txt', 'w')
  N, D = df_train.shape
  file.write("< 説明変数の概要 >\n")
  file.write(str(df_train.info()))
  file.close()


if __name__ =='__main__':
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv('test.csv')
  visualizer(directory_file:='visualfile', df_train, df_test)
