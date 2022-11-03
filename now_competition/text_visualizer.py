import pandas as pd
import os, shutil, io

def text_visualizer(directory_path, df_train, df_test):
  if os.path.exists(directory_path): # ディレクトリが存在するなら、あるものを削除して新規に作成
    shutil.rmtree(directory_path) # ディレクトリを中身ごと削除
    os.mkdir(directory_path) 
  else: #ディレクトリがないなら新規に作成
    os.mkdir(directory_path) 
  
  file = open(directory_path + '/train_describe.txt', 'w')
  file.write("< 説明変数の概要 >\n")
  buffer = io.StringIO()
  df_train.info(buf=buffer)
  df_info = buffer.getvalue().split("\n")
  for i in df_info: 
    file.write(i + "\n")
  file.write(str(df_train.shape))
  
  file.write("\n\n")
  file.write("< 説明変数の基本統計量 >\n")
  file.write("シェイプ" + str(df_train.describe()))

  file.write("\n\n")
  file.write("< 欠損値の確認 >\n")
  train_num = df_train.isnull().sum()[df_train.isnull().sum()>0]
  test_num = df_test.isnull().sum()[df_test.isnull().sum()>0]
  file.write("train : " + str(train_num))
  file.write('\n')
  file.write("test : " + str(test_num))
  file.write("\n\n")


  file.write("< 説明変数の始めの5つのデータ >\n")
  file.write("train : " + str(df_train.head()))
  
  file.close()


if __name__ =='__main__':
  df_train = pd.read_csv("data/train.csv")
  df_test =pd.read_csv('data/test.csv')
  text_visualizer(directory_file:='visualfile', df_train, df_test)
