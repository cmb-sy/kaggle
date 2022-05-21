""""
予測用データは全ての行を残す必要があるので除去作用はしない
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from library.visualizer.visualizer import visualizer
from IPython.display import display

def prior_process(train, test, chg_s_list):
   # 意味のない数字を文字列へ変換
  for column in chg_s_list:
    train[column] = train[column].astype(str)
  return train, test

if __name__ == '__main__':
  df_train = pd.read_csv("csv_file/datasets/train.csv")
  df_test =pd.read_csv("csv_file/datasets/test.csv")
  chg_s_list = ['MSSubClass','YrSold','MoSold']
  p_train, p_test = prior_process(df_train, df_test, chg_s_list)
  visualizer("v_dir", p_train, p_test)




