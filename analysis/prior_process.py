import sys
from unicodedata import name
import pandas as pd
sys.path.append("../")

def prior_process(df_train, df_test):
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv('train.csv')
  
  #欠損値に値を入れる 
  # maleを0、femaleを1へ変換
  train = df_train.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
  test = df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

  #欠損値があったコラムを削除
  category_list = ['Name', 'Cabin', 'Ticket','Embarked']
  train = train.drop(category_list, axis=1, inplace=True)
  test = test.drop(category_list, axis=1, inplace=True)
  return train, test
    
if __name__ == '__main__':
  pass
