from unicodedata import name
import pandas as pd

def prior_process(train_data, test_data):
  #欠損値に値を入れる 
  # maleを0、femaleを1へ変換
  train_data.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
  test_data.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

  #欠損値があったコラムを削除
  category_list = ['Name', 'Cabin', 'Ticket','Embarked']
  train_data.drop(category_list, axis=1, inplace=True)
  test_data.drop(category_list, axis=1, inplace=True)
  return train_data, test_data

if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  df_test =pd.read_csv("test.csv")
  processed_train_data, processed_test_data = prior_process(df_train, df_test)
