import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
df_test =pd.read_csv('train.csv')

#コラムの表示
# print(df_train.columns)

#データフレームの出力
# print(df_train.head())

# 統計量の出力
# print(df_train.describe())

# 欠損値の確認
# print(df_train.isnull().sum())
print(df_test.isnull().sum())
