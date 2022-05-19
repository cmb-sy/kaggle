""""
予測用データは全ての行を残す必要があるので除去作用はしない
"""

from email import message
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df_train = pd.read_csv("train.csv")
df_test =pd.read_csv("test.csv")
ans = str(df_train.isnull().sum())

