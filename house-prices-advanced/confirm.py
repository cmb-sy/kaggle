from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# index_colは
def data_output(train, test, purpose_value):
#目的変数のヒストグラム
  purpose_value = sns.histplot(train[purpose_value]) 
  tmp = purpose_value.get_figure()
  tmp.savefig('purpose_calue_histgram.png',  orientation="landscape")


# plt.show()
# 売却価格の概要をみてみる
# print(df_train["SalePrice"].describe()
# 歪度は正規分布に比べてどれくらい偏りがあるか
# print(f"歪度: {round(df_train['SalePrice'].skew(),4)}" )
# 尖度は正規分布に比べての尖り具合
# print(f"尖度: {round(df_train['SalePrice'].kurt(),4)}" )

# 欠損値の割合を見る
# print(df_train.isnull().sum())
  # plt.savefig("histgram")
  # pd.set_option("display.max_columns", 100)
  # file = open('date_frame_train.txt', 'w')
  # file.write(str(df_train.head()))
  # file.close


# file.write("\n")
# file.write(ans)
# file.write("\n\n尖度や歪度\n\n")
# file.write(ans2)
# file.close()
if __name__ == '__main__':
  df_train = pd.read_csv("train.csv")
  # print(df_train['SalePrice'])
  df_test =pd.read_csv("test.csv")
  data_output(df_train, df_test, "SalePrice")

