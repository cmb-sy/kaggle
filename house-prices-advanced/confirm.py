from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# index_colは
def visualizer(train, test, purpose_value):
#目的変数のヒストグラム
  purpose_value = sns.histplot(train[purpose_value]) 
  purpose_value.set_title("histgram of purpose value")
  tmp = purpose_value.get_figure()
  tmp.savefig('purpose_calue_histgram.png',  orientation="landscape")


# 目的変数の概要
  file = open('purpose_value.txt', 'w')
  file.write("概要\n")
  file.write(str(train["SalePrice"].describe()))
  file.write("\n\n尖度と歪度（どれくらい正規分布から離れているか）\n")
  file.write(f"歪度: {round(train['SalePrice'].skew(),4)}\n" )
  file.write(f"尖度: {round(train['SalePrice'].kurt(),4)}" )
  file.close()

# 欠損値の割合を見る
# print(df_train.isnull().sum())
  # plt.savefig("histgram")
  # pd.set_option("display.max_columns", 50)
  # file = open('frame_train.txt', 'w')
  # file.write("\n")
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
  visualizer(df_train, df_test, "SalePrice")

