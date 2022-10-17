from sklearn.ensemble import RandomForestClassifier as rfc
import pandas as pd
import sys
sys.path.append("../../../")
from library.change_submit_file.change_submit_file import change_submit_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import japanize_matplotlib


def random_forest(train_data, test_data, purpose_variable, explanatory_variable):
  #modelを構築
  rc = rfc(random_state=100)
  # 目的変数の値を取得
  target = train_data[purpose_variable].values
  # 説明変数の値を取得
  train_features = train_data[explanatory_variable].values
  #modelを学習
  rc.fit(train_features, target[:, 0])
  # 説明変数の値を取得
  test_features = test_data[explanatory_variable].values
  # 学習したモデルで分類
  predict = rc.predict(test_features)
  return predict

def visualizer(df_train, df_test):
  fig, axes = plt.subplots(3, 2, figsize=(16, 12))
  axes = axes.ravel() 
  column = ['Pclass', 'Sex', 'Embarked', 'Parch', 'SibSp']

  for col, ax in zip(column, axes):
    sns.countplot(x=col, data = df_train, hue = 'Survived', ax=ax)
  plt.subplots_adjust(wspace=0.4, hspace=0.6)  #グラフ間の余白を設定
  plt.show()

  column = ['Age', 'Fare']
  fig = plt.figure(figsize=(16, 12))
  gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
  axes = [fig.add_subplot(gs[0, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1]),]

# 全体
  sns.distplot(df_train["Age"], kde=True, color='black', ax=axes[0]) 
  sns.distplot(df_train["Fare"], kde=True, color='black', ax=axes[1]) 

# 生存か死亡か
  sns.distplot(df_train[df_train["Survived"]==1]["Age"], kde=True, label=0, ax=axes[2])
  sns.distplot(df_train[df_train["Survived"]==0]["Age"], kde=True, label=1, ax=axes[2])

  sns.distplot(df_train[df_train["Survived"]==1]["Fare"], kde=True, label=0, ax=axes[3])
  sns.distplot(df_train[df_train["Survived"]==0]["Fare"], kde=True, label=1, ax=axes[3])

  plt.legend()
  plt.subplots_adjust(wspace=0.2, hspace=0.2) 
  plt.show()

  # 相関係数の算出
  df_pearson = df_train.corr(method='pearson') 
  df_spearman = df_train.corr(method='spearman')
 
  # ヒートマップで可視化 
  sns.heatmap(df_pearson, annot=True) 
  plt.title('Correlation coefficient (pearson)',fontsize=18) 
  plt.ylim(df_pearson.shape[1],0)
  plt.show() 

  sns.heatmap(df_spearman, annot=True) 
  plt.title('Correlation coefficient (spearman)',fontsize=18) 
  plt.ylim(df_spearman.shape[1],0) 
  plt.show()



def prior_process(df_train, df_test):
    # データの加工
  # EmbarkedのSを代入 
  df_train["Embarked"] = df_train["Embarked"].fillna("S") 
  # Ageの平均値を代入 
  df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median()) 
  df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median()) 
  # Cabinを削除 
  df_train = df_train.drop('Cabin',axis='columns') 
  df_test = df_test.drop('Cabin',axis='columns') 
  # Fareの平均値を代入 
  df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())


  # カテゴリカル変数の変換
  df_train.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
  df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

  df_train.replace({'Embarked': {'S': 0, 'C': 1, 'Q' : 2}}, inplace=True)
  df_test.replace({'Embarked': {'S': 0, 'C': 1, 'Q' : 2}}, inplace=True)

# 特徴量追加
  df_train['FamilySize'] = df_train['Parch'] + df_train['SibSp'] + 1
  df_test['FamilySize'] = df_test['Parch'] + df_test['SibSp'] + 1

  return df_train, df_test

  
if __name__ == '__main__':
  df_train = pd.read_csv("../train.csv")
  df_test =pd.read_csv('../test.csv')

  # 加工
  prior_process(df_train, df_test)

  # 目的変数
  purpose_variable = ["Survived"]
  # 説明変数
  explanatory_variable = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize"]

  pridict = random_forest(df_train, df_test, purpose_variable, explanatory_variable)

  change_submit_file("PassengerId", "Survived", df_test, pridict)


