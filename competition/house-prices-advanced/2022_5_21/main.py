from sklearn.linear_model import Lasso 
import pandas as pd
from prior_process import prior_process
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
import matplotlib.pyplot as plt



def main(train, test):
  # 学習データ内の分割
  train_x = p_train.drop('SalePrice',axis=1)
  train_y = np.log(p_train['SalePrice'])

  # テストデータ内の分割
  test_id = test['Id']
  test_data = test.drop('Id',axis=1)

  scaler = StandardScaler()  #スケーリング

  param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド
  cnt = 0

  for alpha in param_grid:

    ls = Lasso(alpha=alpha) #Lasso回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:
        best_score = test_rmse
        best_estimator = pipeline
        best_param = alpha
    elif best_score > test_rmse:
        best_score = test_rmse
        best_estimator = pipeline
        best_param = alpha
    else:
        pass
    cnt = cnt + 1
    
  print('alpha : ' + str(best_param))
  print('test score is : ' +str(best_score))
 
if __name__ == '__main__':
  df_train = pd.read_csv("../train.csv")
  df_test =pd.read_csv("../test.csv")
  chg_s_list = ['MSSubClass','YrSold','MoSold']
  p_train, p_test = prior_process(df_train, df_test, chg_s_list)
  main(p_train, p_test)
