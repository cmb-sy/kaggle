""""
予測用データは全ての行を残す必要があるので除去作用はしない
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../")
from library.visualizer.visualizer import visualizer

def prior_process(train, test, chg_s_list):
  # Train, Testカラムの追加
  train['WhatIsData'] = 'Train'
  test['WhatIsData'] = 'Test'
  # test['SalePrice'] = 9999999999
#   alldata = pd.concat([train,test],axis=0).reset_index(drop=True)

# # 訓練データ特徴量をリスト化
#   cat_cols = alldata.dtypes[train.dtypes=='object'].index.tolist()
#   num_cols = alldata.dtypes[train.dtypes!='object'].index.tolist()

#   other_cols = ['Id','WhatIsData']
# # 余計な要素をリストから削除
#   cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去
#   num_cols.remove('Id') #Id削除

#   cat = pd.get_dummies(alldata[cat_cols])

# # データ統合
#   all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)

#   p_train = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)
#   p_test = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)
    
#   return p_train, p_test

if __name__ == '__main__':
  df_train = pd.read_csv("../train.csv")
  df_test =pd.read_csv("../test.csv")
  chg_s_list = ['MSSubClass','YrSold','MoSold']
  p_train, p_test = prior_process(df_train, df_test, chg_s_list)
  # visualizer("v_dir", p_train, p_test, "SalePrice", chg_log=True)




