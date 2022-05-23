from sklearn.linear_model import Lasso 
import pandas as pd
from prior_process import prior_process
def main(train, test):
  


if __name__ == '__main__':
  df_train = pd.read_csv("../train.csv")
  df_test =pd.read_csv("../test.csv")
  chg_s_list = ['MSSubClass','YrSold','MoSold']
  p_train, p_test = prior_process(df_train, df_test, chg_s_list)
