# https://www.kaggle.com/code/yuriao/english-language-learning-lgbm-doc2vec

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

def map_visualizer(df_train, df_test):
  purpose_col=list(df_train.columns)
  purpose_col.remove('text_id')
  purpose_col.remove('full_text')

  fig = plt.figure(figsize=(16, 12))
  gs = GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
  axes = [fig.add_subplot(gs[0, 0]),
          fig.add_subplot(gs[0, 1]),
          fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1]),
          fig.add_subplot(gs[2, 0]),
          fig.add_subplot(gs[2, 1]),]

  for i, col in enumerate(purpose_col):
    axes[i].hist(df_train[col])
    axes[i].set_title(col)    
  
  plt.subplots_adjust(wspace=0.2, hspace=0.5)
  plt.show()
    



if __name__ =='__main__':
  df_train = pd.read_csv("data/train.csv")
  df_test =pd.read_csv('data/test.csv')
  map_visualizer( df_train, df_test)
