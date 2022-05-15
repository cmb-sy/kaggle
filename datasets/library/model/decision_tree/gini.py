def gini_score(data, target, feat_idx, threshold):
    gini = 0
    sample_num = len(target)
   
    # targetの中のdata
    div_target = [target[data[:, feat_idx] >= threshold], target[data[:, feat_idx] < threshold]]
   
    for group in div_target:
        score = 0
        classes = np.unique(group)
        for cls in classes:
            p = np.sum(group == cls)/len(group)
            score += p * p
        gini += (1- score) * (len(group)/sample_num)
    return gini



if __name__ == '__main__':
  
  import numpy as np
  import matplotlib.pyplot as plt

  X = np.random.normal(loc=0, scale=1.0, size=(100,2))
  print(X.shape)

  plt.scatter(X[:,0], X[:,1])
  plt.show()
