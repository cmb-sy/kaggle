from platform import node
import sys
from tkinter import Variable
sys.path.append("../")

#ノードクラスの定義
class Node:
  def __init__(self, data): #コンストラクタ
    self.parent_node = data #ノードがもつ数値
    self.kid_left_node = None #左エッジ
    self.kid_right_node = None #右エッジ


class binary_serach_tree:
  def ___init___(self, number):
    self.root = None
    for num in number:
      self.insert(num)
  
  def main():
    pass

# 関数のreturnに行くと自動で関数を出て値を返す
# 木にデータを挿入
  def insert(self, data):
    node = self.root
   #　ここでrootが空であるのならば、上で作成したNodeの構造自体を代入
    if self.root == None:
      self.root = Node(data)
    else:
      while True:
        x = node.parent_node
        
        if data < x:
          if node.kid_left_node == None:
            # 左の子ノードになにもなければ、値ではなく上で作成したNodeの構造自体を左へ代入
            node.kid_left_node = Node(data)
            return
          node = node.kid_left_node
        
        elif data > x:
          if node.kid_right_node == None:
            node.kid_right_node = Node(data)
            return
          node = node.kid_right_node
        
        elif data == x:
          node = node.parent_node
          return

    