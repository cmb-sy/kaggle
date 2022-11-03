https://www.kaggle.com/code/jocker3/feedback-prize-english-language-learning

### 目的
8~12年生の英語学習者の言語能力を評価すること。

### 手法
full_textから、カラムの熟練度を予測する。
評価指標としてMCRMSEを用いる。

### カラム
数値が高いほど熟練度が高いということ
full_text : 全文
cohesion : 凝集（目的変数）
syntax : 構文（目的変数）
vocabulary : 単語（目的変数）
phraseology : 言い回し（目的変数）
grammar : 文法（目的変数）
conventions : 慣例（目的変数）

### 考え
目的変数が文字列であり、種類はデータ数N個ある。
それに対するデータ加工を考える必要がある。

### その他
期限 : 11/30
