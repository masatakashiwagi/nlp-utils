# 特徴量選択
頑健な学習モデルの構築のため、特徴集合のうち意味のある部分集合だけを選択する手法のことを指す。

参考
- [特徴選択 - Wikipedia](https://ja.wikipedia.org/wiki/%E7%89%B9%E5%BE%B4%E9%81%B8%E6%8A%9E)
- [Feature selection - Wikipedia](https://en.wikipedia.org/wiki/Feature_selection)

特徴量選択する目的はWikipediaによると4つある
- モデルを単純化して、ユーザーが解釈しやすくするため
- モデルの学習時間を短くするため
- 次元の呪いを回避するため
- 過学習を避けることで、モデルの汎化性能を高めるため

特徴量選択の方法として、以下の3つに大別されることが多い
- Filter method
- Wrapper method
- Embedded method

この辺りの話は調べれば、すぐに出てくる内容だと思うので、特段ここに書かなくてもいいかもだが。。。
## Filter method
- モデルに関係なく特徴量を選択する方法（統計的な手法で特徴量を選択する）

例えば、以下の方法などがある
- ピアソンの相関関係
- MIC（相互情報量）

といった統計的な手法に基づき削除する方法や以下のものもある
- ドメイン知識などから関係のない特徴量を削除する方法

## Wrapper method
- 機械学習のモデルを用いて特徴量を選択する（最適な特徴量を探索する）
問題点：
- 計算コストの高さ
- モデルの過適合が起こる可能性がある

### 最適な特徴量の探索方法
1. 部分的な特徴量の組み合わせを行う
2. [1]で選んだ特徴量を用いてモデルを作る
3. [2]で作成したモデルの性能を評価する


### 特徴量の組み合わせ方法
1. Forward Selection
    - 特徴量を1つずつ追加していく手法
2. Backward Elimination
    - 特徴量を1つずつ削除していく方法
3. Recursive Feature Elimination
    - 再帰的に特徴量を削除していく方法

例えば、モデルに寄与した特徴量かもしくは寄与しなかった特徴量を取り出す。 残った特徴量を使って再度モデルを作り、寄与した特徴量もしくは寄与しなかった特徴量を取り出すことを特徴量が指定の数になるまで繰り返す。

RFEはsklearnに実装されている
- [sklearn.feature_selection.RFE — scikit-learn 0.20.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)

引数のestimatorに渡せるのは"coef_"or"feature_importances_"があるモデルのみ

RFEの特徴量の数を選択する方法は、ハイパーパラメータとして組み込んで最適な数にする方法もある
    - https://blog.amedama.jp/entry/rfe-feature-selection

他には、[Boruta](https://github.com/scikit-learn-contrib/boruta_py)という特徴量選択をしてくれるものもある

## Embedded method
モデル学習時にアルゴリズムで評価と特徴量選択を同時にしてくれる
有名な方法として、Lasso, Ridgeがある: モデルに必要でない特徴量をスパース化してくれる

他には、Tree-basedのfeature importanceなどがある（feature importanceの上位X項目だけ選択する）

[permutaion importance](https://www.kaggle.com/dansbecker/permutation-importance)で特徴量選択する方法もある: permutaion importanceの結果が0orマイナスの値の特徴量を削除する

## その他
PCAなどの次元削減手法を用いて、無相関状態にしてそれを特徴量とする方法などもある
ただし、元の特徴量がどの主成分方向を向いているかは別途確認する必要がある
（各成分がn次の主成分に押し込められるので、固有ベクトルなどを取り出して各主成分の中身を見ないと元の特徴量がわからない）

## 備考
基本的には削除する方向の特徴量選択が多い気がするので、特徴量を増やす方向の手法も知りたい（交互作用、変化量...とか？）
P.S. 適宜、追加変更修正をお願いします。

## 参考
特徴量選択のまとめ: https://qiita.com/shimopino/items/5fee7504c7acf044a521