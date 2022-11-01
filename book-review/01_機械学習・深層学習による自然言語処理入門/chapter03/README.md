## メモ
### 疑問点などを記載していく
- ラベル付きコーパスのいうラベルとは品詞とかのこと？
    - ラベルとは？（画像などの教師あり学習みたく、0or1のラベルではないと思うが...）
	- 色んな粒度が考えられる
 		- 文全体だったり、名詞句とか...
 		- IOBタグ
 			- この辺が参考になりそう？
			 	- https://qiita.com/Hironsan/items/326b66711eb4196aa9d4
 		- （小ネタ）nishikaの固有表現抽出コンペのアノテーションが大変そうな話
		 	- https://note.com/nishika_inc/n/n78447a423abe

## スクリプトの実行方法
- corpus_marker.pyを動かして、Yelpのレビューデータを取得してコーパスを作成する方法
	- main関数の`API_KEY`を自身のAPI KEYをセットし、`save_file_path`にdataset.jsonを保存するパスを記載する
		- サンプル: save_file_path = './data/dataset.json'