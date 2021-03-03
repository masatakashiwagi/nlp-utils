## メモ
### 疑問点などを記載していく

- ラベル付きコーパスのいうラベルとは品詞とかのこと？
    - ラベルとは？（画像などの教師あり学習みたく、0or1のラベルではないと思うが...）

## スクリプトの実行方法
- corpus_marker.pyを動かして、Yelpのレビューデータを取得してコーパスを作成する方法
	- main関数の`API_KEY`を自身のAPI KEYをセットし、`save_file_path`にdataset.jsonを保存するパスを記載する
		- サンプル: save_file_path = './data/dataset.json'