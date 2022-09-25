## メモ
### 疑問点などを記載していく
- 形態素解析器でよく使われるのはなに？
	- MeCab? / Juman? / Janome?
		- よく使われるという意味ではMecabな気がします（n=1）最近だとSudachiがありますね、簡単に分割の長さ変えられるのが使いやすいと思います。早さはMecabがダントツのよう。
			- 2019年末版 形態素解析器の比較: https://qiita.com/hi-asano/items/aaf406db875f1c81530e
	- 推論を考えるなら、Mecab一択！
	- Sudachiがセンテンスの長さを引数で変えられる
	- PyCon JP 2020:
		- https://speakerdeck.com/taishii/pycon-jp-2020
- 人間でも解釈の違いがあるので、機械で何が正しい解釈かを判断するのは難しいタスクだと感じる。
