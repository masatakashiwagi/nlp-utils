# mecab-compiler
- MeCabのユーザー辞書作成

## ローカルでのビルド
```bash
docker compose build
```

## 辞書のコスト計算
ユーザー辞書のコスト計算を`mecab-ipadic`のモデルを使って行う．
`sample_user_dict.csv`というcsvファイルを用意し，辞書に追加したいワードを準備する．

```bash
docker compose up cost-calculator
```

`sample_user_dict_cost.csv`というcsvファイルが出力される．

## 辞書ファイルのコンパイル
「辞書のコスト計算」で出力されたファイル（`sample_user_dict_cost.csv`）を使用する．

```bash
docker compose up dict-maker
```

`sample_user_dict.dic`という辞書ファイルが出力される．