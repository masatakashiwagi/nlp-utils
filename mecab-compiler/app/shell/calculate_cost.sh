#!/bin/bash
# sample exec: target filename = hoge.csv -> `./shell/calculate_cost.sh hoge`

# ユーザー辞書のコスト計算を「mecab-ipadic」のモデルで行う
# ディレクトリが無ければ作成する
DIR="./mecab-working-dir"
if [[ ! -d $DIR ]]; then
	echo "Not existing working dir."
    echo "Make a working dir."
	mkdir mecab-working-dir
fi

# data/*をworkingディレクトリにコピーする
cd mecab-working-dir
echo "Copy data dir."
cp -r ../data/* ./

DICT_INDEX_PATH=$(mecab-config --libexecdir)/mecab-dict-index
WORKING_DIR=$(pwd)
# ファイル名（拡張子は不要）
FILENAME=$1

# ファイルの存在確認を行う
if [[ -z "$1" ]]; then
	echo "Blank filename :/"
	exit 1
fi

# ipadic辞書の設定
# ファイルの解凍
tar xvzf mecab-ipadic-2.7.0-20070801.tar.gz

# 文字コードをEUC-JPからUTF-8に変換する
echo "Transform string code from EUC-JP to UTF-8."
nkf --overwrite -Ew mecab-ipadic-2.7.0-20070801/*

# mecab-ipadicのディレクトリに移動する
cd mecab-ipadic-2.7.0-20070801
# 文字コードをEUC-JPからUTF-8に変換する
sed -i -e "s/config-charset = EUC-JP/config-charset = UTF-8/g" dicrc

$DICT_INDEX_PATH -f utf-8 -t utf-8
echo "Run configure"
./configure --with-charset=utf8

# ipadicのモデルファイルの設定
cd $WORKING_DIR

# 文字コードをEUC-JPからUTF-8に変換する
nkf --overwrite -Ew mecab-ipadic-2.7.0-20070801.model
sed -i -e "s/charset: EUC-JP/charset: UTF-8/g" mecab-ipadic-2.7.0-20070801.model

$DICT_INDEX_PATH \
	-m "${WORKING_DIR}/mecab-ipadic-2.7.0-20070801.model" \
	-d "${WORKING_DIR}/mecab-ipadic-2.7.0-20070801" \
	-u "${FILENAME}_cost.csv" \
	-f utf-8 -t utf-8 \
	-a "${FILENAME}.csv" \
&& \
echo "Calculated cost for ${FILENAME}.csv to ${FILENAME}_cost.csv."

echo "Copy ${FILENAME}_cost.csv to data dir."
cp "${FILENAME}_cost.csv" "../data/${FILENAME}_cost.csv"