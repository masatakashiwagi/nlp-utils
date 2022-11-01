#!/bin/bash

# コスト付きの辞書をコンパイルする
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

cd $WORKING_DIR

# 辞書ファイル生成
$DICT_INDEX_PATH \
	-d "${WORKING_DIR}/mecab-ipadic-2.7.0-20070801" \
	-u "${FILENAME}.dic" \
	-f utf-8 -t utf-8 \
	"${FILENAME}_cost.csv" \
&& \
echo "Made dict file to ${FILENAME}.dic."

echo "Copy ${FILENAME}.dic to data dir."
cp "${FILENAME}.dic" "../data/${FILENAME}.dic"