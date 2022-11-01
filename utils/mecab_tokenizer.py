"""MeCabを用いてTokenizerを行うクラス
"""

import MeCab


class MeCabTokenizer(object):
    """ MeCabで形態素解析を行う

    表層形  品詞  品詞細分類1  品詞細分類2  品詞細分類3  活用形  活用型  原形  読み  発音を返却する
    sample code:
        option = '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
        mecab = MeCab.Tagger(option)
        parser = mecab.parse
        chunks = parser(text.rstrip()).splitlines()[:-1]
        for chunk in chunks:
            chunk.split('\t')[1].split(',')[0]
            surface, feature = chunk.split('\t')
            feature = feature.split(',')
            if len(feature) <= 7:  # 読みがない場合は追加
                feature.append('*')
            print(f'表層形: {surface}\t| 読み: {feature[7]}\t| 原型: {feature[6]}\t| 品詞: {feature[0]}\
            \t| 品詞細分類1: {feature[1]}\t| 品詞細分類2: {feature[2]}\t| 品詞細分類3: {feature[3]}\
            \t| 活用形: {feature[4]}\t| 活用型: {feature[5]}')

    ref:
        http://www.ic.daito.ac.jp/~mizutani/mining/morphology.html
    """

    def __init__(
            self,
            sys_dic_path: str = '',
            user_dic_path: str = '',
            stopwords_path: str = '',
            scope_pos=None):
        """コンストラクタ（初期化のための関数）

        Args:
            sys_dic_path (str, optional): システム辞書. Defaults to ''.
            user_dic_path (str, optional): ユーザー辞書. Defaults to ''.
            stopwords_path (str, optional): ストップワード. Defaults to ''.
            scope_pos (_type_, optional): 品詞の種類. Defaults to None.
        """

        option = ''
        if sys_dic_path:
            option += ' -d {0}'.format(sys_dic_path)
        if user_dic_path:
            option += ' -u {0}'.format(user_dic_path)
        mecab = MeCab.Tagger(option)
        self.parser = mecab.parse

        if stopwords_path:
            self.stopwords = self._set_stopwords(stopwords_path)
        else:
            self.stopwords = []
        if scope_pos is None:
            self.scope_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.scope_pos = scope_pos

    def tokenize(self, text: str, pos: bool = False) -> list:
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 文字列の分割
        chunks = self.parser(text.rstrip()).splitlines()[:-1]
        res = []
        for chunk in chunks:
            chunk.split('\t')[1].split(',')[0]
            surface, feature = chunk.split('\t')
            feature = feature.split(',')
            p = feature[0]
            base = feature[6]
            if base == '*':
                base = surface  # 原型が存在しないものは、元の単語を返却する
            if p in self.scope_pos and base not in self.stopwords:
                if pos:
                    res.append((base, p))
                else:
                    res.append(base)

        return res

    @staticmethod
    def _set_stopwords(file_path: str) -> list:
        """Stopwordsの設定
        Args:
            file_path (str): 対象となるストップワードのファイル

        Returns:
            list: ストップワードのlist
        """
        with open(file_path) as f:  # テキストファイルからストップワードを追加する
            text_file = f.readlines()

        _stopwords = [line.strip() for line in text_file]
        _stopwords = [ss for ss in _stopwords if not ss == u'']

        # マージして重複の削除
        stopwords_set = set(_stopwords)

        return list(stopwords_set)
