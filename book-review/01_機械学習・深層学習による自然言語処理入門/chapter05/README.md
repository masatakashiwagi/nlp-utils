## メモ
### 疑問点などを記載していく
- P93 Bag-of-Ngrams（BoW）はBoNでは無いの？: Bag-of-Wordsのこと？
- p102 トライグラムだと精度が上がらなかった
    - コーパス自体に3単語までいくと同じような語がでてこないという性質があるのかもしれない
    - 例えば「薄い携帯電話買う」「薄い携帯電話好き」「薄い携帯電話壊れる」みたいなtri-gramまで考慮すると、そもそもこの語順がコーパスの中に全然でてこないので、特徴量としてあまり意味がなさなくなるみたいな... by たかぱい大先生
- EDA
    - ユニグラムとバイグラムでセットを用意して、出現頻度とかどんなかんじの単語が上位に来るかを見るのがいいかも
    
### 確認しておく項目
- Word2Vecの論文とその解説論文
    - Efficient Estimation of Word Representations in Vector Space
        - https://arxiv.org/pdf/1301.3781.pdf
    - word2vec Parameter Learning Explained
        - https://arxiv.org/pdf/1411.2738.pdf
- SWEM: 単語ベクトルを文書ベクトルに変換する方法
    - Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms
        - https://arxiv.org/pdf/1805.09843.pdf
    - https://yag-ays.github.io/project/swem/
- Doc2Vec
    - Distributed Representations of Sentences and Documents
        - https://arxiv.org/pdf/1405.4053.pdf
    - https://qiita.com/g-k/items/5ea94c13281f675302ca
- BM25
    - Elasticsearchのスコアリングで使われているアルゴリズム
    - https://itdepends.hateblo.jp/entry/2020/01/05/112447
