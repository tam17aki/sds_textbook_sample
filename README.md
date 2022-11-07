# 音声対話システム －基礎から実装まで－
**注意：このページは『音声対話システム －基礎から実装まで－』（オーム社, 2022年10月15日 発売）のサポートサイトをforkした改造版**です。各種のスクリプトに色々と手を加えています。

<img src="https://user-images.githubusercontent.com/31427099/192208284-5c1e25a6-2188-401d-a234-8409f84d04cd.jpg" width="30%">

* * *
## 書籍情報

- オーム社（目次などの詳細はこちら）
　https://www.ohmsha.co.jp/book/9784274229541/
- Amazon
　https://www.amazon.co.jp/dp/4274229548/
- honto
　https://honto.jp/netstore/search.html?k=978-4-274-22954-1&srchf=1
- 紀伊国屋書店
　https://www.kinokuniya.co.jp/f/dsg-01-9784274229541

## サンプルソースコード（Hands-on）

Hands-onとして紹介したソースコードのリンクは以下の通りです。  
また、書籍に掲載しきれなかった実装についてもこちらで随時追加・更新しています。 

環境構築の方法の最新情報は[こちら](environment.md)

### 3章（クラウド型音声認識の利用）

- [環境構築](environment.md#3%E7%AB%A0-%E3%82%AF%E3%83%A9%E3%82%A6%E3%83%89%E5%9E%8B%E9%9F%B3%E5%A3%B0%E8%AA%8D%E8%AD%98%E3%81%AE%E5%88%A9%E7%94%A8)
- [音声認識（ストリーミング型）](src/asr_google_streaming.ipynb)
- [音声認識（ストリーミング型・発話区間検出有り）](src/asr_google_streaming_vad.ipynb)

### 4章（言語理解の実装）

- [環境構築](environment.md#4%E7%AB%A0-%E8%A8%80%E8%AA%9E%E7%90%86%E8%A7%A3%E3%81%AE%E5%AE%9F%E8%A3%85)
- [言語理解（ルールによる方法）](src/slu_rule.ipynb)
- [言語理解（機械学習による方法：ドメイン推定）](src/slu_ml_domain.ipynb)
- [言語理解（機械学習による方法：スロット値推定）](src/slu_ml_slot.ipynb)

### 5章（対話管理の実装）

- [対話管理（有限オートマトンによる方法）](src/dm_fst.ipynb)
- [対話管理（フレームによる方法）](src/dm_frame.ipynb)

### 6章（用例ベースの実装）

- [用例ベース](src/example_based.ipynb)

### 7章（クラウド型音声合成の利用）

- [環境構築](environment.md#7%E7%AB%A0-%E3%82%AF%E3%83%A9%E3%82%A6%E3%83%89%E5%9E%8B%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%81%AE%E5%88%A9%E7%94%A8)
- [音声合成](src/tts_google.ipynb)

### 8章（システム統合）

- [システム統合１（有限オートマトン）](src/system1.ipynb)
- [システム統合２（フレーム）](src/system2.ipynb)
- [システム統合３（フレーム＋機械学習言語理解）](src/system3.ipynb)
- [システム統合４（有限オートマトン＋機械学習言語理解）](src/system4.ipynb)
- [システム統合５（用例ベース）](src/system5.ipynb)

* * *

