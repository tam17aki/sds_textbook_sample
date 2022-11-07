import re
import numpy as np
import pickle

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
import MeCab

# 変数の設定
NUM_TRAIN = 80  # 100個のデータのうち最初の80個を学習に利用
NUM_TEST = 20  # 100個のデータのうち残りの20個をテストに利用

LABEL_RESTAURANT = 0  # レストラン検索ドメインのラベル
LABEL_WEATHER = 1  # 天気案内ドメインのラベル

# 単語の系列とBag-of-Words表現を作成するための情報を受け取りベクトルを返す関数を定義
def make_bag_of_words(words, vocab, dim, pos_unk):

    vec = [0] * dim
    for w in words:

        # 未知語
        if w not in vocab:
            vec[pos_unk] = 1

        # 学習データに含まれる単語
        else:
            vec[vocab[w]] = 1

    return vec


# Word2vecで特徴量を作成する関数を定義
# ここでは文内の各単語のWord2vecを足し合わせたものを文ベクトルととして利用する
def make_sentence_vec_with_w2v(words, model_w2v):

    sentence_vec = np.zeros(model_w2v.vector_size)
    num_valid_word = 0
    for w in words:
        if w in model_w2v:
            sentence_vec += model_w2v[w]
            num_valid_word += 1

    # 有効な単語数で割
    sentence_vec /= num_valid_word
    return sentence_vec


def main():
    """main function.

    行数が長いので美しくないのですが、やむを得ない
    """
    # データを読み込む
    # レストランデータ
    with open("./data/slu-restaurant-annotated.csv", "r", encoding="utf-8") as f:
        lines_restaurant = f.readlines()

    with open("./data/slu-weather-annotated.csv", "r", encoding="utf-8") as f:
        lines_weather = f.readlines()

    # 学習とテストデータに分割（80対20）
    # 注）本来は交差検定を行うことが望ましい
    lines_restaurant_train = lines_restaurant[:NUM_TRAIN]
    lines_restaurant_test = lines_restaurant[NUM_TRAIN:]
    lines_weather_train = lines_weather[:NUM_TRAIN]
    lines_weather_test = lines_weather[NUM_TRAIN:]

    data_train = []
    for line in lines_restaurant_train:

        # 既に分割済みの単語系列を使用
        d = line.strip().split(",")[2].split("/")

        # 入力単語系列と正解ラベルのペアを格納
        data_train.append([d, LABEL_RESTAURANT])

    # 以下同様
    for line in lines_weather_train:
        d = line.strip().split(",")[2].split("/")
        data_train.append([d, LABEL_WEATHER])

    data_test = []
    for line in lines_restaurant_test:
        d = line.strip().split(",")[2].split("/")
        data_test.append([d, LABEL_RESTAURANT])

    for line in lines_weather_test:
        d = line.strip().split(",")[2].split("/")
        data_test.append([d, LABEL_WEATHER])

    # Bag-of-Words表現を作成する
    # 学習データの単語を語彙（カバーする単語）とする
    word_list = {}

    for data in data_train:
        for word in data[0]:
            word_list[word] = 1

    # print(word_list.keys())

    # 単語とそのインデクスを作成する
    word_index = {}
    for idx, word in enumerate(word_list.keys()):
        word_index[word] = idx

    # print(word_index)

    # ベクトルの次元数（未知語を扱うためにプラス１）
    vec_len = len(word_list.keys()) + 1
    # print(vec_len)

    # 学習データをBoW表現に変換する
    data_train_bow = []
    for data in data_train:
        feature_vec = make_bag_of_words(data[0], word_index, vec_len, vec_len - 1)
        data_train_bow.append([feature_vec, data[1]])

    # 入力と正解ラベルで別々のデータにする
    train_x = [d[0] for d in data_train_bow]
    train_y = [d[1] for d in data_train_bow]

    # SVMによる学習
    # 注）実際にはパラメータの調整が必要だが今回は行わない
    clf = svm.SVC()
    clf.fit(train_x, train_y)

    # 学習済みモデルを保存
    filename = "./data/slu-domain-svm.model"
    pickle.dump(clf, open(filename, "wb"))

    # テストデータの作成
    data_test_bow = []
    for data in data_test:
        feature_vec = make_bag_of_words(data[0], word_index, vec_len, vec_len - 1)
        data_test_bow.append([feature_vec, data[1]])

    test_x = [d[0] for d in data_test_bow]
    test_y = [d[1] for d in data_test_bow]

    # テストデータでの評価
    predict_y = clf.predict(test_x)

    # 評価結果を表示
    target_names = ["restaurant", "weather"]
    print(classification_report(test_y, predict_y, target_names=target_names))

    # LogisticRegressionによる学習と評価
    # 注）実際にはパラメータの調整が必要だが今回は行わない
    clf_lr = LogisticRegression()
    clf_lr.fit(train_x, train_y)

    # 学習済みモデルを保存
    filename = "./data/slu-domain-lr.model"
    pickle.dump(clf, open(filename, "wb"))

    # テストデータでの評価
    predict_y = clf_lr.predict(test_x)

    # 評価結果を表示
    target_names = ["restaurant", "weather"]
    print(classification_report(test_y, predict_y, target_names=target_names))

    # 学習済みWord2vecファイルを読み込む
    model_filename = "./data/entity_vector.model.bin"
    model_w2v = KeyedVectors.load_word2vec_format(model_filename, binary=True)

    # 単語ベクトルの次元数
    print(model_w2v.vector_size)

    # Word2vecを用いて学習を行う
    data_train_w2v = []
    for data in data_train:
        feature_vec = make_sentence_vec_with_w2v(data[0], model_w2v)
        data_train_w2v.append([feature_vec, data[1]])

    train_x = [d[0] for d in data_train_w2v]
    train_y = [d[1] for d in data_train_w2v]

    clf = svm.SVC()
    clf.fit(train_x, train_y)

    filename = "./data/slu-domain-svm-word2vec.model"
    pickle.dump(clf, open(filename, "wb"))

    # テストデータでの評価
    data_test_w2v = []
    for data in data_test:
        feature_vec = make_sentence_vec_with_w2v(data[0], model_w2v)
        data_test_w2v.append([feature_vec, data[1]])

    test_x = [d[0] for d in data_test_w2v]
    test_y = [d[1] for d in data_test_w2v]

    # テストデータでの評価
    predict_y = clf.predict(test_x)

    # 評価結果を表示
    target_names = ["restaurant", "weather"]
    print(classification_report(test_y, predict_y, target_names=target_names))

    # 自由なデータで試してみる
    # 入力データ
    test_input_list = ["京都駅周辺で美味しいラーメン屋さんを教えて", "横浜は晴れていますか"]

    # MeCabによる分割と特徴量抽出
    m = MeCab.Tagger("-Owakati")
    test_x = []
    for d in test_input_list:
        words_input = m.parse(d).strip().split(" ")
        feature_vec = make_sentence_vec_with_w2v(words_input, model_w2v)
        test_x.append(feature_vec)

    # 予測
    predict_y = clf.predict(test_x)

    for result, text in zip(predict_y, test_input_list):

        print("Input: " + text)

        if result == LABEL_RESTAURANT:
            print("Estimated domain: Restaurant")
        elif result == LABEL_WEATHER:
            print("Estimated domain: Weather")


if __name__ == "__main__":
    main()
