import pickle

import sklearn_crfsuite
import MeCab

from functools import wraps
from sklearn_crfsuite.utils import flatten

# 変数の設定
NUM_TRAIN = 80  # 100個のデータのうち最初の80個を学習に利用
NUM_TEST = 20  # 100個のデータのうち残りの20個をテストに利用


def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)

    return wrapper


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    """
    from sklearn import metrics

    # return metrics.classification_report(y_true, y_pred, labels, **kwargs)
    return metrics.classification_report(y_true, y_pred, labels=labels, **kwargs)


def main():
    """main function.
    行数が長いので美しくないのですが、やむを得ない
    """
    # データを読み込む

    # 使用するドメインを設定する
    USED_DATASET = 1  # 0: 'restaurant', 1: 'weather'

    if USED_DATASET == 0:
        # レストランデータ
        with open("./data/slu-restaurant-annotated.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 正解ラベルの一覧
        LABELS = [
            "B-budget",
            "I-budget",
            "B-mood",
            "I-mood",
            "B-loc",
            "I-loc",
            "B-genre",
            "I-genre",
            "B-style",
            "I-style",
            "B-rate",
            "I-rate",
            "O",
        ]

        SAVED_MODEL = "./data/slu-slot-restaurant-crf.model"

    elif USED_DATASET == 1:
        # 天気案内
        with open("./data/slu-weather-annotated.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()

        LABELS = [
            "B-wh",
            "I-wh",
            "B-place",
            "I-place",
            "B-when",
            "I-when",
            "B-type",
            "I-type",
            "B-temp",
            "I-temp",
            "B-rain",
            "I-rain",
            "B-wind",
            "I-wind",
            "O",
        ]

        SAVED_MODEL = "./data/slu-slot-weather-crf.model"

    # 学習とテストデータに分割（80対20）
    # 注）本来は交差検定を行うことが望ましい
    lines_train = lines[:NUM_TRAIN]
    lines_test = lines[NUM_TRAIN:]

    data_train = []
    for line in lines_train:

        # 既に分割済みの単語系列を使用
        d = line.strip().split(",")[2].split("/")

        # 正解ラベルの系列
        a = line.strip().split(",")[3].split("/")

        # 入力単語系列と正解ラベル系列のペアを格納
        data_train.append([d, a])

    # テストデータも同様
    data_test = []
    for line in lines_test:
        d = line.strip().split(",")[2].split("/")
        a = line.strip().split(",")[3].split("/")
        data_test.append([d, a])

    # 最初のデータだけ表示
    print(data_train[0])
    print(data_test[0])

    # 入力と正解ラベルで別々のデータにする
    # 注）sklearn-crfsuiteのOne-hot表現は単語データをそのまま入力すればよい
    train_x = [d[0] for d in data_train]
    train_y = [d[1] for d in data_train]

    # CRFによる学習
    # 注）実際にはパラメータの調整が必要だが今回は行わない
    clf = sklearn_crfsuite.CRF()
    clf.fit(train_x, train_y)

    # 学習済みモデルを保存
    pickle.dump(clf, open(SAVED_MODEL, "wb"))

    # テストデータの作成
    test_x = [d[0] for d in data_test]
    test_y = [d[1] for d in data_test]

    # テストデータでの評価
    predict_y = clf.predict(test_x)

    # 評価結果を表示
    print(flat_classification_report(test_y, predict_y, labels=LABELS))

    # 自由なデータで試してみる
    # 入力データ
    # input_data = 'この辺りであっさりした四川料理のお店に行きたい'
    input_data = "横浜の今日の天気"

    # MeCabによる分割と特徴量抽出
    m = MeCab.Tagger("-Owakati")
    words_input = m.parse(input_data).strip().split(" ")

    # 予測
    predict_y = clf.predict([words_input])[0]
    for word, tag in zip(words_input, predict_y):
        print(word + "\t" + tag)


if __name__ == "__main__":
    main()
