"""対話管理（フレームによる方法）"""

# 下記のフレームを実装します。 「制約」の列はそのスロットが埋まる必要があるか否かを表します。
# つまり、地域とジャンルは必ず指定する必要がありますが、予算は指定されていなくても検索可能とします。

# スロット名 制約
#
# 地域    　必須
# ジャンル  必須
# 予算    　任意

from __future__ import division


class DmFrame(object):

    # フレームの現在の状態を保持
    # 辞書型のKeyにスロット名、Valueにスロット値を格納する
    current_frame = {}

    # フレームで必須の情報がすべて埋まったかどうかを保持する
    current_frame_filled = False

    def __init__(self):
        """初期化."""
        self.def_frame()
        self.def_serif()
        self.reset()

    # フレームを定義
    def def_frame(self):
        """フレームを定義."""
        # スロット名は言語理解結果のスロット名と対応するようにします
        # スロット名、制約
        self.frame = [
            ["place", "mandatory"],  # 場所, 必須
            ["genre", "mandatory"],  # ジャンル, 必須
            ["budget", "optional"],  # 予算, 任意
        ]

    # システム発話を定義
    def def_serif(self):
        """システム発話を定義."""
        # 必須項目を尋ねるセリフ
        # 辞書型でKeyをスロット名、Valueをセリフにする
        self.utterances = {
            "place": "地域を指定してください",
            "genre": "ジャンルを指定してください",
        }

        # 最初の発話
        self.utterance_start = "こんにちは。京都レストラン案内です。ご質問をどうぞ。"

    # 最後の発話は条件に応じて生成する
    def gen_utterance_last(self):
        """最後のシステム発話を生成."""
        system_utterance = ""

        # Mandatoryである"place"と"genre"が埋まっているかチェック
        if "place" in self.current_frame and "genre" in self.current_frame:

            # Optionalである"budget"が埋まっていれば
            if "budget" in self.current_frame:
                system_utterance = "地域は%sで、ジャンルは%s、予算は%sですね。検索します。" % (
                    self.current_frame["place"],
                    self.current_frame["genre"],
                    self.current_frame["budget"],
                )

            # "budget"が埋まっていなければ
            else:
                system_utterance = "地域は%sで、ジャンルは%sですね。検索します。" % (
                    self.current_frame["place"],
                    self.current_frame["genre"],
                )

        # システム発話を表す文字列が戻り値
        return system_utterance

    # 入力であるユーザ発話に応じて、フレームの状態を更新し、システム発話を出力
    # ただし、ユーザ発話の情報は「意図、スロット名、スロット値」のlistとする
    def enter(self, user_utterance):
        """入力であるユーザ発話に応じて、フレームの状態を更新し、システム発話を出力."""
        # １つのユーザ発話に複数のスロットの値が含まれることもある
        for slot_user_utterance in user_utterance:

            # スロット名とスロット値を取得
            input_slot_name = slot_user_utterance["slot_name"]  # 例 "place"
            input_slot_value = slot_user_utterance["slot_value"]  # 例 "京都駅周辺"

            # フレームの状態を更新
            self.current_frame[input_slot_name] = input_slot_value

        system_utterance = ""

        # 現在のフレームの状態から制約が"mandatory"で不足しているものを探索
        mandatory_need = False
        for slot in self.frame:

            slot_name = slot[0]  # 例 "place"
            slot_condition = slot[1]  # 例 "京都駅周辺"

            if slot_condition == "mandatory" and slot_name not in self.current_frame:
                system_utterance = self.utterances[slot_name]
                mandatory_need = True
                break

        # すべての"mandatory"の要素が埋まっていたら終了
        if mandatory_need == False:

            # システムの発話を生成
            system_utterance = self.gen_utterance_last()

            self.current_frame_filled = True

        return system_utterance

    # 初期状態にリセットする
    def reset(self):
        """初期状態にリセットする."""
        self.current_frame = {}
        self.current_frame_filled = False


if __name__ == "__main__":

    dm = DmFrame()

    print("パターン１")

    # 初期状態の発話を表示
    print("システム発話 : " + dm.utterance_start)

    # ユーザ発話を設定
    user_utterance = [{"slot_name": "place", "slot_value": "京都駅周辺"}]
    print("ユーザ発話")
    print(user_utterance)
    print()

    # 次のシステム発話を表示
    print("システム発話")
    print(dm.enter(user_utterance))

    # ユーザ発話を設定
    user_utterance = [{"slot_name": "genre", "slot_value": "和食"}]
    print("ユーザ発話")
    print(user_utterance)
    print()

    # 次のシステム発話を表示
    print("システム発話")
    print(dm.enter(user_utterance))
    print()

    dm.reset()

    print("-------------------------")
    print("パターン２")

    # 初期状態の発話を表示
    print("システム発話 : " + dm.utterance_start)

    # ユーザ発話を設定
    user_utterance = [
        {"slot_name": "place", "slot_value": "京都駅周辺"},
        {"slot_name": "budget", "slot_value": "5000円"},
    ]
    print("ユーザ発話")
    print(user_utterance)
    print()

    # 次のシステム発話を表示
    print("システム発話")
    print(dm.enter(user_utterance))

    # ユーザ発話を設定
    user_utterance = [{"slot_name": "genre", "slot_value": "和食"}]
    print("ユーザ発話 (2)")
    print(user_utterance)
    print()

    # 次のシステム発話を表示
    print("システム発話(2)")
    print(dm.enter(user_utterance))
