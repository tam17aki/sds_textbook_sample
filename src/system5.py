"""システム統合5（用例ベース対話システム）.

Copyright (C) 2022 by Akira TAMAMORI
Copyright (C) 2022 by Koji INOUE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# これまでに実装してきたモジュールを統合して音声対話システムを作成してみましょう。

# 仕様
# 音声認識 ： VOSK（ストリーミング、オリジナルVAD）
# 音声合成 ： Google
# 対話モデル ： 用例ベース

from asr_vosk_streaming_vad import get_vosk_recognizer, get_asr_result
from tts_google import GoogleTextToSpeech
from example_based import ExampleBased


def main():
    """Perform demo."""
    # 音声認識器の初期化
    vosk_asr = get_vosk_recognizer()

    # 音声合成器の初期化
    tts = GoogleTextToSpeech()

    # 対話モデルの初期化
    example_based = ExampleBased()

    # ユーザ発話に「終了」含まれるまで繰り返す
    while True:

        # 音声認識入力を得る
        # 音声認識入力を得る
        print("<<please speak>>")
        result_asr_utterance = get_asr_result(vosk_asr)
        print("ユーザ： " + result_asr_utterance)

        # ユーザ発話に「終了」含まれていれば終了
        if "終了" in result_asr_utterance:
            print("<<<終了します>>>")
            break

        # 用例を検索（word2vec版を使用）
        result_asr_utterance_mecab = example_based.parse_mecab(result_asr_utterance)
        response, cos_dist_max = example_based.matching_word2vec(
            result_asr_utterance_mecab
        )
        print("%s (%f)" % (response, cos_dist_max))

        # システム応答を再生
        tts.generate(response)
        print("システム： " + response)
        tts.play()

        print()


if __name__ == "__main__":
    main()
