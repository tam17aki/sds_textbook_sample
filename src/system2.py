"""システム統合2（フレーム）.

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
# 言語理解 ： ルールベース（フレーム）
# 対話管理 ： フレーム

from asr_vosk_streaming_vad import get_vosk_recognizer, get_asr_result
from tts_google import GoogleTextToSpeech
from dm_frame import DmFrame
from slu_rule import SluRule


def main():
    """Perform demo."""
    # 音声認識器の初期化
    vosk_asr = get_vosk_recognizer()

    # 音声合成器の初期化
    tts = GoogleTextToSpeech()

    # 対話管理部の初期化
    dm = DmFrame()

    # 言語理解部の初期化
    slu_parser = SluRule()

    # 初期状態の発話
    system_utterance = dm.utterance_start
    tts.generate(system_utterance)
    print("システム： " + system_utterance)
    tts.play()

    # 全てのフレームが埋まるまで対話を続ける
    while dm.current_frame_filled == False:

        # 音声認識入力を得る
        print("<<please speak>>")
        result_asr_utterance = get_asr_result(vosk_asr)
        print("ユーザ： " + result_asr_utterance)

        # 言語理解
        result_slu = slu_parser.parse_frame(result_asr_utterance)
        print(result_slu)

        # 対話管理へ入力
        system_utterance = dm.enter(result_slu)
        tts.generate(system_utterance)
        print("システム： " + system_utterance)
        tts.play()


if __name__ == "__main__":
    main()
