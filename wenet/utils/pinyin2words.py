from typing import List

from pypinyin import Style, pinyin
from Pinyin2Hanzi import DefaultHmmParams, viterbi

hmm_params = DefaultHmmParams()

def get_possible_words(word: str) -> List[str]:
    phones = pinyin(word, neutral_tone_with_five=True, v_to_u=True, style=Style.NORMAL)
    phones = [r[0] for r in phones]
    res = viterbi(hmm_params, phones)
    return [r.path for r in res]

