import io
import os
from typing import Dict, List, Union
import copy
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from wenet.cli.hub import Hub
from wenet.paraformer.search import (gen_timestamps_from_peak,
                                     paraformer_greedy_search, all_paraformer_greedy_search)
from wenet.text.paraformer_tokenizer import ParaformerTokenizer

def get_key_by_value(dict_obj, value):
    return next((k for k, v in dict_obj.items() if v == value), None)

class Paraformer:

    def __init__(self, model_dir: str, resample_rate: int = 16000) -> None:

        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        self.resample_rate = resample_rate
        self.device = torch.device("cpu")
        self.tokenizer = ParaformerTokenizer(symbol_table=units_path)


    @torch.inference_mode()
    def transcribe_batch(self,
                         audio_files: List[Union[str, bytes]],
                         tokens_info: bool = False) -> List[Dict]:
        feats_lst = []
        feats_lens_lst = []
        for audio in audio_files:


            if isinstance(audio, bytes):
                with io.BytesIO(audio) as fobj:
                    waveform, sample_rate = torchaudio.load(fobj,
                                                            normalize=False)
            else:
                waveform, sample_rate = torchaudio.load(audio, normalize=False)

            waveform = waveform.to(torch.float)
            if sample_rate != self.resample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.resample_rate)(waveform)


            feats = kaldi.fbank(waveform,
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                energy_floor=0.0,
                                sample_frequency=self.resample_rate,
                                window_type="hamming")
            feats_lst.append(feats)
            feats_lens_lst.append(torch.tensor(feats.shape[0], dtype=torch.int64))
        feats_tensor = torch.nn.utils.rnn.pad_sequence(feats_lst, batch_first=True).to(device=self.device)
        feats_lens_tensor = torch.tensor(feats_lens_lst, device=self.device)

        decoder_out, token_num, tp_alphas, frames = self.model.forward_paraformer(
            feats_tensor, feats_lens_tensor)
        frames = frames.cpu().numpy()
        cif_peaks = self.model.forward_cif_peaks(tp_alphas, token_num)

        results = paraformer_greedy_search(decoder_out, token_num, cif_peaks)

        r = []
        for (i, res) in enumerate(results):
            result = {}
            result['confidence'] = res.tokens_confidence[0]
            print("res.tokens",res.tokens[0])
            result['text'] = self.tokenizer.detokenize(res.tokens)[0]
            if tokens_info:
                tokens_info_l = []
                times = gen_timestamps_from_peak(res.times,
                                                 num_frames=frames[i],
                                                 frame_rate=0.02)

                for i, x in enumerate(res.tokens[:len(times)]):
                    tokens_info_l.append({
                        'token':
                        self.tokenizer.char_dict[x],
                        'start':
                        round(times[i][0], 3),
                        'end':
                        round(times[i][1], 3),
                        'confidence':
                        round(res.tokens_confidence[i], 2)
                    })
                    result['tokens'] = tokens_info_l
            r.append(result)
            print("r",r)
        return r

    @torch.inference_mode()
    def transcribe_with_labels(self, audio_files: List[Union[str, bytes]], labels_dict: dict, tokens_info: bool = True) -> dict:
        feats_lst = []
        feats_lens_lst = []
        if isinstance(audio_files, str):
            audio_files = [audio_files]

        for audio in audio_files:

            if isinstance(audio, bytes):
                with io.BytesIO(audio) as fobj:
                    waveform, sample_rate = torchaudio.load(fobj,
                                                            normalize=False)
            else:
                waveform, sample_rate = torchaudio.load(audio, normalize=False)

            waveform = waveform.to(torch.float)
            if sample_rate != self.resample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.resample_rate)(waveform)

            feats = kaldi.fbank(waveform,
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                energy_floor=0.0,
                                sample_frequency=self.resample_rate,
                                window_type="hamming")
            feats_lst.append(feats)
            feats_lens_lst.append(torch.tensor(feats.shape[0], dtype=torch.int64))
        feats_tensor = torch.nn.utils.rnn.pad_sequence(feats_lst, batch_first=True).to(device=self.device)
        feats_lens_tensor = torch.tensor(feats_lens_lst, device=self.device)

        decoder_out, token_num, tp_alphas, frames = self.model.forward_paraformer(
            feats_tensor, feats_lens_tensor)

        # 构建yinsu字典
        yinsu_with_grade = {}
        for item in labels_dict:
            all_text = [labels_dict[item][0], labels_dict[item][1]]
            target_text = 'None'
            target_confidence = 0
            for text in all_text:
                key = get_key_by_value(self.tokenizer.char_dict, text)
                if key:
                    log_probs_at_k = decoder_out[0, :, key]
                    # 将对数概率转换为概率，并找到最大值
                    max_confidence_at_k = torch.max(torch.exp(log_probs_at_k)).item()
                    if max_confidence_at_k > target_confidence:
                        target_confidence = max_confidence_at_k
                        target_text = text

            yinsu_with_grade[item] = {
                'text': target_text,
                'confidence': target_confidence
            }
        new_dict = {
            'yinsu_with_grade': yinsu_with_grade
        }
        print("Praformer的输出结果如下:")
        return new_dict

    def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
        result = self.transcribe_batch([audio_file], tokens_info)[0]
        return result

    def align(self, audio_file: str, label: str) -> dict:
        raise NotImplementedError("Align is currently not supported")


def load_model(model_dir: str = None,
               gpu: int = -1,
               device: str = "cpu") -> Paraformer:
    if model_dir is None:
        model_dir = Hub.get_model_by_lang('paraformer')
    if gpu != -1:
        # remain the original usage of gpu
        device = "cuda"
    paraformer = Paraformer(model_dir)
    paraformer.device = torch.device(device)
    paraformer.model.to(device)
    return paraformer
