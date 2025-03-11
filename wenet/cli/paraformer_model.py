import io
import os
from typing import Dict, List, Union
import copy
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from wenet.cli.hub import Hub
from wenet.paraformer.search import (gen_timestamps_from_peak,
                                     paraformer_greedy_search, all_paraformer_greedy_search, paraformer_beam_search)
from wenet.text.paraformer_tokenizer import ParaformerTokenizer


class Paraformer:

    def __init__(self, model_dir: str, resample_rate: int = 16000) -> None:

        model_path = os.path.join(model_dir, 'final.zip')
        units_path = os.path.join(model_dir, 'units.txt')
        self.model = torch.jit.load(model_path)
        self.resample_rate = resample_rate
        self.device = torch.device("cpu")
        self.tokenizer = ParaformerTokenizer(symbol_table=units_path)

    def compute_feats(self, audio_file: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
        waveform = waveform.to(torch.float)
        if sample_rate != self.resample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
        # NOTE (MengqingCao): complex dtype not supported in torch_npu.abs() now,
        # thus, delay placing data on NPU after the calculation of fbank.
        # revert me after complex dtype is supported.
        if "npu" not in self.device.__str__():
            waveform = waveform.to(self.device)
        feats = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            energy_floor=0.0,
                            sample_frequency=self.resample_rate)
        if "npu" in self.device.__str__():
            feats = feats.to(self.device)
        feats = feats.unsqueeze(0)
        print("feats", feats.size())
        return feats


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
            result['confidence'] = res.tokens_confidence
            print("res.tokens",res.token)
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
        frames = frames.cpu().numpy()
        cif_peaks = self.model.forward_cif_peaks(tp_alphas, token_num)

        results = all_paraformer_greedy_search(decoder_out, token_num, cif_peaks)
        # r = []
        # 通过all_greedy_search扒出来的字符及其置信度
        dict_100 = []

        # print("self.tokenizer.char_dict",self.tokenizer.char_dict)
        for (i, res) in enumerate(results):  # 遍历每个录音
            for j in range(len(res.tokens[0])):
                result = {}
                result['confidence'] = res.confidence
                tokens = [res.tokens[0][j], res.tokens[1][j], res.tokens[2][j]]
                # result['text'] = self.tokenizer.detokenize(tokens)[0]
                if tokens_info:
                    # tokens_info_l = []
                    times = gen_timestamps_from_peak(res.times,
                                                     num_frames=frames[i],
                                                     frame_rate=0.02)

                    for k, x in enumerate(tokens[:len(times)]):
                        # tokens_info_l.append({
                        #     'token':
                        #         self.tokenizer.char_dict[x],
                        #     'start':
                        #         round(times[k][0], 3),
                        #     'end':
                        #         round(times[k][1], 3),
                        #     'confidence':
                        #         round(res.tokens_confidence[k][j], 2)
                        # })
                        dict_100.append({
                            'token':
                                self.tokenizer.char_dict[x],
                            'confidence':
                                round(res.tokens_confidence[k][j], 20)
                        })
                        # result['tokens'] = tokens_info_l
                # r.append(result)
                yinsu_with_grade = {}
                for item in labels_dict:
                    text = labels_dict[item][0]
                    for entry in dict_100:
                        if entry['token'] == text:
                            yinsu_with_grade[item] = {
                                'text' : labels_dict[item][0],
                                'confidence' : entry['confidence']
                            }
                        else:
                            yinsu_with_grade[item] = {
                                'text': labels_dict[item][0],
                                'confidence': 0
                            }
                            break  # 找到匹配项后退出内层循环以提高效率
                new_dict = {
                    'yinsu_with_grade': yinsu_with_grade
                }
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
