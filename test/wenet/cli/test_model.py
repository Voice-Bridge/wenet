#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-12] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
from types import SimpleNamespace

import requests
import pytest
import torch

from wenet.cli.hub import download
from wenet.cli.model import Model
import wenet.cli.model as model_module


@pytest.mark.parametrize("model", [
    "aishell_u2pp_conformer_libtorch.tar.gz",
    "aishell2_u2pp_conformer_libtorch.tar.gz",
    "gigaspeech_u2pp_conformer_libtorch.tar.gz",
    "librispeech_u2pp_conformer_libtorch.tar.gz",
    "multi_cn_unified_conformer_libtorch.tar.gz",
    "wenetspeech_u2pp_conformer_libtorch.tar.gz"
])
def test_model(model):
    dest = model.split('.')[0]  # aishell_u2pp_conformer_libtorch
    dataset = model.split('_')[0]  # aishell
    if not os.path.exists(dest):
        os.makedirs(dest)
    response = requests.get(
        "https://modelscope.cn/api/v1/datasets/wenet/wenet_pretrained_models/oss/tree"  # noqa
    )
    model_info = next(data for data in response.json()["Data"]
                      if data["Key"] == model)
    model_url = model_info['Url']
    download(model_url, dest=dest, only_child=True)
    model = Model(dest, gpu=-1, beam=5, resample_rate=16000)
    if dataset in ['gigaspeech', 'librispeech']:
        audio_file = "test/resources/librispeech-1995-1837-0001.wav"
        text = "▁IT▁WAS▁THE▁FIRST▁GREAT▁SORROW▁OF▁HIS▁LIFE▁IT▁WAS▁NOT▁SO▁MUCH" + \
            "▁THE▁LOSS▁OF▁THE▁COTTON▁ITSELF▁BUT▁THE▁FANTASY▁THE▁HOPES▁THE▁DREAMS▁BUILT▁AROUND▁IT"
    else:
        audio_file = "test/resources/aishell-BAC009S0724W0121.wav"
        text = "广州市房地产中介协会分析"
    result = model.transcribe(audio_file)
    print(result)
    assert result['text'] == text


def test_transcribe_with_label_runs_without_grad_and_keeps_shape(monkeypatch):
    model = Model.__new__(Model)
    model.device = torch.device("cpu")

    grad_enabled = {}

    def fake_compute_feats(audio_file):
        grad_enabled["compute_feats"] = torch.is_grad_enabled()
        return torch.zeros(1, 2, 3)

    class DummyTorchModel:

        def forward_encoder_chunk(self, feats, offset, required_cache_size):
            grad_enabled["forward_encoder_chunk"] = torch.is_grad_enabled()
            return torch.zeros(1, 2, 4), None, None

    def fake_attention_rescoring(model, ctc_prefix_results, encoder_out,
                                 encoder_lens, ctc_weight, reverse_weight):
        grad_enabled.setdefault("attention_rescoring", []).append(
            torch.is_grad_enabled())
        token_count = len(ctc_prefix_results[0].tokens)
        return [
            SimpleNamespace(tokens_confidence=[0.9] * token_count,
                            confidence=0.9)
        ]

    monkeypatch.setattr(model, "compute_feats", fake_compute_feats)
    monkeypatch.setattr(model, "tokenize", lambda label: [1, 2])
    monkeypatch.setattr(model_module, "attention_rescoring",
                        fake_attention_rescoring)
    model.model = DummyTorchModel()

    result = model.transcribe_with_label("dummy.wav", "中文",
                                         ["zhongwen", "zhong_wen"])

    assert grad_enabled == {
        "compute_feats": False,
        "forward_encoder_chunk": False,
        "attention_rescoring": [False, False],
    }
    assert result == [{"中": 0.9}, {"文": 0.9}]
