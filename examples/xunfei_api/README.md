# 📘 WeNet ASR API 使用文档

本模块通过 `wenet` 提供多种语音识别模型的统一接口，支持标准转写和带标签识别功能，适用于普通识别、语言学习、发音评估等场景。

## 🔧 模型加载接口

```python
model = wenet.load_model(name)
```

- **参数说明**：

  - `name` (`str`)：模型名称，可选值包括：
    - `"chinese"`：基础中文识别模型。
    - `"paraformer"`：结构化中文模型，识别效果较好。
    - `"XunfeiASR"`：科大讯飞集成模型，语音合成和识别兼顾。

- **返回值**：
  - 返回一个模型对象，支持后续的转写操作。

---

## 🎧 语音识别接口：`transcribe`

```python
result = model.transcribe(audio_path, tokens_info=True)
```

- **功能**：

  - 对语音进行识别并输出识别文本及详细 token 信息。

- **参数说明**：

  - `audio_path` (`str`)：音频文件路径（支持 `.wav` 格式）。
  - `tokens_info` (`bool`)：是否输出每个 token（字或音素）的时间（xunfei 不返回时间）和置信度信息。

- **返回结果结构**：
  ```python
  {
      'text': '识别结果文本',
      'confidence': 平均置信度 (float),
      'tokens': [
          {
              'token': '字/音素',
              'start': 开始时间 (秒),
              'end': 结束时间 (秒),
              'confidence': 单字置信度
          },
          ...
      ]
  }
  ```

### ✅ 示例输出（paraformer）：

```python
{
  'text': '博物馆',
  'confidence': [...],
  'tokens': [
    {'token': '博', 'start': 0.45, 'end': 0.73, 'confidence': 0.95},
    {'token': '物', 'start': 1.35, 'end': 1.63, 'confidence': 0.93},
    {'token': '馆', 'start': 2.53, 'end': 2.915, 'confidence': 0.95},
    {'token': '<eos>', 'start': 2.915, 'end': 3.24, 'confidence': 0.9}
  ]
}
```

---

## 🧪 带标签识别接口：`transcribe_with_labels`

```python
result = model.transcribe_with_labels(audio_path, labels_dict=...)
```

- **功能**：

  - 指定标签识别对应的发音或文本，用于**发音评估**或**特定关键词检测**。

- **参数说明**：

  - `audio_path` (`str`)：音频文件路径。
  - `labels_dict` (`dict`)：字母或标签与其可能发音或文字的映射，如：
    ```python
    {
      "d": ["的", "de"],
      "p": ["坡", "pe"]
    }
    ```

- **返回结果结构**：
  ```python
  {
    'yinsu_with_grade': {
      'd': {'text': '的/de/<unk>', 'confidence': float},
      'p': {'text': '坡/pe/<unk>', 'confidence': float}
    }
  }
  ```

### ✅ 示例输出（XunfeiASR）：

```python
{
  'yinsu_with_grade': {
    'd': {'text': '的', 'confidence': 0},
    'p': {'text': '坡', 'confidence': 0}
  }
}
```
