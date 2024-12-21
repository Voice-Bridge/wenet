import wenet

model = wenet.load_model('xunfeiishere')
result = model.transcribe('D://sample_audios//15_111.193.224.161_20241115141902_面团.wav')
print(result)