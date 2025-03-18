import wenet

# model = wenet.load_model('chinese')
# model = wenet.load_model('paraformer')
# model = wenet.load_model('XunfeiASR')
# result = model.transcribe(r'D:\audio\47_112.225.101.110_20241119074051_博物馆.wav', tokens_info=True)
# print(result)
for name in ['chinese', 'paraformer', 'XunfeiASR']:
    model = wenet.load_model(name)
    result = model.transcribe_with_labels(r'D:\audio\47_112.225.101.110_20241119074051_博物馆.wav',
                                      labels_dict={"b": ["bo", "b"], "p": ["guan", "馆"], "l": ["物", "lia"]})
    print(name, result)
