import wenet

# model = wenet.load_model('chinese')
# model = wenet.load_model('paraformer')
# model = wenet.load_model('XunfeiASR')
# result = model.transcribe(r'D:\sample_audio\selected_files\6_223.104.41.50_20241213040033_丢俩.wav', tokens_info=True)

for name in ['chinese', 'paraformer', 'XunfeiASR']:
    model = wenet.load_model(name)
    print(f"#####{name} transcribe#####")
    result = model.transcribe('./47_112.225.101.110_20241119074051_博物馆.wav', tokens_info=True)
    print(result)
    print(f"#####{name} transcribe_with_labels#####")
    result = model.transcribe_with_labels('./47_112.225.101.110_20241119074051_博物馆.wav',
                                      labels_dict={"d": ["的", "de"], "p": ["坡", "pe"]})
    print(name, result)
