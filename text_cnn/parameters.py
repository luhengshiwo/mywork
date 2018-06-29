class Para:
    embedding_size = 100
    unk = 'uuunnnkkk'
    pad = '<pad>'
    tgt_sos = '\<s>'
    tgt_eos = '\</s>'
    batch_size = 32
    num_units = 128
    max_gradient_norm = 50
    learning_rate = 0.001
    n_epochs = 30
    n_outputs = 15
    train_keep_prob = 0.5
    train_num = 523
    dev_num = 32
    test_num = 32
    threshold = 100.0
    l2_rate = 0.0001
    max_sentence = 64
    filter_sizes = [3,5,7]


