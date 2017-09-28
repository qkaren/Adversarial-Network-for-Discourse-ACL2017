from gensim import models
import numpy as np
import json
from keras.preprocessing import sequence
from keras.utils import np_utils
import pickle

Sense_To_Label = {
                      'Expansion.Conjunction': 6,
                      'Expansion.Instantiation': 7,
                      'Comparison.Concession': 5,
                      'Contingency.Cause': 2,
                      'Expansion.List': 10,
                      'Expansion.Alternative': 9,
                      'Temporal.Asynchronous': 0,
                      'Temporal.Synchrony': 1,
                      'Expansion.Restatement': 8,
                      'Comparison.Contrast': 4,
                      'Contingency.Pragmatic cause': 3
                  }
Rare_Indicator = -1 # indicate the one other than the 11 classes
Conn_Token = "CONN"
np.random.seed(12345)

def get_dict(fn):
    x = [json.loads(l) for l in open(fn)]
    ix = [i for i in x if i["Type"]=="Implicit"]
    cx = [x["Connective"]["RawText"][0] for x in ix]
    return sorted(list(set(cx)))

conn_dict = ['accordingly', 'additionally', 'after', 'afterwards', 'also', 'although', 'and', 'as', 'as a consequence', 'as a matter of fact', 'as a result', 'as it turns out', 'at that time', 'at the same time', 'at the time', 'because', 'before', 'besides', 'but', 'by comparison', 'by contrast', 'consequently', 'earlier', 'even though', 'eventually', 'ever since', 'finally', 'first', 'for', 'for example', 'for instance', 'for one', 'for one thing', 'further', 'furthermore', 'hence', 'however', 'in addition', 'in comparison', 'in contrast', 'in fact', 'in other words', 'in particular', 'in response', 'in return', 'in short', 'in sum', 'in summary', 'in the end', 'in the meantime', 'in turn', 'inasmuch as', 'incidentally', 'indeed', 'insofar as', 'instead', 'later', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'next', 'nonetheless', 'now', 'on the contrary', 'on the one hand', 'on the other hand', 'on the whole', 'or', 'overall', 'particularly', 'plus', 'previously', 'rather', 'regardless', 'second', 'separately', 'similarly', 'simultaneously', 'since', 'since then', 'so', 'so far', 'so that', 'soon', 'specifically', 'still', 'subsequently', 'that is', 'then', 'thereafter', 'therefore', 'third', 'though', 'thus', 'to this end', 'ultimately', "what's more", 'when', 'whereas', 'while', 'yet']

def build_vocab(fileName, freq):

    all_vocab_list = []
    vocab_freq = {}
    all_pos_list = []
    vocab_set = set()
    pos_set = set()

    with open(fileName) as fo:
        relation = [json.loads(x) for x in fo]
    for r in relation:
        all_vocab_list += (r["Arg1"]["Word"])
        all_vocab_list += (r["Arg2"]["Word"])
        if len(r["Connective"]["RawText"]) > 0:
            all_vocab_list += (r["Connective"]["RawText"][0].split())
        all_pos_list += (r["Arg1"]["POS"])
        all_pos_list += (r["Arg2"]["POS"])

    for w in all_vocab_list:
        if w not in vocab_freq:
            vocab_freq[w] = 1
        else:
            vocab_freq[w] += 1

    for item in vocab_freq.keys():
        if vocab_freq[item] <= freq:
            continue
        vocab_set.add(item)

    for pos in all_pos_list:
        pos_set.add(pos)

    return sorted(list(vocab_set)), sorted(list(pos_set))

def build_WE(vocab_list, pos_list, pretrained_file, init_range, word_ndims=300, pos_ndims=50):
    word_WE = np.zeros((45000, word_ndims), dtype='float32')
    pos_WE = np.zeros((100, pos_ndims), dtype='float32')
    w2i_dic = {}
    p2i_dic = {}

    w2vec = {}
    if pretrained_file is not None:
        w2vec = models.Word2Vec.load_word2vec_format(pretrained_file, binary=True)
    index = 2  # word index start from 2, unknown is 1
    for w in vocab_list:
        w2i_dic[w] = index
        index += 1

    word_WE[1, :] = np.array(np.random.uniform(-init_range / word_ndims, init_range / word_ndims, (word_ndims,)),dtype='float32')  # hyperparameter
    for x in vocab_list:
        if x in w2vec:
            word_WE[w2i_dic[x], :] = w2vec[x]
        else:
            word_WE[w2i_dic[x], :] = np.array(np.random.uniform(-init_range / word_ndims, init_range / word_ndims, (word_ndims,)),dtype='float32')  # hyperparameter

    p2i_dic[Conn_Token] = 1   # special token
    for i, y in enumerate(pos_list, start=2):
        p2i_dic[y] = i
        pos_WE[i, :] = np.array(np.random.uniform(-init_range / pos_ndims, init_range / pos_ndims, (pos_ndims,)), dtype='float32')

    return w2i_dic, p2i_dic, word_WE, pos_WE


def process(fileName, w2i_dic, p2i_dic, iftrain, arg_len=80):
    def _arg_process(file):
        fo = open(file)
        relation = [json.loads(x) for x in fo]
        fo.close()
        data = []
        for r in relation:
            if r["Type"] != "Implicit":
                continue
            temp = {}
            temp["Arg1"] = r['Arg1']['Word']
            temp["Arg2"] = r['Arg2']['Word']
            temp["Senses"] = []
            for s in r["Sense"]:
                if s in Sense_To_Label:
                    temp["Senses"].append(Sense_To_Label[s])
                else:
                    temp["Senses"].append(Rare_Indicator)
            temp["Sense"] = r["Sense"]
            temp["Conn"] = r["Connective"]["RawText"][0].split()
            temp["Arg2plus"] = temp["Conn"] + temp["Arg2"]
            temp["POS1"] = r["Arg1"]["POS"]
            temp["POS2"] = r["Arg2"]["POS"]
            temp["POS2plus"] = [Conn_Token for one in temp["Conn"]] + temp["POS2"]    # special token 1
            temp["Conn_index"] = conn_dict.index(r["Connective"]["RawText"][0])
            data.append(temp)
        return data
    # start
    data = _arg_process(fileName)
    arg1 = []
    arg2 = []
    arg2plus = []
    pos1 = []
    pos2 = []
    pos2plus = []
    sense = []
    senses_all = []
    conn_index = []
    for x in data:
        if iftrain:
            for s in x["Senses"]:
                if s != Rare_Indicator:
                    arg1.append(x["Arg1"])
                    arg2.append(x["Arg2"])
                    arg2plus.append(x["Arg2plus"])
                    pos1.append(x["POS1"])
                    pos2.append(x["POS2"])
                    pos2plus.append(x["POS2plus"])
                    sense.append(s)
                    senses_all.append(x["Senses"])  # will not use
                    conn_index.append(x["Conn_index"])
        else:
            arg1.append(x["Arg1"])
            arg2.append(x["Arg2"])
            arg2plus.append(x["Arg2plus"])
            pos1.append(x["POS1"])
            pos2.append(x["POS2"])
            pos2plus.append(x["POS2plus"])
            sense.append(x["Senses"][0])    # will not use
            senses_all.append(x["Senses"])
            conn_index.append(x["Conn_index"])

    "arg words -> word index -> get word_docs"
    arg1_word = [[(w2i_dic[i] if i in w2i_dic else 1) for i in ones]for ones in arg1]
    arg2_word = [[(w2i_dic[i] if i in w2i_dic else 1) for i in ones]for ones in arg2]
    arg2plus_word = [[(w2i_dic[i] if i in w2i_dic else 1) for i in ones]for ones in arg2plus]
    arg1_pos = [[p2i_dic[i] for i in ones] for ones in pos1]
    arg2_pos = [[p2i_dic[i] for i in ones] for ones in pos2]
    arg2plus_pos = [[p2i_dic[i] for i in ones] for ones in pos2plus]

    # padding for sentences
    X_word_1 = sequence.pad_sequences(arg1_word, maxlen=arg_len, padding='pre', truncating='pre')
    X_word_2 = sequence.pad_sequences(arg2_word, maxlen=arg_len, padding='post', truncating='post')
    X_wordplus_2 = sequence.pad_sequences(arg2plus_word, maxlen=arg_len, padding='post', truncating='post')
    X_pos_1 = sequence.pad_sequences(arg1_pos, maxlen=arg_len, padding='pre', truncating='pre')
    X_pos_2 = sequence.pad_sequences(arg2_pos, maxlen=arg_len, padding='post', truncating='post')
    X_posplus_2 = sequence.pad_sequences(arg2plus_pos, maxlen=arg_len, padding='post', truncating='post')
    y = np_utils.to_categorical(np.array(sense))
    ci = np_utils.to_categorical(conn_index,nb_classes=len(conn_dict))
    return {'arg1':X_word_1, 'arg2':X_word_2, 'arg2plus':X_wordplus_2,
            'pos1':X_pos_1, 'pos2':X_pos_2, 'pos2plus':X_posplus_2,
            'sense':y, 'sense_all':senses_all, 'conn':ci}

def write():
    # arguments
    freq = 0
    init_range = 0.5
    # process
    file_prefix = "../mine/pdtb_data/"
    train_file, dev_file, test_file = "train_pdtb.json","dev_pdtb.json","test_pdtb.json"
    print("1. build vocab")
    vocab_list, pos_list = build_vocab(file_prefix+train_file, freq=freq)
    embed_file = file_prefix+'../GoogleNews-vectors-negative300.bin'
    print("2. build WE")
    w2i_dic, p2i_dic, word_WE, pos_WE = build_WE(vocab_list, pos_list, embed_file, init_range)
    print("3. process data")
    train_data = process(file_prefix+train_file, w2i_dic, p2i_dic, True)
    dev_data = process(file_prefix+dev_file, w2i_dic, p2i_dic, False)
    test_data = process(file_prefix+test_file, w2i_dic, p2i_dic, False)
    # write
    label = "f%s-r%s-w%s-p%s" % (freq, init_range, len(w2i_dic), len(p2i_dic))
    print("4. write file %s"%label)
    with open("data_%s.pic" % label, "wb") as f:
        pickle.dump({'w2i_dic':w2i_dic, 'p2i_dic':p2i_dic, 'word_WE':word_WE, 'pos_WE':pos_WE,
             'train_data':train_data, 'dev_data':dev_data, 'test_data':test_data}, f)

def fetch():
    with open("data.pic", "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    write()
