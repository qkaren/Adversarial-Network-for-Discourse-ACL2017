from keras.layers import Input, Embedding, Dense, merge, Dropout, Reshape, Lambda, \
    Activation,Flatten,Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Highway, BatchNormalization
import keras.backend as K
from keras.engine import Model
from keras.optimizers import Adagrad, SGD, Adam
import csv
import numpy as np
import pickle
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from math import isnan, log
import json

# Inspection on 01.04 and for all cases of binary/4way/multi-class (based on the data.pic)
np.random.seed(1337)

class Oracle():
    output_f = None

    @staticmethod
    def evaluate_multi(y_pred_labels, all_senses):
        # y_pred: list of labels, all_senses: list of list of labels
        assert len(y_pred_labels) == len(all_senses)
        count = 0
        for y, ys in zip(y_pred_labels, all_senses):
            if y in ys:
                count += 1
        return {"acc": (count+0.) / len(y_pred_labels)}

    @staticmethod
    def evaluate_cm(y_pred_labels, all_senses, num_class):
        # y_pred: list of labels, all_senses: list of list of labels (only using [0])
        assert len(y_pred_labels) == len(all_senses)
        ss = len(y_pred_labels)
        correct = 0
        cm = [{"tp":0, "fp":0, "fn":0} for i in range(num_class)]
        for y, ys in zip(y_pred_labels, all_senses):
            yt = ys[0]      # only the first one
            if y == yt:
                cm[y]["tp"] += 1
                correct += 1
            else:
                cm[y]["fp"] += 1
                cm[yt]["fn"] += 1
        # return the list of p/r/f1, [-1] will be MACRO average one
        ret = [{"p":0, "r":0, "f1":0, "acc":0} for i in range(num_class)]
        for i in range(num_class):
            ret[i]["p"] = cm[i]["tp"] / (cm[i]["tp"]+cm[i]["fp"]+0.00001)
            ret[i]["r"] = cm[i]["tp"] / (cm[i]["tp"]+cm[i]["fn"]+0.00001)
            ret[i]["f1"] = 2*ret[i]['p']*ret[i]['r'] / (ret[i]['p']+ret[i]['r']+0.00001)
            ret[i]["acc"] = (ss-cm[i]["fp"]-cm[i]["fn"]) / (ss+0.)
        ret.append({"p":np.average([t["p"] for t in ret]), "r":np.average([t["r"] for t in ret]),
                    "f1":np.average([t["f1"] for t in ret]), "acc":correct/(ss+0.)})
        return ret

    @staticmethod
    def count_correct_binary(y_pred, gold):
        # y_pred is batch-size*real, gold is one number 0/1
        num_all = len(y_pred)
        count = 0
        for y in y_pred:
            if y>=0.5:
                label = 1
            else:
                label = 0
            if label == gold:
                count += 1
        # print([y_pred, gold, count])
        return count

    @staticmethod
    def open_f(fname):
        Oracle.output_f = open(fname, 'a+')

    @staticmethod
    def close_f():
        if Oracle.output_f:
            Oracle.output_f.close()
        Oracle.output_f = None

    @staticmethod
    def print(s="\n", end='\n'):
        if s != "\n":
            s = str(s) + str(end)
        print(s, end="")
        try:
            if Oracle.output_f:
                Oracle.output_f.write(s)
        except:
            pass

# adding another layer of kmax-pooling
from keras.layers import Layer
from keras.engine.topology import InputSpec
class GlovalKMaxAveragePooling1D(Layer):
    def __init__(self, k, **kwargs):
        super(GlovalKMaxAveragePooling1D, self).__init__(**kwargs)
        self.kmax = k
        self.input_spec = [InputSpec(ndim=3)]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    def call(self, x, mask=None):
        # must using tensorflow
        x = tf.transpose(x, perm=[0,2,1])
        values, _ = tf.nn.top_k(x, k=self.kmax, sorted=False)
        values = tf.reduce_mean(values, reduction_indices=2)
        return values

class TrainModel:
    def __init__(self, word_WE, params, num_class):
        # firstly, the basic models and compiled models
        # - basic models
        self._block_names = ['cnn_ori', 'cnn_gen', 'cnn_discr', 'clf_ori', 'clf_gen', 'discr']
        self._blocks = {}    # could be alias (for the cnn/clf) or None (for cnn_discr)
        for n in self._block_names:
            self._blocks[n] = None
        # - compiled models for training and testing
        self._model_names = ['ori+clf', 'gen+clf', 'joint+clf', 'discr', 'ori+clf+discr', 'joint+clf+discr']
        self._models = {}
        self._num_class = num_class    # num of classes for the classifier

        # secondly, parameters
        # - general
        self._lr_default = 0.001     # default learning rate
        self._lrs_default = {'ori+clf+discr':0.001, 'joint+clf':0.001, 'discr':0.0001}
        self.batch_size = 64
        self.arg_maxlen = 80         # length of the input
        self.activation = 'tanh'
        self.dropout_embed = 0.1
        self.drop_conn = 0.4
        # - cnn for ori&gen and part for discr-cnn
        self.filter_diff = True
        self.filter_num = 400
        self.filter_lengths = [2, 3, 5]
        self.cnn_dense_num = 0
        self.cnn_dense_size = 300
        self.cnn_avgpool = False    # average pooling or max pooling (only for cnn1)
        self.kmax = 2               # kmax average pooling (only for cnn1)
        self.kmax0 = 1              # kmax pooling for cnn0
        # - discr
        self.discr_ydim = False
        self.discr_filter_num = 0   # 0 means nope, >0 means filter nums
        self.discr_dense_num = 1
        self.discr_dense_size = 300
        self.discr_dense_dropout = 0.4
        # - classifier
        self.cnn01_diff = True
        self.clf_diff = False       # different classifiers for ori and gen
        self.dense_num = 1
        self.dense_size = 300
        self.dense_dropout = 0.4    # dropout for the output of cnn and the denses
        self.low_rank = False       # low rank as the final output
        self.r = 8                  # rank in the low-rank mode
        # trainings (lrs, lambdas, epochs and strategies)
        self.train_clf_later = True
        # -- lambdas for later training
        self.lambda_confuse_binary = 0.    # lambda for the weight of freeze discr when training cnns
        self.lambda_confuse_fm = 0.
        self.lambda_direct_fm = 0.
        self.lambda_classify = 1.
        # -- lambdas for first joint training
        self.lambda_gen = 1.               # lambda for training cnn_gen
        self.lrs = {}
        for k in self._model_names:
            self.lrs[k] = self._lr_default
        for k in self._lrs_default:
            self.lrs[k] = self._lrs_default[k]
        self.epoch = 30
        self.epoch_firstjoint = 0
        self.epoch_firstdiscr = 0
        self.best_rec_epochs = 20   # only record after 20 epochs
        self.cnn_optimizer_name = 'Adagrad'
        self.D_optimizer_name = 'Adam'
        self._strategy_list = [None,None,None,self._fit_epoch_v3, self._fit_epoch_v4]
        self.strategy = 3
        self.kd = 2                 # K m-batches for D
        self.thresh_high = 0.95     # do not train D when > this-one
        self.thresh_low = 0.5       # do not confuse D when < this-one
        self.thresh_by_acc = False
        self.thresh_by_whole = False
        self.verbose = True
        self.shuffle = 0
        self.seed = 1337
        # other parameters
        self.lambda_D_aux = 0.  # aux output of D
        self.alpha = 1.             # label smoothing
        self.enhanceD = ""
        self.cnn_bn = False         # adding batch normal for output of cnn

        # thirdly, assign through the dictionary {params}
        self._title = ""
        for k in sorted(params.keys()):
            if isinstance(params[k], tuple) and len(params[k])==1:   # single choice mode
                params[k] = params[k][0]
            else:
                self._title += k+str(params[k])
        self._title = ''.join(self._title.split())
        self._param = params
        Oracle.print("-- Init with params %s, titled as %s." % (params, self._title))
        for k in params:
            if hasattr(self, k):
                if type(getattr(self, k)) == type(params[k]):
                    v = params[k]
                    if isinstance(v, dict):     # special partial assignment for dictionary
                        origin_d = getattr(self, k)
                        for k in v:
                            origin_d[k] = v[k]
                    else:
                        setattr(self, k, params[k])
                else:
                    Oracle.print("-- WARN: unmatched type for param %s: %s" % (k, params[k]))
            else:
                Oracle.print("-- WARN: unknown param %s: %s" % (k, params[k]))

        # additionally, specify the shared embedding
        self._embed_word = Embedding(input_dim=word_WE.shape[0],input_length=self.arg_maxlen,weights=[word_WE],
                             output_dim=word_WE.shape[1],trainable=False,mask_zero=False,dropout=self.dropout_embed)
        self._thresh_loss_small = -log(self.thresh_high)     # do not train D when loss < this-one
        self._thresh_loss_large = -log(self.thresh_low)      # do not confuse D when loss > this-one
        np.random.seed(self.seed)   # make sure repeatable

    def vprint(self, s):
        if self.verbose:
            Oracle.print(s)

    def get_title(self):       # full param description
        return self._title

    def get_opt(self, name):
        if name == 'Adagrad':
            return lambda lr: Adagrad(lr=lr, clipnorm=1.0)
        elif name == 'Adam':
            return lambda lr: Adam(lr=lr, beta_1=0.5, clipnorm=1.0)
        else:
            return lambda lr: SGD(lr=lr, clipnorm=1.0)

    @property
    def cnn_output_length(self):
        if self.cnn_dense_num > 0:
            return self.dense_size
        else:
            return self.filter_num*2*len(self.filter_lengths)

    # Basic building blocks
    def _build_cnn(self, filter_num, filter_lengths, cnn_dense_num, cnn_dense_size, cnn_kmax, cnn_avgpool):
        '''
        Build the first layer of model, from [arg1, arg2(plus)] to [repr]
        '''
        ''' input '''
        arg1_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg1_word')
        arg2_word_input = Input(shape=(self.arg_maxlen,), dtype='int32', name='arg2_word')
        ''' projection '''
        arg1_word = self._embed_word(arg1_word_input)
        arg2_word = self._embed_word(arg2_word_input)
        ''' word-level cnn + pooling'''
        arg1_cnns = [Convolution1D(nb_filter=filter_num, filter_length=i,
                                   border_mode='same', activation=self.activation) for i in filter_lengths]
        arg2_cnns = arg1_cnns
        if self.filter_diff:   # using different cnn for two args
            arg2_cnns = [Convolution1D(nb_filter=filter_num, filter_length=i,
                                       border_mode='same', activation=self.activation) for i in filter_lengths]
        arg1_cnn_outpus = [cnn(arg1_word) for cnn in arg1_cnns]
        arg2_cnn_outpus = [cnn(arg2_word) for cnn in arg2_cnns]
        arg1_cnn_merge = merge(arg1_cnn_outpus, mode='concat')
        arg2_cnn_merge = merge(arg2_cnn_outpus, mode='concat')
        pooling_part = GlobalMaxPooling1D()
        if cnn_kmax>1:
            pooling_part = GlovalKMaxAveragePooling1D(cnn_kmax)
        if cnn_avgpool:
            pooling_part = GlobalAveragePooling1D()
        arg1_word_mp = pooling_part(arg1_cnn_merge)
        arg2_word_mp = pooling_part(arg2_cnn_merge)
        ''' Output repr '''
        merged_vector = merge([arg1_word_mp, arg2_word_mp], mode='concat', concat_axis=-1)
        ''' Add another denses ? '''
        for i in range(cnn_dense_num):
            merged_vector = Dropout(self.dense_dropout)(merged_vector)      # no dropout for the output layer
            merged_vector = Dense(cnn_dense_size, activation=self.activation)(merged_vector)
        if self.cnn_bn:
            merged_vector = BatchNormalization()(merged_vector)
        input_list = [arg1_word_input, arg2_word_input]
        return Model(input=input_list, output=merged_vector)

    def _enhance_D(self, c):
        # adding another enhancing layer for D
        if self.enhanceD == "LSTM":
            l = int(c.get_shape()[-1])
            c = Reshape((1,l))(c)
            c = LSTM(l)(c)
        elif self.enhanceD == "HW":
            c = Highway()(c)
        return c

    def _build_discr(self):
        '''
        Build the last part of the discriminator, from [repr, [repr-self], [y-label]] to (binary-predict, last-layer)
        '''
        inputs = []
        reprs = []
        # Inputs
        # 1. main input
        inp = Input(shape=(self.cnn_output_length, ))
        inputs.append(inp)
        reprs.append(inp)
        # 2. sinput
        if self.discr_filter_num:   # accept the repr of discr-cnn, sentence input
            sinp = Input(shape=(self.discr_filter_num*2*len(self.filter_lengths), ))
            inputs.append(sinp)
            reprs.append(sinp)
        # 3. yinput
        if self.discr_ydim:   # accept the condition on label y (one-hot input)
            yinp = Input(shape=(self._num_class, ))
            inputs.append(yinp)
            reprs.append(yinp)
        # Merge
        if len(reprs)>1:
            c = merge(reprs, mode='concat')
        else:
            c = reprs[0]
        # Dense
        for i in range(self.discr_dense_num):
            c = Dense(self.discr_dense_size, activation=self.activation)(c)
            c = Dropout(self.discr_dense_dropout)(c)
            # Oracle.print("add %d times discr_dense..."%(i+1))
        # enhancement layer
        c = self._enhance_D(c)
        d_feature = c
        predictions = Dense(1, activation='sigmoid', name="output")(c)
        aux_pred = Dense(self._num_class, activation='softmax', name="output_aux")(c)
        model = Model(input=inputs, output=[predictions, d_feature, aux_pred])
        return model

    def _build_classifier(self):
        '''
        Build the last part of classifier, from [repr] to (multi-predict)
        '''
        inp = Input(shape=(self.cnn_output_length, ))
        c = inp
        for i in range(self.dense_num):
            c = Dense(self.dense_size, activation=self.activation)(c)
            c = Dropout(self.dense_dropout)(c)
            # Oracle.print("add %d times classifier_dense..."%(i+1))
        # whether using the low-rank output
        if self.low_rank:
            KK = self._num_class
            d1 = Dense(KK*self.r)(c)
            d2 = Dense(KK*self.r)(c)
            d1 = Reshape((KK,self.r))(d1)
            d2 = Reshape((KK,self.r))(d2)
            dense = merge([d1,d2], mode='mul')
            merged_vector = Lambda(lambda x: K.sum(x, axis=-1), output_shape=(KK,))(dense)
            predictions = Activation('softmax')(merged_vector)
        else:
            predictions = Dense(self._num_class, activation='softmax')(c)
        model = Model(input=inp, output=predictions)
        return model

    # The compiled modules needed (need to set trainable before them)
    # -- here only stacking models, no more parameters included
    # one cnn+classifier trainer
    def _build_cnn_classifier(self, block_cnn, block_classifier, lr):
        ''' For cnn_gen testing and training, for cnn_origin testing
            [arg1, arg2] to (multi-predict)
        '''
        block_cnn.trainable = True
        block_classifier.trainable = self.train_clf_later       # this is only for the single trainers in later mode, not for the early on joint trainer; bad design choice
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        repr = block_cnn([arg1, arg2])
        output = block_classifier(repr)
        model = Model(input=[arg1, arg2], output=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.get_opt(self.cnn_optimizer_name)(lr))
        return model

    # joint cnn_ori and cnn_gen trainer
    def _build_cnn_joint_trainer(self, block_cnn_ori, block_cnn_gen, block_clf_ori, block_clf_gen, lr):
        ''' Joint training for cnn0 and cnn1
            [arg1, arg2, arg2plus] to [multi-predict0, multi-predict1]
        '''
        block_cnn_ori.trainable = True
        block_cnn_gen.trainable = True
        block_clf_ori.trainable = True
        block_clf_gen.trainable = True
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')
        cnn_ori_repr = block_cnn_ori([arg1, arg2])
        cnn_gen_repr = block_cnn_gen([arg1, arg2plus])
        output_ori = block_clf_ori(cnn_ori_repr)
        output_gen = block_clf_gen(cnn_gen_repr)
        model = Model(input=[arg1, arg2, arg2plus], output=[output_ori, output_gen])
        # compile
        def loss_ori(y_true, y_pred):
            return (1 - self.lambda_gen) * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def loss_gen(y_true, y_pred):
            return self.lambda_gen * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        model.compile(loss=[loss_ori, loss_gen], optimizer=self.get_opt(self.cnn_optimizer_name)(lr))
        return model

    # training the discr
    def _build_cnn_discr(self, block_cnn_ori, block_cnn_gen, block_cnn_discr, block_discr, lr):
        ''' For discriminator training
            [arg1, arg2, arg2plus, [y]] to (binary-precdict)
        '''
        block_cnn_ori.trainable = False
        block_cnn_gen.trainable = False
        block_discr.trainable = True
        if self.discr_filter_num:
            block_cnn_discr.trainable = True
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')
        inputs = [arg1, arg2, arg2plus]
        discr_inputs_rest = []
        if self.discr_filter_num:
            repr_cnn = block_cnn_discr([arg1, arg2plus])    # use the max information
            discr_inputs_rest.append(repr_cnn)
        if self.discr_ydim:
            yinput = Input(shape=(self._num_class,))
            inputs.append(yinput)
            discr_inputs_rest.append(yinput)
        repr_ori = block_cnn_ori([arg1, arg2])
        repr_gen = block_cnn_gen([arg1, arg2plus])
        output_ori, _, aux_ori = block_discr([repr_ori] + discr_inputs_rest)
        output_gen, _, aux_gen = block_discr([repr_gen] + discr_inputs_rest)
        model = Model(input=inputs, output=[output_ori, output_gen, aux_ori, aux_gen])
        # compile
        def multi_ce(y_true, y_pred):
            return 0.5*self.lambda_D_aux * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def binary_ce(y_true, y_pred):
            return 0.5*K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        model.compile(loss=[binary_ce,binary_ce,multi_ce,multi_ce], optimizer=self.get_opt(self.D_optimizer_name)(lr))
        return model

    # train cnn_origin+ori_clf+(freeze)discr
    def _build_cnn_ori_clf_freezediscr(self, block_cnn_ori, block_clf_ori, block_cnn_gen, block_cnn_discr, block_discr, lr):
        ''' For cnn_origin and model_classifier training
            [arg1, arg2, arg2plus, [y]] to [multi-predict, binary-predict]
        '''
        block_cnn_ori.trainable = True
        block_cnn_gen.trainable = False     # fix cnn_gen
        block_clf_ori.trainable = self.train_clf_later
        if self.discr_filter_num:
            block_cnn_discr.trainable = False
        block_discr.trainable = False
        arg1 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2 = Input(shape=(self.arg_maxlen,), dtype='int32')
        arg2plus = Input(shape=(self.arg_maxlen,), dtype='int32')
        inputs = [arg1, arg2, arg2plus]
        discr_inputs_rest = []
        if self.discr_filter_num:
            repr_cnn = block_cnn_discr([arg1, arg2])
            discr_inputs_rest.append(repr_cnn)
        if self.discr_ydim:
            yinput = Input(shape=(self._num_class,))
            inputs.append(yinput)
            discr_inputs_rest.append(yinput)
        # from cnn_ori
        repr_ori = block_cnn_ori([arg1, arg2])
        output_multi = block_clf_ori(repr_ori)
        output_binary, ori_hidden, _ = block_discr([repr_ori] + discr_inputs_rest)
        # from cnn_gen
        repr_gen = block_cnn_gen([arg1, arg2plus])
        _, gen_hidden, _ = block_discr([repr_gen] + discr_inputs_rest)
        fm_loss = Lambda(lambda x: K.sum((x[0]-x[1])**2, axis=-1), output_shape=(1,))([ori_hidden, gen_hidden])
        dfm_loss = Lambda(lambda x: K.sum((x[0]-x[1])**2, axis=-1), output_shape=(1,))([repr_ori, repr_gen])
        model = Model(input=inputs, output=[output_multi, output_binary, fm_loss, dfm_loss])
        # compile
        def multi_crossentropy1(y_true, y_pred):
            return self.lambda_classify * K.mean(K.categorical_crossentropy(y_pred, y_true), axis=-1)
        def binary_crossentropy2(y_true, y_pred):       # y_pred should be all 1.0, otherwise nope
            return self.lambda_confuse_binary * K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
        def fm_lfunc(y_true, y_pred):
            return self.lambda_confuse_fm * K.mean(y_pred, axis=-1)
        def dfm_lfunc(y_true, y_pred):
            return self.lambda_direct_fm * K.mean(y_pred, axis=-1)
        model.compile(loss=[multi_crossentropy1, binary_crossentropy2, fm_lfunc, dfm_lfunc], optimizer=self.get_opt(self.cnn_optimizer_name)(lr))
        return model

    # train cnn_joint+clf+(freeze)discr
    def _build_cnn_joint_clf_freezediscr(self, block_cnn_ori, block_clf_ori, block_cnn_gen, block_clf_gen, block_cnn_discr, block_discr, lr):
        ''' For cnn_origin and model_classifier training
            [arg1, arg2, arg2plus, [y]] to [multi-predict0, multi-predict1, binary-predict]
        '''
        # no using it anymore
        return None

    # build them all
    def build_all_model(self):
        Oracle.print("Start to build them all.")
        blocks = self._blocks
        models = self._models
        lrs = self.lrs
        # basic blocks
        blocks['cnn_ori'] = self._build_cnn(self.filter_num, self.filter_lengths, self.cnn_dense_num, self.cnn_dense_size, self.kmax0, False)
        blocks['cnn_gen'] = blocks['cnn_ori']
        if self.cnn01_diff:
            blocks['cnn_gen'] = self._build_cnn(self.filter_num, self.filter_lengths, self.cnn_dense_num, self.cnn_dense_size, self.kmax, self.cnn_avgpool)
        if self.discr_filter_num:
            # no need for more denses (maybe)
            blocks['cnn_discr'] = self._build_cnn(self.discr_filter_num, self.filter_lengths, 0, 0, 1, False)
        blocks['clf_ori'] = self._build_classifier()
        blocks['clf_gen'] = blocks['clf_ori']
        if self.clf_diff:
            blocks['clf_gen'] = self._build_classifier()
        blocks['discr'] = self._build_discr()
        # compiled models for training and testing
        print(self.lrs)
        models['ori+clf'] = self._build_cnn_classifier(blocks['cnn_ori'], blocks['clf_ori'], lrs['ori+clf'])
        models['gen+clf'] = self._build_cnn_classifier(blocks['cnn_gen'], blocks['clf_gen'], lrs['gen+clf'])
        models['joint+clf'] = self._build_cnn_joint_trainer(blocks['cnn_ori'], blocks['cnn_gen'], blocks['clf_ori'], blocks['clf_gen'], lrs['joint+clf'])
        models['discr'] = self._build_cnn_discr(blocks['cnn_ori'], blocks['cnn_gen'], blocks['cnn_discr'], blocks['discr'], lrs['discr'])
        models['ori+clf+discr'] = self._build_cnn_ori_clf_freezediscr(blocks['cnn_ori'], blocks['clf_ori'], blocks['cnn_gen'], blocks['cnn_discr'], blocks['discr'], lrs['ori+clf+discr'])
        models['joint+clf+discr'] = None
        # for k in models:
        #     if models[k]:
        #         keras_plot(models[k], to_file="%s.png"%k)
        # for b in blocks:
        #     if blocks[b]:
        #         keras_plot(blocks[b], to_file="%s.png"%b)
        return

    def save_models(self, label):
        for k in self._blocks:
            m = self._blocks[k]
            if m:
                m.save_weights("./block_%s-%s.hdf5"%(k,label))

    def load_models(self, label):
        for k in self._blocks:
            m = self._blocks[k]
            if m:
                m.load_weights("./block_%s-%s.hdf5"%(k,label))

    # training and testing
    @staticmethod
    def _generate_batch(data, bs, shuffle, progbar):
        size = len(data['arg1'])
        # shuffle at first
        for i in range(shuffle):
            for cur in range(size):
                target = np.random.randint(cur, size)
                if target != cur:
                    for k in data:
                        tmp = data[k][target].copy()
                        data[k][target] = data[k][cur]
                        data[k][cur] = tmp
        nb_batch = (size+bs-1)//bs
        progress_bar = None
        if(progbar):
            progress_bar = Progbar(target=nb_batch)
        for index in range(nb_batch):
            if(progbar):
                progress_bar.update(index)
            begin, end = index*bs, min((index+1)*bs, size)
            cur_data = {}
            for k in data:
                cur_data[k] = data[k][begin:end]
            yield(cur_data)

    def _prepare_inputs(self, data_batched, add_arg2=1, add_arg2plus=0, drop_arg2plus=0., add_y=0):
        def _prepare_arg2plus(drop):
            if drop <= 0:
                return data_batched['arg2plus']
            size = len(data_batched['arg2'])
            assert size == len(data_batched['arg2plus'])
            ret = np.asarray([(data_batched['arg2'][i] if (np.random.random_sample()<drop) else data_batched['arg2plus'][i]) for i in range(size)])
            return ret
        # prepare the inputs for training or testing,
        # -- it will be [arg1, arg2, arg2plus(condition/drop-to-arg2), ylabel(condition)]
        inputs = []
        inputs.append(data_batched['arg1'])     # arg1 is always there
        if add_arg2:
            inputs.append(data_batched['arg2'])
        if add_arg2plus:
            inputs.append(_prepare_arg2plus(drop_arg2plus))
        if add_y:
            inputs.append(data_batched['sense'])
        return inputs

    def _fit_one(self, model_name, data_batched):
        # input data should be batched
        data = data_batched
        loss = None
        m = self._models[model_name]
        if model_name=='ori+clf':
            loss = m.train_on_batch(self._prepare_inputs(data), [data['sense']])
        elif model_name=='gen+clf':
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2=0,add_arg2plus=1,drop_arg2plus=self.drop_conn), [data['sense']])
        elif model_name=='joint+clf':
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1,drop_arg2plus=self.drop_conn), [data['sense'], data['sense']])
        elif model_name=='discr':
            y_0 = np.asarray([0. for i in data['arg1']])
            y_1 = np.asarray([self.alpha for i in data['arg1']])    # using alpha for label smoothing
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1,add_y=self.discr_ydim), [y_0, y_1, data['sense'], data['sense']])
        elif model_name=='ori+clf+discr':
            y_1 = np.asarray([self.alpha for i in data['arg1']])
            loss = m.train_on_batch(self._prepare_inputs(data,add_arg2plus=1,add_y=self.discr_ydim), [data['sense'], y_1, y_1, y_1])
        else:
            raise("Unkown model %s." % model_name)
        return loss

    def _test_one(self, model_name, data_all):
        def eval_classification(result_labels, all_senses, n):
            assert n in [2,4,11]
            result = None
            if n == 11:
                result = Oracle.evaluate_multi(result_labels, all_senses)
            elif self._num_class == 4:
                result = Oracle.evaluate_cm(result_labels, all_senses, n)[-1]
            else:
                result = Oracle.evaluate_cm(result_labels, all_senses, n)[1]
            return result
        TEST_BSIZE = 128
        ss = len(data_all['arg1'])
        m = self._models[model_name]
        if model_name=='discr':
            # test the discriminator
            ret = {"a0":0., "a1":0., "acc":0., "d0":0., "d1":0., "dloss":0.}
            result_labels_ori, result_labels_gen = [], []
            for data in self._generate_batch(data_all, TEST_BSIZE, 0, False):
                result = m.predict_on_batch(self._prepare_inputs(data,add_arg2plus=1,add_y=self.discr_ydim))
                result_labels_ori += [np.argmax(one, axis=-1) for one in result[2]]
                result_labels_gen += [np.argmax(one, axis=-1) for one in result[3]]
                ret["a0"] += Oracle.count_correct_binary(result[0], 0)
                ret["a1"] += Oracle.count_correct_binary(result[1], 1)
                ret["d0"] += np.sum(result[0])
                ret["d1"] += np.sum(result[1])
                ret["dloss"] += -1 * np.sum(np.log(1.0-result[0])+np.log(result[1]))/2
            for n in ret:
                ret[n] /= ss
            ret["acc"] = (ret["a0"]+ret["a1"]) / 2
            # aux outputs
            for r, prefix in zip([result_labels_ori, result_labels_gen], ["aux_ori_","aux_gen_"]):
                for n, v in eval_classification(r, data_all["sense_all"],  self._num_class).items():
                    ret[prefix+n] = v
            return ret
        elif model_name in ['ori+clf', 'gen+clf']:
            # test the classifier
            result_labels = []
            for data in self._generate_batch(data_all, TEST_BSIZE, 0, False):
                data_arg2 = {'ori+clf': data['arg2'], 'gen+clf': data['arg2plus']}[model_name]
                result = m.predict_on_batch([data['arg1'], data_arg2])
                result_labels += [np.argmax(one, axis=-1) for one in result]
            return eval_classification(result_labels, data_all["sense_all"],  self._num_class)
        else:
            raise("Not for test %s." % model_name)

    def _test_all(self, data):
        # test the four compiled model, return the target acc (origin classifier)
        ori = self._test_one('ori+clf', data)
        gen = self._test_one('gen+clf', data)
        dresult = self._test_one('discr', data)
        ret = {}
        for prefix, r in zip(["ori_", "gen_", "d_"], [ori, gen, dresult]):
            for n in r:
                ret[prefix+n] = r[n]
        ret["result"] = ret[{2:"ori_f1",4:"ori_f1",11:"ori_acc"}[self._num_class]]
        for n in sorted(list(set([s.split("_")[0] for s in ret.keys()]))):
            Oracle.print("--", end="")
            for s in sorted(ret.keys()):
                if s.startswith(n):
                    Oracle.print(" %s: %s"%(s, ret[s]), end=";")
            Oracle.print()
        return ret

    def _test_discr(self, data):
        # maybe data should be a sample of the whole data
        x = self._test_one('discr', data)
        acc, dloss = x['acc'], x['dloss']
        # at which phase (1:high, 0:center, -1:low)
        phase = -100
        if self.thresh_by_acc:
            if acc >= self.thresh_high:
                phase = 1
            elif acc > self.thresh_low:
                phase = 0
            else:
                phase = -1
        else:
            if dloss <= self._thresh_loss_small:
                phase = 1
            elif dloss < self._thresh_loss_large:
                phase = 0
            else:
                phase = -1
        self.vprint("res: %s, acc: %s, dloss: %s, phase: %s" % (x,acc,dloss,phase))
        # _special_test(data)     # to see the specific results
        return phase

    def _test_features(self, data, epoch):
        ori_repr = self._blocks['cnn_ori'].predict_on_batch([data['arg1'], data['arg2']])
        gen_repr = self._blocks['cnn_gen'].predict_on_batch([data['arg1'], data['arg2plus']])
        import os
        os.system("mkdir features")
        with open("features/feat-%s.pic"%epoch, "wb") as f:
            pickle.dump({'ori':ori_repr,'gen':gen_repr}, f)

    # fitting strategies
    def _fit_epoch_v3(self, epoch, train_data):     # just testing
        # Phase 1, train cnn0 and cnn1 for n epochs
        if epoch < self.epoch_firstjoint:
            Oracle.print('First Train cnn0 and cnn1.')
            for data in self._generate_batch(train_data, self.batch_size, self.shuffle, True):
                self._fit_one('joint+clf', data)
            # Oracle.print()
            # self._test_all(train_data)
        # Phase 2, train discr for m epochs
        elif epoch < self.epoch_firstjoint+self.epoch_firstdiscr:
            Oracle.print('First Train discr.')
            for data in self._generate_batch(train_data, self.batch_size, self.shuffle, True):
                self._fit_one('discr', data)
        else:
            Oracle.print('Train v3.')
            datas = [d for d in self._generate_batch(train_data, self.batch_size, 0, False)]
            numD = 0
            for data in self._generate_batch(train_data, self.batch_size, self.shuffle, True):
                # D
                for i in range(self.kd):
                    phase = self._test_discr(train_data if self.thresh_by_whole else datas[np.random.randint(0, len(datas))])
                    if phase != 1:      # only checking high threshold
                        data_sample = datas[np.random.randint(0, len(datas))]
                        self._fit_one('discr', data_sample)
                        numD += 1
                    else:
                        break
                # cnnD
                self._fit_one('ori+clf+discr', data)
            # self._test_all(train_data)
            Oracle.print()
            Oracle.print("Train them all with batch/D: %s/%s" % (len(datas), numD))

    def _fit_epoch_v4(self, epoch, train_data):
        Oracle.print('Only train cnn0.')
        for data in self._generate_batch(train_data, self.batch_size, self.shuffle, True):
            self._fit_one('ori+clf', data)

    def fit(self, train_data, dev_data, save_best=False):
        # use whole epoch for one model
        best_acc = {}
        best_epoch = {}
        history = {}
        data_dt = {'dev': dev_data}
        for n in ['dev']:
            best_acc[n] = 0.
            best_epoch[n] = -1
            history[n] = {}
        # fit it
        nan_flag = False
        for epoch in range(self.epoch):
            Oracle.print('Epoch {} of {}'.format(epoch + 1, self.epoch))
            # # print features for each one
            # self._test_features(dev_data, epoch)
            self._strategy_list[self.strategy](epoch, train_data)
            Oracle.print()
            # test them and save history
            for n in ['dev']:
                Oracle.print("Test on %s"%n)
                result = self._test_all(data_dt[n])
                # check NaN and append to history
                for r in result:
                    if isnan(result[r]):
                        nan_flag = True
                        result[r] = -1      # change to printable one
                    if r not in history[n]:
                        history[n][r] = []
                    history[n][r].append(result[r])
                # change
                acc_ori = result['result']     # must be there
                if epoch>=self.best_rec_epochs and acc_ori >= best_acc[n]:
                    best_acc[n] = acc_ori
                    best_epoch[n] = epoch
                    Oracle.print("--Get best %s--" % n)
                    if save_best:
                        self.save_models(n)
            # break if nan
            if nan_flag:
                Oracle.print("!! NaN, break it ...")
                break
            Oracle.print("----------------\n")
        Oracle.print(">>>> best of dev and test: %s of %s)" % (best_acc, best_epoch))
        return history, best_acc, best_epoch

# =============================================== #

def run_one(data, selections, indexes, output_prefix):
    # data is from pickle, params means the tuned params
    # set seed, setup tf and run
    results = []
    title = ""
    assert len(selections) == len(indexes)
    real_param = {}
    selected_param = {}
    for n, ind in zip(sorted(selections.keys()), indexes):     # must be sorted
        l = selections[n]
        if isinstance(l, tuple):
            if len(l) == 1:
                r = l
            else:       # what is really selected by random
                r = l[ind]
                selected_param[n] = r
                Oracle.print("-- %s: %s" % (n, r))
        else:
            r = (l, )
        real_param[n] = r
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        K.set_session(s)
        with tf.device("/gpu:0"):
            # model
            nclass = len(data['train_data']['sense'][0])    # find an example
            model = TrainModel(word_WE=data['word_WE'], params=real_param, num_class=nclass)
            model.build_all_model()
            title = model.get_title()   # get name for this one
            results += model.fit(data['train_data'], data['dev_data'])
            results.append(title)
    K.clear_session()
    # write records
    with open(output_prefix+'.csv', 'a+') as csvfile:
        try:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow([results[2]['dev'], results[1]['dev']] + [(str(k)+":"+str(selected_param[k])) for k in sorted(selected_param.keys())]+[title])
        except:
            Oracle.print("write result error!")
    with open(output_prefix+'.json', 'a+') as jsonfile:
        jsonfile.write(json.dumps(results)+"\n")

# ============================ #
# basic (default one)
# (~43.5): False(low-rank) 0.4(dense-dropout) 400(f-num) [2, 3, 5](f-length) True(f-diff) 1(dense-num) 300(dense-size)
setting0 = [{}, "setting0", 1]

# s1: on batch sizes and learning rate (to be continued...)
s1adamv3 = [
    {'strategy': (3,), 'lambda_classify':(1.0, ), 'lambda_confuse_binary':(0.1, 0.05), 'lambda_confuse_fm':(0., 0.05, 0.1),
     'lambda_gen':(0.5, ),  'drop_conn':(0.1, 0.3, 0.5), 'epoch_firstjoint':(3, ), 'thresh_by_whole':(False, ),
     'thresh_high':(0.99, 0.95, 0.9), 'thresh_by_acc': (True, ), 'kd': (5, 3, 1),
     'lrs':({'ori+clf+discr':0.001, 'joint+clf':0.001, 'discr':0.0001},), 'D_optimizer_name':("Adam"),
     'discr_dense_num':(2, 1), 'discr_dense_size':(1000, 800), 'discr_dense_dropout':(0.1, ), 'discr_ydim':(False, ),
     'cnn_avgpool': (False, True), 'best_rec_epochs':(10, ), 'verbose':(False,), 'batch_size':(128, 192, 256, 64)},
    "s1adamv3",
    -1
]

#=========================
# best-4way: 27,0.394352909923,29,0.455531692063,27,0.449771102558,dense_dropout:0.4,dense_num:1,dense_size:600,filter_diff:False,"filter_lengths:[2, 4, 5]",filter_num:800,"dense_dropout0.4dense_num1dense_size600filter_diffFalsefilter_lengths[2,4,5]filter_num800"
# best-lin: 14,0.46990291262135925,29,0.4451697127937337,14,0.42950391644908614,dense_dropout:0.3,dense_num:1,dense_size:300,filter_diff:False,"filter_lengths:[2, 4, 5]",filter_num:600,"dense_dropout0.3dense_num1dense_size300filter_diffFalsefilter_lengths[2,4,5]filter_num600"
# best-expansion: 23,0.6918517475888192,23,0.7188400553962218,23,0.7188400553962218,dense_dropout:0.3,dense_num:2,dense_size:300,filter_diff:True,"filter_lengths:[2, 3, 4]",filter_num:400,"dense_dropout0.3dense_num2dense_size300filter_diffTruefilter_lengths[2,3,4]filter_num400"

best_cnn_4way = [  # around 45
    {'strategy': (4, ), 'best_rec_epochs':(10, ), 'batch_size':(128, ), 'shuffle':(0,),
     'filter_num':(800,), 'filter_lengths': ([2,4,5],), 'filter_diff':(False,),
     'dense_num':(1,), 'dense_size':(600,), 'dense_dropout':(0.4,)},
    "best_cnn_4way",
    10
]
best_cnn_lin = [  # around 44
    {'strategy': (4, ), 'best_rec_epochs':(10, ), 'batch_size':(128, ), 'shuffle':(0,),
     'filter_num':(600,), 'filter_lengths': ([2,4,5],), 'filter_diff':(False,),
     'dense_num':(1,), 'dense_size':(300,), 'dense_dropout':(0.3,)},
    "best_cnn_lin",
    10
]
best_cnn_expansion = [  # around 71
    {'strategy': (4, ), 'best_rec_epochs':(10, ), 'batch_size':(128, ), 'shuffle':(3,),
     'filter_num':(400,), 'filter_lengths': ([2,3,4],), 'filter_diff':(True,),
     'dense_num':(2,), 'dense_size':(300,), 'dense_dropout':(0.3,)},
    "best_expansion_4way",
    10
]

import sys
import os
def main():
    def _select(l, name=""):
        ind = 0
        if isinstance(l, tuple):
            if len(l) == 1:
                ind = 0
            else:       # what is really selected by random
                ind = np.random.randint(0, len(l))
        return ind
    try:
        _which_gpu = int(os.environ['RGPU'])
    except:
        _which_gpu = 1
    ss = globals()[sys.argv[1]]
    run_dict, run_name, run_max_time = ss
    if len(sys.argv)==2 and len(run_dict)!=0:
        # loop mode
        while run_max_time != 0:
            indexes = []
            np.random.seed(run_max_time+12345)
            for n in sorted(run_dict.keys()):
                one = _select(run_dict[n], n)
                indexes.append(str(one))
            cmd = "CUDA_VISIBLE_DEVICES=%s python3 %s %s %s" % (_which_gpu, sys.argv[0], sys.argv[1], " ".join(indexes))
            if os.system(cmd) != 0:
                break
            run_max_time -= 1
    else:   # run one
        data = None
        for fn in ["../data.pic", "data.pic"]:
            try:
                with open(fn, "rb") as f:
                    data = pickle.load(f)
            except:
                pass
        run_indexes = [int(i) for i in sys.argv[2:]]
        Oracle.open_f(run_name+'.log')
        Oracle.print(" ".join(sys.argv))
        run_one(data, run_dict, run_indexes, run_name)
        Oracle.close_f()

if __name__ == "__main__":
    main()
