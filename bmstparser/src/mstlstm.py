from dynet import *
from utils import read_conll, write_conll, stream_to_batch
from operator import itemgetter
import utils, time, random, decoder
import numpy as np

random.seed(1)
renew_cg()

class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = Model()
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels


        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        if self.bibiFlag:
            self.builders = [VanillaLSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [VanillaLSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             VanillaLSTMBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.wdims + self.pdims + self.edim, self.ldims, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidBias = self.model.add_parameters((self.hidden_units))

            self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

            self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = self.model.add_parameters((len(self.irels)))


    def  __getExpr(self, s_i, s_j, train, hidbias, hid2bias, hid2layer, outlayer, act):
        
        if s_i.headfov is None:
            s_i.headfov = self.hidLayerFOH.expr() * concatenate([s_i.lstms[0], s_i.lstms[1]])
        if s_j.modfov is None:
            s_j.modfov  = self.hidLayerFOM.expr() * concatenate([s_j.lstms[0], s_j.lstms[1]])

        if self.hidden2_units > 0:
            output = outlayer * act(hid2bias + hid2layer * act(s_i.headfov + s_j.modfov + hidbias)) # + self.outBias
        else:
            output = outlayer * act(s_i.headfov + s_j.modfov + hidbias) # + self.outBias

        return output


    def __evaluate(self, sentence, train):
        ge = self.__getExpr
        hidbias = self.hidBias.expr()
        hid2bias = self.hid2Bias.expr()
        hid2layer = self.hid2Layer.expr()
        outlayer = self.outLayer.expr()
        act = self.activation
        exprs = [ [ge(s_i, s_j, train, hidbias, hid2bias, hid2layer, outlayer, act) for s_j in sentence] for s_i in sentence ]

        return exprs


    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)


    def Predict(self, conll_path, BATCH_SIZE=1):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence_batch in enumerate(stream_to_batch(read_conll(conllFP), BATCH_SIZE)):

                batch_exprs = []
                sents = []
                labels = []
                for sentence in sentence_batch:
                    conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                    for entry in conll_sentence:
                        wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.wdims > 0 else None
                        posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                        evec = self.elookup[int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)))] if self.external_embedding is not None else None
                        entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                        entry.lstms = [entry.vec, entry.vec]
                        entry.headfov = None
                        entry.modfov = None

                        entry.rheadfov = None
                        entry.rmodfov = None

                    if self.blstmFlag:
                        lstm_forward = self.builders[0].initial_state()
                        lstm_backward = self.builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = lstm_forward.add_input(entry.vec)
                            lstm_backward = lstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = lstm_forward.output()
                            rentry.lstms[0] = lstm_backward.output()

                        if self.bibiFlag:
                            for entry in conll_sentence:
                                entry.vec = concatenate(entry.lstms)

                            blstm_forward = self.bbuilders[0].initial_state()
                            blstm_backward = self.bbuilders[1].initial_state()

                            for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                                blstm_forward = blstm_forward.add_input(entry.vec)
                                blstm_backward = blstm_backward.add_input(rentry.vec)

                                entry.lstms[1] = blstm_forward.output()
                                rentry.lstms[0] = blstm_backward.output()
                    batch_exprs.append(self.__evaluate(conll_sentence, True))
                    sents.append(conll_sentence)


                _s=time.time()
                forward(batch_exprs[-1][-1] )
                print "fw1:",time.time()-_s
                batch_heads = []
                _s=time.time()
                for _i, (exprs, conll_sentence) in enumerate(zip(batch_exprs, sents)):
                    scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
                    heads = decoder.parse_proj(scores)

                    for entry, head in zip(conll_sentence, heads):
                        entry.pred_parent_id = head
                        entry.pred_relation = '_'
                    batch_heads.append(heads)
                    dump = False
                print "decode:",time.time()-_s

                if self.labelsFlag: # TODO this is currently not batched..
                    labels = []
                    _exps = []
                    for (heads, conll_sentence) in zip(batch_heads, sents):
                        labels_exprs = []
                        for modifier, head in enumerate(heads[1:]):
                            exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                            _exps.append(exprs)
                            labels_exprs.append((head,modifier,exprs))
                        labels.append(labels_exprs)
                    

                    _s=time.time()
                    forward(_exps)
                    print "fw-L:",time.time()-_s
                    for lbls,conll_sentence in zip(labels, sents):
                        for (head, modifier, exprs) in lbls:
                            scores = exprs.value()
                            conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    for sentence in sentence_batch: yield sentence

    def Train(self, conll_path, BATCH_SIZE=1):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            eeloss = 0.0

            for iSentence, sentence_batch in enumerate(stream_to_batch(shuffledData, BATCH_SIZE)):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start, (100*BATCH_SIZE)/(time.time()-start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                    ltotal = 0

                batch_exprs = []
                sents = []
                golds = []
                labels = []
                for sentence in sentence_batch:
                    conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                    sents.append(conll_sentence)

                    gold = [entry.parent_id for entry in conll_sentence]
                    golds.append(gold)

                    # initialize sentence
                    for entry in conll_sentence:
                        c = float(self.wordsCount.get(entry.norm, 0))
                        dropFlag = (random.random() < (c/(0.25+c)))
                        wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
                        posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                        evec = None

                        if self.external_embedding is not None:
                            evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                        entry.vec = concatenate(filter(None, [wordvec, posvec, evec]))

                        entry.lstms = [entry.vec, entry.vec]
                        entry.headfov = None
                        entry.modfov = None

                        entry.rheadfov = None
                        entry.rmodfov = None

                    # bilstm encode
                    if self.blstmFlag:
                        lstm_forward = self.builders[0].initial_state()
                        lstm_backward = self.builders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            lstm_forward = lstm_forward.add_input(entry.vec)
                            lstm_backward = lstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = lstm_forward.output()
                            rentry.lstms[0] = lstm_backward.output()
                        
                        if self.bibiFlag:
                            for entry in conll_sentence:
                                entry.vec = concatenate(entry.lstms)

                            blstm_forward = self.bbuilders[0].initial_state()
                            blstm_backward = self.bbuilders[1].initial_state()

                            for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                                blstm_forward = blstm_forward.add_input(entry.vec)
                                blstm_backward = blstm_backward.add_input(rentry.vec)

                                entry.lstms[1] = blstm_forward.output()
                                rentry.lstms[0] = blstm_backward.output()

                    # compute all arc score-expressions
                    batch_exprs.append(self.__evaluate(conll_sentence, True))

                    # labeling?
                    _exps = []
                    if self.labelsFlag:
                        labels_exprs = []
                        for modifier, head in enumerate(gold[1:]):
                            rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                            labels_exprs.append((rexprs, head, modifier))
                            _exps.append(rexprs)
                        labels.append(labels_exprs)


                # now do the actual scoring
                _s = time.time()
                forward(batch_exprs[-1][-1] + _exps)
                print "fw1t:",time.time()-_s
                for _i, (exprs, conll_sentence) in enumerate(zip(batch_exprs, sents)):
                    scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
                    gold = golds[_i]
                    heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

                    # TODO labeling is inot batched
                    if self.labelsFlag:
                        for rexprs, head, modifier in labels[_i]:
                            rscores = rexprs.value()
                            goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                            wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                            if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                                lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

                    e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                    eerrors += e
                    if e > 0:
                        loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                        eloss += (e)
                        mloss += (e)
                        errs.extend(loss)

                    etotal += len(conll_sentence)

                    if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                        eeloss = 0.0

                if len(errs) > 0 or len(lerrs) > 0:
                    eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                    _s = time.time()
                    eerrs.scalar_value()
                    print "fw2t",time.time()-_s
                    _s = time.time()
                    eerrs.backward()
                    print "bw2t",time.time()-_s
                    self.trainer.update()
                    errs = []
                    lerrs = []

                renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()

        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
