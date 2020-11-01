import tensorflow as tf
import data_loader_recsys
import generator_recsys_cau
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse
import sys




def generatesubsequence(train_set,padtoken):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0

        copyseq = list(seq)
        padcount = copyseq.count(padtoken)  # the number of padding elements
        copyseq = copyseq[padcount:]  # the remaining elements
        lenseq_nopad = len(copyseq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        if (lenseq_nopad - 4) < 1:
            subseqtrain.append(seq)
            continue

        for j in range(lenseq_nopad - 4):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [padtoken] * j

            subseq = list(subseqbeg) + list(subseqend)

            # subseq= np.append(subseqbeg,subseqbeg)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)

    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train

def GetParser(parser):
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    #history_sequences_20181014_fajie_smalltest.csv
    parser.add_argument('--datapath', type=str, default='Data/Session/history_sequences_20181014_fajie_transfer_pretrain_small.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=10,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=10,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.5,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--padtoken', type=str, default='0',
                        help='is the padding token in the beggining of the sequence')
    return parser

def ShowTrainInfo(loss, iter_num, batch_no, numIters, total_batches):
    return "LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss, iter_num, batch_no, numIters, total_batches)

def main(model_path = ''):
    parser = argparse.ArgumentParser()
    parser = GetParser(parser)
    args = parser.parse_args()

    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.item
    items = dl.item_dict

    if args.padtoken in items:
        padtoken = items[args.padtoken] 
    else:
        padtoken = len(items) + 1

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        train_set = generatesubsequence(train_set,padtoken)

    model_para = {
        'item_size': len(items),
        'dilated_channels': 64,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':2,
        'iterations':400,
        'is_negsample':True #False denotes using full softmax
    }

    itemrec = generator_recsys_cau.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'],
                                       beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(model_para['is_negsample'],reuse=True)

    tf.add_to_collection("dilate_input", itemrec.dilate_input)
    tf.add_to_collection("context_embedding", itemrec.context_embedding)

    sess = tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    numIters = 1
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            start = time.time()
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            _, loss, results = sess.run([optimizer, itemrec.loss, itemrec.arg_max_prediction],
                feed_dict={itemrec.itemseq_input: item_batch})
            end = time.time()
            if numIters % args.eval_iter == 0:
                print(ShowTrainInfo(loss, iter, batch_no, numIters, train_set.shape[0] / batch_size))

            if numIters % args.eval_iter == 0:
                if (batch_no + 1) * batch_size < valid_set.shape[0]:
                    item_batch = valid_set[(batch_no) * batch_size: (batch_no + 1) * batch_size, :]
                loss = sess.run([itemrec.loss_test], feed_dict={itemrec.input_predict: item_batch})
                print(ShowTrainInfo(loss, iter, batch_no, numIters, train_set.shape[0] / batch_size))

            batch_no += 1
            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size*1
                rec_preds_5=[] #1
                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (batch_no_test > 500):
                        break
                    val_beg = batch_no_test * batch_size_test
                    val_end = (batch_no_test + 1) * batch_size_test
                    item_batch = valid_set[val_beg: val_end, :]
                    [top_k_batch] = sess.run([itemrec.top_k], 
                                             feed_dict={itemrec.input_predict: item_batch,})
                    top_k = np.squeeze(top_k_batch[1])
                    for bi in range(top_k.shape[0]):
                        pred_items_5 = top_k[bi][:5]
                        true_item = item_batch[bi][-1]
                        predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                        rank_5 = predictmap_5.get(true_item)
                        if rank_5 == None:
                            rec_preds_5.append(0.0)  # 2
                        else:
                            rec_preds_5.append(1.0)  # 4
                    batch_no_test += 1
                    if (batch_no_test % 10 == 0):
                        hit_5 = sum(rec_preds_5) / float(len(rec_preds_5))
                        print("hit_5:", hit_5)
            numIters += 1
            if numIters % args.save_para_every == 0:
                save_path = saver.save(sess, model_path.format(iter, numIters))

if __name__ == '__main__':
    model_path = "Data/Models/generation_model/model_nextitnet_transfer_pretrain.ckpt"
    main(model_path)
