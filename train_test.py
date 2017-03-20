import logging
import argparse
try:
    import cPickle as pickle
except:
    import pickle
from selektor import data
from selektor import qa_score
import theano


logger = logging.getLogger("selektor.train")

if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="pkl file for dataset")
    parser.add_argument("--dev_refname", help="reference fname for dev set")
    parser.add_argument("--test_refname", help="reference fname for test set")
    parser.add_argument("--dev_ofname", help="output prediction for dev set")
    parser.add_argument("--test_ofname", help="output prediction for test set")
    parser.add_argument("--cnn_cnt", type=int, default=0, help="CNN-Cnt model or not, default is 0")
    args = parser.parse_args()
    
    print ("loading data...")
    x = pickle.load(open(args.dataset,"rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]
    max_l = 40
    print("data loaded!")
    if args.cnn_cnt == 0:
        cnn_cnt = False
    else:
        cnn_cnt = True
    qa_score.tuning_cnn(revs, wordvecs, max_l, args.dev_refname, args.test_refname, args.dev_ofname, args.test_ofname, cnn_cnt)
    
    logger.info('end logging')
