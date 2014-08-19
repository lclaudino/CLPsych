import argparse, os, cPickle as pickle, numpy as np, codecs, re
from sklearn.cross_validation import KFold
from glob import iglob
#from nltk.tokenize import RegexpTokenizer
from topic_model import TopicModel as tm
from shutil import rmtree

def to_tokenized(text_list, stoplist):
    #ret = RegexpTokenizer('[\w\d]+') 
    #texts = [[word.encode('utf-8') for word in ret.tokenize(document.lower()) 
    #          if word not in stoplist and len(word) >= 3] for document in text_list]
    return [[word for word in document.lower().split(" ")] for document in text_list]   


def kfold_train(k, filenames, stoplist, topic_model, tm_args, out_folder):

    dict_kfold = {}
    skf={}
    for ii in filenames:

        # Load texts
        dict_data = pickle.load(open(ii))
        texts=dict_data[4] # --> WILL HAVE TO GENERALIZE THIS BACK IN THE FEATURE GENERATION
        
        # Set up kfold experiments
        skf[ii] = KFold(len(texts), k)
        if not dict_kfold.has_key(ii):
            dict_kfold[ii]={}
        
        for ind,(train_index,_) in enumerate (skf[ii]):
            # Target folder for current topic model       
            curr_dir=out_folder+'/'+ type(topic_model).__name__ + '/' + os.path.basename(ii) + \
            '/topics-'+tm_args['num_topics'] + '-k-'+str(ind)+'/'
            
            if os.path.exists(curr_dir):
                rmtree(curr_dir)

            os.makedirs(curr_dir)

            # Load ids of documents used as training data
            ids=np.array(dict_data[0])[train_index]
            train_corpus = to_tokenized([texts[jj] for jj in train_index], stoplist)
            
            # Train topic model
            topic_model.train(curr_dir, ids, train_corpus, tm_args)

            # Save path to results
            dict_kfold[ii][ind] = curr_dir
            
    # Save pickle with kfold and topic info
    pickle.dump([dict_kfold, skf, k, tm_args['num_topics'], tm.tool_path, type(topic_model).__name__], 
                open(out_folder +'/'+ type(topic_model).__name__ +\
             '/' + os.path.basename(filenames[0]) +\
             '-to-' + os.path.basename(filenames[-1]) +\
             '_'+tm_args['num_topics'] +\
             '_kf_' + str(args.k) + '.pkl','wb'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser( description = 'Experiment' )
    parser.add_argument( '--stoplist_file', type = str, dest = 'stoplist_file', 
                         help = 'File with a list of stopwords, one per row')
    parser.add_argument( '--out_folder', type = str, dest = 'out_folder', default='./', 
                         help = 'Folder where outputs will be dumped')
    parser.add_argument( '--feature_pkl_file_regexp', type = str, dest = 'feature_pkl_file_regexp', 
                         help = 'Regular expression that returns the path of one of more pickle file with features')
    parser.add_argument( '--bin_path', type = str, dest = 'bin_path', 
                         help = 'Path to topic model binary')
    parser.add_argument( '--topic_model_name', type = str, dest = 'topic_model_name', 
                         help = 'Name of the topic model to be used')
    parser.add_argument( '--topic_model_args', type = str, dest = 'topic_model_args', 
                         help = 'Arguments to topic model: arg1=val1,arg2=val2 ...')
    parser.add_argument( '--k', type = int, dest = 'k', help = 'Number of folds to do cross-validation with')
    #parser.add_argument('--retrain', dest='retrain', action='store_true')

    args = parser.parse_args()

    # Extract topic model parameters
    tm_args = re.findall('([^=]+)=([^,]+),*',args.topic_model_args, re.M)
    tm_args = {ii[0]:ii[1] for ii in tm_args}

    # Load stoplist
    stoplist=[]
    if not args.stoplist_file == None:
        stoplist = [word.replace('\'','') for word in codecs.open(args.stoplist_file,'r','utf-8').readlines()]

    # Run all LDA stuff at once
    filenames=list(iglob(args.feature_pkl_file_regexp))
    
    # Pickle with topic model stuff
    tm_file=args.out_folder + '/' + os.path.basename(filenames[0]) + \
    '-to-' + os.path.basename(filenames[-1]) + '.mallet.lda.topics_' + \
    tm_args['num_topics'] +'_kf_' + str(args.k) + '.pkl'
    print tm_file
        
    # Instantiate topic model based on the given name
    tm_class = getattr(__import__('topic_model'), args.topic_model_name)
    tm = tm_class(args.bin_path)
    kfold_train(args.k, filenames, stoplist, tm, tm_args, args.out_folder)
    
    
    
    
    