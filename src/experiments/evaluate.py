import argparse, os, sys, cPickle as pickle, numpy as np, codecs, re
from topic_model import MalletLDA
from sklearn.svm import SVR, SVC
from glob import iglob
from sklearn import preprocessing as pp
from sklearn.metrics import precision_recall_fscore_support as pr_fs

def to_tokenized(text_list, stoplist):
    #ret = RegexpTokenizer('[\w\d]+') 
    #texts = [[word.encode('utf-8') for word in ret.tokenize(document.lower()) 
    #          if word not in stoplist and len(word) >= 3] for document in text_list]
    return [[word for word in document.lower().split(" ")] for document in text_list]   


def word_counts (vocab, text_to_count):
    
    counts = {ii:0 for ii in vocab.keys()}    
    for ii in text_to_count:
        if vocab.has_key(ii):
            counts[ii]+=1
        
    return counts.values()


def compute_prfs(yt, y_feat, p_feat, r_feat, f_feat, s_feat, labels):

    for (iind, (pr,rr,ff,ss)) in enumerate(zip(*pr_fs(yt, y_feat))): # each label seen in gt
        if iind >= len(labels): # to cope with bug 
            continue
        if not p_feat.has_key(labels[iind]):
            p_feat[labels[iind]] = []
            r_feat[labels[iind]] = []
            f_feat[labels[iind]] = []
            s_feat[labels[iind]] = []
        p_feat[labels[iind]].append(pr)
        r_feat[labels[iind]].append(rr)
        f_feat[labels[iind]].append(ff)
        s_feat[labels[iind]].append(ss)
        
    return p_feat, r_feat, f_feat, s_feat



if __name__ == '__main__':
    
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    
    parser = argparse.ArgumentParser( description = 'Run linear SVM regression \
    or predictions of a variable given unigrams, LIWC, topic features and combinations' )
    parser.add_argument( '--stoplist_file', type = str, dest = 'stoplist_file', 
                         help = 'File with a list of stopwords, one per row')
    parser.add_argument( '--out_folder', type = str, dest = 'out_folder', default='./', 
                         help = 'Folder where outputs will be dumped')
    parser.add_argument( '--feature_pkl_file_regexp', type = str, dest = 'feature_pkl_file_regexp', 
                         help = 'Regular expression that returns the path of one of more pickle file with features')
    parser.add_argument( '--target', type = str, dest = 'target', 
                         help = 'Name of the variable to be predicted')
    parser.add_argument( '--experiment_type', type = str, dest = 'experiment_type', 
                         help = 'Experiment to be run: {regression, prediction}')
    parser.add_argument( '--topic_model_pkl', type = str, dest = 'topic_model_pkl', 
                         help = 'Pickle with object that has the topic models')
    parser.add_argument( '--topic_model_args', type = str, dest = 'topic_model_args', 
                         help = 'Arguments to topic model: arg1,val1,arg2,val2 ...')
    parser.add_argument( '--fold', type = int, dest = 'fold', 
                         help = 'Will only compute results for that train/test split')
    parser.add_argument( '--fold_path', type = str, dest = 'fold_path', 
                         help = 'Folder where the results corresponding to the given fold lives')
    parser.add_argument( '--topic_model_name', type = str, dest = 'topic_model_name', 
                         help = 'Name of the topic model to be used. Overrides the one in \"--topic_model_pkl\"')
    parser.add_argument( '--topic_model_path', type = str, dest = 'topic_model_path', 
                         help = 'Path to the topic model.')
    parser.add_argument( '--mallet_bin_path', type = str, dest = 'mallet_bin_path', 
                         help = 'Path to the Mallet topic model.')


    args = parser.parse_args()

    # Load stoplist
    stoplist=[]
    if not args.stoplist_file == None:
        stoplist = [word.replace('\'','') for word in codecs.open(args.stoplist_file,'r','utf-8').readlines()]

    # Load the pickle that has the trained topic models for k different partitions
    [dict_kfold, skf, k, num_topics, tool_path, tm_name] = pickle.load(open(args.topic_model_pkl,'rb'))
        
    # This makes adjustments in case we are testing at a specific fold
    fold_str = ""
    if args.fold <> None and args.fold_path <> None and args.topic_model_path <> None \
    and args.topic_model_name <> None:
        tm_name=args.topic_model_name
        tool_path = args.topic_model_path 
        fold_str += "_fold_%d"%args.fold
        for ii in dict_kfold.iterkeys():
            for jj in dict_kfold[ii].iterkeys():
                dict_kfold[ii][jj]=args.fold_path
        
    tm_class = getattr(__import__('topic_model'), tm_name)
    if tm_name == 'ITMTomcat':
        tm = tm_class(tool_path, args.mallet_bin_path)
    else:
        tm = MalletLDA(args.mallet_bin_path)
        
    # Extract topic model parameters
    tm_args = re.findall('([^=]+)=([^,]+),*',args.topic_model_args, re.M)
    tm_args = {ii[0]:ii[1] for ii in tm_args}
    
    if args.experiment_type == 'regression':
        r={}
        svm = SVR(kernel='linear', verbose=True, max_iter=1E6, tol=1E-4)
    else:
        p, r, f, s = {},{},{},{}
        svm = SVC(kernel='linear', verbose=True, max_iter=1E6, tol=1E-4, class_weight=None)
    
    max_fold, max_fold_dir = {},{}

    # The feature pickle has features computed for n input files, so the experiment is run
    # on each one and the average results are reported
    for ii in iglob(args.feature_pkl_file_regexp):

        dict_data = pickle.load(open(ii))
        var = dict_data[7][0].index(args.target)
        ids = np.asarray(dict_data[0])
        X_hum = dict_data[1]
        X_liwc = dict_data[2]
        y = np.asarray(zip(*dict_data[3])[var],np.float)
        texts=dict_data[4]

        if args.experiment_type == 'regression':
            # Pearson's r score for each feature
            r_uni, r_uni_X_hum, r_uni_lda, r_uni_X_liwc, r_uni_lda_X_liwc, \
            r_uni_X_liwc_X_hum, r_uni_lda_X_hum, r_uni_lda_X_hum_X_liwc=[],[],[],[],[],[],[],[]
            r_lda, r_lda_X_hum, r_lda_X_liwc = [],[],[]
            r_X_liwc, r_X_liwc_X_hum, r_X_hum = [],[],[]
        else:
            # Precision, recall, f-measure and support for each feature
            p_uni, r_uni, f_uni, s_uni = {},{},{},{}
            p_uni_X_liwc, r_uni_X_liwc, f_uni_X_liwc, s_uni_X_liwc = {},{},{},{}
            p_lda, r_lda, f_lda, s_lda = {},{},{},{}
            p_uni_lda, r_uni_lda, f_uni_lda, s_uni_lda = {},{},{},{}
            p_lda_X_liwc, r_lda_X_liwc, f_lda_X_liwc, s_lda_X_liwc ={},{},{},{}
            p_uni_lda_X_liwc, r_uni_lda_X_liwc, f_uni_lda_X_liwc, s_uni_lda_X_liwc = {},{},{},{}
            p_uni_X_hum, r_uni_X_hum, f_uni_X_hum, s_uni_X_hum = {},{},{},{}
            p_uni_X_liwc_X_hum, r_uni_X_liwc_X_hum, f_uni_X_liwc_X_hum, s_uni_X_liwc_X_hum = {},{},{},{}
            p_lda_X_hum, r_lda_X_hum, f_lda_X_hum, s_lda_X_hum = {},{},{},{}
            p_uni_lda_X_hum, r_uni_lda_X_hum, f_uni_lda_X_hum, s_uni_lda_X_hum = {},{},{},{}
            p_uni_lda_X_hum_X_liwc, r_uni_lda_X_hum_X_liwc, f_uni_lda_X_hum_X_liwc, \
            s_uni_lda_X_hum_X_liwc = {},{},{},{}

        for ind, (train_index, test_index) in enumerate(skf[ii]):

            # Check if a single fold will be considered
            #if args.fold <> None and args.fold_path <> None and ind <> args.fold:
            #    continue
            
            print '[[Running k=%d of %d]]'%(ind+1, k)
            y_train, yt = y[train_index], y[test_index]
            ids_train, ids_test = ids[train_index], ids[test_index]
            labels=np.unique(yt)

            X_hum_train = X_hum[train_index]
            X_liwc_train = X_liwc[train_index]
            scaler  = pp.StandardScaler().fit(X_hum_train)
            X_hum_train = scaler.transform(X_hum_train)
            X_hum_test  = scaler.transform(X_hum[test_index])
            scaler  = pp.StandardScaler().fit(X_liwc_train)
            X_liwc_train = scaler.transform(X_liwc_train)
            X_liwc_test = scaler.transform(X_liwc[test_index])
            
            print '> Running inference'
            train_text = [texts[jj] for jj in train_index]
            train_corpus = to_tokenized(train_text, stoplist)
            
            X_train_lda = tm.infer(dict_kfold[ii][ind], True, ids_train, train_corpus, tm_args)
            X_train_lda =  [zip(*jj)[1] for jj in X_train_lda]

            scaler = pp.StandardScaler().fit(X_train_lda)
            X_train_lda = scaler.transform(X_train_lda)
            test_text =  [texts[jj] for jj in test_index]
            test_corpus = to_tokenized(test_text, stoplist)
            
            X_test_lda = tm.infer(dict_kfold[ii][ind], False, ids_test, test_corpus, tm_args)
            X_test_lda = scaler.transform( [zip(*jj)[1] for jj in X_test_lda] )

            vocab = {jj.strip():jj.strip() for kk in train_corpus for jj in kk}
 
            X_train_uni = np.zeros((len(train_corpus),len(vocab)))                
            for row, jj in enumerate(train_corpus):
                X_train_uni[row] = np.array(word_counts(vocab, jj)) + np.ones((1, len(vocab)))  # add-one smoothing
            
            scaler = pp.StandardScaler().fit(X_train_uni)
            X_train_uni = scaler.transform(X_train_uni)

            X_test_uni = np.zeros((len(test_corpus),len(vocab)))
            for row, jj in enumerate(test_corpus):
                X_test_uni[row]=np.array(word_counts(vocab, jj)) + np.ones((1, len(vocab)))  # add-one smoothing
            X_test_uni = scaler.transform(X_test_uni)

            # Fit linear models for each case
            print '> Fitting and scoring models'
 
            # unigrams only
   
            print '>> unigrams'
            svm.fit(X_train_uni,y_train)
            y_uni = svm.predict(X_test_uni)
            
            # LDA only

            print '>> TM only'
            svm.fit(X_train_lda,y_train)
            y_lda = svm.predict(X_test_lda)

            # X_hum only
            print '>> contextual features'
            svm.fit(X_hum_train,y_train)
            y_X_hum = svm.predict(X_hum_test)

            # X_liwc only
            print '>> liwc features'
            svm.fit(X_liwc_train,y_train)
            y_X_liwc = svm.predict(X_liwc_test)
            
            # unigrams + LDA
            print '>> unigrams + TM'
            X_train_comb = np.concatenate((X_train_uni, X_train_lda),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb, y_train)
            X_test_comb = np.concatenate((X_test_uni, X_test_lda),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_lda = svm.predict(X_test_comb)

            # X_liwc + X_hum
            print '>> liwc + humans'
            X_train_comb=np.concatenate((X_liwc_train, X_hum_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_liwc_test, X_hum_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_X_liwc_X_hum = svm.predict(X_test_comb)

            # unigrams + X_hum
            print '>> unigrams + contextual'
            X_train_comb=np.concatenate((X_train_uni, X_hum_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb, y_train)
            X_test_comb=np.concatenate((X_test_uni, X_hum_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_X_hum = svm.predict(X_test_comb)

            # unigrams + X_liwc
            print '>> unigrams + liwc'
            X_train_comb=np.concatenate((X_train_uni, X_liwc_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb, y_train)
            X_test_comb=np.concatenate((X_test_uni, X_liwc_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_X_liwc = svm.predict(X_test_comb)
            
            # LDA + X_hum
            print '>> TM + contextual'
            X_train_comb=np.concatenate((X_train_lda, X_hum_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_lda, X_hum_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_lda_X_hum = svm.predict(X_test_comb)
            
            # LDA + X_liwc
            print '>> TM + contextual'
            X_train_comb=np.concatenate((X_train_lda, X_liwc_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_lda, X_liwc_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_lda_X_liwc = svm.predict(X_test_comb)

            # unigrams + X_liwc + X_hum
            print '>> unigrams + liwc + contextual'
            X_train_comb=np.concatenate((X_train_uni, X_liwc_train, X_hum_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_uni, X_liwc_test, X_hum_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_X_liwc_X_hum = svm.predict(X_test_comb)

            # unigrams + LDA + X_liwc
            print '>> unigrams + TM + liwc'
            X_train_comb=np.concatenate((X_train_uni, X_train_lda, X_liwc_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_uni, X_test_lda, X_liwc_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_lda_X_liwc = svm.predict(X_test_comb)

            # unigrams + LDA + X_hum
            print '>> unigrams + TM + contextual'
            X_train_comb=np.concatenate((X_train_uni, X_train_lda, X_hum_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_uni, X_test_lda, X_hum_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_lda_X_hum = svm.predict(X_test_comb)

            # unigrams + LDA + X_hum + X_liwc
            print '>> unigrams + TM + contextual + liwc'
            X_train_comb=np.concatenate((X_train_uni, X_train_lda, X_hum_train, X_liwc_train),axis=1)
            scaler = pp.StandardScaler().fit(X_train_comb)
            X_train_comb = scaler.transform(X_train_comb)
            svm.fit(X_train_comb,y_train)
            X_test_comb=np.concatenate((X_test_uni, X_test_lda, X_hum_test, X_liwc_test),axis=1)
            X_test_comb = scaler.transform(X_test_comb)
            y_uni_lda_X_hum_X_liwc = svm.predict(X_test_comb)
            
            # Compute scores
            if args.experiment_type == 'regression':
                print '>> Computing Pearson\'s scores'
                r_uni.append(np.corrcoef(y_uni-np.mean(y_uni), yt-np.mean(yt))[0,1])
                r_lda.append(np.corrcoef(y_lda-np.mean(y_lda), yt-np.mean(yt))[0,1])
                r_X_hum.append(np.corrcoef(y_X_hum-np.mean(y_X_hum), yt-np.mean(yt))[0,1])
                r_X_liwc.append(np.corrcoef(y_X_liwc-np.mean(y_X_liwc), yt-np.mean(yt))[0,1])
                r_uni_lda.append(np.corrcoef(y_uni_lda-np.mean(y_uni_lda), yt-np.mean(yt))[0,1])
                r_X_liwc_X_hum.append(np.corrcoef(y_X_liwc_X_hum-np.mean(y_X_liwc_X_hum), yt-np.mean(yt))[0,1])
                r_uni_X_liwc.append(np.corrcoef(y_uni_X_liwc-np.mean(y_uni_X_liwc), yt-np.mean(yt))[0,1])
                r_uni_X_hum.append(np.corrcoef(y_uni_X_hum-np.mean(y_uni_X_hum), yt-np.mean(yt))[0,1])
                r_lda_X_hum.append(np.corrcoef(y_lda_X_hum-np.mean(y_lda_X_hum), yt-np.mean(yt))[0,1])
                r_lda_X_liwc.append(np.corrcoef(y_lda_X_liwc-np.mean(y_lda_X_liwc), yt-np.mean(yt))[0,1])
                r_uni_X_liwc_X_hum.append(np.corrcoef(y_uni_X_liwc_X_hum-np.mean(y_uni_X_liwc_X_hum), yt-np.mean(yt))[0,1])
                r_uni_lda_X_liwc.append(np.corrcoef(y_uni_lda_X_liwc-np.mean(y_uni_lda_X_liwc), yt-np.mean(yt))[0,1])
                r_uni_lda_X_hum.append(np.corrcoef(y_uni_lda_X_hum-np.mean(y_uni_lda_X_hum), yt-np.mean(yt))[0,1])
                r_uni_lda_X_hum_X_liwc.append(np.corrcoef(y_uni_lda_X_hum_X_liwc-np.mean(y_uni_lda_X_hum_X_liwc), yt-np.mean(yt))[0,1])
            else:
                # unigrams
                p_uni, r_uni, f_uni, s_uni = compute_prfs(yt, y_uni, p_uni, r_uni, f_uni, s_uni, labels)
                
                # unigrams and context features
                p_uni_X_hum, r_uni_X_hum, f_uni_X_hum, s_uni_X_hum = compute_prfs(yt, y_uni_X_hum, p_uni_X_hum, 
                                                                                  r_uni_X_hum, f_uni_X_hum, s_uni_X_hum, labels)
                # unigrams + X_liwc
                p_uni_X_liwc, r_uni_X_liwc, f_uni_X_liwc, s_uni_X_liwc = compute_prfs(yt, y_uni_X_liwc, p_uni_X_liwc, 
                                                                                      r_uni_X_liwc, f_uni_X_liwc, s_uni_X_liwc, labels)
                # unigrams + X_liwc + X_hum
                p_uni_X_liwc_X_hum, r_uni_X_liwc_X_hum, f_uni_X_liwc_X_hum, s_uni_X_liwc_X_hum = compute_prfs(yt, y_uni_X_liwc_X_hum, 
                                                                                                              p_uni_X_liwc_X_hum, r_uni_X_liwc_X_hum, 
                                                                                                              f_uni_X_liwc_X_hum, s_uni_X_liwc_X_hum, labels)
                # lda only
                p_lda, r_lda, f_lda, s_lda = compute_prfs(yt, y_lda, p_lda, r_lda, f_lda, s_lda, labels)

                # unigrams + lda 
                p_uni_lda, r_uni_lda, f_uni_lda, s_uni_lda = compute_prfs(yt, y_uni_lda, p_uni_lda, r_uni_lda, f_uni_lda, s_uni_lda, labels)

                # lda + humans
                p_lda_X_hum, r_lda_X_hum, f_lda_X_hum, s_lda_X_hum = compute_prfs(yt, y_lda_X_hum, p_lda_X_hum, r_lda_X_hum, f_lda_X_hum, s_lda_X_hum, labels)

                # lda + liwc
                p_lda_X_liwc, r_lda_X_liwc, f_lda_X_liwc, s_lda_X_liwc = compute_prfs(yt, y_lda_X_liwc, p_lda_X_liwc, r_lda_X_liwc, f_lda_X_liwc, s_lda_X_liwc, labels)

                # unigrams + lda + liwc
                p_uni_lda_X_liwc, r_uni_lda_X_liwc, f_uni_lda_X_liwc, s_uni_lda_X_liwc = compute_prfs(yt, y_uni_lda_X_liwc, p_uni_lda_X_liwc, 
                                                                                                      r_uni_lda_X_liwc, f_uni_lda_X_liwc, 
                                                                                                      s_uni_lda_X_liwc, labels)
                # unigrams + lda + humans
                p_uni_lda_X_hum, r_uni_lda_X_hum, f_uni_lda_X_hum, s_uni_lda_X_hum = compute_prfs(yt, y_uni_lda_X_hum, p_uni_lda_X_hum, 
                                                                                                      r_uni_lda_X_hum, f_uni_lda_X_hum, 
                                                                                                      s_uni_lda_X_hum, labels)
                # unigrams + lda + liwc + humans
                p_uni_lda_X_hum_X_liwc, r_uni_lda_X_hum_X_liwc, f_uni_lda_X_hum_X_liwc, s_uni_lda_X_hum_X_liwc = compute_prfs(yt, y_uni_lda_X_hum_X_liwc, p_uni_lda_X_hum_X_liwc, 
                                                                                                      r_uni_lda_X_hum_X_liwc, f_uni_lda_X_hum_X_liwc, 
                                                                                                      s_uni_lda_X_hum_X_liwc, labels)

        # Will compute the average scores per folder
        if args.experiment_type == 'regression':
            r[ii]=(np.mean(np.array(r_uni)), 
                   np.mean(np.array(r_lda)), 
                   np.mean(np.array(r_X_hum)), 
                   np.mean(np.array(r_X_liwc)),
                   np.mean(np.array(r_uni_lda)), 
                   np.mean(np.array(r_X_liwc_X_hum)), 
                   np.mean(np.array(r_uni_X_liwc)),
                   np.mean(np.array(r_uni_X_hum)),
                   np.mean(np.array(r_lda_X_hum)),
                   np.mean(np.array(r_lda_X_liwc)),
                   np.mean(np.array(r_uni_X_liwc_X_hum)),
                   np.mean(np.array(r_uni_lda_X_liwc)),
                   np.mean(np.array(r_uni_lda_X_hum)), 
                   np.mean(np.array(r_uni_lda_X_hum_X_liwc))) #\,
            
            max_fold[ii]=(r_uni.index(max(r_uni)), 
                   r_lda.index(max(r_lda)), 
                   r_X_hum.index(max(r_X_hum)), 
                   r_X_liwc.index(max(r_X_liwc)),
                   r_uni_lda.index(max(r_uni_lda)), 
                   r_X_liwc_X_hum.index(max(r_X_liwc_X_hum)), 
                   r_uni_X_liwc.index(max(r_uni_X_liwc)),
                   r_uni_X_hum.index(max(r_uni_X_hum)),
                   r_lda_X_hum.index(max(r_lda_X_hum)),
                   r_lda_X_liwc.index(max(r_lda_X_liwc)),
                   r_uni_X_liwc_X_hum.index(max(r_uni_X_liwc_X_hum)),
                   r_uni_lda_X_liwc.index(max(r_uni_lda_X_liwc)),
                   r_uni_lda_X_hum.index(max(r_uni_lda_X_hum)), 
                   r_uni_lda_X_hum_X_liwc.index(max(r_uni_lda_X_hum_X_liwc))) #\,
            
            # Single fold evaluation case
#             if args.fold <> None and args.fold_path <> None and args.topic_model_path <> None \
#             and args.topic_model_name <> None:
#                 max_fold[ii]=[args.fold for jj in range(14)]    
               
            max_fold_dir[ii]=(dict_kfold[ii][ max_fold[ii][0] ],
                   dict_kfold[ii][ max_fold[ii][1] ],
                   dict_kfold[ii][ max_fold[ii][2] ],
                   dict_kfold[ii][ max_fold[ii][3] ],
                   dict_kfold[ii][ max_fold[ii][4] ],                      
                   dict_kfold[ii][ max_fold[ii][5] ],                      
                   dict_kfold[ii][ max_fold[ii][6] ],                      
                   dict_kfold[ii][ max_fold[ii][7] ],                      
                   dict_kfold[ii][ max_fold[ii][8] ],                      
                   dict_kfold[ii][ max_fold[ii][9] ],                      
                   dict_kfold[ii][ max_fold[ii][10] ],                      
                   dict_kfold[ii][ max_fold[ii][11] ],                      
                   dict_kfold[ii][ max_fold[ii][12] ],                      
                   dict_kfold[ii][ max_fold[ii][13] ])
        else:
            for jj in p_uni.iterkeys():
                
                if not p.has_key(jj):
                    p[jj]={}
                    r[jj]={}
                    f[jj]={}
                    s[jj]={}                
                    max_fold[jj]={}                
                    max_fold_dir[jj]={}                
                
                p[jj][ii]=(np.mean(np.array(p_uni[jj])), 
                       np.mean(np.array(p_uni_X_hum[jj])), 
                       np.mean(np.array(p_uni_X_liwc[jj])),
                       np.mean(np.array(p_uni_X_liwc_X_hum[jj])),
                       np.mean(np.array(p_lda[jj])),                      
                       np.mean(np.array(p_uni_lda[jj])),                      
                       np.mean(np.array(p_lda_X_hum[jj])),                      
                       np.mean(np.array(p_lda_X_liwc[jj])),                      
                       np.mean(np.array(p_uni_lda_X_liwc[jj])),                      
                       np.mean(np.array(p_uni_lda_X_hum[jj])),                      
                       np.mean(np.array(p_uni_lda_X_hum_X_liwc[jj])))
                
                r[jj][ii]=(np.mean(np.array(r_uni[jj])), 
                       np.mean(np.array(r_uni_X_hum[jj])), 
                       np.mean(np.array(r_uni_X_liwc[jj])),
                       np.mean(np.array(r_uni_X_liwc_X_hum[jj])),
                       np.mean(np.array(r_lda[jj])),                      
                       np.mean(np.array(r_uni_lda[jj])),                      
                       np.mean(np.array(r_lda_X_hum[jj])),                      
                       np.mean(np.array(r_lda_X_liwc[jj])),                      
                       np.mean(np.array(r_uni_lda_X_liwc[jj])),                      
                       np.mean(np.array(r_uni_lda_X_hum[jj])),                      
                       np.mean(np.array(r_uni_lda_X_hum_X_liwc[jj])))
                
                f[jj][ii]=(np.mean(np.array(f_uni[jj])), 
                       np.mean(np.array(f_uni_X_hum[jj])), 
                       np.mean(np.array(f_uni_X_liwc[jj])),
                       np.mean(np.array(f_uni_X_liwc_X_hum[jj])),
                       np.mean(np.array(f_lda[jj])),                      
                       np.mean(np.array(f_uni_lda[jj])),                      
                       np.mean(np.array(f_lda_X_hum[jj])),                      
                       np.mean(np.array(f_lda_X_liwc[jj])),                      
                       np.mean(np.array(f_uni_lda_X_liwc[jj])),                      
                       np.mean(np.array(f_uni_lda_X_hum[jj])),                      
                       np.mean(np.array(f_uni_lda_X_hum_X_liwc[jj])))
                
                s[jj][ii]=(np.mean(np.array(s_uni[jj])), 
                       np.mean(np.array(s_uni_X_hum[jj])), 
                       np.mean(np.array(s_uni_X_liwc[jj])),
                       np.mean(np.array(s_uni_X_liwc_X_hum[jj])),
                       np.mean(np.array(s_lda[jj])),                      
                       np.mean(np.array(s_uni_lda[jj])),                      
                       np.mean(np.array(s_lda_X_hum[jj])),                      
                       np.mean(np.array(s_lda_X_liwc[jj])),                      
                       np.mean(np.array(s_uni_lda_X_liwc[jj])),                      
                       np.mean(np.array(s_uni_lda_X_hum[jj])),                      
                       np.mean(np.array(s_uni_lda_X_hum_X_liwc[jj])))

                # Single fold evaluation case
#                 if args.fold <> None and args.fold_path <> None and args.topic_model_path <> None \
#                 and args.topic_model_name <> None:
#                     max_fold[ii][jj]=[args.fold for jj in range(14)]    
                
                max_fold[jj][ii]=(f_uni.index(max(f_uni[jj])), 
                       f_uni_X_hum.index(max(f_uni_X_hum[jj])), 
                       f_uni_X_liwc.index(max(f_uni_X_liwc[jj])),
                       f_uni_X_liwc_X_hum.index(max(f_uni_X_liwc_X_hum[jj])),
                       f_lda.index(max(f_lda[jj])),                      
                       f_uni_lda.index(max(f_uni_lda[jj])),                      
                       f_lda_X_hum.index(max(f_lda_X_hum[jj])),                      
                       f_lda_X_liwc.index(max(f_lda_X_liwc[jj])),                      
                       f_uni_lda_X_liwc.index(max(f_uni_lda_X_liwc[jj])),                      
                       f_uni_lda_X_hum.index(max(f_uni_lda_X_hum[jj])),                      
                       f_uni_lda_X_hum_X_liwc.index(max(f_uni_lda_X_hum_X_liwc[jj])))

                max_fold_dir[jj][ii]=(dict_kfold[ii][ max_fold[jj][ii][0] ],
                       dict_kfold[ii][ max_fold[jj][ii][1] ],
                       dict_kfold[ii][ max_fold[jj][ii][2] ],
                       dict_kfold[ii][ max_fold[jj][ii][3] ],
                       dict_kfold[ii][ max_fold[jj][ii][4] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][5] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][6] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][7] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][8] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][9] ],                      
                       dict_kfold[ii][ max_fold[jj][ii][10] ])
    
    
    if not os.path.exists(args.out_folder +'/'+ tm_name+'/'):
        os.makedirs(args.out_folder +'/'+ tm_name+'/')
        
    if args.experiment_type == 'regression':
        pickle.dump([r, max_fold, max_fold_dir], open(args.out_folder +'/'+ tm_name+'/' +args.target+\
        '.regression.topics_' + str(num_topics) + '_k_' + str(k) + fold_str + '.pkl', 'wb'))
    elif args.experiment_type == 'prediction':
        pickle.dump([p, r, f, s, max_fold, max_fold_dir], open(args.out_folder +'/'+ tm_name+'/' +args.target+\
        '.prediction.topics_' + str(num_topics) + '_k_' + str(k) + fold_str + '.pkl', 'wb'))
    
     
    print 'Okay, done!'
    
    
