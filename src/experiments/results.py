import argparse, os, sys, cPickle as pickle, numpy as np

if __name__ == '__main__':
    
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    
    parser = argparse.ArgumentParser( description = 'Results' )
    parser.add_argument( '--pkl_file', type = str, dest = 'pkl_file')
    parser.add_argument( '--exp_type', type = str, dest = 'exp_type')
    
    args = parser.parse_args()

    if args.exp_type == 'regression':    
        [r, max_fold, max_fold_dir] = pickle.load(open(args.pkl_file,'rb'))
        bf = max_fold.values()[0]
        bfd = max_fold_dir.values()[0]
        #uni, lda, x, uni_lda, uni_x, lda_x, uni_lda_x, uni_0_0, zero_lda_X, uni_lda_0 = zip(*r.values())
        uni, lda, xhum, xliwc, uni_lda, xliwc_xhum, uni_xliwc, uni_xhum, lda_xhum, lda_xliwc, uni_xliwc_xhum, uni_lda_xliwc, uni_lda_xhum,uni_lda_xhum_xliwc  = zip(*r.values())
        print 'UNIGRAMS: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni), np.std(uni), bf[0], bfd[0])
        print 'LDA: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(lda), np.std(lda), bf[1], bfd[1])
        print 'X_HUM: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(xhum), np.std(xhum), bf[2], bfd[2])
        print 'X_LIWC: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(xliwc), np.std(xliwc), bf[3], bfd[3])
        print 'UNIGRAMS+LDA: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_lda), np.std(uni_lda), bf[4], bfd[4])
        print 'X_LIWC+X_HUMAN: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(xliwc_xhum), np.std(xliwc_xhum), bf[5], bfd[5])
        print 'UNIGRAMS+X_LIWC: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_xliwc), np.std(uni_xliwc), bf[6], bfd[6])
        print 'UNIGRAMS+X_HUM: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_xhum), np.std(uni_xhum), bf[7], bfd[7])
        print 'LDA+X_HUM: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(lda_xhum), np.std(lda_xhum), bf[8], bfd[8])
        print 'LDA+X_LIWC: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(lda_xliwc), np.std(lda_xliwc), bf[9], bfd[9])
        print 'UNIGRAMS+X_LIWC+X_HUM: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_xliwc_xhum), np.std(uni_xliwc_xhum), bf[10], bfd[10])
        print 'UNIGRAMS+LDA+X_LIWC: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_lda_xliwc), np.std(uni_lda_xliwc), bf[11], bfd[11])
        print 'UNIGRAMS+LDA+X_HUM: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_lda_xhum), np.std(uni_lda_xhum), bf[12], bfd[12])
        print 'UNIGRAMS+LDA+X_HUM+X_LIWC: %.4f +- %.4f, best fold: %d --> %s'%(np.mean(uni_lda_xhum_xliwc), np.std(uni_lda_xhum_xliwc), bf[13], bfd[13])
        #print 'UNIGRAMS+0+0: %.4f +- %.4f'%(np.mean(uni_0_0), np.std(uni_0_0))
        #print '0+LDA+X: %.4f +- %.4f'%(np.mean(zero_lda_X), np.std(zero_lda_X))
        #print 'UNIGRAMS+LDA+0: %.4f +- %.4f'%(np.mean(uni_lda_0), np.std(uni_lda_0))
    else:
        [p,r,f,s] = pickle.load(open(args.pkl_file,'rb'))
        for ii in p.iterkeys():
            print '[Label %s]'%str(ii)
            uni, uni_xhum, uni_xliwc, uni_xliwc_xhum, lda, uni_lda, lda_xhum, lda_xliwc, uni_lda_xliwc, uni_lda_xhum, uni_lda_xhum_xliwc=zip(*p[ii].values())
            print '----- Precision -----'
            print '>>> UNIGRAMS: %.4f +- %.4f'%(np.mean(uni), np.std(uni))
            print '>>> UNIGRAMS+X_HUM: %.4f +- %.4f'%(np.mean(uni_xhum), np.std(uni_xhum))
            print '>>> UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_xliwc), np.std(uni_xliwc))
            print '>>> UNIGRAMS+X_LIWC+X_HUM: %.4f +- %.4f'%(np.mean(uni_xliwc_xhum), np.std(uni_xliwc_xhum))
            print '>>> LDA: %.4f +- %.4f'%(np.mean(lda), np.std(lda))
            print '>>> LDA+UNI: %.4f +- %.4f'%(np.mean(uni_lda), np.std(uni_lda))
            print '>>> LDA+X_HUM: %.4f +- %.4f'%(np.mean(lda_xhum), np.std(lda_xhum))
            print '>>> LDA+X_LIWC: %.4f +- %.4f'%(np.mean(lda_xliwc), np.std(lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_lda_xliwc), np.std(uni_lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_HUMANS %.4f +- %.4f'%(np.mean(uni_lda_xhum), np.std(uni_lda_xhum))
            print '>>> LDA+UNIGRAMS+X_HUMANS+X_LIWC %.4f +- %.4f'%(np.mean(uni_lda_xhum_xliwc), np.std(uni_lda_xhum_xliwc))
            uni, uni_xhum, uni_xliwc, uni_xliwc_xhum, lda, uni_lda, lda_xhum, lda_xliwc, uni_lda_xliwc, uni_lda_xhum, uni_lda_xhum_xliwc=zip(*r[ii].values())
            print '----- Recall -----'
            print '>>> UNIGRAMS: %.4f +- %.4f'%(np.mean(uni), np.std(uni))
            print '>>> UNIGRAMS+X_HUM: %.4f +- %.4f'%(np.mean(uni_xhum), np.std(uni_xhum))
            print '>>> UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_xliwc), np.std(uni_xliwc))
            print '>>> UNIGRAMS+X_LIWC+X_HUM: %.4f +- %.4f'%(np.mean(uni_xliwc_xhum), np.std(uni_xliwc_xhum))
            print '>>> LDA: %.4f +- %.4f'%(np.mean(lda), np.std(lda))
            print '>>> LDA+UNI: %.4f +- %.4f'%(np.mean(uni_lda), np.std(uni_lda))
            print '>>> LDA+X_HUM: %.4f +- %.4f'%(np.mean(lda_xhum), np.std(lda_xhum))
            print '>>> LDA+X_LIWC: %.4f +- %.4f'%(np.mean(lda_xliwc), np.std(lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_lda_xliwc), np.std(uni_lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_HUMANS %.4f +- %.4f'%(np.mean(uni_lda_xhum), np.std(uni_lda_xhum))
            print '>>> LDA+UNIGRAMS+X_HUMANS+X_LIWC %.4f +- %.4f'%(np.mean(uni_lda_xhum_xliwc), np.std(uni_lda_xhum_xliwc))
            uni, uni_xhum, uni_xliwc, uni_xliwc_xhum, lda, uni_lda, lda_xhum, lda_xliwc, uni_lda_xliwc, uni_lda_xhum, uni_lda_xhum_xliwc=zip(*f[ii].values())
            print '----- F-score -----'
            print '>>> UNIGRAMS: %.4f +- %.4f'%(np.mean(uni), np.std(uni))
            print '>>> UNIGRAMS+X_HUM: %.4f +- %.4f'%(np.mean(uni_xhum), np.std(uni_xhum))
            print '>>> UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_xliwc), np.std(uni_xliwc))
            print '>>> UNIGRAMS+X_LIWC+X_HUM: %.4f +- %.4f'%(np.mean(uni_xliwc_xhum), np.std(uni_xliwc_xhum))
            print '>>> LDA: %.4f +- %.4f'%(np.mean(lda), np.std(lda))
            print '>>> LDA+UNI: %.4f +- %.4f'%(np.mean(uni_lda), np.std(uni_lda))
            print '>>> LDA+X_HUM: %.4f +- %.4f'%(np.mean(lda_xhum), np.std(lda_xhum))
            print '>>> LDA+X_LIWC: %.4f +- %.4f'%(np.mean(lda_xliwc), np.std(lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_lda_xliwc), np.std(uni_lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_HUMANS %.4f +- %.4f'%(np.mean(uni_lda_xhum), np.std(uni_lda_xhum))
            print '>>> LDA+UNIGRAMS+X_HUMANS+X_LIWC %.4f +- %.4f'%(np.mean(uni_lda_xhum_xliwc), np.std(uni_lda_xhum_xliwc))
            uni, uni_xhum, uni_xliwc, uni_xliwc_xhum, lda, uni_lda, lda_xhum, lda_xliwc, uni_lda_xliwc, uni_lda_xhum, uni_lda_xhum_xliwc=zip(*s[ii].values())
            print '----- Support -----'
            print '>>> UNIGRAMS: %.4f +- %.4f'%(np.mean(uni), np.std(uni))
            print '>>> UNIGRAMS+X_HUM: %.4f +- %.4f'%(np.mean(uni_xhum), np.std(uni_xhum))
            print '>>> UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_xliwc), np.std(uni_xliwc))
            print '>>> UNIGRAMS+X_LIWC+X_HUM: %.4f +- %.4f'%(np.mean(uni_xliwc_xhum), np.std(uni_xliwc_xhum))
            print '>>> LDA: %.4f +- %.4f'%(np.mean(lda), np.std(lda))
            print '>>> LDA+UNI: %.4f +- %.4f'%(np.mean(uni_lda), np.std(uni_lda))
            print '>>> LDA+X_HUM: %.4f +- %.4f'%(np.mean(lda_xhum), np.std(lda_xhum))
            print '>>> LDA+X_LIWC: %.4f +- %.4f'%(np.mean(lda_xliwc), np.std(lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_LIWC: %.4f +- %.4f'%(np.mean(uni_lda_xliwc), np.std(uni_lda_xliwc))
            print '>>> LDA+UNIGRAMS+X_HUMANS %.4f +- %.4f'%(np.mean(uni_lda_xhum), np.std(uni_lda_xhum))
            print '>>> LDA+UNIGRAMS+X_HUMANS+X_LIWC %.4f +- %.4f'%(np.mean(uni_lda_xhum_xliwc), np.std(uni_lda_xhum_xliwc))
            
            
            
            
        
    #for ii in r:
    #    print ii, r[ii]