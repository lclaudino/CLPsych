import os.path, glob, gzip, argparse, numpy as np, cPickle as pickle
from operator import itemgetter
from liwc import LIWC
import csv_with_encoding as csvwe

class FeatureFilesCasl:

    def __init__(self, feat_name='casl'):
        self.feat_name=feat_name
        self.dict_feats={}

    def create_data_matrices(self, dict_sent):
    
        dict_data = {}
        for ii in dict_sent.iterkeys():
            list_sent = dict_sent[ii]
            ids, human_feats, liwc_feats, sent, text = zip(*list_sent)
            X_humans = [jj.values() for jj in human_feats]
            X_liwc = [jj.values() for jj in liwc_feats]
            text = [jj for jj in text]
            y_sent = [jj.values()  for jj in sent]
            X_humans_names = [jj.keys()  for jj in human_feats]
            X_liwc_names = [jj.keys()  for jj in liwc_feats]
            y_sent_names = [jj.keys()  for jj in sent]
            
            dict_data[ii] = (ids, np.asmatrix(X_humans,np.float), np.asmatrix(X_liwc,np.float), \
                             y_sent, text, X_humans_names, X_liwc_names, y_sent_names)

        return dict_data


    # creates one vw file per label in the big-5 including the liwc features and lda, if posterior file passed
    def create_vw_string(self, dict_human_feats, dict_lda):
    
        dict_str={}
        for ii in dict_human_feats.iterkeys():
            
            dict_str[ii]={}
            
            list_sent = dict_human_feats[ii]
            list_lda  = dict_lda[ii]
    
            sent_str=''
            
            for jj, ll in zip(list_sent, list_lda): # subjects
                score_str = ' '.join([kk+':'+ str(jj[1][kk]) for kk in jj[1].iterkeys()])
                lda_str=''
                lda_str += ' |mallet ' + ' '.join(['topic_' + str(kk[0]) + ':' + str(kk[1]) for kk in sorted(ll, key=itemgetter(0))]) 
                sent_str  +=  str(jj[0]['uid']) + " 1 " + str(jj[0]['sent']) + ' |lda ' + score_str + lda_str + '\n'
                
            dict_str[ii]['sent'] = sent_str
    
        return dict_str   

    # This is to use the features that were annotated by humans: 
    # religious_boolean - {0,1} to whether tweet contains a religious expression in any religion.
    # abusive_boolean - {0, 1} to whether tweet contains abusive language (e.g. racism).
    # profanity_boolean - {0, 1} to whether tweet contains profanity.
    # national_role_boolean - {0, 1} to whether tweet calls upon a national role, a WMD-specific sociocultural feature.
    
    def get_feats(self, args):
        
        liwc = LIWC(args.liwc_dict_filename, 'ar')
        d={}
        for ii in glob.iglob(args.input_tsv_regexp):
            print '\n--> Adding casl data from file: %s'%(ii)
            if os.path.basename(ii) == 'gz':
                csvobj = csvwe.UnicodeReader(gzip.open(ii), delimiter=',')   
            else:
                csvobj = csvwe.UnicodeReader(open(ii, 'rb'), delimiter=',')
    
            header=csvobj.next()
            ind_sent=header.index('sent')
            ind_text=header.index('text')   
            ind_uid=header.index('id')
            ind_religious=header.index('religious')
            ind_abusive=header.index('abusive')
            ind_profanity=header.index('profanity')
            ind_national=header.index('national')
            
            for jj in csvobj:
                try:
                    # Sentiment score/label
                    sent=jj[ind_sent].strip()

                    # Text
                    text=jj[ind_text].strip()

                    # Features
                    human_feats={}
                    #human_feats['uid']=jj[ind_uid].strip()
                    human_feats['relig']=jj[ind_religious].strip()
                    human_feats['abus']=jj[ind_abusive].strip()
                    human_feats['prof']=jj[ind_profanity].strip()
                    human_feats['nat']=jj[ind_national].strip()
                    
                    liwc_feats = liwc.count_liwc_words(text)
                    
                    if not d.has_key(os.path.basename(ii)):
                        d[os.path.basename(ii)] = []
                    d[os.path.basename(ii)].append((jj[ind_uid].strip(), human_feats, liwc_feats, {'sent': sent}, text))
                except:
                    print 'Whoops'
                    continue
        return d

if __name__ == '__main__':

    parser = argparse.ArgumentParser( description = 'Generate feature files for classification' )
    parser.add_argument( '--input_tsv_regexp', type = str, dest = 'input_tsv_regexp', 
                         help = 'Regexp with where the tsv files should be found')
    parser.add_argument( '--feat_folder', type = str, dest = 'feat_folder', default='./', 
                         help = 'Folder where features will be dumped')
    parser.add_argument( '--liwc_dict_filename', type = str, dest = 'liwc_dict_filename', 
                          help = 'Pickle with LIWC dictionary')

    
    args = parser.parse_args()

    ffc = FeatureFilesCasl()
    dict_feats = ffc.get_feats(args)
    dict_data_matrices = ffc.create_data_matrices(dict_feats)
    for ii in dict_data_matrices.iterkeys():
        pickle.dump(dict_data_matrices[ii], open(args.feat_folder + '/' + ii + '.sent.pkl','wb'))

    #dict_lda_str = ffc.create_lda_string(dict_feats, 2)

    #for ii in dict_lda_str.iterkeys(): # NOT WORKING
    #    open(args.feat_folder + '/casl-' + ii + '.mallet','wt').write(dict_lda_str[ii].encode('utf-8'))

# WILL CREATE NEW INPUT FILES WITH ARABIC ALREADY TOKENIZED BY PETER SCRIPTS

