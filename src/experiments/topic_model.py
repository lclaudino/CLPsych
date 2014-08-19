import abc, codecs, re
from subprocess import Popen
from operator import itemgetter

class TopicModel:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, bin_path):
        self.bin_path = bin_path

    @abc.abstractmethod
    def train(self, out_folder, ids, data, args):
        ''' Will run training '''
    
    @abc.abstractmethod
    def infer(self, out_folder, is_train, ids, data, args):
        ''' Will run inference '''

class MalletLDA (TopicModel):

    # Instantiate with: random seed to use always, folder where everything will be dumped in, 
    # and the path to mallet's bin
    def __init__(self, mallet_bin_path):
        TopicModel.__init__(self, mallet_bin_path)

    def train(self, out_folder, ids, data, args):

        # Parse arguments string
        num_ite, seed, num_topics, alpha = int(args['num_ite']), \
        int(args['seed']), int(args['num_topics']), float(args['alpha'])
        
        # Prepare data for mallet-import
        data_str=u''
        for ii, jj in zip(ids, data):
            data_str += ii + ' ' + ' '.join(jj) + '\n'
        codecs.open('%s/corpus.txt.train'%out_folder,'w','utf-8').write(data_str)

        cmd =" ".join(["%(binpath)s/mallet import-file", 
        "--input %(outfolder)s/corpus.txt.train",
        "--output %(outfolder)s/corpus.mallet",
        "--keep-sequence",
        "--token-regex '\S+'",
        "--remove-stopwords"])\
        %{'binpath':self.bin_path, 'outfolder':out_folder}
        Popen(cmd,shell=True).communicate()
        
        # Setup and run training here
        cmd=" ".join(["%(bin_path)s/mallet train-topics",
        "--input %(out_folder)s/corpus.mallet",
        "--random-seed %(seed)d",
        "--num-topics %(num_topics)d",
        "--num-iterations %(num_ite)d",
        "--optimize-interval 0",
        "--alpha %(alpha)f",
        "--output-doc-topics %(out_folder)s/doctopics.txt",
        "--output-topic-keys %(out_folder)s/topickeys.txt",
        "--output-state %(out_folder)s/state.mallet.gz",
        "--topic-word-weights-file %(out_folder)s/wordweights.txt",
        "--word-topic-counts-file %(out_folder)s/wordcounts.txt",
        "--inferencer-filename %(out_folder)s/inferencer.mallet"])\
        %{'bin_path':self.bin_path, 'out_folder':out_folder, 'seed':seed,
          'num_topics':num_topics, 'alpha':alpha, 'num_ite': num_ite}
        Popen(cmd,shell=True).communicate()
        
        # Create file with vocabulary
        word_weights = codecs.open("%s/wordweights.txt"%out_folder,'r','utf-8').readlines()
        #vocab=re.findall('\t(.+)\t0',word_weights,re.M)
        vocab = [ii.split()[1] for ii in word_weights[0:len(word_weights)/num_topics]]
        codecs.open("%s/vocab.txt"%out_folder,'w','utf-8').write('\n'.join(vocab))
        
        
    def infer(self, out_folder, is_train, ids, data, args):
        
        out=[]
        
        # Case the data was used for training -- there is a doctopics.txt in out_folder
        if is_train:
            doctopics=open('%s/doctopics.txt'%out_folder,'r').readlines()[1:]
            out=[]
            for ii in doctopics:
                scores = [(int(jj[0]), float(jj[1])) for jj in re.findall('(\d+)\s(\d\.\d+)',ii)]
                out.append(sorted(scores, key=itemgetter(0)))
            return out
        
        # Case the data was not used for training
        # Parse arguments string
        num_ite, burnin, seed = int(args['num_ite']), int(args['burnin']), int(args['seed'])
        
        # Prepare data for mallet-import
        data_str=u''
        for ii, jj in zip(ids, data):
            data_str += ii + ' ' + ' '.join(jj) + '\n'
        codecs.open('%s/corpus.txt.infer'%out_folder,'w','utf-8').write(data_str)

        cmd =" ".join(["%(bin_path)s/mallet import-file", 
        "--input %(out_folder)s/corpus.txt.infer",
        "--output %(out_folder)s/corpus.mallet.infer",
        "--keep-sequence",
        "--token-regex '\S+'",
        "--remove-stopwords",
        "--use-pipe-from %(out_folder)s/corpus.mallet"])\
        %{'bin_path':self.bin_path, 'out_folder':out_folder}
        Popen(cmd,shell=True).communicate()

        cmd = " ".join(["%(bin_path)s/mallet infer-topics",
        "--burn-in %(burnin)d",
        "--random-seed %(seed)d",
        "--inferencer %(out_folder)s/inferencer.mallet",
        "--input %(out_folder)s/corpus.mallet.infer",
        "--num-iterations %(num_ite)d",
        "--output-doc-topics %(out_folder)s/doctopics.txt.infer"])\
        %{'bin_path':self.bin_path, 'out_folder':out_folder, 'seed':seed,
          'burnin':burnin, 'num_ite': num_ite}
        Popen(cmd,shell=True).communicate()

        doctopics=open('%s/doctopics.txt.infer'%out_folder,'r').readlines()[1:]
        for ii in doctopics:
            scores = [(int(jj[0]), float(jj[1])) for jj in re.findall('(\d+)\s(\d\.\d+)',ii)]
            out.append(sorted(scores, key=itemgetter(0)))
            
        return out
