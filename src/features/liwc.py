from nltk.tokenize import wordpunct_tokenize
import cPickle as pickle, argparse, codecs

class LIWC:

    def __init__(self, dict_pickle_filename, language):
        
        # re-create dict without truncated keys
        if language == 'en':
            d = pickle.load(open(dict_pickle_filename,'rt'))
            #delete 'like' and 'kind' from Pennebaker dictionary (strange values)
            del d['like']
            del d['kind']            
            self.dict = {ii.strip('*'):d[ii] for ii in d.iterkeys()}
            self.cat = range(1,23) + range(121,144) + range(146,151) + range(250,254) + range(354,361) + range(462,465)
        elif language == 'ar':
            self.dict = pickle.load(open(dict_pickle_filename,'rt'))
            self.cat = range(100,109) + [200,201,204,205,206,207,208] + range(300,308) + [310,311,312] + [400,401,402,403,405,406]

    def count_liwc_words(self, multiline_text):
        
        # lowercase and tokenize input text
        words = wordpunct_tokenize(multiline_text.lower())
        
        # initialize counts
        counts={}
        for ii in self.cat:
            counts[str(ii)] = 0;

        # update counts        
        for ii in words:
            if self.dict.has_key(ii): # if observed word in in the dict
                for jj in self.dict[ii]: # every cat for that word has its count incremented
                    counts[jj]+=1

        # normalize counts        
        for ii in counts:
            counts[ii] = float(counts[ii])/len(words)
        
        return counts


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser( description = 'Extract LIWC features from file with text' )
    parser.add_argument( '--input_filename', type = str, dest = 'input_filename', 
                         help = 'File from which LIWC features will be computed')
    parser.add_argument( '--penne_dict_filename', type = str, dest = 'penne_dict_filename', 
                         help = 'Pickle with Pennebaker dictionary')
    parser.add_argument( '--out_folder', type = str, dest = 'out_folder', default='./', 
                         help = 'Folder where outputs will be dumped')

    args = parser.parse_args()

    liwc = LIWC(args.penne_dict_filename, 'en')
    counts = liwc.count_liwc_words(open(args.input_filename).read())

    print counts
    

