import sys
from convert_mallet_state import MalletStateBuffer
from itertools import chain
from math import log

#def export_mallet_states_to_itm(filename, out_folder):
def export_mallet_states_to_itm(filename):
    
    msb = MalletStateBuffer(filename)

    master_vocab={}
    doc_counts={}
    freq={}    
    n_docs=0
    s=u''

    for doc in msb:
        assig = [ii for ii in doc]

        # Compute current doc's contribution to vocab
        vocab = dict((ii.term_id, ii.term) for ii in assig)

        # Add current document to the word's document occurence count and increment word frequency
        for ii in vocab.iteritems():
            if not doc_counts.has_key(ii[1]):
                doc_counts[ii[1]]=[]
                freq[ii[1]]=0
            doc_counts[ii[1]].append(n_docs)
            freq[ii[1]]+=1

        # Merge current vocab with next document's words
        master_vocab=dict(chain(master_vocab.iteritems(), vocab.iteritems()))
        
        word_topics = [str(ii.assignment) for ii in assig]    
        s+=':0\t'.join(word_topics) + ':0\t\n'

        n_docs+=1

        if n_docs%100 == 0:
            print n_docs
            sys.stdout.flush()
    
    #if not os.path.exists(out_folder+'/resume/'):
    #        os.makedirs(out_folder+'/resume/')

    #itm_input=open(out_folder+'/model.states','w')
    #itm_input.write(s.strip())
    #itm_input.close()

    # Sort vocabulary per term_id and print term, tfidf and frequency
    vocab = ['0\t'+ii[1] + '\t' + \
    str( freq[ii[1]] * ( log(n_docs) - log(len(doc_counts[ii[1]])) ) ) + \
    '\t' + str(freq[ii[1]]) for ii in sorted(master_vocab.iteritems(), key=vocab.get)]

    return s, vocab
