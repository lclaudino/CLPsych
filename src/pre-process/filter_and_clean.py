import csv, re, numpy as np, csv, codecs, cStringIO, cPickle as pickle, os, os.path, cfg
from tok import tokenize_ar

def tokenize(ar_text):

    print 'Writing to temp file'
    flat_file_path = os.path.join(cfg.SCRATCH_PATH, str(os.getpid()) + "docs.flat")
    with codecs.open(flat_file_path, "w", "utf-8") as outfile:
        outfile.write(ar_text)
        outfile.write("\n")
    print 'About to run tokenize_ar'
    tokenize_ar(flat_file_path)
    tok_text=""
    with codecs.open(flat_file_path + ".mada","r","utf-8") as infile:
        text = infile.read()
        print text.encode('ascii','xmlcharrefreplace')
        toks = re.findall(r'stem:([^\s]*)', text)
        print "The tokens produced" + " ".join(toks).encode('ascii', 'xmlcharrefreplace')
        tok_text = " ".join(toks)

    return tok_text


class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)



csvobj = UnicodeReader(open('/fs/clip-casl-sentiment/corpora/tweets/wmdtweets_2014-06-09.tsv', 'r'),delimiter='\t')
#pickle.dump(list(csvobj),open('tweets.pkl','w'))
#exit()
header=csvobj.next()

ind_sent=header.index('sentiment')
ind_relev=header.index('relevance')
ind_tweet=header.index('tweet')
ind_tweetid=header.index('tweetid')
ind_file=header.index('tsv_file')
ind_religious=header.index('religious_boolean')
ind_abusive=header.index('abusive_boolean')
ind_profanity=header.index('profanity_boolean')
ind_national=header.index('national_role_boolean')

dict_tweets={}
for ii in csvobj:

    #ii[ind_relev]=re.sub('\s+\t\s+','\t',ii[ind_relev])   
    ii[ind_tweet]=re.sub('\*[^\*]+\*','',ii[ind_tweet])
    ii[ind_tweet]=re.sub('\n','',ii[ind_tweet])
    #ii[ind_tweet]=re.sub('#.+\s','',ii[ind_tweet])

    if not ii[ind_relev] == 'R':
    	continue

    if not len(ii) == len(header):
	continue
    
    try:
        np.float(ii[ind_sent])
    except:
        continue
    
    if not dict_tweets.has_key(ii[ind_tweetid]):
        dict_tweets[ii[ind_tweetid]]=([],[],[],[],[],'')

    l=list(dict_tweets[ii[ind_tweetid]])
    l[0].append(np.float(ii[ind_sent]))
    l[1].append(np.float(ii[ind_religious]))
    l[2].append(np.float(ii[ind_abusive])) 
    l[3].append(np.float(ii[ind_profanity]))
    l[4].append(np.float(ii[ind_national]))
    #l[5]=ii[ind_tweet]
    l[5] = tokenize(ii[ind_tweet])
    dict_tweets[ii[ind_tweetid]]=l

csvobj=UnicodeWriter(open('wmdtweets_2014-06-09.median.tsv.filt', 'w'),delimiter=',')
csvobj.writerow(['id', 'sent', 'religious','abusive','profanity','national','text'])
for ind,ii in enumerate(dict_tweets.iterkeys()):
    csvobj.writerow([ ii, str(round(np.median(dict_tweets[ii][0]))), \
			  str(round(np.median(dict_tweets[ii][1]))), \
			  str(round(np.median(dict_tweets[ii][2]))), \
			  str(round(np.median(dict_tweets[ii][3]))), \
			  str(round(np.median(dict_tweets[ii][4]))), \
			  dict_tweets[ii][5] ])

'''
for ind,ii in enumerate(dict_tweets.iterkeys()):
    csvobj.writerow([ ii, str(np.mean(dict_tweets[ii][0])), \
			  str(np.mean(dict_tweets[ii][1])), \
			  str(np.mean(dict_tweets[ii][2])), \
			  str(np.mean(dict_tweets[ii][3])), \
			  str(np.mean(dict_tweets[ii][4])), \
			  dict_tweets[ii][5] ])
'''

