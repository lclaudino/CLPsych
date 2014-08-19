import codecs
from operator import itemgetter

def export_mallet_topics_to_itm (weights, max_words):

	print 'Reading topic word weights'
	# Create tuples grouped by topic
	f=codecs.open(weights,'r','utf-8').readlines()
	d={}
	wei={}
	for ii in f:
		w=ii.split('\t')
		w[1] = w[1].strip()
		w[0] = int(w[0])
		if not d.has_key(w[0]):
			d[w[0]]=[]
			wei[w[0]]=[]
		#	d[w[0]] = {}

		#if not d[w[0]].has_key(w[1]):
		#	d[w[0]][w[1]]={}
		
		#d[w[0]][w[1]]=w[2]	
		d[w[0]].append((w[1],float(w[2])))
		wei[w[0]].append(float(w[2])) 

	s=''
	#for kk in f:
	keys=sorted(d.keys())
	for kk in keys:

		#t = kk.split('\t')
		#topic=t[0]
		#words=t[2].strip().split(' ')
		#words = d[kk].keys()
		words = sorted(d[kk],key=itemgetter(1), reverse=True)

		#print topic[0]

		#s += '\n--------------\nTopic ' + topic + '\n------------------------\n'
		s += '\n--------------\nTopic ' + str(kk) + '\n------------------------\n'
		for ind, ii in enumerate(words):
			if ind >= max_words:
				break
			#s += d[topic][ii].strip() + '\t' + ii + '\n'
			s +=  str(ii[1]/sum(wei[kk])) + '\t' + ii[0].strip() + '\n'
			#s +=  str(ii[1]) + '\t' + ii[0].strip() + '\n'
		
		
	print 'Writing file'
	return s				
	
