import argparse, os.path, csv_with_encoding as csvwe, codecs
#, re
from shutil import copyfile
from mallet_state_to_itm import export_mallet_states_to_itm
from export_mallet import export_mallet_topics_to_itm
from fake_lhood_file import fake_lhood_file
#from subprocess import Popen

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( description = 'Create ITM files from Mallet LDA topic model' )
    parser.add_argument('--output_folder', type = str, dest = 'output_folder')
    parser.add_argument('--mallet_state_file', type = str, dest = 'mallet_state_file')
    parser.add_argument('--mallet_topic_file', type = str, dest = 'mallet_topic_file')
    parser.add_argument('--mallet_weight_file', type = str, dest = 'mallet_weight_file')
    parser.add_argument('--mallet_doc_file', type = str, dest = 'mallet_doc_file')
    parser.add_argument('--mallet_input_file', type = str, dest = 'mallet_input_file')
    #parser.add_argument('--doc_ids', type = str, dest = 'doc_ids')
    parser.add_argument('--real_docs', type = str, dest = 'real_docs')
    parser.add_argument('--num_ite', type = int, dest = 'num_ite')
    parser.add_argument('--num_topics', type = int, dest = 'num_topics')
    parser.add_argument('--dataset', type=str, dest='dataset')
    parser.add_argument('--template_html', type=str, dest='template_html')

    args = parser.parse_args()
    init_folder = '%s/results/%s/output/T%d/init/'%(args.output_folder, args.dataset, args.num_topics) 
    input_folder= '%s/results/%s/input/'%(args.output_folder, args.dataset)
    data_folder = '%s/data/'%(args.output_folder)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Get doc ids and save model.docs file
    #copyfile(args.mallet_doc_file, docfile)
    doc_ids = [ii.split('\t')[1] for ii in open(args.mallet_doc_file,'r').readlines()[1:]]
    docs = open(args.mallet_doc_file,'r').read()
    open(init_folder+'/model.docs','w').write(docs.replace('\t',' '))

    # csvobj has tuples obtained from the original documents csv
    csvobj = csvwe.UnicodeReader(open(args.real_docs,'r'), delimiter=',')
    header = csvobj.next()
    ind_text=header.index('text')   
    ind_uid=header.index('id')
    html_str=[]
    # Update html template file with documents and ids --> this is very inefficient!
    # Will include only the documents with the given doc_ids

    for ii in csvobj:
        if ii[ind_uid] in doc_ids:
            html_str.append('<div class="segment" id="%s"><p>%s</p></div>'%(ii[ind_uid],ii[ind_text]))
    html_str='\n'.join(html_str)
    html_str+='\n</main>\n</body>\n</html>\n'

    # Write .html file
    html_str=open(args.template_html).read()+html_str
    codecs.open('%s/%s.html'%(data_folder,args.dataset),'w','utf-8').write(html_str)   

    '''
    # Read Mallet documents file here and replace second column with given ids

    doc_topic_rows = open(args.mallet_doc_file,'r').readlines()
    doc_topics_str=[]
    for ind, ii in enumerate(doc_topic_rows[1:]):
        rows=ii.split()
        rows[1] = doc_ids[ind]
        doc_topics_str.append(' '.join(rows))
    doc_topics_str = '\n'.join(doc_topics_str)
    
    itm_docs=open(init_folder+'/model.docs','w')
    itm_docs.write('#doc source topic proportion ...\n' +  doc_topics_str)    
    '''

    # Create .url file
    url = ['%s /data/%s.html#%s'%(ii,args.dataset,ii) for ii in doc_ids]
    urlfile = '%s/%s.url'%(input_folder,args.dataset) 
    open(urlfile,'w').write('\n'.join(url))    
    
    # Copy mallet input file
    inputfile = '%s/%s-topic-input.mallet'%(input_folder,args.dataset)
    copyfile(args.mallet_input_file, inputfile)

    # Convert topic list format from Mallet to ITM
    topic_str=export_mallet_topics_to_itm(args.mallet_weight_file, 100)
    codecs.open(init_folder + '/model.topics', 'w','utf-8').write(topic_str)

    # Create empty likelihood entries for the initial iterations run with Mallet
    fake_lhood_file(args.num_ite, init_folder)

    # Convert state files from Mallet to ITM
    [s, vocab] = export_mallet_states_to_itm(args.mallet_state_file)
    itm_states=open(init_folder+'/model.states','w').write(s) # do not strip s! supposed to have extra tab per line :(

    # Save converted vocabulary
    codecs.open('%s/%s.voc'%(input_folder, args.dataset),'w','utf-8').write('\n'.join(vocab).decode('utf-8'))
    
    # Write file with hyperparameters
    open('%s/tree_hyperparams'%(input_folder),'w').write('DEFAULT_ 0.01\nNL_ 0.01\nML_ 100\nCL_ 0.00000000001')


    #CHANGE TO CREATE AN INIT FOLDER WITH ESSENTIAL FILES. SHOULD INCLUDE THE DOCTOPICS FROM MALLET AS MODEL.DOCS
    #DATASET/OUTPUT/T#TOPICS/INIT/MODEL.*

    #MAY HAVE TO SETUP INPUT AS WELL with DATASET.VOC, DATASET.URL DATASET.TOPIC-INPUT-MALLET AND TREE_HYPERPARAMS

    '''
    cmd='/usr/bin/java -cp %s/tree-TM/class:%s/tree-TM/lib/* '\
        'cc.mallet.topics.tui.GenerateTree --vocab %s '\
        '--tree %s'%(args.path_itm, args.path_itm, '%s/%s.voc'%(input_folder, args.dataset), \
                     '%s/%s.wn'%(init_folder, args.dataset))
     
    Popen(cmd,shell=True).communicate()
    '''

    
