import argparse, os.path, csv_with_encoding as csvwe, codecs
#, re
from shutil import copyfile
from mallet_state_to_itm import export_mallet_states_to_itm
from export_mallet import export_mallet_topics_to_itm
from fake_lhood_file import fake_lhood_file
from subprocess import Popen

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( description = 'Create ITM v.2 files from Mallet LDA topic model' )
    parser.add_argument('--path_itm_software', type = str, dest = 'path_itm_software')
    parser.add_argument('--output_folder', type = str, dest = 'output_folder')
    parser.add_argument('--mallet_state_file', type = str, dest = 'mallet_state_file')
    parser.add_argument('--mallet_topic_file', type = str, dest = 'mallet_topic_file')
    parser.add_argument('--mallet_weight_file', type = str, dest = 'mallet_weight_file')
    parser.add_argument('--mallet_doc_file', type = str, dest = 'mallet_doc_file')
    parser.add_argument('--mallet_input_file', type = str, dest = 'mallet_input_file')
    parser.add_argument('--real_docs', type = str, dest = 'real_docs')
    parser.add_argument('--num_ite', type = int, dest = 'num_ite')
    parser.add_argument('--num_topics', type = int, dest = 'num_topics')
    parser.add_argument('--dataset', type=str, dest='dataset')

    args = parser.parse_args()

    app_folder = '%s/data/%s/'%(args.output_folder, args.dataset)
    if not os.path.exists(app_folder):
        os.makedirs(app_folder)
    model_folder = '%s/model-treetm/'%(app_folder)    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    corpus_folder = '%s/corpus/'%(model_folder)    
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)
    init_folder = '%s/entry-000000/'%(model_folder)
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
        
    doc_ids = [ii.split()[1] for ii in open(args.mallet_doc_file,'r').readlines()[1:]]

    # csvobj has tuples obtained from the original documents csv
    csvobj = csvwe.UnicodeReader(open(args.real_docs,'r'), delimiter=',')
    header = csvobj.next()
    ind_text=header.index('text')   
    ind_uid=header.index('id')
    
    doc_itm2_str=[]
    # Will include only the documents with the given doc_ids
    for ii in csvobj:
        if ii[ind_uid] in doc_ids:
            doc_itm2_str.append("%s\t%s"%(ii[ind_uid], ii[ind_text]))
    doc_itm2_str='\n'.join(doc_itm2_str)

    # Write corpus.txt file
    codecs.open('%s/corpus.txt'%(corpus_folder),'w','utf-8').write(doc_itm2_str)   

    # Write file with hyperparameters, vocabulary and index.json to model folder
    open('%s/tree_hyperparams'%(model_folder),'w').write('DEFAULT_ 0.01\nNL_ 0.01\nML_ 100\nCL_ 0.00000000001')
    open('%s/index.json'%(model_folder),'w').write('{\n "completedEntryID": 0,\n "nextEntryID": 1,\n "numTopics": %d\n}'%(args.num_topics))

    # Copy mallet input file to model folder
    inputfile = '%s/corpus.mallet'%(model_folder)
    copyfile(args.mallet_input_file, inputfile)

    weightfile = '%s/model.topic-words'%(init_folder)
    copyfile(args.mallet_weight_file, weightfile)

    # Convert topic list format from Mallet to ITM and copy to first iteration folder
    topic_str=export_mallet_topics_to_itm(args.mallet_weight_file, 100)
    docs = open(args.mallet_doc_file,'r').read()
    open(init_folder+'/model.docs','w').write(docs.replace('\t',' '))
    codecs.open(init_folder + '/model.topics', 'w','utf-8').write(topic_str)
    open(init_folder + '/states.json', 'w').write('{ "numIters": %d,\n "prevEntryID": -1}'%(args.num_ite))   
    open(init_folder + '/important.keep', 'w')
    open(init_folder + '/constraint.all', 'w')
    open(init_folder + '/removed.all', 'w')
    open(init_folder + '/removed.new', 'w')
    
    # Create empty likelihood entries for the initial iterations run with Mallet
    fake_lhood_file(args.num_ite, init_folder)

    # Convert state files from Mallet to ITM
    [s, vocab] = export_mallet_states_to_itm(args.mallet_state_file)
    # Save converted vocabulary at the model folder
    codecs.open('%s/corpus.voc'%(model_folder),'w','utf-8').write('\n'.join(vocab).decode('utf-8'))

    # Save state file to first iteration folder
    itm_states=open(init_folder+'/model.states','w').write(s) # do not strip s! supposed to have extra tab per line :(
    
    # Create input files in ITM format and save it to the model folder
    cmd='/usr/bin/java -cp %s/tree-TM/class:%s/tree-TM/lib/* '\
        'cc.mallet.topics.tui.GenerateTree --vocab %s '\
        '--tree %s'%(args.path_itm_software, args.path_itm_software, '%s/corpus.voc'%(model_folder), \
                     '%s/corpus.wn'%(init_folder))
     
    Popen(cmd,shell=True).communicate()

    # Run the function that imports the corpus and create the database file
    
    # Run the function that exports the data into the corpus file
    
    
    
    
    
    
    