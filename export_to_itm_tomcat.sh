# Exports best k-fold results to ITM v.1.0 (tomcat based)
# Leo Claudino, 08/2014

#########################
# General settings -- these have to match run_pipeline.sh's <-- change here

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.tsv.madamira.filt

#########################
# Conversion settings -- these have to reflect the results that are printed out by run_pipeline.sh <-- change here

BEST_FOLDER=/tmp/code/output/topic-feats/MalletLDA/wmdtweets_2014-06-09.tsv.madamira.filt.sent.pkl/topics-5-k-8/
DEST_FOLDER=/tmp/code/exported/itm-tomcat/itm-release/
NUM_ITE=1000 # must be the same as the one used in train_topics!
NUM_TOPICS=5 # " "
DATASET=tweets # use whatever name you want
BEST_K=8 # inform the folder with the best results based on "run_pipeline.sh"

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

$PYTHON $BASEDIR/src/mallet-to-itm/mallet_to_itm.py \
--mallet_state_file=$BEST_FOLDER/state.mallet.gz \
--mallet_weight_file=$BEST_FOLDER/wordweights.txt \
--mallet_doc_file=$BEST_FOLDER/doctopics.txt \
--mallet_input_file=$BEST_FOLDER/corpus.mallet \
--real_docs=$DATAFILE \
--output_folder=$DEST_FOLDER \
--num_ite=$NUM_ITE \
--num_topics=$NUM_TOPICS \
--dataset="$DATASET"_topics_"$NUM_TOPICS"_k_"$BEST_K" \
--template_html=$BASEDIR/src/mallet-to-itm/template.html
