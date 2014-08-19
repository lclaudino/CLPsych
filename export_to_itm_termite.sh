#!/bin/bash

# Exports best k-fold results to ITM v.2.0 (termite-based)
# NOTE: this will assume termite-data-server and termite-ui already
# installed
#
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python
BASEDIR=/tmp/code/
INPUTDIR=$BASEDIR/input/
DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.tsv.madamira.filt

#########################
# Compiling treeTM code

mkdir -p $BASEDIR/external/treeTM/classes
javac -cp $BASEDIR/external/treeTM/lib/*:. \
$BASEDIR/external/treeTM/cc/mallet/topics/*/*.java \
-d $BASEDIR/external/treeTM/classes

#########################
# Conversion settings

BEST_FOLDER=/tmp/code/output/topic-feats/MalletLDA/wmdtweets_2014-06-09.tsv.madamira.filt.sent.pkl/topics-5-k-8/
PATH_TREETM=$BASEDIR/external/
NUM_ITE=1000 # must be the same as the one used in train_topics!
NUM_TOPICS=5 # " "
DATASET="tweets_"$(basename $BEST_FOLDER)
EXP_FOLDER=/tmp/code/output/exported/itm-termite/

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

rm -rf $EXP_FOLDER/data/$DATASET
$PYTHON $BASEDIR/src/mallet-to-itm/mallet_to_itm2.py \
--mallet_state_file=$BEST_FOLDER/state.mallet.gz \
--mallet_weight_file=$BEST_FOLDER/wordweights.txt \
--mallet_doc_file=$BEST_FOLDER/doctopics.txt \
--mallet_input_file=$BEST_FOLDER/corpus.mallet \
--real_docs=$DATAFILE \
--output_folder=$EXP_FOLDER \
--num_ite=$NUM_ITE \
--num_topics=$NUM_TOPICS \
--dataset=$DATASET \
--path_itm_software=$PATH_TREETM

#########################
# Importing into server

CURR_DIR=$(pwd)

cd $BASEDIR/external/termite-data-server
bin/import_corpus.py \
$EXP_FOLDER/data/$DATASET/model-treetm/corpus/ \
$EXP_FOLDER/data/$DATASET/model-treetm/corpus/corpus.txt 

bin/read_treetm.py $DATASET \
$EXP_FOLDER/data/$DATASET/model-treetm/ \
$EXP_FOLDER/data/$DATASET/model-treetm/corpus/ \
$EXP_FOLDER/data/$DATASET/model-treetm/corpus/ \
--overwrite

cd $CURR_DIR

#########################
# Prepare services.js in
# the termite ui

ITER_INCREMENT=20
SERVER=http://localhost:8075
ORIGIN=http://localhost:8000

python $BASEDIR/src/mallet-to-itm/setup_termite_ui_services.py \
--server=$SERVER \
--origin=$ORIGIN \
--path_to_services=$BASEDIR/external/termite-ui/public_html/app/js/ \
--iter_increment=$ITER_INCREMENT \
--iter_count=$NUM_ITE

#########################
# Prepare controllers.js in
# the termite ui

python $BASEDIR/src/mallet-to-itm/setup_termite_ui_controllers.py \
--path_to_controllers=$BASEDIR/external/termite-ui/public_html/app/js/ \
--app_name=$DATASET

