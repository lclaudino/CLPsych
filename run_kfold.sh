# Run the cross-validation experiments, given LDA models computed with run_lda.sh
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
#DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.median.tsv.madamira.filt
DATAFILE=$INPUTDIR/wmdtweets_2014.tsv.median # <----- change here

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ # <----- change here

##########################
# Run k fold cross-valid
# regression experiment

K=10
NUM_TOPICS=5
NUM_ITE=1000
ALPHA=100
SEED=11901
MODEL=MalletLDA
PATH_TO_MODEL_BIN=$EXTERNAL/mallet-2.0.7/bin/
TARGET_VAR=sent
EXP_TYPE=prediction
BURNIN=100

$PYTHON $BASEDIR/src/experiments/evaluate.py \
--out_folder=$OUTPUTDIR/results/ \
--feature_pkl_file_regexp=$OUTPUTDIR/feat-files/$( basename $DATAFILE ).sent.pkl \
--target=$TARGET_VAR \
--experiment_type=$EXP_TYPE \
--topic_model_pkl=$OUTPUTDIR/topic-feats/$MODEL/$( basename $DATAFILE ).sent.pkl-to-$( basename $DATAFILE ).sent.pkl_"$NUM_TOPICS"_kf_"$K".pkl \
--topic_model_args=num_ite=$NUM_ITE,burnin=$BURNIN,seed=$SEED \
--mallet_bin_path=$PATH_TO_MODEL_BIN


