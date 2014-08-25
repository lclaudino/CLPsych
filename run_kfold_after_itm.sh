# Run the cross-validation experiments, given ITM'd topics
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.tsv.madamira.filt # <----- change here

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ # <----- change here

##########################
# Run k fold cross-valid
# regression experiment

K=10
NUM_TOPICS=140
NUM_ITE=1000
ALPHA=100
SEED=11901
TARGET_VAR=sent
EXP_TYPE=regression
BURNIN=100

#########################
FOLD_PATH=/Users/claudino/Desktop/collaborations/with-CLIP/2014/summer/vnovak/topic33.r19/ # <----- change here

BEST_FOLD=5 # <----- change here
MODEL=ITMTomcat
MODEL_PATH=src/treeTM/
PATH_TO_MALLET_BIN=$EXTERNAL/mallet-2.0.7/bin/

$PYTHON $BASEDIR/src/experiments/evaluate.py \
--out_folder=$OUTPUTDIR/results/$(basename $FOLD_PATH) \
--feature_pkl_file_regexp=$OUTPUTDIR/feat-files/$( basename $DATAFILE ).sent.pkl \
--target=$TARGET_VAR \
--experiment_type=$EXP_TYPE \
--topic_model_pkl=$OUTPUTDIR/topic-feats/MalletLDA/$( basename $DATAFILE ).sent.pkl-to-$( basename $DATAFILE ).sent.pkl_"$NUM_TOPICS"_kf_"$K".pkl \
--topic_model_args=num_ite=$NUM_ITE,burnin=$BURNIN,seed=$SEED \
--fold=$BEST_FOLD \
--fold_path=$FOLD_PATH \
--topic_model_name=$MODEL \
--topic_model_path=$MODEL_PATH \
--mallet_bin_path=$PATH_TO_MALLET_BIN

