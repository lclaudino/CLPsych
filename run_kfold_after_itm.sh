# Run the cross-validation experiments, given ITM'd topics
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python
export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/

BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
EXTERNAL=$BASEDIR/external/
OUTPUTDIR=/tmp/clpsych/output/

TARGET_VAR=sent
DATAFILE=$INPUTDIR/wmdtweets_2014.tsv.median # <----- change here

CUL_LIWC_FEAT_PICKLE=/tmp/clpsych/output/feat-files/$( basename $DATAFILE ).sent.pkl
TM_FEAT_PICKLE=$OUTPUTDIR/topic-feats/MalletLDA/$( basename $DATAFILE ).$TARGET_VAR.pkl-to-$( basename $DATAFILE ).$TARGET_VAR.pkl_140_kf_10.pkl \

#########################
# Use ITM model to infer
# features

ITM_MODEL_FOLDER=/Users/claudino/Desktop/collaborations/with-CLIP/2014/summer/vnovak/topic1.r7/ # <----- change here

##########################
# Run k fold cross-valid
# regression experiment

NUM_ITE=1000
ALPHA=100
SEED=11901
BURNIN=100
EXP_TYPE=prediction # <----- change here

$PYTHON $BASEDIR/src/experiments/evaluate.py \
--out_folder=$OUTPUTDIR/results/$( basename $ITM_MODEL_FOLDER ) \
--feature_pkl_file_regexp=$CUL_LIWC_FEAT_PICKLE \
--target=$TARGET_VAR \
--experiment_type=$EXP_TYPE \
--topic_model_pkl=$TM_FEAT_PICKLE \
--topic_model_args=num_ite=$NUM_ITE,burnin=$BURNIN,seed=$SEED \
--mallet_bin_path=$BASEDIR/external/mallet-2.0.7/bin/ \
--fold_path=$ITM_MODEL_FOLDER \
--topic_model_name=ITMTomcat \
--topic_model_path=src/treeTM/

