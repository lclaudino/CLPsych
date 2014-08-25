# Pipeline to process .tsv and display regression/prediction results
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.tsv.madamira.filt # <----- change here
LIWC_PICKLE=$INPUTDIR/licw_arabic_dict.pkl

mkdir -p $OUTPUTDIR/feat-files
mkdir -p $OUTPUTDIR/topic_features
mkdir -p $OUTPUTDIR/results

chmod -R 775 $OUTPUTDIR 

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ # <----- change here

#########################
# Create features

$PYTHON $BASEDIR/src/features/prepare_features.py \
--input_tsv_regexp=$DATAFILE \
--feat_folder=$OUTPUTDIR/feat-files/ \
--liwc_dict_filename=$LIWC_PICKLE

##########################
# Train models for k folds
K=10
NUM_TOPICS=5
NUM_ITE=1000
ALPHA=100
SEED=11901
MODEL=MalletLDA
PATH_TO_MODEL_BIN=$EXTERNAL/mallet-2.0.7/bin/

$PYTHON $BASEDIR/src/experiments/train_topics.py \
--out_folder=$OUTPUTDIR/topic-feats/ \
--feature_pkl_file_regexp=$OUTPUTDIR/feat-files/$( basename $DATAFILE ).sent.pkl \
--bin_path=$PATH_TO_MODEL_BIN \
--k=$K \
--topic_model_args=num_topics=$NUM_TOPICS,num_ite=$NUM_ITE,alpha=$ALPHA,seed=$SEED \
--topic_model_name=$MODEL

##########################
# Run k fold cross-valid
# regression experiment

# obs: some of the parameters are the same defined for training the topics
INFER_NUM_ITE=1000
BURNIN=100
TARGET_VAR=sent
EXP_TYPE=regression

$PYTHON $BASEDIR/src/experiments/evaluate.py \
--out_folder=$OUTPUTDIR/results/ \
--feature_pkl_file_regexp=$OUTPUTDIR/feat-files/$( basename $DATAFILE ).sent.pkl \
--target=$TARGET_VAR \
--experiment_type=$EXP_TYPE \
--topic_model_pkl=$OUTPUTDIR/topic-feats/$MODEL/$( basename $DATAFILE ).sent.pkl-to-$( basename $DATAFILE ).sent.pkl_"$NUM_TOPICS"_kf_"$K".pkl \
--topic_model_args=num_ite=$NUM_ITE,burnin=$BURNIN,seed=$SEED

##########################
# Display results

$PYTHON $BASEDIR/src/experiments/results.py \
--pkl_file=$OUTPUTDIR/results/$MODEL/sent.regression.topics_"$NUM_TOPICS"_k_"$K".pkl --exp_type=$EXP_TYPE


