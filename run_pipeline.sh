# Pipeline to process .tsv and display regression/prediction results
# Leo Claudino, 08/2014

#########################
# General settings

#PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
PYTHON=/usr/bin/python2.7
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=$BASEDIR/output # <----- change here
EXTERNAL=$BASEDIR/external/
#DATAFILE=$INPUTDIR/wmdtweets_2014-06-09.tsv.madamira.filt # <----- change here
#DATAFILE=$INPUTDIR/train_statuses.csv # <----- change here
DATAFILE=$INPUTDIR/train_statuses.csv
LIWC_PICKLE=$INPUTDIR/penne_dict.p
#LIWC_PICKLE=$INPUTDIR/licw_arabic_dict.pkl

mkdir -p $OUTPUTDIR/feat-files
mkdir -p $OUTPUTDIR/topic_features
mkdir -p $OUTPUTDIR/results

chmod -R 775 $OUTPUTDIR 

export PYTHONPATH=/usr/bin/python2.7:/usr/bin/python2.7-config:/usr/bin/python:/etc/python2.7:/etc/python:/usr/lib/python2.7:/usr/bin/X11/python2.7:/usr/bin/X11/python2.7-config:/usr/bin/X11/python:/usr/local/lib/python2.7:/usr/include/python2.7:/usr/include/python2.7_d:/usr/share/python:/usr/share/man/man1/python.1.gz

#export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
#/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ # <----- change here

#########################
# Create features

$PYTHON $BASEDIR/src/features/prepare_features.py \
--input_tsv_regexp=$DATAFILE \
--feat_folder=$OUTPUTDIR/feat-files/ \
--liwc_dict_filename=$LIWC_PICKLE \
--prediction="avg" \
--id="userid"
#--feats="q11,q12,q30,q37"
#--feats="religious,abusive,profanity,national"

##########################
# Train models for k folds
K=10
NUM_TOPICS=5
NUM_ITE=300
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
TARGET_VAR=avg
EXP_TYPE=regression

$PYTHON $BASEDIR/src/experiments/evaluate.py \
--out_folder=$OUTPUTDIR/results/ \
--feature_pkl_file_regexp=$OUTPUTDIR/feat-files/$( basename $DATAFILE ).sent.pkl \
--target=$TARGET_VAR \
--experiment_type=$EXP_TYPE \
--topic_model_pkl=$OUTPUTDIR/topic-feats/$MODEL/$( basename $DATAFILE ).sent.pkl-to-$( basename $DATAFILE ).sent.pkl_"$NUM_TOPICS"_kf_"$K".pkl \
--topic_model_args=num_ite=$NUM_ITE,burnin=$BURNIN,seed=$SEED \
--mallet_bin_path=$PATH_TO_MODEL_BIN

##########################
# Display results

$PYTHON $BASEDIR/src/experiments/results.py \
--pkl_file=$OUTPUTDIR/results/$MODEL/$TARGET_VAR.$EXP_TYPE.topics_"$NUM_TOPICS"_k_"$K".pkl --exp_type=$EXP_TYPE


