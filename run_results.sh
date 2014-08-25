# Show overall results
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ # <----- change here

BASEDIR=$(pwd)
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here

#MODEL=MalletLDA # <----- change here
MODEL=ITMTomcat # <----- change here

NUM_TOPICS=140
EXP_TYPE=regression
K=10

PKL_FILE=$OUTPUTDIR/results/topic33.r19/ITMTomcat/sent.regression.topics_140_k_10_fold_5.pkl #<----- change here
echo $PKL_FILE

echo
$PYTHON $BASEDIR/src/experiments/results.py \
--pkl_file=$PKL_FILE --exp_type=$EXP_TYPE | tee $OUTPUTDIR/results/log.txt


