# Pipeline to process .tsv and display regression/prediction results
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
DATAFILE=$INPUTDIR/wmdtweets_2014.tsv.median # <----- change here
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

ls -la $OUTPUTDIR/feat-files/
