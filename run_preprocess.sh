# Pre-process data with Madamira
# Leo Claudino, 08/2014

#########################
# General settings

PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=/tmp/clpsych/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
DATAFILE=$INPUTDIR/wmdtweets_2014.tsv # <----- change here

echo $DATAFILE

export PYTHONPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages:\
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/:\
$EXTERNAL/ArabicPreprocessingScripts/ # <----- change here

python src/pre-process/filter_and_clean.py --filename=$DATAFILE --combine_annotations=mean --out_folder=$OUTPUT_DIR
