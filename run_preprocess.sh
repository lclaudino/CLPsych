# Pre-process data with Madamira
# Leo Claudino, 08/2014

#########################
# General settings


PYTHON=/opt/local/stow/python-2.7.2/bin/python # <----- change here
BASEDIR=$(pwd)
INPUTDIR=$BASEDIR/input/
OUTPUTDIR=$BASEDIR/output/ # <----- change here
EXTERNAL=$BASEDIR/external/
DATAFILE=$INPUTDIR/wmdtweets_2014.tsv # <----- change here
LIWC_PICKLE=$INPUTDIR/licw_arabic_dict.pkl
COMBINE_ANNOTATIONS=median

mkdir -p $OUTPUTDIR
export PYTHONPATH=$PYTHONPATH:$EXTERNAL/ArabicPreprocessingScripts/ # <----- change here

$PYTHON src/pre-process/filter_and_clean.py --filename=$DATAFILE \
--combine_annotations=$COMBINE_ANNOTATIONS --out_folder=$OUTPUTDIR
