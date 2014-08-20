#!/bin/bash

CFG_PATH=$(pwd)/ArabicPreprocessingScripts/cfg.py
MADA_PATH=$(pwd)/$(ls | grep MADAMIRA)
MADA_JAR=$(pwd)$(ls $MADA_PATH/*.jar) 
MADA_CFG=$MADA_PATH/config/madamira.xml
SCRATCH_PATH=/tmp/scratch

mkdir -p $SCRATCH_PATH
echo XML_INPUT_PATH = \"\"> $CFG_PATH
echo SCRATCH_PATH = \"$SCRATCH_PATH\" >> $CFG_PATH
echo ITM_HTML_TEMPLATE_PATH = \"\" >> $CFG_PATH
echo MADA_PATH = \"$MADA_PATH\" >> $CFG_PATH
echo MADA_JAR = \"$MADA_JAR\" >> $CFG_PATH
echo MADA_CFG = \"$MADA_CFG\" >> $CFG_PATH
echo MALLET_BIN_PATH = \"\" >> $CFG_PATH
