#!/bin/bash

CFG_PATH=$(pwd)/ArabicPreprocessingScripts/cfg.py
MADA_PATH=$(pwd)/$(ls | grep MADAMIRA)
MADA_JAR=$(ls $MADA_PATH/*.jar) 
MADA_CFG=$MADA_PATH/config/tok.xml
SCRATCH_PATH=/tmp/scratch

mkdir -p $SCRATCH_PATH
echo XML_INPUT_PATH = \"\"> $CFG_PATH
echo SCRATCH_PATH = \"$SCRATCH_PATH\" >> $CFG_PATH
echo ITM_HTML_TEMPLATE_PATH = \"\" >> $CFG_PATH
echo MADA_PATH = \"$MADA_PATH\" >> $CFG_PATH
echo MADA_JAR = \"$MADA_JAR\" >> $CFG_PATH
echo MADA_CFG = \"$MADA_CFG\" >> $CFG_PATH
echo MALLET_BIN_PATH = \"\" >> $CFG_PATH

echo $MADA_CFG

rm -f $MADA_CFG
echo "<?xml version=\"1.0\" encoding=\"utf-8\"?>" >> $MADA_CFG  
echo "<!-- DOCTYPE madamira_configuration SYSTEM \"$MADA_PATH/resources/schema/MADAMIRA.dtd\" -->" >> $MADA_CFG
echo "<madamira_configuration xmlns=\"urn:edu.columbia.ccls.madamira.configuration:0.1\">" >>  $MADA_CFG
echo "    <preprocessing sentence_ids=\"false\" separate_punct=\"true\" input_encoding=\"UTF8\"/>" >> $MADA_CFG
echo "    <overall_vars output_encoding=\"UTF8\" dialect=\"MSA\" output_analyses=\"TOP\" morph_backoff=\"NONE\" analyze_only=\"false\"/>" >> $MADA_CFG
echo "    <tokenization>" >> $MADA_CFG
echo "        <scheme alias=\"ATB\"/>" >> $MADA_CFG
echo "    </tokenization>" >> $MADA_CFG
echo "</madamira_configuration>" >> $MADA_CFG


