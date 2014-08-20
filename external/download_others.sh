#!/bin/bash

# Download Mallet
wget http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz
tar -zxvf mallet-2.0.7.tar.gz
rm mallet-2.0.7.tar.gz

# Download the Tomcat version of ITM
wget http://www.umiacs.umd.edu/~claudino/summer-2014/itm-release-install.tar.gz
tar -zxvf itm-release-install.tar.gz
rm itm-release-install.tar.gz

# Prompt user to download MADAMIRA
echo Now, please download MADAMIRA and unzip it into this folder
echo It requires registration. As of now, it can be downloaded from:
echo http://innovation.columbia.edu/technologies/cu14012_arabic-language-disambiguation-for-natural-language-processing-applications
echo Then run setup_madamira.sh

