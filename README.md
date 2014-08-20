CLPsych
=======
Pipeline to process text with the enhancement of psychology-driven features.

To run the LDA + LIWC + cultural features pipeline

1. Install the following software/dependencies:
> - Python 2.7.x: https://www.python.org/download/releases/2.7/
> - Latest numpy and scipy modules: http://www.scipy.org/scipylib/download.html
> - Latest scikit-learn module: http://scikit-learn.org/stable/install.html
> - Latest nltk: http://www.nltk.org/install.html

2. Open run_pipeline.sh and set the $PYTHON and $PYTHONPATH variables such that the modules above are visible. Also, copy the pre-processed "file.tsv" file to folder input/ and set DATAFILE=$INPUTDIR/"file.tsv"
and setup the output folder $OUTPUTFOLDER to where all the final results will be stored. Alternatively, you can set other parameters within the file, if you are familiar with the used software.

3. Make sure all ArabicPreprocessingScripts, termite-data-server and termite-ui came along with the bundle. 
They should be in the external/ folder. If not use git to pull the remote versions.

4. Go in external/ and run download_others.sh

5. Still in external/ go to the link below, register and download MADAMIRA.
http://innovation.columbia.edu/technologies/cu14012_arabic-language-disambiguation-for-natural-language-processing-applications

6. Still in external/ run setup_madamira.sh

#### To export the best performing LDA model so it can be used with the Tomcat version of ITM:

1. Go back to the base folder, go in src/treeTM/ and run build.sh (ignore warning messages)



