CLPsych
=======
Pipeline to process text with the enhancement of psychology-driven features.

#### To run the full LDA + LIWC + cultural features pipeline

1) Install the following software/dependencies:
> - Python 2.7.x: https://www.python.org/download/releases/2.7/
> - Latest numpy and scipy modules: http://www.scipy.org/scipylib/download.html
> - Latest scikit-learn module: http://scikit-learn.org/stable/install.html
> - Latest nltk: http://www.nltk.org/install.html

2) Make sure all ArabicPreprocessingScripts, termite-data-server and termite-ui are downloaded into the external/ folder by running "git submodule update".

3) Go in external/ and run download_others.sh

4) Still in external/ go to the link below, register and download MADAMIRA.
http://innovation.columbia.edu/technologies/cu14012_arabic-language-disambiguation-for-natural-language-processing-applications

5) Still in external/ run setup_madamira.sh

6) Open run_pipeline.sh and set the $PYTHON and $PYTHONPATH variables such that the modules above are visible. Also, copy your input csv file to folder input/ and set DATAFILE=$INPUTDIR/"yourfile.csv"
and setup the output folder $OUTPUTFOLDER to where all the final results will be stored. Alternatively, you can set other parameters within the file, if you are familiar with the used software.

7) Go back to the base folder and execute run_pipeline.sh. The results will be shown on the screen and also saved in $OUTPUTFOLDER/results/log.txt

*Note 1: MADAMIRA requires Java 7.

*Note 2: as of now, the only version of MADAMIRA that works is MADAMIRA-release-09232013-1.0-beta-BOLT-MSA-ONLY. I could not get the public version to work yet.*

#### To export the best performing LDA model so it can be used with the Tomcat version of ITM

1) Go in src/treeTM/ and run build.sh to compile the treeTM classes (ignore warning messages).
2) Setup the Tomcat version of ITM by opening, from the base folder, open external/itm-release-install/README.txt and go from there.
3) Look at $OUTPUTFOLDER/results/log.txt that came out of run_pipeline.sh and find out what folder has the best performing results.
4) Copy the path of that folder and edit export_to_itm_tomcat.sh on the base folder.
5) Run export_to_itm_tomcat.sh. This will create a tree structure needed to populate the Tomcat interface of ITM.
6) Copy the contents of the folder data/ and results/ under the folder itm-release/ generated in step 4 to the corresponding folders within the webapps/itm-release/ in the Tomcat home folder.
7) Again, under the Tomcat home folder, open webapps/itm-release/newsession.html and look for:
```
<div class="input">
  <select name="corpus" id="corpus">
  .
  .
  .
  </select>
</div>
```
Then add a new entry to the "select" node corresponding to the exported data. The entry will have the name of the folder under the itm-release/data folder generated in step 4. For example, if that folder is "tweets_topics_5_k_8", you should add an entry like: 

```<option value="tweets_topics_5_k_8">Tweets, topics=5, k=8</option>```

Next, look for:

```
<div class="input">
  <select name="topicsnum" id="topicsnum">
  .
  .
  .
  </select>
</div>
```
and add one or more entries to the select node with numbers of topics the ITM used would want to interact with. For example, to enable 5 topics add: 
```<option>5</option>
```
8) Start the Tomcat server and go to the ITM application URL.

