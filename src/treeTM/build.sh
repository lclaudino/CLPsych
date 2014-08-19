#!/bin/bash

#########################
# Compiling treeTM code

mkdir -p classes
javac -cp lib/*:. \
cc/mallet/topics/*/*.java \
-d classes

