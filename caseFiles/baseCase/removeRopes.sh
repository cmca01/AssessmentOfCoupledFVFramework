#!/bin/sh                                                                                                                           
cd ${0%/*} || exit 1                        # Run from this directory                                                               
#. $WM_PROJECT_DIR/bin/tools/RunFunctions    # Tutorial run functions

rm -r 0.orig/beamone
rm -r 0.orig/beamtwo

rm -r constant/beamone
rm -r constant/beamtwo

rm -r system/beamone
rm -r system/beamtwo
