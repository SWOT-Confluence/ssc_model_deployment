# ssc_model_deployment
Module for input, preprocessing, and deployment of the CV and SSC based models

## Deployment plan
<!-- 
* Static file of a combined vector of sword reaches by continent
* datagen pulls links of delta of overlapping hls tiles since last sos and stores in a json
* paralelize by hls tile
* predict on all reaches and nodes covered by tile
* store parameters in timeseries files -->


* datagen goes through reaches and finds tiles for each of the lines within date range
* stores them in a input json
* process_ssc reads in this and uses the array index to pick out a tile
* checks tile for overlap with nodes
* predicts on nodes in tile
* writes to tile level efs
* ssc_combine module to coalate the results
