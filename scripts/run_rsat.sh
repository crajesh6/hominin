#!/bin/bash

save_path="${1}"

rsat matrix-clustering \
-v 1 \
-max_matrices 300 \
-matrix clustered ${save_path}/filters/filters_3_hits.jaspar jaspar \
-hclust_method average \
-calc sum \
-title test \
-metric_build_tree Ncor \
-lth w 5 \
-lth cor 0.6 \
-lth Ncor 0.4 \
-quick \
-label_in_tree name \
-return json,heatmap \
-o ${save_path}/filters/clustered_filters/clustered_filters \
