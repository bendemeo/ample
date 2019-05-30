library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
library(ggfortify)

setwd('~/Documents/bergerlab/lsh/ample/')

pbmc = fread('data/pbmc_dimred')
labels = fread('data/pbmc_labels')[-1,]

pbmc = cbind(labels[,-1], pbmc)
colnames(pbmc)[1] = 'type'

pbmc_means = pbmc %>% group_by(type) %>% summarize_all(mean)
rownames(pbmc_means) = pbmc_means$type
p = prcomp(pbmc_means[,-1])


dist_euclidean = function(x,y){
  sqrt(sum((x-y)^2))  
}



d = dist_euclidean
for(label in pbmc_means$type) {
  dists = apply(pbmc[,-1],1,function(x)dist_euclidean(x,pbmc_means[label,-1]))
}

