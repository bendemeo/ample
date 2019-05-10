library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')


pbmc_fb = fread('target/experiments/pbmc_fastball_test_backup.txt')
pbmc_fb$method = rep('fastball', nrow(pbmc_fb))

pbmc_gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')
pbmc_gs$lastCounts = as.numeric(gsub("\\[|\\]", "", pbmc_gs$lastCounts))
colnames(pbmc_gs)[4] = 'occSquares'
pbmc_gs$method = rep('gs', nrow(pbmc_gs))

colnames(pbmc_fb)[5] = 'gridSize'

pbmc_all = merge(pbmc_gs, pbmc_fb, all=TRUE)

### make some plots
pbmc_all %>% ggplot(aes(x=occSquares, y=max_min_dist))+
  geom_line(aes(color=method))

pbmc_all %>% ggplot(aes(x=occSquares,y=time))+
  geom_line(aes(color=method))
