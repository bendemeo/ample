library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')


soft_fast = fread('target/experiments/pbmc_softGrid_faster_tests_backup.txt')
soft_fast$lastCounts = gsub("\\[|\\]", "", soft_fast$lastCounts)
soft_fast$lastCounts = as.numeric(soft_fast$lastCounts)

soft_slow = fread('target/experiments/pbmc_softGrid_tests_backup.txt')
soft_slow$lastCounts = gsub("\\[|\\]", "", soft_slow$lastCounts)
soft_slow$lastCounts = as.numeric(soft_slow$lastCounts)

soft_slow$method = rep('exhaustive', nrow(soft_slow))
soft_fast$method = rep('trie', nrow(soft_fast))

soft_all = rbind(soft_fast, soft_slow)

gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')
gs$lastCounts = gsub("\\[|\\]", "", gs$lastCounts)
gs$lastCounts = as.numeric(gs$lastCounts)
gs$method = rep("geometric sketching", nrow(gs))
gs = gs[,-6]

pbmc_all = rbind(soft_all, gs)

ggplot(pbmc_all, aes(x=lastCounts, y=time))+
  geom_line(aes(color=method))

ggplot(soft_all, aes(x=lastCounts, y=max_min_dist))+
  geom_line()
