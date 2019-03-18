library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

##### Read ######
soft_fast = fread('target/experiments/pbmc_softGrid_faster_tests_backup.txt')
soft_slow = fread('target/experiments/pbmc_softGrid_tests_backup.txt')
fast_ball = fread('target/experiments/pbmc_fastBall_tests_backup.txt')
slow_ball = fread('target/experiments/pbmc_slowBall_tests_backup.txt')
gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')

##### transform and combine ######

# get rid of brackets
soft_fast$lastCounts = gsub("\\[|\\]", "", soft_fast$lastCounts)
soft_fast$lastCounts = as.numeric(soft_fast$lastCounts)

soft_slow$lastCounts = gsub("\\[|\\]", "", soft_slow$lastCounts)
soft_slow$lastCounts = as.numeric(soft_slow$lastCounts)

gs$lastCounts = gsub("\\[|\\]", "", gs$lastCounts)
gs$lastCounts = as.numeric(gs$lastCounts)

fast_ball$lastCounts = gsub("\\[|\\]", "", fast_ball$lastCounts)
fast_ball$lastCounts = as.numeric(fast_ball$lastCounts)

slow_ball$lastCounts = gsub("\\[|\\]", "", slow_ball$lastCounts)
slow_ball$lastCounts = as.numeric(slow_ball$lastCounts)


# add method column
soft_slow$method = rep('exhaustive', nrow(soft_slow))
soft_fast$method = rep('trie', nrow(soft_fast))
fast_ball$method = rep('Fastball', nrow(fast_ball))
gs$method = rep("geometric sketching", nrow(gs))
slow_ball$method = rep('Slowball', nrow(slow_ball))

#delete unnecessary columns and rename cols as necessary
gs = gs[,-6]
fast_ball = fast_ball[,-6]
colnames(slow_ball)[5] <- "gridSize"


#combine data
pbmc_all = rbind(soft_fast, soft_slow, fast_ball, slow_ball, gs)
soft_all = rbind(soft_fast, soft_slow)

# melt data
cell_types = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
               "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
               "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
               "CD8+_Cytotoxic_T","Dendritic")


pbmc_all = melt(pbmc_all, id.vars=c('max_min_dist', 'time', 'method', 'lastCounts'), 
               measure.vars=cell_types)


g1 = ggplot(pbmc_all, aes(x=lastCounts, y=time/60))+
  geom_line(aes(color=method))

g2 = ggplot(pbmc_all, aes(x=lastCounts, y=max_min_dist))+
  geom_line(aes(color=method))

pbmc_all %>% filter(lastCounts < 2000) %>%ggplot( aes(x=lastCounts, y=value, color=variable))+
  facet_wrap(~method)+
  geom_line()

soft_all %>% ggplot(aes(x=gridSize, y=lastCounts))+
  geom_line()

gs %>% ggplot(aes(x=gridSize, y=lastCounts))+
  geom_line()
