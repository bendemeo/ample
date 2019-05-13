library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')


pbmc_fb1 = fread('target/experiments/pbmc_fastball_test_2_fulldim_backup.txt')
pbmc_fb2=fread('target/experiments/pbmc_fastball_test_3_fulldim_backup.txt')
pbmc_fb = rbind(pbmc_fb1, pbmc_fb2)

pbmc_fb$method = rep('fastball', nrow(pbmc_fb))

pbmc_pcalsh = fread('target/experiments/pbmc_PCALSH_hausdorff_backup.txt')
pbmc_pcalsh$method = rep('PC-Sketch', nrow(pbmc_pcalsh))

pbmc_gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')
pbmc_gs$lastCounts = as.numeric(gsub("\\[|\\]", "", pbmc_gs$lastCounts))
colnames(pbmc_gs)[4] = 'occSquares'
pbmc_gs$method = rep('gs', nrow(pbmc_gs))
#pbmc_gs$DIMRED = rep('Geometric Sketching', nrow(pbmc_gs))

#pbmc_fb$DIMRED = as.character(pbmc_fb$DIMRED)


colnames(pbmc_fb)[5] = 'gridSize'

pbmc_others = merge(pbmc_gs, pbmc_pcalsh, all=TRUE)
pbmc_all = merge(pbmc_others, pbmc_fb, all=TRUE)

#pbmc_all$DIMRED = as.numeric(pbmc_all$DIMRED)

### make some plots
pbmc_fb %>%
  filter(DIMRED %in% c(2,3,5,10,20,100)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  coord_cartesian(xlim=c(0,10000))+
  geom_line(aes(color=factor(DIMRED)))+
  geom_line(data=pbmc_others, aes(lty=method))

pbmc_all %>%
  ggplot(aes(x=gridSize, y=log(occSquares)))+
  geom_line(aes(color=as.character(DIMRED)))

pbmc_fb %>% filter(occSquares < 25000) %>% ggplot(aes(x=occSquares,y=time))+
  geom_line(aes(color=factor(DIMRED)))+
  geom_line(data=pbmc_others %>% group_by(occSquares, method) %>% summarize(time=mean(time)), aes(lty=method))+
  coord_cartesian(ylim=c(0,500), xlim=c(0,20000))

