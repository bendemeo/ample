library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

# read geometric sketching data
pbmc_gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')
pbmc_gs$lastCounts = as.numeric(gsub("\\[|\\]", "", pbmc_gs$lastCounts))
colnames(pbmc_gs)[4] = 'occSquares'
pbmc_gs$method = rep('gs', nrow(pbmc_gs))
pbmc_gs$N = as.numeric(pbmc_gs$occSquares)


# read fastball data
pbmc_fb1 = fread('target/experiments/pbmc_fastball_test_2_fulldim_backup.txt')
pbmc_fb2=fread('target/experiments/pbmc_fastball_test_3_fulldim_backup.txt')
pbmc_fb = rbind(pbmc_fb1, pbmc_fb2)
pbmc_fb$PCA = rep(FALSE, nrow(pbmc_fb))
pbmc_fb$method = rep('fastball', nrow(pbmc_fb))
pbmc_fb$N = as.numeric(pbmc_fb$occSquares)


# read PCALSH data
pbmc_pcalsh = fread('target/experiments/pbmc_PCALSH_hausdorff_backup.txt')
pbmc_pcalsh$method = rep('PC-Sketch', nrow(pbmc_pcalsh))
pbmc_pcalsh$N = as.numeric(pbmc_pcalsh$occSquares)

# pbmc_adaptive = fread('target/experiments/pbmc_fastball_PCA_adaptive_backup.txt')
# pbmc_adaptive$method = rep('fastball_adapt',nrow(pbmc_adaptive))


# read FT data
pbmc_ft = fread('target/experiments/pbmc_ft_backup.txt')
pbmc_ft$method=rep('Far traversal',nrow(pbmc_ft))

#read refined FT data
pbmc_ft_refined = fread('target/experiments/pbmc_ft_refined_backup.txt')
pbmc_ft_refined$method = rep('Refined far traversal',nrow(pbmc_ft_refined))




pbmc_f = merge(pbmc_ft, pbmc_fb, all=TRUE)
pbmc_others = merge(pbmc_gs, pbmc_pcalsh, all=TRUE)

pbmc_temp = merge(pbmc_f, pbmc_others, all=TRUE)
pbmc_all = merge(pbmc_temp, pbmc_ft_refined, all=TRUE)
# pbmc_fb = merge(pbmc_adaptive, pbmc_fb, all=TRUE)


pbmc_all %>% filter(DIMRED %in% c(100, NA)) %>%
  ggplot(aes(x=N, y=max_min_dist))+
  geom_line(aes(color=method))

pbmc_all %>% filter(time<300) %>% group_by(method, N, DIMRED, gridSize) %>% summarize(time=mean(time)) %>%
  ggplot(aes(x=N, y=time))+
  geom_line(aes(color=factor(DIMRED), lty=method))

pbmc_all %>% filter( method=='Far traversal') %>% mutate(dimension = (log(N))/log((1/max_min_dist))) %>%
  ggplot(aes(x=max_min_dist, y=dimension))+
  geom_line()



#pbmc_gs$DIMRED = rep('Geometric Sketching', nrow(pbmc_gs))

#pbmc_fb$DIMRED = as.character(pbmc_fb$DIMRED)


mb_fb=fread('target/experiments/mouse_brain_fastball_fulldim_backup.txt')

colnames(pbmc_fb)[5] = 'gridSize'

pbmc_others = merge(pbmc_gs, pbmc_pcalsh, all=TRUE)
pbmc_all = merge(pbmc_others, pbmc_fb, all=TRUE)
pbmc_all = merge(pbmc_ft, pbmc_all, all=TRUE)

#pbmc_all$DIMRED = as.numeric(pbmc_all$DIMRED)

### make some plots

pbmc_all %>% ggplot(aes(x=gridSize, y=occSquares))+
  geom_line(aes(color=factor(DIMRED)))

pdf('~/Documents/bergerlab/6.890/mouse_brain_runtimes.pdf',8,3.6)
mb_fb %>% 
  ggplot(aes(x=occSquares, y=time))+
  geom_line(aes(color=factor(DIMRED)))+
  geom_line(data=mouse_brain, aes(x=occSquares, y=time, lty=method))+
  scale_linetype_discrete(labels=c('Geometric Sketch','PC-Sketch'))+
  coord_cartesian(xlim=c(0,7500))+
  ggtitle('Mouse Brain Data')+
  xlab('Sketch size (out of 690K)')+
  ylab('Runtime (seconds)')+
  guides(color=guide_legend(ncol=2))+
  scale_color_discrete(name='Fastball VP-tree\nbuilding dimension')
dev.off()


p1 = pbmc_fb %>%
  filter(DIMRED %in% c(2,3,5,10,20,100)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  coord_cartesian(xlim=c(0,10000))+
  geom_line(aes(color=factor(DIMRED), lty=method))+
  geom_line(data=pbmc_others, aes(lty=method))+
  xlab('Sketch size (out of 68K)')+
  ylab('Hausdorff distance')+
  scale_linetype_discrete(name='Other methods')+
  scale_color_discrete(name='FastBall VP-tree \nbuilding dimension')+
  ggtitle('PBMC data Hausdorff distances')

pdf('~/Documents/bergerlab/6.890/pbmc_fastball_hausdorff.pdf',9,3)
pbmc_fb %>%
  filter(DIMRED %in% c(2,3,5,10,20,100)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  coord_cartesian(xlim=c(0,10000))+
  geom_line(aes(color=factor(DIMRED), lty=method))+
  geom_line(data=pbmc_others, aes(lty=method))+
  xlab('Sketch size (out of 68K)')+
  ylab('Hausdorff distance')+
  scale_linetype_discrete(labels=c('Geometric Sketch','PC-Sketch'))+
  scale_color_discrete(name='FastBall VP-tree \nbuilding dimension')+
  ggtitle('PBMC data Hausdorff distances')+
  guides(color=guide_legend(ncol=2))
dev.off()

pbmc_all %>%
  ggplot(aes(x=gridSize, y=log(occSquares)))+
  geom_line(aes(color=as.character(DIMRED), lty=method))



p2 = pbmc_fb %>% filter(occSquares < 25000) %>% 
  filter(DIMRED %in% c(2,3,5,10,20,100)) %>%
  ggplot(aes(x=occSquares,y=time))+
  geom_line(aes(color=factor(DIMRED), lty=method))+
  geom_line(data=pbmc_others %>% group_by(occSquares, method) %>% summarize(time=mean(time)), aes(lty=method))+
  coord_cartesian(ylim=c(0,500), xlim=c(0,20000))+
  scale_linetype_discrete(name='Other methods',labels=c('Geometric Sketch','PC-Sketch'))+
  scale_color_discrete(name='FastBall VP-tree \nbuilding dimension')+
  ggtitle('PBMC Data Runtimes')+
  xlab('Sketch size (out of 68K)')+
  ylab('Runtime (seconds)')

pdf('~/Documents/bergerlab/6.890/pbmc_fastball_runtimes.pdf', 8,3.6)
pbmc_fb %>% filter(occSquares < 25000) %>% ggplot(aes(x=occSquares,y=time))+
  geom_line(aes(color=factor(DIMRED)))+
  geom_line(data=pbmc_others %>% group_by(occSquares, method) %>% summarize(time=mean(time)), aes(lty=method))+
  coord_cartesian(ylim=c(0,500), xlim=c(0,20000))+
  scale_linetype_discrete(labels=c('Geometric Sketch','PC-Sketch'))+
  scale_color_discrete(name='FastBall VP-tree \nbuilding dimension')+
  ggtitle('PBMC Data Runtimes')+
  xlab('Sketch size (out of 68K)')+
  ylab('Runtime (seconds)')+
  guides(color=guide_legend(ncol=2))
dev.off()

pdf('~/Documents/bergerlab/6.890/pbmc_fastball_alltests.pdf', 8,5)
grid.arrange(p2, p1, nrow=1)

pdf('~/Documents/bergerlab/pbmc_hausdorff_tests.pdf', 8,5)
pbmc_temp %>%
  filter(method %in% c('PC-Sketch','gs','Far traversal', 'fastball'), DIMRED %in% c(NA, 100)) %>%
  ggplot(aes(x=N, y=max_min_dist))+
  coord_cartesian(xlim=c(0,10000))+
  geom_line(aes(color=method))+
  xlab('Sketch size (out of 68K)')+
  ylab('Hausdorff distance')+
  scale_color_discrete(labels=c('FT-Tree Sampling','Ball sampling \n (very slow, 2-approx)',
                                   'Geometric Sketching', 'PC-sketching'))

