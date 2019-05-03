library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

gauss_pcalsh = fread('target/experiments/randomGauss_PCALSH_test.txt.1')
gauss_pcalsh$method = rep('pcaLSH', nrow(gauss_pcalsh))

gauss_grid = fread('target/experiments/randomGauss_gridLSH_test.txt.1')
gauss_grid$method = rep('gridLSH', nrow(gauss_grid))

pbmc_pcalsh = fread('target/experiments/pbmc_PCALSH_hausdorff_backup.txt')
pbmc_gs = fread('target/experiments/pbmc_gsLSH_tests_backup.txt')


pbmc_gs$lastCounts = gsub("\\[|\\]", "", pbmc_gs$lastCounts)
pbmc_gs$lastCounts = as.numeric(pbmc_gs$lastCounts)
colnames(pbmc_gs)[4] <- "occSquares"
pbmc_gs <- pbmc_gs[,-6]

pbmc_gs$method = rep('gsLSH', nrow(pbmc_gs))
pbmc_pcalsh$method = rep('pcaLSH', nrow(pbmc_pcalsh))





multigauss_pcalsh = fread('target/experiments/PCALSH_multigauss_hausdorff_backup.txt')
multigauss_gslsh = fread('target/experiments/gsLSH_multigauss_hausdorff_backup.txt')

multigauss_pcalsh$method = rep('pcaLSH', nrow(multigauss_pcalsh))
multigauss_gslsh$method = rep('gridLSH', nrow(multigauss_gslsh))

gauss_all = rbind(gauss_pcalsh, gauss_grid)
multigauss_all = rbind(multigauss_pcalsh, multigauss_gslsh)
pbmc_all = rbind(pbmc_gs, pbmc_pcalsh)


cell_types = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
               "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
               "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
               "CD8+_Cytotoxic_T","Dendritic")

# cell_counts = c(3817,3306,2812,3126,5859,11445,14112,21975,1865,262)
# 
# c = colnames(pbmc_all)
# pbmc_all=data.frame(pbmc_all)
# colnames(pbmc_all)=c
# for(i in 1:length(cell_counts)){
#   col = which(colnames(pbmc_all) == cell_types[i])[1]
#   pbmc_all[,col]= pbmc_all[,col] / cell_counts[i]
# }




###### Single Gaussian ######



# sample size vs Hausdorff: better for PCALSH
gauss_all %>% filter(occSquares < 1500) %>%
  group_by(occSquares, method) %>%
  summarize(max_min_dist = mean(max_min_dist)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  geom_line(aes(color=method))


# grid size vs. number occupied squares: PCALSH uses grid squares better, also smoother.
gauss_all %>%
  ggplot(aes(x=gridSize, y=occSquares))+
  geom_line(aes(color=method))


#sample size vs time: fairly comparable. Theoretical analysis needed.
gauss_all %>%
  ggplot(aes(x=occSquares, y=time))+
  geom_line(aes(color=method))

###### Multigaussian #####

# sample size vs Hausdorff: better for PCALSH
multigauss_all %>% filter(max_min_dist<2) %>%
  group_by(occSquares, method) %>%
  summarize(max_min_dist = mean(max_min_dist)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  geom_line(aes(color=method))


# grid size vs. number occupied squares: PCALSH uses grid squares better, also smoother.
multigauss_all %>%
  ggplot(aes(x=gridSize, y=occSquares))+
  geom_line(aes(color=method))


#sample size vs time: fairly comparable. Theoretical analysis needed.
multigauss_all %>%
  ggplot(aes(x=gridSize, y=time))+
  geom_line(aes(color=method))




pbmc_clusts = melt(pbmc_all, id.vars=c("max_min_dist","time","occSquares","gridSize", "N", "method"), 
          measure.vars = cell_types)

###### PBMC #######
pbmc_3 = pbmc_all %>% filter(max_min_dist<2, occSquares < 2000) %>%
  group_by(occSquares, method) %>%
  summarize(max_min_dist = mean(max_min_dist)) %>%
  ggplot(aes(x=occSquares, y=max_min_dist))+
  geom_line(aes(color=method))

# grid size vs. number occupied squares: PCALSH uses grid squares better, also smoother.
pbmc_1 = pbmc_all %>%
  filter(occSquares > 50) %>%
  ggplot(aes(x=gridSize, y=occSquares))+
  geom_line(aes(color=method))+
  xlab("Box side length")+
  ylab("Boxes used in cover")+
  scale_color_discrete(name='Covering Method', labels=c("Plaid Covering", "Learned Covers"))
  

pbmc_2 = pbmc_all %>% group_by(occSquares, method) %>%
  summarize(time= mean(time)) %>%
  ggplot(aes(x=occSquares, y=time))+
  geom_line(aes(color=method))



pbmc_clusts %>% filter(method == 'pcaLSH') %>%
  ggplot(aes(x=occSquares, y=value, color=variable))+
  facet_wrap(~variable)+
  geom_line()

pbmc_clusts %>% 
  ggplot(aes(x=occSquares, y=value, color=method))+
  facet_wrap(~variable)+
  geom_line()

pbmc_clusts %>% 
  filter(variable == 'CD14+_Monocyte') %>%
  ggplot(aes(x=occSquares, y=value, color=method))+
  geom_line()

