library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)

###### read #######
diverse = fread('target/experiments/293t_diverseLSHTest_backup.txt')
gs = fread('target/experiments/pbmc_gridLSHTest_clustcounts.txt.1')


###### transform ######

##### plot #####
p1 = diverse %>% filter(N==100) %>%
  ggplot(aes(x=numCenters, y=`293t`))+
  geom_boxplot(aes(group = numCenters), show.legend = FALSE)

gs = melt(gs, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize", "N"), 
          measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
                           "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                           "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
                           "CD8+_Cytotoxic_T","Dendritic"))

##### PDFs #####
pdf('plots/pbmc_diverseLSH_centertest.pdf', 12, 8)
p1+
  ggtitle('diverseLSH on PBMC: N=1000')
dev.off()
