library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')



###### read #######
diverse = fread('target/experiments/pbmc_diverseLSHTest_backup.txt')
gs = fread('target/experiments/pbmc_gridLSHTest_clustcounts.txt.1')

###### transform ######
diverse = melt(diverse, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'batch'), 
               measure.vars=c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
                              "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                              "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
                              "CD8+_Cytotoxic_T","Dendritic"))

# gs = melt(gs, id.vars=c('gridSize','max_min_dist', 'time', 'N'),
#           measure.vars=c('b_cells','cd14_monocytes', "cd4_t_helper",
#                          "cd56_nk", "cytotoxic_t","memory_t", "regulatory_t"))

gs = melt(gs, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize", "N"), 
          measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
                           "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                           "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
                           "CD8+_Cytotoxic_T","Dendritic"))

##### plot #####
p1 = diverse %>% filter(N==500) %>%
  ggplot(aes(x=numCenters, y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group = numCenters), show.legend = FALSE)


p2 = gs %>% filter(N==1000) %>%
  ggplot(aes(x=gridSize, y=value/1000, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group = cut_number(gridSize,50)), show.legend = FALSE)


##### PDFs #####
pdf('plots/pbmc_diverseLSH_centertest.pdf', 12, 8)
p1+
  ggtitle('diverseLSH on PBMC: N=500')
dev.off()

pdf('plots/pbmc_gsLSH_gridTest.pdf', 12, 8)
p2+
  ggtitle('gsLSH on PBMC: N=1000')
dev.off()
