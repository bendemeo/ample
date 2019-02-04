library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

###### read #######
diverse = fread('target/experiments/293t_diverseLSHTest_backup.txt')
gs = fread('target/experiments/gsGridTest_clustcounts_nonwt.txt.1')
cs = fread('target/experiments/293t_centerSamplerTest.txt.1')

###### transform ######



##### plot #####
c1 = cs %>% filter(N==100, steps==10000) %>%
  ggplot(aes(x=numCenters, y=`293t`))+
  geom_boxplot(aes(group=numCenters))

ggplot(cs,aes(x=numCenters, y=`293t`))+
  geom_boxplot(aes(group=numCenters))
# p1 = diverse %>% filter(N==500) %>%
#   ggplot(aes(x=numCenters, y=value, color=variable))+
#   facet_wrap(~variable)+
#   geom_boxplot(aes(group = numCenters), show.legend = FALSE)
# 


##### PDFs #####
pdf('plots/pbmc_diverseLSH_centertest.pdf', 12, 8)
p1+
  ggtitle('diverseLSH on PBMC: N=1000')
dev.off()
