library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')



###### read #######
diverse = fread('target/experiments/pbmc_diverseLSHTest_backup.txt')
diverse_q4 = fread('target/experiments/pbmc_diverseLSHTest_q4_backup.txt')
gs = fread('target/experiments/pbmc_gridLSHTest_clustcounts.txt.1')
cs = fread('target/experiments/pbmc_centerSamplerTest_backup.txt')
cswt = fread('target/experiments/pbmc_centerSamplerTest_weighted_backup.txt')
csnorm = fread('target/experiments/pbmc_centerSamplerTest_l2norm_backup.txt')
cssph = fread('target/experiments/pbmc_centerSamplerTest_spherical_backup.txt')
cs_5000 = fread('plotData/pbmc_centerSampler_plotData_5000_100centers')
cs_densitywt = fread('target/experiments/pbmc_centerSamplerTest_densityWeighted_backup.txt')
cs_cosdensitywt = fread('target/experiments/pbmc_centerSamplerTest_cosinedensityWeighted_backup.txt')
dpp = fread('target/experiments/pbmc_dpp_tests_2_backup.txt')

###### transform ######


cell_types = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg",
          "CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
          "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic",
          "CD8+_Cytotoxic_T","Dendritic")


diverse = melt(diverse, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'batch'), 
               measure.vars=cell_types)

diverse_q4 = melt(diverse_q4, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'batch'), 
               measure.vars=cell_types)


cs = melt(cs, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
               measure.vars=cell_types)

cswt = melt(cswt, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
          measure.vars=cell_types)

cs_densitywt = melt(cs_densitywt, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
                 measure.vars=cell_types)
cs_cosdensitywt = melt(cs_cosdensitywt, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
                    measure.vars=cell_types)

csnorm = melt(csnorm, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
          measure.vars=cell_types)

cssph = melt(cssph, id.vars=c('numCenters','max_min_dist', 'time', 'N', 'steps'), 
              measure.vars=cell_types)

# gs = melt(gs, id.vars=c('gridSize','max_min_dist', 'time', 'N'),
#           measure.vars=c('b_cells','cd14_monocytes', "cd4_t_helper",
#                          "cd56_nk", "cytotoxic_t","memory_t", "regulatory_t"))

gs = melt(gs, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize", "N"), 
          measure.vars = cell_types)

dpp = melt(dpp, id.vars=c("max_min_dist","time","steps", "N"), 
          measure.vars = cell_types)

##### plot #####
divplot = diverse %>% filter(N==1000) %>%
  ggplot(aes(x=numCenters, y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group = numCenters), show.legend = FALSE)

divplot_q4 = diverse_q4 %>% filter(N==500) %>%
  ggplot(aes(x=numCenters, y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group = numCenters), show.legend = FALSE)

gsplot = gs %>% filter(N==1000) %>%
  ggplot(aes(x=gridSize, y=value/1000, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group = cut_number(gridSize,50)), show.legend = FALSE)

csplot = cs %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=numCenters), show.legend = FALSE)

cswtplot = cswt %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=numCenters), show.legend = FALSE)

cs_densitywtplot = cs_densitywt %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=numCenters), show.legend = FALSE)


cs_cosdensitywtplot = cs_densitywt %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=numCenters), show.legend = FALSE)

cssphplot = cssph %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=numCenters), show.legend = FALSE)

csnormplot = csnorm %>% filter(N==500, steps==1000 ) %>%
  ggplot(aes(x=numCenters,y=value/500, color=variable))+
  facet_wrap(~variable)+
  geom_boxplot(aes(group=cut_width(numCenters,1)), show.legend = FALSE)

csmm = cs %>% filter(N==500, steps==1000) %>% 
  ggplot(aes(x=numCenters, y=max_min_dist))+
  geom_smooth()

cs_example = cs_5000 %>% ggplot(aes(x=x,y=y, color = cell_type))+
  geom_point()


gs_hausplot = gs %>% ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=gridSize))+
  ggtitle('Geometric Sketching on PBMC: N=1000')+
  labs(x='Grid Size', y='Hausdorff Distance')


diverse_hausplot = diverse %>% filter(N==1000, batch==5000) %>%
  ggplot(aes(x=numCenters, y=max_min_dist))+
  geom_boxplot(aes(group=numCenters))+
  ggtitle('Diverse LSH on PBMC: N=1000')+
  labs(x='Number of DPP centers', y='Hausdorff Distance')


cs_hausplot = cs %>% filter(N==500, steps==10000) %>%
  ggplot(aes(x=numCenters, y=max_min_dist))+
  geom_boxplot(aes(group=numCenters))+
  ggtitle('Diverse LSH on PBMC: N=1000')+
  labs(x='Number of DPP centers', y='Hausdorff Distance')

##### PDFs #####

pdf('plots/gs_vs_diverse_haus.pdf', 12, 8)
grid.arrange(gs_hausplot, diverse_hausplot)
dev.off()

pdf('plots/pbmc_diverseLSH_centertest.pdf', 12, 8)
divplot+
  ggtitle('diverseLSH on PBMC: N=500')
dev.off()

pdf('plots/pbmc_gsLSH_gridTest.pdf', 12, 8)
gsplot+
  ggtitle('gsLSH on PBMC: N=1000')
dev.off()

pdf('plots/pbmc_cs_centerTest.pdf', 12, 8)
csplot + 
  ggtitle('center sampling on PBMC: N=500')
dev.off()

pdf('plots/pbmc_cswt_centerTest.pdf', 12, 8)
cswtplot + 
  ggtitle('center sampling on PBMC: N=500 (weighted)')
dev.off()