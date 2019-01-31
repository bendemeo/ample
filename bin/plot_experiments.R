library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

### comparing different LSH
#####
lsh_param_test_data_1=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.3')
lsh_param_test_data_2=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.4')
lsh_param_test_data_3=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.5')
lsh_param_test_data_4=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.6')

lsh_param_test_data = rbind(lsh_param_test_data_1, lsh_param_test_data_2, lsh_param_test_data_3, fill=TRUE)

lsh_param_test_data$N = as.character(lsh_param_test_data$N)

by_params <- lsh_param_test_data %>%
  group_by(N, gridSize, sampler, numBands, bandSize) %>%
  summarise(rare = mean(rare), remnants=mean(remnants), max_min_dist = mean(max_min_dist)) %>%
  mutate(percent = 100* rare / as.numeric(N)) %>%
  filter(as.numeric(N)<5000)

ggplot(by_params, aes(x=as.numeric(N), y=rare))+
  geom_point(aes(color=sampler))+
  geom_line(aes(color = sampler, group = paste(sampler,bandSize)))+
  geom_text(aes(label = bandSize))



# the number of rare cells seems stable as N increases, which is good.
by_params %>% ggplot(aes(x=gridSize, y=rare))+
  geom_point(aes(color=N))

by_params %>% ggplot(aes(x=gridSize, y=rare))+
  geom_point(aes(color=N))+
  geom_line(aes(color = N))

by_bandsize = lsh_param_test_data %>% 
  group_by(N, bandSize) %>%
  summarise(rare = mean(rare), remnants = mean(remnants),
            lastCounts = mean(lastCounts))

ggplot(by_bandsize, aes(x=bandSize,y=lastCounts))+
  geom_point(aes(color=N))

ggplot(lsh_param_test_data, aes(x=guess,y=actual))+
  geom_point()


### original vs LSH
#####
orig = fread('target/experiments/293t_gs_orig.txt.7')

lsh = fread('target/experiments/293t_gs_lsh.txt.1')

orig = orig %>% filter(replace=="False", sampling_fn=='gs_gap')

### testing different k values
#####
ktest = fread('target/experiments/293t_gs_lsh_ktest.txt.4')

#ktest <- ktest %>% group_by(target, N, gridSize) %>% summarize(rare = mean(rare), max_min_dist = mean(max_min_dist), kmeans_ami=mean(kmeans_ami))

ktest %>% filter(N==500) %>%
  group_by(gridSize) %>%
  summarize(max_min_dist=mean(max_min_dist), rare=mean(rare)) %>%
  ggplot(mapping=aes(x=gridSize,y=max_min_dist))+
  geom_point(aes(color=rare))+
  geom_line()


ktest %>% filter(N==500) %>%
  ggplot(mapping=aes(x=maxCounts,y=max_min_dist))+
  geom_boxplot(aes(group=maxCounts))

ktest %>% filter(N==500) %>%
  ggplot(mapping=aes(x=k, y=rare)) +
  geom_point(aes(color = max_min_dist))+
  geom_line()

ktest %>% filter(N==500) %>%
  ggplot(mapping=aes(x=k, y=max_min_dist)) +
  geom_point(aes(color = rare))+
  geom_line()


ktest %>% filter(N==500) %>%
  ggplot(mapping=aes(x=k, y=kmeans_ami)) +
  geom_point(aes(color = rare))+
  geom_line()


ktest %>% filter(N==500) %>%
  ggplot(mapping=aes(x=gridSize, y=max_min_dist)) +
  geom_point(aes(color = rare))+
  geom_line()


####testing grid size
######
ktest_randomGrid=fread('target/experiments/293t_randomgrid_lsh_ktest.txt.1')

ktest_randomGrid <- ktest_randomGrid %>% group_by(target, N, maxCounts) %>% summarize(rare = mean(rare), max_min_dist = mean(max_min_dist), kmeans_ami=mean(kmeans_ami))

ktest_randomGrid %>% filter(N==500) %>%
  ggplot(mapping=aes(x=maxCounts, y=rare)) +
  geom_point(aes(color = max_min_dist))+
  geom_line()

ktest_randomGrid %>% filter(N==500) %>%
  ggplot(mapping=aes(x=maxCounts, y=max_min_dist)) +
  geom_point(aes(color = rare))+
  geom_line()


ktest_randomGrid %>% filter(N==500) %>%
  ggplot(mapping=aes(x=k, y=kmeans_ami)) +
  geom_point(aes(color = rare))+
  geom_line()


####more comprehensive grid size test
#######
sizetest=fread('target/experiments/gsGridTest.txt.3')
sizetest_rg=fread('target/experiments/randomGridTest.txt.1') # 3 random grids
sizetest_cos = fread('target/experiments/cosTest.txt.1')
sizetest_proj = fread('target/experiments/projTest.txt.1')
sizetest_ball = fread('target/experiments/ballLSHTest.txt.2')

##gsLSH plots

#gridsize vs max_min_dist
sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))


#gridsize vs rare
g0 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.01)))


g1 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=maxCounts))+
  geom_point()

g2 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.01)))


sizetest %>% filter(N==500) %>%
  ggplot(aes(x=maxCounts,y=max_min_dist))+
  geom_line()

sizetest %>% filter(N==500) %>%
  ggplot(aes(x=maxCounts,y=rare))+
  geom_boxplot(aes(group=cut_width(maxCounts,5)))

sizetest %>% filter(rare==28, N==500) %>%
   ggplot(aes(x=maxCounts))+
   geom_histogram()



sizetest %>% filter(N==500)  %>% ggplot(aes(x=rare,y=max_min_dist))+
  geom_boxplot(aes(group=rare))




#N vs rare
sizetest %>% ggplot(aes(x=N, y=rare))+
  geom_smooth(aes(color=cut_width(gridSize,0.01), span=0.1))



##randomGridLSH plots

sizetest_rg %>% filter(N==500) %>% ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))

sizetest_rg %>% filter(N==500) %>% ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))

sizetest_rg %>% ggplot(aes(x=N, y=rare))+
  geom_smooth(aes(color=cut_width(gridSize,0.05)))

sizetest_rg %>% filter(N==1000) %>%ggplot(aes(x=maxCounts,y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(maxCounts,20)))


##cosineLSH plots

sizetest_cos %>% filter(N==1000) %>%
  ggplot(aes(x=numHashes,y=rare))+
  geom_boxplot(aes(group=numHashes))

sizetest_cos %>% filter(N==1000) %>%
  ggplot(aes(x=lastCounts,y=rare))+
  geom_boxplot(aes(group=cut_width(lastCounts,10)))


##projLSH plots 

#gridsize vs max_min_dist
sizetest_proj %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))

#gridsize vs rare
sizetest_proj %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.05)))

#N vs rare
sizetest_proj %>% ggplot(aes(x=N, y=rare))+
  geom_smooth(aes(color=cut_width(gridSize,0.05)))


##gridLSH with lots of grids
sizetest_rg2 = fread('target/experiments/randomGrid_20.txt.1')


sizetest_rg2 %>% filter(N==500) %>% ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))

sizetest_rg2 %>% filter(N==500) %>% ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.03)))

sizetest_rg2 %>% filter(N==500) %>% ggplot(aes(x=maxCounts,y=rare))+
  geom_line()


sizetest_rg2 %>% filter(N==500) %>% ggplot(aes(x=gridSize, y=maxCounts-remnants))+
  geom_line()

sizetest_rg2 %>% ggplot(aes(x=N, y=rare))+
  geom_smooth(aes(color=cut_width(gridSize,0.05)))

sizetest_rg2 %>% filter(maxCounts > 10) %>% ggplot(aes(x=N, y=rare))+
  geom_line(aes(color=cut_width(maxCounts,10)))

sizetest_rg2 %>% filter(N==1000) %>%ggplot(aes(x=maxCounts,y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(maxCounts,20)))

sizetest_rg2 %>% filter(N==100) %>% ggplot(aes(x=maxCounts, y=rare)) + 
  geom_smooth()

############equal density test results############

gs_equaldens = fread('target/experiments/gsGridTest_equaldens.txt.1')
# gs_equaldens$N <- as.character(gs_equaldens$N)
gs_equaldens %>% ggplot(aes(x=gridSize, y=kl_divergence)) +
  geom_smooth(aes(color=cut_width(N,100)), span=0.1)

######## weighted rare sampling ##############
gs_weighted = fread('target/experiments/gsGridTest_weighted.txt.2')


gs_weighted %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group = cut_width(gridSize,0.01)))


gs_weighted %>% ggplot(aes(x=gridSize,y=rare)) +
  geom_line(aes(color = cut_width(N,100)))

gs_weighted %>% filter(N==100) %>%
  ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.01)))

gs_weighted %>% ggplot(aes(x=N, y=rare))+
  geom_line(aes(color=as.character(gridSize), span=0.1))



#weighted vs nonweighted

unweighted = fread('target/experiments/gsGridTest.txt.4')
gs_compare = full_join(gs_weighted, unweighted)
gs_compare$sampler[which(is.na(gs_compare$maxCounts))] <- 'gsLSH_wt'


gs_compare %>% group_by(sampler, N) %>% summarize(best_size=gridSize[which(rare==max(rare))[1]])


gs_compare %>% filter(as.numeric(gridSize)>=0) %>% View()
gs_compare %>% filter(as.numeric(gridSize)>0.3, as.numeric(gridSize)<0.4) %>% 
  ggplot(aes(x=N, y=rare))+
  geom_smooth(aes(color=sampler, lty=cut_width(gridSize, 0.01)))





gridTest = fread('target/experiments/gsGridTest.txt.3')
gridTest_wt = fread('target/experiments/gsGridTest_weighted.txt.2')



#huge dataframe of all, best vs. best

compare_all = full_join(unweighted, gs_weighted) %>%
  full_join(sizetest_rg2)
            
compare_all$sampler[which(is.na(compare_all$maxCounts))] <- 'gsLSH_wt'

compare_all %>% group_by(sampler, N) %>% summarize(best_size=gridSize[which(rare==max(rare))[1]], best_rare = mean(rare[gridSize==best_size]),
                                                   mean_counts=mean(maxCounts), max_min_dist=mean(max_min_dist)) %>% View()




######## PBMC ##########
pbmc_nonwt = fread('target/experiments/gsGridTest_clustcounts_nonwt.txt.1')
pbmc_nonwt_2 = fread('target/experiments/pbmc_gsGridTest_clustcounts_nonwt.txt.1')
pbmc_wt = fread('target/experiments/gsGridTest_clustcounts_wt.txt.1')
pbmc_grid_nonrandom = fread('target/experiments/pbmc_gridLSHTest_clustcounts.txt.1')
pbmc_grid_random = fread('target/experiments/pbmc_gridLSHTest_clustcounts_randomorigin_.txt.2')

pbmc_nonwt = melt(pbmc_nonwt, id.vars=c('gridSize','max_min_dist', 'time'), measure.vars=c('b_cells','cd14_monocytes', "cd4_t_helper",   "cd56_nk",       
                                                                     "cytotoxic_t",    "memory_t",       "regulatory_t"))

pbmc_wt = melt(pbmc_wt, id.vars=c('gridSize','max_min_dist', 'time'), measure.vars=c('b_cells','cd14_monocytes', "cd4_t_helper",   "cd56_nk",       
                                                                                           "cytotoxic_t",    "memory_t",       "regulatory_t"))
pbmc_grid_nonrandom = melt(pbmc_grid_nonrandom, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize", "N"), 
                           measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                                            "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))
pbmc_nonwt_2 = melt(pbmc_nonwt_2, id.vars=c("max_min_dist","time","maxCounts","gridSize"), 
                           measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                                            "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))
pbmc_grid_random = melt(pbmc_grid_random, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize"), 
                           measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                                            "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))

pdf('plots/pbmc_gridLSH_nonrandom_size_vs_clustCounts.pdf', 15, 8)
pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=value, color=variable))+
acet_wrap(~variable)+
  geom_smooth(span=0.1)
dev.off()

pdf('plots/pbmc_gridLSH_nonrandom_size_vs_clustCounts_all.pdf', 10, 6)
pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.1)
dev.off()


pdf('plots/pbmc_gridLSH_nonrandom_size_vs_clustCounts_all.pdf', 10, 6)
pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_line(span=0.1)
dev.off()

g0 = pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.1)+
  theme(legend.position = 'top')+
  labs(y = 'Membership in size-1000 subsample')+
  ggtitle('PBMC 68K gridLSH')

g1 = pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group = gridSize))+
  labs(y = 'Hausdorff distance')

g2 = pbmc_grid_nonrandom %>% ggplot(aes(x=gridSize, y=maxCounts))+
  geom_line(span=0.1)+
  labs(y = 'grid squares occupied')


pdf('plots/pbmc_68k_gridtests.pdf', 12, 12)
grid.arrange(g0,g1,g2)
dev.off()


pbmc_grid_nonrandom %>% ggplot(aes(x=maxCounts, y=max_min_dist, color = variable))+
  geom_line(span=0.1)

pbmc_grid_nonrandom %>% ggplot(aes(x=maxCounts, y=value, color = variable))+
  geom_line(span=0.1)


ggplot(pbmc_grid_random, aes(x=maxCounts, y=time))+
  geom_point()





pbmc_grid_nonrandom %>% ggplot(aes(x=max_min_dist, y=value, color = variable))+
  geom_line(span=0.1)

pdf('plots/pbmc_grid_vs_clustcounts.pdf', 8,5)
ggplot(pbmc_nonwt, aes(x=gridSize, y=value, color=variable))+
  facet_wrap(~variable)+
  geom_smooth(span=0.1)
  geom_boxplot(aes(group= paste0(variable,cut_width(gridSize,0.03))))
dev.off()

pdf('plots/pbmc_gridLSH_random_size_vs_clustCounts_all.pdf', 10, 6)
pbmc_grid_random %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.2)
dcolev.off()

ggplot(pbmc_nonwt_2, aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.2)
geom_boxplot(aes(group= paste0(variable,cut_width(gridSize,0.03))))
dev.off()



######quick experiments on short data ######
pbmc_grid_random_short = fread('target/experiments/pbmc_gridLSHTest_clustcounts_randomorigin_short.txt.1')
pbmc_grid_nonrandom_short = fread('target/experiments/pbmc_gridLSHTest_clustcounts_randomorigin_short.txt.1')



pbmc_grid_random_short = melt(pbmc_grid_random_short, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize"), 
                        measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                                         "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))
pbmc_grid_nonrandom_short = melt(pbmc_grid_nonrandom_short, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize"), 
                              measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                                               "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))



pbmc_grid_nonrandom_short %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.2)

pbmc_grid_random_short %>% ggplot(aes(x=gridSize, y=value, color=variable))+
  geom_smooth(span=0.2)






#############
scores = fread('target/experiments/pbmc_gridLSHTest_clustcounts_.txt.3')

scores <- melt(scores, id.vars=c("max_min_dist","time","maxCounts","randomize_origin","gridSize"),
               measure.vars = c('CD14+_Monocyte','CD19+_B','CD4+/CD25_T_Reg','CD4+/CD45RA+/CD25-_Naive_T',
                                'CD4+/CD45RO+_Memory','CD4+_T_Helper2','CD56+_NK','CD8+/CD45RA+_Naive_Cytotoxic',
                                'CD8+_Cytotoxic_T','Dendritic','CD14+_Monocyte_score','CD19+_B_score',
                                'CD4+/CD25_T_Reg_score','CD4+/CD45RA+/CD25-_Naive_T_score','CD4+/CD45RO+_Memory_score',
                                'CD4+_T_Helper2_score','CD56+_NK_score','CD8+/CD45RA+_Naive_Cytotoxic_score',
                                'CD8+_Cytotoxic_T_score','Dendritic_score')    )

ggplot(aes(x=gridSize,y=value, color=variable))+
  geom_line()









#######treeLSH testing############
tree = fread('target/experiments/pbmc_treeLSHTest_clustcounts_.txt.1')

tree = melt(tree, id.vars=c("max_min_dist","time","occSquares", "children", "splitSize", "N"), 
     measure.vars = c("CD14+_Monocyte","CD19+_B","CD4+/CD25_T_Reg","CD4+/CD45RA+/CD25-_Naive_T","CD4+/CD45RO+_Memory",
                      "CD4+_T_Helper2","CD56+_NK","CD8+/CD45RA+_Naive_Cytotoxic","CD8+_Cytotoxic_T","Dendritic"))

tree %>% ggplot(aes(x=N, y=value, color = variable))+
  geom_line()

pbmc_grid_nonrandom %>% filter(gridSize==0.3) %>%
  ggplot(aes(x=N, y=value, color = variable))+
  geom_line()



#######
t_tree = fread('target/experiments/293t_treeLSHTest_clustcounts.txt.3')

t_tree %>% ggplot(aes(x=N, y=rare))+
  geom_boxplot(aes(group = N))


##########MAKE PDFS#############
#gridsize vs rare
r0 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=rare))+
  geom_boxplot(aes(group=cut_width(gridSize,0.01)))+
  ggtitle('293T Jurkat grid size tests')+
  labs(y='Rare cells in 500')


r1 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=maxCounts))+
  geom_point()+
  labs(y='Occupied grid squares')

r2 = sizetest %>% filter(N==500) %>%
  ggplot(aes(x=gridSize, y=max_min_dist))+
  geom_boxplot(aes(group=cut_width(gridSize,0.01)))+
  labs(y='Hausdorff distance')


pdf('plots/293Tjurkat_gridtests.pdf')
grid.arrange(r0,r2,r1)
dev.off()


pdf('