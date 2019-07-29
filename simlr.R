library(SIMLR)
library(igraph)
library(dplyr)
library(data.table)
library(ggplot2)
library(Rtsne)

setwd('~/Documents/bergerlab/lsh/ample/')

pbmc_1000 = fread('data/pbmc/subsamples/ft_1000', header=TRUE)[,2:102]
pbmc_200 = fread('data/pbmc/subsamples/ft_200', header=TRUE)[,2:102]

s = SIMLR(t(pbmc_1000[,1:100]), c=10)
s2 = SIMLR(t(pbmc_200[,1:100]),c=10)

stest = SIMLR(X=BuettnerFlorian$in_X, c=BuettnerFlorian$n_clust, cores.ratio=0)
plot(stest$ydata,  
     col=c(topo.colors(BuettnerFlorian$n_clust))[BuettnerFlorian$true_labs[,1]],
     pch=20)

ggplot(data.frame(stest$ydata))+
  geom_point(aes(x=X1, y=X2, color=pbmc_200$labels))

zheng = fread('data/zheng/dimred.txt')
patients
zheng_simlr = SIMLR(t(zheng), c=5)






ggplot(data.frame(s$ydata))+
  geom_point(aes(x=X1, y=X2, color = pbmc_1000$labels))


m = s$S

for(i in 1:nrow(m)){
  m[i,]=rank(m[i,])
}

m=(m+t(m))/2

#t-SNE sanity check that labels aren't mixed up
tsne = Rtsne(pbmc_200[,1:100])
d = data.frame(tsne$Y)
d$labels = pbmc_200$labels
colnames(d)=c('X','Y', 'label')
ggplot(d)+
  geom_point(aes(x=X, y=Y, color=label))
