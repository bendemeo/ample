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

multigauss_pcalsh = fread('target/experiments/PCALSH_multigauss_PCALSH_gridTest.txt.1')
multigauss_gridlsh = fread('target/experiments/gridLSH_multigauss_gridLSH_gridTest.txt.1')

multigauss_pcalsh$method = rep('pcaLSH', nrow(multigauss_pcalsh))
multigauss_gridlsh$method = rep('gridLSH', nrow(multigauss_gridlsh))

gauss_all = rbind(gauss_pcalsh, gauss_grid)
multigauss_all = rbind(multigauss_pcalsh, multigauss_gridlsh)
pbmc_all = rbind(pbmc_gs, pbmc_pcalsh)


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
  ggplot(aes(x=occSquares, y=time))+
  geom_line(aes(color=method))
