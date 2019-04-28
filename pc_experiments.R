library(tidyverse)
library(data.table)
library(dplyr)
library(gridExtra)
setwd('~/Documents/bergerlab/lsh/ample/')

gauss_pcalsh = fread('target/experiments/randomGauss_PCALSH_test.txt.1')
gauss_pcalsh$method = rep('pcaLSH', nrow(gauss_pcalsh))

gauss_grid = fread('target/experiments/randomGauss_gridLSH_test.txt.1')
gauss_grid$method = rep('gridLSH', nrow(gauss_grid))


gauss_all = rbind(gauss_pcalsh, gauss_grid)


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


