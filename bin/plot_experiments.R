library(tidyverse)
library(data.table)

lsh_param_test_data=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_LSHparamtest.txt.1')

by_params <- lsh_param_test_data %>% group_by(numHashes, numBands, bandSize,replace, N) %>%
  summarise(rare=mean(rare))
