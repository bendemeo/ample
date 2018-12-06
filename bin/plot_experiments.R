library(tidyverse)
library(data.table)

lsh_param_test_data=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.1')

by_params <- lsh_param_test_data %>% 
  group_by(numHashes, numBands, bandSize, N) %>%
  summarise(rare=mean(rare), kmeans_ami=mean(kmeans_ami))

# the number of rare cells seems stable as N increases, which is good.
by_params %>% ggplot(aes(x=bandSize, y=kmeans_ami))+
  geom_line()
