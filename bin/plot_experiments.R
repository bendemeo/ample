library(tidyverse)
library(data.table)

lsh_param_test_data=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.3')

lsh_param_test_data$N = as.character(lsh_param_test_data$N)

by_params <- lsh_param_test_data %>%
  group_by(N, gridSize, sampler, numBands, bandSize) %>%
  summarise(rare = mean(rare), remnants=mean(remnants)) %>%
  mutate(percent = rare / as.numeric(N))


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
