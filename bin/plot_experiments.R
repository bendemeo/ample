library(tidyverse)
library(data.table)

lsh_param_test_data_1=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.3')
lsh_param_test_data_2=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.4')
lsh_param_test_data_3=fread('~/Documents/bergerlab/lsh/ample/target/experiments/293t_jurkat_lsh.txt.5')

lsh_param_test_data = rbind(lsh_param_test_data_1, lsh_param_test_data_2, lsh_param_test_data_3)

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

ggplot(by_params, aes(x=



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
