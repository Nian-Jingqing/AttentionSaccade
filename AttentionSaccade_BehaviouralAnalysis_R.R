library('tidyverse')
library(magrittr)
getwd()
dir <- '/Users/user/Desktop/Experiments/Nick/AttentionSaccade'
setwd(dir)
fname <- './AttentionSaccade_BehaviouralData_All.csv'

se <- function(x) sd(x)/sqrt(length(x)) #standard error function

df <- read.csv(fname, header = T, sep = ',')
df %<>% dplyr::select(-X)
ds <- df %>%
  dplyr::select(subject, task,resp, validity, time, corr) %>%
  dplyr::mutate(subject = as.factor(subject), validity = as.factor(validity), task = as.factor(task)) %>%
  dplyr::filter(task == 1) %>% droplevels() %>%
  dplyr::filter(resp != 0) %>% #remove attention block trials where no response was made
  dplyr::group_by(subject, validity) %>%
  summarise_at(.vars = c('time', 'corr'), funs(mean, sd)) %>%
  as.data.frame()
  

ds.plot <- ds %>%
  dplyr::group_by(validity) %>% dplyr::select(-subject) %>%
  summarise_all(.funs = c('mean', 'se')) %>% as.data.frame()

#figure showing reaction time results
ggplot(ds.plot, aes(x = validity, y = time_mean_mean, fill = validity)) + 
  geom_bar(stat = 'identity', width = 0.7, position = position_dodge()) +
  geom_errorbar(aes(x = validity, ymin = time_mean_mean-2*time_sd_se, ymax = time_mean_mean + 2*time_sd_se), width =0.35, position = position_dodge(0.7)) +
  theme_bw() + labs(x = 'Cue validity', y = 'mean RT (s)') +
  geom_point(data = ds, aes(x = validity, y = time_mean)) #overlay single subject data onto plot

#figure for accuracy results
ggplot(ds.plot, aes(x = validity, y = corr_mean_mean, fill = validity)) + 
  geom_bar(stat = 'identity', width = 0.7, position = position_dodge()) +
  geom_errorbar(aes(x = validity, ymin = corr_mean_mean-2*corr_sd_se, ymax = corr_mean_mean + 2*corr_sd_se), width =0.35, position = position_dodge(0.7)) +
  theme_bw() + labs(x = 'Cue validity', y = 'mean proportion correct') +
  geom_point(data = ds, aes(x = validity, y = corr_mean)) #overlay single subject data onto plot
