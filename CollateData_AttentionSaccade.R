# collate all behavioural csv files into one dataframe to be read in elsewhere

library("dplyr")
getwd()
dir <- '/Users/user/Desktop/Experiments/Nick/AttentionSaccade'
setwd(dir)
fileList <- list.files(path = './behaviour/csv', pattern = '.csv') #change to path on diff computer
dataFiles <- list(NULL)
count <- 1
for(i in fileList){
  path <- paste0('./behaviour/csv/', i)
  dataFiles[[count]] <- read.csv(path, header=TRUE, as.is = TRUE)
  count <- count +1
}

data <- do.call('rbind', dataFiles)
data <- data[order(data$subject),]

fname <- './AttentionSaccade_BehaviouralData_All.csv'
write.csv(data, file = fname, sep = ',', eol = '\n', dec = '.', col.names = TRUE)
