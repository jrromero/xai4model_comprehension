install.packages("utiml")
library(utiml)
install.packages("mldr")
library(mldr)
df <- read.csv("modelset_df_final_ecore.csv")
mymldr <- mldr_from_dataframe(df, labelIndices = c(7, 8, 9, 10, 11, 12, 13, 14, 15, 16), name = "modelset")
ds <- utiml::create_holdout_partition(mymldr, c(train=0.70, test=0.30), "stratified")
training <- ds[["train"]][["dataset"]]
testing <- ds[["test"]][["dataset"]]
write.csv(training, "train.csv", row.names = TRUE)
write.csv(testing, "test.csv", row.names = TRUE)