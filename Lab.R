#install.packages("tm")
#install.packages("SnowballC")
#install.packages("topicmodels")
#install.packages("RTextTools")
#install.packages("class")
#install.packages("infotheo")
#install.packages("e1071")
library(tm)
library(SnowballC)
library(topicmodels)
library(RTextTools)
library(class)
library("e1071")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

data <- read.csv(file="reutersCSV.csv", stringsAsFactors=FALSE)
data <- subset(data, data[[140]] != "")    # remove rows with no text section

#for (i in 1:nrow(data)) {
#  if (data[[140]] == "") {
#    data[[140]] <- data[[139]]
#  }
#}

data <- subset(data, data[,3] != "not-used") # remove rows with not-used purpose
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))


classlabels=c("topic.earn", "topic.acq", "topic.money.fx", "topic.grain",
              "topic.crude", "topic.trade", "topic.interest", "topic.ship",
              "topic.wheat", "topic.corn")

#
# Obtain a data frame containing only the ten topics of interest
#

attributeset <- data[,1:10]
for (i in 1:length(classlabels)) {
  for (j in 1:ncol(data)) {
    if (colnames(data)[j] == classlabels[i]) {
      attributeset[i] <- data[,j]
      colnames(attributeset)[i] <- classlabels[i]
    }
  }
}

# Add information about whether the data is in the training set or the data set. When
# the text mining frame is constructed from this frame, this column should be omitted.

attributeset[,11] <- data[,3]
colnames(attributeset)[11] <- colnames(data)[3]
# add the title and the content back into the dataset
attributeset[12] <- data[,139]
colnames(attributeset)[12] <- colnames(data)[139]
attributeset[13] <- data[,140]
colnames(attributeset)[13] <- colnames(data)[140]

#
# At this point, we have a frame containing 10 variables. However, not all
# of the rows in this matrix are guaranteed to have non-zero values. Thus,
# we must first reduce the attrset dataset further by removing these rows,
# because these can't be classified (as they have no identifying features)
#

no.zero.values <- attributeset[1,]
k = 1
for (i in 1:nrow(attributeset)) {
  isnonempty <- FALSE
  for (j in 1:10) {
    if (attributeset[i,j] != 0) {
      isnonempty <- TRUE
      break
    }
  }
  if (isnonempty) {
    no.zero.values[k,] <- attributeset[i,]
    k <- k + 1
  }
}


nzv <- no.zero.values[-c(551, 1782, 4571, 4743, 5796, 8274),];



# Put the text into the corpus, remove the training/test set info
tm.full.frame <- nzv[-11]
cs <- Corpus(DataframeSource(tm.full.frame))

cs <- tm_map(cs, toSpace, "/|@|\\|")
cs <- tm_map(cs, content_transformer(tolower))
cs <- tm_map(cs, removeNumbers)
cs <- tm_map(cs, removePunctuation)
# Remove common English stop words, such as very, for, of, and, etc., as well as potentially
# unhelpful words such as said, and variations of reuters. Also remove unnecessary whitespace,
# and stem the words in the document, removing suffixes such as -ing, -ed, -'s.
cs <- tm_map(cs, removeWords, stopwords("english"))
cs <- tm_map(cs, stemDocument)
cs <- tm_map(cs, removeWords, c("said", "reuter"))
cs <- tm_map(cs, stripWhitespace)

dtm <- DocumentTermMatrix(cs)
rdtm <- removeSparseTerms(dtm, 0.9)
as.matrix(rdtm, classlabels)


rowTotals <- apply(rdtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- rdtm[rowTotals> 0, ]


ldamodel <- LDA(rdtm, 10)
terms(ldamodel, 2)

probs <- posterior(ldamodel, rdtm)
probs <- as.data.frame(as.matrix(probs$topics))



# construct a new data frame which, for each row of the nzv frame, adds the relevant row of the probs 
iter <- 1
k <- 0
super.awesome.set <- nzv[1,1:12]
for (i in 1:nrow(nzv)) {
  numones <- 0
  for (j in 1:10) {
    if (nzv[i,j] != 0) {
      # make a new row in my super.awesome.set containing row(i) of (LDA or posterior)?
      #   + names(no.zero.values)[j] + no.zero.values[i,11(purpose)]
      super.awesome.set[iter,] <- cbind(probs[i,], names(nzv)[j], nzv[i,11])
      iter <- iter + 1
      numones <- numones + 1
      # cat("Topic is", names(no.zero.values)[j], "\n")
    }
  }
  cat("Number of ones to sort out", numones, "\n")
  k <- k + numones
}
cat(k, "\n")
names(super.awesome.set) = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "label", "purpose")
sas.ordered <- super.awesome.set[order(super.awesome.set$purpose, decreasing=TRUE),]

# discretise the data by equal frequencies, and bind the class label and purpose of the data back in
saso.disc <- infotheo::discretize(sas.ordered[, 1:10], disc="equalfreq", nbins=5)
saso.disc <- cbind(saso.disc[,1:10], sas.ordered[,11:12])
saso.disc[1,]
names(saso.disc)[1:10] <- classlabels
saso.disc[,11] <- as.factor(saso.disc[,11])

sasod.train <- subset(saso.disc, saso.disc$purpose == "train")
sasod.train <- sasod.train[,1:11]
sasod.test <- subset(saso.disc, saso.disc$purpose == "test")
sasod.test <- sasod.test[,1:11]

# # # # # # # # # # # # # # # # # # # #
#             CLASSIFYING             #
# # # # # # # # # # # # # # # # # # # #


tenfold <- function(dataframe, classificationMethod="NO CLASSIFICATION METHOD SELECTED") {
  datarand <- dataframe[sample(nrow(dataframe)), ] # randomise the data and put it into datarand
  size <- floor(nrow(datarand)/10)                 # get the floor of frame size / 10
  rem <- nrow(datarand) %% 10                      # get the remainder (in case not wholly divisible)
  
  totaltable <- NULL; startpoint <- 1; endpoint <- 0;
  t.acc <- 0; t.fm.mac <- 0; t.fm.mic <- 0; t.rec.mac <- 0; t.rec.mic <- 0; t.pres.mac <- 0; t.pres.mic <- 0;
  
  cat("\n======== TENFOLD CROSS VALIDATION WITH", classificationMethod,"========\n\n")
  for (i in 1:10) {
    if (rem != 0) { endpoint <- startpoint + size; rem <- rem - 1 }
    else          { endpoint <- startpoint + size - 1 }
    
    testdata <- datarand[startpoint:endpoint,]
    trainingdata <- datarand[-(startpoint:endpoint),]
    
    if (classificationMethod == "nb") {
      nb.model <- naiveBayes(trainingdata[,11] ~., data=trainingdata[,1:10])
      foldtable <- table(ACTUAL = testdata[,11], PREDICTED = predict(nb.model, testdata[,1:10]))
    }
    else if (classificationMethod == "rf") {
      rf.model <- randomForest(trainingdata[,11] ~., data=trainingdata[,1:10], ntree=100)
      foldtable <- table(ACTUAL = testdata[,11], PREDICTED = predict(rf.model, testdata[,1:10]))
    }
    else if (classificationMethod == "svm") {
      svm.model <- svm(trainingdata[,11] ~., data=trainingdata[,1:10])
      foldtable <- table(ACTUAL = testdata[,11], PREDICTED = predict(svm.model, testdata[,1:10]))
    }
    else {
      cat("ERROR: please provide one of the following as a classification method\n")
      cat("nb to perform naiveBayes\n")
      cat("rf to perform randomForest\n")
      cat("svm to perform SVM\n")
      return()
    }

    if (i == 1) { totaltable <- foldtable }
    else { totaltable <- totaltable + foldtable }   # get total values for table
    
    tp = sum(diag(foldtable))
    # the number of positive class instances we recognise correctly
    accuracy = tp / nrow(testdata)
    macrorecall = 0
    for (j in 1:nrow(foldtable)) { macrorecall = macrorecall + (foldtable[j,j]/sum(foldtable[j,])) }
    macrorecall = macrorecall / nrow(foldtable)
    macroprecision = 0
    for (j in 1:nrow(foldtable)) { macroprecision = macroprecision + (foldtable[j,j]/sum(foldtable[,j])) }
    
    macroprecision = macroprecision / nrow(foldtable)
    microrecall = sum(diag(foldtable)) / sum(foldtable)
    microprecision = sum(diag(foldtable)) / sum(foldtable)
    macroFM = ( 2 * macroprecision * macrorecall ) / (macroprecision + macrorecall)
    microFM = ( 2 * microprecision * microrecall ) / (microprecision + microrecall)
    
    cat("==== FOLD", i, "====\n")
    cat("Accuracy =", accuracy, "\n")
    cat("Macrorecall =", macrorecall, "\nMicrorecall =", microrecall, "\n")
    cat("Macroprecision =", macroprecision, "\nMicroprecision =", microprecision, "\n")
    cat("macro F-measure =", macroFM, "\nmicro F-measure =", microFM, "\n")
    
    t.acc <- t.acc + accuracy;
    t.fm.mac <- t.fm.mac + macroFM; t.fm.mic <- t.fm.mic + microFM;
    t.rec.mac <- t.rec.mac + macrorecall; t.rec.mic <- t.rec.mic + microrecall;
    t.pres.mac <- t.pres.mac + macroprecision; t.pres.mic <- t.pres.mic + microprecision;

    startpoint <- endpoint+1
  }
  t.acc <- t.acc / 10;
  t.fm.mac <- t.fm.mac / 10; t.fm.mic <- t.fm.mic / 10;
  t.rec.mac <- t.rec.mac / 10; t.rec.mic <- t.rec.mic / 10;
  t.pres.mac <- t.pres.mac / 10; t.pres.mic <- t.pres.mic / 10;

  cat("=====\n\n\nAvg acc", t.acc, "\nAvg macro F measure", t.fm.mac, "\nAvg micro F measure", t.fm.mic, "\n")
  cat("Avg macro recall", t.rec.mac, "\nAvg micro recall", t.rec.mic, "\n")
  cat("Avg macro precision", t.pres.mac, "\nAvg micro precision", t.pres.mic, "\n")
  print(totaltable)
}
tenfold(sasod.train, "svm")
tenfold(sasod.train, "rf")
tenfold(sasod.train, "nb")




return.measures <- function(inputtable) {
  tp = sum(diag(inputtable))
  # the number of positive class instances we recognise correctly
  accuracy = tp / nrow(sasod.test)
  macrorecall = 0
  for (j in 1:nrow(inputtable)) { macrorecall = macrorecall + (inputtable[j,j]/sum(inputtable[j,])) }
  macrorecall = macrorecall / nrow(inputtable)
  macroprecision = 0
  for (j in 1:nrow(inputtable)) { macroprecision = macroprecision + (inputtable[j,j]/sum(inputtable[,j])) }
  macroprecision = macroprecision / nrow(inputtable)
  microrecall = sum(diag(inputtable)) / sum(inputtable)
  microprecision = sum(diag(inputtable)) / sum(inputtable)
  macroFM = ( 2 * macroprecision * macrorecall ) / (macroprecision + macrorecall)
  microFM = ( 2 * microprecision * microrecall ) / (microprecision + microrecall)

  cat("Accuracy =", accuracy, "\n")
  cat("Macrorecall =", macrorecall, "\nMicrorecall =", microrecall, "\n")
  cat("Macroprecision =", macroprecision, "\nMicroprecision =", microprecision, "\n")
  cat("macro F-measure =", macroFM, "\nmicro F-measure =", microFM, "\n")
}




final.model <- svm(sasod.train[,11] ~., data=sasod.train[,1:10])
return.measures(table(ACTUAL = sasod.test[,11], PREDICTED = predict(final.model, sasod.test[,1:10])))
table(ACTUAL = sasod.test[,11], PREDICTED = predict(final.model, sasod.test[,1:10]))

# Cluster on undiscretised data and discretised data
clusterdata.cont <- sas.ordered[,1:10]
clusterdata.disc <- saso.disc[,1:10]
#K-Means for continuous and discretised
km.cont <- kmeans(clusterdata.cont, 10)
table(ClusterAssignment=km.cont$cluster, ClassLabel=sas.ordered[,11])
km.disc <- kmeans(clusterdata.disc, 10)
table(ClusterAssignment=km.disc$cluster, ClassLabel=saso.disc[,11])
km.disc$cluster

#DBscan
db.cont <- dbscan(clusterdata.cont, 0.1)
db.cont$cluster
db.disc <- dbscan(clusterdata.disc, 0.8)
db.disc$cluster


dist.m.cont <- dist(clusterdata.cont, method = "euclidean") # distance matrix
dist.m.disc <- dist(clusterdata.disc, method = "euclidean") # distance matrix
hc.cont <- hclust(dist.m.cont)
hc.disc <- hclust(dist.m.disc)
plot(hc.cont) # display dendogram
plot(hc.disc) # display dendogram
