
library(magrittr)
library(caret)
library(xgboost)
library(plyr)
library(pROC)
library(ggplot2)


# read in data
train_csv <- read.csv('data/train.csv', row.names='ID') # takes ~ 12s
submit_csv <- read.csv('data/test.csv', row.names='ID') # takes ~ 12s

# cleaning and pre-processing
source('cleaner.R')
train_df <- train_csv %>% f_rename %>% f_cut(train=T) %>% f_replace %>% f_add_bin(train=T)
X <- train_df[,!names(train_df)=='TARGET']
y <- train_df$TARGET
X_submit <- submit_csv %>% f_rename %>% f_cut(train=F) %>% f_replace %>% f_add_bin(train=F)
train_test_l <- list(train_df,X_submit)
save(train_test_l, file='data/train_test.rda')

# train-test split
set.seed(290615)
train_i <- sample(nrow(X),0.7*nrow(X))
X_train <- X[train_i,]
X_test <- X[-train_i,]
y_train <- factor(ifelse(y,'Y','N'))[train_i]
y_test <- factor(ifelse(y,'Y','N'))[-train_i]

# train basic xgboost with simple 5-fold cv
t <- proc.time()
trCtrl <- trainControl(method = 'cv', 
					   number = 5, 
					   verboseIter = T,
					   summaryFunction = twoClassSummary,
					   classProbs = T)
param.grid <- expand.grid(eta = c(0.03,0.02),
						  nrounds = seq(50,500,50),
						  max_depth = c(4,5,6),
						  gamma = 0,
						  colsample_bytree = c(.5,.6,.8),
						  min_child_weight = c(1))
xgb_train <- train(X_train, y_train,
				   method = 'xgbTree',
				   metric = 'ROC',
				   tuneGrid = param.grid,
				   missing = NA,
				   trControl = trCtrl)
print(paste('Training took',round(proc.time()-t)[3],'s'))

# visualize cross-validated scores
best <- xgb_train$bestTune
xgb_name <- paste0('xgb_',paste0(best[1,],'_',names(best),collapse='_'))
p <- ggplot(xgb_train$results)
p <- p + geom_point(aes(x=nrounds,y=ROC,col=max_depth))
p <- p + facet_grid(eta~colsample_bytree)
ggsave(filename=paste0('plots/tune_auc_',xgb_name,'.png'), plot=p)

# predict on test set
df_roc <- data.frame(obs=y_test,
					 pred=predict(xgb_train,X_test,type='prob')$Y)
df_roc <- df_roc[order(df_roc$pred),]
roc_test <- roc(df_roc$obs,df_roc$pred,plot=T)
roc_test$auc

# train on full training data
set.seed(290615)
param_list <- list('objective' = 'binary:logistic',
				   'eta' = best$eta,
				   'max.depth' = best$max_depth,
				   'gamma' = best$gamma,
				   'min_child_weight' = best$min_child_weight,
				   'colsample_bytree' = best$colsample_bytree)
xgb_test <- xgboost(data = as.matrix(X),
					label = y,
					params = param_list,
					nrounds = best$nrounds,
					verbose = 0,
					missing = NA)
save(xgb_test,file=paste0('models/',xgb_name,'.rda'))

# predict on submission set
submit <- data.frame(ID = rownames(X_submit), TARGET = predict(xgb_test,as.matrix(X_submit),missing=NA))
xgb_path <- paste0('data/submit_',xgb_name,'.csv')
write.csv(submit, file=xgb_path, row.names=F)
print(paste('Exported submission to',xgb_path))

