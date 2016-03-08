
library(caret)
library(xgboost)
library(pROC)
library(ggplot2)

if(file.exists('data/train.rda')){
	load('data/train.rda') # when training 
} else {
	# when running it the first time
	train_df <- read.csv('data/train.csv', row.names='ID') # takes ~ 12s
	save(train_df, file='data/train.rda')
}

# EDA to remove duplicates and useless columns
a <- apply(train_df,2,function(x){class(x) %in% c('integer','numeric')})
# sum(a) # 370 all cols are numbers (no strings or dates)
col_var <- apply(train_df,2,var)
sum(col_var==0) # 34 columns with 0 variance, let's remove them
# sum(is.na(train_df)) # no missing values, yay!
dup_rows <-duplicated(train_df)
# sum(dup_rows) # 4807 duplicated rows, let's remove them
dup_cols <- duplicated(t(train_df))
# sum(dup_cols) # 62 duplicated cols, let's remove them
used_col <- ifelse((col_var > 0) & (!dup_cols),T,F)

train_df <- train_df[!dup_rows,used_col]
# train-test split
set.seed(290615)
train_i <- sample(nrow(train_df),0.7*nrow(train_df))
df_train <- train_df[train_i,]
df_test <- train_df[-train_i,]

y <- factor(ifelse(df_train$TARGET,'Y','N'))
X <- df_train[,names(df_train) != 'TARGET']

# train basic xgboost with simple 5-fold cv
t <- proc.time()
trCtrl <- trainControl(method = 'cv', 
					   number = 3, 
					   verboseIter = T,
					   summaryFunction = twoClassSummary,
					   classProbs = T)
param.grid <- expand.grid(eta = c(0.03),
						  nrounds = seq(50,500,50),
						  max_depth = c(2,3,4),
						  gamma = 0,
						  colsample_bytree = c(.7,.5),
						  min_child_weight = 1)
xgb_train <- train(X, y,
				   method = 'xgbTree',
				   metric = 'ROC',
				   tuneGrid = param.grid,
				   trControl = trCtrl)
print(paste('Training took',round(proc.time()-t)[3],'s')) # 12 mins on a 8-core RHEL server


p <- ggplot(xgb_train$results)
p <- p + geom_point(aes(x=nrounds,y=ROC,col=max_depth))
p <- p + facet_grid(colsample_bytree~eta)
p
ggsave(filename='plots/tune_auc.png', plot=p)
best <- xgb_train$bestTune

# predict on test set
df_roc <- data.frame(obs=df_test$TARGET,
					 pred=predict(xgb_train,df_test[,names(df_train)!='TARGET'],type='prob')$Y)
df_roc <- df_roc[order(df_roc$pred),]
roc_test <- roc(df_roc$obs,df_roc$pred,plot=T)
roc_test$auc

# train on full training data
param_list <- list('objective' = 'binary:logistic',
				   'eta' = best$eta,
				   'max.depth' = best$max_depth,
				   'gamma' = best$gamma,
				   'min_child_weight' = best$min_child_weight,
				   'colsample_bytree' = best$colsample_bytree)
xgb_test <- xgboost(data = as.matrix(train_df[,!names(train_df)=='TARGET']),
					label = train_df$TARGET,
					params = param_list,
					nrounds = best$nrounds,
					verbose = 0)

# predict for submission
used_col <- used_col[-length(used_col)]
submit <- read.csv('data/test.csv', row.names='ID')
submit <- data.frame(ID = as.integer(rownames(submit)),
					 TARGET = predict(xgb_train,submit[,used_col], type='prob')$Y)
write.csv(submit, file='data/submit_xgb3.csv', row.names=F)
