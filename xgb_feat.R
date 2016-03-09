
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
# a <- apply(train_df,2,function(x){class(x) %in% c('integer','numeric')})
# sum(a) # 370 all cols are numbers (no strings or dates)
col_var <- apply(train_df,2,var)
print(paste(sum(col_var==0),'columns with 0 variance')) # 34 columns with 0 variance, let's remove them
# sum(is.na(train_df)) # no missing values, yay!
dup_rows <- duplicated(train_df)
# sum(dup_rows) # 4807 duplicated rows, let's remove them
dup_cols <- duplicated(t(train_df))
# sum(dup_cols) # 62 duplicated cols, let's remove them
used_col <- ifelse((col_var > 0) & (!dup_cols),T,F)

train_df <- train_df[!dup_rows,used_col]
y <- factor(ifelse(train_df$TARGET,'Y','N'))

# convert numeric columns with > 2 & <= 5 unique values to factors
n_uniq <- apply(train_df,2,function(x){length(unique(x))})
uniq2_5 <- n_uniq <= 5 & n_uniq > 2
print(paste(sum(uniq2_5),'columns with > 2 & <= 5 unique values'))
train_df[,uniq2_5] <- apply(train_df[,uniq2_5],2,factor)
X <- model.matrix(TARGET~.-1,data=train_df)

# TODO bin instead of splitting by factors to prevent unseen values in test set from breaking code

# train-test split
set.seed(290615)
train_i <- sample(nrow(train_df),0.7*nrow(train_df))
X_train <- X[train_i,]
X_test <- X[-train_i,]
y_train <- y[train_i]
y_test <- y[-train_i]

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
						  min_child_weight = c(1))
xgb_train <- train(X_train, y_train,
				   method = 'xgbTree',
				   metric = 'ROC',
				   tuneGrid = param.grid,
				   trControl = trCtrl)
print(paste('Training took',round(proc.time()-t)[3],'s')) # 12 mins on a 8-core RHEL server


p <- ggplot(xgb_train$results)
p <- p + geom_point(aes(x=nrounds,y=ROC,col=max_depth))
p <- p + facet_grid(colsample_bytree~min_child_weight)
p
ggsave(filename='plots/tune_auc.png', plot=p)
best <- xgb_train$bestTune

# predict on test set
df_roc <- data.frame(obs=y_test,
					 pred=predict(xgb_train,X_test,type='prob')$Y)
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
