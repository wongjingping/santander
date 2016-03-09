
library(caret)
library(xgboost)
library(plyr)
library(pROC)
library(ggplot2)

if(all(file.exists('data/train.rda','data/used_col.rda'))){
	load('data/train.rda') # when training
	load('data/used_col.rda')
} else {
	# when running it the first time
	train_df <- read.csv('data/train.csv', row.names='ID') # takes ~ 12s
	
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
	save(train_df, file='data/train.rda')
	save(used_col,file='data/used_col.rda')
}

# function that bins instead of splitting by converting numeric to factors
# to prevent unseen values in test set from breaking code
bin_column <- function(column,col_bin){
	if(all(is.na(col_bin))){
		return(as.numeric(column))
	} else {
		col_bin <- c(-Inf,col_bin,Inf)
		xi <- cut(column,breaks=col_bin)
		mm = cbind(model.matrix(~.-1,as.data.frame(xi)),column)
		return(mm)
	}
}

if(all(file.exists('data/X.rda','data/y.rda','data/bins.rda'))){
	load('data/X.rda')
	load('data/y.rda')
	load('data/bins.rda')
} else {

	y <- train_df$TARGET
	train_X <- train_df[,names(train_df)!='TARGET']
	# convert numeric columns with > 2 & <= 5 unique values to factors
	bins <- apply(train_X,2,function(x){
		uniq_x <- unique(x)
		if(length(uniq_x) > 1000) {
			q <- quantile(uniq_x,probs=seq(0,1,0.2))
			return(q)
		} else return(NA)
	})
	print(paste(sum(sapply(bins,is.numeric)),'columns with > 2000 unique values'))
	X <- as.matrix(as.data.frame(mapply(FUN=bin_column,train_X,bins)))

	save(X,file='data/X.rda')
	save(y,file='data/y.rda')
	save(bins,file='data/bins.rda')
}


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
param.grid <- expand.grid(eta = c(0.03),
						  nrounds = seq(50,500,50),
						  max_depth = c(3,4,5),
						  gamma = 0,
						  colsample_bytree = c(.3,.5,.7),
						  min_child_weight = c(1))
xgb_train <- train(X_train, y_train,
				   method = 'xgbTree',
				   metric = 'ROC',
				   tuneGrid = param.grid,
				   trControl = trCtrl)
print(paste('Training took',round(proc.time()-t)[3],'s'))


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
xgb_test <- xgboost(data = X,
					label = y,
					params = param_list,
					nrounds = best$nrounds,
					verbose = 0)
save(xgb_test,file=paste0('models/',xgb_name,'.rda'))

# predict for submission
if(file.exists('data/X_submit.rda')){
	load('data/X_submit.rda')
	load('data/ID.rda')
} else {
	submit <- read.csv('data/test.csv', row.names='ID')
	X_submit <- as.matrix(as.data.frame(mapply(FUN=bin_column,submit[,head(used_col,-1)],bins)))
	ID <- as.integer(row.names(submit))
	save(X_submit,file='data/X_submit.rda')
	save(ID,file='data/ID.rda')
}
submit <- data.frame(ID = ID, TARGET = predict(xgb_test,X_submit))
xgb_path <- paste0('data/submit_',xgb_name,'.csv')
write.csv(submit, file=xgb_path, row.names=F)
print(paste('Exported submission to',xgb_path))
