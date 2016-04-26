
library(magrittr)
library(caret)
library(xgboost)
library(plyr)
library(pROC)
library(ggplot2)


# cleaning and pre-processing
if(file.exists('data/train_test.rda')){
    
    cat('Loading pre-cleaned data\n')
	load('data/train_test.rda')
	train_df <- train_test_l[[1]]
	X_submit <- train_test_l[[2]]

} else {
    
    cat('Cleaning data\n')
    
    # read in data
    train_csv <- read.csv('data/train.csv', row.names='ID') # takes ~ 12s
    submit_csv <- read.csv('data/test.csv', row.names='ID') # takes ~ 12s
    
	source('cleaner.R')
	train_df <- train_csv %>% 
	    f_rename %>% 
	    f_cut(train=F) %>% 
	    f_replace %>% 
	    f_add_age %>%
	    f_add_2way(train=T,k=20)
	
	X_submit <- submit_csv %>% 
	    f_rename %>% 
	    f_cut(train=F) %>% 
	    f_replace %>% 
	    f_add_age %>%
	    f_add_2way(train=F)
	
	X_submit <- as.matrix(X_submit)
	train_test_l <- list(train_df,X_submit)
	save(train_test_l, file='data/train_test.rda')
}
X <- as.matrix(train_df[,!names(train_df)=='TARGET'])
y <- train_df$TARGET

# cross-validation on full training data
cat('Beginning Cross-Validation\n')
nrounds <- 1000
params <- expand.grid(eta = c(0.02,0.0175,0.015),
					  max.depth = 5,
					  colsample_bytree = c(0.7,0.5,0.3),
					  subsample = c(0.7,0.5,0.3),
					  gamma = 0,
					  min_child_weight = 1)
cv_res <- data.frame()
for (i in 1:nrow(params)){
	t_i <- proc.time()
	set.seed(290615)
	xgb_i <- xgb.cv(data = X,
					label = y,
					nrounds = nrounds,
					objective = 'binary:logistic',
					eta = params[i,'eta'],
					max.depth = params[i,'max.depth'],
					gamma = params[i,'gamma'],
					min_child_weight = params[i,'min_child_weight'],
					colsample_bytree = params[i,'colsample_bytree'],
					subsample = params[i,'subsample'],
					missing = NA,
					nfold = 5,
					metrics = list('auc','logloss','error'),
					early.stop.round = 150,
					maximize=F,
					verbose=0)
	print(paste('Round',i,'took',round((proc.time()-t_i)[3]),'s'))
	cv_res <- rbind(cv_res,data.frame(auc = xgb_i$test.auc.mean,
								auc_lo = xgb_i$test.auc.mean-2*xgb_i$test.auc.std,
								auc_hi = xgb_i$test.auc.mean+2*xgb_i$test.auc.std,
								nrounds = 1:nrow(xgb_i),
								eta = params[i,'eta'],
								max.depth = params[i,'max.depth'],
								gamma = params[i,'gamma'],
								min_child_weight = params[i,'min_child_weight'],
								colsample_bytree = params[i,'colsample_bytree'],
								subsample = params[i,'subsample']))
	save(cv_res,file='data/cv_res.rda')
}


# visualize cross-validated scores
best <- cv_res[which.max(cv_res$auc),]
xgb_name <- paste0('xgb_',paste0(best[1,],'_',names(best),collapse='_'))
p <- ggplot(cv_res)
p <- p + geom_point(aes(x=nrounds,y=auc,col=max.depth),size=0.5,alpha=0.8)
p <- p + geom_line(aes(x=nrounds,y=auc_lo,col=max.depth),alpha=0.5)
p <- p + geom_line(aes(x=nrounds,y=auc_hi,col=max.depth),alpha=0.5)
p <- p + facet_grid(subsample~colsample_bytree)
ggsave(filename=paste0('plots/tune_auc_',xgb_name,'.png'), plot=p)


# train on full training data
print('Training on full data')
set.seed(290615)
param_best <- list('objective' = 'binary:logistic',
				   'eta' = best$eta,
				   'max.depth' = best$max.depth,
				   'gamma' = best$gamma,
				   'min_child_weight' = best$min_child_weight,
				   'colsample_bytree' = best$colsample_bytree,
				   'subsample' = best$subsample)
xgb_test <- xgboost(data = X,
					label = y,
					params = param_best,
					nrounds = best$nrounds,
					verbose = 0,
					missing = NA)
xgb.save(model = xgb_test, fname = paste0('models/',xgb_name,'.rda'))

# predict on submission set
print('Predicting for submission')
submit <- data.frame(ID = rownames(X_submit), TARGET = predict(xgb_test,X_submit,missing=NA))
xgb_path <- paste0('data/submit_',xgb_name,'.csv')
write.csv(submit, file=xgb_path, row.names=F)
print(paste('Exported submission to',xgb_path))

