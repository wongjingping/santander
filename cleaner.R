
### bunch of helper functions for cleaning the data

# rename columns
f_rename <- function(train_df){
	cnames <- read.csv('ref/Spanish2English.csv',stringsAsFactors=F)
	cnames$English <- gsub(' ','_',cnames$English,fixed=T)
	oldc <- data.frame('Spanish'=names(train_df),stringsAsFactors=F)
	newc <- cnames[match(oldc$Spanish,cnames$Spanish),'English']
	newc <- ifelse(is.na(newc),oldc$Spanish,newc)
	names(train_df) <- newc
	return(train_df)
}


# CUT columns/rows
f_cut <- function(train_df,train=F){
	
	t_clean <- proc.time()
	if(train) {
		
		## column cleaning
		col_var <- apply(train_df,2,var)
		novar_cols <- which(col_var == 0)
		print(paste(sum(col_var==0),'columns with 0 variance'))
		# 34 columns with 0 variance, let's remove them
		dup_cols <- which(duplicated(t(train_df)))
		print(paste(length(dup_cols),'duplicated columns'))
		# 62 duplicated cols, let's remove them
		cor_m <- cor(train_df)
		cor_ix <- which(cor_m==1 & lower.tri(cor_m),arr.ind = T)
		cor_cols <- setdiff(cor_ix[,1],cor_ix[,2])
		print(paste(length(cor_cols),'extra perfectly correlated columns'))
		unused_cols <- names(train_df)[unique(c(novar_cols,dup_cols,cor_cols))]
		write(unused_cols,file='data/unused_cols')
		
		## row cleaning
		# sum(is.na(train_df)) # no missing values, yay!
		# sum(dup_rows) # 4807 duplicated rows, let's remove them
		dup_rows <- duplicated(train_df)
		print(paste(sum(dup_rows),'duplicated rows'))
		train_df <- train_df[!dup_rows,]
		
	} else {
		unused_cols <- read.table('data/unused_cols')[,1]
	}
	train_df <- train_df[,!names(train_df) %in% unused_cols]
	print(paste('Cleaning took',round((proc.time()-t_clean)[3]),'s'))
	return(train_df)
}


# EDIT entries
f_replace <- function(train_df){
	# clean min_int, max_int
	min_int <- apply(train_df,2,function(x){-999999 %in% x})
	# table(train_df$var3)
	for (col_i in which(min_int)) {
		train_df[,col_i] <- ifelse(train_df[,col_i] == -999999, NA, train_df[,col_i])
	}
	max_int <- apply(train_df,2,function(x){9999999999 %in% x})
	for (col_i in which(max_int)) {
		train_df[,col_i] <- ifelse(train_df[,col_i] == 9999999999, NA, train_df[,col_i])
	}
	return(train_df)
}


# ADD binary columns (feature engineering)
f_add_bin <- function(train_df, train=F){
	if (train) {
		# add if > 0 flag for skewed fields
		zeros <- apply(train_df,2,function(x){sum(x==0,na.rm=T)})
		is_bin <- apply(train_df,2,function(x){all(x %in% 0:1)})
		# visualize distribution of columns' zero sums
		# qplot(zeros,binwidth=1e3)
		thresh_zeros <- 6e4
		add_bin <- names(train_df)[zeros > thresh_zeros & !is_bin]
		save(add_bin,file='data/add_bin.rda')
	} else {
		load('data/add_bin.rda')
	}
	new_df <- apply(train_df[add_bin],2,function(x){ifelse(x==0,0,1)})
	colnames(new_df) <- paste0('ind_',colnames(new_df))
	new_df <- cbind(new_df,'ind_sum'=rowSums(new_df))
	train_df <- data.frame(train_df,new_df)
	return(train_df)
}

f_add_age <- function(train_df){
	age_bin <- cut(train_df$age,breaks = c(-Inf,seq(20,100,10),Inf))
	age_m <- model.matrix(~.-1,as.data.frame(age_bin))
	train_df <- cbind(train_df,age_m)
	return(train_df)
}
