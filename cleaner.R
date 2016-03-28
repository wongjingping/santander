
### bunch of helper functions for cleaning the data

# rename columns
f_rename <- function(train_df){
	cnames <- read.csv('data/Spanish2English.csv',stringsAsFactors=F)
	cnames$English <- gsub(' ','_',cnames$English,fixed=T)
	oldc <- data.frame('Spanish'=names(train_df),stringsAsFactors=F)
	newc <- cnames[match(oldc$Spanish,cnames$Spanish),'English']
	newc <- ifelse(is.na(newc),oldc$Spanish,newc)
	names(train_df) <- newc
	return(train_df)
}


# CUT columns/rows
f_cut <- function(train_df,train=F){
	if(train) {
		col_var <- apply(train_df,2,var)
		print(paste(sum(col_var==0),'columns with 0 variance'))
		# names(train_df)[col_var == 0] # uncomment to see the col names
		# 34 columns with 0 variance, let's remove them
		# sum(is.na(train_df)) # no missing values, yay!
		# sum(dup_rows) # 4807 duplicated rows, let's remove them
		dup_rows <- duplicated(train_df)
		train_df <- train_df[!dup_rows,]
		# sum(dup_cols) # 62 duplicated cols, let's remove them
		dup_cols <- duplicated(t(train_df))
		unused_col <- names(col_var)[ifelse((col_var == 0) | dup_cols,T,F)]
		save(unused_col,file='data/unused_col.rda')
	} else {
		load('data/unused_col.rda')
	}
	train_df <- train_df[,!names(train_df) %in% unused_col]
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


