
f_subs <- list.files('data')[grepl('^submit_',list.files('data'))]

sub_l <- lapply(paste0('data/',f_subs),read.csv)
ens <- data.frame(ID = sub_l[[1]]$ID,
				  TARGET = rowMeans(sapply(sub_l,function(x){x$TARGET})))
write.csv(ens,file='data/ens_1.csv',row.names=F)
