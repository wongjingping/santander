
f_subs <- list.files('submissions')[grepl('^submit_',list.files('submissions'))]

sub_l <- lapply(paste0('submissions/',f_subs),read.csv)
ens <- data.frame(ID = sub_l[[1]]$ID,
				  TARGET = rowMeans(sapply(sub_l,function(x){x$TARGET})))
write.csv(ens,file='submissions/ens_4.csv',row.names=F)
