rm(list = ls(all = TRUE))

setwd('/yourDirectory')

library(pracma)
library(plot.matrix)
library(data.table)

#parameters
iterations=8000 #how many iterations, not including 12 testing iterations. 4k iterations gives 1k instances of each input
nnodes=50 #number of nodes
lrate_wmat=.01 #learning rate
lrate_targ=.01
p_link=.1 #probability of a link from input nodes to the reservoir network

## Set up inputs as one-hot encodings. this vocab has 4 possible units, a-d ####
inputs=list(
  a=c(1,0,0,0),
  b=c(0,1,0,0),
  c=c(0,0,1,0),
  d=c(0,0,0,1))

#setting up the transition matrix. this one just goes a,b,c,d, repeat in order
grammarmat=matrix(0,ncol=4,nrow=4)
grammarmat[1,2]=1; grammarmat[2,3]=1; grammarmat[3,4]=1; grammarmat[4,1]=1;

#create random weight matrix mapping input nodes to all other nodes -- these don't change 
input_wmat=matrix(0,ncol=nnodes,nrow=length(inputs))
for(row in 1:nrow(input_wmat)){
  for(col in 1:nnodes){
    input_wmat[row,col]=ifelse(runif(1,0,1) <= p_link, 10,0)
  }
}

#initialize a random weight matrix within the network -- these can change within the lifespan
wmat=matrix(0,ncol=nnodes,nrow=nnodes)

## uncomment below to randomize initial weights 
# for(row in 1:nrow(wmat)){
#   for(col in 1:ncol(wmat)){
#     wmat[row,col]=runif(1,-1,1)
#   }
# }
# diag(wmat)=0 #there are no self-connections

## initialize some arrays for saving data
spikes=c(rep(0,nnodes)) #the activations of all nodes 
act=c(rep(0,nnodes))
targ_min=1 #this the starting & minimum activation value that nodes will try to maintain -- this updates
target=c(rep(targ_min,nnodes))

## uncomment below to randomize initial targets
# for(n in 1:nnodes){ #uncomment this block if we want to start nodes with random targets
#   target[n] = sample(1:5,1)
# }

#these next arrays save the values over time
start_spikes=c()
start_act=c()
start_target=c()
start_stream=c() #this saves the inputs

end_spikes=c()
end_act=c()
end_target=c()
end_stream=c() #this saves the inputs

errors=c()

#grab the first input ('a' by default, but uncomment below to start with a random choice)
input_id=1 # sample(seq(1:length(inputs)),1)
input=inputs[[input_id]]

learning=1 # set this to optionally turn off learning during the test phase

for(i in 1:(iterations+4)){ # we have i iterations of training + 4 iterations with no input for testing
  print(i)
  
  prev_spikes = spikes #save the last spike pattern across all nodes

  #get the current activations
  if(i > iterations){ # if we are in the last 4 iterations, we turn off the inputs
    act= dot(spikes,wmat) 
    
    # can uncomment the below line if we also want to turn off learning during the test phase
    #learning=0 
  }else{ #otherwise, proceed as normal
    act= dot(input,input_wmat) + dot(spikes,wmat)
  }
  
  spikes[act >= target] = 1 #if activation value is >= the current target, the node spikes
  spikes[act < target] = 0
  error = act - target #error is the difference between activation value and target value

  errors=rbind(errors,cbind(mean(error),sd(error))) #save errors
  
  if( i <= 1000){ #save the first 1K iterations of data
    start_spikes=rbind(start_spikes,spikes)
    start_act=rbind(start_act,act)
    start_target=rbind(start_target,target)
    start_stream=rbind(start_stream,cbind(word=names(inputs)[input_id],i=i))
  }
  if( i > iterations-1000){ # save the last 1K iterations of data
    end_spikes=rbind(end_spikes,spikes)
    end_act=rbind(end_act,act)
    end_target=rbind(end_target,target)
    end_stream=rbind(end_stream,cbind(word=names(inputs)[input_id],i=i))
  }
  
  #go thru each node and adjust parameters deterministically
  if(learning==1){
    for(n in 1:nnodes){
      
      #get the set of neighbors who are active on the last iteration, excluding self. 
      prev_active=which(prev_spikes>0)[which(prev_spikes>0) != n]
      N = length(prev_active) + 1
      
      if(length(prev_active>0)){ #so long as some nodes were active...
        for(r in prev_active){ # go through each connection with nodes that were active
          wmat[r,n]=wmat[r,n] - (error[n]/N)*lrate_wmat #take the total error, divide it by the number of active nodes (+1, because we also adjust the target) and multiply by learning rate
        }
      }
      
      target[n] = target[n] + (error[n]/N)*lrate_targ #adjust the target according to the same learning rule
      if(target[n]<targ_min){target[n]=targ_min} # if target goes below min value, reset it to min
      
    } 
  }
  
  #select a new input for next iteration
  input_id=sample(seq(1:length(inputs)),1,prob=c(grammarmat[input_id,]/sum(grammarmat[input_id,])))
  input=inputs[[input_id]]

  
}


## PLOTTING 

#get a histogram of the weight matrix
hist(wmat,breaks=100)

#re-label the last 4 iterations to note that it is expected input -- none was actually received here
for(i in 1:4){
  end_stream[1000+i,1]=paste0('(',end_stream[12+i,1],')')
}

## save & plot the spike-train for the FIRST 12 iterations (12 gives us 3 loops through the grammar matrix of 4 elements)
plot_spikes=start_spikes[1:12,]
plot_spikes=as.matrix(plot_spikes,ncol = nnodes)
rownames(plot_spikes)=start_stream[1:12,1]
plot_spikes=t(plot_spikes) #transpose
pdf('Fig1.pdf',width=8,height=8)
plot(plot_spikes,xlab="Input",ylab='Node',key=NULL,axis.row=NULL)
dev.off()

#get the spike train for the LAST 12 iterations
plot_spikes=end_spikes[(1004-15):1004,]
plot_spikes[13:16,][plot_spikes[13:16,]==0]=NA
plot_spikes=as.matrix(plot_spikes,ncol = nnodes)
rownames(plot_spikes)=end_stream[(1004-15):1004,1]
plot_spikes=t(plot_spikes) #transpose
pdf('Fig2.pdf',width=8,height=8)
plot(plot_spikes,na.col = 'blue',xlab='Input',ylab='Node',key=NULL,axis.row = NULL) # plot
dev.off()
plot_spikes[is.na(plot_spikes)]=0

## create an auto-correlation matrix for the last 16 iterations (12 with input + 4 test iterations)
cor_matrix=matrix(NA,nrow=16,ncol=16)
for(n in 1:16){
  x=end_spikes[988+n,]
  for(r in n:16){
    y=end_spikes[988+r,]
    res=cor(unlist(x),unlist(y))
    cor_matrix[r,n]=res
  }
}

rownames(cor_matrix)=end_stream[(1004-15):1004,1]
colnames(cor_matrix)=end_stream[(1004-15):1004,1]
library(RColorBrewer)
c1=brewer.pal(n = 7, name = "RdYlBu")
par(mar=c(5.1, 4.1, 4.1, 6.1))
pdf('Fig3.pdf',width=15,height=10)
plot(cor_matrix,
     col=c1,
     breaks=7,
     digits=2, 
     text.cell=list(cex=1),
     key=list(side=4, cex.axis=2),
     border=NA,
     xlab='Iteration',
     ylab='Iteration',
     axis.row = list(side=2,las=1,cex.axis=2),
     axis.col = list(side=1,las=1,cex.axis=2),
     na.print=FALSE) # plot
dev.off()


