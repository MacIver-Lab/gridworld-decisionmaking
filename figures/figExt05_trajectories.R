library(dplyr)
library(igraph)

cat("\014") # Clear console
this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)
rm(list = ls()) # Clear environment

XSize <- 15
YSize <- 15

x <- 0:(XSize-1)
y <- 0:(YSize-1)

xy <- expand.grid(x, y)
XY <- as.matrix(xy)

get_grid_adjacency_matrix <- function(occlusion){
  g <- make_lattice(dimvector=c(XSize, YSize))
  ga <- as_adj(g)
  
  ga2 <- matrix(0, nrow=225, ncol=225)
  
  for(i in 1:nrow(XY)){
    #cat("iteration i#: ", i)
    coord1x <- XY[i, 1]
    coord1y <- XY[i, 2]
    df_coord1 <- data.frame(X=coord1x, Y=coord1y)
    if(nrow(inner_join(df_coord1, occlusion, by=c("X","Y")))) next
    
    for(j in 1:nrow(XY)){
      coord2x <- XY[j, 1]
      coord2y <- XY[j, 2]
      df_coord2 <- data.frame(X=coord2x, Y=coord2y)
      if(nrow(inner_join(df_coord2, occlusion, by=c("X", "Y")))) next
      
      ga2[coord1x+1+(coord1y*XSize), coord2x+1+(coord2y*XSize)] <- ga[coord1x+1+(coord1y*XSize), coord2x+1+(coord2y*XSize)]
    }
    #cat("\014")
  }
  
  return(ga2)
  
}

create_world_graph <- function(entropy.val, simulation.val){
  
  occlusion.path <- sprintf("../Data_NatComm/Simulation_%d/Occlusion_%d/OcclusionCoordinates.csv",simulation.val, entropy.val)
  occlusion.coordinates <-read.csv(occlusion.path, header=TRUE)
  world.adjacency <- get_grid_adjacency_matrix(occlusion.coordinates)
  world.graph <- graph_from_adjacency_matrix(world.adjacency, mode="undirected")
  
  V(world.graph)$size <- 0
  V(world.graph)$label <- NA
  occlusion.nodes <- which(degree(world.graph)==0)
  V(world.graph)$color <- "#ffffff00"
  V(world.graph)[occlusion.nodes]$color <- "#000000"
  V(world.graph)[occlusion.nodes]$size <- 15
  V(world.graph)$frame.color <- "#ffffff00"
  E(world.graph)$width <- 0.2
  E(world.graph)$color <- "#ffffff00"
  
  episode.base.path <- sprintf("../Simulation 2 Pseudo-Terrestrial/Data/Simulation_%d/Occlusion_%d",simulation.val, entropy.val)
  episode.paths <- list.files(path=episode.base.path, recursive=T, pattern="Episode_.*\\.csv", full.names=T) # Extract all episode files
  
  agentTrajectory = matrix(0, nrow=XSize*YSize, ncol=1)
  for (episode.path in episode.paths){
    depth <- as.numeric(gsub(".*Depth_", "", gsub("/Episode.*", "", episode.path)))
    if (depth != 5000){
      next
    }
    
    episode <- read.csv(episode.path, header=TRUE)
    terminalReward <- episode$Reward[nrow(episode)]
    if(terminalReward < 0){
      next
    }
    
    episode.new <- episode[!duplicated(episode[,c('Agent.X', 'Agent.Y')]),]
    #episode.new <- episode
    for(t in 1:(nrow(episode.new)-1)){
      agentTrajectory[episode.new$Agent.X[t]+1+(episode.new$Agent.Y[t]*XSize)] = 
        agentTrajectory[episode.new$Agent.X[t]+1+(episode.new$Agent.Y[t]*XSize)] + 1
      if (agentTrajectory[episode.new$Agent.X[t]+1+(episode.new$Agent.Y[t]*XSize)] > 0){
        vertexInd <- episode.new$Agent.X[t]+1+(episode.new$Agent.Y[t]*XSize)
        nextVertexInd <- episode.new$Agent.X[t+1]+1+(episode.new$Agent.Y[t+1]*XSize)
        edgeId <- get.edge.ids(world.graph, c(vertexInd, nextVertexInd))
        E(world.graph)[edgeId]$width <- agentTrajectory[vertexInd]
      }
    }
  }
  
  return(world.graph)
}

palf <- colorRampPalette(c(rgb(255/255, 255/255, 255/255), rgb(192/255, 212/255, 229/255), rgb(140/255, 151/255, 196/255),
                           rgb(127/255, 33/255, 123/255)))

color.bar <- function(lut, min, max=-min, nticks=11, ticks=seq(min, max, len=nticks), title='') {
  scale = (length(lut)-1)/(max-min)
  
  dev.new(width=11, height=5)
  pdf(file="preypath_colorbar.pdf")
  plot(c(0,10), c(min,max), type='n', bty='n', xaxt='n', xlab='', yaxt='n', ylab='', main=title)
  axis(2, ticks, las=1)
  for (i in 1:(length(lut)-1)) {
    y = (i-1)/scale + min
    rect(0,y,10,y+1/scale, col=lut[i], border=NA)
  }
  dev.off()
}


n <- 1
simulation.list <- matrix(c(12, 4, 4, 4, 4, 4, 3, 4, 4, 4, 
                            5, 13, 15, 15, 13, 15, 13, 14, 15, 12,
                            6, 12, 10, 19, 19, 14, 19, 5, 19, 1, 
                            7, 0, 0, 0, 0, 8, 0, 1, 7, 0, 
                            14, 2, 11, 10, 10, 9, 10, 9, 10, 14), nrow=5, ncol=10, byrow = TRUE)

n <- 1
plot.name <- "Plots/figExt05_trajectories/figExt05_trajectories.pdf"
pdf(plot.name, height=9*2, width=7*2+3, useDingbats = FALSE)

par(mfrow= c(5, 10), oma=c(0, 0, 0, 0)+0.1, mar=c(0, 0, 1, 0)+0.1)

create_graph = FALSE
if(file.exists("saved/figExt05_trajectories.rdata")){
  load(file="saved/figExt05_trajectories.rdata")
} else{
  create_graph = TRUE
  world.trajectories = vector("list", length(10*length(simulation.list)))
}

for (n in 1:5){
  simulation.list.row <- simulation.list[n,]
  for (n1 in 1:length(simulation.list.row)){
    
    simulation.val <- simulation.list.row[n1]
    entropy.val <- n1 - 1
    cat("\nSimulation #:, Entropy .# ", simulation.val, entropy.val)
    
    if(create_graph){
      world.graph <- create_world_graph(entropy.val, simulation.val)
      world.trajectories[[((n-1)*10)+n1]] <- world.graph
    } else{
      world.graph <- world.trajectories[[((n-1)*10)+n1]]
    }
    
    szs <- seq(2, 15, length=max(E(world.graph)$width, na.rm=TRUE))
    clrs <- palf(max(E(world.graph)$width, na.rm=TRUE))
    widths = E(world.graph)$width
    for(i in 1:length(E(world.graph)$width)){
      width <- widths[i]
      if(width>0.2){
        E(world.graph)[i]$width <- szs[width]
        E(world.graph)[i]$color <- clrs[width]
      }
    }
    
    plot(world.graph, vertex.shape="square", layout=layout_on_grid(world.graph, width = XSize, height = YSize))
  }
}

if(create_graph){
  save(world.trajectories, file="saved/figExt05_trajectories.rdata")
}

dev.off()