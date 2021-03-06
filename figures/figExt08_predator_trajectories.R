library(dplyr)
library(igraph)

cat("\014") # Clear console
this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(this.dir)
#rm(list = ls()) # Clear environment

XSize <- 15
YSize <- 15

x <- 0:(XSize-1)
y <- 0:(YSize-1)

xy <- expand.grid(x, y)
XY <- as.matrix(xy)

num.simulations <- 20

create_world_graph <- function(visualrange){
  V(world.graph)$size <- 0
  V(world.graph)$label <- NA
  occlusion.nodes <- which(degree(world.graph)==0)
  V(world.graph)$color <- "#ffffff00"
  V(world.graph)[occlusion.nodes]$color <- "#000000"
  V(world.graph)[occlusion.nodes]$size <- 15
  V(world.graph)$frame.color <- "#ffffff00"
  E(world.graph)$width <- 0.2
  E(world.graph)$color <- "#ffffff00"
  
  predatorTrajectory = matrix(0, nrow=XSize*YSize, ncol=1)
  for (simulation in 0:num.simulations-1){
    episode.base.path <- sprintf("../Simulation 1 Pseudo-Aquatic/Data/Simulation_%d/Vision_%d/Depth_5000", (simulation), (visualrange))
    episode.paths<- list.files(path=episode.base.path, recursive=T, pattern="Episode_", full.names=T) # Extract all episode files
    
    for (episode.path in episode.paths){
      episode <- read.csv(episode.path, header=TRUE)
      terminalReward <- episode$Reward[nrow(episode)]
      if(terminalReward > -2){
        next
      }
      
      episode.new <- episode[!duplicated(episode[,c('Predator.X', 'Predator.Y')]),]
      for(t in 1:(nrow(episode.new)-1)){
        predatorTrajectory[episode.new$Predator.X[t]+1+(episode.new$Predator.Y[t]*XSize)] = 
          predatorTrajectory[episode.new$Predator.X[t]+1+(episode.new$Predator.Y[t]*XSize)] + 1
        if (predatorTrajectory[episode.new$Predator.X[t]+1+(episode.new$Predator.Y[t]*XSize)] > 0){
          vertexInd <- episode.new$Predator.X[t]+1+(episode.new$Predator.Y[t]*XSize)
          nextVertexInd <- episode.new$Predator.X[t+1]+1+(episode.new$Predator.Y[t+1]*XSize)
          edgeId <- get.edge.ids(world.graph, c(vertexInd, nextVertexInd))
          E(world.graph)[edgeId]$width <- predatorTrajectory[vertexInd]
        }
      }
    }
  }
  
  return(world.graph)
}

palf <- colorRampPalette(c( rgb(220/255, 208/255, 35/255),rgb(204/255, 161/255, 23/255),rgb(198/255, 110/255, 23/255),
                            rgb(199/255, 56/255, 20/255), rgb(172/255, 0/255, 0/255)))

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

world.graph <- make_lattice(dimvector=c(XSize, YSize))

num.vision <- 5

n <- 1
plot.name <- "Plots/figExt08_predator_trajectories/figExt08.pdf"
pdf(plot.name, height=(4.5+1)/5, width=4.5+1, useDingbats = FALSE)
par(mfrow= c(1, 5), oma=c(0, 0, 0, 0), mar=c(0, 0, 1, 0))

create_graph = FALSE
if(file.exists("saved/figExt08_trajectories.rdata")){
  load(file="saved/figExt08_trajectories.rdata")
} else{
  create_graph = TRUE
  world.trajectories = vector("list", length(5))
}

for (visualrange in 1:num.vision){
  
  if(create_graph){
    world.graph <- create_world_graph(visualrange)
    world.trajectories[[n]] <- world.graph
  } else{
    world.graph <- world.trajectories[[n]]
  }
  
  szs <- seq(1, 4, length=max(E(world.graph)$width, na.rm=TRUE))
  clrs <- palf(max(E(world.graph)$width, na.rm=TRUE))
  widths = E(world.graph)$width
  for(i in 1:length(E(world.graph)$width)){
    width <- widths[i]
    if(width>0.2){
      E(world.graph)[i]$width <- szs[width]
      E(world.graph)[i]$color <- clrs[width]
    }
  }
  plot.title <- sprintf("Visual Range %d",visualrange)
  plot(world.graph, vertex.shape="square", layout=layout_on_grid(world.graph, width = XSize, height = YSize))
  title(plot.title, cex.main=0.445)
  
  n <- n + 1
}

if(create_graph){
  save(world.trajectories, file="saved/figExt08_trajectories.rdata")
}

dev.off()