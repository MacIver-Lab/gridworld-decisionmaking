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

num.simulations <- 20

create_world_graph <- function(visualrange){
  world.graph <- make_lattice(dimvector=c(XSize, YSize))
  
  V(world.graph)$size <- 0
  V(world.graph)$label <- NA
  occlusion.nodes <- which(degree(world.graph)==0)
  V(world.graph)$color <- "#ffffff00"
  V(world.graph)[occlusion.nodes]$color <- "#000000"
  V(world.graph)[occlusion.nodes]$size <- 15
  V(world.graph)$frame.color <- "#ffffff00"
  E(world.graph)$width <- 0.2
  E(world.graph)$color <- "#ffffff00"
  
  agentTrajectory = matrix(0, nrow=XSize*YSize, ncol=1)
  for (simulation in 0:(num.simulations-1)){
    episode.base.path <- sprintf("../Simulation 1 Pseudo-Aquatic/Data/Simulation_%d/Vision_%d/Depth_5000", (simulation), (visualrange))
    episode.paths<- list.files(path=episode.base.path, recursive=T, pattern="Episode_", full.names=T) # Extract all episode files
    
    for (episode.path in episode.paths){
      episode <- read.csv(episode.path, header=TRUE)
      terminalReward <- episode$Reward[nrow(episode)]
      if(terminalReward < 0){
        next
      }
      episode.new <- episode[!duplicated(episode[,c('Agent.X', 'Agent.Y')]),]
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
  }
  
  return(world.graph)
}

palf <- colorRampPalette(c(rgb(255/255, 255/255, 255/255),  rgb(192/255, 212/255, 229/255), rgb(140/255, 151/255, 196/255),
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
plot.name <- "Plots/fig02_visualrange/fig02c.pdf"
pdf(plot.name, height=(4.5+1)/5, width=4.5+1, useDingbats = FALSE)
par(mfrow= c(1, 2), oma=c(0, 0, 0, 0), mar=c(0, 0, 1, 0))

visual.range <- c(1, 5)

create_graph = FALSE
if(file.exists("saved/fig02_trajectories.rdata")){
  load(file="saved/fig02_trajectories.rdata")
} else{
  create_graph = TRUE
  world.trajectories = vector("list", length(2))
}

for (v in 1:length(visual.range)){
  
  if(create_graph){
    world.graph <- create_world_graph(visual.range[v])
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
  plot.title <- sprintf("Visual Range %d",visual.range[v])
  plot(world.graph, vertex.shape="square", layout=layout_on_grid(world.graph, width = XSize, height = YSize))
  title(plot.title, cex.main=0.445)
  
  n <- n + 1
  
}

if(create_graph){
  save(world.trajectories, file="saved/fig02_trajectories.rdata")
}
dev.off()