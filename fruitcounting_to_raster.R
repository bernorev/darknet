library(raster)
library(tidyverse)
library(rgdal)
library(mapview)
library(spatialEco)
library(sp)
library(gstat) # Use gstat's idw routine
# pts <- read.csv("counting_output.csv")
# coordinates(pts) <- ~longitude+latitude 
# proj4string(pts) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84")  
# pts <- spTransform(pts,CRS("+proj=utm +datum=WGS84 +zone=34S +no_defs +ellps=WGS84 +towgs84=0,0,0"))
# 

  block_name <- "K0o1C" 
  block_files <- list.files(pattern = paste0(block_name,".*\\counting.csv$"),full.names = TRUE)
  
  # block_name <- "bardsley_2" 
  # block_files <- list.files(pattern =".*\\counting.csv$",full.names = TRUE)[2]
  # 
  pts <- block_files %>% 
    lapply(read_csv) %>% 
    bind_rows
  
  pts <- pts %>% group_by(Latitude,Longitude)%>% summarise_all(sum)
  
  coordinates(pts) <- ~Longitude+Latitude
  proj4string(pts) = CRS("+init=epsg:4326")
  mapview::mapview(pts)
  
  ###
  writeOGR(pts, "./",paste0(block_name,"_counts") , driver="ESRI Shapefile",overwrite = TRUE)
  ###
  
  pts <- readOGR(paste0(block_name,"_counts.shp"))
  names(pts) <- c("elapsed_time","counter","section_count")
  
  lower_quantile <- quantile(pts$section_count, 0.1)
  
  pts <- pts[pts$section_count > lower_quantile,]
  mapview::mapview(pts)
  # block <- readOGR(paste0(block_name,".shp"))
  block <- readOGR(paste0(block_name,"_block.gpkg"))
  block_utm <- spTransform(block,CRS("+init=epsg:3857"))

  # pts$path <- pts$path %>% as.numeric()
  sum(pts$section_count)

  mapview::mapview(pts,zcol="section_count") 
  # pts <- pts %>% select(section_count)
  pts <- spTransform(pts,CRS("+init=epsg:3857"))
  
  block_grid <- makegrid(block_utm, cellsize = 5,pretty = FALSE)
  colnames(block_grid) <- c("x","y")
  block_grid <- SpatialPoints(block_grid, proj4string = CRS(proj4string(block_utm)))
  block_grid <- SpatialPixels(block_grid[block_utm,])
  
  idw_yield <- gstat::idw(formula = section_count ~ 1, locations = pts, 
                          newdata = block_grid, idp=2)  # apply idw model for the data

  plot(idw_yield)
  r       <- raster(idw_yield)
  r.m     <- mask(r, block_utm)
  
  # r.m <- focal(r.m, w=matrix(1,3,3), fun=mean)
  r.m <- aggregate(r.m,2,method='bilinear')
  r.m <- disaggregate(r.m,2,method='bilinear')
  
  r.m <- mask(r.m, block_utm)
  mapview(r.m)
  
  writeRaster(r.m,paste0(block_name,"_counts.tif"),"GTiff",overwrite=TRUE)   
  
  ######rasterize functions
#   rast <- raster(pts,res=15)
#   r <- rasterize(pts, rast, pts$section_count, fun = sum)
#   r <- disaggregate(r,2,method='bilinear')
#   # pts <- spTransform(pts,CRS("+proj=longlat +ellps=WGS84 +datum=WGS84"))
#   r  <- projectRaster(r ,crs=proj4string(pts))
#   mapview::mapview(r) 
# 
#   
#   writeRaster(r,paste0(block_name,"_counts.tif"),"GTiff",overwrite=TRUE)   
# 
#   
#   
#   
#   
#   
#   r <- raster("allan_counts.tif")
#   r <- raster::shift( r, dx=-10, dy=-4)
#   writeRaster(r,"allan_counts_shifted.tif")
# mapview(r)  
  
  
# 
# 
# r <- raster("T10_counts.tif")
# crs(r) <- CRS("+init=EPSG:32734")
# r <- projectRaster(r,crs='+init=EPSG:3857',method='bilinear')
# plot(r)
# mapview::mapview(r)
# writeRaster(r,paste0(block_name,"_counts2.tif"),"GTiff",overwrite=TRUE)   
# 
# 
# r[r < 10] <- NA
# harvest_map <- r
# harvest_map <- aggregate(harvest_map,2)
# harvest_map <- disaggregate(harvest_map,2,method='bilinear')
# writeRaster(r,"heatmap","GTiff",overwrite=TRUE)   
# 
# EVI <- raster("EVI.tiff")
# EVI <-projectRaster(EVI ,harvest_map)
# EVI <- mask(EVI,harvest_map)
# 
# mapview(harvest_map) + mapview(EVI)
# 
# rasterCorrelation(EVI,harvest_map)
# 
# all <- brick(EVI,harvest_map)
# 
# all <- as.data.frame(rasterToPoints(all))
# 
# plot(all$EVI,all$layer)
# cor(all)
