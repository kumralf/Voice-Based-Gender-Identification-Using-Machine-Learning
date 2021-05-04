finalDataFrame = data.frame()

specan <- function(X, bp=c(0,22), wl=2048, threshold=5, parallel=1, pb=TRUE) {
  
  
  
  r <- tuneR::readWave(as.character(X[1, 1]), 
                       from = X[1, 3], to = X[1, 4],  units = "seconds")
  
  b<- bp #in case bp its higher than can be due to sampling rate
  if(b[2] > ceiling(r@samp.rate/2000) - 1) b[2] <- ceiling(r@samp.rate/2000) - 1  
  
  # frequency spectrum analysis  
  songspec <- seewave::spec(r, f = r@samp.rate, plot = FALSE)
  
  analysis <- seewave::specprop(songspec, f = r@samp.rate, 
                                flim = c(0, 280/1000), plot = FALSE)
  
  #save parameters
  
  meanfreq <- analysis$mean/1000
  sd <- analysis$sd/1000
  median <- analysis$median/1000
  Q25 <- analysis$Q25/1000
  Q75 <- analysis$Q75/1000
  IQR <- analysis$IQR/1000
  skew <- analysis$skewness
  kurt <- analysis$kurtosis
  sp.ent <- analysis$sh
  sfm <- analysis$sfm
  mode <- analysis$mode/1000
  centroid <- analysis$cent/1000

  
  sfm <- analysis$sfm
  
  # Fundamental frequency parameters
  ff <- seewave::fund(r, f = r@samp.rate, ovlp = 50, threshold = threshold, 
                      fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = wl)[, 2]
  meanfun<-mean(ff, na.rm = T)
  minfun<-min(ff, na.rm = T)
  maxfun<-max(ff, na.rm = T)
  
  #Dominant frecuency parameters
  y <- seewave::dfreq(r, f = r@samp.rate, wl = wl, ylim=c(0, 280/1000), ovlp = 0, plot = F, threshold = threshold, bandpass = b * 1000, fftw = TRUE)[, 2]
  meandom <- mean(y, na.rm = TRUE)
  mindom <- min(y, na.rm = TRUE)
  maxdom <- max(y, na.rm = TRUE)
  dfrange <- (maxdom - mindom)

  #modulation index calculation
  changes <- vector()
  for(j in which(!is.na(y))){
    change <- abs(y[j] - y[j + 1])
    changes <- append(changes, change)
  }
  if(mindom==maxdom) modindx<-0 else modindx <- mean(changes, na.rm = T)/dfrange
  
  
  DataFrame = data.frame(meanfreq = meanfreq, sd = sd, median = median, Q25 = Q25, Q75 = Q75, IQR = IQR,
                         skewness = skew, kurtosis = kurt, sp.ent = sp.ent, sfm = sfm,
                         mode = mode, cent = centroid, meanfun = meanfun,
                         minfun = minfun, maxfun = maxfun, meandom = meandom,
                         mindom = mindom, maxdom = maxdom, dfrange = dfrange,
                         modindx = modindx)
  
  finalDataFrame = rbind(finalDataFrame, DataFrame)
  
  
  return (finalDataFrame)
  
}

sampleDataframe = data.frame(sound.files='male40.wav', selec=0, start=0, end=10)
sampleDataframe['start'] = lapply(sampleDataframe['start'], as.numeric)
sampleDataframe['end'] = lapply(sampleDataframe['end'], as.numeric)
sampleDataframe['selec'] = lapply(sampleDataframe['selec'], as.character)

result = specan(sampleDataframe)






