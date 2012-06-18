import pylab,matplotlib,numpy
#smoothing functions from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/
def smoothList(list,strippedXs=False,degree=10):  
     if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  
     smoothed=[0]*(len(list)-degree+1)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(list[i:i+degree])/float(degree)  
     return smoothed  

def smoothListTriangle(list,strippedXs=False,degree=5):  
     weight=[]  
     window=degree*2-1  
     smoothed=[0.0]*(len(list)-window)  
     for x in range(1,2*degree):weight.append(degree-abs(degree-x))  
     w=numpy.array(weight)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(numpy.array(list[i:i+window])*w)/float(sum(w))  
     return smoothed  

def smoothListGaussian(list,strippedXs=False,degree=5):  
     window=degree*2-1  
     weight=numpy.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(numpy.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=numpy.array(weightGauss)*weight  
     smoothed=[0.0]*(len(list)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(numpy.array(list[i:i+window])*weight)/sum(weight)  
     return smoothed  


def band_shape(data):
    """Plots the shape of the entire band by collapsing (averaging) over all time.
     data is a 2D array. Axis0 is time, axis1 is a spectrum.
     bandwidth is the total bandwidth of the system (in Hz). n_chans is extracted from data size. used for labling."""
    fig = matplotlib.pyplot.figure()
    subplot = fig.add_subplot(1, 1, 1)
    n_pts=data.shape[0]
    n_chans=data.shape[1]
    av_spectrum=numpy.sum(data,axis=0)/float(n_pts)
    median_lev = numpy.median(av_spectrum)
    spectrum_db=10*(numpy.log10(av_spectrum)-numpy.log10(median_lev))
    subplot.plot(smoothListGaussian(spectrum_db,degree=5))
    subplot.set_ylabel('Bandpass (dB)')
    subplot.set_title('Bandpass response')
    subplot.set_xlabel('Frequency (channel number)')
    matplotlib.pyplot.show()
