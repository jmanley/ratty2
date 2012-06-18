import pylab,matplotlib,numpy

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


def find_n_max(data,n_max,ignore_adjacents=False):
    max_levs=numpy.zeros(n_max)
    max_locs=numpy.zeros(n_max)
    max_lev=0
    for n,d in enumerate(data[:-1]):
        if d>max_lev and (not ignore_adjacents or (data[n+1]<d and data[n-1]<d)):
            #print 'Peak found at %d: %i. max now at %i'%(n,d,max_lev)
            loc=numpy.argmin(max_levs)
            max_levs[loc]=d
            max_locs[loc]=n
            if numpy.min(max_levs)>max_lev: max_lev=numpy.min(max_levs)
    return max_levs,max_locs


def sfdr_freq(data):
    """Plots the difference between the highest peak and the second-highest peak across the band."""
    
    fig = matplotlib.pyplot.figure()
    subplot = fig.add_subplot(1, 1, 1)
    n_pts=data.shape[0]
    n_chans=data.shape[1]

    sfdr_spectrum=numpy.zeros(n_chans)
    for t in range(n_pts):
        max_arg=numpy.argmax(data[t,:])
        max_levs,max_locs=find_n_max(data[t,:],2,ignore_adjacents=True)
        sfdr=numpy.abs(10*numpy.log10(max_levs[0])-10*numpy.log10(max_levs[1]))
        sfdr_spectrum[max_arg]=sfdr
        print 'found peak for chan %i with sfdr %6.2f dB.'%(max_arg,sfdr)
        
    subplot.plot(smoothListGaussian(sfdr_spectrum))
    subplot.set_xlim(n_chans)
    subplot.set_ylabel('SFDR (dB)')
    subplot.set_title('SFDR')
    subplot.set_xlabel('Frequency (channel number)')
    matplotlib.pyplot.show()
