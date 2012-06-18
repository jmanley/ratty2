import pylab,matplotlib,numpy
def chan_shape(data,cent_chan,freq_step_per_integration,bandwidth):
    """Plots the shape of individual channels.
     chans is a list of the channels you'd like to plot.
     data is a 2D array. Axis0 is time, axis1 is a spectrum.
     freq_step_per_integration is the step size of the signal gen in Hz.
     bandwidth is the total bandwidth of the system (in Hz). n_chans is extracted from data size. used for labling."""
    xax=numpy.arange(len(data))
    fig = matplotlib.pyplot.figure()
    subplot = fig.add_subplot(1, 1, 1)
    n_pts=data.shape[0]
    n_chans=data.shape[1]
    chans=[cent_chan-1,cent_chan,cent_chan+1]
    median_lev = numpy.median(data[:,cent_chan])
    peak_db=numpy.max(10*(numpy.log10(data[:,cent_chan])-numpy.log10(median_lev)))
    loc_peak=numpy.argmax(data[:,cent_chan])
    start_freq=-(loc_peak)*freq_step_per_integration
    stop_freq=(n_pts-loc_peak)*freq_step_per_integration
    xaxis_lab=numpy.arange(start_freq,stop_freq,freq_step_per_integration)[0:n_pts]
    print 'Peak found at time %i with level %i'%(loc_peak,peak_db)
    print ' start: %8.3f, stop %8.3f. len %i/%i'%(start_freq,stop_freq,len(xaxis_lab),n_pts)

    for chan in chans:
        this_chan_db=10*(numpy.log10(data[:,chan])-numpy.log10(median_lev))
        subplot.plot(xaxis_lab,this_chan_db)

    subplot.set_ylabel('Uncalibrated Leveli (dB)')
    subplot.set_title('Channel response')
    subplot.set_xlabel('Frequency (MHz)')
    matplotlib.pyplot.show()
