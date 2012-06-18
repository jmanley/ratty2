#!/usr/bin/env python

'''
Plots a histogram and time-domain sample of the ADC values from a specified antenna and pol.

'''
'''
Revisions:
2011-03-xx  JRM Misc modifications, feature additions etc
2011-02-24  JRM Port to RFI system
2010-12-11: JRM Add printout of number of bits toggling in ADC.
                Add warning for non-8bit ADCs.
2010-08-05: JRM Mods to support variable snap block length.
1.1 PVP Initial.\n

'''

#TODO: Add duty-cycle measurement support.
#TODO: Add trigger count support.

import matplotlib
matplotlib.use('TkAgg')
import rfi_sys, time, corr, numpy, struct, sys, logging, pylab, h5py

# what format are the snap names and how many are there per antenna
snapName = 'snap_adc'
# what is the bram name inside the snap block
bramName = 'bram'

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',
    try:
        f.flush()
        f.close()
        r.lh.printMessages()
        r.fpga.stop()
    except:
        pass
    if verbose:
        raise
    exit()

def exit_clean():
    try:
        print "Closing file."
        f.flush()
        f.close()
        r.fpga.stop()
    except:
        pass
    exit()

# callback function to draw the data for all the required polarisations
def drawDataCallback(n_samples,indep,trig_level):
    unpackedData, timestamp,status = getUnpackedData(trig_level=trig_level)
    filewrite(unpackedData,timestamp,status)
    if indep:
        mean_lev =numpy.mean(unpackedData)
        meana_lev =numpy.mean(unpackedData[0::2])
        meanb_lev =numpy.mean(unpackedData[1::2])
        std_deva = numpy.std(unpackedData[0::2])
        std_devb = numpy.std(unpackedData[1::2])
        print '%s: Mean ADC-A: %4.2f, mean ADC-B: %4.2f, overall: %4.2f. std-dev-ADC-A: %4.2f, std-dev-ADC-B: %4.2f.'%(time.ctime(timestamp),meana_lev,meanb_lev,mean_lev,std_deva,std_devb)

    calData=unpackedData*trig_scale_factor 
    abs_levs=numpy.abs(calData)
    max_lev =numpy.max(abs_levs)
    trigs = numpy.ma.flatnotmasked_edges(numpy.ma.masked_less_equal(abs_levs,(trig_level-4)*trig_scale_factor))
    #print trigs
    if (trigs == None or trigs[0] ==0) and trig_level>0 and (max_lev/trig_scale_factor)<trig_level: 
        max_pos = numpy.argmax(calData)
        print('ERROR: we asked for a trigger level of %4.2f mV and the hardware reported success, but the maximum level in the returned data was only %4.2fmV.'%(trig_level*trig_scale_factor,max_lev))

    if trigs==None:
        max_pos = numpy.argmax(calData)
    else:
        max_pos = trigs[0]
    
    subplots[0].cla()
    if indep:
        max_pos=(max_pos/2)*2
        t_start =max(0,max_pos-n_samples/2)
        t_stop  =min(len(calData),max_pos+n_samples/2)
        p_data=calData[t_start:t_stop]
        x_range=numpy.arange(t_start-max_pos,t_stop-max_pos)

        subplots[0].plot(x_range[0::2],p_data[0::2])
        subplots[0].plot(x_range[1::2],p_data[1::2])
        subplots[0].set_xlim(-n_samples/4*1.e9/sample_clk,n_samples/4)
    else:
        t_start =max(0,max_pos-n_samples/2)
        t_stop  =min(len(calData),max_pos+n_samples/2)
        p_data  =calData[t_start:t_stop]
        x_range =numpy.arange(t_start-max_pos,t_stop-max_pos)*1.e9/sample_clk
        #print max_pos,t_start,t_stop,len(x_range)

        subplots[0].plot(x_range,p_data)
        subplots[0].set_xlim(-n_samples/2*1.e9/sample_clk,n_samples/2*1.e9/sample_clk)

    if status['adc_overrange'] or status['adc_bad']:
        subplots[0].set_title('Time-domain [%i] (max %4.2fmV)'%(cnt,max_lev), bbox=dict(facecolor='red', alpha=0.5))
    else:
        subplots[0].set_title('Time-domain [%i] (max %4.2fmV; ADC %i)'%(cnt,max_lev,numpy.max(numpy.abs(unpackedData))))
    subplots[0].set_ylim(-max_lev-1,max_lev+1)
    subplots[0].set_ylabel('mV')
    subplots[0].set_xlabel('Time (nanoseconds).')

    subplots[1].cla()
    t_start =0
    #t_start =max(0,max_pos-(n_chans*2)-1)
    freqs,emptySpectrum=rfi_sys.cal.get_calibrated_spectrum_from_raw_snapshot(adcdata=unpackedData[t_start:max_pos-1],atten=rf_gain,bandwidth=bandwidth,ant_factor=af,bandshape=bp,n_chans=n_chans)
    freqs,calSpectrum=rfi_sys.cal.get_calibrated_spectrum_from_raw_snapshot(adcdata=unpackedData,atten=rf_gain,bandwidth=bandwidth,ant_factor=af,bandshape=bp,n_chans=n_chans)
    #print 'got a spectrum:',calSpectrum
    #print 'plotting from %i to %i'%(t_start,max_pos-1)
    pylab.hold(True)
    subplots[1].plot(freqs/1e6,calSpectrum,label='Quiescent')
    pylab.hold(True)
    subplots[1].plot(freqs/1e6,emptySpectrum,label='Signal on')
    subplots[1].legend()
    subplots[1].set_title('Spectrum of capture (%i samples)'%(len(unpackedData)))
    subplots[1].set_ylabel('Level (%s)'%units)
    subplots[1].set_xlabel('Frequency (MHz)')
    
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, drawDataCallback, n_samples,indep,trig_level)

# the function that gets data given a required polarisation
def getUnpackedData(trig_level=-1):
    # get the data
    if file==None:
        unpackedBytes = r.get_adc_snapshot(trig_level=trig_level) 
        stat=r.status_get()
        stat['adc_level']=r.adc_amplitudes_get()['adc_dbm']
        timestamp=time.time()
    else:
        global cnt
        print 'Press enter to grab plot number %i...'%cnt,
        raw_input()
        if cnt+1>=f['raw_dumps'].shape[0]: exit_clean()
        unpackedBytes = f['raw_dumps'][cnt] 
        stat={'adc_overrange':f['adc_overrange'][cnt],
                'adc_bad':f['adc_shutdown'][cnt],
                'fft_overrange':f['fft_overrange'][cnt],
                'adc_level':f['adc_level'][cnt]}
        timestamp=f['timestamp'][cnt]
        cnt+=1

    print '%s: ADC input level: %5.2f dBm.'%(time.ctime(timestamp),stat['adc_level']),
    if stat['adc_bad']: print 'ADC selfprotect due to overrange!',
    if stat['adc_overrange']: print 'ADC is clipping!',
    if stat['fft_overrange']: print 'FFT is overflowing!',
    print ''
    return unpackedBytes, timestamp, stat

def filewrite(adc_data,timestamp,status):
    if not file:
        global cnt
        cnt=f['raw_dumps'].shape[0]-1
        print '  Storing entry %i...'%cnt,
        sys.stdout.flush()
        f['raw_dumps'][cnt] = adc_data 
        f['timestamp'][cnt] = timestamp
        f['adc_overrange'][cnt] = status['adc_overrange']
        f['fft_overrange'][cnt] = status['fft_overrange']
        f['adc_shutdown'][cnt] = status['adc_bad']
        f['adc_level'][cnt] = status['adc_level']
        for name in ['raw_dumps','timestamp','adc_overrange','fft_overrange','adc_shutdown','adc_level']:
            f[name].resize(cnt+2, axis=0)
        print 'done'
#    else:
#        print '    Not writing any files.'


if __name__ == '__main__':
    from optparse import OptionParser
    p = OptionParser()
    p.set_usage('%prog [options] LOG_MESSAGE')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true',default=False, 
        help = 'Enable debug mode.')
    p.add_option('-i', '--plot_indep', dest = 'plot_indep', action = 'store_true', 
        help = 'Plot interleaved ADC independantly.')
    p.add_option('-f', '--file', dest = 'file', type='string', 
        help = 'Open an existing file for analysis.')
    p.add_option('-t', '--capture_len', dest = 'capture_len', type='int', default = 100, 
        help = 'Plot this many nano-seconds around the trigger point. Default:100')
    p.add_option('-e', '--entry', dest = 'entry', type='int', default = 0, 
        help = 'Jump directly to this entry. Default:0')
    p.add_option('-a', '--ant', dest = 'ant', type='string', default = 'none', 
        help = 'Choose an antenna calibration file. Note that this will auto-select y-axis units to be dBuV/m. Default:none')
    #p.add_option('-u', '--units', dest = 'units', type='string', default = 'dBm', 
    #    help = 'Choose the units for y-axis in freq plots. Options include dBuV,dBm. Default:dBm')
    p.add_option('-c', '--n_chans', dest = 'n_chans', type='int', default = 256, 
        help = 'Number of frequency channels to resolve in software FFT. Default:256')
    p.add_option('-l', '--trig_level', dest = 'trig_level', type='float', default = 0., 
        help = 'Ask the hardware to wait for a signal with at least this amplitude in mV before capturing. Valid range: 0-127. Default: negative (disabled, just plot current input).')
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])
    verbose=opts.verbose
    n_chans=opts.n_chans
    ant=opts.ant
    usrlog='_'.join(args)
    if usrlog=='': usrlog='No userlog specified. Starting file at %i.'%(int(time.time()))
    if opts.file: file=opts.file
    else: file=None

#    if args==[]:
#        print 'Please specify a mode!\nExiting.'
#        exit()

try:
    if file ==None:
        print 'Connecting to ROACH...',
        # make the correlator object
        r = rfi_sys.cam.spec()
        #r = rfi_sys.rfi_sys(mode=args[0])
        if verbose:
            r.logger.setLevel(logging.DEBUG)
        else:
            r.logger.setLevel(logging.INFO)
        print 'done.'

        rf_gain     =r.rf_status_get()[1]
        trig_scale_factor=rfi_sys.cal.get_adc_cnt_mv_scale_factor(rf_gain)
        n_samples   =int(opts.capture_len/1.e9*r.sample_clk)
        trig_level  =int(opts.trig_level/trig_scale_factor)
        bandwidth   =r.bandwidth
        sample_clk  =r.sample_clk

        filename=str(int(time.time())) + ".time.h5"
        print 'Starting file %s.'%filename
        f = h5py.File(filename, mode="w")
        baseline=r.get_adc_snapshot()
        f.create_dataset('raw_dumps',shape=[1,len(baseline)],dtype=numpy.int8,maxshape=[None,len(baseline)])
        f.create_dataset('timestamp',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f.create_dataset('adc_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('fft_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_shutdown',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_level',shape=[1],maxshape=[None],dtype=numpy.float)
        f['/'].attrs['bitstream']=r.bitstream
        f['/'].attrs['bandwidth']=bandwidth
        f['/'].attrs['adc_type']=r.adc_type
        f['/'].attrs['adc_scale_to_mv']=trig_scale_factor
        f['/'].attrs['rf_gain']=rf_gain
        f['/'].attrs['usrlog']=usrlog
        f['/'].attrs['sample_clk']=sample_clk
        f['/'].attrs['trig_level']=trig_level

    else:
        global cnt
        cnt=opts.entry
        print 'Opening file %s...'%file
        f=h5py.File(file,'r')
        usrlog      =f['/'].attrs['usrlog']
        bandwidth   =f['/'].attrs['bandwidth']
        rf_gain     =f['/'].attrs['rf_gain']
        trig_level  =f['/'].attrs['trig_level']
        print 'USRLOG: %s'%usrlog
        sample_clk=898000000*2
#        sample_clk=f['/'].attrs['sample_clk']
        n_samples=int(opts.capture_len*1e9/sample_clk)
        trig_scale_factor=rfi_sys.cal.get_adc_cnt_mv_scale_factor(rf_gain)

    freqs=numpy.arange(n_chans)*float(bandwidth)/n_chans #channel center freqs in Hz

    if opts.ant != 'none':
        af=rfi_sys.cal.af_from_gain(freqs,rfi_sys.cal.ant_gains(opts.ant,freqs)) #antenna factor
        if file==None:
            f['/'].attrs['antena_factor']=af
            f['/'].attrs['antena_calfile']=opts.ant
        units='dBuV/m'
        #rfi_sys.cal.plot_ant_gain(opts.ant,freqs)
        #rfi_sys.cal.plot_ant_factor(opts.ant,freqs)
        #pylab.show()
    else:
        af=None
        units='dBm'

    bp=rfi_sys.cal.bandshape(freqs)
    if file==None:
        f['bandshape']=bp


    print 'Triggering at a level of %4.2fmV (ADC level of %i).'%(trig_level*trig_scale_factor,trig_level)
    print 'Plotting %i samples.'%n_samples

    # set up the figure with a subplot for each polarisation to be plotted
    fig = matplotlib.pyplot.figure()

    # create the subplots
    subplots = []
    for p in range(2):
        subPlot = fig.add_subplot(2, 1, p + 1)
        subplots.append(subPlot)

    # start the process
    print 'Starting plots...'
    fig.subplots_adjust(hspace=0.8)
    fig.canvas.manager.window.after(100, drawDataCallback, n_samples,opts.plot_indep,trig_level)
    matplotlib.pyplot.show()

except KeyboardInterrupt:
    exit_clean()
except:
#    exit_fail()
    raise

print 'Done with all.'
exit_clean()

# end

