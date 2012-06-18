#!/usr/bin/env python

'''
Plots the spectrum from an RFI monitoring spectrometer.\n

'''
#Revisions:\n
#2011-03-17  JRM Removed lin plots, added cal
#2011-03-xx  JRM Added various features - logging to file, lin/log plots, status reporting etc.
#2011-02-24  JRM Port to RFI system
#2010-12-11: JRM Add printout of number of bits toggling in ADC.
#                Add warning for non-8bit ADCs.
#2010-08-05: JRM Mods to support variable snap block length.
#1.1 PVP Initial.

#TODO: FIX r.freqs now in Hz (not MHz)
#TODO: read back from files rather than live system

import matplotlib
matplotlib.use('TkAgg')
import pylab,h5py,rfi_sys, time, corr, numpy, struct, sys, logging

# what format are the snap names and how many are there per antenna
bram_out_prefix = 'store'
# what is the bram name inside the snap block
bramName = 'bram'
verbose=True
n_chans_ignore_top=900
n_chans_ignore_bot=100
freq_range='0,-1'.split(',')
max_level=numpy.NINF
min_level=numpy.Inf
cal_mode='full'

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
        r.fpga.stop()
        print "Closing file."
        f.flush()
        f.close()
    except:
        pass
    exit()

def filewrite(spectrum,timestamp,acc_cnt,status):
    if not file:
        cnt=f['spectra'].shape[0]-1
        print '  Storing entry %i...'%cnt,
        sys.stdout.flush()
        f['spectra'][cnt]   = spectrum
        f['acc_cnt'][cnt]   = acc_cnt
        f['timestamp'][cnt] = timestamp
        f['adc_overrange'][cnt] = status['adc_overrange']
        f['fft_overrange'][cnt] = status['fft_overrange']
        f['adc_shutdown'][cnt] = status['adc_bad']
        f['adc_level'][cnt] = status['adc_level']
        f['input_level'][cnt] = status['input_level']
        f['adc_temp'][cnt] = status['adc_temp']
        f['ambient_temp'][cnt] = status['ambient_temp']
        for name in ['spectra','acc_cnt','timestamp','adc_overrange','fft_overrange','adc_shutdown','adc_level','input_level','adc_temp','ambient_temp']:
            f[name].resize(cnt+2, axis=0)
        print 'done'


def getUnpackedData(last_cnt):
    if file==None:
        #wait for a new integration:
        while r.fpga.read_uint('acc_cnt') == last_cnt: 
    #        print '.',
    #        sys.stdout.flush()
            time.sleep(0.1)
    #    print ''
        # get the data
        spectrum=numpy.zeros(r.n_chans)
        for i in range(r.n_par_streams):
            spectrum[i::r.n_par_streams] = numpy.fromstring(r.fpga.read('%s%i'%(bram_out_prefix,i),r.n_chans/r.n_par_streams*8),dtype=numpy.uint64).byteswap()
        stat=r.status_get()
        ampls=r.adc_amplitudes_get()
        stat['adc_level']=ampls['adc_dbm']
        stat['input_level']=ampls['input_dbm']
        stat['adc_temp']=r.adc_temp_get()
        stat['ambient_temp']=r.ambient_temp_get()
        last_cnt=r.fpga.read_uint('acc_cnt')
        timestamp=time.time()
    else:
        print 'Press enter to grab plot number %i...'%last_cnt,
        raw_input()
        if last_cnt+1>=f['spectra'].shape[0]: exit_clean()
        spectrum = f['spectra'][last_cnt] 
        stat={'adc_overrange':f['adc_overrange'][last_cnt],
                'adc_bad':f['adc_shutdown'][last_cnt],
                'fft_overrange':f['fft_overrange'][last_cnt],
                'input_level':f['input_level'][last_cnt],
                'adc_level':f['adc_level'][last_cnt]}
        timestamp=f['timestamp'][last_cnt]
        last_cnt+=1
        print 'got all data'

    print '[%i] %s: input level: %5.2f dBm (ADC %5.2f dBm).'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),
    if stat['adc_bad']: print 'ADC selfprotect due to overrange!',
    elif stat['adc_overrange']: print 'ADC is clipping!',
    elif stat['fft_overrange']: print 'FFT is overflowing!',
    else: print 'all ok.',
    print ''
    return spectrum,timestamp,last_cnt,stat

def find_n_max(data,n_max,ignore_adjacents=False):
    max_levs=numpy.ones(n_max)*-numpy.Inf
    max_locs=numpy.ones(n_max)*-numpy.Inf
    this_max_lev=-numpy.Inf
    for n,d in enumerate(data[:-1]):
        if d>this_max_lev and (not ignore_adjacents or (data[n+1]<d and data[n-1]<d)):
            #print 'Peak found at %d: %i. max now at %i'%(n,d,this_max_lev)
            loc=numpy.argmin(max_levs)
            max_levs[loc]=d
            max_locs[loc]=n
            if numpy.min(max_levs)>this_max_lev: this_max_lev=numpy.min(max_levs)
    return max_levs,max_locs     

# callback function to draw the data for all the required polarisations
def drawDataCallback(last_cnt):

    unpackedData, timestamp, last_cnt,stat = getUnpackedData(last_cnt)
    filewrite(unpackedData,timestamp,last_cnt,stat)
    calData=rfi_sys.cal.get_calibrated_spectrum(freqs,unpackedData,n_accs,fft_scale,rf_gain,bandshape=bp,ant_factor=af) #returns spectrum in dBm

    median_lev_db=numpy.median(calData[20:-20])
    calData[0:n_chans_ignore_bot]=calData[n_chans_ignore_bot]
    calData[-n_chans_ignore_top:]=calData[-n_chans_ignore_top]
    maxs,locs=find_n_max(calData[n_chans_ignore_bot:-n_chans_ignore_top],n_top,ignore_adjacents=True)

    subplot1.cla()
    if stat['fft_overrange'] or stat['adc_bad'] or stat['adc_overrange']:
        subplot1.set_title('Spectrum %i as at %s (input power: %5.1fdBm; ADC level %5.1fdBm)'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),bbox=dict(facecolor='red', alpha=0.5))
    else:
        subplot1.set_title('Spectrum %i as at %s (input power: %5.1fdBm; ADC level %5.1fdBm)'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']))
    subplot1.set_xlabel('Frequency (MHz)')
    subplot1.set_ylabel('Level (%s)'%units)

    if plot_baseline or plot_diff:
        subplot1.hold(True)
        subplot1.plot(freqs[chan_low:chan_high]/1.e6,baseline[chan_low:chan_high],'r',linewidth=5,alpha=0.5)

    subplot1.plot(freqs[chan_low:chan_high]/1.e6,calData[chan_low:chan_high],'b')
#    print 'Freqs:',freqs
#    print 'locs:',locs
#    print 'data:',calData

    ##collapse data for plotting:
    #collapse_factor=len(unpackedData)/plot_chans
    #collapseddata=unpackedData.reshape(plot_chans,collapse_factor).sum(1)/collapse_factor
    #if plot_type == 'lin':
    #    subplot.plot(r.freqs[::collapse_factor],collapseddata)
    #    #Plot a horizontal line representing the average noise floor:
    #    subplot.hlines((median_lev),0,r.freqs[-1])
    #elif plot_type == 'log':
    #    #subplot.semilogy(r.freqs[::collapse_factor],collapseddata)
    #    subplot.plot(r.freqs[::collapse_factor],10*numpy.log10(collapseddata))
    #    median_lev_db=10*numpy.log10(median_lev)
    #    #Plot a horizontal line representing the average noise floor:
    #    subplot.hlines(median_lev_db,0,r.freqs[-1])
    #    subplot.annotate('%3.1fdB'%(median_lev_db),(r.freqs[-1],median_lev_db))
    

    if plot_diff:
        subplot2.cla()
        subplot2.plot(freqs[chan_low:chan_high]/1.e6,calData[chan_low:chan_high]-baseline[chan_low:chan_high])
        subplot2.set_ylabel('Difference (dB)')

    #Plot a horizontal line representing the average noise floor:
    subplot1.hlines(median_lev_db,freqs[chan_low]/1e6,freqs[chan_high]/1.e6)
    subplot1.annotate('%3.1f%s'%(median_lev_db,units),(freqs[chan_high]/1.e6,median_lev_db))

   
    #annotate:
    for i in range(n_top):
        freq=freqs[locs[i]+n_chans_ignore_bot]/1.e6
        #lev=10*(numpy.log10(maxs[i])) #-numpy.log10(median_lev))
        lev=maxs[i]
        print '  Local max at chan %5i (%6.2fMHz): %6.2f%s'%(locs[i]+n_chans_ignore_bot,freq,lev,units)
        subplot1.annotate('%iMHz:%3.1f%s'%(numpy.round(freq),lev,units),(freq,lev))

        #if plot_type == 'lin':
        #    subplot.annotate('%iMHz:%3.1fdB'%(freq,lev),(freq,collapseddata[locs[i]/collapse_factor]))
        #elif plot_type == 'log':
        #    subplot.annotate('%iMHz:%3.1fdB'%(freq,lev),(freq,10*numpy.log10(collapseddata[locs[i]/collapse_factor])))
    
    global min_level
    global max_level
    #local_min=min(calData)
    min_level=min(min(calData),min_level)
    max_level=max(max(calData),max_level)
    #subplot1.set_ylim(min_level-10,max_level+10)
    
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, drawDataCallback, last_cnt)

if __name__ == '__main__':
    from optparse import OptionParser
    p = OptionParser()
    p.set_usage('%prog [options] MODE')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Enable debug logging mode.')
    p.add_option('-b', '--baseline', dest = 'baseline', action = 'store_true', default=False,
        help = 'Keep the first trace displayed as a baseline.')
    p.add_option('-d', '--diff', dest = 'diff', action = 'store_true', default=False,
        help = 'Also plot the difference between the first trace and subsequent spectra.')
    #p.add_option('-c', '--cal', dest = 'cal',type='string',default='full',
    #     help = 'Choose the calibration mode (none, full, scaled etc). Default: full.')
    p.add_option('-r', '--freq_range', dest = 'freq_range',type='string',default='0,-1',
         help = 'Select a frequency range in MHz to plot. Negative frequencies are supported. Default: 0,-1.')
    p.add_option('-s', '--n_top', dest='n_top', type='int',default=5,
        help='Find the top N spiky RFI candidates. Default: 5')
    p.add_option('-f', '--file', dest = 'file', type='string', 
        help = 'Open an existing file for analysis.')
    p.add_option('-a', '--ant', dest = 'ant', type='string', default = 'none', 
        help = 'Choose an antenna calibration file. Note that this will auto-select y-axis units to be dBuV/m. Default:none')

#    p.add_option('-n', '--n_chans', dest='n_chans', type='int',default=512,
#        help='Plot this number of channels. Default: 512')
#    p.add_option('-l', '--plot_lin', dest='plot_lin', action='store_true',
#        help='Plot on linear axes. Default: semilogy.')
    p.add_option('-p', '--no_plot', dest='plot', action='store_false',default=True,
        help="Don't plot anything.")
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])

    usrlog='_'.join(args)
    if usrlog=='': usrlog=str(int(time.time()))
    if opts.file: file=opts.file
    else: file=None
    #plot_chans=opts.n_chans
    freq_range=opts.freq_range.split(',')
    ant=opts.ant
    n_top=opts.n_top
    verbose=opts.verbose
    plot_baseline=opts.baseline
    plot_diff = opts.diff
    #cal_mode=opts.cal
    

try:
    if file==None:
        print 'Connecting to ROACH...',
        # make the correlator object
        r = rfi_sys.cam.spec()
        #r = rfi_sys.rfi_sys(mode=args[0])
        if verbose:
            r.logger.setLevel(logging.DEBUG)
        else:
            r.logger.setLevel(logging.INFO)
        print 'done.'

        if r.spectrum_bits != 64: 
            print 'ERR: Sorry, this is only for 64 bit systems.'
            exit()

        acc_time,n_accs = r.acc_time_get()
        freqs=r.freqs
        fft_shift=r.fft_shift_get()
        fft_scale=r.fft_scale
        rf_gain=r.rf_status_get()[1]
        bandwidth=r.bandwidth
        n_chans=r.n_chans

        print 'Scaling back by %i accumulations.'%n_accs

        filename=usrlog + ".spec.h5"
        print 'Starting file %s.'%filename
        f = h5py.File(filename, mode="w")
#        f.create_dataset('spectra',shape=[1,r.n_chans],dtype=numpy.int64,maxshape=[None,r.n_chans])
        f.create_dataset('spectra',shape=[1,r.n_chans],maxshape=[None,r.n_chans])
        f.create_dataset('acc_cnt',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f.create_dataset('timestamp',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f.create_dataset('adc_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('fft_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_shutdown',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_level',shape=[1],maxshape=[None],dtype=numpy.float)
        f.create_dataset('adc_temp',shape=[1],maxshape=[None],dtype=numpy.float)
        f.create_dataset('ambient_temp',shape=[1],maxshape=[None],dtype=numpy.float)
        f.create_dataset('input_level',shape=[1],maxshape=[None],dtype=numpy.float)

        f['/'].attrs['n_chans']=n_chans
        f['/'].attrs['n_accs']=n_accs
        f['/'].attrs['bitstream']=r.bitstream
        f['/'].attrs['bandwidth']=bandwidth
        f['/'].attrs['adc_type']=r.adc_type
        f['/'].attrs['spectrum_bits']=r.spectrum_bits
        f['/'].attrs['fft_shift']=fft_shift
        f['/'].attrs['rf_gain']=rf_gain
        f['/'].attrs['usrlog']=usrlog

        last_cnt=r.fpga.read_uint('acc_cnt')
    else:
        print 'Opening file %s...'%file
        f=h5py.File(file,'r')
        usrlog      =f['/'].attrs['usrlog']
        last_cnt=0
        n_accs      =f['/'].attrs['n_accs']
        n_chans     =f['/'].attrs['n_chans']
        usrlog      =f['/'].attrs['usrlog']
        bandwidth   =f['/'].attrs['bandwidth']
        rf_gain     =f['/'].attrs['rf_gain']
        fft_shift   =f['/'].attrs['fft_shift']
        freqs       =numpy.arange(n_chans)*float(bandwidth)/n_chans #channel center freqs in Hz
        fft_scale   =2**(cal.bitcnt(fft_shift))

    if opts.ant != 'none':
        af=rfi_sys.cal.af_from_gain(freqs,rfi_sys.cal.ant_gains(opts.ant,freqs)) #antenna factor
        if file==None:
            f['antena_factor']=af
            f['/'].attrs['antena_calfile']=opts.ant
        units='dBuV/m'
        #rfi_sys.cal.plot_ant_gain(opts.ant,freqs)
        #rfi_sys.cal.plot_ant_factor(opts.ant,freqs)
        #pylab.show()
    else:
        af=None
        units='dBm'

    bp=rfi_sys.cal.bandshape(freqs)
    #rfi_sys.cal.plot_bandshape(freqs)
    if file==None:
        f['bandshape']=bp


    if opts.plot or file != None:
        chan_low =rfi_sys.cal.freq_to_chan(int(freq_range[0])*1e6,bandwidth,n_chans)
        chan_high=rfi_sys.cal.freq_to_chan(int(freq_range[1])*1e6,bandwidth,n_chans)
        print 'Plotting channels %i (%5.1fMHz) to %i (%5.1fMHz).'%(chan_low,freqs[chan_low]/1.e6,chan_high,freqs[chan_high]/1.e6)
        # set up the figure with a subplot for each polarisation to be plotted
        fig = matplotlib.pyplot.figure()
        if opts.diff or opts.baseline:
            print 'Fetching baseline...',
            sys.stdout.flush()
            unpackedData, timestamp, last_cnt,stat = getUnpackedData(last_cnt)
            filewrite(unpackedData,timestamp,last_cnt,stat)
            baseline=rfi_sys.cal.get_calibrated_spectrum(freqs,unpackedData,n_accs,fft_scale,rf_gain,bandshape=bp,ant_factor=af) #returns spectrum in dBm
            baseline[0:n_chans_ignore_bot]=baseline[n_chans_ignore_bot]
            baseline[-n_chans_ignore_top:]=baseline[-n_chans_ignore_top]
            print 'done'
        if opts.diff: 
            subplot1 = fig.add_subplot(2, 1, 1)
            subplot2 = fig.add_subplot(2, 1, 2)
        else: subplot1 = fig.add_subplot(1, 1, 1)
        fig.canvas.manager.window.after(100, drawDataCallback,last_cnt)
        matplotlib.pyplot.show()
        print 'Plot started.'
    else:
        while(1):
            unpackedData, timestamp, last_cnt,stat = getUnpackedData(last_cnt)
            filewrite(unpackedData,timestamp,last_cnt,stat)

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

print 'Done with all.'
exit_clean()
