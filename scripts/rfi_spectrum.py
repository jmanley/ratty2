#!/usr/bin/env python

'''
Plots the spectrum from an RFI monitoring spectrometer.\n

'''
#Revisions:\n
#2012-06-14  JRM Overhaul to more object-oriented code.
#                Diff plots now have maxhold y-axis and peak annotations.
#                Added option to only plot one capture (useful for interacting with a single spectrum)
#2011-03-17  JRM Removed lin plots, added cal
#2011-03-xx  JRM Added various features - logging to file, lin/log plots, status reporting etc.
#2011-02-24  JRM Port to RFI system
#2010-12-11: JRM Add printout of number of bits toggling in ADC.
#                Add warning for non-8bit ADCs.
#2010-08-05: JRM Mods to support variable snap block length.
#1.1 PVP Initial.


import matplotlib
matplotlib.use('TkAgg')
import pylab,h5py,ratty1, time, corr, numpy, struct, sys, logging, os
import iniparse

# what format are the snap names and how many are there per antenna
bram_out_prefix = 'store'
# what is the bram name inside the snap block
bramName = 'bram'
verbose=True
freq_range='0,-1'.split(',')
max_level=numpy.NINF
min_level=numpy.Inf
dmax_level=numpy.NINF
dmin_level=numpy.Inf
cal_mode='full'

def exit_fail():
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
    cnt=f['spectra'].shape[0]-1
    print '  Storing entry %i...'%cnt,
    sys.stdout.flush()
    f['spectra'][cnt]   = spectrum
    f['acc_cnt'][cnt]   = acc_cnt
    f['timestamp'][cnt] = timestamp
    for name in ['spectra','acc_cnt','timestamp']:
        f[name].resize(cnt+2, axis=0)
    for stat in status:
        try:
            f[stat][cnt]=status[stat]
            f[stat].resize(cnt+2, axis=0)
        except KeyError:
            f.create_dataset(stat,shape=[1],maxshape=[None])
            #f['adc_overrange'][cnt] = status['adc_overrange']
            #f['fft_overrange'][cnt] = status['fft_overrange']
            #f['adc_shutdown'][cnt] = status['adc_shutdown']
            #f['adc_level'][cnt] = status['adc_level']
            #f['input_level'][cnt] = status['input_level']
            #f['adc_temp'][cnt] = status['adc_temp']
            #f['ambient_temp'][cnt] = status['ambient_temp']

    print 'done'


def getUnpackedData(last_cnt):
    if play_filename==None:
        spectrum, timestamp, last_cnt, stat = r.getUnpackedData()
        filewrite(spectrum,timestamp,last_cnt,stat)
    else:
        print 'Press enter to grab plot number %i...'%last_cnt,
        raw_input()
        if last_cnt+1>=f['spectra'].shape[0]: exit_clean()
        spectrum = f['spectra'][last_cnt] 
        stat={'adc_overrange':f['adc_overrange'][last_cnt],
                'adc_shutdown':f['adc_shutdown'][last_cnt],
                'fft_overrange':f['fft_overrange'][last_cnt],
                'input_level':f['input_level'][last_cnt],
                'adc_level':f['adc_level'][last_cnt]}
        timestamp=f['timestamp'][last_cnt]
        last_cnt+=1
        print 'got all data'

    print '[%i] %s: input level: %5.2f dBm (ADC %5.2f dBm).'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),
    if stat['adc_shutdown']: print 'ADC selfprotect due to overrange!',
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
    inds = max_levs.argsort()[::-1]
    return max_levs[inds],max_locs[inds]     

# callback function to draw the data for all the required polarisations
def drawDataCallback(last_cnt):
    unpackedData, timestamp, last_cnt,stat = getUnpackedData(last_cnt)
    calData=co.get_calibrated_spectrum(unpackedData, rf_gain) #returns spectrum in dBm

#    calData[0:chanlow]=calData[n_chans_ignore_bot]
#    calData[chan_high:]=calData[-n_chans_ignore_top]

    subplot1.cla()
    if stat['fft_overrange'] or stat['adc_shutdown'] or stat['adc_overrange']:
        subplot1.set_title('Spectrum %i as at %s (input power: %5.1fdBm; ADC level %5.1fdBm)'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),bbox=dict(facecolor='red', alpha=0.5))
    else:
        subplot1.set_title('Spectrum %i as at %s (input power: %5.1fdBm; ADC level %5.1fdBm)'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']))
    subplot1.set_xlabel('Frequency (MHz)')
    subplot1.set_ylabel('Level (%s)'%units)

    if plot_baseline or plot_diff:
        subplot1.hold(True)
        subplot1.plot(freqs[chan_low:chan_high]/1.e6,baseline[chan_low:chan_high],'r',linewidth=5,alpha=0.5)

    subplot1.plot(freqs[chan_low:chan_high]/1.e6,calData[chan_low:chan_high],'b')

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
        dd=calData[chan_low:chan_high]-baseline[chan_low:chan_high]
        subplot2.cla()
        subplot2.plot(freqs[chan_low:chan_high]/1.e6,dd)
        subplot2.set_ylabel('Difference (dB)')
        maxs,locs=find_n_max(dd,n_top,ignore_adjacents=True)
        maxfreqs=[freqs[locs[i]+chan_low]/1.e6 for i in range(n_top)]
        for i in range(n_top):
            subplot2.annotate('%iMHz:%3.1fdB'%(numpy.round(maxfreqs[i]),maxs[i]),(maxfreqs[i],maxs[i]))
        global dmin_level
        global dmax_level
        dmin_level=min(min(dd),dmin_level)
        dmax_level=max(max(dd),dmax_level)
        subplot2.set_ylim(dmin_level-10,dmax_level+10)

    median_lev_db=numpy.median(calData[chan_low:chan_high])
    #Plot a horizontal line representing the average noise floor:
    subplot1.hlines(median_lev_db,freqs[chan_low+1]/1e6,freqs[chan_high-1]/1.e6)
    subplot1.annotate('%3.1f%s'%(median_lev_db,units),(freqs[chan_high]/1.e6,median_lev_db))

   
    #annotate:
    maxs,locs=find_n_max(calData[chan_low:chan_high],n_top,ignore_adjacents=True)
    if max(locs)==min(locs): locs=[0 for i in range(n_top)] # in case we don't find a max, locs will be [-inf, -inf, -inf...]
    maxfreqs=[freqs[locs[i]+chan_low]/1.e6 for i in range(n_top)]
    for i in range(n_top):
        print '  Local max at chan %5i (%6.2fMHz): %6.2f%s'%(locs[i]+chan_low,maxfreqs[i],maxs[i],units)
        subplot1.annotate('%iMHz:%3.1f%s'%(numpy.round(maxfreqs[i]),maxs[i],units),(maxfreqs[i],maxs[i]))

        #if plot_type == 'lin':
        #    subplot.annotate('%iMHz:%3.1fdB'%(freq,lev),(freq,collapseddata[locs[i]/collapse_factor]))
        #elif plot_type == 'log':
        #    subplot.annotate('%iMHz:%3.1fdB'%(freq,lev),(freq,10*numpy.log10(collapseddata[locs[i]/collapse_factor])))
    
    global min_level
    global max_level
    #local_min=min(calData)
    min_level=min(min(calData[chan_low:chan_high]),min_level)
    max_level=max(max(calData[chan_low:chan_high]),max_level)
    subplot1.set_ylim(min_level-10,max_level+10)
    
    fig.canvas.draw()
    if opts.update:
        fig.canvas.manager.window.after(100, drawDataCallback, last_cnt)

if __name__ == '__main__':
    from optparse import OptionParser
    p = OptionParser()
    p.set_usage('%prog [options]')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Enable debug logging mode.')
    p.add_option('-b', '--baseline', dest = 'baseline', action = 'store_true', default=False,
        help = 'Keep the first trace displayed as a baseline.')
    p.add_option('-d', '--diff', dest = 'diff', action = 'store_true', default=False,
        help = 'Also plot the difference between the first trace and subsequent spectra.')
    p.add_option('-r', '--freq_range', dest = 'freq_range',type='string',default=None,
         help = 'Select a frequency range in MHz to plot. Negative frequencies are supported. Default is to ignore top and bottom 50MHz: 50,-50.')
    p.add_option('-s', '--n_top', dest='n_top', type='int',default=5,
        help='Find the top N spiky RFI candidates. Default: 5')
    p.add_option('-f', '--play_file', dest = 'play_file', type='string', default=None,
        help = 'Open an existing file for analysis.')
    p.add_option('-e', '--save_to_file', dest = 'save_to_file', type='string',default=None,
        help = 'Specify the destination filename.')
    p.add_option('-c', '--config_file', dest = 'config_file', type='string',default=None,
        help = 'Specify the configuration file to use.')
    p.add_option('-u', '--update', dest = 'update', action = 'store_false',default=True,
        help = 'Do not update the plots (only plot a single capture).')

#    p.add_option('-n', '--n_chans', dest='n_chans', type='int',default=512,
#        help='Plot this number of channels. Default: 512')
#    p.add_option('-l', '--plot_lin', dest='plot_lin', action='store_true',
#        help='Plot on linear axes. Default: semilogy.')
    p.add_option('-p', '--no_plot', dest='plot', action='store_false',default=True,
        help="Don't plot anything.")
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])

    usrlog=('Starting file at %i.'%(int(time.time()))).join(args)
    if usrlog=='': usrlog=str(int(time.time()))
    #plot_chans=opts.n_chans
    freq_range=opts.freq_range
    #ant=opts.ant
    n_top=opts.n_top
    verbose=opts.verbose
    plot_baseline=opts.baseline
    plot_diff = opts.diff
    #cal_mode=opts.cal
    config_file = opts.config_file
    play_filename=opts.play_file

try:
    if play_filename==None:
        r = ratty1.cam.spec(config_file=config_file)
        co=r.cal
        print 'Config file %s parsed ok!'%(r.config_file)
        print 'Connecting to ROACH %s...'%r.config['roach_ip_str'],
        r.connect()

        if verbose:
            r.logger.setLevel(logging.DEBUG)
        else:
            r.logger.setLevel(logging.INFO)
        print 'done.'

        acc_time,n_accs = r.acc_time_get()
        freqs=r.freqs
        fft_shift=r.fft_shift_get()
        fft_scale=r.fft_scale
        rf_gain=r.rf_status_get()[1]
        n_chans=r.n_chans
        bandwidth=r.bandwidth

        print 'Scaling back by %i accumulations.'%n_accs

        filename=usrlog + ".spec.h5"
        print 'Starting file %s.'%filename
        f = h5py.File(filename, mode="w")
        f['/'].attrs['usrlog']=usrlog

        f.create_dataset('spectra',shape=[1,r.n_chans],maxshape=[None,r.n_chans])
        f.create_dataset('acc_cnt',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f.create_dataset('timestamp',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f['/'].attrs['n_accs']=n_accs
        f['/'].attrs['rf_gain']=rf_gain

        for key in r.config.config.keys():
            #print 'Storing',key
            try:
                f['/'].attrs[key]=r.config[key]
            except:
                try:
                    f[key]=r.config[key]
                except TypeError:
                    if r.config[key]==None: f['/'].attrs[key]='none'
                    elif type(r.config[key])==dict: 
                        f[key]=r.config[key].items()
                        print 'Stored a dict!'

        last_cnt=r.fpga.read_uint('acc_cnt')

    else:
        last_cnt=0
        print 'Opening file %s...'%play_filename
        f=h5py.File(play_filename,'r')
        usrlog=f['/'].attrs['usrlog']
        print 'USRLOG: %s'%usrlog
        conf_ovr=dict(f['/'].attrs)
        for key in f.keys():
            if not key in ['raw_dumps','timestamp','adc_overrange','fft_overrange','adc_shutdown','adc_level','input_level']:
                print 'trying',key
                if len(f[key])>1: conf_ovr[key]=f[key][:]
                else: conf_ovr[key]=f[key]
        conf_ovr['atten_gain_map']=dict(conf_ovr['atten_gain_map'])
        co=ratty1.cal.cal(**conf_ovr)

        n_accs      =f['/'].attrs['n_accs']
        n_chans     =f['/'].attrs['n_chans']
        usrlog      =f['/'].attrs['usrlog']
        bandwidth   =f['/'].attrs['bandwidth']
        rf_gain     =f['/'].attrs['rf_gain']
        fft_shift   =f['/'].attrs['fft_shift']
        freqs       =co.config['freqs']
        fft_scale   =2**(ratty1.cal.bitcnt(fft_shift))


    if co.config['antenna_bandpass_calfile'] == 'none':
        units='dBm'
    else:
        units='dBuV/m'

    bp=co.system_bandpass
    af=co.ant_factor

    if freq_range==None:    
        chan_low =co.freq_to_chan(co.config['ignore_low_freq'])
        chan_high=co.freq_to_chan(co.config['ignore_high_freq'])
    else:
        chan_low =co.freq_to_chan(int(freq_range.split(',')[0])*1e6)
        chan_high=co.freq_to_chan(int(freq_range.split(',')[1])*1e6)
    print 'Working with channels %i (%5.1fMHz) to %i (%5.1fMHz).'%(chan_low,freqs[chan_low]/1.e6,chan_high,freqs[chan_high]/1.e6)

    if opts.plot or play_filename != None:
        # set up the figure with a subplot for each polarisation to be plotted
        fig = matplotlib.pyplot.figure()
        if opts.diff or opts.baseline:
            print 'Fetching baseline...',
            sys.stdout.flush()
            unpackedData, timestamp, last_cnt,stat = getUnpackedData(last_cnt)
            baseline=co.get_calibrated_spectrum(unpackedData, rf_gain) #returns spectrum in dBm
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
            calData=r.cal.get_calibrated_spectrum(unpackedData, rf_gain) #returns spectrum in dBm
            maxs,locs=find_n_max(calData[chan_low:chan_high],n_top,ignore_adjacents=True)
            maxfreqs=[freqs[locs[i]+chan_low]/1.e6 for i in range(n_top)]
            for i in range(n_top):
                print '  Local max at chan %5i (%6.2fMHz): %6.2f%s'%(locs[i]+chan_low,maxfreqs[i],maxs[i],units)


except KeyboardInterrupt:
    exit_clean()
except Exception as e:
    print e
    exit_fail()

print 'Done with all.'
exit_clean()
