#!/usr/bin/env python

'''
Plots a histogram and time-domain sample of the ADC values from a specified antenna and pol.

'''
'''
Revisions:
2012-06-15  JRM Update to use objectified package
                trig scale factor now referenced to input levels.
                Added option to not plot histogram.
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
import ratty1, time, corr, numpy, struct, sys, logging, pylab, h5py, os, iniparse, csv

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
    cald=co.calibrate_adc_snapshot(raw_data=unpackedData,rf_gain=rf_gain,n_chans=n_chans)
    filewrite(unpackedData,timestamp,status)

    if indep:
        mean_lev =numpy.mean(unpackedData)
        meana_lev =numpy.mean(unpackedData[0::2])
        meanb_lev =numpy.mean(unpackedData[1::2])
        std_deva = numpy.std(unpackedData[0::2])
        std_devb = numpy.std(unpackedData[1::2])
        print '%s: Mean ADC-A: %4.2f, mean ADC-B: %4.2f, overall: %4.2f. std-dev-ADC-A: %4.2f, std-dev-ADC-B: %4.2f.'%(time.ctime(timestamp),meana_lev,meanb_lev,mean_lev,std_deva,std_devb)


    calData=cald['input_v']*1000 #in mV
    freqs=cald['freqs']
    abs_levs=numpy.abs(calData)
    max_lev =numpy.max(abs_levs)
    trigs = numpy.ma.flatnotmasked_edges(numpy.ma.masked_less_equal(abs_levs,(trig_level-4)*trig_scale_factor))
    #print trigs
    if (trigs == None or trigs[0] ==0) and trig_level>0 and (max_lev/trig_scale_factor)<trig_level: 
        #r.logger.error('Error triggering. Found no trigger points.')
        max_pos = numpy.argmax(calData)
        #r.logger.error('ERROR: we asked for a trigger level of %4.2fmV and the hardware reported success, but the maximum level in the returned data was only %4.2fmV.'%(trig_level*trig_scale_factor,max_lev))
        print('ERROR: we asked for a trigger level of %4.2f mV and the hardware reported success, but the maximum level in the returned data was only %4.2fmV.'%(trig_level*trig_scale_factor,max_lev))

    if trigs==None:
        max_pos = numpy.argmax(calData)
    else:
        max_pos = trigs[0]
    
    next_subplot=0
    if opts.plot_hist:
        subplots[0].cla()
        subplots[0].set_xticks(range(-130, 131, 20))
        histData, bins, patches = subplots[0].hist(unpackedData, bins = 256, range = (-128,128))
        if status['adc_overrange'] or status['adc_shutdown']:
            subplots[0].set_title('Histogram as at %s'%(time.ctime(timestamp)),bbox=dict(facecolor='red', alpha=0.5))
        else:
            subplots[0].set_title('Histogram as at %s'%(time.ctime(timestamp)))
        subplots[0].set_ylabel('Counts')
        subplots[0].set_xlabel('ADC sample bins.')
        matplotlib.pyplot.ylim(ymax = (max(histData) * 1.05))
        next_subplot+=1
    subplots[next_subplot].cla()
    if indep:
        max_pos=(max_pos/2)*2
        t_start =max(0,max_pos-n_samples/2)
        t_stop  =min(len(calData),max_pos+n_samples/2)
        p_data=calData[t_start:t_stop]
        x_range=numpy.arange(t_start-max_pos,t_stop-max_pos)
        #print max_pos,t_start,t_stop,len(x_range)

        subplots[next_subplot].plot(x_range[0::2],p_data[0::2])
        subplots[next_subplot].plot(x_range[1::2],p_data[1::2])
        subplots[next_subplot].set_xlim(-n_samples/4*1.e9/sample_clk,n_samples/4)
    else:
        t_start =max(0,max_pos-n_samples/2)
        t_stop  =min(len(calData),max_pos+n_samples/2)
        p_data  =calData[t_start:t_stop]
        x_range =numpy.arange(t_start-max_pos,t_stop-max_pos)*1.e9/sample_clk
        #print max_pos,t_start,t_stop,len(x_range)

        subplots[next_subplot].plot(x_range,p_data)
        subplots[next_subplot].set_xlim(-n_samples/2*1.e9/sample_clk,n_samples/2*1.e9/sample_clk)

    if status['adc_overrange'] or status['adc_shutdown']:
        subplots[next_subplot].set_title('Time-domain [%i] (max >%4.2fmV)'%(cnt-1,max_lev), bbox=dict(facecolor='red', alpha=0.5))
    else:
        subplots[next_subplot].set_title('Time-domain [%i] (max %4.2fmV; ADC %i)'%(cnt-1,max_lev,numpy.max(numpy.abs(unpackedData))))
    subplots[next_subplot].set_ylim(-max_lev-1,max_lev+1)
    subplots[next_subplot].set_ylabel('mV')
    subplots[next_subplot].set_xlabel('Time (nanoseconds).')
    next_subplot+=1

    subplots[next_subplot].cla()
    t_start =0
    #t_start =max(0,max_pos-(n_chans*2)-1)
    if co.config['antenna_bandpass_calfile']=='none':
        calSpectrum=cald['input_spectrum_dbm']
        emptySpectrum=co.calibrate_adc_snapshot(raw_data=unpackedData[t_start:max_pos-1],rf_gain=rf_gain, n_chans = n_chans)['input_spectrum_dbm']
    else:
        calSpectrum=cald['input_spectrum_dbuv']
        emptySpectrum=co.calibrate_adc_snapshot(raw_data=unpackedData[t_start:max_pos-1],rf_gain=rf_gain, n_chans = n_chans)['input_spectrum_dbuv']
        
    #print 'got a spectrum:',calSpectrum
    #print 'plotting from %i to %i'%(t_start,max_pos-1)
    pylab.hold(True)
    subplots[next_subplot].plot(freqs[chan_low:chan_high]/1e6,calSpectrum[chan_low:chan_high],label='Signal on')
    pylab.hold(True)
    subplots[next_subplot].plot(freqs[chan_low:chan_high]/1e6,emptySpectrum[chan_low:chan_high],label='Quiescent')
    subplots[next_subplot].legend()
    subplots[next_subplot].set_title('Spectrum of capture (%i samples)'%(len(unpackedData[chan_low:chan_high])))
    subplots[next_subplot].set_ylabel('Level (%s)'%units)
    subplots[next_subplot].set_xlabel('Frequency (MHz)')

    if opts.csv_file:
        csv_writer(cald,timestamp,status,quiescent=emptySpectrum) 

    fig.canvas.draw()
    if update: 
        if wait_keypress:
            print '\t Press enter to get another capture...'
            raw_input()
        fig.canvas.manager.window.after(100, drawDataCallback, n_samples,indep,trig_level)

# the function that gets data given a required polarisation
def getUnpackedData(trig_level=-1):
    # get the data
    if play_filename==None:
        print 'Fetching data from ROACH...',
        unpackedBytes = r.get_adc_snapshot(trig_level=trig_level) 
        print 'done'
        stat=r.status_get()
        ampls=r.adc_amplitudes_get()
        stat['adc_level']=ampls['adc_dbm']
        stat['input_level']=ampls['input_dbm']
        timestamp=time.time()
    else:
        global cnt
        if cnt+1>=f['raw_dumps'].shape[0]: exit_clean()
        unpackedBytes = f['raw_dumps'][cnt] 
        stat={'adc_overrange':f['adc_overrange'][cnt],
                'adc_shutdown':f['adc_shutdown'][cnt],
                'fft_overrange':f['fft_overrange'][cnt],
                'adc_level':f['adc_level'][cnt],
                'input_level':f['input_level'][cnt]}
        timestamp=f['timestamp'][cnt]
        cnt+=1

    print '%s: input level: %5.2f dBm (%5.2fdBm into ADC).'%(time.ctime(timestamp),stat['input_level'],stat['adc_level']),
    if stat['adc_shutdown']: print 'ADC selfprotect due to overrange!',
    if stat['adc_overrange']: print 'ADC is clipping!',
    if stat['fft_overrange']: print 'FFT is overflowing!',
    print ''
    return unpackedBytes, timestamp, stat

def filewrite(adc_data,timestamp,status):
    if not play_filename:
        global cnt
        cnt=f['raw_dumps'].shape[0]-1
        print '  Storing entry %i...'%cnt,
        sys.stdout.flush()
        f['raw_dumps'][cnt] = adc_data 
        f['timestamp'][cnt] = timestamp
        f['adc_overrange'][cnt] = status['adc_overrange']
        f['fft_overrange'][cnt] = status['fft_overrange']
        f['adc_shutdown'][cnt] = status['adc_shutdown']
        f['adc_level'][cnt] = status['adc_level']
        f['input_level'][cnt] = status['input_level']
        for name in ['raw_dumps','timestamp','adc_overrange','fft_overrange','adc_shutdown','adc_level','input_level']:
            f[name].resize(cnt+2, axis=0)
        print 'done'

def csv_writer(cald,timestamp,status,quiescent):
    fcp=open(str(timestamp)+'.csv','w')
    fc=csv.writer(fcp)
    for key in r.config.config.keys():
        if type(r.config[key])==list:
            fc.writerow([key] + r.config[key])
        elif type(r.config[key])==numpy.ndarray:
            fc.writerow([key] + r.config[key].tolist())
        else:
            fc.writerow([key] + [r.config[key]])
    fc.writerow(['trig_level']+[trig_level])
    fc.writerow(['timestamp']+['%s'%time.ctime(timestamp)])
    fc.writerow(['adc_overrange']+ [status['adc_overrange']])
    fc.writerow(['fft_overrange']+ [status['fft_overrange']])
    fc.writerow(['adc_shutdown']+ [status['adc_shutdown']])
    fc.writerow(['ave_adc_level_dbm']+ [status['adc_level']])
    fc.writerow(['ave_input_level_dbm']+ [status['input_level']])

    fc.writerow(['raw_adc','adc_v','input_v','freq','input_spectrum_dbm','input_spectrum_dbuv','quiescent'])
    for i in range(len(cald['adc_v'])):
        if i < n_chans:
            if co.config['antenna_bandpass_calfile'] != 'none':
                fc.writerow([cald['adc_raw'][i],cald['adc_v'][i],cald['input_v'][i],cald['freqs'][i],cald['input_spectrum_dbm'][i],cald['input_spectrum_dbuv'][i],quiescent[i]])
            else:
                fc.writerow([cald['adc_raw'][i],cald['adc_v'][i],cald['input_v'][i],cald['freqs'][i],cald['input_spectrum_dbm'][i],quiescent[i]])
        else:
            fc.writerow([cald['adc_raw'][i],cald['adc_v'][i],cald['input_v'][i]])

    fcp.close()

if __name__ == '__main__':
    from optparse import OptionParser
    p = OptionParser()
    p.set_usage('%prog [options] LOG_MESSAGE')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true',default=False, 
        help = 'Enable debug mode.')
    p.add_option('-o', '--plot_hist', dest = 'plot_hist', action = 'store_false',default=True, 
        help = 'Do not plot the histogram.')
    p.add_option('-u', '--update', dest = 'update', action = 'store_false',default=True, 
        help = 'Do not update the plots (only plot a single capture).')
    p.add_option('-i', '--plot_indep', dest = 'plot_indep', action = 'store_true', 
        help = 'Plot interleaved ADC independantly.')
    p.add_option('-t', '--capture_len', dest = 'capture_len', type='int', default = 100, 
        help = 'Plot this many nano-seconds around the trigger point. Default:100')
    p.add_option('-a', '--ant', dest = 'ant', type='string', default = 'none', 
        help = 'Choose an antenna calibration file. Note that this will auto-select y-axis units to be dBuV/m. Default:none')
    p.add_option('-n', '--n_chans', dest = 'n_chans', type='int', default = 512, 
        help = 'Number of frequency channels to resolve in software FFT. Default:512')
    p.add_option('-l', '--trig_level', dest = 'trig_level', type='float', default = 0., 
        help = 'Ask the hardware to wait for a signal with at least this amplitude in mV before capturing. Default: 0 (disabled, just plot current input).')
    p.add_option('-f', '--play_file', dest = 'play_file', type='string', default=None,
        help = 'Open an existing file for analysis.')
    p.add_option('-e', '--save_to_file', dest = 'save_to_file', type='string',default=None,
        help = 'Specify the destination filename.') 
    p.add_option('-c', '--config_file', dest = 'config_file', type='string',default=None,
        help = 'Specify the configuration file to use.')
    p.add_option('-s', '--csv_file', dest = 'csv_file', action='store_true', default=False,
        help = 'Output (convert) each timestamp to a separate CSV file.')
    p.add_option('-w', '--wait_keypress', dest = 'wait_keypress', action='store_true', default=False,
        help = 'Wait for a user keypress before storing/plotting the next update.')

    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])
    verbose=opts.verbose
    n_chans=opts.n_chans
    ant=opts.ant
    usrlog=('Starting file at %i.'%(int(time.time()))).join(args)
    config_file = opts.config_file
    play_filename=opts.play_file
    wait_keypress=opts.wait_keypress
    update=opts.update


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

        rf_gain=r.rf_status_get()[1]
        r.config['rf_gain']=rf_gain
        trig_scale_factor=r.cal.get_input_adc_v_scale_factor(rf_gain)*r.config['adc_v_scale_factor']
        n_samples=int(opts.capture_len/1.e9*r.sample_clk)
        trig_level=int(opts.trig_level/trig_scale_factor)
        bandwidth=r.bandwidth
        sample_clk=r.sample_clk
        antenna_bandpass_calfile=r.config['antenna_bandpass_calfile'].strip()
        system_bandpass_calfile=r.config['system_bandpass_calfile'].strip()
    
        unpackedData, timestamp,status = getUnpackedData(trig_level=0)
        cald=co.calibrate_adc_snapshot(raw_data=unpackedData,rf_gain=rf_gain,n_chans=n_chans) 
        freqs=cald['freqs']
        r.config['freqs']=freqs
        if r.config['antenna_bandpass_calfile'] != 'none':
            af=cald['ant_factor']
            abp=cald['antenna_bandpass']
        else:
            af=numpy.zeros(n_chans)
            abp=numpy.zeros(n_chans)
        if r.config['system_bandpass_calfile'] != 'none':
            bp=cald['system_bandpass']
        else:
            bp=numpy.zeros(n_chans)
        r.config['system_bandpass']=bp
        r.config['antenna_bandpass']=abp
        r.config['ant_factor']=af

        filename=str(int(time.time())) + ".time.h5"
        print 'Starting file %s.'%filename
        f = h5py.File(filename, mode="w")
        print 'fetching baseline...',
        sys.stdout.flush()
        baseline=r.get_adc_snapshot()
        print 'done'
        f['/'].attrs['usrlog']=usrlog
        f.create_dataset('raw_dumps',shape=[1,len(baseline)],dtype=numpy.int8,maxshape=[None,len(baseline)])
        f.create_dataset('timestamp',shape=[1],maxshape=[None],dtype=numpy.uint32)
        f.create_dataset('adc_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('fft_overrange',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_shutdown',shape=[1],maxshape=[None],dtype=numpy.bool)
        f.create_dataset('adc_level',shape=[1],maxshape=[None],dtype=numpy.float)
        f.create_dataset('input_level',shape=[1],maxshape=[None],dtype=numpy.float)
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
                        
        f['/'].attrs['rf_gain']=rf_gain
        f['/'].attrs['trig_level']=trig_level


    else:
        global cnt
        cnt=0
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

        rf_gain=co.config['rf_gain']
        trig_scale_factor=co.get_input_adc_v_scale_factor(rf_gain)*co.config['adc_v_scale_factor']
        sample_clk=co.config['sample_clk']
        n_samples=int(opts.capture_len/1.e9*co.config['sample_clk'])
        trig_level=int(opts.trig_level/trig_scale_factor)
        bandwidth=co.config['bandwidth']
        freqs=co.config['freqs']
        antenna_bandpass_calfile=co.config['antenna_bandpass_calfile']
        system_bandpass_calfile=co.config['system_bandpass_calfile']
        af=co.ant_factor
        bp=co.config['system_bandpass']
        trig_level =f['/'].attrs['trig_level']

    if antenna_bandpass_calfile == 'none':
        units='dBm'
    else:
        units='dBuV/m'
    n_samples=int(opts.capture_len*1e9/sample_clk)

    print 'Triggering at a level of %4.2fmV (ADC level of %i).'%(trig_level*trig_scale_factor,trig_level)
    print 'Plotting %i samples.'%n_samples
    chan_low =co.freq_to_chan(co.config['ignore_low_freq'],n_chans=n_chans)
    chan_high=co.freq_to_chan(co.config['ignore_high_freq'],n_chans=n_chans)
    print 'Working with channels %i (%5.1fMHz) to %i (%5.1fMHz).'%(chan_low,freqs[chan_low]/1.e6,chan_high,freqs[chan_high]/1.e6)

    # create the subplots
    fig = matplotlib.pyplot.figure()
    subplots = []

    n_subplots=2
    if opts.plot_hist: n_subplots+=1

    for p in range(n_subplots):
        subPlot = fig.add_subplot(n_subplots, 1, p + 1)
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

