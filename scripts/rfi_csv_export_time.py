#!/usr/bin/env python
"""Exports time-domain data to a CSV file."""
import numpy,rfi_sys,scipy,h5py,matplotlib,pylab,time,csv,sys

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
    p.add_option('-a', '--ant', dest = 'ant', type='string', default = 'none', 
        help = 'Choose an antenna calibration file. Note that this will auto-select y-axis units to be dBuV/m. Default:none')
    #p.add_option('-u', '--units', dest = 'units', type='string', default = 'dBm', 
    #    help = 'Choose the units for y-axis in freq plots. Options include dBuV,dBm. Default:dBm')
    p.add_option('-c', '--n_chans', dest = 'n_chans', type='int', default = 1024, 
        help = 'Number of frequency channels to resolve in software FFT. Default:1024')
    p.add_option('-l', '--trig_level', dest = 'trig_level', type='float', default = 0., 
        help = 'Ask the hardware to wait for a signal with at least this amplitude in mV before capturing. Valid range: 0-127. Default: negative (disabled, just   plot current input).')
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])
    verbose=opts.verbose
    n_chans=opts.n_chans
    ant=opts.ant
    usrlog='_'.join(args)
    if usrlog=='': usrlog='No userlog specified. Starting file at %i.'%(int(time.time()))
    if opts.file: file=opts.file
    else: file=None


if __name__ == '__main__':
    from optparse import OptionParser
    p = OptionParser()
    p.set_usage('%prog [options] h5_file start_index')
    p.add_option('-c', '--n_chans', dest = 'n_chans', type='int', default = 256, 
        help = 'Number of frequency channels to resolve in software FFT. Default:256')
    p.add_option('-t', '--capture_len', dest = 'capture_len', type='int', default = 8000, 
        help = 'Plot this many nano-seconds around the trigger point. Default:8000')
    opts, args = p.parse_args(sys.argv[1:])
    if len(args) !=2: 
        print 'Please specify the h5 file and the start index.'
        exit()
    file=args[0]
    cnt=int(args[1])
    n_chans=opts.n_chans
    ns=opts.capture_len

f=h5py.File(file,'r')

usrlog      =f['/'].attrs['usrlog']
bandwidth   =f['/'].attrs['bandwidth']
rf_gain     =f['/'].attrs['rf_gain']
trig_level  =f['/'].attrs['trig_level']
print 'USRLOG: %s'%usrlog
sample_clk=898000000*2
#        sample_clk=f['/'].attrs['sample_clk']
n_samples=int(ns*1e9/sample_clk)
trig_scale_factor=rfi_sys.cal.get_adc_cnt_mv_scale_factor(rf_gain)

freqs=numpy.arange(n_chans)*float(bandwidth)/n_chans #channel center freqs in Hz
bp=rfi_sys.cal.bandshape(freqs)
if file==None:
    f['bandshape']=bp
af=None

print 'Triggering at a level of %4.2fmV (ADC level of %i).'%(trig_level*trig_scale_factor,trig_level)
print 'Plotting %i samples.'%n_samples

# set up the figure with a subplot for each polarisation to be plotted
fig = matplotlib.pyplot.figure()

# create the subplots
subplots = []
for p in range(2):
    subPlot = fig.add_subplot(2, 1, p + 1)
    subplots.append(subPlot)
fig.subplots_adjust(hspace=0.8)

def getUnpackedData(trig_level=-1):
    if cnt+1>=f['raw_dumps'].shape[0]: exit()
    unpackedBytes = f['raw_dumps'][cnt]
    stat={'adc_overrange':f['adc_overrange'][cnt],
            'adc_bad':f['adc_shutdown'][cnt],
            'fft_overrange':f['fft_overrange'][cnt],
            'adc_level':f['adc_level'][cnt]}
    timestamp=f['timestamp'][cnt]

    print '%s: ADC input level: %5.2f dBm.'%(time.ctime(timestamp),stat['adc_level']),
    if stat['adc_bad']: print 'ADC selfprotect due to overrange!',
    if stat['adc_overrange']: print 'ADC is clipping!',
    if stat['fft_overrange']: print 'FFT is overflowing!',
    print ''
    return unpackedBytes, timestamp, stat

def filewrite(adc_data_raw, adc_data_mv, spectrum_empty, spectrum_dbuv, timestamp,status):
    fcp=open(str(timestamp)+'.csv','w')
    fc=csv.writer(fcp)
    cnt=f['raw_dumps'].shape[0]-1
    print '  Storing entry %i...'%cnt,
    sys.stdout.flush()
    fc.writerow(['timestamp'] + [timestamp])
    fc.writerow(['bandwidth_hz'] + [bandwidth])
    fc.writerow(['sample_clk_hz'] + [sample_clk])
    fc.writerow(['adc_overrange']+ [status['adc_overrange']])
    fc.writerow(['fft_overrange']+ [status['fft_overrange']])
    fc.writerow(['adc_shutdown']+ [status['adc_bad']])
    fc.writerow(['ave_adc_level_dbm']+ [status['adc_level']])
    fc.writerow(['usrlog'] + [usrlog])
    fc.writerow(['rf_gain_db'] + [rf_gain])
    fc.writerow(['trig_level']+[trig_level])
    fc.writerow(['bandshape_db']+ numpy.array(bp).tolist())
    fc.writerow(['adc_data_raw']+ numpy.array(adc_data_raw).tolist())
    fc.writerow(['adc_data_mv']+ numpy.array(adc_data_mv).tolist())
    fc.writerow(['spectrum_freqs_hz']+ numpy.array(freqs).tolist())
    fc.writerow(['spectrum_empty_dbuv']+ numpy.array(spectrum_empty).tolist())
    fc.writerow(['spectrum_dbuv']+ numpy.array(spectrum_dbuv).tolist())
    print 'done writing CSV file'
    fcp.close()

def drawData():
    unpackedData, timestamp,status = getUnpackedData()
    calData=unpackedData*trig_scale_factor
    abs_levs=numpy.abs(calData)
    max_lev =numpy.max(abs_levs)
    trigs = numpy.ma.flatnotmasked_edges(numpy.ma.masked_less_equal(abs_levs,(trig_level-4)*trig_scale_factor))
    if (trigs == None or trigs[0] ==0) and trig_level>0 and (max_lev/trig_scale_factor)<trig_level:
        max_pos = numpy.argmax(calData)
        print('ERROR: we asked for a trigger level of %4.2f mV and the hardware reported success, but the maximum level in the returned data was only %4.2fmV.'%(trig_level*trig_scale_factor,max_lev))

    if trigs==None:
        max_pos = numpy.argmax(calData)
    else:
        max_pos = trigs[0]

    subplots[0].cla()
    t_start =max(0,max_pos-n_samples/2)
    t_stop  =min(len(calData),max_pos+n_samples/2)
    p_data  =calData[t_start:t_stop]
    x_range =numpy.arange(t_start-max_pos,t_stop-max_pos)*1.e9/sample_clk

    subplots[0].plot(x_range,p_data)
    subplots[0].set_xlim(-n_samples/2*1.e9/sample_clk,n_samples/2*1.e9/sample_clk)

    if status['adc_overrange'] or status['adc_bad']:
        subplots[0].set_title('Time-domain [%i] (max %4.2fmV)'%(cnt,max_lev), bbox=dict(facecolor='red', alpha=0.5))
    else:
        subplots[0].set_title('Time-domain [%i] (max %4.2fmV; ADC %i)'%(cnt,max_lev,numpy.max(numpy.abs(unpackedData))))
    subplots[0].set_ylim(-max_lev-1,max_lev+1)
    subplots[0].set_ylabel('mV')
    subplots[0].set_xlabel('Time (nanoseconds).')

    localfreqs,emptySpectrum=rfi_sys.cal.get_calibrated_spectrum_from_raw_snapshot(
        adcdata=unpackedData[0:max_pos-1],
        atten=rf_gain,
        bandwidth=bandwidth,
        ant_factor=af,
        bandshape=bp,
        n_chans=n_chans)

    localfreqs,calSpectrum=rfi_sys.cal.get_calibrated_spectrum_from_raw_snapshot(
        adcdata=unpackedData,
        atten=rf_gain,
        bandwidth=bandwidth,
        ant_factor=af,
        bandshape=bp,
        n_chans=n_chans)

    calSpectrum=rfi_sys.cal.dbm_to_dbuv(calSpectrum)
    emptySpectrum=rfi_sys.cal.dbm_to_dbuv(emptySpectrum)
    print 'plotting from adc sample %i to %i'%(t_start,max_pos-1)
    subplots[1].cla()
    pylab.hold(True)
    subplots[1].plot(freqs/1e6,calSpectrum,label='Signal on')
    pylab.hold(True)
    subplots[1].plot(freqs/1e6,emptySpectrum,label='Quiescent')
    subplots[1].legend()
    subplots[1].set_title('Spectrum of capture (%i samples)'%(len(unpackedData)))
    subplots[1].set_ylabel('Level (dBuv)')
    subplots[1].set_xlabel('Frequency (MHz)')

    filewrite(unpackedData,calData,emptySpectrum,calSpectrum,timestamp,status)

    raw_input()

drawData()
