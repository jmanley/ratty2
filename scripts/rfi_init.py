#!/usr/bin/env python
'''
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/

Hard-coded for 32bit unsigned numbers.
\nAuthor: Jason Manley, Feb 2011.
'''


import corr,time,numpy,struct,sys,logging,rfi_sys

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',
    try:
        r.lh.printMessages()
        r.fpga.stop()
    except:
        pass
    raise
    exit()

def exit_clean():
    try:
        r.fpga.stop()
    except:
        pass
    exit()


if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('spectrometer.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-f', '--fft_shift', dest='fft_shift', type='int',default=-1,
        help='Set the FFT shifting schedule.')
    p.add_option('-t', '--acc_period', dest='acc_period', type='float',default=1,
        help='Set the accumulation period. default is about 1 second.')
    p.add_option('-r', '--rf_gain', dest='rf_gain', type='float',
        help='Override the automatic analogue RF gain adjustment with this value in dB. Default=automatic.')
    p.add_option('-s', '--skip_prog', dest='fpga_prog', action='store_false',default=True,
        help='Skip reprogramming the FPGA.')
    opts, args = p.parse_args(sys.argv[1:])

    fft_shift=opts.fft_shift
    acc_period=opts.acc_period
    rf_gain=opts.rf_gain
    fpga_prog=opts.fpga_prog

try:
    r = rfi_sys.cam.spec()
    #r = rfi_sys.rfi_sys(mode=args[0])
    print 'Connecting to ROACH...',
    r.logger.setLevel(logging.DEBUG)
    print 'done.'

    print '------------------------'
    print 'Programming FPGA...',
    sys.stdout.flush()
    if fpga_prog:
        r.fpga.progdev(r.bitstream)
        print 'done'
    else:
        print 'Skipped.'

    print 'Checking clocks...',
    sys.stdout.flush()
    if fpga_prog:
        est_clk_rate=r.clk_check()
        print 'ok, %i MHz.'%est_clk_rate
    else:
        print 'Skipped.'

    print 'Auto-calibrating ADC...',
    sys.stdout.flush()
    r.adc_selfcal()
    print 'done'

    if opts.rf_gain:
        print 'Manually configuring RF gain to %4.1dB...'%rf_gain,
        sys.stdout.flush()
        r.rf_gain_set(rf_gain)
        print 'done. ADC input is currently %s with level of %4.1fdBm.'%('enabled' if r.rf_status_get()[0] else 'disabled',r.adc_amplitudes_get()['adc_dbm'])
    else:
        print 'Attempting automatic RF gain adjustment...'
        max_n_tries=10
        n_tries=0
        tolerance=1
        rf_gain=r.rf_gain_range[0]
        r.rf_gain_set(rf_gain)
        time.sleep(0.1)
        r.ctrl_set(mrst='pulse',cnt_rst='pulse',clr_status='pulse',flasher_en=True)
        rf_level=r.adc_amplitudes_get()['adc_dbm']
        if r.status_get()['adc_bad'] or r.status_get()['adc_overrange']: 
            raise RuntimeError('Your input levels are too high!')

        while (rf_level < r.desired_rf_level-tolerance or rf_level>r.desired_rf_level+tolerance) and n_tries < max_n_tries:
            rf_level=r.adc_amplitudes_get()['adc_dbm']
            difference = r.desired_rf_level - rf_level 
            rf_gain=r.rf_status_get()[1] + difference
            print '\t Gain was %3.1fdB, resulting in an ADC input level of %5.2fdB. Trying gain of %4.2fdB...'%(r.rf_status_get()[1],rf_level,rf_gain)
            if rf_gain < r.rf_gain_range[0]: 
                print '\tWARNING: Gain at minimum, %4.2fdB.'%r.rf_gain_range[0],
                r.logger.warn('Gain at minimum, %4.2fdB.'%r.rf_gain_range[0])
                r.rf_gain_set(r.rf_gain_range[0])
                break
            elif rf_gain > r.rf_gain_range[1]: 
                print '\t WARNING: Gain at maximum, %4.2fdB.'%r.rf_gain_range[1],
                r.logger.warn('Gain at maximum, %4.2fdB.'%r.rf_gain_range[1])
                r.rf_gain_set(r.rf_gain_range[1])
                break 
            r.rf_gain_set(rf_gain)
            time.sleep(0.1)
            n_tries += 1
        if n_tries >= max_n_tries: print 'Failed.'
        else: print 'done!'

    print 'Setting FFT shift... ',
    sys.stdout.flush()
    r.fft_shift_set(fft_shift)
    print 'set to 0x%x.'%r.fft_shift_get()

    print 'Configuring accumulation period to %2.2f seconds...'%opts.acc_period,
    sys.stdout.flush()
    r.acc_time_set(opts.acc_period)
    print 'done'

    print 'Resetting counters...',
    sys.stdout.flush()
    r.ctrl_set(mrst='pulse',cnt_rst='pulse',clr_status='pulse',flasher_en=False)
    print 'done'

    print 'Current status:',
    sys.stdout.flush()
    stat=r.status_get()
    if stat['adc_bad']: print 'ADC selfprotect due to overrange!',
    elif stat['adc_overrange']: print 'ADC is clipping!',
    elif stat['fft_overrange']: print 'FFT is overflowing!',
    else: print 'all ok',
    print ''

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

