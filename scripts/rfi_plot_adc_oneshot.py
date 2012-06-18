#!/usr/bin/env python

'''
Plots a histogram of the ADC values from a specified antenna and pol. Hard-coded for 8 bit ADCs.\n

Revisions:
2011-02-24  JRM Port to RFI system
2010-12-11: JRM Add printout of number of bits toggling in ADC.
                Add warning for non-8bit ADCs.
2010-08-05: JRM Mods to support variable snap block length.
1.1 PVP Initial.\n

'''
import rfi_sys,matplotlib, time, corr, numpy, struct, sys, logging, pylab

# what format are the snap names and how many are there per antenna
snapName = 'snap_adc'
# what is the bram name inside the snap block
bramName = 'bram'

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
    p.set_usage('%prog [options] MODE')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Print raw output.')
    p.add_option('-n', '--n_samples', dest = 'n_samples', type='int', default = 100, help = 'Plot this many time-domain samples around the maximum value. Default:100')
    p.add_option('-t', '--trig_level', dest = 'trig_level', type='int', default = -1, help = 'Ask the hardware to wait for a signal with at least this amplitude before capturing. Valid range: 0-127. Default: negative (disabled, just plot current input).')
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])
#    if args==[]:
#        print 'Please specify a mode!\nExiting.'
#        exit()

def getUnpackedData(trig_level=-1):
    # get the data
    unpackedBytes = r.get_adc_snapshot(trig_level=trig_level) 
    return unpackedBytes, time.time()

try:
    #r = rfi_sys.rfi_sys(mode=args[0])
    r = rfi_sys.rfi_sys.rfi_sys()
    print 'Connecting to ROACH...',
    if opts.verbose:
        r.logger.setLevel(logging.DEBUG)
    else:
        r.logger.setLevel(logging.INFO)
    print 'done.'

    unpackedData, timestamp = getUnpackedData(opts.trig_level)
    pylab.plot(unpackedData[0:opts.n_samples])
    pylab.title('Raw ADC samples %s'%(time.ctime(timestamp)))
    pylab.xlabel('Time in ADC samples')
    pylab.ylabel('ADC count')
    pylab.show()

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

print 'Done with all.'
exit_clean()

# end

