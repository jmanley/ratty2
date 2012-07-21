#!/usr/bin/env python
'''
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://pypi.python.org/pypi/corr
\nAuthor: Jason Manley
'''

#Revs:
#2012-07-18 JRM New object oriented cam/cal interface.
#2011-02    JRM First release
import corr,time,numpy,struct,sys,logging, os, ratty1

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',
    try:
        r.lh.printMessages()
        r.fpga.stop()
    except Exception as e:
        print e
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
    p.set_usage('%prog <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-s', '--skip_prog', dest='fpga_prog', action='store_false',default=True,
        help='Skip reprogramming the FPGA.')
    p.add_option('-c', '--config_file', dest = 'config_file', type='string',default=None,
        help = 'Specify the configuration file to use.')
    p.add_option('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Enable debug logging mode.')
    opts, args = p.parse_args(sys.argv[1:])
    config_file = opts.config_file
    verbose=opts.verbose

try:
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

    print '------------------------'

    r.initialise(skip_program=(not opts.fpga_prog), print_progress=True)

except KeyboardInterrupt:
    exit_clean()
except Exception as e:
    print e
    exit_fail()

exit_clean()

