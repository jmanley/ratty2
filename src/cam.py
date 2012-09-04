#!/usr/bin/env python
'''
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/

Hard-coded for 32bit unsigned numbers.
\nAuthor: Jason Manley, Feb 2011.
'''

import corr,time,numpy,struct,sys,logging,ratty1,cal,conf,iniparse, os


class spec:
    def __init__(self, config_file='/etc/ratty1/default', connect=False, log_handler=None, log_level=logging.INFO):

        if log_handler == None: log_handler=corr.log_handlers.DebugLogHandler(100)
        if config_file == None: config_file='/etc/ratty1/default'
        self.lh = log_handler
        self.logger = logging.getLogger('RATTY1')
        self.logger.setLevel(log_level)
        self.logger.addHandler(self.lh)
        self.config_file=config_file

        self.config=ratty1.conf.rattyconf(config_file=self.config_file)
        self.cal=ratty1.cal.cal(config_file=self.config_file)

        #self.mode = self.sys_config['digital_system_parameters']['mode'].strip()
        self.n_chans = self.config['n_chans']
        self.bandwidth = self.config['bandwidth']
        self.n_par_streams = self.config['n_par_streams']
        self.bitstream = self.config['bitstream']
        self.fft_shift = self.config['fft_shift']
        self.adc_type = self.config['adc_type']
        self.desired_rf_level = self.config['desired_rf_level']
        self.spectrum_bits = self.config['spectrum_bits']
        self.acc_period = self.config['acc_period']
        self.rf_gain = self.config['rf_gain']
        self.fpga_clk=self.config['fpga_clk']
        self.sample_clk=self.config['sample_clk']
        self.rf_gain_range=self.config['rf_gain_range']
        self.chan_width=self.config['chan_width']
        self.freqs=self.config['freqs']
        self.roach_ip = self.config['roach_ip_str']
        self.katcp_port = self.config['katcp_port']
        self.adc_levels_acc_len=self.config['adc_levels_acc_len']
        self.spectrum_bram_out_prefix=self.config['spectrum_bram_out_prefix']
        self.fe_gain=self.config['fe_gain']

        self.last_cnt=0

        self.antenna_bandpass_calfile=self.config['antenna_bandpass_calfile']
        self.system_bandpass_calfile=self.config['system_bandpass_calfile']
        self.atten_gain_calfile=self.config['atten_gain_calfile']

        self.system_bandpass = self.cal.system_bandpass
        self.antenna_bandpass = self.cal.antenna_bandpass
        self.atten_gain_map = self.cal.atten_gain_map
        #self.af = self.cal.ant_gain()

        if connect:
            self.connect()

    def connect(self):
        self.logger.info('Trying to connect to ROACH %s on port %i...'%(self.roach_ip,self.katcp_port))
        self.fpga=corr.katcp_wrapper.FpgaClient(self.roach_ip,self.katcp_port,timeout=10,logger=self.logger)
        time.sleep(1)
        try:
            self.fpga.ping()
            self.logger.info('KATCP connection ok.')
        except:
            self.logger.error('KATCP connection failure. Connection to ROACH failed.')
            print('KATCP connection failure.')
            raise RuntimeError("Connection to FPGA board failed.")

    def auto_gain(self,print_progress=False):
        self.logger.info('Attempting automatic RF gain adjustment...')
        if print_progress: print ('Attempting automatic RF gain adjustment...')
        max_n_tries=10
        n_tries=0
        tolerance=1
        rf_gain=self.rf_gain_range[0]
        self.rf_gain_set(rf_gain)
        time.sleep(0.1)
        self.ctrl_set(mrst='pulse',cnt_rst='pulse',clr_status='pulse',flasher_en=True)
        rf_level=self.adc_amplitudes_get()['adc_dbm']
        if self.status_get()['adc_shutdown'] or self.status_get()['adc_overrange']:
            self.logger.error('Your input levels are too high!')
            raise RuntimeError('Your input levels are too high!')

        while (rf_level < self.desired_rf_level-tolerance or rf_level>self.desired_rf_level+tolerance) and n_tries < max_n_tries:
            rf_level=self.adc_amplitudes_get()['adc_dbm']
            difference = self.desired_rf_level - rf_level
            rf_gain=self.rf_status_get()[1] + difference
            log_str='Gain was %3.1fdB, resulting in an ADC input level of %5.2fdB. Trying gain of %4.2fdB...'%(self.rf_status_get()[1],rf_level,rf_gain)
            self.logger.info(log_str)
            if print_progress: print log_str
            if self.rf_gain < self.rf_gain_range[0]:
                log_str='Gain at minimum, %4.2fdB.'%self.rf_gain_range[0]
                self.logger.warn(log_str)
                if print_progress: print log_str
                self.rf_gain_set(self.rf_gain_range[0])
                break
            elif rf_gain > self.rf_gain_range[1]:
                log_str='Gain at maximum, %4.2fdB.'%self.rf_gain_range[1]
                self.logger.warn(log_str)
                if print_progress: print log_str
                self.rf_gain_set(self.rf_gain_range[1])
                break
            self.rf_gain_set(rf_gain)
            time.sleep(0.1)
            n_tries += 1
        if n_tries >= max_n_tries: 
            log_str='Auto RF gain adjust failed.'
            self.logger.error(log_str)
            if print_progress: print log_str
        else: 
            log_str='Auto RF gain adjust success.'
            if print_progress: print log_str
            self.logger.info(log_str)

    def initialise(self,skip_program=False, input_sel='Q',print_progress=False):
        """Initialises the system to defaults."""
        if print_progress:
            print '\tProgramming FPGA...',
            sys.stdout.flush()
        if not skip_program:
            self.fpga.progdev(self.bitstream)
            if print_progress: print 'ok'
        elif print_progress: print 'skipped'
                
        if print_progress:
            print '\tChecking clocks...',
            sys.stdout.flush()
        est_rate=self.clk_check()
        if print_progress: print 'ok, %i MHz'%est_rate

        if print_progress:
            print '\tSelf-cal ADC...',
            sys.stdout.flush()
        self.adc_selfcal()
        if print_progress: print 'ok'
        
        if print_progress:
            print '\tConfigure ADC...',
            sys.stdout.flush()
        if self.adc_type == 'katadc':
            corr.katadc.set_interleaved(self.fpga,0,input_sel)
            if print_progress: print 'selected input %s.'%input_sel,
        print 'ok'

        if print_progress:
            print '\tConfiguring RF gain...',
            sys.stdout.flush()
        self.rf_gain_set(self.rf_gain,print_progress=print_progress)
        if print_progress: print 'ok'
    
        if print_progress:
            print '\tConfiguring FFT shift schedule...',
            sys.stdout.flush()
        self.fft_shift_set(self.fft_shift)
        if print_progress: print 'ok'
    
        if print_progress:
            print '\tConfiguring accumulation period to %4.2f seconds...'%self.acc_period,
            sys.stdout.flush()
        self.acc_time_set(self.acc_period)
        if print_progress: print 'ok'

        if print_progress:
            print '\tClearing status...',
            sys.stdout.flush()
        self.ctrl_set(mrst='pulse',cnt_rst='pulse',clr_status='pulse',flasher_en=False)
        if print_progress: print 'ok'

        stat=self.status_get()
        if stat['adc_shutdown']: 
            log_msg='ADC selfprotect due to overrange!'
            self.logger.error(log_msg)
            if print_progress: print log_msg
        elif stat['adc_overrange']: 
            log_msg='ADC is clipping!'
            self.logger.warn(log_msg)
            if print_progress: print log_msg
        elif stat['fft_overrange']: 
            log_msg='FFT is overflowing!'
            self.logger.error(log_msg)
            if print_progress: print log_msg

    def clk_check(self):
        """Performs a clock check and returns an estimate of the FPGA's clock frequency."""
        est_rate=round(self.fpga.est_brd_clk())
        if est_rate>(self.fpga_clk/1e6 +1) or est_rate<(self.fpga_clk/1e6 -1):
            self.logger.error('FPGA clock rate is %i MHz where we expect it to be %i MHz.'%(est_rate,self.fpga_clk/1e6))
            raise RuntimeError('FPGA clock rate is %i MHz where we expect it to be %i MHz.'%(est_rate,self.fpga_clk/1e6))
        return est_rate

    def getSpectra(self,n_acc):
        """Fetches n_acc spectra from hardware. Data is uncalibrated."""
        spectra = []
        acc_cnt = []
        timestamp = []
        adc_overrange = []
        fft_overrange = []
        adc_shutdown = []
        adc_level = []
        input_level = []
        adc_temp = []
        ambient_temp = []

        while n_acc:        #While we still need to grab next spectra
            spectrum, time, self.last_cnt, stat = self.getUnpackedData()
            spectra.append(spectrum)
            acc_cnt.append(self.last_cnt)
            timestamp.append(time)
            adc_overrange.append(stat['adc_overrange'])
            fft_overrange.append(stat['fft_overrange'])
            adc_shutdown.append(stat['adc_shutdown'])
            adc_level.append(stat['adc_level'])
            input_level.append(stat['input_level'])
            adc_temp.append(stat['adc_temp'])
            ambient_temp.append(stat['ambient_temp'])
            n_acc = n_acc - 1

        return { 'spectrum':spectra, 
                'acc_cnt':acc_cnt, 
                'timestamp':timestamp, 
                'adc_overrange':adc_overrange, 
                'fft_overrange': fft_overrange, 
                'adc_shutdown':adc_shutdown, 
                'adc_level':adc_level, 
                'input_level':input_level, 
                'adc_temp':adc_temp, 
                'ambient_temp': ambient_temp, 
                }

    def cal_gains(self,low,high):
        base_gain=self.rf_gain_set((low+high)/2.)
        time.sleep(0.2)
        base_power=self.adc_amplitudes_get()['adc_dbm']-base_gain
        gain_cal=[]
        for g in numpy.arange(low,high+self.config['rf_gain_range'][2],self.config['rf_gain_range'][2]):
            self.rf_gain_set(g)
            time.sleep(0.2)
            gain_cal.append(self.adc_amplitudes_get()['adc_dbm']-base_power)
        return gain_cal

    def getUnpackedData(self):
        """Gets data from ROACH board and returns the spectra and the state of the roach at the last timestamp."""
        if self.config['spectrum_bits'] != 64:
            self.logger.error('ERR: Sorry, this function is only for 64 bit accumulators.')
            raise RuntimeError('ERR: Sorry, this function is only for 64 bit accumulators.')

        while self.fpga.read_uint('acc_cnt') <= (self.last_cnt):  #Wait until the next accumulation has been performed
            time.sleep(0.1)
            #print "cnt = " + str(self.fpga.read_uint('acc_cnt'))
        spectrum = numpy.zeros(self.n_chans)              #Get spectra
        for i in range(self.n_par_streams):
            spectrum[i::self.n_par_streams] = numpy.fromstring(self.fpga.read('%s%i'%(self.spectrum_bram_out_prefix,i),self.n_chans/self.n_par_streams*8),dtype=numpy.uint64).byteswap()
        stat = self.status_get()
        ampls = self.adc_amplitudes_get()
        stat['adc_level'] = ampls['adc_dbm']
        stat['input_level'] = ampls['input_dbm']
        stat['adc_temp'] = self.adc_temp_get()
        stat['ambient_temp'] = self.ambient_temp_get()
        self.last_cnt = self.fpga.read_uint('acc_cnt')
        timestamp = time.time()

        #print '[%i] %s: input level: %5.2f dBm (ADC %5.2f dBm).'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),
        if stat['adc_shutdown']: self.logger.error('ADC selfprotect due to overrange!')
        elif stat['adc_overrange']: self.logger.warning('ADC is clipping!')
        elif stat['fft_overrange']: self.logger.error('FFT is overflowing!')
        return spectrum, timestamp, self.last_cnt, stat

    def getAttributes (self):
        """Returns basic system attributes (n_chans, bandwidth etc)"""
        ret = dict()
        ret['n_chans'] = self.n_chans
        ret['n_accs'] = self.acc_time_get()[1]
        ret['bitstream'] = self.bitstream
        ret['bandwidth'] = self.bandwidth
        ret['adc_type'] = self.adc_type
        ret['spectrum_bits'] = self.spectrum_bits
        ret['fft_shift'] = self.fft_shift_get()
        ret['rf_gain'] = self.rf_status_get()[1]
        ret['fe_gain'] = self.fe_gain
        ret['antenna_calfile'] = self.antenna_bandpass_calfile
        ret['bandpass_calfile'] = self.system_bandpass_calfile
        ret['atten_gain_calfile']=self.atten_gain_calfile
        ret['system_bandpass'] = self.system_bandpass
        ret['antenna_bandpass'] = self.antenna_bandpass
        ret['atten_gain_map'] = self.atten_gain_map
        return ret 

    def adc_selfcal(self):
        if self.adc_type=='iadc':
            corr.iadc.configure(self.fpga,0,mode='inter_I',cal='new',clk_speed=self.bandwidth/1000000)
        elif self.adc_type=='katadc':
            corr.katadc.set_interleaved(self.fpga,0,'I')
            time.sleep(0.1)
            corr.katadc.cal_now(self.fpga,0)
        else:
            return
        
    def fft_shift_set(self,fft_shift_schedule=-1):
        """Sets the FFT shift schedule (divide-by-two) on each FFT stage. 
            Input is an integer representing a binary bitmask for shifting.
            If not specified as a parameter to this function (or a negative value is supplied), program the default level."""
        if fft_shift_schedule<0: fft_shift_schedule=self.fft_shift
        self.fpga.write_int('fft_shift',fft_shift_schedule)
        self.fft_shift=fft_shift_schedule
        self.fft_scale=2**(cal.bitcnt(fft_shift_schedule))
        self.logger.info("Set FFT shift to %8x (scaling down by %i)."%(fft_shift_schedule,self.fft_scale))

    def fft_shift_get(self):
        """Fetches the current FFT shifting schedule from the hardware."""
        self.fft_shift=self.fpga.read_uint('fft_shift')
        self.fft_scale=2**(cal.bitcnt(self.fft_shift))
        return self.fft_shift 
#        return self.fft_scale

    def ctrl_get(self):
        """Reads and decodes the values from the control register."""
        value = self.fpga.read_uint('control')
        return {'mrst':bool(value&(1<<0)),
                'cnt_rst':bool(value&(1<<1)),
                'clr_status':bool(value&(1<<3)),
                'adc_protect_disable':bool(value&(1<<13)),
                'flasher_en':bool(value&(1<<12)),
                'raw':value,
                }

    def ctrl_set(self,**kwargs):
         """Sets bits of all the Fengine control registers. Keeps any previous state.
             \nPossible boolean kwargs:
             \n\t adc_protect_disable 
             \n\t flasher_en
             \n\t clr_status
             \n\t mrst
             \n\t cnt_rst"""

         key_bit_lookup={
             'adc_protect_disable':   13,
             'flasher_en':   12,
             'clr_status':   3,
             'cnt_rst':      1,
             'mrst':         0,
             }
         value = self.ctrl_get()['raw']
         run_cnt=0
         run_cnt_target=1
         while run_cnt < run_cnt_target:
             for key in kwargs:
                 if (kwargs[key] == 'toggle') and (run_cnt==0):
                     value = value ^ (1<<(key_bit_lookup[key]))
                 elif (kwargs[key] == 'pulse'):
                     run_cnt_target = 3
                     if run_cnt == 0: value = value & ~(1<<(key_bit_lookup[key]))
                     elif run_cnt == 1: value = value | (1<<(key_bit_lookup[key]))
                     elif run_cnt == 2: value = value & ~(1<<(key_bit_lookup[key]))
                 elif kwargs[key] == True:
                     value = value | (1<<(key_bit_lookup[key]))
                 elif kwargs[key] == False:
                     value = value & ~(1<<(key_bit_lookup[key]))
                 else:
                     raise RuntimeError("Sorry, you must specify True, False, 'toggle' or 'pulse' for %s."%key)
             self.fpga.write_int('control', value)
             run_cnt = run_cnt +1

    def rf_gain_set(self,gain=None,print_progress=False):
        """Enables the RF switch and configures the RF attenuators on KATADC boards. \n
        \t KATADC's valid range is -11.5 to 20dB. \n"""
        if gain==None:
            self.auto_gain(print_progress=print_progress)
        else:
            if gain < self.config['rf_gain_range'][0] or gain < self.config['rf_gain_range'][1]:
                 raise RuntimeError("Invalid gain setting of %i. Valid range for your ADC is %f to %fdB."%(gain,self.config['rf_gain_range'][0],self.config['rf_gain_range'][1]))
            if self.adc_type == 'katadc':
                #RF switch is in MSb.
                self.fpga.write_int('adc_ctrl0',(1<<31)+int((20-gain)*2))
            elif self.adc_type == 'iadc':
                self.fpga.write_int('adc_ctrl0',(1<<31)+int((0-gain)*2))
                #print 'Set RF gain register to %x'%int((0-gain)*2)
            elif self.adc_type == 'adc1x1800-10':
                self.fpga.write_int('adc_ctrl0',(1<<31)+int((0-gain)*2))
            else: raise RuntimeError("Sorry, your ADC type is not supported.")

    def rf_status_get(self):
        """Grabs the current value of the RF attenuators and RF switch state. returns (enabled,gain in dB)."""
        if self.adc_type == 'katadc':
            value = self.fpga.read_uint('adc_ctrl0')
            self.rf_gain=20.0-(value&0x3f)*0.5
            return (bool(value&(1<<31)),self.rf_gain)
        elif self.adc_type == 'iadc':
            value = self.fpga.read_uint('adc_ctrl0')
            self.rf_gain=0.0-(value&0x3f)*0.5
            return (bool(value&(1<<31)),self.rf_gain)
        elif self.adc_type == 'adc1x1800-10':
            value = self.fpga.read_uint('adc_ctrl0')
            self.rf_gain=0.0-(value&0x3f)*0.5
            return (bool(value&(1<<31)),self.rf_gain)
        else: raise RuntimeError("Sorry, your ADC type is not supported.")

    def adc_amplitudes_get(self):
        """Gets the ADC RMS amplitudes."""
        #TODO: CHECK THESE RETURNS!
        if self.adc_type == 'katadc':
            adc_bits=8
        elif self.adc_type == 'iadc':
            adc_bits=8
        elif self.adc_type == 'adc1x1800-10':
            adc_bits=8
        rv = {}
        rv['adc_raw']=self.fpga.read_uint('adc_sum_sq0')
        rv['adc_rms_raw']=numpy.sqrt(rv['adc_raw']/float(self.adc_levels_acc_len))
        rv['adc_rms_mv']=rv['adc_rms_raw']*self.config['adc_v_scale_factor']*1000
        rv['adc_dbm']=ratty1.cal.v_to_dbm(rv['adc_rms_mv']/1000.)
        #backout fe gain
        rv['input_dbm']=rv['adc_dbm']-self.config['fe_gain']
        #backout variable attenuation
        rv['input_dbm']-=self.atten_gain_map[self.rf_status_get()[1]]
        rv['input_rms_mv']=ratty1.cal.dbm_to_v(rv['input_dbm']*1000)
        return rv

    def status_get(self):
        """Reads and decodes the status register. Resets any error flags after reading."""
        rv={}
        value = self.fpga.read_uint('status')
        self.ctrl_set(clr_status='pulse')
        return {
                'adc_shutdown':bool(value&(1<<4)),
                'adc_overrange':bool(value&(1<<2)),
                'fft_overrange':bool(value&(1<<1))
                }

    def acc_time_set(self,acc_time=None):
        """Set the accumulation length in seconds. If not specified, use the default from the config file."""
        if acc_time >0:
            self.n_accs = int(acc_time * float(self.bandwidth)/self.n_chans)
        else:
            self.n_accs = self.config['n_accs']
            acc_time = self.config['acc_period']
        self.logger.info("Setting accumulation time to %2.2f seconds (%i accumulations)."%(acc_time,self.n_accs))
        self.fpga.write_int('acc_len',self.n_accs)
        self.ctrl_set(mrst='pulse')

    def acc_time_get(self):
        """Set the accumulation length in seconds"""
        self.n_accs = self.fpga.read_uint('acc_len')
        self.acc_time=self.n_accs*self.n_chans/float(self.bandwidth)
        self.logger.info("Accumulation time is %2.2f seconds (%i accumulations)."%(self.acc_time,self.n_accs))
        return self.acc_time,self.n_accs

    def get_adc_snapshot(self,trig_level=-1):
        if self.config['adc_n_bits'] != 8:
            self.logger.error("This function is only designed to work with 8 bit ADCs!")
            raise RuntimeError("This function is only designed to work with 8 bit ADCs!")
        if trig_level>0: 
            self.fpga.write_int('trig_level',trig_level)
            circ_capture=True
        else:
            self.fpga.write_int('trig_level',0)
            circ_capture=False

        return numpy.fromstring(self.fpga.snapshot_get('snap_adc',man_valid=True,man_trig=True,circular_capture=circ_capture,wait_period=-1)['data'],dtype=numpy.int8)

    def adc_temp_get(self):
        if self.adc_type== 'katadc':
            return corr.katadc.get_adc_temp(self.fpga,0)
        else:
            return -1

    def ambient_temp_get(self):
        if self.adc_type== 'katadc':
            return corr.katadc.get_ambient_temp(self.fpga,0)
        else:
            return -1


#    def get_adc_snapshots(self,ant_strs,trig_level=-1,sync_to_pps=True):
#        """Retrieves raw ADC samples from the specified antennas. Optionally capture the data at the same time. Optionally set  a trigger level."""
#        return corr.snap.get_adc_snapshots(self,ant_strs,trig_level=trig_level,sync_to_pps=sync_to_pps)


def ByteToHex( byteStr ):
    """
    Convert a byte string to it's hex string representation e.g. for output.
    """
    
    # Uses list comprehension which is a fractionally faster implementation than
    # the alternative, more readable, implementation below
    #   
    #    hex = []
    #    for aChar in byteStr:
    #        hex.append( "%02X " % ord( aChar ) )
    #
    #    return ''.join( hex ).strip()        

    return ''.join( [ "%02X " % ord( x ) for x in byteStr ] ).strip()

