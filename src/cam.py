#!/usr/bin/env python
'''
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/

Hard-coded for 32bit unsigned numbers.
\nAuthor: Jason Manley, Feb 2011.
'''

import corr,time,numpy,struct,sys,logging,rfi_sys,cal,iniparse, os

front_led_layout=['adc_clip','adc_shutdown','fft_overflow','quantiser_overflow','new_accumulation','sync','NA','NA']

#roach='192.168.64.112' Edit by Chris
#mode_params={'hr': {'bitstream':'r_spec_1ghz_16k_r106_2011_Feb_24_1810.bof',
# mode_params={
#             #'hr': {'bitstream':'r_spec_1ghz_16k_iadc_r106_2011_Mar_10_1724.bof',
#             'hr': {'bitstream':'r_spec_1ghz_16k_iadc_r107_2011_Mar_14_0850.bof',
#                     'n_chans':16384,
#                     'n_par_streams':4,
#                     'bandwidth':898000000,
#                     'desired_rf_level':-25,
#                     'adc_type':'iadc',
#                     'spectrum_bits':64,
#                     'fft_shift':0b001111111111100},
#             'hr_900': {'bitstream':'r_spec_1ghz_16k_iadc_r107_2011_Mar_14_0850.bof',
#                     'n_chans':16384,
#                     'n_par_streams':4,
#                     'bandwidth':900000000,
#                     'adc_type':'iadc',
#                     'desired_rf_level':-25,
#                     'spectrum_bits':64,
#                     'fft_shift':16383},
#             #'hr_kadc': {'bitstream':'r_spec_1ghz_16k_kadc_r108_2011_Jul_26_1810.bof',
#             'hr_kadc': {'bitstream':'r_spec_1ghz_16k_kadc_r108_2011_Nov_09_1541.bof',
#                     'n_chans':16384,
#                     'n_par_streams':4,
#                     'bandwidth':800000000,
#                     'adc_type':'katadc',
#                     'desired_rf_level':-25,
#                     'spectrum_bits':64,
#                     'fft_shift':16383},
#              #'lr': {'bitstream':'r_spec_1ghz_1k_r108lr_2011_Feb_28_1051.bof',
#              'lr': {'bitstream':'r_spec_1ghz_1k_iadc_r108lr_2011_Feb_28_1655.bof',
#                     'n_chans':1024,
#                     'desired_rf_level':-25,
#                     'n_par_streams':4,
#                     'adc_type':'iadc',
#                     'spectrum_bits':32,
#                     'bandwidth':900000000,
#                     'fft_shift':1023},
#            }
#katcp_port=7147 Edit by Chris

class spec:
    def __init__(self, config_file, log_handler=None, log_level=logging.INFO):
        #-------------------------------------Code By Chris--------------------------------------------
        self.config_file = config_file.strip()
        try:
            self.sys_config = iniparse.INIConfig(open(self.config_file, 'rb'))      #load config file
        except IOError as e:
            print "Error opening the config file : ",
            print e
            exit()
        roach = self.sys_config['connection']['roach_ip'].strip()                     #load Roach IP
        katcp_port = int(self.sys_config['connection']['katcp_port'])                  #load Roach port
        #----------------------------------End Code By Chris-------------------------------------------
        if log_handler == None: log_handler=corr.log_handlers.DebugLogHandler(100)
        self.lh = log_handler
        self.logger = logging.getLogger('RFIsys')
        self.fpga=corr.katcp_wrapper.FpgaClient(roach,katcp_port,timeout=10,logger=self.logger)
        self.logger.setLevel(log_level)
        self.logger.addHandler(self.lh)
        time.sleep(1)
        try:
            self.fpga.ping()
            self.logger.info('KATCP connection ok.')
        except:
            self.logger.error('KATCP connection failure. Connection to ROACH failed.')
            print('KATCP connection failure.')
            raise RuntimeError("Connection to FPGA board failed.")
        #--------------------------------Edited by Chris-------------------------------------------------
        #self.mode = self.sys_config['digital_system_parameters']['mode'].strip()
        self.n_chans = int(self.sys_config['digital_system_parameters']['n_chans'])
        self.bandwidth = int(self.sys_config['digital_system_parameters']['bandwidth'])
        self.n_par_streams = int(self.sys_config['digital_system_parameters']['n_par_streams'])
        self.bitstream = self.sys_config['digital_system_parameters']['bitstream']
        self.fft_shift = int(self.sys_config['digital_system_parameters']['fft_shift'])
        self.adc_type = self.sys_config['digital_system_parameters']['adc_type']
        self.desired_rf_level = int(self.sys_config['digital_system_parameters']['desired_rf_level'])
        self.spectrum_bits = int(self.sys_config['digital_system_parameters']['spectrum_bits'])
        if self.adc_type== 'katadc':
            self.fpga_clk=self.bandwidth/4
            self.sample_clk=self.bandwidth*2
            self.rf_gain_range=(-11.5,20)
        elif self.adc_type== 'iadc':
            self.fpga_clk=self.bandwidth/4
            self.sample_clk=self.bandwidth*2
            self.rf_gain_range=(-31.5,0)
        self.chan_width=numpy.float(self.bandwidth)/self.n_chans 
        self.freqs=numpy.arange(self.n_chans)*float(self.bandwidth)/self.n_chans #channel center freqs in Hz
        #self.freqs=numpy.arange(self.n_chans)*float(self.bandwidth)/self.n_chans/1.e6 + float(self.bandwidth)/self.n_chans/2.e6 #channel start freqs in MHz (lower freq half-power boundary)


        #-----------------------------End Edited By Chris -----------------------------------------------

    def initialise(self,rf_gain=-10,acc_time=1,fft_shift=0xffffffff):
        """Initialises the system to defaults."""
        self.fpga.progdev(self.bitstream)
        self.fft_shift_set(fft_shift)
        self.rf_gain_set(rf_gain)
        self.acc_time_set(acc_time)
        self.ctrl_set(flasher_en=False,cnt_rst='pulse',clr_status='pulse')
        #self.ctrl_set(flasher_en=True,cnt_rst='pulse',clr_status='pulse')

    def clk_check(self):
        """Performs a clock check and returns an estimate of the FPGA's clock frequency."""
        est_rate=round(self.fpga.est_brd_clk())
        if est_rate>(self.fpga_clk/1e6 +1) or est_rate<(self.fpga_clk/1e6 -1):
            self.logger.error('FPGA clock rate is %i MHz where we expect it to be %i MHz.'%(est_rate,self.fpga_clk/1e6))
            raise RuntimeError('FPGA clock rate is %i MHz where we expect it to be %i MHz.'%(est_rate,self.fpga_clk/1e6))
        return est_rate

    def adc_selfcal(self):
        if self.adc_type=='iadc':
            corr.iadc.configure(self.fpga,0,mode='inter_I',cal='new',clk_speed=self.bandwidth/1000000)
        elif self.adc_type=='katadc':
            corr.katadc.set_interleaved(self.fpga,0,'I')
            time.sleep(0.1)
            corr.katadc.cal_now(self.fpga,0)
        
    def fft_shift_set(self,fft_shift_schedule=-1):
        """Sets the FFT shift schedule (divide-by-two) on each FFT stage. 
            Input is an integer representing a binary bitmask for shifting.
            If not specified as a parameter to this function (or a negative value is supplied), program the default level."""
        import cal
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

    def rf_gain_set(self,gain=None):
        """Enables the RF switch and configures the RF attenuators on KATADC boards. \n
        \t KATADC's valid range is -11.5 to 20dB. \n"""
        self.rf_gain=gain
        if self.adc_type == 'katadc':
            #RF switch is in MSb.
            if gain > 20 or gain < -11.5:
                 raise RuntimeError("Invalid gain setting of %i. Valid range for KATADC is -11.5 to +20dB.")
            self.fpga.write_int('adc_ctrl0',(1<<31)+int((20-gain)*2))
        elif self.adc_type == 'iadc':
            if gain > 0 or gain < -31.5:
                 raise RuntimeError("Invalid gain setting of %i. Valid range for RFI frontend is -31.5 to 0dB.")
            self.fpga.write_int('adc_ctrl0',(1<<31)+int((0-gain)*2))
            #print 'Set RF gain register to %x'%int((0-gain)*2)
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
        else: raise RuntimeError("Sorry, your ADC type is not supported.")

    def adc_amplitudes_get(self):
        """Gets the ADC RMS amplitudes."""
        adc_levels_acc_len=65536
        if self.adc_type == 'katadc':
            adc_bits=8
        elif self.adc_type == 'iadc':
            adc_bits=8
        rv = {}
        rv['adc_raw']=self.fpga.read_uint('adc_sum_sq0')
        rv['adc_rms_raw']=numpy.sqrt(rv['adc_raw']/float(adc_levels_acc_len))
        rv['adc_rms_mv']=rv['adc_rms_raw']*cal.get_adc_cnt_mv_scale_factor()
        rv['adc_dbm']=cal.v_to_dbm(rv['adc_rms_mv']/1000.)
        rv['input_rms_mv']=rv['adc_rms_raw']*cal.get_adc_cnt_mv_scale_factor(self.rf_status_get()[1])
        rv['input_dbm']=cal.v_to_dbm(rv['input_rms_mv']/1000.)
        return rv

    def status_get(self):
        """Reads and decodes the status register. Resets any error flags after reading."""
        rv={}
        value = self.fpga.read_uint('status')
        self.ctrl_set(clr_status='pulse')
        return {
                'adc_bad':bool(value&(1<<4)),
                'adc_overrange':bool(value&(1<<2)),
                'fft_overrange':bool(value&(1<<1))
                }

    def acc_time_set(self,acc_time=1):
        """Set the accumulation length in seconds"""
        self.n_accs = int(acc_time * float(self.bandwidth)/self.n_chans)
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

#----------------------------------------Code By Chris------------------------------------------------




#---------------------------------------End Code By Chris---------------------------------------------

