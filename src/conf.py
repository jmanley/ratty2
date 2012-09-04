import iniparse, exceptions, socket, struct, numpy, os,scipy
"""
Library for parsing CASPER correlator configuration files

Author: Jason Manley
"""
"""
Revs:
2012-06-14  JRM Initial release 
"""
LISTDELIMIT = ','
PORTDELIMIT = ':'

class rattyconf:    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_file_name = os.path.split(self.config_file)[1]
        self.cp = iniparse.INIConfig(open(self.config_file, 'rb'))
        self.config = dict()
        self.read_common()

    def __getitem__(self, item):
        #if item == 'sync_time':
        #    fp = open(VAR_RUN + '/' + item + '.' + self.config_file_name, 'r')
        #    val = float(fp.readline())
        #    fp.close()
        #    return val
        #elif item == 'antenna_mapping':
        #    fp = open(VAR_RUN + '/' + item + '.' + self.config_file_name, 'r')
        #    val = (fp.readline()).split(LISTDELIMIT)
        #    fp.close()
        #    return val
        #else:
        return self.config[item]

    def __setitem__(self,item,value):
        self.config[item]=value

    def file_exists(self):
        try:
            #f = open(self.config_file)
            f = open(self.config_file, 'r')
        except IOError:
            exists = False
            raise RuntimeError('Error opening config file at %s.'%self.config_file)
        else:
            exists = True
            f.close()
      #  # check for runtime files and create if necessary:
      #  if not os.path.exists(VAR_RUN):
      #      os.mkdir(VAR_RUN)
      #      #os.chmod(VAR_RUN,0o777)
      #  for item in ['antenna_mapping', 'sync_time']:
      #      if not os.path.exists(VAR_RUN + '/' + item + '.' + self.config_file_name):
      #          f = open(VAR_RUN + '/' + item + '.' + self.config_file_name, 'w')
      #          f.write(chr(0))
      #          f.close()
      #          #os.chmod(VAR_RUN+'/' + item,0o777)
        return exists


    def read_common(self):
        if not self.file_exists():
            raise RuntimeError('Error opening config file or runtime variables.')
        self.config['front_led_layout']=['adc_clip','adc_shutdown','fft_overflow','quantiser_overflow','new_accumulation','sync','NA','NA']
        self.config['snap_name'] = 'snap_adc'
        self.config['spectrum_bram_out_prefix'] = 'store'
        
        #self.mode = self.sys_config['digital_system_parameters']['mode'].strip()
        self.read_int('digital_system_parameters','n_chans')
        self.read_int('digital_system_parameters','bandwidth')
        self.read_int('digital_system_parameters','n_par_streams')
        self.read_str('digital_system_parameters','bitstream')
        self.read_int('digital_system_parameters','fft_shift')
        self.read_str('digital_system_parameters','adc_type')
        self.read_float('digital_system_parameters','desired_rf_level')
        self.read_int('digital_system_parameters','spectrum_bits')
        self.read_float('digital_system_parameters','acc_period')
        self.read_float('digital_system_parameters','adc_levels_acc_len')

        self.read_int('connection','katcp_port')
        self.config['roach_ip']=struct.unpack('>I',socket.inet_aton(self.get_line('connection','roach_ip')))[0]
        self.config['roach_ip_str']=self.get_line('connection','roach_ip')
        
        self.read_float('analogue_frontend','fe_gain')
        self.read_float('analogue_frontend','ignore_low_freq')
        self.read_float('analogue_frontend','ignore_high_freq')
        self.config['antenna_bandpass_calfile']=self.get_line('analogue_frontend','antenna_bandpass')
        self.config['system_bandpass_calfile']=self.get_line('analogue_frontend','system_bandpass')
        self.config['atten_gain_calfile']=self.get_line('analogue_frontend','atten_gain_map')
        if self.get_line('analogue_frontend','rf_gain').strip() == 'auto':
            self.config['rf_gain'] = None
        else:  
            self.read_float('analogue_frontend','rf_gain')

        self.read_str('digital_system_parameters','adc_type')
        if self.config['adc_type'] == 'katadc':
            self.config['sample_clk']=self.config['bandwidth']*2
            self.config['rf_gain_range']=(-11.5,20,0.5)
            self.config['adc_demux'] = 4
            self.config['adc_n_bits'] = 8
            self.config['adc_v_scale_factor']=1/260.4
            self.config['adc_low_level_warning']=-32
            self.config['adc_high_level_warning']=0
        elif self.config['adc_type'] == 'iadc':
            self.config['sample_clk']=self.config['bandwidth']*2
            self.config['rf_gain_range']=(-31.5,0,0.5)
            self.config['adc_demux'] = 4
            self.config['adc_n_bits'] = 8
            self.config['adc_v_scale_factor']=1/368.
            self.config['adc_low_level_warning']=-35
            self.config['adc_high_level_warning']=0
        elif self.config['adc_type'] == 'adc1x1800-10':
            self.config['sample_clk']=self.config['bandwidth']*2
            self.config['rf_gain_range']=(-31.5,0,0.5)
            self.config['adc_demux'] = 4
            self.config['adc_n_bits'] = 8
            self.config['adc_v_scale_factor']=1/368.
            self.config['adc_low_level_warning']=-35
            self.config['adc_high_level_warning']=0
        else:
            raise RuntimeError("adc_type not understood. expecting katadc or iadc.")

        self.config['chan_width']=numpy.float(self.config['bandwidth'])/self.config['n_chans']
        self.config['fpga_clk']=self.config['bandwidth']/self.config['adc_demux']
        
        self.config['freqs']=numpy.arange(self.config['n_chans'])*float(self.config['bandwidth'])/self.config['n_chans'] #channel center freqs in Hz        
        self.config['n_accs'] = int(self.config['acc_period'] * float(self.config['bandwidth'])/self.config['n_chans'])

        if (self.config['system_bandpass_calfile'] != 'none'):
            self.config['system_bandpass'] = self.get_interpolated_gains(self.config['system_bandpass_calfile'])
        else:
            self.config['system_bandpass'] = numpy.zeros(self.config['n_chans'])

        if (self.config['antenna_bandpass_calfile'] != 'none'):
            self.config['antenna_bandpass'] = self.get_interpolated_gains(self.config['antenna_bandpass_calfile'])
        else:
            self.config['antenna_bandpass'] = numpy.zeros(self.config['n_chans'])

        if (self.config['atten_gain_calfile'] != 'none'):
            self.config['atten_gain_map'] = getDictFromCSV(cal_files("%s.csv"%(self.config['atten_gain_calfile'])))
        else:
            self.config['atten_gain_map'] = {i:i for i in numpy.arange(self.config['rf_gain_range'][0],self.config['rf_gain_range'][1]+self.config['rf_gain_range'][2],self.config['rf_gain_range'][2])}

    def get_interpolated_gains(self,fileName):
        """Retrieves antenna gain mapping files and interpolates data to return values at 'freqs'."""
        cal_freqs,cal_gains=get_gains_from_csv(cal_files(fileName + '.csv'))
        inter_freqs=scipy.interpolate.interp1d(cal_freqs,cal_gains,kind='linear')
        #print ('self.config["freqs"] = %s'%self.config['freqs'])
        return inter_freqs(self.config['freqs'])

    def write(self,section,variable,value):
        print 'Writing to the config file. Mostly, this is a bad idea. Mostly. Doing nothing.'
        return
        self.config[variable] = value
        self.cp[section][variable] = str(value)
        fpw=open(self.config_file, 'w')
        print >>fpw,self.cp
        fpw.close()

    def write_var(self, filename, value):
        fp=open(VAR_RUN + '/' + filename + '.' + self.config_file_name, 'w')
        fp.write(value)
        fp.close()

    def write_var_list(self, filename, list_to_store):
        fp=open(VAR_RUN + '/' + filename + '.' + self.config_file_name, 'w')
        for v in list_to_store:
            fp.write(v + LISTDELIMIT)
        fp.close()

    def read_int(self,section,variable):
        self.config[variable]=int(self.cp[section][variable])

    def read_bool(self,section,variable):
        self.config[variable]=(self.cp[section][variable] != '0')

    def read_str(self,section,variable):
        self.config[variable]=self.cp[section][variable].strip()

    def get_line(self,section,variable):
        return str(self.cp[section][variable]).strip()

    def read_float(self,section,variable):
        self.config[variable]=float(self.cp[section][variable])



def getDictFromCSV(filename):
        import csv
        f = open(filename)
        f.readline()
        reader = csv.reader(f, delimiter=',')
        mydict = dict()
        for row in reader:
            mydict[float(row[0])] = float(row[1])
        return mydict

def get_gains_from_csv(filename):
    freqs=[]
    gains=[]
    more=True
    fp=open(filename,'r')
    import csv
    fc=csv.DictReader(fp)
    while(more):
        try:
            raw_line=fc.next()
            freqs.append(numpy.float(raw_line['freq_hz']))
            gains.append(numpy.float(raw_line['gain_db']))
        except:
            more=False
            break
    return freqs,gains


cal_file_path = "/etc/ratty1/cal_files/"; #For development
def cal_files(filename):
    return "%s%s"%(cal_file_path, filename)
