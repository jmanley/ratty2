# -*- coding: utf-8 -*-
import numpy,scipy,scipy.interpolate,iniparse,ratty1


#smoothing functions from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/

c=299792458. #speed of light in m/s
#cal_file_path = "/etc/rfi_sys/cal_files/"; #For when you have everything working and ready to install with distutils
cal_file_path = "/etc/ratty1/cal_files/"; #For development
def cal_files(filename):
    return "%s%s"%(cal_file_path, filename)

def smoothList(list,strippedXs=False,degree=10):  
     if strippedXs==True: return Xs[0:-(len(list)-(len(list)-degree+1))]  
     smoothed=[0]*(len(list)-degree+1)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(list[i:i+degree])/float(degree)  
     return smoothed  

def smoothListTriangle(list,strippedXs=False,degree=5):  
     weight=[]  
     window=degree*2-1  
     smoothed=[0.0]*(len(list)-window)  
     for x in range(1,2*degree):weight.append(degree-abs(degree-x))  
     w=numpy.array(weight)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(numpy.array(list[i:i+window])*w)/float(sum(w))  
     return smoothed  

def smoothListGaussian(list,strippedXs=False,degree=5):  
     window=degree*2-1  
     weight=numpy.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(numpy.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=numpy.array(weightGauss)*weight  
     smoothed=[0.0]*(len(list)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(numpy.array(list[i:i+window])*weight)/sum(weight)  
     return smoothed  

def dmw_per_sq_m_to_dbuv(dbmw):
    # from http://www.ahsystems.com/notes/RFconversions.php: dBmW/m2 = dBmV/m - 115.8 
    return dbmw + 115.8

def dbuv_to_dmw_per_sq_m(dbuv):
    # from http://www.ahsystems.com/notes/RFconversions.php: dBmW/m2 = dBmV/m - 115.8 
    return dbuv - 115.8

def dbm_to_dbuv(dbm):
    return dbm+106.98

def dbuv_to_dbm(dbuv):
    return dbm-106.98 

def v_to_dbuv(v):
    return 20*numpy.log10(v*1e6)

def dbuv_to_v(dbuv):
    return (10**(dbuv/20.))/1e6

def dbm_to_v(dbm):
    return numpy.sqrt(10**(dbm/10.)/1000*50)

def v_to_dbm(v):
    return 10*numpy.log10(v*v/50.*1000)

def bitcnt(val):
    '''Counts the number of set bits in the binary value.'''
    ret_val=0
    shift_val=val
    while shift_val>=1:
        if shift_val&1: ret_val +=1
        shift_val = shift_val>>1
    return ret_val

def polyfit(freqs,gains,degree=9):
    """Just calls numpy.polyfit. Mostly here as a reminder."""
    return numpy.polyfit(freqs,gain,deg=degree)

def af_from_gain(freqs,gains):
        """Calculate the antenna factor (in dB/m) from a list of frequencies (in Hz) and gains (in dBi).
            There are a number of assumptions made for this to be valid:
             1) Far-field only (plane wave excitation).
             2) 50ohm system.
             3) antenna is polarisation matched.
             4) effects of impedance mismatch are included."""
        #From Howard's email:
        #The antenna factors are derived from gain by taking 9.73/(lambda(sqrt(Gain))  - note the gain here is the non-dB gain. It is worth noting that in dB’s the conversion is 19.8 – 20log10(lambda) – 10 log(Gain)
        #avoid the divide by zero error:
        if freqs[0]==0:
            freqs[0]=numpy.finfo(numpy.float).eps
        return 19.8 - 20*numpy.log10(c/freqs) - gains

def gain_from_af(freqs,afs):
        """Calculate the gain (in dBi) from the Antenna Factor (in dB/m)."""
        return 19.8 - 20*numpy.log10(c/freqs) - afs

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

class cal:

    def __init__(self, **kwargs):
        """Either specify a config_file, or else specify
            n_chans,bandwidth, n_par_streams, bitstream, fft_shfit, adc_type, desired_rf_level, spectrum_bits, antenna_bandpass, atten_gain, system_bandpass, fe_gain and acc_period"""
        self.config={}
        if kwargs.has_key('config_file'):
            self.config = ratty1.conf.rattyconf(kwargs['config_file'])

        for key in kwargs:
            self.config[key]=kwargs[key]

        self.system_bandpass = self.config['system_bandpass']
        self.antenna_bandpass = self.config['antenna_bandpass']
        self.atten_gain_map = self.config['atten_gain_map']
        self.ant_factor = af_from_gain(self.config['freqs'], self.config['antenna_bandpass'])
        self.config['ant_factor'] = af_from_gain(self.config['freqs'], self.config['antenna_bandpass'])

    def inter_adc_details(self, data):
        print 'DC offset 0: %f'%numpy.mean(data[0::2])
        print 'DC offset 1: %f'%numpy.mean(data[1::2])
        print 'Max 0: %f'%numpy.max(data[0::2])
        print 'Max 1: %f'%numpy.max(data[1::2])
        #print 'Phase difference estimate: %f'%numpy.acos(2*numpy.mean(data[0::2]*data[1::2]))

    def plot_bandshape(self,freqs):
        import pylab
        pylab.plot(self.config['freqs']/1e6,self.config['system_bandpass'])
        pylab.title('Bandpass calibration profile')
        pylab.xlabel('Frequency (Hz)')
        pylab.ylabel('Relative response (dB)')

    def plot_atten_gain_map(self):
        import pylab
        inputs=self.config['atten_gain_map'].keys()
        inputs.sort()
        pylab.plot(inputs,[self.config['atten_gain_map'][k] for k in inputs])
        pylab.title('RF attenuator mapping')
        pylab.xlabel('Requested value (dB)')
        pylab.ylabel('Actual value (dB)')
    
    def get_interpolated_gains(self,fileName):
        """Retrieves antenna gain mapping from /etc/rfi_sys/cal_files/ant.csv file and interpolates data to return values at 'freqs'."""
        cal_freqs,cal_gains=get_gains_from_csv(cal_files(fileName + '.csv'))
        inter_freqs=scipy.interpolate.interp1d(cal_freqs,cal_gains,kind='linear')
        #print ('self.config["freqs"] = %s'%self.config['freqs'])
        return inter_freqs(self.config['freqs'])

    def plot_ant_gain(self):
        """Plots the antenna gain as read from the antenna calibration CSV file specified by the system config file."""
        import pylab
        pylab.plot(self.config['freqs']/1e6,self.config['antenna_bandpass'])
        pylab.title('Antenna gain %s'%self.config['antenna_bandpass_calfile'])
        pylab.xlabel('Frequency (MHz)')
        pylab.ylabel('Relative response (dBi)')

    def plot_ant_factor(self):
        """Plots the antenna factor over the given frequencies as calculated from the specified antenna CSV file."""
        import pylab
        pylab.plot(self.freqs/1e6,self.ant_factor)
        pylab.title('Antenna factor as a function of frequency (%s)'%self.config['antenna_bandpass_calfile'])
        pylab.xlabel('Frequency (MHz)')
        pylab.ylabel('Antenna factor (dBuV/m)')

    def freq_to_chan(self,frequency,n_chans=None):
        """Returns the channel number where a given frequency is to be found. Frequency is in Hz."""
        if frequency<0: 
            frequency=self.config['bandwidth']+frequency
            #print 'you want',frequency
        if frequency>self.config['bandwidth']: raise RuntimeError("that frequency is too high.")
        if n_chans==None:
            n_chans=self.config['n_chans']
        return round(float(frequency)/self.config['bandwidth']*n_chans)%n_chans

    def get_input_adc_v_scale_factor(self,rf_gain):
        """Provide the calibration factor to get from an ADC input voltage to the actual frontend input voltage. Does not perform any frequency-dependent calibration."""
        return 1/(10**((self.atten_gain_map[rf_gain]+self.config['fe_gain'])/20.))

    def calibrate_adc_snapshot(self,raw_data,rf_gain,n_chans=256):
        """Calibrates a raw ADC count timedomain snapshot. Returns ADC samples in V, ADC spectrum in dBm, input spectrum in dBm and input spectrum of n_chans in dBm."""
        ret={}
        ret['adc_raw']=raw_data
        ret['adc_v']=raw_data*self.config['adc_v_scale_factor']
        ret['input_v']=ret['adc_v']*self.get_input_adc_v_scale_factor(rf_gain) 
        #Calculate the spectrum:
        n_accs=len(raw_data)/n_chans/2
        window=numpy.hamming(n_chans*2)
        spectrum=numpy.zeros(n_chans)
        freqs=numpy.arange(n_chans)*float(self.config['bandwidth'])/n_chans
        ret['freqs']=freqs
        for acc in range(n_accs):
            spectrum += numpy.abs((numpy.fft.rfft(ret['adc_v'][n_chans*2*acc:n_chans*2*(acc+1)]*window)[0:n_chans])) 
        ret['adc_spectrum_dbm']  = 20*numpy.log10(spectrum/n_accs/n_chans*6.14)
        #get system_bandpass:
        if self.config['system_bandpass_calfile'] != 'none':
            cal_freqs,cal_gains=get_gains_from_csv(cal_files(self.config['system_bandpass_calfile'] + '.csv'))
            inter_freqs=scipy.interpolate.interp1d(cal_freqs,cal_gains,kind='linear')
            bp=inter_freqs(freqs)
            ret['system_bandpass']=bp
        else: bp=0
        ret['input_spectrum_dbm']=ret['adc_spectrum_dbm']-bp-self.config['fe_gain']-self.atten_gain_map[rf_gain]

        if self.config['antenna_bandpass_calfile'] != 'none':
            #get antenna factor:
            cal_freqs,cal_gains=get_gains_from_csv(cal_files(self.config['antenna_bandpass_calfile'] + '.csv'))
            inter_freqs=scipy.interpolate.interp1d(cal_freqs,cal_gains,kind='linear')
            af=af_from_gain(freqs,inter_freqs(freqs))
            ret['ant_factor']=af
            ret['antenna_bandpass']=inter_freqs(freqs)
            ret['input_spectrum_dbuv'] = dbm_to_dbuv(ret['input_spectrum_dbm']) + af 
        return ret

     
#    def get_calibrated_spectrum_from_raw_snapshot(self,adcdata,atten,n_chans=512):
#        """Will correct for RF frontend attenuator gains, bandshape and optionally antenna response.\n 
#            Units are dBm unless an antenna was specified in your config file, in which case units are dBuV/m.\n"""
#        n_accs=len(adcdata)/n_chans/2
#        freqs=numpy.arange(n_chans)*float(self.config['bandwidth'])/n_chans #channel center freqs in Hz. #linspace(0,float(bandwidth),n_chans) returns incorrect numbers
#        window=numpy.hamming(n_chans*2)
#        spectrum=numpy.zeros(n_chans)
#        fe_gain_db=self.fe_gain+self.atten_gain_map(atten)
#        adc_data_v=self.config['adc_v_scale_factor'] * adcdata * (10**(fe_gain_db/20.))
#        for acc in range(n_accs):
#            spectrum += numpy.abs((numpy.fft.rfft(adc_data_v[self.config['n_chans']*2*acc:self.config['n_chans']*2*(acc+1)]*window)[0:self.config['n_chans']])**2)
#            #print (numpy.fft.rfft(adc_data_dbm[n_chans*2*acc:n_chans*2*(acc+1)]*window)[0:n_chans])
#        spectrum  = 10*numpy.log10(spectrum/n_accs/self.config['n_chans']) #now in dBV
#        spectrum -= 13.034
#        spectrum -= bandshape
#        if self.config['antenna_bandpass_calfile'] != 'none':
#            spectrum = dbm_to_dbuv(spectrum)
#            spectrum += self.ant_factor 
#        return freqs,spectrum

    def get_calibrated_spectrum(self,data, desired_rf_gain):
        '''Returns a calibrated spectrum from a raw hardware spectral dump.
            Units are dBm unless an antenna was specified in your config file, in which case units are dBuV/m.\n
            Performs bandpass correction, fft_scaling adjustment, overall gain compensation, backs out number of accumulations, RF frontend gain etc.\n
        '''
        data_return=numpy.array(data)
        data_return /= float(self.config['n_accs'])
        data_return *= bitcnt(self.config['fft_shift'])
        data_return *= self.config['adc_v_scale_factor']
        #data_return /= self.chan_width
        data_return  = 10*numpy.log10(data_return)
        data_return -= self.atten_gain_map[desired_rf_gain]
        data_return -= self.config['fe_gain'] 
        data_return -= 67. #overall system/algorithm gain
        data_return -= self.system_bandpass
        if self.config['antenna_bandpass_calfile'] != 'none':
            data_return = dbm_to_dbuv(data_return)
            data_return += self.ant_factor 
        return data_return

