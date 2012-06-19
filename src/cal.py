# -*- coding: utf-8 -*-
import numpy,scipy,scipy.interpolate, iniparse


#smoothing functions from http://www.swharden.com/blog/2008-11-17-linear-data-smoothing-in-python/

c=299792458. #speed of light in m/s

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

def inter_adc_details(data):
    print 'DC offset 0: %f'%numpy.mean(data[0::2])
    print 'DC offset 1: %f'%numpy.mean(data[1::2])
    print 'Max 0: %f'%numpy.max(data[0::2])
    print 'Max 1: %f'%numpy.max(data[1::2])
    #print 'Phase difference estimate: %f'%numpy.acos(2*numpy.mean(data[0::2]*data[1::2]))


def run_gains_iadc(camobj):
    gain_cal=[]
    for g in numpy.arange(-31.5,0.5,0.5):
        camobj.rf_gain_set(g)
        time.sleep(0.2)
        gain_cal.append(r.adc_amplitudes_get()['adc_dbm'])
    return gain_cal

def run_gains_katadc(camobj):
    gain_cal=[]
    for g in numpy.arange(-11.5,20,0.5):
        camobj.rf_gain_set(g)
        time.sleep(0.2)
        gain_cal.append(r.adc_amplitudes_get()['adc_dbm'])
    return gain_cal


#def get_smoothed_bandshape(spectrum,poly_order=10):
#    import pylab
#    max_lev=numpy.max(spectrum)
#    norm_spec=numpy.array(spectrum)/float(max_lev)
#    norm_spec_db=10*numpy.log10(norm_spec)
#    pylab.plot(norm_spec_db)
#    norm_spec_db_smoothed=smoothListGaussian(norm_spec_db,degree=100)
#    norm_spec_db_smoothed_resized=numpy.ones(16384)
#    norm_spec_db_smoothed_resized[0:100]=norm_spec_db_smoothed[0]
#    norm_spec_db_smoothed_resized[100:100+len(norm_spec_db_smoothed)]=norm_spec_db_smoothed
#    norm_spec_db_smoothed_resized[100+len(norm_spec_db_smoothed):]=norm_spec_db_smoothed[-1]
#    pylab.plot(norm_spec_db_smoothed_resized)
#    poly=numpy.polyfit(range(len(norm_spec_db_smoothed)),norm_spec_db_smoothed,poly_order)
#    pylab.plot(numpy.polyval(poly,range(len(norm_spec_db_smoothed))))
#    pylab.show()
#    return poly,norm_spec_db_smoothed_resized

def plot_bandshape(freqs):
    import pylab
    pylab.plot(bandshape(freqs))
    pylab.title('Bandpass calibration profile')
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Relative response (dB)')

def plot_atten_gain_map():
    import pylab
    inputs=atten_gain_map.keys()
    inputs.sort()
    pylab.plot(inputs,[atten_gain_map[k] for k in inputs])
    pylab.title('RF attenuator mapping')
    pylab.xlabel('Requested value (dB)')
    pylab.ylabel('Actual value (dB)')

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

def af_from_gain(freqs,gains):
    """Calculate the antenna factor (in dB/m) from a list of frequencies (in Hz) and gains (in dBi).
        There are a number of assumptions made for this to be valid:
         1) Far-field only (plane wave excitation).
         2) 50ohm system.
         3) antenna is polarisation matched.
         4) effects of impedance mismatch are included."""
    #From Howard's email:
    #The antenna factors are derived from gain by taking 9.73/(lambda(sqrt(Gain))  - note the gain here is the non-dB gain. It is worth noting that in dB’s the conversion is 19.8 – 20log10(lambda) – 10 log(Gain)
    return 19.8 - 20*numpy.log10(c/freqs) - gains

def gain_from_af(freqs,afs):
    """Calculate the gain (in dBi) from the Antenna Factor (in dB/m)."""
    return 19.8 - 20*numpy.log10(c/freqs) - afs

def ant_gains(ant,freqs):
    """Retrieves antenna gain mapping from /etc/rfi_sys/cal_files/ant.csv file and interpolates data to return values at 'freqs'."""
    #---------------------------------------------------------Edit By Chris------------------------------------------------
    cal_freqs,cal_gains=get_gains_from_csv(cal_files(ant +'.csv'))
    #------------------------------------------------------End Edit By Chris----------------------------------------------
    inter_freqs=scipy.interpolate.interp1d(cal_freqs,cal_gains,kind='linear')
    return inter_freqs(freqs)

def plot_ant_gain(ant,freqs):
    """Plots the antenna gain as read from a CSV file specified as "ant"."""
    import pylab
    pylab.plot(freqs/1e6,ant_gains(ant,freqs))
    pylab.title('Antenna gain %s'%ant)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Relative response (dBi)')

def plot_ant_factor(ant,freqs):
    """Plots the antenna factor over the given frequencies as calculated from the specified antenna CSV file."""
    import pylab
    pylab.plot(freqs/1e6,af_from_gain(freqs,ant_gains(ant,freqs)))
    pylab.title('Antenna factor as a function of frequency (%s)'%ant)
    pylab.xlabel('Frequency (MHz)')
    pylab.ylabel('Antenna factor (dBuV/m)')


#--------------------------------------------------------------Edited By Chris----------------------------------------------------------
def bandshape(freqs):
    """Returns the system bandshape in dB, evaluated at 'freqs'."""
    import iniparse, os
    config_file = os.path.join('..', 'src', 'system_parameters')
    try:
        sys_config = iniparse.INIConfig(open(config_file, 'rb'))
        return ant_gains(sys_config['analogue_frontend']['system_bandpass'],freqs)
    except Exception as e:
        print e
        raise

#--------------------------------------------------------End Edit By Chris---------------------------------------------------------------



def polyfit(freqs,gains,degree=9):
    """Just calls numpy.polyfit. Mostly here as a reminder."""
    return numpy.polyfit(freqs,gain,deg=degree)

def get_adc_cnt_mv_scale_factor(atten_gain=None):
    """Calculate and return a scale factor for calibrating a raw ADC count to millivolts. Optional atten_gain in dB to map to input levels."""
    if atten_gain==None:
        return 3.93
    else:
        return 3.93/(10**((atten_gain_map[atten_gain]+fe_gain)/20.))

def freq_to_chan(frequency,bandwidth,n_chans):
    """Returns the channel number where a given frequency is to be found. Frequency is in Hz."""
    if frequency<0: 
        frequency=bandwidth+frequency
        #print 'you want',frequency
    if frequency>bandwidth: raise RuntimeError("that frequency is too high.")
    return round(float(frequency)/bandwidth*n_chans)%n_chans
 
def get_calibrated_spectrum_from_raw_snapshot(adcdata,atten,bandwidth,ant_factor=None,bandshape=None,n_chans=512):
    """Will correct for RF frontend attenuator gains, bandshape and optionally antenna response. Returns dBm unless antenna is specified, in which case returns dBuV/m."""
#TODO: TEST THIS
    n_accs=len(adcdata)/n_chans/2
    freqs=numpy.arange(n_chans)*float(bandwidth)/n_chans #channel center freqs in Hz. #linspace(0,float(bandwidth),n_chans) returns incorrect numbers
    window=numpy.hamming(n_chans*2)
    spectrum=numpy.zeros(n_chans)
    adc_data_v=get_adc_cnt_mv_scale_factor(atten_gain=atten)*adcdata/1000. #factors-in atten_gain_map and fe_gain
    for acc in range(n_accs):
        spectrum += numpy.abs((numpy.fft.rfft(adc_data_v[n_chans*2*acc:n_chans*2*(acc+1)]*window)[0:n_chans])**2)
        #print (numpy.fft.rfft(adc_data_dbm[n_chans*2*acc:n_chans*2*(acc+1)]*window)[0:n_chans])
    spectrum  = 10*numpy.log10(spectrum/n_accs/n_chans) #now in dBV
    spectrum -= 13.034
    if bandshape != None:
        spectrum -= bandshape
    if ant_factor != None: 
        spectrum = dbm_to_dbuv(spectrum)
        spectrum += ant_factor 
    return freqs,spectrum

def get_calibrated_spectrum(freqs,data,n_accs,fft_scale,rf_gain,ant_factor=None,bandshape=None):
    '''Returns a calibrated spectrum from a raw hardware spectral dump.
        Units are dBm unless an antenna is specified in which case units are dBuV/m.\n
        Performs bandpass correction, fft_scaling adjustment, overall gain compensation, backs out number of accumulations, RF frontend gain etc.\n
    '''
    #SQRT?
    data_return=numpy.array(data)
    data_return /= float(n_accs)
    data_return *= fft_scale
    #data_return /= self.chan_width
    data_return  = 10*numpy.log10(data_return)
    data_return -= atten_gain_map[rf_gain]
    data_return -= fe_gain 
    data_return -= 120. #overall system/algorithm gain
    if bandshape != None:
        data_return -= bandshape
    if ant_factor !=None: 
        data_return = dbm_to_dbuv(data_return)
        data_return += ant_factor 
    return data_return

def getDictFromCSV(filename):
    import csv
    f = open(filename)
    f.readline()
    reader = csv.reader(f, delimiter=',')
    mydict = dict()
    for row in reader:
        mydict[float(row[0])] = float(row[1])
    return mydict

def cal_files(filename):
    import os
    return os.path.join('..', 'cal_files', filename)

#front-end gain in dB.
fe_gain=20.0

#------------------------------------------------------Edited by Chris----------------------------------------------------------
atten_gain_map = getDictFromCSV(cal_files("%s.csv"%(iniparse.INIConfig(open("../src/system_parameters", 'rb'))['analogue_frontend']['atten_gain_map'])))
#--------------------------------------------------------End Edited By Chris----------------------------------------------
#atten_gain_map maps the selected attenuator level to actual system gain. 
# atten_gain_map={
# -11.50:-11.5,
# -11.00:-11.,vv
# -10.50:-10.5,
# -10.00:-10.,
# -9.50:-9.5,
# -9.00:-9.,
# -8.50:-8.5,
# -8.00:-8.,
# -7.50:-7.5,
# -7.00:-7.,
# -6.50:-6.5,
# -6.00:-6.,
# -5.50:-5.5,
# -5.00:-5.,
# -4.50:-4.5,
# -4.00:-4.,
# -3.50:-3.5,
# -3.00:-3.,
# -2.50:-2.5,
# -2.00:-2.,
# -1.50:-1.5,
# -1.00:-1.,
# -0.50:-0.5,
# 0    : 0.,
# 0.50:0.5,
# 1.    : 1.   ,
# 1.5   : 1.5  ,
# 2.    : 2.   ,
# 2.5   : 2.5  ,
# 3.    : 3.   ,
# 3.5   : 3.5  ,
# 4.    : 4.   ,
# 4.5   : 4.5  ,
# 5.    : 5.   ,
# 5.5   : 5.5  ,
# 6.    : 6.   ,
# 6.5   : 6.5  ,
# 7.    : 7.   ,
# 7.5   : 7.5  ,
# 8.    : 8.   ,
# 8.5   : 8.5  ,
# 9.    : 9.   ,
# 9.5   : 9.5  ,
# 10.   : 10.  ,
# 10.5  : 10.5 ,  
# 11.   : 11.  ,
# 11.5  : 11.5 ,
# 12.   : 12.  ,
# 12.5  : 12.5 ,
# 13.   : 13.  ,
# 13.5  : 13.5 ,
# 14.   : 14.  ,
# 14.5  : 14.5 ,
# 15.   : 15.  ,
# 15.5  : 15.5 ,
# 16.   : 16.  ,
# 16.5  : 16.5 ,
# 17.   : 17.  ,
# 17.5  : 17.5 ,
# 18.   : 18.  ,
# 18.5  : 18.5 ,
# 19.   : 19.  ,
# 19.5  : 19.5 ,
# 20.   : 20.  ,
# }   

atten_gain_map_iadc={
-31.50:-31.9181910417206964325487206224352,
-31.00:-31.4170978194694896501459879800677,
-30.50:-30.9060324277381610613701923284680,
-30.00:-30.3916967524695671443168976111338,
-29.50:-29.8780930151225234681078291032463,
-29.00:-29.3673596195642829798089223913848,
-28.50:-28.8604017167465869420084345620126,
-28.00:-28.3573549006826048923812777502462,
-27.50:-27.8579153415863203235858236439526,
-27.00:-27.3615649830771836548137798672542,
-26.50:-26.8677161642545243580570968333632,
-26.00:-26.3757961562352143403131776722148,
-25.50:-25.8852886093700327307942643528804,
-25.00:-25.3957457746709884816027624765411,
-24.50:-24.9067825736737553654620569432154,
-24.00:-24.4180611277831367544877139152959,
-23.50:-23.9292722039444569759325531776994,
-23.00:-23.4401181709396624341934511903673,
-22.50:-22.9503004725254697859782027080655,
-22.00:-22.4595132928498202318223775364459,
-21.50:-21.9674439986927190204824000829831,
-21.00:-21.4737800750742557909234164981171,
-20.50:-20.9782216082037713533736678073183,
-20.00:-20.4804978955706786791779450140893,
-19.50:-19.9803864598088942727827088674530,
-19.00:-19.4777325936905398862108995672315,
-18.50:-18.9724675509017366437092277919874,
-18.00:-18.4646236039735853751153626944870,
-17.50:-17.9543443995554099501532618887722,
-17.00:-17.4418893350002832676182151772082,
-16.50:-16.9276310416648883006018877495080,
-16.00:-16.4120454722212869569375470746309,
-15.50:-15.8956945344049387358609237708151,
-15.00:-15.3792016747165050105650152545422,
-14.50:-14.8632212754721990677353460341692,
-14.00:-14.3484031699743539434166450519115,
-13.50:-13.8353539862681937933075460023247,
-13.00:-13.3245973826885943935849354602396,
-12.50:-12.8165355209807714231828867923468,
-12.00:-12.3114143179512591785851327585988,
-11.50:-11.8092951071486957914657978108153,
-11.00:-11.3100353107499209670550044393167,
-10.50:-10.8132805514061303142625547479838,
-10.00:-10.3184703070576198058461159234866,
-9.50:-9.8248587114155956356853494071402,
-9.00:-9.3315514117102118518687348114327,
-8.50:-8.8375584961802076122694415971637,
-8.00:-8.3418623794003181615153152961284,
-7.50:-7.8434981666768139874079679429997,
-7.00:-7.3416423921561619891917871427722,
-6.50:-6.8357041217567431345969453104772,
-6.00:-6.3254102143153971837818971835077,
-5.50:-5.8108740252087525135493706329726,
-5.00:-5.2926339989315875911302100575995,
-4.50:-4.7716454134591064217829625704326,
-4.00:-4.2492049924553647599623218411580,
-3.50:-3.7267841742843863528378278715536,
-3.00:-3.2057425021019527733301401895005,
-2.50:-2.6868878598229524179430427466286,
-2.00:-2.1698451072397135774849630251992,
-1.50:-1.6521890467792683665493314038031,
-1.00:-1.1282915671001498836290011240635,
-0.50:-0.5878262377104350733247883908916,
0    : 0
}


