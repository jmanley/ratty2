import pylab,h5py,rfi_sys, time, corr, numpy, struct, sys, logging

bram_out_prefix = 'store'

class roach_handle:
    def __init__(self):
        try:
            print 'Connecting to ROACH...'
            self.r = rfi_sys.cam.spec()

            if self.r.spectrum_bits != 64: 
                print 'ERR: Sorry, this is only for 64 bit systems.'
                exit()

            #Access configuration of RATTY
            self.acc_time, self.n_accs = self.r.acc_time_get()       #Get time for each accumulation and number of accumulations
            self.freqs = self.r.freqs
            self.fft_shift = self.r.fft_shift_get()
            self.fft_scale = self.r.fft_scale
            self.rf_gain = self.r.rf_status_get()[1]
            self.bandwidth = self.r.bandwidth
            self.n_chans = self.r.n_chans
            self.bandshape = rfi_sys.cal.bandshape(self.freqs)

            print 'Scaling back by %i accumulations.'%self.n_accs

            self.last_cnt = self.r.fpga.read_uint('acc_cnt')

            self.af=None
            self.units='dBm'

        except Exception as e:
            print 'Runtime error: ',e
            raise e
            exit()


    def getSpectrum(self,n_acc):
    	spectrum = []
    	acc_cnt = []
    	timestamp = []
    	adc_overrange = []
    	fft_overrange = []
    	adc_shutdown = []
    	adc_level = []
    	input_level = []
    	adc_temp = []
    	ambient_temp = []

        while n_acc:		#While we still need to grab next spectra
            spectra, time, self.last_cnt, stat = self.getUnpackedData(self.last_cnt)
            spectrum.append(spectra)
            acc_cnt.append(self.last_cnt)
            timestamp.append(time)
            adc_overrange.append(stat['adc_overrange'])
            fft_overrange.append(stat['fft_overrange'])
            adc_shutdown.append(stat['adc_bad'])
            adc_level.append(stat['adc_level'])
            input_level.append(stat['input_level'])
            adc_temp.append(stat['adc_temp'])
            ambient_temp.append(stat['ambient_temp'])
            n_acc = n_acc - 1

    	return spectrum, acc_cnt, timestamp, adc_overrange, fft_overrange, adc_shutdown, adc_level, input_level, adc_temp, ambient_temp

    def getUnpackedData(self,last_cnt):
        """Gets data from ROACH board and returns the spectra, the state of the roach at the last timestamp"""
        while self.r.fpga.read_uint('acc_cnt') == last_cnt:  #Wait untill the next accumulation has been performed
            time.sleep(0.1)
        spectrum = numpy.zeros(self.r.n_chans) 				#Get spectra
        for i in range(self.r.n_par_streams):
            spectrum[i::self.r.n_par_streams] = numpy.fromstring(self.r.fpga.read('%s%i'%(bram_out_prefix,i),self.r.n_chans/self.r.n_par_streams*8),dtype=numpy.uint64).byteswap()
        stat = self.r.status_get()
        ampls = self.r.adc_amplitudes_get()
        stat['adc_level'] = ampls['adc_dbm']
        stat['input_level'] = ampls['input_dbm']
        stat['adc_temp'] = self.r.adc_temp_get()
        stat['ambient_temp'] = self.r.ambient_temp_get()
        last_cnt = self.r.fpga.read_uint('acc_cnt')
        timestamp = time.time()

        print '[%i] %s: input level: %5.2f dBm (ADC %5.2f dBm).'%(last_cnt,time.ctime(timestamp),stat['input_level'],stat['adc_level']),
        if stat['adc_bad']: print 'ADC selfprotect due to overrange!',
        elif stat['adc_overrange']: print 'ADC is clipping!',
        elif stat['fft_overrange']: print 'FFT is overflowing!',
        else: print 'all ok.',
        print ''
        return spectrum, timestamp, last_cnt, stat

    def exit_clean(self):
        try:
            self.r.fpga.stop()
        except:
            pass
        exit()

if __name__ == '__main__':
    print 'yo'
    test = roach_handle()
    spectrum, acc_cnt, timestamp, adc_overrange, fft_overrange, adc_shutdown, adc_level, input_level, adc_temp, ambient_temp = test.getSpectrum(5)
    for i in range(len(acc_cnt)):
        print '\n---------------%i------------------'%i
        print spectrum[i]
        print acc_cnt[i]
        print timestamp[i]
        print adc_overrange[i]
        print adc_level[i]
        print input_level[i]
        print adc_temp[i]
        print ambient_temp[i]



