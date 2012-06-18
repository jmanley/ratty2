from distutils.core import setup, Extension
import os, sys, glob

__version__ = '0.0.7'

setup(name = 'rfi_sys',
    version = __version__,
    description = 'Interfaces to an RFI monitoring spectrometer',
    long_description = 'Provides interfaces to a ROACH-based RFI monitoring spectrometer.',
    license = 'GPL',
    author = 'Jason Manley',
    author_email = 'jason_manley at hotmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    requires=['h5py', 'pylab','matplotlib','numpy','corr'],
    provides=['rfi_sys'],
    package_dir = {'rfi_sys':'src'},
    packages = ['rfi_sys'],
    scripts=glob.glob('scripts/*'),
    data_files=[('/etc/rfi_sys/cal_files', glob.glob('cal_files/*'))]
    #package_data={'rfi_sys': ['cal_files/*']}

    #data_files=[('/etc',['corr_example.conf'])]
)

