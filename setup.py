from distutils.core import setup, Extension
import os, sys, glob

__version__ = '0.1.0'

setup(name = 'ratty1',
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
    provides=['ratty1'],
    package_dir = {'ratty1':'src'},
    packages = ['ratty1'],
    scripts=glob.glob('scripts/*'),
    data_files=[('/etc/ratty1', glob.glob('config_files/*')),('/etc/ratty1/cal_files', glob.glob('cal_files/*'))]
)

