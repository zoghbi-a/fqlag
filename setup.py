from distutils.core import setup
setup(
  name = 'fqlag', 
  packages = ['fqlag'],
  version = '0.3',
  license='MIT',
  description = ('Calculating Periodogram and Time delays in the frequency domain '
                'from unevenly-sample time series'),
  author = 'Abdu Zoghbi',
  author_email = 'astrozoghbi@gmail.com',
  url = 'https://github.com/zoghbi-a/fqlag',
  download_url = 'https://github.com/zoghbi-a/fqlag/archive/v0.3.tar.gz',
  keywords = ['Astronomy', 'Time-Series', 'Delays'],
  install_requires=[ 
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
