from setuptools import setup

setup(
    name='orthojax',
    version='0.1.0',    
    description='Orthogonal polynomials in Jax',
    url='https://github.com/PredictiveScienceLab/orthojax',
    author='Iilias Bilionis',
    author_email='ibilion@purdue.edu',
    license='MIT License',
    packages=['orthojax'],
    install_requires=['jax>=0.4.19',
                      'numpy', 
                      'equinox>=0.11.2'                    
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3'
    ],
)