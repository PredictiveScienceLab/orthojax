from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='orthojax',
    version='0.1.2',    
    description='Orthogonal polynomials in Jxax',
    long_description=long_description,
    long_description_content_type='text/markdown',
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