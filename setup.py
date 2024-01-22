from setuptools import setup, find_packages

setup(
    name='NTK',
    version='0.1.0',
    url='https://github.com/filipposchiazza/NTK',
    author='Filippo Schiazza',
    description='Implementation of the Multi-Layer Neural Tangent Kernel',
    packages=find_packages(),    
    install_requires=[
        'numpy==1.25.0',
        'matplotlib==3.7.1',
        'torch==1.12.1',
        'tqdm==4.65.0',
    ],
)