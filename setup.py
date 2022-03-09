import os
import setuptools

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = [r.strip() for r in f]

setuptools.setup(
    name='jax_dft',
    version='0.0.0',
    license='Apache 2.0',
    author='Google LLC and Riksi',
    install_requires=requirements,
    url='https://github.com/Riksi/jax_dft/tree/master/jax_dft',
    packages=setuptools.find_packages(),
    python_requires='>3.6'
)