from setuptools import setup

setup(name='lrfinder',
      version='0.1',
      description='Learning rate finder.',
      url='http://github.com/beringresearch/lrfinder',
      author='Bering Limited',
      license='Apache 2.0',
      packages=['lrfinder'],
      install_requires=[
            'tensorflow',
      ],
      zip_safe=False)
