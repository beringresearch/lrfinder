from setuptools import setup

setup(name='lrfinder',
      version='0.1',
      description='Learning rate finder.',
      url='http://github.com/beringresearch/lrfinder',
      author='Ignat Drozdov',
      author_email='idrozdov@beringresearch',
      license='TBD',
      packages=['lrfinder'],
      install_requires=[
            'tensorflow>=2.1.0'
      ],
      zip_safe=False)