from setuptools import setup, find_packages

setup(name='icsoc-2022',
      version='0.1.0',
      description='Implementation of stochastic service composition.',
      url='http://github.com/luusi/ICSOC-2022',
      author='Luciana Silo',
      author_email='silo@diag.uniroma1.it',
      license='MIT',
      packages=find_packages(include='icsoc_2022*'),
      zip_safe=False,
      install_requires=[
            "numpy",
            "graphviz",
            "websockets",
            "paho-mqtt",
            "requests",
            "mdp_dp_rl @ git+https://github.com/luusi/mdp-dp-rl.git#egg=mdp_dp_rl"
      ]
)
