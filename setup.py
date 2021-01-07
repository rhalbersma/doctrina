#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from setuptools import setup, find_packages

setup(
    name='doctrina',
    version='0.1.0-dev0',
    description='Reinforcement learning',
    url='https://github.com/rhalbersma/doctrina',
    author='Rein Halbersma',
    license='Boost Software License 1.0 (BSL-1.0)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym', 'ipykernel', 'jinja2', 'matplotlib', 'numpy', 'pandas', 'plotnine', 'pylint', 'scipy', 'seaborn', 'setuptools', 'tqdm', 'wheel',
        'gym_blackjack_v1 @ git+https://github.com/rhalbersma/gym-blackjack-v1.git#egg=gym_blackjack_v1',
        'gym_jcr @ git+https://github.com/gsverhoeven/gym_jcr.git#egg=gym_jcr'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha'
        'Intended Audience :: Science/Research'
        'License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
