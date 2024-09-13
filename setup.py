from setuptools import setup, find_packages

setup(
    name='LDGD',
    version='0.2',
    description='A Python package for Gaussian Process models implemented in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Navid-Ziaei/gp_project_pytorch',
    author='Navid Ziaei',
    author_email='nziaei@wpi.edu',  # Replace with your email
    license='MIT',
    install_requires=[
        'colorcet>=3.0.1',
        'GPy>=1.12.0',
        'gpytorch>=1.11',
        'h5py>=2.10.0',
        'joblib>=1.1.1',
        'linear_operator>=0.5.2',
        'matplotlib>=3.6.1',
        'mne>=1.2.1',
        'mogptk>=0.3.4',
        'mpi4py>=3.1.5',
        'networkx>=3.1',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'paramz>=0.9.5',
        'plotly>=5.15.0',
        'pods>=0.1.14',
        'scikit_learn>=1.1.2',
        'seaborn>=0.10.1',
        'tables>=3.7.0',
        'torch>=1.12.1',
        'tqdm>=4.64.1'
        # Add other dependencies here
    ],
    package_data={
        # If any package contains *.txt or *.json files, include them:
        '': ['*.txt', '*.json'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6"
)
