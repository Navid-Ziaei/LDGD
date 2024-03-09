[![arXiv](https://img.shields.io/badge/arXiv-2206.03992-b31b1b.svg)](https://arxiv.org/abs/2401.16497)

# A Bayesian Gaussian Process-Based Latent Discriminative Generative Decoder (LDGD) Model for High-Dimensional Data

## Table of Contents
* [General Information](#general-information)
* [Reference](#reference)
* [Getting Started](#getting-started)
* [Example](#example)
* [Reading in Data](#reading-in-edf-data)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
<br/>
* 
## General Information
Extracting meaningful information from high-dimensional data poses a formidable modeling challenge, particularly when the data is obscured by noise or represented through different modalities. In this research, we propose a novel non-parametric modeling approach, leveraging the Gaussian Process (GP), to characterize high-dimensional data by mapping it to a latent low-dimensional manifold. This model, named the Latent Discriminative Generative Decoder (LDGD), utilizes both the data (or its features) and associated labels (such as category or stimulus) in the manifold discovery process. To infer the latent variables, we derive a Bayesian solution, allowing LDGD to effectively capture inherent uncertainties in the data while enhancing the model's predictive accuracy and robustness. We demonstrate the application of LDGD on both synthetic and benchmark datasets. Not only does LDGD infer the manifold accurately, but its prediction accuracy in anticipating labels surpasses state-of-the-art approaches. We have introduced inducing points to reduce the computational complexity of Gaussian Processes (GPs) for large datasets. This enhancement facilitates batch training, allowing for more efficient processing and scalability in handling extensive data collections. Additionally, we illustrate that LDGD achieves higher accuracy in predicting labels and operates effectively with a limited training dataset, underscoring its efficiency and effectiveness in scenarios where data availability is constrained. These attributes set the stage for the development of non-parametric modeling approaches in the analysis of high-dimensional data; especially in fields where data are both high-dimensional and complex.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2401.16497). Cite this paper using its [DOI](https://doi.org/10.48550/arXiv.2401.16497).

## Getting Started

1. Step 1: Clone the Repository 
`git clone https://github.com/Navid-Ziaei/gp_project_pytorch.git`

2. Step 2: Navigate to the Project Directory
`cd gp_project_pytorch`

3. Step 3: Update pip (optional but recommended)
`python -m pip install --upgrade pip`

4. Step 4: Install virtualenv if you haven't already and activate it (you can use conda env as well)

5. Step 5: Install the LDGD Package
`python -m pip install`

6. After installation, to verify if everything is set up correctly, run:
`python -c "import ldgd; print(ldgd.__version__)`"

## Examples
To help you get started with LDGD and to demonstrate its capabilities, we have included a variety of examples in the examples folder of the repository. These examples cover a range of applications and use cases, showing you how to implement Gaussian Process models with LDGD in your projects.



## Repository Structure
This repository is organized as follows:

- `src/main.py`: The main script to run the LDGD model.

- `src/LDGD/data`: Contains scripts for data loading  and preprocessing .

- `src/LDGD/experiments`: Contains scripts for different experiments.

- `src/LDGD/model`: Contains the main LDGD and Fast LDGD model .

- `src/LDGD/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `src/LDGD/utils`: Contains utility scripts.. 

- `src/LDGD/visualization`: Script for data and result visualization.
<br/>

## Citations
The code contained in this repository for LDGD is companion to the paper:  

```
@article{ziaei2024discriminative,
  title={A Discriminative Bayesian Gaussian Process Latent Variable Model for High-Dimensional Data},
  author={Ziaei, Navid and Nazari, Behzad and Yousefi, Ali},
  journal={arXiv preprint arXiv:2401.16497},
  year={2024}
}
```
which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to LDGD! 

## License

This project is licensed under the terms of the MIT license.