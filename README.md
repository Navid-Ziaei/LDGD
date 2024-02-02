# A Discriminative Bayesian Gaussian Process Latent Variable Model for High-Dimensional Data
# Latent Discriminative Generative Decoder (LDGD)

## Overview
Extracting meaningful information from high-dimensional data poses a formidable modeling challenge, particularly when the data is obscured by noise or represented through different modalities. In this research, we propose a novel non-parametric modeling approach, leveraging the Gaussian Process (GP), to characterize high-dimensional data by mapping it to a latent low-dimensional manifold. This model, named the Latent Discriminative Generative Decoder (LDGD), utilizes both the data (or its features) and associated labels (such as category or stimulus) in the manifold discovery process. To infer the latent variables, we derive a Bayesian solution, allowing LDGD to effectively capture inherent uncertainties in the data while enhancing the model's predictive accuracy and robustness. We demonstrate the application of LDGD on both synthetic and benchmark datasets. Not only does LDGD infer the manifold accurately, but its prediction accuracy in anticipating labels surpasses state-of-the-art approaches. We have introduced inducing points to reduce the computational complexity of Gaussian Processes (GPs) for large datasets. This enhancement facilitates batch training, allowing for more efficient processing and scalability in handling extensive data collections. Additionally, we illustrate that LDGD achieves higher accuracy in predicting labels and operates effectively with a limited training dataset, underscoring its efficiency and effectiveness in scenarios where data availability is constrained. These attributes set the stage for the development of non-parametric modeling approaches in the analysis of high-dimensional data; especially in fields where data are both high-dimensional and complex.

## Reference
For more details on our work and to cite it in your research, please visit our paper: [See the details in ArXiv, 2024](https://arxiv.org/abs/2401.16497). Cite this paper using its [DOI](https://doi.org/10.48550/arXiv.2401.16497).
