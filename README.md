## Hydro-KAN-V1

**Official Code for "Exploring Kolmogorov-Arnold Neural Networks for Hybrid and Transparent Hydrological Modeling"**

This repository contains the official source code and experimental framework for the research paper, "Exploring Kolmogorov-Arnold Neural Networks for Hybrid and Transparent Hydrological Modeling." This project investigates the application of Kolmogorov-Arnold Neural Networks (KANs) to advance hydrological modeling, with a focus on creating hybrid models that are both accurate and interpretable.

### Overview

The core of this research is to leverage the unique properties of KANs to build transparent and efficient hydrological models. Traditional neural networks often function as "black boxes," making it difficult to understand their internal decision-making processes. KANs, with their learnable activation functions, offer a pathway to more interpretable models. This project explores this potential by:

* **Developing Hybrid Models:** Integrating KANs with established hydrological models to enhance predictive performance while maintaining a degree of physical realism.
* **Enhancing Transparency:** Utilizing the inherent structure of KANs for easier interpretation and analysis of the learned hydrological relationships.
* **Symbolic Regression:** Deriving mathematical equations from the trained KAN models to provide clear and concise representations of the underlying hydrological processes.

### Directory Structure

The `src` directory is organized into the following subdirectories, each responsible for a specific part of the research workflow:

* `finetune`: Contains scripts and notebooks for fine-tuning the architecture of the KAN models.
* `models`: Defines the structure of the KAN models, built using the `HydroModels.jl` framework.
* `plots`: Includes code to generate the figures and visualizations presented in the paper.
* `reg`: Holds the implementation for the regularization techniques applied during the training of the KAN models.
* `run`: The primary directory for conducting the modeling experiments. It includes scripts to train the standalone hydrological models (exp-hydro) and the hybrid models (K50/M50).
* `stats`: Provides a collection of statistical tools used for model evaluation and analysis.
* `sym`: Contains the code for performing symbolic regression on the activation functions of the trained KAN models to extract mathematical formulas.
* `utils`: A library of utility functions that support the various components of the project.
* `xai`: Includes tools and methodologies for eXplainable AI (XAI), aimed at interpreting the behavior of the trained models.

### Getting Started

To replicate the experiments and utilize the models from this study, please follow the steps outlined below.

#### Prerequisites

Ensure you have the necessary dependencies installed. The core modeling framework is built upon `HydroModels.jl`. Additional Python and Julia packages may be required and can be found in the respective script files.

#### Usage Workflow

The intended sequence for running the code and reproducing the results is as follows:

1.  **Model Calibration (`run` directory):**
    Begin by calibrating the baseline hydrological model (`exp-hydro`) and the hybrid KAN-based models (`K50`/`M50`). The scripts within the `run` directory are used for this purpose. These scripts will train the models on the specified datasets.

2.  **Regularization (`reg` directory):**
    After the initial calibration, apply regularization techniques to the trained models. The code in the `reg` directory facilitates this process, which helps in preventing overfitting and improving the generalizability and interpretability of the models.

3.  **Symbolic Regression (`sym` directory):**
    With the regularized and trained KAN models, proceed to the `sym` directory. The scripts here are used to perform symbolic regression on the learned activation functions of the KANs. This step is crucial for translating the learned patterns into explicit mathematical equations, a key aspect of the model's transparency.

4.  **Analysis and Visualization:**
    The remaining directories support the analysis and presentation of the results:
    * The model definitions are located in the `models` directory.
    * The `plots` directory can be used to generate the figures that visualize the model outputs and comparisons.
    * The `stats` directory provides tools for statistical analysis of the model performance.
    * The `xai` directory offers methods for a deeper, explainable AI-driven analysis of the model's behavior.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.