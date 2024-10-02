# **Enhancing Graph Neural Networks for Particle Tracking by Leveraging Attention Mechanisms and Computational Efficiency**

## **Overview**
This repository contains the code and resources used for the research paper **"Enhancing Graph Neural Networks for Particle Tracking by Leveraging Attention Mechanisms and Computational Efficiency"**. The paper introduces improvements to GNN models for particle tracking by enhancing jet tagging accuracy and optimizing computational efficiency. Key contributions include the integration of attention mechanisms, graph sparsification, and the use of automatic mixed precision (AMP).

## **Abstract**
The main focus of this paper is to enhance GNNs for particle tracking at the Large Hadron Collider (LHC) by improving both jet tagging accuracy and computational efficiency. By incorporating attention mechanisms, the model better captures important node dependencies, while graph sparsification reduces memory consumption. Additionally, AMP optimizes training speed, reducing memory usage without compromising accuracy.

## **Installation Instructions**
To run the code provided in this repository, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Set up the environment:**
   We recommend using **Micromamba** to manage your environment. Please refer to the `environment.yml` file in the repository for a list of dependencies.
   ```bash
   micromamba env create -f environment.yml
   micromamba activate your-env-name
   ```

## **Data**
The data used in this project can be obtained from [TrackML](https://www.kaggle.com/c/trackml-particle-identification). Ensure the data is placed in the appropriate directory as outlined in the paper for proper execution of the models.

## **Usage**
### **Training**
For training and experimentation, see the relevant scripts and configuration files in the repository. All training parameters, datasets, and configurations are available in the `configs/` folder.

### **Evaluation**
You can evaluate the trained models by running:
   ```bash
   python evaluate.py --config configs/your_config.yaml
   ```
Ensure that the appropriate paths to trained models and datasets are specified in the config files.

## **Acknowledgments**
This work was heavily inspired by the amazing work done in the [gnn-tracking](https://github.com/gnn-tracking) repository. A huge thank you to their team, whose work laid the foundation for many of the techniques implemented in this research. Their dedication to advancing GNNs in particle tracking has been instrumental in the development of this project.

Special thanks to the core developers of the gnn-tracking project:
- **Gage DeZoort** ![💻](https://github.githubassets.com/images/icons/emoji/unicode/1f4bb.png)
- **Kilian Lieret** ![⚠️](https://github.githubassets.com/images/icons/emoji/unicode/26a0.png)

Their work in GNN particle tracking has greatly influenced the direction and methods used in this project.

## **Contact**
For any questions or collaborations, feel free to contact me at **[cbharathulwar3@gmail.com](mailto:cbharathulwar3@gmail.com)**.
