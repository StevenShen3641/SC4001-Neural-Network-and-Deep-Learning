# Testing Model Adaption Ability via Domain Shift on Movie Review Sentiment Analysis

## Overall Description

This is our project implementation for SC4001, 2024 Spring. This project is done by Phee Kian Ann, Mohamed Ali Mohamed Umar and Shen Chihao. For project  details, you can refer to the project report. 

## Project Introduction

This project is a study on Text Sentiment Analysis (TSA), which is the identification of emotions or opinions conveyed by the portion of text in question. It is a common classification task, typically resulting in either positive or negative outcomes. TSA has a wide variety of use cases that are commonly found in the business sector, and analysis of user feedback, such as reviews, can bring about valuable insights on the performance and success of certain products. With deep learning, we are able to analyse user-generated data automatically and thus be able to process a huge quantity of data that would have been impossible manually to garner more insights into user behaviour and feelings in the domain. 

For this project, we are challenging the feasibility of domain adaptation so that a model that has been trained on one particular dataset can be re-used on multiple other datasets. The model is required to have flexibility to adapt to the introduction of new words, the absence of specific words, and possibly different phrasing contexts. The datasets used in this project are the SST-2 dataset for the initial model training and testing and the IDMB review dataset for the evaluation of our model performance for domain shift. We have experimented with three architectures of neural networks: Convolutional Neural Networks, Long-Short-Term Memory, and Hierarchical Attention Networks.

## Implementation

| Name     | Part                 |
| -------- | -------------------- |
| Kian Ann | HAN                  |
| Umar     | LSTM, LSTM-Attention |
| Chihao   | CNN-WV, CNN-Char     |

## Set Up Instruction

We implement all our models using Jupyter Notebook. Each notebook contains a specific model.

### Structure

- All notebooks and Python scripts are located in the root directory.
- Datasets are stored under `./data` directory.
- Images (such as t-SNE visualization) are stored under `./image` directory.

### Prerequisites

Create a new Python virtual environment. After activating the environment, run the following command in the terminal to install all necessary packages:

```powershell
pip install -r requirement.txt
```

### Run the Project

Step 1:

Run the following command to pre-processing the datasets to onehot tensors for **CNN-Char** model:

```powershell
python .\data_preprocessing.py
```

Step 2:

Run `jupyter notebook` to start the notebook. Each model has its separated `.ipynb` file. You can try different models you need.

Available notebooks along with models:

```powershell
CNN_onehot.ipynb  # character-wise embedding CNN
CNN_vector.ipynb  # word vector embedding CNN 
# CNN_vector_val.ipynb is for hyperparameter tuning demonstration
HAN.ipynb  # Hierarchical Attention Network model
LSTM.ipynb  # bidirectional LSTM
LSTM-Attention.ipynb  # bidirectional LSTM with attention layers
```

Appendix:

If you want to visualize t-SNE of both datasets, you can execute `t-SNE.ipynb` to visualize them.

