# Project Title
Few-Shot Learning with Adversarial Robustness using Prototypical Networks and MAML

## Description
This project focuses on addressing the research problems of adversarial robustness with MAML and prototypical networks in few-shot learning.

## Installation
1. Install the required Deep Learning framework, PyTorch.
2. Clone this repository to your local machine.

## Data Collection and Preprocessing
- Three datasets were used: Omniglot, Mini-Imagenet, and CUB.
- Data was preprocessed and split into train and test sets.
- Ensure the data is formatted correctly for the model training.

## Model Implementation
### Generic Prototypical Model
- Structure: [Describe the structure of the model]
- Implementation: [Explain the implementation details in PyTorch]

### Basic MAML
- Structure: [Describe the structure of the model]
- Implementation: [Explain the implementation details in PyTorch]

### Training and Evaluation
- The models were trained using SGD optimizer.
- The Evaluation function calculates accuracy and cross-entropy loss on the test data.

## Adversarial Attack Implementation
- Implemented k-step PGD attack for generating adversarial samples.
- Modified the train function to incorporate adversarial regularization.
- Adjusted the evaluate function to output the Robustness metric.

## Hyperparameter Tuning
- Identified and listed all hyperparameters of the models.
- Employed [Hyper-param Optimization Method] for optimizing parameters on each dataset.

## Experimentation and Analysis
- Presented experimental results on accuracy and robustness.
- Utilized various experiments and averaged the results.
- Analyzed the outcomes and prepared visualizations for better understanding.

## Overall Implementation structure
# data
- dataloader 
  - return x_spt, y_spt, x_qry, y_qry 
  - x: Meta batck size (num_task) x N_way*K_shot x C x H x W
  - y: Meta batck size (num_task) x N_way*K_shot
    (1 common N_way for the support and querry sets)
    (K_shot_sp, K_shot_querry)
- omniglot
- mini-imagenet
- cub

# model
- meta_learner: can be implemented with any models like protonet used for maml
- protonet
- proposednet

# tools
- attacks
  - fgsm (white-box)
  - pgd (white-box)
  - pixle (black-box)
- defense
- losses
- train
- test

# experiments
- omniglot
  - meta_learner
  - protonet
  - proposednet
- mini-imagenet
  - meta_learner
  - protonet
  - proposednet
- cub
  - meta_learner
  - protonet
  - proposednet

# main

## References
- [List relevant references here]
