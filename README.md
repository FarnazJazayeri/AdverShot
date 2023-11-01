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

## References
- [List relevant references here]
