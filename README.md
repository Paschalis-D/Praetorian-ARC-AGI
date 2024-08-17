# Praetorian-ARC-AGI
A repository that includes the model architecture and training procedure of PraetorianAI team for the ARC_AGI competition.

The core logic followed by this submission is: 
1. A diffuser will be trained with every ARC dataset that is publicly available and found in (ENTER REPOSITORY).
2. We will then apply transfer learning to this diffuser with the training examples of each task in the ARC-AGI dataset. This will happen during the inference process.
3. A Relational Neural Network will also be trained on all of the datasets to learn to calculate a difference score between the various outputs of each task of each dataset.
4. Finally a Refinement Neural Network will refine the output of the diffuser using the output of the Relational NN as a loss function.