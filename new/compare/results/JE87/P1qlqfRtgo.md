# Review

## Summary
The paper explores the impact of neural network architecture on the accuracy of data-driven modeling for thermal explosions in hydrogen-oxygen-air mixtures. Using a reduced kinetic mechanism for 11 reactants, the study simulates the thermal explosion process under specific initial pressure and temperature conditions, generating time-resolved data. Three architectures are compared: a standard multilayer perceptron (MLP), a DeepONet-inspired model, and a U-Net-style residual network, evaluating their ability to capture transient dynamics and key reaction regimes. The results show that the U-Net architecture consistently outperformed the other models, achieving a lower mean squared error (MSE) and reduced standard deviation (STD) in capturing both rapid transients and slower reaction dynamics.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
The experimental results indicate that the U-Net architecture consistently outperformed the other models, achieving a lower mean squared error (MSE) and reduced standard deviation (STD) in capturing both rapid transients and slower reaction dynamics.

## Weaknesses
1. The paper's writing quality is subpar, with numerous errors, particularly in the experimental results section, where the authors mistakenly included the discussion of experimental results from the appendix as part of the main text. This resulted in an incomplete and confusing discussion of the findings.

2. The paper's novelty is limited, as it primarily focuses on comparing different network architectures without introducing any new methodologies or significant contributions to the field.

3. The experimental results lack credibility, as the authors did not provide detailed information about the training process, such as the training time, hyperparameters used, or whether the experiments were conducted multiple times with different random seeds. This omission raises concerns about the reliability and reproducibility of the reported results.

4. The authors did not conduct any ablation studies to analyze the impact of different hyperparameters or architectural choices on the model's performance. This lack of analysis makes it difficult to draw meaningful conclusions about the effectiveness of the proposed approaches.

## Questions
The authors did not provide detailed information about the training process, such as the training time, hyperparameters used, or whether the experiments were conducted multiple times with different random seeds. This omission raises concerns about the reliability and reproducibility of the reported results.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4