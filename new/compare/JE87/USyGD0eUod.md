# Review

## Summary
This paper investigates the effectiveness of sparse autoencoders (SAEs) in extracting interpretable features from transformer models, specifically focusing on whether commonly used evaluation metrics can distinguish trained transformers from randomly initialized ones. The authors find that, in many cases, SAEs trained on randomly initialized transformers produce auto-interpretability scores and reconstruction metrics similar to those from trained models. This suggests that high aggregate auto-interpretability scores do not necessarily indicate the presence of learned, computationally relevant features. The paper argues for the need to treat common SAE metrics as insufficient proxies for mechanistic interpretability and proposes the use of routine randomized baselines and targeted measures of feature "abstractness" to improve the evaluation of SAEs.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper identifies a critical limitation in the use of aggregate auto-interpretability scores for evaluating SAEs, highlighting that these scores do not always reflect the complexity or computational relevance of learned features.
- The experimental methodology is robust, comparing trained transformers with randomly initialized models across a range of Pythia model sizes and multiple randomization schemes. This provides a thorough basis for the findings.
- The paper provides a theoretical foundation for understanding superposition in random networks, which contributes to the broader discourse on feature learning in neural networks.

## Weaknesses
- The paper primarily focuses on the Pythia family of models and the RedPajama dataset, which may limit the generalizability of the findings to other architectures and datasets. Further experiments on diverse model architectures would strengthen the claims.
- While the paper introduces entropy as a measure of feature abstractness, it acknowledges that this is a preliminary proof-of-concept. Developing more robust metrics for evaluating feature complexity would enhance the contributions.
- The study does not thoroughly explore the implications of its findings for the development of future SAEs or other interpretability methods. Providing more detailed guidance on how to apply these insights would increase the practical value.

## Questions
- How do the findings of this study translate to other types of neural networks beyond transformers, or to SAEs applied to different domains (e.g., vision or reinforcement learning)?
- What specific steps can researchers take to implement the recommended metrics and randomized baselines in their own work?
- How do the authors envision the field moving forward in developing more robust metrics for evaluating the interpretability of learned features?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4