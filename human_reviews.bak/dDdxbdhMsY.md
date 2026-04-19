# Deep Temporal Deaggregation: Large-Scale Spatio-Temporal Generative Models

- Decision: Reject
- Scores: 3, 6, 6

## Abstract
Access to spatio-temporal trajectory data is essential for improving infrastructure, preventing the spread of disease and for building autonomous vehicles. However, it remains underutilized due to limited availability, as it cannot be shared publicly due privacy concerns or other sensitive attributes. Generative time-series models have shown promise in generating non-sensitive data, but show poor performance for large-scale and complex environments. In this paper we propose a spatio-temporal generative model for trajectories, TDDPM, which outperforms and scales substantially better than state-of-the-art. The focus is primarily on trajectories of peoples' movement in cities. We propose a conditional distribution approach which unlock out-of-distribution generalization, such as to city-areas not trained on, from a spatial aggregate prior. We also show that data can be generated in a privacy-preserving manner using $k$-anonymity. Further, we propose a new comprehensive benchmark across several standard datasets, and evaluation measures, considering key distribution properties.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper propose TDDPM, a spatiotemporal generative model for trajectories. It aims to address two issues, (1) shortage of publicly available trajectories data; (2) model should conduct predictions about unobserved parts in space. To address these issues, TDDPM applies occupancy frequency marginal distribution as local information and hierarchical occupancy frequency mixture. The authors demonstrate the effectiveness of TDDPM on real-world datasets, showing improvements in forecasting accuracy compared to baseline models.

### Strengths
1. The paper focuses on the privacy issues of spatio-temporal trajectory data, as well as enhancing the prediction about unobserved parts in space. This is a very valuable and meaningful research topic.
2. The paper provides detailed visualizations and What-if analysis.

### Weaknesses
1. This paper proposes a privacy-preserving for generating synthetic trajectory samples, however, its contribution to privacy protection is limited. The paper uses only k-anonymity to protect local information in Section 5.3, which is a straightforward design. 
2. There are no baselines in Section 5.2 and Section 5.3. The author should demonstrate how other methods perform in the Generalization experiment and with k-anonymity.

### Questions
Could you provide brief descriptions of your baselines?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents TDDPM, a model for generating high-fidelity, privacy-preserving trajectory data in complex environments. TDDPM leverages a denoising diffusion approach to deaggregate spatial data into individual trajectories, allowing realistic time-series generation that generalizes to unseen areas. By conditioning on spatial aggregates, it achieves strong out-of-distribution performance. The model also uses k-anonymity for privacy and introduces a new benchmark for evaluating synthetic trajectory data, making it a valuable tool for urban planning and autonomous driving applications.

### Strengths
(1) The demonstration of proposed method TDDPM is detailed. 
(2) The authors describe the motivation and background of generating out-of-distribution trajectories in detail.

### Weaknesses
1. The contributions of TDDPM are unclear. The authors should clearly claim the contributions in the end of introduction.
2. In Table 1 and Table 2, standard deviation of KL and JS divergence are not reported.
3. This paper lacks an ablation study part. Experiments should be added to verify the effectiveness of proposed two steps mentioned in Section 4.
4. The experiment setting is confusing. The authors claim that TDDPM could achieve out-of-distribution generalization in Abstract. More analysis should be added to demonstrate the diffirence between the synthetic dataset and Geolife/Porto.
5. Line 308 to line 325 seems making no sense.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new task—out-of-distribution generalization—for synthetic trajectory generation, which current models in the literature cannot address. To enable this generalization capability, the paper proposes using a heatmap as a conditioning constraint in a generative denoising diffusion model. Experiments on both unconditional trajectory generation and generation for new environments and hypothetical scenarios are conducted to evaluate its effectiveness. Additionally, the generation performance is analyzed under private heatmap scenarios.

### Strengths
1. This paper addresses a novel task, enabling the synthetic trajectory generation model to generalize to new areas.
2. A heatmap is used as a novel conditioning mechanism to achieve this generalization capability in the trajectory generation model.
3. Extensive experiments are conducted to evaluate the proposal.

### Weaknesses
1. Additional metrics, such as Density Error, Trip Error, Length Error, and Pattern Score (as used in the TrajDiff paper), should be considered for evaluation.
2. The proposed method splits regions for training, but it is unclear whether the model can correctly generate cross-region trajectories under these conditions.
3. In Table 2, the reasons why the model trained on 25% outperforms the model trained on 100% in most cases should be explained more clearly.
4. As an ablation study, it would be helpful to include results when the model does not use the conditioning component for the unconditional trajectory generation task.
5. Providing information on hyperparameters and including a complexity analysis would improve the paper.
6. Typo:  "ell(..." should be "l(..." on Page 5.

### Questions
Same as the weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
