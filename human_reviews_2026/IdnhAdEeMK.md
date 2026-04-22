# Subspace Inference Enables Active Preference based Learning of Neural Network Reward Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Reinforcement learning from human feedback (RLHF) has emerged as a powerful approach for aligning decision-making agents with human intentions, primarily through the use of reward models trained on human preferences. However, RLHF suffers from poor sample efficiency, as each feedback provides minimal information, making it necessary to collect large amounts of human feedback. Active learning addresses this by enabling agents to select informative queries, but effective uncertainty quantification required for active learning remains a challenge. While ensemble methods and dropout are popular for their simplicity, they are computationally expensive at scale and do not always provide good posterior approximation.
Inspired by the recent advances in approximate Bayesian inference, we develop a method that leverages Bayesian filtering in neural network subspaces to efficiently maintain model posterior for active reward modeling. Our approach enables scalable sampling of neural network reward models to efficiently compute active learning acquisition functions. Experiments on the D4RL benchmark demonstrate that our approach achieves superior sample efficiency, scalability, and calibration compared to other Bayesian deep learning approaches, and leads to competitive offline reinforcement learning policy performance. This highlights the potential of scalable Bayesian methods for preference-based reward modeling in RLHF.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses sample efficiency in reward modeling for RLHF. The authors propose a Bayesian approach to select queries that maximize mutual information between labels and reward model parameters. To ensure practical applicability, the method integrates Bayesian learning with subspace inference, limiting computation to a subset of parameters. Experimental validation on 12 D4RL tasks shows the proposed method achieves competitive performance compared to dropout and ensemble baselines while requiring substantially fewer active labels.

### Strengths
1. The paper presents a novel method by applying Bayesian methods to select queries with maximum information gain for active preference labeling.

### Weaknesses
1. The paper lacks clarity on the underlying mechanism by which the Bayesian-based method achieves superior sample efficiency. The theoretical or empirical justification for why this approach outperforms alternatives is insufficient.
2. The choice of test-set log-likelihood as an evaluation metric for reward model quality is inadequately justified. The claim of improved sample efficiency rests on showing that the proposed method reaches equivalent test-set log-probability with fewer annotated samples than random sampling. This conclusion requires more rigorous empirical analysis and validation.
3. The baseline comparison is limited, incorporating only dropout and ensemble methods. The evaluation would benefit from including additional state-of-the-art active learning baselines.

### Questions
1. In Figure 1, why do methods using random sampling exhibit significant performance variation according to your evaluation metric? Additionally, why does the normalized score for random sampling sometimes exceed that of active labeling with your proposed method?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Reinforcement learning from human feedback (RLHF) relies on preference labels provided by humans, which is time-consuming. Following the line of active learning, this paper proposes leveraging Bayesian filtering in neural network subspaces to efficiently maintain a model posterior for active reward modeling. Compared with ensemble and dropout methods, PreferenceEKF demonstrates superior sample efficiency on the D4RL benchmark.

### Strengths
- The paper is well-structured and easy to follow.

- Adopting extended Kalman filters for training neural networks is intuitively sound.

- The experiments demonstrate performance gains compared with ensemble and dropout methods.

### Weaknesses
- The proposed method is an adaptation of existing methods for network training. However, some assumptions remain unverified in the experiments, making it unclear how well the method can scale. For instance, the authors cite a paper to support the claim that "neural networks are overparameterized and that solutions actually live in a much smaller subspace." Yet, the phrase "much smaller subspace" is vague, and the values of $|z|$ and $|\theta|$ used in the experiments do not differ significantly.

- One motivation mentioned is that "ensemble methods and dropout are computationally expensive at scale." However, the experiments are tested on a toy model with only two layers, and there are no results demonstrating the proposed method’s performance at scale. This means the experiments fail to support the broader claims introduced in the paper.

- The paper conflates sample efficiency and time efficiency. Conducting experiments with a small network and comparing against only two very basic baselines does not clearly validate either type of efficiency. Additionally, several RLHF or PBRL methods—though not within the active learning paradigm—could have been included as baselines to enable a more comprehensive comparison of sample efficiency [1, 2].

[1]Park, Jongjin, et al. "Surf: Semi-supervised reward learning with data augmentation for feedback-efficient preference-based reinforcement learning." arXiv preprint arXiv:2203.10050 (2022).

[2] Liu, Runze, et al. "Meta-reward-net: Implicitly differentiable reward learning for preference-based reinforcement learning." Advances in Neural Information Processing Systems 35 (2022): 22270-22284.

### Questions
- Following your claim that "neural networks are overparameterized and that solutions actually live in a much smaller subspace," I am curious: if we use ensemble methods with N models—each having $|\theta|/N$ parameters, would such an approach achieve good performance and be efficient?

- Please refer to the "Weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PreferenceEKF, which maintains a Gaussian posterior over reward-model parameters in a learned low-dimensional subspace and updates it online via an extended Kalman filter (EKF) under a Bradley–Terry preference likelihood. Sampling from this subspace posterior enables efficient, posterior-based active learning via mutual information (MI) without training large ensembles or relying on dropout. Experiments on D4RL tasks report improved label efficiency over random querying, competitive or better calibration, scalable inference, and competitive downstream offline RL returns.

### Strengths
Principled active learning: Uses MI over pairwise preferences with a maintained posterior, which is solid Bayesian practice for label efficiency.

Scalable posterior sampling: Subspace + EKF yields cheap sampling for acquisition, avoiding K-way ensembles or heavy MCMC; practical for neural reward models.

Clarity of core idea: The subspace mapping and EKF update are presented clearly; ablations on subspace dimension aid interpretation.

Potential impact: Addresses a central RLHF bottleneck (costly preference labels) by making posterior-based acquisitions feasible.

### Weaknesses
Missing Bayesian baselines: Does not compare against competitive and computationally efficient approximate Bayes methods (Laplace redux, last-layer VI, SGLD/SWAG). This weakens novelty and empirical support.

Posterior expressivity: A unimodal Gaussian in a learned subspace with first-order linearization can misrepresent uncertainty under misspecification, multi-modality, or annotator heterogeneity; these regimes are not stress-tested. (More reasoning: The posterior is a single Gaussian in a learned subspace θ=Az+θ\* where A comes from SVD of SGD iterates and the prior in z is isotropic. This is an empirical-Bayes choice whose geometry is optimizer-induced; isotropy in z becomes anisotropy in θ, and coverage depends on whether the iterate subspace spans high-mass posterior directions. There is no evidence-based selection of ∣z∣ (e.g., WAIC/PSIS-LOO) and no hierarchical/shrinkage prior on z. Meanwhile, the iterated EKF with a Bradley–Terry likelihood (n_linearize=5) implements a sequential Laplace/ADF update. It is reasonable near a single mode with mild curvature, but can under-estimate uncertainty under sharp curvature/skew and cannot represent multimodality (e.g., heterogeneous annotators), which can in turn bias MI-based query selection.)

Statistical reporting: Few seeds and limited uncertainty reporting; lacking confidence intervals, effect sizes, and paired tests at matched budgets.

Human data gap: Reliance on synthetic preferences leaves external validity to real annotators unclear.

Compute transparency: Per-query acquisition budget (number of posterior samples M, candidate-pool size/refresh, wall-clock) is under-specified, making it hard to judge real-world efficiency.

### Questions
Results for Laplace (full/last-layer), variational last-layer, and SGLD/SWAG on a representative subset?

Exact acquisition budget per query: M, pool size/refresh, and wall-clock on your stated hardware?

Sensitivity to ∣z∣, SVD-iterate selection, and offset θ\*; can you provide evidence-based subspace selection (WAIC/PSIS-LOO)?

Robustness with human or heterogeneous annotators; any hierarchical BT results?

Can you add 95% CIs, paired tests, and effect sizes; plus NLL and reliability diagrams with ECE binning details?

Do conclusions hold under CQL or TD3+BC given the same learned rewards?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper is motivated by the poor sample efficiency of reinforcement learning from human feedback (RLHF) and the lack of scalable methods for active learning based mitigations. To address this issue, adopting recent advances in Bayesian subspace filtering, the authors propose the PreferenceEKF (Preference Extended Kalman Filter) method which maintains a posterior distribution over reward model (RM) parameters and enables sampling-based acquisition functions such as InfoGain to perform active learning at scale.

The key contributions are:

- PreferenceEKF, to the best of my knowledge also, the first method to apply Bayesian subspace filtering for active preference-based reward learning.
    
- Scaling InfoGain for active preference-based reward learning from linear models to neural networks.
    
- Showing empirically, using the D4RL offline RL benchmark, that PreferenceEKF outperforms DeepEnsemble and Dropout baselines in terms of sample efficiency (~42% fewer samples), scalability (Figure 2) and model calibration (Figure 3b) whilst  maintaining or slightly exceeding in trained policy performance.
    
- An open source implementation.

### Strengths
The key strength of this paper, contributing to both its originality and significance, is the PreferenceEKF method which is the first to apply Bayesian subspace filtering for active preference-based reward learning. By extending EKF in a low-dimensional subspace the $\mathcal{O}(|{\theta}|^2)$ posterior scaling is avoided, enabling the use of InfoGain for active preference-based reward learning to scale from linear models to neural networks.

The clarity of the work is high and it’s clear how and why PreferenceEKF enables sampling from high-dimensional distributions to scale InfoGain to neural network models. The experiments are high-quality (e.g., well-structured research questions, appropriate baselines and task selection, multiple runs, hyperparameters reported, ablation analysis etc) and the resulting discussion is also insightful. The limitations are clearly and (to the best of my understanding) faithfully discussed.

I think the potential significance of this work is high given the limitations of linear reward models and the importance of active preference-based reward learning.

### Weaknesses
As noted by the authors, it remains to be determined whether PreferenceEKF will scale to foundation model-scale reward models and this is not considered within the current paper (in fact the models trained are modest at 2x64 - 2x256 hidden units). The evaluation, although thorough, is also limited to the D4RL benchmark. It would be interesting if more could be said about the Pen Human outlier case where active PreferenceEKF "severely underperforms".

The work could be improved further by establishing theoretical bounds or guarantees on Bayesian subspace filtering for active preference-based reward learning. Where this is not possible in the scope of the current work it would nevertheless benefit from signposting the most closely-related or promising theoretical foundations.

Minor:

- I don’t think the ‘RM’ acronym is explicitly defined anywhere (reward model?)
    
- line 226: the footnote (3) is hanging between the end of the previous sentence and the start of the next.

- In A.3 you mention "in the case of the three layer neural networks with 1024 units each" but this seems to be the only reference to such a case.

### Questions
1. Are you able to offer any insights about the Pen Human outlier case where active PreferenceEKF "severely underperforms"?

2. Can you please comment on how significant the jump from linear reward models to modest neural networks is in contrast to foundation-model sized reward models?

### Soundness
3

### Presentation
3

### Contribution
3
