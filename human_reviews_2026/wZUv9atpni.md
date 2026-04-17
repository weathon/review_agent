# SAMOSA: Sharpness Aware Minimization for Open Set Active learning

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Modern machine learning solutions require extensive data collection where labeling remains costly. To reduce this burden, open set active learning approaches aim to select informative samples from a large pool of unlabeled dataset that includes irrelevant or unknown classes. In this context, we propose Sharpness Aware Minimization for Open Set Active Learning (SAMOSA) as an effective querying algorithm. Building on theoretical findings concerning the impact that data typicality has on generalization properties of traditional SGD and sharpness-aware minimization, SAMOSA actively queries samples based on their typicality. SAMOSA effectively identifies atypical samples that belong to regions of the embedding manifold clustered close to the model decision boundaries. Therefore, SAMOSA prioritizes the samples which are both highly informative for the targeted classes, and useful for distinguishing between targeted and unwanted classes.
Extensive experiments show that SAMOSA achieves up to 3\% accuracy improvement over the state-of-the-art across several datasets. The source code of our experiments is available at \url{https://anonymous.4open.science/r/samosa-EF8C}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SAMOSA, an inconsistency-based querying strategy that leverages the prediction discrepancy between models trained with SGD and Sharpness-Aware Minimization (SAM). The inconsistency measure, termed SAMIS-P, is defined as the $L_1$ difference between the output probabilities of the two models. SAMOSA adopts a two-step querying framework: first, both models act as $(K+1)$-class classifiers, where the additional class helps filter out potential unknown samples; second, samples with higher SAMIS-P scores are prioritized for labeling. Extensive experiments on CIFAR10, CIFAR100, and TinyImageNet show that the proposed method performs well.

### Strengths
1. This paper attempts to provide a new perspective on open-set active learning from a theoretical point of view.
2. The paper provides high-quality visualization results.

### Weaknesses
1. The entropy-based experiments in this paper seem somewhat problematic. Normally, in active learning, the model tends to select samples with higher entropy. However, as shown in Figure 3, the selected samples actually have lower entropy values. Similarly, in Figure 1(c), the LfOSA method is not uncertainty-based; it instead favors samples that are most likely to belong to known classes, i.e., it is purity-based. In Figure 1(b), the samples visualized by the authors also do not appear to correspond to those with high entropy values.

2. The authors essentially adopt a typical inconsistency-based sampling strategy commonly used in active learning. What is relatively new is their emphasis on training models using SAM. However, it is unclear what the particular advantage of SAM is in this context—it seems mainly used to increase the prediction discrepancy between the two models by altering the training procedure.

3. A key challenge in open-set active learning is how to distinguish hard samples within known classes from unknown-class samples. In this regard, the paper does not provide any new insights.

4. The authors only include EOAL as the most recent baseline, lacking comparisons with other newer state-of-the-art methods. Moreover, the reported results seem somewhat questionable — in Table 1, EOAL performs worse than LfOSA in several cases, which raises some doubts.

5. In Figure 4, the curves of the compared methods are not clearly presented; in some cases, certain baselines even appear to perform better, but the visualization makes this unclear.

### Questions
Please see the Weaknesses.

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
4

### Summary
This paper proposes SAMOSA, a method for open-set active learning that leverages the difference in predicted probabilities between a model trained with SGD and one trained with sharpness-aware minimization (SAM). This discrepancy, called SAMIS-P, is proposed as a proxy for data atypicality. The approach iteratively selects unlabeled samples with the highest SAMIS-P scores for labeling. Extensive experiments on CIFAR10, CIFAR100, and TinyImageNet show modest accuracy improvements over several baselines.

### Strengths
S1. **Clear problem motivation.** The paper correctly identifies the limitation of entropy-based sampling under open-set settings, where uncertainty measures become unreliable due to distributional shift.

S2. **Solid experimental setup.** Multiple datasets (CIFAR-10, CIFAR-100, TinyImageNet) and several baselines are used. The results are consistently positive, suggesting that the proposed typicality metric is at least empirically stable.

S3. **Readable presentation.** The figures and method overview are clear, and the paper is well written overall. It is easy to understand the intuition and pipeline of the approach.

### Weaknesses
W1. **Limited novelty in methodology.** The approach primarily combines two existing techniques, SAM optimization and active learning, in a quiet straightforward way. The “typicality discrepancy” is a small conceptual extension rather than a fundamentally new algorithmic idea. The authors should more emphasize why SAM optimization has a large synergy when applied to open set active learning domain.

### Questions
I have no question.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces SAMOSA, an open-set active learning (OSAL) method that scores unlabeled samples by the probability gap between SAM and SGD and filters unknowns with a K+1 head, prioritizing atypical, boundary-proximal examples. It is theoretically motivated that SGD–SAM predictions diverge on atypical samples but not on typical ones, and this is validated via embedding analyses across training. It outperforms many OSAL baselines by up to 2–3% on CIFAR/Tiny-Imagenet datasets with comparable runtime.

### Strengths
- Theory-to-practice link: a simple, label-free atypicality proxy (|pSAM−pSGD|) that correlates with decision-boundary informativeness.
- Empirical strength: multi-dataset gains, boundary visualizations, ablations (OOD selection value, ensemble vs. SAM effect), and robustness to label noise.
- Practicality and efficiency: easy to plug in (two standard models with a K+1 head) and competitive runtime comparable to strong OSAL baselines.

### Weaknesses
Unclear Motivation

- It is unclear whether Fig 2 and 3 convincingly demonstrate that SAMOSA is a better OSAL metric compared to entropy. A good open-set AL metric should effectively filter out OOD samples while selecting informative in-distribution (IN) samples. Showing the proportion of true IN vs. OOD samples among the selected points would better highlight SAMOSA’s efficacy.
- Even after carefully reading the Introduction and Section 2, the reviewer remains unclear about the high-level intuition behind why the L1 norm between ( f_{SAM}(x) ) and ( f_{SGD}(x) ) leads to better OSAL sample selection.

Questionable Experimental Setup

- The experiments rely solely on synthetic datasets derived from CIFAR-10/100 and Tiny-ImageNet (via random class splits). Evaluation on a real open-set noise dataset would provide stronger evidence.
- There are no results for 0% mismatch ratios. Since real-world data collection often occurs in very low-noise settings, it is important to verify whether SAMOSA remains effective under such conditions.

### Questions
Open-set noise can arise in various domains beyond vision, such as NLP. It would be interesting to see whether SAMOSA still remains effective in non-vision domains.

### Soundness
2

### Presentation
3

### Contribution
3
