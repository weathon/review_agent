# Prior-aware and Context-guided Group Sampling for Active Probabilistic Subsampling

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Subsampling significantly reduces the number of measurements, thereby streamlining data processing and transfer overhead, and shortening acquisition time across diverse real-world applications. The recently introduced Active Deep Probabilistic Subsampling (A-DPS) approach jointly optimizes both the subsampling pattern and the downstream task model, enabling instance- and subject-specific sampling trajectories and effective adaptation to new data at inference time. However, this approach does not fully leverage valuable dataset priors and relies on top-1 sampling, which can impede the optimization process. Herein, we enhance A-DPS by integrating a deterministic (fixed) prior-informed sampling pattern derived from the training dataset, along with group-based sampling via top-k sampling, to achieve more robust optimization—method we call Prior-aware and context-guided Group-based Active DPS (PGA-DPS). We also provide a theoretical analysis supporting improved optimization via group sampling, and validate this with empirical results. We evaluated PGA-DPS on three tasks: classification, image reconstruction, and segmentation, using the MNIST, CIFAR-10, fastMRI knee, and hyperspectral AeroRIT datasets, respectively. In every case, PGA-DPS outperformed A-DPS, DPS, and all other sampling methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Prior-aware and Context-guided Group-based Active Deep Probabilistic Subsampling (PGA-DPS), which enhances Active Deep Probabilistic Subsampling (A-DPS) by incorporating two key innovations: (1) integrating deterministic prior-informed sampling patterns derived from training data statistics with instance-specific active sampling; (2) introducing DPS-top-k group sampling to replace sequential top-1 sampling. The authors provide theoretical analysis showing that group sampling achieves smaller effective Lipschitz constants, leading to smoother optimization landscapes. Experiments on MNIST classification, fastMRI reconstruction, and hyperspectral segmentation demonstrate that PGA-DPS outperforms existing sampling methods across all evaluation metrics.

### Strengths
1. The dual sampling strategy combining deterministic priors with adaptive active sampling is well-motivated and novel. While group sampling is not entirely new, its application and theoretical analysis in this context show originality.

2. The method is technically sound with solid implementation. The theoretical analysis (Theorem 1 on Lipschitz constants) provides valuable insights into optimization stability. The experimental design covers three diverse domains, demonstrating method generality.

3. The paper is well-structured with clear method description and reasonable figure presentations.

### Weaknesses
1. Questionable practical relevance: The MNIST pixel-level sampling has limited real-world significance as pixel acquisition costs are identical and downsampling doesn't provide substantial practical benefits. For MRI and HSI tasks, while acceleration is mentioned, detailed time cost analysis and end-to-end system evaluation are missing.

2. Incomplete experimental evaluation: All experiments lack 100% data sampling baselines, making it impossible to assess performance degradation at different sampling rates. This significantly undermines the assessment of the method's practical value.

3. Limited dataset scale: MNIST is too simple to validate method scalability. With sufficient computational resources, evaluation on CIFAR-10/100 or larger datasets would strengthen the claims. While fastMRI and AeroRIT are more realistic, their scales remain relatively limited.

### Questions
1. Annotation dependency of prior sampling: Prior sampling requires complete training data annotations. How is this handled in annotation-scarce scenarios? How to update prior sampling patterns when new data continuously arrives?

2. Hyperparameter sensitivity: The (Ps, As) ratios vary significantly across tasks. Is there an automatic selection strategy?

Others can be seen in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The main idea of this paper is to augment Deep Probabilisitic Subsampling (DPS) for task adaptive subsampling with Active Deep Probabilistic Subsampling method (A-DPS). The main idea behind the DPS method is to use Gumbel-softmax trick for back-propagation to estimate the gradients. 

In this paper, the proposed method first performs the deterministic subsampling from the training data and then performs the A-DPS method for contextually selecting more samples. Also, another modification proposed is to use DPS-K for sampling instead of DPS-1 in the A-DPS step.

Empirical results on MNIST and MRI datasets show significant improvements over existing baselines.

### Strengths
1. Empirical results show that the usefulness of the proposed method. Significant gains on MNIST and MRI reconstruction dataset.

2. The method is benchmarked against a wide range of alternatives—including greedy algorithms, learned masks (LOUPE), RL methods, and spectral selection approaches proposed for other domains.

### Weaknesses
1. The core algorithm is a combination of already published ideas: deterministic mask learning (training-set prior as used in DPS and LOUPE), sequential active sampling (A-DPS), and top-k/group sampling (DPS-top-k, Gumbel-top-k) in a straightforward manner.

2. The tuning of how many samples come from the prior mask versus active sampling (Ps/As) is hand-picked for different tasks, and the contributions are hyperparameter-dependent.

3. The prior sampling part is not clear, and the key details are hidden away in appendix.

### Questions
1. How does this method compare against Attention guided methods [1]?

2. Expand on section 3.3.  Also, is Theorem 1 a known result or new contribution?

3. How sensitive is the method to the fixed temperature parameter 

4. Can the hyperparameters P_s and A_s (proportions of prior and active sampling) be learned during training instead of fixed?

5. How does the prior-informed deterministic mask compare to purely learned or data-driven static masks in terms of interpretability and robustness?

6. How does the choice of task model (classification vs. reconstruction) affect the optimal proportion of prior vs. active samples?


References:

[1] Shankaranarayana, Sharath M., et al. "Deep Attention-guided Adaptive Subsampling." arXiv preprint arXiv:2510.12376 (2025).

Typos:

Line 017: impedes -> impede
Line 280: accrucay -> accuracy
Line 479: DSP-1 -> DPS-1

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PGA-DPS, a method that integrates a fixed prior sampling mask with group-based active sampling. The proposed approach builds upon DPS and A-DPS. It is evaluated across three tasks: MNIST, MRI reconstruction, and hyperspectral image segmentation.

### Strengths
- Extensive experiments are conducted across three tasks.

- The paper is well-written and easy to follow.

- The proposed method is supported by theoretical analysis.

- The approach is an effective variant of the existing DPS-based methods.

### Weaknesses
- For all experiments, the number of training epochs is fixed according to the implementation details. It is unclear how the performance would change if the best-performing checkpoint were selected instead of the final epoch model.

- In the MRI reconstruction experiments, one setup crops the slices to 208x208, while another uses 320x320 images. The reason for this inconsistency is not explained.

- The ratios for the fixed prior (Ps) and active sampling (As) are central to the proposed method but are heuristically chosen for each task. While the authors present results across multiple Ps and As settings for the MRI reconstruction task, similar analyses are missing for the classification and segmentation tasks. The segmentation task uses (80%, 20%), which differs from MRI, and the MNIST configuration also varies. The search space for these ratios is large, but no principled optimization strategy is provided.

### Questions
- How would the performance change if model selection were based on validation performance (best checkpoint) rather than the final training epoch?

- What is the rationale behind using different image resolutions (208x208 vs. 320x320) across MRI experiments?

- How sensitive is the proposed approach to Ps or As settings in MINST and segmentation tasks?

- Since the choice of Ps and As ratios is crucial, what is the justification for determining these values?

### Soundness
3

### Presentation
3

### Contribution
2
