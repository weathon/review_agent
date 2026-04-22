# BEARD: Benchmarking the Adversarial Robustness for Dataset Distillation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Dataset Distillation (DD) compresses large-scale datasets into smaller synthesized datasets, enabling efficient model training while preserving high test performance. However, existing DD methods primarily focus on accuracy and largely neglect adversarial robustness, potentially exposing models to security risks in critical applications. Evaluating robustness is therefore essential but remains challenging due to complex interactions among distillation methods, model architectures, and attack strategies. Moreover, current benchmarks provide only partial coverage and lack a unified perspective in the DD domain. To address this gap, we introduce BEARD, an open and unified benchmark for systematically evaluating the adversarial robustness of models trained on distilled datasets from representative DD methods such as DM, IDM, and BACON. BEARD supports diverse adversarial attacks, including FGSM, PGD, and C&W, and widely used datasets such as CIFAR-10/100 and TinyImageNet. Using an adversarial game framework, we define three key metrics: Robustness Ratio (RR), Attack Efficiency Ratio (AE), and Comprehensive Robustness-Efficiency Index (CREI). We conduct systematic evaluations and analyses across unified benchmarks, varying images-per-class (IPC) settings and adversarial training strategies, showing that dataset distillation consistently enhances adversarial robustness, with adversarial training providing further improvements. The leaderboard is available at https://beard-leaderboard.github.io/, along with a library of model and dataset pools to support reproducible research. Code is accessible at https://anonymous.4open.science/r/BEARD-6B8A/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BEARD, a unified benchmark for evaluating the adversarial robustness of models trained on distilled datasets. It proposes an adversarial game framework and three new metrics (RR, AE, CREI) to systematically assess robustness across multiple datasets, distillation methods, and attack strategies. The benchmark includes a dataset pool, model pool, and leaderboard, with extensive experiments showing that dataset distillation can improve robustness, especially when combined with adversarial training.

### Strengths
- Novel and well-defined evaluation framework and metrics.
- Comprehensive experiments across datasets, methods, and attacks.
- Open-source code and leaderboard support reproducibility and community adoption.

### Weaknesses
- Limited to image classification; does not cover other modalities.
- Does not include very large-scale datasets like ImageNet.
- Some results (e.g., robustness improvements) are not thoroughly analyzed or compared to non-distilled baselines.
- Some metrics (e.g., AST) may be sensitive to hardware and implementation details.

### Questions
1. Why does dataset distillation often improve adversarial robustness? Is it due to implicit regularization or reduced capacity?
2. How does BEARD compare to training on random subsets of the original data?
3. Could the benchmark include more recent distillation methods (e.g., SRe2L, CAFE)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a benchmark for evaluating the robustness of dataset distillation approaches against adversarial examples. The authors wrap several models, datasets, and attacks, and provide a set of metrics to evaluate the adversarial robustness. The wide range of experimental results is also collected in a leaderboard, which is publicly released (together with the implementation code).

### Strengths
- The paper is clear and well written
- The addressed problem is relevant, and I think those benchmarks and their codebase are very valuable for the research community and can serve as a baseline both for attacks and defenses
- The authors made a lot of effort to wrap together models, datasets, and attacks, and run a considerable amount of experiments

### Weaknesses
- I am concerned about the contribution, as it appears weak (particularly considering this venue), both from a technical and novelty point of view. The authors (although I recognize the hard work that has been made) "simply" wrap together existing works, whereas the most novel contribution appears to be the proposed metrics (on which I have some concern, see below). Additionally, there is a non-negligible overlap with the competing DD-RobustBench work, with only incremental improvements over it.
- I don't understand the reason to define relative metrics (RR and AE), as the maximum ASR and AST, which serve as baselines for them, are strongly influenced by several factors (model pool, attack performance, which in turn depends on many parameters, etc.). Why not use absolute metrics, such as an average?
- Using the GPU time to compute the computational cost is not reliable, as this can be influenced by multiple side effects. A suitable metric to compute the attacker cost is the number of model inferences and gradient computations. In this way, both the model itself and other overheads unrelated to the attack are excluded.
- I also have concerns about the attack hyperparameters. Unlike AutoAttack, which is parameter-free (and thus suitable for benchmarking different models), the other approaches require careful tuning of the hyperparameters for each attacked model (e.g., iterations, step size) to achieve the best results. For this reason, the results of those attacks might not be reliable. Moreover, using 10 iterations for PGD is unlikely to lead to convergence of the optimization.

### Questions
- Could you please justify the use of relative metrics (RR and AE), especially considering that adding other models/attacks to the benchmark might lead to recomputing the entire results?

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
2

### Summary
This paper presents the BEARD benchmarking framework to evaluatethe robustness of models trained via dataset distillation against adversarial attacks. The authors point out that existing benchmarks such as DD-RobustBench and RobustBench fail to sufficiently reflect the actual performance of dataset distillation techniques in adversarial scenarios. To address this issue, BEARD integrates multiple dataset distillation techniques, attack methods, and datasets, while introducing three new metrics: RR, AE, and CREI. These metrics employ game-theoretic approaches to simultaneously evaluate attack effectiveness and efficiency. The benchmarking platform is publicly available, featuring leaderboards and a curated model dataset collection to support reproducible research. Experiments demonstrate that dataset distillation significantly enhances adversarial robustness, particularly when combined with adversarial training.

### Strengths
1. The paper presents the first unified benchmark for adversarial robustness in dataset distillation, introducing a novel adversarial game framework and three tailored metrics (RR, AE, CREI).

2. As dataset distillation gains traction in resource-constrained settings, understanding its robustness is critical. BEARD provides a standardized platform for comparative evaluation.

3. The paper is well-structured, with clear descriptions of the framework, metrics, and experimental setup. The public release of code and leaderboard enhances transparency and usability.

4. The benchmark covers multiple DD methods, attack types, and datasets, offering a comprehensive evaluation landscape.

### Weaknesses
1. Completion is slightly insufficient. This paper has systematically expanded and deepened DD-RobustBench through introducing unified evaluation metrics, incorporating more attack types, and proposing a game-theoretic framework. yet it is unable to prove on other larger datasets, more complex architectures and algorithms, and only remains in relatively simple scenarios.
2. In Section 3.10, the CREI metric locks α at 0.5 without explanation. Giving robustness and efficiency equal weight might not suit every task; an ablation on α or a data-driven reason for this choice would help.
3. In Section 5.1, the claim that “dataset distillation improves adversarial robustness” is counter-intuitive and lacks mechanistic explanation. The observed CREI drop with increasing IPC is noted but not interpreted. Include a discussion on why distilled datasets may enhance robustness—e.g., whether they filter out non-robust features or reduce overfitting. Analyze the IPC–robustness trade-off more deeply.
4. In Section 5.1, the performance differences among DD methods (e.g., why DSA/DM/BACON perform better) are reported but not explained. The analysis remains descriptive. Provide hypotheses or further experiments (e.g., feature analysis, robustness curvature) to explain why certain methods excel.
5.In Figure 4 & Figure 5, the paper claims that “distilled datasets improve adversarial robustness,” a conclusion that runs counter to intuition (smaller datasets are usually expected to yield more fragile models) yet no convincing explanation is provided. Discuss the interaction between dataset scale, distillation method, and adversarial training to provide more actionable insights.

### Questions
1. Why was α=0.5 chosen for CREI? Have you experimented with other values, and how sensitive are the rankings to this parameter?

2. Can you provide a deeper explanation for why some DD methods (e.g., DSA, DM, BACON) exhibit stronger adversarial robustness? Is it related to their distillation objectives or synthetic data diversity?

3. The conclusion that “distillation improves robustness” contradicts the common belief that smaller datasets lead to weaker models. Can you discuss potential reasons for this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an open and unified benchmark designed to systematically evaluate the adversarial robustness of models trained via dataset distillation  methods called BEARD. The benchmark covers multiple DD algorithms, adversarial attacks  and widely-used image datasets. The authors formalize an adversarial game framework and employ three key evaluation metrics, Robustness Ratio,  Attack Efficiency Ratio and Comprehensive Robustness-Efficiency Index respectively.

### Strengths
1. The code, leaderboard, and data pools are open-sourced, which can help facilitate future research.

2. The adversarial game formalism is thoughtfully articulated.

### Weaknesses
1. The empirical results do not directly benchmark against some newer strategies for adversarial training (e.g, [1]), adversarial attacks  (transformation-based attacks [2] and generative approachs [3] )and other widely used datasets (e.g., cinic10, imagenet and mnist)

2. Section 5 reports trends but lacks deeper causal explanations (e.g., why DM improves CREI).

3. Section 3 introduces too many mathematical definitions, but provides limited experimental interpretation or discussion later.

4. typo and errors in grammar.  1) "DISCUSSION THE DIFFERENCES BETWEEN BEARD AND OTHER BENCHMARKS" (B.5) -> "THE DIFFERENCES BETWEEN BEARD AND OTHER BENCHMARKS" in  2) Missing space between “IDM” and “BACON” in figure 3. 3) TinyImageNet” or “Tiny-ImageNet? should be consistent. 4) 

5. From the current description, BEARD appears conceptually similarly  to  DD-RobustBench in both purpose and experimental scope, though the authors claim they provide a more holistic assessment.  But RRM does not provide substantial novelty beyond existing robustness evaluation metrics.  And it is easy to integrate target settings in DD-RobustBench. Furthermore, the DD methods, attack methods and  provided in BEARD are also limited.  The paper reads more like an engineering consolidation than a fundamentally new contribution.  I am not sure I understand it correctly.

[1] Yang, Zhuolin, et al. "Trs: Transferability reduced ensemble via promoting gradient diversity and model smoothness." Advances in Neural Information Processing Systems 34 (2021): 17642-17655.

[2] Yun, Zebin, et al. "The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations." Proceedings of the 2024 Workshop on Artificial Intelligence and Security. 2024.

[3] Wei, Zhipeng, et al. "Enhancing the self-universality for transferable targeted attacks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

### Questions
What is the meaning of $|\epsilon|$? Why the authors use $\epsilon= 8/255$ and $|\epsilon|= 8/255$ interchangeably?

### Soundness
2

### Presentation
3

### Contribution
2
