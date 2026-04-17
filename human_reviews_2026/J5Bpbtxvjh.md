# RoCA: Robust Cross-Domain End-to-End Autonomous Driving

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
End-to-end (E2E) autonomous driving has recently emerged as a new paradigm, offering significant potential. However, few studies have looked into the practical challenge of deployment across domains (e.g., cities). Although several works have incorporated Large Language Models (LLMs) to leverage their open-world knowledge, LLMs do not guarantee cross-domain driving performance and may incur prohibitive retraining costs during domain adaptation. In this paper, we propose RoCA, a novel framework for robust cross-domain E2E autonomous driving. RoCA formulates the joint probabilistic distribution over the tokens that encode ego and surrounding vehicle information in the E2E pipeline. Instantiating with a Gaussian process (GP), RoCA learns a set of basis tokens with corresponding trajectories, which span diverse driving scenarios. Then, given any driving scene, it is able to probabilistically infer the future trajectory. By using RoCA together with a base E2E model in source-domain training, we improve the generalizability of the base model, without requiring extra inference computation. In addition, RoCA enables robust adaptation on new target domains, significantly outperforming direct finetuning. We extensively evaluate RoCA on various cross-domain scenarios and show that it achieves strong domain generalization and adaptation performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes RoCA (Robust Cross-domain end-to-end Autonomous driving), a Gaussian Process–based framework that enhances the robustness and adaptability of end-to-end driving models across different domains. Key Contributions: (1) Introduces a probabilistic Gaussian Process module that models joint distributions of ego and agent tokens for trajectory prediction. (2) Builds a learnable codebook of basis tokens and trajectories to improve cross-domain generalization. (3) Enables uncertainty-aware domain adaptation and active learning without heavy retraining. (4)Demonstrates state-of-the-art cross-domain performance on Bench2Drive and nuScenes benchmarks.

### Strengths
1. Cross-domain robustness: Effectively improves generalization across different cities, lighting, and weather conditions.
2. Uncertainty modeling: Provides principled uncertainty estimates that aid in active learning and online adaptation.
3. Plug-and-play design: Can be attached to existing E2E models (e.g., VAD, SparseDrive) without changing their architectures.
4. Strong empirical results: Consistently outperforms baselines and several methods on multiple benchmarks.

### Weaknesses
1. The paper’s “cross-domain” claim is somewhat overstated. While the title suggests broad domain generalization (e.g., across cities, weather, or sensors), the experiments mainly cover cross-city and sim-to-real settings, which represent limited domain shifts.
2. The paper argues that recent LLM/VLM-based E2E driving models do not guarantee cross-domain generalization and may incur high retraining costs. However, this claim is not experimentally supported, the authors do not provide many quantitative comparisons with such models (except for DiMA) under domain-shift scenarios. Given that LLMs and VLMs are often credited with strong open-world or cross-domain capabilities, the lack of empirical evidence weakens this argument and makes the motivation for replacing them with RoCA less convincing.

### Questions
1. Corresponding to W1, the authors are suggested to clearly define the scope of “domain” in this context or add some clarification.
2. Corresponding to W2, the authors are suggested to compare with more LLM/VLM-based methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel end-to-end autonomous driving framework, RoCA, which aims to address the insufficient robustness of E2E models in cross-domain generalization and long-tail scenarios. The core innovation of ROCA lies in the introduction of a Gaussian Process-based module for jointly modeling the probability distribution of Ego and Agent Tokens extracted from the E2E model and their corresponding future trajectories. By learning a set of "basis tokens" representing diverse driving scenarios and their corresponding trajectories, RoCA can perform probabilistic inference for new scenarios and estimate prediction uncertainty.

Extensive experiments demonstrate that RoCA significantly outperforms existing baseline methods and some LLM-based methods in terms of domain generalization and adaptation.

### Strengths
- For the first time, Gaussian processes are introduced as an uncertainty-aware module into the token/feature space of E2E autonomous driving for probabilistic trajectory modeling. This approach provides a theoretically complete Bayesian framework for autonomous driving decision-making, naturally quantifying the uncertainty of predictions.

- Utilizing an active learning strategy guided by Gaussian process prediction variance, performance comparable to or even better than randomly selected 10% or 15% of the target domain data is achieved using only 5% of the target domain data, significantly improving the efficiency of target domain data labeling and adaptation.

- Compelling closed-loop experiments were conducted on the Bench2Drive dataset, achieving promising results. Extensive robustness testing, active learning testing, and ample ablation studies strongly demonstrate the contributions of each component within the framework.

### Weaknesses
ROCA is an add-on module that relies on the quality of the Ego/Agent Tokens output by the Scene Encoder of the baseline E2E model. If the base model's tokens exhibit significant instability and domain shift across domains, ROCA's performance will be limited.

### Questions
1. Learning the Basis Tokens B is central to ROCA. How can you ensure that B doesn't collapse during training to represent only a few common scenarios? Could you provide a t-SNE visualization of B at different training stages (similar to Figure 2, but showing the evolution) to further demonstrate its coverage of diverse driving modes?

2. Figure 2 shows clear token clustering. Could the authors provide more analysis explaining which specific driving behaviors or scenarios (e.g., "sharp turns," "lane changes," etc.) these learned basis token B clusters represent to enhance the interpretability of the method?

3. How does RoCA behave when encountering a completely unseen scenario that far exceeds the coverage of the source domain basis tokens B? In this corner case, will the prediction variance σ² reliably spike, thus correctly indicating a lack of model confidence?

4. Formatting issue: The citation style in the paper does not comply with the conference requirements. Please revise it.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces RoCA, a robust cross-domain end-to-end autonomous driving framework. RoCA learns the joint probabilistic distribution over the tokens in the end-to-end pipeline with a set of basis tokens via Gaussian process. Then it infers the future trajectories with GP-based regression. The framework shows strong domain generalization and adaptation performance.

### Strengths
1. The proposed framework is general and works with several end-to-end models.
2. The motivation of integrating Gaussian process (GP) is clear.

### Weaknesses
1. The closed-loop evaluation in Table 1 should be on the whole validation set (220 routes) of Bench2Drive rather than Dev10 subset. The Dev10 is proposed for quick development and ablation studies. For main results, it is better to evaluate on the whole validation set for comprehensive evaluation and convenient comparation.
2. Although the Table 1 reports the closed-loop results, the experiments of domain adaptation and robustness are all open-loop. It is recommended to measure these significant abilities of RoCA in closed-loop manner too. 
3. The set of basis trajectories are constructed by sampling from dataset and clustering. Thus, the set can only represent the data distribution of the selected dataset rather than the general distribution. It may be hard to formulate the distribution in other domain with the set.
4. The citiation format does not meet the requirements of ICLR that "citations within the text should include the authors’ last names and year".

### Questions
1. In cross-city experiments, is the set of basis trajectories constructed from the data of a single city, or from the whole nuScenes dataset?

### Soundness
2

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
This paper proposes RoCA, a novel framework for robust cross-domain end-to-end autonomous driving. The core idea is to learn a codebook of basis tokens that represent diverse ego and agent states, with each token mapping to a corresponding trajectory. RoCA uses a Gaussian Process (GP) to model the joint probabilistic distribution over these tokens. This GP-based module is then used in two ways: 1) as a regularizer during source-domain training to improve the generalization of a base E2E model, and 2) as an adaptation module to improve performance when transferring to new target domains.

### Strengths
1. The paper tackles the critical and practical problem of cross-domain generalization and adaptation for E2E models, which is a significant barrier to real-world deployment.
2. The approach of using a Gaussian Process over a learned codebook of tokens. This probabilistic formulation allows for principled uncertainty estimation.
3. The framework demonstrates consistent performance improvements across multiple base E2E models, and across various challenging settings, including closed-loop simulation, sim-to-real, cross-city transfer, and robustness to image degradations.

### Weaknesses
1. The paper is hard to follow, and the writing needs to be greatly improved. This is a major barrier to understanding the contribution and evaluating the method's soundness.
    - Key terminologies (e.g., 'codebook', 'basis token', 'ego and agent states') are used extensively in the abstract and introduction before being formally defined in the methodology. The author is expected to define or explain implementations of terminologies when they first appear.
    - The paper lacks a clear, formal problem definition. It jumps into the system overview without first defining the precise inputs (e.g., multi-view images, ego status?) and the target outputs (e.g., ego waypoints, agent trajectories?).
    - The entire method is based on Gaussian Processes, but there is no preliminary section or background provided. A brief mathematical overview of GP regression in the main text would be helpful for readers.
    - The experiment section is dense and difficult to parse. It would be much clearer if it started with a summary of the experimental goals, the base models used, and explained more about the different settings being tested (e.g., source-domain generalization, unsupervised adaptation, active learning) before presenting the tables. Also, it will be clearer to bold the highest scores in tables.

2. The paper is missing critical analysis to justify why the proposed formulation is superior to simpler alternatives.
    - The core claim is that the GP formulation adds robustness. The t-SNE plot (Fig 2) shows better cluster separation, but the paper does not deeply analyze why this probabilistic approach leads to better generalization compared to a deterministic one.
    - The "codebook" of "basis trajectories" sounds very similar to a set of anchors. The paper is missing a crucial comparison to a strong anchor-based method. What is the explicit advantage of this complex GP formulation over a simpler, non-probabilistic method that also learns a codebook of "anchors" (e.g., via k-means or simple end-to-end learning) and predicts a residual? An ablation is needed to isolate the benefit of the GP itself.

I am willing to increase my score if the authors can substantially improve the paper's clarity and address the key questions regarding the method's justification.

### Questions
1. How do you define the problem? What are the precise inputs and outputs for the entire system?
2. How are the initial token embeddings (e and a from the base model) learned, and what information do they carry? How does RoCA handle cases where the base model fails to recognize a new driving scenario and produces a problematic or out-of-distribution token embedding?
3. Why is a set of tokens needed for each group? What are the special characteristics of the learned basis tokens within a single group? How were the group size and the number of groups chosen?
4. How is the group classification performed? The paper mentions an MLP and ground-truth labels. How are these ground-truth group labels obtained?

### Soundness
2

### Presentation
2

### Contribution
3
