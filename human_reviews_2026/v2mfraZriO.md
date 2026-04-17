# Fidelity-Aware Data Composition for Robust Robot Generalization

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Generalist robot policies trained on large-scale, visually homogeneous datasets can be susceptible to shortcut learning, which impairs their out-of-distribution (OOD) generalization. While generative data augmentation is a common approach to introduce diversity, it presents a subtle challenge: data composition. Naively mixing real and synthetic data can corrupt the learning signal, as this process often prioritizes visual diversity at the expense of information fidelity. This paper suggests that robust generalization depends on principled, fidelity-aware data composition. We introduce Coherent Information Fidelity Tuning (CIFT), a framework that treats data composition as an optimization problem. CIFT uses a practical proxy for Information Fidelity based on the feature-space geometry of a dataset. This enables the identification of a phase transition, termed the Decoherence Point, where training stability degrades. The framework includes a generative engine, Multi-View Video Augmentation (MVAug), to synthesize a causally disentangled data spectrum for this tuning process. Applying CIFT to policy architectures such as $\pi_0$ and GE-Act improves OOD success rates by over 54\%. 
The datasets used in this study are available in the anonymous repository provided. All model checkpoints will be released in a public repository after the review process to facilitate reproducibility. The anonymous code repository is available at: https://anonymous.4open.science/r/CIFT-code.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the problem of "shortcut learning" in robot policies, where models trained on visually homogeneous data fail to generalize to out-of-distribution (OOD) environments. The authors posit that the solution lies not just in generative data augmentation but in the principled composition of real and synthetic data. They introduce a framework, Coherent Information Fidelity Tuning (CIFT), which formalizes this composition as an optimization problem. The framework consists of two main components: 1) MVAug, a multi-view video-to-video generative model for creating a diverse spectrum of disentangled data, and 2) a composition algorithm that uses a feature-space Signal-to-Noise Ratio (SNR) as a proxy for "Information Fidelity." By analyzing this SNR, CIFT aims to identify an optimal mixing ratio (λ) of real-to-synthetic data that maximizes robustness while avoiding a "Decoherence Point" where training stability collapses. The authors apply CIFT to train π₀ and Diffusion Policy, reporting significant improvements (over 54%) in OOD task success rates on physical robots.

### Strengths
- The paper correctly identifies a critical and subtle challenge in modern robot learning. Moving the focus from just data synthesis to principled data composition is a valuable contribution to the field. The concept of a "Diversity-Information Fidelity trade-off" is insightful and well-articulated.
- The attempt to create a formal, data-driven framework (CIFT) to select the data mixing ratio is commendable. It moves beyond the common practice of ad-hoc hyperparameter tuning and seeks a more systematic solution.
- The authors present a complete system, from a sophisticated generative model (MVAug) to a composition algorithm, and validate it with closed-loop experiments on physical robots. The reported OOD performance improvements are substantial and demonstrate the potential of the underlying idea.

### Weaknesses
Despite its promising direction, the paper is undermined by significant methodological flaws, a lack of critical experimental validation, and poor clarity in key areas. The claims made are not sufficiently supported by the evidence provided.
- The central hypothesis is that CIFT's SNR proxy can identify an optimal mixing ratio λ* for closed-loop policy performance. However, the experiments fail to validate this.
1. The validation of the SNR proxy is done entirely in an open-loop setting (Section 5.2, Figure 5), where the metric of success is the "Robustness Score (RS)," an unvalidated metric based on action prediction MSE. It is well-known in robotics that open-loop performance (i.e., imitation accuracy on a static dataset) is a poor predictor of closed-loop performance, where compounding errors and unexpected states dominate.
2. The on-robot, closed-loop experiments (Table 3) only compare the baseline policy (λ=0, "w/o CIFT") against the policy trained with the CIFT-selected ratio (λ=λ*, "w/ CIFT"). Crucially, they do not evaluate other mixing ratios (e.g., λ values around the supposed "Decoherence Point") in the closed-loop setting. Without this comparison, the authors have only shown that their specific data augmentation is beneficial. They have provided no evidence that their CIFT selection procedure finds a better λ than any other randomly chosen λ, or that the "Decoherence Point" identified via open-loop analysis corresponds to a real collapse in closed-loop task success. This is the single most critical experiment needed to validate the paper's core contribution, and its absence is a fatal flaw.

- The paper is heavily motivated by the goal of solving shortcut learning (i.e., reliance on spurious features). However, it provides no direct evidence that the method actually reduces this phenomenon. Showing improved OOD generalization is not direct proof of it. The authors should have included experiments to investigate what the baseline vs. CIFT-trained policies are paying attention to in ID vs. OOD settings. Alternatively, they could have constructed controlled experiments where a specific spurious cue is present or absent to directly measure the policy's reliance on it. Without this, the claims about how the method works are purely speculative.

- Weak justification and opacity of proposed methods:
1. The paper provides insufficient detail on how the MVAug video-to-video model was trained (Appendix B). Crucially, it omits the dataset used for fine-tuning the Cosmos-Predict2-2B-Video2World foundation model. What data was used for paired multi-view video data? Without this information, the results are not reproducible, and it is impossible to assess whether the generative model's quality unfairly advantages their method over baselines.
2. The justification for using the SNR of the first principal component of features from a generic, pre-trained Inception-v3 model is weak and poorly explained. The text simply states, "Analysis shows a non-monotonic relationship..." but provides no citation or pointer to where this analysis is. Why should the axis of maximum variance in a general-purpose feature space be the most informative signal for the training dynamics of a specific robot policy? It is an unsubstantiated leap of faith.

- The authors report that training a Diffusion Policy required "approximately 80 hours on 16 H100 GPUs." This is an astronomical amount of computation for fine-tuning on a small dataset of 200 real episodes. This implies that the CIFT-composed dataset is enormous, meaning MVAug must generate a massive volume of synthetic data. This is not just a resource issue; it fundamentally questions the method's efficiency and practicality. It suggests the policy's robustness comes from brute-force exposure to an immense quantity of generated data, rather than a finely-tuned "optimal" composition. The paper should be more transparent about the scale of data generation required.

### Questions
- The core validation experiment (Figure 5, Table 1) shows that the peak Robustness Score (RS) occurs at a 100:200 ratio, while the CIFT method selects the 100:100 ratio based on peak SNR. Why should one trust a proxy (SNR) that selects a demonstrably sub-optimal point according to the open-loop evaluation metric (RS)?
- Could you elaborate on the justification for using the first principal component of Inception-v3 features as a proxy for information fidelity? A policy's learned representation is task- and architecture-specific. Why would the primary axis of variance from a generic, pre-trained image classifier be a reliable indicator of the complex learning dynamics and gradient alignment for a visuomotor policy like π₀?

### Soundness
1

### Presentation
2

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
The paper introduces **Coherent Information Fidelity Tuning (CIFT)**, a new framework to optimize the composition of real and synthetic data for robot policy training. CIFT consists of two major components: Multi-View Video Augmentation (MVAug) — a latent diffusion transformer that generates multi-view, causally consistent video demonstrations; Information Fidelity Metric — a quantitative proxy for data quality that measures coherence between real and synthetic data using the feature-space signal-to-noise ratio. Empirical evaluations on real-world robotic manipulation tasks show that CIFT achieves superior performance over existing baselines.

### Strengths
- Using SNR to quantitatively measures the coherence is both interesting and novel.
- The authors conduct comprehensive experiments, including extensive ablation studies, demonstrating the effectiveness of CIFT.
- The Robustness Score of MVAug surpasses that of the baseline methods by a large margin across all mixing ratios, demonstrating the superiority of the generated data.
- The proposed method can be integrated with various pretrained backbones or downstream algorithms, making it broadly applicable across different robot learning setups.

### Weaknesses
- The Feature SNR used as a proxy for robustness does not appear to align well with the actual robustness score, as shown in Figure 5(a).
- The authors state that all code and datasets have been made publicly available in an anonymous repository. However, I was unable to locate or access this repository. I only found that some video demonstrations are included in the supplementary material.

### Questions
- It is interesting to observe that quite different pretrained backbones (Inception-v3, CLIP, DINO-v2) yield exactly the same Decoherence Point. Could the authors provide an explanation for why this happens?
- Could the authors clarify how to access the code and dataset mentioned in the paper?

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
3

### Summary
The paper proposes a unified framework that balances data diversity and realism in robot learning by combining Multi-View Video Augmentation (MVAug) and Coherent Information Fidelity Tuning (CIFT). MVAug is a latent diffusion-based generator that produces physically consistent and multi-view coherent augmented videos through cross-view attention and structure/appearance conditioning, enriching visual diversity without breaking causality. CIFT then analyzes the feature-space Signal-to-Noise Ratio (SNR) to quantify data fidelity and identify the optimal real-to-synthetic mixing ratio. Experiments on simulated and real robots demonstrate that the fidelity-aware composition strategy enhances policy robustness and generalization under unseen visual conditions, though its scalability and reliance on generative model quality remain limitations.

### Strengths
1. The paper proposes a well-founded framework CIFT that balances data diversity and realism. It also helps researchers predict the optimal real/synthetic data ratio without post-training, improving policy generalization and lowering computational cost.
2. This paper establishes a measurable relationship between feature-space SNR and robustness score (RS). SNR provides an interpretable and computationally efficient proxy for assessing data quality before training.
3. The latent diffusion transformer with periodic cross-view attention and structure/appearance conditioning achieves impressive results in maintaining geometric and temporal consistency across views.

### Weaknesses
1. The “decoherence point” $\lambda_{dc}$ is detected empirically from SNR minima, but this paper does not provide a rigorous mathematical criterion or theoretical justification for its general validity.
2. The experimental results are based on a single checkpoint with 20 trials and limited task types, making the improvement in Table 3 less reliable. Additional policy testing results from tasks in MVAug synthesis examples (in Appendix) and details on settings, such as the object's initial position range, should be provided.

### Questions
1. What is the computational overhead (training/inference time) for generating augmented datasets using MVAug?
2. A minor suggestion: in Figure 23, please consider changing “bottom” and “top” to “left” and “right” for clarity.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a multi-view data augmentation approach to achieve better policy performance when finetuned on a target domain and when tested especially under distribution shift. The author propose a multi-view consistent diffusion model that augments the robot demonstration data with different backgrounds, distractors, relighting and foreground objects etc. They propose a metric which guides how much to mix the original data with synthetic data. Both open and close loop experiments are performed with existing foundation models i.e. pi-0 as the base model.

### Strengths
In my opinion, here are the strengths of the paper:

1. A systematic study for robot learning foundation models on finding the feature space SNR correlates well with final post training policy stability.  

2. Significant performance improvement vs the base model finetuning using the augmented dataset proposed by the paper. 

3. The paper is clearly written and easy to follow, and figures complement the text nicely.

### Weaknesses
In my opinion, below are the weaknesses of the paper:

1. While it is great to study open-loop correlation, I am curious if the same correlation holds with the closed-loop SR. It is also not clear if open-loop RS score even correlates well with closed-loop SR; which makes the decisions on mixing ratio being calculated on open-loop RS seems a bit questionable.

2. While it is great to see qualitative results with baseline, I am wondering will the baselines i.e. RoboTransfer etc. would also result in the same performance improvements as the authors see in Table 3 with their proposed augmentations?

3. No statistical error bars are presented, which doesn't show the full variance b/w runs in these generative policies [1]. 

4. The qualitative results for RoboEngine are presented in a bit too extreme case where in every frame the background etc changes, is there a more principled baseline where one could take masked robot embodiment from RoboEngine and applies let's say same lightning to all frames or same color augmentation to foreground objects like shirt in all frames, where the results could look a lot better and cleaner for the baseline than currently presented. 

5. Does the same augmentation strategy hold for pretraining the foundation model as opposed to finetuning where still 200 real-world demos are collected which is a large number. 


[1] TRI LBM Team, A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation, arXiv 2025

### Questions
Please see questions in the weakness section. I look forward to seeing the author's response in rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
