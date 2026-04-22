# EigenScore: OOD Detection using Posterior Covariance in Diffusion Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 2

## Abstract
Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems in safety-sensitive domains. Diffusion models have recently emerged as powerful generative models, capable of capturing complex data distributions through iterative denoising. Building on this progress, recent work has explored their potential for OOD detection.
We propose *EigenScore*, a new OOD detection method that leverages the eigenvalue spectrum of the posterior covariance induced by a diffusion model.
We argue that posterior covariance provides a consistent signal of distribution shift, leading to larger trace and leading eigenvalues on OOD inputs, yielding a clear spectral signature. We further provide analysis explicitly linking posterior covariance to distribution mismatch, establishing it as a reliable signal for OOD detection.
To ensure tractability, we adopt a Jacobian-free subspace iteration method to estimate the leading eigenvalues using only forward evaluations of the denoiser.
Empirically, EigenScore achieves state-of-the-art  performance, with up to 2% AUROC improvement over the best baseline. Notably, it remains robust in near-OOD settings such as CIFAR-10 vs CIFAR-100, where existing diffusion-based methods often fail.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes to adopt the posterior covariance in diffusion models for out-of-distribution (OOD) detection. To be specific, through theoretical and empirical results, earlier diffusion-based scores, like likelihood and score dynamics, are shown to be not reliable enough. Then, in the proposed EigenScore, during the reverse diffusion process, the aggregated eigenvalues of the intermediate diffused results are leveraged to capture the differences between OOD and InD (in-distribution) as a detection score. Experiments verify the effectiveness of EigenScore in some small-scale low-resolution images.

### Strengths
1.	Exploring the posterior covariance in the reverse diffusion process is novel and interesting, which contributes new insights into diffusion-based OOD detection.
2.	Extensive theoretical results are provided, which is appreciated.
3.	To enhance the computational complexity, acceleration techniques are introduced into the proposed EigenScore

### Weaknesses
1.	About the validation part of EigenScore  

The proposed EigenScore requires a validation set, which further requires OOD samples to tune some hyper-parameters such as the selected time steps. In this sense, the selection of OOD samples directly affects the evaluation fairness. That is, if the OOD samples in the validation set are exactly from the same dataset in the test phase, then such a setup is not a fair one for comparisons, since EigenScore has “saw” OOD samples during validation. The authors are suggested to specify how the OOD samples in the validation set are collected. Those OOD samples for validation should be from an additional dataset that is different from those in the test phase.

2.	Experiments are not sufficient enough  

The included datasets in experiments are all small-scale low-resolution image datasets, such as the 32x32 CIFAR10 and CIFAR100. Nevertheless, note that the powerful generation ability of diffusion models is mainly reflected on large-scale high-resolution images. Besides, in existing OOD detection methods, experiments on the large-scale ImageNet-1K dataset are essential. Considering the two aspects, it is highly recommended for the authors to supplement results on the large-scale 224x224 ImageNet-1K dataset in order to better demonstrate the spectrum differences between InD and OOD from diffusion models, which could provide an even comprehensive evaluation and further strengthen this work.

3.	Some reconstruction-based OOD detection methods that do not rely on generative models should be cited and discussed for an enriched literature review. For example, in [a-c], proper subspaces are identified via (kernel) PCA and reconstructions are executed in features and gradients to distinguish OOD from InD. It would be much appreciated to also provide comparisons with [a-c].

[a] Revisit pca-based technique for out-of-distribution detection. ICCV 2023.  
[b] Kernel PCA for out-of-distribution detection. NeurIPS 2024.  
[c] GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients. NeurIPS 2023.

### Questions
Questions correspond to the three weaknesses above. I will raise my rating if all the concerns are well addressed.

### Soundness
1

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
4

### Summary
This paper introduces EigenScore, a diffusion-based method for out-of-distribution detection. The core idea is that when a denoiser trained on in-distribution data is evaluated on OOD inputs, its denoising error increases. The authors leverage the fact that the MMSE denoising error corresponds to the expected total posterior variance of the clean image, and that this posterior variance can be expressed through the denoiser’s Jacobian via Miyasawa’s identity. Since the trace of the covariance equals the sum of its eigenvalues, inflation of the leading eigenvalues provides a practical surrogate for detecting excess uncertainty induced by distribution shift. Based on this intuition, EigenScore estimates the top K eigenvalues at selected low-to-medium noise levels and aggregates them into an OOD score. Experiments on standard benchmarks demonstrate improved behavior over norm-based diffusion OOD metrics, with consistently strong performance and particularly clear gains in near-OOD scenarios.

### Strengths
- **Novelty and strong theoretical motivation**.

     The paper presents a new OOD detection metric based on the spectral properties of the diffusion denoiser’s Jacobian. The theoretical path, from excess denoising error to posterior variance and then to dominant eigenvalues, is well established and provides a compelling motivation for the method.

- **Clear and well-structured presentation**.

     The paper is easy to follow, with a logical progression from the theoretical insights to the algorithmic design and experimental validation.

- **Consistent and competitive empirical performance**.

     The method performs reliably across a variety of OOD detection scenarios, showing steady improvements over diffusion norm-based baselines. These results suggest that the proposed spectral metric captures a stable indicator of distribution shift.

### Weaknesses
- **Sensitivity to multiple hyperparameters.**

    The method depends on several tunable parameters (K, T, I, aggregation rule, finite-difference step size c) and performance appears fragile when deviating from tuned configurations.

- **Tuning strategy may limit real-world applicability.**

    Hyperparameters are selected using ID–OOD validation splits, which assumes prior knowledge of the exact OOD distribution being evaluated. This setup does not reflect realistic deployment scenarios where the nature of the shift is unknown. Demonstrating strong performance with a single default configuration that does not rely on ID–OOD validation would improve the method’s practical usability.

- **Computational efficiency not addressed.**

    Repeated denoiser evaluations and eigenvalue estimation introduce substantial computational overhead, yet there is no runtime or throughput comparison with alternative detectors.

- **Limited evaluation scope.**

    The experiments use relatively small and low-resolution datasets. The effectiveness of EigenScore on larger-scale OOD tasks, for example ImageNet-based settings, remains unverified.

### Questions
1. Hyperparameters such as K, T, I, and the finite-difference step size c appear to be tuned specifically for each ID–OOD setting. Could you report results using a single fixed configuration across all experiments to better evaluate generalization when the nature of the OOD data is not known in advance.

2. Could you include runtime and memory comparisons with classifier-based or other diffusion baselines.

3. The finite-difference step size c influences Jacobian estimation quality and numerical stability, yet its impact is not analyzed. Can you comment on how sensitive EigenScore is to this parameter.

4. Could you provide a denser analysis over diffusion timesteps to better understand which parts of the trajectory contribute most to OOD separation. For example, reporting AUROC as a function of t on a finer noise grid, and including a heatmap over (t × K) to visualize where additional eigenvalues help versus hurt. Evaluating whether restricting the score to the most informative t values improves robustness would also clarify the design choices for T.

5. How does EigenScore perform on more complex and high-resolution OOD benchmarks, for example ImageNet vs Places365 or SUN (OpenOOD).

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
The authors propose EIGENSCORE, an unsupervised framework for out-of-distribution (OOD) detection using uncertainty estimates derived from the posterior covariance of diffusion models during denoising. They highlight the limitations of likelihood and score-based OOD methods and establish a connection between the KL divergence of in-distribution (InD) and OOD data and the MSE of the optimal denoiser. Since denoising error (MSE) is higher for OOD inputs, they express this MSE as the trace of the conditional (posterior) covariance, which is approximated via the Jacobian of the denoiser using the Miyasawa identity. Assuming a symmetric positive semi-definite Jacobian, they perform an eigenvalue decomposition and use the inflation of its trace (the sum of eigenvalues) as the OOD metric for several timesteps. They further show that top-K eigenvalues provide stronger OOD discrimination than using all eigenvalues. EIGENSCORE achieves competitive AUROC results across multiple datasets (CIFAR-10, CIFAR-100, CelebA, SVHN) and shows promising performance on near-OOD detection, supported by ablations over key hyperparameters.

### Strengths
The paper clearly lists the limitations of existing diffusion-based OOD frameworks with evidence in near-OOD scenarios as well. and highlighted theoretically why the proposed method maintains consistent ordering in tasks. 


The paper first lists the existing work, their limitations with evidence, an overview of their method, and a logical flow of the framework supported by theory and empirical evidence.

The paper combines existing theoretical concepts and intuition about covariance and eigenvalues, and applies it for diffusion posteriors.


The method performs better compared to different SOTA OOD detection framework paradigms. and shows promising results in a near-OOD scenario, an improvement compared to the previous diffusion-based SOTA, DiffPath.

### Weaknesses
It seems, the theory, proposition 1, and equations 5-8, which directly talk about MSE instead of top-K, predict good performance, but in Table 4, MSE performs poorly in some tasks (worse than random). This is a disconnect between prop 1 and equations 5-8, and lemma 1 and Proposition 2. Lemma 1 attempts to justify use of top-k but difference in performance on MSE (supported by prop 1) and top-K (supported by lemma 1) seems big. Can you please clarify?


The assumption that covariance or jacobian of diffusion model is symmetric positive semi-definite might be strong. Can you please clarify?


The method involves explicit training of diffusion model on InD data for OOD detection, whereas previous SOTA, DiffPath, uses diffusion model trained on different/generic dataset (ImageNet or CelebA). 


Comparing table 1 (main result) and table 3 (ablation), for C100 vs CelebA, there is a difference in hyperparameter setting (timesteps are different). C100 vs CelebA in table 3 performs poorly compared to the same pair in table 1. Table 1 for this pair uses high noise levels (450, 500) and table 3 is for (100-300). For the same pair, MSE performs better compared to EIGENSCORE in table 4, which for higher noise values, according to lemma 1, should perform badly. More analysis on this could help. 


C100 vs CelebA in table 1 uses higher timesteps (450, 500) as compared to other input pairs (100-300), is there any hyperparameter sensitivity? Does higher noise help better performance in table 1?


Why does MSE fail so catastrophically (AUROC < 0.5) when theory predicts it should work? Can you please provide an explanation that reconciles Proposition 1 with Table 4?


Can  you please mention the diffusion models used?


DiffPath gets 0.328 AUROC, which is way less than their original paper, any insights would be great.

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an unsupervised OOD detector for diffusion models that uses the leading eigenvalues of the posterior covariance. 
EigenScore leverages the covariance structure of the denoising process to capture uncertainty signals, which is theoretically grounded and interpretable.
Empirically, EigenScore achieves state-of-the-art performance

### Strengths
1. Generally clear and readable; 

2. figures and tables are informative; 

3. Codes are publicly available.

4. Investigate OOD detection for Duffusion models is timely.

5. the use of Jacobian-free eigenvalue estimation algorithm is new to me.

### Weaknesses
1. lack of experiments on large-scale dataset e.g. ImageNet-1K
2. when c10 and c100 as ID data, SVHN LSUN iSUN Textures Places365 are standard OOD dataset in the literature of OOD detection. but the corresponding results are missing
3. lack of time computation analysis since QR requires O(n^3) computation complexity
4. this paper only introduce 1 baseline published on/after 2024. more advanced baseline should be included.
5. given a ID dataset, the optimizied hyper-parameter setting are not identical for OOD datasets.
6. The paper  mostly restates these connections and asserts that “inflation” will occur OOD; but formal conditions ensuring consistent ordering for individual samples (beyond in-expectation statements) are not established.

### Questions
see weakness

### Soundness
1

### Presentation
2

### Contribution
2
