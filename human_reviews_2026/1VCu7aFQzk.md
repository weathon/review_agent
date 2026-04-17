# Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapter

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
When deploying large language models (LLMs) to safety-critical applications, uncertainty quantification (UQ) is of utmost importance to self-assess the reliability of the LLM-based decisions. However, such decisions typically suffer from overconfidence, particularly after parameter-efficient fine-tuning (PEFT) for downstream domain-specific tasks with limited data.
To address these limitations,  we build on the Bayesian last layer (BLL) model, where the LLM-based ${\it deterministic}$ feature extractor is followed by random LL parameters for uncertainty reasoning. 
Since existing low-rank adapters (LoRA) for PEFT have limited expressiveness due to rank collapse, we address this with Polar-decomposed Low-rank Adapter Representation (PoLAR), an orthogonalized parameterization paired with Riemannian optimization to enable more stable and expressive adaptation.
The resulting PoLAR-VBLL is a flexible framework that nicely integrates architecture-enhanced optimization with scalable Bayesian inference to endow LLMs with well-calibrated UQ.
Our empirical results verify the effectiveness of PoLAR-VBLL in terms of generalization and uncertainty estimation on both in-distribution and out-of-distribution data for various common-sense reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes PoLAR-VBLL, a principled framework that integrates Polar-decomposed Low-rank Adapter Representation (PoLAR) with a Variational Bayesian Last Layer (VBLL) for uncertainty-aware fine-tuning of large language models (LLMs). The central motivation stems from two widely-known observations: Ii) Fine-tuned LLMs tend to be overconfident, especially after parameter-efficient fine-tuning (PEFT) on small, domain-specific datasets; and (ii) Existing PEFT methods such as LoRA suffer from rank collapse, limiting their expressiveness and thereby harming calibration and generalization. Empirical evaluations on common-sense reasoning and domain generalization tasks show that PoLAR-VBLL consistently outperforms existing methods (LoRA, BLoB).

### Strengths
+ The paper addresses a highly relevant challenge: quantifying uncertainty in fine-tuned LLMs for safety-critical domains (e.g., healthcare, legal, and autonomous systems). This is a crucial step toward trustworthy LLM deployment.

+ The discussion on rank collapse in LoRA is also a widely recognized phenomenon. Exploration of polar decomposition with orthogonality constraints and Riemannian optimization is definitely helpful.

### Weaknesses
- The paper lacks a convincing formal analysis of how orthogonality in PoLAR enhances uncertainty calibration.

- The paper does not compare conceptually or empirically against SOTA on Bayesian LORA techniques. For example, https://dl.acm.org/doi/10.5555/3762387.3762543

### Questions
Can authors include experiments with more recent methods, such as https://dl.acm.org/doi/10.5555/3762387.3762543 and add discussion on additional parameters needed by PoLAR vs other techniques? The reviewer is ready to raise the score if the paper can argue for demonstrated improvement over SOTA Bayesian LORA techniques.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for Bayesian fine-tuning of LLMs that integrates a variational Bayesian last layer with an orthogonalized low-rank adapter. The goal is to achieve parameter-efficient adaptation with well-calibrated uncertainty estimates.

### Strengths
- The combination of VBLL (analytic variational training) and orthogonalized adapters (PoLAR) appears mathematically sound and computationally efficient.
- The Jensen-tightened ELBO yields a closed-form expression for expectations under Gaussian, avoiding additional Monte Carlo sampling during training.
- The paper uses an efficient landing-field update rule to maintain orthogonality in the adapter weights, avoiding the costly retraction steps used in standard manifold optimization.
- Across several reasoning and OOD tasks, the method outperforms similar approaches in both calibration and accuracy.
- The study carefully separates the contributions of PoLAR, VBLL, and the Laplace refinement, providing clear evidence for each component’s effect.

### Weaknesses
- The paper makes the ELBO tractable by applying Jensen’s inequality to the log-sum-exp term, leading to a convenient closed-form objective. However, this simplification produces a rather loose approximation for softmax models, which likely explains the need for an additional Laplace correction.
- After substituting Laplace covariances $\Sigma_c$ for $S_c$, the model no longer optimizes a coherent variational objective, becoming a two-stage hybrid (VB + Laplace) approximation.
- The posterior factorization $q(\Theta) = \prod_c \mathcal{N}(\theta_c; \mu_c, S_c)$ ignores cross-class correlations induced by the softmax, which may yield overconfident predictions.
- All results are obtained on a single LLM backbone (Llama-2-7B) and only for classification tasks. This is a narrow evaluation for a paper claiming general "LLM fine-tuning."
- The framework claims $O(C d^2)$ memory due to per-class covariances $S_c$. This can become substantial for practical feature dimensions and a moderate number of classes.

### Questions
#### Questions
1. After Laplace substitution, is there still a formal ELBO objective, or is the procedure purely post-hoc?
2. How sensitive are the results to the choice of prior or initialization of the variational parameters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper integrates the Bayesian Last Layer (BLL) framework with the Polar-decomposed Low-rank Adapter Representation (PoLAR) to enable effective and efficient Bayesian fine-tuning of LLMs. In addition, the authors introduce a post-hoc LA strategy to further optimize the learned covariance matrix. The results show that the proposed method achieves comparable performance to baseline method while reducing additional training and inference costs.

### Strengths
* The organization of the paper is clear.
* The proposed method is simple and effective. 
* The paper effectively mitigates the high computational cost of Bayesian fine-tuning while maintaining comparable performance.

### Weaknesses
* The experimental details are insufficient. For baselines such as BLoB, the authors state that “we report the better result between our reproduced numbers and those seen in BLoB.” However, the experimental settings used to reproduce the baselines are not clearly described.

* The proposed method is essentially a direct combination of BLL and PoLAR. If the authors used standard LoRA for other baselines, then the observed performance improvement may mainly stem from the effective training brought by PoLAR. It remains unclear how much performance gain individually comes from BLL and PoLAR, as relevant ablation studies are missing.

### Questions
* The computational cost analysis in Table 4 appears unusual. BLoB typically exhibits nearly the same memory usage as the standard LoRA when using the same LoRA rank $r$. Did the authors use the same LoRA rank $r$ for all three methods? Relevant details are lacking. The authors should also specify whether Tables 4 corresponds to the inference or training stage.

* The content of line 950 is missing.

* Do the authors train on all the datasets for 500 epochs? The description in line 944 is ambiguous.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes PoLAR-VBLL, a scalable variational Bayesian fine-tuning framework for LLMs that improves UQ and calibration. It combines a Polar-decomposed Low-Rank Adapter Representation (PoLAR) with Variational Bayesian Last Layer (VBLL), which models uncertainty in the classifier weights through analytical variational inference (instead of MC sampling). The framework jointly optimizes PoLAR parameters and the variational posterior for efficient Bayesian fine-tuning, optionally refined by a post-hoc Laplace Approximation. Experiments on common-sense reasoning tasks (e.g., Winogrande, ARC, OBQA) show that PoLAR-VBLL consistently achieves higher accuracy and better calibration (lower ECE and NLL) than prior UQ methods like BLoB and Laplace-LoRA, while being more memory- and computation-efficient, thus enabling reliable and scalable uncertainty-aware LLM deployment.

### Strengths
+ **Good writing.** The paper is presented in a clear and concise way, making it most accessible to the audience. 
+ **Empirical effectiveness.** The experiments are (somewhat) extensive (while missing two common sub-datasets as in a standard setting), covering multiple reasoning datasets and both in- and out-of-distribution settings, demonstrating consistent improvements in accuracy, calibration (ECE/NLL), and robustness over state-of-the-art baselines such as BLoB and LAP.
+ **Methodological integration.** The paper presents a well-motivated and elegant combination of PoLAR and VBLL, effectively bridging PEFT with scalable Bayesian UQ methods.

### Weaknesses
+ **Lack of Technical Novelties.** It looks like this paper is a combination of three existing techniques: (I) Polar-decomposed Low-Rank Adapter Representation (PoLAR); (II) Variational Bayesian Last Layer (VBLL); and (III) Laplace Approximation (LA). All three techniques are well established: PoLAR is a major improvement over the vanilla LoRA method; VBLL is an "exact and non-MC sampling-based" variational inference framework (which stabilizes the training of VI, while suffers from its loose bound); LA is a well-studied method for the Bayesian Inference. Section 3.1 and 3.2 simply repeat the content from the original work and I could not find contribution made by this paper other than applying VBLL and LA to PoLAR. This is my major concern. 
+ **Lack of Clear Motivation.** What is the major motivation of this paper? I find it hard to be persuaded by the claim *"(BLoB) require expensive Monte Carlo sampling with prohibitive memory overhead that scales poorly with model size, making them impractical for large-scale deployment."* In BLoB [1], the MC sampling size during training is set to $K=1$ and produces almost no extra computational overhead. During testing, this paper uses the same MC sampling scheme (as in Eq. 14) and have no advantage over the other baseline methods that rely on BMA. Hence I think the actual problems solved with this paper needs to be further clarified. Besides, in the original paper of BLoB and its subsequent work TFB [2], the authors studied the application of last-layer Bayesianization (while it's not the exact VBLL) and showing even better performance in terms of sample efficiency and calibration. The edge of this paper needs to be established upon the comparison with the variants.
+ **Lack of Sufficient Ablative Study.** VBLL might not be the working factor that causes the success of the paper. In fact, the loose bound of VBLL derived from Jensen's Inequality may cause the whole method fail, which is remedied by the final Laplace Approximation. I would like to see the performance of just PoLAR and LA.  
+ **Lack of Most Recent Baselines.** The following recent baselines accepted at NeurIPS 2025 need to be considered: 
  + TFB [2]
  + C-LoRA [3]

**References**
- [1] Wang, Yibin, et al. "Blob: Bayesian low-rank adaptation by backpropagation for large language models." Advances in Neural Information Processing Systems 37 (2024): 67758-67794.
- [2] Shi, Haizhou, et al. "Training-free bayesianization for low-rank adapters of large language models." arXiv preprint arXiv:2412.05723 (2024).
- [3] Rahmati, Amir Hossein, et al. "C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models." arXiv preprint arXiv:2505.17773 (2025).

### Questions
See above (Weaknesses).

### Soundness
2

### Presentation
3

### Contribution
2
