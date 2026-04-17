# BERNOULLI FLOW MODELS

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Diffusion-based generative modeling for data with Bernoulli distributions has broad potential applications, but it relies on carefully designed forward processes. Recently, flow matching-based methods have addressed this issue. However, when these methods are naively applied to the Bernoulli distribution, their dependence on predicting the instantaneous velocity field during sampling can introduce invalid Bernoulli parameters, leading to model collapse. To address this challenge, we introduce **Bernoulli Flow Models (BFM)**, a novel generative framework that fuses flow matching with vanilla binary diffusion. BFM ensures valid Bernoulli parameters throughout the sampling process by deriving a one-step forward transition kernel and a closed-form, normalized posterior based on the pre-defined flow-matching probability path in the Bernoulli parameter space. As a result, BFM simplifies the training process of current binary diffusion models and can be easily integrated into existing  architectures with minimal modification. We empirically validate the generative performance of BFM on high-dimensional binary manifolds, including Ising model simulations, both unconditional and conditional image generation. Experiments show that our model achieves comparable performance to both continuous and discrete space generative models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Binary Flow Matching (BFM), a new training loss for Binary Diffusion Models that draws inspiration from Flow Matching (FM). Specifically, BFM modifies the noise scheduling strategy of Binary Latent Diffusion Models (BLM) [1] to follow a linear interpolation between the data and noise distributions, analogous to flow matching in continuous diffusion frameworks.

### Strengths
1. **Extensive Experimental Evaluation**  
   The authors conduct a comprehensive set of experiments on high-dimensional datasets.  

2. **Novel Integration of Flow Matching Concepts**  
   Incorporating the principles of flow matching into binary diffusion models is an novel idea that extends the applicability of flow-based training to discrete domains.  

3. **Clear and Accessible Writing**  
   The paper is generally well-structured and easy to follow.

### Weaknesses
1. **Unclear Motivation for Flow Matching Noise Scheduler**  
   The rationale for replacing the standard BLM noise scheduling [1] with a flow-matching-inspired one is not well articulated. It remains unclear why the new schedule is expected to improve over the established BLM formulation.  

2. **Underwhelming Empirical Performance**  
   The main results indicate that BFM underperforms relative to BLM, raising doubts about its practical utility. Further discussion is needed to clarify potential advantages (e.g., interpretability, stability, or low-NFE behavior) that could justify the approach.  

3. **Lack of Direct Comparisons with BLM**  
   Given that BFM is a close variant of BLM, a side-by-side comparison under identical conditions (same architecture, datasets, and training budget) would be highly informative for evaluating the claimed improvements.

### Questions
1. **Generalization Beyond Gaussian Transitions**  
   Flow matching enables transformations between arbitrary distributions, not limited to Gaussian-based processes as in diffusion models. Does BFM similarly generalize BLM to handle arbitrary unknown initial distributions, potentially enabling tasks such as image-to-image translation?  

2. **Clarify Technical Motivation**  
   The motivation for BFM remains vague, as the proposed formulation still relies on transition kernels, and the primary difference lies in the variable ordering rather than a fundamental theoretical innovation. A deeper discussion of what this change accomplishes in practice would strengthen the paper.  

3. **Low-NFE Performance Analysis**  
   Flow matching methods are known to perform well in low NFE (number of function evaluations) regimes. It would be valuable to investigate whether BFM exhibits similar behavior compared to BLM, potentially highlighting cases where it provides computational benefits despite lower overall accuracy.

---

## References  
[1] **Binary Latent Diffusion Models.**  
[2] **Flow Matching for Generative Modeling.**

### Soundness
1

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
The paper proposes Bernoulli Flow Models (BFM), a binary generative framework that defines a probability path in Bernoulli space and derives a closed-form forward kernel and posterior. This avoids invalid Bernoulli parameters when applying standard flow matching to binary data. Overall, BFM is a principled, practical alternative to heuristic binary diffusion and discrete score-based / flow-based methods, with clear theory and solid results.

### Strengths
1. BFM unifies flow matching with diffusion in the discrete domain,and guarantees all intermediate parameters valid.It is stable on binary systems,suggesting good robustness and cross-domain transferability.

2. A major strength is the derivation of an analytical one-step transition and posterior for the Bernoulli diffusion process.

3. Empirically, BFM shows competitive results across diverse tasks, often with advantages in efficiency.And it’s engineering friendly,both training and sampling are operationally lightweight.

### Weaknesses
1. The claimed advantages on Ising-like systems are under-substantiated: evaluation is confined to a single small 2D Ising setting with limited metrics and baselines. Moreover, the paper does not clearly demonstrate superiority on binary data.

2. While BFM performs well, it does not decisively outperform some specialized prior models in terms of raw image fidelity on certain benchmarks.For instance, on high-resolution image synthesis (LSUN/FFHQ in Table 2), BLD (Wang et al., 2023) still achieves significantly better FID scores (e.g., 5.85 vs BFM’s 10.87 on FFHQ).

3. Scope Limited to Binary Data: By design, BFM focuses on Bernoulli distributions. This specialization is logical, but it means the method is not directly applicable to non-binary categorical data (except by binarizing them).

4. The paper lacks a systematic ablation and sensitivity analysis of the probability-path design , temperature, and step count.

Ze Wang, Jiang Wang, Zicheng Liu, and Qiang Qiu. Binary latent diffusion. In Proceedings of the 
IEEE/CVF conference on computer vision and pattern recognition, pp. 22576–22585, 2023.

### Questions
1.	Blackout Diffusion (Santos et al., 2023) achieves an extremely low FID (0.02) on binarized MNIST. Could the authors clarify what might account for this near-perfect score?

2.	In the high-resolution image synthesis experiments, the Precision/Recall gap between BFM and BLD (Wang et al., 2023) appears small and dataset-dependent.What factors do the authors think drive these differences? Are there inherent limitations of the linear probability path that affect sample fidelity? Would increasing the step count reduce the FID gap?

Javier E Santos, Zachary R Fox, Nicholas Lubbers, and Yen Ting Lin. Blackout diffusion: generative 
diffusion models in discrete-state spaces. In International Conference on Machine Learning, pp. 
9034–9059. PMLR, 2023.

Ze Wang, Jiang Wang, Zicheng Liu, and Qiang Qiu. Binary latent diffusion. In Proceedings of the 
IEEE/CVF conference on computer vision and pattern recognition, pp. 22576–22585, 2023.

### Soundness
2

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
4

### Summary
The authors propose a method to learn discrete flows on binary data. My understanding is the training procedure is roughly the following: given a batch of binary data, flip some of the bits according to a scheduler, and use a neural net to predict the data given the noise. This is similar to BERT style training (as most things are), but instead of masking random elements you flip them.

Authors experiment on image generation by directly modelling the pixel bits, which is an interesting application.

I did not check the math.

### Strengths
- The method is simple to understand and has visually pleasing images.
- Given how image generation metrics are largely overfitted by the community, and don't serve much as a signal when fighting for decimal points, I believe the authors have a nice method regardless of FID and other results.

### Weaknesses
- The method is akin to discrete flow matching and "discrete diffusion", but the authors don't review discrete flow matching: instead they review regular continuous FM. Am I missing something?

- Some discussion around inference time speed would be nice. For example, image quality as a function of inference steps, throughput on whatever GPU they have at their disposal, etc.

- Suggestion for future work: multimodal data.

### Questions
1. I couldn't tell if this "Heuristic Bernoulli Diffusion Models" is prior work or if the authors are proposing it in this paper.

2. Can the authors please give an example and clarify what this means: "However, when these methods are naively applied to the Bernoulli distribution, their dependence on predicting the instantaneous velocity field during sampling can introduce invalid Bernoulli parameters, leading to model collapse."

3. I believe the probability path is similar to the uniform one present in https://arxiv.org/abs/2412.03487, but on binary data (I'm not advocating you cite this paper, just calling your attention to it). Can the authors comment on this?

4. My understanding is the training procedure is roughly the following: given a batch of binary data, flip some of the bits according to a scheduler, and use a neural net to predict the data given the noise. This is similar to BERT style training (as most things are), but instead of masking random elements you flip them. Is this correct?

5. Could the authors add some discussion around speed, as mentioned in the box above? I'm willing to raise my score if this is properly addressed (the rest is less important imo). Thanks.

### Soundness
4

### Presentation
3

### Contribution
4
