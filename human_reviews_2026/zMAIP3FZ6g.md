# Sculpting Latent Spaces With MMD: Disentanglement With Programmable Priors

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Learning disentangled representations, where semantic features are captured by independent variables, is dominated by the Variational Autoencoder (VAE) which uses the Kullback-Leibler (KL) penalty to learn a factorized representation in the latent space. In this paper, we provide direct visual and quantitative evidence that the VAE-based methods consistently fail to enforce this target distribution on the aggregate posterior, subsequently falling short of a mutually independent representation -- the training objective of unsupervised disentanglement. We quantify this failure and resulting entanglement using a stable, unsupervised Latent Predictability Score (LPS). To address this, we propose the Programmable Prior Framework: a non-parametric method built on the Maximum Mean Discrepancy (MMD). We verify our framework allows practitioners to explicitly sculpt the latent space, achieving (1) state-of-the-art unsupervised statistical independence (measured by LPS), (2) alignment to semantic features using an internal semi-supervised mechanism, and (3) aggregate posterior distribution shaping (validated through quantization-aware training), all without reconstruction trade-offs. Ultimately, the framework is one of a kind in that it provides a reliable foundational tool for balancing these three key training objectives, opening new avenues for model identifiability, interpretability, causal reasoning, and efficient compression.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript presents a method to improve learning of disentangled latent space. The key component is using MMD instead of KL for regularization. The paper also proposes a metric LPS, latent predictability score.

### Strengths
1. The paper tries to address an important and open challenge. 
2. The paper is written reasonably well, easy to follow.

### Weaknesses
The manuscript has the following issues:

1. It is well understood that MMD is better than KL in many applications. However the motivation in Fig 1, which empirically shows the proposed method learns better Gaussian latent space, is insufficient. Learning isotropic Gaussian latent space is closely related to learning disentangled latent space, but not the same. Especially for VAE and $\beta-$VAE, since latent space is subject to rotation, a more rigorous study is needed.
2. The paper is highly questionable in terms of execution. It's acceptable to propose a new metric, while propose a new method. But the new metric should be thoroughly examined before using it to champaign the proposed method. This manuscript failed to achieve this.
3. Missing an important reference. The proposed LPS is very similar to a recent paper Yeats et al, especially the concept of using reconstruction loss $d-1$ dimensions. 
4. Lack of commonly used experimental benchmarks.

### Questions
1. Using a newly defined metric, LPS, to demonstrated the superiority of a proposed method needs to be executed thoroughly. The authors need to first establish the fact that the new metric is better than the existing metrics (e.g., MIG, DCI), in terms of consistency and robustness, across major existing disentanglement methods and datasets. I would encourage the authors to conduct such a comprehensive study before using the same metric to champaign the new method proposed by the same set of authors.

2. The details of LPS bear a close resemblance to a recent paper on disentanglement:

Yeats, E., Liu, F., Womble, D. and Li, H., 2022, October. Nashae: Disentangling representations through adversarial covariance minimization. In European Conference on Computer Vision (pp. 36-51). Cham: Springer Nature Switzerland.

which is not included in the reference section. 

3. In LPS, "a regression model is trained to predict $z_i$", however there is no mentioning on how this model is trained, and more importantly, whether the quality of this model can be trusted to calculate LPS.  This is a much more nuanced usage of the regression model than in Yeats et al, where the purpose of the regression model is to **encourage** the disentanglement, not as a final yardstick to **evaluate the quality of disentanglement**. Again this brings up the question whether the LPS should be used to champaign the proposed method.

4. The experimental results are weak. As in Higgins et al, CelebA is commonly accepted as one of the baseline benchmarks. There is also a small datasets with known generating factors in Yeats et al. These benchmarks should be included in the manuscript.

### Soundness
1

### Presentation
2

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
This paper presents a critique of the conventional Variational Autoencoder (VAE) framework for learning disentangled representations, arguing that its Kullback-Leibler (KL) divergence penalty is an unreliable mechanism for enforcing a factorized prior on the latent space. To address this, the authors introduce the Programmable Prior Framework, which replaces the KL divergence with a non-parametric Maximum Mean Discrepancy (MMD) regularizer. This MMD-based approach allows VAEs to sculpt the distribution of posterior for latent space which samples can be drawn from any target distribution, such as Gaussian, Uniform, or even Gaussian Mixture Models.
The authors further propose a novel unsupervised metric, the Latent Predictability Score (LPS), to quantify the mutual independence of latent features by measuring how well one latent dimension can be predicted from the others using a regression model. A lower score indicates greater independence.

### Strengths
1. The paper clearly identifies and provides compelling visual evidence (Figure 1) for a critical weakness in VAE-based disentanglement methods—the failure of the KL term to shape the aggregate posterior.
2. The proposal of the Latent Predictability Score (LPS) is a significant contribution. Its unsupervised nature makes it applicable to real-world datasets where ground-truth factors are unavailable. Furthermore, the authors convincingly demonstrate its superior stability compared to the high variance of alignment-based metrics like DCI, which is a crucial point for reliable model evaluation.

### Weaknesses
1. The paper's primary strength—the programmability of the prior—is also its main practical limitation. How to determine a proper target distribution for posterior remains unsolved. We can see the significant performance drop for choosing a wrong distribution from Table 12. The gaussian prior may not be the best one, but is robust for the most cases.
2. Due to the non-linearity of deep  neural networks, a gaussian distribution can be mapped to arbitrary distributions. Therefore, the posterior distribution is not a emergent problem in disentanglement learning.
3. From table 5, the proposed methods (AE-MMD) have lower MIG scores, low SAP, and so on. The methods did not exhibit strong disentangled representations on dSprites.

### Questions
Why AE-MMD has a high Covariance Ratio but low number of MIG or SAP?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, the authors tackle the limitations of KL-divergence-based VAE methods for learning disentangled representations. It introduces a flexible, architecture-agnostic framework using Maximum Mean Discrepancy to explicitly sculpt the latent space to match arbitrary priors, enabling what the authors term a programmable prior” The framework achieves state-of-the-art statistical independence of latent features without sacrificing reconstruction quality and provides a novel unsupervised metric, the Latent Predictability Score, to quantify disentanglement. Experiments across synthetic and real-world datasets demonstrate that MMD regularization can enforce both marginal and joint properties of latent distributions, allow alignment with interpretable features, and scale across varying latent dimensions.

### Strengths
This paper makes a significant contribution to the field of disentangled representation learning by directly addressing the limitations of KL-divergence-based VAE regularization. The proposed MMD-based framework is architecture-agnostic and provides a flexible, sample-based mechanism for sculpting the latent space to match arbitrary priors. This programmability is a clear strength, allowing practitioners to inject task-specific inductive biases into the representation, which is demonstrated through strong empirical results on complex datasets like CIFAR-10 and TinyImageNet. The framework also achieves state-of-the-art statistical independence of latent features, as measured by the novel Latent Predictability Score, without sacrificing reconstruction quality—a common trade-off in prior work.

Additionally, the introduction of the unsupervised LPS metric is an important methodological advance. By evaluating mutual independence without relying on ground-truth factors, LPS provides a robust and widely applicable tool for quantifying disentanglement. The experiments convincingly demonstrate that MMD regularization consistently enforces true statistical independence across diverse datasets and latent dimensions, highlighting the scalability and generality of the approach. The visualization and latent space copying experiments further illustrate the precision and flexibility of the programmable prior framework.

### Weaknesses
Despite its strengths, the framework has limitations in real-world applicability due to the challenge of selecting an optimal prior. While simple priors like factorized Gaussians are effective for achieving statistical independence, engineering priors that align with semantically meaningful features often requires domain knowledge that may not be available. This limits the ease of deploying the method in fully unsupervised scenarios where the underlying generative factors are unknown.
Another potential weakness lies in the computational complexity of MMD regularization in high-dimensional latent spaces. Although the paper demonstrates strong empirical performance, fitting complex priors may require careful kernel selection and tuning, which could hinder scalability or reproducibility for very large datasets. Furthermore, while the framework excels at matching marginal and aggregate distributions, it does not guarantee learning identical latent representations across models, which may limit its effectiveness in applications like knowledge distillation or causal representation learning

### Questions
How sensitive is the framework’s performance to the choice of kernel in the MMD regularizer, especially for high-dimensional latent spaces?

### Soundness
3

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
3

### Summary
This paper proposes a novel framework for learning disentangled representations by replacing the traditional KL divergence penalty in VAEs with the Maximum Mean Discrepancy (MMD). The authors argue that the per-sample KL-divergence is an unreliable mechanism for enforcing the desired factorized structure on the aggregate posterior distribution. By leveraging MMD, the proposed Programmable Prior Framework is claimed to be architecture-agnostic and non-parametric, allowing practitioners to "sculpt" the latent space to match any target prior distribution. The method demonstrates disentanglement performance on datasets like CIFAR-10 and Tiny ImageNet without the reconstruction quality trade-off common in $\beta$-VAE. Additionally, the authors introduce Latent Predictability Score (LPS), a unsupervised metric for quantifying mutual independence.

### Strengths
Clarity: Overall the paper is clearly written.

Novel Unsupervised Metric: The Latent Predictability Score (LPS) offers a new tool for quantifying mutual independence without relying on ground-truth factor labels.

### Weaknesses
Limited Theoretical Grounding & Novelty: The core technical idea is closely related to a lot of VAE variants e.g. WAE framework. The paper also need to formally justify the claim that $L_{ours}$ (Eq 5) is a lower bound on the log-likelihood.

Unsupported Claim Regarding Prior Flexibility: The paper claims that the proposed method supports "any prior", but the standard VAE objective is also theoretically valid for any analytic prior. The paper's strength is that MMD is empirically and practically more effective at sculpting the aggregate posterior to a non-Gaussian geometry due to its non-parametric nature. This distinction must be made clearer.

MMD Implementation Challenges: MMD is sensitive to the choice and tuning of the kernel function. A discussion or experiment on the robustness of the results to kernel choice would strengthen the practical utility.

Reproducibility: The provided code link is empty.

### Questions
Can the authors provide a rigorous theoretical derivation or reference for the claim that the objective ${L}_{ours}$ (Eq 5) is a lower bound on the log-likelihood?

MMD performance can be sensitive to the kernel function (e.g., Gaussian RBF) and its parameters. Could the authors include a robustness study?

### Soundness
2

### Presentation
2

### Contribution
1
