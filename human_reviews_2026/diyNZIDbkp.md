# Hierarchical Parametrization with Gaussian Process for Bayesian Meta-Learning

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Meta-learning has emerged as a key approach to preparing neural networks for deployment with limited training data. Mainstream solutions focus on parameter initialization across training episodes to enable rapid generalization, which can be interpreted as adjusting the episode-specific posterior based on a cross-episode prior distribution in the context of Bayesian meta-learning. Despite the demonstrated efficacy of this probabilistic meta-learning paradigm, existing methodologies encounter performance bottlenecks, particularly as the scale and number of episodes increase. A promising strategy involves the integration of a hyper-network, which establishes a parameter memorization space across diverse episodes. In this paper, we propose Hierarchical Parametrization with Gaussian Process (HP-GP), a novel probabilistic meta-learning method that leverages the power of Gaussian Process. By implementing the amortization network layer-wise with decoupling variational Gaussian Process and normalizing flow, HP-GP offers probabilistic parametrization for meta-learning while requiring minimal modifications to the network architecture. This enables flexible and scalable integration of meta-learning into existing neural networks. Our experiments demonstrate the flexibility and robust generalization of HP-GP, outperforming other popular meta-learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a Bayesian meta-learning method that utilizes Stochastic Variational Gaussian Processes (SVGP) and Normalizing Flows, implementing the amortization network in a layer-wise manner. It shows that SVGP enables input–output tensor decoupling, which reduces computational cost, while Normalizing Flows improve performance by mapping the Gaussian posterior from SVGP to a more expressive distribution.

### Strengths
- The idea of using inducing points in SVGP and mapping the posterior to a more expressive, non-Gaussian distribution using Normalizing Flows is innovative. The decoupled calculation derived from the product of two Gaussians also significantly reduces computational complexity.
- It also proposes an additional optimization by leveraging the recently introduced KAN as a substitute for MLPs, which shows a performance improvement in the ablation. An addition that I found adds meaningful value to the paper.
- The paper’s connection to related work is well-researched and presented in a timely and relevant manner.

### Weaknesses
- The experimental section could be expanded further. For example, by including the final experiment from [1].
- The limitations of the proposed method should be articulated more clearly in the conclusion section.

[1] Meta-learning without memorization (Yin, et al. 2020)

### Questions
__Q1.__ Could you comment on why the performance of HP-GP appears to be relatively unaffected to the choice of kernel function? Additionally, have you considered evaluating a non-smooth kernel, such as the Matérn $\frac{1}{2}$ kernel?

__Q2.__ What is the time comparison between using KAN and other architectures in the Normalizing Flow ablation experiment presented in Section A.4.3?

__Q3.__ Regarding the inducing points $\mathbf{u}$, how is their number determined? Is the number of inducing points learned during training, or must it be specified manually by the user? In any case, it would be helpful if you could provide more details on the choice of number of $\mathbf{u}$ for each experiment. If this is specified manually, an ablation study would add further value to the paper.

__Q4.__ In Table 2, your proposed method uses two different base networks (ResNet-18 and Conv-4). Could you comment on how fair or comparable this setup is relative to the baselines? For example, do the architecture-agnostic baselines also use ResNet-18 or Conv-4? And for the non-agnostic baselines, are they using a comparable number of parameters? As at the moment, the results for the baselines from Yin et al. (2020) appear to be taken directly from their paper.

__Q5.__ I wonder if replacing the Normalizing Flows with Diffusion or Flow Matching models would work in your setting, especially given their strong performance in high-dimensional setups such as neural network weights. Could you comment on this?

__Q6.__ Could you also disclose the complete setup used to produce the t-SNE experiment in Figure 4? It might be helpful to include this information in the Appendix for completeness.

__Minor__

- In line 149, the abbreviation ABML should be expanded. For instance, you could provide the full term Amortized Bayesian Meta Learning (ABML) (Ravi & Beatson, 2019) in line 147.

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
4

### Summary
The paper proposes HP-GP (Hierarchical Parametrization with Gaussian Process), a Bayesian meta-learning framework that combines Sparse Variational Gaussian Processes (SVGP) and Inverse Autoregressive Flow (IAF) with Kolmogorov–Arnold Networks (KAN) to generate episode-specific neural network weights in a probabilistic, model-agnostic, and scalable manner. HP-GP uses a layer-wise decoupled GP to reduce computational cost and leverages normalizing flows to transform Gaussian posteriors into richer weight distributions. It is evaluated on few-shot regression tasks (time series prediction and object pose estimation), where it consistently outperforms state-of-the-art methods, especially in challenging non-mutually exclusive settings that induce memorization.

### Strengths
The main strength of the proposed approach are:

* Model-agnostic and lightweight. The method requires minimal architectural changes required, and can be integrated into existing networks.

*Strong empirical performance. The proposed approach achieves SOTA results on both in-distribution and out-of-distribution few-shot regression benchmarks.

*Robust to memorization. The approach excels in non-mutually exclusive tasks by leveraging cross-class information via inducing variables.

*Theoretically grounded. The method is theoretically grounded and provides bounds on norm, Rademacher complexity, and approximation error using RKHS and Barron space frameworks.

* Uncertainty-aware. Finally, being rooted on Gaussian Processes, the method naturally supports uncertainty quantification through Bayesian inference.

### Weaknesses
The main limitations of the paper are:

* Computational overhead: Despite decoupling, SVGP + IAF introduces nontrivial compute/memory costs vs. simpler meta-learners (e.g., MAML). It would also be useful to compare the method against existing probabilistic methods such as BMAML (Yoon et al.) to further elucidate its strengths/weaknesses when it comes to computational usage.

* Limited task scope. The proposed method is evaluated only on regression (not classification) tasks, whereas existing methods such as BMAML, which is also probabilistic, is evaluated on classification, active learning, and reinforcement learning tasks. I believe that this reduces the generalizability claims of the paper.

* Hyper-parameter sensitivity. The performance of the proposed approach hinges heavily on kernel choice, number of inducing points, and IAF depth (risk of over-parameterization).

* Scalability concerns. While layer-wise, the extension of the paper to very deep architectures (e.g., transformers) is not demonstrated.

* Ablation gaps. Because of the limited analysis of the relative contribution of KAN vs. standard MLPs beyond one table, it is unclear to me if KAN is essential.

### Questions
Please refer to the section above.

### Soundness
2

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
The paper proposes a new meta-learning method that integrates Gaussian Process into the modeling. Conventionally, probabilistic meta-learning often relies on a hyper-model (also known as hyper-net) to generate task-specific parameters to adapt to the task of interest. Instead of following a parametric approach, the paper proposes to use Gaussian Process coupled with normalizing flow. Such a modeling is claimed to be flexible and scalable with minimal change to the model used. Empirical evaluation shows that the newly-proposed method outperforms existing meta-learning methods on some benchmarks.

### Strengths
The paper provides an extensive study about related studies and presents detailed background knowledge (especially section 3). These make the paper self-contained and easier to follow.

The paper also includes theoretical analysis on the generalization of the proposed methods, and in particular upper bounding the error in the form of PAC learning. This provides certain level of generalization guarantee.

The proposed method is evaluated on two families of tasks: one is about time series prediction and the other about object pose prediction. The proposed method has been demonstrated to out-perform existing meta-learning methods in both evaluation benchmarks.

### Weaknesses
In the current form, the paper is too dense. The notations used are not well explained. For example: at line 203, "u and $\omega$ are inducing latent functions and design functions". It is unclear what these functions are. In addition, the generative process in the modeling shown in Fig. 1 is not explained at all. Could the authors provide the data generation process like other graphical model papers do? For example, the authors could refer to the template in section 3 of the paper: "Latent Dirichlet Allocation". That would help to understand how data is generated, what the role of each parameter and which kind of parameter inference approach the paper is proposing. In addition, introducing some variables like the description right after Eq. (3) is not recommendable because it suddenly jumps to a much later equation. This causes surprise and hard to understand the paper.

The usage of normalizing flow introduced in the paper is not clear. Could the authors elaborate why normalizing flow is needed, while other hyper-net based meta-learning methods do not?

The empirical evaluation is not too standard to the meta-learning literature. I am aware of the standard evaluation may be old. However, it would be fairer to evaluate on the standard few-shot learning on mini-ImageNet, then move to the new datasets like the one considered in the paper. In addition, the current baselines are outdated (most baselines were published before Covid-19). Should newer baselines be included to reflect the high-performance of the proposed method.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
