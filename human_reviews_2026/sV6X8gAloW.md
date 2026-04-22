# SVRG and Beyond via Posterior Correction

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Stochastic Variance Reduced Gradient (SVRG) and its variants aim to speed-up training by using gradient corrections, but have seen limited success in deep learning.
   Here, we show surprising new foundational connections of SVRG to a recently proposed Bayesian method called posterior correction. Specifically, we show that SVRG is recovered as a special case of posterior correction over the isotropic-Gaussian family, while novel extensions are automatically obtained by using more flexible exponential families.
   We derive two new SVRG variants by using Gaussian families: First, a Newton-like variant that employs novel Hessian corrections, and second, an Adam-like extension that improves (continual) pretraining and finetuning of Transformer language models. This is the first work to connect SVRG to Bayes and use it to boost variational training for deep networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper establishes a connection between SVRG (Johnson & Zhang, 2013) and posterior correction (PC)
(Khan, 2025), showing that SVRG can be understood as a special case of PC assuming isotropic Gaussian distributions.
PC is subsequently used to improve the IVON (Shen et al., 2024) optimizer across a wide range of experiments.

### Strengths
- The mathematical connection between SVRG and PC is novel.
- A strong and wide-ranging set of experiments together with a comprehensive set of ablations demonstrating the power of combining IVON with PC.

### Weaknesses
- While the equivalence of SVRG as a special case of PC is interesting, the paper reads like a
follow-up to Khan (2025) that demonstrates how PC and IVON can be combined to markedly improve
the performance of the latter, a comparison that was missing in Khan (2025).
The SVRG connection feels like an afterthought that does not provide deeper insights,
e.g., the extent to which convergence properties from SVRG still hold or can be extended to non-isotropic Gaussians, or the extent to which smoothness assumptions and variance reduction empirically hold.
- The paper lacks an analysis of the added computational cost, both from SVRG and from the Hessian approximations.
- All runs appear to be single runs, without any results on variance across training runs.
- No code is provided.

### Questions
- Q1: Does PC gain any theoretical properties from the derived SVRG connection?




_____
*Note: The low score is primarily due to the strong focus on SVRG without the paper gaining anything from this focus. If one were to ignore that and simply read it as an application paper combining IVON with PC and evaluating it extensively, the score would be higher.*

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a connection between Stochastic Variance Reduced Gradient (SVRG) with Posterior Correction (PC). The authors show that SVRG emerges as a special case of posterior correction when using isotropic Gaussian distributions. This insight allows them to derive new SVRG variants by applying posterior correction to broader exponential-family distributions. Two extensions are proposed: 1. Newton-like variant (VON-PC) — incorporates stochastic variance reduction for both gradients and Hessians, improving stability and enabling second-order corrections. 2. Adam-like variant (IVON-PC/IVON-PCM) — adapts posterior correction over the IVON optimizer, showing strong performance in continual pre-training and fine-tuning of large models like GPT-2 and ViT.

### Strengths
1. The original contribution by establishing a connection between SVRG and posterior correction. The interpretation reframes variance reduction as a form of knowledge transfer, a new perspective that unifies two previously separate research threads.
2. The authors successfully extend this connection to derive two new SVRG variants: One is a Newton-like method incorporating stochastic variance reduction for both gradients and Hessians, introducing Hessian corrections rarely explored in prior work; another is an Adam-like method (IVON-PC/IVON-PCM) with diagonal covariance approximations.

### Weaknesses
1. Compared with the well established optimization method e.g., AdamW, posterior correction does not yield clear improvements in deep learning tasks. 
2. Can the variance reduction method be applied to reinforcement learning? The authors can explore the experiments on either RLVR of LLMs post-training or some other traditional tasks in RL area. It can also compared with TRPO (Trust Region Policy Optimization).

### Questions
How about its computational overhead (e.g., Hessian estimation, mega-batch refresh) ?

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
The paper establishes a connection between SVRG and posterior correction, enabling the generalization of variance reduction method to the IVON optimizer. At a high level, it aims to establish a relationship between non-Bayesian optimization methods and variational inference within a common theoretical framework.

### Strengths
- The connection between SVRG and posterior correction is sound.
- An advantage of the connection is that it generalizes the variance reduction method to higher-order optimizers, leading to a novel IVON-PC.

### Weaknesses
- While the theoretical analysis provides a fresh perspective on SVRG, the resulting IVON-PC method does not appear to offer practical benefits.

  - Figure 3 shows an initial improvement, but IVON-PC ultimately requires a similar number of gradient computations as SVRG to reach comparable final performance. Likewise, Figure 5 demonstrates that their final performance remains nearly identical.

  - The gains reported in Table 5 are also insignificant, especially when considering the error bars.

- Another limitation is that the established connection does not clarify why SVRG or the proposed IVON-PC fail to deliver stronger empirical results.

The above two points raise questions about whether we obtained any value through the established connection. Nonetheless, I lean toward acceptance, as the work introduces a novel viewpoint and may inspire deeper future investigations on SVRG.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
