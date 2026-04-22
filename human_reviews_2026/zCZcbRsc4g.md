# Avoid Catastrophic Forgetting with Rank-1 Fisher from Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Catastrophic forgetting remains a central obstacle for continual learning in neural models.
Popular approaches---replay and elastic weight consolidation (EWC)---have limitations: replay requires a strong generator and is prone to distributional drift, while EWC implicitly assumes a shared optimum across tasks and typically uses a diagonal Fisher approximation.
In this work, we study the gradient geometry of diffusion models, which can already produce high-quality replay data.
We provide theoretical and empirical evidence that, in the low signal-to-noise ratio (SNR) regime, per-sample gradients become strongly collinear, yielding an empirical Fisher that is effectively rank-1 and aligned with the mean gradient.
Leveraging this structure, we propose a rank-1 variant of EWC that is as cheap as the diagonal approximation yet captures the dominant curvature direction.
We pair this penalty with a replay-based approach to encourage parameter sharing across tasks while mitigating drift.
On class-incremental image generation datasets (MNIST, FashionMNIST, CIFAR-10, ImageNet-1k), our method consistently improves average FID and reduces forgetting relative to replay-only and diagonal-EWC baselines. In particular, forgetting is nearly eliminated on MNIST and FashionMNIST and is roughly halved on ImageNet-1k.
These results suggest that diffusion models admit an approximately rank-1 Fisher.
With a better Fisher estimate, EWC becomes a strong complement to replay: replay encourages parameter sharing across tasks, while EWC effectively constrains replay-induced drift.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the gradient geometry of diffusion models in low-SNR timesteps and argues that per-sample gradients become nearly collinear, making the empirical Fisher effectively rank-1. Building on this, the authors propose a rank-1 EWC penalty (cheap to compute) and pair it with generative distillation for continual image generation. Across MNIST, FMNIST, CIFAR-10, and downsampled ImageNet-1k (32×32), the method improves average FID and notably reduces forgetting versus diagonal-Fisher EWC and distillation alone (e.g., near-zero forgetting on MNIST/FMNIst and roughly halved on ImageNet-1k). Theory (Propositions/Theorem) and empirical analyses (eigenspectra, cosine similarities) support the rank-1 claim.

### Strengths
Clear theory → practice bridge: formal propositions/theorem (low-SNR → gradient collinearity → rank-1 Fisher) with convincing empirical validation (eigenvalue dominance, Frobenius error vs. diagonal). 

Simple & efficient: rank-1 EWC captures dominant curvature at cost comparable to diagonal EWC; easy to add to standard UNet diffusion setups. 

Consistent gains: with generative distillation, improves AFID and reduces forgetting on four datasets; forgetting ≈ 0 on MNIST/FMNIst and ~halved on ImageNet-1k. 

Robust learning dynamics: smoother curves over long horizons; qualitative samples maintain object sharpness compared to baselines that drift.

### Weaknesses
Scope/scale limits: experiments are at 32×32 resolution (even for ImageNet-1k) and on UNet backbones; no results on larger resolutions or Transformer/DiT diffusion models, which the paper itself flags as future work. 

Heavy reliance on distillation: EWC alone underperforms; the strongest results require rank-1 EWC + generative distillation, so the standalone benefit of the rank-1 penalty is limited.

### Questions
see weakness

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
This paper investigates the problem of catastrophic forgetting in the continual learning of generative diffusion models. The authors identify a key limitation of Elastic Weight Consolidation (EWC), a popular regularization method, which typically relies on a diagonal approximation of the Fisher Information Matrix (FIM) that fails to capture important parameter correlations. The paper's main contribution is a theoretical and empirical analysis showing that for diffusion models in the low signal-to-noise ratio (SNR) regime, the empirical FIM is effectively rank-1 and aligned with the mean gradient. Leveraging this insight, the authors propose a "Rank-1 EWC" penalty that is computationally efficient yet captures this dominant curvature direction. This regularizer is combined with generative distillation (a form of replay) to create a synergistic approach that encourages a shared parameter space while mitigating drift. Experiments on class-incremental image generation tasks (MNIST, FashionMNIST, CIFAR-10, and ImageNet-1k) show that the proposed method significantly reduces forgetting and improves sample quality (FID) compared to replay-only and diagonal-EWC baselines.

### Strengths
1. Novel and Insightful Analysis of Fisher Geometry: The core of the paper is a novel and non-obvious finding about the gradient structure of diffusion models.
2. Practical and Computationally Efficient Algorithm: The proposed Rank-1 EWC penalty, as formulated in Equation 6, is highly practical.
3. Strong Empirical Performance and Validation: The method demonstrates impressive empirical results, substantially reducing catastrophic forgetting and improving generation quality (FID) across four different benchmarks.

### Weaknesses
1. Heavy Reliance on Generative Distillation: A significant weakness is the method's apparent dependence on the replay component.
2. Strength of Theoretical Assumptions: The theoretical argument hinges on Assumption 1, which posits that the score network $s_{\theta}(x_t, t)$ can be approximated as a linear function of its parameters, $s_{\theta}(x_t, t) \approx x_t\theta$, in the low-SNR regime.
3. Limited Scope of Fisher Matrix Analysis: The detailed empirical analysis of the FIM in Section 3.2 is conducted on a small-scale diffusion model trained on MNIST

### Questions
1. The effectiveness of your Rank-1 EWC is clearly dependent on the generative distillation component. Could you elaborate on the precise role of the penalty?
2. Your theoretical analysis highlights the emergence of the rank-1 structure in the low-SNR (late timestep) regime. However, the EWC penalty in Equation 6 appears to be based on a mean gradient $\mu$ that is averaged over all timesteps. How does the inclusion of gradients from high-SNR timesteps, where the Fisher may not be rank-1, affect the validity and performance of your proposed penalty?
3. The analysis in Figure 3a shows that the mean gradients $\mu_t(\theta)$ are highly aligned across timesteps but are not perfectly collinear. This suggests that a single rank-1 approximation for the entire FIM might be discarding some useful information.

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
4

### Summary
This paper concentrates on the continual learning of diffusion models with an improved EWC method (i.e., rank-1 EWC). The main contributions are summarized as:

a. This paper provides both theoretical and empirical characterizations of Fisher information geometry in diffusion models, showing that low SNR induces a near rank-1 Fisher aligned with the mean gradient.

b. This paper proposes a practical rank-1 EWC penalty that is as cheap as a diagonal penalty but captures more curvature information for diffusion models.

c. This paper provides extensive experiments to prove the effectiveness of the proposed method.

### Strengths
a. This paper is well written and easy to follow.

b. This paper is technically solid.

c. It is interesting and significant to prove that the empirical Fisher matrix of diffusion models is effectively rank-1.

### Weaknesses
I am generally satisfied with the content presented in the paper; however, I still have the following concerns:

a. I believe that the most critical premise supporting this paper is Assumption 1. However, its explanation is overly intuitive and lacks rigor. Please use a concrete example and demonstrate, from a mathematical perspective, how this example is connected to Assumption 1. This is the major concern.

b. The Theorem considers the low SNR region (i.e, at later diffusion timesteps). However, SNR will be high at most diffusion timesteps. How does the paper account for this situation?

c. How to determine that a SNR is low?

### Questions
a. In Section 3, it is mentioned that "This is plausible in practice because a trivial solution for UNet is to directly route the inputs to the output due to the skip connections". Please theoretically discuss Assumption 1 according to this example in details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the gradient geometry of diffusion models and argues that, at low SNR timesteps, per-sample gradients align strongly with their mean, making the empirical Fisher approximately rank-1. Building on this, the authors propose a rank-1 EWC penalty that is as cheap as the diagonal approximation but better aligned with the dominant curvature direction. They pair it with diffusion-based generative distillation/replay. On class-incremental image generation (MNIST, FMNIST, CIFAR-10, downsampled ImageNet-1k), the method improves AFID and reduces forgetting versus diagonal-EWC and replay alone; forgetting is nearly eliminated on MNIST/FM and roughly halved on ImageNet-1k.

### Strengths
- The motivation example is well positioned, the story of the paper is clear and easy to follow.
- Clean connection from low SNR → gradient collinearity → rank-1 Fisher, with both theory and measurement
- Rank-1 EWC that is drop-in and cost-comparable to diagonal while capturing dominant curvature. 
- Consistent AFID/forgetting improvements; notably near-zero forgetting and significant improvement on different benchmarks.

### Weaknesses
- Ablations missing/limited: (i) k-rank (>1) Fisher vs rank-1 vs diagonal; (ii) Sensitivity to λ, μ-estimation schedule, and the SNR/timestep sampling strategy; (iii) replay buffer size; teacher choice (EMA vs last checkpoint) in distillation.

- No wall-clock/memory comparisons vs Diag-EWC and low-rank baselines (e.g., diagonal + small K-FAC block, diagonal + momentum/EMA Fisher smoothing).

- Main long-horizon test uses 32×32 ImageNet-1k; higher-res or domain/task-incremental setups would strengthen claims.

- I suspect that other baselines are not well tuned. Figure 4 shows that the proposed method is consistently better than others in adapting to new task, which means the method can obtain both stability and plasticity overall. I am wondering how the result would look like if λ=0 (no regularization).

### Questions
- Is this the first work about continual learning for diffusion model? If it is not, I expect the author to compare against other related work. The main table result looks like an ablation studies for Fisher approximation approaches.

- Can you conduct ablation studies on rank-k (k=2–8) variants to test diminishing returns beyond rank-1?

### Soundness
3

### Presentation
3

### Contribution
3
