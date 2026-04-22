# Beyond Loss Guidance: Using PDE Residuals as Spectral Attention in Diffusion Neural Operators

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Diffusion-based solvers for partial differential equations (PDEs) are often bottle-necked by slow gradient-based test-time optimization routines that use PDE residuals for loss guidance. They additionally suffer from optimization instabilities and are unable to dynamically adapt their inference scheme in the presence of noisy PDE residuals. 
To address these limitations, we introduce PRISMA (PDE Residual Informed Spectral Modulation with Attention), a conditional diffusion neural operator that embeds PDE residuals directly into the model's architecture via attention mechanisms in the spectral domain, enabling gradient-descent free inference. 
In contrast to previous methods that use PDE loss solely as external optimization targets, PRISMA integrates PDE residuals as integral architectural features, making it inherently fast, robust, accurate, and free from sensitive hyperparameter tuning. We show that PRISMA is at-par or better in accuracy compared to previous methods across five benchmark PDEs especially with noisy observations, while using 10x to 100x fewer denoising steps, leading to 15x to 250x faster inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a diffusion-based PDE solver that can significantly accelerate the problem solving process. The new design is that the author introduces a conditional diffusion neural operator that embeds PDE residuals directly into the model’s architecture via attention mechanisms in the spectral domain, enabling gradient-free inference. The experimental results show PRISMA is at-par or better in accuracy compared to previous methods across five benchmark PDEs especially with noisy observations, while using 10x to 100x fewer denoising steps, leading to 15x to 250x faster inference.

### Strengths
(1) The novelty is somewhat good. Unified Framework for Diverse PDE Tasks: PRISMA’s conditional design, enabled by input masks unifies forward and inverse PDE solving under a single model, supporting full, sparse, and noisy observation regimes. Unlike baselines (e.g., FNO, PINO) that require task-specific models or inference pipelines, PRISMA seamlessly adapts to different problem settings (e.g., sparse Darcy flow inverse problems, noisy Navier-Stokes forward problems) without reconfiguration. This versatility is a significant advance for real-world applications where observation quality varies.

(2) The experimental results. The paper demonstrates compelling empirical performance across five benchmark PDEs (Darcy Flow, Poisson, Helmholtz, Navier-Stokes with/without BCs). PRISMA achieves 15x–250x faster inference (0.18–0.8 seconds per sample) compared to diffusion-based baselines (e.g., DiffusionPDE: 213s, FunDPS: 11.8s) by using only 20–50 denoising steps (vs. 200–2000 steps for baselines). Crucially, this speedup does not compromise accuracy: PRISMA outperforms baselines in noisy settings (e.g., Darcy Flow forward error: 12.28% vs. FunDPS’s 55.09% and DiffusionPDE’s 49.18%) and matches top performers (e.g., PINO, FunDPS) in full/sparse observations.

### Weaknesses
(1) Limited Discussion of SRA Block Mechanics: While the SRA block is core to PRISMA’s success, the paper provides insufficient detail on its internal workings. For example:
The calculation of the compatibility score 
S^l(k) (complex inner product of Fourier-transformed features and residuals) is described, but the intuition for why spectral-domain attention outperforms spatial-domain methods (e.g., residual concatenation) is underdeveloped.
The MLP that learns g_res (guidance strength) is not specified (e.g., architecture, input features beyond r_avg and c_σ), making it hard to replicate or extend.
The paper does not explain how SRA handles frequency modes with conflicting residual signals (e.g., high-frequency noise vs. low-frequency physical signals).


(2) Spatio-Temporal and Irregular Mesh Limitations: PRISMA is evaluated exclusively on static (time-independent) PDEs with regular grid inputs. The authors acknowledge future work will extend to spatio-temporal problems and irregular meshes, but the current limitation narrows PRISMA’s applicability. Many real-world PDEs (e.g., time-dependent Navier-Stokes, heat equation on complex geometries) require these capabilities, and the paper provides no insight into how PRISMA’s architecture might adapt (e.g., integrating temporal attention, handling unstructured grids). Are there any insights for this problem.

(3) Scalability to High Resolution: The paper evaluates PRISMA on 64×64 and 128×128 grids, but does not address scalability to larger resolutions (e.g., 256×256, 512×512). Diffusion models and neural operators often face computational bottlenecks at high resolution (e.g., Fourier transform costs, memory usage). The authors should clarify whether PRISMA’s 64M parameter count (similar to baselines) remains feasible at higher resolutions, or if modifications (e.g., sparse Fourier transforms) are needed.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a PDE solver framework called PRISMA. This framework is a diffusion model which encodes the physical information into the structure of the denoiser, but not reinforced through physical loss. The embedding of physical information is through a mechanism called Spectral Residual Attention (SRA), which includes the transformation of physical residual into frequency domain, and then attention with the observed information also in frequency domain. The performance of their proposed method is better than other models on PDE problems with 100% unit Gaussian noise corruption.

### Strengths
- (Originality) This work introduces a new framework that can incorporate physical information other than direct calculation and backpropagation of physical loss, which seems to reduce the training time and inference steps for diffusion-based frameworks.
- (Clarity) Besides some minor issues with mathematical symbols (see weaknesses part), the general presentation of this work is easy to follow.

### Weaknesses
- (Wrong Physical Residual Calculation of NS) One of the most serious problems with this paper is that, the calculation of physical residual for nonbounded NS equations, which was adopted from DiffusionPDE, is actually wrong. The vorticity $\vec{\omega}(x,y)=\vec{\nabla} \times \vec{v}(x, y)$ is an (axial-)vector which only has $z$ component and is a function of $x, y$. Therefore, its zero divergency, $\vec{\nabla} \cdot \vec{\omega}(x,y) = \frac{\partial \omega}{\partial z} = 0$ cannot be regarded as a meaningful physical residual. The same thing happens to bounded NS. In that case, only the magnitude of $\vec{v}(x, y)$ is recorded, and one cannot take divergence on a magnitude field. This serious problem would make all the arguments for physical embedding useless, as a comparison of performance without physical residual and with a wrong physical residual does not make any sense.
- (Problems Tested) This paper compares the performance of models on a rather rare case of observations with 100% Gaussian noise. In reality if one observation is with 100% noise, it is considered a failed observation. One more serious problem is about the physical residual calculated on observation with 100% noise. This noise is high frequency and would hurt the physical residual calculated with finite difference. One cannot get convincing physical information from a physical residual that is not reliable.
- (Performance and Baseline) On more common full observation and partial observation, the performance of this model is inferior to other models like PINO and FunDPS, as reported in the appendices. The baseline of DiffusionPDE on full observation is actually available in their paper, and is much better than the results reported in the Table 6 of this paper.
- (Mathematical Symbol) In equation (1), $\mathcal{F}$ stands for the operator of PDE, but in equation (2), $\mathcal{F}$ stands for the Fourier transform operator. This reuse of symbol hurts the clarity of this paper.

### Questions
- To show that the proposed method works for NS, maybe the authors can try out with vector form of velocity field. The zero divergency of velocity vector field can be implemented as: $\vec{\nabla}\cdot \vec{v}(x, y) = \frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0$.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose PRISMA (PDE Residual Informed Spectral Modulation with Attention). They modify the global spectral path of a UNO, via a residual guided attention mechanism; which depends on the phase alignment of said PDE residuals, as well as a gate dependent on the diffusion step/noise level. This speeds up the inference time of the models by a 15-250x, while maintaining or surpassing their compared models.

### Strengths
### Novel PDE-informed guidance mechanism
Using complex-valued attention in Fourier space to modulate frequencies based on PDE residuals is genuinely novel. The authors established via gating and normalization a well rounded method to perform this task.

### Impressive speedup of the inference (for diffusion models)
The 20-step inference while achieving comparative accuracy in some tasks is impressive. Having a speedup of 15-250x is the strongest contribution of this work. This brings the neural PDE solvers a step closer to being practical in real-time applications.

### Unified framework (forward-inverse problems)
Training on both forward and inverse problem simultaneously with task-specific probabilities is elegant.

### Provided code
This made the review process easier to understand how the experiments were setup/run.

### Weaknesses
### Method is only decent on full observations, compared to other models (Table 6).
In this case PINO and the other models seem to outperform the proposed method (in speed and accuracy)

### Noise robustness claims (Table 3).
Overall this table shows that diffusion models are better suited for noisy data. You should compare PINO/FNO trained with the same data augmentation (noise injection during training). The current comparison falls flat, as this setting would be out-of-distribution data for PINO/FNO, while it is in-distribution for most diffusion methods.
A similar claim can be made for Table 7 results (sparse observations).

### More ablation runs for Table 4.
Without confidence intervals, the differences in Table 4 are questionable. Some tasks show minimal improvment or even worse performance without PDE residuals. The paper should report mean and std over multiple seeds (code only shows seed:33).

### Claim of multi-scale guidance is somewhat misleading
The paper claims to provide "multi-scale guidance at every layer of UNO", but the implementation uses interpolation by downsampling to the same resolution to match the feature map spatial dimensions. Unlike hierarchical feature extractors, this method appears to not learn different representations of PDE residuals at different scales (simple resize of the same signal). This means the multi-scale is rather a multi-resolution.

### The caption of Figure 4 appears to be incorrect
Caption does not match the figure content (mentions "inference time vs accuracy", but this is not shown)

### Questions
- Is there any intuition why FUNDPS outperforms the other methods in the sparse observation case? (Table 7)

- Did you also try out to integrate magnitude alignment into the attention? Phase alignment and using w_gain appears to perform well, I wonder if magnitude information could be used to omit the w_gain term.

- Can you provide results where PINO and other baselines are trained with the similar noise augmentation schedule?

- Can you report mean and std for +5 random seeds? (Table 4)

- Have you ablated using magnitude-only vs. phase-only alignment in frequency domain?

- Why is complex-valued attention necessary, compared to for example attention in the spatial path?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces PRISMA (PDE Residual Informed Spectral Modulation with Attention), a novel conditional diffusion neural operator for solving forward and inverse problems in the context of PDEs. To avoid the cost and potential instabilities of inference-time guidance, they propose to condition the operator directly on the (noisy and masked) PDE residuals using a spectral residual attention block. Similar to existing solvers, PRISMA operates on a joint space of observations and parameters to enable the solution of forward and inverse problems with sparse and noisy observations. Due to the guidance-free inference, PRISMA can outperform existing solvers in terms of accuracy vs. inference cost.

### Strengths
1. The spectral residual attention block seems to be a novel module for conditioning diffusion models/operators on residuals from sparse and noise observations. This can lead to improved performance, in particular for noisy observations (potentially due to the learned gating mechanism). Since no guidance is needed during inference, this further reduces the required number of steps and inference time compared to DiffusionPDE and FunDPS.
2. The paper includes a comprehensive set of experiments (from DiffusionPDE) and provides ablation studies on the conditioning type and number of required diffusion steps (showing that performance saturates quickly at 20-50 steps and validating the use of a low number of steps). Moreover, it is shown that PRISMA can improve statistics of the PDE residuals over the considered two baselines.

### Weaknesses
1. A major concern is that PRISMA considers a different setting than the two main baselines. The baselines only require paired data, but no prior knowledge of the PDE or the corruptions (type of masking/noise/etc.) during training. In other words, these guidance-based methods are *agnostic* to the corruptions and just rely on inference-time control. On the other hand, PRISMA assumes knowledge of the type of masks, noise, and PDE equations *during training*. This is a more restrictive setting and (unsurprisingly) also leads to better performance.
2. The framework of jointly modeling observations and parameters (and thus being able to solve forward and inverse problems) is taken directly from DiffusionPDE (finite-dim.) and FunDPS (infinite-dim.).
3. As mentioned in the paper, the performance is only evaluated on the dataset and baselines provided by DiffusionPDE which do not consider practically relevant problems on full time-intervals (instead of initial/terminal time problems). Due to its dependence on FFT, it seems also non-trivial to extend PRISMA to problems on irregular geometries.
4. Some of the claims seem to be too strong:
    - The proposed guidance is still not “pointwise” since the Fourier transform has a global dependency.
    - The method is not fully "gradient-free" during inference, since the gradients appearing in the PDE still need to computed (e.g. using finite-differences or Fourier differentiation).
    - While the methods has a better performance vs. inference cost trade-off, the statement that it is “at-par or better in accuracy” is not true given the sparse-observation results reported in Table 7.

**Minor:**

- The caption of Figure 6 seems to be wrong (and coincides with the caption of Figure 4).

### Questions
- It would be interesting to see the ablation when directly conditioning on the sparse and noisy observations instead of the PDE residual.
- It seems that the guidance weight for DiffusionPDE/FunDPS is not tuned properly since the reported performance is sometimes decreasing when increasing the number of steps.
- How much slower is the training due to the computations of residuals and the additional spectral residual attention block?
- Does PRISMA train different models for different noise levels for the (sparse) observation (and if not, how is the noise level sampled during training). Moreover, how are the masks sampled during training?
- It would be interesting to ablate the effect of the gating mechanism in the noisy setting.

### Soundness
1

### Presentation
3

### Contribution
2
