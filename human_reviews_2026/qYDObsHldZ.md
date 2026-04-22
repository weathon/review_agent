# Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 2, 4, 8

## Abstract
Guided diffusion sampling relies on approximating often intractable likelihood scores, which introduces significant noise into the sampling dynamics. We propose using adaptive moment estimation to stabilize these noisy likelihood scores during sampling. Despite its simplicity, our approach achieves state-of-the-art results on image restoration and class-conditional generation tasks, outperforming more complicated methods, which are often computationally more expensive. We provide empirical analysis of our method on both synthetic and real data, demonstrating that mitigating gradient noise through adaptive moments offers an effective way to improve alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a very simple plug-in modification to plug-and-play diffusion guidance: keep exponential moving averages of the likelihood-gradient during sampling (Adam-style first/second moments) and use the stabilized gradient to guide updates. The authors instantiate this on DPS ("AdamDPS") and on classifier guidance ("AdamCG"), provide a 2-D GMM toy study, and report empirical gains on various inverse problems.

### Strengths
- The change is minimal (Adam-style moments around the guidance gradient) and easy to graft onto existing DPS/CG code paths.

- The toy GMM illustrates how DPS becomes unstable under noisy likelihood gradients whereas AdamDPS stabilizes trajectories; several restoration benchmarks also show improvements in LPIPS/FID vs. DPS and some training-free baselines.

### Weaknesses
- Technical novelty is minimal. The paper just adds Adam-style gradient update to DPS/CG at inference time. There is no new objective, solver, or estimator; the contribution is only an optimizer wrapper around an existing guidance gradient (Algorithms 1–2). This is a straightforward, well-known idea from stochastic optimization brought over without new theory specific to diffusion guidance beyond intuition. This falls short of ICLR's bar for conceptual advance. The empirical results are useful but do not compensate for the lack of technical depth or new insight into diffusion guidance.

- Potential overfitting in Fig. 5. The combination of highest accuracy with worst FID suggests the guidance may be overfitting to the evaluation classifier rather than improving image quality. Classification accuracy alone does not imply good samples if the images are classifier-friendly but perceptually poor. Please use a different (held-out) classifier for evaluation than the one used during guidance/training, and report cross-model accuracy (and, ideally, human or ImageReward/VQA scores) to confirm the gains are not due to classifier overfitting.

### Questions
- Can you replicate gains on other plug-and-play frameworks (e.g., DDRM, PiGDM, MPGD-style data-space updates) and other backbones/samplers to support the "surprisingly effective" claim beyond DPS/CG?

- Beyond demonstrating that AdamDPS smooths noisy guidance, is there substantive novelty beyond applying an optimizer wrapper to DPS/CG?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the classifier guidance and finds out that classifier guidance approximates the likelihood score which often includes a lot of noise. The paper proposes to regularize the guidance sampling process with ADAM, a popular method in neural network optimization.

### Strengths
1. The paper is well written
2. the method is easy to understand

### Weaknesses
1. The novelty might be the concern. The Adam is not novel and the guidance is also not novel. The use of Adam during sampling process is also not new (this paper has similar idea https://www.ijcai.org/proceedings/2024/0157.pdf). The only difference is that Adam is applied to guidance term instead of denoising term as in the paper. 
2. The paper does not provide a new fundamental observations or scientific hypothesis.
3. In terms of applications, the proposed method currently could not extend to classifier-free guidance which is a very popular guidance right now. 
4. The quantitative results are not provided, different metrics are not considered e.g FID, IS, Rec, Prec, CLIP, GenEval, T2I Bench.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the likelihood approximation noise in training-free plug-and-play diffusion models such as DPS. These models estimate the posterior score using $\nabla_xt log⁡p(xt∣y)=\nabla_xt log⁡p(y∣xt) + \nabla_xt log⁡p(xt)$, where the likelihood term $\nabla_{x_t}\log p(y|x_t)$ is approximated by a differentiable surrogate $\nabla_{x_t}L(f_\phi(\hat x_{0|t}),y)$. At high noise levels (early and mid diffusion steps), this estimate becomes highly unstable, leading to biased and noisy sampling.

The paper proposes maintaining exponential moving averages (EMAs) of the first and second moments of the likelihood gradient, similar to Adam. By tracking running averages of the gradient and its squared values across timesteps and adaptively rescaling updates, the approach stabilizes the guidance direction and reduces variance. Two variants are presented: AdamDPS for diffusion posterior sampling and AdamCG for classifier guidance.

Experiments demonstrate consistent improvements: on inpainting, deblurring, and $16\times$ super-resolution tasks (ImageNet and Cat datasets), AdamDPS achieves 1–2 dB PSNR gains and lower LPIPS/FID than DPS, UGD, and LGD. On CIFAR-10 and ImageNet 64×64 class-conditional generation, AdamCG improves top-1 and top-10 accuracy without significant degradation in FID. Synthetic 2-D Gaussian mixture experiments confirm that adaptive moments smooth the gradient variance peak occurring at intermediate diffusion steps.

### Strengths
Addresses a key limitation—noisy and biased likelihood gradients—with a simple, general, and low-cost modification. 

Empirically robust across datasets, with clear ablations on noise scales and $(\beta_1, \beta_2)$.

Minimal implementation effort with measurable gains in quality and stability.

### Weaknesses
Limited to gradient-based plug-and-play samplers; not applicable to variational or optimization-based methods such as RED-Diff.

No theoretical analysis showing that adaptive moments preserve unbiasedness or convergence to the correct posterior $p(x_0)\exp[-r(x_0,y)]$.

Missing comparisons with RED-Diff, MPGD, and TMPD, which already incorporate momentum or adaptive scaling.

No examination of how the moving averages interact with the diffusion dynamics or affect the stationary distribution.


RED-Diff — M. Mardani, J. Song, J. Kautz, A. Vahdat. A Variational Perspective on Solving Inverse Problems with Diffusion Models, arXiv:2305.04391 (2023).

MPGD — Y. He, N. Murata, C.-H. Lai, Y. Takida, T. Uesaka, D. Kim, W.-H. Liao, Y. Mitsufuji, J. Z. Kolter, R. Salakhutdinov, S. Ermon. Manifold Preserving Guided Diffusion, ICLR 2024.

TMPD — B. Boys, et al. Tweedie Moment Projected Diffusions for Inverse Problems, arXiv:2310.06721 (2023).

### Questions
How does AdamDPS fundamentally differ from optimization-based samplers such as RED-Diff that already employ Adam updates?

Does maintaining EMAs across timesteps bias the effective sampling trajectory?

How would AdamDPS perform against RED-Diff or TMPD under the same compute and step budgets?

Could combining adaptive moments with Monte-Carlo smoothing (e.g., LGD) provide further gains?

How sensitive is the performance to the number of diffusion steps or when applied to ODE-based samplers such as DDIM?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces AdamDPS, a new plug-and-play method for diffusion sampling. By noting that the guided sampling update is gradient ascent using the score of the likelihood distribution, the authors introduce Adam-style moment exponential moving averages to stabilize the overall update. The resulting method is tested on a toy GMM example, demonstrating improvements in sampling quality over the baseline diffusion posterior sampling (DPS). Experiments are then extended to reconstruction tasks on image datasets, where AdamDPS demonstrates the best combination of both FID and LPIPs. Ablations are performed on task-difficulty, Adam hyperparams, and wall-clock time.

### Strengths
This paper is complete and well-presented. Overall, the method is effective and experiments are thorough. I would recommend it for acceptance.

**(S1)**: Simple and effective. The core idea of introducing Adam-style moment stabilization is simple and effective for diffusion guided plug-and-play sampling. The paper clearly motivates the problem with previous approaches, proposes a sound improvement, and demonstrates the improvement via experimental results.

**(S2)**: Clear improvement over prior work on reconstruction tasks. Results in Fig 3 clearly demonstrate superior reconstruction quality and sample quality of AdamDPS. 

**(S3)**: Comprehensive ablations and analysis. The ablations on task difficulty is valuable and demonstrates the core motivation for the method-- as the task gets noisier, the stabilization introduced by Adam-style moments results in better performance. Other ablations on number of sampling steps and moment coefficient values are useful for any future users of this method. 

**(S4)**: Computational efficiency. The ablation on wall-clock time confirms that this method induces little to no overhead over baselines, which is valuable.

### Weaknesses
**(W1)**: Slightly mixed results on class-conditional sampling. The FID on ImageNet and CIFAR-10 seems worse than baselines, even as the classification accuracy is much better. 

**(W2)**: Slight regressions over baselines on easy tasks. In Fig 6, for easier super-resolution or deblurring tasks, AdamDPS is slightly worse than TFG. This again slightly clouds the otherwise clear narrative of the paper. No explanation for this is given.

**(W3)**: Details on the method are missing. A clearer explanation of the models used, experimental setup, metrics would be very valuable for clarity. For example, while an ablation on $\beta_1$ and $\beta_2$ were performed, it's not clear what the optimal values are for each task.

### Questions
**(Q1)**: Is there a missing increment to $k$ in algorithms 1 and 2?

### Soundness
4

### Presentation
4

### Contribution
4
