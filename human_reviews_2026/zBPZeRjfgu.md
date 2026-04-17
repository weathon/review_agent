# Cautious Optimizers: Improving Training with One Line of Code

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6, 6

## Abstract
AdamW has been the default optimizer for transformer pretraining. For many years, our community searched for faster and more stable optimizers with only constrained positive outcomes. In this work, we propose a \textbf{single-line modification in Pytorch} to any momentum-based optimizer, which we rename cautious optimizer, e.g. C-AdamW and C-Lion.  Our theoretical result shows that this modification preserves Adam's Hamiltonian function and it does not break the convergence guarantee under the Lyapunov analysis. In addition, a whole new family of optimizers is revealed by our theoretical insight. Among them, we pick the simplest one for empirical experiments, showing not only consistent speed-up on LLM pretraining and post-training tasks, but also better results in MAE pretraining, with minimum extra tuning on hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents "Cautious Optimizers," a simple, one-line modification for momentum-based optimizers like AdamW. The method only applies parameter updates when their direction aligns with the sign of the current gradient, preventing counter-productive steps. This change is claimed to accelerate model training (including for LLMs and computer vision) with minimum extra tuning on hyperparameters, all while preserving theoretical convergence guarantees under the Lyapunov analysis.

### Strengths
- The change is trivial (a single line of code) and applicable to existing momentum-based optimizers.

- The method boosts performance without the need for costly and time-consuming hyperparameter retuning.

- The intuitive idea is supported by theoretical analysis, ensuring that convergence guarantees are maintained.

- It demonstrates consistent speed-ups in various high-impact domains, including LLMs and image classification.

### Weaknesses
N/A

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a simple modification to momentum-based optimizers called Cautious Optimizers, implemented by masking update directions that disagree in sign with the current gradient. The authors argue that this one-line change ensures monotonic decrease in loss for small step sizes, preserves the Lyapunov/Hamiltonian structure of momentum dynamics, and empirically improves convergence speed and training stability. Experiments on toy problems, LLM pretraining (100M LLaMA on C4 and FineWeb-Edu), and vision tasks (Mini-ImageNet with ViT) confirm that the proposed modification can improve the performance of some popular optimizers.

### Strengths
- The paper addresses a practically relevant problem.

- The proposed method is very simple (a one-line change in PyTorch) and broadly applicable to many optimizers.

- The experiments show consistent (but modest) gains in LLM pretraining, image classification, and toy settings, with improved stability and slightly better downstream performance.

- The theoretical analysis tries to conceptually connect the masking heuristic to theoretical convergence guaranties.

### Weaknesses
1. Theoretical claims are not fully convincing. The results demonstrate that cautious optimizers can reduce the loss more than the original optimizers in a *single step*, but not throughout the full optimization trajectory. In line 84, the authors claim that "Our theoretical analysis shows that the modified algorithm converges to local optima under mild conditions on the base optimizers", but the presented results do not establish such convergence.

2. Lack of stochastic analysis (the paper only considers deterministic gradient setting).

3. Empirical improvements are modest. Most improvements (e.g., Table 2) are around $0.1–1$%, which may fall within variance across training runs.

4. The abstract claims "consistent speed-up", but training time or throughput overhead is not reported (the results focus on perplexity and accuracy).

5. No results are shown on other architectures (CNNs, RNNs) or non-transformer tasks.

6. Some equations in the appendix overflow the page margins.

7. Minor issues: several typos and grammatical errors (e.g., 'methods normally requires' in line 43, 'The follow is a comparison result' in line 240, 'catuious' in line 390, 'generalit' in line 444),  inconsistent boldface notation (e.g., equation (1)), formatting issues in References.

### Questions
1. Could the authors address the concerns above?

2. In Table 1, why were C-AdamW runs with learning rates 1e-4 and 3e-4 omitted?

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
4

### Summary
The paper proposes “Cautious Optimizers,” a one-line masking of momentum updates that zeros coordinates where the proposed update and current gradient have opposite signs, optionally rescaled by the active-mask ratio (Alg. 1). The theory is framed via a Hamiltonian+Descent view (continuous time) and per-step comparison results (discrete time). Experiments include a 2-D toy, Mini-ImageNet, and LLM pretraining up to 1.2B parameters.

### Strengths
1. This paper proposes a simple mechanism (coordinate-wise sign check) that is easy to implement; clear statement that it promotes monotone decrease of the loss for small steps. 


2. Empirical results cover both vision and language, and generally show small but consistent improvements

### Weaknesses
1. The key idea of masking coordinates where the gradient and velocity have opposite signs is meant to promote descent. But requiring sign consistency on every coordinate feels too strict — usually, it's enough for the update direction and gradient to have a positive inner product. Also, in stochastic settings with noisy gradients, enforcing per-coordinate alignment could hurt rather than help. Plus, the paper motivates this with “monotonic decrease,” but per-step monotonicity isn’t necessary for faster convergence — e.g., Nesterov's method is non-monotone. This raises the question: is this motivation really essential?

2. In Theorem 2.3 and 2.4, the discrete-time analysis relies on certain properties of the masking function (like Δ(vₓ) ≥ 0). But Algorithm 1 uses a non-smooth hard indicator (1(uᵢgᵢ > 0)) and a heuristic rescaling. These look inconsistent. Is this difference purely due to the hard masking being non-smooth? If so, it would be good to clarify the gap between what’s proved and what’s implemented.

3. The toy experiment in §3.1 is just 2D and too simple. The LLM results in §3.2 only go up to 1.2B parameters, which is relatively small. To claim relevance to large-scale pretraining, results on models at 7B scale or above would be much more convincing.


4. It feels odd to postpone related work to the very end (§4). This makes §2.1 hard to follow since many readers won’t be familiar with the Hamiltonian perspective. I'd recommend moving related work earlier or giving at least a brief summary in §2.

### Questions
See comments 2 of weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce cautious optimizers which is a modification applicable to any momentum-based optimizer where the main idea is to update parameters mainly when the proposed update direction and the gradient have the same sign. The authors show theoretically that it preserves the optimizer’s Hamiltonian structure and guarantees monotonic decrease of the loss under sufficiently small steps. Empirically, C-AdamW, C-Lion, etc. yield performance improvements in large-scale pretraining and image classification.

### Strengths
- The idea is quite elegant can be seamlessly applied to all momentum-based optimizers without introducing new hyperparameters. 

- The authors situate the method within the Hamiltonian Descent framework, which were used to analyze Adam and Lion. Theorem 2.1 and Corollary 2.2 offers a clear interpretation of cautious masking as energy-preserving damping.

- The discrete-time results confirm that each cautious step decreases the loss more efficiently than the base optimizer under µ-smoothness.

- The experiments show improvements without re-tuning hyper-parameters — showing that the cautious modification preserves hyperparameter stability.

- Overall, writing in the paper is clear.

### Weaknesses
- The claim that cautious optimizers “do not get stuck at non-stationary points even when the update is fully masked out” is only partially justified. The authors state that momentum dynamics will eventually realign updates and gradients, but did not provide no empirical analysis of how long this alignment requires. Maybe exploring the failure cases near saddle points or flat regions (regimes that dominate the modern deep learning optimization) could help?

- Empirically, the improvements are modest and sometimes fall within the noise range of large-scale training. For example, Table 2 reports ≤ 1 % perplexity gains at max. The comparison is also relatively limited i.e., Section 4 mainly discusses related optimizers like AdamW, Lion, but omits several recent and directly relevant baselines such as AdaBelief, Adan and SOAP even when these methods employ simple directional or normalization modifications. 

- the effect of the scaling factor introduced in Eq. (1) on the magnitude of updates and on the effective learning-rate distribution is not thoroughly analyzed. Since α directly scales updates in proportion to the ratio, could it change the convergence behavior in anisotropic curvature regions?

### Questions
Please refer to the comments in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a plug-in method for improving momentum-based optimization methods (such as AdamW) named as Cautious optimizers. This method applies a coordinate-wise mask according to if the proposed update direction aligns in sign with the current gradient. Theoretically, the authors analyze the method in a proposed Hamiltonian/Lyapunov framework. With smoothness-based arguments, they show that cautious optimizers preserve the base optimizer’s Hamiltonian descent, can ensure monotonic decrease of the loss and further accelerates it. Empirically, they evaluate the proposed method first on a 2D toy objective function, then on language and vision tasks. Consistent improvements on convergence rates and training performances are shown.

### Strengths
1. The paper proposes cautious optimizers, a single yet effective trick for improving momentum-based gradient methods. The method is lightweight and easy to implement.  
2. The authors argue the soundness of their proposed approach from a Hamiltonian/Lyapunov perspective, which might be of individual interest for future optimizer design.  
3. Performance gains are shown empirically across different tasks, demonstrating the effectiveness and wide applicability of the method.

### Weaknesses
1. The LLM pretraining experiments use the same learning rate when comparing with original AdamW and Lion. This might not be a fair comparison since cautious optimizers use a rescaling factor $\\alpha$. Although the rescaling factor can be normalized, it is still hard to say if the same learning rate means the same thing for different optimizers.  
2. The reported performance gains in Table 3 are modest, making interpretation sensitive to data and random noise. Reporting the number of random seeds tried would strengthen the claims.

Minor comments: Typo in line 390: “catuious”.

### Questions
Why is weight decay not included in cautious optimizers but performed after it? What if one uses other regularizers instead of the L2 norm, such as the KL regularizer in RL?

### Soundness
3

### Presentation
3

### Contribution
3
