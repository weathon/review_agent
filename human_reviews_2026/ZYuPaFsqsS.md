# Accelerating Denoising Generative Models is as Easy as Predicting Second-Order Difference

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
High-fidelity diffusion and flow models remain latency-bound at inference, motivating acceleration that leaves pretrained weights untouched. We ask: what is the $\\textit{minimal yet principled}$ way to accelerate sampling? Under a simple and mild budget, when uniform reduction targets more than $2\\times$ speedup, each three-step window contains at most one fresh denoiser call, creating a structural scarcity of real signals.
From this constraint, we isolate the $\\textit{observed}$ information at step $t$—the fresh output $\\psi_t$ and its backward difference $\\Delta \\psi_{t}^{(1)}=\\psi_t-\\psi_{t+1}$—and show it induces a uniquely minimal, affine-exact second-order predictor $\\hat\\psi_{t-1}=2 \\psi_t- \\psi_{t+1}$.
We prove that, under this scarcity, the two-point second-order rule is the information-consistent optimum: it is BLUE among linear two-point estimators.
Naively chaining this predictor across consecutive steps destabilizes sampling by compounding approximation errors.
We resolve this by $\textit{reusing the observed tuple}$ in an interleaved zig–zag schedule that prevents back-to-back extrapolations and controls variance. 
The resulting method, $\textbf{ZEUS}$, is a zero-overhead, backbone- and parameterization-agnostic plug-in requiring no retraining, no feature caches, and no architectural changes.
Across images and video, ZEUS consistently moves the speed–fidelity Pareto frontier outward versus recent state-of-the-art, delivering up to $3.2\\times$ end-to-end speedup while improving perceptual similarity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ZEUS accelerates diffusion/flow sampling without retraining by leveraging a budget‑induced “signal scarcity,” where each three‑step window contains at most one fresh denoiser call. Using only the fresh output and its backward difference, it derives a uniquely minimal, affine‑exact second‑order predictor proven BLUE among linear two‑point estimators, and stabilizes chaining via an interleaved zig–zag schedule that avoids back‑to‑back extrapolations. The zero‑overhead plug‑in is backbone‑ and parameterization‑agnostic, requiring no caches or architectural changes, and consistently pushes the speed–fidelity Pareto frontier outward on images and video with substantial end‑to‑end speedups and improved perceptual similarity.

### Strengths
1. This paper conducts extensive experiments to show the superiority.
2. It provides a theoretical analysis.

### Weaknesses
1. This paper is hard to read due to poor writing.
2. The proposed method is very similar to Taylor ($\mathcal{O}=1$) for obtaining $\widehat\psi_{t-1}$.
3. I believe the method induces lots of memory costs, which is not justified by the authors.
4. The authors only provide comparisons with DiCache for FLUX. I believe they should include comparisons for the other models.  Also, the settings for baselines in this paper are not clear.
5. The performance of TaylorSeer is inconsistent (a huge gap) with its original paper (Tab. 1 for FLUX).

### Questions
N/A

### Soundness
2

### Presentation
2

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
This paper introduces ZEUS (Zero-cost Extrapolation-based Unified Sparsity), a training-free and backbone-agnostic method for accelerating denoising generative models—such as diffusion and flow-based architectures—without modifying model weights, retraining, or using feature caches. Empirically, ZEUS is evaluated across five backbones (e.g., Stable Diffusion 2, SDXL, Flux.1-dev, Wan 2.1, CogVideoX v1.5) and two solvers (Euler and DPM++), consistently pushing the speed–fidelity Pareto frontier forward. It achieves up to 3.2× end-to-end speedup while maintaining or improving perceptual metrics such as LPIPS, FID, and PSNR.

### Strengths
1. The authors formally prove the BLUE optimality and second-order accuracy of the proposed predictor, and justify why higher-order extrapolants are suboptimal or unstable under limited fresh computation. The combination of theory (bias–variance characterization, affine invariance) with practical implementation (zig–zag reuse schedule) demonstrates high technical soundness.

2. ZEUS has strong practical and scientific significance. It provides a zero-cost, architecture-agnostic plug-in that can be directly applied to large-scale diffusion and flow models such as SDXL, Flux, and Wan2.1. The method delivers up to 3.2× inference acceleration while maintaining or even improving fidelity (LPIPS/FID), which represents a meaningful advance in the efficiency of generative modeling.

### Weaknesses
1. Evaluation scope and metric coverage.

Although the experiments are extensive, the paper primarily reports traditional similarity metrics (PSNR, LPIPS, FID) and qualitative visual comparisons. However, these metrics capture perceptual closeness rather than semantic or compositional fidelity. For text-to-image generation, including evaluations such as GenEval or DPG-Bench would better reflect whether ZEUS preserves prompt alignment and fine-grained attribute consistency after aggressive skipping. This would strengthen claims of maintaining “fidelity” beyond pixel-level similarity.

2. No discussion on potential integration with caching or mixed-precision methods.

While ZEUS is intentionally designed as a zero-overhead plug-in, the paper stops short of exploring synergy with complementary acceleration families, such as feature-cache re-use (DeepCache, AB-Cache). Discussing how ZEUS could combine with these orthogonal techniques to achieve compound acceleration would expand its utility.

### Questions
Lack of analysis on strong modern backbones.

All reported results focus on mid-scale or publicly available diffusion/flow models (e.g., SDXL, Flux.1-dev, Wan2.1). The paper does not evaluate on more recent state-of-the-art text-to-image backbones like FLUX-Krea, and Qwen-Image that exhibit stronger coupling between text and visual features. Since ZEUS claims architecture-agnostic generalization, demonstrating performance on these modern DiT-based models would further substantiate scalability and practical relevance.

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
3

### Summary
This paper proposes a training-free acceleration method for diffusion models named ZEUS. Based on an information scarcity assumption, the authors derive that a second-order difference predictor is the only form satisfying the BLUE condition. They further introduce an alternating zig–zag reuse mechanism to balance stability and accuracy. The method is zero-cost, requires no feature cache or structural modification, and can be directly embedded into any diffusion or flow-based model during inference. Experiments on Stable Diffusion, SDXL, Flux, Wan, and CogVideoX demonstrate consistent acceleration while maintaining or even improving generation quality, showing that ZEUS is a simple, stable, and general acceleration paradigm.

### Strengths
1. **Theoretical simplicity with solid foundation.** The paper rigorously derives that, under specific conditions, the second-order difference serves as the only optimal estimator satisfying the BLUE criterion, leading to a training-free and architecture-invariant extrapolation scheme.

2. **Efficient and stable design.** The proposed Zig–Zag reuse mechanism effectively suppresses multi-step extrapolation drift while maintaining the accuracy of second-order prediction, achieving stable acceleration with zero additional computational cost.

3. **Comprehensive experiments and strong generality.** ZEUS consistently improves the speed–quality trade-off across multiple image and video diffusion models, verifying its practicality and plug-and-play generalization capability.

### Weaknesses
1. **Overly structured core assumption.** The key theoretical derivation relies on a constrained premise—when aiming for acceleration beyond 2×, only one of every three consecutive steps can involve a fresh denoiser call. In real-world inference, non-uniform or adaptive scheduling is common, where this assumption may not hold. The paper has not provided theoretical or empirical evidence that the optimality can generalize to such practical cases.

2. **Lack of global stability and error analysis.** While the paper discusses overshoot phenomena and gives local bias–variance scaling laws, these analyses are confined to fixed short segments. In actual diffusion sampling, the process is a long chained integration. The paper does not provide a global upper bound or long-term error propagation analysis, making current stability conclusions largely empirical.

3. **Limited comparison and ablation scope.** The experiments mainly compare with training-free baselines, which are not strictly comparable to ZEUS. Stronger baselines or hybrid cache–predict strategies could be added. Moreover, the ablation design focuses primarily on the authors’ own setting, limiting the generality of the conclusions.

### Questions
1. **Theory.** Under non-uniform or adaptive time-step scheduling, does the theoretical optimality of the second-order predictor still hold?
Can the authors provide a global stability or error bound for the zig–zag reuse process across long sampling trajectories?
If a formal bound is not feasible, what is the empirically observed maximum jump length before instability occurs?

2. **Experiments.** It would strengthen the paper to include stronger baselines and high-order solvers for comparison. In addition, although the paper repeatedly claims model-agnostic generalization, experiments are mostly in visual diffusion models; evaluations on audio or text diffusion tasks (with different smoothness characteristics) are needed to validate the generality claim.

3. **Implementation cost.** Although the method is described as zero-cost at the operator level, additional tensor construction, copying, and logic overhead may still exist. Could the authors provide quantitative profiling data to verify that the reported acceleration indeed reflects the end-to-end wall-clock gain?

### Soundness
3

### Presentation
3

### Contribution
2
