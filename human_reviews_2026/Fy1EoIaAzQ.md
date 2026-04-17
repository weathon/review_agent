# EasyTune: Efficient Step-Aware Fine-Tuning for Diffusion-Based Motion Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 10, 6

## Abstract
In recent years, motion generative models have undergone significant advancement, yet pose challenges in aligning with downstream objectives. Recent studies have shown that using differentiable rewards to directly align the preference of diffusion models yields promising results. However, these methods suffer from (1) inefficient and coarse-grained optimization with (2) high memory consumption. In this work, we first theoretically and empirically identify the *key reason* of these limitations: the recursive dependence between different steps in the denoising trajectory. Inspired by this insight, we propose **EasyTune**, which fine-tunes diffusion at each denoising step rather than over the entire trajectory. This decouples the recursive dependence, allowing us to perform (1) a dense and fine-grained, and (2) memory-efficient optimization. Furthermore, the scarcity of preference motion pairs restricts the availability of motion reward model training. To this end, we further introduce a **S**elf-refinement **P**reference **L**earning (**SPL**) mechanism that dynamically identifies preference pairs and conducts preference learning. Extensive experiments demonstrate that EasyTune outperforms DRaFT-50 by 7.7% in alignment (MM-Dist) improvement while requiring only 31.16% of its additional memory overhead and achieving a **7.3$\times$** training speedup. The project page is available at this [link](https://xiaofeng-tan.github.io/projects/EasyTune/index.html).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EasyTune, a step-aware fine-tuning scheme for diffusion-based motion generation that optimizes at each denoising step using stop-gradient to break recursive dependencies; this yields fine-grained updates and O(1) memory rather than O(T). It trains a reward model without human labels via Self-refinement Preference Learning (SPL), mining preference pairs from retrieval results and fine-tuning a retrieval backbone as a differentiable reward.

### Strengths
- Clear motivation and theory; step-wise objective avoids vanishing gradients and heavy graph storage.
- Strong empirical results on alignment metrics (R-Precision/MM-Dist) with reduced memory footprint.
- Practical reward learning via SPL removes dependence on expensive human preference data.

### Weaknesses
- Limited discussion on when to use noise-aware vs. one-step (ODE) rewards and their sensitivity.
- Reward model is retrieval-based, which may under-capture physics/realism and encourage metric chasing. (Authors note limited physical grounding.)

### Questions
- Are the metrics giving a distorted signal? R-Precision exceeds the “Real” upper bound (0.581 > 0.511), yet FID is still notably higher than Real (0.132 vs 0.002). How do you interpret this divergence, and does it indicate semantic alignment without true distributional realism?
- Why must rewards be differentiable here? In RLHF for diffusion, rewards need not be differentiable. For motion, why is a differentiable reward essential—especially when non-differentiable motion metrics (e.g., foot-skate or slip measures) could be used as rewards? 
- What exactly is being aligned? In EasyTune + SPL, is the alignment primarily to text–motion retrieval similarity, to human preference proxies, or to motion-quality attributes (e.g., fluency/physicality)? Please specify the concrete targets the gradients optimize toward.
- Missing citations ( PhysDiff, ReinDiffuse, Aligning Human Motion Generation with Human Perceptions ).
- I'm glad to raise the score if the author can address my concerns.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
# Summary
The paper aims at aligning motion diffusion models with downstream objectives using differentiable rewards. The authors first diagnose a root cause behind prior methods’ inefficiency and memory use: recursive gradient dependence across denoising steps. They propose EasyTune, which fine‑tunes at each denoising step by stopping gradients through the current sample and backpropagating only through the local denoiser. This decouples steps, enabling dense, fine‑grained updates with O(1) step memory. To address the scarcity of motion preference data, they introduce Self‑refinement Preference Learning (SPL) that adapts a retrieval model into a reward model without human‑labeled pairs. 
Experiments on HumanML3D and KIT‑ML across multiple backbones (MLD/MLD++, MotionLCM, MDM, MotionDiffuse) report consistent improvements and lower peak memory than trajectory‑level fine‑tuning baselines.

### Strengths
- The paper cleanly shows how chain‑rule recursion creates sparse/vanishing gradients and large memory graphs and contrasts chain vs. step optimization. The motivation is clear and reasonable.
- The proposed method seems to be a simple, general, and effective idea: The per‑step objective with `sg(·)` is easy to implement on standard motion diffusers.
- Results span two datasets and several backbones; on HumanML3D, EasyTune improves R‑Precision/FID/MM‑Dist vs. DRaFT/AlignProp/DRTune while using less peak memory than those baselines. EasyTune also achieves faster convergence curves for optimization.
- The authors promised to release the code and models.
- Most details are provided in the paper and appendix. Video result (website) is well-made.

### Weaknesses
I am a bit worried about the technical novelty (but it seems ok: The fine-tuning input x_t can be obtained by denoising sampling - seems new). But Theorem 1 is a direct chain‑rule decomposition (can it be called a Theorem?); Theorem 2 is the local derivative after inserting `sg(·)`. Both are too straightforward to be called Theorem...  Besides, SPL improvements are modest... On HumanML3D retrieval, SPL improves ReAlign by R@1 +2.5% / R@3 +1.4%; on KIT‑ML, R@5 +2.2%. In the Limitation section, the authors note SPL’s reliance on retrieval mining may introduce noisy/ambiguous pairs.

Minor:
- The Implementation section states LR=2×10⁻⁴, batch=128, whereas Tab.S1 shows LR=1×10⁻⁵, batch=256 for all methods including EasyTune?
- Please also end‑to‑end costs, EasyTune’s total compute/time…Fig.5’s “w/o BP” curves illustrate the theoretical O(1) vs. O(T) difference without backprop; training, however, always with backpropagates.

### Questions
Besides the questions in the above section: 

The objective uses uniform \(t \sim U(0, T)\), yet the paper argues early, high‑noise steps are critical and under‑optimized by chain methods. Would non‑uniform sampling or up‑weighting early steps help?

In SPL, the authors set K=10. How do different K and different retrieval pools (subset vs. full) impact reward quality and final generation?

Any failure cases (I mean figures/videos)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper addresses the challenge of efficiently fine-tuning diffusion-based motion generation models using differentiable rewards, which suffer from coarse-grained optimization and extremely high memory usage due to recursive gradient dependencies across denoising steps. The paper propose EasyTune, a step-aware fine-tuning framework that optimizes the diffusion model at each denoising step independently instead of over the full trajectory. In EasyTune, the model computes the reward of the current noised motion, applies a stop-gradient operation to decouple dependencies, updates parameters locally, and clears the computation graph before moving to the next step—achieving fine-grained supervision and constant memory cost. To overcome the lack of human preference data, the paper further introduces Self-refinement Preference Learning (SPL), which adapts a pre-trained text-motion retrieval model into a reward model by automatically mining motion preference pairs from retrieval errors and fine-tuning them using a KL-divergence objective. Experiments on HumanML3D and KIT-ML with six pre-trained motion diffusion models show that EasyTune significantly outperforms prior fine-tuning methods such as DRaFT and DRTune.

### Strengths
- The paper addresses an important and timely problem in diffusion model fine-tuning, providing a thorough analysis of the limitations in existing differentiable reward methods.
- The proposed EasyTune framework is conceptually clear and mathematically well-formulated, offering a principled solution to recursive gradient and memory inefficiency issues.
- The authors conduct extensive experiments and comparisons across multiple datasets and backbone models, demonstrating consistent and substantial performance gains with strong quantitative and qualitative evidence.

### Weaknesses
The paper’s presentation could be improved — some sections are densely formatted and may benefit from clearer visual structure (e.g., spacing, figure placement, and paragraph organization) to enhance overall readability.

### Questions
Can the proposed method be applied to other domains such as image generation?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents EasyTune, a step-aware fine-tuning method for diffusion-based motion generation. The approach aims to improve the efficiency of differentiable reward optimization by decoupling recursive dependencies across denoising steps. A Self-refinement Preference Learning (SPL) mechanism is also introduced to train reward models without human annotations. Experimental results on standard benchmarks indicate performance and memory advantages over existing methods, suggesting that the proposed framework is effective and practically relevant.

### Strengths
1. The paper clearly explains the main problem in existing diffusion fine-tuning methods and gives a straightforward way to make the optimization more efficient.

2. The experiments are thorough and show steady improvements on several benchmarks and models.

3. The method can be applied in practice since it does not rely on human-labeled data for reward learning.

4. The writing and presentation are clear and easy to follow.

5. The video results presented in the supplementary materials look decent.

### Weaknesses
### Major Concerns
1. In the proposed Self-refinement Preference Learning (SPL), a pre-trained text-to-motion retrieval model is used for preference evaluation. It would be important to clarify how critical the choice of this model is. For example, if a weaker retrieval model were used, would the overall performance of EasyTune be significantly affected? Some discussion or analysis on this point would strengthen the argument, especially since the retrieval model is further fine-tuned.

2. While the paper emphasizes the large reduction in GPU memory, the efficiency analysis could be expanded. It would be helpful to report additional metrics such as GFLOPs, inference/training time, and the GPU type used, to give a more complete picture of the claimed efficiency gains.

3. The theoretical analysis, although helpful for understanding the recursive dependency issue, remains somewhat limited. A more detailed examination of convergence behavior or optimization stability under step-wise updates would improve the technical depth.

4. The SPL module depends on automatically mined preference pairs, which could introduce noise or bias. Some quantitative analysis on the reliability or noise level of these pseudo pairs would make the approach more convincing.

### Minor Concerns
- In Line 78, the symbol “θ” appears too close to the “p” in “depended”; it would be better to slightly adjust the spacing for readability.

### Questions
1. Could the authors provide more details on how the step-wise optimization interacts with the diffusion time schedule? For instance, does the reward weighting or learning rate vary across steps, and how sensitive is the method to this design?

2. In Eq. (7), the stop-gradient operation ensures local optimization per step. Does this affect the global consistency of the final denoised result? Have the authors observed any degradation in smoothness or temporal coherence of the generated motions?

3. Regarding the SPL mechanism, how often are new preference pairs mined during training? Is the mining process iterative (i.e., reward model updated → new pairs mined), or is it performed only once before fine-tuning?

4. The proposed method is demonstrated on motion generation tasks. Do the authors believe EasyTune can generalize to other diffusion domains (e.g., image or video generation), or are there task-specific assumptions that limit its applicability?

Looking forward to hearing from the authors for more explanation and discussion.

### Soundness
3

### Presentation
4

### Contribution
3
