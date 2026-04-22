# BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Recent progress in aligning image and video generative models with Group Relative Policy Optimization (GRPO) has improved human preference alignment, but existing variants remain inefficient due to sequential rollouts and large numbers of sampling steps, unreliable credit assignment,as sparse terminal rewards are uniformly propagated across timesteps, failing to capture the varying criticality of decisions during denoising.
In this paper, we present BranchGRPO, a method that restructures the rollout process into a branching tree, where shared prefixes amortize computation and pruning removes low-value paths and redundant depths.  
BranchGRPO introduces three contributions: 
(1) a branching scheme that amortizes rollout cost through shared prefixes while preserving exploration diversity; 
(2) a reward fusion and depth-wise advantage estimator that transforms sparse terminal rewards into dense step-level signals; and 
(3) pruning strategies that cut gradient computation but leave forward rollouts and exploration unaffected.  
On HPSv2.1 image alignment, BranchGRPO improves alignment scores by up to \textbf{16\%} over DanceGRPO, while reducing per-iteration training time by nearly \textbf{55\%}.  
A hybrid variant, BranchGRPO-Mix, further accelerates training to 4.7× faster than DanceGRPO without degrading alignment.
On WanX video generation, it further achieves higher motion quality reward with sharper and temporally consistent frames.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a dynamic branching and pruning approach that seems to amortize roll-out costs, and improve efficiency. Authors provide detailed mathematical formulation of the proposed procedure as well as a suite of quantitative and qualitative results.

### Strengths
- The paper is reads well, and is well structured offering a coherent narrative and useful insights. 
- The idea of dynamic branching and pruning is an interesting perspective applied to GRPO, and it seems to have some impact in terms of run-time efficiency.
- The results seem to be promising, even though the impact might not be as significant. More on this in the next section.

### Weaknesses
- The impact of the proposed idea feels slightly inflated. The results in Table 1, Figure 4 and 7 confirm this point. Could you please elaborate how/why you find the impact (beyond relative decrease in run-time) significant? How should one read and interpret an improvement of $1.594 \rightarrow 1.625$ or $3.59 \rightarrow 3.69$? I don't think it also makes sense to accentuate on the iteration time $148$ while the arguments regarding performance are not based upon BranchGRPO-Mix ... Speaking of the results, if possible during rebuttal, I would advise comparing against one or two extra recent baselines (2024/5); as it stands most of the argument is built around outperforming DanceGRPO as the key baseline. Same goes with Section C.3 Table 4 (vBench) the performance gain (over DanceGRPO) is almost negligible. 

- On the qualitative results, a similar argument applies, in my view. I agree that more elaborate details (of BranchGRPO) might lead to higher preference or aesthetics scores, but in practice this might also come at the cost of divergence in prompt adherence. Fig. 4 right hand side is an example, I would argue Flux dinosaur and Gundam adhere much better with the prompt. To be more specific, I would argue the Gundam of BranchGRPO has little to no connection to Shrek anymore, and its Dinosaur does not look like a toy but more so an aesthetically elaborate 3D graphics. Extending on this, in Section C.2 (Appendix) top figure I would argue BranchGRPO does not look like an Anime anymore, same goes with DanceGRPO. In contrast, Flux seems to adhere to that part of prompt (at least) better. Please consider elaborating on this point. 

- I think pareto-front type of results can be way more insightful for your audience. A first step in that direction would be to consider adding prompt adherence using CLIPScore (text part --- please see https://arxiv.org/pdf/2104.08718) or something along those lines to your tables. Going beyond that, given that you propose an alignment or guidance approach, I would suggest considering paret-front reward vs. KL divergence plots, if possible. In one way or another you need to establish when your approach offers a better rewards (say HPS), in a faster fashion, it does not diverge drastically from the base model or does not sacrifice prompt adherence. I would suggest taking a look at the approach as well as the plots illustrated on this topic in the following two refs for inspiration: https://arxiv.org/pdf/2502.00968 & https://arxiv.org/pdf/2501.05803v2

### Questions
- Nowhere in the draft I can figure out which Diffusion model has been used? SDXL? 
- Possible to add video frames of DanceGRPO in Section C.3 as well for consistency?

### Soundness
2

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
4

### Summary
BranchGRPO restructures GRPO for diffusion models into a branching rollout to simultaneously remove inefficiency and unstable credit assignment caused by sparse terminal rewards. By branching at intermediate denoising steps while sharing prefixes, it preserves exploration diversity with far less redundant computation, and by propagating leaf rewards upward into dense, step-level advantages, it stabilizes optimization. Lightweight width/depth pruning further reduces backprop cost without affecting forward exploration. Experiments show faster convergence and higher or comparable alignment than DanceGRPO across image and video tasks, improving both efficiency and final quality.

### Strengths
The paper is clearly written and well-presented, making the motivation, problem formulation, and key ideas easy to follow. The proposed BranchGRPO achieves a favorable efficiency–performance trade-off: it reduces computation relative to the baseline DanceGRPO (as evidenced by lower NFE and shorter iteration time) while maintaining or even improving alignment metrics. Finally, the method is validated not only on image generation but also on video generation, demonstrating its effectiveness and generality across modalities.

### Weaknesses
1) Concern about generational diversity.  
Although Sec. 3.1 presents figures and KID/MMD distances (Inception/CLIP space) to claim that branching does not harm diversity, these metrics only measure distributional proximity, not conditional diversity nor modality-level mode coverage across prompts. Since branching may suppress global structure formation at high-noise regimes (where large-scale semantics are determined), the risk of mode collapse or reduced coverage cannot be ruled out. A direct and prompt-conditioned evaluation of diversity (rather than distributional closeness alone) would strengthen the claim. For evaluating diversity more directly, the evaluation methods in [1] may serve as a helpful reference.

2) Questions on evaluation validity.  
- For image experiments, no human evaluation is reported, making it unclear whether the improvements reflected by proxy rewards correlate with gold human preference. 
- For video experiments, the observed improvement might reflect reward hacking rather than a genuine multi-objective gain. Prior work has shown that video quality metrics often exhibit non-trivial trade-offs [2, 3], demonstrating that consistent improvement along multiple independent axes (e.g., visual quality, text alignment, dynamic degree) would be significant. 

3) Limited experimental coverage
- Despite Sec. 2 acknowledging the existence of many alternative alignment approaches, the experiments primarily compare against DanceGRPO and MixGRPO. It would be informative to include other competitive baselines. 
- The tested backbones are also limited; validating the method on more recent or diverse models such as Qwen-Image [4] or SD3.5 [5] would support claims of generality. 
- Finally, almost all rewards used are human-preference proxies. It remains unclear whether the method is applicable to other classes of rewards (e.g., object count or visual text rendering, as explored in [6]).

References 

[1] Kim et al., Test-time Alignment of Diffusion Models without Reward Over-optimization, ICLR 2025.  
[2] Liu et al., VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, CVPR 2025.  
[3] Oshima et al., Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search, NeurIPS 2025.  
[4] Wu et al., Qwen-Image Technical Report, arXiv:2508.02324.  
[5] Esser et al., Scaling rectified flow transformers for high-resolution image synthesis, ICML 2024.  
[6] Liu et al., Flow-GRPO: Training Flow Matching Models via Online RL, NeurIPS 2025.

### Questions
Please see the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
BranchGRPO introduces a new reinforcement learning alignment framework tailored for Diffusion and Flow-Matching models, addressing the instability and inefficiency of prior approaches like DanceGRPO. Its key innovation, “Structured Branching,” enables multiple branches of sampled trajectories to share common prefixes, allowing computation reuse and faster convergence. Combined with techniques such as Reward Fusion and lightweight Pruning, BranchGRPO achieves significant gains in training efficiency, optimization stability, and alignment quality, while maintaining scalability across larger branching factors and deeper training configurations.

### Strengths
* The paper is well-written and clearly structured, with a well-motivated approach that effectively addresses the inefficiency and instability challenges in existing diffusion model alignment methods.

* The structured branching design achieves approximately a 4.7× training speedup and smoother optimization through depth-wise normalization and reward fusion, leading to superior alignment stability.

* The proposed method surpasses DanceGRPO in both image and video generation quality (e.g., higher HPS-v2.1, ImageReward, and Unified Reward scores) and demonstrates strong scalability, maintaining predictable improvements under larger branch factors and depths.

### Weaknesses
* The current validation is primarily conducted on the FLUX.1-Dev model. To demonstrate the generality and robustness of the proposed framework, it would be valuable to include experiments on additional architectures (e.g., Qwen-Image or other text-to-image backbones).

* The evaluation primarily relies on HPS-v2.1, PickScore, and ImageReward. Incorporating more comprehensive and multidimensional benchmarks such as GenVal could provide deeper insights into alignment quality, compositional reasoning, and spatial understanding.

* The manuscript contains several typos (e.g., redundant brackets in Line 67). These should be carefully reviewed and corrected in the revised version.

### Questions
* In Section 3.1, the authors address the question “Does Branch rollout harm diversity?” and provide quantitative results. Specifically, they project the generated samples into both CLIP and Inception feature spaces and compare the distribution similarity between BranchGRPO and DanceGRPO using KID and FID metrics. While the authors report specific numerical values, it remains unclear whether these values are sufficiently small to substantiate that the two distributions are indeed close. Can the authors provide further justification supporting this claim?

* Building upon the first point, it seems intuitive that the diversity of generated samples may depend on the branching stage. In particular, if branching occurs earlier, the resulting samples might exhibit greater diversity, whereas later branching could limit variation. Have the authors conducted any analysis or observed patterns that validate this hypothesis?

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
This paper redesigns GRPO for diffusion models into a branching rollout (BranchGRPO) that amortizes computation via shared prefixes, converts sparse terminal rewards into dense stepwise signals through leaf-reward fusion with depth-wise advantages, and reduces only the backprop cost via width/depth pruning. In experiments on image and video alignment, it trains faster and more stably than prior work (e.g., DanceGRPO), improves metrics such as HPS v2.1, and substantially shortens iteration time while maintaining or improving diversity and temporal consistency.

### Strengths
- The paper is well organized and easy to follow: the problem setup, methodology, and objectives are presented coherently, with clear sectioning and notation that make the technical ideas accessible.
- By redesigning sequential rollouts into branching rollouts, it amortizes computation via shared prefixes while preserving search breadth, which has novelty.
- With selective backpropagation (width/depth pruning) that keeps forward generation and evaluation intact while reducing only the learning compute, it enables flexible speed–quality trade-offs, so it has usefulness.

### Weaknesses
The proposed method introduces multiple new degrees of design freedom, e.g., branching positions/counts, branching factor, branch-noise correlation, reward-fusion temperature, pruning choice, depth window and its sliding schedule, and the Hybrid ODE–SDE mixing ratio. Although the paper includes ablations of these components, it does not provide practical guidance for selecting them when tasks or compute budgets change. As a result, there is a concern about benchmark-specific overfitting and variability in reproducibility.

### Questions
Please address the above concerns with concrete clarifications.

### Soundness
3

### Presentation
3

### Contribution
3
