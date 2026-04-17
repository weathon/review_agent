# FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Flow-matching models deliver state-of-the-art fidelity in image and video generation, but the inherent sequential denoising process renders them slower. Existing acceleration methods like distillation, trajectory truncation, and consistency approaches are static, require retraining, and often fail to generalize across tasks. We propose FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost. This enables skipping computation at intermediary steps. We model the decision of how many steps to safely skip before requiring a full model computation as a multi-armed bandit problem. The bandit learns the optimal skips to balance speed with performance. FastFlow integrates seamlessly with existing pipelines and generalizes across image generation, video generation, and editing tasks. Experiments demonstrate a speedup of over $2.6\times$ while maintaining high-quality outputs. The source code for this work can be found at https://github.com/Div290/FastFlow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FastFlow, a plug-and-play adaptive inference framework that accelerates flow-matching generative models without retraining. The key idea is to approximate redundant denoising steps using a finite-difference Taylor expansion of the model’s velocity field, thereby skipping expensive neural evaluations when the dynamics are locally smooth. To decide how many steps can be safely skipped, FastFlow formulates the process as a multi-armed bandit (MAB) problem that adaptively balances efficiency and fidelity during sampling. Overall, FastFlow achieves acceleration while maintaining perceptual and semantic fidelity across multiple generative domains.

### Strengths
The paper is clearly written and easy to follow, with a well-motivated goal: accelerating flow-matching models to benefit the broader generative modeling ecosystem. I especially appreciate the inclusion of image editing, where real-time interaction is critical. While reusing the previous step’s velocity is not new, bandit-driven policy for adaptive step skipping is novel in this context to my knowledge and is presented in a concrete, convincing way.

### Weaknesses
The proposed method relies on heuristic parameters such as $p$ for velocity approximation and $\mu$ for the reward regularization term. It would be helpful to clarify how these parameters are chosen in practice and whether the method is robust to variations in their values.

Fig. 2–3 consistently show that FastFlow has higher latency than TeaCache (for comparable compute). Is this overhead coming from the multi-armed bandits? It seems that the overhead is not negligible. Can the authors clarify this behavior? Similarly, the result for FastFlow-10 in image generation (Table 1) shows only minor gains compared to Full-10. Additionally, the authors mention that the baselines follow the official hyperparameters. What are these parameters, and could this violate an apples-to-apples comparison? Overall, my concern is that the experiments either show marginal gains or may have presented in an unfair manner. 

Lastly, the paper would be strengthened by including an ablation study on the contribution of the multi-armed bandit algorithm. For instance, comparing FastFlow against simpler alternatives such as uniform or piecewise-constant skipping schedules.

### Questions
How does the method perform when coupled with quantization method? Also can this be utilized in flow models after reflow training? Ideally, reflow models have straight, non-crossing paths where the proposed method might not be as effective. I supposed experiment presented in Figure 4 using rectified flow models FLUX schnell can show a different trend.

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
3

### Summary
This paper introduces FastFlow, a plug-and-play adaptive inference framework to accelerate generative flow-matching models by skipping redundant denoising steps. The key insight is that flow-matching generative trajectories are often approximately linear, so intermediate states can be extrapolated cheaply instead of recomputing every step. FastFlow uses a finite-difference (Taylor series) approximation of the model’s velocity field to predict future states, allowing the method to advance multiple steps at zero neural network cost. Crucially, the framework employs a multi-armed bandit (MAB) at each timestep to adaptively decide how many steps to skip before the next full model evaluation. The bandit’s reward balances two objectives: (i) speed (skipping more steps) and (ii) accuracy (penalizing deviation from the true model trajectory). By learning this trade-off online per sample, FastFlow dynamically skips only those steps that would have minimal effect on final output. The approach is model-agnostic (no retraining or extra networks required) and integrates seamlessly into existing flow-matching pipelines.

The paper provides a theoretical bound on the error induced by skipping steps, formulates the skip decision as an online bandit problem, and demonstrates various experiments on text-to-image generation, image editing, and text-to-video generation. Empirically, FastFlow achieves over 2.6× speedup in inference while maintaining output quality comparable to the full model across these tasks. This represents a significant improvement over prior static acceleration methods, which often require retraining or sacrifice fidelity.

Though this method is effective when multiple steps are necessary, there are already many few-step or even one-step models (e.g., distillation or shortcut models) available today, so I am not sure whether this method is truly useful in practical scenarios.

### Strengths
Unlike static acceleration schemes, FastFlow adapts to each sample’s complexity. The multi-armed bandit dynamically decides per timestep how many steps to skip, meaning simpler cases automatically run faster while complex cases get more compute. This adaptive inference is novel and ensures no one-size-fits-all schedule, leading to greater robustness across diverse inputs.

The proposed framework is model-agnostic and plug-and-play, so it can be applied to existing pretrained flow-matching models without any retraining or fine-tuning. There’s no need for distilling a new model or training an auxiliary network, which makes the method very practical. It can be integrated into current pipelines with minimal effort, offering immediate speed benefits.

### Weaknesses
It requires some exploration to learn the optimal skipping policy. You acknowledge that the speedup may not fully materialize in the very first steps or first few samples due to this exploration phase. In practice, you mitigate this by seeding the bandit with one full generation, but if a user only generates a handful of samples, the adaptive policy might not have time to reach peak efficiency. In scenarios with very few inference runs, the benefit of FastFlow could be less pronounced.

The effectiveness of FastFlow rests on the assumption that the generative trajectories are locally smooth/linear enough to be extrapolated. While flow-matching models do encourage linear paths, there might be cases of highly non-linear or complex dynamics where the Taylor approximation could be less accurate. Therefore, essentially, FastFlow may be less effective if the model’s velocity field changes rapidly in unpredictable ways.

Table 1 and Table 2 are overlapped. Please adjust the margin via \vspace.

This paper contains some typos and grammatical issues. Here are the ones I found just by skimming through it:
* L67: a a theoretical -> a theoretical
* L95: We setup -> We set up
* L170: a static criteria -> a static criterion
* L364: is applied is as -> is applied as
* L388: an the -> and the
* L490: it’s content -> its content

### Questions
How many samples or iterations does it typically take for the bandit policy to stabilize? In your experiments, after seeding with one full generation, does FastFlow achieve near-optimal skipping immediately on the next sample, or does it require a few generations to fully adapt? Clarifying this can help understand use-cases. Any insight into how the bandit’s learning curve looks would be helpful.

Did you observe any failure cases or significantly reduced speedups for particular input types or prompts that might cause non-linear dynamics? Analyzing a case where FastFlow nearly defaults to the full model would illustrate its limits and robustness.

Theorem 3.1 gives an error bound $O(|S|/T^3)$. Did you empirically measure how close the practical error comes to this bound? In other words, is the bound reasonably tight or very conservative? Some intuition or experiment on how the final output error grows with number of skips in practice.

### Soundness
4

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
This paper proposes FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost.

### Strengths
- the motivation to accelerate flow-matching models is reasonable.
- the proposed method is plug-and-play and introduce negalectable extra costs

### Weaknesses
- missing comparisons on ImageNet 256

### Questions
can the method combined with modern fast samplers instead of Euler?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training free, plug and play acceleration scheme for flow matching Euler sampler at inference time using multi-arm bandit algorithms to choose the most relevant steps given a low-budget step sizes. More specifically, FastFlow aims to skip a variable number of intermediate time steps and approximate the missing velocities using a finite difference in time, and the choice of how many steps to skip is cast as a multiarmed bandit with reward. A theorem gives a bound on the terminal deviation between the approximated and full trajectories under smoothness assumptions, with uniform step size and a set $\mathcal{S}$ of skipped steps. Experiments on image generation, image editing, and video generation claim speedups up to about 2.6 times while maintaining GenEval and CLIP based IQA metrics near full sampling. Qualitative examples are shown for BAGEL and FLUX models and HunyuanVideo.

### Strengths
I think the most notable point is that the method is training-free and easy to integrate into existing flow matching pipelines. The speedup figures are plausible given the cost model of flow samplers where every velocity evaluation dominates wall time. I also like recasting the step-size selection as a bandit objective, whichh directly encodes the speed-accuracy tradeoff

### Weaknesses
- I think the novelty is thinner than the paper suggests. The velocity extrapolator collapses to a two step Adams Bashforth style predictor in uniform time-step. The work should at least acknowledge this equivalence and position itself relative to, for example,  PNDM [1], and other linear multistep sampling strategies already common in diffusion code bases. Empirically, a direct comparison to a simple two step predictor that still evaluates the model at checkpoints would be informative. Moreover, the statement that most alternative accelerators require retraining is not fully accurate. TeaCache and DeepCache are training free, and the recent adaptive skipping line is also training free in some variants. These should be acknowledged and compared.

- The theory is reassuring but optimistic in scale. For example, with $T=50$ and $∣S∣=25$, the error upper bound term $O(|S|/T^3)$ suggests very small terminal deviations unless the bounding constants are large. However, in the empirical evaluation, the experiments do show quality drop at aggressive skip levels, so either the constants are large or the bound does not capture the dominant error channel. A local error monitor beyond the velocity mismatch would be more principled, for example, an embedded predictor-corrector or curvature proxy, as in adaptive time-stepping literature.

- The experimental section omits several highly related baselines. AdaptiveDiffusion and AdaDiff are the most obvious, but there are also solver learning baselines such as Bespoke Solvers and S4S that reduce NFE without training the base generator. A comparison would help position FastFlow on the quality versus NFE Pareto.


[1] Luping Liu, Yi Ren, Zhijie Lin, Zhou Zhao (2022); Pseudo Numerical Methods for Diffusion Models on Manifolds, ICLR 2022.

[2] Neta Shaul, Juan Perez, Ricky T. Q. Chen, Ali Thabet, Albert Pumarola, Yaron Lipman (2023), Bespoke Solvers for Generative Flow Models, ICLR 2024.

### Questions
- Please quantify compute precisely. Report average number of model calls per sample and the distribution of skip lengths $\alpha_t$.
- Please compare against AdaptiveDiffusion and AdaDiff under the same backbones and prompts, and include DeepCache on image tasks and TeaCache on both image and video. Use the same target speed levels and report NFE matched comparisons.
- See also other remarks in Weaknesses on the theoretical bound.

### Soundness
3

### Presentation
2

### Contribution
2
