# DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
We propose **DiFFPO**, **Di**ffusion **F**ast and **F**urious **P**olicy **O**ptimization, a unified framework for training masked diffusion large language models (dLLMs) to reason not only *better (furious)*, but also *faster* via reinforcement learning (RL). We first generalize the existing baseline approach such as d1 by proposing to train surrogate policies via off-policy RL, whose likelihood is much more tractable as an approximation to the true dLLM policy. This naturally motivates a more accurate two-stage likelihood approximation combined with importance sampling correction, which leads to generalized RL algorithms with better sample efficiency and superior task performance. Second, we propose a new direction of training efficient samplers/controllers of dLLMs policy. Via RL, we incentivize dLLMs' natural multi-token prediction capabilities by letting the model learn to adaptively allocate an inference threshold for each prompt, which yields better accuracies with lower number of function evaluations (NFEs) compared to the base model. Finally, we consider joint training of the dLLM policy and the sampler together to obtain the best performance in improving the Pareto frontier of the inference-time compute of dLLMs. We showcase the effectiveness of our pipeline by training open source large diffusion language models over math and planning benchmark tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose DiFFPO, an RL framework for diffusion LLMs that replaces the intractable dLLM likelihood with a better surrogate via a two-times mean-field approximation and an importance-sampling correction. The method trains an efficient sampler by learning a prompt-aware entropy threshold γ(c), with optional joint training of model and sampler. The results show DiFFPO improving sample efficiency and accuracy on math and planning tasks while reducing NFEs versus d1 and fixed samplers.

### Strengths
1. The proposal of the surrogate-likelihood route for diffusion LLMs, the importance-sampling correction, and the prompt-aware sampler thresholds are all nice additions. 

2. The paper figure and tables show  show consistent accuracy gains and better sample efficiency than d1.

3. The framing and equations are easy to follow.

### Weaknesses
1. The scope of the results (tested baselines + benchmarks) is quite small.
2. The method is not compared to other GRPO-like baselines (even though these are mentioned in the prior work section.
3. There are no experiments testing the method's stability & gains (the role of clip sweeps, ess/weight histograms, and a no-clip baseline.
4. Minor: The quality of the figures is low (e.g. fig 3).

### Questions
1. Can the authors explain the lack of benchmarks against comparative methods, and why comparing to d1 would suffice?

2. It would be worthwhile to see a light sweep of ε (and any hard cap C), report ESS/weight histograms per timestep, and include a no-clip (or self-normalized IS) reference, or have an intuition for what to expect.

3. Under what conditions do you expect DiFFPO to degrade? For example, 1. highly multimodal posteriors where the TTMF surrogate (single-τ conditioning) misses key latents, or 2. sparse/peaky rewards where clipping trims the few high-weight successes?

### Soundness
3

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
4

### Summary
This work tackles the problem of RL fine-tuning for diffusion LLMs by proposing a modification of the GRPO objective used in prior work (namely d1 (Zhao et al., 2025)). The objective uses a more accurate token-level likelihood approximation (by evaluating the conditional likelihood $\pi_\theta(\hat{o}^t_i | c, z^{s}_i)$ at some intermediate partial completion $z^s_i$, for $t<s$ rather than just on the prompt as in d1), as well as an importance sampling correction to weight the GRPO advantage.  Additionally, a method is proposed to improve inference speed, whereby an input dependent threshold is learned  for use in Entropy Bounded sampling. The proposed policy gradient method is used to fine-tune both the model, and the sampling threshold predictor, for tasks such as Sudoku, Math, GSM8K and Countdown. An improvement in both task performance and inference efficiency (measured by NFEs) is demonstrated.

### Strengths
1. The experimental results reported are clear and convincing, both for the increase in reward, and the speed of inference (when tuning the Entropy Bounded threshold)
2. The analysis of the error in d1’s likelihood approximation is convincing (Figure 2 a, b)).
3. The idea of training the inference threshold in a sampler with RL is useful

### Weaknesses
1. Numerous spelling and grammatical mistakes throughout. The paper would benefit from more thorough proofreading.
    - Related to presentation Figure 2 c) is unclear, and includes symbols not defined
    - Figure 4 should indicate the threshold’s used when tracing out the frontier
2. For the efficiency comparison (eg. Figure 3), some plots should also be included where the x-axis is the number of calls to the denoiser, since the proposed two-times mean field approximation requires twice as many denoiser calls per gradient update (compared to d1’s mean-field approximation) when updating on a fixed set of trajectories. Without this, its difficult to judge whether the two times mean field approximation adds much benefit over the d1 baseline.
3. Several experimental details missing from the writeup - for instance there are no details provided for  batch sizes used, or whether remasking is performed (as described in the d1 paper) - these can be critical in evaluating whether the comparisons are fair 
4. The impact of the importance weighting in eq 12 is unclear: we can sample trajectories from the surrogate (two-times mean field) policy and directly optimize it without any importance weights. How does this approach compare to using the importance weights? 
    - Analogously, the importance weights can be directly applied to the d1 method. Comparing to this setup would help clarify the benefits of the two-times mean field approximation. 
5. In terms of theory - Assumption 2 requires some sort of justification or argument (possibly empirical evidence). Theorem 3 is a straightforward corollary of Assumption 2, but it shifts the need for justification to Assumption 2.

For now I am recommending a weak reject, due to points 2 and 3 above.

### Questions
1. It would be useful to see a comparison against full likelihood evaluation to see the impact of the proposed corrections (eg. importance sampling), in a setting where it is feasible
2. Between table 2 and 3, why is the performance of the base model and d1 different?
3. Related to weakness 2 above:, one can extend the approach to an m-times mean field approximation. Is there some justification or evidence for why stopping at 2 is reasonable or adequate?

### Soundness
2

### Presentation
2

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
The paper presents a novel framework for fine-tuning diffusion LLMs, called DiFFPO. By utilising surrogate policies, a two stage likelihood approximation is adopted, which leads to improved sampling efficiency and task performance across a range of benchmarks.

### Strengths
The paper displays a high degree of novelty, and addresses the highly impactful and timely topic of diffusion LLMs.

The proposed method is theoretically well motivated, and this is clearly presented within the manuscript.

Experimental performance of the proposed model appears strong, showing substantial improvements in both speed and accuracy.

### Weaknesses
The main experimental results, while very promising and interesting,  are only presented with the best performing seed. This is not ideal as we then lack statistical uncertainty, and robustness. For example, the reader cannot be confident whether the improvement on MATH500 in Table 1 from 34.20 to 36.40 would persist if we had chosen a different set of random seeds.  RL can lead to a high variance across seeds and it is therefore especially important to fully present these findings. 

Ideally it would have been informative to explore variations to the model architecture, or to  assess how this impact is expected to scale with the size of the model. Some comparisons to an 8b autoregressive LLM may also be informative.

### Questions
What metric was used to determine the best performing seed, is this on test performance or some validation metric?

What is the variance in empirical performance across runs with different seeds, and does this change significantly between the different methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the problem reinforcement learning with masked diffusion language models. The authors proposes DiFFPO, an off-policy RL approach that (i) optimizes a surrogate policy whose likelihood is tractable; (ii) improves that approximation with a two-times mean-field conditioning on an intermediate latent and adds importance-sampling (IS) correction; and (iii) trains the sampler itself by learning a prompt-aware entropy-bounded threshold and optionally jointly training policy + sampler. DiffPO is evaluated empirically by fine-tuning LLaDA-8B-Instruct on math and planning (GSM8K, MATH500, Sudoku, Countdown). The authors compare DiffPO with d1/diffu-GRPO baseline: with Top-k sampling, DiFFPO improves accuracy over base by +30.5 (Countdown) and +35.8 (Sudoku) points. With EB sampling and joint sampler training, DiffPO shifts the speed–accuracy frontier.

### Strengths
* The paper is generally well written. The problem is defined quite clearly and the proposed solution explained clearly.
* The 2MF approximation intuitively makes sense, and Theorem 3 shows it to be a tighter likelihood approximation than prompt-only (though I have some concerns about Assumption 2). The approximation is also handled in the objective with a IS correction. 
* I like the idea of jointly training some sampler parameters instead of being fixed post-hoc to enable efficient sampling.

### Weaknesses
* I have a major concern about the empirical results. Specifically, the authors mention they select the best checkpoint from seeds for comparions. This can skew comparisons, putting the results in question. I suggest reporting the mean performance with some error estimates.
* Another concern is that DiffPO results are using top-k sampling and it seems that the baselines are not using it, this also makes the comparisons invalid.
* The paper misses some closely related prior work on RL fine-tuning of diffusion language models [1, 2, 3]. I believe a comparison to these baselines would be critical, as it is quite an active area. 
* The experiments are also limited to a single base model. I think it would benefit the paper to have some experiments with another base model (e.g. https://github.com/DreamLM/Dream)

[1] Venkatraman et al., 2024. Amortizing intractable inference in diffusion models for vision, language, and control.

[2] Zekri and Boullé, 2025. Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods.

[3] Tang et al. 2025. wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models.

### Questions
* Could you clarify the exact compute cost for the method and how it comapres to the baselines?
* Could you clarify the choice of the experimental setup, using the best seed and checkpoint?

### Soundness
2

### Presentation
2

### Contribution
2
