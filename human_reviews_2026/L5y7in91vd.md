# TEST-TIME SCALING IN DIFFUSION LLMS VIA HIDDEN SEMI-AUTOREGRESSIVE EXPERTS

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 6

## Abstract
Diffusion-based large language models (dLLMs) are trained to model extreme flexibility/dependence in the data-distribution; however, how to best utilize this at inference time remains an open problem. In this work, we uncover an interesting property of these models: dLLMs {trained on textual data} implicitly learn a mixture of semi-autoregressive experts, where different generation orders reveal different specialized behaviors. We show that committing to any single, fixed inference time schedule, a common practice, collapses performance by failing to leverage this latent ensemble. To address this, we introduce HEX (Hidden semi-autoregressive EXperts for test-time scaling), a training-free inference method that ensembles across heterogeneous block schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such as GSM8K, it boosts accuracy by up to 3.56× (from 24.72\% to 88.10\%), outperforming top-K margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX even yields significant gains on MATH benchmark from 16.40\% to 40.00\%, scientific reasoning on ARC-C from 54.18\% to 87.80\%, and TruthfulQA from 28.36\% to 57.46\%. Our results establish test-time scaling as a powerful principle for dLLMs, showing that the sequence in which masking is done can play a significant role in test-time scaling/inferencing of dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes HEX (Hidden semi-autoregressive EXperts), a training-free inference method for diffusion-based large language models (dLLMs). Through a series of intuitive experimental observations, it reveals that dLLMs implicitly learn a mixture of semi-autoregressive experts, where different masking schedules activate distinct expert behaviors. By ensembling multiple decoding paths with majority voting, HEX achieves remarkable performance on reasoning benchmarks such as GSM8K and MATH.

### Strengths
1. Introducing the proposed method through a simple toy example makes the paper intuitive and easy to follow.

2. The experimental results are impressive, showing substantial improvements across multiple reasoning benchmarks.

### Weaknesses
1. The authors repeatedly emphasize (for example, in Sec. 1 and Sec. 3) that using sampling strategies such as top-K margin can harm performance and that this issue has not been sufficiently discussed. However, previous literature has already provided explanations for this phenomenon, such as Appendix B.4 of [1].

2. The presentation is poor. For example, Eq. (1) is incorrect because it lacks the necessary reweighted coefficient. The description of the main method—such as how the majority vote is performed—is neither clearly introduced in the text nor reflected in the algorithmic formulas.

3. The evaluation is neither fair nor comprehensive. For instance, in the abstract the authors state that LLaDA-8B-Instruct achieves 16.4 % and 54.2 % accuracy on MATH and ARC-C, which HEX improves to 40.0 % and 87.8 %. However, the original LLaDA paper reports 42.2 % and 88.5 % for these datasets. The authors should not use the widely known underperforming random remasking as the baseline. In addition, the paper seems to lack a comparison of inference efficiency.

[1] Nie et al. Large Language Diffusion Models.

### Questions
What are the differences and connections between the proposed method and beam search?

### Soundness
2

### Presentation
1

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
The paper argues that dLLMs trained with masked-token objectives implicitly encode a mixture of semi-autoregressive experts: different left-to-right block sizes activate different conditional distributions learned during training. Committing to a single decoding schedule can collapse performance. The proposed training-free method HEX ensembles across heterogeneous semi-AR block schedules and aggregates final answers via majority vote. On LLaDA-8B-Instruct, HEX reports strong gains while avoiding collapse modes.

### Strengths
1. The paper shows dLLMs are sensitive to semi-autoregressive schedules, which provides confirmed motivation for HEX to ensemble diverse schedules for robust inference.

2. HEX is a training-free plug-and-play method, enabling a trade-off between accuracy and computation via the number of voting trajectories, and is competitive with (or exceeds) RL-fine-tuned variants under the reported settings.

### Weaknesses
1. Although the paper treats the number of voting trajectories as a tunable inference time knob, the marginal gains from adding schedules are modest relative to the near linear growth in compute (for example, GSM8K improves only from 81.96% to 84.15%). Moreover, while HEX can surpass RL fine tuned baselines, the cost profiles differ: RL incurs a one time training expense and then serves at single pass inference cost, whereas HEX must sustain a $k$ times inference overhead on every request to remain competitive, which is an unfavorable trade at deployment scale.

2. While HEX’s wall-clock latency is reported for GSM8K, the paper does not provide budget-matched comparisons across all datasets, which limits a fair cost-effectiveness assessment.

3. Despite a persuasive mixture-of-experts interpretation, the method remains theoretically under-specified: there are no conditions or guarantees for the consistency/convergence of majority voting, sample-complexity requirements, or optimal schedule priors, weakening the theoretical closure of the contribution.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a specific choice of block schedule for sampling from diffusion language models which is “semi-autoregressive”, meaning that parallel sampling is carried out in blocks that are autoregressively revealed left to right. The authors associate different choices of block schedule with different “expert” components of the dLLM and reveal large gains in accuracy by taking the majority vote over predictions by such experts (in effect marginalizing over different choices of block schedules).

### Strengths
- Massive performance gains
- Interesting interpretation of block schedules as experts

### Weaknesses
- Identification of block schedules as experts is tenuous - do these block schedules really encode specific knowledge/expertise in the way that, say, the experts in an MoE model do? Can you provide more evidence of this?
- If semi-autoregressive decoding uses all block schedules down to very small granularity (b=1), can performance gains be explained in terms of increased granularity (and higher computational cost)? The b=1 case in particular seems to perfectly mimic standard auto-regressive sampling in transformer-based LLMs.

### Questions
- Can you clarify the impact of your proposed methodology on computational efficiency? Please compare both to regular sampling from diffusion LLMs and to autoregressive sampling from transformer-based LLMs.

See the section on weaknesses for additional questions.

### Soundness
2

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
2

### Summary
This paper shows that by ensembling over multiple semi-autoregressive decoding orders, the current dLLMs can get a significant improvement in capability. They empirically show that confidence-based decoding fails to beat the random order baseline. With autoregressive sampling, the performance jumps more than 3x on benchmarks including GSM8k. Ensembling over block-size and randomness further improve the performance.

### Strengths
1. The paper clearly illustrates that previous methods such as confidence-based sampling fail in the context of long reasoning. This provides a clear motivation for designing new algorithms.

2. The method proposed is very intuitive and easy to apply.

3. The potential mechanism behind the empirical improvement is clearly outlined in Sections 3 and 4.

### Weaknesses
1. It should be noted that in the abstract, the 24.72% performance is not using semi-autoregressive baseline, which is widely adopted.

2. In section 5.4, regarding the importance of block-diversity, the author shows that ensembling 30 different block schedules can improve performance but is worse than 'structural diversity' with a fixed set of 5 block schedules and 6 random seeds. However, it remains unclear whether the proposed algorithm can beat the plain method with only 1 block schedule and 30 seeds.

### Questions
1. Please refer to weakness 2. 

2. Have authors think about how to incorporate HEX into the post-training of dllm, in order to get either faster decoding or better performance?

### Soundness
3

### Presentation
3

### Contribution
3
