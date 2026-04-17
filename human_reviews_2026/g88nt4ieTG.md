# Diffusion Language Model Knows the Answer Before It Decodes

- Decision: Accept (Oral)
- Scores: 8, 4, 6, 8

## Abstract
Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high-quality outputs.
In this work, we highlight and leverage an overlooked property of DLMs—**early answer convergence**: in many cases, the correct answer can be internally identified by half steps before the final decoding step, under both semi-autoregressive and random remasking schedules. 
For example, on GSM8K and MMLU, up to 97\% and 99\% of instances, respectively, can be decoded correctly using only half of the refinement steps.
Building on this observation, we introduce **Prophet**, a training-free fast decoding paradigm that enables **early commit decoding**. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e. decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training.
Empirical evaluations on LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4$\times$ while preserving high generation quality, and yields additional speedups when combined with existing acceleration methods.
These results recast DLM decoding as a problem of *when to stop sampling*, and demonstrate that early answer convergence provides a simple yet powerful mechanism for accelerating DLMs on reasoning, code, and planning tasks with identifiable answer regions. Our code is available at \url{https://github.com/pixeli99/Prophet}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a new decoding algorithm, Prophet, for diffusion language  models that enables early stopping of refinement using a dynamic schedule based on token logits. The paper provides empirical observation that motivates the methodology: In a lot of instances the correct answer is already decoded halfway through the decoding steps, and the remaining steps are only additional refinements on the reasoning. The paper evaluates their decoding algorithm and shows that early stopping provides speedup without significant drop in accuracy.

### Strengths
- The early stopping idea is simple but effective. The presented decoding algorithm clearly presents performance benefits and is relatively simple to implement and deploy. There does not seem to be any downsides to default to using this algorithm.
- The algorithm works for different generation lengths, re-masking strategies, and even improves course-grained block refinements
- The paper is well-written; the empirical observation and resulting algorithm are described clearly.

### Weaknesses
- While the presented experimental evaluation in section 5 is fair and sound, the work would benefit from more ablations to fully explore the limits (or capability) of the methodology. See Questions for what experiments should be added.
- The paper is well written in general, but Section 5 seems to have some typos that makes the section a bit hard to read. Namely, the text refers to "three decoding strategies" but only show full baseline and prophet. Section 5.2 also refer to a "half baseline" that is undefined and not shown in Table 1.
- The table column headers should be made clearer. In table 1, it was not immediately clear that columns 2,3,5,6 were showing accuracy percentages and the orange is the speedup. The column header should clearly label "accuracy" and have the Prophet speedup as its own column with label "Speedup". In table 2, It was also confusing what 16, 32, 64, 128 refer to at first. There should be a header label that says "step budget"

### Questions
- To clarify, is the experimental evaluation in section 5 using suffix prompting? If not, what does the performance looks like for Prophet vs baseline? It seems to me from the early convergence analysis that suffix prompting and Prophet should synergize?
- What are the percentages of the step budget used by Prophet in the benchmarks? Does the distribution of used steps correspond to the early convergence analysis?
- Why use a staged threshold function? Why not make $\tau(p)$ a continuous function? What does the performance look like if it is simply a linear function?

- The early answer convergence analysis only shows the histograms for correct answer. What does the distribution look like for incorrect answer? For all answers? I am curious to see if the distribution for incorrect answers skews right (implying the model is simply uncertain of the answer) or skews left (implying the model is overconfidently wrong).
- In Figure 2, how come there are some orange marks to the right of the blue marks? Is it that the token is changed and then rechanged back?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address a key challenge in Diffusion Language Models (DLMs): although DLMs inherently offer the potential for parallel decoding, their inference speed in practice often lags behind that of autoregressive (AR) models.
The core contribution of the paper lies in identifying and leveraging an overlooked property, Early Answer Convergence. Through extensive experiments, the authors demonstrate that in many cases, DLMs internally determine the correct answer well before the decoding process concludes, often around halfway through the total number of steps.
Building on this insight, the authors propose Prophet, a training-free fast decoding paradigm. Prophet introduces an Early Commit Decoding strategy that dynamically monitors the model’s confidence in its predictions at each decoding step. Specifically, it measures the Confidence Gap between the top-1 and top-2 predicted candidates and employs an adaptive threshold that evolves with decoding progress to decide when to go all-in.
Experimental results show that Prophet achieves substantial decoding acceleration—up to 3.4× speedup—while maintaining, and in some tasks even slightly improving, the generation quality.

### Strengths
1. The observation of early answer convergence is novel and interesting.
2. The proposed method is lightweight and requires no additional training.
3. It demonstrates a significant acceleration in inference.

### Weaknesses
1. The paper appears to heavily rely on the suffix prompt “Answer:”, yet the implementation details are not provided. It is unclear whether this suffix is appended to the end of the question or to the generated sequence.
2. I am also puzzled by why adding the suffix prompt “Answer:” leads to such a significant difference in Figures 1 and 2. The authors should provide a deeper explanation or interpretation of this phenomenon.
3. The paper does not seem to explain how the answer region positions are determined. Moreover, for general question-answering tasks beyond math problems, it is possible that the entire generated sequence could be considered as answer region positions.
4. Although the “go all-in” strategy appears to maintain accuracy on the “answer,” it is unclear whether it might compromise aspects such as the correctness of intermediate reasoning steps, grammatical accuracy, or overall fluency.
5. The paper should provide some complete, real cases to help evaluate and analyze the decoding behavior and dynamics of the proposed method.
6. The notion of “L” appears to be used inconsistently.
7. This appears to be a rapidly evolving field. In the past few months, several works have emerged on accelerating DLM inference by reducing sampling steps (e.g., arXiv:2505.22618, arXiv:2506.10848, arXiv:2507.18578). Some of these methods seem to achieve even more pronounced acceleration effects compared to this paper. A more detailed comparison and discussion with these contemporaneous approaches would be very valuable. (I am not certain about ICLR’s policy regarding “concurrent works.” If such comparisons are deemed unnecessary under that policy, the authors may clarify and omit this comment accordingly.)

### Questions
Included in Weakness.

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
2

### Summary
This paper investigates sequence generation with discrete diffusion models and makes an interesting observation: in many cases, it is possible to decode the correct answer early in the denoising process such as 25%-50% of the total steps. Motivated by this observation, the authors propose a method to monitor answer convergence and propose to track the difference between the top1 and top2 logits of the answer tokens. If that difference exceeds a threshold (dependent on the step t) then the complete answer is decoded in a single step. The authors evaluate this protocol on a variety of benchmarks and find that performance degradation is minimal, while at the same time speedups are very high.

### Strengths
1. The idea is very well-motivated with an initial exploration on GSM8K and MMLU, which makes it easy for the reader to follow the thought-process of the authors and also appreciate the idea of answer convergence. 
2. The proposed algorithm is really simple, does not need any training and can be used for any discrete diffusion model. The speedups are quite strong and the evaluation suite is quite broad, so this technique does seem to work on most benchmarks of interest.

### Weaknesses
1. The formatting and writing of the paper can be improved, some things seem imprecise in the main text, e.g. how is $\hat{g}_t$ exactly defined? Section 4 seems to suggest that it is an average over all positions, but Algorithm 1 seems to suggest you only average over the answer tokens? The latter seems to make more sense to me but it would be great if you could clarify this in the main text. In general, the nature of answer tokens is not super clear to me; for GSM8K the definition seems pretty straightforward but for problems like HumanEval, where the answer comprises a piece of code, the definition is a bit less clear. Things like “Answer length” and “Block length” should also be defined better so that the reader can more easily follow.
2. There seem to be quite some hyper parameters for the method, i.e. a schedule needs to be set which right now is somewhat arbitrarily divided into three uniform sections, and then each section needs a corresponding threshold. How exactly did you determine these values? You mention some validation experiments but I couldn’t find much more detail in the paper. Does setting those parameters on one benchmark transfer to another, or does the validation data need to be diverse enough?

### Questions
1. I was a bit surprised that you observed even some gains on some benchmarks when applying your early-decoding strategy. Do you have any intuition regarding why this happens? Showing such an example where correct tokens are later changed would be interesting.
2. I could imagine that looking at top1 and top2 logits per position can also bring some issues? E.g. in coding, the model might be unsure whether to call a given variable “x” or “y”, keeping the difference quite small although both immediate decodings might be fine. Do you observe any such behavior, where even earlier decoding would have worked but “Prophet” was kind of stalled due to such ambiguous tokens in the answer?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a novel method to accelerate sampling in diffusion language models. In particular, they focus on the observation that answers often converge in intermediate steps. To this end, they introduce a method that enables up to 3.4x reduction of the sampling steps training-free. The efficacy of the method is demonstrated in two models and multiple benchmarks.

### Strengths
- The main idea is novel and clearly presented.

- The studied problem is relevant and the proposed solution appropriately motivated.

- The empirical evidence regarding convergence in intermediate steps is valuable for the community.

- The experimental results showcase that the proposed methods can achieve significant speed-ups with similar performance.

- I appreciate the thorough ablations.

### Weaknesses
- The only major weakness I see is the lack of comparison with other methods that aim to reduce the number of steps, e.g., distillation-based. This would help put the latency-reduction in perspective.

- Could the decision to terminate a decoding loop be learnable, for example by a judge network, e.g., similar to [1]? A discussion on this would be interesting and potentially fruitful for future directions.

- I am missing some related works that aim to reduce the number of sampling steps, e.g., [2].

[1]. Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment

[2]. Beyond Autoregression: Fast LLMs via Self-Distillation Through Time

### Questions
I would appreciate it if the authors could address the issues and questions raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
