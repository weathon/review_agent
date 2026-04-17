# e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Test-time scaling offers a promising path to improve LLM reasoning by utilizing more compute at inference time; however, the true promise of this paradigm lies in extrapolation (i.e., improvement in performance on hard problems as LLMs keep "thinking" for longer, beyond the maximum token budget they were trained on). Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration: training the LLM to effectively spend its test time budget by chaining operations (such as generation, verification, refinement, etc.), or testing multiple hypotheses before it commits to an answer. To enable in-context exploration, we identify three key ingredients as part of our recipe e3: (1) chaining skills that the base LLM has asymmetric competence in, e.g., chaining verification (easy) with generation (hard), as a way to implement in-context search; (2) leveraging "negative" gradients from incorrect traces to amplify exploration during RL, resulting in longer search traces that chains additional asymmetries; and (3) coupling task difficulty with training token budget during training via a specifically-designed curriculum to structure in-context exploration. Our recipe e3 produces the best known 1.7B model according to AIME'25 and HMMT'25 scores, and extrapolates to 2x the training token budget. Our e3-1.7B model not only attains high pass@1 scores, but also improves pass@k over the base model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies different aspects of test-time scaling driven by Reinforcement Learning. In particular, the authors study the role of chaining asymmetric capabilities (e.g., generation and verification), negative gradients and training length budget in improving RL performance in-distribution, out-of-distribution and extrapolating to longer test budgets. The authors conclude that exploration in RL is enabled by asymmetric capabilities chaining and that negative gradients are important for driving exploration and extrapolation beyond the training budget. Following this, the authors introduce a new method for improving RL performance, by using a curriculum that increases the training length budget together with the data complexity. This results in a model that achieves SOTA performance under 2B.

### Strengths
- Overall, the paper is well-written, and the introduction section did a good job of motivating the problem setting.
- The paper studies a fundamental problem in scaling test-time compute using RL and provides novel insights on the role of different aspects of the training pipeline.
- The discussion of the role of asymmetric capabilities is interesting and novel, and the authors provide experimental evidence to the important of this chaining.
- I found the analysis of the effect of varying the training length budget on the performance of RL extrapolation to be very interesting, particularly the fact that smaller training length can sometimes perform better when extrapolating to longer sequence lengths compared to training on a larger length. The "trade-off" between training length and problem complexity is also interesting.
- The curriculum method introduced in the paper seems to be convincingly better than alternatives in the setting it was tested.

### Weaknesses
- One main issue that I have with the results and analysis (especially in Section 4 and Section 5) is the fact that in many cases it seems very hard to attribute the improvement/degradation in performance to the properties identified by the authors. Examples for this:
  - Section 5 discusses the role of the negative gradients in RL, and concludes that they are important for driving exploration, increase in length and extrapolation. However, the authors test this by "masking" negative gradients from the RL algorithm, which results in an overall degradation in performance. The "masking" clearly makes the algorithm worse, and then a lot of metrics become worse, but it is hard to tie the negative gradients to particular aspects of this, like the authors try to do. For example, we can make the algorithm worse by e.g. choosing wrong hyper-parameters, under-training or over-fitting. Would this not result in a similar behavior, even though negative gradients are used? An experiment like this, i.e. comparing "masked" negative gradients to other modifications to the algorithm that result in the same overall accuracy would help to shed light on this.
  - Section 4 suffers from similar issues, though I think this section is more convincing. Here the authors claim that generation-verification gap drives performance improvement in RL, but this is mostly shown by measuring the number of "chaining" occurrences and showing that it increases with RL steps. But it is known that RL increases length, and that this is correlated with improved performance, so it seems that this just finds another property (number of chained asymmetries) that is correlated with increased length. 
- Another main issue is the apparent disconnect between the different parts of the paper. The role of negative gradient seems unrelated to the "chaining asymmetries", and these two seem unrelated to the new method introduced in the paper. While the authors claim that the scientific analysis drives the resulting method, it's not clear how this method relates to the analysis.
More minor issues:
- The authors discuss the asymmetric capabilities chaining in general, but in practice the paper is focused on generation-verification gap. Not clear what this more general characterization is adding.
- The discussion on the role of negative gradients in relation to the position of the EOS token is unclear. In particular, while negative gradients reduce the probability of the EOS token appearing at a given place, they could also increase its probability for appearing earlier. In general, I think that a more theoretical discussion of the role of negative gradients can be introduced into the main paper (now only in the appendix). This could perhaps clarify things.
- The method introduced in the paper is relatively simple: training in two phases, starting with a simple problems and short sequence length followed by harder problems and longer sequence length. I think that the method is introduced makes this seem more complex than it actually is. While I don't see any problem with a simple method that improves performance, introducing it in a convoluted way seems to suggest that the method is more sophisticated or general than it actually is.
- The plots in Figure 8 are very hard to parse, with some of the labels overlapping.
- Line 333: "Fig. 5(a) confirms this...." - I think this is not referring to the right figure.
- The description of the n-digit multiplication experiment in Section 4 is not clear.
- Lines 321: "the policy gradient update reinforces chaining and improves in-context exploration, and this is reinforced" - unclear what this is saying.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors state that the model's reasoning capability does not extrapolate well beyond the maximum budget observed during training. The paper explains why the output increases during RL training, with asymmetries. They argue that constraining the budget would encourage the model to become more precise rather than interleaving "skills". They suggest recipe "e3", which dynamically adjusts the training budget according to the difficulty of the prompts.

### Strengths
- The concept of "chain of asymmetric skills" offers a compelling explanation for the reasoning trajectory extension during RL. Likewise, the "verification-generation gap" (VG Gap) clearly accounts for the reasoning path that appears in the pattern of solution-verification-solution-....
- The authors trained a state-of-the-art  (<2.0B) model on AIME/HMMT 2025 using the suggested method e3.
- The paper presents a comprehensive experiments that strongly support the authors' claims.

### Weaknesses
- The proposed method "e3" appears to lack novelty. The paper does not suggest a clear approach to determine a proper value of $\kappa$, nor does it include additional experiments on the impact of different $\kappa$ values. Also, the budgets it explores (4k, 8k, 16k) are too coarse, leading to suboptimal performance.
- The definition of problem hardness is somewhat ambiguous. For example, the authors define the difficulty level for each dataset separately.
- The scalability of e3 remains uncertain. If the asymmetry (VG gap) appears only in the small-scale model (I believe so), the broader implications of the paper would be significantly weakened.

### Questions
- Would enforcing asymmetry lead to improved final performance? For instance, one could explicitly teach the base model on verification tasks before training how to solve the problems.
- Can e3 be extended to explore more fine-grained training budgets? Would the final performance benefit from that?

### Soundness
3

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
The paper argues that realizing the promise of test‑time scaling requires learning in‑context exploration, and proposes e3 which is a training recipe that teaches models to spend extra tokens on structured generate‑verify‑revise behavior so performance extrapolates beyond the training budget. e3 leverages chained asymmetries in the base model (verification–generation gap), preserves negative RL gradients so failures shift probability mass from premature endings into longer, verification‑heavy traces, and uses a coupled curriculum that pairs data difficulty with token budgets to reward chaining within budget and beyond. On Qwen3‑1.7B trained up to 16k tokens, the resulting e3‑1.7B extrapolates to 32k and achieves state‑of‑the‑art results among sub‑2B models on AIME’25 and HMMT’25, with higher pass@k; good analyses and ablations link negative gradients to longer responses, higher entropy, and more verification.

### Strengths
- The paper offers a curriculum based training recipe chaining asymmetries provides remedy to the problem that many open models fail to extrapolate beyond their training budget. The empirical results are strong on the math benchmarks and the choice of Countdown and Multiplication tasks are illustrative in depicting the VG gap.
- The ablations and mechanistic analyses are well conducted and support the presented claims, and the theory section also provides good insights. Overall, I believe the paper is well-written with clear contributions and gains.

### Weaknesses
- The paper does not directly measure/report how verification capability or the VG gap evolve during training, which I believe is an important direction to investigate to better understand the verification‑generation training dynamics.
- As an ablation study, the paper demonstrates masking negative gradients in GRPO Mask baseline. Yet, masking negatives also changes the update magnitude and how GRPO’s advantage calculation & clipping behave. So, a more controlled study could have provided better evidence, as the observed gap could stem from these rather than the gradient sign itself.

### Questions
I don't have any more questions, please see weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that most open-source “reasoning” LLMs don’t actually improve when you give them much more test-time compute beyond what they were trained on, and proposes a training recipe, e3, to fix that by teaching models to explore in-context. e3 has three parts: (1) exploit asymmetric skills in the base model (especially a verification-better-than-generation gap) so the model can chain “generate → verify → revise”; (2) keep the negative gradient during RL so failed traces push probability mass toward longer traces that chain more of those asymmetries; and (3) use a coupled curriculum that jointly schedules problem difficulty and token budget so exploration is rewarded and optimization stays stable.

### Strengths
1. The paper tackles an important gap: most open-source “reasoning” LLMs fail to extrapolate to longer test-time compute than they were trained on.  

2. It analyzes the verification-generation (VG) asymmetry and argues it is a key driver of RL gains beyond the training length by enabling chains of “generate → verify → revise.”  

3. The coupled curriculum is clearly specified (including a concrete budget-selection rule) and empirically outperforms many SOTA models.

### Weaknesses
1. In Fig. 8(a), Qwen3-1.7B with naïve RL appears to extrapolate similarly to e3 at some budgets; the large headline gains seem to come primarily from better training-length performance (≤16k). The analogous naïve-RL baseline is missing from Fig. 8(b), which makes cross-dataset claims harder to assess.  


2. Figures 8(a) and 9 both report AIME’25 accuracy for Qwen3-1.7B/e3 but show different values and ranges; the relationship between the two protocols (SOTA comparison vs. “wait”-prompt budget forcing) should be reconciled to avoid confusion.

### Questions
1. Fig. 8(c): Is the teal curve measuring accuracy for an 8k-budget model, with the jump caused by switching the same checkpoint to 16k at evaluation? If so, could you clarify exactly which components change across the plotted segments?    


2. How is “verification” detected in traces—pure string heuristics (e.g., segments beginning with “Wait, …”, “Let me verify …”), or a more robust parser?

### Soundness
3

### Presentation
2

### Contribution
3
