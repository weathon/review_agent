# PACR: Progressively Ascending Confidence Reward for LLM Reasoning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model’s evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PACR (Progressively Ascending Confidence Reward), a dense reward signal for training LLM reasoning via reinforcement learning. The core idea is to reward intermediate reasoning steps based on whether they increase the model's probability of the ground-truth answer. The authors provide empirical observations and theoretical justification for this approach, implementing both sparse (trajectory-level) and dense (step-level) variants.

### Strengths
Conceptual clarity: The idea of using internal confidence dynamics as a reward shaping signal is intuitive and well-motivated.

Implementation simplicity: PACR does not require an external reward model, making it lightweight and practical.

Empirical validation: The paper includes quantitative results on multiple math reasoning benchmarks (MATH500, AIME, AMC, etc.) showing some performance gains.

### Weaknesses
## Flawed Theoretical FoundationProposition 1 is trivial and doesn't validate the method's utility:

The proposition proves that E[C_k] ≥ 0 when h_k is sampled from π_θ(·|q, Y_gt, H_<k) (the "oracle policy"). 
However, during actual RL training, steps are sampled from π_θ(·|q, H_<k) without access to Y_gt. 
The proof merely shows that conditioning on the answer increases confidence in that answer—this is tautological. 
The critical gap: the paper never establishes that actually sampled steps (without Y_gt conditioning) will exhibit this property. 
This undermines the entire theoretical motivation for the method. 

## Weak and Inconsistent Empirical Results

Table 1 shows marginal and unstable improvements:

- Many improvements are within noise margins (e.g., +0.6, +0.8 on MATH500)
- I admire the authors' frankness in posting those results. Several datasets show degradations (e.g., Minerva: -2.9 for Sparse-PACR on 1.5B; AMC: -1.2 for Sparse-PACR on 7B)
- The average improvements (≤3.0 points) are modest given the added computational complexity
- No statistical significance testing provided despite claiming results over 3 seeds

## Circular Reasoning in Observational Studies

Observation 1 (Section 4.1) lacks proper controls:

- Finding that correct trajectories have more positive C_k steps could simply mean: correct reasoning leads to correct answers (obvious)
- No analysis controlling for confounding factors (e.g., trajectory length, model confidence calibration)
- Causality is unclear: does confidence growth cause correctness, or does correctness cause confidence growth?

Observation 2's methodology is questionable:

- Using GPT-5 to judge "logical coherence" introduces significant subjectivity and potential bias
- The distinction between "coherent" and "spurious" reasoning that reaches correct answers is ill-defined
- High confidence in a spurious path might actually indicate a model failure rather than validate the reward signal

### Questions
Why not compare against actual trained process reward models?

Can you show results with statistical significance tests?

What happens when the model is highly confident in the wrong answer—does PACR still provide useful signal?

Compare against recent dense-reward or implicit-reward baselines (e.g., ConfPO, TLCR, DAPO, DeepSearch).

### Soundness
1

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
This paper proposes Progressively Ascending Confidence Reward, called PACR, to improve LLM reasoning inside the RL with verifiable reward setting. The key idea is simple. While the model writes a chain of thought, at each step the authors compute how much the log probability of the true answer increases, and they use this confidence gain as a dense, model intrinsic reward. They give two variants, Sparse PACR at trajectory level and Dense PACR at step level, and combine them with GRPO under RLVR. On three Qwen math models, the paper reports faster exploration and small but consistent gains on AIME 2024, AMC 2023, MATH500, Minerva Math, and OlympiadBench.

### Strengths
1. The paper gives a very clear inductive bias: along a good reasoning path, confidence in the ground truth should tend to go up. Turning this into a dense reward that needs no extra reward model is practical and clean.
2. The paper checks three open models and five math benchmarks. The main table shows Dense PACR improves average pass@1 over a strong Dr GRPO baseline, for example +2.5 on the 1.5B model and +3.0 on the 7B model.

### Weaknesses
1. Experiments only cover math datasets. Many recent results also evaluate general reasoning and code. It is not clear if PACR transfers beyond numeric answers or beyond tasks where the final answer is exactly verifiable, for example, long form QA or proofs with multiple valid forms. Comparison to broader settings in R1-style training or DAPO-like systems would strengthen the claim.
2. The proof shows that the expected confidence gain is non-negative when steps are drawn from the ground truth conditioned policy. In practice, training never samples from that oracle. This gives a nice intuition but a weak guarantee for the real policy, and it leaves open how often confidence will increase under noisy steps. A discussion that connects the oracle gap to measured gains would help.
3. Because the reward comes from the model’s own probability of the final answer, the model may learn to raise that probability early by adding hints or formatting patterns that correlate with the answer tokens, without improving reasoning quality. PRM works try to avoid such reward hacking with different credit assignment, for example, min form credit or verifier designs that measure true progress. A safety check or a negative control would be useful.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

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
Arguably, dense process rewards can lead to greater improvement with RL; however, acquiring quality process rewards can be costly. This paper introduces PACR, a novel method to enhance Reinforcement Learning with Verifiable Rewards (RLVR) for LLM reasoning. It provides a dense, model-intrinsic reward based on the idea that along a successful reasoning trajectory, the model's confidence in the correct final answer should progressively increase. The correlation between this increase in confidence and outcome correctness implies that it can serve as a process reward.

### Strengths
The paper addresses a key problem in RL training: that dense rewards are hard and expensive to acquire.

The paper's presentation is clear. The supporting evidence (observations 1, 2, and 3) for the method is mostly relevant and well thought-out.

The gain from the methods is good over the baseline Dr.DRPO , though in some certain test datasets it is negative.

### Weaknesses
The experiment is slightly limited, with only one training dataset and three models, and one baseline algorithm (Dr.GRPO). I would like to see one reasoning model (e.g., DeepSeek-R1-Distill-Qwen-1.5B) tested to see if the effectiveness of your proposed process reward still holds. Also, report the accuracy on AIME 2025.

I dislike the inclusion of Section 4.2, as it makes the proposed method look deep, whereas the key to the proof is really the artificial "oracle policy assumption". I think that Section 4.2 doesn't meaningfully justify your proposed process reward, or you can leave it in the appendix.

The authors only evaluate with pass@1 greedy decoding, yet it is debated whether RL post-trained models increase sampling efficiency (better pass@1) while reducing coverage (lower pass@k for large k compared to the base model). Therefore, the authors should also report $\text{pass@K}$ with a positive temperature.

The author can use this new process-reward for beam search. In my opinion, this is an important experiment, as if beam search leads to improvement, then I'm more confident that RL's improvement comes mainly from the process-reward and not your other design choices (e.g. mixing with outcome reward, mix-max scaling).

### Questions
How do the authors justify the process-wise group normalization (in lines 363, before the Min-Max scaling explanation)? Does there exist similar method in the literature? Does the way they divide the sequence into steps of varied lengths have any impact on this?

It is hard for me to read the small colored numbers in Table 1.

What is the setting of Figure 8?

Qualitatively, it would be nice to attempt to find failure cases when high growth doesn't correspond to an informative step, or when low growth doesn't correspond to an uninformative or wrong step. I can see in Figure 12 and 13 steps that have negative, large magnitude growth that are nevertheless informative.

I'm also concerned about the way you divide the sequence into steps. It is very hard to do that nicely by just pattern. I can see that you have many short steps. Did you consider other alternatives?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PACR, a dense, model-intrinsic reward for reinforcement learning in LLM reasoning. Instead of relying solely on sparse, outcome-based signals, PACR rewards stepwise increases in the model’s confidence in the ground-truth answer. The authors show that this “confidence ascent” acts as a strong training signal for training and can empirically enhance RL training performance. Experiments on multiple math-reasoning benchmarks (e.g., MATH500, AIME24, AMC23) demonstrate that PACR improves RL training performance.

### Strengths
1. The problem addressed in this paper is important. Incorporating more dense reward information into current RL pipelines remains an underexplored direction.


2. The proposed idea is reasonable and aligns with recent studies showing that a model’s reasoning confidence often correlates with the correctness of its answers, suggesting that such confidence signals could be valuable for training.


3. The writing is clear and easy to follow.

### Weaknesses
1. The approach for determining the reasoning step appears rather ad-hoc. It is unclear how this mechanism would transfer to other domains such as code generation, where the output often contains many new lines. Would this lead to an excessive number of reasoning steps for tasks involving long-context generation?

2. The training process seems to introduce additional computational overhead, particularly as the generation length increases, which could significantly inflate the number of reasoning steps. A detailed analysis of the training cost and the number of reasoning steps would help clarify the efficiency of the proposed method.

3. The experiments use a maximum generation length of only 3k tokens, which is much shorter than the 40k-token context supported by the Qwen3-4B model. Given that the proposed method may introduce substantial overhead for long-context scenarios, it would be valuable to examine its scalability with respect to context length to better assess its effectiveness.

4. Another possible explanation for the observed performance improvement could be more stable training due to the absence of zero-advantage samples. In vanilla GRPO, many responses have zero advantage and thus contribute nothing to training (though dynamic sampling can alleviate this issue). Since PACR avoids this, it would be helpful to compare against (or combine with) GRPO with dynamic sampling to determine whether the gains primarily stem from denser rewards or from improved training stability.

### Questions
listed in weakness

### Soundness
2

### Presentation
3

### Contribution
2
