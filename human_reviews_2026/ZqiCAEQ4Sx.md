# The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
We revisit test-time scaling for language model reasoning and ask a fundamental question: at equal token budget and compute, is it better to run multiple independent chains in parallel, or to run fewer chains that iteratively refine through sequential steps? Through comprehensive evaluation across 5 state-of-the-art open source models and 3 challenging reasoning benchmarks, we find that sequential scaling where chains explicitly build upon previous attempts consistently outperforms the dominant parallel self-consistency paradigm in 95.6% of configurations with gains in accuracy upto 46.7%. Further, we introduce inverse-entropy weighted voting, a novel training-free method to further boost the accuracy of sequential scaling. By weighing answers in proportion to the inverse entropy of their reasoning chains, we increase our success rate over parallel majority and establish it as the optimal test-time scaling strategy. Our findings fundamentally challenge the parallel reasoning orthodoxy that has dominated test-time scaling since Wang et al.’s self-consistency decoding (Wang et al., 2022), positioning sequential refinement as the robust default for modern LLM reasoning and necessitating a paradigm shift in how we approach inference-time optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The proposed sequential inference-time scaling (iterative reasoning) is superior to parallel self-consistency (independent reasoning paths) under matched compute. The presentation is clear, technically rigorous, and supported by extensive experiments across multiple models and benchmarks. However, a few aspects could be improved to enhance its credibility and accessibility for reviewers.

### Strengths
1. Introduces a training-free, information-theoretic voting method based on Shannon entropy. Improves accuracy in nearly all configurations (97% sequential, 100% parallel).

2. Careful matched compute design ensures fair comparison (identical token budgets). Extensive reproducibility details (prompts, hyperparameters, seeds, API configs).

### Weaknesses
1. The paper lacks a formal explanation for why sequential refinement works better (e.g., in terms of uncertainty reduction or information accumulation).

2. Builds on prior “self-refine” and “reflexion” frameworks; reviewers may see it as an empirical extension rather than a fundamentally new paradigm.

3. Uses small sample sizes (≈30 problems per benchmark), so some reported differences may not be statistically robust. Needs clearer reporting of p-values or confidence intervals in the main text.

4. Sequential reasoning is slower in wall-clock time since it runs steps serially. No quantitative latency analysis or discussion of mitigation (e.g., parallelized refinement).

5. The creative-writing ablation feels loosely connected to the paper’s core reasoning claim and could be trimmed or moved to the appendix.

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
4

### Summary
This paper revisits inference-time scaling for large language model (LLM) reasoning and challenges the long-standing assumption that parallel self-consistency decoding (Wang et al., 2022) is the optimal test-time scaling method. The authors propose a systematic comparison between parallel and sequential reasoning paradigms under matched compute budgets across five open-source LLMs (GPT-OSS, Qwen3, Kimi-K2) and three reasoning benchmarks (AIME-2024/2025 and GPQA-Diamond).

They find that sequential reasoning, where each reasoning chain iteratively refines previous attempts, outperforms parallel reasoning in 95.6% of settings, achieving up to 46.7% accuracy gains. Furthermore, the paper introduces Inverse-Entropy Weighted (IEW) Voting, an information-theoretic aggregation method that weights answers by inverse Shannon entropy of token-level log probabilities. This method yields consistent improvements over majority voting in both sequential and parallel setups, achieving optimality in 97% of configurations.

### Strengths
1, This paper challenges a widely accepted inference-time scaling orthodoxy (parallel self-consistency) with compelling evidence favoring sequential reasoning.

2, Controlled matched-compute setup and multi-model, multi-domain evaluation ensure fairness and reproducibility.

3, Inverse-entropy voting introduces a principled, information-theoretic mechanism that improves upon heuristic majority voting.

4, This paper demonstrates generality across reasoning, scientific, and creative tasks, reinforcing the universality of sequential refinement.

### Weaknesses
1, More related works should be discussed. e.g. https://aclanthology.org/2024.findings-emnlp.135.pdf, https://arxiv.org/abs/2401.02009, https://arxiv.org/abs/2308.00436. For example, at the same cost, does the proposed method perform better than mirror-consistency, self-contrast & self-check?

2, The main benchmarks (AIME, GPQA) focus on mathematical and scientific reasoning; inclusion of commonsense or real-world tasks (e.g., MMLU, GSM8K) would further support generality.

3, Self-refinement is somehow a doubtful pathway (http://arxiv.org/abs/2310.01798). The paper could better explain its qualitative decision process and failure modes under extreme conditions.

### Questions
1, What's the performance comparison between the consistency-based methods and other inference-time methods? e.g. multi-agent systems or other prompting methods like step-back https://arxiv.org/abs/2310.06117. Or let me ask in another way, why should we keep optimizing consistency-based methods, given all other prompting strategies?

2, The method is mainly a prompting engineering work. Can the llm be trained to be better at self-refinement?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the prevalent parallel reasoning paradigm (Self-Consistency) in large language model (LLM) inference scaling. Through rigorous experimentation across five state-of-the-art open-source models and multiple benchmarks (AIME, GPQA-Diamond), the authors demonstrate that sequential refinement (where LLM reasoning iteratively builds upon and corrects prior outputs) outperforms parallel approaches in 95.6% of configurations under a crucial condition: matched token budget/compute. This superiority is achieved without additional fine-tuning, leveraging the inherent mechanisms of iterative error correction and progressive context accumulation unique to the sequential process.

One  contribution is the introduction of Inverse-Entropy Weighted (IEW) Voting, a training-free method that uses token-level log-probabilities to quantify model confidence (lower Shannon entropy equals higher confidence) and assign higher weight to more confident chains. IEW Voting proves to be the optimal aggregation strategy across both sequential and parallel paradigms, achieving optimal performance in 97% of sequential configurations. The paper advocates for a paradigm shift, positioning sequential refinement as the robust default for LLM reasoning, with the 6-chain configuration emerging as the optimal balance of compute and performance gains.

### Strengths
* The paper offers near-universal evidence (95.6% win rate) that sequential reasoning outperforms the parallel method (Self-Consistency) across diverse LLMs and complex reasoning tasks

* The technical contribution of Inverse-Entropy Weighted (IEW) Voting is elegant and training-free, providing a principled way to leverage the LLM's inherent uncertainty (via logprobs) to aggregate results.

* The paper is in an important area, and we definitely need more analysis and interesting studies about detailed aspects of reasoning and test-time scaling. The paper is also well written, and well organized, and relatively easy to follow (despite being fairly technical). Good work!

* The comparison is fair and scientific, strictly matching the total token budget between sequential and parallel configurations (e.g., $N \times 4096$ tokens). The paper also studies multiple benchmarks (math and creative) and studies multiple base models (GPT, Qwen, Kimi).

### Weaknesses
* The paper acknowledges that sequential, serial execution has a substantial wall-clock time overhead compared to parallel methods, making it challenging for real-time applications

* The core advantage is hypothesized to come from Error Correction and Context Accumulation, but the experiments do not empirically decouple and quantify the contribution of these two distinct mechanisms

* The Creative Tasks ablation shows a divergent trade-off (Sequential: high lexical diversity; Parallel: high semantic diversity), suggesting the "universal superiority" may only hold for correctness-focused, convergent reasoning tasks

* In my opinion, the paper is somewhat borderline because the new method has a practical limitation of high latency. Specifically, it seems difficult to parallelize the method, and so the wall clock time is high. I am curious if there are ways to mitigate this (see questions later).

* I am open to raising my score, but I have a handful of technical questions that I would like some clarity on.

* Another big issue, which you should just fix (we don't need to talk about it). The related work is super limited. I don't seem to find a related work section. Please add this. Also fix the parentheses and the citations for the related work -- there is often no space between the text and the starting parens, which is sloppy.

### Questions
* To mitigate the latency, have the authors explored an ablation where the token limit of each individual refinement step is aggressively constrained (e.g., to 512 tokens instead of 4096) to reduce the time-per-step while still utilizing the same total budget? This would demonstrate if the iterative refinement loop itself, rather than the length of each attempt, is the true source of the sequential edge, making the method more practically viable.

* The paper attributes the sequential advantage to three mechanisms: (1) Iterative Error Correction, (2) Progressive Context Accumulation, and (3) Answer Verification. The current experiments combine all three. Can the authors conduct an ablation to decouple the effects of the explicit error correction instruction versus the passive context accumulation? Compare Sequential Refinement (prompt: "Review your previous reasoning, identify any gaps or errors...") against a new baseline Sequential Re-Prompting, which uses the history as context but with a neutral continuation prompt (e.g., "Continue the analysis with the next step.") This would isolate the gain derived from the LLM's ability to respond to an explicit error-correction instruction, strengthening the claim about the mechanism.

* The Inverse-Entropy Weighted (IEW) Voting metric uses the mean entropy across all tokens in the reasoning chain (Equation 1). However, the Appendix F ablation showed **surprisingly identical results** whether using the mean, median, maximum, or minimum entropy of the full chain. This suggests that important localized confidence signals might be diluted by the mean or are entirely sufficient using minimal tokens. Could the authors present an ablation comparing the current mean-chain entropy weighting to a Localized Answer-Token Entropy weight? This weight would be based solely on the log-probabilities of a small window of tokens (e.g., the final $N=100$ tokens) immediately preceding the extracted answer tag. This would test the hypothesis that the confidence signal is localized to the concluding segments of the chain, rather than being distributed across the entire, potentially noisy, reasoning sequence.

* The Creative Task Ablation (Figure 3) reveals a crucial trade-off: parallel generation achieves higher Semantic Diversity, while sequential refinement achieves higher Lexical Diversity. Sequential is superior for refinement and depth, while parallel is better for exploration and breadth. Does the claim that Sequential is the "robust default" still hold for tasks requiring high divergent exploration (e.g., novel code generation, brainstorming)? Could the authors propose a Hybrid Gated Scaling approach? This approach would execute a small initial set of parallel chains (for breadth) and then use the IEW metric to select the top-performing parallel result, which is then fed into a subsequent, short sequential refinement chain (for depth and error correction). This framework would leverage the strengths of both paradigms and provide a more nuanced "optimal" strategy.

* For the limited related work, one question. I don't know this area very well -- are there baselines to compare against from recent papers?

### Soundness
3

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
3

### Summary
This paper compares sequential test-time scaling (iterative self-refinement where each chain conditions on earlier reasoning) to the dominant parallel self-consistency approach at matched token budgets. Across five OSS models (GPT-OSS-20B/120B, Qwen3-30B/235B, Kimi-K2) and three benchmarks (AIME-2024/2025, GPQA-Diamond), the authors report that sequential reasoning wins in 95.6% of configurations with gains up to 46.7 pp; they also introduce inverse-entropy weighted (IEW) voting, which weights each chain’s answer by the inverse of the chain’s mean token-level Shannon entropy computed from logprobs, and claim it is near optimal among tested aggregators. Key experimental choices include strict token-budget matching (e.g., 6×4096 tokens for both paradigms) and fixed system/refinement prompts.

### Strengths
1.	Claim clearly presented with wide variety of evidence. The author claims that sequential self-refinement beats parallel self-consistency at matched token budgets. The claim is supported by supported by results across 5 models, 3 benchmarks, and multiple chain counts (3/6/9), and indeed show the higher accuracy in almost all the settings. The wide range of configurations ensures the generalizability of the claim.
2.	Training-free and cross-model. The author avoid additional fine-tuning and show the effect across different families (GPT-OSS, Qwen3, Kimi-K2), which strengthens generality beyond a single architecture or a special-trained model.
3.	Attempted fairness via matched token budgets and fixed decoding. In the experiments, author ensures the fairness of comparison by keep the temperature fixed and top k disabled. Most critically, the authors matched the total tokens generated between sequential and parallel. Ablation studies were conducted to also analyze how the token budge affect the performance

### Weaknesses
1.	Hypothesis on token-level entropy and model confidence as a metric to weigh the chains’ quality is not verified. The author proposed to use token-wise entropy as a weighing factor for generated chains. The critical assumption here is model confidence is positively correlated with quality or correctness of the response. It has been a common phenomenon that model tends to generated confidently the wrong answer under certain given prompt. The test to verify the effectiveness of token level entropy as a metric is missing.
2.	Matching tokens budget does not equate match compute. Equal tokens do not equal compute because sequential steps repeatedly read longer contexts (quadratic attention cost), while parallel chains don’t. Token budgets therefore may understate sequential compute and the comparison for CoT quality is unfair since more computes are spent on generating a new token in sequential than parallel generation.

### Questions
1.	The first is whether the sequential generation benefits more from iterative refining or the voting process. If only the final answer is taken, do you almost always get the same results? Is voting more central in the sequential generation than in previous generations? While the author conducted a variety of experiments on voting methods, it might be best for comparison if an ablation experiment without any voting is conducted.
2.	Could you justify the correlation between token-level entropy and the correctness of the response?
3.	Are questions drawn randomly from GPQA-Diamond? Would a small sample size of 30 bias towards a specific domain of knowledge?
4.	Refinement Prompts: How sensitive are the sequential results to the specific refinement prompts used? Did the authors experiment with other self-correction or refinement prompting strategies, and if so, how did their performance compare?

### Soundness
2

### Presentation
2

### Contribution
3
