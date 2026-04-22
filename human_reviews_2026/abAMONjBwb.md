# Native Reasoning Models: Training Language Models to Reason on Unverifiable Data

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 2, 8

## Abstract
The dominant paradigm for training large reasoning models—combining Supervised Fine-Tuning (SFT) with Reinforcement Learning with Verifiable Rewards (RLVR)—is fundamentally constrained by its reliance on high-quality, human-annotated reasoning data and external verifiers.
This dependency incurs significant data-collection costs, risks embedding human cognitive biases, and confines the reinforcement learning stage to objectively assessable domains like mathematics and coding, leaving a vast landscape of unverifiable tasks unaddressed.
To overcome these limitations, we introduce Native Reasoning Training (NRT), a novel framework that cultivates complex reasoning by having the model generate its own reasoning traces using only standard question-answer pairs, thereby obviating the need for expert-written demonstrations.
NRT reframes the training problem by treating the reasoning process as a latent variable. 
It employs a unified training objective that models reasoning as an optimization problem, intrinsically rewarding paths that increase the model's likelihood of producing the ground-truth answer.
This unified perspective allows us to analyze intrinsic failure modes of prior methods, such as policy collapse, and systematically design more robust reward aggregation functions, creating a self-correcting feedback loop where the model learns to \textit{think} in ways that resolve its own uncertainty.
Empirical evaluation on Llama and Mistral model families demonstrates that NRT achieves state-of-the-art performance among verifier-free methods, significantly outperforming standard SFT baselines and prior verifier-free RL methods.
Our approach yields particularly strong performance gains in complex reasoning domains and exhibits high robustness to policy collapse, offering a general, scalable path toward building more powerful and broadly applicable reasoning systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a unified verifier-free framework (NRT) that treats the reasoning trace as a latent variable and maximizes an intrinsic reward computed only from question-answer (QA) pairs without expert CoT or external verifiers. The authors say this re-frames prior methods, exposes failure modes (e.g., policy collapse), and yields new reward aggregations, etc and present several experiments focusing on the llama model family.

### Strengths
* The paper presents a unified formulation that subsumes several verifier-free ideas into one token-level objective and gradient, making design levers more explicit by casting them as choices of $f$, which is a helpful perspective.

* NRT relies only on QA pairs which can potentially help lessen/reduce dependence on costly human chain-of-thought (CoT) annotations or judge models.

* It appears that NRT might avoids the mode collapse observed in RLPR, maintaining diverse reasoning traces while still converging on higher answer accuracy.

### Weaknesses
* The experimentation is quite limited in restricting attention to just two models within the same model family; it's unclear how well these results can generalize to other more model families as a result. Results on bigger models, alternative corpora, and domains with very long outputs or strong factuality constraints would strengthen the claims. 

* The results in Table 3 do not seem to show consistent or state-of-the-art results by NRT as claimed in the paper; the results seem mixed as claims of significant gains are made only on the average score. The individual scores tell quite a different story. Moreover, it's unclear what variant of NRT to use consistently over others and why as the results, again, seem to be mixed.
  * Just as one example, TruthfulQA hardly moves in the results; if NRT induced genuinely more truthful reasoning, I would expect clearer TruthfulQA gains, no?

* The paper also notes prior baselines use different datasets and response lengths (Appendix D, p18), while NRT trains on long-answer QA (longer avg response). From this, it seems like the data regimes also differ across prior work; Appendix D (Table 5, p18) emphasizes that earlier baselines trained on short-answer data (Verifree) or specific long-form proofs (JLB), while NRT trains on a general instruction-following mix with long reference answers. 
  * Longer reference answers can provide denser supervision (e.g., more tokens with explicit targets) and thus richer learning signals even before any reward optimization. Instruction-tuned datasets like tulu-3 encode more linguistic variety than the narrower math-reasoning data used by RLPR. So when the paper attributes higher benchmark scores to NRT, if I understand correctly, how do we separate how much comes from the NRT objective vs. the richer data regime? In effect, two variables have changed at once: the learning rule and the data distribution. From my view, that weakens the causal chain needed to claim “NRT is a better training method” rather than “NRT + better data does better.”

* One thing that caught my attention is that the paper re-implements RL baselines and, “to ensure a fair comparison,” injects the same reward stabilization and format-supervision used by NRT into those baselines, because the original methods were designed for instruction-tuned models (Appendix F, p20). Doesn't this change the baselines’ training conditions from what their original authors intended? I have not seen the paper quantify how this affects each baseline’s behavior (e.g., ablations of RLPR with/without format loss, length normalization, etc.). Without such controls, I fear that these modified baselines may risk confound how fair of a comparison this is.
  * More specifically, my concern is that this may actually change what those baselines were originally optimizing for. The original RLPR paper, for instance (if I remember correctly), did not use format-supervision or clipping relative to a "no-reasoning" baseline; those are distinct design choices that can likely alter gradient magnitudes and trace-length dynamics. As such, when the paper shows NRT outperforming RLPR (Table 3, p7), it’s unclear whether NRT is truly better than the RLPR method or just better than their internally modified version of RLPR. For a fair method-level comparison, the baselines should be reproduced under their original settings and under the shared settings etc. with results reported for both. Otherwise, it's difficult in my view to isolate the causal factor of improvement.
  * Similarly, it seems that NRT can benefit from some extra things like reward clipping, group-wise normalization, and a mild format loss to maintain thinking/reasoning delimiters (Appendix F); however, these may be problems that prior baselines did not address in the same way (i.e., the converse of the earlier point: if those aids disproportionately help NRT but aren’t tuned equally for baselines, the comparison again favors NRT not because its underlying intrinsic-reward formulation is superior, but because the training recipe is more stable).

### Questions
See weaknesses.

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
The paper proposes Native Reasoning Training (NRT), a verifier-free reinforcement learning framework for reasoning models. Instead of relying on Reinforcement Learning with Verifiable Rewards (RLVR)—which needs an external programmatic checker for correctness—NRT treats the reasoning trace as a latent variable and optimizes rewards that reflect how much a generated trace increases the model’s likelihood of producing the ground-truth answer. The authors present a unified view of intrinsic rewards and discuss several aggregation schemes (e.g., arithmetic mean, geometric mean, weighted-sum that prioritizes difficult tokens) to avoid failure modes such as policy collapse. They couple this with a GRPO-style group advantage estimator. Experiments use a standardized evaluation harness across nine benchmarks (e.g., MMLU, BBH, GSM8K, MATH, HumanEval, DROP, PopQA, TruthfulQA, IFEval). Reported results claim state-of-the-art among verifier-free methods and robustness to training instabilities.

### Strengths
Clear stance vs RLVR: positions NRT for unverifiable tasks, removing dependency on external checkers. 

Unified reward view with explicit analysis of aggregation pitfalls  and a weighted-sum that targets low-probability “hard” answer tokens. 

Stability measures: GRPO-style relative advantages over an empty-trace baseline; practical rollout details are transparent. 

Standardized evaluation harness across a broad benchmark suite, improving comparability.

### Weaknesses
Self-referential reward: Even with improved aggregation, the intrinsic signal is driven by the model’s own likelihood over the ground-truth answer. This can inflate confidence without proving better reasoning. The paper lacks external evidence that higher intrinsic reward correlates with logically valid intermediate steps.

Scope vs RLVR: While NRT addresses unverifiable domains, the paper does not show how it competes where verifiers do exist (math/code). The claims are limited to “verifier-free SOTA,” which weakens broader impact. 

Aggregation novelty is incremental: GM and difficulty-weighted sums are reasonable but not conceptually new. The unification is helpful but largely consolidates known heuristics. 

Evidence gaps: No analysis with external judges or rule checkers to validate that selected traces embody correct reasoning, rather than patterns that merely raise answer likelihood.

### Questions
Correlation with reasoning correctness: Can you show that higher intrinsic rewards (under your aggregation) correlate with externally judged reasoning quality (e.g., logic checkers, stronger LLM judges) rather than only internal likelihood gains?

When verifiers exist: On math/code tasks where RLVR is available, how does NRT compare under similar compute? Could NRT complement RLVR (e.g., pre-train with NRT, then fine-tune with RLVR)?

Choice of aggregation: Why is the weighted-sum superior to geometric mean in practice? Please provide ablations isolating the effect of each aggregation on collapse mitigation and accuracy.

Generalization to open-ended outputs: Since rewards depend on ground-truth answers, how does NRT extend to truly open-ended tasks without reference answers?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new framework called "Native Reasoning Training (NRT)" aimed at addressing the limitations of current training paradigms for large-scale reasoning models (SFT+RLVR). By treating the reasoning process as a latent variable and using the model’s own confidence in predicting standard question-answer pairs as an intrinsic reward, NRT eliminates the need for expert-annotated reasoning trajectories or external verifiers. However, the authors do not explain why this approach works. Since the quality of the reference answers cannot be guaranteed, it is unclear whether this method is justified. The experiments were conducted only on Llama-3.1-8B and Llama-3.2-3B models, which are too small to adequately validate the authors’ claims. Furthermore, in the experimental section, all comparative methods were evaluated under the authors’ own baseline implementations, raising doubts about the reliability of the results. The authors could have instead used existing baseline settings for a more convincing comparison.

### Strengths
- This paper is commendable for its exceptional clarity in presentation, which significantly facilitates the review process. The precise and well-structured descriptions allow readers to quickly grasp the core contributions of the work, as well as to readily identify its strengths and potential limitations. 
- The cover letter provides a remarkably clear and concise summary of the manuscript's key findings. This thoughtful presentation is highly efficient, as it enables reviewers to understand the essence of the paper at a glance, thereby saving considerable time and effort during the initial evaluation.

### Weaknesses
- The author did not explain why the method is effective. The author's "native" responses are derived from the "think-off" mode. Since the quality of the reference answers cannot be guaranteed, I cannot confirm whether this approach is reasonable.
- The experiments in the paper were conducted solely on the Llama-3.1-8B and Llama-3.2-3B models. The models are too small to adequately validate the author's claims, and the persuasiveness of experimental results from smaller-scale models is relatively weak. In contrast, many existing methods validate their claims on larger models [1].
- In the experimental section, all comparative methods were evaluated under the "Our baseline implementations" setting. The reliability of the experimental results is questionable, as the author could have chosen existing baseline settings for the experiments instead.

[1] Yunhao Tang, Sid Wang, Lovish Madaan, and Remi Munos. Beyond verifiable rewards: Scaling reinforcement learning for language models to unverifiable data. arXiv preprint arXiv:2503.19618, 2025.

### Questions
1. Why are the results generated with the "think" mode disabled used as the reference answers?

2. Why wasn't the method's effectiveness validated on larger-scale models?

3. Why weren't the existing baseline experimental setups and results adopted?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduce NRT - a new LLM framework for verifier-free reasoning .The new method is based on new reward functions which defined on model generated think traces. Next they optimize the model using off-policy RL.  The update is based on 2 parts: (1) Trace reward (2) Token reward for right answer prediction. They propose 3 new schemas for reward: geometric mean, arithmetic mean and weighted sum that that prioritize difficult tokens. 
The authors empirically demonstrated their models using 2 small base models (Llama - 3B / 8B)

### Strengths
The paper is original. It proposed a more  general schema which unify 3 previous verifier free methods  (JLB, Verifree, and RLPR) . The paper is well written. The quality of ablation study is good.

### Weaknesses
No obvious weak points

### Questions
How would you modify NRT reward aggregation function  to encourage short answers and penalize too long answers?

### Soundness
3

### Presentation
4

### Contribution
3
