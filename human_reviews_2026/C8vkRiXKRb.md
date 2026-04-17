# RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Reinforcement learning (RL) can refine the reasoning abilities of large language models (LLMs), but critically depends on a key prerequisite: the LLM can already generate high‑utility reasoning paths with reasonable probability. For tasks beyond the LLM’s current competence, such reasoning path can be hard to sample, and learning risks reinforcing familiar but suboptimal reasoning. 
We are motivated by the insight from cognitive science that *Why is this the answer?* is often an easier question than *What is the answer?*, as it avoids the heavy cognitive load of open-ended exploration, opting instead for explanatory reconstruction—systematically retracing the reasoning that links a question to its answer. 
We show that LLMs can similarly leverage answers to derive high-quality reasoning paths. 
We formalize this phenomenon and prove that conditioning on answer provably increases the expected utility of sampled reasoning paths, thereby transforming intractable problems into learnable ones. Building on this insight, we introduce RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end framework that uses answer-conditioned reasoning as a variational surrogate for question-only reasoning. Experiments in both general and math domains demonstrate consistent improvements over strong baselines. We further analyze the reasoning behavior and find that RAVR reduces hesitation, strengthens conclusion consolidation, and promotes problem-specific strategies in reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes RAVR (Reference-Answer-Guided Variational Reasoning), an end-to-end framework that leverages reference answers to enhance large language models’ (LLMs) reasoning capabilities in reinforcement learning (RL). It formally proves that conditioning on reference answers increases the expected utility of sampled reasoning paths, transforming intractable tasks into learnable ones by drawing on the cognitive science insight that explaining answers is easier than generating them.

### Strengths
solid theoretical foundation, as it provides rigorous mathematical proofs to validate that reference-answer conditioning amplifies high-utility reasoning paths

Its innovative integration of variational inference, using answer-conditioned reasoning as a surrogate for question-only reasoning, enables end-to-end training, avoiding the multi-stage complexity of existing methods like STaR.

### Weaknesses
- The reliance on specific LLMs (Qwen3-1.7B) and benchmarks raises concerns about generalizability

- the paper focuses on tasks with clear reference answers (e.g., multiple-choice, math problems), leaving untested whether RAVR works for scenarios where answers are ambiguous or multi-faceted.

- I'm not quite sure how the method performs RL training, e.g., GRPO. Is it that the input is x and y, and the output is z? 
What are the reward signals in GRPO? As far as I know, the thought path (z) is not a verifiable signal.

- Additionally, during test-time, since there is no ground truth, how does it perform inference?

### Questions
see Weaknesses

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
This paper proposes RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end training framework that leverages the reference answer during training to improve LLM's question-only reasoning at inference. The core idea is to treat answer-conditioned reasoning as a variational surrogate for the question-only prior. 

The authors first formalize why conditioning on the answer should bias sampling toward higher-utility reasoning paths: using Bayes rule, the posterior over reasoning traces reweights the prior, leading to a provable increase in both the probability of above-average traces and the expected utility under the posterior. They then derive an ELBO target that maximizes answer likelihood under the posterior while minimizing a KL from posterior to prior, and instantiate a practical objective with (i) a utility-baseline reward measuring posterior improvement over the prior, (ii) KL sample reweighting by the utility, and (iii) a simple answer-prefix (“The answer is …”) trick for stable probability estimation. 

Prompt templates are provided to make posterior traces read like first-person, think-aloud reasoning without revealing access to the answer. On Qwen3-1.7B, RAVR shows consistent gains across GPQA-Diamond, MMLU-Pro, AIME24/25, AMC23, and Minerva, and appears more sampling-efficient than GRPO (similar accuracy with 8 vs 24 rollouts).

### Strengths
1. Framing answer-conditioned reasoning as an amortized posterior that teaches the question-only prior via an ELBO-style objective, plus practical touches (utility-baseline, KL reweighting, answer-prefix, posterior “think-aloud” instructions), is novel. I mean using answers to induce better reasoning is standard, but incorporating it into training objectives is interesting.
2. The theoretical part is clearly derived， posterior re-weighting by s(z) implies a larger mass on good traces and a higher expected utility.
3. Empirically, RAVR improves Qwen3-1.7B's average over RL baselines (GRPO, DAPO, VeriFree, RLPR) across both general and math setups.

### Weaknesses
1. I am still concerned that models can inflate likelihood via stylistic cues ("answer-prefix"), instead of improving true correctness. The paper partly addresses this with a baseline and length normalization, but there is no more discussion.
2. Not sure if RAVR's advantage persists under equal wall-clock or token budgets as RL baselines.
3. All results use Qwen3-1.7B. If during rebuttal the authors can show similar improvements on other and even better larger models, I will raise my score.
4. Numerous minor grammar/typo issues ("refernce answe", "postive", "noval").

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes RAVR, a reference-answer-guided RL framework that enhances LLM reasoning on math and logic benchmarks. RAVR trains the model using an answer-conditioned “posterior” distribution that maximizes the likelihood of the reference answer while remaining close to the question-only “prior,” incorporating a prior-based utility baseline, reward-weighted KL, a think-aloud prompt, and an answer-prefix cue, end-to-end optimized via GRPO. On Qwen3-1.7B, RAVR achieves notable gains over previous baselines: when trained on CrossThink-QA it reaches 40.91 GPQA-Diamond (+5.56 over DAPO) with the best overall average of 44.75, and when trained on DeepMath it obtains the top overall score of 45.02, demonstrating strong out-of-domain transfer.

### Strengths
+ The paper introduces an answer-conditioned variational objective that meaningfully improves exploration in RL-based reasoning, offering a conceptual advance over prior reward-only or STaR-style methods.

+ The experiment results show consistent performance gains across multiple reasoning benchmarks, supported by various ablations and behavioral analyses that validate the source of improvement.

### Weaknesses
+ RAVR treats the reference-answer token probability as the reward signal for the generated reasoning trace. However, this self-supervised likelihood-based reward is known to be noisy and susceptible to reward hacking, where models can optimize for probability shortcuts and could easily lead to training collapse.


+ The method depends on having gold reference answers during training, which may limit its applicability in weakly supervised or open-ended settings where answers are unavailable or ambiguous.


+ While the paper reports gains on the selected benchmarks, it does not provide qualitative evaluations to show whether the performance transfers beyond multiple-choice and math tasks to other broader settings that require strong reasoning capabilities.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
