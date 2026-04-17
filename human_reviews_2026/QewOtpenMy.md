# One-Token Verification for Reasoning LLMs, Anytime, Anywhere

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Reasoning large language models (LLMs) have recently achieved breakthrough performance on complex tasks such as mathematical problem solving. A widely used strategy to further improve performance is parallel thinking, wherein multiple reasoning traces are generated, and the final prediction is chosen using methods such as Best-of-$N$ or majority voting. However, two limitations remain: most existing methods lack effective mechanisms for assessing the quality of reasoning traces, typically relying only on final logits or answers, and multi-sample decoding incurs substantial inference latency for long outputs. To address these challenges, we propose **One-Token Verification** (OTV), a lightweight framework for assessing reasoning quality via a single-token forward during generation. The proposed OTV method introduces a learnable LoRA-based role vector that, without interfering with the primary reasoning process, enables the LLM to assume a verification role and probe the past KV cache for correctness confidence estimation. Unlike generic verifiers or external reward models, OTV is trained natively for each LLM, directly leveraging internal computations for token-level scoring. As a result, OTV can provide confidence signals at any point in generation and for any token position, realizing "anytime, anywhere'' verification.
Experiments on math benchmarks show that OTV consistently outperforms the state-of-the-art baselines in parallel thinking. Moreover, based on OTV, we introduce efficient variants that terminate most traces early and retain only a single complete reasoning path, reducing token usage by up to 90\%. In this setting, OTV maintains superior performance while favoring shorter and more reliable solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an efficient way (using an adapter, based on LoRA) to train a verifier to assess intermediate results of reasoning traces. The experimental results demonstrate the superior performance of the proposed method on math benchmarks against SOTA baselines.

### Strengths
1. This paper proposes a clever way to address the computational efficiency for the intermediate assessment, which could also be easily adapted to other scenarios such as actor-critic based PPO training.
2. The experimental results show a clever gain against the baselines, especially for the weighted average scenario
3. The paper is well-written and clearly presented

### Weaknesses
1. The second contribution in the Introduction part does not hold 
> We design a heuristic supervision and parallel training scheme that provides dense step-level signals, making the proposed OTV method efficient, scalable, and adaptable to metrics beyond answer correctness

[1,2] have already pointed out the value/advantage function should be the correct training target for step-wise signals, it remains unclear to me why these heuristics are developed. There lacks the discussion and comparison to these prior works.

2. Main evaluation is only conducted on Qwen models and AIME benchmarks. Given the evaluation issues raised by recent works [3,4], I think the evaluation is not comprehensive.

3. The BoN results do not show a consistent improvement (Table 2). 

[1] Feng et al. Step-by-Step Reasoning for Math Problems via Twisted Sequential Monte Carlo. ICLR 2025.

[2] Setlur et al. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. ICLR 2025

[3] Shao et al. Spurious Rewards: Rethinking Training Signals in RLVR. 

[4] Hochlehnert et al. A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility. COLM 2025.

### Questions
The novelty part of this work seems not so clear. Given my list of previous works, this paper only stands out in proposing a more efficient strategy to finetune/store the verifier rather than making any algorithmic innovations. Most importantly, there lacks a clear efficiency comparison with baselines.

For example, within the same time budget, could DeepConf have more sampled trajectories?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To assess the quality of reasoning traces at any point in generation and for any token position, the authors introduce OTV, a lightweight framework for assessing reasoning quality via a single one-token forward during generation. They conduct extensive experiments across multiple reasoning LLMs and math benchmarks.

### Strengths
1. The paper shows that OTV can surpass recent PRMs across various LLMs and settings (e.g., self-consistency, BoN).
2. The authors visualizes the confidence trajectories of different scoring methods, highlighting that OTV more faithfully captures the
underlying reasoning quality.

### Weaknesses
1. The authors miss an important reference: "Token-Supervised Value Models for Enhancing Mathematical Problem-Solving Capabilities of Large Language Model, ICLR 2025". The missing reference is totally critical due to the fact that this work already offers anytime, anywhere verification even without any special token like `[ToT]`. Moreover, this work already shows its effectiveness for step-level signals, but the authors do not compare OTV with this work at all. Finally, this work theoretically shows that its method can allow tree search algorithms to be value-guided, but OTV is based on heuristic confidence. In other words, there is no theoretical analysis why OTV can work well. In this sense, the novelty of OTV seems negligible.
2. In addition to Figure 1, it would be instructive if the authors share the example of confidence decreasing for inference, whether this example really results in a wrong answer, and at which token the confidence decreases dramatically.
3. It seems necessary to show that OTV can be helpful for MCTS or Step-by-Step Beam Search to show that OTV can be also applied to advanced search strategies.

### Questions
Can OTV be extended to other tasks (e.g., code generation, neural theorem proving)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose One-Token Verification (OTV), a very lightweight method to assess reasoning quality during the model reasoning process. OTV trains a LoRA for self-verification as a role vector and uses special token [ToT] to measure the verification confidence. The experiment shows promising result of OTV in parallel thinking.

### Strengths
1. Paper is relatively well written and easy to follow.
2. OTV is sound and well constructed in the methodology.

### Weaknesses
1. The concept of OTV is not super new in that it uses (an adapted) model internal confidence to measure the certainty of a verifiable result.
2. Evaluation is not comprehensive enough. The evaluation only shows result against math reasoning, and have not shown OTV can generalize beyond the standard math benchmarks. 
3. Related works lack the various works related to model confidence measure and early-stopping in LLM reasoning.

### Questions
1. Why choosing the DAPO17K dataset as the dataset to train against?
2. How will this method perform against non-mathematical datasets?
3. The experiment & Appendix D shows the various traces of problem (with correct vs incorrect) and the change of the score. Statistically, do you observe the answer flipping phenomenon to happen, meaning the retained cases actually contains the right answer (in the trajectory) but then flipped to the wrong one in later stage?

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
The manuscript proposes One-Token Verification (OTV), which uses a LoRA to mode-switch the LLM to switch to the “role” of a verifier. By feedforwarding a special verification token, the model can probe the base model’s KV cache and output a token level confidence score. For training signals, heuristic pseudo-labels are used for confidence trajectories. The authors also propose a method to perform parallel verification, making it possible to score all positions in one pass. On AIME datasets, OTV improves majority-voting/Best-of-N and outperforms Reward Models, and shows best or near-best performance for efficient variants of BoN.

### Strengths
- Does not require a fully separate verifier model, saving memory and compute.
- LoRA instead of full finetuning for the “verifier role” can mitigate catastrophic forgetting on the verifier’s side.
- Efficient parallel verification is possible.

### Weaknesses
- Confidence labels are synthetic and heuristic, which can drift significantly from actual process correctness. Also, as far as I know, the heuristic label scheme cannot model situations where the model succeeds in a major self-correction during its trajectory.
- Although the manuscript briefly claims to justify the fairness of its evaluation protocol, I believe it is unfair. OTV is tuned with DAPO17K for each base backbone model, but other baseline Reward Models are off-the-shelf. Thus, OTV is tailored for each idiosyncratic backbone, while other baselines are fixed to a single backbone. Also, the baseline RMs are built on backbones that are older than Qwen 3, which is the backbone for OTV.
- Experiments are conducted on only a single model family(Qwen 3) and single benchmark family (AIME 24, 25), which is arguably a narrow coverage.

### Questions
- How does OTV fare for the setting where the generator and the verifier are heterogeneous (e.g. OTV is built on Qwen 3 and the generator is built on Gemma 3)?

### Soundness
3

### Presentation
3

### Contribution
3
