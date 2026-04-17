# The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key method for improving Large Language Models' reasoning capabilities, yet recent evidence suggests it may paradoxically shrink the reasoning boundary rather than expand it. This paper investigates this shrinkage issue of RLVR by analyzing its learning dynamics and reveals two critical phenomena that explain this failure. First, we expose the negative interference phenomenon in RLVR, where learning to solve certain training problems actively reduces the likelihood of correct solutions for others, leading to the decline of Pass@$k$ performance or the probability of generating a correct solution within $k$ attempts. Second, we uncover the winner-take-all effect: RLVR disproportionately reinforces problems with high likelihood under the base model while suppressing initially low-likelihood, correct solutions. Through extensive theoretical and empirical analysis on multiple mathematical reasoning benchmarks, we show that this effect arises from the inherent on-policy sampling in standard RL objectives, causing the model to converge toward narrow solution strategies. Based on these insights, we propose a simple yet effective data curation algorithm that focuses RLVR learning on low-likelihood problems, achieving notable improvement in Pass@$k$ performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
1. They show the phenomena of negative interference where improving likelihood of correct responses on some prompts reduces likelihood of correct responses on other prompts. This leads to reduction in pass@k performance during RLVR
2. They show that RLVR disproportionately reinforces problems which have high likelihood correct solutions under the base models while suppressing others.
3. Using their insights, they provide a data curation algorithm that helps achieve higher pass@k performance.

### Strengths
The problem is quite important and it's important to theoretically understand the root of the issue. The paper tries to look at it from a theoretical and empirical perspective. The paper is presented very well and makes the correct observation that pass@k goes down because pass@1 goes down for certain problems while doing RLVR.

### Weaknesses
The analysis done in "Why does RLVR reinforce problems with high-likelihood correct solutions in the base model" from line 342 is erroneous. 

1. Low-likelihood tokens provide less meaningful updates:

This is not followed from equation 8 because the gradient's value is a function of the network parameters as well. Also, "less meaningful" is hand wavy. What is exactly meant by less meaningful? Low gradient norm? 

2. When there are multiple optimal actions....

This is also not followed from equation 8 because it doesn't take into account the value of the gradient.

### Questions
Could the authors comment on the weaknesses and clearly describe any assumptions they've made in section mentioned above?

### Soundness
1

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies why Reinforcement Learning with Verifiable Rewards (RLVR) can sometimes shrink rather than expand reasoning capability in large language models. Through a detailed theoretical and empirical investigation, the authors identify two core failure modes:
(1) Negative interference, where learning to solve some problems reduces the likelihood of solving others, and
(2) Winner-take-all reinforcement, where RLVR disproportionately strengthens problems already solvable by the base model while suppressing low-likelihood ones. They formalize these effects via a per-step influence analysis, empirically validate them on multiple math benchmarks (AIME 2024/25, Math500, Minerva), and propose Selective Examples with Low-likelihood and Forward-KL (SELF), a data-curation algorithm that focuses learning on low-success-rate examples. SELF mitigates coverage shrinkage and improves Pass@k consistency without extra computational cost.

### Strengths
1. The per-step influence formulation provides an insightful view of how on-policy RLVR gradients cause cross-problem interference and loss of coverage. This formalism connects RLVR behavior to known interference and plasticity-loss phenomena in RL
2. The proposed method SELF is conceptually simple (i.e., curating low-likelihood examples and using forward-KL regularization) yet effectively improves coverage and diversity (entropy, trust-region violations). 
3. The experiments are thorough, spanning three model families and four reasoning benchmarks, showing consistent Pass@k degradation under standard RLVR and improvement with SELF. The correlation analyses between interference metrics and Pass@k drop are particularly strong and interpretable

### Weaknesses
1. Additional exploration beyond math reasoning would further strengthen the work. All benchmarks are math-centric, it remains unclear whether the observed interference and SELF’s benefits generalize to code/commonsense reasoning tasks.
2. Missing discussion of potential trade-offs. SELF improves Pass@k but sometimes slightly reduces Pass@1. A deeper analysis of why emphasizing low-likelihood problems doesn’t over-regularize high-confidence ones would strengthen understanding of its limits.

### Questions
- Would the negative interference also appears in non-verifiable/non-math RL scenarioes?

- How sensitive is SELF to the accuracy of low-likelihood identification? For example, does mis-labeling easy problems as “low-likelihood” hurt stability?

- Regarding the training stability, was mixed-precision training used (bf16 or fp16)? Given that RLVR often face numerical precision issues, it would be helpful to clarify whether the reported instability of baselines (e.g., GRPO/W-REINFORCE) arises intrinsically from the algorithms or from precision-mode effects. Would GRPO/W-REINFORCE still suffer from training divergence if training in fp16?

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
3

### Summary
This paper investigates how RLVR affects reasoning boundary of LLMs, measured by pass@k performance. The authors identify two phenomena: 1. negative interference, where learning to solve certain problems reduces likelihood of others, and 2. winner-take-all, where RLVR disproportionately reinforces problems with high-likelihood correct solutions while suppressing low-likelihood ones. The proposed method SELF, which focuses on low-likelihood problems and uses forward KL regularization, proves to be effective in improving LLM reasoning while preserve base model's pass@k performance.

### Strengths
1. The per-step influence provides a principled analysis to understand how updates on one problem affect others, which is comprehensive and well supported by the experiments in section 4.2.
2. The proposed method SELF is simple to implement and shows consistent improvements in pass@k across benchmarks while maintaining comparable pass@1 performance.

### Weaknesses
1. The main finding of this paper is well studied in many priors works, as also mentioned in the paper, therefore weakening the novelty.
2. The paper does not provide sufficient justification for why filtering based on greedy response failure is an appropriate or effective way to identify low-likelihood problems. Other potential strategies for targeting such problems could be explored. Moreover, the proposed filtering approach may not generalize well to models whose optimal decoding strategy is sampling-based rather than greedy, such as Qwen3 Thinking models.

### Questions
1. The proposed SELF objective is intended to train only on problems where the greedy response fails. However, Eq. (9) appears to optimize examples where the greedy answers ( $y^\*$ ) are correct, since the indicator function ( $\mathbf{1}(r(x, y^\*) \in C(x))$ ) selects successful responses. Could the authors clarify this discrepancy?
2. Are the improvements reported in Table 1 statistically significant? For example, are they at least beyond the 2-sigma or 95% confidence level?

### Soundness
3

### Presentation
3

### Contribution
2
