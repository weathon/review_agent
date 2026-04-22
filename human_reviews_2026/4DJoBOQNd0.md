# Linking Process to Outcome: Conditional Reward Modeling for LLM Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Process Reward Models (PRMs) have emerged as a promising approach to enhance the reasoning capabilities of large language models (LLMs) by guiding their step-by-step reasoning toward a final answer.
However, existing PRMs either treat each reasoning step in isolation, failing to capture inter-step dependencies, or struggle to align process rewards with the final outcome. 
Consequently, the reward signal fails to respect temporal causality in sequential reasoning and faces ambiguous credit assignment.
These limitations make downstream models vulnerable to reward hacking and lead to suboptimal performance.
In this work, we propose Conditional Reward Modeling (CRM) that frames LLM reasoning as a temporal process leading to a correct answer.
The reward of each reasoning step is not only conditioned on the preceding steps but also explicitly linked to the final outcome of the reasoning trajectory. By enforcing conditional probability rules, our design captures the causal relationships among reasoning steps, with the link to the outcome allowing precise attribution of each intermediate step, thereby resolving credit assignment ambiguity.
Further, through this consistent probabilistic modeling, the rewards produced by CRM enable more reliable cross-sample comparison.
Experiments across Best-of-N sampling, beam search and reinforcement learning demonstrate that CRM consistently outperforms existing reward models, offering a principled framework for enhancing LLM reasoning.
In particular, CRM is more robust to reward hacking and delivers stable downstream improvements without relying on verifiable rewards derived from ground truth.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new theoretical framework with Potential-Based Reward Shaping to train PRMs.  Their framework not only captures the dependencies between reasoning steps, but also between the reasoning steps and the final outcome. The empirical studies also validate the effectiveness of their frameworks.

### Strengths
- The intuition behind the framework is well-motivated and easy to follow.
- The empirical studies are comprehensive, including BoN, PRM-guided beam-search, and RL experiments. The cross-sample comparability appears appealing.

### Weaknesses
> Ambiguity of why choose $\log S(t)$ as the potential function.

The authors directly claim "A natural choice for the potential function is one that estimates the likelihood of eventually reaching a correct answer from the current state", but abridge the motivations and insights behind this substitution. Since this substitution is the key step to derive the final objective, the ambiguity here confuses.

### Questions
> Implementation details on how to combine CRM + VR would enhance the reproducibility.

> Finding 3 (Line 466) claims CRM exhibits a high data efficiency, but without comparison with previous works. Does CRM exhibits a higher  or on-par data efficiency compared to previous works?

> Confused about what each peak of CRM in Figure 1 means? It cannot show your CRM can capture dependency between reasoning steps and outcome.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Conditional Reward Modeling (CRM) — a framework for enhancing the reasoning capability of large language models by modeling the reasoning trajectory as a temporal probabilistic process. CRM explicitly links intermediate reasoning steps to the final outcome by defining step-level rewards as conditional probabilities of maintaining correct reasoning. This design aims to resolve credit assignment ambiguity and improve robustness against reward hacking.
Empirical results across Best-of-N sampling, beam search, and reinforcement learning (RL) demonstrate that CRM outperforms baseline reward models (PRM, PQM, IPRM) in reasoning tasks such as GSM-Plus, MATH500, and AIME benchmarks. The method shows stable optimization and increased self-reflection behaviors in RL-trained models.

### Strengths
1.	Formal rigor — CRM introduces a clear probabilistic derivation linking reasoning steps to final outcomes, which elegantly resolves step-level credit ambiguity.
	2.	Empirical robustness — Demonstrates consistent performance gains across multiple reasoning benchmarks and settings (Best-of-N, beam search, RL).
	3.	Reward hacking analysis — The paper includes a thoughtful investigation into degenerate reward optimization and shows CRM’s improved stability.
	4.	Data efficiency — The ablation on L_z shows CRM achieves high data utilization, improving with limited supervision.

### Weaknesses
1.	Flawed motivation — The key claim that verifiable rewards are costly and difficult to scale is unconvincing and outdated.
	- Recent large-scale reasoning systems such as DeepSeek-R1 have demonstrated that verifiable rewards are both accurate and scalable, making CRM’s justification inconsistent with current research trends.
	- Moreover, DeepSeek-R1 explicitly abandoned PRM-style process rewards for their instability, undermining the central motivation of this paper.
	2.	Lack of comparison to verifiable-reward models —
	- Although CRM shows improvement over PRM-family models (PRM, PQM, IPRM), the experiments exclude comparisons with contemporary verifier-based RL systems like Qwen3-RL, DeepSeek-R1, or OpenAI o1.
	- Without these baselines, it’s difficult to assess whether CRM’s improvements are competitive in modern reasoning settings.
	3.	Limited scalability and applicability —
	- CRM is only tested on LLM-based reasoning (Best-of-N, beam search, RL) but not on large reasoning models (LRMs), where process reasoning involves extended multi-branch trajectories and hierarchical dependencies.
	- The proposed CRM-based search mechanism may also introduce additional computational cost compared to verifier-based approaches, which is not analyzed or mitigated.
	4.	Incremental novelty —
	- The use of conditional probability chains and potential-based reward shaping is mathematically elegant but conceptually derivative of standard PRM extensions (e.g., IPRM, PQM).
	- The claimed robustness advantage could likely be replicated by integrating verifiable signals with existing PRMs.

### Questions
1.	How does CRM compare quantitatively to DeepSeek-R1 or Qwen3-RL in reasoning benchmarks (e.g., AIME24, GAIA)?
	2.	What is the computational overhead introduced by CRM’s step-level conditional modeling during beam search or RL?
	3.	Could CRM be integrated with verifiable rewards rather than replacing them, to combine interpretability with scalability?
	4.	Have you attempted to apply CRM to large reasoning models or long-horizon tasks beyond math reasoning (e.g., proof or planning tasks)?

### Soundness
2

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
This paper proposes Conditional Reward Modeling (CRM) that frames LLM reasoning as a temporal process leading to a correct answer. The core idea is to model the reasoning process as a conditional probability process, associating the reward of each step with the preceding tokens and the final result. The author conducted experiments on various applications of their reward model, in most of which they showed good performance. The cost of this article is also acceptable, and the experimental comparison is fair from this perspective. The current improvement in the reward model mostly comes from the reasoning reward model, which is easily hacked, or training data scaling. I think this is a good paper that should be accepted as a poster.

### Strengths
1. The paper is well written, and the proposed method is reasonable.
2. The experiment is sufficient to demonstrate the effectiveness of the method.
3. The experiment can prove that the reward learned by the method is robust and is not easily hacked.

### Weaknesses
The author can try to expand the method to more fields, such as code, or even fields without verifiers.

### Questions
1. The results in Figure 4 show the behavior of reward hacking. But the reviewer has conducted a similar experiment using Math-Shepherd, and although the effect was not good, the AIME2024 accuracy did not decrease to 0. Is there a problem with this part of the experiment? That is, the author obtained a good training hyperparameter on CRM and directly used it on other baselines, which require different hyperparameters.
2. In the BoN exps, the authors use the score S(T) for CRM and take the min value for other models. What would happen if other models also use S(T)?

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
3

### Summary
The paper introduces Conditional Reward Modeling (CRM) for multi-step reasoning, aligning step-level rewards with the final outcome through a principled probabilistic framing centered on the “first error” in a chain of thought. This yields a single, comparable trajectory score that supports better sample selection (Best-of-N), stronger beam-search guidance, and more stable RL training with reduced reward hacking, while remaining complementary to verifiable outcome rewards when available. Experiments on math reasoning benchmarks and multiple backbones show consistent gains over PRM/PQM/IPRM/ORM baselines and clearer localization of the first wrong step. Ablations indicate robustness to limited step-level supervision and to reward-shaping choices, suggesting a simple and practical framework for outcome-aligned process rewards.

### Strengths
The paper poses an interesting problem and ties the method well to practical RL applications.

### Weaknesses
The experiments do not appear to discuss the latest PRM (including trainable PRM) designs. [1][2]

CRM seems to converge with small n; please add Major@N and Pass@N baselines to Table 1 to contextualize performance and the gap to an upper bound.

Table 3 should include an RLVR baseline.

[1] R-PRM: Reasoning-Driven Process Reward Modeling

[2] GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3
