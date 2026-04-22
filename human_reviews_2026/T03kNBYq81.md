# RLPR: Extrapolating RLVR to General Domains without Verifiers

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that the LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability signal is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.0 points on TheoremQA and 8.4 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.7 average points across seven benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a "self-reward" strategy for replacing RLVR, using instead the model's own average token log-probability, minus a term for the average token log-probability of the answer without reasoning. A second critical trick is to remove examples for which the standard deviation of this reward is below a running average. In experiments with Qwen2.5, this approach is apparently superior to RLVR trained *with* a verifier, and also apparently superior to a closely-related approach that uses sequence probability instead of average token probability.

### Strengths
- The paper tackles an important problem: removing the need for a verifier could in principle make reasoning models applicable in domains in which it is difficult to construct a reasoner.
- The proposed approach is explained clearly.
- The paper's evaluation are relatively comprehensive, including both mathematical and non-mathematical tasks, and going beyond Qwen to include evluations on Llama and Gemma.
- The paper ablates the main methodological components: debiasing, filtering, and per-token probability.

### Weaknesses
The idea of the paper is closely related to Verifree, which appeared on arXiv in May. While I could imagine that per-token probability could be superior to sequence probability, it also seems relatively easy to hack, by having reasoning traces that insert a lot of high-probability tokens (like function words).

Given this, the empirical comparison with Verifree is critical, but unfortunately it's hard to know what to make of the evaluations in the two papers. In the Verifree paper, the experiments are performed with Qwen3-8b, and it appears that Verifree improves significantly on Qwen3-8b-base, on MMLU-Pro and SuperGPQA. In this paper, all experiments are performed with Qwen2.5-7b, and the replication of Verifree shows no improvement over Qwen2.5-7B-Inst on MMLU-Pro (no evaluation of SuperGPQA is performed). It's hard to know why Verifree appears to be useless here, and I don't feel confident drawing firm conclusions about sequence vs token reward based on these mixed experiments.

### Questions
- Given that the Verifree paper was published in May, would it be possible to conduct comparable experiments on the same datasets and with Qwen3-8B?
- What prevents hacking  the token probability reward by producing reasoning chains that are dominated by high-probability low-content tokens?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RLPR, a verifier-free reinforcement learning framework that uses the model’s own intrinsic probability as a reward signal. Specifically, the method leverages the conditional probability of the ground-truth answer given the model’s CoT output as the reward. This self-probing approach aims to improve performance in domains where external verifiers are unavailable or unreliable. Experiments demonstrate performance improvements across multiple datasets and model families.

### Strengths
1. It introduces a new verifier-free RL method leveraging model probability as intrinsic feedback. Overall it is an interesting idea to use the model's own confidence to evaluate the COT and the final answer to assign reward. I can see it is very promising for the field without an easily verifiable answer. The paper also demonstrated empirically the improvements over the general dataset without an easily verifiable answer. 

2. The experiments are comprehensive enough to cover both general and math datasets, and the model comparisons are sufficient.

3. Probability Reward formulation is significant enough, and it opens a direction for RL fine-tuning where explicit reward models are hard to define.

### Weaknesses
Presentation clarity:
===============
Section 2.2 should have a more clear description. For example, using $z$ and $y$ to denote the response rather than just $o$, then you could use notation $z_1, …, z_n, y_1, …, y_m$. The modified sequence could have more clarity and aligns with previous notations. Otherwise it makes it confusing and might regard the entire sequence as modified. In addition, the $f_seq$ average in l167 should have a more clear definition rather than abusing the notation.

Clarity of the algorithm description:
========================== 
1. In Eq.5, it uses $\hat{r}$ directly for the update but in l260 implementation details, the roll out is 8 responses per prompt. It is unclear which algorithm authors are using (PPO or GRPO). Other than that, both algorithms use advantage $A$ to update the policy, the Eq.5 makes it unclear whether the advantage is directly equal to $ \hat{r}$. It is an important detail and needs proper explanations in the paper

2. Which model are you using for the General-Verifier (L342)? It is unclear to me since no model is defined. 

3. Figure 3 graphics states o1’, o2’ but captions and texts do not reflect that.

4. Section 3.3 is an important quantitative analysis of the probability reward validity, and it is important to expand it with more details. There should be more data and analysis over the 50 prompts selected in order to understand it better. How many responses for each prompt, percentage for correct and incorrect, how many are math/general data.

Overall the presentation should be improved with more clarity.

### Questions
(1) Could the authors report the AUC of  $r’$ (L184 ) to evaluate how well the biased probability aligns with true correctness?

(2) How sensitive is the proposed reward formulation to the base model’s quality? How do other models PR perform in the AUC metric?  

(3) Does the method require CoT reasoning to produce reliable probability rewards, or could it also work well with short-form answers?

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
4

### Summary
This paper introduces RLPR, a verifier-free reinforcement learning method that uses an LLM's intrinsic token probabilities for reference answers as a reward signal to extend RLVR to general domains. Experiments on Qwen2.5-7B, Llama3.1-8B, and Gemma2-2B models show consistent gains on general (e.g., MMLU-Pro, GPQA, TheoremQA) and math benchmarks (e.g., MATH-500, Minerva), outperforming verifier-dependent methods (e.g., General Reasoner) and other verifier-free approaches (e.g., VeriFree).

### Strengths
1. This paper addresses an important research question of RL for general domains. The verifier-free approach is a meaningful step toward broadening RLVR to free-form natural language domains, where rule-based or model-based verifiers are impractical or costly.
2. The results are impressive on paper, with RLPR outperforming strong baselines. It also boosts math performance without math-specific data, suggesting transferability.

### Weaknesses
1. Using token-level probabilities as a reward signal is conceptually close to likelihood-based or entropy-based self-reward methods (e.g., VeriFree)
2. The paper presents a series of empirical engineering techniques (essentially a "bag of tricks") without providing rigorous theoretical justifications
3. Depending on the LLM's intrinsic probabilities as a reward signal introduces circularity and uncertainty. If the base model is biased, or overconfident, this could amplify errors by reinforcing flawed internal evaluations rather than correcting them. Moreover, the method's generalizability to tasks involving longer, sentence-level answers (e.g., in natural reasoning) is not adequately demonstrated (see question 3)
4. Asserting "sheds valuable light on paths to AGI" in the intro is gratuitous and unsubstantiated

### Questions
1. The proposed method seems to have the risk of reward hacking. Have the authors discover any reward hacking in the  trajectories?
2. The debiasing subtracts probs without reasoning. But if the base model already knows the answer, doesn't this penalize good reasoning?
3. Can RLPR generalize to long, compositional, or sentence-level answers, like the answers in natural reasoning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents RLPR, a token probability based reward for extending reasoning to non-verifiable domains. RLPR consists of token probability rewards, reward debiasing, and filtering to enhance model performance across multiple domains.

### Strengths
- The paper evaluates RLPR across multiple tasks using multiple model families.
- Paper is well-written and easy to follow.

### Weaknesses
- Equation 2 seems flawed. Consider two responses "I'm good, not bad" and "I'm bad, not good". Although they are semantically different, Equation 2 seems incapable of distinguishing them in reward. In other words, the token probability reward does not evaluate correctness and is not semantically reliable. 
- Why latent factors are additive decomposable? The authors could provide more intuition behind it.
- The choice of Avg@k seems inconsistent. I understand this is due to the budget concern. But showing consist choice of k will make the results more convincing.
- Can the authors also report training dynamics and efficiency using RLPR?
- Sensitivity analysis to parameters such as $\beta$ and temperature is missing.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
