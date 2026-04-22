# AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Large language models (LLMs), despite possessing latent safety understanding from their vast pretraining data, remain vulnerable to generating harmful content and exhibit issues such as over-refusal and utility degradation after safety alignment. Current safety alignment methods often result in superficial refusal shortcuts or rely on intensive supervision for reasoning-based approaches, failing to fully leverage the model's intrinsic safety self-awareness. We propose \textbf{AlphaAlign}, a simple yet effective pure reinforcement learning (RL) framework with verifiable safety reward designed to incentivize this latent safety awareness through proactive safety reasoning. AlphaAlign employs a dual-reward system: a verifiable safety reward encourages correctly formatted and explicitly justified refusals for harmful queries while penalizing over-refusals, and a normalized helpfulness reward guides high-quality responses to benign inputs. This allows the model to develop proactive safety reasoning capabilities without depending on supervised safety-specific reasoning data. AlphaAlign demonstrates three key advantages: (1) Simplicity and efficiency, requiring only binary prompt safety labels and minimal RL steps for substantial improvements. (2) Breaking the safety-utility trade-off, by enhancing refusal of harmful content and reducing over-refusals, while simultaneously maintaining or even improving general task performance and robustness to unseen jailbreaks. (3) Deep alignment, fostering proactive safety reasoning that generates explicit safety rationales rather than relying on shallow refusal patterns. Our codes are available at \url{https://github.com/zy20031230/AlphaAlign}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AlphaAlign, a RL framework that aims to elicit LLMs’ latent “safety self-awareness” while preserving utility. The method has two reward components: (i) a verifiable safety reward computed with a refusal verifier plus a format verifier that enforces explicit safety reasoning before the answer; and (ii) a normalized helpfulness reward that scores non-refusal responses on benign prompts and discourages over-refusal.

### Strengths
1. The algorithmic design and insights are clear and simple, and the verifiable signals are cheap and reproducible.
2. The empirical study shows strong performance, lowering ASR and reducing over-refusal on CoCoNot. The finding that binary labels and very few RL updates suffice addresses a well-known barrier to scaling safety tuning.

### Weaknesses
1. While the authors present AlphaAlign as an RL framework for safety alignment, the approach appears primarily driven by a structured prompt that elicits safety reasoning, with RL mainly reinforcing this behavior. This likely explains why only a small number of RL updates suffice, the gains seem to stem from the reasoning scaffold rather than heavy RL optimization.
2. The need for explicit safety reasoning during both training and inference raises concerns about general capability and usability. The authors should quantify helpfulness trade-offs and discuss the practicality of such outputs.
3. The method requires labels on question prompts (harmful vs. benign) during training, introducing additional cost.

Minor:

1. Given the RLVR framing, why adopt PPO rather than a critic-free GRPO-style algorithm?
2. What happens if the safety-reasoning tags are omitted at inference? Does the RL training still improve general capability on safety?

### Questions
See weaknesses.

### Soundness
3

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
3

### Summary
This paper proposes AlphaAlign, a reinforcement learning framework designed to cultivate inherent safety awareness in LLMs through active safety reasoning. The authors demonstrate through extensive experiments that AlphaAlign offers three key advantages: simplicity and efficiency, an effective mitigation of the trade-off between safety and utility, and a profound degree of alignment.

### Strengths
The idea of leveraging CoT to enhance LLMs' safety reasoning is novel and promising. The public release of the source code is a commendable practice that significantly facilitates reproducibility and future research in this area.

### Weaknesses
+ The paper lacks convincing evidence to demonstrate that AlphaAlign achieves a deeper alignment beyond a superficial or shallow safety adjustment. 
+ The claim on line 81 that "fewer than 200 RL steps are sufficient to yield substantial improvements" is not sufficiently substantiated. 
+ The design choices for the reward function and the RL loss are confusing.

### Questions
1. The claim that AlphaAlign achieves more than a shallow safety alignment is not fully convincing. The evidence, primarily based on comparing CKAS scores of the initial and aligned Qwen2.5-3B-Instruct models, is insufficient. It remains unclear how this improvement qualitatively differs from the effects of prior alignment methods, such as Beaver [1] or SACPO [2]. A more compelling demonstration of "deep alignment" would involve testing on challenging benchmarks such as XSTest [3], which require nuanced reasoning (e.g., distinguishing between real-world and fictional contexts). Providing such comparative analyses and evaluations is crucial to substantiate the alleged advantage of AlphaAlign.

2. The experimental validation appears to be limited to 3B and 7B scale models. Consequently, the generalizability of the claim that "fewer than 200 RL steps are sufficient" may be premature, as the optimal number of RL steps could vary significantly with model scale. Verification on larger-scale models would strengthen this finding.

3. I note several similarities between the design of AlphaAlign and existing methods. The reward formulation, which includes a format reward and a rule-based reward, appears conceptually similar to the approach in DeepSeek-R1 [4]. Furthermore, the use of a normalized helpfulness reward closely mirrors the design in GRPO. Given these similarities, the core methodological innovation requires clearer articulation. Specifically, if the reward $\tilde{r}_i$ is defined as in GRPO, it already functions as an advantage estimate. The subsequent use of GAE to compute advantages then seems redundant. A clarification of the design intuition behind this specific choice—and how it differs from or improves upon GRPO—is necessary to fully appreciate the contribution of AlphaAlign.

4. There might be a typo: The $r_a$ in equation (5) appears to be undefined. Based on the preceding paragraph, it seems this should likely be $r_s$? Please clarify or correct this for consistency.

[1] Dai, Josef, et al. "Safe rlhf: Safe reinforcement learning from human feedback." arXiv preprint arXiv:2310.12773 (2023).

[2] Wachi, Akifumi, et al. "Stepwise alignment for constrained language model policy optimization." Advances in Neural Information Processing Systems 37 (2024): 104471-104520.

[3] Röttger, Paul, et al. "Xstest: A test suite for identifying exaggerated safety behaviours in large language models." arXiv preprint arXiv:2308.01263 (2023).

[4] Guo, Daya, et al. "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning." arXiv preprint arXiv:2501.12948 (2025).

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
The paper proposes AlphaAlign, a reinforcement learning (RL) framework for LLM safety alignment. The authors argue that existing methods either lead to superficial refusal shortcuts or rely on expensive, supervised safety-reasoning datasets. AlphaAlign aims to incentivize the model's latent safety self-awareness using a pure RL approach. The core of the method is a dual-reward system: (1) a verifiable safety reward that encourages structured, reasoned refusals for harmful prompts and penalizes over-refusals, based only on binary prompt labels and rule-based verifiers; and (2) a normalized helpfulness reward, derived from a standard reward model, to maintain utility on benign queries. The paper claims this simple framework can improve safety (including robustness to jailbreaks) and reduce over-refusal, thereby breaking the common safety-utility trade-off, all without requiring supervised safety-reasoning data.

### Strengths
- The primary strength of this paper is its simplicity and novelty. The proposal to use a pure RL framework with a simple, verifiable reward system—bypassing the need for expensive supervised safety-reasoning datasets—is a valuable contribution to the alignment field. The data-efficiency (requiring only binary prompt labels) is a significant practical advantage.

- The method directly addresses the critical safety-utility trade-off. The dual-reward system is well-motivated, and the experimental results, particularly in Table 1 and 2, suggest that AlphaAlign can simultaneously reduce attack success rates (ASR) and improve or maintain utility on benchmarks like AlpacaEval and GSM8K, which is a non-trivial claim.

- The framework's efficiency is compelling. The authors report substantial improvements within a small number of RL steps (fewer than 200, as stated in the abstract), which supports their hypothesis that the method is incentivizing latent safety awareness rather than teaching a new, complex behavior from scratch.

### Weaknesses
- **The reliance on a simple, rule-based refusal verifier ($V_r$)** As described in Appendix B.1 and mentioned on Line 207, $V_r$ appears to be a simple string-matching function. This is brittle and prone to both false positives and false negatives. The paper needs empirical validation of this verifier's accuracy and its impact on the diversity of refusal responses.

- **The formalism of the reward functions lacks clarity, hindering reproducibility.** In Equation 5, the safety reward $R_s$ is defined using reward values $r_f$ and $r_a$. However, the paper never specifies what these scalar values are.

- **The utility reward mechanism (Eq 6) introduces potential instability.** The helpfulness scores are normalized using the mean and standard deviation of a small number of rollouts (n=8, per Line 986). With such a small batch, the sample mean and std are high-variance estimators, which will in turn make the normalized reward signal $\tilde{r}_i$ very noisy. This can destabilize the PPO training. The authors do not discuss or analyze this potential instability.

### Questions
There appears to be a notational inconsistency. Line 211 and Equation 4 mention a safety reward $r_s$. However, Equation 5 then defines a seemingly different reward $R_s(x, o_i)$. Could the authors clarify the relationship between $r_s$ and $R_s$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AlphaAlign, a novel safety alignment framework that leverages LLM's inherent safety understanding through simplified reinforcement learning. Rather than relying on complex supervised learning or extensive safety-specific datasets, AlphaAlign uses a dual-reward system to incentivize models to engage their latent safety awareness through proactive reasoning.

### Strengths
Originality: The paper presents a fundamentally original approach by reframing safety alignment from an external constraint problem to an internal capability activation problem. The adaptation of Reinforcement Learning with Verifiable Rewards (RLVR) to safety alignment is highly creative. 
Quality: The evaluation is rigorous and comprehensive, encompassing multiple model architectures, diverse safety benchmarks, utility preservation metrics, and thorough ablation studies.
Clarity: The paper offers clear technical exposition, complemented by an accessible high-level framework overview and detailed case studies.
Significance: The empirical success of this approach lends support to the hypothesis that LLMs possess substantial latent safety understanding. This finding carries broader implications for AI alignment research and advances our understanding of emergent capabilities in large language models.

### Weaknesses
1）Evaluation Metric Limitation
- Heavy emphasis on ASR may miss other important safety dimensions, such as consistency between reasoning and the final answer.
- The model is rewarded for generating these exact phrases during training, then evaluated based on detecting these same phrases. This doesn't validate genuine safety understanding. 

2) Insufficient Analysis of "Deep Alignment"
The paper claims to "incentivize safety awareness" but doesn't clearly distinguish this from simply training on binary labels (e.g., sft, standard RLHF). The paper needs to:
- Provide ablations showing that the RL framework is necessary versus simpler alternatives
- Demonstrate what aspects of "safety awareness" emerge that couldn't be achieved through other methods.

3) Computational Efficiency
- The paper claims efficiency with "fewer than 200 RL steps" but provides no computational cost comparisons (GPU hours, wall-clock time, FLOPs) against baselines. Each RL step requires 8 rollouts per prompt, likely making it more expensive than SFT methods, yet this cost difference is never quantified.
- Missing inference overhead analysis: The mandatory <safety_reasoning> generation before every response increases token count and latency, but no measurements are provided.

### Questions
1. The application of CKAR to capture genuine safety understanding lacks sufficient justification, as the use of certain keywords may be artificially amplified through training rather than reflecting authentic alignment. Empirical support—such as case studies or quantitative analyses demonstrating a clear correlation between CKAR frequency and deep safety alignment, or comparison with other fine-tuning methods—would strengthen the validity of this approach. The selection of keywords also requires justification.
2. Table 2 shows Llama3.2-3B has 8.3% drop on GSM8K. The paper dismisses this, but this is a substantial degradation that contradicts the "maintaining utility" claim.

### Soundness
3

### Presentation
3

### Contribution
4
