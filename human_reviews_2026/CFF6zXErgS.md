# GTPO AND GRPO-S: TOKEN AND SEQUENCE-LEVEL REWARD SHAPING WITH POLICY ENTROPY

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Reinforcement learning (RL) is a pivotal task for enhancing Large Language Model (LLM) reasoning. Conventional algorithms, however, typically adhere to a coarse-grained credit assignment paradigm, applying a uniform reward to all tokens in a sequence—a critical flaw in long-chain reasoning tasks. In this paper, we address this challenge and propose Dynamic Entropy Weighting, a novel mechanism that facilitates fine-grained rewards through two new algorithms: Group Token Policy Optimization (GTPO), which assigns an entropy-weighted reward to each token, and the analogous algorithm Sequence-Level GRPO (GRPO-S). Our approach is founded on the hypothesis that high policy entropy within a reasoning path is a powerful heuristic for "cognitive effort" at pivotal junctures, which can be repurposed into a learning signal. By repurposing policy entropy for reward shaping, we achieve true per-token credit assignment. Experimental results across challenging reasoning benchmarks validate the superiority of our approach, showing our methods significantly outperform a strong DAPO baseline and confirming our entropy-weighting mechanism as the key driver of this performance boost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tries to addresses the coarse-grained reward assignment challenge in reasoning by leveraging policy entropy to redistribute reward signals. The authors operate on the central hypothesis that the moments of high policy entropy within a reasoning sequence are not random noise but strong correlates of pivotal reasoning exploration. To exploit this, they propose the Dynamic Entropy Weighting framework, which reshapes rewards to be proportional to token-level or sequence-level entropy. This framework instantiates two methods: GTPO, derived from token-level entropy, and GRPO-S, based on sequence-level entropy. The methods are evaluated on the AIME 2024 and AIME 2025 mathematical reasoning benchmarks.

### Strengths
- This paper is generally well-written and easy to follow.

- This paper tackles the important challenge of reward sparsity in RLVR scenario. The core idea of redistributing sequence-level rewards to token-level actions based on action entropy in successful sequences is intuitively sound.

- Unlike previous dense reward modeling techniques that rely on model-based rewards and risk reward hacking, GTPO offers a more robust alternative. It derives dense rewards directly from rule-based verification, which can help mitigate this risk.

### Weaknesses
- **Need for Stronger Justification of the Core Hypothesis**

The proposed method is based on the hypothesis that "high-entropy moments within a reasoning sequence are signatures of pivotal junctures, not noise." The paper would be strengthened by a more rigorous justification for this claim. For example, it would be beneficial to discuss the empirical conditions under which this hypothesis holds and the situations in which it might fail.

- **Concern Regarding the Penalization of Confident Tokens**

The motivation for penalizing low entropy (or confident) tokens in incorrect trajectories requires further justification. As highlighted by prior work [1], the most confident tokens in a sequence (including unsuccessful one) are often related to syntactically correct tokens (e.g., a LaTeX equation: the token "{" after "\boxed" in ["\boxed", "{"]). It is questionable whether penalizing such tokens is meaningful, as their high confidence is tied to grammatical correctness rather than reasoning error. The authors should clarify how their method distinguishes between harmful overconfidence and valid syntactic certainty.

[1] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning, arxiv 2025.

- **Lack of Clarity in Theoretical Guarantees (Section 2.4.2)**

The theoretical derivation in Section 2.4.2, particularly the approximations at lines 308 and 313, is confusing. The equations do not explicitly provide theoretical bounds (upper or lower) for these approximations, making it difficult to assess their validity and impact. Providing more explicit and rigorous theoretical evidence would be beneficial.

- **Limited Scope of Evaluation**

The empirical evaluation is constrained to only two mathematical reasoning benchmarks (AIME 2024/2025). This narrow scope raises concerns about the robustness and generalizability of the proposed methods. Evaluation on a wider range of tasks (code generation, commonsense reasoning, or instruction following) is necessary.

- **Missing Experimental Details for Training Data**

### Questions
- The captions for Figures 1 and 2 lack key details, making it difficult to distinguish between high and low-entropy tokens/sequences or to identify their corresponding reward values.

- Typo: at line 260, $\tilde{r}\_{i,t}$  $\to$ $\tilde{r}^+\_{i,t}$?

### Soundness
3

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
This paper studies the sparse reward challenge in reinforcement learning when training large language models. To tackle this, the authors proposed Group Token Policy Optimization (GTPO), which assigns an entropy-weighted reward to each token, and the analogous algorithm Sequence-Level GRPO (GRPO-S).

### Strengths
1. The method is well motivated and easy to understand or implement.
2. The experiments show a clear advantage over GRPO and DAPO.
3. The authors discussed future works and limitations of the paper in detail.

### Weaknesses
1. The mathematical derivation of why we have Equations 3 and 5 is unclear. Why does this give us a better policy gradient in theory?
2. The authors tried to give some theoretical guarantees in the paper. However, all the results are in the appendix. I'm expecting at least the theorems themselves to appear in the main text. Besides, it is unclear to me why the proposed PG is unbiased. The 'approximately equal to' statement is not a formal statement that can act as a theoretical guarantee.
3. It is also unclear to me why adding entropy reduces the PG variance. I'm even more confused after reading Appendix C.2. It seems to show that directly computing the mean of all samples enjoys a smaller variance than 'First Compute Subgroup Means, Then Average'. What is this conclusion's relationship with the Dynamic Entropy Weighting method?
4. Can the improvements over GRPO/DAPO in the experiments be attributed to the different averaging method described above, instead of coming from the entropy design?
5. The novelty is also limited. Entropy is known to be an important metric for RLVR. The specific entropy design might be new, but due to the lack of a clear math motivation, why we need this design instead of other entropy-related designs is unclear.

Minor: Figure 1 is uninformative. What are the orange bars? We also cannot tell the differences between (c) and (d) from the figures themselves.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses a central limitation in reinforcement learning (RL) for large language model (LLM) reasoning—coarse-grained credit assignment, where a uniform reward is assigned to all tokens in a generated sequence. The authors propose Dynamic Entropy Weighting, a novel mechanism that uses policy entropy as a signal of “cognitive effort,” repurposing it for fine-grained reward shaping. Two new algorithms are introduced: Group Token Policy Optimization (GTPO), which performs entropy-weighted reward assignment at the token level, and Sequence-Level Group Relative Policy Optimization (GRPO-S), which applies entropy-based modulation at the sequence level. Both methods are designed as value-function-free extensions to the GRPO framework, maintaining convergence guarantees while reducing gradient variance. Experimental results on AIME 2024 and 2025 benchmarks using Qwen2.5 models demonstrate significant performance gains over DAPO and GRPO baselines, especially in small model settings.

### Strengths
- The proposed solution is simple yet effective.
- The paper is easy to follow.

### Weaknesses
- The major concern is the experimental evaluation. The experiments are only conducted on two Qwen2.5 series models and AIME benchmarks only. The authors are suggested to conduct more experiments on other base models (e.g., Llama and DeepSeek-R1-Distill series, etc.) and other benchmarks (e.g., AMC23, Minerva Math, OlympiadBench, LiveCodeBench) to validate the effectiveness and generalization of the proposed methods.
- The proposed methods introduce four hyperparameters (i.e., $\alpha_1$, $\alpha_2$, $\beta_1$ and $\beta_2$), which make it difficult to tune the methods in practice.
- The authors are suggested to provide more analysis on the sensitivity of these hyperparameters. The authors only conduct ablation study using three sets of these hyperparameters. However, there are still many other combinations of these hyperparameters. It is unclear how sensitive the proposed methods are to these hyperparameters, e.g., what if $\alpha_2$ and $\beta_2$ are set to 0.05.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
2

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
The paper addresses the coarse-grained credit assignment problem in reinforcement learning for large language models, where existing methods (e.g. GRPO) give the same reward to every token based only on final outcome. To enable fine-grained credit assignment in long-chain reasoning tasks, the authors introduce Dynamic Entropy Weighting, which uses a model’s policy entropy as a proxy for cognitive effort to shape rewards at a finer level. Two algorithms are proposed: Group Token Policy Optimization (GTPO), assigning an entropy-weighted reward to each token, and Sequence-Level GRPO (GRPO-S), scaling an entire sequence’s reward by its average entropy. The paper formalizes coarse credit assignment as a bottleneck and hypothesizes that high-entropy decisions indicate pivotal reasoning steps to be reinforced. A theoretical analysis shows the new reward shaping preserves the policy’s optimality while reducing variance. Experiments on mathematical reasoning benchmarks (AIME 2024/2025) demonstrate that GTPO and GRPO-S significantly outperform baselines (DAPO and GRPO), achieving higher Pass@k and mean scores across model scales. Key contributions include introducing per-token reward shaping via entropy, a sequence-level variant for efficiency, and empirical validation of improved learning signals for complex reasoning.

### Strengths
1. The paper clearly identifies coarse-grained credit assignment as a fundamental limitation in current RL fine-tuning of LLMs.
2. Instead of treating policy entropy as mere “uncertainty,” the paper repurposes it as a proxy for cognitive effort, rewarding high-entropy decisions in correct answers and penalizing overconfident errors.

### Weaknesses
1. The paper discussion of related literature is limited in scope, potentially missing important context. Many earlier works have attempted token-level rewards or alternative credit assignment heuristics, the paper should contrast with them. Besides, given the focus on entropy, one might expect references to prior uses of entropy or uncertainty in exploration or credit assignment.
2. Some aspects of the technical presentation suffer from notation inconsistencies or ambiguity, which could confuse readers. 
* For example, the formula for the shaped negative sequence reward uses $|o_j|$ in the product $\prod_{k=1}^{|o_j|}(1/\hat{H}_k)^{1/|o_j|}$, whereas in the original definition (Eq. (6) in Sec. 2.3) the normalization for negative sequences was over the count of unsuccessful sequences $m$. This suggests a likely typo or misuse of $|o_j|$ (sequence length) where $m$ (number of sequences) was intended, but the text does not clarify it. Such a notation error can be misleading, as it conflates two different concepts (sequence length vs. batch size of negative examples). 
* Similarly, in the token-level formula,  $\tilde{r}{j,t}^{-}$ uses a normalization over $\sum{k=1}^{m}(1/H_{k,t})$ in Eq. (3), but the geometric mean version in Sec. 2.4 writes $\prod_{k=1}^{|o_j|}1/H_{k,t}$ , again inconsistent in indexing. These minor inconsistencies indicate that the math notation was not thoroughly proofread for consistency with the earlier definitions. Readers might have to infer the intended meaning (e.g., assume the product is actually over all $m$ negative sequences at timestep $t$) which disrupts clarity.
3. The conservation of total reward assumption (Eq. 9) and, by extension, all subsequent theoretical claims that depend on it. This includes the claim that the expected mean reward is unchanged (Eq. 10), the claim that the expected advantage is preserved (Section 2.4.2, Para 4), and the final conclusion that the expected policy gradient direction is preserved (Section 2.4.2, Para 5). The theory presented does not apply to the method that was experimentally evaluated.

### Questions
1. Which prior works on token-level or segment-level credit assignment have you considered, and how does your approach differ from or improve upon them?
2. How consistently does high policy entropy truly correspond to “pivotal” reasoning steps that deserve more credit?
3. Do you expect GTPO/GRPO-S to help in other domains beyond math reasoning?
4. GTPO in particular introduces per-token calculations. How much slower or more memory-intensive was training with GTPO compared to GRPO? Is this method feasible for very long sequences (since entropy must be computed at each position)? Also, did you encounter any training instabilities when combining PPO with these dynamically changing rewards?

### Soundness
2

### Presentation
2

### Contribution
2
