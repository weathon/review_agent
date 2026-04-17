# UpSkill: Mutual Information Skill learning for Structured Response Diversity in LLMs

- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of large language models (LLMs) on mathematics and programming tasks, often by maximizing *pass@1* correctness. However, optimizing single-attempt accuracy can inadvertently suppress response diversity across repeated attempts, narrowing exploration and overlooking underrepresented strategies. We introduce UpSkill, a training time method that adapts *Mutual Information Skill Learning* (MISL) to LLMs
to induce *structured response diversity*: a discrete latent $z$ selects a reproducible ``strategy" that steers the token distribution toward distinct modes. We propose a novel reward that we implement within Group Relative Policy Optimization (GRPO): a *token-level* mutual information (MI) reward that encourages trajectory specificity to $z$. Experiments on GSM8K with three open-weight models, Llama 3.1--8B, Qwen 2.5-7B, and R1-Distilled Qwen2.5-Math-1.5B show that UpSkill improves multi-attempt metrics, yielding median gains of $\sim$4\% in pass@k and $\sim$7\% in consensus@k without degrading pass@1. Additionally, we show theoretically that mutual information is closely tied to pass@k, providing a theoretical justification for UpSkill.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the critical issue of "diversity collapse" in LLMs, where standard reinforcement learning (RLVR) techniques aimed at optimizing single-attempt accuracy (pass@1) inadvertently reduce response variety, thereby limiting multi-attempt (pass@k) performance. The authors introduce UpSkill, a novel training-time method that integrates Mutual Information Skill Learning (MISL) into the GRPO framework. This is achieved by conditioning the model on a discrete latent "strategy" variable $z$ and introducing a token-level mutual information reward ($r_{TMI}$). This reward incentivizes the model to produce distinct, reproducible strategies for each $z$. Experiments demonstrate the method's effectiveness, starting with a controlled arithmetic environment where UpSkill prevents diversity collapse and significantly boosts pass@5, and extending to the GSM8K benchmark, where it achieves median gains in pass@k and consensus@k across several models without degrading pass@1 accuracy.

### Strengths
Basically, in my opinion, this paper's primary strength lies in its principled approach, tightly coupling the proposed method with a strong theoretical justification. 
The authors provide a formal proof (Lemma 1) that directly links the mutual information objective $\mathcal{I}(\tau;z|x)$ to a bound on pass@k improvements, moving the work beyond simple empirical heuristics. 
Moreover, the theoretical proof is well-supported by the empirical results. The "arithmetic environment" (Section 5.1, Figure 4) serves as an good controlled experiment, clearly visualizing how the baseline model's diversity completely collapses (pass@1 converges to pass@5) while UpSkill maintains a significant performance gap, proving the concept. The method addresses a good implement potential, showing that on the complex GSM8K task, it is possible to enhance multi-attempt metrics (pass@k, consensus@k) without the common side effect of sacrificing single-attempt (pass@1) performance.

### Weaknesses
Basically, I think the experiments and analysis should be more sufficient and solid.
1） The experiment results for the Llama 3.1-8B model is decently negative and seems negative, where pass@k performance decreased by 2% (from 88% to 86%). The lack of discussion on this contradictory finding in the main text undermines the method's claimed robustness, especially on high-capability models.

2）The presentation of the main GSM8K results is incomplete. The specific value of $k$ used for the pass@k, plurality@k, and consensus@k metrics is not explicitly stated in the main body of the paper, making the magnitude of the improvements difficult to interpret.

3) The main results chart (Figure 5) is missing a numerical label for the Llama 3.1-8B pass@1 "with MI" condition, which prevents a full and clear comparison of this key metric on the strongest baseline model.

### Questions
1) Could the authors elaborate on the negative pass@k result for Llama 3.1-8B? Does this finding suggest a fundamental limitation of UpSkill when applied to stronger base models, perhaps aligning with the findings in Appendix F, or is it potentially an artifact of hyperparameter selection?

2) What is the computational overhead of the token-level MI reward (Equation 5) during training? Does calculating the mixture distribution $p_{\pi}(y_t|x,y_{<t})$ require N separate forward passes, and if so, how does this impact training efficiency?

3) Given that other contemporary work (e.g., Chen et al., 2025) explores the direct optimization of the pass@k objective, what is the primary advantage of using the mutual information approach? Does UpSkill, for instance, produce more semantically distinct or interpretable strategies than direct pass@k optimization would?

### Soundness
3

### Presentation
2

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
The paper introduces UpSkill, a method to enhance "structured response diversity" in LLMs through Mutual Information Skill Learning (MISL). By conditioning on a discrete latent variable *z*, the model learns distinct reasoning strategies, which impoves multi-attempt metrics (on the arithmetic environment and gsm8k) while preserving single-attempt accuracy (only on gsm8k).

### Strengths
This paper presents a comprehensive theoretical analysis aimed at establishing a connection between mutual information and the llm's pass@K bounds. By introducing token-level mutual information into the reward, the approach enhances the diversity of the model's output strategies, thereby increasing the success rate of Pass@K. The proposed idea is novel enough, and experimental results indicate that it effectively improves the model's Pass@K capability.

### Weaknesses
1. The experiment results of the paper are not convincing. First, the results across the two different experimental settings are inconsistent: in the first experiment, Pass@K improves but Pass@1 stays low, while in the second experiment, Pass@K improves without degrading Pass@1. This indicates that the results are highly sensitive to the environment/dataset, but the paper only tests this method on two environments/datasets, which makes the results lack robustness. Moreover, the experiments only include in-distribution evaluations, without any out-of-distribution (OOD) benchmarks, making it difficult to see the generalization of this approach. Additionally, the analysis of the experiment section is also not enough to explain the results.
2. In the proof section, the paper introduces a strong assumption, but later states that this assumption is difficult to hold in practice. As a result, despite extensive theoretical analysis, the theory does not sufficiently support the final experimental results.
3. Finally, the contribution of the approach remains unclear. Although this approach may be beneficial for improving Pass@K, its performance on Pass@1 is unstable, and it requires careful tuning of the number of *z* (which significantly affects the performance and stability). Moreover, while *z* can control the diversity of responses, the actual meanings of different *z*s are not interpretable, making it nearly impossible for users to use *z* to control the strategies in llm's solutions.

### Questions
1. Have you analyzed whether the learned *z* acquires any generalizable semantic meaning after training?
2. Have you attempted to evaluate the method's generalization ability across different domains?
3. The experimental analysis in gsm8k explains that the pass@1 is maintained due to the existence of a larger set of correct
reasoning approaches. But later in the ablation part, it is mentioned that many GSM8K problems admit only a limited number
of distinct solution paths. There seems to be a contradiction between these statements, and this further strengthens my concern about the generalizability of *z*.
4. The analysis "UpSkill improvement is negatively related to pass@1 (equivalently
pass@kB ) capability" which explains the low Pass@1 in the Arithmetic Environment seems to conflict with the observation in gsm8k, where Pass@1 is maintained or even improved. Could you please provide an analysis to clarify this discrepancy?
5. During testing, if only a single response is generated, how should one select *z* to ensure the quality of the response? Is it possible to let the model choose *z* automatically, rather than requiring manual specification?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the diversity collapse problem in RLVR, where existing methods like GRPO, which optimize for single-attempt accuracy (pass@1), can inadvertently reduce output diversity. The authors propose UpSkill, a training-time method that conditions the model on a latent discrete variable z and employs a novel token-level mutual information reward to encourage distinct responses. Experiments on GSM8K demonstrate that UpSkill can improve pass@k and consensus@k metrics without degrading pass@1 accuracy. The method is also supported by a theoretical analysis linking the mutual information objective to a lower bound on pass@k improvement.

### Strengths
1. The paper targets a crucial problem in existing RL training: the trade-off between single-attempt accuracy and multi-attempt diversity. The authors proposed a novel adaptation of MISL, and the formulation of the token-level mutual information reward made a valuable contribution.
2. The proposed method is supported by theoretical analysis, which links the mutual information objective to a lower bound on pass@k improvement.
3. Experiments show that UpSkill is effective at improving both the single-attempt accuracy and multi-attempt diversity.

### Weaknesses
1. The scalability of UpSkill appears limited, as its experiments are confined to the GSM8K dataset with N=5, a relatively simple benchmark for mathematical reasoning. 
2. The method's stability is questionable, as it shows inconsistent effects on the pass@1 metric across different arithmetic and GSM8K tasks. It also remains unclear whether UpSkill is effective when applied to more powerful base models.

### Questions
1. What if we experiment with UpSkill on more complicated reasoning tasks? I would like to see the results of whether UpSkill can still **effectively** improve the multi-attempt diversity when the query is inherently complex (this would potentially yield more diverse solutions).

### Soundness
3

### Presentation
3

### Contribution
3
