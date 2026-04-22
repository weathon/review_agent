# MTIR-SQL:Multi-turn Tool-Integrated  Reasoning Reinforcement Learning for Text-to-SQL

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
As large language models (LLMs) are increasingly used in Text-to-SQL tasks, Reinforcement Learning (RL) has become a common method for improving performance. Existing methods primarily rely on static execution feedback, which restricts real-time error correction. However, integrating multi-turn tool invocation along with dynamic feedback could significantly improve adaptability and robustness, ultimately enhancing model performance. To address these issues, we propose MTIR-SQL, an innovative Multi-turn Tool-Integrated Reasoning reinforcement learning framework for Text-to-SQL. Our approach introduces an execution-aware multi-turn reasoning paradigm that seamlessly incorporates database execution feedback at each reasoning step, enabling context-sensitive query generation and progressive refinement throughout the reasoning process. The framework extends the GRPO algorithm to accommodate complex multi-turn interaction scenarios. Considering the training instability characteristics of MTIR and the potential for significant Deviation of model distribution from the initial model, we enhance the GRPO algorithm by adding a trajectory filtering mechanism and removing KL loss constraints. Experimental results demonstrate that MTIR-SQL, with 4B parameters, achieves 64.4\% accuracy in the BIRD Dev and 84.6\% execution accuracy in the SPIDER Dev, significantly outperforming existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MTIR-SQL, a multi-turn RL NL2SQL framework with SQL execution feedback. It is trained with a modified GRPO algorithm with trajectory filtering and without KL regularization. The experiment results on the Qwen3-4B model suggest good accuracy on BIRD and SPIDER benchmarks.

### Strengths
- The empirical results on BIRD dev seem to be strong for a 4B model.
- The paper addresses a well-motivated NL2SQL problem.

### Weaknesses
- **Limited Methodological Novelty:** The paper's primary weakness is its limited methodological contribution. The core ideas are largely applications or incremental modifications of existing techniques.
    - The MTIR framework closely resembles the self-correction/multi-turn reasoning workflows already established in prior work (e.g., DART-SQL, CHASE-SQL, Arctic-Text2SQL-R1). Integrating this loop with RL, while effective, appears to be a straightforward incremental step.
    - The extensions to the GRPO algorithm, such as removing KL regularization and adding trajectory filtering, are common practices in modern RL training pipelines (e.g., verl, DAPO, rStar-Math). Any potential novelty in the *design* of the filtering functions is unfortunately not discussed, making the contribution in this area unclear.
- **Insufficient Baseline Comparisons:** The performance evaluation lacks a fair comparison against key baselines. While Tables 1-3 show the 4B model outperforming some larger models, they omit direct comparisons against current state-of-the-art (SOTA) RL-based models from the BIRD benchmark leaderboard, such as arctic-r1. This omission makes it difficult to properly situate the paper's performance claims.
- **Lack of Scalability Analysis:** The empirical validation is limited to a single 4B model. The paper provides no analysis of how the methodology performs with models of different scales (e.g., 7B, 14B, or 32B), making it difficult to judge the generalizability of the approach.
- **Contribution Feels More Engineering than Research:** The work reads more like a strong engineering report, demonstrating a successful combination of existing components, than a fundamental research paper. As a result, I am concerned that it offers limited new insights or generalizable lessons for the ICLR community.

### Questions
- What is “retrieval-based token masking” in line 99, page 2? There is no explanation for this term in the rest of this paper.
- What is the filtering function for selective rollout filtering?
- In figure 3, given the fact that GRPO_Filter starts to collapse after step 140, is it possible that GRPO finally outperform GRPO_Filter? What makes GRPO_Filter collapse earlier than the original GRPO method?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MTIR-SQL, a reinforcement learning framework for Text-to-SQL that leverages Multi-turn Tool-Integrated Reasoning (MTIR). The core problem it addresses is that existing RL methods often use static, final-execution feedback, which limits the model's ability to perform real-time error correction. MTIR-SQL proposes an execution-aware paradigm where the model's reasoning is interleaved with database execution at each step, allowing for progressive query refinement. The method extends the GRPO algorithm with a "GRPO-Filter" mechanism, which incorporates trajectory filtering and removes the KL divergence constraint to stabilize multi-turn training. Experimental results show that the 4B parameter MTIR-SQL model achieves 64.4% execution accuracy on the BIRD Dev set and 84.6% on the SPIDER Dev set, demonstrating strong performance and efficiency.

### Strengths
- **S1.** The paper's core idea is intuitive and well-motivated. Shifting from static, end-of-process rewards to a multi-turn, interactive paradigm that integrates execution feedback at each reasoning step is a logical and promising approach for improving complex, multi-step SQL generation.
- **S2.** The empirical results are strong, especially considering the model's parameter count. The 4B model achieves competitive performance (64.4% on BIRD Dev), outperforming several larger models.

### Weaknesses
- **W1. Limited Novelty and Contribution:** The primary technical contribution, "GRPO-Filter" (Section 3.1), appears to be a direct adaptation of a pre-existing method (SimpleTIR) to the Text-to-SQL domain. The paper does not sufficiently clarify the novel, task-specific innovations or distinctions. This makes the core contribution feel incremental and diminishes its perceived value, which is critical for assessing the paper's impact.
- **W2. Insufficient Experimental Baselines:** The experimental evaluation (Section 4.2) is not comprehensive.
    - It lacks comparisons against many state-of-the-art, specialized Text-to-SQL methods, including both prompting-based (e.g., OpenSearch-SQL[1], Alpha-SQL[2], CHESS-SQL[3]) and other fine-tuning methods (e.g., CHASE-SQL[4], Reasoning-SQL[5], Think2SQL[6]). Comparisons to general-purpose LLMs (like GPT-4) are insufficient for a task-specific paper.
    - The main results compare the proposed method on a Qwen3-4B model against baselines running on older Qwen2.5 models. This is an unfair comparison, as the performance gains may be attributable to the stronger base model (Qwen3) rather than the MTIR-SQL method itself.
- **W3. Key Methodological Ambiguities:** Several key components of the method are not clearly defined in the main text.
    - "Selective Rollout Filtering" (Section 3.1): This is a central component of the "GRPO-Filter", yet the paper never specifies how trajectory quality is evaluated. The filtering function $F(x,y)$ is undefined.
    - "Execution Reward Function" (Section 3.3): In a multi-turn reasoning trace with multiple SQL calls, it is unclear if this reward is applied to every executed SQL query or only to the final SQL in the answer. This is a critical detail for reproducibility.
- **W4. Missing Practicality Analysis:** For a Text-to-SQL method, practical deployment metrics are essential. The paper provides no analysis of inference latency or token cost. A multi-turn, self-correcting method is intuitively more expensive, and the paper must quantify this accuracy-vs-cost trade-off.
- **W5. Clarity and Presentation Issues:** There is a significant discrepancy between the methodology and the ablation study. Section 4.3 introduces and ablates an "$R_f$ (Exploration Reward)", but this reward is never mentioned or defined in the "Reward Design" (Section 3.3). This creates major confusion for the reader regarding the actual reward structure used.

[1] Xie, X., Xu, G., Zhao, L., & Guo, R. (2025). Opensearch-sql: Enhancing text-to-sql with dynamic few-shot and consistency alignment.

[2] Li, B., Zhang, J., Fan, J., Xu, Y., Chen, C., Tang, N., & Luo, Y. (2025). Alpha-sql: Zero-shot text-to-sql using monte carlo tree search.

[3] Talaei, S., Pourreza, M., Chang, Y. C., Mirhoseini, A., & Saberi, A. (2024). Chess: Contextual harnessing for efficient sql synthesis.

[4] Pourreza, M., Li, H., Sun, R., Chung, Y., Talaei, S., Kakkar, G. T., ... & Arik, S. O. (2024). Chase-sql: Multi-path reasoning and preference optimized candidate selection in text-to-sql.

[5] Pourreza, M., Talaei, S., Sun, R., Wan, X., Li, H., Mirhoseini, A., ... & Arik, S. (2025). Reasoning-sql: Reinforcement learning with sql tailored partial rewards for reasoning-enhanced text-to-sql.

[6] Papicchio, S., Rossi, S., Cagliero, L., & Papotti, P. (2025). Think2sql: Reinforce llm reasoning capabilities for text2sql.

### Questions
- **Q1. Clarification on "Selective Rollout Filtering"**: In Section 3.1, you introduce "Selective Rollout Filtering" as a key component of GRPO-Filter. However, the paper does not define the filtering function $\mathcal{F}(\cdot)$ or the quality threshold $\tau$. Could you please specify the exact criteria used to evaluate trajectory quality and filter these rollouts?
- **Q2. Novelty of GRPO-Filter**: The proposed "GRPO-Filter", particularly its use of trajectory filtering for multi-turn reasoning, appears to share significant conceptual overlap with the SimpleTIR method (which you cite for mathematical reasoning). Could the authors please explicitly articulate the core innovations and distinctions of GRPO-Filter as applied to Text-to-SQL, beyond the adaptation of this filtering concept? This is essential for evaluating the paper's novelty.
- **Q3. Scope of Execution Reward**: Regarding the "Execution Reward" (Section 3.3), in a multi-turn trajectory that may involve multiple SQL tool calls (as seen in Appendix C.3), is this reward applied to every intermediate SQL execution, or only to the final SQL query presented in the \<answer\> tag?
- **Q4. Undefined "Exploration Reward"**: Section 4.3 introduces and ablates an "$R_f$ (Exploration Reward)". However, this reward component is not clearly defined in the methodology's "Reward Design" (Section 3.3). Please clarify precisely what this reward component is, how it is calculated, and where it fits into the overall reward function described in Section 3.3.
- **Q5. Baseline Comparisons**: The experimental comparison in Tables 1-3 is a significant concern for evaluating the method's effectiveness:
    - Why were many state-of-the-art, specialized Text-to-SQL baselines (both prompting-based like Alpha-SQL, and fine-tuning/RL-based like SQL-R1, Think2SQL, and Reasoning-SQL) omitted in favor of general-purpose LLMs?
    - The main results compare MTIR-SQL on Qwen3-4B against baselines on different, often weaker, models (e.g., Qwen2.5-Coder-7B). This makes it difficult to isolate the gains. A more direct SOTA comparison controlling for the base model is needed.
- **Q6. Practicality and Cost**: The multi-turn framework intuitively adds significant inference latency and token cost. Could the authors provide an analysis of these practical metrics? For example, what is the average number of tool-call turns (out of the max N=6) used at inference, and what is the latency/cost trade-off compared to the Text-to-SQL baselines?

### Soundness
3

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
4

### Summary
This paper introduces MTIR-SQL, a reinforcement learning framework for Text-to-SQL that leverages multi-turn, tool-integrated reasoning. The model iteratively refines SQL predictions using real-time database feedback, and incorporates a modified Group Relative Policy Optimization (GRPO-Filter) algorithm that removes KL regularization and filters out low-quality trajectories to stabilize learning. Experiments on the BIRD and SPIDER development sets show 64.4% and 84.6% accuracy respectively with a compact 4B-parameter model, demonstrating that multi-turn reasoning can yield competitive results even at smaller scales.

While results are only reported on development splits rather than test sets, the overall performance is reasonable and the approach is methodologically interesting. Compared to XYZ-Text2SQL-R1, which attains higher accuracies (e.g., 68.9% on BIRD-Dev and 88.8% on SPIDER-Test) with a larger 7B model and a simpler single-turn RL setup, MTIR-SQL offers a more sophisticated exploration of interactive reasoning.

### Strengths
Somewhat framework (only if applied to txt2sql not coding in general) : Introduces a multi-turn, execution-aware RL loop that allows models to refine SQL queries dynamically.

Algorithmic contribution: The GRPO-Filter variant is a thoughtful modification addressing instability and low-quality rollouts.

Parameter efficiency: Competitive results achieved with only 4B parameters, smaller than most contemporary baselines.

### Weaknesses
Limited evaluation: Results are only on development sets (no held-out test benchmarks).

Comparative performance: Slightly below state-of-the-art systems like XYZ-Text2SQL-R1, which achieve higher accuracies with simpler pipelines.

Complexity vs. gain: Multi-turn SQL execution adds computation and engineering overhead, with moderate empirical benefit.

Generalization scope: Evaluation restricted to BIRD and SPIDER; cross-domain robustness not demonstrated.

### Questions
none right now.

### Soundness
3

### Presentation
3

### Contribution
3
