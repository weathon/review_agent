=== CALIBRATION EXAMPLE 44 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the paper's focus on shifting from explicit to latent reasoning. The abstract succinctly states the problem (inference cost of long reasoning trajectories), the proposed solution (Latent Reasoning Tuning, LRT), and the key claims (efficient, outperforms baselines, matches/exceeds Qwen3's hybrid reasoning). The claims are specific and appear to be supported by the experiments later. No major issues.

### Introduction & Motivation
The problem is well-motivated: "overthinking" in slow-thinking LLMs leads to high computational costs. The limitations of existing approaches (post-training compression and prompt-based bypassing) are clearly explained. The introduction effectively sets up the gap: methods that either remain decoding-intensive or are brittle. The contributions are clearly listed. The core idea—using a lightweight reasoning network to generate latent representations—is introduced logically. A minor concern: the introduction could more explicitly foreshadow the empirical finding (Section 2) that fragmented trajectories are sufficient, as this is a key motivation for the latent approach.

### Reasoning Trajectory Analysis (Section 2)
This section provides empirical evidence for the redundancy in reasoning trajectories. The experimental setup (token- and step-level skipping) is reasonable. However, there are significant concerns:
1. **Lack of statistical rigor**: The results are presented without any measures of variance or significance. For example, the accuracy drops from 92.80% to 90.35% when 30% of steps are skipped—is this drop meaningful? The conclusions about "substantial redundancy" and "resilience" are drawn from a single model (Deepseek-R1-Distill-Qwen-7B) on one dataset (Math-500). The generalization of this finding to other models and tasks is assumed but not demonstrated.
2. **Confusing presentation**: The table in Figure 2 is garbled (likely a parsing artifact), making it difficult to interpret the exact numbers. The text references "Figure 2" but the figure caption is incomplete. The paper should include a clear, well-formatted table.
3. **Missing analysis of skip patterns**: Random skipping may not reflect realistic compression; structured skipping (e.g., based on importance) might yield different insights. The connection from this analysis to the latent method is somewhat hand-wavy: the method doesn't skip tokens but replaces the entire trajectory with a latent vector. A stronger justification for why latent representations are a natural next step from fragmented text is needed.

### Method (Section 3)
The method is clearly described. The two-stage training (SFT then RL) is standard and appropriate. However, several critical details are missing or unclear, affecting reproducibility:
1. **Architecture of the reasoning network**: Section 3.2 and Algorithm 1 mention a "lightweight reasoning network" but do not specify its architecture. The appendix (Section C) reveals it is based on Qwen3-Embedding-0.6B with modifications, but the exact architecture (e.g., number of layers, hidden size) and the initialization of the learnable vectors are not detailed. The projection layers \(f_{in}\) and \(f_{out}\) are mentioned but their dimensions are not specified.
2. **Integration with the base model**: How exactly is the latent representation \(z\) fed into the base LLM? Is it prepended as a sequence of continuous embeddings? The paper states it conditions the LLM, but the mechanics (e.g., whether it's used as a prefix to the answer generation) are ambiguous. Figure 1 suggests the latent tokens replace the reasoning tokens, but the text does not clarify the interface.
3. **Training details**: The SFT loss (Eq. 4) uses the base model's hidden states \(H_X\). How are these extracted? At which layer? The RL stage uses GRPO, but the reward function is not defined (Algorithm 1 says "ComputeReward" but gives no formula). Is it simply binary correctness? The KL penalty coefficient and other hyperparameters are in the appendix, but the reward formulation is critical for RL.
4. **Theoretical justification**: The method is motivated by modeling the reasoning trajectory as a function \(h\). However, the step from "fragmented trajectories work" to "a single latent vector works" is a leap. The latent representation is fixed-length (e.g., 256 tokens), but how does this length relate to the original trajectory length? No theoretical or empirical justification for the chosen length is provided.

### Experiments & Results (Section 4)
The experimental setup is comprehensive, with multiple benchmarks and baselines. However, there are major concerns:
1. **Budget enforcement**: The paper uses "budget-forcing" from S1 to limit tokens. This is a reasonable way to compare efficiency, but the details are lacking. How exactly is the budget enforced for each method? For LRT, does the budget include the latent tokens? The latency comparison in Table 7 suggests latent tokens are processed in parallel, but their computational cost should be accounted for in the budget.
2. **Comparison with Qwen3**: Table 2 compares LRT-enhanced Qwen3 with its non-thinking mode. This is a valuable comparison, but it's unclear whether the non-thinking mode also uses a budget. If not, the comparison may be unfair. The paper claims LRT "surpasses" Qwen3's non-thinking mode, but the results are mixed: for Qwen3-1.7B pass@1, LRT wins on average but loses on MATH-500 (60.90 vs. 66.05). The pass@4 results are stronger, but the paper should discuss why pass@1 sometimes underperforms.
3. **Ablation studies**: The ablation on the number of latent tokens (Table 3) is useful, but the explanation for why performance drops at 512 tokens is vague ("larger training scales may be necessary"). A deeper analysis is needed. The ablation on training methods (Table 4) shows RL helps, but what RL algorithm? GRPO? The reward function is still not specified.
4. **Statistical significance**: The paper acknowledges variance in Appendix D.5 and provides standard deviations for one setting. However, these are only for LRT on DeepSeek-R1 under 512 budget. No significance tests or confidence intervals are provided for the comparisons against baselines. Given the sometimes small margins (e.g., 38.00 vs. 37.75 on AMC in Table 1), it's unclear if improvements are statistically significant.
5. **Efficiency claims**: Table 7 shows latency and throughput improvements. However, the peak memory of LRT is higher than non-thinking mode (6528 MB vs. 3946 MB). The paper should discuss this trade-off. Also, the throughput calculation including latent tokens (73.02 tokens/sec) is misleading because latent tokens are not standard tokens; comparing this to textual throughput is apples-to-oranges.

### Related Work (Section 5)
The related work is thorough and covers both efficient reasoning and latent reasoning. The distinction from prior latent reasoning methods (e.g., Coconut) is clearly articulated in Appendix E, but this critical discussion should be in the main text. The paper correctly positions LRT as adapting a pre-trained explicit reasoning LLM with an auxiliary network, unlike methods that retrain the base model.

### Writing & Clarity
Overall, the paper is well-structured and clearly written. However, there are some confusing points:
- The garbled table in Section 2 (Figure 2) impedes understanding.
- The interface between the reasoning network and the base LLM is not fully explained.
- The reward function for RL is never defined.
- Some claims are overstated (e.g., "surpasses the state-of-the-art Qwen3 hybrid reasoning framework" in the abstract) when the results show a more nuanced picture.

### Limitations & Broader Impact
The paper briefly mentions limitations in Appendix D.1: when inference cost is not constrained, thinking mode can achieve higher accuracy. This is an important caveat. However, other limitations are not discussed:
- The method requires training an auxiliary network, which adds complexity.
- The latent representations are not interpretable, which may be a concern for safety-critical applications.
- The experiments are limited to mathematical and logical reasoning; performance on creative or open-ended tasks is unknown.
- The method assumes access to a pre-trained reasoning LLM and its training data (for SFT). The broader impact section is missing; the paper should at least note the positive societal impact of efficient reasoning and potential negative impacts (e.g., misuse of more efficient models).

### Overall Assessment
The paper presents a novel and promising approach to efficient reasoning by replacing explicit trajectories with latent representations. The core idea is interesting and the experimental results show competitive performance. However, the paper has significant shortcomings in methodological detail (architecture, integration, reward function), empirical rigor (lack of statistical significance, unclear budget enforcement), and clarity (missing explanations, garbled table). These issues hinder reproducibility and make it difficult to fully assess the contribution. For ICLR, where technical soundness and clarity are paramount, the paper in its current form does not meet the acceptance bar. With major revisions—particularly providing full methodological details, rigorous statistical analysis, and clearer presentation—the contribution could be strengthened. The potential impact is significant if the efficiency gains hold up under more thorough evaluation.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **Latent Reasoning Tuning (LRT)**, a framework designed to improve the inference efficiency of reasoning-capable Large Language Models (LLMs). LRT replaces the autoregressive generation of lengthy, explicit reasoning trajectories (chain-of-thought) with a single forward pass through a lightweight auxiliary "reasoning network" that produces a compact latent representation. This latent representation conditions the frozen base LLM to generate the final answer directly. The method is motivated by an empirical finding that LLMs can maintain high accuracy even when conditioned on fragmented or incomplete reasoning paths. Experiments on mathematical (GSM8K, MATH-500, AMC) and out-of-domain (GPQA, LSAT) benchmarks demonstrate that LRT outperforms several efficient reasoning baselines and can match or exceed the performance of hybrid models like Qwen3 in non-thinking mode.

### Strengths
1.  **Novel and Pragmatic Approach:** The core idea of compressing explicit reasoning into a learnable, fixed-length latent vector is innovative for efficiency gains. The modular design, which keeps the base LLM frozen and uses an auxiliary network, is practical as it allows seamless switching between latent and explicit reasoning modes.
2.  **Comprehensive Empirical Evaluation:** The paper provides extensive experiments across multiple reasoning benchmarks (5 total, including in-domain and out-of-domain), model families (DeepSeek-R1, Qwen3), and model scales (1.5B to 8B). The results consistently show LRT outperforming strong baselines like NoThinking, ShorterBetter, and LC-R1 under constrained token budgets (Tables 1 & 2).
3.  **Strong Ablation Studies:** The paper includes informative ablations analyzing the impact of the number of latent tokens (Table 3) and the two-stage training strategy (SFT + RL, Table 4). The analysis of inference efficiency (latency, throughput) in Appendix D.3 provides concrete evidence of the practical benefits.
4.  **Insightful Analysis:** The initial analysis in Section 2, demonstrating model resilience to fragmented reasoning trajectories, provides a clear, empirical motivation for the proposed method. The analysis of latent space geometry (Appendix D.4) offers additional, though preliminary, insight into how the representations are structured.

### Weaknesses
1.  **Limited Technical and Conceptual Clarity:** The paper lacks a detailed explanation of *what* the latent representations encode and *how* they function as a substitute for explicit reasoning. The description of the reasoning network's architecture and the interaction mechanism (Eq. 5) is high-level, making the method difficult to reproduce precisely. The claim of "performing reasoning in compact latent representations" (Fig. 3) is not sufficiently substantiated.
2.  **Incomplete Comparison to the State-of-the-Art:** While compared to efficient reasoning methods, the comparison to other contemporary **latent reasoning** approaches (e.g., Coconut, Geiping et al. 2025, Ruan et al. 2025) is relegated to a brief discussion in the Appendix (E). A direct empirical or conceptual comparison in the main text is necessary to properly position LRT's novelty within this active research direction.
3.  **Insufficient Analysis of Limitations:** The paper does not adequately discuss the limitations of LRT. Key questions remain unanswered: What is the computational overhead of the reasoning network? Does the method fail on certain types of problems that inherently require long-form, explicit verification? The performance drop for Qwen3-4B on some out-of-domain tasks at *pass@1* (Table 2) is noted but not analyzed.
4.  **Theoretical Grounding is Light:** The work is primarily empirical. While the fragmented reasoning analysis is compelling, a deeper theoretical discussion on why LLMs can operate effectively on latent representations of reasoning is missing. The connection to concepts like information bottleneck or mechanistic interpretability is not explored.

### Novelty & Significance
**Novelty:** The specific formulation—using a frozen, pre-trained reasoning LLM conditioned on latent vectors from a separately trained, lightweight network—is novel. While the broad concept of latent reasoning is being explored, this paper's focus on *efficiency* and *modularity* for already-capable reasoning models (not training a latent reasoner from scratch) is a distinct and valuable contribution.

**Significance:** The work addresses a critical pain point in deploying powerful "slow-thinking" LLMs: high inference cost. The results demonstrate a promising path to significantly reduce latency and compute while preserving, and sometimes enhancing, reasoning accuracy. If the method generalizes further, it could have substantial practical impact. However, the significance is currently tempered by the need for clearer reproducibility and a more thorough situating within related latent reasoning literature to meet the high bar of ICLR.

### Suggestions for Improvement
1.  **Enhance Methodological Clarity:** Add a detailed diagram or pseudo-code for the exact architecture of the reasoning network and its integration point with the base LLM. Clearly specify the dimensions of inputs/outputs, the nature of the learnable vectors `[r̂1, r̂2, ..., r̂t]`, and the projection layers. This is crucial for reproducibility.
2.  **Deepen the Related Work and Comparison:** Integrate the discussion from Appendix E into the main Related Work section. Provide a concise table or paragraph directly comparing LRT's mechanism, training cost, and intended use case with 2-3 other key latent reasoning methods (e.g., Coconut, "Scaling up test-time compute").
3.  **Conduct a Failure Mode Analysis:** Include a qualitative analysis or a dedicated experiment to understand *when* and *why* LRT might underperform compared to full explicit reasoning. Analyzing incorrect predictions could reveal the limitations of the latent representation and guide future improvements.
4.  **Strengthen the Writing and Presentation:** The introduction and problem statement can be more sharply focused. Avoid overly broad claims (e.g., "fundamentally reimagining reasoning computation") and instead precisely articulate the scope of the contribution (e.g., "a novel efficient inference method for pre-trained reasoning LLMs"). Ensure all figure references in the text are correct (e.g., reference to Fig. 2 in the text seems misplaced).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison with state-of-the-art latent reasoning methods (e.g., Coconut, Geiping et al. 2025).** The paper claims a novel latent reasoning framework but only compares to explicit efficiency methods and Qwen3's non-thinking mode. Without comparison to existing latent reasoning works, the claimed contribution and superiority are not substantiated.
2. **Ablation replacing the learned reasoning network with random/fixed vectors or a simple projection.** This is critical to validate that the learned latent representations are indeed meaningful and necessary, rather than the model simply benefiting from any additional conditioning signal.
3. **Failure case analysis: identify problem types where latent reasoning underperforms compared to full explicit reasoning.** The paper shows aggregate improvements but does not diagnose when the method fails, undermining a clear understanding of its limitations and the validity of the claim that explicit trajectories are unnecessary.
4. **Efficiency comparison in FLOPs or theoretical operation counts, not just system-dependent latency/throughput.** The efficiency claim rests on implementation-specific measurements; a hardware-agnostic computational cost analysis is needed to robustly support the efficiency argument.

### Deeper Analysis Needed (top 3-5 only)
1. **Interpretability of latent representations beyond cosine similarity.** The paper shows clustering patterns but does not analyze what semantic or reasoning content the latent vectors encode. Without this, it is unclear if the network learns structured reasoning logic or just task-specific shortcuts.
2. **Detailed trade-off curve between accuracy, latent token length, and inference cost (latency, memory).** Table 3 shows accuracy vs. token count, but the corresponding efficiency metrics are missing. This is essential to validate the core claim of achieving efficiency without sacrificing performance.
3. **Component analysis of the two-stage training: what specific capabilities does RL add over SFT?** The ablation shows SFT+RL outperforms SFT alone, but it is unclear whether the gain comes from better exploration, reward shaping, or simply more training. A breakdown of reward components and learning dynamics is needed.
4. **Sensitivity analysis of the reasoning network architecture and size.** The method uses a fixed 0.6B embedding model as the reasoning network. The impact of its capacity and architecture choice on performance and efficiency is unexplored, leaving the design decision unjustified.

### Visualizations & Case Studies
1. **Side-by-side case studies contrasting successful and failed examples of latent reasoning vs. explicit reasoning.** Concrete examples would illustrate what types of reasoning steps are captured or lost in the latent representation, making the method's operation and failure modes tangible.
2. **t-SNE/PCA visualizations of latent representations colored by problem type or difficulty.** The cosine similarity table is a start, but a spatial visualization would more clearly reveal whether the latent space organizes problems by domain or reasoning structure as claimed.

### Obvious Next Steps
1. **Dynamic switching mechanism between latent and explicit reasoning based on problem difficulty or uncertainty.** The paper mentions the ability to switch modes but does not implement or evaluate an adaptive policy, which is a natural and practical extension for a hybrid reasoning system.
2. **Application to a broader range of base models (especially larger ones >8B) and reasoning tasks (e.g., code generation, planning).** The evaluation is limited to a few models and primarily mathematical/logical benchmarks; generalization to other domains and scales is necessary to demonstrate broad applicability.
3. **Investigation of variable-length latent sequences instead of fixed-length.** Fixed latent tokens may not be optimal for all problems; adaptive length generation could improve efficiency and performance, aligning with the goal of reducing overthinking.

# Final Consolidated Review
## Summary
This paper introduces Latent Reasoning Tuning (LRT), a framework for improving the inference efficiency of reasoning-capable Large Language Models. LRT replaces the autoregressive generation of lengthy, explicit reasoning chains (chain-of-thought) with a compact, fixed-length latent representation produced by a lightweight auxiliary network. This latent representation conditions the frozen base LLM to generate the final answer directly, significantly reducing decoding steps.

## Strengths
- **Strong empirical results across scales and domains:** The method demonstrates consistent performance improvements over efficient reasoning baselines (NoThinking, ShorterBetter, LC-R1) under constrained token budgets on five diverse reasoning benchmarks (GSM8K, MATH-500, AMC, GPQA, LSAT). It also matches or exceeds the performance of Qwen3's non-thinking mode on models from 1.7B to 8B parameters (Tables 1, 2, 5).
- **Practical and modular design:** The approach keeps the base reasoning LLM completely frozen, attaching only a trainable, lightweight reasoning network. This enables seamless switching between latent and explicit reasoning modes without modifying the core model, a practical advantage for deployment.
- **Concrete efficiency gains:** The inference evaluation (Table 7) shows LRT achieves the lowest latency and highest effective throughput compared to standard thinking and non-thinking modes, providing tangible evidence for its core efficiency claim.

## Weaknesses
- **The mechanism of latent reasoning is insufficiently explained:** While the architecture is outlined in Appendix C, the paper lacks a clear description of *how* the fixed-length latent vector functionally substitutes for a multi-step reasoning trajectory. The interface between the latent representation and the base LLM's generation process is described at a high level (conditioning via concatenation) but the precise operational mechanism and what the latent tokens encode remain opaque, hindering reproducibility and deep understanding.
- **Incomplete positioning within the latent reasoning literature:** The comparison is focused on methods that shorten explicit reasoning (e.g., ShorterBetter) or bypass it via prompting (NoThinking). A direct empirical or conceptual comparison to contemporary works on latent/continuous reasoning (e.g., Coconut, Geiping et al. 2025, Ruan et al. 2025), which is a closely related and active subfield, is relegated to an appendix. This omission makes it difficult to assess the novelty and relative merits of LRT's specific approach.
- **Limited analysis of failure modes and limitations:** The paper shows aggregate performance gains but does not analyze *when* or *why* LRT might fail compared to full explicit reasoning. A qualitative analysis of incorrect predictions could reveal important limitations of the latent compression (e.g., on problems requiring long, verifiable derivations) and clarify the boundary conditions of the method's applicability.

## Nice-to-Haves
- **Variable-length latent sequences:** Exploring adaptive generation of latent sequence length per problem, rather than a fixed budget, could better align with the goal of reducing overthinking and further optimize the efficiency-accuracy trade-off.
- **Uncertainty-aware mode switching:** Implementing and evaluating a policy to dynamically choose between latent and explicit reasoning based on predicted problem difficulty or model confidence would be a natural and valuable extension of the hybrid capability the framework enables.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Lack of statistical rigor / no significance tests":** The paper provides standard deviations for its main results in Appendix D.5 and reports averages over multiple stochastic decodings. While more rigorous statistical reporting is always beneficial, the performance margins (e.g., +5-10% on key benchmarks) are substantial relative to the reported variance, making the improvements clear.
- **Weakness: "Missing details on reward function for RL":** The paper specifies the use of GRPO and mentions rule-based reward signals. While the exact reward formulation (e.g., binary correctness) could be stated more explicitly, the training framework and outcome are sufficiently clear for the core claim.
- **Weakness: "Budget enforcement details are lacking":** The paper states it uses budget-forcing from S1 and enforces the same token budget for all methods. This is a standard and reasonable approach for efficiency comparisons in this line of work.
- **Weakness: "Theoretical justification is a leap":** The paper is primarily an empirical contribution. The foundational analysis in Section 2 (model resilience to fragmented trajectories) provides adequate, empirically-grounded motivation for exploring latent compression, which is sufficient for this type of work.
- **Strength: "The paper is well-written / topic is important":** These are generic strengths that apply to many papers and do not highlight what is specific and effective about this work.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- In the main text, add a concise subsection or paragraph directly comparing LRT's mechanism (frozen base LLM + auxiliary network) and intended use case with 2-3 other key latent reasoning methods (e.g., Coconut, "Scaling up test-time compute"). This is crucial for properly situating the contribution.
- Include a qualitative failure analysis, examining a sample of problems where LRT underperforms compared to the explicit reasoning baseline. Discuss potential reasons (e.g., loss of specific intermediate verification steps) to better define the method's limitations.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 8.0, 4.0]
Average score: 5.0
Binary outcome: Accept
