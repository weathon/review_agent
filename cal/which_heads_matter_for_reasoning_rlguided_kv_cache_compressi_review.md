=== CALIBRATION EXAMPLE 48 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title is clear and reflective of the core contribution. The abstract succinctly states the problem (existing KV cache methods fail for reasoning models), the key insight (functional heterogeneity of heads), the proposed solution (RLKV), and the main result (20-50% cache reduction with near-lossless performance). The claim of being the "first to identify a set of heads that matter for reasoning behaviors" is a strong contribution claim that the rest of the paper must solidly support.

**Introduction & Motivation:** This section is a major strength. The problem is well-motivated with concrete memory usage numbers. Figure 1 effectively illustrates the failure of existing methods (token-dropping and head-reallocation) on reasoning models, isolating extended CoT generation as the key challenge. The analysis of distinct error modes (repetitive vs. over-extended CoT) provides clear intuition for why these methods fail. The hypothesis of "reasoning heads" is logically derived from these observations. The three contributions are clearly stated.

**Methodology:**
*   **Clarity of Formulation:** The core idea—using RL-optimized gating adapters under sparsity pressure to identify heads critical for reasoning—is novel and well-explained in Figure 2. The use of mixed attention (Eq. 1) as a probing mechanism is appropriate.
*   **Operational Definition:** The definition of "reasoning heads" as those that "significantly degrade reasoning performance under local KV cache access" is operational but somewhat circular. It identifies heads important for the task used in RL training, but the paper later claims these heads are *for reasoning behaviors*. This conflation needs clearer discussion: are these heads fundamentally specialized for reasoning, or simply important for maintaining performance on the specific training tasks?
*   **RL Formulation Details:** The description of the RL objective (Eq. 2) has gaps crucial for reproducibility and understanding.
    1.  **Reward Signal (`r_i`)**: The paper states rewards are based on "final answer correctness." How is this binary (or potentially partial) correctness score transformed into the reward `r_i` used in Eq. 3? Is it simply 1 for correct, 0 for incorrect? Or is there a verifier score? This is a critical detail.
    2.  **Policy `π_α`**: The policy is said to be the "model’s generation probability distribution conditioned on the current gating parameters `α`." This is vague. Since the LLM parameters are frozen, the policy is effectively a function that, given `α`, produces a sequence. The optimization is directly on `α`. The phrasing should be clarified to avoid confusion with policies that output actions.
    3.  **KL Penalty Removal:** The removal of the KL penalty is noted but not justified in depth. While the aim is to maximize reward signal strength, this could lead to unstable training or overfitting to the reward function. The successful training suggests the adaptive penalty weighting mitigates this, but the rationale deserves more discussion.
*   **Stabilization Techniques:** The conflict between sparse reward and dense L1 penalty is a key insight. The proposed solutions—self-distillation sampling (curriculum based on output length) and adaptive penalty weighting (Eq. 4)—are sensible and well-motivated by Figure 4. The ablation studies later effectively validate their importance.

**Experiments & Results:**
*   **Setup & Baselines:** The experimental setup is thorough. Using two popular reasoning models (Llama-3.1-8B-R1, Qwen-2.5-7B-R1) and four benchmarks spanning math and code is excellent. The choice of baselines (H2O, R-KV, DuoAttention) is comprehensive and representative of both token-dropping and head-reallocation categories. The adjustments to baselines for fair comparison (dynamic budget, overhead tokens) are commendable and detailed in the appendix.
*   **Main Results (Figure 5, Tables 1, 3, 4):** The results are compelling. RLKV consistently outperforms baselines, especially at higher sparsity (0.4, 0.6). The counter-intuitive result of sometimes *exceeding* full KV cache performance is intriguing and well-discussed (noise reduction hypothesis). The trend of degradation at 0.8 sparsity logically supports the claim that reasoning requires a sufficient number of full-cache heads.
*   **Analysis on Reasoning vs. Retrieval Heads (Figure 6):** This is a critical analysis that strengthens the paper's core thesis. Showing that RLKV-identified heads are more sensitive to compression than retrieval heads (from DuoAttention) or random heads provides strong evidence that they capture a distinct, reasoning-critical function.
*   **Error Mode Analysis (Figure 7):** The qualitative analysis of different error modes (repetitive for reasoning-head compression vs. varied for retrieval-head compression) offers valuable mechanistic insight and aligns well with the initial motivation.
*   **Ablation Studies (Figure 8):** These are well-designed and clearly show the importance of the two stabilization components and the choice of `β`.
*   **Generalization & Additional Evaluations (Appendix A.5, A.6):** The results on Qwen-3-4B-Thinking and MMLU-Pro subsets are valuable for demonstrating generality beyond the primary training/test models and domains. The performance on some MMLU-Pro subsets (e.g., Law) not being lossless even at low sparsity is an honest presentation of a limitation.
*   **Memory Efficiency & Latency (Section 3.4, Appendix A.9):** The memory reduction claims (20-50%) are supported by the results. The latency analysis in the appendix is a crucial addition, as a compression method must not introduce excessive overhead. The analysis showing that latency approaches full attention for long sequences and that end-to-end speedups are possible (Table 7) addresses a key practical concern. The discussion of continuous batching is appropriate.
*   **Missing Analysis - Computational Cost of RLKV:** The training cost is mentioned (40, 22, 36 GPU-hours), which is reasonable. However, a discussion is needed on whether this one-time cost per model is justified by the inference savings, and how it compares to the training cost of baseline learning-based methods like DuoAttention.

**Writing & Clarity:** Overall, the paper is well-written. However, there are points of confusion:
1.  The term "reasoning behaviors" is used frequently but never formally defined. It is intuitively understood as the coherent, multi-step CoT process, but a more precise description would help.
2.  In Section 2.2, "The reward signal preserves high `α_i,j` values for *reasoning heads*..." implies the reward has direct knowledge of which heads are reasoning heads, which is not the case. The reward is based on final answer correctness. The sentence should be rephrased to reflect that high `α` values emerge *because* they lead to correct answers.
3.  Some figure references in the main text are broken (e.g., references to Figure 1(b) column labels), but this appears to be a parser/formatting artifact as per the note.

**Limitations & Broader Impact:** The limitations section (Future Work) appropriately identifies key avenues: variability across models/tasks, understanding the full functional role of heads, and pushing compression ratios higher. A more explicit "Limitations" subsection could strengthen the paper by directly acknowledging current constraints, such as: the need for task-specific training data (mathematical reasoning), the performance drop at very high sparsity (0.8), and the potential overhead of the head-reallocation implementation. The societal impact is implicitly positive (efficient deployment of reasoning models), and no significant negative impacts are evident.

### Overall Assessment
This paper presents a novel, well-motivated, and empirically strong approach to a pressing problem: KV cache compression for reasoning LLMs. The core idea of using RL with sparsity pressure to identify reasoning-critical heads is clever and clearly justified by the failure modes of existing methods. The experimental validation is extensive, convincing, and includes important analyses on head importance, error modes, latency, and generalization. The main weaknesses lie in the methodological details: the RL formulation requires clearer specification of the reward, and the conflation between "heads important for the training task" and "reasoning heads" needs more careful discussion. Furthermore, a more direct comparison of training costs with learning-based baselines would be beneficial. These issues are addressable and do not undermine the paper's solid contribution. With revisions to clarify the methodology, this paper meets the bar for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces RLKV, a reinforcement learning method to identify "reasoning-critical" attention heads in large language models (LLMs) that perform chain-of-thought reasoning. The identified heads are allocated a full Key-Value (KV) cache, while the remaining heads use a compressed cache, enabling a 20-50% reduction in KV cache memory with minimal performance loss on reasoning tasks. The core idea is that only a small subset of heads are essential for maintaining reasoning integrity, and RLKV uses reward signals from generated reasoning traces to learn which heads require full cache access.

### Strengths
1. **Novel Methodology**: The use of reinforcement learning (specifically GRPO) to directly optimize the relationship between head-level KV cache allocation and reasoning quality is a creative and novel approach. It moves beyond static heuristics or retrieval-oriented head identification used in prior work.
2. **Comprehensive Empirical Evaluation**: The paper provides extensive experiments across three reasoning models (Llama-3.1-8B-R1, Qwen-2.5-7B-R1, Qwen-3-4B-Thinking) and multiple benchmarks (GSM8K, MATH, AIME24, MBPP, and MMLU-Pro subsets). Results consistently show RLKV outperforms strong baselines (H2O, R-KV, DuoAttention), especially at moderate to high compression ratios (0.4-0.6 sparsity).
3. **Insightful Analysis**: The paper includes a detailed ablation study (adaptive penalty weighting, self-distillation sampling), error mode analysis (showing repetitive errors when reasoning heads are compressed), and a comparison between "reasoning heads" and "retrieval heads." This provides valuable insights into the functional heterogeneity of attention heads in reasoning models.

### Weaknesses
1. **Limited Model Scale and Task Diversity**: All experiments are conducted on models with up to 8B parameters. It remains unclear how well the method scales to much larger models (e.g., 70B+), which are commonly used for complex reasoning. Additionally, the evaluation is heavily focused on mathematical and coding tasks; performance on other reasoning types (e.g., commonsense, logical deduction) is not explored.
2. **Training Complexity and Cost**: While the training cost is reported as "modest" (e.g., 40 GPU-hours for Llama-3.1-8B-R1), the RL-based approach is inherently more complex and less straightforward to implement than training-free or single-pass methods. The need for a curated dataset of correctly solved problems and careful stabilization techniques (adaptive penalty, self-distillation) adds to the implementation burden.
3. **Incomplete System Performance Picture**: The latency analysis in the appendix shows that the current PyTorch implementation does not always yield end-to-end speedups due to the overhead of splitting and recombining attention heads. The paper acknowledges that custom kernels are needed for optimal speed, but the actual inference-time benefits are not yet fully realized or demonstrated in a production-grade serving system.

### Novelty & Significance
The paper presents a novel and significant contribution. The core idea—using RL to dynamically identify reasoning-critical heads for KV cache compression—is distinct from prior token-dropping or retrieval-head methods. The finding that reasoning capability relies on a sparse set of heads, and that compressing others can be near-lossless, offers a new perspective on model efficiency and interpretability. The work has high practical significance for deploying reasoning LLMs under memory constraints and could influence future research on efficient inference and head specialization.

### Suggestions for Improvement
1. **Scale Up Experiments**: Include results on at least one larger reasoning model (e.g., 30B+ parameters) to demonstrate the method's scalability and to see if the fraction of reasoning heads changes with model size.
2. **Broaden Task Evaluation**: Evaluate on a wider range of reasoning benchmarks (e.g., logical reasoning from BIG-Bench, science QA) to better assess the generality of the identified reasoning heads beyond mathematical domains.
3. **Deeper Head Analysis**: Provide a more detailed analysis of the identified reasoning heads (e.g., their layer distribution, attention patterns, or relationship to known mechanisms like induction heads) to better explain *why* they are critical for reasoning.
4. **System Optimization and Benchmarking**: Implement and evaluate a more optimized inference kernel (e.g., in CUDA or via integration with vLLM/SGLang) to provide a clearer picture of the end-to-end latency and throughput gains in a realistic serving scenario.
5. **Clarify Comparison with Baselines**: The paper notes that fixed-budget evaluations can be unfair for reasoning tasks. While a dynamic budget is used, a more detailed discussion in the main text (not just the appendix) on why this is necessary and how it affects comparisons would strengthen the methodological rigor.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation comparing RL-based selection to a simple, task-agnostic baseline.** The paper lacks a comparison to a straightforward method like selecting heads based on the magnitude of their gating parameters (`α`) after training the *uncompressed* model on the same data. This is necessary to prove that the RL optimization (and its associated reward signal) is crucial, rather than just the act of identifying heads with high activity on reasoning tasks.
2. **Evaluation with an oracle upper bound.** To validate the claim of identifying "reasoning heads," an experiment is needed where heads are ablated one-by-one (or in groups) on a held-out validation set to establish a ground-truth ranking of head importance. The correlation between this oracle ranking and RLKV's ranking would directly measure identification accuracy.
3. **Cross-domain training/evaluation.** The core method is trained exclusively on mathematical reasoning data. A critical missing experiment is to train RLKV on a *different* domain (e.g., coding) and evaluate on math (and vice-versa) to test the claim that identified heads are general "reasoning heads" and not simply overfitted to the training task distribution.
4. **Comparison to a compute-equivalent baseline.** The RL training has non-trivial cost (40+ GPU-hours). An experiment is needed where the same compute budget is given to a baseline (e.g., DuoAttention) for more extensive training or search, to ensure the gains are from the method itself and not just additional compute.

### Deeper Analysis Needed (top 3-5 only)
1. **Characterization of what "reasoning heads" actually do.** The paper claims these heads are critical for CoT consistency but provides zero analysis of their mechanistic function. A necessary analysis is to examine the attention patterns of identified reasoning heads versus non-reasoning heads across reasoning steps to show they attend to critical previous tokens (e.g., premises, intermediate results).
2. **Sensitivity analysis of head selection stability.** The analysis is missing any measure of how stable the identified set of reasoning heads is across different random seeds or across different subsets of the training data. If the set varies wildly, the interpretation of a consistent "reasoning head" role is undermined.
3. **Quantification of the trade-off: sparsity vs. necessary heads.** The paper notes performance drops at high sparsity but doesn't analyze the relationship between the *number* of reasoning heads and task difficulty/complexity. An analysis correlating the minimum number of heads required for near-lossless performance with metrics of problem difficulty (e.g., solution length, accuracy of base model) is needed to understand the limits of compression.

### Visualizations & Case Studies
1. **Visualize attention patterns for selected "reasoning heads" in successful vs. failed reasoning traces.** Case studies should show, for a few problems, which past tokens a identified reasoning head attends to during key reasoning steps. Comparing this to a head identified as a "retrieval head" would concretely demonstrate the claimed functional difference.
2. **Case studies of failure modes at high compression.** The error mode analysis is aggregated. Specific examples are needed showing the *content* of generations when top RLKV-identified heads are compressed, illustrating how the reasoning chain breaks (e.g., which logical step is missed or repeated), to validate the hypothesis about broken CoT consistency.

### Obvious Next Steps
1. **Compare to random head selection.** A fundamental baseline is missing: for each sparsity level, compare performance when randomly selecting heads to keep at full cache. This would establish the non-triviality of the learned selection; if random selection performs similarly, the entire premise is weak.
2. **Proper cost-benefit analysis.** The paper mentions training cost but does not amortize it against inference savings. A clear calculation is needed: for a given deployment scenario (e.g., serving X queries), does the total cost (training + inference) of RLKV actually beat just using a larger GPU with the uncompressed model or other baselines?
3. **Validate on a broader suite of reasoning models.** The method is evaluated on R1-distilled models and one "Thinking" model. An obvious step is to test on a fundamentally different reasoning architecture, such as OpenAI's o1-preview series (if accessible) or Gemini Flash Thinking, to claim generality across reasoning LLMs.

# Final Consolidated Review
## Summary
RLKV introduces a reinforcement learning method to identify reasoning-critical attention heads in large language models, enabling efficient KV cache compression. By allocating full cache only to these heads and compressing others, the method achieves 20-50% memory reduction with near-lossless performance on mathematical and coding reasoning tasks.

## Strengths
- **Novel RL-based formulation**: The use of GRPO with sparsity pressure to directly optimize head-level cache allocation based on reasoning quality is a creative departure from static heuristics or retrieval-oriented methods, effectively capturing dynamic reasoning behaviors.
- **Comprehensive and insightful evaluation**: Experiments across three reasoning models and multiple benchmarks consistently show RLKV outperforms strong baselines, with detailed analysis of error modes, head importance (vs. retrieval heads), and ablation studies validating key design choices.
- **Effective stabilization techniques**: The introduction of adaptive penalty weighting and self-distillation sampling addresses the sparse reward vs. dense penalty conflict, enabling stable training as demonstrated in ablations.

## Weaknesses
- **Incomplete RL formulation details**: The paper does not specify how the reward signal \( r_i \) is computed from final answer correctness (e.g., binary or continuous), which is critical for reproducibility and understanding the optimization dynamics.
- **Limited scalability evidence**: Experiments are conducted only on models up to 8B parameters; the method's effectiveness on larger-scale reasoning models (e.g., 70B+) remains unverified, raising questions about generalizability.
- **Narrow task focus**: Evaluation is heavily skewed toward mathematical and coding tasks, with only limited results on MMLU-Pro subsets; generalization to other reasoning types (e.g., commonsense, logical deduction) is not adequately demonstrated.
- **Insufficient baseline comparisons**: The paper lacks comparison to simple head selection strategies (e.g., random selection or selection based on attention scores without RL), making it unclear if the RL optimization is necessary or if gains stem from selective compression alone.
- **Training cost justification unclear**: While training costs are reported, there is no direct comparison to the training overhead of learning-based baselines like DuoAttention, obscuring the practical cost-benefit trade-off.
- **Latency overhead in current implementation**: The PyTorch-based head reallocation introduces computational overhead, and end-to-end speedups are not fully realized without custom kernels, limiting immediate deployment benefits.

## Nice-to-Haves
- Broader evaluation on diverse reasoning benchmarks (e.g., logical reasoning, science QA) to strengthen claims of generality.
- Deeper mechanistic analysis of identified reasoning heads, such as their attention patterns or layer distribution, to explain their role in maintaining CoT consistency.
- Sensitivity analysis of head selection stability across random seeds or data subsets to ensure robustness.
- Cross-domain training experiments (e.g., train on coding, test on math) to test if identified heads are task-specific or general reasoning components.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Circular definition of "reasoning heads"**: The operational definition (heads that degrade performance under local cache access) is valid for the paper's scope and not circular.
- **Minor writing clarity issues**: Assumed to be parser artifacts or easily fixable in revision.
- **Demand for oracle upper bound experiments**: Not standard practice and beyond the paper's core contribution.
- **Request for theoretical proofs or user studies**: Inappropriate for this empirical systems paper.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Clarify the reward computation in the RL formulation (e.g., explicitly state if \( r_i \) is binary based on final answer correctness or uses a verifier score).
- Include a comparison to random head selection at various sparsity levels to baseline the performance gains and demonstrate the non-triviality of learned selection.
- Implement and evaluate optimized kernels (e.g., in CUDA or integrated with vLLM/SGLang) to provide concrete latency/throughput improvements in a production-like setting.
- Add a brief discussion comparing training costs to learning-based baselines and amortizing them against inference savings for typical deployment scenarios.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 8.0]
Average score: 4.5
Binary outcome: Reject
