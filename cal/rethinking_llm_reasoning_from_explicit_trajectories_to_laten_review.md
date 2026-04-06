=== CALIBRATION EXAMPLE 52 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title clearly indicates a shift from explicit to latent reasoning. The abstract succinctly states the problem (overthinking, inefficiency), the proposed method (LRT), and claims superior performance over relevant baselines and Qwen3’s hybrid reasoning. These claims are strong and need to be backed by rigorous experiments. The abstract is well-written and sets appropriate expectations.

**Introduction & Motivation**  
The introduction effectively motivates the problem of computational inefficiency due to lengthy reasoning trajectories in “slow-thinking” models. It reviews existing approaches (post-training compression, prompt-based methods) and identifies their limitations, creating a clear gap for LRT. The contributions are stated clearly. However, the introduction does not explicitly mention how LRT differs from prior latent reasoning methods (e.g., Coconut) beyond being modular and non-intrusive—this distinction is later discussed in the appendix but should be highlighted earlier.

**Section 2: Reasoning Trajectory Analysis**  
This section aims to demonstrate redundancy in reasoning trajectories via random token/step skipping. The core finding—that models retain high accuracy even with 50% skipping—supports the premise that full trajectories are unnecessary. However, the analysis has several weaknesses:
- The random skipping strategy is simplistic and may not reflect intelligent compression; it would be more convincing to compare with importance-based skipping (e.g., based on perplexity or gradient signals).
- The presentation of results (Figure 2) suffers from formatting artifacts (the table is garbled), making it difficult to interpret exact numbers. While this is likely a parser issue, the authors should ensure the data is clearly presented in the final version.
- The analysis uses only one model (Deepseek-R1-Distill-Qwen-7B) on one dataset (Math-500). Generalizability to other models/tasks is not shown, though later experiments on other benchmarks partially address this.
a- The claim that models are resilient to “noisy or fragmental input” is interesting but not deeply analyzed: why does this happen? Is it because the model relies on early salient tokens, or because it can recover from gaps? A deeper analysis would strengthen the motivation.

**Section 3: Method**  
The method is novel and well-motivated. The two-stage training (SFT + RL) is standard but appropriately applied. However, several details are unclear and could hinder reproducibility:
- How exactly are the latent representations \( z \) integrated into the base model? Algorithm 1 shows concatenation of \( E_X \) (input embeddings) and \( z \), but are they concatenated along the sequence dimension? If so, does \( z \) have a fixed length, and how is positional information handled? Appendix C mentions learnable vectors and projection layers, but the exact mechanism of combining \( H_X \) and \( z \) needs a clearer description, possibly with a diagram.
- The SFT stage uses only \((X, Y)\) pairs, not the reasoning trajectories \( R \). This is fine, but it raises the question: how does the reasoning network learn to encapsulate reasoning without any direct supervision on the latent space? The authors should discuss how the network avoids collapsing to trivial representations.
- In Equation (4), \( f_\theta \) is used instead of \( P_\theta \); this seems like a typo.
- The reinforcement learning stage uses GRPO, but the reward function is not specified. It is presumably based on answer correctness (e.g., exact match), but this should be explicitly stated.

**Section 4: Experiments**  
Experiments are comprehensive, covering multiple benchmarks (mathematical and out-of-domain) and comparisons with strong baselines. However, several issues need attention:
- **Statistical significance**: While Table 9 provides standard deviations for LRT, no significance tests or comparisons with baseline variances are given. The improvements, though consistent, are sometimes marginal (e.g., 38.00 vs. 37.75 on AMC under 512 budget). The authors should report significance tests (e.g., paired bootstrap) to confirm that differences are not due to chance.
- **Fairness of comparisons**: Comparing LRT (which fine-tunes an auxiliary network) with prompt-based methods like NoThinking is acceptable, but it should be noted that LRT requires additional training data and computation. The comparison with Qwen3’s non-thinking mode is fair because both use the same base model and budget.
- **Ablation studies**: The ablation on the number of latent tokens (Table 3) is useful, but why does performance drop when going from 256 to 512 tokens? The authors suggest larger training scales may be needed, but this is speculative. An ablation on the architecture of the reasoning network (e.g., size, initialization) is missing.
- **Efficiency metrics**: Table 7 in Appendix D.3 shows latency and throughput gains. However, the comparison is not entirely fair: LRT uses a 0.6B reasoning network, which adds computational overhead. The reported throughput when accounting for latent tokens (73.02 tokens/sec) is an “effective” metric that may not reflect real wall-clock time if the reasoning network is computationally heavy. A clearer breakdown of time spent in the reasoning network vs. the base model would be helpful.
- **Pass@4 results**: The consistent improvement in pass@4 over pass@1 suggests increased diversity, but this is not analyzed. Is the diversity due to stochasticity in the base model or the latent representation? A brief discussion would be valuable.

**Section 5: Related Work**  
The related work adequately covers chain-of-thought reasoning and efficient reasoning methods. The discussion of latent reasoning methods is brief; the appendix (Section E) provides a more detailed comparison, which should be integrated into the main text to better position the contribution.

**Writing & Clarity**  
Overall, the paper is well-structured and clearly written. However, there are some confusing points:
- In Section 3.2, the statement “Under greedy decoding, the generation of the reasoning trajectory becomes a deterministic process” is an approximation (greedy decoding can still have ties), but it is acceptable. More importantly, the transition from function \( h \) to \( G_\phi \) is abrupt: how does \( G_\phi \) learn to approximate \( h \) without autoregressive constraints? Some intuition or discussion would help.
- Algorithm 1: The notation “Embedding_θ(X)” and “HiddenStates_θ(E_X)” is ambiguous; it should be clarified that these are operations of the base model.
- Figure 1 and Figure 3 are helpful, but Figure 3’s caption is redundant with the text.

**Limitations & Broader Impact**  
The paper lacks a dedicated limitations section. Key limitations include:
- Training cost: The two-stage training requires substantial data and compute (8 A100 GPUs).
- Generalizability: The method is tested only on a few models (DeepSeek-R1 distillate and Qwen3). Would it work on other architectures (e.g., Gemini, Claude)?
- Interpretability: Latent representations are not human-readable, which may hinder debugging and trust in sensitive applications.
- The broader impact is not discussed; while the work aims at efficiency, potential negative societal impacts (e.g., enabling more powerful models with less compute) are minimal but could be briefly noted.

## Overall Assessment
The paper presents a novel and promising approach to improving the efficiency of reasoning LLMs by replacing explicit reasoning trajectories with latent representations generated by an auxiliary network. The core idea is well-motivated by an empirical analysis of redundancy, and the method is non-intrusive, allowing flexible switching between latent and explicit modes. Experiments show consistent improvements over strong baselines on multiple benchmarks, and ablation studies support design choices. However, the paper has several weaknesses: the redundancy analysis is somewhat superficial, methodological details are occasionally unclear, statistical significance of improvements is not rigorously established, and limitations are underexplored. With revisions that address these concerns—particularly clarifying the integration of latent representations, providing significance tests, and discussing limitations—the contribution would meet ICLR’s standards for novelty, rigor, and impact.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Latent Reasoning Tuning (LRT), a framework to improve the inference efficiency of reasoning LLMs by replacing explicit, token-by-token generation of chain-of-thought with compact latent representations produced by a lightweight auxiliary network. The base LLM remains frozen, and the reasoning network is trained via supervised fine-tuning and reinforcement learning to generate latent vectors that condition the LLM to produce the final answer. Experiments on mathematical and out-of-domain benchmarks show that LRT outperforms efficient reasoning baselines in accuracy under constrained token budgets and reduces latency.

### Strengths
1. **Novel and practical approach**: The idea of using a separate, trainable network to produce latent reasoning representations is innovative compared to prior work that shortens explicit chains or uses fixed prompts. The modular design allows flexible switching between latent and explicit reasoning without modifying the base LLM.
2. **Comprehensive empirical evaluation**: The paper evaluates on diverse benchmarks (GSM8K, MATH-500, AMC, LSAT, GPQA) and compares against strong baselines (NoThinking, ShorterBetter, LC-R1, Qwen3). Results consistently show accuracy improvements under token budgets, with detailed ablations on latent token count and training stages.
3. **Clear demonstration of efficiency gains**: Table 7 provides concrete measurements showing reduced latency and increased throughput compared to both thinking and non-thinking modes, making a compelling case for practical deployment.
4. **Thorough supplementary analysis**: The appendix includes valuable experiments on larger base models, inference efficiency, geometric analysis of latent representations, and statistical variance, adding depth to the claims.

### Weaknesses
1. **Superficial analysis of reasoning redundancy**: The motivation in Section 2 relies on random token/step skipping to argue redundancy. A more principled analysis (e.g., importance scoring or structural patterns) would strengthen the foundation and provide insights into what makes reasoning compressible.
2. **Missing direct comparison with latent reasoning methods**: While related work discusses latent CoT approaches (e.g., Coconut, Geiping et al.), no experimental comparison is provided in the main results. This omission makes it difficult to assess how LRT advances the state-of-the-art in latent reasoning specifically.
3. **Limited generalization evidence**: Experiments are conducted only on DeepSeek-R1-Distill-Qwen and Qwen3 series. It remains unclear whether LRT generalizes effectively to other reasoning LLMs (e.g., OpenAI o1, Gemini) or different architectural families.
4. **Increased memory overhead**: Although latency improves, Table 7 shows peak memory usage is higher than non-thinking mode due to the auxiliary network. This trade-off is not thoroughly discussed, and its impact on memory-constrained deployments is overlooked.
5. **Insufficient statistical rigor**: While Appendix D.5 reports standard deviations, the main results lack confidence intervals or significance tests. Given the stochastic nature of LLM generation, more rigorous statistical analysis (e.g., paired tests across multiple runs) is needed to substantiate the claimed improvements.

### Novelty & Significance
**Novelty**: The paper introduces a distinct instantiation of latent reasoning by employing a parallel, auxiliary network to generate fixed-length latent trajectories, differing from prior iterative latent refinement methods. The analysis of redundancy in reasoning trajectories provides a fresh empirical perspective. However, the core concept of latent reasoning is not entirely new, as acknowledged in the related work.

**Significance**: Improving the efficiency of reasoning LLMs is a high-impact problem for real-world applications. LRT demonstrates meaningful gains in accuracy and latency, offering a practical solution that balances performance and cost. The modular design enables hybrid reasoning capabilities, which could facilitate wider adoption. Nevertheless, the requirement for additional training data and compute, along with increased memory, may limit immediate scalability.

### Suggestions for Improvement
1. **Deepen the redundancy analysis**: Replace random skipping with importance-based methods (e.g., gradient saliency, attention weights) to identify critical reasoning components and provide a more principled justification for latent compression.
2. **Include direct comparisons with latent reasoning baselines**: Add experiments against recent latent reasoning methods (e.g., Coconut, Geiping et al.) to clearly demonstrate LRT's advantages in accuracy, efficiency, or training stability.
3. **Test generalization to more base models**: Evaluate LRT on other reasoning LLMs like o1-preview or Gemini Flash Thinking to show broader applicability and identify potential limitations.
4. **Enhance statistical reporting**: Perform significance tests (e.g., bootstrap confidence intervals, paired t-tests) across multiple random seeds for key comparisons, and report these in the main text or appendix.
5. **Expand discussion of limitations and trade-offs**: Explicitly address the memory overhead, interpretability loss due to latent representations, data dependency (need for reasoning trajectories), and potential failure cases for complex problems requiring long chains.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with state-of-the-art latent reasoning methods (e.g., Coconut, Geiping et al. 2025).** The paper claims a novel latent reasoning framework but only compares against methods that shorten *explicit* trajectories (e.g., ShorterBetter, LC-R1) or prompt-based tricks (NoThinking). Without a direct comparison to other works that perform reasoning in a latent space (cited in Section 5.2), the claim of novelty and superior effectiveness is not substantiated.
2. **Ablation on the necessity of the pre-trained embedding model as the reasoning network.** The reasoning network is initialized from Qwen3-Embedding-0.6B. A critical ablation is missing: what is the performance if the reasoning network is a simple MLP or randomly initialized? This is needed to verify if the performance gains come from the specific pre-trained knowledge in the embedding model or the LRT framework itself.
3. **Evaluation on tasks where explicit, verifiable reasoning is crucial (e.g., proof generation, symbolic reasoning).** The benchmarks (GSM8K, MATH, etc.) primarily evaluate answer correctness. To claim that latent representations adequately capture reasoning, tests on datasets requiring step-by-step justification or where the reasoning trace itself is the output (e.g., PrOntoQA, ProofNet) are necessary. Its absence leaves open whether LRT is learning to reason or just learning better answer shortcuts.
4. **Performance under a compute-equivalent comparison with the "thinking" mode.** The efficiency claims (Table 7) compare latency but not final performance under equal total FLOPs or time budgets. A critical experiment is to run the standard "thinking" mode with a budget matched to LRT's total compute (including the reasoner forward pass) to see if LRT truly provides a Pareto improvement.

### Deeper Analysis Needed (top 3-5 only)
1. **Causal analysis of what information in the latent representation drives the answer.** The cosine similarity analysis (Appendix D.4) is correlational. To trust that the latent vector encapsulates *reasoning*, the authors should perform interventions (e.g., perturbing specific dimensions of the latent vector) and measure the impact on specific reasoning sub-steps in the final answer generation.
2. **Failure mode analysis.** The paper only reports aggregate accuracy. A qualitative analysis of problems where LRT fails but the explicit reasoning baseline succeeds is essential to understand the limitations of the latent representation and whether it discards critical logical steps or nuances needed for hard problems.
3. **Analysis of the reward model's role in RL training.** The RL stage uses a reward based on final answer correctness. An analysis is missing on whether this reward leads the latent space to simply overfit to answer patterns or actually learns robust reasoning representations. Showing the correlation between latent space structure (e.g., clustering by problem type) and reward during training would be insightful.
4. **Sensitivity analysis of the latent token length hyperparameter.** Table 3 shows performance varies with token count, but there's no analysis explaining *why* 256 is a sweet spot for Qwen3-1.7B. Is it related to model capacity, dataset complexity, or the embedding model's architecture? This lack of understanding makes the method seem heuristic.

### Visualizations & Case Studies
1. **Side-by-side examples of explicit CoT vs. latent-conditioned generation.** For a set of problems (especially where performance differs), show the full explicit reasoning trajectory, the learned latent vectors (e.g., via PCA/t-SNE projections), and the final answer generated by LRT. This would visually demonstrate what is compressed or lost.
2. **Attention visualization from the base LLM when conditioned on the latent vector.** Showing how the attention pattern differs when the model uses a latent prefix versus an explicit reasoning prefix would help validate that the latent vectors are serving a similar conditioning role, and reveal if the model attends to them in a meaningfully structured way.

### Obvious Next Steps
1. **Adaptive generation of latent token length.** The paper uses a fixed number of latent tokens. An obvious extension is to make this dynamic (e.g., via a halting mechanism), which should have been explored as it directly addresses the core efficiency claim—why generate 256 tokens if 64 suffice for a simple problem?
2. **Applying LRT to a wider range of base models, including larger (e.g., 70B) and non-reasoning-specialist LLMs.** The experiments are limited to a few models (DeepSeek-R1-Distill-Qwen, Qwen3 series). Testing on models like Llama or Gemma is necessary to claim general applicability of the framework.
3. **Integration with speculative decoding or other inference acceleration techniques.** Since LRT generates a fixed-length latent representation in one pass, it naturally complements speculative decoding. A combined efficiency benchmark comparing LRT+speculative decoding against standard thinking modes would be a strong addition.

# Final Consolidated Review
## Summary
This paper introduces Latent Reasoning Tuning (LRT), a framework that improves the inference efficiency of reasoning LLMs by replacing explicit, token-by-token generation of reasoning chains with compact latent representations produced by a lightweight auxiliary network. The base LLM remains frozen, and the auxiliary reasoning network is trained via supervised fine-tuning and reinforcement learning to generate latent vectors that condition the LLM to produce the final answer. Experiments on mathematical and out-of-domain benchmarks show that LRT outperforms efficient reasoning baselines in accuracy under constrained token budgets and reduces latency.

## Strengths
- **Novel and practical modular design:** The method introduces a lightweight, trainable reasoning network that generates fixed-length latent representations to condition a frozen base LLM. This non-intrusive approach allows seamless switching between latent and explicit reasoning modes without modifying the base model's parameters, offering a flexible and practical solution for hybrid reasoning systems.
- **Strong empirical performance across diverse benchmarks:** The paper demonstrates consistent improvements over strong efficient reasoning baselines (NoThinking, ShorterBetter, LC-R1) and surpasses Qwen3's non-thinking mode on multiple in-domain (GSM8K, MATH-500, AMC) and out-of-domain (LSAT, GPQA) tasks under constrained token budgets. The results are supported by thorough ablation studies on latent token count and training stages.
- **Concrete efficiency gains:** The method reduces inference latency and increases throughput compared to both standard thinking and non-thinking modes, as shown in Table 7, while maintaining or improving accuracy. This provides a compelling case for real-world deployment where computational efficiency is critical.

## Weaknesses
- **Increased memory overhead:** While latency improves, the auxiliary reasoning network and the generation of latent representations increase peak memory usage compared to a standard non-thinking mode (Table 7). This trade-off is not deeply analyzed and could impact deployment in memory-constrained environments.
- **Limited direct comparison with other latent reasoning methods:** The paper positions itself within the latent reasoning literature but does not include experimental comparisons with contemporary latent reasoning approaches (e.g., Coconut, Geiping et al. 2025). This omission makes it difficult to assess the specific advantages of LRT's parallel, auxiliary-network design over other latent reasoning paradigms.
- **Statistical reporting could be more rigorous:** Although Appendix D.5 reports standard deviations, the main results lack confidence intervals or statistical significance tests. Given the stochastic nature of LLM generation, more rigorous statistical analysis (e.g., paired bootstrap tests across multiple runs) would strengthen the claims of improvement.

## Nice-to-Haves
- A deeper analysis of what specific information the latent representations encode (beyond the cosine similarity analysis in the appendix) and how they interact with the base model's attention mechanisms could provide more insight into the method's inner workings.
- Testing the framework on a broader set of base model architectures (beyond DeepSeek-R1-Distill-Qwen and Qwen3) would help demonstrate generalizability.

## Novel Insights
The paper's core novel insight is that explicit, token-by-token reasoning trajectories in slow-thinking LLMs contain substantial redundancy, and this reasoning process can be effectively compressed into a fixed-length, non-autoregressive latent representation generated by a separate network. This allows the base model's reasoning capability to be preserved and even enhanced while drastically reducing the sequential generation cost. The method is distinct from prior latent reasoning approaches that typically require iterative refinement of a recurrent state or retraining of the base model; instead, LRT keeps the base model frozen and uses a parallel, auxiliary network to produce the entire latent reasoning trajectory in one forward pass, enabling a modular and switchable hybrid reasoning system.

## Suggestions
- Include a direct experimental comparison with at least one state-of-the-art latent reasoning baseline (e.g., Coconut) to clearly delineate the performance and efficiency advantages of the proposed auxiliary-network approach.
- Expand the discussion of limitations to explicitly address the memory overhead trade-off, the interpretability loss due to non-textual latent representations, and the dependency on training data containing reasoning trajectories.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 8.0, 4.0]
Average score: 5.0
Binary outcome: Accept
