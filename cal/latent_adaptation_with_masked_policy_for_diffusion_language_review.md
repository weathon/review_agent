=== CALIBRATION EXAMPLE 40 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the contribution (LAMP: Latent Adaptation via Masked Policy). The abstract succinctly states the problem (underexplored test-time reasoning in dLLMs), method (training-free, reward-guided policy-gradient updates on sparse token latents, clamp-and-inpaint), and results (improvements on GSM8K, MATH-500, AIME across LLaDA and Dream). Claims are supported with specific benchmarks and models. However, the abstract mentions "modest compute" without quantification—this is later addressed in the method but could be more precise.

### Introduction & Motivation
The introduction effectively motivates the problem: autoregressive LLMs have limitations for multi-step reasoning due to sequential decoding, while diffusion LMs offer parallel, revisable decoding but their reasoning capabilities are underexplored. It reviews relevant work on dLLMs and test-time strategies, identifying a gap for diffusion-specific adaptation. Contributions are clearly listed (LAMP framework, diffusion-specific loop design, experimental gains). The narrative sets up the paper well.

### Method / Approach
Section 2 provides a detailed description of LAMP, including preliminaries, overview, reward models, and latent policy adaptation. The method is algorithmic (Algorithm 1) and appears reproducible. Key ideas: baseline decode, low-confidence token selection, policy-gradient updates on hidden states (treated as editable latents), clamp-and-inpaint decoding. The reward models (self-reward and Perfect Sparse Reward Model/PSRM) are clearly defined.

**Major concerns:**
1. **Lack of detail on constrained diffusion:** The "clamp-and-inpaint" step relies on a "constrained diffusion pass," but the mechanics (e.g., how tokens are clamped, how the diffusion sampler handles fixed tokens) are not specified. This is crucial for reproducibility and understanding how edits propagate.
2. **Gradient estimation and stability:** The policy-gradient update (Equation 4) uses REINFORCE on discrete samples from categorical policies parameterized by continuous latents. While a moving baseline and trust-region regularization are mentioned, the exact form of regularization \(R_{\text{stab}}\) is not detailed in the main text (though pseudo-code in the appendix shows KL and L2 penalties). High variance could be an issue; more discussion on stability is needed.
3. **Practicality of PSRM:** The primary reward (PSRM) requires ground-truth answers at test time, making it an oracle setup. This limits real-world applicability. The self-reward alternative yields only modest gains, raising questions about how effective LAMP would be with noisy or learned rewards.
4. **Token selection rationale:** Selecting tokens based on low confidence (max probability or margin) may not always align with tokens critical for correctness. Ablation studies on selection criteria are missing from the main text.
5. **Computational overhead:** The claim of "modest compute" is not quantified. Test-time gradient updates and multiple diffusion passes add overhead; a comparison of inference time vs. baseline would strengthen the claim.

### Experiments & Results
Experiments cover three math reasoning benchmarks (GSM8K, MATH-500, AIME2024) and three dLLM backbones (LLaDA, LLaDA-1.5, Dream). Table 1 shows consistent gains with PSRM (e.g., +13.3 on GSM8K for LLaDA) but modest gains with self-reward. Figure 2 illustrates scaling with adaptation iterations, and Figure 3 analyzes self-reward transitions. Qualitative examples (Table 9) provide insight.

**Major concerns:**
1. **Missing comparisons to other test-time scaling methods:** The paper mentions inference-time scaling techniques for dLLMs (e.g., particle Gibbs, remasking, search) in the related work but does not compare LAMP against them. This is a critical omission for evaluating the contribution's novelty and effectiveness relative to existing approaches.
2. **Over-reliance on oracle reward:** PSRM uses ground-truth answers, which is not realistic for deployment. The paper does not explore using a learned reward model or more practical supervision, weakening the practical impact.
3. **Insufficient ablation studies:** The paper claims that "ablations confirm that diffusion-specific ingredients... are essential," but these are not presented in the main text. Without ablations (e.g., on token selection, reward design, clamp-and-inpaint), it is unclear which components are necessary.
4. **Lack of computational cost analysis:** No data on extra inference time or memory overhead is provided. For a test-time adaptation method, this is essential to assess efficiency.
5. **Statistical significance and robustness:** No statistical tests are reported, and hyperparameters are fixed across experiments without sensitivity analysis (e.g., edit budget \(k\), learning rate \(\eta\), confidence thresholds). The robustness of LAMP to these choices is unclear.
6. **Self-reward transition analysis** is descriptive but lacks a quantitative summary of net improvement vs. regression rates.

### Writing & Clarity
The paper is well-structured and clearly written. Figures and tables are informative. Algorithm 1 and the appendix pseudo-code enhance reproducibility. Minor issues: Equation 4 notation could be more explicit (e.g., clarifying the gradient w.r.t. \(z_i\)), and some terms like "constrained diffusion" need elaboration.

### Limitations & Broader Impact
The conclusion and future work discuss limitations: reliance on sparse outcome-based rewards, potential for process supervision, extension to interactive settings. The ethics and reproducibility statements are appropriate. However, the paper does not adequately address:
- The impracticality of PSRM in real-world scenarios and the need for learned reward models.
- Performance on non-reasoning tasks (e.g., open-ended generation).
- Sensitivity to hyperparameters and failure modes when base model confidence is low.
- Environmental impact of extra computation is mentioned but not quantified.

## Overall Assessment
The paper introduces a novel training-free framework (LAMP) for reward-guided latent adaptation in diffusion language models, leveraging policy-gradient updates on sparse token latents and clamp-and-inpaint decoding. The core idea is innovative and well-motivated, with experiments demonstrating substantial improvements when using an oracle reward (PSRM) on math reasoning benchmarks across multiple dLLM backbones. However, the contribution is significantly undermined by several weaknesses: (1) lack of comparison to existing inference-time scaling methods for dLLMs, leaving the reader unsure of LAMP's relative advantage; (2) reliance on ground-truth rewards limits practical relevance, and the self-reward version yields only modest gains; (3) insufficient ablation studies and computational overhead analysis; (4) missing sensitivity analysis and statistical rigor. While the method is technically sound and the writing is clear, these issues impact the paper's readiness for ICLR. Addressing these concerns—particularly by adding comparisons, exploring more realistic reward settings, and providing ablations—would strengthen the paper considerably.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces LAMP, a training-free framework for reward-guided latent adaptation in masked diffusion language models (dLLMs). LAMP identifies low-confidence token positions, applies sparse policy-gradient updates to their hidden states using either a lightweight self-reward or a Perfect Sparse Reward Model (PSRM), and then uses clamp-and-inpaint decoding to propagate edits globally. Experiments on mathematical reasoning benchmarks (GSM8K, MATH-500, AIME) demonstrate consistent accuracy improvements across multiple dLLM backbones (LLaDA, Dream), particularly when using PSRM supervision.

### Strengths
1. **Training-Free Efficiency**: The method requires no model retraining or fine-tuning, operating purely at inference time with modest computational overhead (only a few gradient steps on a sparse set of latents). This is evidenced by the reported experimental setup and runtime comparisons.
2. **Consistent Empirical Gains**: With PSRM supervision, LAMP achieves substantial improvements (e.g., +13.3 points on GSM8K for LLaDA, +16.0 points on MATH-500) across multiple models and datasets, as shown in Table 1. The gains are robust and demonstrate the potential of reward-guided latent adaptation.
3. **Innovative Use of Diffusion Properties**: The clamp-and-inpaint decoding effectively leverages the bidirectional, parallel nature of masked diffusion models to maintain global coherence after local edits. The method is specifically designed for diffusion's unique inference characteristics (parallel scoring, constrained infilling), as motivated in Sections 2.1 and 2.2.
4. **Thorough Analysis**: The paper includes insightful ablation studies (e.g., scaling behavior in Figure 2, reward transition dynamics in Figure 3, and qualitative examples in Table 9) that validate design choices and provide nuanced understanding of when and why the method works or fails.

### Weaknesses
1. **Dependence on Ground-Truth Reward (PSRM)**: The most significant gains require PSRM, which is an oracle reward based on the ground-truth answer. This limits real-world applicability where such supervision is unavailable. The self-reward variant yields only modest and inconsistent improvements (Table 1), and even causes regressions on some tasks (e.g., AIME).
2. **Limited Exploration of Reward Design**: The self-reward signals are simple, rule-based checks (format, consistency). The paper does not explore more sophisticated or learned reward models (e.g., process supervision, verifiers) that could bridge the gap between self-reward and PSRM performance.
3. **Instability and Regressions**: Qualitative analysis (Table 9) shows cases where edits degrade correct answers (TRUE→FALSE transitions), indicating that the adaptation can sometimes break global reasoning consistency. While confidence gating is used to mitigate this, the phenomenon persists.
4. **Incomplete Comparison to Inference-Time Baselines**: While related work is discussed, the empirical comparison is primarily against vanilla diffusion decoding. A more direct comparison to other inference-time scaling methods for dLLMs (e.g., particle Gibbs sampling, search-based strategies) would better contextualize LAMP's advantages and trade-offs.
5. **Theoretical Justification is Light**: The connection between the policy-gradient updates and optimizing a reward-weighted posterior is mentioned but not deeply analyzed. The stability and convergence properties of the latent updates are not formally examined.

### Novelty & Significance
**Novelty**: The work introduces a novel, training-free adaptation paradigm specifically tailored for masked diffusion LMs. While latent optimization for test-time reasoning has been explored in autoregressive models (e.g., LatentSeek), adapting this idea to the non-sequential, bidirectional setting of diffusion models is a meaningful contribution. The clamp-and-inpaint mechanism is a clever diffusion-specific innovation.

**Significance**: The results convincingly demonstrate that reward-guided latent adaptation is a viable and effective axis for improving dLLM reasoning, complementing existing approaches like prompt engineering or inference-time scaling via sampling. This could inspire further research into test-time optimization for diffusion models. However, the reliance on PSRM for large gains tempers the immediate practical significance.

### Suggestions for Improvement
1. **Explore Richer Reward Signals**: Investigate learned verifiers, process supervision, or LLM-based critique models as rewards to reduce dependency on ground-truth answers while maintaining stronger signal than simple self-reward.
2. **Benchmark Against More Baselines**: Include explicit comparisons to state-of-the-art inference-time methods for dLLMs (e.g., ReMDM, particle Gibbs sampling) on the same tasks to better establish LAMP's relative performance and efficiency.
3. **Strengthen Theoretical Grounding**: Provide a more formal analysis linking the policy-gradient updates to Bayesian inference or distributional shift, and discuss the optimization landscape (e.g., local minima, gradient stability).
4. **Expand Task Diversity**: Evaluate LAMP on a broader range of reasoning tasks (e.g., code generation, logical deduction, scientific QA) to assess its generalizability beyond mathematics.
5. **Address Instability**: Propose and test more robust mechanisms (e.g., better confidence estimation, ensemble over edits, backtracking) to reduce TRUE→FALSE regressions, potentially using the transition analysis as a diagnostic tool.
6. **Clarify Computational Cost**: Provide a clearer breakdown of the added latency (wall-clock time) and memory overhead compared to baseline decoding, as "modest compute" is claimed but not quantified.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to a simple "sample-and-select" baseline.** The core claim is that latent adaptation is an effective test-time scaling axis. This is undermined without showing that LAMP outperforms the trivial baseline of running the vanilla model multiple times and selecting the best output according to the same reward (PSRM). This is a standard sanity check for any inference-time optimization method.
2. **Ablation on the necessity of gradient updates.** The method mixes gradient-based latent updates with clamp-and-inpaint decoding. An experiment is needed where low-confidence tokens are simply re-masked and re-sampled (i.e., more diffusion steps) without gradient updates, to isolate the benefit of the policy-gradient component versus just more compute spent on diffusion refinement.
3. **Evaluation with a learned reward model, not an oracle.** The most impressive gains use a Perfect Sparse Reward Model (PSRM), which is an oracle using the ground-truth answer. This is not a practical setting. The paper must show results using a *learned* reward model (e.g., a verifier) to demonstrate the method's utility in realistic scenarios where the answer is unknown.
4. **Test on non-mathematical, open-ended generation tasks.** The claims about enhancing "diffusion-based reasoning" are only tested on closed-form math problems. Performance on tasks like code generation, long-form QA, or creative writing is necessary to argue for the general applicability of the method.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what is being edited.** The paper lacks a clear analysis of *which tokens* are typically selected for editing (e.g., numbers, operators, reasoning keywords) and how the edits propagate through the reasoning chain. Without this, it's unclear if LAMP improves the logical reasoning or just corrects superficial answer tokens.
2. **Sensitivity analysis of hyperparameters (k, K, η).** The method has several key hyperparameters: the edit budget `k`, the number of adaptation steps `K`, and the learning rate `η`. The paper provides default values but no analysis of their sensitivity or how performance degrades with suboptimal choices, which is critical for reproducibility and understanding the method's robustness.
3. **Quantification of compute overhead.** The claim of "modest compute" is not substantiated. A direct comparison of FLOPs or wall-clock time versus the vanilla model and versus simple sampling baselines is missing. This is essential for evaluating the practical efficiency of the proposed test-time scaling.
4. **Breakdown of performance by problem type/difficulty.** The aggregate results hide whether improvements come from easier or harder problems. An analysis correlating gains with initial model confidence or problem complexity would reveal the method's true capabilities and limitations.

### Visualizations & Case Studies
1. **Visualization of the latent edit trajectory.** For several cases, plot how the selected token latents evolve across gradient steps (e.g., via PCA) and how their associated token distributions change. This would visually demonstrate whether the gradient steps are making meaningful, monotonic improvements or chaotic jumps.
2. **Side-by-side case studies of successful and failed adaptations.** The paper shows only two brief examples. A more systematic presentation of 5-10 diverse examples is needed, showing the initial output, the edited latent positions, the intermediate provisional sequences, and the final output. This would expose failure modes (e.g., breaking coherent reasoning) and successful correction patterns.
3. **Attention map comparison before and after clamping.** To support the claim that "bidirectional re-inpainting propagates local edits globally," visualize attention patterns in the final decode for a key example, comparing the vanilla run to the LAMP run with clamped tokens. This would provide mechanistic evidence for how the edit propagates.

### Obvious Next Steps
1. **Compare to inference-time scaling methods for autoregressive models.** The related work mentions LatentSeek for AR models. A direct comparison on the same benchmarks (using comparable model sizes and compute budgets) is necessary to position LAMP's contribution within the broader field of test-time adaptation, not just among dLLM methods.
2. **Investigate the use of process reward.** The future work mentions process supervision, but a preliminary experiment using step-by-step correctness (e.g., on datasets with intermediate solution steps) should have been included to show the potential beyond sparse outcome rewards.
3. **Run experiments on a wider variety of diffusion LM architectures.** The paper tests LLaDA and Dream. Testing on another major family (e.g., the "Mercury" model cited) would strengthen the claim of general applicability across masked diffusion LMs.
4. **Analyze the variance of improvements.** Report standard errors or confidence intervals for the accuracy improvements. The gains with PSRM are large, but it's unclear how stable they are across different random seeds or prompt variations.

# Final Consolidated Review
## Summary
This paper introduces LAMP, a training-free framework for reward-guided latent adaptation in masked diffusion language models (dLLMs). LAMP performs sparse policy-gradient updates on token-level hidden states and uses clamp-and-inpaint decoding to propagate edits globally. Experiments on math reasoning benchmarks show significant accuracy improvements when using an oracle reward (PSRM), but only modest gains with a lightweight self-reward.

## Strengths
- **Training-free efficiency**: LAMP operates entirely at inference time without model retraining, applying gradient updates only to a small subset of token latents, which aligns with the goal of test-time adaptation.
- **Consistent empirical gains with PSRM**: Using the Perfect Sparse Reward Model (PSRM), LAMP achieves substantial improvements across multiple dLLM backbones (e.g., +13.3 points on GSM8K for LLaDA, +16.0 points on MATH-500), demonstrating the potential of reward-guided latent optimization.
- **Innovative use of diffusion properties**: The clamp-and-inpaint mechanism leverages the bidirectional, parallel nature of masked diffusion models to maintain global coherence after local edits, a design specifically tailored for dLLMs.

## Weaknesses
- **Dependence on oracle reward**: The primary results rely on PSRM, which requires ground-truth answers at test time, severely limiting practical applicability in real-world scenarios. The self-reward variant yields only modest and inconsistent improvements.
- **Lack of comparison to existing inference-time methods**: The paper does not empirically compare LAMP to other test-time scaling techniques for dLLMs (e.g., particle Gibbs sampling, remasking strategies mentioned in related work), making it unclear whether LAMP offers a tangible advantage over prior approaches.
- **Insufficient ablation studies**: While the paper claims that ablations confirm the necessity of diffusion-specific components, these are not presented in the main text, leaving the contribution of individual elements (e.g., token selection, gradient updates, clamp-and-inpaint) uncertain.
- **Computational overhead not quantified**: The claim of "modest compute" is unsupported by data on added latency, memory overhead, or FLOPs compared to baseline decoding, which is critical for evaluating the efficiency of a test-time adaptation method.

## Nice-to-Haves
- Sensitivity analysis of key hyperparameters (e.g., edit budget, learning rate, confidence thresholds) to demonstrate robustness.
- Exploration of more realistic reward signals, such as learned verifiers or process supervision, to bridge the gap between self-reward and PSRM performance.
- Evaluation on a broader range of tasks (e.g., code generation, open-ended QA) to assess generalizability beyond mathematical reasoning.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Self-reward transition analysis lacks quantitative summary**: The paper includes Figure 3 with quantitative transition matrices, so this criticism is factually incorrect.
- **Demand for statistical significance tests**: While reporting variance could strengthen the paper, single-run evaluation on standard benchmarks is common in the field; this is not a core flaw.
- **Criticism of missing details on constrained diffusion**: The clamp-and-inpaint mechanism is described in Section 2.2, Algorithm 1, and appendix pseudo-code, though deeper exposition could be beneficial.

## Novel Insights
The paper demonstrates that reward-guided latent optimization is a viable axis for improving reasoning in diffusion language models, uniquely leveraging their bidirectional denoising process. The clamp-and-inpaint decode is a novel mechanism for integrating local edits into globally coherent sequences, highlighting how diffusion's revisable nature can be harnessed for targeted test-time adaptation.

## Suggestions
- Add empirical comparisons to state-of-the-art inference-time scaling methods for dLLMs (e.g., ReMDM, particle Gibbs sampling) on the same benchmarks to contextualize LAMP's performance.
- Provide a clear breakdown of computational overhead (e.g., wall-clock time, memory usage) relative to vanilla decoding and simple sampling baselines.
- Include ablation studies in the main text to validate the necessity of each core component (sparse selection, policy-gradient updates, clamp-and-inpaint).

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
