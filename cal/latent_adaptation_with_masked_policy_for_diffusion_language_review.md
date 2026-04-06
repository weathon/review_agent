=== CALIBRATION EXAMPLE 35 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title "Latent Adaptation with Masked Policy for Diffusion Language Models" accurately reflects the core contribution. The abstract clearly states the method (LAMP), its key features (training-free, reward-guided latent updates, clamp-and-inpaint), and the main empirical finding (consistent improvements on reasoning benchmarks). The claim that "test-time reasoning has been little explored" for dLLMs is fair and sets up the motivation. The abstract is well-supported by the rest of the paper.

**Introduction & Motivation**
The introduction effectively contrasts autoregressive and diffusion LMs, establishing the unique opportunities (parallel, revisable decoding) and the underexplored nature of test-time reasoning for dLLMs. It correctly cites recent relevant work (Diffusion-of-Thoughts, inference-time scaling methods). The contributions are listed clearly and align with the paper's content. The problem is well-motivated within the growing field of dLLMs.

**Method / Approach**
The core idea is novel and well-adapted to the diffusion setting: performing lightweight policy-gradient updates on a sparse set of token latents and using the model's own clamp-and-inpaint mechanism for global coherence. The formulation of editable latents and the policy-gradient update (Eq. 4) is clear.
*   **Logical Gaps & Assumptions:** A key assumption is that gradient updates on isolated hidden states (`z_i`) will produce meaningful, composable token distributions when decoded. The method relies on the model's ability to inpaint coherently around these locally edited latents. While the results support this, the paper could more deeply discuss the validity of this assumption or its potential failure modes (e.g., when edits create inconsistent contexts).
*   **Reward Design:** The dual reward design is sensible. However, the **Perfect Sparse Reward Model (PSRM)** is essentially an oracle using the ground-truth answer. Its use is justified for a proof-of-concept but significantly limits the practical applicability of the method, as noted in the limitations. The description of "self-reward" is vague (e.g., "format or consistency checks"); Appendix B provides some detail but it remains a relatively weak, heuristic component.
*   **Reproducibility:** Algorithm 1 and the pseudo-code in Appendix C provide a good high-level overview. However, crucial details for exact reproduction are missing: How is the "moving baseline `b`" computed and updated (line 7 of Alg. 1)? The pseudo-code shows a simple exponential moving average, but the algorithm text does not specify this. The exact mechanism for `CONSTRAINEDDIFFUSE` (how clamped tokens are fixed during the diffusion steps) is also not detailed, though it's implied to use the model's native capability.

**Experiments & Results**
*   **Main Results (Table 1):** The results are compelling. The large gains with PSRM (+10-20 points) robustly demonstrate the potential of the latent adaptation concept. The modest/fluctuating gains with self-reward honestly show its limitations. Testing across multiple dLLM backbones (LLaDA, Dream) strengthens the claim of generality.
*   **Baselines:** The primary baseline is "Vanilla DLM," which is fair. However, for ICLR, a more thorough comparison with **other test-time scaling methods for dLLMs** is expected. The related work cites several (ReMDM, particle Gibbs, classical search). The paper would be significantly strengthened by including 1-2 strong, recent baselines from this set to contextualize LAMP's compute/performance trade-off. The claim that LAMP "complements existing inference-time scaling methods" is not tested.
*   **Ablations:** The ablations mentioned in the contributions ("sparse selection, reward choice, and clamp-and-inpaint—are essential") are not presented in a dedicated section or table. Quantifying the impact of each component (e.g., what happens if you edit random tokens, or skip clamp-and-inpaint) is necessary to support these claims rigorously.
*   **Analysis Figures:** Figure 2 (scaling) and Figure 3 (reward transitions) are excellent and provide valuable insights into the dynamics of the method. The qualitative analysis (Table 9/§3.5) is also very good, showing both successes and failure modes.
*   **Compute Overhead:** The claim of "modest compute overhead" is plausible given the sparse updates but is not quantified. Reporting the average increase in forward passes or wall-clock time relative to a vanilla decode would be important.

**Writing & Clarity**
The paper is generally well-written. The structure is logical. Some sections could be clearer:
*   Section 2.4, "Latent Policy Adaptation": The connection between optimizing the intractable posterior `p*(y)` and the practical REINFORCE update on `z` is slightly abrupt. A sentence or two elaborating on this variational perspective would help.
*   The description of confidence gating and the final decode is clear.
*   The tables in the appendix (C, D) are helpful for reproducibility.

**Limitations & Broader Impact**
The conclusion and future work section thoughtfully discusses limitations: the reliance on outcome-based rewards (PSRM) and the potential for richer process supervision. The ethics statement is appropriate. A critical limitation **not sufficiently addressed** is the **practical utility of the method in real-world scenarios without ground-truth answers**. While PSRM demonstrates upper-bound potential, the self-reward results are weak. The paper should more forcefully discuss this gap and the challenge of obtaining high-quality, dense reward signals in practice. The broader impact statement is standard and reasonable.

### Overall Assessment
This paper presents a novel and compelling idea: leveraging the bidirectional, revisable nature of diffusion language models for training-free, reward-guided latent optimization. The core contribution is solid, and the empirical demonstration using a perfect reward (PSRM) is strong and clearly shows the method's potential. However, the paper has significant weaknesses for an ICLR acceptance in its current form. The practical applicability is limited by the reliance on ground-truth rewards, the comparison to other test-time scaling methods is lacking, and key ablation studies are omitted. Addressing these issues—particularly by adding strong baselines, detailed ablations, and a more nuanced discussion of the reward practicality—is essential for the paper to meet the high bar of ICLR. The underlying idea is promising and likely worthy of publication if these concerns are adequately resolved.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces LAMP (Latent Adaptation via Masked Policy), a training-free, instance-level adaptation method for masked diffusion language models (dLLMs). LAMP performs sparse policy-gradient updates on the hidden states of low-confidence tokens, guided by reward signals (self-reward or a Perfect Sparse Reward Model), and then uses a clamp-and-inpaint decode to propagate edits globally. The method consistently improves reasoning accuracy on mathematical benchmarks (GSM8K, MATH-500, AIME) across multiple dLLM backbones (LLaDA, LLaDA-1.5, Dream) with modest compute overhead.

### Strengths
1. **Novel adaptation of RL techniques to diffusion LMs:** The paper creatively applies policy-gradient updates to editable token latents within a diffusion decoding process, leveraging the bidirectional, parallel nature of dLLMs. This is a fresh take on test-time adaptation, distinct from autoregressive methods.
2. **Comprehensive and rigorous experimentation:** The evaluation spans three challenging math reasoning benchmarks and three modern dLLM architectures. The results are substantial, with double-digit accuracy gains when using the Perfect Sparse Reward Model (PSRM), convincingly demonstrating the method's effectiveness. Ablation studies and scaling analyses are included.
3. **Practical and efficient design:** LAMP is training-free and adds minimal inference overhead by editing only a sparse set (≈10%) of tokens. The "clamp-and-inpaint" mechanism is a diffusion-native way to ensure global coherence after local edits, making the approach computationally feasible.

### Weaknesses
1. **Heavy reliance on the Perfect Sparse Reward Model (PSRM):** The most significant gains require PSRM, a binary oracle that checks final answer correctness. This limits the method's real-world applicability where ground truth is unavailable. The self-reward variant yields only modest, inconsistent improvements, as the paper openly acknowledges.
2. **Limited analysis of scalability and broader impact:** While test-time scaling via iteration is studied, the compute cost (e.g., total forward passes vs. baseline) is not quantified in detail. Furthermore, the evaluation is confined to mathematical reasoning; it remains unclear if LAMP generalizes to other reasoning domains (e.g., code, commonsense) or if its benefits persist in much larger models.
3. **Incomplete comparison to the state-of-the-art:** The baselines are "vanilla" dLLMs. A comparison to other inference-time scaling methods for dLLMs (e.g., ReMDM, particle Gibbs sampling) or a strong autoregressive model with chain-of-thought/self-consistency is missing, making it hard to gauge LAMP's relative advancement.

### Novelty & Significance
**Novelty:** The core idea of performing instance-level, reward-guided policy-gradient updates on token latents within a diffusion process is novel. It effectively bridges concepts from reinforcement learning and discrete diffusion models for language. The "clamp-and-inpaint" decode is a clever, diffusion-specific mechanism for integrating edits.
**Significance:** The work successfully demonstrates that the latent states in dLLMs are a viable and effective substrate for test-time optimization, opening a new axis for improving reasoning without retraining. For the growing community working on diffusion language models, this provides a practical tool and a new perspective on leveraging their unique bidirectional dynamics.

### Suggestions for Improvement
1. **Investigate learned or process-based rewards:** To move beyond PSRM, future work should explore training a lightweight verifier or using process supervision (evaluating reasoning steps) to provide a more generally applicable reward signal. An initial experiment in this direction would greatly strengthen the paper's contribution.
2. **Conduct a more thorough comparative analysis:** Add comparisons to other inference-time scaling methods for dLLMs and a strong autoregressive baseline (e.g., LLaMA with self-consistency). This would better position LAMP within the current landscape and clarify its unique advantages.
3. **Expand the evaluation scope:** Test LAMP on non-mathematical reasoning tasks (e.g., logical deduction, code generation) to demonstrate the generality of the latent adaptation principle. Additionally, a deeper analysis of failure cases, especially the TRUE→FALSE regressions under self-reward, could yield insights for making edits more robust.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare LAMP against existing inference-time scaling methods for dLLMs (e.g., particle Gibbs, classical search, ReMDM).** Without this, it’s unclear whether the reported gains are novel or simply match prior techniques, undermining the claim of a new practical axis for improvement.
2. **Ablate the sparsity level (edit budget) beyond the fixed 10%.** The claim that sparse editing is essential requires showing how performance varies with different budgets (1%, 5%, 20%, 50%) to justify the design choice and demonstrate sensitivity.
3. **Include a baseline that applies gradient updates to all tokens (or a random subset).** To validate that low-confidence selection is critical, compare against a dense-update variant; otherwise, the benefit of sparsity is unsubstantiated.
4. **Evaluate on non-mathematical reasoning tasks (e.g., logical deduction, commonsense QA).** The paper only tests math problems, so claims about enhancing “reasoning” in diffusion LMs lack evidence for generalizability.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze what the latent updates actually change in hidden states.** Without probing whether updates move latents toward specific token embeddings or alter distributions, the mechanism of “reward-guided latent optimization” is not validated.
2. **Systematically categorize failure modes and success patterns.** The qualitative section shows isolated cases; a quantitative breakdown of error types (e.g., arithmetic, logical, unit errors) fixed/introduced by LAMP is needed to understand its limitations.
3. **Assess sensitivity to key hyperparameters (learning rate, trust-region coefficients).** Using fixed values across models/datasets leaves open whether gains are robust or due to brittle tuning; a sensitivity analysis is required for reproducibility.
4. **Examine the impact of reward sparsity/density.** The paper contrasts self-reward and PSRM but does not explore dense (e.g., per-step) rewards, which could better illuminate the trade-offs and potential for process supervision.

### Visualizations & Case Studies
1. **Visualize token confidence heatmaps and edit locations across the sequence.** This would reveal whether LAMP consistently selects semantically critical tokens (e.g., numbers, operators) or if edits are arbitrary, questioning the selection heuristic.
2. **Show diffusion probability trajectories for key answer tokens before/after adaptation.** Tracking how token logits evolve over diffusion steps with and without LAMP would demonstrate how clamp-and-inpaint propagates changes globally.
3. **Provide case studies on complex MATH/AIME problems with multi-step reasoning.** The current examples are mostly from GSM8K; harder cases would better test if LAMP can correct deeper logical errors beyond arithmetic slips.

### Obvious Next Steps
1. **Compare directly with LatentSeek (AR latent optimization) on the same benchmarks.** The paper cites it as inspiration but does not show whether LAMP’s diffusion-specific approach offers advantages over AR-based latent editing.
2. **Quantify computational overhead (wall-clock time, extra FLOPs) precisely.** Claiming “modest compute” is insufficient; reporting actual slowdowns relative to baseline decoding is essential for assessing practicality.
3. **Experiment with learned reward models (e.g., verifiers) instead of perfect oracle.** Using PSRM is unrealistic for real-world deployment; testing with a trained verifier would show if LAMP works without ground-truth access.

# Final Consolidated Review
## Summary
LAMP is a training-free framework that performs reward-guided latent optimization in masked diffusion language models. It identifies low-confidence tokens, applies sparse policy-gradient updates to their hidden states, and uses a clamp-andpaint decode to propagate edits while maintaining global coherence. The method yields substantial accuracy gains on mathematical reasoning benchmarks when using a perfect reward signal, demonstrating that token-level latent adaptation is a viable axis for improving diffusion-based reasoning without retraining.

## Strengths
- **Novel adaptation of reinforcement learning to diffusion decoding.** The paper introduces a non-obvious but natural extension: treating hidden token states in a diffusion model as editable latents and applying per-instance policy-gradient updates. This creatively leverages the parallel, revisable nature of diffusion decoding, distinguishing it from autoregressive latent optimization methods.
- **Robust empirical demonstration of the core concept.** Using a perfect sparse reward (PSRM), LAMP achieves large, consistent improvements (e.g., +13–20 points on GSM8K, +16–17 on MATH-500) across multiple diffusion backbones (LLaDA, LLaDA-1.5, Dream). The scaling analysis shows smooth improvement with iteration, and the failure-mode analysis (TRUE→FALSE regressions) honestly examines limitations.

## Weaknesses
- **Heavy reliance on ground-truth reward for strong results.** The most impressive gains require a Perfect Sparse Reward Model (PSRM), which is a binary oracle using the ground-truth answer. The self-reward variant yields only modest, inconsistent improvements. This severely limits the method's practical applicability in real-world settings where ground truth is unavailable.
- **Lack of comparison to contemporary inference-time scaling methods for diffusion LMs.** The paper only compares against a vanilla diffusion decode. To contextualize LAMP's contribution and compute-performance trade-off, it should be compared against other test-time scaling techniques for diffusion LMs (e.g., ReMDM, particle Gibbs sampling, classical search) cited in the related work.
- **Insufficient ablation to justify key design choices.** The paper claims that sparse low-confidence selection, clamp-and-inpaint, and the specific reward are "essential," but does not provide systematic ablations (e.g., varying edit sparsity, updating random vs. low-confidence tokens, or removing clamp-and-inpaint). This undermines confidence that each component is necessary.

## Nice-to-Haves
- Quantify the computational overhead (e.g., additional forward passes or wall-clock time) to substantiate the "modest compute" claim.
- Explore the method's applicability to non-mathematical reasoning tasks (e.g., code generation, logical deduction) to assess generality.
- Conduct a sensitivity analysis of key hyperparameters (learning rate, trust-region coefficients) to demonstrate robustness.

## Novel Insights
The paper convincingly demonstrates that token-level latent states in masked diffusion models are a viable and effective substrate for test-time optimization. Unlike autoregressive models where latent optimization is constrained by causal structure, diffusion's bidirectional dynamics allow local edits to be harmonized globally via clamp-and-inpaint. The stark performance gap between self-reward and perfect reward highlights that reward quality, not just the adaptation mechanism, is the primary bottleneck for practical deployment. This insight opens a new research direction: designing dense, process-level rewards that can approach the upper bound set by perfect outcome supervision.

## Suggestions
- Add a comparison against at least one recent inference-time scaling method for diffusion LMs (e.g., ReMDM or particle Gibbs sampling) to clarify LAMP's relative advantage.
- Include a dedicated ablation table or section quantifying the contribution of each component (sparse selection, clamp-and-inpaint, reward type) to validate the design.
- Discuss concrete pathways to obtain stronger reward signals without ground truth (e.g., training a verifier, using process supervision) to address the practicality limitation.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
