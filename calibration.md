=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary
The paper proposes GOC (Gradient Order Combination), a new optimization method for convex quadratic problems that combines gradients with products of Hessian matrices. The authors interpret steepest descent as "first-order" and CBB as "second-order" based on the parameter r (reciprocal of step length), and extend this framework to construct "higher-order" methods using combinations like g + 3Ag/r + A²g/r² in the update formula.

## Strengths
- **Geometric insight**: The paper provides a novel geometric interpretation connecting SD and CBB methods through the parameter r, showing how CBB can be viewed as a symmetric reflection in the Ag direction (Figure 1, Equations 17-18).
- **Hessian-free implementation**: Algorithm 1 computes Hessian-vector products using finite differences of gradients (g_k - g_k¹)/d and (g_k - 2g_k¹ + g_k²)/d², avoiding explicit Hessian construction, which is practical for large-scale problems.
- **Empirical iteration reduction**: On the 100,000-dimensional quadratic problem, GOC converges in fewer iterations than CBB (1864 vs 3194) and BB (4930) for the fixed-initialization case.

## Weaknesses
- **Misleading terminology**: The use of "order" (first-order, second-order, third-order) conflicts with standard optimization terminology where order refers to derivative information used (gradient-based vs. Hessian-based). This creates unnecessary confusion and obscures the actual contribution.
- **Unfair computational cost comparison**: Each GOC iteration requires 3 gradient evaluations (g_k, g_k¹, g_k² in Algorithm 1). A fair comparison by gradient evaluations shows GOC uses ~5592 (1864×3), while CBB uses ~3194—CBB is actually more efficient computationally, contradicting the claimed efficiency.
- **Severely limited experimental validation**: Only one test function—a separable quadratic with diagonal Hessian—is used. This is exactly the case matching the paper's analytical assumptions. No non-quadratic functions, no non-diagonal Hessian cases, no machine learning applications are tested.
- **Missing critical baselines**: No comparison to conjugate gradient (the standard method for quadratic problems), L-BFGS, Nesterov acceleration, or modern adaptive methods (Adam, AdaGrad). The comparison only includes BB and CBB, which are not competitive baselines for convex quadratic optimization.
- **No theoretical convergence analysis**: The paper claims "faster convergence" but provides no convergence rate theorems or proofs. The empirical observations are not backed by theoretical guarantees.
- **Underspecified hyperparameter**: The step size d in Algorithm 1 is used for Hessian-vector product approximation but is never theoretically justified or empirically tuned. No analysis of how d affects numerical stability or convergence.

## Nice-to-Haves
- Extension to non-quadratic optimization problems with analysis of how the method behaves for non-convex landscapes (relevant for neural network training)
- Wall-clock time comparisons and total gradient evaluation counts to properly assess computational efficiency
- Parameter sensitivity analysis for d and examination of different condition numbers

## Novel Insights
The paper's core insight—that the update formulas for SD and CBB can be unified through the lens of the parameter r and extended to include higher powers of Hessian-vector products—is genuinely interesting. The geometric interpretation of CBB as a symmetric reflection about the gradient direction (Figure 1) provides a clean conceptual framework. The derivation showing that x_1(i) = x_0(i)(1 - a^(i)/r_0)^m relates the "order" m to polynomial expansion of eigenvalue ratios offers a theoretical foundation for constructing these methods. However, the insight remains underdeveloped without proper theoretical convergence bounds and computational cost analysis.

## Potentially Missed Related Work
- None identified beyond general optimization textbooks (Nocedal & Wright) covering conjugate gradient methods which would provide important baseline comparisons for quadratic optimization.

## Suggestions
- **Report total computational cost**: Present results in terms of gradient/Hessian-vector product evaluations and wall-clock time, not just iteration counts, to enable fair comparison.
- **Add standard baselines**: Include conjugate gradient method comparison for quadratic problems (the gold standard), plus at least one modern adaptive method.
- **Provide convergence analysis**: Derive theoretical convergence rate bounds for GOC on quadratic problems to substantiate the "faster convergence" claim.
- **Rename the concept**: Use alternative terminology (e.g., "degree" or "level") instead of "order" to avoid conflict with standard optimization nomenclature.
- **Extend empirical validation**: Test on non-diagonal quadratic problems and at least one non-quadratic problem (e.g., logistic regression) to demonstrate broader applicability.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary

The paper introduces HASTE, a framework for retrieving and compressing code context for LLMs. HASTE combines AST-aware chunking, hybrid information retrieval (BM25 + semantic search), and call graph expansion to provide structurally coherent, semantically relevant code context under token budget constraints. The framework is evaluated on a curated dataset of 6 Python files and the SWE-PolyBench benchmark using LLM-as-Judge evaluation.

## Strengths

- **Well-motivated problem formulation**: The paper clearly articulates the fundamental trade-off between structure-aware approaches (which preserve syntax but may miss semantic relevance) and relevance-focused approaches (which find relevant code but may break structural integrity). This addresses a real bottleneck for LLM-based software engineering.

- **Comprehensive architectural design**: The modular pipeline—Scanner, Chunker, Identifier Extraction, Payload Builder, Embedding Generator, and Hybrid Ranker—addresses multiple aspects of the context retrieval problem in a systematic way.

- **Strong curated dataset results**: The framework achieves a 97.3 average Judge Score with compression ratios up to 6.8× (85% reduction), demonstrating that aggressive compression can coexist with high-quality code generation when structural integrity is preserved.

- **Honest failure mode discussion**: Section 5.3 candidly reports cases where HASTE performed poorly (scores of 0, 5, 10), attributing some failures to flawed task suggestions rather than context retrieval issues.

## Weaknesses

- **Missing baseline comparisons in results**: Section 4.1.3 defines three baselines (IR-only, AST-only, naïve truncation), but Table 2 reports only HASTE results with no quantitative comparison to any baseline. This fundamentally undermines the paper's central claim that HASTE outperforms alternative approaches—the evidence is simply absent.

- **Defined metrics never reported**: AST Fidelity and Hallucination Rate are defined as key evaluation metrics in Sections 4.2.2 and 4.2.3, yet neither metric appears in any results table. The claim that HASTE "reduces model-generated hallucinations" remains unsubstantiated by the paper's own evidence.

- **Insufficient evaluation scale**: The curated dataset comprises only 6 files (52–1317 LOC). Drawing general conclusions about compression-quality trade-offs from n=6 is statistically unsound, and the r=-0.97 correlation claim is based on merely 6 data points.

- **Underspecified implementation details**: The "Token-bounded Extraction" from the title is never clearly explained—the paper mentions filtering "under a strict token budget" without describing the actual mechanism. Additionally, the call graph construction methodology is not described, and the embedding model for semantic search is unspecified, all affecting reproducibility.

## Nice-to-Haves

- **Ablation studies**: The paper lacks systematic evaluation of individual components. Readers cannot determine how much call graph expansion contributes versus hybrid ranking, or whether simpler alternatives would suffice.

- **Concrete examples of compressed context**: No side-by-side comparison showing original code versus HASTE's compressed output, making it difficult to assess whether the "structurally coherent" output is genuinely useful.

## Novel Insights

The paper's key insight is that the trade-off between compression and quality can be navigated successfully by preserving AST validity during extraction—ensuring that even aggressive compression maintains compilable, structurally sound context. The finding that compression and Judge Score correlate negatively (r=-0.97) is interesting but requires validation at scale; the paper correctly frames this as a trade-off rather than a flaw. The observation that context quality matters even for simple "no-op" tasks (POLYBENCH-NOOP cases achieving perfect scores) suggests that HASTE's approach is broadly applicable beyond complex edits.

## Potentially Missed Related Work

- **LLMLingua series** (Li et al.) — prompt compression methods for LLMs that could serve as meaningful baselines for comparison.
- **CoCoMIC** and similar code context compression approaches — relevant for positioning against specialized code compression methods.
- **RepoCoder** and **RepoBench** — established benchmarks for repository-level code completion that could strengthen evaluation.

## Suggestions

- **Add baseline comparison table**: Include compression ratios and Judge Scores for all three defined baselines (IR-only, AST-only, naïve truncation) across the same tasks to substantiate improvement claims.

- **Report the defined metrics**: Either include AST Fidelity and Hallucination Rate results in the evaluation, or remove these metric definitions from the methodology if they were not actually measured.

- **Expand evaluation scale**: Increase the curated dataset size substantially (e.g., dozens of files across multiple repositories) or report full SWE-PolyBench quantitative results in a structured table with baseline comparisons.

- **Clarify token-bounded extraction mechanism**: Explain how the token budget is enforced—does the system prune AST nodes greedily by relevance rank, truncate at the budget boundary, or use another strategy? This is central to the claimed contribution.

- **Specify implementation details**: Document the embedding model used for semantic search, the LLM-as-judge configuration, and the call graph construction methodology.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary

AutoNFS proposes a neural feature selection method that uses Gumbel-Sigmoid sampling with temperature annealing and a cardinality penalty to automatically determine both the feature subset and its size for tabular data. The architecture consists of a masking network (generating selection masks from a learned embedding) and a task network (evaluating feature relevance), trained end-to-end. The key contribution is eliminating the need to pre-specify the number of features—a practical pain point in existing FS methods.

## Strengths

- **Automatic feature count determination**: The cardinality penalty L_select allows the model to discover the minimal sufficient feature set without user-specified budgets, addressing a genuine practical limitation of existing FS methods. The approach achieves strong performance while selecting substantially fewer features (e.g., reducing features from 136 to 47 on the Microsoft dataset without performance degradation).

- **Strong empirical results**: AutoNFS achieves the best average rankings across all three corruption scenarios (random: 3.5, corrupted: 2.1, second-order: 3.6) on the OpenML benchmark, with consistent performance improvements on 24 metagenomic datasets while selecting only 7.7% of original features on average.

- **Empirical computational efficiency**: Figure 4 demonstrates near-constant scaling with input dimensionality (α ≈ 0.08), which is practically significant for high-dimensional applications where conventional methods scale linearly or superlinearly.

- **Reproducibility and clarity**: Algorithm 1 provides complete pseudocode, hyperparameters are specified in Appendix C, and code is available via anonymous repository.

## Weaknesses

- **Missing direct comparison with Stochastic Gates (STG)**: The paper cites Yamada et al. (2020) for Stochastic Gates, which uses nearly identical Gumbel-based continuous relaxation for feature selection, but omits it from experiments. This is a significant gap—STG is the most directly comparable prior work and its absence undermines claims of novelty and superiority.

- **Unfair baseline configuration**: Tables 3-5 state that "all baseline methods select the same number of features as were in the initial representation," meaning methods like Lasso, LassoNet, and Deep Lasso are run without their sparsity mechanisms properly tuned. These baselines have built-in feature selection capabilities that should be configured fairly. Without this, it's unclear whether AutoNFS's advantage stems from better methodology or simply from using fewer features while baselines use all features.

- **Missing statistical significance reporting**: Main results (Tables 3-5) report single values without standard deviations or confidence intervals across multiple runs. Given the stochastic nature of Gumbel-Sigmoid sampling, uncertainty quantification is essential to assess result reliability.

- **Overstated theoretical complexity claim**: The paper claims "nearly constant computational overhead regardless of the input dimensionality," but the masking network's output layer has dimension D, requiring O(D) computation. The empirical scaling results are valid, but the theoretical claim is incorrect and should be qualified.

- **Insufficient ablation studies**: Critical design choices lack empirical justification: the temperature annealing schedule (claimed "critical" without comparison to fixed temperature), embedding size (fixed at 32), and task network architecture. The contribution of each component remains unclear.

## Nice-to-Haves

- Stability analysis reporting which features are consistently selected across multiple training runs, since neural FS methods are known to exhibit instability that undermines interpretability claims.
- Instance-specific feature selection analysis or discussion of scenarios where the global mask assumption may fail.

## Novel Insights

The exploration-exploitation framing (Appendix B) provides a useful conceptual lens for understanding temperature annealing: high temperature enables broad feature space exploration, while annealing enforces exploitation of learned important features. This connects FS optimization to reinforcement learning principles, suggesting that the curriculum-like transition from continuous to discrete selection is not merely a computational trick but reflects a principled trade-off between discovering relevant features and committing to decisions.

## Potentially Missed Related Work

- None identified that would fundamentally change the contribution. The already-cited Stochastic Gates (Yamada et al., 2020) and L0-regularization (Louizos et al., 2017) are the most relevant prior works and should be included as experimental baselines.

## Suggestions

1. **Add STG (Yamada et al., 2020) as a baseline** with comparable automatic feature count determination—this is essential for establishing the contribution relative to the most similar prior work.

2. **Configure baselines for fair comparison**: Either tune Lasso, LassoNet, and other embedded methods with proper sparsity settings, or clearly frame the comparison as "AutoNFS's automatic sparsity discovery vs. untuned baselines."

3. **Report standard deviations across multiple runs** for all main results to establish statistical significance.

4. **Add ablation on temperature annealing**: Compare against fixed low temperature, fixed high temperature, and different decay schedules to justify this design choice.

5. **Qualify the complexity claim**: State that empirical scaling is nearly constant while acknowledging the theoretical O(D) dependency in the output layer.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0, 2.0, 4.0, 4.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
The paper proposes a model-agnostic technique for improving the accuracy of small interpretable models by learning an optimized training distribution. The key innovation is using uncertainty scores from a probabilistic "oracle" model to project instances to 1D, then learning a distribution over these scores via a Dirichlet Process-based Infinite Beta Mixture Model optimized through Bayesian Optimization. The method is demonstrated on linear models, decision trees, and gradient boosted models across 13 datasets.

## Strengths
- **Extensive empirical validation**: The paper evaluates the method across 13 datasets, multiple model families (LPMs, decision trees), multiple oracles (GBM, RF), and various size constraints. The improvements range from modest to dramatic (over 100% relative improvement in some small-model settings), with appropriate statistical testing via Wilcoxon signed-rank and Friedman tests.
- **Genuine model agnosticism**: The technique works across different notions of model size—number of non-zero coefficients for linear models, depth for trees, and both depth and number of trees for GBMs (Appendix A.12). This flexibility is a practical advantage.
- **Cross-feature space capability**: Section A.13 demonstrates that the oracle and interpretable model can use different feature representations (e.g., a GRU on character sequences for the oracle, and character n-grams for a decision tree). This significantly expands practical applicability.
- **Competitive with specialized methods**: The comparison against IMM (cluster explanation) and SNC (prototype classification) in Appendix A.10 shows the technique can elevate simple methods like CART to be competitive with specialized algorithms designed for those tasks.
- **Honest acknowledgment of limitations**: The paper transparently discusses runtime issues (Section 5) and provides preliminary results showing BoTorch can reduce runtime from ~1 hour to ~2 minutes (Appendix A.11).

## Weaknesses
- **Missing comparison to simpler baselines**: The paper briefly dismisses simple uncertainty sampling by citing Ghose & Nguyen (2024), but provides no experimental comparison. Without demonstrating that straightforward approaches (e.g., sampling proportionally to uncertainty scores, class-balanced sampling, or hard example mining) underperform, the complexity of the DP-based IBMM + BayesOpt approach lacks adequate justification.
- **No ablation on oracle quality**: The entire method depends on the oracle providing meaningful uncertainty estimates. The paper does not analyze what happens when the oracle is poorly calibrated, has lower accuracy, or is itself a small model. This is critical for understanding practical applicability and failure modes.
- **Limited novelty beyond prior work**: The core idea of learning training distributions is from Ghose & Ravindran (2020) by the same author. The main novelty is substituting uncertainty scores for density trees—a reasonable improvement, but incremental rather than transformative.
- **No analysis of when the method helps vs. hurts**: Table 1 shows approximately 2% of cases with negative improvements (underlined values). The paper notes this low incidence rate but provides no analysis of what characterizes these failure cases, leaving practitioners without guidance on when to apply the method.
- **Misleading "one hyperparameter" claim**: The paper states T (iterations) is the only parameter users need to set, but Appendix A.3 reveals many fixed parameters (scale=10000, α bounds, parameter ranges for BayesOpt). Sensitivity to these defaults is not analyzed.

## Nice-to-Haves
- **Visualization of learned distributions**: Show the sampling weights learned across the uncertainty spectrum for representative successful and unsuccessful cases. This would reveal *why* the method works and what the learned distributions prioritize.
- **Analysis of the po parameter**: The paper includes po (proportion of original distribution) as a "safety net"—analyzing typical learned values would clarify whether good solutions rely primarily on the learned distribution or mostly revert to uniform sampling.

## Novel Insights
The cross-feature space capability is the paper's most interesting contribution. The demonstration that a GRU trained on character sequences can serve as an oracle for a decision tree using n-gram features—and that this configuration produces substantial improvements—suggests a broader principle: informative training distributions can be transferred across representations. This hints at a decoupling between the information source (oracle) and the interpretable model that could be explored more deeply in future work. The result that simple CART decision trees, when combined with this method, can compete with specialized cluster explanation algorithms (IMM) is also noteworthy, suggesting that instance weighting/resampling is an underexplored axis for improving interpretable models.

## Potentially Missed Related Work
None identified.

## Suggestions
1. **Add comparison to simple baselines**: Include uncertainty-weighted sampling and/or class-balanced sampling as baselines to justify the method's complexity. This is essential for establishing that the DP-based approach provides benefits beyond intuitive alternatives.
2. **Analyze oracle quality requirements**: Run experiments varying oracle accuracy/calibration to establish minimum requirements. If weak oracles also work, applicability expands; if only strong oracles help, that limits practical use.
3. **Characterize failure cases**: Analyze the settings where the method produces negative improvements to provide practitioners with guidance on when to avoid the approach.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 0.0]
Average score: 3.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

SHALLOW introduces a benchmark framework for evaluating hallucinations in automatic speech recognition (ASR) systems by decomposing errors into four dimensions: Lexical Fabrications, Phonetic Fabrications, Morphological Errors, and Semantic Errors. The authors demonstrate through evaluation across 12 ASR models and 10 datasets that their multi-dimensional metrics reveal error patterns invisible to Word Error Rate (WER), and that correlations between WER and hallucination metrics break down as transcription quality degrades—precisely when fine-grained evaluation matters most.

## Strengths

- **Timely and well-motivated contribution**: The paper addresses a critical gap in ASR evaluation, with clear motivation from high-stakes applications (healthcare, legal transcription) where semantic inversions like "take medication" vs. "skip medication" have low WER but dangerous consequences.

- **Linguistically grounded taxonomy**: The four-dimensional decomposition (lexical, phonetic, morphological, semantic) captures distinct error types with clear definitions and computational instantiations. The distinction between acoustic fidelity errors (ASR hallucinations) and factuality errors (LLM/VLM hallucinations) is clearly articulated.

- **Empirical validation of key claim**: Figures 3 and 7 compellingly demonstrate that WER-hallucination correlations degrade substantially at higher error rates (e.g., morphological-semantic correlations becoming negative at WER > 70%), validating that WER obscures error structure precisely when evaluation matters most.

- **Architectural insights**: The cross-model analysis reveals meaningful differences—encoder-decoder models show balanced error profiles while SpeechLLMs like Phi-4 exhibit stronger linguistic fluency but higher phonetic confusions—insights invisible to WER alone.

- **Synthetic benchmark design**: The 1,050-sample synthetic dataset with targeted perturbations validates metric specificity and demonstrates orthogonality among the four dimensions (Figure 2, 4).

- **Practical case study**: Table 3 provides concrete medical examples where SHALLOW correctly flags dangerous errors (e.g., polarity reversals) that receive low WER scores, demonstrating real-world utility.

## Weaknesses

- **Insufficient justification for composite metric weights**: The weighting schemes (e.g., LF: 0.5/0.3/0.2 for insertion/substitution/deletion; ME: 0.4/0.6 for structural/grammatical) are described as "empirically validated" but the paper provides no sensitivity analysis, ablation study, or human evaluation to demonstrate that these specific weights reflect meaningful trade-offs or are robust to alternative parameterizations. This gap undermines confidence in adopting SHALLOW as a standard benchmark.

- **No statistical significance testing**: Results in Table 2 and throughout the paper report mean values without error bars, confidence intervals, or significance tests. Given variability across test samples, it remains unclear whether reported architectural differences (e.g., between Whisper-v3 and Canary) reflect true effects or random variation.

- **Missing comparison with alternative evaluation metrics**: The paper compares against WER but does not compare against other relevant metrics such as Semantic Distance (Kim et al., 2021), BERTScore, or information preservation metrics. Without these baselines, the claim that SHALLOW captures what existing metrics miss is only partially validated.

- **No comparison with prior hallucination detection methods**: The paper cites Frieske & Shi (2024), Barański et al. (2025), and Atwany et al. (2025) as related work but provides no quantitative comparison of SHALLOW against these perturbation-based or LLM-based approaches. This leaves unclear whether SHALLOW offers practical advantages beyond existing methods.

- **External model dependencies as potential failure modes**: The semantic metrics rely on BERT embeddings and NLI models, and morphological metrics use LanguageTool. The paper does not discuss how biases or failures in these underlying systems might propagate to SHALLOW scores, nor does it provide ablation showing robustness to alternative embedding/NLI choices.

- **Limited validation on real ASR outputs**: While the synthetic benchmark shows orthogonality, the paper does not report inter-metric correlation matrices on the 12 models × 10 datasets of real ASR outputs. If LF, PF, ME, SE are highly correlated on natural errors, the multi-dimensional diagnostic value would be diminished.

- **Medical case study lacks systematic quantification**: The case study examines hand-picked examples from only Phi-4. A systematic evaluation across all models showing what fraction of critical errors SHALLOW catches versus WER would strengthen claims about practical utility.

## Nice-to-Haves

- Human evaluation study correlating SHALLOW scores with expert judgments of hallucination severity
- Sensitivity analysis on composite metric weights (e.g., ±0.1 perturbations)
- Failure mode analysis: cases where SHALLOW produces counterintuitive scores (benign paraphrases with high SE, or obvious hallucinations with low scores)
- Guidance on translating SHALLOW diagnoses into targeted model improvements (e.g., "elevated PF suggests acoustic model improvements; elevated SE suggests language model calibration")
- Runtime comparison table for WER vs. each SHALLOW component metric

## Novel Insights

The finding that WER-hallucination correlations systematically degrade as recognition quality worsens has important implications: aggregate metrics are least informative precisely when models fail most severely. This validates SHALLOW's diagnostic value for challenging acoustic conditions (noisy environments, accented speech, child speech) where understanding error structure matters more than raw accuracy. The architectural signature analysis—encoder-decoder models showing balanced profiles while SpeechLLMs exhibit semantic coherence at the cost of phonetic fidelity—provides actionable insights for model selection in domain-specific deployments.

## Potentially Missed Related Work

None identified. The paper covers the relevant ASR hallucination literature (Frieske & Shi 2024, Barański et al. 2025, Atwany et al. 2025) and semantic evaluation metrics (Kim et al. 2021 Semantic Distance, BERTScore). However, explicit quantitative comparison with these methods would strengthen the contribution.

## Suggestions

1. Add sensitivity analysis showing how rankings and conclusions change under alternative weightings for the composite metrics.
2. Include statistical significance tests (bootstrap confidence intervals or standard errors) for main results.
3. Add correlation analysis of LF/PF/ME/SE on real ASR outputs to validate orthogonality claims beyond synthetic data.
4. Expand the medical case study to systematic evaluation across all models rather than hand-picked Phi-4 examples.
5. Discuss known failure modes and edge cases where SHALLOW may produce misleading scores, and provide guidance for interpretation.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 2.0, 2.0]
Average score: 4.5
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
## Summary
The paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that repairs human-specified proxy reward functions by learning an additive correction term from trajectory preferences. PBRR uses a reference policy for targeted exploration and proposes a novel three-term loss function that regularizes corrections toward transitions where the proxy reward incorrectly assigns high values. The authors prove regret bounds matching prior strategic RLHF methods and demonstrate improved data efficiency across four reward-hacking benchmark environments.

## Strengths
- **Novel problem formulation**: The key insight—that repairing a misspecified proxy reward function may require fewer preferences than learning from scratch—addresses a practical gap between manual reward design and pure RLHF, with clear real-world motivation.

- **Principled algorithmic design**: The three-term loss (Eq. 3) exploits the structural assumption that proxy rewards are typically "aligned or overly optimistic" by regularizing corrections on correctly-ranked pairs (L⁺) and prioritizing downward corrections on misclassified pairs (L⁻), with intuitive motivation provided in Appendix I.1.

- **Theoretical grounding**: Theorems 5.1 and 5.2 establish sub-linear regret bounds in tabular settings, providing formal sample-efficiency guarantees. The analysis correctly identifies that using (π_ref, π_{r̂_t}^*) for exploration introduces a constant factor C₁ relative to optimal uncertainty-maximizing exploration.

- **Comprehensive empirical evaluation**: Experiments span four diverse domains (pandemic mitigation, glucose monitoring, traffic control, AI safety gridworld) with meaningful proxy misspecifications that induce substantial policy suboptimality, demonstrating consistent improvements over strong baselines including RRM (Cao et al., 2025) and Online-RLHF.

- **Demonstrated robustness**: Appendix G.6 shows PBRR succeeds even when the optimism assumption fails, and Appendix G.8 empirically validates that randomly initialized reference policies suffice, reducing practical deployment barriers.

## Weaknesses
- **Simulated preferences only**: All experiments use preferences sampled from ground-truth rewards via Boltzmann distribution, not actual human feedback. Real human preferences exhibit noise patterns, cognitive biases, and inconsistencies not captured by this model, leaving the core claim of "human feedback" partially unvalidated.

- **Theory-practice disconnect**: The theoretical regret bounds assume C₁ > 0 (enabling uncertainty-maximizing exploration from the undominated policy set), but all experiments use C₁ = 0. The paper does not empirically evaluate whether the theoretical exploration strategy provides benefits over the simpler reference-policy comparison used in practice.

- **Limited scalability evidence**: Evaluated environments have state dimensions ranging from 36 to 312, with action spaces ≤ 10 dimensions. The approach is not tested on larger-scale domains (e.g., LLM fine-tuning, high-dimensional robotics) where reward hacking is also prevalent.

- **Statistical significance limitations**: Most experiments report only 3 random seeds. While Appendix G.9 provides 10-seed analysis for one environment showing statistical significance, stronger claims about data efficiency would benefit from more seeds across all environments.

## Nice-to-Haves
- Visualization of learned corrections g(s,a,s') across states/transitions in the AI Safety Gridworld to verify whether the method correctly identifies and penalizes the sprinkler exploit rather than learning opaque corrections.

- Systematic study of how reference policy quality affects sample efficiency (beyond the random policy test in Appendix G.8), as the theoretical guarantee only ensures the repaired policy matches or exceeds π_ref.

## Novel Insights
The paper's most valuable insight is that proxy reward functions, even when misspecified, contain valuable inductive bias that can be leveraged for efficient reward learning. Rather than discarding human-specified rewards entirely in favor of learning from preferences (standard RLHF), or manually iterating through trial-and-error design, PBRR treats the proxy as a starting point and learns targeted corrections. The loss function design specifically exploits the observation that proxy rewards tend to be "optimistic"—assigning high reward to behaviors that seem desirable at design time but fail under optimization. The L⁻ term's prioritization of downward corrections (rather than symmetric adjustments) embodies this insight: if the proxy incorrectly prefers an undesirable trajectory, decrease the proxy's reward rather than increase the preferred trajectory's reward, preserving correct aspects of the proxy while fixing misalignments.

## Potentially Missed Related Work
- **Inverse Reinforcement Learning (IRL) with prior rewards**: Methods like Maximum Entropy IRL or Bayesian IRL often incorporate priors over reward functions. The paper discusses this in Section 3 (Eisenstein et al., 2023; Novoseller et al., 2020), but the relationship to classical IRL with informative priors could be more thoroughly explored.

- **Reward shaping and potential-based corrections**: Ng et al. (1999) established conditions under which reward shaping preserves optimal policies. The additive correction approach could be compared to potential-based shaping to understand when corrections preserve vs. modify optimal behavior.

- **Concurrent work on reward model fine-tuning**: Several recent works explore fine-tuning reward models from human preferences (e.g., Dong et al., 2024), which shares conceptual similarities but differs in not assuming a starting proxy.

## Suggestions
- Add experiments comparing PBRR's exploration strategy (sampling from π_ref and π_{r̂_t}^*) against ensemble-based uncertainty sampling, both using PBRR's loss function, to cleanly isolate the contribution of each component.

- Provide sensitivity analysis for the regularization weights λ₁ and λ₂, which are set to 10/|D⁺| with decay but not ablated—these control the balance between preference loss and regularization and could significantly affect performance in different domains.

- Report computational overhead (training time per iteration) for PBRR versus baselines, since each iteration requires training a new policy with PPO.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
This paper proposes Latent Reasoning Tuning (LRT), a framework that replaces the costly autoregressive generation of explicit reasoning trajectories with compact latent representations computed by an auxiliary reasoning network. The authors first demonstrate that reasoning LLMs maintain high accuracy even when conditioned on fragmented reasoning paths (50% token/step skipping), then introduce a two-stage training paradigm (SFT followed by RL with GRPO) to optimize a lightweight reasoning network that generates latent reasoning tokens which condition a frozen base model to produce final answers efficiently.

## Strengths
- **Strong empirical motivation**: The fragmentation experiments in Section 2 provide compelling evidence that explicit reasoning trajectories contain substantial redundancy—models maintain ~92% accuracy even with 50% token skipping—justifying the core premise that full explicit trajectories are unnecessary.
- **Novel architectural approach**: Unlike prior work that either shortens explicit reasoning (via RL penalties) or bypasses it entirely (via prompting), LRT genuinely internalizes reasoning into a learned latent space while preserving the ability to switch back to explicit reasoning modes through its modular design.
- **Strong empirical results against efficient reasoning baselines**: The method consistently outperforms NoThinking, ShorterBetter, and LC-R1 across mathematical and out-of-domain benchmarks (e.g., 77.16% vs 74.26% on GSM8K under 512-token budget; average improvements of 5-9% across baselines).
- **Modular and practical design**: By freezing base model parameters and training only the reasoning network (initialized from Qwen3-Embedding-0.6B), the framework enables seamless transitions between latent and explicit reasoning without modifying the original LLM weights.
- **Comprehensive ablation studies**: The paper includes thoughtful ablations on the number of latent tokens (64-512), training strategies (SFT vs. SFT+RL), and interaction with base model capacity ( Tables 3, 4, 6), plus efficiency analysis (Table 7) and latent representation analysis (Table 8).

## Weaknesses
- **Misleading claim about surpassing Qwen3**: The abstract states the approach "surpasses the state-of-the-art Qwen3 hybrid reasoning framework," but Table 2 only compares against Qwen3's non-thinking mode. Table 5 shows Qwen3 thinking mode achieves 69.61% at 4096 tokens versus LRT's 59.63% at 1024 tokens. The claim should be qualified—LRT surpasses Qwen3's *non-thinking mode* and offers an efficiency-accuracy tradeoff, not a blanket improvement over hybrid reasoning.
- **Core mechanism poorly explained in main text**: The critical mechanism for how latent representations condition generation—$z = f_{out}(G_\phi(f_{in}(H_X) \odot [\hat{r}_1, ..., \hat{r}_t]))$—only appears in Appendix C (Equation 5). Readers must navigate to the appendix to understand the Hadamard product with learnable vectors and the projection layers. This architectural detail belongs in Section 3.
- **No experimental comparison with prior latent reasoning methods**: Appendix E discusses Coconut and related work conceptually, but provides no direct experimental comparison. Given that Coconut represents the most relevant prior work on latent reasoning, empirical comparison on shared benchmarks is needed to establish the claimed distinctions (parallel trajectory generation vs. iterative hidden state refinement).
- **Missing baseline error bars**: Table 9 provides standard deviations only for the proposed method. Without variance estimates for baselines (NoThinking, ShorterBetter, LC-R1), claims of "consistent outperformance" lack statistical grounding.

## Nice-to-Haves
- **FLOPs comparison alongside latency**: While Table 7 reports latency (11.79s vs. 71.09s for thinking mode), a FLOPs accounting of base model forward pass + reasoning network would clarify the computational tradeoff more precisely than token budgets.
- **Probing experiments on latent representations**: The cosine similarity analysis (Table 8) shows domain clustering but doesn't reveal what semantic or logical content is captured. Probing experiments or attention analysis would strengthen claims that latents serve as "reasoning instructions."
- **Failure case analysis**: The paper shows aggregate improvements but never analyzes where latent reasoning fails versus explicit reasoning—critical for assessing practical applicability.
- **RL hyperparameter justification**: Algorithm 1 mentions GRPO training, but key hyperparameters (8 rollouts per question, KL penalty of 2×10⁻³, 100 RL steps) lack sensitivity analysis.

## Novel Insights
The fragmentation analysis (Section 2) provides a valuable empirical insight: reasoning LLMs exhibit "resilience to noisy or fragmental input," maintaining high accuracy despite degraded trajectories with elevated perplexity. This suggests models exploit salient reasoning components amid substantial noise, which motivates the latent approach. The finding that larger models (Qwen3-8B) continue to benefit from more latent tokens while smaller models (Qwen3-1.7B) saturate at 256 tokens (Tables 3 and 6) reveals an interesting interaction between model capacity and latent representation complexity—larger models can leverage richer latent information. The empirical analysis of latent representations (Table 8) showing domain-specific clustering (AMC/MATH-500 similarity, LSAT separation) suggests the learned latents encode task-specific reasoning patterns rather than generic embeddings.

## Potentially Missed Related Work
- **Training Large Language Models to Reason in a Continuous Latent Space (Hao et al., 2024)** — This is Coconut, which the appendix discusses but does not experimentally compare against. Direct comparison would strengthen novelty claims about parallel trajectory generation versus iterative refinement.
- **Reasoning with Latent Thoughts: On the Power of Looped Transformers (Saunshi et al., 2025)** — Provides theoretical analysis of latent reasoning that could inform the paper's theoretical justification for why latent representations suffice.

## Suggestions
1. **Revise the abstract and claims** to accurately state that LRT surpasses Qwen3's *non-thinking mode* and offers a favorable efficiency-accuracy tradeoff, rather than claiming to surpass the hybrid reasoning framework as a whole.
2. **Move Equation 5 and the architecture description from Appendix C to Section 3.2** so readers understand the mechanism (projection layers, Hadamard product with learnable vectors) without consulting supplementary material.
3. **Add experimental comparison with Coconut** or similar latent reasoning methods on shared benchmarks to establish the claimed architectural distinctions empirically.
4. **Report baseline error bars** across multiple random seeds to support claims of consistent improvement.
5. **Clarify the reward function** used in GRPO training—currently Algorithm 1 mentions "ComputeReward" without specifying whether it's binary correctness, format rewards, or task-specific formulations for LSAT/GPQA.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 8.0, 4.0]
Average score: 5.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary
This paper addresses hardware underutilization during LoRA hyperparameter tuning. The key insight is that individual LoRA configurations (with small batch sizes and low rank) underutilize GPU resources, with SM occupancy around 16.7% and memory utilization below 55%. PLORA packs multiple configurations into concurrent training jobs, sharing the frozen base model while training adapters simultaneously. The system combines an ILP-based offline packing planner with custom CUDA kernels for efficient batched LoRA computation, achieving up to 7.52× makespan reduction and 12.8× throughput improvement.

## Strengths
- **Strong empirical motivation**: Over 1,000 experiments demonstrate LoRA hyperparameter sensitivity (up to 14.2% accuracy difference from learning rate variation alone, up to 23.4% improvement over default configurations), convincingly establishing the need for efficient hyperparameter tuning across tasks and models.

- **Novel problem identification**: The observation that LoRA training underutilizes hardware is well-supported by profiling data. This identifies a genuine gap—not addressed by prior multi-LoRA systems focused exclusively on inference serving (SLoRA, Punica, vLLM).

- **Substantial efficiency gains**: Experiments demonstrate consistent improvements across multiple model families (QWen-2.5, LLAMA-3) and sizes (3B to 32B), with makespan reductions from 6.33× to 7.52× and throughput improvements up to 12.8×.

- **Sound technical approach**: The ILP-based scheduling algorithm includes provable approximation bounds (empirical approximation ratio between 1.05-1.14), and the custom CUDA kernels achieve near-linear speedup up to 32 adapters across different GPU architectures.

## Weaknesses
- **Missing comparison with intelligent HPO methods**: The baselines (Min GPU, Max GPU) represent naive sequential strategies. Real practitioners use methods like Hyperband/ASHA or Bayesian optimization that share computation across configurations. Comparing PLORA combined with such methods would demonstrate that packing gains are complementary to smarter search strategies—a meaningful comparison currently absent.

- **Training equivalence under packing not verified**: The paper claims PLORA finds the "best" configuration but does not validate that a configuration trained in packed mode achieves equivalent accuracy to sequential training. If packing introduces gradient interference or alters optimization trajectories, the best configuration found might differ from the true optimum. A comparison of identical configurations trained both ways is missing.

- **No scheduling algorithm ablation**: The DTM algorithm involves recursive ILP solving, but it's unclear whether simpler heuristics (e.g., greedy-by-memory-fit) would achieve similar results. Figure 6 separates kernel vs. scheduling contributions but doesn't justify the ILP complexity against simpler alternatives.

- **Limited search space analysis**: All experiments use exactly 120 configurations without justification. How does makespan reduction scale with search space size? This matters for real-world tuning that often explores thousands of configurations.

- **Experimental results lack statistical rigor**: All reported numbers are single values without error bars or standard deviations across multiple runs, reducing confidence in exact speedup claims given potential GPU workload variability.

## Nice-to-Haves
- Validation on larger models (70B+) would strengthen generalization claims
- Analysis of how the planner's ~10-minute overhead scales with search space size
- Explicit discussion of failure handling when memory model predictions underestimate requirements

## Novel Insights
The core insight—that LoRA hyperparameter tuning creates hardware underutilization because adapters are small relative to modern GPUs—is well-motivated and the packing solution is elegant. The paper correctly identifies that this approach is orthogonal to existing HPO methods (which reduce configuration count) rather than competing with them. The theoretical bound on the scheduling approximation ratio, while loose in worst case, provides analytical grounding rare in systems papers. The empirical study of LoRA hyperparameter sensitivity across tasks and models (Table 2-3) is a valuable contribution showing optimal configurations vary significantly—no single rule of thumb applies.

## Potentially Missed Related Work
None identified by the review process that would significantly impact the paper's positioning.

## Suggestions
1. Add an experiment comparing the same LoRA configuration trained packed vs. sequentially to verify training equivalence
2. Compare against at least one intelligent HPO baseline (e.g., ASHA with early stopping) to demonstrate complementarity
3. Include error bars from at least 3 runs for key experimental results
4. Add an ablation comparing DTM against a simple greedy packing heuristic to justify the ILP complexity
5. Analyze makespan reduction across different search space sizes (e.g., 30, 120, 500 configurations)

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0]
Average score: 5.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

The paper proposes SigMap, a multimodal foundation model for wireless localization that introduces: (1) a cycle-adaptive masking strategy for self-supervised pre-training on CSI data that dynamically adjusts to signal periodicity to prevent shortcut learning, and (2) a "map-as-prompt" framework that encodes 3D geographic information via graph neural networks as lightweight soft prompts for cross-scenario adaptation. Experiments on ray-tracing datasets demonstrate improvements over baselines with parameter-efficient fine-tuning.

## Strengths

- **Well-motivated domain-specific innovations**: The cycle-adaptive masking strategy addresses a genuine limitation of applying MAE-style pre-training to CSI data, where naive masking can be circumvented by exploiting periodic signal structures. This represents principled adaptation of foundation model techniques to wireless domain characteristics.

- **Novel geographic integration mechanism**: The map-as-prompt formulation using GNN-encoded 3D building geometry provides an interpretable way to inject environmental constraints without modifying the pre-trained backbone, achieving strong results with only 0.7% of parameters updated during fine-tuning.

- **Comprehensive empirical evaluation**: The paper evaluates across single-BS, multi-BS, and cross-scenario generalization settings with meaningful ablations comparing masking strategies (Table 3) and map modalities (Table 4). The relative improvements over LWLM (34.4% MAE reduction for single-BS, 53.2% for generalization to DeepMIMO O2) are substantial.

- **Reproducibility support**: Code and detailed hyperparameters are provided via anonymous repository, with dataset configurations clearly specified in Tables 6-7.

## Weaknesses

- **Mischaracterization of generalization claims**: The abstract claims "strong zero-shot generalization in unseen environments," but Section 4.5 reveals that task heads are fine-tuned using "approximately 100 instances per scenario." This is few-shot transfer learning, not zero-shot. The backbone being frozen does not make the approach zero-shot—this distinction matters for accurately positioning the contribution.

- **No validation on real-world CSI measurements**: All experiments use ray-tracing simulations (DeepMIMO, WAIR-D). While ray-tracing is standard for controlled experiments, the claimed applicability to 5G/6G applications lacks substantiation without real-world CSI data that includes hardware imperfections, synchronization errors, and environmental noise.

- **Missing statistical significance reporting**: Tables report only mean values despite stating results are "averaged over 5 independent runs." Standard deviations or confidence intervals are essential for assessing reliability of claimed improvements.

- **Incomplete baseline coverage**: The introduction cites LWM (Alikhani et al., 2024) and WirelessGPT (Yang et al., 2025) as relevant foundation models for wireless, but these are not included in experimental comparisons. Comparing against these directly relevant baselines would strengthen claims of state-of-the-art performance.

- **NLoS attention mechanism introduced post-hoc**: Equation 11 (NLoS-aware attention) appears in Section 4.2 (results) describing single-BS performance, but was not defined in Section 3 (methodology). This creates confusion about whether NLoS attention is a core contribution or an afterthought added to explain results.

## Nice-to-Haves

- Analysis of computational overhead for geographic prompt generation, including sensitivity to mesh resolution and number of GNN layers
- Failure case analysis identifying scenarios where the method degrades (e.g., extreme multipath, sparse map data)
- Visualization of learned geographic prompts to validate that they encode meaningful spatial relationships rather than environment-specific embeddings

## Novel Insights

The cycle-adaptive masking strategy represents a thoughtful response to domain-specific structure in CSI data. Unlike vision transformers that process images as generic 2D patches, CSI exhibits inherent periodicity across antenna and subcarrier dimensions. By detecting this periodicity via cross-correlation and generating masks that disrupt periodic shortcuts, the model is forced to learn genuinely useful signal representations. The ablation in Table 3 supports this intuition: adaptive masking achieves 84.5% CDF@1m versus 80.3% for grid masking and 75.3% for strip masking. The map-as-prompt mechanism similarly exploits domain structure—encoding building geometry via GNN and injecting it as soft prompts leverages the physical relationship between signal propagation and environment topology.

## Potentially Missed Related Work

- **LWM (Alikhani et al., 2024)**: Cited in introduction as "Large wireless model" for channel representation learning, but not experimentally compared. Would provide direct comparison of foundation model approaches for wireless tasks.

- **WirelessGPT (Yang et al., 2025)**: Cited as "generative pre-trained multi-task learning framework for wireless communication" but not included in experiments. Relevant for understanding how task-specific adaptation compares across foundation models.

## Suggestions

- Correct the abstract to describe the approach as "few-shot" or "parameter-efficient" transfer rather than "zero-shot" generalization

- Report standard deviations or confidence intervals across the 5 experimental runs to enable statistical significance assessment

- Add LWM and WirelessGPT as experimental baselines, or explicitly justify their exclusion

- Move Equation 11 (NLoS attention) to Section 3 with proper motivation and definition

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0]
Average score: 5.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary
This paper investigates whether Large Language Models (LLMs) exhibit human-like inductive biases toward efficient semantic categorization, using the Information Bottleneck (IB) framework from cognitive science. Through two experiments—English color naming across 39 models and a novel Iterated In-Context Language Learning (IICLL) paradigm—the authors show that larger instruction-tuned models achieve better alignment with human color naming, and that LLMs iteratively restructure random category systems toward IB-efficiency. Notably, only Gemini 2.0 recapitulates the full range of human-like IB tradeoffs observed in cultural transmission experiments.

## Strengths
- **Novel methodological contribution**: The IICLL paradigm extends iterated in-context learning to study semantic category evolution in LLMs, enabling direct comparison with human cultural transmission experiments from cognitive science.
- **Comprehensive model evaluation**: Testing 39 models across 6 families with systematic variations in size, instruction-tuning, and modality provides robust evidence linking model capabilities to semantic alignment. The analysis of Olmo training checkpoints offers insight into how alignment emerges during training.
- **Strong cognitive science grounding**: The paper meaningfully bridges established cognitive science theories (IB principle, iterated learning) with LLM evaluation, using well-established human behavioral datasets (WCS, Xu et al. 2013) as comparison benchmarks.
- **Evidence against pure memorization**: The Nearest Neighbor baseline comparison (Appendix M) demonstrates that Gemini's evolved categories are more human-like and IB-aligned than simple feature-based clustering, suggesting genuine inductive bias rather than trivial pattern matching.
- **Principled quantitative evaluation**: Using IB-efficiency loss, NID-based alignment measures, and comparison to theoretical IB bounds provides rigorous quantitative evaluation of semantic systems.

## Weaknesses
- **Insufficient statistical rigor for IICLL experiments**: The paper does not clearly report how many independent chains were run per model per condition, nor does it provide error bars, variance estimates, or significance tests for key comparisons. Human experiments averaged 20 chains across 4 replications per condition; unclear replication counts make it difficult to assess whether observed trajectories reflect systematic behavior or stochasticity.
- **Uncontrolled training data contamination**: The central claim that LLMs have an "inductive bias" toward IB-efficiency rather than merely mimicking training data lacks systematic control. The paper does not analyze whether WCS data, IB-optimal color systems, or related materials appear in model training corpora, making it difficult to distinguish emergence from memorization.
- **Lack of mechanistic explanation for model differences**: The finding that only Gemini 2.0 achieves human-like IB tradeoffs while other state-of-the-art models converge to low-complexity solutions is central, yet the paper offers no systematic ablation or investigation into what capabilities (context length, training procedure, architecture) enable this success.
- **Underdeveloped generalization claim**: The paper briefly mentions testing "a domain distinct from color" but provides minimal detail in the main text. Without robust evidence from other semantic domains, claims about general inductive biases remain speculative.

## Nice-to-Haves
- Analysis of why CIELAB inputs degrade performance (Appendix C) could illuminate properties of LLM color representations and perceptual understanding.
- Inclusion of additional frontier models (e.g., GPT-4) would strengthen claims about which model capabilities enable IB-efficient categorization.

## Novel Insights
The IICLL paradigm represents a genuine methodological innovation for eliciting and studying implicit inductive biases in LLMs. The finding that iterated cultural transmission alone can drive category systems toward IB-efficiency—without explicit training for compression—provides evidence that efficiency principles may emerge spontaneously in sufficiently capable models. The contrast between Gemini's success and other models' convergence to low-complexity solutions raises important questions about what architectural or training properties enable human-like category emergence, questions that have implications for understanding both LLM cognition and the cognitive science of categorization.

## Potentially Missed Related Work
- None identified in the related work report.

## Suggestions
- Report the number of independent IICLL chains per condition explicitly, add error bars or confidence intervals to trajectory plots, and conduct significance tests for key comparisons between models and human baselines.
- Provide control experiments testing whether models can reproduce color naming systems from languages or synthetic systems unlikely to appear in training data, to strengthen the claim about inductive bias versus memorization.

# Actual Human Scores
Individual reviewer scores: [4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary

The paper introduces Direct Group Preference Optimization (DGPO), a novel RL method for post-training diffusion models that bypasses the policy-gradient framework used by prior GRPO-style methods. The key insight is that GRPO's effectiveness stems from group-level preference information rather than stochastic policies—DGPO enables efficient ODE-based sampling and directly optimizes group preferences via Bradley-Terry likelihood with advantage-based weighting. The method achieves approximately 20× faster training than Flow-GRPO while improving GenEval performance from 0.63 to 0.97.

## Strengths

- **Strong empirical results with clear efficiency gains**: DGPO achieves SOTA GenEval scores (0.97, surpassing GPT-4o's 0.84 and Flow-GRPO's 0.95) while training ~20× faster than Flow-GRPO, as demonstrated in Figures 1 and 3. The method maintains out-of-domain quality metrics (Aesthetic Score, DeQA, ImageReward, UnifiedReward) while improving task-specific rewards.

- **Well-motivated problem formulation**: The paper clearly articulates three fundamental issues with applying GRPO to diffusion models: (i) SDE-based rollouts are less efficient than ODE, (ii) stochasticity from model-agnostic Gaussian noise provides weak learning signal, and (iii) training over entire trajectories is computationally expensive.

- **Principled theoretical derivation**: The method is derived from the Bradley-Terry model through to a practical objective (Eq. 17), with the clever insight that advantage-based weights can eliminate the intractable partition function Z(c) by satisfying Σ_{G+} w = Σ_{G-} w.

- **Comprehensive experimental evaluation across three tasks**: Compositional image generation (GenEval), visual text rendering (OCR accuracy), and human preference alignment (PickScore) are all evaluated, with appropriate out-of-domain metrics to guard against reward hacking.

- **Effective practical techniques**: The Timestep Clip Strategy (sampling timesteps from [t_min, T]) addresses artifacts from few-step generation and is validated through ablation (Figure 4).

## Weaknesses

- **Missing statistical significance**: Tables 1 and 2 report single metric values without standard deviations or confidence intervals. Without error bars, readers cannot assess whether claimed improvements (e.g., 0.95 vs 0.97 on GenEval) are statistically meaningful.

- **Incomplete ablation on key hyperparameters**: Group size G=24, β=100, and t_min are used throughout but never systematically ablated. Understanding sensitivity to these hyperparameters is important for practical deployment and would strengthen claims about the method's robustness.

- **Bound tightness not analyzed**: The derivation uses Jensen's inequality to obtain an upper bound (Eq. 16), but how close this bound is to the original objective (Eq. 15) is neither theoretically analyzed nor empirically validated.

- **Edge case handling undefined**: When all samples in a group have positive (or all negative) advantages, the partition into G+ and G- would yield one empty group. This edge case is not discussed.

## Nice-to-Haves

- **Comparison with Diffusion-KTO**: The paper mentions KTO in related work but does not empirically compare against it. Since KTO also handles preference optimization without pairwise constraints, this comparison would help contextualize DGPO's contribution.

- **Memory requirements for practical deployment**: DGPO requires generating and storing G=24 samples simultaneously per prompt. Reporting GPU memory requirements would help practitioners assess practical feasibility.

## Novel Insights

The paper's core insight—that GRPO's effectiveness stems from fine-grained relative preference information within groups rather than from policy-gradient optimization—is significant and enables practical efficiency gains. This reconceptualization allows DGPO to combine the benefits of group-level preference learning (leveraging relative information across multiple samples) with the efficiency of direct preference optimization (avoiding stochastic policies and trajectory-wide training). The advantage-based weight design that eliminates the partition function is a clever technical solution that connects GRPO-style normalization with DPO-style direct optimization.

## Potentially Missed Related Work

- None identified beyond the concurrent work (Chen et al., 2025a) already discussed in the paper, which enumerates all pairwise comparisons rather than using a single group-level reward.

## Suggestions

- Add standard deviations or confidence intervals to Tables 1 and 2 from multiple experimental runs to establish statistical significance of improvements.

- Include an ablation study on group size G (e.g., G∈{8,16,24,32}) and β values to clarify hyperparameter sensitivity.

- Clarify the inconsistency between Figure 1 ("~30× Faster") and the text ("~20× faster")—either use consistent numbers or explain the difference (e.g., different metrics yield different speedup factors).

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
The paper introduces GROUNDCUA, a large-scale desktop GUI grounding dataset with 56K screenshots and over 3.56M human-verified element annotations across 87 applications. The authors also present GROUNDNEXT, a family of vision-language models (3B and 7B) trained via supervised fine-tuning and reinforcement learning, achieving state-of-the-art results on five benchmarks while using less than one-tenth the training data of prior work.

## Strengths
- **Significant dataset contribution**: GROUNDCUA fills a clear gap as the largest human-annotated desktop grounding dataset, with substantially denser annotations (64 elements per screenshot average) compared to OS-Atlas (7.8), UGround (11.6), and JEDI (7.0). The high-resolution coverage (0.5–7MP) and focus on small UI elements (average 0.13% of image area) addresses genuine challenges in desktop environments.

- **Strong empirical results with data efficiency**: GROUNDNEXT-3B achieves 66.4 average across five benchmarks, outperforming JEDI-3B (52.2) and GUI-Actor-3B (54.3) while using only 700K training samples versus JEDI's 9M. The controlled comparison (Figure 3) training Qwen2.5-VL-3B on 100K samples from different datasets provides clean evidence for data quality effects.

- **Comprehensive evaluation scope**: The paper evaluates across five diverse benchmarks (ScreenSpot-Pro, OSWorld-G, MMBench-GUI, ScreenSpot-v2, UI-Vision) covering desktop, mobile, and web domains. The agentic evaluation on OSWorld-Verified shows practical utility, with GROUNDNEXT-3B (50.6 overall) competing with the much larger JEDI-7B (51.0) using o3 as planner.

- **Insightful analysis of SFT-RL interaction**: The paper provides a thoughtful empirical analysis of why RL gains are modest when SFT is already strong, supported by evidence that models trained on GROUNDCUA show smaller RL gains than those trained on other datasets (Figure 3, Appendix D.6).

## Weaknesses
- **UI-Vision overlap not quantified**: The paper acknowledges UI-Vision overlaps with GROUNDCUA in platform coverage but treats it as "in-domain" without quantifying the exact overlap. Without explicit analysis of how many test samples use applications present in training data, the in-domain results may partially reflect memorization rather than generalization.

- **No statistical significance or uncertainty estimates**: All results are reported as point estimates without confidence intervals, standard errors, or significance tests. Differences of 1–3 points (e.g., 70.5 vs 69.2 average for RL vs SFT at 7B scale) are presented as meaningful without establishing they exceed measurement variance.

- **SFT instruction composition not ablated**: Table 7 shows individual instruction type performance (Functional best at 56.6, Spatial worst at 47.5), but the final 700K mix (50% Direct, 35% Functional, 15% Spatial) is presented without ablation against alternative compositions. Given Spatial underperforms, the 15% allocation requires justification.

- **No inter-annotator agreement metrics**: For a dataset paper whose central claim is quality over quantity, the absence of inter-annotator agreement metrics (IoU agreement, label consistency) is a notable omission. The 4% error rate from human evaluation (Section B.6) addresses instruction generation but not bounding box annotation quality.

- **No held-out application evaluation**: With 87 applications, aggregate results obscure whether the model learns transferable UI understanding or merely memorizes specific applications. A leave-one-app-out or held-out application evaluation would strengthen generalization claims.

## Nice-to-Haves
- The reward function thresholds (−0.5, −0.1, 0.1, 0.5) are motivated by error distribution analysis but not systematically ablated. Appendix C.5 ablates granularity levels but not threshold placement.
- The 700K subset is described as having "balanced coverage and diversity" but the exact sampling strategy and balancing criteria are not detailed, limiting reproducibility of the exact SFT data.

## Novel Insights
The paper surfaces an important empirical finding: when SFT data is sufficiently high-quality and diverse, it creates a "strong ceiling" that leaves fewer actionable errors for RL to correct. This is supported by the observation that models trained on GROUNDCUA during SFT show smaller RL gains than those trained on noisier datasets. The controlled experiment (Figure 3, Appendix D.6) demonstrates this interaction systematically, challenging the assumption that RL universally provides substantial gains regardless of SFT quality. This has practical implications for resource allocation in model development.

## Potentially Missed Related Work
- None identified. The related work section adequately covers GUI grounding datasets (RICO, UIBert, AMEX, SeeClick, AriaUI, OS-Atlas, UGround, JEDI) and computer-use agent models (CogAgent, ShowUI, Ferret-UI, OS-ATLAS, GUI-Actor, etc.).

## Suggestions
1. **Quantify UI-Vision overlap**: Report the number or percentage of UI-Vision test samples whose applications appear in GROUNDCUA, and ideally evaluate on a held-out subset of applications not seen during training.

2. **Add confidence intervals**: Report standard deviations or confidence intervals across multiple runs for key benchmark results, particularly when claiming improvements over baselines.

3. **Ablate instruction composition**: Provide an ablation showing performance with different Direct/Functional/Spatial proportions to justify the 50/35/15 split, especially given Spatial's lower individual performance.

4. **Report inter-annotator agreement**: Include IoU agreement metrics for bounding box annotations and label consistency statistics to substantiate the quality claims central to the paper's contribution.

5. **Conduct held-out application evaluation**: Train on a subset of the 87 applications and evaluate on the remainder to demonstrate genuine transfer rather than application-specific learning.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0, 6.0]
Average score: 5.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
The paper proposes History-Guided Sampling (HiGS), a training-free method that enhances diffusion model sampling by incorporating the difference between current predictions and an EMA-weighted average of past predictions. The method includes additional components: time-varying weight schedules, optional orthogonal projection, and DCT-based high-pass filtering. HiGS achieves consistent quality improvements across diverse models and sampling regimes, with a state-of-the-art FID of 1.61 for unguided ImageNet generation using only 30 steps.

## Strengths
- **Plug-and-play applicability without retraining**: HiGS requires no model fine-tuning and works across diverse architectures (DiT, SiT, SDXL, SD3, SD3.5, Flux) and sampling regimes (standard sampling, distilled models, various NFEs and CFG scales), as demonstrated in Tables 1-4.

- **Strong empirical results with efficiency gains**: Table 3 shows HiGS achieves FID 1.61 on ImageNet 256×256 with 30 steps versus baseline FID 1.83 with 250 steps—a substantial efficiency improvement. Tables 1-2 show consistent HPSv2 win rates exceeding 80% across Stable Diffusion variants.

- **Negligible computational overhead**: Appendix D explicitly confirms that inference speed matches standard CFG (~6.50 iterations/second on SD3), as HiGS only adds buffer operations and DCT filtering without additional forward passes.

- **Comprehensive ablation studies**: Appendix E thoroughly evaluates design choices (EMA vs. averaging, projection necessity, DCT filtering, weight schedules, EMA parameter α, DCT threshold R_c) with supporting figures demonstrating their impact.

- **Theoretical grounding for core mechanism**: Appendix B proves that the history-based update reduces local truncation error from O(h²) to O(h³) compared to Euler sampling, with corresponding global error reduction from O(h) to O(h²), explaining why fewer steps with HiGS can match or exceed more steps without.

## Weaknesses
- **Theoretical analysis does not cover full implementation**: Appendix B proves error reduction for the basic momentum update (Equation 17), but the actual HiGS algorithm includes DCT filtering and optional orthogonal projection, which are not analyzed. The connection between the O(h²) global error claim and the practical implementation remains unclear.

- **Missing comparison to related plug-and-play guidance methods**: No empirical comparison to autoguidance (Karras et al., 2024), guidance interval scheduling, or other training-free enhancements. This limits the ability to position HiGS relative to existing methods in the same category.

- **No statistical significance testing**: All FID, IS, Precision, and Recall values in Tables 1-3 are reported without error bars or confidence intervals. Given FID's known sensitivity to implementation details and random seeds, the statistical significance of reported improvements cannot be verified.

- **Substantial hyperparameter tuning overhead**: The method introduces six hyperparameters (α, t_min, t_max, w_HiGS, η, R_c) with Tables 10-12 showing different optimal values per model. This complicates the "plug-and-play" claim and raises questions about tuning requirements for new domains.

## Nice-to-Haves
- **Ablation on history window size W**: The paper uses all past predictions but does not analyze whether smaller windows (e.g., last 1-3 steps) would suffice. Understanding this would clarify whether the "momentum" framing or simpler recency weighting is the key mechanism.
- **Failure case analysis**: No discussion of scenarios where HiGS might degrade quality or introduce artifacts, nor quantitative diversity evaluation beyond Recall to validate the claim that diversity is preserved while avoiding high-CFG oversaturation.

## Novel Insights
The paper makes an original connection between momentum-based variance reduction in optimization (STORM) and diffusion sampling. The key insight is that the prediction difference ΔD_{t_k} = D_CFG(z_{t_k}) - g(H_k) approximates a quality-improving direction in the sampling space. The ODE interpretation in Section 4.1—showing that Euler sampling corresponds to gradient descent on a time-varying energy function—is elegant and provides theoretical grounding. The observation that DCT filtering removes "unrealistic color compositions" from the history signal suggests the prediction history contains both structural detail information (high-frequency, beneficial) and color/mean information (low-frequency, potentially harmful). However, the gap between the theoretical error analysis (pure momentum update) and the full algorithm (with DCT/projection corrections) leaves open the question of which components are theoretically justified versus empirically motivated.

## Potentially Missed Related Work
- **Autoguidance (Karras et al., 2024)**: A training-free guidance method using a weaker model to improve generation quality. Appendix C briefly discusses HiGS's relation to autoguidance but provides no empirical comparison. Comparison would contextualize HiGS among plug-and-play enhancements.
- **Perturbed Attention Guidance (PAG)** and other training-free guidance techniques: Not discussed despite being relevant baselines for sampling enhancement without retraining.

## Suggestions
- Add empirical comparison to autoguidance on identical benchmarks to establish HiGS's relative merit as a plug-and-play enhancement.
- Report FID, IS, and preference metric results with error bars or confidence intervals across multiple random seeds to establish statistical significance.
- Conduct ablation on history window size to determine whether full history is necessary or whether recent predictions (1-3 steps) suffice, which would simplify the method and clarify the mechanism.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary

The paper introduces Latent Basis Function Neural Posterior Estimation (LBF-NPE), a variational family for amortized inference that parameterizes log densities through linear combinations of basis functions. The method represents the posterior as an exponential family where coefficients are predicted by a neural network from observations, and basis functions can be either fixed (B-splines, wavelets) or learned adaptively. The key insight is that this parameterization yields marginally convex optimization landscapes when either the basis functions or coefficients are held fixed, offering a middle ground between simple Gaussian families (stable optimization but limited expressiveness) and normalizing flows (expressive but complex optimization).

## Strengths
- **Novel contribution**: LBF-NPE is the first work to apply neural exponential families with basis expansions to amortized posterior estimation in the NPE framework, filling a genuine gap between simple and complex variational families.

- **Theoretical grounding**: Proposition 1 proves marginal convexity in both the coefficient network f and basis function network s. For fixed basis functions, Appendix B connects to NTK theory to establish global convergence guarantees—a notable theoretical result for NPE methods.

- **Empirical effectiveness**: Table 1 shows LBF-NPE achieves order-of-magnitude improvements in KL divergence over MDN and substantial improvements over neural spline flows across 2D benchmarks. Figure 1 demonstrates that LBF-NPE consistently converges to the same optimum across 20 random seeds, while MDN frequently converges to inferior local minima.

- **Real-world validation**: The redshift estimation experiment on LSST DESC DC2 data (Table 2) demonstrates practical utility, with LBF-NPE achieving lower NLL (-57,220) than NSF (-55,389) and MDN (-50,648) on 153,000 test objects.

- **Clear algorithmic presentation**: Algorithm 1 and Appendix D provide sufficient implementation details for reproduction, with code availability.

## Weaknesses
- **Theoretical-experimental mismatch**: Proposition 1's convexity guarantee applies only when basis functions are fixed, yet most experiments (Sections 6.2, 6.3) use adaptive basis functions. The primary experimental setting thus lacks the theoretical guarantees claimed as a motivation.

- **Limited high-dimensional validation**: The abstract claims computational efficiency "even for problems with high-dimensional latent spaces," but all main experiments are 1D or 2D. The 50D experiment in Appendix E.4 uses a highly structured annulus model without comparison to baselines. Systematic experiments on 5D-20D posteriors with complex geometry would better substantiate scalability claims.

- **Underspecified implementation details**: The proposal distribution r(z) for SNIS gradient estimation (Algorithm 1) is critical for performance but not discussed. The choice of base measure h(z) is mentioned only abstractly. These omissions affect reproducibility.

- **Missing calibration analysis**: The paper reports KL divergence and NLL but no calibration metrics (simulation-based calibration, coverage probability). For Bayesian inference applications, demonstrating well-calibrated posteriors is essential.

## Nice-to-Haves
- **Guidance on basis function selection**: The paper proposes both fixed (B-splines, wavelets) and adaptive basis functions but provides limited guidance on when to use which. Practitioners would benefit from heuristics based on problem characteristics.

- **Sample complexity comparison**: No analysis of how many simulation samples each method requires for convergence. Since simulation cost often dominates in SBI applications, this comparison would help practitioners assess practical tradeoffs.

## Novel Insights

The key insight is that exponential family parameterizations of variational posteriors inherit favorable optimization properties when the sufficient statistics (basis functions) are decoupled from the natural parameters (coefficients). By parameterizing these components with separate networks and exploiting the marginal convexity in each, LBF-NPE avoids the local minima that plague mixture models and normalizing flows while maintaining expressiveness. The stereographic projection reparameterization (Section 4.4) stabilizes training of the adaptive variant, and the empirical analysis (Appendix E.1) shows it produces more structured, interpretable basis functions. The method is particularly well-suited for low-dimensional posterior projections—exactly the setting common in scientific inference where nuisance variables are marginalized out.

## Potentially Missed Related Work
- None identified.

## Suggestions
- Add systematic experiments on 5D, 10D, and 20D posteriors with varying complexity to substantiate scalability claims and clarify practical dimension limits.

- Include simulation-based calibration (SBC) or posterior predictive checks to validate that the posteriors are statistically calibrated for downstream inference.

- Provide explicit guidance on choosing the proposal distribution r(z) for SNIS, and analyze its sensitivity to misspecification.

- Conduct an ablation comparing fixed vs. adaptive basis functions on the same problem to clarify the expressivity vs. optimization stability tradeoff.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept

=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
## Summary

The paper proposes NCRL, a method for leveraging non-curated offline data—reward-free, mixed-quality, and multi-embodiment—to improve sample efficiency in online reinforcement learning. The key insight is that naive world model fine-tuning fails due to distributional shift between offline pre-training data and online fine-tuning data. The authors address this through experience rehearsal (retrieving task-relevant trajectories during fine-tuning) and execution guidance (using a behavior-cloned prior policy to guide exploration), achieving nearly 2× improvement in aggregate score over training-from-scratch baselines across 72 visuomotor tasks.

## Strengths

- **Practical problem setting**: The paper addresses a realistic and valuable problem—leveraging abundant non-curated offline data—expanding the pool of usable data beyond curated reward-labeled datasets common in prior offline RL work.

- **Strong empirical coverage**: Evaluation on 72 tasks spanning Meta-World manipulation and DMControl locomotion benchmarks is comprehensive. NCRL achieves 0.748 vs 0.360 (DreamerV3) and 0.430 (DrQ-v2) success rates on Meta-World at 150k samples, approximately matching baseline performance at 3.3-6.7× more samples.

- **Clear problem identification**: The paper identifies three failure modes of naive world model fine-tuning—model degradation from distributional shift, catastrophic forgetting on narrow online data, and limited rollout diversity from narrow initial states—supported by t-SNE visualization and Wasserstein distance measurements (Figure 2).

- **Diverse baselines**: Comparisons against representation learning (R3M), offline-to-online methods (UDS-RLPD, ExPLORe, JSRL-BC), and world model pre-training approaches (iVideoGPT, DreamerV3 w/ pre-training) provide meaningful context. Notably, baselines receive preprocessed task-relevant data while NCRL handles raw multi-embodiment data, making NCRL's improvements more impressive.

## Weaknesses

- **Limited ablation scope**: Figure 6 shows ablations on only 4 tasks out of 72. Given that the method has three main components (pre-training, experience rehearsal, execution guidance) addressing distinct failure modes, more comprehensive ablation coverage is needed to validate component contributions across the full task spectrum.

- **No analysis of world model prediction quality**: The core mechanism—pre-trained world models improving RL—assumes the world model learns useful dynamics. However, the paper never measures prediction accuracy on downstream tasks, leaving the connection between world model quality and policy improvement unvalidated.

- **Single-task distributional shift demonstration**: Figure 2 visualizes distributional shift for only Shelf Place. Without analysis across multiple tasks—particularly tasks where NCRL succeeds versus fails—it remains unclear whether distributional shift is consistently the actual failure cause.

- **Incomplete statistical reporting**: Tables 3-5 report only mean performance without confidence intervals or standard deviations, making statistical significance of improvements difficult to assess despite 3-5 seeds per method.

- **Partial iVideoGPT comparison**: The aligned comparison against iVideoGPT (Figure 10) covers only 2 tasks, while the main comparison (Figure 4) uses iVideoGPT's original results with reward shaping and demonstration pre-filling that NCRL doesn't use. A more comprehensive aligned comparison would strengthen the claims.

## Nice-to-Haves

- **Failure mode analysis**: Tasks like Disassemble (0% success), Soccer (16%), and Pick Place (20%) perform poorly. Understanding why—whether coverage gaps in offline data, retrieval failures, or inherent task difficulty—would clarify the method's limitations.

- **World model prediction visualizations**: Showing predicted versus ground-truth future frames would reveal what dynamics information transfers across embodiments and what doesn't, providing insight into what makes non-curated data useful.

## Novel Insights

The paper's key insight is that distributional shift between offline pre-training data and online fine-tuning data causes world model degradation in three specific ways: (i) accuracy degradation when offline coverage is narrow, (ii) catastrophic forgetting when online data distribution is narrow, and (iii) limited rollout diversity for policy updates. The proposed solutions directly target each failure mode—experience rehearsal addresses (i) and (ii) by maintaining training on retrieved offline data, while execution guidance addresses (iii) by steering exploration toward regions where the world model has high confidence. This mechanistic decomposition is valuable for understanding when and why world model pre-transfer succeeds or fails.

## Potentially Missed Related Work

- None identified. The related work section covers offline RL, multi-task offline RL, unsupervised RL, generalist agents, and world model approaches comprehensively.

## Suggestions

- Expand ablation studies to at least 10-12 diverse tasks (including both successful and challenging tasks) to validate component contributions across the full benchmark.

- Add world model prediction error metrics (e.g., MSE on held-out trajectories from downstream tasks) to empirically validate that the pre-trained world model transfers useful dynamics knowledge.

- Include confidence intervals in all summary tables to enable proper statistical comparison.

# Actual Human Scores
Individual reviewer scores: [6.0, 10.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
