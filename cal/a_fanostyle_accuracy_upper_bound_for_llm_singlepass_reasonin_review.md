=== CALIBRATION EXAMPLE 53 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the paper's dual contributions: a theoretical bound and a proof-of-concept framework. The abstract is well-structured, clearly stating the problem (single-pass capacity bottleneck), the theoretical contribution (Fano-style bound revealing an Accuracy Cliff), and the practical outcome (InfoQA framework). All claims are supported by the paper's content.

**Introduction & Motivation:** The introduction effectively motivates the problem. It clearly explains why multi-hop QA (MHQA) is challenging for LLMs, introducing the intuitive concept of a finite per-pass output capacity. It cleanly transitions from this intuition to the need for a formal theoretical analysis, setting the stage for the bound. The contributions are listed explicitly and correspond to the paper's structure.

**Method / Approach (Sections 2 & 3):**
*   **Theoretical Analysis (Sec 2):** The formalization is rigorous and builds correctly on standard information-theoretic tools (Conditional Fano Inequality, Output Entropy Bound). Theorem 1 is the core result, and its derivation in the appendix appears sound. A major strength is the translation of the precise bound into intuitive, testable corollaries (Eq. 4, 5) and the compelling "Accuracy Cliff" visualization (Fig 2). This makes the theory accessible and actionable.
*   **Key Concern:** The most significant assumption is modeling the LLM's *output* `Y` as the information-carrying channel, with capacity `C = H(Y)`. The paper bounds `H(Y)` by maximum token length/vocabulary size. However, this treats every output token as potentially carrying independent information about the answer `A`. In reality, LLM outputs are highly structured and correlated, and the *relevant* information `I(A; Y | Q, C)` may be far less than `H(Y)`. The chosen bound `I(A; Y) ≤ H(Y)` is loose. The theory would be more compelling if it engaged with or justified this looseness—e.g., arguing that for the purpose of an *upper bound* on accuracy, using the maximum possible entropy is a conservative (and thus valid) choice, even if it may overestimate the true capacity `C`. This is a nontrivial point that reviewers are likely to question.
*   **Anatomy of MHQA (Sec 3):** This section successfully bridges theory and practice. The parametric model for information demand `β(h, L)` (Eq. 6) is a sensible, simplified operationalization. It directly leads to the testable plug-in bound (Eq. 7). The analysis of Cross-Step Error Accumulation is clear and correctly highlights the compounding nature of the problem. The dual-crisis framing is effective.

**Method / Approach (Section 4 - InfoQA):** The framework is a logical, direct application of the principles derived from the theory. The three components (Capacity-Aware Decomposition, Dependency-Explicit Workflow, Iterative Query Contraction) are clearly described and aligned with addressing the identified challenges. As a proof-of-concept, its design is sound. A minor point: the term "multi-call" is used interchangeably with "multi-step"; clarifying that "call" refers to a separate LLM invocation would prevent ambiguity.

**Experiments & Results (Section 5):**
*   **Benchmark & Setup:** The decision to create a new synthetic benchmark is justified based on the need for fine-grained control over `h` and `L`. The construction principles (systematic control, high semantic similarity, path maximization) are well-explained in the appendix and are appropriate for testing the theory. Using publicly available models (Qwen) is good practice.
*   **Theory Validation (Fig 5, Parameter Fitting):** This is a central part of the paper. The protocol for fitting the parameters `(α, γ, β0, C)` to empirical F1 scores is detailed in the appendix. The results show a striking qualitative alignment between the theoretical cliff curves and the empirical performance drop across methods. This strongly supports the core theoretical claim.
    *   **Concern 1:** The fitting procedure uses F1 as a proxy for accuracy `Acc` and fits the bound *to the data*. Since the bound is an *upper limit*, the fact that empirical points lie at or below the fitted curve is expected. However, a skeptic might argue that with four free parameters (`α, γ, β0, C`), fitting a curve to 24 data points could lead to overfitting, making the alignment less surprising. The authors should discuss the robustness of the fit (e.g., via cross-validation or by showing fits on a held-out subset of `(h, L)` combinations). The low MAE values are encouraging but not definitive proof against overfitting.
    *   **Concern 2:** The interpretation of fitted parameters (`C`, `γ`) as "effective single-pass capacity" and "hop inflation" is insightful but post-hoc. The paper should more explicitly state that these are *interpretations* of the best-fit parameters within the model, not direct measurements of fundamental properties.
*   **Framework Validation & Ablations:** The results in Table 2 are comprehensive. InfoQA's superior and more robust performance, especially on high-hop/long-context tasks, validates the practical utility of the multi-call paradigm. The ablation study (w/o D., w/o P.) cleanly demonstrates the contribution of each key component. The error analysis is thoughtful, correctly identifying residual failure modes (semantic drift, base comprehension limits) that are distinct from capacity overflow.

**Writing & Clarity:** The paper is generally well-written. The progression from theory to problem analysis to solution is logical. Figures are helpful. Some minor notes: The initial explanation of the output capacity bound in Section 2.1 is very brief; referencing the detailed appendix earlier might help. The phrase "Fano-style" is used appropriately.

**Limitations & Broader Impact:**
*   **Limitations:** The paper acknowledges some limitations in the error analysis (semantic drift, base model capacity). However, it should more explicitly discuss the limitations of its **theoretical modeling**: the looseness of the `H(Y)` bound as discussed above, the simplifications in the `β(h,L)` model (e.g., linear scaling with `L`, exponential scaling with `h`), and the fact that the theory applies to a *closed-book* setting. The impact of the model's *internal* reasoning capacity (separate from output length) is also not addressed by the theory.
*   **Broader Impact:** The ethics statement is appropriate for the synthetic data used. The broader impact of promoting more reliable, multi-step LLM reasoning systems is positive. The reproducibility statement is excellent.

### Overall Assessment
This is a strong, theoretically grounded paper with clear empirical validation. The core idea—formalizing the single-pass output capacity bottleneck via an information-theoretic bound—is novel, significant, and well-executed. The derivation of the "Accuracy Cliff" is a compelling and useful conceptual tool. The experimental validation, while relying on a synthetic benchmark and parameter fitting, is carefully done and provides convincing evidence for the theory. The InfoQA framework effectively demonstrates the practical implications. The main reservations are the looseness of the key theoretical bound `I(A;Y) ≤ H(Y)` and the potential for overfitting in the parameter estimation. However, the overall contribution—a principled explanation for the failure modes of single-pass reasoning and a paradigm shift towards capacity-aware multi-call design—stands and meets the high bar of ICLR. Addressing the noted concerns in a revision would strengthen the paper further.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents an information-theoretic analysis of the limitations of single-pass reasoning in large language models (LLMs) for multi-hop question answering (MHQA). The authors derive a Fano-style accuracy upper bound, demonstrating that performance collapses sharply (an "Accuracy Cliff") when task information demand exceeds the model's per-pass output capacity. Motivated by this bound, they propose InfoQA, a multi-call reasoning framework that decomposes tasks, prunes reasoning traces, and contracts queries to keep per-step information load manageable, and validate both the theory and the framework on a controlled synthetic benchmark.

### Strengths
1. **Theoretical novelty and rigor**: The paper introduces a novel information-theoretic bound (Theorem 1) grounded in Fano's inequality and the output entropy bound, formalizing the concept of an "Accuracy Cliff" in LLM single-pass reasoning. This provides a principled explanation for the observed performance degradation in complex MHQA tasks.
2. **Clear identification of key challenges**: The work convincingly dissects MHQA into two compounding challenges—Stepwise Capacity Overflow (due to super-linear growth in information demand) and Cross-Step Error Accumulation—offering a structured understanding of why single-pass reasoning fails.
3. **Well-motivated and transparent framework**: InfoQA is directly derived from the theoretical insights, featuring capacity-aware decomposition, dependency-explicit workflow, and iterative query contraction. The design is clearly explained and aligns with the theoretical principles.
4. **Rigorous and controlled experimental validation**: The authors construct a synthetic benchmark that systematically varies hop count and context length, enabling precise testing of the theoretical predictions. The fitting of empirical F1 scores to the derived bound (Figure 5) shows strong alignment, providing solid evidence for the Accuracy Cliff phenomenon.
5. **Comprehensive evaluation and ablation**: The paper compares InfoQA against a wide range of single-pass baselines and includes informative ablations (e.g., without decomposition or pruning), demonstrating the contribution of each component and the framework's robustness across depths and context lengths (Table 2).
6. **Strong reproducibility**: The paper includes a reproducibility statement, detailed appendices with proofs, benchmark construction algorithms, and fitting procedures, and promises code and data release, facilitating replication.

### Weaknesses
1. **Limited empirical scope on real-world data**: The experiments are conducted solely on a synthetic benchmark. While this allows fine-grained control, it remains unclear whether the Accuracy Cliff and InfoQA's advantages generalize to established, real-world MHQA datasets (e.g., HotpotQA), where noise and reasoning patterns may differ.
2. **Narrow model evaluation**: Experiments are limited to the Qwen model family (8B and 14B parameters). The theory claims universality, but validation across diverse architectures (e.g., Llama, GPT) would strengthen the claim that the capacity bottleneck is inherent to the single-pass paradigm, not model-specific.
3. **Ambiguous treatment of multi-call baselines**: The paper categorizes Self-Ask as a single-pass baseline and states that all baselines are implemented as "zero-shot, single-pass methods." However, Self-Ask is inherently a multi-call method. This inconsistency should be clarified, and a more direct comparison to existing multi-call approaches (beyond Self-Refine) would better situate InfoQA.
4. **Theoretical assumptions and metric mismatch**: The bound assumes a closed-book setting and, in the simplified form, a near-uniform answer distribution. The practical validity of these assumptions is not thoroughly discussed. Additionally, the empirical fitting uses F1 as a proxy for accuracy without justification; a direct accuracy metric would be more appropriate for testing an accuracy bound.
5. **Superficial error analysis**: The error analysis identifies semantic drift and intrinsic capacity limits as failure modes for InfoQA but does not quantify their prevalence or provide concrete examples, limiting insights for future improvement.

### Novelty & Significance
The paper makes a significant contribution by providing a formal information-theoretic foundation for understanding LLM reasoning capacity bottlenecks, a topic of high interest in the community. The Accuracy Cliff concept and the decomposition of MHQA challenges are novel and insightful. The work successfully bridges theory and practice by deriving a concrete framework from the theoretical analysis. This perspective is likely to influence the design of future multi-step reasoning methods and stimulate further theoretical analysis of LLM capabilities.

### Suggestions for Improvement
1. **Validate on real-world benchmarks**: Supplement the synthetic experiments with results on at least one standard MHQA dataset (e.g., HotpotQA or 2WikiMultiHopQA) to demonstrate the practical relevance and generalizability of the theory and framework.
2. **Expand model coverage**: Run key experiments (e.g., the Accuracy Cliff validation) on additional LLM families (e.g., Llama 3, Mistral) to confirm that the phenomenon is not architecture-specific.
3. **Clarify baseline implementations and include multi-call comparisons**: Explicitly describe how Self-Ask was adapted to a single-pass setup or re-implement it as a true multi-call baseline. Also, compare InfoQA to other recent multi-call reasoning frameworks to better highlight its unique contributions.
4. **Address theoretical assumptions and metrics**: Discuss the limitations of the closed-book and uniformity assumptions. In experiments, report accuracy (exact match) alongside F1 to more directly test the theoretical bound.
5. **Deepen error analysis**: Quantify the proportion of errors attributable to semantic drift versus intrinsic capacity limits, and provide concrete examples of each from the benchmark to guide future improvements.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validate on established, real-world MHQA datasets (e.g., HotpotQA, MuSiQue).** The theory and method are only tested on a synthetic benchmark. Without showing that the Accuracy Cliff phenomenon and InfoQA's advantages hold on real data, the practical relevance and generalizability of the contributions are severely undermined.
2. **Test on a diverse set of LLM families and scales.** Experiments are limited to Qwen3 models. To support claims about universal capacity limits and paradigm superiority, results must be shown across varied architectures (e.g., Llama, Gemma, GPT) and sizes to rule out model-specific biases.
3. **Include strong multi-call baselines for comparison.** The paper only compares against single-pass methods. To substantiate that InfoQA's design is superior, it must be benchmarked against other multi-call frameworks (e.g., ReAct or Self-Refine in their intended multi-step setups) to isolate the benefit of its specific components.
4. **Perform a comprehensive ablation study.** The current ablation only removes decomposition and pruning. Critical components like the dependency-explicit workflow and the contraction mechanism lack isolated evaluations, making it impossible to attribute gains to specific design choices.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the tightness and robustness of the theoretical bound.** The paper fits parameters from performance data, creating a circular argument. An independent, predictive estimation of information demand (β) and capacity (C) is needed to truly validate the theory. Furthermore, discuss how violations of key assumptions (e.g., uniform answer distribution) affect the bound's applicability.
2. **Provide a detailed, quantitative error analysis for InfoQA.** The error discussion is qualitative and vague. A systematic categorization of failure modes (e.g., percentages due to semantic drift, base comprehension errors, or error propagation) across different task complexities is essential to understand the method's limitations and guide improvement.
3. **Clarify the relationship between fitted parameters and actual model properties.** The parameters (γ, C, β0) are curve-fitting outputs. The paper must explain what these fitted values correspond to in terms of measurable model or task characteristics (e.g., is C related to max token length?) to move from a descriptive fit to an explanatory theory.

### Visualizations & Case Studies
1. **Visualize side-by-side reasoning traces for success and failure cases.** Concrete examples comparing single-pass and InfoQA outputs are needed to illustrate *how* capacity overflow manifests (e.g., truncated or garbled reasoning) and *how* InfoQA's decomposition and pruning avert it. This would make the theoretical claims tangible.
2. **Plot per-step information load (estimated β_k) against capacity C for sample queries.** A visualization showing how the information demand evolves across hops in single-pass vs. InfoQA would directly demonstrate the core mechanism of keeping per-step demand below capacity.

### Obvious Next Steps
1. **Derive a theoretical bound for the multi-call paradigm.** The paper's core theory only addresses the single-pass limit. A natural and critical extension is to formulate a capacity law for multi-call reasoning, showing how decomposition alleviates the bottleneck and what new limits might emerge.
2. **Incorporate an adaptive decomposition strategy.** The current decomposition is fixed (single-hop). A direct application of the theory would be to dynamically adjust the granularity of sub-questions based on estimated information demand, making the framework more robust and efficient.

# Final Consolidated Review
## Summary
This paper provides an information-theoretic analysis of the fundamental limitations of single-pass reasoning in Large Language Models for Multi-Hop Question Answering (MHQA). It derives a Fano-style upper bound on accuracy, formalizing an "Accuracy Cliff" where performance collapses when a task's information demand exceeds a model's per-pass output capacity. Motivated by this theory, the authors introduce InfoQA, a proof-of-concept multi-call framework that decomposes tasks and manages information load to circumvent the single-pass bottleneck.

## Strengths
- **Novel Theoretical Foundation:** The paper provides a rigorous, novel information-theoretic analysis that formalizes an intuitive but previously unquantified bottleneck. The derivation of a Fano-style accuracy upper bound (Theorem 1) and the resulting "Accuracy Cliff" concept offer a principled explanation for performance degradation in complex reasoning tasks.
- **Bridging Theory and Practice:** The work successfully connects theory to actionable design principles. The decomposition of MHQA challenges into "Stepwise Capacity Overflow" and "Cross-Step Error Accumulation" directly informs the design of the InfoQA framework, whose components (capacity-aware decomposition, dependency-explicit workflow, iterative query contraction) are clearly motivated by the theoretical insights.
- **Rigorous and Controlled Empirical Validation:** The construction of a synthetic benchmark allows for precise, systematic control over key variables (hop count, context length) to test the theoretical predictions. The protocol for fitting empirical performance to the theoretical curves demonstrates a strong qualitative alignment with the predicted Accuracy Cliff across multiple single-pass methods, providing compelling evidence for the core theory.
- **Comprehensive Evaluation and Ablations:** InfoQA is evaluated against a wide suite of strong single-pass baselines and shows consistent, robust performance gains, especially on high-hop and long-context tasks. The ablation studies effectively demonstrate the contribution of its core components (decomposition and pruning).

## Weaknesses
- **Limited Generalization Evidence:** The empirical validation is conducted exclusively on a controlled, synthetic benchmark. While this is justified for testing the theory, it leaves open the question of whether the Accuracy Cliff phenomenon and InfoQA's advantages generalize to established, real-world MHQA datasets (e.g., HotpotQA) with different noise and reasoning patterns.
- **Narrow Model Validation for a Universal Claim:** The theory posits a universal limit of the single-pass paradigm. However, experiments are limited to the Qwen model family. To strongly support the claim that the bottleneck is inherent to the paradigm and not model-specific, validation across diverse architectures (e.g., Llama, GPT) is needed.
- **Theoretical Bound is a Loose Upper Limit:** The core capacity bound, \( C = H(Y) \), uses the maximum entropy of the output sequence, which is a conservative (loose) upper bound on the mutual information \( I(A; Y) \). The paper acknowledges this but does not deeply discuss the implications of this looseness for the practical tightness of the predicted cliff or explore potential refinements.

## Nice-to-Haves
- **Visualizations of Failure Modes:** Side-by-side examples of single-pass vs. InfoQA reasoning traces, illustrating how capacity overflow manifests (e.g., garbled reasoning) and how decomposition averts it, would make the theoretical claims more tangible.
- **Adaptive Decomposition:** An adaptive strategy that dynamically adjusts sub-question granularity based on estimated information demand would be a direct and valuable application of the theoretical framework.
- **Deeper Error Analysis:** A more quantitative breakdown of InfoQA's residual errors (e.g., percentage due to semantic drift vs. base comprehension failure) would provide clearer guidance for future improvements.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Ambiguous treatment of multi-call baselines/Self-Ask."** The paper states all baselines were implemented as "zero-shot, single-pass methods." While the adaptation of Self-Ask to a single-pass setup could be clarified, this is an implementation detail that does not invalidate the core comparison between paradigms.
- **Weakness: "Theoretical assumptions and metric mismatch (F1 vs. Accuracy)."** Using F1 score as a proxy for accuracy is standard in QA evaluation and is a reasonable approximation for testing the theoretical trend. A direct accuracy metric would not change the fundamental results.
- **Weakness: "Parameter fitting is circular/overfitted."** The fitting procedure uses multiple parameters, but the consistent, interpretable patterns across methods (e.g., CoT increasing effective capacity `C`) and the validation on a separate model (Qwen3-8B in the appendix) suggest the fits capture meaningful trends rather than mere overfitting.
- **Suggestion: "Derive a theoretical bound for the multi-call paradigm."** This is an interesting research direction but is a clear extension beyond the paper's stated scope of analyzing the single-pass bottleneck.

## Novel Insights
The paper's central novel insight is the formalization of the single-pass output channel as a finite-capacity information bottleneck for LLM reasoning. This perspective cleanly explains why performance does not degrade gracefully but hits an "Accuracy Cliff," and it fundamentally shifts the design goal from improving single-pass prompts to managing per-step information demand via decomposition. The decomposition of MHQA failures into compounding "Capacity Overflow" and "Error Accumulation" provides a structured framework for diagnosing and addressing reasoning failures in LLMs.

## Suggestions
- **Demonstrate Generalizability:** Include an experiment on at least one standard, real-world MHQA dataset (e.g., HotpotQA or MuSiQue) to show that the practical performance advantages of the multi-call paradigm hold outside the synthetic controlled environment.
- **Expand Model Validation:** Perform the key Accuracy Cliff validation experiment (fitting the bound) on one additional, architecturally different LLM family (e.g., Llama 3) to strengthen the claim of paradigm-level universality.
- **Discuss Theoretical Tightness:** In the limitations or discussion, add a brief paragraph more explicitly addressing the conservatism of the \( H(Y) \) bound and what it implies for the practical predictive power of the theory.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
