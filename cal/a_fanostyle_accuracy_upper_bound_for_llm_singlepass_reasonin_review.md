=== CALIBRATION EXAMPLE 57 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the paper's dual contribution: a theoretical bound and a framework (InfoQA). The abstract clearly states the core problem (single-pass capacity overflow), the theoretical contribution (Fano-style bound revealing an "Accuracy Cliff"), and the practical framework designed to circumvent it. Claims are specific and map directly to the paper's sections.

**Introduction & Motivation:** The introduction effectively motivates the problem. It clearly explains the inherent limitation of LLMs in single-pass, multi-hop reasoning due to finite output capacity and provides an intuitive graphical summary (Fig 1). The contributions are stated precisely and map to subsequent sections. The transition from the identified bottleneck to the need for a new paradigm is logical.

**Method / Approach (Sections 2, 3, 4):**
*   **Section 2 (Theoretical Bound):** The formalization is clear. The derivation of Theorem 1 from the Conditional Fano Inequality and Output Entropy Bound is sound in its information-theoretic reasoning. The core insight—that perfect accuracy is impossible when information demand (β) exceeds output capacity (C)—is well-founded. The subsequent corollaries (Eq. 4, Eq. 5) and the conceptual "Accuracy Cliff" (Fig. 2) effectively translate the theorem into an intuitive, testable phenomenon. A key concern is the **operationalization of "output capacity" C = H(Y)**. The paper correctly notes it can be bounded by sequence length and vocabulary size, but in practice, for a given model and decoding strategy, *H(Y|Q,C)* is not a fixed, known constant but a property of the model's distribution. The experiments later *fit C* as a parameter, which is a valid way to test the theory's *shape*, but it means C is an empirical property of the method, not a pre-computable limit like a context window. This should be more explicitly discussed as a bridge between theory and experiment.
*   **Section 3 (Anatomy of MHQA):** This is a strong section that contextualizes the abstract bound. Modeling β as a function of hop count *h* and context length *L* (Eq. 6) is sensible, and the identification of "Stepwise Capacity Overflow" and "Cross-Step Error Accumulation" precisely articulates why MHQA hits the Accuracy Cliff. The mathematical model for error accumulation (Eq. 8-10, Fig. 3) clearly shows the compounding crisis. The assumption of a uniform per-step error rate ε for Eq. 10 is a simplification but sufficient to illustrate the point.
*   **Section 4 (InfoQA Framework):** The framework is a logical, direct implementation of the principles derived from the theory. The three components (Capacity-Aware Task Decomposition, Dependency-Explicit Workflow, Iterative Query Contraction) are clearly described and align with mitigating the dual crises identified in Section 3. The design is reproducible. A notable point is that InfoQA, while presented as a "proof-of-concept," is conceptually similar to some existing multi-step QA and decomposition methods (e.g., self-ask). Its novelty lies in being explicitly motivated and designed around the theoretical capacity analysis. The paper would be strengthened by a more formal argument or lemma showing how this design ensures per-step β_k < C.

**Experiments & Results (Section 5):**
*   **Benchmark Construction:** The decision to create a new synthetic dataset is justified given the need for fine-grained control over *h* and *L* and to avoid artifacts. The construction principles (systematic control, high similarity, path maximization) are sound. A major **limitation** is the lack of validation on established, real-world MHQA benchmarks (e.g., HotpotQA, 2WikiMultihop). While the controlled setting is perfect for theory validation, demonstrating that the *phenomenon* (if not the exact fitted curves) and the *advantage of InfoQA* generalize to realistic, noisy data is critical for the paper's broader impact. This should be added or strongly highlighted as a necessary future step.
*   **Theory Validation (Fig. 5):** The experimental methodology is clever and rigorous. Fitting the parametric form of β(h,L) and the bound to empirical F1 scores is an excellent way to test the theory. The results show a compelling alignment: the empirical points largely respect the theoretical envelope, and the "cliff" is observed. The fitted parameters (Table 3) provide insightful, post-hoc explanations for method performance (e.g., CoT increases effective C). This is strong evidence for the theory's descriptive power.
*   **Framework Validation & Ablations:** Table 2 convincingly shows InfoQA's superior and more robust performance compared to a strong set of single-pass baselines, especially as hop count and context length increase. The ablation study (w/o D., w/o P.) clearly demonstrates the contribution of both key components. The error analysis is thoughtful, correctly identifying that failures shift from capacity overflow to semantic drift/contraction errors.
*   **Baselines:** The selection of baselines is comprehensive. It's good that all baselines are implemented as zero-shot, single-pass for a fair comparison. The use of public Qwen models is appropriate.

**Writing & Clarity:** The paper is generally well-written. The logical flow from theory to analysis to framework to experiment is excellent. Some sections, particularly the information-theoretic derivations in Section 2 and the Appendix, are dense but necessary. The figures are effective. No major clarity issues impede understanding.

**Limitations & Broader Impact:**
*   **Limitations:** The paper acknowledges some limitations in the error analysis (semantic drift, base model capacity). However, it misses several key points: (1) The computational cost (latency, financial) of the multi-call approach vs. single-pass is not discussed, which is a significant practical consideration. (2) As noted, the reliance on a synthetic benchmark, while methodologically sound, limits claims about real-world applicability. (3) The theory assumes a closed-book setting; its implications for retrieval-augmented generation (RAG) systems, where the "context" is dynamically retrieved, are unexplored but highly relevant.
*   **Broader Impact:** The ethics statement is appropriate for the synthetic data. The broader impact is positive: providing a theoretical lens to understand LLM reasoning failures and a principled design for multi-step systems. The reproducibility statement is thorough.

### Overall Assessment

This is a strong paper with a meaningful contribution to ICLR. It successfully marries a novel theoretical analysis (the Fano-style accuracy bound and the "Accuracy Cliff") with a well-designed, principled framework (InfoQA) and rigorous experimental validation. The core theoretical insight is compelling and well-supported. The main weaknesses are the lack of validation on real-world datasets, which limits the immediate practical demonstration, and the somewhat idealized treatment of capacity *C*. These do not undermine the core theoretical contribution but indicate areas for future work and temper the claims about immediate practical utility. The paper is clearly above the ICLR acceptance bar due to its theoretical novelty, rigorous methodology, and cohesive narrative linking theory and practice.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents an information-theoretic analysis of the limitations of single-pass reasoning in LLMs for Multi-Hop Question Answering (MHQA). It derives a Fano-style accuracy upper bound, demonstrating an "Accuracy Cliff" where performance collapses when task complexity exceeds model capacity. Based on this analysis, the authors propose InfoQA, a multi-call reasoning framework that decomposes tasks, prunes reasoning traces, and iteratively contracts queries to manage information load. Experiments on a synthetic benchmark validate the theoretical bound and show InfoQA outperforming single-pass baselines.

### Strengths
1. **Novel Theoretical Contribution**: The paper formalizes the single-pass capacity bottleneck using information theory, deriving a rigorous Fano-style upper bound (Theorem 1) that predicts the Accuracy Cliff. This provides a principled explanation for LLM failures in complex reasoning tasks.
2. **Clear Problem Decomposition**: The analysis cleanly dissects MHQA challenges into Stepwise Capacity Overflow (modeled via information demand scaling) and Cross-Step Error Accumulation (formalized via chain probability), offering actionable insights for solution design.
3. **Rigorous Experimental Validation**: The authors construct a controlled synthetic benchmark with systematic variation of hop count and context length, enabling precise testing of theoretical predictions. The empirical results closely align with the derived capacity curves (Figure 5), strongly supporting the theory.
4. **Well-Designed Framework**: InfoQA thoughtfully implements capacity-aware decomposition, dependency-explicit workflow, and iterative query contraction—directly addressing the identified bottlenecks. Ablation studies (Table 2) demonstrate the importance of each component.
5. **Strong Reproducibility**: The paper provides detailed appendices with proofs, benchmark construction algorithms, and fitting procedures. The reproducibility statement includes code and data access, meeting ICLR standards.

### Weaknesses
1. **Limited Real-World Benchmark Evaluation**: While the synthetic benchmark enables controlled testing, the paper lacks validation on established MHQA datasets (e.g., HotpotQA, MuSiQue). This raises questions about generalizability to real-world complexities like diverse reasoning types and natural noise.
2. **Incomplete Comparison to Multi-Call Baselines**: The evaluation primarily compares InfoQA to single-pass methods. While Self-Ask and Self-Refine are included, they are implemented as single-pass. A comparison to true multi-call frameworks (e.g., recursive decomposition methods) would better contextualize InfoQA's contributions.
3. **Simplified Theoretical Assumptions**: The theoretical model assumes a closed-book setting and uses a parametric form for information demand (Equation 6) with fitted parameters. The uniform-distribution approximation (Equation 5) may not hold in many practical scenarios, limiting the bound's direct applicability.
4. **Computational Overhead Not Quantified**: The multi-call approach inevitably increases inference cost (more API calls/tokens), but the paper does not analyze this trade-off between performance gains and computational expense, which is important for practical deployment.
5. **Superficial Error Analysis**: The error analysis in Section 5.3 is brief and qualitative. A deeper categorization of failure modes (e.g., decomposition errors vs. contraction errors) with quantitative breakdowns would provide clearer directions for improvement.

### Novelty & Significance
The paper makes a **novel** theoretical contribution by formalizing the single-pass capacity bottleneck in LLMs using information theory—a perspective underexplored in existing MHQA literature. The Accuracy Cliff concept and its derivation provide a principled explanation for empirical observations of performance degradation. The work is **significant** as it bridges theory and practice: the theoretical bound motivates the design of multi-call systems, and InfoQA demonstrates how to operationalize these insights. The synthetic benchmark also offers a valuable controlled testbed for studying reasoning capacity. The work aligns well with ICLR's emphasis on foundational insights and rigorous experimentation.

### Suggestions for Improvement
1. **Evaluate on Standard MHQA Benchmarks**: Add experiments on at least one public dataset (e.g., HotpotQA) to demonstrate InfoQA's effectiveness in more realistic settings and facilitate comparison with state-of-the-art methods.
2. **Compare with True Multi-Call Baselines**: Implement and compare against contemporary multi-call frameworks (e.g., recursive decomposition methods, pipeline approaches) to better establish InfoQA's relative advantages.
3. **Discuss Theoretical Limitations More Explicitly**: In the main text, discuss assumptions (closed-book, uniform approximation) and how violations might affect the bound's applicability. Consider extending the analysis to open-book/retrieval-augmented settings.
4. **Analyze Computational Efficiency**: Report the number of calls, total tokens processed, and latency for InfoQA versus baselines. Discuss the accuracy-cost trade-off to inform practical use.
5. **Deepen Error Analysis**: Categorize failure cases quantitatively (e.g., percentage due to semantic drift vs. base capacity limits) and provide examples to guide future improvements.
6. **Explore Adaptive Decomposition**: As suggested in the conclusion, briefly outline how dynamic decomposition based on estimated complexity could be implemented, perhaps as a direction for future work.

**Overall Recommendation**: This is a strong paper with a compelling theoretical foundation and thorough experimental validation. It offers novel insights into LLM reasoning limitations and presents a well-designed framework that directly addresses these limitations. With the suggested improvements—particularly adding real-world benchmark results—it would be a valuable contribution to ICLR.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validate on established real-world MHQA datasets (e.g., HotpotQA, MuSiQue).** The synthetic benchmark, while controlled, does not demonstrate that the theory or method generalizes to realistic, noisy, and naturally distributed data. Without this, the paper’s practical relevance and claim of a “general principle” is unsubstantiated.
2. **Compare InfoQA against modern multi-call/iterative reasoning baselines (e.g., Self-Refine, IR-based methods, or recent decomposition methods).** The paper only compares to single-pass methods. To credibly claim the advantage of a multi-call paradigm, it must be shown to outperform or meaningfully differ from other multi-call approaches.
3. **Perform an ablation study on the “dependency-explicit workflow” (query contraction).** The paper ablates decomposition and pruning but not the core mechanism that maintains state across steps. Without this, it’s unclear whether the dependency tracking itself is a critical component or if benefits come solely from decomposition.
4. **Test the theory across a wider range of model families and scales (e.g., Llama, GPT, 70B+ models).** Experiments are limited to two sizes of Qwen3. The claim of a universal capacity bottleneck requires evidence that the “Accuracy Cliff” manifests similarly in architectures with different training data, attention mechanisms, and scaling laws.

### Deeper Analysis Needed (top 3-5 only)
1. **Provide a theoretical or empirical justification for the parametric form of β(h,L) = β0 + αLγ^(h-1).** The equation is presented as an assumption-driven model. It must be derived from information-theoretic principles or rigorously validated (e.g., by showing it fits data significantly better than simpler linear/multiplicative models). Otherwise, the fitted curves are not convincingly tied to the theory.
2. **Analyze the statistical significance and uncertainty of the fitted parameters (α, γ, β0, C).** The paper reports point estimates from grid search but no confidence intervals or sensitivity analysis. This makes it impossible to judge whether differences in fitted capacity (C) across methods are meaningful or artifacts of noise.
3. **Examine the breakdown of errors: capacity overflow vs. semantic/comprehension failures.** The error analysis mentions “semantic drift” but does not quantify how often failures are due to genuine capacity overflow (predicted by theory) versus other limitations (e.g., entity linking, textual understanding). This is needed to verify the root cause of the cliff.

### Visualizations & Case Studies
1. **Show concrete examples of prompts, reasoning traces, and failure cases for both single-pass and InfoQA.** The paper describes failures abstractly. Side-by-side case studies would visually demonstrate how single-pass outputs become garbled at capacity overflow, and how InfoQA’ pruning and contraction keep prompts manageable but can introduce semantic drift.
2. **Plot the actual output length (or estimated entropy) of the model against task difficulty.** The theory hinges on output entropy C being bounded. A plot showing that generated token length or entropy saturates as context length/hop count increases would provide direct evidence for the bottleneck, rather than indirect fitting.

### Obvious Next Steps
1. **Derive a formal bound or analysis for the multi-call setting.** The paper identifies the single-pass bottleneck but does not provide a theoretical framework for why multi-call should work or what its new limits are. A complementary bound showing how decomposition keeps per-step β below C would strengthen the foundation of InfoQA.
2. **Implement and test an adaptive decomposition strategy.** The paper uses fixed single-hop decomposition. A logical extension is to dynamically decide the granularity of sub-questions based on estimated complexity (e.g., via a lightweight estimator), which would better align with the “capacity-aware” principle.
3. **Evaluate on a broader class of reasoning tasks (e.g., mathematical, symbolic, or long-form generation).** The theory is presented as general for “single-pass reasoning,” but validation is only on entity-chaining QA. Testing on other compositional tasks would demonstrate the breadth of the accuracy cliff phenomenon.

# Final Consolidated Review
## Summary
This paper derives a Fano-style accuracy upper bound for LLM single-pass reasoning in multi-hop QA, formalizing an "Accuracy Cliff" where performance collapses when task complexity exceeds model capacity. Building on this analysis, it introduces InfoQA, a proof-of-concept multi-call framework that decomposes tasks and manages information load through capacity-aware decomposition, dependency-explicit workflow, and iterative query contraction.

## Strengths
- **Novel theoretical contribution:** The paper rigorously derives an information-theoretic bound (Theorem 1) that connects task information demand (β) and model output capacity (C), providing a principled explanation for the failure of single-pass reasoning in complex tasks. The subsequent analysis of stepwise capacity overflow and cross-step error accumulation (Section 3) clearly dissects the MHQA challenge.
- **Rigorous experimental validation:** The construction of a controlled synthetic benchmark with systematic variation of hop count and context length enables precise testing of the theoretical predictions. The empirical results closely align with the fitted capacity curves (Figure 5), strongly supporting the existence of the Accuracy Cliff.
- **Well-motivated framework design:** InfoQA’s components directly address the bottlenecks identified by the theory—capacity-aware decomposition mitigates stepwise overflow, while iterative pruning and dependency-explicit workflow counteract error accumulation. Ablation studies (Table 2) confirm the importance of these design choices.

## Weaknesses
- **Limited generalizability to real-world benchmarks:** The exclusive reliance on a synthetic dataset, while methodologically sound for controlled validation, leaves the theory and framework untested on established MHQA benchmarks (e.g., HotpotQA). This raises questions about whether the predicted phenomena and performance gains hold under natural noise and diverse reasoning types, which is important for practical relevance.
- **Simplified theoretical modeling:** The parametric form for information demand β(h,L) = β₀ + αLγ^(h-1) (Equation 6) and the uniform-distribution approximation (Equation 5) are assumptions not derived from first principles. While they enable empirical fitting, their arbitrariness may limit the bound's direct applicability to scenarios where these assumptions do not hold.

## Nice-to-Haves
- Analysis of the computational efficiency trade-offs (e.g., number of calls, total tokens) between InfoQA and single-pass baselines, as multi-call approaches inherently increase inference cost.
- Deeper error analysis quantitatively categorizing failure modes (e.g., semantic drift vs. base comprehension errors) to provide clearer directions for improvement.
- Validation across a wider range of model architectures and scales to reinforce the universality of the capacity bottleneck.

## Novel Insights
The paper provides a novel information-theoretic lens on LLM reasoning limitations, formally linking the finite output entropy of single-pass generation to a hard accuracy ceiling when task demand exceeds capacity. This insight—the Accuracy Cliff—not only explains empirical degradation in MHQA but also motivates the design of multi-call systems that explicitly manage per-step information load. The decomposition of MHQA into stepwise capacity overflow and cross-step error accumulation offers a structured framework for diagnosing and addressing reasoning failures.

## Suggestions
- Evaluate InfoQA on at least one standard MHQA dataset (e.g., HotpotQA) to demonstrate its effectiveness in realistic settings and facilitate comparison with state-of-the-art methods.
- Strengthen the empirical validation by reporting confidence intervals or sensitivity analysis for the fitted parameters (α, γ, β₀, C) to assess the robustness of the theoretical curves.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
