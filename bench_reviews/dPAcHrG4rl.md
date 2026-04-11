## Summary
This paper presents an information-theoretic analysis of the limitations of single-pass reasoning in Large Language Models (LLMs) for Multi-Hop Question Answering (MHQA). It derives a Fano-style upper bound on accuracy, formalizing an "Accuracy Cliff" where performance collapses when task information demand exceeds the model's output capacity. Based on this theory, the authors propose InfoQA, a proof-of-concept multi-call framework that decomposes tasks, prunes reasoning traces, and uses an explicit workflow to manage information load. The theory and framework are validated on a controlled synthetic benchmark.

## Strengths
- **Rigorous Theoretical Foundation:** The paper provides a formal, information-theoretic derivation of a performance bound for single-pass LLM reasoning (Theorem 1). The analysis cleanly connects classical tools (conditional Fano inequality, output entropy bound) to a modern LLM bottleneck, yielding an interpretable "Accuracy Cliff" prediction.
- **Controlled and Systematic Empirical Validation:** The authors construct a novel synthetic benchmark that allows fine-grained, independent control over key difficulty factors (hop count, context length). This design enables a clean test of the theoretical predictions, and the results show single-pass baselines following the predicted capacity curves.
- **Well-Motivated Proof-of-Concept Framework:** InfoQA is a direct, operational implementation of the principles derived from the theoretical analysis. Its components (capacity-aware decomposition, dependency-explicit workflow, iterative query contraction) are clearly explained, and ablation studies demonstrate the necessity of its core design choices.

## Weaknesses
### Major:
- **Circularity in Theoretical Validation:** The empirical validation of the Fano-style bound is not independent. The parameters of the information demand model (β₀, α, γ) and the model capacity (C) are fitted to the observed performance data (F1 scores) via grid search (Section 5.2, Eq. 11, Appendix A.5). The resulting close alignment between theory and data (Figure 5) is therefore a post-hoc fit, not a prediction from first principles. The paper does not provide an independent, task-side method to estimate β or C, weakening the claim that the bound *governs* model behavior.
- **Lack of Real-World Benchmark Validation:** The entire empirical evaluation is conducted on a synthetic, controlled benchmark. While this is suitable for testing the theory in isolation, it leaves the practical efficacy and generalizability of both the theoretical insight and the InfoQA framework unproven. There is no validation on established, real-world MHQA datasets (e.g., HotpotQA, MuSiQue), where natural noise, diverse question structures, and potential shortcuts could yield different results.
- **No Direct Manipulation of the Theorized Bottleneck (C):** The theory centers on output capacity `C = H(Y)`. A strong causal test would involve experimentally manipulating `C` (e.g., by capping the maximum allowed output tokens) and observing if the accuracy cliff shifts accordingly. The paper infers `C` from performance curves but does not perform such a manipulation, leaving the causal link between the derived bound and model behavior less firmly established.

### Minor:
- **Over-Simplifying Theoretical Assumptions:** The elegant, interpretable bound (Eq. 5, `Acc ≤ (C+1)/β`) and the demand model (Eq. 6, `β(h,L) = β₀ + αLγ^(h-1)`) rely on simplifying assumptions (e.g., uniform answer distribution, exponential hop amplification). The paper acknowledges these but does not analyze how often these assumptions hold in practice or how violations affect the bound's tightness, limiting the bound's claimed generality.
- **Narrow Model and Task Scope Evaluation:** Experiments are limited to two sizes of a single model family (Qwen3). Testing on models with different architectures, training regimes, and scales is necessary to establish the "Accuracy Cliff" as a universal LLM phenomenon, not an artifact of a specific model. Furthermore, reasoning chains are limited to 4 hops; generalization to longer, more complex chains remains unverified.
- **Superficial Error Analysis:** The error analysis is brief and generic, identifying "semantic drift" and "intrinsic model capacity" as failure modes for InfoQA. A more systematic breakdown comparing error types between single-pass (e.g., capacity overflow) and multi-call (e.g., faulty decomposition) failures would provide clearer guidance for future improvements.

## Nice-to-Haves
- **Validation on Standard MHQA Benchmarks:** Adding results on datasets like HotpotQA or 2WikiMultihopQA would significantly strengthen the paper's practical relevance.
- **Deeper Analysis of Capacity (C):** A discussion on how `C` relates to tangible LLM properties (e.g., max generation length, decoding distribution entropy) would make the theory more actionable.
- **Comparison with Other Multi-Call Frameworks:** While the paper compares to single-pass baselines, a direct comparison with other modern multi-call reasoning systems would better contextualize InfoQA's contribution within the multi-call paradigm.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength - "Well-written" or "Important Topic":** Removed as generic.
- **Weakness - "Unfair Baseline Comparison":** The critic claimed ReAct and Self-Ask were unfairly implemented as single-pass. The paper explicitly compares single-pass variants of all baselines to isolate the effect of the reasoning paradigm, which is a valid experimental design. This is not an unfair asymmetry that favors the author's method.
- **Weakness - "Lack of Novelty in Framework Components":** While the individual components of InfoQA are not novel in isolation, the framework's contribution is its principled derivation from and validation of a new theoretical analysis. The criticism is overly reductive.
- **Weakness - "Formatting Nitpicks":** Any minor stylistic comments are removed.
- **Weakness - "Reproducibility Nitpicks":** Concerns about undisclosed hyperparameters or large training logs are removed as trivial.

## Suggestions
- To address the major circularity issue, consider developing an independent method to estimate the information demand β from task statistics (e.g., context characteristics, answer space size) without fitting to performance data. Alternatively, reframe the contribution as providing a *descriptive* model that fits observed data well, rather than a predictive theoretical law.
- Run InfoQA and key baselines on 1-2 standard, real-world MHQA benchmarks to demonstrate generalizability beyond the synthetic setting.
- Include an experiment where the output capacity `C` is directly manipulated (e.g., by restricting the maximum generation length) to test if the accuracy cliff's location shifts as predicted by the theory.