=== CALIBRATION EXAMPLE 32 ===

# Harsh Critic Review
Now I have read the full paper. Let me write the review.

---

## Section-by-Section Critical Review

---

### Title & Abstract

The title is accurate. The abstract's central claim — that SFDD achieves "over 2× training speedup using only 50% of the data, while keeping the final model's inference speedup within 4% of the full-dataset baseline" — is technically defensible but somewhat misleading as framing. A 2× training speedup from using 50% of data is simply the linear scaling of compute; the genuinely interesting result is the 4% inference degradation, which deserves the headline emphasis. The abstract also presents the cosine-similarity-to-uniform metric as a principled theoretical result, when it is arguably a heuristic motivated by an idealized Gaussian proxy — a nuance that matters for calibrating reader expectations.

---

### Introduction & Motivation

The motivation is clear and the paper fills a real gap: prior work on train-based SD focuses almost exclusively on loss functions, and the data-selection angle is underexplored. The observation that acceptance rate links to the L1-norm (Leviathan et al., 2023) and the subsequent question of which training samples maximally reduce that norm is a natural and well-framed research question.

However, the introduction pre-emptively answers a question ("which target-side token distributions yield the largest per-step gains?") before presenting the theory, and some of the causal language in the framing ("tokens inducing flatter predictive distributions... are more valuable") is stated as a firm finding before it has been derived. The distinction between the **stochastic observation** that high-flatness tokens correlate with more training movement versus the **causal mechanism** that they *cause* better acceptance-rate improvement is blurred throughout.

---

### Method: Theoretical Analysis (Section 3.2 + Appendix A/B)

This is the most consequential section and has several issues that deserve scrutiny.

**1. The Gaussian assumption is structurally inappropriate.** LLM output distributions are discrete probability vectors over vocabularies of 32k–128k tokens; they are typically extremely peaked (effective support of a few tokens) with heavy tails. Modeling them as univariate Gaussians *N(μ, σ²)* discards the multinomial structure entirely. The "vocabulary size" plays no role in the Gaussian model, yet it appears critically in the cosine-similarity proxy (Appendix B). The authors acknowledge the gap ("we cannot directly compute this continuous variance") but the bridge — cosine similarity to uniform — is validated only through the asymptotic Theorem B.2, which holds *as V→∞ with L fixed*. This limit is the precise opposite of the LLM regime: LLM distributions have small effective support regardless of V, so the Gaussian variance σ² does *not* grow proportionally to vocabulary size in practice.

**2. The single-step budget model is not a model of gradient descent.** The proxy update r* = argmin D_KL(p‖r) s.t. D_KL(r‖q) ≤ θ is an analytical convenience, not an approximation of stochastic gradient descent on the KL or L1 loss. SGD on KL(p‖q) updates logits proportionally to (p(x) - q(x)), which is very different from the "optimal budget-constrained move" in distribution space. The authors claim "our insights do not depend strongly on the specific choice of budget measurement; alternative measures such as Lp norm, Jensen–Shannon divergence... would yield similar conclusions" — but this is stated without proof or empirical demonstration.

**3. The key result is numerical, not analytical.** The claim that ΔL₁ increases with σ²_p (Figure 1a) is shown only via simulation with fixed q = N(0,1) and fixed budget θ=2. There is no theorem establishing the monotonicity of ΔL₁ in σ²_p under the Gaussian model for arbitrary q and θ. For large separations (|μp - μq| large), the effect might saturate or reverse; Figure 1a shows only three separation values and they all happen to agree. This is a significant gap in the theoretical argumentation.

**4. The cosine-similarity proxy (Appendix B) has a subtle flaw.** Theorem B.2 establishes cos(p, U) ∝ σ^(1/2) — that is, cosine similarity grows as *the square root* of σ, not linearly. But ΔL₁ in Figure 1a appears to grow roughly linearly with σ². A monotone relationship is preserved (since x^(1/2) is monotone), but the authors present flatness as a "proxy for variance" without noting that the relationship is highly nonlinear. In practical low-entropy regimes (small σ), the cosine similarity will be compressed near 1.0 for most tokens, reducing discriminability where the metric is needed most (identifying the high-σ end). This could attenuate SFDD's effectiveness without being visible in the aggregate speedup numbers.

**5. The token-to-sample aggregation step (Section 4.2, Eq. 8) has no theoretical justification.** The paper derives a token-level importance criterion, then aggregates by simple averaging with no analysis of whether averaging is optimal, whether it preserves the token-level ranking, or whether it can be dominated by outlier tokens. No ablation over aggregation strategies (mean vs. median vs. top-k% percentile) is provided.

---

### Empirical Validation (Section 4.1)

The empirical validation in Figure 2 is helpful and broadly supports the claim that high-flatness tokens correlate with larger ΔL₁ values. However:

- **Sample size is extremely small:** Only 10 training samples are used. This is insufficient to establish a reliable correlation across the diversity of real data. Outlier samples or particular data domains could dominate the trend.
- **Figure 2d (entropy vs. flatness gap):** The comparison uses N ∈ {10, 20, 30, 40, 50} examples — again very small — and reports a single averaged difference g. No confidence interval or variance estimate is provided. The trend (g > 0) is consistent but its practical significance is unclear from such a small sample.
- The empirical validation is done with LLaMA3-8B-Instruct only. The claimed mechanism (high-flatness tokens drive ΔL₁ changes) has not been verified on other model families.

---

### Experiments & Results (Section 5)

**Positive aspects:** The main comparison (Table 1) is thorough, covers five diverse tasks and six baselines, and SFDD wins consistently. The ablation over retain ratios (Table 2, Table 3) is a genuine strength — the method works at low ratios (5–30%), not just at 50%. Temperature-0 results (Appendix C) are a valuable robustness check.

**Concerns:**

1. **Single model, single dataset, single framework.** All experiments use LLaMA3-8B-Instruct with EAGLE-2 on ShareGPT. There are no results on: (a) a different target model size (e.g., 70B, or a smaller 3B); (b) a different model family (Mistral, Qwen, Gemma); (c) a different training dataset (e.g., the WizardLM data used in some EAGLE ablations); (d) a different framework (Medusa, CORAL, EAGLE-3, which the paper cites). This makes it impossible to assess whether the insight generalizes or is specific to the LLaMA3/ShareGPT/EAGLE-2 combination.

2. **No statistical reporting.** Every result is a single run. Given that the differences between methods in Table 1 can be small (e.g., Entropy at 2.20× vs SFDD at 2.41× on average — only one standard deviation's worth of difference if runs have ~5% variance), the absence of error bars is a meaningful gap. For ICLR, this is increasingly expected.

3. **The 2× training speedup framing is misleading.** Using 50% of data reduces training time by roughly half *by definition*, modulo data loading and batching effects. The training speedup at 50% retain is ~2.02×, essentially the data-reduction ratio. The meaningful contribution is that inference quality is maintained (within 4%). Framing the compute savings as the headline metric overstates the methodological contribution.

4. **Cherry-picked result flagged.** The claim "on certain datasets like Alpaca, it even surpasses the full-dataset speedup (2.77× vs. 2.71×)" (Section 5.3) is a cherry-picked individual benchmark result. The paper should not foreground this without noting that the average at 70% (2.44×) is still below the No-Filter baseline (2.49×).

5. **Relationship between flatness and entropy is closer than presented.** Appendix F.2 shows that entropy produces nearly identical training dynamics curves (Figures 2 and 5 look very similar), yet entropy underperforms in Table 1. The authors explain this via Figure 2d, but the gap there (computed on only 10–50 samples) seems insufficient to explain an average speedup difference of 2.41× vs 2.20× (a ~9% gap). A rigorous analysis of *why* flatness outperforms entropy at the sample-selection level would strengthen the paper considerably. One plausible explanation — that cosine similarity weights the L2 norm while entropy uses logarithmic weighting, making them non-equivalent at extreme values — is not explored.

6. **No diversity analysis of selected samples.** If flatness is correlated with a particular data domain (e.g., open-ended creative text tends to have flatter distributions than structured math), then SFDD at 50% retain could be selecting a domain-shifted subset. This would partly explain the performance drop on GSM8K (math) at extreme retain ratios but is never analyzed. A topic/domain distribution of retained vs. filtered samples would substantially strengthen the method's characterization.

---

### Writing & Clarity

The paper is generally well-written. One genuine clarity issue: the distinction between "token-level flatness" (the theoretically motivated criterion) and "sample-level flatness" (the operational selection criterion, Eq. 8) is made in Section 4.2 but is somewhat buried. The theoretical narrative in Section 3.2 is at the token level, while the method operates at the sample level; the gap between these is acknowledged but not closely analyzed. Readers may wrongly assume the method directly implements the token-level criterion.

---

### Limitations & Broader Impact

The paper's limitation discussion is minimal. Key omissions:

- **Distribution shift from filtering:** No analysis of how SFDD changes the distribution of the training set and whether this affects domains with inherently sharp target distributions (code, arithmetic, factual QA).
- **Interaction with training duration:** SFDD is validated for 30 epochs on 50% of data. At some point, the draft model may overfit the smaller, filtered dataset. No investigation of this effect is provided.
- **Generalization to multi-token draft models:** EAGLE-2 uses a single transformer layer as draft; architectures like Medusa use multiple independent heads. The flatness criterion is applied per-position; how it extends to joint multi-token prediction is not discussed.

---

### Overall Assessment

The paper addresses a legitimate and underexplored problem — training efficiency for speculative decoding — and proposes a simple, practically effective method (SFDD). The core empirical finding (selecting high-flatness samples preserves inference quality while substantially reducing training cost) is consistent across many experimental conditions and is a real contribution. However, the theoretical underpinning is considerably weaker than it appears: the Gaussian single-step model is not a good approximation of the actual LLM-training setting, the key monotonicity result is numerical rather than analytical, and the bridge from continuous Gaussian variance to discrete cosine similarity relies on asymptotic assumptions that may fail precisely in the regime of interest. Moreover, the experiments are confined to a single model family, dataset, and framework, with no statistical reporting of variance across runs. The method's practical superiority over entropy is demonstrated but not explained mechanistically, and the aggregation from token-level scores to sample-level scores is theoretically unmotivated. For ICLR acceptance, the authors should at minimum: (1) test on at least two model families or sizes; (2) report standard deviations across multiple training runs; (3) add an ablation on sample-level aggregation strategy; (4) provide a clearer and more honest characterization of where the Gaussian theory does and does not apply. In its current form, the contribution stands as a solid empirical method paper with an oversold theoretical framing.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates data efficiency for training draft models in Speculative Decoding (SD), arguing that tokens inducing flatter (more uniform) predictive distributions in the frozen target model yield greater acceptance-rate improvements than sharply peaked tokens. Guided by a toy theoretical analysis and empirical training dynamics, the authors propose a "flatness" metric (cosine similarity to a uniform distribution) and a corresponding sample-level filtering pipeline (SFDD). Experiments on the EAGLE-2 framework demonstrate that training on 50% of the data selected via SFDD achieves over 2× training speedup while preserving inference speedup within 4% of the full-dataset baseline.

### Strengths
1. **Clear, well-motivated data-centric insight:** The paper effectively reframes SD draft training from a pure loss-design problem to a data-value problem. The observation that high-flatness tokens provide more "headroom" for L1-distance reduction is intuitively grounded in the SD acceptance formula (Eq. 1) and empirically validated through training dynamics (Fig. 2b,c, Fig. 7).
2. **Comprehensive empirical evaluation:** The experimental suite is thorough, benchmarking SFDD against six strong distribution-dispersion baselines across a wide range of retain ratios (5%–100%) and five diverse downstream tasks. Results consistently show SFDD's superiority in both average acceptance length and wall-clock speedup (Tables 1–3), with clear ablation isolating the metric's contribution from random filtering.
3. **High practicality and reproducibility:** The method is lightweight, requiring only a single offline forward pass of the target model, and is architecture-agnostic. The authors provide full training hyperparameters (Appendix E), timing protocols (Appendix D), theoretical derivations (Appendix A–B), and public code, ensuring the results can be independently replicated and deployed.

### Weaknesses
1. **Limited theoretical depth and validation scope:** The core theoretical justification relies on a single-step KL-constrained Gaussian toy model (Appendix A), which is a strong abstraction of real multi-epoch, AdamW-optimization training over discrete categorical distributions. The empirical bridge to the theory is demonstrated on only 10 randomly sampled sequences over 15 epochs, which limits the statistical strength of the saturation claims.
2. **High redundancy with entropy:** The authors acknowledge that cosine similarity to a uniform distribution is strongly correlated with entropy and shares similar training dynamics (Appendix F.2, Fig. 5). While Figure 2d shows flatness yields a larger filtering gap, the paper does not provide a mechanistic or theoretical explanation for *why* flatness consistently outperforms entropy in practice, leaving the distinction somewhat heuristic.
3. **Unclear scalability of the scoring overhead:** Although the data-selection overhead is reported as ~3.85% of total training time, the wall-clock cost and memory footprint of running a frozen 8B-parameter target model over a large dataset like ShareGPT are not fully quantified. For larger target models (e.g., 70B+) or longer-context corpora, this one-off pass could become a non-trivial bottleneck.
4. **Narrow architectural and methodological scope:** Experiments are restricted to the EAGLE-2 draft architecture with a single target model family. The paper mentions but does not evaluate SFDD on more complex SD paradigms (e.g., draft-tree verification, multi-token prediction heads, or RL-based SD training like GTO), limiting claims of generalizability across the rapidly evolving SD landscape.

### Novelty & Significance
**Novelty:** Moderate to High. The idea of prioritizing uncertain/uniform tokens has parallels in active learning and curriculum learning, but applying this specifically to SD draft training via an acceptance-centric lens (L1-norm reduction) is novel and distinct from standard KD practices. **Clarity:** High. The paper is well-structured, with clear notation, logical progression from theory to empirical validation, and intuitive visualizations. **Reproducibility:** High. Code release, detailed hyperparameters, explicit timing protocols, and mathematical derivations in appendices align well with ICLR's reproducibility standards. **Significance:** High for practitioners. As SD training pipelines grow more complex and compute-intensive, a simple, effective data-filtering strategy that halves training time with minimal inference degradation is highly valuable. However, to meet ICLR's top-tier bar, the work would benefit from deeper theoretical grounding or broader cross-framework validation to elevate it beyond an empirical engineering contribution.

### Suggestions for Improvement
1. **Deepen the flatness vs. entropy comparison:** Provide a controlled analysis isolating the cases where flatness and entropy diverge (e.g., token subsets with high entropy but low flatness, or vice versa) and report their respective gradient norms/loss trajectories to clarify *why* flatness is a superior filter.
2. **Expand architectural evaluation:** Test SFDD on at least one additional SD paradigm beyond EAGLE-2 (e.g., a multi-head draft model or a tree-based verifier) to demonstrate that the flatness insight transfers across different draft training objectives and verification mechanisms.
3. **Quantify scoring overhead transparently:** Report the exact base dataset size (number of tokens/sequences), the hardware configuration, and the wall-clock time required for the target model scoring pass. Discuss how this scales with target model size and context length, and suggest practical mitigations (e.g., chunked scoring, smaller proxy models).
4. **Strengthen the theory-practice bridge:** Add a brief discussion or supplemental experiment linking the single-step KL-budget toy model to actual AdamW optimization dynamics, perhaps by analyzing the relationship between initial flatness and Hessian trace or gradient variance over multiple epochs.
5. **Explore soft alternatives to hard filtering:** Investigate whether reweighting training samples by their flatness score (rather than binary discard) yields smoother training dynamics or better preserves long-tail capabilities, especially at very low retain ratios (<20%).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Downstream Task Accuracy:** Evaluate generation quality (e.g., exact match on GSM8K, ROUGE on CNN/DM) rather than just inference speed, as faster inference is useless if answer quality degrades.
2. **Target Model Scaling:** Validate findings on larger target models (e.g., 70B parameters) to ensure the "flatness" phenomenon is not specific to small 7B/8B models.
3. **Draft Architecture Generalization:** Test on standard independent draft models (not just EAGLE-style feature heads) to prove the insight applies to general speculative decoding, not just one framework.
4. **Metric Divergence Cases:** Provide experiments where Flatness and Entropy selection yield significantly different data subsets to justify Flatness as a novel metric rather than an entropy proxy.

### Deeper Analysis Needed (top 3-5 only)
1. **Flatness vs. Entropy Theory:** Explain theoretically why Flatness outperforms Entropy despite Appendix F.2 showing highly correlated trends and similar mathematical properties.
2. **Distributional Bias:** Investigate whether filtering low-flatness tokens removes rare but critical knowledge (e.g., specific entities or facts) from the training distribution.
3. **Saturation Timing:** Provide empirical evidence on *when* low-flatness tokens saturate during training to validate the claim that they are useless after early epochs.
4. **Overhead Scaling Analysis:** Rigorously analyze data selection overhead on larger datasets (e.g., 100B+ tokens) where a full target forward pass becomes prohibitively expensive.

### Visualizations & Case Studies
1. **Token Distribution Examples:** Visualize concrete examples of "high-flatness" vs. "low-flatness" token distributions from the target model to make the metric interpretable.
2. **Acceptance Heatmaps:** Plot acceptance rates across sequence positions for SFDD vs. Full Data to identify where the speedup gains originate.
3. **Group-wise Loss Curves:** Plot training loss separately for high-flatness and low-flatness token groups to visually confirm the saturation hypothesis.

### Obvious Next Steps
1. **Online Selection:** Explore dynamic data selection during training rather than one-off offline filtering to adapt to the draft model's changing state.
2. **Loss Function Synergy:** Investigate combining SFDD with L1-norm based loss functions to see if data selection resolves the inconsistencies mentioned in the Introduction.
3. **Quality-Speed Pareto:** Evaluate the method on a Pareto frontier of generation quality vs. inference speed rather than optimizing for speed alone.

# Final Consolidated Review
## Summary

This paper addresses training efficiency for speculative decoding draft models from a data-centric perspective. The authors propose that tokens with flatter (more uniform) predictive distributions from the target model provide greater "headroom" for acceptance-rate improvement than sharply peaked tokens. They formalize this insight through a "flatness" metric (cosine similarity to uniform distribution) and propose SFDD, a sample-level filtering method that achieves ~2× training speedup using 50% of data while preserving inference speedup within 4% of the full-dataset baseline.

## Strengths

- **Novel data-centric framing of SD training:** The paper reframes speculative decoding draft training as a data-value problem rather than purely a loss-design problem. The insight that acceptance-rate improvement correlates with target distribution flatness—grounded in the L1-norm relationship (Eq. 1)—is conceptually clean and empirically supported by training dynamics analysis (Figure 2b,c and Figure 7 gradient norms/loss curves).

- **Comprehensive empirical evaluation:** The method is benchmarked against six baselines (entropy, top-1 probability, margin, energy score, perplexity, random) across five downstream tasks and a wide range of retain ratios (5%–70%). Ablation studies at extreme retain ratios (Table 3) demonstrate robustness even under severe data reduction, and experiments on Vicuna-7B-v1.3 (Table 9) and GSM8K training split (Table 10) provide evidence of generalization beyond the primary experimental setting.

- **Practical efficiency with transparent overhead analysis:** The one-time scoring pass requires only a single forward pass through the frozen target model, with overhead of ~3.85% of total training time (2,242s vs 58,227s total training). The method is architecture-agnostic and can be integrated into existing SD pipelines like EAGLE-2 without modification to the training objective.

- **Reproducibility guarantees:** The paper provides complete training hyperparameters (Appendix E), timing protocols (Appendix D), theoretical derivations (Appendix A–B), and public code, meeting ICLR reproducibility standards.

## Weaknesses

- **Theoretical framework relies on strong idealizations:** The Gaussian assumption for analyzing token distributions (Section 3.2) abstracts away the discrete, peaked nature of actual LLM output distributions. While the authors acknowledge this gap and use cosine similarity as a proxy, the bridge from continuous variance to discrete cosine similarity (Appendix B) relies on asymptotic assumptions that may not hold in the low-entropy regime typical of LLM predictions. The claim that insights "do not depend strongly on the specific choice of budget measurement" is stated without empirical validation.

- **Limited generalization evidence across SD frameworks:** The primary experiments use EAGLE-2 with LLaMA3-8B-Instruct. While Vicuna-7B-v1.3 results (Table 9) provide some cross-model evidence, there is no evaluation on alternative SD paradigms (multi-head draft models like Medusa, tree-based verification, or RL-based training like GTO) to establish that the flatness insight transfers broadly.

- **No statistical significance reporting:** All results report single-run metrics without confidence intervals or standard deviations. Given that some comparisons involve modest gaps (e.g., 2.41× vs 2.20× average speedup over entropy), statistical significance cannot be assessed.

- **Mechanistic explanation for flatness vs. entropy gap is incomplete:** Appendix F.2 shows entropy and flatness produce similar training dynamics (Figures 2 and 5), and Figure 2d shows flatness yields larger filtering gaps on small sample sizes (N=10–50). However, the paper does not provide a principled explanation for *why* flatness consistently outperforms entropy in sample-level selection, leaving the distinction partially heuristic.

- **No analysis of distributional bias from filtering:** Selecting high-flatness samples may inadvertently shift the training distribution toward particular domains (e.g., open-ended text vs. structured reasoning). The paper does not analyze whether critical knowledge (facts, entities, reasoning patterns) is disproportionately lost in the filtered data, particularly at extreme retain ratios.

## Nice-to-Haves

- Investigation of dynamic or curriculum-style data selection that adapts as the draft model's state changes during training
- Analysis combining SFDD with L1-norm-based loss functions to examine potential synergies with prior optimization-focused approaches
- Ablation over sample-level aggregation strategies (mean vs. median vs. top-k percentile) to validate the design choice in Eq. 8

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The single-step budget model is not a model of gradient descent" (Harsh Critic):** The paper explicitly states that r* is "a theoretical proxy... not employed as a practical training target; instead, it functions solely as an analytical tool." The goal is to derive directional insights about which tokens are valuable, not to approximate SGD. This criticism misreads the paper's stated purpose.

- **"Downstream Task Accuracy evaluation missing" (Spark Finder):** This misunderstands how speculative decoding works. SD preserves the target model's output distribution by construction—acceptance rate preservation directly implies generation quality preservation. Reporting perplexity, speedup, and acceptance length is appropriate for this setting.

- **"The 2× training speedup claim is misleading" (Harsh Critic):** The 2× speedup includes data-selection overhead and reflects actual wall-clock reduction (58,227s → 28,787s). The claim is factually accurate. The inference speedup preservation (within 4%) is the genuinely meaningful contribution, which the paper appropriately emphasizes.

- **"Cosine similarity grows as σ^(1/2), not linearly" (Harsh Critic):** This is technically correct but not a meaningful weakness. A monotonic relationship is sufficient for ranking tokens by importance, which is the method's purpose. The nonlinear scaling does not invalidate the approach if it works empirically.

## Novel Insights

The paper's core insight—that the training value of tokens for SD draft models is determined by the target model's predictive uncertainty (flatness) rather than by traditional knowledge-distillation heuristics—is genuinely novel. This reframes SD training from "minimize divergence from target" to "prioritize tokens where the target is uncertain," which aligns naturally with the acceptance-rate objective. The observation that low-flatness tokens exhibit negligible gradient norms and rapid loss saturation (Figure 7) provides empirical grounding for the saturation hypothesis that motivates the filtering strategy. However, beyond these contributions, the paper does not offer deeper mechanistic insight into *why* cosine similarity to uniform outperforms entropy despite their similar mathematical properties—this remains an empirical finding in search of theoretical explanation.

## Suggestions

- Report standard deviations across at least 3 random seeds for key experiments (Table 1, Table 2) to enable statistical significance assessment
- Add a brief analysis of the data distribution shift induced by SFDD—for example, compare topic/domain distributions in retained vs. filtered samples to identify potential biases
- Include experiments on at least one alternative SD framework (e.g., Medusa or a standard independent draft model) to strengthen generalization claims
- Provide 2–3 concrete examples of high-flatness vs. low-flatness token distributions (with actual probability vectors) to make the metric interpretable for practitioners

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
