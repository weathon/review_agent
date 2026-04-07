=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "From Past to Path: Masked History Learning for Next-Item Prediction in Generative Recommendation" is evocative and accurately reflects the paper's content. The abstract claims MHL "significantly outperforms state-of-the-art generative models" on three datasets — a claim that is broadly supported by the results, though the restricted evaluation scope (three small Amazon datasets from 2014) qualifies what "state-of-the-art" means here. The abstract correctly identifies the key ingredients (entropy-guided masking, curriculum learning) and is generally well-written.

---

### Introduction & Motivation

The core motivation — that purely autoregressive training emphasizes local next-step transitions while ignoring richer long-range structural patterns in user history — is reasonable and clearly articulated. The running camera-enthusiast example is intuitive. However, two weaknesses stand out:

**Average sequence length undermines the motivation.** Table 10 reveals that the average sequence length is only 8.87 (Beauty), 8.63 (Toys), and 8.32 (Sports). The paper's central argument about "capturing long-range dependencies" and "understanding the underlying intent behind items along the purchasing path" rings hollow when the typical user has fewer than 9 interactions. The motivation is much stronger for long sequences, but those are a minority in the evaluation data.

**The pilot experiment (Appendix A) is circular.** The authors artificially truncate sequences of length >20 by removing the last 15 items, then evaluate models on predicting the *first item* of the truncated sequence. This is a contrived evaluation; models were never trained to perform this task. The dramatic improvement claimed (18.23% → 43.95% relative gain) is partly an artifact of the evaluation setup rather than a genuine demonstration of long-range modeling capability. The absolute numbers in the truncated setting (RPG R@5 = 0.6355, MHL R@5 = 0.8460) are suspiciously high compared to the full-sequence setting, which further suggests a task mismatch rather than a principled long-range dependency comparison.

---

### Related Work

The coverage is adequate and correctly positions MHL relative to BERT4Rec and S³-Rec. The key distinction — those methods use bidirectional encoders for discriminative scoring, while MHL augments a unidirectional decoder for generative recommendation — is clearly stated. No major gaps.

---

### Method

This is the most technically problematic section.

**Critical ambiguity: how is non-autoregressive reconstruction performed in a decoder-only model?** Equation 6 states that the masked history reconstruction loss reconstructs "original semantic IDs at masked positions in a non-autoregressive manner." However, the backbone is explicitly described as a "Transformer decoder" with causal (left-to-right) attention. In a decoder-only architecture with causal masking, a token at position *k* can only attend to tokens at positions ≤ *k*. Reconstructing a masked token in the *middle* of the sequence using this constraint means the model can only use left context — which is much weaker than BERT-style bidirectional reconstruction. The paper never clarifies:
- Is the causal attention mask modified for the reconstruction task?
- Or does the reconstruction use only the available left context for each masked codeword?

This is a fundamental design choice with major implications for the method's power, yet it is never addressed.

**The entropy computation is architecturally ambiguous.** Section 3.3 introduces a "transformer decoder (Dec(·))" in Equation 9 that operates on *item-level* representations (mean-pooled embeddings). However, the main backbone operates on *codeword-level* token sequences (since each item is represented as a sequence of K=32 tokens). This creates ambiguity: is there a *separate* item-level decoder used only for entropy estimation, or is it the same decoder? If separate, the additional parameters and computational overhead should be quantified. If it's the same decoder, the description is inconsistent with the item-level operation described for the main generation task (which uses token-level sequences, not item-level embeddings).

**The curriculum scheduler introduces significant complexity with vague phase transitions.** Phase II transitions via "masking ratio adaptively decreases from γ₀ as the validation performance plateaus" — effectively an ad-hoc ReduceLROnPlateau applied to the masking ratio. Appendix C clarifies this somewhat ("5 consecutive evaluations without improvement → linear decay"), but the interaction between this adaptive decay and the overall training schedule is not made fully transparent. Phase III removes masking entirely and fine-tunes only with the prediction objective, which adds a third distinct training regime. The complexity of this three-phase setup (each with its own triggers, hyperparameters, and objectives) is nontrivial to reproduce.

**The loss weight is not a free hyperparameter in practice.** Equation 7 introduces λ₁ and λ₂, yet Section 4.1 states "equal weights" are used, and Table 5 shows performance is mostly monotonically increasing with reconstruction weight up to 1.0. If λ₁=λ₂=1.0 is uniformly optimal, the presented parameterization is misleading. The ablation confirms the reconstruction signal is helpful but does not require tuning λ, which slightly weakens the motivation for Table 5.

---

### Experiments & Results

**Numerical inconsistency between Table 1 and downstream tables.** This is a significant concern. The paper reports the following for MHL on Beauty (Table 1): R@5=0.0574, N@5=0.0424. However:
- Table 2 (ablation, codebook=32): best token-level R→E→Inf gives R@5=0.0568.
- Table 3 (codebook sensitivity): token-level with codebook **16** gives exactly R@5=0.0574, N@5=0.0424 — matching Table 1 precisely.

But Section 4.1 states the implementation uses codebook size 32. Moreover, Table 4 confirms codebook 32 for Beauty in the masking ratio analysis. It appears Table 1's MHL results for Beauty actually use codebook *16*, not 32 as described. A similar issue appears for Sports and Outdoors — Table 4 confirms the best results there use codebook 16. This strongly suggests **the main table reports the best numbers cherry-picked across different codebook configurations**, without disclosing this selection process. This undermines the reliability of the presented results and must be clarified.

**Limited and dated benchmark.** All experiments use three Amazon Reviews 2014 datasets with average sequence lengths of ~8-9 items. No newer, larger-scale, or more diverse benchmarks (e.g., ML-1M, Yelp, or the Amazon Reviews 2023 datasets) are used. Given that the paper's central claim is about capturing "history" and "long-range dependencies," evaluation on datasets with longer sequences is important for validating this claim. ICLR 2026 reviewers will likely push for more comprehensive benchmarks.

**Ablation is incomplete on one key dimension.** The ablation covers masking strategy (random vs. entropy-guided) and curriculum (R, E, R→Inf, E→Inf, R→E→Inf), but does not test the direct counterpart of BERT4Rec masking — i.e., reconstruction with bidirectional context. Since the claim is that a decoder-only model can benefit from reconstruction, a natural baseline would be a model with bidirectional attention during training and causal attention during inference (as in some semi-autoregressive approaches). Without this baseline, it is unclear whether the gains come from the masking objective itself or the specific combination of unidirectional architecture + masking.

**Ablation results show inconsistent patterns for Sports and Outdoors.** In Table 2, for token-level masking on Sports: R→Inf (.0332/.0225) actually outperforms R→E→Inf (.0191/.0123) by a large margin. The curriculum with entropy guidance performs *worse* than stopping at random masking. This is acknowledged only briefly ("the complete R→E→Inf curriculum learning framework achieves optimal performance") — but that statement is not true for Sports at token-level. The paper should discuss this inconsistency rather than glossing over it.

**No statistical significance testing.** No error bars, variance estimates, or significance tests are reported. Given that many reported improvements are in the 3rd–4th decimal place and some ablation differences are marginal, this is a meaningful omission.

**Case study contains a factual error (Table 7).** The paper states MHL's predictions "aligns logically with the user's sustained interest in firearm-related products." However, the listed history (Footwear Adhesive → Running Waist Pack → Cardio Trampoline → Heavyweight T-Shirt → BMX Pads) and the prediction (Youth Multi-Sport Helmet) contain nothing related to firearms. This description is plainly incorrect and undermines confidence in the qualitative analysis.

**Generalization experiment (Table 6) is limited.** The generalization to text sequences is tested only on Beauty for all three metrics per dataset, and the absolute improvements are modest. Only three datasets are shown, and the improvement is numerically smaller in magnitude than the semantic ID case. More thorough analysis would strengthen this claim.

---

### Writing & Clarity

The main source of potential confusion is the architectural ambiguity in Section 3.3 (entropy computation), which genuinely impedes understanding of the method. The three-phase curriculum is described with enough detail in Appendix C to be reproducible, though the description in the main text is vague about phase transition conditions.

---

### Limitations & Broader Impact

The paper does not include a limitations section. Key limitations that should be acknowledged:

1. **Computational overhead of entropy-guided masking.** Computing token entropies requires a forward pass through the model for every training example at each epoch. For a training set of ~150K interactions with sequences tokenized into 32 codewords, this overhead is non-trivial and is never quantified.

2. **Short sequence setting.** The average sequence length of ~8 interactions means most users have very little "history" to mask and reconstruct. The method's advantage is most pronounced on long sequences (as shown in Table 9), yet long sequences are a minority in the evaluation data.

3. **Train-test discrepancy.** Despite Phase III fine-tuning being designed to address this, the paper does not ablate whether Phase III is actually necessary, or analyze how much of the gap it closes.

---

### Overall Assessment

MHL is a sensible and incrementally novel idea: augmenting autoregressive generative recommendation with an auxiliary masked history reconstruction objective, enhanced by entropy-guided masking selection and a three-phase curriculum schedule. The empirical results on three Amazon datasets are positive and the ablations, while incomplete, provide reasonable support for each component. However, the paper has several problems that collectively weaken the contribution at ICLR's standard: **(1)** a critical, unresolved ambiguity about how non-autoregressive reconstruction operates within a causal decoder; **(2)** what appears to be undisclosed cherry-picking of codebook configurations across tables, creating a misleading picture of the "default" model; **(3)** the entire empirical evaluation rests on three small (~150K interaction), short-sequence (~8–9 items) Amazon datasets from 2014, directly undermining the paper's central claim about capturing long-range user history; and **(4)** unexplained inconsistencies in ablation results (Sports and Outdoors, token-level). The core idea is more of an engineering contribution — BERT-style masking applied inside a generative recommender — than a fundamental scientific advance, and the paper needs to more rigorously justify why this specific combination of components (rather than simpler alternatives) is necessary. For ICLR, the absence of theoretical analysis and the limited experimental scope are likely to be disqualifying concerns without substantial revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **Masked History Learning (MHL)**, a training framework for generative recommendation that augments standard autoregressive objectives with a masked history reconstruction task. It introduces two key strategies to optimize the masking process: an **entropy-guided masking** policy to target informative historical items and a **curriculum learning** scheduler to transition from reconstruction to future prediction. Experiments on three Amazon datasets demonstrate that MHL significantly outperforms state-of-the-art generative recommendation baselines like TIGER and RPG.

### Strengths
1.  **Strong Empirical Performance:** MHL consistently achieves State-of-the-Art (SOTA) results across all metrics (Recall@K, NDCG@K) and datasets (Beauty, Toys, Sports). For instance, it achieves a 27.1% improvement in NDCG@5 over the strong TIGER baseline in the Sports and Outdoors category (Table 1, Section 4.2).
2.  **Comprehensive Ablation Studies:** The paper provides rigorous ablation studies validating the necessity of each component. Table 2 explicitly compares masking strategies (Random vs. Entropy) and curriculum configurations (R→Inf vs. R→E→Inf), confirming that the full proposed framework is necessary for optimal performance.
3.  **Generalizability:** The authors validate the framework beyond discrete semantic IDs by applying it to unstructured text sequences (Table 6). MHL outperforms the RPG baseline on raw text tokens, suggesting the learning paradigm captures fundamental sequential dynamics rather than relying solely on specific ID tokenization properties.
4.  **Clear Motivation:** The paper effectively argues that current generative recommenders prioritize local transitions ("what comes next") at the expense of global intent understanding ("why this path matters"). The pilot experiment in Appendix A supporting long-term dependency capture strengthens this claim.

### Weaknesses
1.  **Architectural Ambiguity (Decoder-only vs. Masked Context):** The paper states the model is a "unidirectional, decoder-only" model (Section 1, Section 3.2), yet the masked history reconstruction loss implies the model attempts to reconstruct masked items using context. Standard causal decoders cannot see "future" tokens relative to a mask position. If the mask allows only left-context (causal MLM), this is methodologically distinct from BERT-style bidirectional reconstruction, potentially limiting the effectiveness compared to encoder-based approaches. The exact mechanism for handling causal masking during the reconstruction loss (Eq. 6) is unclear.
2.  **Computational Overhead of Entropy Guiding:** Section 3.3 describes computing predictive entropy for codebooks (Eq. 10). This requires forward passes over the vocabulary to compute probabilities. Given the codebook size (32) and sequence length, this may incur significant computational overhead during training compared to random masking. The paper does not report training time or FLOPs comparisons to baselines.
3.  **Curriculum Scheduler Complexity:** The curriculum involves three phases (Random → Entropy → Fine-tuning) with decay mechanisms on masking ratios. While Ablation shows it works, the added complexity of scheduling hyperparameters (e.g., decay factors in Section 4 Implementation) compared to a simple joint loss training objective introduces more implementation variance without a guaranteed improvement over a static high-ratio masking strategy.
4.  **Loss Weight Sensitivity:** Table 5 shows that performance is sensitive to the reconstruction loss weight ($\lambda_2$). While 1.0 is optimal, the drop at 0.2 suggests the two objectives (prediction vs. reconstruction) might be conflicting or the signal strength varies significantly, requiring careful tuning per dataset which reduces robustness for deployment.

### Novelty & Significance
**Novelty:** The integration of masked language modeling objectives specifically into the *generative* (decoder-based) recommendation paradigm, particularly with entropy-guided masking, is novel. While MLM is standard in discriminative recommenders (e.g., BERT4Rec), applying it with causal constraints to enhance generative sequence production is a distinct contribution to the ICLR audience interested in generative foundations.

**Significance:** The findings suggest that understanding user history as a reconstruction task improves path prediction, moving beyond simple co-occurrence patterns. If successfully generalized, this could improve the robustness of generative recommendation systems in capturing long-term user intent, which is a significant challenge in current sequential recommendation literature.

### Suggestions for Improvement
1.  **Clarify Causal Attention Mechanism:** Explicitly describe the attention mask used during the training step where history reconstruction occurs. Confirm whether the reconstruction task uses bidirectional attention (contradicting "decoder-only") or strictly left-context. If left-context, clarify how it differs from next-item prediction and why it still improves global understanding.
2.  **Report Efficiency:** Include a analysis of training time and computational cost overhead introduced by the entropy-guided masking and curriculum learning compared to the standard TIGER/RPG baselines.
3.  **Hyperparameter Sensitivity:** Expand the discussion on the loss weight $\lambda_1/\lambda_2$ (Table 5). Suggest whether adaptive weighting strategies (e.g., uncertainty-weighted loss) could stabilize the training without manual tuning.
4.  **Long-Sequence Analysis:** While Appendix A touches on sequence length, provide a more detailed breakdown of performance on very long sequences (e.g., $T > 50$) specifically to validate the claim of "learning robust and generalizable representations" mentioned in the Introduction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Modern and Diverse Benchmarks:** Evaluate on newer datasets (e.g., Amazon 2018, Yelp, or TikTok) because relying solely on Amazon 2014 limits the generalizability of claims to current interaction patterns.
2. **Computational Efficiency Analysis:** Report training time and inference latency compared to baselines, as the entropy calculation and curriculum scheduling introduce significant overhead that may negate performance gains.
3. **Statistical Significance Tests:** Include standard deviations over multiple runs and significance testing (e.g., t-tests) because the marginal improvements over strong baselines like RPG could be due to random variance.
4. **Bidirectional Encoder Baseline:** Compare against a bidirectional encoder-generator pipeline to isolate whether gains come from "history understanding" or simply accessing bidirectional context during training.
5. **Cold-Start User Evaluation:** Measure performance on users with sparse interaction histories to verify if the proposed "history understanding" genuinely helps where data is scarce.

### Deeper Analysis Needed (top 3-5 only)
1. **Attention Mask Mechanism:** Explicitly define the attention mask during history reconstruction, as using bidirectional context in a "decoder-only" model creates a train-test mismatch that undermines the architectural claim.
2. **Curriculum Phase III Justification:** Explain why removing the reconstruction task in the final fine-tuning phase improves performance, as this contradicts the core claim that history understanding is crucial for inference.
3. **Entropy Cost-Benefit Trade-off:** Analyze whether the marginal performance gain of entropy-guided masking over random masking (seen in Table 2) justifies the $O(V)$ computational cost of entropy estimation.
4. **Loss Interference Analysis:** Investigate gradient conflicts between the reconstruction and prediction losses to ensure the auxiliary task does not degrade the primary generative objective.
5. **Masking Ratio Sensitivity:** Explain why the optimal masking ratio varies significantly across datasets (0.10 vs. 0.20) rather than demonstrating a robust, dataset-agnostic hyperparameter.

### Visualizations & Case Studies
1. **Attention Heatmaps:** Visualize attention weights during prediction to confirm the model attends to long-term history rather than just recent items, validating the "path understanding" claim.
2. **Entropy Distribution Plot:** Show the distribution of entropy scores to verify that high-entropy items correspond to meaningful intent shifts rather than simply rare or noisy interactions.
3. **Training Loss Curves:** Plot both loss objectives over epochs to demonstrate convergent behavior and lack of catastrophic forgetting during curriculum transitions.
4. **Embedding Space Visualization:** Use t-SNE to compare item representations learned by MHL vs. RPG to prove that reconstruction yields more semantically clustered embeddings.
5. **Failure Case Examples:** Provide specific instances where MHL fails to reconstruct history or predicts incorrectly despite the auxiliary task to expose method limitations.

### Obvious Next Steps
1. **Simplify Masking Strategy:** If random masking achieves comparable results in ablation studies, remove entropy guidance to improve reproducibility and reduce computational complexity.
2. **Inference Latency Measurement:** Report actual milliseconds-per-request latency to assess real-world deployability, as generative models are often too slow for production without optimization.
3. **Cross-Domain Generalization:** Test on non-e-commerce sequential data (e.g., news reading or video watching) to validate the claim of learning general "user paths" rather than product co-occurrence.
4. **Hyperparameter Robustness:** Conduct a broader sensitivity analysis on loss weights ($\lambda_1, \lambda_2$) to ensure results are not tuned to a fragile balance specific to these datasets.
5. **Reproducibility Details:** Provide exact details on the entropy calculation implementation (e.g., temperature scaling, sampling) in the main text rather than Appendix C to ensure the method can be replicated.

# Final Consolidated Review
## Summary

This paper proposes **Masked History Learning (MHL)**, a training framework for generative recommendation that augments standard autoregressive objectives with an auxiliary task of reconstructing masked historical items. The framework includes entropy-guided masking (targeting high-entropy positions) and a three-phase curriculum scheduler transitioning from random masking to entropy-guided masking to pure prediction. Experiments on three Amazon Reviews 2014 datasets demonstrate consistent improvements over generative recommendation baselines.

## Strengths

- **Strong empirical results with substantial margins:** MHL achieves consistent improvements across three datasets, with gains up to 27.1% NDCG@5 over TIGER on Sports and Outdoors (Table 1). The ablation study (Table 2) provides reasonable evidence that both entropy-guided masking and the curriculum schedule contribute to performance.

- **Well-motivated problem formulation:** The paper correctly identifies that purely autoregressive training focuses on local transitions, potentially missing deeper user intent. The motivating camera-enthusiast example (Figure 1) intuitively illustrates the limitation and how history reconstruction could help.

- **Generalization to text sequences:** Table 6 demonstrates that the approach works on raw text token sequences, not just discrete semantic IDs, suggesting the learning principle captures fundamental sequential dynamics rather than exploiting specific tokenization properties.

## Weaknesses

- **Critical architectural ambiguity: how does non-autoregressive reconstruction work in a decoder-only model?** Equation 6 states that masked history reconstruction is performed "in a non-autoregressive manner," and Section 2 claims MHL uses a "unidirectional, decoder-only" model. In a standard causal decoder, position *k* can only attend to positions ≤ *k*. Reconstructing a masked item at position *k* using only left context is fundamentally different from BERT-style bidirectional reconstruction. The paper never clarifies: (1) whether the attention mask is modified during reconstruction, (2) whether reconstruction uses only available left context, or (3) whether there's a separate encoder. This is a central methodological question that directly affects the claims about "history understanding."

- **Numerical inconsistency suggests undisclosed hyperparameter selection:** Table 1 reports MHL Beauty results as R@5=0.0574, N@5=0.0424. Table 3 shows these exact numbers occur at codebook size 16 for token-level masking. However, Section 4.1 states "We encode item metadata... into sequences of 32 semantic tokens" and Table 4's caption confirms codebook size 32 for Beauty. This discrepancy strongly suggests the main results table may report the best configuration across codebook sizes without transparent disclosure, undermining reproducibility claims.

- **Inconsistent ablation results unexplained:** In Table 2, for Sports and Outdoors with token-level masking, R→Inf (R@5=0.0332) substantially outperforms R→E→Inf (R@5=0.0191). The paper claims "the complete R→E→Inf curriculum learning framework achieves optimal performance" but this is false for this configuration. The inconsistency deserves discussion rather than being glossed over.

- **Limited benchmark scope undermines long-range dependency claims:** All experiments use Amazon Reviews 2014 datasets with average sequence lengths of 8-9 items. The paper's central claim is about capturing "long-range dependencies" and "understanding the user's path," yet typical users have fewer than 10 interactions. While Appendix A analyzes longer sequences (>20 items), these represent a small fraction of the evaluation data. The motivation would be substantially strengthened by evaluation on datasets with genuinely long interaction histories.

- **No statistical significance testing:** Improvements over baselines are often in the 3rd-4th decimal place (e.g., 0.0574 vs 0.0500 for Beauty R@5). Without variance estimates or significance tests, it's unclear whether these improvements are robust or within noise margins.

- **Factual error in case study:** Table 7 claims MHL's prediction "aligns logically with the user's sustained interest in firearm-related products." The listed history (Footwear Adhesive → Running Waist Pack → Cardio Trampoline → Heavyweight T-Shirt → BMX Pads) and prediction (Youth Multi-Sport Helmet) contain no firearms references. This error undermines confidence in the qualitative analysis.

- **Entropy computation architecture unclear:** Section 3.3 introduces a "transformer decoder Dec(·)" for entropy computation that operates on item-level mean-pooled embeddings. The main model operates on codeword-level token sequences. It's unclear whether this is a separate decoder (adding parameters and computation) or an architectural inconsistency in the description.

## Nice-to-Haves

- **Computational efficiency analysis:** Entropy-guided masking requires forward passes to compute predictive entropy for masking decisions. The paper provides no training time or FLOPs comparison to baselines, which is important for practical adoption.

- **Broader dataset evaluation:** Modern datasets (Amazon 2018+, MovieLens, Yelp) would strengthen generalization claims and better reflect contemporary user behavior patterns.

- **Analysis of train-test discrepancy mitigation:** Phase III removes masking for fine-tuning to address train-test mismatch, but no ablation shows whether this phase is actually necessary or how much gap it closes.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Claim that pilot experiment is "circular":** The harsh critic calls Appendix A's truncated-sequence evaluation circular because models weren't trained for this task. This overstates the issue—the evaluation intentionally tests whether models capture early-sequence intent when later context is removed. The dramatic improvement percentages are valid evidence of MHL's long-range modeling, though the absolute numbers being higher in truncated setting (easier task with shorter sequences) is unsurprising.

- **Demand for bidirectional encoder baseline:** One reviewer requests comparing against models that use bidirectional attention during training and causal attention during inference. While interesting, this would be a different architecture family. The paper's contribution is augmenting decoder-only generative recommenders, which is a valid scope. This is a nice-to-have, not a weakness.

- **Loss weight being "misleadingly parameterized":** The claim that λ₁=λ₂=1.0 being optimal makes the parameterization misleading is overblown. Table 5 shows sensitivity analysis is useful for understanding the contribution of each objective, and equal weighting is a reasonable default.

- **"Short sequences undermine motivation entirely":** While average sequences are short, Appendix A explicitly analyzes longer sequences and shows MHL's advantages are more pronounced there. The method still improves on shorter sequences (as shown in main results), so the concern is mitigated rather than fatal.

## Novel Insights

The paper's core insight—that forcing generative recommenders to reconstruct masked history compels learning of "why this path matters" rather than just "what comes next"—is compelling. The entropy-guided masking strategy specifically targets positions where the model is uncertain, which aligns with the intuition that these positions encode meaningful user intent transitions. However, the architectural ambiguity about causal attention during reconstruction raises a fundamental question: if masked tokens can only use left context, how is this fundamentally different from the next-item prediction that the paper argues is insufficient? If bidirectional context is somehow enabled during reconstruction, this should be explicitly stated as it represents a significant departure from "decoder-only" claims.

## Suggestions

1. **Explicitly describe the attention mechanism during reconstruction:** Add a clear statement (with equations or pseudocode) showing whether causal masking is modified during the masked history reconstruction loss computation. If bidirectional context is used, acknowledge the train-test mismatch and justify Phase III fine-tuning as mitigation.

2. **Disclose hyperparameter selection transparently:** Either use consistent codebook sizes across all tables and the main results, or clearly state that different configurations were selected per dataset and report the selection process.

3. **Add statistical significance tests:** Report mean and standard deviation across multiple runs for key comparisons, especially when improvements are marginal.

4. **Correct the case study description:** The "firearm-related products" claim is factually incorrect and should be fixed.

5. **Clarify the entropy decoder architecture:** Explicitly state whether Dec(·) is a separate model component and quantify any additional parameters/computation it introduces.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 2.0, 2.0]
Average score: 3.0
Binary outcome: Reject
