# Why Attention Fails: The Degeneration of Transformers into MLPs in Time Series Forecasting

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Transformer-based architectures have achieved high performance in natural language processing and computer vision, yet many studies have shown that they have not demonstrated a clear advantage in time series forecasting. However, most of these studies have not thoroughly investigated the reasons behind the failure of transformers. To gain a deeper understanding of time-series transformers (TST), we designed a series of experiments and observed a degeneration phenomenon in the attention mechanism, which means that the attention mechanism does not contribute to the model's performance, causing the model before the final flatten head to become a stack of residual MLPs. We further conducted experiments on an interpretable dataset and found that the failure of representation learning is the cause of this degeneration of the attention mechanism. Finally, we provided a comprehensive analysis to explain why the current embedding methods fail to allow transformers to function in a well-structured latent space, and further analyzed the deeper underlying causes of the failure of embedding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates why time-series Transformers (TSTs) often underperform simple baselines in forecasting. The main claim is that, in many widely-used TST variants (e.g., PatchTST, iTransformer), attention contributes little to accuracy and the backbone effectively degenerates into an MLP/FFN.

### Strengths
1. A simple diagnostic offers a unified way to understand attention’s functional role in TSTs.
2. The toy state-machine dataset provides interpretable ground truth about inter-patch dependencies

### Weaknesses
1. Replacement/perturbation experiments show robustness to disabling attention, but they do not establish why attention underperforms relative to FFN, nor isolate which design choices (tokenization stride/length, residual pathways, normalization, optimizer/sharpness) make attention redundant. Without causal attribution, the paper does not show understanding beyond “attention might not matter”.
2. Disabling attention changes model capacity and computation. It remains unclear whether comparable performance arises because attention is redundant or because remaining layers dominate the capacity. When attention is removed or replaced, the model’s parameter count, compute, and capacity all change. The paper does not control for these factors, so it is impossible to attribute performance parity to attention’s irrelevance rather than architectural imbalance.
3. The “linear embedding hypothesis” is plausible but lacks formal backing. The paper makes a strong conceptual claim without proof or assumptions. 
4. Present results on shared-axis figures, making small changes look negligible. Have you tried confidence intervals or statistical significance tests, so “no difference” may be an artifact of noise? 
4. The title and narrative (“Why Attention Fails”) imply a universal conclusion, but the study covers only a narrow class of models (mostly TST-style architectures with linear embeddings). It omits settings where attention could matter (e.g., irregular sampling, exogenous covariates, long-horizon dependencies). The authors do not demonstrate any counterexamples or boundary conditions where the attention module could help.
5. The toy dataset is interesting but disconnected from the real-world findings. It visualizes “attention not focusing correctly,” but the authors did not bridge the toy results to why real models fail.
6. The writing is uneven and sometimes confusing, with grammatical errors, and vague reasoning (“we think linear embedding is bad,” “attention does not help much”).

### Questions
1. Are the capacity-matched MLP/TSMixer baselines trained under identical compute and regularization?
2. Does introducing a nonlinear embedding layer (e.g., small CNN or random Fourier features) restore the usefulness of attention?
3. Are there datasets or regimes (e.g., long-context forecasting, irregular sampling, exogenous signals) where attention demonstrably helps?
4. Would you consider using per-dataset metric deltas or significance tests to confirm that “no difference” results are statistically meaningful?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that Transformer architectures often degenerate into simple MLP when applied to time series forecasting. The attention mechanism fails to function as intended, rendering the model effectively equivalent to an MLP.

### Strengths
* Clear, Counter-intuitive Finding: It identifies and systematically demonstrates a surprising phenomenon: Transformers effectively function as MLPs in time series forecasting, making a strong challenge to the field.

* Interpretable Analysis: The design of a custom toy dataset allows for a clear inspection of attention failure, moving beyond just performance metrics on complex, black-box real-world data.

* Broad Validation: The findings are shown to hold across multiple modern Transformer architectures (PatchTST, iTransformer) and even large time series foundation models, proving the issue is widespread.

* Actionable Insight: It moves beyond just identifying a problem to pinpointing a root cause: the failure of representation learning and the inadequacy of simple linear embeddings, giving a clear direction for future work.

### Weaknesses
1. "Degeneration" is Loosely Defined: The term "degeneration into an MLP" is powerful but imprecise. A true MLP on a flattened sequence would still have weights connecting all input points. In their experiments (like Zero attention), the model still processes tokens independently through the FFN and only mixes them at the final linear layer. This is a specific type of architecture, not a standard MLP. The claim is more accurately that the model becomes token-wise independent until the very end, which is a crucial distinction.

2. The Primacy of the Embedding Layer is an Assertion, Not a Proven Cause: The paper identifies linear embedding as the root cause but provides only correlative, not causative, evidence. They show that linear embedding is ineffective, but their other experiments (Table 4) show that other embedding methods (Conv, MLP) also fail to prevent attention degeneration. This strongly suggests the problem is deeper and more fundamental than just the choice of a linear layer. The core issue might lie in the nature of time series data itself or the overall training objective, which the paper only briefly touches on in the appendix.

3. Insufficient Distinction Between "Not Necessary" and "Not Working": The experiments brilliantly show that attention is not necessary for good performance on these tasks. However, this is not the same as proving it is "not working." It's possible that in the standard model, the attention mechanism is learning meaningful patterns, but the FFN is learning a redundant, more efficient pathway. Ablating attention forces the model to rely solely on the FFN, which, it turns out, is sufficient. This is a subtle but important difference in interpretation.

4. Lack of Analysis on Learned Representations: The paper claims the latent space is poorly structured but provides limited direct analysis of the learned token embeddings. A more convincing case could be made by using probes or similarity measures to show that the representations for tokens that should be related (e.g., similar events in the time series) are not closer in the latent space than unrelated tokens.

5. The section labeled "Theoretical Analysis" (Section 5.1) is more of a conceptual discussion or a position statement than a rigorous mathematical analysis. It lacks the formal definitions, theorems, proofs, and mathematical modeling that would qualify as a theoretical foundation.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a crucial and timely diagnostic study investigating why Transformer-based models (TSTs) consistently fail to outperform simple linear models in Time-Series Forecasting (TSF), a phenomenon widely observed since the DLinear paper.

The authors' core contribution is the compelling demonstration that TSTs often degenerate into simple MLPs. Through a meticulous series of experiments, they show that the self-attention mechanism becomes functionally inert. The paper traces this failure back to a single root cause: the ineffectiveness of the standard linear embedding layer. This layer, adopted from NLP/CV, fails to project time-series data into a meaningful latent space, leaving the attention mechanism with no structured information to operate on.

### Strengths
1. This is an exceptionally strong diagnostic paper that provides immense value to the TSF community, even without proposing a new SOTA model.

- Meticulous and Rigorous Experimental Design: The paper's primary strength is its evidence. The authors don't just ablate attention; they show its progressive degeneration through multiple, clever experiments:
- Attention Replacement (Sec 3.1): Replacing attention with Zero, Identity, and Mean matrices proves the model's performance is not derived from context-aware attention.
- Module Perturbation (Sec 3.2): Demonstrates that perturbing the FFN module drastically harms performance, while perturbing the attention module does almost nothing.

Patch Length Variation (Sec 3.3): Shows that performance is stable even when the model is degenerated to a single-token MLP by increasing patch length.

2. The comparison with ViT (Sec 3.4) is perhaps the most powerful argument. By showing that ViT breaks when positional encoding is removed while PatchTST does not, the authors masterfully isolate the problem. It's not the Transformer architecture that's flawed, but its application to poorly represented time-series data.

3. This paper provides a clear, evidence-based answer to a puzzle that has occupied the TSF community. It will save many researchers from the flawed path of "building a better attention mechanism" and correctly refocuses the field on the true bottleneck: representation learning for time-series data.

### Weaknesses
1. Contribution is Diagnostic, Not Prescriptive: The most significant weakness is that the paper provides an excellent diagnosis but no validated cure. It "closes the loop" on the problem but not on the solution. While Appendix D speculates on non-linear encoders (like VQ-VAE), the paper would have been far more complete if it had demonstrated a fix.

2. Missed Opportunity on the Toy Dataset: The authors created the perfect testbed (the toy dataset) to validate a solution. A key missed opportunity was not applying a simple non-linear encoder (e.g., an MLP-based embedder) to this dataset. Showing that a better embedding re-activates attention (i.e., makes it focus on the "event" patch in Fig 7) would have transformed this from a great diagnostic paper to a groundbreaking complete paper.

3. Novelty of the Problem is Low: As the authors acknowledge, the problem that TSTs underperform linear models is well-known. The novelty lies entirely in the depth and clarity of the analysis, not in the discovery of the problem itself.

### Questions
This is an insightful paper, and I have a few questions to better understand the boundaries of its conclusions.

1. Closing the Loop on the Toy Dataset: Your toy dataset was the most compelling part of the analysis. Have you attempted to replace the linear embedding with a simple non-linear encoder (e.g., a small MLP or 1D-CNN) on this specific dataset? If so, were you able to observe the attention mechanism becoming "re-activated" and focusing on the critical "event" patches as one would expect? This seems like a crucial missing experiment to fully validate your hypothesis.

2. ViT's Success: Data Density vs. Architecture? You argue that ViT succeeds because its early FFNs perform representation learning (Sec 5.3), which TSTs fail to do even with more blocks (Fig 19, 20). Do you attribute this fundamental difference to (a) the nature of the data (i.e., image patches have a much higher "information density" than time-series patches) or (b) a subtle architectural difference in how TSTs are structured (e.g., the final linear head)?

3. Implications for Large Foundation Models: Your analysis is based on training models on individual datasets. However, large models like ViT learned rich representations because they were trained on massive, diverse datasets (like ImageNet). Do you believe your findings still apply to large-scale Time-Series Foundation Models (TSFMs) like Moirai or TimesFM? Or is it possible that "scale" (i.e., pre-training on millions of time series) can overcome the poor linear embedding and teach the model a meaningful latent space, thereby re-activating the attention mechanism?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper titled “Why Attention Fails: The Degeneration of Transformers into MLPs in Time Series Forecasting ”  investigates why Transformer architectures do  not have a clear advantage in time series forecasting and often shown to underperform simpler baselines. The main result (found empreically) is that in time-series Transformers (TSTs), the attention blocks often degenerate into MLPs, as the attention mechanism fails to function as intended. Extensive experiments, including attention replacement and perturbation, validate that the Feed-Forward Network (FFN) is the dominant player in performance, while the contribution of the attention module is minimal. The authors empirically argue that the problem stems from ineffective linear embedding methods, that hinder the attention mechanism from capturing crucial inter-token dependencies in time series data.

### Strengths
The paper observes an interesting phenomenon in that it identifies and substantiates the critical finding that Transformer blocks in Time Series Transformers (TSTs) frequently degenerate into simple Multi-Layer Perceptrons (MLPs).

Experimental validation is multifaceted and extensive and has managed to cover many aspects such as attention replacement, perturbation experiments, varying patch length among others

The findings are validated across a variety of architectures, including recent patch-wise and channel-wise Transformer models as well.

### Weaknesses
The theoretical analysis regarding linear embedding is largely an assertion or intuition that the isomorphic linear transformation of a full-rank linear layer cannot map a complex time series manifold into a "well-structured latent space". This is presented without providing any formal mathematical proofs, theorems, or guarantees (e.g., lower bounds, expressivity analysis) that rigorously demonstrate why this structural mapping is fundamentally impossible or sub-optimal in a general sense.

The primary conclusion is that the issue lies with poor representation learning due to ineffective linear embeddings.  But it begs the question, what would be a good embedding then? On this front paper explicitly does not include a thorough discussion or evaluation of alternative embedding strategies, except in a discussion section

The key experiments focus o attention replacements (zero, identity, mean) or simple perturbations (noise/attenuation). While effective at showing the lack of necessity of attention, this approach does not entirely rule out the possibility that a different, better-designed attention mechanism could be learned.

### Questions
na

### Soundness
3

### Presentation
3

### Contribution
3
