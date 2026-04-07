=== CALIBRATION EXAMPLE 6 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:**  
The title accurately reflects the contribution: a pluggable mechanism for balanced contrastive learning using a semantic disentanglement error. The abstract clearly states the problem, solution, and expected gains. However, it claims the method can be plugged into frameworks like CoST “without architectural changes,” which is slightly misleading: CoST already has separate trend/seasonal encoders and losses, so the plugin modifies the objective but doesn’t require new architecture; for other frameworks (e.g., TS2Vec), architectural changes would be needed. The abstract’s mention of “semantic skew conditions” is not defined—readers must infer from later sections.

**Introduction & Motivation:**  
The introduction effectively motivates the semantic imbalance problem in contrastive time-series learning, listing three concrete limitations (no inductive bias for decomposition, time‑domain‑only objectives, isotropic embedding collapse). The contributions are clearly stated (diagnosis, ablations, rebalancing framework). A minor weakness: terms like “spectral‑domain regularization” and “frequency‑aware contrastive learning” are introduced but not explained until later, which may confuse readers unfamiliar with CoST.

**Methods:**  
The core method is described in Sections 3.2–3.4. The Semantic Disentanglement Error (SDE) is a novel directional metric inspired by word‑embedding arithmetic. However, there are several critical ambiguities and inconsistencies:  
- The metric is alternately called “Semantic Disentanglement Error,” “Semantic Separability Error,” and “Semantic Decomposition Error” in different sections. This must be fixed for clarity.  
- Equation and definition: SDE_{a,b} = 1 − cos( v(a+b) − v(b), v(a) ). This assumes linear recoverability in embedding space, but v is a nonlinear encoder; the justification for this linearity assumption is lacking.  
- The computation of ∆ during training is not specified. Is it computed per batch, on a held‑out set, or moving average? This affects reproducibility.  
- The hyperparameters γ, γ′ are introduced without guidance on how to set them.  
- Section 3.6 references Figure 1, but the figure is garbled in the text (parser artifact). The description suggests the figure illustrates the pipeline, but its absence hinders understanding.  

Section 4.4.2 introduces an MLP to produce the composite embedding v(a+b) from concatenated component embeddings. This deviates from the original definition where v(a+b) should be the embedding of the original signal. The authors should justify why an MLP is needed and how it interacts with the SDE computation. Without this, the method appears circular: the MLP used to compute SDE is itself trained with the reweighted loss that depends on SDE.

**Experiments & Results:**  
- Section 4.2 provides a convincing synthetic analysis showing TS2Vec’s imbalance.  
- Section 4.3 reports a negative result (direct SDE regularization fails), which is honest and informative.  
- The main results in Section 4.4.4 are severely compromised because **Table 3 is unreadable** due to formatting artifacts. The table appears to contain forecasting results (MSE/MAE) for multiple datasets and horizons, but rows and columns are jumbled, making it impossible to verify the claimed improvements. This is a critical flaw because the paper’s primary empirical claim rests on this table.  
- Even if the table were clean, the experiments lack statistical significance tests, standard deviations, or multiple runs—essential for ICLR.  
- Ablation studies are insufficient: the paper mentions “systematic ablations” in the introduction, but only the direct SDE regularization is ablated. There is no ablation of the proposed components (e.g., removing the MLP, fixing ∆=0, testing different weighting schemes).  
- The SDE metric is only evaluated on synthetic data for TS2Vec; no SDE values are reported for the proposed method on real datasets, making it hard to assess whether the method actually reduces semantic imbalance in practice.  
- Baseline comparisons are appropriate (TS2Vec, TNC, CoST), but the paper does not compare to other rebalancing techniques (e.g., loss weighting based on component variances) or to fully supervised decomposition methods.

**Writing & Clarity:**  
Apart from the terminology inconsistency (SDE naming) and the garbled Table 3, the writing is generally clear. However, the method description (especially the interplay between the MLP, SDE, and weighting) is convoluted and needs a more step‑by‑step explanation. The missing figure also reduces clarity.

**Limitations & Broader Impact:**  
The paper acknowledges that direct SDE regularization fails and that multi‑view contrastive learning (CoST) already helps. However, it does not discuss limitations of the asymmetric weighting approach: e.g., sensitivity to the hyperparameters γ, γ′; potential instability if ∆ varies widely across batches; or the computational overhead of the MLP and SDE computation. Broader impact is not discussed, but the method is unlikely to have negative societal implications beyond typical ML concerns.

### Overall Assessment
The paper identifies a meaningful problem (semantic imbalance in contrastive time‑series learning) and proposes an intuitive solution (asymmetric weighting based on a disentanglement error). The idea is novel and could be valuable to the community. However, the current submission has critical flaws: inconsistent terminology, unclear methodological details (especially regarding training‑time computation of ∆ and the role of the MLP), and—most damaging—an unreadable results table that prevents evaluation of the empirical claims. Additionally, the experiments lack statistical rigor and necessary ablations. For ICLR, where empirical soundness and clarity are paramount, these issues are severe. If the authors can provide a clear, consistent manuscript with a properly formatted table, statistical tests, and thorough ablations, the contribution could be acceptable. In its present form, however, the paper does not meet the bar for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a problem in contrastive time-series representation learning where dominant semantic components (e.g., trend) suppress weaker ones (e.g., seasonality), degrading downstream task performance. To address this, the authors propose a Semantic Disentanglement Error (SDE) metric to quantify component recoverability and an adaptive loss-weighting mechanism that uses SDE to dynamically rebalance contrastive objectives. The method is designed as a plug-in module for existing frameworks like CoST.

### Strengths
1. **Clear Problem Identification**: The paper provides a well-motivated diagnosis of semantic imbalance in contrastive time-series learning, supported by controlled ablation studies on synthetic data (e.g., Table 1 showing SDE skew with varying component ratios).
2. **Innovative Diagnostic Metric**: The proposed SDE metric offers a principled, interpretable way to measure component recoverability in embedding space, drawing inspiration from vector arithmetic in word embeddings.
3. **Practical and Modular Solution**: The adaptive perceptual weighting mechanism is simple to implement, requires no architectural changes to base models like CoST, and demonstrates consistent performance gains across multiple real-world benchmarks (Table 3 shows improved MSE/MAE over baselines).
4. **Thorough Empirical Validation**: Experiments are conducted on three standard datasets (ETT, Electricity, Weather) with multiple forecasting horizons, and the paper includes ablation studies (e.g., testing direct SDE regularization) to validate design choices.

### Weaknesses
1. **Limited Novelty in Core Mechanism**: The adaptive weighting strategy is a straightforward application of dynamic loss balancing based on an observed metric (∆). Similar re-weighting ideas exist in other domains (e.g., class-imbalance learning), and its application to time-series, while sensible, is incremental.
2. **Incomplete Evaluation of Representations**: The paper primarily evaluates forecasting performance. It does not assess the learned representations on other critical downstream tasks common in time-series literature (e.g., classification, anomaly detection), limiting claims about general representation quality.
3. **Under-Specified Components and Hyperparameters**: The fusion MLP and the hyperparameters (γ, γ′) are not ablated or analyzed in depth. Their impact on performance and sensitivity are unclear, affecting reproducibility.
4. **Weakened Claim from SDE Regularization**: The paper finds that directly using SDE as a regularization term fails to improve performance (Table 2), which undermines SDE's utility as a direct optimization signal and suggests the success hinges on the specific weighting scheme.
5. **Lack of Theoretical Foundation**: While the SDE metric is intuitively defined, there is no theoretical analysis linking the SDE-based weighting to guaranteed improvements in representation balance or generalization bounds.

### Novelty & Significance
**Novelty**: The introduction of the SDE metric for diagnosing semantic imbalance in time-series embeddings is novel, as is its integration into a dynamic loss-weighting scheme for contrastive learning. However, the overall framework builds heavily on CoST, and the weighting mechanism itself is a relatively simple extension.
**Significance**: The work addresses a practical and often overlooked issue in time-series representation learning. The plug-and-play nature and consistent empirical gains make it a useful contribution for practitioners. However, the significance is somewhat tempered by the incremental nature of the core weighting idea and the lack of broader task evaluation.

### Suggestions for Improvement
1. **Expand Downstream Evaluation**: Include results on classification and anomaly detection tasks to demonstrate the general utility of the balanced representations beyond forecasting.
2. **Conduct Detailed Ablation Studies**: Systematically analyze the impact of the fusion MLP (e.g., depth, width), the sensitivity to γ/γ′, and the effect of different decomposition methods (beyond simple filtering) on performance.
3. **Provide Theoretical Insights**: Offer a theoretical justification or analysis for why the proposed weighting scheme mitigates imbalance, perhaps connecting it to gradient alignment or optimization dynamics.
4. **Compare to Alternative Balancing Strategies**: Benchmark against other simple re-balancing baselines (e.g., static loss weighting based on component variance) to better isolate the contribution of the dynamic SDE-based mechanism.
5. **Clarify Limitations and Scope**: Explicitly discuss scenarios where the method might fail (e.g., when components are not additive or when the decomposition is noisy) and its computational overhead.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Controlled synthetic experiments demonstrating the core benefit.** The paper lacks a systematic experiment on synthetic data with *known* and *variable* semantic imbalance (e.g., adjustable trend-to-seasonality SNR). The authors must show that their method's performance gain over CoST directly correlates with the degree of imbalance, proving it addresses the stated problem. Without this, the claim that it "mitigates semantic imbalance" is not causally established.
2. **Ablation of the asymmetric weighting mechanism.** The entire contribution hinges on the proposed adaptive weighting. The authors must ablate it by comparing to: a) CoST with fixed, equal weights; b) CoST with a fixed but increased weight on the seasonal loss (a simple heuristic). This is critical to prove the *adaptive* component is necessary and superior to a naive static fix.
3. **Comparison to a straightforward frequency-domain regularization baseline.** The paper dismisses direct SDE regularization but does not compare against established spectral preservation techniques (e.g., a simple Fourier-domain reconstruction loss). This gap undermines the claim that their integrated multi-view approach is uniquely effective, as the improvement might be achieved by simpler means.
4. **Evaluation on classification/anomaly detection tasks.** The paper only evaluates forecasting. To claim "improved representation learning," the authors must demonstrate benefits on other core time-series tasks (especially classification, where seasonal patterns are often critical). Without this, the contribution's generality is unproven.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the optimization dynamics of the adaptive weights.** The paper does not show how the asymmetry factor ∆ evolves during training. Does it stabilize? Does it correlate with the actual component recoverability in the final embeddings? Without tracking ∆, it's unclear if the re-weighting mechanism behaves as intended or is unstable.
2. **Causal link between reduced SDE and improved forecasting.** The authors report lower SDE and better MSE but do not establish that the former *causes* the latter. A per-dataset or per-series analysis correlating the reduction in ∆ with the improvement in forecast error is needed to validate the core hypothesis.
3. **Sensitivity and hyperparameter analysis for *γ*, *γ'*.** The method introduces new hyperparameters. A sensitivity analysis is required to show the method is robust and doesn't require extensive tuning. The paper provides no guidance on setting these values, making reproducibility and adoption difficult.

### Visualizations & Case Studies
1. **Visualization of embeddings for imbalanced synthetic examples.** For a synthetic series with a dominant trend, show t-SNE/PCA plots of the trend/seasonal embeddings from CoST vs. CoST+APW. This would visually demonstrate whether the seasonal component's representation space becomes more distinct and structured after re-weighting.
2. **Case studies of failure modes for baselines.** Select specific time series from the datasets where TS2Vec or CoST fail dramatically in forecasting (e.g., missing a key seasonal peak). Visualize the predictions and the decomposed embeddings to illustrate how the proposed method corrects this by better capturing the weak component.

### Obvious Next Steps
1. **Apply the asymmetric weighting framework to TS2Vec itself.** The problem is first diagnosed in TS2Vec, but the solution is only applied to CoST. A direct application to TS2Vec (perhaps by creating simulated "views" via filtering) is a necessary validation of the pluggable claim and would significantly strengthen the paper's contribution.
2. **Analyze the role of the fusion MLP *g_φ*.** The paper introduces an MLP to create the composite embedding but does not analyze what it learns. A simple linear probe or visualization of its weights is needed to confirm it performs meaningful fusion rather than trivial operations, which is central to the SDE computation.

# Final Consolidated Review
## Summary
This paper identifies and addresses the problem of semantic imbalance in contrastive time-series representation learning, where dominant components (e.g., trend) suppress weaker ones (e.g., seasonality). The authors propose a novel diagnostic metric, the Semantic Disentanglement Error (SDE), and integrate it into an adaptive loss-weighting mechanism that dynamically rebalances contrastive objectives within frameworks like CoST.

## Strengths
- **Clear Problem Diagnosis with Quantitative Evidence**: The paper provides a well-motivated and empirically grounded diagnosis of semantic imbalance. The controlled synthetic experiment (Table 1) convincingly demonstrates how SDE skews with varying component amplitude ratios in a standard baseline (TS2Vec).
- **Novel and Interpretable Diagnostic Tool**: The introduction of the SDE metric is a principled contribution. Inspired by vector arithmetic in word embeddings, it offers an interpretable, directional measure of component recoverability in the learned representation space.
- **Effective and Modular Solution**: The proposed asymmetric perceptual weighting (APW) mechanism is a simple, pluggable addition to existing frameworks like CoST. Empirical results on standard forecasting benchmarks (ETT, Electricity, Weather) show consistent improvements in MSE/MAE over strong baselines, validating its practical utility.

## Weaknesses
- **Inconsistent Terminology and Methodological Ambiguity**: The core metric is referred to interchangeably as "Semantic Disentanglement Error," "Semantic Separability Error," and "Semantic Decomposition Error" within the paper, harming clarity. Furthermore, key implementation details are underspecified: the procedure for computing the asymmetry factor (∆) during training (per batch, moving average?) is not described, and the role of the introduced fusion MLP in the SDE computation is not sufficiently justified, creating a circularity concern.
- **Insufficient Ablation and Analysis of Core Components**: The paper lacks systematic ablation studies for its novel components. The necessity and impact of the adaptive weighting mechanism (vs. a static heuristic) and the fusion MLP are not analyzed. Similarly, the sensitivity to the newly introduced hyperparameters (γ, γ′) is not explored, affecting reproducibility and understanding of the method's robustness.
- **Limited Evaluation of Representation Quality**: The evaluation is confined to forecasting performance. To substantiate the claim of improved general-purpose representation learning, the balanced embeddings should be evaluated on other core time-series tasks such as classification or anomaly detection.
- **Weak Empirical Rigor**: Results are presented as single runs without reporting standard deviations or statistical significance tests, which is insufficient for a conference with high standards like ICLR. While some fields use single-run evaluations, the community standard for machine learning requires more rigorous reporting.

## Nice-to-Haves
- A theoretical analysis or justification linking the SDE-based weighting to guaranteed improvements in representation balance.
- A comparison to alternative, simpler re-balancing strategies (e.g., static weighting based on component variance) to better isolate the contribution of the dynamic, SDE-driven mechanism.
- Visualization of embedding spaces or the evolution of ∆ during training to provide deeper insight into the method's behavior.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Table 3 is unreadable"**: This criticism is based on a parser artifact. In the provided paper content, Table 3 is present and clearly shows results.
- **"The method requires architectural changes for frameworks like TS2Vec"**: The paper explicitly scopes its contribution as a plugin for frameworks *like* CoST, which already has the necessary decomposition. Criticizing its applicability to TS2Vec is scope creep.
- **"Lack of comparison to frequency-domain regularization baselines"**: The paper includes an ablation showing direct SDE regularization fails (Table 2). Demanding comparison to every possible alternative regularizer is not required for the core claim.
- **"Abstract claim of 'no architectural changes' is misleading"**: The method modifies the loss function within CoST's existing dual-encoder architecture, which is reasonably described as "pluggable without architectural changes."

## Novel Insights
The paper's novel insight lies in formally defining and quantifying the semantic imbalance problem in time-series contrastive learning via the SDE metric. It then innovatively uses this diagnostic not as a static regularizer, but as a dynamic signal to adaptively reweight contrastive objectives, creating a closed-loop system that explicitly optimizes for balanced representations. This turns a measurement tool into an actionable optimization driver.

## Suggestions
- Standardize the terminology for the proposed metric (e.g., "Semantic Disentanglement Error (SDE)") throughout the paper and clarify the training-time procedure for computing ∆.
- Conduct essential ablation studies: (1) compare adaptive weighting to fixed weighting schemes, (2) analyze the contribution of the fusion MLP, and (3) perform a sensitivity analysis for γ and γ′.
- Strengthen the empirical evaluation by reporting results with standard deviations over multiple runs and extending representation evaluation to at least one other downstream task (e.g., classification).
- Provide a clearer, step-by-step explanation of the method's pipeline, ideally with a correctly rendered figure, to resolve ambiguities around the MLP and SDE computation.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
