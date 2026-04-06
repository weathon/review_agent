=== CALIBRATION EXAMPLE 13 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly conveys the core idea—a pluggable mechanism for balancing contrastive time-series representations via a Semantic Disentanglement Error (SDE). The abstract succinctly states the problem (dominant components suppress minor ones), the proposed solution (SDE + adaptive weighting), and the outcome (consistent gains in forecasting/robustness). All claims are specific and appear supported by the experiments described later. The typo "PLUG- GABLE" is a minor formatting artifact and does not affect understanding.

### Introduction & Motivation
The introduction effectively motivates the problem: current contrastive methods fail to balance semantic components like trend and seasonality, degrading downstream utility. The three identified limitations (no decomposition bias, time-domain-only objectives, isotropic embedding collapse) are well-grounded in prior work and set up a clear need for intervention. The contributions are listed (diagnosis, ablations, rebalancing framework) and align with the paper’s content. One minor note: the numbering of contributions is slightly inconsistent in formatting (first is a dash, others numbered), but this does not obscure meaning.

### Related Work
The section adequately covers relevant areas: self-supervised learning in NLP/vision, contrastive time-series methods (TS2Vec, TNC), frequency-aware models (Autoformer, FEDformer, CoST), and embedding collapse literature. It correctly positions the work as bridging the gap between asymmetry-aware weighting (from vision) and time-series representation learning. The discussion of CoST is particularly relevant as the proposed method builds directly upon it.

### Methods
The method is conceptually clear and presented as pluggable into frameworks like CoST. However, several points require clarification or correction:
1. **SDE Definition (Sec 3.2)**: The equation `SDE_{a,b} = 1 - cos( v(a+b) - v(b), v(a) )` is sensible, but the notation is ambiguous—does `v(a+b)` use the same encoder as `v(a)` and `v(b)`? In Section 4.4.2, an MLP is introduced to compute `v(a+b)` from concatenated embeddings, which differs from the initial definition. This inconsistency should be resolved.
2. **Asymmetric Weighting (Sec 3.4)**: The weighting scheme `L = (1 + γ·Δ) L_season + (1 + γ′·(−Δ)) L_trend` is intuitive. However, it is unclear how Δ is computed during training—is it estimated on a batch, or using a running average? The choice of hyperparameters γ, γ′ is not discussed (e.g., are they tuned or fixed?). This affects reproducibility.
3. **Composite Embedding MLP (Sec 4.4.2)**: The introduction of an MLP to fuse trend and seasonal embeddings is a substantive addition not mentioned in the earlier method description. This should be integrated into Section 3 for consistency. The role of this MLP (linear alignment vs. nonlinear interaction) is left as future work, but its inclusion here means the method is not entirely “without architectural changes” as claimed in the abstract—it adds a new module.
4. **Figure Reference**: Figure 1 is referenced but not visible in the text (only a placeholder appears). While this is likely a parser artifact, the description in Sec 3.6 is sufficient to understand the framework.

### Experiments & Results
The experimental design is solid: standard benchmarks (ETT, Electricity, Weather), fair baselines (TS2Vec, TNC, CoST), and multiple horizons. However, several issues affect interpretability:
1. **SDE Analysis (Sec 4.2)**: Table 1 convincingly shows TS2Vec’s semantic imbalance across amplitude ratios. This validates the core problem. The synthetic data construction is appropriate.
2. **Direct SDE Regularization (Sec 4.3)**: Table 2 shows that adding SDE as a regularizer to TS2Vec does not improve (and sometimes harms) forecasting performance. This ablation is valuable—it shows SDE alone is not an effective regularizer, justifying the need for asymmetric weighting.
3. **Main Results (Sec 4.4.4)**: Table 3 is critically important but **poorly formatted** in the provided text. Numbers are interleaved with text, making it impossible to fully assess the claims. From the narrative, CoST+APW outperforms baselines, but the exact numbers and statistical significance are unclear. The authors must ensure the table is clearly presented in the final version. Additionally, SDE values are mentioned as being reported in Table 3, but they are not visible in the extracted text.
4. **Ablation Missing**: The paper introduces three components (SDE, multi-view contrastive learning, asymmetric weighting), but there is no ablation study isolating the contribution of asymmetric weighting versus the multi-view baseline (CoST). The claim that “asymmetry-aware perceptual weighting further ensures” improvement needs direct comparison—e.g., CoST vs. CoST+APW with all else fixed.
5. **Implementation Details**: Training details (lr=1e-3, 100 epochs) are given, but hyperparameters for γ, γ′ and the MLP architecture are omitted. This hinders reproducibility.

### Writing & Clarity
The paper is generally well-written and logically structured. However, terminology is inconsistent: the error metric is called “Semantic Separability Error” in Sec 3.2, “Semantic Decomposition Error” in Sec 4.3, and “Semantic Disentanglement Error” in the title. This should be unified. Section 4.4.1 repeats the CoST framework description, which is redundant given earlier coverage. The flow from problem to solution is clear, but the method-experiment mismatch (MLP introduced only in experiments) is confusing.

### Limitations & Broader Impact
Limitations are briefly discussed in the conclusion: the need for frequency-domain encoders (CoST) and the unknown role of the fusion MLP. The authors note plans to test if low-pass filtering suffices and to analyze the MLP. These are appropriate future directions. A broader impact statement is absent; while the work is methodological, ICLR typically expects a short statement (even if societal impact is neutral). The authors should add one.

## Overall Assessment
This paper identifies a genuine issue in contrastive time-series learning—semantic imbalance—and proposes a novel diagnostic (SDE) and a pluggable reweighting mechanism to address it. The core idea is sound, and the preliminary analysis convincingly demonstrates the problem. However, the experimental evaluation is marred by a poorly formatted results table and missing ablations (e.g., contribution of asymmetric weighting alone). Additionally, the method description is inconsistent regarding the use of an MLP for composite embeddings, and key implementation details are omitted. With these issues addressed (clear presentation of results, ablation studies, method clarification), the contribution would meet ICLR’s standards for novelty and empirical validation. As it stands, the paper is promising but requires revision to solidify its claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a semantic imbalance problem in contrastive time-series representation learning, where dominant components (e.g., trend) suppress weaker ones (e.g., seasonality). The authors propose a pluggable mechanism based on a novel Semantic Disentanglement Error (SDE) metric, which quantifies component recoverability, and an adaptive asymmetric weighting strategy that dynamically re-balances contrastive losses. The method is integrated into the CoST framework, improving forecasting accuracy and representation robustness on standard benchmarks.

### Strengths
1. **Clear Problem Formulation**: The paper convincingly diagnoses a concrete limitation in existing methods (e.g., TS2Vec, CoST) via controlled synthetic experiments (Table 1), showing that SDE asymmetry correlates with component amplitude ratios.
2. **Practical, Low-Overhead Solution**: The proposed asymmetric weighting mechanism is simple and pluggable, requiring no architectural changes to base frameworks like CoST. This enhances its utility and ease of adoption.
3. **Comprehensive Empirical Validation**: Experiments on multiple real-world benchmarks (ETT, Electricity, Weather) demonstrate consistent improvements in forecasting MSE/MAE over strong baselines (TS2Vec, TNC, CoST). The inclusion of SDE as an evaluation metric provides direct evidence of improved semantic balance.

### Weaknesses
1. **Methodological Clarity and Consistency**: The method description is fragmented. SDE is initially defined for a general encoder (Sec 3.2), but the operational procedure (Sec 4.4) introduces an MLP to generate the composite embedding \(v(a+b)\) without justifying its necessity or detailing its architecture. The relationship between the "SDE regularization" attempt (Sec 4.3) and the final asymmetric weighting is unclear.
2. **Incomplete Ablation Study**: While the paper claims three complementary strategies (SDE regularization, multi-view learning, asymmetric weighting), results only show the full combination (CoST+APW) versus baselines. The individual contribution of asymmetric weighting on top of CoST is not isolated, making it difficult to attribute gains precisely.
3. **Limited Theoretical Insight**: The paper is empirically driven but lacks a theoretical analysis of why asymmetric weighting works or how SDE relates to the gradient dynamics of contrastive learning. The failure of direct SDE regularization is noted but not deeply analyzed.

### Novelty & Significance
**Novelty**: The core novelty lies in defining SDE as a *directional* measure of semantic recoverability for time-series components and using it to *dynamically* re-weight contrastive objectives. This addresses a specific, underexplored form of representation collapse in time-series contrastive learning.
**Significance**: The work tackles a practical issue affecting representation utility in downstream tasks. The pluggable nature and consistent gains make it a meaningful incremental contribution for researchers and practitioners using frameworks like CoST. It meets ICLR's emphasis on clear empirical advances and reusable techniques.

### Suggestions for Improvement
1. **Streamline Method Description**: Consolidate Sections 3 and 4.4 into a single, coherent algorithm. Explicitly state whether the MLP is essential or if a simpler alternative (e.g., linear projection) works, and clarify how SDE is computed during training (e.g., using a running average).
2. **Perform Targeted Ablations**: Add experiments that isolate the effect of asymmetric weighting by comparing: (a) CoST, (b) CoST with fixed asymmetric weights (e.g., always up-weight seasonality), and (c) CoST with dynamic weighting (APW). This would clarify the value of the adaptive mechanism.
3. **Deepen Analysis**: Provide a qualitative or quantitative analysis (e.g., visualization of embedding spaces or gradient norms) showing how asymmetric weighting alters the learning dynamics to preserve weak components. Discuss the sensitivity of hyperparameters \(\gamma, \gamma'\).
4. **Clarify Limitations and Scope**: Explicitly discuss scenarios where the method might not help (e.g., when no clear trend/seasonality decomposition exists) and computational overhead. The conclusion mentions future work on low-pass filtering; briefly situating that as a current limitation would strengthen the paper.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study on the MLP fusion component.** The paper adds an MLP to create the composite embedding *v(a+b)*. A controlled ablation (e.g., replacing the MLP with a simple linear layer or averaging) is required to verify that performance gains are due to the asymmetric weighting mechanism and not simply from adding more parameters/nonlinearity.
2. **Statistical significance testing for forecasting results.** The forecasting tables (e.g., Table 3) show marginal improvements. Without significance tests (e.g., paired t-tests across multiple runs/seeds), it is impossible to claim "consistent gains," which is a core contribution. This is a critical standard for ICLR.
3. **Comparison with modern supervised forecasting baselines.** The paper only compares against self-supervised representation learners (TS2Vec, CoST). To justify the value of the learned representations, a comparison with strong end-to-end supervised models (e.g., FEDformer, Autoformer, PatchTST) on the same forecasting task is necessary.
4. **Experiments on datasets with varying semantic skew.** The core problem is "semantic imbalance," but experiments are only on standard benchmarks. The paper should include a dedicated experiment on synthetic or real data where the trend/seasonality amplitude ratio is systematically varied to conclusively show the method's robustness to skew.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of SDE on real dataset components.** SDE is analyzed only on synthetic data. To trust it as a diagnostic, the authors must compute SDE on the *actual decomposed components* (trend, season) of the real benchmark datasets (ETT, Electricity) for all models, showing that lower SDE correlates with better downstream performance.
2. **Sensitivity analysis of hyperparameters γ and γ′.** The asymmetric weighting introduces new hyperparameters. A sensitivity analysis is needed to show the method's performance is not brittle to their choice and to provide practical guidance for setting them.
3. **Investigation of what the model learns when periodicity is absent.** Many real-world series are non-stationary or lack strong periodicity. The paper must analyze the method's behavior on such data: does it incorrectly amplify noise, or does the asymmetry factor ∆ correctly remain near zero?
4. **Explanation for why direct SDE regularization fails.** The paper notes SDE regularization doesn't work (Table 2) but only hypothesizes about "unconstructive gradients." A gradient analysis or visualization of the loss landscape is needed to substantiate this claim and justify the more complex asymmetric weighting approach.

### Visualizations & Case Studies
1. **Visualization of the embedding space for trend/seasonal components.** A 2D projection (e.g., t-SNE/PCA) of the embeddings *v(a)*, *v(b)*, and *v(a+b)* for baseline CoST and CoST+APW would visually demonstrate whether the proposed method better separates or linearly relates these semantic components.
2. **Case studies of forecasting failures/successes.** Show specific time-series examples where CoST+APW significantly outperforms CoST, and link this visually to the estimated asymmetry factor ∆ and the component recoverability (SDE). Conversely, show where it fails.
3. **Visual trace of the dynamic loss weights during training.** Plot the evolution of the weights (1+γ·∆) and (1+γ′·(-∆)) over training epochs for different datasets to demonstrate the adaptive rebalancing in action.

### Obvious Next Steps
1. **Include long-horizon forecasting results.** The paper evaluates up to 720 steps but the results (e.g., for ETTh1 at 720) show CoST outperforming CoST+APW. This must be discussed and analyzed, as it directly contradicts the claim of improved representation robustness.
2. **Clarify the evaluation protocol for the representations.** The paper is unclear on whether the frozen representations are used for forecasting or if a linear/probe model is fine-tuned. This needs explicit description, as it affects comparability and interpretation.
3. **Compare against a simple, non-learned reweighting baseline.** A critical baseline is to use a fixed, pre-defined weighting scheme for *L_season* and *L_trend* based on prior knowledge of dataset seasonality strength. This would test whether the adaptive mechanism is truly necessary.

# Final Consolidated Review
## Summary
This paper identifies a semantic imbalance problem in contrastive time-series representation learning, where dominant components (e.g., trend) suppress weaker ones (e.g., seasonality). It proposes a Semantic Disentanglement Error (SDE) to quantify this imbalance and an adaptive asymmetric weighting mechanism that can be integrated into frameworks like CoST, demonstrating improved forecasting accuracy on standard benchmarks.

## Strengths
- Clear diagnosis of semantic imbalance via the SDE metric, empirically validated through controlled synthetic experiments (Table 1) showing that SDE asymmetry correlates with component amplitude ratios in TS2Vec.
- Practical, pluggable solution that enhances existing contrastive frameworks without major architectural overhaul, as evidenced by consistent forecasting improvements over baselines like CoST on real-world datasets (ETT, Electricity, Weather).

## Weaknesses
- Methodological clarity and consistency: The description is fragmented, with SDE defined generally in Section 3.2 but operationalized using an MLP for composite embeddings only in Section 4.4.2, and terminology fluctuates between "Separability," "Decomposition," and "Disentanglement." This obscures the approach and undermines reproducibility.
- Missing ablation study: The paper claims gains from asymmetric weighting but does not isolate its contribution by comparing CoST with and without asymmetric weighting (APW). Without this, it is difficult to attribute improvements specifically to the proposed mechanism.
- Omission of key implementation details: Hyperparameters for the asymmetric weighting (γ, γ′) and the architecture of the MLP used for composite embeddings are not specified, hindering reproducibility and fair comparison.
- Presentation issues: The main results table (Table 3) is poorly formatted in the submission, making it challenging to assess the claimed performance gains and SDE metrics fully.
- Insufficient discussion of limitations: While the conclusion notes future work, there is limited analysis of scenarios where the method might fail (e.g., on data without clear periodicity) or why direct SDE regularization fails, leaving gaps in understanding.

## Nice-to-Haves
- Statistical significance testing for forecasting results to bolster claims of consistent gains, though single-run evaluation may be the norm in this area.
- Sensitivity analysis of the hyperparameters γ and γ′ to provide guidance for practitioners.
- Visualizations of embedding spaces or dynamic weight evolution during training to offer qualitative insights into the rebalancing mechanism.
- Ablation study on the MLP fusion component to determine if gains stem from the weighting mechanism or added model capacity.

## Novel Insights
The paper introduces SDE as a directional measure of component recoverability in time-series embeddings and leverages it to dynamically re-weight contrastive losses, addressing an underexplored form of representation collapse. This provides a novel diagnostic and optimization strategy for balancing semantic components in self-supervised time-series learning.

## Suggestions
- Consolidate the method description into a single, coherent section, clarifying how SDE is computed during training (e.g., via batch estimates) and unifying terminology.
- Perform an ablation study comparing CoST, CoST with fixed asymmetric weights, and CoST with dynamic weighting (APW) to isolate the contribution of the adaptive mechanism.
- Include key implementation details in the main text or appendix: specify hyperparameters γ, γ′ and the MLP architecture.
- Ensure Table 3 is clearly formatted in the final version, with all metrics (MSE, MAE, SDE) presented legibly for full transparency.
- Add a brief broader impact statement as per ICLR guidelines, even if societal impact is neutral.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
