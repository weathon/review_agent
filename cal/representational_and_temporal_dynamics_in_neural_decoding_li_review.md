=== CALIBRATION EXAMPLE 14 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Abstract
The abstract clearly states the problem, approach, and high-level findings. It claims the temporal lag analysis provides a "principled framework" for evaluating behavioral relevance. This is a strong claim that the paper must substantiate. The abstract correctly summarizes the three main contributions but does not oversell them.

### Introduction & Motivation
The introduction effectively surveys relevant literature on linear vs. nonlinear decoding and the debate over position vs. velocity representations. The gap is clearly identified: it remains unclear whether models capture true causal dynamics or superficial correlations. The three stated contributions are concrete and address this gap. However, the novelty of the temporal lag analysis could be more sharply contrasted with prior work (e.g., how does it go beyond standard cross-correlation or "optimal lag" analyses?).

### Method
*   **Data & Preprocessing:** The description is adequate for reproducibility. A significant limitation is acknowledged later (single mouse, 3 sessions) but is a major weakness that undermines the study's generalizability and statistical power. The use of a 2ms bin and Gaussian smoothing is standard. The creation of lagged datasets is clear.
*   **Models:** The linear model description has a minor error: **X** ∈ ℝ^(D×N) typically has D samples (time bins) and N features (neurons), making **w** ∈ ℝ^(N×K). The closed-form solution is correct but assumes no regularization; it's unclear if regularization was used. The LSTM description is standard.
    *   **Major Concern:** The FingerFlex model is introduced with its composite loss (MSE + cosine distance). The rationale for using this specific loss, designed for finger movement decoding from ECoG, is **not provided** for decoding 3D hand kinematics from spikes. This is a significant methodological choice that needs justification. Cosine distance is sensible for directional velocity but less so for position; its impact on results is not analyzed.
*   **Experimental Protocol:** The use of Nested Cross-Validation is appropriate. However, key details are missing:
    1.  **Hyperparameter Tuning:** What hyperparameters were tuned for LSTM and FingerFlex (e.g., hidden layers, units, kernel sizes)? The paper states the inner loop was used for "hyperparameter tuning or early stopping" but reports no specifics.
    2.  **Inconsistency:** The text states "the number of training epochs is 1000" and later "Each deep learning model was trained for 3000 epochs." This needs clarification.
    3.  **Evaluation Metric:** Using R² and retaining negative values is correct. However, the two evaluation schemes (trial-average vs. concatenated-predictions) yield dramatically different results (Fig 1a). The paper argues concatenation is "commonly adopted," but this effectively mixes trial structure and may artificially reduce the impact of trial-by-trial variability, potentially inflating scores. The validity of comparing results from these two different computations is questionable and needs stronger justification.

### Experiments & Results
*   **Figure 1 & Section 3.1:** The result that position decoding outperforms velocity is clear. The explanation that velocity has higher variance is plausible but not directly verified (e.g., by reporting signal variance). The dramatic difference between evaluation methods is a red flag; it suggests model performance is highly sensitive to how trial variability is handled, a crucial detail for interpreting all subsequent results.
*   **Figure 2 & Section 3.2 (Temporal Analysis):** This is the core novel analysis. The heatmaps effectively show performance across lags. The finding that performance is similar across a broad range of lags (especially for position) is interesting and supports the "projection" hypothesis over precise temporal decoding.
    *   **Key Missing Analysis:** The claim that LSTM shows "no noticeable performance improvement over the linear model" is visual. A statistical comparison of performance (e.g., mean R² across lags) between linear and nonlinear models is **absent**. Without this, the conclusion that nonlinear models offer no advantage is not rigorously supported.
    *   The observation about the "Right" coordinate is interesting but not explored. Is this a consistent feature of the task or recording geometry?
*   **Figure 3 & Section 3.3 (Linear Model Visualization):** This is a creative analysis. The UMAP visualization suggests smoother manifolds for position decoders across lags compared to velocity.
    *   **Major Concerns:**
        1.  **Interpretation:** The authors interpret smoothness as indicating "structurally aligned" information. However, smoothness in UMAP space could also arise simply because the optimal linear mapping for position changes more continuously with lag than for velocity. This might be a property of the signal, not necessarily of its alignment with neural space. The analysis is suggestive, not conclusive.
        2.  **UMAP Parameters:** UMAP visualizations are highly sensitive to hyperparameters (e.g., `n_neighbors`, `min_dist`). These are not reported, making the visualization non-reproducible and potentially misleading.
        3.  **Quantification:** The cosine similarity matrices (Fig 3b) partially quantify the observation, but a more direct metric (e.g., the rate of change of weights as a function of lag) would strengthen the claim.

### Writing & Clarity
The writing is generally clear. The figures are central to the narrative. A significant point of confusion is the inconsistent reporting of training epochs. The logic flow from results to the "projection mechanism" conclusion is reasonable.

### Limitations & Broader Impact
The limitations section is good and honest, covering the small dataset, limited model exploration, lack of statistical testing, and the correlative nature of the lag analysis. The broader impact is implicit (better understanding of neural coding for improved BMIs) and does not raise significant ethical concerns.

### Overall Assessment
The paper presents an interesting idea: using systematic temporal lags to probe whether decoders capture dynamics or perform static projection. The central finding—that position is decoded more robustly and seems to have a more stable relationship with neural activity across time—is potentially valuable for the field. However, the work is currently **undermined by methodological shortcomings and insufficient rigor** for ICLR. The extremely limited dataset (1 mouse), lack of statistical comparisons between models, unclear hyperparameter tuning, unjustified loss function choice, and qualitative (non-reproducible) visualization analysis mean the evidence for the conclusions is weak. The core insight is promising, but the paper in its current form does not meet the expected bar for acceptance. Major revisions addressing these issues, particularly more robust statistics and validation on a larger dataset, would be required.

# Neutral Reviewer
## Balanced Review

### Summary
This paper systematically compares linear and nonlinear decoders for predicting hand position and velocity from neural spike trains in a mouse reach-to-grab task. Its core methodological contribution is the use of artificial temporal lags between neural and behavioral data to probe whether decoders capture genuine, causally relevant information. The main findings are that position decoding is consistently more accurate and robust than velocity decoding across models, and that linear models perform on par with more complex deep networks (LSTM, FingerFlex), suggesting the decoders act more as static projectors than dynamic interpreters.

### Strengths
1.  **Rigorous Temporal Analysis:** The introduction of artificial time lags is a principled and clever method to dissociate true neural encoding from spurious correlations. The analysis of performance across lags and the realignment of predictions to a common movement onset (Figure 2) provides strong, visual evidence for how information is distributed in time.
2.  **Thorough Model Comparison & Evaluation:** The paper employs a solid nested cross-validation scheme and reports R² scores (including negatives) without truncation, which is methodologically sound and avoids over-optimistic reporting. Comparing a simple linear baseline to two distinct deep learning architectures (LSTM and FingerFlex) across two behavioral representations (position and velocity) is comprehensive.
3.  **Insightful Representational Analysis:** The use of UMAP and cosine similarity to visualize and compare the learned weights of linear models across different lags (Figure 3) is a significant strength. It provides an intuitive, data-driven explanation for why position decoding is more robust, showing smooth manifolds versus the fragmented clusters found for velocity.

### Weaknesses
1.  **Limited Dataset and Generalizability:** The analysis is based on only 3 recording sessions from a single mouse. As noted in the limitations, this severely constrains the statistical power and the ability to generalize the conclusions to other animals, brain regions, or more complex behaviors. For ICLR, where scale and robustness are often valued, this is a major weakness.
2.  **Lack of Statistical Rigor in Claims:** While performance trends are shown, the paper lacks formal statistical testing to support its key claims (e.g., that position decoding is "significantly" better, or that nonlinear models offer "no significant advantage"). Statements about model equivalence or performance differences across coordinates (e.g., the "Right" axis) are presented visually without quantification of confidence or significance.
3.  **Underdeveloped Causal/Interpretive Narrative:** The core finding—that models are "projectors" not true temporal decoders—is interesting but not deeply explored. The discussion does not sufficiently engage with why this might be the case from a neuroscientific perspective (e.g., properties of the recorded population, the task) or what it implies for model design. The link between the lag analysis and causality is asserted but not rigorously defended.

### Novelty & Significance
**Novelty:** The specific combination of temporal lag manipulation with representational visualization (UMAP of model weights) to compare linear/nonlinear models on position vs. velocity decoding is a novel methodological contribution. However, the individual components—using lags to test decoders, finding linear models competitive with nonlinear ones for motor cortex decoding, and position being easier to decode than velocity—have been noted in prior literature.
**Significance:** The paper provides a valuable methodological framework for critically evaluating neural decoders, which is important for the field. It offers concrete evidence that increased model complexity does not necessarily yield better decoding, advocating for simplicity and careful evaluation. Its significance is more methodological than theoretical, offering tools for better practice rather than a new theoretical insight into neural coding.

### Suggestions for Improvement
1.  **Strengthen the Empirical Basis:** The most critical improvement is to validate the core findings on a larger dataset, ideally from multiple animals. This would allow for proper statistical testing of performance differences (e.g., using mixed-effects models) and would dramatically increase the paper's impact and credibility.
2.  **Deepen the Analysis and Discussion:** Move beyond reporting performance to explain it. For instance, analyze the spectral or variance properties of the position vs. velocity signals that might make one easier to decode. More critically discuss *why* the LSTM fails to outperform a linear model—is it a training/data limitation, or does it genuinely indicate the absence of complex long-range temporal dynamics relevant for this decoding task?
3.  **Clarify and Extend the "Projector" Hypothesis:** The paper would be strengthened by a more formal analysis to support the "projector" claim. For example, one could compute the mutual information between neural activity and behavior at different lags and compare it to the decoder performance profiles. Additionally, explicitly testing if a time-wrapped linear model (using a window of binned spikes) performs as well as the LSTM could solidify the argument.
4.  **Improve Presentation and Precision:** The manuscript has minor inconsistencies (e.g., epochs listed as 1000 then 3000). The abstract and conclusion should more precisely state the limited scope (single mouse, M1-dominated recordings). Figures, while informative, could benefit from clearer captions explaining exactly what is being shown in each panel (e.g., in Figure 1a, specify if the boxplots are over trials or folds).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test generalizability across subjects.** The core claims are based on data from a single mouse (3 sessions). Performance could be idiosyncratic to this animal. To claim robust conclusions about position vs. velocity decoding or model comparisons, results must be replicated across multiple mice.
2. **Region-specific decoding analysis.** The paper records from M1, thalamus, striatum, and cerebellum but pools all neurons. A critical missing experiment is to decode separately from each region to test if the reported findings (e.g., position superiority) are consistent across brain areas or driven by a specific region.
3. **Comparison to standard neural decoding baselines.** The linear model is a simple regression. To contextualize the "no advantage for nonlinear models" claim, results must be compared against standard neural decoding baselines like Wiener filters or Kalman filters, which are the established linear benchmarks for continuous decoding.
4. **Ablation of velocity computation method.** The poor velocity decoding could stem from the 8th-order central difference filter amplifying noise. An essential experiment is to decode velocity computed via alternative methods (e.g., smoothing then differentiation) to test if the result is an artifact of preprocessing.

### Deeper Analysis Needed (top 3-5 only)
1. **Statistical significance testing for all comparisons.** The paper presents performance scores but provides no statistical tests (e.g., ANOVA, pairwise tests with correction) to support claims that position is "consistently decoded with higher accuracy" or that models do not differ significantly. Without this, the results are merely suggestive.
2. **Analysis of temporal error structure.** The claim that models act as "projection mechanisms" rather than capturing dynamics requires analyzing the temporal structure of errors. For instance, does error autocorrelation differ between linear and LSTM models? A simple comparison of R² masks potential differences in how errors unfold over time.
3. **Explain the "Right" axis performance drop.** The consistently poor decoding for the "Right" coordinate is noted but not investigated. A crucial analysis is to examine if this is due to lower behavioral variance, poorer neural tuning, or a coordinate frame misalignment, as this undermines the claim of robust position decoding.
4. **Quantify representation smoothness/fragmentation.** The UMAP visualization in Figure 3 is qualitative. To substantiate the claim that position representations are "smoother," the analysis must provide a quantitative measure (e.g., neighborhood preservation metric across lags, or a smoothness index) comparing position and velocity model spaces.

### Visualizations & Case Studies
1. **Show predicted vs. true trajectories for all lags and models.** Figure 1b shows one linear model trajectory at zero lag. To evaluate the lag manipulation's effect and the "projection" claim, show overlaid predicted trajectories (for both position and velocity) across multiple lags for single trials, contrasting linear and LSTM outputs.
2. **Visualize model failures, especially for velocity.** Case studies of trials where velocity decoding fails catastrophically (negative R²) are needed to diagnose whether failures are due to noise, phase misalignment, or representational limits. This would validate the claim that velocity is a harder decoding problem.
3. **Visualize latent dynamics of the LSTM.** To test if the LSTM is merely performing projection, visualize its hidden state dynamics (e.g., via PCA) during a trial and compare to neural PCA trajectories. If they are similar, it supports the projection claim; if the LSTM exhibits different dynamics, it may be capturing transformed features.

### Obvious Next Steps
1. **Incorporate multiple animals.** This is not a "future work" item; it is a fundamental requirement for a credible systems neuroscience study at ICLR. The paper's conclusions cannot be trusted until shown to hold across subjects.
2. **Perform a proper baseline comparison.** The paper should have included Wiener/Kalman filters as standard linear dynamical decoders. Their absence makes the model comparison incomplete and less convincing for the neuroscience community.
3. **Conduct a statistical evaluation of results.** Every major comparison (position vs. velocity, linear vs. nonlinear, across lags) needs formal statistical testing with appropriate corrections. This is standard practice and should have been in the main paper.
4. **Analyze decoding per brain region.** Given that different regions were recorded, the natural and obvious next step is to report if the primary motor cortex (M1) drives the results or if other areas contribute differentially to position/velocity decoding. This is a missed opportunity to deepen the mechanistic insight.

# Final Consolidated Review
## Summary
This paper systematically compares linear and nonlinear models for decoding hand position and velocity from neural spike trains during a mouse reach-to-grab task. Its core methodological contribution is the introduction of artificial temporal lags between neural and behavioral data to probe whether decoders capture genuine, time-specific information or act as static projectors. The main findings are that position decoding is more accurate and robust than velocity decoding across lags, and that linear models perform comparably to more complex deep networks (LSTM, FingerFlex).

## Strengths
- **Principled Temporal Analysis:** The systematic introduction of artificial time lags and the subsequent realignment of predictions to a common movement onset (Figure 2) provides a clever and rigorous framework to dissect how behavioral information is distributed in time and to test whether decoders capture dynamics or superficial correlations.
- **Insightful Representational Analysis:** The visualization and comparison of linear model weights across different lags using UMAP and cosine similarity (Figure 3) offers a data-driven, intuitive explanation for the robustness of position decoding, showing smooth manifolds versus the fragmented clusters found for velocity.

## Weaknesses
- **Limited Dataset Undermines Generalizability:** The analysis is based on only 3 recording sessions from a single mouse. This severely constrains the statistical power and the ability to generalize the conclusions to other animals, brain regions, or behavioral contexts, which is a significant concern for a systems neuroscience study.
- **Lack of Statistical Rigor for Key Claims:** While performance trends are shown visually, the paper lacks formal statistical testing to support its central claims (e.g., that position decoding is significantly better than velocity, or that linear and nonlinear models do not differ significantly). Statements about performance differences across spatial coordinates or model types remain qualitative and unquantified.

## Nice-to-Haves
- A more quantitative measure of the "smoothness" observed in the UMAP visualizations (e.g., a neighborhood preservation metric) would strengthen the representational analysis beyond qualitative description.
- A deeper discussion on *why* an LSTM, designed for temporal dynamics, fails to outperform a simple linear projection in this context—whether it is a data limitation, a property of the neural population code, or a training issue—would enrich the interpretive narrative.

## Novel Insights
The combination of the temporal lag manipulation with representational visualization provides a novel methodological lens. It offers concrete evidence that, for this dataset and task, both linear and nonlinear decoders function more as projectors mapping neural activity onto behavior rather than as interpreters of underlying temporal dynamics. The finding that position information forms a more stable and continuous manifold in model weight space across time lags, compared to velocity, is a genuinely novel observation that helps explain the empirical robustness of position decoding.

## Suggestions
- The most critical improvement is to validate the core findings on data from multiple animals to establish generalizability and enable proper statistical testing of performance differences.
- Incorporate formal statistical comparisons (e.g., pairwise tests with correction) for all major claims regarding performance differences between behavioral variables, model architectures, and spatial coordinates.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0, 0.0]
Average score: 1.0
Binary outcome: Reject
