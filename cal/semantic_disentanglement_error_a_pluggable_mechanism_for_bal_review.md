=== CALIBRATION EXAMPLE 3 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title effectively highlights the core contribution (a disentanglement error metric and a plug-and-play weighting mechanism). However, the abstract claims the approach integrates into existing frameworks like CoST **"without architectural changes,"** which directly contradicts Section 4.4.2, where you introduce a learnable MLP $g_\phi$ to compute the composite embedding. Additionally, the metric is referred to as "Semantic Disentanglement Error" in the title/abstract, "Semantic Separability Error" in Sec 3.2, and "Semantic Decomposition Error" in Sec 4.3. This naming drift should be resolved for consistency. The abstract's claims of "consistent gains" need stronger backing from the results tables, where improvements are sometimes marginal or horizon-dependent.

### Introduction & Motivation
The motivation is well-grounded: contrastive time-series models often bias dominant components (trends) over weaker ones (seasonality), harming downstream utility. The identification of three limitations is clear. However, the conflation of **component suppression** with **isotropic embedding collapse** needs refinement. Isotropic collapse typically refers to representations losing directional information and collapsing to a uniform distribution on the hypersphere (Wang & Isola, 2020), whereas your problem is about *relative representational strength* across semantic subspaces. While related, they are distinct phenomena; attributing the issue partly to collapse without distinguishing the mechanisms weakens the theoretical framing.

### Method / Approach (Sections 3 & 4.4)
This section contains the most significant technical concerns:

1.  **Inconsistent Definition of Composite Embedding:** In Sec 3.2 and 4.3, $v(\mathbf{a}+\mathbf{b})$ is defined as $f_\theta(\mathbf{x})$ (the raw encoder applied to the composite signal). However, Sec 4.4.2 replaces this with an MLP-based mapping: $v(\mathbf{a}+\mathbf{b}) = g_\phi([v(\mathbf{a}) \parallel v(\mathbf{b})])$. This is a substantial architectural change, not just a loss weighting. It must be explicitly clarified which definition is used during training and evaluation. If the MLP is used, SDE is no longer probing the *encoder's* ability to preserve linearity but rather the *fusion module's* capacity to interpolate, which changes the interpretation of the metric.
2.  **Linearity Assumption in Latent Space:** The SDE metric relies on the assumption that $v(\mathbf{a}+\mathbf{b}) - v(\mathbf{b}) \approx v(\mathbf{a})$ (cosine similarity of the difference). This linear superposition property is well-known in static word embeddings but is rarely guaranteed in contrastive representations, especially those optimized with InfoNCE-style objectives that map to hyperspheres. No theoretical or empirical justification is provided for why time-series contrastive embeddings should exhibit this linearity. If this assumption fails, SDE measures correlation rather than true separability.
3.  **Optimization Stability Risk in Asymmetric Weighting:** The weighting formula is $\mathcal{L} = (1 + \gamma \Delta)\mathcal{L}_{season} + (1 + \gamma' (-\Delta))\mathcal{L}_{trend}$. From Table 1, $\Delta$ can easily exceed 1.0. If $\Delta > 1/\gamma'$, the weight on $\mathcal{L}_{trend}$ becomes **negative**. Assigning a negative weight to a contrastive loss effectively maximizes it, pushing trend embeddings apart rather than aligning them, which is mathematically unsound for this objective and will likely destabilize training. You must clarify whether weights are clamped (e.g., $\max(0, \cdot)$), normalized, or exponentiated.
4.  **Gradient Flow Ambiguity:** Section 4.4.3 states gradients flow end-to-end, but calculating SDE requires passing decomposed components $\mathbf{a}$ and $\mathbf{b}$ through the network. In CoST, components are extracted internally; does the SDE computation use detached embeddings to compute the diagnostic, or does it require additional forward passes? The computational graph and stop-gradient strategy are missing.

### Experiments & Results
1.  **Implementation Details:** You state `lr=1e3`, which is almost certainly a typo for $1e^{-3}$. Additionally, 100 epochs with batch size 32 is notably low for self-supervised time-series representation, where convergence often requires longer schedules or cosine annealing. Please confirm the exact training protocol to ensure fair comparison.
2.  **Synthetic Analysis (Table 1):** Table 1 effectively demonstrates asymmetry, but it is critical to clarify the state of the TS2Vec encoder when SDE was measured. If the encoder is randomly initialized, SDE values will be dominated by random projection geometry rather than learned bias. If pre-trained, please state the training procedure.
3.  **Missing Key Ablations:** The paper argues for *dynamic* asymmetric weighting, but there is no ablation against a **static** asymmetric weighting scheme (e.g., fixed weights like $1.2\mathcal{L}_{season} + 0.8\mathcal{L}_{trend}$). Without this, it's unclear whether gains come from the SDE-driven dynamics or simply from rebalancing the loss. Furthermore, does adding MLP fusion $g_\phi$ alone improve CoST? An ablation of "CoST + MLP" vs "CoST + MLP + APW" is needed to isolate the contribution of the weighting mechanism.
4.  **Result Reporting & Metrics:** Table 3 is severely misaligned in the provided text, making precise numerical evaluation difficult. Even accounting for parsing errors, the reported gains of CoST+APW over CoST are often modest (e.g., ETTm1 24-step: 0.012 vs 0.015; Electricity 168-step: 0.566 vs 0.425 where APW actually performs *worse* on MSE). The claim of "consistent gains" is not fully supported by these numbers. Standard deviations or multiple seeds are necessary to establish significance. Additionally, while Sec 4.1 lists SDE as a metric, Table 3 only reports MSE/MAE. Real-dataset SDE values are needed to verify that imbalance is actually reduced in practice.

### Writing & Clarity
The logical flow is generally good, but Sections 4.3 and 4.4.1 repeat definitions introduced in Section 3. Section 4.4.2 introduces the MLP abruptly; it should be integrated into the method section earlier. The explanation of how SDE is computed during training vs. evaluation needs to be unified. Avoid phrases like "We hypothesize that SDE... fails to provide constructive optimization gradients" in Sec 4.3; instead, analyze the gradient norms or Hessian properties to support this empirically.

### Limitations & Broader Impact
The future work section correctly identifies the need to analyze the MLP and LPF approximations. However, the current limitations section misses critical discussion points:
*   **Computational Overhead:** Dynamic SDE requires additional forward passes (or MLP evaluations) and cosine computations per batch. What is the training time overhead compared to CoST?
*   **Multivariate Handling:** Section 3.1 defines $\mathbf{X} \in \mathbb{R}^{T \times d}$. SDE computes a single cosine similarity. How is this aggregated across $d$ channels? Global averaging? Channel-wise computation? This choice significantly impacts the metric's scalability and stability.
*   **Failure Modes:** What happens if the initial decomposition (LPF) misclassifies components? The method assumes clean separation of trend/seasonality; in noisy signals with broad spectral overlap, $\Delta$ may reflect decomposition error rather than representation bias.

### Overall Assessment
The paper addresses a genuine problem in contrastive time-series learning: semantic imbalance where dominant dynamics suppress informative weak signals. The proposal to use a directional recoverability metric (SDE) to guide loss weighting is conceptually sound and potentially useful. However, the manuscript currently falls short of the ICLR acceptance bar due to critical methodological ambiguities and experimental gaps. The inconsistency in defining the composite embedding, the risk of negative loss weights destabilizing optimization, the unverified linearity assumption in latent space, and the lack of ablations isolating the dynamic weighting from architectural changes are substantial technical concerns. Additionally, the experimental results require clearer presentation, statistical validation, and proof that SDE actually decreases on real-world benchmarks. Addressing the optimization safety (e.g., weight clamping), clarifying the computational graph for SDE, providing static-weighting ablations, and fixing the contradiction regarding "no architectural changes" would significantly strengthen the paper for resubmission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a semantic imbalance problem in contrastive time-series representation learning, where dominant components (e.g., trends) suppress weaker but meaningful signals (e.g., seasonality). To address this, the authors introduce the Semantic Separability Error (SDE), a metric to quantify component recoverability, and propose an asymmetric perceptual weighting scheme that dynamically rebalances multi-view contrastive losses. The method integrates into existing dual-stream frameworks like CoST and demonstrates improved forecasting accuracy across several benchmark datasets.

### Strengths
1. **Well-Motivated Diagnostic with Empirical Validation:** The SDE metric effectively exposes representation asymmetry under controlled conditions. Table 1 demonstrates a clear monotonic relationship between component amplitude ratio and SDE/∆, validating that the proposed metric reliably detects semantic skew in TS2Vec embeddings.
2. **Scientifically Sound Iterative Design:** The authors systematically test direct SDE regularization, observe its failure (Table 2), and correctly hypothesize that the diagnostic metric lacks constructive optimization gradients. Pivoting to asymmetry-aware loss reweighting shows strong methodological maturity and justifies the final design choice.
3. **Lightweight, Architecture-Compatible Integration:** By operating at the loss level rather than modifying encoder architectures, the approach maintains compatibility with frameworks like CoST that already separate trend/seasonal streams. This aligns with the stated goal of a low-overhead mechanism.

### Weaknesses
1. **Overstated "Plug-and-Play" Claim & Decomposition Dependency:** The SDE computation and weighting mechanism require explicit access to decomposed components (**a** and **b**). In practice, these are extracted via heuristic low-pass filtering, and the method cannot be directly applied to single-encoder contrastive models (e.g., TS2Vec) without architectural changes. The claim of plug-and-play applicability is therefore inaccurate.
2. **Outdated Baselines & Inconsistent Performance Claims:** The evaluation relies primarily on methods from 2018–2022 (TS2Vec, TNC, CoST), missing recent self-supervised and foundation time-series models. Furthermore, Table 3 shows that TS2Vec occasionally matches or outperforms CoST+APW on the Weather dataset at longer horizons (168+), which contradicts the abstract's claim of "consistent gains." A more nuanced discussion of failure cases is missing.
3. **Reproducibility & Hyperparameter Gaps:** Critical implementation details are absent or ambiguous. The learning rate is listed as `1e3` (likely a typo for `1e-3`), low-pass filter specifications (cutoff, order) are unspecified, and hyperparameters `γ, γ′` lack tuning ranges or sensitivity analysis. This hinders faithful reproduction.
4. **Theoretical Fragility of SDE:** SDE assumes linear superposition in the embedding space (`cos(v(a+b)−v(b)), v(a)`). Deep encoders are highly nonlinear, making this approximation dataset- and architecture-dependent. The paper lacks analysis of how encoder depth, nonlinearity, or noisy decompositions affect SDE reliability.

### Novelty & Significance
**Novelty:** Moderate. The idea of dynamically weighting multi-view or multi-objective contributions exists in other representation learning subfields. However, deriving an adaptive weighting signal from a component-recoverability error specifically for temporal contrastive learning is a novel and practically useful incremental contribution. The SDE metric itself offers a fresh diagnostic perspective on representation geometry.
**Significance:** The work tackles a recognized yet under-explored issue in temporal self-supervision: spectral dominance and embedding collapse toward high-amplitude signals. The demonstrated improvements in forecasting, particularly under skewed distributions, provide practical value. For ICLR, however, the paper falls slightly short of the acceptance bar due to limited benchmark context, unresolved theoretical assumptions, and missing reproducibility details. Strengthening empirical rigor and theoretical grounding would elevate it to a strong poster or oral candidate.

### Suggestions for Improvement
1. **Revise Claims & Clarify Scope:** Explicitly state that the method requires architectures with explicit or extractable component streams (e.g., CoST, decomposition-based models). Update the abstract and introduction to reflect this constraint rather than claiming universal plug-and-play applicability.
2. **Modernize Baselines & Discuss Mixed Outcomes:** Include comparisons with more recent temporal SSL methods (e.g., TS-TCC, SimMTM, or contemporary time-series foundation models) to situate contributions in the current landscape. Additionally, provide a dedicated analysis of instances where baselines outperform the proposed method (e.g., Weather dataset), discussing whether data characteristics (low SNR, high noise, weak seasonality) explain the gap.
3. **Provide Full Reproducibility Details & Ablations:** Correct the learning rate notation, specify decomposition parameters (filter type, order, cutoff frequency), and add a hyperparameter sensitivity study for `γ` and `γ′`. A brief FLOPs/memory footprint comparison against vanilla CoST would also strengthen the practicality argument.
4. **Strengthen Theoretical Grounding of SDE:** Add an empirical or analytical discussion on when the linear superposition assumption holds. Consider validating SDE with embedding geometry metrics (e.g., centered kernel alignment, pairwise distance distributions, or spectral analysis of learned representations) to demonstrate that SDE tracks true semantic separability beyond heuristic cosine approximations.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add anomaly detection and classification benchmarks.** The introduction claims semantic imbalance degrades performance across multiple tasks, yet all experiments are strictly forecasting. Without non-regression downstream results, the core claim of improved universal representation remains unverified.
2. **Run a strict ablation isolating dynamic weighting from the fusion MLP.** Section 4.4 adds a multi-layer perceptron $g_\phi$ while claiming a "pluggable" method; without removing the MLP, gains cannot be attributed to asymmetric weighting rather than added capacity.
3. **Report SDE values on real-world datasets for all compared methods.** The central hypothesis is that the method balances semantics, yet Table 3 only reports MSE/MAE; proving SDE actually decreases on ETT/Weather in the proposed model is essential to validate the mechanism.

### Deeper Analysis Needed (top 3-5 only)
1. **Provide mathematical or gradient-flow analysis explaining why SDE fails as a direct regularizer but succeeds as a loss multiplier.** Dismissing SDE regularization as lacking "constructive gradients" without empirical loss landscape or Jacobian evidence undermines the design rationale.
2. **Quantify the statistical correlation between reduced SDE and improved forecasting error.** If semantic balance does not strongly correlate with downstream gains, the mechanistic narrative linking imbalance to poor forecasting is speculative rather than proven.
3. **Analyze optimization stability under $(1+\gamma\Delta)$ scaling.** Dynamic reweighting risks gradient explosion or training oscillation; demonstrating bounded weight behavior or loss variance over time is necessary to prove practical reliability.

### Visualizations & Case Studies
1. **Plot training trajectories of $\Delta$ and the resulting component weights across epochs.** Visualizing adaptation dynamics will reveal whether the mechanism actively rebalances optimization or quickly saturates into static weighting.
2. **Show subspace alignment/uniformity plots for trend vs. seasonal embeddings across models.** Direct geometric comparison will expose whether the method genuinely mitigates isotropic collapse or merely shifts error between components.
3. **Provide decomposition residual plots on signals with extreme amplitude ratios ($r$).** Case studies visualizing recovered trend and seasonal components under high skew will prove the method captures intended semantics versus simply improving overall fit via confounding correlations.

### Obvious Next Steps
1. **Report mean $\pm$ std over $\geq 3$ random seeds with statistical significance tests.** Table 3 presents single-point results, which ignores forecasting variance and fails to meet ICLR’s empirical rigor standards.
2. **Test the weighting mechanism on at least one non-CoST baseline architecture.** Limiting evaluation to CoST contradicts the abstract's "pluggable" claim and leaves it unclear whether this generalizes to other contrastive or transformer-based TS models.
3. **Include a hyperparameter sensitivity sweep for $\gamma$ and $\gamma'$.** Showing performance across a range of scaling sensitivities is required to prove the method is robust rather than brittlely tuned to specific dataset properties.

# Final Consolidated Review
## Summary
This paper identifies a semantic imbalance in contrastive time-series representation learning, where dominant trend components suppress weaker seasonal signals, and proposes a diagnostic metric (SDE) alongside an asymmetric perceptual weighting (APW) scheme to dynamically rebalance loss objectives. Integrated into the CoST framework, the method demonstrates improved forecasting performance across several standard benchmarks while attempting to preserve representation of underrepresented temporal components.

## Strengths
- **Clear diagnostic with controlled empirical validation:** The SDE metric successfully exposes representation asymmetry under synthetic conditions. Table 1 demonstrates a monotonic relationship between component amplitude ratios and SDE/$\Delta$, validating that the metric reliably detects semantic skew when one temporal factor dominates.
- **Methodologically mature iterative design:** The authors systematically explore direct SDE regularization, empirically observe its failure (Table 2), and correctly pivot to multiplicative loss reweighting. This progression demonstrates strong scientific reasoning and justifies the final architectural choice over simpler alternatives.
- **Targeted mitigation of a documented failure mode:** By explicitly quantifying and correcting seasonal-trend imbalance, the method yields consistent MSE/MAE improvements over vanilla CoST and TS2Vec on ETT and Electricity datasets, directly addressing a known limitation in time-domain-only contrastive learning.

## Weaknesses
- **Contradictory claims regarding architectural modifications:** The abstract claims the approach integrates into frameworks like CoST "without architectural changes," yet Section 4.4.2 explicitly introduces a learnable MLP $g_\phi$ to compute the composite embedding $v(a+b)$. This addition increases parameter count, alters the computational path for SDE evaluation, and undermines the stated plug-and-play claim for single-stream contrastive models.
- **Unbounded loss weights risk optimization instability:** The weighting formula $(1 + \gamma' \cdot (-\Delta))\mathcal{L}_{trend}$ can produce negative coefficients when $\Delta > 1/\gamma'$. Table 1 shows $\Delta$ values exceeding 1.1, and the paper does not specify clamping, normalization, or bounded activation functions. Negative weights on contrastive objectives would actively repel trend embeddings, contradicting alignment goals and potentially destabilizing training.
- **Unverified linearity assumption in latent space:** SDE relies on vector arithmetic $v(a+b) - v(b) \approx v(a)$, assuming additive composability in the embedding space. While this property holds for some static word embeddings, contrastive time-series models typically normalize representations to a hypersphere and optimize non-linear objectives, making linear superposition unguaranteed. Without empirical or theoretical justification, SDE may capture superficial cosine alignments rather than true semantic separability.
- **Missing empirical validation of the core mechanism on real data:** The central hypothesis is that APW reduces semantic imbalance, and the text claims "consistently lower SDE values." However, Table 3 reports only MSE and MAE on real-world datasets (ETT, Electricity, Weather) without corresponding SDE or $\Delta$ scores. Demonstrating actual SDE reduction is essential to verify that the weighting mechanism successfully rebalances representations rather than improving forecasting through other means.
- **Insufficient ablations isolate dynamic weighting gains:** The paper lacks comparisons against (1) a static asymmetric weighting scheme (e.g., fixed multipliers) and (2) CoST augmented with only the fusion MLP $g_\phi$ (CoST+MLP). Without these baselines, it remains unclear whether performance gains originate from the SDE-driven dynamic adaptation or simply from added representational capacity and fixed loss rebalancing.

## Nice-to-Haves
- **Reproducibility specifications:** Clarify low-pass filter parameters (type, cutoff frequency, order), channel aggregation strategy for multivariate SDE computation, and exact learning rate notation (`lr=1e3` is likely a typo for $10^{-3}$).
- **Optimization dynamics analysis:** Plot $\Delta$ and resulting weight trajectories over training epochs to visualize whether the mechanism actively adapts or quickly saturates, and provide a brief sensitivity sweep for $\gamma, \gamma'$.
- **Statistical reporting:** Report mean $\pm$ standard deviation across multiple random seeds and clarify whether single-run evaluations reflect typical variance in time-series forecasting benchmarks.

## Novel Insights
The paper’s experimental pivot from SDE regularization to SDE-driven loss weighting reveals a broader principle in self-supervised representation learning: geometric diagnostics derived from vector-space relationships often suffer from misaligned gradients when used as direct optimization penalties, yet become highly effective when repurposed as dynamic multipliers that modulate training focus. This suggests that diagnostic metrics in deep representation learning should be evaluated primarily for their correlation with semantic structure rather than their mathematical tractability as loss functions, and that adaptive optimization signals can compensate for structural limitations in contrastive objectives.

## Suggestions
- Explicitly reconcile the abstract's "no architectural changes" claim with the MLP fusion in Section 4.4.2, and clearly state that the method assumes access to decomposable or multi-view streams.
- Introduce a weight-bounding strategy (e.g., $\max(0, \cdot)$ or softmax normalization) for the asymmetric weighting formula to prevent negative loss coefficients, and report the exact $\gamma, \gamma'$ values used.
- Conduct and report ablations for CoST+MLP (no APW) and CoST+APW with static weights to isolate the contribution of dynamic SDE-driven adaptation.
- Include real-dataset SDE/$\Delta$ values alongside forecasting metrics in Table 3 to empirically validate that semantic imbalance is actually mitigated in practice.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
