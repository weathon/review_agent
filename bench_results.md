# ICLR Benchmark Results

Date: 2026-04-12 23:55
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## 1c73HCZpbo

- GT: Reject (avg 4.0)
- Predicted: Reject (4.5/10)
- Match: YES

### Final Review

## Summary
REVEAL-IT proposes a framework for explaining and optimizing the learning process of RL agents by visualizing policy weight updates as node-link graphs and using GNN-based models to (1) predict learning progress across training tasks and (2) highlight important policy updates. The method simultaneously provides interpretability about *why* an agent succeeds or fails while using the predicted learning progress to dynamically optimize task sequences (curriculum learning). Experiments are conducted on ALFWorld and OpenAI Gym environments.

## Strengths
- **Dual-purpose framework combining interpretability with curriculum optimization**: The paper attempts to bridge two related but often separate goals—understanding agent behavior and using that understanding to improve training. This aligns well with the broader goal of actionable interpretability.
- **Algorithm-agnostic design**: Table 2 demonstrates the method works across multiple RL algorithms (PPO, A2C, PG), showing generality beyond a single training paradigm.
- **Novel visualization approach**: Using node-link diagrams to track weight updates across training provides a dynamic view of policy evolution, contrasting with static saliency maps or 2D-only value function visualizations that prior work relies on.
- **Strong empirical results on ALFWorld if valid**: The reported 0.80 success rate substantially outperforms baselines, though the validity of this result is tempered by comparison methodology concerns (see Weaknesses).

## Weaknesses
- **Inappropriate baseline comparisons in Table 1 isolate curriculum effects poorly**: REVEAL-IT combines curriculum learning with GNN-based task optimization. The primary baselines (MiniGPT-4, BLIP-2, LLaMA-Adapter, InstructBLIP) are zero-shot or few-shot VLMs with fundamentally different learning paradigms and compute budgets. While PPO (0.04) is included as a baseline, it does not use curriculum learning, making it impossible to determine whether gains come from (a) the curriculum structure, (b) the GNN-based optimization, or (c) other factors. The paper cites curriculum RL literature (Narvekar et al., 2020; Held et al., 2017) but does not evaluate against any standard curriculum learning baselines such as self-paced learning or ALP-GMM. This undermines the central claim that the *GNN-based explainer* specifically drives the improvements.

- **No quantitative evaluation of explanation quality**: The paper claims to provide "intuitive and comprehensible explanations" but validates this entirely through visual inspection of Figure 2. There is no fidelity metric (does masking the highlighted weights actually degrade performance?), no faithfulness score, no comparison to ground-truth importance, and no human study verifying that the explanations help users understand agent behavior. Without any of these, the interpretability contribution remains unsubstantiated empirically.

- **Missing statistical significance reporting**: Neither Table 1 nor Table 2 reports standard deviations, confidence intervals, or variance across seeds. Given the well-documented high variance in RL experiments (Henderson et al., 2018), this omission makes it impossible to assess whether reported differences are statistically meaningful.

- **Mixed results on OpenAI Gym environments are not discussed**: Table 2 shows REVEAL-IT degrades performance on several environments: Hopper (PPO: 2250.46 → 2104.88), InvertedPendulum (A2C: 1002.48 → 966.20), Hopper (PG: 2489.07 → 2253.70). These failures are not acknowledged or analyzed. The efficiency framing (reporting results at fewer training steps) is valid for measuring sample efficiency, but performance regressions at those step counts should be discussed.

- **Ambiguous relationship between GNN predictor and explainer**: The paper introduces both a GNN predictor (for learning progress estimation) and a GNN explainer (for highlighting important weight updates), but their interaction is unclear. Algorithm 1, Line 7 uses {P(task_n, π_t)}—the *true* learning progress—for task sampling, not the predicted value. It remains ambiguous whether the predictor is actually used during training or only for offline analysis. Additionally, Section 4.2 initially states "The overall goal of the GNN explainer is to learn to optimize the sequences of training tasks" before distinguishing the two components, creating conceptual confusion.

- **Missing implementation detail for task sampling**: Algorithm 1, Line 7 states "Sample training task sequence Seq_t in terms of {P(task_n, π_t)}" without specifying the sampling distribution (softmax? proportional? greedy?). This is a critical detail for reproducibility.

- **Scalability concerns unaddressed**: The method is demonstrated on a 4-layer × 64-node MLP (~12,000 edges). Policy networks for complex tasks often use CNNs, transformers, or much larger architectures. The paper claims "no limitations on the RL algorithm" but the visualization and GNN construction fundamentally assume MLP structure with fixed layer sizes. How this scales to deeper/wider networks or non-MLP architectures is not addressed.

## Nice-to-Haves
- Evaluation on CNN or transformer policies to validate generalizability beyond small MLPs
- Comparison of the optimized task sequence against random task scheduling as an ablation
- Human study or quantitative interpretability metric (e.g., fidelity, sparsity) to validate explanation quality
- Computational overhead analysis for training the GNN explainer concurrently with the RL agent

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"Citation missing BUTLER"**: The reviewer claims BUTLER (Micheli & Fleuret, 2021) should be included as a baseline. We cannot verify this reference exists or is relevant without external sources, and requesting missing citations is outside scope.

- **"POMDP framing is mentioned then dropped"**: While true, this is a minor writing issue (an unused sentence) rather than a substantive flaw affecting the method's validity.

- **"Table 2 comparison is unfair because REVEAL-IT uses fewer steps"**: The efficiency framing (better performance at fewer steps) is actually a valid way to measure sample efficiency. The real concern is that some environments show performance regression, which is a different issue.

- **"Demanding theoretical justification for learning progress"**: The learning progress signal (Eq. 1) is a standard formulation from curriculum learning literature. Requesting theoretical proofs for what is empirically evaluated is scope creep.

## Novel Insights
The paper's visualization reveals an interesting pattern: certain policy weights are updated across multiple related subtasks (e.g., "open microwave" and "take apple from microwave" share spatial reasoning components), forming a kind of "shared capability" structure in the policy. The GNN explainer appears to identify these intersections, which aligns with the intuition that curriculum learning should prioritize teaching foundational skills before composite tasks. However, this insight is presented only qualitatively through Figure 2—the paper would be substantially stronger if it quantified this phenomenon (e.g., measuring overlap between highlighted subgraphs across tasks) and correlated it with curriculum effectiveness.

## Suggestions
- Replace or supplement VLM baselines in Table 1 with standard curriculum RL methods (e.g., self-paced learning, teacher-student curriculum, PLR) to isolate whether the GNN-based optimization specifically contributes beyond naive curriculum approaches.
- Add a simple ablation: random task scheduling vs. GNN-predicted task scheduling to quantify the curriculum optimization contribution.
- Include at least one quantitative interpretability metric. A straightforward choice: mask the top-k weights highlighted by the explainer and measure performance degradation—if the highlighted weights are truly important, masking them should hurt performance.
- Report mean ± std across multiple random seeds (at least 3-5) for Table 2 to establish statistical significance of improvements and acknowledge environments where performance regresses.
- Clarify in Algorithm 1 and text: (a) the exact sampling distribution for task selection, (b) whether the GNN predictor's predictions are used during training or only for post-hoc analysis, and (c) the distinction between "explainer" (visualization) and "predictor" (curriculum optimization) roles throughout.

## Assessment
**Novelty**: Moderate. The idea of treating policy weight updates as a graph and using GNN explainers for curriculum optimization is creative, but GNN explainers and learning progress signals are borrowed from existing work.

**Technical Soundness**: Weakened by missing implementation details, unclear role of predictor vs. explainer, and lack of proper curriculum baselines.

**Empirical Support**: Mixed. Strong ALFWorld results are undermined by inappropriate baselines; OpenAI Gym results show improvements in some environments but regressions in others; no statistical significance; interpretability claims are entirely qualitative.

**Significance**: Potentially useful if the method's contributions can be properly isolated and validated, but currently conflates curriculum learning gains with GNN-specific gains.

**Clarity**: Moderate. The dual goals (interpretability + curriculum) create some terminological confusion; the relationship between predictor and explainer could be clearer.

MY FINAL SCORE: <pineapple>4.5</pineapple>

---

## 33P4evE2ej

- GT: Reject (avg 4.8)
- Predicted: Accept (5.5/10)
- Match: NO

### Final Review

## Summary

This paper proposes DynaMer Adapter, a method for adapting Vision Transformers to medical imaging tasks by dynamically merging tokens from two pre-trained models: a general-domain ViT (DINO v2) and a medical-domain ViT. The architecture combines a Gated Mixture-of-Experts adapter for token-level fusion with a layer-wise skipping router for computational efficiency. Experiments on the Med-VTAB benchmark demonstrate improvements over prior adapter methods across medical imaging tasks, with particular emphasis on out-of-distribution and data-scarce scenarios.

## Strengths

- **Comprehensive evaluation scope**: The paper evaluates on 23 datasets across color medical images, X-rays, and OCT/CT/MRI modalities, providing broad coverage of medical imaging tasks. Performance improvements over strong baselines (GMoE-Adapter, VPT variants) are consistent across nearly all datasets.

- **Novel token-level fusion mechanism**: Unlike prior work that combines features at the layer or network level, DynaMer operates at the token level, enabling finer-grained integration of general and medical domain knowledge. The gating mechanism that balances original vs. adapter-processed tokens addresses training stability concerns.

- **Practical efficiency focus**: The layer-wise skipping router provides a configurable trade-off between inference time and accuracy (Table 7 shows inference time reduction from 0.165s to 0.086s with minimal accuracy change), which is relevant for deployment-constrained medical settings.

- **Thorough ablation coverage**: Tables 4-7 examine the contribution of gating mechanisms, gating dimensions, gating layers, and token-skipping ratios, providing insight into architectural design choices.

## Weaknesses

- **Misleading parameter efficiency claims**: The paper reports "Total Params" as ~1.21X relative to single-backbone baselines, but this appears to count only adapter parameters while **running two complete ViT-B/16 backbones** simultaneously. Since ViT-B has ~86M parameters, the true system uses roughly 2× the backbone parameters of single-backbone methods. The efficiency comparison is fundamentally incomplete without reporting total FLOPs or end-to-end inference cost accounting for both backbones.

- **Missing architectural details for reproducibility**: The MoE expert architecture `AdapE_j` is never specified. The paper states that experts "take two tokens from general and medical models, and output an integrated one" but does not describe whether this is implemented as concatenation followed by an MLP, cross-attention, or another mechanism. Additionally, critical hyperparameters (number of experts `n`, top-`k` value) are not reported in the main text.

- **Unexplained experimental anomaly in Table 7**: Processing 50% of tokens yields *higher* accuracy than processing 100% (e.g., HyperKvasir: 70.85 vs 70.82). This counterintuitive result—less computation producing better performance—warrants discussion. Possible explanations include regularization effects or overfitting at 100%, but the paper does not address this.

- **Naming inconsistencies suggesting editorial errors**: Table 9 labels the proposed method as "**GL-MoF Adapter**" instead of "DynaMer Adapter" used throughout the rest of the paper. Tables 1 and 2 use "MoE-Adapter" vs "MoF-Adapter" inconsistently for the same baseline method (Tong et al., 2024b). These inconsistencies undermine confidence in the experimental reporting.

- **Domain mismatch between medical backbone and evaluation tasks**: The medical ViT is pre-trained on cell images (Nguyen et al., 2023), yet evaluation includes X-ray, CT, and MRI datasets. The paper provides no ablation testing whether a domain-matched medical backbone would improve results, or whether the cell-image backbone provides any signal beyond noise for non-pathology modalities.

- **Marginal improvements without statistical validation**: Improvements over GMoE-Adapter are consistently small (often <0.5% absolute, e.g., 70.82 vs 70.75 on HyperKvasir). No confidence intervals, standard deviations, or significance tests are reported, making it difficult to assess whether these improvements are meaningful or within experimental noise.

- **Table 4 ablation presentation is unclear**: The table shows four rows with identical checkmarks for "General Gate" and "Medical Gate" but varying parameter counts (1.19X to 1.21X). What differs between these configurations is not explained, making the ablation uninterpretable.

## Nice-to-Haves

- **Simple ensemble baseline**: Compare against a naive ensemble of two single-backbone adapters to determine whether "dynamic merging" outperforms standard ensembling without additional architectural complexity.

- **Expert specialization analysis**: Visualize which MoE experts activate for which modalities (X-ray vs. pathology vs. MRI) to validate that the routing mechanism learns meaningful specialization.

- **Dual general-backbone ablation**: Test merging two general-domain ViTs (e.g., DINO v2 + CLIP) to isolate whether gains come from domain complementarity or simply from using two models.

- **Significance testing**: Report mean ± std over multiple seeds (at least 3-5 runs) to confirm improvements are statistically meaningful.

## Removed Points

*These points are flagged to be removed or treated with caution:*

- **Formatting nitpicks** (mentioned by reviewers but not substantive): Minor grammatical issues like "four four folds" do not affect technical contribution.

- **Claim about "stand on two shoulders" title being imprecise**: The title is metaphorical and appropriate; this is stylistic criticism.

- **Claim that contribution 4 (generalizability) is speculative**: The paper does demonstrate on FGVC and VTAB-1K in Table 10, providing empirical evidence beyond speculation.

- **Demand for "standard vs. full fine-tuning comparison"**: This asks the paper to address a different evaluation paradigm. The paper focuses on parameter-efficient adaptation; comparing against full fine-tuning would be a different scope.

## Novel Insights

The token-level merging approach represents a meaningful shift from feature-level or layer-level fusion in multi-model adaptation. The observation that 50% token processing outperforms 100% suggests potential regularization benefits from selective token processing—this could inform future work on when and how to prune token computation. The consistently better OOD performance (Tables 8-9) suggests that dual-domain knowledge genuinely helps generalization, though stronger evidence (e.g., statistical tests, broader OOD datasets) would strengthen this claim.

## Suggestions

1. **Report total system parameters and FLOPs**: Explicitly state the full parameter count and computational cost including both backbones. If dual-backbone overhead is acceptable in context, defend it clearly.

2. **Specify expert architecture**: Add a sentence describing `AdapE_j` (e.g., "Each expert is a 2-layer MLP with hidden dimension 256 that takes concatenated token pairs as input").

3. **Report hyperparameters in a dedicated table**: Include number of experts, top-`k`, learning rate, batch size, and training epochs for reproducibility.

4. **Explain or fix Table 4**: Clarify what varies between the rows, or restructure the ablation to show clearer variable manipulation.

5. **Fix naming inconsistencies**: Standardize method names across all tables (DynaMer vs GL-MoF, MoE-Adapter vs MoF-Adapter).

---

**Quality Assessment:**

- **Novelty**: The token-level MoE fusion mechanism for dual-domain adaptation is a reasonable architectural contribution, building on but extending prior MoE adapter work.

- **Technical Soundness**: Core methodology is sensible, but missing implementation details (expert architecture, hyperparameters) and unexplained anomalies (Table 7) raise reproducibility concerns.

- **Empirical Support**: Extensive evaluation across 23 datasets is commendable, but marginal improvements (<0.5%) without statistical testing weaken the evidence. The dual-backbone computational cost being omitted from efficiency claims is a significant transparency issue.

- **Significance**: If the improvements hold up under statistical scrutiny, the method could be valuable for medical imaging adaptation. However, the domain mismatch between the medical backbone (cell images) and evaluation tasks (X-ray, CT, MRI) limits claims about principled domain fusion.

- **Clarity**: The paper has notable issues including table naming inconsistencies, unclear ablation presentations, and missing architectural details that should be addressed.

MY FINAL SCORE: <pineapple>5.5</pineapple>

---

## 3lXZjsir0e

- GT: Reject (avg 5.6)
- Predicted: Accept (6.0/10)
- Match: NO

### Final Review

## Summary

This paper proposes RTZ-VI-LCB, a model-based algorithm for offline robust two-player zero-sum Markov games (RTZMGs) under partial data coverage. The authors introduce a novel "robust unilateral clipped concentrability coefficient" to characterize data quality without requiring full coverage, derive finite-sample complexity bounds that are near-optimal in state and action space dimensions, establish information-theoretic lower bounds for various uncertainty regimes, and extend the algorithm to multi-player general-sum games.

## Strengths

- **Novel algorithmic contribution for a challenging setting:** The paper addresses the intersection of offline RL, robustness, and multi-agent learning—a setting with limited prior theoretical treatment. The RTZ-VI-LCB algorithm properly integrates pessimism principles with robust value iteration and TV distance uncertainty sets (Section 3, Algorithm 2).

- **Weaker data coverage assumption:** The robust unilateral clipped concentrability coefficient $C_r^*$ is a genuine improvement over the maximum density ratio $C_r$ used in prior work like P²M²PO. As stated in Section 1.1, $C_r^* \in [\frac{1}{S(A+B)}, \infty)$ captures distribution shift without requiring proportional scaling when occupancy distributions exceed $\frac{1}{S(A+B)}$, enabling learning under partial coverage.

- **Near-optimal sample complexity for key parameters:** The derived sample complexity $\tilde{O}\left(\frac{C_r^* H^4 S(A+B)}{\varepsilon^2} f(\sigma^+, \sigma^-, H)\right)$ matches information-theoretic lower bounds (Theorem 2) with respect to state space $S$ and action spaces $\{A, B\}$—a first for offline RTZMGs (Table 1 and Theorem 1 discussion).

- **Meaningful lower bound analysis:** Theorem 2 establishes that learning RTZMGs is at least as hard as learning standard TZMGs when uncertainty is small ($\min\{\sigma^+,\sigma^-\} \lesssim 1/H$), and provides tighter bounds for larger uncertainty levels. This contextualizes the algorithm's difficulty relative to well-studied settings.

## Weaknesses

- **Inconsistency between Table 1 and main text regarding horizon exponent:** Table 1 states the lower bound for small uncertainty as $\Omega\left(\frac{C_r^* SH^3(A+B)}{\varepsilon^2}\right)$, while the discussion following Theorem 2 in Section 4 states $\Omega\left(\frac{C_r^* SH^4(A+B)}{\varepsilon^2}\right)$. This discrepancy directly affects the paper's optimality claims and must be resolved. The upper bound is $H^4$ in both places, so either the lower bound exponent is wrong in one location, or the claimed "optimality except for $H$" is inaccurate.

- **Confusing notation in Assumption 1 (Equation 22):** The definition uses $\sup_{(s, a, b, h, P) \in \Delta(A) \times \mathcal{S} \times \mathcal{A} \times \mathcal{B} \times [H] \times \mathcal{U}^{\sigma^-}(P^0)}$, which writes $\Delta(A)$ (a probability simplex) as an element in a tuple subscript—this is not standard mathematical notation. The intent appears to be taking a supremum over policies $\mu^-$, but this should be written explicitly for clarity.

- **Computational tractability gap for general-sum extension:** Theorem 3 claims the extension to multi-player general-sum games "break[s] the curse of multiagency," achieving sample complexity depending on $\sum_i A_i$ rather than $\prod_i A_i$. However, Section 3.2 briefly acknowledges that "solving these robust matrix games is generally PPAD-hard." The paper does not clarify whether this extension assumes a computational oracle, relies on a weaker equilibrium concept (e.g., CCE), or is purely information-theoretic. Without this clarification, the practical relevance of the general-sum result is unclear.

- **No empirical validation:** The paper is purely theoretical with no experiments demonstrating the algorithm's practical viability, convergence behavior, or sensitivity to uncertainty parameters. While not strictly required for theory papers at ICLR, empirical validation would strengthen the contribution—particularly given the novel concentrability coefficient and penalty term construction.

## Nice-to-Haves

- **Tighten the horizon dependency:** Investigate whether the $H^4$ upper bound can be reduced toward $H^3$ to fully close the gap with the lower bound. Even an explanation of why the current proof techniques yield $H^4$ would help readers understand the bottleneck.

- **Provide guidance on uncertainty set calibration:** The sample complexity depends on $f(\sigma^+, \sigma^-, H)$, but no discussion is offered on how practitioners should select $\sigma^+$ and $\sigma^-$ in practice or how sensitive performance is to these choices.

- **Add discussion on verifying concentrability:** The assumption on $C_r^*$ is theoretically useful but practically unverifiable from data alone. Discussing how one might estimate or bound this coefficient would enhance applicability.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Criticism of (s,a,b)-rectangularity as "strong assumption":** This structural condition is standard in robust RL literature (cited Iyengar, 2005) and enables the Bellman recursion. Requesting comparison with weaker conditions like s-rectangularity is scope creep beyond the paper's stated contribution.

- **Request for discussion of different divergence functions per player:** The paper already notes both players can use different divergence functions but restricts to the same one for the main results. This is a modeling choice within scope.

- **Minor notational inconsistencies (e.g., $C_n$ vs $C_h$):** These are proofreading issues that do not affect technical correctness.

- **Burn-in cost dependence on $d_m^n$ being "prohibitive":** This is standard in offline RL; the paper provides the standard $d_m^n$ condition alongside the concentrability assumption. Flagging this as a flaw misrepresents the offline RL literature.

## Novel Insights

The robust unilateral clipped concentrability coefficient $C_r^*$ represents a meaningful conceptual advance: by clipping the occupancy distribution at $\frac{1}{S(A+B)}$ before taking the ratio, the assumption becomes significantly weaker than requiring full coverage or proportional scaling. This insight—that partial coverage suffices when combined with pessimism—transfers the single-agent offline RL insight of Li et al. (2024a) to the multi-agent robust setting. The key innovation is adapting the clipped concentrability concept to account for adversarial perturbations in both transition dynamics and opponent policy, requiring consideration of worst-case occupancies under model perturbations. The derivation of the Bernstein-style penalty that properly accounts for the uncertainty set's non-linear transformation of the transition kernel is also noteworthy, though the proof intuition is deferred to the appendix.

## Suggestions

- **Resolve the H^3 vs H^4 discrepancy immediately:** Verify whether Table 1 or the main text is correct for the lower bound exponent, and update consistently. If both are correct under different conditions, explicitly state the condition boundaries.

- **Clarify Assumption 1 notation:** Rewrite Equation 22 to explicitly show the supremum over policies $\mu^-$ and $\nu^+$, matching how $d_h^{\mu^-, \nu^+, P}$ is used.

- **Add an explicit statement about computational assumptions in Theorem 3:** Either state that the general-sum result assumes an oracle for solving robust matrix games, or specify that CCE/CE is targeted instead of NE. Currently, readers cannot assess the practical meaning of the "curse of multiagency" claim.

- **Consider adding synthetic experiments:** Even simple matrix game experiments would demonstrate that the penalty term correctly handles uncertainty and that convergence follows the theoretical rate.

## Evaluation

**Novelty:** High. This is the first work to achieve near-optimal sample complexity for offline robust Markov games under partial coverage, with a novel concentrability coefficient adapted to the robust multi-agent setting.

**Technical soundness:** The core theoretical framework is sound, but the inconsistency between Table 1 and the main text regarding the horizon exponent is a significant oversight that affects optimality claims. The notation in Assumption 1 needs repair.

**Empirical support:** None provided. The paper is entirely theoretical.

**Significance:** Significant for the offline robust MARL literature. The theoretical tools (concentrability coefficient, robust Bellman analysis with TV distance, lower bound construction) are valuable for future work.

**Clarity:** Generally well-written, but the Assumption 1 notation and the H-exponent inconsistency impede full understanding of the key technical claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>

---

## F9JZiGradI

- GT: Reject (avg 5.2)
- Predicted: Reject (4.5/10)
- Match: YES

### Final Review

## Summary

The paper proposes MLP-KAN, a unified architecture combining Multi-Layer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs) within a Mixture-of-Experts framework embedded in a Transformer. The goal is to eliminate manual model selection between representation learning (MLP) and function learning (KAN) by dynamically routing inputs to appropriate experts. Experiments span computer vision (CIFAR-10/100, mini-ImageNet), NLP (SST-2), and symbolic regression (Feynman dataset).

## Strengths

- **Clear conceptual motivation**: The paper identifies a genuine tension between architectures optimized for representation learning versus function approximation, and proposes a principled integration strategy using Soft MoE routing to combine MLP and KAN experts within a Transformer backbone.
- **Comprehensive cross-domain evaluation**: The experiments cover three distinct task types—image classification, sentiment analysis, and symbolic regression—which demonstrates effort to validate the "universal" capability claim across domains requiring different inductive biases.
- **Function learning results show clear MLP advantage**: On the Feynman dataset (Table 2), MLP-KAN dramatically outperforms standalone MLP on most equations (often by orders of magnitude in RMSE), validating that KAN-based experts add substantial value for symbolic regression tasks.
- **Ablation studies included**: Tables 4 and 5 provide useful analysis of expert count and Top-K routing sensitivity, showing that 8 experts with Top-2 routing provides reasonable trade-offs.

## Weaknesses

- **Claims contradict experimental results in function learning**: The abstract states MLP-KAN "consistently outperforms" baselines, and Section 5.2 claims it "outperforms both KAN and MLP across a variety of equations." However, Table 2 shows KAN achieves a **better average RMSE (2.09×10⁻²) than MLP-KAN (2.58×10⁻²)**. Additionally, KAN wins on 13 of 30 individual equations. The paper overstates MLP-KAN's function learning performance. — This matters because the core claim of unified superiority is not supported by the data.

- **Text contains factual error about results**: Section 5.2 states "MLP-KAN outperforms both KAN and MLP with an RMSE of 7.18×10⁻²" for equation 1.15.3t, but Table 2 shows KAN achieves 3.69×10⁻², which is nearly half the error. The table correctly marks MLP-KAN as second-best (underlined), contradicting the text. — This matters for scientific accuracy and reader trust.

- **Representation learning shows no consistent advantage**: On 6 of 8 CV/NLP metrics in Table 3, MLP-KAN is **second-best behind standalone MLP**, with differences within or near one standard deviation. MLP-KAN wins only on SST-2 accuracy. — This matters because the unified model does not match specialist performance on the majority of representation tasks.

- **Missing critical ablation: heterogeneous vs. homogeneous MoE**: The paper does not compare MLP-KAN against an all-MLP MoE or all-KAN MoE with equivalent expert count. Without this comparison, we cannot determine whether any performance changes stem from expert heterogeneity versus simply having more capacity through the MoE structure. — This matters because it's the core technical claim that mixing expert types provides benefit.

- **No routing behavior analysis**: The paper claims dynamic adaptation between representation and function experts but provides no analysis of actual routing weights. Do CV tokens preferentially route to MLP experts? Do symbolic regression inputs route to KAN experts? Without this, the "dynamic adaptation" claim is unsupported. — This matters because it's central to the paper's motivation.

- **Equation 13 inconsistent with Soft MoE description**: The Transformer block output uses a uniform average (1/NE)∑Fₑ over all experts, while Section 4.1 describes the Soft MoE routing with learned dispatch weights α. The relationship between these is unclear. If outputs are uniformly averaged at the block level, the Soft MoE routing plays no role in the final output.

- **No computational efficiency comparison**: KAN operations (spline evaluations) are computationally heavier than MLP matrix multiplications, yet the paper provides no FLOPs, training time, or inference latency comparisons. The claim of "maintaining efficiency across diverse datasets" cannot be evaluated.

## Nice-to-Haves

- Comparison against standard MoE Transformers (e.g., Switch Transformer baseline with all-MLP experts) to isolate the contribution of KAN integration from MoE capacity benefits.
- Visualization of routing weights across task types to validate that the mechanism learns to separate representation and function inputs.
- Clarification of the relationship between Soft MoE routing and the uniform averaging in Equation 13.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Citation of MoE literature**: The harsh critic notes Soft MoE is not cited in related work. While this should be addressed, it's a fixable omission rather than a fundamental flaw.
- **Formula formatting in Table 2**: Some equation labels appear duplicated or incorrect (e.g., multiple rows showing the same formula). While this affects presentation, it's not central to evaluating the method's validity.
- **Comparison against state-of-the-art models**: Demanding comparison against ResNet, ViT, or BERT is scope creep. The paper's contribution is combining MLP and KAN experts; comparing against these baselines is appropriate for establishing that baseline.
- **Parameter count matching**: The harsh critic suggests MLP-KAN may have more parameters than standalone KAN. While relevant for fairness, parameter-matched comparisons are not standard practice in MoE papers where the goal is to leverage additional capacity.

## Novel Insights

The paper identifies a meaningful architectural insight: KANs' spline-based structure provides superior function approximation inductive biases for symbolic tasks, while MLPs' fixed activations may be preferable for high-dimensional representation learning. The MoE combination is a natural but non-obvious integration that, if successful, would simplify practitioner decisions about model selection. However, the empirical results suggest the integration introduces overhead without consistently improving over either specialist—particularly for representation learning where standalone MLP performs equivalently or better. The paper would be significantly stronger if it analyzed why the MoE routing doesn't preferentially select KAN experts for symbolic tasks (or whether it does), which would validate or invalidate the core premise.

## Suggestions

1. **Correct the claims**: Revise the abstract and Section 5.2 to accurately reflect that MLP-KAN performs comparably to (not better than) KAN on function learning averages, and does not consistently outperform MLP on representation tasks.
2. **Add routing analysis**: Include a heatmap or statistics showing expert selection distributions for different task types. This is essential to verify the mechanism works as claimed.
3. **Add homogeneous MoE baseline**: Train an all-MLP MoE with 8 experts to isolate whether benefits come from heterogeneity versus capacity.
4. **Report compute metrics**: Include training time, parameter counts, and FLOPs for fair comparison.

## Evaluation

**Novelty**: Moderate. Combining KAN and MLP experts via Soft MoE in Transformers is a reasonable architectural contribution, but builds directly on existing MoE frameworks without substantial algorithmic innovation.

**Technical Soundness**: Mixed. The Soft MoE formulation is correctly described, but Equation 13's uniform averaging is inconsistent with the routing mechanism. The empirical methodology is standard, but missing critical ablations.

**Empirical Support**: Weak. The main claims (consistent superiority in both domains) are not supported by the data. Function learning shows worse average RMSE than KAN; representation learning shows no consistent advantage over MLP.

**Significance**: Limited. Without demonstrating that expert heterogeneity provides benefit over homogeneous MoE, or that routing actually separates tasks, the practical utility is unclear.

**Clarity**: Generally clear but undermined by inaccurate textual claims that contradict the presented tables.

MY FINAL SCORE: <pineapple>4.5</pineapple>

---

