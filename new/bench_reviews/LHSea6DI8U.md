Now I have sufficient context from calibration papers. Let me write the final review.

## Summary

The paper proposes STBP, a framework for continual spatio-temporal forecasting that pairs a frozen general backbone (featuring a frequency-domain network FreNet and dual-stream linear graph attention DLGA) with a scalable contextual pattern bank that expands incrementally as new nodes arrive. The backbone captures stable spatio-temporal patterns after initial training and is then frozen, while the pattern bank adapts via parameter expansion and prompt-based gating. Experiments on three streaming datasets demonstrate improvements over existing CSTF baselines, particularly on traffic datasets.

## Strengths

- **Clear and principled architecture design**: The separation of stable general knowledge (frozen backbone) from scenario-specific adaptation (expanding pattern bank) is a logical and well-motivated design for the continual learning setting. The prompt-based interaction mechanism (Eq. 5, 7-9) between pattern bank and backbone is cleanly formulated and enables node-level heterogeneity modeling without modifying the backbone.

- **Architectural innovations with practical benefits**: The combination of FreNet (frequency-domain processing for stable temporal components) and DLGA (linear attention reducing O(N²) to O(N) complexity while incorporating pattern bank as additional key) is technically sound and well-suited for the graph expansion scenario. The efficiency study (Sec. 5.5) confirms that these design choices keep computational costs competitive.

- **Strong empirical performance on traffic datasets**: STBP achieves substantial margins over CSTF baselines on PEMS-Stream (~21% MAE reduction) and CA-Stream (~22% MAE reduction). The few-shot evaluation (Table 2) further demonstrates robustness under data scarcity, and the comprehensive ablation study isolates key component contributions.

- **Good experimental coverage**: Multiple datasets, metrics, forecasting horizons, ablation variants, few-shot settings, parameter sensitivity analysis, and efficiency comparisons provide a thorough evaluation.

## Weaknesses

### Major:

- **No direct measurement of catastrophic forgetting**: A core claim is that the pattern bank "mitigates catastrophic forgetting" (Abstract, Intro challenge ❸, Sec. 4.2). However, no standard continual learning forgetting metrics are reported. There is no evaluation of performance on earlier periods after training on later ones, no backward transfer metric, and no per-period performance breakdown. All main results report averages across all periods (Table 1), which cannot distinguish a method that retains old knowledge from one that simply performs well on recent periods. The paper would be substantially strengthened by per-period performance curves and forgetting scores. Without these, the central anti-forgetting claim remains plausible but empirically unsupported.

- **Evaluation conflates backbone capacity with continual learning strategy**: STBP uses a substantially more capable backbone (two FreNets + DLGA + FFN) compared to the simpler CSTF baselines (e.g., TrafficStream, STKEC use CNN+GCN). The ablation "w/o Backbone" (Sec. 5.3) replaces STBP's backbone with weaker CNN+GCN and shows degradation, but this only demonstrates that a stronger backbone helps—not that the continual learning strategy is superior. Missing is the symmetric control: pairing STBP's backbone with a simpler continual mechanism (e.g., EAC-style prompts or regularization-based fine-tuning) to isolate whether the gains come from the backbone architecture or the continual learning design. This makes it difficult to attribute the substantial performance improvements to the claimed bridging of STGNNs and CSTF.

- **Distributional drift claims are not substantiated**: The paper repeatedly asserts FreNet "mitigates distributional drift" by focusing on "stable low-frequency components" (Abstract, Sec. 4.3). However, there is no quantitative characterization of how distributions shift across periods, no comparison of FreNet against standard temporal modules (TCN/RNN/Transformer) within the same continual framework with the pattern bank, and no per-period degradation analysis showing that FreNet specifically prevents drift-related degradation. The ablation "w/o Backbone" replaces both FreNet and DLGA with CNN+GCN, conflating temporal and spatial design choices. The causal link between frequency-domain processing and drift robustness remains speculative.

### Minor:

- **Disproportionate improvement across datasets is unexplained**: STBP reduces average MAE by ~21% on traffic datasets but only 2.35% on AIR-Stream. This large gap raises questions about generalizability beyond traffic domains. The paper does not discuss whether the frequency-domain stability assumption holds less well for air quality data, or whether the pattern bank mechanism is less suited for meteorological dynamics. Even a brief discussion would strengthen the paper.

- **Pattern bank growth without bound**: The pattern bank P_τ grows linearly with cumulative node count via concatenation (Eq. 4). While the paper claims "scalable" design, there is no analysis of the parameter footprint over many incremental periods or discussion of when bank size could become problematic. The "Efficiency Study" (Sec. 5.5) mentions linear additional cost but provides no quantitative breakdown of pattern bank overhead as a fraction of total parameters or memory.

- **Pattern bank interpretation claims outrun evidence**: The t-SNE visualization (Fig. 6) shows clusters in P_τ, which is interpreted as encoding node heterogeneity and relevance. However, no ground-truth grouping (district labels, road types, sensor categories) is overlaid on the clusters, and no quantitative cluster quality metrics are provided. Clusters in learned high-dimensional embeddings are common and do not by themselves substantiate the semantic claims about encoding geography, policy, or events. Similarly, the claim that the bank "autonomously distinguishes heterogeneous and relevant nodes" (Sec. 4.2) is strong but only supported by qualitative visualization.

### Trivial:

- The term "general" backbone could be clarified: since the backbone requires the pattern bank P(0), P(1), P(2) for its forward pass, it is "general" in structure (node-count independent) but cannot operate independently. A brief clarifying note would help reader expectations.

## Nice-to-Haves

- **Per-period performance curves for all models and datasets**: Line plots of MAE at each incremental period would directly reveal whether STBP maintains stable performance over time while baselines degrade—validating the anti-forgetting claim.

- **Forgetting metric evaluation**: Standard continual learning metrics (backward transfer, forgetting measure) would quantitatively support the anti-forgetting claim.

- **Visualize learned frequency embedding F_τ**: Showing the frequency response of the learned embedding across periods would validate whether FreNet actually enforces stability by emphasizing low-frequency components.

- **Pair STBP's pattern bank with alternative modern backbones** (e.g., GWNet, STID as frozen backbones) to rigorously test the claimed portability of the pattern bank.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Baseline comparison fairness regarding parameter count"** (Human Finder, citing PromptST review): The harsh critic raises a valid variant of this about backbone capacity differences, which I've kept as a major weakness. However, the specific claim that comparing against baselines "without any pretraining" is unfair is not directly applicable—STBP's backbone is trained on data, not pretrained in a foundation model sense, and the CSTF baselines similarly train from scratch on their first period. The fairness concern is about architecture capacity, not pretraining status.

- **"Unmatched continual protocols for baselines regarding replay/data storage"** (Harsh Critic, point 2.2): The paper states CSTF baselines use their published protocols. While different methods have different data requirements (some use replay, some don't), this follows the standard practice of using each method's own designed protocol. This is not an unfair comparison—it reflects each method's design choices.

- **"Missing related works"** (Human Finder, SKI-CL review reference): Per hard rules, I do not include criticisms about missing related works.

- **"Formatting errors in Table 1"** (Neutral Reviewer): The paper text mentions parsing artifacts from PDF extraction. This is a formatting nitpick that should be removed per hard rules.

- **"Reproducibility concerns about FFT implementation details"** (Harsh Critic, Sec 4.3 notes): Questions about how FFT is taken, whether F_τ is per frequency/channel/global, how real/imag parts are handled—these are implementation details that can be found in the code repository. Per hard rules, minor reproducibility nitpicks are removed.

- **"Eq. 5 ambiguity about which submodule h_θ refers to"** (Harsh Critic): While this could be clearer, the code is available and the paper provides sufficient information for readers to understand the gating mechanism at a conceptual level. This falls under minor reproducibility/formatting concerns.

- **"Conventional STGNN baselines should use online fine-tuning with replay/regularization"** (Harsh Critor, point 2): The baseline GWNet/STID are tested with retraining-from-scratch following the prior work (Chen & Liang, 2025) protocol. Adding more baseline configurations would be a nice-to-have, not a flaw in the current evaluation.

- **"Individual component novelty is limited"** (Human Finder): While true that FreNet, linear attention, and prompt pools have precedents, the paper's contribution is their integration for a specific problem. This is a common pattern in systems papers and does not diminish the contribution of the overall design.

## Novel Insights

The paper implicitly demonstrates an important but underexplored trade-off in continual ST forecasting: the choice between backbone adaptability and backbone stability is not simply about "freezing vs. fine-tuning" but about whether the backbone's inductive biases (frequency-domain processing, graph-attention structure) are sufficiently general to project future unseen distributions into a useful representation space. The large performance gap between traffic and air quality datasets suggests that the generalization of these inductive biases is domain-dependent—a finding the paper does not reflect on but that has significant implications for the broader goal of building "foundation" spatio-temporal models.

## Suggestions

- Add per-period performance breakdowns (tables or line plots) showing MAE at each incremental period for STBP and key baselines. This is the single most impactful addition that would validate the core anti-forgetting claim.

- Include a forgetting metric (e.g., average accuracy drop on period 1 test data after training through all periods) to quantitatively support the claim about mitigating catastrophic forgetting.

- Discuss the AIR-Stream performance gap and articulate why the method's benefits are less pronounced for air quality data (e.g., different periodicity structure, less pronounced node heterogeneity).

- Add a controlled experiment isolating the contribution of FreNet to drift robustness: replace only FreNet with a standard temporal module (keeping DLGA and the pattern bank) within the same continual framework.

## Score and Decision

**Calibration reasoning**: I compared this paper against several papers with similar topics and quality patterns:

- **SKI-CL** (B1TnT6lUnU): Continual learning for time series forecasting, scores 3/5/3/3/8 (Reject). Had similar issues with conflated contributions and insufficient forgetting measurement, but weaker empirical results and less architectural novelty.

- **TFMoE** (vJGKYWC8j8): Continual traffic forecasting via MoE, scores 6/3/3 (Reject). Similar problem setting but limited datasets and baselines.

- **N-ForGOT** (rLlDt2FQvz): Continual learning on temporal graphs with explicit forgetting measurement, scores 6/6/5/8 (Accept Poster). Provides theoretical guarantees and explicit forgetting evaluation that this paper lacks, but has less architectural novelty.

- **FreDF** (4A9IdSa1ul): Frequency domain + forecasting, scores 6/8/6/8 (Accept Poster). Strong empirical results with clear, focused contribution.

- **PromptST** (YUNnVFlpjp): Prompt learning for ST prediction, scores 3/5/5/5 (Reject). Fairness of comparison issues similar to this paper.

STBP has stronger empirical results than most comparable rejected papers, a well-designed architecture, and addresses an important problem. However, the lack of direct forgetting measurement and the conflation of backbone capacity with continual learning strategy are significant empirical gaps that weaken the core claims. The paper's results are impressive but the evaluation doesn't adequately support the most ambitious claims about what each component contributes. This places it below accepted papers like N-ForGOT and FreDF that had more rigorous evaluation of their specific claims, but above rejected papers like SKI-CL and TFMoE that had both evaluation gaps and weaker results.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>