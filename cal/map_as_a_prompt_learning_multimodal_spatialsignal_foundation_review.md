=== CALIBRATION EXAMPLE 51 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the core contributions: using a map as a prompt and building a foundation model for wireless localization. The abstract concisely states the problem, the two key innovations (cycle-adaptive masking and map-as-prompt), and the main results (SOTA performance and zero-shot generalization). The claims are strong and supported by the experimental results later in the paper.

### Introduction & Motivation
The introduction effectively situates the work within the evolution from model-based to data-driven and foundation model approaches for wireless localization. It clearly identifies two research gaps: 1) inadequate handling of signal periodicity in SSL, and 2) superficial integration of geographic information. The three contributions are stated clearly and map directly to these gaps. The motivation is strong and well-argued.

### Preliminaries (Section 2)
This section provides necessary background on wireless channel modeling and problem formulation. The mathematical model (Eq. 1) is standard and correctly links CSI to geometry. The inclusion of the 3D map \(M\) in the problem definition (Eq. 3) is a key differentiator. One minor point: the ray-tracing abstraction (Eq. 2) is mentioned but its use in generating training data is only clarified later in the appendix; a brief sentence connecting it to the dataset (DeepMIMO) here would improve flow.

### Methodology (Section 3)
This is the core of the paper. The two-stage framework (pre-train + prompt-based fine-tuning) is clearly illustrated in Figure 2.
*   **Cycle-Adaptive Masked Modeling:** The intuition is excellent—preventing models from exploiting periodic shortcuts in CSI. However, the description in the main text is somewhat high-level. Equation (6) and the surrounding text are not fully self-contained. Critical details on how the periodicity shift \(d_{\text{final}}\) is detected are deferred to Appendix B.4. For a methods section, a more complete algorithmic description or a reference to a clear, numbered algorithm in the main text would enhance reproducibility. The claim that this forces learning of "globally meaningful representations" is plausible but not directly verified; an analysis (e.g., of learned attention patterns) would strengthen this claim.
*   **Geographic Prompt Tuning:** This is a creative and well-formulated idea. Algorithm 1 and the accompanying GCN formulation are clear. However, a significant concern arises: the graph is built from the union of **all building mesh vertices** and base station positions. For a large 3D map, the number of vertices \(|V|\) could be enormous (thousands to millions). The computational and memory cost of constructing and processing this full graph via Delaunay triangulation and a GCN is not discussed. Do the authors subsample vertices? If not, this could be a major practical bottleneck. Furthermore, the rationale for using Delaunay triangulation over simpler geometric relationships (e.g., k-NN based on distance) is not justified. The parameter efficiency claim is supported by the low number of fine-tuned parameters.
*   **Task-Specific Adaptation:** The attention-based fusion for multi-BS scenarios (Eq. 9, 10) is sensible. However, it is not explicitly stated how the model receives multi-BS input. Does it process CSI from each BS separately and then fuse the [CLS] tokens? Or is there a unified input sequence? Figure 2 suggests a unified sequence, but the text does not specify the tokenization for multiple BSs.

### Experiments & Results (Section 4)
The experimental design is comprehensive, addressing the key questions of performance, ablation, and generalization.
*   **Main Results (Tables 1, 2):** SIGMAP with map prompts shows impressive gains, especially in the challenging single-BS NLoS setting. The inclusion of a strong recent baseline (LWLM) is good. However, **Equation (11) appears abruptly** in the middle of the results narrative. This equation describes an "NLoS-aware attention mechanism" that was not introduced in the Methodology section. Is this a component of SIGMAP? If so, it must be described in Section 3. If it's an analysis tool, it should be clearly marked as such. Its sudden appearance is confusing and breaks the flow.
*   **Ablation Studies:** Tables 3 and 4 effectively validate the core components (masking strategy and map prompts). The finding that 2D maps retain most of the benefit is interesting and practical.
*   **Generalization (Sec. 4.5):** This is a strong experiment demonstrating zero/few-shot cross-domain adaptation. However, a **critical clarification is needed**: When fine-tuning on the unseen scenarios (O2 and WAIR-D), do the baselines (LWLM, SIGMAP w/o map) also use the same 100-instance fine-tuning set and the same frozen backbone? The text states "only the downstream task heads are fine-tuned... while the self-supervised backbone remains frozen." It must be explicitly confirmed that this *same protocol* (fine-tuning only the head on 100 samples) was applied to **all methods** for a fair comparison. If LWLM was trained from scratch or with a different data budget, the comparison is invalid.
*   **Parameter Efficiency (Table 5):** The numbers support the claim of efficient fine-tuning. It would be helpful to state the total parameter count of the pre-trained model to contextualize the 0.085M fine-tuned parameters (11.73M total is mentioned later, so ~0.7% is fine).

### Writing & Clarity
The paper is generally well-written. The major clarity issue is the **unexplained Equation (11)** in the results section, which needs integration into the method description. The figures are referenced appropriately and support the text. The appendices are detailed and aid reproducibility.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Important limitations to acknowledge include:
1.  **Dependence on Simulated Data:** All experiments use ray-traced data (DeepMIMO, WAIR-D). Performance on real-world, noisy CSI data is unknown.
2.  **Scalability of Geographic Prompts:** The computational cost of building and processing the full vertex graph for large maps is a potential practical limitation that should be discussed.
3.  **Map Availability:** The method assumes access to a detailed 3D/2D map, which may not be available in all deployment scenarios. The 2D ablation mitigates this, but it's still a prerequisite.
4.  **Broader Impact:** A brief statement on potential positive (enabling new applications) and negative (privacy risks from precise tracking) societal impacts should be included.

## Overall Assessment

This paper presents a novel and compelling foundation model approach for wireless localization. The core ideas—cycle-adaptive masking to defeat periodic shortcuts and map-conditioned prompt tuning for efficient cross-scenario adaptation—are innovative, well-motivated, and demonstrate strong empirical results. The experiments are thorough and generally support the claims, showing significant gains over relevant baselines. However, the paper has notable weaknesses that must be addressed: 1) a key component (NLoS-aware attention, Eq. 11) is not properly integrated into the methodology, 2) the fairness of the generalization experiment comparisons needs explicit verification, and 3) the computational scalability of the geographic prompt generation is not discussed. Furthermore, the absence of a limitations section is a notable omission for ICLR. If these issues are adequately clarified and addressed in a revision, the paper's significant contribution would stand and make it a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SigMap, a multimodal foundation model for wireless localization. Its core contributions are a cycle-adaptive masking strategy that disrupts periodic shortcuts in Channel State Information (CSI) during self-supervised pre-training, and a "map-as-prompt" framework that injects 3D geographic information via lightweight graph neural network prompts for efficient cross-scenario fine-tuning. The model demonstrates strong performance in both single and multi-base station localization tasks and shows promising zero-shot generalization to unseen environments.

### Strengths
1. **Novel and Well-Motivated Technical Contributions**: The cycle-adaptive masking strategy is a direct, insightful response to a clearly identified gap (exploiting CSI periodicity as a shortcut). The method of computing row-wise cross-correlation to generate shift-aware mask patterns (Sec. 3.3, Eq. 6) is a concrete, physics-aware innovation. Similarly, the geographic prompt tuning mechanism (Sec. 3.4, Alg. 1) offers an elegant, parameter-efficient method for fusing environmental semantics.
2. **Comprehensive and Rigorous Evaluation**: The experiments are extensive, answering multiple key research questions. The paper benchmarks against strong, recent baselines (e.g., LWLM, SWiT) across varied tasks (single/multi-BS, NLoS), uses standard metrics (MAE, RMSE, CDF@1m), and includes critical ablation studies (masking strategies, map modalities). The zero-shot/few-shot generalization tests on completely unseen datasets (DeepMIMO O2, WAIR-D) are particularly compelling for a foundation model claim (Sec. 4.5).
3. **Strong Empirical Results**: The performance gains are substantial and consistently demonstrated. For instance, SigMap with a map reduces MAE by 34.4% over the best baseline (LWLM) in challenging single-BS NLoS localization and achieves high parameter efficiency (fine-tuning only 0.7% of parameters) (Tables 1, 2, 5).

### Weaknesses
1. **Limited Real-World Validation and Practical Deployment Considerations**: All experiments are conducted on ray-traced datasets (DeepMIMO, WAIR-D). While valuable, this does not fully capture the complexities of real-world signal noise, hardware imperfections, dynamic environments, or the challenge of acquiring accurate, aligned 3D maps in practice. The reliance on pre-computed ray-tracing for map-channel alignment (Eq. 2) may not be feasible in many operational settings.
2. **Incomplete Analysis of Computational Overhead and Comparative Efficiency**: Although parameter efficiency during fine-tuning is highlighted (0.085M trainable params), the computational cost of the two-stage pre-training/fine-tuning pipeline is not directly compared to end-to-end supervised or other self-supervised baselines. The pre-training itself requires 36 hours on 6x A800 GPUs (Table 5), which is significant. A more thorough cost-benefit analysis for foundation model adoption is needed.
3. **Ambiguity in Map Dependency and Prompt Interpretation**: The ablation shows that 2D maps provide most of the benefit (Table 4), suggesting the model may rely more on topological (e.g., LoS/NLoS) cues than detailed 3D geometry. This raises questions about the necessity of complex 3D map processing and the specific spatial relationships the GNN prompts actually learn. A deeper analysis of what the prompts encode would strengthen the interpretability claim.

### Novelty & Significance
**Novelty:** The work is highly novel within the wireless localization domain. The cycle-adaptive masking is a fresh take on SSL for structured signal data, moving beyond generic image/audio-inspired masking. The "map-as-prompt" concept is a creative and timely adaptation of prompt-tuning paradigms from NLP/Vision to a spatially grounded multimodal problem. The integration of a GNN to generate environmental prompts from a 3D scene graph is also innovative.
**Significance:** The work is significant for the field. It pushes the state-of-the-art in data-driven localization, particularly for cross-scenario generalization—a major pain point. The proposed foundation model framework, if validated in real-world settings, could substantially reduce data labeling burdens and enable more robust systems for critical 5G/6G applications like autonomous driving and XR, aligning well with ICLR's focus on impactful machine learning advances.

### Suggestions for Improvement
1. **Include Real-World or Hardware-in-the-Loop Validation**: To address the primary weakness, the authors should validate the model on at least one dataset containing measured CSI from a real testbed or public dataset (even if limited in scale). An analysis of performance degradation under realistic noise, hardware calibration errors, and imperfect map data would greatly strengthen the claims of practical robustness.
2. **Provide a More Detailed Efficiency and Complexity Analysis**: Expand Table 5 to include a direct comparison of total computational cost (FLOPs, training time) and memory footprint against key baselines (e.g., LWLM, SWiT) for achieving similar accuracy. This is crucial for assessing the practical value of the foundation model approach beyond just final accuracy metrics.
3. **Deepen the Analysis of the Geographic Prompts**: Conduct experiments to probe the learned representations in the prompt tokens. For example, visualize attention maps between the prompt and CSI tokens to see which environmental features (e.g., specific buildings, corridors) the model attends to. Alternatively, perform a sensitivity analysis by perturbing parts of the input map (e.g., removing a building) to quantify its impact on the prompt and final accuracy. This would solidify the claim of "interpretable fusion."

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validation on real-world measured CSI data.** The paper relies entirely on simulated/ray-traced datasets (DeepMIMO, WAIR-D). Without demonstrating performance on even one real-world measured dataset (e.g., from a public testbed), the claims of robust generalization to "complex real-world environments" and practical applicability are severely undermined. The simulation-to-real gap is a major hurdle in wireless learning, and its absence is a critical omission.
2. **Ablation of the geographic prompt mechanism against simpler map integration baselines.** The paper compares "w/ map" vs. "w/o map" but does not ablate its specific GNN-based prompt against simpler alternatives (e.g., concatenating map features, using a non-graph MLP, or using a hard-coded geometric filter). This leaves the contribution of the novel "map-as-prompt" framework unsubstantiated, as the gains could stem from any map inclusion rather than the proposed method.
3. **Comparison to state-of-the-art foundation models and adaptation methods for wireless.** The baselines (OMP, CNN, SWiT, LWLM) are insufficient. Missing are comparisons to recent large wireless models (e.g., WirelessGPT, LWM, CrowdBERT) and to standard parameter-efficient fine-tuning (PEFT) techniques like LoRA or adapters. Without this, the claimed superiority in "parameter-efficient generalization" and SOTA performance is not convincingly established.
4. **Sensitivity analysis to imperfect or partial map information.** The method's core premise is integrating 3D map data. Experiments are needed where the map is incomplete, noisy, or misaligned (common in practice) to show robustness. The current ablation (3D vs. 2D vs. none) is too idealized and does not test the method's practicality.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what the cycle-adaptive masking strategy actually learns.** The paper claims it prevents "periodic shortcuts" and learns "globally meaningful representations," but provides zero evidence. A frequency or correlation analysis of the learned features compared to those from baseline masking strategies is needed to validate this core claim. Without it, the innovation's mechanism is speculative.
2. **Interpretability of the map prompt's influence on model decisions.** It is unclear *how* the geographic prompt guides localization. A case study analyzing attention weights between the prompt token and CSI tokens, or visualizing how the model's error changes relative to specific map features (e.g., walls, corners), is necessary to substantiate the "interpretable fusion" claim.
3. **Detailed error analysis to identify failure modes.** The paper reports aggregate metrics but does not dissect *where* and *why* the model fails (e.g., specific NLoS geometries, distances from BS, areas with high multipath density). This analysis is critical for assessing the method's true robustness and for understanding its limitations, which is expected for a foundational contribution.

### Visualizations & Case Studies
1. **Visualization of predicted locations overlaid on the map for challenging NLoS cases.** Side-by-side plots comparing predictions of SIGMAP (w/ map) and the best baseline for several representative, difficult user positions would directly illustrate the claimed benefit of map integration and expose when it fails.
2. **Visualization of the learned shift patterns and masking strategy.** While Figure 3 is referenced, a clear visualization comparing the input CSI, the detected periodicity, and the applied adaptive mask for a few examples is needed to make the novel masking strategy concrete and understandable.

### Obvious Next Steps
1. **Test on real-world data.** This is not a "future work" item; it is a necessary validation step for a paper making claims about practical wireless localization and should have been included.
2. **Perform a rigorous ablation study on the prompt architecture.** The paper must deconstruct the GNN-based prompt to show which components (graph construction, aggregation, projection) are essential, rather than presenting it as a monolithic block.
3. **Evaluate cross-domain generalization with varying system parameters.** The zero-shot test uses new geographic scenes but similar ray-tracing parameters. Testing generalization to different carrier frequencies, bandwidths, or antenna array geometries would better demonstrate the model's foundation capabilities.

# Final Consolidated Review
## Summary
This paper proposes SigMap, a multimodal foundation model for wireless localization. Its core innovations are a cycle-adaptive masking strategy that prevents shortcut learning on periodic CSI patterns during self-supervised pre-training, and a "map-as-prompt" framework that uses a GNN to generate geographic prompts from 3D maps for efficient, interpretable fine-tuning.

## Strengths
- **Novel, physics-informed pre-training objective:** The cycle-adaptive masking strategy is a direct and well-motivated solution to a known challenge (periodic shortcuts in CSI), moving beyond generic masking used in vision or NLP.
- **Effective and parameter-efficient cross-scenario adaptation:** The geographic prompt-tuning mechanism enables strong zero/few-shot generalization to unseen environments (e.g., 53.2% MAE improvement on DeepMIMO O2) while fine-tuning only 0.7% of parameters, demonstrating a practical foundation model paradigm.

## Weaknesses
- **Incomplete methodological description:** A key component—the "NLoS-aware attention mechanism" described by Equation (11) in the results—is not introduced or explained in the Methodology section (Section 3). This omission makes the method incomplete and hinders reproducibility.
- **Lack of validation on real-world data:** All experiments are conducted on simulated, ray-traced datasets (DeepMIMO, WAIR-D). For a paper targeting practical wireless localization, the absence of any evaluation on measured CSI data is a significant limitation that weakens claims of robustness and real-world applicability.
- **Unclear experimental fairness in generalization tests:** While the paper states a few-shot fine-tuning protocol for SigMap on unseen scenarios, it does not explicitly confirm that the same protocol (fine-tuning only the task head on 100 samples) was applied to all baselines (LWLM, SIGMAP w/o map). This ambiguity calls the fairness of the reported generalization comparisons into question.

## Nice-to-Haves
- A deeper analysis of what the geographic prompts learn (e.g., via attention visualization) to substantiate the claim of "interpretable fusion."
- An ablation comparing the proposed GNN-based prompt against simpler map-feature integration baselines to isolate the contribution of the novel prompt architecture.
- A discussion of the computational scalability of building a graph from all building mesh vertices for large maps.

## Novel Insights
The paper's core insight is that treating geographic information as a learnable "prompt" for a pre-trained wireless encoder is a highly effective and parameter-efficient strategy for cross-scenario adaptation. Furthermore, it identifies and directly addresses a previously overlooked pitfall in applying masked autoencoding to wireless signals: the risk of models exploiting local periodicities as shortcuts, which is mitigated by the proposed cycle-adaptive masking.

## Suggestions
- Integrate a description of the NLoS-aware attention mechanism (currently Eq. 11) into the Methodology section (Section 3) to provide a complete technical description.
- Clarify the fine-tuning protocol used for all methods in the generalization experiments (Section 4.5) to ensure a fair comparison is being presented.
- Acknowledge the limitation of using only simulated data and discuss plans or challenges for real-world validation in a revised manuscript.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0]
Average score: 5.3
Binary outcome: Accept
