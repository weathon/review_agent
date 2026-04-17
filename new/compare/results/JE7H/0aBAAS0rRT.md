---
job_id: 302768aa-fb86-465c-8516-2c88310e4718
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0aBAAS0rRT.pdf
paper: Map as a Prompt: Learning Multi-Modal Spatial-Signal Foundation Models for Cross-Scenario Wireless Localization
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a self-supervised transformer with masking and prompt tuning for CSI-based localization, clearly within ICLR’s scope on representation learning, multimodal modeling, and foundation models.

## Minimum Quality
Pass ✅.  
The submission is in English and has all essential components: Abstract, Introduction, background/preliminaries, a clear Methodology (Section 3), Experiments and quantitative Results (Section 4 with multiple tables and figures), and Conclusion. The method is technically coherent and experiments are non-trivial, using standard datasets with comparisons to relevant baselines. No obvious fatal theoretical or experimental flaw is evident from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, AI-targeted instructions, or suspicious formatting in the main content.

---

# Expected Review Outcome:

## Summary

The paper introduces **SigMap**, a transformer-based “wireless localization foundation model” trained on CSI using a **cycle-adaptive masked modeling** objective and then adapted to downstream localization tasks by **map-conditioned prompt tuning**. The masking strategy exploits cross-correlation across antennas/subcarriers to disrupt periodic patterns in CSI, while a GNN-based module converts 3D map geometry and base-station positions into a soft prompt token injected into the frozen transformer during fine-tuning. Experiments on DeepMIMO (single-BS and multi-BS), generalization to DeepMIMO O2 and WAIR-D, and ablations on masking and map quality show strong performance and good parameter efficiency compared with several supervised and self-supervised baselines.

---

## Strengths

1. **Clear, well-motivated integration of physics and representation learning.**  
   The paper starts from the physical MIMO-OFDM model in **Equation (1)** and the ray-tracing mapping in **Equation (2)**, then uses these to justify why CSI encodes geometric constraints and how map information can help disambiguate NLoS paths. This grounding in channel physics helps justify both the periodicity-aware masking and the graph-based prompt design rather than presenting them as arbitrary architectural tweaks.

2. **Cycle-adaptive masking is a thoughtful response to a CSI-specific shortcut.**  
   Section 3.3 and Appendix B.4 provide a reasonably detailed formulation of the masking. The high-level rule in **Equation (6)** and the more complete algorithm in **Equations (13)–(21)** describe how cross-correlation–derived shifts determine diagonal mask bands across the antenna–subcarrier plane. **Figure 3** and **Figure 7** visually support the claim that CSI often exhibits stripe-like periodic structure: the “CSI Amplitude Heatmaps for Different Shift Patterns” make it apparent that naive random or purely grid masking would be easy to interpolate. The empirical gain in **Table 3** (MAE 0.770 → 0.673, CDF@1m 80.3% → 84.5%) reinforces that this is not just cosmetic: the masking choices measurably affect downstream localization.

3. **“Map-as-prompt” mechanism is clean and parameter-efficient.**  
   The geographic prompt tuning pipeline in **Section 3.4**, **Algorithm 1**, and **Figure 4** is conceptually neat: Delaunay triangulation over building vertices and BS positions creates a graph, two GCN layers aggregate geometry, and a simple projection produces a prompt token prepended to the transformer sequence. Importantly, only the GNN, projection MLP, and small task heads are updated during fine-tuning, keeping the number of trainable parameters at **0.7%** of the total (see **Table 5**). This is a reasonable and clearly explained adaptation of prompt tuning ideas to a signal–geometry setting.

4. **Strong empirical results with informative tables and figures.**  
   - **Table 1** and **Table 2** show that SIGMAP (w/ map) improves MAE substantially over strong baselines (e.g., single-BS MAE from 2.382 m for LWLM to 1.564 m; 4-BS MAE from 0.828 m to 0.673 m), with a large CDF@1m gain (25.3% → 60.5% in the single-BS NLoS regime). These numbers directly support the claim that the proposed combination of pre-training and map prompts is beneficial for localization.  
   - **Figure 5** summarizes many metrics in a radar plot; visually, SIGMAP (w/ map) consistently occupies the outermost region across axes (SingleBS, MultiBS, NLoS, etc.), confirming that improvements are not narrowly confined to a specific configuration.  
   - **Figures 8 and 9** (CDF curves, Appendix B.5) show the effect of SIGMAP shifting the entire error distribution leftwards, not just improving mean metrics, which is relevant for reliability claims.
   
5. **Non-trivial cross-scenario generalization and few-shot adaptation.**  
   The experiments in **Section 4.5** transfer from DeepMIMO O1_3p5 to DeepMIMO O2 and the WAIR-D real-city dataset with only limited fine-tuning (approximately 100 labeled samples per scenario), freezing the backbone. The results table on **Page 10** shows that SIGMAP (with map) nearly halves MAE relative to LWLM in both domains (e.g., 2.213 → 1.026 m on DeepMIMO O2 and 3.375 → 1.880 m on WAIR-D), suggesting that the backbone representations are meaningfully reusable, not overly tied to a single map.

6. **Ablation on map fidelity gives useful insight into what “map-as-prompt” is doing.**  
   **Table 4** compares 3D mesh, 2D birdview, and no-map variants for single-BS localization. The modest gap between 3D and 2D (1.564 m vs. 1.692 m MAE) compared to the much larger gap to no-map (2.275 m) indicates that most gains are from topological / LoS-geometry cues rather than full 3D detail. This is an interesting empirical insight and helps demystify the role of the geographic prompt.

7. **Exposition is mostly clear and accessible despite technical domain.**  
   The narrative from CSI physics (Section 2) to architecture (Figure 2), masking (Figure 3), and prompt generation (Figure 4) flows logically. Definitions of inputs and objectives, such as **Equation (3)**, the MAE pre-training loss in **Equation (7)**, and the multi-BS fusion in **Equations (9)–(10)**, are self-contained and readable for a non-wireless specialist.

---

## Weaknesses

1. **Limited and slightly inconsistent mathematical specification of the masking strategy.**  
   While the paper claims a key contribution in “cycle-adaptive masking”, the formalism is fragmented and occasionally inconsistent. In Section 3.3, **Equation (6)** defines a simple band mask depending on a pre-computed periodicity shift \(d_{\text{final}}\), but it does not specify how \(d_{\text{final}}\), \(j_0\), or \(w\) are derived from the cross-correlation analysis discussed earlier. Appendix B.4 introduces a more elaborate shift-mask generation via **Equations (13)–(21)** with parameters \(d, N_a, N_s, T, F\), but the notational link to Equation (6) is unclear: is \(d_{\text{final}} = d\)? Are the “bandwidth” \(bw = |d|\) and “half-bandwidth” \(hw\) exactly the mask width \(w\) used in Equation (6)? Also, the main text only hints at cross-correlation, but does not write down the exact statistic, e.g.,  
   \[
   \rho(\Delta) = \frac{\sum_k H_i[k] H_{i+1}[k+\Delta]^*}{\sqrt{\sum_k |H_i[k]|^2 \sum_k |H_{i+1}[k+\Delta]|^2}},
   \]
   or explain how the argmax over \(\Delta\) yields \(d_{\text{final}}\). For a core algorithmic contribution, this level of under-specification makes reproducing the exact masking scheme difficult and weakens the claim that it leverages signal periodicity in a principled way rather than heuristic tuning.

2. **Geographic prompt design is somewhat ad hoc and lacks strong ablations on graph choices.**  
   The geographic prompt pipeline in **Figure 4** relies on a very specific set of design choices: (i) building vertices and BS positions as separate node types; (ii) Delaunay triangulation for edges; (iii) exactly two GCN layers and global mean pooling; (iv) a single prompt token. While this is reasonable, the paper provides almost no analysis of why these choices are preferable to simple alternatives. For example, there is no comparison to:  
   - encoding only BS positions via an MLP and concatenating to the CLS token;  
   - using k-nearest-neighbor graphs instead of Delaunay;  
   - multiple prompt tokens (per BS or per region) vs a single global token.  
   The only map ablation in **Table 4** focuses on 3D vs 2D vs no-map, which answers “is map info useful?” but not “is this specific GCN-prompt architecture necessary?”. Without such analysis, it is hard to credit the map-as-prompt design as more than a plausible, but not particularly optimized, instantiation.

3. **Experimental scope is predominantly simulated; real data usage is indirect.**  
   Most of the main quantitative results (**Tables 1–4** and **Table 3**) are on DeepMIMO ray-traced data. The WAIR-D results on **Page 10** are still produced using ray-tracing over OpenStreetMap-derived city layouts, not over measured CSI, so the method’s robustness to real RF imperfections (hardware non-idealities, calibration drift, coarse synchronization, etc.) is not evaluated. Given the strong claims about cross-scenario robustness and “practical deployability,” it is problematic that there is no validation on a purely measured dataset (even a small one). This gap matters because both the masking strategy (exploiting neat periodic structure) and the map prompts (from high-quality 3D meshes) may rely on properties that degrade substantially in real deployments.

4. **Limited diversity and somewhat unclear implementation of baselines.**  
   The baselines in **Tables 1 and 2** are OMP, CNN, SWiT, and LWLM. While LWLM is an appropriate foundation-model baseline, the paper does not clearly state whether the baselines are all trained under the same data regime (e.g., using only the 2% labeled subset for fine-tuning vs full supervision) nor whether they receive any map information. The text on **Page 8** mentions an “NLoS-aware attention mechanism” using **Equation (11)** for SIGMAP, but it is unclear if comparable architectural sophistication (e.g., similar attention to multipath) was added to baseline models or whether they are fairly strong implementations. This leaves some ambiguity about whether the improvements are due to the new pre-training and map prompts, or simply more careful modeling of NLoS effects in the proposed method.

5. **Some architectural and training details are under-specified or slightly contradictory.**  
   - In Section 3.2, CSI is represented as \(\mathbf{X}=[\Re(\mathcal{H}), \Im(\mathcal{H})]\) (**Equation (5)**), while Appendix B.2 rewrites the input as magnitude and phase \(\overline{\boldsymbol{H}}_s = [|\boldsymbol{H}_s|, \angle \boldsymbol{H}_s]\) (**Equation (12)**). The paper then states it will reuse \(H_s\) “for notational consistency” but does not clarify which representation is actually used in the backbone and in pre-training. This is not a trivial detail because the loss in **Equation (7)** reconstructs \(\mathbf{X}\), and magnitude/phase vs real/imag have different training dynamics and phase unwrapping issues.  
   - The description of the backbone in Appendix B.2 references both “encoder-decoder framework” and ViT-like masking, but the main text (**Figure 2**) mainly shows a masked encoder with separate heads, leaving the exact architecture slightly ambiguous.  
   These inconsistencies make it harder to reimplement the model and may hide subtle design choices critical for performance.

6. **Positioning w.r.t. other multimodal wireless foundation models is incomplete.**  
   The related work overview touches on LWM, WirelessGPT, LWLM, and several localization-focused SSL methods, but omits some directly related recent work on *multimodal* wireless foundation models that also mix wireless channels with other modalities. In particular, there is no mention of models that integrate wireless with images or other environment descriptions for localization and sensing, which are conceptually very similar to the “map-as-prompt” idea (see next section for specific citations). This omission makes it harder to judge how much of the multimodal integration is new versus a domain-specific variant of existing wireless multimodal FMs.

7. **Over-claiming “foundation model” status and scope relative to actual experiments.**  
   The paper repeatedly refers to SigMap as a “foundation model” and suggests that it underpins many wireless tasks, mentioning beamforming and signal processing in the conclusion. However, all experiments are on localization from CSI, with one dataset for pre-training and several related datasets for testing, and no evidence is provided that the learned representations transfer effectively to non-localization tasks. While the cross-scenario generalization in **Section 4.5** is commendable, calling the model a general-purpose “spatial-signal foundation model” is somewhat overstated compared with the presented empirical scope.

---

## Potentially Missing Related Work

1. **A. Aboulfotouh and H. Abou-Zeid, “Multimodal Wireless Foundation Models,” 2025.**  
   This work reportedly introduces a multimodal wireless foundation model that can process both raw IQ streams and image-like representations across multiple tasks. It is directly relevant to the paper’s positioning as a “multimodal spatial-signal foundation model,” especially in the way it combines modalities and claims general-purpose usability. It should be cited and discussed in the Introduction and Related Work discussion (around Sections 1–2), and if it includes localization benchmarks, compared in Section 4.

2. **M. Farzanullah, H. Zhang, A. B. Sediq, “Wireless Multimodal Foundation Model (WMFM): Integrating Vision and Communication Modalities for 6G ISAC Systems,” 2025.**  
   This work integrates wireless channel coefficients with visual imagery for user localization and classification in an ISAC context, which is very close in spirit to using maps as prompts to condition channel-based inference. It should be discussed as prior art for multimodal wireless–vision integration, likely in Section 1.1 (Research Gaps) and Section 3.4 (Geographic Prompt Tuning), to better contextualize the novelty of using a GCN-derived prompt token vs other fusion mechanisms. If WMFM reports localization performance, including it as a baseline or at least in a qualitative comparison would strengthen the empirical positioning.

---

## Questions

1. **Clarification of the exact masking algorithm and hyperparameters.**  
   - How exactly is the periodicity shift \(d_{\text{final}}\) in **Equation (6)** computed from cross-correlation? Please provide the formula, the search range for the shift, and whether it is computed per-antenna pair, per-subcarrier, or globally per CSI sample.  
   - What are the ranges or typical values for \(w\), \(N_a\), and \(N_s\) in **Equations (13)–(21)**, and how sensitive are results in **Table 3** to these values?  
   A precise algorithmic description (pseudo-code or math) would increase confidence that the observed performance gains are not fragile to these choices.

2. **Representation choice: real/imag vs magnitude/phase.**  
   There is a discrepancy between **Equation (5)** and **Equation (12)** regarding how CSI is represented. Which representation is actually used in the backbone during pre-training and fine-tuning? If both are used in different experiments, how do their performances compare? If only one is used, please align the notation and clarify whether phase wrapping or normalization strategies were necessary.

3. **Fairness and training regime of baselines.**  
   For LWLM, SWiT, and CNN baselines in **Tables 1–3**, were they pre-trained and fine-tuned with the same data splits and label budget as SIGMAP (e.g., 2% user subsampling for fine-tuning)? Did any baseline receive map information, and if not, could they in principle do so (e.g., by concatenating map features to the input)? A clear fairness statement would help interpret the strength of the reported performance gains.

4. **Ablation on graph construction and prompt design.**  
   Could you provide experiments comparing (a) Delaunay vs k-NN edge construction, and (b) GCN-based prompt vs a simple MLP over a global feature (e.g., averaged building vertices and BS positions), especially for single-BS NLoS in **Table 1**? This would help justify the more complex GNN prompt as opposed to simpler conditioning.

5. **Robustness to map errors and missing information.**  
   Real-world maps (e.g., from OpenStreetMap) can be noisy or incomplete. Have you tried perturbing building positions, removing some buildings, or adding random obstacles to the 3D mesh when generating prompts? How does performance in **Table 4** or on WAIR-D change under such perturbations? Such a study would strengthen the claim that the approach is robust and practically deployable when perfect 3D maps are not available.

6. **Potential for non-localization tasks.**  
   Since the paper positions SigMap as a “foundation model,” can you provide at least preliminary evidence (even small-scale) that the pre-trained backbone helps with another channel-related task, such as beam selection or channel extrapolation? Even qualitative or single-baseline results would make the foundation-model claim more convincing.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The modeling choices are physically motivated and the core algorithms (masking, GNN prompt, attention-based multi-BS fusion) are coherent and produce strong empirical results, but some algorithmic details (especially masking and representation choices) are under-specified, and experiments are limited to (high-quality) simulated data.

---

## Presentation Rating

3: good.  
The paper is generally well-written with clear figures (e.g., **Figures 1–4, 5, 8, 9**) and tables, and it explains the physics background and methodological ideas clearly. However, some notational inconsistencies (real/imag vs magnitude/phase, masking equations) and missing ablations on key design choices reduce clarity and reproducibility.

---

## Contribution Rating

3: good.  
The paper makes a meaningful and timely contribution to representation learning for wireless localization, combining a CSI-specific masking strategy with map-conditioned prompt tuning and demonstrating substantial gains over existing deep and foundation-model baselines. The “foundation model” scope is somewhat overstated, and empirical validation is limited to localization tasks on simulated data, but within that scope the contribution is solid.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper offers a well-motivated and empirically strong approach to CSI-based localization, with two reasonably interesting ideas (cycle-adaptive masking and map-as-prompt GNN conditioning) that are supported by quantitative gains across several scenarios and ablations (especially **Tables 1–4** and **Figure 5**). At the same time, the masking algorithm is not fully specified, the geographic prompt design lacks thorough ablations, and all primary results rely on simulated data. These issues prevent a higher recommendation, but given the demonstrated improvements and conceptual clarity, I lean toward acceptance.

---

## Reviewer Confidence

4: confident.  
I am familiar with representation learning and, to a lesser extent, wireless localization and CSI modeling. I have carefully checked the math at the level presented and the experimental tables/figures, though I did not re-derive every detail of the masking algorithm or GCN prompting.