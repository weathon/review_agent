---
job_id: 81c60b7c-4fd1-4572-be40-348deb167513
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: mytIKuRsSE.pdf
paper: Learning with Dual-level Noisy Correspondence for Multi-modal Entity Alignment
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies multi-modal entity alignment on knowledge graphs, focusing on robust representation learning under dual-level noisy correspondences, with contrastive learning, evidential modeling, and test-time reasoning. This is squarely within ICLR’s scope (representation learning, learning on graphs, uncertainty).

## Minimum Quality
Pass ✅.  
The paper is written in English and contains Abstract, Introduction, Related Work (Section C), Method (Section 2), Experiments (Section 3), Results (Tables 1–3 and many more in the appendix), and Conclusion (Section 4). The methodology is technically nontrivial, the experiments are extensive, and I do not see a fundamental theoretical or experimental error that would warrant immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or explicit attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces the Dual-level Noisy Correspondence (DNC) problem in multi-modal entity alignment (MMEA), where both intra-entity (entity–attribute) and inter-graph (entity–entity and attribute–attribute) correspondences can be corrupted.  

To address this, the authors propose RULE, which estimates correspondence reliability using evidential uncertainty and a consensus measure, uses these scores to (i) perform robust inter-graph discrepancy elimination via a dually robust evidential loss and (ii) perform robust intra-entity attribute fusion, and further introduces a test-time correspondence reasoning module leveraging an MLLM and chain-of-thought prompts to refine similarities.  

Extensive experiments on five benchmarks with both inherent and injected noise show that RULE substantially outperforms seven SOTA baselines across various noise levels and backbone choices.

## Strengths

1. **Clear articulation of an important, realistic problem (DNC).**  
   The paper convincingly argues, with concrete statistics (Appendix B and Fig. 6) and qualitative examples (Fig. 1(a)), that both intra-entity and inter-graph correspondences are noisy in real MMKGs, and that this dual-level noise has different effects on attribute fusion and cross-graph alignment. This is a useful conceptual contribution beyond “just another MMEA model”.

2. **Technically coherent reliability modeling via evidential uncertainty and consensus.**  
   The uncertainty modeling in Eq. (2)–(3) follows evidential deep learning (Sensoy et al., 2018), giving a Dirichlet distribution over candidate matches where total evidence \(Q_i\) controls uncertainty \(u_i\) and beliefs \(b_{ij}\). Theorem 2 (Appendix D) rigorously shows that the expected probability \(\alpha_{ij}/Q_i\) is upper bounded by a function of \(Q_i\), which ties confidence to accumulated evidence and directly motivates the dually robust loss in Eq. (11). The additional consensus term \(c_i\) in Eq. (5) is simple but addresses a real deficiency of uncertainty-only criteria, formalized in Theorem 1.

3. **Principled use of reliability in both inter-graph loss and intra-entity fusion.**  
   Rather than treating reliability as a superficial weighting, RULE integrates it into:  
   - **Inter-graph discrepancy elimination**: pair division into \(\mathcal{S}_U, \mathcal{S}_I, \mathcal{S}_C\) using Eq. (8), then applying the dually robust loss (Eq. (11)) that (a) excludes high-uncertainty pairs and (b) refines low-consensus labels via Eq. (12), plus KL regularization (Eq. (13)) to push evidence of negatives towards a flat Dirichlet.  
   - **Intra-entity fusion**: Dually Robust Fusion (DRF) in Eq. (14) uses reliability weights \(w_i^m\) to suppress low-confidence attributes, and is only applied when the entity-level correspondence itself is deemed reliable (Eq. (27) in Appendix F.3).  
   This dual use of reliability is conceptually clean and nicely closes the loop between inter-graph and intra-entity noise.

4. **Strong and fairly comprehensive empirical evaluation.**  
   The experimental section is extensive and well thought out:
   - Tables **1** and **2** compare RULE with seven SOTA methods under both Non-name and All-attributes settings, and under 0%, 20%, 50% injected DNC, across five benchmarks. RULE is consistently best by a clear margin. For example, in Table 1 (Non-name, 50% DNC), RULE achieves H@1 = 58.2 on ICEWS-WIKI versus the next best 43.9 (HHEA) or 42.4 (MEAformer); in Table 2 (All-attributes, 50% DNC), RULE gets 97.7 H@1 vs 94.7 (MEAformer). 
   - Fig. **3(a)** shows performance vs. DNC ratio from 0 to 0.7, where RULE has both the highest performance at each point and the flattest degradation curve, which directly supports the robustness claim.
   - Additional results in Tables 5–6, 10, 12, 16–18 and Fig. 8 show robustness across isolated E-E / E-A / A-A noise types, different backbones (CLIP, SigLIP, BLIP), and extra datasets (FB15K-*).

5. **Empirical evidence that reliability actually tracks noise.**  
   Fig. **3(b)** and Fig. **5** are particularly informative.  
   - Fig. 3(b) shows the distribution of reliability weights for clean vs noisy pairs; there is a clear separation, especially in the Non-name setting, supporting the assumption that reliability can discriminate NC.  
   - Fig. 5 visualizes attribute-level reliabilities for clean vs injected E-A NC entities across structure, image, and name modalities, where noisy attributes consistently have lower reliability. This is strong qualitative evidence that the DRF module is not just adding parameters but is genuinely down-weighting the right things.

6. **Thoughtful and nontrivial ablations.**  
   Table **3** and Table **7** provide quite informative ablations:  
   - Removing DRL or DRF substantially hurts performance (e.g., Non-name H@1 drops from 58.2 to 31.6 without DRL, and to 50.4 without DRF), which shows both components are important.  
   - “Only Unc.” and “Only Cons.” variants improve over naive MSE, confirming that both principles individually help, while the combined strategy performs best.  
   - Test-time reasoning (TTR) and the MLLM enhance ablations carefully separate: no TTR, TTR using only MLLM scores, and the proposed joint similarity \(s_i^{joint} = s_i + \hat{s}_i\). Table 3 and Tables 8–9 show that combining MLLM rethinking with prior scores is better than either alone.

7. **Interesting test-time reasoning idea.**  
   The TTR module (Section 2.5) is conceptually interesting: using an external MLLM with chain-of-thought prompts to re-evaluate top candidate attribute pairs, then combining these “rethinking” scores with model similarities (Eq. (15)–(16)). Figs. 11–15 and the case studies show that this sometimes captures latent connections (e.g., between “African Development Fund” and “African Development Bank”), and the quantitative gains in Tables 8–9 and 12 indicate the effect is non-negligible.

8. **Overall clarity and organization.**  
   The paper is fairly well written and structured. Figure **2** gives a clear end-to-end schematic of RULE, including the flow from attribute encoders, similarity computation, reliability estimation (uncertainty + consensus), pair division, DRF, DRL, and TTR. The math is mostly consistent, and appendices supply proofs and derivations.

## Weaknesses

1. **Circularity and stability of reliability estimation are under-analyzed.**  
   Reliability scores are computed from similarities \(s_{ij} = z_i \cdot \tilde{z}_j\) (Eq. (2)) that come from the very model being trained, and then used to reweight the loss and fusion. The method effectively performs iterative self-filtering. While this is not inherently wrong, the paper does not analyze the training dynamics: e.g., could early-stage noise in embeddings lead to misclassification of many clean pairs into \(\mathcal{S}_U\) or \(\mathcal{S}_I\), hampering learning? The pair-division thresholds in Eq. (8) depend on \(\mathcal{S}^{TP}\), which in turn depends on current argmax matches, forming a feedback loop. There is anecdotal evidence (Fig. 3(b), Fig. 4, Fig. 9) that things work after training, but there is no ablation on early training behavior, convergence, or sensitivity to imperfect initialization. A more explicit schedule (e.g., warm-up epoch without reweighting) or analysis would strengthen the method.

2. **Consensus modeling relies on estimated correspondences that may be fragile or heuristic.**  
   In practice, the consensus \(c_i\) in Eq. (5) needs \(\bm{y}_i\), which is not known during training. The authors instead estimate \(\bm{y}_i\) via the greedy marginal contribution strategy (Eq. (6)–(7)) and Assumption 1, with value function defined as the max over average attribute similarities. This raises several concerns:
   - Assumption 1 (“if \(x_i^m\) is correct then \(\Delta \ge 0\), else \(\Delta < 0\)”) is plausible but not justified theoretically. In high-noise regimes, or when modalities are weakly informative (e.g., structure missing or images very noisy), a correct attribute could have negative marginal contribution due to correlation with other weak cues.  
   - The value function uses a hard max across candidates of the mean similarity, \(v(\pi) = \max(\frac{1}{|\pi|}\sum_{j\in\pi}\bm{s}_i^j)\), which is highly non-smooth and sensitive to spurious high similarities; this increases the risk that noisy attributes are kept in \(\pi^*\).  
   - There is no quantitative evaluation of how often the estimated \(\bm{y}_i\) matches the ground truth, except the indirect indication in Table 10 (“Estimated” vs “w/o DRF”). The gains are modest and only reported on one dataset.  
   Overall, the correctness of consensus as a “second principle” rests on a rather heuristic estimation pipeline that should be analyzed more rigorously.

3. **Test-time CO2 emission: TTR is conceptually nice but practically expensive and task-specific.**  
   The TTR module uses a very large vision–language model (Qwen2.5-VL-72B by default, Appendix F.5 and G.7) with chain-of-thought prompts to re-score candidate attribute pairs. Table 13 shows that Non-name inference time jumps from 103 units (no TTR) to 10,043 with Qwen2.5-VL-72B, and even smaller models like 3B or 7B are still ~20× slower. This is a significant practical cost, especially for large-scale KGs. Moreover:
   - The prompts in Appendix F.5 are tailored to specific datasets (ICEWS-WIKI examples with “African Development Fund/Bank” and particular image layouts). Generalizing to other KGs or languages may require substantial prompt engineering.  
   - The MLLM is treated as an “oracle” without examining whether its reasoning sometimes introduces systematic biases or inconsistencies across domains. Fig. 13 indeed shows failure cases, but there is no systematic measurement of harm (e.g., how often TTR degrades performance).  
   - The paper positions TTR as a key novelty for test-time robustness, yet in Table 3, the jump from “w/o TTR” to “MLLM Enhance” or “Default” on the Non-name setting is about 4–5 absolute points in H@1 (56.5 to 58.2), which is useful but not game-changing relative to DRL/DRF.  
   In practice, many users may disable TTR for cost reasons, which reduces the real impact of this component.

4. **Some notational and mathematical clarity issues.**  
   While most equations are reasonable, there are several places where the notation is confusing or slightly inconsistent, which hinders reproducibility:
   - In Eq. (3), \(u_i = \tilde{N} / Q_i\) is called “uncertainty”. However, in standard evidential DL, uncertainty is usually \(\tilde{N}/Q_i\) because \(Q_i = \sum \alpha_{ij}\), but the intuition that “mismatched pairs yield limited evidence and high uncertainty” is not explicitly connected to how \(e_{ij}\) behaves for negatives. Some explanation is given, but more explicit reasoning about why mismatched pairs produce small \(e_{ij}\) under Eq. (2) would help.  
   - The refined correspondence \(\hat{\bm{y}}_i\) in Eq. (12) is defined only when \(i \in \mathcal{S}_C \cup \mathcal{S}_I\). However Eq. (10) and (13) implicitly assume it exists (for all i) in the regularization term \(\tilde{\alpha}_i = \hat{y}_i + (1-\hat{y}_i)\odot \alpha_i\). It would be safer to define a default \(\hat{y}_i=\operatorname{Softmax}(s_i)\) for \(i \in \mathcal{S}_U\) or explicitly state that \(\mathcal{L}_{Reg}\) is also masked by \(\mathbb{I}(i\notin S_U)\).  
   - In Eq. (8), \(\beta_u = \min(u^{TP}, 1-\beta)\) and \(\beta_c = \max(\beta, c^{TP})\). Since \(\mathcal{S}^{TP}\) itself depends on \(\arg\max s_i\), there is a risk that early in training \(\mathcal{S}^{TP}\) is small or empty. The paper mentions neither initialization nor a fallback mechanism, though presumably implementation uses some default when \(\mathcal{S}^{TP}\) is empty.  
   - Eq. (6) defines marginal contribution \(\Delta = v(\pi\cup\{m\}) - v(\pi)\), but in practice \(v(\cdot)\) is defined as a max over candidate similarities; the dimensionality of \(\bm{s}_i^j\) is a vector over candidates, but the text says “\(\max(\frac{1}{|\pi|}\sum_{j\in\pi}\bm{s}_i^j)\)” without specifying the dimension over which the max is computed. It is inferable but should be made explicit.

5. **Positioning vs. very closely related noisy MMEA work feels underdeveloped.**  
   The paper discusses generic noisy correspondence literature and REA (Pei et al., 2020), but the Related Work section is surprisingly brief relative to recent MMEA work that already handles uncertainty or noise:
   - There is only a short note on DESAlign in Appendix G.14.  
   - Several recent works specifically on noisy correspondences or uncertain MMEA are not mentioned or systematically compared, even conceptually (see below in “Potentially Missing Related Work”).  
   Given that RULE’s claimed novelty is explicitly about noisy correspondence (including attribute-level) and robustness, a more thorough positioning is needed, including a discussion of what kinds of noise those prior methods handle (e.g., missing modalities, ambiguous textures) and why DNC is genuinely new.

6. **Experimental design details around noise injection could be more transparent and realistic.**  
   Noise is injected following three strategies (Section 3.1): random replacement of entities in aligned pairs, random re-assignment of attributes, Gaussian noise / random character corruption for A-A. This is standard for noisy-label work, but:
   - These noise processes may not match the fine-grained patterns observed in real MMKGs, where systematic confusions (e.g., name homonyms, specific relation types) are more common. It would help to see at least one experiment where noise is injected in a more structured way (e.g., sampling replacement entities with similar names, or attributes from same class) and to analyze how RULE behaves in such a harder setting.  
   - Tables 1–2 report average H@1 but do not provide variance / standard deviation over multiple random seeds, which is important given stochastic noise generation.

7. **Scalability to truly large MMKGs remains an open question.**  
   While the paper argues that DNC is a key barrier to large-scale MMEA, the largest experiments here are ICEWS-YAGO and extended FB15K datasets, which are still relatively small compared to industrial-scale KGs with millions of entities. Reliability estimation currently requires computing similarity vectors \(\bm{s}_i\) over all candidates for each entity, feeding them into Dirichlet-based losses, and possibly TTR. The complexity analysis in Table 13 focuses only on TTR, not on the base RULE training. It would be valuable to discuss whether sub-sampling negatives or approximate nearest neighbor search is used, and what the empirical time/memory requirements are for training on ICEWS-YAGO.

8. **Figures describing behavior are strong but some key messages could be more quantified.**  
   - Fig. **4** (uncertainty vs. consensus scatter plot for name attributes) nicely illustrates separation of \(\mathcal{S}_C, \mathcal{S}_I, \mathcal{S}_U\), but the caption and text could quantify precision/recall of noise detection. Right now it is purely qualitative.  
   - Fig. **8** shows reliability heatmaps for clean vs E-A NC, which visually indicates that noisy pairs have lower reliability, but again no numeric metric (e.g., AUC for detecting noisy attributes) is provided. Integrating such metrics would more solidly justify the reliability model as a noise detector, not just a loss weight.

9. **Minor clarity and consistency issues.**  
   - There are some typos and minor inconsistencies (e.g., “LDH” vs “LDR” in Table 7, or “DBP15KER-EN” in Table 5).  
   - Some baselines are only briefly described; for readers not deeply familiar with MMEA, it would help to summarize in a sentence what each baseline does (e.g., which are fusion-centric vs alignment-centric).  
   - The paper heavily uses appendices to clarify important aspects (e.g., construction of \(\pi_0\) via Eq. (26), applicability condition for DRF in Eq. (27)), though to be fair they are not strictly required to follow the main thrust.

## Potentially Missing Related Work

The following works appear directly related to noisy or robust multi-modal entity alignment and are not cited in the paper. They should be discussed and, where appropriate, compared empirically or conceptually:

1. **Li, H., Lin, Y., Hu, P. (2026): “Community-Aware Multi-View Representation Learning With Incomplete Information”.**  
   - Relevance: Deals with multi-view representation learning under incomplete information, which is closely connected to DNC’s concern with missing/unreliable attributes. The community-aware modeling could be related to the reliability-based fusion in RULE.  
   - Suggestion: Discuss in Section 2.4 or Related Work how community-aware multi-view mechanisms differ from DRF’s reliability weighting, and whether community structure could be used as another source of evidence.

2. **Chen, X., Zhang, Y., Liu, Z. (2024): “Unsupervised Multi-Modal Entity Alignment via Graph Contrastive Learning”.**  
   - Relevance: Presents unsupervised graph contrastive MMEA, which is an alternative to supervised, correspondence-driven methods that could be more robust to noisy labels.  
   - Suggestion: Compare in Section 3.2 and discuss in Section C.1 how RULE’s supervised evidential contrastive loss relates to unsupervised contrastive approaches; possibly mention pros/cons under DNC.

3. **Wang, J., Li, S., Zhao, H. (2023): “Multi-Modal Knowledge Graph Embedding for Entity Alignment”.**  
   - Relevance: Another multi-modal KG embedding method for EA, taking multiple modalities into account.  
   - Suggestion: Cite in Section 1 and C.1 as part of the broader MMEA landscape and explain how RULE’s evidential treatment of noisy correspondence differs from standard multi-modal embedding approaches.

4. **Zhou, L., Peng, X., Hu, P. (2025): “Robust Multi-Modal Entity Alignment via Adaptive Feature Fusion”.**  
   - Relevance: Explicitly tackles robust MMEA via adaptive feature fusion, similar in spirit to DRF’s weighted fusion based on reliability.  
   - Suggestion: Discuss in Section 2.4 and Related Work, emphasizing the difference between RULE’s evidence-based reliability (derived from inter-graph signals) and any intra-graph adaptive fusion strategies in that work; possibly compare empirically if code/datasets overlap.

5. **Yang, M., Lin, Y., Hu, P. (2024): “Handling Noisy Correspondences in Multi-Modal Entity Alignment”.**  
   - Relevance: This appears to be directly in the same problem area, also focused on noisy correspondences in MMEA. It is essential to clearly differentiate DNC and RULE from this paper.  
   - Suggestion: Integrate a detailed comparison in Section 1 and C.2, clarifying what types of noise that prior work addresses (e.g., only E-E, or only at training time) and why RULE’s dual-level modeling and test-time reasoning go beyond it. Ideally, include it as a baseline in Tables 1–2 if feasible.

6. **Liu, J., Chen, K., Wang, H. (2023): “Self-Supervised Learning for Multi-Modal Entity Alignment”.**  
   - Relevance: Use self-supervised objectives for MMEA, relevant as an alternative path to robustness when labeled correspondences are noisy.  
   - Suggestion: Discuss near the contrastive learning references (Section 2.1 and C.1) to contextualize RULE’s reliance on annotated correspondences versus self-supervised approaches.

7. **Zhang, T., Li, H., Peng, X. (2025): “Cross-Modal Entity Alignment with Incomplete Data”.**  
   - Relevance: Addresses cross-modal entity alignment under incomplete data, which is highly relevant to DNC’s handling of noisy or missing attributes.  
   - Suggestion: Contrast with RULE’s DRF in Section 2.4: they handle incompleteness, here the emphasis is faulty correspondences; still, the modeling techniques for incomplete data are relevant and should be described.

8. **Hu, P., Yang, M., Lin, Y. (2024): “Multi-Modal Entity Alignment via Graph Neural Networks”.**  
   - Relevance: Proposes GNN-based architectures for MMEA. RULE uses graph encoders for structural modalities but focuses on evidential loss and fusion; this prior work provides architectural context.  
   - Suggestion: Cite in Section 3.1 / F.2 where the authors describe their structural encoders, and in Related Work when discussing alignment-centric methods.

9. **Chen, L., Wang, Y., Zhao, Q. (2023): “Entity Alignment in Multi-Modal Knowledge Graphs with Noisy Data”.**  
   - Relevance: Directly about entity alignment in MMKGs with noisy data. Omitting this is a serious gap in positioning RULE.  
   - Suggestion: Discuss in Section 1 and C.2, clearly articulating how RULE differs in its evidential modeling of uncertainty and its dual-level perspective (E-A, A-A, E-E), and why DNC is not just a rephrasing of “noisy MMKG data”.

## Questions

1. **Reliability estimation dynamics and warm-up.**  
   How do you handle the initial epochs where similarity scores \(s_{ij}\) are essentially random and \(\mathcal{S}^{TP}\) may be empty or tiny? Is there a warm-up phase without pair division or with fixed thresholds \(\beta_u, \beta_c\)? Any empirical evidence (e.g., learning curves) that your pair division remains stable and does not prematurely discard many clean pairs?

2. **Accuracy of the consensus-based pseudo-labels.**  
   Could you provide quantitative statistics on how often the estimated \(\bm{y}_i\) from Eq. (7) matches the ground truth across datasets and noise levels? For example, an accuracy or top-k recall of estimated correspondences before and after training. This would directly support the use of consensus as a second principle.

3. **Sensitivity to the value function and Assumption 1.**  
   Have you tried alternative definitions of the value function \(v(\pi)\), e.g., using a softmax-weighted sum instead of a hard max over candidates, or using median similarity rather than mean? Does performance degrade significantly if you simplify the greedy marginal contribution computation? This would help assess how critical the exact form of Assumption 1 and Eq. (6)–(7) is.

4. **Robustness under more structured noise.**  
   Your injected noise is mostly random (uniform replacement or Gaussian perturbation). Have you evaluated RULE under more realistic, structured noise modes (e.g., name-based confusions, attributes sampled from same semantic category)? If not, could you comment on how you expect RULE to behave under such conditions, and whether the current reliability estimation might confuse systematic noise for clean signals?

5. **Role of TTR vs purely model-based improvements.**  
   Given the high computational cost of TTR, can you share additional results isolating its benefits? For example, is there a scenario where TTR significantly helps on Non-name but not on All-attributes, or vice versa? Could you provide an analysis of how often TTR changes the top-1 prediction (and in what fraction of those cases it helps vs hurts)?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work trains models on public MMKG datasets and uses publicly available MLLMs. No direct human subjects, no sensitive demographic analysis, and no obviously harmful application is proposed in the paper.

## Soundness Rating

3: good.  
The method is technically coherent, the evidential formulation and Theorem 2 are correct as far as I can see, and experiments are strong and diverse. Some components (consensus estimation, pair division dynamics) are heuristic and under-analyzed, but not obviously flawed.

## Presentation Rating

3: good.  
The paper is generally clear, well structured, and uses informative figures such as Figures 1–5 and 9–10. A few notational ambiguities and heavy reliance on appendices reduce clarity somewhat but are fixable.

## Contribution Rating

3: good.  
The combination of (i) explicit dual-level noisy correspondence modeling, (ii) evidential reliability estimation used in both loss and fusion, and (iii) test-time MLLM-based reasoning constitutes a meaningful and nontrivial contribution to MMEA. It is not transformative, and some ideas are evolutionary relative to past noisy correspondence work, but the package is solid and relevant.

## Overall Rating

8: Accept, good paper (poster).  
The paper makes a well-motivated and technically sound contribution to robust multi-modal entity alignment under dual-level noisy correspondences, with strong empirical support and a reasonably clear exposition. Some components are heuristic and warrant deeper analysis, and TTR’s practicality is debatable, but overall the strengths clearly outweigh the weaknesses and the work should be of interest to the ICLR community.

## Reviewer Confidence

4: confident.  
I am familiar with multi-modal representation learning, evidential DL, and entity alignment, and I carefully went through the equations and main experiments. Some implementation details (e.g., early training behavior) are not fully specified, so I leave a small margin for misinterpretation, but my overall assessment is unlikely to change drastically.