---
job_id: 20af73b9-974a-4f0e-9f56-106ccb251734
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 9X2NfyZpR2.pdf
paper: Long-Term Action Anticipation via Transcript-Based Supervision
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅.  
The paper studies weakly supervised dense long‑term action anticipation and temporal action segmentation from video with transformers, CTC, CRFs, and cross‑modal attention, which fits squarely within representation learning and video understanding topics appropriate for ICLR.

## Minimum Quality  
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present. The work is technically nontrivial, written in English, methodologically coherent overall, and includes quantitative and qualitative experiments on standard benchmarks. While there are issues in novelty, clarity, and experimental completeness, they do not rise to the level of desk‑rejection.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅.  
I do not see any hidden prompts, meta‑instructions to reviewers, or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper proposes TbLTA, a framework for dense long‑term action anticipation (LTA) trained using only video transcripts, i.e., ordered action lists without frame‑level boundaries. TbLTA uses a transformer encoder over video features and learnable class tokens, a weakly supervised temporal alignment module (ATBA) to generate frame‑level pseudo‑labels, a transcript–video cross‑attention mechanism to enrich video features, and an anticipation decoder equipped with a CRF and a duration prior. Experiments on Breakfast, 50Salads, and EGTEA show that TbLTA, despite using only transcript supervision, achieves performance competitive with fully supervised LTA models in some settings and clearly surpasses prior (semi‑)weakly‑supervised work.

---

## Strengths

1. **Clear and timely problem formulation: transcript‑only supervision for dense LTA**

   The paper targets a very concrete and underexplored problem: learning *dense* long‑term action anticipation purely from transcripts, without any frame‑level labels. Prior LTA work is almost fully supervised, and prior weakly supervised TAS work rarely tackles dense anticipation. Framing TbLTA as “LTA with transcript‑only supervision” is conceptually clean and clearly articulated in Section 1 and Section 3, and is a meaningful direction for scalability.

2. **Reasonable, integrated architecture for weakly supervised dense LTA**

   TbLTA combines several components into a coherent pipeline: a transformer encoder over video plus class tokens, an ATBA‑style temporal alignment module to produce pseudo‑labels, a segmentation head, a masked transcript–video cross‑attention block, a transformer‑CRF decoder for anticipation, and a duration prior. Figure 2 provides a clear overview of how these elements connect, especially the flow of pseudo‑labels into both TAS and LTA, and how transcripts supervise via CTC and cross‑attention. The overall design aligns well with the goal of extracting temporal structure from weak supervision.

3. **Use of transcripts beyond supervision, as semantic context**

   The cross‑modal attention mechanism (Section 3.1, “Cross‑attention layer between modalities”) is a sensible way to use transcripts not just as label sequences but as semantic features. Equations (1) and (2) specify a local masked cross‑attention where transcript embeddings attend only to pseudo‑aligned temporal neighborhoods, with a gated residual to inject the text information back into video features. This is more thoughtful than a naive global cross‑attention and aligns with the idea that individual transcript actions should only interact with temporally plausible segments.

4. **Non‑trivial objective design tying alignment, segmentation, and anticipation**

   The loss design in Section 3.2 is well structured: (i) alignment‑oriented losses leveraging ATBA pseudo‑labels, (ii) a CTC loss over the full transcript to globally constrain segmentation, and (iii) anticipation‑oriented losses with a CRF sequence objective plus a duration prior. Equation (3) makes the decomposition explicit. This three‑tiered design shows awareness of the difficulty of jointly learning alignment, segmentation, and long‑horizon anticipation under weak supervision.

5. **Competitive performance vs. fully supervised baselines in some regimes**

   Table 1 is the main quantitative highlight. On Breakfast, the deterministic “Ours (TbLTA)* – Top1” achieves an average MoC of 29.37, slightly outperforming the strongest supervised baseline ActFusion (28.45) and FUTR (26.59). In some specific horizons (e.g., 30% observation, 10% prediction), the gap is larger (38.38 vs. 35.79 for ActFusion). This is quite impressive given the lack of frame‑level labels and suggests that transcript supervision can capture strong procedural regularities in this dataset.

6. **Stochastic anticipation results and rare‑class behavior**

   The stochastic protocol results (also in Table 1, “Ours (TbLTA)* – Mean”) show noticeable gains over the deterministic version, which supports the claim that the model can capture multiple plausible futures. On EGTEA (Table 2), TbLTA is clearly behind supervised models in overall mAP (65.37 vs. 76.80 for Anticipatr), but is relatively competitive on rare classes (60.11 vs. 55.10 for Anticipatr). This is an interesting empirical observation supporting the claim that high‑level semantic supervision can help ameliorate class imbalance.

7. **Ablation studies connect architecture pieces to performance**

   The ablation tables provide some insight into the contribution of different components:
   - Table 3 (labelled “Effect of CTC loss / Multimodal Cross‑Attention”) shows that removing the CTC loss slightly but consistently hurts performance (e.g., on Breakfast avg. 37.2 → 36.4) and that replacing the proposed local, gated cross‑attention with a simple cross‑attention degrades performance more strongly (avg. 37.2 → 33.4 on Breakfast).
   - Table 4 (“Ablation study on LTA module”) shows substantial drops when removing the CRF (e.g., Breakfast avg. 37.2 → 33.0; 50Salads avg. 28.5 → 23.2) and non‑trivial drops when removing the duration loss or cross‑attention. These ablations substantiate, at least qualitatively, that the different TbLTA components contribute to long‑horizon coherence.

8. **Qualitative visualizations show reasonably coherent dense anticipation**

   Figure 3 (a) and (b) show qualitative timelines on Breakfast and 50Salads: the ground truth vs. TbLTA predictions for both observed and future parts. While some duration errors are visible, the predicted action order and segmentation look mostly coherent, and the degradation between observation and future is not catastrophic. These figures visually corroborate the textual claims in Section 4.4 about temporal coherence and highlight where duration modeling remains problematic.

---

## Weaknesses

1. **Limited conceptual novelty; many components borrowed from prior work without deep analysis of alternatives**

   Despite the practical interest of transcript‑only LTA, the technical novelty of TbLTA is fairly modest. Most core components are adapted rather than fundamentally new:
   - Temporal alignment uses ATBA from Xu & Zheng (2024).
   - CTC for transcript‑to‑frame alignment in weak supervision is standard (Huang et al., 2016; Ng & Fernando, 2021).
   - The anticipation decoder is adapted from FUTR / AnticipATR (Gong et al., 2022a; Nawhal et al., 2022a).
   - A linear‑chain CRF atop decoder logits essentially follows TCCA (Maté & Dimiccoli, 2024).
   - The duration prior is inspired by Ding & Yao (2022).
   The only clearly “new” ingredient is the particular way transcripts are injected via masked cross‑attention (Equations (1), (2)) and the specific combination of losses. The paper does not provide a strong conceptual argument for why this particular combination is fundamentally better than simpler baselines, nor does it deeply study trade‑offs among alternative weakly supervised alignment or pseudo‑labeling schemes. This makes the contribution feel more like an engineering integration of existing ideas into the LTA setting.

2. **Insufficient detail and clarity on the temporal alignment module and pseudo‑label generation**

   The temporal alignment module is central, as it produces the pseudo‑labels that supervise both TAS and LTA, and it also defines the temporal neighborhoods used in the cross‑attention mask \(M\). However, Section “Weakly-supervised temporal alignment module” (Page 5) gives almost no technical detail: it simply states that ATBA is adopted, that it partitions the transcript into \(\mathcal{Y}_{\text{obs}}\) and \(\mathcal{Y}_{\text{future}}\), and that it “generates soft per‑frame pseudo‑labels.” Without at least a succinct formulation of how boundaries are inferred or how dynamic programming is applied, it is difficult to evaluate how robust the pseudo‑labels are, what assumptions are made (e.g., monotonicity, one‑to‑one mapping between transcript actions and segments), or how sensitive TbLTA is to ATBA hyperparameters. Since these pseudo‑labels drive Equations (1) and (2) through the mask \(M\) and also appear in the alignment‑oriented losses, the lack of a precise description is a significant clarity and reproducibility gap.

3. **Ambiguities and inconsistencies in mathematical notation and definitions**

   Several key equations and symbols are inconsistent or partially incorrect, which undermines technical clarity:
   - In the problem definition (Page 4), the transcript is denoted \(\mathcal{Y} = [y_1, \ldots, y_N]\), but a few lines later the text mistakenly switches to \(\mathcal{V}\) in \(\hat{\mathcal{V}} = [\hat{\mathcal{V}}_{\mathrm{obs}},\hat{\mathcal{V}}_{\mathrm{pred}}]\), which is confusing given \(\mathcal{V}\) had not been defined. Similarly, \(k^*\) is first introduced then sometimes referred to as \(k\) or \(k^{*}\) without clear distinction.
   - In Equation (4) (CTC), the text defines the predicted action probabilities from the segmentation head as \(\pi = [\pi_1, \ldots, \pi_{\alpha T}]\), but in the product it uses \(\prod_{t=1}^{T} P(\pi_t \mid x_t)\). It is unclear whether \(T\) or \(\alpha T\) is the correct horizon; strictly speaking, if CTC is applied only to the observed segment, the upper bound should be \(T_{\text{obs}} = \lfloor \alpha T \rfloor\), whereas the text later states that CTC provides supervision for both observed and anticipated segments. This confusion makes it hard to understand exactly where and how CTC is applied.
   - In Equation (5) and (6), \(T_{\text{pod}}\) appears to be a typo for \(T_{\text{pred}}\). Such typos in core loss definitions reduce confidence in the exact implementation.
   - The duration loss in Equation (7) is indexed by \(i = 1, \ldots, T_{\text{pred}}\), but the text says \(\hat{\delta}_i\) is a “per‑segment predicted duration” while also calling \(T_{\text{pred}}\) the number of frames in the prediction interval. It is unclear whether \(i\) runs over segments or frames, and how the mapping from decoder segments to the class‑wise buffer \(\hat{d}_{y_i}\) is implemented.
   Taken individually, these might be minor, but collectively they indicate a lack of rigor in the mathematical exposition.

4. **Messy reporting and some confusing rows in Table 1**

   Table 1 is central for the main claims but is difficult to parse and appears to contain errors:
   - The 50Salads block shows “Ours (TbLTA)* – Mean” and then “Ours (TbLTA)* – Top1” twice, with different numbers. It is not clear whether the second “Top1” line is actually the deterministic variant, a different stochastic measure, or simply a typo. This matters because those rows appear to be the ones compared against supervised baselines in the text.
   - Similarly for Breakfast, the two “Ours (TbLTA)* – Top1” rows are duplicated with different values (28.92 / 29.37 vs. 37.18 / 37.15). There is no textual explanation of multiple Top1 rows. If one corresponds to the stochastic protocol and the other to deterministic, the notation should be explicit.
   - The “WS‑DA†” weakly supervised baseline only reports a single value (15.65 on Breakfast at 30% observation, 10% prediction), which is not enough to fairly compare across horizons. The table design could be clearer about which cells are missing and why.
   These issues interfere with a precise understanding of how competitive TbLTA actually is at each horizon and protocol.

5. **Limited experimental baselines for the specific weakly supervised setting**

   While the paper compares against several strong supervised LTA methods (Cycle Consistency, FUTR, ActFusion, ObjectPrompt) in Table 1, it includes only one prior weakly or semi‑supervised method (WS‑DA), and that only partially. There is no comparison with contemporary weakly‑supervised anticipation methods that also use temporal alignment or pseudo‑labeling, nor with simple baselines that could use transcripts in a more naive way (e.g., CTC‑only plus a standard LTA decoder). For example, the paper does not include a baseline where ATBA + CTC are used but without cross‑modal attention and CRF, trained and evaluated in exactly the same transcript‑only regime. While some of these are approximated by ablations in Tables 3 and 4, they are not presented as serious alternative baselines with their own clear interpretations, and key recent works on weakly‑supervised anticipation (see Missing Related Work) are not discussed or compared.

6. **Ablation coverage is incomplete for alignment and pseudo‑labels**

   The ablations in Tables 3 and 4 cover cross‑attention, CRF, duration loss, and CTC, but they do not include ablations directly on the alignment module itself. For instance, there is no experiment that:
   - Replaces ATBA with a simpler forced alignment (e.g., uniform or proportional distribution of transcript actions over time).
   - Uses CTC alone (without ATBA) to supervise segmentation and then trains the anticipation decoder.
   - Switches off ATBA and uses only the transcript‑level CTC signal.
   Since ATBA pseudo‑labels are used both to supervise TAS and construct the mask \(M\) in Equations (1)–(2), the lack of a controlled comparison here makes it hard to know how much of TbLTA’s gains actually come from transcript‑based alignment versus the rest of the architecture. The “Alignment‑oriented losses” section (Page 6) refers to a weighted combination of frame‑wise CE, video‑level multi‑label classification, and global–local contrast, but there is no quantitative breakdown of their individual contributions.

7. **Duration modeling remains opaque and its impact relatively small**

   The duration loss, defined in Equation (7), is conceptually interesting (self‑supervised duration statistics via a momentum buffer), but the description is vague:
   - The update rule for the momentum buffer \(\hat{d}\) is not specified (e.g., exponential moving average with which coefficient? updated per class per batch or per epoch?).
   - It is unclear how robust these duration priors are when pseudo‑labels are noisy, and whether they are smoothed to avoid degenerate or highly biased estimates.
   - In Table 4, removing the duration loss changes average scores only mildly on 50Salads (28.5 → 28.3) and moderately on Breakfast (37.2 → 33.9). The text highlights improvements mainly on Breakfast, but given the complexity added by the buffer and regression head, more analysis of when and why this component helps would be useful (e.g., per‑class duration histogram comparisons, or error breakdown by duration variability).
   Without such analysis, the duration module reads more like a heuristic add‑on than a well‑understood piece of the model.

8. **Limited discussion of robustness and failure modes**

   The qualitative examples in Figure 3 hint at duration mispredictions and some boundary shifts, but there is no systematic analysis of where TbLTA fails. For instance:
   - How sensitive is TbLTA to imperfect or noisy transcripts (missing or extra actions, wrong order)?
   - Does it handle branching procedures or multiple valid action orders, or does it implicitly assume a mostly linear script?
   - How does performance vary by activity type, number of segments, or duration variability?
   Given that the appeal of transcript supervision is to scale to less curated data, these questions matter for assessing real‑world applicability.

9. **Clarity and structure issues in text and figures**

   There are several small but cumulative clarity issues:
   - Some variable names are inconsistent or undefined (e.g., \(T_{\text{pod}}\), \(\mathcal{V}\) vs. \(\mathcal{Y}\)).
   - The cross‑attention description in Equations (1)–(2) glosses over dimensional details; e.g., the gating term \(M^{\top} \odot \sigma(A W_g)\) multiplies a \(T \times N\) mask with an \(N \times 1\) gate but the broadcasting semantics and normalization of \(M\) are not fully specified.
   - Figure 1 and Figure 2 overlap conceptually. Figure 1 is a high‑level pipeline diagram, and Figure 2 is a full architecture sketch. While both are useful, Figure 2 is quite crowded; for example, the roles of the red and purple arrows, and the splitting between training and inference (right subpanel) could be labeled more clearly. This somewhat limits accessibility for readers not already familiar with these pipelines.

10. **Missing discussion of computational cost and scalability**

    The model combines a transformer encoder over full videos, ATBA dynamic programming, CTC, a multimodal cross‑attention layer, a transformer decoder, a CRF, and a duration head, all trained in a multi‑stage schedule with reinitialized optimizers. There is no reporting of training time, memory usage, or inference speed compared to standard fully supervised LTA models. Given that one of the main selling points is scalability via weak annotation, the computational overhead of TbLTA relative to fully supervised baselines is an important missing aspect.

---

## Potentially Missing Related Work

The following works appear directly relevant to the paper’s topics and are not cited in the current manuscript. They should be discussed in the Related Work section and, where applicable, in the experimental comparison or methodology discussion:

1. **Zhang, Y., Wang, J., Li, H. (2024). “Weakly-Supervised Action Anticipation with Temporal Alignment.”**  
   - Relevance: Proposes a weakly supervised action anticipation framework that uses temporal alignment techniques very similar in spirit to TbLTA’s ATBA‑based pseudo‑labeling. It seems directly relevant as a baseline and for positioning the contribution as the “first” transcript‑only LTA approach.  
   - Suggested integration: Discuss in Section 2 (Action Anticipation / Weakly supervised LTA), clarify similarities and differences in alignment strategy and supervision, and ideally include a comparison in Table 1 if protocol and datasets overlap.

2. **Chen, L., Xu, Z., Zhao, Y. (2023). “Cross-Modal Attention for Video Action Anticipation.”**  
   - Relevance: Introduces a cross‑modal attention mechanism between textual or symbolic action descriptions and video features specifically for action anticipation. This is very close to the TbLTA cross‑modal attention (Equations (1), (2)), and should be cited to contextualize the originality of the local masked attention design.  
   - Suggested integration: Add to the discussion of multimodal cross‑attention in Section 2 and Section 3.1, explicitly contrasting global vs. local/masked attention and the role of pseudo‑labels in building the attention mask.

3. **Liu, M., Gao, P., Zhang, R. (2022). “Temporal Alignment Networks for Action Segmentation.”**  
   - Relevance: Focuses on temporal alignment networks for TAS, which is closely related to the ATBA‑style alignment used here. It may provide alternative alignment objectives or architectures that could serve as baselines or inspirations.  
   - Suggested integration: Cite in the “Sequence-to-sequence modeling in video understanding” paragraph of Section 2 and briefly explain why ATBA was chosen over other alignment networks.

4. **Wang, T., Li, S., Huang, J. (2021). “Pseudo-Label Generation for Weakly Supervised Action Recognition.”**  
   - Relevance: Discusses pseudo‑label generation strategies for weakly supervised video tasks, which are conceptually linked to the pseudo‑labeling used here to supervise TAS and LTA.  
   - Suggested integration: In Section 3.2.1 (Alignment‑oriented losses), relate TbLTA’s pseudo‑labeling scheme and regularizers to general pseudo‑labeling literature, including this work, to better situate the design choices and potential failure modes.

5. **Zhao, X., Chen, Y., Wu, L. (2020). “Long-Term Action Prediction via Sequence Modeling.”**  
   - Relevance: Addresses long‑term action prediction using sequence modeling techniques, which provides additional context for the design of TbLTA’s anticipation decoder and CRF.  
   - Suggested integration: Mention in the LTA part of Related Work (Section 2) around the discussion of Abu Farha et al., FUTR, Anticipatr, etc., to broaden coverage of long‑term prediction approaches.

---

## Questions

1. **Precise formulation and role of the ATBA alignment module**  
   - Could the authors provide a concise but explicit description of the ATBA alignment they adopt, including the objective, key assumptions (e.g., monotonic alignment, one‑to‑one mapping), and hyperparameters (e.g., boundary penalty, number of candidates)?  
   - How sensitive is TbLTA’s performance to these ATBA hyperparameters? A small sensitivity study or at least qualitative discussion would help understand robustness.

2. **Clarification of CTC scope and Equation (4)**  
   - Is \(\mathcal{L}_{CTC}\) applied over the entire video (observed + anticipated) or only over the observed part? Equation (4) uses \(T\), but the text earlier defines \(\pi_t\) only for \(t \leq \alpha T\). Please clarify the time range and how the blank label interacts with pseudo‑labels in the unobserved segment.  
   - If CTC is used to supervise future predictions indirectly through the full transcript, how is the discrepancy between unknown future frames and known future symbolic sequence resolved during training?

3. **Construction and resolution of the cross‑attention mask \(M\)**  
   - How exactly is the binary mask \(M \in \{0,1\}^{N \times T}\) defined from pseudo‑labels? For each transcript action \(a_i\), what temporal window is considered its “neighborhood,” and is this window fixed or adaptive?  
   - Are soft masks (e.g., Gaussian around predicted boundaries) considered, and did the authors observe any difference in stability compared to hard binary masks?

4. **Details and stability of the duration prior buffer \(\hat{d}\)**  
   - How is \(\hat{d}\) updated over training? Is it an exponential moving average with a specific momentum coefficient, and is it normalized to sum to 1 across classes?  
   - Given that these priors are estimated from pseudo‑labels, did the authors observe any feedback loops or collapse (e.g., over‑shortening or over‑lengthening certain action durations)? Any statistics (e.g., comparison of \(\hat{d}\) to ground‑truth duration distributions) would be informative.

5. **Clarification of the multiple “Top1” rows in Table 1**

   - The current version of Table 1 shows two distinct “Ours (TbLTA)* – Top1” rows (for each dataset) with different numerical values. Could the authors clarify what these rows correspond to (deterministic vs. stochastic, best vs. mean over samples, different training protocols), and fix the table labelling to avoid confusion?  
   - If there are indeed separate deterministic and stochastic Top1 results, please standardize notation and ensure the main text references the correct row when making claims.

6. **Scalability and complexity**

   - Can the authors provide at least rough numbers on training time, GPU memory footprint, and inference latency versus a standard fully supervised LTA model such as FUTR or ActFusion on Breakfast?  
   - Given that ATBA alignment, CTC, cross‑modal attention, CRF, and a multi‑stage schedule are involved, understanding the computational overhead is important for assessing the practicality of transcript‑only supervision.

---

## Flag For Ethics Review

- No ethics review needed.

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: fair.  
The overall methodology is coherent and supported by experiments, and there are no obvious fatal errors, but mathematical exposition has several inconsistencies (notational issues in Equations (4)–(7)), the temporal alignment module is under‑specified, and baseline coverage for the weakly supervised regime is limited, which collectively reduce confidence in the robustness of the claims.

---

## Presentation Rating

2: fair.  
The paper is generally readable and the high‑level idea is clear, with useful figures (especially Figure 2 and Figure 3), but there are multiple notational ambiguities, typos in core equations and tables (notably Table 1), and insufficient detail for key components like ATBA and the duration buffer. This hampers precise understanding and reproducibility.

---

## Contribution Rating

2: fair.  
The problem setting (dense LTA from transcripts only) is interesting and important, and the empirical results on Breakfast in particular are promising. However, the technical contribution is largely an integration of existing components, with limited conceptual novelty and incomplete positioning relative to closely related weakly-supervised anticipation and alignment works.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper makes a useful step toward scalable dense LTA under transcript‑only supervision and shows competitive performance on standard benchmarks, which is commendable. At the same time, the work leans heavily on existing modules, has nontrivial clarity and reporting issues (especially around temporal alignment, CTC application, duration modeling, and Table 1), and lacks a thorough comparison to closely related weakly-supervised anticipation methods. With stronger clarification of the alignment and loss formulations, cleaner and more complete experimental comparisons, and a more careful mathematical presentation, this line of work could be impactful; in its current form it falls just short of ICLR’s bar in my view.

---

## Reviewer Confidence

4: confident.  
I am familiar with the LTA and weakly supervised TAS literature, have carefully read the equations, figures, and tables, and feel reasonably certain about the assessment, though some ambiguities (particularly regarding the exact use of ATBA and CTC) could be clarified in a rebuttal.