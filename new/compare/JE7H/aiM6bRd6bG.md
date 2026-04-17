---
job_id: 239e6308-1a25-4f26-8891-625fb3d553dd
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: aiM6bRd6bG.pdf
paper: PPI Candidate Ranking: Large-Scale Evaluation of a Domain Knowledge-Guided Pipeline
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work focuses on representation learning for PPI prediction and ranking, using deep sequence models, contact maps, knowledge-guided re-ranking, and large-scale evaluation on STRING; this is squarely within ML for biology / learning on graphs / interpretability and representation use.

## Minimum Quality
Pass ✅.  
The paper is in English and contains all key sections (Abstract, Introduction, Related Work, Background/Methodology, Results/Experiments, Conclusions). Methods are nontrivial, the math is mostly coherent, and there is substantial experimental work. While I see issues in clarity, methodological rigor, and evaluation design, none rise to the level of automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to address or manipulate automated reviewing systems, nor any hidden prompts or suspicious text.

---

# Expected Review Outcome:

## Summary

The paper introduces the task of PPI candidate ranking: given a target protein and its known interaction partners from an earlier STRING release, rank novel candidate partners that are most likely to correspond to interactions that appear only in a later STRING release.  

The authors propose an interpretability-guided retrieval strategy that leverages D-SCRIPT or Topsy-Turvy contact maps to identify “active” residue regions in known partners, then ranks candidates by maximum cosine similarity to these active embeddings (Equations (3–5), Figure 1). They further apply a second-stage re-ranking using interaction scores, structural plausibility (SpeedPPI / pDockQ), and semantic/LLM-based similarity on UniProt-derived annotations, and report large improvements in ranking metrics (Table 1 and Table 2) compared to directly using model interaction probabilities on STRING v11→v12 and PiNUI.

---

## Strengths

1. **Clearly articulated and practically relevant task formulation (PPI candidate ranking).**  
   The shift from binary PPI classification to “retrieving” novel partners of a given protein across STRING versions is well motivated in the Introduction (Pages 1–2): it aligns with real experimental prioritization, and the use of STRING v11 vs v12 makes the evaluation at least temporally separated from training data. This is an appreciable step beyond static, single-release link prediction benchmarks.

2. **Interpretability-guided embedding use is conceptually interesting and technically nontrivial.**  
   The core idea, formalized in Section 4.1 and Figure 1, is to exploit the residue-level contact maps from D-SCRIPT/Topsy-Turvy to identify high-activation segments \(I_k\) in known partners \(p_k\) and then search candidate proteins for similar segments by sliding-window cosine similarity (Equation (3)). This is a concrete and reasonably well-defined way of using internal model structure, rather than just scalar interaction probabilities, to define a task-specific similarity metric.

3. **Substantial empirical evaluation, including a large STRING v11→v12 setup.**  
   The experiments in Section 5 cover multiple aspects:  
   - Large-scale evaluation on STRING v11→v12 with several metrics at cutoffs \(k\in\{5,10,50,100,200,500\}\) (Table 1).  
   - Pairwise rank-shift analysis across different re-ranking signals (Table 2).  
   - Runtime analysis of retrieval vs re-ranking (Figures 2 and 3).  
   - Additional experiments on PiNUI (Table 4), which is more stringent and less entangled with STRING’s evidence sources.  
   This breadth increases confidence that the proposed pipeline was actually implemented and run at scale.

4. **Clear quantitative evidence that raw prediction scores are suboptimal for ranking, and that the proposed method reshapes rankings substantially.**  
   In Table 1, comparing the D-SCRIPT / Topsy-Turvy / xCAPT5 “Prediction Probability” rows to “Our Approach”, Recall@10 increases from below ~2% to ~26% (D-SCRIPT-side setting) and from ~1% to ~11% (Topsy-Turvy-side setting), with consistent gains in MAP@nDCG. Even accounting for table formatting issues, the magnitude of improvement suggests that contact-map-guided embedding similarities capture complementary signals to scalar probabilities. This strongly supports the argument that “interpretability-aware” use of these models can materially affect ranking.

5. **Systematic analysis of complementary signals and tradeoffs.**  
   The re-ranking module in Section 4.2 includes a nice spectrum of evidence types: D-SCRIPT interaction scores (Equation (6)), structural plausibility via SpeedPPI / pDockQ (Equation (7)), simple semantic overlaps (Equations (8–10)), and LLM-based bi- and cross-encoders (Equation (11–12)). Table 2’s pairwise comparison of rank shifts is a useful diagnostic: for example, PubMedBERT improves or maintains 75.5% of rediscoveries relative to cosine, while IS complements cosine (~63% improved) and pDockQ appears less effective for early ranking. This table gives a more nuanced view than reporting only raw metrics.

6. **Runtime / scalability considerations are explicitly measured.**  
   Figures 2 and 3, along with Appendix A.2, show absolute runtimes for retrieval and re-ranking. This is valuable for practitioners: e.g., Figure 2 reveals that the interpretability-guided retrieval is roughly half the runtime of the probability-based baseline, and Figure 3 highlights how SpeedPPI is prohibitively slow compared to semantic methods. This helps position the work as an actual pipeline rather than just an algorithmic sketch.

7. **Useful reflection on limitations.**  
   The Conclusion (Section 6, Pages 10–11) honestly notes that the method fundamentally relies on having known partners for a target and may fail for under-studied proteins, and that despite the “interpretability-guided” label, the final rankings are not truly explanatory, since embeddings remain black-box. This balanced discussion improves the credibility of the work.

8. **Figures are helpful for understanding the method and tradeoffs.**  
   - **Figure 1** lays out the end-to-end flow from target and known partners, through contact maps and active residues, to windowed cosine similarity and ranking. It clarifies how indices \(I_k\) are used to extract regions from known partners and slide against candidates.  
   - **Figure 2** directly contrasts retrieval runtimes of probability-based vs interpretability-guided methods, reinforcing that the new method is not more expensive at retrieval.  
   - **Figure 3** vividly shows the orders-of-magnitude runtime cost of SpeedPPI in the re-ranking step relative to semantic methods, which nicely supports the argument that structural signals are computationally heavy and may be better reserved for later filtering.

---

## Weaknesses

I see several substantial issues, both methodological and presentational. The list is long; many are fixable but they currently keep the paper below ICLR standards.

1. **Ambiguity and possible inconsistency in the definition of the interaction score (IS), especially Equation (6).**  
   - In **Section 3**, D-SCRIPT is described as computing a residue-residue contact map \(C\in[0,1]^{n\times m}\), applying convolution/pooling, then compressing to a scalar interaction probability \(\hat p\in[0,1]\) via logistic activation. This aligns with the original D-SCRIPT paper.  
   - In **Section 4.2**, the IS used for re-ranking is defined as  
     \[
     \hat{p} = \max_{i \le n, j \le m} C(p,p_c)_{ij},
     \tag{6}
     \]
     and then called “sharpened through a logistic activation” although no logistic is applied in Equation (6). This conflicts with the earlier description that \(\hat p\) is the output of an interaction module aggregating over contact patterns, not a simple elementwise max.  
   - It is unclear whether the authors actually use D-SCRIPT’s official scalar score (post-logistic) or this new “max contact” score as their IS. This discrepancy matters because Table 1 compares “Prediction Probability” vs “Our Approach” presumably using the original probability; but the IS re-ranking in Table 2 may be using a different signal. The paper should either:  
     * Correct Equation (6) to match the actual implementation of D-SCRIPT’s score, or  
     * Explicitly state that they replace the original interaction module with a max-pooling baseline and justify why this is acceptable.  
   Without this clarity, it is hard to interpret the role of the IS signal across Tables 1 and 2.

2. **Methodological opacity around candidate sets and ranking protocol.**  
   The problem setup in Section 4 defines \(CP(p) = P \setminus KP(p)\) and \(NP(p)\) as novel partners, but the actual construction of the ranking candidate universe in experiments is not fully specified. Issues include:
   - In **Section 5.1**, negatives are generated with a 10:1 negative-to-positive ratio, but it is unclear whether these negatives are included in the ranking candidate pool for each \(p\) or only used for training the original models (D-SCRIPT, Topsy-Turvy, xCAPT5). Are rankings performed over all non-known partners in the filtered STRING graph, or over a downsampled set of negatives plus the future positives?  
   - In **Section 5.3**, the text says “retrieval remains the computational bottleneck, with runtimes in the order of hundreds of hours (Figure 2)”, but does not quantify how many candidate proteins are evaluated per query, nor how the runtime scales with dataset size.  
   - For re-ranking, they say “top 10 ranked candidates for each target protein” are considered, resulting in 2,280 pairs. That implies roughly 228 proteins, but we are never clearly told how many target proteins there are overall in STRING v11→v12 under their filtering, or what fraction of all positive v12 edges are covered by these.  
   The lack of transparent description of candidate space and ranking procedure makes it hard to evaluate how realistic these metrics are in an interactome-scale setting.

3. **Inconsistent and partially unreadable presentation of Table 1, undermining interpretability of the claimed gains.**  
   - **Table 1** merges several models (D-SCRIPT, Topsy-Turvy, xCAPT5, “Our Approach” for both D-SCRIPT and Topsy-Turvy) but the rows are not clearly labeled. There are blocks that clearly correspond to different models (e.g., the rows starting with “Prediction Probability” then a cluster of k rows, then another cluster likely for Topsy-Turvy, then for xCAPT5, then “Our Approach” for D-SCRIPT, then another “Our Approach” for Topsy-Turvy); however the “Model” column is mostly blank.  
   - Many metric columns such as “Pred. Cov.” and “MRR” are empty for almost all rows. The text in **Section 5.3** claims MRR increases by “4–6 times”, but MRR is not actually populated in Table 1 for most entries.  
   - There are typos (e.g., “Prediction Probability” rows duplicated; “D-SCRIPTand-, Topsy-Turvy, Both baselinesrecover”) that make it harder to map narrative text to the numbers.  
   As a result, while one can qualitatively see that “Our Approach” has much higher Recall@k and nDCG@k, the exact quantitative comparison to each baseline is awkward and prone to misinterpretation. For a core result table, this level of sloppiness is problematic.

4. **Limited and potentially optimistic evaluation design with respect to STRING’s evidence composition.**  
   - The purported “prospective” evaluation uses STRING v11 for training and v12 for evaluation (Section 5.1), but STRING v12 integrates heterogeneous evidence sources, many of which are already present in v11 (e.g., literature-based scores, homology-based evidence). It is not clear to what extent v12 positives are genuinely unseen by D-SCRIPT/Topsy-Turvy/xCAPT5 training or whether some interactions are simply upgraded based on more evidence.  
   - Topsy-Turvy uses GLIDE scores derived from the protein interaction network, and STRING itself already aggregates network context; the dependence between training data and evaluation labels is not discussed.  
   - Without careful analysis of how many v12 edges correspond to completely new experimental discoveries versus re-annotations, the “future interactions” framing might be overstated. There is some mitigation via PiNUI (Table 4), but results on PiNUI are relatively weak and not deeply analyzed.  
   This does not invalidate the ranking method, but it does weaken the claim that the evaluation truly measures ability to anticipate entirely novel interactions.

5. **Interpretability claim is somewhat overstated relative to what is provided.**  
   While the paper correctly notes in the Conclusion that rankings are not fully explanatory, the overall framing is heavily “interpretability-guided”. However:
   - The only interpretability-related mechanism used is the selection of high-activation segments \(I_k\) from contact maps, which are then flattened and used in cosine similarity (Equations (3–4)).  
   - There is no analysis of whether selected residues correspond to known binding interfaces, conserved motifs, or functionally important regions. No qualitative examples are shown (e.g., highlighting \(I_k\) on structures).  
   - Figure 1 is helpful for understanding the process but does not provide any biological interpretability.  
   As a result, the method is essentially a clever use of internal activations for similarity, not an interpretability study per se. The paper’s title and framing could be toned down or, alternatively, more analysis provided to justify the interpretability angle.

6. **Mathematical and notational clarity issues in Section 4.1.**  
   The technical core, Equations (3–5), has several weaknesses:
   - The definition of \(I_k\) is described in prose but the notation is muddled. On **Page 5**, the text oscillates between “activation score of residue \(j\) in \(p_k\) is defined as the maximum contact probability with any residue of \(p\)” and “we identify all maximal contiguous segments of highly activated residues” and then pick the highest-average one. However, there is no formal definition of the activation threshold, nor of what counts as “maximal contiguous segment”. Is it all segments above some percentile? Are gaps allowed? This is important, since the sliding window size \(|I_k|\) directly drives the similarity and runtime.  
   - In Equation (3), the similarity is
     \[
     \mathrm{sim}(p_c,p_k) = \max_{i=0}^{i < n_c - |I_k|} \frac{\langle z_k[I_k], z_c[i:i+|I_k|]\rangle}{\|z_k[I_k]\|_2 \cdot \|z_c[i:i+|I_k|]\|_2}.
     \]
     Here, \(z_k[I_k]\) and \(z_c[i:i+|I_k|]\) are said to be “flattened embeddings”, but the resulting dimension is \(|I_k|\times d\), so vectorization order matters. This is not specified. Also, the index range \(i=0\) to \(n_c - |I_k|\) suggests 0-based indexing, while \(I_k \subseteq \{1,\dots,n_{p_c}\}\) uses 1-based. This is a minor but unnecessary confusion.  
   - There is no mention of normalization or scaling across different \(|I_k|\): do longer segments systematically change the cosine distribution? Is there any regularization or limit on \(|I_k|\) to avoid cases where essentially the entire sequence is compared?  
   A more rigorous mathematical formulation (perhaps with pseudocode) would make the core contribution easier to scrutinize and reproduce.

7. **Experimental analysis on PiNUI is underdeveloped and not fully convincing.**  
   - Table 4 shows that on PiNUI, “Our Approach” improves rediscovery ratio (0.3849 vs 0.0080) and Recall@k/MAP@k at large k, but the Success@k numbers are extremely low; e.g., Success@50 remains 0 for “Our Approach” (which is odd and suggests an error in the table, since Recall@50 > 0).  
   - The narrative in **Appendix A.3** spends most of its space explaining why the higher Average Rank for “Our Approach” does not mean worse performance, due to more rediscoveries being included. That is reasonable, but there is no discussion of *why* the method struggles on PiNUI in terms of early ranking (e.g., Recall@5/10 close to zero) nor any ablation to understand which components fail.  
   - Given that PiNUI is explicitly less tied to STRING’s own evidence mix (and thus arguably a more objective test), the relative underperformance should be confronted more directly.

8. **Re-ranking evaluation is somewhat indirect and omits absolute ranking metrics.**  
   - For re-ranking (Section 5.2–5.3, Table 2), the evaluation is based on pairwise comparisons of whether rediscoveries “maintain or improve” their rank when switching from one method to another. This is informative about relative shifts but does not report direct Recall@k/MAP@k/nDCG@k after re-ranking, so we cannot see absolute performance gains.  
   - The candidate set used for re-ranking is restricted (2,280 pairs), but there is no reporting of how these improvements translate to metrics at top-k for the full ranking problem. For example, how much does PubMedBERT re-ranking improve Recall@10 beyond the 26.4% already achieved by the interpretability-guided method?  
   Without absolute metrics, it is difficult to assess whether the re-ranking step is of major practical value or just a modest refinement.

9. **Missing or under-discussed related work on interpretable interaction prediction.**  
   The Related Work section is reasonably broad but omits several directly relevant recent works that explicitly use interpretable deep architectures or attention for biological interactions (see next section). This weakens the positioning of the proposed “interpretability-guided” pipeline and makes it sound more unique than it actually is.

10. **Clarity and polish issues throughout the text.**  
   There are many small but pervasive editorial problems that make the paper read as somewhat rushed:
   - Broken or duplicated citations (e.g., D-SCRIPT cited twice with slightly different years on Page 1–3).  
   - Typos and sentence glitches in Section 4.1 and 5.3 (“design a the two-stage framework”, “D-SCRIPTand-, Topsy-Turvy, Both baselinesrecover”).  
   - Inconsistent tense and style.  
   While not fatal, for a multi-component pipeline paper this level of noise does hinder understanding.

Given the instructions, I count well over seven substantive weaknesses affecting methodological clarity, evaluation credibility, and exposition. This constrains the overall recommendation even though some ideas are promising.

---

## Potentially Missing Related Work

The following directly relevant works are not cited and should be discussed, especially in Section 2 (Related Work) and when positioning the interpretability-guided aspects in Sections 3–4:

1. **Mou et al., “A Transformer-Based Ensemble Framework for the Prediction of Protein-Protein Interaction Sites”, 2023.**  
   - This paper focuses on PPI *sites* rather than pairwise interaction scores, but it uses deep sequence models with attention to capture interface regions, which is closely related to the use of contact maps and activated residues in this submission.  
   - It should be discussed in Section 2 as an example of deep, somewhat interpretable models for PPI site prediction and contrasted with the use of contact-map activations here.

2. **Han et al., “Predicting Protein-Protein Interaction with Interpretable Bilinear Attention Network”, 2025.**  
   - Introduces an interpretable attention mechanism for PPI prediction that highlights key residue-residue interactions.  
   - This is directly relevant to the “interpretability-guided” theme and should be compared to the current use of D-SCRIPT/Topsy-Turvy contact maps, particularly in Section 3 (Background) and 4.1 (Interpretability-Guided Retrieval).

3. **Kurata & Tsukiyama, “ICAN: Interpretable Cross-Attention Network for Identifying Drug and Target Protein Interactions”, 2022.**  
   - Although focused on drug–target rather than PPI, it proposes an interpretable cross-attention mechanism that closely parallels the idea of using internal attention patterns for candidate ranking.  
   - It could be cited in Section 2 as part of the broader trend of interpretable interaction prediction architectures, and briefly compared in the discussion.

4. **Liu et al., “RNA-Protein Interaction Prediction Using Network-Guided Deep Learning”, 2025.**  
   - Uses network-guided deep learning for RNA–protein interactions, integrating multiple biological signals (sequence, structure, network).  
   - Given that Topsy-Turvy also injects network information and the current paper combines multiple evidence sources (sequence, structure, text), this work could be discussed in Section 2 as an analogous approach in a different interaction domain.

5. **Wang et al., “A Bidirectional Interpretable Compound-Protein Interaction Prediction Framework Based on Cross Attention”, 2024.**  
   - Presents an interpretable cross-attention framework for compound–protein interactions. Again, the interpretability aspect and cross-modal attention parallels the use of contact-map-derived regions here.  
   - It should be discussed in Related Work and briefly compared when motivating the use of interpretable architectures for interaction ranking.

6. **Lin et al., “DeepRLI: A Multi-Objective Framework for Universal Protein–Ligand Interaction Prediction”, 2025.**  
   - Emphasizes integration of diverse biological signals and multi-objective optimization for interaction prediction.  
   - This is conceptually similar to the multi-signal re-ranking (IS, pDockQ, semantic, LLMs) and could be cited in Section 2 and referenced when discussing the complementary nature of these signals in Section 5.3.

7. **Kim et al., “PriorCCI: Interpretable Deep Learning Framework for Identifying Key Ligand–Receptor Interactions Between Specific Cell Types from Single-Cell Transcriptomes”, 2025.**  
   - Uses interpretability in a different but related setting (ligand–receptor interactions across cell types). The idea of prioritizing key interactions via interpretable deep learning connects naturally to PPI candidate ranking.  
   - Should be referenced in Section 2 as part of the broader landscape of interpretable deep models for biological interactions.

Including and contrasting these works would help position the paper as part of a broader movement toward interpretable, multi-signal interaction prediction, rather than as an isolated proposal.

---

## Questions

1. **Precise definition and implementation of the interaction score (IS).**  
   - Please clarify whether Equation (6) actually defines the interaction score used in experiments, or if you instead use D-SCRIPT’s original scalar output (after convolution/pooling + logistic).  
   - If Equation (6) is indeed used, why deviate from the original D-SCRIPT interaction module? Have you compared performance of max-contact vs original IS as a re-ranking signal?

2. **Formal definition of active segments \(I_k\).**  
   - What is the precise threshold or rule used to define “highly activated” residues from the activation profile along \(p_k\)? Is it a fixed threshold, a percentile, or something learned?  
   - How do you handle cases where multiple segments have similar average activation? Do you always pick exactly one segment \(I_k\), and is there a minimum or maximum allowed length?

3. **Candidate pool and negatives in STRING v11→v12.**  
   - For each target protein \(p\), what is the size of \(CP(p)\) in practice? Is it literally all proteins minus known partners, or a pruned subset?  
   - How do the artificially generated negatives (10:1 ratio) in Section 5.1 interact with candidate generation? Are these negatives included in the ranking pool? If so, does this distort the realism of the candidate space relative to the actual human proteome?

4. **Absolute metrics for re-ranked lists.**  
   - Can you provide Recall@k, MAP@k, nDCG@k for the full ranking pipeline *after* applying the best re-ranking signals (e.g., PubMedBERT, TF-IDF) on top of the interpretability-guided retrieval?  
   - This would concretely show how much the re-ranking step helps beyond the already large improvement from the interpretability-guided method in Table 1.

5. **Nature of v12 “novel” interactions.**  
   - Have you analyzed what proportion of v12 positives are:  
     (a) brand-new experimental PPIs,  
     (b) interactions already suggested by low-confidence evidence in v11, or  
     (c) edges added due to updated text mining or homology propagation?  
   - Such an analysis would help contextualize how “prospective” your evaluation really is.

6. **Failure modes on PiNUI.**  
   - Can you investigate why early Recall@k and Success@k remain almost zero on PiNUI (Table 4) despite higher rediscovery ratio? For example, is it due to missing known partners for many proteins, or because D-SCRIPT contact maps are poorly calibrated on PiNUI’s sequence distribution?  
   - Some qualitative examples or ablations would be instructive.

7. **Interpretability analysis of \(I_k\) segments.**  
   - Can you provide at least a few case studies where the selected active segments \(I_k\) align with known binding motifs, domains, or interfaces (e.g., mapping to Pfam/InterPro or PDB structures)?  
   - This would significantly strengthen the claim that the approach is interpretability-guided in a biologically meaningful sense.

Addressing these questions clearly in the rebuttal and possibly tightening notation / tables could substantially improve my confidence.

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: fair.  
The core ideas are technically plausible and mostly well-founded (contact-map-based region selection, cosine-based retrieval, multi-signal re-ranking), and the empirical results are promising. However, unresolved ambiguities in the definition of key quantities (IS, \(I_k\)), incomplete specification of candidate pools, and the somewhat optimistic nature of the evaluation relative to STRING evidence prevent a higher soundness score.

---

## Presentation Rating

2: fair.  
The paper is readable and the high-level story is clear, with helpful figures (especially Figure 1). However, there are many notation inconsistencies, typos, and a confusing main results table (Table 1) with missing entries, which collectively hinder careful scrutiny. More rigorous mathematical specification and cleaner tables are needed.

---

## Contribution Rating

2: fair.  
The paper tackles an important, practically relevant question (prospective PPI partner ranking) and offers a reasonably creative use of existing PPI models and semantic/structural signals. Yet, the methodological novelty is moderate (recombining existing components), interpretability is under-explored, and the evaluation design has limitations, so the overall contribution is useful but not strong enough for a clear accept.

---

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The idea of using contact-map activations from D-SCRIPT / Topsy-Turvy to define region-guided similarity for PPI candidate ranking is interesting and, coupled with the multi-signal re-ranking, leads to substantial improvements over naive probability-based baselines on STRING v11→v12. The evaluation is large-scale and the runtime measurements are appreciated.  

However, several weaknesses significantly limit the paper’s impact at this stage: fuzzy mathematical definitions (Equations (3) and (6)), incomplete description of the ranking candidate space, confusing presentation of the main results (Table 1), underdeveloped analysis on a more challenging dataset (PiNUI), and somewhat overstated interpretability claims. With a thorough revision addressing these issues and better positioning vs closely related interpretable interaction models, this could become a solid contribution, but in its current form it falls just short of ICLR standards.

---

## Reviewer Confidence

4: confident.  
I am familiar with D-SCRIPT, Topsy-Turvy, and STRING-based evaluations, and I have carefully checked the core equations and experimental setup as far as they are described. Some ambiguities remain due to presentation issues, but my overall assessment is unlikely to change drastically unless key methodological clarifications or additional results are provided.