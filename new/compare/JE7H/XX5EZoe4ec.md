---
job_id: 8a7a024d-4240-45d5-9428-41dc41f26ae2
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: XX5EZoe4ec.pdf
paper: RetrievalFormer: A Dual-Encoder Transformer for Efficient Approximate Nearest Neighbor Retrieval and Cold-Item Recommendation
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a dual-encoder transformer architecture for recommendation, focused on representation learning, metric learning, and large‑scale inference with ANN; this is squarely within ICLR’s core topics.

## Minimum Quality
Pass ✅.  
All core sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present and reasonably detailed. The method is clearly specified, the math is standard and appears consistent, and the experiments are substantial on three public benchmarks plus a production dataset, with reasonable baselines. I do not see fatal methodological flaws or data leakage.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any instructions targeting automated reviewers or other hidden manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary
The paper introduces RetrievalFormer, a dual-encoder sequential recommender that replaces the standard ID-softmax output layer with a shared user–item embedding space trained via an InfoNCE objective. The user tower is a transformer over enriched interaction sequences, while the item tower encodes heterogeneous item features using an AttentionFusion module with shared embedding tables across towers. Experiments on Amazon Beauty, Amazon Toys & Games, MovieLens‑1M, and an internal email-marketing dataset show that RetrievalFormer attains 86–91% of the Recall@20 of strong transformer baselines, supports zero‑shot cold-item recommendation under a Leave‑One‑Out Cold (LOOC) protocol, and achieves up to 288× lower latency at 10M items using ANN retrieval.

## Strengths
1. **Clear, well-motivated reformulation of sequential recommendation as dual-encoder retrieval.**  
   The paper articulates convincingly why ID-softmax transformers are problematic at scale (both in latency and cold-start handling) and shows how a dual-encoder with ANN search addresses these issues. Section 3.1 and Equation (9) give a clean and standard contrastive formulation, and the inference-time story (precompute item embeddings, ANN over them) is practical and easily adoptable.

2. **Attention-based heterogeneous feature fusion with shared embeddings is thoughtfully designed and empirically justified.**  
   The AttentionFusion module (Equations (1)–(4) on Pages 4–5) provides a principled, permutation-invariant way to aggregate mixed feature types. The design choice of sharing embedding tables across user and item towers (Section 3.2.2 and Appendix B.2) is interesting for cold-start and parameter efficiency, and the ablations in Table 3 (Appendix E, Page 20) show measurable gains from self-attention fusion and from the “uniformity” training setup.

3. **Comprehensive experimental evaluation including strong recent baselines and a production case study.**  
   Table 1 (Page 8) benchmarks RetrievalFormer against a wide spectrum of sequential recommenders, including recent attribute-aware transformers such as AttrFormer and MT4SR, on three standard datasets. While RetrievalFormer is not the top performer in accuracy, it consistently lands close to the cluster of strong transformers. The email marketing case study (Table 5, Page 21) shows clear gains over a content-based baseline in a 100% cold-start regime, which is quite relevant for real-world deployment.

4. **Well-designed cold-start evaluation protocol.**  
   The LOOC protocol in Section 4.4 and Appendix F is more carefully constructed than typical “cold-start” experiments. Table 2 (Page 9) and Table 4 (Page 21) quantify the drop from LOO to LOOC, the size of cold item sets, and the fraction of interactions removed. This makes clear that the cold items are nontrivial and that the model still achieves non-negligible Recall@20 on completely unseen items, whereas ID-softmax baselines are inapplicable.

5. **Strong evidence for inference-time efficiency and scalability.**  
   The latency analysis in Section 4.5 and Figure 2 (Page 10) is one of the more carefully quantified parts of the paper. The figure clearly shows linear growth for exhaustive scoring vs sub-linear for IVF-PQ on a log-scale latency axis, and the text provides concrete numbers (e.g., 292 ms vs 1.02 ms at 10M items). This backs the central claim that the dual-encoder + ANN formulation materially changes the scaling regime.

6. **Figures effectively explain architecture and input structure.**  
   Figure 1 (Page 4) gives a reasonably understandable high-level view of how shared feature embeddings feed both towers, where AttentionFusion sits, and how the dot-product scoring is used. Figure 3 (Appendix I, Page 22) adds useful detail on how interaction-type embeddings, [SEP], and [CLS] are organized in the user sequence. These diagrams make the otherwise quite involved feature-processing pipeline easier to follow.

7. **Reproducibility and implementation details are unusually thorough.**  
   The appendices provide detailed hyperparameters, training schedules, feature-noising scheme, and FAISS configuration. Table 3’s ablations, along with the InfoNCE/negative-sampling discussion in Appendix C, give implementers concrete guidance on design choices and expected sensitivities.

## Weaknesses
1. **Accuracy vs. baselines is modest and sometimes significantly behind the strongest models.**  
   From Table 1 (Page 8), RetrievalFormer’s Recall@20 on MovieLens‑1M is 0.337, which is below several established baselines (e.g., GRU4Rec 0.3579, LightSANs 0.3590, TiSASRec 0.3558). Even on Amazon datasets, RetrievalFormer typically trails AttrFormer and others (e.g., on Amazon Toys Recall@20, RetrievalFormer 0.1169 vs DIF-SR 0.1342, AttrFormer 0.1357). The paper frames this as “86–91% of transformer performance,” but from a recsys perspective that is a 9–14% relative degradation, which is not obviously small. The work would be stronger with a deeper analysis of when this tradeoff is acceptable and whether there are regimes (certain user segments, tail items, longer histories) where RetrievalFormer actually wins in quality rather than simply being “almost as good.”

2. **Comparisons conflate architectural changes with retrieval approximations.**  
   The paper repeatedly attributes the performance gap to ANN vs. exact softmax scoring. However, in all reported metrics (Table 1), RetrievalFormer is evaluated without any candidate truncation from ANN (it is trained and evaluated as a dual-encoder dot-product scorer). There is no direct accuracy comparison between (a) an exact top‑K over all item embeddings and (b) ANN-based approximate retrieval for the same user and item embeddings. Figure 2 (Page 10) only reports latency, not Recall@K degradation due to IVF-PQ. Without such a recall-vs-latency curve, it is speculative to say that the accuracy drop relative to SASRec et al. is fundamentally due to the dual-encoder formulation rather than training objective, featureization, or other modeling choices.

3. **Positioning relative to existing dual-encoder retrieval and dense embedding work is thin.**  
   While the paper cites two-tower recommenders like Yi et al. (2019) and embedding-based retrieval in industry, it omits several closely related dense retrieval / dual-encoder works from the IR community that have already explored transformer-based dual-encoders and efficient ANN retrieval (see “Potentially Missing Related Work” below). This weakens the claim of conceptual novelty: much of the high-level recipe “transformer user encoder + feature-based item encoder + contrastive loss + ANN” is standard in dense retrieval and entity retrieval. The distinctiveness here seems to be mainly the particular feature fusion and cold-start protocol; the related work section should make this more explicit.

4. **LOOC evaluation is informative but somewhat one-sided.**  
   Table 2 (Page 9) demonstrates that RetrievalFormer retains some performance under strict cold-item conditions, which is valuable. However, the analysis is mostly descriptive. There is no quantitative comparison to other content-based recommenders on public data (e.g., a dual-encoder trained purely on features without sequences, or a hybrid that combines MF-style embeddings with attributes). The only baseline mentioned for cold-start is a Content-based KNN, and its results on public datasets are not shown. Consequently it is unclear whether the LOOC numbers (e.g., Recall@20 = 0.2267 on MovieLens‑1M) are actually competitive in an absolute sense, or simply “nonzero.”

5. **Lack of per-dataset and per-setting latency baselines for competing methods.**  
   The latency study (Section 4.5, Figure 2) compares: exhaustive dot-product scoring (which appears to correspond to using the RetrievalFormer embedding space but brute-force over all items), IVF-PQ ANN, and external SASRec measurements from ETUDE. However, there is no direct head‑to‑head system-level comparison of “SASRec + ANN candidate generation + small re-ranker” which is a very standard practical setup, nor of more recent sparsified retrieval models (e.g., Su et al., 2023 referenced in the intro). As a result, the claim that RetrievalFormer “changes the scaling behavior” is somewhat overstated; in practice many systems use a separate retriever + re-ranker arrangement, which is not evaluated here.

6. **Some empirical choices lack rigorous justification or are only partially explored.**  
   The ablation Table 3 (Page 20) is useful but quite narrow: all results are on a single dataset (Amazon Toys) and one metric. Important design elements like shared embeddings (discussed extensively in Section 3.2.2 and Appendix B.2) are not ablated directly there, even though Table 5 (Page 21) suggests a large contribution on the email dataset. Similarly, the “uniformity loss” is described as “implicit through InfoNCE” but then toggled as an enabled/disabled binary; it is not fully clear whether this refers to additional regularization or a hyperparameter regime. A more precise mathematical description (e.g., an explicit added term corresponding to Equation (15), or a clear change in training setup) would help.

7. **Mathematical exposition around InfoNCE + MNS is occasionally hand-wavy.**  
   Equation (9) presents the standard in-batch InfoNCE loss, and Equation (16) introduces a generalized MNS loss. However, the main text (Section 3.5) asserts that “we employ Mixed Negative Sampling (MNS), augmenting each batch with uniformly sampled items from the catalog,” without clearly specifying how these are integrated into the normalization term in Equation (9) and whether the implementation matches Equation (16) with importance weights \(w_j\). For example, are the additional negatives encoded with the item tower on the fly in the same batch, or are precomputed embeddings used? How many such negatives are added relative to batch size B? Since the geometry of the embedding space is central to ANN effectiveness, the precise structure of the negative distribution is not a minor detail.

8. **Theoretical claims on complexity and uniformity are high level and not fully substantiated.**  
   Section 4.5 and Appendix J argue that ANN search implies \(O(\log N)\) complexity for “tree-based indices,” but the concrete index used in the experiments is IVF-PQ in FAISS, whose empirical behavior is sub-linear but not asymptotically \(O(\log N)\) in a clear theoretical sense. Similarly, Appendix C cites mutual information bounds (Equation (17)) and uniformity loss (Equation (15)) mostly to justify the use of InfoNCE, but no concrete quantitative uniformity metrics are actually reported for the trained models. The theory discussion is more of a high-level recap than a substantive analysis of RetrievalFormer’s behavior.

9. **Some dependence on a single strong prior work (AttrFormer) for baseline protocol.**  
   The paper repeatedly says it follows Liu et al. (2025) for splits and baselines. While that is reasonable, it also raises questions around robustness: for example, AttrFormer’s MovieLens‑1M Recall@20 of 0.4128 in Table 1 is unusually high relative to other baselines, and the paper itself labels it an “outlier.” A brief sanity check (e.g., cross-referencing other published numbers or re-running a simpler method with the authors’ code) would strengthen confidence that results are not artifacts of dataset handling.

10. **Clarity and minor consistency issues.**  
   - Equation (1) uses \(W_m E_{f_m}(f_m)\) and concatenation over features to form \(H\), but later text introduces “text features” handled as multi-valued categorical without explicitly writing down the aggregation operator within \(E\).  
   - Section 3.5 mentions “one in-batch negative per positive example unless otherwise noted,” but Equation (9) clearly uses all \(B-1\) other items as negatives; this discrepancy is confusing and should be resolved.  
   - In the latency experiments, the hardware description is slightly inconsistent (an “ml.g6.xlarge instance” vs later “a single NVIDIA V100 GPU with 32GB memory”), which could be clarified.

Overall, the paper has several good ideas and solid engineering, but the conceptual novelty is moderate and some empirical arguments (especially regarding where performance loss comes from, and how RetrievalFormer compares to alternative scalable pipelines) remain underdeveloped.

## Potentially Missing Related Work
1. **Yadav et al., “Efficient k-NN Search with Cross-Encoders using Adaptive Multi-Round CUR Decomposition” (2023).**  
   This work focuses on efficient k-NN search in settings where cross-encoders are used for scoring, which is related to the paper’s theme of efficient nearest neighbor retrieval under deep models. It should be discussed in Section 2 when contrasting dual-encoder retrieval with approaches that approximate cross-encoder scoring and should help clarify how RetrievalFormer compares to methods that retain richer interaction functions but approximate their use at scale.

2. **Yadav et al., “Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization” (2022).**  
   Similar to the above, this paper deals with scalable nearest neighbor search for complex scoring models via factorization; it is directly relevant to the “reformulating recommendation as retrieval” narrative. It would fit naturally into the “Two-Stage and Retrieval Models in Recommenders” part of Section 2, framing RetrievalFormer as one point in the design space of efficient retrieval for expressive models.

3. **Miech et al., “Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers” (2021).**  
   This work uses transformer-based dual-encoders for retrieval efficiency, with a fast tower and a slower cross-encoder re-ranking. Citing it in Section 2 under “Approximate Nearest Neighbors for Recommendation” (or a short separate subsection on dual-encoder transformers in retrieval) would better align this paper with the broader literature on transformer-based dense retrieval and make clearer that the architectural pattern is well‑established in other domains.

4. **Gillick et al., “Learning Dense Representations for Entity Retrieval” (2019).**  
   A classic dual-encoder model in IR for efficient entity retrieval. Including it in Section 2 would help position RetrievalFormer’s contributions primarily in the recommender and cold-start feature-fusion space, acknowledging that the general dual-encoder + ANN pattern is well-known.

## Questions
1. **Accuracy vs. exact vs. approximate scoring.**  
   Can the authors provide an experiment, at least on one dataset (e.g., MovieLens‑1M), that compares: (a) exact top‑K search over all item embeddings, and (b) IVF-PQ retrieval with varying \(n_{\mathrm{probe}}\) and PQ configurations, reporting Recall@20 and NDCG@20 in each setting? This would directly quantify how much of the observed accuracy gap to SASRec is due to the dual-encoder formulation vs. ANN approximation error.

2. **Content-based and other cold-start baselines under LOOC.**  
   For the public datasets, can you report LOOC results for at least one or two content-based baselines (e.g., a pure item-tower dual-encoder, or a content-based KNN per item)? Even if these baselines are weaker overall, this would contextualize whether RetrievalFormer’s 25–35% drop from LOO to LOOC is a good outcome compared to standard content-based methods.

3. **Clarification of Mixed Negative Sampling implementation.**  
   In practice, how many uniformly sampled negatives per batch are added relative to in-batch negatives, and are they encoded online with the item tower or precomputed? Does your implementation match Equation (16) with weights \(w_j\), or is it a simpler unweighted mixture? A short algorithmic description would help others replicate the training behavior.

4. **Role and definition of “uniformity loss.”**  
   In Table 3, “Uniformity Loss Enabled/Disabled” yields a nontrivial difference in Recall@20. Is this an explicit additional term approximating Equation (15), or does “Disabled” mean that some piece of the InfoNCE or negative sampling strategy was changed? Please spell out the exact objective in each case.

5. **Ablation of shared embeddings on public datasets.**  
   You report a ~3% gain on MovieLens‑1M and a larger gain in the email dataset when using shared embeddings. Could you include a simple ablation in Table 3 (or a separate table) for Amazon Toys/Beauty showing the effect of shared vs. separate embedding tables? This would empirically support the extensive motivation given in Section 3.2.2 and Appendix B.2.

6. **Impact of interaction-type features and profile tokens.**  
   Figure 3 and Appendix H describe multiple strategies (profile-as-token, late fusion). Can you provide a small ablation table showing Recall@20 changes for (a) with vs. without interaction types, and (b) profile-as-token vs. profile-side-input vs. fusion, at least on one dataset? This would help quantify how much of the gain comes from the specific user tower design versus simply switching to a dual-encoder.

7. **Fairer system-level comparison to two-stage pipelines.**  
   Do you have any internal or public benchmarks where you compare RetrievalFormer as a single-stage retriever to a more typical “separate dual-encoder retriever + SASRec re-ranker” pipeline, both in terms of final Recall/NDCG and latency? Even a rough comparison would be valuable to assess if RetrievalFormer meaningfully simplifies the pipeline at comparable quality.

## Flag For Ethics Review
No ethics review needed.  

## Details Of Ethics Concerns
N/A.

## Soundness Rating
3: good.  
The methodology is technically sound and uses well-established components (transformers, contrastive learning, FAISS ANN), with reasonable ablations and multiple datasets. Some empirical arguments could be sharper, especially regarding the decomposition of accuracy loss sources and more complete baselines, but there are no obvious fatal flaws.

## Presentation Rating
3: good.  
The paper is generally clear and well-structured, with helpful figures (Figures 1–3) and detailed appendices. A few inconsistencies and overloaded terminology around “uniformity loss” and negative sampling should be fixed, and the related work on dense retrieval could be broadened, but overall exposition is above average.

## Contribution Rating
2: fair.  
The main conceptual move (dual-encoder + ANN for sequential recsys) is relatively standard in the broader ML/IR ecosystem, though less common in the specific sequential recommendation literature. The novel pieces are the attention-based heterogeneous feature fusion with shared embeddings and the rigorous LOOC protocol plus production case study. These are valuable but not transformative.

## Overall Rating
4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper addresses an important practical problem and presents a well-engineered solution with solid evaluation and nice insights on cold-start and latency. However, the conceptual novelty is moderate, the accuracy gap to strong baselines is nontrivial, and some empirical comparisons (especially around ANN approximation and alternative scalable pipelines) are incomplete. With additional experiments clarifying these points and broader positioning relative to existing dense retrieval work, this could become a strong contribution.

## Reviewer Confidence
4: confident.  
I am familiar with dual-encoder retrieval, transformers for recommendation, and ANN methods, and I carefully read the math, experiments, and appendices. There is some uncertainty around engineering details of the training pipeline that are not fully specified, but they are unlikely to change the overall assessment.