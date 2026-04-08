=== CALIBRATION EXAMPLE 20 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the detailed review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "VocSim: A Training-Free Benchmark for Zero-Shot Content Identity in Single-Source Audio" is accurate and specific. The abstract is well-structured and does not overclaim. The description of the two metrics (P@k, GSR with permutation calibration), the main finding (sharp OOD gap on blind speech), and the external validations is appropriately compressed.

One mild concern: the abstract states that the Whisper-based pipeline "yields strong zero-shot performance" and that VocSim "surfaces a consistent generalization gap"—these two claims are framed as separate contributions, but they are in some tension. On the blind sets, P@1 drops from 66.8% to 11.5% and the GSR lift collapses from 16.9 to 5.8 points. Whether calling P@1=11.5% "strong" or "nearly random" is a framing choice that deserves care. The abstract handles this reasonably, but readers may be confused.

---

### Introduction & Motivation

The motivation is clearly articulated: existing evaluation suites (HEAR, SUPERB) measure task-specific adaptability via fine-tuning or linear probing, leaving the *intrinsic* zero-shot geometry of frozen embeddings unmeasured. This is a legitimate gap. The connection to biological auditory systems as motivation is appropriate but slightly overstated—the benchmark does not actually attempt to replicate or formally test biologically grounded invariances, just class-level retrieval.

The contributions list is precise and includes an important clarification (lifted later from the main text) that models are *ranked by raw GSR within subsets*, not by lift-over-random. This is good. However, the claim that the paper "establishes state-of-the-art performance on the HEAR benchmark" (line 28, abstract; lines 635-646, main text) conflates zero-shot geometry with supervised linear probing. HEAR evaluates K-fold linear probing, not zero-shot retrieval. The finding is interesting but weakens the internal coherence of the "training-free" framing.

---

### Dataset (Section 3)

**Scope and coverage.** The aggregation of 19 corpora is thoughtful, and the three-domain coverage (human speech, animal vocalizations, environmental sounds) is genuinely broad. The exclusion of polyphonic material is principled and well-argued. The minimum class size of 6 is operationally necessary for P@k stability, and the authors acknowledge it.

**The blind test set is narrow.** Only 4 subsets (HW3, HW4, HU3, HU4—all from Shipibo-Conibo and Chintang) serve as the primary OOD evidence. These are exclusively low-resource *human* speech subsets. The paper's central finding—a "generalization gap"—is therefore a claim about generalization to unseen *languages* and *speakers*, not to unseen acoustic domains (e.g., unseen animal species, novel environmental sounds). The existing animal vocalization subsets (BS, BC, OC1) are held publicly available, so they are not true zero-shot. This limitation is acknowledged in the Discussion but not sufficiently flagged as a qualification of the main claim. The paper should be more careful: the gap may be specific to language-level or phonotactic generalization, not a universal failure of audio representations.

**Asymmetric preprocessing.** Top-frequency Zipfian filtering is applied to public speech corpora but *not* to the blind sets (Section D.6), with a reasonable justification. However, this means the blind sets have a different task structure (possibly skewed by function words in the transcriptions), and the difference in difficulty relative to the public sets cannot be cleanly attributed to OOD-ness versus preprocessing choices. The label permutation random baseline will absorb some of this, but the confound is real.

**Acoustic-to-label fidelity.** The paper combines sub-classes with very different semantic grain—phone-level (HP: 68 classes, avg. 0.09 s), word-level (HW1: 754 classes), utterance-level (HU1: 1245 classes), bird syllables (BS3: 46 classes), and full environmental events (ES1: 50 classes). This is intentionally varied, but the definition of "content identity" is stretched to cover very different notions of acoustic similarity. Phone identity and full-utterance identity are not the same thing, and it is unclear whether a single metric can meaningfully summarize performance across all these scales.

**Class granularity and P@k validity.** For subsets with very few samples per class (average 6.7 for HW2), P@5 is structurally ill-defined (there may be fewer than 5 true positives). The minimum class size of 6 guarantees that P@5 can reach 1.0 but makes P@5 trivially high for the smallest classes—essentially requiring retrieval of all class members. This is worth discussing more carefully.

---

### Metrics (Section 4.2)

**Precision@k.** Standard, well-understood, appropriate for dense class evaluation.

**Global Separation Rate (GSR).** The metric formula (NID − Avg_ID)/(NID + Avg_ID + ε) is novel and the design rationale (capturing global class boundary integrity) is reasonable. However, there are several concerns:

1. **Asymmetric aggregation (min vs. mean).** GSR computes NID as a *minimum* inter-class distance and Avg_ID as a *mean* intra-class distance. This is deliberately asymmetric and makes the metric sensitive to outlier inter-class points (a single near-miss with a point of a different class will collapse NID even if the class is otherwise well-separated). The authors justify this as measuring "worst-case boundary integrity," which is conceptually clear, but it means GSR may penalize models for a structurally different reason than P@k—not because global class structure is absent, but because boundary violations occur at individual outlier points. This is not fully acknowledged.

2. **High absolute GSR on blind sets despite collapsed lift.** The most puzzling reported finding is that the top model achieves GSR = 41.7% on public data and GSR = 39.4% on blind sets—a modest drop. Yet the lift-over-random collapses from 16.9 to 5.8 percentage points (micro-averaged). This implies the random permutation baseline is approximately 33.4% on public data and approximately 33.6% on blind data—nearly the same. The geometric reason for this should be stated explicitly. The current phrasing ("despite a superficially high absolute score, the model's ability to organize novel classes is only marginally better than random assignment") may give a misleading impression that GSR ≈ 39.4% is "near chance" when it is 5.8 points above a baseline that itself sits at ~33.6%. A clearer exposition with explicit baseline values in Table 2 would help. Reporting absolute GSR *and* random baseline *and* lift in the same table (or at minimum citing the random baseline values explicitly in the main text) is necessary.

3. **Cross-subset non-comparability.** The authors note that absolute GSR cannot be compared across subsets because permutation baselines differ. This is a significant limitation for a benchmark intended to aggregate scores. The macro-average of raw GSR across subsets (Table 2) is therefore potentially misleading if baselines vary substantially across subsets.

4. **Relationship to established metrics.** GSR bears a conceptual resemblance to the silhouette coefficient. The paper compares P@k and GSR to NMI, Purity, and ARI (Appendix G.1) and finds consistency, but does not formally compare GSR to the silhouette coefficient, which is a well-established global class separation metric. This comparison would strengthen or contextualize the case for GSR as a novel contribution.

---

### Results (Section 5)

**Whisper dominance.** The finding that Whisper large-v3 with mean time-frequency pooling and PCA outperforms WavLM, Wav2Vec 2.0, BEATs, CLAP, EAT, and AudioMAE in zero-shot retrieval is notable. However, the models compared differ substantially in size, pre-training data, and pre-training objective. Whisper large-v3 is a very large model trained on 680,000 hours of multilingual data with a supervised ASR objective—arguably better aligned with the *content identity* task (recognizing same phonetic content regardless of speaker) than contrastive or masked prediction objectives. The authors acknowledge this but do not provide any capacity-controlled comparison. The conclusion that Whisper "yields strong zero-shot performance" due to its architecture or pre-training *recipe* is at best weakly supported.

**Divergence from SUPERB.** The observation that SUPERB top-performers (WavLM, HuBERT, Wav2Vec 2.0 Large) do not top-rank on VocSim is presented as a finding reinforcing that SUPERB measures "supervised adaptability" vs. VocSim's "intrinsic geometry." This is reasonable, but an important confounder is that SUPERB rankings use fine-tuned or probed representations while VocSim uses frozen features. The divergence could reflect *this* difference rather than the "intrinsic geometry" story.

**OOD gap interpretation.** The finding that P@1 drops from 66.8% to 11.5% on blind sets is striking and important. But as noted above, only 4 subsets form the blind set, all human speech, and all from agglutinative low-resource languages that Whisper almost certainly has never seen in training. Whether this is evidence that "current embeddings retain structured geometry but fail to align it with novel class boundaries" (Discussion) or simply that Whisper has weak representations for languages totally absent from its training data is indistinguishable from the evidence.

**Layer-wise stability (Appendix H).** The report that performance is "remarkably stable across all 32 Whisper encoder layers" is interesting but not shown in the main paper. A brief discussion of this in the main text would strengthen reproducibility.

**DTW baseline.** The comparison with DTW re-ranking (Appendix H.3) showing no average gain over simple pooling is a useful negative result. It would be worth including a brief mention in the main text rather than relegating it entirely to an appendix.

---

### External Validation (Section 6)

**Avian perception alignment (80.9% triplet accuracy).** This is a compelling external validation. The comparison to inter-bird agreement (80–90%) is informative. However: (a) this is evaluated on the "high-consistency subset" of the Zandberg et al. dataset—the fraction this represents of the full dataset and the criterion for this subsetting should be stated; (b) no confidence intervals are reported for this 80.9% figure in the main text; (c) the comparison models for triplet accuracy are mentioned as baselines (Table 25 in appendix) but not summarized in the main text, making it hard to assess how competitive this result is.

**Mouse USV classification (99.49% strain, 53.1% individual).** Interesting results, but this uses a "high-frequency spectrogram frontend" *in addition* to Whisper embeddings. The 99.49% strain classification is likely driven primarily by the acoustic preprocessing frontend, which operates above 16 kHz—outside the range of Whisper's 16 kHz input. The authors note this ("appropriate frontends can recover information beyond this band") but this means the bioacoustic result is not purely a validation of the VocSim-favorable representations. It is misleading to present this as evidence that "the geometry favored by VocSim metrics translates into practical gains."

**HEAR benchmark.** Whisper EWMTF D100 achieves SOTA on HEAR tasks under K-fold *linear probing*. This is a genuinely interesting result—Whisper frozen features are highly linearly separable for downstream tasks. But it contradicts or at least is orthogonal to the zero-shot finding: on blind OOD speech, Whisper is nearly random, yet on HEAR tasks (which include labeled training folds) it is SOTA. The paper frames both as confirming VocSim's predictive validity. The generalization gap finding suggests HEAR's labels are what matter, not VocSim-style zero-shot geometry. This tension is underplayed.

---

### Discussion & Limitations (Section 7)

The discussion is candid about scope (single-source, no polyphonic audio) and the bird syllable labeling convention. The acknowledgment that the OOD gap "likely requires pretraining objectives that explicitly reward cross-domain content identity" is reasonable, if somewhat vague.

The "Threats to validity" paragraph notes that the heterogeneity of encoders prevents causal analysis of training regimes. This is appropriate. However, a key unacknowledged threat is the **hubness problem** in high-dimensional spaces (Radovanović et al., 2010, cited in the references but never discussed in the text). In high dimensions, a small number of "hub" points become the nearest neighbors of many other points regardless of class membership, biasing P@k upward for hubs and downward for anti-hubs. Whether PCA (which the best model uses) mitigates this, and whether the results for raw pooled features are affected, is not discussed.

Another unaddressed limitation: **per-subset PCA is applied to the full subset data before evaluation**. This is a transductive step that uses information from the evaluation data itself, even if labels are not used. For the blind test sets, this means the PCA projection adapts to the blind data distribution. While label-free, this is not strictly "training-free" in the classical sense, and it provides an advantage that would not be available in a true deployment scenario. The paper refers to this consistently as "label-free" but not as "transductive," which is slightly misleading.

---

### Writing & Clarity

The paper is generally well-written. A few genuinely confusing passages:

- The description of the GSR formula in the main text omits ε and describes the range as "normalized to a percentage" without explaining the normalization. The formal definition appears elsewhere and is clear, but the main text summary is imprecise.

- The dual use of "zero-shot" (meaning both "no labels used for evaluation" and "no fine-tuning") leads to some ambiguity, especially given that PCA is applied per-subset using unlabeled data and the HEAR results involve supervised probing.

- The abbreviation system (EWMTF D100, EWTF, ETF, etc.) is dense and requires constant reference to the glossary. While the glossary helps, the main text becomes difficult to parse for readers not fully conversant in the abbreviation scheme.

---

### Overall Assessment

VocSim addresses a genuine gap: existing audio benchmarks test fine-tuned or linearly probed representations, whereas the intrinsic zero-shot geometry of frozen embeddings is understudied. The benchmark aggregation is methodologically sound, the metrics are complementary and reasonably justified, and the external validations (avian perception, bioacoustics, HEAR) add real credibility. The main empirical finding—that the GSR lift-over-random collapses on blind OOD speech—is interesting and clearly quantified.

However, several issues limit the current submission. The central OOD finding rests on just 4 subsets (all human speech from 2 low-resource languages), making it impossible to distinguish language-specific generalization failure from a universal representation failure. The GSR metric's asymmetric design (min NID vs. mean Avg_ID) and its variable random baseline across subsets create interpretational difficulties that are only partially resolved by the permutation calibration. The claim that Whisper EWMTF D100 establishes SOTA on HEAR under linear probing is in tension with the zero-shot framing and the OOD gap finding. The mouse USV result confounds the Whisper embedding with a non-16 kHz frontend. The per-subset PCA is transductive, not fully training-free. And the model comparison lacks capacity control—Whisper large-v3 is substantially larger than competing models in the comparison. These are important clarifications and limitations that, taken together, suggest the contribution is valuable as a benchmark resource but requires more careful framing and additional experiments (e.g., truly held-out animal vocalization subsets as OOD data, a formal comparison of GSR to the silhouette coefficient, and a discussion of hubness) before the strongest claims can be substantiated. The paper is at the boundary of acceptance for ICLR; the benchmark itself merits publication, but the framing of several key claims should be tightened.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces VocSim, a large-scale, training-free benchmark designed to evaluate the intrinsic zero-shot content identity of frozen audio representations across 125k single-source clips from 19 diverse corpora. The authors propose two complementary geometric metrics—Precision@k for local coherence and a permutation-calibrated Global Separation Rate (GSR) for global class separability—and use them to benchmark a wide range of SSL, autoencoder, and foundation models. Key findings reveal that a simple Whisper-encoder pipeline with pooling, label-free PCA, and Spearman distance dominates zero-shot performance, while also exposing a sharp generalization gap on blind low-resource speech; the benchmark's validity is further supported by alignment with avian perception, improved fixed-feature bioacoustic classification, and new SOTA results on the HEAR benchmark.

### Strengths
1. **Clear gap identification and principled benchmark design**: The paper explicitly contrasts its zero-shot geometric evaluation with existing fine-tuning/linear-probing suites (HEAR, SUPERB), justifying the focus on intrinsic representation quality. The restriction to single-source audio cleanly isolates content similarity from separation artifacts, and the aggregation of 19 corpora across human speech, animal vocalizations, and environmental sounds provides structured stress-testing across duration, granularity, and recording realism (Section 3).
2. **Rigorous metric development and validation**: GSR is carefully formulated to compare nearest inter-class to average intra-class distances, and crucially, it is calibrated against label-permutation baselines to establish meaningful lift-over-random. The paper includes extensive robustness checks: layer-depth sweeps, pooling ablations, DTW re-rankings, label-noise sensitivity, and convergence with clustering metrics (NMI, Purity, ARI) and silhouette scores (Appendices F, G, H).
3. **Strong empirical findings and external validation**: The demonstration that Whisper + mean time-F pooling + PCA + Spearman distance outperforms specialized SSL models in frozen similarity is clear and practically useful (Table 2). The blind low-resource speech sets effectively reveal an OOD ceiling, providing actionable diagnostics for future representation learning. External validations (80.9% triplet accuracy aligning with zebra finch perception, strong mouse USV classification, HEAR SOTA) convincingly show that zero-shot geometric quality translates to real-world utility (Section 6).
4. **Excellent reproducibility and ethical transparency**: The paper provides deterministic pipelines, fixed seeds, detailed preprocessing steps (Algorithm 1), pre-training overlap audits (Appendix D.2), and clear plans for code/data/leaderboard release. The secure server-side evaluation for low-resource speech respects data sovereignty and IRB protocols, aligning with modern responsible AI practices (Ethics Statement).

### Weaknesses
1. **Potential confounding from Whisper’s web-scale pre-training exposure**: While the overlap audit is thorough, Whisper’s training on ~680k hours of web-scraped audio likely encompasses distributions highly similar to many VocSim public subsets. This makes it difficult to disentangle whether Whisper’s dominance stems from architectural properties or implicit data exposure. The blind sets mitigate this, but the public-set rankings may overstate "zero-shot" generalization in a strict sense (Appendix D.2, Section 4.2).
2. **GSR’s limited cross-subset interpretability**: The authors correctly note that absolute GSR values cannot be compared across subsets due to differing permutation baselines, and they rank models by raw GSR within each subset. However, this complicates macro-level aggregation for a unified leaderboard. Practitioners seeking a single score may find the reporting schema cumbersome without a normalized aggregation strategy (Section 4.2, Appendix G.2).
3. **Narrow coverage of recent architectural families**: The evaluated model suite focuses on classic SSL (Wav2Vec 2.0, WavLM, HuBERT), spectrogram transformers (BEATs, EAT, AudioMAE), and large foundation/multimodal models (Whisper, CLAP). Newer large audio-language models, open-weight encoders, or explicitly contrastive representation learners are underrepresented, which limits the breadth of architectural conclusions (Section 4.1).
4. **Heavy reliance on post-hoc PCA for top performance**: The best results require label-free, per-subset PCA compression. While this is practically useful, it suggests that raw frozen embeddings are not inherently optimized for similarity tasks. The paper does not deeply analyze whether PCA primarily mitigates hubness, removes pooling-induced noise, or compensates for misaligned embedding axes, leaving an open question about intrinsic encoder quality versus post-processing gains (Section 4.1, Appendix E).

### Novelty & Significance
The paper is highly novel and significant for the ICLR community. It shifts audio evaluation from supervised task adaptability to intrinsic, training-free geometric diagnostics—a paradigm well-established in vision and NLP but underdeveloped in audio. The introduction of permutation-calibrated GSR addresses a known methodological gap in geometric evaluation, and the systematic quantification of the OOD generalization ceiling provides a clear research target for future self-supervised objectives. The benchmark’s design, extensive ablations, and cross-domain validations meet ICLR’s standards for methodological rigor, reproducibility, and broad impact. The release of data, code, and a living leaderboard positions VocSim as a foundational tool that will likely catalyze new work in robust audio representation learning.

### Suggestions for Improvement
1. **Disentangle data exposure from architectural merit**: Add a controlled analysis or ablation that isolates the impact of Whisper’s training distribution. For example, compare performance on subsets with verified zero pre-training overlap versus those with likely overlap, or discuss findings in the context of scaling laws and known ASR pre-training biases to clarify how much of the gain is architecture vs. data coverage.
2. **Provide a practical aggregation scheme for GSR**: While respecting the authors’ caution against cross-subset GSR comparison, consider reporting an average lift-over-random score alongside raw metrics for leaderboard convenience. Clearly document its statistical interpretation so practitioners can use it as a quick diagnostic without overinterpreting absolute magnitudes.
3. **Broaden the architectural scope**: Include recent large open audio encoders, newer contrastive audio models, or audio-text joint embeddings to test whether the observed zero-shot patterns (e.g., Whisper’s dominance, Spearman’s robustness) generalize across newer architectural families and training objectives.
4. **Analyze the mechanistic role of PCA and distance metrics**: Provide a brief empirical study on why PCA boosts performance (e.g., variance explained, hubness reduction via Spearman vs. Euclidean, or alignment of latent axes). This would strengthen the "practical blueprint" claim and directly inform future pre-training or pooling strategies that aim to bake similarity-ready geometry into the encoder itself.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add audio-audio contrastive baselines** (e.g., audio-only SimCLR or CPC) because comparing primarily ASR/tagging models (Whisper, Wav2Vec) undermines the claim that VocSim evaluates intrinsic *similarity* geometry rather than task-specific feature utility.
2. **Evaluate HEAR tasks with training-free protocols** (e.g., k-NN) because using linear probing (supervised) contradicts the paper's core "training-free" premise and weakens the link between VocSim scores and downstream utility.
3. **Perform controlled OOD ablations** by systematically adding noise or channel shifts to public sets because without isolating specific factors, the reported "generalization gap" on blind sets remains a superficial observation rather than a diagnosed failure mode.
4. **Test animal subsets at native sample rates** (>16kHz) because resampling to 16kHz may artificially cap performance on bioacoustics, invalidating the claim that the benchmark fairly evaluates cross-domain generalization.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze OOD failure modes** via confusion matrices on blind speech sets because identifying whether errors are linguistic (phoneme) or acoustic (channel) is necessary to make the "generalization gap" finding scientifically actionable.
2. **Correlate GSR with downstream accuracy across all models** because showing this relationship only for the top performer is insufficient to validate GSR as a reliable predictor of representation quality.
3. **Quantify pre-training overlap impact** on public set performance because admitting likely overlap without measuring its effect makes it unclear if Whisper wins due to architecture or data leakage.
4. **Analyze metric sensitivity to class imbalance** because subsets range from 6 to 1000+ samples per class, and without proving robustness here, the aggregated averages may be skewed by large subsets.

### Visualizations & Case Studies
1. **Plot GSR vs. Downstream Accuracy scatter** for all models to visually demonstrate whether GSR consistently predicts real-world utility better than standard metrics like Silhouette.
2. **Overlay UMAP projections of Public vs. Blind embeddings** to visually expose the distribution shift that causes the performance collapse, making the OOD gap tangible.
3. **Display confusion matrices for blind speech subsets** to reveal specific phonetic or acoustic confusions, providing concrete evidence of where the representation fails.
4. **Plot performance vs. SNR/Duration for blind sets** to identify if the generalization gap is driven by specific acoustic conditions rather than linguistic OOD factors.

### Obvious Next Steps
1. **Release acoustic statistics for blind sets** (without labels) because reviewers cannot verify the OOD claim or reproduce the conditions without knowing the SNR, bandwidth, and duration distributions.
2. **Justify the "content identity" definition** across heterogeneous labels (phones vs. bird syllables) because aggregating these without deeper semantic analysis risks conflating distinct acoustic tasks.
3. **Expand ethics statement on low-resource languages** because using indigenous speech data for benchmarking requires explicit discussion of data sovereignty beyond a standard IRB citation.
4. **Include a training-free HEAR leaderboard** because maintaining consistency between the benchmark's training-free constraint and its downstream validation is essential for the paper's internal logic.

# Final Consolidated Review
## Summary
VocSim introduces a training-free benchmark for evaluating the intrinsic geometry of frozen audio embeddings on single-source audio across 125k clips from 19 corpora spanning speech, animal vocalizations, and environmental sounds. The paper proposes two complementary metrics (P@k and permutation-calibrated GSR), identifies a strong zero-shot baseline (Whisper encoder with time-F pooling and PCA), and reveals a sharp OOD generalization gap on blind low-resource speech. External validations include alignment with avian perception and SOTA results on HEAR linear probing.

## Strengths
- **Clear methodological contribution:** VocSim fills a genuine gap by evaluating frozen embeddings' zero-shot geometry, complementing fine-tuning/linear-probing suites like HEAR and SUPERB. The restriction to single-source audio cleanly isolates content identity from source separation confounds.
- **Rigorous metric development:** The GSR metric is thoughtfully designed (point-wise NID vs. Avg_ID) and calibrated against permutation baselines. Extensive ablations (layer sweeps, pooling strategies, DTW re-ranking, label-noise sensitivity, clustering metric correlations in Appendix G) validate the methodology.
- **Important empirical finding:** The sharp drop in P@k on blind sets (66.8%→11.5%) combined with the collapse in GSR lift-over-random (16.9→5.8 percentage points micro-averaged) provides a concrete, quantified diagnostic of where current representations fail.
- **Reproducibility and ethics:** Deterministic pipelines, fixed seeds, preprocessing documentation (Algorithm 1), pre-training overlap audits (Appendix D.2), and secure evaluation for low-resource speech data demonstrate responsible research practices.

## Weaknesses
- **Narrow OOD evidence base:** The blind test sets comprise only 4 subsets (HW3, HW4, HU3, HU4), all low-resource human speech from two languages (Shipibo-Conibo and Chintang). The "generalization gap" finding therefore applies specifically to language-level OOD, not to acoustic domain shifts like unseen animal species or novel environmental sounds. The paper should clearly frame this limitation rather than presenting it as a universal failure of audio representations.

- **Asymmetric preprocessing confounds comparison:** Zipfian token filtering is applied to large public speech corpora (e.g., removing high-frequency stop words) but *not* to the blind test sets (Section D.6). This creates a structural difference in task difficulty that cannot be cleanly attributed to OOD-ness versus preprocessing. The authors justify this ("to ensure a meaningful evaluation"), but the confound remains when interpreting the magnitude of the performance gap.

- **Per-subset PCA is transductive:** The PCA projection is fit on each subset's evaluation data (label-free but data-aware), which provides information not available in true deployment. This is correctly labeled "label-free" but technically not fully training-free for the blind sets. The paper is transparent about this, but the distinction matters for interpreting zero-shot claims.

- **External validation limitations:** (1) The mouse USV results use a high-frequency frontend beyond Whisper's 16kHz band, confounding the embedding quality with preprocessing. (2) The avian perception alignment uses only the "high-consistency subset" without clearly defining the selection criterion in the main text. (3) HEAR results use supervised linear probing, not training-free evaluation, making the link to VocSim scores correlational rather than causal.

- **Whisper dominance may reflect data exposure:** Whisper large-v3 was trained on ~680k hours of web-scale audio, likely covering distributions similar to many public VocSim subsets. The authors provide an overlap audit (Appendix D.2), but cannot fully disentangle architectural advantages from implicit data leakage. Capacity-controlled comparisons would strengthen the architectural claims.

## Nice-to-Haves
- A controlled ablation isolating data exposure from architecture (e.g., comparing models of similar size with different training data)
- Analysis of GSR's relationship to silhouette coefficient and more formal discussion of hubness effects in the main text (currently only briefly in Appendix E)
- Clarification of the "high-consistency subset" criterion for avian perception validation in the main text
- Release of acoustic statistics for blind sets (SNR, duration distributions) to help readers understand the OOD conditions

## Removed Points
*These points are flagged to be removed, treat them with caution*
- Harsh critic's complaint about "strong" vs "nearly random" framing: The paper reasonably presents both absolute scores and lift-over-random, making the distinction clear to attentive readers.
- Harsh critic's claim that HEAR SOTA results contradict the zero-shot framing: The paper is transparent that HEAR uses linear probing and presents this as transfer validation, not zero-shot performance. The two findings address different aspects of representation quality.
- Spark Finder's request for training-free HEAR evaluation: This contradicts HEAR's established protocol and would not provide a meaningful comparison to existing benchmarks.
- Spark Finder's request to test animal subsets at native sample rates: The paper acknowledges this limitation (Appendix L) and shows appropriate frontends can recover high-frequency information. Resampling is a standard practice for benchmark compatibility.

## Novel Insights
The paper's most novel finding is the divergence between local and global metrics on OOD data: P@k drops sharply (66.8%→11.5%) while GSR remains relatively stable (41.7%→39.4%), but the *lift-over-random* collapses. This reveals that embeddings retain geometric structure on novel classes but fail to align that structure with the correct semantic boundaries. This diagnostic—the separation of "structure preservation" from "boundary alignment"—is a genuinely useful contribution to understanding representation quality.

## Suggestions
- In the main text, briefly summarize why the blind sets are restricted to speech (ethical constraints on releasing low-resource language data) and explicitly frame the OOD gap as a finding about language-level generalization, not universal acoustic generalization.
- Add a table in the main paper reporting the permutation baseline values for key subsets alongside observed GSR, making the lift calculation transparent without requiring readers to dig into appendices.
- Consider adding a small "held-out animal vocalization" test set (if ethically feasible) to broaden the OOD evidence beyond speech, strengthening the claim about general representation failure.

# Actual Human Scores
Individual reviewer scores: [0.0, 8.0, 2.0, 2.0]
Average score: 3.0
Binary outcome: Reject
