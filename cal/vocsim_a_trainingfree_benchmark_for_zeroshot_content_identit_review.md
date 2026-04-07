=== CALIBRATION EXAMPLE 41 ===

# Harsh Critic Review
Now I have enough information to write a comprehensive review. Let me compose it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "VocSim: A Training-Free Benchmark for Zero-Shot Content Identity in Single-Source Audio" is accurate and descriptive. The abstract correctly characterizes the contributions: the 125k-clip benchmark, two training-free metrics (P@k and GSR), the key empirical result (Whisper encoder with TF-pooling + PCA), and the generalization gap finding. The claim of "state-of-the-art performance on the HEAR benchmark" in the abstract is slightly misleading—Section 6 clarifies this is achieved via the official *linear probing* protocol, which is not zero-shot. A single sentence of disambiguation would prevent a reader from assuming the SOTA result also holds under frozen, training-free evaluation. The mention of "80.9% triplet accuracy" as external validation is compelling but appears without context in the abstract; the connection to *VocSim scores predicting* this performance (rather than just Whisper being good) is not established here.

---

### Introduction & Motivation

The framing is clear and the gap is real: existing benchmarks (HEAR, SUPERB) measure task-specific adaptability via fine-tuning, not the intrinsic geometry of frozen embeddings. The biologically motivated framing (content identity across nuisance variation) adds depth. The restriction to single-source audio is a principled design choice that is well-argued.

One concern: the introduction states that VocSim measures whether embeddings "already express content identity without any task-specific heads or labels," but PCA—even when label-free—is fit *per-subset on the evaluation data itself*. This is not a pure zero-shot setup; the post-processing step adapts to each subset's distributional statistics. The introduction and contributions bullet should be more explicit that "label-free, per-subset PCA" is still data-dependent adaptation, even if it is unsupervised.

The contributions bullet about the "OOD gap" ("on blind low-resource speech, P@k drops sharply and GSR lift is small") is the strongest empirical finding of the paper and is well highlighted.

---

### Related Work

The related work is thorough and well-organized. The comparison to vision (CUB-200) and NLP (MMTEB) analogues is apt. The audit of existing audio benchmarks (HEAR, SUPERB, BirdSet, BIRB) correctly identifies the gap VocSim fills.

One missed comparison: the BEANS benchmark (Hagiwara et al., 2023) is cited but not discussed in detail. Given that BEANS also evaluates audio representations on animal sounds in a retrieval-adjacent setting, it deserves a sentence or two explaining how VocSim differs.

---

### Dataset (Section 3)

**Strengths:** The aggregation of 19 corpora spanning human speech, animal vocalizations, and environmental sounds is impressive in scope. The blind test set design (Shipibo-Conibo and Chintang from ACQDIV) is methodologically sound and ethically handled. The per-individual labeling of bird syllables is appropriate—syllable labels (A, B, C) have no cross-individual acoustic meaning.

**Concerns:**

1. **Asymmetric preprocessing**: Top-frequency words/utterances are filtered from large public speech corpora (LibriSpeech, AMI) but *not* from blind test sets. The justification (Zipfian skew in large corpora; low-resource languages lack this skew) is reasonable, but this means the benchmark difficulty structure differs between public and blind sets in a way that goes beyond language domain. Specifically, the public HW sets are filtered to retain a diverse, mid-frequency vocabulary, while the blind HW sets retain their natural distribution including potentially common function words. This could confound interpretation of the OOD gap—some of the P@k drop on blind sets may reflect the harder class-balance structure, not just OOD generalization.

2. **Single environmental sound dataset**: ESC-50 (ES1) provides only 2,000 clips in 50 classes. With 40 clips/class and perfectly balanced classes, this subset is structurally very different from the other 18 subsets (which have highly variable samples/class). Including only one environmental sound corpus—and one that is extremely well-structured—seems an underrepresentation of this domain, especially given the paper's claim of broad coverage.

3. **Only 4 blind test subsets**: The OOD generalization claim rests on 4 subsets (HW3, HU3, HW4, HU4) from only 2 language families. The reported averages over blind sets are averages of 4 numbers; adding confidence intervals over subsets is good practice, but the small-N makes any subset-level findings fragile. Claims about generalization to "low-resource speech" in general are strong given this constraint.

4. **The HW (Human Words) task structure**: Across public sets (HW1, HW2), classes are words (up to 1,324 distinct word types with only 6-7 samples per class in HW2). This is a many-class, few-shot-per-class setting that is structurally very different from the few-class, many-shot settings (e.g., bird syllables). P@k is sensitive to this class structure (as Figure 2b confirms), making cross-subset comparisons of absolute P@k values difficult.

---

### Method / Approach (Section 4)

**Feature representations:**

The selection of models is broad and representative, spanning raw-waveform SSL (Wav2Vec 2.0, WavLM), spectrogram transformers (EAT, BEATs, AudioMAE), and foundation models (Whisper, CLAP, EnCodec). However, **HuBERT** is listed in the Related Work as a top SUPERB model but is absent from all evaluation tables. Given the claim in Section 5 that "models that rank at or near the top of SUPERB (e.g., WavLM Large, HuBERT Large, wav2vec 2.0 Large) do not consistently top-rank on VocSim," this statement is only partially supported: HuBERT and wav2vec 2.0 are cited but HuBERT's results do not appear in Table 2. This is a notable omission that should be resolved.

**GSR metric design:**

The GSR formula—(NID − Avg_ID) / (NID + Avg_ID + ε), then normalized to [0%–100%]—uses an asymmetric comparison: a *minimum* (the nearest inter-class neighbor, NID) vs. an *average* (mean intra-class distance, Avg_ID). The authors correctly flag this asymmetry (Section 4.2, Appendix G.2), noting that the expected random baseline is not 50% but depends on the embedding's geometry. This is why they introduce permutation-based calibration.

However, this design choice has an underappreciated consequence: **NID is highly sensitive to outliers and hubness in high-dimensional spaces.** A single "hub" point from a different class that happens to land near a cluster boundary will drive NID down for many points, systematically depressing GSR. The Silhouette score (which also appears in Table 11 with ρ=0.82 correlation to GSR) uses the *mean* nearest-cluster distance rather than the *minimum* nearest-neighbor distance, which is more robust to this effect. The paper should discuss whether hubness-related artifacts are observed in the high-dimensional pooled representations, particularly since the paper cites Radovanović et al. (2010) on hubness in high-D spaces as motivation for Spearman distance.

**PCA as adaptation:**

The "top configuration" (EWMTF D100) fits PCA per-subset on the evaluation data. This is labeled "label-free" and is distinct from supervised fine-tuning, but it still uses test-distribution statistics. For the blind test sets, is PCA fit on the blind test data itself? If so, models that produce representations poorly aligned with the target domain may still benefit from PCA whitening the blind test distribution. The paper should be explicit about whether PCA for blind sets is fit on blind data or transferred from public-set components.

**Whisper's zero-shot framing:**

Whisper is an ASR model trained on 680,000 hours of speech. Its encoder is trained to produce representations useful for transcription, not for within-class acoustic similarity. The fact that it outperforms self-supervised models (WavLM, Wav2Vec 2.0) in this training-free setting is an interesting empirical finding, but the paper offers no mechanistic explanation. Does Whisper's strong performance stem from (a) scale, (b) its specific ASR objective enforcing phonetic discriminability, (c) its multilingual training enabling broad phonetic coverage, or (d) its encoder architecture? Without ablation, this remains unexplained and limits actionability. At minimum, comparing Whisper large-v3 to a smaller Whisper variant (small or medium) would help isolate scale effects.

---

### Experiments & Results (Section 5)

**Table 2 analysis:**

The reported values show EWMTF D100 at 66.8% P@1 on public sets vs. 11.5% on blind sets. The random baseline for P@1 is ~5.8% (public) and ~0.92% (blind). This means the model achieves roughly 11× random baseline on public sets but only 12× on blind sets (11.5/0.92 ≈ 12.5)—in fact the multiplicative lift is similar. The authors focus on the additive lift in GSR (16.9 points vs. 5.8 points), which does show meaningful collapse. However, for P@k, the drop is partly structural (many more classes in blind sets relative to samples per class), and the paper does not decompose how much of the P@k drop is attributable to (a) class-balance structure differences vs. (b) genuine representation failure on OOD phonotactics.

**Figure 2:**

Figure 2 (performance vs. subset properties) is informative. The finding that GSR is more stable than P@k across class-structure variations is useful. However, this raises a question: if GSR is stable across structural variation, why does it *also* show a large lift collapse on blind sets? The answer must be that both the model's GSR and the random baseline's GSR increase together on blind sets—and the random baseline is much closer to the model's GSR on blind data. This is a subtle but important distinction that is not fully articulated in the main text.

**Missing ablations:**

The paper discusses several ablations in appendices: layer-sweep over Whisper layers (stable across depth—notable!), DTW re-ranking (no gain), and label-flip noise (GSR more robust than P@k). These are appropriate and thorough. However, one important ablation is missing from the main text: **masked vs. unmasked pooling for Whisper.** The paper mentions this in Appendix D.8: they include padding frames in mean pooling, noting this has a "mild regularizing effect." Since Whisper pads short clips to a 30-second context window, the padding proportion varies dramatically across clips (a 0.07s bird syllable is mostly padding; a 5s environmental sound is mostly signal). Whether masked pooling significantly changes results—especially on short-duration subsets like BS1–BS4 (0.07–0.13s clips)—is a crucial question that deserves more prominence.

---

### Section 6: External Validation

This section is conceptually important but has significant methodological weaknesses that undermine the claim that "VocSim scores predict real-world utility."

**Avian perceptual similarity (80.9%):** The Whisper model achieves 80.9% triplet accuracy, "approaching inter-bird agreement (80–90%)." This is impressive. But the paper does not show that *VocSim scores correlate with* this triplet accuracy across models. To validate VocSim as a predictor, one needs a Spearman rank correlation: do models with high VocSim GSR/P@k also score high on triplet accuracy? Without this, the section only shows that Whisper is good at both tasks—not that VocSim scores are predictive.

**HEAR benchmark SOTA:** The Whisper EWMTF D100 achieves SOTA on HEAR under the official *linear probing* protocol. This is noteworthy, but the paper describes VocSim as a benchmark for "training-free" evaluation. The HEAR validation contradicts this framing because linear probing trains a task-specific classifier. More precisely: the HEAR result demonstrates that Whisper's frozen features are linearly separable for downstream tasks. This is related to, but not the same as, the zero-shot content identity that VocSim measures. This should be stated more carefully.

**Bioacoustics (mouse USV, 99.49%):** The mouse USV classification result is achieved "when paired with an appropriate frontend" (a high-frequency spectrogram frontend beyond 16 kHz). This is outside the VocSim evaluation pipeline, which standardizes to 16 kHz. Pairing Whisper with a specialized frontend is a practical recipe, but it is not a direct demonstration of VocSim's predictive validity—it demonstrates that frozen Whisper features + custom preprocessing are effective for a specific bioacoustic task.

---

### Writing & Clarity

The writing is generally clear and the paper is well-organized. The dual usage of GSR as a ranking metric (raw score within-subset) and an interpretive metric (lift over random for OOD diagnosis) is correctly explained but requires careful reading. Section 5's paragraph "Interpreting P@k and GSR jointly" and the near-duplicate text in Section 7 ("A generalization ceiling on OOD speech") cover this well, but the repetition feels like it could be condensed.

The abbreviation system (EWMTF D100, etc.) is dense. The glossary in Appendix C helps, but requiring readers to memorize two-letter codes for 11 models and 5 pooling variants creates a barrier in the main text.

---

### Limitations & Broader Impact

The limitations discussion in Section 7 is more honest than average for the genre: the single-source scope, the 16 kHz standardization, the heterogeneous encoder comparison, and the distinction from causal analysis are all acknowledged. The ethical treatment of blind low-resource language data (IRB, Nagoya Protocol, server-side evaluation) is exemplary.

One limitation that deserves mention: the benchmark's "content identity" task is defined by source corpus labels, which were not always designed for acoustic similarity evaluation. For example, HW classes are word types, but speakers and prosodic conditions vary; the task implicitly rewards representations that are speaker-invariant but prosody-agnostic. Whether this is the right definition of "content identity" for human words is debatable and is not discussed.

---

### Overall Assessment

VocSim makes a genuine contribution to the audio representation evaluation literature. The benchmark fills a real gap—the intrinsic, training-free geometry of frozen audio embeddings has been undercharacterized—and the 19-corpus aggregation with blind OOD test sets is a thoughtful design. The permutation-calibrated GSR metric is a methodologically principled addition to the evaluation toolkit, and the core finding (that the GSR lift over random collapses on OOD data despite a superficially high absolute score) is important and well-documented. The reproducibility commitments (code, data, leaderboard) are strong.

However, several weaknesses need to be addressed before this work fully delivers on its promises. First, the omission of HuBERT from Table 2 leaves a claim in Section 5 partially unsupported. Second, the external validation section (Section 6) does not actually demonstrate that VocSim *scores predict* downstream utility—it shows that Whisper is good at multiple tasks, which is a different claim. Third, the PCA adaptation per-subset is a form of data-dependent processing that complicates the "zero-shot" framing and needs clearer treatment, especially for blind sets. Fourth, the GSR metric's sensitivity to hubness effects in high-D spaces, particularly for short-clip subsets where most of the Whisper representation is padding, is unexplored. Finally, the OOD test relies on only 4 subsets from 2 low-resource languages, making it narrow as a test of "generalization" in general. These concerns are material but most are addressable with additional experiments and clarification. Given ICLR's standards for both methodological rigor and empirical breadth, the paper is a borderline accept contingent on resolving the HuBERT evaluation, strengthening the predictive validity analysis with explicit rank correlations across models, and clarifying the zero-shot framing around PCA.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **VocSim**, a training-free benchmark designed to evaluate the zero-shot content identity capability of audio representations across 125,382 single-source clips from 19 diverse corpora. The authors propose two key evaluation metrics: **Precision@k** (local neighborhood purity) and a novel **Global Separation Rate (GSR)** (global class separability) calibrated against permutation baselines. Their evaluation reveals that a simple frozen Whisper encoder pipeline yields strong zero-shot performance but identifies a critical generalization gap on low-resource, out-of-distribution speech, where class structure barely exceeds chance. The benchmark's validity is further supported by alignment with avian perceptual judgments and state-of-the-art results on the HEAR benchmark.

### Strengths
1.  **Rigorous Evaluation Paradigm:** The paper correctly identifies a gap in existing benchmarks (e.g., SUPERB, HEAR) which focus on supervised adaptability rather than intrinsic, training-free geometry. The inclusion of a "blind" low-resource test set allows for a genuine stress test of out-of-distribution (OOD) generalization that is rarely available in audio benchmarks.
2.  **Novel Metric Construction:** The **Global Separation Rate (GSR)** is a well-motivated addition to the evaluation toolkit. By comparing Nearest Inter-class Distance (NID) to Average Intra-class Distance (Avg_ID) and calibrating via label permutation, GSR provides a robust, geometry-aware check against random baselines, addressing potential issues with absolute metric values on different embedding manifolds.
3.  **External Validation:** The validation against **avian perceptual similarity** (80.9% triplet accuracy) is a particularly strong contribution. It moves beyond task-specific performance (like classification accuracy) to measure biological plausibility and intrinsic similarity, which is crucial for general-purpose audio intelligence.
4.  **Reproducibility and Transparency:** The authors provide code, data processing pipelines (including specific filtering logic like frequency filtering for Zipfian distributions), and a plan for a public leaderboard. The detailed ablation studies in the appendix (e.g., pooling strategies, layer depth) ensure that findings are not artifacts of specific implementation choices.

### Weaknesses
1.  **Access Barrier for Blind Set:** While the blind test set is ethically necessary for the low-resource languages (Shipibo–Conibo and Chintang), the evaluation protocol relies on a "secure, server-side process." This creates a higher barrier to submission compared to fully open benchmarks, potentially slowing community iteration on the OOD gap identified.
2.  **Theoretical Grounding of GSR:** While GSR is empirically robust (shown via ablation and correlation analysis), the derivation is heuristic. A deeper theoretical discussion on why the NID/Avg_ID ratio is superior to established structural metrics like Silhouette Score or Cluster Separability Index would strengthen the methodological claim, as Silhouette is already widely used for this purpose.
3.  **Scope Limitation (Single-Source):** The strict exclusion of polyphonic mixtures limits the direct applicability to real-world scenarios (e.g., music, cityscapes). While justified as isolating content representation, a discussion on how these representations might behave on multi-source audio would help contextualize the "good" performance found on clean, single-source data.

### Novelty & Significance
**Novelty:** The paper is significant primarily as a **methodological contribution** rather than a model architecture proposal. It introduces a specific benchmark infrastructure (VocSim) to solve a specific problem under-evaluated in the community: the *intrinsic zero-shot geometry* of audio embeddings. The GSR metric is also a novel component of this evaluation suite.

**Significance:** The findings challenge the assumption that foundation models (like Whisper) inherently learn robust content identity across domains. Demonstrating a "generalization ceiling" on low-resource speech is a critical insight for the field, suggesting current pre-training objectives may not adequately capture phonetic content invariance without linguistic adaptation. The successful transfer to biological perception benchmarks adds high significance to the work.

### Suggestions for Improvement
1.  **Clarify Blind Set Submission Process:** To maintain community engagement, the authors should detail exactly how external researchers can contribute models to the blind leaderboard (e.g., Docker container submission, API endpoint). This reduces ambiguity and maintains benchmark integrity without exposing ground truth.
2.  **Expand the GSR Discussion:** Briefly compare GSR mathematically to the Silhouette Coefficient. While the paper states they measure similar properties, explicitly showing where they diverge (e.g., on outliers or specific manifold shapes) would solidify GSR's value proposition.
3.  **Visualize the Generalization Gap:** Include a UMAP or t-SNE visualization contrasting the embedding space of the public sets versus the blind OOD set for the top model. Visualizing how the "class clusters" dissolve or collapse in the OOD set would intuitively convey the severity of the generalization gap mentioned in the text.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Cross-Domain PCA Ablation:** The top configuration fits PCA on each evaluation subset, effectively adapting to the test distribution. Evaluate performance using PCA weights fitted on a source domain and applied frozen to the blind target sets to verify the generalization gap isn't masked by test-time adaptation.
2. **OOD Downstream Correlation:** The claim that VocSim predicts utility relies on in-domain HEAR results. Demonstrate a correlation between VocSim GSR Lift and performance on downstream tasks trained on the blind OOD sets to prove the benchmark predicts true out-of-distribution transfer.
3. **Pre-training Data Proximity Analysis:** The "blind" low-resource speech claim assumes no overlap with Whisper's 680k-hour web corpus. Provide a quantitative analysis comparing the blind sets to common web-scraped speech data to substantiate the "true OOD" status.
4. **Polyphonic Robustness Check:** Since the goal is "general-purpose audio representations," evaluate top models on a polyphonic subset to show that single-source zero-shot geometry correlates with performance in realistic, overlapping acoustic scenes.

### Deeper Analysis Needed (top 3-5 only)
1. **Random Baseline Geometry Discrepancy:** The empirical random baseline for GSR is significantly higher on blind sets (33.7%) than public sets (24.9%). Analyze the intrinsic geometric properties driving this difference to ensure the "lift collapse" isn't an artifact of the blind set's embedding space structure.
2. **P@k vs. GSR Divergence Explanation:** Local retrieval (P@1) collapses (66% to 11%) while Global Separation (GSR) remains relatively stable (41% to 39%). Explain this divergence mechanistically—does the model preserve global manifold structure while failing local neighborhood coherence?
3. **Class Imbalance Control:** The blind subsets have different samples-per-class distributions compared to public subsets. Control for this by subsampling public sets to match blind set statistics, ensuring the performance drop is due to domain shift rather than metric sensitivity to class frequency.
4. **Metric Sensitivity to Cluster Density:** Analyze how GSR behaves when class clusters have varying densities, as the blind sets might exhibit different cluster compactness that artificially inflates the random baseline.

### Visualizations & Case Studies
1. **Public vs. Blind UMAP Comparison:** Provide side-by-side UMAP projections of the same model on a Public subset vs. a Blind subset to visually verify whether the "generalization gap" manifests as cluster overlap or dispersion.
2. **Failure Case Spectrograms:** Display spectrograms of blind set queries where the nearest neighbor is incorrect, juxtaposed with successful public set queries, to identify specific acoustic nuisances driving the zero-shot failure.
3. **GSR Lift vs. OOD Accuracy Scatter:** Plot GSR Lift-over-Random against downstream classification accuracy on OOD tasks for all evaluated models to visually validate the claim that Lift is a predictor of real-world utility.

### Obvious Next Steps
1. **Freeze Normalization Protocol:** Establish a standard protocol where normalization/PCA parameters are fixed from a source domain to prevent test-time adaptation leakage in future benchmark submissions.
2. **Polyphonic VocSim Extension:** Extend the benchmark to include mixed-source audio to test separation-invariant content identity, as single-source is a narrow slice of audio intelligence.
3. **Pre-training Objective Dissection:** Systematically evaluate how specific pre-training objectives influence zero-shot geometry to guide future model design beyond just benchmarking existing encoders.
4. **Longitudinal Blind Set Expansion:** Add new blind subsets periodically to prevent leaderboard overfitting, as the current 4 blind subsets may become de facto training targets.

# Final Consolidated Review
## Summary

VocSim introduces a benchmark for evaluating zero-shot content identity in single-source audio representations across 125k clips from 19 diverse corpora. The paper proposes two training-free metrics—Precision@k and a permutation-calibrated Global Separation Rate (GSR)—and identifies a critical generalization gap: on blind low-resource speech, local retrieval (P@k) collapses sharply while GSR lift over random remains marginal, indicating that current embeddings organize novel classes only slightly better than chance. The benchmark includes blind test sets from low-resource languages (Shipibo-Conibo and Chintang) and demonstrates external validity via avian perceptual alignment (80.9% triplet accuracy) and SOTA HEAR benchmark results.

## Strengths

- **Addresses a genuine gap in audio representation evaluation**: Existing benchmarks (HEAR, SUPERB) measure supervised adaptability via fine-tuning, not the intrinsic geometry of frozen embeddings. VocSim provides a principled training-free evaluation that probes whether embeddings already organize content identity without task-specific adaptation.

- **Methodologically sound design for OOD testing**: The inclusion of blind test sets from genuinely low-resource languages (protected under IRB-approved protocols) enables rigorous out-of-distribution evaluation that goes beyond the typical train/test splits of public corpora. The permutation-calibrated GSR provides a geometry-aware baseline that accounts for embedding space structure.

- **Clear and important empirical finding**: The paper documents a generalization ceiling—despite strong public-set performance (P@1 = 66.8%), models achieve only marginal GSR lift over random on blind OOD data. This quantifies precisely *where* and *how badly* current embeddings fail to generalize, which is more actionable than aggregate metrics.

- **Comprehensive ablation studies**: The appendices include layer-wise sweeps, DTW re-ranking comparisons, label-noise robustness tests, and pooling strategy comparisons. The finding that simple temporal pooling matches or exceeds DTW re-ranking (Appendix H.3) is practically useful.

- **Strong reproducibility commitments**: Code, data, and a public leaderboard are promised. The detailed preprocessing pipeline (Algorithm 1) and model configuration glossary (Appendix C) enable replication.

## Weaknesses

- **HuBERT evaluation missing despite claims in the paper**: Section 5 states that "models that rank at or near the top of SUPERB (e.g., WavLM Large, HuBERT Large, wav2vec 2.0 Large) do not consistently top-rank on VocSim," but HuBERT results do not appear in any evaluation tables (Tables 18–24). WavLM and Wav2Vec 2.0 are evaluated; HuBERT is not. This leaves the comparative claim partially unsupported and should be clarified—either by including HuBERT results or by correcting the text.

- **External validation does not establish predictive correlation**: Section 6 shows that Whisper achieves strong results on avian perception (80.9%) and HEAR, but this does not demonstrate that *VocSim scores predict* downstream utility. To validate the benchmark as a predictor, the paper would need to show that models with higher VocSim GSR/P@k also achieve higher triplet accuracy and downstream accuracy *across the model zoo*. Currently, the section only shows that one model (Whisper) performs well on multiple tasks—not that VocSim scores are predictive across models.

- **"Zero-shot" framing complicated by per-subset PCA**: The top configuration fits PCA separately on each evaluation subset, including blind sets. While labeled "label-free," this is still data-dependent adaptation—PCA learns the covariance structure of the test distribution itself. The paper acknowledges this in passing but should more explicitly discuss whether this affects the interpretation of OOD results and whether PCA weights transferred from public sets would yield different conclusions.

- **GSR's theoretical properties relative to established metrics not deeply analyzed**: The paper claims GSR is a "robust global metric" but provides limited theoretical justification for preferring NID/Avg_ID ratio over Silhouette Score (which achieves ρ=0.82 correlation with GSR in Table 11). More importantly, GSR uses a *minimum* inter-class distance (NID), making it potentially sensitive to hubness effects in high-dimensional spaces. The paper cites Radovanović et al. (2010) on hubness for Spearman distance justification but does not analyze whether hubs affect GSR differently across models or subsets.

- **Limited environmental sound coverage**: Only one environmental sound corpus (ESC-50, 2,000 clips) is included among the 19 subsets, despite environmental sounds being one of the three target domains. The other 18 subsets span speech and animal vocalizations more comprehensively.

- **Narrow OOD test scope**: The blind test sets comprise only 4 subsets from 2 language families. Generalization claims about "low-resource speech" rest on this limited foundation. Adding just 1–2 more blind sets from diverse language families would substantially strengthen these claims.

## Nice-to-Haves

- A cross-domain PCA ablation: report blind-set performance using PCA weights fitted on public sets (transferred frozen) rather than fit independently per blind set, to isolate test-time adaptation effects.

- Explicit correlation between VocSim scores and downstream OOD performance across the model zoo (e.g., scatter plot of GSR lift vs. HEAR accuracy across all evaluated models).

- Side-by-side UMAP visualization of a representative public subset versus a blind subset to visually illustrate the generalization gap.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Claim that the HEAR validation "contradicts the zero-shot framing"**: The paper correctly describes HEAR as using linear probing (not zero-shot), and this is explicitly stated in Section 6. There is no contradiction—just a clarification that frozen features are linearly separable.

- **Criticism that the avian perception result lacks context in the abstract**: The abstract explicitly states that "top embeddings predict zebra finch perceptual similarity (80.9% triplet accuracy)" as external validation. This is not out of context—it is presented precisely as validation.

- **Criticism about "only 4 blind test subsets" as too few for generalization claims**: While sample size is limited, the paper correctly reports bootstrap confidence intervals and permutation baselines. The concern is valid but has been addressed to reasonable standards for an ICLR paper; this is a matter of degree rather than a fundamental flaw.

- **Request for polyphonic robustness testing**: The paper explicitly scopes VocSim to single-source audio "to isolate content representation from source separation confounds" (Section 7). Requesting polyphonic evaluation is scope creep, not a legitimate weakness of this contribution.

- **Claim that GSR needs "deeper theoretical grounding"**: GSR is empirically validated against permutation baselines, correlates strongly with established metrics (ρ=0.82 with Silhouette), and is mathematically well-defined. Demanding additional theoretical justification beyond empirical soundness is an excessive standard not applied to other benchmarks.

## Novel Insights

The paper's most significant insight is the *divergence between local and global metrics on OOD data*: local retrieval (P@k) collapses dramatically (66.8% → 11.5% on blind sets) while the GSR *lift over random* collapses more informatively (16.9 → 5.8 points). This reveals that current embeddings retain structured geometry on novel classes but fail to align that geometry with correct class boundaries—a subtle but important distinction that would be invisible to absolute metric comparisons. The finding that Spearman distance consistently outperforms Cosine and Euclidean across configurations suggests that rank-based similarity is particularly suited to content-identity geometry in high-dimensional embedding spaces.

## Suggestions

- Add HuBERT results to the evaluation tables or revise the comparative claim in Section 5 to accurately reflect which models were evaluated.

- In Section 6, add a figure showing Spearman rank correlation between VocSim GSR/P@k and (a) avian triplet accuracy across models, and (b) HEAR task accuracy across models, to directly support the claim that VocSim scores *predict* downstream utility.

- Clarify in the main text whether PCA for blind sets is fit on blind data itself or uses weights transferred from public sets, and discuss implications for interpreting the OOD generalization gap.

- Add one more blind subset from a different language family (if possible under ethical constraints) to broaden the OOD claim.

# Actual Human Scores
Individual reviewer scores: [0.0, 8.0, 2.0, 2.0]
Average score: 3.0
Binary outcome: Reject
