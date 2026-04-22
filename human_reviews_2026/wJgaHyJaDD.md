# TPOUR: Temporal Preference Optimization for Unsupervised Retrieval

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Unsupervised retrievers offer scalability by learning semantic similarity from unlabeled documents via contrastive learning. However, they struggle to capture the temporal relevance, often retrieving semantically related but temporally misaligned documents--an important aspect when a document collection spans multiple time periods (e.g., For the query "Who is the president in 2019?" retrieving from related documents spanning 2018–2025 introduces temporal ambiguity if relying solely on semantics). Existing methods rely on supervised training with explicit timestamps, which are not always feasible. We propose TPOUR (Temporal Preference Optimization for Unsupervised Retriever), which integrates our novel training method Temporal Retrieval Preference Optimization (TRPO). TRPO reinterprets preference learning in the temporal dimension, guiding the retriever to favor temporally aligned documents. TPOUR constructs temporally aligned and misaligned document pairs by leveraging document corpora collected at different times and trains the retriever without supervision to prioritize temporally aligned over misaligned documents. Furthermore, TPOUR generalizes to unseen time periods by interpolating time vectors, enabling continuous temporal alignment. Experiments on temporal QA with a mixed-timestamp document collection show that TPOUR outperforms both unsupervised and supervised baselines. Compared to Nomic Embed v2 MoE, TPOUR Contriever improves nDCG@5 by +7.13 (+23.5%) on explicit and +7.76 (+25.5%) on implicit queries on average.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TPOUR (Temporal Preference Optimization for Unsupervised Retriever), a novel method designed to solve the problem of temporal misalignment in scalable, unsupervised retrieval by teaching the retriever a time bias without labeled data. Current unsupervised dense retrievers fail to distinguish between semantically relevant documents that are temporally aligned and those that are misaligned. TPOUR addresses this by integrating a new training signal, Temporal Retrieval Preference Optimization (TRPO), into contrastive learning. TRPO utilizes unlabeled versioned corpora (e.g., historical Wikipedia snapshots) to create implicit preference pairs, teaching the retriever to prioritize documents from a temporally aligned version over a misaligned version, thereby learning a continuous temporal preference from content updates. TPOUR substantially outperforms both unsupervised and supervised baselines on temporal QA tasks and demonstrates that Time Vector Interpolation allows it to generalize its temporal preference to intermediate, unseen time periods without requiring full retraining.

### Strengths
The core strength lies in how TPOUR achieves temporal awareness without requiring costly human-labeled data or explicit time annotations on every document. Traditional temporal retrieval demands supervised training with explicitly timestamped relevance scores, which is expensive and unscalable. TPOUR cleverly addresses this by introducing a Temporal Retrieval Preference Optimization (TRPO) loss and adapting the DPO framework to retrieval. This simple but ingenious mechanism enables the dense retriever to learn a complex temporal bias directly from changes in semantic content over time.

In addition, TPOUR provides a foundational fix by embedding a temporal preference directly into the latent embedding space. This allows the retriever to efficiently distinguish between documents that are merely semantically similar (low temporal score) and those that are both semantically similar and temporally correct (high temporal score), thereby significantly improving retrieval precision for time-sensitive queries. This fundamental fix moves dense retrieval closer to reliable, real-world deployment.

### Weaknesses
1. Mismatch Between Training Signal and Evaluation Benchmark
>TPOUR intentionally trains the model on implicit semantic drift (changes in content across document versions). However, the primary evaluation uses questions that require alignment with explicit temporal anchors ("in 2019"). This raises questions about whether the achieved performance truly reflects a learned temporal map or simply an effective scoring bias toward the document version containing the latest relevant semantic update. Closing this conceptual gap requires further demonstration.

2. Ambiguity of Generalization Results (Figure 5)
> The correlation in Figure 5 is not visually supported. Although a rising regression line is shown, the wide spread of data points does not convincingly demonstrate a strong correlation between a dataset's age and its optimal timing interpolation. This ambiguity weakens the conclusion that TPOUR shows a reliable general time sensitivity in external datasets.

3. Missing Diagnostic Evidence for Foundational Problem
> The paper asserts that time-unaware retrievers tend to retrieve semantically relevant but temporally misaligned documents, which is the central problem TPOUR is designed to solve. However, the paper does not provide a dedicated diagnostic analysis to formally quantify and prove the severity of this foundational problem in a baseline model. Consequently, in the absence of this explicit diagnostic proof, the necessity and added complexity of the TRPO optimization are not adequately anchored to a rigorously quantified problem. This weakens the overall justification and motivation for introducing the complex temporal preference framework.

### Questions
1. Since the model is trained on implicit semantic changes across document versions, how can the authors provide a dedicated diagnostic evaluation (e.g., using a different benchmark, or a targeted analysis of the retrieved documents) to verify that the model has effectively learned to distinguish content updates over time?

2. Could the authors please provide diagnostic evidence that rigorously quantifies the severity of temporal misalignment in baseline models, thereby justifying the necessity and added complexity of the TRPO optimization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles temporal misalignment in unsupervised dense retrieval. The authors propose TPOUR, which augments MoCo-style contrastive training with a preference loss (TRPO) that encourages a retriever to favor temporally aligned documents over temporally misaligned ones by leveraging multiple snapshots of the same corpus (Wikipedia) collected at different times. They further introduce “time vectors” extracted from fine-tuned models at different timestamps and linearly interpolate them to generalize to intermediate time periods without retraining. Experiments on custom retrieval versions of SituatedQA (2018–2021) and RealTimeQA (Jan–Dec 2023) show sizable gains over Contriever and other baselines, particularly on queries with temporal intent. They also demonstrate improvements in timestamp prediction with a mixture of TPOUR and a correlation between the creation year of the BEIR dataset and the optimal interpolation weight.

### Strengths
1. The problem is well-motivated and clearly formulated. It tackles temporal misalignment in unsupervised retrieval, an important and under-addressed challenge.

2. The method is simple and effective. It combines contrastive learning with a DPO-style preference loss guided by similarity gaps and integrates smoothly into existing training pipelines.

3. The time-vector interpolation is a neat and practical idea. It is supported by experiments and enables time alignment without retraining, making deployment easier.

4. The empirical analysis is thorough. The paper covers yearly and monthly settings, explicit and implicit queries, timestamp prediction, and a BEIR case study, complemented by qualitative examples, distribution visualizations, and clear reproducibility details.

### Weaknesses
1. Dataset construction is central to the method, but key details are missing. For example, the paper should clearly specify how preference triplets (Q, D^t, D^{t'}) are sampled, how content changes across snapshots are detected, and related implementation choices. In addition, the appendix assumes topics across years are similar; while intuitively plausible, this assumption requires empirical validation.

2. The baseline comparisons are not fully fair. According to the appendix, baselines rely on public checkpoints and are not trained or adapted on the same corpora, which may give the proposed method a domain/time adaptation advantage.

3. The model may have learned temporal shortcuts or biases. Although Appendix E.6 shows some robustness, it does not rule out reliance on spurious time-correlated cues.

4. A substantial amount of essential information is placed in the appendix, which reduces the readability and self-containedness of the main text.

### Questions
1. Could you clarify how preference triplets are formed and how temporal content changes are identified, and consider adding a brief sensitivity analysis to show robustness to these design choices? Additionally, please provide evidence that the assumption of topical similarity across years holds in practice.

2. To ensure fair comparison, could you discuss the extent of time/domain adaptation applied to baselines and, where feasible, include adapted or better‑tuned baselines—or justify why such adaptation is not practicable?

3. Can you demonstrate that the method’s gains are not driven by explicit or implicit temporal cues？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an unsupervised retrieval framework called TPOUR to address the “temporal misalignment” problem in traditional retrieval models when handling time-sensitive queries.

### Strengths
The paper’s main contribution is transferring the idea of DPO to unsupervised retrieval (TRPO) and combining it with time vector interpolation, offering a solution to the problem of “temporal awareness.”

### Weaknesses
1. Unclear “Unsupervised” Claim
Although the method is presented as unsupervised, it relies on document timestamp metadata to construct aligned vs. misaligned preference pairs. This constitutes weak supervision, and the paper currently does not clearly acknowledge or justify this discrepancy.

2. Evaluation Pipeline May Introduce Retrieval Bias
The construction of evaluation document sets uses Contriever itself to retrieve candidate documents before filtering. This risks closed-loop bias, potentially favoring methods architecturally similar to Contriever and inflating gains.

### Questions
Could the authors clarify whether TPOUR should be categorized as unsupervised, self-supervised, or weakly supervised?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
TPOUR proposes a training methodology for time-aware unsupervised retrieval by adding a temporal preference signal (TRPO) on top of MoCo contrastive learning. In particular, queries are paired with (aligned, misaligned) documents drawn from Wikipedia snapshots at different dates; the model is trained to increase similarity to aligned items while decreasing it for misaligned ones. The paper also
adapts “time vectors” (which record differences between fine-tuned weights on distinct periods) to bi-encoders, enabling interpolation across intermediate timepoints without retraining, and builds a mixture-of-TPOUR classifiers for timestamp prediction. On SituatedQA and RealTimeQA (customized for retrieval) and a BEIR case study, the method reports consistent nDCG@5/10 gains over baselines (Contriever, DPR, Nomic Embed v2 MoE, TimeR4), smooth performance peaks at interpolated α matching evaluation time, and improved year/month
timestamp prediction.

### Strengths
The paper proposes a clear and grounded integration of temporal awareness into unsupervised retriever training, since TRPO re-frames preference learning along the time axis while preserving standard contrastive semantics. Despite its simplicity, the idea does not require manual labeling and is reproducible. The usage of time-vector interpolation for bi-encoders is useful in practice. Performance peaks align with test time across years/months, achieving temporal generalization without per-period retraining. The paper presents a good experimental evaluation. In particular, it employs temporal QA with explicit/implicit timestamps and a BEIR analysis indicating temporal sensitivity and enabling specific α by creation date; Results show consistent improvements over baselines. The usage of mixture of retrievers for auxiliary timestamp prediction is effective to encode usable temporal signals (e.g., 76.56% year accuracy vs. 50.18% baseline).

### Weaknesses
Dataset construction for “gold” documents retrieves top-k with Contriever and then filters by answer presence. Such a procedure may lead to evaluation bias, since systems may become similar to the constructor and under represent hard/rare temporal cases. The paper mentions  an alternative retriever check, but it seems to me that it is necessary a stronger, retriever-agnostic construction or multi-constructor consensus to increase trust. The paper needs to be improved regarding statistical analysis and significance. There is no multi-seed variance, confidence intervals, or significance tests across the main tables; sensitivity to random seeds and snapshot choice (details about Wikipedia dump) are not clear. The selection of the α parameter is heuristic (based on known test time) and it is not clear a practical and sound strategy to infer α at inference time without test-time leakage (beyond the separate timestamp-prediction head). A unified, end-to-end approach seems to be necessary. The mixture of TPOUR timestamp predictor is compared to a single-retriever baseline with adjusted classifier parameters, but it still benefits from multiple specialized encoders. It should be necessary to include a capacity-matched or distilled single-encoder control to clarify the source of the gains. The experimental ablations need to be extended beyond λ,  and include, for instance, contrastive-only vs. TRPO-only, with/without interpolation, different negative sampling schemes, varying temporal granularity. Such evaluations would help understanding the contributions of each component.  It seems necessary to look for some external validation, once the experiments focus on Wikipedia-style corpora and QA; it’s unclear how robust TRPO is for bursty and non-encyclopedic domains with  drift (e.g., newswire, finance, code, biomedical preprints).

### Questions
1. How exactly are “aligned/misaligned” pairs sampled to avoid topical confounds (e.g., same entity with different years vs. different entities altogether)? 
2. Can you report multi-seed mean±std and a simple paired significance test for the main nDCG tables?
3. For α at inference: did you try to use the timestamp predictor to pick α automatically, or a small calibration set to learn α per query/domain?
4. In the BEIR analysis, please clarify how overfitting is prevented through α tuning per dataset? Is it possible to freeze α by creation year only and still see the trend?
5. How sensitive are results to the specific Wikipedia dumps/months chosen? Have you tried different adjacent snapshots?
6. For the mixture-of-retrievers, can you include a distilled single-encoder baseline (knowledge-distill multiple πt into one) so that we are able to separate capacity from temporal specialization?

### Soundness
3

### Presentation
3

### Contribution
3
