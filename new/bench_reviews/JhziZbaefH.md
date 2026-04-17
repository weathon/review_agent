## Summary

The paper proposes OML, a brain-inspired hierarchical modular neural network for online multimodal learning with human-in-the-loop interaction. The architecture consists of feature neurons, unimodal association neurons, and multimodal association neurons connected via ascending, descending, and lateral pathways, enabling continuous learning of new multimodal concepts without catastrophic forgetting. Key innovations include a reference extraction algorithm that identifies which features a word refers to (e.g., "red" refers to color, not shape), and a conflict detection mechanism that can pose questions to users when new inputs conflict with prior knowledge.

## Strengths

- **Novel problem formulation with integrated capabilities**: The paper tackles an under-explored but meaningful problem—online multimodal learning that integrates reference extraction, conflict detection, and interactive questioning in a single framework, going beyond standard continual learning or standard multimodal retrieval.

- **Well-motivated hierarchical architecture**: The FN–UAN–MAN structure with ascending, descending, and lateral pathways is thoughtfully designed, with principled distinctions between order-independent (visual) and order-dependent (auditory) activation modes that reflect real differences in how modalities encode information.

- **Consistent improvement over existing online baselines**: Across all experiments (Tables 1–3), OML outperforms the only comparable online baselines (ART and AEN) in cross-modal retrieval accuracy under both close and open environments. In open environments, the improvements are notable (e.g., Fruits V→A: 89.8 vs. 86.2 for ART; VAT T→A: 93.9 vs. 89.0 for AEN).

- **Modality extension results**: The VAT and VAT-HomeF experiments demonstrate that the network can be extended with a new modality (taste) while preserving prior associations, which is a genuinely valuable and underexplored capability.

- **Reference extraction mechanism as a conceptually interesting idea**: The variance-based method for determining which features a word refers to (Section 3.4) is a creative attempt to address a real limitation of prior binding approaches, even if its empirical validation is incomplete.

## Weaknesses

### Major:

- **Core conceptual claims are not directly evaluated**: The paper's headline contributions—precise reference extraction, conflict detection, and human-in-the-loop interactive learning—are not empirically validated in a way that matches the claims. The experiments measure only cross-modal retrieval accuracy. Conflict detection accuracy, false positive/negative rates, the quality/frequency of questions asked, and performance as a function of user answers are never quantified. The conflict resolution mechanism is bypassed in the main experiments by auto-setting unanswered questions to "positive" (Section 4: "if the question posed to the user by OLM remains unanswered for a certain period of time, we set the answer to be positive"). The claim that "OML is able to detect all conflicts" under 10% noise injection (Conclusion, paragraph 3) is stated anecdotally without any table, statistical test, or analysis of false positives on clean data. This is a fundamental mismatch between the paper's framing and its experimental evidence.

- **"Precise referring" evaluation does not operationalize the claimed capability**: The E-Fruits/E-HomeF experiments use standard cross-modal retrieval accuracy as the metric. As the authors themselves acknowledge, baselines like ART and AEN "return all features (shape and color) of red objects" and this is still "count[ed] as a correct result" (Section 4.1, paragraph 2). This means the retrieval metric is blind to whether the model's internal representation correctly attributes "red" to color features only. There is no quantitative measurement of which feature types are selected by the reference extraction function, what proportion of words are correctly classified as attribute vs. object terms, or any failure case analysis. A model that simply memorizes co-occurrence of word and full-image feature vectors could achieve similar retrieval performance.

- **No ablation studies**: The paper claims several innovations (reference extraction, conflict detection, lateral pathways, frequency-based routing), but none are isolated experimentally. It is impossible to determine which components drive the performance gains over ART/AEN. This is a significant gap given the complexity of the architecture and the multiple interacting mechanisms.

- **Limited datasets and outdated baselines**: Experiments are conducted only on small, narrow-domain datasets (Chinese fruit names, home objects with fruit subsets). The offline baselines include DAE (2011) and DBM (2014), which are over a decade old. No modern deep-learning baselines for continual learning (e.g., experience replay with deep networks, regularization-based methods) are compared. While direct comparison with recent deep multimodal models on these specific datasets may be limited by the availability of code, the absence of any modern baseline makes it hard to assess the method's competitiveness.

- **Unfair comparison with offline methods in open environments**: In the open environment setting, offline methods are evaluated by training them sequentially on disjoint data subsets without any continual learning mechanism, which amounts to measuring catastrophic forgetting under an adversarial protocol the methods were never designed for. The paper then highlights that these offline methods' accuracy "drops significantly" (Conclusion, paragraph 1). This is an expected result, not a fair comparison. The valid comparison is between OML and the online methods (ART, AEN), where OML genuinely excels.

### Minor:

- **Frequency/Fourier signaling mechanism lacks empirical isolation**: The λ-frequency-based routing and cosine-activation functions are central to the architecture but are never compared against simpler alternatives (e.g., index-coded one-hot channel IDs). The claim that λ enables modality disambiguation (Section 5) is stated but not demonstrated in isolation.

- **Threshold-dependent mechanisms without sensitivity analysis**: The conflict detection and reference extraction depend on multiple hand-set thresholds (θ, ϑ=0.8, r=0.5). No sensitivity analysis is provided, making it unclear how robust these mechanisms are to threshold choices or how they would generalize to different data distributions.

### Trivial:

- Equation numbering is inconsistent in Section 3 (multiple equations share labels like "Eq. (1)" appearing for both Eq. 1 and Eq. 3). This appears to be a formatting issue but may hinder reproducibility.

## Nice-to-Haves

- Evaluation on larger-scale or more diverse multimodal benchmarks to assess scalability beyond fruit/fruit-like datasets.
- Ablation studies isolating each proposed component (reference extraction, conflict detection, lateral pathways, frequency routing).
- Quantitative evaluation of reference extraction accuracy (e.g., proportion of words correctly attributed to the right feature types).
- Analysis of network growth (number of neurons/connections) as a function of learned concepts to assess lifelong learning feasibility.

## Removed Points

- **Demand for modern continual learning baselines (e.g., EWC, progressive networks) applied to multimodal settings**: While this would strengthen the paper, applying standard CL methods to this specific multimodal binding formulation is not straightforward, and the paper already compares against the two most directly comparable online methods (ART, AEN). This is a nice-to-have, not a core flaw for the paper's stated scope.

- **Criticism that hand-crafted features (SAM + Fourier descriptors, MFCCs) limit representational power**: The paper's architecture is designed for interpretable, neuro-inspired feature binding where feature types must be explicitly defined. Using end-to-end deep features would fundamentally change the nature of the approach. This is a scope/limitation worth noting but not a flaw in the current design.

- **Demand for real human-in-the-loop evaluation with actual participants**: The paper clearly scopes its evaluation protocol. While using auto-yes for unanswered questions weakens the human-in-the-loop claim (and this IS noted as a major weakness above), demanding a full user study goes beyond what is standard for this type of systems paper.

- **Criticism of "overselling" in abstract/conclusion**: While the claims are indeed stronger than the evidence warrants, this is captured in the major weaknesses about unevaluated core contributions. Flagging it separately risks double-counting.

## Novel Insights

The most important insight emerging from the reviews is the fundamental disconnect between the paper's conceptual ambitions and its empirical validation. The architecture is rich and interesting—integrating hierarchical neuron types, frequency-based routing, reference extraction, and conflict detection—yet the experiments reduce everything to cross-modal retrieval accuracy, a metric that cannot distinguish the novel mechanisms from simple co-occurrence memorization. The paper would be significantly stronger if the evaluation directly measured what the mechanisms are designed to do: the accuracy of reference extraction (does "red" correctly bind to color features?), the precision/recall of conflict detection, and the effect of user responses on learning outcomes. Without such targeted evaluation, the paper demonstrates an effective online multimodal retrieval system but does not substantiate its more ambitious claims.

## Suggestions

- Design an evaluation protocol that directly tests reference extraction: for each color/attribute word, measure whether the model's internal binding correctly selects the relevant feature type and suppresses irrelevant ones. Report precision/recall of feature-type attribution.
- Quantify conflict detection: inject known conflicts at known rates into the data stream and report detection rate, false positive rate, and how correct vs. incorrect answers affect downstream retrieval.
- Run ablation experiments removing each major component (reference extraction, lateral connections, frequency coding, conflict checking) and measure the impact on both retrieval accuracy and the claimed capabilities.

## Score and Decision

Calibration against comparable papers:
- **MC² (K1VLZ5rNuZ)**: Scores 3,3,5,3 → Reject. Small datasets, weak baselines, novelty concerns, insufficient evaluation of claimed capabilities. Very similar weakness profile to the paper under review.
- **Beyond Unimodal Learning (Pa6SiS66p0)**: Scores 5,5,3 → Reject. Limited baselines, single dataset, insufficient validation of the method over alternatives.
- **Brain-inspired Multi-View Incremental Learning (hyb6NCjS8G)**: Scores 3,6,3,3 → Withdrawn/Reject. Brain-inspired architecture, scalability/presentation concerns, lacking basic evaluation metrics.
- **Human-in-the-Loop TTDA (OsuV40VuZo)**: Scores 3,3,5,5 → Withdrawn/Reject. Human-in-the-loop aspect overclaimed relative to evaluation.

This paper has a similar weakness profile: interesting ideas but insufficient validation of core claims, narrow datasets, and outdated baselines. However, the architecture itself is more novel and the online learning results against the direct comparables (ART, AEN) are solid. The paper is above the MC²/brain-inspired papers because it does show genuine improvements over the most relevant baselines, but the gap between what is claimed and what is tested is substantial.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>