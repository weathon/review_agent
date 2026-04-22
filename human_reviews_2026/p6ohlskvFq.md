# PGN: A Polar Geodesic Network for Multimodal Emotion Recognition

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Multimodal emotion recognition faces semantic ambiguity, significant noise, and cross-modal interference, including missing modalities. Psychological research supports a radial structure of emotions, yet many methods overlook this geometry and accumulate directional noise during fusion. The Polar Geodesic Network maps modality embeddings into a radial space, performs reliability-aware geodesic fusion to preserve circular topology, and then uses a Transformer to refine the fused representation and capture cross-dimensional interactions. Under a unified frozen-backbone protocol, PGN attains 0.6835 Accuracy and 0.6756 Weighted-F1 on MELD, and 0.7340 Accuracy and 0.690 Macro-F1 on IEMOCAP. Ablation results indicate complementary gains from geometry-aware fusion and the subsequent Transformer. These findings show that explicit modelling in radial space improves recognition accuracy and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes PGN (Polar Geodesic Network) for multimodal emotion recognition (MER). Each modality’s embedding is mapped into a polar form, radius and form, then fused via reliability-weighted geodesic means. Concretely, phases are averaged by the circular Fréchet mean (angle of the resultant), amplitudes by a reliability-weighted average, and a lightweight Transformer with a geometry-aware attention bias refines the fused representation. Under a unified frozen-backbone protocol, PGN achieves better results than reproduced attention baselines (MulT, MemoCMT, MultiEMO) on MELD and IEMOCAP; ablations show both the geodesic fusion and the Transformer contribute complementary gains, and robustness tests (noise, occlusion, missing modalities) favor PGN.

### Strengths
(1) A promising geometry-aware MER formulation: decouple intensity and direction and aggregate directions via shortest-arc (geodesic) means to avoid wrap-around artifacts (e.g., opposite angles averaging to “neutral”). The method defines the signed angular difference, geodesic distance, and uses the resultant phase as the circular mean. 

(2) A reliability-weighted fusion that combines local phase consistency and amplitude to down-weight noisy/missing modalities during fusion. 

(3) A frozen-backbone evaluation protocol with multi-seed reporting and paired tests, reducing confounds from encoder tuning; PGN shows consistent improvements vs. reproduced baselines on MELD/IEMOCAP. 

(4) Ablations (remove geodesic / remove Transformer) and order-sensitivity (PGT>PTG) support the design choices; robustness sweeps (SNR, token drop, missing modalities) suggest graceful degradation.

### Weaknesses
(1) The paper lays out the circular distance and circular mean carefully, but the link from psychology’s circumplex to concrete MER decision boundaries is mostly qualitative. For instance, while §2.2 defines d(\theta,\theta') and justifies arc-over-chord, the work does not clearly illustrate how class prototypes/decision regions would actually look in the polar space or how geodesic fusion moves ambiguous points relative to Euclidean fusion on real samples. Stronger mechanistic evidence (e.g., before/after angular distributions, prototype geometry) would make the insight more convincing.

(2) Key steps are scattered between main text and appendices (e.g., gradient gating by resultant length, complex-to-real interface for attention, and exact reliability logit formation). A smooth and more informative mapping between figure and methodology would improve readability.

(3) Only two ERC datasets (MELD, IEMOCAP) are in the main text; MOSEI appears only in the appendix and mainly as sentiment (and without apples-to-apples reproduced baselines). Cross-dataset transfer, domain shift, or multilingual settings that are common in MER are not fully covered, and the authors themselves note scope limits. 

(4) While reproduced baselines are fair internally, the set (MulT, MemoCMT, MultiEMO) omits several recent robust-fusion or missing-modality MER approaches; literature-reported numbers are intentionally not used for significance, but this also narrows the external positioning of PGN.

(5) The on/off ablations (no geodesic, no Transformer) and order test (PGT vs PTG) are helpful, but finer-grained ablations are missing: e.g., (i) remove reliability weighting but keep circular mean; (ii) replace geodesic mean with chord/Euclidean on the unit circle; (iii) vary masked-softmax temperature and the phase-gradient gate systematically in main text (some sweeps are in appendix but not tied to core claims).

### Questions
(1) Why polar per-dimension? Are phases defined per hidden dimension semantically meaningful, or would a single global angle (or low-dimensional angular subspace) suffice? Any analysis of phase correlations across dimensions?

(2) What is the contribution of the local phase-consistency term vs amplitude alone?

(3) How does PGN perform if the phase mean uses Euclidean chord? What about simply averaging raw angles with wrap correction? Or if we drop geodesic-aware attention bias in the Transformer?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a polar geodesic network for the MER task. The core idea is to disentangle affective representations into amplitude (intensity/confidence) and phase (affective direction). It then performs reliability-weighted geodesic fusion (using the circular Fréchet mean for phases) to preserve the circular topology of emotions, inspired by psychological models like the circumplex. Finally, a lightweight Transformer with geometry-aware attention refines the fused representation for classification.

### Strengths
1. Disentangling affective representations into amplitude (intensity/confidence) and phase (affective direction).
2. Designing a lightweight Transformer with geometry-aware attention to refine the fused representation.

### Weaknesses
1. The core components, i.e., polar coordinate representations and geodesic fusion via the circular mean, are well-established concepts in directional statistics and geometric deep learning. The paper primarily applies these ideas to multimodal emotion recognition rather than introducing a fundamental innovation. The actual architectural novelty, specifically the integration path, feels incremental.
2. I think the biggest flaw of this paper is the integrity and validity of the experiments. First, the compared methods are only three; this is incredible for the MER field. This field has been developing for decades, for example, the IEMOCAP dataset used in this paper was proposed in 2008. Therefore, the three compared methods are definitely insufficient. Secondly, the three compared algorithms do not have references, which is not friendly to beginners.
3. The decision to use a unified frozen-backbone protocol, while ensuring fairness, severely constrains the model's representational power and may not reflect real-world performance where fine-tuning is common. More critically, all baseline comparisons are based on the authors' own reproductions. Without direct comparisons to officially reported results using their original code and hyperparameters, it's impossible to verify if the baselines were implemented optimally, introducing a potential source of bias that undermines the claimed superiority.
4. The ablation study reveals a critical weakness: removing the subsequent Transformer causes the largest performance drop. This suggests that the proposed geometric fusion alone is insufficient and that the model heavily relies on a standard Transformer to capture the necessary dependencies, thereby reducing the perceived standalone contribution of the geometric components.

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Polar Geodesic Network (PGN), a geometry-aware multimodal emotion recognition framework operating in polar coordinates. PGN maps modality features into amplitude and phase representations, performs reliability-weighted geodesic fusion to preserve the circular structure of emotions, and suppresses interference from noisy modalities. A geometry-aware Transformer then refines cross-modal interactions. Experiments on MELD and IEMOCAP demonstrate that PGN significantly outperforms existing methods in both accuracy and F1 score, showing greater robustness and stability, and validating the effectiveness of explicitly modeling geometric structures in emotion recognition.

### Strengths
1. The method is innovative. Learning emotional representations in polar coordinate space is a very reasonable and intuitive idea.
2. The method combines model design with circular statistics and Riemannian geometry. It has good interpretability and mathematical consistency.
3. It achieves significantly better results than multiple baselines on both the MELD and IEMOCAP datasets.

### Weaknesses
1. The comparison is limited to a small set of reproduced Transformer-based baselines. This makes it difficult to evaluate the performance of PGN based on the current MER standards. 

2. Could you provide the feature visualization? Are the differences in the distribution of features more distinct for samples of different emotions?

3. Could you provide and discuss some examples of successful and unsuccessful cases?

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
