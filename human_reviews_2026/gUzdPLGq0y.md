# Negatives-Dominant Contrastive Learning for Generalization in Imbalanced Domains

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Imbalanced Domain Generalization (IDG) focuses on mitigating both *domain and label shifts*, both of which fundamentally shape the model's decision boundaries, particularly under heterogeneous long-tailed distributions across domains. Despite its practical significance, it remains underexplored, primarily due to the *technical* complexity of handling their entanglement and the paucity of *theoretical* foundations. In this paper, we begin by *theoretically* establishing the generalization bound for IDG, highlighting the role of posterior discrepancy and decision margin. This bound motivates us to focus on directly steering decision boundaries, marking a clear departure from existing methods. Subsequently, we *technically* propose a novel Negative-Dominant Contrastive Learning (NDCL) for IDG to enhance discriminability while enforce posterior consistency across domains. Specifically, inter-class decision-boundary separation is enhanced by placing greater emphasis on negatives as the primary signal in our contrastive learning, naturally amplifying gradient signals for minority classes to avoid the decision boundary being biased toward majority classes. Meanwhile, intra-class compactness is encouraged through a reweighted cross-entropy strategy, and posterior consistency across domains is enforced through a prediction-central alignment strategy. Finally, rigorous yet challenging experiments on benchmarks validate the effectiveness of our NDCL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the Imbalanced Domain Generalization (IDG) problem, where both domain and label shifts co-occur. It derives a novel generalization bound highlighting the roles of posterior discrepancy and decision margin, and introduces Negative-Dominant Contrastive Learning (NDCL) — a contrastive framework emphasizing negative samples to enlarge margins and balance gradients. Experiments on several datasets show consistent improvements across multiple imbalance settings.

### Strengths
1. The paper provides the a novel theoretical generalization bound tailored to IDG.
2. The paper proposes an innovative negative-dominant contrastive mechanism that avoids explicit resampling. The theoretical insights are linked to the algorithmic formulation, making the overall framework coherent and conceptually grounded.
3. The method demonstrates consistent gains across multiple benchmarks and imbalance scenarios, including severe long-tailed and heterogeneous shifts, showing both stability and scalability.

### Weaknesses
1. The paper does not provide quantitative descriptions of dataset sizes, the number of domains per dataset, or the imbalance ratios (e.g., majority-to-minority class counts). Such details are critical for understanding the severity of imbalance under each scenario. Experiments on large-scale or high-resolution datasets could better demonstrate the effectiveness of NDCL.
2. Some key hyperparameters—including α, β (loss trade-offs), and the Beta distribution coefficient ρ for hard negative mixing—are not explicitly listed. It is suggested the author provide more sensitivity analysis such as their ranges.
3. Although an ablation table is provided, it only tests the presence/absence of NDCL components. It would be better to quantify how the margin term and posterior discrepancy term individually influence performance. A controlled ablation aligning with Theorem 1 would strengthen the link between theoretical and empirical results.
4. NDCL involves multiple sub-objectives and prototype computations. It is suggested to provide more details about the training time, GPU memory usage, or complexity comparisons with baseline contrastive methods, which is essential to assess the practicality.
5. The paper would be strengthened by including several more concrete real-world IDG scenarios, which would better illustrate the practical relevance of the problem under investigation.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles Imbalanced Domain Generalization (IDG), where domain and label shifts co-occur, leading to biased decision boundaries. The authors first derive a generalization bound highlighting the roles of posterior discrepancy P(Y∣X)P(Y|X)P(Y∣X) and decision margin, extending classical DG theory. Guided by this, they propose Negative-Dominant Contrastive Learning (NDCL), which reshapes decision boundaries via three objectives: (1) a negative-dominant contrastive loss to enlarge inter-class margins, (2) a re-weighted cross-entropy for intra-class compactness, and (3) a prediction-central alignment ensuring cross-domain posterior consistency. Experiments on VLCS, PACS, and OfficeHome show NDCL consistently outperforms 21 baselines, achieving stronger generalization under severe imbalance.

### Strengths
The paper’s key strength lies in addressing an underexplored yet practically important problem — Imbalanced Domain Generalization (IDG) which combines challenges of domain and label shift. Its theoretical formulation is a notable step forward, introducing a generalization bound that jointly accounts for posterior discrepancy and decision margin, offering a fresh lens on generalization under imbalance. Methodologically, the proposed Negative-Dominant Contrastive Learning (NDCL) framework is a creative adaptation of existing contrastive paradigms, emphasizing negatives as a dominant learning signal an idea that is both intuitive and empirically supported. From a quality standpoint, the experimental design is thorough, spanning multiple benchmarks (VLCS, PACS, OfficeHome) and including ablations and new imbalance settings, demonstrating reproducibility and effort. In terms of clarity, the visualizations (e.g., margin discrepancy analysis) effectively highlight the intended behavior of NDCL, even though some sections are dense. Finally, in terms of significance, the paper opens a promising direction for robust representation learning under long-tailed, multi-domain settings, which can inspire follow-up work on fairness, calibration, and federated generalization. Overall, it offers moderate originality, solid experimental quality, and conceptual value for the robustness and DG community.

### Weaknesses
The paper’s main weakness lies in the gap between its theoretical claims and empirical validation. While the proposed generalization bound elegantly integrates posterior discrepancy and decision margin, it remains largely unverified quantitatively — no experiments explicitly measure or correlate these terms with observed performance. A small-scale synthetic or analytical validation could strengthen this theoretical link.
From a novelty standpoint, NDCL’s “negative-dominant” formulation is conceptually interesting but derivative of existing ideas such as SupCon (Khosla et al., 2020), hard-negative mining (Kalantidis et al., 2020), and re-weighted CE losses (Cao et al., 2019). The paper would benefit from clearer differentiation, perhaps through ablations isolating the effect of negative weighting versus standard InfoNCE.
The experimental analysis, though broad, remains performance-centric — lacking statistical tests, calibration or robustness metrics, and computational cost evaluations (important since NDCL adds prototype alignment and hard negative generation). Moreover, all evaluations are limited to standard DomainBed datasets; assessing performance in real-world long-tailed or medical domains could enhance generalizability.
Finally, the writing density and notation complexity obscure key intuitions, and the connection between Theorem 1 and NDCL’s design is more rhetorical than formally derived. Making this linkage more explicit or providing geometric visualizations of how NDCL reshapes decision boundaries would significantly improve clarity and credibility.

### Questions
Theoretical–Empirical Link: The generalization bound is central, but no quantitative validation is shown. Can the authors empirically measure how posterior discrepancy or margin width correlates with target-domain accuracy to substantiate Theorem 1? Negative-Dominant Contrastive Design: NDCL’s main novelty lies in prioritizing negatives. Could the authors clarify how this differs from SupCon or debiased contrastive learning and provide ablations isolating this effect?  Hard Negative Mixup Justification: The mixup strategy adds complexity. How much performance gain does it offer compared to standard hard-negative sampling? Posterior Alignment Mechanism: How does the prediction-central alignment (Eq. 3) improve over traditional feature or prototype alignment methods?  Training Efficiency: What is the computational overhead of NDCL relative to standard DG baselines, and is it practical for large-scale or real-time applications?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of Imbalanced Domain Generalization (IDG), where both domain shift and label imbalance coexist across training domains. The authors first derive a theoretical generalization bound based on H-divergence, emphasizing the importance of posterior discrepancy and decision margin for robust generalization. Motivated by this analysis, they propose Negative-Dominant Contrastive Learning (NDCL), which reformulates contrastive learning to emphasize negatives as the main signal, combines it with adaptive re-weighted cross-entropy for intra-class compactness, and enforces cross-domain posterior alignment.

### Strengths
1. Imbalanced Domain Generalization is an important and practical extension of the DG setting, and the paper provides a good formalization.

2. The generalization bound highlights posterior discrepancy and margin effects, offering a potentially useful conceptual perspective.

3. NDCL integrates several mechanisms (contrastive, reweighting, alignment) into a unified training objective.

4. The authors compare with over twenty baselines across multiple imbalance configurations and datasets.

5. The paper is well structured, with helpful visualizations (e.g., Figures 1–4).

### Weaknesses
1. The proposed NDCL mainly combines known components: (a) The “negative-dominant” contrastive loss is a simple reformulation of the InfoNCE / SupCon objective with reversed emphasis; similar ideas exist in hard-negative mining and OOD contrastive learning. (b) The re-weighted CE and prototype alignment are standard practices in long-tailed learning and multi-domain contrastive frameworks.
The overall design feels incremental rather than conceptually new.

2. The presented generalization bound resembles existing DG theory with added posterior and margin terms. The derivation lacks rigor and empirical verification. The connection between the bound and NDCL’s specific loss terms is qualitative rather than principled.

3. Improvements over strong baselines (e.g., Fish, PGrad, BoDA, SAMALTDG) are within 1–2% and not statistically analyzed.
The paper lacks confidence intervals or significance tests, and does not show clear advantages in difficult subgroups.

4. NDCL introduces multiple intertwined losses (contrastive, reweighted CE, posterior alignment) and additional hard-negative mixup.
The motivation for each component is scattered and lacks concrete ablation or efficiency analysis (e.g., computational cost, convergence stability).

5. Although the authors claim to enlarge margins and reduce posterior discrepancy, these quantities are not quantitatively measured or visualized beyond a brief example. The lack of deeper diagnostic analysis weakens the paper’s empirical evidence.

6. The paper is lengthy and occasionally repetitive. The theoretical and methodological sections could be more concise and better connected.

### Questions
1. How does the proposed generalization bound differ mathematically from previous DG bounds, beyond adding a posterior term?

2. Can you provide empirical evidence that the NDCL loss indeed enlarges margins or reduces posterior discrepancy?

3. How sensitive is NDCL to hyperparameters (e.g., α, β, ρ for Beta mixup)?

4. What is the computational overhead compared with standard SupCon or BoDA?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies Imbalanced Domain Generalization (IDG), where both domain shift and label imbalance jointly degrade model generalization. The authors first present a theoretical generalization bound under the H-divergence framework, revealing that the key to IDG lies in controlling the posterior discrepancy and decision margin across domains rather than merely aligning feature distributions.
To address this, the authors propose Negatives-Dominant Contrastive Learning, a contrastive learning framework emphasizing negative samples as the dominant supervision signal. Experiments on VLCS, PACS, and OfficeHome with three imbalanced setups show consistent improvements over prior domain generalization and imbalance baselines.

### Strengths
1. The paper is clearly structured and easy to follow.
2. The derived generalization bound motivates the design choices in NDCL, linking domain discrepancy and class imbalance to posterior alignment.
3. Instead of relying on positive pair supervision, it is interesting that NDCL focuses on negative gradients to enlarge inter-class margins, which is conceptually fresh and practically effective.
4. Several experiments validate the proposed method, which improves significantly compared to previous methods.

### Weaknesses
1. Although the authors provide a theory, it seems not to be related to the label imbalance and domain imbalance.
2. It is better to provide the visualization of the learned feature.
3. The experiments should also be done on the large-scale DomainNet dataset to further demonstrate the effectiveness of the method.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
