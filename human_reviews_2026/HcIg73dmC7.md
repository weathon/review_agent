# Revisiting Theory of Contrastive Learning for Domain Generalization

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Contrastive learning is among the most popular and powerful approaches for self-supervised representation learning, where the goal is to map semantically similar samples close together while separating dissimilar ones in the latent space. Existing theoretical methods assume that downstream task classes are drawn from the same latent class distribution used during the pretraining phase. However, in real-world settings, downstream tasks may not only exhibit *distributional shifts* within the same label space but also introduce new or broader label spaces, leading to *domain generalization* challenges. In this work, we introduce \emph{novel generalization bounds} that explicitly account for both types of mismatch: domain shift and domain generalization. Specifically, we analyze scenarios where downstream tasks either (i) draw classes from the same latent class space but with shifted distributions, or (ii) involve new label spaces beyond those seen during pretraining. Our analysis reveals how the performance of contrastively learned representations depends on the statistical discrepancy between pretraining and downstream distributions. This extended perspective allows us to derive provable guarantees on the performance of learned representations on average classification tasks involving class distributions outside the pretraining latent class set.
We present theoretical results on new bounds for various downstream tasks, including out-of-distribution detection, dataset corruption, clustering, and transfer learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel and *realistic* generalization bounds for contrastive learning under both domain shift and domain generalization (distinguishing the two is one of the paper's main ideas). The authors show that downstream performance depends on the statistical discrepancy between pretraining and downstream distributions through an explicit bias term, thereby providing guarantees in both scenarios. In other words, representations learned through contrastive learning transfer well when downstream class means remain close to the pretraining latent means.

### Strengths
Thank you for your work, I have truly enjoyed reading the paper. Below, I have listed the strengths of this paper. 

- Realistic Assumptions in Contrastive Learning (CL) X Domain Generalization (DG): First and foremost, it is worth mentioning that the paper addresses an important, yet often overlooked issue in DG: the in-distribution assumption. I believe this is the largest, and most important contribution of this paper.
- Technical Correctness: The derivations proposed in the paper are sound and mathematically consistent.
- Presentation: The paper is well-written, and the motivations are very clear and easy to understand.

### Weaknesses
While the paper has some strengths, it also has critical weaknesses. 

- Lack of empirical grounding: While the theoretical analysis is sound, there is a visible lack of experimental results. While I acknowledge that this paper's contributions derive in its theoretical aspects, the paper would benefit from even small-scale experiments. Frankly, this is my largest concern regarding the paper. Please refer to the following Questions section for potential experiments.
- Partially novel, yet incremental contribution: The gap between CL and DG is a topic deeply studied in the last few years. While the paper extends Saunshi et al's [1] framework to a more realistic setting, the main observations (e.g., CL provably degrades under realistic distribution shifts and thus better sampling/alignment is needed) are already present.

***

### References
[1] Saunshi et al., A theoretical analysis of contrastive unsupervised representation learning, ICML, 2019

### Questions
Overall, I view this paper to be theoretically solid and promising, but there are some questions that *must* be addressed.


- I believe adding some empirical analysis would strengthen the paper substantially. For instance:
    - Estimate mean shifts in distributions and correlate them with transfer accuracy.  
    - Could you provide some experiments, at least the ones performed in [1]? The lack of empirical results fades the theoretical contributions of this paper, especially in a venue like ICLR, where empirical results are also expected.
    - Validate the predicted role of the intra-class variance $s(f)$
    - How does increasing $k$ trade off with collision driven penalities?
- Correct me if I'm wrong, but the guarantees focus on mean classifiers. While the paper assumes the use of mean classifiers, in reality, fine-tuned heads are used. Could you elaborate how the paper's conclusions could be extended to scenarios where fine-tuned classifiers are used? e.g., Are there stability assumptions under which the conclusions still go through?


***

### References
[1] Saunshi et al., A theoretical analysis of contrastive unsupervised representation learning, ICML, 2019

### Soundness
2

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
This paper extends the theoretical framework of contrastive learning (CL), originally formalized by Saunshi et al. (2019), to explicitly account for domain shift and domain generalization. Classical theory assumes that downstream tasks share the same latent class distribution as pretraining, an assumption violated in realworld transfer scenarios. The authors introduce a generalization bound for contrastive representations that includes a bias term $B(f)$, quantifying representational misalignment between pretraining and downstream distributions.


They derive bounds for the supervised downstream loss under two settings: (i) shifted distributions within the same label space and (ii) novel downstream classes outside the pretraining latent set. The paper also extends the theoretical analysis to multiple negative samples, bounding how class collisions and intra-class variance affect guarantees. Additional analysis is provided under Lipschitz continuity and sub-Gaussian assumptions for the encoder, yielding interpretable upper bounds that scale linearly with the shift magnitude $\epsilon$. The authors conclude by situating their work as a unifying theoretical account of contrastive learning robustness and transferability.

### Strengths
Overall, the paper demonstrates theoretical soundness. Its central claim that distributional mismatches introduce a quantifiable bias term in the generalization bound is well supported. The proofs are largely self-contained and presented with sufficient rigor, and the results hold consistently across multiple loss functions (hinge and logistic), which strengthens their generality. No major logical flaws or unjustified leaps were found. Other notable points are stated below. 

--Ambitious attempt to bridge contrastive-learning theory with domain-shift analysis.

--The derived bias bounds (scaling with shift magnitude $\epsilon$) are interpretable and could guide empirical model evaluation

--Useful decomposition of the unsupervised loss into shifted-mean and intra-class-variance terms; conceptually interesting direction for future work.

### Weaknesses
The work relies on idealized assumptions, such as access to latent class means and uniform downstream sampling, which may not hold in realistic settings as they agree in the end of Section 4.


Lack of literature review: Although the paper builds directly upon and extends the framework of Saunshi et al. (2019), the authors do not clearly summarize that prior work or provide a comprehensive comparison with its results. This omission weakens the contextual grounding of their contribution.


The abstract promises guarantees for new downstream classes, but no theorem contains a term accounting for unseen classes. All bounds assume shared label spaces with bounded mean shifts, i.e., domain shift only.


The appendix introduces $\tau_k, \tau_0, \rho_{\min}^+, p_{\max}$ but not provide explicit scaling or normalization, making it impossible to check consistency with Theorem 5.1’s informal statement.


Incomplete presentation: Several presentation issues reduce the clarity of the work and make it hard to fully evaluate.  Some examples are noted below. 

- Missing definitions for key notations:  $W^\mu$ in line 114, $R$ and $\Re_S(\mathcal{F})$ in line 168
 
- The inequality on line 720 extends beyond the visible text boundary, leaving it unreadable

 - An incomplete sentence appears around lines 237–238

### Questions
1. In Section 4.2, both similar pairs and negative examples can originate from either the same or different classes. So what specific criterion is used to define similar and negative examples in this context? Does “similar” refer to an augmentation of the anchor sample, or is there another underlying assumption?

2. The paper mentions measuring the discrepancy of novel downstream class means by their distance to the convex hull of pretraining means. However, this concept does not appear explicitly in any theorem. Could you clarify where this notion is formally incorporated or demonstrate how it can be derived within your framework?

3. Please provide a clearer comparison between your theoretical results and those of Saunshi et al. (2019). Specifically, what key assumptions or bounds have been relaxed or strengthened relative to their work?

4. Could your theoretical framework be extended to contrastive multimodal pretraining (e.g., CLIP), where the distributional shifts occur across different modalities rather than domains?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper extends the theoretical framework of contrastive learning to account for distribution shift and domain generalization. Building upon Saunshi et al. (2019)'s latent class model, the authors introduce generalization bounds that explicitly handle: 1. Domain shift where downstream class distributions differ from pretraining; 2. Domain generalization where downstream tasks involve novel label spaces. The key contribution is a bias term that quantifies representational misalignment between pretraining and downstream distributions, bounded under various assumptions.

### Strengths
1. Well-motivated extension: The paper addresses a genuine gap in contrastive learning theory.
2. Clean mathematical framework: The paper provides an elegant way to quantify distribution mismatch. The first-order Taylor expansion approach in Lemma 4.3 is technically sound.

### Weaknesses
1. Limited practical impact: No experiments showing whether minimizing the theoretical quantities leads to better downstream performance
2. Incremental technical contribution: The proof technique is a straightforward extension of Saunshi et al. (2019)

### Questions
1. Tightness: Can you provide any evidence (theoretical or empirical) that your bounds are not vacuous? 
2. Can you provide more experiments to demonstrate the effectiveness of your theory?
3. Connection to recent work: How does your framework relate to:
- Provable guarantees for self-supervised learning under distribution shift ?
- The recent analysis of foundation models' transferability?

### Soundness
2

### Presentation
2

### Contribution
2
