# ATOM of Understanding: Information-Theoretic Decomposition for Interpretable 3D Question Answering

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
3D Question Answering remains largely opaque, with existing approaches functioning as black boxes that provide no insight into how different modalities contribute to spatial reasoning. This lack of interpretability limits trust and understanding in critical applications like robotics and autonomous systems. We present ATOM (\textbf{A}daptive \textbf{T}ask-aware m\textbf{O}dular \textbf{M}odel), the first information-theoretic framework that operationalizes Partial Information Decomposition (PID) to achieve fully interpretable 3D question answering. ATOM explicitly decomposes multimodal interactions into four theoretically-grounded information atoms: point cloud uniqueness, image uniqueness, redundancy (shared cross-modal information), and synergy (emergent complementary information). Our framework provides transparent reasoning through a Query-driven View Aggregator for geometrically consistent visual features, a Contextual Grounding Module for description-guided visual grounding, a Question-aware PID module with theoretically-grounded regularization, and a Dynamic Atom Modulation mechanism that provides direct, quantifiable interpretability of each atom's contribution. Extensive experiments on ScanQA and SQA3D datasets demonstrate that ATOM achieves the performance comparable to prior work and, to the best of our knowledge, are the first to enable transparency in 3D reasoning. Our analysis reveals that different question types systematically rely on distinct information patterns that align with human spatial cognition.n.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces ATOM (Adaptive Task-aware mOdular Model), a framework for interpretable 3D Question Answering (3D QA) based on Partial Information Decomposition (PID) theory.
The authors aim to move beyond “black-box” fusion models by decomposing multimodal interactions into four interpretable information atoms: redundancy, uniqueness (for each modality), and synergy. ATOM integrates several modules:

- Query-driven View Aggregator (QVA) for selecting geometrically relevant views,

- Contextual Grounding Module (CGM) for injecting scene description context,

- Question-aware PID (Q-PID) to decompose multimodal features, and

- Dynamic Atom Modulation (DAM) for question-dependent weighting of these atoms.

Experiments are conducted on ScanQA and SQA3D datasets, with comparisons against 3D QA baselines (e.g., DSPNet, 3DGraphQA, 3D-VisTA). Results show comparable or slightly lower performance than the strongest baselines, with claims of improved interpretability.

### Strengths
1. Conceptual novelty: The paper presents an interesting attempt to incorporate information-theoretic decomposition (PID) into multimodal QA, a direction that could enrich interpretability research in 3D reasoning.
2. Clear motivation: The motivation for interpretability in 3D QA—addressing black-box fusion—is well framed and relevant to embodied AI. 
3. Theoretical grounding: The PID formulation (Equations 1–3, page 4) provides a rigorous theoretical foundation.

### Weaknesses
1. Weak empirical justification of the proposed theory.
The ablation study (Table 3, page 8) shows minimal performance gains for most components. Removing the core PID module (“w/o Q-PID”) leads to only minor drops in ScanQA (22.5 → 23.5 EM@1) and SQA3D (48.76 → 49.71 EM@1). This 0.7–1 point difference is within noise for such datasets, suggesting the proposed information decomposition does not substantively improve task performance.

2. Poor quantitative results compared to state-of-the-art.

3. On SQA3D, ATOM achieves only 49.7 EM@1, below DSPNet’s 50.4.

4. On ScanQA, ATOM’s EM@1 = 26.7 vs. DSPNet = 26.5, but CIDEr and BLEU-4 scores are lower (77.3 vs. 78.1, 15.0 vs. 15.4).
These marginal improvements or regressions do not justify the architectural complexity introduced.

5. Ablation design does not convincingly isolate interpretability benefits.
The PID components (U1, U2, R, S) are claimed to provide “interpretable decomposition,” but the paper only shows weight histograms (Fig. 4) without human evaluation, visualization of PID contributions, or quantifiable interpretability metrics. The interpretability claims remain qualitative and anecdotal, not scientifically validated.

6. Overly complex framework with minimal gain.
The model combines PointNet++, Swin Transformer, SBERT, QVA, CGM, Q-PID, and DAM, introducing a large number of modules, parameters, and hyperparameters. Despite this, the gains are statistically insignificant, raising concerns about over-engineering.

7. Experimental setup not fully convincing.
The authors emphasize “no pre-training,” but this makes comparisons unfair since methods like 3D-VisTA and Multi-CLIP rely on pretraining. For a fair comparison, results should be grouped and analyzed under equivalent training regimes.

### Questions
1. The claimed interpretability advantage is qualitative. Can the authors provide quantitative metrics or user studies showing that the decomposition indeed improves transparency or error diagnosis?

2. The ablation results (Table 3) show very small gains from adding PID regularization. Can the authors explain how these minimal differences support their claim of “theoretically grounded interpretability”?

3. Could you report statistical significance tests (e.g., confidence intervals or variance) on the EM@1 and EM@10 results to verify that improvements are meaningful?

4. Given the heavy reliance on prior architectures (e.g., DSPNet backbone, MCGR), to what extent does the proposed contribution stand as an independent framework rather than an incremental modification?

5. How does the method scale to open-ended QA or additional modalities (text + depth + point cloud), as briefly mentioned in Appendix B? Any early results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ATOM, an end-to-end 3D question answering framework that incorporates Partial Information Decomposition. The proposed framework produces compatible performance with SOTA methods on ScanQA and SQA3D benchmarks, while providing different information atoms.

### Strengths
1. The proposed information theoretic components indeed have several loss functions for regularization to make them more compatible with the theories.

### Weaknesses
1. Lack of comprehensive analysis of the learned information atoms despite the claim that the proposed framework introduces interpretability.

2. The baseline used for evaluation on SQA3D is not the current best model for this benchmark. For example, the author didn't compare against SID3D [1]

3. Despite using image and point cloud together, the proposed method is not able to provide superior performance compared to single-modal models like 3D-VisTA.

4. The notations in the paper are so much that the paper is hard to follow.


[1] Man, Y., Gui, L.Y. and Wang, Y.X., 2024. Situational awareness matters in 3d vision language reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13678-13688).

### Questions
1. In Ln.19-21 and Ln. 59-61, the author claimed that there are four information atoms whereas in the following sentences and throughout the whole paper there's only three.

2. For the loss term $L_{div}$, why does there exist case where atoms are not used inside a batch? What is the range for the importance weights $\beta_k$?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ATOM, an information-theoretic framework to integrate Partial Information Decomposition (PID) into an end-to-end 3D Question Answering (3D QA) model. ATOM emphasizes question-aware decomposition, which is essential for spatial reasoning. The framework comprises 4 key components: the Query-driven View Aggregator (QVA), the Contextual Grounding Module (CGM), the Question-aware PID (Q-PID) module, and the Dynamic Atom Modulation (DAM) mechanism. Validated on the ScanQA and SQA3D datasets, ATOM shows performance that is competitive with or marginally below the current SOTA, while achieving principled interpretability through information-theoretic decomposition.

### Strengths
1. ATOM explicitly decomposes multimodal interactions into 4 theoretically-grounded information atoms (redundancy, uniqueness, and synergy) in a question-aware manner. This rigorous decomposition provides transparent reasoning pathways. Furthermore, the model achieves competitive results without requiring external pre-training, enhancing its practical utility.
2. The experiments successfully validate that ATOM achieves performance comparable to prior work while providing unprecedented interpretability. This makes ATOM a valuable framework for bridging the gap between high-performing end-to-end 3D QA and the demands of explainable AI.
3.  The framework is clearly explained through a concise overview figure (Fig. 3) and detailed descriptions in Section 3. The authors also commit to providing anonymous source code and comprehensive implementation details in the supplementary materials, enhancing reproducibility.

### Weaknesses
1. The placement of figures is suboptimal, with Figure 1 appearing on page 3, one and a half pages after its initial textual mention in the Introduction, and following Figure 2. The manuscript layout could be more logically organized to align figures closer to their first mention.
2. The incorporation of PID introduces 4 specific regularization losses in addition to the main task loss, making the final objective complex. The authors themselves admit that this explicit information decomposition requires careful hyperparameter tuning, which increases training complexity and engineering effort.

### Questions
1. According to Figure 4 (Left), the mean DAM weight distributions for the ScanQA and SQA3D datasets exhibit notably different patterns. Could the authors elaborate on how they reconcile the general conclusions with the significantly different behavior observed between the two datasets, and why the final interpretation seems to align more closely with the SQA3D patterns?
2. The ablation study in Table 4(b) systematically examines the individual and combined effects of the atoms, showing R, R+S, and the full ATOM. Given the emphasis on the modality-specific Uniqueness atoms U1 and U2 as crucial complementary signals that conventional methods overlook, why was the combined effect of uniqueness U1+U2 or U1/U2-only not included in the ablation study? Including this variant would be highly informative for direct comparison against the baseline w/o Q-PID, R, and R+S, thereby better isolating the contribution of different information atoms in 3D QA.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
They attempted to apply Partial Information Decomposition (PID) theory to 3D-QA literature. Based on PID theory, the following information atoms are introduced: redundancy of the shared information between two sources; uniqueness of the evidence contributed by each source individually; and synergy of the complementary information emerging from combination. Their proposed ATOM works in three stages: stage 1 comprises Query-driven View Aggregator (QVA) and  Contextual Grounding Module (CGM),modules; stage 2 is Question-aware PID (Q-PID) that include redundancy, uniqueness and synergy modules; stage 3 is Dynamic Atom Modulation (DAM). Their training loss is the addition of task objective loss and the regularization losses of redundancy, uniqueness and synergy by PID-theory.

In experiments with ScanQA and SQA3D, their model performed on par with their previous models. They confirmed the effectiveness of each information atom (redundancy, uniqueness and synergy) with the ablation experiment.

Overall, they proposed a model based on an interesting viewpoint of PID theory. Considering the final score, their final model doesn’t necessarily perform better than previous models. However, their approach can be worth further investigating.

### Strengths
- S1: Interesting approach to apply Partial Information Decomposition (PID) theory to 3D-QA literature
- S2: CGM generates two different representations from different perspectives of images and 3D point clouds that are compared in the way of redundancy, uniqueness and synergy by PID-theory.
- S3: Experimental results suggest the effectiveness of the proposed to some extent.
- S4: It is interesting that authors provide the variations of architectures (ATOM-MoE and ATOM-Flat) in Table 5.

### Weaknesses
- W1: The entire model becomes too complex and it is difficult to grasp how each module processes what kind of information and how it affects others at a glance. This can become an important limitation to further analyze and improve upon this model.
- W2: The entire performance is mostly in the 2nd place or later in most of ScanQA and SQA3D datasets.
- W3 (minor): Characters in some figures are too tiny and really difficult to read. (Especially for subscripts in Fig4).

### Questions
See weakness.
L. 059: four information atoms: three?

### Soundness
4

### Presentation
2

### Contribution
3
