# CELAD: Compositional Evaluation for Logical Anomaly Detection

- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Anomaly detection (AD) has attracted significant research interest and now achieves near-perfect performance on most existing benchmarks. However, the majority of prior work has focused on detecting structural anomalies, where anomalies manifest as localized defective regions. Recently, logical anomaly detection (LAD) has emerged, extending AD to cases where violations occur at the level of compositional or relational rules rather than individual regions—a setting particularly relevant for industrial inspection. Despite its importance, LAD research remains hindered by limited dedicated datasets, raising concerns about the generalization ability of current methods. We address this gap with two contributions. First, we introduce CELAD, a new benchmark designed to test compositional understanding in LAD. CELAD features greater variation in both normal and anomalous samples, along with more intricate anomaly types, resulting in a substantial performance drop in state-of-the-art methods. Second, we propose ROMAD, a simple yet effective framework that leverages DETR, an object detector with strong relational modeling, to construct rich object embeddings. ROMAD computes anomaly scores via a training-free matching pipeline and requires only a small number of annotated samples. Extensive experiments show that ROMAD achieves a new state of the art on CELAD, outperforming the next-best method by nearly 15% while maintaining competitive performance on existing datasets. In few-shot regimes, ROMAD further delivers the strongest average results across both CELAD and prior benchmarks, demonstrating its ability to generalize beyond memorization and capture the underlying logical rules. Code and data are available at: https://github.com/neutral-coder-737/Home-Page.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Paper contributes a new dataset, CELAD, for the logical anomaly detection (LAD) problem, which contains images of beaded bracelets. Each bracelet is defined by the letter beads spelling "SPARK" and a combination of black and white colored beads. There are 530 normal samples and 220 anomalous samples across 5 distinct anomaly categories.

It also proposes a novel LAD method based on a DETR object detector - ROMAD. ROMAD uses DETR to detect the objects of interests (OOI) and compare the detected OOI with a set of pretrained memory bank for anomaly detection. The logical pattern of the numerical distributions of normal OOIs are represented as a reference histogram and L2 distance is computed for the inferred histogram and the reference histograms to detect anomalous arrangement of OOIs.

Each reference image is also represented as set of representative embeddings of the OOIs, the set of embeddings are then linearly assigned to the detected embeddings. The complete Distance is thus: D_matched + D_unmatched.

### Strengths
Presentation: Paper is well written and easy to follow.

Motivations: Motivations are clearly explained and supported by the proposed model and the experimental setup.

Methodology: Methodology is sound and efficient.

### Weaknesses
1. Dataset is only limited to beaded bracelets and does not represent the complexity of the real world problem of LAD.

2. The proposed method is specifically designed for the specific type of LAD. There is no formal proof, nor experimental results to support that the proposed method is generalizable across different types of LAD.

3. Experimental results show that SOTA methods outperform the proposed method in the full-shot setting.  

4. Novelty and contributions are limited. 
- The dataset is only for one specific type of objects. 
- Methodology performance

### Questions
1. Dataset is not sufficiently representative of the LAD diversity and complexity.
2. Proposed method does not outperform SOTA in the full-shot setting.

Please comment on the above.

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
This paper introduces CELAD, a new benchmark for logical anomaly detection (LAD) that emphasizes compositional and relational rule violations rather than simple structural defects. It also proposes ROMAD, a DETR-based framework that performs training-free anomaly matching with minimal annotations. Experiments show that ROMAD achieves state-of-the-art results on CELAD

### Strengths
1.CELAD is a more challenging LAD benchmark with richer compositional rules and diverse anomalies, exposing poor generalization of current SOTA methods.
2.ROMAD is simple, few-shot efficient, and leverages DETR’s relational embeddings with a training-free matching pipeline.

### Weaknesses
1.Ignores structural anomalies, limiting real-world applicability where both anomaly types coexist.

2.No pixel-level anomaly maps, reducing interpretability and compatibility with standard AD evaluation.

3.Annotation cost comparison with segmentation-based methods lacks empirical justification.

### Questions
1.Why does ROMAD underperform slightly on LOCO despite strong results on CELAD? Is LOCO too simple to benefit from relational modeling?

2.How sensitive is the distance-based attention to object density variations across datasets?

3.Are there plans to extend CELAD to more object categories or dynamic (video) settings?

If my main concerns are properly addressed, I would be willing to raise my evaluation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CELAD, a new benchmark designed to test compositional logical anomaly detection, and proposes ROMAD (Relation-aware Object Matching for Anomaly Detection). ROMAD leverages a DETR-based detector to extract contextual object features, enriches them with distance-aware attention and area cues, and performs training-free bipartite matching against a memory bank of normal samples.  Experiments on MVTec LOCO AD and the new CELAD dataset show solid gains (≈ 14–15 %) and good few-shot performance.  While CELAD as a dataset is a meaningful step toward richer logical-composition evaluation, ROMAD itself mainly assembles existing components. Claims of logical or compositional reasoning are not well supported as shown form the experiments.

### Strengths
1. Well-timed attempt to move LAD research toward compositional evaluation.
2. The CELAD dataset appears carefully annotated and publicly released.
3. The proposed ROMAD model achieves strong few-shot anomaly detection for logical anomalies.
4. Experimental comparison with seven baselines (PSAD, CSAD, ComAD, SINBAD, EfficientAD, ULSAD, SA-PatchCore) shows superior performance of the proposed ROMAD method.

### Weaknesses
1. CELAD provides good benchmark contribution, but ROMAD showed limited methodological novelty. ROMAD offers little conceptual novelty, since it essentially combines DETR features with nearest-neighbor matching.
2. The proposed method heavily relies on the pretrained models; originality lies mainly in combination.
3. CELAD’s focus on bracelet-type compositions, but it provides limited generality.
4. The “compositional reasoning” claims exceed the presented evidence.
5. The experimental evaluation is performed on the logical anomalies in MVTec LOCO AD and CELAD datasets only. Including experimental results on more datasets, such as MVTec AD, would strengthen the paper by demonstrating broader applicability and robustness across different anomaly domains.

### Questions
1. The proposed ROMAD method was developed for detecting logical anomalies only. It seems to be quite restricted.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces both a new pipeline for industrial (primarily logical) anomaly detection tasks (ROMAD), as well as a small dataset of logical bracelet-based anomalies (CELAD). ROMAD leverages a combination of DETR encoder-decoder to produce memory bank (during training) and anomaly candidates (during testing). These are then refined via SAM segmentation maps, producing corresponding features to both contrast against (as part of a separate memory bank), and used as queries (for test images). The resulting pipeline performs on state-of-the-art level in lower- to few-shot scenarios, and achieves highly competitive performance  in full-shot settings, tested on both the established MVTEC LOCO AD, and the newly introduced CELAD.

### Strengths
* The ROMAD pipeline is, to the best of my knowledge, a novel combination of memory-based AD methods combined with DETR & SAM detection plus segmentation for feature generation. The corresponding method pipeline seems very sensible. Given the pretraining of the associated modules and coupled with finetuning of the detection module, the resulting pipeline should be quite generally applicable.
* The corresponding performance on both MVTEC LOCO AD & their own CELAD benchmark is very convincing.
* CELAD itself is a novel, albeit rather small, contribution to the set of logical AD benchmarks & datasets.
* The paper itself is well written and structured.

### Weaknesses
The proposed method, and the corresponding benchmark, are two independent contributions. As CELAD itself is a simple additional benchmark for logical AD that does not exist yet, and the setup appears quite logical, I don't have any major issues to note there.
Regarding ROMAD, I have the following issues / questions:

* The detection module requires finetuning on manually annotated examples. It would be great if the authors could provide some context as to how the performance scales as a function of annotated examples, and how well it works out of the box without.
* Comparisons made in Table 1 includes methods that utilize components pretrained on different datasets and at different scales, s.a. e.g. EAD or SA-PC. It's always difficult to compare pipelines where technically, for competing methods, one could also consider swapping associated feature encoders with more modern or stronger ones. It would be great if the authors could provide additional information on how the numbers / relative differences should be best understood between different methods.
* How well can ROMAD be applied to non-logical AD benchmarks, e.g. plain MVTEC? It would be great to understand the limits of ROMAD as a general anomaly detection pipeline.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
