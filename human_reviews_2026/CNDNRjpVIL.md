# HEEGNet: Hyperbolic Embeddings for EEG

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Electroencephalography (EEG)-based brain-computer interfaces facilitate direct communication with a computer, enabling promising applications in human-computer interactions. However, their utility is currently limited because EEG decoding often suffers from poor generalization due to distribution shifts across domains (e.g., subjects). Learning robust representations that capture underlying task-relevant information would mitigate these shifts and improve generalization. One promising approach is to exploit the underlying hierarchical structure in EEG, as recent studies suggest that hierarchical cognitive processes, such as visual processing, can be encoded in EEG. Yet, most existing decoding methods rely on Euclidean embeddings, which are not well-suited for capturing hierarchical structures. 
In contrast, hyperbolic spaces, regarded as the continuous analogue of tree structures, provide a natural geometry for representing hierarchical data. In this study, we first demonstrate that EEG data exhibit hyperbolicity and show that hyperbolic embeddings improve generalization. Motivated by these findings, we propose HEEGNet, a hybrid hyperbolic network architecture to capture the hierarchical structure in EEG and learn domain-invariant hyperbolic embeddings. To this end, HEEGNet combines both Euclidean and hyperbolic encoders and employs a novel coarse-to-fine domain adaptation strategy. Extensive experiments on multiple public EEG datasets, covering visual evoked potentials, emotion recognition, and intracranial EEG, demonstrate that HEEGNet achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work focuses on the hyperbolic representation of the EEG signal to enhance classification tasks. Based on the demonstration of EEG showing hyperbolicity to improve generalization, the authors propose HEEGNet to capture hierarchical structure in EEG and learn domain-invariant information. HEEGNet achieves superior results in various EEG classification tasks.

### Strengths
1. The work leveraged hyperbolic geometry to represent EEG features, which gave technical novelty and a good basis for EEG classification tasks.
2. Extensive evaluations have been performed to demonstrate the usability of the HEEGNet. Various SOTA methods were involved for comparison.
3. The authors gave a clear description of the model and source code for reproduction.

### Weaknesses
1. The roadmap on why a hyperbolic manifold is necessary for EEG and further comparison with other methods is necessary to show the improvement of the current method.
2. The overall performance of HEEGNet outperformed other methods. However, it's not clear which part of the model contributes more to the final results. 
3. Apart from the classification results, it's not clear if the hyperbolic embeddings could help us find brain patterns precisely. Further analysis would be beneficial.
4. The motor imagery paradigm is an important paradigm to explore the significance of EEG geometry. It would be better to include the motor imagery comparison in the manuscript instead of the Appendix.
5. Several datasets from different paradigms were included for comparison. It would be better to clarify the validation manner to help understand the use of the methods. Is HEEGNet also useful in the generalization cases, such as cross-subject classification?

### Questions
1. Is the model only suitable for the EEGNet used in the paper, or also works for other methods like spatial-temporal convolution and some self-attention?
2. How much computational cost is taken by the hyperbolic calculation?
3. Is there a Discussion section for the work?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Overall this paper is working on a promising research direction of EEG-based BCIs/recognitions. 
However the paper needs to be significantly improved to better clarify both its contributions and performance comparisons.

### Strengths
1. The research direction of using hyperbolic embeddings for EEG-based BCIs/recognitions is emerging and promising. 
2. The proposed method is well explained. 
3. The performance improvements compared with EEGNet are significant.

### Weaknesses
1. The novelty is not well clarified; and the idea of exploring hyperbolic space /embeddings for EEG-based recognition is not new. The authors didn't clearly explain the difference. 
2. My main concern is about the performance comparison part. 
-- The proposed HEEGNet was not compared with other hyperbolic embeddings based methods, e.g., Jing Chang 2025 in the ref list and related refs/citations of that paper. 
-- For different tasks/datasets, it doesn't seem that the proposed methods were compared with SOTAs. Why only baselines were compared? The authors should compare their results with SOTA performances of the datasets, e.g., Seed, Faced. 
E.g., the results from comparison methods for VEP in table 3 are far away from SOTA performances reported in the literature. The gap between the proposed method and other methods is too good to believe. More clarification is needed.
-- It is also not clear whether the methods are tested in the cross-dataset setting. E.g, for the Emotion recognition task, are the Seed and Faced datasets combined or tested separately? 
3. Minor comment: Abstract is a bit misleading. Reviewers would think this is the first work on exploring hyperbolic embeddings for EEG.

### Questions
1. The authors should carefully clarify their novelty. 
2. The performance comparison part need to be significantly improved and justified.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a  hyperbolic architecture based on EEGNet for EEG‑based task with a novel domain‑shift alignment method in hyperbolic space.

### Strengths
The two-step alignment strategy, DSMDBN, operates in hyperbolic space: it first aligns source and target domain distributions by matching their first and second moments using a hyperbolic BatchNorm module, and then further aligns the moment-normalized features to a standard hyperbolic Gaussian via the Horospherical Sliced-Wasserstein (HHSW) loss. This approach is well-motivated and offers a moderately novel contribution to the domain adaptation literature.

### Weaknesses
1. Hyperbolicity evidence may reflect model bias:

The observed hyperbolicity is derived from learned embeddings rather than the intrinsic geometry of the EEG signals. Therfore, the results may reflect model-induced structure rather than inherent data-level hyperbolic properties.

2. Numerical stabilization not addressed:

The paper does not discuss numerical stabilization techniques essential for reliable hyperbolic training, such as gradient clipping, norm normalization, or overflow prevention during Lorentzian operations.

3. Missing comparisons with recent foundation models

This work lacks comparison with recent high-impact foundation models such as BIOT (NeurIPS' 2023) [1], LaBraM (ICLR' 2024) [2], and CBraMod (ICLR' 2025) [3]. These models are widely cited, have open-source code and model weights provided, and achieve state-of-the-art results across EEG decoding tasks.


Refs:

[1] Yang, Chaoqi, M. Westover, and Jimeng Sun. "Biot: Biosignal transformer for cross-data learning in the wild." Advances in Neural Information Processing Systems 36 (2023): 78240-78260.

[2] Jiang, Weibang, Liming Zhao, and Bao-liang Lu. "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI." The Twelfth International Conference on Learning Representations.

[3 Wang, Jiquan, et al. "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding." The Thirteenth International Conference on Learning Representations.

### Questions
1. How is the curvature K selected?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes HEEGNet, a hybrid hyperbolic embedding network designed to improve cross-domain generalization in EEG-based emotion recognition tasks. By jointly leveraging Euclidean and hyperbolic geometries, the model aims to capture both local and hierarchical relations within EEG representations. The proposed framework is evaluated on the SEED dataset and demonstrates competitive or superior performance compared to state-of-the-art methods. The manuscript is generally well-structured and technically sound, addressing an important challenge in EEG domain adaptation. However, several conceptual and methodological aspects require clarification, particularly regarding the motivation and interpretation of the hybrid embedding design and the underlying assumptions about EEG data structure.

### Strengths
1. The paper tackles an important issue—cross-domain generalization in EEG signals—where conventional Euclidean-based networks often fail to model hierarchical dependencies or inter-domain variations effectively.
2. The hybrid use of Euclidean and hyperbolic embeddings provides an interesting perspective, allowing the model to balance local feature encoding and hierarchical structural representation. This design is conceptually meaningful for EEG data, which may contain both flat temporal dynamics and hierarchical cognitive relations.
3. Experiments on SEED show consistent improvements over relevant baselines, suggesting that hyperbolic modeling can indeed improve domain transfer and robustness in EEG-based emotion recognition.

### Weaknesses
1. The manuscript claims that EEG, video, and language modalities can all be represented in hyperbolic space, but does not provide sufficient theoretical or empirical justification for why EEG data, in particular, exhibits hierarchical or tree-like properties that make hyperbolic geometry appropriate.
2. The rationale for combining Euclidean and hyperbolic representations is underdeveloped. It remains unclear what complementary features each space captures and how their joint use specifically benefits cross-domain transfer.
3. Terms such as “neural information” lack precision, and the formulation of domain adaptation within the SEED dataset (e.g., session-wise vs. subject-wise domains) should be clarified. These ambiguities make it difficult to interpret the mechanism behind improved generalization.
4. While t-SNE visualizations are provided, they reveal sub-clustering within the same class. The manuscript does not discuss why this occurs or whether it reflects subject-level variation, embedding instability, or meaningful substructure. Additional visualization or embedding analysis could better support the claimed advantages of hyperbolic modeling.

### Questions
1. Please clarify whether EEG, video, and natural language modalities share structural similarities that justify their representation in hyperbolic space. How do the manifold assumptions differ across these modalities?
2. Could you elaborate on the motivation for integrating Euclidean and hyperbolic embeddings? What distinct geometric properties does each space capture in EEG data?
3. The term “neural information” (Line 93, Page 2) is vague. Please specify the type of neural features it refers to—temporal, spectral, or spatial—and how they are encoded in your model.
4. In the SEED dataset experiments, how are domains defined? Are sessions or subjects treated as distinct domains, and how is cross-domain generalization evaluated?
5. It is recommended to visualize intermediate representations from both Euclidean and hyperbolic branches to more directly show their complementary effects and the benefits of the hyperbolic layer.
6. In Fig. 2(d), samples of the same class form multiple clusters. Please explain this behavior—does it arise from subject variability, intra-class diversity, or properties of hyperbolic projection?

### Soundness
3

### Presentation
3

### Contribution
3
