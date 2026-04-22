# Geometry-Aware Metric for Dataset Diversity via Persistence Landscapes

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 2, 6

## Abstract
Diversity can be broadly defined as the presence of meaningful variation across elements, which may be viewed from multiple perspectives, including statistical variation and geometric structural richness in the dataset. 
Existing diversity metrics, such as feature-space dispersion and metric-space magnitude, primarily capture distributional variation or entropy, while largely neglecting the geometric structure of datasets. To address this gap, we introduce a framework based on topological data analysis (TDA) and persistence landscapes (PLs) to extract and quantify geometric features from data.
This approach provides a theoretically grounded means of measuring diversity beyond entropy,  capturing the rich geometric and structural properties of datasets. Through extensive experiments across diverse modalities, we demonstrate that our proposed PLs-based metric (PLDiv) is powerful, flexible, and interpretable, directly linking data diversity to its underlying geometry and offering new insights for dataset construction, augmentation, and evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission focuses on the diversity measure of datasets, which is broadly defined as the presence of meaningful variation across elements. Existing methods primarily consider distributional variation or entropy. This submission futher considers the geometric structure of datasets. To this end, a framework (PLDiv) ) based on topological data analysis (TDA) and persistence landscapes (PLs) is used to characterize the geometric structure. PLDiv is able to link data diversity to its underlying geometry as shown in the experiments.

### Strengths
+ It is useful to consider the geometric structure for the diversity measure. This submission uses the topological data analysis (TDA) and persistence landscapes (PLs) to reflect geometry

+ PLDiv has several mathematical properties: it satisfies key diversity axioms (effective size, monotonicity, twin property, symmetry)

### Weaknesses
- [**Validation of Geometry-awareness**] The experiments include image embeddings of Colored MNIST and text embeddings of prompt tasks based on MiniLM embeddings. However, there’s no large-scale empirical validation on diverse real-world datasets (e.g., ImageNet variants, multimodal data, or foundation model embeddings).

- [**Comparison with prior metrics**] First, the experiments include Vendi Score, DCScore, and MAGAREA; it does not clearly demonstrate the new information offered by PLDiv. Second, according to Fig. 5, it is hard to see the advantage of PLDiv over other methods. Also, the subset selection in Fig. 3 needs clarification on the advantage of PLDiv

- [**Discussion on sensitivity**] PLDiv depends on distance metrics and filtration parameters, but the paper lacks an analysis of its sensitivity to noise, scaling, or feature normalization

- [**High computational cost**] According to Table 2, PLDiv remains significantly slower than simpler alternatives like Vendi Score or DCScore. Then, there is a need to discuss the scalability of PLDiv, especially for large-scale and high-dimensional datasets.

### Questions
- The paper claims PLDiv can guide dataset design and evaluation, but provides no quantitative link between PLDiv scores and downstream model performance or robustness.

- The paper does not include a discussion of its own limitations. Please discuss the scenarios where PLDiv may fail or provide unreliable results. For example, in high-dimensional sparse embeddings or when data points are highly correlated?

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
The paper proposes PLDiv, a new diversity measure based on persistent homology (PH), which aims to quantify diversity in data from a geometric and topological perspective. By using 0-dimensional persistence landscapes, PLDiv captures clustering behaviour and satisfies four axioms of diversity. PLDiv is computed for synthetic, image, and text datasets. Results show the potential of PLDiv for capturing geometric information by (i) distinguishing simulated point patterns, (ii) predicting curvature, (iii) measuring the diversity of generated text as determined by the softmax temperature, and (iv) correlating with the number of unique classes sampled from an image dataset when computed from image embeddings. Further, empirical runtimes are compared to alternative diversity measures.

### Strengths
- The text is clearly written and the main question it addresses on how to measure diversity is relevant in application.

- The application of persistent homology (PH) to diversity evaluation for ML tasks is novel and theoretically interesting. 

- The proposed summary, PLDiv, is geometry-aware and measures a notion of diversity related to 0-dimensional PH features e.g. clustering behaviour.

- The proposed diversity measure is theoretically motivated and fulfills four basic axioms of diversity.

- Evaluation across multiple data modalities (synthetic, image, text) demonstrates the flexibility of the proposed diversity measure.

- It is beneficial that uncertainty is evaluated for some of the results (Table 1 and Figure 4) and that a sparsification approach is considered to speed-up PH computations.

### Weaknesses
The paper’s empirical and conceptual claims are not fully substantiated, which limits its impact.

**Overstated Claims about Geometry-Awareness:** The paper claims that existing diversity measures “neglect geometric structure”, or do not "genuinely consider data from a geometric perspective". However, measures like VS, MagArea, and MagDiff can be computed from the same pairwise distance matrices used to calculate PLDiv and are inherently geometry-dependent. Exactly how geometry is summarised by either of these diversity measures is a more nuanced debate that requires clearer formalisation or further empirical investigation. 

**Limited Advantages over Baselines:** The practical benefits of PLDiv over existing diversity measures, such as the Vendi Score (VS), MagArea, or MagDiff are not convincingly demonstrated (as detailed below). It would be of interest to show stronger examples of diversity evaluation tasks that uniquely require PLDiv and cannot be addressed by established measures. 

**Point Pattern Analysis:**  Figure 3 shows that PLDiv can distinguish between simulated point patterns of varying diversity. However, it is not evaluated  if PLDiv has an unique advantage at this task or if other diversity measures could also tell the difference between these examples. 

**Curvature prediction (Table 1):** VS and MagArea can predict curvature with MSEs of 0.05 compared to PLDiv with an MSE of 0.04 (see Limbeck et al. and Turks et al. for comparison). Given the experimental setup, and the small absolute difference in MSE, this does not seem like conclusive evidence that PLDiv is uniquely geometry-aware. For further clarifications, it would also be of interest to simply report the values of each diversity score plotted against the curvature values.

**Text Evaluation (Figures 4, 6, 7,8):** Considering uncertainties, all diversity measures reach similar Spearman correlations on the response task, and PLDiv shows the lowest Spearman correlation on the story task. The authors state that PLDiv exhibits linear correlation, but if the stronger linear relationship is its main advantage over other diversity measures, Pearson correlation should be evaluated instead. Questions of interest beyond the correlation with the softmax temperature, such as the alignment with human evaluation scores are not evaluated. The reported text evaluation tasks already seem to be sufficiently addressed in more detail by baseline diversity measures (see Zhu et al. or Limbeck et al. for comparison).

**Image Evaluation:** MagArea seems to show very similar trends and perform on-par with PLDiv in Figure 5. To clarify whether PLDiv performs best at this experiment, it would be beneficial to report uncertainties e.g. by repeating the experiment across varying different seeds or subsamples. Further, it would be relevant to specify which parameter and distance choices have been used to compute alternative diversity measures and how these impact results to e.g. discuss why the DCScore here shows larger fluctuations than reported by Zhu et al..

**Limited Scalability:** PLDiv shows e.g. MagArea with the parameter settings reported here. However, the runtimes of PLDiv are not necessarily advantageous over diversity measures that are computed at one fixed scale of (dis)similarity e.g. the VS score, which is still notably faster than PLDiv for the examples in Table 2.. 

**Reproducibility:** Questions remain on the reproducibility of the experimental results. A reproducibility section and relevant supplementary materials detailing the specific experimental setups and parameter choices would improve transparency.

**Theoretical Novelty:** Existing entropy-based diversity measures are also theoretically justified and the authors state no unique theoretic property of PLDiv that is not also already fulfilled by an alternative diversity measure.

**Overall Assessment:** This paper introduces a valid and mathematically sound approach to measure diversity via persistent homology. However, the claimed advantages over existing diversity measures are not sufficiently substantiated, both conceptually and empirically. The work would be strengthened by more careful positioning relative to prior methods, clearer definitions of what “geometry-aware” means in this context, and broader, reproducible experiments that highlight for exactly which datasets PLDiv provides unique insights.

### Questions
**Main Questions:**

- Line 139 claims that magnitude-based methods "abstract away the geometric or topological structures that can differentiate datasets with the same dispersion."
Can the authors provide examples where PLDiv distinguishes datasets that alternative diversity measures cannot?

- Does PLDiv fulfill any theoretic properties relating to diversity (beyond what is shown in 4.2.) that are not fulfilled by other diversity measures ? 

- Why is PLDiv proposed as a new summary of PH rather than applying existing summary statistics? Would the trends shown by PLDiv be different to e.g. total persistence or other one-number summaries of PH? Further comparison could be relevant to clearly motivate the introduction of PLDiv. 

- Is PLDiv (sufficiently) scalable to large datasets compared to e.g. the Vendi Score? Further empirical evaluation on increasing sample sizes would be of interest.

- Could PLDiv also be generalised via alternative filtrations (e.g., using cosine distances) to capture structure in other modalities such as language embeddings?  Would being similarity-dependent then not be a strength rather than a limitation?

- Which implementations of the alternative diversity measures, dissimilarity choices, and other parameter choices have been used to report the results in Table 2, Figure 4 or Figure 5? Does the choice of dissimilarity impact results and does it ensure a fair comparison?

---

**Further Questions on Reproducibility:**

- How is the k-DPP sampling implemented and can you cite a reference?

- How exactly is the experiment in Section 5.4. simulated? Does the number of observations stay the same or does it decrease as the number of labels decreases? 

- Do you have an intuition on why PLDiv shows the lowest standard deviation in the MSE in Table 1? 

- Missing implementation details are given as a reason for not reporting MagArea in Figure 4. Why could MagArea be computed for other experiments but not for this one? Could it not be calculated from e.g. the same distances used to calculate PLDiv?

- Which dataset is Table 5 computed on?

- Line 062 cites Bubenik et al. directly after the statement that "curvature is inherently linked to diversity", but their paper never once mentions diversity. Isn’t the statement taken from another reference?

- Conclusions state that “these results establish PLDiv as a versatile tool for dataset construction, augmentation, model evaluation, and robustness analysis”. But as far as I am aware, dataset construction, augmentation, and robustness analysis are not explicitly analysed in the experiments?

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
The paper proposes a new dataset diversity metric, PLDiv, based on topological data analysis (TDA) and persistence landscapes. The authors argue that existing diversity measures—such as entropy, magnitude, or kernel-based metrics—fail to capture geometric structures in the data. By leveraging persistent homology and integrating the resulting “tent” functions, PLDiv aims to provide a geometry-aware quantification of diversity. The paper claims theoretical justification (via diversity axioms) and reports experiments across synthetic data, text embeddings, and image embeddings, suggesting that PLDiv correlates with intuitive diversity notions.

### Strengths
1). The topic of dataset diversity measurement is important and increasingly relevant for evaluating generative models and representation learning.

2). The exposition is clear and the paper is well written overall, with some effort spent on theoretical motivation.

### Weaknesses
1). Lack of substantial novelty. The proposed PLDiv essentially computes a quadratic function of the lifetimes in persistent homology (𝑃𝐿𝐷𝑖𝑣=1/4∑𝑖(𝑑𝑖−𝑏𝑖)^2). This is a straightforward statistic derived from standard persistence diagrams, not a fundamentally new methodology. The connection to diversity measurement appears superficial, and the theoretical content (axioms, proofs) largely restates basic TDA properties rather than introducing new insights.

2). Weak theoretical contribution. The “axiomatic” analysis simply verifies trivial properties (monotonicity, symmetry, etc.) that hold for any nonnegative function of lifetimes. There is no rigorous justification that PLDiv truly measures diversity in a meaningful or generalizable sense, beyond restating intuition.

3). Empirical evaluation lacks depth and rigor. The experimental settings are small-scale and largely qualitative. Comparisons with existing metrics (Vendi Score, DCScore, MAGAREA) rely on correlation or regression but lack clear baselines, statistical tests, or ablation analyses. There is no real-world application where PLDiv produces actionable improvements or insights.

4). The method requires persistent homology computations over the full distance matrix, which is expensive for large datasets. The “sparse” approximation in Table 2 still consumes considerable time without clear scalability to modern data scales.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes PLDiv, a diversity metric for datasets based on persistence landscapes from topological data analysis (TDA). The authors argue that existing diversity measures, such as entropy- or kernel-based approaches, largely overlook geometric and structural properties of data. PLDiv quantifies diversity by integrating persistence landscapes derived from persistent homology, capturing both local and global topological features. The method is evaluated across synthetic, text, and image datasets, demonstrating advantages in interpretability, robustness, and geometry-awareness over baseline methods like Vendi Score, DCScore, and MAGAREA.

### Strengths
1.	Theoretical Grounding: The paper provides a closed-form expression for PLDiv and proves it satisfies key diversity axioms (effective size, monotonicity, twin property, symmetry), enhancing its credibility.
2.	Empirical Validation: Extensive experiments across modalities (synthetic, text, image) and tasks (subset selection, curvature regression, semantic diversity) show PLDiv’s consistency and superiority over competing methods.
3.	Interpretability: The connection between persistence lifetimes and diversity offers an intuitive and geometrically meaningful interpretation.

### Weaknesses
1.	The fundamental limitation of the proposed metric is that its geometric perspective on diversity may not align with human intuition, which greatly hinder its practical value. Please consider this example: in the experiment of Section 5.2, do texts generated with a higher temperature truly possess more meaningful diversity that mitigates the self-enforcing homogenization, which is the objective declared in the paper, or do they merely produce more variations of "LLM-ish" text?
2.	Computation Cost: Although sparse PLDiv is proposed, the full method remains computationally expensive compared to lighter baselines like Vendi Score or DCScore.

### Questions
1.	As the scale of training data is continuously increasing in practice, how does PLDiv scale to very large datasets in terms of cost, e.g. time and memory? Is it possible that the algorithmic complexity be explicitly stated?
2.	How sensitive is PLDiv to the choice of distance metric or embedding model, especially in semantic tasks where embedding quality varies significantly?
3.	As the Weakness 2 stated, there is a gap between the degree of geometric dispersion and semantically rich diversity in data or generated content. In this regard, what is your perspective?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a topology-based diversity metric, PLDiv, which quantifies dataset diversity without relying on labels or reference sets. By integrating persistence landscapes over Vietoris–Rips filtrations, it captures both spatial dispersion and topological connectivity. Experiments on curvature point clouds, text, and image embeddings show that PLDiv effectively characterizes the structural properties of embedding distributions.

### Strengths
1. The paper introduces a novel perspective to dataset diversity measurement by integrating topological persistence with geometric embeddings.

2. The experiment results shows this paper effectively captures structural properties in both geometric point clouds (e.g., curvature data) and high-dimensional embedding spaces (e.g., textual and visual representations), demonstrating strong generality and adaptability across different data modalities.

3. The paper is generally clear and well written and the methodology is explained step by step with illustrative figures.

### Weaknesses
1. Given the potential computational and memory overhead of the topological process, it would be important to evaluate how the proposed method performs on large-scale datasets.

2. Although the method is applied to high-dimensional embeddings, the paper lacks a systematic analysis across different feature dimensions.

3. If the dataset follows a long-tailed or highly imbalanced distribution, it remains unclear how this diversity measure should be interpreted or whether it might be dominated by the majority clusters.

4. Unlike prior diversity metrics such as Vendi Score, DCScore, or MAGAREA, which include human-annotated or human-evaluated benchmarks to validate semantic diversity, this paper relies solely on embedding-level geometric correlations without external semantic validation.

5. Lack of discussions of theoretical or empirical analysis of space complexity, leaving the memory scalability of PLDiv unclear.

6. The authors could clarify the distinction between \epsilon (filtration) and \epsilon (sparsification rate).

7. The definition, proposition and remarks in Section 4 are numbered as (3.x) instead of (4.x).

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
