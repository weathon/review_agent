# GSPRec: Temporal-Aware Graph Spectral Filtering for Recommendation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Graph-based recommendation systems are effective at modeling collaborative patterns but often suffer from two limitations: overreliance on low-pass filtering, which suppresses user-specific signals, and omission of sequential dynamics in graph construction. We introduce GSPRec, a graph spectral model that integrates temporal transitions through sequentially-informed graph construction and applies frequency-aware filtering in the spectral domain. GSPRec encodes item transitions via multi-hop diffusion to enable the use of symmetric Laplacians for spectral processing. To capture user preferences, we design a dual-filtering mechanism: a Gaussian bandpass filter to extract mid-frequency, user-level patterns, and a low-pass filter to retain global trends. Extensive experiments on four public datasets show that GSPRec consistently outperforms baselines, with an average improvement of 6.77% in NDCG@10. Ablation studies show the complementary benefits of both sequential graph augmentation and bandpass filtering.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a novel approach that incorporates sequential information to construct an item–item relation graph and to design a bandpass filter. By integrating sequential patterns into user–item interactions, the authors argued that the model can better capture meaningful semantic relationships among users and items. In addition, the proposed method exploited not only low-frequency components, but also mid- and high-frequency components that encode localized and task-relevant variations.

### Strengths
(S1) The paper proposed a new approach to modeling item–item relations by explicitly utilizing sequential information in a bidirectional manner.

(S2) The newly proposed band-pass filter was theoretically well defined, enabling the model to exploit not only low-frequency components but also mid- and high-frequency components.

### Weaknesses
(W1) The method increases computational complexity, while the performance gain from the multi-hop diffusion component is relatively limited.

(W2) The experimental results for hyperparameter sensitivity showed an inconsistent tendency that does not align with the implementation details.

(W3) Most importantly, due to the fact that the method leverages additional information (e.g., sequential information) that is not typically available to conventional collaborative filtering baselines, the comparison with those baselines is not entirely fair.

### Questions
1. The proposed method constructed item-item relation graph using sequential information in a bidirectional formulation. However, in sequential settings where order is crucial, doesn’t a bidirectional formulation break the order signal?

2. There exists a graph signal processing-based model for sequential recommendation, namely FIRE (WWW’22). However, the paper did not provide a comparison between FIRE and the proposed model. It would be valuable if the authors could include a performance comparison between the two models.

3. The design of the filter function appeared somewhat arbitrary, as it adopted an exponential-shaped form without a clear theoretical justification. In addition, when the model filtering low-frequency components, it performed eigen-decomposition for the adjacency matrix. If the eigenbasis is already available, why not directly using mid- or high-frequency components instead of designing a bandpass filter?

4. In parameter sensitivity shown in Figures 4 and 5, the proposed models with w = 0.1 appeared to achieve the best performance in most settings. Nevertheless, the models treated w as a tunable hyperparameter. In addition, Table 3 represented the filter parameters across dataset, but these reported values do not seem to match the trends observed in parameter sensitivity. Could the authors clarify why w is left as a tunable hyperparameter, and how the reported parameter choices in Table 3 were determined?

5. In Tables 5 and 6, the results included the notation “SE”, which I interpret as the standard error. However, according to my understanding, the proposed method is based on graph signal processing and does not involve a learning or training process. In that case, the model should be deterministic, and it is not clear why there would be any standard error. Could the authors clarify the source of this standard error and how it was computed?

6. In the proposed model, the band-pass filter component in Eq (8) performs normalization on the item side, while the low-pass filter in Eq (9) performs normalization that jointly considers both users and items. I agree that each choice can be justified individually. However, I am not fully convinced that it is appropriate to apply two different normalization schemes to components derived from the same adjacency structure, and then simply combine them with a weighted sum at the end.

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
This paper proposes GSPRec, a temporal-aware graph spectral filtering approach for collaborative filtering. The method addresses limitations of existing graph-based methods that rely primarily on user-item interactions and low-pass filtering. The key contributions are: (1) a sequential graph construction approach that integrates temporal transition patterns from user sequences through symmetrization and multi-hop diffusion, capturing both direct and indirect item relationships; (2) a novel bandpass filtering design that targets mid-frequency spectral components to extract community-level personalization patterns, distinguishing it from prior work using low-pass or high-pass filtering; (3) a dual-filter architecture combining Gaussian bandpass and lowpass filters to balance personalized recommendations with global popularity. Experiments on four datasets demonstrate consistent performance improvements over baselines, with ablation studies showing bandpass filtering as the primary contributor. The work offers a frequency-aware perspective for analyzing collaborative signals in recommendation systems.

### Strengths
1. The paper introduces bandpass filtering to recommendation systems, explicitly targeting mid-frequency components—a distinct approach from prior GSP methods that focus on low-pass or high-pass filtering. The dual-filter architecture combining bandpass and lowpass filters represents a new design for balancing personalization and popularity in spectral collaborative filtering.

2. The paper offers a new lens for understanding collaborative signals through spectral analysis, proposing that different frequency components may capture different aspects of user preferences (e.g., low-frequency for global popularity, mid-frequency for community patterns). While the universality of this hypothesis requires further investigation (as noted in Weaknesses), this frequency-aware perspective provides a valuable framework for analyzing recommendation signals.

3. Experimental results demonstrate the method's effectiveness across multiple datasets with varying characteristics. The approach shows consistent improvements over baseline methods, and ablation analysis indicates that bandpass filtering contributes substantially to overall performance. The results suggest that the proposed frequency-aware filtering approach offers practical benefits for recommendation tasks.

### Weaknesses
1.  Unclear Utilization of Temporal Information
	The paper claims to be "Temporal-Aware" throughout, but the utilization of temporal order is not clearly demonstrated. Lines 196-198 symmetrize the sequential transition matrix (S⁰[i,j] = 1 if i→j OR j→i exists), and Equation (3) produces a symmetric matrix S̃ where S̃[a,b] = S̃[b,a]. While the authors note this symmetrization is required for spectral analysis (lines 183-185), it remains unclear how the method distinguishes "a→b" from "b→a" given that undirected graphs with symmetric adjacency do not naturally encode directionality. The method may primarily capture sequential co-occurrence rather than temporal order. The authors should clarify how directional information is preserved or acknowledge this as a limitation.

2.Parameter Inconsistencies with Sensitivity Analysis
	Figure 4 reveals apparent contradictions with Table 3 for the Beauty dataset. The sensitivity curves suggest better performance at lower c values (around c=0.25 with w=0.1), while Table 3 reports c=0.8 and w=0.3 as optimal. Additionally, the monotonic decrease in Figure 4 appears inconsistent with the paper's hypothesis that mid-frequency (c≈0.5) captures optimal personalization (lines 294-299). This raises questions about: (1) which parameters were used for Table 4 results, (2) whether the mid-frequency hypothesis applies to all datasets, and (3) whether bandpass filtering is necessary for Beauty. Beauty-specific ablation experiments (analogous to Table 5) would help clarify these points.

3. Missing Comparisons with Sequential Baselines
	Table 1 lists BERT4Rec and SASRec as relevant methods, but Table 4 does not include experimental comparisons with them. Given that these methods preserve temporal order through positional encodings and attention mechanisms, they would serve as appropriate baselines for evaluating "Temporal-Aware" claims. Comparisons with these sequential recommendation approaches would help assess the trade-offs between the proposed graph-based approach and methods that explicitly maintain temporal information.

### Questions
1.How is directional information from sequences preserved after symmetrization, and should the "Temporal-Aware" claim be reconsidered?
2.What explains the parameter discrepancy between Table 3 (c=0.8, w=0.3) and Figure 4 (suggesting c≈0.25, w=0.1) for Beauty? Which parameters were used for Table 4?
3.Why were BERT4Rec and SASRec (listed in Table 1) not included in experimental comparisons?
4.Can the authors provide Beauty-specific ablation experiments and clarify the definitions of ablation variants (GSPRec-NB/NL/NS/SE)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes GSPRec, a graph-spectral recommendation model that first turns user sequences into a symmetric item-item graph via multi-hop diffusion and then applies a dual spectral filter, which is a Gaussian bandpass to mine mid-frequency, user-specific patterns and a lowpass to keep global popularity, before fusing both signals to predict the next interaction.

### Strengths
1. By transforming raw click sequences into a symmetric graph before spectral filtering, GSPRec preserves temporal cues while maintaining the stability of eigendecomposition.

2. The entire pipeline requires only matrix decomposition and lightweight filters, demonstrating high efficiency as shown in the experiments.

3. The conducted experiments demonstrate the superior performance of the proposed method.

### Weaknesses
1. Throughout the paper the mid-frequency Gaussian bandpass is said to capture ‘user-specific sequential patterns’. Is there any case study to demonstrate it?

2. The experiments only report aggregate top-k accuracy, but emphasizing mid-frequency components may systematically boost long-tail items and thus change the exposure distribution. Could the authors include fairness-aware metrics in the experiments?

### Questions
See Weakness.

### Soundness
3

### Presentation
2

### Contribution
3
