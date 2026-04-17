# Hierarchical One-Class Data Description via Probabilistic Granular-ball Computing

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
One-class data description aims to model the distribution of target data by constructing a compact representation of the target class. This approach is widely applied in tasks like anomaly detection, where the objective is to differentiate target data from outliers. Traditional methods typically rely on single-sphere or pre-defined multi-sphere representations. However, these simplistic assumptions often fail to capture the anisotropic structures and intricate patterns present in real-world data, limiting their effectiveness in representing distributions across multiple scales. 
To address these limitations, we propose Probabilistic Granular-ball Computing (PGBC), a hierarchical framework for one-class data description. PGBC uses ellipsoidal granular-balls to align with the anisotropic geometry of data and recursively refines them through statistical splitting, achieving precise and adaptive data representation. Moreover, PGBC approximates a hierarchical Gaussian mixture model by aggregating data description scores via granular-ball distribution entropy at each layer. This enables PGBC to capture data patterns at multiple levels of granularity, modeling both global structures and fine local variations. 
Extensive experiments on benchmark datasets demonstrate that PGBC consistently outperforms related strong baselines, offering superior accuracy for hierarchical one-class data description while maintaining a low false positive rate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Probabilistic Granular-Ball Computing (PGBC), a novel framework for hierarchical one-class data description, particularly suited for anomaly detection. Unlike traditional methods that use simple spherical boundaries, PGBC constructs ellipsoidal granular-balls that adapt to the anisotropic geometry of real-world data. These granular-balls are refined recursively using statistical criteria, enabling multi-scale representation of normal data. PGBC also builds a hierarchical Gaussian mixture model, where anomaly scores are aggregated across layers using entropy-based weighting, allowing detection of both global and local deviations. Experiments show PGBC consistently outperforms strong baselines with higher accuracy and lower false positive rates, making it a robust solution for complex anomaly detection tasks. Experiments are conducted on teh

### Strengths
- The paper is generally well-written and well-organise. The structure and the motivation of the paper is clear.
- The idea of modeling anisotropic geometries is interesting.
- The paper evaluates the proposed method on various datasets and demonstrates its effectiveness.

### Weaknesses
- It is not clearly articulated how the proposed method significantly differs from or advances beyond GBDO in terms of conceptual innovation or technical contribution.
- The proposed PGBC framework relies on BIC and log-likelihood gain (LLG) as statistical criteria to determine whether a granular-ball should be recursively split. While these are standard model selection metrics, I wonder whether they are sufficient to robustly prevent over-partitioning, especially in high-dimensional.
- The experimental datasets used in the paper are primarily tabular. However, considering that DeepSVDD has been successfully applied to computer vision datasets such as CIFAR and MVTec—which are widely used benchmarks for one-class classification—I wonder whether the proposed method can be naturally extended to such scenarios. It would be interesting to explore how PGBC performs on these vision-based benchmarks.
- Would the proposed method remain effective if the training data is contaminated with a small proportion of anomalous samples?

### Questions
n/a

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
This work proposes PGBC (Probabilistic Granular ball Computing) by combining the GB tree with Gaussian, to describe intrinsic structures and patterns of the data. Grounded in the nature of PGBC for one-class description, this work designs a hierarchical anomaly score for anomaly detection task. The empirical evaluation on tabular and time-series demonstrates the effectiveness of the proposed method.

### Strengths
1.	The work is well-written with clear organization, statement and illustrations.
2.	The proposed method is theoretically reasonable for anomaly detection under one class classification.
3.	The evaluation given in the manuscript seems to be convinced and could demonstrate the superior detection performance to some extent.

### Weaknesses
1.	There is a defect for current version of the PGBC. Based the pseudo-code of hierarchical anomaly scoring (Algorithm 2), this algorithm only works when $l \geq 1$ that means it is ill-defined for the Granular ball Queue $\mathcal{Q} = [B^{(0)}]$. The case is possible, right? In other words, current algorithm is valid only for the data with more intrinsic structures. This is crucial for the rigor and completeness of the proposed algorithm.

2.	The time complexity (and real implementation cost) both on the training and inference has not been analyzed and compared. From my view, the time cost maybe is a big problem for the proposed method both in training and inference since there is an iterative GMM process during the training and iterative density estimation during the inference.

3.	There is a lack for some key ablation and exploration analysis.
 - 1). What is the relation between detection performance and the depth of GB Tree or the number of granular balls? If such analysis can help to decrease the inference time cost?
 - 2).  Authors mention the importance of the dynamic reassignment. The related ablation study should be included to provide the empirical evidence.
 - 3).  what if only use the granular balls as leaves to compute the anomaly score?
 - 4).  The proposed method is developed from GBC. Consequently, the GBC should be included as a baseline and the anomaly score should attempt the vanilla version (showing in Figure 1) and another simple strategy (the average distance to the centroids of all granular balls)


4.	Unconvinced empirical evaluation and findings in Section 3.5.

The findings presented in Section 3.5 may lack generalizability, primarily due to the limited scope and specialized nature of the datasets used. The tabular datasets in Table 1 maybe are better choice for demonstrating performance against the GBC.

5.	Limited experiments

If possible, complementing experiments on more benchmark datasets, (like ADBench) to improve the statistical reliability.

### Questions
1.	The issue illustrated in Figure 1 seems to be overcame easily if calculating anomaly score using the average distance to the centroids of all granular balls. The proposed method needs to more suit-well motivation.

2.	Why the hierarchical anomaly scoring is important? Some empirical evaluation should be provided if without theoretical guarantees.

3.	The normalization for $s^{(l)}$ is necessary? and why?

4.	Authors mentioned that the proposed method captures both global and local features, but based on the design of algorithm 2, the entropy weight for $B^{(0)}$ is always zero (of course, I noticed that the $B^{(0)}$ is eliminated in algorithms 2 which also leads to other issue). So, how to understand the global feature and it is possible to utilize the $B^{(0)}$ reasonably?

5.	Where is randomness from for LOF, KNN and the proposed method PGBC since I noticed that the related hyperparameters are fixed?

6.	I understand the experiments on tabular datasets, but how to obtain the results on time-series showing in Figure 4? It seems to yield the anomaly score in each time point. Please provide more explanation to help me understand this part.

7.	For the task: time series open-set recognition, I don’t think this is a different task compared to the anomaly detection on the tabular datasets, particularly the embeddings of data are used in experiments.

8.	More explanation about the statement “these findings highlights …” in line 430-432.

9.	While the FPR is a crucial metric for anomaly detection systems, the FNR (False Negative Rate) is often more of greater practical significance in real applications, like medical diagnosis, intrusion detection in cybersecurity. I suggest including a FNR analysis for the proposed method.

**Minor errors:**

By aligning Figure 2 with Algorithm 1, the $B^{(1)}_j$ in Figure 1 should be $B^{(0)}_j$, right?

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
4

### Summary
The paper proposes Probabilistic Granular-Ball Computing (PGBC), a hierarchical one-class data description method designed to model complex, anisotropic data distributions. PGBC extends granular-ball computing by introducing ellipsoidal, probabilistic components that adapt to local data geometry through covariance estimation. The model recursively refines data representations via statistical splitting using BIC and log-likelihood criteria and incorporates entropy-weighted hierarchical scoring for anomaly detection. Experiments on tabular, time series, and open-set datasets show that PGBC outperforms existing baselines (e.g., DeepSVDD, DAGMM, HGAD, and granular-ball variants) in AUC and false positive rate.

### Strengths
- Conceptual novelty: Extends granular-ball computing into a probabilistic, hierarchical framework with ellipsoidal modeling, enabling anisotropic and multiscale representation of data.

- Statistical rigor: Splitting and reassignment are governed by BIC and log-likelihood improvement criteria, avoiding ad hoc partitioning.

- Hierarchical anomaly scoring: The entropy-weighted aggregation across levels provides a principled way to combine coarse- and fine-grained representations.

- Comprehensive experiments: Evaluations span 19 tabular, 6 time series, and 5 open-set benchmarks, providing convincing empirical evidence.

- Robustness and efficiency: The paper reports competitive AUC with lower false positive rates and significant model compactness

### Weaknesses
- Limited theoretical analysis: The work lacks a formal justification or convergence analysis for the recursive splitting and entropy-based weighting. The connection to mixture model theory remains largely empirical.

- Comparison coverage: The baselines are extensive, but the paper omits recent deep one-class or diffusion-based anomaly detection methods (e.g., OCFlow, CutPaste, or normalizing-flow variants beyond HGAD).

- Ablation and sensitivity analysis: The paper does not quantify the impact of each component (e.g., BIC criterion, entropy weighting, or covariance adaptivity) on performance. This limits interpretability of where the gains come from.

- Scalability concerns: Although PGBC reduces the number of components, its per-iteration Gaussian fitting and reassignment steps may become computationally expensive for very large datasets. No runtime comparison is provided.

- Presentation clarity: While figures are informative, the text is at times dense and overly detailed in algorithmic description. The narrative could be streamlined to highlight key intuitions.

- Evaluation metric bias: The dominance of AUC as the main metric could obscure limitations under other measures (e.g., precision-recall or calibration error).

### Questions
- How sensitive is the entropy-based weighting to the number of hierarchical levels or to irregular layer sizes? Would uniform or learned weights perform similarly?

- Does the probabilistic splitting procedure risk overfitting when the local data variance is small but noise-induced? Is there any early-stopping or regularization strategy beyond the BIC penalty?

- How does PGBC perform under distributional drift or streaming data conditions? The conclusion mentions potential for online adaptation—has any preliminary evaluation been conducted?

- Given that each granular-ball corresponds to a Gaussian, can the resulting model be interpreted as an adaptive Gaussian mixture? If so, how does it compare to incremental or variational GMMs in both likelihood and efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work tackles the problem of anomaly detection using one-class data description. To incorporate the diverse nature of available samples, it proposes a probabilistic granular ball computing framework, where samples are aggregated in hierarchical clusters, maintaining an ellipsoidal structure along the principal component. By leveraging the hierarchical structure, an aggregated entropy-based metric is proposed, which acts as an anomaly score to distinguish between normal and anomaly samples in testing. Comprehensive experiments are conducted against 19 tabular datasets, 6 time series anomaly datasets, and 5 time series open-set recognition datasets.

### Strengths
- The paper is easy to follow, with intuitive illustrations of the proposed concepts.
- The motivation to propose an ellipsoidal structure along the principal component is well-formulated.
- Experiments are comprehensive with a diverse range of datasets.

### Weaknesses
- The performance is sub-optimal in a significant number of cases.	
- Anomaly detection problem can be approached from an open-set recognition (OSR) perspective as well. It is suggested to include comparisons with some of the baselines from OSR [1,2].
- It is suggested to include the ablation of the proposed components. For example, how does it impact the performance if the dynamic reassignment step is omitted? 


References:

[1] Zhang, Hongjie, et al. "Hybrid models for open set recognition." European Conference on Computer Vision. Cham: Springer International Publishing, 2020.

[2] Bendale, Abhijit, and Terrance E. Boult. "Towards open set deep networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

### Questions
- How can this framework be extended to vision datasets?
- Is one-class data description applicable to other fields than anomaly detection?
- Does the method support a single principal component, or the method can be extended?

### Soundness
3

### Presentation
4

### Contribution
3
