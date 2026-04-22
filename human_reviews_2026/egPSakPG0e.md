# Beyond Single Views: Achieving Significant Gains in Text Clustering via Informative Diversification

- Avg Score: 2.40
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2, 2

## Abstract
Clustering text into coherent groups is a long-standing challenge, complicated by high-dimensional embeddings, semantic ambiguity, and distributional shifts in unseen data. Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) systems have further underscored the need for robust and scalable knowledge representation methods. In this work, we introduce a novel clustering framework based on informative diversification. Our method applies a set of semantic-preserving transformations to generate multiple views of the data, and then harnesses their collective structure through a spectral consensus process. We prove that consensus clustering achieves an exponentially lower expected error rate compared to any single view, provided the views are diverse and informative. We then propose an iterative co-training procedure that learns a cluster-friendly latent space by jointly minimizing a contrastive InfoNCE loss and a Gaussian mixture negative log-likelihood loss. This training sharpens assignments and pulls embeddings toward their cluster centroids, while dynamically updating cluster assignments to accommodate the evolving latent space. The result is a robust and generalizable model that not only outperforms baselines on benchmark datasets but also maintains strong accuracy on unseen text, making it a powerful tool for real-world knowledge discovery and retrieval-augmented generation systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new multi-view consensus clustering framework for text data, designed to overcome the limitations of single-view clustering methods. It leverages informative diversification—creating multiple, semantically varied versions of embeddings—to achieve lower misclustering error and higher robustness.

The paper theoretically proves that, in multi-view consensus clustering, the expected misclustering error decays exponentially with the number of views (m), under diversity and informativeness conditions, whereas single-view clustering retains a positive lower bound on the misclustering rate.

The proposed methodology is three-fold: 1. Multi-view generation;
2.Consensus clustering, and 3. Latent space learning.  Experimentally,
the proposed consensus clustering consistently outperforms baseline
methods such as K-Means, single-view GMM, and Spectral Clustering.

### Strengths
1. The paper introduces a novel approach that integrates multiple
semantically diverse embeddings (“views”) into a spectral consensus
clustering framework. This idea—aggregating information from
different embedding transformations—represents a creative extension
of ensemble learning to modern text clustering, improving stability
and robustness over single-view methods.


2. It provides formal proofs showing that the expected misclustering
error decreases exponentially with the number of independent,
informative views.  The derivations connect clustering performance
with statistical guarantees, offering a clear theoretical
justification for why and when multi-view consensus is superior.


3. The method elegantly combines contrastive learning (InfoNCE loss) with
Gaussian Mixture Modeling (GMM) in a joint optimization loop.  This
hybrid objective balances representation learning and cluster density
modeling, yielding embeddings that are semantically rich.


4. Experiments show consistent improvements over baseline methods
(K-Means, GMM, Spectral Clustering) across multiple configurations.
The model maintains robust clustering accuracy on unseen data,
demonstrating generalization beyond the training set—a key challenge
in unsupervised learning.


5. The algorithms (Algorithm 1 and 2) are clearly described,
step-by-step, with well-defined mathematical notation and transparent
design choices (e.g., transformation types, consensus computation).
The combination of deterministic and stochastic transformations (like
PCA, WPT, Gaussian noise) provides practical reproducibility for
future studies.

### Weaknesses
1. The evaluation is narrow, using only two clean English datasets
(DBPedia and Reuters R8). There’s no evidence of scalability to
large, noisy, or multilingual corpora, nor any analysis of
computational cost.



2. The proof of exponential error reduction assumes that the multiple
views are independent and informative. In reality, the generated views
(e.g., PCA or similar BERT models) are highly correlated, so these
conditions are unlikely to hold.


3. The method is only compared against basic clustering algorithms
(K-Means, GMM, Spectral Clustering), omitting modern deep or
contrastive clustering baselines. There is also no ablation or
sensitivity analysis to show which components truly drive the
improvement. This limitation appears in the short length of the reference list.
Clustering is one of the most extensively studied areas in machine
learning, and any new significant proposal should relate to much more
recent related work—most of which this paper ignores.




4. "Informative diversification” is not formally defined or adaptively
measured; the approach relies heavily on pretrained embeddings without
clarifying how much the gains come from the transformations versus the
base models. This weakens interpretability and reproducibility.

### Questions
1. The paper only benchmarks against K-Means, GMM, and Spectral
Clustering. How would it perform against modern methods like DEC,
IDEC, SCAN, or graph-based and contrastive clustering models that
already integrate multiple representations?


2. The framework requires multiple clustering runs and spectral
decomposition steps. How well does it scale with large corpora or
high-dimensional embeddings, especially when the number of views (m)
increases?


3. Since the model relies heavily on high-quality sentence embeddings
(e.g., from BERT), are the observed gains primarily due to the
diversification strategy, or simply from strong pretrained
representations?



4. The theoretical guarantees rely on mutually independent and
informative views. However, deterministic transformations (e.g., PCA
or similar BERT encoders) are highly correlated. Can the claimed
exponential error decay still hold when view independence is violated?

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
4

### Summary
The paper proposes a new approach to do unsupervised-text-clustering. Particularly, they introduce the usage of multi-views combined with Gaussian Mixture Model to aggregate the views information.

### Strengths
The paper has clear writing as well as the reasoning process. I also find the claims made in the paper reasonable. While I have not rigorously validated the theoretical claims of the paper (in the appendix), I think they make sense intuitively on a high-level, thus, I believe they are correct.

### Weaknesses
First, I find the benchmark is lacking. The paper only evaluates their methods on 2 datasets. Furthermore, as mentioned in the related works, there are many other methods for text clustering and they only compare the proposed method to K-Mean, GMM, and Spectral clustering. Thus, I find the current benchmarking is not satisfiable for ICLR.

Second, while I believe the theoretical claims are correct, I question the assumptions that the authors use to make it works. For instance, the "mutual independent" (line 235) condition is too strong as I do not believe it ever holds in practice for a reasonable number of views m > 2. From my understanding, as all the views are generated on one entity/data point, it is much more likely for the views to be strongly dependent. On the other hand, I find the second condition "informative" is stated in a very unintuitive manner (overly formulated to fit the theoretical claim?). I also think it is difficult to validate how the assumption can hold in practice. Thus, I do not think it makes sense to say they are "mild conditions" as stated at line 224.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Briefly summarize the paper and its contributions. You can incorporate Markdown and Latex into your review. See https://openreview.net/faq.
The paper proposes a novel clustering scheme based on multi-view consensus clustering and dual-objective latent space optimization. Multi-view consensus is established by 1) first applying a view-specific randomly parameterized transforms, 2) conducting clustering for each view with Gaussian Mixture Model (GMM) clustering, and finally 3) merging the views via spectral clustering to yield the consensus cluster. Dual-objective latent space optimization refines the encoder latent space by minimizing both 1) an InfoNCE objective to sharpen the cluster assignment and 2) a regularization prior to map sample encodings onto the GMM manifold. The entire clustering process is iteratively conducted through alternation between the two processes. The authors further presents mathematical proof based on Hoeffding's inequality to demonstrate exponential error decrease induced by multi-view clustering.

### Strengths
In general, the presentation of the manuscript is good, and the proofs and methods are sound.

(Quality) The authors present theoretical backing to their scheme through discussions on the lowering of expected error caused by multi-view representation.

(Quality) Overall, the alternating optimization scheme proposed is interesting and methodologically sound.

(Quality) The authors demonstrate that their experiment outperforms single-view clustering.

(Clarity) The authors have presented the necessary equations and pseudocode to ensure a clear understanding for the audience.

### Weaknesses
In general, the work has issues with novelty and motivation, caused by the lack of a Related Work section and (more importantly) analysis of more recent literature (within the past 3 years).

(Quality) Experiments for clustering on unseen data are insufficient, as no single-view methods are presented for comparison.

(Originality) The work is somewhat limited in originality. Works on multi-view clustering with GMM [1] and InfoNCE [2] are already well-known. Thus, the proposed work incrementally builds upon existing work, with the main contribution being its application to LLM.

(Significance) The motivation of this work is detracted by the references selected. Aside from lacking a Related Works section, most of the references in the introduction are over 5 years old. An analysis of more recent multi-view clustering work (i.e. [3]) is necessary.

[1] Kumar, A., & Daumé, H. (2011). A co-training approach for multi-view spectral clustering. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 393-400).

[2] Oord, A. V. D., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

[3] Pattnaik, A., George, C., Tripathi, R., Vutla, S., & Vepa, J. (2024, November). Improving hierarchical text clustering with llm-guided multi-view cluster representation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track (pp. 719-727).

### Questions
1. Table 1 shows several sentence embedding models for generating multi-view representations. But their application is unclear. Does each model replace/merge its representation along with the Sentence-Bert output?

2. Given that multiple transformations are applied, would the cost (i.e. runtime, compute, memory) also increase significantly?

3. How many views m are used for the different multi-view schemes?

### Soundness
3

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
This paper proposes a new text clustering method based on multi-view clustering and information diversification. The authors introduce an iterative framework that alternately refines the clustering step and the text representation learning step by jointly minimizing a combined loss function consisting of a likelihood term and an InfoNCE term. Multiple feature extraction methods and models are employed to obtain diverse views of the texts, and a voting mechanism is used to measure how many views support that any two texts belong to the same cluster. A spectral clustering algorithm is then applied to this voting matrix (an NxN similarity matrix, where N is the number of texts) to produce the final clustering result. The authors also provide a theoretical analysis explaining why multi-view clustering outperforms single-view clustering, based on the minimax risk of the misclassification rate. Experimental results demonstrate that multi-view clustering with information diversification yields significant performance improvements over single-view embeddings.

### Strengths
1. The paper is well written and easy to follow.

2. The experimental results are positive and provide support for the authors’ claims.

3. The proposed algorithms are simple, reproducible, and conceptually sound.

### Weaknesses
1. **Lack of comparison with stronger baselines.** The proposed method is compared only against standard embeddings (e.g., SBERT) combined with conventional clustering algorithms such as KMeans and GMM, as well as its own variants. However, comparisons with existing multi-view clustering methods or ensemble-based approaches are missing.

2. **Theoretical contribution appears limited.** The authors show that multi-view consensus clustering achieves an arbitrarily low minimax risk as the number of views increases, whereas single-view clustering retains a constant lower bound. This result, however, is rather straightforward and well known in ensemble learning theory. Moreover, the assumption that the multiple views are sufficiently diversified to achieve a large effective number of independent views is quite strong, making the comparison with the single-view case somewhat unfair.

3. **Limited novelty.** The main contribution—combining multi-view clustering with contrastive learning for text clustering—appears incremental relative to prior work.

4. **Questionable scalability.** The proposed algorithm involves computing eigenvalues during the spectral clustering step on an N×N matrix, where N is the number of texts. While this matrix can be stored sparsely, the voting mechanism across multiple diversified views may considerably reduce its sparsity, making the spectral clustering step computationally expensive. More discussion on scalability is warranted.

### Questions
1. How should Figure 2 and Table 2 be interpreted? In Table 2, performance drops after adding Gaussian noise, whereas Figure 2 suggests the opposite trend. Please clarify this inconsistency.

2. In Table 2 (first row), what exactly are the “original embeddings”? Which SBERT model was used to produce them, or were they obtained by concatenating embeddings from multiple models?

3. Are there other possible ways to combine multiple views besides the proposed consensus approach (voting + spectral clustering)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a text clustering framework based on informative diversification, supported by a theoretical analysis of its multi-view consensus mechanism. While the reported gains over baselines such as K-Means merit recognition, the work's overall impact is limited by the restrictive nature of its theoretical assumptions and a lack of comprehensive experimental validation, which collectively undermine the validity of its claims.

### Strengths
1. Technical Framework: The paper presents a text clustering framework that integrates multi-view consensus clustering with deep representation learning in an end-to-end manner. An iterative co-training strategy is used to jointly optimize view generation, consensus clustering, and representation learning.

2. Theoretical Contributions: Theoretical analysis is provided, including an exponential upper bound on the misclustering error in multi-view consensus. The authors also connect the minimization of InfoNCE loss to the maximization of mutual information, giving a theoretical basis for the objective.

3. Experimental Results: Experiments on DBPedia and Reuters R8 report improvements in NMI and ARI over several baselines. The model shows some generalization capability, and t-SNE visualizations suggest that the learned embeddings form relatively compact clusters.

### Weaknesses
1. Limited Theoretical Novelty and Strong Assumptions: Proof 1 restates the known connection between InfoNCE and mutual information, contributing minimal theoretical innovation. Proof 2 relies on the strong—and often impractical—assumption of view independence, yet fails to discuss its validity or consequences in real-world scenarios.

2. Narrow Experimental Scope:  Evaluation is limited to two clean datasets and traditional baselines, lacking tests under noisy conditions or comparisons with recent deep learning-based clustering methods.

3. Insufficient Computational Analysis: The paper does not address the computational cost of multi-view generation or scalability, leaving practical feasibility unclear for larger datasets.

4. Incomplete Ablation and Hyperparameter Analysis: Ablation studies only explore view combinations without justifying core design decisions (e.g., spectral clustering vs. majority voting). Key hyperparameters—such as the number of views, loss weights, and iteration counts—lack systematic analysis, hindering reproducibility and insight.

5. Lack of Experimental Rigor and Presentation Issues: Critical implementation details and hyperparameter settings are inadequately documented. Additionally, Figure 2 uses inconsistent plot types for the same metric, impairing readability, and references are not alphabetized, reflecting a lack of attention to presentation quality.

### Questions
1.	The independence assumption for multiple views in Proof 2 is critical yet often impractical. How would violations of this assumption (i.e., correlated views) affect your theoretical guarantees, and did you empirically measure the dependence between the generated views?
2.	The experimental comparisons are limited to traditional methods. To firmly establish the advancement of your work, could you include results comparing against recent deep learning-based clustering approaches?
3.	Could you provide an analysis of the computational cost (e.g., training time scaling with dataset size and number of views) and include in the appendix the detailed settings for key hyperparameters (e.g., α, β, τ) to ensure reproducibility?

### Soundness
2

### Presentation
3

### Contribution
1
