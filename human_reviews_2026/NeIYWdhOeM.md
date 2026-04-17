# Bridged Clustering for Representation Learning: Semi-Supervised Sparse Bridging

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We introduce Bridged Clustering, a semi-supervised framework to learn predictors from any unpaired input $\mathcal{X}$ and output $\mathcal{Y}$ dataset. Our method first clusters $\mathcal{X}$ and $\mathcal{Y}$ independently, then learns a sparse, interpretable bridge between clusters using only a few paired examples. At inference, a new input $x$ is assigned to its nearest input cluster, and the centroid of the linked output cluster is returned as the prediction $\hat{y}$. Unlike traditional SSL, Bridged Clustering  explicitly leverages output-only data, and unlike dense transport-based methods, it maintains a sparse and interpretable alignment. Through theoretical analysis, we show that with bounded mis-clustering and mis-bridging rates, our algorithm becomes an effective and efficient predictor. Empirically, our method is competitive with SOTA methods while remaining simple, model-agnostic, and highly label-efficient in low-supervision settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces Bridged Clustering, a semi-supervised learning framework designed to learn predictors from unpaired input and output datasets. The method first clusters the input space and output space independently, then learns a sparse, interpretable bridge between clusters using a small set of paired examples. At inference, a new input is assigned to its nearest input cluster, and the centroid of the linked output cluster is returned as the prediction. The approach is model-agnostic, computationally efficient, and supports bidirectional inference. Theoretical analysis shows that the predictor's risk is bounded under certain mis-clustering and mis-bridging rates. Empirical results on multimodal datasets (BIOSCAN, WIT, Flickr30k, COCO) demonstrate competitive performance against a wide range of baselines, especially in low-supervision regimes.

### Strengths
1. The proposed BC is model-agnostic and easy to implement, relying on standard clustering and majority-vote bridging. Its sparse, interpretable mapping between input and output clusters offers transparency absent in dense transport or deep generative models.
2. Evaluation across multiple modalities (vision, language, genomics) under both inductive and transductive settings, along with comparisons to a wide range of baselines, demonstrates robustness and generality.

### Weaknesses
1. The method assumes that input and output spaces can be cleanly partitioned into clusters that align one-to-one. This may not hold in real-world data with overlapping or hierarchical categories. When clusters poorly capture latent structure (e.g., in the WIT dataset), predictive performance degrades notably. This limits robustness in high-dimensional or weakly separable spaces.
2. Although the method’s interpretability is emphasized, no concrete visualization or case study (e.g., cluster bridges on BIOSCAN or COCO) is provided to illustrate interpretability in practice.
3. Predictions are always the centroid of the output cluster, which may be too coarse for tasks requiring fine-grained outputs. The model does not refine predictions with additional supervision beyond the bridge.

### Questions
1. Since the method heavily relies on representation separability (∆X, ∆Y), how sensitive is performance to pretrained encoder choice? Have the authors experimented with self-supervised embeddings?
2. Empirically, the paper reports success with as few as one labeled pair per cluster. Is there a theoretical lower bound on the number of pairs required to guarantee reliable bridging under bounded εX, εY?

### Soundness
3

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
3

### Summary
The authors propose BRIDGED CLUSTERING, an algorithm leveraging both input x, output y data and a small paired (x,y) dataset. The algorithm is very simple, both input and output sets are independently clustered, and an input to output cluster mapping is found through majority voting in the paired dataset. The authors provide a mis-clustering analysis with data generated under sub-gaussian mixtures. The effectiveness is validated empirically on 4 datasets.

### Strengths
- The paper is well written and easy to follow. The notation is also coherent and intuitive.
- The algorithm is extremely simple yet well motivated, defined and atomized. The algorithm seems to work well empirically.

### Weaknesses
### Major
- My main issue is that all this method reduces all intra cluster variation to 0, which is even more pronounced if the output-only data has high variations. For instance in the image captioning task (L337), any intra-cluster variation in images is lost and all images get the same caption. This strongly limits the applicability of such algorithm. Based on the problem formulation and analysis, the algorithm is only applicable to data derived from categorical latents.

### Minor
- Limited empirical validation.
- L083: “leverages output-only data” this is confusing because it also leverages input data.

### Questions
- About the terminology: what does “output-only data mean?” In my understanding, this can only be defined is you define a learner that maps inputs to outputs. I think this should be better explained in the introduction. Because given an unnannotated dataset, there is no notion of input/output. L089 reinforces my claim because you state that you can do bidirectional inference.
- All empirical evaluations seem to be done in a very controlled setup. Why did you limit yourselves to 7 groups? Do you think this would work on more complex datasets?
- L063: “once” should be “if” ?

Overall, I think the paper reads well and my rating would be around borderline. Given that I cannot give a rating of 5, I put 4 as a start but am willing to increase my rating based on the reply and the other reviews.

### Soundness
3

### Presentation
3

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
The paper proposes Bridged Clustering (BC), a semi-supervised framework for learning predictors when one has unpaired inputs $X$ and unpaired outputs $Y$ plus a small paired set $S$.

The method: (i) clusters $X$ and $Y$ independently; (ii) learns a sparse cluster-to-cluster bridge from the few paired examples via majority vote; (iii) predicts by assigning a test input to its nearest input cluster and returning the centroid of the linked output cluster (and symmetrically $Y \rightarrow X$).

Experiments on BIOSCAN (bioinformatics) and vision-language datasets (COCO, Flickr30k, WIT) show BC is competitive with SSL, unmatched regression, and transport baselines; it tends to win on BIOSCAN/COCO/Flickr30k and is competitive (but not best) on WIT.

### Strengths
1. Focuses on the under-served regime with large unpaired $X$ and $Y$ plus a tiny paired set, using independent clustering and a sparse bridge-distinct from classical SSL and from purely distributional coupling.

2. Model-agnostic encoders and off-the-shelf clustering.

3. Four datasets across modalities; many seeds and settings; BC generally wins on BIOSCAN/COCO/Flickr30k and is competitive on WIT.

### Weaknesses
1. Performance hinges on embedding quality and (near-)correct $C$. There is no systematic study of mis-specifying $C$ or robustness to imbalanced/overlapping clusters; theory assumes separation.

2. Majority-vote induces a deterministic mapping $A:[C]\to[C]$. In multi-modal or hierarchical relations, a soft/multi-edge bridge may reduce $\varepsilon_B$.

3. The main metric is embedding-space MSE; this may not fully capture downstream utility.

### Questions
1. For captioning/retrieval, could you report Recall@K / median rank and human-readable examples to complement embedding MSE?

2. Section 5.3 claims linear-time scaling once $C \ll n$. Please include wall-clock vs. $n,d$, and $C$ across datasets.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The article presents Bridged Clustering (BC), a semi-supervised framework for learning mappings between unpaired input–output datasets. The method first clusters each domain independently and then learns a sparse, interpretable “bridge” between clusters using a small set of paired samples. During inference, a new input is assigned to its input cluster and mapped to the centroid of the linked output cluster. The authors provide theoretical analysis (risk bounds under mis-clustering and mis-bridging rates) and evaluate BC across four multimodal datasets (BIOSCAN-5M, WIT, Flickr30k, COCO). The method outperforms or matches strong baselines in most cases while being model-agnostic and computationally efficient.

### Strengths
- The paper is clear about its motivation: learning from both input-only and output-only data is an underexplored problem.
- Simple and interpretable design that remains competitive with complex baselines.
- Strong empirical performance on diverse modalities with extremely low supervision.
- Training and inference scale linearly in data size, unlike OT/GW baselines.

### Weaknesses
- Method performance is heavily dependent on clustering quality; the paper could discuss robustness or adaptive clustering strategies.

- Theoretical analysis, while rigorous, lacks intuitive explanation or ablation support.

- Limited discussion of failure modes and sensitivity to hyperparameters like the number of clusters.

- Sparse bridge formulation may be too restrictive for many-to-many mappings.

- Baseline tuning is relatively light, might weaken empirical fairness while comparing.
- No sufficient discussion on fixed cluster selection. 
- Missing comparison with recent simCLR based methods like SCAN, NNM or so.

### Questions
- How sensitive is BC to the choice of cluster count?

- Majority voting seems too simple, could soft or probabilistic bridges (e.g., weighted votes) improve results in overlapping clusters?
- Is it possible to provide runtime comparison in actual time not in complexity?

- How does clustering method choice (k-means vs spectral, DBSCAN, etc.) affect performance?

- Have you tested robustness under noisy embeddings or partially misaligned clusters?

- Could the approach generalize to continuous or hierarchical output spaces?
- What about imbalance dataset?

### Soundness
2

### Presentation
2

### Contribution
2
