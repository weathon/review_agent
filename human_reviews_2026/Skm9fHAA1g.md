# Gene-M1: Advancing Cross-Species Genomic Discovery via Taxon-Specific Mixture-of-Experts

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Prevailing genomic foundation models rely on a uniform architecture across all species, which overlooks evolutionary divergence and leads to feature interference and limited cross-species generalization. To address this, we introduce GENE-M1, a novel Mixture-of-Experts (MoE) framework strictly governed by biological taxonomy. Our method builds on three core components: (1) a hierarchical expert architecture that instantiates specialized modules for taxonomic ranks (Domain, Kingdom, Phylum, Class) to enable taxon-specific processing; (2) a dynamic router that activates expert pathways aligned with a sequence’s taxonomy, ensuring hierarchical feature extraction; and (3) a progressive training strategy that transfers knowledge from higher to lower taxonomic ranks for stable optimization. In addition, we construct GM-DATA, a large-scale, taxonomically aligned benchmark comprising 294 species spanning 5 Kingdoms, 18 Phyla, and 62 Classes, with broad and balanced coverage across major clades, as well as a held-out GM-DATA(eval) set of 15 unseen species for rigorous cross-species evaluation. Extensive experiments on this benchmark show that GENE-M1 significantly outperforms state-of-the-art baselines in few-shot classification and unsupervised clustering, demonstrating that explicit taxonomic alignment is key to robust and interpretable genomic representation learning. We will release our model, code, and dataset soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces two main contributions: (1) GENE-M1, a Mixture-of-Experts (MoE) model that mitigates feature interference when jointly learning gene representations across diverse species; and (2) GM-DATA, a dataset covering 294 species with a more balanced class distribution. The proposed approach achieves good performance on several tasks such as  few-shot classification and unsupervised clustering.

### Strengths
1. The idea of utilizing the MoE structure to decouple the learning across species and reduce feature interference is reasonable.
2. The hierarchical model design aligned with the taxonomic ranks Domain, Kingdom, Phylum, and Class is clear and interpretable.
3. The proposed GM-DATA offers a more balanced data distribution and is likely to serve as a viable alternative for future model evaluation.

### Weaknesses
1. The proposed GENE-MI heavily relies on an accurate classification tree, e.g., the training of L_router is supervised by explicit classification labels, and for unknown or contentious species, the model's capability may be limited.
2. Although the evaluation set in GM-DATA contains 15 "unknown" species, the classes of most of those species are present in the training set, e.g., "Danio rerio" belongs to "Actinopterygii". Therefore, the experiments largely test "generalization ability to new species within known classifications," rather than "generalization ability to entirely new classifications (such as an unseen 'Class' or 'Phylum')."
3. The progressive training method may have certain limitations, although this hierarchical training with frozen parameters can reduce model forgetting, it also limits the model's ability to fine-tune more general-level features while learning finer-grained features, which is a relatively rigid, one-way optimization approach.
4. While the paper compares performance with models such as DNABert and HyenaDNA in Table 2 and Table 3, it does not list the differences in model size and computational efficiency; I believe we need to consider accuracy, model size, and computational efficiency comprehensively before making an overall assessment of the model's strengths and weaknesses, which would be more reasonable.
5. There are some typos in the paper, for example "evolutionary distant species" at L114, "Class Labal" in Fig. 2, "After flitering" at L284, "specie" at L463, etc.

### Questions
Please kindly refer to the Weaknesses section, and I would consider to raise my rating if the authors could answer most of my questions.

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
4

### Summary
The paper presents the GENE-M1, an MOE model whose architecture is governed by taxonomy.
GENE-M1 explicitly mirrors the biological hierarchy by using: (i) Hierarchical experts specialized per taxonomic level; (ii) A dynamic router; (iii) A coarse-to-fine training strategy.
The work also provides GM-DATA for training and evaluation,
Extensive experiments on this benchmark show that GENE-M1 significantly outperforms state-of-the-art baselines.

### Strengths
(i) Architecture: The paper provides an MOE architecture for the genomic models. There has been almost no MOE architecture in the genomic model domain.

(ii) Dataset and benchmark: The paper provides both a training dataset and a benchmark.

(iii) Strong empirical gains: The experimental results are very positive.

### Weaknesses
(i) Presentation: The paper presentation is a little poor. It can be hard to understand the details, and there are some typos. E.g., there is a "?" in line 114.

(ii) No comparison between the model size/training cost of GENE-M1 and DNABERT-2.

(iii) Not enough evidence to show this is a general method: Only show the model based on DNABERT-2 and NT.

### Questions
(i) What is the importance of this scientific classification/clustering problem? Is it the same as DNABERT-S?

(ii) What is the computational cost and model size of GENE-M1, compared with DNABERT-2 and DNABERT-S? Are the positive experimental results under fair comparison?

(iii)  Can the method be generalized to other genomic models?

### Soundness
2

### Presentation
2

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
The paper presents GENE-M1, a taxonomy-aware Mixture-of-Experts (MoE) framework for genomic foundation models, designed to improve cross-species generalization by explicitly aligning model architecture with biological taxonomy (Domain > Kingdom > Phylum > Class). The authors also release GM-DATA, a new benchmark with 294 species across 5 kingdoms, 18 phyla, and 62 classes, including 15 unseen species for evaluation.
Experiments show that GENE-M1 significantly outperforms DNABERT-2, Nucleotide Transformer, and other baselines on few-shot gene classification and unsupervised clustering tasks, particularly at fine taxonomic levels (e.g., Class), while producing more disentangled and interpretable embeddings.

### Strengths
•	Novel biological alignment: The paper introduces a principled and interpretable architectural mapping between taxonomy and neural network modularity, which is rare in genomic foundation models.
•	Comprehensive dataset construction: GM-DATA is carefully curated with hierarchical structure and balanced taxonomic representation, a clear advancement over previous bacteria-dominated datasets.
•	Strong empirical results: Consistent and substantial improvements (10–30 % macro-F1 gains) on cross-species few-shot classification and clustering metrics demonstrate real advantages over large baselines.
•	Methodological clarity: Each module (expert composition, routing, training) is mathematically and conceptually well-defined.

### Weaknesses
•	GENE-M1’s design presumes that every training sequence has a known and correctly assigned taxonomy down to fine levels. That’s realistic for curated species genomes but not for metagenomic or environmental samples, where taxonomy is noisy or incomplete. 
•	All baselines use masked-language pretraining (DNABERT, NT). There’s no comparison to alternative pretraining paradigms such as contrastive learning (DNASimCLR), masked-k-mer prediction, or retrieval-based transformers, which might already encode cross-species structure without hierarchical supervision. It is hard to tell whether the gains come from architecture or from additional supervision signal.

•	Computational efficiency not quantified: Although the paper claims parameter efficiency via sparse expert activation, training cost and scaling behavior (vs dense models) are not empirically reported.
•	In MoE models, a known challenge is expert collapse where a few experts dominate routing, reducing effective capacity. The authors show qualitative router heatmaps, but don’t quantify load balancing or entropy of expert usage.
•	While the authors present unsupervised clustering as a secondary “task,” it is effectively a post-hoc analysis of embeddings obtained from the supervised taxonomic classification model. The resulting alignment of clusters with taxonomy is thus expected to some extent, since the embeddings were optimized to separate taxonomic labels. The analysis would have been more interesting if instead of k-means clustering, hierarchical clustering techniques were used, since they seem to align more naturally with the problem. The resulting dendrogram could be evaluated against the ground truth hierarchy.

Minor points/typos:
•	Typo in last sentence of abstract: ‘modal’  ‘model’
•	When mentioning things like “…supporting a variety of downstream biological applications” it would be useful to include some examples of the applications.
•	In section 4.2 under Backbone & Tasks, the authors mention ‘three’ tasks, but only two are described.
•	In the results for the clustering tasks, the full form of the metrics should be mentioned at least once in the main paper, even if they are defined in detail in the appendix.

### Questions
The paper states that GENE-M1 is trained with hierarchical supervision through its taxonomy-aligned Mixture-of-Experts structure, but the embeddings used for the clustering analysis are extracted from this same model. Could the authors clarify whether the model receives supervision at all taxonomic levels during training (e.g., Domain, Kingdom, Phylum, Class), or only at the level used for classification evaluation? Since the model has explicit access to hierarchical labels through its experts and routing mechanism, to what extent can the clustering results be considered an unsupervised validation of emergent structure rather than a byproduct of supervised hierarchical training?

### Soundness
2

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
4

### Summary
This paper addresses the one-size-fits-all limitation of current genomic foundation models, which perform poorly across diverse species. The authors introduce GENE-M1, a Mixture-of-Experts (MoE) framework explicitly aligned with biological taxonomic hierarchies (Domain, Kingdom, Phylum, Class). GENE-M1 uses hierarchical experts, dynamic routing, and progressive training to learn specialized representations that improve cross-species generalization. To support this, the authors also built GM-DATA, a new large-scale, taxonomically structured genomic dataset. Extensive experiments demonstrate that GENE-M1 significantly outperforms state-of-the-art baselines on cross-species tasks, offering superior performance and biological interpretability.

### Strengths
1) This paper is clear written and easy to understand. The motivation is clear.

2) Better performance than the comparison methods.

3) It makes sence to introduce the MoE architecture with the ground-truth biological taxonomic hierarchy.

4) A new dataset is released.

### Weaknesses
1) The paper doesn't explicitly state whether the baseline models (NT, DNABERT-2) were re-trained from scratch or fine-tuned on the new GM-DATA(train) dataset. If the authors used off-the-shelf baselines, the comparison may be unfair, as GENE-M1's superior performance could be partially attributed to being trained on this new, balanced dataset, rather than purely to the architectural advantages.

2) Figure 2 of the paper illustrates the paper's vision: Domain → Kingdom → Phylum → Class. However, the experiments only involve Kingdom → Phylum → Class. Although the authors mentioned the issue of finer-grained levels in the limitations, from the perspective of the paper, Figure 2 depicts future work. This is because the problem of the significant parameter increase caused by more finer-grained levels has not been solved in this paper.

3) While this design based on a hierarchical mechanism is a core contribution of the paper, it has already been widely used in many relevant classification tasks (e.g., image classification with ImageNet's hierarchical labels), which limits the innovation of the proposed methodology.

### Questions
1) Were the baseline models (NT, DNABERT-2, etc.) re-trained on GM-DATA(train) before evaluation? It is important to tell the performance gains come from the GENE-M1 architecture versus the benefit of training on the new, taxonomically-balanced GM-DATA dataset.

2）Due to the introduction of the MoE structural design, the network also introduces a large number of learnable parameters. Therefore, the authors need to compare the increase in parameters, or the computational cost of the network, to ensure that the performance improvement is brought by the structural advantages of the proposed MoE, not just by the increase in number of parameters.

### Soundness
3

### Presentation
3

### Contribution
2
