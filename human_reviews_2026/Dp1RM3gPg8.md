# Fast Proteome-Scale Protein Interaction Retrieval via Residue-Level Factorization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Protein-protein interactions (PPIs) are mediated at the residue level. Most sequence-based PPI models consider residue-residue interactions across two proteins, which can yield accurate interaction scores but are too slow to scale. At proteome scale, identifying candidate PPIs requires evaluating nearly *all possible protein pairs*. For $N$ proteins of average length $L$, exhaustive all-against-all search requires $\mathcal{O}(N^2L^2)$ computation, rendering conventional approaches computationally impractical. We introduce RaftPPI, a scalable framework that approximates residue-level PPI modeling while enabling efficient large-scale retrieval. RaftPPI represents residue interactions with a Gaussian kernel, approximated efficiently via structured random Fourier features, and applies a low-rank factorized attention mechanism that admits pooling into a compact embedding per protein. Each protein is encoded once into an indexable embedding, allowing approximate nearest-neighbor search to replace exhaustive pairwise scoring, reducing proteome-wide retrieval from *months* to *minutes* on a single GPU or CPU. On the human proteome with the D-SCRIPT dataset, RaftPPI retrieves the top 20% pairs from $\sim$200M candidate pairs in 5.7 minutes on an A100 GPU, or 3.3 minutes on an Intel Xeon 6980P CPU, covering 75.1% of the true interacting pairs, compared to 4.9 GPU months for the best prior method (61.2%). Across seven benchmarks with sequence- and degree-controlled splits, RaftPPI achieves state-of-the-art PPI classification and retrieval performance, while enabling residue-aware, retrieval-friendly screening at proteome scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes RaftPPI, which models PPI by approximating residue-level interactions while enabling scalable protein retrieval. RaftPPI computes residue–residue scores using a Gaussian kernel and aggregates them into a protein-level score. The non-linear kernel is efficiently approximated via random Fourier features, and pooling is implemented with a low-rank factorized attention mechanism that allows linear-time approximation during inference. Each protein is encoded once into an indexable embedding suitable for ANN search, enabling retrieval of likely interactors while preserving residue-level interactions and avoiding explicit per-pair computation.

### Strengths
This work transforms nonlinear residue-level interaction modeling into a factorizable embedding form, substantially enhancing the scalability of proteome-level retrieval. It demonstrates outstanding computational efficiency, achieving an acceleration from GPU-months to minutes.

### Weaknesses
1.	The results in Table 2 show no significant improvement, with RaftPPI surpassing PLM-Interact by only 0.15 on the Average metric.
2.	RaftPPI, short for Residue-interaction Approximation with Fourier FeaTures, only validates the effectiveness of the residue-interaction approximation in the ablation studies, lacking evidence for the contribution of the Fourier features.
3.	The paper lacks a discussion on limitations and future research directions, which would help readers gain a more comprehensive understanding of the work.

### Questions
Please refer to the above "weakness" part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SummaryThis paper proposes RaftPPI, a method to speed up proteome-scale protein-protein interaction (PPI) screening. The main problem it addresses is that existing sequence-based models are too slow for an all-against-all search, often requiring $\mathcal{O}(N^{2}L^{2})$ computation. RaftPPI approximates a residue-level interaction model with a factorizable one. It models residue interactions with a Gaussian kernel, approximates this kernel using random Fourier features (SORF), and uses a low-rank (rank-1) attention mechanism to pool these features. This allows each protein to be encoded into a single embedding. The interaction score is then just a dot product of these embeddings, which enables fast retrieval using Approximate Nearest Neighbor (ANN) search. The authors also use an adaptive negative weighting loss to focus on harder negative examples during training. The method is shown to be much faster than prior work, reducing screening time from months to minutes on the human proteome, while achieving strong performance on several classification and retrieval benchmarks.

### Strengths
- The primary strength is the massive, practical speedup. Reducing a computation from 4.9 GPU-months to 6 GPU-minutes is a game-changer for the field and makes routine proteome-scale screening a reality.
- The experimental setup is rigorous. By using 7 datasets that are explicitly controlled for sequence similarity and degree bias, the authors provide a robust and trustworthy evaluation of their model's performance.
- The paper is very clear, especially Figure 1, which provides an excellent intuitive explanation of how the factorization is achieved and why it leads to a speedup.
- The ablation study confirms that the different components of the model (the kernel, the attention, and the loss function) all positively contribute to the final performance.

### Weaknesses
1. The main weakness is the limited novelty of the individual components. The work is a clever integration of existing techniques: PLM embeddings, kernel approximation with Random Fourier Features (specifically SORF), and low-rank (rank-1) attention. This makes the contribution feel more incremental and engineering-focused rather than a fundamental breakthrough.
2. The use of a rank-1 approximation for the attention mechanism ($r=1)$ is a very strong simplification. It's surprising this works so well and it's not well-justified beyond just stating it "achieves strong performance". This could be a significant limitation, as a rank-1 matrix is unlikely to capture complex residue-residue interaction patterns.
3. The justification for using the ESM2-8M model is weak. The appendix (B.1) bases this choice on an unsupervised scaling analysis of [CLS] token dot products. This is not a good proxy for the full, finetuned RaftPPI model, which uses all residue embeddings in a complex, kernelized way. It's very possible the model's performance is being left on the table by not using a larger, more powerful PLM.
4. There is no discussion of how the Gaussian kernel bandwidth $\hat{\sigma}$ was selected. This is a critical hyperparameter for kernel methods, and its tuning (or lack thereof) could significantly impact performance.

### Questions
1. The use of a rank-1 ($r=1$) approximation for the attention mechanismseems like a major simplification. Did you experiment with higher ranks (e.g., $r=4, 8, 16$)? What is the performance-vs-computation trade-off here?
2. How was the Gaussian kernel bandwidth $\hat{\sigma}^{2}$ selected? Can you provide a sensitivity analysis for this hyperparameter?
3. The justification for using the small ESM2-8M model rests on an unsupervised analysis of [CLS] token dot products. Why should this analysis apply to the full RaftPPI model, which is finetuned and uses all residue embeddings in a kernelized way? Have you tried finetuning RaftPPI with a larger PLM (e.g., ESM2-35M) to see if performance improves?
4. The adaptive negative weighting loss is one of several ways to handle hard negatives. How does it compare, both in performance and simplicity, to a more standard, well-tuned focal loss?

### Soundness
3

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
This paper introduces RaftPPI, a method that is a computationally efficient framework which is designed to quickly and accurately predict protein-protein interactions (PPIs) on massive protein datasets. The central problem it solves is that existing high-accuracy models are too slow, requiring months of computation to screen an entire dataset. RaftPPI approximates complex, residue-level (amino acid) interactions using an efficient method that encodes each protein into a single, compact, and indexable embedding. This "factorizable" approach, which uses a Gaussian kernel approximated by random Fourier features, allows the model to replace exhaustive pairwise comparisons with a fast nearest-neighbor search. As a result, RaftPPI can bring multiple folds of speedup, while achieving superior accuracy in both PPI classification and retrieval.

### Strengths
Here are some strengths of the paper:

1. $\textbf{Computational Efficiency}$

The paper's main strength is that it directly tackles the most significant bottleneck in protein-protein interaction, which is computational cost. Traditional high-accuracy models are too slow, taking "GPU-months" as described in the paper. This paper proposes a method that reduces this significantly by multiple folds. This makes large-scale interaction screening very practical.

2. $\textbf{Proposed Method}$

The proposed method solves the efficiency problem in a clever way without compromising the performance of protein-protein interaction performance. Instead of comparing every residue on one protein to every residue on another protein for all possible pairs ($O(N^2L^2)$), it creates a single, compact embedding for each protein which allows the problem to be re-framed as an efficient nearest-neighbor search with $O(NL^2)$ complexity.

3. $\textbf{Superior Results }$

The authors show that in average their method performs better than baselines in different datasets.

### Weaknesses
Here are some weakness/questions:

1. $\textbf{Additional datasets}$

How will the proposed method perform on datasets like the one shown in [1] which is a dataset proposed for protein-protein interaction for ML pipeline?

2. $\textbf{Comparison with bigger models}$

Is it possible that your model misses subtle, complex, or non-linear interactions that a full, slower model (like AlphaFold) might capture? Is there a way to compare this quantitatively?

3. $\textbf{Limitations of the work}$

I don't see any place where the limitation of the proposed work being mentioned. What are some discussions/limitations of this approach? It would be beneficial to include those in the paper or appendix as well.

4. $\textbf{Interpretability}$ 

How interpretable is your approach? Can it be used with works that require interpretability?


References

1. PiNUI: A Dataset of Protein-Protein Interactions for Machine Learning; Geoffroy Dubourg-Felonneau and Eyal Akiva and Daniel Wesego and Ranjani Varadan; NeurIPS 2023 Workshop on New Frontiers of AI for Drug Discovery and Development; 2023

### Questions
Please look at the limitation section

### Soundness
2

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
The paper introduces RaftPPI, a sequence‑only framework that preserves residue‑level fidelity yet enables proteome‑scale retrieval. The method (i) models residue–residue scores by a Gaussian kernel on PLM residue embeddings and (ii) uses low‑rank (separable) attention so that the aggregation over residue pairs becomes a dot product between per‑protein embeddings after a Structured Orthogonal Random Features (SORF) transform; this permits HNSW‑based ANN search using one‑time per‑protein encodings. The training objective applies adaptive negative weighting to emphasize hard negatives. On seven-degree and homology‑controlled datasets, RaftPPI attains the best mean AUROC and best mean Recall@20%, while reducing whole‑human‑proteome retrieval from ~148.5 A100 GPU‑days for PLM‑Interact to minutes using a single GPU.

### Strengths
- **(S1) Effective methods.** Modeling residue interactions via a Gaussian kernel and approximating it with RFF/SORF yields an inner‑product surrogate for the attention‑weighted residue sum. The low‑rank attention gives interpretable residue weights while keeping the index dimension small.

- **(S3) Strong experimental results.** Firstly, the comparison experiment results are impressive. On seven datasets curated to mitigate sequence/degree leakage, RaftPPI achieves the SOTA results, while compressing human‑proteome retrieval from ~148.47 A100 GPU‑days for PLM‑Interact to 241s for Recall@20% after a one‑time encoding. Meanwhile, the paper evaluates against classical and PLM baselines with a shared ESM2‑8M backbone where applicable, runs five seeds, and includes ablations showing that kernelization (SORF), attention pooling, and adaptive negative weighting are each necessary. The Appendix scaling study (Figs. 4–5) justifies the 8M backbone for throughput without sacrificing accuracy.

- **(S3) Clear motivation and professional presentation.** The manuscript is well organized and written in clear language with comprehensive figures and tables. The methodology section is well structured. The technical descriptions are coherent and self-contained.

### Weaknesses
- **(W1) Approximation fidelity lacks characterization.** The end‑to‑end error induced by rank‑1 separable attention and finite‑feature RFF is not quantified; no bounds or diagnostics relate the surrogate $<\hat{h}_A, \hat{h}_B>$ to the original attention‑weighted kernel sum. This leaves open whether interfaces requiring multiple spatial patches(multi‑epitope interactions) demand r>1 or larger feature budgets.

- **(W2) Retrieval evaluation underrepresents full proteome use‑cases.** The protocol uses 100 query proteins per dataset rather than all queries in a proteome or cross‑species searches (Fig. 2), so robustness of HNSW hyper‑parameters, index memory/latency vs. N, and failure modes at scale remain unclear. End‑to‑end curves vs. database size or per‑species full‑coverage runs would better support the proteome‑scale claim.

- **(W3) Metrics for extreme imbalance are incomplete.** The paper reports AUROC and Recall@K%, which are informative but insufficient in heavily imbalanced retrieval; AUPRC/average precision, MAP, and NDCG would make performance at the head of the ranking clearer and facilitate comparisons with practical screening budgets.

- **(W4) Limited sensitivity analysis of key design knobs.** The model fixes r=1, d′=2048, and a single bandwidth, and it stops gradients through the negative weights. However, there is no study of how rank r, feature dimension d′, or bandwidth selection trades accuracy, memory, and latency, nor a comparison to allowing gradients through $p_i$. These are central for practitioners who must size the index and allocate compute.

### Questions
- **(Q1)** Please detail the negative sampling protocol(s) used in training across datasets (e.g., random, compartment‑aware, topology‑aware) and the ratio of negatives to positives per batch, so others can reproduce the adaptive weighting behavior.

- **(Q2)** Did the authors compare inner‑product vs. cosine retrieval (with/without L2 normalization) in HNSW, and if so, how did this choice affect recall and calibration of scores? A short note or table would guide practitioners.

- **(Q3)** Did the authors try index‑time compression (e.g., product quantization) to reduce memory for whole‑proteome indices, and how does it trade a small loss in recall for a large RAM reduction? It would be valuable for deployment at a larger N.

### Soundness
2

### Presentation
3

### Contribution
3
