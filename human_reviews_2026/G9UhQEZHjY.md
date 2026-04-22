# NC-Bench and NCfold: A Benchmark and Closed-Loop Framework for RNA Non-Canonical Base-Pair Prediction

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 4

## Abstract
RNA secondary structure forms the basis for folding and function, with non-canonical (NC) interactions indispensable for catalysis, regulation, and molecular recognition.
Despite their importance, predicting NC base pairs remains challenging due to the absence of a standardized benchmark for systematic evaluation.
To address this, we introduce NC-Bench, the first benchmark dedicated to NC base-pair prediction. NC-Bench provides 925 curated RNA sequences with 6,708 high-quality NC annotations, fine-grained edge and orientation classification tasks, and IsoScore-based embedding evaluation, offering a rigorous foundation for systematic assessment.
Building on this, we propose NCfold, a dual-branch framework that couples sequence features with structural priors derived from RNA foundation models (RFMs) via Representative Embedding Fusion (REF) and REF-weighted self-attention.
The closed-loop design iteratively refines sequence and structure representations, alleviating data sparsity and enhancing predictive accuracy. 
Experiments on NC-Bench show that NCfold outperforms existing methods, with zero-shot and ablation studies confirming its effectiveness and underscoring the need for NC-specific benchmarks.
Together, NC-Bench and NCfold establish a systematic foundation for NC base-pair prediction, advancing our understanding of RNA structure and enabling next-generation RNA-centric applications. The datasets and codes are publicly available at https://github.com/heqin-zhu/NCBench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NC-Bench, a small (N=925) dataset for non-canonical (NC) RNA base pair prediction. It also proposes NCfold, a complex dual-branch transformer. NCfold uses an IsoScore heuristic to select top-k RFM embeddings, which are then fused as structural priors into the attention mechanism.

### Strengths
1. The NC-Bench dataset is a useful, though modest, contribution to the field.

2. The problem of NC pair prediction is important and challenging.

3. The zero-shot analysis clearly shows that existing canonical structure predictors fail at this specific task.

4. The paper is well written and clearly structured. The problem, the proposed solutions (dataset and model), and the results are all explained logically and are easy to follow.

### Weaknesses
1. The proposed architecture contains multiple novel components (IsoScore ranking, REF, dual-branch fusion), but their theoretical justification and individual impact could be analyzed more rigorously.

2. Some baselines are missing, particularly against MSA-based RNA language models such as RNA-MSM, which represent strong baselines for RNA structure understanding.

3. The use of 4-fold cross-validation instead of the standard 5-fold setup is not explained, and the sensitivity of performance to this choice is unclear.

### Questions
1. What motivates the use of 4-fold instead of 5-fold cross-validation, and how sensitive are the results to this choice?

2. How does NCfold perform relative to MSA-based RNA LMs (e.g., RNA-MSM or other alignment-informed models)?

3. How robust is the performance of NCfold to different IsoScore thresholds or numbers of selected embeddings (top-k)?

4. Could the benchmark and model be extended to capture tertiary contacts or multi-body interactions in longer RNAs?

### Soundness
3

### Presentation
3

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
The paper presents NC-Bench, a new benchmark dataset of 925 curated RNA sequences derived from the PDB. The benchmark is specifically designed for non-canonical RNA prediction tasks. Based on this benchmark, the authors propose NCFold, a framework for non-canonical base-pair prediction. NCFold utilizes embeddings from different existing RNA foundation models that are ranked by IsoScore with subsequent fusion and a specialized attention approach. The authors evaluate NCFold on NC-Bench with comparisons against different RNA foundation model baselines and general secondary structure prediction algorithms.

### Strengths
- The paper tackles an important problem in RNA structure prediction, namely non-coding base-pair prediction
- To my knowledge, a dedicated benchmark for NC-pair prediction is missing
- Including the interaction edge of nucleotides as a classification task via LW-nomenclature in the prediction tasks is a good and to my knowledge novel task.
- The proposed REF approach and REF-weighted self-attention appear interesting.
- NCFold seems to outperform different existing baselines on NC-Bench.

### Weaknesses
**Major**:
1. In the Introduction, the authors support their new benchmark with the claim that previous methods are “primarily designed for canonical base-pairs” predictions. I do not agree with this statement. Already early DL methods like [1] clearly state in their Abstract:

>[...] Here, we propose the use of deep contextual learning for base-pair prediction including those noncanonical and non-nested (pseudoknot) base pairs stabilized by tertiary interactions. [...]

Following methods like [2-4] continue leveraging the advantage of DL methods to output an L x L matrix which allows them to include nc-pairs but also pseudoknots and even multiplets. So these methods can predict all kinds of nucleotide interactions in general. The performance on nc-interactions, however, remains far behind the performance for canonical pairs. To the best of my knowledge, even the used baseline provided in [5] outputs ct files, which allows them to predict all kinds of nucleotide interactions. The prediction quality then mainly depends on the chosen weights (see questions).

However, I generally agree that the LW-classification is missing in all these approaches.

2. I’m particularly concerned about the data set construction. The authors use mmseqs to reduce redundancy between datapoints. However, the splitting of the data into train, validation, and test is not further defined. There was a lot of work recently that showed that careful data splitting is an essential part during evaluation of learning-based methods for RNA secondary structure prediction [6-9]. In this regard, I cannot fully trust the reported results. At least two recent RNA benchmarks tackle the problem of data curation for different tasks [10, 11]. In particular, RNA3DB would be a perfect fit here, since it provides a strong split based on RNA families, is based on PDB samples, and the authors could run their exact same data processing pipeline to extract pairs.
3. Following up on this: Why do the authors exclude all WC interactions? This strongly reduces the practical usefulness of NC-Bench and makes the evaluation look a bit like it is constructed for NCFold. I agree that this procedure increases the focus on NC pairs, and from Fig4, we clearly see that there is need to improve these predictions, but it would be much more interesting (and challenging) to improve NC-pair predictions alongside general pair predictions. There could still be an option to only evaluate NC-pairs, but there is no need to exclude other pairs from the start.
4. For a  benchmark, the API is one of the major aspects (Is it easy to use? Can I integrate it into my research without any effort?) and so is reproducibility (How is the data processed exactly? Is this all in line with the results in the paper?). I would encourage the authors to publish the code anonymously. Today, that is really easy to do (e.g. via https://anonymous.4open.science/) and allows reviewers (and only reviewers) to check the code.
5. Generally, I'm not convinced that the proposed integration (and ranking) of multiple embeddings from RFMs is a giant step forward from the methodological perspective, even if the ranking approach via IsoScore is interesting.

**Minor**:
1. It would be interesting to see how far pseudoknots are considered here. Using PDB samples, we typically see a lot of non-canonical interactions and these include pseudoknots and base multiplets. I think these details should be shown somewhere in the paper.

[1] Singh, J., Hanson, J., Paliwal, K., & Zhou, Y. (2019). RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nature communications, 10(1), 5407.

[2] Fu, L., Cao, Y., Wu, J., Peng, Q., Nie, Q., & Xie, X. (2022). UFold: fast and accurate RNA secondary structure prediction with deep learning. Nucleic acids research, 50(3), e14-e14.

[3] Franke, J. K., Runge, F., & Hutter, F. Scalable Deep Learning for RNA Secondary Structure Prediction. CompBio workshop at ICML’23

[4] Singh, J., Paliwal, K., Zhang, T., Singh, J., Litfin, T., & Zhou, Y. (2021). Improved RNA secondary structure and tertiary base-pairing prediction using evolutionary profile, mutational coupling and two-dimensional transfer learning. Bioinformatics, 37(17), 2589-2600. → MSA input

[5] Chen, J., Hu, Z., Sun, S., Tan, Q., Wang, Y., Yu, Q., ... & Li, Y. (2022). Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions. arXiv preprint arXiv:2204.00300.

[6] A range of complex probabilistic models for RNA secondary structure prediction that includes the nearest-neighbor model and more

[7] Szikszai, Marcell, et al. "Deep learning models for RNA secondary structure prediction (probably) do not generalize across families." Bioinformatics 38.16 (2022): 3892-3899.

[8] Flamm, Christoph, et al. "Caveats to deep learning approaches to RNA secondary structure prediction." Frontiers in Bioinformatics 2 (2022): 835422.

[9] Qiu, X. (2023). Sequence similarity governs generalizability of de novo deep learning models for RNA secondary structure prediction. PLOS Computational Biology, 19(4), e1011047.

[10] Szikszai, M., Magnus, M., Sanghi, S., Kadyan, S., Bouatta, N., & Rivas, E. (2024). RNA3DB: A structurally-dissimilar dataset split for training and benchmarking deep learning models for RNA structure prediction. Journal of Molecular Biology, 436(17), 168552.

[11] Runge, F., Farid, K., Franke, J. K., & Hutter, F. (2024). Rnabench: A comprehensive library for in silico rna modelling. bioRxiv, 2024-01.

### Questions
1. Which exact model (weights) did you use for RNA-FM? Which for e.g. UFold? As far as I know there are different models available, some of which perform very well on e.g. data derived from PDB (although these models also typically ignore structure similarity during training which is unfortunate).
2. What was the rationale for using IsoScore as a score for embedding quality? Why not use some embedding metrics as e.g. described in [12] for DNA Language Models? I’m not super familiar with the current literature on RFM embeddings, but it seems that there are multiple approaches available to study the embedding quality and the cited paper for IsoScore from 2021 appears quite old when considering the recent progress in FM research.
3. As far as I can see, the matrix input in the REF-weighted self-attention is processed with convolutions to get a more local view. Did the authors also test other architectural choices? For example, some row- and column-wise attention should also work and result in a more global view, while preserving local features as well, or?

[12] Awasthi, R., Mend Mend Arachchige, G. S., & Zhu, X. (2025). Unsupervised evaluation of pre-trained DNA language model embeddings. BMC genomics, 26(1), 710.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper has two major contributions: a benchmark for predicting non-canonical base pairing interactions from 3D sequences, and a novel model to attack the proposed task. The benchmark pulls RNA structures from the PDB databank and annotates non-canonical base pairs using the RNAVIEW software. These interactions are particularly relevant for RNA function prediction and design as they often encode for highly specific interactions with other molecules, whereas the more stable canonical basepairs tend to provide the overall scaffold for the RNA fold. The proposed model is a transformer-based architecture which combines RNA foundation model embeddings with a decoder head to predict the base pair class across an input RNA sequence.

### Strengths
This is a highly valuable contribution. Direct RNA 3D structure prediction remains challenging, with all-atom models still showing limited performance due to the limited availability of 3D structures. Several new and prominent methods are turning to working with this level of structural detail with encouraging results [1][2]. Predicting non-canonicals (often termed 2.5D) provides a lot of useful structural detail and might be a better prediction target than full 3D for the time being. The proposed model also appears to be well structured. 

1. Karan, Aayush, and Elena Rivas. "All-at-once RNA folding with 3D motif prediction framed by evolutionary information." Nature Methods (2025): 1-13.
2. Carvajal-Patiño, Juan G., et al. "RNAmigos2: accelerated structure-based RNA virtual screening with deep graph learning." Nature Communications 16.1 (2025): 1-12.

### Weaknesses
1. Source code was not made available. This can be done [anonymously](https://anonymous.4open.science/). Without the source code I cannot verify the reproducibility or soundness of the results. (This is a major contributing factor for my current score)
2. There are no measurements of variance across the 4 splits, or across different model seeds.
3. Two important classes of structure prediction are absent from the benchmark: non-canonical motif-based prediction (e.g. CaCoFold [1], BayesPairing2 [2], JAR3d [3] and full 3D predictions (e.g. AlphaFold3-like models, RhoFold [4])
4. IsoScore is not defined in the paper with much detail, one has to go to the original reference and since it's an important component should be better explained in the main text.
5. The sequence similarity threshold remains somewhat unexplored. I suggest the authors try some more and less stringent settings to better understand the generalizability of proposed models.



[1] Karan, Aayush, and Elena Rivas. "All-at-once RNA folding with 3D motif prediction framed by evolutionary information." Nature Methods (2025): 1-13.
[2] Sarrazin-Gendron, Roman, et al. "Stochastic sampling of structural contexts improves the scalability and accuracy of RNA 3d module identification." International Conference on Research in Computational Molecular Biology. Cham: Springer International Publishing, 2020.
[3] Roll, James, et al. "JAR3D Webserver: Scoring and aligning RNA loop sequences to known 3D motifs." Nucleic acids research 44.W1 (2016): W320-W327.
[4] Shen, Tao, et al. "Accurate RNA 3D structure prediction using a language model-based deep learning approach." Nature Methods 21.12 (2024): 2287-2298.

### Questions
* Did you check that RNAVIEW is able to assign base pair geometries to chemically modified nucleotides? I know this was a problem for FR3D until recently. If not, it could introduce some significant bias to the benchmark. 
* Did you have a look at the performance broken down by base pair class? Given the sharp imbalance I think we would get a better picture of model performance by reporting the performance per class as well as in aggregate as already reported.
* For Fig 3 and section 4, how did you obtain predictions for the canonical pairs? From reading the problem formulation, it seems that the models were only trained to predict non-canonicals.
* For future work, I would suggest some more domain-specific evaluation functions. For example, some NC classes are geometrically more similar than others, so a wrong prediction but within a similar NC class might not be as bad, see Table 4 [1]. Likewise the location of an incorrect edge call if it is slightly shifted from the ground truth could also not be so deleterious so you could use some structure-aware evaluation functions (e.g. [2], but there are many others).
* Do you keep multi-chain RNAs?

[1] Stombaugh, Jesse, et al. "Frequency and isostericity of RNA base pairs." Nucleic acids research 37.7 (2009): 2294-2312.
[2] Agius, Phaedra, Kristin P. Bennett, and Michael Zuker. "Comparing RNA secondary structures using a relaxed base-pair score." Rna 16.5 (2010): 865-878.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces NC-Bench, a new benchmark for predicting RNA non-canonical (NC) base pairs with 925 sequences and 6,708 curated labels covering fine-grained edge (W/H/S) and orientation (cis/trans) tasks. Building on this, it proposes NCfold, a dual-branch transformer that selects top-k RNA foundation model (RFM) embeddings via IsoScore, fuses them with Representative Embedding Fusion (REF), and injects them into the model through REF-weighted self-attention. Experiments on NC-Bench include traditional ML and RFM baselines, ablations, and a zero-shot comparison to canonical-focused secondary-structure methods.

### Strengths
First standardized benchmark dedicated to NC base-pair prediction with defined edge/orientation subtasks.

IsoScore-based ranking plus REF and REF-weighted attention to couple sequence features with structural priors.

Traditional baselines, multiple RFMs, ablations, and zero-shot references are reported.

### Weaknesses
925 sequences and heavily skewed class distributions limit learning and can inflate simple metrics

All frozen-RFM + linear baselines fail to predict positives on the edge task (MCC≈0), weakening fairness/interpretability of comparisons.

Canonical-focused methods largely predict “non-pair,” yielding high accuracy but near-zero recall—making the reference comparison hard to interpret.

### Questions
How do you justify that 925 sequences / 6,708 labels provide enough statistical power and representativeness for a reliable benchmark? Any sampling/power analyses?

To what extent do gains come from REF selection vs. REF-weighted attention? Could modest fine-tuning of RFMs with stronger heads close the gap?

Why do frozen RFM linear probes predict all negatives on edge classification (MCC=0)? Would balanced losses, calibrated thresholds, or shallow MLPs recover positives?

Do predicted NC pairs align with known geometric/biophysical patterns across RNA families? What error modes appear most often under the observed class imbalance?

### Soundness
3

### Presentation
2

### Contribution
3
