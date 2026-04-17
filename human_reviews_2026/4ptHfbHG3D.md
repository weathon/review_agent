# Multi-state Protein Sequence Design with DynamicMPNN

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
Structural biology has long been dominated by the one sequence, one structure, one function paradigm, yet many critical biological processes—from enzyme catalysis to membrane transport—depend on proteins that adopt multiple conformational states. Existing multi-state design approaches rely on post-hoc aggregation of single-state predictions, achieving poor experimental success rates compared to single-state design. We introduce DynamicMPNN, an inverse folding model explicitly trained to generate sequences compatible with multiple conformations through joint learning across conformational ensembles. Trained on 46,033 conformational pairs covering 75% of CATH superfamilies and evaluated using Alphafold 3, DynamicMPNN outperforms ProteinMPNN by up to 31% on decoy-normalized RMSD and by 12% on sequence recovery across our challenging multi-state protein benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors curated a dataset of similar sequences and their multiple conformations from the PDB. They trained an inverse folding model on this dataset to enable inverse folding with two target conformations. The paper proposes a new metric for evaluating multi-target inverse folding by comparing the target folded structure to the AF3-predicted structure, using the target as a template.

### Strengths
This paper challenges the “one sequence–one structure” assumption in inverse folding, which is an important direction to explore.  
The dataset curation approach is a clever use of the existing PDB database and MSA alignments to extract sequences with multiple possible protein conformations.

### Weaknesses
1. Using AF3 as a validation metric to assess whether a sequence folds into the desired structure is reasonable in the single conformation case. However, in the multiple conformation setting, where AF3 is known to underperform, this approach is questionable, particularly when the target structure is provided as a template. The choice of decoy structure in such cases could significantly affect the results, and there is insufficient evidence that this metric is meaningful or correlates well with real world folding behavior.  
2. The claim that the target sequence can form both target conformations is supported only by computational validation, with no experimental evidence demonstrating successful dual conformation inverse folding.
3. The paper feels somewhat rushed. The authors still have an extra page available to include additional results, while Table 2 and Figure 3 present essentially the same information, which is not an efficient use of space.

### Questions
1. How is the decoy structure selected for the metric? If it is chosen to be dissimilar to the targets but represents a structure the sequence is unlikely to fold into, it could significantly distort the normalized metric.  

2. Why was AF3 used for evaluation, given that other models such as Boltz or AlphaFlow are known to perform better at predicting structural ensembles?


3. Line 105 broken reference

### Soundness
2

### Presentation
2

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
The paper proposes a novel approach for protein design given multiple states, which correspond to multiple structures. Instead of an ad-hoc aggregation of predictions from different states, DynamicMPNN first encodes multiple states of protein structures, along with their binding partners, into an aggregated representation in latent space. Then, an autoregressive sequence decoder is applied to decode from the pooled representation. In addition, the paper proposes a more reasonable multi-state protein design evaluation based on folding with AF3 with structures of different states as templates. The paper also provides a valuable two-state protein dataset of more than 40000 conformer pairs. Benchmarked with a wide variety of baseline models, DynamicMPNN shows superior performance in both sequence recovery and refoldability.

### Strengths
1.The paper presents a complete contribution to the problem of multistate protein design, covering from dataset building, model design and evaluation methods.

2.The paper is well written with the idea being clearly demonstrated.

3.The benchmark provides a thorough comparison between various combinations of models and training strategies, providing a strong support to the pretraining-finetuning training pipeline.

4.Performance on the foldability evaluation based on AF3 shows significant improvement of DynamicMPNN.

### Weaknesses
1.Though it is argued that refoldability with a single protein folding model prediction is not suitable for multistate design, it would still be interesting to see how the refoldability would differ if the state conformers are not provided as templates in $\mathrm{AF3}(Y, X_k)$. In other words, it would still be interesting to see the refoldability with only $\mathrm{AF3}(Y)$. If the strong bias towards a single dominant state of the existing folding models (in this case, AF3) can be demonstrated over the curated evaluation datasets, it would be valuable to support the newly proposed refoldability definition.

2.DynamicMPNN is trained and evaluated with only on two-state proteins. Extending to proteins with more states can be non-trivial since it may require more sophisticated pooling strategy across the states. DSS in this paper shows no advantage over the simple pooling strategy. The case can be different if we extend to more states, where richer interaction between conformational channels could help.

3.There are minor flaws in the presentation of the paper, see the first 2 items in the Questions section.

### Questions
1.The Appendix number is missing in line 105.

2.In the caption under Figure 2, the description of 2(b) and 2(c) are mismatched with the figures. Their description in the caption should be swapped. 

3.How the binding partners are treated in AF3 in multi-state design evaluation.

4.Instead of using states as templates in AF3 evaluation, is it reasonable to provide only the designed sequence along with the sequence of binding partners to AF3, and use the folded structure, without using the ground truth state conformer as a template?

5.How does DynamicMPNN perform when using Single Training + MSD, using the same logit averaging strategy as ProteinMPNN-MSD? It would be interesting to see this result so that the effectiveness of unified encoding approach in multi training can be more clearly validated.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DynamicMPNN, an inverse folding model for generating protein sequence based on multiple conformational states of a given protein. It proposes to cluster sequences and conformations in PDB to build an augmented dataset with conformational heterogeneity for training. It uses template-based AF3 for self consistency refoldability check as a metrics, and have outperformed existing inverse folding baselines.

### Strengths
- Develop a novel method to include explicit multi-state conditioning in the model
- Design a new data processing pipeline to effectively augment PDB for structural heterogeneity
- Rigorous evaluation: decoy normalization mitigates template bias and focuses on relative compatibility with each state.
- Clear ablations show that combined training beats single-only or multi-only; simple Deep Set pooling matches more complex DSS.

### Weaknesses
- Dataset uses only the max-RMSD pair, potentially biasing toward extreme transitions and discarding informative intermediates.
- Entirely in silico; mapping the normalized AF3 gains to experimental hit rates is unknown.
- Missing appendix reference (line 105)

### Questions
- Since there're models that predict conformational ensembles, e.g. BioEmu, AlphaFlow. Have the authors tested the generated sequences with those models, and compare with the input structures?
- While during the training, the model uses pairs of conformations from the augmented dataset. However, how much those pairs would match the distribution of practical design demands is questionable.

### Soundness
4

### Presentation
4

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
The paper introduces DynamicMPNN, which designs protein sequences that are compatible with multiple conformational states. Instead of aggregating single-state predictions post hoc, the model jointly learns across conformational ensembles. Using a dataset of 46,033 conformational pairs spanning 75% of CATH superfamilies and an AlphaFold–based evaluation, DynamicMPNN outperforms ProteinMPNN on a multi-state benchmark.

### Strengths
1. The motivation is clear: the paper tackles a well-defined gap by explicitly modeling multi-state behavior—a cornerstone of many biological functions—where post-hoc aggregation approaches have historically shown low success rates.

2. The training dataset curation is careful and comprehensive.

### Weaknesses
1. The empirical analysis lacks stratification by biologically and structurally meaningful factors. A breakdown by motion class (metamorphic, hinge, transporter) would contextualize the relatively high absolute RMSDs and reveal where DynamicMPNN provides the largest benefits.

2. The paper does not report scalability or runtime characteristics, leaving unclear the training and inference costs (GPU hours, memory footprint), AF3 evaluation throughput per sequence-state pair, and how computation and memory scale with the number of conformational states m.

3. The demonstrated scope is restricted to two-state systems. Although broader applicability is discussed, there is no evidence on proteins with more than two conformational states or on more continuous conformational landscapes. At least one m > 2 case study would substantiate the abstract’s claim of multi-state generality.

4. The title and text use “protein design” broadly, but the contribution is specifically an inverse folding method for protein sequence design conditioned on multiple conformations. To avoid overclaiming and to align with community terminology, the manuscript should consistently use “protein sequence design” (or “inverse folding”) where appropriate, and reserve “protein design” for pipelines that include backbone generation. The related work should be expanded to cover widely used single-state protein sequence design methods, including but not limited to ProteinMPNN, ESM-IF, CarbonDesign, and GeoEvoBuilder, with a brief comparison of how DynamicMPNN differs (e.g., multi-state conditioning, encoding of binding context, pooling across conformers) and where single-state advances might transfer or serve as baselines.

5. Writing and formatting require polish. The unresolved cross-reference “Appendix ??” (around line 105) should be fixed; the duplicated citation “Praetorius et al. (2023) (2023)” should be corrected.

### Questions
Please address the questions in the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
