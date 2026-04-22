# h-MINT: Modeling Pocket-Ligand Binding with Hierarchical Molecular Interaction Network

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 2, 4, 4

## Abstract
Accurate molecular representations are critical for drug discovery, and a central
challenge lies in capturing the chemical environment of molecular fragments,
as key interactions, such as H-bond and π stacking, which occur only under specific
local conditions. Most existing approaches represent molecules as atom-level
graphs; however, individual atoms cannot express stereochemistry, lone pairs,
conjugation, and other complex features. Fragment-based methods (e.g., principal
subgraph or functional group libraries) fail to preserve essential information such
as chirality, aromatic bond integrity, and ionic states. This work addresses these
limitations from two aspects. (i) OverlapBPE tokenization. We propose a
novel data-driven molecule tokenization method. Unlike existing approaches, our
method allows overlapping fragments, reflecting the inherently fuzzy boundaries
of small-molecule substructures and, together with enriched chemical information
at the token level, thereby preserving a more complete chemical context. (ii) h-
MINT model. We develop a hierarchical molecular interaction network capable
of jointly modeling drug–target interactions at both atom and fragment levels. By
supporting fragment overlaps, the model naturally accommodates the many-to-
many atom–fragment mappings introduced by the OverlapBPE scheme. Extensive
evaluation against state-of-the-art methods shows our method improves binding
affinity prediction by 2-4% Pearson/Spearman correlation on PDBBind and LBA,
enhances virtual screening by 1-3% in key metrics on DUD-E and LIT-PCBA, and
achieves the best overall HTS performance on PubChem assays. Further analysis
demonstrates that our method effectively captures interactive information while
maintaining good generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a method for biomolecular interaction modelling (protein-ligand binding, specifically). This method has two novel components: 1) OverlapBPE tokenization, which is a novel tokenization technique building upon BPE, where different tokens may overlap each other, capturing more complex molecular substructures, like rings and their functionalizations; 2) h-MINT, a hierarchical SE(3)-equivariant graph neural network built to exploit both the 3D information about protein-ligand pockets and the fragment information derived from OverlapBPE.

The results show that the method improves with regards to relevant baselines

### Strengths
1. The OverlapBPE tokenization is an elegant way of capturing molecular substructures, while preserving functional units.
2. The h-MINT, although not particularly novel on its own (as it is a hierarchical GNN), is a reasonable choice for exploiting the information from the protein-ligand binding poses and the fragment information derived from OverlapBPE.
3. The results show the advantage of the method.
4. The evaluation describes the deviation between multiple experiments which provides a necessary idea of the significance of the differences with regards to other methods.

### Weaknesses
1. The choice of evaluation benchmarks might lead to undiagnosed overfitting, as the controls for data leakage focus on the protein targets, rather than the molecular ligands. This has been shown to be problematic in some cases. As this problem occurs both on the method and the baselines is not a critical problem, but it is a limitation that should be acknowledged.

### Questions
None

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
This paper introduces a novel framework for modeling protein-ligand interactions, for the purpose of improving performance on drug discovery tasks like binding affinity prediction and virtual screening. The authors identify that existing molecular representation methods, particularly those based on fragmenting molecules into disjoint sets, often fail to preserve crucial chemical information such as aromaticity, chirality, and ionic states.  A new molecular tokenization algorithm is proposed. Unlike prior methods, OverlapBPE allows fragments to overlap, better preserving the local chemical context and molecular substructures. The authors also developed a hierarchical Molecular Interaction Network for interaction prediction. Extensive experiments are conducted on several benchmarks.

### Strengths
1. The idea of OverlapBPE is novel and well-motivated. It effectively addresses a clear limitation in prior work.

2. The experiment is extensive. The authors evaluate their method on four different datasets.

3. The paper is very well-written and easy to follow.

### Weaknesses
1. The OverlapBPE is primarily for molecular representation. How does it perform on molecular tasks? It seems to me that it is more naturel to test OverlapBPE on molecular tasks than the ligand-protein interaction task. 

2. The novelty of h-MINT is limited. The core architecture of the h-MINT model is heavily built upon existing equivariant graph transformers. The paper states that h-MINT follows GET's architecture. What are the precise architectural changes made for handling the overlapping tokens? Are the underlying equivariant transformer blocks identical to those in GET?

3. There is no study of the characteristics of the fragment vocabulary, such as the size and statistical information, the hierarchical structure, and the correspondence to common functional groups; thus, although the motivation is good, little chemical insight is presented. 

4. What is the training corpus to compute the token frequency? Does it affect the vocabulary?

5. The contributions of the tokenization and the h-mint model are convoluted. There is no ablation to show the effect of the tokenization clearly. 

6. In the experiments of VS, DrugCLIP and LigUnity are re-trained with PDBBind. Why do they have such a big difference in performance? The only difference is the ranking loss. The ranking loss in LigUnity needs samples of the same protein and different ligands, which is not common in PDBBind. How many such samples are there in PDBBind? 

7. No code available. The reproducibility statement is weak.

### Questions
See Weaknesses

### Soundness
3

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
5

### Summary
The paper proposes an approach for tokenization of molecular graphs by considering overlapping tokens. It starts with frequent basic subgraphs treated as words in  an initial vocabulary. Adjacent subgraphs are merged to  form new tokens,  ovelapping words are allowed in the context. New tokens' frequency is calculated across the database and most frequent new tokens are kept. 
The tokenized graphs are transformed into new graphs whereas the nodes are subgraphs and edge are overlapping information. The authors use an equivariant  GNN to transform the given subgraphs for predicting binding affinity with target. The experimental results carried out on two standard benchmarks and comparison was done to compare the proposed approaches with baselines regarding equivariant  GNN used for neural potential learning and other variation of all atoms GNN. The proposed approach outperformed the given approaches with a significant margin.

### Strengths
The idea is interesting, the results look significant compared to related baselines.

### Weaknesses
I have a concern on the chosen baselines. It is well known that  representation learning approaches is behind  molecular fingerprint on molecular property prediction tasks. Molecular fingerprint such as Morgan Fingerprints, ERP, Avalon available in the RDKit tools can summary important substructure and functional groups in a small molecule and being better or comparative to other molecular representation learning approaches (see the leaderboard for ADMET drug property prediction by TDC).

The idea provided in this paper is not new if considering molecular fingerprint where overlappping subgraphs or functional graphs are considered as a fingerprint of the molecules. The new part may concern the preserve of the connectivity between substructure in the graphs but it is unclear whether that information is useful. I would suggest the authors to perform additional experiments whereas fingerprints are used as feature representation of ligands and ESMas feature representation of proteins, then use simple approach such as lightgbm on the top of the provided features to predict binding affinity. That would be a strong confirmation of the significance of the proposed methods compared to existing fingerprint based approaches.

### Questions
Could you please compare your approach to simple baselines: taking ESM-2 as representation of protein, MorganFingerprint , ERP and Avalon as combined representation of ligands, run HPO in lightgbm models to find best set of hyperparameters of lightGBM and report the results of that baseline approaches in your paper?
If the experiments show significance results w.r.t to that baseline I am happy to raise my score.

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
4

### Summary
This paper proposes OverlapBPE, an overlapping, chemically consistent fragment tokenization that preserves aromatic systems, chirality, and ionic states, and h-MINT, a hierarchical, SE(3)-equivariant interaction network that couples atom–fragment representations via bilevel attention and token-expanded geometric edges. On PDBBind and LBA, h-MINT improves binding-affinity prediction over baselines; on DUD-E/LIT-PCBA it yields better zero-shot virtual screening metrics.

### Strengths
1. The proposed OverlapBPE tokenization effectively preserves crucial chemical information.

2. The h-MINT architecture enables many-to-many atom–fragment interactions via bilevel attention and equivariant message passing.

### Weaknesses
1. The tokenizer and architecture are co-developed, but their effects are not separated. Baselines isolating each component are needed to attribute gains.

2. The many-to-many mapping between atoms and fragments results in more computationally expensive graph construction. The effect of added overhead is acknowledged but not quantitatively discussed. Since the proposed h-MINT model targets the same virtual screening setting as LigUnity and DrugCLIP, a runtime comparison would be highly informative.

3. Virtual screening results are relatively weak, showing only marginal improvements and limited evidence of practical benefit.

4. The comparison with LigUnity seems not entirely fair, since h-MINT adds an extra MSE loss and no ablation is provided.

5. The claim in Section 5 about false-positive fragments is important but underexplored. There is no discussion on how such fragments might affect virtual screening results.

### Questions
1. I would appreciate it if the authors could share results on benchmarks such as JACS/Merck for affinity ranking and DEKOIS for virtual screening. These datasets are commonly used in related works like LigUnity and could help clarify the model’s generalization performance.

2. In Table 3, LigUnity performs substantially better than DrugCLIP on virtual screening metrics. I would be interested in the authors’ view on the role of the ranking loss in virtual screening. The original LigUnity paper reported that this term had little to no effect on screening performance. Have the authors observed different behavior in h-MINT or performed ablations to assess its impact?

### Soundness
2

### Presentation
3

### Contribution
2
