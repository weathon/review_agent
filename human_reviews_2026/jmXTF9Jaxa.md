# Unifying Structure- and Ligand-based Drug Design via Contrastive Geometric Learning

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Structure-based computational drug design, which employs methods trained on large datasets of protein-ligand complex structures, 
has been revolutionized by breakthroughs such as AlphaFold. In parallel, ligand-based computational drug design, driven by models 
trained on extensive bioactivity resources, has impacted drug discovery by enabling the simultaneous prediction of numerous biological effects of small-molecule ligands. Yet, despite recent advances in both structure- and ligand-based approaches, no existing method integrates them effectively at scale. We introduce **Con**trastive **G**eometric **L**earning for **U**nified Computational **D**rug D**e**sign (ConGLUDe), an approach that leverages both structure- and ligand-based training data through geometric and contrastive learning. The ConGLUDe architecture combines a geometric protein encoder, producing both spatial binding pocket and global protein representations, with a ligand encoder. The encoders are trained jointly via contrastive learning on 20K protein-ligand complexes from PDBbind and 77M ligand-based datapoints from ChEMBL, PubChem, and BindingDB. With ConGLUDe, multiple key drug discovery tasks, including virtual screening, binding pocket prediction, ligand-conditioned pocket selection and target fishing, can be addressed within a single model. ConGLUDe achieves state-of-the-art performance on virtual screening benchmarks and strong results across other tasks, demonstrating the benefit of joint structure-ligand training. By replacing a set of specialized models with a single system and by unifying structure- and ligand-based paradigms, ConGLUDe represents a major step toward foundation models for drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ConGLUDe, a unified framework for drug discovery that jointly leverages both structure-based and ligand-based information through contrastive geometric learning. The architecture integrates a geometric protein encoder (based on VN-EGNN) and a ligand encoder, trained via contrastive learning on structure complexes from pdbbind and ligand data from ChEMBL, PubChem, and BindingDB. ConGLUDe is evaluated on four tasks: virtual screening, binding site prediction, ligand-conditioned pocket selection, and zero-shot target fishing. The method achieves good performance those tasks.

### Strengths
1. Different types of tasks are conducted, including virutal screening, binding site prediction, ligand conditioned pocket selection, and target fishing.
2. The improvements on LIT-PCBA dataset is impressive

### Weaknesses
1. The paper feels somewhat disjointed, lacking a clear focus. It covers many tasks, but the analysis is not sufficiently deep. Important components like the ablation study are placed in the appendix, and the main text lacks further analysis and visualization.
2. The technical contribution is relatively limited—the work largely appears to be an application of VN-EGNN across different tasks.
3. The performance gains on several tasks are marginal, such as on DUD-E for virtual screening, and the LIGAND-CONDITIONED POCKET SELECTION task in Table 3.

### Questions
see Weaknesses

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
This paper introduces ConGLUDe framework, which aims to unify structure-based and ligand-based drug design through contrastive geometric learning. ConGLUDe combines geometric protein encoders and ligand encoders, and is jointly trained on multi-source structural and bioactivity data. This unified model can handle multiple drug discovery tasks, achieving SOTA performance and marking a step towards a foundational model for drug discovery.

### Strengths
1. This work has interesting motivation, proposing a novel unified framework with very broad applications.

2. The writing is clear and well-structured, with a straightforward model that is easy to understand and follow.

3. The experimental results are comprehensive, demonstrating strong multi-task and zero-shot generalization capabilities.

### Weaknesses
1. The loss term calculations ($\mathcal{L}\_{m2p}$ and $\mathcal{L}\_{p2m}$) in Eqn. 9-11 do not align with Figure 2. In Figure 2, the illustrations and descriptions of $\mathcal{L}\_{m2p}$ and $\mathcal{L}\_{p2m}$ appear to be reversed.

2. The current model integrates two data types through alternating training across different batches, which is similar to multi-task learning rather than deep information fusion. Could there be a more intrinsic fusion mechanism? For example, can knowledge learned from large-scale ligand data (such as target selectivity) more directly guide or constrain the learning process from structural data?

3. The discussion of ablation results is insufficient. The ablation experiments in the appendix (Table G4) show that on the DUD-E dataset, the structure-only model significantly outperforms the complete model. While the authors note that DUD-E is less realistic than LIT-PCBA, this result somewhat undermines the core argument that "fusing both data types is crucial." Consider moving this discussion to the main text with a deeper analysis of why this phenomenon occurs.

Minor issues:

a. Many equations lack proper punctuation.

b. Why is there no experimental results analysis in Section 4.2 BINDING SITE PREDICTION?

c. What is the bolding criterion in Table 1? Why do two values appear bolded in the same column? Many appendix tables have similar issues.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
### Summary
The paper presents ConGLUDE, a unified framework that combines structure-based and ligand-based drug discovery via geometric contrastive learning. It co-trains a protein encoder (for both binding-pocket and global representations) and a ligand encoder on large structural (PDBbind) and bioactivity datasets. The model achieves state-of-the-art zero-shot virtual-screening results on DUD-E and LIT-PCBA, outperforming other methods on some metrics.

### Strengths
---

### Strengths
- Clear unification: Bridges SBDD and LBDD with a simple, well-motivated contrastive framework.  
- Scalable and generalizable: Handles diverse drug-discovery tasks (screening, pocket prediction, target fishing).  
- Timely contribution: Moves toward foundation-model paradigms for computational drug design.

### Weaknesses
---

### Weaknesses
- Limited empirical advantage: The reported results do not consistently surpass other ML-based benchmarks. For example, in virtual screening, DrugCLIP achieves comparable results on DUD-E, despite being trained on less data and without ligand-based datasets.  
- Low methodological novelty: The structure encoder is adopted from prior work, while the ligand encoder consists of only a few MLP layers. The contrastive loss also appears to be heavily inspired by DrugCLIP.  
- Missing baselines: The ligand-conditioned pocket selection task should be compared with stronger baselines, such as those in the DiffDock paper or all-atom structure prediction models like AlphaFold3. Additionally, no baseline method is provided for the target-fishing task.  
- Unclear characterization of ligand-based design: Ligand-based drug design typically applies to cases where target structures are unavailable or unknown, relying on 2D/3D similarity of known active molecules. However, in this paper, the authors still use protein representations during “ligand-based training,” which may not fully align with that definition.

### Questions
### Questions
- Why not encode small molecules as graphs? This is a more common choice in molecular representation learning.  
- How does the speed of this method compare to DrugCLIP?  
- Since the data split is based on protein sequences, how similar are the most similar molecules between the training and test sets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents ConGLUDe (Contrastive Geometric Learning for Unified Computational Drug Design), a framework that unifies structure-based (SBDD) and ligand-based (LBDD) drug design. It jointly trains a geometric protein encoder and a ligand encoder using contrastive learning on both 3D protein–ligand complexes (PDBbind) and large-scale bioactivity datasets (ChEMBL, PubChem, BindingDB). ConGLUDe performs multiple drug discovery tasks—including virtual screening, binding pocket prediction, ligand-conditioned pocket selection, and target fishing—within one model. Experiments show state-of-the-art or competitive results across benchmarks, demonstrating effective integration of structure and ligand data. The method’s limitations include reliance on available 3D protein structures and chemical space coverage, but it represents a step toward foundation models for drug discovery

### Strengths
The paper is clearly written, and the description of the proposed model is well articulated. I agree with the idea of leveraging both large-scale protein data and extensive molecular data for modeling interactions.

### Weaknesses
1. Although the authors emphasize the goal of unifying large-scale protein and molecular data to train a unified model, I believe their proposed method does not fully achieve this. In practice, the model is mainly trained on PDBBind or BindingDB or CHEMBL, containing protein–ligand pairwise data. Such datasets are far smaller in scale than purely protein datasets (e.g., UniProt) or purely molecular datasets. Therefore, in terms of data utilization, the approach is not fundamentally different from previous methods.
2.	Regarding the model, the main innovation lies in improving the protein encoder to simultaneously encode both the pocket and the protein. However, this is essentially an enhancement of the encoder design rather than a change in the overall training paradigm. Architecturally, the structure-based training remains similar to other contrastive learning–based approaches, the ligand-based training only adds a new loss term and dataset, without introducing a novel architecturally.
3.	The virtual screening results appear relatively marginal, performing roughly on par with DrugCLIP and SPRINT on two benchmark datasets.

### Questions
1.	The authors should show that their adaptive pocket–encoder approach outperforms the two-step method (e.g., Fpocket + pocket encoder) to justify the value of unifying protein and pocket representations in one model.
2.	How do the authors view GNN/EGNN models’ ability to encode geometry, particularly the trade-off between equivariance and efficiency? Could data augmentation replace strict equivariance to enable usage of more powerful architectures?

### Soundness
3

### Presentation
3

### Contribution
2
