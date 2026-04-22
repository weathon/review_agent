# Pallatom-Ligand: an All-Atom Diffusion Model for Designing Ligand-Binding Proteins

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Small-molecule ligands extend protein functionality beyond natural amino acids, enabling sophisticated processes like catalysis, signal transduction, and light harvesting. However, designing proteins with high affinity and selectivity for arbitrary ligands remains a major challenge. We present Pallatom-Ligand, a diffusion model that performs end-to-end generation of ligand-binding proteins at atomic resolution. By directly learning the joint distribution of all atoms in the protein–ligand complexes, Pallatom-Ligand delivers state-of-the-art performance, achieving the highest *in silico* success rates in a comprehensive benchmark. In addition, Pallatom-Ligand's novel conditioning framework enables programmable control over global protein fold and atomic-level ligand solvent accessibility. With these capabilities, Pallatom-Ligand opens new opportunities for exploring the protein function space, advancing both generative modeling and computational protein engineering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method for generating protein–ligand complexes based on small-molecule ligands. The approach outperforms existing baseline methods on in silico evaluation metrics and, to some extent, addresses the challenge of generating protein–ligand complexes with non-helical protein structures.

### Strengths
1. The proposed method outperforms the current state-of-the-art on most evaluation metrics and, to some extent, increases the proportion of designable non-helical structures in the generated results.
2. The paper introduces a well-designed benchmarking strategy for this problem, which enriches the evaluation process.
3. The writing is clear and well organized.

### Weaknesses
1. In terms of novelty, the model architecture is almost identical to those of AF3 and P(allatom), with the only additions being the alpha ratio and solvent accessibility (SA) as inputs. There are no significant differences in the generation or training methods, indicating a lack of machine learning novelty.
2. Although the paper claims to generate more non-helical structures, this aspect is not compared against other models; the evaluation is only performed through self-comparison.

### Questions
Why was the SA experiment conducted on only three ligands?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Pallatom-Ligand, an end-to-end all-atom diffusion model for generating ligand-binding proteins. The key idea is to unify the representation of small molecules and protein residues at the atomic level and to jointly model their spatial and chemical interactions. The model introduces a ligand-aware all-atom diffusion transformer that integrates atom- and token-level representations to capture both global and local dependencies within protein-ligand complexes. Furthermore, a conditioning framework is proposed to control global protein fold diversity (through α/β ratio conditioning) and atomic-level ligand solvent accessibility (SA). Evaluation on eight chemically diverse small molecules using AlphaFold3-based component-specific metrics demonstrates improved in-silico structural plausibility over RFdiffusion2 and RFdiffusionAA.

### Strengths
1. The idea of a unified atomic representation for both proteins and small molecules is conceptually elegant and well-motivated. Modeling all atoms within one generative framework could in principle enhance data efficiency and the precision of atomic interactions.
2. The paper proposes a multi-level conditioning framework that allows explicit control over protein fold topology (via α/β ratio) and ligand solvent accessibility. This design makes the generation process more interpretable and programmable.
3. The authors conduct a broader benchmark than prior work, testing on eight ligands with diverse physicochemical properties, and systematically report AlphaFold3-based component-specific metrics. This gives a relatively thorough in-silico comparison.

### Weaknesses
1. The architecture introduces a “triple-attention” transformer combining token-, atom-, and pair-level streams, but the paper does not include any ablation study to clarify what each component contributes. Without that, it is unclear which part is essential for the observed performance gain.
2. The paper describes two sampling modes for dual-objective training, but there is no experimental analysis of how the chosen ratio between them influences training dynamics or the final results, as no sensitivity or ablation study is provided.
3. The evaluation framework relies entirely on AlphaFold3 predictions as pseudo–ground truth. This is acceptable for initial screening, but the paper could be clearer that these metrics measure structural consistency rather than true binding affinity or physical validity.
4. The presentation is difficult to follow for readers outside the protein-design community. Several evaluation metrics (e.g., pLDDT, ipAE) are introduced as part of the “AlphaFold3-based component-specific metrics,” but the paper provides little intuition about their meaning or justification. While these values come from AlphaFold3 predictions, the manuscript does not clarify how they reflect the underlying biophysical objectives or why they are appropriate for assessing generative quality. This lack of context makes it difficult for readers from a machine-learning background to interpret the reported results.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new diffusion model that jointly generates protein-ligand complexes operating at the atomic resolution. The key contributions are a novel diffusion transformer architecture for protein-ligand complexes and a novel conditioning framework that allows conditioning on global protein properties and atomic-level properties. Extensive experiments are presented to showcase the strong performance of this model.

### Strengths
* The paper is well-written and the key ideas are clearly presented.
* The idea of modeling the joint distribution of all atoms in a protein-ligand complex to improve generation quality and data efficiency is sound and promising.
* The presented experiments are quite extensive and showcase the superior performance of the proposed model compared to baselines.
* The introduced set of component-specific metrics that reflect the multi-faceted nature of the protein design problem is reasonable and should better evaluate the different models.

### Weaknesses
* The reliance on LigandMPNN to redesign the generated sequences hints at some limitations in the proposed model, especially because the model with MPNN outperforms the one without quite significantly (Figure 4). The authors should discuss the problems with the structures directly generated by their model and explore possible solutions to address them.
* The model achieves a quite poor diversity rate compared to the baselines (0.14-0.17 vs 0.30). In the global control setting, this seems to be improved, but at the cost of lower success rates. The authors should discuss this and some potential fixes.
* The paper assumes that the reader is familiar with the field of protein design. It would greatly improve the accessibility of the paper if some technical terms were explained, e.g., the metrics discussed in section 3.1: pLDDT, ipAE, etc.

### Questions
1. What is the Transition module used in Equations 2 and 5? I couldn't find any explanations in the paper.
2. The atom-level pair features $p$ used in Equation 4 should be defined along with the other variables at the start of that section.
3. In Table 1, I assume Nov. refers to novelty. If so, why is a lower value better? If not, please clarify what it means. Also, Div. is not formally introduced in the paper.

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
2

### Summary
The paper presents Pallatom-Ligand, an all-atom diffusion model for generating ligand-binding proteins directly from atomic coordinates. It jointly models all atoms in protein–ligand complexes using a triple-attention transformer and introduces two conditioning mechanisms: global control of protein fold (α/β ratio) and atomic-level control of ligand solvent accessibility. Benchmarked against RFdiffusionAA and RFdiffusion2 on eight ligands, Pallatom-Ligand achieves higher in silico success rates and better control over structure and binding geometry.

### Strengths
1. Unlike backbone-only or inverse folding models, Pallatom-Ligand directly generates protein–ligand complexes at the atomic level, capturing fine-grained chemical and spatial interactions critical for ligand specificity.
2. The model introduces multi-level conditioning (fold ratio and solvent accessibility), enabling programmable structural control—something that earlier diffusion-based models lack.

### Weaknesses
1. The training relies on a relatively small number of protein–ligand complexes with strong bias in ligand–fold distribution. Although the authors introduce a dual sampling strategy, scalability to rare or novel ligands may be limited.
2. All-atom diffusion transformers with block-sparse attention and multi-level conditioning are computationally intensive, potentially restricting usability for large complexes or extensive sampling.

### Questions
Can Pallatom-Ligand generalize to unseen ligand chemotypes or novel binding motifs outside the training distribution, and how does its performance degrade in such zero-shot settings?

### Soundness
3

### Presentation
3

### Contribution
2
