# Towards All-Atom Foundation Models for Biomolecular Binding Affinity Prediction

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
Biomolecular interactions play a critical role in biological processes. While recent breakthroughs like AlphaFold 3 have enabled accurate modeling of biomolecular complex structures, predicting binding affinity remains challenging mainly due to limited high-quality data. Recent methods are often specialized for specific types of biomolecular interactions, limiting their generalizability. In this work, we repurpose AlphaFold 3 for representation learning to predict binding affinity, a non-trivial task that requires shifting from generative structure prediction to encoding observed geometry, simplifying the heavily conditioned trunk module, and designing a framework to jointly capture sequence and structural information. To address these challenges, we introduce the **Atom-level Diffusion Transformer (ADiT)**, which takes sequence and structure as inputs, employs a unified tokenization scheme, integrates diffusion transformers, and removes dependencies on multiple sequence alignments and templates. We pre-train three ADiT variants on the PDB dataset with a denoising objective and evaluate them across protein-ligand, drug-target, protein-protein, and antibody-antigen interactions. The model achieves state-of-the-art or competitive performance across benchmarks, scales effectively with model size, and successfully identifies wet-lab validated affinity-enhancing antibody mutations, establishing a generalizable framework for biomolecular interactions. Our open-source implementation is available at https://github.com/VectorShi/ADiT.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes the Atom‑level Diffusion Transformer (ADiT), a unified foundation model that aims to learn transferable representations for biomolecular complexes and predict binding affinity across diverse interaction types. The authors repurpose the AlphaFold 3 architecture for representation learning by:

- **Simplifying inputs.** they remove multiple‑sequence alignments and template conditioning, instead using a unified tokenization scheme where each protein residue and each heavy atom in a small molecule is treated as a token and enriched with features from the ESM‑2 protein language model.

- **Using diffusion transformers.** the model alternates between atom‑level and token‑level DiT blocks, performs atom‑to‑token pooling and token‑to‑atom unpooling, and applies a denoising pre‑training objective where Gaussian noise is added to atom coordinates.

- **Training at scale.** three model sizes (12M, 35M and 253M parameters) are pre‑trained on ~480k protein‑protein, ~433k protein‑ligand and ~427k protein–ligand structures from the PDB without labels. Fine‑tuning is performed on downstream tasks.

- **Evaluating on four benchmarks.** the model is tested on protein‑ligand, drug‑target, protein‑protein and antibody-antigen tasks. According to the authors, the largest variant ADiT‑L achieves the best or comparable results across tasks, shows performance improvements with larger model size, and assigns better ranks to affinity‑enhancing antibody mutations in a case study.

### Strengths
- ADiT attempts to unify representation learning across protein–protein, protein–ligand and antibody–antigen interactions. The use of a single architecture trained on all‑atom inputs is novel and aligns with the trend of foundation models.

- The authors detail a simplified featurization scheme that avoids heavy reliance on MSAs and templates while leveraging pre‑trained ESM embeddings. They clearly describe the hierarchical DiT architecture (Fig. 2) with atom‑to‑token and token‑to‑atom operations, diffusion transformer blocks and gating mechanisms, and provide algorithmic descriptions in the appendices.

- Results are presented on four different tasks. On the protein–ligand benchmark, ADiT‑L outperforms previous models such as ProNet and ProFSA. The authors provide an ablation table and pre‑training loss curves (Fig. 4) demonstrating scaling benefits.

- The antibody affinity maturation case illustrates that ADiT assigns lower (better) ranks to most experimentally validated affinity‑enhancing mutations compared with the smaller variants. Figure 3 in the paper visualizes the ranking of mutations for two antibodies, and the authors claim this suggests utility for antibody design.

- Figures 1–4 are crisp, readable, and self-contained (architecture overview, module layout, case study, scaling/ablation), and each experiment section cleanly separates Setup and Results. This structure makes assumptions, metrics, and splits explicit, significantly improving readability.

### Weaknesses
- **Claim of being the first all‑atom “foundation model” might be overstated.** The paper positions ADiT as a “first step toward developing all‑atom structure foundation models”. However, Boltz‑2 is a biomolecular all-atom foundation model, I believe. Because Boltz‑2 pre‑dates ICLR 2026, the authors should compare to it and clarify what is novel beyond prior work. 

- **Limited attention to data leakage and generalization.** Deep models trained on PDBbind and similar benchmarks are known to memorize training complexes due to train–test contamination. A 2025 study in Nature Machine Intelligence introduced the CleanSplit dataset to remove overlaps and found that performance of many models drops significantly when leakage is eliminated, suggesting poor generalization. The authors curate PDB clusters but do not quantify overlap between pre‑training and fine‑tuning sets or evaluate ADiT on leak‑proof benchmarks (e.g., CleanSplit). Without such evaluation, it is unclear if the model genuinely learns transferable representations rather than memorizing structural motifs. Currently, this is the main concern in the field.

[CleanSplit]: Resolving data bias improves generalization in binding affinity prediction

- **Potential over‑claim about scaling.** While larger ADiT variants exhibit modest improvements (e.g., Pearson increases from 0.626 to 0.645 on LBA‑30 and from 0.690 to 0.751 on the Davis dataset), the gains are sometimes small relative to parameter increases. The ablation study shows a 3.8% improvement in Pearson when using all‑atom information, but no comparison is made against equally large equivariant models or models trained on bigger synthetic datasets. It remains unclear whether scaling alone, rather than improved data or architecture, drives gains.

- **Unclear justification for ESM‑2 features and neglect of small‑molecule semantics.** ADiT uses the ESM‑2 (650 M) language model to embed protein residues; however, this model is trained only on protein sequences and may not capture ligand or nucleic‑acid features. For small molecules, the token type embedding is set to zero, meaning no evolutionary or contextual information is added.

### Questions
- How did you ensure that there is no overlap between pre‑training complexes and those used in downstream evaluation? Will you evaluate ADiT on leak‑free datasets (e.g., CleanSplit) or the synthetic–data‑augmented GatorAffinity benchmark to assess generalization?

[GatorAffinity]: GatorAffinity: Boosting Protein-Ligand Binding Affinity Prediction with Large-Scale Synthetic Structural Data

- Given evidence that equivariant networks like ELGN improve binding affinity prediction by capturing rotational symmetries, why did you choose a non‑equivariant architecture and not include geometric priors? 

[ELGN]: Equivariant Line Graph Neural Network for Protein-Ligand Binding Affinity Prediction

- Can ADiT be extended to DNA/RNA–protein interactions or multi‑chain assemblies? Does the unified tokenization scheme generalize to nucleic acids or small peptides? 

- Boltz‑2 (2025) jointly predicts structures and binding affinities, matches FEP accuracy, and includes dynamic ensemble data. Did you benchmark ADiT against Boltz‑2? If not, why?

If the authors address the weaknesses outlined above, or provide a right justification for the points raised in my questions, or clarify points I may have misunderstood, I would be willing to raise my score.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The model proposes an all-atom foundation model for the representation learning of biomolecular interactions. The interaction structure is tokenized into atom level and token (protein residue or ligand atom) level representation, in both single and pair-wise manners. The backbone of the model is a diffusion transformer that involves hierarchical atom-token feature aggregation and unpooling. The model is pre-trained on general protein-protein interaction data from PDB and then fine-tuned for multiple downstream prediction tasks: protein-ligand, protein-protein and antibody-antigen interactions, achieving competitive results.

### Strengths
- The hierarchical atom and token level representation is a good solution that addresses the balance between expressiveness and computational efficiency. The methods are clearly formulated and presented.

- The experimental results show clear improvements on protein-drug interaction prediction tasks and also competitive performance on protein-protein interactions. Also, the comparison between small, medium and large models demonstrates the scalability of the model.

- Additional case studies such as in silico antibody affinity maturation show good generalizability and applicability of the learned patterns.

### Weaknesses
See Questions

### Questions
- How is Token2Atom implemented?

- Some interpretability is needed. For example, does the attention capture atom or residue level patterns that facilitate the interaction?

- Appendix E3: if the protein-ligand edges are removed, how is interaction information represented and captured? In this way, the problem decomposes into one that separate embeds protein and drug (dual encoders) then combines them for prediction. How would atom level and pairwise information contribute in this scenario?



Minor:
- As the model didn't seem to rely on AlphaFold 3's tokenization or trained parameter, only some of the architecture, it is not really "repurposing AlphaFold 3".

- Protein-ligand interaction and drug-target interaction largely overlap, so it would be clearer if the two experiments are defined as different datasets instead of different tasks.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ADiT, an atom-level diffusion transformer that repurposes AlphaFold 3-style machinery from structure generation to representation learning for binding affinity prediction, jointly encoding sequence and 3D geometry without MSAs or templates. Pretrained on PDB and evaluated across protein–ligand, drug–target, PPI, and antibody–antigen tasks, ADiT achieves state-of-the-art or competitive performance, scales with model size, and identifies wet-lab validated affinity-enhancing antibody mutations.

### Strengths
1.  The work adopts an all-atom representation and enables interaction between atom-level and token-level features. The all-atom representation unifies the encoding of various biomolecular types, facilitating knowledge transfer across different interaction prediction tasks.

2.  The work  leverages the atom- and token-level representation modules of AF3 while removing dependencies on MSAs and templates, and simplifying the Pairformer blocks to enhance efficiency.

### Weaknesses
1.  When using AlphaFold 3, is the model trained from scratch or initialized with AF3 parameters? The Atom2Token average pooling is simple and reasonable, but the Token2Atom operation is insufficiently explained, does it simply assign token features to all constituent atoms, or are weights/attention mechanisms used?

2.  The performance is relatively weak, at comparable model sizes, ADiT shows no clear advantage, especially given the substantial pretraining cost. On protein–ligand tasks ADiT-S/M trails lighter, task-specific baselines such as CheapNet and BindNet, and on protein–protein binding affinity even the largest variant (ADiT-L) underperforms recent state-of-the-art models like BA-DDG and Light-DDG. This raises concerns about the practical benefit of the proposed pretraining strategy.

3.  The authors emphasize that the model is non-equivariant but do not explain why abandoning SE(3)-invariance is advantageous for binding prediction. Appendix E.1 mentions random rotation augmentation (centered on protein centroids) to encourage robustness and generalization, yet no ablation study is provided to support.

4.  The model uses only a structure-denoising task, which may be insufficient to capture complex features relevant to binding affinity. It would be beneficial to incorporate additional self-supervised tasks such as masked atom/residue prediction.

5.  The paper lacks efficiency analysis and training time reporting. All-atom representation models are typically computationally expensive. It would be helpful to include comparisons of ADiT’s training and inference speeds with other baselines

### Questions
1.  Do you train ADiT entirely from scratch or initialize any modules with AlphaFold 3 parameters?

2.  How is Token2Atom implemented: are atom features a direct copy from their parent token, or do you use learned weights/attention or distance-aware mixing? Please provide formulas and ablations.

3.  What empirical or theoretical reasons justify a non-SE(3)-equivariant design for affinity prediction?

4.  What is the performance impact of rotation augmentation (and centroid-centering) versus no augmentation? 

5.  Why rely solely on structure-denoising? Have you compared against masked atom/residue prediction, contrastive objectives, or multi-task pretraining?

6.  What factors drive ADiT’s underperformance on protein-protein affinity benchmarks relative to BA-DDG/Light-DDG?

7.  What are the training compute (FLOPs/GPU hours), wall-clock times, parameter counts, peak memory, and throughput for each model size?

8.  How do ADiT’s training and inference speeds compare to strong baselines (including all-atom and residue-level models) on standardized batch/complex sizes? Please include scaling curves with atom count.

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
5

### Summary
This paper repurposes AlphaFold3 to a representation learning model and then achieves protein binding affinity prediction. The paper applies two-stage learning algorithm with first pretraining a general model and then finetuning the pretrained model on specific tasks. The pretrained model is finetuned and evaluated on four tasks: protein-ligand, drug-target, protein-protein and antibody-antigen binding predictions.

### Strengths
The idea of repurposing AlphaFold3 to a binding prediction model is very interesting and useful in real world.

### Weaknesses
The weaknesses of this paper are listed as follows:

1. I appreciate the idea of the building upon AlphaFold3 achieves protein binding prediction and the proposed method is evaluated across several important tasks. However, the paper finetuned four tasks and there are four task-specific models, which is inconvenient for practical use in reality. If there are even more tasks like enzyme-cofactor, enzyme-substrate, target protein-binder affinity prediction,  that means there are will be more models. If different model size is considered, there will be ever more models. That would really reduce the utility of the proposed method. If the paper could merge all downstream tasks together and finetune the pretrained model on all merged tasks, I believe that might both increase the  performance due to multitask learning and also improve the convenience for practical use.

2. The paper lacks some baseline models for affinity prediction, like Rosseta, NERE [1].

[1] Unsupervised Protein-Ligand Binding Energy Prediction via Neural Euler’s Rotation Equation.

3. AlphaFold3 itself provides a metric ipTM to evaluate the binding interface geometry of the binding molecules, which can tell the binding affinity to some extent. On one hand, the author doesn't explain why they need to construct a different metric based on AlphaFold3 instead of directly using ipTM. On the other hand, the proposed method should compare with ipTM.

4. Since the proposed method heavily relies on AlphaFold3, the paper should clearly explain their difference from AF3, like what modules are reused and what modules are newly proposed in this paper.

5. The performance on some tasks are worse than the baseline methods like protein-protein interaction.

6. Some small issues like grammar error in line 214, "We employs ...".

### Questions
1. In 4.1, the paper mentioned they curated 427,947 protein-ligand interaction examples from PDB. However, generally there are just around 20k protein-ligand complexes in PDB as introduced in previous works. Can the authors explain how they curated each kind of complex data from the PDB?

### Soundness
3

### Presentation
3

### Contribution
2
