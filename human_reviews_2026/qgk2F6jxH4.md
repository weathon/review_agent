# SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Accurate prediction of protein-ligand binding affinities remains a cornerstone problem in drug discovery. While binding affinity is inherently dictated by the 3D structure and dynamics of protein-ligand complexes, current deep learning approaches are limited by the lack of high-quality experimental structures with annotated binding affinities. To address this limitation, we introduce the Structurally Augmented IC50 Repository (SAIR), the largest publicly available dataset of protein-ligand 3D structures with associated activity data. The dataset comprises $5,244,285$ structures across $1,048,857$ unique protein-ligand systems, curated from the ChEMBL and BindingDB databases, which were then computationally folded using the Boltz-1x model. We provide a comprehensive characterization of the dataset, including distributional statistics of proteins and ligands, and evaluate the structural fidelity of the folded complexes using PoseBusters. Our analysis reveals that approximately $3 \%$ of structures exhibit physical anomalies, predominantly related to internal energy violations. As an initial demonstration, we benchmark several binding affinity prediction methods, including empirical scoring functions (Vina, Vinardo), a 3D convolutional neural network (Onionnet-2), and a graph neural network (AEV-PLIG). While machine learning-based models consistently outperform traditional scoring function methods, neither exhibit a high correlation with ground truth affinities, highlighting the need for models specifically fine-tuned to synthetic structure distributions. This work provides a foundation for developing and evaluating next-generation structure and binding-affinity prediction models and offers insights into the structural and physical underpinnings of protein-ligand interactions. 
The link to the data will be added upon publication, to preserve anonymity of the submission.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Structurally Augmented IC50 Repository (SAIR), a dataset of over 1 million unique protein-ligand pairs with associated activity data curated from ChEMBL and BindingDB. The 3D structures for these complexes were computationally generated using the Boltz-1x co-folding model. The paper presents a characterization of the dataset, an evaluation of the structural quality using PoseBusters, and benchmarks for several binding affinity prediction models on the new data.

### Strengths
1. The dataset's size is a significant contribution that may contribute to addressing the data scarcity problem in structure-based modeling. The authors' commitment to making the dataset publicly available is highly commendable.
2. The use of a standard validation tool (PoseBusters) to confirm the physical plausibility of the computationally generated structures adds credibility and provides users with confidence in the data's quality.
3. The analysis correlating Boltz-1x confidence metrics (e.g., iPTM) with experimental binding affinity is a novel and interesting finding. This is timely, and it echos recent observations about the potential of AlphaFold-like models to rank binding affinities[1].

[1] Hong X, Gao B, Jia Y, et al. How Good is AlphaFold3 at Ranking Drug Binding Affinities?[J]. bioRxiv, 2025: 2025.05. 27.656341.

### Weaknesses
**Major Concern**: 
The use of PoseBusters is a commendable step for ensuring the physical and chemical plausibility of the generated structures. However, this validation does not—and cannot—confirm the biological correctness of the binding pose. Boltz-1x performs prediction from sequence, which is analogous to a "blind docking" experiment. This creates a significant risk that the model places a ligand in a physically plausible but biologically incorrect pocket, which would fundamentally mislead any downstream model. 

For proteins with structures in PDB, I recommend performing structural alignments between their generated structures and known holo-structures to quantify pocket overlap and correctness. For the proteins without known structures—where the predictions are inherently more speculative as they fall outside Boltz's structural training domain—this increased uncertainty should be explicitly discussed. I recommend comparing the predicted pocket against annotated functional domains from UniProt or the known pockets of homologous proteins. Furthermore, some datasets constructed with template-docking methods (e.g., BindingNet v2[1]), which assume similar ligands bind in the same pocket and can give mostly correct pockets, should be more adequately discussed. 

**Minor Concern**:
The term "assay" appeared many times in this paper. However, in the bioactivity prediction field, an "assay" is often defined as a set of measurements for a single target conducted under a consistent experimental protocol (often corresponding to a unique ChEMBL assay ID). To avoid ambiguity, I recommend using the more precise term "assay type"(binding, functional, ADMET, ...), "assay level"(cell, target, ...), or "measurement"(IC50, Ki, Kd). However, I still strongly recommend providing the ChEMBL assay ID and organizing data in BindingDB by assays following BindingNet[1] and ActFound[2]. Data from different assays, even if they report the same "assay type," can vary dramatically due to differences in experimental protocols (e.g., cell-based vs. biochemical), conditions (pH, temperature, cofactors), and instrumentation. Lumping these data together without distinction introduces severe batch effects, and there is also evidence for the benefit of considering "assay" in modeling [3].

[1] Zhu H, Li X, Chen B, et al. Augmented BindingNet dataset for enhanced ligand binding pose predictions using deep learning[J]. npj Drug Discovery, 2025, 2(1): 1.
[2] Feng B, Liu Z, Huang N, et al. A bioactivity foundation model using pairwise meta-learning[J]. Nature Machine Intelligence, 2024, 6(8): 962-974.
[3] Feng B, Liu Z, Li H, et al. Hierarchical affinity landscape navigation through learning a shared pocket-ligand space[J]. Patterns, 2025, 6(10).

### Questions
No questions.

### Soundness
3

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
4

### Summary
The paper introduces **SAIR**, a million-scale synthetic structure–activity dataset ($\approx$ 1.05M protein–ligand systems) built by curating bioactivity from ChEMBL and BindingDB, then generating 3D complexes using Boltz-1x co-folding. The curation removes entries with missing identifiers, validity flags, inequalities, and measurements outside a defined dynamic range; ligands are standardized (desalting, neutral-pH protonation, RDKit canonical SMILES). To avoid leakage from experimental structures, (UniProt, CCD) pairs present in PDB are excluded. The authors assess structural plausibility with PoseBusters, report overall low failure rates (only $\approx$ 0.53% of complexes have all five poses failing), analyze confidence metrics (iPTM, interaction-PTM, complex iPLDDT) vs potency, and benchmark a small set of affinity predictors (Vina/Vinardo, OnionNet-2, AEV-PLIG).

### Strengths
- **New dataset for proper demand.** If fully released, SAIR could be a valuable pretraining/analysis data source for structure-based binding affinity prediction models. Since the major limitation of training PLI models is lack of paired data, this approach is suitable in terms of its purpose and time.
- **Scale and transparency.** Clear end-to-end pipeline from bioactivity ingestion to synthetic structure generation; explicit de-duplication and PDB-leakage control. As the authors stated, the computational cost for making this data is very expensive and thus this dataset is invaluable.
- **Interesting diagnostics.** Introduction/usage of interface-focused confidence metrics (interaction-PTM, iPTM) and assay-aware correlation analyses (biochemical > cell).

### Weaknesses
Despite the strengths of the proposed dataset, my major concern is rooted in the fact that creating a synthetic structure-affinity dataset should minimize uncertainty in both structure and affinity.
1. **More rigorous validation of predicted structure.** All complexes are treated as monomers with canonical UniProt sequences, despite many targets (e.g., GPCRs) being oligomeric or using non-canonical constructs. Recent studies show that co-folding models can predict plausible-looking yet physically inconsistent pockets/poses (including persistent ligand placement after pocket disruption), yielding false positives in downstream analyses. [1,2] Thus, I suggest that the authors include an additional assessment of pocket validity. For instance, they could examine whether the predicted pocket residues are consistent with experimentally validated binding sites observed in highly similar structures associated with proteins with similar sequence, especially for the proteins from rare protein family.
2. **Potential distillation bias.** Because Boltz-1x uses deterministic MSA subsampling and results improve most on a high-confidence subset, the dataset may over-represent 'easy' (well-constrained/high-MSA or training-set–similar) proteins and under-represent rare/low-MSA targets. I would welcome analyses quantifying how confidence filtering shifts family/MSA composition and whether conclusions hold under family-balanced or MSA-balanced evaluations. I think this may results in making false positive data for rare protein family.
3. **Confidence wording vs effect size.** In the page 7's first and second paragraphs, calling $r_{s}\approx 0.25$ "strong" overstates the effect; the paper's own narrative elsewhere acknowledges low absolute correlations comparable to interface metrics.

[1] Masters, Matthew R., Amr H. Mahmoud, and Markus A. Lill. "Investigating whether deep learning models for co-folding learn the physics of protein-ligand interactions." _Nature Communications_ 16.1 (2025): 8854.
[2] Škrinjar, Peter, et al. "Have protein-ligand co-folding methods moved beyond memorisation?." _BioRxiv_ (2025): 2025-02.

### Questions
1. **Affinity aggregation strategy over five structures.** For each protein–ligand pair, do you report model performance per-structure, or do you aggregate (mean/min/best) across the five predicted structures?
2. **Pocket diversity in main text.** I saw that there are no information about pocket diversity in the main text despite its detailed explanation is provided in appendix. In my opinion, high pocket diversity for several proteins may introduce unintended bias in data. I'd like to know about the authors opinion of this issue. If this is problematic, giving diversity of protein pocket and let the users handling about that issue themselves can be a solution.
3. **Cofactors and tiny ligands.** How are cofactors/ions handled (e.g., heavy-atom count near 1, as described in table 3)? Are such entries filtered or annotated, and how do they affect pocket identification and affinity correlations?
4. **About the degradation of performance of OnionNet-2.** In Fig. 6, OnionNet-2 attains only Pearson r $\approx$ 0.35 on SAIR, whereas its reported performance on the CASF-2016 scoring set is r $\approx$ 0.864. Given that OnionNet-2 relies primarily on protein–ligand contact patterns, this large gap suggests potential issues with pocket localization/contacts in the co-folded structures.

### Soundness
3

### Presentation
3

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
This paper introduces SAIR, the largest dataset of 3D protein-ligand structures with annotated binding affinity data. The dataset comprises over 1 million unique protein-ligand systems and a total of 5.2 million structures, was created by computationally folding complexes from the ChEMBL and BindingDB databases using the Boltz-1x model. The primary goal of SAIR is to provide a comprehensive resource for training and evaluating structure-based deep learning models in drug discovery.

### Strengths
1. The paper addresses an important real-world problem in structure-based drug discovery, i.e., lacking of large-scale protein-ligand 3D structural data with bioactivity annotations.
2. The authors provide detailed descriptions of the specific steps for filtering and processing data from ChEMBL and BindingDB.
3. The authors provide comprehensive and insightful experimental analysis.

### Weaknesses
1. One core conclusion of this work is that models trained on experimental structures show modest correlations on SAIR's synthetic structures, which the authors attribute to data distribution differences. However, it would be better to include an important validation experiment: fine-tuning or training (from scratch) a model on a SAIR subset, then evaluating on a held-out SAIR test set. Demonstrating significant performance improvement would more convincingly prove SAIR's value as a "training resource." Currently, the paper positions SAIR more as a "benchmark exposing existing model limitations" rather than demonstrating its potential for "enabling new model development".

2. The generated dataset relies on the structural quality generated by Boltz-1x. While the authors validated using PoseBusters with a low failure rate (~3%), complete dependence on a single generative model may introduce biases specific to that model. For instance, the model might underperform on specific protein families. It would be beneficial to explore these risks more thoroughly in the discussion.

3. Fig. 4 shows that "internal energy" is the primary cause of PoseBusters validation failures. Does this suggest the generated conformations may not be physically reasonable?

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

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
The manuscript introduces SAIR, a large-scale dataset of protein–ligand 3D complexes paired with activity labels. Activity data are curated from ChEMBL and BindingDB, and corresponding complex structures are computationally generated using the Boltz-1x model—this synthetic-structure generation is the core contribution. The work contributes: (1) a large, publicly available synthetic 3D structure–activity resource built with Boltz-1x from ChEMBL/BindingDB, (2) quality analysis highlighting failure modes, and (3) an initial benchmarking suite motivating methods calibrated to this data regime.

As an initial benchmark, they compare empirical scoring functions (Vina, Vinardo), a 3D CNN (OnionNet-2), and a GNN (AEV-PLIG). While ML models outperform classical scoring, correlations with ground-truth affinities remain modest, underscoring the need for models fine-tuned to synthetic structure distributions.

### Strengths
1.  Constructs the largest dataset of predicted protein–ligand 3D complexes paired with activity labels; if released, it would be a valuable community resource for training and benchmarking.

2.  Provides a preliminary benchmarking of multiple models on the dataset, offering a baseline reference for future methods.

### Weaknesses
1) Model choice: The use of Boltz-1 for complex generation is not justified against more robust contemporaries (e.g., AlphaFold 3).

2) Benchmarking impact: Current results do not show that the dataset materially improves performance of existing models.

3) Baselines and finetuning: Baselines omit modern 3D algorithms; no fine-tuning of foundation models (e.g., Boltz-2) or comparisons of frozen vs. full training.

4) 3D utility ablations: No comparisons to sequence-/2D-only models or to randomized/alternative poses to assess sensitivity to 3D quality and pose source.

5) IC50 Values from diverse assay conditions (system, temperature, readout) are pooled without harmonization or consistency analysis.

6) Over-reliance on correlation/AUC; missing RMSE/MAE and decision-centric metrics (e.g., enrichment, hit rate).

7) Experimental clarity: Dataset splitting (protein/scaffold/time) and leakage prevention strategies are not specified.

### Questions
1.  Why was Boltz-1 selected over more robust alternatives (e.g., AlphaFold 3)

2.  Show evidence that training on SAIR improves performance vs. existing datasets (e.g., pretrain/fine-tune on SAIR vs. not)


3.  Will the authors include stronger modern 3D baselines and fine-tune foundation models (e.g., Boltz-2), and compare frozen vs. full fine-tuning?

4.  Add ablations comparing sequence-only, 2D-only, randomized poses, and alternative pose generators to quantify sensitivity to 3D quality and pose source.


5.  Will the authors report RMSE/MAE, enrichment/hit-rate metrics, and calibration with confidence intervals?

6.  The exact split protocols (by protein/scaffold/time), identity/scaffold thresholds, preprocessing steps, and measures to prevent leakage

### Soundness
3

### Presentation
3

### Contribution
3
