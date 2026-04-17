# EDMolGPT: A Decoder-Only Framework for 3D Drug Design via Electron Density

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Electron density-guided drug design is a promising structure-based drug discovery (SBDD) frontier, crucial for delineating dynamic molecular features and intermolecular interactions. Existing methods leveraging electron density for \textit{de novo} molecule generation employ a two-stage process: generating hypothetical binder electron densities within a pocket, then interpreting them into molecules. While mitigating bias from binders pre-existing in the pocket, these approaches' two-stage nature can lead to error accumulation. Furthermore, these methods are limited by rigid pocket assumptions, which may compromise the diversity of the generated electron density. These limitations often result in drug-like molecules lacking favorable three-dimensional (3D) conformations or conversely, 3D conformations without assured drug-likeness. We introduce EDMolGPT, a novel decoder-only framework that directly synthesizes molecules from the low-resolution electron density point cloud derived from an existing binder. By leveraging this existing binder’s low-resolution electron density and avoiding explicit pocket structures, our strategy effectively mitigates bias, circumvents two-stage error, and negates rigid pocket limitations. EDMolGPT's autoregressive decoder-only architecture, guided by robust low-resolution electron density, efficiently generates binding molecules with high drug-likeness and favorable 3D conformations. Rigorous validation across 101 biological targets underscores its potential to accelerate novel therapeutic agent discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a drug design model based on the electron density representation of known ligands, employing an autoregressive Transformer architecture and introducing the FSMILES representation to achieve 3D molecular generation. The method extracts low-resolution point cloud representations via FFT and a high-frequency cutoff, and incorporates hydrogen-bonding information to provide a novel guidance mechanism for molecular generation.

Experiments are conducted on the DUD-E dataset to validate the approach on new ligands, and the Glide score is used as evaluation metrics, which better reflects real-world drug design scenarios.

However, the paper suffers from unclear representation, weak argumentation, insufficient baselines, and incomplete experimental analysis, all of which require further improvement and refinement.

### Strengths
- Method: This paper proposes a novel representation to guide 3D drug design, which integrates fragment, binding, and spatial information — all critical for real world application. The method also addresses key challenges in efficiency and scalability associated with spatial point cloud representations.

- Experiments: The study uses the DUD-E dataset as the test set and reports Glide scores, which provide more practically relevant and application-oriented evaluations compared to baselines that use CrossDocked datasets and Vina scores.

### Weaknesses
- Task: The reviewer argues that this work still essentially belongs to ligand-based drug design (LBDD), merely adopting a different form of conditional representation. Besides, the paper lacks controlled comparisons to demonstrate that the proposed representation is superior to those used in ED2Mol or ECloudGen.
- Presentation: Several main claims are not rigorously supported.
    - Error accumulation: The proposed point cloud representation itself still contains errors, which may arise from multiple sources — including binding pose, electron density calculation, h bond annotation, and discretization precision loss. Although the authors claim that their approach reduces error accumulation compared to two-stage methods, the representation they use is also error-prone, and the paper does not report any quantitative error analysis.
    - Flexibility: The claimed advantage of not requiring explicit pocket structures is actually a common feature of LBDD, rather than a unique contribution of this work. Moreover, since the point cloud is derived from a single ligand conformation, it cannot capture pocket flexibility.
    - Minor points: The paper lacks a clear demonstration or ablation of FSMILES effectiveness, and the biological activity metric ECFP4 Tanimoto similarity (TS) is not clearly explained.
    - Docking score: How is the initial binding pose determined when calculating min score? Can you provide the experiment affinity or docking scores for reference ligands?
- Method:
    - Sorting point clouds purely by coordinate order may disrupt spatial locality.
    - The paper does not explain how positional embeddings — a critical component of Transformer architectures — are implemented in this context.
- Experiment:
    - Table 1 does not report the average molecular weight of each model. ED2Mol tends to generate smaller molecules, which typically exhibit lower docking scores, lower strain energy, and higher drug-likeness. The paper only reports ED2Mol’s molecular weight, but ED2Mol performs weakly overall (Table 1). The authors should have compared against stronger baselines.
    - Table 1 also omits important metrics such as diversity and novelty. Furthermore, Figure 6 only verifies that there is no data leakage between training and test sets, but does not confirm that the generated molecules are novel relative to the training data. Intuitively, for certain small molecules, using 199 points may provide excessive information about the reference ligand, effectively making the ligand-based design task easier. The authors need to demonstrate that their method can generate molecules that are significantly different from the reference ligand, rather than merely reproducing or slightly modifying it.

### Questions
see weakness

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
3

### Summary
The paper proposes EDMolGPT, a decoder-only (GPT-style) autoregressive model for structure-based drug design (SBDD) that generates 3D ligand molecules directly from low-resolution electron-density (ED) point clouds.

Unlike conventional SBDD approaches that condition on atom-level protein pocket structures or two-stage “ED→pocket→ligand” pipelines, EDMolGPT takes a 3D ED point cloud as the sole condition. Each sampled point is labeled with a pharmacophore type (e.g., H-bond donor/acceptor) to encode coarse chemical semantics. The point cloud and molecular tokens (fragmented SMILES and discretized 3D geometry tokens) are concatenated into one sequence, and a decoder-only Transformer is trained to predict the next molecular token autoregressively. During inference, the model generates ligand atoms and their relative coordinates step-by-step, constrained by predicted bond lengths, angles, and dihedrals.

On the DUD-E benchmark (101 targets), EDMolGPT achieves the highest bioactive recovery rate (41%) and competitive docking scores compared with Pocket2Mol, ED2Mol, Lingo3DMol, and MolCRAFT, while maintaining conformational stability without post-hoc relaxation.

Contributions:

- Introduces ED point clouds as a new conditional representation for SBDD that encodes both spatial and chemical context.

- Designs a decoder-only 3D molecular generator that unifies condition and generation in one autoregressive sequence model.

- Demonstrates strong performance and diversity on DUD-E, validating the feasibility of learning directly from physical electron-density signals.

### Strengths
Novel conditioning modality: Using electron density as the generative condition is a fresh, physically grounded idea that captures pocket flexibility and avoids hard atomic constraints.

Unified autoregressive formulation: The decoder-only design eliminates the need for a separate encoder, simplifying architecture and enabling efficient joint modeling of condition and ligand.

3D-aware tokenization: Integrating FSMILES with discretized coordinate and relative geometry tokens is elegant and practical for coupling chemical and spatial features.

Strong empirical results: Substantial gains on DUD-E demonstrate that electron-density conditioning provides meaningful guidance for 3D molecular generation.

### Weaknesses
Dependence on holo complexes: The conditioning ED maps are derived from known ligand–protein complexes, meaning the model is not applicable to apo pockets where no ligand is known. This limits its practical use in true de-novo design.

Pharmacophore labeling ambiguity: The pharmacophore features require ligand knowledge; the paper does not clarify how these labels could be inferred from ED alone during inference.

Arbitrary point-cloud ordering: Sorting ED points by xyz coordinates is heuristic and breaks rotational symmetry, which may hurt generalization.

Limited evaluation scope: Experiments are restricted to DUD-E; no validation on unseen protein families or cryo-EM-derived ED data.

Insufficient ablation and interpretability: The influence of pharmacophore labeling, sorting, and geometry tokens is under-analyzed; more visualization or ablation would strengthen claims.

### Questions
Inference scenario realism:
In practice, we often have only an apo pocket or predicted density map. How would EDMolGPT operate without a known holo-derived ED map? Could you approximate ED from the pocket atoms or use a learned ED predictor?

Pharmacophore derivation:
Since pharmacophore labels are computed from the known ligand during training, what is their source during inference? Are they predicted jointly, or assumed from pre-computed ED channels?

Permutation robustness:
How sensitive is the model to coordinate frame rotation or point-ordering perturbations? Would training with random permutations improve invariance?

Generalization to unseen systems:
Have you evaluated EDMolGPT on unseen protein families or on experimental cryo-EM densities to confirm cross-domain robustness?

Computational efficiency:
How does the autoregressive decoding speed and scaling behavior compare with diffusion-based SBDD models such as Pocket2Mol or TargetDiff?

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
3

### Summary
The paper conditions generation on the low-resolution electron-density point cloud of a known ligand and employs a decoder-only autoregressive framework to produce 3D molecules. While results on DUD-E show some advantages, reproducibility and evaluation fairness are problematic.

### Strengths
Using a low-resolution electron-density point cloud as the conditioning signal and a pure decoder autoregressive architecture to directly generate molecules with 3D conformations is a simple, efficient idea that is engineering-friendly and scalable.

### Weaknesses
1) Reproducibility is insufficient: no released code, model weights, data processing or evaluation scripts, and missing environment and random seed settings, and heavy reliance on commercial software makes the results difficult to reproduce.
2) The training set’s exact sources, versions, licenses, and cleaning/normalization procedures are not specified.
The method conditions on the reference ligand’s ED, which is essentially ligand-based generation, yet it is directly compared against SBDD baselines conditioned on protein pockets; this is not a fair comparison.
3) The ED is derived from the ligand rather than the protein/complex, making it difficult to substantiate claims about capturing pocket flexibility or avoiding rigid pocket assumptions; this reflects a conceptual mismatch.
4) The method conditions on the reference ligand’s ED, which is essentially ligand-based generation, yet it is directly compared against SBDD baselines conditioned on protein pockets, this is not a fair comparison.
5) Ablations are incomplete: there is no systematic study of the number of sampled points Np, coordinate quantization step σ, tolerances for relative geometry , or the effect of turning pharmacophore labels on/off.

### Questions
Code and model availability: You currently do not provide training/inference code, pretrained weights, evaluation scripts, or environment configuration, making it impossible to reproduce key results. Suggestion: release a complete pipeline in an anonymous repository (training, inference, evaluation), including Docker/conda environments, random seed and determinism settings, logs and hyperparameters; provide pretrained weights and a small set of example data.
Training data provenance and cleaning: The statement “~8M → ~2M after filtering” lacks detail on specific sources (e.g., ChEMBL/ZINC/PubChem), versions, download dates, licenses, deduplication, and standardization (salt stripping, stereochemistry/tautomer handling, normalization). Please document these aspects and provide the corresponding scripts and statistics.

### Soundness
2

### Presentation
2

### Contribution
2
