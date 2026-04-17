# ProtoBind-Diff: Protein-Conditioned Discrete Diffusion for Structure-Free Ligand Generation

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Designing small molecules that selectively bind to protein targets remains a central challenge in drug discovery. While recent generative models leverage 3D structural data to guide ligand generation, their applicability is limited by the sparsity and bias of experimentally determined complexes. Here, we introduce ProtoBind-Diff, a structure-free masked diffusion model that conditions molecular generation directly on protein sequences via pre-trained language model embeddings. Trained on over one million active protein-ligand pairs from BindingDB, ProtoBind-Diff generates chemically valid, novel, and target-specific ligands without requiring 3D structures for inference. In extensive benchmarking against 3D structure-based models, ProtoBind-Diff achieves competitive predicted binding affinity scores and performs well on challenging targets, including those with limited training data. Despite never being trained on the data that contain binding pockets, its attention maps align with contact residues, suggesting the model learns spatially meaningful interaction priors from sequence alone. These results demonstrate that sequence-conditioned diffusion can enable structure-free, scalable ligand discovery across the proteome, including orphan or rapidly emerging targets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Authors propose a new masked diffusion language model to generate molecules with sequence-based representations of proteins and ligands.

### Strengths
Paper is well-written and easy to follow.
Comparisons are good. I appreciate the inclusion of very recent methods such as TamGen.

### Weaknesses
Doesn’t seem to be doing better than existing methods. ProtoBind only seems to do better on diversity and is on par with or worse than other methods on all other metrics, including vina docking score. Could the authors include error bars to show statistical significance of results?

Masked diffusion language modeling already exists. What is the methodological novelty in this method? Why limit to sequence data when structure data already exists?

### Questions
see weaknesses, above

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
4

### Summary
This paper introduces ProtoBind-Diff, a masked discrete diffusion model for structure-free ligand generation using protein sequence embeddings from the pretrained language model ESM-2. Unlike traditional methods relying on 3D structures, ProtoBind-Diff generates drug-like molecules for diverse protein targets, including those without known structures. Trained on over one million protein-ligand pairs from BindingDB, it outperforms or matches state-of-the-art 3D-structure-based and sequence-based models, demonstrating competitive binding affinity (notably with Boltz-1/2 predictors) and biologically interpretable attention patterns aligned with true binding residues.

### Strengths
* This work broadens applicability by generating ligands conditioned on protein sequences, eliminating the need for 3D structural data and enabling its use for targets with little to no structural information.
* This paper is well-written, with a clear presentation of both the motivation and the methodology.
* This paper provides a rich benchmark, evaluating 12 diverse targets from easy to hard classes and systematically compares against 3D generative models.
* The quantitative analysis of attention head distributions and their correlation with key protein binding sites is insightful.

### Weaknesses
* The architectural innovation of the model is quite limited, as it employs a standard cross-attention mechanism for condition injection, with the only distinction being the use of protein sequences as input.
* Although the idea for conditioning ligand generation on pure protein sequences is feasible, the authors need to further elaborate on why sequence-based conditioning offers advantages over 3D structure-based conditioning, especially in cases where real or predicted 3D structures are available.
* The paper lacks an in-depth discussion and experiments regarding the model architecture. For instance, it remains unclear how ProtoBind-Diff performs with varying parameter scales, different types of protein encoders, and alternative condition injection methods.
* The performance improvement of ProtoBind-Diff over the baseline models is minimal.

### Questions
* Could the authors provide a comparison with alternative condition injection methods? For example: 1) Pooling the protein sequence and injecting it using AdaLN; 2) Treating the protein sequence as the context for the ligand sequence and injecting by sharing q, k, v (similar to MM-DiT); 3) A hybrid approach combining both methods.
* Please provide a wider range of parameter scale combinations to validate the model's scalability.
* Please include ablations with a broader range of protein encoders, such as testing more advanced models like ESM-C to evaluate if they yield better results.
* One of my major concerns is whether replacing the protein encoder in ProtoBind-Diff with a structure-aware encoder of similar scale, such as SAProt-650M, while keeping the rest of the architecture unchanged, would result in significant performance differences. If the performance remains similar or is lower, it could support the authors' claim in the introduction regarding the limitations of using 3D structural inputs.
* Please include a comparison with baseline models in terms of parameter scale and inference speed.
* Please revisit the necessity of introducing the sequence-based condition in light of W.2 and Q.4.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel ligand generation model named ProtoBind-Diff, which uses the protein's amino acid sequence as a condition to guide molecule generation. The model is based on a masked discrete diffusion framework and is trained on the large-scale BindingDB dataset. Experiments demonstrate that ProtoBind-Diff generates ligands with chemical properties and binding affinity closely matching real active molecules.

### Strengths
- This paper proposes a "structure-free" paradigm for target-aware ligand generation
- By breaking free from 3D structure constraints, the model can be trained on BindingDB, which vastly exceeds the scale of traditional structural datasets.
- This paper conducts interpretability analysis, showing the models' capability to capture key binding residues even when not trained on 3D structures.

### Weaknesses
- The model does not resolve the same limitations it critiques in SBDD. The authors claim that SBDD models often ignore conformational flexibility and induced-fit effects, suffer from limited structural data, and may produce poor results for diversity and other key properties. However, the paper fails to demonstrate how the proposed sequence-based model addresses these same challenges. 
- Poor conceptual and technical novelty. It doesn't discuss why sequence-based design would be preferable to structure-based design if structural data were available. The authors do not provide evidence of using sequence over structure, nor do they sufficiently argue against the strong and obvious alternative of using high-accuracy structure prediction tools (e.g., AlphaFold) or pocket prediction tools to generate inputs for established SBDD models. The core components, using a discrete diffusion model for molecular generation and a cross-attention mechanism to fuse conditioning information, are both well-established techniques.
- The evaluation framework is insufficient and potentially unreasonable. (As detailed in questions below).
- The paper lacks essential visualization results and case studies.

### Questions
- As for the potential advantage of this work in considering conformation flexibility and induced-fit effects, I recommend following evaluation works on related works (e.g., [1]). And it is recommended to discuss more about related works like this. [1] is a generative SBDD method that explicitly considers protein flexibility.
- During evaluation, how is the number of molecule atoms determined for baseline models? Is there any bias?
- Regarding the clustering method: Is it based on scaffold novelty? Or clustering based on other features?
- How the author ensure that the test set does not overlap with the training set? Was a target (sequence) similarity analysis performed?
- "Lower MMD values indicate greater similarity to the BindingDB reference set and thus better generation quality." How to explain this similarity regarding the goal of diversity?
- The author admits that validity cannot be optimized simultaneously with other molecular properties. But compared with SBDD models, where molecules are designed together with sequence and structure, this work only designs molecular sequence, yet still performs poorly in terms of validity. Could the authors explain the reason? The bad validity may indicate that the model does not learn the basic chemical rules.
- In Table 6, why are only MMD values reported instead of the real values/scores for the properties?
- Why is a low MMD (being closer to the distribution of known actives) considered better? For drug discovery, the goal is often to find novel candidates that surpass existing molecules (e.g., stronger competitive inhibitors or activators), not just reproduce the existing distribution.
- Have you tried fine-tuning the generative model by reinforcement learning [2] or preference optimization [3,4]? More discussion about this might further improve this submission.
- "All generative models performed well on targets where docking effectively distinguished active from inactive compounds, for example, ESR1, GRIK1, and CCR9." How is "effectively distinguished" formally defined? Is it defined by the Enrichment Factor?
- I understand the concern that Vina docking may have a high False Positive rate, as indicated by EF study in this paper, so I recommend the author to run other traditional evaluations for binding affinity, such as MMGBSA or MMPBSA. 
- The attention-based analysis does not prove that the generated ligands actually form key interactions. Could the authors use PLIP [5] analysis to confirm if Non-Covalent Interactions were formed in accordance with the key residues and motifs?

**Note: I am willing to change my rating according to the rebuttal discussion and the paper revision.**

References:

[1] Integrating Protein Dynamics into Structure-Based Drug Design via Full-Atom Stochastic Flows. ICLR 2025

[2] Stabilizing policy gradients for stochastic differential equations via consistency with perturbation process, ICML 2024

[3] Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization, NeurIPS 2024

[4] Decomposed Direct Preference Optimization for Structure-Based Drug Design, TMLR 2025

[5] PLIP: fully automated protein–ligand interaction profiler. Nucleic acids research, Nucleic Acids Research 2015

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ProtoBind-Diff, a structure-free masked discrete diffusion model that conditions ligand generation on protein sequence embeddings (ESM-2) via cross-attention. Trained on >1M protein–ligand activity pairs, the method aims to avoid dependencies on 3D pocket structures. The authors report: (i) molecular property distributions close to actives, (ii) favorable enrichment under Boltz-1 on a 12-target benchmark, and (iii) attention maps aligning with binding-site residues, suggesting biologically reasonable priors learned from sequence alone. Pocket-free generation is positioned as a scalable pathway for orphan/rapidly emerging targets where 3D structures are unavailable.

### Strengths
1. **Clear problem motivation & scope**: Removing the dependency on 3D pocket structures addresses a real deployment bottleneck (limited and biased complex structures in PDBbind compared to activity-centric resources like BindingDB).
2. **Competitive performance vs 3D baselines**: Empirical comparisons against Pocket2Mol and PocketFlow (flow-matching–based SBDD) support the claim that sequence-conditioned diffusion can rival pocket-conditioned methods on multiple targets.

### Weaknesses
1. **Heavy reliance on proxy metrics; risk of bias**
   - The core claims depend on Boltz-1/2 and Vina. While Boltz-1/2 are strong modern predictors, their training distribution and the overlap with commonly used ligand/target corpora can confound absolute gains. Please (a) report distributional overlap checks (ligand scaffolds and protein family homology) against your training sources; (b) complement with leak-controlled external tests such as LP-PDBbind or other curated splits explicitly designed to reduce leakage.
2. **Comparisons to target-aware sequence models need tighter controls**
   - TamGen is a natural, strong comparison for target-aware chemical LMs. Please ensure identical novelty filters, deduplication, sampling budgets, and stratified reporting on “easy vs. low-data” targets; add statistical tests (e.g., stratified bootstrap CIs) on key metrics. 
3. **Validity/uniqueness vs. quality trade-offs**
   - Provide sampling curves vs. step count, mask rate, and nucleus-p, reporting Validity, Uniqueness, Novelty, Property MMD, and Boltz-1 to show practical operating points.
4. **Compute & scalability reporting**
   - Since the paper’s value proposition includes scalability without 3D pockets, please add throughput/latency/memory comparisons to Pocket2Mol/PocketFlow/TamGen at equal batch sizes and similar sampling budgets.

### Questions
1. **Leak-control protocol**: How do you guard against scaffold-level and protein-family leakage between BindingDB training and your 12-target test set? Please include a similarity matrix and thresholding policy, then recompute headline metrics after removing near-neighbors.
2. **Low-data targets**: Which components (ESM-2 conditioning vs. resampling vs. augmentation) drive robustness on sparse targets? Please include ablation tables.
3. **Generalization beyond BindingDB**: Any early results on curated, leak-controlled external testbeds (e.g., LP-PDBbind) or other activity datasets to support broader claims?
4. **Contextualization of baselines**: Briefly summarize Pocket2Mol (3D pocket-conditioned autoregressive assembly) and PocketFlow (flow-matching with interaction priors) to help readers who are less familiar with SBDD baselines appreciate where sequence-only diffusion stands.

### Soundness
2

### Presentation
2

### Contribution
2
