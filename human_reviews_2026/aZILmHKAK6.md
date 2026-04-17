# From Genomic Whispers to Therapeutics: Multi-Resolution Transcriptome-Guided Diffusion Models for Drug Design and Screening

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Traditional drug discovery is protracted and extremely expensive. While Structure-based Drug Design (SBDD) has advanced AI-driven molecular generation, target-centric models struggle with diseases arising from the dysregulation of complex physiological systems. To bridge this gap, we introduce Transcriptome-based Drug Design (TBDD): designing molecules from a cell’s transcriptomic response to perturbations. We present scTrans-Gen, a diffusion model that conditions generation on multi-resolution transcriptomic data (bulk and single-cell). Central to our approach is a transcriptome-centric condition extractor that aligns perturbation signals across domains into a function-oriented chemical space, avoiding the ill-posed reconstruction of microscopic structures from macroscopic signals. To exploit single-cell data, we propose a Gene Pseudoimage mechanism for robust high-resolution conditioning. Across diverse benchmarks, scTrans-Gen outperforms strong baselines on multiple metrics. We further demonstrate novel inhibitor design for specified gene knockouts and an efficient generate-then-search screening workflow suitable for time-sensitive clinical scenarios. Altogether, scTrans-Gen offers a practical route to function-oriented drug discovery and personalized precision medicine.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces scTrans gen, a diffusion model for drug design conditioned on transcriptomic data rather than protein structures. Given pre- and post-perturbation transcriptomes, the model generates molecules that induce that cellular response.
The model uses a "Transcriptome Pseudoimage" module to handle sparse single-cell data (converts gene expression to images processed by a vision encoder), a transcriptome interaction module to extract perturbation features, and a graph diffusion generator with multi-domain alignment to molecular fingerprints and graph VAE representations via two-stage training.
They evaluate on L1000 (bulk, ~20k drugs) and Tahoe-100M (single-cell, 300+ drugs). Results show large improvements over baselines on structural similarity (Fraggle Sim: 0.89 vs 0.33, Morgan Sim: 0.82 vs 0.11). They also demonstrate zero-shot gene inhibitor design and a screening workflow. The paper claims this avoids the "ill-posed" inverse mapping problem by learning a "function-oriented chemical space" as an intermediate representation.

### Strengths
* Substantial improvements over baselines (e.g., Fraggle Sim: 0.89 vs 0.33, Morgan Sim: 0.82 vs 0.11 for GexMolGen). Evaluation spans macro (validity/QED/diversity/FCD) and micro (Fraggle/Morgan/MACCS) plus a screening workflow.
* The model is complex, but there's good ablation studies

### Weaknesses
* Presentation issues:
	* Figure 4 is hard to read. Why is this not just another table?
	* Table 1, the red diff numbers are distracting and add no information
	* Table 4 is unmentioned in the text. Instead the text points to Table 6 for ablation?
* Their model is very complex. Multi stage training, complex loss functions, multiple pretrained parts. This limits reproducibility, and scalability, and makes modifications harder. I know this is not actionable feedback, but I still wonder if a simpler architecture using the same insights wouldn't have worked similiarly well (I've seen the ablations).
* I don't think that forcing the generation through a feature extraction bottleneck makes the problem any less ill-posed. It may make the model easier to evaluate, and maybe improves scores but it doesn't fix the fundamental issue?
* I would've liked to see an ablation of the Image Encoder. This component seem ad-hoc to me. Why is a vision encoder appropriate for gene expression patterns?

### Questions
* "de novo generation may be too slow for urgent clinical needs". How long does the de novo generation take? Even if it's in the few-minutes range, I find it hard to imagine a clinical scenario where that could possibly be the bottleneck?
* You align to a Mol Graph VAE and Morgan fingerprints (Fig. 3). Were those components pretrained on any molecules that appear in our eval splits? Is the extractor pretrained on a larger corpus or only on the train split here?
* TRIOMPHE has very bad Morgan Sim. Was this number taken from the prior paper or did the authors reproduce it themselves?
* What was SCimilarity trained on? Is there leakage with the test set?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces transcriptome-based drug design (TBDD), a “novel” paradigm for generating drug molecules conditioned on cellular transcriptomic responses to perturbations. The authors propose scTrans-Gen, a diffusion-based generative model that accepts both bulk and scRNA-seq data as conditioning signals. The core technical contribution is a multi-domain alignment architecture that bridges the gap between transcriptomic space (biological signals) and molecular space (chemical structures) through a function-centric condition extractor. For single-cell data, the authors introduce a transcriptome pseudoimage mechanism that converts sparse, noisy single-cell profiles into dense representations via pre-trained encoders. The model is trained in two stages: first aligning transcriptome features with a molecular graph VAE latent space, then aligning with Morgan fingerprint space. The paper also proposes a generate-then-search screening workflow for identifying similar compounds in existing drug libraries. Experiments on L1000 and Tahoe-100M datasets demonstrate improvements over baselines across multiple metrics. The authors validate their approach through zero-shot gene inhibitor design tasks.

### Strengths
1.	The paper presents the first conditional molecular generation model that successfully incorporates single-cell transcriptomic data, addressing a significant technical challenge with sparse and noisy data through the creative Transcriptome Pseudoimage mechanism.
2.	The authors establish a rigorous evaluation suite spanning macroscopic metrics (coverage, diversity, validity), microscopic structural similarity measures (Fraggle, Morgan, MACCS), and functional assessments (PRnet MSE), providing multiple perspectives on model performance.
3.	The unified framework handles both bulk and single-cell data without requiring separate architectures, demonstrating architectural flexibility and potentially broader applicability across different data modalities.
4.	The paper includes detailed ablations validating the contribution of each architectural component, particularly demonstrating the necessity of the dual-domain alignment mechanism for single-cell data.
5.	The gene inhibitor prediction experiment tests model performance on truly held-out targets, providing evidence for generalization beyond the training distribution.

### Weaknesses
1.	The extremely high structural similarity scores (>0.80 across multiple metrics) between generated and target molecules suggest the model may be reconstructing training examples rather than performing true de novo design. The paper lacks analysis of structural diversity between generated molecules and their training set counterparts, making it questionable to distinguish memorization from functional equivalence.
2.	The drug screening evaluation is potentially redundant since it uses high-quality generated molecules as queries to retrieve similar compounds. The high hit rates primarily validate that generation worked rather than demonstrating unique value of the screening component. Additionally, using PRnet predictions to evaluate functional similarity creates circularity without experimental ground truth, with no information about what data it was trained on.
3.	The unseen drug split evaluates the model on generating held-out drugs from their transcriptomic signatures, but this formulation is problematic because multiple structurally diverse molecules can produce similar transcriptomic effects. The high reconstruction accuracy suggests the model learned drug-specific mappings rather than perturbation-to-function relationships. Besides, common cancer cell line panels have heavy redundancy, while in the hold-out cell split experiment, the paper didn’t discuss if same cancer type were involved in both training and testing.
4.	The baselines show surprisingly poor performance (Morgan similarity around 0.10-0.15 versus 0.82 for the proposed method; wheras, e.g. Gx2Mol reported 0.83 uniqueness and 0.95 QED in their paper), raising concerns about implementation quality, hyperparameter tuning, or whether the evaluation framework inherently favors the proposed architecture. This massive gap undermines confidence in the comparative analysis.
5.	The paper provides no wet-lab validation of generated molecules, no binding affinity measurements, no cellular assay results, and no synthesis feasibility analysis. All functional claims rest on computational predictions, which is insufficient for a drug design paper making strong therapeutic claims. I understand that this could be majorly tied by the lab’s resources, however, since major evaluations were on generated molecules, it could still raise a concern.
6.	While the paper claims function-oriented design addresses limitations of structure-based approaches for systemic diseases, no experiments demonstrate superior performance in multi-pathway disease scenarios. The gene inhibitor task tests single-target effects, not the claimed advantage for complex systemic perturbations, if that’s what the authors claimed.
7.	The pseudoimage construction mechanism is the paper's primary novelty for single-cell data, yet critical parameters lack justification: k equals 15 cells is never compared against alternatives, cell cycle-based sampling is not compared to random or uniform sampling, and the total number of cells N is never specified. The choice to encode gene expression data using a pretrained Stable Diffusion VAE (trained on natural images) is counterintuitive and unexplored, with no comparison to training from scratch or simpler encoders. Multiple loss function weights (alpha equals 0.4, lambda equals 0.15, unspecified lambda subscript KL) appear arbitrary with no sensitivity analysis, raising concerns about test-set tuning or cherry-picking results.

### Questions
1.	Can the authors provide analysis of structural diversity between generated molecules and their nearest neighbors in the training set? What is the distribution of Tanimoto similarities between generated molecules and all training molecules? For cases where the model generates molecules with high structural similarity (>0.80) to training drugs, can the authors demonstrate that these represent functional equivalence rather than memorization? 
2.	What data was PRnet trained on? If PRnet was trained on the same L1000 or Tahoe-100M datasets used to train scTrans-Gen, this creates severe circular reasoning where both the generator and evaluator learn from the same distribution. The authors must clarify PRnet's training data source or provide alternative functional validation methods. 
3.	What are the specific cell lines and their cancer type annotations in train versus test splits? What proportion of test cell lines come from cancer types already represented in training (e.g., multiple breast cancer lines in both sets)? Can the authors provide performance stratified by whether test cells come from seen versus truly novel cancer types, and ideally demonstrate generalization to completely held-out cancer types? 
4.	How should the unseen drug split be interpreted when multiple structurally diverse molecules can produce similar transcriptomic effects? Should the evaluation focus on functional equivalence (any molecule producing the desired perturbation) rather than structural reconstruction of the specific held-out drug? The current high structural similarity (>0.77) to exact held-out drugs suggests drug identification rather than function-based design. 
5.	Given that baselines report much higher performance in their original papers (e.g., Gx2Mol: 0.83 uniqueness, 0.95 QED) compared to this evaluation (0.10 uniqueness, 0.60 QED), can the authors provide evidence that baselines were properly implemented with appropriate hyperparameter tuning? Were official implementations used or were methods reimplemented? 
6.	What is the sensitivity to pseudoimage parameters (k equals 15, cell cycle sampling, total N cells, aggregation strategy)? Why is a Stable Diffusion VAE trained on natural images appropriate for gene expression data compared to alternatives (trained-from-scratch VAE, simple CNN, direct MLP)? How were loss function weights (alpha, lambda, lambda subscript KL) determined, and what is the performance sensitivity to these choices? 
7.	The paper claims function-oriented design addresses limitations of structure-based approaches for systemic, multi-pathway diseases, but experiments only test single-target gene inhibitors. Can the authors provide experiments demonstrating superior performance for complex disease scenarios involving multiple dysregulated pathways, or clarify that this claimed advantage remains theoretical?
8.	Single-cell results show substantially lower structural similarity (Morgan: 0.6114) compared to bulk data (Morgan: 0.8228), despite being the paper's primary novelty. Why does performance degrade on single-cell data? Additionally, the paper states baselines lack methods for single-cell data, but couldn't the authors adapt baselines (e.g., aggregate single-cell to pseudo-bulk) to provide any comparison, or at minimum demonstrate that baselines completely fail on this data? 
9.	All reported diversity metrics measure variation across different conditions. Can the authors show that the model generates structurally diverse molecules when sampling multiple times from the SAME perturbation condition? This would demonstrate the model learns one-to-many mappings (one perturbation to multiple valid drugs) rather than deterministic one-to-one mappings, which is critical for de novo design claims. 
10.	The paper reports no error bars, confidence intervals, or results from multiple training runs. Are the performance differences statistically significant? What is the variance across different random seeds? Given the dramatic improvements claimed, statistical validation is essential to rule out fortunate initialization or dataset-specific overfitting.
11.	The zero-shot gene inhibitor experiment uses 10 selected genes with highly variable inhibitor counts (1,200 to 23,000 per gene). Which specific genes were selected and why? How were baselines evaluated on this zero-shot task if they were never trained on gene knockout data? The evaluation protocol needs clarification to ensure fair comparison.

### Soundness
2

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
4

### Summary
This paper introduces a generative framework, scTrans-Gen,  that could design molecules conditioned on their cellular responses where they use gene expressions to guide with the molecular generation. scTrans-Gen, addresses two tasks: Transcriptome-based Drug Design, and Transcriptome-based Drug Screening. Finally, they test their framework on two datasets: L1000, and Tahoe where it outperforms prior works on structural similarity, and MSE.

### Strengths
- The problem addressed in this work is novel and important.
- utilizing the transcriptomic effect of drugs instead of structural similarities is a novel way of looking at this problem and more powerful for drug discovery. 
- The paper has a comprehensive experiments and ablation studies.

### Weaknesses
- My main issue is that the architecture is very complex. The framework combines different architectures such as graph diffusion, LLM encoder, VAEs, etc without enough justification on such choices. 
- While using the transcriptomes is interesting and useful for this task, it has limited biological interpretability. 
- Since the authors are promoting this framework for drug discovery, other qualities such as toxicity or scaffold novelty needs to be explored as well for it to be useful in this domain. 
- The related work section is limited given the architectural choices.

### Questions
- Could you explain the experimental setting in more detail?
- Is it possible to explain how the generated molecules related to the transcriptomic change produce such perturbation effect?
- Could the authors clarify if the transcriptomic change is mainly modeling the effect of the drug or the cell line? 
- Is it correct to assume that the intermediate function oriented chemical space is only capturing the statistical correlation between the two modalities (molecular and cellular)?

### Soundness
2

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
4

### Summary
Recent works in Structure-Based Drug Design (SBDD) advance but remain brittle to complex disease types and molecular compositions. To this end, the paper approaches drug design from the perspective of Transcriptomic representations. The paper presents scTrans-Gen, a diffusion model framework for conditionally generating molecules using multi-resolution transcriptomic data. scTrans-Gen makes use of a feature extraction module that extracts representations coresponding to the perturbed and unperturbed transcriptome, a transcriptome pseudoimage block for processing single cell representations and a transcriptome interaction block for modelling relationships between pre and post perturbation samples. scTrans-Gen is trained using classifier-free guided diffusion over Graph Transformers wherein alignment is carried out w.r.t functional features and morgan fingerprint using variational inference and contrastive learning respectively. Authors present improved performance on benchmarks compared to architecture baselines.

### Strengths
* The paper is well-written and motivated.
* The paper presents an innovative and novel approach for post-training generation.

### Weaknesses
* **Empirical Comparisons:** My main concern is the empirical comparisons drawn and baselines considered in the paper. The paper currently limits baselines to GexMolGen, TRIOMPHE and Gx2Mol as baselines. Note that none of these methods were explicitly trained using a combination of either of the losses utilized in scTrans-Gen. That is, the paper does not consider an explicit contrastive learning or diffusion baseline. Furthermore, while all baselines are VAE variants, scTrans-Gen adopts additional training as well as architecture augmentations. This puts the baselines out of perspective of the proposed framework. Authors could consider using relevant baselines such as DiffDock [1] and Boltz variants [2]. This would also allow for the comparison of drug screener on established architectures as well as benchmarks. Furthermore, authors should consider additional cross-modal baselines using alternative objectives or inductive biases [3].

* **Generation and Search:** The paper presents the drug screening strategy wherein generated samples are used as queries to screen compound libraries. However, it remain unclear on why this design decision is executed. Why bootstrap the search from learned biases of the model? Assuming scTrans-Gen can be further aligned on a given library, why not conduct few-shot alignment / feature regression over the library? Authors mention that the approach is best suited for molecules with similar structure. Could the authors elucidate on the structural similarity between a few compounds and generated samples (eg- tanimoto similarity, laplacian similarity, etc.)? Furthermore, it remains unclear as to how effective the drug screener becomes as with increasing number of hits, we observe lower similarity scores for a larger top-k sample size.

* **Role of Feature Extractor:** It is unclear as to what role the feature exrtractor plays in the final training scheme. scTrans-Gen is trained using classifier-free guidance which already reduces the contribution of the embedding. Additionally, training phase implements random feature dropping wherein guidance is randomly dropped with probability p. Ideally, the role of conditioning is to provide more information during the learning phase to the base model. Authors provide ablations to support this argument. However, algorithm design indicates that a subset of molecules suffer from additional conditional information.

[1]. Corso et al, DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking, ICLR 2023.  
[2]. Wohlwend et al, Boltz-1 Democratizing Biomolecular Interaction Modeling, https://doi.org/10.1101/2024.11.19.624167.  
[3]. Bendidi et al, A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features, ICML 2025.

### Questions
Refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
