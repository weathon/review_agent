# CellxPert: An Efficient Reasoning Language Model for Single-Cell and Spatial Multi-Omics

- Decision: Reject
- Scores: 8, 6, 2, 2

## Abstract
In this work, we introduce CellxPert, a scalable multimodal foundation model that unifies single-cell and spatial multi-omics within a common representation space. CellxPert jointly encodes transcriptomic (scRNA-seq), chromatin-accessibility (ATAC-seq), and surface-proteomic (CITE-seq) measurements, while directly incorporating MERFISH and imaging mass-cytometry data as 2D or 3D spatial–visual layers. CellxPert facilitates four key downstream tasks out of the box: (i) cell‑type annotation across a broad ontology of 154 largely overlapping identities—the largest label space addressed to date and a stringent test of fine‑grained discrimination, (ii) efficient fine-tuning using Low Rank Adaptation (LoRA), (iii) genome-wide transcriptomic response prediction to in silico perturbations (ISP), and (iv) seamless multi-omic integration across various assays and platforms. Unlike current single-cell foundation models, which approximate gene perturbations by deleting or reordering tokenized gene expression ranks, CellxPert employs a Metropolis–Hastings sampler whose proposal kernel uses the model’s masked conditional distributions to transition to new transcriptomic states conditioned on the perturbed genes. This Markov‑chain procedure mitigates out‑of‑distribution artifacts introduced by abrupt token manipulation and produces trajectories that are biologically interpretable. Evaluations on PBMC68K, Replogle Perturb-seq, Systema and BMMC benchmarks show CellxPert outperforming classical and state-of-the-art baselines in cell-type annotation, perturbation-aware reasoning, and multi-omic integration by a significant margin.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript introduces CELLXPERT, a large multimodal foundation model designed for single-cell and spatial multi-omics data. The model unifies diverse assay types (scRNA-seq, scATAC-seq, CITE-seq, MERFISH, imaging mass cytometry) into a shared latent representation, supports cell-type annotation across 154 identities, and enables downstream fine-tuning through low-rank adapters. A central contribution is an in silico perturbation framework based on blockwise Metropolis–Hastings sampling, which iteratively proposes and accepts expression updates so that predicted perturbation responses remain biologically plausible and close to the data manifold. Architecturally, the model incorporates mixture-of-experts layers, expanded token context, and spatial neighborhood encoding. The method is evaluated on perturbation benchmarks, multimodal integration tasks, spatial imaging data, and disease fine-tuning scenarios, where it demonstrates improved performance compared to existing single-cell foundation models such as Geneformer, scGPT, and CellPLM. Overall, the manuscript aims to provide a unified, biologically grounded approach to reasoning over complex cellular states.

### Strengths
A notable strength of this work is its conceptual unification of disparate single-cell and spatial modalities into a single latent framework, which is extremely relevant given the rapidly diversifying assay landscape. The perturbation modeling approach is both theoretically motivated and practically useful: by leveraging iterative Metropolis–Hastings masking rather than single-pass imputation, the method avoids off-manifold drift and preserves gene–gene relationships, an issue that has limited existing perturbation simulators. The architectural choices—including mixture-of-experts and long-context reasoning—appear well justified by the hierarchical structure of cellular data. Empirically, the method shows consistent improvements across perturbation benchmarks, multimodal integration, and cell-type classification, and the manuscript provides ablations demonstrating that several architectural pieces contribute meaningfully to performance. The integration of LoRA adapters is also practical, offering domain customization for real disease datasets without catastrophic forgetting. Finally, the spatial results on MERFISH/IMC add evidence that the model generalizes beyond purely transcriptomic inputs, which strengthens its claims of multimodal reasoning.

### Weaknesses
While the perturbation sampling strategy is a highlight of the paper, its computational cost relative to single-pass inference is not fully characterized; iterative MCMC steps may limit scalability on large studies or high-target gene screens. The manuscript would also benefit from more analysis of robustness across training seeds, especially for perturbation outcomes, since stochastic sampling procedures can introduce variance. Although the model claims broad multimodal generality, most biological interpretation is still shown in hematopoietic or immune contexts, leaving open how well it handles highly heterogeneous solid tissues. The claims regarding improved biological realism of perturbation responses rely primarily on embedding-space metrics; direct comparison to differential expression profiles or pathway-level ground truth would strengthen the validation. Additionally, while the energy-based interpretation is discussed, the manuscript stops short of providing formal properties such as convergence guarantees or mixing behavior, which would help justify the correctness of the sampler. Finally, although LoRA fine-tuning is appealing, potential interactions between adapters (e.g., composition across tasks) are not explored, which may be relevant for translational workflows.

### Questions
1. How sensitive are MH-based perturbation results to initialization, random seeds, or the number of sampling iterations? Can the authors quantify variance?
2. What is the computational overhead of blockwise MH compared to single-pass masked modeling? Are there practical limits on dataset size or number of perturbation targets?
3. How do the authors detect or guarantee adequate mixing of the sampling chain? Is there a termination criterion beyond a fixed step budget?
4. In settings where the target gene is expressed in very rare subpopulations, can the sampler capture niche perturbation responses, or does averaging wash them out?
5. How stable are attention/importance patterns across different model initializations or LoRA adapter configurations?
6. Have the authors evaluated CELLXPERT on tissues with complex spatial gradients (e.g., tumors, brain) where context is not purely discrete?
7. Can LoRA adapters trained on different diseases be composed, or do they interfere with one another when stacked?
8. For well-studied perturbations with known pathway targets, can the authors demonstrate pathway-level expression shifts rather than only embedding-space movement?

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
The authors of this paper introduce CELLXPERT, a large-scale multimodal foundation model designed to unify single-cell and spatial omics within a shared latent representation. The model integrates scRNA-seq, ATAC-seq, and CITE-seq data, while incorporating spatial omics. CELLXPERT employs an encoder–decoder MoE architecture with a Metropolis–Hastings (MH) sampler for in-silico perturbation (ISP) reasoning, which mitigates out-of-distribution shifts resulting from naïve token deletions or reorderings. This method interprets the masked language model as an implicit energy-based model and samples transcriptomic states iteratively to produce biologically interpretable perturbation trajectories.
Empirical evaluations on multiple benchmarks and spatial datasets show that CELLXPERT outperforms previous single-cell foundation models in perturbation response prediction, cell-type annotation, and multi-omic integration.

### Strengths
- The MCMC-based ISP modeling is novel in the field of single-cell foundation models, providing a probabilistic framework that corrects limitations of token-level perturbation approximations in prior models.

- CELLXPERT effectively handles transcriptomic, chromatin, proteomic, and spatial modalities in a unified architecture.

- Across diverse benchmarks, the model achieves superior accuracy and F1 scores, especially on complex multimodal integration tasks.

### Weaknesses
- A key limitation is the absence of comparisons to traditional statistical or linear models that have proven highly competitive in recent benchmarking efforts. Prior studies (e.g., Ahlmann-Eltze et al., 2025) demonstrated that simple linear or regression-based models can match or even outperform recent single-cell foundation models on perturbation response prediction tasks under rigorous evaluation metrics.

- Several recent works (e.g., GEARS, MultiVI-GNN) incorporate known gene–gene or spatial relationships using graph neural networks and have shown strong performance in similar perturbation or integration tasks. These methods are only briefly mentioned, leaving unclear how CELLXPERT performs against models that explicitly encode biological priors.

### Questions
- Did you evaluate CELLXPERT against such non-deep-learning baselines (e.g., ridge regression, linear mixed models, PCA- or kNN-based predictors)?

- The paper mentions joint modeling of RNA, ATAC, CITE, and spatial data. Did the authors try to use specialized pretrained encoders for these modalities?

- Graph-aware models like GEARS or MultiVI-GNN explicitly encode gene–gene interactions. Could CELLXPERT benefit from incorporating such priors?

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
3

### Summary
CELLXPERT is a multimodal transformer for single-cell and spatial omics with an MCMC inference scheme for realistic perturbation simulation. It achieves strong results on multiple benchmarks, though its main contribution lies in inference rather than backbone innovation.

### Strengths
- The inference approach for **perturbation simulation** is a creative and well-motivated idea that directly targets known limitations of prior token-based perturbation schemes.

- The paper extends **multimodal modeling** to include spatial omics, broadening the scope beyond RNA-only foundation models.

- **Empirical results** are strong, and the ablations on modality fusion are informative.

### Weaknesses
- The architectural novelty is limited to engineering refinements, and so similar to previous works. The conceptual novelty lies instead in the proposed perturbation simulation method, which operates at the inference stage and is largely independent of the backbone design.  

- As a foundation model, CELLXPERT is expected to demonstrate broad generalization across datasets and modalities. Many standard public datasets used in prior works such as scGPT are not covered in this paper, limiting the assessment of generalization.

- The main novelty (the MCMC inference) remains under-evaluated. Its impact is not isolated from the backbone in Table 3, and comparisons to simpler inference methods are missing. This perturbation simulation method should be applied to embeddings from other foundation models and compared against previously used perturbation approaches under identical setups.  

- Several figures lack clarity and proper referencing. ّn Figure 1, the legend text is too small, axis labels are missing, and the subfigures do not show consistent point distributions as implied by the caption, which suggests identical embeddings colored by different labels. Figures 1, 2, 6, and 10 are not referenced or discussed in the main text, leaving their interpretive role unclear.  

- No public code or model is provided, which limits verifiability for a foundation model.

Justification: The paper makes interesting steps toward multimodal and perturbation-aware single-cell modeling, but the claims are not sufficiently supported. The foundation backbone adds little architectural (engineering) novelty and does not cover many datasets used in previous works, making it hard to judge generalization. The proposed perturbation simulation approach is novel but not evaluated independently of the backbone. Addressing these points would require new experiments and goes beyond minor revisions, though the direction is promising for future work.

### Questions
1. Can you apply the MCMC-ISP wrapper to prior encoders (scGPT, Geneformer) and/or run token-editing/one-shot inference on CELLXPERT to isolate contributions?

2. Report MCMC convergence diagnostics: acceptance-rate curves, stabilization trends, per-chain variance, iteration/compute budgets.

3. Clarify anchor construction, train-split restriction (to avoid leakage), number of medoids (K), and sensitivity to distance choice (Wasserstein vs. cosine/L2).

### Soundness
3

### Presentation
1

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
The paper introduces CellxPert, a new single-cell foundation model that uses a Transformer encoder-decoder trained with a masked-language modeling objective and sparsely gated Mixture-of-Experts feed-forward layers. The model allows for incorporating multi-omic measurements, where each feature is assigned an identity embedding and a discretized bin embedding based on its magnitude of activity in a given cell. Additionally, the model accommodates spatial transcriptomics-based positional encoding, when such data is available. A key distinction from existing single-cell foundation models is their approach to perturbation modeling, where, rather than performing one-shot imputations, they adopt a Metropolis-Hastings MCMC approach. This approach repeatedly masks a small block of (non-directly-perturbed) genes and proposes updates to impute them based on the encoder's conditional distributions and accepts proposals with probability proportional to how closely they move the cell profile to perturbed-cell anchors. 

While there are several interesting ideas in the proposed model that could potentially advance the state of the single-cell foundation models, the evaluation rigor is lacking, leaving it unclear whether these ideas actually provide any advances (thus my current rating on Soudness), and certain key modeling choices appear underpowered.

### Strengths
**1.** The submission proposes a single-cell foundation model that incorporates multi-omic data. This is a biologically well-motivated and timely problem with very little development, thus, has a promising direction.  
**2.** The identity-plus-binned-magnitude scheme (with permutation training for order invariance) is biologically sensible. They also have practical engineering choices with, e.g., sparsely-gated MoE layers.  
**3.** The authors rightly identify some of the problems with the approaches current single-cell foundation models take to simulate perturbations (particularly the one-shot imputation approaches and OOD issues) and propose a new approach to improve upon these.    
**4.** There is a range of datasets used for experimental benchmarking, involving various types of -omics data, clinical datasets with patient-hold-out experiments, and genetic perturbations.  
**5.** Introduction and Related Works sections are well-written and motivated.  
**6.** The method is generally clearly described although some experimental details may be lacking.

### Weaknesses
**1.** The largest concern I have is the lack of appropriate benchmarking that clearly establishes whether the proposed model is a stronger foundation model than what already exists in the single cell literature (at least for transcriptomics). There are only two experiments where other single-cell foundation models (scFM) are used as baselines, and both have “confounding factors” for interpretation:  
&nbsp;&nbsp;&nbsp;&nbsp; **(i)** Genetic Perturbation Clustering Task: Since the submission introduces both a new FM architecture *and* a new perturbation simulation approach (Metropolis-Hastings scheme), it is unclear which is responsible for the performance difference in the genetic perturbation tasks since such ablations are missing.  
&nbsp;&nbsp;&nbsp;&nbsp; **(ii)** Multi-omic Cell-Type Annotation Task: In the multi-omic case, the scFM baselines are transcriptomics-only models. So, I believe the authors compare the performance of their model when using multi-omic data vs the performance of the scFM models when they only use transcriptomics data. Unless I misinterpret the writing, there are no results on CellxPert classifying BMMC cells with transcriptomics only. This makes it unclear if the performance gain is from the multi-omic data or the model architecture. A typical benchmarking experiment used by existing foundation models is to test cell type annotation performance under the same conditions (e.g. with transcriptomics, like in Section 4 paragraph 2). I don’t believe we, as a field, have figured out the “right benchmarks” yet, but I'd expect to see such comparison.  

**2.** Most tasks involve classification but there is no linear classifier baseline. This is important because one of the main criticisms in the field is that the current foundation models fail to robust outperform linear models in most tasks *(e.g. Systema paper by Vinas Torne et al, Nature Biotechnology 2025; Boiarsky et al Nature Machine Intelligence 2024 etc)*. Such a baseline should always be included in ML papers. 

**3.** On a similar note, it is not clear in the spatial transcriptomic + proteomic experiments whether the incorporation of spatial coordinates as positional encodings actually provides anything useful to the model, e.g. I would expect to see a comparison on cell type annotation when these coordinates are completely ignored.

**4.** While I excitedly appreciate the effort to move towards incorporating multi-omic data, the current design appears underpowered. The model enforces a fixed per-cell token budget shared across all modalities. A typical single-cell ATAC-seq data involves tens of thousands peaks and often resolves cell states that transcriptomics does not *(e.g. Jindal et al Nature Biotechnology 2023)*, but the current modeling approach will discard most of the features that would explain a large portion of the variability. In its current state, the multi-omic fusion feels naïve. Additionally, there is no principled token-allocation policy across modalities so it is unclear how a user would go about determining how many tokens to allocate to one modality vs the other. Lastly, Figure 5 results are demonstrated as if RNA features are more informative than ATAC features but it is unclear if the model was pre-trained on ATAC (or ADT) data at all. I believe the 2023 version of CELLxGENE only has transcriptomic data, so it would not be surprising if the model did not learn as informative embeddings for ATAC and ADT  features, for example.  

**5.** One of the goals of scFM field is to eventually simulate effects of unseen perturbations. While we are far from this goal, it is unclear how one would simulate perturbation profiles when there is no data to pick anchors for the Metropolis-Hastings procedure. Additionally, a few key information is missing in this section:  
&nbsp;&nbsp;&nbsp;&nbsp; **(a)** It is unclear how the temperature parameter \tau (for exploration vs exploitation tradeoff in proposals) and the \beta parameter in Wasserstein-1 objective are picked and whether the perturbation simulation performance is sensitive to these.  
&nbsp;&nbsp;&nbsp;&nbsp; **(b)** A fair comparison would require that scFM baselines are fine-tuned on the training split since training split is used for anchors in CellxPert. I assume this is the case but I could not locate any information on this in the paper.  
&nbsp;&nbsp;&nbsp;&nbsp; **(c)** Some baselines used with K562 cells are not used with RPE-1 cells and it is unclear why, unless I am missing a note in the manuscript.  
&nbsp;&nbsp;&nbsp;&nbsp; **(d)** For cosine-shift calculations, which cells' embeddings are used: training split, test split, all? This is quite important to clarify since depending on the choice, you may be coupling the benchmark/metric to the model itself in some ways.

From the point of view of my own evaluation, I think the most reasonable and effective way to improve the submission would be to include appropriate, robust benchmarking experiments that address concerns #1-#3. I appreciate the effort toward multi-omic scFM and believe there are some interesting ideas in the manuscript, so I would be happy to raise my score with demonstrated improvements regarding these concerns.

### Questions
**1.** How do the authors propose users to determine how many tokens to allocate per modality?  
**2.** Is the quantile binning performed on a gene-by-gene basis or are bin boundaries determined on expression levels across all genes?  
**3.** Are there any experiments showing whether four random permutations is sufficient to achieve position invariance?  
**4.** Are the scFM baselines on Table 3 fine-tuned on the training split?  
**5.** For cosine-shift calculations, which cells' embeddings are used: training split, test split, all?  
**6.** Do the scFM baselines in Table 4 all only take in transcriptomic data?  
**7.** How do the authors propose to simulate perturbation effects with no training data to pick anchors?

### Soundness
1

### Presentation
3

### Contribution
2
