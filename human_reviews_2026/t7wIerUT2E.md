# Controllable diffusion-based generation for multi-channel biological data

- Decision: Accept (Poster)
- Scores: 4, 2, 0, 8

## Abstract
Biological profiling technologies, such as imaging mass cytometry (IMC) and spatial transcriptomics (ST), generate multi-channel data with strong spatial alignment and complex inter-channel relationships. 
Modeling such data requires generative frameworks that jointly model spatial structure and inter-channel dependencies and generalize across arbitrary subsets of observed and missing channels. Existing generative models typically assume low-dimensional inputs (e.g., RGB images) and rely on simple conditioning mechanisms that disrupt spatial correspondence and overlook inter-channel dependencies. This work proposes a unified multi-channel diffusion (MCD) framework for controllable generation of structured biological data with complex inter-channel relationships. Our model introduces two key innovations: (1) a hierarchical feature injection mechanism that enables multi-resolution conditioning on spatially aligned observed channels, and (2) two complementary channel attention modules to capture inter-channel relationships and recalibrate latent features. To support flexible conditioning and generalization to arbitrary sets of observed channels, we train the model using a random channel masking strategy, enabling it to reconstruct missing channels given any combination of observed channels as the spatial condition. We demonstrate state-of-the-art performance across both spatial and non-spatial biological data generation tasks, including imputation in spatial proteomics and clinical imaging, as well as gene-to-protein translation in single-cell datasets, and show strong generalizability to unseen conditional configurations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Many biological modalities are inherently multi‑channel and spatially co‑registered (IMC, clinical imaging, spatial transcriptomics). Off‑the‑shelf conditional diffusion often assumes low‑dimensional channels and crude conditioning that breaks spatial correspondence. The paper proposes MCD (multi‑channel diffusion) for controllable generation and imputation given arbitrary subsets of observed channels. Core claims: (1) hierarchical feature injection to keep spatial alignment; (2) two channel‑attention modules to model inter‑channel dependencies; and (3) random channel masking for amortized training across any condition configuration. Experiments cover single‑cell gene‑to‑protein prediction, IMC imputation, and cross‑dataset generalization.

### Strengths
- The hierarchical feature‑injection pipeline preserves spatial correspondence at every resolution, which is exactly what biological channels need. The injection is simple to deploy and avoids brittle concatenation tricks common in vision diffusion.
- Pairing SE gating with transformer‑style channel attention gives a plausible division of labor: per‑sample recalibration and higher‑order inter‑channel structure. The additional SE head at the output enforces cross‑channel coupling where it matters.

### Weaknesses
- Comparators underpowered for spatial tasks. ControlNet is a fair reference, but the paper doesn’t benchmark strong end‑to‑end, jointly‑trained spatial conditioners or ablate cross‑attention conditioning versus the proposed SE‑gated injection. Given that the method already computes $E_{\ell}(c)$, a direct cross‑attention baseline would be informative.
- Masking policy and distribution shift. Random masking zeros unobserved channels in the condition (Algorithm 1). The paper does not analyze whether zero‑masking creates a train/test mismatch when “missing” at test time means “physically unmeasured but nonzero distribution,” especially for modalities where absolute intensity carries semantics.
- Metrics don’t reflect calibration. Spatial evaluation is Pearson r. No per‑channel calibration (e.g., bias/variance decomposition), no uncertainty quality, no region‑level histograms for key biomarkers. For clinical‑adjacent imaging, correlation alone can be misleading.

### Questions
- (Multi-channel vs multi-source) The setting is similar to multi-source integration problem, and the masking idea is similar to scVAEIT. It would be interesting to discuss the connection. (Du, J.‑H., Cai, Z., Roeder, K. (2022), Robust probabilistic modeling for single‑cell multimodal mosaic integration and imputation via scVAEIT.”)
- (Masking policy) What masking probability $p$ and schedule work best? Any performance cliffs when masks are extremely sparse/dense? Show sensitivity curves and OOD mask combos not seen in training.
- (Scalability) Provide computational complexity (e.g., wall‑clock, memory, and FLOPs) versus $C$, $D$, and $H\times W$ for both attention modules. Where does channel self‑attention become the bottleneck?

### Soundness
2

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
3

### Summary
The paper proposes MCD, a conditional diffusion framework for multi-channel biological data. Conditioning is handled via masked images. The proposed architecture is a dual network with hierarchical feature injection from a contextual encoder into the denoiser, plus two channel-attention mechanisms (SE for injection and transformer-style channel self-attention inside unet blocks). Experiments show state of the art level results on CITE-seq protein prediction, IMC channel imputation including a union vs intersection multi-dataset study, hybrid controls versus ControlNet/BrushNet, and BraTS MRI.

### Strengths
- **Problem relevance.** Training with random channel masking yields one model that accepts arbitrary observed subsets and making it flexible. The union and intersection result supports cross-dataset integration under partial channel overlap.

- **Strong empirical results.** When reported, the method consistently outperforms baselines. Experiments are broad and span single/multi dataset setups and including hybrid controls.

- **Ablations.** Stepwise ablations and ControlNet/BrushNet hybrids help attribute gains to hierarchical injection.

### Weaknesses
- **Subset-size stress-tests are missing.** One of the core claims is robustness to arbitrary observed subsets, but there is no sweep of performance vs. #observed channels / masking-probability p, nor targeted leave a group out per channel families. Single vs multi channel and union vs intersection is positive but partial.

- **Efficiency evidence.** Table 1 lists SiD(1-step) with near identical accuracy and claims two orders of magnitude speedup, but there are no wallclock analysis for readers to observe the real performance gains. 

- **Method clarity.** I find the pieces in the methods section hard to put together. It requires stitching together several sections to reconstruct the exact flow of input, context encoder, SE-gated injections per scale, denoiser block attention, output SE flow. A single forward schematic/pseudocode can improve the clarity significantly. 

- **CFG analogy is conceptual.** I could not directly link random masking with CFG. Instead, the authors can provide a deeper the analysis of how the compared baselines actually operate and where they fall short.

- **Reproducibility.** The paper promises code upon publication without anonymous repo or supplementary materials. This limits the verification significantly.

### Questions
See the questions and actionable items below. I find the core idea is strong, but the draft feels rushed; clearer presentation and a few added analyses would better show the paper's potential.


- **MedVAE ablation.** It would be nice to see the effect of the latent space choice on the results. One can convert each channel into grayscale (or stack each channel 3 times to imitate RGB) and run through a stable diffusion VAE pipeline do the latent diffusion.

- **Efficiency plots.** Single step generation has a clear advantage from its strong results and efficiency but it would be nice to see more details on how you distilled together with the performance comparisons with other baselines.

- **Confidence intervals.** Report mean $\pm95\%$ CI over multiple seeds for all main tables to quantify variance from training and random masking. 

### Mistakes in text.

- **Section 2.2** The part describing classical RGB image imputation and classical RGB colorization problems are not correct. If we have $C_m = 0$ and $C_o = C = 3$ there is nothing to impute. Additionally, for the colorization example $C_m = 3$ and $C_o = 1$ implies $C = C_m + C_o = 4$ which is clearly not RGB. 

- **Dataset reference for BMMC.** Table 1 report BMMC (bone marrow mononuclear cells) as BMNC both in the body and in the caption.

- **Different acronym on Table 3.** Throughout the paper the authors refer their method as MCD however Table 3 uses DiffuseMRI

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This work proposes a unified multi-channel diffusion (MCD) framework for controllable generation of structured biological data.

### Strengths
1. The idea of developing a framework capable of controllably generating multi-channel biological data using diffusion models is interesting.

### Weaknesses
1. The paper is quite obscure and its objective remains unclear. The title suggests that it focuses on developing a generative framework for multi-channel biological data, but the type of data is not specified. I assumed the authors were referring to images, yet in the experiments they attempt to predict protein expression from paired scRNA-seq data, and later they evaluate their method on MRI images. This inconsistency makes the overall methodology difficult to understand and significantly undermines the coherence of the paper.

2. The paper contains several incorrect claims and assertions. To list a few: “Existing generative models typically assume low-dimensional inputs (e.g., RGB images)” and “In spatial profiling data, each channel designates a specific molecule of interest (e.g., proteins n ≥ 30 and genes n ≥ 100), and each pixel (or cell)…”.

3. The manuscript contains numerous writing inconsistencies, redundant and buzzword-heavy claims, and incorrect or oversimplified descriptions of diffusion theory. Core method elements (random channel masking, SE attention) are incremental and poorly justified as novel contributions.

### Questions
1. The authors declared the following: "All experiments were trained on NVIDIA A5000 GPUs with 24 GB of VRAM. The model was trained for 2000k imgs with a batch size of 256, taking approximately 2 hours to complete at 16 × 16 resolution. All results were obtained using a single-GPU setup unless otherwise specified." Which kind of biological data have a resolution of 16 × 16?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a Multi-Channel Diffusion (MCD) framework, designed for controllable synthesis and imputation of multi-channel biological data such as spatial proteomics, single-cell omics, and MRI. This framework integrates a mechanism for hierarchical spatial feature injection with a dual channel-attention module. This enables the resulting models to preserve spatial alignment while capturing complex inter-channel dependencies. A wide range of experiments on publicly available benchmarks suggest that MCD outperforms existing baselines.

### Strengths
1) This paper combines two different methodological innovations in a quite ingenious way, effectively addressing the spatial and inter-channel complexity of biological data.

2) The resulting models demonstrates versatility across multiple domains, including spatial proteomics, single-cell omics, and MRI modality synthesis, showing strong generalization and scalability.

3) Finally, the presented evaluation is quite comprehensive. I particularly appreciate the ablation studies that assess the individual contributions of model components.

### Weaknesses
1) All comparisons reported in Tables 1 to 3 lack any assessement of stastistical significance. This makes it difficult to gauge whether differences in performances are actually significant.
2) There is not biologically-grounded evaluation of the imputed data. For example, are known protein markers expressed in their corresponding cells?

### Questions
I would ask the authors to address the weaknesses highlighted above. Particularly:
1) Evaluate differences in predictive performance between their method and the others through appropriate statistical tests
2) Check whether expected cell-specific markers are expressed in the inputed data. You could also scale up this assessement to the pathway level, and check which biological pathways are enriched in the imputed data and if the enrichment results are biologically meaningful

### Soundness
3

### Presentation
3

### Contribution
3
