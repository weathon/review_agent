# Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Computational modeling of single-cell gene expression is crucial for understanding cellular processes, but generating realistic expression profiles remains a major challenge. This difficulty arises from the count nature of gene expression data and complex latent dependencies among genes. Existing generative models often impose artificial gene orderings or rely on shallow neural network architectures. We introduce a scalable latent diffusion model for single-cell gene expression data, which we refer to as scLDM, that respects the fundamental exchangeability property of the data. Our VAE uses fixed-size latent variables leveraging a unified Multi-head Cross-Attention Block (MCAB) architecture, which serves dual roles: permutation-invariant pooling in the encoder and permutation-equivariant unpooling in the decoder. We enhance this framework by replacing the Gaussian prior with a latent diffusion model using Diffusion Transformers and linear interpolants, enabling high-quality generation with multi-conditional classifier-free guidance. We show its superior performance in a variety of experiments for both observational and perturbational single-cell data, as well as downstream tasks like cell-level classification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes scLDM, a two‑stage generative framework for single‑cell gene expression that (i) uses a fully transformer VAE designed for exchangeable sets of genes and (ii) replaces the VAE’s Gaussian prior with a latent diffusion model trained with flow matching and linear interpolants (SiT) and parameterized with Diffusion Transformers (DiT). The key architectural element is a Multi‑head Cross‑Attention Block (MCAB) that’s used for permutation‑invariant pooling in the encoder and permutation‑equivariant “unpooling” in the decoder via index‑selected embeddings. The decoder parametrizes a Negative Binomial likelihood for counts. Experiments cover: (1) reconstruction on observational scRNA‑seq, (2) unconditional and conditional generation on observational data, (3) multi‑attribute conditional generation on perturbational datasets (cell context × perturbation), and (4) downstream classification using the learned embeddings. In tables and plots, scLDM typically outperforms scVI, CFGen, CPA, and scDiffusion on W2/MMD²/Fréchet and reconstruction metrics; the classifier‑free guidance is extended to multi‑hot conditioning. Figure 1 gives the system diagram; Tables 1–3 report core wins; Figure 2–3 show qualitative UMAPs; Table 4 summarizes downstream classification.

### Strengths
- Exchangeability by construction. The encoder’s MCAB uses fixed “pseudo‑inputs” $S$ to pool arbitrary gene sets into a fixed number of latent tokens; the decoder restores per‑gene outputs by cross‑attending from latents to the index‑selected embeddings $E_I$. This yields permutation‑invariant posteriors and permutation‑equivariant likelihoods, matching the set‑valued nature of genes. The design is clean and scalable relative to naïve all‑token attention.
- Multi‑attribute conditioning at sampling time. The extended classifier‑free guidance supports joint multi‑hot attributes, not just additive singletons, and the ablation in the supplement claims the standard joint form outperforms the additive variant for multi‑attribute control.
- Downstream utility. VAE embeddings competitive with scRNA representation models on COVID‑19 classification; F1 up to 0.820 with the largest scLDM encoder, on par with TranscriptFormer and better than scVI/scGPT/Geneformer.

### Weaknesses
- Novelty is incremental on the set‑architecture axis. MCAB is essentially a Perceiver‑style cross‑attention with learned inducing tokens in the encoder and index‑selected keys in the decoder. The “unified block for pooling/unpooling” is neat, but the paper underplays overlap with SetTransformer/Perceiver‑family parameterizations. The exchangeability story is correct but not conceptually new; the new part is applying this cleanly to scRNA with a counts‑likelihood, then pairing it with a DiT latent prior. The framing should be more honest about what’s new vs re‑packaged.
- Evaluation uses HVGs in several core tables, partially undercutting the “scalable to all genes” narrative. For perturbational datasets, analyses restrict to the top 2,000 HVGs. This weakens the claim that scLDM intrinsically handles arbitrarily large, unordered gene sets without feature selection.
- Metrics and calibration risk. W2, MMD²‑RBF, and Fréchet on embeddings are common, but the paper doesn’t show per‑gene marginal calibration (e.g., KS or QQ vs Negative Binomial), zero‑inflation fidelity, or gene‑gene correlation structure retention. The qualitative UMAPs are persuasive but can be misleading without per‑marker distribution checks.

### Questions
- (Related work) "two key challenges limit existing methods." The author ignore some recent work like scGPT, scFoundation, scdiffusion, etc, that are not based on GAN. The claim of the second limit is thus invalid. Additionally, related statistical generative models/simulators like Splatter are not compared.
- Even though the gene order in general datasets does not has specific biological meaningful, their corresponding location on DNA could provide some information, if one orders the genes in this way.
- What are the computational complexities for compared methods? Provide FLOPs/throughput and memory scaling with number of genes, latent tokens m, and pseudo‑input size $∣S∣$, for both encoder/decoder and the DiT prior. Include wall‑clock on representative datasets.
- Exchangeability baselines. Please run an ablation comparing MCAB to SetTransformer/Perceiver variants with and without the index‑selected $E_I$ trick. Are the gains architectural or due to the latent diffusion prior? Report reconstruction and generation deltas.

### Soundness
2

### Presentation
3

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
The paper proposes scLDM, a fully transformer-based VAE for exchangeable single-cell gene expression that uses a Multi-head Cross-Attention Block (MCAB) both as a permutation-invariant pooling operator in the encoder and a permutation-equivariant unpooling operator in the decoder. The standard Gaussian prior is replaced with a latent diffusion model (DiT + SiT with flow matching) to improve sample quality and enable multi-attribute classifier-free guidance. Empirically, scLDM is evaluated on observational datasets (Dentate Gyrus, Tabula Muris, HLCA) and perturbational datasets (Parse 1M, Replogle), reporting strong reconstruction and generation results and competitive downstream classification from VAE embeddings.

### Strengths
* The paper is well-written and well-motivated.
* They showed competitive or superior performance of their method across different experiments. 
* The idea of MCAB in the single-cell generative model context is novel.

### Weaknesses
* The text, especially in the method section and experiment description, lacks clarity, includes incorrect cross-referencing, and overloaded notation (e.g, equation 33). The trains and test sets across experiments are not well specified. 
 * In Table 1, the HLCA Pearson correlation (~0.1 for other methods vs. ~0.4 for scLDM) does not appear consistent with the UMAPs in Figure 2. 
* Metrics and baselines for the perturbation benchmark are limited; more recent methods (e.g., CellFlow, GEARS) should be included. Because perturbation effects are subtle, it is important to report metrics on differentially expressed genes (DEGs).  
* Results in Tables 1 and 2 do not match the CFGen paper on the same benchmarks; this may be due to hyperparameters or data splits and should be reconciled.

### Questions
* The method comprises several components, but there is no ablation in the main text. What is the effect of each component (e.g., MCAB design, latent diffusion prior, guidance strategy)?
* Were the foundation models in the benchmark fine-tuned on the training data, or used zero-shot? Please clarify.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces a two-stage generative model for single-cell RNA-seq. First, a transformer VAE with a single Multi-head Cross-Attention Block (MCAB) encodes unordered gene sets and decodes them back, staying permutation-aware. Second, instead of a Gaussian prior, the latent space is modeled with a Diffusion Transformer trained with flow matching and classifier-free guidance, so the model can generate cells under multiple conditions (cell type, context, perturbation).

### Strengths
1. Doing the diffusion in the latent tokens learned by the VAE avoids operating over tens of thousands of genes. Using DiT in latent space is interesting.

### Weaknesses
1. The paper says: train VAE -> freeze -> train DiT diffusion with classifier-free guidance. But we don’t get wall-clock, number of diffusion steps, model sizes per dataset, or memory footprint, yet the model is transformer-based end to end and is evaluated on million-cell datasets. This matters for reproducibility and to support the “scalable” claim.
2. The MCAB is the key novelty, but we don’t see a direct ablation like “replace MCAB with SetTransformer pooling” or “use separate pooling/unpooling” to isolate how much of the gain comes from this block vs. the diffusion prior.
3. Comparison set is strong but not exhaustive. scLDM is compared to CPA and scVI on perturbations, but there are more recent OOD-perturbation models (e.g. transport/OT-style, flow-matching-on-manifolds).
4. Biological faithfulness is shown mostly through distributional metrics + UMAPs. W2, MMD², FD are good, but we need lineage/trajectory or marker-level checks (for example “does the model preserve marker co-expression under perturbation?”).

### Questions
1. If you replace MCAB with (i) pooling by multihead attention (SetTransformer-style) or (ii) a Perceiver IO block with separate latent arrays, how much do reconstruction and FD/W2 degrade?
2. You report better F1 than scGPT/Geneformer on COVID. Is this purely because the VAE is trained on a big Human Census dataset, or is there a benefit from diffusion pretraining too?

### Soundness
2

### Presentation
2

### Contribution
2
