000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Molminer: Towards Controllable, 3D-Aware, Fragment-Based Molecular Design

Anonymous authors Paper under double-blind review

## Abstract

We introduce MolMiner, a fragment-based, geometry-aware, and order-agnostic autoregressive model for molecular design. MolMiner supports conditional generation of molecules over twelve properties, enabling flexible control across physicochemical and structural targets. Molecules are built via symmetry-aware fragment attachments, with 3D geometry dynamically updated during generation using forcefields. A probabilistic conditioning mechanism allows users to specify any subset of target properties while sampling the rest. MolMiner achieves calibrated conditional generation across most properties and offers competitive unconditional performance. We also propose improved benchmarking methods for both unconditional and conditional generation, including distributional comparisons via Wasserstein distance and calibration plots for property control. To our knowledge, this is the first model to unify dynamic geometry, symmetry handling, order-agnostic fragment-based generation, and high-dimensional multi-property conditioning.

## 1 Introduction

Deep generative models are increasingly central to modern high-throughput screening (HTS) pipelines (Westermayr et al., 2023; Ortega Ochoa et al., 2023), where they generate candidate molecules tailored to specific properties before being filtered through successively more expensive stages: from machine-learning surrogates (Schutt et al., 2017) to quantum chemical calculations ¨ such as density functional theory (Kohn & Sham, 1965). These models span a wide range of molecular representations (e.g., SMILES (Weininger, 1988), molecular graphs) and generative approaches (e.g., VAEs (Gomez-Bombarelli et al., 2018; Lim et al., 2018), diffusion models (Hoogeboom et al., ´ 2022b; Wu et al., 2022)). While many methods address isolated challenges—such as chemical validity, structural diversity, or property control—it remains rare to find models that simultaneously support the full range of capabilities required for practical molecular design. In real-world settings, models must go beyond oneshot generation to support multi-step, interpretable generation processes that flexibly adjust molecular size, incorporate chemically meaningful fragments, and maintain validity throughout. Multistep generation also enables human-in-the-loop design, offering greater transparency and interactive control. Furthermore, capturing 3D geometry is essential when structure-dependent properties are targeted—yet few autoregressive frameworks incorporate this effectively (Voloboev, 2024). Another limitation in existing methods is rigid rollout order: most autoregressive models grow molecules from a fixed atom or fragment, reducing flexibility and diversity. We address this with an order-agnostic rollout strategy that allows growth from any starting fragment in any valid order. Finally, conditional generation is critical for use in HTS pipelines, where desired molecular properties are specified upfront. While most models support only single-target conditioning, we enable multi-property control over twelve physicochemical and structural properties. Users can condition on any subset of properties, and the remaining ones are automatically sampled—facilitating efficient, targeted exploration of chemical space. We introduce MolMiner, a unified, fragment-based generative model designed for flexible and controllable molecular generation. Our key contributions are:
054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082

## 083

084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Our work builds on fragment-based molecular generation approaches such as JTNN (Jin et al., 2019) and HierVAE (Jin et al., 2020), which assemble molecules sequentially while enforcing chemical validity. Like these models, we use coarse-grained molecular fragments and an autoregressive decoding process. Our model is also order-agnostic, similar in spirit to G-SchNet (Gebauer et al., 2022), allowing flexible rollout without fixing a starting point or strict atom ordering, whereas JTNN and HierVAE are fragment-based but order-fixed, and G-SchNet is order-agnostic but atom-based. Additionally, unlike G-SchNet, we allow the geometry of the partial molecule to remain dynamic during generation, rather than freezing atom positions prematurely. We also explicitly introduce a systematic method to handle fragment symmetries during attachment, an aspect not clearly detailed in earlier fragment-based models such as MoLeR (Maziarz et al., 2024). Finally, we demonstrate conditional generation across twelve molecular properties simultaneously; a scale of multi-target control that, to the best of our knowledge, has not previously been achieved in molecular generative modeling.

## 3 Method

We model molecular generation as a fragment-based, order-agnostic (Uria et al., 2014; Hoogeboom et al., 2022a), autoregressive process. Molecules are first decomposed into non-overlapping fragments based on rings and bonds, with attachment points standardized to account for fragment symmetries. Generation proceeds step-by-step: at each step, the model is queried with a focal attachment point on the current partial structure and predicts either a new fragment to attach or a decision to terminate that site. To incorporate 3D information, the partial molecular structure is relaxed using a forcefield (e.g., UFF (Rappe et al., 1992)) and the spatial arrangement is used to inform the prediction. This avoids the rigid, frozen geometries seen in prior methods (Gebauer et al., 2022) and ensures that predictions are conditioned on realistic intermediate structures. Formally, we define the probability of a molecule M as the expected likelihood over all valid rollout trajectories R, each consisting of a sequence of fragment attachment actions:

$$p({\mathcal M})=\mathbb{E}_{R\sim{\mathcal U}({\mathcal R}({\mathcal M}))}\left[\prod_{i=1}^{|R|}p_{\theta}\big(x_{i}^{(R)}|{\bf x}_{<i}^{(R)},c\big)\right],$$
$$(1)$$

Here, xi = (fi, ai) is a fragment-attachment pair where fi ∈ Vf is a fragment from the vocabulary and ai ∈ Va(fi) is a valid attachment configuration for fi. The model may also select a special
"termination" action, which marks an attachment site as closed. The sequence x
(R)
<i denotes the partial structure up to step i, c represents optional conditioning information (e.g., target properties), |R| is the length of the rollout, and U denotes uniform distribution over the set of valid rollout orders R(M). Generation proceeds by alternately attaching fragments or terminating open sites. The process continues until all sites are resolved, yielding a chemically valid, fully assembled molecule. Figure 1 illustrates a step in the autoregressive process of molecular generation with MolMiner.

- **Multi-property conditional generation**: MolMiner supports conditioning on any subset of twelve molecular properties, enabling flexible, user-defined control. It achieves accurate and calibrated generation across a wide range of targets.

- **Symmetry-aware 3D modeling**: We incorporate a dynamic, forcefield-driven geometry update during generation and introduce a standardized protocol to handle fragment symmetries.

- **Order-agnostic generation**: Our rollout strategy avoids fixed atom ordering, improving flexibility and acting as a regularizer.

- **Targeted evaluation protocols**: We propose Wasserstein-based distributional metrics and calibration plots to rigorously assess both unconditional and conditional performance.

## 2 Related Work

![2_image_0.png](2_image_0.png)

## 3.1 Fragment-Based Molecular Representation

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Although SMILES syntax allows for explicit encoding of atom-specific metadata, such as attachment points using atom-map numbers, incorporating such information would interfere with the canonicalization procedure itself, altering the resulting SMILES string. Furthermore, explicit labeling does not resolve fragment symmetries, where multiple attachment sites may be chemically indistinguishable. A simple example is benzene, where all carbon atoms are symmetry-equivalent. Molecules naturally exhibit hierarchical structure, often containing repeating substructures such as rings and functional groups. To capture this, we represent molecules as assemblies of chemically meaningful, non-overlapping fragments. Specifically, we apply a coarse-graining procedure that decomposes each molecule into a set of fragments corresponding to rings, identified via the RDKit's Smallest Set of Smallest Rings (SSSR) (Landrum, 2024), and isolated bonds not within a ring. This decomposition strategy is similar to the "small motif" variant explored in HierVAE (Jin et al., 2020), where molecules are fragmented into minimal cyclic and bond-based motifs. Each extracted fragment is uniquely represented by its Canonical SMILES string (Weininger et al., 1989), computed using RDKit's implementation (Landrum, 2024), providing a compact, humanreadable encoding that is invariant to atom indexing within this scheme1. However, canonical SMILES do not retain explicit information about how fragments were connected in the original molecule. To preserve attachment information, we track the mapping between each atom's original index in the full molecule and its local index in the extracted fragment. This allows us to recover the attachment points necessary for reassembling molecules from fragments and sets the foundation for our symmetry-aware attachment modeling described next. We treat each fragment as a discrete token, analogous to tokenization in natural language processing (NLP). This abstraction enables us to associate each fragment with a learnable embedding and formulate molecular generation as a stepwise prediction over a sequence of fragment tokens.

## 3.2 Symmetry-Aware Attachment Modeling

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 To ensure that fragment attachments are consistently and unambiguously represented, we introduce a symmetry-aware standardization procedure. Since our coarse-graining process extracts fragments corresponding to rings and bonds—both of which are single cycles—the problem of matching atom indices before and after canonicalization reduces to finding valid cyclic permutations. This is because RDKit's canonicalization relies on graph traversal (variants of depth-first or breadth-first search) that follow the topology of the cycle, making reindexing predictable up to a rotation. We exploit this structure by computing similarities between atom environments and identifying cyclic shifts consistent with the fragment's chemical graph. To reconstruct the atom index correspondences, we compute pairwise similarities between atoms based on their local chemical environments, using Morgan fingerprints (Rogers & Hahn, 2010) and Tanimoto (Tanimoto, 1958) similarity. This yields a similarity matrix that captures correspondences between atom environments in different indexing orders. Valid cyclic permutations are then extracted by identifying rotations that maintain high-similarity mappings across the fragment. Once valid shifts are found, we select a consistent common frame that unifies attachment configurations across symmetric cases, ensuring that generation decisions are invariant to fragment symmetries. Further technical details on the fragment extraction and attachment point handling are provided in Appendix A.6.

## 3.3 Order-Agnostic Molecular Rollouts

Molecular generation is framed as a sequence of fragment attachments. At each step, the model is queried with a specific focal attachment point on the current partial structure and predicts either a new fragment to attach or a decision to leave the point vacant. Unlike previous methods that use a fixed rollout order (e.g., breadth-first or depth-first traversal), we adopt an order-agnostic strategy: the next focal attachment point is sampled randomly from the available open sites. The only constraint is that new fragments must attach directly to the existing structure, ensuring the molecule grows as a single connected component. By avoiding any specific traversal scheme and allowing arbitrary selection among open sites, we maximize the flexibility and diversity of possible rollouts for each molecule. The rollout is initialized by selecting a starting fragment at random from the molecule's fragment set, and identifying its available attachment points. These open sites are placed into an exploration queue. At each step, an attachment point is sampled from the queue, and the model predicts either a fragment to attach or a decision to terminate the site. Unlike linear sequence generation in natural language models, where a single global termination token signals the end of generation, our process is inherently parallel: termination occurs locally at each attachment point. The molecule is considered complete only when all open attachment sites have either been connected to fragments or explicitly closed. This decentralized termination mechanism reflects the graph-like structure of molecules and allows generation to proceed flexibly through multiple concurrent growth paths. During training, rollouts are precomputed: for each molecule, a sequence of attachment actions and intermediate geometries is generated in advance. This allows efficient learning without the need for force field optimization during training epochs. In contrast, during generation, the molecule is built incrementally, with geometry relaxed after each attachment step via a classical force field. This dynamic procedure ensures that predictions remain geometry-aware throughout autoregressive sampling.

## 3.4 Model Architecture

The model is implemented as a decoder-only transformer (Vaswani et al., 2023) operating over a sequence of fragment tokens. Each fragment is associated with a learnable embedding vector. To incorporate local chemical context, we augment each embedding with three normalized features indicating the fraction of attachment sites that are occupied, free, or sealed. These enriched representations serve as inputs to the transformer layers and help distinguish between fully bonded, partially open, and terminated fragments. To make the model geometry-aware, we incorporate spatial information directly into the attention mechanism via a global attention bias (Shehzad et al., 2024). Specifically, the attention coefficients between fragments i and j are given by 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

$$\alpha_{i j}=\frac{e^{\;g(h_{i},h_{j})+\theta\cdot D_{i j}}}{\sum_{k=1}^{N}e^{\;g(h_{i},h_{k})+\theta\cdot D_{i k}}},\quad D_{i j}=e^{-\frac{||\mathbf{x}_{i}-\mathbf{x}_{j}||^{2}}{2\sigma^{2}}},\quad g(h_{i},h_{j})=\frac{h_{i}\cdot h_{j}^{\top}}{\sqrt{d_{h}}},$$
√dh, (2)
where Dij is a Gaussian-decayed distance kernel, θ is a learnable scalar controlling the strength of the geometric bias, and hi denotes the hidden representation of fragment i, as produced by the self-attention mechanism of the previous transformer layer. This mechanism allows the model to attend more strongly to nearby fragments without requiring explicit positional encodings. Unlike sequence-based tasks in NLP, molecules do not follow a canonical linear order. Instead, spatial relationships emerge from the 3D configuration of fragments. Our attention bias thus acts as a spatial inductive prior that replaces standard positional embeddings with a structure-aware alternative. During generation, the model is conditioned on the current fragment set, a designated focal fragment, and a specific attachment site (the "hit location"). After processing the structure through the transformer, we perform a focalized readout: the focal embedding attends to all fragments, with attention scores further biased by distances to the hit location. This aggregates global context while emphasizing the local growth site. The resulting vector is concatenated with the conditioning properties, passed through a feed-forward layer, and projected onto the vocabulary of fragment-attachment actions, including the termination action.

## 3.5 Training Objective

We train the model to maximize the log-likelihood of each molecule M under the order-agnostic rollout factorization (Uria et al., 2014; Hoogeboom et al., 2022a), conditioned on target properties c:

$$\mathcal{L}(\theta\mid\mathcal{M})=\log\mathbb{E}_{R\sim\mathcal{U}(\mathbb{R}(\mathcal{M}))}\left[\prod_{i=1}^{|R|}p_{\theta}\big{(}x_{i}^{(R)}|\mathbf{x}_{<^{i}}^{(R)},c\big{)}\right]\geq\mathbb{E}_{R\sim\mathcal{U}(\mathbb{R}(\mathcal{M}))}\left[\sum_{i=1}^{|R|}\log p_{\theta}\big{(}x_{i}^{(R)}|\mathbf{x}_{<^{i}}^{(R)},c\big{)}\right]\tag{3}$$
$$(2)^{\frac{1}{2}}$$

The expectation is over all valid rollouts R of M, with the lower bound derived via Jensen's inequality (Jensen, 1905). In practice, we use a Monte Carlo approximation of the expectation and randomly sample one rollout per molecule per epoch, providing natural data augmentation by exposing the model to diverse construction orders. At each step, it is trained to predict the next fragment-attachment pair or a termination action, conditioned on the current partial structure and target properties. To initiate rollouts, we jointly train an auxiliary model to predict a suitable starting fragment from the target properties. This predictor is a feed-forward network that outputs independent probabilities for each fragment in the vocabulary, framing the task as multi-label classification. It is trained with binary cross-entropy loss to encourage high scores for fragments present in the molecule. Both models share the same training splits. Together, these components enable end-to-end conditional generation—from fragment selection to flexible, geometry-aware rollouts. Importantly, conditioning is implemented in a fully implicit manner: target properties are provided as inputs during training, but no auxiliary loss is applied to enforce property compliance. This allows the model to learn property alignment organically from the data distribution.

## 3.6 Sampling Procedure

To generate a molecule conditioned on user-specified properties, we begin by completing the conditioning vector when only a subset of target properties is provided. The missing properties are sampled using a Gaussian Mixture Model (GMM) (McLachlan, 2000) fitted to the empirical distribution of training data. This ensures that completed conditioning vectors remain realistic and consistent with the underlying data distribution. Further details on GMM training and validation are provided in Appendix A.2. Once the conditioning vector is completed, generation is initialized by selecting a starting fragment. A trained fragment predictor assigns independent probabilities over the fragment vocabulary, from which a seed fragment is sampled. The molecule is then constructed autoregressively. In Appendix A.7, we evaluate several sampling strategies, including greedy and probabilistic decoding, as well as seed fragment selection from the top-k predictions (with k = 3, 5, 10). We also investigate how conditioning values influence the choice of starting fragment in Appendix A.4.

## 4 Experiments

We evaluate our model on a subset of the ZINC dataset (Irwin et al., 2012) originally curated for ChemicalVAE (Gomez-Bombarelli et al., 2018), containing approximately 200,000 drug-like ´ molecules. Each molecule is annotated with 12 properties computed using RDKit, which are used both for conditioning and evaluation (see Appendix A.1 for details). We adopt an 80/10/10 train/validation/test split.

## 4.1 Training And Ablation Summary

We train an 8-layer decoder-only transformer trained with AdamW (Kingma & Ba, 2017; Loshchilov & Hutter, 2019) and a linear warmup-decay schedule. Hyperparameters were selected via grid search (Appendix A.3), with the final configuration using a 0.3 dropout rate, 64 attention heads, 0.15 warmup ratio and a 5e-5 peak learning rate. Ablation studies confirm three key findings: (i) conditioning on more properties improves performance, consistent with the "tomographic effect" (Ortega-Ochoa et al., 2025), where richer conditioning helps disambiguate structure, (ii) geometryaware attention aids performance when initialized with positive bias, and (iii) rollout resampling serves as effective regularization, reducing overfitting. These inform our final model, trained with resampling for 50 epochs.

## 4.2 Benchmarking Unconditional Generation

While our model is optimized for conditional generation, we evaluate it under unconditional settings for completeness. We evaluate unconditional generation by measuring how closely the model reproduces the property distributions of the training data. Because direct comparison of molecular graphs is challenging, we instead compare the distributions of twelve physicochemical and structural properties between 5,000 generated molecules and the dataset. These properties include: logP (logarithm of water partition coefficient, used as a measure of lipophilicity) (Wildman & Crippen, 1999), QED (quantitative estimate of drug-likeness) (Bickerton et al., 2012), SAS (synthetic accessibility) (Ertl & Schuffenhauer, 2009), FractionCSP3 (sp3carbon fraction), molecular weight, TPSA
(topological polar surface area) (Ertl et al., 2000), MR (molar refractivity, descriptor accounting for molecular size and polarizability) (Wildman & Crippen, 1999), hydrogen bond donors and acceptors (HBD, HBA), ring count, number of rotatable bonds (flexibility), and number of chiral centers (stereochemical complexity). To compare distributions, we use the 1D Wasserstein distance for each property, following (Polykovskiy et al., 2018), providing a robust measure of distributional similarity. In addition, we report three standard metrics: Uniqueness (fraction of distinct molecules), Novelty (fraction of valid, unique molecules not present in the dataset), and Diversity (average pairwise Tanimoto distance among generated molecules). We omit validity, as our model enforces valence constraints during generation and consistently produces valid molecules. Molecular identity is determined solely by connectivity, using the first block of the InChIKey (Heller et al., 2015; Pletnev et al., 2012), which encodes the molecular skeleton and excludes variation due to stereoisomerism, tautomerism, and related forms of isomerism.

As our model is inherently conditional, we simulate unconditional generation by sampling conditions to match the training distribution. We evaluate two variants: MolMinerD, which samples conditions directly from the dataset, and MolMinerS, which samples conditions from the GMM. We benchmark against HierVAE (Jin et al., 2020) an unconditional model, which is the most comparable in terms of generation strategy and architectural design. We exclude MARS (Xie et al., 2021), as it accesses ground-truth molecular properties during generation to guide sampling. Specifically, 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 MARS evaluates properties such as QED, SA, or activity scores on-the-fly for proposed molecules and uses these values to shape the acceptance probability in a Markov Chain Monte Carlo (MCMC) loop. This fundamentally differs from our approach, in which molecules are generated solely from prompted (i.e., user-specified or sampled) properties, without access to oracle evaluations at inference time. As such, a direct comparison would be misleading in practical scenarios like highthroughput screening, where true property values are unavailable during generation. We also experimented with MoLeR (Maziarz et al., 2024), using the official implementation and training configuration. The model was run for seven days on an NVIDIA RTX 3090 GPU, completing two 5,000-step validation intervals ("mini-epochs") as defined in the authors' logging protocol. Molecules sampled from the latent prior were often chemically implausible and showed poor alignment with training property distributions. These results are consistent with known limitations of VAE-based molecular models—particularly the mismatch between prior and posterior distributions—and with previously reported decoding issues in MoLeR2. We therefore exclude MoLeR from our main quantitative comparisons but include these results in the Appendix A.9. Additional sampling strategy comparisons for MolMiner are provided in Appendix A.7, along with kernel density plots of the generated property distributions for visual reference. Table 1: Wasserstein distances between the property distributions of generated molecules (N ≈ 5,000) and the reference dataset are reported, along with uniqueness (%), novelty (%), and mean Tanimoto distance, for HierVAE and Molminer in two different sampling approaches.

$$\begin{array}{l}{{\frac{\pi}{6}}}\\ {{\frac{\pi}{2}}}\\ {\approx}\end{array}$$
$\frac{\pi}{4}$. 

| TPSA      | MR   | HBD   | HBA   | #Rings   | #RotBonds   | #Chiral   | %    |      |      |      |      |      |     |      |      |
|-----------|------|-------|-------|----------|-------------|-----------|------|------|------|------|------|------|-----|------|------|
| molWt     |      |       |       |          |             |           |      |      |      |      |      |      |     |      |      |
| P3 tCS    |      |       |       |          |             |           |      |      |      |      |      |      |     |      |      |
| gP        | QED  | SAS   | Frac  |          |             |           |      |      |      |      |      |      |     |      |      |
| Model     | lo   |       |       |          |             |           |      |      |      |      |      |      |     |      |      |
| HierVAE   | 0.26 | 0.01  | 0.13  | 0.03     | 15          | 2.3       | 3.8  | 0.08 | 0.20 | 0.39 | 0.33 | 0.08 | 100 | 99.9 | 0.88 |
| MolMinerD | 0.31 | 0.01  | 0.07  | 0.02     | 47          | 7.6       | 11.9 | 0.14 | 0.36 | 0.41 | 0.64 | 0.19 | 99  | 99.5 | 0.89 |
| MolMinerS | 0.46 | 0.02  | 0.09  | 0.02     | 65          | 10.9      | 16.3 | 0.16 | 0.56 | 0.59 | 0.88 | 0.26 | 98  | 99.8 | 0.89 |

Our model performs slightly below HierVAE in unconditional generation, with modest differences across most properties. For further analysis we refer to Fig. 15 illustrating the full distributions and a more detailed comparison. The largest gaps—observed in molecular weight, TPSA, and molar refractivity—are partly attributable to approximation error in GMM-based conditioning. While this explains some degradation from MolMinerD to MolMinerS, it does not fully account for the gap. Crucially, MolMiner is optimized for conditional generation, where it enables flexible, multiproperty control. We now evaluate its performance in that setting.

## 4.3 Benchmarking Conditional Generation

To evaluate conditional generation, we measure how accurately the model produces molecules that match specified target property values. For each of the twelve physicochemical and structural properties, we uniformly sample target values across the range µ ± 2σ, based on their empirical distributions in the dataset. The remaining eleven properties are sampled conditionally from the GMM prior, and the full twelve-dimensional vector is used to guide generation. This process is repeated 30 times per target value, enabling a robust estimate across the full range of each property. Calibration plots compare the prompted (target) values with the properties predicted from the generated molecules. For continuous properties, we show mean trends with ±1 standard deviation bands; for discrete properties, we report confusion matrices. This setup evaluates how faithfully the model responds to conditioning across the entire dynamic range of each property, providing insight into its capacity for simultaneous, multi-property control.

As shown in Figure 2, the model achieves calibrated conditional generation for most of the twelve properties. QED is a notable exception, where control accuracy degrades. In addition, molWt and MR exhibit systematic deviations, consistent with their performance in the unconditional benchmark, suggesting areas for further improvement. Overall, to our knowledge, this is the first model to support simultaneous conditioning across as many as twelve molecular properties—representing a significant advance in controllable molecular design.

2https://github.com/microsoft/molecule-generation/issues/77 7 378 379 380 31 382 383 384 385 386 387 38 389 39 391 392 393 394 395 39 397 39 39 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

## 5 Limitations

While MolMiner demonstrates strong performance in conditional generation and introduces several architectural innovations, certain limitations remain. Notably, the model underperforms its predecessor in unconditional generation for some properties, particularly molecular weight, MR, and TPSA. We hypothesize that this arises from a tendency to terminate rollouts early, producing slightly smaller molecules on average. This behavior stems from an imbalance in the training data: the orderagnostic rollouts used in MolMiner contain a higher proportion of termination actions than in prior models, potentially biasing the model toward early termination. This effect likely contributes to the systematic deviations observed in the calibration plots, especially for molecular weight. Addressing this may require balancing termination actions during rollout sampling or introducing reinforcement learning based fine-tuning to better calibrate the model's termination policy.

## 432

433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 6 Conclusion

All the models in this work were trained using PyTorch 2.5.0 on a NVIDIA RTX3090. Training these models took approximately 7 days, or 30 epochs, using a batch size of 256 with AdamW as the optimizer, and RAM usage of 70 GB.

## 8 Code And Data Availability

We introduce MolMiner, a novel generative model for inverse molecular design that is autoregressive, fragment-based, geometry-aware, and order agnostic. Crucially, MolMiner supports conditional generation on up to twelve key molecular properties, including logP, QED, SAS, FractionCSP3, molecular weight, TPSA, molar refractivity, hydrogen bond donors and acceptors, ring count, rotatable bonds, and chiral centers.

We show that MolMiner enables controllable and calibrated generation across most of these properties. To make the process more flexible and user-friendly, we introduce a GMM that allows users to specify any subset of properties while the remaining values are sampled conditionally. In unconditional benchmarks, MolMiner performs comparably to existing models across many properties, though some—particularly molecular weight, TPSA, and molar refractivity—still exhibit systematic deviation. More importantly, the model demonstrates strong performance in the more challenging setting of conditional generation. To our knowledge, this is the first model to unify the following capabilities within a single generative framework: (A) Dynamic incorporation of 3D molecular geometry during autoregressive generation, (B) A symmetry-aware protocol for fragment attachment, (C) Order-agnostic rollout with demonstrated regularization benefits, (D) Scalable, high-dimensional conditional generation using a GMM-based prior. Together, these contributions advance the state of controllable molecular generation and lay the foundation for more interpretable, flexible, and accessible tools for molecular design. Beyond methodological advances, MolMiner has the potential to accelerate discovery in domains of high environmental and biomedical relevance. By enabling inverse design of molecules with precise control over structural and physicochemical properties, our model could assist in the development of next-generation materials for sustainable energy storage and conversion —such as organic redox flow batteries and organic photovoltaics— facilitate early-stage drug discovery, and support green chemistry initiatives aimed at more environmentally responsible molecular design.

## 7 Computational Requirements

All code, model checkpoints, and processed data used in this work are available at https:// github.com/xxxx. This includes the ZINC subset with computed properties, dataset splits, training scripts, and evaluation tools needed to reproduce all experiments.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

G. Richard Bickerton, Gaia V. Paolini, Jer´ emy Besnard, Sorel Muresan, and Andrew L. Hopkins. ´
Quantifying the chemical beauty of drugs. *Nature Chemistry*, 4(2):90–98, Feb 2012. ISSN 17554349. doi: 10.1038/nchem.1243. URL https://doi.org/10.1038/nchem.1243.

Hesam Dashti, William M. Westler, John L. Markley, and Hamid R. Eghbalnia. Unique identifiers for small molecules enable rigorous labeling of their atoms. *Scientific Data*, 4(1):170073, May 2017. ISSN 2052-4463. doi: 10.1038/sdata.2017.73. URL https://doi.org/10.1038/ sdata.2017.73.

Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics, 1(1):8, Jun 2009. ISSN 1758-2946. doi: 10.1186/1758-2946-1-8. URL https: //doi.org/10.1186/1758-2946-1-8.

Peter Ertl, Bernhard Rohde, and Paul Selzer. Fast calculation of molecular polar surface area as a sum of fragment-based contributions and its application to the prediction of drug transport properties. *Journal of Medicinal Chemistry*, 43(20):3714–3717, 2000. doi: 10.1021/jm000942e. URL https://doi.org/10.1021/jm000942e. PMID: 11020286.

Niklas W. A. Gebauer, Michael Gastegger, Stefaan S. P. Hessmann, Klaus-Robert Muller, and ¨
Kristof T. Schutt. Inverse design of 3d molecular structures with conditional generative neu- ¨ ral networks. *Nature Communications*, 13(1):973, Feb 2022. ISSN 2041-1723. doi: 10.1038/ s41467-022-28526-y. URL https://doi.org/10.1038/s41467-022-28526-y.

Rafael Gomez-Bombarelli, Jennifer N. Wei, David Duvenaud, Jos ´ e Miguel Hern ´ andez-Lobato, ´
Benjam´ın Sanchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, ´ Ryan P. Adams, and Alan Aspuru-Guzik. Automatic chemical design using a data-driven con- ´ tinuous representation of molecules. *ACS Central Science*, 4(2):268–276, 2018. doi: 10.

1021/acscentsci.7b00572. URL https://doi.org/10.1021/acscentsci.7b00572.

PMID: 29532027.

Stephen R. Heller, Alan McNaught, Igor Pletnev, Stephen Stein, and Dmitrii Tchekhovskoi. Inchi, the iupac international chemical identifier. *Journal of Cheminformatics*, 7(1):23, May 2015.

ISSN 1758-2946. doi: 10.1186/s13321-015-0068-4. URL https://doi.org/10.1186/ s13321-015-0068-4.

Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=Lm8T39vLDTE.

Emiel Hoogeboom, Victor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant diffu- ´
sion for molecule generation in 3d, 2022b.

John J. Irwin, Teague Sterling, Michael M. Mysinger, Erin S. Bolstad, and Ryan G. Coleman. Zinc:
A free tool to discover chemistry for biology. Journal of Chemical Information and Modeling, 52(7):1757–1768, 2012. doi: 10.1021/ci3001277. URL https://doi.org/10.1021/
ci3001277. PMID: 22587354.

J. Jensen. Om konvekse funktioner og uligheder imellem middelværdier. Nyt tidsskrift for matematik, 16:49–68, 1905. ISSN 09093524. URL http://www.jstor.org/stable/ 24528332.

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation, 2019.

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Hierarchical generation of molecular graphs using structural motifs, 2020.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL
https://arxiv.org/abs/1412.6980.

W. Kohn and L. J. Sham. Self-consistent equations including exchange and correlation effects.

Phys. Rev., 140:A1133–A1138, Nov 1965. doi: 10.1103/PhysRev.140.A1133. URL https: //link.aps.org/doi/10.1103/PhysRev.140.A1133.

Greg Landrum. Rdkit: Open-source cheminformatics software, 2024. http://www.rdkit.

org/.

Jaechang Lim, Seongok Ryu, Jin Woo Kim, and Woo Youn Kim. Molecular generative model based on conditional variational autoencoder for de novo molecular design. *Journal of Cheminformatics*,
10(1):31, Jul 2018. ISSN 1758-2946. doi: 10.1186/s13321-018-0286-7. URL https://doi. org/10.1186/s13321-018-0286-7.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https:
//arxiv.org/abs/1711.05101.

Krzysztof Maziarz, Henry Jackson-Flux, Pashmina Cameron, Finton Sirockin, Nadine Schneider, Nikolaus Stiefl, Marwin Segler, and Marc Brockschmidt. Learning to extend molecular scaffolds with structural motifs, 2024. URL https://arxiv.org/abs/2103.03864.

Geoffrey McLachlan. Finite mixture models. *A wiley-interscience publication*, 2000. Noel M. O'Boyle, Michael Banck, Craig A. James, Chris Morley, Tim Vandermeersch, and Geoffrey R. Hutchison. Open babel: An open chemical toolbox. Journal of Cheminformatics, 3(1):33, Oct 2011. ISSN 1758-2946. doi: 10.1186/1758-2946-3-33. URL https:
//doi.org/10.1186/1758-2946-3-33.

Raul Ortega Ochoa, Bardi Benediktsson, Renata Sechi, Peter Bjørn Jørgensen, and Arghya Bhowmik. Materials funnel 2.0 - data-driven hierarchical search for exploration of vast chemical spaces. *J. Mater. Chem. A*, 11:26551–26561, 2023. doi: 10.1039/D3TA05860C. URL http://dx.doi.org/10.1039/D3TA05860C.

Raul Ortega-Ochoa, Alan Aspuru-Guzik, Tejs Vegge, and Tonio Buonassisi. A tomographic inter- ´
pretation of structure-property relations for materials discovery, 2025. URL https://arxiv. org/abs/2501.18163.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Igor Pletnev, Andrey Erin, Alan McNaught, Kirill Blinov, Dmitrii Tchekhovskoi, and Steve Heller.

Inchikey collision resistance: an experimental testing. *Journal of Cheminformatics*, 4(1):39, Dec 2012. ISSN 1758-2946. doi: 10.1186/1758-2946-4-39. URL https://doi.org/10.1186/ 1758-2946-4-39.

Daniil Polykovskiy, Alexander Zhebrak, Benjam´ın Sanchez-Lengeling, Sergey Golovanov, Oktai ´
Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, Artur Kadurin, Sergey I. Nikolenko, Alan Aspuru-Guzik, and Alex Zhavoronkov. ´ Molecular sets (MOSES): A benchmarking platform for molecular generation models. *CoRR*,
abs/1811.12823, 2018. URL http://arxiv.org/abs/1811.12823.

A. K. Rappe, C. J. Casewit, K. S. Colwell, W. A. III Goddard, and W. M. Skiff. Uff, a full periodic table force field for molecular mechanics and molecular dynamics simulations. *Journal of the* American Chemical Society, 114(25):10024–10035, 1992. doi: 10.1021/ja00051a040. URL
https://doi.org/10.1021/ja00051a040.

David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5):742–754, 2010. doi: 10.1021/ci100050t. URL https://doi. org/10.1021/ci100050t. PMID: 20426451.

Nadine Schneider, Roger A. Sayle, and Gregory A. Landrum. Get your atoms in order—an opensource implementation of a novel and robust molecular canonicalization algorithm. *Journal of* Chemical Information and Modeling, 55(10):2111–2120, 2015. doi: 10.1021/acs.jcim.5b00543. URL https://doi.org/10.1021/acs.jcim.5b00543. PMID: 26441310.

Kristof T. Schutt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre ¨
Tkatchenko, and Klaus-Robert Muller. Schnet: A continuous-filter convolutional neural network ¨ for modeling quantum interactions, 2017. URL https://arxiv.org/abs/1706.08566.

Ahsan Shehzad, Feng Xia, Shagufta Abid, Ciyuan Peng, Shuo Yu, Dongyu Zhang, and Karin Verspoor. Graph transformers: A survey, 2024. URL https://arxiv.org/abs/2407. 09777.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014. URL http://jmlr.org/papers/v15/ srivastava14a.html.

T.T. Tanimoto. *An Elementary Mathematical Theory of Classification and Prediction*. International Business Machines Corporation, 1958. URL https://books.google.dk/books?id= yp34HAAACAAJ.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647

## A Appendix A.1 Calculated Properties For Controlled Generation

Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In Eric P.

Xing and Tony Jebara (eds.), Proceedings of the 31st International Conference on Machine Learning, volume 32 of *Proceedings of Machine Learning Research*, pp. 467–475, Bejing, China, 22–24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/uria14.html.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. URL https://arxiv. org/abs/1706.03762.

Sergei Voloboev. A review on fragment-based de novo 2d molecule generation, 2024. URL https:
//arxiv.org/abs/2405.05293.

David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. *Journal of Chemical Information and Computer Sciences*, 28(1):31–36, 1988. doi: 10.1021/ci00057a005. URL https://doi.org/10.1021/ci00057a005.

David Weininger, Arthur Weininger, and Joseph L. Weininger. Smiles. 2. algorithm for generation of unique smiles notation. *J. Chem. Inf. Comput. Sci.*, 29:97–101, 1989.

Julia Westermayr, Joe Gilkes, Rhyan Barrett, and Reinhard J. Maurer. High-throughput propertydriven generative design of functional organic molecules. *Nature Computational Science*, 3(2): 139–148, Feb 2023. ISSN 2662-8457. doi: 10.1038/s43588-022-00391-1. URL https:// doi.org/10.1038/s43588-022-00391-1.

Scott A. Wildman and Gordon M. Crippen. Prediction of physicochemical parameters by atomic contributions. *J. Chem. Inf. Comput. Sci.*, 39:868–873, 1999. URL https://api. semanticscholar.org/CorpusID:15271440.

Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusionbased molecule generation with informative prior bridges. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 36533–36545. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/ file/eccc6e11878857e87ec7dd109eaa9eeb-Paper-Conference.pdf.

Yutong Xie, Chence Shi, Hao Zhou, Yuwei Yang, Weinan Zhang, Yong Yu, and Lei Li. Mars:
Markov molecular sampling for multi-objective drug discovery, 2021. URL https://arxiv. org/abs/2103.10432.

- **logP**: Logarithm of water partition coefficient, used as a measure of lipophilicity.

This work included twelve annotated molecular properties for the compounds in the dataset calculated using RDKit version 2024.3.5, whose statistics for the dataset used in this work are summarized in Table 2.