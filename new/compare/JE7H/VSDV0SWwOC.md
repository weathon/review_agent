---
job_id: a4eabd9e-14ce-463b-9857-e629ff5d95ec
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: VSDV0SWwOC.pdf
paper: LS-Merge: Merging Language Models in Latent Space
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely on representation learning for language models, generative latent models over weights, and model merging, which fits ICLR’s core topics (representation learning, generative models, transfer/model reuse, optimization over weight spaces).

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Method, Experiments/Results, Discussion, Conclusion). The method is technically nontrivial, empirical evaluations are fairly extensive across several LLMs and LoRA experts, and there are no obvious fatal theoretical or experimental flaws, though there are points needing clarification and several weaknesses.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, instructions to reviewers, or manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes LS-Merge, a framework that learns a latent space over LLM weights using a transformer-based VAE and performs model merging in latent space rather than directly in parameter space. The method supports both homogeneous (same architecture) and heterogeneous (different depth/width/family) merges via layer-wise chunking, a two-stage VAE training curriculum, and an optimal-transport-based alignment between latent distributions. Experiments on Gemma, LLaMA, and LoRA experts show that latent-space merging can match or outperform standard weight-space merging and some representation-merging baselines, and can handle cross-architecture merges that are difficult in raw weight space.

## Strengths

1. **Clear high-level idea and solid conceptual framing.**  
   Moving merging operations from raw weight space into a learned latent space is a clean conceptual step. The framework in **Figure 1** (encode → latent align/merge → decode) is well articulated and makes it easy to see how self-merging, expert fusion, and heterogeneous merging are all special cases of the same pipeline.

2. **Thoughtful analysis of LLM weight statistics informing design.**  
   Section 3.1 and **Table 1** + **Table 9** provide a detailed empirical study of mean/variance/skewness/kurtosis across attention and MLP layers in Gemma and LLaMA. The observation that self-attention weights exhibit high excess kurtosis and heavier tails than MLP weights, together with the PCA spectra in **Figure 2** and **Figure 6**, provides a good empirical justification for (i) not assuming simple Gaussian structure, and (ii) using expressive, tail-preserving encoders. This is more convincing than the usual hand-wavy “weights might be compressible” argument.

3. **Technically interesting OT-based latent alignment for heterogeneous merging.**  
   For cross-architecture merging, the paper does not stop at matching latent dimensionality. It uses 2-Wasserstein optimal transport under a Gaussian approximation to derive a closed-form affine map aligning means and covariances of source and target latent distributions (Eq. (2) and surrounding text). **Figure 3** and **Figure 9b** nicely visualize why this matters: Gemma and LLaMA latents lie on disjoint manifolds, so naive interpolation would fall off the target decoder’s manifold. The OT alignment is a principled and relatively lightweight solution.

4. **Strong evidence that the latent manifold is genuinely non-linear.**  
   Section 5.3 and **Table 8** provide a very nice diagnostic: PCA vs. Transformer-VAE reconstruction at several compression ratios. PCA reconstructions collapse performance to near-random on MMLU and other tasks even at mild compression (e.g., MMLU 25.5% vs 41.44% base at \(r=1.6\)), while VAE reconstructions retain almost all performance (39.89% at \(r=1.6\)). This is a compelling argument that the “functional” weight manifold is significantly curved and not well approximated by any linear subspace.

5. **Empirical gains on expert fusion and comparison to strong baselines.**  
   For LoRA experts, **Table 3** shows that LS-Merge variants outperform a strong collection of weight-space baselines (Uniform/Greedy Soup, SLERP, Dare-Ties, Data Merge) on most benchmarks, often by a nontrivial margin (e.g., HellaSwag: 60.1 vs 54.6; MMLU: 56.0 vs 52.5–52.5, etc.). For merging fine-tuned LLaMA-2-13B models, **Table 4** demonstrates that latent weight-space merging is competitive with AIM (activation-informed merging) and clearly better than Task Arithmetic. This positions LS-Merge not just as a new idea but as practically competitive with specialized merging methods.

6. **Evidence that latent-space self-merging can modestly improve a single model.**  
   **Table 2** shows that for both Gemma-3-4B-it and especially Gemma-3-1B-it, “self-merging” multiple latent samples gives consistent improvements over the base model and a single-sample VAE reconstruction (e.g., MMLU-pro 7.1 → 10.3 on Gemma-3-1B-it). **Figure 5a** and **Figure 5b** (on MMLU/MMLU-Pro) further visualize that carefully tuned latent mixing yields gains. While not dramatic, this is an intriguing indication that exploring a single model’s latent neighborhood can enhance performance.

7. **Heterogeneous merging experiments show real, not just toy, benefits.**  
   Section 4.4 and **Figure 4** + **Table 5** provide results for both intra-family (Gemma-3-4B-it → Gemma-3-1B-it) and cross-family (LLaMA-3.2-1B → Gemma-3-1B-it) merges. OT+interpolation clearly outperforms OT-only or no-alignment interpolations (e.g., WinoGrande 56.83 → 57.75, ARC-C 42.78 → 43.34) for small mixing coefficients. This supports the claim that alignment is not cosmetic but actually improves downstream performance.

8. **Nontrivial ablations revealing useful insights.**  
   - **Table 6** (MLP vs Attention vs both) shows that merging attention-only hurts performance, MLP-only provides limited gains, and combining both gives the best performance. This is a useful, mechanistic insight on which submodules actually carry complementary knowledge.  
   - **Table 7** systematically studies VAE generalization vs. compression ratio \(r\) across unseen models, making concrete the trade-off between compression and out-of-distribution generalization.  
   - **Figure 8** tracks reconstruction error vs. training for different compression ratios and shows that the two-stage (AE → VAE) curriculum stabilizes training.

9. **Implementation details and reproducibility.**  
   The paper includes reasonably detailed architectural and training hyperparameters (e.g., **Table 10**, compression ratios, chunk size, sequence length, number of layers, optimizer settings) and algorithmic pseudo-code (**Algorithm 1**, **Algorithm 2**), and promises open-sourcing, which should make reproduction feasible after some clarifications.

## Weaknesses

1. **Positioning vs closely related model- and latent-space merging work is incomplete.**  
   Beyond weight-space soups and some representation-merging methods, the paper does not adequately engage with several directly relevant recent works on merging LLMs in various latent spaces or with explicit alignment:

   - There is no discussion of latent or semantic alignment based model merging such as:
     - Kim & Lee, “Latent Merging: Dynamic and Reversible Composition of Large Language Models” (merging in hidden-representation space).  
     - Gu et al., “SeMe: Training-Free Language Model Merging via Semantic Alignment.”  
     - Roy et al., “AlignMerge – Alignment-Preserving Large Language Model Merging via Fisher-Guided Geometric Constraints.”  
     - Chen et al., “Can Heterogeneous Language Models Be Fused?” which tackles heterogeneous LMs with topology-based alignment.  
     - Reza et al., “SSAM: Singular Subspace Alignment for Merging Multimodal Large Language Models.”  

   These are not cited anywhere in the text, and several address almost the same “heterogeneous merging” problem with other alignment strategies (semantic/topological/Fisher/subspace vs OT). Without a clear comparison in Section 2 and 4, it is difficult to judge how LS-Merge’s contributions compare in novelty and significance; some of the claimed uniqueness on cross-architecture merging and alignment comes across as overstated.

2. **Some key methodological details are underspecified or internally inconsistent.**  
   Several aspects of the method are described at a high level but lack precise definitions, which will inhibit faithful replication and obscure what exactly is being optimized:

   - **Compression ratio \(r\).** The paper repeatedly uses ratios like \(r=1.6,2,4\) (Sections 4.1, 5.2, 5.3; **Table 7**, **Table 8**) but never gives a precise mathematical definition. It seems to be the ratio of original weight dimensionality to latent dimension per chunk, but then **Section B.1** talks about “expansion \(r<1\)” without clarifying how \(r\) is computed. A precise formula such as  
     \[
     r = \frac{\text{dim}(w)}{\text{dim}(z)}
     \]
     or per-layer/chunk equivalent would be needed to interpret the trade-offs in Tables 7–8.

   - **Latent dimensionality mapping for heterogeneous models.** In Section 3.3, the “proportional mapping” defines  
     \[
     r = \frac{n_t N}{n_s M},\quad Z^{(\text{src.mapped})}\in\mathbb{R}^{n_t\times d},\quad
     Z^{(\text{tgt})}\in\mathbb{R}^{n_t\times d},
     \]
     but it is unclear how exactly per-layer latents of shapes \((n_s,M)\) and \((n_t,N)\) are transformed into \(n_t\) layers of dimension \(d\). Are some layers averaged, split, or subsampled? Is there a mapping from source layers to target layers beyond “proportional capacity”? This is crucial to reproduce cross-depth/width merges.

   - **Confusing notation between algorithms and main text.**  
     In Section 3.3, interpolation is defined with \(z_\lambda = (1-\lambda) z_a + \lambda z_b\). In **Algorithm 1** and **Algorithm 2**, the merge step suddenly uses “\(\lambda\)” and later “\(\beta\)” (line 10 of Algorithm 2: \(z^{(1)}_{\text{merged}} \leftarrow z^{(1)} + \beta(z^{(1)}_{\text{align}} - z^{(1)}_{\text{tgt}})\)). The relationship between \(\lambda\) and \(\beta\) is not explained, and the two algorithms differ slightly in notation and steps. It is unclear which one corresponds to the experiments in Section 4.4 and **Table 5**.

   - **VAE objective notation.** In Eq. (1) the loss is written as
     \[
     \mathcal{L} = -\mathbb{E}_{q_\phi(z\mid w)}[\log p_\theta(w\mid z)] + \beta \mathrm{KL}(q_\phi(z\mid w)\Vert p(z)).
     \]
     But earlier the encoder is denoted \(E_\theta\) and the decoder \(D_\phi\). Here \(\theta\) and \(\phi\) are apparently swapped. This mismatch may sound minor, but with a large, nontrivial VAE architecture it matters to know which part is stochastic and where the KL applies.

   Overall, the paper needs more careful treatment of notation and explicit formulas to lift the current “recipe-style” description into a rigorous method specification.

3. **The OT alignment formulation vs. implementation is ambiguous.**  
   Section 3.3 derives the 2-Wasserstein OT map between Gaussians,
   \[
   T^*(z_{\text{src}}) = \mu_t + A (z_{\text{src}} - \mu_s)
   \quad\text{with}\quad
   A = \Sigma_s^{-1/2}(\Sigma_s^{1/2} \Sigma_t \Sigma_s^{1/2})^{1/2} \Sigma_s^{-1/2},
   \]
   but then states “In practice, we use existing OT library from Flamary et al. (2021; 2024) in our work.” It remains unclear whether they actually use the closed-form Gaussian map (which does not require a numerical OT solver) or estimate a discrete OT transport plan and regress an affine map from it. Section C further complicates the picture with Algorithm 2’s OT step and a separate “OT only” baseline in **Table 5**, but does not say whether that baseline applies the affine \(T^*\) directly or uses the library’s discrete plan. This ambiguity matters because the Gaussian approximation is a strong assumption on the latent distributions; if they instead use full entropic OT, the theoretical discussion around Eq. (2) is somewhat misleading.

4. **Some of the theoretical discussion in Section 3.1 is loose and symbolically inconsistent.**  
   The “Theoretical Compressibility” subsection tries to ground the use of VAEs in manifold theory and the Eckart–Young theorem, but there are issues:

   - The text references a projection map dimension bound
     \[
     k = O\Big(\frac{d}{\sqrt{s}}\log\frac{V}{\varepsilon}\Big) \ll D
     \]
     with \(V\) the manifold volume and “\(\tau\) its reach”, yet the variable \(s\) appears in the formula without any definition. From Lahiri et al. (2016) and related results, the dependence usually involves reach \(\tau\), ambient dimension, and curvature bounds. As written, the bound is mathematically opaque and cannot be checked.

   - The manifold \( \mathcal{M} \subset \mathbb{R}^D\) is assumed “smooth \(d\)-dimensional” but there is no explicit assumption on how many checkpoints or layers are needed to empirically approximate such a manifold. Since the method is trained on a small finite set of pretrained checkpoints, the theoretical narrative risks overselling the guarantees without connecting to the finite-sample setting.

   These are not fatal flaws, but they undermine the rigor of the theory section. A clearer statement such as “there exist random projections preserving pairwise distances on manifolds (citing exact theorems) and a VAE can approximate these” without the unexplained big-O bound would be more honest.

5. **Empirical improvements in some main scenarios are modest and sometimes within variance.**  
   While **Table 3**, **Table 4**, and **Table 8** show strong, convincing gains, some other core results are comparatively small:

   - In **Table 2** (self-merging on Gemma-3-4B-it and 3-1B-it), LS-Merge improves MMLU from 53.1 → 54.2 and GSM8K 29.9 → 32.2. There is no comparison against weight-space self-merging baselines (e.g., sampling checkpoints along a training trajectory, or adding noise in weight space) to show that the latent sampling per se is crucial. It could be that applying mild structured noise to weights yields similar or larger gains.

   - For cross-family merging (**Table 5**), improvements are on the order of 0.5–1.5 points. These are meaningful but relatively small compared to the overhead of training and running a 200M-parameter VAE. A stronger case would require showing that weight-space or semantic/subspace-based heterogeneous merging (where available) fail badly on the same pairs.

   - In **Figure 4a** and **Figure 4b** (Gemma-3-4B → 3-1B), the best gains occur for very small mixing coefficients (\(\lambda\in[0.05, 0.2]\)), and quickly plateau or regress as \(\lambda\) grows. This is sensible, but it highlights that LS-Merge is injecting only a small amount of source information, which limits the “capacity transfer” story.

6. **Limited coverage of merging scenarios and scale, given the ambition of the method.**  
   The method is advertised as a general recipe for merging “billion-parameter LLMs including cross-family cases”. However, the experiments mostly use 1B–4B Gemma and 1B LLaMA checkpoints, plus LoRA experts on 7B-scale models. There is a single LLaMA-2-13B experiment, but LS-Merge is applied only after fine-tuning on a few domains. It is unclear whether the method remains stable and computationally practical for more challenging settings, such as merging two full 70B-class models or multiple 13B–34B models. Given the cost of training the VAE (~200M parameters, several hours on a 6000 Ada GPU), readers will want a more explicit analysis or at least a discussion of scaling behavior and failure modes.

7. **Cost/benefit trade-off vs simple weight-space baselines is under-explored.**  
   The paper acknowledges in Section 6 that training VAEs at high compression ratios is challenging and that overcomplete latents are acceptable. However:

   - There is no quantitative comparison of wall-clock or energy cost between LS-Merge and simple baselines such as Uniform/Greedy Soup or SLERP when applied to the same number of models. For many applications, a small performance gain may not justify training a large VAE and running a full encode/decode over all layers.

   - The “expert merging” experiment in **Table 3** uses 10 LoRA experts. It would be informative to see how performance and runtime scale as the number of experts increases (e.g., 20, 50), since latent merging presumably scales linearly in the number of experts, like soups.

   Without a more explicit cost/benefit story, it is hard to assess where LS-Merge is the right tool vs an interesting but overkill alternative.

8. **Some experimental design choices reduce comparability and clarity.**  
   A few aspects of the evaluation could be better justified or standardized:

   - The paper uses two different evaluation toolchains (a custom subset evaluation and lm-eval) across different sections, and even mentions “some issues with llama model when using the previous evaluation code”. This raises questions about how comparable different tables are, and whether the anomalies were fully resolved.

   - For LoRA expert merging (**Table 3**), the baselines are exclusively weight-space methods (plus Data Merge). It would be interesting to see AIM or other representation-based methods on the same setup, since the LS-Merge variant there is also a kind of representation merging, albeit in weight-latent space.

   - The expert merging narrative claims advantages from “sampling multiple latent codes per expert”, but there is no ablation quantifying how much improvement is due to stochastic sampling vs a single deterministic encoding per expert.

9. **Minor clarity issues and typos.**  
   - **Algorithm 1** has several typos that make lines confusing (e.g., “ltrc” instead of “lsrc”, “wtrc, wgt, wgt” in the same line, inconsistent arrows `<` instead of `←`).  
   - Equation references mix “Algorithm 1 / algorithm 2” and the OT map is said to be summarized in “algorithm 2” in the main text, but Algorithm 1 is the one actually in the main section; Algorithm 2 appears only in the appendix.  
   - A few sentences in Section 3.2’s description of chunking and token-wise vs pooled latents are garbled (“(p o o l e d o v e r t o k e n s)” with spaces), likely due to formatting, but this still hinders precise understanding.

   These are not fatal, but for a method as intricate as LS-Merge, small notational slips can compound into real ambiguity.

## Potentially Missing Related Work

1. **Kim, J. S., Lee, S. (2025). “Latent Merging: Dynamic and Reversible Composition of Large Language Models.”**  
   - Directly related because it also composes LLMs in a latent space (of hidden representations rather than weights), with an emphasis on reversibility and dynamic composition.  
   - Should be discussed in **Section 2 (Related Work)** as an alternative form of latent-space merging for LMs, contrasting hidden-representation composition with weight-latent composition, and possibly referenced in the discussion on self-merging and dynamic reuse.

2. **Chen, S., Zhou, J., Chen, Q. (2026). “Can Heterogeneous Language Models Be Fused?”**  
   - Addresses the same high-level question as LS-Merge regarding fusing heterogeneous LMs, using topology-based alignment and conflict-aware denoising.  
   - Should be cited in **Section 2** and compared to LS-Merge’s OT-based latent alignment in **Section 4.4**; a short discussion after Eq. (2) could highlight differences in the alignment assumptions and costs.

3. **Reza, M. K., Patil, A., Ayrapetian, E. (2026). “SSAM: Singular Subspace Alignment for Merging Multimodal Large Language Models.”**  
   - Proposes training-free subspace alignment for merging multimodal LMs; closely related to the idea of aligning latent/parameter spaces across models of different modalities or architectures.  
   - Should be mentioned in **Section 2** as another alignment-based merging approach and could be briefly discussed alongside LS-Merge’s PCA vs VAE results in **Section 5.3**, since SSAM also exploits low-rank structure.

4. **Gu, J., Aleti, A., Chen, C. (2025). “SeMe: Training-Free Language Model Merging via Semantic Alignment.”**  
   - Uses semantic (representation-based) alignment to merge language models without training, which is highly relevant in spirit to LS-Merge’s OT-based latent alignment.  
   - Should be added to the discussion of activation / representation level merging methods in **Section 2** and **Section 4.3**, and potentially considered as an additional baseline where feasible.

5. **Roy, A., Patel, J., Chadha, A. (2025). “AlignMerge – Alignment-Preserving Large Language Model Merging via Fisher-Guided Geometric Constraints.”**  
   - Introduces a geometry-aware merging method preserving alignment via Fisher information, closely related to LS-Merge’s discussion of preserving manifold structure in latent space.  
   - Should be compared in **Section 2** and in the discussion of OT alignment in **Section 3.3**, emphasizing pros/cons of Fisher-based vs OT-based geometry preservation.

6. **Li, C., Gao, X., Li, Y. (2020). “Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space.”**  
   - A large-scale VAE-based latent model for language, directly relevant to the idea of using VAEs with transformers in the language domain.  
   - Could be referenced in **Section 3.2** where the transformer-VAE architecture is introduced, as an example of prior successful large-scale VAEs for language, and in the related work on generative latent models.

Including and briefly contrasting these works would significantly strengthen the paper’s positioning and clarify what is genuinely new in LS-Merge.

## Questions

1. **Exact definition of compression ratio \(r\).**  
   Can you provide a precise definition of the compression ratio \(r\) used in Tables 2, 7, and 8 and in Section B.1, including how it is computed per chunk/layer and how “expansion (\(r<1\))” is defined in the heterogeneous setting?

2. **Details of the layer mapping for heterogeneous merges.**  
   In Section 3.3, you define \(r = \frac{n_t N}{n_s M}\) and say that per-layer latents are proportionally mapped to a fixed dimension \(d\). Concrete questions:
   - How do you map source layers to target layers when their counts differ (e.g., Gemma-3-4B vs Gemma-3-1B)?  
   - Is there a specific rule (e.g., contiguous blocks, interpolation over depth, attention/MLP treated differently), or do you simply truncate or repeat layers?  
   - How sensitive are the results in **Figure 4** and **Table 5** to this mapping?

3. **OT map implementation vs Gaussian closed form.**  
   Are your experiments using the closed-form Gaussian OT map
   \[
   T^*(z) = \mu_t + A(z-\mu_s), \quad
   A = \Sigma_s^{-1/2}(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2})^{1/2}\Sigma_s^{-1/2},
   \]
   or are you computing discrete transport plans with POT and then approximating a map? If you use the closed form, why is POT needed at all? Clarifying this would help interpret the “OT only” vs “OT+interp” results in **Table 5**.

4. **Comparison with training-free / semantic alignment methods.**  
   Given the existence of training-free semantic and geometric alignment methods (e.g., SeMe, AlignMerge, SSAM), can you comment on:
   - How LS-Merge’s performance compares to such methods in heterogeneous merging scenarios like Gemma ↔ LLaMA?  
   - Whether LS-Merge could reuse ideas from these works in the latent space (e.g., aligning singular subspaces or semantics in latent instead of raw parameter space)?

5. **Sensitivity to the number of latent samples per expert.**  
   For LoRA expert merging in **Table 3**, you mention sampling multiple latent codes per expert to increase robustness. Could you provide an ablation where you vary the number of samples per expert (e.g., 1, 4, 16) and show the effect on accuracy and variance? It would help quantify the benefit of exploiting the VAE’s posterior vs a single deterministic embedding.

6. **Scalability beyond 7B-scale models.**  
   Based on your current implementation and the training curves in **Figure 8**, how would you expect training time and memory to scale for, say, 13B or 34B models? Are there obvious bottlenecks (e.g., number of chunks, attention over 10k+ tokens) that would require architectural changes like sparse attention or per-layer encoders?

7. **Failure modes at larger mixing coefficients.**  
   In **Figure 4b**, performance clearly deteriorates when the mixing coefficient grows beyond 0.2–0.3. Could you shed light on what failure mode you observe in the decoded weights (e.g., norms exploding, layer normalization breakdown, attention logits distribution)? A small qualitative or quantitative analysis of these failure cases would make the trade-off clearer.

Author responses on these points, especially 1–3 and 5, would substantially increase my confidence in the method’s clarity and applicability.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A. The work focuses on model merging mechanisms and does not introduce new datasets or application domains with clear safety or fairness concerns. The authors briefly acknowledge possible bias-aggregation risks in the appendix impact statement.

## Soundness Rating

3: good.  
The core ideas (transformer-VAE over weights, OT-based latent alignment, latent merging operators) are technically reasonable and supported by multiple experiments, including strong PCA vs VAE and ablations. However, some mathematical and algorithmic details are ambiguous or loosely stated, and the empirical evaluation, while diverse, leaves open questions on scalability and cost/benefit.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with helpful figures (especially **Figure 1**, **Figure 2**, **Figure 4**, and **Figure 9**) and detailed tables. That said, notation inconsistencies, minor typos, and underspecified algorithmic steps make some parts harder to follow than necessary.

## Contribution Rating

3: good.  
The contribution is a meaningful step in weight-space generative modeling and model merging: a practical latent-space pipeline that enables cross-architecture merging with empirical benefits, plus insightful analysis of weight distributions and non-linearity of the weight manifold. The novelty is somewhat tempered by the lack of comparison to several closely related alignment/merging methods, but overall the work is a valuable addition to the area.

## Overall Rating

8: Accept, good paper (poster).  
Despite some clarity issues and incomplete positioning vs very recent related work, LS-Merge presents a coherent and empirically validated framework for latent-space model merging, with a nontrivial OT-based heterogeneous alignment component and strong evidence that a non-linear latent manifold over weights is beneficial. The LoRA expert fusion and PCA vs VAE results in particular are compelling. With improved discussion of related methods and more precise algorithmic specifications, this paper is well suited for presentation at ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with model merging, VAEs over weights, and optimal transport, and I carefully checked the main equations and experimental tables. Some implementation specifics remain unclear, but they do not overturn the main claims.