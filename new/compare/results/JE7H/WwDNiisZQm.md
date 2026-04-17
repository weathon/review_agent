---
job_id: 20321ec7-0efb-40ab-88b5-38ce429d76a1
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: WwDNiisZQm.pdf
paper: Content-Aware Mamba for Learned Image Compression
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper proposes a learned image compression architecture using Mamba-style state-space models and focuses on representation learning and efficient neural architectures for vision, which is fully within ICLR’s scope.

## Minimum Quality
Pass ✅. The submission includes all required sections (Abstract, Introduction, Related Work, Method, Experiments, Results/Analysis, Conclusion). The methodology and experiments are substantial and clearly described, with comparisons to strong baselines. I do not see fatal methodological, theoretical, or experimental flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I do not see any attempts to manipulate automated reviewing (no hidden prompts, odd meta-instructions, or similar content).

---

# Expected Review Outcome:

## Summary

The paper proposes CMIC, a learned image compression model that integrates a Content-Aware Mamba (CAM) block into the analysis/synthesis transforms. CAM introduces two mechanisms: (1) Content-Adaptive Token Permutation (CTP), which clusters latent tokens via a shared codebook and permutes them so content-similar tokens are adjacent in the 1D scan used by Mamba; and (2) Global-Prior Prompting (GPP), which maps cluster centroids into a prompt dictionary that modulates the Mamba output projection to weaken strict causality. Built into a VAE-style compression pipeline with an enhanced SCCTX entropy model, CMIC achieves strong rate–distortion performance, outperforming VTM-21.0 and several state-of-the-art CNN, Transformer, and Mamba-based LIC baselines on Kodak, Tecnick, and CLIC.

## Strengths

1. **Clear architectural idea with solid empirical payoff.**  
   The paper identifies two concrete mismatches between vanilla Mamba and image compression: raster-ordered, content-agnostic scanning and strict causality. The proposed remedies, CTP and GPP, are conceptually straightforward yet well-integrated into the overall transform. The empirical gains in **Table 1** are substantial: CMIC improves BD-rate over VTM-21.0 by −15.91% (Kodak), −21.34% (Tecnick), and −17.58% (CLIC), and also beats strong LIC baselines like FTIC, MLICv2, DCAE, and LALIC while keeping FLOPs and latency in a reasonable range.

2. **Strong and fair comparison against prior Mamba-based LIC.**  
   The work specifically targets limitations of Mamba-based compressors. The comparisons to MambaVC and MambaIC are careful: CMIC outperforms both by non-trivial margins (e.g., BD-rate improvements of 7.51% and 2.36% on Kodak) and does so with lower parameters, FLOPs, and memory as shown in **Table 1** and the throughput numbers in **Table 3** and **Table 10**. This directly supports the claim that content-aware scanning and global prompting materially improve how Mamba is used in this domain.

3. **Good analysis and ablations isolating the contributions.**  
   The ablations in **Table 2** neatly disentangle CTP and GPP from the vanilla single-scan Mamba block. CTP alone gives consistent BD-rate gains (≈2% on Kodak/Tecnick, 1.8% on CLIC), while GPP alone yields smaller but still meaningful improvements. Combined, they provide 2.7–3.6% BD-rate gains over the baseline. Additional architectural ablations in **Table 4** (replacing CAM with Conv blocks, 2D Mamba, attention-only, CAM-only) and in **Table 7/8** (entropy model variants) show that CAM is doing real work beyond simple capacity increase or entropy-model tuning.

4. **Compelling qualitative analysis of receptive fields and content adaptivity.**  
   The ERF visualizations in **Figure 7**, **Figure 8**, and **Figure 9** are a highlight. Figure 7 shows CMIC’s effective receptive field spreading much more globally than CNN and Transformer baselines at the same depth. Figure 8’s per-image ERFs convincingly show that regions with high ERF align with semantically redundant content (hair, feathers, shoreline, aircraft), suggesting the model truly adjusts its dependency structure to content. Figure 9 further decomposes the contributions: without CTP/GPP the ERF is strictly causal and truncated at the raster boundary; enabling GPP creates non-zero ERF after the scan boundary; and adding CTP reshapes ERFs into semantic patterns, supporting the core narrative.

5. **Clustering mechanism is reasonably well motivated and practically efficient.**  
   The codebook-based cosine K-means strategy for token clustering (Section 3.3) is a sensible compromise between per-sample online K-means and fully learnable token routers. The paper describes how the centroids are updated via EMA during training and then used deterministically at inference. The cluster-activation statistics in **Table 5** and the cluster-number ablation in **Table 6** show that the fixed codebook size is not a hard bottleneck and that the effective number of active clusters is content-dependent. **Figure 10**’s masks demonstrate that clusters correspond to semantically coherent regions (e.g., doors, sky, feathers), which supports the claim that permutation is content-aware rather than arbitrary.

6. **Complexity vs. performance trade-off is carefully evaluated.**  
   Beyond raw BD-rate, the paper reports FLOPs, params, latency, and memory (Table 1, Table 3, Table 10). The model is not the lightest in absolute terms, but it compares favorably to strong baselines with similar RD performance (e.g., vs. TCM-L, MLICv2, MambaIC) and especially to 2D Mamba schemes. The fact that CTP+GPP only reduce training throughput about 5% and add ~4% inference latency while giving sizable BD-rate improvements suggests the design is practically viable.

7. **Good contextualization vs. closely related ideas (SegPIC, cluster-based LIC, MambaIRv2).**  
   The related work and appendix carefully distinguish this method from SegPIC’s segmentation-guided adaptivity and Zhang et al.’s grid-anchored clustering, emphasizing permutation-equivariance and fine-grained, non-Euclidean grouping. The comparison with MambaIRv2 in Appendix A.13 goes beyond name-dropping: it discusses the difference between redundancy-aware clustering and a purely classification-head-based routing, and reports BD-rate degradations when swapping in MambaIRv2-style components (Table 9). This shows the authors have thought carefully about neighboring designs.

## Weaknesses

1. **Novelty is somewhat incremental at the level of core mechanisms.**  
   At a high level, both key ingredients are conceptually familiar: clustering tokens and reordering them to improve sequence modeling (akin to routing transformers / adaptive dictionaries) and modulating SSM parameters using prompts/global priors (as in MambaIRv2 and prompt-based attention). The main novelty here lies in (i) plugging a codebook-based clustering into Mamba’s scanning mechanism, and (ii) tying the prompt dictionary to cluster centroids for compression-specific redundancy modeling. This is a meaningful combination and well executed, but the paper could more explicitly articulate what is fundamentally new over existing content-adaptive token routing plus prompt-conditioned SSMs. Currently, the distinction is somewhat blurred between “different task focus” and “different algorithmic principle.”

2. **Mathematical formulation of clustering and prompting is underspecified or slightly inconsistent.**  
   - In Section 3.3, the K-means / cosine clustering is only described qualitatively. The distance matrix $Distance_{i,j}$ is mentioned but never formally defined. Since the text says “cosine-based clustering” and “normalized centroids,” presumably $Distance_{i,j} = 1 - \langle \hat x_i, \hat c_j \rangle$ with $\|\hat x_i\|=\|\hat c_j\|=1$, but this should be spelled out. Likewise, Algorithm 1 is referenced but not included in the main text, and the number of iterations $T$ and any convergence criteria are not given here. For a core architectural mechanism that runs 5 iterations per step, more explicit equations and pseudo-code in the main paper would be appropriate.  
   - In Section 3.4, dimensions around the prompting modification are ambiguous. The vanilla SSM uses $C\in\mathbb{R}^{d\times d_s}$ (implicitly) to map hidden state $h_i\in\mathbb{R}^{d_s}$ to an output token in $\mathbb{R}^{d}$, but the prompt matrix $P$ is defined as $\mathbf{P}\in\mathbb{R}^{N\times d_s}$ via $\mathbf{P}=\Gamma U$, where $U\in\mathbb{R}^{K\times d_s}$. Then Equation
     \[
       \mathbf{O}_i = (\mathbf{C}+\mathbf{P})\mathbf{h}_i + \mathbf{D}\mathbf{x}_i
     \]
     appears dimensionally inconsistent unless there is an implicit per-token projection or broadcasting scheme (e.g., $P_i\in\mathbb{R}^{d\times d_s}$ after another linear layer). As written, $C$ and $P$ live in different spaces. This needs clarification: are prompts applied as a diagonal scaling of $C$, as an additive low-rank update, or as a separate projection whose output is summed with $C h_i$? Without this, the exact parameterization of the “Attentive State-Space” variant is unclear and may hinder reproducibility.

3. **Content-adaptive permutation introduces non-differentiable decisions but gradients and bias are not deeply analyzed.**  
   The permutation $\pi$ is computed from hard K-means assignments $g_i$, and the authors explicitly acknowledge in Appendix A.8 that clustering and token sorting are non-differentiable. Their argument is essentially empirical: EMA and codebook-based updates yield stable training curves (Figure 18). However, no discussion is provided about the gradient flow through the clustering step: are gradients simply ignored for the codebook during backprop (straight-through style)? Do the non-differentiable permutations bias learning in ways that may affect optimality or stability? Since the permutation is applied multiple times per sample and deeply intertwined with Mamba’s recurrent dynamics, some analysis or at least ablation contrasting soft vs hard clustering or randomization would strengthen the scientific claims beyond “it works in practice.”

4. **Limited empirical diversity and robustness checks.**  
   While Kodak, Tecnick, and CLIC are standard and acceptable, all are natural photographic images with similar statistics. The method heavily relies on a dataset-level codebook of centroids learned from Flickr2W, and it is plausible that performance may degrade on out-of-distribution data (e.g., line art, medical, satellite, or synthetic images) where learned centroids are misaligned. There are no experiments probing robustness to such shifts or even a qualitative example where clustering fails. Similarly, variable-rate compression or performance under different $\lambda$ schedules is not discussed, though many modern LIC models address this. At minimum, a short discussion of how CMIC might behave under significant domain shifts or rate-adaptation settings would help understand its generality.

5. **Entropy model remains relatively conventional; the role of CAM there is limited and under-discussed.**  
   Section 4.5 notes that inserting CAM into the entropy model brings “negligible performance gains while increasing latency,” and **Table 7** confirms that the CAM-augmented SCCTX has almost identical BD-rate to the efficient Conv SCCTX baseline but higher decoding time. This basically confines CAM’s impact to the non-linear transforms only. While this is a reasonable design choice, it somewhat weakens the overarching claim that “content-aware Mamba is broadly useful for compression.” It would be valuable to analyze *why* the entropy model seems insensitive to content-aware global modeling (e.g., is the latent distribution already close to factorized after transform, or do the autoregressive dependencies not align with the cluster structure?), and to make that limitation more explicit in the main text rather than relegating it to the appendix.

6. **Ablations focus on BD-rate but are light on more diagnostic metrics or visual failure analyses.**  
   The paper does a good job with BD-rate ablations, but there is relatively little discussion of *when* CMIC helps most and when it might hurt. For instance, are gains larger on highly textured, globally repetitive content (where long-range redundancy dominates) versus smooth, low-entropy images? Figure 4–6 show RD curves but do not break down performance by content type or bit-rate region. A simple stratified analysis (e.g., partitioning Tecnick by texture or edge density) would give more insight into what structures CTP and GPP actually capture. Similarly, while **Figure 10** shows nice cluster masks, there is no example where clustering is clearly suboptimal or confuses unrelated regions, which would help bound expectations.

7. **Some aspects of fairness in comparison and tuning are not fully detailed.**  
   All baselines appear to be taken from the literature and compared via BD-rate curves, but the paper does not state explicitly whether they re-trained any baselines under the same dataset/protocol, or whether they rely entirely on published models/curves. Given that some baselines (e.g., FTIC, TCM) may have been trained on slightly different data or loss configurations, there is a residual risk that performance differences are partly due to training recipe rather than architecture. The claim “CMIC surpasses FTIC by 0.32 dB on CLIC” would be more convincing if supported by a short note explaining the source of baseline checkpoints and confirming comparable training data and evaluation pipelines.

8. **Minor clarity issues and typos.**  
   There are various small notational and editorial issues: in Section 3.1 the analysis transform is denoted both $g_a$ and $g_s$ in the same sentence; the rate term uses $p_{\hat{y}}$ and $p_{\hat{z}}$ while the variables in the expectation are sometimes written without bold; Section 3.3 mentions “Algorithm 1” which is absent in the main body; some references (e.g., to “Taxens” in A.10) look like leftover typos. These are not fatal but detract from polish.

## Potentially Missing Related Work

1. **Mentzer et al., "Conditional Probability Models for Deep Image Compression", 2018.**  
   This early work on conditional probability models for deep image compression is directly relevant to the entropy modeling side and contextualizes SCCTX-style designs. It should be discussed in Section 2.1 when describing autoregressive and conditional entropy models and possibly referenced alongside Minnen et al. (2018).

2. **Choi et al., "Variable Rate Deep Image Compression with a Conditional Autoencoder", 2019.**  
   While this paper focuses on variable-rate coding rather than SSMs, it is a standard reference for conditional autoencoder-based LIC and is relevant to the broader design space of transform/entropy models. It could be briefly covered in Section 2.1, especially when discussing architectures trained under multiple $\lambda$ settings or potential extensions of CMIC to variable-rate coding.

3. **Yang et al., "Learning for Video Compression", 2018.**  
   Even though this is about video rather than still images, it is an influential early work on learned compression. A brief mention in Section 2.1 or the introduction, perhaps in a paragraph bridging image and video compression, would help place CMIC within the larger trajectory of learned compression methods.

4. **Li et al., "Hybrid Image Compression Using Deep Learning", 2020.**  
   This work proposes hybrid approaches that fuse classical and deep components for image compression. Since CMIC emphasizes the interaction between linear-time sequence modeling and VAE-style transforms, it would be natural to cite and contrast this hybrid line of work in Section 2.1, clarifying how CMIC differs from codec-hybrid approaches.

## Questions

1. **Clarification of prompt parameterization (Equation for $O_i$).**  
   Can you precisely specify the shapes of $C$, $D$, $h_i$, $x_i$, and $P$ in Equation $O_i = (C+P)h_i + Dx_i$ and how the addition $C+P$ is implemented in practice? For example, is $P$ first projected to a $d\times d_s$ matrix per token, or does it act as a diagonal scaling of $C$? A short explicit formula would both resolve the dimensional ambiguity and help others reproduce your Attentive State-Space variant.

2. **Gradient flow through clustering and permutation.**  
   How exactly are gradients handled for the codebook centroids and assignments? Are centroids updated purely via EMA with no gradient flow, while tokens and subsequent layers receive gradients as if the permutation were fixed? Did you experiment with any soft-assignment or straight-through estimator variants, and if so, how did they compare? Evidence on whether the current hard clustering introduces noticeable gradient bias would be helpful.

3. **Robustness to domain shifts and non-natural images.**  
   Have you tested CMIC on non-photographic images (e.g., cartoons, line drawings, medical or remote sensing data)? Given that the clustering codebooks are learned from Flickr2W, I am curious whether the content-adaptive scan and redundancy-aware prompts still behave sensibly when the underlying visual statistics diverge significantly.

4. **Failure modes of clustering.**  
   In Figure 10, clustering looks very clean. Could you provide examples where clustering is *not* semantically aligned, or where many clusters become nearly empty? How does such misclustering manifest in RD performance or ERFs? This would provide a more balanced picture of the strengths and limitations of CTP.

5. **Baseline training details.**  
   For the main RD comparisons (Table 1, Figures 4–6), are all baselines taken from reference implementations trained on the same Flickr2W dataset with similar optimization schedules, or from their original releases? If the latter, can you comment on potential discrepancies in data or loss functions and how you mitigated them (e.g., re-evaluation pipelines, same test splits, etc.)?

6. **Potential for variable-rate or scale-general models.**  
   Given that CMIC is trained separately for each $\lambda$, do you see a straightforward way to adapt CTP+GPP to a single variable-rate model (e.g., conditioning codebooks or prompts on a rate-control token)? Would the clustering codebooks remain stable across a wider range of bitrates, or do you anticipate that low-rate and high-rate regimes require different centroid structures?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3 (good): The method is technically sound overall and backed by solid experiments, though some mathematical details (especially around the prompt parameterization and clustering mechanics) are under-specified and would benefit from clarification.

## Presentation Rating

3 (good): The paper is generally clear, well structured, and supported by strong figures and tables, despite some notational ambiguities and minor typos.

## Contribution Rating

3 (good): The work offers a useful and empirically validated way of making Mamba more content-aware for image compression, with meaningful gains over strong baselines, although the core ideas build on existing concepts like clustering-based routing and prompt-conditioned SSMs.

## Overall Rating

8 (Accept, good paper (poster)): The paper presents a well-motivated and carefully evaluated enhancement of Mamba-based learned image compression, with clear empirical benefits and insightful analyses. While the conceptual novelty is moderate and some mathematical details need tightening, the contribution is solid and of clear interest to the ICLR community.

## Reviewer Confidence

4 (confident): I am familiar with learned image compression, attention/SSM architectures, and content-adaptive mechanisms, and I examined the equations, ablations, and comparisons in detail, though I did not independently re-implement the method.