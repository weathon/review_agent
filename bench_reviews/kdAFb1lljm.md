## Summary
This paper proposes **Med-SegNet**, a compact encoder–decoder for binary medical image segmentation that inserts a single **Circulant Layer Token Mixer (CLTM)** at the bottleneck. The central empirical claim is that this lightweight, attention-free cross-scale mixer improves performance consistently across a broad suite of 20 public datasets while maintaining a very small model size (~2.07M parameters).

## Strengths
- **The paper demonstrates unusually broad within-paper validation of the proposed module across many medical domains.** The same architecture is evaluated on **20 datasets spanning 12 modalities**, and the ablation in Table 1 reports improvements on **20/20 datasets** when adding CLTM, with the mean Dice improving from **0.8977 to 0.9161**. Regardless of how one interprets external SOTA comparisons, this breadth of internal evaluation is specific and valuable.
- **The core architectural choice is targeted and efficient rather than brute-force.** CLTM is inserted **once at the bottleneck**, not throughout the network, and the paper gives a concrete complexity argument for the mixer: depthwise 1D circular convolution with parameter cost **\(k d\)** and mixing cost **\(O(Nkd)\)**. This is a specific design decision that plausibly explains why the full model remains at **~2.07M parameters**.
- **The empirical gains are most pronounced on precisely the difficult regimes where additional context should matter.** The largest reported ablation gains are on harder, low-contrast or structurally challenging datasets such as **BUSI (+6.31 Dice points)** and **RaViR (+6.12)**, while performance on easier near-ceiling datasets is largely preserved. This pattern is consistent with the intended role of the bottleneck mixer.
- **The paper is explicit about an important limitation instead of hiding it.** The conclusion clearly states that the evaluation is **confined to 2D inputs** and that robustness under distribution shift is not deeply analyzed. This scope clarity matters when judging the claims.

## Weaknesses

###: Fatal
- **The paper overclaims what CLTM actually computes: as specified, it is not a true single-step global interaction mechanism.**  
  In Section 3.4, the mixer is explicitly
  > “a depthwise one-dimensional circular convolution ... with learnable kernel of length \(k\)”  
  and the paper later states:
  > “We use \(k=5\) by default.”
  
  A single depthwise 1D convolution with fixed small kernel size has a **local receptive field along the token sequence**, even with circular padding. Circular padding wraps boundaries, but it does not make the operator globally dense in one pass. The text repeatedly describes CLTM as performing a **“single global information exchange,” “global token interaction,”** and as supplying **“global context”** in the same sense used to motivate replacing self-attention. That characterization is mathematically too strong for the operation actually defined. This does not mean the module is useless—the cross-scale concatenation and bottleneck placement may still help—but it **undermines the paper’s core conceptual framing** and some of its strongest claims.

### Major:
- **The external “state-of-the-art” comparisons in Table 2 are not methodologically strong enough to support the paper’s strongest comparative claims.**  
  The paper explicitly states in Table 2:
  > “Results for other methodologies are copied as reported in their original papers (not re-trained here). Our Med-SegNet results are produced under the unified setup described in Experimental Setup.”
  
  This means the model is being compared against numbers obtained under **different preprocessing, splits, resolutions, losses, and training schedules**. As a result, claims like “establishes a new benchmark,” “state-of-the-art,” or “decisively outperforms” are not adequately supported by Table 2 alone. The internal CLTM ablation is still meaningful, but the external superiority claims should be softened unless at least a few strong baselines are retrained under the same protocol.
- **The paper contains a nontrivial inconsistency in the training setup.**  
  Section 4 says:
  > “Adam optimizer (learning rate: 0.0175)”  
  whereas the appendix says:
  > “Adam (base learning rate \(7.5 \times 10^{-4}\)) and a cosine-decay schedule”
  
  This is a large discrepancy, not a minor typo, and it affects interpretation of the results and reproducibility of the reported numbers. Since the paper’s contribution is empirical and optimization-sensitive, this should be resolved clearly.
- **The ablation study is too shallow to isolate what aspect of CLTM is responsible for the gains.**  
  The paper only compares **with vs. without CLTM** across datasets. It does **not** ablate:
  - kernel size \(k\),
  - whether circular padding matters versus ordinary 1D conv,
  - whether cross-scale concatenation is necessary,
  - whether the gains come primarily from pre/post normalization and residual reprojection,
  - or whether a simpler bottleneck module with similar parameter count would achieve similar improvements.
  
  Because the central contribution is a specific mixer design, these missing controls matter. Without them, the evidence supports “this added bottleneck module helps,” but does not yet convincingly establish that the **circulant cross-scale design itself** is the key reason.
- **Efficiency claims are only partially substantiated.**  
  The paper argues that CLTM is near-linear and hardware-friendly, and the appendix does provide some runtime information on TPU (step times and a test-set throughput figure). However, the main paper does not provide a clean comparative table of **latency / FLOPs / peak memory** against baselines, nor scaling with input resolution. Given the emphasis on “practical latency,” “low memory,” and “hardware-friendly deployment,” stronger empirical efficiency evidence is needed, especially for comparison-driven claims.

### Minor
- **The paper’s novelty is somewhat incremental at the mechanism level.**  
  The work is a sensible adaptation of attention-free token mixing ideas to a medical segmentation bottleneck, but the mixer itself is not a large conceptual leap over existing structured/token-mixing approaches. The practical integration is more convincing than the underlying methodological novelty.
- **The significance is limited by the 2D-only scope.**  
  The paper acknowledges this limitation. Since many high-impact medical segmentation settings are volumetric, the current contribution is better viewed as a promising 2D segmentation architecture than a broadly complete medical segmentation solution.
- **Some claims of “statistically meaningful” gains are not fully supported by the reported evidence.**  
  The paper says the improvements are “statistically meaningful,” but no statistical testing, variance estimates, or multi-seed results are shown in the main text or appendix excerpt provided. The empirical trend is encouraging, but that wording is stronger than the presented evidence.

### Trivial
- None.

## Nice-to-Haves
- Retrain a **small but strong subset of baselines** (e.g., U-Net/UNet++, TransUNet or Swin-UNet, and one recent efficient mixer/SSM model) under the paper’s exact training protocol to make the comparative section much more credible.
- Add a **component-level ablation** for CLTM: vary kernel size \(k\), remove cross-scale concatenation, replace circular conv with plain 1D conv, and test normalization variants.
- Replace “global” phrasing with more precise language unless a larger-kernel / multi-hop / globally dense variant is implemented and validated.
- Provide a compact **Pareto table or plot** showing Dice vs. parameters/FLOPs/latency.
- Include a short discussion or visualization clarifying how token linearization interacts with 2D spatial structure. This is not currently a fatal flaw, but it would help interpret what the mixer is actually learning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that CLTM’s 1D flattening is inherently destructive or invalid for 2D segmentation.**  
  This concern is plausible as a question, but the harsh review stated it too strongly as a fundamental flaw. The paper does flatten multi-scale features into sequences and mix them with 1D convolution, but many successful vision architectures also tokenize spatial maps. The current evidence does not justify calling this inherently broken.
- **Complaint that some external comparisons may involve models with different task dimensionality or setup specifics.**  
  The broader concern about unfair external comparison is valid and kept, but specific claims such as “those models are often 3D” are not verified from the paper and should not be asserted.
- **Criticism about missing related works.**  
  Per instruction, this is removed. The review should not fault the paper for omitted literature that cannot be externally verified here.
- **SE reduction ratio \(R=24\) as an intrinsic design flaw.**  
  The paper gives this value, but there is no evidence in the manuscript that it is unreasonable or harmful. The real issue is lack of ablation, not the number itself.
- **Claim that there are zero efficiency benchmarks.**  
  This is factually incorrect. The appendix does report TPU timing/throughput information. The valid criticism is that comparative efficiency evidence is insufficient, not absent.
- **Demand for complete training logs.**  
  This is a reproducibility nitpick beyond normal submission standards and is not necessary for the core scientific evaluation.

## Novel Insights
The strongest synthesis across the reviews is that this paper is **better empirically than conceptually framed**. The evidence that “adding this bottleneck module helps across many datasets” is fairly persuasive, especially because the gains are broad and largest on hard cases. But the paper’s conceptual sales pitch—that a single small-kernel circular depthwise convolution constitutes a true global interaction mechanism replacing self-attention—is overstated. If reframed more honestly as an efficient **cross-scale bottleneck mixer with local sequence mixing and broad empirical utility**, the work would read as a more credible and solid engineering contribution.

## Suggestions
- **Reframe the core claim**: avoid describing CLTM as a true one-step global interaction module unless the operator is changed; present it instead as an efficient cross-scale bottleneck mixer.
- **Fix the learning-rate inconsistency** between Section 4 and the appendix, and ensure the final camera-ready text has one unambiguous training protocol.
- **Strengthen Table 2** by retraining a representative subset of strong baselines under the same setup; otherwise, tone down SOTA language.
- **Run targeted CLTM ablations**: \(k\in\{3,5,7,9\}\), circular vs. standard padding, single-scale vs. cross-scale mixing, and with/without pre/post normalization.
- **Add comparative efficiency evidence** in the main paper: FLOPs, peak memory, and latency at one or two standard resolutions.
- **Clarify the contribution axis in the narrative**: the paper appears strongest on empirical robustness and parameter efficiency, and weaker on fundamental novelty and theoretical justification.