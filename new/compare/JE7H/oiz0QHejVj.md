---
job_id: 2173fb1c-d783-40e6-976f-a66a107d65cd
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: oiz0QHejVj.pdf
paper: CLIP-Map: Structured Matrix Mapping for Parameter-Efficient CLIP Compression
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a parameter‑efficient compression method for CLIP, squarely in representation learning, model compression, and optimization, all well within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is in English and has all required components: Abstract, Introduction, Related Work, Method, Experiments (with several tables and figures), and Conclusion. The method is technically coherent and supported by non‑trivial experiments; no obvious fatal methodological flaw or test leakage is visible.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any explicit attempt to manipulate automated reviewing systems or hidden prompts in the paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces CLIP‑Map, a mapping‑based framework for compressing CLIP‑like vision–language models. Instead of select‑based pruning, CLIP‑Map learns structured mapping matrices that transform a large model’s parameters into a smaller model using Kronecker‑factorized “full mapping” for width compression and linear combinations of layers for depth compression. A “Diagonal Inheritance Initialization” is used to stabilize optimization, and the mapped model is subsequently retrained with knowledge distillation; experiments on YFCC‑trained CLIP/OpenCLIP show improved zero‑shot retrieval and classification accuracy compared with TinyCLIP at similar compression ratios, especially under heavy compression.

## Strengths

1. **Conceptual shift from selection to mapping.**  
   The paper makes a clear conceptual move away from select‑based pruning (masking / dropping parameters) to learnable mapping of all pretrained weights into a smaller network. This is well illustrated in **Figure 1**, which contrasts select‑based pruning (left) with the proposed mapping‑based process (right). This framing is easy to understand and compelling from an information‑preservation perspective.

2. **Structured Kronecker mapping is parameter‑efficient and mathematically clean.**  
   The derivation in **Equations (3)–(4)** shows how a dense mapping matrix \(\mathbf{R}_l\in\mathbb{R}^{D_2^2\times D_1^2}\) can be replaced by a Kronecker product \(\mathbf{F}_l^{in}\otimes \mathbf{F}_l^{out}\), which is equivalent to applying \(\mathbf{F}_l^{out}\) and \(\mathbf{F}_l^{in}\) via standard matrix multiplications. This reduces parameters from \(\mathcal{O}(D_1^2D_2^2)\) to \(\mathcal{O}(D_1D_2)\) and avoids explicitly constructing huge mapping matrices, which is a sensible and well‑justified design.

3. **Diagonal Inheritance Initialization is simple and empirically important.**  
   The discussion around **Equations (5)–(8)** articulates the variance explosion issue when factor matrices in a Kronecker product are randomly initialized. The proposed Diagonal Inheritance Initialization in **Equation (9)** and **Figure 3** is a straightforward but effective way to preserve approximate identity mappings at the start, thus inheriting part of the pretrained weights. **Table 5** and **Figure 6** provide strong empirical support: random / Xavier / Kaiming inits yield near‑random performance (IN‑1K accuracy ≈ 0–5%), whereas diagonal init gives 28.9% IN‑1K and a much faster loss decrease, which convincingly shows that initialization is not a cosmetic tweak but central to making mapping actually trainable.

4. **Unified handling of width and depth compression.**  
   The method treats width compression via \(\mathbf{F}^{in}, \mathbf{F}^{out}\) and depth compression via \(\mathbf{L}_{depth}\) jointly in one differentiable pipeline. **Figure 3** nicely illustrates how input and output dimension mappings are combined with layer‑wise linear combinations to form fewer, narrower layers. This unified approach is conceptually cleaner than multi‑stage pruning heuristics often used in prior CLIP compression work.

5. **Consistent empirical gains over TinyCLIP under matched budgets.**  
   The main results in **Table 1** and **Table 2** are quite favorable. For instance, at the extreme 1% compression ratio on YFCC‑15M, CLIP‑Map\(_{\text{tiny}}\) improves MSCOCO TR@1 from 12.5 to 15.8 and Flickr30K TR@1 from 24.5 to 30.3 over a re‑implemented TinyCLIP (5×25ep). At 10.8% compression, CLIP‑Map\(_{\text{small}}\) improves TR@1/IR@1 on MSCOCO from 36.2/21.5 to 38.4/24.3 and consistently outperforms TinyCLIP on most of the 21 zero‑shot classification datasets in **Table 2**. These are reasonably large margins given identical student architectures and training data.

6. **Better efficiency / fewer epochs for similar or better performance.**  
   **Table 11** compares wall‑clock training time; for the “small” configuration, CLIP‑Map requires 22h10m vs 32h36m for TinyCLIP, yet achieves better MSCOCO TR@1 (64.6 vs 62.6 in **Table 1**). The training‑curve comparison in **Figure 4** further shows faster convergence of TR@1 on MSCOCO for CLIP‑Map relative to TinyCLIP. This speaks to the practical value of the mapping‑based initialization beyond academic metrics.

7. **Ablations and visualizations give insight into the mapping behavior.**  
   The ablation in **Table 4** on varying mapping‑stage duration, together with **Figure 5** (evolution of a mapping matrix from a diagonal pattern to a more uniform structure), provides nice intuition that the mapping matrices gradually move away from pure identity and “search” a better compression. The monotonic performance improvement up to 5 epochs of mapping, then degradation when mapping is too long, is a useful practical guideline.

8. **Broad evaluation across compression ratios and backbones.**  
   The authors test three compression levels (≈1%, ≈10%, ≈50%) and several teacher models (OpenCLIP, MetaCLIP, and a ResNet‑based CLIP). While the ResNet experiments are limited to the mapping stage only, they still suggest that the approach is not tightly coupled to ViT backbones.

## Weaknesses

1. **Depth‑compression formulation and constraints are underspecified and somewhat hand‑wavy.**  
   The depth compression is given in **Equation (2)** as \(\mathbf{W}_{l'}^{\text{new}} = \sum_{l=1}^{L_1} \mathbf{L}_{depth}[l',l]\mathbf{W}_l\), but it is never clearly stated:
   - which layers are being linearly combined (e.g., are MHA and FFN blocks mixed together or treated separately? How are norms or scales handled?),
   - what constraints are placed on \(\mathbf{L}_{depth}\) (non‑negativity, normalization, sparsity, monotonicity w.r.t. depth?),  
   - and whether this mixing preserves any notion of representational hierarchy.  
   Without such constraints, \(\mathbf{L}_{depth}\) could in principle create arbitrary mixtures of early and late layers, potentially breaking any “shallower” semantics. This is important, because the method claims a “unified, end‑to‑end optimization of width and depth”, yet the depth side is only superficially described. A more explicit parameterization (e.g., soft assignments with row‑stochasticity) or at least empirical analysis of learned \(\mathbf{L}_{depth}\) patterns would strengthen the work considerably.

2. **Initialization analysis is only variance‑level and somewhat incomplete.**  
   The formulation in **Equations (5)–(8)** focuses solely on the variance of a Kronecker‑structured mapping under independent initialization of \(\mathbf{A},\mathbf{B}\). Two issues:
   - The paper does not connect this variance analysis to preservation of activation distributions or gradient scales in the actual CLIP network. There is no attempt at matching fan‑in/fan‑out or using established initialization theory for linear transformations composed with nonlinearities. As a result, the “distribution shifting problem” claim remains qualitative.
   - Diagonal Inheritance Initialization in **Equation (9)** treats \(\mathbf{F}^{in},\mathbf{F}^{out}\in\mathbb{R}^{D_2\times D_1}\) as identity‑like, but when \(D_2<D_1\), this is a rectangular matrix. The paper glosses over how “identity” is defined in that case. Concretely, are the first \(D_2\) rows of \(\mathbf{F}\) set to select the first \(D_2\) channels, or are rows aligned with some heuristic ordering? How does this interact with multi‑head attention where heads are typically interleaved in the dimension? These details matter for reproducibility and for understanding what “part of the original parameter structure” is actually inherited.

3. **Loss definitions and notation are sloppy and occasionally inconsistent.**  
   There are several issues:
   - In **Equation (11)**, the distillation loss is written as \(\mathcal{L}_{distill} = CE(logits_{I2T}^{*}, logits_{I2T}^{t}) + CE(logits_{T2I}^{*}, logits_{T2I}^{t})\). The star vs superscript \(s\) (student) notation is never clearly defined, and the equation appears to treat teacher logits as targets in a standard cross‑entropy without mention of temperature scaling, which is atypical in KD.  
   - In **Equation (12)**, the hard loss is written as \(CE(logits_{T2T}^s,\text{labels}) + CE(logits_{T2I}^s,\text{labels})\), which looks like a typo (should be I2T/T2I or text‑image logits, not T2T).  
   - In **Equation (13)**, \(\mathcal{L}_{\text{soft}}\) is used but earlier the paper defined \(\mathcal{L}_{distill}\).  
   These may seem minor, but for a method that hinges on a two‑stage pipeline and a mix of hard vs soft losses, sloppy notation is a red flag and makes it harder to be confident that the reported implementation exactly matches the described method.

4. **Limited comparison to alternative structured / mapping‑based compression approaches.**  
   The main empirical comparison is to TinyCLIP (select‑based pruning + distillation). Other CLIP compression methods like MoPE‑CLIP, EfficientVLM, MobileCLIP appear in **Table 3**, but only as single ImageNet‑1K numbers; there is no head‑to‑head comparison under controlled training budgets or matching student architectures, and no retrieval results vs these methods. More critically, there is no comparison to other Kronecker / decomposition‑based or mapping‑based compression methods that have appeared in the broader literature (even if not specifically for CLIP), which is directly relevant given the core use of Kronecker factorization. This makes it hard to disentangle whether the gains come from “learning a mapping” vs simply having a good KD recipe plus large‑scale training.

5. **Experimental scope is still relatively narrow: one main pretraining dataset and mostly retrieval / classification.**  
   All main experiments are trained on YFCC‑15M, whereas many CLIP compression works also consider LAION‑2B, CC12M, or larger curated datasets. The authors mention limited compute as a constraint, which is fair, but it does limit external validity. In addition, the evaluation focuses only on zero‑shot retrieval (MSCOCO, Flickr30K) and zero‑shot classification on a set of datasets. It would be informative to see at least one downstream fine‑tuning task (e.g., open‑vocabulary detection or segmentation), especially since the mapping mixes transformer layers in a non‑trivial way and might affect adaptation.

6. **Some design choices are under‑motivated and lack ablations.**  
   A few examples:
   - The choice of using only hard CLIP loss in the mapping stage and only distillation loss (with \(\lambda=1\)) in retraining is somewhat ad hoc. **Table 10** is used to justify \(\lambda=1\), but it is run only on CC3M and MSCOCO; there is no indication whether this generalizes across scales and datasets.  
   - Depth compression is barely analyzed: no ablation on mapping only width vs mapping both width and depth, or on varying depth ratios independently. **Table 4** covers mapping steps but not the effect of turning off depth mapping.  
   - In Appendix A.3, they share \(\mathbf{F}^{out}_{emb}\) among multiple attention matrices (Q,K,V,O) as a parameter‑reduction trick, but there is no ablation to show whether this sharing hurts performance or not.

7. **Missing clarity on what exactly is frozen vs updated when.**  
   **Figure 2** gives a high‑level picture: mapping stage freezes the teacher and learns mapping params; retraining stage uses the mapped model as a student and distills from the teacher. However, key implementation details are not explicit:
   - During the mapping stage, are only \(\mathbf{F}^{in},\mathbf{F}^{out},\mathbf{L}_{depth}\) updated while all compressed model parameters are implicit functions of the teacher? Or are the mapped weights materialized and further fine‑tuned?  
   - At the start of retraining, do they discard the mapping parameters and use the resulting compressed weights as a normal CLIP, or are mapping parameters still active?  
   This matters for understanding the true parameter count at inference and for reproducing the exact initialization of the student.

8. **Baseline fairness at 50% compression is not fully clear.**  
   In **Table 1**, at ≈50% compression (\(39\times 10\) params), CLIP‑Map and TinyCLIP have very close retrieval performance, sometimes slightly worse (e.g., Flickr30K TR@1: 81.0 vs 84.6). But the TinyCLIP model there is structurally different (TinyCLIP uses 512×24 ViT layers; CLIP‑Map uses 512×12). **Table 6** clarifies architectures but the discussion in Section 4.2 is somewhat one‑sided, emphasizing superiority under extreme compression while glossing over that at moderate compression, gains are small or reversed on some metrics. This is not fatal, but a more balanced discussion would be appropriate.

9. **Minor math / notation issues and typos.**  
   Examples:
   - In **Equation (1)**, \(Vec(\mathbf{W}_l')\in\mathbb{R}^{D_2\times D_2}\) is dimensionally wrong; it should be \(\mathbb{R}^{D_2^2}\).  
   - Mixing boldface vs non‑bold symbols and inconsistent index ranges (e.g., using \(l\) both as summation variable and as layer index in **Equation (2)**) reduces readability.  
   - Some references are slightly garbled (e.g., “Cheris et al., 2023; Ebarco et al., 2021” instead of Cherti / Ilharco in **Table 6**).  
   These do not invalidate the method, but they detract from clarity.

Overall, the paper is technically solid and empirically convincing, but there are clear gaps in clarity around depth compression, initialization semantics, and role of mapping parameters, and the experimental narrative is more narrow and slightly more positive‑spun than the raw tables justify.

## Potentially Missing Related Work

1. **Chekalina et al., “Generalized Fisher‑Weighted SVD: Scalable Kronecker‑Factored Fisher Approximation for Compressing Large Language Models” (2025).**  
   Uses Kronecker‑factored approximations and structured decompositions for compression, directly related to the Kronecker‑factorized mappings used in CLIP‑Map. It should be discussed in Section 2.2 and/or around **Equations (3)–(4)** to position CLIP‑Map relative to other Kronecker‑based compression strategies.

2. **Gamal et al., “SeKron: A Decomposition Method Supporting Many Factorization Structures” (2023).**  
   Proposes tensor decompositions via sequences of Kronecker products. This is thematically close to the structured matrix mapping in Section 3.2.2 and should be cited when introducing Full‑Mapping with Kronecker Factorization, to clarify differences between generic Kronecker decompositions and the specific factorization used here.

3. **Guo et al., “CALIP: Zero‑Shot Enhancement of CLIP with Parameter‑free Attention” (2022).**  
   Focuses on parameter‑efficient adaptation of CLIP via parameter‑free modules. Although not a compression method per se, it is highly relevant to the goal of making CLIP more efficient and should be referenced in Section 2.1 when discussing model‑efficient CLIP adaptations.

4. **Shafiullah et al., “CLIP‑Fields: Weakly Supervised Semantic Fields for Robotic Memory” (2022).**  
   Uses CLIP in robotics and semantic mapping. This paper is not about compression, but mentioning it in Section 1 or 2.1 would better contextualize why CLIP compression for resource‑constrained robotic platforms might be particularly important.

5. **Song et al., “CLIP Models are Few‑shot Learners: Empirical Studies on VQA and Visual Entailment” (2022).**  
   Examines CLIP’s few‑shot performance in downstream tasks. Including it in Section 2.1 or in the discussion of downstream evaluation would help argue for the importance of preserving few‑shot capabilities under compression, which CLIP‑Map currently does not evaluate.

## Questions

1. **Details of depth compression and constraints on \(\mathbf{L}_{depth}\).**  
   - Are rows of \(\mathbf{L}_{depth}\) constrained (e.g., row‑stochastic, sparse, or monotone)?  
   - Can you provide statistics or visualizations of learned \(\mathbf{L}_{depth}\) matrices, similar to **Figure 5** for \(F\), to show whether certain original layers dominate each compressed layer?

2. **Exact semantics of Diagonal Inheritance Initialization when \(D_2<D_1\).**  
   - Which specific channels are inherited? For ViT, do you inherit contiguous dimensions (e.g., the first \(D_2\)) or some head‑aligned subset?  
   - Have you tried permutations of channels (e.g., distributing heads more evenly) and, if so, how sensitive is performance to this choice?

3. **Role of mapping parameters after retraining.**  
   - At inference time, are \(\mathbf{F}^{in},\mathbf{F}^{out},\mathbf{L}_{depth}\) still present and applied, or are they only used to initialize the compressed model’s weights and then discarded?  
   - If they are discarded, did you observe any performance drop between “just after mapping” and “after retraining”? Clarifying this would help readers reason about parameter and FLOP counts.

4. **Comparisons with MoPE‑CLIP / CLIP‑KD under matched settings.**  
   - In **Table 3**, CLIP‑Map\(_{base}\) looks competitive or slightly better than these methods but under different seen‑sample counts and sometimes different student sizes. Could you provide a controlled experiment where all methods are run with the same student architecture and total number of optimization steps, at least on one compression ratio?

5. **Effect of depth compression alone vs width compression alone.**  
   - Have you tried ablations where you only apply width mapping but keep depth unchanged, and vice versa? This would clarify how much of the gain comes from each component.

6. **Choice of loss mixture and \(\lambda\) beyond CC3M.**  
   - **Table 10** suggests \(\lambda=1\) works best on CC3M+MSCOCO for the small model. Did you check this on YFCC‑15M and larger models (base / 39M)? If not, could you comment on why you are confident in fixing \(\lambda=1\) universally?

Clear answers, especially to (1)–(3), could significantly increase my confidence in both the soundness and reproducibility of the approach.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work focuses on compressing existing CLIP‑like models using standard public datasets (YFCC‑15M, MSCOCO, Flickr30K, ImageNet‑1K). There is no indication of new sensitive data, problematic deployment scenarios, or unusual data practices beyond typical CLIP‑style training.

## Soundness Rating

3: good.  
The core method (Kronecker‑factorized mapping + diagonal init + KD retraining) is technically coherent and well supported by ablations and multiple benchmarks. However, underspecification of depth compression, some notational sloppiness, and limited analysis of alternative design choices prevent a rating of “excellent”.

## Presentation Rating

3: good.  
The paper is generally readable, the main ideas are well illustrated (especially **Figures 1–3**), and tables are comprehensive. Nonetheless, there are several typos, inconsistent symbols (e.g., in **Equations (11–13)**), and missing implementation details that reduce clarity.

## Contribution Rating

3: good.  
Replacing select‑based pruning with a mapping‑based, Kronecker‑factorized compression pipeline for CLIP is a meaningful and non‑trivial contribution, and the empirical improvements under heavy compression are practically relevant. The conceptual novelty is moderate but solid, and could be strengthened with better depth‑mapping analysis and broader comparisons.

## Overall Rating

8: Accept, good paper (poster).  
The paper proposes a reasonably original, well‑engineered mapping‑based alternative to CLIP pruning, shows consistent gains over a strong baseline (TinyCLIP) with fewer training epochs, and provides insightful ablations on initialization and mapping duration. Despite some gaps in clarity and breadth of comparisons, the method appears sound and practically useful to the community working on efficient multimodal models.

## Reviewer Confidence

4: confident.  
I am familiar with CLIP, model compression, and Kronecker‑factorized mappings, and I carefully checked the key equations, figures, and tables. Some unclarified implementation details prevent absolute certainty, but it is unlikely that I misunderstood the central contributions.