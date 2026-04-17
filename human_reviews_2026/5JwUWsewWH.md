# PRISM: Progressive Robust Learning for Open-World Continual Category Discovery

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Continual Category Discovery (CCD) aims to leverage models trained on known categories to automatically discover novel category concepts from continuously arriving streams of unlabeled data, while retaining the ability to recognize previously known classes. Despite recent progress, existing methods often assume that data across all stages are drawn from a single, stationary distribution—a condition rarely satisfied in open-world scenarios. In this paper, we challenge this stationary-distribution assumption by introducing the Open-World Continual Category Discovery (OW-CCD) setting. We address this challenge with PRISM (\underline{P}rogressive \underline{R}obust d\underline{I}scovery under \underline{S}trea\underline{M}ing data), an adaptive continual discovery framework consisting of three key components. First, inspired by spectral properties, we develop a high-frequency-driven category separation technique that exploits high-frequency components—preserving more global information—to distinguish known from unknown categories. Second, for known categories, we design a sparse assignment matching strategy, which performs proximal sparse sample-to-label matching to assign reliable cluster labels to known-class samples. Finally, to better recognize novel categories, we propose an invariant knowledge transfer module that enforces domain-invariant category relation consistency, thereby facilitating robust knowledge transfer from known to unknown classes under domain shifts. Extensive experiments on the SSB-C and DomainNet benchmarks demonstrate that our method significantly outperforms state-of-the-art CCD approaches, highlighting its effectiveness and superiority.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
#### **Summary**: This paper studies Open-World Continual Category Discovery (OW-CCD) where unlabeled streams mix known and novel classes under domain shift across sessions, and proposes PRISM, a progressive, divide-and-conquer framework. PRISM first separates known vs. unknown via a high-frequency image decomposition and a 2-component GMM over prototype similarity scores (HCS). It then assigns sparse, reliable pseudo-labels to known-class samples using proximal optimal transport (SAM). For the unknown pool, it enforces domain-invariant category relations by aligning Plackett-Luce ranking distributions of similarities across style-perturbed views (IKT), and clusters novel classes with Affinity Propagation without a class-count prior. On SSB-C and DomainNet, PRISM reports consistent gains over strong CCD/GCD baselines (e.g., +15.1% / +12.3% vs. VB-CGCD on CUB-C original/corrupted), with ablations supporting each module’s contribution, tested with both DINOv1 & DINOv2.

### Strengths
- #### **S1. Well-posed, timely setting.:** The paper clearly formulates OW-CCD and explains why naive domain alignment can harm discovery (negative transfer), motivating the split-then-learn design.

- #### **S2. Compelling empirical evidence:** Strong improvements on SSB-C and DomainNet with multi-seed means ± std and component ablations; HCS beats entropy/energy separations.

- #### **S3. Clear technical details:** HCS→SAM→IKT is technically concrete (DFT mask + 2-GMM; proximal OT; PL-ranking KL) and presented with end-to-end pseudocode, aiding reproducibility.

- #### **S4.** Clear presentation, make it easy to understand the overal paper claim.

### Weaknesses
- #### **W1. Concerns on HCS (high-frequency separation) and a brittle bimodality assumption:**  PRISM’s HCS assumes that high-frequency components carry domain-invariant “global semantics”, then splits a stream into known/unknown with a 2-component GMM on a cosine-similarity–based score S(x) and a fixed 0.5 decision rule. The concern is due to the assumption that high frequencies are more domain-invariant/global than low frequencies is non-standard and is supported only by a few visuals and an empirical “often bimodal” observation of S(x). There’s no stress test for cases where the density is not bimodal (e.g., overlapping or multimodal mixtures), or where frequency energy shifts with style/texture corruptions. The split hinges on the mask ratio r and the 2-GMM fit; failure modes (bad EM fits, non-bimodal spectra, class-imbalance drift) aren’t characterized. The paper introduces r and report sensitivity plots, but the text doesn’t establish robustness across tasks/backbones/shifts—only that varied r, ε in Appendix Fig. 6.

- #### **W2. Scalability gap around clustering (Affinity Propagation) in an open-world streaming setting:** Novel-class discovery uses Affinity Propagation (AP) each session on the unknown subset embeddings. This concerns is due to AP is O(n^2) in similarities and memory; in a streaming, open-world scenario this is a bottleneck. The appendix’s compute profile discusses FFTs/SAM/IKT and quotes “5.2 GFLOPs” and “0.65 s” per iteration, claiming the overhead “scales well”, but it doesn’t analyze AP’s time/memory cost or performance under large, long streams. What to show: either (i) replace AP with a streaming/mini-batchable clustering method (e.g., online k-means, DP-means, coresets), or (ii) add a clear AP complexity plus wall-clock and memory study at increasing stream sizes (curves), and (iii) demonstrate stability when unknowns accumulate over many stages.

- #### **W3. Positioning/novelty risk:** OW-CCD may read as “CCD plus domain shift” (an incremental protocol tweak). The paper presents OW-CCD as the first step beyond single-domain-per-stage CCD, arguing prior CCD/GCD assume a stationary domain per stage, but the setting itself can also be interpreted as a composed variant—continual discovery combined with domain shift—rather than a fundamentally new learning paradigm. The paper would benefit from a crisper separation from prior “domain-varying” or “corruption-varying” CCD/GCD works and from OSDA/SFDA lines  or domain-class incremental learning [a] adapted to novelty discovery (too much of this currently sits in the appendix).

### Questions
- #### **Q1. Bimodality & fallback:** How consistently does S(x) exhibit a bimodal pattern across sessions/datasets, and is there a fallback (e.g., energy/entropy split) when EM finds overlapping components?

- #### **Q2.** Results include synthetic corruptions and cross-style domains; how does PRISM fare under spatial distortions (large occlusions, extreme down-sampling) that specifically degrade high-frequency cues presumed to be invariant?

- #### **Q3.** Why Affinity Propagation over alternatives (e.g., density-based methods) for unknowns? Any observed failure modes (e.g., over-fragmentation) and mitigations?

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
4

### Summary
This paper introduces the Open-World CCD setting and proposes PRISM, a divide-and-conquer framework with three components: (i) High-Frequency-driven Category Separation (HCS) to split known vs. unknown using spectral cues and a GMM on a density score; (ii) Sparse Assignment Matching (SAM) for proximal-OT based pseudo-labeling of known classes; and (iii) Invariant Knowledge Transfer (IKT) that aligns Plackett–Luce ranking distributions across styles to stabilize transfer to novel classes. Extensive experiments on SSB-C and DomainNet show performance improvement.

### Strengths
- Overall, I appreciate the clear problem formulation. The proposed OWCCD is well motivated and distinct from the static GCD and CCD assumptions under single domain settings.
- The proposed method seems reasonable. HCS uses DFT high-frequency components with a 2-component GMM for known/unknown separation. SAM applies a proximal ℓ2 OT for sparser. IKT models listwise relations via PL and enforces consistency between original and style-perturbed rankings.
- The paper is well organized and easy to follow.

### Weaknesses
- The paper requires substantial revisions before acceptance. The font sizes in Tables 1 and 2 should be unified, as the current versions are difficult to read. Figure 1 also has fonts that are too small. The derivation of Equation 3 needs to be clarified, particularly the relationships among F^{l}, F^{h}, and F^{-1}. In addition, the introduction should be improved to enhance readability.
- The method lacks ablations on key design choices, maybe could include: (i) removing frequency decomposition to test spectral necessity, (ii) replacing PL with pairwise or listwise surrogates and varying \lambda_{IKT}, and (iii) adjusting OT regularization ε and SAM steps to analyze the accuracy–sparsity–speed trade-off.
- Provide a small theory/intuition section on why high-frequency cues are more domain-invariant in practice (beyond Fig. 1).
- Although the method is prototype-based, it does not compare with recent prototype-based approaches, such as [1].

[1] Happy: A debiased learning framework for continual generalized category discovery. NIPS 2024

### Questions
- Does HCS remain reliable when S(x) is not clearly bimodal (e.g., high domain overlap)? Any fallback beyond the 0.5 posterior cut?
- How often are prototypes updated across sessions, and how sensitive are results to stale prototypes feeding HCS/IKT?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new category discovery task: Open-World Continual Category Discovery (OW-CCD), which challenges the single-domain data assumption in the previous CCD task.

### Strengths
1.	New problem setting:   This paper addresses a realistic open-world streaming setting (OW-CCD) that extends and generalizes prior continual and generalized category discovery work, directly tackling the limitations of stationary-domain assumptions in earlier CCD literature.
2.	The proposed method is interesting, especially since using the frequency domain to provide auxiliary supervision signals is reasonable.
3.	The proposed method achieved good results on the OW-CCD.

### Weaknesses
1.	I agree that introducing frequency domain analysis into the GCD task is an interesting and intuitively plausible approach. However, the paper's core claim that high-frequency components contain more domain-invariant category discriminative information lacks sufficient experimental support. The current Fig. 1 serves more as an idea demonstration than a rigorous quantitative argument. To enhance persuasiveness, the authors should provide more direct evidence, such as comparing the separation levels of known and unknown classes in high- and low-frequency feature spaces via t-SNE visualization, or quantifying the accuracy advantage of high- and low-frequency features in classifying new versus old categories.
2.	The proposed method introduces a non-trivial number of hyperparameters, among which the mask ratio r is particularly critical and appears to be highly sensitive.  
3.	I don't entirely agree with the motivation behind the IKT section. Its optimization principle requires that, regardless of style variations, the similarity ranking between unknown samples and known classes remains consistent—essentially treating known classes as anchors for discovering new ones. However, whether this approach genuinely promotes effective discovery of new classes is debatable. Moreover, in current GCD/CCD research, known classes on some datasets exhibit low accuracy. I think this constraint may limit the representational space for new classes.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a new setting called Open-World Continual Category Discovery (OW-CCD), arguing that existing CCD works usually assume a single domain, which is unrealistic for real data streams. To address this, the authors introduce PRISM, a three-part framework:

1. High-Frequency-Driven Category Separation (HCS): use Fourier-domain decomposition to keep high-frequency components and compute a density score over high-frequency features to separate “known-like” vs. “unknown-like” samples. 
2. Sparse Assignment Matching (SAM): for samples considered “known-like”, perform proximal-regularized optimal transport to assign them to known-class prototypes, making the cluster assignment sparser and clearer.
3. Invariant Knowledge Transfer (IKT): for unknown samples, perform low-frequency perturbation and enforce ranking-level consistency to known prototypes using a Plackett–Luce distribution, so that semantic relations to known classes are invariant across domains. 

Experiments on SSB-C and DomainNet show consistent gains over recent CCD baselines; ablations show that each of HCS / SAM / IKT contributes and all three together give the best numbers.

### Strengths
1. Plausible problem setting: the paper targets CCD under domain shift, which is a realistic gap in current CCD/GCD work.

2. Good empirical performance: shows strong results on multiple benchmarks.

3. Readable: pipeline (HCS → SAM → IKT) is clearly described and easy to follow.

### Weaknesses
1. **Weak and insufficiently argued link to continual learning.**
   The paper claims to tackle an open-world, continual setting, but the method itself is almost entirely tailored to domain shift. In practice, the pipeline treats the problem as “domain-shift GCD + CCD,” without discussing what actually becomes new or harder once we make it continual (e.g., catastrophic forgetting, order sensitivity, memory constraints). All three modules are clearly shift-oriented — HCS uses frequency cues to be domain-robust, SAM does cross-domain pseudo-labeling, and IKT enforces cross-domain relation consistency — yet none of them is a CL-specific mechanism. As a result, there is a visible gap between the problem the paper advertises (open-world *continual* category discovery) and the solution it delivers (a per-session, domain-shift-aware GCD pipeline). This naturally raises the question: why isn’t this simply framed as “domain-shift GCD” [1,2]? To make the claim convincing, the paper should spell out (i) what challenge is *unique* to their OW-CCD formulation and cannot be handled by combining existing domain-GCD [1, 2] and CCD methods, and (ii) how PRISM actually uses the *continual* nature of the stream, instead of just re-running the same 3-stage block on every session.

2. **Unclear tractability of IKT.**
   In IKT, the paper models the distribution over **all** permutations of the known-class prototypes using a Plackett–Luce distribution. This is conceptually reasonable, but as the number of known classes grows (e.g., C > 100), enumerating or even implicitly handling all C! permutations becomes infeasible. The paper does not explain how this is solved in the actual implementation. Without this detail, it is hard to judge whether IKT is an actually runnable module or mainly a theoretical description.

3. **Comparisons do not fully match the stated goal.**
   Since the method is explicitly designed to handle domain shift, it should also be compared to domain-shift GCD methods [1,2]. At minimum, the authors could (a) adapt HiLo [1] or CDAD-Net [2] to the same domain-shift + CCD, or (b) evaluate PRISM directly in a domain-GCD setup to demonstrate that it handles shift better than prior work [1, 2]. At present, most baselines are standard CCD methods that were *not* written for domain shift, so the performance gaps are less conclusive — the method may simply be capitalizing on a challenge (domain shift) that the competing methods were never tuned for.

[1] Hilo: A learning framework for generalized category discovery robust to domain shifts.

[2] Cdad-net: Bridging domain gaps in generalized category discovery.

### Questions
1. **Failure cases of HCS.** On which SSB-C corruption types does HCS fail to produce a clearly bimodal similarity distribution? Showing 1–2 negative examples in the appendix would make the claim “HCS yields separable scores” more convincing.

2. **Session/domain order.** On DomainNet, do you randomize the order of arriving domains, or do you always use the same source (e.g., Real) and the same progression of target domains? If the order is fixed, to what extent does PRISM rely on this order (i.e., does it overfit to an “easy → hard” schedule)? Have you tried choosing a non-Real source domain?

### Soundness
2

### Presentation
3

### Contribution
2
