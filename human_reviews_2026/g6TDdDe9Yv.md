# Metadata-Agnostic Decentralized Learning

- Decision: Reject
- Scores: 8, 2, 4, 2

## Abstract
Decentralized learning enables collaborative model training without sharing raw data, offering strong privacy benefits. However, many existing studies in decentralized learning research rely on an unrealistic assumption that all participants can share metadata such as class labels and the total number of categories. This assumption, which we term Metadata-Dependent Supervised Learning (MDSL), fails to reflect the diversity and autonomy of real-world participants. In contrast, we propose MAZEL: Metadata-Agnostic Zero-shot Learning, a framework that eliminates the need for shared metadata by leveraging CLIP-based zero-shot classification. MAZEL enables more realistic and flexible decentralized learning, where clients can dynamically join or leave without requiring predefined output heads. Our contributions are fourfold: (1) We formalize the distinction between MDSL and MAZEL; (2) we show that standard claims about performance degradation and slow convergence in MSDL-based decentralized learning may not hold under MAZEL; (3) we provide benchmarks using up to 8–16 diverse datasets to rigorously evaluate newly proposed decentralized learning methods under real metadata-agnostic cases; and (4) we propose two-stage and cosine gossip schedulers to optimize communication efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Proposes a new framework for decentralized learning under realistic conditions without requiring shared metadata. Uses CLIP-based image-text embeddings for zero-shot classification. Eliminates the need for shared metadata or predefined classification heads. Tackles for: 
Heterogeneous label spaces (different classes per participant).
Dynamic participation (nodes join/leave).
Privacy constraints

### Strengths
- Metadata Independence: No need for shared class labels or predefined classification heads
making it suitable for heterogeneous environments.
- Scalability; dynamic participation of nodes (joining/leaving) without major reconfiguration.
- Improved Generalization: accuracy across diverse datasets.
- Efficient Communication: Gossip-based schedulers reduce communication overhead and increase privacy and avoid sharing sensitive metadata

### Weaknesses
This design reduces transparency, making it harder to audit or explain the model, and makes specific predictions, raising accountability issues.

It relies heavily on CLIP embeddings for zero-shot classification. These models have pre-biases, which can lead to unfair outcomes in sensitive domains like healthcare or finance, reducing their real-time capability.

Writing:
- Dense sentences that pack too many ideas, reducing readability. Break the sentences for clarity.
- Inconsistent capitalization: like Zero-Shot Learning and metadata-agnostic

### Questions
- How do you tackle equity in participation? Nodes with limited resources may struggle to compute embeddings or participate effectively, creating a digital divide.
- While MAZEL avoids sharing metadata, gossip-based communication still involves exchanging model updates, which could be reverse-engineered to infer private data. Is there a study to test for this reverse generation of data for privacy analysis?
-

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces MAZEL (Metadata-Agnostic Zero-shot Learning), a new decentralized learning framework that eliminates the common but unrealistic assumption that all participants share metadata (e.g., class labels or the number of categories). Instead, MAZEL allows each node to maintain distinct label spaces and uses CLIP-based zero-shot classification to align local models without metadata exchange. The authors further propose a new evaluation metric, Gossip Gain (GG), to quantify improvements in global generalization through model gossiping. Experimental results demonstrate that MAZEL achieves faster convergence and stronger generalization than traditional Metadata-Dependent Supervised Learning (MDSL) across multiple decentralized settings, highlighting its practicality and scalability.

### Strengths
1. The paper formulates a highly relevant setting in which clients can have different category spaces and participate without any metadata sharing. This setup better reflects real-world decentralized systems, where label synchronization is impractical or privacy-restricted.

2. The newly proposed Gossip Gain (GG) provides a meaningful measure of communication efficiency and model improvement across nodes. It helps evaluate how gossip-based communication contributes to global performance—a useful addition to decentralized learning research.

3. The experiments demonstrate that MAZEL outperforms MDSL in both global generalization and convergence speed, suggesting that the common assumptions in prior work may not hold in more realistic settings.

### Weaknesses
1. The paper reports several findings that contradict established conclusions in metadata-dependent supervised learning, such as faster convergence under MAZEL. However, it lacks a theoretical analysis to explain these results or to formalize why CLIP-based zero-shot learning behaves differently in decentralized optimization.

2. The proposed framework heavily relies on CLIP’s pretrained text–image representations. It remains unclear whether the same conclusions would hold when using weaker or non-multimodal backbones.

3. Although the paper introduces extensive benchmarks, the core analysis of MAZEL vs. MDSL is only conducted on CIFAR-100 and Kvasir V2. Including additional datasets (e.g., CIFAR-10, Tiny-ImageNet, or DomainNet) would reduce the risk of dataset-specific conclusions.

4. Experiments are mainly conducted with ViT-B/32. Results with stronger or smaller models (e.g., ViT-B/16, ViT-L/14) would strengthen the evidence for the generality and scalability of the proposed method and the Gossip Gain metric.

5. The manuscript contains several small errors, such as a citation formatting issue at line 188 and a line-break error at line 190. Proofreading is recommended to improve readability.

### Questions
1. Can the observed differences between MAZEL and MDSL (e.g., faster convergence, better global generalization) be theoretically explained under certain optimization or information-sharing assumptions?

2. If CLIP performs poorly on downstream tasks with weak text–image alignment, would MAZEL still maintain its advantages over MDSL?

3. Have you considered evaluating the MAZEL framework with other multimodal or vision-only backbones, such as BLIP, DINOv2, or SigLIP, to verify the generality of your findings?

### Soundness
3

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
This paper identifies a critical discrepancy between the typical evaluation of decentralized learning algorithms and their real-world applicability. The authors term the common experimental setup, which assumes shared knowledge of metadata (e.g., class labels and counts), as Metadata-Dependent Supervised Learning (MDSL). They propose an alternative framework, Metadata-Agnostic Zero-shot Learning (MAZEL), which leverages pre-trained vision-language models like CLIP to eliminate the need for such metadata exchange. Through empirical evaluation, the paper argues that widely-held beliefs about decentralized learning—namely, poor generalization of local models and slow convergence—are artifacts of the unrealistic MDSL setting and do not necessarily hold under the more practical MAZEL framework. The authors also contribute 8-site and 16-site benchmarks for evaluating algorithms in this new setting and propose simple gossip scheduling heuristics.

### Strengths
1. The paper's primary strength is its conceptual contribution. The formalization of the MDSL vs. MAZEL dichotomy is a sharp and significant insight that challenges the foundations of how a large body of work in decentralized learning is evaluated. This is not an incremental algorithmic improvement but a call for a paradigm shift in experimental methodology, which is highly valuable.

2. The paper is exceptionally well-written and clearly articulated. The problem is well-motivated with real-world examples (e.g., healthcare, business confidentiality). Figure 1 provides an excellent visual explanation of the core difference between MDSL and MAZEL. The structure is logical, and the authors' arguments are easy to follow.

### Weaknesses
1.  While the problem formulation (MAZEL) is novel and insightful, the proposed solution is a rather straightforward application of existing technology. The core idea is to replace a standard classification head with CLIP's zero-shot classification mechanism. This is not, in itself, a novel technical contribution. The paper feels more like an evaluation framework proposal or a position paper with supporting experiments, rather than a paper presenting a new, technically deep algorithm. The primary novelty is in identifying the problem, not in the complexity of the solution.

2. The introduction of the "Gossip Gain (GG)" metric feels ad-hoc. The paper provides a motivation, but the metric is not standard and its properties are not analyzed. It is unclear why this specific formulation, $GG = (\frac{MMGA}{AGA} - 1) \times 100%$, is superior to simply reporting and analyzing the gap between MMGA and AGA.

### Questions
1. The most significant omission is the lack of comparison to personalized decentralized learning methods. Could the authors provide an experimental comparison against a baseline where only the model backbone (feature extractor) is shared via gossip, while each node maintains its own private classification head? This seems to be a natural approach to handle heterogeneous label spaces without sharing metadata and would serve as a much stronger baseline to demonstrate the advantages of the MAZEL framework.

2. The MAZEL framework's success is tied to CLIP's performance. How does the framework's effectiveness degrade if a weaker vision-language model is used? Furthermore, can the authors elaborate on a potential path forward for applying MAZEL in domains where no aligned multi-modal embeddings are available? Does the entire concept collapse without a model like CLIP?

3. Regarding the MDSL baseline, the experiments use a frozen CLIP encoder and only train a linear classification head. Is this a fair and strong representation of MDSL? Would fine-tuning the entire model, or using a more complex (e.g., 2-layer MLP) head, result in a stronger MDSL baseline that might narrow the reported performance gap with MAZEL?

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
3

### Summary
The paper argues that much decentralized learning (DL) work implicitly assumes shared metadata (labels, #classes). It proposes MAZEL, a “metadata-agnostic” alternative that replaces supervised heads with CLIP-style zero-shot classification and runs gossip averaging; it also introduces 8-site and 16-site heterogeneous benchmarks and explores simple gossip schedulers. Empirically, MAZEL often attains higher “global test accuracy” than supervised MDSL under Dirichlet non-IID CIFAR-100 and on broader multi-dataset suites.

### Strengths
1. The paper clearly articulates the “metadata-dependent” assumption and offers multi-dataset (8/16-site) benchmarks that better reflect cross-domain heterogeneity in DL evaluations. 
2. MAZEL’s formulation (text embeddings as class weights + gossip) is straightforward and compatible with standard vision backbones; the paper also explores practical gossip schedulers. 
3. On CIFAR-100 and Kvasir V2, MAZEL improves AGA substantially vs. MDSL; extended 8/16-site results show competitive ALA/AGA/MMGA under several scheduler settings.

### Weaknesses
1. Too many typos in this paper. (See Questions)
2. MAZEL objective. The normalized cosine scores are introduced and then dropped “for analytical convenience,” but no justification is provided that removing the normalization keeps the classifier calibrated or preserves the intended geometry. 
3. Algorithm 1 contains syntactic errors (“for t = 0 T”, “end for=0”) and half-line breaks in variable names, which makes the update ambiguous (e.g., θ^(t+½)ᵢ). This should be fixed to unambiguously specify the local-then-gossip update. 
4. In §3, the transition from population risks to ERM skips the standard normalization by total sample count (or per-site weighting), which matters when site sizes differ.
5. Conceptually, MAZEL is a combination of CLIP zero-shot classification and standard gossip averaging. The paper does not sufficiently differentiate this from existing decentralized model merging or zero-shot decentralized/FL alternatives; theory is limited to restating objectives. A deeper comparison and, ideally, some analysis (e.g., when zero-shot priors help consensus) are missing. 
6. No error bars or multi-seed statistics are shown; convergence step counts are reported but without runtime/throughput or memory. (See Table 1.) 
7. Missing comparisons to strong decentralized personalization/merging methods under heterogeneous tasks (e.g., task-vector/TIES-merging as a gossip step; decentralized CLIP adaptation baselines).

### Questions
1. Too many typos in this paper:
- Abstract: “exisitng studies” → existing. 
- §3.1 first sentence: “allows collaboratively train… without the of control of a central sever” → allows collaborative training… without the control of a central server. 
- §3.1 personalized setting: stray bracket after citation “Kharrat et al., 2024)]”. 
- Algorithm 1: “for t = 0 T do” (missing “to”); last line “end for=0”. 
- §5 datasets list: “SHVN Netzer et al. (2011)” → SVHN.  
- Minor subject–verb agreement: “The dataset assigned to each site are biased…” → is biased. 
- “In ??, we illustrate a typical decentralized learning process” 
2. MAZEL objective. The normalized cosine scores are introduced and then dropped “for analytical convenience,” but no justification is provided that removing the normalization keeps the classifier calibrated or preserves the intended geometry. 
3. Algorithm 1 contains syntactic errors (“for t = 0 T”, “end for=0”) and half-line breaks in variable names, which makes the update ambiguous (e.g., θ^(t+½)ᵢ). This should be fixed to unambiguously specify the local-then-gossip update. 
4. In §3, the transition from population risks to ERM skips the standard normalization by total sample count (or per-site weighting), which matters when site sizes differ.
5. Conceptually, MAZEL is a combination of CLIP zero-shot classification and standard gossip averaging. The paper does not sufficiently differentiate this from existing decentralized model merging or zero-shot decentralized/FL alternatives; theory is limited to restating objectives. A deeper comparison and, ideally, some analysis (e.g., when zero-shot priors help consensus) are missing. 
6. No error bars or multi-seed statistics are shown; convergence step counts are reported but without runtime/throughput or memory. (See Table 1.) 
7. Missing comparisons to strong decentralized personalization/merging methods under heterogeneous tasks (e.g., task-vector/TIES-merging as a gossip step; decentralized CLIP adaptation baselines).

### Soundness
3

### Presentation
3

### Contribution
2
