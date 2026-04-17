# UDANG: Unsupervised Domain Adaptation with Neural Gating for learning invariant representation of subspaces

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
The key assumption of deep learning is that the data that the model will be tested on (target domain) are drawn from the same distribution as the data it was trained on (source domain). Breaking this assumption can lead to a significant drop in performance despite having similar underlying features between the source and target domains. Unsupervised Domain Adaptation (UDA) involves using unlabeled samples from the target domain, in addition to labeled samples from source domain, to train a model that can perform well on the target domain. Many existing UDA approaches rely on domain adversarial training (DAT) to reduce domain shift. Although effective, they do not explicitly disentangle the learned features into task-specific and domain-specific components. As a result, the features despite appearing to be domain invariant, may still contain domain-specific biases. To address this, we propose a novel method, UDA with Neural Gating (UDANG), that utilizes a dual adversarial objective to learn an adaptive gating which dynamically route each feature dimension to either the domain or task subspace. Using our strategy, networks have the ability to effectively disentangle task-specific features from domain-specific ones. We validated our approach in multiple datasets and network architectures for image classification, demonstrating strong adaptation performance while retaining the features for discerning the domain.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UDANG, an unsupervised domain adaptation framework that introduces a neural gating mechanism to disentangle task-relevant and domain-specific subspaces. Building upon the Domain Adversarial Training (DAT) paradigm, UDANG adds a dual adversarial objective: one discriminator for task alignment and another for domain alignment. A Gumbel–Sigmoid gating function dynamically routes feature dimensions into the task or domain subspace during training, aiming to learn invariant representations. Experiments on VisDA-2017 and Office-31 demonstrate competitive performance compared to recent UDA baselines such as MIC, SDAT, and TVT.

### Strengths
1. The paper is built around a well-defined goal — separating task-relevant and domain-specific representations in UDA — and follows this motivation throughout. The narrative is coherent and easy to follow.

2. The experimental setup is carefully designed, with appropriate baselines, multiple datasets, and detailed ablations. The reported gains are consistent and appear reproducible.

3. The introduction of a gating module makes the feature routing process more transparent, and the visualizations help to qualitatively understand the model’s behavior.

4. The manuscript reads smoothly and provides sufficient detail to reproduce the method. Figures are informative and the comparisons are fair.

### Weaknesses
1. The overall framework is still built upon standard adversarial adaptation (DAT) and domain separation ideas. The proposed neural gating resembles earlier disentanglement strategies, offering more of a structural variation than a conceptual breakthrough.

2. The gating mechanism is introduced intuitively, but there is no analysis explaining why it should improve invariance or stability. The dual-adversarial setting also lacks formal discussion on convergence or disentanglement guarantees.

3. Although GradCAM plots are shown, the work does not measure or validate how well the gated subspaces are separated or independent.

4. Experiments are restricted to image classification tasks. It is unclear whether the method generalizes to other modalities or more complex adaptation settings such as segmentation or multi-source transfer.

5. The paper would benefit from a sharper articulation of how UDANG differs from established models like DSN or CDAN. Currently, the overlap is substantial, which makes the contribution appear incremental.

### Questions
1. How is UDANG fundamentally different from Domain Separation Networks (DSN) or Conditional Adversarial DA (CDAN)?

2. If the gating module is removed, does the dual-adversarial structure still perform comparably?

3. How sensitive is the model to the gating temperature parameter (τ)?

4. Have you tested cross-domain generalization beyond visual datasets?

5. Is it possible to drop the domain discriminator during inference to reduce computational overhead?

### Soundness
3

### Presentation
4

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
The paper proposes UDANG, a UDA method that disentangles task-specific vs domain-specific features via a dual adversarial objective and an attention-style gating that routes each feature dimension into a “task” or “domain” subspace. The loss couples DAT with a task-side adversary and MCC on target; gates use a Gumbel-Sigmoid hard/soft path. Evaluated on VisDA-2017 and Office-31 with ResNet/ViT backbones, UDANG is competitive with recent SOTAs and wins some Office-31 transfers; qualitative UMAP/Grad-CAM support is provided.

### Strengths
1. **Clear objective & architecture:** a tidy extension of DAT with symmetric adversaries and MCC; the gating mechanism is simple and differentiable.  
2. **Empirical competitiveness:** ≥90% mean on VisDA-2017 (ViT) and SOTA on 4/6 Office-31 tasks; results span CNN/ViT. 
3. **Diagnostics:** UMAP class/domain separation and Grad-CAM analyses are helpful; authors discuss an observed bias case.

### Weaknesses
1. **Novelty vs prior disentanglement/DAT:** The idea of separating task/domain subspaces with adversaries echoes domain-separation lines; the paper would benefit from a sharper contrast to DSN-style methods and MIC-like masking beyond qualitative claims. (Related work listed, but positioning remains light.) 
2. **Gating evidence:** No quantitative probe of gate assignments (e.g., sparsity, stability across seeds, correlation with domain cues). The Gumbel temperature is fixed; sensitivity is unknown.  
3. **Ablation depth:** Missing ablations for each loss term (remove MCC / each adversary / gate ->  identity), and for alternative routers (soft masks, top-k, per-token vs per-channel). Current gains are solid but not uniformly superior to MIC/TVT on VisDA. 
4. **Theory claims:** The method is positioned as learning “invariant representations of subspaces,” but there’s no formal identifiability or invariance guarantee, largely empirical.

### Questions
See weakness above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to solve the drawback of the previous domain adversarial training methods. The authors propose UDA with Neural Gating, that utilizes a dual adversarial objective to learn an adaptive gating which dynamically route each feature dimension to either the domain or task subspace.

### Strengths
- The proposed method is straightforward.

### Weaknesses
- It seems that the proposed method is not superior to previous methods in most cases. It is hard to persuade the reviewer that the proposed method is ready for an ICLR publication.
- The presentation of this work may need improvement. For example, replacing the current figures with some high-definition ones.
- Some arguments in the paper need further justification.

### Questions
- In Table 1 and Table 2, the reviewer sees that the method is no better than some methods proposed in 2019 (CAN in Table 1 and Table 2, TVT in Table 2). In this case, why would we choose the proposed UDANG method?
  - Are there any more recent works, for example, some works published in the 2024/2025 conference and journals. The baselines come from more than two years ago in Table 1 and Table 2. The reviewer thinks we need more recent baselines to show the superiority of the proposed method.

- What are the core contributions of the work? The Attention Gating Network or the dual branch design? How do different components contribute to the whole system? The reviewer thinks we need some analysis of the different components in the system.

- In lines 072-073, the authors claim "While foundation models learn broad representations, they do not inherently address the fundamental problem of domain shift." Why "they do not inherently address the fundamental problem of domain shift."? Is there any empirical evidence for this claim. We know that there are recent VLMs like Qwen-VL, the training data of such VLMs is large-scale, will the suffer from the issue?

- When the reviewer zooms in the figures in the paper, the reviewer finds that it is not clear. Could the authors replace these figures with some high-definition ones?

### Soundness
2

### Presentation
2

### Contribution
2
