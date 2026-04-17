# Unveiling the Power of Shared Spaces: A Gating-Driven Mechanism for Semi-Supervised Domain Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Domain adaptation (DA) aims to enhance the generalization ability of models in scenarios where labeled data in the target domain is scarce. In DA research, semi-supervised domain adaptation (SSDA) can utilize the labeled information in the target domain more effectively compared to unsupervised domain adaptation (UDA), thus achieving superior transfer performance and gaining widespread attention. Existing SSDA methods implicitly learn feature spaces in the process of aligning feature spaces between domains; however, the underlying mechanisms remain insufficiently explored. To address this issue, this paper first theoretically reveals the advantages of learning a shared feature space for enhancing transferability. Based on our theoretical insights, we develop a  framework to learn a shared space, which is implemented by a gating-driven SSDA enhancement mechanism.  It is feasible to explicitly filters out inconsistent features across domains compared with existing methods. Extensive experimental results demonstrate the significant improvements of the proposed gating-driven enhancement mechanism on state-of-the-art SSDA models. Our code is anonymously provided in https://anonymous.4open.science/r/ICLR_8979.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies semi-supervised domain adaptation (SSDA) and proposes a gating-driven shared-space mechanism. A lightweight gating module is inserted between the feature extractor and classifier to “turn off” domain-specific channels and retain domain-invariant (shared) features. The authors provide a theoretical analysis arguing that enlarging the proportion of shared space reduces the source–target total variation (TV) distance, and they show plug-and-play gains when adding the gate to several SSDA baselines.

### Strengths
1）Simplicity & plug-and-play: The gating block is easy to integrate into existing SSDA pipelines without changing loss functions or training protocols.

2）Theory–practice linkage attempt: The paper tries to connect the idea of “more shared space → smaller TV → lower target error” with empirical TV measurements.

3）Broad empirical coverage: Multiple datasets and several representative SSDA baselines are evaluated; complexity overhead is small.

### Weaknesses
1）Limited novelty of the core premise :
The claim that “explicitly learning a shared space benefits SSDA” is already widely recognized and underpins many prior SSDA/DA approaches (e.g., domain-invariant representation learning, feature disentanglement, conditional alignment). The theoretical section mostly formalizes a well-known intuition rather than delivering new insights or substantially stronger guarantees. As presented, the theory’s necessity to the method is questionable and feels incremental.

2）“Explicitly turning off domain-specific channels” is not truly explicit:
The paper repeatedly emphasizes explicitly shutting down domain-specific features. However, the proposed gating is still a learned soft mask over latent channels driven by task loss. There is no external signal, constraint, or supervision that explicitly identifies domain-specific factors (style codes, backgrounds, frequency bands, etc.). Without architectural or optimization mechanisms that tie gates to domain cues, “explicit” remains more a narrative than a property of the method.

3）Modest and inconsistent performance gains:
Reported improvements over strong baselines are generally small (often ~0.5–2% and not universal across directions/shots). Some tasks show overlapping confidence intervals or marginal deltas. Given the simplicity of the gating, small gains are plausible, but then the contribution should be framed as a lightweight regularizer rather than a fundamentally new mechanism.

4）Theory–empirics gap:
The TV analysis assumes a clean separation between shared and domain-specific factors and (in parts) independence; actual deep features violate these assumptions. The measured TV drops after gating are encouraging, but do not isolate whether the effect comes from generic capacity control/sparsification rather than “shared-space enlargement” per se.

### Questions
Please refer to Weaknesses

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
This paper first analyzes SSDA and separates the learning feature space into an essential shared space and a domain-related space. They analyze the error bound of shared space learning and conclude that it is helpful to highlight the shared space learning. Then they propose a gate network to learn shared features across both domains, and experiments with the proposed network as a SOTA method's plugin demonstrate the effectiveness of the gating mechanism.

### Strengths
1. The paper is well written and easy to follow.
2. a straightforward but effective method.

### Weaknesses
1. Lack of novelty in the theoretical analysis. The shared space analysis is mainly based on the community's common sense and does not add anything new to the SSDA study, which severely limits the submission's novelty.
2. The details of the gated network are missing. The submission should clearly specify the gated network's design, including the number of layers, parameters, and channels, and so on. In the current version, it is hard to see why a gated network can improve learning in the shared feature space, which also limits the submission's novelty.

### Questions
1. Please clarify the details of the gated network.
2. Please clarify why gated network can improve the learning of shared feature space.
3. Please provide a variant analysis of gated network design. For example. if cnn/transformer/cross-attention/MLP is used as the gated network, how about the performance?

### Soundness
3

### Presentation
3

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
This paper investigates the role of shared feature spaces in Semi-Supervised Domain Adaptation (SSDA) and proposes a gating-driven mechanism to explicitly filter out domain-specific features and emphasize domain-invariant ones. The authors provide a theoretical analysis showing that focusing on shared features reduces the total variation distance between domains and improves target-domain generalization. They implement a lightweight gating module that can be easily integrated into existing SSDA frameworks (e.g., MME, CDAC, ECB), achieving consistent accuracy gains across several benchmarks such as DomainNet and Office-Home.

### Strengths
(1)	The paper provides a clear theoretical analysis linking shared feature spaces to reduced domain discrepancy, which grounds the proposed method in formal reasoning.
(2)	The proposed gating mechanism can be seamlessly incorporated into existing SSDA frameworks (e.g., MME, CDAC, ECB) without modifying their core objectives.
(3)	The method consistently improves performance across multiple benchmarks and settings, showing both effectiveness and stability.

### Weaknesses
(1)	The idea of leveraging shared feature spaces has already been explored in many prior SSDA and domain adaptation studies [1,2,3,4]. The authors’ related work does not discuss how this work provides a fundamentally new insight.
(2)	Some baselines, including a 2025 work cited by the authors, are not the most representative in the SSDA literature. It would strengthen the paper to compare with more recent and competitive baselines.
(3)	Although proposed mechanism improves the overall average performance, it does not achieve the best results on some specific transfer directions such as A→C and A→R in Table 3. The paper does not analyze or explain why the method underperforms in these cases.

[1] Shared space transfer learning for analyzing multi-site fmri data, NeurIPS’20
[2] Domain-specific feature unlearning for semi-supervised and unsupervised domain adaptation, ECCV’24
[3] Domain Separation Networks, NeurIPS’16
[4] Bridging Domains with Approximately Shared Features, Arxiv’24

### Questions
(1) Could the authors clarify what specific gap in prior shared-space research this paper aims to fill?
(2) What are the possible reasons for lower performance on specific transfer pairs such as A→C and A→R (Table 3)?
(3) Why were only subsets of DomainNet, Office-Home, and Office-31 chosen? (4) Do results generalize to other visual or non-visual domains (e.g., graph)?
(5) Can the gating mechanism also benefit unsupervised DA (UDA) or multi-source DA? Have the authors tried zero-shot domain shifts?
(6) Appendix C.5 mentions “minimal computational cost.” Can the authors provide concrete FLOPs or runtime comparisons versus baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops a framework to learn a shared space, which is implemented by a gating-driven SSDA enhancement mechanism. Furthermore, the paper theoretically reveals the advantages of learning a shared feature space for enhancing transferability.

### Strengths
While the paper theoretically analyzes the advantages of the shared feature space, offering valuable insights for newcomers to the field.  The overall logic of the paper is clear, the presentation is detailed, the theoretical hypotheses are elaborated on in depth, and the experimental results accurately demonstrate the experimental effects.

### Weaknesses
1. Novelty is limited. "Shared feature space" is a fundamental concept in domain adaptation research, and this idea has been prevalent in the field for over a decade. The paper theoretically analyzes the advantages of the shared feature space, offering valuable insights for newcomers to the field. However, it does not introduce groundbreaking theoretical innovations to the core paradigm.
2.  From an experimental validation perspective, the performance improvement brought by the proposed gating-driven SSDA enhancement mechanism is limited. As shown in Figure 3: (a)-(b), there is no significant observable change, making it difficult to discern the advantage of the proposed mechanism.
3. Results comparison in Table 2 and Table 3 has no reference citation. It is suggested to add these citations.
4. The dataset used in the paper can be expanded to include more datasets, and the scale of the dataset should also be verified using large-scale data.

### Questions
As listed in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
