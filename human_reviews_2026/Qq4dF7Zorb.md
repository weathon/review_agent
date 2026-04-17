# Class-Conditional Autoencoders with Adversarial Alignment for Multimodal Fusion

- Decision: Reject
- Scores: 4, 8, 4, 2, 2

## Abstract
Large-scale multimodal transformers excel at cross-modal reasoning but incur prohibitive computational costs and lack theoretical grounding. We propose **DEF+AAF**, combining *Discriminative Embedding (DEF)* with *Adversarial Alignment (AAF)* to achieve provably robust multimodal fusion. We prove that class-conditional variance contraction + Wasserstein barycenter alignment provides a tighter generalization bound (**Theorem 3**) than standard contrastive methods, reducing expected error by $O(\sqrt{M/N})$ where $M$ is modality count. On emotion recognition (IEMOCAP, MOSEI) and translation (Multi30k, How2), DEF+AAF matches transformer baselines at 2.4× fewer parameters and 1.6× lower FLOPs, with +8.4% robustness gain under 50% missing modalities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents DEF+AAF, a comprehensive multimodal fusion framework that integrates Discriminative Embedding Framework (DEF), a Class-Conditional Autoencoder (CCAE), and an Adversarial Alignment Framework (AAF). The model is theoretically grounded, aiming to enhance cross-modal alignment, reduce modality discrepancies, and improve generalization in multimodal tasks. The proposed framework is evaluated on both emotion recognition (IEMOCAP, MOSEI) and multimodal machine translation (How2, Multi30k) benchmarks. Experimental results show that DEF+AAF achieves comparable or superior performance to existing baselines while being more parameter-efficient and faster in training and inference.

### Strengths
1. Proposes a unified multimodal fusion framework (DEF+AAF) that combines discriminative embeddings, class-conditional autoencoders, and adversarial alignment, supported by theoretical justification.

2. Compared to large models like Transformers, DEF+AAF has lower parameter count and FLOPs, faster training and inference speed, while maintaining or even improving performance.

### Weaknesses
1. The baselines selected for machine translation (How2, Multi30k) are all from before 2020, and those for emotion recognition (IEMOCAP, MOSEI) are all from before 2022, lacking comparisons with the current state-of-the-art models.

2. Given that all baselines are pre-2022, the improvements of DEF+AAF on emotion recognition (IEMOCAP, MOSEI) are relatively limited.

3. Hyperparameter studies are insufficient. In Appendix B.2, the authors only compare three values for λ and two values for γ, which provides a very limited view of the model’s sensitivity. 

4. Although lighter than large Transformers, DEF+AAF still consists of multiple modules, which makes implementation and tuning complex. Moreover, the theoretical guarantees for homologous variance contraction and adversarial alignment rely on several assumptions, which may be difficult to satisfy in practice.

### Questions
1. How stable is the training of DEF+AAF considering its multiple interdependent modules?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a novel and theoretically grounded framework for multimodal fusion, combining a Class-Conditional Autoencoder (CCAE) with a Discriminative Embedding Framework (DEF) and an Adversarial Alignment Framework (AAF). The core objective is to learn compact, discriminative, and distributionally aligned multimodal embeddings in a computationally efficient manner. The method is extensively evaluated on machine translation (How2, Multi30k) and emotion recognition (IEMOCAP, MOSEI) tasks, demonstrating strong performance improvements over several state-of-the-art baselines while being more parameter- and compute-efficient.

The paper is well-written, methodologically sound, and makes significant contributions. The unification of discriminative and adversarial objectives under a single optimization perspective is a key strength. The empirical evaluation is thorough, including robustness analyses and efficiency comparisons.

### Strengths
- The proposed DEF+AAF framework offers a unified optimization perspective that elegantly combines variance contraction (via homologous and reconstruction losses), class separability, and distributional alignment (via adversarial training). This provides a more principled and interpretable approach compared to many ad-hoc fusion strategies.
- The paper is commendable for its theoretical contributions. Propositions, along with the proofs in the appendix, provide formal guarantees on intra-class variance contraction and distribution alignment, strengthening the methodological claims.
- Emphasis on Efficiency and Robustness.

### Weaknesses
- My primary concern is the selection of baselines in the mail part of the experiment, which are of date (2021, 2022, etc). Are there any recent SOTA baselines that can be added?
- The ablation definitions need to be clearer. While the main method is a combination of DEF and AAF, is the ablation of DEF or AAF itself needed? Or has it already been included in Table 3?

### Questions
- My primary concern is the selection of baselines in the mail part of the experiment, which are of date (2021, 2022, etc). Are there any recent SOTA baselines that can be added?
- The ablation definitions need to be clearer. While the main method is a combination of DEF and AAF, is the ablation of DEF or AAF itself needed? Or has it already been included in Table 3?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a lightweight framework for multimodal fusion, which aims to address the high computational cost and lack of theoretical grounding in existing large Transformer models. The proposed framework consists of several modules: 1) Class-Conditional Autoencoder
 Is used to map inputs from different modalities into a shared latent space that is conditioned on class information; 2) Discriminative Embedding Framework (DEF) enforces compactness and class separability using homologous and reconstruction losses, ensuring modality-aligned and semantically robust embeddings; 3) Adversarial Alignment Framework (AAF) introduces a dynamic fusion mechanism (similar to attention) to weight different modalities and uses Wasserstein-based adversarial training to align the distribution of the fused embedding with the distributions of the individual modal embeddings. The authors claim that this framework (DEF+AAF) surpasses strong existing baselines (like Transformer, MulT, etc.) on machine translation (How2, Multi30k) and emotion recognition (IEMOCAP, MOSEI) tasks with lower computational cost (FLOPS).

### Strengths
- The core idea of combining class information, modal cohesion, and distributional alignment is conceptually clear and technically sound;
- The paper reports not only on performance but also on parameters, FLOPs, and training/inference speed (Table 5);

### Weaknesses
- The paper writing & organization is poor. The illustration of introduction is too short, which makes readers hard to fully understand the motivation & goal of this work; In the related work part (Multimodal representation learning), the paper misses many Refs, e.g., “, such as early fusion (feature concatenation) or late fusion (decision-level combination),”, “Autoencoding-based methods extended”; The current writing quality significantly hinders readability and makes it difficult for readers to follow the paper’s logic and contributions, which is not acceptable for a top-tier venue such as ICLR.  I hope the authors could carefully revise these typos & writing issues before resubmission.
- In Tables 1&2, the latest compared approach is proposed in 2022, please include latest SOTA approaches for comparison.

The writing quality of this paper falls well below the standards expected at ICLR, making it difficult to follow. Moreover, the paper lacks comparisons with state-of-the-art methods. Therefore, I recommend rejection.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles the heavy computation and weak theoretical grounding in multimodal learning by proposing a lightweight yet principled fusion framework. Based on a Class-Conditional Autoencoder (CCAE), the method maps inputs into a class-aware latent space, while the Discriminative Embedding Framework (DEF) enhances intra-class compactness and preserves semantic consistency. To address cross-modal distribution gaps, the Adversarial Alignment Framework (AAF) employs a Wasserstein-based objective for dynamic alignment. Unified under a coherent optimization view, DEF and AAF achieve both efficiency and theoretical interpretability. Experiments on translation and emotion recognition benchmarks show consistent gains over Transformer, MulT, and MISA with significantly reduced FLOPs.

### Strengths
1. Proposed strategies are theoretically solid.
2. Experiments are partially effective.

### Weaknesses
1. Related work and baselines are outdated, mostly before 2022. Including recent multimodal fusion methods and robustness comparisons against missing/noisy modality approaches would strengthen the evaluation.
2. Mathematical notation is inconsistent and sometimes ambiguous. Theoretical analysis lacks formal proofs or derivations to support the claimed guarantees.
3. Prior methods such as conditional autoencoder, InfoNCE, and Wasserstein GAN with Gradient Penalty and the baseline methods are mentioned without proper citations.
4. The paper lacks implementation details regarding the hyperparameters used in Eq.9.
5. Figures 1–3 are not referenced in the main text. Figure 1 lacks a legend, and its visualization appears unrelated to the objective of the Homogeneous Loss. Moreover, Figures 2 and 3 do not specify the datasets used for the experiments.
6. The writing could be improved. Section 3 mentions two proposed methods, which should refer to DEF and AAF; however, AAF is described separately in Section 4, showing a structural oversight in the paper’s organization.

### Questions
1. The paper uses a mean squared error–based objective to reduce modality discrepancy. Has the potential loss of modality-specific information been considered when enforcing such cross-modal similarity?
2. In the contrastive regularization loss, how are the positive and negative sample pairs defined?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a lightweight multimodal method that combines a class-conditional autoencoder  with a discriminative embedding module and an adversarial aligner that learns sample-wise modality weights and aligns the fused code to each modality using Wasserstein training. The authors argue this unifies intra-class variance reduction, semantic preservation, and cross-modal distribution alignment with good efficiency. Experiments on translation and affect show consistent gains and robustness to missing/noisy inputs, supported by ablations. However, the role of class conditioning in MT settings, fairness of FLOP accounting, and some implementation details require clearer exposition.

### Strengths
1. Combines a class-conditional autoencoder with discriminative embedding and adversarial alignment to jointly achieve intra-class compactness, semantic preservation, and cross-modal distribution alignment under one objective.
2. Learns per-example modality weights and aligns the fused code to each modality, improving resilience when a modality is noisy or missing.
3. Demonstrates consistent gains on translation and affective benchmarks while using fewer parameters/FLOPs, indicating a favorable accuracy–efficiency trade-off.

### Weaknesses
1. The paper under-specifies how class conditioning is defined on tasks without explicit labels, whether class cues are needed at inference, and key implementation details, making replication difficult.
2. FLOP/latency comparisons appear to exclude external feature extractors, and decoding/tokenization protocols aren’t fully standardized across baselines; end-to-end efficiency and broader datasets/metrics would make the gains more convincing.
3. Pushing the fused code toward a Wasserstein barycenter can dilute rare but discriminative cues when modalities disagree; the paper lacks analyses or ablations vs. reliability-aware or top-k alignment variants to rule out this failure mode.

### Questions
1. How are class embeddings defined on Multi30k/How2, and are they required at inference?
2. Does Wasserstein-barycenter alignment dilute rare but discriminative cues when modalities conflict?

### Soundness
2

### Presentation
2

### Contribution
2
