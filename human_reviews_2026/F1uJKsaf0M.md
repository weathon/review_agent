# Modality Alignment across Trees on Heterogeneous Hyperbolic Manifolds

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Modality alignment is critical for vision-language models (VLMs) to effectively integrate information across modalities. However, existing methods extract hierarchical features from text while representing each image with a single feature, leading to asymmetric and suboptimal alignment. To address this, we propose Alignment across Trees, a method that constructs and aligns tree-like hierarchical features for both image and text modalities. Specifically, we introduce a semantic-aware visual feature extraction framework that applies a cross-attention mechanism to visual class tokens from intermediate Transformer layers, guided by textual cues to extract visual features with coarse-to-fine semantics. We then embed the feature trees of the two modalities into hyperbolic manifolds with distinct curvatures to effectively model their hierarchical structures. To align across the heterogeneous hyperbolic manifolds with different curvatures, we formulate a KL distance measure between distributions on heterogeneous manifolds, and learn an intermediate manifold for manifold alignment by minimizing the distance. We prove the existence and uniqueness of the optimal intermediate manifold. Experiments on taxonomic open-set classification tasks across multiple image datasets demonstrate that our method consistently outperforms strong baselines under few-shot and cross-domain settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes "Alignment across Trees," a novel method for aligning vision and language modalities by constructing symmetric, hierarchical feature representations for both. The core idea is to address the asymmetry in prior work where hierarchical text features are aligned with a single, flat image feature. The method introduces two main contributions: 1) a semantic-aware visual feature extraction framework that uses cross-attention with textual cues to build a hierarchical tree of visual features from intermediate Transformer layers, and 2) a heterogeneous manifold alignment algorithm that embeds the image and text feature trees into separate hyperbolic manifolds with distinct, learnable curvatures. To align these, the paper defines a novel distance measure between heterogeneous hyperbolic manifolds and learns an optimal intermediate manifold. The approach is evaluated on taxonomic open-set classification, where it shows consistent and significant improvements over strong baselines.

### Strengths
1. The paper clearly identifies a key limitation in existing VLMs, asymmetric feature representation, and proposes a well-motivated solution.

2. The use of heterogeneous hyperbolic manifolds with learnable curvatures to capture the intrinsic geometry of each modality is technically sophisticated. The formulation of an intermediate manifold for alignment, supported by proofs of existence and uniqueness, provides a strong theoretical foundation. Even though I do not have enough experience in this line of work to confirm with certainty the correctness of the proofs.

3. The method demonstrates substantial and consistent performance gains over strong baselines across multiple datasets, metrics, and experimental settings (few-shot and generalization).

### Weaknesses
1. The paper’s literature review, while good, overlooks several relevant lines of work. It would be stronger if it discussed the concept of emergent alignment in models, and positioned the proposed "intermediate manifold" with respect to related concepts such as:
    - The idea of a universal or shared latent space, as explored in work on relative representations [1,2]. The proposed "intermediate manifold" seems to be a generalization of this concept to heterogeneous hyperbolic spaces.
    - The general problem of identifiability in representation learning (e.g., [3]), as the proposed work is implicitly trying to learn an identifiable hierarchical structure


2.  Some of the visual results used to support the paper's claims are difficult to interpret without further evidence. For instance, the qualitative improvement in clustering in Figure 5 is not immediately obvious, and the attention map visualizations in Figure 6 lack a crucial comparison to the baseline.

3. While the paper includes ablations, further analysis would strengthen the claims. For example, an ablation on the novel manifold distance definition and an analysis of the learned curvatures would provide deeper insight into the method's behavior.

--- 

[1] Moschella, et al, "Relative representations enable zero-shot latent space communication," in ICLR, 2023.

[2] Cannistraci, et al, "From Bricks to Bridges: Product of Invariances to Enhance Latent Space Communication," in ICLR, 2024.

[3] Kivva, et al, "Identifiability of deep generative models without auxiliary information," NeuriIPS 2022.

### Questions
1. The manuscript introduces a new function D to measure the distance between hyperbolic manifolds. Is this function a formal distance metric (i.e., does it satisfy non-negativity, identity of indiscernibles, symmetry, and the triangle inequality)? A brief discussion on this would be helpful.

2. The learnable curvatures are an interesting component. Could the authors provide some analysis on the learned values? For instance, do textual and visual manifolds for semantically similar/distinct data end up with correspondingly similar/distinct curvatures? This could provide valuable insight.

3.  Figure 5: The claim that the method achieves "improved feature separability" is hard to verify by visual inspection alone, especially for the "Order" and "Class" levels. Could it be supplemented with a quantitative clustering metric (e.g., Silhouette Score or Calinski-Harabasz Index) to more formally support this claim?

4. Figure 6: These attention maps are very compelling. However, to demonstrate that the method is responsible for this hierarchical attention, it is essential to show the corresponding attention maps from the baseline. Does the regularization and alignment strategy actually cause the model to shift its attention in this structured way compared to the baseline?

5. The abstract mentions that existing methods represent each image with a "single feature." Models like ViT produce patch-level features. What is the final, single-vector representation (e.g., the [CLS] token) used for alignment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the modality gap in hierarchical vision-language learning. The key idea is to construct symmetric tree-like features for both images and texts, and then align them on hyperbolic manifolds. Specifically, the method extracts hierarchical visual features from intermediate transformer layers using a text-guided attention module. To handle geometric differences between modalities, it embeds them into separate hyperbolic spaces with learnable curvatures and aligns them through an optimally constructed intermediate manifold. The approach demonstrates strong performance on taxonomic open-set classification across several datasets.

### Strengths
- The paper addresses an important and interesting problem of achieving efficient and symmetric modality alignment for hierarchical semantic structures.
- The work presents a novel approach by leveraging hyperbolic spaces with learnable curvatures and an intermediate manifold for modality alignment between images and text.
- Comprehensive experiments demonstrate the method's effectiveness, showing consistent improvements across multiple datasets and settings for taxonomic classification.

### Weaknesses
- The design of the semantic-aware feature extraction module appears somewhat heuristic and lacks strong theoretical justification. The critical choices of which specific intermediate transformer layers to use and the decision to disable cross-token attention are based on empirical observations rather than derived from first principles. This raises concerns about the generalizability and robustness of this core component, as its effectiveness might be sensitive to the specific architecture (e.g., different ViT depths) or pre-training dataset. Furthermore, the utilization of intermediate features is a well-established technique, diminishing the perceived novelty of this part of the proposed framework.
- Concern about sensitivity to hyperparameters. The proposed method introduces a non-trivial number of hyperparameters, and its performance is likely sensitive to their careful tuning. Key hyperparameters include the choice of intermediate transformer layers, the initial values for the learnable curvatures, and the loss weighting factor α. The paper would be strengthened by a sensitivity analysis demonstrating the robustness of the results to variations in these critical settings. Without such an analysis, it is difficult to assess the true practicality and generalizability of the approach, as the reported strong performance might be contingent on a specific, finely-tuned configuration.

### Questions
Please kindly refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a key limitation in current vision-language models (VLMs): the asymmetric and potentially suboptimal alignment of modalities, where images are represented by single features while text uses multi-level features. The proposed method, Alignment across Trees, constructs hierarchical (tree-structured) features for both modalities and embeds them in separate hyperbolic manifolds with learnable curvatures. It further introduces a semantic-aware visual feature extraction framework based on cross-attention guided by textual prompts, and a heterogeneous manifold alignment technique that uses KL divergence between distributions on different manifolds via an intermediate curvature. Experiments on multiple taxonomic open-set classification benchmarks show that this approach outperforms existing methods in aligning hierarchical multimodal data.

### Strengths
S1. The paper identifies and tackles a issue in VLMs—the mismatch in representational hierarchy between vision and language. Figure 1 clearly illustrates this contrast, highlighting how the proposed tree-based structure achieves more symmetric alignment.

S2.This paper has comprehensive ablation and visualization.

### Weaknesses
W1. The Taylor approximation used to compute the KL-based manifold distance (Appendix A) lacks empirical evaluation. No sensitivity analysis is provided for the approximation constant 𝑟, raising concerns about robustness and stability.

W2. The paper does not analyze potential failure modes, computational overhead from learning multiple curvatures.

W3. The approach to class imbalance across different tree levels is not explained, leaving open questions about potential bias in curvature learning or alignment.

### Questions
Could visualizations be supported by metrics like inter/intra-class variance or cluster separation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a hyperbolic representation approach for vision–language alignment in a hierarchical manner. 

The approach assumes we have textual labels at multiple levels (e.g. not just the original animal species but also the biological family, order, and class). It hypothesizes that different layers of a deep model similarly discern representations at different levels of granularity that can be aligned to the different textual hierarchy levels.

Based on this idea, the paper investigates hyberbolic representations. First, hierarchical representations are induced for the two modalities in a deep model by eliminating cross-token self-attention, adding cross-modality attention. The two modality-specific manifolds are bridged by constructing an intermediate hyperbolic space, to which the original ones are mapped subject to hierarchical entailment constraints.

### Strengths
- Interesting idea worth exploring
- Some promising experimental results
- Generalization to novel classes investigated

### Weaknesses
- Most of the examples focus on hierarchical relationships from biological taxonomy. While there are experiments on more diverse datasets such as ImageNet, the paper does not provide much analysis and insight into how the model behaves on other kinds of hierarchies, especially when the data is more diverse.

- On the Rare Species Dataset, the authors do not stick with the original labels but instead use hierarchical label names created by concatenating the various hierarchy levels in Algorithm 1 (Hierarchical Tree Construction). This seems to massively alter the nature of the task, as now the parent hierarchy is already given in each textual label. If the motivation for this step, as claimed by the authors, is just to deal with ambiguous labels, then why not add a small extra term just to ambiguous labels?

- The paper currently lacks a larger discussion of methods for hierarchy/taxonomy alignment. The paper presents a single method but doesn't explain the design space and design choices well enough. What alternatives could have been chosen? The relationship to the broader field of hierarchical/taxonomy alignment could be explored further.

- The alignment occurs in multiple phases rather than a single one, in particular it occurs via the Heterogeneous Manifold Alignment Algorithm but also earlier via cross-attention in the Semantic-Aware Visual Feature Extraction Framework. This cross-attention between image and text in the Semantic-Aware Visual Feature Extraction Framework (Eq. 5) appears to allow information to be copied over between modalities. It is not obvious how the approach ensures that each modality indeed yields a hierarchy for that particular modality rather than just copying over information from the other modality. How well does the model work without this cross-attention?

### Questions
- Understanding how the model avoids degenerate solutions is important. Can you explain to readers of the paper which parts of the model ensure that the fine-grained features do not become overly coarse?

- What do we observe on truly diverse classes, e.g. WordNet's "artifact" class?

Minor issues:

- The explanation of Eq. 3 mistakenly refers to $\delta$ and $u$ that do not appear in the formula.

- bad formatting in L. 197 "fine-grained information(Chen et al., 2024)."

- typo "can not" 

- The paper suffers from extensive negative vspace hacking. There are various orphan headings such as "4.2 HETEROGENEOUS MANIFOLD ALIGNMENT ALGORITHM" and "4.2.2 INTER-MODAL GEOMETRIC ALIGNMENT MECHANISM". There are also many headings that have insufficient spacing, e.g. "4.2.1 INTERMEDIATE MANIFOLD CONSTRUCTION" and "4.4 OPTIMIZATION STRATEGY".

- There are a number of poorly typeset mathematical expressions, e.g. Eq. (49).

### Soundness
3

### Presentation
2

### Contribution
3
