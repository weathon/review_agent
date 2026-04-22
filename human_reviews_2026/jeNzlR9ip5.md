# Retina Vision Transformer (RetinaViT): Measuring the Importance of Spatial Frequencies in Vision Transformers

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Humans perceive the visual scene based on low spatial frequency components, before high spatial frequency components refine the perception through another neural pathway.
  Drawing on this neuroscientific inspiration, we introduce patches from different spatial frequencies into Vision Transformers (ViTs), and investigate how attention is assigned to them.
  We name this model Retina Vision Transformer (RetinaViT) due to its inspiration from the human visual system, where different types of photoreceptors with different receptive field sizes form the bases for neural pathways.
  Our experiments on benchmark data show that RetinaViT exhibits a strong tendency to attend to low spatial frequency components in the early layers, and shifts its attention to high spatial frequency components as the network goes deeper.
  This tendency emerged by itself without any additional inductive bias, and aligns with the visual processing order of the human visual system.
  We hypothesize that RetinaViT captures structural features, or the gist of the scene, in earlier layers, before attending to fine details in subsequent layers, which is the reverse of the processing order of mainstream backbone vision models, such as CNNs and other ViT family models.
  We also observe that RetinaViT is more robust to image corruptions, and significant reductions in model size, compared to the original ViT.
  We hypothesize that its higher information density led to this result.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work is focused on the design of Vision Transformers (ViTs) and making them able to work with patches of different spatial frequencies. The approach is motivated from the perspective of neuroscience, where the human visual system processes low spatial frequencies initially then followed by high spatial frequencies. A ViT that can handle input patches of different sizes is proposed, entitled RetinaViT, which is evaluated by investigating the attention in early layers and the removal of layers in the ViT.

### Strengths
1. A clear and understandable idea.
2. A well structured manuscript.
3. Nice figures illustrating the different patching strategies.

### Weaknesses
1. The novelty is low. As the paper itself discusses in the related work, numerous works have investigated processing patches at different scales [1, 2, 3, 4]. While there seems to be minor difference between how the process is implemented in practice, the paper does not explain how the RetinaViT differs from existing approaches and also why this particular approach would be beneficial compared to already established technique.

2. The experimental evaluation is limited. Two experiments are conducted, one where the magnitude of the attention weights is calculated and one where layers are iteratively removed and performance is evaluated. These experiments do not give insights into the benefits of RetinaViT. No baselines are included, so there is no way to compare with existing methods. For Figure 6 to meaningful, a standard ViT should also be presented so a comparison can be made. For Table 1, RetinaViT seems to have very little influence on the performance of the model. Compare to previous work along the same lines [1, 2], where numerous datasets, baselines, and tasks are considered, the analysis in this paper is to limited to provide information about the usefulness of RetinaViT.

3. The motivation is unclear. The argument seems to be that "Humans start to process low spatial frequency components earlier (30 ms after exposure to a visual scene) than high spatial frequency components (150 ms after)", and that CNNs and ViTs do not. However, looking to the literature, it appear to me that many works point towards CNNs [5] and ViTs [6] processing low frequency simple features first before combining them into more complex high frequency feature later. It is therefore unclear to what extend the motivation of the paper is sound.

- [1] Tian et al., ResFormer: Scaling ViTs with Multi-Resolution Training, CVPR 2023
- [2] Beyer et al., FlexiViT: One Model for All Patch Sizes, CVPR 2023
- [3] Fan et al., Multiscale Vision Transformers, ICCV 2021
- [4] Liu et al., MSPE: Multi-Scale Patch Embedding Prompts Vision Transformers to Any Resolution, NeurIPS, 2024
- [5] Bau et al., Network Dissection:Quantifying Interpretability of Deep Visual Representations, CVPR 2017
- [6] Dorszewski et al., FROM COLORS TO CLASSES: EMERGENCE OF CONCEPTS IN VISION TRANSFORMERS, Explainable AI, 2025

### Questions
1. In what way does RetinaViT differ from existing works [1, 2, 3, 4]?
2. What would be the benefit of using RetinaViT over these works, and can you experimentally show these benefits?
3. Can it be shown quantitatively a clear difference in the way the ViT and the RetinaViT process information?
4. Can experimental evidence supporting the motivation be provided, and how does the motivation connect to related work [5,6]?

- [1] Tian et al., ResFormer: Scaling ViTs with Multi-Resolution Training, CVPR 2023
- [2] Beyer et al., FlexiViT: One Model for All Patch Sizes, CVPR 2023
- [3] Fan et al., Multiscale Vision Transformers, ICCV 2021
- [4] Liu et al., MSPE: Multi-Scale Patch Embedding Prompts Vision Transformers to Any Resolution, NeurIPS, 2024
- [5] Bau et al., Network Dissection:Quantifying Interpretability of Deep Visual Representations, CVPR 2017
- [6] Dorszewski et al., FROM COLORS TO CLASSES: EMERGENCE OF CONCEPTS IN VISION TRANSFORMERS, Explainable AI, 2025

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces RetinaViT, a simple ViT variant with two contributions: (1) using patches from a multi-scale image pyramid, and (2) a scaled average position embedding that encodes a patch's location as well as its scale.

### Strengths
- Clean, clear paper with an intuitive idea taken from the human visual system
- The novel position encoding strategy is a clever way to encode both 2d position and scale
- Interesting finding that the model learns a low-to-high frequency processing order; this is genuinely notable

### Weaknesses
- Extremely weak empirical results; the entire performance claim seems to rely on one tiny table, which shows a marginal delta over the plain ViT
- Doesn't compare to related multi-scale ViT variants, little explanation as for why this approach is better
- Paper is mostly speculative discussion about inductive biases, parse trees, scale invariance, etc. -- most of these claims being unsubstantiated by experiments

### Questions
Are there any additional pieces of empirical evidence that the RetinaViT strategy has superior performance, controlling for the number of patches processed? For example, you could show that RetinaViT is actually more scale-invariant by testing on a dataset of "corrupted" images where the scale is changed (something akin to ImageNet-C). This could even be training-free (aside from the models you've already trained) -- if you could find a set of transforms that this model is more robust to I would find the paper a lot more compelling. 

Due to the lack of experiments, I find this to be more of a workshop contribution.

### Soundness
1

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
5

### Summary
The study describes a way to integrate information from different frequency bands within Vision Transformers (RetinaViT) without increasing the model’s parameter count. The key idea is to construct an image pyramid by progressively downsampling the input image to multiple resolutions. From each resolution, image patches are extracted and concatenated into a single token sequence, which is then processed by a standard ViT-S/16 backbone. This design allows each transformer block to jointly attend to tokens representing both coarse, low-frequency structures and fine, high-frequency details of the same image. The motivation of this work is human behavior which processes the image gist before the finer details.

Empirically, RetinaViT delivers modest accuracy gains on ImageNet-scale benchmarks while keeping the total parameter count unchanged. Since the core transformer architecture remains intact and no new learnable components are introduced, the approach is both lightweight and easily integrable with existing ViT variants. While this offers an interesting perspective on how hierarchical frequency integration might emerge in transformer-based vision models, the results overall feel preliminary and premature. The neuroscientific framing appears more like an afterthought than a guiding principle, and the current implementation of “RetinaViT” bears little resemblance to the actual retina or early visual processing mechanisms it seeks to emulate.

### Strengths
1. The paper introduces an interesting perspective on integrating information from multiple spatial frequency bands within a ViT framework. This idea of fusing low- and high-frequency representations at the token level is conceptually interesting and expands the discussion around how transformers might capture hierarchical visual structure.

2. The idea is conceptually simple and easy to follow.

### Weaknesses
1. The biological analogy: The paper leans on the claim that humans process low SF ~30 ms and high SF ~150 ms earlier and maps this onto ViT depth but in humans it is in fact in behavior.
2. Extending this further: The human retina performs spatially nonuniform sampling (not concatenation over multi-frequency bands). Naming the model RetinaViT is highly misleading because the proposed mechanism is nothing like the human retina. 
3. Also see neuroscience evidence (e.g., Issa et al. 2000; Nauhaus et al. 2012 - figure 2c and d) shows that early visual cortex (V1) already represents both high and low spatial frequencies. This contradicts the paper’s finding that early ViT modules prioritize lower frequencies. The authors should discuss this discrepancy and clarify how RetinaViT’s processing hierarchy aligns—or intentionally diverges—from biological evidence.
4. The paper does not have direct comparisons with standard ViTs trained under identical conditions, nor with other established multi-scale transformer architectures (e.g., Swin, PVT, CrossViT). Without these baselines, it’s hard to determine whether the reported gains stem from the proposed frequency-integration mechanism or simply from generic multi-scale processing effects already explored in prior work.
5. The concatenation of patches across frequency bands likely increases the embedding dimensionality. As wider ViTs are known to perform better (see Saratchandran et al. 2025), it is important to decouple performance gains due to increased width from those due to the proposed frequency integration. Could the authors constrain the embedding dimension to remain constant (e.g., via projection) to control for this effect?
6. In Section 4.1 (line 275), the authors mention analyzing “all examples in the dataset.” This should ideally be restricted to the validation set only, to avoid inadvertently analyzing the model on its training data (i.e., double dipping).
7. The paper infers how the model processes frequency components based on attention-weight patterns. However, prior work (see Darcet et al. 2023, figure 2) shows that ViTs tend tend to pay lots of attention to arbitrary image patches (attention “artifacts”). Thus to control for this phenomena, the authors could train their model with register tokens (see https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_with_register_tokens.py)
8. Figure 6 is very hard to follow and can be improved by averaging the activation values  (with errorbars) corresponding to each frequency band within each layer. 
9. For Table 1, where performance differences are small, it is important to report accuracy variance across multiple random seeds to establish statistical significance.
10. Is this architecture more adversarially robust? (because of the partial dependence on lower frequency components)
11. How does the model perform on other perceptual benchmarks that depend on spatial frequency?: eg. https://openreview.net/pdf?id=KvPwXVcslY

### Questions
Q1. How does the claimed mapping between human behavioral temporal processing meaningfully map to RetinaViT?

Q2. Relatedly, how do the authors justify the name RetinaViT (given the number of differences with retinas)?

Q3.  Why are there no direct comparisons with standard ViTs or existing multi-scale transformer baselines (e.g., Swin, PVT, CrossViT) trained under identical conditions to isolate the benefit of the proposed frequency integration?

Q4. Given known artifacts in ViT attention maps (Darcet et al., 2023), how do the authors ensure that their attention-based frequency interpretations are reliable?

Q5. Have the authors evaluated statistical significance by reporting variance across multiple random seeds, especially where accuracy differences are small?

Q6. Does RetinaViT exhibit improved robustness to adversarial or high-frequency perturbations, and how does it perform on spatial frequency–sensitive perceptual benchmarks (Subramanian et al)?

### Soundness
2

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
4

### Summary
This paper introduces RetinaViT, a modified Vision Transformer designed to examine whether the low-to-high spatial frequency processing order observed in human vision can naturally emerge within transformer-based visual models. The proposed method extends the standard ViT-S/16 architecture by incorporating a multi-resolution image pyramid input, where patches from multiple downsampled versions of the same image are concatenated into a single token sequence. Additionally, it proposes a Scaled Average Positional Embedding (SAPE) that encodes both spatial location and receptive field scale by averaging and scaling the original ViT positional embeddings. Using ImageNet-1K, the authors probe layer-wise attention weights, scores, and residual magnitudes to observe a consistent shift from low-resolution to high-resolution focus across layers. Experimental results further show that RetinaViT maintains slightly better accuracy under severe depth reduction, suggesting improved robustness in shallow configurations.

### Strengths
- The paper presents a conceptually original perspective by framing transformer attention through the lens of biologically inspired frequency processing, offering an interesting bridge between computational vision and perceptual neuroscience.
- The proposed design is simple yet interpretable, preserving the original ViT architecture while clearly isolating the effects of multi-scale input on attention behavior.
- The method provides a technically neat way to integrate multi-scale image information into a ViT without architectural modification, and its positional embedding formulation that encodes scale information could serve as a useful reference for future multi-scale transformer research.

### Weaknesses
- Architectural novelty is limited. RetinaViT remains functionally identical to ViT aside from multi-scale patch concatenation and a heuristic positional embedding scaling. Unlike prior multi-scale transformers (e.g., PVT, CrossViT, ResFormer), it does not incorporate hierarchical or cross-scale attention mechanisms within the transformer blocks, making the architectural contribution relatively shallow.

- Empirical validation is limited. Experiments are confined to ImageNet-1K classification, despite the method’s stated goal of improving multi-scale understanding. Since such cues are most critical in dense prediction tasks (e.g., detection, segmentation), the absence of these evaluations limits the generalizability and relevance of the claims.

- Marginal Scalability of performance gains. While the authors suggest that RetinaViT enhances robustness in shallow configurations by better capturing low-frequency or high-level structural information, the quantitative results in Table 1 indicate that meaningful gains occur only at very low depths (+0.6–0.8 pp for 2–4 layers) and become marginal (+0.1–0.2 pp) for standard or deeper configurations (10–12 layers). Because the representational power of Transformers largely arises from depth scaling, improvements that appear only in under-parameterized regimes limit the architectural significance and suggest that RetinaViT may not effectively enhance model capacity at scale. Moreover the discrepancy between the visually consistent coarse-to-fine trend across all 12 layers (Fig. 6) and the marginal accuracy gains at full depth (Table 1) further casts doubt on whether this hierarchical processing meaningfully persists or contributes to deeper representations.

Efficiency and complexity analysis are missing. The concatenated multi-resolution patches significantly increase the total token count. This high token count inherently increases the computational complexity of the self-attention mechanism, yet crucial metrics such as FLOPs, memory footprint, and latency relative to the baseline ViT are entirely absent. Without this analysis, it is impossible to assess the practical utility or efficiency trade-offs of the proposed model.

### Questions
- Did the authors compare attention maps when low- or high-frequency patches are selectively masked?
- Can the same phenomenon be observed on larger ViT backbones (e.g., ViT-B/L) or different datasets?

### Soundness
1

### Presentation
2

### Contribution
1
