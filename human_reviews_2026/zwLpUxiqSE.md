# Space Filling Curves as Spatial Priors for Small or Data-Scarce Vision Transformers

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Vision Transformers (ViTs) have become a dominant backbone in computer vision, yet their attention mechanism lacks inherent spatial inductive biases, which are especially crucial in small models and low-data regimes. Inspired by the masking in Linear Transformers and the scanning patterns of Vision SSMs, we propose VIOLIN, a lightweight masked attention mechanism that integrates Space Filling Curves (SFCs) to enhance spatial awareness with negligible computational overhead. VIOLIN scans the input image with multiple SFCs to build curve specific decay masks, which are averaged and multiplied with the attention matrix to encode spatial relationships. It yields notable gains in data-scarce settings: when fine-tuning on VTAB-1K, VIOLIN improves accuracy by up to 8.7% on the Structured group, and it can be combined with parameter-efficient tuning methods such as LoRA. Beyond fine-tuning, VIOLIN consistently improves various tiny or small scale ViT architectures (e.g., DeiT, DINO) during pretraining on ImageNet-1K, achieving gains of up to 0.9\% on  on ImageNet-1K and 7.2\% on pixel level CIFAR-100. Overall, VIOLIN offers a computationally efficient yet effective way to inject spatial inductive bias into ViTs, particularly benefiting small models and data-scarce scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes **VIOLIN**, a simple and plug-and-play spatial prior module for Vision Transformers (ViTs).  
The method introduces *Space Filling Curves (SFCs)* (e.g., Snake, Zig-zag, Peano, Hilbert) to define alternative scanning orders of image patches.  
For each curve \(c\), a decaying mask \(M_c[i,j] = \gamma_c^{|i-j|}\) is constructed to encourage locality in attention.  
After aligning these masks back to the standard patch order and averaging, the resulting mask \(M_{\text{VIOLIN}}\) is multiplied with the attention score matrix before softmax.

The approach is extremely lightweight (+0.0002% params, +0.64% FLOPs) and can be applied to pretrained or finetuned ViTs without architectural changes.  
Extensive experiments on **VTAB-1K**, **ImageNet-1K**, **DINO**, **pixel-level CIFAR-100**, and dense tasks (ADE20K / COCO) show consistent gains, especially on “Structured” VTAB tasks (+8.7%).

### Strengths
- **Well-defined target problem:** Focuses on *small models and data-scarce regimes* where ViTs lack spatial inductive bias — a meaningful and under-explored setting.  
- **Simplicity and generality:** VIOLIN requires no retraining or re-architecture changes, making it truly plug-and-play.  
- **Elegant formulation:** The SFC-based decaying masks are clearly derived; the permutation and averaging operations are well explained.  
- **Strong empirical results:** Significant improvement on VTAB-1K (Structured group +8.7%) and pixel-level CIFAR-100 (+7.2%) convincingly show the benefit of spatial priors.  
- **Low computational cost:** The added overhead is negligible, suitable for real-world low-resource finetuning.  
- **Broad applicability:** Small but consistent gains on segmentation and detection tasks further validate its generality.

### Weaknesses
1. **Novelty is limited.**  
   The core idea—distance-decayed attention weights—is reminiscent of *linear attention*, *RMT*, and *RetNet*–style exponential decay mechanisms.  
   The use of multiple SFCs and their averaged mask is an incremental extension rather than a fundamentally new concept.

2. **Missing comparisons with strong baselines.**  
   The paper compares mainly to vanilla DeiT/DeiT-III/DINO backbones.  
   It lacks direct comparisons with existing locality-enforcing methods, such as:
   - Relative positional bias (Swin / ViT-RPB),
   - Convolutional stems or LocalViT,
   - Manhattan-distance masks (RMT),
   - Single-curve or random-curve baselines.  
   Without these, it is unclear whether the large Structured-task gains stem from the proposed multi-SFC averaging or from any reasonable local bias.

3. **Training details and fairness are under-specified.**  
   VTAB-1K finetuning recipes (learning rate, γ initialization, α sharing) are buried in the appendix.  
   It remains unclear whether baselines were tuned equivalently.  
   The surprising claim that *untrained masks outperform pretrained ones* needs stronger justification.

4. **Questionable mask effectiveness.**  
   Figure 7 shows most γ₍c₎ values approach 1, suggesting the mask becomes nearly uniform.  
   If so, why does the Structured group improve so dramatically?  
   More analysis of per-head γ values and locality visualization is needed.

5. **Computational overhead claim is not empirically verified.**  
   Only theoretical FLOPs/parameter ratios are reported.  
   Actual GPU memory and runtime increase (especially on dense tasks) should be measured.

6. **Overstated framing.**  
   The paper sometimes overclaims by calling VIOLIN a *principled spatial prior via SFCs*.  
   In fact, the method does not exploit the geometric guarantees of SFCs; it only uses index distance \(|i-j|\) with exponential decay.  
   Theoretical justification for averaging multiple SFC-induced metrics is weak.

### Questions
Please refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes VIOLIN, a masked attention mechanism for Vision Transformers (ViTs) that incorporates Space Filling Curves (SFCs) to improve spatial inductive biases. Standard ViTs suffer from the lack of spatial awareness due to the permutation-equivalent nature of self-attention. Inspired by linear attention and SSMs, VIOLIN constructs curve-specific decay masks that model the relative spatial distance between image patches. These masks are averaged and applied to the attention matrix, introducing spatial priors without modifying the core ViT architecture.

### Strengths
- Extensive empirical validation: tested on diverse model scales (5M–86M parameters) and training setups (supervised and self-supervised).

- The paper is easy to follow.

- The proposed method shows some improvment.

### Weaknesses
- **Limited Contribution from the Core Method**: [6] has shown that average pooling can boost the DeiT's performance. Tab. 14 suggests that **the performance gain mainly comes from the average pooling**. The VIOLIN only provide marginal improvement for small models, and **even harms the performance of the large model ViT-B**.

- **Limited Generalization**: Based on the the results in Tab. 8, **the improvements on Swin-T and Swin-S are below 0.2%**, which is likely within run-to-run variance and not statistically significant. This suggests that the proposed method  is rather an engineering optimization technique, which does not generalize well to different models.

- **Limited Comparison**. The baselines used for comparison primarily rely on absolute positional embeddings, which are known to be suboptimal. Relative positional encodings, widely adopted in modern architectures [1-5], are simpler, more flexible, and have been shown to outperform absolute encodings in multiple settings. **It is not clear that space-filling curves offer any meaningful advantage over such approaches**. Without direct comparisons to relative positional encoding, the benefits of VIOLIN are difficult to justify.

[1] Wu, Kan, et al. "Rethinking and improving relative position encoding for vision transformer." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[2] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In ACL, 2019. 1, 3, 7, 8

[3] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[4] Liu, Ze, et al. "Swin transformer v2: Scaling up capacity and resolution." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[5] Zhou, Yuxuan, et al. "SP-ViT: Learning 2D Spatial Priors for Vision Transformers." 33rd British Machine Vision Conference. BMVA Press, 2022.

[6] Conditional Positional Encodings for Vision Transformers, ICLR2023.

### Questions
Could the authors also compare VIOLIN to other methods related to spatial prior, such as relative positional encoding/bias in [1-5] ?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the use of space filling curves as a way to introduce spatial priors to vision transformers. It extends upon the use of decay masks with image flattening as determined by different space filling curves. The use of different curves effectively reorders the patches of the image in different spatially meaningful ways as compared to a single zig-zag line scan used in transformer architectures. The proposed method improves upon previous data efficient methods under similar settings and can also be applied solely in the fine-tuning stage.

### Strengths
The authors proposed a novel way to include hand designed spatial priors thru the use of SFCs and proposed an efficient and effective way to incorporate into ViT architectures. Their proposed method can also be included into pretrained models with fine-tuning only. The proposed method improves on previous data-efficient methods like DeiT. Well designed ablation studies were also included to show the effects of each of their proposed changes to the attention mechanism. The authors also include a rather commendable and substantial appendix with important key prior art.

### Weaknesses
- Training flow is not immediately clear in the main paper. Since there are multiple stages to train a ViT with VIOLIN masks, it would be good to recap on the stages even though DeiT’s training recipe was followed. This would make the experiment section and the ablation studies clearer.  
- The authors proposed the use of different hand selected SFCs, it would be interesting to see how a separately learned patch ordering, e.g. from Kutscher 2025, compares. After all, the mask decay method can take in any form of ordering.  
- Minor issue: Typo in Figure 2\. In the center block, VIOLIN is misspelled.

### Questions
- DieT uses CNN as a teacher network. Is a CNN also used in this case?  
- Since CNNs have the strongest spatial prior, could the authors also include a similarly size SOTA CNN? Especially if, similar to DieT training recipe, a CNN is used as a teacher network

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The manuscript proposes a lightweight masked attention mechanism named VIOLIN that integrates Space Filling Curves (SFCs) to enhance spatial awareness in smaller visual transformers (ViT). By better filling the space in 2D images through specifically designed curves, a better neighborhood representation is achieved when applying ViTs. VIOLIN scans the input image with multiple SFCs to build curve specific decay masks which are averaged and then weighted with the attention matrix to encode spatial relationships.

As SFCs the authors use Snake, Zig-zag, Peano, and Hilbert curves together with their transposed variants to capture diverse scanning patterns in both row and column major order.

### Strengths
- The author propose an approach to represent better the neighbourhoods through Space filling curves (SFC) in order to enhance the processing of the image with ViT networks.
- The manuscript concludes that by using SFCs improves the performance in performance in small models and limited-data settings.
- Extensive experimental results are provided.
- The approach requires only limited extra computational demands
- Extending the application of SFCs to video understanding is also assessed.

### Weaknesses
- There is no systematic or any theoretical study about what the space filling curves are useful for in ViTs
- It is not clear what applications can be used for such SPCs based representations in ViTs except for some particular filtering. In the manuscript it is indicated that it can be applied for classification, semantic segmentation or object detection.
- It is not clear how such SPCs can be used to some other ViT models.

### Questions
Could the multiple SFC scans be combined in a more efficient way than by simply averaging?

How would the proposed multiple SFC work in the case of other transformer networks than those tested in the manuscript? 
For example how would they work for the Swin transformer proposed in the paper:
Z. Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021.

How it would work, when applied on videos, on some video transformers, like for:
Limin, W. el al, VideoAME V2: Scaling}video masked autoencoders with dual masking, CVPR 2023.

### Soundness
3

### Presentation
3

### Contribution
2
