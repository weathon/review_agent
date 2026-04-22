# Reference-Guided Identity Preserving Face Restoration

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 8

## Abstract
Preserving face identity is a critical yet persistent challenge in diffusion-based
image restoration. While reference faces offer a path forward, existing methods
typically suffer from partial reference information and inefficient identity losses.
This paper introduces a novel approach that directly solves both issues, involving
three key contributions: 1) Composite Context, a representation that fuses high- and
low-level facial information to provide comprehensive guidance than traditional
singular representations, 2) Hard Example Identity Loss, a novel loss function
that uses the reference face to address the identity learning inefficiencies of the
standard identity loss, 3) Training-free multi-reference inference, a new method
that leverages multiple references for restoration, despite being trained with only a
single reference. The proposed method demonstrably restores high-quality faces
and achieves state-of-the-art identity preserving restoration on benchmarks such as
FFHQ-Ref and CelebA-Ref-Test, consistently outperforming previous work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a reference-based diffusion framework for face restoration, emphasizing identity preservation. The authors introduce two key modules: (1) Composite Context, which combines multiple levels of facial features from pre-trained ArcFace and FaRL to provide high- and low-level representations of the reference face; and (2) Hard Example Identity Loss, which uses the reference face as a “hard example” to alleviate training inefficiency in identity learning. The method can also perform training-free multi-reference inference using classifier-free guidance. Experiments show that the proposed approach achieves higher identity similarity compared to prior works such as RefLDM and RestorerID, while maintaining competitive perceptual quality.

### Strengths
+ The paper identifies the common issue of insufficient identity preservation in diffusion-based face restoration and addresses it via composite context (face recognition and representation) and hard example identity loss.

+ The two modules are orthogonal, making the approach easily integrable with other LDM backbones.

+ The authors provide results against both reference-based (RefLDM, RestorerID) and no-reference methods (DiffBIR, CodeFormer), with great metrics and ablations.

### Weaknesses
- Limited novelty over existing IP-Adapter-like paradigms and other diffusion based face restoration models.
The proposed Composite Context essentially acts as a fixed feature adaptor combining ArcFace and FaRL representations. This is conceptually close to IP-Adapter–style feature injection, except with multiple frozen encoders. The claimed advantage of mixing high-level and general representations is not convincingly demonstrated—visual results do not show clear benefits from these two branches. A more direct comparison with a trainable face encoder (e.g., IP-Adapter with finetuning or other low-level feature extraction model) is necessary to justify the contribution.

- While “reference-guided face super-resolution” has some relevance, the paper focuses mainly on preserving high-level identity rather than fine-grained personal textures, which are typically more desirable for human perception. It is better to show some close-up regions to show the detailed textures. I note that sometimes these fine-grained personalized textures are not preserved well.

- Lack of real-world evaluation.
The entire study uses synthetically degraded training and text data. The absence of experiments on real low-quality or in-the-wild images weakens claims about robustness and applicability, as acknowledged in the limitations section.

- When both ​s_i and s_c are set to 1.2 in Eqs. (5)–(6), the middle term becomes redundant and has no practical meaning.

- The authors should also discuss whether diffusion-based approaches indeed offer substantial advantages over GAN-based ref face restoration in this specific reference-conditioned setting.

- Honestly, the improvement of some components like identity loss, FaRL, or ArcFace is not obvious.

### Questions
- How does the Composite Context differ in practice from IP-Adapter or other reference-conditioned feature injectors? Could you compare against a fine-tuned, learnable adapter to validate the necessity of using two frozen encoders?

- Since only synthetic degradations are used, how would the method behave on real-world low-quality faces (e.g., surveillance or historical photos)?

- Can the Hard Example Identity Loss be generalized to other perceptual constraints, or is its effect limited to face identity preservation?

- As for the VAE Encoder, some works claim that it is not suitable for low-quality image reconstruction, so they attempt to fine-tune the vae encoder with LORA. However, this work did not consider this.

### Soundness
2

### Presentation
3

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
This paper presents an interesting method for reference-based face restoration, focusing on improving identity preservation through two key contributions: a Composite Context representation and a Hard Example Identity Loss. The work is well-motivated, addressing clear limitations in existing methods (partial reference information, inefficient identity loss). The experimental results are comprehensive, demonstrating state-of-the-art performance on standard benchmarks.

### Strengths
Well-Motivated: The core ideas are reasonable in the context of reference-based face restoration. The critique of the "learning inefficiency" of standard identity loss is insightful, and the proposed HID loss is a simple yet effective solution. 

Empirical Evidence: The paper provides extensive quantitative evaluations on multiple benchmarks (FFHQ-Ref Moderate/Severe, CelebA-Ref-Test), consistently showing superior performance in identity preservation (IDS, FaceNet) while maintaining competitive image quality. The comparison with recent SOTA methods (RefLDM, RestorerID) is fair and convincing.

Comprehensive Ablation Studies: The hierarchical ablation (module-wise and component-wise) effectively validates the contribution of each proposed component.

### Weaknesses
Limited technical novelty:  While the paper proposes two modules—Composite Context (CC) and Hard Example Identity Loss (HID)—the technical innovations appear incremental. The CC module combines existing face representations (ArcFace and FaRL), which is conceptually similar to prior multi-modal fusion approaches (e.g., SDXL, PGDiff). Though the authors claim to be the first to combine specialized face encoders for restoration, this primarily constitutes an engineering integration rather than a fundamental algorithmic breakthrough. The HID loss, while effective, builds directly on standard metric learning techniques (e.g., hard example mining) without novel theoretical contributions. The training-free multi-reference inference is pragmatic but relies on straightforward ensemble averaging, lacking architectural or methodological novelty. Overall, the work would benefit from deeper ablation studies or theoretical analysis to justify its uniqueness beyond empirical improvements.

Insufficient quantitative and qualitative results:
1)	Limited Quantitative Superiority: The proposed method does not consistently outperform existing approaches. Key perceptual quality metrics like LPIPS and MUSIQ often fail to show a decisive advantage over competing methods.
2)	Deficient Qualitative Results: The visual evidence is unconvincing. As shown in Figures 3-4, the outputs frequently exhibit noticeable artifacts and a significant disparity in identity preservation compared to the high-quality ground truth. In some cases (e.g., Figure 3), the restored facial texture and structure are inferior even to CodeFormer, a non-reference-guided baseline.

Incomplete Comparative Analysis: The experimental comparisons are limited to only two reference-guided works, neglecting several recent and relevant state-of-the-art methods in the field, such as DMDNet, FaceMe, and Gen2Res. This omission makes it difficult to fairly assess the method's true standing.

### Questions
Please refer to the paper weakness

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
5

### Summary
This paper addresses identity preservation in reference-based face restoration by proposing two key innovations: (1) Composite Context, a multi-level representation that fuses high-level identity information and low-level facial details from reference faces, and (2) Hard Example Identity Loss, a novel loss function that uses reference faces as hard examples to improve identity learning efficiency. The method is trained with single references but can leverage multiple reference images during inference in a training-free manner. Experiments on FFHQ-Ref and CelebA-Ref-Test datasets demonstrate superior performance in identity preservation compared to existing methods.

### Strengths
1.Well-designed Composite Context that addresses multi-level information fusion. Unlike prior reference-based methods that only leverage partial information from reference faces, this work comprehensively combines: (a) high-level identity features via pre-trained ArcFace embeddings that enforce angular margin constraints; (b) general facial attributes via FaRL including skin texture, lighting, and semantic information; and (c) cross-attention projection through UNet for spatial alignment. The ablation study in Table 5 validates that all components contribute meaningfully, demonstrating the complementary nature of multi-level information.

2.Strong quantitative improvements in identity preservation metrics. The method achieves substantial and consistent gains in identity metrics. Notably, the method maintains these identity gains while achieving competitive LPIPS and sometimes better perceptual quality scores, suggesting the approach does not simply overfit to identity at the expense of visual quality.

### Weaknesses
1.Limited novelty in individual technical components 
While the overall system is effective, each core component builds heavily on existing techniques, the contribution feels more like good engineering than fundamental innovation.

2. Insufficient Analysis of Multi-Reference Degradation Phenomenon
Table 2 reveals a counterintuitive and concerning result: identity similarity IDS(REF) sometimes decreases with more reference faces. This directly contradicts the fundamental premise that more reference information should improve identity preservation. This weakness raises concerns about whether the multi-reference capability is truly beneficial or just a side effect of the architecture. The lack of analysis makes it difficult to recommend best practices for real-world deployment.

3.The comparative analysis is limited in scope, and notably lacks comparison with recent state-of-the-art methods from 2025.

### Questions
1.Counterintuitive multi-reference degradation phenomenon inadequately explained. Table 2 reveals a puzzling result: IDS(REF) sometimes decreases as more reference faces are added. This contradicts the intuition that more reference information should improve identity matching to references. The paper acknowledges this but provides no analysis of: (a) Why does this degradation occur? Is it due to conflicting information from different references, averaging artifacts, or limitations in the ensemble mechanism? (b) Is there an optimal number of references, or does it vary by degradation severity? (c) How should practitioners select which references to use when multiple are available?

2.Unclear notation and missing implementation details. Several technical details are insufficiently specified: (a) Equation (1) mentions positional encoding epsilon_position but never defines its form, dimensionality, or initialization. (b) Table 2 caption states IDS(REF) is calculated using 'the first available reference face' - why this arbitrary choice rather than averaging across all references or using the highest-quality one? These ambiguities would make reproduction challenging.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the task of reference-guided face restoration and proposes two key components: a Composite Context mechanism and a Hard Example Identity Loss. These components enhance the utilization of reference image information and improve identity consistency. Furthermore, the paper introduces a multi-reference inference framework that, despite being trained with only a single reference image, can effectively handle multiple references during inference.

Although the proposed approach shows limited novelty and shares similarities with prior works, the method is well designed and empirically validated through comprehensive experiments. Overall, the results demonstrate the effectiveness of the proposed framework, and I recommend acceptance.

### Strengths
1. The paper is clearly written and well organized.
2. The overall framework is well structured, it effectively incorporates reference facial representations into the restoration pipeline and extends RefLDM models to better preserve identity similarity.
3. The experimental evaluation is comprehensive and detailed, providing strong empirical evidence for the method’s effectiveness.

### Weaknesses
1. The overall novelty is limited. The proposed framework is conceptually similar to several existing works, and the Hard Example Identity Loss is closely related to the loss function commonly used in [1]. Its main difference lies in the additional use of reference image information, which resembles strategies adopted in recent personalized generation methods[2].
2. The Composite Context module relies heavily on face recognition features. It would be interesting to explore how the performance changes when different face recognition backbones are employed.

[1]: Refldm: A latent diffusion model for reference-based face image restoration.

[2]: PuLID: Pure and Lightning ID Customization via Contrastive Alignment

### Questions
See Weakness section above

### Soundness
4

### Presentation
4

### Contribution
3
