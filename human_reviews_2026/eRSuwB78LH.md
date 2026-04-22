# CoCoDiff: Correspondence-Consistent Diffusion Model for Fine-grained Style Transfer

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Transferring visual style between images while preserving semantic correspondence between similar objects remains a central challenge in computer vision. While existing methods have made great strides, most of them operate at global level but overlook region-wise and even pixel-wise semantic correspondence. To address this, we propose **CoCoDiff**, a novel *training-free* and *low-cost* style transfer framework that leverages pretrained latent diffusion models to achieve fine-grained, semantically consistent stylization. We identify that correspondence cues within generative diffusion models are under-explored and that content consistency across semantically matched regions is often neglected. CoCoDiff introduces a pixel-wise semantic correspondence module that mines intermediate diffusion features to construct a dense alignment map between content and style images. Furthermore, a cycle-consistency module then enforces structural and perceptual alignment across iterations, yielding object and region level stylization that preserves geometry and detail. Despite requiring no additional training or supervision, CoCoDiff delivers state-of-the-art visual quality and strong quantitative results, outperforming methods that rely on extra training or annotations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposed a training-free diffusion-based framework for fine-grained, correspondence-consistent style transfer. It leverages pretrained diffusion models to extract pixel-level semantic correspondences between content and style images, enabling region- and object-aware stylization.

The main contribution includes 1. Pixel-Wise Semantic Correspondence Module to extract intermediate diffusion features to build dense, semantically meaningful alignment maps between content and style images; and 2. Cycle-Consistency Optimization which 
integrates attention-guided feature injection with iterative refinement to enhance structural stability and stylization coherence.

### Strengths
1. This work has enough metrics to prove that the solution is effective, across FID, LPIPS, ArtFID, and CFSD. The ablation and user studies further validate design choices such as the feature injection weight and AdaIN harmonization.

2. This paper is well written and easy to follow.

### Weaknesses
1. This work claimed training free but it still need iteration during sampling. the complexity is not stated in the work
2. Some innovations were introduced rather suddenly, lacking sufficient theoretical foundation and clear explanations.
3. Why sobel and g works for style and context similarity. I look forward to a more reasonable explanation
4. This work lacks many recent references. I hope the author can include it as a baseline for comparison. e.g. [1] [2]


[1] Ahn, Namhyuk, et al. "Dreamstyler: Paint by style inversion with text-to-image diffusion models." aaai24

[2] He, Huiang, et al. "Semantix: An Energy Guided Sampler for Semantic Style Transfer." iclr25

### Questions
1. I hope the author can provide the specific time and memory consumption of the method.
2. The work lacks ablation experiments. I hope the authors can demonstrate the role of each module they proposed. And provide more credible theoretical explanations for sobel and g network.
3. I hope the author can add some newer baselines. Or provide some objective theoretical analysis to explain the similarities and differences.

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
This paper focuses on enhancing the performance of artistic style transfer between images with similar semantics. A key limitation of most existing works is that they operate at the global level, while overlooking region-wise and even pixel-wise semantic correspondence. To address this gap, the authors propose CoCoDiff, a training-free diffusion-based framework. This framework extracts intermediate diffusion features to establish pixel-wise correspondences and leverages cyclic optimization techniques, achieving fine-grained stylization with semantic consistency.

### Strengths
+ The integration of semantic consistency into diffusion-based style transfer is interesting and beneficial for stylization tasks between images with similar semantics.

+ The proposed method is training-free.

### Weaknesses
- My primary concern is that this paper appears to overlook a crucial research direction in style transfer, namely patch-based style transfer. Aligned with the motivation of this paper, works in this direction primarily leverage semantic correspondence between features to perform style transfer between objects with similar semantics. Representative methods include CNNMRF [A], Style-Swap [B], DIA [C], DivSwapper [D], Avatar-Net [E], and SCSA [F]. This paper neither discusses these methods in the related work section nor compares with them in experiments. Consequently, it is hard to effectively evaluate the technical innovation and performance of this work.

[A] Chuan Li and Michael Wand. Combining markov random fields and convolutional neural networks for image synthesis. In CVPR, pages 2479–2486, 2016.

[B] Tian Qi Chen and Mark Schmidt. Fast patch-based style transfer of arbitrary style. arXiv preprint
arXiv:1612.04337, 2016.

[C] Jing Liao, Yuan Yao, Lu Yuan, Gang Hua, and Sing Bing Kang. Visual attribute transfer through deep image analogy. TOG, 2017.

[D] Zhizhong Wang, Lei Zhao, Haibo Chen, Zhiwen Zuo, Ailin Li, Wei Xing, and Dongming Lu. Divswapper: Towards diversified patch-based arbitrary style transfer. In IJCAI, pages 4980–4987, 2022.

[E] Lu Sheng, Ziyi Lin, Jing Shao, and Xiaogang Wang. Avatarnet: Multi-scale zero-shot style transfer by feature decoration. In CVPR, pages 8242–8250, 2018.

[F] Chunnan Shang, Zhizhong Wang, Hongwei Wang, Xiangming Meng. SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer. In CVPR, pages 13051–13060, 2025.

- In L247–248, the authors state: “The optimal pair $(t^∗, l^∗)$ is selected by maximizing a correspondence quality metric $\mathcal{M}(t, l)$ over predefined candidate sets $\mathcal{T}$ and $\mathcal{L}$” How are $\mathcal{T}$ and $\mathcal{L}$ determined? Are they set based on empirical values?

- It seems that the reason that the proposed method can perform style transfer in a training-free manner mainly relies on the adjustment of attention weights in Eq. (7). Why can this adjustment effectively enhance the semantic consistency of the feature maps? Some necessary theoretical explanations are lacking.

- In L314–316, it is mentioned that “the fitting cycle's iteration process terminates when both ... are predefined thresholds.” It is unclear what kind of constraint a content loss ($\mathcal{L}_{content}$) greater than a threshold ($\tau_c$) can impose. What is the difference between this and constraining only the style loss?

- Section 4.6 (User Study) lacks necessary details. For example: what were the scoring instructions provided to users? What were the scoring rules? Which specific aspects were evaluated under the “Style” and “Content” dimensions in Table 3?

- Some minor detail issues: In L144–145, ArtFlow does not belong to the diffusion-based methods. In Eq. (6), $F_s$ is not defined.

### Questions
Please see weaknesses above.

### Soundness
2

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
This paper  proposes a training-free, diffusion-based framework for fine-grained, structure-preserving style transfer that operates directly on pretrained backbones without additional supervision or fine-tuning. The propose method delivers state-of-the-art visual quality and strong quantitative results, outperforming methods that rely on extra training or annotations.

### Strengths
(1) The proposed training-free style transfer method, which ensures style consistency, represents a valuable contribution to the field.
(2) A clear algorithm is presented for better understanding.
(3) CoCoDiff outperforms the six representative methods to some extend.

### Weaknesses
(1) Line 252 says M(t, l) that evaluates the alignment quality based on the extracted feature maps at timestep t and layer l. What exactly is M(t, l)?

(2) Fig. 2 requires significant refinement for improved clarity. FFM appears to have two inputs, but only I_sty_c is explicitly shown in the main framework. It seems that I_sty is inputted and reconstructed within the U-Net. If this is used for self-attention extraction, I suggest that the output should also be included. If this is the case, the self-attention section should be moved from Section 3.1 to Section 3.2 for better understanding. Besides, FIC also has two inputs where I can see only one.

The structure of Fig.2 should be better organized. The input images are put in the center, which is not easy to understand

(3) The used SDv1.4 is a very old version. Do the authors try newer diffusion models like SD2.1, SDXL or FLUX.

(4) The inference time should be listed as it consists of many complex steps like feature exchange, fitting cycle, iterative control.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

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
This paper proposes CoCoDiff, a training-free fine-grained style transfer framework based on pretrained latent diffusion models. The method establishes dense pixel-level semantic correspondences by mining intermediate diffusion features and introduces a cycle-consistency module to enforce structural and perceptual alignment. The approach achieves good visual quality and quantitative results.

### Strengths
1. The method demonstrates certain innovation by mining intermediate diffusion features to construct dense pixel-level semantic correspondences between content and style images, and introducing a cycle-consistency module to enhance structural and perceptual alignment.

2. The paper achieves superior quantitative and qualitative results compared to existing methods across multiple datasets.

### Weaknesses
1. Why does cycle consistency improve semantic correspondence? The paper lacks theoretical analysis or deeper explanation.

2. The correspondence quality metric M(t,l) in Equation (5) is not clearly defined. How is "correspondence quality" quantified?

3. Inconsistent notation usage: symbols such as p_c, p_s, p*_s are defined inconsistently across different sections.

4. Figure 8 has low comparison quality with text labels that are too small.

4. Insufficient experiments. Table 2 shows that on Mip-NeRF 360, GENIE's PSNR is 5-7 dB lower than Mip-NeRF, with even larger gaps in SSIM and LPIPS metrics. The authors attribute this to Gaussian volume density, but lack systematic analysis. The authors need to strengthen the persuasiveness.

5. Missing training time comparisons.

### Questions
1. The grid search for finding optimal (t*, l*) lacks theoretical guidance.

2. Why does "first converting the style image to content style" improve matching? Please provide theoretical analysis or more detailed mechanism explanation.

3. The authors need to provide sufficient validation on different diffusion models.

4. The authors need to report runtime and computational resource consumption.

5. How are the candidate sets T and L for grid search selected?

### Soundness
3

### Presentation
2

### Contribution
2
