# UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Tokenizer is a crucial component for both visual understanding and generation. To advance toward the ultimate goal of universal modeling, recent research has focused on developing a unified tokenizer. However, existing tokenizers face a significant performance trade-off between understanding and generation, stemming from the inherent conflict between high-level semantic abstraction and low-level pixel reconstruction. To tackle this challenge, we propose a generic and unified tokenizer, namely $\textbf{UniFlow}$, by flexibly adapting any visual encoder with a concise reconstruction decoder. 
Specifically, we introduce $\textit{layer-wise adaptive self-distillation}$ applied to the well-pretrained visual encoders, which enables UniFlow to simultaneously inherit the strong semantic features for visual understanding and flexibly adapt to model fine-grained details for visual generation.
Moreover, we propose a lightweight $\textit{patch-wise pixel flow decoder}$, which efficiently achieves high-fidelity pixel reconstruction by modeling a conditional flow from the noisy state back to the patch-wise pixel domain. 
By leveraging the semantic features as visual conditions for the decoder, we effectively alleviate the training conflicts between understanding and generation. Furthermore, the patch-wise learning strategy simplifies the data distribution, thereby improving training efficiency.
For instance, our 7B UniFlow-XL not only surpasses the 14B TokenFlow-XL by 6.05\% on average understanding benchmarks, but also achieves a competitive results in both visual reconstruction and generation, surpassing UniTok by 0.15 in rFID and 0.09 in gFID (without guidance), respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes UniFlow, which fine-tunes a pre-trained visual encoder through layer-wise adaptive self-distillation. This method preserves semantics while supplementing the details required for image reconstruction. Additionally, the authors introduce a patch-wise pixel flow decoder that can map directly from noise to the pixel space, which improves the image reconstruction performance.

### Strengths
1. UniFlow can simultaneously achieve strong understanding performance and high-fidelity reconstruction performance.
2. The pixel flow decoder simplifies the training loss and complexity compared to the VAE.

### Weaknesses
1. Despite using of a distillation loss to preserve semantics, a comprehensive comparison of the full understanding performance against the baseline encoder is missing, which prevents validation of the effectiveness of distillation loss. While a comparison between InternViT and UniFlow(InternViT) is provided in Figure 4, the performance gain on MME-S is smaller than on MME-P, suggesting a performance degradation issue for UniFlow on the MME-C benchmark.
2. The lack of understanding of the performance comparison with the baseline also makes the comparison with other methods unfair. For example, in Table 4, it is not clear whether the improvement over other methods comes from better pretraining or Uniflow itself.
3. UniFlow's generative performance does not show a significant advantage and is even weaker than a VAE when using guidance.
4. Considering the generative results of UniFlow, the gain in reconstruction performance from the pixel flow decoder appears to have little positive impact on generation. This leads to concern that UniFlow may have over-optimized for the reconstruction task and consequently failed to learn a feature distribution conducive to generation.
5. The evaluation of the generation is unfair. UniFlow uses a 448x448 resolution during generative training, but the other methods only use a 256x256 resolution during training.
6. The UniFlow-LV notation is not defined within the main text.

### Questions
1. Why is the reconstruction performance for the UniFlow (SigLIP) model different between Table 1 and Table 3, especially regarding the SSIM scores (e.g., 0.93 vs. 0.85)?
2. Can UniFlow be applied to fine-tune the visual encoder already integrated into a pre-trained MLLM (like LLaVA), or must it be trained on a standalone VFM first?
3. What are the benefits of UniFlow for generation? Can it achieve faster convergence?
4. What are the benefits of pixel flow decoder for understanding performance?
5. UniFlow currently only performs generative training on ImageNet. Can it be applied to text-to-image and image editing tasks?

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
This paper introduce a unified image tokenizer, named UniFlow, for visual understanding and generation.
To efficiently unify visual representations within a single tokenizer to achieve both powerful semantic understanding and high-fidelity reconstruction, this paper enhances the tokenizer's ability to extract semantic information and pixel information through layer-wise adaptive self-distillation and patch-wise pixel flow decoder, respectively. This method achieves competitive performance across multiple downstream tasks.

### Strengths
1. Competitive experimental results. UniFlow achieves good performance on multiple downstream tasks.
2. The paper expresses ideas accurately and organizes them clearly. The two proposed strategies: layer-wise adaptive self-distillation and patch-wise pixel flow decoder, effectively improve the model's performance.

### Weaknesses
1. Sufficient comparative experiments. Many of the experiments in the paper do not provide a fair comparison with other methods. For example, in the multi-modal understanding benchmark, although comparisons are made with other methods on LLaVa's data, the initialized semantic encoder and resolution differ. In fact, these two modules are the key reasons for multimodal understanding. I understand that fair comparison experiments with large models are more expensive, but I still hope to reduce variables as much as possible to better illustrate the effectiveness of the method. In addition to this, there is a lack of comparisons with other unified tokenizers in vision-centered tasks. Concerning the ablation experiment section, I believe that merely comparing it to InternViT in multi-modal tasks is inadequate. More comprehensive experiments on downstream tasks are necessary.
2. The paper do not discuss or compare the approach of concatenating features on the CLIP+VAE feature dimensions. It seems that simply concatenating the dimensions can also achieve both powerful semantic understanding and high-fidelity reconstruction, requiring only simple fine-tuning of the pixel decoder.
3. Lack of text-to-image experiments. I think just deploy experiments on Imagenet is not enough. Referring to the settings of LlamaGen or TokenFlow, it is more comprehensive to deploy text-to-image experiments under a considerable data scale.
4.  Comparative experiments with state-of-the-art methods are needed. In Tab.4, there is no comparison with classic representation models such as DINO and the existing unified tokenizer.

### Questions
1. Refer to the weaknesses
2. Confusion about the motivation of the patch-wise pixel flow decoder. From my personal understanding, the core of representation learning work should be on the encoder part rather than the decoder. Because the encoder is the module used for multiple downstream task applications. Although an advanced pixel decoder can achieve better reconstruction and generation performance, this does not seem to match the core purpose of the paper. In previous work, such as BeiTv2, MAE, in the choice of decoder, they tend to have a lighter and simpler structure.

### Soundness
2

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
This paper introduces UniFlow, a unified visual tokenizer designed to simultaneously support both visual understanding and generation. The key innovation lies in combining a layer-wise adaptive self-distillation strategy, which preserves hierarchical semantic knowledge from pretrained vision foundation models, with a patch-wise pixel flow decoder, which reconstructs fine-grained visual details via conditional flow matching in pixel space. Extensive experiments across 13 benchmarks and 7 vision tasks demonstrate that UniFlow achieves state-of-the-art results among unified tokenizers.

### Strengths
1.UniFlow mitigates the representational conflict between understanding and generation by combining semantic-guided self-distillation and pixel-space flow reconstruction. The proposed idea is both technically novel and interesting.
2.The paper benchmarks UniFlow across a diverse range of tasks (VQA, classification, segmentation, detection, depth estimation, generation, reconstruction), and the results show that it consistently outperforms or matches existing unified tokenizers like TokenFlow-XL and UniTok.
3.Extensive ablations in Table 5 clearly demonstrate the effectiveness of adaptive distillation and flow decoder design.

### Weaknesses
1. This paper chose several different semantic teachers, which are pretrained with different resolutions and downsample ratios. It would be better to include a deeper analysis and comparison between them, in addition to the reconstruction, multimodal understanding and generation performance.
2. As UniFlow is trained with both reconstruction and distillation (semantic) objectives, it would be valuable to further analyze the effects of them, e.g., mutual benefits or conflicts.

### Questions
1. As stated in the implementation detail part, UniFlow can do visual reconstruction with one-step Euler inference. It would be interesting to analyze whether using more steps will improve or hurt the reconstruction performance.
2. With only 150M parameters, the flow-based decoder can achieve superior reconstruction performance on ImageNet. How was this model size determined? Could similar performance be achieved with a smaller model, and would scaling up the decoder further enhance reconstruction quality?
3. It would be helpful to include more visualizations of face and text reconstructions, since vision encoders trained with contrastive losses are typically rich in semantic representation but tend to lose fine-grained visual details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents UniFlow, a unified and generic tokenizer designed to bridge the trade-off between visual understanding and generation. The method adapts any pretrained visual encoder with a lightweight reconstruction decoder, enhanced through layer-wise adaptive self-distillation to balance semantic abstraction and fine-grained detail modeling. A patch-wise pixel flow decoder further improves reconstruction efficiency and fidelity by learning conditional flows in the pixel domain. Through this design, UniFlow effectively mitigates the conflict between understanding and generation, achieving consistent improvements across 13 benchmarks. Notably, the 7B UniFlow-XL surpasses larger models such as TokenFlow-XL (14B) in understanding tasks and achieves competitive performance in generation quality metrics.

### Strengths
1. The paper is very clearly written, with a well-motivated problem statement and a logically coherent narrative that effectively highlights the trade-off between visual understanding and generation.

2. The authors provide extensive and detailed experiments across a wide range of benchmarks, offering strong empirical evidence for the proposed method’s effectiveness.

3. The model demonstrates impressive performance in both multimodal understanding and visual reconstruction, showing that UniFlow achieves a well-balanced trade-off without compromising either side.

4. The introduction of layer-wise adaptive self-distillation and the patch-wise pixel flow decoder is both conceptually sound and practically meaningful, representing a thoughtful and generalizable design applicable to various pretrained vision encoders.

### Weaknesses
1. The paper lacks quantitative results on text-to-image generation benchmarks, which limits the evaluation of UniFlow’s generative capability in open-ended visual synthesis.

2. The study does not explore how different frozen teacher encoders or alternative decoder types (e.g., conventional VQ-VAE decoder) would perform under the UniFlow framework, leaving an open question about the generality of its design choices.

3. It remains unclear whether using UniFlow’s learned VQ tokenizer as a visual encoder for multimodal understanding actually surpasses or lags behind directly adopting the frozen teacher encoder; such a comparison would help clarify UniFlow’s benefits.

4. The computational cost and training efficiency of the proposed flow-based decoder are not discussed in detail, making it difficult to assess the practicality of scaling UniFlow to larger models or broader datasets.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
