## Human Reviewer 1

### Summary
This paper tackles the problem of inefficient training issues in diffusion models, which often rely on large-scale text–image pairs for supervision. The authors propose Generative Modeling with Explicit Memory (GMem), a framework that conditions generation on high-quality semantic features extracted directly from data and stored in an external memory bank. This memory-based conditioning provides a more accurate and efficient guidance signal, enabling up to 50× faster training and state-of-the-art performance on ImageNet. Moreover, GMem allows rapid domain adaptation and a data-efficient text-to-image pipeline, reducing both dataset size and computational cost while maintaining competitive quality.

### Strengths
1. This paper addresses an important and timely problem: improving the efficiency of training generative models. 

2. The proposed methods are conceptually sound, clearly presented, and relatively easy to follow. 

3. The paper presents a comprehensive set of experiments, demonstrating the proposed method’s effectiveness across multiple dimensions, including generation quality, adaptability to new tasks, and training efficiency.

### Weaknesses
1. The dependency on REPA appears to be a crucial factor that requires further clarification. In Table 1, most of GMem’s results are reported on top of REPA, making it unclear whether GMem alone yields significant improvements. A comparison isolating GMem’s contribution would strengthen the claim of its independent effectiveness. 

2. The claimed advantages of the Bank-Free T2I design (lines 307–314) require more careful justification. For instance, regarding privacy preservation, the paper argues that Bank-Free T2I eliminates privacy concerns by internalizing semantic information into model parameters. However, traditional T2I models also encode semantic information within their parameters. Similarly, since standard T2I approaches do not require external memory banks, the claim of superior storage efficiency needs more substantiation. 

3. To support the reported superiority in text-to-image (T2I) performance, it would be beneficial to include additional, widely recognized metrics such as T2I-CompBench[A] or HPSv2[B]. The current reliance on MJHQ-FID provides a narrow perspective on T2I performance. Incorporating these broader benchmarks, especially since the baseline PixArt-α also uses T2I-CompBench, would offer a more convincing evaluation. 

4. The discussion on efficient downstream adaptation (Section 5.3) would benefit from deeper ablation studies and comparative analysis. While GMem demonstrates rapid adaptation in "absolute" training steps, efficiency should also be evaluated "relative" to strong baselines (e.g., REPA) by comparing computational overhead and convergence speed. 

5. The manuscript would greatly benefit from extensive proofreading and consistency checks. For example, the important baseline PixArt-α is repeatedly miswritten as PixelArt-α, and the number of training epochs (450 in line 362) seems a wrong information. Although minor, such issues may confuse readers and reduce the paper’s overall polish and credibility.

[A] T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation, NeurIPS 2023

[B] Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces GMem (Generative Modeling with Explicit Memory), a diffusion-based generative framework that conditions
generation on semantic information, which stored in an external memory bank, extracted directly from the data. GMem significantly improved training efficiency. Moreover, the authors also extend GMem to a text-to-image (T2I) setup with strong data and time efficiency.

### Strengths
- The paper is well written and easy to follow.
- GMem demonstrates comprehensive empirical validation - the experiments are broad, spanning CIFAR-10, ImageNet (256×256, 512×512), and T2I tasks. And it also shows strong emperical resultls - outperforming prior baselines such as SiT, REPA, and PixelArt-α on various metrics covering efficiency and quality.

### Weaknesses
- Limited conceptual novelty. The paper does not sufficiently clarify what is fundamentally new. The core idea is conceptually close to retrieval-augmented diffusion (e.g., REPA, KNN-Diffusion, Memory-Driven T2I).
- Lack of theoretical or mechanistic insight. The paper does not analyze why explicit memory accelerates convergence or improves sample quality.
- Extracting features from a pretrained model to build the memory bank requires running inference on millions of images, which can take hundreds of GPU-hours for ImageNet-scale datasets, but this cost is probably not included in training time.
- Training converges faster because the network conditions on semantically rich embeddings rather than noisy text or random class tokens, experical results are very strong but it's not a new algorithmic insight.

### Questions
- The reported “50×” training speedup, is it due to property of GMem? Is the speedups measured under identical optimization and data settings as baselines?

### Soundness
3

### Presentation
4

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposes GMem, a diffusion-based generative framework that conditions image synthesis on high-quality semantic representations stored in an external memory bank instead of noisy text prompts. This design enables dramatically faster training, which achieves an SOTA FID of 1.53 on the ImageNet dataset, 50× faster than SiT. It also enjoys rapid downstream adaptation to new domains.

### Strengths
1. The paper is well written. Figures (especially Figure 3) effectively illustrate the architecture, training/sampling pipelines, and relationships to prior work.
2. The core idea is conceptually fresh and thoughtfully motivated.
3. The paper focuses on three timely and important challenges in generative modeling: training inefficiency, poor adaptability, and reliance on large paired datasets.

### Weaknesses
1. GMem’s core innovation relies on storing a full-dataset memory bank (e.g., 1.28M snippets for ImageNet), the cost is manageable for ImageNet but prohibitive for LAION-scale T2I.
2. Section 5.3 states that an ImageNet-pretrained GMem can adapt to new domains (anime, faces) in ~20K fine-tuning steps. However, there is no comparison to fine-tuning a strong baseline model on the same downstream tasks.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper presents GMem, a method to improve diffusion image generation models by adding an external memory bank of semantic features (also referred to as memory snippets in the paper). These features are extracted during training, and, at inference time, are used as additional conditioning. The authors show that adding this external memory significantly speeds up training compared to a SiT and still reaches a good FID score. In addition, the memory bank can also be easily finetuned to adapt to new domains, and easily converted to help text-to-image generation.

### Strengths
-	Authors’ experiments show Gmem brings an order of magnitude speedup in training time without quality degradation. This is exciting.
-	GMem has the potential to become an easily-adaptable model in few-shot transfer learning. The memory snippet could be used for test-time editing.
-	The author discusses storage optimization techniques plus big-O analysis.
-	The authors tried multiple image encoders to show the generalizability of the method.
-	The idea is simple, thus practical to reproduce elsewhere.
-	The paper is well-structured and easy to follow. Discussion of related works is ample.

### Weaknesses
-	Using an external memory bank is novel, but there are similar works in RAG (retrieval-augmented generation), like RDM. Compared to them, the author’s innovation is in compressing the database, which is incremental.
-	Section 4.1 deserves more details. The authors say the memory bank is constructed by collecting N snippets from dataset D. But how does the author determine which N snippets? We can safely assume N << size(D), right? Or is N determined by a fixed ratio on top of the size of the data? This is not explained in the appendix either.
-	While the authors apply masking techniques to avoid over-memorization (the model just decodes a snippet), the metrics in the experiments cannot measure the true diversity of the output. It is recommended to compare the similarity with the training set images to see if it has over-memorization.
-	In Section 4.3, the reviewer does not fully understand why mapping a sampled noise to some element in the bank ensures good generation diversity. Why is a normal distribution not a uniform distribution used? Would this make the center indices more likely to be chosen?
-	In Fig.3 right side, the arrows between “class embed”, “Memory bank”, and “Text encoder” are a bit confusing. At first sight, readers might think there is some data transformation between them and not realize that these refer to different models.

### Questions
-	Line 693 and 696 have duplicated reference.
-	Line 514, the cited reference seems wrong. Should be Retrieval-Augmented Diffusion Models by Blattmann et al.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3