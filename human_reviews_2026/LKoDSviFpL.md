# SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
In this paper, we introduce SemHiTok, a unified image Tokenizer via Semantic Guided Hierarchical codebook that provides consistent discrete representations for multimodal understanding and generation. Recently, unified image tokenizers have sparked exploration within the research community, which is designed to capture high-level semantic features for understanding and retaining low-level pixel features for generation. Previous works attempt to train a unified image tokenizer by combining loss for semantic distillation and pixel reconstruction. However, due to the differing levels of features prioritized by multimodal understanding and generation, joint training methods face significant challenges in achieving a good trade-off. SemHiTok addresses this challenge through a novel semantic-guided
hierarchical codebook, which builds pixel sub-codebooks on a pretrained semantic codebook. This design decouples the semantic and pixel in terms of structure and training strategy, enabling the tokenizer to capture pixel features while retaining its ability to comprehend high-level semantic information. Our experiments demonstrate that SemHiTok achieves leading performance in image reconstruction and multimodal understanding under the LLaVA-v1.5 setting. Further, we develop a unified MLLM with SemHiTok, which exhibits superior performance across multimodal understanding and generation tasks. Extensive experiments confirm our analysis, showing that our unified image tokenizer architecture achieves a better trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a unified tokenzier for both image understanding and generation. By linking a set of pixel code with a semantic code, the proposed SemHiTok makes a good trade-off between semantic image understanding and pixel image generation. Experiments on image reconstruction, understanding and generation confirm its effectiveness.

### Strengths
- The motivation and illustration of the proposed method are clear and easy to follow.
- Experiment results are good on multiple tasks, including image reconstruction, understanding and generation.
- The ablation experiments are clear to demonstrate the effect of each component.

### Weaknesses
- Image reconstruction. The rFID is not enough to prove the real performance of a tokenizer for image reconstruction. Other metrics like PSNR are encouraged. Besides, unified tokenizers like UniTok, MUSE-VL should be compared in Tab.1.
- Training setting. Most unified models are jointly trained on a mixture of multimodal understanding and text-to-image data, the proposed method are only trained on LLaVA-1.5 and text-to-image settings seperately, which may not reflect the relation between image understanding and generation under a unified model.
- Generation benchmarks. Recent popular benchmarks such as Gen-Eval and DPG are missing. Besides, it's better to include more recent methods on Tab.4.

### Questions
no.

### Soundness
3

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
4

### Summary
The paper proposes SemHiTok, an image tokenizer for unified MLLMs, which provides both semantic features and pixel-level features for multimodal understanding and generation tasks. The tokenizer features a hierarchical architecture, where an image is first quantized into semantic codes and then the pixel-level tokens are selected based on the corresponding pixel sub-codebook. Experiments demonstrate the effectiveness of SemHiTok on both image understanding & generation tasks.

### Strengths
* The concept of the hierarchical codebook for high-level semantic features and low-level pixel features is novel, and well-motivated. The method decouples the optimization of the two conflicting objectives for semantics and pixel-level details.
* There are comprehensive experiments which effectively demonstrate SemHiTok's strong performance both at the tokenizer level and in its application on a unified MLLM.

### Weaknesses
* While the selection of pixel sub-codebooks is guided by the semantic codes, the training of the pixel-branch is decoupled, so the link between the two hierarchical levels is weak. The authors argue that SemHiTok is superior to naively combining two separately trained tokenizers (in Table 6 Exp 3 & 4), but the conceptual difference feels incremental. Perhaps the authors should provide a more explicit discussion on the specific advantages of this hierarchical design over a simpler, two-stage concatenation approach.

* Though the authors state that SemHiTok avoids codebook overexpansion, the codebook size of SemHiTok is still large (196608), much larger than baseline methods. So the comparison of performances may not be fair, since a much larger codebook inherently allows for higher-fidelity reconstruction.
* There are several grammatical errors, and maybe some mistakes in the formulations (Eq 2, 5).

### Questions
About the observation that image patches with the same semantic code tend to have similar pixel feature. Are there any quantitative verification (e.g., specific metrics or analysis) for this observation? This would better solidify the motivation for using the semantic code to index pixel sub-codebooks

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SemHiTok, a unified image tokenizer enhanced by semantic information guidance. It innovatively introduces a hierarchical codebook structure, which builds a pixel sub-codebook based on a pre-trained semantic codebook. The semantic part and the pixel part are trained separately to decouple the structure and training strategy. This enables the tokenizer to capture pixel features while retaining its ability to comprehend high-level semantic information. Additionally, SemHiTok is applied to the MLLM structure, and the experimental results demonstrate the performance of this method.

### Strengths
- The experiments are comprehensive, comparing with multiple state-of-the-art (SOTA) models and demonstrating the superiority of the SemHiTok method.
- The writing language is accessible, and the diagrams are clear.
- As a Tokenizer, it underwent complete training on the MLLM architecture, validating its effectiveness.

### Weaknesses
- The paper only conducted ablation experiments on the joint training vs. phased training of the SemHiTok architecture, but did not compare joint training and phased training across other methods. Thus, the conclusion that "Joint training degrades performance" lacks sufficient support.
- The roles of the sub-codebook and phased training have not been ablated individually in the experiments, making it insufficient to demonstrate the degree of validity of each component on its own.

### Questions
- Typo: In the header of Table 5, "Ehance" should be corrected to "Enhance".
- How is the size m of the sub-codebook determined, and does it impact the model's performance?
- Is it necessary to retrain the entire LLM to verify SemHiTok's capabilities? Perhaps one can only replace the Tokenizer in the existing MLLM (Multimodal Large Language Model) structure. Is this new MLLM architecture necessary?

### Soundness
3

### Presentation
4

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
This paper introduces SemHiTok, a unified image tokenizer designed to effectively capture both high-level semantic features for understanding and low-level pixel details for generation. The authors identify a key challenge in joint training: the inherent conflict between the feature priorities of multimodal understanding and generation tasks. To bridge this gap, they propose a novel Semantic-Guided Hierarchical Codebook (SGHC), which employs a set of sub-codebooks to model the pixel-level space under the guidance of each semantic code. A notable advantage of SemHiTok is its compatibility with existing next-token-prediction-based MLLMs, achieved through a straightforward codebook flattening operation.

### Strengths
This paper is well-structured and clearly written, making it a pleasure to read. The core idea of SemHiTok is both novel and compelling, presenting a fresh perspective on the problem. Consequently, I have no major questions regarding the technical content presented. Should I have overlooked any aspect of the work, I welcome the authors to clarify it in their rebuttal.

### Weaknesses
Although this paper proposes an alternative method to bridge the gap between low-level visual cues and high-level semantic features, I still have several concerns.

First, the hierarchical structure significantly increases the codebook size. As shown in Table 1, SemHiTok has a total size of K × M = 196,000. It appears impractical to expand this codebook further. However, this already large size limits the value of K. I wonder whether a small K is sufficient to represent the full diversity of semantics in visual content.

Second, the comparisons regarding codebook capacity do not seem entirely fair. As shown in the table, methods like FQGAN and IBQ achieve better reconstruction quality with the same resolution and codebook dimension, yet have a smaller codebook size. This raises a question: given abundant data (far beyond ImageNet-50K), would simply using a large enough standard codebook be sufficient, thereby diminishing the necessity of the proposed SemHiTok? In other words, can SemHiTok maintain its competitiveness under such conditions? For instance, the EMU series (e.g., the recently released EMU-3.5) employs a very large visual codebook and massive pre-training data, which leads to strong performance.

I would appreciate it if the authors could address these concerns.

### Questions
* Does the authors mention the specific value of K and m for the SemHiTok's codebook? Have you ever carried out ablation of different value sets of (K, m) and their effectiveness? It could bring more insights if the authors could provide more details about how they decide the scale of codebook.

### Soundness
2

### Presentation
3

### Contribution
2
