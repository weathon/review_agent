# One Patch Doesn’t Fit All: Adaptive Patching for Native-Resolution Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Real-world visual signals are inherently variable in resolution, and it is natural to endow multimodal large language models (MLLMs) with such native-resolution perception capabilities. In principle, for general and straightforward multimodal understanding, low-resolution images are sufficient. While for images with nuanced details like documents and charts, it is crucial to preserve fine-grained details using high-resolution inputs, as naive resizing inevitably results in information loss. Recent advances employ sequence packing to process images of any resolution and aspect ratios. Despite these efforts, model performance degrades at both low and high resolutions, and high-resolution inputs incur substantial computational costs. We argue that the rigid use of a single patch size is the primary cause: when image resolution or information density varies, fixing patch size is intrinsically suboptimal.  To address this issue, we introduce Adaptive Patching (AdaPatch), a simple yet effective strategy that adjusts patch size according to image resolution and information density and could be seamlessly plugged into pre-trained fixed-patch MLLMs without any training efforts. Extensive evaluations demonstrate consistent improvements in native resolution performance without additional training. Besides, we provide a training-based method to further adapt MLLMs with dynamic patch sizes and enhance the performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the native resolution issues of MLLMs, and propose an adaptive patching (AdaPatch) method to achieve the optimal patching and visual embedding of input images. The experiments on a set of experiments show the performance gains obtained by AdaPatch for a set of MLLMs.

### Strengths
1. The motivation is clear and plausible. This paper shows that the existing any-res. solution are not optimal due the use of fixed patching. Also, the authors also provide experimental analyses to support their intuitions. 

2. The effectiveness of the proposed methods are validated on a set of recent MLLMs. As shown in Tab.1, the proposed AdaPatch can bring performance gains to the compared MLLMs.

### Weaknesses
1. The method description is vague. The method section introduces various concepts,  but the most important method procedure lacks a clear and intuitive description. For example, how AdaPatch chooses the optimal patch sizes based on the resolution and information density? Subjectively speaking, this paper over-packages theoretical descriptions but lacks explanation of specific details and procedures. 

2. More ablation are required. Although this papers provides a lot of comparisons and charts, it still lacks some critical ablations to show the superiority of the proposed methods. For instance, what about the alternatives to AdaPatch that can also adaptively set patch size? What about the results of using task-specific patch size?  

3. Insufficient qualitative analyses. The authors take more efforts to show the resolution inference to MLLMs on different tasks. However, this seems not the key to AdaPatch, since AdaPatch focuses more on the size of patching, as shown in Fig.1. Besides, the authors are required to give more information about the results of AdaPatch. For instance, what are the patch size set to different tasks? the avg. path size of different benchmarks?

### Questions
1. In Sec.3.2, what are the purpose and motivations of Pseudo-inverse and Multi-scale patch embedding for AdaPatch? In particular, why MS patch embedding should be proposed considering the training-free target claimed in Abs and Intro. 

2. Following last question, what are the settings of AdaPatch for the results of Tab.1? Similarly, the settings in other tables should be also specified. 

3. The comparison of Tab.2 is unfair. Most MLLMs train SFT data only for once, and the additional fine-tune is likely to get performance gains. In other words, Tab.2 should also provide the SFT performance of MLLMs being trained for twice. 

4. What about the results of using the optimal patch sizes for different tasks, i..e., task-specific setting rather than example-level adaptive settings. 

5. In terms of Fig.1, does AdaPatch has other differences to dynamic-res operations of QWen and InternVL in addition to adaptive patching?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem that multimodal large language models (MLLMs) with a fixed image patch size perform suboptimally across varying image resolutions. The authors propose a plug-and-play method that dynamically adjusts the patch size based on an image's resolution and information density. Additionally,  a trainable version is also offered for further performance gains. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed AdaPatch can be seamlessly plugged into pre-trained fixed-patch MLLMs without any training efforts.
2. The proposed AdaPatch achieves promising results on several benchmarks.

### Weaknesses
1. The most significant issue with this paper is that the baseline metrics do not align with the results from the original paper or the vlmevalkit. For instance, on MME, the reported Qwen2.5-VL-3B metric in this paper is 2135.90, whereas the vlmevalkit reports 2199.9. On MMMU, the paper reports 49.22 for Qwen2.5-VL-3B, compared to vlmevalkit's 51.2. On MMStar, the paper reports 54.27, while vlmevalkit reports 56.3. The discrepancies are quite substantial. Since the authors used vlmevalkit as the evaluation framework, it would be better to align the baseline with the official vlmevalkit evaluation results. This would enhance the credibility of the experiment results.

2. Lack of metrics regarding the additional computational load and the impact on inference speed brought by AdaPatch.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the limitation of fixed patch sizes in MLLMs that claim “any-resolution” capability. The authors propose AdaPatch, a training-free and training-based framework that adjusts patch size according to both image resolution and information density. The method estimates information density using feature similarity between original and downsampled representations, then determines the patch size via a simple power-law mapping. A “pseudo-inverse resize” operation enables pretrained fixed-patch MLLMs to operate under any patch size without retraining. Experiments on multiple benchmarks (MME, MMBench, OCRBench, DocVQA, etc.) across four representative MLLMs (Qwen2.5-VL, Ovis2.5, SAIL-VL, Kimi-VL) demonstrate consistent improvements in native-resolution stability and accuracy.

### Strengths
(1) This paper thoroughly evaluates the performance of AnyRes by systematically rescaling benchmark datasets across a broad spectrum of input resolutions, uncovering substantial degradation and instability in recent state-of-the-art multimodal large language models (MLLMs) that employ fixed patch sizes.

(2) This paper introduces AdaPatch, a simple yet highly effective approach that dynamically adjusts patch size based on both input resolution and estimated information density, enabling more adaptive and content-aware visual tokenization.

(3) This paper provides two complementary implementations: a training-free variant that allows immediate deployment without any model modification, and a training-based version that supports further fine-tuning for enhanced performance.

(4)  This paper demonstrates consistent improvements in accuracy, stability, and computational efficiency across multiple benchmarks and model architectures, with especially notable gains at high input resolutions.

(5) This paper leverages publicly available models and open-source evaluation toolkits, ensuring that all empirical results are transparent, reproducible, and easily verifiable by the research community.

### Weaknesses
(1) The information-density formulation (Eq.3) lacks theoretical justification and comparison with simpler statistical metrics.
(2) The patch-size range [6,56] is empirically set without analysis of generalization to other domains.
(3) Notation and figures show minor inconsistencies and labeling issues that reduce clarity.
(4) The method’s performance on LLaVA-Bench is relatively weak and not discussed.
(5) The tuning process of α and β is unclear and may not reflect joint optimization.
(6) The authors can address these points by responding to the Questions section.

### Questions
(1) Information density estimation (Eq.3): Have you compared this definition with more intuitive statistical measures (e.g., gradient entropy or Laplacian variance), and how do the results differ? What is the rationale for choosing the 0-th layer features of the visual encoder for this computation? How do you verify the effectiveness of Eq.3 as a valid information-density estimator?

(2) Candidate patch sizes ([6, 56]): How was this specific range determined? For other image domains such as medical or remote sensing, would this range need to be extended or adjusted?

(3) Notation inconsistency: The symbol r̃ is referred to as “target resolution” in Section 2.1 but as “base resolution” in Eq.4, which is inconsistent. The notation throughout the paper should be unified, and it is recommended to include a notation table in the appendix for clarity.
(4) Figures:
(a) In Figure 1, several labels use “patchsize” instead of “patch size.”
(b) In Figure 3, the proportional relation s ∝ r / ρ is not visually evident.
(c) Figure 7 lacks a descriptive title.

(5) Performance on LLaVA-Bench: According to Tables 1 and 3, the proposed method performs relatively modestly on LLaVA-Bench. Could the authors analyze the potential reasons for this difference?

(6) Hyperparameter tuning (α, β): Were α and β tuned jointly or separately? Based on Figure 7(a), the final selected values appear to be determined independently.

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
This paper identifies a critical weakness in current "native-resolution" Multimodal Large Language Models (MLLMs): their performance is unstable and often degrades at very low or very high resolutions. The authors convincingly argue that the root cause is the rigid use of a fixed patch size, which creates a mismatch between the receptive field granularity and the image's resolution or information density.
To address this, the paper proposes Adaptive Patching (AdaPatch), a simple yet effective method that dynamically selects an appropriate patch size for each input image based on its resolution and a novel information density metric. The authors also introduce PI-resize, a training-free technique to adapt existing fixed-patch MLLMs to handle variable patch sizes, making AdaPatch a practical drop-in solution. Through extensive experiments on several state-of-the-art MLLMs and a wide range of benchmarks, the paper demonstrates that AdaPatch not only consistently improves performance but also significantly enhances performance stability across resolutions, while reducing computational costs for high-resolution inputs.

### Strengths
1.	The proposed AdaPatch method is intuitive, simple, and addresses the identified problem directly. The core idea of adapting patch size to resolution and information density is logical, and the proposed implementation is straightforward.
2.	The introduction of PI-resize, a training-free method to convert existing models, is a significant strength. It allows AdaPatch to be seamlessly integrated into powerful, pre-trained MLLMs without requiring costly retraining, which dramatically increases the practical utility and potential impact of this work.
3.	The evaluation is comprehensive, covering multiple base models (Qwen2.5-VL, SAIL-VL, Ovis2.5, Kimi-VL), a diverse suite of benchmarks, and various model scales. The head-to-head comparisons showing improved stability across pixel ranges (Figure 5) are particularly compelling and strongly support the paper's claims.

### Weaknesses
1.	Limited Improvements on Information-Dense Tasks and Scaling: While AdaPatch shows consistent overall gains (Table 1), the improvements on VQA and OCR tasks are often surprisingly marginal. This is counter-intuitive, as OCR  tasks should benefit most from the adaptive preservation of fine-grained details. Furthermore, upon scaling the models (Table 3), the performance uplift provided by AdaPatch becomes extremely limited for larger backbones (e.g., Qwen2.5-VL 7B and 32B), suggesting that the technique's benefits might not scale effectively with model capacity.
2.	Performance Plateau and Ceiling Limitations: As demonstrated in the pixel-range evaluations (Figure 5), although AdaPatch successfully maintains superior stability compared to fixed-patch baselines, it appears to primarily mitigate performance degradation rather than significantly elevating the overall performance ceiling. Specifically, performance still seems to plateau or slightly decline at the highest tested resolutions. This raises questions about the long-term potential of the method to achieve true native resolution robustness.

### Questions
1.	Given that AdaPatch is specifically designed to optimize detail preservation, which is crucial for OCR and document tasks, could the authors provide a deeper analysis into why the absolute performance gains on benchmarks like OCRBench and DocVQA are marginal, particularly for the largest models in Table 3?
2.	The results in Figure 5 indicate that stability is dramatically improved, but the performance ceiling does not rise substantially as resolution increases. Do the authors view this plateauing effect as an inherent limitation of the current ViT-MLLM paradigm, or are there planned extensions to the Adaptive Patching Law (Eq. 4) that could further unlock performance gains at ultra-high resolutions?

### Soundness
3

### Presentation
3

### Contribution
3
