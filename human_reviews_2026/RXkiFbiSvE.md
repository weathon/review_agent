# LlamaSeg: Image Segmentation via Autoregressive Mask Generation

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
We present LlamaSeg, a visual autoregressive framework that unifies multiple image segmentation tasks via natural language instructions. By reformulating segmentation as visual generation, LlamaSeg encodes masks as visual tokens and uses a LLaMA-style Transformer for direct next-token prediction, naturally fitting segmentation into autoregressive architectures. To support large-scale training, we introduce a data annotation pipeline and construct the SA-OVRS dataset, which contains 2M segmentation masks annotated with over 5,800 open-vocabulary labels or diverse textual descriptions, spanning diverse real-world scenarios. This enables our model to localize objects in images based on text prompts and to generate fine-grained masks. We further introduce the composite metric average Hausdorff Distance ($d_{\mathrm{AHD}}$) to evaluate mask contour fidelity for generative models better. Experiments show that LlamaSeg consistently outperforms existing generative approaches on multiple segmentation benchmarks and delivers finer, more accurate segmentation masks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes LlamaSeg, a visual autoregressive model that reformulates image segmentation as visual generation using a LLaMA-style Transformer. Trained on the large-scale SA-OVRS dataset with 2M annotated masks, it supports text-guided segmentation and fine-grained mask generation.

### Strengths
1. LlamaSeg introduces the idea of using an image tokenizer to encode segmentation masks, effectively unifying various segmentation tasks within a discrete autoregressive framework. 

2. The paper is clearly written and easy to follow.

### Weaknesses
1. The comparison baselines are outdated, and LlamaSeg’s segmentation performance is not competitive (e.g., around 56 on RefCOCO), which is significantly lower than recent methods such as Ferret-v2 [1] (≈90).

2. The relatively poor performance raises doubts about whether encoding masks using image tokenizer truly offers advantages over encoding them as discrete position tokens or point sequences. A more detailed comparison and ablation studies (including performance and efficiency) across different mask encoding strategies is needed to justify this design choice.

3. Encoding a single mask requires hundreds of visual tokens, which appears less efficient than directly encoding the mask as a compact sequence of points(represented as position tokens in Kosmos-2).

[1] Ferret-v2: An Improved Baseline for Referring and Grounding with Large Language Models. arXiv preprint arXiv:2404.07973, 2024.

[2] Kosmos-2: Grounding multimodal large language models to the world[J]. arXiv preprint arXiv:2306.14824, 2023.

### Questions
1. My main concern lies in the motivation of this work. It remains unclear what specific advantages the use of an image tokenizer offers over existing mask encoding methods. Is it intended to improve performance, efficiency, or both?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper present the LlamaSeg network which is a visual autoregressive model that can apply image segmentation tasks with natural language instructions. The proposed model LlamaSeg encode the input as visual tokens and use LLAMA style network for next token prediction.

The paper also introduce a new data annotation framework which contains 2M segmentation masks over 5800 labels (SA-OVRS dataset). The new dataset allow the model to localized object based on text prompts. They also introduce a new metric called, Hausdorff Distance to measure the mask contour fidelity. The evaluation of the proposed model shows that in outperform existing methods.

### Strengths
1. The proposed method have unified formulation for multiple segmentation tasks such as - semantic, referring, open-vocabulary in one autoregressive model.

2. The proposed method has strong boundary fidelity which cause due to the use of mask-tokenizer and autoregressive decoding. 

3. The new dataset SA-OVRS is a large one, with open-vocabulary supervision which improve the performance in multiple tasks.

### Weaknesses
1. The proposed method has lower performance on some tasks when comparing to discriminative models

2. The tokens that used has fixed downsample of ×16, which can miss fine details

3. The usage of autoregressive model has some latency issues which is much slower than discriminative models

### Questions
1. What is the latency of the proposed model? Is there any trade of between the performance and the runtime latency?

2. Does using finer stride can improve the results of the proposed mode? for example if you use ×8 in the mask tokenizer does the performance imporve?

3. How does the model behave for out of distribution data such as medical or aerial imagery without finetune?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LlamaSeg, an autoregressive framework for image segmentation that unifies multiple segmentation tasks under the paradigm of next-token prediction.  
The key idea is to reformulate segmentation as visual generation, encoding segmentation masks as discrete visual tokens through a VQGAN tokenizer and generating them autoregressively using a LLaMA-style Transformer.  
To support large-scale training, the authors construct SA-OVRS, a new dataset containing 2M segmentation masks annotated with over 5,800 open-vocabulary labels and textual descriptions.  
Additionally, a novel evaluation metric, average Hausdorff Distance (dAHD), is proposed to assess contour fidelity of generated masks.  
Extensive experiments show that LlamaSeg surpasses existing visual generative models (e.g., Unified-IO and Unified-IO2) on both semantic and referring segmentation benchmarks, while producing finer and more accurate mask boundaries.

### Strengths
### 1. Conceptual novelty  
Reformulating image segmentation as an autoregressive mask generation problem is a creative and elegant extension of large language model principles to pixel-level prediction.  
This perspective bridges the gap between generative modeling and structured visual understanding.  

### 2. Unified framework  
The proposed approach enables seamless integration of segmentation tasks into LLM-based architectures through a consistent tokenization and generation pipeline.  
It provides a promising step toward unifying pixel-level vision tasks with autoregressive multimodal modeling.  

### 3. Dataset contribution  
The introduced SA-OVRS dataset offers large-scale, open-vocabulary segmentation data paired with rich textual descriptions.  
This resource can support future research in open-vocabulary and multimodal segmentation.  

### 4. Evaluation rigor  
The introduction of the dAHD metric is an insightful addition, providing a more nuanced measure of boundary accuracy and contour fidelity compared to traditional IoU-based metrics.  

### 5. Empirical validation  
Comprehensive experiments across multiple benchmarks and datasets demonstrate consistent quantitative and qualitative improvements, validating the effectiveness of the proposed framework.

### Weaknesses
### 1.  Limited scope and contribution
The contribution feels more foundational within a narrow scope rather than broadly transformative.  The method primarily focuses on segmentation and language alignment, without clear extensions to other modalities or tasks such as vision-language reasoning, instruction following, or general multimodal generation.  Compared with highly integrative multimodal frameworks like Unified-IO, 4M-21 (Bachmann, Roman, et al. "4m-21: An any-to-any vision model for tens of tasks and modalities." Advances in Neural Information Processing Systems 37 (2024): 61872-61911.), this work appears less comprehensive and serves more as an initial step toward unifying segmentation within the LLM paradigm rather than a fundamentally new multimodal foundation.

### 2. Incomplete comparison with recent foundation models
Beyond Unified-IO, there exist broader any-to-any vision models such as 4M-21, which support a wider range of tasks and modalities while achieving comparable semantic segmentation results.  The paper should include a direct comparison and a detailed discussion to better contextualize its contribution relative to such works.

### 3. High complexity and unclear efficiency
The proposed segmentation process is quite complex. It involves label generation, mask matching, and a separate inference step for each mask.  This design is inefficient for dense semantic segmentation scenarios.  The paper should report inference time, throughput, and computational cost to clarify the practical feasibility of the approach.

### Questions
1. Could the authors clarify whether the model can generalize beyond segmentation to other vision-language tasks, such as referring expression comprehension or open-ended visual reasoning? 

2. The segmentation process appears computationally heavy, requiring separate inferences for each mask.  Could the authors provide concrete runtime statistics (e.g., FPS, latency per image, GPU hours) and discuss possible optimizations for dense segmentation settings?

3. Since the model leverages VQ-based mask tokenization, how sensitive is it to the quality of the VQ tokenizer?  Would retraining or substituting the tokenizer significantly impact segmentation accuracy?

### Soundness
3

### Presentation
3

### Contribution
3
