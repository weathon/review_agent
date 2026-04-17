# v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning

- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
When thinking with images, humans rarely rely on a single glance: they revisit visual information repeatedly during reasoning. However, existing models typically process images only once and thereafter generate reasoning entirely in text, lacking mechanisms to re-access or ground inference in visual representations. We empirically confirm this: as reasoning chains lengthen, models progressively lose focus on relevant regions. In response, we introduce v1, a lightweight extension that enables active visual referencing through a simple point-and-copy approach. This allows the model to identify relevant image patches and copy their embeddings back into the reasoning stream, ensuring that evolving hypotheses remain grounded in perceptual evidence. Crucially, our pointing strategy lets the MLLM directly select image patches using their semantic representations as keys, keeping perceptual evidence embedded in the same space as the model’s reasoning. To train this capability, we construct v1g, a dataset of 300K multimodal reasoning traces with interleaved visual grounding annotations. Across various multimodal mathematical reasoning benchmarks, v1 consistently outperforms comparable baselines, establishing dynamic visual access based on point-and-copy as a practical mechanism for grounded reasoning. We will release the model checkpoint and data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a lightweight extension that explicit re-access region in the input images to avoid visual groundting decay, which helps MLLM to do complex visual reasoning tasks. Their experiments show good performance on visual-math problems and they also claim the training data will be released.

### Strengths
1. The proposed methods sound reasonable and it is a lightweight extension that can be easily used on many MLLM.
2. The section 3 provides a solid support for the motivation of the methods. 
3. The authors provide dataset to finetune its model. Also, the generation method is interesting and inspired.

### Weaknesses
1. The Figure 1 seems to sometimes invisible or incorrectly rendering. Sometimes, the region 2z-15 is too large and covering surrounding text. Please test your image in different devices.
2. Table 1 is too far from the related text.
3. [**MAJOR**] The ablation study in Table 2 is not significant. Specifically, the coord-method improve the backbone by 8.3% while your pointing method improve the w/o pointer version by 9.2%. The difference is not significant to show the benefits of your pointing comparing to the coord.

If the point 3 has been solved, I will raise my score to 6. If some of the following questions can be answered reasonably, I will further raise my score.

### Questions
1. One of the benifit of the method is that it is a lightweight extension and can be used with many existing pretrained MLLM. Another way that has been used in GPT-4o is to let MLLM to write some code to operate the images and put the new images into the context again. The latter one shows better flexible because it can do more on the original images rather than only point the original region. Can the authors provide more comparison between these two methods? 
2. Also, the authors claim the "coordinate" method will "fail in cases where relevant visual cues are abstract or not spatially localized". Can the authors provide some experiments to show the better localizing ability of the point method?
3. The experiments in section 3 can be better explained. First, it is possible that the useful information has been extracted and reprensented in the text in the early stage of the generation, so that it is not necessary to revisit the images. Then, in Figure 3(b), 0.8 is not a significant low value, which raise a question that whether the decay will influnce the performance. I believe the author should provide more evidence about the relation between the decay and the performance.
4. Using cross-attention to ground the region is an interesting idea but it requires some experiments to support its effectiveness. Is there any post-examination or double-check to support its alignment with human intuition or "correct" useful region?

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
The paper "v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning" proposes a novel mechanism to enhance multimodal large language models (MLLMs) by enabling them to re-access visual information during reasoning. Inspired by how humans repeatedly revisit visual stimuli while thinking, the authors design a point-and-copy module that allows the model to identify relevant image patches and inject their embeddings back into the reasoning stream.

To train this capability, they build v1g, a dataset containing 300K multimodal reasoning traces with interleaved visual grounding annotations. Experiments across several multimodal mathematical reasoning benchmarks show that v1 achieves strong performance gains over comparable baselines, demonstrating the potential of dynamic visual access for grounded reasoning.

### Strengths
### 1. Conceptually inspired and technically elegant:

The paper draws inspiration from human problem-solving processes and introduces a clever point-and-copy mechanism that allows re-referencing of visual regions without introducing additional vocabulary tokens.

### 2. Insightful analysis of visual grounding behavior:

The authors identify and analyze issues such as attention degradation and mismatch in visual token importance during long reasoning chains. This motivates the design of mechanisms that preserve visual grounding, making the argumentation coherent and compelling.

### 3. Strong empirical results and valuable dataset:

Through rigorous experiments, v1 demonstrates clear improvements on multimodal mathematical reasoning benchmarks. The proposed v1g dataset is likely to be a useful resource for future research on multimodal reasoning and grounding.

### Weaknesses
### 1. Dependence on external text-trace generation:

The SFT data pipeline relies on an external model (Gemini) to produce text-based traces. This is fine, but it remains unclear how reliably v1 can autonomously propose and execute such traces after training.

**Question:** Is there a quantitative evaluation of the success rate and reliability of detect-call usage during inference?

### 2. Lack of clarity in visual trace generation:

Section 4.3 states that v1’s visual traces are derived via heuristic post-processing of cross-attention maps, yet the details of this algorithm are not provided.

**Suggestion:** Include pseudo-code or a concise algorithmic description in the main text (possibly condensed from Appendix E) along with statistical analysis of the heuristic’s stability.

### 3. Ambiguity in model composition:

It is unclear whether the visual traces during inference are extracted by v1 itself or by a separate pretrained model (e.g., Qwen).

**Question:** How many models are actually involved in the inference loop, and are all functionalities integrated within v1 after training?

### 4. Limited interpretability discussion:

The ablation study on “How does v1 utilize pointed visual regions?” lacks depth.

**Suggestion:** A more detailed explanation would enhance understanding of how v1 internally uses the pointed regions to support reasoning.

### 5. Training procedure justification:

The choice of training for five epochs deviates from common MLLM practice (typically 1–2 epochs).

**Question:** Could the authors provide additional insight into training dynamics and the rationale behind this choice?

### Questions
See the Weaknesses part for details.

**Minor Issues:**

Figures 1 and 2 embedded in the PDF fail to render correctly in Safari and some other browsers. The authors are encouraged to check compatibility or provide rasterized alternatives.

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
3

### Summary
This paper addresses a critical issue in vision-language models (VLMs): the diminishing influence of visual tokens as the chain-of-thought (CoT) lengthens. To mitigate this, the authors introduce a novel training dataset, v1g, and a resulting model, v1. The v1g dataset is constructed from VQA samples, where CoT sequences are augmented by interleaving relevant visual tokens, copied directly from the input, to reinforce visual grounding. For instance, a reasoning step like "query z" is explicitly followed by the visual tokens corresponding to object "z". The authors fine-tune a base VLM on V1G to obtain the V1 model. Experimental results on benchmarks, including MathVista, MathVision, and MathVerse, demonstrate the effectiveness of the proposed approach.

### Strengths
1. This paper proposes a novel training dataset named v1g to address a critical issue in VLMs: the diminishing influence of visual tokens as the chain-of-thought (CoT) length increases.

2. By fine-tuning VLMs on v1g, the authors obtain the v1 model. Experimental results on three mathematical VQA benchmarks demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The evaluation of the proposed method is currently limited to mathematical VQA tasks. Its performance on other domains remains unclear, and further experiments on general VQA benchmarks are needed to assess the generalization capability of both the dataset and the fine-tuned VLM.

2. Does the reuse of visual tokens in the chain-of-thought introduce additional computational overhead compared to text-only CoT? A quantitative comparison of computational costs between the proposed method and the baseline would help clarify this practical concern.

### Questions
1. How does the method generalize to other VQA domains?

2. What is the computational overhead compared to baseline methods?

### Soundness
3

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
5

### Summary
MLLMs often lose visual grounding as reasoning unfolds, since they process images only once before generating purely textual inferences. This paper proposes v1, a lightweight extension that enables active visual referencing through a simple point-and-copy mechanism. v1 allows the model to dynamically select relevant image patches and copy their embeddings into the reasoning stream, keeping inference grounded in perceptual evidence. To train this ability, the authors construct v1g, a dataset of 300 K multimodal reasoning traces with interleaved visual-grounding annotations. Evaluated on three multimodal mathematical reasoning benchmarks, v1 consistently surpasses comparable baselines, particularly on tasks requiring fine-grained visual grounding. These results demonstrate that dynamic visual access via point-and-copy offers an effective and efficient mechanism for grounded multimodal reasoning.

### Strengths
1)	The paper identifies a concrete weakness in current MLLMs, the visual grounding decay problem during reasoning, and provides empirical evidence to substantiate this observation.
2)	The proposed point-and-copy approach offers a simple yet elegant solution to dynamically re-access visual representations during reasoning, conceptually bridging textual reasoning and visual perception.
3)	The method introduces minimal additional parameters (two linear heads), making it easy to integrate into existing MLLMs without significant computational or architectural overhead.

### Weaknesses
1)	Empirical validation is confined to MathVista/MathVision/MathVerse. Broader domains (charts beyond math, documents, VQA, OCR-heavy tasks) are not evaluated, limiting claims of generality. Consider adding non-math benchmarks.
2)	Although the method is said to be generally compatible, experiments are instantiated only on Qwen2.5-VL-7B; cross-backbone results (e.g., InternVL/LLaVA variants) would strengthen the case for portability.
3)	The paper offers only brief training descriptions, making it difficult to assess reproducibility and generalization. More details on optimization settings, schedule, and potential use of reinforcement-style methods (e.g., R1-type reasoning training) would help clarify robustness and strengthen confidence in the reported results.
4)	In Figure 3(b), although the ratio of attention to salient regions decreases as generation progresses, the overall attention level remains relatively high. It is unclear how the authors interpret this result: does a high absolute ratio still indicate grounding decay, or could it reflect stable focus on visual areas? 
5)	In Figure 3(a), the decrease of attention on visual tokens with longer generation steps seems natural for autoregressive reasoning, where attention gradually shifts from perception to internal memory. The key question is not whether decay occurs, but how fast it happens. The paper interprets the observed decline as evidence of model deficiency, yet does not justify why the speed of attention decay indicates unreasonable behavior. A more principled analysis or comparison across models with different decay rates would make this argument more convincing.
6)	Swapping Figures 2 and 3 would improve narrative clarity: the paper should first show the attention-decay problem (Fig. 3) before introducing the proposed point-and-copy solution (Fig. 2).

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
