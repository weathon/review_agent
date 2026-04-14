# From Training-Free to Adaptive: Empirical Insights into MLLMs' Understanding of Detection Information

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Despite the impressive capabilities of Multimodal Large Language Models (MLLMs) in integrating text and image modalities, challenges remain in accurately interpreting detailed visual elements. Fortunately, vision detection models have shown superior performance in recognizing fine-grained image details, leading to their increased deployment by researchers to enhance the ability of MLLMs. Among the feasible strategies, infusing detection information in text format is easy to use and effective. However, most studies apply this method in a training-free manner. There is limited research on the effects of adaptive training, which has great potential for helping LLMs better comprehend the special input and discard irrelevant information. In this paper, we address the key research question: How does training influence MLLMs' understanding of infused textual detection information? We systematically conduct experiments with numerous representative models to explore the performance implications of training-free, retraining, and fine-tuning strategies when infusing textual detection information into MLLMs. Additionally, we investigate the impact of training on the original abilities of MLLMs, as well as the interchangeability of detection models. We find that fine-tuning the pre-trained MLLM to adapt to textual detection information yields better results compared to the training-free strategy and the retraining strategy, with the fine-tuned MLLM outperforms the training-free MLLM by 6.71\% across 10 widely recognized benchmarks. Besides, we find that fine-tuning allows the MLLM to maintain performance improvements even after replacing the deployed detection models, which means that it enables the MLLM to better understand the specially formatted textual information. We release our codes to facilitate further exploration into the fusion strategies of vision detection models and improving the fine-grained multimodal capabilities of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
MLLMs struggle with accurately interpreting fine-grained visual details. While vision detection models excel at this, most studies simply insert the detection information as text into the MLLM without further training (training-free).  This paper investigates whether adaptive training can improve the MLLM's understanding of this added textual detection information, leading to better performance than training-free methods.

### Strengths
1. Reasonable motivation. Additional vision experts can further enhance the visual capacities of MLLMs. The paper finds the adaptive training can achieve great potential for helping LLMs better comprehend the special detection input.

2. The conducted experiments and visualizations are extensive and well-organized.

3. The paper is well-written and easy to understand.

### Weaknesses
1. The paper lacks analysis regarding the impact of detector performance. Would a detector with significantly higher mAP lead to greater MLLM improvement?

2. Detectors trained on larger datasets with more categories, such as LVIS (1.2k categories) compared to COCO (80 categories), potentially achieve finer-grained visual understanding. Would using the LVIS-trained detector, like Co-DETR-LVIS [1], improve FTBI performance?

3. The proposed method with an open-set detector is similar to VisualCOT [2]. Both first locate the box region that is relevant to the user question and leverage the region information to help MLLM better answer the question.

4. Can FTBI further improve performance upon stronger open-source baselines like LLaVA-NeXT [3] and LLaVA-OneVision [4]?

5. There are two paradigms to incorporate detection experts into MLLMs in the community. One converts detector outputs directly into text descriptions for the MLLM (as in this paper, MoAI [5] and IVE [6]), while the other fuses detector vision backbones with CLIP features (MoVA [7] and Eagle [8]). What advantages does the former approach offer?

[1] Detrs with collaborative hybrid assignments training. ICCV 2023.

[2] Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models. NeurIPS 2024.

[3] Llava-next: Improved reasoning, ocr, and world knowledge.

[4] Llava-onevision: Easy visual task transfer.

[5] Moai: Mixture of all intelligence for large language and vision models. ECCV 2024.

[6] Incorporating visual experts to resolve the information loss in multimodal large language models.

[7] Mova: Adapting mixture of vision experts to multimodal context. NeurIPS 2024.

[8] Eagle: Exploring the design space for multimodal llms with mixture of encoders.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Inspired by the absence of adaptive training methods to integrate textual detection information into MLLMs, the paper empirically explored the effect of fine-tuning MLLMs equipped with textual detection information. The key insights were 1) fine-tuning strategy yields better performance than training-free and retraining strategies, 2) retraining rather impairs the original image comprehension ability of MLLMs, and 3) Swapping the deployed detection model with open-set object detector further improves MLLM performance.

### Strengths
- The paper conducted a comprehensive set of experiments with thorough analysis for integrating textual detection information into MLLM
- The empirical observations are straightforward and the paper is written in easy-to-understand manner

### Weaknesses
- The key insights and the empirical observations this paper investigated may seem to reiterate the existing observations of the related papers. Specifically, [MiniGPT4-v2](https://arxiv.org/abs/2310.09478) and  [VisionLLM](https://arxiv.org/abs/2305.11175) are the pioneering works that demonstrated the positive impact of integrating object detection information into MLLMs in terms of object detection and several MLLM tasks (e.g., VQA).
- Additionally, the paper overlooks the effectiveness of training-free methods, which avoid the need for a huge amount of labor-intensive annotations required for equipping such large-scale MLLMs with object detection ability.
- The novelty of the proposed methods is significantly limited, which is a simple adoption of training modules onto training-free infusion models.
- The technical soundness of the proposed methods seems deficient. Why is the retraining strategy that trains MLLM from scratch not training visual encoders? Also, there is no justification for why the different backbone (vicuna) is used for retraining, compared to the other fine-tuning and training-free strategies.

### Questions
See above weaknesses

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
3

### Summary
The paper investigates the impact of training strategies on Multimodal Large Language Models (MLLMs) when integrating textual detection information from vision models. While current methods often utilize a training-free approach, the researchers systematically explore the effects of adaptive training, retraining, and fine-tuning strategies. Their findings indicate that fine-tuning significantly enhances the MLLMs' performance—improving results compared to the training-free method across various benchmarks. Additionally, fine-tuning enables MLLMs to retain performance benefits even after replacing detection models, suggesting better comprehension of the specialized textual information.

### Strengths
- The paper is well-written and easy to follow.
- Through extensive empirical validation, the study rigorously evaluates the performance of various training strategies across different experimental settings.

### Weaknesses
- Limited Contribution
    - The primary findings of this study demonstrate limited novelty in their conclusions. The superiority of fine-tuning over training-free methods has been well-established in the literature, making this result somewhat predictable. Furthermore, the inclusion of comparison with retraining from scratch adds limited value, as it is rarely considered a preferable option in practice.
- Ambiguities in Dataset Construction
    - The proportional distribution of various data types, including textual detection information needs to be more adequately specified. Moreover, the paper's use of the term "infusion" lacks a precise definition, leaving uncertainty about whether it refers to data addition or conversion processes. The paper's ambiguous description of data processing methods is problematic, especially since data conversion, if implemented, would reduce conventional question-answer pairs and potentially affect benchmark performance, particularly in the retraining strategy.

### Questions
- How is the textual detection instruction data infused during training? (See weakness)

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
4

### Summary
The paper  investigates the impact of various training strategies on the multimodal large language models' (MLLMs) ability to utilize infused detection information. The authors propose three training strategies—training-free infusion, retraining, and fine-tuning—for incorporating detection data in textual format. Through extensive experimentation across benchmarks, they conclude that fine-tuning yields the best results, enhancing MLLM performance in tasks requiring fine-grained image recognition by up to 6.71% over training-free methods. The study also explores model adaptability to different detection models, suggesting fine-tuning as a robust approach for integrating detection information.

### Strengths
1. The paper addresses a crucial aspect of MLLMs' limitations—difficulty in interpreting detailed visual elements. By exploring methods to effectively integrate detection information, it has significant implications for real-world applications where precision in visual recognition is essential, such as autonomous driving, medical imaging, and other fields that rely on high-detail visual data.
2. The authors conduct a wide-ranging analysis across ten well-regarded benchmarks, providing robust evidence for the effectiveness of each training strategy. 
3. A key strength is the demonstration of fine-tuning’s adaptability when incorporating different detection models. The authors showcase that fine-tuned models retain performance gains even when switching from closed-set to open-set detectors, underscoring fine-tuning as a resilient strategy for enhancing MLLMs.
4. The findings from comparing training-free, retraining, and fine-tuning strategies offer valuable empirical insights. By quantitatively showing the superiority of fine-tuning, the paper guides future work on the practical application of training strategies for MLLMs that require fine-grained detail recognition.

### Weaknesses
1. Fine-tuning MLLMs with detection information likely introduces computational overhead, which is not sufficiently addressed. An analysis of training costs and memory requirements across the three strategies would provide valuable insights into the feasibility of each approach for large-scale applications.
2. While the paper includes multiple benchmarks focused on fine-grained visual tasks, the evaluation could benefit from additional benchmarks that test broader language-vision capabilities. Tasks like DocumentVQA.
3. The paper does not examine how variations in detection model accuracy (e.g., OCR quality) impact the MLLM’s performance. Given that the approach depends on external detection outputs, this vulnerability could lead to inconsistent performance if detection quality fluctuates across different scenarios or datasets.

### Questions
1. Could you elaborate on the computational requirements for each training strategy, particularly the memory and time costs associated with fine-tuning compared to the training-free approach?
2. Is there a risk of the model overfitting to the textual detection information during fine-tuning? Has the paper examined the impact of fine-tuning on tasks unrelated to detection, to confirm that broader language comprehension capabilities are maintained?

### Soundness
3

### Presentation
3

### Contribution
3
