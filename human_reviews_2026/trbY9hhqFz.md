# SpurLens: Automatic Detection of Spurious Cues in Multimodal LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Unimodal vision models are known to rely on spurious correlations, but it remains unclear to what extent Multimodal Large Language Models (MLLMs) exhibit similar biases despite language supervision. In this paper, we investigate spurious bias in MLLMs and introduce SpurLens, a pipeline that leverages GPT-4 and open-set object detectors to automatically identify spurious visual cues without human supervision. Our findings reveal that spurious correlations cause two major failure modes in MLLMs: (1) over-reliance on spurious cues for object recognition, where removing these cues reduces accuracy, and (2) object hallucination, where spurious cues amplify the hallucination by over 10x. We investigate various MLLMs and datasets, and validate our findings with multiple robustness checks. Beyond diagnosing these failures, we explore potential mitigation strategies, such as prompt ensembling and reasoning-based prompting, and conduct ablation studies to examine the root causes of spurious bias in MLLMs. By exposing the persistence of spurious correlations, our study calls for more rigorous evaluation methods and mitigation strategies to enhance the reliability of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper extends the ICLR 2025 Workshop paper “SpurLens: Finding Spurious Correlations in Multimodal LLMs.” It presents a framework combining GPT-4 and open-set object detection to automatically discover and quantify spurious visual correlations that affect MLLMs. SpurLens identifies interpretable spurious cues and computes “Spurious Gaps” in perception accuracy and hallucination rate.

### Strengths
1. Spurious correlations remain a genuine and pressing issue in MLLMs, and this work contributes a practical diagnostic framework that could be broadly useful to the community.

2. The paper appropriately frames its findings as diagnostic rather than as mitigation breakthroughs; its conclusion, that prompting alone cannot effectively address spurious correlations in MLLMs, is meaningful.

### Weaknesses
1. The contribution is incremental. The main novelty is the inclusion of more base models and expanded analyses (vision-encoder ablation, prompt variants), but the core method, evaluation setup, and key conclusions are essentially unchanged from the ICLR 2025 workshop version. Many figures, metrics, and formulations (PA, HR, Spurious Gap) are nearly identical to the workshop paper, making this feel like a polished version rather than a new contribution.

2.  Since SpurLens depends heavily GPT-4 outputs, errors or biases in those models may propagate without strong theoretical justification for robustness.

3. The study concludes that CoT and prompting fail to fix spurious bias, but provides no new mitigation direction.

### Questions
1. The paper identifies spurious gaps, but what insights do the authors have into why certain spurious features dominate or how aspects of the training data lead to these biases?

2. Can SpurLens be extended beyond evaluation to support training-time mitigation or data curation, leveraging its diagnostic findings to improve model robustness?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
SpurLens introduces an automatic pipeline for detecting multi-modal spurious correlations in
multi-modal large language models (MLLMs) under object recognition tasks. The study primarily
uses GPT-4 and OWLv2 to extract and rank image features. Experiments on HardImageNet, an
ImageNet subset, and the COCO dataset demonstrate the effectiveness of the extracted
features in increasing perception accuracy (PA) and hallucination rates (HR), indicating
a strong (spurious) correlation between the extracted features and the corresponding object class. Three open-source models and GPT-4o are tested against extracted biases, and all demonstrate significant reliance on bias. Further testing of the proposed pipeline with different backbones
and human verification further supports the method's credibility. Ablation studies of hallucination
with dropped object patches, prompt-based mitigation, and visual spurious cues during
encoding extend the findings of this work.

### Strengths
1. SpurLens focuses on spurious bias in MLLMs —an underexplored field that could
enhance the robustness of MLLM applications.
2. SpurLens is a fully automated pipeline that extracts spurious features that significantly
degrade MLLMs’ detection performance. Such scalability indicates broad potential
applications.
3. Human verification and cross-backbone verification enhance the design's soundness.

### Weaknesses
1. The feature filter is required for the OWLv2 object detector to function. However, the
filtered-out features could be valuable for studying spurious bias, e.g., the backgrounds
in the WaterBirds dataset.
2. The extraction of spurious bias is limited to object detection, whereas MLLMs have a
broader capacity. Modern MLLMs need to handle tasks that require accurate processing
of information beyond objects’ presence or absence, e.g., time, actions, and scenes.
This work's scope is comparably limited in this sense, and would not help the tasks
above as a result.
3. The patch drop method is debatable, as missing patch tokens fail to pass positional
information to the transformer blocks. Zero-filling on the patches could have been a more
robust approach, as it reduces the domain shift that might have contributed to the observations.

### Questions
1. Does the PA gap remain high if the target object is cropped out with OWLv2 and used as input? I am
curious whether the images that contain spurious features depict intrinsically easier
class object instances, contributing to the PA gap.
2. Do HR and PA gaps correlate across target objects? What other insights can we draw from the split of PA and HR compared to their sum, which would also reveal the detection bias?
3. How does your ranking work when there are more than K images with zero scores?

### Soundness
3

### Presentation
3

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
This paper proposes SpurLens, an automatic pipeline to detect spurious correlations in multimodal large language models (MLLMs). The method uses GPT-4 to suggest potential spurious cues associated with object classes and OWLv2 as an open-set detector to quantify their co-occurrence with these objects. Based on this, the framework ranks images by spuriosity and evaluates the perception accuracy (PA) and hallucination rate (HR) differences across the ranking. Experiments on COCO, ImageNet, and HardImageNet show consistent spurious gaps across models including GPT-4o-mini, Qwen2-VL, LLaVA-1.6, and Llama-3.2-Vision. Additional analyses such as patch dropping and prompt-based mitigation highlight that removing spurious cues lowers performance but does not eliminate model dependence on them. The framework is timely, but its contribution is mainly an automation of existing ideas rather than a fundamentally new perspective.

### Strengths
1. The end-to-end framework is well constructed and reproducible, combining large language models with open-set detectors to extract spurious cues automatically.
2. Cross-model and cross-backbone verification strengthens the reliability of the findings.
3. The experiments consistently reveal strong spurious dependencies, especially in models that exhibit hallucination under complex visual contexts.
4. The scalability of the method makes it a useful diagnostic tool for future robustness studies.

### Weaknesses
1. The conceptual novelty is limited, as the paper builds directly on prior works in spurious bias analysis and multimodal robustness [1,2].
2. The OWLv2 detector may itself encode contextual/spurious bias, which could affect the measured spuriosity.
3. The study is limited to object-level recognition and does not address relational or compositional tasks that better reflect multimodal reasoning. There are more spurious correlation such as color or textual biases.
4. The patch-dropping experiment may introduce a domain shift rather than isolating causal visual features.
5. The analysis primarily confirms existing findings without offering a deeper explanation of why these biases emerge.

[1] Moayeri et al., “Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases,” NeurIPS 2023. \
[2] Varma et al., “RAVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models,” ICCV 2024.

### Questions
1. How consistent are the discovered spurious cues when using a different detector such as Grounding-DINO or DETR?
2. Do GPT-4-generated cues tend to favor specific linguistic patterns or cultural biases that could influence the detected correlations?
3. How much overlap exists between cues found by SpurLens and those manually identified in benchmarks such as MM-SpuBench or Waterbirds?
4. What happens if the cue filtering is relaxed, do we recover subtler correlations missed under stricter thresholds?
5. Could causal probing or concept activation vector [3] analysis complement the current ranking-based approach?

[3] B. Kim et al., “Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV),” ICML 2018.

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
4

### Summary
The paper proposes a pipeline designed to produce interpretable spuriosity rankings of images to study spurious bias in Multimodal Large Language Models (MLLMs). The pipeline leverages GPT-4 and open-set object detectors to automatically identify spurious visual cues without human supervision. By applying the pipeline to various MLLMs and datasets, the paper reveals that MLLMs often exhibit over-reliance on spurious cues for object recognition and object hallucination. These findings are validated with multiple robustness checks. The paper also explores potential mitigation strategies, such as prompt ensembling and reasoning-based prompting, and shows that spurious bias in MLLMs cannot be trivially overcome through language-based techniques, calling for more rigorous evaluation methods and mitigation strategies to enhance the reliability of MLLMs.

### Strengths
- The paper studies spurious bias, an important and fundamental problem in MLLMs, and provides detailed and robust analyses on how spurious correlations influence both object recognition and hallucination in MLLMs.

- The proposed Spurious Gap for object recognition and object hallucination is a useful metric to measure an MLLM's reliance on spurious correlations.

- The experiments are comprehensive, covering object recognition and object hallucination as well as ablation studies on visual cues of the target object, prompting strategies for mitigation, and spuriousity in the vision encoder, demonstrating that spurious bias is a fundamental issue in MLLMs.

### Weaknesses
- The paper only analyzes spurious visual cues. In a multimodal setting, spurious correlations may also exist in the text modality. For example, an MLLM may completely ignore the given visual content. I think it is important to emphasize that the paper focuses on spurious visual cues and to provide justifications on why only considering visual features in analyzing spurious bias in MLLMs.

- The spurious bias detection capability of SpurLens is limited to a language model such as GPT-4, as it relies on the language model to generate potential spurious attributes. It may work well for general image datasets where GPT-4 can generate spurious features that indeed commonly appear in images. However, the method may not work well if there is a mismatch between GPT-4 and the dataset being studied, such as a domain-specific dataset. On a similar vein, using an open-set object detector may further limit the spurious bias detection capability.

- The paper assumes that that the image dataset being studied is reasonably large and diverse but does not provide concrete criteria for selecting such datasets. It would be beneficial to provide some practical guidelines to select datasets for effectively analyzing spurious bias in MLLMs.

### Questions
- Why does the paper only study spurious visual cues in MLLMs?
- Can the method generalize to domain-specific datasets?
- What are the guidelines to select datasets for an effective analysis of spurious bias in an MLLM?

### Soundness
3

### Presentation
3

### Contribution
3
