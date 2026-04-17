# v-HUB: A Visual-Centric Humor Understanding Benchmark for Video LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 8

## Abstract
AI models capable of comprehending humor hold real-world promise—for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel visual-centric video humor understanding benchmark. v-HUB comprises a curated collection of minimally verbal short videos, sourced from classic silent films and online resources, and reflecting real-world scenarios where humor can be appreciated purely through visual cues. Each video clip is paired with rich annotations, including captions, descriptions, and explanations, supporting evaluation tasks like caption matching and humor explanation. To broaden its applicability, we further construct an open-ended video QA task, making it readily integrable into existing video understanding benchmarks. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. For example, all models exhibit a marked performance drop on caption matching when moving from text-based to video-based evaluation (without audio). Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the informativeness of sound and the
promise of integrating richer modalities for complex video understanding tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces v-HUB, a visual-centric benchmark for evaluating multimodal large language models (MLLMs) on video humor understanding, addressing the gap in existing benchmarks that either rely on spoken language or require both video and linguistic cues to comprehend humor. The key findings of the paper include: MLLMs heavily rely on linguistic cues, struggle with active humor discovery, benefit slightly from audio + video, and perform worse on historically distant silent films than contemporary videos.

### Strengths
1. The paper performs a comprehensive comparison of existing datasets (e.g., NYCC, MUStARD, SMILE, ExFunTube) in the area of multimodal humor understanding and systematically diagnoses their inherent limitations.
2. This paper designs three complementary tasks: Caption Matching, Humor Explanation, Open-ended QA. They cover diverse cognitive dimensions of humor understanding—from deep video-text alignment to active humor discovery and general video reasoning.

### Weaknesses
1. The dataset scale is relatively limited, with only 900+ videos included. This small sample size may restrict the evaluation breadth. For instance, it fails to fully cover humorous scenarios across diverse cultural backgrounds or visual styles, making it difficult to disentangle whether the observed MLLM underperformance stems from inherent capability gaps or insufficient dataset coverage.
2. Additionally, this dataset may lack of diversity (e.g., different types of humor, different topics of videos).  This lack of transparent cultural diversity validation restricts the generalizability of v-HUB to MLLMs evaluated on cross-cultural humor understanding tasks. 
3. The above limited scale and diversity of historical humor samples undermines the statistical robustness of the claim regarding MLLMs’ poorer comprehension of historical versus contemporary humor, as the conclusion may be biased by small-sample variation rather than genuine model limitations.
4. The paper lacks explicit documentation of the parameter sizes for most evaluated models (e.g., Video-SALMONN-2, MiniCPM-2.6-o) and omits controlled experiments to isolate the impacts of model parameters versus architecture. While a few models (e.g., Qwen2.5-VL-72B) have parameter sizes specified, the absence of comparative tests between models with identical architectures but different parameter scales (or vice versa) makes it impossible to attribute performance differences to either pre-trained knowledge storage gaps (due to parameter size) or architectural design nuances (e.g., visual encoders, multimodal fusion modules, the pre-training method of different model). This ambiguity weakens the depth of analysis into MLLMs’ humor understanding limitations. 
5. The evaluation framework does not include comparative experiments on performance-boosting methods such as few-shot prompting, chain-of-thought (CoT), or fine-tuning. The paper cannot assess whether MLLMs’ visual-centric humor understanding capabilities can be activated or enhanced via these approaches. 
6. While the paper mentions that some qualified annotators conducted three rounds of annotation for each video, it does not report quantitative metrics (e.g., Cohen’s Kappa coefficient, Fleiss’ Kappa) to measure inter-annotator agreement.
7. This work mainly uses traditional NLG automatic metrics to evaluate the ability of LLMs. Personally, I feel that these metrics are no longer sufficient to indicate whether the generated text conforms to human preferences.

### Questions
1.V-HUB only contains 900+ videos and is sourced by a single X account and silent flims from Charlie Chaplin, how do you address the concern that the small and undiverse dataset scale may limit the evaluation breadth, diversity and statistical robustness of the claim about MLLMs’ poorer comprehension of historical versus contemporary humor? What specific strategies could be adopted to expand the dataset to better disentangle model inherent capability gaps from dataset coverage limitations?
2.The paper claims v-HUB is "visual-centric" (99% of videos rely on visual cues) but provides no operationalized threshold for "visual reliance". How were annotators instructed to distinguish "visual as primary" from "visual as supplementary" when audio/text elements existed? Why was this definition not quantified, and how might ambiguity here affect the interpretability of experimental results?
3.Most evaluated models (e.g., Video-SALMONN-2, MiniCPM-2.6-o) lack explicit parameter size documentation, and no controlled experiments were designed to isolate the effects of parameter scale versus architectural design on performance. Why were these critical details and experiments omitted? What experimental design (e.g., comparing models with identical architectures but different parameter sizes) would help attribute performance differences to pre-trained knowledge storage or architectural nuances?
4.Lack of experiments on performance-boosting methods (e.g., few-shot prompting, chain-of-thought, fine-tuning) were included in the evaluation. Why were these approaches not tested, and how does this omission limit insights into MLLMs’ potential for visual-centric humor understanding?

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
This work contributes a novel dataset named v-HUB as a video humor understanding benchmark. It consists of minimally verbal short videos, and labels for three target tasks of caption matching, humor explanation, and open-ended QA. With the new dataset, this work empirically reveals that existing MLLMs suffer from the issue of strong bias toward text over visuals.

### Strengths
1. This work contributes a novel dataset that can evaluate the MLLMs’ ability of humor understanding from visuals. This dataset can be a valuable resource for this line of research to the community.

### Weaknesses
1. The dataset novelty is limited. 
- Table 1 summarizes the novelty of the proposed v-HUB over existing humor video datasets. However, it contains highly exaggerated arguments. 
- Surely, v-Hub is more visual-centric than the other datasets, but it is an overclaimed argument that only v-Hub is O and the others are X, since it is a matter of degree not a binary O/X decision. For example, ExFunTube contains many visual-centric datapoints, although all datapoints would not. 
- Also, another important novelty argument is that v-Hub supports three target tasks. However, once one has humor explanation text labels for humor videos, it is almost automatic to transform their label formats for those of caption matching and QA tasks. Thus, it is not a notable contribution. 

2. The experimental findings are quite predictable with little surprising novelty. 
- It is a well-known issue that current multimodal models (1) are highly biased toward text information over visual one, and (2) often ignore subtle information in the visuals. These phenomena have been observed in almost all multimodal tasks. 
- Likewise, findings summarized in the text in bold in section 4.2-4.3 have little novel perspective on the multimodal task. 

Based on the points in 1-2 (limited novelty), this work may be better fit to a second-tier venue rather than ICLR. 

3. The ethics review is not possible for a reviewer. 
- Often humor videos in social media are highly likely to contain sensitive, offensive, disturbing, or private content. However, this work does not provide reviewers with the means to assess such issues in the dataset, as only four examples in Fig.1 are viewable to a reviewer. 
- The copyright issue on Charlie Chaplin’s Silent Films (one of the two main video sources) is not discussed.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper present a visual-centric video humor understanding benchmark, and give a comprehensive evaluation of many sota MLLMs.

### Strengths
1. The paper collects humor videos from both User-Generated Funny Videos and films, introduces new humor video sources.
2. Compare to existing video humor benchmarks, this paper evaluates a new generation of MLLM models (newer, lager, wider), demonstrating improvements in model capabilities and introducing new challenges in humor comprehension.

### Weaknesses
1. **Novelty**: The humor videoQA data type is already included in MVBench [1], Table 1 Action - Unexpected Action (What unexpected event contributes to the humor in the video?). The authors cited Mvbench in this paper but they didn't aware that their topic is already covered in it.
2. **Repetitive work**: Upon more precise tracing of the MVbench source, this paper's overall conceptual framework **nearly entirely overlaps with the HumorQA subset within FunQA [2]**.  This raises concerns about duplicate publication.
After a more precise comparison (FunQA vs v-HUB):  
i) **Tasks**: Counter-intuitive timestamp - Caption matching, Title generation - Caption matching, Counter-intuitiveness reasoning - Humor explanation, FunQA-MCQA & Dialog subset - Open-ended QA.   
ii) **Datasize**: 1,769' avg 7s vs. 960'.   
iii) **Anno**: both annotate caption, description, explanation by human annotators.  
iv) **Common result**: The models heavily relies on text cues and weak visual reasoning in humor comprehension.   
v) **Eval metrics**: BLEURT, GPT4 vs. BERTScore, METEOR

This is more like a **coincidental repetition of research topics** (humor videoQA). However, give the existing weakness, even this paper introduces new benchmark and evaluation, **the omissions in its literature review are significant**, leading the authors to overestimate the novelty of their work and limiting the paper's potential for further advancement in MLLM humor comprehension.

[1] Li, K, et al. "Mvbench: A comprehensive multi-modal video understanding benchmark." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.  
[2] Xie, B., et al. "Funqa: Towards surprising video comprehension." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
1. Although existing research points are repetitive, authors can still leverage their existing data for in-depth studies, such as model training, ablation experiments involving more modalities, and thorough analysis of human cognitive patterns regarding humor.
2. Finally, **what are your thoughts and explanations regarding the overlap with HumorQA in FunQA**, and how do you plan to enhance  and reconstruct the value of your paper?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces V-HUB, a visual-centric video humor understanding benchmark that aims to address the limitations of prior works which primarily relied on text inclusion for humor comprehension.  
V-HUB emphasizes visual understanding by providing rich multimodal annotations and a diverse set of video types.
It supports multiple tasks, such as humor explanation, caption matching, and open-ended question answering, making it a comprehensive resource for studying humor in videos.  
The benchmark highlights the complex, multimodal nature of humor and the challenges of modeling subjectivity and cultural context in computational humor understanding.

### Strengths
**Well-defined problem statement and novelty:** The paper clearly defines its problem and contributes a visual-centric benchmark for humor understanding, addressing the gap in prior benchmarks that relied on natural language cues.

**High-quality benchmark:** Each video is accompanied by rich annotations, including closed captions, textual descriptions, explanations, and humor-related elements. These extensive annotations enable multifaceted evaluation of humor understanding across different dimensions.

**Support for multiple tasks:** The benchmark is applicable to various tasks related to video and humor understanding, such as humor explanation, caption matching, and open-ended question answering.

**Annotation methodology:** Annotations are collected using a dual-caption strategy, which explicitly addresses the subjectivity of humor by incorporating multiple perspectives.

**Diverse video sources:** V-HUB includes both silent and user-generated videos, spanning combinations of visual, visual+text, visual+audio, and video+text+audio modalities. This diversity ensures broad coverage of humor types and scenarios.

### Weaknesses
**Ambiguity between description and explanation:** The example in Figure 3 shows minimal distinction between a description and an explanation—the main difference being phrases like “which viewers found very humorous.” A deeper analysis is needed to clarify the conceptual and functional differences between these two annotation types to help readers understand the necessity and distinct role of each.

**Limited annotation granularity:** Some videos contain humor concentrated in short segments rather than throughout the entire clip. 
Identifying these time segments could improve humor understanding. 
Also, for videos with multiple independent humorous moments, the benchmark could benefit from segment-level annotations, ensuring that each humor instance is represented distinctly for more accurate analysis and model training.

**Insufficient detail about background knowledge (L421):** The paper states that background knowledge aids humor understanding but does not specify what this knowledge entails.
 Providing a concrete description or examples of such knowledge would enhance interpretability and reader comprehension.

### Questions
1. What do the authors see as the key differences between description and explanation?
 Were there any specific annotation guidelines to ensure they capture distinct aspects of humor?

2. Do all videos contain a single humorous moment, or are there cases with multiple independent humor instances?
 If so, does the annotation process capture these moments individually or collectively?

3. What exactly constitutes the background knowledge mentioned in L421? Could the authors provide examples or clarify how it was incorporated into the evaluation process?

### Soundness
3

### Presentation
4

### Contribution
3
