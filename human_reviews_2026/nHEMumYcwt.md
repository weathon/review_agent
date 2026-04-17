# Video Language Models are Human-Aligned Evaluators for Text to Motion Generation

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recently, text-to-motion (T2M) has become a basic setting for human motion generation. This work studies the evaluation of alignment between text and generated motion, to credit the reliable use of T2M models. We consider solving the T2M evaluation task by making use of a video language model (VLM). Our basic idea is: render the generated human motion into a skinned video, and then use a VLM for evaluation. To address information loss problem when 3D motion is rendered into 2D video, we develop a method, which ensures reliable evaluation score by analyzing VLM entropy. Our evaluation method, named VeMo, frees T2M evaluation from reliance on motion data while seamlessly leveraging the semantic understanding and reasoning capabilities of advanced VLMs trained on Internet-scale data. To systematically compare the empirical usefulness of different evaluation methods, we manually annotate a meta-evaluation benchmark that includes coarse-grained alignment labels and fine-grained reasons. Extensive experiments and case studies demonstrate the effectiveness of VeMo.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VeMo, a method for evaluating text-to-motion (T2M) generation by leveraging Video Language Models (VLMs). The core idea is to render 3D generated motions into 2D videos and use a VLM to assess the alignment between the video and the input text prompt. To mitigate information loss from the 3D-to-2D projection, the method employs an entropy-based view selection technique to identify the most informative camera angle for evaluation. Furthermore, the authors contribute a new meta-evaluation benchmark, including human annotations for text-motion alignment, to facilitate a standardized comparison of different evaluation metrics. Experimental results demonstrate that VeMo's scores correlate more strongly with human judgments than existing reference-based and reference-free methods.

### Strengths
1. Important Problem: The paper addresses a critical bottleneck in the T2M field. Developing reliable, automatic evaluation metrics that align with human perception is essential.
2. Intuitive and Effective Method: The core idea of rendering 3D motions into 2D videos to leverage the powerful semantic understanding of pre-trained VLMs is clever and well-motivated. As demonstrated in the main results (Table 3), this approach proves effective, achieving a significantly higher correlation with human judgments across multiple metrics compared to previous automatic evaluation techniques.
3. Valuable Benchmark: The creation of a meta-evaluation benchmark with coarse- and fine-grained human annotations is a strong contribution to the research community. This resource enables a fair and standardized comparison of existing and future T2M evaluation methods, which has been a significant gap in the field. The quality of the benchmark is supported by a high inter-annotator agreement.

### Weaknesses
1. (Major Concern) Computational Overhead: The primary limitation of VeMo is its significant computational expense. The pipeline requires rendering motions into videos, potentially from multiple viewpoints, and then performing inference with a large-scale VLM. This high overhead makes the metric impractical for use as an online reward signal during model training. And for large-scale evaluations, this effectively limits its utility to offline analysis. The paper would be strengthened by a discussion of this limitation and an exploration of potential mitigation strategies, such as distilling the VLM evaluator into a smaller, more efficient scoring model.
2. Lack of Analysis on Number of Views: The paper proposes rendering motion from K different views and selecting the one with the lowest entropy. However, the experiments only compare a human-optimized view against one other randomly generated view. There is no ablation study or discussion on the impact of the number of views (K) on evaluation performance and the trade-off with computational cost. It is unclear how many views are necessary to achieve reliable scores.
3. VLM Score Stability: The paper does not address the potential instability of VLM-generated scores. It is known that even with deterministic decoding settings (e.g., temperature=0), language models can exhibit slight variations in output likelihoods. Since the VLM score is the fundamental output of the metric, an analysis of its variance across multiple runs is necessary to establish the reliability and consistency of VeMo as an evaluator.

### Questions
1. Could the authors provide a more detailed discussion on the computational overhead of the VeMo pipeline? Given the expense of rendering and VLM inference, how do they see its practical utility beyond offline analysis, for example, as a reward signal during training?
2. Following up on the computational cost, have the authors explored any mitigation strategies to create a more efficient version of the metric, such as distilling the VLM evaluator into a smaller, specialized scoring model?
3. The paper proposes selecting from K views based on entropy, but the experiments only seem to compare a human-optimized view against one random view. Could the authors provide an ablation study on the impact of K (the number of views) on evaluation performance? What is the trade-off between the number of views, score reliability, and computational cost?
4. What is the stability or variance of the VLM-generated scores? Could the authors provide an analysis of VeMo's reliability across multiple runs on the same inputs (even with deterministic decoding) to establish the consistency of the metric?

### Soundness
3

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
3

### Summary
Text-to-motion generation is a fundamental task in our community. This work proposes an evaluation model, VeMo, to provide a reliable assessment for the task. The core idea is to render generated human motions as skinned videos and then use a video-language model (VLM) to evaluate how well the generated motion aligns with the given prompt. Experiments show that VeMo is an efficient and effective metric for measuring prompt–motion alignment.

### Strengths
- The paper is well written and easy to follow.

- The proposed evaluation model is solid and well motivated: it leverages a large video–language model to assess how well generated motions align with prompts, rather than relying on reference-motion metrics. 

- Comprehensive experiments demonstrate that the model provides an effective and efficient benchmark for evaluating T2M tasks.

- The ablation study offers sufficient analysis to clarify the design choices.

### Weaknesses
The paper lacks a comprehensive evaluation of current text-to-motion methods.
Can the authors provide a thorough benchmarking of existing approaches so the community has a clear reference for comparison?
Such a benchmark would allow future work to make direct, convenient comparisons.

### Questions
N/A

### Soundness
3

### Presentation
3

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
The paper considers solving the T2M evaluation task by making use of a video language model, and provides a meta-evaluation dataset.

### Strengths
1.The paper addresses a critical limitation in T2M evaluation, over-reliance on scarce motion data by proposing VeMo, a novel framework that repurposes VLMs (trained on internet-scale text-vision data) for text-motion alignment assessment.

2.VeMo provides a plug-and-play solution for T2M evaluation—no motion data or human labels are needed for training, making it easily integrable into existing T2M pipelines.

### Weaknesses
1.The main experiments only use InternVL3-14B as the core VLM, with limited testing of other open-source VLMs (only 4 models in ablation). This fails to verify VeMo’s robustness across VLM architectures (e.g., video-specialized vs. general multimodal models). Additionally, the benchmark only includes motions from HumanML3D and two T2M models—excluding complex motions or newer T2M models,limiting validation of VeMo’s generalizability .

2.VeMo requires rendering 3D motions into multi-view videos (Blender) and running large VLMs. The paper provides no analysis of computational efficiency or optimization strategies for low-resource scenarios, limiting its practical applicability for real-time T2M evaluation. This will lead to reduced usability.

3.There are too few video demos.

4.The headings in the supplementary materials should be numbered as 1., 2. instead of .1, .2.

### Questions
1.The benchmark excludes complex motions and newer T2M models. Could author extend the benchmark to include these, and reevaluate VeMo to confirm its generalizability? 
 
2.Author note VeMo inherits VLM biases but provide no details.

3.VeMo’s computational cost is high. Could you test lightweight VLMs (e.g., InternVL3-8B) or optimize rendering to reduce resources, while quantifying the trade-off between efficiency and evaluation performance?

4.The entropy-based view selection assumes low entropy equals low information loss. Could you provide a deeper explanation for this correlation? Are there edge cases where this strategy fails, and how might you address them?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new evaluation metric for text-to-motion generation by utilizing a pretrained video language model. The idea is to leverage the prior knowledge residing in the VLM and render generated motions in different viewpoints for the VLM to determine its consistency with the text. A meta-evaluation benchmark is proposed to evaluate the effectiveness of the proposed metrics.

### Strengths
- Utilizing VLM as a prior model to evaluate motion generation is interesting and introduces a new perspective for text-to-motion generation evaluation.
- Experimental results and analysis are performed on the benchmark dataset to validate the proposed evaluation method.
- The writing is good and the paper is easy to follow.

### Weaknesses
- The benchmark data is generated by two motion generators, which may contain inductive bias in the motion generator, which may not be the real-world distribution. 
- The evaluation metric has its limitations. Since a human annotator only annotates yes or no options. It cannot reveal the extent of failure cases, e.g., the generated motions perform mostly correctly following the text, while with some small mistakes, or the motion is totally wrong.
- The high entropy scores selection strategy in Eqn. 3 only utilizes the highest score single-view for the decision. However, some complex motions may suffer from the overlapped problem in different sub-sequences from a single view, e.g., performing some motion while turning in a circle at the same time. Any strategies to combine information from different views for a better performance?
- The user study only contains two users, which may contain variation.

### Questions
Null.

### Soundness
3

### Presentation
3

### Contribution
2
