# Voila-A: Aligning Vision-Language Models with User's Gaze Attention

- Decision: Reject
- Scores: 5, 6, 3, 6

## Abstract
In recent years, the integration of vision and language understanding has led to significant advancements in artificial intelligence, particularly through Vision-Language Models (VLMs). However, existing VLMs face challenges in handling real-world applications with complex scenes and multiple objects, as well as aligning their focus with the diverse attention patterns of human users. In this paper, we introduce gaze information, feasibly collected by AR or VR devices, as a proxy for human attention to guide VLMs and propose a novel approach, Voila-A, for gaze alignment to enhance the interpretability and effectiveness of these models in real-world applications.
First, we collect hundreds of minutes of gaze data to demonstrate that we can mimic human gaze modalities using localized narratives. We then design an automatic data annotation pipeline utilizing GPT-4 to generate the VOILA-COCO dataset. Additionally, we innovate the Voila Perceiver modules to integrate gaze information into VLMs while preserving their pretrained knowledge.
We evaluate Voila-A using a hold-out validation set and a newly collected VOILA-GAZE Testset, which features real-life scenarios captured with a gaze-tracking device. Our experimental results demonstrate that Voila-A significantly outperforms several baseline models. By aligning model attention with human gaze patterns, Voila-A paves the way for more intuitive, user-centric VLMs and fosters engaging human-AI interaction across a wide range of applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces the concept of incorporating gaze information (referred to as human attention) into VLM models to enhance their performance and potentially improve their interpretability. To support this, the authors collected hundreds of minutes of gaze data and developed an automated data annotation pipeline using GPT-4 to generate the VOIA-COCO dataset.

The paper makes a valuable contribution to the relevant academic community by demonstrating the significance of gaze information for VLM tasks, such as visual question answering. The presented results show that the proposed method is qualitatively and quantitatively superior to baseline models like Otter.

### Strengths
•	The authors introduce the novel concept of utilizing gaze information in the development of VLMs.
•	Unlike baseline and several other studies, the experimental analysis is not limited to qualitative results but also demonstrates quantitative results.

### Weaknesses
1.	The paper's presentation is lacking. Many important sections have been relegated to the appendix, especially the technical details. For example, Section 4.1 is challenging to understand due to the limited text. 2)The model heavily depends on the baseline model Otter. The method of injecting gaze information is quite straightforward. The way in which the authors handle catastrophic forgetting can be observed in the literature, thus not introducing technical novelty.
2.	The experimental analysis appears somewhat unfair because the proposed method uses additional modalities to achieve the same results. Therefore, its better performance is not surprising, particularly in cases where the query does not clearly define the object's name, and several other objects are present in the scene, with the gaze heatmap aligning with the queried object. It is also worth to mention that both baseline methods are still only in ArXiv.
3.	There is uncertainty about the cases in which gaze information was found to be less relevant.
4.	The caption for Figure 3 lacks informativeness.
5.	There exist a few typos to be fixed, e.g., Fiture
6.	Hallucination issue can be better presented qualitatively and better discussed.
7.	Gaze data collection procedure is also very scarse and it is not possible to understand if the annotators have a reliable consensus to use the collected data in model evaluation and comparisons.
8.	It is doubtful whether 100 gaze samples are sufficient for conducting a comprehensive comparative study. I have reservations about the potential bias in the collected dataset.

### Questions
•	Section 4.1 and gaze data annotation should be described in detail. It is not possible to validate the procedures perform in these context.
•	Weakness Q4
•	Pls. comment on Weakness (2) for the technical novelty of the method.
•	Pls. comment on Weakness (3).
•	How the authors evaluate the interpretability? Several places in the paper interpretability was mentioned, however its evaluation is unclear given that such a keyword is being used in several different content of AI.
•	Weakness Q9

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents "Voila-A," a novel approach aimed at aligning Vision-Language Models (VLMs) with user gaze attention. The authors highlight the challenges faced by existing VLMs in handling complex scenes and diverse human attention patterns. They propose utilizing gaze information collected from AR or VR devices as a proxy for human attention to guide VLMs.

The paper provides a thorough explanation of the methodology, including data collection, automatic data annotation using GPT-4, and the design of the Voila Perceiver modules. The authors conduct experiments, comparing Voila-A with baseline models (Otter and Kosmos-2) on both synthesized and real-world gaze datasets.

The results demonstrate the effectiveness of Voila-A, showcasing its balanced capability between helpfulness and fact grounding. The evaluation metrics, including GPT-4 Ranking and Reward Score, support the authors' claims. Additionally, ablation studies and qualitative analyses provide further insights into the model's performance and capabilities.

One notable contribution is the introduction of trace data as a proxy for gaze data, offering a cost-effective and scalable alternative for aligning VLMs with user gaze attention. The method of transforming trace data to mimic gaze data is well-described and substantiated with empirical evidence.

### Strengths
The proposed method for aligning gaze data with trace data proves to be both effective and straightforward. It introduces a fresh approach to integrating gaze information with Vision-Language Models (VLLM). The approach has been rigorously examined through studies, yielding results that substantiate its efficacy.

### Weaknesses
The data collection section (4.1) lacks detailed information on the methodology and content of the dataset. Providing specific examples and clearer explanations would enhance comprehension. Additionally, Figure 3 needs a caption and more in-depth explanations to convey its intended message. The figures also need higher resolution for better readability when printed.

Section 4.2 requires clearer explanations, particularly regarding the parameters X, L, and G. The concept of 'latent data' (L) needs better elucidation. A structured approach, starting with an explanation of the inputs and employed encoders, followed by a deep dive into the new approach, would enhance clarity. A comprehensive figure illustrating how gaze is integrated into visual language models would be beneficial. It's unclear how the 'Voila Perceiver Resampler' module is integrated with the VLLM.

In Section 4.3, the meaning and specifics of 'gaze linear layer' and 'gaze key linear of attention' need clarification. It's not clear which layers these terms refer to or if there's a specific formula involved.

Merging Section 5.1.2 with Section 4.1 would improve the overall clarity of the paper.

The summary of the main results in Figure 5 is not easily understandable. Using a table with percentages might provide clearer insights.

The claim that Voila exhibits a 'superior ability to handle multi-turn real conversations' in the last sentence of Section 5.3 needs stronger support or clarification in the results section.

### Questions
Could you provide a clearer depiction of how the ViolapercieverBlock and Resampler are integrated within the larger VLLM framework? A simplified architectural overview would be immensely helpful in understanding the bigger picture.

It would be beneficial to have more details on what exactly is included in the automatic data annotation process and how it is carried out. Providing specific examples in the main paper would greatly enhance comprehension.

For Figure 4 and 5, additional guidance on how to interpret the results would be appreciated. Specifically, clarification on what constitutes the 'Overall score' and a detailed explanation of how the 'Helpfulness' and 'Grounding score' are calculated would be invaluable.

I would also onsider providing a brief discussion of potential applications and future directions in the conclusion section.
Clarify any specific limitations or potential challenges associated with the proposed approach.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a way to integrate gaze information into large VLMs. It uses mouse trace data as a proxy for gaze track with proper sampling. Such gaze information is then used in an attention mechanism to enhance visual feature perception. The authors report that the proposed approach outperforms baselines.

### Strengths
* Introduces a scalable way to leverage human attention cue in VLM models.

### Weaknesses
* Technical or scientific contribution is very incremental and limited.
* Writing can be improved; not always easy to follow and clear. 
* Baseline methods considered are not comprehensive or fair. Mouse trace data as a proxy for gaze sounds reasonable but there are many off-the-shelf saliency model that are designed to mimic human gaze. Some of the existing saliency model can be used or at least need to be discussed and reviewed in the paper.

### Questions
Please address the comments above regarding baseline.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In AR/VR scenarios, gaze is one of the most natural way to represent the regions interesting to users. This paper studied an interesting problem: suppose we are using vision-language model under AR/VR, how to incorporate human gaze attention into vision-language model and how much improvement can it bring? The paper proposed to use mouse trace to approximate gaze and use the collected gaze heatmap into attention mechanism in vision language models (Otter) while freezing the language encoder MPT-7B and vision encoder CLIP ViT-L/14. The models is evaluated on the collected Voila-COCO data set and a VOILA Gaze data which is more close to real life scenarios. The proposed method with extra gaze information outperforms baselines Otter and Kosmos-2. Ablation study also shows that gaze heatmap is better than alternatives ways to use gaze data like discrete gaze position, gaze bounding box as patch, etc.

### Strengths
--The idea of including gaze information to vision-language model is quite interesting, which might be one important aspect when people use the vision-language model in VR/AR scenarios. The idea of human using gaze/attention in compute vision models is not new, but the idea of using gaze/attention to improve vision-language model is relatively novel to the best of my knowledge.

--Some interesting experiment results are shown to demonstrate that the gaze/attention data are helpful for VQA tasks of vision-language models.

### Weaknesses
--Will the data set VOILA-COCO be released? I did not see this information in the paper. 

--Using mouse trace to approximate human gaze/attention is a popular approach in attention related area, however, the authors does not mention existing works like BubbleView https://bubbleview.namwkim.org/ or Salicon http://salicon.net/

--The organization and presentation of the paper can be improved. It is not clear how the gaze data will be used in vision-language model until section 4.3. Instead, the authors can provide an illustrator figure about it at the beginning.

### Questions
See the weakness part. Especially, the dataset might be an important contribution of this paper. However, it is not clear whether the data set will be released or not.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
