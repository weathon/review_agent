# VideoDirectorGPT: Consistent Multi-Scene Video Generation via LLM-Guided Planning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Although recent text-to-video (T2V) generation methods have seen significant advancements, the majority of these works focus on producing short video clips of a single event with a single background (i.e., single-scene videos). Meanwhile, recent large language models (LLMs) have demonstrated their capability in generating layouts and programs to control downstream visual modules such as image generation models. This prompts an important question: can we leverage the knowledge embedded in these LLMs for temporally consistent long video generation? In this paper, we propose VideoDirectorGPT, a novel framework for consistent multi-scene video generation that uses the knowledge of LLMs for video content planning and grounded video generation. Specifically, given a single text prompt, we first ask our video planner LLM (GPT-4) to expand it into a ‘video plan’, which involves generating the scene descriptions, the entities with their respective layouts, the background for each scene, and consistency groupings of the entities and backgrounds. Next, guided by this output from the video planner, our video generator, named Layout2Vid, has explicit control over spatial layouts and can maintain temporal consistency of entities/backgrounds across multiple scenes, while being trained only with image-level annotations. Our experiments demonstrate that our proposed VideoDirectorGPT framework substantially improves layout and movement control in both single- and multi-scene video generation and can generate multi-scene videos with visual consistency across scenes, while achieving competitive performance with SOTAs in open-domain single-scene text-to-video generation. We also demonstrate that our framework can dynamically control the strength for layout guidance and can also generate videos with user-provided images. We hope our framework can inspire future work on integrating the planning ability of LLMs into consistent long video generation.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
VideoDirectorGPT presents an innovative framework for consistent multi-scene video generation, utilizing the capabilities of GPT-4 for video content planning and scene description. The process begins with a single text prompt, which is expanded by the video planner LLM (GPT-4) into a comprehensive ‘video plan’, detailing scene descriptions, entity layouts, background settings, and consistency groupings. This information guides the Layout2Vid video generator, enabling explicit control over spatial layouts and maintaining temporal consistency across multiple scenes. VideoDirectorGPT demonstrates competitive performance against state-of-the-art models in single-scene text-to-video generation. The framework showcases potential for innovative applications, offers dynamic control features, and supports user interaction, setting a promising precedent for the integration of LLMs in long video generation and laying the groundwork for future advancements in the field.

### Strengths
1. Reasonable Pipeline Design: The framework adeptly utilizes GPT-4 for meticulous video content planning, optimally harnessing the extensive capabilities of this large language model to bring an innovative and groundbreaking approach to the realm of video generation.

2. Comprehensive Experimental Validation: The paper meticulously outlines a thorough and extensive experimental setup, ensuring a robust and all-encompassing evaluation of the framework’s performance and capabilities. It highlights a diverse range of scenarios and use cases, showcasing the framework’s exceptional versatility and its ability to seamlessly adapt to varying contexts.

3. Diverse Visual Illustrations: The paper and accompanying demo are enriched with a wide array of visual examples, vividly demonstrating the framework’s proficiency in generating multi-scene videos with varied themes and settings. Furthermore, the demo uniquely features support for user-provided images, thereby significantly enhancing user interaction and engagement with VideoDirectorGPT.

### Weaknesses
**Lack of Technical Contribution in Layout2Vid:** The Layout2Vid component of the VideoDirectorGPT framework, responsible for the actual video generation, appears to draw heavily from existing image generation work, particularly the Gilgen model which also operates based on layouts. There seems to be a noticeable lack of substantial technical differentiation or advancement in Layout2Vid, raising concerns about the novelty and contribution of this particular component to the field.

**Lack of Environment Consistency:** The VideoDirectorGPT framework, while innovative in its approach to multi-scene video generation, exhibits a notable lack of consistency in environmental elements across different scenes, as prominently seen in the "make caraway cakes" demo example. Although the object (the woman) maintains a consistent appearance throughout, the environment suffers from visible discontinuities, leading to a disjointed, montage-like effect in the resulting video. This issue seems to be a direct consequence of the framework’s heavy reliance on GPT-4 for generating scene descriptions, coupled with potential shortcomings in the Layout2Vid model's capability to translate these textual plans into visually cohesive sequences.

**Minor: Limited Developmental Space:** The director model in VideoDirectorGPT demonstrates a substantial dependency on GPT-4 for generating scene descriptions. This reliance not only makes the system vulnerable to any limitations and performance issues inherent in GPT-4, but it also raises questions about the developmental prospects of the director model. Given that GPT-4 is a pre-trained model with fixed capabilities, improvements in scene planning and description generation may be challenging to achieve without significant advancements in language model technology or a change in approach. Additionally, despite the innovative integration of GPT-4, the current state of long video generation still leaves much to be desired in terms of visual continuity and narrative coherence, indicating a need for further refinement and development.

### Questions
See the weakness above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper elegantly deconstructs the process of video generation into two distinct phases: planning and execution. In the planning stage, GPT-4 takes on the director's role, meticulously crafting scene descriptions, arranging entities with their corresponding layouts, setting the backdrop for each scene, and ensuring consistency among the groupings of entities and backgrounds. Following this intricate planning process, the baton is passed to the video generation model, Layout2Vid. Leveraging the foundation laid by the pre-trained ModelScopeT2V, a Gated Self-Attention module has been fine-tuned, enabling the direct input of text, images, and layout as conditions for manifesting videos. Remarkably, it transcends the boundaries set forth by ModelScopeT2V, excelling in metrics such as accuracy and the spatial placement of generated entities.

### Strengths
1. The task of video generation has been dissected, enabling the application of LLM's knowledge into the realm of video creation.
2. The method of guiding video generation through layouts has been introduced to the task of video creation.
3. The training process of Layout2Vid is notably efficient.

### Weaknesses
1. As per the CLIPSIM indicator in Table 2 and Appendix F, even with the employment of the costly GPT-4 for video planning, it still lags behind Make-A-Video and VideoLDM, and is even outperformed by ModelScopeT2V. This suggests that there may be certain issues with Layout2Vid's model fine-tuning method.
2. As a T2V model, Layout2Vid's serious oversight lies in its failure to compare the FVD metrics with other models on UCF-101, a benchmark commonly used for T2V tasks.
3. While the paper claims its ability to generate long videos, it merely compares model capabilities with ModelScopeT2V within its own VideoDirectorGPT framework. This seems more like a comparison between Layout2Vid and ModelScopeT2V rather than a qualitative or quantitative comparison with other long video generation models such as Phenaki and NUWA-XL.
4. As a generation task, qualitative comparisons are crucial. However, the qualitative comparison in the paper only presents results between Layout2Vid and ModelScopeT2V, which is evidently inadequate.
5. The techniques employed on the Layout2Vid model are primarily based on ModelScopeT2V and GLIDEN, which is not novelty enough.

### Questions
1. The human evaluation results presented in Table 5 leave us wondering about the number of people involved in rating, as well as the fairness and reliability of the process?
2. Do you think that merely fine-tuning the Gated Self-Attention module might substantially diminish the original potent T2V generation capabilities of ModelScopeT2V?
3. The accesibility to GPT-4 is not always possible. If other open-sourced LLMs are used, would they still be able to generate video plans of the same high quality? Moreover, are the same prompts still effective for other LLMs?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents VideoDirectGPT, a novel framework for generating consistent multi-scene videos by leveraging large language models (LLMs) for video content planning and grounded video generation. It expands a text prompt into a 'video plan' using an LLM, enabling explicit control over spatial layouts and ensuring temporal consistency across scenes, achieving improved video quality and movement control.

### Strengths
In a word, the paper proposes a straightforward solution for video generation by leveraging GPT planning capability and pretrained text-to-video model.

### Weaknesses
1. How does the unclip prior affect the video quality and text image alignment?

2. What if representations are not shared?

3. Except for using GPT an the planner, the novelty is quite limited.  In particular, compared with both GLIGEN and ModelScopeT2V, the only contribution is unclip prior?

4. Have the authors tried only finetune gated self-attn layer? What does the performance look like?

5. Does the baseline ModelScopeT2V use the same dataset as the papers uses for fine-tuning?

6. Comparing ModelScopeT2V and the proposed method on the actionbench-direction prompts, the object is actually not moving. Quality is not good as the baseline. And the advantage of text guidance actually comes from layout control, making the method contribution rather limited.

### Questions
Please see above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a two-stage text-to-video generation framework, consisting of video content planning and grounded multi-scene video generation. The first module employs a large language model (LLM), such as GPT-4, to generate a video plan. The second module, trained with image-level layout annotations, generates a consistent multi-scene video given the video plan. The authors conduct various evaluations to demonstrate the effectiveness of their work.

### Strengths
1. The proposed framework achieves high efficiency by not requiring video training data and maintaining good results with 87% of total parameters fixed.
2. The authors develop several novel evaluation methods that provide solid comparisons between the proposed framework and previous works.
3. The framework uses both high-level and low-level conditioning to enable fine-grained control over generated videos.
4. Intuitively and effectively, the framework uses shared features for the same subject across different scenes to ensure multi-scene consistency.

### Weaknesses
### A. Main paper
1. This paper is somewhat too abstract throughout. The introduction is adequate, but I would expect to see more technical content in the following sections. For example, the loss function for the proposed image fine-tuning is not provided. Additionally, there is little evidence to support the correctness of the proposed methods beyond empirical results.
2. The introduction to the datasets is limited. Some datasets provide fine-grained descriptions for each scene, while others only provide a single sentence for an entire video. Furthermore, the authors customize the Pororo-SV dataset by replacing character names with pronouns, but they do not justify this procedure. The lack of a clear explanation of the datasets makes it difficult to understand the task, such as the inputs and outputs for training and testing, and whether a large language model (LLM) is used.
3. An ablation study should be conducted to demonstrate the effectiveness of the LLM. Additionally, if the LLM was used to refine prompts, these prompts should also be given to ModelScopeT2V to enable a fair comparison and provide readers with better insights.
4. Since the consistency should be maintained regardless of the temporal distance between scenes, the authors should consider using the variance of CLIP features of all scenes instead of the average of similarities across adjacent scene pairs.
5. The human evaluation does not have enough participants to provide reliable results.

### B. Qualitative results
1. The objects not exactly follow the bounding boxes.
2. The "pushing object" video examples appear to show camera movement rather than object movement.

### Questions
1. How do you replace the original animation characters in Pororo-SV with real-world entities? Do the edited videos look natural enough? Why do you use pronouns to replace character names, and wouldn't this make it difficult for ModelScopeT2V to guess the correct content, leading to an unfair comparison?
2. What is the exact loss function used for finetuning?
3. Why are some numbers not available in the results, such as FVD and FID for Coref-SV and Consistency for HiREST?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
