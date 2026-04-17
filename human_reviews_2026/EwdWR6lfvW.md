# Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Recent advances in text-to-vision generation excel in visual fidelity but struggle with compositional generalization and semantic alignment. Existing datasets are noisy and weakly compositional, limiting models' understanding of complex scenes, while scalable solutions for dense, high-quality annotations remain a challenge. We introduce **Generate Any Scene**, a data engine that systematically enumerates scene graphs representing the combinatorial array of possible visual scenes. Generate Any Scene dynamically constructs scene graphs of varying complexity from a structured taxonomy of objects, attributes, and relations. Given a sampled scene graph, Generate Any Scene translates it into a caption for text-to-image or text-to-video generation; it also translates it into a set of visual question answers that allow automatic evaluation and reward modeling of semantic alignment. Using Generate Any Scene, we first design a self-improving framework where models iteratively enhance their performance using generated data. SDv1.5 achieves an average ***4%*** improvement over baselines and surpassing fine-tuning on CC3M. Second, we also design a distillation algorithm to transfer specific strengths from proprietary models to their open-source counterparts. Using fewer than 800 synthetic captions, we fine-tune SDv1.5 and achieve a ***10%*** increase in TIFA score on compositional and hard concept generation. Third, we create a reward model to align model generation with semantic accuracy at a low cost. Using GRPO algorithm, we fine-tune SimpleAR-0.5B-SFT and surpass CLIP-based methods by ***+5%*** on DPG-Bench. Finally, we apply these ideas to the downstream task of content moderation where we train models to identify challenging cases by learning from synthetic data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles poor compositional generalization in text-to-vision models using a data engine called GENERATE ANY SCENE. The engine systematically enumerates complex scene graphs from a rich taxonomy and translates them into captions for training and VQA pairs for reward modeling. This synthetic data enables model self-improvement, targeted distillation of proprietary model abilities , and low-cost reward functions.

### Strengths
1. The paper demonstrates originality by proposing a programmatic data engine, GENERATE ANY SCENE, for systematically synthesizing scene graphs and translating them into captions and accompanying VQA pairs. This method of tightly coupling data generation (captions) with automated evaluation (QA pairs) is novel, bypassing the reliance on LLM-generated prompts or expensive manual annotations.
2. The method provides high-quality experimental support through four comprehensive application studies (self-improvement, capability distillation, reward modeling, content moderation). The experimental results are strong; for example, the self-improvement framework surpasses fine-tuning on real CC3M data, and targeted distillation using fewer than 800 samples improves the TIFA score by 10%.
3. The paper is well-structured and easy to follow. The five-step process of GENERATE ANY SCENE (from structure enumeration to QA pair conversion) is clearly articulated in Section 2.1. The architectural overview in Figure 1 effectively communicates the proposed framework.
4. This work is significant as it provides a scalable, low-cost, and highly controllable data generation solution for the text-to-vision field. Its applications (especially low-cost reward modeling and targeted capability distillation) offer efficient and practical pathways for improving and evaluating generative models.

### Weaknesses
1. The paper mentions a "commonsense plausibility filtering" in Step 1, but lacks a detailed discussion of how this mechanism is implemented. It is unclear how it balances plausibility with compositional diversity.
2. The system relies on external metadata sources (like WordNet), which fundamentally limits its output scope and carries the risk of inheriting and amplifying inherent social biases from these sources, which the authors do not discuss.

### Questions
1. Could you elaborate on the implementation mechanism of the "commonsense plausibility filtering" in Step 1? Is it rule-based or model-based?
2. In the iterative self-improving framework in Section 3, the model learns from its own high-scoring outputs. After multiple rounds of iteration, is there a risk of mode collapse or a reduction in diversity?
3. Regarding the capability distillation in Section 4, if DALL-E 3 also fails to correctly generate a complex synthetic caption, is this "failed" sample still valuable for fine-tuning SDv1.5?
4. Given the system's reliance on metadata sources like WordNet, have the authors analyzed the risk of amplifying inherent social biases in the synthetic data?
5. Why is your programmatic enumeration approach fundamentally superior to using a well-prompted SOTA LLM to generate both the scene graphs and the diverse captions?
6. Why is your "flat," systematically enumerated data distribution superior to the "natural" distribution of real-world data, and does it risk harming model performance on common, real-world prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a data engine that can generate scene graphs based on libraries of objects, attributes and relations. The proposed engine also generates detailed captions and visual question-answer pairs for each scene graph. The paper further introduces several applications of the proposed data engine, including self-improving text-to-image generation, model capability transfer, reinforcement finetuning, and content moderation. Experimental results show that proposed engine can improve performance of baseline models.

### Strengths
1. The paper is well written and easy to follow. Multiple tasks are tackled with the proposed approach, demonstrating the broad applicability.

2. Extensive experiments are performed, trying to support the claims of the paper.

### Weaknesses
1. It is understandable that the proposed data engine can generate faithfully compositional captions. However, it is not convincing that such an engine can improve the compositional generation performance of a text-to-image generative model. Once a text-to-image generative model is trained, the distribution is there. When data engine and VQA score are used to select self-generated images and finetune the model, it is just biasing the model to overfit region of high VQA score. Then of course, the finetuned model will have better VQA score than baseline model. Is there any other metric that can demonstrate compositionality? Will such overfitting destroy the diversity of generated samples? How about the sample quality of the finetuned model? In other words, are you trading generation quality for VQA score? 

2. If the approach faithfully improves compositionality, then generalization performance would be enhanced at the same time. For example, the finetuning data contains combinations of a set of objects and attributes, the model can generate samples containing unseen combination of the same set of objects and attributes. Is this the case?

3. Is there any reason why some tasks use TIFA score, while others use VQA score?

4. In the main text, it mentions multiple times that more details are provided in the Appendix, which, however, is not attached. Especially, Section 7 give some conclusions, but no results are provided.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces "Generate Any Scene", a data engine that systematically generates diverse and compositional scene graphs from a large taxonomy of objects, attributes, and relations. These scene graphs are translated into textual captions for text-to-image/video/3D generation and into exhaustive visual question-answering (VQA) pairs for automatic evaluation. The authors demonstrate the system's utility across four applications: iterative self-improvement of generative models, targeted capability distillation from proprietary to open-source models, reinforcement learning with a scene-graph-based reward function, and enhanced AI-generated content detection. The core claim is that this structured, programmatic approach to synthetic data generation overcomes the limitations of noisy, weakly compositional web-crawled datasets, leading to significant improvements in model performance and semantic alignment.

### Strengths
1. The paper identifies a clear and significant problem in text-to-vision generation: the lack of structured, compositionally rich training data. The proposed solution using a programmatic scene graph engine to systematically enumerate the visual space is well-motivated.

2. The paper provides extensive experimental validation across multiple tasks (text-to-image, video, 3D), models, and benchmarks (DPG-Bench, GenEval, GenAI-Bench). 

3. A significant strength is the demonstration of the system's versatility across four distinct applications (self-improvement, distillation, RLHF, content moderation). This shows the proposed engine is a general-purpose tool for advancing the field.

4. The programmatic generation of both captions and exhaustive QA pairs for evaluation/reward modeling is highly scalable and avoids the costs and biases associated with LLMs or manual annotation.

### Weaknesses
1. I am concerned that randomly sampling metadata (objects, attributes, relations) may result in implausible or nonsensical scenes (e.g., "crispy dog holding a rabbit reciting a poem"). I am not sure if he "commonsense plausibility filtering" mentioned in Section 2.1 addresses this problem, it is not clearly described, making it difficult to assess its effectiveness. For scalable, fully automated operation, this remains a potential source of noise and inefficiency, as the T2I model and evaluator would waste resources on unrealistic prompts.

2. The method's effectiveness, particularly in the self-improvement loop, is heavily dependent on the initial T2I model's generation efficiency and is computationally expensive. It requires generating multiple images per caption (e.g., 8 in the paper) and running VQA evaluation on all of them to select the best. This process of generation, storage, and evaluation consumes significant resources and time, which may limit its practicality.

3. The process for identifying "hard concepts" for distillation (Section 4) is a great idea, but its description is somewhat vague. A more detailed explanation of how concepts are selected from the taxonomy and how the performance gap is quantified would strengthen this part of the methodology.

### Questions
please see weaknesses

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
3

### Summary
This paper introduces “Generate Any Scene”, a data generation pipeline to sample scene graphs from a set of ~29k objects, ~1.5k attributes, ~10k relations, and ~2k scenes. Generate Any Scene also generates corresponding captions and QA pairs from sampled scene graphs in a programmatic way. In the paper, the authors incorporate the approach into a model self-improvement pipeline where graphs and captions are first sampled, and images / videos are then synthetically generated from those captions (using SDv1.5 for images and Text2Video-Zero for video). This synthesized data is then used for LoRA fine-tuning. The authors compare their approach to CC3M as a baseline and evaluate on text-to-vision generation benchmarks including GenAI-Bench. The authors also experiment with incorporating the proprietary DaLL-E 3 model into the pipeline as a more powerful model to distill from. Additionally, the paper experiments with GRPO in an RL context to fine-tune SimpleAR-0.5B-SFT using captions generated by the pipeline, and Qwen2.5-VL-3B as the reward evaluator. Finally, the paper explores the usage of Generate Any Scene in the context of content moderation.

### Strengths
This paper presents a useful data generation pipeline based on scene graphs that can be incorporated into model self-improvement pipelines for image and video generation. The paper provides extensive experimental results and the proposed approach is technically sound. Given the reported results this approach can be useful for researchers working across this field.

### Weaknesses
My main concerns with this paper are as follows:
* Technical novelty: the paper contributes very little technical novelty as its main contribution is the provision of components for scene graph sampling. It would have been interesting to see (at least a discussion of) different, more advanced sampling methods from the data repository to further improve downstream task performance.
* Limitations: the paper does not seem to discuss limitations of their proposed approach. It would be great if additional limitations and directions for future work could be discussed.

### Questions
The paper mentions the usage of commonsense plausibility filtering to prune implausible contents while retaining compositional diversity. Can the authors elaborate on how this works in detail?

### Soundness
3

### Presentation
3

### Contribution
3
