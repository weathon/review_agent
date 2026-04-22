# WorldEdit: Towards Open-World Image Editing with a Knowledge-Informed Benchmark

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent advances in image editing models have demonstrated remarkable capabilities in executing explicit instructions, such as attribute manipulation, style transfer, and pose synthesis.    However, these models often face challenges when dealing with implicit editing instructions, which describe the cause of a visual change without explicitly detailing the resulting outcome.    These limitations arise because existing models rely on uniform editing strategies that are not equipped to handle the complex world knowledge and reasoning required for implicit instructions. To address this gap, we introduce WorldEdit, a dataset specifically designed to enable world-driven image editing. WorldEdit consists of high-quality editing samples, guided by paraphrased instructions that align with real-world causal logic. 
Furthermore, we provide WorldEdit-Test for evaluating the existing model's performance on causal editing scenarios. With WorldEdit, we use a two-stage training framework for fine-tuning models like Bagel, integrating with a causal verification reward. Our results show that the proposed dataset and methods significantly narrow the gap with GPT-4o and Nano-Banana, demonstrating competitive performance not only in instruction following but also in knowledge plausibility, where many open-source systems typically struggle.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper designs a dataset called WorldEdit, which is designed to enable world-driven image editing. It boasts high-quality images and more realistic text. They also provide WorldEdit-Test as an evaluation benchmark. When it comes to the training framework, they use a fine-tune framework with a causal verification reward. Experiment results prove the effectiveness of the proposed method.

### Strengths
1. This paper is clearly organized and easy to understand. 
2. Constructing a dataset, along with a benchmark, is beneficial to the community and future research. 
3. The experiment results are abundant, with clear analysis. The baseline methods cover the mainstream methods. All the results are listed in a very clear organization. So I think the experiment section is good, and the results are convincing. 
4. Some figures in this paper helps a lot, for example, Figure 4 shows comparison results of a lot of methods in one figure very clearly.

### Weaknesses
1. As it is a dataset work, I think the authors can have more discussions about the limitation or future work, so that the research community can further expand the current dataset. 
2. When it comes to expanding the current dataset, the authors can have discussions about the budget or computational costs of data generation and collection.

### Questions
Since this is majorly a dataset work, I think the authors can have more discussions about the problems I mention in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of implicit editing instructions in image manipulation by introducing WorldEdit, a dataset specifically designed for world-driven image editing. The authors identify that existing models struggle with instructions that describe the cause of visual changes without specifying the outcome, due to their reliance on uniform editing strategies. The proposed solution includes a curated dataset of 11k high-quality samples generated through instruction decomposition and multi-step editing, rigorously filtered for world knowledge consistency. Using this dataset, a two-stage training framework fine-tunes models like Bagel: first through supervised fine-tuning with paraphrased instructions, then via reinforcement learning with a causal verification reward. The approach demonstrates state-of-the-art performance among open-source models, significantly bridging the gap with leading commercial systems like GPT-4o in knowledge plausibility and causal reasoning.

### Strengths
- The paper presents a novel world knowledge editing method and demonstrates state-of-the-art results with convincing showcased outcomes.
- The authors provide a new dataset along with a corresponding test set, and employ Flow-GRPO to train the Bagel model, reflecting substantial research effort.
- The paper is logically structured and clearly written, with effective comparative results.

### Weaknesses
See the "Question" part.

### Questions
From the perspectives of dataset collection methodology, training strategy, the base models used, and the generated results, I find it difficult to identify significant flaws in the proposed approach. I believe the paper warrants acceptance, but I acknowledge that my view might be partial. Therefore, I am providing a conservatively initial score and am willing to adjust it based on subsequent discussions.

Additionally, several prior works have explored world knowledge editing. The authors may consider citing the following relevant studies:

  [1] Unireal: Universal image generation and editing via learning real-world dynamics \
  [2] Editworld: Simulating world dynamics for instruction-following image editing \
  [3] Reasonpix2pix: instruction reasoning dataset for advanced image editing

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces WorldEdit, a dataset and benchmark targeting implicit, cause-driven image edits that require world knowledge and physical reasoning. The authors (i) curate 11k editing pairs via an automated pipeline (1. object detection, 2. instruction generation/paraphrase, 3. multi-step editing, 4. pre/post filtering for causal plausibility and visual quality; (ii) define WorldEdit-Test with ten causal categories (environment- and mechanics-driven); and (iii) finetune a base editor (based on BAGEL) using a two-stage scheme: SFT on paraphrased CoT instructions and RL (Flow-GRPO) with three rewards (reasoning, image quality, causal verification). Evaluations (automatic and human) show competitive results vs. commercial systems (GPT-4o, Nano-Banana) and SOTA among open-source models on knowledge plausibility and instruction following.

### Strengths
1. The problem of implicit, world-knowledge image editing is a timely research problem and opens new possibilities for image editing models.
2. The proposed dataset design contains comprehensive editing categories (Time, Temperature, Humidity, Acidity, Light, Break, Inflate, Squeeze, Twist, Stretch).

### Weaknesses
1. Both the dataset construction pipeline and the benchmark evaluation pipeline rely on proprietary models: Data synthesis and filtering are based on GPT-4o, and model evaluation is based on Qwen-VL-Max. This makes the method difficult to scale up, especially when constructing training datasets, where all the training samples are generated from GPT-4o.
2. Following weakness 1, since the evaluation pipeline is based on Qwen-VL-Max, there could be potential bias from the model during evaluation. Although there are some human evaluation results, these results only cover a limited set of models. I would recommend conducting a more detailed human-VLM agreement and inter-rater agreement study to validate the effectiveness of the automatic evaluation pipeline.
3. The method is only validated on WorldEdit-Test, which is an in-domain benchmark with respect to the training set (both training data and evaluation data are generated by GPT-4o). Therefore, it is naturally expected that WorldEdit performs the best among open-source models. There should be more evaluation results on out-of-domain evaluation benchmarks (such as KRIS-Bench and RISEBench as mentioned in the paper) to further validate the effectiveness of WorldEdit.
4. The proposed method (dataset + two-stage training pipeline) has only been experimented on one model (BAGEL), without further validation on more candidate models.

### Questions
See weaknesses above

### Soundness
2

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
4

### Summary
The paper aims to address the limitations of existing image editing models in processing implicit instructions (which describe the cause of change rather than explicit visual outcomes) and their lack of world knowledge reasoning, introduces the WorldEdit dataset comprising 11,000 high-quality editing samples across 10 causal transformation types, and proposes a two-stage training framework based on the Bagel model (supervised fine-tuning followed by reinforcement learning, integrating triple rewards of CoT reasoning, image fidelity, and causal verification), achieving state-of-the-art performance among open-source models and significantly narrowing the gap with commercial models such as GPT-4o in instruction following and knowledge grounding.

### Strengths
### originality
1. The paper primarily constructs a dataset designed to elicit existing models' capabilities for world knowledge understanding and generation.
2. It employs supervised fine-tuning (SFT) and reinforcement learning with multiple reward signals to train models, thereby validating the effectiveness of the proposed dataset.

### quality
1. The data construction pipeline is rigorous and reliable. The collection, rewriting, construction, and generation of instructions undergo mature filtering mechanisms, substantially ensuring data quality, which is critical for model training.
2. The baseline comparisons are comprehensive. The four evaluation criteria—visual consistency, visual quality, instruction following, and knowledge plausibility—are well-justified and, combined with human evaluation, demonstrate reasonable results.
3. The progressive framework from supervised fine-tuning with Chain-of-Thought (CoT) reasoning to reinforcement learning with composite rewards is well-motivated, and ablation studies confirm the contribution of each component.
### clarity
The paper is well-structured and clearly written.
### significance
This work demonstrates that high-quality data fine-tuning can effectively leverage existing models' prior knowledge to elicit world knowledge understanding and generation capabilities, holding both research and practical value.

### Weaknesses
1. From a visual inspection perspective, the color tone appears to inherit characteristics from GPT-4o. For instance, in Fig. 8, the pizza and instant noodles exhibit noticeably intensified color saturation. The model may learn to mimic GPT-4o's output distribution, raising concerns about the actual physical accuracy.

2. A discussion of how dataset scale affects model performance and generalization would add significant value to the work.

3. The paper lacks object-level metrics to verify whether the target object was actually modified (rather than background elements), whether non-target objects remained unchanged, and whether the transformation was appropriately spatially localized.

4. The paper does not discuss whether baselines were fine-tuned on WorldEdit or tested in a zero-shot setting, which may affect the fairness of comparisons and could enhance the demonstration of the dataset's impact.

5. While the paper mentions extensive filtering operations, it provides limited quantitative details, such as what percentage of generated pairs were rejected during filtering.

### Questions
1. The paper could benefit from including comparisons and distinctions with existing similar data benchmarks to better highlight its significance, such as ReasonPix2Pix [1] and EditWorld [2].

2. Table 1 demonstrates good quality, but what is the scale of the test set?

3. While the model can now handle implicit instructions, does it maintain editing capabilities for explicit instructions? For example, object removal and style transfer also align with world knowledge.

4. Regarding reproducibility, the paper does not mention whether code or data will be released, nor does it clarify the computational resource requirements for training.

5. Minor Points
- The paper mentions "11,000 high-quality editing samples" but could more explicitly state upfront how these are split between training and test sets.

[1] Reasonpix2pix: instruction reasoning dataset for advanced image editing
[2] EditWorld: Simulating World Dynamics for Instruction-Following Image Editing

### Soundness
3

### Presentation
3

### Contribution
3
