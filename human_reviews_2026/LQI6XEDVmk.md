# SafetyPairs: Isolating Safety Critical Image Features with Counterfactual Image Generation

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
What exactly makes a particular image unsafe? Systematically differentiating between benign and problematic images is a challenging problem, as subtle changes to an image, such as an insulting gesture or symbol, can drastically alter its safety implications. However, existing image safety datasets are coarse and ambiguous, offering only broad safety labels without isolating the specific features that drive these differences. We introduce SafetyPairs, a scalable framework for generating counterfactual pairs of images, that differ only in the features relevant to the given safety policy, thus flipping their safety label. By leveraging image editing models, we make targeted changes to images that alter their safety labels while leaving safety-irrelevant details unchanged. Using SafetyPairs, we construct a new safety benchmark, which serves as a powerful source of evaluation data that highlights weaknesses in vision-language models' abilities to distinguish between subtly different images. Beyond evaluation, we find our pipeline serves as an effective data augmentation strategy that improves the sample efficiency of training lightweight guard models.  We release a benchmark containing over 3,020 SafetyPair images spanning a diverse taxonomy of 9 safety categories, providing the first systematic resource for studying fine-grained image safety distinctions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes SAFETYPAIRS, a scalable framework that generates counterfactual image pairs differing only in safety-relevant features. Using instruction-based image editing and automated consistency checks with LLMs, the authors construct a dataset of 3,020 images (1,510 pairs) across 9 safety categories. The benchmark allows for systematic evaluation of the guardrail model’s sensitivity to subtle safety cues and can also serve as effective data augmentation for lightweight guard models.

### Strengths
1. The paper introduces an automated counterfactual image generation pipeline that isolates safety-critical visual features via targeted image editing.
2. The paper presents a new benchmark dataset (SAFETYPAIRS) that captures fine-grained safe/unsafe distinctions, which can be a valuable resource for future research.

### Weaknesses
1. The edit success rate is relatively low (23%), which limits scalability and makes it difficult to expand the dataset for large-scale training purposes. Could the authors try more advanced models, like Nano banana, or do more preprocessing before editing the image to improve the success rate?
2. The data source is too narrow—SafetyPairs is entirely built upon the LlavaGuard dataset. The authors should consider incorporating additional unsafe datasets (e.g., Unsafebench) to improve data diversity and make the benchmark more comprehensive and challenging.
3. The paper does not evaluate any guardrail-specific models, such as LlavaGuard, LlamaGuard3, or classifier-based guard models like Q16 or NudeNet. This omission is strange given that SafetyPairs is derived from LlavaGuard.
4. The training discussion is brief, and the experiments are limited to training a simple linear probe model, which reduces the practical impact of the work.

### Questions
1. During evaluation, the authors only consider a binary classification setup. Could the authors provide more detailed metrics or insights into multi-class performance?
2. Have the authors tried to fine-tune VLMs or classifier-based models using the SafetyPairs dataset to assess potential performance improvements?
3. How do the authors handle ambiguous or borderline cases where it is difficult to determine whether an image is safe or unsafe—are such cases manually labeled by multiple labelers or filtered out?

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
This paper mainly constructs a benchmark dataset that consists of image pairs differing in harmful features. Specifically, the authors leverage image editing models to modify safety-related attributes, making the images safe while keeping other safety-irrelevant aspects unchanged. By using this new safety benchmark, the paper sheds light on the weakness of vision-language models in distinguishing between subtly different images. The benchmark dataset can also serve as a data augmentation resource.

### Strengths
- The idea is interesting that SAFETYPAIRS contains a pair of images that differ only in safety-related details. So that it can help clearly see if the VLMs understand the safety-related features.
- New dataset.
- Shed light on the weaknesses of the current VLMs.

### Weaknesses
- In the data construction, unsafe images are real-world samples, while safe images are synthetically generated through editing. This introduces a potential confounding variable: a classifier trained on such data might learn to distinguish between real and synthetic images rather than between unsafe and safe content. For example, in Section 4.3, the authors train linear probe models. Could the authors analyze to disentangle these two effects and demonstrate that the model is not just learning a real vs. fake classifier?

- The discussion on related work could be significantly improved. The paper should provide a more detailed comparison with recent works like LlavaGuard and UnsafeBench (Qu et al., 2025). A deeper analysis of the datasets is needed. What are the key advantages of the proposed image pair structure compared to the datasets in LlavaGuard and UnsafeBench? A more thorough discussion would better highlight the unique contributions of this work. The authors should also discuss the guardrail models proposed in those papers. Even including them as baselines to provide a more comprehensive assessment. There appears to be a factual error. The paper claims it contains "entirely synthetically generated images," but the UnsafeBench paper states it includes both real-world and AI-generated images.

- Another concern lies in the evaluation of generalization. The authors train the linear probe model on the SafetyPairs dataset, which is constructed from edited samples of the LlavaGuard dataset, and then evaluate it again on LlavaGuard. This setup means that both the training and evaluation data come from highly similar distributions. As a result, the reported improvements may only reflect in-distribution performance, rather than demonstrating true out-of-distribution generalization. To convincingly support the claimed generalization ability, the method should be evaluated on a dataset that the model has never encountered during training.

- Overall, while the proposed benchmark effectively reveals the nuanced weaknesses of current vision-language models, it falls short of providing a clear roadmap for how this dataset can be leveraged to substantially advance the state of the art in guardrail models.

### Questions
- Could the authors analyze to disentangle these two effects and demonstrate that the model is not just learning a real vs. fake classifier?
- What are the key advantages of the proposed image pair structure compared to the datasets in LlavaGuard and UnsafeBench?
- What about the out-of-distribution generalization?
- What are the roadmaps for pushing forward current guardrail models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces safetypairs, a dataset of counterfactual image for safety assessment consisting of 1510 image pairs. 
The authors also propose a general data generation framework that can be utilized to generate counterfactual image pairs for safety testing. Given an unsafe image the safetypairs setup leverages VLMs and image editing models to generate a safe counterfactual that remains as close as possible to the original image. This works experiments demonstrate that the SafetyPairs benchmarks remains challenging for models that perform well on non-counterfactual evaluations. Additionally, the authors provide insights into failure mode of VLMs and demonstrate correlations with pair similarity.

### Strengths
- counterfactual pairs for safety assessments mark a valuable contribution that unearth prevalent failure modes in safeguard models despite saturation of existing benchmarks 

- assuming that the benchmark and code will be made publicly available these will prove useful to the community 
- easily builds on top of established safety taxonomies and datasets (here LlavaGuard)
- the data creation framework is easily reproducible. The overall setup is described well and relevant prompts for reproduction are shared. 
- demonstrates meaningful improvements in training safeguards on SafetyPair augmentations 
- analysis provides some interesting details on failure modes and correlations with image pair similarity

### Weaknesses
# Major 

- The paper lacks details on the human verification of edit attempted edit pairs and only states that that this was conducted by the authors. While this is generally acceptable (especially in safety related research), at least the Appendix should include some details on the specific setup of this validation, measurements taken to ensure no confounding or bias by the authors and usual demographic statistics on the involved annotators. 

- Creating synthetic counterfactuals through generative image editing may lead to a distribution shift of the outputted images [1,2]. The paper provides no control experiment to ensure that potential gaps in qualification accuracy for example are due to a mismatch in real vs synthetic images. Additionally, when using this dataset in training there is a clear confounder with all unsafe image being real and all safe ones synthetically manipulated which can serve as a potential confounder leading to Clever Hans behavior in the classifier 

- The evaluation only considers zero-shot VLMs but no dedicated image safety models (LlavaGuard, SHIELDAGENT [3], OpenAI moderation API [4], etc.) 

- at the same time one demonstrated benefit of SafetyPairs is their benefit on training/tuning models but these results are lacking. While the small-scale analysis on classification heads is nice the paper would benefit from applying this training to an actual VLM-based guardrail that was also the focus in the evaluation. 
Further, since the method constructs pairs DPO would be an obvious application that the paper does not explore currently. 

-The paper should provide a datasheet for the newly introduced dataset (https://arxiv.org/abs/1803.09010) to adhere with standardized documentation practices, especially in safety related fields. 

## Minor Comments

- paper is limited to one source taxonomy and unsafe dataset (i.e. LlavaGuard). While the methodology should be easily transferrable to other setups the paper would benefit from results along those lines 

- the Appendix could be extended with further qualitative samples (ideally some randomly drawn once) to also provide better coverage across models and safety categories 

- building safety classifiers w/ linear heads on CLIP representations was first proposed by Q16 and should be cited in Sec. 4.3 [5] (https://arxiv.org/abs/2202.06675)

[1] https://www.semanticscholar.org/paper/Convergence-Dynamics-and-Stabilization-Strategies-Gao-Li/9330b7a20bd171e1a418518c18a21c80630fd8f9

[2] https://www.semanticscholar.org/paper/PRISM%3A-Precision-Recall-Informed-Data-Free-via-He-Wang/83a8776e8f6f8aa711357c4362e166882c8368c2

[3] https://arxiv.org/pdf/2503.22738

[4] https://platform.openai.com/docs/guides/moderation

 [5] https://arxiv.org/abs/2202.06675

### Questions
- Q1) can the authors provide further details on their analyses by types of unsafely? For example considering the used LLavaGuard taxonomy are there certain categories that models perform significantly better or worse in? How do more fine-grained performance evaluations translate to observations on pair similarity and ROC tradeoff?

- Q2) How do counterfactuals effect models trained w/o them? For example consider LlavaGuard models. Given that they were trained on the unsafe version of these images, do models exhibit a stronger bias in missclaifiyng the safe counterfactuals? 

- Q3) Do you have any insights on further scaling counterfactual data curation and using it in large scale training?

- Q4) Do you have any results indicating if SafetyPairs can be used for paired post-training like DPO? Is there an expected mode collapse here w/ one side of the pair always being synthetic?

### Soundness
3

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
3

### Summary
This work proposes to identify visual features that contribute to "unsafe" images by synthesizing counterfactual edits. Specifically, the authors start from a dataset of unsafe images and prompt LLM to generate edit instructions, such that the edited image no longer violates the safety policy. They use an instruction-based image editing model to perform the edits, and a VQA model to verify the output images.
The counterfactual image pairs are used to evaluate LVLMs on detecting unsafe images. Experimental results suggest that this benchmark is more challenging than those using natural images like LlavaGuard, and that most LVLMs struggle to differentiate between safety-critical features between images. Training with counterfactual editing as data augmentation is also found to improve sample efficiency.

### Strengths
- The paper presents a promising data pipeline to perform counterfactual edits on safety features in images. The method appears to be scalable and yields realistic examples that serve as hard negatives for unsafe image detection.
- Experimental results on various VLMs of varying size, both open-source and proprietary. The generated data appears universally challenging for all models tested.
- Even linear probing with a small number of counterfactual samples (<32) can significantly improve guard model performance.

### Weaknesses
- I feel that the work has yet to realize the full potential of counterfactual probing. I would have liked to see a more in-depth experimental analysis, providing insights into questions such as
  - What features mattered the most to ground-truth image safety (by analyzing the edited regions/objects? even obtaining the ROIs for unsafe images would be informative),
  - What features the guard models are the most sensitive to (maybe repeat the data pipeline but only change unimportant features?), or
  - Whether certain spurious correlation exists to mislead the models to consider the counterfactual images as unsafe (even after removing the true unsafe features).

  I believe understanding these problems will be helpful for building guard models with higher precision and minimize unwanted false positives from biased data.

- The data pipeline involves human in the loop for 1) verifying image edits and 2) annotating unsafe images w/ rationale (as in LlavaGuard), both of which crucial for high-quality benchmarking data but limits scaling to larger-scale training (e.g., if one were to incorporate a significant amount of safety data into VLM post-training). It would be interesting to explore whether training with more unverified counterfactual data can also improve guard performance, relying on the automated filtering by the VQA models.

### Questions
I am curious if the original captions $c_p$ are necessary for generating edit instructions, given the GPT 4o model can understand the images natively. Have the authors done any preliminary experiments comparing the edit quality to a simpler pipeline without the captioning step?

### Soundness
3

### Presentation
3

### Contribution
2
