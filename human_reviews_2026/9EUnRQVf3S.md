# DVT-LLaVA: Vision-Language Model Personalization with Disentangled Visual Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Personalizing foundational vision-language models (VLMs) specifically for individual users could enhance user experience when interacting with VLMs. Most existing methods rely on introducing additional trainable tokens and finetuning VLMs to fit the data of users. Despite the demonstrated improvements on some Visual Question Answering (VQA) benchmarks, we reveal that the improvements come mostly from the shortcut approach to memorizing the information from the introduced textual training dataset. The capability of visually understanding the user's target concepts -- key to the VQA tasks -- however remains mostly not improved after finetuning. This is especially true for visual concepts residing in complex backgrounds, as these methods often learn representations with concept-relevant and concept-irrelevant information intertwined. To tackle these issues, we introduce DVT-LLaVA, which learns disentangled visual representations for target concepts by jointly learning the concept-relevant tokens and concept-irrelevant tokens via a crafted vision-text dataset derived from image captions. We further propose to tune the LayerNorm layers to enhance the learning capacity and adopt a text embedding augmentation strategy to mitigate overfitting on the training text-image pairs. In addition, we reveal that the existing evaluation benchmarks in this field are mainly based on multiple-choice questions, which fail to accurately assess model performance in the open-set setting. To remedy this, we establish a new benchmark to evaluate performance on this aspect. Extensive evaluations demonstrate the superiority and versatility of DVT-LLaVA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents DVT-LLaVA, a personalization method for VLMs that learns a disentangled concept representation by jointly training a shared concept-relevant token and per-image concept-irrelevant tokens, while tuning LayerNorm parameters to keep the trainable footprint small. It also injects text-embedding noise to curb shortcut and overfitting. The authors introduce OPBench, an open-ended VQA benchmark for personalized concepts, and report gains over MyVLM, Yo’LLaVA, and MC-LLaVA on recognition and attribute understanding, with reasonable retention of base capabilities. The dataset and setup are sensible and reproducible.

### Strengths
- Interesting, low-overhead adaptation: LayerNorm tuning strikes a good accuracy/params balance for personalization.
- Disentanglement idea: Splitting concept-relevant vs. concept-irrelevant tokens encourages visual grounding rather than prompt-only cues.
- Evaluation shift toward realistic use: OPBench’s open-ended judging better reflects real personalization than pure multiple-choice.
- Data & setup: The data composition and training protocol are reasonable and easy to reproduce.

### Weaknesses
- Limited comparison breadth: Baselines and benchmarks are too few; stronger/newer methods(RAP-MLLM, SLC, etc.) are under-covered.

- No explicit multi-concept setting: Despite comparing to MC-LLaVA, the paper does not test multi-concept interference.

### Questions
- Multi-concept personalization: Please add experiments where multiple learned concepts co-occur (composition/disambiguation), and analyze interference between concept tokens. Any changes needed to the disentanglement design to scale to multiple concepts?

- Benchmark breadth & fairness: Can you evaluate on additional public suites and include more recent, strong baselines—ideally reproduced under your pipeline—or at least standardized, directly comparable metrics?

- Judge reliability: Since the evaluation uses LLM-as-judge, please report judge-model/prompt variance and include a small human study with inter-rater agreement to validate the pipeline.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper conducts research on three core pain points in the personalization of vision-language models (VLMs): existing methods rely on the shortcut of "memorizing textual descriptions" rather than learning visual features, concept-relevant/irrelevant information is easily entangled in complex backgrounds, and multiple-choice evaluation benchmarks cannot accurately measure performance in open-set scenarios. To this end, the authors propose the DVT-LLaVA framework, whose core designs include three parts: 1) Disentangled Visual Representation Learning: by jointly training shared concept-relevant tokens  and image-specific concept-irrelevant tokens , visual features are separated without textual descriptions of target concepts; 2) LayerNorm Tuning Strategy: only fine-tuning the LayerNorm layers in VLMs (accounting for approximately 0.003% of parameters) to enhance the ability to learn complex concepts while avoiding catastrophic forgetting; 3) Text Embedding Augmentation: adding controllable uniform noise to query embeddings to alleviate overfitting. In addition, the authors construct the OPBench open-style benchmark, covering 28 concepts, 335 images, and 3115 open VQA questions, evaluating from four dimensions: coarse recognition, fine recognition, concept attributes, and personalized captioning. Experiments show that DVT-LLaVA achieves an average accuracy of 43.4% on OPBench, significantly outperforming baseline methods such as MyVLM (28.5%) and Yo’LLaVA (27.2%), while maintaining the performance of the pre-trained model on benchmarks like POPE and RealWorldQA, verifying the balance between personalized learning and pre-trained knowledge retention.

### Strengths
1. Accurate and quantitative problem positioning: Instead of only qualitatively criticizing the "text memorization shortcut", it clarifies the performance gap of existing methods in visual tasks (e.g., counting) and textual tasks (e.g., color description) through controlled experiments with/without textual descriptions (Tab.5), providing quantitative basis for subsequent method design.

2. Excellent balance between parameter efficiency and performance: LayerNorm Tuning involves only 0.003% of model parameters (Fig.10), with training time (11min23s) slightly lower than Yo’LLaVA (11min40s). Without textual descriptions, the fine recognition accuracy (48.0%) is more than twice that of Yo’LLaVA (23.2%) (Tab.1), balancing practical deployment and performance advantages.

3. Benchmark design close to real needs: OPBench adopts an open VQA format to avoid the "random guessing" flaw of multiple-choice benchmarks. Its questions cover real user interaction scenarios such as "redundant object description" and "concept interaction" (e.g., "What other items are there besides  in Fig.7"), making it more practically valuable than existing benchmarks.

### Weaknesses
1. Lack of interpretability and robustness in the disentanglement mechanism: The separation of  and  is only qualitatively proven through attention visualization (Fig.8), without using quantitative tools such as Concept Activation Vectors (CAV) or feature mutual information to verify the disentanglement effect. Additionally, it does not test "concept neighbor interference" scenarios (e.g.,  = "red pig piggy bank",  = "red cup"), failing to prove that  will not mistakenly absorb key visual features of  or that the two interact, leaving the core innovation without essential support.

2. Weak generalization and theoretical basis of LayerNorm Tuning: It does not explain the theoretical mechanism by which "LayerNorm Tuning promotes visual learning" (e.g., how mean/variance adjustment affects vision-language feature alignment), only verifying it retroactively through performance improvement, resulting in insufficient theoretical support.

3. Lack of adversarial and practical tests in the evaluation system: It does not design confusing concept pairs (e.g., "apple" vs. "tomato"), which are precisely the core pain points of VLM personalization. Meanwhile, it does not evaluate the model’s performance in multi-concept learning.

### Questions
1. The number of benchmarks used in the paper is insufficient, and the results are not enough to illustrate the effectiveness of the proposed scheme. Additional comparisons with other methods on common benchmarks are needed. (E.g., Yo'Chameleon[1], UniCTokens[2] and MyVLM/ Yo'LlaVA / MC-LLaVA Datasets, if you do these experiments, I will consider raise my score.)

2. In the "concept neighbor interference" scenario ( = "red pig piggy bank",  = "red cup"), what is the feature purity of  (e.g., accuracy of probing and classifying target regions)?

3. On VLMs with different architectures such as Qwen-VL or GPT-4V, what is the performance degradation of LayerNorm Tuning? Is it necessary to adjust the noise coefficient α or the selection of tuned layers?

4. In tests with confusing concept pairs ("apple" vs. "tomato"), how much will DVT-LLaVA’s coarse recognition accuracy decrease? This can also reflect whether  truly learns comprehensive features.

[1] YoChameleon: Personalized Vision and Language Generation, https://arxiv.org/abs/2504.20998

[2] UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens, https://arxiv.org/abs/2505.14671

### Soundness
3

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
3

### Summary
This paper focuses on VLM personalization and proposes a method to enhance the visual learning capability for target concepts in complex background. Specifically, the authors introduce concept-relevant and concept-irrelevant tokens to learn disentangled visual representations for target concepts, enabling finer control over concept-specific features. To mitigate overfitting, they tune the LayerNorm layers and employ a text embedding augmentation strategy. Additionally, they introduce OPBench, a benchmark for evaluation in an open-set setting, which includes an open-style VQA format.

### Strengths
1. The core idea is straightforward and the method is easy-to-follow.
2. The experiments demonstrate notable performance improvements, supported by good visualization results that validate the paper’s claims.

### Weaknesses
1. The study would benefit from comparisons with additional VLM personalization datasets, such as those from Yo’LLaVA and MyVLM.
2. The text embedding augmentation and concept-irrelevant tokens lead to significant accuracy drops in Personalized Caption (Table 2). Could you explain it?

### Questions
1. What methods ensure disentangled visual representations for target concepts in VLMs? Are there losses or regularizers for concept-relevant vs. irrelevant tokens?

2. During inference, what are the VLM's inputs? Are all concept-irrelevant tokens included? 

3. Does the number of irrelevant tokens affect performance? Since this likely varies with background complexity, are there strategies to handle such variability?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of personalizing vision-language models (VLMs) to make the model better recognize and describe that concept in visual-language tasks. Accordingly, this work introduces DVT-LLaVA with disentangled visual tuning, layernorm tuning, and text embedding augmentation. Experiments show the effectiveness of the proposed method.

### Strengths
* This work identifies a realistic and understudied failure mode in personalized VLMs, which over relay on textual memory. 

* The proposed method requires simple yet effective tuning, which could be a low-cost solution compared with existing fine-tuning methods. 

* This work introduce a benchmark, which should be supported if the benchmark can be properly released to the academy.

### Weaknesses
* It seems that there lacks sufficient evidence for the disentanglement. Quantitative results, rather than the qualitative analysis are expected. 

* Although the method avoids explicit textual descriptions of the personalized concept, the caption-derived QA pairs still carry strong linguistic priors (object names, contexts, co-occurrences). Thus, improvements might still stem from language-side memorization rather than purely visual disentanglement.

* OPBench includes only 335 images / 28 concepts / 3115 QAs, which is relatively small, and relies on LLM-as-a-judge (GPT-4o-mini) for scoring. Such evaluation may introduce style or prompt bias.

### Questions
* Is that possible to analyze the effect of decoding parameters (temperature, beam size, length penalty) to confirm that the gain is not merely due to text generation bias.

* It seems that the results show that LayerNorm tuning performs better and avoids overfitting compared to LoRA or MLP tuning, but the reason behind this is unclear.

### Soundness
3

### Presentation
3

### Contribution
3
