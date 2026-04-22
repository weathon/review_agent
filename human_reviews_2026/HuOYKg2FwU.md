# MagiC: Evaluating Multimodal Cognition Toward Grounded Visual Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Recent advances in large vision-language models have led to impressive performance in visual question answering and multimodal reasoning. However, it remains unclear whether these models genuinely perform grounded visual reasoning or rely on superficial patterns and dataset biases. In this work, we introduce MagiC, a comprehensive benchmark designed to evaluate grounded multimodal cognition—assessing not only answer accuracy but also the quality of step-by-step reasoning and its alignment with relevant visual evidence. Our benchmark includes 5,534 weakly supervised QA examples generated from strong model outputs and 896 human-curated examples with fine-grained annotations, including answers, rationales, and bounding box groundings. We evaluate 15 vision-language models ranging from 7B to 70B+ parameters across four dimensions: final answer correctness, reasoning validity, grounding fidelity, and self-correction ability. MagiC further includes diagnostic settings to probe model robustness under adversarial visual cues and assess their capacity for introspective error correction. We introduce new metrics such as MagiScore and StepSense, and provide comprehensive analyses that reveal key limitations and opportunities in current approaches to grounded visual reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed another VQA benchmark dataset. The idea is to enhance grounded multimodal cognition. A set of evaluation metrics were also proposed with the benchmark. The benchmark was evaluated on 15 VLMs and revealed some interesting failure modes of these models.

### Strengths
- The benchmark dataset is well designed with broad coverage on different dimensions related to grounded vision cognition.
- The proposed metrics align well with the design target.
- The analysis of model failure modes is informative.

### Weaknesses
- There are already many benchmark datasets for VLMs. This one seems to be object-centric with heavy use of bounding boxes. Discussion on the limitations and how it fits into the broader zoo of benchmarks would be helpful.
- The use of LLM judges could introduce biases.

### Questions
It would be helpful to provide side-by-side contrasts with existing benchmark datasets.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper observes that most VQA tasks only evaluate the final answer, while there are few structured evaluations of the reasoning steps themselves. Therefore this paper proposes to enhance part of the GQA dataset with both reasoning steps and visual grounding annotations to be able to explicitly assess these aspects. More specifically, they take 896 questions from the validation set of this benchmark and feed these to four LLMs: InternVL-2.5B, Gema-3-27B. Qwen2.5-VL-7B, and OpenAI GPT4O. This results in reasoning steps which are verified by humans into being True or False. Then the reasoning is corrected in such a way that the reasoning pattern is still valid (e.g. a wrong reasoning step will be followed by a human annotation observing its mistake and correcting it). But I am actually not sure how this would work for multiple wrong reasoning steps in a row. The corrections are used to evaluate the model ability to ‘self-correct’. This is an interesting experiment where they feed the model the question with the reasoning trace up to the mistake, after which they ask the model to complete the rest. But it is also slightly flawed in that it measures an off-policy situation which may be easier to correct for a model. 
In addition, they use existing ground truth bounding boxes and divide them into ‘relevant’ and ‘irrelevant’ for the question using a saliency map. The saliency map is obtained through a human which solves these questions (previously done by Chen et al. ECCV2020). The idea is to link these boxes to the reasoning steps to understand whether the model ‘looks’ at the correct parts of the image. Unfortunately, it is not properly explained how (L264 mentions a region extraction function without any details).

Results show that accuracy of a model is correlated with correct visual grounding, which seems unsurprising but the grounding itself seems also very implicit. Next, they show that stronger models are better in (off-policy) self-correction, which is also unsurprising. They also mention a StepSense measure which apparently should measure whether a reasoning step is correct, but this measure is never explained so I am not sure how to interpret this.

### Strengths
* It makes sense to evaluate grounding. So collecting reasoning traces makes sense.
* The idea of explicit human corrections to evaluate the self-correction capabilities of a model is fun in theory. However, since it measures correcting off-policy behaviour, it is unclear if it really measures ‘self-healing’ behavior. Models typically tend to be very certain of the reasoning steps which they just produced.

### Weaknesses
* There are existing papers like LLM-as-a-judge [Judging LLM-as-a-judge with MT-bench and Chatbot Arena, NeurIPS’23] which analyze reasoning steps fully automatically. While I do believe that using ground truth should be better, we at least want to know how good such an approach is.
* Gemini 2.5 Pro and GPT-5 yield >85% accuracy on GQA, while GQA is noisy. This suggests that GQA is nearly saturated hence I do not believe that this dataset will be relevant for research much longer. It would be much better to collect reasoning traces for more modern and difficult benchmarks.
* The data collection only consists of collecting reasoning annotations. While it could be valuable, by itself the contribution seems rather limited for a top conference like ICLR.
* The insights obtained through the experiments are not very surprising: better grounding leads to better results for a VQA task which looks at relations between objects. Bigger models are better in correcting mistakes (made by other models).
* StepSense remains undefined while the abstract suggests it is an important part of the paper.
* The error analysis is purely qualitative and therefore not very informative.
* Human saliency maps are generally quite noisy. And since there is no quality control of the relevant and irrelevant boxes, this makes it hard to judge the quality of the data.
* More problematically, it remains undefined how regions are mapped to reasoning traces. The qualitative examples seem to suggest that regions are fed in the input by directly drawing on the image in a set-of-marks fashion, but it could also be just a visualization. This is a crucial detail missing from the paper.
* A single extra sentence about what Chen et al. do would prevent requiring the reader to check the AiR-D paper.
* Using LLMs to compare predicted with ground truth responses is not new. Please add some references like [A].
* Not really a weakness, but there is contemporary work in Video QA benchmarks which collects an explicit reasoning trace [Minerva, Nagrani et al. https://arxiv.org/abs/2505.00681]. Please consider citing.
* There is also a portion of synthetically annotated reasoning traces, but these remain unused and it is also unclear whether these would even be useful to the community.
* It would help to have an example of a human correction in L236. Since this is really an important aspect of the paper, it should be clear how these could look like.
* I did not understand the difference between short answer accuracy and long answer accuracy. Especially since the reasoning traces seem to be evaluated in the StepSense metric.
* Table 5.1 is really dense. To make better the point, I would recommend moving the whole table to the appendix. Then I don’t see any significant differences between Full and Short answer correctness, nor between any of the Micro/Macro variants. It would be much easier to present a single scatterplot with Short (or Full) answer correctnes vs the F1 measure of Micro (or Macro) MagiC score.

[A]Jannis Bulian, Christian Buck, Wojciech Gajewski, Benjamin Boerschinger, and Tal Schuster. Tomayto, tomahto.
beyond token-level answer equivalence for question answering evaluation, EMNPL 2022

### Questions
Please focus on the major weaknesses.

Please also note that while my review is very critical, I do think the general direction has a lot of potential and I strongly encourage the authors to keep pursuing this direction. But it will require quite a lot of extra work before it reaches the required level for a top-tier conference.

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
4

### Summary
This paper introduces MagiC, a new benchmark for evaluating grounded multimodal cognition in large vision-language models (LVLMs). The benchmark combines 5,534 weakly supervised and 896 human-curated question-answer pairs, each annotated with step-by-step reasoning traces and bounding-box groundings. MagiC proposes several evaluation axes beyond final answer accuracy, including (i) reasoning validity, (ii) grounding fidelity, and (iii) self-correction ability. It also introduces new metrics MagiScore for region focus, StepSense for reasoning quality, and Self-Heal for introspective correction. 15 LVLMs (ranging from 7B to 90B parameters, including GPT-4o and Qwen-2.5-VL variants) are systematically evaluated. The authors find that models focusing more precisely on relevant image regions tend to produce better answers, and that larger models show stronger self-correction behavior.

### Strengths
* The dataset covers multiple reasoning types with detailed human annotations and saliency-based grounding boxes. The dual-source design (weakly supervised + curated) balances scalability and annotation quality.
* Testing 15 diverse models, including both open and proprietary ones, provides a convincing comparative landscape.
* The addition of StepSense and Self-Heal metrics is creative and useful.
* The paper's main strength lies in its multi dimensional evaluation. The explicit assessment of (1) answer correctness, (2) reasoning validity, (3) visual grounding, and (4) self-correction provides a holistic and insightful view of a model's capabilities than traditional VQA benchmarks.

### Weaknesses
* The evaluation results shows the result for a single run, without any standard deviations.
* The dataset is derived from GQA, which limits its diversity, this may not generalize to broader tasks.
* Many metrics rely on a LLM as-a-judge. Without a human agreement study or cross-model validation, it’s unclear how reliable or unbiased these judgements are.
- Formulation of MagiScore seems a bit weak and ambiguous. No concrete details are given on how the various components of this score are computed and how Precision, Recall and F1 of MagiScore relates to standard models of evaluation metrics like attention (please see "Questions" section)
- The formulation of StepSense is not discussed.
- The paper introduces a large, 5,534-example weakly-supervised (WS) dataset, but its role in the evaluation is not fully clarified.
- While the benchmark measures grounded reasoning and attention, “cognition” suggests higher-level abstraction and compositional skills that aren’t really tested here. The framing may be slightly overselling the scope.

### Questions
* Section 4.2: What type of prompts are given to the LLM judge?
* Box_q has adversarial boxes as well. So, if the entire reasoning focuses on the adversarial regions, would not the MagiScore still be high?
* What is the region extractor function $\phi$, and how does it map each reasoning step to an index in Box_q ?
* If y_k{i) and y^*{i} is 1, does that mean that the model predicted the exact bounding box as GT or that there is a significant percentage of overlap between these two bounding boxes?
* Section 4.1: IoU measure the overlap between GT and predicted BBoxes. In MagiScore, given a GT BBox and a predicted BBox, how exactly are we checking their similarity?
* In Section 4.3 for Self-heal, how can S_rest = [s_i \dot s_j] semantically match with S_wrong = [s_i \dot s_j ] given i < j? This needs more clarification?
-  How do you define Stepsense?
- Section 5.2: How does the F1 Score of MagiScore relate to attention? Line 371 claims “GEMMA 3 12B, which scored 67.7% in attention” which corresponds to F1 score of MagiScore(micro) is Table 2.
- Could a verbose or generic reasoning chain inflate StepSense?

### Soundness
2

### Presentation
2

### Contribution
2
