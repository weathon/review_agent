# CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Image captioning is a fundamental task that bridges the visual and linguistic
domains, playing a critical role in pre-training Large Vision-Language Models
(LVLMs). Current state-of-the-art captioning models are typically trained with
Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable
data annotated by humans or proprietary models. This approach often leads to
models that memorize specific ground-truth answers, limiting their generality and
ability to generate diverse, creative descriptions. To overcome the limitation of
SFT, we propose applying the Reinforcement Learning with Verifiable Rewards
(RLVR) paradigm to the open-ended task of image captioning. A primary challenge,
however, is designing an objective reward function for the inherently subjective
nature of what constitutes a "good" caption. We introduce Captioning Reinforce-
ment Learning (CapRL), a novel training framework that redefines caption quality
through its utility: a high-quality caption should enable a non-visual language
model to accurately answer questions about the corresponding image. CapRL
employs a decoupled two-stage pipeline where an LVLM generates a caption, and
the objective reward is derived from the accuracy of a separate, vision-free LLM
answering Multiple-Choice Questions based solely on that caption. As the first
study to apply RLVR to the subjective image captioning task, we demonstrate
that CapRL significantly enhances multiple settings. Pretraining on the CapRL-
5M caption dataset annotated by CapRL-3B results in substantial gains across 12
benchmarks. Moreover, within the Prism Framework for caption quality evaluation,
CapRL achieves performance comparable to Qwen2.5-VL-72B, while exceeding
the baseline by an average margin of 8.4%. Results validate that our CapRL effec-
tively trains models to produce a more general and accurate image descriptions,
moving beyond the limitations of traditional SFT-based image captioning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CapRL (Captioning Reinforcement Learning), a new framework applying Reinforcement Learning with Verifiable Rewards (RLVR) to the open-ended image captioning task. CapRL defines caption quality in terms of utility: a caption is good if it enables a non-visual LLM to correctly answer multiple-choice VQA questions about the original image. The method uses an LVLM to generate a caption, and a vision-free LLM to answer questions based solely on that caption, with the resulting accuracy serving as a verifiable binary reward. Then, Group Relative Policy Optimization (GRPO) is used to train the LVLM with the reward. To support question creation, the authors develop a specific QA curation pipeline to ensure the quality of questions and answers. The authors introduce a large-scale CapRL-5M dataset generated using CapRL-3B, and demonstrate significant improvements across 12 multimodal benchmarks (InfoVQA, ChartQA, SEED, MMStar, etc.), showing performance comparable to much larger models (e.g., Qwen2.5-VL-72B).

### Strengths
The core idea of redefining caption quality through its verifiable downstream utility (i.e., enabling correct VQA responses) is interesting to overcome the long-standing reward-design challenge in open-ended language generation.

The paper is well organized and readable. The methodology is clearly structured, with detailed explanations.

The dataset construction (CapRL-5M) is beneficial to the community if the author makes it open-source.

### Weaknesses
The two-stage pipeline (multiple QA samples per image, multiple-choice shuffling, etc.) introduces heavy computational overhead.

The reward signal depends entirely on a single non-visual LLM’s comprehension and question-answering ability. If that LLM is weak, biased, or miscalibrated, CapRL may learn suboptimal behaviors.

### Questions
How does CapRL perform when the reward LLM is replaced with models of varying strengths (weaker, stronger models)?

What is the actual computational cost of each stage in the CapRL pipeline? 

How sensitive is CapRL to the accuracy of QA filtering? What happens if weakly grounded questions are included?

How does CapRL handle creative or abstract images that do not have factual QAs (e.g., artwork, memes)?

Do the authors open CapRL-5M to the community?

### Soundness
2

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
4

### Summary
This paper introduces CapRL, a framework for synthesizing richer captions to enhance VLM training. Their contributions are two-fold. 1. They incorporate the RLVR paradigm to train CapRL-3B, where the reward is defined by the average score of an LLM answering questions based on the generated captions. 2. They utilize the RL-trained model to produce an additional 3M image-caption pairs, which are combined with ShareGPT4V-1M and DenseFusion-1M to form the CapRL-5M dataset. Experiments show that pretraining on this curated dataset yields consistent performance gains over existing pretraining datasets under the same model architecture.

### Strengths
1. The idea of combining RLVR + data synthesis is novel and effective, as demonstrated in the results of Table 1.

2. As mentioned in appendix E, they employ a data filtering pipeline to alleviate the data-leakage issue in their pretraining corpus.

3. The proposed synthesis pipeline is scalable with the model scale being only 3B.

### Weaknesses
1. While the paper claims that CapRL reduces hallucination, no quantitative evidence is provided for either CapRL-3B or models pretrained on CapRL-1M/5M. The authors are encouraged to report hallucination metrics such as POPE [1] or CHAIR [2] to support this claim.

2. The experiments in Table 3 do not appear to provide a fully fair comparison. The evaluation relies on text-only captions to generate responses, whereas only CapRL has been RL-tuned specifically to produce detailed captions, giving it an inherent advantage. In real-world applications, models are typically used in an end-to-end manner rather than in such a disjoint caption–response setup. Therefore, weaker performance under this prism evaluation does not necessarily indicate poorer downstream capability. I understand the authors’ intention to demonstrate that higher-quality captions can improve downstream QA performance, but a more direct and convincing evaluation would be to apply CapRL as a post-training stage to a VLM and then compare it with other models on these benchmarks under standard end-to-end settings.

3. In the QA curation process, the authors first use QwenVL-72B to generate QA pairs and then QwenVL-3B to verify that the answers are visually grounded and image-dependent. However, this filtering step may remove examples where the smaller model disagrees with the larger one, potentially biasing the dataset toward easier instances (i.e., those correctly answered by the 3B model) and excluding more challenging or informative pairs that only the 72B model answers correctly.

4. It would be helpful to include captioning benchmarks and quantitative metrics to assess the improvements of CapRL-3B over QwenVL-3B, providing a clearer measure of caption quality.

5. (Minor) The title and abstract may be somewhat misleading, as the paper does not present quantitative analyses of caption quality; instead, the evaluation primarily focuses on downstream task performance when using the constructed datasets for pretraining.

[1] Evaluating Object Hallucination in Large Vision-Language Models, EMNLP 2023.

[2] Object Hallucination in Image Captioning, ACL 2018.

### Questions
Please see the above weakness section.

### Soundness
2

### Presentation
2

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
The paper proposes CapRL, a reinforcement-learning framework for vision-language models that tackles the problem of reward hacking in open-ended image captioning. Instead of letting a reward model or llm judege directly score captions (which the authors show quickly collapses to trivial, short outputs), CapRL splits evaluation into two stages: (1) use a strong VL model to generate multiple vision-grounded MCQs about an image, and (2) judge a caption by asking a (cheaper) LLM to answer those MCQs using only the caption, averaging over shuffled options to reduce bias. This verifiable reward is then used to RL-tune Qwen2.5-VL variants, and the authors report that the resulting models improve across a wide suite of benchmarks.

### Strengths
* The idea of evaluating the generated caption, not in terms of the caption itself, but in terms of how well it provides information for downstream QA seems very interesting and novel. 

* More QA data leading to better performance shows the scalability of the method.

### Weaknesses
[minor] In Figure 1 of introduction, the authors use the term 'Unified Reward Model as a Judge', but its not exactly clear what this is. Is it an open-source reward model trained to evaluate captionings?

* In Tables 1 and 2 the authors report results on 12 LVLM benchmarks (InfoVQA, DocVQA, ChartQA, RealWorldQA, MathVista, SEED-Bench 2 Plus, MME, RW MMB, MMStar, MMVet, AI2D, GQA). Could the authors clarify what metric is being reported for each dataset (e.g., accuracy / exact match / task-specific score)? Since the main task in the paper is captioning, it would help to state explicitly how these VQA-style evaluations are computed.

* When the authors mention Qwen2.5-3B + Qwen2.5-ViT, does this mean Qwen2.5-VL-3B? If not, what is the difference?

* In the proposed 2-stage reward, the caption is judged by whether an LLM can answer MCQs based solely on that caption. Could you clarify how well this proxy aligns with human-oriented caption quality (fluency, succinctness, informativeness)? In other words, is there a risk that the model is being optimized for captions that are good for downstream QA rather than captions that are good for people, and if so, can the authors provide human evals or qualitative examples to show the proxy is realistic?

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

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
This paper proposes the CapRL reframeing the captioning task as “utility for QA.” A caption is good if a vision-free LLM can answer the questions about the image using only that caption. The captioner is trained with RL using this verifiable reward (QA accuracy) in a decoupled two-stage pipeline. They start from a small LVLM (Qwen2.5-VL-3B), optimize it with their RL scheme to get CapRL-3B, and then mass-caption images to create a 5M-caption dataset used in standard three-stage LVLM pretraining.

### Strengths
- Using QA accuracy as a verifiable signal rather than judge models or reference captions is well-motivated and an interesting way to improve caption quality.
- The proposed approach shows a 3B CapRL-trained captioner matching a **72B** VL model in the decoupled QA-from-caption setting.
- The paper provides useful analyses on (i) training-dataset size, (ii) the number of GRPO generations per step, and (iii) the number of QA pairs per image.

### Weaknesses
- Could the method incentivize VQA-specific cues rather than balanced, detailed captions? Please report caption quality directly (standard automatic metrics and a small human study) to show quantitative gains.
- For a fair comparison, evaluate methods on the exact same image set, varying only the training approach. One approach can be evaluating a GRPO-trained VQA model that generates description → reasoning → answer on your QA dataset as a direct comparison.
- The paper measures caption utility only via MCQs. Does the approach transfer to open-ended QA (and related tasks like grounding or summarization)?
- It would be beneficial to include the computation required to generate the QA data, produce captions at scale, and run QA (GPU type/count, GPU-hours, throughput, wall-clock time).
- I wonder if there is any potential bias introduced by using the 72B model to generate the QA dataset used for training.
- Answer-model sensitivity: How much do results change when the answer LLM is swapped (different sizes/models)?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2
