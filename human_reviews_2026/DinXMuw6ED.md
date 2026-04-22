# Aya Vision: Advancing the Frontier of Multilingual Multimodality

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 2, 8, 4

## Abstract
Building multimodal language models is fundamentally challenging: requiring alignment of vision and language modalities, curating high-quality instruction data, and preserving existing text-only capabilities once vision is introduced. These difficulties are further magnified in multilingual settings, where the need for multimodal data in different languages exacerbates existing data scarcity, machine translation often distorts meaning, and catastrophic forgetting is more pronounced. To address these issues, we propose: (1) a synthetic annotation framework that curates high-quality, diverse multilingual multimodal instruction data across many languages; (2) a cross-modal model merging technique that mitigates catastrophic forgetting, effectively preserving text-only capabilities while simultaneously enhancing multimodal generative performance. Together, these contributions yield \textbf{Aya Vision}, a family of open-weights multilingual multimodal models (8B and 32B) that achieve \textbf{leading performance across both multimodal and text-only tasks}, outperforming significantly larger models. Our work provides guidance and reusable components for scalable multilingual data curation, robust multimodal training, and advancing meaningful evaluation in multilingual multimodal AI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Aya Vision, a family of open-weight 8B and 32B multilingual multimodal models (MLLMs) designed to address key challenges in non-English vision-language understanding. The authors identify two primary obstacles: (1) the scarcity of high-quality, diverse multilingual data, as simple machine translation often introduces errors and "translationese", and (2) the "catastrophic forgetting" of text-only capabilities that occurs when vision is introduced. To solve these problems, the paper proposes two main contributions: A Multilingual Multimodal Synthetic Annotation Framework and a Cross-Modal Model Merging. The paper also introduces new benchmarks, including Aya VisionBench, for evaluating open-ended multilingual generation

### Strengths
1. The paper’s most significant contribution is the cross-modal model merging technique. It provides an elegant, simple, and training-free solution to the well-documented problem of catastrophic forgetting.
2. The authors provide a thorough evaluation by not only testing multimodal performance but also rigorously assessing text-only capabilities, which is critical given the paper's motivation.

### Weaknesses
1. The paper criticizes machine translation for causing cultural bias and misalignment but still relies on a translation-based pipeline, i.e., machine translation + LLM post-editing. While this improves fluency, it fails to address the core cultural issue—a translated American-centric caption remains American-centric. Although Section P mentions limited image re-rendering, the method itself does not bridge the cultural-context gap it highlights.
2. The data-balance ablation (Figure 8) suggests issues with the quality of the synthesized multilingual data. Performance peaks at 35% multilingual data and drops at 67%, which the paper attributes to upsampling and the need for diverse English data. This implies the synthesized multilingual data is inherently less diverse or useful than English, undermining the claim of a “high-quality, diverse multilingual multimodal instruction data” framework.
3. One of my main concerns is that the paper claims to “set a new standard for multilingual performance,” yet its results do not consistently support this. In Table 1, Aya-Vision-8B scores 46.16 versus Qwen-2.5-VL-7B’s 50.00, and the gap widens at larger scales (Aya-Vision-32B: 52.81 vs. Qwen-2.5-VL-72B: 59.72). Despite claims of being “optimized for open-ended generation,” the model underperforms on structured tasks like VQA (CVQA and MTVQA) and reasoning.

### Questions
Please refer to the Weaknesses.

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
The paper introduces AyaVision, multilingual multimodal LLMs (8B/32B) that handle up to 23 languages. The motivation is to address three challenges: (i) data scarcity and translationese to train multilingual MLLMs, addressed via a synthetic data engine that enriches the annotations with additional context about he images, (ii) text-only capability loss after adding vision, which is addressed via a training-free cross-modal weight-merging method, (iii) lack of open-ended multilingual evaluation, addressed via new benchmarks (AyaVisionBench, m-WildVision and xChatBench). The paper reports strong preference win rates on the proposed open-ended AyaVisionBench.

### Strengths
- The paper makes a meaningful data contribution by using a pipeline that combines distillation, filtering, and context-aware rephrasing to reduce translation artefacts. The data engine also increases lexical diversity and sequence length of instruction pairs, which is quantified by gains and results in an automatic quality improvement on COMET.

- The paper also makes a meaningful benchmark contribution, it proposes three benchmarks: AyaVisionBench,  m-WildVision and xChatBench offer open-ended, multimodal, multilingual evaluation. This extends evaluation beyond multiple-choice or english-only setups and should help standardize evaluation for future multilingual multimodal work.

- The training-free cross-modal merging method is simple and effective, as shown in the ablations. The reported deltas show large recovery on text-only win rates and non-trivial gains on multimodal win rates.

### Weaknesses
- The model shows large gains on the two proposed benchmarks, AyaVisionBench and m-WildVision, measured as open-ended pairwise win rates averaged across 23 languages. However, on standard accuracy-scored benchmarks, its performance is clearly below strong baselines. The paper does not explain why a model that improves under open-ended evaluations falls behind by 6 to 17 points on benchmarks like xMMMU, CVQA, and MTVQA. If this gap comes from answer-format constraints (for example, multiple choice or one-word responses) or a mismatch between free-form outputs and the expected scoring format, the paper should include a simple normalization step that maps free-form text to the required label, for example, an LLM-based post-processing that converts a sentence to option B or to the exact short answer. Without such a comparison, it is hard to tell whether the wins on AyaVisionBench reflect alignment with the response style used in training and evaluation, rather than improvements that transfer to widely used benchmarks.

- The paper mixes several motivations, but the evidence for each one is narrow. The paper claims translationese in training data is a major problem in data annotation methods of existing multilingual MLLM and proposes translation plus LLM rephrasing. Yet it does not show that prior datasets have this issue (quantitatively or qualitatively), nor that the proposed method fixes it beyond a COMET jump. There is no human verification/alignment and no comparison to a simple off-the-shelf LLM translation, so it is unclear whether any benefit comes from the proposed method or from the tools used. The paper also argues that text-only performance worsens when additional languages are added in multilingual multimodal performance and uses linear weight merging, but this relation is not explained, and there is no evidence that higher text-only scores lead to higher multimodal accuracy. Altogether, the motivations feel mixed. 

- The citation format does not match the official ICLR citation requirement. 

Overall, the main technical contribution is not supported by clear evidence, and its novelty is limited. The method is a simple linear weight mix between the text-only LLM and the multimodal model while keeping vision modules fixed. The paper does not examine basic design alternatives or stronger baselines that would test whether this is the right choice, nor does it provide clear guidance on when and why the method helps. At the same time, the performance improvements are only seen in the pairwise LLM-judge win rates on proposed open-ended benchmarks, while scores on widely used benchmarks are lower than the baselines.

### Questions
Please explain why the scores on CVQA, MTVQA, and xMMMU are lower. Is the main cause answer-format mismatch, short-answer matching, domain differences, or lower model accuracy? If it is mostly formatting, please say whether a small LLM post-processing step that maps free-form outputs to the exact expected label would raise the scores, and by roughly how much.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Aya Vision, a family of multilingual multimodal large language models (8B and 32B parameters) designed to handle both image and text inputs across 23 languages. The authors propose two main ideas: **(1)** a synthetic multilingual data generation and translation pipeline, and **(2)** a cross-modal model merging technique to preserve text-only performance after multimodal training. The paper also presents AyaVisionBench, a new benchmark for evaluating multilingual multimodal models. Experimental results show Aya Vision performing reasonably against several existing open-weight models and stronger on its own proposed benchmark.

### Strengths
The following are the strengths of the paper:

1. The paper tackles an important problem of extending multimodal models beyond English, making progress toward inclusive and global AI systems.

2. The effort to maintain text-only capability while introducing visual understanding is valuable, as it tries to address the catastrophic forgetting issue common in multimodal training. The cross-model merging approach is interesting and if works in general, could be very useful for MLLM post training.

### Weaknesses
The following are the weaknesses of the paper:

1. Many important implementation details are missing or unclear. For example, how dataset categories were selected and sampled to reach 2.29M samples (lines 126-128) is not explained. The process of “regularization” and sampling remains undefined.

2. The keyword-based filtering step is unclear. The “curated list of keywords” is never provided, making it unclear how such filtering can reliably detect errors in generated descriptions (refer to lines 156-160).

3. In Stage 2: LLM-based semantic filtering, the LLM-based semantic filtering is a weak proxy since the LLM cannot see images. As stated, complex samples are often discarded, which limits the model’s ability to learn complex reasoning tasks.

4. The hybrid translation pipeline lacks justification. Simply chaining NLLB and command-r-plus does not guarantee better quality, and the paper does not explain why this choice was made over stronger multilingual multimodal translation models such as LLaMA-4 or Qwen3. The subset of data chosen for translation (lines 182–183) is also not described.

5. There seems to be data size inconsistencies.  Synthetic re-annotation (3.5M of 2.29M original) and downsampled original data (3.7M from 6M) do not match the claimed final training set size of 2.75M samples. How is this 2.75M number exactly calculated?

6. The cross-model merging method is interesting but conceptually unclear. Theoretically, the text-only and multimodal models occupy different weight spaces, and the paper does not justify how linear interpolation of weights can work reliably.

7. AyaVisionBench cannot be considered a major contribution. All its details are buried in the appendix, and it is small, only 135 samples per language (around 3K total). The comparison with other multilingual multimodal benchmarks is missing. In fact, stronger public benchmarks like “All Languages Matter (ALM-Bench) [1]” already exist, covering 100 languages and 22K samples.

8. Aya Vision performs poorly on academic and structured benchmarks (Table 1), with models such as Qwen2.5-VL outperforming it consistently. Its strong results appear only on its own benchmark, which weakens the claim of “best-in-class performance.”

9. Many components depend on proprietary LLMs (command-r-plus, GPT-based judges) and claim “human-preferred” outputs without any real human evaluation. For example, as show in Appendix K, simply asking LLM/MLLM to generate "human-preferred" does not mean that the resulting text is "human-preferred".

Overall, the paper lacks clarity, key methodological details, and fair validation. The improvements shown are narrow and not convincingly demonstrated beyond the proposed benchmark.

---

[1] Vayani, A., Dissanayake, D., Watawana, H., Ahsan, N., Sasikumar, N., Thawakar, O., Ademtew, H. B., Hmaiti, Y., Kumar, A., Kuckreja, K., et al. (2024). All languages matter: Evaluating LMMs on culturally diverse 100 languages. CVPR 2025.

### Questions
Please refer to the Weaknesses section for detailed points requiring clarification.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
In this paper, the authors propose Aya Vision, a model family of multilingual multimodal open models. The authors also propose a new framework for synthetic annotation of multilingual multimodal instruction tuning data. The authors also introduce a new benchmark AyaVisionBench, a multilingual multimodal benchmark.

### Strengths
- The authors introduce a strong set of models Aya Vision, achieving state of the art performance on multilingual multimodal benchmarks. 
- The authors performed comprehensive experiments, comparing various different models against Aya Vision, demonstrating the effectiveness of their models.
- The proposed new framework of generating synthetic annotation could help scale up multilingual multimodal training by increasing the amount of data that could be gathered for large scale training.
- The authors constructed a new benchmark AyaVisionBench, which has a broad coverage of language and topics that could comprehensively evaluate multilingual multimodal models on generation quality.

All these resources are very helpful to the multilingual community.

### Weaknesses
- The authors employed LLM as a judge approach for evaluation of win rate, which could be very unreliable. It would be good if they could have alternative evaluation format or perform a small set of manual evaluation to verify the results.
- The authors claim that they proposed a novel multimodal merging technique, yet multimodal merging is commonly discussed in literature. It would be good if the authors could compare their work against existing work that discuss multimodal merging.
- The synthetic data relies on expensive proprietary models. It would be great if the authors could compare with existing open models and demonstrate on par performance.

### Questions
- What are the computational resources required to train the Aya Vision model family?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the important challenge of building multilingual multimodal language models, focusing on data scarcity, translation distortion, and catastrophic forgetting when integrating visual understanding into multilingual LLMs. To tackle these issues, the authors introduce (1) a synthetic annotation framework for high-quality multilingual multimodal data and (2) a cross-modal model merging technique to preserve text-only capabilities while enhancing multimodal generation. The resulting Aya Vision models (8B and 32B) demonstrate strong performance on both text-only and multimodal benchmarks, surpassing larger counterparts.

### Strengths
1. The paper tackles a highly relevant and timely problem. Building multilingual multimodal systems remains a core challenge for the next generation of foundation models, and the proposed dataset construction framework provides valuable insights for the community.

2. The writing is clear, well-structured, and easy to follow. The motivation and contributions are concisely articulated, making the work accessible to a broad audience.

### Weaknesses
Major Concerns

1. Clarity on Key Concepts (Lines 62–63): The terms context-aware and human-preferred are mentioned as central ideas but are not elaborated in Section 2. The paper should explicitly explain how these notions are implemented or reflected in the proposed framework, particularly the human-preferred aspect.

2. Justification for Translation Choice (Lines 171–187): It remains unclear why the NLLB-3.3B model is preferred over GPT-based translation. The paper lacks empirical or qualitative evidence supporting this decision. A brief comparative evaluation or rationale would strengthen the argument.

3. Missing Experimental Analyses:

- The paper notes that Aya-Vision is built upon multilingual post-trained base models. The authors should discuss why this multilingual prior is essential and whether similar results can be achieved by directly fine-tuning strong visual-language models such as Qwen-VL.

- Figure 9’s ablation fails to demonstrate the effectiveness of the proposed context-aware rephrasing step. The authors should include a comparison with models trained on unprocessed data to quantify its benefit.

- There is little discussion on alternative model-merging strategies. Incorporating or at least discussing comparisons with TIES-Merging [1] and DARE [2] would provide deeper insight into the design choices.

Minor Issues

1. The Related Works section should not be confined entirely to the Appendix; key discussions should appear in the main text to improve context and readability.

2. Appendix C merely lists supported languages without justifying their selection. The authors should briefly explain the inclusion criteria or regional balance.

3. Appendix J introduces SafeGuards, but the main text does not reference or evaluate safety performance. A clear link to this appendix and a short discussion in the results section would help readers understand its relevance.

[1] TIES-Merging: Resolving interference when merging models. NeurIPS 2023.

[2] Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. ICML 2024.

### Questions
Please see Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
