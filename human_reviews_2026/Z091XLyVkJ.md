# Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
Fine-grained perception of multimodal information is critical for advancing human–AI interaction. With recent progress in audio–visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and accurately describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent ``co-growth'' between the level of detail and the degree of hallucination in current OLMs. To address this, we propose \textbf{Omni-Detective}, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: \textbf{Audio-Captioner} for audio-only detailed perception, and \textbf{Omni-Captioner} for audio–visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design \textbf{Omni-Cloze}, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority and human preference alignment of Omni-Cloze in evaluating such detailed captions. All the data pipeline, models, and the benchmark are open-source to facilitate further research for omni detailed perception.\footnote{\url{https://github.com/ddlBoJack/Omni-Captioner}}

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Omni-Detective, an agentic data generation pipeline that uses iterative Query-Observation cycles to produce high-quality, low-hallucination multimodal captions. Based on this data, the authors train Audio-Captioner and Omni-Captioner models for audio and audio-visual detailed perception, respectively, and propose Omni-Cloze, a novel cloze-style benchmark for evaluation. Experimental results show that Omni-Captioner outperforms existing open-source models on benchmarks like VDC and video-SALMONN 2, achieving a strong balance between detail and hallucination.

### Strengths
# Strengths:
(+) Innovative data generation：Omni-Detective’s iterative approach with tool-calling effectively reduces hallucination, providing a novel solution for generating high-quality multimodal data.

(+) Superior model performance：Omni-Captioner sets new state-of-the-art results on VDC and achieves the best detail-hallucination trade-off on video-SALMONN 2, surpassing most open-source models.

(+) Robust evaluation framework：Omni-Cloze offers a comprehensive, efficient, and reliable cloze-style evaluation across audio, visual, and audio-visual modalities.

### Weaknesses
# Weakness:
(-) The reliance on a single detective module, while effective, lacks the depth of insight offered by multi-agent systems, potentially limiting its innovation in the generative AI field.

(-) The paper lacks experimental data on the computational cost per iteration, leaving the method’s feasibility in resource-constrained environments unclear.

(-) The paper does not specify how to assess the detective agent’s performance, raising concerns that poor agent output could result in longer, meaningless text rather than valuable information.

### Questions
# Questions:
1. Can Omni-Detective maintain low hallucination rates during multi-round iterations in highly complex scenarios?
2. Are there plans to evaluate Omni-Captioner’s performance on devices with limited computational resources?
3. How can specific metrics be designed to quantify the quality of information extracted by the detective agent in each round?
4. Is there a mechanism to correct errors if the detective agent misidentifies information and mitigate subsequent impacts?
5. Can the Omni-Cloze benchmark be extended to evaluate longer videos or more complex multimodal interactions?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes Omni-Detective, an agent-based data generation pipeline that automatically produces highly detailed multimodal data with minimal hallucination by integrating tool-calling. Specifically, Omni-Detective interacts with diverse tools, including MLLMs, OCR, and ASR, to extract information from videos and summarize detailed perceptions. Using this information, the LLM integrates all perceptions to generate detailed captions, which can later be used for question–answer tasks. Two captioning models, Audio-Captioner and Omni-Captioner, are trained as part of Omni-Detective.
To further validate detailed perception capabilities, the authors introduce Omni-Cloze, a cloze-style evaluation dataset for detailed audio, visual, and audio-visual captioning. Experimental results show that Omni-Detective outperforms existing methods on both existing benchmarks and the Omni-Cloze dataset.

### Strengths
- Designing Omni-Detective is a valid and practical approach to fully utilize the perceptual capabilities of existing models for detailed captioning.
- Introducing the Omni-Cloze evaluation dataset for detailed captioning is valuable to the community. It enables cloze-style evaluation, addressing current benchmark limitations that require multiple LLM calls for assessment.

### Weaknesses
- Several important details are missing. For instance:
  - Which dataset is used for Figure 2 and how are hallucinations detected in long captions?
  - Which model serves as the detective agent in Figure 3?
  - What metrics are used in Table 3?
  - Which datasets are used for joint training and audio-only training?
  - How is the detail rate evaluated in L406?

- Although AudioCaps or Clotho are not specifically designed for detailed captioning, they can still be used to evaluate audio captioning capabilities. However, they are not included in the experiments.

- In Figure 4, the Omni-Cloze samples appear to include audio-only, visual-only, and audio-visual QAs. In Table 4, for the audio-only model, was the evaluation conducted only on audio-only QAs within Omni-Cloze?

- Error analysis of the data generation pipeline is missing. Even with human validation, such analysis is essential to assess the robustness and reliability of the generated dataset.

- For training Omni-Captioner (and Audio-Captioner), is full finetuning reallly effective? Why not LoRA?

### Questions
### Questions and Suggested Experiments

- Although AudioCaps and Clotho are not targeted for detailed captioning, evaluating on these datasets would help assess the model’s audio captioning capability.

- Since the data generation pipeline involves automation, conducting error analysis after human verification would clarify its robustness.

- Ablation on each modality to examine whether the audio module is necessary for the omni benchmark will help to understand that the model fully exploit all the modalities, and the benchmark requires all the modalities.

### Minor Questions and Suggestions

- In L606, VGGSound was used for training the audio-captioner. Does VGGSound provide ground-truth captions for training?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Omni-Detective, an agentic data generation pipeline designed to enhance fine-grained multimodal perception in Omni Language Models (OLMs). It produces highly detailed yet low-hallucination audio–visual data and supports training of Audio-Captioner and Omni-Captioner, which achieve performance comparable to closed-source models (ex, Gemini). The authors also propose Omni-Cloze, a new cloze-style benchmark for evaluating detailed multimodal captioning across audio, visual, and audio–visual domains.

### Strengths
- The paper is well written and well structured, clearly presenting the motivation, methodology, and results.

- It achieves competitive or superior performance to closed-source models (e.g., Gemini 2.5 Flash and Pro) across multiple benchmarks.

- The experimental setup, dataset description, and evaluation protocol are presented in exceptional detail, ensuring transparency and reproducibility.

### Weaknesses
There are no major weaknesses identified in the paper. The overall work is solid and well executed; only a few minor points of curiosity or clarification remain, which are addressed in the Questions section below.

### Questions
- In Table 2, why are proprietary models such as Gemini 2.0 and 2.5 not included in the comparison, despite being used as baselines in other sections? Were these results unavailable or intentionally omitted for consistency?

- Since the core models are fine-tuned from existing architectures, what specific aspects of Omni-Detective or Omni-Captioner represent the novel contribution beyond data quality improvements?

- Are there plans to release the generated dataset and benchmark publicly, and if so, will accompanying annotation or tool-calling logs be shared for transparency?

- When conducting these experiments, how sensitive are the results to hyperparameters such as gradient accumulation, number of training epochs, or learning rate? Even with a strong data pipeline like Omni-Detective, it would be valuable to understand how much small tuning variations influence the final outcomes, based on the authors’ experience.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This submission introduces Omni-Detective, an agentic data pipeline for generating highly detailed yet low-hallucination multimodal captions via iterative tool-calling and evidence verification. 
Using this data, the authors train Audio-Captioner (audio-only) followed by Omni-Captioner (audio–visual), and claimed that they achieve state-of-the-art results (but this reviewer got some concerns and this matter should be discussed further). 
To evaluate fine-grained multimodal perception, they also propose Omni-Cloze, a cloze-style benchmark covering audio, visual, and audio–visual tasks, offering stable and efficient assessment with strong alignment to human preference. 
Overall, the work presents a whole stack, including data pipeline, models, and benchmark, for fine-grained, low-hallucination multimodal understanding.

This reviewer saw the contribution and potential of this submission to the community. However, this reviewer found that the comparative evaluations are conducted in unfair ways. This issue may nullify most of the claims made by comparing with the competing methods. This should be further discussed during the rebuttal.

### Strengths
- Conceptual clarity: Clearly articulates the "co-growth" issue between detail and hallucination and provides an actionable framework to mitigate it.

- Clear demonstration of the effectiveness of the Omni-Detective

- Comprehensive evaluation: Omni-Cloze provides stable, low-cost, human-aligned evaluation across modalities.

- Reproducibility: Extensive appendices with prompts, hyperparameters, and validation details enhance transparency.

Overall, the paper structure, motivation, contribution, and analyses are well implemented.

### Weaknesses
- Unfair comparison due to unfair training scheme: The comparison in Table 2 appears not fully fair, as the proposed Omni-Captioner is trained with the additional proposed large-scale, detailed captioning data (≈ 55k audio-only + 15k audio-visual samples) from VGGSound and FineVideo using a two-stage curriculum. In contrast, all baseline results are "obtained from their original papers" (as described in the caption of Table 2), meaning those models were applied without any chance to be fine-tuned with comparable detailed captioning datasets, but evaluated in their own (likely general or zero-shot) settings. Although the authors mentioned the proposed Captioners were applied in a zero-shot way, the zero-shot meanings are different (the proposed method is not fine-tuned on any training set related to the benchmark, but it still has an unfair advantage of fine-tuning on additional large-scale detailed captioning data, while the other competing methods do not).
Thus, the proposed model benefits from substantially more task-specific supervision, making the performance gains partly attributable to data scale rather than modeling improvements. The authors should clarify this mismatch or conduct controlled experiments with at least comparable training datasets, so that Table 2 clearly shows the effectiveness of the proposed dataset and training recipe. If there is no comparable dataset for the audio-only case, the authors should provide some baselines at least in an ablation way.

- Due to the above concern, all the claims made with the comparisons against the competing methods are deemed to be nullified. For example, in Table 3, even if the proposed method is additionally fine-tuned, the performance gaps against the other competing methods seem marginal.

- Scalability: While the design of Omni-Detective seems effective according to Fig. 6, it may be bulky and has a trade-off between the effectiveness and computational costs. The multi-step Omni-Detective pipeline may be computationally expensive for large-scale data generation, but no analysis and discussion have been reported.

### Questions
Please check the weakness for further discussion.

Personally, this reviewer would like the work, but the following critical concern hinders giving a positive opinion. 
However, the unfair comparison is tightly entangled with the overall contribution claims. It's indeed unfortunate. 
Thus, carefully addressing this reviewer's concern would be essential to re-evaluate the submission. Otherwise, the rate would not be changed.

### Soundness
1

### Presentation
4

### Contribution
4
