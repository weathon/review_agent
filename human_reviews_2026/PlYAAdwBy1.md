# OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large Language Models have emerged as powerful tools for interpreting multimodal data (e.g., images, audio, text), often surpassing specialized models. In medicine, they hold particular promise for synthesizing large volumes of clinical information into actionable insights and patient-facing digital health applications. Yet, a major limitation remains their inability to handle time series data. To overcome this gap, we present OpenTSLM, a family of Time Series Language Models created by integrating time series as a native modality to pretrained LLMs, enabling natural-language prompting and reasoning over multiple time series of any length. We investigate two architectures that differ in how they model time series. The first, OpenTSLM-SoftPrompt, models time series implicitly by concatenating learnable time series tokens with text tokens via soft prompting. Although parameter-efficient, we hypothesize that explicit time series modeling scales better and outperforms implicit approaches. We thus introduce OpenTSLM-Flamingo, which integrates time series with text via cross-attention. We benchmark both variants with LLaMa and Gemma backbones against baselines that treat time series as text tokens or plots, across a suite of text–time-series reasoning tasks. We introduce three time-series Chain-of-Thought (CoT) datasets: HAR-CoT (human activity recognition), Sleep-CoT (sleep staging), and ECG-QA-CoT (ECG question answering). Across all, OpenTSLM models consistently outperform baselines, reaching 69.9% F1 in sleep staging and 65.4% in HAR, compared to 9.05% and 52.2% for finetuned text-only models. Notably, even 1B-parameter OpenTSLM models surpass GPT-4o (15.47% and 2.95%). OpenTSLM-Flamingo matches OpenTSLM-SoftPrompt in performance and outperforms on longer sequences, while maintaining stable memory requirements. By contrast, SoftPrompt exhibits exponential memory growth with sequence length, requiring ~110 GB compared to ~40 GB VRAM when training on ECG-QA with LLaMA-3B. Expert reviews by clinicians find strong reasoning capabilities and temporal understanding of raw sensor data exhibited by OpenTSLMs on ECG-QA. To facilitate further research, we provide all code, datasets, and models open-source.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work introduces two architectures for medical time series and text multi-modal tasks.

### Strengths
1. The methodology section on the model architecture was quite clear to follow.
2. Including expert evaluation on model responses is valuable and provides useful insight into the model’s reasoning capability.

### Weaknesses
1. The authors propose two model architectures but only compare against vanilla finetuned LLM baselines. Additional baselines would strengthen the study:
-  Since the architectures repurpose VLM-style designs, it would be useful to test encoding time series as images and see how a fine-tuned VLM on the same dataset split compare to explicit time series modeling.
- Existing time series–text alignment methods such as [1](https://arxiv.org/pdf/2412.03104), [2](https://arxiv.org/pdf/2408.07773) should be considered for comparison.
- For repurposed tasks like ECG-QA, it would make sense to compare against task specific models, even if they are not multimodal. If adding textual components does not improve results, the benefit of enabling reasoning becomes unclear.

2. The paper presents two frameworks but does not analyze the motivation behind keeping both. The abstract hypothesizes that Flamingo would outperform SoftPrompt, yet there is no follow-up analysis of this hypothesis, and Table 2 shows SoftPrompt performing better on most datasets.

3. The experimental results would benefit from more fine-grained analysis. For example, in Table 2, LLaMA-1B consistently outperforms LLaMA-3B with softprompt, which is counter-intuitive. It would also help to show how performance scales with model size to understand whether the method is scalable.

4. The three CoT datasets were annotated using ChatGPT, with only limited manual review mentioned. To increase reliability, it would be helpful to provide dataset review statistics, reviewers’ backgrounds, and potentially include studies such as human detectability or deliverability checks for future community use.

### Questions
Please see weaknesses

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
This paper presents OpenTSLM, a family of time-series language models that integrate time-series data as a native modality into large language models for reasoning over multivariate medical text-and-time-series data. The work proposes two architectural variants: OpenTSLM-SoftPrompt, which directly maps time-series segments to learnable tokens, and OpenTSLM-Flamingo, which uses a Perceiver Resampler to compress sequences into fixed-size latent representations. The paper demonstrates strong performance on medical tasks including sleep staging, activity recognition, and ECG interpretation.

The paper is well-organized and clearly written, with high-quality figures and substantial experimental work in the medical domain. However, several significant limitations impact the contribution: insufficient coverage of related work on time-series-text multimodal fusion, questions about the training methodology for chain-of-thought reasoning, narrow evaluation metrics that focus only on accuracy without assessing generation quality, and potential concerns about the data collection methodology.

### Strengths
1. **Clear presentation and organization**: The paper is well-structured and easy to follow, with high-quality figures that effectively illustrate the architectures and experimental results.
2. **Comprehensive medical domain evaluation**: The authors conduct extensive experiments across multiple medical tasks (sleep staging, activity recognition, ECG interpretation) with clinical validation, demonstrating practical relevance.
3. **Dual architecture design**: Providing both SoftPrompt and Flamingo variants addresses different computational constraints and sequence length requirements, offering flexibility for different use cases.
4. **Strong empirical results**: The models achieve competitive performance on medical tasks, with notable improvements over baseline approaches.
5. **Open-source contribution**: Releasing code, datasets, and model weights promotes reproducibility and future research.

### Weaknesses
1. **Insufficient coverage of related work**: The paper lacks a systematic discussion of existing approaches to time-series-text multimodal fusion. Several relevant works that address similar problems are missing or inadequately discussed:

   - **ITFormer** and similar models that bridge time series and natural language
   - **Time-VLM** and other vision-language models adapted for time-series
   - Recent work on addressing modality gap in time-series-text fusion

   The paper does not clearly articulate how OpenTSLM differs from or advances beyond these existing solutions. Given that the problem of modality gap in time-series-text fusion has been addressed through various approaches, the technical contribution needs better positioning relative to this body of work.
2. **Training methodology for chain-of-thought reasoning**: The paper uses supervised fine-tuning (SFT) for training chain-of-thought (CoT) reasoning capabilities. However, there are open questions about whether SFT alone is sufficient:

   - CoT reasoning often requires learning complex reasoning patterns that may benefit from reinforcement learning (RL) approaches, as demonstrated in recent LLM training (e.g., RLHF, RLAIF)
   - The paper does not provide ablation studies comparing SFT-only training with methods incorporating RL
   - No discussion of why RL was not considered or whether it would improve reasoning quality

   This is particularly important for medical applications where reasoning quality and interpretability are critical.
3. **Limited evaluation metrics**: The evaluation focuses primarily on accuracy for structured outputs (multiple-choice questions) but provides limited assessment of the quality of generated text, especially for chain-of-thought reasoning:

   - No metrics for evaluating the quality of generated CoT explanations (fluency, coherence, medical correctness)
   - No human evaluation of explanation quality
   - No automated metrics for text generation (BLEU, ROUGE, semantic similarity, medical knowledge correctness)
   - The paper emphasizes "human-interpretable reasoning outputs" but does not evaluate interpretability

   For a paper focused on reasoning and explanation generation, this is a significant gap.
4. **Data collection methodology concerns**: The paper mentions using images to generate captions and chain-of-thought reasoning for multimodal models, but several questions arise:

   - Is it appropriate to use image-to-text generation for creating training data for time-series models? This seems like a domain mismatch
   - How do captions and CoT explanations generated from images transfer to time-series data?
   - No ablation or analysis showing the impact of this data collection strategy
   - Potential concerns about data quality and relevance
5. **Missing technical details**:

   - Insufficient analysis of when to use SoftPrompt vs. Flamingo (decision criteria)
   - Limited ablation studies on key components (Perceiver Resampler compression ratio, cross-attention mechanisms)
   - No analysis of information loss in the Flamingo compression process

### Questions
* **Related work positioning**: Please provide a systematic comparison with existing time-series-text multimodal approaches, particularly ITFormer, Time-VLM, and other recent works. How does OpenTSLM's contribution differ from or build upon these methods? What specific limitations do existing approaches have that OpenTSLM addresses?
* **Training methodology**: Why was reinforcement learning not considered for training chain-of-thought reasoning? Recent work has shown that RL can significantly improve reasoning quality in LLMs. Please provide:
  - Ablation study comparing SFT-only vs. SFT+RL training
  - Discussion of trade-offs and rationale for the chosen approach
  - Analysis of reasoning quality with current training method
* **Evaluation of generation quality**: The paper should include comprehensive evaluation of generated CoT explanations:
  - Add human evaluation of explanation quality (medical correctness, clarity, completeness)
  - Include automated metrics: BLEU, ROUGE, semantic similarity, domain-specific metrics for medical reasoning
  - Compare generated explanations with expert-written ones
  - Assess the correlation between explanation quality and task performance
* **Data collection methodology**: Please clarify and justify the approach of using images to generate captions and CoT for time-series training data:
  - Provide ablation study showing the contribution of image-generated data
  - Analyze potential domain mismatch issues
  - Consider alternatives such as expert-written explanations or synthetic time-series-specific generation
  - Discuss data quality and filtering processes
* **Architecture selection criteria**: Provide clear guidance on when to use SoftPrompt vs. Flamingo:
  - Decision criteria based on sequence length, available resources, task type
  - Comparative analysis of trade-offs
  - Recommendation framework for practitioners
* **Technical ablations**: Include ablation studies on:
  - Perceiver Resampler compression ratio selection (why 64×N?)
  - Impact of different cross-attention mechanisms
  - Information loss analysis in the compression process

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents two new approaches for training multimodal models for health applications. The resulting models take time series and language inputs, and produce a text output containing a time series classification and chain of thought reasoning. Evaluations are done on health datasets that the authors augment with chain of thought reasoning, and include expert evaluation by clinicians.

### Strengths
- The proposed design makes sense for the task and is a well-motivated use of LLMs that also appropriately integrates time series data without over-relying on pre-trained LLM capabilities.
- Prior works in this area have not emphasized generating and evaluating reasoning results, so this work provides novel contributions in that direction.
- The expert evaluation of model outputs provides a clinically relevant view of the model's utility.
- The presentation of the model is generally clear and limitations are discussed.

### Weaknesses
- Unimodal time series classification methods are not evaluated in Section 4. I would hope to see somewhat similar accuracies, otherwise the practical applicability of this method would be limited.
- Other time series language models from the literature are not evaluated in Section 4.

**Minor**

- InstrucTime appears to be a relevant related model to cite and discuss: https://dl.acm.org/doi/10.1145/3701551.3703499 https://arxiv.org/abs/2403.12371

### Questions
- What would be the advantage of the proposed method over using a unimodal time series classification model and then generating a rationale for the output using an LLM in the same way you did to create your CoT datasets? It's not guaranteed that your model's reasoning accurately reflects the internal computations that led to the classification output.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes two approaches to model time series and text jointly. Both the approaches are adopted from the image domain: soft prompting and multimodal fusion based on cross-attention, initially proposed in Flamingo. The paper also introduces 3 datasets, which comprises of existing public datasets along with chain of thought rationales. The authors show that none of the approaches perform uniformly the best across all the compared datasets.

### Strengths
The paper addresses an important and understudied problem, and provides open-source datasets and code to facilitate research in this direction.

### Weaknesses
1. ** Incorrect claims:** The paper makes multiple incorrect claims, subtly. For example, the paper proposes "time-series language models", however their proposed model are trained and evaluated on the same datasets. Language models are widely known to generalize across domains, and therefore I believe the title misleads the reader.

> Prior work has primarily used soft prompting, encoding time series as learned token embeddings concatenated with text tokens.
This is incorrect. Prior work [1] has considered non-prompting based approaches, and have shown value, in the clinical context. 

2. **Some design decisions are not explained:** For example, the authors train their time series encoder from scratch. Why not use a large-scale pre-trained encoder-based time series foundation model (e.g. MOMENT)? 

> we preserve scale and temporal context by adding the original mean, standard deviation, and time scale to the textual description. For example: This is heart-rate data over 24 hours sampled at 50 Hz with mean=61 and std=12.

How is this expected to help, especially when large language models are known to have poor understanding of numbers?

> two synthetic time-series datasets to pretrain the encoder

Why not use existing time series datasets used to pre-train foundation models, e.g. LOTSA [4], Chronos [5] or Time Series Pile [6]? Why do you need time series and text datasets to pre-train the time series encoder? 

3. **Datasets do not inspire confidence:** The proposed models are evaluated on multimodal time series datasets, which are generated using LLMs. However, the authors later show that LLMs do not have a good understanding of time series data, which has also been established by prior work [2]. Were the rationales produced by the LLMs vetted by human experts? I also wonder why the authors did not use existing time series reasoning, captioning and question-answering datasets such as [2], [3] or PTB-XL (as used in [1]). Finally, the proposed datasets on human activity recognition (HAR) and sleep staging are not considered to be within the "medical" domain. 

4. **Proposed methods are not novel:** The proposed soft prompting and Flamingo-based methods are not novel, even with the context of time series & language modeling. I would encourage the authors to pick one method and dive deep into it, while the remaining methods can be baselines. Outside of Flamingo, there are other methods such as LLaVa [7] which have shown promise in the vision language modeling field.  

5. **The evaluation setting tests memorization, not generalization:** The evaluation setting, where the models are trained and evaluated on the same datasets, primarily test memorization of facts, and not generalization. I would encourage the authors to test generalization using held-out datasets. 

6. **Evaluation metrics:** I recommend providing a summary of the evaluation metrics used to evaluate the methods, prior to discussion.

7. ** Baselines:** Without baselines such as statistical and foundation models for classification and forecasting models, it is hard to evaluate the utility of the proposed methods. I would encourage the authors to compare their proposed methods against widely-used task and domain specific and agnostic baselines.


### References
1. Cai, Yifu, et al. "Jolt: Jointly learned representations of language and time-series." Deep Generative Models for Health Workshop NeurIPS 2023. 2023.
2. Cai, Yifu, et al. "Timeseriesexam: A time series understanding exam." arXiv preprint arXiv:2410.14752 (2024).
3. Fons, Elizabeth, et al. "TADACap: Time-series Adaptive Domain-Aware Captioning." Proceedings of the 5th ACM International Conference on AI in Finance. 2024.
4. Woo, Gerald, et al. "Unified training of universal time series forecasting transformers." (2024): 53140.
5. Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).
6. Goswami, Mononito, et al. "Moment: A family of open time-series foundation models." arXiv preprint arXiv:2402.03885 (2024).
7. Huang, Jiaxing, et al. "Visual instruction tuning towards general-purpose multimodal model: A survey." arXiv preprint arXiv:2312.16602 (2023).

### Questions
> Based on ECG-QA Oh et al. (2023), which provides 12-lead 10s ECGs and clinical context, we excluded comparison questions, retaining 42/70 templates.

- Why do you exclude comparison-type questions?

- How are multivariate time series passed to the models, e.g. figure 6(b)?

> Finetuned baselines improve substantially on HAR-CoT (60.44% F1 vs. 0% for Llama-3.2-1B) but only slightly on Sleep-CoT (9.05 vs. 2.14). 
- Do the authors have any hypotheses why this is the case?

> ECG-QA finetuning was infeasible due to high VRAM demands (80k tokens require 100GB per sample).
- Isn't this a limitation of the proposed method?

> Our results show that even frontier LLMs like GPT-4o are poorly suited for time-series reasoning and that time series must be treated as a distinct modality. With OpenTSLM, even small models like Gemma3 270M outperform GPT-4o (∼200B parameters Abacha et al. (2025)) at a fraction of the compute and cost, enabling efficient on-device or mobile deployment.

- How can we rule out this finding as a direct consequence of the fact that the proposed models were fine-tuned on the dataset. Did you test few-shot prompting with frontier LLMs?

### Soundness
2

### Presentation
2

### Contribution
2
