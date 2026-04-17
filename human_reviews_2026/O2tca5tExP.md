# video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We present video-SALMONN 2, a family of audio-visual large language models that set new state-of-the-art (SOTA) results in video description and question answering (QA). Our core contribution is multi-round direct preference optimisation (MrDPO), paired with a caption-quality objective that jointly rewards completeness and factual accuracy. Unlike standard DPO with a fixed reference policy, MrDPO periodically refreshes the reference by bootstrapping from a newly re-initialised lightweight adapter trained on the latest preferences, avoiding reference staleness and enabling continual improvement. This strategy produces captions that are consistently more detailed and accurate than those from proprietary systems such as GPT-4o and Gemini-1.5 Pro. We further distil these gains by using our model to generate a high-quality video-caption corpus for supervised fine-tuning of new models, transferring benefits beyond captioning to strong performance on complex video-QA tasks. Across widely used audio-visual and visual-only understanding benchmarks (including Video-MME, WorldSense, AVUT, Video-Holmes, DailyOmni, MLVU, and LVBench), our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems. Our source code, models, and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Video-SALMONN 2, a family of audio-visual large language models that achieve state-of-the-art results in video captioning and question answering through multi-round direct preference optimization (MrDPO) with a caption-quality objective.
MrDPO continually refreshes training references via lightweight adapters, producing more detailed and accurate captions that transfer effectively to video-QA tasks, surpassing proprietary systems like GPT-4o and Gemini-1.5 Pro.

### Strengths
•	The paper introduces an original multi-round DPO (MrDPO) framework that continuously updates reference models to prevent staleness, leading to more accurate and faithful video captioning.
	•	It demonstrates strong methodological quality and clarity, effectively combining reinforcement-based preference optimization with lightweight adapters for stable and scalable training.
	•	Extensive experiments across diverse benchmarks show consistent SOTA performance in both captioning and video-QA, highlighting the method’s strong generalization and practical impact.

### Weaknesses
See questions.

### Questions
To ensure fair evaluation, the paper should also report results on long-video benchmarks without audio input to distinguish improvements from the proposed training strategy versus those merely due to the additional audio modality.

### Soundness
3

### Presentation
2

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
The paper presents video-SALMONN 2, a family of audio-visual LLMs that attain state-of-the-art captioning and strong video-QA by (i) introducing Multi-round Direct Preference Optimization (MrDPO)—a preference-learning scheme that repeatedly refreshes the DPO reference policy via a re-initialized LoRA proxy and a guided DPO loss mixed with SFT—and (ii) turning the improved captioner into a data generator to re-annotate videos and then SFT new models on these higher-quality captions. Caption quality is measured and rewarded with an atomic-event–based evaluation that decomposes captions into missing vs. hallucinated events. Models at 3B/7B achieve SOTA at comparable scales and the 72B model surpasses all open-source systems and rivals proprietary ones on benchmarks such as Video-MME, WorldSense, AVUT, Video-Holmes, DailyOmni, MLVU, and LVBench.

### Strengths
* **Principled, iterative preference learning for captioning.** MrDPO mitigates reference staleness by **merging the previous LoRA into the backbone and re-initializing a fresh LoRA proxy each round**, stabilized by a guided DPO loss with a small SFT term; ablations show cumulative reductions in caption error.  
* **Task-aligned reward with atomic events.** The **missing + hallucination** metric operationalizes caption *completeness and factuality*, enabling LLM-judged preference pairs for RL. 
* **Strong, broad empirical results and transfer.** The captioner beats strong baselines (incl. GPT-4o visual-only) on a human-validated caption metric and VDC-Detailed, and the **re-captioned data** lifts downstream QA across diverse benchmarks and scales.  
* **Clear architecture for synchronized A/V modeling.** Interleaved audio/visual tokenization with frozen encoders and an aligner cleanly integrates audio without degrading visual performance.

### Weaknesses
* **Evaluator dependency and potential bias.** The atomic-event pipeline relies on **text LLMs** (e.g., GPT-3.5/4o) for both event extraction and preference decisions; while some human checks exist, the paper still **inherits evaluator bias/noise** and only partially audits it.  
* **Caption→QA transfer hinges on data regeneration, not RL directly.** Authors note MrDPO mainly boosts captioning; general QA gains arrive **after** re-annotating and SFT, raising questions about how much MrDPO helps end-to-end QA absent the data-generation step. 
* **Compute/engineering cost and robustness details.** Training uses substantial H800 resources across stages, and robustness to deployment artifacts (codec changes, audio dropouts) is not deeply probed.

### Questions
1. How convincing is the atomic-event LLM-evaluation pipeline as a reward signal—what additional **human studies, inter-evaluator agreement, or bias audits** would you require to fully trust the missing/hallucination estimates? 
2. To what extent should the paper **disentangle** improvements from MrDPO itself versus the **re-captioned SFT data**? Which ablations (e.g., MrDPO-trained model evaluated on QA *without* regenerated data) would most clarify causal contribution? 
3. Given the reported **H800 budgets**, what **efficiency analyses** (e.g., fewer rounds, smaller evaluators) and **robustness tests** (e.g., compression/noisy audio) would you consider necessary to judge real-world deployability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose MrDPO to enhance the model's video captioning capability, and leverage this capability to generate high-quality video caption data, which is then used to train a general-purpose model, thereby improving its general video question-answering ability.

### Strengths
1. video-SALMON 2 demonstrates strong video captioning capability, surpassing many well-known models such as Gemini 1.5, VideoLLaMA3, and Qwen2.5-VL.
2. video-SALMON 2 provides a self-constructed caption benchmark, which helps advance the video captioning capabilities of other models.
3. video-SALMON 2 exhibits powerful audio-visual question-answering performance, achieving strong results on VideoMME.

### Weaknesses
1. The paper’s first main contribution MrDPO (a DPO variant that updates the reference model during training to improve stability) is not novel; a similar approach was already proposed in TR-DPO [1]. Moreover, the paper’s primary focus is on enhancing video captioning performance, yet the design of MrDPO is unrelated to video captioning and appears to be a generic DPO method. The motivation behind MrDPO and the mechanism by which it improves video captioning capability remain unclear.

2. The second main contribution (using video captioning to enhance video question-answering performance) is also problematic. On one hand, this idea is already well established in the field and has been thoroughly validated in prior work. On the other hand, the authors fail to explain why or how improved video captioning capability leads to better video QA performance, leaving the underlying rationale unaddressed.

[1] Learn Your Reference Model for Real Good Alignment

### Questions
1. What is the design rationale behind the Video Caption Benchmark? Why did you develop a benchmark that evaluates video caption quality based on the extraction of various types of events? Is GPT-3.5 truly capable of reliably extracting diverse events from video captions?
2. Why does improving video captioning capability lead to enhanced video question-answering performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces video-SALMONN 2, a audio-visual large language model (MLLMs) designed to improve video captioning and question answering. The contribution of this paper is Multi-round Direct Preference Optimization (MrDPO), a novel RL-based training strategy. This method iteratively refines the model's captioning ability by using a GPT-based reward signal with "atomic events" to measure caption completeness and factuality. Unlike standard DPO, which uses a fixed reference model, MrDPO periodically merges the latest model improvements (via a LoRA proxy) into the reference model, preventing "reference staleness" and enabling continual improvement. The authors then use their best MrDPO-trained model to distill its knowledge by re-annotating a large dataset, creating a high-quality caption corpus. This new corpus is used to SFT a final set of models (video-SALMONN 2+), which not only excel at captioning but also transfer these gains to achieve new state-of-the-art results on a wide range of video QA benchmarks.

### Strengths
- The paper is well-written and easy to follow.
- The proposed method achieves the best performance compared to other methods in diverse metrics and datasets. In particular, it shows better performance than GPT-4o.

### Weaknesses
- MrDPO seems to be very similar to the concept of GRPO. As GRPO updates and adopts old policy model, MrDPO uses the updated model as a reference model. Could you compare MrDPO with GRPO?
- I’m concerned about the complexity and cost of the MrDPO pipeline. The proposed MrDPO requires: evaluating the quality of every video caption using GPT-3.5. In comparison of using the verifiable reward function such as BLEU, ROUGE, the caption evaluation with GPT is time-consuming and cost-intensive.
- It would be better to include the prompt that evaluates the quality of video caption. Since finding out and evaluating atomic events in the video caption is a core component of the proposed algorithm, the prompt utilized for the validation should be included.
- Could the authors further clarify the distinction between the "gDPO" loss and the common practice of adding a standard SFT loss (using ground-truth data) as a regularizer to the DPO objective? The paper claims a distinction, but Equation 3 appears to be a standard DPO loss plus a standard SFT loss.

### Questions
Please refer to weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
