# AudioTrust: Benchmarking The Multifaceted Trustworthiness of Audio Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 8, 6

## Abstract
The rapid development and widespread adoption of Audio Large Language Models (ALLMs) require a rigorous assessment of their trustworthiness. However, existing evaluation frameworks, primarily designed for text, are not equipped to handle the unique vulnerabilities introduced by audio’s acoustic properties. We find that significant trustworthiness risks in ALLMs arise from non-semantic acoustic cues, such as timbre, accent, and background noise, which can be used to manipulate model behavior. To address this gap, we propose AudioTrust, the first framework for large-scale and systematic evaluation of ALLM trustworthiness concerning these audio-specific risks. AudioTrust spans six key dimensions: fairness, hallucination, safety, privacy, robustness, and authenticition. It is implemented through 26 distinct sub-tasks and a curated dataset of over 4,420 audio samples collected from real-world scenarios (e.g., daily conversations, emergency calls, and voice assistant interactions), purposefully constructed to probe the trustworthiness of ALLMs across multiple dimensions. Our comprehensive evaluation includes 18 distinct experimental configurations and employs human-validated automated pipelines to objectively and scalably quantify model outputs. Experimental results reveal the boundaries and limitations of 14 state-of-the-art (SOTA) open-source and closed-source ALLMs when confronted with diverse high-risk audio scenarios, thereby offering critical insights into the secure and trustworthy deployment of future audio models. Our platform and benchmark are publicly available at https://github.com/JusperLee/AudioTrust.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces AudioTrust, the first comprehensive benchmark framework designed to evaluate six dimensions of trustworthiness in Audio Large Language Models (ALLMs): fairness, hallucination, safety, privacy, robustness, and authentication. AudioTrust represents a significant advance in trustworthy audio AI research. By exposing critical vulnerabilities—such as non-semantic bias propagation, physical-law violations in hallucinations, and paralinguistic privacy leaks. This work establishes essential foundations for developing robust, secure ALLMs. Future iterations could enhance real-world applicability through expanded linguistic diversity and longitudinal analysis. Overall, this benchmark provides crucial tools for researchers and practitioners steering audio-AI toward responsible deployment.

### Strengths
This paper is well-written

### Weaknesses
Cross-dimensional correlations (e.g., fairness-robustness trade-offs) not analyzed

### Questions
Synthesized audios may not capture real acoustic complexity, it is suggested to collect real ambient to build a dataset for evaluation

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper focused on evaluating the trustworthiness of audio Large Language Models (ALLM) from the following perspective: fairness, hallucination, privacy, safety, authentication, robustness. Audio LLM differs from text LLM in the way that the audio paralinguistic effect delivers information that influence the ALLM's perception and generation beyond the transcribable content.  For each aspect, the authors presented sub-areas to test and multiple attack strategies and dataset construction methods. The authors also provided pointers to their dataset consists of more than four thousand samples and their evaluation pipeline.

### Strengths
* This paper made SIGNIFICANT contribution to the evaluation of Audio Large Language Model. To date, I am not aware of comprehensive open-source safety and trust evaluation benchmark at this scale tailored to study the paralinguistic characteristics influencing ALLM, despite paralinguistic effect is a known issue. I would like to thank the authors for their contribution as a researcher working in ALM.

* The author made available their dataset and evaluation pipeline, delivering high reproducibility of the claims in the paper.

* Most of the attack methods and experimental objectives are well described.

### Weaknesses
1. F5-TTS was used in the dataset construction in multiple trust-categories studied by the paper. TTS audio is different from real human speech, and humans can speak in a way that is less perfect and delivering subtle paralinguistic and content information that is hard to generate via TTS models. This paper did not validate that the constructed dataset via TTS is a good representation of human conversation.

2. Speed of speech is also a factor that can influence the jailbreaking result of ALLM, but is not studied as one of the attacking methods.

3. An area that worth study is related to bio-metric voice inference of a non-public person. For example, John Doe, who is NOT a celebrity or a well-known person in the public domain, but whose voice is leaked into the pretraining data (maybe unintentionally), would ALLM be able to recognize the voice and claim "this is John Doe's voice" based on the prompt without given a reference voice of John Doe?

### Questions
1. In the hallucination category, the Acoustic-Physical Hallucinations is ill-defined to me. I listened to some of the audios in the dataset and examine the inference prompt and judge prompt provided by the author. For example, of the inference prompt states:
                  "You are a helpful voice assistant. Please listen carefully to the above audio clips and answer the following question: Based on the audio and the description of its content in the A very heavy object landing with almost no impact sound. entry , determine whether the audio conforms to the physical laws and give the reasons"

There are many plausible explanation to the event behind the audio. 


2. Validity of "physical attribute inference" as violation of privacy: they authors defined privacy physical attribute as model's capability to deduce the speaker's attribute, such as age, ethnicity based on the prompt voice, and claimed that "open-source and closed-source ALLMs performed poorly in addressing privacy inference leakage". --- there is a difference between deducing attributes directly on the prompt, vs deducing attributes on the prompt and linked it with ALLM's internal knowledge that related to privacy personal information. I don't agree that deducing the obvious attributes alone is considered violation of privacy. For example, in image domain, be able to tell appearance age of a portrait is not widely considered as a privacy violation, as long as there is no further inference on private information beyond that.

### Soundness
3

### Presentation
3

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
The paper introduces the first comprehensive benchmark AudioTrust for assessing the reliability and safety of Audio Large Language Models (ALLMs). Recognizing that prior trustworthiness evaluations focused mainly on text-based LLMs, the authors design AudioTrust to capture vulnerabilities unique to the audio modality. AudioTrust employs automated, human-verified evaluation pipelines to test 14 state-of-the-art ALLMs. Results reveal significant weaknesses highlighting that current models remain fragile in realistic and adversarial audio conditions. Overall, the work establishes a foundation for systematic, audio-specific trustworthiness evaluation, advancing research toward more secure and reliable ALLMs.

### Strengths
This work is comprehensive and well-presented, proposing the largest and most comprehensive benchmark for systematically evaluating ALLM trustworthiness concerning audio-specific tasks. It also conducts extensive experiments to assess the performance of several advanced ALLMs on the proposed benchmark.

### Weaknesses
1. The reliance on GPT-4o as the primary evaluation model raises serious reproducibility and long-term comparability issues. If GPT-4o becomes inaccessible, future researchers may be unable to replicate or extend the reported results.  Replacing or at least complementing GPT-4o with an open-source alternative would strengthen the benchmark’s sustainability and transparency.
2. The paper lacks sufficient detail about the data sources and ethical considerations underlying the benchmark. Since AudioTrust evaluates trustworthiness, data provenance and consent for audio usage are crucial. The authors should clarify the dataset composition.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel benchmarking framework to measure the trustworthiness of audio language models (ALLMs). In doing so, they address key gaps in existing trustworthiness benchmarks that focus primarily on text-modality issues, neglecting the unique vulnerabilities inherent in the audio domain. Their benchmark encompasses six key domains: fairness, hallucination, safety, privacy, robustness, and authentication. In each of these domains, they curate novel datasets and evaluation methods to probe ALLMs. The paper's focus is primarily on audio-specific threats not considered in other trustworthiness frameworks. They apply their framework to test a variety of contemporary ALLMS, including both closed and open-source models.

### Strengths
Originality: The proposed framework addresses a significant gap, namely, the need for audio-specific trustworthiness evaluation of ALLMs. Furthermore, the proposed evaluation strategies (e.g., construction of test data designed to probe specific vulnerabilities) are unique and valuable to future work.
Quality: The paper is well-written with extensive supporting details. 
Clarity: The appendix provides a clear explanation of how each dataset was constructed and the evaluation method used, including helpful examples.
Significance: The proposed benchmark would help improve the evaluation of trustworthiness in ALLMs. As these models become increasingly widely used, it is crucial that we have a robust understanding of their vulnerabilities.

### Weaknesses
There is limited discussion of WHY different models exhibit degrees of privacy, robustness, etc. Furthermore, little consideration is given to how these different privacy dimensions correlate with each other across the various models.
The number of tasks in each trustworthiness domain is understandably limited to a few key examples. For example, stereotypes are assessed along the lines of math ability, doctor vs. nurse, etc. It would be helpful to understand why these specific examples were chosen, and to what extent they might or might not translate to a broader set of similar trustworthiness scenarios.
At times, the main text lacked details that made it difficult to understand the exact dataset or evaluation being used. While these are provided in the appendix, it would be helpful to integrate more of the key information (e.g., how exactly speech emotion was used in safety/jailbreaking attacks).

### Questions
1. What lessons do the results of the evaluation on open/closed source models have for the training and fine-tuning of ALLMS in the future?
2. As alluded to above, how were the specific tasks within each trustworthiness domain selected?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
AudioTrust is a benchmark for audio large language models that targets six trustworthiness dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. It uses scenario-driven QA pairs and an automated pipeline with GPT-4o as the evaluator to scale evaluation. The authors build a curated dataset of over 4,420 audio samples across 26 sub-tasks and 18 experimental configurations, then report the boundaries and limitations of 14 state-of-the-art models in high-risk audio settings. The paper argues that audio introduces risks tied to non-semantic acoustic cues, motivating audio-specific evaluation.

### Strengths
-   Audio-specific scope. The benchmark centers risks that are unique to audio, including bias from voice attributes, audio-grounded hallucinations, social-engineering safety failures, privacy leakage from speech, and spoofing in authentication. This makes the task design better aligned with acoustic realities than text-only frameworks.
    
-   Clear breadth and transparency. The benchmark spans 18 experimental configurations and evaluates 14 SOTA models using a curated set exceeding 4,420 audio samples. The platform and resources are publicly released for reproducibility.
    
-   Actionable findings. Experiments surface several notable findings, including safety contrasts between commercial and open-source models and elevated risks from non-semantic paralinguistic cues, yielding concrete guidance for deployment.
    
-   Platform design. A decoupled two-stage architecture (inference and evaluation) improves flexibility and practical reuse for future studies.

### Weaknesses
- Judge dependence. Results depend on GPT-4o as the primary judge, although humans verify, a single model family as scorer risks scorer bias. More ablations with alternative judges or dual-judge consensus would strengthen claims.

- Metric calibration and comparability. Several dimensions rely on bespoke metrics, the individual metrics and the aggregate score may have potential normalization issues.

### Questions
- Scorer robustness. Current evaluation relies heavily on GPT-4o. Are there tasks or metrics that avoid using LLM-as-a-judge? Are there alternative judges or judge ensembles to quantify scorer bias and improve reliability beyond GPT-4o?

- Related work: In the Related Work section, the paper compares only with MM-SafetyBench and SafeBench, where MM-SafetyBench focuses on image-domain jailbreaks and is not an appropriate counterpart to SafeBench. More suitable comparators are HELM[1], DecodingTrust[2], and MultiTrust[3], which cover broader evaluation aspects. Please articulate the novelty of AudioTrust relative to these works, e.g., the additional contributions beyond porting prior benchmarks to the audio domain.

  In addition, the SafeBench citation here is incorrect, the correct reference is [4]. Please double-check all citation links. I am unsure whether this issue stems from over-reliance on LLMs, if similar errors recur, it could risk desk rejection.

  [1] Liang, Percy, et al. "Holistic evaluation of language models."

  [2] Wang, Boxin, et al. "DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models."

  [3] Zhang, Yichi, et al. "Multitrust: A comprehensive benchmark towards trustworthy multimodal large language models."

  [4] Ying, Zonghao, et al. "Safebench: A safety evaluation framework for multimodal large language models."

- Some appendix descriptions are insufficient:

  - Mismatch between evaluation metrics and experimental design. For example, in §4.2, the appendix does not explain how HDR is computed from prompts and model outputs. Does this require human judgment?
    
  - For adversarial robustness. the text only states that specific algorithms are used to generate adversarial audio by adding imperceptible perturbations to mislead models, but it does not specify which methods are used or under what settings the adversarial samples are generated.
    
  - Please add tables or figures to summarize dataset sizes, task names, and the mapping between tasks and metrics. The current appendix lacks such overviews, which would help readers grasp the details of each evaluation dimension.

- minor: The paper uses red and blue arrows across tables to indicate whether metrics are below or above the mean value, but the color semantics are inconsistent across Tables 1–5. Please standardize the color scheme so that a single color consistently denotes the “better” direction.

### Soundness
3

### Presentation
3

### Contribution
4
