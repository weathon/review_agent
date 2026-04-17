# Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The rapid advancement of AI-generated content (AIGC) has enabled the synthesis of visually convincing images; however, many such outputs exhibit subtle \textbf{semantic anomalies}, including unrealistic object configurations, violations of physical laws, or commonsense inconsistencies, which compromise the overall plausibility of the generated scenes. Detecting these semantic-level anomalies is essential for assessing the trustworthiness of AIGC media, especially in AIGC image analysis, explainable deepfake detection and semantic authenticity assessment.In this paper,  we formalize \textbf{semantic anomaly detection and reasoning} for AIGC images and  introduce \textbf{AnomReason}, a large-scale benchmark with structured annotations as quadruples \emph{(Name, Phenomenon, Reasoning, Severity)}. Annotations are produced by  a modular multi-agent pipeline (\textbf{AnomAgent}) with lightweight human-in-the-loop verification, enabling scale while preserving quality. At construction time, AnomAgent processed approximately 4.17\,B GPT-4o tokens, providing scale evidence for the resulting structured annotations. We further  show that models fine-tuned on AnomReason achieve consistent gains over strong vision-language baselines under our proposed semantic matching metric (\textit{SemAP} and \textit{SemF1}). Applications to {explainable deepfake detection} and {semantic reasonableness assessment of image generators} demonstrate practical utility. In summary, AnomReason and AnomAgent serve as a foundation for measuring and improving the semantic plausibility of AI-generated images. The code is available at \url{https://github.com/chuangchuangtan/Semantic-Visual-Anomaly-Detection-and-Reasoning}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AnomReason, a large-scale benchmark annotated with structured quadruples to capture commonsense, physical, and relational inconsistencies in generated images. The dataset is built using AnomAgent, a modular multi-agent framework that decomposes anomaly detection into entity parsing, anomaly mining, and structured output generation, with lightweight human-in-the-loop verification. The author also provides AnomReasonor-7B, a model fine-tuned on AnomReason, which achieves comparable performance with GPT-4o.

### Strengths
1. The paper is clearly written and well-structured.
2. The introduction of AnomReason, AnomReason Deepfake, and AnomAgent provides both a comprehensive benchmark and a scalable annotation framework for semantic-level anomaly detection.

### Weaknesses
1. The evaluations in Sections 4.1 and 4.2 are limited to the proposed AnomReason and AnomReason Deepfake benchmark. The paper should include results on additional third-party anomaly detection datasets, such as FakeReasoning, Spot the Fake, etc.
2. The evaluations in Sections 4.1 and 4.2 primarily evaluate general-purpose VLMs rather than models specifically designed or fine-tuned for anomaly or forgery detection, such as FakeReasoning, Spot the Fake, Aigi-holmes, etc. 
3. The proposed pipeline mainly relies on prompting and LoRA fine-tuning. There is limited method innovation in model architecture or training strategy, which somewhat reduces the technical novelty of the work.

### Questions
1. Could you further analyze the results to determine what kinds of anomalies current T2I models are most likely to generate?
2. Given that AnomReasonor-7B shows the strongest anomaly detection ability, could you analyze why its performance is inferior to GPT-4o on AnomReason-Deepfake? Is it because not all AI-generated images in this benchmark exhibit visible anomalies?
3. How will the selection of hyperparameters in SFT influence AnomReasonor-7B’s performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes **AnomAgent**, a multi-agent system capable of automatically annotating perceptible anomalies in images, effectively replacing the need for manual labeling. Furthermore, the authors construct a benchmark containing over 20,000 samples generated through this fully automated annotation pipeline, featuring a structured output format. In the experiments, the authors validate the effectiveness of their approach from three complementary perspectives.

### Strengths
1. The fully automated annotation pipeline effectively addresses the inconsistency issues that commonly arise among human annotators, greatly facilitating the process of dataset construction.  
2. The constructed benchmark is highly valuable for research on interpretable image forgery analysis.  
3. The paper introduces the first metric designed to evaluate **semantic anomalies**, filling an important gap in semantic-level assessment for image authenticity evaluation.

### Weaknesses
1. It is unclear what role the **Severity Score** plays in the overall framework — please clarify its purpose and how it contributes to the evaluation or annotation process.  
2. Please provide more detailed information about the **manual verification** procedure. Specifically, explain how high-quality and image-relevant annotations are selected and validated, and justify the rationality and reliability of this filtering process.

### Questions
Please refer to the *Weakness* section for detailed explanations.  
If the authors can adequately address my concerns, I would be very willing to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AnomReason, a large-scale benchmark for semantic anomaly detection and reasoning in AI-generated images. It introduces AnomAgent, a modular multi-agent framework that detects and explains semantic inconsistencies through structured outputs (Name, Phenomenon, Reasoning, Severity). The dataset includes 21K photorealistic AIGC images with 170K verified anomalies. New BERTScore-based metrics (SemAP, SemF1) evaluate phenomenon- and reasoning-level accuracy. Experiments show that existing VLMs struggle with commonsense reasoning, while the fine-tuned AnomReasonor-7B achieves state-of-the-art performance.

### Strengths
1. The proposed AnomReason dataset and AnomAgent framework are well-designed and provide valuable resources for both the forgery detection and image generation communities.
2. The paper conducts extensive reasoning experiments across a wide range of VLMs and generative models, providing a comprehensive analysis of their capabilities in semantic-level reasoning, detection and generation.

### Weaknesses
1. The paper claims to employ an agent-based framework for benchmark construction; however, the system is essentially composed of prompt-driven GPT-4o modules, which cannot be considered true autonomous agents. The human-in-the-loop stage merely filters GPT-4o outputs rather than providing substantive corrections, thus retaining the inherent biases of GPT-4o. This design choice results in the strong performance of GPT-4o in Table 1—surpassing even newer models such as GPT-5 and GPT-o3—leading to a unfair comparison.
2. The evaluation metrics (SemAP and SemF1) rely on text similarity thresholds for anomaly matching, which can significantly influence performance outcomes across the Phe, Rea, and Full evaluation views. However, the paper does not provide a clear rationale or sensitivity analysis for the chosen threshold values, making it difficult to assess the stability and robustness of the reported results.
3. The paper conducts extensive experiments on various VLMs and generative models, but it fails to distill clear and generalizable conclusions from the results. Moreover, in the deepfake detection task, the evaluation lacks comparison with specialized forgery detection models.
4. In the deepfake detection task presented in Table 2, the fine-tuned AnomReasonor-7B achieves only 82% accuracy, performing even worse than GPT-4o’s zero-shot reasoning despite the absence of any cross-domain generalization challenge. This raises concerns about the validity and robustness of the proposed dataset and the effectiveness of its supervision strategy.
5. The paper defines a new task, semantic anomaly detection and reasoning, claiming that existing MLLM-based detectors operate only at the surface level while the proposed method captures content-level semantics. However, in the context of forgery detection, all MLLM-based approaches inherently perform high-level semantic analysis, as low-level forensic clues are beyond their perceptual capacity. While structuring semantic anomalies is beneficial, this additional task definition appears unnecessary.

### Questions
please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel task and dataset for semantic visual anomaly detection and reasoning, called AnomReason. The dataset was constructed using an iterative hybrid annotation framework, AnomAgent, which combines VLM generations with human verification.
The authors conduct extensive benchmarking of both open-weight and proprietary models on their dataset. Furthermore, they fine-tune a VLM on the AnomReason training set, demonstrating notable performance improvements as a result.

### Strengths
- The paper introduces a novel dataset that addresses an important gap in semantic visual anomaly detection and reasoning.

- The dataset’s design is rigorously supported by large-scale human annotation efforts, ensuring the validity and reliability of the resource.

- The authors provide extensive benchmarking of various vision-language models, offering valuable insights into model performance on this new task.

### Weaknesses
**[W1]** The paper does not provide statistics on the diversity of visual phenomena present in the benchmark. This omission makes it difficult to assess how the dataset splits are affected, particularly regarding potential entity overlap. Without this information, there is a risk of data leakage across splits in terms of topic, domain, or subject, which could impact the validity of fine-tuning and evaluation.

**[W2]** A significant limitation in the experimental results (Sections 4.1 and 4.2) is the lack of counterfactual examples, i.e. AI-generated images that do not contain semantic anomalies, or real-world images. Including such examples would allow for a more thorough assessment of the fine-tuned model’s ability to distinguish between anomalous and non-anomalous cases, and between AIGC and non-AIGC.

**[W3]** The authors rely on BERTScore to compare extracted anomalies to the ground truth. Given the known limitations of BERTScore (especially its insensitivity to subtle semantic differences) it may be more appropriate to use an LLM-based judge for evaluation, as suggested in recent literature. Also, it is important to validate the proposed metrics with human evaluation. Conducting a sampled error analysis would help identify failure modes and provide additional support for the validity of the chosen metrics.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
