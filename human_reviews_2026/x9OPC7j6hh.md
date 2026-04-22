# Low-Resource Finetuning for Hallucination Mitigation in Language Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Hallucinations in Large Language Models (LLMs) pose a significant challenge to their reliable deployment across domains, arising inherently from their design as statistical models that maximize next-token prediction probability based on training data. While methods such as LettuceDetect, RAG-HAT, and prompting techniques have demonstrated efficacy in hallucination detection and mitigation within Retrieval-Augmented Generation (RAG) frameworks, limitations persist. To address these, we propose a novel low-resource hallucination mitigation pipeline that fine-tunes LLMs on synthetic dataset using feedback from LettuceDetect. Our approach reduces hallucination rates in open-source small language models, as validated through evaluations on RAGTruth and PILE-10K benchmarks. We further discuss the pipeline’s extensibility to domain-specific applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors propose a fine-tuning method aimed at reducing hallucinations in small language models. They create a synthetic fine-tuning dataset using a three-step approach: (1) generating 20K nouns with Gemma-3-4b-it, (2) generating 5–15 related attributes for each noun, and (3) producing a text for each noun–attribute pair, resulting in approximately 180K samples. Several small language models are then fine-tuned on this dataset.

During training, a hallucination detector (lettucedetect-large-v1) is used to classify each output token as either a hallucination or non-hallucination. Tokens identified as hallucinations are penalized, while non-hallucinated tokens are rewarded. The models, both before and after fine-tuning, are evaluated on the PILE-10K and RAGTruth datasets to measure perplexity and non-hallucination rate, respectively.

The evaluation results show that while the fine-tuned models exhibit slightly worse perplexity, they achieve a higher non-hallucination rate—indicating reduced hallucination compared to the base models. The authors also find that LoRA fine-tuning outperforms both QLoRA and full fine-tuning.

### Strengths
•	The paper tackles an important and highly relevant problem: mitigating hallucinations in decoder-only language models.

•	The proposed fine-tuning approach demonstrates consistent improvements in reducing hallucinations across all evaluated models.

•	The authors include a diverse set of small language models from different families and sizes, which strengthens the generalizability of the findings.

•	The evaluation considers two complementary metrics—perplexity and non-hallucination rate—providing a balanced view of model quality and factual reliability.

### Weaknesses
•	While the paper reports hallucination rates, it does not include any task-specific evaluations. Including such experiments could help demonstrate whether the reduced hallucination rate also translates to improved or stable downstream task performance.

•	Although Section 1.3 lists several fine-tuning-based mitigation methods, a direct comparison with one or two of these techniques would make the empirical contribution stronger.

•	It seems that some hyperparameters may have been adjusted using the test set, which could lead to optimistic estimates of performance. Clarifying or separating validation and test data would improve the rigor of the evaluation.

•	Because the same hallucination detector is used for both training and evaluation, improvements on this metric may partially reflect training bias. Using an additional, independent detector (even if less powerful) could help validate the robustness of the results.

•	The results show that while hallucination decreases, perplexity increases somewhat (e.g., from 17 to 21 for Gemma-3-4b-it). It might be helpful to discuss this trade-off and potential strategies to balance factual accuracy and fluency.

•	The related work section is quite detailed and informative but could be made more concise. The saved space could be used to strengthen the empirical section—for example, by including comparisons with other mitigation techniques, visualizing sample fine-tuning data, or showing token-level hallucination rates before and after fine-tuning.

### Questions
Q1. Clarity on “Low-Resource” Terminology
The title mentions “low-resource,” which can be confusing since the experiments are conducted in English—a high-resource language. Could the authors clarify in what sense the setting is considered “low-resource”? For example, does it refer to limited model size, limited fine-tuning data, or computational constraints rather than language resource availability?

Q2. Comparison with Existing Mitigation Techniques
The paper introduces an interesting fine-tuning-based approach for hallucination mitigation, but it does not include comparisons with existing methods. Could the authors elaborate on this decision? Were existing techniques difficult to reproduce or incompatible with the proposed setup?

Q3. Evaluation on Downstream Tasks
It would be helpful to understand how the fine-tuned models perform on downstream tasks (e.g., summarization or question answering). Have the authors considered evaluating their models on such tasks to assess whether hallucination reduction affects task-specific performance?

Q4. Hyperparameter Optimization Details
The paper does not clearly specify how hyperparameters were tuned. Could the authors clarify which dataset was used for this purpose? From the current description, it seems that hyperparameters may have been selected based on the RAGTruth test set, which could lead to overfitting. If that’s not the case, additional details on the validation procedure would be appreciated.

### Soundness
2

### Presentation
3

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
This paper proposes a low-resource fine-tuning pipeline for hallucination mitigation in large language models. The method uses LettuceDetect, a pre-trained hallucination detection model, as both a weak supervisor and evaluator. The authors generate a synthetic dataset of noun–attribute pairs using Gemma-3-4B, then fine-tune several small open-source models, including LLaMA-3-8B-Instruct, Qwen3-1.7B, and Gemma variants. The training objective penalizes tokens identified as hallucinated by LettuceDetect and rewards non-hallucinatory tokens, using a Jump ReLU-based loss. Evaluation is conducted on RAGTruth (for hallucination rate) and PILE-10K (for perplexity). The results show modest improvements in non-hallucinatory rate (around 5–8%) with minimal changes in perplexity. The authors claim the approach is architecture-agnostic, computationally efficient, and suitable for small models under resource constraints.

### Strengths
1. The topic of hallucination mitigation in language models is relevant and practically important, especially for low-resource or on-device applications.  
2. The proposed pipeline is simple, computationally light, and architecture-independent, making it applicable to smaller open models without retraining large detectors.  
3. The use of a weak supervision setup is a pragmatic approach to avoid heavy human annotation.  
4. The goal of integrating hallucination control into fine-tuning rather than prompting or retrieval methods is conceptually reasonable.

### Weaknesses
1. The novelty of the work is minimal. The method merely reuses LettuceDetect as a weak labeler, without introducing new learning objectives, data strategies, or theoretical insights.  
2. The experimental validation is insufficient. The model is evaluated on only two small datasets, and the metrics are derived from the same LettuceDetect model used for training, introducing circularity.  
3. The writing and structure are poor, and unclear narrative flow, which makes it difficult to follow.  
4. The paper contains no figures, diagrams, or visual explanations of the proposed pipeline, which severely limits clarity.  
5. The synthetic dataset is overly simplistic and unrelated to the evaluation tasks, weakening the relevance of the training setup.  
6. There is no human evaluation or comparison to other fine-tuning strategies like F2, HIPO, or preference optimization methods.  
7. The mathematical section offers no real theoretical contribution and introduces unnecessary notation for a simple loss function.  
8. The use of LettuceDetect as both training signal and evaluator risks overfitting to detector-specific biases rather than improving genuine factual reliability.

### Questions
1. Provide independent evaluation using either human annotation or alternative hallucination metrics to avoid circular reasoning.  
2. Include visual diagrams of the training pipeline, loss computation, and evaluation flow to improve readability.  
3. Add qualitative examples comparing pre- and post-fine-tuning responses to illustrate concrete behavioral changes.  
4. Perform ablation studies to separate the effects of LettuceDetect supervision, loss configuration, and fine-tuning strategy.  
5. Analyze efficiency in terms of training time, memory usage, and scalability across model sizes.  
6. Reorganize and clean the manuscript for clarity, consistent formatting, and concise writing.  
7. Compare quantitatively against recent fine-tuning and preference optimization approaches to position the contribution properly.  
8. Discuss the limitations of using a noisy weak supervisor and suggest potential methods to mitigate detector bias.  
9. Given the small technical novelty and weak empirical results, the paper would be more suitable as a workshop or system report rather than a full ICLR submission.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a low-resource fine-tuning pipeline that uses a token-level hallucination detector (LettuceDetect) as both weak supervisor and evaluator. The loss penalizes tokens the detector flags and lightly rewards tokens considered grounded, with LoRA, QLoRA, and full fine-tuning variants. Training uses about 180k synthetic noun-attribute texts and evaluation reports Non-Hallucinatory Rate on RAGTruth plus perplexity on PILE-10K. Reported gains in NHR are clearest for smaller models, while perplexity generally increases after fine-tuning.

### Strengths
1. Nice idea and approach - Token-level supervision for hallucinations is a sensible way to train model with dense signals and could be architecture-agnostic and cheap to run.  

2. Training dataset is simple - The noun-attribute generator is straightforward and scales, which makes the recipe easy to reproduce.

### Weaknesses
1. Flawed evaluation - The same detector is used both for supervision and evaluation. The detector’s own precision and recall on RAGTruth are modest, which raises the risk that the model learns the judge rather than factuality. Human checks are described only at a high level, without any details that would establish reliability. Consider adding additional evaluations such as FAVA-Bench [1] or FactScore [2] to distinguish overfitting from generalization.  

2. Lack of baselines - Results compare only base vs fine-tuned models. This problem is widely studied and has established previously proposed alternatives. At minimum include SFT on the same data with standard cross-entropy, refusal tuning [3], “corrected data” training, and simpler token-weighting or DPO variants to test whether the detector-guided loss is essential. Effects of data size are also important to study. The paper itself mentions many prior work on finetuning models for addressing hallucination which should be considered as baselines 

3. Flawed perplexity findings - The paper’s narrative suggests mixed perplexity effects, yet Table 1 shows perplexity increases for all listed models after fine-tuning, which implies degradation in general language modeling. Additional evaluations for broader capability checks such as MMLU or GLUE need to be done ensure core skills are not harmed.  

References 
[1] FAVA-Bench - Mishra, A., Zhou, Y., Wang, S., et al. Fine-grained Hallucination Detection and Editing for Large Language Models. arXiv:2401.06855, 2024. 

[2] FActScore - Min, S., Krishna, K., Lyu, X., et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long-Form Text Generation. EMNLP 2023 (Main), 2023. ACL Anthology+1 

[3] Refusal tuning - Zhang, H., Diao, S., Lin, Y., et al. R-Tuning: Instructing Large Language Models to Say “I Don’t Know”. NAACL 2024 (Long), 2024.

### Questions
see weaknesses.

### Soundness
1

### Presentation
2

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
This paper proposes a low-resource finetuning pipeline designed to mitigate hallucinations in Small Language Models (SLMs), particularly within Retrieval-Augmented Generation (RAG) contexts. Unlike existing methods that often rely on large, expensive "judge" models like GPT-4 (e.g., RAG-HAT), this work employs a lightweight, open-source hallucination detector, LettuceDetect, as a "weak supervisor".

### Strengths
The pipeline is resource-efficient and practical. By avoiding reliance on large, closed-source models (like GPT-4) or complex optimization methods (like DPO), it provides an accessible pathway for improving SLM factuality, especially for on-device applications.
This is possible by introducing a loss function (JReLU) specifically tailored to the task of token-level hallucination mitigation. This allows for direct optimization against the detected undesirable behavior, rather than just mimicking a style. (unlike DPO)

The method demonstrates strong empirical results when combined with LoRA. It successfully improves the Non-Hallucinatory Rate (NHR) for several SLMs, showing its potential utility.

### Weaknesses
The proposed training pipeline appears to be incomplete and fundamentally flawed. Its success is entirely dependent on the implicit regularization of LoRA. The experiments clearly show that when using Full-Finetuning, the method fails and even degrades model performance (Table 2). This indicates the pipeline, on its own, is unstable.

The evaluation is very narrow. It only reports the target metric (NHR on RAGTruth) and a general language metric (Perplexity on PILE-10K). There is no evaluation on standard downstream benchmarks (e.g., NQ, or standard open-ended QA tasks). This makes it impossible to assess if the finetuning has catastrophically damaged the model's general reasoning and knowledge capabilities, which is a critical concern given the observed Perplexity degradation.

The paper states that the reward hyperparameter R for non-hallucinatory tokens is "crucial", yet it fails to provide a principled methodology for its selection. The optimal choice of R=1e-5 appears to be the result of a simple grid search and not discussed properly.

### Questions
* Given that the pipeline fails under full finetuning, does the paper suggest that the core contribution is not the pipeline itself but rather the finding that this JReLU loss function only works when combined with a strong implicit regularizer like LoRA?

* A clear trade-off between NHR improvement and Perplexity degradation is shown. From a practical standpoint, do you believe this degradation in fundamental language capability is an acceptable price to pay for the observed NHR gains?

* Why did you not report performance on standard downstream benchmarks (e.g., open-ended QA tasks)?

* Why was the JReLU threshold $\tau$ arbitrarily fixed at 0.5, rather than being treated as a critical hyperparameter to be calibrated on a validation set? Given that similar thresholds are pivotal for performance in many works (e.g., conformal/selective prediction and uncertainty quantification), this "natural" choice seems unprincipled and suboptimal.

### Soundness
1

### Presentation
2

### Contribution
1
