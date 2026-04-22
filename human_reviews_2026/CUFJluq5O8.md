# U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Universal multimodal retrieval (UMR), which aims to address complex retrieval tasks where both queries and candidates span diverse modalities, has been significantly advanced by the emergence of MLLMs. While state-of-the-art MLLM-based methods in the literature predominantly adopt contrastive learning principles, they often differ in their specific training recipes. Despite their success, the mechanisms underlying their retrieval capabilities remain largely unexplored, potentially resulting in suboptimal performance and limited generalization ability. To address these issues, we present a comprehensive study aimed at uncovering the key factors that drive effective embedding learning for UMR using MLLMs. We begin by implementing a general MLLM-based embedding learning pipeline, and systematically analyze the primary contributors to high-performing universal retrieval systems. Based on this, we explore various aspects of the details in embedding generation and training strategies, including progressive transition, hard negative mining and re-ranker distillation. Notably, our findings reveal that often-overlooked factors can have a substantial impact on model performance. Building on these discoveries, we introduce a unified framework termed U-MARVEL (Universal MultimodAl RetrieVal via Embedding Learning), which outperforms state-of-the-art competitors on the M-BEIR benchmark by a large margin in supervised settings, and also exhibits strong zero-shot performance on several tasks such as composed image retrieval and text-to-video retrieval. These results underscore the generalization potential of our framework across various embedding-based retrieval tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates key design decisions for universal multimodal retrieval (UMR) with MLLMs. 
The analysis spans three axes: (1) embedding generation choices, (2) Contrastive learning training factors (i.e., batch size, learning rate, temperature; hard-negative mining), and (3) re-ranker distillation into a single model. 
Based on these findings, the authors propose U-MARVEL, a unified embedding-learning framework that targets single-model retrieval on M-BEIR.

### Strengths
1. Practically impactful problem: design decisions in universal multimodal retrieval are often overlooked and this work advocates for more attention on this, in which can be a good initiative for UMR deployability.
2. Performance: the proposed U-MARVEL framework improves retriever performance on M-BEIR, particularly in single model retrieval.

### Weaknesses
1. Writing clarity: some paragraphs can be revised to improve readability. (e.g., L44: define concrete examples for “diverse and complex requirements in the real-world” in the intro; L161: state the exact metric used for Table 1 (and other tables))
2. Evidence strength: e.g., Gains of 0.1/0.3 (ID-0 vs ID-1) are tiny -- consider significance tests or a larger sample. L263: The square-root rule is argued via only two datapoints (ID-0 vs ID-2); The “from simple to complex” motivation would benefit from a curriculum-learning citation or an ablation isolating each step’s contribution.
3. Some motivations needs to be justified: e.g., L224: "following the principle of advancing from simpler to more complex tasks" would benefit from either a curriculum-learning citation or an ablation isolating each step’s contribution.

### Questions
1. L413, 414: there should be a space between method name and inline citation.
2. L268-269: please make the claim ("we argue that...") more rigorous or use evidence to back it.
3. How exactly do you filter hard negatives? (i.e., what threshold value and how do you select the threshold)
4. Can you provide empirical results (e.g., latency) to compare efficiency between traditional distillation and the improved distillation?
5. Please fix the typo "exihibits".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a systematic empirical study on using multimodal large language models as embedding models for Universal Multimodal Retrieval. The authors implement a baseline contrastive learning pipeline (Qwen2-VL-7B with LoRA) and investigate three main axes: (1) adapting decoder-only MLLMs into instruction-aware embedders (embedding extraction strategies, instruction handling, progressive transition); (2) training MLLM-based embedders under InfoNCE (batch size / learning rate / temperature interactions, hard negative mining and filtering); and (3) compressing the recall-then-rerank pipeline into a single model via improved distillation. Key findings include that bidirectional attention + mean pooling outperforms last-token/compression prompt schemes, masking instruction tokens during pooling helps, learnable temperature improves contrastive learning, and filtered hard negatives stabilize training. Based on these insights the authors propose U-MARVEL, and demonstrate substantial gains on M-BEIR and many zero-shot benchmarks.

### Strengths
1.	the paper offers actionable, well-motivated findings (embedding extraction, instruction masking, progressive transition, filtering of hard negatives, distillation).
2.	backbone, datasets, and training configs are explicitly listed (Tables in Appendix). This makes replication feasible.

### Weaknesses
1.	Limited novelty, this work is just a conbination of existing tricks in retreiver training, some techs are already presented and disscussed by previous works, such as mean token and bidirection attention [1], learnable temperature [2]. Porting them from LLM to MLLM offers little scientific significance.
2.	Insufficient detail on hard-negative filtering criteria, the choice and sensitivity of the threshold for filtering presumed false negatives needs more analysis.
3.	Some experimental comparisons are not entirely fair, and certain conclusions appear to be inconsistent across experiments. Please refer to the Questions section for detailed discussion.
4.  Lacks sufficient details regarding experimental implementation, parameter selection, and description of observed phenomena, such as why masking instruction tokens during mean pooling enhances embedding performance. In such cases, a deeper analysis is required; merely stating the results is insufficient for a valid research contribution.

[1] LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders (BehnamGhader, COLM 2024)

[2] Analyzing the Impact of Learnable Softmax Temperature in Contrastive Visual-Textual Alignment Systems: Benefits, Drawbacks, and Alternative Approaches (Sun and Li, TMLR 2024)

### Questions
1.	In Section 3.1.2, the paper discusses the effectiveness of multi-stage training. However, it is not clear whether this effectiveness comes from the increased amount of training data or from the multi-stage training strategy itself. Is the comparison and the resulting conclusion objective and fair under this ambiguity?
2.	In Section 3.1.1, the paper states that the presence or absence of the Compression Prompt has a significant impact on model performance. Is this effect truly as substantial as reported? Based on my experience, the impact may not be that pronounced.
3.	How exactly is the similarity threshold chosen for filtering false negatives in hard-negative mining? Is it dataset-specific, and how sensitive are results to this threshold?
4.	What is the initialization and optimizer settings of the learnable temperature? Do you observe stable convergence of τ across different batch sizes? Has the variation process of the temperature been recorded？
5.	In the recall-then-rerank pipeline, what is the value of α? Does it vary during training or remain fixed? How is its magnitude determined?
6.	In Section 3.2.2, what is the value of k used in the top-k selection? There are several unclear parameter choices throughout the paper — could the authors specify the exact values of these parameters and the criteria used for their selection?
7.	In Section 3.3, the “continue-hard” experiments show that using only hard negatives leads to performance improvement, while in Section 3.2.2, using only hard negatives resulted in a failure. These conclusions appear inconsistent — is this discrepancy caused by differences in experimental settings or by the choice of data?
8.	The distillation method proposed in Section 3.3 appears relatively conventional — can it truly be considered an innovation? Moreover, the paper only provides a theoretical derivation of the time savings. How does the proposed method’s performance compare with traditional approaches in practice, and is there any empirical comparison of the actual time consumption?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel framework, U-MARVEL, aimed at improving Universal Multimodal Retrieval (UMR) using Multimodal Large Language Models (MLLMs). While current state-of-the-art UMR methods have made significant strides, they often face limitations in terms of retrieval capability, generalization, and the underlying mechanisms driving performance. The authors systematically explore the key factors that contribute to effective embedding learning and retrieval performance, such as embedding generation strategies, progressive training transitions, hard negative mining, and re-ranker distillation. They present U-MARVEL, a unified framework that integrates these findings, demonstrating superior performance over existing methods on the M-BEIR benchmark and strong zero-shot capabilities for tasks like text-to-video retrieval.

### Strengths
1. The introduction of U-MARVEL as a unified framework is a significant contribution. It successfully integrates multiple advanced techniques to create a more efficient and effective retrieval system, and the results demonstrate clear improvements in performance compared to the state-of-the-art methods.
2. The framework is validated through extensive experiments on the M-BEIR benchmark, where U-MARVEL outperforms other methods in multiple retrieval tasks. It also shows strong zero-shot generalization, which is a critical aspect of real-world applications.

### Weaknesses
1. Retrieval and reranking are common pipelines in information retrieval. The authors apply this framework to MLLMs, seemingly leveraging the powerful capabilities of MLLMs to obtain better embeddings. The authors should further clarify the differences and novelty of the MLLM-based framework compared to traditional retrieval frameworks.
2. The authors use Qwen2-VL-7B as the base model for experiments. However, the performance of this retrieval framework on other MLLMs is unclear. The authors should further validate the generalization ability of this retrieval framework.
3. From Table 3 in the paper, I do not see a significant gain from the progressive transition design. The authors should further analyze this in the rebuttal.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

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
The authors perform a systemic investigation to identify an optimized configuration for universal multimodal retrieval (UMR) with a 'general' MLLM-based pipeline. Specifically, they finetune a Qwen2-VL-7B-Instruct model with LoRA as the retriever-reranker embedding model (with distillation) using contrastive learning (InfoNCE) on the M-BEIR dataset (Figure 1b). Based on this 'basic' architecture, they explore three primary questions: (1) how to adapt decoder-only MLLMs into instruction-aware embedding function, (2) how to train embedding models with contrastive learning, and (3) the value of distillation in recall-then-rerank methods. 

For (1), the first question addressed is whether to use {last token, mean token}, {unidirectional, bidirectional}, and the use of a compression prompt instruction (i.e., one-word) in extracting embeddings from the MLLM [bidirectional, mean pooling wins]. The second question addressed for (1) is instruction inclusion and their masking out during mean pooling [masking out instructions during mean pooling wins]. The third question is testing progressive transition in fine-tuning MLLMs with step wise training (instruction tuning -> text-only retrieval (NLI) -> text-image retrieval (CC3M)) [it works].

For (2), the first question asked is the hyperparameter grid search over {batch size, learnable temperature, learning rate} [increasing batch size with appropriate rate scaling and learnable temperature parameters enhance effectiveness]. The second question asked for (2) explores hard negative mining, in-batch negatives, and negative filtering when training InfoNCE [in-batch negatives and filtered top-k hard negatives wins (there are some details on the hard negative filtering strategy)].

For (3), they show that reranker distillation reduces computational requirements and increases feature diversity (i.e., effective distillation via a teacher-student model). This section has a lot of technical details (especially if also considering the appendices) [the method distills the teacher model with minimal performance degredatation]

Put all of this together, and you have U-MARVEL, which is shown to outperform several single model (e.g., LamRA-retriever) and recall-then-rerank (e.g., LamRA) models on M-BEIR in local and global pool settings. On zero-shot settings with unseen datasets, U-MARVEL outperforms all single model settings, is competitive with LamRA on the recall-then-rerank setting for image/text retrieval, and outperforms all baseline models on text-to-video retrieval benchmarks.

### Strengths
Strengths of this work include:
- Variants of the base architecture captures most state-of-the-art systems for universal multi-modal retrieval with VLLM backbones.
- The configuration/hyperparameter search space addresses several relevant questions and provides good evidence for particular design choices.
- The resulting empirical performance is strong across multiple datasets and settings and is compared to multiple recent strong baseline systems.

### Weaknesses
Weaknesses of this work include: 
- While this is partially attributable to space limitations, it was difficult to assess the soundness of the precise method without reading the appendices (I would make some different choices in terms of details, etc.). In the same vein, many of the empirical results in the appendices are useful and not incorporated into the discussion. Finally, with respect to the empirical results, the discussion/analysis is mostly 'just' restating the tables with limited interpretation.
- Maybe I am missing a description of the notation in the empirical results, but I wasn't able to discern statistical significance. Many of the results don't appear statistically significant. 
- As the authors are aware, limiting experiments to 7B models is limiting as I would expect that the findings may differ for larger models (specifically with respect to instruction following). While these are comparable to other models in the literature, as this work follows these, it is going to be necessary to make these comparisons soon. 
- Modulo the ranker distillation details, there is limited methodological novelty. This is a classic "optimize all the pieces" of a architecture to create a definitive strong baseline for this moment in time that more innovative variants can directly compare to.

### Questions
Some questions of mine would include:
- Statistical significance of the empirical results
- Based on your experience in conducting all of these experiments, what are some directions you are thinking about for further improvements that may lead to notable improvements (if any)?
- (For an appendix), were there any additional directions explored that led to inconclusive or interesting negative results?

### Soundness
4

### Presentation
3

### Contribution
2
