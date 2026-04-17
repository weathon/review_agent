# Lifelong Unlearning for Multimodal Large Language Models

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Multimodal large language models (MLLMs) are trained on massive multimodal data, making data unlearning increasingly important as data owners may request the removal of specific content. In practice, these requests often arrive sequentially over time, creating the problem of *MLLM Lifelong Unlearning*. 
However, existing benchmarks have not considered the MLLM lifelong unlearning scenario.
To study this problem, we introduce MLUBench, a comprehensive benchmark for assessing the performance of unlearning methods under MLLM lifelong unlearning. MLUBench comprises 127 entities of 9 classes and covers sequential unlearning requests. 
We evaluate existing unlearning methods and find that sequential unlearning severely degrades model utility and forget quality.
To address this challenge, we propose an efficient method called LUMoE, which leverages switchable LoRA adapters through a gate module, eliminating the need for incremental training.
Experiments demonstrate that LUMoE significantly outperforms baselines in both model utility and forget quality without degradation. 
Source code and the MLUBench dataset are presented in this anonymous [URL](https://anonymous.4open.science/r/Lifelong_Unlearning_main-72EC/).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
They propose MLUBench. It evaluates the performance of unlearning in a lifelong setting, where multiple unlearning requests happen sequentially. They reveal that typical unlearning approaches (GA, NPO etc) fail in this setting, and propose a MoE-inspired method called LUMoE, which leverages switchable LoRA adapters, each of which is responsible for a refusal against the corresponding task, achieving high forget quality without utility degradation.

### Strengths
1. **Extensive benchmark coverage**: Although comparison with related works is insufficient (detailed in weakness), their proposed benchmark is pretty extensive in terms of size and domain coverage.

2. **Diverse analysis**: The authors conduct extensive analysis ranging from adversarial robustness and key component ablations.

3. **Reasonable task design**: Evaluation of “lifelong” unlearning is practically important. I think it is also practical that they aim at evaluating the unlearning of pretrained knowledge (instead of the more common yet easier finetuned knowledge). Their definition of forget quality (focusing on refusal and not counting hallucinating answers as a success, described in Section 6.1.1) is reasonable as well.

### Weaknesses
Despite their benchmark contributions seeming meaningful, I find myself slightly doubting them due to their inadequate presentation.

1. **Limited related work coverage on multimodal unlearning** (Section 2, line 112-121): First, the authors compare their proposed MLUBench with MMUBench (and FIUBench in the appendix), but comparisons with other recent ones, such as MLLMU-Bench [1], are missing. Second, they argue as if Liu et al.’s SIU is the only method, but I am at least aware of MMUnlearner [2] as another method. Third, PULSE [3] reports a similar performance drop in sequential unlearning; the authors can clarify how their contribution relates to it. In summary, the related work survey feels incomplete. Survey papers (e.g., [4]) and some "awesome-unlearning" GitHub repos offer a more extensive coverage of multimodal unlearning works that the authors could build on.

2. **License concern** (Section 4.1, line 210): The authors mention that they crawled images from Google Images. I like its wide domain coverage, including e.g., cartoons, but did the authors confirm that the images used have no license issues? I suspect this has been one of the bottlenecks in creating extensive benchmarks (i.e., realistic unlearning target domains are protected by copyright), so I encourage them to elaborate on it.

3. **Mismatch between evaluation scheme and baseline** (Table 1): Based on their task design, a hallucinating answer may not be considered as a successful unlearning (line 329-330). To achieve a high unlearning score in such a design, LUMoE conducts preference optimization w.r.t. the refusal response set (Section 5.2 and Table 19).  I like this design itself, but if the authors go in this direction, the baseline approach should be some DPO or PO-based approach (perhaps with some regularizer to retain its utility) that explicitly guides the model to refuse forget set queries. I don’t doubt the effectiveness of their proposed LUMoE approach, but GA, NPO, etc, (which only decrease the likelihood of correct output without an explicit positive guidance toward refusal) have an inevitable issue of hallucination, and are not the best baseline to compare against.

4. **Generalizability beyond Llava evaluation** (Section 4.1, line 238): The content of the MLUBench is selected such that LLaVA models can answer them perfectly, enabling an unlearning efficiency measured from an initial 100% accuracy if (and perhaps only if) users want to evaluate LLaVA. Do other LMMs achieve similarly high pre-unlearning accuracy on MLUBench before unlearning? If not, how should performance be compared across models?

5. **Limited information in ablation** (Section 6.4): Firstly, why is LUMoE not evaluated on TruthfulQA (Appendix D.6)? I think currently it is unclear whether LUMoE can preserve general utility after sequential unlearning. Secondly, while they perform a pretty extensive ablation, most of the results are fully deferred to the appendix, only mentioning e.g., “it does not affect our primary conclusion (line 463)”. While I believe the authors did it mainly because of the space constraint, this makes me feel they are hiding something in the appendix. I would expect to have at least one sentence of quantitative argument for each of the discussions within the main paper.

[1] Liu et al, Protecting Privacy in Multimodal Large Language Models with MLLMU-Bench

[2] Huo et al, MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models

[3] Kawakami et al, PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning

[4] Feng et al, A Survey on Generative Model Unlearning: Fundamentals, Taxonomy, Evaluation, and Future Direction

### Questions
I list up the points that could further improve their presentation. However, these points are relatively minor. Therefore, I would urge the authors to address the points listed in weaknesses first, which could potentially affect their claimed contribution significantly.

1. **Domain-stratified analysis** (Section 6.4): A key claimed contribution is broad domain coverage, but the paper doesn’t analyze whether this breadth yields insights that existing face-focused benchmarks would miss. The only discussion I find somewhat related is the task-ordering ablation (lines 455-459), where the authors concluded that it does not affect the unlearning efficiency. Are there any phenomena that are observable only in some specific domains?

2. **Slightly more sophisticated adversarial attacks** (Section 6.4): Currently, they confirm that the AutoDAN attack does not work. This discussion is useful given the increasing amount of work in adversarial unlearning. However, since LUMoE aims at routing to a refusal path (and not fundamentally removing the knowledge), there should be a more diverse attack vector. In particular, LUMoE may be vulnerable if the routing module (Section 5.2) misroutes and the query is handled by the base model. Can adversaries craft prompts that trigger routing failures and force processing by the base model?

### Soundness
2

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
4

### Summary
This paper addresses the lifelong unlearning problem in Multimodal Large Language Models (MLLMs). To investigate this issue, this paper constructs MLUBench, a new benchmark consisting of 5,105 images and 15,414 Visual Question Answering (VQA) pairs. To solve this task, this paper proposes LUMoE, a novel unlearning method based on the LoRA modules. Experimental results demonstrate that LUMoE significantly outperforms existing approaches, achieving superior performance across multiple evaluation settings.

### Strengths
- S1. This problem setting, which addresses continual unlearning requests on MLLMs, is both practical and important, as it aligns with real-world needs for handling sequential unlearning scenarios.

- S2. A major limitation in this research area is that existing benchmarks often contain only a small number of tasks and are restricted to specific domains, such as facial images. Consequently, the construction of a large-scale VQA dataset in this study represents a certain contribution to advancing research on unlearning in LMMs.

### Weaknesses
- W1. **Lack of the mention of similar work**.  Although this paper states that it is the first to address continual unlearning for MLLMs, similar ideas have been explored in prior research, such as [1]. It would be helpful if this paper could acknowledge and discuss this related work to better position its novelty and contribution within the existing literature.

- W2. **Lack of general-purpose benchmark evaluation.** Although the retain set and forget set are both sampled from the same VQA pool, evaluating performance only on this pool may not be sufficient to support the claim that the model’s overall capability is preserved. It would be valuable for this paper to additionally evaluate the model on a general-purpose benchmark, such as MMBench, by excluding questions similar to those in the forget set. This would provide stronger evidence that the model maintains its generalization ability.

- W3. **The proposed method appears somewhat naive, and not practical.** This paper provides only a shallow analysis of why existing approaches fail, so the proposed LUMoE is the main contribution. However, to my understanding, the proposed approach essentially trains a separate LoRA for each unlearning task and employs a powerful LLM (GLM-4V-Plus) for routing among them. While this design may yield strong empirical results, it also introduces significant operational overhead, since a new set of LoRA parameters must be maintained for each unlearning request, and the routing model depends on a large, resource-intensive LLM. These factors raise concerns about the scalability and practicality of the approach, and the overall method still feels too naive in its design.

- W4. **A deeper analysis would be desirable.** The method proposed in this paper appears to be only a naive solution to the problem. It would be more valuable if the paper provided a deeper investigation into why existing methods fail: for example, a detailed analysis of whether the failure of unlearning stems from the Vision side or the LLM side, or how performance changes when the training parameter (etc., projector, LoRA, full-tuning) is altered. Designing a proposed method built on such an analysis would make this paper much more meaningful.


From these perspectives, I would give this paper a Borderline Reject rating.

[1] Kawakami+, PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning, NeurIPS Workshop 2025

### Questions
- Q1. I ask the authors to emphasize the importance of the LUMoE method from both a practical and an academic perspective.

- Q2. I would like to recommend that the authors add the evaluation results on general-purpose benchmarks.

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
3

### Summary
This paper focuses on sequential (lifelong) unlearning for multimodal large language models (MLLMs). The main contributions claimed in this paper are:

- MLUBench: A new benchmark, which consists of 127 entities across 9 domains, including 5,105 images and 15,414 visual question–answer (VQA) pairs, designed to evaluate the degradation and accumulation effects that occur in continual unlearning scenarios.

- LUMoE: A sequential unlearning method for MLLM based on LoRA-based Mixture-of-Experts (MoE). Each unlearning task is handled by a dedicated LoRA adapter, and a routing (gate) module dynamically assigns the appropriate adapter at inference time. This design enables the model to forget specific knowledge continuously without modifying the base model.

Experimental results on LLaVA-1.6-7B and 13B demonstrate that conventional unlearning methods (e.g., GA, GD, KL, NPO) suffer from severe cumulative forgetting and language degradation, whereas LUMoE maintains both high forget quality and model utility across all sequential unlearning steps.

### Strengths
**S1.** Sequential unlearning is a highly important yet challenging problem, representing one of the key open issues in current machine unlearning research. It is valuable that this paper explicitly focuses on this problem in the context of MLLMs.

**S2.** MLUBench is one of the largest benchmarks for MLLM unlearning, in terms of the number of images and VQA pairs.

**S3.** LUMoE demonstrates that a MoE architecture is highly effective for sequential unlearning in MLLMs, similar to recent findings in sequential unlearning for LLMs and other model architectures.

**S4.** Experimental results show consistently strong performance, significantly outperforming existing unlearning baselines.

### Weaknesses
**W1. Novelty of MLLM Lifelong Unlearning**

This paper claims, as one of its contributions, to define a new problem, "MLLM Lifelong Unlearning." While the problem is indeed important and challenging, a highly similar setting has already been introduced in MUSE (Shi et al., 2024) and [a]. Although the authors cite MUSE, the paper does not explicitly acknowledge or discuss this overlap. 

These prior works focus on LLMs rather than MLLMs, but the conceptual difference appears minor. Moreover, they already demonstrated that existing unlearning methods such as GA, NPO, and their variants fail in sequential unlearning settings, which is the same conclusion presented in this paper. 

This paper should explicitly describe existing attempts on sequential unlearning, including MUSE and [a]. Without such clarification, the claimed novelty becomes ambiguous and may undermine the credibility of the work.


**W2. Novelty of LUMoE**

From a technical perspective, the novelty of LUMoE is rather limited. LoRA-based adapters with routing mechanisms have already been used for sequential unlearning in LLMs [a]. Furthermore, the use of MoE architectures for continual learning or lifelong model editing is well established (e.g., [b]). The key components (e.g., LoRA, learning algorithm, and GLM-4V-Plus) are all existing techniques. Consequently, the contribution of LUMoE lies more in engineering integration than in algorithmic novelty.


**W3. Method Design**

While LUMoE effectively handles sequential unlearning by isolating LoRA adapters per entity, the current routing design activates only a single adapter for each input, as the paper notes "If a request matches multiple existing tasks, any of the corresponding adapters can be routed into the base model." This implies that when multiple unlearned entities co-occur in a single query, the model may still output information associated with the non-selected adapter(s), leading to partial forgetting and potential leakage.


[a] Gao et al., On Large Language Model Continual Unlearning, ICLR 2025.

[b] Wang & Li, LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models, EMNLP 2024.

### Questions
**W1.** This is the most critical issue and should be resolved before the paper can be accepted.

**W2.** Please correct me if there is any misunderstanding.

**W3.** When multiple forgotten entities co-occur in a query, do the authors have any mitigation strategy or evidence showing that partial forgetting or leakage does not occur? Alternatively, is there any reasonable solution to address this problem?

### Soundness
2

### Presentation
2

### Contribution
3
