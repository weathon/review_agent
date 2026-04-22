# FedVLMBench: Benchmarking Federated Fine-Tuning of Vision-Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Vision-Language Models (VLMs) have demonstrated remarkable capabilities in cross-modal understanding and generation by integrating visual and textual information. While instruction tuning and parameter-efficient fine-tuning methods have substantially improved the generalization of VLMs, most existing approaches rely on centralized training, posing challenges for deployment in domains with strict privacy requirements like healthcare. Recent efforts have introduced Federated Learning (FL) into VLM fine-tuning to address these privacy concerns, yet comprehensive benchmarks for evaluating federated fine-tuning strategies, model architectures, and task generalization remain lacking.
In this work, we present FedVLMBench, the first systematic benchmark for federated fine-tuning of VLMs. FedVLMBench integrates two mainstream VLM architectures (encoder-based and encoder-free), four fine-tuning strategies, seven FL algorithms, six multimodal datasets spanning four cross-domain single-task scenarios and two cross-domain multitask settings, covering four distinct downstream task categories.  Through extensive experiments, we uncover key insights into the interplay between VLM architectures, fine-tuning strategies, data heterogeneity, and multi-task federated optimization. Notably, we find that a 2-layer multilayer perceptron (MLP) connector, along with concurrent connector-LLM tuning emerges as the optimal configuration for encoder-based VLMs in FL. Furthermore, current FL methods exhibit significantly higher sensitivity to data heterogeneity in vision-centric tasks than text-centric ones, across both encoder-free and encoder-based VLM architectures. Our benchmark provides essential tools, datasets, and empirical guidance for the research community, offering a standardized platform to advance privacy-preserving, federated training of multimodal foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents FedVLMBench, the first systematic benchmark for federated fine-tuning of Vision-Language Models (VLMs), integrating two architectures (encoder-based and encoder-free), four fine-tuning strategies, six FL algorithms, and six datasets across single-task (VQA, captioning, classification, detection) and multi-task settings in natural and medical domains.

### Strengths
Innovation and Timeliness: FedVLMBench fills a gap in FL-VLMs benchmarks with novel multi-task datasets (Fed-Nature, Fed-Med) and expands task coverage from 2 to 4 types, providing timely value for advanced community research.

Theoretical and Practical Guidance: Findings highlight FL’s sensitivity to heterogeneity in vision-centric tasks and multi-task FL’s ability to mitigate non-IID issues, offering actionable research directions.

### Weaknesses
Motivation Insufficient: The Introduction notes that VLMs’ centralized training fails to meet privacy needs in healthcare and finance, yet lacks experimental validation of FL’s privacy capabilities, resembling a domain migration rather than a problem-driven contribution.

Gaps in Prior Work Lacking Depth: Motivation hinges on limited FL understanding across VLMs and a narrow task focus, essentially an extension of prior work; adding empirical evidence (e.g., why these gaps cause practical issues) would strengthen it.

Dataset Construction Limitations: Some datasets have small client numbers (e.g., 3 for Fed-SLAKE), deviating from typical FL scenarios (10-20+ clients), reducing usability and generalizability.

Limited Model Selection: Experiments rely on small models like LLAMA3.2-3B, lacking support for larger or diverse open-source models, limiting robustness. VLMs' performance is highly scale-dependent; this configuration underestimates real-world performance.

Resembles a Technical Report: The paper focuses on descriptive benchmarks and results but lacks deep theoretical discussion (e.g., why vision tasks are heterogeneity-sensitive), diminishing its research value.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary:

This paper aims to provide a benchmark for federally finetuning VLM, which covers 2 architectures (encoder-based / encoder-free), 4 tuning strategies, 6 FL algorithms, and 6 datasets with different settings. The main findings from the paper are that: 1) for encoder-based VLM, a lightweight (typically 2-layer) MLP connector is the best; 2) joint tuning of connector and LLM works better than two-stage training. 3) Text-centric tasks benefit from LLM tuning while vision-centric tasks need connector tuning. 4) Non-IID hurts vision tasks much more. 5) Federated multi-task can reach near centralized level. Overall, both benchmark and findings are valuable for researchers in the community.

### Strengths
Strength:

1. The paper considered a wide range of downstream tasks, 2 main VLM architectures, 4 mainstream finetuning strategies, and multiple datasets across different domains, which makes the benchmark more comprehensive.
2. While current VLMs are strong on natural image and language tasks, they still underperform on domain-specific tasks like medical imaging. Adapting VLMs faces two real issues: limited data and strict privacy. This paper gives a practical, comprehensive guideline to personalize VLMs for the medical domain using federated fine-tuning.

### Weaknesses
Weakness & question:

1. While the considered VLM approaches are very up-to-dated, most compared baselines are from 5 years ago. Is there any reason for choosing these methods?
2. While the 5 takeaways seems very valuable and reasonable, I'm interested in takeaway 1 - why 2-layer should be better than 6-layer? Do you have any insight or analysis experiments to further explore this phenomenon?
3. In domains that have privacy concerns, like heathcare, the fairness problem is very pronounced.  While this paper focuses on benchmarking VLM finetuning methods, it's worth discussing the fairness issues and potential methods to address them in related works [1-4].
4. Another suggestion is that, non-iid is a general assumption in federated learning, so it would be better to show do takeaways 1,2,3,5 still hold under data heterogeneity. Given the computational limitation, it's fine to just provide some analysis and leave this for future work.
5. Although federated finetune VLM is a good research question in my opinion, adding results on decentralized and centralized training baselines in the main table would help readers to understand the importance of doing federated finetuning. [mainly in table 4 & 5, table 3] 
6. It's also better to provide a figure to illustrate pipelines/workflows of each federated learning strategy; people in VLM or the medical domain may not be familiar with different FL strategies' settings.



[1] Huang W, Ye M, Shi Z, et al. Federated learning for generalization, robustness, fairness: A survey and benchmark[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024, 46(12): 9387-9406.

[2] Zhang F, Shuai Z, Kuang K, et al. Unified fair federated learning for digital healthcare[J]. Patterns, 2024, 5(1).

[3] Wang H, Chen W, Luo X, et al. Toward Fair and Accurate Cross-Domain Medical Image Segmentation: A VLM-Driven Active Domain Adaptation Paradigm[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 24102-24112.

[4] Zeng H, Yue Z, Zhang Y, et al. Fair federated learning with biased vision-language models[C]//Findings of the Association for Computational Linguistics ACL 2024. 2024: 10002-10017.

### Questions
see weakness

### Soundness
3

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
This paper focuses on benchmarking the integration of Vision-Language Models (VLMs) with Federated Learning (FL) and identifies several key limitations in existing benchmarks: the limited range of VLM architectures studied, the restricted task diversity, and the lack of systematic investigation into the relationships among connectors, LoRA, and data heterogeneity. To address these issues, the paper explores two types of VLM architectures—encoder-based and encoder-free, and extends the benchmark to include a wider variety of downstream tasks, such as report generation. It further examines different fine-tuning combinations of connectors and LoRA, revealing several important experimental phenomena and insights.

### Strengths
This benchmark is meaningful as it incorporates a wider variety of VLM architectures, a more diverse set of tasks, and different fine-tuning strategies for VLMs under the Federated Learning (FL) framework. Such a design enhances the comprehensiveness and practical value of the evaluation, providing valuable insights and contributions to the research community.

### Weaknesses
Although the benchmark is meaningful, it still has several shortcomings:

1. Experimental aspect: It only presents the performance of different VLMs across various tasks and fine-tuning strategies. However, the experimental results, especially under different modes (F-C, F-L, F-CL, F-2stage), show limited differences, and it is unclear whether testing such combinations of fine-tuning strategies is truly necessary for FL;
2. Technical aspect: The work lacks technical innovation. It merely evaluates more tasks and provides conclusions without deeply analyzing the issues that arise in different tasks or proposing any technical solutions to address them;
3. Model aspect: Although it claims to cover both encoder-based and encoder-free VLMs, only two specific models are actually tested, which is insufficient to represent the diversity within these two categories;
4. Federated Learning aspect: Key factors that the FL community cares about, such as heterogeneous environments and varying numbers of clients, are not well explored or reflected in the benchmark.

### Questions
The main issues lie in the fact that this benchmark does not thoroughly identify the limitations across different tasks, VLM types, and fine-tuning strategies, nor does it propose technically innovative solutions to address these shortcomings. Moreover, the performance differences among various fine-tuning strategies are minimal, making it difficult to fully support the conclusions drawn. In addition, the benchmark fails to address the core concerns of the Federated Learning community, and the range of model architectures included remains quite limited.

### Soundness
2

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
5

### Summary
This paper introduces FedVLMBench, a comprehensive benchmark for federated fine-tuning of Vision-Language Models. The benchmark systematically evaluates two VLM architectures (encoder-based and encoder-free), four fine-tuning strategies, six federated learning algorithms, and six multimodal datasets spanning single-task and multi-task scenarios across four downstream task categories. Through extensive experiments, the paper investigates the interplay between architectural choices, fine-tuning strategies, and data heterogeneity, providing empirical guidelines for federated VLM deployment in privacy-sensitive domains.

### Strengths
- The experiments are comprehensive, covering comparisons across model architectures, fine-tuning methods, and heterogeneity levels. 

- The paper is well-organized and clearly written, with strong visual support through well-designed figures and tables.

### Weaknesses
- While incorporating both encoder-based and encoder-free architectures enhances the comprehensiveness of the benchmark, this inclusion appears to be driven by empirical considerations rather than theoretical motivation. The introduction briefly states that existing FL studies mainly focus on encoder-based VLMs, and that encoder-free architectures have recently emerged, but it does not sufficiently justify why this comparison is fundamentally necessary in the federated context. The authors do not clearly provide a conceptual analysis of how the two architectures fundamentally differ under FL constraints. 

- In Table 3, the linear layer does not consistently outperform the 6-layer MLP, which slightly contradicts the statement in Section 5.2. It is recommended that the authors moderate this claim.
- In Section 5.2, the comparison between F-CL and F-2stage lacks detailed analysis. As shown in Figure 4, the two strategies also exhibit distinct behaviors in text-dominant and vision-dominant tasks. It would be beneficial to further analyze the results from the perspective of modality dominance.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3
