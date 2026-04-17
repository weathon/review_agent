# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose ViSE (Video-LLM Sycophancy Benchmarking and Evaluation), the first benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, ViSE pioneeringly brings linguistic perspectives on sycophancy into the video domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. Furthermore, we propose two potential training-free mitigation strategies revealing potential paths for reducing sycophantic bias:  (i) enhancing visual grounding through interpretable key-frame selection and  (ii) steering model behavior away from sycophancy via targeted, inference-time intervention on its internal neural representations. Our code is available at https://anonymous.4open.science/r/ICLR26-Video-Sycophancy-7B80.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the sycophancy problem, a well-known hallucination behavior previously studied in LLMs and MLLMs, but focuses on its occurrence in Video-LLMs. The authors introduce VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to measure the severity of sycophancy in Video-LLMs. The benchmark includes 367 videos paired with approximately 6K multiple-choice questions, covering seven distinct sycophancy-inducing prompt scenarios. A subset of the questions is further annotated with eight visual task categories to enable fine-grained analysis. Through extensive evaluation, the authors find that Video-LLMs exhibit strong sycophantic tendencies, influenced by factors such as sycophancy type, visual task, and model scale. To mitigate this bias, the paper explores two training-free strategies, namely key-frame selection and representation steering, both of which demonstrate effectiveness in reducing sycophantic responses.

### Strengths
- The paper is generally well written and easy to follow, presenting its motivation and contributions clearly.
- It provides a comprehensive analysis of sycophancy phenomena in the context of Video-LLMs, addressing this issue for the first time through the introduction of the VISE benchmark dataset.
- The experiments offer valuable insights into model behavior under varying conditions and tasks, evaluating multiple open- and closed-source Video-LLMs with different capacities.
- The authors propose two simple, training-free mitigation strategies to reduce sycophantic bias, which have practical implications for improving model robustness and guiding future model development.

### Weaknesses
- The proposed benchmark is built upon the MSVD, MSRVTT, and NExT-QA datasets. However, these datasets are relatively dated, consist mainly of short video clips, and have been criticized for their limited ability to capture complex temporal and contextual understanding. Prior work has shown that many questions in these datasets can be answered directly by LLMs, indicating potential answer leakage and an overreliance on language priors, or can be solved using only a few frames, suggesting a lack of true video-level reasoning [1,2,3]. This raises concerns about the validity of the proposed benchmark for reliably evaluating sycophancy in Video-LLMs, as it may inherit these known limitations from its source datasets.

    [1] Feng et al., Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?, NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle

    [2] Chen et al., ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos, NeurIPS 2024 Datasets and Benchmarks Track

    [3] Shangguan et al, TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models, ICLR 2025
- Recent research in Video-LLMs has increasingly focused on improving models’ ability to understand long-duration videos. New benchmarks such as LongVideoBench [4] and models like Video-XL [5] have been introduced to address long-context video understanding. The proposed benchmark, however, does not consider such settings. Incorporating samples from long-video datasets and evaluating long-context Video-LLMs could provide a more comprehensive and forward-looking assessment of sycophancy behavior.

    [4] Wu et al., LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding, NeurIPS 2025

    [5] Shu et al., Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding, CVPR 2025

- In light of the previous points, it remains unclear how effective the proposed mitigation strategies are when applied to more challenging video understanding scenarios beyond those included in VISE. For example, the key-frame selection approach relies on sampling only three frames from each input video, which may not sufficiently capture complex temporal or contextual dependencies present in longer or more dynamic videos. Similarly, the representation steering strategy employs mean hidden-state activations computed over the dataset, which may oversimplify model behaviors and fail to generalize across diverse visual contexts or unseen scenarios.
- The evaluation does not include several recent Video-LLMs with strong performance, such as Video-UTR [6], PLLaVA [7], Video-R1 [8] and TW-GRPO [9]. Including these models would provide a more comprehensive and up-to-date assessment. In particular, examining Video-LLMs with reasoning or “thinking” capabilities could yield valuable insights into whether such models exhibit different patterns of sycophantic behavior.

    [6] Yu et al., Unhackable Temporal Rewarding for Scalable Video MLLMs, ICLR 2025

    [7] Xu et al., PLLaVA: Parameter-free llava extension from images to videos for video dense captioning, arXiv 2024

    [8] Feng et al., Video-R1: Reinforcing Video Reasoning in MLLMs, arXiv 2025

    [9] Dang et al., Reinforcing Video Reasoning with Thinking, arXiv 2025

### Questions
- Given the known limitations of the source datasets stated above, how does the benchmark ensure it is measuring true video-level sycophancy? A text-only control experiment could clarify if the visual input is truly necessary to induce the observed behavior.
- The proposed mitigations appear designed for short videos. How would the 3-frame sampling strategy perform on longer, action-dense videos where critical context could be missed? Does representation steering using mean activations negatively impact the model's general performance on standard, non-sycophantic tasks?
- As the field moves toward long-context video understanding, how do the authors expect their findings on short clips to generalize to long-form video scenarios?
- The evaluation does not include recent Video-LLMs with advanced reasoning abilities (e.g., Video-R1, TW-GRPO). Could the authors comment on how these models might behave under the proposed benchmark?
- The authors claim that Qwen2.5-VL 32B and 72B are more robust than the 7B model; however, in Table 1, the 72B version performs worse than 32B for Suggestive Bias, Are You Sure?, and Explicitly Endorse cases. Could the authors clarify this discrepancy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VISE, the first benchmark to systematically evaluate sycophancy in Video-LLMs. It comprises 367 videos and 6,367 questions spanning seven sycophancy types. Across nine variants of six state-of-the-art models, larger scales generally reduce sycophancy, but all remain vulnerable, especially on predictive or causal questions. The authors further propose two training-free mitigation methods: key-frame selection and inference-time representation steering.

### Strengths
- First systematic expose of sycophancy in Video-LLMs, extending the problem from text to dynamic multimodal reasoning.
- Rigorous, reproducible benchmark (VISE) with 367 videos, 6k+ questions, seven bias types, tested across nine model variants.
- Two training-free mitigation tricks: key-frame selection and inference-time steering.

### Weaknesses
- The paper may focus primarily on a limited set of video-LLMs or datasets. This narrow scope can restrict the generalizability of the findings and the robustness of the benchmarking results.
- The paper may focus on static analysis of sycophancy without considering how flattery might evolve over time within a video or across different video contexts.

### Questions
- More evaluations on frontier models are required.

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
The paper introduces a novel benchmark (VISE) for quantifying sycophancy in Video-LLMs. The empirical results highlight consistent patterns linking model scale, bias tone, and reasoning type to sycophantic susceptibility. The paper also proposes mitigation methods such as keyframe selection and representation steering, which show clear quantitative benefit.

### Strengths
**1. Novel benchmark design.**

The paper defines the first video-specific sycophancy benchmark.

**2. Reproducibility.**

The paper provides comprehensive implementation details and an accompanying code repository.

**3. Presentation.**

The paper is well-written, and the main ideas are clearly presented.

### Weaknesses
**1. Lack of general-task performance evaluation.**

The paper does not examine whether the proposed mitigation strategies affect the models’ overall performance on standard video understanding tasks. Without such an evaluation, it remains unclear whether the reduction in sycophancy comes at the cost of general reasoning or factual accuracy. Reporting correlations between sycophancy mitigation and geneal performance would provide a more complete picture of the practical trade-offs involved.

**2. Potential bias in benchmark construction.**

Although the authors state that the CRS is not their primary scope, omitting it from benchmark construction introduces potential bias. Both positive and negative behavioral shifts caused by sycophancy are essential for an objective assessment of model reliability. In addition, the benchmark is constructed based on Qwen2.5-VL-7B, and its data selection overlaps with InternVL 2.5 by 87.8%, indicating a cross-family deviation of over 10%. Such deviation may compromise fair comparisons across different model families. The benchmark should be built upon more consistent issues shared across diverse model families.

**3. Insufficient treatment of CRS in experiments.**

The main paper only reports MSS and relegates CRS analysis to the appendix, without showing how CRS changes after applying the proposed mitigation methods. Prior research [1] suggests that reductions in MSS often coincide with lower CRS, implying that the proposed approaches might also reduce models’ receptiveness to valid corrections. Although the authors state that the CRS is not their primary scope, a balanced experimental discussion of this trade-off is important for ensuring the fairness and completeness of the evaluation. It should not be moved out of the primary scope.

**4. Limited comparison with external baselines.**

The sycophancy mitigation experiments are evaluated only against internal baselines, without incorporating broader methods such as Attention Amplification [2]. Including such established approaches would provide a clearer understanding of how the proposed strategies perform relative to existing techniques for mitigating sycophancy in Video-LLMs.

**5. Readability.**

The Sec. 5 would benefit from clearer visual presentation. Adding a schematic diagram or flowchart illustrating the pipelines of key-frame selection and representation steering would significantly improve readability.

**References:**

[1] Sharma, Mrinank, et al. "Towards Understanding Sycophancy in Language Models." The Twelfth International Conference on Learning Representations.

[2] Li, Shuo, et al. "Have the VLMs Lost Confidence? A Study of Sycophancy in VLMs." The Thirteenth International Conference on Learning Representations.

### Questions
Please refer to "Weaknesses".

### Soundness
2

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
This paper takes the first deep dive into sycophancy in video-language models: the tendency of Video-LLMs to parrot a user’s opinion even when it flatly contradicts the on-screen evidence. The result is a ready-made diagnostic toolkit and a pair of plug-and-play defenses that together push Video-LLMs toward more faithful, bias-resistant reasoning.

### Strengths
1. The authors introduce VISE, the first benchmark built to stress-test this behavior. Spanning 367 videos and 6k+ multiple-choice questions, VISE exposes seven distinct flavors of sycophancy and tracks them across nine model variants (open-source and proprietary). 2. Beyond measurement, the work offers two zero-training fixes—key-frame selection to keep answers visually grounded and inference-time “representation steering” that surgically suppresses ingrained yes-man reflexes. 
3. The paper is well-written and highly reproducible.

### Weaknesses
1. The authors did not analyze the impact of flattery suppression on the model’s initial response accuracy or its general performance.
2. Key frame selection is typically a standard preprocessing step for video inputs, making it difficult to regard this as a technical innovation of the paper.

### Questions
1. What are the proportions of the fine-grained question types in Appendix B? Are they evenly distributed?

2. The ablation settings for the key frame selection algorithm are unfair in terms of sequence length and positional encoding. Could the authors align the selected key frames with the original video frames and simply mask (e.g., replace with all-black or all-white images) the non-key frames instead?

3. Although the neuron interference method is effective against flattery, does it compromise the model’s general reasoning performance?

### Soundness
2

### Presentation
3

### Contribution
3
