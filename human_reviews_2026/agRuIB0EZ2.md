# Do MLLMs Really Understand the Charts?

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Although Multimodal Large Language Models (MLLMs) have demonstrated increasingly impressive performance in chart understanding, most of them exhibit alarming hallucinations and significant performance degradation when handling non-annotated charts. We argue that current MLLMs rely largely on visual $\textit{recognition}$ rather than visual $\textit{reasoning}$ to interpret the charts, and visual estimation of numerical values is one of the most fundamental capabilities in chart understanding that require complex visual reasoning.
To prove this, we introduce ChartVRBench, a benchmark meticulously designed to isolate and evaluate visual reasoning ability in chart understanding. Furthermore, we propose ChartVR-3B/7B trained with a novel Visual Reasoning Reinforcement Finetuning (VR-RFT) strategy to strengthen genuine chart visual reasoning abilities. Extensive experiments show that ChartVR achieves superior performance on ChartVRBench, outperforming even powerful proprietary models. Moreover, the visual reasoning skills cultivated by the proposed VR-RFT demonstrate strong generalization, leading to significant performance gains across a diverse suite of public chart understanding benchmarks. The code and dataset will be publicly available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates a critical failure mode in Multimodal Large Language Models (MLLMs): their inability to perform visual reasoning on charts that lack explicit numerical annotations. The authors argue that MLLMs excel at text recognition but fail at genuine visual reasoning, which they define as the ability to estimate numerical values from a chart's visual geometry (e.g., axes, scales, and positions).

To address this, the paper makes two primary contributions. First, it introduces ChartVRBench, a new benchmark specifically designed to isolate and evaluate this numerical estimation skill. The benchmark contains both synthetic and real-world charts, all deliberately stripped of explicit data labels. Second, it proposes ChartVR, a series of MLLMs trained with a novel two-stage Visual Reasoning Reinforcement Finetuning (VR-RFT) strategy. Stage 1 ("Activation") uses Supervised Fine-Tuning on distilled Chain-of-Thought (CoT) data to teach the model a structured reasoning process. Stage 2 ("Generalization") uses Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm, to refine the model's precision, guided by a continuous quadratic accuracy reward function.

### Strengths
1. Excellent Problem Formulation: The paper's core premise is strong, clear, and important. It correctly identifies a key weakness in current MLLMs and frames a well-defined research question: are models reasoning or just recognizing? The focus on non-annotated charts is a methodologically sound way to isolate this specific capability.

2. Purpose-Built Benchmark: The creation of ChartVRBench, a benchmark designed specifically to test a single, well-defined skill (visual numerical estimation), is a good scientific practice. The inclusion of both synthetic and real-world data is a positive design choice.

### Weaknesses
1. High Risk of Data Contamination and Invalid Evaluation: The paper states that the training data for both the SFT and RL stages is derived directly from the ChartVRBench dataset itself. The SFT data consists of reasoning traces distilled from ChartVRBench's question-answer pairs, and the RL data consists of samples from ChartVRBench where the SFT model was inconsistent. The paper never explicitly states that it maintained a strict separation between the subset of ChartVRBench used for training data generation and the subset used for final evaluation. This strongly suggests that the model was trained on the same data distribution it was tested on, making the impressive results on ChartVRBench a measure of memorization or overfitting, not generalization. This potential train-test overlap invalidates the paper's primary empirical results.
2. Incremental Methodological Contribution: All the techniques are from prior works.
3. Narrow Evaluation Scope and Unconvincing Generalization Claims: The paper's claim that it imparts a "foundational and highly generalizable" reasoning ability is not sufficiently supported. The evaluation of generalization is restricted only to other chart-specific benchmarks (CharXiv, ChartBench, ChartQAPro). This demonstrates in-domain transfer, but it does not prove that the model has learned a general reasoning skill. To validate a claim about foundational reasoning, the evaluation should have included out-of-domain benchmarks that require similar geometric or logical skills, such as mathematical reasoning.
4. Concerns with Distilled Data Quality and Utility: The SFT stage relies on reasoning data distilled from a powerful teacher model (Qwen2.5-VL-32B). This introduces two issues: i) The student model's reasoning quality is capped by the teacher's ability. It is learning to mimic one specific, potentially flawed or suboptimal, reasoning style. ii) The very fact that a 32B model can generate these reasoning traces (even post-hoc) raises questions about the dataset's utility for training even more powerful models in the future. It creates a potential "ceiling effect" for the task itself.

### Questions
1. Can you please provide an unambiguous confirmation that a strict and held-out test set from ChartVRBench was used for evaluation, and that no part of this test set (images, questions, or answers) was used in any capacity during the SFT or RL data construction phases?

2. Given that VR-RFT is a pipeline of existing techniques (CoT-SFT, GRPO), could you clarify what you see as the core novel algorithmic contribution beyond the application to this specific task and the design of the reward function?

3. How can you substantiate the claim that the model has learned a "foundational" reasoning ability, rather than just a set of chart-specific interpretation skills? Why were no out-of-domain reasoning benchmarks (e.g., math, general VQA) included to test this claim of generalizability?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper examines whether Multimodal Large Language Models (MLLMs) truly comprehend charts, especially non-annotated ones requiring visual estimation of numerical values. It introduces ChartVRBench, a benchmark with 2,453 Q&A pairs across seven chart types, using synthetic and real-world data to test visual reasoning. The authors propose ChartVR-3B/7B, trained with a two-stage Visual Reasoning Reinforcement Finetuning (VR-RFT) method—Chain-of-Thought Supervised Finetuning (CoT-SFT) followed by Group Relative Policy Optimization (GRPO) with a continuous accuracy reward. Experiments show ChartVR outperforming models like Gemini-2.5-Flash on ChartVRBench and generalizing well to other benchmarks.

### Strengths
**Important Topic**: Chart understanding is a crucial area for advancing MLLM applications in real-world data analysis, such as scientific papers and financial reports.

**Comprehensive Training Approach**: The VR-RFT strategy effectively combines SFT and RL, with the continuous quadratic accuracy reward delivering a dense, fine-grained signal to refine visual estimation precision.

**Strong Empirical Results and Generalization**: ChartVR demonstrates superior performance on the new benchmark and transfers well to diverse public datasets, suggesting the cultivated visual reasoning is foundational and not task-specific. The paper includes qualitative case studies highlighting improved reasoning chains.

### Weaknesses
**Limited Novelty**:  I think this is the main weakness of this paper. The core idea of visual reasoning on non-annotated charts is not entirely new; benchmarks like PlotQA (cited in the paper, but several years before) already include such data without explicit annotations. Also, the synthetic data generation relies on established methods like Code-as-Intermediary Translation and Self-Instruct, reducing the perceived innovation in dataset curation.

**Insufficient Comparison to Recent SOTA Models**: The evaluation lacks direct comparisons with stronger open-source chart-specific models like Chart-R1, which employs similar CoT supervision and RL strategies to achieve state-of-the-art results on chart understanding benchmarks—potentially undermining claims of superiority.

### Questions
See Cons.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem that MLLMs fails to undertand charts without annotations. A specific benchmark ChartVRBench is developed to comprehensively evaluate the ablites of MLLMs in such a scenario. Besides, to boost the performance of MLLMs in undertanding annotation-free charts, ChartVR is proposed.

### Strengths
1. The paper is well-organized, clearly introducing the background and motivation.
2. The motivation and problem studied in this paper are significant, highlighting current MLLMs could not understand charts without annoations accurately.
3. The benchmark is necessary for the field of chart understanding, which measures the ability of MLLMs in chart understanding in a specific scenario.
4. The performance of ChartVR on both ChartVRBench and public general chart understanding benchmark is great.

### Weaknesses
1. In Table 2, since MLLMs with size of 3B are compared, the performance of ChartVR-3B on public chart understanding benchmarks is supposed to be compared.
2. The performance of MLLMs on Radar charts is poor and far from human baseline, does it mean the reasoning pattern could not generalize to this type?

### Questions
See the weaknesses.

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
This paper introduces ChartVRBench, a benchmark designed to isolate and evaluate visual reasoning capability in chart understanding. The authors proposed ChartVR, a series of LVLM with significantly enhanced visual reasoning capabilities for chart understanding. The paper tackles an important problem at the intersection of LVLM and data visualization.

### Strengths
- The paper is well-written and easy to follow, with a clear structure.
- The motivation behind the work is interesting and well-presented.
- The proposed method (ChartVR) is technically sound and clearly described.

### Weaknesses
1. The paper is missing a discussion and analysis of the proposed benchmark. This paper claims the benchmark, ChartVRBench, is part of its contribution. However, there is limited analysis of the benchmark's quality or comparison with existing benchmarks.
2. The chart data in the benchmark are mostly synthesized by an LVLM. Without assessing its quality, the performance on the benchmark cannot be supported. Evaluation, such as using human evaluators to check the correctness of the generated answers and the diversity of the generated questions, is highly recommended. Also, using an LVLM to generate data can introduce its own biases; for example, generated charts tend to have monotonic trends (only going upward or downward). One can observe in Figure 5 that the charts seem to have only simple trends, which is largely different from the real-world distribution. How is this addressed in the data generation pipeline? Is there any post-filtering or evaluation to ensure correctness and diversity?
3. The benchmark features charts without annotations. However, existing benchmarks (such as ChartXiv, ChartBench, and EvoChart) also contain charts without annotations. What is the difference between the proposed benchmark and these existing works? What are the novel benefits that this benchmark brings to the community? Is the data higher quality? Does the data cover more chart types? Or is the data aligned more closely with the real-world distribution?
4. A head-to-head comparison is needed to show the effectiveness of the proposed solution. The training of ChartVR involves an RL algorithm (GRPO), which is different from existing supervised finetuning techniques. A head-to-head comparison with previous methods on published benchmarks is necessary to support the effectiveness of the proposed method. For example, the comparison in the table is unfair: ChartVR-7B is built on Qwen 2.5VL, which is a much stronger backbone model than those used in existing approaches. A fairer comparison would involve keeping the backbone models the same as in previous work to showcase the effectiveness of both the proposed dataset and the training algorithm. Additionally, you should fix the model and data for both previous works and your model, then train using the previous SFT algorithm and your RL training to showcase the effectiveness of the proposed training strategies. Without these head-to-head comparisons, I cannot be convinced of the approach's effectiveness.

### Questions
- How are the intermediate steps of the reasoning evaluated on ChartVRBench? I cannot find this in Section 3.2 of the evaluation protocol.

### Soundness
2

### Presentation
3

### Contribution
2
