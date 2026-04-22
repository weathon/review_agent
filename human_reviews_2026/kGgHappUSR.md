# ChartReasoner: Code-Driven Modality Bridging for Long-Chain Reasoning in Chart Question Answering

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Recently, large language models have shown remarkable reasoning capabilities through long-chain reasoning before responding. However, how to extend this capability to visual reasoning tasks remains an open challenge. Existing multimodal reasoning approaches transfer such visual reasoning task into textual reasoning task via several image-to-text conversions, which often lose critical structural and semantic information embedded in visualizations, especially for tasks like chart question answering that require a large amount of visual details. To bridge this gap, we propose ChartReasoner, a code-driven novel two-stage framework designed to enable precise, interpretable reasoning over charts. We first train a high-fidelity model to convert diverse chart images into structured ECharts codes, preserving both layout and data semantics as lossless as possible. Then, we design a general chart reasoning data synthesis pipeline, which leverages this pretrained transport model to automatically and scalably generate chart reasoning trajectories and utilizes a code validator to filter out low-quality samples. Finally, we train the final multimodal model using a combination of supervised fine-tuning and reinforcement learning on our synthesized chart reasoning dataset and experimental results on four public benchmarks clearly demonstrate the effectiveness of our proposed ChartReasoner. It can preserve the original details of the charts as much as possible and perform comparably with state-of-the-art open-source models while using fewer parameters, approaching the performance of proprietary systems like GPT-4o in out-of-domain settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ChartReasoner, a two-stage, code-driven framework designed to enhance long-chain reasoning for MLLMs on Chart Question Answering (ChartQA) tasks. The authors argue that conventional image-to-text methods lose critical information. Their proposed solution first uses a model, Chart2Code, to translate a chart image into a high-fidelity, structured ECharts code representation. This stage is supported by a new synthetic dataset of 110K image-code pairs. In the second stage, this Chart2Code model is applied to existing benchmarks (ChartQA, ChartBench, etc.) to generate code, which is then fed to a long-chain reasoning LLM (DeepSeek-R1) to generate reasoning paths. These paths, filtered by final answer correctness, form a new 140K-sample dataset called ChartThink. The final ChartReasoner model is then trained on this dataset using a combination of Supervised Fine-Tuning (SFT) and Reinforcement Learning (GRPO). The central idea is that code serves as a superior, lossless intermediate modality for complex reasoning.

### Strengths
1. Interesting Idea: The core idea of using executable code (ECharts) as a symbolic, intermediate representation to bridge the visual-textual modality gap is interesting.

2. Substantial Dataset Contributions: The paper introduces two large-scale datasets: Chart2Code for image-to-code translation and ChartThink for code-based reasoning. The creation and release of these resources are a significant contribution to the community.

### Weaknesses
1. High Risk of Train-Test Overlap: This is the most critical issue. The paper states that the ChartThink training dataset is constructed by processing samples from existing benchmarks, including ChartQA and ChartBench. It then evaluates the final model on the test sets of these same benchmarks, labeling them as "in-domain." The paper provides no clarification on whether it excluded the official test splits of these benchmarks during the creation of its training data. Without this explicit separation, there is a high probability of data contamination, where the model has been trained on samples derived from the test set it is being evaluated on. This potential data leakage makes the reported results on these benchmarks unreliable and possibly invalid.

2. Questionable Necessity of the Intermediate Code Step: Although the idea is interesting, the paper's core premise is conceptually questionable. The ability of the Chart2Code model to generate accurate, detailed ECharts code from an image implies that it has already achieved a deep and structured understanding of the chart's components, layout, and data. If the model already possesses this rich internal representation, the necessity of an explicit, separate code-generation step is unclear. It seems plausible that a model with this level of visual parsing capability could be trained to reason directly on its internal representations, making the two-stage pipeline an unnecessarily complex and potentially inefficient detour. The paper fails to justify why this "extra step" is indispensable.

3. Quality of Generated Reasoning Traces: The reasoning paths in the ChartThink dataset, which form the basis for the final model's training, are generated automatically by prompting an LLM (DeepSeek-R1). The only quality control measure is to filter out samples where the LLM's final answer does not match the ground truth. This is a very weak form of supervision. It does not guarantee that the reasoning path itself is correct, logical, human-like, or even the most efficient way to solve the problem. The final ChartReasoner model is therefore trained via imitation learning on potentially flawed, unnatural, or suboptimal reasoning logic, which limits the quality and robustness of what it can learn.

### Questions
1. Can you please explicitly clarify how you handled the data splits from benchmarks like ChartQA and ChartBench when constructing the ChartThink training dataset? Specifically, did you ensure that no data (images, questions, or answers) from the official test splits were used in any part of your training data generation pipeline?

2. Could you provide a stronger justification for the necessity of the intermediate code generation step? If a model is capable of generating accurate code, it already understands the chart's structure deeply. Why is it not possible to train this model to reason directly, and what evidence do you have that the explicit code-based reasoning is superior?

3. Beyond filtering by final answer correctness, what steps did you take to validate the quality, logical correctness, and naturalness of the LLM-generated reasoning paths in the ChartThink dataset? How can you be sure your model is not simply learning to mimic flawed or unnatural reasoning?

### Soundness
1

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
The authors address the problem of chart understanding and argue that existing methods, which convert charts into textual representations, often result in the loss of structured information. To mitigate this, they propose using ECharts code as an intermediate representation. Specifically, they first train a Chart2Code model, which is then used to construct ChartThink inference data. This data is subsequently employed in a two-stage training process for the chart understanding model, resulting in improved performance.

### Strengths
1. The paper is well-written and comprehensive, presenting a clear and detailed methodology.
2. The proposed Chart2Code the ChartThink dataset provide valuable resources for the community.

### Weaknesses
1. The proposed approach is largely incremental and lacks substantial novelty, with the main contribution being the construction of datasets.
2. The performance gains are limited; for example, the method underperforms compared to Chart-R1[1] on ChartQA.
3. Unlike approaches based on Python code, which are more widely applicable, the method relies on ECharts templates. This limits its ability to handle complex or non-standard charts, as well as real-world data not generated with ECharts.
4. The improvement from RL compared to SFT on ChartQA and ChartBench is minimal, and the authors do not provide a discussion of this observation.

[1]: Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner

### Questions
1. Can the authors clarify the key differences between the proposed method and prior approaches?
2. Regarding the ChartThink dataset, what is the evaluation procedure for the reasoning chains?

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
3

### Summary
This paper proposes a method for chare reasoning by converting the charts into code for accurate understanding. It first introduces the Chart2Code to convert charts into ECode, then builds ChartReasoning for reasoning training. The resulted model ChartReasoner achieves the best performance compared with baseline methods.

### Strengths
1. The motivation of this paper is clear, demonstrating the significance of chart reasoning.
2. The performance is good, demonstrating the effectiveness of the method.

### Weaknesses
1. In the Chart2Code stage, how to ensure that the code could preserve all information of the charts that the texts could not do? In the quality filtering stage, will there conduct a comparison between the raw chart and the chart that the code corresponds to?
2. In Fig.12-15, if there lacks digital annotation in the charts, could the LLM generates accurate approximation for the data in the ECode?
3. In the ChartThink construction process, it seems the reasoning process has not been verified, only the answer is checked to be correct.
4. The detailed definition of the reward function in the GRPO has not been clearly introduced.
5. The ablation on SFT and GRPO is supposed to be analyzed.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ChartReasoner, a procedure for training multimodal large language models (MLLMs) to improve their reasoning capabilities on chart question answering tasks. First, the Chart2Code MLLM is trained to translate chart images into Apache ECharts code. Then, the ChartThink dataset is created by pooling chart question answering tasks from four previous datasets, translating the charts to Apache ECharts code using Chart2Code, and using a reasoning LLM to annotate each task with a detailed reasoning trace. Finally, the ChartReasoner model is trained on the ChartThink dataset.

### Strengths
Strengths
- The methodology is presented clearly, with enough details to reproduce the dataset generations and model training.
- The annotated reasoning traces in the ChartThink dataset could be useful for future work.
- The method itself seems intuitive, with the motivation being clear of "bridging" the text-vision modality gap by using code as an intermediate modality.

### Weaknesses
Weaknesses
- It is unclear if translating charts to Apache ECharts code has any tangible performance improvement. There are many existing reasoning LLMs which can take image inputs, including the QvQ-preview model the authors include in their main results. Why not simply pass the chart image itself to these multimodal reasoning LLMs and ask them to generate the reasoning trace?
- The gains from the ChartReasoner training are very minimal over Qwen2.5-VL 7B, which was the model used for finetuning. ChartReasoner only improves Qwen2.5-VL's performance by 1.9% on average across the four datasets, despite training on in distribution data from 2/4 evaluation datasets. 
- The authors mention multimodal long-chain reasoning approaches such as R1-OneVision and Vision-R1 in the related works section, but don't compare their method to these methods.
- The Chart2Code performance comparison is questionable, as there is no indication as to how well GPT-4V is able to judge the faithfulness of a chart rendered from code to the original chart.

### Questions
- Could the authors clarify the advantages of the ECharts framework over other popular plotting libraries such as matplotlib?
- It would be nice to know how many tokens the other models in the main results used on average, to compare with ChartReasoner.
- How does the performance of the SFT + GRPO pipeline compare to a GRPO only pipeline (or in general, how much does training on annotated reasoning traces improve performance)?
- The authors should explicitly state the reward function used during GRPO.

### Soundness
1

### Presentation
2

### Contribution
2
