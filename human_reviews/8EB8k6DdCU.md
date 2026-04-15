# ToolACE: Winning the Points of LLM Function Calling

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Function calling significantly extends the application boundary of large language models (LLMs), where high-quality and diverse training data is critical for unlocking this capability. However, collecting and annotating real function-calling data is challenging, while synthetic data from existing pipelines often lack coverage and accuracy. In this paper, we present ToolACE, an automatic agentic pipeline designed to generate accurate, complex, and diverse tool-learning data, specifically tailored to the capabilities of LLMs. ToolACE leverages a novel self-evolution synthesis process to curate a comprehensive API pool of 26,507 diverse APIs. Dialogs are further generated through the interplay among multiple agents, under the guidance of a complexity evaluator. To ensure data accuracy, we implement a dual-layer verification system combining rule-based and model-based checks. We demonstrate that models trained on our synthesized data---even with only 8B parameters---achieve state-of-the-art performance, comparable to the latest GPT-4 models. Our model and a subset of the data are publicly available at https://huggingface.co/Team-ACE.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
ToolACE sampled API-related data from LLM pre-training corpora, obtaining 26,507 APIs, and used a User Agent - Assistant Agent - Tool Agent structure to synthesize appropriate conversational data to obtain API call training data. Rules + LLM were used during the process to ensure the effectiveness of the synthetic data. Finally, it used an 8B model with LoRA to validate the effects.

### Strengths
1. Ranks third on the BFCL-v3 leaderboard (updated on 09/20/2024), and first among open-source models on API-Bank.
2. The relatively large volume of synthetic data demonstrates the benefits of model fine-tuning at a larger scale.
3. Covers various categories of tool calls including Nested, Parallel, Dependent, and Multi-type.

### Weaknesses
1. The paper shows high complexity with many unclear details. For example, how are the varying complexity levels (easy, medium, hard) actually defined?
2. While the paper repeatedly mentions Nested, Parallel, Dependent, and Multi-type, it doesn't analyze their connection to actual performance or conduct ablation studies.
3. The relationship between data scale and performance is not demonstrated, making it difficult to determine which aspects actually contributed to the effectiveness.
4. There are concerns about potential data leakage between BFCL and TSS - can the authors prove there isn't significant data leakage?

### Questions
1. Since Nested, Parallel, Dependent, and Multi-type are essentially subsets of programming language capabilities, if they are indeed effective, does this suggest that direct training with programming languages (like Python) would be better? Furthermore, is the Data Interpreter[1] approach of directly exposing tool interfaces through Python a better solution? This needs further analysis.

[1] Hong, Sirui, et al. "Data interpreter: An LLM agent for data science." arXiv preprint arXiv:2402.18679 (2024).

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper "ToolACE: Enhancing Function Calling with Accuracy, Complexity, and Diversity" presents a novel data generation pipeline for function-calling tasks LLMs. The approach leverages a tool self-evolution synthesis module, a self-guided dialog generation module, and a dual-layer verification module to create accurate, complex, and diverse tool-calling scenarios. ToolACE aims to improve LLMs' zero-shot function-calling capabilities by generating comprehensive training data that is validated through rule-based and model-based checks. The experiments show promising results, particularly with the ToolACE-8B model, which outperforms several existing LLMs.

### Strengths
- The introduction of ToolACE's multi-step data generation, including evolutionary diversity and self-guided complexity, provides an innovative solution for generating complex and diverse function-calling data.

- The DLV system, combining rule-based and model-based checks, enhances the reliability of the generated data. This is a strong point, as it helps maintain data quality, which is critical for training LLMs effectively.

- The paper provides an extensive set of experiments, including comparisons with state-of-the-art models and an ablation study to assess the contribution of different components like accuracy, complexity, and diversity in the dataset. These experiments illustrate the potential benefits of the proposed pipeline.

### Weaknesses
- The evaluation scenarios are limited to synthetic function-calling tasks and benchmarks like BFCL and APIBank. The paper would benefit from more realistic evaluations or applications in real-world tool usage scenarios. This would better demonstrate ToolACE’s utility beyond controlled benchmark settings.

- The self-guided dialog generation process heavily relies on the LLM being trained to evaluate the complexity of generated data. This creates a circular dependency where the model is used both as a learner and an evaluator, which may introduce bias in the complexity estimation. More external validation or use of independent evaluators would make the results more robust.

- The use of complexity-based sampling to dynamically adjust dialog difficulty has merit but may lead to unintended biases, as data that is either too simple or too complex is filtered out. The approach may fail to fully explore the impact of diverse and extreme cases, leading to gaps in the model’s capabilities in certain contexts.

- While the paper compares ToolACE to several other function-calling models, the comparison is often superficial. The benefits of using ToolACE versus simpler data augmentation techniques are not well articulated, and it is unclear how much of the improvement can be attributed to the synthesis method versus the increased volume of data.

- The paper claims that ToolACE-8B is competitive with GPT-4 series models. However, it does not fully address the limitations of ToolACE-8B in terms of generalization and applicability to a broader range of tasks beyond function calling. A more detailed discussion of these limitations would provide a more balanced perspective.

- The font size in Figures is too small, which is unclear for readers.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Great idea and well written. The authors address the challenges of collecting accurate, diverse, and complex tool-usage data for LLMs by introducing a novel self-evolution synthesis process. ToolACE synthesizes a comprehensive API pool and generates data through agent-based dialogues, guided by a complexity evaluator to ensure the difficulty level is suited to the model's capabilities. The paper presents dual-layer verification (DLV) to maintain data quality, combining rule-based checks and model-based validation. Experiments demonstrate that models trained on ToolACE data achieve state-of-the-art performance in function calling, outperforming models such as GPT-4 in specific benchmarks like BFCL and APIBank.

### Strengths
ToolACE introduces a unique self-evolution synthesis method, which is a systematic approach to generating diverse and complex data for function calling, addressing a key limitation in existing tool-augmented LLMs. The paper provides extensive experiments and ablation studies, comparing ToolACE-trained models with existing benchmarks on widely used datasets like BFCL and APIBank, and demonstrating superior performance.

### Weaknesses
1. Please include a complete example of a prompt and LLM response in the appendix so that readers can intuitively understand how the process works in practice.

2. The paper lacks clarity and involves overly complex technical concepts. Although constructing a simulated dataset and fine-tuning the model are effective approaches to enhancing the LLM's function call capabilities, the additional concepts introduced, such as Self-Evolution, Self-Guided, Dual-Layer, and Multi-Agent, make the main idea harder to discern, leading to confusion for the reader. While the authors may believe these terms add richness to the paper, they detract from its central focus.

3. In the ablation study, it would be valuable to compare the Retrieval-Augmented Generation (RAG) approach for retrieving task-relevant tools with In-Context Learning to optimize tool usage. Given the same level of engineering effort, explore whether these methods could achieve results comparable to fine-tuning.

### Questions
In Figure 3, the "without model" approach occasionally outperforms the "with model" approach. Please provide an analysis to explain the reasons for this phenomenon.

### Soundness
3

### Presentation
2

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
This paper tries to improve the function calling capability of LLM by finetuning on a newly collected function calling dataset. Specifically, the authors propose a pipeline to collect new API usage data. This dataset is then used to finetune llama 8b model and shows comparative performances on two API using benchmarks.

### Strengths
- Collecting new data in a scalable way is important
- The performance looks interesting as well by finetuning a small model

### Weaknesses
The experiment section is not quite convincing yet. Since the authors want to show the effectiveness of using their newly collected API (which according to table 1) is much more comprehensive, the authors should compare the performance obtained by finetuning on Table 1 datasets e.g. ToolLLM and that obtained by finetuning on their newly collected API data

### Questions
- As mentioned in weakness, additional experiment with other baselines should be included e.g. ToolLLM. Even the authors provide some results of xLAM in Table 2, I noticed that they are tuned on different base models other than 8b. So it is hard to draw conclusions and give credit to the dataset itself or to the base models. According to Table 3, 8b seems already very good at API calling evaluations.
- The evaluation and experiment process is not quite clear. e.g. what are the benchmark APIBank, BFCL evaluating? What are their input, output, ground truth, etc? What is the metric used in Table 2,  Table 3?
- Table 2 has many categories in the performances: Single Turn, Multi Turn, Live, Hallucination. Compared to the base model used by author (llama 8b), some categories have only limited performance gain while some could be much higher due to finetuning (Multi turn), thus leading to a higher average score. I don't understand these comparison categories and I don't see authors' analysis in that.

### Soundness
2

### Presentation
2

### Contribution
2
