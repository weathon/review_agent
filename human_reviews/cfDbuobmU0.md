# SymTex: A New Benchmark for Non-monotonic Reasoning Capability of Large Language Models

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
Non-monotonic reasoning (NMR) plays a crucial role in logical reasoning, allowing inference to adjust as new information arises. This adaptability is key for large language models (LLMs) to handle complex problems and adjust reasoning in dynamic environments, mimicking human-like flexibility in thought. Recent works mainly explore using LLMs to address non-monotonic reasoning through textual logic representation, as LLMs excel in understanding natural language. However, textual logic representation often leads to ambiguity and complexity, especially in complex situations, while symbolic logic representation is more clear and precise, avoiding these issues. In this work, we introduce a framework called Multi-step Generation for Symbolic and Textual NMR Samples (MG-SymTex) to generate diverse non-monotonic samples automatically, and build a non-monotonic reasoning benchmark, called SymTex, which is used to evaluate the non-monotonic reasoning capability of LLMs. SymTex comprises two types of description and three types of predicate, facilitating two primary tasks: Tri-State Boolean Querying and Answer Set Computation. Through our comprehensive evaluations, we demonstrate that state-of-the-art LLMs such as gpt-4o, claude-3.5-sonnet, and o1-mini encounter significant challenges when addressing our proposed benchmark, highlighting the difficulty of non-monotonic reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces SymTex, a benchmark for evaluating non-monotonic reasoning abilities of large language models. It also proposes a framework MG-SymTex, for generating non-monotonic reasoning samples in both symbolic and textual forms. The evaluation includes two tasks: tri-state boolean querying and answer set computation, and the authors analyze the performance of several state-of-the-art LLMs based on the proposed dataset.

### Strengths
(1) Non-monotonic reasoning is critical in logic as it allows models to revise conclusions when new information is introduced, thus the task is worth investigating but is currently underexplored.

(2) The process of creating the benchmark dataset is transparent and easy to follow.

(3) The paper provides detailed analyses for the LLMs' performance, including the impact of symbolic vs. textual representations, which highlights specific challenges faced by LLMs in NMR.

### Weaknesses
(1) The tasks designed in this benchmark seem highly artificial and do not reflect real-world non-monotonic reasoning challenges. For instance, the use of random strings or words as predicates (as in Figure 2) seems unrealistic for reasoning scenarios that LLMs encounter in natural language tasks. This limits the generalizability of the results. Real-world non-monotonic reasoning involves handling contextual and dynamic information, such as conversational changes, rather than simple logical queries over synthetic facts and rules. In other words, it would be necessary for the authors to justify the relevance of the proposed benchmark to real-world scenarios.

(2) The two proposed tasks—tri-state boolean querying and answer set computation—are not clearly defined in practical terms. The descriptions of both tasks lack details on **how the LLMs are supposed to process the input**, leading to confusion about what is being evaluated. For example, it is unclear how the "M" label is determined in Tri-State Boolean Querying, and why this label significantly complicates reasoning tasks. The evaluation metrics for the tasks (wF1, Acc, etc.) are briefly mentioned but not adequately explained. This weakens the reader's ability to assess the relevance of the tasks. More specific descriptions for the tasks should be provided, such as step-by-step examples of how an LLM should process a sample input for each task.

(3) The paper heavily focuses on the technical process of generating the SymTex dataset but does not sufficiently validate the quality or appropriateness of the generated samples. For example, the authors state that they use a tool called DLV2 to check the correctness of symbolic samples, but there is no discussion of how well the samples reflect actual non-monotonic reasoning problems. 

Moreover, there is no attempt to benchmark the dataset against real-world datasets, making it hard to assess whether SymTex genuinely improves the evaluation of NMR in LLMs. It could make it better to compare SymTex against existing NMR datasets, or incorporate human experts to review the quality of the generated data samples.

(4) The novelty of approach is limited. Several recent works, such as $\delta$-NLI and LogicNMR, have introduced benchmarks for logical reasoning, including non-monotonic reasoning. The paper does not clearly differentiate itself from these works, except by offering symbolic samples, which still lacks clear justification. The limited novelty is also reflected in the shallow experimental results, where the findings mostly confirm known limitations of LLMs in handling complex logical tasks.

### Questions
The paper claims to analyze the gap between symbolic and textual reasoning but does not offer an in-depth analysis of why the performance differs significantly between the two. While it notes that there is a 13.5% performance drop in symbolic reasoning for tri-state boolean query, there is no deep dive into the causes. For example, why symbolic tasks pose such a challenge for LLMs is not explored sufficiently. Are the LLMs' errors due to the format of the data, the complexity of the logic, or limitations in their architecture?

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
3

### Summary
The paper introduces the MG-SymTex framework, which can automatically generate a benchmark for non-monotonic reasoning problems with symbolic and textual representations. The framework consists of three stages: first, it generates a template; second, it modifies the template to create symbolic samples; and finally, it textualizes the modified samples. The paper evaluates several state-of-the-art LLMs and finds that they struggle with non-monotonic reasoning tasks. Additionally, it provides a detailed analysis of the extent to which LLMs perform non-monotonic reasoning, the performance gap between symbolic and textual representations, and the influence of predicate descriptions. Lastly, the paper includes an error analysis and examines how new information impacts LLMs' opinions.

### Strengths
(1)	The method introduces a novel framework capable of automatically generating a benchmark for non-monotonic reasoning tasks.

(2)	The paper presents extensive experiments to evaluate the ability of various state-of-the-art (SOTA) LLMs on this task.

(3)	The paper offers a comprehensive analysis of LLMs' performance on non-monotonic tasks, focusing on symbolic and textual representations, the impact of predicate descriptions and new information, and provides a detailed error analysis.

### Weaknesses
(1)	The motivation for testing LLMs' ability in the symbolic format requires further clarification. There are well-developed rule-based solvers for ASP problems that can solve ASP tasks effectively. Therefore, why focus on testing LLMs' ability on the ASP format instead of relying on rule-based solvers? In the analysis, you mention that improving LLMs' understanding of symbolic structures could enhance their ability to translate natural language into symbolic formats and leverage external solvers. However, solving ASP questions is not equivalent to translating natural language into ASP. Please correct me if my understanding is incorrect.

(2)	Further details are needed in Stage 1: Generation. How are the rules generated in this stage? There are various types of symbolic rules, so how do you determine which specific rule to use? Additionally, it is unclear why the modification is necessary. What purpose does modification serve when building the benchmark?

(3)	More details are needed for Stages 2 and 3. In Stage 2 (Predicate Modification), while you modify the predicate, it appears from Figure 2 that the argument (e.g., Tom, Jack) is also modified. However, in Stage 3 (Textualization), those specific arguments revert back to symbols (e.g., Tom becomes name_0). This process is somewhat confusing, and further elaboration is needed in the methodology to clarify the reasoning behind these steps. 

(4)	More information is needed regarding the experimental setup. In Section 5.1.3, what is the main distinction between Subset 2 and Subset 3? Both seem to evaluate non-monotonic reasoning abilities, so why are there two separate subsets for this purpose? While I understand you may want to evaluate different aspects of the LLM, it would be helpful to explain the specific reason for using different subsets and how each one helps you achieve your goal, rather than simply describing how the subsets are constructed and letting the readers figuring it out themselves. Otherwise, readers might be unclear about the purpose of employing different subsets. 

I will adjust my rating accordingly if the author addresses the concerns raised here.

### Questions
(1)	What is the main difference of your benchmark with the previous one? Is it with symbolic language? And what do you mean by pure non-monotonic and why the previous benchmarks are not pure in the introduction?

(2)	At stage 2 in your method, does structure modification refer to modifying the rules template generated at stage 1 and predicate modification refers to the facts from stage 1 as well?

(3)	Why is the argument in stage 2 modified to Tom and Jack but changed back to symbols at stage 3 according to Fig 2?

(4)	What is the purpose of using the three subsets in section 5?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
A new dataset to evaluate the reasoning powers of LLMs is proposed and several state-of-the-art LLMs are evaluated on this dataset. The main purpose is to evaluate whether an LLM can perform non-monotonic reasoning. The framework is based on Answer Set Programming and a symbolic program is generated and then mapped to a textual description. The tasks evaluated are computing a boolean query and computing an answer set for the program. Several state-of-the-art LLMs are evaluated on the generated dataset and conclusions are presented on their performance in the 2 tasks.

### Strengths
Strengths
- A new dataset based on ASP seems like a novel contribution and can have significance and impact in the Neuro-symbolic community
- The experiments seem well thought out and include several LLMs and helps us evaluate logical reasoning in LLMs
- Sharing the datasets will benefit the community

### Weaknesses
Weaknesses
- As far as I understood, existing work (and mentioned in the related work) has focussed on more text-based non-monotonic reasoning capabilities and the main contribution of the proposed dataset is to evaluate if the LLMs can process symbolic representations. What were the results from the earlier work, how does this new work add/modify our views of LLM capabilities? In general, I felt like placing the new dataset in the context of what is known about LLMs using existing work and what new knowledge the dataset and results are adding is important in an empirical paper such as this one.
- In general, why would it make sense for an LLM (built primarily for language) to be evaluated with a symbolic program was not something that was very clear. Related to this, is there a real-world use-case that motivates this usage of LLMs? (There are a couple of sentences mentioned but a more clear motivation would be useful)
- One of the steps in the proposed approach also performs textualization of the program, how sensitive is the LLM to the quality of going from symbolic models to an equivalent text representation?
- Regarding the number of instances, it is mentioned 1000 instances were used (in the experiments) but the dataset is much larger (178K) , is there a reason for this disparity or is there some additional explanations needed here to replicate the results?

### Questions
- Some more detail about the benefit of the proposed method (in relation to the types of evaluation already conducted in a similar domain) would be helpful. (See weaknesses)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces SymTex, a benchmark designed to evaluate large language models' (LLMs) non-monotonic reasoning (NMR) capabilities, where conclusions may be retracted upon receiving new information. SymTex includes both symbolic and textual representations to facilitate an analysis of LLMs' reasoning abilities across these forms. The authors propose a novel data generation framework, MG-SymTex, that generates diverse non-monotonic samples in symbolic and textual formats, enabling a balanced evaluation across forms. Two main tasks are defined for the benchmark: Tri-State Boolean Querying and Answer Set Computation. Comprehensive experiments show that LLMs face substantial challenges with non-monotonic reasoning tasks, particularly in symbolic settings.

### Strengths
1. well-written and good presentation. 
2. interesting benchmark for a "challenging" task: The paper focuses on non-monotonic reasoning. By setting up a dual-format dataset (symbolic and textual), the benchmark provides a balanced framework that probes the models' adaptability in response to new information, an essential aspect of human-like reasoning.
3. sound methodology: The generation process (MG-SymTex) is well-structured, detailing how symbolic samples are transformed into text, ensuring consistency between both formats. This methodological rigor allows for a controlled comparison of LLM performance on both textual and symbolic reasoning tasks, making the dataset a reliable evaluation tool.
4. comprehensive evaluation: The authors evaluate eight LLMs across various reasoning tasks in SymTex, providing detailed analyses and metrics. This analysis reveals the strengths and limitations of models in handling symbolic logic compared to textual logic, offering a clearer understanding of their reasoning capabilities.

### Weaknesses
1. The authors note that LLMs perform better on textual representations than on symbolic ones, but the paper lacks specific solutions or insights into how to address this gap. An in-depth exploration of architectural or methodological changes that could help LLMs better handle symbolic reasoning would add value.
2. One area for potential improvement in the benchmark's symbolic setting is the integration of reasoning interpreters (e.g., ASP solvers) alongside LLMs. While evaluating LLMs independently provides an insightful baseline of their inherent reasoning capabilities, it might also be valuable to explore how these models perform in combination with interpreters. This would reflect a practical, real-world scenario where LLMs often rely on auxiliary tools for enhanced reasoning.
3. Potential Dataset Bias: As with any generated dataset, SymTex may contain unintentional biases due to the template-based generation process. Without rigorous diversity checks, the dataset might fail to capture the full spectrum of possible reasoning challenges, leading models to overfit specific structures. Providing a bias analysis or diversity evaluation for the generated samples would help ensure a robust assessment of LLMs’ reasoning abilities.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
2
