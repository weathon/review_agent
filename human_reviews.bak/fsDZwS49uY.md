# OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Large language models (LLMs) have exhibited their problem-solving abilities in mathematical reasoning. Solving realistic optimization (OPT) problems in application scenarios requires advanced and applied mathematics ability. However, current OPT benchmarks that merely solve linear programming are far from complex realistic situations. In this work, we propose **OptiBench**, a benchmark for End-to-end optimization problem-solving with human-readable inputs and outputs. **OptiBench** contains rich optimization problems, including linear and nonlinear programming with or without tabular data, which can comprehensively evaluate LLMs' solving ability. In our benchmark, LLMs are required to call a code solver to provide precise numerical answers.
Furthermore, to alleviate the data scarcity for optimization problems, and to bridge the gap between open-source LLMs on a small scale (e.g., Llama-3-8b) and closed-source LLMs (e.g., GPT-4), we further propose a data synthesis method namely ***ReSocratic***. Unlike general data synthesis methods that proceed from questions to answers, \ReSocratic first incrementally synthesizes formatted optimization demonstration with mathematical formulations step by step and then back-translates the generated demonstrations into questions. Based on this, we synthesize the ***ReSocratic-29k*** dataset. We further conduct supervised fine-tuning with ***ReSocratic-29k*** on multiple open-source models. Experimental results show that ***ReSocratic-29k*** significantly improves the performance of open-source models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a study on the challenge of evaluating and improving large language models with respect to solving optimization problems. While the capabilities of large language models have recently improved with respect to mathematical and logical reasoning, the few existing benchmarks are limited to very simple tasks in linear programming and include little complexity representative of real-world applications. The authors present OPTIBENCH-a benchmark that attempts to evaluate the end-to-end ability of LLMs to solve optimization problems. It includes 605 diverse problems in the areas of linear and nonlinear programming, both tabular and nontabular; models are expected to generate code to execute and produce numerical answers.

Motivation: The authors mention that all existing optimization benchmarks are small-scaled and overly simplistic, where most of their problems are linear. Besides, open-sourced LLMs are substantially outperformed by more complex closed-source models like GPT-4. This demonstrates that a more challenging benchmark is needed along with efficient data augmentation methods to make smaller, open-source models perform better.
Main Contributions: OPTIBENCH Benchmark: The authors propose a new benchmark that includes various optimization problems, intended to test the LLM's capabilities in handling optimization tasks of varying complexity. The benchmark problems come in natural language formulation, which requires the generation of Python code that can be executed.
ReSocratic Data Synthesis Method: Based on this, the paper proposes ReSocratic-an approach for attempting to synthesize synthetic data in optimization. Unlike other methods, which first generate questions, the proposed ReSocratic works backward, constructing incremental step-by-step optimization demonstrations, and back-translates those into questions afterwards. This process aims to produce varied and consistent data. In this way, a dataset of questions, called RESOCRATIC-29K, is created.
Performance Improvement: This work presents data suggesting that fine-tuning open-source models over RESOCRATIC-29K might help narrow some of the gap with proprietary models. It reports accuracy improvements from 0.0% to 30.6% for Llama-2-7B-Chat and from 13.6% to 51.1% for Llama-3-8B-Instruct.
Evaluation and Analysis: The authors carried out extensive experiments in which the various models compared against each other in zero-shot, few-shot, and supervised fine-tuning settings. They mentioned nonlinear data and tabular data to be really challenging; increasing problem difficulty considerably raises the challenge, and their experiments indicate the ReSocratic method performs better than the straightforward forward synthesis techniques. Ablation studies further support the influence of key factors such as mechanisms for filtering and step-by-step synthesis.
The paper presents arguments for the potential utility of OPTIBENCH, and examines the effects of ReSocratic on open-source LLMs' optimization capabilities.

### Strengths
Originality: This work introduces the OPTIBENCH benchmark, a comprehensive tool designed to rigorously evaluate large language models (LLMs) on complex optimization problems, significantly advancing beyond existing benchmarks like MAMO and NLP4LP. Unlike prior efforts that primarily focus on simplistic, linear problems or abstract formulations, OPTIBENCH incorporates nonlinear and tabular data, simulating real-world scenarios more accurately. Additionally, the introduction of the ReSocratic data synthesis method is an inventive departure from traditional forward synthesis techniques, employing a reverse step-wise approach that yields a higher quality and diversity of synthetic optimization problems. While similar methods have been applied in other contexts, the adaptation and customization for optimization modeling distinguish this work meaningfully.

Quality: The paper is grounded in rigorous experimental design and analysis. The authors perform extensive evaluations using a variety of open-source and proprietary models, including detailed comparisons in zero-shot, few-shot, and fine-tuning settings. The results are compelling, showcasing substantial performance improvements, particularly when using ReSocratic-generated data, as demonstrated by a 30.6% accuracy improvement for Llama-2-7B-Chat and a 37.5% boost for Llama-3-8B-Instruct. The ablation studies and the analysis of data distribution and variable types further bolster the credibility of the experimental findings, highlighting the robustness and effectiveness of the proposed benchmark and synthesis method.

Clarity: The paper is well-structured and communicates its contributions clearly. The presentation of the OPTIBENCH benchmark, complete with illustrative examples of linear and nonlinear problems, aids in understanding its scope and novelty. The ReSocratic method is also carefully detailed, with step-by-step descriptions and visual aids that effectively convey the synthesis process. Moreover, the inclusion of a thorough comparison with existing benchmarks and methods provides a clear context for the advancements made.

Significance: The significance of this work lies in its practical applicability and potential impact on the field of optimization problem-solving with LLMs. By addressing the limitations of previous benchmarks and providing a more realistic and challenging testbed, OPTIBENCH has the potential to become a standard for evaluating and advancing LLMs in applied mathematics and operations research. Furthermore, the ReSocratic approach offers a scalable method to generate high-quality data, which could influence future research in data synthesis for complex reasoning tasks. The reported performance improvements also suggest practical benefits for enhancing open-source models, bridging the performance gap with proprietary models like GPT-4.

### Weaknesses
## Benchmark Contribution:
The present paper introduces OPTIBENCH for the evaluation of large language models in respect to solving optimization problems. However, such benchmarks already exist, for example MAMO, ComplexOR, and NLP4LP, which test LLMs concerning their ability to interpret natural language descriptions of optimization problems and create corresponding mathematical models. The extension in OPTIBENCH relates to nonlinear elements and tabular data. This feels like an incremental extension rather than a transformative one. The paper should better highlight the aspect of how OPTIBENCH goes significantly beyond the established benchmarks.
## Data Generation Process: 
It proposes the data synthesis methodology, namely ReSocratic, and claims novelty in generating high-quality synthetic data. Earlier research has addressed reverse synthesis methodologies or back-translational techniques, especially for data augmentation and machine translation. Following are some related works which prove similarity:
"Data Augmentation for Code Generation Tasks" shows back-translation-style data augmentation that aims at improving code generation, just like ReSocratic. These techniques create synthetic variations from existing data. There are a number of similarities with this paper which raises the suspicion that the contribution of ReSocratic is more of an extension of the currently known methods and not genuinely new.
MathGenie: Synthetic Data Generation through Question Back-translation to Improve LLMs' Mathematics Reasoning, 2024: MathGenie synthesizes questions from mathematical solutions through a backward process, just like ReSocratic. Be that as it may, both methods are distinguished by their enhancing of reasoning through backward data synthesis, an issue that raises a question as to the novelty of ReSocratic in the given context.
"Data Augmentation for Code Translation with Comparable Corpora and Back-Translation" (2023): This work describes the generation of diverse code examples for translation tasks using back-translation, which aligns with ReSocratic's structured data synthesis approach. In fact, this would mean there is some correlation in the methodological procedure between the current work and existing strategies.
These related works demonstrate that such techniques as back-translation and reverse synthesis have already been successfully applied for code generation and mathematical reasoning. This paper could better explain how ReSocratic is different or improves these established methods. Moreover, discussing the possible biases of the reverse synthesis process would help to enhance the reliability and applicability of the generated data.
## Conclusion:
In general, this paper contains some interesting ideas; however, the contributions of both OPTIBENCH and the ReSocratic data generation method appear relatively modest with respect to prior work. It also shares similarities with approaches that have previously been proposed, indicating potential for better establishing the unique footprint and benefits introduced by the methods. An extended comparative study and some empirical evidence could be more enriching in showing points of strength for the proposed contributions.

### Questions
My major questions are around the data curation process.
# Data Labeling Process Questions

1. Scale of Labeling Team
- Total number of data labelers involved

2. Labeler Qualifications
- Required experience
- Necessary skills
- Background requirements

3. Source Data Details
- Data collection methods
- Criteria for benchmark curation
- Original data sources

### Soundness
3

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
4

### Summary
This paper proposes a benchmark for end-to-end optimization problem-solving and a data synthesis method. The benchmark contains various optimization problems, including linear and non-linear programming with or without tabular data. With the constructed benchmark, the fine-tuned LLMs exhibit a stronger performance.

### Strengths
- This paper proposes a benchmark containing various problems, including linear and non-linear programming with or without tabular data, which can better evaluate the ability of LLMs.
- The reverse data synthesis approach is novel and reasonable.
- The experimental results show that ReSocratic outperforms the forward data synthesis method, and the fine-tuning results are promising.

### Weaknesses
- The authors may want to generate instances with more constraints and variables, as few instances in the paper have more than 7 variables. Thus, this raises my concern about LLMs' ability to model problems with large instance sizes.
- Given that a single optimization problem can have multiple valid formulations, it would be beneficial for the authors to verify the accuracy and equivalence of these formulations with ground-truth ones.
- There are questions regarding the solving efficiency of the generated codes. It would be valuable to assess whether the code produced by LLMs can outperform human-designed formulations and codes.

### Questions
- Could you please provide some results on LLMs handling the large-size problem instances?
- Could you please verify the accuracy of the formulation given by the LLMs?
- Would it be possible to compare the solving efficiency of the codes produced by the LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes OPTIBENCH, an end-to-end benchmark for evaluating large language models (LLMs) on optimization modeling. OPTIBENCH covers various problem types, including linear and nonlinear programming with/without tabular data. To further alleviate data scarcity, the authors introduce ReSocratic, a `reverse` data synthesis method that generates the RESOCRATIC-29K dataset. The authors demonstrated that fine-tuning open-source LLMs (Llama2 7b and Llama3 8B) on RESOCRATIC-29K significantly improves their performance on OPTIBENCH. The paper contributes a comprehensive benchmark, a reverse data synthesis method, and a synthetic dataset to advance evaluations on optimization modeling with LLMs.

### Strengths
OPTIBENCH is a comprehensive benchmark that effectively evaluates the optimization problem-solving abilities. It inlcudes nonlinear programming problems, along with tabular data, reflecting realistic scenarios. It also makes the benchmark challenging enough.
By requiring LLMs to understand the problem, perform sound reasoning, and generate code to invoke a solver, OPTIBENCH provides a holistic assessment of LLMs' reasoning and coding skills. 
This general evaluation approach is valuable for measuring the models' ability to apply their knowledge to practical scenarios and can guide future research on LLMs' capabilities in reasoning tasks. Moreover, the data annotation with strong human guidance looks reliable.

The benchmark evaluates the correctness of LLMs' solutions by comparing the execution results with ground truth answers. This objective evaluation method is more reliable than pure text-based benchmarks in my perspective.

The authors conduct an ablation study that provides valuable insights into the effectiveness of their benchmark and proposed data synthesis method. This includes examining the impact of removing reasoning steps and filters, as well as comparing forward and reverse data synthesis approaches, along with the visualization of the dataset.

This work fills a gap in existing benchmarks by providing a diverse and challenging dataset for end-to-end optimization problem-solving. Moreover, the proposed ReSocratic method and the resulting RESOCRATIC-29K dataset offer a promising approach to address the data scarcity issue.

### Weaknesses
The paper lacks a fine-grained error analysis, which could provide valuable insights into the specific challenges faced by LLMs in optimization tasks. A breakdown of error types, such as errors in understanding the problem, formulating the optimization model, or transferring the mathematical model to code, or summarizing the execution into required formatted outputs, would be helpful to guide future improvements in LLM design and training.

The authors didn't provide an analysis (maybe just a few case studies) of false positives and false negatives in the benchmark results. As with any benchmark, it is expected that both types of errors can exist, and understanding their characteristics could contribute to the benchmark's reliability.

The data collection and annotation process lacks details on deduplication and quality control measures. This helps to ensure the collected optimization problems were diverse enough to measure LLM's modeling capability. 

The paper didn't explore the impact of different prompting strategies during inference. For example, investigating whether step-by-step reasoning then generates the code can help improve model performance.

The range of models tested is relatively limited. As a benchmark paper, one would expect a wider variety of LLMs to be included, such as code-specialized models like CodeLLaMA or the Mistral line of models. 

The reliability of the back-translation step in the ReSocratic data synthesis is not thoroughly investigated. This helps measure the quality and consistency of the questions generated through back-translation, as well as the potential impact of any errors.

### Questions
How does the current perform if using the pass@k metric, which measures the performance of the generated code after k attempts? 

The experimental results show that DeepSeek-V2 severely underperforms in the zero-shot setting compared to its few-shot version, while other models do not exhibit such a drastic difference. Is there any reason behind this?

### Soundness
2

### Presentation
3

### Contribution
3
