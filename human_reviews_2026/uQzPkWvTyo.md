# CauSciBench: A Comprehensive Benchmark on End-to-End Causal Inference for Scientific Research

- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
Large language models (LLMs) are showing increasingly promising progress in accelerating scientific research, yet their ability to facilitate causal inference for scientific discovery remains underexplored.
We introduce CauSciBench, the first comprehensive benchmark to evaluate end-to-end causal inference for scientific research.
CauSciBench comprises 367 evaluation tasks based on 100+ real-world research papers across 9 disciplines, augmented with synthetic scenarios and textbook examples.
CauSciBench is the first to probe the complete causal analysis pipeline, from natural language problem formulation through variable selection and method choice to statistical model implementation and result interpretation---all without any intermediate hints.
We evaluate 6 state-of-the-art models with various test-time scaling techniques, including Chain-of-Thought, Program-of-Thought, and ReAct prompting.
The best-performing OpenAI-o3 with CoT prompting still attains a mean relative error (MRE) of 48.96\% on problems derived from real-world research papers, highlighting a substantial gap between current model capabilities and the demands of research-level causal analysis. 
We call on the community to further explore new methods and rigorous evaluation for building agents that can reliably facilitate causal inference in the context of scientific research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel comprehensive benchmark named CauSciBench , designed to evaluate the ability of large language models (LLMs) to perform end-to-end causal inference tasks in scientific research. The authors point out that existing causal inference benchmarks have limitations; for example, they either focus only on text-based commonsense causal reasoning or only evaluate code implementation after a method is given (like QRData), neglecting the complete analysis pipeline from a natural language problem to the final result. CauSciBench is a benchmark that evaluates the complete causal analysis pipeline , requiring models to start from a natural language description and data to autonomously complete variable selection, method choice, statistical model implementation, and result interpretation. The research results indicate that the primary reasons for model failure include a systematic bias towards using OLS , and that even when the correct method is chosen, there are still serious errors in the specific implementation.

### Strengths
The benchmark combines three complementary data sources: real-world research papers, synthetic data, and textbook examples. This design is very clever, as it makes it possible to not only measure the model's performance but also to diagnose the stages at which the model fails.

The paper not only reports the models' scores but also deeply analyzes their "vulnerabilities". For example, it reveals through confusion matrices that models (especially smaller ones) have an over-reliance on OLS. Furthermore, analysis of Table 4 shows that "incorrect method selection" is the primary driver of high errors. These findings are highly instructive for the future improvement of LLMs.

The paper's structure is clear, and the diagrams and experimental results are very intuitive and straightforward. The authors also provided detailed information in the appendix, including the prompt templates used , the synthetic data generation method , and the technical details of the causal inference methods involved in the evaluation. This greatly enhances the paper's authenticity and credibility.

### Weaknesses
1. This paper designs a benchmark model that focuses on testing the ability of Large Language Models (LLMs) to perform end-to-end causal inference tasks. I do not deny the significance of this design, but I am more concerned with how to solve these problems rather than just discovering them. Therefore, the contribution of this paper might be somewhat diminished. Perhaps the authors should have at least discussed how to use the results found in this paper to guide the correction of large language models.

2. The authors generously acknowledge the limitations of this study, which is good. However, the excessive number of limitations declared suggests that this is still a method in its preliminary stages. As the authors stated, the benchmark primarily focuses on the "Potential Outcomes Framework," mainly focuses on binary treatment scenarios, the evaluation was conducted at pass@1 (i.e., the model passes on a single attempt), etc. The method still has significant room for improvement.

3. The paper uses two main metrics: Method Selection Accuracy (MSA) and Mean Relative Error (MRE), but the measurement by these two metrics still appears rather coarse. MSA is a binary metric; it cannot distinguish between selecting a highly related but incorrect model (e.g., using IPW instead of Matching) and selecting a completely irrelevant model (e.g., using OLS on an IV problem). MRE, while focusing on the estimated value, may also "fail to fully capture the severity of the estimation failure."

### Questions
1. As the authors stated, the core of causal inference is not only calculating an effect value but also verifying the model's assumptions. For example, how to validate the parallel trends assumption for DiD or the instrumental variable validity for IV. It is recommended that future versions of the benchmark add this type of task.

2. Based on the results, CoT prompting is generally superior to direct prompting, but the performance of PoT and ReAct is, on the contrary, unstable. This result seems somewhat counter-intuitive. It is hoped that the authors can provide a more in-depth case study on the specific failure modes of different prompting strategies.

3. The authors mention the evaluation pipeline can "pinpointing key vulnerabilities" (Page 2), but the current analysis in Table 4 is still too coarse-grained, only providing an analysis of the accuracy of method selection. Analyses of variable accuracy, code accuracy, etc., could also be added.

4. The paper claims to evaluate "result interpretation" (in Abstract), but the evaluation metrics (MSA and MRE) seem to only focus on point estimates and methods.

### Soundness
3

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
The paper construct a benchmark aiming for end-to-end causal analysis for scientific research. The data sources covers real-world research paper, synthetic framework, and textbook datasets.
It employs two matrices: Method Selection Accuracy (MSA) and Mean Relative Error (MRE). In the experiment section, 8 LLMs are evaluated in the above three settings.

### Strengths
- It focuses on the causal inference task that is essential to scientific research.
- The proposed benchmark has more broad coverage on data sources and scientific disciplines.
- The empirical results reveals the difficulty and complexity of automating causal inference tasks with LLMs.

### Weaknesses
- It needs more elaboration on the unique challenges behind *end-to-end causal analysis*. Based on the line 53~80, my best understanding of *end-to-end causal analysis* is the pipeline of selecting both *variables* and *method*, and these are highly related to the data preprocessing and model specifications that covered in the existing benchmarks.
- It needs more evaluation matrices to match the stated benchmarking target. Currently, there are only two matrices: Method Selection Accuracy (MSA) and Mean Relative Error (MRE). There are some key steps stated by the paper but not evaluated, for example: "formulation of problems from natural language descriptions", "choice of treatment/effect/confounders", and “result interpretation”. I suggest to employ more advanced evaluating method like LLM-as-a-judge to enrich the set of matrices.
- It needs more diverse querying templates mimicking both expert and non-expert users. The queries are already well formulated causal questions. For example, in figure 2, the query is "What is the effect of education on earnings?" This query has already stated the expected variables. It is not surprising that LLMs can select correct data columns in the .csv files with variable description.
- Implicite data pre-processing in the csv files. Take the example in the figure 2. Some pre-processing is from the original dataset by the experts, like *log of the wage* and *indicator of locations*; another type of pre-processing is from incomplete variable list. For example, variables about family structure (*daded*, *nodaded*, *momed*, and *nomomed*, ..) are not presented. There are about 50 variables in the original dataset [1], while figure 2 only displays 8 variables. If the remaining variables are omitted due to page limitation, it would be much better to provide these information in appendix; If the remaining variables are filtered, then what is the filtering criteria? and how would this influence the difficulty of the tasks? 
- Insufficient discussion on contamination. Although the contamination concern is discussed in section 3 and limitation section. It is unclear how such potential contamination can influence the evaluation results and interpretation. This uncertainty weaken the practical utility of the proposed benchmark.

[1] Using Geographic Variation in College Proximity to Estimate the Return to Schooling

### Questions
Please refer to the concerns stated in the previous section. And also the following questions:
 - In Table 2, textbook datasets has the best method selection accuracy. Is this a signal of perceivable data contamination that memorizing the textbook's method choice? How can we interpret such results?
 - In Table 4, when using incorrect methods, why LLMs are more likely to give lower error in the textbook group? The current explanation is "both methods yield similar results" (line 407). I failed to find out why this could explain the large difference in mean relative errors. I suggest more detailed discussion to avoid potential misunderstanding.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a large-scale benchmark designed to evaluate an LLM's ability to perform end-to-end causal inference in realistic scientific contexts. In contrast to prior benchmarks that focus on specific causal inference queries, CauSciBench focuses an entire causal analysis pipeline, including natural language problem formulation, variable selection, methodological choice, model implementation, and interpretation of estimates.

Contributions:

* A benchmark that requires models to go through an end-to-end inference processs -- identify treatment, outcome, and confounding variables, choose appropriate identification strategies, implement them in Python, and interpret results.
* Integrates  research papers across several scientific domains, supplemented with synthetic datasets generated.
* Introduces quantitative metrics called Method Selection Accuracy (MSA) and Mean Relative Error (MRE), and analyzes model errors across steps of the causal pipeline.
* Comparative study of six leading LLM models under multiple prompting paradigms (Direct, Chain-of-Thought, Program-of-Thought, and ReAct). Results show that the best system a mean relative error high enough to cast doubt on LLM ability to do end-to-end causal inference. 
* The benchmark reveals systematic overreliance on OLS, cascading implementation errors, and limited transfer from synthetic or textbook data to real research data.

### Strengths
The paper is original in framing causal inference as a full end-to-end capability for LLMs, as opposed to focusing on sub-tasks. The quality of the benchmark construction is high, combining real-world research data, textbook examples, and synthetic cases. The clarity of presentation is strong—methodology, examples, and evaluation setup are clearly described, with well-structured figures and prompt templates help with reproducibility. The significance lies in establishing a standardized way to measure whether LLMs can autonomously perform scientific causal analysis.

### Weaknesses
* The evaluation pipeline may not fully capture partial reasoning competence, e.g., correctly identifying treatment/outcome but failing in implementation. 
* The contextualization of failures could be deepened—for instance, more analysis of why models default to OLS or how CoT reasoning collapses under noise.

### Questions
How exactly are the “ground truth” causal effects verified for the real-world paper-derived tasks? Are they always replicated numerically in Python, or are some drawn from published tables without independent validation? Clarifying this would help assess the reliability and reproducibility of the benchmark’s reference values.

You mention two rounds of expert validation for causal queries—could you provide inter-rater agreement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CauSciBench, a benchmark for evaluating large language models (LLMs) on end-to-end causal inference tasks grounded in real scientific research. It claims to assess models across the entire causal inference pipeline—from identifying treatment and outcome variables to implementing and interpreting statistical methods—using real, synthetic, and textbook-based datasets. The results show that even leading models such as OpenAI-o3, have high mean relative errors, implying a large performance gap in research-level causal reasoning.

### Strengths
- Overall, this paper provides a comprehensive evaluation framework for assessing Causal Inference agents in realistic scenarios, effectively complementing various elements that prior works lacked. To perform causal inference as an agent, the system should be able to execute everything from variable selection to modeling independently, which this framework evaluates. 
- Additionally, unlike previous works that relied entirely on synthetic data, this framework includes both real-world scenario developed from research paper, seems to be more extensive than relying only textbook, and synthetic scenarios, which enhances its completeness as an evaluation framework.
- Particularly, the aspect of providing freedom in method selection and implementation within the model is intriguing. While prior works dealt with frameworks limited to specific methods, this approach evaluates whether model implementation is possible without method constraints, making it a scalable framework in terms of LLM reasoning and tool implementation. It appears appropriate for assessing the reasoning and agentic abilities of LLMs that will continue to improve. For example, in Source 2, removing direct constraints on specific methods that prior works imposed and giving models freedom seems to be a reasonable setting for scalably evaluating an agent's reasoning ability.

### Weaknesses
- When generating LLM-generated data, there is a need to introduce filtering or fixation logic to resolve or mitigate hallucination problems, which seems to be absent in this work. The authors claim that for synthetically generated data (Data source 2), they created data scenarios corresponding to synthetic causal effects in Eq (1) through GPT-4o prompting, but it is unclear how they detect, filter, or fix low-quality data.
- Regarding evaluation metrics, I'm not sure if it's necessary to introduce MSA separately in addition to MRE. First, in the definition of MSA, one has to question whether it's optimal design that a candidate method approaching the performance of a reference method receives the same evaluation as a candidate method that doesn't. Even if a method other than the reference method is used, isn't MRE itself what ultimately matters? For a comprehensive analysis, introducing additional different metrics would have been more appropriate. How would one explain a situation where a "wrong method" is selected but achieves a low MRE?
- The interpretive discussion of model behavior remains broad and descriptive, without deeper causal analysis of why specific prompting strategies or architectures succeed or fail across datasets. The study lacks granular failure categorization, providing limited insight into the precise failure modes such as variable misidentification, code logic errors, or assumption mismatches.

### Questions
- It appears that information obtained during real-world dataset generation could be utilized to improve synthetic scenarios. What do you think about this? For example, the authors curate causal queries reviewed by 2 experts in data source 1, which could be used as few-shot examples for synthetic data generation to improve the quality of synthetic data (for instance, to reduce the probability of hallucinated data). Besides this, there seems to be room to apply techniques commonly used in general LLM synthetic data generation.
- As the authors mentioned, when evaluating LLMs in the academic domain, analysis related to data contamination is necessary, which requires reproducibility in terms of data generation. In lines 203-204, although the authors stated that the main focus of this paper is "introducing this dataset," with a bit more effort, it seems this could function not only as a generated dataset but also as an evaluation dataset generation framework. What do you think about this possibility?
- Looking at Appendix F, it states that previously used contexts are incorporated to ensure diversity, but it appears as though the entire history utilized during synthetic data generation is concatenated in the prompt. Is this correct? This approach seems inappropriate as it would be limited in terms of context length and long instruction following once the dataset size to be generated exceeds a certain threshold (perhaps several tens of scenarios would already be enough to hamper instruction following). However, if the paper's main focus is on the generated dataset itself rather than data generation algorithms, I don't consider this a major weakness, which is why I've framed this as a question rather than a weakness.
- The authors stated that the main focus of this paper is the generated dataset itself rather than the data generation algorithm. However, while there is a visual overview of data generation in the main body figures, there doesn't seem to be a visual overview of how LLMs are evaluated using the generated dataset. Given that the authors appear to be highlighting differentiated evaluation elements compared to prior worksas their main contribution, shouldn't a visualization of how the generated dataset is used for evaluation be a priority?

### Soundness
3

### Presentation
2

### Contribution
2
