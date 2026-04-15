# Beyond Forecasting: Compositional Time Series Reasoning for End-to-End Task Execution

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 1, 5, 6, 3, 6

## Abstract
In recent decades, there have been substantial advances in time series models and benchmarks across various individual tasks, such as time series forecasting, classification, and anomaly detection.
Meanwhile, compositional reasoning in time series prevalent in real-world applications (e.g., decision-making and compositional question answering) is in great demand. Unlike simple tasks that primarily focus on predictive accuracy, compositional reasoning emphasizes the synthesis of diverse information from both time series data and various domain knowledge, making it distinct and extremely more challenging. In this paper, we introduce Compositional Time Series Reasoning, a new task of handling intricate multistep reasoning tasks from time series data. Specifically, this new task focuses on various question instances requiring structural and compositional reasoning abilities on time series data, such as decision-making and compositional question answering.
As an initial attempt to tackle this novel task, we developed \modelname, a program-aided approach that utilizes large language model (LLM) to decompose a complex task into steps of programs that leverage existing time series models and numerical subroutines. Through a comprehensive set of experiments, we demonstrate that our simple but effective \modelnamespace outperforms existing standalone reasoning approaches. These promising results indicate potential opportunities in the new task of time series reasoning and highlight the need for further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes an unified pipeline to solve time-series analysis task. Authors transcribe operations into custom programs and use in-context learning with GPT models to write programs. Authors demonstrate that this approach performs better than some reasoning baselines such as Chain of Thoughts (CoT).

### Strengths
1. The problem formulation is both practical and innovative. An unified pipeline for timeseries related task is a significant next step towards advancing timeseries foundation models.
2. Effectively decompose reasoning tasks into steps for streamlined execution and evaluation.

### Weaknesses
Method
1. Tree search / Iterative Refinement: Recent work on LLM agents shows that incorporating techniques like tree search or iterative refinement through feedback can enhance an agent's reasoning abilities. It could be useful for the authors to consider implementing these strategies in the future, as they might improve both the compile success rate and overall performance.

Experiment
1. High Standard Deviation in Table. 2: Some rows seem to have very high standard deviation (such as Load Variability Limit in Energy Power). I wonder if this might be due to a small number of evaluation samples or if it’s an aspect of the method itself. It would be helpful to know the number of samples used for each task to better understand these results.

2. Baseline Experiment Details: In the paper, the setup details for baseline experiments are a bit unclear. My concern is whether the baselines have access to the same tool box as TS-Reasoner, such as forecasting models? In addition to access tool boxes, I also wonder if baselines are performed with same number of in-context learning examples used for TS-Reasoner.

3. Token Economics: Since the author used commercial models (GPT) for TS-Reasoner, it would be helpful to provide number of input/output tokens per task evaluated to determine the cost-effectiveness of the pipeline.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper introduces the "Compositional Time Series Reasoning Task". It proposes a TS-Reasoner model that uses an ChatGPT-4-turbo and time series models to create time series forecasting pipelines based on 1) question, 2) program, 3) output, 4) evaluation. While this is a creative concept, the overall contribution seems to lack depth and rigor in advancing either the NLP or time series forecasting fields.

The paper would benefit from a clearer focus on advancing state-of-the-art methodologies, rather than merely leveraging LLMs for the sake of "novelty".

### Strengths
Somehow interesting ChatGPT-4-turbo prompt engineering for time series analysis.

### Weaknesses
1. The paper presents a caricature of a time series analysis task. Their examples on Figure 1 are trivial tasks.

2. The process of arbitrarily requesting the ChatGPT-4-turbo to predict stock prices and receiving an ARIMA output is highly caricaturesque. This illustrates a superficial application of time series models, where the task of forecasting is treated trivially. Similarly, the request for a causal analysis that results in a standard Granger causality output reinforces this pattern of overly simplistic and mechanical responses. These examples undermine the paper's claim to tackle complex reasoning tasks, as the system is merely pairing standard methods with generic prompts, without demonstrating true depth or sophistication in either reasoning or time series analysis.

### Questions
1. The evaluation of the proompt engineering tool is poorly defined. How can an LLM tuned for complex time series reasoning be evaluated with Profit Percent, Risk Tolerance, and budget allocation. What are those metrics measuring, is the model being evaluated on a simulator, or is the model taking decisions on real scenarios? Everything is so high level and unclear.

2. Experiments are mostly limited to making calls to ChatGPT-4-turbo. Is this a real submission to ICLR?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel task in time series analysis for the financial and energy domains, termed "Compositional Time Series Reasoning," designed to address complex, multistep reasoning tasks that integrate time series data with domain knowledge for applications like decision-making and compositional question-answering. Traditional time series tasks (e.g., forecasting, classification, and anomaly detection) emphasize single-task predictive accuracy, whereas the proposed task requires synthesis across multiple reasoning steps. To tackle this challenge, the authors present TS-Reasoner, a program-aided approach that leverages large language models to decompose complex tasks into sequential steps of programs. TS-Reasoner supports the creation of custom modules and can integrate domain knowledge and user-specified constraints, offering flexibility in reasoning tasks. The paper validates TS-Reasoner’s effectiveness on new datasets within finance and energy domains, showcasing improved performance across a series of question types and reasoning requirements.

### Strengths
**S1:** The paper introduces a novel task in time series analysis, where step-by-step reasoning with large language models (LLMs) for compositional reasoning is an innovative approach. 

**S2:** The accompanying video aids readers in understanding the proposed methodology. 

**S3:** The introduction clearly presents relevant concepts, situates the paper’s contributions, and effectively outlines the paper's positioning within the field.

### Weaknesses
**W1:** The technical contribution of the paper appears limited, as the core technique mainly involves adapting Chain-of-Thought (CoT) reasoning to time series tasks without offering any significant advancements.

**W2:** The research is confined to financial and energy domains, and it’s unclear how the proposed method could be adapted to other time series domains, such as weather or traffic. While it may not be necessary for the authors to demonstrate explicit adaptations, the approach lacks generality and seems overly dependent on prompt design.

**W3:** Using LLMs for compositional reasoning in time series tasks presents various challenges, including understanding time series data in text form, selecting appropriate models based on that understanding, executing the models, and verifying the results. These challenges are not discussed or resolved in the paper; the authors instead rely solely on CoT and simple prompts to guide the LLM, with phrases like “Understand and parse the input data from text” (Line 970) and “Choose an appropriate model… Make sure the model is suitable” (Line 971).

**W4:** The functionality of the Task Decomposer is critical, as it involves guiding the LLM in understanding time series and making decisions based on that understanding. However, the authors do not clearly present how this module operates, especially in terms of how the three modules are coordinated.

**W5:** The paper lacks a sufficient number of baselines, including only two, which is too few. Other relevant baselines, such as various time series forecasting models (e.g., statistical models, transformer-based models) and other in-context learning models (e.g., REACT), should be included for a more comprehensive comparison.

### Questions
**Regarding the Methodology:**
- In Line 363, the paper claims that "Time Series Model Modules are grounded in foundation time series models." However, in Appendix C.3, the prompts reveal that the final predictions utilize models like ARIMA and LSTM.

- For model selection, the prompt used by the authors is “Choose an appropriate model: You will select an appropriate time series forecasting model (e.g., ARIMA, LSTM, etc.) based on the input data.” This approach raises two concerns. Firstly, it contradicts the statement about grounding in foundation models (line 363). Secondly, I question the effectiveness of such prompts. As a simple example, could the authors demonstrate which models were ultimately selected, and in what proportions? Based on my experience, the model likely defaults to ARIMA in most cases due to inherent biases in LLMs rather than a true understanding of the data.

**Regarding the Experiments:**

- Although the authors design different tasks and metrics specifically for financial and energy domains, like absolute average profit (AAP) and relative average profit (RAP), I question the effectiveness of the proposed model in handling these tasks. For instance, in line 275, the authors prompt the LLM with specific requirements like: “[1. I want to make at least {profit percent}% profit. 2. I have a risk tolerance of {risk percent}%.” It’s unclear whether the LLM can genuinely understand and resolve these requirements. The omission of a substantial portion of baselines makes it difficult to find evidence supporting the effectiveness of the proposed method.

- For decision-making tasks, the authors omit other key evaluation metrics, such as MSE and RMSE, which are essential for gauging model performance.

### Soundness
1

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
2

### Summary
This paper proposes a new task, Compositional Time Series Reasoning (TSR), which focuses on handling complex, multi-step reasoning tasks on time series data. The paper introduces a novel approach called TS-Reasoner that utilizes LLMs to decompose complex tasks into structured programs and solve these tasks by calling predefined tool modules. Additionally, new datasets have been compiled for evaluation purposes.

### Strengths
**Novel Task Definition** The paper introduces a well-defined and relevant new task, TSR, which addresses a gap in the time series analysis literature by focusing on complex reasoning tasks.

**Interesting Approach** TS-Reasoner is an innovative approach that effectively combines LLMs with program-based decomposition, offering a promising solution for TSR. By leveraging the in-context learning and logical inference capabilities of LLMs while avoiding the direct management of numerical information, this method achieves impressive results.

**Comprehensive Evaluation** The paper presents a thorough evaluation of TS-Reasoner on datasets from two domains, demonstrating its effectiveness and superiority over CoT based methods.

### Weaknesses
**Limited Analysis of Error Cases** While the paper discusses error types, a more detailed analysis of specific error cases and their root causes would provide a deeper understanding of the method’s limitations.

**Dataset Size and Diversification** Although the paper presents evaluations on datasets from finance and energy domains, the size and diversity of these datasets are not clearly specified. It would be beneficial to provide more details about the dataset composition. Additionally, exploring the generalizability of TS-Reasoner to other domains would be valuable.

### Questions
See Weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
Since many real-world tasks require additional reasonings in addition to time series forecasting, this paper introduces the task of compositional time series reasoning and designs a dataset for this task. The task includes three categories: 1) optimizing some objectives using the forecast results, 2) refining forecast results based on some given constraints, and 3) discovering the causal relationship among variables. The authors predefine a set of functions that a LLM can call, and directly prompts the LLM to solve the above tasks using these functions.

### Strengths
1. The paper is reasonably well-written. It is easy to follow and understand its approach and motivations.
2. The tasks proposed in the dataset are realistic and have practical values. The dataset will also be valuable to the community.

### Weaknesses
1. From my understanding, each type of question has a fixed template, where only the numbers indicated by "{}" will differ from question to question. Since this dataset might be valuable if future work decides to use it to finetune language models. I would suggest to ask off-the-shelf LLMs to rephrase the questions to make them more diverse. Currently, the contribution of this dataset is limited.
2. My biggest concern is that the dataset only proposes several types of questions. If there are only several types of questions, why would someone need an LLM to decide which programs to call? There should be some rules or some decision tree that can simply decide which programs should be called first. LLMs should be more beneficial if there are external language information in the task, or the task will benefit from some general world knowledge (e.g., a hot summer will increase ice cream sales). It is not clear from the current paper how applying LLMs will benefit this task. Please include additional case studies and explanations if so.

### Questions
1. Should the LLM have access to the list of the predefined tools and their explanations in the prompt (Section C.2)? If so, you might want to make this more clear in Section C.1.
2. The paper chooses to only prompt the LLM, instead of finetuning an LLM. I think teaching the model to read the time series directly should enable more power from LLMs. It would be interesting if you could show additional results on that.

Minor:
- Youtube link in the paper doesn't work. Please update it to the link in the repo.
- L348 on page 7, broken figure reference.
- L500 on page 10, "it addiditonally introduce ..."
- Some of the figure/table captions end with periods, and some don't. Please make them consistent.
- Table 3 is not captioned correctly: "Table 3: Overall performance of causal relationship recognition."
- L300 on page 6, in the question template, "for the future future length hours." It should be "{future length}". Same for L317, "variable names".

=================================================

Post Rebuttal:
Thanks to the authors for responding to my comments/questions. I will keep my score as 3. 

The big picture this paper works on is quite interesting. This paper has the potential to be above the accept bar, but needs the following components that I do not think can be thoroughly addressed in the rebuttal:
1) Allow the reviewers access to your dataset so we can check the quality manually and see if there are any misunderstandings;
2) Introduce more operations and tasks that requires a true understanding of time series analysis tasks, instead of using LLMs merely as a tool to extract input parameters to a limited set of functions.
3) Finetune the LLM using a time series encoder instead of just tokenizing the time series values as text.

My main criticism on the current version is that there's only a very limited number of tasks, and each task has a very small solution space with only three to four possible operations. Also, I highly suspect the dataset itself is very redundant since finetuning doesn't show any improvement over simple prompting (that's why I want to see the entire dataset). I feel like the current version of this paper is too early-stage, and is more suitable for publishing at a workshop first.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a new task of compositional time series reasoning which involves conducting multistep reasoning from time series data. The proposed approach leverages LLM to decompose a complex task into a series of programs that call specific time-series models, numerical methods or custom generation modules. The experiments conducted across three types of tasks—decision-making, compositional question answering, and causal mining—demonstrate promising results when compared to directly prompting LLMs.

### Strengths
1. The paper studies a new task of time-series reasoning, constructs specific scenarios that cover decision-making, compositional question answering, causal mining, and composes new datasets in finance and energy domains. 

2. The paper proposes an effective method that leverages LLMs for task decomposition, calling dedicated time-series models, numerical methods or custom generation modules. Experiments on multiple tasks show clear improvement over direct LLM prompting.

### Weaknesses
1. It is unclear how scalable the proposed approach is to new tasks and scenarios. The model might not be able to generalize to new tasks without manually-designed prompts for in-context learning. Additionally, the modules studied in the paper are limited and do not cover real-world scenarios, and the errors might increase as the number of modules increases.

2. The compositional question answering on financial time series appears similar to a traditional forecasting scenario. How does the model compare with existing state-of-the-art forecasting model for this task?

3. How to trace errors back to specific steps within the reasoning chain?

4. Are LLM-generated synthetic data for causal mining reliable?

3. In Line 348, there is a typo of "as illustrated in Fig".

### Questions
How does TS-Reasoner compare to o1?

### Soundness
2

### Presentation
3

### Contribution
3
