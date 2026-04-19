# An Intelligent Agentic System for Complex Image Restoration Problems

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 6

## Abstract
Real-world image restoration (IR) is inherently complex and often requires combining multiple specialized models to address diverse degradations. Inspired by human problem-solving, we propose AgenticIR, an agentic system that mimics the human approach to image processing by following five key stages: Perception, Scheduling, Execution, Reflection, and Rescheduling. AgenticIR leverages large language models (LLMs) and vision-language models (VLMs) that interact via text generation to dynamically operate a toolbox of IR models. We fine-tune VLMs for image quality analysis and employ LLMs for reasoning, guiding the system step by step. To compensate for LLMs' lack of specific IR knowledge and experience, we introduce a self-exploration method, allowing the LLM to observe and summarize restoration results into referenceable documents. Experiments demonstrate AgenticIR's potential in handling complex IR tasks, representing a promising path toward achieving general intelligence in visual processing.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces AgenticIR, an intelligent system designed to handle complex image restoration tasks by emulating human-like problem-solving methods. The system operates through five stages: Perception, Scheduling, Execution, Reflection, and Rescheduling. AgenticIR leverages LLM and VLM, using their text generation capabilities to operate a set of IR tools dynamically. It relies on VLMs for image quality assessment and LLMs for step-by-step reasoning, enhancing its adaptability to various IR challenges.

It also incorporates a self-exploration mechanism that generates referenceable summaries from past restoration attempts, which improves its decision-making. Experimental results show AgenticIR’s effectiveness in complex restoration scenarios, highlighting its potential for real-world automated image processing and broader AI applications in visual processing.

### Strengths
1. Human-Centric Design: AgenticIR provides an image restoration approach that mirrors human actions, incorporating processes like reflection and iterative rescheduling into its pipeline. This design enhances action interpretability and facilitates meaningful human interaction with the system.
2. Clear and Concise Expression: The paper presents complex ideas with clarity, accompanied by detailed images and diagrams that enhance comprehension and support the technical explanations.
3. Comprehensive Experiments and Ablation Studies: Thorough experimental evaluations are provided, with well-structured ablation studies for each module. This approach effectively validates the system's design and performance.
4. Illustrative Pipeline Examples: The pipeline is illustrated with specific cases, offering a clear understanding of how each component functions within real-world scenarios.

### Weaknesses
1. Limitations Compared to Optimal Solutions: AgenticIR tends to encounter an early stopping issue, where it may settle on a satisfactory solution prematurely, halting further exploration and potentially missing the optimal outcome. Addressing this limitation is important, and it might be beneficial to add an additional row to Table 4 to reflect this aspect.
2. Insufficient Reporting on Iteration Count and Processing Time: Although the paper emphasizes the role of experiential information and provides illustrative examples, it lacks concrete data on the actual reduction in iterations or time consumption. Including specific metrics on these improvements would strengthen the evaluation of AgenticIR’s efficiency and practical advantages.

### Questions
In addition to the weaknesses mentioned, I have two further questions:

1. Why Not Use VLMs Exclusively Throughout the Pipeline? Given VLMs' strong capabilities in image quality assessment and reasoning, could a VLM-only approach be more efficient or effective for the entire pipeline?
2. Would Online Updates to the Reference Data Benefit the Pipeline? Could implementing real-time updates to the experiential knowledge base further enhance the pipeline’s adaptability and performance?

I hope the authors can make up for the weaknesses mentioned and address these questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address the complex image restoration(IR) problems in real-world scenarios, the authors propose AgenticIR, which is an agent-based system that comprises five stages: perception, scheduling, execution, reflection, and rescheduling. The system leverages a Vision-Language Model (VLM) and a Large Language Model (LLM) to reason and connect these five stages, utilizing an IR toolbox during the execution phase to perform actual image reconstruction.

The three main elements in this process are the LLM, VLM, and the IR toolbox. The VLM primarily analyzes the image quality and summarizes the degradation issues, fine-tuning its capabilities based on existing work. The LLM is responsible for planning and strategizing based on the VLM's results, utilizing GPT-4 and employing a self-exploration method to gain experience with IR problems. The IR toolbox consists of a combination of 3-6 existing models tailored to each type of image degradation.

### Strengths
-	This paper is well-written and easy to follow.
-	The experimental setup is comprehensive, with sufficient ablation and comparative experiments demonstrating the effectiveness of their proposed methods
-	The discovery that execution order is key to restoring complex degraded images is compelling.

### Weaknesses
-	The main concern is that this work resembles several widely-used frameworks (e.g., large language models (LLMs), vision-language models (VLMs) and image restoration models), giving it a predominantly engineering-focused approach. 
-	Additionally, this work appears complex, so providing statistics on the time and complexity involved in a single inference would enhance clarity.
-	Given that LLMs and VLMs often struggle with the issue of 'hallucination,' does this work encounter a similar challenge? If so, how does it address this problem?
-	What are the limitations of the proposed framework?
-	Regarding the point that 'execution order is crucial', are the documents (knowledge) remain consistent during inference across different test samples?

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The proposed AgenticIR system addresses the inherent complexity of real-world image restoration (IR) tasks by emulating a human-like, multi-stage processing workflow. The system operates through distinct phases: Perception, Scheduling, Execution, Reflection, and Rescheduling. It integrates Large Language Models (LLMs) and Vision Language Models (VLMs) into a collaborative framework, allowing text-based interactions to direct the application of specialized IR models. This agentic approach builds on existing multi-modal IR frameworks by dynamically adapting its restoration strategy, actively reassessing and adjusting to handle various complex degradations.

### Strengths
1. The proposed system offers a comprehensive approach to image restoration, addressing a broad range of degradation types through a structured, adaptable methodology.
2. It incorporates human-interaction-inspired insights into the image restoration process, potentially enhancing adaptability and effectiveness in handling complex restoration tasks.

### Weaknesses
1. **Comparison fairness**: The comparative experiments appear to lack fairness, as the baselines (e.g., InstructIR) are designed as unified IR models trained to handle multiple degradation types in a single framework. In contrast, AgenticIR leverages specialized off-the-shelf models for each type of degradation. Therefore, it would be more appropriate to compare AgenticIR to the state-of-the-art models for each specific degradation task rather than to unified restoration models.
  
2. **Efficiency Concerns**: While the system is comprehensive, its workflow is notably complex and lengthy. Compared to regular image restoration models, how efficient is AgenticIR in processing images? This is a critical aspect for real-world applications of image restoration and should be addressed with precise comparative evaluations.

3. **Toolbox Ablation Study**: Lines 159-160 state, "For each degradation, we collect three to six advanced models to build the ‘toolbox’ that the intelligent agent can use." There is no ablation study analyzing the impact of selecting these advanced models on the system’s effectiveness. Understanding the influence of each selected model in the toolbox could provide valuable insights.

4. **GPT-4 Usage and Reproducibility**: AgenticIR uses GPT-4, but GPT-4 lacks a fixed version, which raises concerns about the reproducibility of experimental results. Additionally, there is no ablation study on the effects of using alternative LLMs, particularly open-source options, on performance outcomes.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents an agentic workflow based on LLM/VLMs for image restoration. The agentic system follows how actual humans would process images, consisting of five stages: Perception, Scheduling, Execution, Reflection, and Rescheduling. Since the existing VLMs are not sufficiently capable of analyzing image quality or reasoning about the order of image processing tasks, the VLMs are finetuned and allowed for (self-)exploration to understand the effects of scheduling the image restoration components. Experimental results clearly demonstrate the effects of scheduling with learned experience and the other proposed components.

### Strengths
1. Clear presentation of the benefits of the proposed methodologies. I especially liked Figure 3, 6, 7, and 8, where the authors show dramatic improvements on why choosing a good scheduler for the image restoration components is important, as well as reflection and rollback.

2. Novelty in connecting the human process of IR with LLM-based agentic systems. Though the idea of mimicking the human workflow is being widely adopted more recently, application to image restoration tasks and showing effectiveness is not demonstrated before, to my knowledge.

3. Thorough justification of the design choices and careful experiment designs. Reasons for the proposed workflow and the capabilities the authors are trying to give to the LLM/VLMs are well described, and the evaluations seem to be fairly performed.

### Weaknesses
1. No cost analysis. Using such agentic systems require numerous requests to the LLM/VLM APIs; if the system chooses to perform "Reflection" with the tree search, the worst case scenario would be extremely costly. Compared to the existing image restoration models, the proposed model uses significantly more compute. In this sense, given that many previous works (roughly) match the FLOPs when comparing the restoration quality, one might argue that the comparisons are unfair.

2. Relatively subtle improvements for quantitative metrics (though qualitative improvements look quite significant). I would suggest also adding quantitative measures on the figures so that the readers can compare both aspects with a single glance.

### Questions
1. How are the discovered workflows similar to the original motivation of following the human workflow? For instance, is the subtask scheduling by GPT-4 (w. experience) match the best practices performed by a human? It would be better if the authors could provide more insights or discussions.

2. How does the proposed model perform when there is only a single type of degradation? Does it also perform competitively?

3. What is the criterion for deciding whether Execution step is Failed or Successful? (I might have missed)

### Soundness
4

### Presentation
4

### Contribution
3
