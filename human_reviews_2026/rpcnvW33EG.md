# Operationalizing Data Minimization for Privacy-Preserving LLM Prompting

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
The rapid deployment of large language models (LLMs) in consumer applications has led to frequent exchanges of personal information. To obtain useful responses, users often share more than necessary, increasing privacy risks via memorization, context-based personalization, or security breaches. We present a framework to formally define and operationalize *data minimization*: for a given user prompt and response model, quantifying the least privacy-revealing disclosure that maintains utility, and propose a priority-queue tree search to locate this optimal point within a privacy-ordered transformation space. We evaluated the framework on four datasets spanning open-ended conversations (ShareGPT, WildChat) and knowledge-intensive tasks with single-ground-truth answers (CaseHOLD, MedQA), quantifying achievable data minimization with nine LLMs as the response model. Our results demonstrate that larger frontier LLMs can tolerate stronger data minimization while maintaining task quality than smaller open-source models (*85.7%* redaction for GPT-5 vs. *19.3%* for Qwen2.5-0.5B). By comparing with our search-derived benchmarks, we find that LLMs struggle to predict optimal data minimization directly, showing a bias toward abstraction that leads to oversharing. This suggests not just a privacy gap, but a capability gap: models may lack awareness of what information they actually need to solve a task.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a framework to formally define and operationalize data minimization, in the context of users sharing personal information to LLMs. For a given user prompt and response model, they quantify the least privacy-revealing disclosure that maintains utility, and propose a priority-queue tree search to locate this. They evaluated this on a diverse set of four datasets and nine LLMs, and show that for the same user prompts, larger frontier LLMs can have stronger data minimization while maintaining task quality. They also show that LLMs are poor predictors of data minimization and have a bias towards oversharing.

### Strengths
- This paper formally operationalizes data minimization in the context of privacy-preserving prompts for LLMs and presents an algorithm for doing this, which is different from existing approaches that mainly focus on detecting personal information and redacting or abstracting it.
- The authors ran a systematic evaluation on a good range of datasets and LLMs and were able to show how more powerful frontier models can be better for data minimization.

### Weaknesses
- It is not clear how to directly apply these findings to real applications, since the data minimization algorithm involves querying and checking the utility for multiple variants of the original prompt that will reveal the personal information, and might also take significant cost and latency.
- It would be good to include more details about the PII annotation, such as how the annotators were chosen, and more information about the amount of consensus. For instance, in table 1, how many examples are there with human consensus at least 0.8?

### Questions
- How could the results be applied to real applications?
- Could more details about the PII annotation be provided?

### Soundness
4

### Presentation
4

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
The paper proposes a framework for reducing privacy leakage in LLM prompting. It first identifies the sensitive spans in the prompt and introduces a priority-queue tree search algorithm that explores combinations of RETAIN / ABSTRACT / REDACT operations on the spans.

### Strengths
1.	The paper addresses a important problem as how to minimize the privacy leakage of LLM prompt.
The paper addresses an important and timely problem — practical data minimization for LLM usage — with clear motivation.
2.	The paper includes comprehensive experimental evaluation with  multiple datasets, models, and attack evaluations.
3.	Although without strict privacy leakage guarantee, the proposed ranking-based evaluation method offers an insightful and practical perspective for quantifying privacy leakage.

### Weaknesses
1. The evaluation of privacy relies on pairwise comparison, lacking of theoretical guarantees.
2. The paper evaluates utility by comparing the generated output with the target response, treating it as a binary classification problem. This oversimplification fails to reflect the continuous performance characteristics of modern LLMs.
3. The practicability of the proposed method remains questionable, as it assumes there is a local deployed LLM while the aim of the paper is to protect privacy of  prompt  when using LLM via API.

### Questions
1. Could you please present how sensitive are the utility threshold gamma? From my point of view, a little gamma may lead to significant performance degradation in LLMs.
2. I am curious about  whether the “utility predictor” can be generalized across tasks or must be retrained for each task?

3.During the search process, are all intermediate prompts exposed to the main LLM? If so, how is privacy leakage mitigated? If using different LLMs, assuming a smaller LlM local deployed, can the obtained privacy-preserved still transfere effectively to the larger main LLM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work addresses the problem of "oversharing" of personal data in LLM prompts. The authors study the data minimization problem as a constrained optimization problem where they want to maximize privacy with a strict constraint on utility degradation. The work aims to find minimal prompts for each model that does not degrade utility. The paper proposes a "Freeze-then-Search" algorithm to find this optimal prompt. The search space consists of span-level transformations: {RETAIN, ABSTRACT, REDACT} where RETAIN is the least private and REDACT is the most private. To obtain an full ordering on the space, the authors utility a privacy comparator model that is a fine-tuned Qwen2.5-7B-Instruct model.  The search is conduced using a priority-queue tree search and terminates once a prompt is found that satisfies the utility constraint. To measure that utility constraint, the authors use a judge model (GPT-4o) to measure whether the answer from the minimized prompt is compared to a reference response.

The work finds these minimized prompts for 9 LLMs (from small models like Qwen-0.5B to large ones like GPT-5) on 4 datasets. The authors find that large models are tolerant of far more minimization than small models. 

The work also tries to understand if LLMs are good predictors of these minimized prompts and find that all LLMs are poor predictors of this. They also find that the predictors exhibit a strong bias towards ABSTRACT rather than using REDACT as in the minimized prompts.

### Strengths
- Paper does a good job of formulating the data minimization problem as a constrained optimization problem including defining a well ordered search space.
- The generated minimized prompts seem useful to make progress on data minimization predictors.
- The insight that models are poor predictors of predicting the minimal prompt required is interesting.

### Weaknesses
- The minimized prompts are model specific.
-

### Questions
- Is it the case that large models require less information is probably because they are able to infer the missing pieces of information better? 
- The "abstract" bias seems interesting. Could it be related to how the model is instructed to generate its minimal prompt?

### Soundness
3

### Presentation
3

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
This paper introduces a framework to define and operationalize the principle of data minimization (PII redaction) for LLM prompting. The authors formulate data minimization as an optimization problem: finding the most privacy-preserving version of a user prompt (by applying a series of actions to sensitive spans) that still maintains a minimum level of task utility for a given response LLM.

### Strengths
The paper is well-written, with motivated and formalized problem settings. The introduction and related work sections do a good job of comparing the data minimization formulation within existing privacy-preserving techniques like DP-training or simple sanitization.

### Weaknesses
1. The most significant weakness is the lack of comparison against simple, non-LLM baselines. The paper compares its computationally expensive oracle to single-pass LLM predictors. However, it fails to include a much simpler and more practical baseline: using a standard PII identification or NER tool to identify and redact PIIs. 
2.  The methodology relies on an LLM-as-a-Judge (GPT-4o) to evaluate utility for open-ended tasks. While the authors commendably validate their privacy comparator LLM against human consensus, the utility judge is not validated at all. The authors provide no data on the judge's accuracy, its agreement with human assessments of utility, or the robustness of its rubric.

### Questions
1. What PII is commonly present in MedQA, a curated medical expert QA dataset? How representative is this of PII that users would actually share, as opposed to PII that is simply embedded in the original exam questions?
2. Could using test-time compute (Gemini Pro Thinking or Claude Extended Thinking) help close the gap in PII removal between single-pass and expensive search-based methods?

### Soundness
3

### Presentation
3

### Contribution
2
