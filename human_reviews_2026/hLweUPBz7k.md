# From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 4

## Abstract
Despite the impressive performance of LLM-powered agents, their adoption for Electronic Health Record (EHR) data access remains limited by the absence of benchmarks that adequately capture real-world clinical data access flows. In practice, two core challenges hinder deployment: query ambiguity from vague user questions and value mismatch between user terminology and database entries. To address this, we introduce EHR-ChatQA, an interactive database question answering benchmark that evaluates the end-to-end workflow of database agents: clarifying user questions, using tools to resolve value mismatches, and generating correct SQL to deliver accurate answers. To cover diverse patterns of query ambiguity and value mismatch, EHR-ChatQA assesses agents in a simulated environment with an LLM-based user across two interaction flows: Incremental Query Refinement (IncreQA), where users add constraints to existing queries, and Adaptive Query Refinement (AdaptQA), where users adjust their search goals mid-conversation. Experiments with state-of-the-art LLMs (e.g., o4-mini and Gemini-2.5-Flash) over five i.i.d. trials show that while the best-performing agents achieve Pass@5 of over 90% (at least one of five trials) on IncreQA and 60–70% on AdaptQA, their Pass^5 (consistent success across all five trials) is substantially lower, with gaps of up to about 60%. These results underscore the need to build agents that are not only performant but also robust for the safety-critical EHR domain. Finally, we provide diagnostic insights into common failure modes to guide future agent development. Our code and data are publicly available at https://github.com/glee4810/EHR-ChatQA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces EHR-ChatQA, a new interactive benchmark for evaluating Large Language Model (LLM) agents accessing Electronic Health Record (EHR) databases, specifically addressing the practical challenges of query ambiguity and value mismatch common in clinical settings. EHR-ChatQA assesses the full end-to-end agentic workflow, including conversational refinement using an LLM-based user simulator, active schema and value tool use, and accurate SQL generation, across two distinct interactive flows: Incremental Query Refinement (IncreQA) and the more adaptive AdaptQA. Evaluations conducted across state-of-the-art closed- and open-source models reveal a critical robustness gap (Pass@5 versus Pass^5), exceeding 35-60% in the challenging AdaptQA flow, demonstrating that current agents, while occasionally capable, are fundamentally unreliable for safety-critical EHR tasks.

### Strengths
- The benchmark is the first interactive evaluation specifically designed for EHR question answering, holistically assessing the full agentic workflow including tool use and conversational refinement.
- Construction of the task instances is rigorously grounded in real-world clinical QA scenarios, utilizing publicly available EHR databases with renamed schemas to enforce genuine exploration rather than memorization.
- A sophisticated simulation environment employs a stochastic LLM user and a dedicated LLM-as-a-judge validator to mitigate simulation noise and ensure evaluation fidelity.
- Evaluation metrics centered on consistent success (Pass^k) and the resulting robustness gap (Gap-k) provide vital diagnostic insights into agent reliability, critical for safety-sensitive applications.

### Weaknesses
- The Adaptive Query Refinement (AdaptQA) flow, which represents the more challenging and novel adaptation scenario, comprises only 64 instances, potentially limiting the statistical robustness of evaluations in this critical area.
- Although the user is simulated by an LLM, its conversational spontaneity is heavily constrained by strict behavioral rules (Table 5), raising questions about the fidelity of the simulation to real clinical dialogue complexity.
- Evaluation relies significantly on extensive, database-specific SQL generation rules (Tables 8 and 9) provided to the agent, suggesting the benchmark may test adherence to prompting constraints rather than intrinsic database reasoning ability.
- Performance of open-source models on AdaptQA is critically low (0.0% Pass^5), which diminishes the utility of the benchmark for diagnosing failures in current open LLM architectures.
- More quantitative diagnostic insights are needed, detailing which specific types of user utterances or tool outputs most frequently trigger the observed dramatic drop in consistent success (Gap-k).
- Lack of transparency regarding the hyperparameters and thresholds used for the `value_similarity_search` tool makes replication or external validation of agent interaction logic difficult.

### Questions
- Given the importance of AdaptQA in revealing the robustness gap, what efforts are planned to expand its size to improve the statistical reliability of results for this difficult and novel task type?
- Could authors provide a more detailed, quantitative breakdown of the error types (Value Linking vs. SQL Generation) specifically correlated with the Gap-k metric, differentiating brittle failures from consistent failures?
- Since simulation validity is crucial, please provide a few detailed examples of dialogues deemed "invalid" by the LLM-as-a-judge validator to better illustrate how the validator enforces the specified user behavior rules.

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
3

### Summary
The authors seek to improve the benchmarking of language model agents in EHR-relevant database tasks, such as clarifying questions, solving value mismatches, and generating accurate SQL. They find that the current LLMs they test struggle with consistent success in this regard. They describe the method of benchmark creation as well as testing, including the use of LLM-based validation.

### Strengths
- I commend the authors for engaging with this particular topic within the Health AI field - optimizing the nuts and bolts of back-end EHR agent interaction will be very important to ensuring these tools are practically useful. Their recognition of the inherent ambiguity and clinical context-specificity of clinician questions is also important. 
- The authors' introduction includes a reasonable summary of the relevant literature in the field and the purpose of their benchmark. 
- The figures are clear and add to the quality of the work. The overall workflow and structure is quite clear. 
- The core QA tasks are reasonable, and the level of human annotation appears overall strong (although I would love further detail on the "38 graduate-level contributors" and what was actually done). 
- I greatly appreciate the depth of analysis the authors offer regarding the nature and presumed causes of errors. This strongly supports the validity of this system as a useful benchmark.
- In general, this paper appears to make a strong contribution to the broader literature.

### Weaknesses
- It is always difficult, when presenting such a benchmark, to know the extent to which the failures reported by the authors (e.g. the worse Pass^5 vs Pass@5) are inherent to the model vs related to flaws in the authors' implementation of that model. While this is inevitable to benchmarking, I do think it should clearly be acknowledged that these results represent a plausible floor, but not at all a ceiling for that which the authors seek to evaluate. 
- Benchmarks using LLMs as user-agents have the same set of concerns, regarding the prospect for the questioner-LLM to inadvertently "tip off" the answerer-LLM, but that does not render them useless in this regard. The authors overall do a good job of highlighting these errors in their section 6, but perhaps this should also be discussed. That is, there are failures which make the test too difficult, but there also may be those which make it too easy.  
- Perhaps more a direction for fture work, but as a clinician I feel that these systems can be further elevated through the use of clinical context itself to improve the questioning process, rather than just general clinical knowledge.
- I am quite concerned, however, about the authors' use of the LLM-as-judge paradigm without any human annotated gold-standard set for whether the validator actually achieves its goals. How are we to know, for example, whether these invalidations are correct, or whether this "validator" is itself discarding valid behaviors? Quis custodiet ipsos custodes? This use of unvalidated LLM-as-judge is very concerning here, and I recommend that either further validation of this approach is offered, or the section is removed.

### Questions
1. Who were these 38 graduate-level contributors? How were they trained? What is their background? What was their specific input?
How was the "LLM-as-judge" validator system itself validated? What were the impacts of implementing this system?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper builds a benchmark called EHR-ChatQA to test how well LLM agents can answer questions over EHR databases in a real conversation, not just single direct questions by focusing on cases where the user starts vague, clarifies over time, or changes their question mid-way — like how clinicians actually talk in real world. The benchmark requires the model to interpret the question, ask clarifying questions, write SQL, and return an answer. The contribution is mostly - a realistic test setup + showing there’s still a big gap in making LLM agents dependable for EHR querying.

### Strengths
I believe the authors thought well and paper is original in framing EHR QA as a conversational, iterative task rather than single-turn SQL. The quality of the benchmark construction is solid, though it could be better validated. The writing is clear, making the setup and results easy to follow. I do think the significance is moderate — it highlights an important reliability gap, but the contribution is mainly diagnostic rather than solution-oriented.

### Weaknesses
I believe the analysis focuses mostly on pass/fail metrics, without deeper breakdowns of failure modes or model behavior, which limits insights into improving performance. Finally, the work emphasizes problem framing rather than proposing strategies to address the reliability gap, so the contribution feels diagnostic rather than forward-moving.

### Questions
Couuld the authors clarify how the proposed method handles edge cases or scenarios where the stated assumptions do not hold?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new benchmark to better capture the real-world complexities of text-to-sql queries on EHR databases. They propose an environment leveraging LLMs that simulate real user interactions. By going beyond static benchmarks, the paper argues to better validate the query ambiguity and value mismatch, and their interactive resolution that are common in practice. 

Disclaimer: I am not well aware of text-to-sql datasets let alone EHR-specific ones. My assessment is based only on the paper's content. It is possible that I missed any important related work.

### Strengths
The paper is well-written and easy to follow. The motivation for EHR text-to-sql is clear and the need for going beyond static benchmarks is also imparted well. 

The paper carefully controlled contamination by changing the names of tables and columns in the database. They also sourced database and queries from real hospitals, which makes the dataset interesting.

### Weaknesses
W1) Details on how they simulated the two user interaction workflows: IncreQA and AdaptQA in Sections 4.2.2, 4.2.3 are not clear to me. The authors should consider elaborating or elucidating with an example. 

W2) I also could not follow how they validated the final response from the interaction. Since the queries can be altered mid-way (in AdaptQA) somewhat arbitrarily, how's the evaluation done? 

W3) In order to establish the reliability of dataset, it is required that the authors report some validation to ensure that the numbers reported in Table 3 are not inflated. In other words, how many of the interactions are misjudged due to failures of user-simulator, user validator, or response validator?  

W4) The proposed benchmark has a heavy system component because they are proposing an environment for validation. But I did not find system setup instructions in the supplementary folder of main paper.

### Questions
Please comment on the questions mentioned in the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
3
