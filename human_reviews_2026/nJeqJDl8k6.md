# SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Large language models (LLMs) show promise in
solving scientific problems. They can help
generate long-form answers for scientific questions, which are crucial for
comprehensive understanding of complex phenomena that require detailed
explanations spanning multiple interconnected concepts and evidence.
However, LLMs often suffer from hallucination, especially in
the challenging task of long-form scientific question answering.
Retrieval-Augmented Generation (RAG) approaches can ground LLMs by
incorporating external knowledge sources to improve trustworthiness.
In this context, scientific simulators, which play a vital role in
validating hypotheses, offer a particularly promising retrieval source 
to mitigate hallucination and enhance answer factuality.
However, existing RAG approaches cannot be directly applied for
scientific simulation-based retrieval due to two
fundamental challenges: how to retrieve from scientific
simulators, and how to efficiently verify and update long-form answers.
To overcome these challenges, we propose the 
simulator-based RAG framework (SimulRAG)
and provide a long-form scientific QA benchmark covering climate science and
epidemiology with ground truth verified by both simulations and
human annotators. In this framework, we propose a generalized simulator retrieval interface
to transform between textual and numerical modalities. We further design 
a claim-level generation method that utilizes uncertainty estimation scores
and simulator boundary assessment (UE+SBA) to efficiently verify and update claims.
Extensive experiments demonstrate SimulRAG outperforms traditional
RAG baselines by 30.4\% in informativeness and
16.3\% in factuality. UE+SBA further improves efficiency
and quality for claim-level generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper shows a new RAG framework that lets language models query scientific simulators instead of static text. It turns natural questions into simulator inputs, retrieves numerical results, and uses them to fact-check and refine long answers at the claim level. A benchmark in (1) climate and (2) epidemiology shows clear gains (30% more informative and 16% more factual responses) than standard RAG systems.

### Strengths
The proposed simulator retrieval interface and claim-level uncertainty framework (UE+SBA) together is original combination of ideas that enables grounding in dynamic, quantitative environments. also the paper presentation is easy to follow.

### Weaknesses
Experiments are limited to two domains (climate and epidemiology). so that leaves questions about generalization as other domains are not tested.

### Questions
1. Can the framework generalize across different simulators without re-engineering the retrieval interface, or is it domain-specific?

2. What is the computational overhead of simulator calls compared to standard RAG retrieval, and how does SBA help reduce it quantitatively? Some clarity on this would help readers understand the difference and make balanced understanding

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work improves the drawbacks of Retrieval-Augmented Generation (RAG) by proposing simulation processes for Retrieval and a claim-level generator. Their approach, called SimulRAG, is applied to climate and epidemiology domains. The main challenge that SimulRAG attempts to solve is the issue of failing to construct a correct long answer in the existing RAG-based approach for domain-specific question-answering. In the first process, given an open-ended question, SimulRAG will get simulation outputs as multiple possible contexts. Next, the second process will use a generator to generate a set of multiple atomic claims that form the final answer. The main contribution of the simulation retrieval interface is to leverage a parameter extraction process to be an input for context generation, given a pre-defined set of templates called the simulator output format. The claim-level generation requires a rigorous process of claim evaluation. The authors design the generation process with the following components. First, a metric for claim-level uncertainty estimation leveraged graph construction between answer and claim sets to see which claim has a higher level of confident. The simulator boundary assessment leveraged GPT-4o to see whether a claim parameter is suitable for a simulator. SimulRAG performed multiple rounds of uncertainty estimation (UE) and simulator boundary assessment (SBA) until getting the final answer. The number of rounds is determined by the percentage of claims that need to be verified, called the verification budget. They construct two domain-specific datasets of question-answer with claims, including 200 entities with human-in-the-loop for data annotation. Their reported accuracy shows that the proposed strategy for claim generation outperformed other traditional algorithms on both GPT-4o and Claude 3.5 (Table 1) while it requires only 45% of verification budget to get comparable accuracy with the All RAG configuration.

### Strengths
- The paper is well-written.
- The paper’s proposed research domains are sound and clear. LLMs often struggle with solving long-form questions in specific domains, such as climate and epidemiology.
- The claim-generation process was constructed properly and has potential for applications in other domains.

### Weaknesses
- In Line 245, the authors mentioned that the simulator boundary assessment process used GPT-4 as the judge. This introduces a risk of bias, as a high percentage of cases in the simulator were evaluated as suitable for the claim’s parameters and conditions.
- Lack of details on human-in-the-loop for evaluation data construction. For example, how can authors ensure that humans’ decisions are correct? Is there a protocol that a question/claim will be verified by more than one expert to avoid the bias of a single decision?
- In Table 3, in some configurations, the improvement of UE+SBA appears to be marginal compared to the Uncertainty strategy, with an improvement of over 1%. While it is still valid, authors are recommended to make a case study on when uncertainty claim generators perform better claims than UE+SBA.
- The simulation for the simulator retrieval interface generated output based on predefined templates (L184), which poses the risk for generating answers with a restricted set of templates.

### Questions
- I would like to learn more about the human-in-the-loop process for creating evaluation data. For example, for each question, how many experts are assigned to it?
- This work uses close models for every process. I would like to see how this framework performs when built on open LLMs such as Qwen, Llama models.
- Can you specify how many predefined templates you use for the simulator retrieval interface?

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
Thiswork introduces SimulRAG, a Retrieval-Augmented Generation framework that uses scientific simulators to ground LLMs for long-form scientific question answering. The key contributions include: (1) a generalized simulator retrieval interface for transforming between textual and numerical modalities, (2) a claim-level generation method with uncertainty estimation and simulator boundary assessment (UE+SBA) for efficient verification, (3) benchmark datasets for climate science and epidemiology with simulator-verified ground truth, and (4) experimental validation showing 30.4% improvement in informativeness and 16.3% in factuality over traditional RAG baselines.

### Strengths
THis work uses simulators as retrieval sources which is innovative and addresses a real gap in scientific QA systems. Simulators provide dynamic, quantitative evidence that static text corpora cannot offer. The simulator retrieval interface is well-designed. It can handle the challenging transformation between textual questions and numerical simulator inputs/outputs without requiring fine-tuning. This work tested multiple baselines and the UE+SBA method demonstrates that selective claim verification can achieve near-optimal performance while reducing computational costs by around 50%.

### Weaknesses
First, the statistical significance concern: the claim decomposition process could be better illustrated with examples in the main text. With only 200 questions per domain, the statistical power of the evaluation may be limited. Ane there is no comparison with fine-tuned models on the benchmark datasets. Are there systematic patterns in which types of claims UE+SBA misidentifies? And there is no confidence intervals or significance tests are provided for the main results

### Questions
When does the simulator retrieval interface extract incorrect parameters?
The average 3.6-5.3 claims per answer is relatively small, how does performance scale with longer, more complex answers?
What percentage of errors come from parameter extraction vs. claim verification vs. answer generation?
some minor issues: The sensitivity to these hyperparameters is not analyzed and training/computational costs are not reported
Algorithm 1,  the Merge function is underspecified, how does the merge work? (can provide more details).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper try to use RAG-style approach to conduct long form QA on scientific domain, focusing on queries that could include some scientific projection or simulation numbers. To integrate numerical scientific simulators into RAG pipeline, they propose SimulRAG that grounds LLMs in scientific simulators, rather than static textual knowledge bases, to improve trustworthiness of the results. To handle the complexity of long-form answers, the framework decomposes answers into fine-grained, atomic claims, allowing for more precise, targeted verification and updating via their proposed UE+SBA method.

### Strengths
1. Integrating simulation tools is important contribution to generalize RAG to more applications
2. Valuable new resource on climate modeling and epidemiology domains

### Weaknesses
1. *Mismatch Between Broad Claims and Limited Evaluation Scope*: My first significant concern is the potential overstatement of the framework's applicability. The paper positions itself as a solution for "long-form scientific QA" (line 85), a very broad problem domain. This domain includes numerous existing long-form QA benchmarks based on textual knowledge, such as PeerQA [1] or datasets included in the RAG-QA Arena [2].

However, the paper's empirical evaluation is constrained exclusively to a new benchmark constructed by the authors themselves. This benchmark, covering climate science and epidemiology, is explicitly designed to be answerable using the specific simulators the SimulRAG framework integrates. This evaluation strategy feels somewhat circular, as the framework's core novelty (integrating simulators) is tested only on tasks guaranteed to require them. It remains unclear how SimulRAG would perform on the wider class of scientific QA problems where a relevant simulator does not exist, and grounding must be performed against static, text-based knowledge bases

2. The comparison between the proposed method and existing RAG methods should clearly identify the extra cost needed. For example, the Figure 4's results. SimulRAG needs to sample multiple different answer sets (line 193) in order to do the followup UE-SBA, while existing RAG only sample once. Given nowadays LLM are known to get better performance of test-time scaling, this sampling cost difference could mean a lot. Without clearly compare, the effectiveness of SimulRAG is doubtful.



[1] Baumgärtner, Tim, Ted Briscoe, and Iryna Gurevych. "PeerQA: A Scientific Question Answering Dataset from Peer Reviews." arXiv preprint arXiv:2502.13668 (2025).
[2] Han, Rujun, et al. "RAG-QA arena: Evaluating domain robustness for long-form retrieval augmented question answering." arXiv preprint arXiv:2407.13998 (2024).

### Questions
Can you properly define Scientific Simulator in the overall framework section?

### Soundness
1

### Presentation
3

### Contribution
3
