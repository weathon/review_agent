# GSM-Agent: Understanding Agentic Reasoning Using Controllable Environments

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
As LLMs are increasingly deployed as agents, agentic reasoning—the ability to combine tool use, especially search, and reasoning—becomes a critical skill. 
However, it is hard to disentangle agentic reasoning when evaluated in complex environments and tasks. Current agent benchmarks often mix agentic reasoning with challenging math reasoning, expert-level knowledge, and other advanced capabilities.
To fill this gap, we build a novel benchmark, GSM-Agent, where an LLM agent is required to solve grade-school-level reasoning problems, but is only presented with the question in the prompt without the premises that contain the necessary information to solve the task, and needs to proactively collect that information using tools. 
Although the original tasks are grade-school math problems, we observe that even frontier models like GPT-5 only achieve 67\% accuracy.
To understand and analyze the agentic reasoning patterns, we propose the concept of *agentic reasoning graph*: cluster the environment’s document embeddings into nodes, and map each tool call to its nearest node to build a reasoning path. Surprisingly,  we identify that revisit, returning to a previously visited node after leaving--widely taken as a crucial pattern in static reasoning, is a missing ability for agentic reasoning among many models. Based on the insight, we propose a tool-augmented test-time scaling method to improve LLM's agentic reasoning performance by adding tools to encourage models to revisit. We expect our benchmark and the agentic reasoning framework to aid future studies of understanding and pushing the boundaries of agentic reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new benchmark, GSM-Agent, aimed at addressing the difficulty of existing benchmarks in isolating the evaluation of LLM "Agentic Reasoning". The benchmark is derived from GSM8K, decomposing the original problems into a "question" and multiple "premises." The agent only sees the "question" and must use tools to proactively search a database for the hidden "premise" information to solve the task.
Experiments reveal a significant gap between "static reasoning" and "agentic reasoning," with even frontier models achieving only about 67% accuracy. To analyze the reasons, the authors propose the "Agentic Reasoning Graph" framework, which maps agent behavior onto a path on the graph. A core finding is that the 'revisit' ability—returning to a previously visited node—is highly correlated with task success, and this is what many models lack. Based on this, the authors propose a "tool-augmented scaling" method, which encourages 'revisit' by adding new tools, and confirm its effectiveness.

### Strengths
1. **Novel and Controlled Benchmark Design:** The design of GSM-Agent is clever. It successfully decouples "agentic reasoning" (information gathering) from "complex reasoning" (the problem itself). Using grade-school math problems as a base ensures the task's inherent logical difficulty is controlled, allowing the evaluation to truly focus on the agent's information-gathering and decision-making capabilities. This ability to compare "static" vs. "agentic" settings on the same core task is a significant contribution.
2. **Insightful Analytical Framework and Key Finding:** The "Agentic Reasoning Graph" is a novel and powerful analytical tool. Discretizing the agent's trajectory into "exploration," "exploitation," and "revisit" behaviors provides clear, quantitative metrics for understanding agent behavior. This framework leads to the paper's core contribution: identifying "revisit" as a key skill for agent success. This point is strongly supported by the high correlation found between the "revisit ratio" and "accuracy".

### Weaknesses
1. **Generalizability:** The benchmark is built entirely on GSM8K (math problems). While this provides good control, it also limits the generalizability of the conclusions. Is the identified importance of "revisit" equally applicable to other types of agentic tasks? The paper claims to study broad "agentic reasoning," but the evidence is derived from a single domain.
2. **Sensitivity of the Analytical Framework:** The entire "revisit" conclusion is highly dependent on the construction of the "Agentic Reasoning Graph". This construction, in turn, relies on K-means clustering (with $K=250$) and a specific embedding model (`text-embedding-3-large`). The authors do not discuss the sensitivity of these hyperparameter choices. If the value of K were changed or a different embedding model were used, the granularity of the nodes would change. Would the strong correlation between "revisit ratio" and "accuracy" still hold?

### Questions
1. **Regarding Generalizability**: How do the authors view the generalizability of the "revisit" insight to non-mathematical, more agentic tasks? Is "revisit" still expected to be a key bottleneck in those tasks?
2. **Regarding the Robustness of the Analytical Framework:** Can the authors provide ablation studies on the construction of the "Agentic Reasoning Graph"? Specifically, how much does the strong correlation between "revisit ratio" and "accuracy" change if the number of clusters in K-means is varied or if a different document embedding model is used?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper first proposes GSM-Agent, a benchmark designed to evaluate the agentic reasoning of LLMs. The benchmark is adapted from the GSM8K dataset, which decomposes and rewrites each original problem into an under-defined question (task) and relevant documents transformed from the problem premises. To answer the question, LLMs are supposed to use search tools to find the relevant documents from the whole dataset’s document collection. Based on GSM-Agent, this paper then defines an agentic reasoning graph to track the exploration, exploitation and revisit steps in the tool-use traces of LLM agents, based on clustering document embeddings into graph nodes and mapping document searching in the traces into paths on the nodes. Experimental analysis reveals that revisit behavior, i.e., returning to previously accessed documents, is strongly correlated with successful agentic reasoning. Based on that, this paper further proposes a tool-augmented scaling method to encourage the revisit behavior and thus improve reasoning accuracy of LLM agents.

### Strengths
- GSM-Agent provides a controllable testbed for grade school math reasoning, which enables the evaluation to switch between fully-defined question answering and under-defined agentic task solving, which is not covered by prior GSM benchmarks.
- The experimental analysis based on the agentic reasoning graph is insightful, especially, the finding of the strong correlation between revisit behavior and agentic reasoning accuracy has valuable potential to benefit future design of LLM agentic frameworks.
- The experimental study spans a comprehensive set of LLM agents, which avoids model-specific conclusions.

### Weaknesses
- The study is limited to a single GSM domain, which makes it questionable whether the proposed findings and methods could generalize to other broader domains. In particular, with respect to GSM-Agent, similar benchmarks that support retrieval-based reasoning have long existed, namely HotpotQA [1] and MuSiQue [2] based on Wikipedia documents. Controllable experiments could also be done by either plugging in a Wikipedia search engine or directly giving the relevant Wikipedia documents to LLM agents.
- It is unclear how the defined agentic reasoning graph is different from or improved upon the related work of reasoning topology (Minegishi et al., 2025) cited in the paper, which limits the novelty of the proposed analytical methodology of agentic reasoning.
- The study overlooks the clarification of confounders that may affect agentic reasoning accuracy. For example, after encouraging more proportion of revisit steps, it is unclear whether the improvement is due to rechecking previous relevant documents to trigger better reflection, or just because of accessing less kinds of irrelevant documents in the top-k retrieval, thus being less distracted but not performing better thinking.

[1] HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. Yang et al., 2018.

[2] MuSiQue: Multihop Questions via Single-hop Question Composition. Trivedi et al., 2022.

### Questions
- In Table 1, why is the search-complete (SC) rate, i.e., the proportion of tasks that an agent finds all relevant documents, always lower than the final answer accuracy? Does this mean that the agent often correctly guesses the answer even if the relevant documents (premises) are not fully found? If so, the results of accuracy may not be reliable enough to measure the agentic reasoning performance.
- In section 3.2.4, should 32315 be the number of unique documents (premises) in the database instead of unique questions? This may be a typo.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GSM-Agent, a benchmark designed to evaluate and analyze agentic reasoning—the ability of LLMs to reason while interacting with external tools—under a clean and controllable setup.
The authors adapt the GSM8K dataset into a simulated environment where models must actively retrieve missing premises using search tools before solving math problems. The benchmark allows precise control of difficulty via distractor documents and provides a consistent comparison between static and agentic reasoning. To interpret agent behavior, the paper proposes the Agentic Reasoning Graph (ARG), which maps search traces into discrete nodes representing clusters of environment embeddings. Analysis reveals that the revisit ratio (returning to a previously explored node) strongly correlates with task accuracy. Finally, a tool-augmented scaling method introduces explicit revisit tools to encourage iterative reasoning, improving model performance compared with naive interaction scaling.

### Strengths
1. Well-constructed benchmark. The GSM-Agent dataset offers a rare controlled setup where agentic reasoning can be compared apples-to-apples with static reasoning on identical base problems.
2. Quantitative interpretability. The Agentic Reasoning Graph provides a useful formalism to visualize reasoning dynamics and correlate graph-level metrics (explore, exploit, revisit) with performance.
3. Insightful empirical finding. The identification of revisit behavior as a core agentic skill offers an interesting direction for future reasoning-oriented architectures.

### Weaknesses
1. The benchmark mainly extends GSM8K into a retrieval environment. While the controllability is commendable, the core idea—transforming reasoning problems into search tasks—is incremental relative to prior works such as AppWorld, ToolSandbox, and τ-Bench. The “agentic reasoning graph” is conceptually borrowed from earlier reasoning topology analyses (e.g., Minegishi et al., 2025) with minimal methodological novelty.

2. A large portion of the paper describes data processing (entity renaming, timestamping, anonymization), which, while meticulous, dilutes the conceptual contribution. The central analytical claim—the link between revisit ratio and accuracy—could have been explored more deeply or theoretically justified.

3. The benchmark is fully built from grade-school math problems, limiting its applicability to broader agentic domains (e.g., open-domain QA, code synthesis, multimodal planning). The paper does not test whether the identified patterns generalize beyond arithmetic reasoning.

4. While the Agentic Reasoning Graph provides an interpretable diagnostic, it lacks formal rigor—no quantitative justification is given for why the revisit ratio should causally imply stronger reasoning, or how it might relate to planning-oriented cognitive models.

5. Insufficient ablations and reliability checks.
   The paper lacks:
   * Statistical validation of correlations (e.g., confidence intervals for (R^2 = 0.914) in Fig. 4).
   * Sensitivity analysis for the chosen clustering hyperparameters (e.g., K = 250).
   * Verification of annotation or tool-use consistency across random seeds.

6. The authors acknowledge that adding “revisit tools” improves performance for weaker models, but the improvement magnitude is small and inconsistent (Fig. 5). The work does not discuss why some models (e.g., GPT-5, o3) barely benefit—suggesting the method might not scale universally.

7. Since all results are collected via LangChain-based calls, the paper may confound *prompt engineering* effects with genuine reasoning ability. The evaluation framework’s reproducibility under open-source conditions remains uncertain.

### Questions
1. Please report statistical significance (e.g., Pearson/Spearman coefficients with p-values) for the correlation between revisit ratio and accuracy (Fig. 4b). The current (R^2) alone does not confirm robustness.
2. The Agentic Reasoning Graph uses (K = 250) clusters. Please include ablations for different K values (e.g., 100, 500) to test stability of exploration/exploitation/revisit metrics.
3. Extend GSM-Agent to non-math tasks (e.g., commonsense QA or simple retrieval puzzles) to test whether the “revisit hypothesis” generalizes beyond arithmetic reasoning.
4. Report whether the “Revisit” tool improves reasoning in unseen environments (e.g., new document distributions). If gains vanish, it would suggest overfitting to the current dataset.
5. Provide qualitative examples of failed reasoning traces with low revisit ratios—do these failures stem from early search termination, irrelevant document focus, or token limit exhaustion?
6. Include open-source code or pseudo-code for your LangChain evaluation pipeline to ensure replicability.
7. A fair baseline against ToolSandbox (Lu et al., 2024) or AppWorld (Trivedi et al., 2024) would strengthen the claim that GSM-Agent uniquely isolates agentic reasoning effects.

### Soundness
2

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
This work:
1) builds the GSM-Agent benchmark to isolate agentic reasoning (by requiring LLMs to collect necessary premises via tools for grade-school-level problems, avoiding mixing it with other advanced capabilities);
2) proposes the agentic reasoning graph to analyze reasoning patterns and identify that many models lack the "revisit" ability critical for agentic reasoning;
3) develops a tool-augmented test-time scaling method to improve LLMs’ agentic reasoning performance by encouraging revisit.

### Strengths
1. GSM-Agent’s controllable environment enables unique static-agentic reasoning comparisons by isolating agentic skills via simple math, avoiding confounding factors like advanced expertise.
2. The agentic reasoning graph provides a reproducible framework to quantify exploration/exploitation/revisit, with strong data $R^2=0.914$ validating revisit as a core skill.
3. The tool-augmented method outperforms naive interaction scaling, offering a practical, low-cost way to boost agentic reasoning without retraining.
4. Comprehensive evaluation of 12 models across metrics gives a holistic view of agentic performance.

### Weaknesses
1. The choice of K=250 clusters for the agentic reasoning graph lacks justification or ablation, weakening the graph’s robustness.
2. Missing critical baselines (e.g., perfect retrieval) and tool ablations (e.g., Revisit(·) alone) make it hard to rule out confounding factors.
3. No evidence that revisit’s importance generalizes beyond grade-school math, limiting claims about its role as a general agentic skill.
4. "The performance of GPT-5 stands at only 67%". Yet in my view, this is a relatively high score when considering the benchmark’s complexity. For context, take OSWorld in another domain: when this benchmark was first proposed, a mere 30% performance rate was already regarded as high.

### Questions
Address the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
