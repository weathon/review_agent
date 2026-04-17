# Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Driven by recent advancements in tool-augmented Large Language Model (LLM) agents, comprehensive benchmark datasets for evaluating these tool-augmented agents are being actively developed. Although these benchmarks incorporate increasingly complex user requests and a diverse array of tools, the evaluation methods for most of them remain limited to answer matching. However, as the number of steps required to resolve a user request increases, a proper evaluation of an agent's performance must go beyond the final answer to also assess the problem-solving trajectory, including previously ignored aspects such as efficiency, hallucinations, and adaptivity. The most straightforward method for evaluating these aspects is to compare the trajectory of the agent with a ground-truth trajectory, but this approach is fundamentally limited since annotating all possible ground-truth trajectories is prohibitively expensive. To address these significant gaps, we introduce TRACE, a framework for the multi-dimensional evaluation of tool-augmented LLM agent performance. By incorporating evidence store, TRACE enables a multi-faceted analysis and evaluation of an agent's reasoning trajectory, eliminating the need for a predefined ground-truth trajectory. To validate our framework, we develop a new meta-evaluation dataset by augmenting existing benchmarks with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates these complex behaviors in a scalable and cost-effective manner, even with small open-source LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TRACE, a framework for multi-dimensional evaluation of tool-augmented LLM agents. The authors argue that existing benchmarks mostly rely on answer matching, which fails to assess important aspects such as efficiency, hallucination, and adaptivity in long, multi-step tasks. TRACE introduces an evidence bank that accumulates information from previous reasoning steps, enabling detailed trajectory-level analysis without requiring exhaustive ground-truth annotations. To validate the framework, the authors construct a meta-evaluation dataset by augmenting existing benchmarks with diverse, flawed trajectories labeled across multiple performance dimensions. Experiments show that TRACE can accurately and cost-effectively evaluate complex agent behaviors, even using small open-source LLMs, and reveal new insights into how agents perform across long, tool-augmented workflows.

### Strengths
1. The paper introduces a novel benchmark that focuses on step-level performance, representing meaningful progress in evaluating LLM agents.
2. It proposes a new evaluation framework capable of automatically assessing agent behaviors in a more fine-grained and systematic manner.
3. This method gets rid of the drawback of relying on fixed annotations when offline for online interaction data. I hope the author can open source it, which will contribute to the development of this field.

### Weaknesses
1. The work lacks sufficient novelty, the step-level storage and verification mechanism is essentially similar to traditional checkpointing, which is not an innovative idea. Moreover, the term “evidence bank” does not effectively conceal this conceptual limitation. 
2. The three proposed evaluation metrics are not clearly differentiated from those in prior work, which limits the contribution mainly to data collection rather than methodological advancement (such as metrcs although authors said that).
3. The evaluation process largely relies on prompt engineering; although the experiments show improvements over previous methods, the paper fails to provide human evaluation benchmarks for validation.
4. The writing quality needs improvement (starting from Section 3), the paper reads more like an experimental report than a coherent, well-structured research narrative.

### Questions
1. Section 5.1 devotes long passages to describing the experimental procedure and metrics, but many of these details should be moved to the appendix. How do you handle the same instance when there is no gold trajectory, how can we tell when the model has reached an optimal solution?
2. What is the difference between the information stored in the evidence bank and simply feeding the whole trajectory directly to LLMs?
3. How do you validate the quality of the newly annotated data?
4.	In Table 1 your method shows a clear improvement over “LLM-as-a-Judge,” but higher scores are not necessarily better — shouldn’t the scores aim to match human evaluation?
5.	More experiments with additional thinking models are needed, for example doubao-thinking-pro or models specialized for tool use.
6.	The experimental analysis lacks concrete, interesting conclusions; it reads vague. Are there cases where other methods fail but your method succeeds? I think the paper needs a systematic summary of such success/failure cases.

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
This paper addresses a critical limitation in evaluating tool-augmented Large Language Model (LLM) agents: the heavy reliance on final-answer accuracy or comparison against a single ground-truth trajectory. The authors argue that merely matching the final answer fails to assess crucial aspects of the problem-solving process, such as efficiency, hallucination, and adaptivity. To solve this, the authors introduce TRACE (Trajectory-based Reasoning Assessment and Comprehensive Evaluation). TRACE is an effective, simple LLM-based evaluation framework that assesses the logical soundness of an agent’s reasoning trajectory without relying on a single, pre-defined ground-truth path. The core mechanism is the evidence bank, a dynamically constructed knowledge base that accumulates factual information from each reasoning step. To validate TRACE, the authors developed a novel meta-evaluation dataset (Meta-GTA and Meta-m&m's) by augmenting existing benchmarks (GTA and m&m’s) with diverse, flawed trajectories (inefficiency, hallucination, adaptivity) labeled with multi-faceted performance scores. Experiments show that TRACE significantly outperforms a naive LLM-as-a-Judge baseline and existing trajectory evaluation methods like PIPA. The final experiments apply TRACE to real-world agents, revealing significant performance differences that were obscured by standard final-answer accuracy metrics.

### Strengths
- The paper's core premise is strong and interesting. The field is moving beyond simple task completion, and evaluating the process of reasoning, not just the result, is a critical and necessary next step for building robust agents.
- The idea of creating a meta-evaluation dataset by synthetically injecting flaws is a clever and pragmatic approach. It creates a clear ground truth for testing the evaluator (TRACE) without facing the impossible-to-scale task of manually annotating all possible flawed trajectories.

### Weaknesses
- The efficiency metric is defined by identifying a "minimal subset of evidence" ($\mathcal{E}_{min}$). This task is delegated to an LLM evaluator, which is a complex reasoning task in itself. The paper does not adequately validate the evaluator's ability to correctly identify this minimal path.
- The validation shows that TRACE is good at detecting synthetically injected flaws (e.g., a "find-and-replace" style hallucination, as per the prompt in Fig. 9). This is a much easier task than finding subtle, naturally-occurring agent hallucinations, which may be more about omission or logical leaps rather than direct contradiction. The paper doesn't prove that success on its synthetic benchmark translates to success in finding these "in-the-wild" errors.

### Questions
- The adaptivity test is very narrow (api failure). How do the authors think TRACE would perform on more nuanced failures, such as a tool returning an empty list or a rate limit error?
- The paper claims the evidence bank improves both accuracy and speed, do the authors have any ablation experiments to back up this claim? Did the authors consider any other representations? e.g. not tuple, or to the extreme, just the original thinking trace.
- Given that TRACE itself depends on LLM evaluators, how sensitive are the results to evaluator choice (e.g., GPT-4 vs. smaller open-source LLMs)?
- How would TRACE handle multi-agent or parallel reasoning settings, where multiple concurrent tool calls are valid?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces TRACE, an LLM-based framework that builds an evidence bank from each tool call (action, input, observation) and then scores a trajectory along three dimensions: efficiency, hallucination, and adaptivity. The authors also construct meta-evaluation datasets. TRACE outperforms a naïve LLM-as-judge and shows better robustness than PIPA’s state-consistency measure. They further apply TRACE to real agents on GTA, revealing differences among models that overall accuracy obscures (e.g., trade-offs between hallucination and adaptivity), and analyze token/turn-length effects.

### Strengths
1. Originality. Clear reframing from “final answer only” to process-level, multi-dimensional assessment without relying on a single ground-truth trajectory; the evidence-bank abstraction is a neat way to make evaluation modular and scalable.

2. Clarity: The paper reads cleanly; Fig. 2 and the formalization make the pipeline easy to follow (trajectory to evidence bank to per-dimension scoring). 

3. Significance: Applying TRACE to several agents on GTA exposes efficiency–hallucination–adaptivity trade-offs that final accuracy masks; the token/turn analysis offers actionable signals for system builders (e.g., shorter trajectories correlate with better accuracy).

### Weaknesses
1. Evaluator dependence & construct validity
TRACE still relies on an LLM to: (i) select a “minimal” evidence subset, (ii) judge grounding, and (iii) qualify adaptivity. This introduces evaluator bias and potential non-identifiability (different minimal sets may be equally valid).

2. Minimal-evidence identification Current approach asks an LLM to pick E_min; there’s no guarantee of subset minimality or monotonicity.
No algorithmic ablation (e.g., backward elimination / counterfactual removal tests) to verify that removing any item from E_min breaks correctness.

3. Scope limitation to ReAct-style traces. TRACE is designed around action–observation loops; programmatic agents (planner–executor, code-gen with tests) may require different evidence schemas. Lack demonstration of adaptation to at least one non-ReAct agent family and discuss evidence schemas for program synthesis or planner-executor traces. 

4. Efficiency measured only on successful trajectories
This can yield survivorship bias (inefficient-but-correct vs. efficient-but-failed cases).
Actionable: Also report efficiency diagnostics on failed trajectories (e.g., minimal set relative to the intended answer or checklist), or provide a joint metric correlating efficiency with success probability.

### Questions
1. Evaluator reliability: What are the inter-evaluator agreements (κ / %-agreement) across metrics when swapping Claude/GPT/Llama/o3-mini as the judge, and how sensitive are results to prompt templates? Please include per-metric κ and A/B prompts. 

2. Hallucination policy: How do you treat thoughts that rely on unstated commonsense/background facts? Could TRACE allow an explicit “import fact” step so those facts are added to the evidence bank before being used, for example, using google search tool for latest factual info? 

3. Coverage outside ReAct: Can TRACE readily score trajectories from planner-executor or code-synthesis agents? A small demonstration would broaden impact. 

4. Efficiency on failures: Do “almost-there” failures look efficient under TRACE? Reporting efficiency distributions on incorrect runs would help diagnose whether inefficiency is a cause or consequence.

### Soundness
2

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
The paper proposes the TRACE framework, which utilizes an "evidence bank" to enable LLM evaluators to assess an agent's reasoning trajectory across three dimensions—efficiency, hallucination, and adaptivity—without relying on a single ground-truth trajectory.

### Strengths
- The paper clearly demonstrates through experiments that its proposed trajectory metrics are highly correlated with final accuracy, thereby validating the importance of these metrics.

- The meta-evaluation experiment provides a rigorous method for validating the accuracy of the evaluation framework itself, by injecting controlled, labeled flaws into existing benchmarks.

### Weaknesses
- The calculation of the efficiency metric relies on the LLM evaluator to accurately identify $\epsilon_{min}$. Can smaller models really measure this effectiveness? The meta-evaluation results in Table 1 show that the accuracy of smaller models on efficiency evaluation is much lower than on other metrics, which casts doubt on the reliability of this specific metric when used with non-SOTA evaluators.

- The scope of the adaptivity evaluation is too narrow. The metric currently only measures the response to an "unavailable tool" error. However, real-world failure modes are much more complex (e.g., the tool returns a syntactically correct but semantically wrong answer, tool timeouts, or incomplete information). The definition of this metric is likely too narrow.

- Hallucination in this work is defined as: TRACE identifies hallucinations by assessing whether an agent's thought at a given step can be logically derived from the evidence collected so far. However, isn't it possible that in some cases, the model's own parameterized knowledge already contains this information, and it doesn't need to be derived from previous thoughts? Would this still be considered a hallucination? This is actually a very important part of knowledge.

- Although the authors used cosine similarity to follow the established GTA benchmark protocol, I think the protocol itself is flawed. It will be interesting to compare this with other available methods, such as using LLM-as-a-Judge, or an embedding model with world knowledge like QWEN3-Embedding.

- To increase the work's impact and generalizability, I recommend adding experiments that use TRACE to evaluate SOTA tool-augmented agent systems.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
