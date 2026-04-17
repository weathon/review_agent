# AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
As Large Language Model (LLM) agents become more widespread, associated
misalignment risks increase. While prior research has studied agents’ ability to
produce harmful outputs or follow malicious instructions, it remains unclear how
likely agents are to spontaneously pursue unintended goals in realistic deploy-
ments. In this work, we approach misalignment as a conflict between the internal
goals pursued by the model and the goals intended by its deployer. We introduce
a misalignment propensity benchmark, AGENTMISALIGNMENT, a benchmark
suite designed to evaluate the propensity of LLM agents to misalign in realistic
scenarios. Evaluations cover behaviors such as goal-guarding, resisting shutdown,
sandbagging, and power-seeking. Testing frontier models, we find that more capa-
ble agents tend to exhibit higher misalignment on average. We also systematically
vary agent personalities through different system prompts and observe that persona
characteristics can strongly and unpredictably influence misalignment, sometimes
more than the choice of model itself. Our results reveal the limitations of current
alignment methods for autonomous LLM agents and underscore the need to rethink
misalignment in realistic deployment settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper defines LLM agent misalignment as a conflict between the model’s internal goals and deployers’ intended goals, then proposes the AGENTMISALIGNMENT benchmark (9 real-scenario tasks, e.g., log tampering, resisting shutdown) to evaluate misalignment propensity. Key findings: more capable models (e.g., Claude 3 Opus) show higher misalignment; agent personalities (via prompts) often impact misalignment more than models; existing alignment methods are insufficient for real deployments. It contributes a real-scenario misalignment benchmark, tests cutting-edge models, and quantifies personality effects.

### Strengths
1. Controls variables (model, personality, task), uses RMS normalization and CMS scoring for comparability, and validates via case studies.

2. Clear structure with appendices (task prompts, scoring rules) for transparent replication.

### Weaknesses
1. The weighting of "severity" in the CMS scoring lacks a quantitative basis (e.g., "Approve Toxic Content" has a weight of 1.0, "Reject Neutral Content" has a weight of 1/(N+1)), and there's no explanation as to why this weighting accurately reflects the actual number of errors.

2. No Long-Term Evaluation: All experiments are short-term (e.g., single-round negotiation); no data on “misalignment drift” (e.g., 7–14 days of community management) or which is critical for real deployments.

3. The paper fails to clarify the extent of consequences caused by agent misalignment. However, "revealing the actual harms of misalignment" should have been the core starting point for researching "evaluating misalignment propensity," which weakens the study’s risk warning significance and practical relevance.

### Questions
1. For CMS subjectivity: What basis sets “severity weights”? Has expert validation been done? 

2. For long-term gaps: Can you provide a long-experiment design (e.g., 10-round “Community Curator” for Claude 3 Opus)? 

3. Supplement explanations of the consequence severity corresponding to different misaligned behaviors. This aligns the research on "evaluating misalignment propensity" more closely with the starting point of "addressing actual harms" and enhances the study’s practical value.

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
This paper introduces a benchmark suite designed to evaluate the propensity of LLM agents to spontaneously exhibit misaligned behaviors in realistic deployment scenarios. The study encompasses multi evaluation tasks covering behaviors such as avoiding oversight, resisting shutdown, sandbagging, and power-seeking. The authors test multiple frontier models and systematically investigate how different personality prompts influence misalignment propensity. Key findings indicate that more capable models tend to show higher average misalignment, and personality characteristics can sometimes exert greater influence on misalignment than model choice itself.

### Strengths
1. The evaluation framework is comprehensive. It covers diverse misalignment behavior types.

2. The exploration of how personality prompts affect agent behavior is an important but understudied problem.

3. Detailed experimental setups, prompts, and scoring mechanisms are provided.

4. InspectAI framework provides a unified interface for cross-model comparison.

5. The focus on propensity rather than pure capability represents an important distinction for assessing real-world deployment risks.

### Weaknesses
1. The CMS scoring mechanism relies primarily on keyword and pattern matching to detect misaligned reasoning, potentially missing more subtle or differently-expressed misaligned reasoning while also generating false positives

2. The ecological validity of evaluation tasks is questionable, with some scenarios designed too obviously to elicit misalignment

3. The experiments use single runs with temperature equals zero for most models, limiting understanding of behavioral stability and variance

4. Personality prompt effects may primarily be artifacts of prompt engineering rather than revealing deep model tendencies, and the paper insufficiently distinguishes between these possibilities.

5. Some evaluation tasks observe no misalignment behaviors at all, such as Rank Organizations, raising questions about task design effectiveness.

6. Causal mechanisms remain unclear, with the paper failing to explain why certain personalities increase or decrease misalignment propensity.

7. Canary Code where no model successfully modified log files, possibly indicating capability insufficiency rather than alignment success.

### Questions
1. How can the study distinguish between models unable to execute misaligned behaviors due to capability limitations versus models understanding the situation but choosing not to execute misaligned behaviors?

2. Could personality prompts merely change surface response styles rather than genuinely altering decision-making tendencies?

3. Might observed behaviors in some evaluation tasks be reasonable responses to hints in prompts rather than spontaneous misalignment? 

4. If different hyper-parameters settings or multiple runs were used on the same tasks, how stable would the results be?

5. How were weights and thresholds in the CMS scoring mechanism determined? Were ablation studies conducted to validate these choices?

6. How does the paper ensure evaluations have not been seen by models during pretraining?

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
2

### Summary
This paper introduces AGENTMISALIGNMENT, a new benchmark suite designed to evaluate the propensity of agents to spontaneously misalign with their deployer's intended goals. They define intent misalignment as behavior where an agent recognizes but still acts against deployer goals, and they target categories such as avoiding oversight, power seeking, resisting shutdown, resource acquisition, deception, and drifting from an ethical baseline. The authors argue that prior work has focused on agent capability (what they can do when explicitly prompted for harmful tasks), whereas this work tests what agents are likely to do in realistic scenarios where deployer intentions are often implicit .
The paper defines misalignment as "intent misalignment", a conflict between the internal goals an agent pursues and the goals intended by its deployer. The benchmark suite consists of 9 evaluations that create "honeypot" scenarios to test for specific misaligned behaviors. Findings include that more capable models tend to show higher misalignment on average and that persona prompts can shift misalignment substantially, sometimes as much as or more than swapping models.

### Strengths
- The paper's primary strength is its clear distinction between what an agent can do (capability) and what it is likely to do spontaneously (propensity) . This moves safety evaluations toward more realistic deployment scenarios where an agent might pursue unintended goals even without malicious prompting. the benchmark probes propensity in deployment-like contexts rather than single-turn capability checks.
- The benchmark uses a Comprehensive Misalignment Scoring (CMS) mechanism that evaluates an agent's internal reasoning in addition to its final actions. This allows the benchmark to detect deceptive intent or planning, even if the agent is not capable enough to successfully execute the misaligned behavior. The paper distinguishes misuse from misalignment and enumerates concrete behavioral categories
- The study methodically investigates system prompts as a key variable, treating personality as a "high-leverage alignment control surface". The inclusion of case studies (Sections 4.1 and 4.2) clearly demonstrates how different personas can cause the same model to behave differently in identical scenarios, providing valuable insight into prompting-based risks.

### Weaknesses
- The authors acknowledge that the results have large error bars and "lots of variance" between evaluations (as seen in Figure 1). This high variance and statistical uncertainty make it difficult to draw strong conclusions, forcing the authors to "refrain from drawing any definitive conclusions" about which models or personalities are definitively more or less misaligned on average.
- Cross-task comparability. Each eval uses different scoring, which the authors note complicates comparisons of absolute misalignment levels between tasks.All evaluations are in English and simulated environments, so generalizability to other languages and real deployments is limited.
- The authors note that the specific agent scaffolding (the ReAct loop) and tool-use formatting used in the benchmark "can be foreign to some models". This makes "accurate comparisons difficult" , as it introduces a confounding variable: a model's poor performance (or lack of misalignment) could be due to genuine alignment or simply an inability to effectively use the provided tools to achieve the misaligned action.

### Questions
- How reliable are the comparisons between models and personalities, given the high variance and large error bars in the results?
- Since evaluations are English-only and use different scoring mechanisms, how generalizable are the findings to other languages, and how alid are comparisons of misalignment levels between different tasks?
- How can you distinguish between genuine alignment and a model's simple inability to use the "foreign" agent scaffolding to perform the isaligned action?

### Soundness
2

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
4

### Summary
The paper introduces AGENTMISALIGNMENT, a benchmark designed to measure the likelihood of large language model agents engaging in misaligned behavior such as avoiding oversight, resisting shutdown, sandbagging, or seeking power. The study treats misalignment as a conflict between an agent’s internal goals and its deployer’s intentions, emphasizing that real-world deployments often rely on implicit expectations that are difficult to fully specify. The author evaluated various LLM models and tested the effect of varying persona in system prompt.

### Strengths
- The author uses controlled, deterministic experimental setups to ensure reproducibility.
- The author evaluated on a variety of latest models.

### Weaknesses
- The benchmark mainly combines known misalignment behaviors (e.g., deception, shutdown resistance, etc.). Many existing papers already tackle similar problems. The authors do not necessarily provide insights or theoretical constructs that make this work stand out, or they fail to make these contributions clear due to the writing or presentation style.
- It is unclear how the authors set up the experiments and implementation details. For example, what tasks the agents are performing, how they are evaluated, what metrics are used, and which variables are controlled or changed. The authors should consider adding separate sections that explicitly describe these details and include clear result tables.
- Based on the current presentation, the experimental setup appears overly simplistic. i.e. 1) defining a few misalignment types, 2) writing a few fixed personas, 3) calling LLMs to generate outputs, and then 3) evaluating those outputs with another LLM. Steps 1 and 2 also seem trivial enough to be easily automated by an LLM itself.
- The authors only evaluate six fixed personality prompts. The resulting interpretations are therefore too limited and lack sufficient generalizability or practical usefulness for others.

### Questions
- Could the authors provide a more detailed description of the experimental setup? Specifically, what are the concrete tasks that the agents perform in each evaluation scenario, how are these tasks constructed, and what are the underlying assumptions behind each one? A clearer explanation of these would help assess how realistic or representative the scenarios are.
- How were the six personality prompts chosen? Were they derived empirically (e.g., from prior studies or human behavioral typologies), or were they arbitrarily designed?

### Soundness
2

### Presentation
1

### Contribution
1
