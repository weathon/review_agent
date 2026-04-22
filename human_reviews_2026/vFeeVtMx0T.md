# Active Confusion Expression in Large Language Models: Leveraging World Models toward Better Social Reasoning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
While large language models (LLMs) excel in mathematical and code reasoning, we observe they struggle with social reasoning tasks, exhibiting cognitive confusion, logical inconsistencies, and conflation between objective world states and subjective belief states. Through deteiled analysis of DeepSeek-R1's reasoning trajectories, we find that LLMs frequently encounter reasoning impasses and tend to output contradictory terms like "tricky" and "confused" when processing scenarios with multiple participants and timelines, leading to erroneous reasoning or infinite loops. The core issue is their inability to disentangle objective reality from agents' subjective beliefs. To address this, we propose an adaptive world model-enhanced reasoning mechanism that constructs a dynamic textual world model to track entity states and temporal sequences. It dynamically monitors reasoning trajectories for confusion indicators and promptly intervenes by providing clear world state descriptions, helping models navigate through cognitive dilemmas. The mechanism mimics how humans use implicit world models to distinguish between external events and internal beliefs. Evaluations on three social benchmarks demonstrate significant improvements in accuracy (e.g., +10\% in Hi-ToM) while reducing computational costs (up to 33.8\% token reduction), offering a simple yet effective solution for deploying LLMs in social contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an Active Confusion Expression with Adaptive World Model framework to improve LLMs’ social reasoning.
It detects moments when a model shows uncertainty (e.g., “I’m confused”) and dynamically injects a structured “world state” and “belief state” clarification to guide reasoning.
Experiments on three Theory-of-Mind benchmarks demonstrate consistent accuracy and efficiency improvements over existing reasoning methods.

### Strengths
Novel and intuitive idea.
Using confusion signals as triggers for reasoning intervention is creative and aligns with human metacognitive behavior.

Empirical effectiveness.
The approach improves accuracy and reduces token usage across multiple ToM datasets.

Interpretability.
The intervention makes reasoning steps more explicit and easier to analyze.

### Weaknesses
The main limitation of this paper lies in its lack of theoretical rigor and generalization analysis. While the distinction between “world state” and “belief state” is conceptually appealing, it remains informally defined without a formal cognitive or probabilistic framework. The confusion detection mechanism relies on a fixed list of trigger words, making it heuristic and potentially fragile across languages, models, and task domains. Moreover, the paper does not analyze how the adaptive world model intervention changes the model’s internal reasoning dynamics, leaving the mechanism’s effectiveness somewhat opaque. Experimental validation is restricted to one primary model (DeepSeek-R1-Distill-Qwen-32B) and a few social reasoning benchmarks, with no exploration of broader reasoning contexts.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the phenomenon of cognitive confusion in social reasoning within LLMs. While LLMs have achieved strong performance in mathematical and code reasoning, the authors show that they often fail at theory-of-mind tasks that require distinguishing objective world states from subjective belief states.
The central contribution is an adaptive world-model–enhanced reasoning mechanism that detects “confusion” during reasoning and injects structured world-state information to help the model recover coherent reasoning trajectories. The method improves accuracy on three social reasoning benchmarks — ToMi, Hi-ToM, and Explore-ToM.

### Strengths
1. Novel diagnosis of reasoning breakdowns. The paper introduces Active Confusion Expression, which is an observable linguistic signal of reasoning failure.

2. Detailed semantic similarity analysis. The paper conducts a Semantic Similarity Analysis of Context-Disrupting Words.
For each word in DeepSeek-R1’s reasoning trajectories, they compute cosine similarity between the surrounding context embeddings.

### Weaknesses
1. Limited applicability to non-reasoning models.

2. Heuristic trigger detection.

3. Insufficient methodological details.

### Questions
1. The method relies on detecting confusion during explicit reasoning traces (e.g., CoT). Non-reasoning models, which generate short direct outputs without introspective tokens, do not provide a trajectory where intervention can occur. Would you explain whether the method can be applied to a non-reasoning model?

2. Although the paper employs Semantic Similarity Analysis offline to identify confusion words, the runtime detection mechanism is limited to simple keyword matching rather than continuous semantic monitoring. I wonder if the authors can compare their approach with alternative runtime intervention strategies, such as confidence-based examination or other uncertainty-aware methods, to better assess the effectiveness of their trigger design.

3. The process for constructing the textual world model is under-specified. The paper briefly states that it “tracks entity and character states,” but does not explain how these states are parsed or updated. In addition, the paper does not provide enough information about how the Semantic Similarity Analysis of Context-Disrupting Words was implemented.

### Soundness
3

### Presentation
2

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
This paper studies the limitations of large language models (LLMs) in social reasoning scenarios, where models often exhibit “cognitive confusion” — mixing objective world states with agents’ subjective belief states. Through analysis of DeepSeek-R1’s reasoning trajectories, the authors observe frequent self-contradictory phrases (e.g., “tricky”, “confused”) when the model handles multi-agent, multi-timeline tasks.

To mitigate this issue, the paper proposes an adaptive world model–enhanced reasoning mechanism.
The approach constructs a dynamic textual world model that tracks entities, characters, and events during reasoning.
When confusion indicators appear in the chain-of-thought, the system triggers an intervention step that injects structured world-state descriptions to help the model re-orient its reasoning trajectory.
Experiments on three social reasoning benchmarks (ToMi, Hi-ToM, ExploreToM) show improved accuracy (up to +7%) and reduced token usage (up to −33%).

### Strengths
Clarity and Motivation:
The paper clearly articulates a genuine weakness of current LLMs — their difficulty in disentangling subjective beliefs from objective reality in social reasoning. The examples of “confusion moments” are intuitive and effectively motivate the need for intervention-based reasoning control.

Methodological Simplicity:
The proposed mechanism is simple to implement — no retraining is required — and can be easily applied to existing LLMs. The structured “world model buffer” is an interpretable way to represent intermediate state information.

### Weaknesses
1. **Methodological originality is questionable**
   Conceptually, I find the proposed “adaptive world model–enhanced reasoning” to be a lightweight modification of the *ReAct*-style reasoning paradigm — the model detects confusion and injects contextual state summaries, which is essentially an intervention step within a reasoning loop. Because of this, comparing only against ReAct and other older prompting methods (CoT, ToT, Reflexion) is insufficient.
   To properly assess novelty, the paper should include comparisons with more recent *agentic workflow frameworks* that also implement structured intervention or state-tracking logic, such as **AFlow** [1], **DyFlow** [2], and **MaAS** [3]. Without these, it is hard to tell whether the claimed improvement stems from a fundamentally new mechanism or simply from a re-packaged form of existing agent reasoning workflows.

2. **Limited empirical gains**
   Even within the existing baselines, the improvements remain modest . Considering the extra complexity introduced by confusion detection and world-model updates, the benefit seems marginal. The results do not convincingly support the claim that this approach substantially enhances reasoning ability.

3. **Narrow evaluation scope**
   All experiments are run on **DeepSeek-R1-Distill-Qwen** variants, which are relatively weak reasoning models. Since the method does not involve training, it would be entirely feasible to apply it to stronger models such as **GPT-4o/5** or **Claude-4**. Evaluating only on weaker open-source models makes the practical significance unclear — current frontier models already handle social reasoning reasonably well, so the proposed mechanism may have limited relevance.

4. **Static trigger design and brittleness**
   The confusion-detection mechanism relies on a *static list* of trigger words (e.g., *tricky*, *confused*, *ambiguous*). This design feels fragile and language-dependent — small lexical variations or stylistic differences could easily bypass the trigger, leading to inconsistent intervention behavior. A more adaptive or semantic trigger mechanism would be needed to make the method robust and scalable across models and languages.

---

**References**

[1] *AFlow: Automating Agentic Workflow Generation*, [https://openreview.net/forum?id=z5uVAKwmjf](https://openreview.net/forum?id=z5uVAKwmjf)

[2] *DyFlow: Dynamic Workflow Framework for Agentic Reasoning*, [https://openreview.net/forum?id=0pbUfmwNTy](https://openreview.net/forum?id=0pbUfmwNTy)

[3] *Multi-agent Architecture Search via Agentic Supernet (MaAS)*, [https://openreview.net/forum?id=imcyVlzpXh](https://openreview.net/forum?id=imcyVlzpXh)

### Questions
My main questions are already reflected in the Weaknesses section — in particular, regarding the methodological originality (relation to ReAct-style frameworks), the choice of baselines, the limited empirical gains, and the static trigger design.
I would appreciate detailed clarifications or additional results addressing these points.

While I currently have substantial reservations about the contribution and positioning of this work, I am open to further discussion and would be willing to reconsider my evaluation if the authors can provide convincing responses or evidence during the rebuttal phase.

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
The paper studies why LLMs struggle on social reasoning / Theory-of-Mind (ToM) tasks and proposes an adaptive world-model–enhanced reasoning framework. The system monitors a model’s chain-of-thought for “confusion” indicators (e.g., words like tricky, confused) and, when detected, injects a dynamically maintained textual “world model” that tracks entities, agents, and timelines to disambiguate objective states vs. subjective beliefs. On ToMi, Hi-ToM, and ExploreToM, the method reports higher accuracy (e.g., +7.34 points on Hi-ToM with DeepSeek-R1-Distill-Qwen-32B) and fewer tokens (up to 33.8% reduction) compared to a baseline, and outperforms prompting strategies such as CoT, ToT, ReAct, and Reflexion.

### Strengths
Clear problem framing. The paper sharply identifies objective state vs. belief state conflation as a recurring failure mode in multi-agent, multi-timeline stories, supported by qualitative examples and token/word statistics.

Broad baselines (prompting). The comparison against CoT/ToT/ReAct/Reflexion helps position the contribution within test-time reasoning methods; the proposed approach wins by a comfortable margin on the three datasets.

Token-efficiency angle. Reporting token reductions alongside accuracy is useful; the claim of up to ~33.8% fewer tokens is appealing for deployment.

### Weaknesses
1. Limited generalization; risk of negative transfer. Evidence is almost entirely confined to ToM/social-reasoning tasks. There is no systematic evaluation on general benchmarks (e.g., GSM8K/MATH, MMLU-Pro, BBH), raising concern that the intervention could degrade performance on non-ToM tasks by injecting irrelevant structure.

2. Opaque feasibility and cost of streaming world-model updates under API constraints. Continual updates and repeated injections entail extra input/output tokens and more interaction rounds; the paper does not provide a complete, auditable accounting of tokens, latency, or failure/retry rates, making the “token savings” claim uncertain.

3. Epistemic leakage risk. The injected “world model” presents globally coherent objective states that may conflict with agents’ local observability and beliefs, potentially short-circuiting the very skill being tested—distinguishing facts from beliefs.

4. Brittle trigger mechanism. Reliance on “confusion words” as the intervention trigger is correlation-based, vulnerable to phrasing avoidance, sensitive to domain/language, and biased toward helping “loud” failures while missing confident mistakes.

5. Evaluation and reproducibility gaps. Possible coupling of the max-intervention count k and trigger lexicon to the test sets; heterogeneous sampling/budget policies across models/providers; absence of stratified metrics by belief order/visibility, confidence intervals, and multi-seed runs; and insufficiently strict token/compute budget controls, reducing the credibility and comparability of results.

### Questions
The same as the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
