# When Disagreements Elicit Robustness: Investigating Self-Repair Capabilities under LLM Multi-Agent Disagreements

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of cooperation and tool use in multi-agent systems (MAS). However, it remains unclear how disagreements shape collective decision-making. In this paper, we revisit the role of disagreement and argue that general, partially overlapping disagreements prevent premature consensus and expand the explored solution space, while disagreements on task-critical steps can derail collaboration depending on the topology of solution paths. We investigate two collaborative settings with distinct path structures: collaborative reasoning (CounterFact, MQuAKE-cf), which typically follows a single evidential chain, whereas collaborative programming (HumanEval, GAIA) often adopts multiple valid implementations. Disagreements are instantiated as general heterogeneity among agents and as task-critical counterfactual knowledge edits injected into context or parameters. Experiments reveal that general disagreements consistently improve success by encouraging complementary exploration. By contrast, task-critical disagreements substantially reduce success on single-path reasoning, yet have a limited impact on programming, where agents can choose alternative solutions. Trace analyses show that MAS frequently bypasses the edited facts in programming but rarely does so in reasoning, revealing an emergent self-repair capability that depends on solution-path rather than scale alone. Our code is available at *anonymity*.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines how disagreements affect the performance of multi-agent systems. The authors propose that not all disagreements are equal; they distinguish between general disagreements, which can be beneficial, and task-critical disagreements, which can be harmful. General disagreements, where agents have slightly different perspectives, are shown to help prevent premature consensus and encourage the exploration of more solutions. Conversely, disagreements on crucial facts can derail collaboration, particularly when there's only a single correct path to a solution. The researchers tested this by creating two scenarios: collaborative reasoning, which usually has one correct answer, and collaborative programming, which can have multiple valid solutions. Their experiments showed that general disagreements consistently led to better outcomes in both settings. However, task-critical disagreements significantly lowered success rates in reasoning tasks but had a much smaller impact on programming. In the programming tasks, the agents were often able to find alternative solutions, working around the incorrect information. This suggests that multi-agent systems have a self-repair capability that depends more on the nature of the problem than on the scale of the AI models themselves.

### Strengths
1. The paper is easy to understand. 
2. The authors made an early exploration of disagreement in multi-agent systems.
3. Some of the findings are interesting, and the experiments and framework are comprehensive.

### Weaknesses
My main criticism is that many of the insights claimed in this paper are already broadly accepted within the community, offering limited novelty.

1. The paper's first key insight (Fig. 1 caption Insight I), "partial disagreements expand the joint decision space of multi-agents", essentially restates the fundamental rationale for studying multi-agent systems (MAS) in the first place. The community explores MAS because distributed agents, whether through "partial disagreement" or general heterogeneity, inherently expand the collective solution space. While the authors explicitly label it "partial disagreements," this concept is already implicitly or explicitly covered in prior MAS research.

2. The second claimed insight (Fig. 1 caption Insight I), "unique-path tasks are brittle to local task-critical disagreements, whereas multi-path tasks can route around localized disagreements", is not a surprising conclusion. It logically follows that systems with multiple valid pathways are more robust to localized failures than those with a single, linear dependency. More importantly, the paper lacks a mathematical model or a theoretical framework for defining and analyzing "localized disagreements". Given that LLMs and MASs are already known to be somewhat insensitive to noisy or self-contradictory context (which occurs frequently in daily LLM use), what is truly new here is unclear.

3. I question the necessity of artificially injecting disagreements into the system. LLMs used in the experiments appear to be relatively small and less capable. Furthermore, the type of "devised disagreement" is unlikely to occur in SOTA LLMs, let alone modern MASs (such as Cursor and Copilot). A more capable base LLM might simply solve the issues being investigated. Considering the capability of SOTA LLMs,  I do not think the proposed problem, or at its current form, is a fundamental flaw in collective decision-making.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper investigates how disagreements among agents in LLM-based multi-agent systems affect collaborative decision-making and robustness. The authors distinguish between general disagreements and task-critical disagreements. Results show that general disagreements often can improve collaboration, in contrast to task-critical disagreements that harm single-path reasoning. Results and analyses reveal findings for multi-agent systems.

### Strengths
1. This paper presents analyses through a path-aware view of robustness, and the distinction between single-path and multi-path tasks provides useful findings for understanding MAS behavior.

2. The experimental design is rigorous, with controlled manipulation of disagreements through knowledge editing.

3. The findings and analyses have practical implications for MAS design and challenge the assumption that all disagreements are harmful. The self-repair capability is an interesting emergent phenomenon worthy of further study.

### Weaknesses
**1. Limitations in mechanistic understanding:** While the paper documents self-repair behavior, it provides limited insight into why it occurs. 

**2. Limitations in using knowledge editing for disagreements**: The use of knowledge editing, especially parametric methods like ROME, may not faithfully represent naturally occurring disagreements in deployed systems. 

**3. Limitations in theoretical grounding**: The formalization in Section 2 is intuitive but lacks rigor. For example, there seems to present no formal proof that path multiplicity enables self-repair. Also, the connection between $\mathcal{M}(\tau)|$ and robustness is asserted but not validated empirically. It’s also unclear how to measure or estimate ∣M(τ)∣ for real tasks?

**4. Incomplete analysis**: Table 2 shows mixed results (some lower, some higher), while the claim that "general disagreements trigger complementary exploration" needs more support. Also, programming vs. reasoning tasks differ in many ways beyond path multiplicity, such as task complexity, evaluation metrics, context length, etc., while there is no attempt to control for these confounds.

**5. Limitations in statistical rigor**: There is no error bars or significance tests despite running experiments 5 times. Also, the claims about "substantial" vs. "marginal" changes are qualitative. Confidence intervals are also missing on adoption probabilities.

**6. Limitations in scalability**: Figure 3 shows performance degrades with >1 disagreement, but real systems may face many disagreements. This raises concerns regarding scaling to larger teams and longer interactions.

### Questions
My questions are following several aspects mentioned in weakness:

- Can you provide more analysis on how and why self-repair occurs?
- How do you operationalize $\mathcal{M}(\tau)|$ in practice? How do you measure or estimate $\mathcal{M}(\tau)|$ for real-world tasks?
- Why do heterogeneous agents sometimes decrease performance, while sometimes gain higher performance? Have you tested with more heterogeneity, like more different models in their sizes and capabilities? Is there an optimal level of diversity?
- How do results change with naturally occurring disagreements, e.g., agents trained on different corpora?
- Do findings hold for other collaborative tasks? Can your analyses and conclusions scale to larger teams and longer interactions?
- Based on your findings, what specific design recommendations would you give for building robust MAS?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the role of disagreements in LLM-based multi-agent systems (MAS) and their impact on robustness. The authors posit that the effect of disagreement is not uniformly negative but depends critically on two factors: the type of disagreement (general or task-critical) and the solution path topology of the task (single-path or multi-path). The central finding is that while single-path tasks are fragile to task-critical disagreements, multi-path tasks exhibit a self-repair capability.

### Strengths
1. The paper effectively operationalizes its hypotheses by contrasting collaborative reasoning with collaborative programming. The use of knowledge editing (ROME, MEND, IKE) to inject controlled, task-critical disagreements is robust.
2. The trace analysis for RQ3 provides evidence of the self-repair mechanism 
the authors hypothesize. The example in Table 6, where the MAS avoids the edited append() function by using a different implementation, clearly demonstrates this rerouting capability in practice.
3. The paper is for the most part well structured and written, and the diagrams, tables, etc. are appropriate and clear.

### Weaknesses
1. The paper frames the task topology as a binary (single-path vs. multi-path). This distinction, while useful for the experiment, may be an oversimplification. Many complex reasoning tasks might admit multiple evidential paths, and some programming tasks may have only one optimal solution.
2. While the paper demonstrates that self-repair occurs, it does not analyze the mechanism of this repair. The trace analysis shows what happens, but not the communicative or reasoning dynamics of how the agents collectively decide to abandon a conflicting path. The authors should provide the dialogues to illustrate this process.
3. The analysis of self-repair limits (Fig. 3) is conducted on LLaMA-based MAS for HUMANEVAL. It is unclear how this breaking point scales with model size or capability. This aspect feels underexplored.
4. There is a significant lack of clarity in the experimental setup for the Mixed Systems in Collaborative Reasoning. The methodology described in Section 3.2.1 implies all three mixed systems should have an identical agent composition, yet Table 2 reports different performance results for each. The paper does not discuss the reason for this discrepancy.
5. The paper's claim of emergent self-repair is potentially confounded by the experimental design. The task-critical disagreement experiments involve editing only one agent. In the multi-path setting, this means the majority of agents still possess the correct knowledge. The system's success might be less about the emergent self-repair of the MAS and more about a simple, non-emergent majority vote. The study fails to test if self-repair would still occur if a majority of agents held the task-critical disagreement.
6. Typo: In Section 1, the sentence begins "In collaborative, a group of coders...". It appears the word "programming" is missing after "collaborative".

### Questions
See Weaknesses

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
This paper investigates how disagreements affect multi-agent systems (MAS) with LLMs across two task types: collaborative reasoning (single-path tasks like COUNTERFACT, MQUAKE-CF) and collaborative programming (multi-path tasks like HUMANEVAL, GAIA). The authors distinguish between "general disagreements" (heterogeneous agent diversity) and "task-critical disagreements" (conflicting knowledge injected via knowledge editing methods). Their main findings suggest that general disagreements improve performance, while task-critical disagreements harm single-path reasoning but have a limited impact on multi-path programming, where systems show "self-repair" by avoiding edited facts.

### Strengths
- The distinction between general disagreements (beneficial diversity) and task-critical disagreements (potentially harmful conflicts) is well-motivated and provides a useful lens for understanding MAS robustness.

- The use of knowledge editing methods (IKE, ROME, MEND) to inject controlled disagreements is novel and allows for reproducible experiments.


- The paper tests multiple models (LLaMA, Qwen, InternLM) across multiple datasets with multiple metrics, providing reasonable breadth.

### Weaknesses
The paper uses AutoGen for collaborative programming but doesn't compare against ChatDev [1] or MetaGPT [2], which are established frameworks specifically designed for multi-agent software development. These frameworks have been shown to significantly outperform simpler multi-agent setups and would be natural baselines. For instance, ChatDev achieves quality scores of 0.3953 compared to 0.1523 for MetaGPT on software development benchmarks through its cooperative communication method [1].


The claim that self-repair is an "emergent capability" is overstated. Table 5 shows adoption rates declining only modestly (e.g., from 34.76% to 32.93% for LLaMA on HUMANEVAL), which could simply reflect the probabilistic nature of API selection rather than systematic avoidance. More rigorous causality analysis is needed.


Recent work like "On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents" [5] directly studies faulty agents in MAS and finds hierarchical structures most robust (losing only ~5% accuracy vs. ~24% for chain structures). This paper isn't cited despite clear topical overlap. Your work would be strengthened by comparing different MAS topologies beyond the flat peer structure you use.


- Minor 

The formalism in Section 2 (information atoms, minimal sufficient knowledge sets M(τ)) is introduced but not really leveraged. It reads more like hand-waving than rigorous analysis.

### Questions
Could you show that the avoidance is systematic rather than just probabilistic sampling effects? Perhaps compare against temperature=0 for more deterministic generation?

IKE (in-context) is quite different from ROME/MEND (parametric). Do they represent fundamentally different types of disagreements, and if so, shouldn't they be analyzed separately?

Could you provide token/time costs to assess whether these benefits justify the computational overhead?

### Soundness
3

### Presentation
3

### Contribution
2
