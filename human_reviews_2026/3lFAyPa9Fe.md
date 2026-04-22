# CausalSteward: An Agentic Divide-Conquer-Combine Copilot for Causal Discovery

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Learning causal models from high-dimensional data is a significant challenge, particularly in real-world settings where violations of core assumptions lead to causal identifiability issues. Although massive amounts of prior knowledge are available, and contain valuable causal information, effectively integrating this knowledge into the causal discovery process remains an open problem. We introduce CausalSteward (CaST), a novel human-in-the-loop framework for interactively assembling large causal models. CausalSteward is a multi-agent collaborative system that tackles high-dimensional causality through a divide-and-conquer approach where large clusters of variables are iteratively partitioned and then separately analyzed. Our framework fuses prior knowledge with a data-driven approach by using tailored tools such as retrieval augmented generation and conditional independence tests. Finally, we use this work to examine the capabilities and limitations of causal reasoning in multi-agent frameworks, and how the human-in-the-loop can contribute to accurate and trustworthy results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
LLMs have recently been applied to providing background knowledge to causal discovery algorithms; however LLMs struggle in graphs with many nodes. This submission proposes a new method that employs divide-and-conquer to partition the graph into smaller graphs, that LLMs can then infer. These are combined into a larger graph.

### Strengths
- Original approach, as I have never seen divide-and-conquer in this context, while it does naturally fit into helping with large graphs.

- Theoretical properties; whose proofs look correct.

- Extensive empirical simulations

### Weaknesses
There are some glitches in experiments:
- LLM-BFS and Pairwise only seem evaluated on one LLM each, and a different one, making it hard to rule out an effet from the LLM alone in these baselines.
- "Therefore, we confirm (Q3)." (l.429). What do you confirm? (Q3) is an open-ended question, not a statement.
- Baselines are not evaluated on CausalChambers, and I actually do not see any results for Earthquake.
- There is no ablation study for the Explainer.
- ""Regarding scalability, the decrease in performance from CausalMan Small to Medium is only sub-linear, whereas data-driven baselines degrade by a factor of 4-6, while C A ST decreases sub-linearly by a factor of ≈ 2" (l.416-418) : for which methods and metrics? There is little change for BOSS on F1, and a sub-2 decrease in performance for XGES on SHD.
- CAST HITL+RAG is not consistently better than HITL or RAG alone: only 2 out of 6 instances on F1 in Table 1

### Questions
Can you answer/address the above?

In addition, which humans are interacting with the model? If these are authors themselves, how do you ensure that comparisons with baselines fair?

### Soundness
2

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
This paper proposes CAST for causal discovery, and CAST is the first attempt to aggregate human-in-loop, multi-agent collaborate and divide&conquer mechanisms. Experiments on  medium to large datasets all demonstrate its superiority on scability, and CAST achieves a good balance between F1 score and runtime.

### Strengths
1.	The paper provides sufficient theoretical analysis, including proofs for the ideal case and error bounds under realistic conditions.
2.	It’s the first attempt to integrate human-in-loop, multi-agent collaborate and divide&conquer to solve the causal identifiability and high dimensionality data challenges in causal discovery.
3.	Experiments show the superiority of the proposed method.

### Weaknesses
1.	The paper’s novelty may be somewhat limited, as its main contribution seems to lie in aggregating existing techniques. It would be helpful if the authors could better clarify the specific methodological innovations beyond this integration.
2.	The paper reports results for CAST without HITL but does not explain how the “human-in-the-loop” setting was conducted. There is no description of real human involvement or feedback collection. This makes the claimed benefit of human–AI collaboration unclear.
3.	The paper does not evaluate the effect of using different causal discovery algorithms within the Conquer phase. It remains unclear whether the observed improvements are due to the LLM hypothesis, the underlying causal discovery method, or their combination. An ablation study with alternative algorithms could clarify the contribution of each component.
4.	There is a minor error in the subsection Ablation for Divide & Conquer and Ablation for Critic Agents, it should refer to Figure 4 rather than Table 4.

### Questions
1.	How was the human-in-the-loop setting implemented? Can the authors provide quantitative or qualitative evidence for the claimed benefits of HITL?
2.	Can the authors clarify which aspects of their method constitute a novel contribution beyond the aggregation of existing techniques?
3.	Have the authors evaluated alternative causal discovery algorithms within the Conquer phase? How much of the observed performance improvement is attributable to the LLM hypothesis versus the underlying causal discovery method?
4.	In the Ablation subsections for Divide & Conquer and Critic Agents, the text refers to Table 4, but it should refer to Figure 4.

### Soundness
3

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
The paper introduces CausalSteward (CAST), a multi-agent human-in-the-loop (HITL) framework for causal discovery. CAST uses a divide-conquer-combine paradigm that integrates retrieval-augmented generation, human interaction, and data-driven causal discovery algorithms. 
The system decomposes high-dimensional variable sets into causally coherent partitions (divide), estimates local causal graphs through LLM-assisted hypothesis and critic agents combined with constraint-based discovery (conquer), and merges these into a global graph (combine). 
The authors provide a theoretical analysis demonstrating consistency under ideal causal partitioning (Theorem 1) and decomposing structural errors according to partition and merge inaccuracies (Theorem 2). 
They then compare CAST to other baselines with experiments on the CausalMan, Neuropathic-Pain, and CausalChambers benchmarks.

### Strengths
- CAST is the first agentic framework combining Divide-and-Conquer, RAG, and human-in-the-loop mechanisms for causal discovery.
- Theorem 2 provides a useful decomposition of SHD in terms of partitioning, merging, and LLM components. The formalization of causal partitioning and its link to theoretical guarantees is valuable.
- Interesting experimental questions. The paper explores whether LLM-assisted agents can meaningfully contribute to causal discovery and how human feedback and retrieval impact the results.
- Has promising early results. CAST performs competitively on moderately high-dimensional datasets where traditional algorithms (e.g., FCI) become intractable.

### Weaknesses
(I write one bulletpoint per weakness and then detail it below)

- Writing and presentation

1. Several sections (notably the Preliminaries) are written in overly informal language and contain repeated or redundant phrasing. Citations are missing or improperly formatted (see detailed feedback below in questions/suggestions). Some notation is inconsistent e.g., between $Pa_i$ (line 93) and $Pa_G$ (line 88), or Agent $\mathcal{D}$ (line 244) vs $\mathcal{D}_{hyp}$ (line 253).
2. Definition 3 ("Causal Partitioning") originates from Zhang et al. (2022) but is presented as if novel ("A mathematically rigorous way of grouping variables is the one of Causal Partitionings, which we define below."). It also violates the disjointness property of partitions. This should be clarified explicitly. It's more of an overlapping clustering, not a partition in the strict sense.
3. Figures 1 and 2 display only three phases, although the text describes four, which is a bit confusing.



- Conceptual and methodological concerns

1. The **technical novelty** is somewhat limited. CAST mainly orchestrates existing causal discovery and RAG/HITL components rather than introducing a new algorithm or estimator. The "method" is largely a structured prompting pipeline rather than a model or optimization procedure.
2. The theoretical results hinge on ideal causal partitioning and infinite data. No identifiability or performance guarantees exist when these conditions fail: precisely the realistic case.
3. The linear workflow may be restrictive. In real expert-in-the-loop settings, humans often revise earlier causal assumptions as understanding improves. Allowing back-and-forth updates between phases (rather than a one-directional pipeline) would perhaps better reflect practice.


- Orchestration vs. Model

Performance varies drastically with the chosen LLM (e.g., Qwen 14B > o3-mini).
This suggests that CAST's success may depend more on the LLM's reasoning power than on the method itself. A useful comparison would be a single-query baseline (running the whole causal discovery pipeline with a single model query) to demonstrate the added value of the multi-agent design.


- Experimental issues

1. Scalability interpretation is overstated: claiming sub-quadratic or "$\times{}2$ scaling" from only two data points (Small->Medium) is not statistically meaningful ("This indicates that CAST scales more robustly" on line 418). More datasets or controlled scaling experiments are needed.
2. Metric choice. Reporting orientation accuracy would clarify whether CAST's improvements concern causal directionality or merely edge presence. Additionally, reporting Structural Intervention Distance (SID) would be valuable, since the future work section mentions extending CAST to more general causal reasoning tasks involving interventional queries.
3. Stability not studied. Both LLMs and human interactions are stochastic and context-sensitive. No experiment measures how variable CAST's outputs are across runs, agents, or human responses.
4. Dataset suitability bias. The authors acknowledge that CausalMan Medium contains repeated patterns easily picked up by LLMs, which likely inflates CAST's apparent scalability.
5. CausalChambers interpretation: The explanation that extra predicted edges "might reflect unmodeled physics" seems unjustified and self-serving. The ground truth is physical. Such edges are just false positives.
6. Agentic RAG performs a web search. The paper does not describe any guardrails against retrieving benchmark ground truths (e.g., filtering domains, blocking known solution pages, offline corpora, or logging/inspecting retrieved URLs). Leakage seems possible here (and RAG seems important in the ablations).
7. While the experiments are interesting, they cover only three cases and do not empirically validate key claims about CAST's internal behavior. For example, there are no checks that agents successfully "translate human-provided information into the language of causality" or that the Divide phase produces causally consistent partitions. 
8. Similarly, the frequently stated advantage that CAST "does not require tedious manual specification" is not demonstrated, as all experiments were conducted by the authors themselves. A human-grounded study involving users with varying levels of causal expertise would help substantiate this claim.


- Ethical and safety aspects.

Even though the ethical statement is optional, its absence in this paper is notable. 
It's important to warn users about the risks of using public agents. Such agents can store/steal data. They can also be unfair and discriminatory (as is the internet sometimes), etc.


- Missing limitations (some mentioned already in other weaknesses)

CAST provides no guarantees or identifiability results when the Divide phase is imperfect, which could make the approach unreliable for sensitive applications. Moreover, the authors acknowledge that the evaluated datasets are particularly well-suited to the Divide-and-Conquer assumption, so performance might degrade on less structured data. Finally, no stability study was conducted to assess robustness to stochasticity in LLM outputs or human inputs.

### Questions
- Could RAG "leak" the exact target graph from the web? Did you prevent that?

- Did you log all agent-human interactions and retrieved documents? Will these be released for reproducibility?

- Can humans revise earlier decisions (e.g., change partitioning after seeing partial results)? If not, could CAST support iterative feedback loops rather than a linear pipeline?

- Why are only additional edges inserted rather than also removing edges inconsistent with the data? Is this due to relaxing faithfulness assumptions? Wouldn't this risk over-connectivity? (c.f. CausalChambers)

- We can use HITL for the divide phase, but it seems very difficult for a human to do such a task. How feasible is this in practice?

- Have you run any human-grounded experiments with real users of varying causal expertise to test whether CAST indeed avoids tedious manual specification?

- Add an ethics statement addressing data leakage, bias in retrieved content, and human accountability when CAST assists in high-stakes domains.

- Does "on demand" mean triggered by the agent or the human? Can it be parameterized (e.g., frequency of HITL queries)?


- Writing issues

1. Figures 1-2: The diagrams currently show only three phases, while the method actually consists of four (Explain, Divide, Conquer, Combine). The Explain phase is mentioned for the first time on line 199.
2. Subsection numbering: There is a 2.1 without a 2.2 (Line 157). 
3. Figure 1: Not referenced in the text.
4. Table 2 (precision column): The bolded value should be for LLM-BFS, not CAST, since LLM-BFS achieves the best precision.
5. Line 32: "Existing approaches" lacks a supporting citation.
6. Lines 38-39: Citations should appear in parentheses.
7. Line 98-99: Add a reference for MAG (Maximal Ancestral Graphs).
8. Line 183-184: Missing parentheses around citations.
9. Line 216: A citation is needed to support the statement.
10. Line 488: The author name "adamos hadjivasiliou" is entirely lowercase in the reference, capitalize properly.
11. The contributions paragraph in the introduction repeats content from the introduction itself. Consider combining them.
12. The paragraph on causal identifiability (Lines 108-116) is unclear and overly informal. Clarify wording and provide examples for "certain sets of mathematical examples" (Line 108).
13. Phrases such as "often not possible" and "often result" (Lines 111, 115) should either include supporting citations or be replaced with neutral statements like "cannot be" / "can result."
14. Line 360: The word "are" is missing.
15. Line 415: "Linked to the sparsity" is vague-clarify whether the issue is too sparse or not sparse enough.
16. Line 430: The notation $O(D^2)$ is unclear. Define $D$
17. There are two notations for parents: $Pa$ and $Pa^G$, but G is never introduced properly.
18. In the Divide phase, parameters appear inconsistently. Line 247: first occurrence of $p_{div.hyp}$, but only $p_{div}$ was defined. Line 252: first occurrence of $p_{div.critic}$, likewise undefined earlier.
19. $\mathcal{D}$ vs $\mathcal{D}_{hyp}$: The same symbol is used for both the dataset and the Divide-hypothesis agent. Please clarify.
20. Line 268: Notation inconsistency between $H$ and $C_{hyp}$.
Theorem 1: Relies on the infinite-data assumption; state this explicitly.
In the appendix, include at least one full dialogue between CAST and a human to illustrate the interaction process.

### Soundness
2

### Presentation
1

### Contribution
2
