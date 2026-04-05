=== CALIBRATION EXAMPLE 56 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title captures the broad idea—using Mermaid-based graphs plus evolutionary programming for agentic workflow generation—but “safety-constrained evolutionary programming” is somewhat stronger than what the paper actually substantiates. The core safety claim is mostly about syntactic/structural validity, not broader safety in the ML sense.
- The abstract clearly states the problem and the proposed framework, but the main result claims are vague. It says “consistent improvements in success rates and faster convergence,” yet does not name the benchmark(s), the magnitude of gains, or what “faster convergence” means.
- The claim that MermaidFlow “redefines the agentic search space” and provides a “verifiable intermediate representation” is plausible, but the abstract overstates the novelty and strength of guarantees relative to the actual implementation, which still relies on an LLM-based Mermaid-to-Python translation and a custom checker.

### Introduction & Motivation
- The motivation is relevant for ICLR: workflow generation for multi-agent LLM systems is indeed fragile, and a more structured search space is a sensible idea.
- The gap in prior work is partially identified: code-centric or prompt-centric workflows are brittle and hard to verify. That said, the introduction largely positions MermaidFlow as if it solves the core reliability problem, without sufficiently distinguishing syntax/structure verification from semantic correctness.
- The contributions are stated clearly, but several are overstated:
  - “first agentic workflow framework to guarantee static graph-level correctness across the entire generation process” is too strong, because correctness is only with respect to a limited set of handcrafted constraints in Appendix A.2, not general semantic correctness.
  - “programmable and task-agnostic programming layer” is not convincingly demonstrated by the experiments, which are limited to four benchmarks and two task families.
- For ICLR standards, the introduction needs a more careful delimitation of what is genuinely guaranteed versus what is empirically improved.

### Method / Approach
- The method is understandable at a high level: represent workflows as Mermaid graphs, enforce a small set of structural constraints, and search over graph edits with evolutionary operators.
- However, the technical description has several gaps:
  - Section 3.1/3.2 define the graph representation, but the formalism is not precise enough to support the strong claims. Equation (1) is informal, and the “types” and “static verifiability” are described in prose more than in a rigorous specification.
  - The “static verifier” \(Q\) in Definition 1 is only a binary validator for membership in \(S\), where \(S\) is defined by a small set of constraints. This is much weaker than the paper’s repeated “guarantee” language suggests.
  - The search space closure claim in Lemma 1 depends on the operators being constraint-preserving, but the “proof” is effectively definitional. There is no real proof of preservation beyond restating the operators’ intended conditions.
- The operators in Section 4.1 are conceptually sensible, but several are underspecified:
  - “Node substitution,” “edge rewiring,” “subgraph mutation,” and “crossover” need more exact pre/postconditions to be reproducible.
  - The paper does not explain how the system ensures that the Mermaid graph remains executable after translation into Python beyond the checker and an LLM translator.
- Important failure modes are under-discussed:
  - A graph can be syntactically valid yet semantically poor.
  - The LLM-based translation step can reintroduce errors after a valid Mermaid graph is found.
  - The operators may preserve type/interface constraints while still degrading task performance.
- The method also appears to mix roles between search, evaluation, and generation in a way that is not fully cleanly separated. For instance, the optimization LLM both proposes workflow graphs and translates them to Python, which can confound the claimed modularity.

### Experiments & Results
- The experiments do test the claimed setting: benchmark-driven workflow generation for GSM8K, MATH, HumanEval, and MBPP, compared against workflow-generation baselines.
- The baseline set is broad and relevant overall, though it mixes very different paradigms (non-agentic prompting, hand-crafted systems, and autonomous systems). That is acceptable if the goal is to situate the method broadly, but the paper should be more explicit about which baselines are the closest competitors on the same optimization problem.
- There are several concerns about fairness and interpretation:
  - The method uses GPT-4o-mini for both optimization and execution, while some baselines may not be tuned identically or may rely on different prompting/implementation details. The paper says it uses the same base LLM “consistent with MaAS,” but the setup is still not fully enough to rule out prompt- or implementation-level confounds.
  - Table 1 reports averages over three runs, but no error bars, standard deviations, or significance tests are given. For an ICLR paper, especially one claiming consistent improvement, this is a notable omission.
  - The average improvement over MaAS and AFlow is modest, and without variance it is hard to tell whether the gains are robust.
- Some experimental claims are not fully supported:
  - The paper attributes gains partly to “faster convergence” and “token efficiency,” but only one learning-curve figure is shown, and it is not enough to establish systematic efficiency gains across tasks.
  - The claim that MermaidFlow “consistently yields >90% success rate in producing valid Python code” is important, but the paper does not provide a direct table or ablation quantifying this success rate across tasks, only a textual statement in Section 5.3.
- Material missing ablations:
  - There is no clear ablation isolating the contribution of the Mermaid representation versus the evolutionary operators versus the checker.
  - A crucial ablation would compare: code-level search, Mermaid search without constraint enforcement, and Mermaid search with only some operators.
  - Another important ablation would remove the Mermaid-to-Python translation step or compare different translators, because the final performance depends on it.
  - The paper also does not test how much of the gain comes from the extra candidate generation budget versus the representation itself.
- The evaluation metrics are appropriate for the tasks chosen, but the paper’s framing would benefit from a more direct measure of structural validity, search efficiency, and translation success rate, not just downstream task score.

### Writing & Clarity
- The paper is generally readable at the high level, but several sections are conceptually repetitive and some claims are not cleanly distinguished from implementation details.
- The most confusing part is the interplay between Mermaid, the static checker, evolutionary search, and Python generation. The pipeline is described in multiple places, but the exact flow of information is hard to reconstruct without reading the appendix carefully.
- Figures and tables are mostly informative, but they do not always provide the quantitative support needed for the claims:
  - Figure 3 is about MATH learning curves, but it is used to support broader claims about efficiency.
  - Table 1 shows final scores but not statistical uncertainty.
  - The appendix cases are useful, but the main paper should include at least one concrete example of a successful Mermaid-to-Python translation path.
- The method section would benefit from more explicit notation and a sharper separation between the formal representation, the operator design, and the empirical pipeline.

### Limitations & Broader Impact
- The paper does acknowledge some limitations in the conclusion and future work, especially that real-world integration and user-in-the-loop workflows require more study.
- However, the key limitation is underemphasized: MermaidFlow does not guarantee semantic correctness or task correctness, only a constrained form of structural validity.
- Another major limitation is that the system still depends on an LLM for graph generation, graph selection, translation, and judging. So the “safety” and “verifiability” gains are only partial.
- Broader impact is lightly treated. For an ICLR submission, it would be helpful to discuss:
  - whether more structured agent workflows could enable more scalable deployment of autonomous systems,
  - whether the approach could inadvertently make agentic systems more convincing without being more reliable,
  - and whether the workflow-search machinery could be repurposed in ways that increase misuse risk.
- The paper also does not discuss whether the Mermaid-based abstraction constrains expressivity enough to hinder tasks that need richer control flow, which the future work section briefly hints at but does not analyze.

### Overall Assessment
MermaidFlow is a promising and timely idea: it introduces a structured workflow representation and constraint-preserving search that plausibly improve reliability and search efficiency for agentic systems. The empirical gains in Table 1 are real and the approach is interesting for ICLR. However, the paper’s strongest claims are overstated relative to the evidence. The “static correctness” guarantee is only about limited graph constraints, the method still relies heavily on LLMs for generation and translation, and the experiments lack key ablations, uncertainty estimates, and stronger evidence that the Mermaid representation itself—not just more search or more prompting—drives the gains. I think the paper has a solid core contribution, but to meet ICLR’s bar convincingly, it needs a much more careful technical framing and substantially stronger experimental disentanglement.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes MermaidFlow, a workflow-generation framework that represents agentic workflows as Mermaid graphs and applies safety-constrained evolutionary programming to search this space. The main claim is that a declarative, statically verifiable representation enables safer mutations, better search efficiency, and improved downstream performance on math and code benchmarks compared with existing agentic workflow methods.

### Strengths
1. **Clear systems-level idea: declarative, graph-based workflow representation.** The paper’s central design choice—using Mermaid as a human-readable intermediate representation for agent workflows—is concrete and well-motivated. The authors argue that it separates planning from execution and enables static checking before runtime, which is aligned with ICLR’s interest in principled abstractions for LLM systems.

2. **Safety-aware evolutionary search is a sensible contribution.** The paper defines several graph-level operators (substitution, addition, rewiring, deletion, subgraph mutation, crossover) intended to preserve validity by construction. This is a meaningful attempt to constrain the search space, and the paper explicitly ties these operators to compiler/checker-level validation.

3. **Broad empirical evaluation across four benchmarks.** MermaidFlow is evaluated on GSM8K, MATH, HumanEval, and MBPP, and the paper reports consistent gains over many baselines, including strong workflow baselines like AFlow and MaAS. The table suggests nontrivial improvements, especially on MATH and overall average score.

4. **Emphasis on interpretability and modularity.** Compared with code-centric workflow generation, the Mermaid representation is easier to visualize and inspect. The paper repeatedly highlights the benefit of human readability, which is a practical advantage for workflow debugging and iteration.

5. **Includes implementation details and validation rules.** The appendix provides checker logic, operator descriptions, and algorithmic pseudocode, which improves the transparency of the proposed pipeline relative to many agentic-workflow papers.

### Weaknesses
1. **Novelty is somewhat incremental relative to prior workflow-search literature.** The paper combines known ingredients—graph representations, evolutionary search, validation/checking, and LLM-as-judge selection—but does not clearly isolate a new algorithmic principle beyond “search in a safer representation.” At ICLR, this may read as a useful engineering synthesis rather than a fundamentally new learning or optimization method.

2. **The “static verifiability” claim appears overstated.** The paper repeatedly says MermaidFlow “guarantees” correctness, but the implemented checker only enforces syntactic and a few structural rules (e.g., interface nodes, connectivity, ensemble arity, valid types). That does not imply semantic correctness, task correctness, or even executable correctness in the strong sense suggested by the writing.

3. **Empirical methodology raises reproducibility and fairness concerns.** The system uses closed-source GPT-4o-mini both for optimization and execution, plus an LLM-as-judge for candidate selection. This introduces a large amount of hidden variance and makes it difficult to attribute gains to the representation itself rather than to the judge/optimizer model. The paper also does not clearly establish whether all baselines are equally tuned under the same budget and interface constraints.

4. **Ablations are limited relative to the paper’s claims.** The main message is that the representation and the evolutionary operators are responsible for the gains, but the paper does not convincingly disentangle these factors. For example, there is no strong ablation isolating Mermaid representation vs. the checker vs. the evolutionary operators vs. the LLM judge.

5. **Theoretical results are weak and somewhat formalistic.** The lemma about transformation invariance is essentially a restatement of closure under the allowed operations, but the proof content is minimal. It does not yield a deeper guarantee about optimization quality, convergence, or robustness.

6. **Some experimental claims are hard to interpret.** The improvements are reported over averages across heterogeneous tasks, but the paper does not clearly analyze variance, significance, or whether gains are practically meaningful on each benchmark. The reported “faster convergence” is plausible, but the learning-curve evidence appears limited.

7. **Prompting and workflow design may dominate performance.** The appendix shows highly task-specific prompts and extensive handcrafted instructions. This makes it difficult to know whether MermaidFlow’s improvement comes from the graph representation or from careful prompt engineering and workflow tailoring.

### Novelty & Significance
**Novelty:** Moderate. The paper’s novelty lies in combining Mermaid-based declarative workflow representation with constrained evolutionary search for agentic systems. While the representation choice is interesting and potentially useful, the core optimization idea is close to existing evolutionary/graph-based workflow search work.

**Significance:** Moderate to potentially strong if the claims hold up. For ICLR, the idea is relevant because it targets robustness, interpretability, and structured search for LLM agents. However, the significance is limited by the reliance on closed models, the lack of stronger ablations, and the somewhat inflated correctness guarantees.

**Clarity:** Moderate. The high-level story is understandable, but the paper sometimes conflates syntactic validity with semantic correctness and overclaims “guarantees.” The appendix is more operationally informative than the main text, but the presentation of operators, validation, and evaluation is not always crisp.

**Reproducibility:** Moderate to low. The paper provides substantial implementation detail, but the use of proprietary models, an LLM judge, and task-specific prompts makes full reproduction difficult. It would be stronger with open-model experiments and a more controlled accounting of compute and prompt budgets.

**ICLR acceptance bar assessment:** Promising but below a clear accept threshold in its current form. ICLR typically expects either a clearly novel methodological advance or a strong empirical result with careful analysis. This paper has a compelling systems idea and solid results, but the methodological novelty and empirical rigor need strengthening.

### Suggestions for Improvement
1. **Add stronger ablations.** Separate the effects of:
   - Mermaid representation vs. Python/code representation,
   - evolutionary operators vs. plain search,
   - checker-based filtering vs. no filtering,
   - LLM-as-judge vs. simpler selection strategies.

2. **Tone down and formalize correctness claims.** Replace “guarantee static graph-level correctness” with a more precise statement such as “guarantee syntactic and structural validity under the implemented checker rules.” Distinguish structural validity from task success.

3. **Run open-model and controlled-budget experiments.** To improve reproducibility and fairness, test with open-source LLMs and report compute/token budgets in a normalized way across methods.

4. **Provide statistical significance and variance.** Report confidence intervals or standard deviations across more runs, and test whether gains are statistically reliable, especially for smaller improvements.

5. **Clarify the role of the LLM judge.** The paper should explain how the judge is prompted, whether it introduces bias, and how sensitive results are to judge/model choice.

6. **Improve the theoretical contribution.** If possible, add a more meaningful analysis of the search space, such as properties of operator closure, constraints on convergence, or why the representation reduces invalid proposals in practice.

7. **Include stronger qualitative case studies.** Show concrete workflow evolution traces: parent graphs, mutations, candidate rejection by the checker, and how the final graph improves task performance. This would substantiate the claimed interpretability benefit.

8. **Reduce task-specific handcrafting or isolate it.** Since much of the performance may depend on carefully tuned prompts, explain what is generic and what is benchmark-specific, and test transfer to a new task suite if possible.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct ablation of the representation itself: compare MermaidFlow against the same EP/search procedure over Python/JSON/LangGraph plans with the same operator budget. Without this, the claim that Mermaid’s declarative graph space is the cause of gains is not established.  
2. Add an ablation of each evolution operator and the safety checker: remove crossover, subgraph mutation, insertion/deletion, and static validation one at a time. ICLR reviewers will expect evidence that the reported gains are not just from more sampling or more tokens.  
3. Compare against stronger and more recent workflow-search baselines under matched budgets, especially EvoFlow, ScoreFlow, and multi-agent design/search methods with identical LLMs and iteration limits. The current comparison mixes methods with different search resources and does not isolate algorithmic advantage.  
4. Report compute-normalized results: success rate vs. token cost, wall-clock time, number of valid candidates generated, and number of LLM calls. The paper claims faster convergence and efficiency, but only one token comparison on one dataset is shown.  
5. Add cross-dataset generalization experiments: tune workflow structures on one benchmark and transfer to another within the same domain. Without transfer evidence, the “task-agnostic” and reusable workflow claims are too weak for ICLR.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much of the improvement comes from “static verifiability” versus ordinary prompt ensembling. The paper attributes gains to the Mermaid search space, but never measures whether validity constraints actually improve downstream accuracy or just reduce malformed outputs.  
2. Analyze search dynamics over rounds: valid-rate, candidate diversity, parent-selection entropy, and improvement trajectory per round. The current convergence claim is not trustworthy without showing whether search truly explores better workflows or merely converges to a narrow template.  
3. Provide a formal or empirical account of the correctness guarantee. The paper says “guarantee static graph-level correctness,” but the validator only checks a limited set of syntactic/structural rules and still relies on LLM-generated translation to Python.  
4. Evaluate sensitivity to the judge model and judge bias. Since an LLM-as-judge selects candidates, the method may simply optimize to the judge rather than the task; this needs inter-judge agreement or oracle-vs-judge analysis.  
5. Break down performance by task difficulty and failure mode. ICLR reviewers will want to know where MermaidFlow helps, where it fails, and whether gains concentrate on easy instances, code tasks, or specific workflow motifs.

### Visualizations & Case Studies
1. Show before/after workflow evolution traces for several instances, including rejected candidates and why they were rejected. This would reveal whether the operators make meaningful improvements or just cosmetic edits.  
2. Visualize the distribution of workflow graph sizes, operator usage frequencies, and how often each operator leads to score gains. Without this, the evolutionary mechanism looks underspecified and potentially arbitrary.  
3. Provide failure case studies where MermaidFlow generates a valid graph but a poor solution, and where the validator accepts a graph that still fails at execution. This is necessary to distinguish structural validity from actual reasoning quality.  
4. Show a side-by-side comparison of Mermaid graph, translated Python, and execution traces for a few representative tasks. The claimed separation between planning and realization is only convincing if the translation is faithful in practice.  
5. Include an error taxonomy with concrete examples of invalidity, translation failure, and judge failure. This would expose whether the system’s robustness comes from true workflow quality or from filtering out obvious failures.

### Obvious Next Steps
1. Replace the current LLM-based Mermaid-to-Python translation with a deterministic compiler or verified transpiler. The paper itself identifies this as future work, and without it the “compiler-verifiable” claim is not convincing end to end.  
2. Extend the evaluation to harder and more realistic agentic tasks beyond GSM8K/MATH/HumanEval/MBPP, such as long-horizon tool use or retrieval-augmented multi-step tasks. ICLR expects broader evidence than standard benchmark gains on short-form tasks.  
3. Test whether MermaidFlow can operate with smaller or open models for both optimization and execution. Relying on gpt-4o-mini everywhere limits the contribution’s methodological significance and raises concerns about cost and portability.  
4. Add a human study on interpretability and editability of Mermaid workflows versus Python workflows. The paper repeatedly claims readability and reusable structure, but never demonstrates that humans can actually inspect or modify the graphs better.

# Final Consolidated Review
## Summary
This paper proposes MermaidFlow, a framework that represents agentic workflows as Mermaid graphs and searches over them with safety-constrained evolutionary programming. The main idea is to make workflow generation more structured, statically checkable, and easier to mutate than code-level or prompt-level agent pipelines. The paper reports improvements on GSM8K, MATH, HumanEval, and MBPP over a range of agentic baselines.

## Strengths
- **The core representation choice is sensible and practically useful.** Modeling workflows as Mermaid graphs does make the planning structure explicit, human-readable, and easier to validate than raw Python or loosely structured prompts. The appendix shows concrete node types, checker rules, and translation steps, so this is not just an abstract idea.
- **The empirical results are consistently better than the compared baselines.** Table 1 shows MermaidFlow leading on all four benchmarks, including nontrivial margins over strong workflow baselines like AFlow and MaAS. The paper also provides evidence that the graph-space search is more token-efficient than AFlow on MATH, and that stronger optimization LLMs improve results, which is at least consistent with the proposed search formulation.

## Weaknesses
- **The main correctness/safety claim is overstated.** The paper repeatedly talks about “guaranteed” or “static graph-level correctness,” but the actual validator only enforces a narrow set of syntactic and structural constraints: connectivity, node types, interface presence, and ensemble arity. That is useful, but it is not semantic correctness, not executable correctness end-to-end, and not enough to justify the strongest claims in the abstract and introduction.
- **The experimental evidence does not isolate what is actually helping.** The paper reports better end-task performance, but it does not convincingly disentangle the effect of Mermaid representation, the checker, the evolutionary operators, the LLM judge, and the heavy prompt engineering. Without ablations against code-level search, no-checker search, and operator-by-operator removal, it is hard to know whether the gains come from the representation or simply from a larger, more carefully prompted search pipeline.
- **Methodological rigor is limited for a paper making robustness claims.** There are no error bars, confidence intervals, or significance tests in the main results, and the “faster convergence” and “token efficiency” claims are supported only weakly. Given the modest absolute gains over some baselines, the lack of variance reporting makes the robustness of the improvement hard to assess.
- **The system still depends heavily on LLMs at every stage.** The same or similar models are used for workflow generation, judging, translation to Python, and execution. That undermines the framing of MermaidFlow as a cleanly “compiler-verifiable” pipeline, because the most error-prone parts are deferred to LLM components rather than removed.

## Nice-to-Haves
- Add a clearer breakdown of which improvements come from the Mermaid representation itself versus the checker, judge, and evolutionary search.
- Report standard deviations or confidence intervals over more runs, especially for the smaller gains.
- Show a few end-to-end examples of workflow evolution, including rejected candidates and the corresponding checker failures.

## Novel Insights
The most interesting aspect of the paper is not the evolutionary search itself, but the decision to move workflow generation into a declarative graph language with explicit type- and interface-like constraints. That shift changes the optimization problem from “generate a correct program directly” to “search over a constrained, inspectable workflow space,” which is a legitimate systems-level insight for agentic LLM pipelines. The paper’s stronger-than-justified claim is that this yields broad safety guarantees; in reality, the contribution is narrower but still meaningful: it reduces obvious structural failure modes and makes workflow search more stable and editable, which plausibly explains the observed gains.

## Potentially Missed Related Work
- **LangGraph / graph-based agent orchestration frameworks** — relevant because they also represent agent workflows as graphs, though not necessarily with the same constrained evolutionary search setup.
- **EvoFlow** — directly relevant as a recent evolutionary workflow search baseline the paper already cites and should discuss more explicitly in relation to its operator design.
- **ScoreFlow** — relevant as another workflow optimization approach; useful for comparing whether search over structured workflows is actually the key ingredient.

## Suggestions
- Replace “guarantee static correctness” language with a precise statement about the implemented structural checks.
- Add ablations for: representation, checker, each evolutionary operator, and LLM judge.
- Include variance/significance reporting and normalize results by token cost and number of LLM calls.
- Show a direct comparison between Mermaid-based search and the same search procedure over Python or JSON workflow representations under matched budgets.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0]
Average score: 5.3
Binary outcome: Reject
