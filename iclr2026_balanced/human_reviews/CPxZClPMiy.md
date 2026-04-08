## Human Reviewer 1

### Summary
The authors introduce two contributions, Aria, an autoformalization method/pipeline which decomposes and grounds an informal statement in Mathlib to ensure correctness, and AriaScore, a new metric which improves upon other alignment scorers such as LeanScore and back-translation. The authors conduct experiments to demonstrate the effectiveness of their contributions.

### Strengths
Both AriaScore and Aria are significant contributions to the community. The authors conduct experiments and error analyses to show that AriaScore performs better at measuring alignment than previous metrics. The authors also show that each of their three components in Aria are important to high performance as shown by an ablation study. Overall, the authors do a good job supporting their claims and have a strong contribution.

### Weaknesses
I think it's worth distinguishing this approach from papers such as (Liu et al, 2025), which also do RAG autoformalization.

An error analysis of the components of Aria would be very useful to understanding how this method improves. The authors make a claim (~L361) that hallucination of incorrect interfaces is the main source of error. However, why does a single-retrieval RAG system not accomplish a similar improvement in that case? Or perhaps it does, but just not to the same degree? Your ablation study answers this to some extent. But I believe an error analysis would be most beneficial.

### Questions
How does your system know the **informal** definitions of things it can't find in Mathlib, e.g., Cohen-Macaulay Module? My understanding is that this would be necessary information to be able to continue breaking down the definition into grounded terms, but I'm not clear on how this works.

While it's not in autoformalization, a similar technique as your "reflection" module has been used in similar fields such as automated theorem proving (see: COPRA from Thakur et al. 2024) among others. Maybe worth mentioning in related works.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper proposes **ARIA**, an agent for translating informal mathematical statements into Lean code through **Graph-of-Thought (GoT)** reasoning, **retrieval-augmented generation (RAG)** from Mathlib, and a **compiler-in-the-loop** feedback mechanism. It also introduces **AriaScorer**, a term-grounded semantic checker ensuring meaning consistency between informal and formal statements. Experiments on **ProofNet**, **FATE**, and real conjectures show **state-of-the-art** performance, demonstrating ARIA’s ability to handle complex, research-level formalization tasks.

### Strengths
- Introduces a **dependency-graph (GoT)** framework that mirrors human reasoning, enabling structured decomposition and synthesis of unseen mathematical concepts.
- Combines **RAG grounding**, **GoT planning**, and **compiler reflection** into a robust, self-correcting pipeline, with solid ablation evidence.
- Well-structured and clearly illustrated; figures and examples effectively explain the dependency graph process.
- Achieves **research-level formalization** beyond prior methods and improves semantic reliability through **term-level grounding** (AriaScorer).

### Weaknesses
- The paper does not provide the detailed prompts used in the key stages—decomposition, grounding, synthesis, and reflection. Since the entire pipeline relies on prompt-driven reasoning, the absence of these examples makes it difficult to reproduce the workflow or evaluate design choices. Including representative prompts or templates would significantly improve clarity and reproducibility.  

- All experiments are conducted on datasets from algebra and commutative algebra, limiting the demonstrated generality of ARIA. It remains unclear whether the dependency-graph and grounding strategies would perform equally well in other mathematical domains such as analysis, topology, or geometry. Broader testing or discussion on domain adaptation would strengthen the paper’s scope and impact.

- The LeanSearch component relies on a locally indexed Mathlib database, which introduces significant computational and storage overhead. A discussion of indexing efficiency, caching, or alternative lightweight retrieval strategies would make the system more practical for large-scale or real-time deployment.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper presents ARIA, an agent for auto-formalization of mathematical statements in Lean 4. ARIA integrates Retrieval-Augmented Generation (RAG), a Graph-of-Thought (GoT) planning mechanism, and a reflection loop guided by the Lean compiler. It also introduces AriaScorer, a semantic checker that retrieves term definitions from Mathlib for grounding-based verification. Experiments show improvements over prior baselines.

### Strengths
1. The paper clearly motivates improving statement-level formalization before theorem proving by introducing ARIA, an agent designed to convert natural language mathematical statements into Lean 4 formalizations.

2. It presents a well-structured and competently engineered system that integrates Retrieval-Augmented Generation (RAG), a Graph-of-Thought (GoT) planning mechanism, and a reflection loop guided by the Lean compiler, alongside AriaScorer, a semantic checker leveraging Mathlib for grounding-based verification.

3.It evaluates the model on various benchmarks, including conjecture-level tasks.

### Weaknesses
1. The paper claims novelty by integrating RAG, GoT, and the reflection loop guided by the Lean compiler. However, each of these components already exists in prior work. ARIA merely combines these elements without introducing a new algorithmic principle or theoretical insight. This makes the paper engineering incremental rather than conceptually innovative
2. The core metric is defined using AriaScorer, a tool introduced in the same paper, which introduces a potential self-evaluation bias: the observed performance gains may reflect alignment with the metric rather than genuine capability. Furthermore, only the Conjectures dataset has been human-verified, leaving the main benchmarks unchecked. Alternative metrics for autoformalizers, such as typecheck and BEq [1], exist, and reporting results under these metrics would provide a more robust and credible evaluation.
3. Table 1 compares ARIA with Goedel-V2, although most results reported are for pass@1. However, “pass@k” sampling is not equivalent to multi-stage agentic reasoning. Additionally, regarding the Conjectures dataset, I would like to ask why ARIA exhibits such a significant advantage on this data. It seems likely that other models may not have encountered this type of data during training, resulting in 0% accuracy for those models.

[1] Qi Liu, Xinhao Zheng, Xudong Lu, Qinxiang Cao, Junchi Yan. Rethinking and Improving Autoformalization: Towards a Faithful Metric and a Dependency Retrieval-based Approach

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper presents Aria, an agent system for auto-formalizing mathematical statements through a two-phase Graph-of-Thought (GoT) process: recursively decomposing statements into conceptual dependency graphs anchored to Mathlib via RAG, then bottom-up construction of formalizations synthesizing new definitions through compiler-in-the-loop reflection. The paper also introduces AriaScorer, a semantic checker achieving term-level grounding by retrieving actual definitions of Lean terms. Experiments demonstrate that Aria achieves 44.0% accuracy on FATE-X (baseline: 24.0%) and 42.9% on Homological Conjectures (all baselines: 0%), significantly outperforming existing methods.

### Strengths
- **Effective system architecture**: The paper successfully integrates GoT planning, RAG, and compiler-in-the-loop reflection to address the critical failure point of prior models in synthesizing new definitions. Ablation studies in Table 3 validate the necessity of each component: removing RAG or reflection causes accuracy to collapse from 42.9% to 0% on the Conjectures dataset.

- **Innovative contribution of AriaScorer**: This checker transcends text similarity limitations by employing the static analyzer jixia to retrieve actual definitions of Lean terms. Table 2 validates that this term-level grounding step is crucial for improving accuracy.

- **Breakthrough experimental results**: The system achieves 42.9% success rate on the Homological Conjectures benchmark where all baseline models completely fail (scoring 0%), demonstrating the unique capability and practical value of this compound approach for conjecture-level formalization.

### Weaknesses
- **Compound propagation risk of semantic errors**: The paper explicitly states that the GoT synthesis phase ensures only syntactic correctness and "cannot preclude correctly-typed but semantically wrong translations," while AriaScorer's semantic checking is described as a final step (Figure 1(C)). This creates a critical logical gap: if a base concept (e.g., "Cohen-Macaulay Module" in Figure 1) is semantically incorrect during synthesis, it will serve as a syntactically valid but semantically flawed premise for synthesizing all dependent parent nodes, invalidating entire dependency branches. The paper lacks crucial discussion or experimental validation on whether AriaScorer should be iteratively applied to each node during the synthesis phase to prevent this issue.

- **Insufficient explanation of contradictory phenomena in GoT ablation**: Table 5 shows that removing GoT paradoxically increases compilation rate (89%→95%) while decreasing final accuracy (71%→54%) on FATE-H. Although Section C.2 mentions this leads to "synthesis failure" and "interface hallucination" failure modes, it inadequately explains why a simpler monolithic approach systematically produces more syntactically simple (thus easier to compile) but semantically incorrect code. The fundamental mechanism of this "high compilation, low accuracy" failure mode remains unclear, weakening understanding of GoT's core value.

### Questions
1. **Regarding semantic error propagation mechanism**: Is AriaScorer's semantic checking iteratively applied to each newly synthesized node during the GoT synthesis phase (Figure 1(B)), or only performed once at the end? If not iteratively applied, how does the system prevent or recover from compound semantic errors caused by intermediate definitions that are "correctly-typed but semantically wrong"? Is there experimental data quantifying the impact and system robustness under such error scenarios?

2. **Regarding the deep mechanism of Table 5's anomaly**: Removing GoT increases compilation rate (89.0%→95.0%) but decreases accuracy (71.0%→54.0%) on FATE-H. Does this suggest that the non-GoT monolithic approach tends to produce syntactically simpler (thus easier to compile) but semantically incorrect formalizations? Can you provide concrete cases for in-depth analysis of the root cause of this "high compilation, low accuracy" failure mode? What does this reveal about GoT's core value (enforcing semantic structure vs. syntactic simplification)?

3. **Regarding reliability of critical decision points in GoT decomposition**: This process relies on an LLM reasoner to judge whether retrieved Mathlib candidates are "suitable matches" and decide whether to use candidates or trigger synthesis of new concepts. Overall pipeline performance critically depends on this decision's accuracy. How is this LLM reasoner's accuracy evaluated? Is there data showing the specific impact of its erroneous decisions (e.g., deciding to synthesize an existing concept or incorrectly matching an irrelevant concept) on end-to-end performance?

4. **Regarding robustness boundaries of AriaScorer**: This checker's term-level grounding depends on retrieving term information from the Herald dataset. What are AriaScorer's failure modes if Lean terms used in formalized statements are not included in this dataset, or if informal descriptions of terms in the dataset are inaccurate or ambiguous? Does the paper experimentally quantify the checker's robustness to incomplete grounding datasets? How does this affect the method's generalization capability to new domains or rapidly evolving Mathlib versions?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper introduces Aria, a system designed to perform autoformalization of high-level mathematical statements into Lean 4 syntax. Aria emulates human reasoning via a two-phase Graph-of-Thought (GoT) process: (1) recursively decomposing a conjecture into a dependency graph of definitions and sub-statements, and (2) constructing Lean-formalized statements using retrieved definitions from Mathlib. To enforce semantic precision, it employs AriaScorer, a retrieval-based verifier that grounds terms and checks logical consistency.
The model combines retrieval-augmented generation, graph reasoning, and self-reflection for improved correctness. On three benchmarks (ProofNet, FATE-X, and Homological Conjectures) Aria achieves strong improvements. The paper claims Aria is the first system to synthesize novel definitions during autoformalization, addressing a key bottleneck in automated mathematical reasoning.

### Strengths
1. The combination of retrieval, graph-of-thought reasoning, and verification is well-motivated and logically coherent. Autoformalization of research-level statements is a frontier problem, and this paper is impactful both for Lean and general symbolic reasoning research.

2. Performance is satisfactory. The system achieves SOTA results across diverse benchmarks, especially on difficult tasks where all prior models can't do well.

3. It tackles hallucination and definition synthesis, the two well-known weaknesses of current formalization LLMs.

4. The use of term-level retrieval (AriaScorer) for semantic grounding is a technically meaningful contribution that enhances reliability.

### Weaknesses
1. More ablation study can make it even better. The contribution of each submodule (retrieval, GoT reasoning, reflection, AriaScorer) is not quantified in detail; ablation or controlled comparison would strengthen claims.

2. Some discussion on Aria's generality to other systems (e.g., Coq, Isabelle, Lean 3) will make it more impactful.

3. While promising, the "first to synthesize novel definitions" claim may need stronger empirical evidence (e.g., human evaluation verifying novelty and correctness).

4. More detailed discussion on computational cost or efficiency will be very beneficial for assessing scalability.

### Questions
Addressing the points I mentioned in weakness part should be enough.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 6

### Summary
This paper proposes Aria, a comprehensive agent-based autoformalization pipeline designed to emulate human reasoning for research-level mathematics. The system first performs graph-of-thought decomposition, transforming natural language statements into dependency graphs and grounding the identified concepts in mathlib when possible. For ungrounded concepts, it employs graph-of-thought synthesis to generate Lean representations through a self-reflective loop.

The authors further introduce AriaScorer, which decomposes informal statements into atomic assumptions and conclusions, then aligns them with formal clauses retrieved from Lean’s library and their corresponding natural language versions to compute an aggregated correctness score.

Experimental results demonstrate that Aria significantly outperforms existing LLM-based autoformalization baselines on the ProofNet, FATE benchmarks, and real-world conjectures. Moreover, AriaScorer achieves more accurate verification of formalization correctness compared to other evaluation methods.

### Strengths
* The paper is well-written and well-motivated. Its focus on research-level mathematical autoformalization is interesting.  
* The proposed Aria and AriaScorer frameworks are novel, integrating several intuitively and carefully designed components into a cohesive pipeline.  
* Experimental results demonstrate that both Aria and AriaScorer are effective, outperforming existing methods by a substantial and consistent margin.  
* The key insights underlying AriaScorer are also inspiring, offering a new perspective on evaluating the correctness of autoformalization.

### Weaknesses
There is no human evaluation in the main results, which makes the reported performance less reliable. Moreover, since AriaScorer uses Gemini for both generation and evaluation, the results may be biased toward its own generated responses, especially when compared to other baselines such as Goedel-Autoformalizer.

### Questions
* Could you include some manual evaluation on the main experiments to better assess the quality and reliability of the autoformalized results?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4