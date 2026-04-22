# LocalV: Exploiting Information Locality for IP-level Verilog Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
The generation of Register-Transfer Level (RTL) code is a crucial yet labor-intensive step in digital hardware design, traditionally requiring engineers to manually translate complex specifications into thousands of lines of synthesizable Hardware Description Language (HDL) code. While Large Language Models (LLMs) have shown promise in automating this process, existing approaches—including fine-tuned domain-specific models and advanced agent-based systems—struggle to scale to industrial IP-level design tasks. We identify three key challenges: (1) handling long, highly detailed documents, where critical interface constraints become buried in unrelated submodule descriptions; (2) generating long RTL code, where both syntactic and semantic correctness degrade sharply with increasing output length; and (3) navigating the complex debugging cycles required for functional verification through simulation and waveform analysis.
To overcome these challenges, we propose \textit{LocalV}, a multi-agent framework that leverages the inherent \textit{information locality} in modular hardware design. LocalV decomposes the long-document to long-code generation problem into a set of short-document, short-code tasks, enabling scalable generation and debugging. Specifically, LocalV integrates hierarchical document partitioning, task planning, localized code generation, interface-consistent merging, and AST-guided locality-aware debugging. Experiments on \textsc{RealBench} demonstrate that LocalV substantially outperforms state-of-the-art (SOTA) LLMs and agents, showing the potential of generating Verilog for IP-level RTL design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LocalV, a multi-agent framework designed for IP-level Verilog generation that leverages the inherent information locality of modular hardware design. Instead of tackling the long-document to long-code generation challenge directly, LocalV decomposes it into a series of manageable short-document to short-code subtasks, thereby enhancing scalability in both generation and debugging. The framework incorporates three core innovations: a hierarchical indexing mechanism, a fragment-oriented task decomposition scheme, and a locality-aware debugging loop. Experimental results on the RealBench benchmark demonstrate that LocalV significantly surpasses existing state-of-the-art LLMs and agent-based systems, underscoring its potential for automating IP-level RTL design.

### Strengths
This paper contributes to automated hardware design by addressing the practical challenge of IP-level Verilog generation. The proposed LocalV framework introduces a coherent and technically grounded approach that leverages information locality in modular hardware design to decompose long-document to long-code generation into smaller, locality-aware subtasks. Its design combines a hierarchical indexing mechanism, fragment-based task decomposition, and a locality-guided debugging process, offering a structured solution to scalability and correctness issues in IP-level RTL synthesis. The experimental evaluation on RealBench provides evidence of LocalV’s effectiveness, showing consistent improvements over existing LLMs and agent-based methods. The paper is clearly presented and logically organized, making its motivation, methodology, and results easy to follow. Overall, it offers a meaningful step toward scalable and reliable RTL generation for practical IP-level design tasks.

### Weaknesses
While the paper introduces a promising framework, its technical differentiation from prior agent-based systems is not entirely clear. The authors emphasize that LocalV avoids cascading ambiguity by maintaining a direct link between specifications and code, but it remains uncertain whether this alone constitutes a substantial methodological advancement over existing frameworks such as MAGE or Spec2RTL-Agent. A more detailed comparison, highlighting architectural distinctions, workflow advantages, or specific design decisions beyond the specification-to-code linkage, would help clarify the scope of novelty. Furthermore, the approach fundamentally relies on the information locality hypothesis, yet the quantitative results in Section 3.2 show a noticeable gap in normalized entropy between idealized and real-world cases, suggesting that the assumption may not consistently hold across complex or loosely structured specifications. This raises concerns about robustness and generalizability of the proposed method. Lastly, the paper lacks a reproducibility statement or code-release plan, which limits the ability of other researchers to validate or extend the work.

### Questions
1. In Section 3.2, the authors use Qwen3-Embedding-0.6B to compute semantic similarities between specification and code fragments when validating the locality assumption. Could the authors elaborate on the rationale behind this choice? Have other embedding models been considered to verify the robustness of this assumption? Given that Qwen3-Embedding-0.6B may have limited understanding of Verilog semantics, it might produce misleading similarity scores. There are domain-specific embedding models such as DeepRTL2[1] that are trained on RTL data. Have the authors explored or considered using such models to obtain more accurate and representative results?
    
    [1] Liu, Yi, et al. "DeepRTL2: A Versatile Model for RTL-Related Tasks.” *Findings of the Association for Computational Linguistics: ACL 2025*. 2025.
    
2. The paper mentions that the raw specification documents are split into *coherent paragraphs* before hierarchical indexing. Could the authors clarify how this segmentation is performed? Is it based purely on structural markers or on semantic coherence using an LLM?
3. During the planning and task decomposition stage, the retriever is responsible for fetching the most relevant document fragments for each subtask. Have the authors quantitatively evaluated the retriever’s accuracy? Since errors at this stage could propagate through the generation pipeline, an assessment of retrieval precision and recall would strengthen confidence in the overall system reliability.
4. Although LocalV outperforms existing baselines, the overall functional accuracy on RealBench remains modest. Have the authors conducted any failure analysis to identify common error patterns or limitations in the failed cases? Insights into where LocalV struggles would help clarify the boundaries of the proposed method’s effectiveness and inform directions for future improvement.

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
5

### Summary
This paper proposes LocalV, a multi-agent framework for generating IP-level Verilog code from natural language specifications. The key idea is to leverage the information locality inherent in modular hardware design -- where each code fragment primarily depends on a small, corresponding portion of the specification. LocalV decomposes long-document to long-code generation into manageable sub-tasks, integrating hierarchical document indexing, structured planning, localized generation, merging with interface consistency, and locality-aware debugging.

### Strengths
1. The information locality is intuitive. By grounding the workflow in the modular structure of hardware specifications, this paper establishes a conceptual foundation for decomposing long, complex design documents into smaller, tractable tasks.

### Weaknesses
1. The granularity of document partitioning and task decomposition is unclear. Since different hardware modules can span multiple paragraphs or share interfaces, it is unclear how partition boundaries are chosen or adapted to maintain coherence across related fragments.
2. The merging and debugging stages appear to rely primarily on observations rather than formal verification. Without stronger guarantees of merging correctness or synthesized design quality (e.g., timing, area), it is difficult to assess how close LocalV comes to generating deployable industrial RTL.

### Questions
1. Does partition granularity impact performance? How does LocalV ensure coherence when multiple paragraphs describe interdependent modules?
2. Are the extracted paragraphs validated as meaningful semantic units, or could irrelevant segments mislead sub-task generation?
3. How is correctness ensured in the merging process -- does the system invoke simulators or rely only on syntactic checks?
4. The three challenges listed in the introduction section seem broadly applicable to other coding domains. What aspects are unique to Verilog generation?
5. Could you show per-iteration LocalV results in Fig.5(a)? Basically how does the debugging process in LocalV help improve performance?
6. How does LocalV's retrieval mechanism differ from standard RAG approaches? Any results comparing with RAG?
7. Apart from function correctness, are the implemented IP blocks efficient enough? Have you examined hardware efficiency metrics such as area or timing?
8. Would reasoning-enhanced models like DeepSeek-R1 further improve LocalV's planning or debugging performance?

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
This paper tackles the scalability bottleneck of large-language-model (LLM)–based RTL code generation.
Existing approaches struggle with IP-level (industrial-scale) designs because specifications are long, code is lengthy and inter-dependent, and debugging is hard.
The authors formalize the Information Locality principle — the assumption that each RTL semantic unit depends mainly on a small, localized portion of the specification. They quantify locality using a normalized-entropy metric based on embedding similarity and show that hardware specifications display stronger locality than software tasks.
Building on this, they propose LocalV, a five-stage multi-agent pipeline: hierarchical document indexing; task planning and skeleton generation; localized RTL generation; interface-consistent merging; AST-guided locality-aware debugging.
On the REALBENCH benchmark (AES, SDC, E203 CPU), LocalV achieves 75 % syntax and 45 % functional Pass@1 — about 23 % higher than the best agent baseline (MAGE).

### Strengths
1.	Quantitative validation of Information Locality (Eqs. 1–4, Fig. 3) is a novel and principled idea beyond heuristic agent planning.
2.	Dual-level indexing, deterministic planning, and AST-guided debugging form a coherent, effective pipeline.
3.	Strong improvement on REALBENCH over model and agent baselines.

### Weaknesses
1.	The framework's core assumption is the availability of a complete, detailed, and well-structured natural-language specification as input. This is a significant prerequisite that defines a more constrained problem setting than what many general-purpose agent systems address. In practice, such a document may not exist and would need to be generated from higher-level requirements, a task not covered by the proposed workflow. The paper should discuss this dependency, as the method's performance is tightly coupled to the quality of this assumed input.
2.	The Information Locality hypothesis is validated on a small set (one CPU and a handful of modules). The reported entropy gaps are modest, and the analysis lacks confidence intervals or sensitivity analyses for the embedding model used. The external validity across more IPs and different document styles (e.g., less structured or "flat" specifications) remains unsubstantiated.   
3.	The pipeline depends critically on the initial indexing and planning stages, yet there are no metrics for the accuracy of these steps or an analysis of how early mistakes propagate or are corrected. The current ablations only report final outcomes, while insights into process-level robustness are missing.   
4.	The ablation study shows large gains from debugging, but the paper fixes a 10-iteration debug cap without justification. There is no analysis of the cost-benefit trade-off (e.g., a Pass@k vs. iterations curve) or an equal-budget comparison against baselines. This makes it difficult to distinguish whether the performance advantage stems from methodological superiority or simply a larger budget for repair and search.   
5.	The debug agent lacks a reproducible algorithmic description. While the paper claims "AST-guided, locality-aware" debugging, it does not provide a clear workflow, localization hit-rates, or controlled ablations against simpler debugging strategies (e.g., without retrieval or without AST guidance) under the same budget.   
6.	Key details regarding efficiency and resource usage are missing, such as end-to-end runtime, total token consumption, and the complexity of the merging stage. The stopping criteria for the debugging loop are also unspecified.
7.	While the evaluation on REALBENCH is well-aligned with the paper's goals, its scope is narrow. The findings could be further strengthened by demonstrating performance on a broader set of benchmarks—such as HDLBench for larger SoCs and the relevant subsets of CVDP for agent-level tool integration—to provide additional insights into the method's scalability and robustness.
8.	Reproducibility is limited by the lack of variance reporting over multiple runs, full baseline configurations (e.g., temperature, sampling counts), and public access to data splits or evaluation scripts.
9.	The mechanism for ensuring interface consistency during the merging stage is not formalized. It is unclear how potential conflicts (e.g., macros, parameters) are resolved or what the computational complexity of this step is for larger designs

### Questions
1.	Regarding the baseline comparisons in Table 1, could you provide more details to justify the fairness of the evaluation? Specifically, can you clarify the resource budgets (e.g., total tokens, time, or iterations) allocated to LocalV versus MAGE/VerilogCoder? This would help distinguish the methodological advantages from potential differences in computational effort.
2.	The concept of 'information locality' is central to the paper. Could you elaborate on the characteristics of a specification document that are necessary for this principle to hold? For instance, how detailed or hierarchically structured must the document be for the locality to be strong? A discussion on the boundary conditions or the types of specifications where this principle might weaken would be valuable.
3.	For the debugging stage, could you provide a Pass@1 vs. #iterations curve (from 0 to 10) along with the associated time/token costs? This would help clarify the cost-benefit trade-off of the iterative repair process.
4.	Could you provide a more detailed, reproducible workflow (e.g., pseudocode) for the Debug Agent? It would be helpful to understand the distinct roles of AST analysis and locality-aware retrieval in the error localization and patching process.
5.	What is the "localization hit rate" of the debug agent? Specifically, what percentage of the time does it correctly map a simulation error to the responsible code fragment and, crucially, to the correct specification fragment?
6.	Could you provide accuracy/recall metrics for the initial hierarchical indexing and skeleton planning stages? It would be useful to understand how robust these critical early steps are.
7. Others see Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LocalV, a multi-agent framework for IP-level Verilog generation. The authors identify three fundamental challenges in generating IP-level Verilog code: long-document handling, long-code generation, and the complex debugging process. To address these challenges, LocalV leverages the inherent information locality in modular hardware design. The framework incorporates three key components: an index-driven document partitioning mechanism, a fragment-based generation strategy that decomposes complex tasks into manageable subtasks, and a traceable debugging pipeline that maps errors back to relevant specification fragments through AST-guided analysis. LocalV achieves a 45.0% pass rate on RealBench, surpassing state-of-the-art agent-based frameworks by 23.4%.

### Strengths
This paper addresses an underexplored problem: generating functionally correct Verilog code for large-scale, IP-level designs using large language models (LLMs). The proposed LocalV framework builds on the insightful observation of *information locality* in hardware specifications, demonstrating that modular design documents can be decomposed into semantically cohesive segments for localized code generation. The authors provide empirical justification for this hypothesis through an entropy-based locality analysis and integrate it into a multi-agent architecture comprising document partitioning, planning, localized generation, merging, and debugging. Experimental results on the challenging RealBench dataset show noticeable improvements over previous baselines, with clear component-level contributions validated by an ablation study. The paper is well written and systematically organized, and the workflow diagrams enhance readability and make the overall system easy to follow.

### Weaknesses
Despite its strong motivation, the technical novelty of LocalV is somewhat incremental compared to prior multi-agent systems such as MAGE and VerilogCoder. The main conceptual contribution lies in leveraging information locality to guide task granularity rather than introducing fundamentally new agent coordination mechanisms. The evaluation scope is also somewhat narrow, as RealBench is the only benchmark used. While the results are encouraging, it remains unclear how well the proposed approach generalizes to other hardware design domains or specification styles beyond those captured in RealBench. Moreover, the empirical analysis focuses primarily on Pass@k metrics and lacks a deeper investigation of failure cases. For example, scenarios where LocalV fails to produce functionally correct or syntactically valid code. Such analysis would provide valuable insights into the method’s current limitations and potential directions for improvement. Overall, while the observed gains are promising, they may not fully justify the added complexity of the multi-agent pipeline. The work would benefit from broader evaluation, more transparent implementation details, and stronger empirical evidence supporting the generality of the information-locality hypothesis.

### Questions
1. How sensitive is LocalV to the accuracy of the document indexing and retrieval steps?
2. Can LocalV effectively scale to system-level designs exceeding 2,000 lines of RTL code or to multi-module integration scenarios?
3. Will the authors release the LocalV implementation to facilitate reproducibility and community validation?

### Soundness
3

### Presentation
3

### Contribution
2
