# RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 2

## Abstract
Large Language Models (LLMs) can enhance their reasoning capabilities by using external tools. However, many tasks lack predefined tools. Prior works have explored instructing LLMs to generate tools on their own, but such approaches depend heavily on internal knowledge and struggle when tasks fall outside the model’s knowledge scope. To address this limitation, we propose RefTool, a reference-guided framework for automatic tool creation that leverages external materials, such as textbooks and knowledge snippets. RefTool consists of two modules: (1) tool creation, where LLMs generate executable tools from reference content, validate them using illustrative examples, and organize them hierarchically into a toolbox; and (2) tool utilization, where LLMs navigate the toolbox structure to select and apply the appropriate tools to solve problems. Experiments on causality, physics, and chemistry benchmarks demonstrate that RefTool outperforms existing tool-creation and domain-specific reasoning methods by 12.3% on average accuracy, while being cost-efficient and broadly generalizable to non-scientific tasks, e.g., extremely low-resource language translation. Analyses reveal that grounding tool creation in references produces accurate and faithful tools, and that the hierarchical structure facilitates effective tool selection. RefTool enables LLMs to overcome internal knowledge limitations, advancing generalizable reasoning in knowledge-intensive domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces REFTOOL, a framework for tool-augmented reasoning designed to enhance the inferential capabilities of  LLMs in knowledge-intensive domains that lack predefined tools. REFTOOL leverages external materials to perform hierarchical tool creation, operating through two primary modules:

Tool Creation: The LLM processes reference content to generate executable tools. Each tool is generated with a description, its function code, and an illustrative example. These tools are subsequently validated using their own examples and optimized as necessary. Finally, the validated tools are organized into a hierarchical toolbox that generally mirrors the structure of the reference material.

Tool Utilization: During the reasoning process, when a query is encountered, the LLM executes a hierarchical selection procedure. It will select the most relevant category and subsequently selects the most appropriate tool from within that category. The selected tool is then employed to generate the final solution.

Experiments were conducted on causality, physics, and chemistry benchmarks. The results demonstrate that REFTOOL's mean accuracy significantly outperforms existing tool-creation methods, domain-specific models, and RAG-based approaches by an average of 12.3%. The paper also shows that the framework is cost-effective and exhibits high generalizability, having been successfully applied to non-scientific tasks using unstructured reference materials.

### Strengths
1. Novel Framework for Knowledge Proceduralization: The paper introduces a novel paradigm that bridges declarative knowledge retrieval with executable tool use. By compiling procedural knowledge from static reference materials into a persistent, validated, and structured toolbox, the framework (REFTOOL) presents a significant conceptual advance over standard Retrieval-Augmented Generation (RAG) and pre-defined tool-use methodologies.

2. Robust Methodological Design and Validation: The framework's two-module architecture is methodologically sound, incorporating critical components for robustness, such as automated tool verification, refinement loops , and a hierarchical selection mechanism. The efficacy of these design choices is rigorously validated through comprehensive ablation studies, which demonstrate the clear superiority of code-based tools over raw text retrieval and hierarchical selection over standard similarity-based retrieval.

3. Rigorous Experiments and Generalizability Validation:  The effectiveness of REFTOOL is substantiated through extensive experiments across three distinct, knowledge-intensive scientific domains, showing consistent and significant performance gains over a wide array of strong baselines. Furthermore, the framework's generalizability is impressively demonstrated by its successful application to a non-scientific task using unstructured reference materials, highlighting its flexibility and broad applicability.

### Weaknesses
1. Dependency on the Capability of the Base Model for Tool Creation: The core function of the REFTOOL framework relies on a specific base model to generate high-quality and validated tools. However, the framework's performance varies significantly under different base models. Supplemental experiments in Appendix B.4 show that other models, such as Gemini-1.5-Pro and Llama-3.1-70B, have much lower initial validation success rates. For example, in the physics domain, the success rate was 54% versus 82%.

2. Insufficient handling of prerequisite foundational knowledge. The design of the framework implicitly assumes that the knowledge in reference materials is entirely comprehensible. However, in specialized fields such as chemistry, textbooks often presuppose that readers have mastered undefined foundational knowledge (e.g., IUPAC nomenclature, SMILES notation). If the tool creation model misinterprets these prerequisite foundational concepts, the model may fail to generate correct tools.

### Questions
This paper's core premise is that REFTOOL can overcome an LLM's internal knowledge limitations by leveraging reference materials. However, in specialized domains like chemistry, reference materials often assume a baseline of foundational knowledge rather than explicitly teaching it. For instance, a textbook may freely use representations like SMILES strings or IUPAC nomenclature in its examples and formulas, assuming the reader is already proficient.

During Tool Creation: If the base LLM lacks reliable pre-trained knowledge of (e.g.) SMILES, it may fundamentally misinterpret a textbook example and thus generate a semantically flawed tool. The Tool Verification step, which relies on a model-generated example, could also fail to catch this error if the model produces a self-consistent but incorrect validation case, leading to the propagation of a bad tool.

During Tool Utilization: Even if a tool is perfectly created (e.g., calculate_property(smiles_string)), it becomes practically useless if the LLM, when given a natural language query ("calculate the boiling point of ethanol"), cannot perform the implicit prerequisite task of translating "ethanol" into its required SMILES format ("CCO"). This adds an additional reasoning burden that the framework was intended to solve, not create.

Does the REFTOOL architecture possess a specific design or mechanism to address this gap in implicit foundational knowledge?

For example, does the framework (a) simply rely on the comprehensiveness of the reference material to also teach these fundamentals, or (b) does it include a more active mechanism to detect these dependencies and recursively build a foundational library of prerequisite tools (e.g., a chemical_name_to_smiles converter) that the higher-level scientific tools can then rely upon?

### Soundness
4

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
This paper presents REFTOOL, a framework that automatically builds and uses code-based tools for knowledge-intensive reasoning. It works in two stages: (1) tool creation, where an LLM reads reference materials, generates and validates functions, and organizes them into a hierarchical toolbox; and (2) tool use, where the LLM selects and applies these tools to solve new problems. Experiments in causality, physics, and chemistry show that REFTOOL outperforms existing tool-creation and domain-specific methods, and it also generalizes well to a low-resource language translation task.

### Strengths
1. Principled approach to grounding tool generation in external, authoritative sources. It is a well-motivated and logical solution for specialized domains.
2. The introduction of a hierarchical toolbox structure that mirrors the organization of the reference material is a strong design choice.
3. Demonstrated generalization.

### Weaknesses
1. The framework's performance depends on the quality and structure of the reference material. It assumes a well-organized, comprehensive textbook with content that can be easily split into discrete functions. How does the framework cope with noisy, theoretical, or poorly structured references?
2. The core components are incremental and miss the core baseline, such as fine-tuning the base LLM directly on textbook content. It is unclear whether the explicit, complex, and potentially brittle tool-creation pipeline is truly superior to simply teaching the model the domain knowledge through continued pre-training or fine-tuning.
3. Human evaluation of tool quality uses a small sample—just 20 tools per domain—making it hard to generalize across hundreds created. While REFTOOL outperforms baselines, the gains are often modest. For example, chemistry scores improve by only 2% over POT + RAG (Table 2), questioning the complexity of the pipeline. Also, Table 5's cost analysis omits the considerable manual and computational effort required for reference curation and initial tool generation.

### Questions
The hierarchical selection is a critical component. Could you provide metrics on the accuracy of the category and tool selection steps themselves, perhaps against an oracle or human judgment? What is the failure rate at these stages?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
REFTOOL introduces a reference-guided framework that enables large language models (LLMs) to automatically create and use tools grounded in external materials (e.g., textbooks, knowledge snippets) rather than relying solely on internal knowledge. This allows LLMs to tackle knowledge-intensive reasoning tasks across domains where pre-defined tools or sufficient prior knowledge are unavailable.
- LLMs generate executable code tools from reference materials (structured like textbooks or unstructured like text snippets). Each tool includes a natural-language description, Python implementation, and example usage. Tools are validated and refined automatically using example-based execution checks.
- Tools are organized into a hierarchical toolbox reflecting the reference’s knowledge structure. During inference, LLMs hierarchically select relevant categories and tools to solve input problems. Integrates with reasoning paradigms like Program-of-Thoughts (PoT) and ReAct, allowing graceful fallback when no tools apply.
- Benchmarked on causality (QRData), physics (TheoremQA), and chemistry (SciBench). Outperforms state-of-the-art tool-creation methods and domain-specific reasoning systems (e.g., Physics Reasoner, ChemAgent) by ~12–13% average accuracy across domains. Achieves dataset-agnostic generalization - tools built from one dataset transfer effectively to others in the same domain. Extends to non-scientific tasks, such as extremely low-resource language translation, yielding a +10 BLEU improvement.
- Reduces computational cost and inference time by up to 99% compared to domain-specific tool systems. Human evaluation shows ≥90% correctness and usefulness of generated tools.

### Strengths
- The paper introduces a new perspective on tool creation shifting from internally generated tools (as in Creator, TroVE, etc.) to reference-guided tool generation grounded in external textual materials. This is elegant and underexplored.
- The approach generalizes to multiple datasets within the domain and also to areas beyond scientific reasoning.
- The experiments are extensive, spanning multiple scientific domains and extending to non-scientific domains. The evaluation is robust and covers tool reuse across datasets, ablations, alternative base models for tool creation, cost analysis, and human evaluation.
- The improvements (~12–13%) are consistent and statistically significant.
- The paper is well-written and logically structured, has illustrative figures and examples.
- This work contributes a scalable and general paradigm for extending LLMs’ reasoning ability beyond internal knowledge via reference-grounded code tools. It has clear implications for scientific reasoning, educational AI, and self-improving agents.

### Weaknesses
- the validation of generated tools is based on available examples or generation of appropriate examples. It is unclear how one would ensure the correctness of the generated examples and their solution.
- the choice of GPT-4o for tool creation, the number of tools and categories is unclear - what is the basis for these choices?
- While causality, physics, and chemistry benchmarks are informative, they are narrow in scope and format (mostly numerical or formula-based). The framework’s generality would be more convincing if tested on applied or multi-modal reasoning tasks.
- Although the authors claim REFTOOL can extend LLM knowledge boundaries, all experiments use well-bounded domains with relatively small references. It remains unclear whether this approach scales when references are large, diverse, or ambiguous (e.g., thousands of documents).

### Questions
covered in the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a novel framework that enables large language models (LLMs) to create and use their own tools by leveraging external reference materials (e.g. textbooks, knowledge documents). The approach is motivated by the observation that many complex tasks (causal reasoning, physics, chemistry, etc.) require domain knowledge that may not be encoded in the LLM’s parameters. Unlike prior tool-using paradigms, which rely purely on the model’s internal knowledge or a fixed set of APIs, REFTOOL “grounds” the tool creation process in external references.

### Strengths
1.	Enabling LLMs to automatically create executable tools from unstructured text references is more efficient than traditional text retrieval. This structured generation approach (including pseudocode and code templates) is more effective than simple text retrieval. By transforming textbooks into a toolbox, REFTOOL empowers models to solve problems previously beyond their capabilities.
2.	REFTOOL achieves significant performance improvements on knowledge-intensive benchmarks like causal reasoning, physics, and chemistry, increasing accuracy by 12.3% compared to existing tool-generation methods.

### Weaknesses
1.	The method currently relies on GPT-4 (or similar large models) for tool generation and verification. While understandable (tool synthesis is difficult), this means the initial setup can be expensive and dependent on proprietary models. Without GPT-4 or similar high-performance models, the quality of the generated tools may decline. This introduces an unfairness in experimental baseline comparisons. The article does not deeply explore using open-source LLMs for tool creation; it would be worth knowing how a 70B open-source model performs on this task, even if some performance loss is expected.
2.	Although overall tool quality is high, the generated tools might be relatively unreliable in some highly complex domains (like chemistry). For instance, chemical reasoning may require extremely complex domain knowledge, and even with textbooks and GPT-4, generating completely correct code isn't guaranteed. The article notes that chemistry tools received relatively low scores in human evaluation. Therefore, for highly specialized or obscure knowledge, the method might require additional fine-tuning (perhaps incorporating symbolic systems or domain expert review).
3.	REFTOOL generates hundreds of tools per textbook (e.g., over 500 for physics). Storing and managing such a large toolbox could become overwhelming, especially when considering broader knowledge bases or multiple textbooks. Although hierarchical selection helps, it can become quite complex – for example, if a category still contains dozens of tools, providing all their descriptions to the LLM might exceed its context window limit.

### Questions
1.	The article's innovation seems extremely limited relative to existing work [1], essentially just pre-generating tools in a RAG-like manner. What do the authors believe is the innovative contribution of their research?
[1] Cai, T., Wang, X., Ma, T., Chen, X., & Zhou, D. Large Language Models as Tool Makers. In The Twelfth International Conference on Learning Representations.

### Soundness
3

### Presentation
2

### Contribution
2
