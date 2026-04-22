# AutoQG: An Automated Framework for Evidence-Traceable Question Generation via Ontology-Guided Knowledge Graph Construction

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Large Language Models (LLMs) present unprecedented opportunities for generating scientific questions. However, existing approaches face two key limitations: heavy reliance on costly human annotations and the production of brittle, unverifiable outputs. To address these challenges, we propose AutoQG, a fully automated multi-agent framework for evidence-grounded scientific QA generation. AutoQG comprises three complementary agents: (i) KG Extraction Agent, which performs ontology-guided knowledge graph construction with section-aware prompts for precise information retrieval; (ii) KG Evaluation Agent, a multi-dimensional evaluation module with iterative refinement to ensure accuracy and consistency; and (iii) QA Generation Agent, which produces schema-constrained QA pairs grounded in reasoning paths and explicit textual evidence. Applied to over 4,000 scientific papers, AutoQG constructs 243k triples and introduces AutoQG-20k, a benchmark containing more than 20,000 QA pairs. Each pair is explicitly linked to its reasoning chains and supporting evidence, ensuring transparency and verifiability. We further release AutoQG-7k, a challenging subset designed with hard questions that strong LLMs struggle to answer. Extensive experiments demonstrate that AutoQG consistently outperforms strong baselines in both human evaluation and LLM-as-a-Judge assessments. By transforming LLM output into a controlled and auditable pipeline, AutoQG advances evidence-based AI for the understanding of reliable scientific knowledge. Source code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AutoQG, a fully automated, multi-agent framework that generates evidence-traceable, scientific question-answer (QA) pairs from academic papers without human annotation. It addresses the key challenges of existing methods: costly manual annotation and the generation of unverifiable outputs by LLMs.

AutoQG's pipeline consists of three agents:

* KG Extraction Agent: Uses section-aware and ontology-guided prompts to build a structured KG from a paper's text.

* KG Evaluation Agent: Performs a multi-dimensional evaluation of the KG to ensure its accuracy and consistency, providing feedback in an iterative refinement loop.

* QA Generation Agent: Uses the validated KG to produce schema-constrained QA pairs that are explicitly grounded in the graph's reasoning paths and the original source evidence.

### Strengths
* The agents' tasks are simple but intelligently implemented, providing a useful, scalable framework to explore potentially for other non QA generation domains.

* The auditability (as the KG provides an "auditable backbone" for the generation process), traceability (it allows for the generation of QA pairs that are "explicitly traceable to the source text"), and the ability to create questions with multihop reasoning of the method.

### Weaknesses
* In Table 3, you compare against now quite outdated frontier models. How would the table look with newly released models like GPT-5?

* Given the multiagent setup, you do not address the increase in cost vs a standard LLM call.

### Questions
* While this appears to be a useful, unique QA generation method, it is often the follow-up questions (and follow-ups to those follow-ups etc) that are more akin to real-life science work. Could you extend your method to evaluate this in any way?

* Can you share the more granular details of what types of questions were judged highly by humans or not in Table 3 across all methods?

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
4

### Summary
AutoQG is a fully automated multi-agent framework for generating scientific question-answer (QA) pairs that are explicitly grounded in evidence from research papers. The system comprises three coordinated agents: (i) a Knowledge Graph (KG) Extraction Agent that uses section-aware prompts and an automatically induced ontology to convert a paper’s text into a structured knowledge graph; (ii) a KG Evaluation Agent that performs multi-dimensional quality checks (domain relevance, factual accuracy, consistency across sections, completeness, granularity) and iteratively feeds back corrections to improve the graph; and (iii) a QA Generation Agent that identifies multi-hop reasoning paths in the refined graph and produces schema-constrained multiple-choice QA pairs, each linked to its reasoning chain and supporting text segments. By transforming Large Language Model outputs into this controlled pipeline, AutoQG generates high-fidelity, evidence-traceable questions instead of brittle or unverifiable ones.

### Strengths
This paper presents a highly original and rigorous contribution to verifiable scientific QA generation through its AutoQG framework, which innovatively uses knowledge graphs as structured intermediates to ensure traceability and reduce hallucinations. The evaluation is exceptionally thorough, employing multi-modal assessment including human expert annotations, double-blind scoring, and LLM judges that demonstrate strong inter-rater reliability, with AutoQG consistently outperforming baselines across critical dimensions like relevance, reasoning depth, and evidence support. The paper excels in presentation quality with clear articulation of contributions, effective visualizations, and comprehensive contextualization within existing literature. Beyond its technical merits, the work offers significant practical impact by producing large-scale datasets (AutoQG20k/7k) that address a critical gap in multi-hop scientific QA benchmarking, with planned open-source release promising to catalyze future research in auditable knowledge generation for scientific education and literature analysis.

### Weaknesses
- Ontology Induction Details: The paper automatically induces an ontology but provides minimal details on its construction in the main text (relegated to Appendix A.4). This creates a reproducibility concern—the KG Extraction Agent's efficacy depends heavily on this ontology, yet readers lack insight into its contents, derivation method, size, or domain coverage. Did the authors use generic scholarly relation types or statistically induce them? The authors should summarize the ontology induction process in the main paper with key statistics (e.g., number of relation types) to help practitioners understand the system's limits and adapt it to new fields.
- Reliance on Proprietary LLMs and Compute Costs: AutoQG relies heavily on GPT-4 variants without discussing computational costs or API calls needed for 4,435 papers—likely substantial. This raises scalability and accessibility concerns: can others reproduce this without significant resources? Dependency on closed models also hinders long-term reproducibility. The authors should discuss optimization strategies, such as fine-tuning open-source LLMs for some tasks or using GPT-4 only for critical steps. The method's resource intensity is a limitation that needs acknowledgment and mitigation strategies.
- Generality and Domain Scope: AutoQG was demonstrated primarily on CS/STEM papers (arXiv, ACL). Its generalization to different domains (medical, social sciences) remains unclear, and the authors note "domain sensitivity" as a challenge. The ontology may not cover domain-specific relations, and different fields have varying writing conventions. While this isn't critical for a single paper, future work should address domain generality through adaptive prompts or automatic ontology expansion. A cross-domain performance experiment would strengthen claims about the approach's breadth.
- Evaluation Dependence on LLM Judges: Using multiple LLM judges is innovative, but introduces potential subjectivity and bias. Judges may share biases with GPT-4 (used for generation) or miss nuances humans would catch. Human evaluation covered only subsets (30 papers for KG), so most results rely on LLM judgments. The authors should clarify what "GPT-5" refers to, conduct larger human studies, and open-source evaluation prompts for independent validation. While the strategy is mostly convincing, more transparency would strengthen credibility.
- Potential Biases and Ethical Considerations: The paper briefly mentions bias propagation but doesn't explore it deeply. Biases in GPT-4's knowledge (field over-representation, stylistic preferences) could affect generated questions. A CS-heavy corpus might create ontologies biased toward that style. LLM reliance could also propagate factual errors if incorrect triples pass evaluation. The authors should acknowledge specific observed biases, provide error examples, and consider implementing bias checks or diverse judges to ensure fairness across subdomains.

### Questions
## 

1. Ontology Induction Procedure: Could you provide more details on how the ontology is automatically induced? It's unclear whether it was mined from the corpus, derived from external knowledge bases, or hand-designed. How many relation types does it include, and does it cover common scientific discourse relations (e.g., method-used, problem-stated, result-shown)?
2. Domain and Corpus Diversity: What is the scope of the corpus in terms of scientific fields? If the 4,435 papers are mostly in computer science, have you tried applying the system to other fields (biomedicine, physics, social sciences)? How does performance vary by domain?
3. Iteration and Convergence of KG Refinement: How many iterations of the extract→evaluate feedback loop were typically required for the knowledge graph to reach the quality threshold τ? Was one pass usually sufficient, or were multiple cycles needed? Was there a point of diminishing returns?
4. Multiple-Choice QA Format: How are the distractor options generated? Did you consider or experiment with *open-ended question answering* (without provided options)? Could AutoQG's approach easily extend to generating and evaluating free-form answers?
5. Use of AutoQG-Generated Data: Do you plan to use the AutoQG20k (and 7k) datasets to train or fine-tune QA models? Would a model fine-tuned on this evidence-traceable QA data perform better on scientific QA tasks than models fine-tuned on surface-level data?
6. Error Analysis and Future Improvements: Did you perform any qualitative error analysis on the generated triples or QA pairs? Understanding typical failure modes (e.g., trivial questions, incomplete reasoning chains, or erroneous triples) would be useful. What improvements would you prioritize to address these errors?

### Soundness
2

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
5

### Summary
The authors proposed AutoQG, a fully automated multi-agent framework for evidence-grounded scientific QA generation.

### Strengths
Well organized. The presentation is clear.

### Weaknesses
1. In KG Extraction Agent module, how did the authors convert a pdf document to text? can you process the context over two pages? Will you control the size of academic KG in LLMs? Will the author limit the type of nodes in Academic KG?
2. In QA Generation Agent module, the reasoning path is randomly sampled from the Academic KG or by some strategies? How could you make the path from a KG is meaningful? beyond the checklist in figure 2. 
3. Did the author compare the questions generated by KG and corresponded sentences? What is the difference between them?
4. Will the KG lose content information during the question generation?
5. Does the evidence-based refer to the multi-hop path extracted from the Academic KG?

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
