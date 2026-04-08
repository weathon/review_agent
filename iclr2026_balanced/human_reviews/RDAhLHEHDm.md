## Human Reviewer 1

### Summary
This paper tackles how Sci-LLMs handle biomolecular sequences. It argues current methods are stuck in a "tokenization dilemma": either they treat sequences as language, breaking up important motifs, or as a separate modality, which creates an alignment gap. The authors propose a "context-driven" approach, skipping raw sequences entirely. Instead, they use bioinformatics tools (BLAST, Pfam) to create a text summary for the LLM . Their experiments show this context-only method works best, and that adding the raw sequence back in actually hurts performance, acting like noise.

### Strengths
The paper's "tokenization dilemma" concept is a really clear and smart way to frame a major hurdle for Sci-LLMs. The main idea—that feeding LLMs text context from tools like BLAST is better than giving them the raw sequence—is surprising but backed up well by the experiments. The finding that raw sequences just add "noise" and make things worse is a big deal. The visualizations (like in Figure 3) showing how alignment fails are also very convincing . This work is important because it questions the push for end-to-end models and offers a practical, hybrid alternative.

### Weaknesses
The main drawback, which the authors rightly point out, is that this method can't handle mutation effect prediction. The bio-tools (BLAST, etc.) used to create the context just aren't sensitive to tiny, single-point changes, so the context for a normal protein and its mutant look the same . This is a major limitation, as it rules out a big area of computational biology. Also, the claims about it working on DNA are mostly tucked away in the appendix, not fully explored in the main paper.

### Questions
Given the issue with mutations, do you have ideas for how this context-driven method could be adapted for those tasks? Maybe by using different tools that are sensitive to mutations to generate the context?

You mention your method is efficient because it avoids retraining, but running tools like InterProScan and BLAST for every query isn't free. How does the real-world inference time/cost of your pipeline compare to running a big, end-to-end model?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a “paradigm shift” for how Scientific Large Language Models (Sci-LLMs) are trained, leveraging context-centric approaches driven by high-level structured knowledge from bioinformatics tools (e.g., GeneOntology, ProTrek, BLASTp, etc.).  The solution addresses two key tokenization “dilemmas” that have posed challenges on the Sci-LLM space: sequence-as-language and sequence-as-modality.  This approach accounts for multiple levels of language used to describe biomolecular phenomena – from human-encoded knowledge to genetics/evolutionary-encoded knowledge. Strikingly, the context-only approach largely outperforms joint context + raw sequences, suggesting that raw sequences contribute more to information noise.  The contribution suggests that Sci-LLMs don’t necessarily require solving complex biological “language” from scratch but can leverage decades of accumulated biological knowledge contained within structured databases.

### Strengths
1.	Overall: The paper and aims to address a novel challenge in the Sci-LLM space, making a case that Sci-LLMs are better served as “reasoning engines over expert knowledge”, rather than pure sequence decoders. While this is noted and there is some evidence that this is the case, it does raise some circular logic around the quality of the annotations derived from the bioinformatics knowledgebases (addressed below in the weaknesses).
2.	Generalizability: The solution in generalizable, with applications ranging from known proteins to “novel” proteins, as well as different biomolecular types.
3.	Practicality: The solution as it is described is practical, as it allows to more easily keep models up to date with new biological knowledge with lower development costs.  (Although it could be argued that most of the effort is derived from maintaining the bioinformatics knowledgebases).

### Weaknesses
1.	Circular Logic: The approach works well when high-quality annotations exist, yet the solution also exists to propose annotations to fill in knowledge gaps. This counter-intuitively raises a bit of a “Catch 22” scenario.
2.	Core Argument: The basis of the manuscript suggests that there is in fact valuable information encoded within the evolutionary language through sequence tokens, yet the results suggest the opposite – and that human context exclusively drives the value.

### Questions
1.	How do you address the circular reasoning between the strengths of the approach (incorporating high-quality expert annotations) and using this approach to predict those annotations where they do not yet exist?  Could tool-calling agents solve this rather than building directly into the LLM?  What are the tradeoffs?
2.	Along this line of questioning, does the core contribution put a focus on the LLMs, or are you simply demonstrating that tradition bioinformatics pipelines already solve most of the problems around understanding protein function?
3.	Have these results been validated against human expert annotators?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper argues that Scientific Large Language Models face a "tokenization dilemma," struggling to interpret raw biomolecular sequences, which are either broken down into meaningless components or difficult to align with natural language. Through systematic experiments, the authors demonstrate that a "context-only" approach, where models are given high-level, human-readable knowledge from bioinformatics tools (like BLAST or Pfam) , consistently and substantially outperforms models given the raw sequence.

### Strengths
Pros:
- The authors proposed a new “context-only” method, which achieved significantly 
- The context-driven approach achieve good performance.

### Weaknesses
Cons:
- Context-only approach sounds interesting. However, compared with raw biomolecular sequences input, an inevitable con of this approach would be significant information loss (by discarding too many detailed information).
- The capability of this approach is capped by the bioinformatics tools being used, e.g., InterProScan and BLAST.
- As the context-only model relies majority on prior, it may not be a good tool for exploring “novel” findings (which may be out of distribution a bit).
- Why in Table 1, QWEN series of models are not considered, while in Figure 2, for “ours” model, the author choose to use Qwen-embedding. What about the embedding visualization for specialized language models [1] like ESM series


[1] Zheng, Y., Koh, H. Y., Ju, J., Yang, M., May, L. T., Webb, G. I., ... & Church, G. (2025). Large language models for drug discovery and development. Patterns.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper identifies and investigates a fundamental challenge in Scientific Large Language Models (Sci-LLMs) for biomolecular understanding, which the authors term the "tokenization dilemma." They argue that existing paradigms—"sequence-as-language" (tokenizing sequences into atomic units) and "sequence-as-modality" (encoding sequences via specialized encoders)—suffer from weak representation and semantic misalignment, respectively. As a solution, the authors propose a "context-driven" paradigm, which bypasses raw sequence input. Instead, it leverages established bioinformatics tools (e.g., InterProScan, BLASTp) to generate high-level, human-readable textual context (e.g., functional domains, GO terms) that is natively aligned with the LLM's linguistic space.  The authors evaluated three input modes: sequence-only, context-only, and a combination of both. Through extensive empirical evaluation on protein QA, EC number prediction, and DNA mutation tasks, the authors demonstrate that the context-only approach consistently and substantially outperforms all other modes. They find that adding raw sequence information to context often degrades performance, acting as "informational noise."

### Strengths
- The paper clearly articulates the "tokenization dilemma" as a critical, yet overlooked, bottleneck in Sci-LLMs. The conceptual framing of the two existing paradigms and their respective weaknesses is compelling and well-supported by prior work.
- The central claim—that raw sequences can be detrimental when combined with high-level context—is counter-intuitive and strongly supported by systematic experiments across multiple models (Intern-S1, Evolla, NatureLM, GPT-4o, etc.) and tasks (protein function, pathway, localization, EC prediction). The consistent performance drop in "Sequence + Context" settings is a powerful result.
- The authors evaluate their method on a wide range of benchmarks, including their own reconstructed dataset, temporal splits, and sequence identity-based splits (Easy/Medium/Hard). The inclusion of DNA-based tasks also demonstrates generalizability beyond proteomics.
- The paper goes beyond mere performance comparisons. The layer-wise analysis of Evolla (Section 5.3, Appendix F) convincingly shows how semantic alignment (via Q-Former) erases fine-grained mutation signals, providing a mechanistic explanation for the limitations of the sequence-as-modality approach.

### Weaknesses
- The context-driven approach relies heavily on the quality and coverage of external tools (InterProScan, BLAST). While an ablation study is provided (Appendix E), it does not fully explore the performance ceiling—what happens when these tools fail completely on highly novel proteins? The method's performance is inherently tied to the underlying databases' completeness and timeliness.
- The paper equates "biomolecular understanding" primarily with high-level functional annotation (GO terms, pathways). It does not assess whether the model gains *mechanistic* or *structural* insights that might require raw sequence analysis (e.g., predicting the effect of a point mutation). The limitation section (Appendix J) correctly notes this but underscores a fundamental constraint of the proposed paradigm.
- The strong performance of general LLMs (Gemini, GPT) in the context-only setting raises questions about potential memorization of public protein annotations from their vast pre-training corpora. While the authors take care to prevent label leakage in their *context generation*, they do not explicitly audit whether the test proteins' annotations were already in the LLMs' training data.
- The primary metric (LLM-Score) relies on another LLM (DeepSeek-V3) to judge answer quality. While this is a reasonable approach for open-ended QA, it introduces potential biases and lacks the objectivity of exact-match metrics used in tasks like EC prediction.
- Code is not provided in the current submission, providing it would be helpful to make work reproducible.

### Questions
- Given the high performance of general-purpose LLMs like Deepseek-v3, Gemini2.5 Pro and GPT-5, what steps did you take to ensure that the ground-truth annotations for your test proteins were not present in these models' pre-training data? Could the results be partly explained by memorization rather than reasoning?
- Your approach depends on external tools. Can you provide a qualitative analysis or failure case study for proteins where InterProScan and BLASTp return no or incorrect hits? How does the performance of your method degrade in such "orphan" scenarios, and what are the potential remedies?
- The paper convincingly shows that context is superior for *retrieving* known functional annotations. However, do you believe your paradigm can be extended to tasks that require *discovering* novel functions or reasoning about structure-sequence relationships that are not yet captured in existing databases?
- You note your method is computationally efficient as it avoids Sci-LLM retraining. However, running InterProScan and BLASTp for every query in a real-time application could be costly and slow. Could you comment on the latency and scalability of the full context-generation pipeline compared to a single forward pass of a sequence-as-language and a sequence-as-modality model?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4