# QLCoder: A Query Synthesizer For Static Analysis of Security Vulnerabilities

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 8, 6

## Abstract
CodeQL is a powerful static analysis engine that represents programs’ abstract syntax trees as databases that can be queried to detect security vulnerabilities. While CodeQL supports expressive interprocedural dataflow queries, the coverage and precision of its existing security queries remain limited, and writing new queries is challenging even for experts. Automatically synthesizing CodeQL queries from known vulnerabilities (CVEs) can provide fine-grained vulnerability signatures, enabling both improved detection and systematic variant analysis. We present QLCoder, an agentic framework for synthesizing CodeQL queries from known CVE descriptions. QLCoder leverages the Model Context Protocol (MCP) for agentic tool use, integrates abstract syntax tree guidance, and incorporates CodeQL’s language infrastructure and documentation into the synthesis loop. A key challenge is that state-of-the-art large language models hallucinate deprecated CodeQL syntax due to limited training data and outdated knowledge. QLCoder addresses this by combining contextual engineering, iterative query feedback, and structured tool interaction to reliably generate executable, up-to-date queries.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present FineNib – an agentic framework that automatically synthesizes queries in CodeQL, a powerful static analysis engine,
directly from a given CVE metadata. FineNib embeds an LLM in a synthesis loop with execution feedback, while constraining its reasoning using a custom MCP interface that allows structured interaction with a Language Server Protocol (for syntax guidance) and a RAG database (for semantic retrieval of queries and documentation). This approach allows FineNib to generate syntactically and  emantically valid security queries. The authors evaluate FineNib on 176 existing CVEs across
111 Java projects. Building upon the Claude Code agent framework, FineNib
synthesizes correct queries that detect the CVE in the vulnerable but not in the
patched versions for 53.4% of CVEs. In comparison, using only Claude Code
synthesizes 10% correct queries. The generated queries achieve an F1 score of
0.7. In comparison, the general query suites in IRIS (a recent LLM-assisted static
analyzer) and CodeQL only achieve F1 scores of 0.048 and 0.073, highlighting
the benefit of FineNib’s specialized synthesized queries

### Strengths
- The topic of CodeQL query generation for vulnerability detection is both timely and technically demanding. It presents significantly greater challenges than code generation for mainstream programming languages.
- The development of an MCP interface demonstrates substantial engineering effort. This contribution is valuable and enables structured reasoning and provides a foundation for future research in automated vulnerability analysis.
- FineNib achieves notably higher compilation, success, and F1 scores compared to existing methods.

### Weaknesses
- The authors claim that "Permitting free online search for vulnerability patterns or snippets similarly proved problematic." It would be helpful to provide empirical evidence or examples supporting this observation.
- While it is understandable that the presence of both vulnerable and fixed code versions simplifies the synthesis process, the realism of this assumption is uncertain. In practice, software updates often contain numerous unrelated edits (e.g., functional updates) to the affected CVE. How robust is the proposed method to such irrelevant changes between the two versions?
- The evaluation section refers to “CodeQL (version 2.22.2) query suites” as a baseline. This description is somewhat vague—could the authors clarify what specific query suites or rule sets were used? Since CodeQL itself does not autonomously generate queries, it would be useful to understand how these suites were selected and executed for comparison, and what they represent in the context of automated query synthesis.
- A broader weakness of the work is that it appears predominantly engineering/application-oriented, with limited discussion of principled methodologies for code generation in low-resource or domain-specific languages such as CodeQL. A stronger methodological foundation could enhance the generalizability and scientific contribution of the framework.

### Questions
Please address my concerns in **Weaknesses**.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents FineNib, an agentic framework that synthesizes CodeQL queries directly from CVE text and patch signals. An LLM runs in a generate→execute→feedback loop and is constrained via a minimal MCP tool suite: (i) an LSP for syntax/API diagnostics and completion, and (ii) a RAG component to retrieve documentation, example queries, and small AST snippets. On CWE-Bench-Java (176 CVEs across 111 Java projects), the authors report a 53.4% success rate (fires on the vulnerable version and remains silent on the patched version) and strong F1, outperforming IRIS and off-the-shelf CodeQL query suites.

### Strengths
1) Translating CVE/patch information into executable static-analysis queries has clear value for regression testing, variant hunting, and patch verification in security engineering.
2) The constrained (MCP) tool set with LSP+RAG is a practical and novel combination that reduces hallucinated identifiers, version drift, and brittle syntax—common failure modes in code-query synthesis.
3) Treating “patched-version silence” as a first-class success criterion is well-motivated. The path-level metrics that intersect with patch deltas provide a principled notion of precision/recall beyond simple hit counts.

4) FineNib achieves substantial gains over IRIS and generic CodeQL suites on shared CVEs, and the reported overall success rate indicates meaningful practical headroom for automated coverage.

### Weaknesses
1) Limited external validity across languages and analysis engines; the study focuses on Java with CodeQL, leaving portability to Python/JS/C/C++ or to engines like Semgrep/Infer unclear.
2) Insufficient transparency on cost and throughput; the paper does not report detailed per-CVE runtime, tool-call/token breakdowns, or scaling with project size/CWE category.
3) Potential bias from metric design; scoring based on intersections with patch deltas may favor patch-aware synthesis and disadvantage baselines not modeling sanitizers or taint steps.

### Questions
1) Can you report a detailed cost profile per CVE (time cost, token usage), along with a breakdown by project size and CWE category?
2) What minimal changes are needed to port FineNib to Python/JS/C/C++? How would you replace or adapt LSP/RAG for other engines (e.g., Semgrep rules, Infer)?
3) Could you provide a controlled cross-agent evaluation where multiple LLMs use the same MCP interface and tool set, with end-to-end success rates and error categories?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
FineNib uses LLM's to synthesize CodeQL queries for a given CVE. Instead of doing it directly with an LLM, they integrate it into a loop using execution feedback to iteratively repair the CodeQL query that is generated. This execution feedback helps ensure that the query is correct syntactically and semantically. They show large improvements on CWE-Bench-Java compared to current state-of-the-art approaches, achieving an F1 score of .7.

### Strengths
I think there are some interesting ideas in this paper when it comes to neurosymbolic program analysis. I think that using execution feedback to guide the LLM to synthesize a query is a very good idea. There are many parts of program analysis that are tedious and difficult (writing CodeQL queries is a great example of this!). By incorporating an LLM into the loop and providing some feedback, I think we can make program analyses much more accessible and easier to write. I also liked the discussion section on alternative designs - it really helps highlight that more information is not always better when it comes to using these models. Also, performance is greatly improved using FineNib compared to other methods.

### Weaknesses
Unfortunately, where this paper falls apart for me is in the motivation. I don't understand why we need a query for one *specific* CVE, especially given the fact that that CVE has been patched. To my understanding, IRIS and other tools write queries that detect a general weakness pattern, like code injection (CWE-94). These queries can be used to detect that weakness pattern in other projects. But a CVE is just *one* instance of a CWE: "CVE-2024-12345: Buffer overflow in `foo()` of `libbar` 2.3.1 allows remote code execution".  I don't understand why we need a query for just that instance, *especially* given that it has been patched. It seems to me that the query would be useless from thereon out. 

If the authors could show how FineNib could be used to discover *new* CVE's, that would be a huge improvement to this paper. But from the current version, it seems that FineNib is just detecting vulnerabilities that have already been detected and patched.

Some other small nitpicks:
- Figure 1 implies that FineNib needs only the CVE information. But it uses the CVE information, the vulnerable code, and the patched code.
- The first sentence of the introduction (line 030) states that security vulnerabilities are growing, but doesn't state how many CVE's were reported in 2023, 2022, etc. 
- I'm a little confused at why IRIS has such a low F1 score. IRIS reports an F1 score of .177 in their paper, but it is .048 here. Specifically, the authors state that "The lack of true positive recall is why CodeQL and IRIS have significantly lower precision". Does that mean that IRIS still reports the vulnerability when it has been patched? I think this could be clearer.

### Questions
- Why do we need a CodeQL query for one specific vulnerability? 
- Along the same lines, what can that query be used for besides the vulnerability that has been patched?
- Can the authors elaborate more on why IRIS has such a low F1 score? Is it because IRIS still reports the vulnerability in the patched version?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces FineNib, an agentic framework that generates CodeQL queries from CVE metadata. To mitigate the pitfalls of pure LLM-based approaches, FineNib utilizes iterative reasoning with syntax guidance and a RAG database for semantic retrieval of queries and documentation. FineNib is evaluated on CWE-Bench-Java with Claude Sonnet 4, where 100% of generated queries complied, and 53.4% successfully detected the vulnerability. This is a much-improved result over the vanilla agentic baselines.

### Strengths
+ The application of script generation for vulnerability detection is a novel application, and with the help of RAC, the approach can mitigate the issues with low-resource static analysis query languages.
+ The approach shows great efficiency, outperforming vanilla LLMs and original CodeQL analysis by a big margin.

### Weaknesses
- The evaluation shows that FineNib can successfully generate good queries. However, the evaluation of a single LLM makes it somewhat incomplete. Claude Sonnet 4 is a very powerful, expensive, closed-source LLM; knowing the effectiveness of FineNib with a weaker open-source LLM would make the evaluation more complete.
- Since agentic approaches tend to be more costly due to their iterative refinement process, a cost vs effectiveness analysis could inform the reader of the cost per performance gain. Having an additional open-source LLM in the evaluation with this cost analysis can also paint a more complete picture of the overall efficiency of FineNib.
- Since the baseline agents with Claude Sonnet 4 are not presented in Table 4, it is hard to follow (it is kind of presented in the ablation study). Having Claude Sonnet 4 as the baseline agent would help highlight the effectiveness of the contribution more.

### Questions
1. It is not clear from the text if the performance was measured under a regression testing scenario or a variant analysis. It is important to know the performance separately between these two scenarios. What is the performance of FineNib in the variance analysis scenario (i.e., generate queries from one issue and detect other issues of the same type)?
2. Since the static analysis query is a low-resource language, would finetuning a smaller model specifically to generate the code for this language be a feasible approach? Would this potentially reduce the operational cost for a trade-off of a slightly larger cost upfront?
3. Could an additional open-source LLM be added for the evaluation to create a more complete evaluation picture?
4. Do you have some cost statistics that you can quickly perform a cost vs effectiveness analysis of FineNib?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents FineNib, a system that automatically synthesizes CodeQL vulnerability queries from CVE text and vulnerable/patched code. It uses an LLM agent with execution feedback and structured tool access (CodeQL LSP + RAG) to iteratively refine queries. On 176 Java CVEs, FineNib achieves a high rate of successful queries and good F1, substantially outperforming IRIS and built-in CodeQL rules.

### Strengths
1. Tackles a practical and timely challenge in security automation.
2. Clear design: LSP for syntax, RAG for semantics, feedback loop for correctness.
3. Strong results across real-world projects; large margin over baselines.
4. Good ablations showing importance of LSP + retrieval.

### Weaknesses
1. Only Java CVEs are supported; unclear how easily this extends to other CodeQL languages (C/C++, JS, Python). Authors mention RAG replacement, but actual cross-language results are not shown.
2. The system leans on Claude’s coding agent + CodeQL LSP. It is unclear how reproducible this is without Anthropic tooling or whether open-model support is viable (Gemini/GPT baselines fail).
3. Iterative refinement with code execution can be slow. Wall-clock time, computational budget, and iteration cost are not reported.
4. The metric assumes "raises alarm on vulnerable version and silent on patched version" as correctness. But some vulnerabilities have multiple exploitation paths; authors explicitly allow false positives, which may inflate success numbers.
5. The work is strong for static analysis automation, but the conceptual novelty in LLM reasoning or program analysis theory is incremental. It feels like an engineering system rather than theoretical innovation.

### Questions
1. What is the average runtime per CVE?
2. How sensitive is the system to CodeQL version changes?
3. Could the same iterative loop work with a weaker LLM if aided by stronger program-analysis constraints?
4. For multi-file or multi-snapshot vulnerabilities, how do you avoid overfitting to the specific patch change?
5. Can the pipeline detect variants of vulnerabilities (not only the exact CVE instance)?

### Soundness
3

### Presentation
4

### Contribution
3
