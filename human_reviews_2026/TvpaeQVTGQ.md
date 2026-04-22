# A Fast, Reliable, and Secure Programming Language for LLM Agents with Code Actions

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Modern large language models (LLMs) are often deployed as agents, calling external tools adaptively to solve tasks. Rather than directly calling tools, it can be more effective for LLMs to write code to perform the tool calls, enabling them to automatically generate complex control flow such as conditionals and loops. Such code actions are typically provided as Python code, since LLMs are quite proficient at it; however, Python may not be the ideal language due to limited built-in support for performance, security, and reliability. We propose a novel programming language for code actions, called QUASAR, which has several benefits: (1) automated parallelization to improve performance, (2) uncertainty quantification to improve reliability and mitigate hallucinations, and (3) security features enabling the user to validate actions. LLMs can write code in a subset of Python, which is automatically transpiled to QUASAR. We evaluate our approach on the ViperGPT and CaMeL agents, applied to the GQA visual question answering and AgentDojo AI assistant datasets, demonstrating that LLMs with QUASAR actions instead of Python actions retain strong performance, while reducing execution time by up to 56%, improving security by reducing user approvals by up to 53%, and improving reliability by applying conformal prediction to achieve a desired target coverage level.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper falls outside my area of expertise.  I'm unable to assess this paper.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

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
This paper introduces QUASAR, a novel programming language designed to improve the performance, security, and reliability of LLM-based agents. QUASAR achieves this by separating internal computations from external side effects, supporting parallel execution, and utilizing conformal semantics for uncertainty quantification. The language is implemented through a workflow where LLMs generate Python code, which is then transpiled to QUASAR. Experimental results show that QUASAR improves execution speed (up to 56% faster), reduces user interaction for security validation, and maintains high reliability with a target error rate of 0.1.

### Strengths
- The paper effectively identifies challenges in LLM-based agents which write Python code to invoke tool APIs, and presents a practical solution through QUASAR.
- The "internal computation - external side effects separation" architecture and the introduction of conformal semantics are novel and offer significant advantages in performance, security, and reliability.
- Experiments on real-world agents like ViperGPT and CaMeL, covering performance, security, and reliability, demonstrate the practical benefits of QUASAR.

### Weaknesses
- Lack of Detailed Technical Explanation: The paper lacks in-depth descriptions of key components like QUASAR’s rewrite rules, Python subset syntax, and transpiler implementation, which could impact reproducibility and understanding.
- Flexibility Concerns in Tool-Calling Scenarios: While QUASAR improves upon Python in certain areas, there is a concern about whether it can maintain the same flexibility as Python in all tool-calling scenarios. Python’s ecosystem is rich with libraries that facilitate diverse use cases (e.g., system administration tasks, network programming, data processing). It’s unclear if QUASAR can handle such diverse scenarios with the same ease and flexibility, particularly in more dynamic, real-time applications where Python’s built-in flexibility is often crucial. A clearer discussion on this aspect and how QUASAR addresses such scenarios, if at all, would be valuable.

### Questions
- What optimizations would you suggest for fine-tuning on small datasets like AgentDojo? Are techniques like transfer learning or data augmentation being considered to improve performance on such tasks?
- How does QUASAR address tasks that cannot be parallelized due to dependencies? Could you provide more insight into the characteristics of tasks that hinder parallel execution?
- Could QUASAR be considered more of a specialized Python interpreter rather than a new programming language? How does it differ from existing solutions such as parallel-execution Python interpreters (PyPy, Cython) or security frameworks (e.g., Sandboxed Python), which already provide performance improvements for Python code? What makes QUASAR’s approach more beneficial than these well-established, mature solutions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces QUASAR, a new programming language designed to make LLM-driven code execution faster, safer, and statistically reliable. It combines a pure functional core, explicit side-effect isolation, automatic parallelization, and conformal prediction–based uncertainty propagation. The work is ambitious and conceptually motivated, aiming to establish a formal, language-level foundation for trustworthy agent behavior. Broadening experiments and addressing realistic LLM integration would make it stronger in practice and more convincing.

### Strengths
- Designing an LLM-native programming language for code generation action is innovative and promising. 
- QUASAR introduces a pure functional core that separates computation from side effects.
This separation allows deterministic execution, simplifies formal reasoning, and makes program behavior easier to verify and audit.
- The runtime system can automatically detect independent external calls and execute them concurrently.
Experiments show up to 56% reduction in total execution time, demonstrating concrete performance gains compared to sequential baselines.
- QUASAR enforces strict external-call isolation through explicit user approval.
It introduces a batch-approval mechanism that reduces the number of user interactions by over 50%, balancing usability and safety while preventing unverified API calls.

### Weaknesses
- **Narrow evaluation scope:**
The experiments are confined to small, synthetic benchmarks (GQA and AgentDojo). These tasks are short and prestructured, which limits the external validity of the claims. There is no evaluation in complex or dynamic environments that real LLM agents operate in.
- **Limited language expressiveness:**
QUASAR only supports a very restricted subset of Python (functions, variables, simple control flow). It does not handle classes, exceptions, pattern matching, or early returns. This simplicity makes formal analysis easier but severely limits applicability to realistic agent workflows that depend on richer language features.
- **Unclear LLM integration strategy:**
The paper proposes transpiling from a restricted Python subset but does not explain how LLMs are constrained to generate only this subset. There is no discussion of prompting or fine-tuning when the model produces invalid constructs. This leaves a major usability gap between theory and practice.

### Questions
How are LLMs guided or constrained to produce valid Python subsets that can be reliably transpiled into QUASAR, and what is the success rate of this process in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces QUASAR, a novel programming language designed specifically for LLM agents that use code actions. Unlike Python—which is the standard medium for LLM-generated code—QUASAR provides built-in mechanisms for performance optimization (via automatic parallelization), security (via dynamic access control and user approval of external calls), and reliability (via conformal semantics for uncertainty quantification).

### Strengths
+ The rewrite-rule semantics and external call dispatch mechanism are rigorously formalized.

+ The ability to propagate model uncertainty at the program level is a novel contribution that could inspire future work on trustworthy agent execution.

+ The use of a Python subset and a transpiler ensures backward compatibility with current LLMs, addressing real-world deployability concerns (without performance degradation).

### Weaknesses
- The paper does not specify how QUASAR manages external call failures, exceptions, or thread-level errors. For example, what happens if an external API call fails, times out, or returns an invalid response? Is the failure propagated, retried, or absorbed?

- While QUASAR executes external calls “as soon as all their arguments are available,” it is not clear whether “futures” or deferred results are explicitly represented in the language. How does the interpreter manage dependencies among pending external calls or enforce order when results are reused?

### Questions
It wasn’t clear to me what the sentence “There is only one external rule Rext = {Rext}. This rule is designed to enable calls to external functions f ∈ Fext” means in practice. Does this imply that all side-effecting operations (e.g., API calls, LLM queries) are handled uniformly through this single rewrite rule? How does QUASAR distinguish between different external APIs at runtime?

Could QUASAR’s transpilation strategy be generalized to other host languages (e.g., typescript) or is it fundamentally tied to Python’s semantics?

### Soundness
4

### Presentation
3

### Contribution
3
