# RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Code agents have gained widespread adoption due to their strong code generation capabilities and integration with code interpreters, enabling dynamic execution, debugging, and interactive programming capabilities. While these advancements have streamlined complex workflows, they have also introduced critical safety and security risks. Current static safety benchmarks and red-teaming tools are inadequate for identifying emerging real-world risky scenarios, as they fail to cover certain boundary conditions, such as the combined effects of different jailbreak tools.
In this work, we propose RedCodeAgent, the first automated red-teaming agent designed to systematically uncover vulnerabilities in diverse code agents. 
With an adaptive memory module, RedCodeAgent can leverage existing jailbreak knowledge, dynamically select the most effective red-teaming tools and tool combinations in a tailored
toolbox for a given input query, thus identifying vulnerabilities that might otherwise be overlooked.
For reliable evaluation, we develop simulated sandbox environments to additionally evaluate the execution results of code agents, mitigating potential biases of LLM-based judges that only rely on static code.
Through extensive evaluations across multiple state-of-the-art code agents, diverse risky scenarios, and various programming languages, RedCodeAgent consistently outperforms existing red-teaming methods, achieving higher attack success rates and lower rejection rates with high efficiency. We further validate RedCodeAgent on real-world code assistants, e.g., Cursor and Codeium, exposing previously unidentified security risks. By automating and optimizing red-teaming processes, RedCodeAgent enables scalable, adaptive, and effective safety assessments of code agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RedCodeAgent, an automated red-teaming agent designed to identify security vulnerabilities in LLM-based code agents. The system comprises three key components: (1) an adaptive memory module that stores and retrieves successful attack experiences, (2) a toolbox integrating both general jailbreak methods (GCG, AmpleGCG, AdvPrompter, AutoDAN) and a specialized code substitution tool, and (3) simulated sandbox environments for unbiased evaluation. Through extensive experiments across multiple code agents (OpenCodeInterpreter, ReAct, MetaGPT, Cursor, Codeium), benchmarks (RedCode-Exec, RedCode-Gen, RMCbench), and programming languages, the authors demonstrate that RedCodeAgent achieves higher attack success rates and lower rejection rates compared to existing jailbreak methods.

### Strengths
1. Red-teaming code agents is a critical but understudied area. As code agents become more widely deployed with execution capabilities, systematic security evaluation is essential. The motivation is well-articulated.
2. The integration of memory retrieval, dynamic tool selection, and execution-based evaluation is thoughtful. The memory module with trajectory logging and similarity-based retrieval (Algorithm 1) is elegant and effective.
3. RedCodeAgent consistently outperforms baselines, achieving 72.47% ASR vs 55.46% for no jailbreak on OCI, while maintaining efficiency (121.17s vs comparable baseline costs).
4. Validation on real-world tools (Cursor, Codeium) and discovery of 82 unique vulnerabilities that all baselines missed demonstrates real-world impact.

### Weaknesses
1. The paper relies entirely on automated evaluation methods without any human validation. This raises concerns about evaluation validity, as even spot-checking a subset of results with human annotators would significantly strengthen the validity of the findings. And the evaluation approach is particularly weak for RMCbench, where keyword-matching is used to detect rejections, which could easily miss sophisticated or nuanced refusals that do not contain the predefined rejection keywords.
2. Section D.4 shows memory helps, but provides minimal insight into what the agent learns. What patterns emerge in successful attacks? What makes certain tool combinations effective? The memory structure includes "self-reflection" but no analysis of its quality or utility
3. Nearly all experiments use GPT-4o-mini, only one ablation with GPT-4o (Section D.9). No exploration of open-source base LLMs (e.g., Llama, Mistral), which limits generalizability claims.

### Questions
1. How does RedCodeAgent perform on risk scenarios completely absent from the memory? The current setup accumulates memory during sequential execution.
2. What is your process for disclosing vulnerabilities to commercial vendors? Have Cursor and Codeium been notified?

### Soundness
2

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
3

### Summary
The paper proposes RedCodeAgent, an automated, adaptive red-teaming agent for LLM-based code agents. It combines (i) a memory module that stores successful attack trajectories and retrieves top-K similar experiences with a length penalty, (ii) a toolbox integrating multiple jailbreak optimizers and a code-substitution module to refine prompts via function-calling, and (iii) a simulation-based evaluation with Docker sandboxes for verifying code execution outcomes.
- The study evaluates RedCodeAgent on multiple targets: OpenCodeInterpreter (OCI), a ReAct-based agent, MetaGPT, and real-world assistants (Cursor, Codeium). Benchmarks include RedCode-Exec (27 risk scenarios across 8 categories), RedCode-Gen (malware-style function docstrings), and RMCBench. Metrics: Attack Success Rate (ASR), Rejection Rate (RR), and time/efficiency.
- Results: RedCodeAgent achieves higher ASR and lower RR than jailbreaking baselines (GCG, AmpleGCG, Advprompter, AutoDAN) on OCI/RA and across benchmarks. It also shows effectiveness across languages (Python/C/C++/Java) and on Cursor/Codeium. Ablations suggest retries alone do not close the gap; memory and multi-tool orchestration matter; the agent uncovers vulnerabilities other methods miss (e.g., reverse shell).

### Strengths
- Originality:
  - Integrates memory-guided retrieval with a length penalty to favor efficient prior trajectories: $S = \mathrm{CosSim}(e_{\mathrm{risk}}^q, e_{\mathrm{risk}}^m) + \mathrm{CosSim}(e_{\mathrm{desc}}^q, e_{\mathrm{desc}}^m) - \rho \cdot \mathrm{len}(m)$.
  - Systematic orchestration of jailbreak and code-specific tools via function-calling; includes code-substitution tailored to code-agent risks.
  - Execution-grounded evaluation in Docker, moving beyond LLM-as-judge for code tasks.
- Quality:
  - Broad and careful evaluation (multiple agents, benchmarks, languages; real-world assistants). Clear metrics (ASR, RR, time) and ablations (effect of retries; number of tools; memory modes; $\rho$).
  - Demonstrates discovery of previously missed vulnerabilities and improved efficiency with memory/tooling.
- Clarity:
  - Clear pipeline: retrieval → tool-driven prompt optimization → query → sandbox evaluation → reflection/memory update.
  - Tables summarize comparative performance; design choices (e.g., top-K, $\rho$) are stated, with ablations indicating robustness.
- Significance:
  - Addresses a pressing problem (code-agent safety) with a scalable, automatable methodology. Real-world assistant results underline practical risk and relevance.

### Weaknesses
- Fairness of baselines:
  - Comparisons pit an iterative, memory-augmented agent against baselines mostly evaluated as single-shot optimized prompts. The “retry” study covers two subtasks; a comprehensive best-of-N or multi-round baseline across all scenarios—budget-matched by iterations/API calls/time—would strengthen claims.
- Evaluation biases and coverage:
  - RedCode-Gen relies on LLM-as-judge; despite reasonableness, potential bias remains. Consider cross-checking with lightweight execution proxies where feasible or multi-judge consensus.
  - While RedCode-Exec covers 27 scenarios, important classes (e.g., SQL injection) are acknowledged as missing. Expanding or reporting generalization to additional realistic risks would improve coverage.
- Reproducibility and cost:
  - Core backbone model (GPT-4o-mini) and real-world assistant interfaces are not fully open/API-stable; semi-automated pipelines may hinder replication. Detailed reporting of token/compute budgets per method and per scenario would clarify cost-effectiveness beyond wall-clock time.
- Safety-of-release considerations:
  - The agent surfaces workable exploit prompts and reverse-shell procedures. While sandboxing mitigates local risk, the paper should articulate a more concrete responsible-release plan for tools, memory logs, and prompts (redactions, access controls).

### Questions
- Budget-matched baselines:
  - Can you provide a comprehensive, budget-matched comparison where baselines are allowed the same number of optimization/agent-query rounds as RedCodeAgent across all scenarios? If total iteration parity is hard, report best-of-N (N comparable to your median trajectory length) to bound the gap.
- Cost accounting:
  - Please include token counts and API costs per method (aggregate and per-risk) in addition to time and trajectory length, to assess sample and cost efficiency.
- Memory influence:
  - In the main results, to what extent do memory entries span across risk indices? You mention “Independent” mode for main tables; can you confirm there is no cross-index leakage? Also, how sensitive are outcomes to $\rho$ and top-K beyond the reported settings?
- Real-world assistants:
  - Were interactions compliant with the platforms’ terms of service? Could you release the semi-automated scripts and detailed instructions to reproduce Table 4 results (with redactions if needed)?
- LLM-as-judge reliability:
  - For RedCode-Gen, did you perform any human verification on a stratified sample to estimate judge accuracy? If so, please report agreement rates and common failure modes.
- Safety release plan:
  - What is the exact policy for releasing prompts/memories/tools to avoid enabling misuse? Will you gate high-risk artifacts (e.g., reverse shell prompts) for registered researchers?

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
5

### Summary
This paper proposes RedCodeAgent, an automated red-teaming framework for LLM-based code agents. The system integrates an adaptive memory module, a toolbox of existing jailbreak methods, and an evaluation module that uses sandbox execution to verify whether the generated code truly performs risky operations. Through extensive experiments on multiple benchmarks, the paper shows that RedCodeAgent achieves higher attack success rate and lower rejection rate compared to these baseline jailbreak methods.

### Strengths
1. The experiments cover multiple agents, programming languages, and benchmarks, offering a broad empirical view of code-agent vulnerabilities.
2. The sandbox-based evaluation and execution-level validation go beyond prior “LLM-as-a-judge” settings, improving measurement reliability.

### Weaknesses
1. The design of RedCodeAgent demonstrates limited novelty.
2. The paper does not include recent state-of-the-art jailbreak methods.

### Questions
The paper is well-written and includes comprehensive experiments. However, the design of RedCodeAgent shows limited novelty, as it mainly combines well-known components such as a memory module, jailbreak tools, and sandbox-based testing. The contribution lies in integrating and empirically evaluating these elements within a single framework for code agents.

The selected baselines (GCG, AutoDAN, AdvPrompter, and AmpleGCG) are relatively basic and outdated, while more recent and stronger methods are not considered. If existing jailbreak techniques already perform well, what is the necessity of developing RedCodeAgent?

Additionally, it remains unclear how gradient-based methods like GCG are applied to test a black-box code agent, since gradients are typically inaccessible in such settings.

Finally, the paper focuses solely on attack performance and does not evaluate or discuss potential defensive measures, which would be essential for providing a balanced view of code agent safety.

### Soundness
3

### Presentation
3

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
A framework called RedCodeAgent is proposed to automatically red-team LLM-based code agents and find security flaws. It has three parts: a memory module that retrieves past successful attacks, a toolbox of jailbreak and code-specific tools to craft prompts, and an evaluation module that runs generated code in a sandbox to check whether the risky action actually occurs. The agent iteratively optimizes prompts, queries the target agent, and stores successful trajectories for future attacks. Experiments show higher attack success rates and lower rejection rates than prior jailbreak methods in various benchmarks, languages, and real-world code assistants.

### Strengths
1. The model proposed can continuously learn from past attacks, which makes it more scalable and practical than static benchmarks.

2. The model proposed does not require additional knowledge other then API calls to the code agent.

3. The experiments are done using various benchmarks, programming languages, and real-world code agents.

### Weaknesses
1. The discussion and evaluations seems to be based on a pre-existing memory; It would be great to see how performance degrades when some tools are removed or when memory is empty.

2. The paper’s memory and embedding setup relies on general-purpose sentence embeddings and natural-language similarity, which weren’t specifically designed for attack semantics. That means the system may retrieve examples that are superficially similar but not actually useful for crafting successful exploits, reducing effectiveness on nuanced or codespecific vulnerabilities. It could be better if the authors adapt embeddings and memory structure for attack-relevance (e.g., code-aware or action-aware representations) to ensure the memory truly helps the inference process.

3. Most of the tools in the toolbox are pre-existing methods taken from prior work. Also, the tools appears to be largely hand-picked rather than systematically derived. The paper provided little justification for why these particular tools were chosen or how they complement each other.

4. The evaluation primarily focuses on attack success rate and rejection rate, with limited analysis of real-world impact or severity of the discovered vulnerabilities.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
