# TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
While integrating tools like Code Interpreter and Search has significantly enhanced Large Language Model (LLM) reasoning in models like ChatGPT Agent and Gemini-Pro, practical guidance on optimal tool use is lacking. The core challenge is effectively combining textual reasoning, coding, and search for diverse questions. In this paper, we propose Tool-Use Mixture (TUMIX), an ensemble framework that runs multiple agents in parallel, each employing distinct tool-use strategies and answer paths. Agents in TUMIX iteratively share and refine responses based on the question and previous answers. In experiments, TUMIX achieves significant gains over state-of-the-art tool-augmented and test-time scaling methods, delivering an average accuracy improvement of up to 3.55\% over the best baseline on Gemini-2.5-Pro and Gemini-2.5-Flash across key reasoning benchmarks, with near-equal inference costs. We find that agent diversity and quality are crucial and can be enhanced by using LLMs to auto-optimize agent designs. Furthermore, TUMIX can halt refinement upon reaching sufficient confidence, preserving performance at only 49\% of the inference cost. Further scaling can achieve higher performance, albeit at a greater cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TUMIX, an ensemble framework designed to solve the challenge of effectively combining LLM textual reasoning with Code Interpreter and Search tools. The framework operates by running multiple agents with diverse tool-use strategies in parallel, where agents iteratively refine their solutions by considering all outputs from the previous round. Experiments on Gemini-2.5-Pro and Flash show TUMIX significantly outperforms existing test-time scaling methods, achieving an average accuracy improvement of 3.55% over the best baseline.

### Strengths
1. The experiments show that TUMIX can beat several baselines across several highly challenging reasoning benchmarks (HLE, GPQA, AIME). 
2. By using an LLM-as-Judge to determine the optimal stopping round, the framework achieves nearly identical performance to a fixed-round policy but reduces the inference cost.
3. The paper provides a key insight that, unlike in text-only scaling, agent diversity is crucial for tool-augmented reasoning and outperforms repeated sampling from a single strong agent.

### Weaknesses
1. The main results are averaged over only three repetitive runs. On complex benchmarks like GPQA and AIME, the variance in LLM reasoning can be high. The reported gains, while positive, are sometimes small (e.g., 96.7% vs. 94.7% on AIME-Pro; 87.9% vs. 86.9% on GPQA-Pro). With only three runs, it is difficult to ascertain if these differences are statistically significant or fall within the margin of error, which undermines the robustness of the SOTA claims. Additionally, the results omit the actual inference count and token metrics
2. The paper's reliance on "inference counts"  as a primary cost proxy is problematic; a single inference from a simple CoT agent is not "comparable" in cost to a single inference from a complex, multi-step dual-tool agent that consumes far more tokens.
3. The paper notes that performance gains plateau after ~12 agent types (Figure 10) and attributes this to the increasing difficulty of answer selection. However, this analysis fails to consider that all 30 agents (human- and LLM-designed) are fundamentally limited to the same two tools (Code and Search). The performance plateau may not be due to the number of agents, but rather the lack of tool diversity. See more details in Question.

### Questions
1. Regarding Agent Comparison (Figure 5): In the 3-agent vs. 15-agent comparison, the 3-agent group used three "strong agents" ($C^{+}$, $CS_{gs}$, and $CSG_{gs}$), each sampling 5 times. Were these three agents empirically determined to be the top-3 best-performing agents from the pool of 15? 
2. The process for generating new agents (Section 5.3) is described as "query[ing] Gemini-2.5-Pro with current agent code examples". Could you provide more detail on this methodology? Specifically, what prompts were used to guide the LLM to generate strategies that "differ substantially from the original ones" (as seen in Table 14 ) rather than just minor prompt variations?
3. The framework was evaluated exclusively on Gemini-2.5-Pro and Gemini-2.5-Flash. Have you explored TUMIX's applicability to other models (e.g., GPT-5, Claude 4, or open-source models like DeepSeek and Qwen)? Furthermore, given the paper's emphasis on agent diversity, do you believe a heterogeneous mixture of models (i.e., agents powered by different LLMs) could be a viable strategy to further scale agent diversity and performance?
4. You correctly note that in later rounds, as answers converge, the choice of selection method (Majority Vote, LLM Selector, Random) becomes negligible. Does this imply that the primary benefit of the framework lies in the iterative refinement process itself rather than the final selection, and that for a highly-converged set of answers, a simple majority vote is sufficient?
5. You plot both inference number and token number against score in Figure 9. Given that token consumption is a more accurate measure of computational cost, and that different agents (e.g., CoT vs. Dual-Tool) have vastly different token profiles per "inference," would it be more transparent to frame the primary cost analysis around token count rather than inference count?

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
3

### Summary
This paper proposes TUMIX, a multi-agent framework for test-time scaling of tool-augmented LLMs. By orchestrating diverse specialized agents and enabling iterative response refinement, the system achieves an average gain of +3.55% on benchmarks including HLE and GPQA, using constrained inference budgets.

A key contribution of this work is the insight that agent diversity yields better results than homogeneous replication. Furthermore, the proposed adaptive termination mechanism reduces inference costs by approximately 49% while maintaining model accuracy. The experimental analysis on performance improvement in multi-agent systems also provides valuable reference for the research community.

### Strengths
- The TUMIX framework utilizes a coordinated multi-agent system in which specialized, tool-augmented agents perform parallel iterative refinement. Experiments report an average performance gain of 3.55% across several challenging benchmarks (HLE, GPQA, AIME 2024/2025), indicating the potential of this approach for test-time scaling.

- Through the introduction of a coverage rate metric and systematic ablation studies, the authors provide empirical support for the relationship between agent complementarity and overall task performance. This analytical methodology offers useful insights into the functioning of multi-agent ensembles.

- Empirical results consistently show that strategically diversifying agent composition leads to greater performance improvements than merely increasing the number of agents. This trend holds across various benchmark settings and model architectures.

- The proposed adaptive termination mechanism reduces computational costs by approximately 49%, while largely preserving model performance.

### Weaknesses
- Based on the experimental analysis presented in the paper, increasing the number of rounds generally leads to a decrease in agent coverage rate, suggesting reduced answer diversity and eventual performance convergence. However, in the GPQA experiment (Figure 5, Section 5.1), an increase in the number of rounds conversely results in a higher coverage rate. It would be interesting to understand the factors behind the contrasting behavior observed in the GPQA benchmark (Figure 5, Section 5.1).
- In Section 3.3, the authors note that many correct answers were incorrectly discarded by the LLM. Does this imply that the performance improvement is primarily influenced by the diversity of answers, as reflected by the coverage rate, while the use of the LLM to select responses may not contribute significantly to the performance gain?

### Questions
- Based on the experimental results in Figure 5, it appears that increasing the diversity of agent answers and mitigating the decline in coverage rate contribute to performance improvement. However, this observation seems to differ from the trends shown in Section 5.3 (Table 13). Some clarification regarding these differing outcomes would be helpful for a more comprehensive understanding.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TUMIX, a method run multiple tool strategies at test time (text, code, search, and hybrids) in parallel, share and revise answers across rounds, finish with majority voting; use an LLM-as-Judge for early stopping. Reports modest gains on HLE, GPQA-Diamond, and AIME with reduced cost under similar call budgets.

### Strengths
1. Clear problem setup: practical test-time tool mixing and scheduling.

2. Reusable engineering: parallel strategies + cross-round sharing + majority vote; pseudocode and prompt lists provided.

3. Useful mechanism analysis: coverage drops over rounds; early stopping cuts cost; includes cost–performance curves and several ablations.

### Weaknesses
1. Loss of ensemble independence: cross-round sharing homogenizes answers; later selection policies become near-equivalent; majority vote is no longer independent experts and is prone to correlated failure.

2. Tool–cost mismatch: cost is approximated by tokens or call counts only; wall-clock time, currency cost, search and code-execution delays and retries are not accounted for, so “equal-cost” comparisons are not valid and efficiency claims are unverifiable.

### Questions
1. Novelty vs. SciMaster: Could you clarify differences in architecture, judge/selector design, diversity control with the baseline SciMaster since I am still confused about the novelty?

2. Early-stopping robustness: The judge-score heuristic can prematurely stop on small benchmarks. Do you have a concrete method to prevent this? For example, risk-limiting stopping with error control, SPRT-style sequential testing, minimum rounds per task type, judge–solver model decoupling, and sensitivity curves for thresholds.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TUMIX (Tool-Use Mixture), a framework for test time scaling that integrates multiple LLM agents equipped with distinct tool use strategies such as textual reasoning, code execution, and search, to enhance reasoning capabilities. The framework operates by running 15 diverse agents in parallel, iteratively refining answers across multiple rounds and adaptively halting based on confidence estimation. The authors report consistent accuracy gains of up to 3.55% over strong baselines such as Self-MoA, Symbolic-MoE, and DEI on three reasoning benchmarks (HLE, GPQA, and AIME) using Gemini-2.5-Pro and Gemini-2.5-Flash. The work’s practical contribution lies in the combination of agentic diversity, tool integration, and cost effective refinement termination.

This work represents a thorough and thoughtful empirical study that advances understanding of tool augmented, multi-agent reasoning. It delivers consistent performance improvements and introduces practically relevant techniques such as adaptive termination and LLM driven agent generation. However, the methodological simplicity of the refinement mechanism, lack of theoretical or architectural novelty, and uncertain generalizability prevent it from reaching the standard expected at a premier venue. The contribution is valuable as a rigorous empirical report and could serve as a foundation for more principled frameworks combining structured agent communication with tool integration. Strengthening the theoretical underpinnings, expanding evaluation to multiple model families, and providing formal statistical validation would substantially improve the paper’s readiness for acceptance.

### Strengths
The paper is empirically strong and demonstrates commendable experimental rigor. It includes comprehensive ablations exploring agent diversity, tool combinations, and termination strategies, supported by clear figures and well documented appendices. The adaptive early termination strategy is an especially practical feature, halving inference costs with negligible performance loss. The empirical results establish that combining code and search tools within diverse agents produces measurable improvements across benchmarks. The use of LLMs to auto generate new agent types adds creative value, revealing potential for self improving agent systems. Clarity of presentation, transparency in reporting results, and a focus on practical reproducibility further enhance the paper’s quality.

### Weaknesses
The core methodological contribution is limited. The refinement mechanism relies on concatenating all prior responses into the prompt, a brute force aggregation that lacks principled structure or communication modeling. While empirically effective, it feels more like an engineering heuristic than a conceptual advance. The paper also depends heavily on proprietary, high capacity models (Gemini-2.5), raising concerns about reproducibility and generalization to open models. Claims about agent diversity remain loosely defined and most agents differ only in prompt phrasing or API configurations rather than fundamentally distinct reasoning styles. Furthermore, the evaluation lacks formal statistical significance tests and theoretical grounding for why diversity improves performance. The formulation in Section 3.1 sets up an optimization problem but doesn’t solve or analyze it, leaving the work largely descriptive. Finally, the reported improvements, though consistent, are modest relative to the significant computational and implementation cost.

### Questions
Could the authors clarify why concatenating all agent outputs yields such robust gains? Have they explored more structured synthesis mechanisms (e.g., critique based debate or weighted aggregation)?

How sensitive is TUMIX to the underlying LLM’s capacity? Would smaller or open source models (Llama, Mistral) still benefit meaningfully from this framework?

Can the authors provide confidence intervals or significance tests to confirm that improvements on smaller benchmarks (GPQA, AIME) are statistically valid?

How was the two round minimum for the LLM termination judge determined? Could a dynamic rule adaptively estimate the optimal refinement depth?

Regarding the LLM generated agents, what specific prompting or selection process was used? Do these agents exhibit measurable differences in reasoning strategy or simply better prompt quality?

Have the authors evaluated the method on open ended or real world tasks beyond academic benchmarks where consensus or correctness may be ambiguous?

### Soundness
2

### Presentation
3

### Contribution
2
