# Ca$^2$P: Cache-Augmented Code-as-Policies for Open-Domain Embodied Tasks

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 8, 4, 6

## Abstract
Embodied agents deployed in open-domain environments must continuously handle unpredictable tasks beyond predefined action policies. Such tasks are often given as natural language instructions, and recent progress in code-writing large language models (CodeLLMs) has inspired the Code-as-Policies (CaP) paradigm, where instructions are translated into executable control code when issued. However, generating full code from scratch for each instruction incurs high latency and inconsistency, limiting CaP's practicality in real-world, time-sensitive scenarios. To address these limitations, we present Ca$^2$P, a Cache-Augmented Code-as-Policies framework that improves CodeLLM-based robotic programming by introducing function-level key-value (KV) caching, a repurposed and extended form of the native KV caching mechanism tailored for function reuse, together with cache-augmented code policy synthesis. Ca$^2$P decomposes previously generated and validated code policies and stores them as function-level KV caches, supporting efficient compositional programming, where new policies are synthesized by invoking cached functions directly through their KV states. Furthermore, by revisiting and editing cached functions within their KV states, Ca$^2$P provides cache-refactoring, thereby enabling efficient synthesis of task-specific code policies without the need for full regeneration. Evaluated on ALFRED, TEACh, and RLBench benchmarks together with real-world robot manipulation, Ca$^2$P achieves the best trade-off between robustness and latency, with $19.80\%$ higher task success rate and $2.91\times$ faster policy synthesis than the CaP baseline.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a caching mechanism for the code as policies base. At a high-level, a coding LLM is used to write composable functions for executing manipulation tasks given a set of perception and control APIs (the idea previously in code as policies). In this work, the authors propose to augment it with an explicit cache across domains, which store separately the function description and the function implementation. At deployment time, functions are accumulated from successful runs and reused later in other tasks by referencing the stored function description.

### Strengths
- The paper is written with clarity in general and easy to follow.
- The empirical results are extensive across several evaluation domains.

### Weaknesses
- The contribution of this work appears to solely focus on the cache management for code generation (though for applications in embodied tasks), which has been very well understood and explored in both research and commercial systems such as coding agents. As a result, it is questionable whether the contribution is meaningful enough for ICLR community. Importantly, the major benefit in the context of embodied tasks appears to be mainly efficiency gain for LLM calls (which typically happens before any robot execution). Although it may be argued that this is important for certain tasks (such as that in figure 1), the scope remains limited and there are many ways to improve efficiency that has been widely deployed in existing tool boxes (such as token caching, model quantization).
- The open-sourced implementation of code as policies already contains a caching mechanism, albeit within the per-episode execution. It would enhance the paper if it can made further clearer the differences in the proposed caching mechanism to the existing one. Notably, what advantages does it offer?

### Questions
See weaknesses section above.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles a practical issue with Code-as-Policies (CaP) for robot control—every new instruction forces the model to rewrite all the code, making it slow and sometimes inconsistent. The proposed approach, CA2P, reuses previously verified functions by caching key–value (KV) states and updating code in place instead of regenerating everything from scratch. It builds a function-level caching system that supports both code reuse and quick local edits. Tests on ALFRED, TEACh, and RLBench show higher success rates and up to 2.9× faster responses than standard CaP methods. Real-robot trials confirm the improvements in both speed and stability.

### Strengths
- Clear and well-structured technical design (two-tier cache: Function-Interface and Function-Code) with coherent equations and pseudocode.

- Writing is clear and figures are informative

- Extensive experimental setup across multiple environments, baselines, and metrics.

### Weaknesses
- In Algorithm 1, the try–except block appears tailored for simulation, where errors can be easily caught. It remains unclear how such failures would be handled in real-world deployments.

### Questions
N/A

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
This paper aims to address the high latency and inconsistency of the Code-as-Policies (CaP) paradigm, where large language models (LLMs) generate full control code from scratch for every embodied task. This paper presents $CA^{2}P$, a Cache-Augmented Code-as-Policies framework for embodied AI agents in open-domain environments. The core innovation is function-level key-value (KV) caching that repurposes native transformer attention caching to enable code reuse. The system maintains a two-tier cache (Function-Interface and Function-Code) indexed by function identifiers, supporting compositional programming (assembling new policies from cached functions) and cache-refactoring (editing cached functions via fill-in-the-middle). A cache management scheme based on recency, frequency, co-occurrence, and semantic diversity scores determines retention. Experiments on ALFRED, TEACh, and RLBench 8benchmarks, as well as real-world robot manipulation, show that $CA^{2}P$ achieves a superior trade-off between task success rate (SR) and policy synthesis latency (PSL), outperforming CaP baselines.

### Strengths
- The two-tier cache design (Function-Interface I and Function-Code C) is well-motivated for separating lightweight references from full implementations, enabling efficient compositional programming without redundant attention computation

- Evaluation spans three simulation benchmarks (ALFRED, TEACh, RLBench) with different task characteristics plus real-world manipulation. Thorough baseline comparisons and ablation studies are conducted

### Weaknesses
- The locality score $l(f_k)$ in Equation (3) is central to the method, but its components are defined only at a high level. $l_{freq}$ ("usage frequency") and $l_{asso}$ ("conditional association") are not given precise mathematical definitions, making it difficult to reimplement the scoring function exactly. The weights $\alpha, \beta, \gamma$ in Equation (3) are set to 0.4, 0.3, and 0.3, but this choice is presented without justification. No ablation or sensitivity analysis is provided.

- The "code cache warm-up" analysis (Fig 4) explicitly states it starts from an "empty cache". However, Appendix D.1 states: "All cache-based baselines and $CA^{2}P$ begin with the same initial KV cache states derived from basic success code policies" for the main benchmark results. This implies the main results in Table 1 use a pre-populated cache, not one warmed-up from empty. This is a crucial distinction that is not clarified in the main paper.

- The "open-domain" framing is somewhat overstated—all benchmarks provide predefined API sets (Tables 6-9) with fixed primitives; true open-domain deployment would require handling novel APIs or learning from demonstrations, which is not addressed.

### Questions
- For the real-world experiments (Table 2), the paper states the cache is built by "first solves simpler tasks in RLBench". Is the performance therefore dependent on this pre-population step?

- In Algorithm 1, when is generate() versus edit() called? How are exceptions E detected and categorized? What triggers cache updates beyond task success?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors propose to utilize caching in order to improve the latency of code as policies framework for LLM based robotic control. Authors perform experiments with real world robot.

### Strengths
- Good empirical performance.
- Clear contribution as identified knowledge gap in code as policies framework. 
- Well engineered solution.

### Weaknesses
- Related work does not have any citations. Please rewrite, the purpose of related work is to cite previous works. 
- In respect to these hyperparameters, such as  $\alpha$,  $\beta$, $\gamma$ and others, I did not see experiments where those values were varied. This is strange considering the amount of results in presented in the paper. Maybe authors can explain this?

### Questions
- In line 203, why $\alpha$,  $\beta$, $\gamma$ needs to sum to one?

### Soundness
4

### Presentation
3

### Contribution
3
