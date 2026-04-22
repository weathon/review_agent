# LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Materials discovery requires navigating vast chemical and structural spaces while satisfying multiple, often conflicting, objectives. We present LLM-guided Evolution for MAterials discovery (**LLEMA**), a unified framework that couples the scientific knowledge embedded in large language models with chemistry-informed evolutionary rules and memory-based refinement. At each iteration, an LLM proposes crystallographically specified candidates under explicit property constraints; a surrogate-augmented oracle estimates physicochemical properties; and a multi-objective scorer updates success/failure memories to guide subsequent generations. Evaluated on **14 realistic tasks** that span electronics, energy, coatings, optics, and aerospace, LLEMA discovers candidates that are chemically plausible, thermodynamically stable, and property-aligned, achieving higher hit rates and improved Pareto front quality relative to generative and LLM-only baselines. Ablation studies confirm the importance of rule-guided generation, memory-based refinement, and surrogate prediction. By enforcing synthesizability and multi-objective trade-offs, LLEMA provides a principled approach to accelerating practical materials discovery. Project website: https://scientific-discovery.github.io/llema-project/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduce a unified framework called LLEMA to intergrates LLM-knowledge and domain knowledge to generate high-quality sample through iterative search and refinement. The authors test the framwork on 10 diverse tasks and achieve significant performance gain on some of the tasks.

### Strengths
1. The framework provides good insight of combining LLM as search engine and modules that provides domain knowledge feedback.
2. Significant improvements are made on some tasks.

### Weaknesses
1. The originality of the method is somewhat limited as its high-level framework is similar to LLMatDesign except for some module design like stage b and multi-objective optimization. For example, the implementation of surrogate predictor comes from previous work but provides major improvements according to figure 7.
2. The experimental results are somewhat suspicious. LLEMA achieves very good results on tasks which LLMatDesign has nearly trivial values and has only comparable and slightly worse results on the tasks which LLMatDesign has non-trivial values. Although it's unreasonable to require SOTA result on every task, but more analysis should be made to explain this observation. For example, the author could demonstrate that the performance gain from trivial values to large values is not caused by some invalid setup of baseline methods but the design of some modules through ablation study.
3. Different prompts are used to in candidate generation stage. It is neccessary to clarify how prompt engineering affects the results to illustrates the robustness of the method.

### Questions
Please address my concerns in the weakness part. Also please check the accessibility of the code link in the abstract. I'm willing to raise the score if they're properly addressed.

### Soundness
2

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
This paper presents LLEMA, which employs chemistry-informed evolutionary rules and memory-based refinement to guide large language models (LLM) in multi-objective materials discovery. Its performance is evaluated on 10 different tasks, showing higher hit rates and chemical plausibility (characterized by thermodynamic stability) than baselines including generative models and vanilla LLMs. The importance of components of the proposed method is assessed via ablation tests.

### Strengths
- The proposed method combines LLM’s advantage of utilizing unstructured data sources with principled chemical knowledge guidance.
- The ablation studies are comprehensive, in particular, the analysis of memorization vs guided exploration investigates a key question that LLM4Sci should answer.

### Weaknesses
- The application scenarios are oversimplifications of materials design. For example, thermodynamic stability is the minimum requirement of chemical plausibility.
- Some common methods for constrained multi-objective optimization are not included in the benchmark tests (see Q1).

### Questions
## Technical 
1. Would statistically more principled design optimization methods, such as Bayesian optimization, work for the problem setting? How would they compare to the proposed method and other LLM-based or generative methods?
2. In Table 2, the hit rate of generative models such as CDVAE are surprisingly low. What might be the reason? Is the tested task too different from what the methods are designed for?
3. Following up on Q2, more recent generative models, such as MatterGen, are not covered in the benchmark. Would they perform better than the old ones?

## Clarity
4. In Sec. 2.4, how are the islands determined? This should be briefly explained in the main text. Besides, how much would the method choice or randomness in island partitioning affect the method’s performance?
5. Minor issues
    - in Line 33, “materials discovery remains a formidable challenge…” is an overstatement.
    - Line 183, what format does the LLM output, CIF or JSON?

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
The paper proposes an LLM-guided evolutionary framework for generating candidate materials under explicit structural and property constraints. In each iteration, an LLM is prompted to produce crystallographically specified or composition-level candidates; chemistry-informed evolutionary rules and a success/failure memory then filter and refine these candidates; finally, a surrogate (learned) oracle estimates multiple target properties and a multi-objective scorer selects promising samples to feed back into the next round.

### Strengths
1. The proposed method puts the LLM inside an iterative optimization loop (LLM $\rightarrow$ screen $\rightarrow$ refine $\rightarrow$ re-prompt). That aligns with current best practices in LLM-for-science/LLM-as-optimizer work and makes the contribution intelligible from an ML perspective.

2. The framework explicitly tries to stay inside chemically plausible regions via rule-based evolution and feasibility checks, rather than treating materials generation as free-text generation. That’s an important realism step for this domain.

3. The paper shows improvements on several tasks and on Pareto fronts (not just single-property top-1), making the method look more generally useful.

### Weaknesses
1. Several recent works also use pretrained LLMs to propose materials/structures and then improve them using an external scorer (e.g., [1]). The paper needs to spell out what is actually new here. Right now, the novelty can look like “a solid engineering combination” rather than a clearly new algorithmic design.

2. The paper claims to “leverage scientific knowledge embedded in LLMs,” but it does not show: (1) performance when replacing the LLM with a lighter template-based (or rule-based) generator; (2) performance across LLM scales or domain specialization; or (3) how often the LLM actually proposes candidates that could not be reached by rules alone. Without that, it’s hard to justify the LLM cost.

3. Surrogate reliability in out-of-distribution regions is not fully addressed. Evolutionary loops guided by learned oracles can be misled by over-optimistic predictions. The paper mentions memory-based refinement, but does not clearly report surrogate error over iterations or how many top candidates were re-evaluated with higher-fidelity methods. This weakens the “accelerating materials discovery” claim.

4. Evaluation realism is partly unclear. Many of the reported constraints (stability, synthesizability, plausibility) can be enforced by heuristics or low-cost predictors. It is not obvious how many of the “hits” would remain if checked with higher-fidelity simulation or expert curation.

5. The framework seems to rely on hand-crafted chemistry-informed rules. It is not clear how much effort is needed to move from the materials considered here to, say, MOFs, polymers, or battery electrolytes.

### Questions
1. Can you provide quantitative comparisons to the most similar recent frameworks that also do “LLM proposal + evolutionary/active search” for materials (e.g., [1])? Right now, the difference to those systems is mostly described at a high level.


2. What happens if you (1) use a smaller/general-purpose model; (2) use a materials-tuned LLM; or (3) replace the LLM with a rule/template generator? This is important to justify that the gain comes from LLM knowledge, not only from the evolutionary loop.


3. The current version seems to rely solely on surrogate-augmented oracles for property evaluation. I did not find evidence of a higher-fidelity (e.g., DFT) re-evaluation of the top candidates. Could you provide such validation or at least report surrogate error for late-stage candidates?


4. Since every generation invokes an LLM under constraints, what is the total token/runtime cost across the 14 tasks? Are there caching or retrieval-augmentation tricks to keep the loop affordable?

[1] Gan J, Zhong P, Du Y, et al. Large language models are innate crystal structure generators[C]//AI for Accelerated Materials Design-ICLR 2025. 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The proposed framework LLEMA is designed for efficient materials discovery, which includes an LLM-driven candidate generator with chemistry-aware evolutionary rules, memory pools of successful or failed candidates, and ML-based property predicting oracles. The authors designed ten materials design tasks with multi-constraint targets and stability screening. The effectiveness of the framework is demonstrated through the high valid hit rates and Pareto fronts than both generative models and LLM-only baselines. The key contribution of the work is its demonstration of materials discovery through constraining the LLM’s search with domain rules and iterative feedback, while grounding candidates with a surrogate oracle and enforcing stability.

### Strengths
1. The work is a good demonstration of employing LLMs while injecting chemistry heuristics for 'multi-objective and synthesizability-aware materials design'.
2. The diverse benchmarking tasks are well designed which align with realistic multi-constraint targets

### Weaknesses
1. No DFT validation provided on the top novel successful candidates for higher-fidelity stability evaluation.
2. The quality of the accepted or rejected designs of the pool is highly dependent on the reliability of surrogate models. Exploration is potentially limited due to the unintentional error cumulated, which the system would favor correct known designs.
3. The efficiency of the proposed framework is not fully explained.

### Questions
1. It is mentioned in A.2 that equal weights are assigned to each property in multi-objective scoring, but the section also suggests prioritized weighting. What are the specific weighting strategy used in each of the tasks? 
2. In the same section, it is also stated that  'performance-critical properties' prioritized over 'feasibility constraints' like formation energy, density, and hull stability. Is this the case for all tasks? Can you justify the design? An unstable crystal is still invalid with optimal shear modulus. How likely will the design sacrifice stability for certain performance metrics? 
3. How does the distribution of formation energy evolve over iterations in the tasks, e.g. stable wide-bandgap semiconductor task?
4. Have you examined the failure pool for failure modes? The oracles can mislabel novel candidates as failures due to OOD, e.g. novel composition missing from the patched phase diagram or absent from the Materials Project API. Will this error being cumulated over iterations and lead to a conservative search bias toward known chemistries?
5. How do you measure novelty and diversity beyond Materials Project, to ensure that the generated designs are not near-duplicates of known phases?
6. What are the average prompt token lengths, number of LLM calls, API calls and pool sizes when executing each task? How does LLEMA compare to baselines in terms of efficiency and resources required, e.g. inference time and cost.

### Soundness
3

### Presentation
3

### Contribution
3
