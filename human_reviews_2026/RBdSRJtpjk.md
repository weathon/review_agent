# SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Large language models (LLMs) have opened new paradigms in optimization modeling by enabling the generation of executable solver code from natural language descriptions. Despite this promise, existing approaches typically remain solver-driven: they rely on single-pass forward generation and apply limited post-hoc fixes based on solver error messages, leaving undetected semantic errors that silently produce syntactically correct but logically flawed models. To address this challenge, we propose SAC-Opt, a backward-guided correction framework that grounds optimization modeling in problem semantics rather than solver feedback. At each step, SAC-Opt aligns the original semantic anchors with those reconstructed from the generated code and selectively corrects only the mismatched components, driving convergence toward a semantically faithful model. This anchor-driven correction enables fine-grained refinement of constraint and objective logic, enhancing both fidelity and robustness without requiring additional training or supervision. Empirical results on seven public datasets demonstrate that SAC-Opt improves average modeling accuracy by 7.7\%, with gains of up to 21.9\% on the ComplexLP dataset. These findings highlight the importance of semantic-anchored correction in LLM-based optimization workflows to ensure faithful translation from problem intent to solver-executable code.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a backward-guided correction framework, SAC-Opt, to better utilize problem semantics to model optimization problems.

### Strengths
- A semantically driven correction method was developed.
- The authors evaluated it on several different datasets.

### Weaknesses
- As the article states, the authors cleaned all the evaluation sets, which raised a series of issues. For example, IndustryOR originally had 100 samples, but after cleaning, it only had 42 samples, less than half of the original size. But in Table 1, the authors directly cited other people's results for comparison; is this fair? I checked the repository of Xiao et al. (2025)'s article (https://github.com/LLM4OR/LLM4OR) and found that the IndustryOR test set still maintains 100 test data points. The authors need to carefully check the rigor of their experiments.
- In Table 1, SAC-Opt shows an IndustryOR of 63.7%, but no figure divided by 42 yields this result. How was this figure calculated? How many of the 42 questions were answered correctly? Similarly, the reproduction rates of 44.0% for OptiMUS-0.2 and 54.0% for OptiMUS-0.3 are unexplained. Therefore, **I seriously doubt the validity of the entire Table 1 and indeed all the experimental results presented in this paper**.
- Similarly, of the 211 samples in the ComplexLP dataset, only 111 were retained, nearly half were removed. These complex datasets have been processed, but detailed explanations are lacking. Are the authors avoiding the problem that SAC-OPT is unsuitable for solving?
- Solving optimization problems using large models is a hot topic, and authors should make a comprehensive comparison with the latest work in the field, such as LLMOPT, OptMATH, SIRL, ORLM, etc.
- Does gradual convergence lead to a longer time to obtain the answer? How to efficiently obtain a better answer should be evaluated.
- In line 34, Pyomo appears to be a modeling language rather than a solver.

### Questions
As described in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SAC-Opt, a semantics-anchored, backward correction framework for translating natural-language optimization problems into solver-executable code. The method first extracts structured elements (parameters, variables, constraints, objective), generates an initial program, and then iteratively reconstructs "semantic anchors" from the code to compare against the originals; only misaligned components are regenerated until alignment or a cap on iterations is reached. Empirically, across seven public datasets, SAC-Opt reports an average +7.8% absolute gain in modeling accuracy and up to +21.9% on ComplexLP, with ablations, cross-LLM generalization, and verification strategy comparisons. The paper provides prompt details and a reproducibility statement.

### Strengths
- The paper shows how solver-driven debugging misses semantic mistakes and frames a backward, semantics-first loop to detect them. The end-to-end workflow is clearly depicted.  The introduction concretely illustrates semantic failure modes (e.g., "upper bound" encoded as "lower bound"). The decomposition into structured vs. semantic components reduces cumulative translation errors.

-  The core mechanism is well-specified, including formalization of reconstruction, verification, and correction. Two verification instantiations (LLM-based and similarity-based) operationalize semantic equivalence. Concrete prompts for reconstruction and verification are provided.

- Across seven datasets, SAC-Opt attains best accuracy with large gains on difficult sets (e.g., +21.9% on ComplexLP). Ablations isolate the effect of semantic correction vs. solver debugging; removing correction causes substantial drops. Cross-LLM generalization indicates benefits even with weaker backbones. Verification strategy comparison shows LLM-based checking is more accurate/efficient than similarity-based alternatives.

### Weaknesses
- The "first semantics-driven correction method" claim in the Contributions is strong but not substantiated with a systematic comparison against other semantics-aware correction lines. 

- The Related Work section lists many inference frameworks but does not explicitly rule out semantics-driven correction precedents.

- The paper's empirical comparisons focus on method families but do not include a semantics-anchoring baseline beyond the two \delta instantiations internal to SAC-Opt. 

- Convergence is evidenced via a single-case error-set plot; no conditions on \delta, regeneration policy, or anchor space guarantee monotonic decrease or termination beyond a hard cap. Operation "M ← Msimp + M(t)sem" is not formalized. Composition boundaries between deterministic and regenerated segments are not precisely specified.

- Table 1 includes starred results imported from prior work; while stated to be under the same setting, such reuse can mask differences in extraction, prompting, or seeds. 

- Although the paper averages over five runs, Table 1 reports no dispersion; only Table 4 includes mean and std for internal variants. (Sec. 4.4, p.7–8; Table 4, p.8.) Why it matters: statistical robustness is unclear. :contentReference[oaicite:39]{index=39} 

- SAC-Opt increases runtime vs. the best baseline on several datasets. Average per-instance runtimes for SAC-Opt-LLM exceed many simple inference baselines in Table 7. (Table 7, p.18.) 

- The approach assumes reliable anchor extraction and adopts an external pipeline. Reported manual checks cover three datasets and suggest high extraction accuracy, but do not explore adversarial or systematic extraction errors. No controlled stress test perturbs anchor quality to quantify SAC-Opt’s tolerance.

### Questions
See Weakness

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
This paper is about turning problem descriptions given in text to optimization models (like TSP) to be solved by mixed-integer programming solvers (like Cplex, Gurobi).

The main angle of the paper is the introduction of SAC-Opt which focuses on "semantic" correctness of the constraint model along with execution correctness of the code. 

The idea is to go from text to constraint and then back; constraint to text, which allows comparing the initial text VS. generated text to have a proxy of semantic correctness. In this step, the paper proposes two methods: an LLM used as a judge for a binary agree/disagree decision and sentence similarity to detect same/not-same within some embedding distance threshold. Mismatches are corrected and the process is repeated in a loop until alignment is achieved on all constraints or iteration limit. 

The overall setup is heavily based on structured data initially generated from the given problem description text.

Experiments on existing datasets and methods show ~7% average improvement.

### Strengths
- A study on semantic-correction is a nice addition to natural language to optimization model translation.
- The paper write-up is clear (though I have a few suggestions below) 
- Experiments show added value of semantic correction (albeit at the expense of runtime)
- Shares code/data which seems reproducible --thank you!

### Weaknesses
My main concern with the paper is: the significance is not clear with respect to ICLR. 

Overall, I like this paper and it is a nice addition to an emerging array of approaches to NL4Opt. That said, the fact that adding more verification improves the correct code generation is somewhat expected, no? Here, we are looking at a %7 increase (notice most of that comes from one dataset with 21% improvement when averaged). Btw, what does this mean? Are we solving 1 or 2 (or 10) more problems correctly? It would be nice to quantify that in the number of problems to better understand the impact. So an expected improvement (at the expense of heavier approach, higher runtime) with marginal improvement.

The other thing is; the semantic verification step is an LLM detecting an inconsistency by looking at the backward step (constraint to text). I like the simplicity of this (again, at time/query expense) but what do we learn from an AI/ICLR point of view? So the overall contribution is the framework. Given the existing several frameworks, this specific improvement would be an excellent contribution, say at INFORMS. 

Other (minor) comments: 
- The presentation is clear but it fails in one important aspect. How does the re-generation works? This is one of the most important steps of the framework so I am surprised it is not explained in more detail. So say a constraint generation is detected for inconsistency. Then what happens? Do we present the LLM with the previous version and say that it's incorrect and point the incorrect example and why so? Or, something else? Most importantly, is there some "learning" here feeds back into the next generation. This is important details that must be explained in more detail. 

- It is not clear to me why the semantic verification step comes before the code verification. Code verification is easier/cheaper, no? Cannot you just run the code verification and ONLY IF it succeeds proceed with semantic verification. In the current form, this choice seems quite arbitrary. Consider an ablation. 

- Similar semantic loops are discussed as early as 2023, see e.g., the NL4Opt blueprint of Holy Grail 2.0 which is not cited https://arxiv.org/pdf/2308.01589 (btw, here the semantic step like MUS verification comes after the code verification) 

- The extract agents here is exactly "Ner4Opt" which is unfortunately not cited. pls fix. 
https://link.springer.com/article/10.1007/s10601-024-09376-5 

- The authors rightly point out the annotation contamination in their experimental datasets and referred to a "cleaned" version (thank you!). Consider citing & using Text2Zinc, which also fixes these annotation leakage issue in test datasets. 
https://arxiv.org/pdf/2503.10642

- The higher runtimes are only disclosed in the appendix. Please be more up front about it in the main paper. 

- Similarly, there is a heavy dependency on structured extraction being correct in the first place. Please be more up front about this dependency in the paper. IF extraction does not work, the rest will not matter (and plz correct me if I am correct). In Appendix, there is a comment that it actually does work well, so that seems to work in practice (OR.. the datasets are easy) 

- The experiments jump to direct results on "x% improvement". What I would have liked more is a detailed analysis/statistics on the whole pipeline. In how many problems, the extraction works perfect. In how many it fails? And by how much? Rather a binary works/not-works, how many parameters/constraints it omits, how often? Then, for how many of those the initial generation is deemed incorrect by the LLM backward verification? Is that correct at all times, OR is there a case LLM semantics, itself, is incorrect? Then, how many loops are needed to fix it. Etc. etc.. I think there is a great empirical story/study that remains untold here, and that's more interesting than getting +1 more problem modelled correctly (these problems are easy to model anyways..). But even than, that's a niche study better fit for an optimization journal.

- Curiosity question; how did you tune/decide similarity threshold? 

- The paper says that this process converge, but I don't see why (without better understanding the re-generation step)

- There are a few places with the claim "model-agnostic". Are you referring to constraint models (as in swab Gurobi with Cplex) or LLM models. IF llm models, is not this obvious? We can feed in the prompt to LLM-A or LLM-B. This seems trivial to me so I am not sure why the paper is making this claim. Do you mean you found "universal prompts" (by some definition) that works for all LLMs? I am not really following this part. Please revise (or better, drop that part)

### Questions
- Can you say more about re-generation step once inconsistencies are found? This is a crucial step of the paper that I am surprised not discussed in more detail? 

- I really do like the paper but what do we learn from AI/ICLR point of view? (in the light of several works have done this already)

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SAC-Opt, a novel framework designed to improve large language models’ (LLMs) ability to translate natural-language descriptions of optimization problems into solver-ready mathematical formulations (e.g., linear or mixed-integer programs). The key idea is the introduction of Semantic Anchors (SAs) — structured symbolic hints that connect linguistic elements (e.g., “maximum capacity”, “budget limit”) with optimization primitives (variables, constraints, and objective terms).

### Strengths
1. Clear and Motivated Problem: The task, mapping natural-language optimization problems into executable mathematical programs, is both underexplored and highly relevant for LLM reasoning research.

2. Methodological Soundness: The two-phase design (supervised + consistency regularization) is technically reasonable and fits within modern neuro-symbolic learning trends.

3. Readable & Reproducible: Dataset and training pipeline are clearly documented.

### Weaknesses
1. Semantic Anchors are manually curated and lack scalability: Claim of “neurosymbolic generality” (p.2, paragraph 3) is not substantiated: the current anchors are symbolic labels, not learned semantics.	There’s no automated anchor discovery or learning, which limits scalability to unseen optimization structures or domains (e.g., stochastic, nonlinear, combinatorial).

2. Anchor Consistency Loss is heuristic and lacks formal grounding: There is no analysis of failure cases where paraphrase consistency fails (e.g., negation or quantity inversion).

3.  The framework does not actually perform symbolic reasoning: Although labeled “neurosymbolic,” the framework never executes or verifies symbolic constraints during training in line 120-150. The term “neuro symbolic optimization” seems to be overstated; the system remains purely neural.

4.  Missing related work: “ReFT (Luong et al. 2024)” and “DeepSeek-RFT (2025)” are relevant to semantic alignment.

### Questions
As shown in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
