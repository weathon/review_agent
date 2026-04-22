# Neural Theorem Proving for Verification Conditions: A Real-World Benchmark

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Theorem proving is fundamental to program verification, where the automated proof of Verification Conditions (VCs) remains a primary bottleneck. Real-world program verification frequently encounters hard VCs that existing Automated Theorem Provers cannot prove, leading to a critical need for extensive manual proofs that burden practical application. While Neural Theorem Proving (NTP) has achieved significant success in mathematical competitions, demonstrating the potential of machine learning approaches to formal reasoning, its application to program verification—particularly VC proving—remains largely unexplored. 
Despite existing work on annotation synthesis and verification-related theorem proving, no benchmark has specifically targeted this fundamental bottleneck: automated VC proving.
This work introduces Neural Theorem Proving for Verification Conditions (NTP4VC) and presents the first real-world multi-lingual benchmark for this task. Specifically, from real-world projects such as Linux and Contiki-OS kernel, our benchmark leverages industrial pipelines (Why3 and Frama-C) to generate semantically equivalent test cases across formal languages of Isabelle, Lean, and Rocq. We evaluate large language models (LLMs), both general-purpose and those fine-tuned for theorem proving, on NTP4VC. Results indicate that although LLMs show promise in VC proving, significant challenges remain for program verification, highlighting a large gap and opportunity for future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces NTP4VC, a new task and benchmark that targets neural theorem proving of verification conditions (VCs)—arguably a core bottleneck in automated program verification. The authors (i) extract VCs from real-world projects via industrial pipelines (Why3/Frama-C), (ii) translate them with ~2.4K expert-written mapping/rewriting rules into three ITPs (Lean, Isabelle, Rocq), (iii) complicate the VCs by erasing helper annotations to make proofs harder while remaining provable, and (iv) evaluate LLM provers and hammer baselines. The released benchmark contains 672 VCs spanning pearls (algorithms/data structures) and real C verification (Contiki OS, Linux scheduler, etc.), with difficulty balanced so Why3’s strongest ATP tactic hovers around ~20% success. Across languages, modern LLM provers achieve <4% pass@8 while Isabelle/Coq hammers perform better but still modestly (up to ~15% pass@1).

### Strengths
1. Problem focus is well-motivated. VC theorem proving sits at the heart of verification workflows and is less explored by NTP than math benchmarks. By targeting VCs directly (rather than only annotation synthesis), the paper addresses a real bottleneck and usefully complements recent annotation-centric efforts such as Laurel and DafnyBench. 

2. Extracting from Why3/Frama-C, translating to Isabelle/Lean/Rocq, and sourcing from Contiki/Linux bring the setting closer to practice than e.g., purely math (miniF2F/PutnamBench) or toy code-property suites.  The two halves (pearls vs. real C) and per-category balancing around a target ATP pass rate are thoughtfully constructed; the complication step measurably increases hardness. 

3. Results convincingly show VCs are qualitatively harder for current NTP models than math/formal benchmarks.

### Weaknesses
1. Concerns regarding translation correctness: The benchmark depends on ~2.4K hand-written translation/rewriting rules and occasionally axiomatizes termination when Isabelle/Rocq can’t discharge it. There is no formal correctness proof of the translation or quantitative post-hoc validation beyond syntax checks/cross-review, leaving risk of subtle semantic drift or unsoundness through added axioms. A minimal mechanized proof of semantics-preservation for a core calculus would be beneficial.

2. Contamination analysis is qualitative. Given that “engineering” cases show unusually high NTP success relative to others, please run decontamination (code/proof snippet search against common training corpora) to support the low-contamination claim.

### Questions
1. From >5.3k extracted VCs, 672 are “carefully selected,” but criteria are only qualitatively described. Can you please release the full pool with metadata (size, quantifier alternation, AST depth, symbol counts, dependency graph) and the sampling procedure?

2. For a representative subset, can you show end-to-end proofs exist in at least one target ITP (perhaps with generous hammer help), confirming that difficulty stems from complexity rather than unprovability introduced by erasing?

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
The authors propose a benchmark for proving verification conditions. They extract these benchmarks from a combination of real world code, where the VCs are extracted using existing industrial tools, and are then simplified massively by removing hints like asserts and lemmas. The authors then evaluate existing LLMs, NTPs and Hammers.

### Strengths
1. The work is novel to the best of my knowledge, and addresses a fundamental gap: automating program verification for real-world complex programs.
2. Thorough evaluation of models and NTPs on the generated datasets. I also appreciate the datasets being created for different languages.

### Weaknesses
1. Lack of details on how the 5.3K VCs get filtered down to 627: I'm not sure why this was done, or what were the criteria for eliminating majority of the VCs. Were they too similar to each other, making them redundant? Or was it some deeper issue? Regardless, the elimination process was not described in the paper. Furthermore, out of the 5.3K, how many were extracted from the real-world programs and how many were extracted from the pearls of programs?
2. The pearls of programs: Does the inclusion of these pearls of programs not contradict the real-world claim of the title of the paper? I understand the need for diverse VCs, but the pearls seem more as an exercise in niche settings rather than VCs that occur frequently in realistic programs.
3. Discussion centered around improving NTPs: I would like a discussion that focuses on how NTPs could improve on such VCs. I assume that these VCs would be difficult to prove even for humans. Even if one would try to perform RL, the initial accuracies of NTPs are not enough to provide straightforward gains. How could one try to improve NTPs on such complex benchmarks?

### Questions
See weaknesses. Also:
1. Have you thought about composing different VCs to try and create even more complex benchmarks?
2. How easy is it to add more projects to the set of real-world verified programs?
3. Can you elaborate on your statement in lines 403 -- 408? I am not sure I really understand why one category performs better than the other.

### Soundness
3

### Presentation
4

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
The authors introduce NTP4VC, a novel benchmark targeting neural theorem proving (NTP) for program verification conditions (VCs). Unlike prior datasets focused on mathematical or auxiliary lemmas, NTP4VC consists of real-world VCs automatically extracted from industrial projects using mature verification pipelines. The benchmark covers multiple languages and a diverse range of verification scenarios, aiming to reflect practical challenges in automated program verification. Experimental results show that existing neural and automated provers perform poorly on NTP4VC.

### Strengths
1. **Benchmark contribution:** 

The authors introduce a new dataset specifically addressing VC proving, which is a practical bottleneck in automated program verification.

2. **Industrial relevance:** 

VCs are extracted from real-world industrial projects using established verification pipelines (Why3, Frama-C), increasing practical relevance.

3. **Multi-language and scenario coverage:** 

The dataset spans multiple proof assistant languages and aims to cover a variety of verification scenarios.

### Weaknesses
1. **Limited dataset novelty:** 

Although the focus on VCs is clear, the novelty is moderate. Previous datasets (e.g., CoqStop, FVEL, CoqGym) have also included verification-related theorems, even if the VC proportion was lower. The paper should better highlight what fundamentally differentiates NTP4VC—beyond just the percentage of VCs.

2. **Lack of technical contribution:** 

The work mainly involves dataset construction and benchmarking; there is no new technical advance.

3. **Provability not manually verified:** 

While VCs are generated using automated tools, there is no evidence of manual or sampled verification to guarantee their provability. This raises the risk that some VCs may be invalid or unprovable due to the toolchain.

4. **No quantitative analysis of VC diversity:** 

The paper focuses on diversity of program sources, but does not quantitatively analyze the types of generated VCs. It is unclear whether the dataset covers a broad range of VC properties or is dominated by a few repetitive types. A breakdown of VC categories/properties would strengthen the empirical evaluation.

### Questions
See Weaknesses

### Soundness
2

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
5

### Summary
This paper propose a challenging and real-world program verification benchmark, theorem proving for verification conditions (VCs). Different from most theorem proving benchmarks, VCs are the primary (and transient) intermediate representations of program verification in the real world. This work first collects VCs by instrumenting a popular program verifier, Why3, and then converting them into three popular interactive theorem provers (ITPs), i.e., Isabelle, Lean, and Rocq,  based on manually designed rules (around 800 for each ITP). The verification tasks span two categories: 1) small challenging puzzles that involve non-trivial data structures and algorithms, and 2) C programs from real-world projects. The experimental evaluations show that neural theorem provers (NTP) achieve less than 4% pass rate, which is consistently worse than classic ATPs like CoqHammer/Sledgehammer. The new benchmark remains challenging for both ITPs and ATPs, motivating technical innovations in both.

### Strengths
- Constructing a real-world program verification benchmark using VCs is well-motivated and the paper is well-written. 
- The benchmark construction is based on high-quality expert-written rules, which is reliable and one-time effort. 
- The NTP4VC benchmark strikes a good balance of two categories of challenging program verifications, specifically, small programming puzzles and real-world projects. Furthermore, the transient nature of VCs make potential contamination less of a concern.
- Experimental evaluations show the clear gap as well as promises of neural theorem provers, motivating future follow-up works.

### Weaknesses
- compared to programs and annotations at the source code level, VCs and relevant proofs are perhaps too low-level and less suitable for LLMs to generate proofs, since human experts rarely write proofs for VCs. Although NTP4VCs is an interesting and challenging benchmark, but may not be necessary to tackle directly by LLMs. The cost is also prohibitive, as indicated by the limited evaluation with GPT-o4-mini-high. 
- three themes of errors (presented in sec 5.2) are well expected; having a quantitative analysis will be more helpful.

### Questions
How large are the translated theorems in ITPs? Is there always a single proof goal? Are they human readable?

The results presented in Table 5 are really interesting. Besides comparing the absolute number of solved theorems, are they complementary or roughly a subset of another?

### Soundness
4

### Presentation
4

### Contribution
3
