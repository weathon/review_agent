=== CALIBRATION EXAMPLE 5 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately describes the system, but the abstract sets up expectations that the paper cannot deliver. The abstract claims the framework is "remarkably effective at proposing such decompositions" and that it addresses a question posed by Terry Tao — but the paper ultimately demonstrates this on only two examples, both drawn directly from Tao's own blog post/MathOverflow answer. Calling this an answer to Tao's question is a significant overstatement; the paper demonstrates feasibility on illustrative toy examples, not a systematic study. The phrase "No existing AI tools are able to complete and symbolically verify proofs of this kind" is stated without empirical evidence.

---

### Introduction & Motivation

The motivation is genuinely interesting and well-chosen — domain decomposition as the creative bottleneck in asymptotic analysis is a real problem that Tao and others have articulated. However, several formulas that are central to the argument (the Riemann Hypothesis asymptotic form, the series in Case Study 2, the AM-GM variant) are simply missing from the rendered text — these are significant enough that the paper's story is incomplete without them. While this may be a PDF parsing issue, the introduction's argumentative thread is hard to follow in their absence.

The contributions bullet point states "Frontier LLMs often provide incorrect proofs" as a motivation for the tool, but there is no citation or quantitative evidence of how often LLMs fail on these specific types of problems.

---

### Section 2: Framework (LLM-Proposed Decomposition + CAS Verification)

This is the core technical section, and it is severely underdeveloped.

**Missing prompt template.** The prompt in Section 4 (referenced here) is entirely empty — the XML tags `<guiding_principles>`, `<task>`, `<requirements_for_breakpoints>`, `<output_format>` all have no content. This is the single most critical reproducibility failure in the paper. The LLM's proposed decomposition is the creative bottleneck the authors themselves identify, and the prompt engineering driving that decomposition is completely hidden. It is impossible to reproduce, evaluate, or build upon the system without knowing what was prompted.

**Missing Mathematica code.** The code snippet in Section 4 is clearly truncated — only a few lines are shown with no surrounding context. The reader cannot understand how LLM output is parsed, how variable names are extracted, or how the Resolve call is structured.

**Search over C.** The paper mentions searching C over a grid from 1 to 10⁴ (Step 4). This is a non-trivial algorithmic decision. If the true constant is very large, the system will report failure even when the inequality holds. The paper asserts all tested examples used C ≤ 2, but this is an empirical claim without justification as a property of the problem class. This bound could silently fail on problems where a large constant is needed.

**No formal algorithm box.** There is no pseudocode or precise algorithmic description. "Regime-wise simplification" (Step 3) is described in one paragraph without specifying when or how denominators are detected as sums of positive terms.

---

### Section 3: Case Studies

**Case Study 1.** The asymptotic inequality xy ≪ x log x + eʸ is indeed a nice example. The manual proof shown (y ≤ 2 log x vs. y > 2 log x) is clear and correct. However, the paper claims the LLM "correctly" proposed this decomposition — but there is no evidence of this. Was the decomposition proposed by the LLM verbatim? Did the LLM propose it on the first try? How many attempts were needed? Was this specific decomposition hand-verified, or did the tool just run Resolve and succeed? Without this detail, the reader cannot assess the LLM's actual contribution vs. the CAS's.

**Case Study 2.** The series decomposition case is more ambitious, but the actual series formula is missing from the parsed text, making the discussion nearly impossible to evaluate. The breakpoints {⌈h⌉, ⌈hm⌉} are described but the reader cannot verify that the LLM actually found these, versus the authors feeding them in manually.

**Structural disorganization.** The paper's sections appear out of order in significant ways. Text that compares O-Forge with AlphaGeometry (the two bullet points about not needing to train from scratch, and using Resolve) appears mid-paper between the two case studies, disconnected from the Related Work section where it belongs. This reads as if sections were moved without updating transitions.

---

### Section 5: Empirical Evaluation

This section is the paper's most serious weakness from an ICLR perspective.

- **No quantitative results.** "40–50 easier problems" are described, but there is no table, figure, or even a number reporting the success rate.
- **No dataset description.** What are these problems? Where do they come from? How were they selected? Are they publicly available?
- **No baselines.** The paper claims superiority over direct LLM use, over Z3/CVC5/MetiTarski, and over Lean tactics — but provides no head-to-head comparison on a common set of problems. The statement that CVC5 and MetiTarski cannot prove "log x ≤ log y ⟹ exp(x) ≤ exp(y)" is offered as a single anecdote, not a systematic benchmark.
- **No failure analysis.** What happens when the LLM proposes an incorrect decomposition? Does the system fail gracefully? How often does this happen? What is the fallback strategy?
- **No ablation.** The paper does not study what happens if one removes the LLM (i.e., tries random or systematic decompositions) or uses a weaker LLM. The LLM's actual contribution to success is entirely unclear.
- **No runtime or scalability data.** How long does Resolve take as the number of variables or the complexity of the decomposition increases?

The empirical section as written amounts to a qualitative claim that the system "generally works."

---

### Related Work

The related work is thin and selective. It engages with AlphaGeometry, Lean tactics, and autoformalization, but omits:

- DSOS/SDSOS/SOS (Sum-of-Squares) methods for proving polynomial and algebraic inequalities — a well-established algorithmic approach.
- Tools like SAGE's `qepcad` interface and Polyrith.
- The large literature on automated inequality proving (e.g., Sturm's theorem applications, polyrith in Lean/Coq).
- Work on LLMs + formal verification more broadly (e.g., Draft-Sketch-Prove, COPRA).
- Whether any symbolic computation system can already prove the two case-study examples out of the box, without LLM assistance.

The comparison to Tao's own estimates tool (Tao, 2025b) is described as "O-Forge extends this work greatly" — but the difference is primarily the choice of CAS (Lean/linarith vs. Mathematica/Resolve), not a fundamentally new approach.

---

### Writing & Clarity

Beyond the structural disorganization noted above, there are substantive clarity failures that impede understanding:

- The paper uses an unfinished placeholder at line 94: `"(** describe the structure of the prompt**)"` — this appears to be an author's internal note that was never filled in.
- The website is referred to as both "o-forge.com" (§1.1, §4, §8) and "o-forge.net" (Appendix B) — a contradictory detail that raises reproducibility questions.
- The reference to `Anonymous (2025)` for the code repository links to a GitHub URL (breaking double-blind review), which is a protocol violation.
- The claim about Wikipedia as a citation for the AM-GM inequality (Wikipedia contributors, 2025) is not appropriate for an academic venue.

---

### Limitations & Broader Impact

The limitations section honestly acknowledges the lack of proof objects, which is the most significant foundational concern. However, additional limitations go unaddressed:

1. **Completeness is not guaranteed.** If Resolve times out or returns an indeterminate result on a subdomain, the system fails silently. This is not discussed.
2. **Scope is very narrow.** The system appears to handle only inequalities of the form f ≤ C·g where f and g involve elementary and transcendental functions of a small number of variables. The claim that this is "research-level" mathematics needs more careful qualification.
3. **LLM reliability.** The paper notes that "making API calls to Gemini only sporadically gave us the correct simplifications" — this is a significant admission of unreliability that is not quantified.
4. **The tool is not interactive.** If the LLM proposes a wrong decomposition and Resolve fails, there is no described mechanism for refinement or feedback within the loop. The "In-Context Symbolic Feedback loop" described in the abstract is not detailed or demonstrated.

---

### Overall Assessment

O-Forge addresses a genuine and interesting problem — automating asymptotic inequality proofs via LLM-guided domain decomposition + CAS verification — and the core idea has merit. However, the paper in its current form falls well short of ICLR's acceptance bar on nearly every dimension. The prompt template, which is the paper's central technical contribution, is completely missing from the submission. The empirical evaluation consists of two case-study examples and an unreported set of 40–50 problems with no quantitative outcomes, no baselines, and no ablations. Several formulas central to the argument are absent from the submitted text. The writing contains internal editorial placeholders and apparent double-blind violations. The claims of novelty ("No existing AI tools...") are not substantiated by systematic comparison. The contribution is essentially: use an off-the-shelf LLM to propose a decomposition, call Mathematica's Resolve — a useful engineering idea, but not a scientific advance that has been empirically validated. In its current state, this paper requires major revision before it can be evaluated fairly, let alone accepted.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents O-Forge, a framework that couples frontier Large Language Models (LLMs) with a Computer Algebra System (CAS), specifically Mathematica's Resolve function, to rigorously verify asymptotic inequalities. The system leverages the LLM to propose domain decompositions that simplify complex estimates, while the CAS provides symbolic verification across the proposed subdomains. The authors demonstrate this approach through two case studies involving research-level inequalities and series, claiming to bridge the gap between contest math and automated research assistance.

### Strengths
1.  **Practical Utility and Targeted Impact:** The tool addresses a specific, painful bottleneck in mathematical analysis (asymptotic estimation) where standard theorem provers often fail. The claim that Mathematica's Resolve handles transcendental functions better than Z3/CVC5 or current Lean tactics is a valuable practical insight for AI4Math.
2.  **Decomposition Strategy:** The separation of "creative reasoning" (LLM domain decomposition) from "rigorous verification" (CAS quantifier elimination) is a sound architectural choice that mirrors successful strategies in systems like AlphaGeometry (Trinh et al., 2024). It mitigates the LLM's hallucination problem by offloading proof validity to a symbolic engine.
3.  **Accessibility:** The provision of both a CLI and a web interface (o-forge.com) significantly lowers the barrier to entry for mathematicians who may not be comfortable with command-line tools or complex coding environments, potentially increasing adoption within the research community.

### Weaknesses
1.  **Limited Empirical Evaluation:** The evaluation consists of only two specific case studies and a vague mention of "40-50 easier problems." There is no benchmark dataset, no baseline comparison against other LLM+CAS pipelines, and no analysis of failure modes or false positives/negatives. Without statistics on success rates over a diverse dataset, claims of general effectiveness are unsubstantiated.
2.  **Reliance on Closed-Source Verification:** The core verification relies on Mathematica's Resolve function, which does not produce externally verifiable proof objects. As acknowledged in the paper, this introduces a "trust" element with a commercial black box, which contradicts the rigor expected in mathematical tools compared to open-source alternatives like Lean or Coq, even if those are currently less capable with transcendental functions.
3.  **Lack of ML Novelty:** The method primarily integrates existing APIs (LLM + CAS) without novel algorithmic contributions. There is no discussion of prompt engineering optimization, finetuning, or how the decomposition quality correlates with performance. It reads more as an engineering integration paper than a methodological contribution suitable for ICLR's algorithmic depth standards.

### Novelty & Significance
**Novelty:** The novelty is moderate. While the specific combination of LLM decomposition + Mathematica Resolve for *this specific* domain (asymptotic analysis) is novel, the high-level "LLM propose steps + Verifier check" pattern is well-established (e.g., AlphaProof, AlphaGeometry). The contribution lies more in the system assembly and domain adaptation than in new theory.

**Significance:** The potential significance is high for the niche of analysis and number theory where manual verification is tedious. If validated, it offers a genuine research partner tool. However, the inability to produce formal proof objects limits its integration into formal verification pipelines, capping its broader theoretical impact.

**Reproducibility:** Code is provided (via GitHub), but the workflow is heavily dependent on proprietary software (Mathematica) and paid API keys for frontier LLMs. The "Reproducibility" section notes Python 3.9+ and Mathematica access, which is standard but creates a dependency barrier for full independent verification.

### Suggestions for Improvement
1.  **Rigorous Benchmarking:** Implement a standardized benchmark set of asymptotic inequalities with known ground truths. Report precision, recall, and failure rates compared to baselines (e.g., LLM alone, CAS alone, other solvers like Z3/CVC5).
2.  **Ablation and Sensitivity Analysis:** Conduct an ablation study to determine if the LLM's specific prompts or specific decompositions are sensitive to noise. Quantify how often the LLM suggests a decomposition that the CAS cannot verify and whether subsequent loops (self-correction) improve success.
3.  **Open-Source Verification:** Consider experimenting with open-source alternatives or discussing a path toward autoformalization (e.g., Lean/Coq) that could eventually replace the closed-source Mathematica Resolve step, addressing the "proof object" limitation.
4.  **Case Study Detail:** The "Case Study" sections are overly qualitative. Provide explicit examples of the LaTeX input, the specific prompt used to the LLM, the exact decomp output, and the CAS verification time/log for the reader to understand the actual workflow mechanics.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Defined Benchmark & Success Rates:** Provide the exact list and success statistics for the claimed "40-50 easier problems" rather than vague descriptions. Without quantitative success rates on a fixed dataset, the claim of robustness is unsubstantiated.
2.  **Baseline Comparisons:** Compare O-Forge against pure LLM proof generation, standalone CAS usage, and Lean-based tactics (e.g., `linarith`). Without baselines, it is impossible to determine if the LLM+CAS combination offers any advantage over existing tools.
3.  **Ablation on Decomposition Strategy:** Test whether the LLM's decomposition performs better than random splits or standard heuristic thresholds. This is necessary to verify the core claim that the LLM provides unique "creative" value versus simple automation.
4.  **False Positive Evaluation:** Test the system on known false inequalities to measure the false positive rate of the `Resolve` verifier. A verification tool must demonstrate it rejects incorrect claims, not just verifies correct ones.
5.  **Iterative Refinement Performance:** The abstract claims an "In-Context Symbolic Feedback loop," but Section 3 states the LLM is prompted only once. Experiment with multi-turn refinement to validate the claimed feedback mechanism.

### Deeper Analysis Needed (top 3-5 only)
1.  **Verifier Trustworthiness:** Analyze the risk of relying on Mathematica's closed-source `Resolve` without proof objects for a paper claiming "rigorous verification." This directly undermines the credibility of the proofs in a formal mathematics context.
2.  **Loop Discrepancy:** Reconcile the contradiction between the abstract's "feedback loop" and the method's "single-shot" LLM prompt. If no feedback loop exists, the core architectural claim is misleading.
3.  **Simplification Validity:** Rigorously justify the "leading-order term" simplification assumption used before verification. If this heuristic fails on complex summands, the entire verification pipeline collapses.
4.  **Scalability Analysis:** Analyze how verification time and success rates scale with the number of variables and decomposition subdomains. Research-level problems often involve high dimensions; without this, utility is limited to toy examples.
5.  **Error Propagation:** Analyze how errors in the LLM's decomposition proposal propagate to the CAS verification step. Understanding failure modes is critical for users to trust the "True/False" output.

### Visualizations & Case Studies
1.  **Success/Failure Distribution Plot:** Visualize the success rate across problem complexities (variable count, term count) to expose performance cliffs. This reveals whether the method works generally or only on specific, simple classes of inequalities.
2.  **Decomposition Quality Comparison:** Visualize an LLM-proposed domain split versus the mathematically optimal split for a specific case study. This exposes whether the LLM is finding non-trivial decompositions or just obvious thresholds.
3.  **Failure Case Deep Dive:** Present a concrete case where the system fails (either wrong decomposition or verifier timeout) and analyze why. Showing failure modes is essential to establish the boundary of the method's applicability.

### Obvious Next Steps
1.  **Integrate Open-Source Verifier:** Replace or supplement Mathematica with an open-source proof assistant (e.g., Lean 4) to generate independently verifiable proof objects. This is required to meet the standard of "rigorous" mathematical tooling.
2.  **Implement Actual Feedback Loop:** Modify the system to allow the LLM to retry decompositions when `Resolve` returns False, fulfilling the "feedback loop" promise in the abstract.
3.  **Release Evaluation Dataset:** Publicly release the "40-50 problem" dataset as a benchmark for asymptotic analysis. Without this, the empirical claims cannot be reproduced or validated by the community.

# Final Consolidated Review
## Summary

O-Forge presents a framework combining frontier LLMs with Mathematica's Resolve function for proving asymptotic inequalities. The LLM proposes domain decompositions, and the CAS verifies each subdomain via quantifier elimination. The paper demonstrates feasibility on two case studies from Terence Tao's blog posts and claims testing on 40-50 additional problems.

## Strengths

- **Addresses a genuine research bottleneck**: Domain decomposition is widely recognized as the "creative" step in proving asymptotic inequalities. Automating this step addresses a real pain point for analysts and number theorists, as noted by Tao's own commentary on AI-assisted mathematics.

- **Sound architectural separation**: The division of labor between LLM-guided decomposition and CAS verification follows a principled design pattern that mitigates LLM hallucinations by requiring symbolic proof for each subdomain. This mirrors successful approaches like AlphaGeometry while targeting a different mathematical domain.

- **Practical accessibility**: The web interface (o-forge.com) lowers barriers for mathematicians without programming expertise, making the tool immediately usable by the target research community.

## Weaknesses

- **Missing prompt template destroys reproducibility**: Section 4 contains "(** describe the structure of the prompt**)" followed by empty XML tags for `<guiding_principles>`, `<task>`, `<requirements_for_breakpoints>`, and `<output_format>`. The LLM prompt is the core "creative" contribution of the system—without it, the work cannot be reproduced, evaluated, or improved upon. This is a critical omission.

- **No quantitative evaluation despite claiming 40-50 test problems**: The paper states it tested on "40-50 easier problems" but provides no success rates, failure analysis, or even a description of these problems. Without quantitative data, claims of robustness and effectiveness are unsubstantiated.

- **No baselines or false-positive testing**: The paper asserts superiority over Z3, CVC5, and MetiTarski based on a single anecdote (log x ≤ log y ⟹ exp(x) ≤ exp(y)). There is no systematic comparison, and crucially, no testing on *false* inequalities to verify the system correctly rejects invalid claims. A verification tool must demonstrate it does not produce false positives.

- **Abstract claims "feedback loop" but method is single-shot**: The abstract describes an "In-Context Symbolic Feedback loop," yet Section 3 explicitly states "we only prompt the LLM once in the entire process." There is no mechanism for iterative refinement when Resolve fails. This mismatch between claimed architecture and actual implementation is misleading.

- **Simplification assumptions lack justification**: Step 3 ("Regime-wise simplification") extracts leading-order terms from numerators/denominators. The paper acknowledges this "may not be valid simplification for more complex summands" but provides no analysis of when it fails or how often. The entire pipeline's correctness depends on this heuristic.

- **Closed-source verification undermines formal credibility**: The paper acknowledges that Mathematica's Resolve does not produce proof objects. While practical, this means "verified" proofs cannot be independently audited—contradicting the paper's emphasis on "rigorous" verification for research mathematics.

## Nice-to-Haves

- **Algorithm pseudocode**: The method is described procedurally but lacks a formal algorithm box specifying exact inputs, outputs, and decision points.

- **Ablation on decomposition quality**: Testing whether LLM-proposed decompositions outperform simple heuristics (dyadic splits, random thresholds) would clarify the LLM's actual contribution versus baseline automation.

- **Path toward open-source verification**: Discussion of how autoformalization or open proof assistants could eventually replace the closed-source Mathematica dependency.

- **Iterative refinement when decomposition fails**: The natural extension—prompting the LLM with failure information when Resolve returns False—is suggested by the abstract but never implemented.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Missing formulas (Harsh Critic)*: The reviewer claimed central formulas were missing from the paper. The formulas ARE present (e.g., xy ≪ x log x + eʸ). The confusion appears to stem from PDF parsing artifacts, not author omissions.

- *Double-blind review violation (Harsh Critic)*: The reviewer claimed the GitHub URL breaks double-blind protocol. However, the citation is properly anonymized as "Anonymous (2025)" with a blinded repository URL, which follows correct double-blind procedures.

- *Wikipedia citation inappropriate (Harsh Critic)*: Citing Wikipedia for the well-known AM-GM inequality is a minor style issue, not a substantive flaw that affects the paper's contribution.

- *Lack of ML novelty (Balanced Reviewer)*: This criticism demands the paper be an ML methodology contribution when it is clearly an AI4Math systems paper. The integration of LLM+CAS for this specific domain is the claimed contribution, not novel ML algorithms.

- *Scalability analysis demand (Spark Finder)*: Requesting analysis of scaling with dimensions and complexity is reasonable for future work but exceeds what is expected for a first paper introducing a new tool with demonstrated feasibility.

## Novel Insights

The observation that quantifier elimination via Mathematica's Resolve can verify transcendental function inequalities that defeat current SMT solvers and Lean tactics is a genuinely useful practical finding for the AI4Math community. This points to a capability gap worth addressing in open-source proof assistants. Additionally, the decomposition-based approach correctly identifies that the "creative bottleneck" in these proofs is not the final verification step (which is mechanical) but the domain splitting strategy—this suggests LLMs may be best deployed for proposal tasks rather than end-to-end proof generation in mathematical domains where verification can be offloaded to symbolic engines.

## Suggestions

1. **Complete the prompt template** before any revision. This is non-negotiable for reproducibility. Include the exact prompt used for the case studies.

2. **Report actual success rates** for the claimed 40-50 problems. At minimum: (a) the problem set description, (b) number of successful verifications, (c) types of failures.

3. **Test on false inequalities** to demonstrate the system does not verify invalid claims. Report false-positive rate.

4. **Clarify the feedback loop**: Either implement iterative refinement when decomposition fails, or remove "feedback loop" language from the abstract and framing.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject
