=== CALIBRATION EXAMPLE 5 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is reasonable, though "asymptotic analysis" is somewhat broad — the actual scope is narrower: asymptotic *inequalities* amenable to domain decomposition followed by quantifier elimination. The abstract makes strong claims: the framework "produces proofs that are both creative and symbolically verified," and "answers a question posed by Terry Tao." Both claims need scrutiny.

The "In-Context Symbolic Feedback loop" described in the abstract is never formally defined anywhere in the paper. Reading the actual system description in Sections 2–4, the process appears to be: one LLM call → Mathematica verification, with no feedback loop back to the LLM if verification fails. If there is no actual iterative feedback mechanism, calling it a "loop" is misleading. If there is one, it is never described.

The claim about "answering Tao's question" is oversold. Tao's MathOverflow post (2024) identifies domain decomposition as a useful strategy; the paper applies this strategy to two illustrative examples from that same post. This is a proof-of-concept demonstration, not a research-level answer.

---

### Introduction & Motivation

The motivation is well-chosen and genuinely compelling: asymptotic inequalities are central to analysis and number theory, and the "find the right decomposition" insight is mathematically accurate. The running example of `xy ≪ x log x + e^y` is concrete and appropriately illustrative.

However, the introduction contains an editorial artifact that should never appear in a submitted paper: the parenthetical **(** describe the structure of the prompt**)** on page 6 within the description of Step 2. This strongly suggests the paper was submitted in an unfinished state.

The contribution list is also thin relative to the claims. Contribution 1 (the O-Forge tool and website) is engineering, not research. Contributions 2 and 3 are two specific examples, not a systematic study. There is no general theorem, no complexity analysis, no formal characterization of the problem class the method handles.

---

### Method / Approach (Sections 2–3)

**Step 2 (Decomposition proposal)** is described only at a high level. The reader is told the LLM "proposes a finite cover guided by cues such as dominant terms and monotonic regimes," but the actual prompt structure — which is the core intellectual contribution of this component — is entirely absent. Section 4 provides only empty XML tags (`<guiding_principles></guiding_principles>`, `<task></task>`, `<requirements_for_breakpoints></requirements_for_breakpoints>`, `<output_format></output_format>`), which are clearly unfilled placeholders. **This is a critical omission**: the paper claims that a "structured prompt" is what enables reliable LLM performance, but that prompt is not disclosed.

**Step 4 (Symbolic verification via Resolve)** raises legitimate concerns about the scope of the method. The paper relies on Mathematica's Resolve function via quantifier elimination over the reals (Tarski-Seidenberg). This is a decision procedure for the first-order theory of the reals, which is decidable but has doubly exponential complexity in the alternation of quantifiers. For the specific examples shown (two-variable inequalities over simple regimes), this works. But the paper provides no characterization of the class of problems for which this terminates in practice, nor any discussion of timeouts or undecidability risks for more complex expressions. This matters substantially for the claimed applicability to "research-level" problems.

**Case Study 1** (the `xy ≪ x log x + e^y` example): The mathematical walkthrough is clear and the proof by domain decomposition is correct. However, the paper does not report that O-Forge successfully found the decomposition `{y ≤ 2 log x, y > 2 log x}` and verified it — it merely explains the decomposition manually and asserts that Mathematica can verify each piece. There is no experiment showing that the LLM actually proposed exactly this decomposition, how many attempts were needed, or whether the CAS successfully returned True. This is the central empirical claim of the paper and it is unsubstantiated.

**Case Study 2** (the series decomposition): The actual series formula appears to have been lost in PDF parsing, which makes it impossible to evaluate the claim. The discussion of breaking points `{⌈h⌉, ⌈hm⌉}` is present but the target estimate is missing. Setting aside the parsing issue, the same problem applies: no concrete experimental record is provided — no LLM transcript, no Mathematica output, no runtime.

---

### Implementation (Section 4)

This section is effectively empty. The Mathematica code snippet provided is a fragment with no context or explanation:

```
Resolve[ForAll[{series.other_variables},
    logForm["Resolve results", res2];
If[AllTrue[res2,TrueQ],True,res2]
```

This is syntactically incomplete, and the ellipsis is never filled. The prompt shown consists entirely of empty XML tags. The CLI description is brief and provides no insight into system design.

For a paper whose primary contribution is a software tool, this is deeply inadequate. A reviewer cannot assess reproducibility, correctness, or generalizability from what is provided.

---

### Empirical Evaluation (Section 5)

This section is the most critical failure of the paper. The entire empirical evaluation consists of:

- Testing "around 40-50 easier problems" (the vagueness of "around 40-50" is itself a red flag)
- Three qualitative observations about decomposition counts and leading-term simplification
- The conclusion that "our approach is robust"

There are **no tables**, **no quantitative success rates**, **no failure analysis**, **no comparison with any baseline system on the same problem set**, **no statistical reporting of any kind**. The 40-50 problem dataset is not released, not described in any systematic way, and not analyzed. ICLR expects rigorous empirical evaluation; this section does not come close to meeting that bar.

Specifically absent:
- Success rate of the LLM in proposing decompositions that lead to successful verification (first-try and after retries)
- Comparison: what fraction of these problems can Mathematica's Resolve solve *directly*, without any LLM-proposed decomposition?
- Ablation: what is the LLM's contribution over simply calling Resolve with a generic decomposition heuristic?
- Runtime statistics: how long do Resolve calls take across problem types?
- Failure modes: what categories of problems does the system fail on, and why?

The claim that "the number of decompositions grows linearly with the number of variables" is stated as an empirical observation but supported by no data. This could be a genuinely interesting finding if substantiated.

---

### Choice of CAS (Section 3, inline)

The justification for Mathematica over Lean tactics and SMT solvers is reasonable and the comparison point about Z3's limitations with transcendentals is accurate. The demonstration that CVC5/MetiTarski fail on `log x ≤ log y ⟹ exp(x) ≤ exp(y)` is a concrete and useful data point — though a single example is not a systematic comparison.

The honest acknowledgment that Resolve does not produce verifiable proof objects is appreciated, but the implications are understated. For a tool claiming to assist "research mathematicians," the absence of a proof certificate is a significant limitation. The paper essentially asks users to trust Wolfram's implementation, which introduces an unverifiable oracle in what is presented as a rigorous verification pipeline.

---

### Related Work (Section 6)

The related work is thin. AlphaGeometry and Tao's Lean-based tool are discussed. The autoformalization paragraph is reasonable. Missing from the discussion:

- There is a substantial literature on *automated theorem proving for inequalities* (e.g., RAHD, QEPCAD, Polyrith in Lean/Mathlib, sum-of-squares methods) that is entirely absent.
- The LLM-guided proof search literature (e.g., Hypertree Proof Search, Draft-Sketch-Prove) is not discussed.
- Prior work on LLM+CAS combinations for mathematical reasoning beyond AlphaGeometry is not surveyed.

The "key differences" framing is appropriate but the coverage is too narrow.

---

### Limitations & Future Work (Section 7)

Credit to the authors for acknowledging the Resolve trust issue honestly. The summand simplification limitation is appropriately flagged.

However, several important limitations go unacknowledged:
- **Scope**: The approach is fundamentally limited to inequalities that (a) admit a finite domain decomposition and (b) are decidable by quantifier elimination after regime-wise simplification. Many important research-level asymptotic inequalities do not have this structure.
- **LLM reliability**: The paper mentions that LLM calls "only sporadically gave correct simplifications" (Section 3) but does not quantify this unreliability or discuss how the system behaves when the LLM proposes an incorrect decomposition.
- **Scalability**: No analysis of how the system scales with problem complexity, number of variables, or expression depth.
- **Soundness gap**: If the LLM proposes an incorrect decomposition that doesn't cover the full domain, and Resolve verifies each piece, the global proof is invalid. It is unclear whether the system checks that the proposed subdomains form a complete cover.

---

### Writing & Clarity

Beyond the already-noted incompleteness issues (empty placeholders, missing formulas, displaced paragraphs), Section 3's Case Study 1 walkthrough appears in the middle of the paper in an unexpected location — the mathematical discussion of `xy ≪ x log x + e^y` appears to be split across pages 3–5 non-contiguously. The overall structure is difficult to follow: the introduction already describes the algorithm and the case studies in some detail, making Sections 2 and 3 feel redundant.

---

### Overall Assessment

This paper presents a genuinely interesting idea — using LLMs to propose domain decompositions for asymptotic inequalities, then verifying each piece with Mathematica's Resolve function — and the core concept is well-motivated by a real pain point in analytic mathematics. However, the paper is **clearly unfinished** and falls substantially short of ICLR's standards across nearly every dimension. The implementation section contains unfilled template placeholders and incomplete code. The empirical evaluation reports no quantitative results. The two case studies are illustrative walkthroughs rather than experiments. The prompt — described as the key to reliable LLM performance — is never revealed. The paper cannot be reproduced from what is provided. Beyond completeness issues, there are substantive technical concerns: the scope of the approach is narrower than claimed, the "feedback loop" is never shown to exist, the soundness of domain cover completeness is not addressed, and the "research-level mathematics" framing substantially overstates two blog-post exercises. In its current form, this paper is not suitable for acceptance. The core idea merits development, but the paper requires complete reconstruction of its empirical evaluation, full disclosure of the prompting strategy, a rigorous characterization of the problem class addressed, and an honest recalibration of its claims.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces O-Forge, a neuro-symbolic pipeline that uses a frontier LLM to propose domain or series decompositions for asymptotic inequalities, followed by Mathematica’s `Resolve` function to rigorously verify the bound on each subdomain. By automating the divide-and-conquer strategy for proving $f \ll g$ estimates, the system aims to offload tedious verification work from research mathematicians. The authors illustrate feasibility using Tao-motivated examples and report qualitative performance on a small suite of simpler inequalities.

### Strengths
1. **Well-motivated application to a real research bottleneck:** Asymptotic estimation is a routine but time-intensive task in analysis, number theory, and TCS. The paper correctly identifies domain decomposition as the primary creative step and automated verification as the natural computational complement (Sec. 1, Sec. 3).
2. **Practical, accessible system design:** The modular LLM→CAS loop, combined with a web UI and CLI, directly addresses adoption barriers for non-technical mathematicians who cannot run local codebases (Sec. 1, Sec. 8).
3. **Effective use of a neuro-symbolic verification loop:** Offloading the "creative guess" to an LLM while delegating logical verification to a CAS aligns with proven paradigms (e.g., AlphaGeometry) and correctly mitigates LLM hallucination risks in high-stakes math (Sec. 2, Sec. 6).
4. **Demonstrates feasibility on non-trivial examples:** The pipeline successfully handles inequalities and series decompositions that standard SMT solvers (Z3, CVC5) and proof assistants (Lean tactics) struggle with, particularly due to transcendental functions like $\log$ and $\exp$ (Sec. 2, Sec. 6).

### Weaknesses
1. **Insufficient empirical evaluation for ICLR standards:** Section 5 reports tests on only ~40–50 "easier" problems plus two case studies, with no quantitative metrics (e.g., decomposition success rate, error rates, statistical distributions, baselines, or ablation studies). ICLR expects systematic, reproducible benchmarking.
2. **Heavy reliance on proprietary, unverifiable software:** The core verification depends entirely on Mathematica’s closed-source `Resolve`, which returns a boolean without proof certificates. This limits auditability, contradicts the ML community’s push for verifiable reasoning, and is acknowledged as a major limitation without mitigation (Sec. 2, Sec. 7).
3. **Minimal methodological or algorithmic novelty:** The framework consists of standard LLM prompting paired with an off-the-shelf CAS. There is no novel optimization, learning signal, structured reasoning technique, or theoretical analysis explaining *why* certain decompositions succeed or how to improve LLM proposal quality.
4. **Structural and academic writing issues:** Case studies are redundantly introduced in Sections 1 and 3. Technical justification relies heavily on blog posts and MathOverflow threads (e.g., Tao 2024, 2025a) rather than peer-reviewed or arXiv literature. The prompt template is left as placeholder tags `<guiding_principles>`, etc. (Sec. 4).

### Novelty & Significance
**Novelty:** Low-to-moderate. The LLM+solver/verifier paradigm is established in AI-for-Math literature; the paper applies it to a specific, underexplored niche (asymptotic $O(\cdot)$ analysis) without algorithmic innovation. **Clarity:** Moderate. The high-level idea is easily understood, but the manuscript suffers from redundant sections, informal citations, placeholder code/prompt text, and missing formal definitions for the "regime-wise simplification" module. **Reproducibility:** Limited. While code and a CLI are provided, the pipeline requires a commercial Mathematica license and proprietary LLM APIs. No exact model versions, temperatures, temperature seeds, dataset splits, or full prompt texts are disclosed, hindering independent replication. **Significance:** Niche-to-low for ICLR. The tool is practically useful for mathematicians, but the current submission lacks the algorithmic depth, rigorous evaluation, and open science practices typically required for main-track acceptance at a premier ML venue. It would be better suited for a focused AI-for-Math workshop in its present form.

### Suggestions for Improvement
1. **Conduct a standardized, quantitative evaluation:** Construct a public benchmark of asymptotic inequalities with ground-truth decompositions or expert annotations. Report metrics such as proposal success rate, CAS verification rate, failure modes, latency, and cost. Include ablations (e.g., zero-shot vs. few-shot LLM, different decomposition granularities, effect of the regime-wise simplification step).
2. **Audit or compare against open alternatives:** To reduce the closed-source trust dependency, run the same pipeline using open CAS backends (e.g., SymPy’s logic module, SageMath’s `qepcad`, Redlog/Reduced CAS) and report where they fail vs. `Resolve`. Alternatively, integrate a formal backend (e.g., Lean/Isabelle) and discuss the trade-offs in automation vs. certification rigor.
3. **Deepen the methodological contribution:** Investigate *how* to improve LLM decomposition quality. Examine prompt structures, self-consistency decoding, iterative refinement (feedback from CAS failures to guide resplitting), or lightweight fine-tuning/LoRA on math decomposition corpora. Characterize the theoretical conditions under which subdomain splitting guarantees tractability for quantifier elimination.
4. **Improve scholarly rigor and manuscript organization:** Consolidate the duplicated case studies, replace heavy reliance on informal web posts with peer-reviewed or preprint sources where appropriate, and provide the complete, exact prompt template and Mathematica invocation scripts in the appendix. Clearly define the "regime-wise simplification" algorithm and its mathematical validity conditions.
5. **Enhance reproducibility documentation:** Specify exact LLM model names, API versions, temperature/top-$p$ settings, random seeds, Mathematica version, Python environment, and exact dataset composition. Provide a Dockerized setup or fallback scripts that allow evaluation even when Mathematica is unavailable.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Quantitative success rates on the 40-50 problem suite are absent; without precision metrics on decomposition validity, the claim of effectiveness is unsubstantiated.
2. No baseline comparison against LLM-only proof generation or standard CAS workflows; without this, the additive value of the hybrid approach is unclear.
3. Experiments validating the "In-Context Symbolic Feedback loop" claimed in the abstract are missing, as Section 3 describes a single-shot prompt; an ablation is needed to confirm if iteration exists or improves performance.

### Deeper Analysis Needed (top 3-5 only)
1. Failure mode analysis is absent; specifically, does the system fail due to LLM hallucination of domains or CAS limitations, and how often?
2. The reliance on closed-source Mathematica undermines the "rigorous" claim; an analysis of verification confidence or comparison with open-source verifiers is needed to assess trustworthiness.
3. The paper claims research-level utility but only tests "easier problems" quantitatively; an analysis of complexity scaling (variables, transcendental functions) is needed to trust generalization.

### Visualizations & Case Studies
1. Visualizations of the proposed domain decompositions overlaid on the function surfaces would reveal whether the LLM identifies mathematically meaningful boundaries or arbitrary splits.
2. A detailed case study of a failed proof attempt is necessary to expose the system's limitations and error propagation between LLM and CAS.
3. A comparison visualization of proof time/steps between human experts and O-Forge would substantiate the claim of saving "lots of time and effort."

### Obvious Next Steps
1. The authors must resolve the direct contradiction between the claimed "feedback loop" in the abstract and the "single prompt" implementation described in Section 3.
2. The dataset of 40-50 problems must be publicly released with ground truth decompositions to enable independent verification of the claims.
3. Integration with an open-source proof checker (like Lean) should be explored or justified more rigorously than dismissing it based on limited tactic testing.

# Final Consolidated Review
## Summary

O-Forge is a neuro-symbolic tool that uses frontier LLMs to propose domain decompositions for asymptotic inequalities, then verifies each subdomain using Mathematica's Resolve function via quantifier elimination over the reals. The system aims to automate the tedious verification work that analysts and number theorists routinely face when proving $O(\cdot)$ estimates. The authors demonstrate feasibility on two case studies motivated by Terence Tao and report qualitative performance on a suite of 40-50 simpler inequalities.

## Strengths

- **Well-motivated application to a real mathematical bottleneck.** Asymptotic estimation is genuinely time-consuming in analysis, PDEs, and analytic number theory. The insight that "finding the right decomposition is the creative step; verification is mechanical" is mathematically sound and correctly identifies where LLM assistance adds value (Section 1, Section 3).

- **Novel application domain for neuro-symbolic methods.** While LLM+CAS combinations exist (e.g., AlphaGeometry), applying this paradigm to asymptotic inequalities—with their transcendental functions and domain decomposition challenges—addresses an underexplored niche that standard SMT solvers cannot handle well (Section 2, Section 6).

- **Practical, accessible system design.** The combination of CLI and web interface (o-forge.com) lowers adoption barriers for mathematicians who may not be comfortable with command-line tools or local installations (Section 1, Section 8).

- **Correct handling of transcendental functions.** The paper demonstrates that Mathematica's Resolve can verify inequalities involving $\log$ and $\exp$ that cause failures in Z3, CVC5, and MetiTarski—this is a legitimate technical advantage worth highlighting (Section 3, inline discussion).

## Weaknesses

- **Critical missing content: The prompt template is never disclosed.** Section 4 shows only empty XML placeholder tags (`<guiding_principles></guiding_principles>`, etc.) where the actual prompt should be. The paper explicitly claims that "a structured prompt" enables reliable LLM performance, but this prompt—the core intellectual contribution of the LLM component—is entirely absent. This makes the method unreproducible and unauditable. Page 2 also contains the placeholder text "(describe the structure of the prompt)" that should never appear in a submitted manuscript.

- **Empirical evaluation is insufficient for ICLR standards.** Section 5 reports testing on "around 40-50 easier problems" with zero quantitative metrics: no success rates, no failure analysis, no comparison baselines, no ablation studies, no runtime statistics. The vagueness of "around 40-50" itself signals a lack of rigor. The two case studies are mathematical walkthroughs, not experimental records—there is no transcript showing the LLM actually proposed the claimed decompositions, no number of attempts needed, no verification that Resolve returned True.

- **The claimed "feedback loop" does not exist in the described system.** The abstract promises an "In-Context Symbolic Feedback loop," but Section 2 and Section 3 describe a single-shot pipeline: one LLM call for decomposition, then CAS verification. The paper states: "we only prompt the LLM once in the entire process, and the rest of the proof completion is carried out by Mathematica." There is no iteration where CAS failure triggers re-prompting. This is a direct contradiction between claims and implementation.

- **No proof of completeness for domain covers.** If the LLM proposes subdomains that do not form a complete cover of the original domain, and Resolve verifies each piece, the global proof is invalid. The paper does not describe any mechanism to verify that proposed decompositions are exhaustive.

- **Reliance on closed-source verification without proof certificates.** Mathematica's Resolve returns a boolean but produces no verifiable proof object. For a tool claiming to provide "rigorous" verification, this introduces an untrustworthy oracle. The paper acknowledges this limitation but offers no mitigation or comparison with open alternatives (Section 7).

- **Case study formulas appear corrupted in the PDF.** The series formula in Case Study 2 is missing or garbled, making it impossible to evaluate that claim fully. While this may be a parsing artifact, it affects readability and verification.

## Nice-to-Haves

- **Quantitative benchmarking.** A public dataset of asymptotic inequalities with ground-truth decompositions, reporting success rate, failure modes, and comparison against baseline approaches (e.g., direct Resolve without decomposition, LLM-only proof attempts).

- **Failure mode analysis.** When the system fails, does it fail because the LLM proposes bad decompositions, or because Resolve times out on certain expressions? This would clarify the bottleneck.

- **Exploration of iterative refinement.** If the paper's "feedback loop" framing is intentional, implement and evaluate an actual loop where CAS failures guide LLM re-decomposition.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"No theoretical characterization of decidable problem class."** The paper could benefit from characterizing which inequalities the method handles, but this is outside the paper's stated scope of building a practical tool. Demanding formal complexity analysis is scope creep.

- **"No comparison with QEPCAD, RAHD, or sum-of-squares methods."** The paper discusses relevant prior work (AlphaGeometry, Tao's Lean tool, autoformalization). Demanding coverage of the full automated inequality proving literature is excessive for a paper focused on a specific application.

- **"Case studies are from blog posts, not peer-reviewed sources."** The Tao references are legitimate research discussions; this is common in emerging areas. The content matters more than venue prestige.

- **"Claims research-level mathematics but only tests easy problems."** The two main case studies ARE research-level examples from Tao's work. The 40-50 simpler problems are supplementary stress-testing, not the main contribution.

- **"Generic claim that paper is well-written or topic is important."** These are generic strengths that apply to many papers; the actual paper has significant writing issues (placeholders, redundancy).

## Novel Insights

The paper identifies a genuine asymmetry in mathematical work: domain decomposition for asymptotic inequalities requires creative "guessing" that LLMs are reasonably good at, while verification on each subdomain is mechanical and well-suited to symbolic systems. This division of labor is elegant. The observation that Resolve handles transcendental functions while SMT solvers fail is technically useful. However, the insight is undercut by the missing prompt template—if the decomposition quality depends entirely on prompting, that prompt IS the contribution, and withholding it prevents evaluation.

## Suggestions

1. **Release the complete prompt template.** The XML structure shown is empty; fill it in or provide it in an appendix. This is essential for reproducibility.

2. **Provide actual experimental evidence for the case studies.** Show LLM transcripts, Mathematica outputs, and runtime statistics. Demonstrate that the system found the decompositions it claims.

3. **Define and measure success quantitatively.** Report: How many of the 40-50 problems succeeded? How many required how many decomposition pieces? What fraction failed and why?

4. **Clarify or implement the feedback loop.** Either remove the "In-Context Symbolic Feedback loop" claim from the abstract or describe the actual iterative mechanism.

5. **Add a completeness check.** Verify that proposed subdomains cover the full original domain, and describe this mechanism in the paper.

6. **Fix the placeholder text.** Remove "(describe the structure of the prompt)" and other editorial artifacts before resubmission.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 0.0]
Average score: 0.5
Binary outcome: Reject
