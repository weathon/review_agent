# VerifyThisBench: Generating Code, Specifications, and Proofs All at Once

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable progress in code generation, but many existing benchmarks are approaching saturation and offer little guarantee on the trustworthiness of the generated programs.
To improve visibility into model reasoning on formal correctness, we introduce $VerifyThisBench$, 
a new benchmark that evaluates end‑to‑end program generation and proof verification from natural language descriptions: models must (i) extract formal specifications, (ii) implement in a verification‑aware language, and (iii) construct machine‑checkable proofs.
Our evaluation reveals that even state-of-the-art (SOTA) models, such as o3-mini, achieve a pass rate of less than 4\%, with many outputs failing to compile. To isolate sources of difficulty, we further propose $VerifyThisBenchXS$, a relaxed variant in which partial implementations or proofs are provided.  Across nine models and seven verification tools on both benchmarks, we observe consistent gains with feedback‑driven refinement, but overall pass rates remain low, underscoring substantial gaps in formal reasoning. We release the benchmark and the unified evaluation environment to catalyze the verification capabilities for future models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VerifyThisBench, a challenging benchmark for evaluating LLMs' ability to perform end-to-end program verification from natural language descriptions. The benchmark requires models to simultaneously generate formal specifications, implementations in verification-aware languages, and machine-checkable correctness proofs.

### Strengths
- A good benchmark evaluating end-to-end verification from natural language across multiple frameworks
- Competition-grade tasks (up to 648 lines) are more challenging than prior work (Verina: 225 lines max)
- The benchmark spans seven verification tools (Dafny, Why3, VeriFast, VerCors, Frama-C, Verus, CBMC) across multiple programming paradigms and languages, providing broad assessment of verification capabilities

### Weaknesses
- No soundness analysis: The paper provides no detection or analysis of whether "passing" solutions achieve verification through unsound means: inserting `assume false` or `#[verifier::external_body]`
- No benchmark contamination analysis: The benchmark uses publicly available VerifyThis competition problems and solutions (2011-2024), all accessible online before model training cutoffs.
- Backwards spec-gen task design: In spec-gen tasks, models receive implementation + complete proof but must generate specifications. This contradicts realistic verification workflows where specs come first, and critically, the provided proof leaks what the specification should be since proofs explicitly reference specification properties.

### Questions
- In spec-gen tasks, why do you provide the proof to models when asking them to generate specifications? Doesn't the proof leak what the spec should be? In practice, the spec should be designed first and then write the proofs.
- How many "passing" solutions use unsound verification shortcuts like `assume false`, overly broad assumptions, or unsafe functions?
Have you implemented automated detection for common unsound patterns?
- Have you analyzed whether VerifyThis competition problems appear in model training data?
- Why does o3-mini, your best model on VerifyThisBench (9.37%), perform worse on the supposedly easier VerifyThisBenchXS (8.28%)?
- What percentage of your 154 benchmark tasks have verified reference solutions? Please report verification rates for all ground-truth solutions.
- How many of the "passed" solutions have specifications that are verifiable but semantically wrong (e.g., proving a weaker property than intended)? What is the performance of LLM-as-a-judge in this task?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a benchmark set for generating fully verified code from a natural language prompt. The benchmarks are based on tasks drawn from the long-running VerifyThis Competition Series. The paper evaluates a range of existing models on the benchmarks for a wide range of verification frameworks and shows that the benchmarks are very challenging.

### Strengths
- Given the increasing usage of LLM in writing code, evaluating their ability to write verifiably correct code is a timely topic.
- I find it a very nice and clever idea to use the tasks described in VerifyThis competition. This choice helps to focus on tasks that existing verification tools are more likely to be able to handle. 
- The evaluation considers a wide range of existing models and is therefore quite thorough.

### Weaknesses
- My main issue with the benchmark is that the success rate is not a very meaningful metric. If I understand correctly, the LLM can achieve 100% success rate by generating a trivial and fully verified program that is completely irrelevant to the natural language prompt. Certainly, LLMs are trained to follow instructions. However, the evaluation metric is too easy to game. Imagine having a leaderboard based on success rate for VerifyThisBench, it would be really unclear whether top performers are simply generating easily verifiable programs that do not (fully) follow the natural language instructions, or whether they are actually performing well. 
- The author did try to mitigate the issue above by proposing a coherence check based on LLM self-assessing its generated artifact. Self-assessed coherence is something that can be easily gamed; the model is incentivized to just lie. So in short, one can easily get 100% success rate and 100% self-assessed coherence by generating a trivial and verified program and lying about the coherence. As a developer of practical LLMs, this is not good practice. However, a benchmarking framework should be designed to have a level of robustness against these gaming strategies. 
- There are different potential ways to mitigate the problems above. 1) Instead of self-assessment, perhaps assess using an independent LLM or even employ mutual assessment; 2) one could associate each benchmark with a set of test cases; 3) This would be much harder, but ideally, the evaluation could perform some equivalence checking between the generated code and the ground truth, or the generated specification and the ground specification. 1) and 2) are both not ideal. 2) is probably better than 1). I acknowledge that the paper did explore 2) to a limited degree, but it does not apply it to all benchmarks. If something along the line of 3) can be achieved, then I believe the benchmark set would differentiate itself significantly from concurrent work and consitute a solid evaluation framework.

### Questions
N/A

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VerifyThisBench, benchmark for evaluating LLM coders on end-to-end formal program verification. The benchmark is derived from the VerifyThis competition series, and includes generating code, formal specifications, and machine-checkable proofs simultaneously from natural language descriptions. The authors evaluate nine models (including both closed frontier models and leading open models) across seven verification frameworks (Dafny, Verus, Frama-C, etc.) on over 700 tasks. The results show that performance is quite low, although iterative improvement seems to offer potential for significant gains.  To isolate sources of difficulty, the authors introduce VerifyThisBenchXS, a relaxed variant providing partial implementations or proofs, where performance improves moderately but still demonstrates substantial gaps.

### Strengths
The paper's core strength is its novel and rigorous benchmark, VerifyThisBench, which addresses a critical gap in LLM evaluation by moving beyond simple functional testing.

-- Unlike most benchmarks, it requires models to generate specifications, code, and machine-checkable proofs from only a natural language prompt. 

-- It introduces a unified environment for seven distinct verification tools (e.g., Dafny, Verus, Frama-C), a significant step up from single-framework benchmarks like Verina and Clever.

-- The sourcing is from the VerifyThis competition, which is arguably more "real-world" than previous benchmarks in this space.

-- I appreciated the design of VerifyThisBenchXS, and the analysis in isolating specific failure modes.

### Weaknesses
The paper's core premise is to evaluate verification from natural language, but doesn't really directly measure this goal.  The main quantitative experiments check if the generated proof is valid for the generated specification.  The closest result in the main paper is in Section 4.5, where "coherence" is measured (whether the LLM believes the generated spec matches the natural language description).  Thus, I believe the abstract & intro need to be somewhat repositioned, or is otherwise a bit misleading in this respect.

The extremely low success rates on this benchmark can also need to noise, where "randomly" getting a few extra problems correct can lead to huge relative gains in improvement.  Worth doing a more detailed analysis on variance/reproducibility of results.  The common approach is to run the same setup several times and observe the variance.  I realize this is expensive, but could be worth doing for a few LLMs.

### Questions
No further questions, other than commenting on my weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
