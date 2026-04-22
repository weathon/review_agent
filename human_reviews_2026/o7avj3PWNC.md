# BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians.
However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. 
To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review.
Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work finds that the LLMs are prone to hallucination and sophistry for theorem proving. Therefore, this work builds a benchmark named BrokenMath to evaluate the sycophantic behavior in LLMs in the natural language theorem proving.

### Strengths
* This problem is interesting. This work proposes to evaluate the hallucination and sycophancy, focusing on theorem proving.
* The dataset construction is clear. The figure 1 present the dataset construction pipeline.
* This work discuss several interesting observation, such as self-sycophancy.
* The experiment parts contain various models, suggesting that the sycophancy is common in current SOTA models.
* The prompt is given in Appendix to help reproduce the resutls.

### Weaknesses
* The evaluation uses LLM. Therefore, it is not clear whether the evaluation is correct
* Qwen3-4B sycophancy is even lower than Qwen3-235B and DS-V3.1. The question is why the smaller model has a lower sycophancy rate?
* Figure 7 presents that the model performance is significantly related to the prompt, which suggests that the evaluation variance may be large with different methods.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces BrokenMath, presented as "the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving." The benchmark contains 504 problems from 2025 mathematical competitions, where each problem is perturbed to produce a false statement. The authors evaluate 10 state-of-the-art LLMs and find that even the best model (GPT-5) exhibits "sycophantic" behavior 29% of the time. The paper investigates factors influencing this behavior (difficulty, problem type) and evaluates mitigation strategies including prompt engineering and fine-tuning.

### Strengths
1. **High-quality data curation**: The benchmark uses recent 2025 competition problems with expert verification, reducing contamination risks compared to prior work using GSM8k or AIME.

2. **Comprehensive evaluation**: Systematic assessment of 10 frontier models across multiple dimensions, providing valuable empirical comparisons.

3. **Fine-grained behavioral classification**: The four-category taxonomy (Ideal/Corrected/Detected/Sycophant) provides more nuanced information than binary correct/incorrect classification.

4. **Thorough experimental execution**: Careful validation of LLM-as-a-judge (95% agreement with human labels), detailed ablation studies, and multiple experimental settings.

### Weaknesses
### 1. **Fundamental construct validity failure: Confounding competence with alignment**

The paper claims to measure "sycophancy" (an alignment issue) but provides no mechanism to distinguish it from mathematical incompetence. True sycophancy requires that a model **can** judge a statement's validity but chooses not to due to user-pleasing tendencies. The paper's methodology cannot differentiate:

- **Phenomenon A**: Model accepts false statement because it cannot judge truth/falsity (competence deficit)
- **Phenomenon B**: Model can judge but suppresses critical thinking to please user (alignment sycophancy)

A rigorous operationalization would require:
```
For proposition P and its negation ¬P:
  Step 1: Filter to problems where model can correctly prove P
  Step 2: Further filter to problems where model can correctly disprove ¬P
  Step 3: Only on this filtered set, if the model still attempts to prove ¬P without questioning or refuting it, does this represent sycophancy.
```

The paper performs no such filtering, rendering all claims about "sycophancy as an alignment problem" unsubstantiated.

### 2. **Table 2 reveals that the measurement is fundamentally contaminated**

Table 2 presents "sycophantic behavior" split by whether models can solve the original problem:
- Solved problems: 21.5% (GPT-5 example)
- Unsolved problems: 47.7% (GPT-5 example)
- Gap: 26.2 percentage points

The existence of this table itself demonstrates the methodological failure. The "sycophancy" measurements on unsolved problems are **not measuring sycophancy at all**—they are measuring the model's inability to judge mathematical validity. This is pure competence deficit misclassified as alignment failure.

The solved/unsolved gap does not reveal "difficulty as an influencing factor." It reveals that the reported sycophancy rates are **contaminated metrics** mixing:
- True potential sycophancy (from solved subset, and even this requires further validation per Step 2 above)
- Competence limitations (from unsolved subset, which should not be in the dataset at all)

There should be no "All/Solved/Unsolved" breakdown because only the subset satisfying both Steps 1 and 2 should be included in the benchmark. The presence of unsolved problems in the evaluation fundamentally invalidates the sycophancy measurements.

### 3. **Cascading invalidation of all derivative analyses**

With the core measurement conflating competence and alignment, all subsequent analyses become uninterpretable within the paper's claimed framework:

- **Main results (§4.1)**: The reported rates (29%-70%) are inflated by competence limitations, cannot quantify alignment issues
- **Difficulty analysis (§4.2)**: The paper interprets the solved/unsolved gap as "higher difficulty implies higher sycophancy," but this correlation precisely demonstrates that the measurement captures the mixture of alignment and competence rather than only alignment
- **Problem type comparison (§4.2, Fig 4)**: Cannot distinguish whether proof-style vs. final-answer differences reflect difficulty (competence) or alignment dynamics  
- **Self-sycophancy (§4.3)**: Cannot determine if increased rates reflect consistency bias (competence-related) or alignment failures
- **Mitigation strategies (§5)**: The "modest" improvements are uninterpretable—are they trying to enhance critical reasoning ability or adjust alignment? The experimental design cannot answer this

### 4. **Mischaracterization of related work**

The paper states that prior works "typically modify existing final-answer problems... Results consistently show that frontier models are prone to sycophancy" (§2, citing Xue et al., Kirichenko et al., Liu et al., Sun et al., Ouyang, Rahman et al., Ma et al.). But indeed none of these seven papers use "sycophancy" as their core framework.

### 5. **Method is isomorphic to prior work despite claims of novelty**

Despite criticizing prior work for using "ill-posed questions" versus "well-posed but false" statements (§1), the perturbation approaches are structurally identical:
- Prior work: Perturb problems (remove constraints/add contradictions) → test detection
- This work: Perturb problems (change conclusions) → test detection

Both evaluate the same capability: detecting problematic mathematical inputs. The "ill-posed" vs. "well-posed but false" distinction does not constitute methodological innovation—both test critical reasoning.

### Questions
The main questions and concerns are detailed in the Weaknesses section above.

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
The paper introduces BROKENMATH, a benchmark to measure sycophancy - LLMs "going along" with incorrect prompts - in natural-language theorem proving. Authors build 504 problems by perturbing recent (2025) competition problems into false but plausible statements with LLM assistance and expert verification, then evaluate frontier models using an LLM-as-a-judge protocol. They find sycophancy is widespread and worse on harder/proof-style tasks; several mitigations (prompting, agentic variants, fine-tuning) help but don’t eliminate it.

### Strengths
1. Interesting and original approach that moves beyond final-answer math to proof-style tasks with verified false statements, addressing contamination and ill-posedness critiques of prior datasets.

2. Studies self-sycophancy and agentic sycophancy (best-of-n, iterative verification), expanding the phenomenon’s scope.

3. Careful comparisons across problem types and difficulties; shows sycophancy rises on problems a model cannot solve.

4. Empirically important finding: even top models are sycophantic a non-trivial fraction of the time (29% for GPT-5), especially on proofs, helping recalibrate expectations for theorem-proving deployments.

### Weaknesses
1. The main classification relies on an LLM-as-a-judge. Although a 95% agreement is claimed, the paper would benefit from a larger human-labeled audit, inter-annotator agreement, and/or error analysis (e.g., where the judge mistakes “Detected” vs “Corrected”). Also, the judge choice could correlate with family-level behaviors and subtly advantage similar model families. 

2. Perturbations are LLM-generated then expert-tuned; there’s a risk that models learn to spot stylistic artifacts of the perturbation procedure. 

3. Focuses on advanced high-school/undergrad level problems; unclear generalization to research-level math or to formal-proof ecosystems. Authors note this, but it constrains impact.

4. While the dataset is constructed to span algebra, geometry, combinatorics, and number theory, the results don’t report per-domain sycophancy/utility or domain-specific failure modes. Given well-known differences in LLM math behavior (e.g., geometry often needs diagrammatic or synthetic reasoning; number theory leans on modular arguments), a topic-level analysis could surface systematic vulnerabilities and make the benchmark more diagnostic.

### Questions
1. Can you report more details on the bracket/judge prompts for best-of-n, stopping criteria for iterative verification, and cost/latency, or did I miss them somewhere?

2. The mitigation evaluation depth can be potentially extended to cover broader prompt families.

### Soundness
2

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
4

### Summary
The paper introduces the first benchmark for evaluating sycophancy in natural-language theorem proving. The benchmark uses challenging 2025 math olympiad problems to minimize contamination, generating subtle false statements via an LLM and verifying them with experts. The dataset has 504 examples (321 proof, 183 final-answer). Evaluation of frontier models shows pervasive sycophancy, especially in proof tasks and with increasing difficulty. Existing mitigation techniques provide only limited benefit.

### Strengths
1. Important and timely focus on high-stakes alignment failure in mathematical reasoning.
2. Rigorous counterfactual construction: LLM perturbation + expert verification → plausible and difficult false theorems.
3. Clear exposition; strong motivation and methodology description.
4. Empirical results provide useful diagnostic insights: proof settings and harder tasks produce more sycophancy.

### Weaknesses
1. LLM-as-judge introduces circularity and evaluation risk; the judge may share the same biases.
2. Dataset size (504 samples) limits robustness and granularity in difficulty-stratified analysis.
3. Assumption of minimal contamination relies on recency; no quantitative verification.
4. Limited mechanistic analysis of how proofs fail (e.g., where sycophancy manifests in reasoning chains).

### Questions
1. Can you provide quantitative evidence (e.g., perplexity checks, memory probing) supporting minimal pre-training contamination?
2. What is the human vs. judge agreement rate on incorrect proofs where sycophancy is subtle?

### Soundness
3

### Presentation
3

### Contribution
3
