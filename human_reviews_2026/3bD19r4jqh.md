# FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Recent advances in large language models (LLMs) have demonstrated impressive capabilities in formal theorem proving, particularly on contest-based mathematical benchmarks like the IMO. However, these contests do not reflect the depth, breadth, and abstraction of modern mathematical research. To bridge this gap, we introduce **FATE**, a new benchmark series in formal algebra designed to chart a course toward advanced mathematical reasoning. We present two new components, FATE-H and FATE-X, each with 100 problems in abstract and commutative algebra. The FATE series spans a difficulty spectrum from undergraduate exercises to problems exceeding PhD qualifying exams. Notably, FATE-X is the first formal benchmark to surpass both PhD-level exam difficulty and the coverage of the Mathlib library. Our evaluations of state-of-the-art LLM provers on this new benchmark reveal a stark performance gap compared to contest math: the best model achieves only 3\% (pass@64) accuracy on FATE-H and 0\% on FATE-X. Our two-stage evaluation reveals that models' natural-language reasoning is notably more accurate than their ability to formalize this reasoning. We systematically classify the common errors that arise during this formalization process. Furthermore, a comparative study shows that a specialized prover can exhibit less effective reflection than general-purpose models, reducing its accuracy at the natural-language stage. We believe FATE provides a robust and challenging benchmark that establishes essential checkpoints on the path toward research-level formal mathematical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce FATE: a set of three benchmarks consisting of problems in abstract and commutative algebra:
- FATE-M: 150 undergrad-level problems (published previously, here extended by 9 new problems),
- FATE-H: 100 graduate-level problems,
- FATE-X: PhD-qualifying problems.

The problems are presented in LaTeX and in a formal math language (Lean).

A set of various LLMs are evaluated on the benchmark (including models RL-tuned for formal theorem proving in Lean).

The results are analyzed, which includes interesting qualitative study and discussion.

### Strengths
1. There are not enough formal math benchmarks, especially those targeting research-level math. Therefore, benchmarks like FATE constitute valuable contribution for the research community of AI for formal math.
2. The benchmark seems to be carefully designed and build, involving mathematicians and Lean experts -- this is important since it is relatively easy to introduce mistakes in the formal statements.
3. The analysis of the evaluation results are informative, interesting, and quite deep.

### Weaknesses
1. The authors do not experiment with any more sophisticated inference strategies and jus use a fixed prompt and temperature to sample 64 times. It would be interesting to see, e.g., what performance can be achieved when the model receives Lean feedback for proof repair.
2. It would be good to see more examples of problems from the benchmark to get better feeling what is included.
3. It would be good to additionally see the performance of models from the Qwen 3 series.
4. The authors do not specify what mathlib version they use. Different LLMs could be trained for different versions of mathlib and therefore some number of formalization mistakes may be cause by using a different version of mathlib.


**Minor**
* Figure 1 (a) should perhaps simply be a bar plot instead of using smoothed lines.
* Table 2: Prover-V2 --> Goedel-Prover-V2

### Questions
1. What version of mathlib do you use for evaluation?
2. Did you experiment with the "decoupled" approach you suggest where a general model is used to generate informal reasoning, and another, Lean-specific model is used to turn the reasoning into a formal proof?
3. Did you experiment with using Lean feedback for proof repair?

### Soundness
3

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
2

### Summary
The paper lies in automated formal theorem proving with large language models (LLMs), evaluated in Lean. The authors argue that current contest-style or introductory benchmarks do not reflect research-level mathematics, and they propose a graded benchmark in abstract and commutative algebra—FATE-M/H/X—to measure progress from undergraduate to beyond PhD qualifying exam difficulty. The core question is whether we can reliably measure and drive progress toward research-level formal reasoning, and whether current systems can formalize such mathematics. The paper builds a benchmark series and a two-stage evaluation protocol to answer this. It shows very low pass rates at higher difficulty and analyzes the bottleneck between natural-language reasoning and Lean formalization.

### Strengths
The paper tackles an important gap: modern, abstract algebra at graduate and post-graduate levels, formalized in Lean. The graded design and the curation pipeline are carefully described, with inputs from expert mathematicians and Mathlib contributors, improving both difficulty control and content quality. The two-stage analysis is informative and supports a clear technical takeaway: current systems often find a plausible informal argument but fail to turn it into correct Lean code, mainly due to library hallucinations and gaps in Lean skill. The study also includes a model comparison that proposes a reasonable training direction (decouple NL proof from autoformalization). Together, these aspects make the benchmark useful to the community.

### Weaknesses
First, the claim that FATE-X is the first formal benchmark exceeding PhD qualifying exam difficulty and surpassing Mathlib coverage is important. Although the paper provides expert survey results and internal statistics, the case would be stronger with an external audit or a direct side-by-side with other recent research-level testbeds (e.g., RealMath or FrontierMath) in terms of formalization coverage and difficulty distribution.  

Second, the baseline suite is good, but the paper’s main scientific message points to a decoupled pipeline. It would be helpful to include at least one concrete baseline that actually implements “NL proof → autoformalizer,” so that the analysis connects to an end-to-end number under the proposed design. Right now, the ablations argue for decoupling, but the paper does not show how far a simple, engineered two-module system can go on H or X.

Third, library hallucinations are frequent. It would help to instrument and report “library hits” versus “misses” more explicitly: for each formal step, how often is a cited lemma present, and how often is a name or signature slightly off. A standardized “lemma grounding rate” could become a trackable metric across future systems. The current table is informative but high-level. 

Fourth, compute and sampling fairness could be clarified. The paper uses pass@64 and long contexts; it would help to normalize sampling budgets across models of very different sizes and search strategies, and to document Lean kernel timeouts and caching settings in a machine-readable form to improve reproducibility beyond high-level text. 

Fifth, to support the claim about coverage beyond Mathlib, a compact map of “new definitions introduced per problem” in FATE-X, with tags for the minimal Mathlib extensions that would make them solvable, would be useful for method developers. Right now we are told that new definitions are supplied in X, but a table of these would help users plan retrieval or tool-use strategies. 

Sixth, reproducibility of the natural-language grading is described, but since this evaluation underpins the main bottleneck claim, adding a small public “adjudication set” with anonymized model outputs and gold adjudications would help outside groups reproduce the gap. The appendices outline the team and process; releasing an artifact would make the result stronger. 

Finally, the comparative section reports interesting behaviors of a specialized prover, including “cheating” patterns. Given the sensitivity of such claims, it would help to provide a small gallery of redacted transcripts and a measurable definition of the behaviors flagged.

### Questions
See Weakness

### Soundness
3

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
3

### Summary
This paper introduces FATE, a benchmark suite designed to evaluate the frontier of automated formal theorem proving in algebra. It extends previous work (FATE-M) with two new datasets: FATE-H, representing graduate-level difficulty, and FATE-X, targeting or exceeding PhD qualifying-exam level. Each contains 100 formalized problems in abstract and commutative algebra, curated and verified by experts using Lean. The authors evaluate a range of state-of-the-art LLM provers, finding drastic performance drops from FATE-M to FATE-H and FATE-X. Through a two-stage analysis, they show that the main bottleneck lies not in mathematical reasoning per se, but in translating correct natural-language proofs into formal Lean code. They further classify common formalization errors and compare specialized vs. general reasoning models, highlighting that general models exhibit stronger “reflection” capabilities.

### Strengths
Originality: The paper fills a major gap between contest-level and research-level mathematical benchmarks. The design of a progressive benchmark series is conceptually novel and important for longitudinal evaluation of reasoning systems. The detailed error taxonomy (Mathlib hallucination, Lean proficiency, misalignment, etc.) provides a structured lens for analyzing formalization failures.
Quality: The benchmark construction is rigorous: problems collected, curated, and formalized through expert workshops, with verification by Lean specialists.
Clarity: The paper is well organized, with clear motivation, design rationale, and transparent methodology.
Significance: FATE establishes the first formal benchmark exceeding PhD exam difficulty, thus setting a new standard for evaluating research-level formal reasoning.

### Weaknesses
Lack of Actionable Insight from Core Finding: The central finding is that models fail at the translation/formalization stage. However, the paper offers little new, actionable mechanism or technique to address this challenge. It merely describes the phenomenon, classifies the symptoms (hallucinations, proficiency issues), and proposes a decoupled architecture as a future direction.

Limited Scope for Deeper Behavioral Analysis (General vs. Prover): The argument for the general model's superior "effective reflection" is compelling, but the empirical evidence for this key hypothesis relies heavily on comparing only the DeepSeek series of models. While DeepSeek-R1 significantly outperforms DeepSeek-Prover-V2 in natural language accuracy, this could be an artifact of the specific architecture/training of this lineage rather than a fundamental truth about general versus specialized models. A comparison with more diverse models, such as other specialized provers at the natural language stage (if feasible to extract for them), would significantly strengthen the claim that the general/specialized distinction is the source of the difference in "effective reflection.

While expert surveys are conducted, more quantitative evidence (e.g., human success rates on FATE-H/X problems) could help validate the claimed difficulty scaling.

### Questions
1. The paper mentions that abstract and commutative algebra is suitable for testing research-level reasoning due to its abstract and self-contained nature. Could the authors elaborate on whether they considered other fields for a frontier benchmark (e.g., topology, functional analysis) and what made algebra uniquely suitable as the first domain to push past the PhD-level frontier?
2. In Table 2, the natural language (NL) accuracy is only reported for DeepSeek-R1 and DeepSeek-Prover-V219. Given the compelling finding regarding "effective reflection" (Section 4.5), obtaining NL accuracy for Goedel-Prover-V2-32B or Kimina-Prover-72B would be highly valuable to generalize the claim about specialized provers. Could the authors confirm whether these models also generate an explicit natural language proof (or Chain-of-Thought) that could be manually evaluated, and if so, provide those additional data points in the final version?
3. Have you measured approximate human success rates or time to solve for graduate students or PhD candidates? This would ground the “difficulty” claim quantitatively.

### Soundness
3

### Presentation
2

### Contribution
2
