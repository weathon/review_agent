# SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Community-driven Text-to-SQL evaluation platforms play a pivotal role in tracking the state of the art of Text-to-SQL performance. The reliability of the evaluation process is critical for driving progress in the field. Current evaluation methods are largely test-based, which involves comparing the execution results of a generated SQL query and a human-labeled ground-truth on a static test database. Such an evaluation is optimistic, as two queries can coincidentally produce the same output on the test database while actually being different. In this work, we propose a new alternative evaluation pipeline, called SpotIt, where a formal bounded equivalence verification engine actively searches for a database that differentiates the generated and ground-truth SQL queries. We develop techniques to extend existing verifiers to support a richer SQL subset relevant to Text-to-SQL. A performance evaluation of ten Text-to-SQL methods on the high-profile BIRD dataset suggests that test-based methods can often overlook differences between the generated query and the ground-truth. Further analysis of the verification results reveals a more complex picture of the current Text-to-SQL evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SPOTIT, a framework for improving the evaluation of Text-to-SQL systems via formal verification. Instead of relying solely on benchmark-provided test cases, SPOTIT aims to automatically generate counterexample databases that can distinguish between a generated SQL query and the corresponding gold (ground-truth) query.

SPOTIT extends VeriEQL (He et al., 2024) to handle a broader range of SQL constructs frequently used in real-world datasets such as BIRD, including string and date operations (strftime, JulianDay, ToStr, substr, LIKE, etc.). The system is applied to outputs from ten Text-to-SQL models, revealing that many incorrect or semantically mismatched queries pass traditional dataset-based evaluations but are correctly identified as incorrect by SPOTIT.

### Strengths
* **S1- Clear motivation and formulation**: The paper identifies a well-motivated gap in evaluating Text-to-SQL systems due to incomplete or non-discriminative test databases and formalizes the problem elegantly.
 
* **S2- Extension of formal verification**: Extending VeriEQL to cover a richer subset of SQL operators is a nontrivial and practically useful engineering contribution.
 
* **S3- Comprehensive evaluation**: Applying the verifier to ten different Text-to-SQL models on BIRD demonstrates empirical value.
 
* **S4- Readable and well-structured**: The paper includes illustrative examples that make the approach and findings easy to follow.

### Weaknesses
* **W1- Limited related work coverage**:
  The paper omits prior research on test data generation and SQL verification, including approaches based on constraint solving and fuzzing [a,b,c].
  
  * [a] Chandra et al. Data generation for testing and grading SQL queries. VLDB Journal, 2015.
  
  * [b] Somwase et al. Data Generation for Testing Complex Queries. arXiv, 2024.
  
  * [c] Zhong et al. Semantic Evaluation for Text-to-SQL with Distilled Test Suites. EMNLP, 2020.

* **W2- Context and positioning**:
  Related work is moved to the appendix, which weakens the contextual framing. A short related work section in the main text would help readers understand how SPOTIT builds upon and differs from existing formal or data-driven SQL evaluation approaches.

* **W3- Limited scope of expressivity**:
While the supported SQL subset has been expanded, the framework still cannot handle aggregation subtleties (e.g., nested subqueries, HAVING clauses, or complex joins) as far as can be inferred from Fig 2. Clarifying this limitation would improve transparency.
 
* **W4- Scalability concerns**:
The reliance on a SAT solver introduces potential scalability issues as query and schema complexity increase. The paper would benefit from runtime statistics or analysis of computational overhead.

### Questions
1. How is the value of K (the search bound) determined? Larger K improves completeness but increases runtime—what trade-offs were observed, and what values were used in experiments?
  
2. Which query classes cannot be tested using bounded equivalence? Are such queries present in benchmarks such as BIRD?
  
3. How does runtime scale with the number of tables, query complexity, and schema size?

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
5

### Summary
In the text to SQL systems, evaluation is an important component and the present evaluation methods are largely based on the execution of the generated SQL and ground truth on a static database. Such kind of evaluation may not lead to the best results as two queries can coincidentally produce the same output on the test database while actually being different. SPOTIT proposes a new alternative evaluation pipeline, where a formal bounded equivalence verification engine actively searches for a database that differentiates the generated and ground-truth SQL queries.

### Strengths
In text to SQL evaluation plays a vital role and this paper acknowledged the issue. Current methods relying on test-based execution on a static database are optimistic. The proposed solution, SPOTIT, replaces the weak test-based check with a much stronger, formal verification method. This also showed that the reported accuracy of these methods drops by 11.3%–14.2% when switching from the official test-based evaluation to SPOTIT

### Weaknesses
The sensitivity of the bound parameter (k) is not analyzed, leaving unclear how verification completeness and runtime scale with k. The paper does not report statistics on timeouts or unsupported queries.

### Questions
How is the experiment impacted on changing bound (K) and does increasingly reveal more counter examples.

### Soundness
3

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
3

### Summary
Traditional Text-to-SQL evaluation typically relies on comparing execution results. However, this can miss subtle semantic differences between the generated and gold SQL, as different queries may produce the same result on a particular database. SPOTIT addresses this limitation by introducing a formal verification-based evaluation method that uses an SMT solver to construct minimal test databases where two SQLs have different execution results. SpotIt supports richer SQL queries than existing methods. Existing text2SQL methods show a significant performance drop after employing SpotIt, highlighting the need for more rigorous SQL verification.

### Strengths
1. This paper tackles an important problem in Text-to-SQL, assessing SQL correctness beyond execution equivalence. By searching for minimal counterexample databases where two SQL queries diverge, SpotIt offers a more rigorous correctness check and greatly simplifies debugging (e.g., identifying faulty ground truths in BIRD).
2. The paper extends the SQL syntax that can be encoded into an SMT solver, which is a great contribution. It consistently covers more than 90 percent of the queries generated by various models.
3. I liked the error analysis the authors did on the BIRD benchmark regarding why the gold SQLs are incorrect. I’ve also personally encountered many erroneous ground truth SQL queries in BIRD, and it used to take me a lot of time to figure out why the reference SQL was wrong. With SpotIt, it would save researchers a significant amount of time identifying these issues.

### Weaknesses
1. SPOTIT only searches for divergences up to a fixed tuple-count bound K. Semantic differences that manifest only on larger databases (e.g., join multiplicity) may go undetected if no small counterexample exists.
2. As schema size or query complexity grows (many tables, join many tables together, deep nesting, large numbers of predicates), the SMT solver’s search space can explode, leading to longer solve times.
3. There are still some unsupported SQL features, such as window and analytics functions. I know those might be out of the scope of this paper, but it would be good to discuss them and give examples of SQL queries that can’t be supported.

### Questions
1. Did you run any experiments with varying K? How does the latency and accuracy change with increasing K? Would we discover more erroneous SQL queries if we increase K?
2. In practice, how do users choose the tuple-count bound K for counterexample search? Is there a systematic way to trade off between “smallness” and coverage of semantic differences?
3. How does SpotIt latency grow with SQL complexity?
4. On a more complex text2sql benchmark, such as Spider 2.0, could you estimate the percentage of SQL queries that can be covered by SpotIt? It's ok if the percentage is low because not everything can be encoded into SMT solver. But it would be nice to give readers more context because the community is shifting to more complicated benchmarks.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SpotIt, a verification-based evaluation pipeline for Text-to-SQL that replaces single-database execution tests with SMT-based bounded equivalence checking. Building on VeriEQL, the authors extend the supported SQL fragment with dates/strings, implicit type casts, set-semantics equivalence, and provide correctness proofs for new encodings. On BIRD-dev (1,533 items) across 10 top-performing systems, accuracy reported by standard EX-TEST drops by roughly 9.8–13.5% under SpotIt/SpotIt+, and rankings change substantially. Minimal counterexample databases expose three root causes for mismatches: incorrect gold SQL, ambiguous NL questions, and incorrect generated SQL. Manual audits suggest problematic gold SQLs are common; in some cases, ambiguity makes multiple SQLs “reasonable”. Algorithms for iterative bounds, validation against a real DBMS to filter spurious CEX, and cross-checking of counterexamples across systems are described. Runtime is practical (seconds per pair) with good coverage gains over baseline VeriEQL.

### Strengths
S1. Strong and timely problem framing; evaluation reliability is critical for Text-to-SQL progress.
S2. Clean pipeline with validation and cross-checking; minimal CEXs make root causes transparent.
S3. Substantive empirical findings (accuracy/ranking shifts; gold SQL issues; ambiguity prevalence).
S4. Formal treatment and proofs for extended string/date operators and set-semantics equivalence.
S5. Practicality: seconds-scale per-instance runtime; high coverage relative to prior verifier.

### Weaknesses
W1. Bound sensitivity (K): The paper does not quantify how many non-equivalences require bounds beyond K=5, so the false-negative rate is unknown. A detection-vs-K curve and examples missed at K=5 are needed to calibrate trust in results.
W2. Dataset scope: All results are on BIRD-dev, leaving external validity uncertain. Including Spider 2.0 and one enterprise-style schema would test generality and robustness of conclusions.
W3. Cause attribution (gold vs ambiguity vs model): The labeling relies on a small, non-adjudicated manual audit, which risks subjectivity. A larger, blinded, multi-annotator study with agreement stats would strengthen these claims.
W4. DBMS semantics alignment: The SMT encodings may not match engine-specific behaviors (NULLs, casts, date ops), but this isn’t systematically evaluated. Cross-engine validation (SQLite/MySQL/Postgres) and documenting divergences are necessary.
W5. Presentation quality: Frequent typos/formatting glitches and occasional terminology inconsistencies reduce clarity (e.g., “SQL ITE”, “hosptial”). A thorough proofreading and added navigational aids for dense formal sections would improve readability.

### Questions
1. How sensitive are SpotIt’s findings to the choice of K? Any empirical curve of detected non-equivalences vs K and the diminishing returns?
2. Did you observe cases where K=5 missed mismatches that appeared at higher K? Any characteristic patterns?
3. How many spurious CEXs did the validation filter out, and which operators were most responsible?
4. Can you report results on Spider 2.0 (or BIRD-Interact) to assess generalization beyond BIRD-dev?
5. How do encodings reconcile engine idiosyncrasies (e.g., string-to-int casting behavior, date boundaries)? Any cross-engine discrepancies observed?
6. For ambiguity cases, would adding light-weight interaction (à la BIRD-Interact) change pass/fail judgments under SpotIt?

### Soundness
3

### Presentation
2

### Contribution
3
