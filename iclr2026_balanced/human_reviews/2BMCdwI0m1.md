## Human Reviewer 1

### Summary
This paper introduces Post-SQLFix, a neuro-symbolic framework replacing ambiguous Execution Feedback for Text-to-SQL. It formally analyzes queries via a Canonical Query Structure , synthesizes verifiable repair plans , and significantly boosts execution accuracy.

### Strengths
S1: This paper addresses the core issue of Text-to-SQL: the "diagnostic poverty" flaw in execution feedback (EF), and the "arbitration paradox" of LLM self-correction. The proposed paradigm shift from "fuzzy feedback" to "verifiable fixes" is highly original and significant, offering a new perspective for building reliable Text-to-SQL systems.

S2: The methodology in the paper is rigorous and clear. Drawing from compiler theory, it constructs a robust neural-symbolic framework. The design of converting SQL into a dialect-agnostic "Canonical Query Structure" (CQS) as an intermediate representation (IR) is an effective approach. On this basis, the design of a hierarchical validation engine (α, β, γ) with a strict partial order (α < β < γ) is logically sound and effectively prevents "cascading false positives." Detailed formal definitions and soundness proofs (Appendix B) greatly enhance the quality of the work.

S3: Another major innovation in this paper is the redefinition of the role of LLMs: from an unreliable "arbiter" to a "constrained agent." Diagnosis and repair plan synthesis are handled by the symbolic engine (providing "what" and "how"), while the LLM is responsible for selecting from the valid plans using its semantic understanding (providing "which"). This neural-symbolic collaboration is outstanding.

S4: Experimental results are convincing. Post-SQLFix significantly improves accuracy on the BIRD and Spider benchmarks (up to +11.6%) and achieves far better repair efficiency (50% fewer iterations) compared to the EF baseline. The multi-dialect analysis (Table 9) is particularly compelling, exposing the vulnerabilities of the EF paradigm in handling errors across different dialects, which contrasts with the robustness of CQS and formalized diagnosis.

### Weaknesses
W1: The limitation of the framework is that it mainly addresses "formally incorrect" queries but struggles with "semantically valid but logically erroneous" queries (e.g., incorrect joins, aggregation scope). The paper admits that EF provides "no signal" for such errors. The proposed "empty result alignment" strategy is a narrow heuristic, and does not address a broader range of logical bugs. Although "intent completeness" is undecidable (Appendix B.4.3), the gap between "formally correct" and "intentionally correct" remains significant.

W2: The complexity of building the core "strategy map M" in the "repair synthesis" module seems to be underestimated. The paper should clarify how M is constructed (e.g., is it manually defined based on NL2SQL-BUGs?). The claim that M is a "total function" to ensure "repair completeness" may imply an extremely large and difficult-to-maintain engineering task, particularly when covering all SQL errors and dialects.

W3: The setup of the ablation study (Table 8) is not sufficiently clear. Removing "Plan Synthesis" leads to the largest performance drop (~3%), which suggests that the LLM only receives diagnosis (V_abs) and is forced to perform "speculative" repairs. The meaning of "w/o Hierarchical Validation" is also unclear (does it refer to flattening the validation?). Since the paper emphasizes that hierarchical validation prevents "cascading false positives," providing a specific failure case would make the ablation study more convincing.

W4: CQS is claimed to be "dialect-agnostic," but this is misleading. Validation (especially Layer γ) and repair synthesis are clearly "dialect-aware," relying on the "dialect specification Δ." The paper should clarify the engineering effort required to support a new dialect (e.g., MS SQL Server or Oracle), which is closely related to the maintenance cost of the "strategy map M" mentioned in W2.

### Questions
Q1: (Related to W1) Given that the current framework primarily addresses formal errors, what are the plans for handling "semantically valid but logically erroneous" queries (i.e., logical bugs)? The "empty result alignment" strategy is a narrow heuristic—could your formal synthesis framework be extended to diagnose and fix a wider range of logical errors?

Q2: (Related to W2/W4) For queries with multiple complex violations, does the set of repair plans {Λ} generated during "repair synthesis" face combinatorial explosion? How does the framework manage this complexity (e.g., pruning)? Or does the LLM need to handle a large "repair specification" (S_R) containing many candidate plans?

Q3: (Related to Appendix D.2) The residual error analysis attributes failures to the "engineering challenges" of fixing complex SQL (e.g., nested, recursive). Does this specifically refer to the incompleteness of the "strategy map M" for such cases (i.e., unable to synthesize plans), or does it indicate that the symbolic engine generated valid plans, but the LLM failed to successfully "follow instructions" for complex code transformations?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper proposes Post-SQLFix, a neuro-symbolic framework designed for verifiable SQL repair in text-to-SQL tasks. The core idea is to replace ambiguous execution feedback—which often provides low-level, dialect-dependent error information—with a formal synthesis-based repair mechanism. The framework introduces a dialect-independent canonical query structure (CQS) as an intermediate representation, performs layered validation (including syntax, context-free, and context-sensitive semantic checks), and formulates the repair problem as a constraint synthesis task, generating a set of formally valid repair candidates. The LLM then performs constrained generation based on these verified candidates. Experiments on the BIRD and Spider datasets demonstrate an 11.6% improvement in execution accuracy and a 50% reduction in repair iterations compared to EF baselines. The authors also provide proofs of reliability for diagnosis and repair, along with ablation studies and efficiency analysis.

### Strengths
1. The paper correctly identifies a major limitation in current text-to-SQL pipelines: the reliance on ambiguous execution feedback that provides low-level, semantically sparse information. Addressing this bottleneck is both timely and significant for the field.

2. The proposed Canonical Query Structure (CQS) and the hierarchical validation layers (α, β, γ) are systematically designed. Formulating the repair synthesis as a constraint satisfaction problem demonstrates formal rigor and represents a well-constructed framework.

### Weaknesses
1. The formal verification and repair synthesis rely heavily on explicit schema checking and rule-based validation. The scalability of this approach remains unclear when applied to large or noisy real-world schemas, complex subqueries, or non-standard SQL constructs.

2. While the quantitative accuracy is reported in the tables, the paper would benefit from deeper qualitative examples demonstrating how Post-SQLFix resolves specific semantic errors more effectively than EF or LLM reflection. Concrete before-and-after SQL repair examples would help readers better understand its advantages.

3. The "formal verification + constrained repair" concept closely resembles compiler-inspired static analysis pipelines, with the primary innovation lying in its integration with LLMs—which feels incremental. While the authors extensively formalize the symbolic component, the role of the "neural agent" remains underspecified.

### Questions
1. How does the proposed method handle semantically incorrect yet executable queries that return non-empty but erroneous results?

2. How robust is the system to LLM instruction non-compliance? What safeguards exist if the LLM disregards the formal repair constraints and generates a syntactically valid but semantically divergent query?

3. To what extent can this framework be generalized to other code generation tasks beyond text-to-SQL?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper targets the problem that SQL queries generated by LLMs often contain formal errors and rely on vague "Execution Feedback" (EF) for repairs. It proposes the Post-SQLFix neuro-symbolic framework. This method uses a "Canonical Query Structure" (CQS) to perform a formal diagnosis of the query and employs "constrained synthesis" to generate a guaranteed-correct repair plan. This transforms the LLM from a guesser into a constrained executor, significantly improving accuracy by up to 11.6% on BIRD and Spider and reducing repair iterations by 50%.

### Strengths
1. The paper's greatest contribution is proposing a new paradigm for Text-to-SQL repair. It successfully transforms the uncertain, LLM-guess-dependent "execution feedback" loop into a deterministic, formal "diagnose-synthesize-execute" workflow.
2. The core innovation lies in the "constrained synthesis" of a guaranteed-valid repair plan space. This fundamentally changes the LLM's role in SQL repair and reduces the risk of introducing new errors during the process.
3. The framework's improvements in accuracy and reduction in iteration costs are quite significant. In particular, the results on multi-dialect benchmarks demonstrate that this method, when handling dialect-specific errors, is superior to EF methods that rely on the LLM to interpret different error messages.

### Weaknesses
1. The framework only guarantees "formal correctness," not "intent correctness." A SQL query might fully comply with syntax, type, and schema constraints but be completely misaligned with the user's intent due to faulty join logic. Post-SQLFix does not seem capable of detecting or fixing such errors.
2. The paper also mentions that residual errors mainly come from "complex SQL structures" (e.g., deep nesting, recursion) because their repair logic is "combinatorially explosive." This indicates that "constrained synthesis" faces complexity challenges in practice, and its "repair completeness" is bounded.
3. The paper's writing when introducing the overall workflow is somewhat overly obscure, with a suspicion of intentionally piling on formulas.

### Questions
1. There are actually many trained SQL generation models for the Bird-SQL dataset. Do these types of models exhibit a phenomenon where formal correctness capability increases while intent correctness capability decreases? In such cases, would the Post-SQLFix method still be effective? The paper only uses two sizes of the Qwen-coder model, making it somewhat difficult to prove the method's generalizability.
2. The paper mentions the "combinatorial explosion" problem for complex repairs. Does this mean the method is powerless against errors in cases like multi-level nesting or complex functions? Can you analyze where the practical capability boundary of "constrained synthesis" lies?
3. When handling empty results, why not continue using the neuro-symbolic collaborative approach (e.g., having the symbolic engine generate candidate repairs and then having the LLM judge the intent)? Why does it instead revert to a purely deterministic, restricted rule?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a neurosymbolic repair for SQL that is applied to incorrect SQL
generated from NL using LLMs.

The popular approach here is self-debugging, where if the generated SQL throws error
on execution, the errors are presented to the LLM and the LLM is asked to repair its
incorrect generation conditioned on the errors. This "execution feedback" (EF) does not
guarantee that errors will be fixed, and it could get into multistep feedback loops.

In this work, the authors build a symbolic engine that analyzes an incorrect SQL 
and identifies the violations. These violations are then mapped to potential repairs
(schemas of repairs, not the concrete repair). The LLM is now presented the findings
of the symbolic analyzer, and the LLM suggests repairs, and the one that does not throw
any errors on execution is finally accepted. This post processor is called Post-SQLFix.

The paper shows that Post-SQLFix performs better than EF and even uses fewer iterations
than EF. The paper also shows that Post-SQLFix can also be used to improve SQL generation 
from any model since it can be plugged to the output of any model. This is shown on two
datasets, BIRD and Spider.

### Strengths
Strengths:
1. The idea of using a symbolic analyzer to guide the repair is a powerful paradigm,
and the authors successfully demonstrated that it works for Text-to-SQL.
2. The experiments are fairly exhaustive and provide good evidence for the main claims
of the paper.

### Weaknesses
Weaknesses:
1. The writing is rushed, which makes the paper a confusing read at many places.
2. The symbolic parts of the Post-SQLFix are not entirely trivial and may have
required a lot of effort. Most of the insights are specific to SQL. 
Question: How much work was it to get all the symbolic pieces together, and what learnings could 
one take from here when working on another domain?

Details:
1. Line 262-263 says "we provide a detailed instantiation of this mapping for key violation
types in Appendix C.4" -- but there is no such thing in Appendix C.4
2. Section 3.3.2 is really hard to follow due to a lack of details. Who composes the violations
into candidate repair plans? Is it a symbolic procedure or LLM? If it is an LLM, then why is
it a repair plan and not just a repair? 
3. Theorem 1 mentions "our synthesis function" -- it is not clear what this is. Is this Algorithm 1?
But Algorithm 1 is not called "synthesis". Please try to be rigorous when writing 
statements of theorems. In the next section, "Synthesize" function is not even generating a CQS.
4. Definition 5: Last line, should = be op? If not, then what is the role of "op" ?
5. Table 5, does bold mean "better"? If so, then Time for EF is better than time for "Our" and yet
time for "Our" is in bold.
6. The distinction between "Universal Efficacy and Integration" and "Plug-and-Play" is unclear,
and I can't even guess what the difference might be in this case. Line 402 "By intercepting and resolving
errors early" hints that I am missing some important detail.
7. Line 400 mentions 54.9% vs 53.1%  -- is this statistically significant? What was the variance 
observed in the 5 runs?
8. Line 460: "the limitation of EF is architectural, not informational": what does this mean?

The authors may wish to see some related work in [Poesia ICLR 2022], [Bavishi OOPSLA 2022],
and https://www.arxiv.org/abs/2508.09324  that all share some of the same high-level philosophy
as this work.

Other typos/errors:
l26: EF in abstract is undefined
l67: "correctness-intent tradeoff" : Why would this be a trade-off, they are not mutually 
adversarial.
l130: "In contrast,regardless of their architectural complexity, converge on a different
fundamental dependency" -- What does that mean?
l229: Equation (3) - Is the negation needed there? 
l262: "a Algorithm 1 first identifies the set of violations V" -- incomplete sentence.
l266: "Only those plans produce a verifiably valid successor state are included in the final output." --
unclear what this means.
l269: "for any error our identifies" -- missing word.
l320: "We seeks to ..."

### Questions
See the Weaknesses section above.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4