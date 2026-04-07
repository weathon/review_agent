=== CALIBRATION EXAMPLE 71 ===

# Harsh Critic Review
Now I have enough of the paper to write a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the contribution. The abstract correctly summarizes the three main positive results (O(1)-Coreset+ for general metrics, for bounded-doubling metrics, and an ε-Coreset+ refinement for k-median). However, the abstract does not foreground a crucial caveat: the algorithms do *not* output exactly k centers but rather O(k·polylog(n)) centers. Since this is a significant departure from the standard clustering problem specification, it should be stated clearly in the abstract rather than deferred to Section 1.1. The LLM motivation in the abstract ("demonstrates how noisy, low-cost oracles...can be systematically integrated") is not validated experimentally at all; no LLM oracle is tested.

---

### Introduction & Motivation

The introduction cleanly situates the work relative to Addanki et al. (2021) and the RM-model results of Galhotra et al. (2024) and Raychaudhury et al. (2025). The impossibility argument for o(n)-approximation with k centers (motivating the relaxation to O(k polylog n) centers) is well-framed.

**Key concern — the persistence assumption.** The formal oracle model (Section 1.1) stipulates that the oracle is *persistent*: repeated calls to Q̃(e1, e2) always return the same answer. This is an extremely non-standard assumption that significantly limits the practical relevance of the model. LLMs and crowdsourcing workers — the applications cited — do *not* satisfy persistence; each call is an independent draw. The introduction presents these as natural oracle realizations, but they violate the model's defining property. This gap between the motivating examples and the formal model should be discussed explicitly and prominently. At present, persistence is mentioned only parenthetically in the formal definition and never again.

**Concern — near-optimality claim.** The introduction claims query complexities are "near-optimal within logarithmic factors," but no query-complexity lower bound in the pure R-model (for k-median/means) is established for the specific complexities O(nk polylog n) and O((n+k²) polylog n). The lower bound in Appendix A concerns the *number of centers*, not the number of queries. The authors should either state the query lower bound explicitly or soften this claim.

---

### Problem Setup & Contributions (Section 1.1 / Coreset Definitions)

The definition hierarchy (O(1)-coreset → O(1)-Coreset+ → ε-Coreset+) is useful. The Coreset+ notion is a genuine contribution; the observation that an O(1)-Coreset+ immediately yields an O(1)-coreset is important.

**Concern — duplication artifact possibly masking a real issue.** The O(1)-coreset definition (lines 147–153) and the (k,ε)-coreset definition (lines 156–166) each appear twice in succession. Although this is likely a PDF-parsing artifact, it is worth confirming that the definitions are internally consistent — specifically, the O(1)-coreset definition uses a weight function ω: T → R>0 over *T*, while the (k,ε)-coreset definition uses ω: V → R>0 over all of *V* despite the coreset being a set T ⊆ V. This inconsistency should be corrected in the final version.

**Concern — approximation ratio clarity.** Table 1 labels all results as "O(1)-approximation algorithms," but the algorithms produce O(k polylog n) centers rather than k centers, so the notion of "approximation" is non-standard. The table caption mentions this but should be more emphatic: the approximation guarantee is relative to OPT^p_k even though the output has Θ(k polylog n) centers. This is legitimate (as the Coreset+ cost is bounded by O(1)·OPT_k^p) but requires care.

---

### Technical Overview (Section 3) and Algorithms (Appendices B, C, D)

This is the strongest part of the paper. The technical overview is clear and the three-level hierarchy (ALG-G → ALG-D → ALG-DI) is well motivated.

**Core novelty: ALG-TESTER.** The mechanism that emulates an adversarial oracle from a probabilistic one — using kernel/guard sets and a majority vote — is the most interesting contribution. The reduction "when both query points are farther than the kernel radius, ALG-TESTER behaves like an adversarial oracle with µ=1" (Lemma B.5) is the crux, and it is carefully proven.

**Concern — dependence on Geissmann et al. (2025).** The key primitive PROBSORT (Lemma 2.1) relies entirely on Geissmann et al. (2025), which at the time of submission is an arXiv preprint (arXiv:2508.19785). The entire approach collapses if this result has a bug or is not yet peer-reviewed. The authors should either (a) reproduce the proof of PROBSORT inline or in an appendix, or (b) more explicitly state this external dependence as a caveat.

**Concern — ADVSORT query model mismatch.** Lemma 2.2 (ADVSORT) is stated for an oracle that can answer queries of the form (e1, e2) where both edges are arbitrary. However, in ALG-TESTER, the queries issued to the probabilistic oracle Q̃ are of a specific structured form (one endpoint fixed). The paper states ALG-TESTER "behaves like" an adversarial oracle, but the transition from probabilistic oracle calls to adversarial oracle behavior (Lemma B.5) requires the oracle queries within ALG-TESTER to be independent. Isolation is partially addressed (Lemma B.1 proves no query is *shared* between rounds or between PROBSORT and the tester), but within a single invocation of ADVSORT on Y_i, ALG-TESTER is called O(nk polylog n) times. The independence condition across these calls needs to be established. Specifically: if ALG-TESTER is called on the same pair (s, v1), (s, v2) twice during ADVSORT, could the same oracle answer be reused, violating the i.i.d. assumption for the majority test? This deserves clarification.

**ALG-D analysis — lazy filtering.** The description of ALG-D mentions "lazy filtering" but the analysis in Appendix C (not fully read but sketched in Section 3.3) is presented at high level. The claim that "the overall number of quadruplet oracle required in this step is O(n polylog n)" is not easy to verify from the overview alone; this is precisely where the key complexity savings occur, and a clearer complexity accounting would strengthen confidence.

**Theorem 3.3 (ALG-DI) — limited scope.** The ε-Coreset+ result applies only to *k-median* (p=1), while Theorems 3.1 and 3.2 apply for any constant p. This restriction is stated in Theorem 3.3 but is not mentioned in the abstract or introduction. The abstract says "further improve the approximation from constant to 1+ε" without the p=1 caveat. This should be fixed.

**Lower bound (Appendix A).** The lower bound construction is clean and correct. The proof shows that any o(n)-approximation algorithm with at most 2k-2 centers fails on at least one of two metric instances that are indistinguishable by the oracle. One note: the theorem requires 3 ≤ k = O(1); whether the condition k = O(1) is essential or an artifact of the proof is not discussed.

---

### Experiments & Results (Section 4)

The experiments validate the basic correctness of ALG-G but fall well short of the standards expected at ICLR.

**Concern — scale.** The real datasets are subsampled to only n=2,000 points (from 30k–50k original points). At this scale, it is unclear whether the coreset complexity advantages are meaningful. Given that the theoretical query complexity is O(nk polylog n), the paper should test at larger n to show practical scalability.

**Concern — "optimal" baseline is not optimal.** The paper acknowledges (footnote 3) that k-means++ on ground truth data is not the true optimum. Calling it "optimal" throughout Figures 2–4 is misleading, especially since k-means++ can be far from optimal in adversarial cases.

**Concern — distance oracle used in implementation.** Section 4.1 states: "we then apply k-means++ algorithm on the weighted coreset to obtain the final set of k centers, invoking the distance oracle when necessary." This means the experimental evaluation is *not* in the R-model — it uses a distance oracle for the final optimization step. The theoretical contribution is about the R-model without distance access, so the experiments do not cleanly validate the theory. This gap should be explicitly discussed: is the distance oracle for the final k-means++ step unavoidable in practice?

**Concern — weak baseline.** The baseline compared against is the "Generic algorithm" that trusts the oracle blindly. This is an extremely weak competitor. The paper provides no comparison against Addanki et al. (2021) (which also works in the R-model, though only for k-center and under structural assumptions), nor against any RM-model algorithm that adds a distance oracle. Without such comparisons, it is hard to assess whether the practical performance gains justify the algorithmic complexity of ALG-G.

**Concern — no runtime measurement.** For a work targeting ICLR's ML audience, the wall-clock runtime and the actual number of oracle calls in experiments are not reported. Does ALG-G terminate in reasonable time on n=10,000 points? The theoretical bound is O(nk polylog n) oracle calls, which for k=5, n=10,000 would be roughly O(10^6) calls — this should be verifiable and reportable.

**Concern — no test of LLM oracle.** The introduction specifically motivates quadruplet oracles via LLMs, but no experiment uses an actual LLM. Since the persistence assumption is violated by LLMs, at minimum the authors should discuss what happens empirically when persistence fails.

---

### Writing & Clarity

The paper is generally well-written, particularly the Technical Overview. The main text is accessible for a theory-oriented audience, though some definitions (Coreset+) may be unfamiliar to a broader ML audience. One significant clarity issue: the algorithm description in Appendix B refers to the "isolation property" before formally establishing it, which creates a forward reference that complicates reading the proofs in sequence.

---

### Limitations & Broader Impact

The paper does not include a limitations section. Key limitations that should be acknowledged:

1. **Persistence assumption:** Practical oracles (LLMs, crowdsourcing) typically provide i.i.d. answers, not persistent ones. The entire analysis relies on persistence (particularly PROBSORT and the isolation property). Whether the results extend to non-persistent noise is an important open question.

2. **Number of centers:** Outputting O(k polylog n) centers rather than exactly k is a significant practical limitation. For large k (e.g., k=100), this could mean thousands of centers, defeating the purpose of clustering.

3. **Approximation constant:** The O(1) approximation factor is not made explicit. From the proof, the constant appears to be at least 5 (Corollary B.11 bounds the total cost by 5·OPT_k^1), but the true constant for p>1 or after composing all the algorithm steps is larger and unstated. ICLR readers expect to know the approximation constant.

4. **Computational complexity:** Query complexity ≠ runtime complexity. The paper focuses exclusively on oracle calls and never analyzes computational (time) complexity beyond "O(m polylog n) queries to ADVSORT." For a practical contribution, the running time should be analyzed.

---

### Overall Assessment

This paper makes a genuine and non-trivial theoretical contribution: it is the first to achieve a constant-factor approximation for k-median and k-means in the pure R-model (no distance oracle), achieving query complexities that match the RM-model results of prior work. The core algorithmic ideas — kernel/guard sets, the ALG-TESTER emulation of an adversarial oracle, and lazy filtering for doubling metrics — are novel and carefully analyzed. However, several issues temper enthusiasm for acceptance at ICLR in its current form. First, the experimental evaluation is preliminary: it tests only small-scale data, uses a weak baseline, invokes a distance oracle in the implementation (undermining the R-model story), and provides no runtime measurements. Second, the practical relevance is limited by the persistence assumption, which is unacknowledged as a major caveat despite being violated by every motivating application cited (LLMs, crowdsourcing). Third, the O(k polylog n) center count and the unspecified O(1) approximation constant are significant gaps between the theory and usable practice. For a strong theory conference (SODA, STOC, PODS), this work would be competitive. For ICLR, the weak empirical validation and the gap between practical motivation and formal model make it difficult to recommend acceptance without major revisions to address experiments, the persistence caveat, and the ALG-DI scope limitation in the abstract.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates $k$-clustering (specifically $k$-median and $k$-means) in the Rank-model (R-model), where access to exact distances is replaced by a noisy quadruplet oracle. The authors design randomized algorithms that construct a "Coreset+" of size $O(k \cdot \text{polylog}(n))$ with a mapping to these centers, achieving a constant-factor approximation of the optimal clustering cost. They achieve a query complexity of $O(nk \cdot \text{polylog}(n))$ for general metrics and improve this to $O((n + k^2) \cdot \text{polylog}(n))$ for metrics with bounded doubling dimension, while also providing an $(1+\epsilon)$-approximation for the latter case.

### Strengths
1.  **Theoretical Novelty in Weak Oracle Models:** The paper addresses a significant gap in oracle-based clustering literature. While prior work (e.g., Galhotra et al. 2024; Raychaudhury et al. 2025) relied on the Weak-Strong model (requiring some exact distance queries), this work proves that constant-factor approximations for $k$-median and $k$-means are achievable using *only* noisy comparison oracles (Pure R-model), which was previously thought to require exact distances for these objectives.
2.  **Robust Algorithmic Design:** The "Kernel" and "Guard" set mechanism is a clever and rigorous adaptation of coreset construction techniques to handle noisy, persistent comparison errors. By simulating an adversarial oracle using probabilistic queries, the authors leverage existing sorting lemmas (ADVSort) effectively to handle distance uncertainty, which is technically sound and elegant.
3.  **Practical Relevance:** The motivation aligns well with current trends in Learning-Augmented Algorithms. The authors explicitly connect their model to LLMs and embedding services as potential implementations of the quadruplet oracle, making the results relevant for real-world applications where computing exact distances is costly or noisy.

### Weaknesses
1.  **Relaxation of Center Count:** The paper outputs $O(k \cdot \text{polylog}(n))$ centers rather than exactly $k$ centers. While the authors justify this by citing a lower bound ($2k-1$ centers necessary) and practical utility (mapping smaller set to human annotators), this deviates from the standard $k$-clustering problem definition. The community might prefer a method that returns exactly $k$ centers with a slightly sub-constant approximation or higher query cost, though this relaxation is arguably necessary in the R-model.
2.  **Experimental Validity:** The experimental section relies on simulating the oracle using ground truth distances from real datasets. While the baseline comparison (blindly trusting the noisy oracle vs. their filtering) is valid for the *algorithmic* step, it does not capture the cost/latency implications of a real-world oracle (e.g., LLM inference time). A deeper discussion on the trade-off between filtering accuracy and oracle query cost would be beneficial.
3.  **Proof Accessibility:** While Section 3 provides a Technical Overview, the full proofs are deferred to Appendices B, C, and D. Given ICLR's focus on reproducibility and technical verification, a more streamlined presentation of the core analysis in the main text (e.g., bounding the dislocation or injection of bad to good vertices) would strengthen the paper's impact.

### Novelty & Significance
*   **Novelty:** **High.** The primary contribution is extending coreset-based clustering to the Pure R-model for $k$-median/means, a problem where prior theoretical results required even minimal access to distance oracles. The specific technique for handling probabilistic noise via kernel/guard sets in a sorting context adds new methodological tools to the learning-augmented algorithms community.
*   **Significance:** **Moderate to High.** The results provide a theoretical foundation for using comparison-based data (from LLMs or crowdsourcing) for metric clustering tasks. By removing the need for exact distances, it lowers the barrier for applying clustering in scenarios where distance metrics are semantic and expensive to compute.
*   **Clarity:** Moderate. The Technical Overview is well-structured, but the reliance on "Coreset+" definitions and complex sampling rounds requires careful reading. The notation in the extracted text has some artifacts (e.g., `O_ ( _n_ \cdot _k_`), which obscures formulas but does not detract from the logic.
*   **Reproducibility:** Theoretical results appear reproducible given the detailed algorithms and lemmas in the appendices. However, the stochastic nature of the R-model noise requires careful parameter tuning (e.g., `phi < 1/4`) which should be highlighted for practical implementation.

### Suggestions for Improvement
1.  **Clarify the "Center Count" Trade-off:** In the Conclusion or Discussion, briefly address the scenario where the application strictly requires exactly $k$ centers (e.g., hardware constraints). Could the extra $O(k \text{polylog}(n))$ centers be pruned post-hoc with minimal cost increase? Acknowledging this limitation explicitly would manage reader expectations.
2.  **Strengthen Experimental Analysis:** Consider adding an analysis of the *number of queries* required to achieve convergence in practice. The theoretical bound is $O(nk \text{polylog}(n))$, but in the experiments, does the algorithm converge faster? Visualizing query usage vs. cost might add value.
3.  **Refine the Baseline:** The "Baseline" in the experiments assumes the oracle is perfect. A more comprehensive baseline might include "Optimal Distance" (if accessible, to show the gap) or "Naive Sorting" without Kernel/Guard filtering, to further highlight the specific necessity of their filtering technique.
4.  **Proof Summary:** If space permits, moving the core structural Lemma (e.g., Lemma B.10 regarding the injection from bad to good vertices) into the main body would improve the flow for readers not interested in reading the full appendices immediately.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Real Oracle Implementation:** Replace simulated noise with actual LLM or human quadruplet queries. Without this, the claim that the method works for "LLMs or human feedback" is unsubstantiated speculation rather than empirical fact.
2. **Pure R-Model Evaluation:** The experiments use a distance oracle for the final $k$-means++ step on the coreset, violating the "Only Weak Comparison Oracles" premise. You must demonstrate clustering the coreset using only quadruplet queries to validate the core constraint.
3. **Query Complexity Scaling:** Plot actual query counts against $n$ and $k$. The theoretical bounds ($O(nk \text{polylog} n)$) are central contributions; without empirical verification of this scaling, the efficiency claims remain unproven.

### Deeper Analysis Needed (top 3-5 only)
1. **Doubling Dimension Verification:** Estimate the doubling dimension of the Adult and Credit datasets. The improved bounds (ALG-D/ALG-DI) rely on this; applying them to high-dimensional tabular data without verification undermines the validity of those specific results.
2. **Noise Model Robustness:** Analyze performance under structured or adversarial noise rather than just probabilistic $\phi$. LLMs often exhibit bias or consistent errors rather than independent random noise, which threatens the theoretical guarantees.
3. **Constant Factor Analysis:** Theoretical bounds hide large constants in $\text{polylog}(n)$ and approximation factors. Provide an analysis of the practical magnitude of these constants to assess real-world viability.

### Visualizations & Case Studies
1. **Query Cost vs. Accuracy Trade-off:** Visualize the Pareto frontier of query budget vs. clustering cost. This reveals whether the theoretical query savings actually translate to practical efficiency gains.
2. **Coreset Representation Quality:** Show t-SNE/UMAP of the selected coreset points vs. random sampling. This exposes whether the algorithm actually captures structure or just reduces size.
3. **Failure Mode Visualization:** Show examples where the method fails (e.g., high doubling dimension or high noise). This establishes the boundaries of the method's applicability.

### Obvious Next Steps
1. **Remove Distance Oracle Dependency:** Modify the experimental pipeline to cluster the coreset using only comparison queries, aligning practice with the theoretical R-model definition.
2. **Integrate Real LLM API:** Run the quadruplet oracle using an actual LLM (e.g., GPT-4) on image or text data to validate the primary motivation.
3. **Dataset Characterization:** Explicitly measure and report the intrinsic dimensionality of used datasets to justify the use of doubling-dimension-specific algorithms.

# Final Consolidated Review
## Summary

This paper studies $k$-clustering ($k$-median and $k$-means) in the Rank-model (R-model), where pairwise distances are inaccessible and only a noisy quadruplet comparison oracle is available. The authors design randomized algorithms that compute an $O(1)$-Coreset+—a set of $O(k \cdot \text{polylog}(n))$ centers with a mapping function—using only quadruplet queries, achieving constant-factor approximations to the optimal $k$-clustering cost. The query complexity is $O(nk \cdot \text{polylog}(n))$ for general metrics and improves to $O((n + k^2) \cdot \text{polylog}(n))$ for bounded doubling dimension; for $k$-median in doubling metrics, the approximation improves to $(1 + \epsilon)$.

## Strengths

- **Novel theoretical contribution:** This is the first work to achieve constant-factor approximations for $k$-median and $k$-means in the *pure* R-model without any distance oracle access. Prior work (Galhotra et al. 2024; Raychaudhury et al. 2025) required the RM-model with at least some distance queries. The lower bound (Appendix A) showing that $2k-1$ centers are necessary for any $o(n)$-approximation motivates the Coreset+ relaxation.

- **Clever algorithmic mechanism:** The kernel/guard set construction and ALG-TESTER procedure that emulates an adversarial oracle from a probabilistic oracle (Lemma B.5) is technically elegant. By ensuring query points are farther than the kernel radius, the majority vote over guard comparisons achieves the adversarial behavior needed for ADVSORT, which is a non-trivial insight.

- **Comprehensive theoretical analysis:** The three-tier algorithmic hierarchy (ALG-G for general metrics, ALG-D for doubling metrics, ALG-DI for refined approximation) is well-motivated, and the proofs establish isolation properties (Lemma B.1), dislocation bounds, and injection arguments for bad-to-good vertex cost accounting (Lemmas B.9–B.10) with appropriate rigor.

- **Clear presentation of the technical overview:** Section 3 provides an accessible roadmap of the algorithmic ideas, making the appendices follow naturally for readers seeking full detail.

## Weaknesses

- **Persistence assumption limits practical relevance:** The oracle model assumes *persistent* answers—repeated queries return the same result. However, the motivating applications (LLMs, crowdsourcing) typically provide independent samples each time. The paper acknowledges this assumption only parenthetically in Section 1.1, but it fundamentally affects applicability. Whether the results extend to non-persistent noise remains unaddressed.

- **Experimental evaluation departs from the R-model:** The implementation invokes a distance oracle for the final $k$-means++ step on the coreset (Section 4.1), which contradicts the paper's premise of using *only* quadruplet queries. The experiments thus do not validate the pure R-model contribution. A complete evaluation would require clustering the coreset using only comparison queries.

- **No validation with real oracles:** The introduction motivates quadruplet oracles via LLMs and embedding services, yet all experiments simulate oracle noise synthetically. Without testing against actual LLM or human annotators, the practical applicability remains unverified—particularly concerning given the persistence assumption mismatch.

- **Unsubstantiated "near-optimal" query complexity claim:** The introduction states that query complexities are "near-optimal within logarithmic factors," but no query-complexity lower bound is established. The Appendix A lower bound concerns the *number of centers*, not the number of queries. This claim should be either substantiated or qualified.

- **Dependence on an unreviewed arXiv preprint:** The key PROBSORT primitive (Lemma 2.1) relies entirely on Geissmann et al. (2025), an arXiv preprint. While this is methodologically acceptable for establishing results, the correctness of the entire approach hinges on this external result.

- **Experimental scale and baseline limitations:** The real datasets are subsampled to $n = 2{,}000$ points, which is too small to demonstrate scalability. Additionally, the only baseline is a naive algorithm that blindly trusts the noisy oracle; there is no comparison to prior R-model or RM-model algorithms.

- **Approximation constant is opaque:** The theoretical results guarantee $O(1)$-approximation, but the constant factor is never made explicit. From the analysis (Corollary B.11), the constant appears to be at least 5 for general metrics, but the composite constant after all algorithm stages is unclear.

- **Datasets not validated for bounded doubling dimension:** ALG-D and ALG-DI assume bounded doubling dimension, but the Adult and Credit datasets used in experiments are not verified to satisfy this assumption. Applying these algorithms to arbitrary high-dimensional data without justification may undermine the improved bounds.

## Nice-to-Haves

- Analysis of algorithm performance under non-persistent (i.i.d.) noise models
- Empirical validation of query complexity scaling against $n$ and $k$
- Runtime measurements (wall-clock time, not just theoretical query bounds)
- Larger-scale experiments to demonstrate practical utility

## Removed Points

*These points were flagged for removal as they are factually incorrect, minor nitpicks, or already addressed in the paper:*

- **Claim that abstract omits the $O(k \cdot \text{polylog}(n))$ center count:** Incorrect—the abstract explicitly states "computes a set of $O(k \cdot \text{polylog}(n))$ centers."

- **Claim that the $p=1$ limitation of ALG-DI is not mentioned in the abstract:** Incorrect—the abstract clearly states "for the special case of $k$-median."

- **Duplication artifact in coreset definitions:** This is a PDF parsing artifact; the definitions are consistent.

- **Forward reference to isolation property in Appendix B:** The isolation property is properly stated before being used in subsequent lemmas; this is standard proof structure.

- **Citation formatting artifacts:** Parser issues that do not reflect paper quality.

## Novel Insights

Beyond the paper's contributions, a fundamental tension emerges between theoretical elegance and practical deployment: the Coreset+ relaxation (outputting $O(k \cdot \text{polylog}(n))$ centers instead of exactly $k$) is mathematically necessary in the R-model, but it shifts the burden of selecting $k$ final centers to a downstream process that may require exact distances. If that downstream step must use a distance oracle, then the "pure R-model" contribution becomes a preprocessing reduction rather than a complete solution. The paper would benefit from explicitly characterizing the information-theoretic or computational barriers to returning exactly $k$ centers with $O(1)$-approximation in the R-model. Additionally, the hidden constants in $\text{polylog}(n)$ grow with the success probability $1 - n^{-\Omega(1)}$; for practical use at scale, understanding the trade-off between success probability and query budget is more relevant than the asymptotic guarantee.

## Suggestions

1. **Address the persistence assumption:** Either (a) discuss explicitly why it is reasonable or (b) provide an extension to the standard i.i.d. noise model with analysis of the increased query complexity.

2. **Complete the R-model experimental pipeline:** Implement a coreset clustering phase using only quadruplet queries (even if suboptimal), or clearly articulate what additional information would be needed for the final step.

3. **Verify or discuss dataset assumptions:** Either estimate the doubling dimension of the datasets used or restrict experiments to synthetic data with controlled dimension.

4. **Add empirical query complexity validation:** Report the actual number of quadruplet queries made during experiments to bridge the gap between theoretical bounds and practical cost.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0, 6.0]
Average score: 7.2
Binary outcome: Accept
