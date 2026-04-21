## Summary

The paper proves that the backward gradient of tensor attention training can be computed in almost linear time $n^{1+o(1)}$, matching the forward pass complexity under the bounded entries assumption $B = o(\sqrt[3]{\log n})$. It derives a closed-form expression for the gradient (Lemma 4.1), proposes a fast algorithm (Algorithm 1) using polynomial approximation and novel tensor algebraic techniques (swap rule, distribution rule, Hadamard-product reduction), and proves a tight hardness result (Theorem 6.3) showing that slightly weakening the assumption renders the problem intractable in sub-cubic time under SETH. This completes the 2×2 grid of matrix/tensor × forward/backward attention complexity.

## Strengths

- **Completes a natural complexity-theoretic gap.** The paper fills the last cell in Table 1 (tensor attention backward), which is a meaningful milestone within the Alman–Song research program. The result shows forward and backward tensor attention share the same $n^{1+o(1)}$ complexity under identical assumptions (Theorem 5.2).

- **Novel tensor algebraic techniques.** The swap rule (Fact 5.4), distribution rule (Fact 5.5), and especially the Hadamard-product reduction (Lemma 5.6) — which reduces $\mathcal{T}_{\text{mat}}(d, n^2, d)$ to $\mathcal{T}_{\text{mat}}(d, n, d)$ — are genuine technical contributions that are specific to the backward pass and do not appear in the forward computation setting. These techniques may find use beyond this specific problem.

- **Tight hardness result.** Theorem 6.3 shows that under SETH, slightly weakening $B = O(\sqrt[3]{\log n})$ to $B = \Theta(\sqrt[\gamma]{\gamma(n) \cdot \log n})$ makes the gradient problem require $\Omega(n^{3-\delta})$ time. The reduction from gradient hardness to forward hardness via interpolation/integration is clean and leverages the existing forward lower bound. This gives a sharp complexity transition rather than a loose separation.

- **Clear presentation of the proof strategy.** The computational graph (Figure 4) and the staged structure of Algorithm 1 (five low-rank propagation stages with dedicated lemmas) make the overall approach understandable even when details are deferred to appendices. The explicit "Technical novelty over previous works" paragraph in Section 5.2 transparently delineates contributions from prior settings.

## Weaknesses

### Fatal
None.

### Major

- **The bounded entries assumption $B = o(\sqrt[3]{\log n})$ severely restricts the regime where the result applies, and the paper overclaims practical relevance.** For practically relevant $n$ (e.g., $n = 2 \times 10^6$), $\sqrt[3]{\log n} \approx 2.4$, meaning the entries of $A_iX_j$ must be vanishingly small. Since the attention logits are $A_1X(A_2 \otimes A_3)^\top/d$ with $d = O(\log n)$, the effective bound on attention scores becomes $B^3/d = o(\log n)/O(\log n) = o(1)$, implying near-uniform softmax distributions and limited model expressiveness. Remark 5.3 defends the assumption as "practical" by citing half-precision floating-point, but this conflates numerical representation bounds with the semantic constraint that attention scores must be small — which is precisely what makes softmax attention useful in practice. The hardness result (Theorem 6.3) proves this boundary is inherent to the problem, which is a genuine contribution, but the abstract and conclusion claim the results "establish the feasibility of efficient higher-order transformer training" and "may facilitate practical applications," which overclaims. The honest characterization is: efficient gradient computation is possible only in the low-expressiveness regime, and this boundary is tight.

- **The paper does not analyze the gradient with respect to $Y$ (and hence $W_{V_1}, W_{V_2}$).** The paper states (line 139) that "the gradient for $Y$ is not the bottleneck, since $A_1X(A_2 \otimes A_3)^\top \in \mathbb{R}^{n \times n^2}$ lies in the non-linear function Softmax." While it is true that the $Y$ gradient path avoids the softmax nonlinearity, the paper does not formally show that $\frac{d\text{Loss}}{dY}$ can also be computed in $n^{1+o(1)}$ time. The $Y$ gradient still involves multiplication by $(A_4 \otimes A_5)^\top$ and the intermediate $n \times n^2$ matrix $S = D^{-1}\exp(A_1X(A_2 \otimes A_3)^\top/d)$, so the same low-rank structure exploited for the $X$ gradient would need to be leveraged. This is likely possible with analogous techniques, but without an explicit argument or proof, the result does not constitute a complete training algorithm.

### Minor

- **The $n^{1+o(1)}$ complexity hides potentially large polynomial factors** from the polynomial approximation degree, tensor algebraic overhead, and the $n^{o(1)}$ column count of the low-rank factors $U_i, V_i, W_i$. While this is standard in fine-grained complexity theory and the paper's contribution is asymptotic, the gap between the theoretical guarantee and any concrete crossover point where Algorithm 1 outperforms the naive $O(n^3)$ computation remains undetermined. Providing even rough estimates of the $n^{o(1)}$ terms as functions of $B$ and $d$ would strengthen the paper's practical argument.

- **The practical motivation connecting tensor attention to GPT-4o and Project Astra is aspirational** (Section 1, line 23). These models do not use tensor attention as defined in this paper, and the connection is speculative rather than grounded. This is a minor framing issue rather than a technical concern.

### Trivial
None.

## Nice-to-Haves

- An empirical evaluation of Algorithm 1 against naive gradient computation for moderate $n$, showing crossover points or at minimum verifying correctness on small instances.
- A brief formal argument (even a lemma sketch) showing the $Y$ gradient can be computed efficiently using the same low-rank structure of $S$.
- Concrete bounds on the $n^{o(1)}$ factors as functions of $B$ and $d$.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No empirical validation that the algorithm is faster than naive $O(n^3)$ computation for any practical $n$" (from Harsh Critic, stated as a critical issue).** Downgraded to minor/nice-to-have because this is a theoretical paper in the learning theory track. The absence of experiments is standard for this venue and research program (e.g., Alman & Song 2024a, 2024b are also purely theoretical). The paper explicitly acknowledges this as future work in the conclusion.

- **"The lengthy catalog of LLM citations (paragraph 1) is uninformative and could be cut entirely" (from Harsh Critic).** This is a formatting/presentation nitpick. The introduction follows a standard motivational structure common in ML theory papers.

- **"Definition 3.9 introduces the assumption on $\max\{\|A_1 X_1\|_\infty, \ldots\}$ but this is a bound on the *products* $A_i X_j$, not on the individual matrices... the relationship should be clarified" (from Harsh Critic).** The bound on products rather than individual matrices is the natural quantity to control, as it directly bounds the entries of the attention matrix $A_1X(A_2 \otimes A_3)^\top/d$. This is consistent with the formulation in Alman & Song (2024a) for matrix attention backward and is not an ambiguity requiring clarification.

- **"Show a concrete worked example" for a small instance (from Harsh Critic).** This is a nice-to-have presentation suggestion, not a weakness.

- **"Fact 5.5 as stated is hard to parse" (from Harsh Critic).** Minor notation concern, not a substantive weakness.

- **Missing related works claims.** Per instructions, not evaluated due to lack of external verification.

## Novel Insights

The tight interplay between the upper and lower bounds is this paper's most distinctive feature: the bounded entries assumption is not merely a convenience for the algorithm but is provably necessary (under SETH) for any sub-cubic gradient algorithm. This means the paper's contribution is not just "here is an efficient algorithm" but "here is a complete complexity characterization of when efficient gradient computation for tensor attention is possible." The real insight, however, is somewhat bittersweet — it reveals that the tractable regime for tensor attention training is precisely the low-expressiveness regime, which tempers the practical aspirations. The paper would be stronger if it acknowledged this tension directly rather than claiming the assumption is "practical."

## Suggestions

- Qualify the abstract and conclusion to state that efficient gradient computation is feasible under bounded entries assumptions, and acknowledge that this regime restricts the expressiveness of the attention mechanism. The current phrasing ("establish the feasibility of efficient higher-order transformer training") implies broader applicability than the results support.

- Add a brief argument (even a paragraph or lemma sketch) showing how the $Y$ gradient can be computed efficiently using the low-rank structure of $S$, or explicitly state this as an open direction if the analysis differs from the $X$ case.

## Calibration

**Anchors retrieved:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Streaming attention (sublinear space) | rKMz6cDE7W | 2.33 | Much worse: poor writing, no novelty, impractical. Our paper has clear novelty and tight results. |
| Fast RoPE Attention (polynomial + FFT) | AozPzKE0oc | 4.80 | Similar: purely theoretical, polynomial method, bounded entries. Rejected due to a technical flaw found during rebuttal. Our paper has no such flaw and has a matching lower bound. |
| PolySketchFormer | YkCjojDG3l | 5.00 | Had experiments + theory but mixed reviews and some impractical aspects. Our paper is purely theoretical but has a more complete theoretical picture (tight upper + lower bounds). |
| Benign overfitting in attention | uVDwunWsLz | 5.25 | Similar score range; had experiments but was deemed incremental. Our paper has more novel techniques. |
| Hardness of learning under symmetries | ARPrtuzAnQ | 7.33 | Had experiments + tight hardness results + clear practical relevance. Our paper has tight results but less practical grounding. |
| Optimal computational-statistical tradeoff | is4nCVkSFA | 7.50 | Matching upper/lower bounds with algorithmic contribution; much more practical relevance. Our paper's contribution is narrower. |

The paper sits above the low-scoring streaming attention paper and the flawed RoPE paper, roughly comparable to PolySketchFormer (5.0) but with a more complete theoretical picture (matching lower bound). It falls below the high-scoring theory papers (7.0+) which had stronger practical grounding and/or experiments. The bounded entries limitation and the Y gradient gap are real but not fatal. A score of 5.5 reflects a solid theory contribution with meaningful technical novelty, tempered by overclaimed practical relevance and the incomplete treatment of the full training gradient.

## Score and Decision

**Originality:** The paper extends the Alman–Song program to tensor attention backward computation with genuinely novel tensor algebraic techniques (Facts 5.4, 5.5, Lemma 5.6). The overall framework (polynomial approximation → low-rank → propagate) is incremental relative to prior work, but the backward-specific techniques are non-trivial.

**Importance of research question:** Efficient tensor attention training is a well-motivated theoretical question. The practical impact is limited by the bounded entries constraint, which the hardness result shows is inherent.

**Claims well supported:** The upper bound (Theorem 5.2) and lower bound (Theorem 6.3) are stated precisely and the proof sketches are reasonable. The practical relevance claims in the abstract and Remark 5.3 are overstated.

**Soundness of experiments:** No experiments, which is standard for this type of theory paper but limits the ability to assess practical utility.

**Clarity:** Well-structured with clear computational graph (Figure 4) and algorithm (Algorithm 1). The technical novelty section is helpful.

**Value to research community:** Completes a natural complexity-theoretic gap and provides tools (tensor algebraic techniques) that may be reusable. The tight characterization is valuable for the fine-grained complexity community.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>