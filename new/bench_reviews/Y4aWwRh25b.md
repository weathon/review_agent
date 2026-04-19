## Summary

The paper identifies and empirically demonstrates a security vulnerability in Retrieval-Augmented Generation (RAG) systems: instruction-tuned language models can be prompted to reveal retrieved context verbatim through simple prompt injection. The authors evaluate 10 open-source models showing instruction tuning increases extractability ~5x over base models, and present a production attack achieving 100% success on 25 customized GPTs with up to 41.73% reconstruction of a 77k-word book from 100 queries. A position-bias elimination strategy (PINE) reduces reconstruction by ~40%.

## Strengths

- **Extensive empirical scope across the model zoo:** The open-source evaluation spans models from 7B to 72B parameters across seven families (Table 1), showing the vulnerability is consistent and scales with model size. Qwen1.5-72B reaches ROUGE-L of 99.15 and BERTScore of 99.75, demonstrating near-perfect verbatim extraction.

- **Strong production-level evidence with practical attack vector:** Section 4's attack on customized GPTs is the paper's most compelling contribution. Achieving 100% success rate on 25 GPTs (Section 4, Exp 1) using tool-invocation exploitation (`myfiles_browser`) and extracting 41.73% of a copyrighted book with only 100 queries (Figure 6) elevates this from an academic observation to a real-world security concern.

- **Clear ablation studies characterizing operational risk factors:** Figure 3 (context size and chunk count analysis) and Figure 5 (position/U-shaped curve analysis) provide actionable insight into when leakage is most severe. The finding that longer chunks and fewer chunks increase reconstruction length is practically useful for RAG system designers.

## Weaknesses

### Major

- **The claim of "scalable data extraction" is partially established for open-source models but strongly established for production systems:** In the abstract and title, the paper frames the contribution as "scalable data extraction from RAG systems." For open-source models (Table 1), the primary metrics are per-query similarity scores (ROUGE/BLEU/F1). While the mitigation section (Table 3) introduces Reconstruction Rate (88.88% baseline), this is evaluated only on Llama3 8b and does not include multi-query reconstruction curves showing marginal gain per query or datastore coverage growth — the evidence needed to support "scalability." However, Section 4 partially addresses this concern: the GPT experiments DO report reconstruction curves (Figure 6) across query counts, making the production-side scalability claim well-supported. The gap is that the open-model section, which forms the empirical backbone, lacks equivalent multi-query reconstruction analysis.

- **Mitigation evaluation is too narrow to justify broad defense claims:** The abstract and conclusion frame the mitigation as generally applicable ("vulnerability can be greatly mitigated by position bias elimination strategies"), but Table 3 evaluates only Llama3 8b Instruct with one attack setup. There is no utility-preservation evaluation on benign RAG tasks (e.g., whether PINE degrades normal answer quality), no robustness check against alternative attacks, and no evidence that the mitigation transfers to production systems where the vulnerability was most clearly demonstrated. Without this, the defense reads as a proof-of-concept rather than a validated solution.

### Minor

- **Causal mechanism attribution is stronger than evidence supports:** The abstract states the vulnerability is an "outcome of failure in effectively utilizing contexts." The body appropriately hedges this ("might stem from the presence of position bias" in Section 1), but the abstract and Section 3.1 present the U-shaped reconstruction curve (Figure 5) as mechanistic evidence. The data show positional sensitivity, not necessarily context-utilization failure — a simpler explanation is that models are stronger at following proximal instructions. This matters because the proposed defense (PINE) is motivated by this mechanism, and if the mechanism is incorrect, the defense's transferability becomes questionable.

## Nice-to-Have
- Multi-query reconstruction curves for open-source models would bridge the gap between per-query similarity evidence and the datastore-level reconstruction demonstrated for GPTs.
- Testing whether PINE (or similar position-aware strategies) preserves normal RAG answer quality would make the mitigation claim more actionable for practitioners.
- Evaluating retrieval/output filtering baselines in the production-style setting could complement the attack demonstration with the full defense-and-attack picture.

## Removed Points
These points are flagged to be removed, treat them with caution:

1. **Missing experimental details for instruction-tuning comparison (decoding settings, prompt formatting, etc.):** Cross-checked against the paper — these details may be in the stripped appendix or in the implementation. The stark gap shown in Figure 2 (~10 vs ~80 ROUGE-L) is large enough that even with format variations, the qualitative result would hold. Removing as a potential strip artifact and not substantive.

2. **Harry Potter ablation confounds datastore familiarity with query quality:** The paper explicitly notes this limitation in Section 3.1 ("Although we have no knowledge of Llama2's training data, the gains...lead to a hypothesis that they have been trained on Harry Potter"). The confounder is acknowledged and the conclusion is appropriately framed as a hypothesis. The critic overstates this as a flaw when the paper is honest about it.

3. **"Scalable extraction not established" as a wholesale criticism:** As noted above, Section 4 DOES establish scalability for production systems with 100-query reconstruction at 41.73%. The criticism is valid for the open-model section specifically but not for the paper as a whole. Moved to a major weakness with this nuance rather than removed entirely.

4. **BERTScore being "not very informative" for verbatim leakage:** This is a reasonable preference but BERTScore adds semantic context that complements the other metrics. Removing as a metric preference, not a substantive flaw.

5. **Requesting more architectural comparison (non-RIC systems):** This would be outside the paper's stated scope, which is explicitly about RIC-based RAG. Removing as scope creep.

## Novel Insights

The paper's most novel contribution is the empirical demonstration that RIC-based RAG systems, by design, create a tension between context augmentation and data protection: the very mechanism that makes RAG useful (prepending retrieved knowledge to context) also makes it vulnerable to extraction when combined with instruction-tuning. The finding that instruction tuning, which is typically considered purely beneficial, introduces a ~5× increase in extractability over base models is practically significant and reframes the security conversation around alignment capabilities. The production attack showing that customized GPTs can be exploited through tool-invocation bypass (rather than direct prompt injection alone) reveals that real-world RAG implementations have additional attack surfaces beyond the academic literature's typical focus on prompt manipulation.

## Suggestions

1. **Add multi-query reconstruction curves for at least one open-source model** to bridge the gap between per-query similarity results and the scalability claim in the title. This would require running the attack across increasing query budgets and reporting datastore coverage growth, similar to Figure 6 for GPTs.

2. **Include a utility-preservation evaluation for PINE** on standard RAG QA tasks (e.g., answer quality metrics like exact match or F1 on normal queries) so readers can assess the tradeoff between extraction resistance and normal functionality.

3. **Tone down the abstract's causal claim** to match the body's hedging ("might stem from" → "is consistent with position-bias effects") and clarify in the conclusion that PINE's transfer to production systems is untested.

## Score and Decision

I calibrated against several anchor sets:
- **High-scoring security papers with production attacks** (bhK7U37VW8, zZ8fgXHkXi, GEcwtMk1uA — scores 8-8-8, 8-5-8, 6-8-8): The paper under review has comparably strong production demonstrations (100% attack success on 25 GPTs) but slightly narrower open-source scalability evidence.
- **Papers with overclaimed contributions but strong experiments** (Kz3yckpCN5 — scores 8-6-8-6; the "False Promise" paper): Similar pattern where the title/abstract overstate but the core findings are valuable — accepted at spotlight.
- **Borderline security papers** (JqKh7FLUw1, 41uZB8bDFh, ei3qCntB66 — scores 5-6 range): The production results here exceed those papers in practical impact.
- **Training data extraction paper** (vjel3nWP2a — scores 6-6-8-6-8-6, accepted): Similar theme, comparable empirical weight, accepted.

This paper's production evidence exceeds many borderline papers and approaches high-scoring ones. The weaknesses (narrow mitigation evaluation, partial open-model scalability) are real but do not undermine the core contribution. The False Promise calibration paper (Kz3yckpCN5) accepted at 8-6-8-6 despite title-level overclaim; the production extraction paper (vjel3nWP2a) accepted at 6-6-8-6-8-6 with similar scope limitations. This paper is comparable in quality to the latter and slightly below the former on overclaim severity but stronger on production results.

A score of **6.5** places it between borderline and strong accept territory, reflecting genuine contributions that would be valued at ICLR, while acknowledging that the overclaim and mitigation gaps prevent it from reaching 7+.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>