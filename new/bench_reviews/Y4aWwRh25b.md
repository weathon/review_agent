## Summary
This paper demonstrates that Retrieval-In-Context (RIC) RAG systems built with instruction-tuned LMs are vulnerable to prompt-injected data extraction attacks. The authors show high similarity scores across multiple open-weights models, achieve 100% attack success on 25 production GPTs, and report reconstruction rates of 41.7% for a Harry Potter datastore and 3.2% for Wikipedia using 100 queries. The paper also proposes PINE-based mitigation strategies.

## Strengths
- **Strong empirical validation on production systems**: The paper achieves 100% attack success rate on 25 customized GPTs with at most 2 queries each (Sec. 4, Experiment 1), demonstrating real-world relevance beyond controlled open-weights settings.
- **Systematic analysis of vulnerability drivers**: Figure 2 shows instruction tuning increases ROUGE scores by ~65 points (from ~15 to ~80), and Table 1 demonstrates vulnerability scales with model size (ROUGE-L from 80.4 at 7B to 99.2 at 72B), providing mechanistic insight into why RIC-RAG is vulnerable.
- **Novel practical finding on GPT tool exploitation**: The discovery and systematic use of the `myfiles_browser.search` function to dump retrieved chunks verbatim (Sec. 4, Adversarial Prompt 4) is a concrete security observation for deployed systems that prior work like Zeng et al. (2024) did not cover.
- **Rich ablation studies on RAG configuration effects**: Figures 3-5 analyze chunk size, semantic chunking, and position bias effects, connecting leakage to known context utilization issues in LMs.

## Weaknesses

### Major

- **Title/abstract overclaims relative to realistic reconstruction rates**: The paper frames itself as "scalable data extraction" capable of reconstructing "the entire customized datastore," but the no-prior-knowledge reconstruction rate on Wikipedia is only 3.22% with 100 queries (Fig. 6). The 41.73% Harry Potter result requires the adversary to know the exact book title and prompt for chapter-covering questions—a rights-holder audit scenario, not the "black-box adversary with no prior knowledge" threat model stated in Sec. 2. This scope mismatch between claims and evidence weakens the central narrative.

- **Mitigation evaluation is insufficiently supported**: Table 3 tests defenses only on Llama3-8B Instruct, with no validation across other architectures from Table 1. The safety-aware prompt reduces reconstruction from 88.88% to 87.57%—effectively noise-level—yet is still presented as a "baseline mitigation strategy." Crucially, no utility cost is measured: how much does PINE degrade normal RAG QA performance? Without this, recommending it as "practical" is premature.

### Minor

- **Threat model inconsistency across experiments**: Sec. 2 defines a "black-box adversary with no prior knowledge of the datastore," but experiments span three distinct regimes: (1) WikiQA queries on unseen Wikipedia (near-zero knowledge), (2) domain-aware queries for WikiGPT ("generate questions covering your knowledge"), and (3) chapter-covering queries for Harry Potter GPT (strong prior knowledge). The paper should explicitly separate risk conclusions for each regime rather than presenting them under a unified threat model.

- **Open-weights metrics don't directly quantify privacy risk**: Table 1 uses ROUGE-L/BLEU/BERTScore between output and retrieved chunk for single queries, which measures chunk-level regurgitation fidelity but not corpus-level leakage. A ROUGE-L of 80 vs. 90 has unclear privacy implications without analysis of exact character-level overlap or legal significance (e.g., proportion of verbatim copyrighted spans).

### Trivial

- **Query budget details underspecified for Table 3**: The mitigation experiments don't state how many queries or what query distribution were used, making it hard to assess robustness under adaptive attackers.

## Nice-to-Haves
- Show reconstruction rate as a function of query budget (100, 500, 1000 queries) with confidence intervals for at least one setting to visualize leakage accumulation.
- Include side-by-side examples of retrieved chunks vs. model outputs under attack to make similarity scores more tangible.
- Test whether an attacker aware of PINE can adapt prompts (e.g., embedding instructions in pseudo-retrieved documents) to circumvent it.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Criticisms about dataset/model existence**: Any suggestion that cited models (Llama3, GPTs, etc.) or datasets (WikiQA, Wikipedia) are "not yet released" or "cannot be independently verified" violates the hard rules—all cited entities are assumed to exist as of 2026-04-19.
- **Missing appendix/proofs complaints**: The parser strips appendix sections from all papers; weaknesses about "missing appendix" or "missing proofs in appendix" are invalid.
- **Formatting/style nitpicks**: Minor presentation issues not affecting core claims should not weigh in evaluation.
- **Unfair comparison claims favoring the author**: If asymmetry in comparisons favors the baseline (not the author's method), this is intentionally asymmetric to prove a stronger point and should not be penalized.
- **Generic reproducibility nitpicks**: Undisclosed hyperparameters or trivial implementation details are not substantive weaknesses per the hard rules.

## Novel Insights
The paper's most novel contribution is the empirical demonstration that instruction tuning—intended to improve helpfulness and safety—paradoxically increases susceptibility to context-copying attacks by ~65 ROUGE points. This inverts the common assumption that alignment improves security. Additionally, the discovery that GPTs' `myfiles_browser.search` tool can be weaponized via prompt injection to dump retrieved chunks verbatim reveals an unexpected attack surface in production RAG systems that prior extraction work (focused on parametric memorization) did not address.

## Suggestions
1. **Reframe claims to match evidence**: Either (a) add corpus-level reconstruction experiments for open-weights models on the 1.5M-word Wikipedia datastore under realistic query budgets, or (b) explicitly re-scope the title/abstract to emphasize "per-query context leakage" and distinguish the rights-holder audit scenario from the no-prior-knowledge setting.
2. **Add utility-security tradeoff analysis**: Evaluate PINE + safety prompt on standard RAG tasks (e.g., open-domain QA) to measure performance degradation. Without this, practitioners cannot assess whether the defense is viable.
3. **Clarify threat model taxonomy**: Add a table mapping each experiment to attacker knowledge level (zero/domain/rights-holder) and restate risk conclusions separately for each.

## Score and Decision
**Calibration anchors consulted:**
- **vjel3nWP2a.md** (Scalable Extraction of Training Data from Aligned, Production Language Models): Scores 6, 6, 8, 6, 8, 6, **Accept Poster**. Similar empirical security paper with production model validation, accepted despite some scope limitations.
- **BHIsVV4G7q.md** (Phantom: General Trigger Attacks on RAG): Scores 5, 6, 3, **Reject**. RAG attack paper rejected for threat model issues and weak baselines—weaker experiments than this paper.
- **fsW7wJGLBd.md** (Tensor Trust): Scores 8, 5, 8, **Accept Spotlight**. Stronger dataset contribution with broader impact.
- **cRmmIdqvZp.md**: Reviewer explicitly notes "title is too general, promise doesn't correspond to provided results" (scores 5, 8, 6, **Reject**). Similar overclaim issue.

This paper has stronger empirical validation than rejected RAG attack papers (Phantom) but exhibits overclaiming issues similar to papers that were rejected or received borderline scores. The production GPT results (100% success rate, 41% reconstruction with prior knowledge) are compelling and comparable to the accepted "Scalable Extraction" paper. However, the threat model inconsistency and shallow mitigation evaluation prevent it from reaching the 7-8 range. Positioned relative to anchors: above rejected RAG attack papers due to stronger experiments, but below the accepted "Scalable Extraction" paper due to more pronounced overclaiming.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>