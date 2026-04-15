## Summary
This paper proposes **StepProof**, an autoformalization workflow that verifies natural-language mathematical proofs incrementally, sentence by sentence, rather than translating and checking an entire proof at once. The core idea is to maintain a stack of previously verified formalized steps, enabling localized regeneration, partial progress tracking, and step-level feedback; experiments compare this strategy against a full-proof baseline and report modest gains in pass rate and larger reductions in formalization/proof time.

## Strengths
- **The paper makes a concrete shift from whole-proof verification to step-level verification with a clearly specified workflow.** Section 3.2 is explicit that “each sentence in the proof is a verifiable sub-proposition,” and the system keeps a “formal proof stack” of verified steps. This is a real systems contribution, not just a vague appeal to chain-of-thought.
- **The work exposes a practically useful interaction model that most autoformalization papers do not target directly.** The interface in Figure 2 supports `REGEN`, `HOLD`, and `UNDO`, and Section 3.2 explains that erroneous steps can be retracted while “retaining previously verified steps,” which directly addresses the poor error localization of full-proof methods.
- **The direct FULL-PROOF vs STEP-PROOF comparison is run under matched dataset/model settings.** In Section 4.1, both strategies are tested on GSM8K with the same base model (Llama3 8B), same theorem prover environment, and the same evaluation split; this supports the claim that stepwise verification can improve efficiency under a controlled comparison.
- **The paper surfaces an important and specific bottleneck: the verifiability of informal proof phrasing.** The Number Theory experiment in Table 4 is limited, but it does provide evidence that rewriting proofs to better align with step verification can improve complete proof success (from 6% to 12%), which is a useful observation for future dataset and interface design.

## Weaknesses
###: Fatal

### Major:
- **The paper’s strongest comparative claims are not supported by fair baseline evidence.** Table 2 compares StepProof against Majority Voting and DTV under different models and very different attempt budgets (e.g., 10 attempts for StepProof vs 64 for the others). Moreover, the DTV result is explicitly a reimplementation with a substituted model (“we use the same method in DTV, but replace the LLM into Llama3”). This means claims such as “surpassed DTV” and especially the conclusion that performance “reached the level of state-of-the-art” are not established by the presented evidence.
- **The main improvement over FULL-PROOF is directionally positive but empirically modest, and the paper overstates it.** Table 1 reports 6.10% vs 5.30% one-attempt proof pass rate. That is only a **0.8 percentage point absolute gain**, yet the paper repeatedly calls the improvement “significant” and says StepProof is “significantly improved in all aspects of performance.” Without uncertainty estimates, repeated runs, or statistical testing, that language is too strong.
- **The evaluation scope is narrow relative to the paper’s broad framing as natural-language mathematical proof verification.** The main benchmark is GSM8K, whose proofs average 4–5 steps according to the paper itself. That is a reasonable starting point for short sequential reasoning, but it is not enough to support broad claims about mathematical proof verification in general, especially since the limitations section itself admits weaker handling of “structured proof methods” and that StepProof is geared toward sequential proofs.
- **Several proposed advantages are argued mostly by intuition rather than directly measured.** Section 3 claims STEP-PROOF reduces generation loops, avoids token budgeting issues, improves stability, and yields stronger formal/informal alignment. But in the experiments, “stability” is only indirectly proxied through runtime variance, and there is no direct quantification of loop frequency, regeneration success, mapping quality, or repair efficiency. So some mechanism claims are plausible but under-validated.

### Minor
- **The efficiency reporting is conditional on passed proofs only, which weakens the interpretation.** Section 4.1 defines average formalization/proof time “for passed proofs,” so Table 1 does not provide a full end-to-end cost picture across all problems, especially when later evaluations allow retries.
- **The new step-pass metric is potentially useful but insufficiently normalized.** The paper defines step pass rate \(r_s\) as the fraction of proof steps verified, but this depends heavily on proof segmentation granularity. Since StepProof’s core idea is decomposition, the lack of rigorous segmentation rules makes cross-method and cross-dataset interpretation harder.
- **The Number Theory rewriting study is suggestive but too loosely controlled for strong causal conclusions.** Section 4.1 says the authors made “simple manual modifications” to 100 proofs, but the paper does not characterize what was changed or isolate whether gains came from shorter steps, more explicit premises, altered proof content, or clearer wording.
- **Some experimental details are under-specified or unclearly reported.** The use of “\(\mu \pm \sigma^2\)” is unusual and harder to interpret than mean ± standard deviation; “Comments Rate” in Table 2 is not clearly defined in the main text; and the paper does not give enough failure analysis to explain why such a large fraction of proofs remain at \(r_s=0\).

### Trivial
- **The paper would benefit from tighter claim wording around novelty and superiority.** Phrases like “we pioneered,” “first,” and “state-of-the-art” are broader than what the experiments securely support.

## Nice-to-Haves
- A direct ablation separating the contributions of stepwise decomposition, proof-stack reuse, and interface actions such as HOLD/REGEN.
- A clearer taxonomy of failure modes: unformalizable informal step, incorrect LLM formalization, theorem prover insufficiency, or proof-structure mismatch.
- Evaluation on more structured proof styles (e.g., case analysis or induction), which the paper itself identifies as a current limitation.
- A small user study or simulated repair experiment to support the claim that step-level feedback is more useful to users, beyond being more granular.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claims that cited baselines/models are unavailable or unverifiable.** Per instruction, if the paper cites them, they are assumed to exist; criticisms based on release status or availability were removed.
- **Requests for unspecified external related work.** Some reviewer complaints about missing related work were not retained because they require external verification.
- **Pure formatting/style complaints.** Minor typos, grammar issues, or parser-related presentation complaints were removed.
- **Reproducibility nitpicks about full prompts/hyperparameters/logs.** These are not core weaknesses for this submission.
- **The Human Finder’s point that the step-by-step idea lacks novelty because it resembles generic process supervision / dense rewards.** That analogy is too broad and does not negate the paper’s concrete autoformalization system contribution.
- **A complaint that the paper fails to compare FULL-PROOF and STEP-PROOF directly.** This is factually incorrect: Table 1 is exactly that direct comparison under matched Llama3 8B / GSM8K settings.
- **A criticism centered on comparing against missing related methods like LEGO-Prover experimentally.** This becomes a missing-related-work/baseline-demand issue; the more defensible version is simply that the novelty claim should be scoped more carefully.

## Novel Insights
The paper is more convincing as a **workflow/interface paper for interactive autoformalization** than as a strong empirical claim of a superior verification strategy. The best-supported contribution is not that StepProof decisively advances proof success rates—it does not—but that it reframes autoformalization from a brittle one-shot translation problem into an incremental verification loop with persistent verified state. That shift is genuinely useful and could matter more in practice than the modest headline pass-rate gain. However, the current experiments mostly validate StepProof on short, sequential reasoning chains; they do not yet show that the approach scales to the kinds of structured proofs where step decomposition is most nontrivial and potentially most valuable.

## Suggestions
- Reframe the contribution more conservatively: emphasize **interactive, localized verification and repair** as the primary contribution, and soften broad superiority/state-of-the-art claims.
- Redo Table 2 under a matched protocol: same model, same attempt budget, same stopping rule, same prover environment for all methods.
- Add uncertainty estimates or repeated-run statistics for pass rates, especially for the 6.1% vs 5.3% comparison.
- Report end-to-end cost over the full dataset, not only over passed proofs.
- Define “Comments Rate” explicitly and justify why it is a meaningful metric.
- Provide concrete examples of proof rewrites in Table 4 and categorize what kinds of edits help StepProof most.
- Include at least a modest failure analysis of the large \(r_s=0\) bucket to reveal whether the bottleneck is data phrasing, formalization errors, or prover limitations.
- Test at least one dataset with less purely sequential proof structure to support broader claims.

## Score and Decision
**Novelty:** Moderate. The stepwise verification workflow and proof-stack interaction are meaningful and more concrete than a generic “decompose the proof” idea, but some novelty claims are overstated.

**Technical soundness:** Moderate-to-weak. The core method is sensible, and the direct FULL vs STEP comparison is technically valid as far as it goes, but several stronger claims are insufficiently supported.

**Empirical support:** Weak-to-moderate. The matched comparison in Table 1 is useful, but the gain is small in absolute terms; Table 2 does not justify the stronger comparative claims; and the evaluation scope is limited.

**Significance:** Moderate. The interactive verification framing is practically interesting, but the current success rates are still low and the evidence for broad impact is not yet strong.

**Clarity:** Moderate. The high-level idea is understandable, but several metrics and claims are imprecisely defined or overstated.

**Calibration against similar papers:**
- **DTV (V5tdi14ple)** was accepted because it paired a concrete autoformalization+verification idea with broader and more convincing empirical gains across multiple datasets; this paper is materially weaker in comparative rigor and breadth.
- **Let’s Verify Step by Step (v8L0pN6EOi)** was accepted with strong evidence that step-level/process supervision yields substantial performance gains; relative to that calibration, this submission has the right intuition but much weaker empirical backing.
- **Process-Driven Autoformalization in Lean 4 (k8KsI84Ds7)** drew strong criticism when evaluation methodology and dataset quality claims outran the evidence. This paper is not as problematic, but it has a similar overclaiming pattern in its comparative framing.
- **Formal Theorem Proving by Rewarding LLMs to Decompose Proofs Hierarchically (D23JcXiUwf)** was rejected despite an interesting decomposition idea because gains were limited and evaluation breadth was insufficient. This submission is close to that pattern: interesting direction, but evidence not yet strong enough for ICLR.
- **Multilingual Mathematical Autoformalization (QqdloE1QH2)** suggests that papers in this area can clear the bar when they either contribute a strong dataset or show robust gains; this paper currently does neither at that level.

Overall, I find the core idea promising and the interactive stepwise framing genuinely useful, but the submission **overclaims relative to the evidence**. For ICLR, the current empirical case is not yet strong enough.

**Score: 4.9**

**Decision: Reject**

MY FINAL SCORE: <pineapple>4.9</pineapple>
MY FINAL DECISION: <orange>Reject</orange>