Now I have a complete picture. Let me synthesize the final review.

## Summary

This paper studies the vulnerability of Retrieval-In-Context RAG systems to prompt-injected data extraction attacks, where an adversary instructs instruction-tuned LMs to regurgitate retrieved context verbatim. The paper demonstrates this attack across 10 open-source instruction-tuned LMs (with ROUGE scores up to 99), presents ablation studies on the effects of instruction tuning, data memorization, chunk size, semantic chunking, and position bias, proposes position-bias-elimination mitigation strategies, and extends the attack to production GPTs systems with 100% attack success rate and 41%/3% reconstruction rates on a Harry Potter book and a Wikipedia corpus respectively.

## Strengths

- **Systematic empirical sweep across model families and sizes (Table 1):** Testing 10 instruction-tuned LMs spanning 7B–72B parameters with consistent methodology and multiple similarity metrics (ROUGE-L, BLEU, F1, BERTScore) provides a useful empirical benchmark. The scaling trend (ROUGE-L from ~80 at 7B to ~99 at 72B for Qwen1.5) is clearly demonstrated.

- **Ablation studies providing mechanistic insight (Figures 2–5, Table 2):** The studies on instruction tuning vs. base models (Figure 2), seen vs. unseen data (Table 2), chunk size/number (Figure 3), semantic chunking (Figure 4), and position bias (Figure 5) collectively explain *why* the vulnerability exists. The U-shaped position bias curve (Figure 5, right) connecting RAG vulnerability to known position bias phenomena is a genuine finding with defense implications.

- **Successful attack on production GPTs (Section 4):** Demonstrating 100% attack success rate on 25 randomly selected customized GPTs with at most 2 queries establishes real-world practical impact beyond open-source models. The system prompt extraction methodology is well-documented.

- **Principled mitigation informed by ablations (Table 3):** Connecting the position bias finding to the PINE mitigation strategy, and showing combined mitigation reduces reconstruction rate from 88.88% to 52.34%, is a reasonable first step, even though the defense is incomplete.

## Weaknesses

### Fatal
None.

### Major

- **The headline 41% reconstruction rate conflates memorization with RAG datastore leakage, while the most policy-relevant setting yields only 3%.** The paper's abstract and introduction prominently feature the 41% reconstruction rate from Harry Potter, while the 3% Wikipedia reconstruction rate (the more realistic scenario where datastore content is unseen during pre-training) is buried. The paper itself acknowledges this confound (Section 3.1: "it is unclear whether our result is an artifact of LMs' memorization and pre-training data regurgitation") and uses the Harry Potter experiment specifically to test the memorization hypothesis, finding consistent gains (~9 ROUGE points) from seen data. However, the framing in the abstract does not adequately communicate this distinction. For the most relevant threat model the paper itself identifies—private data unseen during pre-training in the datastore—the 3% rate with 100 queries is far less alarming than the 41% figure suggests. This is not just a presentation issue; it directly affects assessment of the real-world severity of the vulnerability claimed.

- **The combined mitigation still leaves 52.34% reconstruction rate, yet the abstract claims the vulnerability "can be greatly mitigated."** Table 3 shows the best combined defense (Safety-Aware Prompt + PINE) reduces reconstruction rate from 88.88% to 52.34%—still more than half of the datastore being reconstructable. The abstract's claim of "greatly mitigated" is an overstatement relative to the evidence. A 52% reconstruction rate after mitigation means the defense is suggestive rather than definitive.

- **The GPTs attack relies on a fundamentally different mechanism (internal API tool execution) than the open-source attack (context regurgitation), undermining the paper's claim of a unified vulnerability.** The open-source attack (Prompts 1–3) asks the model to repeat its context verbatim—a simple instruction-following exploit. The GPTs attack (Prompt 4) first extracts system prompts to discover internal tool names like `myfiles_browser.search()`, then instructs the model to execute search function calls. The paper itself acknowledges that the naive prompt injection fails on GPTs ("GPTs either output nothing or say 'Sorry, I cannot fulfill that request'"). The 100% attack success rate is thus achieved through a tool-invocation vulnerability, not the same mechanism studied in the open-source experiments. While both are security concerns, they are different vulnerability classes and the paper does not adequately clarify this distinction.

### Minor

- **The definition of "Prompt-Injected Data Extraction" (Definition 1) targets reconstructing retrieved context R_D(q), but the Reconstruction Rate metric (Section 3.2) targets reconstructing the entire datastore.** These are different quantities: extracting the context retrieved for a specific query is easier than reconstructing the whole datastore. The paper uses both notions without always distinguishing which is being discussed.

- **The "no prior knowledge" adversary claim for the open-source experiments is partially softened by using WikiQA questions as anchor queries.** Section 3 states "the adversary has no prior knowledge of the datastore" but then uses WikiQA questions that, while outdated, are semantically structured to target information-dense content. The paper acknowledges that "certain prior knowledge about the datastore would favor the adversary," but this partial concession could be stated more explicitly as a scope limitation.

- **The threat model for the GPTs experiment involves a two-step attack (extract system prompt → discover internal APIs → exploit tool calls).** The paper describes this process in Section 4 but the framing of "no prior knowledge" in Experiment 1 is somewhat misleading: the adversary first obtains knowledge of internal APIs through a separate attack step. The paper should be more explicit that the threat model requires prior successful extraction of system prompts.

### Trivial
None.

## Nice-to-Haves

- Scaling curves for the Wikipedia reconstruction (e.g., 200, 500, 1000 queries) would help assess whether reconstruction grows linearly or plateaus, directly informing severity assessment.
- A controlled experiment with known training data membership (e.g., using canary data) could cleanly disentangle memorization from RAG context leakage, strengthening the core causal claims.
- Qualitative examples of extracted text alongside originals for both Wikipedia and Harry Potter would help readers assess whether extraction is truly verbatim or merely topically similar.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The open-source attack is trivially obvious" (Harsh Critic Point 2):** The simplicity of the attack is actually a *strength* of the paper—it demonstrates that RAG systems are vulnerable to even the most basic prompt injection without requiring any optimization, gradients, or special tokens. This makes the vulnerability broadly applicable and difficult to patch. The tautology concern about base vs. instruction-tuned models is partially valid but overstates itself; the comparison is informative for security practitioners choosing model architectures, and the ~66 ROUGE point jump is a meaningful empirical finding even if conceptually expected.

- **"Demand for evaluation against production defenses" (Harsh Critic Missing Experiment 2):** The paper explicitly shows that GPTs' built-in alignment filtering blocks the naive attack (Section 4) and develops a bypass. Demanding evaluation against additional production defenses goes beyond the paper's scope and represents an unreasonable bar for a vulnerability demonstration paper.

- **"The definition targets retrieved context but Reconstruction Rate targets entire datastore" (partially kept):** While this inconsistency exists, it is a minor clarity issue, not a major flaw. The paper uses both notions in appropriate contexts.

- **"Missing related works" (Harsh Critic general demand):** Per instructions, removed since we cannot verify existence of cited works not in the paper.

- **"Format/style nitpicks" and "typo criticism":** Per instructions, these are parser artifacts, not paper issues.

## Novel Insights

The paper's most important insight is that instruction tuning creates a broad attack surface for RAG datastore extraction precisely because it simultaneously enables the desirable behavior (following user instructions) and the vulnerability (following adversarial instructions). The position bias analysis (U-shaped extraction curve) connects this RAG vulnerability to an already-known LM limitation, suggesting that fundamental LM capabilities and RAG security may be in tension—a finding with direct design implications for practitioners.

## Suggestions

- Reframe the abstract and introduction to lead with the 3% Wikipedia reconstruction rate (the more realistic threat model) and position the 41% Harry Potter rate as a worst-case bound under memorization. This would improve the paper's credibility while still communicating the seriousness of the vulnerability.
- In the mitigation section, explicitly acknowledge that 52% reconstruction rate is far from a complete defense, and discuss what additional mitigation layers (e.g., output filtering, retrieval access controls) would be needed to bring this to acceptable levels.
- Clarify upfront that the GPTs attack exploits a different mechanism (internal tool invocation) than the open-source attack (context regurgitation), even though both stem from instruction-following capability. This strengthens rather than weakens the paper by accurately delineating the attack surface.

<context>
Original reviewer signal: The Harsh Critic argues the paper overstates severity—the 41% headline is confounded by memorization (the realistic Wikipedia rate is only 3%), the open-source attack is trivially obvious, the GPTs attack exploits a different mechanism (tool execution vs. context regurgitation), and the 52% residual reconstruction rate after mitigation means the defense fails. The Strength Finder emphasizes the systematic empirical sweep across 10 models, the principled ablations connecting position bias to the vulnerability, and the successful production-system attack.

What was dropped and why:
- "The open-source attack is trivially obvious"—the simplicity is actually a strength (shows vulnerability without sophisticated techniques); partially valid concern about base-vs-instruction-tuned comparison being tautological is real but overstates itself—the ~66 ROUGE point jump is informative regardless.
- "Demand for evaluation against production defenses"—the paper already shows GPTs' alignment blocks the naive attack; demanding additional defense evaluation is beyond scope.
- "Missing related works"—cannot verify existence; removed per rules.
- Format/style nitpicks—parser artifacts; removed per rules.

Cross-checks performed:
- Verified the 41% vs 3% distinction: The abstract leads with "41% from a book" and "3% from a corpus of 1,569,000 words"—both are present but 3% is significantly less prominent.
- Verified the memorization confound: Section 3.1 explicitly acknowledges the confound ("it is unclear whether our result is an artifact of LMs' memorization and pre-training data regurgitation") but frames it as hypothesis-generating rather than a fundamental limitation.
- Verified the "greatly mitigated" claim: The abstract says "such vulnerability can be greatly mitigated by position bias elimination strategies" but Table 3 shows 52.34% reconstruction remains after combined defense.
- Verified the GPTs attack mechanism: Section 4 explicitly states the naive attack fails on GPTs ("GPTs either output nothing or say 'Sorry, I cannot fulfill that request'") and the actual attack uses internal search API function calls—a different mechanism.
- Verified the Definition 1 vs Reconstruction Rate discrepancy: Definition 1 targets R_D(q) (retrieved context for a query) while Reconstruction Rate in Section 3.2 targets the entire datastore—these are indeed different quantities.

Severity read: The surviving major weaknesses are significant but not fatal. The conflation of memorization with RAG leakage in framing is the most serious—it overstates severity in the most visible parts of the paper but does not invalidate the core vulnerability finding. The incomplete mitigation and the different attack mechanism for GPTs are substantive but addressable through better framing. The paper makes real contributions (systematic evaluation, position bias connection, production system demonstration) that survive these weaknesses.
</context>