Now I have a good understanding of the calibration papers and their scores. Let me consolidate my review:

**Calibration anchors:**
- "Safety Alignment Should be Made More Than Just a Few Tokens Deep" (ICLR Oral, scores 10,8,10,10): This is the most topically similar paper. It identifies shallow safety alignment in ARMs, proposes data augmentation defense. Conceptually very similar to this paper (identifying a vulnerability mechanism + proposing targeted defense). This paper got very high scores.
- "Catastrophic Jailbreak of Open-source LLMs" (ICLR Spotlight, scores 8,8,6,6): Identifies a vulnerability based on decoding strategies, proposes defense. Similar structure.
- "Jailbreaking as a Reward Misspecification Problem" (ICLR Poster, scores 6,5,6,6): Novel perspective on jailbreaks, proposes ReMiss attack.
- "Weak-to-Strong Jailbreaking" (Rejected, scores 6,6,6,3,6): Attack requiring model access, limited defense evaluation.
- "Baseline Defenses for Adversarial Attacks" (Rejected, scores 5,8,5,3): Limited evaluation scope.

Now let me analyze the paper under review relative to these anchors:

**Strengths of the current paper:**
1. Novel vulnerability identification specific to MDLMs (a growing model class)
2. Formal theoretical contribution (Theorem 4.1) connecting priming to attack optimization
3. Comprehensive evaluation across 3 MDLMs, multiple attack types, multiple evaluators
4. Strong empirical results for RA defense
5. Well-designed ablations

**Weaknesses:**
1. The anchoring attack threat model is very strong (requires direct intervention) - but this is acknowledged and the First-Step GCG provides a non-intervention attack
2. The monotonicity assumption in Theorem 4.1 needs more scrutiny
3. Residual vulnerability at late intervention steps
4. Limited models (only 7B-scale MDLMs)
5. RA is tightly coupled to specific contamination pattern and single reward model

Compared to the "Safety Alignment" paper (Oral, avg ~9.5), the current paper:
- Has a similarly novel insight (priming vulnerability ↔ shallow alignment)
- Has a similarly targeted defense
- But has weaker experimental validation in some dimensions (no human eval, limited benchmark breadth for safety)
- Has a formal theoretical contribution that the Safety Alignment paper lacked
- The attack threat model concern is more significant here (requiring intervention in denoising)

Compared to "Catastrophic Jailbreak" (Spotlight, avg ~7), the current paper:
- Has a more formal theoretical grounding
- Has a more principled defense
- Has similar threat model concerns (open-source/white-box access needed)
- Addresses both attack and defense more thoroughly

I think this paper is solidly in the 6-7 range. It makes genuine contributions to an important and growing area (MDLMs), has clear empirical results, and proposes a principled defense. However, the theoretical claim about the lower bound is somewhat fragile (monotonicity assumption), and the defense evaluation has gaps. It's not at the level of the "Safety Alignment" oral paper, which was more comprehensive and addressed a broader set of vulnerabilities, but it's clearly above the rejected papers.

Final score: I'll give this a 6.5 - it's a good paper with solid contributions but some notable gaps that prevent it from being in the "strong accept" range.

Let me now write the full review.</think>Now I have a thorough understanding of the paper and relevant calibration anchors. Let me write the final review.

## Summary
The paper identifies a "priming vulnerability" in Masked Diffusion Language Models (MDLMs), where affirmative tokens appearing at intermediate denoising steps can steer subsequent generation toward harmful responses even in safety-aligned models. The authors demonstrate this through an anchoring attack (intervention in denoising) and a novel First-Step GCG attack (no intervention, enabled by a theoretical lower bound), and propose Recovery Alignment (RA), which trains models to generate safe responses from contaminated intermediate states. Experiments across three MDLMs show RA substantially reduces attack success rates while preserving general capability.

## Strengths
- **Novel and well-motivated vulnerability identification.** The paper identifies a genuinely MDLM-specific vulnerability arising from the parallel, iterative denoising mechanism. The demonstration that a single-token intervention at step 1 raises ASR from 2% to 21% (LLaDA Instruct) is compelling and highlights a real safety gap distinct from ARM prefilling attacks.
- **Principled theoretical contribution connecting vulnerability to attack optimization.** Theorem 4.1 provides a formal lower bound justifying First-Step GCG as a tractable surrogate for the intractable full-denoising objective, and the resulting attack achieves ~20× speedup and up to 4× ASR improvement over Monte Carlo GCG (Table 1).
- **Comprehensive defense evaluation.** RA is evaluated against multiple attack families (anchoring, PAD, DiJA, First-Step GCG, PAIR, ReNeLLM, Crescendo) using three evaluators across three MDLMs. The "RA w/o inter" ablation (Table 2) cleanly demonstrates that training on contaminated intermediate states is essential, moving beyond correlational evidence.
- **Strong empirical results with minimal capability degradation.** RA reduces anchoring attack ASR at t_inter=1 from 17.3% to 0.0% for LLaDA Instruct while maintaining average benchmark performance at ~52.6% (Table 4). The curriculum-based training schedule is well-justified by ablation studies (Figure 3).
- **Practical implementation.** RA reuses existing datasets (BeaverTails) and reward models (DeBERTaV3), requiring no additional data construction, which enhances adoptability.

## Weaknesses

### Major:
- **The anchoring attack threat model requires direct intervention in the denoising process, limiting practical significance.** The most dramatic vulnerability results (e.g., ASR >80% at t_inter=16) assume an attacker who can replace the model's internal predicted tokens at intermediate steps. While the paper acknowledges this and proposes First-Step GCG as a more realistic alternative, the gap is significant: First-Step GCG achieves 58% ASR on LLaDA Instruct versus 88.7% for anchoring at the same effective step. The practical relevance of the anchoring attack rests on scenarios where an attacker has white-box access and can modify internal generation states—which, as reviewers of related jailbreaking papers have noted, is primarily a diagnostic tool rather than a realistic deployment threat. This does not invalidate the vulnerability, but the framing should more clearly distinguish diagnostic findings from practical threats.

- **The monotonicity assumption in Theorem 4.1 is empirically under-verified for the harmful targets that matter.** The theorem requires log πθ(r̃_{t+1}=r | q, r_t) ≥ log πθ(r̃_1=r | q, r_0) for all t. The paper claims this holds "across a broad range of models" (Appendix C.2), but the main text provides no quantitative summary of violation rates or severity for the specific harmful targets r used in the attacks. For safety-aligned models, one might expect the opposite: the probability of generating a harmful response r could *decrease* over denoising steps as the model "recovers" toward safe trajectories. If monotonicity fails for typical harmful r, then L_first is not actually a lower bound on the true attack objective, and First-Step GCG's success becomes a heuristic phenomenon rather than evidence of a theoretically grounded exploitation of the priming vulnerability. The paper should, at minimum, report violation rates specifically for harmful r and discuss the consequences.

- **Recovery Alignment is tightly coupled to the specific contamination mechanism used in training.** RA trains on contaminated states constructed by the same masking procedure as the anchoring attack (mt_inter(·|r) where r is a harmful response from BeaverTails). While RA shows gains against PAD and DiJA, both use relatively stylized templates. There is no evaluation against contamination patterns that differ substantially from training (e.g., partial insertions of unknown harmful tokens, adversarial masking schedules, or harmful tokens distributed across multiple steps). This makes it unclear whether RA is learning a general "recovery trajectory" or overfitting to the trained contamination pattern. The paper's claim that RA "improves robustness against conventional jailbreak attacks" (Section 6.2) through a "recovery capability" is plausible but mechanistically unverified—no intermediate-step analysis shows that RA models actually redirect harmful trajectories at intermediate denoising steps.

- **Residual vulnerability at late intervention steps is substantial and unexplained.** At t_inter=32, RA still yields ASR of 50.7% (LLaDA Instruct) and 43.0% (LLaDA 1.5) under the anchoring attack. The paper acknowledges this ("generating a contextually safe response becomes challenging") but does not analyze whether this reflects a fundamental limitation of recovery from heavily contaminated states or a training deficiency. MMaDA's performance under RA remains particularly poor (First-Step GCG ASR of 45.7%, ReNeLLM ASR of 81.7%), and its baseline No-Attack ASR of 79.7% suggests minimal initial alignment, making it hard to attribute improvements specifically to the priming vulnerability mitigation versus adding basic alignment.

### Minor:
- **Single reward model dependency.** All RA experiments use DeBERTaV3 without fine-tuning as the reward model. The paper acknowledges reward hacking at large t_max (Section 6.4) but does not analyze sensitivity to reward model choice or assess whether RA's improvements reflect genuine safety improvements versus surface-level refusal patterns preferred by DeBERTa.
- **No comparison with inference-time defenses.** Simple alternatives—such as monitoring intermediate denoising states for harmful tokens and restarting generation, or applying output-level safety classifiers at each step—are natural baselines that could complement or replace training-time RA. Their absence leaves open whether RA is the most effective approach.
- **Capability evaluation does not probe over-refusal.** The eleven benchmarks (Table 4) are knowledge/reasoning tasks that cannot detect over-refusal on borderline benign queries. A model that reflexively refuses more often could perform similarly on multiple-choice tasks while being less helpful in practice.

## Nice-to-Haves
- **Trajectory-level analysis of RA's mechanism.** Showing actual denoising trajectories—how harmful tokens at intermediate steps are progressively replaced by safe tokens in RA-trained models—would make the "recovery" mechanism concrete and allow readers to verify whether genuine recovery or generic refusal is occurring.
- **Evaluation on larger/more capable MDLMs** to validate whether findings scale beyond the 7B-parameter models tested.
- **Multiple reward models** to demonstrate RA's robustness to reward choice.
- **Explicit comparison with ARM architectures under analogous attacks** to firmly establish that the priming vulnerability is uniquely MDLM-specific rather than a general property of conditional generation.

## Removed Points
These points are flagged to be removed; treat them with caution:
- *Theoretical lower bound is "not established" because monotonicity may not hold for harmful r*: This is addressed partially—the paper does provide empirical validation in Appendix C.2 and acknowledges the assumption. The concern about specific violation rates for harmful r is valid and kept above, but the claim that the theorem is "on shaky ground" overstates the issue given that First-Step GCG empirically succeeds, supporting the practical relevance even if the bound is not tight everywhere.
- *The anchoring attack conflates mechanism with outcome / doesn't isolate "priming" from "prefix conditioning"*: This is a nuanced point but the paper does include the t_inter=1 experiment showing that even a single token makes ASR jump to 21%. While additional controls (neutral tokens, safe tokens) would strengthen the isolation, the current evidence is sufficient to demonstrate a genuine vulnerability specific to the denoising mechanism, even if the causal attribution would benefit from further controls.
- *MMaDA's poor baseline alignment invalidates RA results on MMaDA*: The paper does discuss this implicitly and the MMaDA results, while harder to interpret in isolation, contribute to the overall picture across three models. This is a minor limitation, not a fatal flaw.
- *Reproducibility concerns about hyperparameters, sampling configurations*: These are detailed in the appendices. Standard for the field.
- *Missing related works*: Cannot verify existence of specific works not cited.
- *Formatting/style nitpicks*: Removed per rules.

## Novel Insights
The connection between the priming vulnerability in MDLMs and ARM "shallow safety alignment" is instructive. Both phenomena reveal that alignment methods which only train models from clean initial states (fully masked sequences for MDLMs, empty prefixes for ARMs) fail to equip models for recovery from corrupted internal states. RA's core insight—that one must explicitly train on contaminated intermediates to build resilience—parallels the data augmentation approach in "Safety Alignment Should be Made More Than Just a Few Tokens Deep," but adapts it to the fundamentally different, parallel denoising mechanism of MDLMs. The Theorem 4.1 result, while its monotonicity assumption needs more scrutiny, provides an interesting theoretical bridge connecting single-step predictions to full-trajectory generation probability, which could be useful beyond the attack setting.

## Suggestions
- Report quantitative violation rates of the monotonicity assumption specifically for harmful targets used in attacks, and discuss the practical consequences of violations.
- Add trajectory-level visualizations showing how RA-trained models redirect harmful intermediate states toward safe outputs, rather than relying solely on endpoint ASR metrics.
- Include evaluation on a benign-query over-refusal benchmark (e.g., XRiskQA, OR-Bench) to verify that RA does not cause excessive refusal on safe requests.
- Compare RA against at least one inference-time baseline (e.g., intermediate-step safety monitoring with regeneration) to establish that training-time intervention is necessary.

## Score and Decision

**Calibration comparison:**
- "Safety Alignment Should be Made More Than Just a Few Tokens Deep" (ICLR Oral, scores 10,8,10,10, avg 9.5): Topically closest paper. Both identify a structural vulnerability in safety alignment (shallow alignment for ARMs, priming for MDLMs), both propose targeted defenses, both show ASR improvements. The current paper has a theoretical contribution (Theorem 4.1) but the theoretical foundation is less solid due to the monotonicity assumption. The current paper's defense evaluation is somewhat narrower (3 MDLMs vs broader ARM evaluation). Below this anchor.
- "Catastrophic Jailbreak" (ICLR Spotlight, scores 8,8,6,6, avg 7.0): Similar structure—novel vulnerability + defense. The current paper has stronger theoretical grounding and a more principled defense, but has a more restrictive threat model for its main vulnerability demonstration. Roughly comparable.
- "Jailbreaking as Reward Misspecification" (ICLR Poster, scores 6,5,6,6, avg 5.75): Novel perspective on safety, thorough evaluation. The current paper has a more concrete defense and clearer mechanism. Above this anchor.
- "Weak-to-Strong Jailbreaking" (Rejected, scores 6,6,6,3,6, avg 5.4): Attack requiring model access, weak defense. The current paper has a more complete defense story and more rigorous evaluation. Well above this anchor.

The paper makes genuine contributions to an important and underexplored area (MDLM safety). The priming vulnerability is real, the First-Step GCG attack is an impactful practical contribution, and RA shows strong empirical performance. However, the overclaims about the theoretical foundation (monotonicity assumption), the tight coupling of RA to a specific contamination pattern, the high residual vulnerability at late intervention steps, and the absence of inference-time baselines prevent this from being in the "strong accept" range. It sits solidly in the "good paper, accept" territory.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>