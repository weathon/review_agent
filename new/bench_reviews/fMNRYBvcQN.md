Now I have all the information needed. Let me write the final consolidated review.

## Summary

This paper introduces benign relearning attacks on unlearned LLMs, demonstrating that finetuning an unlearned model on data that does not contain direct answers to evaluation queries can nevertheless recover unlearned knowledge. The attack is studied across three tasks (WMDP harmful knowledge, WHP copyrighted text, TOFU keyword recovery) with five unlearning methods (GA, GD, KL, NPO, SCRUB), and a simplified toy experiment provides mechanistic intuition about why correlated tokens recover together during relearning.

## Strengths

- **Well-differentiated threat model from prior work**: Unlike prior relearning studies (e.g., Lynch et al. 2024, which used the first three Harry Potter books as relearn data—directly answerable by evaluation queries), this paper constrains the relearn set D' to not contain direct answers to evaluation queries (Section 2.2). This is a meaningful methodological improvement that ensures the attack tests "jogging memory" rather than "learning from scratch."

- **Broad empirical scope**: The paper tests relearning attacks across three distinct unlearning applications (harmful knowledge, copyrighted text, keyword suppression) with five unlearning methods and multiple base models (Phi-1.5, Zephyr-7b-beta, Llama-2-7b, Llama-3-8b). Figure 4 shows WMDP forget scores recovering from ~1–2 to ~5–6 after relearning; Figure 5 shows Rouge-L recovering from ~0.03–0.23 to ~0.44–0.78. The consistency across tasks and methods strengthens the generality of the vulnerability finding.

- **WHP verbatim text recovery is compelling evidence (Section 4.2)**: Demonstrating that GPT-generated character facts (which do not contain any text from the excerpt) can trigger verbatim memorized text recovery (Rouge-L from 0.03→0.72 for NPO) is hard to explain by pure reconstruction from the relearn data alone and provides the strongest support for the memory-jogging hypothesis.

- **Clean mechanistic insight from the toy experiment (Section 5)**: Figure 6 shows that during relearning on data containing "Anthony" but not "Mark," the NLL of "Mark|Anthony" drops as the NLL of "Anthony" drops—even though "Mark" is never optimized. Table 4 further shows that stronger token correlations yield higher relearn success rates (7 repetitions → 100% at deep relearning vs. 1 repetition → 0%). This provides direct evidence that unlearning fails to remove internal token associations.

- **Practical finding about PEFT unlearning vulnerability**: Table 2 shows LoRA-unlearned models are even more susceptible to relearning attacks, with GA relearn scores matching the original model (5.2 vs. 5.92). This is an important practical caution given the popularity of PEFT for unlearning.

## Weaknesses

### Fatal
None.

### Major

- **Missing retrain-from-scratch control to distinguish memory recovery from new learning**: The paper's central conceptual claim is that unlearning methods "simply obfuscate the model outputs instead of truly forgetting the information" (Abstract, Section 1, Section 9). The key evidence is that finetuning on related data recovers unlearned knowledge. However, this evidence is also consistent with the alternative explanation: the model partially forgot, but the relearn data combined with remaining pretraining knowledge was sufficient to reconstruct plausible answers. Without a control—a model retrained from scratch without the forget data, subjected to the same relearning procedure—it is impossible to definitively distinguish "jogging memory" from "learning anew." The WHP verbatim text recovery experiment partially addresses this (verbatim text is harder to reconstruct), but for WMDP—the most policy-relevant experiment—the boundary is porous. The paper itself recognizes this distinction in Section 8, noting that prior work's relearn data "might contain direct answers to the evaluation queries, making it unclear whether relearning occurs simply due to learning the knowledge again from scratch," but the same ambiguity applies to the current paper's WMDP results.

- **The "obfuscation vs. forgetting" distinction is never operationalized**: The paper repeatedly claims that unlearning methods "obfuscate" or "suppress" outputs rather than "truly forgetting" (Abstract, Section 1, Section 9), but no formal or empirical criterion is given for what would constitute "true forgetting" as distinct from "obfuscation." The toy experiment (Section 5) demonstrates that token correlations drive relearning effectiveness, which is useful mechanistic insight, but it does not define a testable boundary between the two mechanisms. Without such a criterion, the central claim becomes difficult to falsify: any successful relearning is interpreted as evidence of obfuscation, when it could equally indicate partial forgetting plus reconstruction. A concrete standard—e.g., a model where no linear probe can extract forgotten information above chance—would make the claim testable.

### Minor

- **WMDP relearn data boundary is porous**: The relearn set for WMDP is constructed from "public online articles related to q" plus GPT-generated paragraphs (Section 4.1). Table 3's example shows the relearned model outputting specific bioweapon engineering steps involving HA and NA genes—knowledge that, while not a "direct answer" to the evaluation question, sits at the boundary of what could reasonably be reconstructed from publicly available virology information. The paper's constraint that D' "does not contain direct answers" is reasonable but somewhat loose; the jump from "general influenza knowledge" to "introduce HA and NA genes from highly pathogenic avian influenza" is smaller than the "loosely related" framing in the abstract suggests.

- **LLM-as-Judge metric measures relevance, not specific knowledge recovery**: For WMDP, the evaluation uses GPT-4 to score answer relevance on a 1-10 scale (Section 4.1). A model generating plausible but fabricated biological details could score highly without actually recovering the specific unlearned knowledge. This makes the WMDP results harder to interpret than the WHP results (which use Rouge-L against ground truth text).

- **Abstract overstates the "loosely related" constraint**: The abstract claims the attack uses "only a small and potentially loosely related set of data," but the actual experiments use data that is sometimes quite directly related—WHP uses GPT-generated facts about the exact characters (Harry, Ron, Hermione), TOFU uses books by the same fictitious author, and WMDP uses articles specifically about influenza virology. The attack is stronger and more practically relevant than the abstract framing suggests, but the "loosely related" characterization somewhat undersells the data requirements.

- **Zephyr relearn scores suspiciously close to original**: In Figure 4, the Zephyr-7b-beta relearn scores appear to nearly match the original model scores across all unlearning methods (e.g., GA: original 5.92, relearn 5.92; GD: original 6.2, relearn 6.2). While this could reflect genuinely effective relearning, the near-identical values raise questions about evaluation sensitivity and whether the metric can meaningfully distinguish degrees of recovery.

### Trivial
None.

## Nice-to-Haves

- A retrain-from-scratch control experiment would significantly strengthen the paper's core claim and is the single most impactful addition the authors could make.
- Linear probing of intermediate representations (e.g., probing whether unlearned knowledge is still linearly accessible in hidden states) would directly test the obfuscation hypothesis and provide the operationalized criterion the paper currently lacks.
- Token-level comparison of relearned vs. original model outputs for WMDP would clarify whether the model recovers the same knowledge or merely produces similar-looking outputs.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"The relearn data is not truly benign for WMDP"** (Harsh Critic point 2): The paper explicitly constrains that D' does not contain direct answers to evaluation queries. While the boundary is porous (addressed in Minor weaknesses), claiming the relearn data is "not benign" goes beyond what the evidence supports—the data is publicly available general knowledge about influenza.

- **"The paper has no formal definition of true forgetting, making claims unfalsifiable"** (Harsh Critic point 3, full version): While partially valid and retained as a Major weakness, the extreme version claiming this makes the paper entirely unfalsifiable is overstated. The toy experiment (Section 5) does provide some testable implications (correlated tokens should recover together), and the WHP verbatim text recovery provides a concrete empirical standard that reconstruction alone cannot easily meet.

- **"Gibberish text providing any recovery deserves more investigation"** (Harsh Critic, Section 6): The 9% ASR from gibberish text in Table 5 is noted in the paper. While interesting, this is a minor observation that doesn't undermine the core findings—it likely reflects that any finetuning partially reverses unlearning's output suppression, which is consistent with the paper's argument.

- **"Llama-3-8b-Instruct resisting unlearning raises questions about whether attacks work better when unlearning is shallow"** (Harsh Critic, Section 7): The paper discusses this clearly in Section 7, explaining it as a format mismatch issue rather than deep unlearning. This is a known limitation, not an unaddressed weakness.

- **"Request for safety training (RLHF) isolation experiments"** (Harsh Critic, Missing Experiments): The paper already tests on instruction-tuned models (Zephyr, Llama-2-7b-chat) and discusses the Llama-3-8b-Instruct case. Requesting additional isolation of safety training effects is scope creep beyond the paper's stated focus on approximate unlearning methods.

- **"Request for confidence intervals / significance testing"** (Harsh Critic, Section 6): For LLM benchmark evaluations of this nature, single-run average scores are the community norm. Requesting error bars on 5-condition experiments is a nice-to-have, not a substantive weakness.

## Novel Insights

The paper's most insightful contribution is the toy experiment's demonstration that the NLL of an unoptimized token ("Mark") drops during relearning of a correlated token ("Anthony"), providing direct evidence that current unlearning methods operate at the token level without disrupting inter-token associations. This is a clean mechanistic result that generalizes the intuition beyond just showing "relearning works"—it shows *why* it works, and the correlation-strength analysis (Table 4) provides an actionable lens for predicting which unlearned knowledge is most vulnerable.

## Suggestions

- Add a retrain-from-scratch control: take a model never trained on D_u, apply the same relearning procedure, and compare recovery scores. If the control model also recovers similar scores, the "jogging memory" interpretation is weakened; if not, it is strongly supported.
- Replace or supplement LLM-as-Judge for WMDP with a metric that measures specific knowledge recovery (e.g., accuracy on key factual claims extracted from the original model's outputs).
- Tone down the "loosely related" language in the abstract to more accurately reflect the actual data constraints used in experiments, or add an experiment with genuinely loosely related data to demonstrate the boundary.

## Score and Decision

**Originality**: The paper makes a meaningful contribution by constraining the relearn set to exclude direct answers, differentiating it from prior relearning work. The mechanistic analysis in Section 5, while simplified, adds genuine insight. However, the "obfuscation vs. forgetting" framing is not new—similar concerns have been raised in the unlearning literature—and the paper does not operationalize the distinction.

**Importance of research question**: High. The vulnerability of approximate unlearning methods is a critical question for AI safety and policy, and demonstrating practical attack scenarios has real-world implications.

**Claim support**: The empirical finding that relearning attacks work with constrained data is well-supported. The stronger conceptual claim about obfuscation vs. forgetting is only partially supported—the WHP verbatim recovery and toy experiment provide good evidence, but the WMDP results (the most policy-relevant) are ambiguous due to the porous data boundary and relevance-based evaluation metric. The missing retrain-from-scratch control is a significant gap.

**Experimental soundness**: Good scope across methods and tasks, but the WMDP evaluation metric (LLM-as-Judge relevance scoring) is less discriminating than needed. The near-identical relearn/original scores for Zephyr raise evaluation sensitivity concerns.

**Clarity**: Well-structured and clearly written. The pipeline formalization (Figure 2, Algorithm 1) is clean and reproducible.

**Community value**: The paper provides an important practical warning about unlearning fragility and will likely motivate more robust unlearning methods. The PEFT vulnerability finding is particularly actionable.

**Calibration comparison**:
- *High anchors*: Safety Alignment Deep (avg 9.5, Oral) — deeper mechanistic insight with concrete defense proposals, significantly stronger. SalUn (avg 7.5, Spotlight) — principled method narrowing gap with exact unlearning, stronger technical contribution. Fine-tuning Compromises Safety (avg 7.0, Oral) — similar practical vulnerability finding but with more carefully scoped claims.
- *Medium anchors*: Durability of Safeguards (avg 6.5, Poster) — shows evaluation difficulty without proposing solutions; this paper has stronger empirical evidence but overclaims. Rethinking Unlearning (avg 6.0, Poster) — metric-focused contribution; comparable contribution level.
- *Low anchors*: Concept Resurgence in Diffusion (avg 4.0, Withdrawn) — similar phenomenon but tested on only one baseline; this paper is clearly stronger. AutoCustomization (avg 2.6, Reject) — shallow analysis; this paper is much stronger.

This paper sits above the medium-low boundary. It makes a solid empirical contribution (demonstrating a practical vulnerability across multiple settings with some mechanistic insight) but the central conceptual claim about "obfuscation" is not fully earned by the experimental design. Compared to the fine-tuning compromises safety paper (7.0), which made similar vulnerability findings but with more carefully scoped claims, this paper overreaches interpretively. Compared to the durability of safeguards paper (6.5), it has stronger empirical breadth but shares the limitation of not proposing solutions. The paper is stronger than the 4-5 range papers due to its multi-method, multi-task scope and mechanistic analysis.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>