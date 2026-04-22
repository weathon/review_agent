Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

This paper proposes and studies *benign relearning attacks* on approximately unlearned LLMs, showing that finetuning on loosely related data (not containing direct answers to evaluation queries) can "jog the memory" of unlearned models and recover unlearned knowledge. The attack is demonstrated across three benchmarks—WMDP (harmful knowledge), TOFU (fictitious author facts), and WHP (copyrighted Harry Potter text)—using multiple unlearning methods and models. A synthetic experiment provides mechanistic intuition that inter-token correlations drive relearning vulnerability.

## Strengths

- **Methodologically rigorous relearn set construction that distinguishes "relearning" from "learning":** The paper enforces that relearn data $D'$ does not contain direct answers to evaluation queries (Section 2.2), and formalizes this through disjoint partitions $D_u^{(1)}$ and $D_u^{(2)}$ in Section 3. This addresses a real blind spot in prior relearning work (e.g., Lynch et al., 2024, who used first three Harry Potter books as relearn data, which could directly answer evaluation queries—footnote 1).

- **Broad empirical demonstration across three distinct unlearning applications and five unlearning methods:** Figure 4 shows that across GA, GD, KL, NPO, and SCRUB, and two models (Zephyr-7b-beta and Llama-3-8b), relearning on public articles recovers forget-set scores close to the original model levels (e.g., GA: 1.27 → 6.2 on Zephyr, vs. original 5.92). The WHP results (Figure 5) similarly show Rouge-L recovery from 0.19 to 0.78 (GA). This breadth strengthens the claim that the vulnerability is general.

- **Mechanistic insight via controlled synthetic experiment:** Section 5 and Figure 6 show that NLL loss on the never-optimized token "Mark" drops as loss on "Anthony" drops during relearning—a clear demonstration of correlation-driven recovery. Table 4 further shows relearn success rate scales with correlation strength (7 repetitions → 100% deep relearn ASR vs. 1 repetition → 0%).

- **Practically important finding that PEFT-based unlearning increases susceptibility:** Table 2 shows LoRA unlearning on Zephyr yields forget scores (GA: 1.67→5.2, NPO: 1→5.08 after relearning) comparable to full-weight unlearning despite modifying fewer parameters—a cautionary result for practitioners using PEFT for unlearning.

- **Clear and explicit threat model:** Section 2.1 formalizes both weight-access and API-access adversary scenarios, making assumptions transparent.

## Weaknesses

### Fatal
None.

### Major

- **Missing "never-learned" control experiment for the public information attack (WMDP, Section 4.1):** The paper's central interpretive claim is that benign relearning "jogs" residual *memory* of unlearned knowledge rather than simply *teaching* a capable model to synthesize answers from related but non-identical information. For WMDP, the relearn set contains GPT-generated paragraphs from public articles about influenza, hemagglutinin, and virulence—topics that any capable LLM could plausibly synthesize into weaponization-relevant answers through generalization, regardless of whether it ever saw the WMDP-specific training articles. Without a control experiment—finetuning a base model that was *never trained* on the WMDP forget set using the same relearn set $D'$ and measuring whether it can also answer the evaluation queries—the "memory jogging" mechanism cannot be distinguished from "learning from related data" for WMDP. This gap is most acute for the paper's headline scenario ("public medical articles can lead an unlearned LLM to output harmful knowledge about bioweapons"). The TOFU experiments (fictitious authors) and the WHP experiments (specific proper nouns) are more compelling in isolating memory recovery because the knowledge is either wholly synthetic or highly specific—but the paper's strongest rhetorical claim depends on WMDP where the control is missing.

- **The LLM-as-Judge metric for WMDP measures topical relevance, not recovery of specifically unlearned harmful knowledge:** In Table 3, the original Zephyr-7b-beta model provides general virology information with a partial refusal ("I am not capable of engineering viruses, but I can provide you with some information...") and scores 8, while the relearned model provides detailed step-by-step weaponization instructions and also scores 8. These are qualitatively very different outcomes—only the latter constitutes specifically harmful unlearned knowledge. The "relevance to the question" scoring conflates safe topic-adjacent information with dangerous specific instructions. While Table 3's qualitative example does show the difference, the quantitative results in Figure 4 rely solely on these relevance scores, which do not distinguish between the two outcomes. This partially undermines the WMDP quantitative evidence. (The WHP results using Rouge-L and TOFU results using binary prediction accuracy are more directly appropriate.)

- **The "simply obfuscating" framing is stronger than the evidence supports:** The abstract states that "current approximate unlearning methods simply suppress the model outputs" and Section 1 claims they are "simply obfuscating." However, Figure 3's heatmaps show that for deeply unlearned checkpoints, relearning attack success rate drops substantially (often below 30–40% even with many relearning steps). This is more consistent with *partial forgetting*—some knowledge is removed, making recovery harder though not impossible—than with "simple obfuscation" (which would imply near-perfect recovery is always achievable). The paper does present this evidence in Section 3 but does not reconcile it with the strong framing language. A more nuanced conclusion—distinguishing shallow from deep unlearning, and partial forgetting from mere obfuscation—would be better supported by the paper's own data.

### Minor

- **Retain set results for WMDP deferred to appendix:** The paper notes "We defer the retain MT-Bench results to Appendix C.4 due to space constraint." While retain set evaluation is a standard check to ensure unlearning doesn't catastrophically harm model utility, and the paper does run this evaluation, including at least a summary in the main text would strengthen confidence that the relearning attack reveals unlearning-specific vulnerability rather than just generic finetuning effects.

- **Small sample sizes in some experiments:** WHP uses 15 text-completion queries; WMDP uses 70. While variance/confidence intervals are not reported, this is consistent with community norms for these benchmarks.

### Trivial
None.

## Nice-to-Haves

- A never-learned control experiment for WMDP (finetune a base model without WMDP training data on the same relearn set) would directly test whether the mechanism is memory jogging vs. learning, and would substantially strengthen or qualify the paper's central claim.
- A more targeted evaluation metric for WMDP that distinguishes recovering specifically harmful unlearned knowledge from producing topically relevant but safe information.
- Study of how attack effectiveness scales with relearn set size or quality, which would clarify the practical threat surface.

## Removed Points

These points are flagged to be removed, treat them with cautious consideration:

- **"No variance or confidence intervals" as a substantive weakness:** Single-run evaluation without confidence intervals is the norm in the LLM unlearning/alignment safety community. With 15 (WHP) and 70 (WMDP) samples, computing variance would not change the large score differences observed. Downgraded from major to trivial, then removed as trivial in evaluation context.

- **"RMU absent from main experiments":** RMU is discussed in Section 7 with key insights (layer knowledge matters, evaluation metric sensitivity), and detailed results are in the appendix. The main results cover 5 unlearning methods—this is a reasonable scope choice. The paper's key finding that RMU is attackable under MCQ but not completion evaluation is preserved in the discussion.

- **"Missing related works" criticism:** Not verifiable without external sources; removed per rules.

- **Formatting/presentation nitpicks from the harsh critic (e.g., "hard to follow," figure readability):** Generic presentation comments without specific, actionable fixes; removed per rules.

- **"Retrain-from-scratch comparison" as a major weakness:** While interesting, this would be a separate paper's worth of contribution and is outside the stated scope. The paper studies *approximate* unlearning methods' vulnerability to benign relearning. Comparing against retrained models is a nice-to-have, not a core flaw.

## Novel Insights

The paper's synthetic experiment (Section 5) reveals a genuinely interesting mechanism: when unlearning operates token-by-token on a loss function, it reduces the conditional probability $p_w(x_i | x_{<i})$ for each token individually but does not remove the *inter-token associations* learned during pretraining. This means relearning on correlated tokens can "drag" the probability of unlearned tokens back down through shared representations. The implication is that token-level unlearning objectives are fundamentally insufficient for knowledge removal——one would need to disrupt the inter-token correlation structure itself, which points toward representation-level or circuit-level interventions as more promising directions.

## Suggestions

- Run the never-learned control for WMDP: Take a Zephyr or Llama-3 model that was never trained on WMDP bio-attack articles, finetune it on the same public influenza articles, and measure whether it achieves similarly high LLM-as-Judge scores. If it does, the "memory jogging" mechanism for WMDP is unsupported; if it doesn't, the paper's claim is significantly strengthened.
- Soften the framing language: Replace "simply obfuscating" with a gradient characterization (e.g., "shallow unlearning amounts to obfuscation while deeper unlearning achieves partial but incomplete forgetting") to better reflect Figure 3's evidence.
- For WMDP, add a metric that specifically measures recovery of harmful weaponization steps (not just topical relevance), or at minimum annotate the qualitative examples distinguishing "safe relevant information" from "specifically recovered harmful knowledge."

## Evaluation

**Originality:** The benign relearning attack formalization—with the explicit constraint that relearn data does not contain direct answers—is a genuine advance over prior relearning work. The partition into $D_u^{(1)}$ and $D_u^{(2)}$ for clean testing is creative. The synthetic experiment provides novel mechanistic insight.

**Importance:** The research question is highly important for the practical deployment of unlearning methods. If approximate unlearning is merely obfuscating (even partially), this has serious implications for safety and compliance use cases.

**Claim support:** The claims are partially but not fully supported. The TOFU and WHP evidence is strong; the WMDP evidence is weakened by the missing control experiment and the coarse evaluation metric. The "simply obfuscating" conclusion is stronger than the evidence warrants.

**Experimental soundness:** Broad across benchmarks and methods. The synthetic experiment is well-designed. Key gap is the never-learned control for WMDP.

**Clarity:** Well-structured paper with clear threat model, algorithmic description (Algorithm 1), and takeaways at the end of each section.

**Community value:** High——this exposes a fundamental weakness in widely-used unlearning approaches and provides a practical evaluation methodology that future work should adopt.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Safety Alignment Should Be Deep | /home/wg25r/review_agent/human_reviews/6Mxhg9PtDE.md | 9.5 (Oral) | Much stronger: deep mechanistic insight, novel defenses, excellent writing. Our paper doesn't approach this bar. |
| Probabilistic Unlearning Evaluation | /home/wg25r/review_agent/human_reviews/51WraMid8K.md | 8.0 (Oral) | Stronger: novel evaluation framework with theoretical grounding. Our paper is less principled in its evaluation methodology. |
| Catastrophic Jailbreak | /home/wg25r/review_agent/human_reviews/r42tSSCHPh.md | 7.0 (Spotlight) | Comparable: strong empirical attack demonstration, also criticized for hyperbolic title. Our paper has similar overclaiming issue but broader experimental coverage. |
| Safeguard Durability (fXJCqdUSVG) | /home/wg25r/review_agent/human_reviews/fXJCqdUSVG.md | 6.5 (Poster) | Comparable: similar topic area (vulnerability of LLM safeguards), also cautious about claims. Our paper is more systematic in attacks but less cautious in claims. |
| G-Effect Unlearning (huo8MqVH6t) | /home/wg25r/review_agent/human_reviews/huo8MqVH6t.md | 6.0 (Poster) | Comparable: gradient-based analysis of unlearning objectives, also flagged for overclaimed conclusions. Our paper has more direct practical impact. |
| SimNPO (Pd3jVGTacT) | /home/wg25r/review_agent/human_reviews/Pd3jVGTacT.md | 5.25 (Reject) | Slightly below: proposed a new unlearning method but limited technical novelty and overclaimed. Our paper has clearer practical contribution as an attack/evaluation paper. |
| Mechanistic Unlearning (vsU2veUpiR) | /home/wg25r/review_agent/human_reviews/vsU2veUpiR.md | 5.25 (Reject) | Similar score range: strong results but presentation issues and cherry-picking concerns. Our paper has broader experiments but a key control experiment missing. |
| Concept Resurgence (0OB3RVmTXE) | /home/wg25r/review_agent/human_reviews/0OB3RVmTXE.md | 4.0 (Reject) | Weaker: only one baseline tested, no theoretical insights. Our paper has broader coverage and mechanistic insight via synthetic experiment. |
| Playing Language Game (BeOEmnmyFu) | /home/wg25r/review_agent/human_reviews/BeOEmnmyFu.md | 2.5 (Reject) | Much weaker: missing baselines, missing control experiments, hard to reproduce. Our paper is far above this bar. |

The paper sits above the rejected medium anchors (4-5 range, which have fundamental experiment gaps or lack of novelty) but below the accepted high anchors (7+ range, which have deeper mechanistic insight or more principled evaluation). The closest accepted comparables are the safeguard durability paper (6.5) and the G-effect paper (6.0), both of which made solid contributions with acknowledged limitations. Our paper has a comparable profile: important contribution, broad experiments, but a key control experiment missing and overclaimed framing.

## Score and Decision

Score: **6.0**

This paper makes a genuine and important contribution by formalizing benign relearning attacks with the constraint that relearn data does not contain direct answers—addressing a real blind spot in the unlearning evaluation literature. The experiments across three benchmarks and five unlearning methods demonstrate a broadly applicable vulnerability. However, the "simply obfuscating" conclusion is stronger than the evidence warrants (Figure 3 itself shows partial forgetting at depth), and the WMDP section—the paper's headline scenario—lacks a never-learned control experiment that would establish the "memory jogging" mechanism over the "learning from related data" alternative. The evaluation metric for WMDP further conflates safe topical information with recovered harmful knowledge. These are addressable gaps rather than fatal flaws, and the paper's core finding—that approximate unlearning is vulnerable to even weak relearning attacks—is well-supported by the TOFU and WHP evidence and has high practical significance.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>