Now I have enough context to write the final review. Let me carefully synthesize all the information.

The human reviewers gave scores of 8, 3, 6, 6 (average 5.75) and the decision was Reject. Let me now carefully consider which criticisms are valid and which should be removed.

Key issues to verify against the paper:

1. Harsh Critic point #1: "Decoupling is only instantiated via hierarchical labels" - This is a valid and substantive concern. The paper's title and abstract claim broad conceptual novelty about decoupling class labels from target concepts, but all instantiations are through CIFAR class/superclass hierarchies.

2. Harsh Critic point #2: "Theorem 3.2 doesn't justify the mechanism" - Valid concern. The theorem is a generic Lipschitz bound that doesn't specifically connect to the 3-phase schedule.

3. Harsh Critic point #3: "Gap metric conflates competing objectives" - Valid. The averaged L1 gap treats UA/RA/TA/MIA as equally weighted and equally directed, but they have different units and directions.

4. Harsh Critic point #4: "Over-claiming generality" - Valid concern. TOFU/diffusion experiments are thin.

5. Harsh Critic point #5: "τ and β only heuristically justified" - Valid but more of a minor/moderate concern.

Now let me check the human finder weaknesses:

W1 (Unclear metrics for model mismatch) - Valid. The paper says UA of the retrained reference is non-zero because "evaluated with superclass label." This does need clarification.

W2 (Limited real-world motivation) - Valid. The CIFAR hierarchy examples are somewhat artificial.

W3 (Requires class info) - Valid. The paper does assume knowing the number of classes in the target concept.

W4 (Weaker performance in model mismatch) - Need to check. The UA in model mismatch is actually close to the retrained reference (which itself is non-zero), so this needs careful interpretation.

W5 (Sequential three-phase pipeline) - Somewhat valid but more of a nice-to-have concern.

W6 (Limited adversarial evaluation) - This is a reasonable point but more of a nice-to-have.

W7 (Paper organization) - Formatting nitpick, remove.

Let me now also address which harsh critic points to keep vs. remove:

- Point #1: Keep as major. The gap between "decoupling class labels from target concepts" (broad framing) and what's actually done (hierarchical label mismatch) is real and substantive.

- Point #2: Keep as moderate. The theory is weak but not fatally so.

- Point #3: Keep as major. The Gap metric really does conflate competing objectives.

- Point #4: Keep as moderate. Over-claiming is real but not fatal to the core contribution.

- Point #5: Keep as moderate but downgrade to minor. Heuristic hyperparameters are common.

Now, let me calibrate the score. Looking at the human reviews:
- Same paper (OHOmpkGiYK.md): 8, 3, 6, 6 → Reject
- Concept forgetting (2L7KQ4qbHi.md): 3, 1, 3, 5 → Withdrawn/Reject (weaker concepts, simpler method)
- Deep unlearning (pUOesbrlw4.md): 5, 5, 3, 8 → Reject (reasonable ideas, implementation issues)
- Blind unlearning (KEeTRb8GLf.md): 3, 3, 6, 3, 3 → Reject (setting issues)
- Unlearn and Burn: 3, 8, 8 → Accept Poster (novel setting + effective demonstration)
- Unlearning via Sparse: 5, 5, 6, 5 → Reject
- Effective Evaluations LLM: 6, 5, 8, 3 → Accept Poster

This paper has genuine contributions: the taxonomy of mismatch settings is useful, the TARF algorithm works reasonably well, and the empirical scope is broad. But it has significant conceptual overclaiming (hierarchical labels ≠ decoupled concepts), weak theoretical grounding, and a problematic evaluation metric. These are real but not fatal weaknesses.

The paper is not bad enough for a score below 4, but the overclaiming and evaluation issues prevent it from being above 5. Given the comparison papers and that human reviewers averaged ~5.75 but recommended reject, I'll assess accordingly.

Let me write the final review.</think>## Summary

This paper identifies an under-explored gap in class-wise machine unlearning: the target concept to be unlearned may not coincide with the training class labels. It formalizes three new "mismatch" settings (target, model, data) beyond the conventional all-matched scenario, analyzes challenges via a "representation gravity" theorem linking representation distance to forgetting dynamics, and proposes TARF, a three-phase framework combining annealed gradient ascent on forgetting data with target-aware selective gradient descent on retaining data.

## Strengths

- **Novel problem formulation:** The taxonomy of four settings (all-matched, target mismatch, model mismatch, data mismatch) with explicit set notation (Table 1) is a genuine conceptual contribution. It identifies a real limitation of prior class-wise unlearning work, which assumes D_f = D_t, and provides a clean formalization that future work can build on.

- **Insightful empirical diagnosis:** The analysis of why existing methods fail under mismatch (Figure 2)—insufficient representation when LD ≺ LT, entangled features when LT ≺ LM—is supported by empirical evidence and provides actionable intuition. The "representation gravity" concept, even if theoretically loose, captures a real phenomenon.

- **Effective algorithm with broad empirical validation:** TARF achieves the lowest Gap in 6/8 main settings (Table 3) and particularly excels in the challenging mismatch scenarios. Experiments span CIFAR-10/100, TinyImageNet, ImageNet-1k, LLMs (TOFU), and diffusion models, with multiple architectures (ResNet-18, VGG-16bn, WideResNet-50). This breadth exceeds typical unlearning papers.

- **Fine-grained evaluation in model mismatch:** Table 2 evaluates forgetting within superclasses (UA-F vs. UA-R) rather than just coarse UA/RA, which properly assesses whether the model distinguishes forgetting from retaining subclasses—a useful contribution.

## Weaknesses

### Major:

- **Gap between broad conceptual framing and narrow experimental instantiation:** The paper's title, abstract, and introduction frame "decoupling the class label and the target concept" as handling realistic unlearning requests involving privacy, copyright, harmful capabilities, etc. However, every core experiment instantiates "concepts" via known label hierarchies (CIFAR class ↔ superclass). No experiment tests genuinely unlabeled or attribute-level concepts (e.g., "forget all images containing red cars" without a corresponding label). The real contribution is a training procedure for *hierarchical label mismatch*, which is narrower and more specific than the paper's grander claims. The TOFU and diffusion experiments (Table 5, Figure 6) are too lightly specified to bridge this gap—they are proof-of-concept sketches rather than substantive evidence of generality.

- **The "Gap" evaluation metric conflates competing objectives and masks trade-offs:** The primary metric across all tables is an unweighted L1 average of absolute differences between the unlearned model and retrained reference on four metrics (UA, RA, TA, MIA) that encode opposing goals (lower UA is better; higher RA/TA is better). This treats a 1-point UA gain as equivalent to a 1-point RA loss, and a 1-point MIA change as equivalent to a 1-point TA change, with no task-specific justification. The paper also acknowledges that the retrained baseline itself has non-zero UA in model mismatch settings (because of superclass evaluation), yet still uses Gap mechanically. This makes it difficult to assess whether TARF's small Gap advantages reflect genuine superiority or simply favorable averaging of trade-offs.

### Minor:

- **Theoretical contribution is largely narrative:** Theorem 3.2's bound (Eq. 2) is a generic Lipschitz-smoothness argument that loss changes on one subset are proportional to representation distance from another. It provides qualitative intuition but does not yield concrete, testable predictions about the specific TARF 3-phase schedule, the choice of annealing, the β threshold, or the binary mask τ. The connection from Assumption 3.1 → Theorem 3.2 → Definition 3.3 → the actual algorithm is a chain of heuristic steps, not a derivation. The paper presents this as if the theory underpins the method, but it mainly retrofits intuition.

- **Assumption of known target concept class count:** Section 2 explicitly assumes "the number of classes in D_un belonging to the target concept is known in target mismatch forgetting." This is a practical limitation that is acknowledged but underplayed. In realistic scenarios, the whole point is that users provide only partial identification—knowing exactly how many classes the target spans seems to require substantial prior knowledge.

- **Hyperparameter sensitivity of the three-phase schedule:** TARF introduces several interdependent hyperparameters (k, t₀, t₁, T, β). Figure 7 (left) shows meaningful sensitivity to k, and the β threshold for target identification (Phase I) is set via a top-10% heuristic without systematic sensitivity analysis. The paper provides some guidance in Appendix E, but no principled or adaptive selection method.

- **Comparison fairness in model mismatch on UA:** In model mismatch (Table 3), TARF achieves UA of 91.11% (CIFAR-10) vs. the retrained reference's 87.76%, while GA achieves 5.76%. Since UA is the accuracy on the forgetting set, *lower* is typically "more forgetting." The retrained reference's high UA reflects that it was trained on superclasses (so subclass-level forgetting is incomplete). TARF's even-higher UA means it forgets *less* at the instance level, which makes sense under the superclass evaluation—but this nuance is not clearly explained, and the Gap metric obscures it.

### Trivial:

- The notation L₁ ≺ L₂ and subscript-heavy definitions (D_fr, D_ar, etc.) add some reading difficulty, but are adequately defined in Table 1.

## Nice-to-Haves

- Testing on genuinely unlabeled target concepts (e.g., attribute-based groupings that don't correspond to any label hierarchy) would substantially strengthen the claim of generality.
- Providing an adaptive or principled method for setting β and k (e.g., via validation) rather than manual tuning would improve practical usability.
- Adding a relabeling baseline (relabel data to match the target concept and then fine-tune/retrain) would clarify whether the mismatch scenarios truly require TARF-style solutions vs. simpler re-labelling approaches.
- Including error bars / multiple runs in the main tables (stated to be in Appendix F.7) should be in the main text for the primary comparisons.

## Removed Points

- **"Paper is too long / too many appendix pages"** (Human Reviewer 2): This is a formatting/style nitpick. Conference page limits are enforced by the program, not reviewers.

- **"CIFAR-10 does not natively include superclass information"** (Human Reviewer 4): The paper explicitly states (Section 4.1) that "the coarse-to-fine label structure of CIFAR-10 is obtained by grouping based on semantic proximity (Dhakad et al., 2024)," addressing this directly.

- **"Comparison against re-labelling baseline"** (Spark reviewer): While this would be informative, the paper's primary claim is about *approximate* unlearning (avoiding retraining), and re-labelling still requires retraining. The comparison would be nice but is not a core flaw.

- **"Limited adversarial evaluation / privacy guarantees"** (Human Finder W6): This is outside the paper's stated scope, which is about approximating a retrained reference model. Providing formal privacy guarantees or defending against strong adversarial attacks is a separate research direction.

- **"The three phases should be more tightly integrated"** (Human Reviewer 2 / Human Finder W5): This is speculative design feedback without evidence that integration would help; the current design already blends phases II (ascent + descent simultaneously).

- **"Computation is not the main problem with exact unlearning—data access is"** (Human Reviewer 2): The paper discusses both data access and computational cost as motivations. This is a minor framing disagreement, not an error.

## Novel Insights

The paper's identification of "representation gravity" as a diagnostic tool for target identification in mismatched unlearning is genuinely novel. The observation that gradient ascent on forgetting data can be *leveraged* (rather than merely mitigated) to identify semantically related data in the remaining set (Phase I) is an elegant reversal of the typical concern about gradient ascent's collateral damage. This insight—that the unwanted spillover effect IS the signal for identifying what else to forget—deserves attention beyond the paper's specific algorithmic contribution.

## Suggestions

- Narrow the conceptual framing to match what's actually demonstrated: the paper makes a real contribution on *hierarchical label mismatch in unlearning*, but overclaims by equating this with full decoupling of concept and label. Tightening the framing would strengthen rather than weaken the paper.
- Decompose the "Gap" metric or present UA, RA, TA, MIA separately with explicit discussion of trade-offs, rather than relying on a scalar average as the primary claim of superiority.
- Clarify UA computation in model mismatch: explain that UA here reflects superclass-level evaluation where the reference retrained model also has non-zero UA, and discuss whether this is the right evaluation setup.

## Score and Decision

Calibrating against similar papers:
- **OHOmpkGiYK (this paper, human scores: 8,3,6,6, avg 5.75)** — same paper, scored by humans
- **2L7KQ4qbHi (concept forgetting, scores 3,1,3,5, avg 3)** — much weaker, ill-posed problem, simple method → Reject
- **KEeTRb8GLf (blind unlearning, scores 3,3,6,3, avg 3.75)** — setting issues, method not convincing → Reject  
- **pUOesbrlw4 (deep unlearning, scores 5,5,3,8, avg 5.25)** — interesting idea but significant methodological gaps → Reject
- **TLBPjECC5D (sparse unlearning, scores 5,5,6,5, avg 5.25)** — incremental, limited evaluation → Reject
- **5xxGP9x5dZ (unlearn and burn, scores 3,8,8, avg ~6.3)** — novel threat model, effective demonstration → Accept Poster
- **HaX48yksVL (unlearning fails on poisoning, scores 6,6,6,6, avg 6)** — solid empirical study, clear takeaways → Accept Poster

This paper is above the clearly-weak papers (~3-4 range) but below the accepted papers (~6+ range). It has genuine contributions—the taxonomy of mismatch settings is novel and useful, TARF works well empirically—but the overclaiming (hierarchical labels ≠ decoupled concepts), weak theoretical grounding, and problematic evaluation metric prevent it from reaching acceptance quality. The gap between the broad conceptual pitch and the narrow instantiation is the most damaging issue. The paper would be significantly stronger with a more honest framing.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>