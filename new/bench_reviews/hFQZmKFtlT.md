Now I have a comprehensive understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

This paper introduces a formal-grammar-based experimental framework to disentangle memorization ("learning by rote") from generalization ("learning with understanding") in LLMs. Using random and hierarchical grammars with controlled entropy, the authors train LLMs (Pythia-1B, Llama3-8B, Mistral-Nemo-12B) and track training/test loss dynamics, finding: (a) memorization and generalization phases overlap and are "at odds," (b) lower-entropy distributions are easier to generalize to while higher-entropy ones are easier to memorize, and (c) recollection accuracy alone cannot determine whether a model has memorized or generalized.

## Strengths

- **Formal grammar framework is a genuine methodological contribution**: Unlike natural language, formal grammars guarantee no pre-training contamination, enable generation of i.i.d. test data from the same distribution, and allow precise control over entropy and alphabet size (Section 2). This is a clean experimental setup that addresses real limitations of prior memorization studies on natural language.

- **Figure 3 provides a concrete empirical demonstration that recollection-based memorization measures are ambiguous**: Two models (trained with n=8 vs. n=64) can achieve identical training loss (~0.6 for Pythia-1B) on the hierarchical grammar while one is in the memorization phase (test loss diverges to ~0.8) and the other is still generalizing (test loss remains close to training loss). This is a specific, empirically grounded argument against the approach taken by prior memorization measures (Carlini et al., 2022; Tirumala et al., 2022).

- **Consistency across model families spanning >10× parameter range**: All major results (learning dynamics, entropy effects, sequential memorization) are replicated across Pythia-1B, Llama3-8B, and Mistral-Nemo-12B (Figures 2, 4, 5), strengthening the generality of claims.

- **Practical implications for unlearning**: The sequential memorization experiment (Section 4) suggests a concrete strategy—memorizing new data from the same distribution can trigger forgetting of previously memorized data, which has direct implications for privacy-preserving ML.

## Weaknesses

### Fatal
None.

### Major

- **The core observations largely repack well-known ML phenomena with overclaimed novelty**: The paper's central finding that "rote learning harms understanding" (claim a) is that overfitting hurts generalization—a foundational observation in ML. The sequential training result (Section 4) showing that memorizing D_train,2 causes forgetting of D_train,1 is catastrophic forgetting under distribution shift—well-studied since the 1990s. The entropy finding (claim b) that lower-entropy distributions generalize better is consistent with basic information theory: less information to learn means easier approximation. The paper frames these as "striking" (abstract, Section 1) and "surprising and unexpected" (Section 1), which significantly overstates their novelty. While the *application* of these ideas to the LLM memorization literature is useful, the underlying phenomena are not new discoveries.

- **"Understanding" is operationally just i.i.d. generalization, making the framing misleading**: The paper explicitly equates "learning with understanding" with "generalization" (Section 1: "generalization, i.e., learning with understanding"), and the motivating thought experiment (German vs. English speaker reciting German) invokes genuine syntactic/semantic understanding. However, the actual experiments only test whether models achieve low loss on i.i.d. test data from the same grammar—this is distribution learning, not understanding. For the random grammar (each token independently sampled uniformly), "understanding" reduces to learning marginal character frequencies, which stretches the term beyond any reasonable cognitive definition. No experiment tests whether models can recognize grammatically invalid strings, compositional generalize to longer strings, or manipulate grammar rules—all of which would more directly test understanding. The human cognition analogy (Section 4: "humans can both memorize poems while also being able to write new ones") is therefore unwarranted by the evidence presented.

- **The proposed memorization measure cannot decompose the claimed overlap between phases**: Equation (2) defines memorization as 1 − Loss(train)/Loss(test), which is a normalized generalization gap. The paper claims that memorization and generalization phases *overlap* (Section 3.1: "During these overlapping epochs, the total recollection of an LLM is partly with understanding and partly by rote"), but the proposed measure produces a single scalar that increases monotonically as the gap grows—it provides no decomposition of a given epoch's performance into "rote" and "understanding" components. Figure 1(b) appears to decompose accuracy into "recollection by rote" and "recollection by understanding," but this is simply training accuracy minus test accuracy vs. test accuracy—again just the generalization gap. The overlap claim thus remains unfalsifiable with the tools provided.

### Minor

- **Very small training sets may limit generalizability**: The primary experiments use n=8 or n=64 strings of length 64–72, meaning models train on only ~4K tokens for 100+ epochs. The dynamics observed in this extreme small-data, many-epoch regime may differ qualitatively from typical LLM fine-tuning or pre-training. The paper acknowledges this limitation (Section 6) but understates it.

- **The 5% threshold for declaring memorization onset is arbitrary**: The criterion that memorization starts when Loss(test)/Loss(train) > 1.05 (Section 3.1) is not justified. Different thresholds would shift the claimed epoch numbers and the extent of the "overlap" between phases.

- **The "impossibility" claim in Section 3.2 is overstated**: The paper claims it is "impossible" to estimate memorization based on recollection, but the demonstration (Figure 3) shows that recollection alone is insufficient—one also needs to measure generalization. This is essentially the observation that training loss alone doesn't determine generalization, which is precisely why test sets exist. The Figure 3 demonstration is concrete and useful, but calling it an "impossibility" overstates the case.

- **Entropy levels are not quantified numerically**: The three methods of varying entropy (alphabet size, oversampling, production rule skewness) provide qualitative variation, but the actual entropy values are not reported in the main text, making it difficult to assess the quantitative relationship between entropy and memorization/generalization dynamics.

## Nice-to-Haves

- Testing whether models can distinguish valid grammar strings from invalid ones (grammaticality judgment) would substantiate the "understanding" framing and distinguish distribution learning from structural understanding.
- Experiments at larger data scales (n=1000+, fewer epochs) would test whether observed dynamics transfer to more realistic training regimes.
- Token-level analysis showing which positions in a string are predicted via "understanding" vs. "rote" (e.g., structurally constrained vs. unconstrained positions) would make the decomposition more concrete than aggregate loss curves.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic: "The sequential training experiment is just catastrophic forgetting"** — The paper does show more than just forgetting: it shows that when switching to D_train,2, the test loss *briefly re-decreases* (re-generalization) before rising again, and that D_train,1 loss rises to match the test loss rather than just spiking. However, the core mechanism is indeed catastrophic forgetting, and the additional nuance doesn't constitute a fundamentally new phenomenon. Kept as part of the major weakness about repackaging known phenomena, but weakened from the original claim that it's "nothing new."

- **Strength Finder: "Proposed alternative memorization measure grounded in the train/test loss gap"** — This is listed as a strength, but Eq. (2) is simply the normalized generalization gap, not a novel measure. Moved to Removed Points as it conflicts with the verified major weakness that the measure is just the generalization gap.

- **Strength Finder: "Clear operational definitions for the onset of memorization"** — The 5% threshold is acknowledged as arbitrary even in the minor weaknesses. This is not a strength when the definition is unjustified and shifts results. Moved to Removed Points.

- **Harsh Critic: Request for grammaticality judgment, compositional generalization experiments** — These are reasonable suggestions but fall outside the paper's stated scope. The paper explicitly focuses on *what happens* during memorization and generalization at an input-output level, not on proving understanding. These are nice-to-haves, not core flaws. Moved to Nice-to-Haves.

- **Harsh Critic: Request for experiments at realistic data scales** — Valid concern but addressed as a minor weakness since the paper already acknowledges the limitation. The small-data regime is a feature (enables controlled observation), not purely a bug.

- **Harsh Critic: Request for quantitative entropy analysis** — Valid but minor. The qualitative trends are clear from Figure 5; numerical values would strengthen but not change the conclusions.

- **Harsh Critic: Missing related works** — Cannot verify existence of specific works; removed per rules.

## Novel Insights

The paper's most valuable insight is not the "rote vs. understanding" framing itself but rather the concrete demonstration that the LLM memorization literature's reliance on recollection-based measures creates a fundamental ambiguity: two models that appear identical on training data can be in qualitatively different learning phases (Figure 3). This is a specific, actionable critique of existing methodology that goes beyond the general observation that overfitting exists. However, the paper does not follow through on this insight—the proposed alternative measure (Eq. 2) is the generalization gap, which is well-known and impractical in settings where test data from the same distribution is unavailable (as the paper itself acknowledges).

## Suggestions

- Tone down the "striking"/"surprising" language and the human cognition analogies. Present the findings as controlled empirical confirmations of expected phenomena in the specific context of LLM memorization measurement, which is valuable without being oversold.
- Replace or supplement Eq. (2) with a measure that can actually decompose the "overlap" phase rather than just computing the generalization gap. For instance, compare model performance on strings that differ from training data only in structurally predictable vs. unpredictable positions.
- Report numerical entropy values for each grammar variant to enable quantitative analysis of the entropy–memorization relationship.
- Add at least one experiment with a larger training set (e.g., n=512) and fewer epochs to test whether the phase dynamics persist at more realistic data regimes.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Emergent properties with repeated examples | xrXci5YGm7 | 5.50 | Similar: memorization/generalization on synthetic data, criticized for known phenomena. This paper has a cleaner framework but overclaims more. |
| Meta-OCL | I7kpf3mZ4n | 5.25 | Similar: repackaging known phenomena with new framing. This paper has a stronger methodological contribution (formal grammar framework) and clearer experiments. |
| Think or Remember? | w4gkS9RsWh | 4.50 | Similar topic but weaker execution. This paper is stronger due to the controlled framework and Figure 3 demonstration. |
| Mitigating Memorization in LMs | MGKDBuyv4p | 7.33 | Much stronger: practical methods, comprehensive evaluation, new tools (TinyMem). This paper is below this level—it identifies phenomena but doesn't provide solutions. |
| On inherent limitations of GPT/LLM | JNZ3Om6NPS | 1.67 | Far weaker: incoherent presentation, unsubstantiated claims. This paper is clearly above this level. |
| Learning Dynamics of LLM Finetuning | tPNHOoZFl9 | 8.00 | Much stronger: novel theoretical framework with practical implications. This paper's contributions are more observational than explanatory. |

This paper sits in the lower-medium range. It makes a genuine methodological contribution (formal grammar framework) and has one strong empirical finding (Figure 3), but the core observations are largely repackaged known phenomena with overclaimed novelty, the "understanding" framing is misleading, and the proposed measure doesn't deliver on its promise. It's above the low-scoring papers (real framework, clear writing, concrete experiments) but below the medium-scoring acceptance threshold papers (less novelty than "Emergent properties," which itself was rejected). Score: 5.0.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>