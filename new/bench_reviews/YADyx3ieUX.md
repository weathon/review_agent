## Summary

This paper argues that the widespread characterization of neural networks as "black boxes" rests on a fallacy—the assumption that causal continuity necessarily implies "correlative continuity," meaning that whenever a distal cause produces an effect, individuable intermediary features correlating with that effect must exist. Using a potter's clay analogy (where a wobble at time t₁ causes a wobble at t₃, but allegedly no individuable correlate exists at t₂), the author contends that some neural network behaviors may have no hidden intermediary features at all—they are not opaque but simply lack an inner box to open. The paper applies this framework to the "secret owls" subliminal learning phenomenon, discussions of trust in AI, and the language of opacity in XAI.

## Strengths

- **The paper identifies and names a genuine, under-examined assumption.** The claim that XAI discourse implicitly assumes every distal causal relation must be mediated by identificable intermediate correlates is correct and important to surface. The Dwivedi et al. (2023) quote is well-chosen to illustrate this tendency, and the philosophical question—whether causal continuity entails correlative continuity—is genuinely worth investigating.

- **The potter's clay example is vivid and thought-provoking.** It provides an intuitive, low-level physical system where the distinction between causal and correlative continuity can be illustrated clearly, and the contrast with the photic sneeze reflex (where we expect intermediary features to eventually be found) is pedagogically effective.

- **The "secret owls" application is well-chosen and concretizes the stakes.** The Cloud et al. (2025) subliminal learning study is a genuinely puzzling phenomenon, and framing it through the lens of correlative discontinuity opens a candidate explanation that avoids positing mysterious "encodings" of owl preference in number sequences.

- **The writing is engaging and the paper is well-structured.** The progression from problem statement through analogy to consequences is logical, and the prose is clear throughout.

## Weaknesses

### Major:

- **The central counterexample fails to establish its claim.** The clay wobble example is supposed to demonstrate that there can be causal continuity without any intermediate correlate. But the "overall state of the clay" at t₂ *does* uniquely determine the wobble at t₃—the clay's mass distribution, residual deformation, and internal stress state constitute a structural property at t₂ that correlates with and causally explains the t₃ wobble. The paper dismisses this by saying such an explanation would be "right to be unimpressed" (line ~296) and that "nothing more fine-grained than 'the state of the clay' can be picked out" (line ~317). But explanatory *dissatisfaction* is an epistemic/pragmatic judgment, not an ontological one. The fact that a holistic explanation is less illuminating than a distal-cause explanation does not establish that no intermediary correlate exists. The paper's claim that "even an omniscient god could not identify a feature" (line ~357) is asserted rather than argued for; a god who could read off the full microstructural state of the clay could certainly predict the wobble frequency. This is not merely an incidental weakness—**the entire argument that "there is no box" rests on this example**, and the example does not rule out its most natural alternative interpretation.

- **The concept of "meaningful correlate" is never adequately defined, making the central thesis unfalsifiable.** The paper requires intermediaries that "meaningfully correlate" (line ~222) or "intelligibly correspond" (line ~189) to the explanandum, but never specifies what makes a correlate "meaningful" versus trivial. Without this criterion, the claim that no correlate exists in the clay case is question-begging: the paper treats holistic structural properties as insufficient to count, which is precisely what an opponent would deny. This lack of definition also makes the thesis difficult to evaluate or test in neural network contexts—one person's "no individuable feature" is another person's "high-dimensional distributed representation that serves as the correlate."

- **The extension from clay to neural networks is analogical rather than argumentative.** The paper acknowledges the clay is "something of a special case" (line ~324) and that "the degree to which causally intermediary features can be individuated will not be binary" (line ~368). Yet the conclusion declares "there is simply no box" (line ~323) and the title proclaims "the myth of the box." Neural networks are discrete computational systems with individuable weights, activations, and attention patterns—quite unlike a homogeneous lump of clay. Mechanistic interpretability research has had documented success in identifying intermediate features that correlate with specific outputs (e.g., circuits analysis, sparse autoencoders). The paper neither addresses this body of work nor explains why the clay-case logic should transfer despite these structural differences. At best, the paper establishes a *conceptual possibility* that some features may lack intermediary correlates; it does not establish that this is a typical or important feature of neural networks.

- **The paper redefines "opacity" in a narrow, ontologically loaded way that doesn't faithfully engage with how the term is used.** In the XAI literature, "black box" typically marks **intractability of mechanistic understanding** or **lack of usable explanatory mappings**—an epistemic and pragmatic claim, not an ontological one about the existence of hidden individuable correlates. By targeting only the ontological reading and then concluding "there is no box," the paper argues against a strawman. Section 3.2 itself concedes that reframing opacity as ontological vs. epistemic "may make no ultimate difference to the trust we do, or should, have in a system" (line ~419), which significantly undermines the practical import of the "no box" conclusion.

### Minor:

- **The "secret owls" application asserts rather than establishes that no statistical correlate exists in the number sequences.** The paper claims "there is no finer-grained analysis of the data set's features available" (line ~405), but provides no formal or empirical argument for this. It is entirely plausible that subtle statistical patterns in the number sequences carry information about owl preference in the training dynamics—the kind of information that representation analysis could potentially reveal. The paper does hedge ("nothing in the above argumentation guarantees that this is the correct explanation," line ~411), but then immediately inflates this to "a very strong candidate" without engaging with competing explanations.

- **Limited engagement with relevant philosophical literature on causation and explanation.** The paper references Liang & Yang (2021) and Nathan (2023) in the bibliography but does not substantively discuss interventionist theories of causation (Woodward, Pearl), mechanistic accounts of explanation (Craver, Bechtel), or the extensive debate on distributed vs. localist representation. These literatures directly bear on whether holistic system states can serve as correlates and would strengthen or refine the argument.

- **No criteria for distinguishing genuine ontological correlative discontinuity from epistemic gaps in practice.** The paper acknowledges this is "feature-dependent" (line ~368) but provides no heuristics or diagnostic criteria for identifying which features in which neural networks exhibit genuine discontinuity versus which simply await better interpretive tools. This limits the practical upshot to a conceptual caution rather than an actionable insight.

### Trivial:

- None worth flagging.

## Nice-to-Haves

- A formal or semi-formal definition of "meaningful correlate" that makes the correlative continuity claim falsifiable
- Empirical analysis of the Cloud et al. student model's internal representations to test whether owl-correlated features exist in intermediate layers
- Engagement with mechanistic interpretability results on circuits and feature geometry as potential counterexamples
- A more moderate conclusion that reframes opacity as potentially epistemic rather than ontological, which would preserve the insight while avoiding overclaim

## Removed Points

- **Criticism that the paper is "not an ML paper" or is a "position paper" unsuitable for the venue.** While several comparison papers received this criticism, venue fitness is an editorial matter, not a substantive weakness of the paper's argument. The paper's ideas are relevant to the XAI community regardless of venue conventions.
- **Criticism that no experiments were run.** This is a philosophical/conceptual paper and demanding empirical validation as a prerequisite is scope creep. However, the *absence of engagement with empirical results that exist* (mechanistic interpretability findings) is a valid weakness, noted above.
- **Criticism that the Cloud et al. (2025) study may not exist or be available.** Per the rules, cited works are assumed to exist.
- **Criticism demanding formalization of all key terms.** While more precision would help, demanding full formalization of a philosophical argument is beyond what is standard for this type of contribution.
- **Demand for concrete trust-policy implications.** The paper explicitly scopes the trust discussion as depending on particular arguments; requiring a specific policy change is scope creep.

## Novel Insights

The identification of the "correlative continuity" assumption—intuitively, that causal chains must always have identifiable intermediate links that correlate with the endpoints—is genuinely novel and philosophically sharp. Even if the clay counterexample doesn't fully establish the thesis, the challenge it poses reveals that much "black box" discourse implicitly requires identifiable intermediary correlates, and forces clarification about whether opacity claims are ontological ("there are hidden features") or epistemic ("we can't find useful features"). This distinction is underappreciated in XAI, and the paper deserves credit for surfacing it clearly.

## Suggestions

- **Strengthen the clay example by anticipating and addressing the obvious objection:** that the holistic microstructural state of the clay at t₂ *is* a correlate. Either provide principled criteria for why holistic states don't count (which would require engaging with philosophy of causation on dispositional properties), or concede that the clay case involves a *gradient* of correlative accessibility rather than a clean absence—and rebuild the argument accordingly.
- **Moderate the conclusion.** The strongest defensible version of this paper's insight is: "Not all causal chains in neural networks need have neatly decomposable intermediate correlates; opacity may sometimes be an ontological feature of certain causal pathways, not merely an epistemic limitation." This is genuinely interesting without requiring the overclaim that "there is no box."
- **Engage with mechanistic interpretability literature directly.** Address whether successful identification of intermediate features (circuits, SAE features) is evidence against correlative discontinuity, or whether the paper's thesis applies only to specific feature types or layers.

## Score and Decision

This paper raises an interesting philosophical question—whether causal continuity implies correlative continuity in neural networks—but its central argument relies on a single informal analogy (the clay wobble) that does not convincingly establish the claimed result, as the obvious objection (that the clay's holistic state at t₂ *is* a correlate) goes unanswered. The overclaim in the conclusion ("there is simply no box") significantly outpaces what the argument supports, and the paper does not adequately engage with either the mechanistic interpretability literature that has found intermediate correlates in neural networks, or with philosophical literature on causation that addresses holistic/dispositional properties. The practical implications are acknowledged to be limited (Section 3.2). These weaknesses are comparable to those in similarly-positioned philosophical papers that received scores in the 3–5 range and were rejected (YuwxDSqNXw at 3.5, LsZxlxA9da at 4.0, dKPzWyaOsK at 3.7). The philosophical insight is genuinely interesting, elevating this slightly above the weakest comparable papers, but the argument's failure to establish its central claim is a significant deficit.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>