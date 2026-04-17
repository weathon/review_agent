Now I have a thorough understanding of the paper. Let me write the final consolidated review.

## Summary

This paper argues that the widespread characterization of neural networks as "black boxes" rests on a fallacy—the assumption that causal continuity necessarily implies "correlative continuity" (i.e., that if a feature at t₁ causally produces a feature at t₃, there must be an individuable intermediate correlate at t₂). Using the analogy of a potter's wheel (where a wobble at t₁ causes a wobble at t₃ through a stationary clay state at t₂ that allegedly contains no feature corresponding to the wobble) and the Cloud et al. (2025) "secret owls" subliminal learning study, the author argues that causation can subsist without intermediating correlates, reframes opacity as an ontological rather than epistemic matter, and concludes that "this ubiquitous box is mere myth."

## Strengths

- **Identifies a genuine and underexamined philosophical assumption in XAI discourse.** The paper correctly notes that much "black box" framing implicitly assumes that causal continuity entails the existence of individuable intermediary correlates. Subjecting this assumption to critical scrutiny is a novel and potentially valuable contribution. The paper explicitly engages with specific claims in the literature (Dwivedi et al. 2023, Zerilli 2022, Chesterman 2021) about the nature of opacity.

- **The clay/wobble analogy is vivid and pedagogically effective.** The example makes a subtle philosophical claim concrete: something causally continuous (wobble → stationary clay → wobble) may resist decomposition into identifiable intermediate causal correlates. This forces the reader to confront a genuine puzzle about the relationship between causation and explanation.

- **The paper is well-organized and clearly written.** The progression from motivating example (§1.3) to philosophical argument (§2) to consequences (§3) is logical and accessible. The distinction between epistemic and ontological opacity (§2.3) is philosophically important and underappreciated in the XAI literature.

- **Engagement with a real empirical case study.** The Cloud et al. "secret owls" phenomenon provides a concrete, recent, and genuinely puzzling example where standard explainability frameworks struggle, grounding the philosophical argument beyond pure abstraction.

## Weaknesses

### Major:

- **The central counterexample (the clay wobble) does not establish the strong ontological claim the paper needs.** The paper's core argumentative move is the claim that in the clay case, there is *no individuable intermediate correlate* of the wobble frequency at t₂—this is asserted as an ontological fact, not merely an epistemic limitation. However, physically, the stationary clay at t₂ has a well-defined mass distribution, internal stress field, and deformation pattern that jointly determine the wobble frequency at t₃. The paper explicitly acknowledges that "the holistic form of the clay at t₂ has structure" and that "the properties of this structure are causally implicated in the wobble at t₃," then denies that any feature can be "individuated as a causal correlate." But the argument for why the entire microstructural state should not count as a correlate—and why a god-like observer could not identify regions of state-space that predict wobble frequency—is asserted rather than argued. If the clay state at t₂ is merely *hard for humans to parse* but physically decomposable in principle, the example illustrates epistemic difficulty, not ontological absence. This distinction is load-bearing: the paper's conclusion that "there is simply no box" in neural networks requires the stronger ontological reading.

- **The argument equivocates between different notions of "correlate" and "feature."** At key points, "correlate" slides between a weak sense (any systematic relation between variations in two quantities) and a stronger sense (a salient, humanly graspable, low-dimensional property that can be *named* as the cause). The paper acknowledges that "the overall form of the clay" is a necessary condition carrying the causation, but denies it is a "correlate" because citing it is explanatorily unsatisfying ("someone who asked why the clay wobbled thus and not otherwise at t₃, if met with 'because of the overall form of the clay at t₂', would be right to be unimpressed"). But explanatory unsatisfyingness is an epistemic/pragmatic judgment, not an ontological one. The XAI literature that the paper targets generally does *not* claim that there must exist a simple, humanly salient intermediate feature—they claim that there exists internal structure that, if known, would make the behavior intelligible. By targeting a straw version of this claim, the paper's conclusion that the "black box is a myth" is overdrawn.

- **The extension from the clay case to neural networks is unsupported.** Neural networks are discrete, explicitly parametrized computational systems where, for a given model and input, there exists a well-defined mapping from intermediate activations to outputs. The paper provides no argument for why the clay analogy—where the relevant analogy is to continuous physical deformation—should transfer to this setting. The paper acknowledges that "to what extent we should expect there to exist intermediary features that correlate with output features will depend on the details" (§2.3), but then applies the conclusion to neural networks without analyzing those details. In the owl case, the paper asserts "there is no finer-grained analysis of the data set's features available, to either humans or gods" (§3.1), but this is precisely what is at issue: whether statistical patterns in the number sequences transmit the owl disposition. The paper does not rule out this possibility; it merely denies it.

- **The paper does not engage with the mechanistic interpretability literature.** Work on sparse autoencoders, probing, circuit analysis, and feature extraction has successfully identified intermediate correlates in neural networks. Whether these constitute the kind of "correlates" the paper discusses is exactly the question at issue, yet the paper does not address this empirical research program. This is a significant gap for a paper whose central claim is that such correlates do not exist.

### Minor:

- **The scope of the argument is ambiguous.** The paper alternates between claiming that correlative discontinuity *sometimes* occurs (§2.3: "we cannot assume ahead of time that every instance of causal continuity will also demonstrate the correlative continuity") and making much stronger claims ("the ubiquitous box is mere myth," "there is simply no box"). The former is modest and defensible; the latter is not established by the argument given.

- **The practical implications of the "myth of the box" thesis are unclear.** The paper concedes (§3.2) that reframing opacity as ontological rather than epistemic "may or may not" change trust judgments, and acknowledges (§3.3) that eliminating "opacity" from the language "in no way undermines" ongoing work in representation analysis and XAI. This raises the question: if nothing methodological changes, what is the practical import of the conceptual revision?

- **Insufficient engagement with interventionist accounts of causation.** On Woodward-style or Pearl-style interventionist accounts, if an intervention on the clay at t₁ changes the wobble at t₃, then there must be features at t₂ that mediate this effect. The paper does not address how its thesis interacts with this widely-accepted framework.

## Nice-to-Haves

- Engagement with the literature on distributed and superposed representations in neural networks (polysemantic neurons, superposition, etc.) would sharpen the argument about what counts as an "individuable" feature.
- A concrete neural network example where the paper's prediction (no individuable intermediate correlates) could be tested—even a toy model demonstration—would greatly strengthen the argument.
- Discussion of interventionist (Woodward, Pearl) accounts of causation and how the thesis interacts with them.
- Clarification of what precisely counts as "individuating" a feature, which is the linchpin of the argument.

## Novel Insights

The paper's most valuable contribution is the identification and naming of the "correlative continuity fallacy"—the assumption that causal continuity must entail identifiable intermediate correlates. Even if the clay counterexample fails to establish the strong ontological reading, the formal identification of this assumption and the demonstration that it is not self-evident is a genuine philosophical contribution that could usefully reorient discussions in XAI. The distinction between epistemic opacity (features exist but are hard to find) and ontological opacity (no features exist to find) is philosophically important, even if the paper overreaches in asserting the latter.

## Suggestions

- Either defend the strong ontological reading of the clay example against the objection that microstructural properties of the clay are individuable correlates of the wobble, or weaken the conclusion to the epistemic reading and acknowledge that this significantly reduces the paper's claim about the "myth" of the box.
- Engage with mechanistic interpretability results and explain which sense of "individuable correlate" they fail to satisfy, or concede that they provide intermediate correlates in the standard sense.
- Clarify whether the thesis is that correlative continuity *sometimes* fails (modest) or that it *typically* fails in neural networks (strong), and adjust the rhetorical conclusions accordingly.
- Provide criteria for what counts as a "feature" or "correlate" that could, in principle, be individuated, so that the thesis is empirically testable rather than merely stipulative.

## Evaluation

**Originality:** The identification of the "correlative continuity fallacy" is genuinely novel and philosophically interesting. The application of this idea to XAI discourse is original. However, the central example (clay) does not bear the philosophical weight required.

**Importance of research question:** The question of what "black box" really means and whether there are hidden correlates to find is important for the field.

**Whether claims are well supported:** This is the primary weakness. The central ontological claim—that the clay case demonstrates causal continuity without intermediate correlates even in principle—is asserted rather than argued, and is vulnerable to the obvious physical objection that the clay's microstructural state constitutes such a correlate. The extension to neural networks lacks argumentative support.

**Soundness of experiments:** N/A (conceptual/philosophical paper).

**Clarity of writing:** Clear and well-organized.

**Value to the research community:** Moderate. The philosophical clarification is useful but limited by the unsupported core claim.

## Score and Decision Calibration

I compared this paper against several calibration anchors:

- **ToUla3c9kW** (Explanatory Virtues Framework, scores 4/4/2/2, withdrawn/reject): Similarly a philosophy-of-science paper about XAI/MI. Reviewers found it lacked novelty, formal substance, and empirical contribution. Scores averaged ~3.

- **cUAhqSUfeK** (Progressive Coarse-graining, scores 2/0/2/2, reject): Purely conceptual perspective with no experiments or formal proofs. Scores averaged ~1.5.

- **CHwweyjn5z** (Reflexivity in AI, scores 0/2/2/0, withdrawn): Position paper with no technical contribution, described as "speculative" with claims "not supported by experimental evidence or formal argumentation." Scores averaged ~1.

- **Q3MisVkuTu** (Demystifying Common Beliefs in GML, scores 6/4/4/2, accept poster): Another "demystifying common beliefs" paper with counterexamples. Scored higher because it provided formal counterexamples and toy experiments, even though the contributions were partly overlapping with prior work. Scores averaged ~4.

- **GVIei1IdmC** (LLMs as Nondeterministic Causal Models, scores 6/2/4/4, withdrawn): Conceptual/philosophical paper about causal models and LLMs with a specific formal proposal. Scores averaged ~4.

This paper is more substantive and original than the purely speculative position papers (CHwweyjn5z at ~1, cUAhqSUfeK at ~1.5). It makes a genuine philosophical argument with a specific counterexample and engages with real empirical work. However, its central philosophical argument has a significant weaknesses: the clay counterexample does not convincingly establish the ontological claim it needs, and the extension to neural networks is argued by analogy rather than demonstrated. It is stronger than papers that have no argument at all, but weaker than papers like Q3MisVkuTu that provide formal counterexamples. The paper is similar in spirit to ToUla3c9kW (philosophy of XAI, no experiments) but somewhat more original in its central thesis.

I place this paper below the "demystifying" paper (which at least provided formal counterexamples) and above the purely speculative position papers. A score of 3 reflects that the paper identifies an interesting and novel philosophical question but fails to convincingly establish its central claim, which is load-bearing for the broader conclusions about XAI and trust.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>