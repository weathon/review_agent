## Summary
The paper argues that visual afterimages are cortical rather than retinal in origin, using the phenomenon of perceiving the physiological blind spot as an afterimage to localize the first-stage substrate to V1 layer 4. Building on this, it proposes a broad cortical architecture in which superficial layers (L2/3) are feedforward, deep layers (L5/6) are feedback, and middle layer 4 serves as short-term memory.

## Strengths
- **The paper makes a concrete, distinctive anatomical-theoretical link rather than staying at the usual vague “cortical involvement” level.** In Section 2.3, it ties the blind-spot-afterimage argument specifically to the known representation of the blind spot in V1-L4, yielding a sharper claim than generic statements that afterimages are “in the brain.”
- **It revives and synthesizes an unusual historical phenomenology in a way that is genuinely central to the thesis.** The La Hire–Purkinje phenomenon and the Franklin effect are not decorative historical material here; they are used as the main evidential bridge from phenomenology to neural interpretation.
- **The paper offers a clear, falsifiable architectural hypothesis.** The proposed assignment “L2/3 feedforward, L4 STM, L5/6 feedback” is easy to understand and potentially testable, even though it is not yet validated here. Figure 5 crystallizes the intended claim at the level of cortical computation.

## Weaknesses
### Fatal
- **The submission does not substantiate its central inferential leap from blind-spot phenomenology to “V1-L4 is the neural site for afterimages,” yet that claim is used as the foundation for the rest of the paper.** The paper’s key move is: blind spots can be seen as afterimages; blind spots are represented in V1-L4; therefore afterimages are localized to V1-L4. But Section 2.3 only provides correlational reasoning, not a causal or exclusionary argument. The text explicitly claims that these findings “decisively and precisely pinpoint the first-stage neural substrate of afterimages to V1-L4,” which is much stronger than the evidence provided. Even if the phenomenon is real and interesting, the paper does not rule out multi-stage contributions or establish that V1-L4 is the locus of storage rather than an early representational stage.
- **The claimed “computational architecture of the human brain” is not actually developed as a computational model.** Despite the title and repeated use of “computational architecture,” the paper offers no formalization, no dynamical mechanism, no learning rule, no simulation, and no computational test of the proposed layer roles. Section 4 moves from “afterimages are neural persistence” to “afterimages are STM” to “L4 is STM” and then to the architecture in Figure 5, but these transitions are conceptual assertions rather than computational derivations. For an ICLR paper, this is a fundamental mismatch between the claimed contribution and what is delivered.

### Major:
- **The paper overgeneralizes from a specific visual phenomenon in V1 to a universal cortical architecture.** The evidence discussed concerns afterimages and blind-spot representations in early visual cortex, yet the abstract and conclusion generalize to “the computational architecture of the brain” and to “each cortical area.” The manuscript does not provide evidence that the same role assignment holds outside the visual system, nor does it address obvious scope limitations. This makes the main claim read as substantially broader than the paper can support.
- **The identification of afterimages with short-term memory is underdefined and insufficiently justified.** Section 4 says that because afterimages are cortical and persistent, they “should better be conceived as visual STM.” But persistence alone is not enough to establish a memory mechanism in the computational sense. The paper does not define STM operationally, specify how maintenance occurs in L4, or explain what properties distinguish this proposed STM from sensory persistence or adaptation. Since the architecture depends on L4 being a memory substrate, this conceptual gap matters directly.
- **The manuscript provides no new controlled empirical evidence for its central observational premise.** The “rediscovery” of the La Hire–Purkinje phenomenon is presented as important motivation, but the paper does not report a modern psychophysical experiment quantifying reliability, duration, subject variability, or experimental controls. As written, the empirical basis is a historical synthesis plus the authors’ own observations, which is too thin for the strength of the claims being made.
- **The treatment of positive and negative afterimages remains too assertive relative to the evidence presented.** Section 3 uses the Franklin effect to argue against distinct mechanisms for positive and negative afterimages and then adopts a shared-persistence account. The phenomenon certainly motivates a unified account, but the paper does not actually provide a mechanistic explanation of the polarity reversals or show that alternative multi-stage explanations fail. Thus the conclusion that positive and negative afterimages “share the same neural substrate” is plausible as a hypothesis but not established here.

### Minor
- **The paper’s framing is often more absolute than the evidence warrants.** Phrases such as “the Retinal View is erroneous and only the Brain View is correct” and “decisively and precisely pinpoint” overstate what is defended in the manuscript. A more scoped presentation as a hypothesis or reinterpretation would be better aligned with the evidential level.
- **The proposed architectural role of L4 lacks a mechanism consistent with the paper’s own level of analysis.** Even setting aside the need for a full model, the manuscript should at least explain how L4 could maintain state over time and support the claimed memory function, rather than only naming it as STM.

### Trivial

## Nice-to-Haves
- Add a simple computational instantiation of the proposed laminar architecture, even at a toy level, to show what “L4 as STM” means operationally.
- Run a controlled psychophysical study of blind-spot afterimages with fixation control and time-decay measurements.
- Narrow the scope of the claims to visual cortex unless the paper can justify extension to the entire cortex.
- State falsifiable predictions that would distinguish this account from other cortical or multi-stage explanations of afterimages.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper ignores alternative related theories/models.”** I removed this as a main weakness because I cannot verify missing related work beyond what is in the manuscript, and the review should not speculate about uncited literature.
- **“The paper is weak because comparisons are unfair / missing against specific external methods.”** Not applicable here: the paper does not present a benchmarked algorithmic comparison setup.
- **Generic strengths such as “the paper is well-written,” “the topic is important,” or “the experiments are extensive.”** These are too generic or inapplicable.
- **Pure reproducibility nitpicks about missing implementation details.** There is no implemented model; the substantive issue is absence of a computational model, not omitted hyperparameters.
- **Claims that the paper’s concerns about cited models, references, or historical sources are unverifiable.** Such concerns are disallowed and not evidence-based.
- **Some overly strong reviewer assertions about established neuroscience facts contradicting the paper.** For example, statements that specific laminar physiology “contradicts” the proposal are stronger than what can be verified from the paper alone. The more defensible criticism is that the paper does not justify the L4-memory claim, not that the opposite has been conclusively proven here.

## Novel Insights
The strongest synthesis across the reviews is that the paper has an unusual asymmetry: its most interesting contribution is not the grand cortical architecture, but the narrower observation that blind-spot afterimages may provide an anatomically anchored probe of subjective visual persistence. If the paper were reframed around that phenomenon as a hypothesis-generating bridge between entoptic perception and laminar neuroscience, it would read as an intriguing conceptual contribution. The current version weakens itself by escalating that insight into a universal cortical architecture and a definitive localization claim without the computational or empirical support needed to sustain those leaps.

## Suggestions
- Reframe the paper around a **scoped hypothesis**: blind-spot afterimages suggest a cortical contribution, plausibly involving V1-L4, rather than claiming decisive localization.
- Either **add an actual computational model** of the proposed laminar roles or substantially soften the “computational architecture” claims.
- Define **short-term memory** operationally and explain what dynamical or circuit mechanism in L4 is supposed to instantiate it.
- Add at least one **modern empirical validation**, ideally a controlled psychophysical experiment on blind-spot afterimages.
- Separate clearly what is **evidence**, what is **inference**, and what is **speculation**, especially in Sections 2.3 and 4.
- Narrow claims about general cortical organization unless the manuscript can justify extension beyond early visual cortex.