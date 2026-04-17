---
job_id: 12c42c7f-ea4b-4065-bce8-5a09eecc7e50
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: YADyx3ieUX.pdf
paper: The Myth of the Box: Causation and Comprehensibility in Neural Network Behavior
main_score_norm: 0.2
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is a conceptual / philosophical analysis of causation, opacity, and explainability in neural networks, directly engaging with causal reasoning, interpretability, and societal considerations around trust in AI, which are all explicitly within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is written in clear English, has an abstract, an introductory section, a body that develops the main argument, and a “Three Consequences” section that functions as a discussion/conclusion. It is a position/philosophy-of-ML paper, so lack of empirical sections is not itself a structural defect. However, as discussed below, its substantive scientific contribution to ICLR is limited.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, instructions to LLM reviewers, or other manipulative content.

---

# Expected Review Outcome:

## Summary

The paper argues that the common description of neural networks as “black boxes” relies on a mistaken assumption that causal continuity necessarily entails “correlative continuity,” i.e., the existence of intermediate features that can in principle be individuated as causes of specific outputs. Using a thought experiment involving a wobbling lump of clay on a potter’s wheel and the “Secret Owls” / “subliminal learning” style setup (Cloud et al., 2025) as examples, the author claims there can be cases where distal causal relations are clear but no meaningful intermediate correlates exist, even in principle. The paper concludes that for at least some neural network behaviors, there is no hidden internal structure to be “opened,” which alters how we should talk about opacity, trust, and explainability.

## Strengths

1. **Clear articulation of a conceptual target and structure of the argument.**  
   The paper is unusually explicit about the central assumption it is challenging: that for any distal causal relation between \(f_j(z_i)\) and \(f_m(z_k)\) there must exist intermediate correlates in the system (Section 2, particularly around Page 4–5). The desiderata for a counterexample to “correlative continuity” are carefully spelled out in Section 2.1, which makes the philosophical move transparent and easy to follow.

2. **Engaging and well-written thought experiment.**  
   The potter’s wheel / wobbling clay example (Section 2.2, Pages 5–6) is intuitively compelling and nicely motivated. It does a good job of illustrating how a system can carry causal structure over time without there being any obviously individuable local feature at the intermediate time \(t_2\) that “is” the future wobble. This is a sharp, memorable way to get readers to question default intuitions about mechanisms and intermediate causes.

3. **Interesting connection to a concrete LLM phenomenon (Secret Owls).**  
   The application to the “subliminal learning” scenario of Cloud et al. (2025) (Section 1.3 and revisited in 3.1) is timely. Framing that setup as a case of distal causal continuity from the teacher’s “owl preference” to the student’s “owl preference,” carried by apparently semantically meaningless 3‑digit sequences, is a smart way of motivating the philosophical point in an ML-relevant context.

4. **Potentially valuable conceptual clarification for XAI/trust debates.**  
   The discussion in Sections 3.2 and 3.3 usefully separates two notions: (i) epistemic opacity due to hidden or inaccessible intermediate causes versus (ii) ontological limits where such intermediate features simply do not exist. This could help sharpen arguments in the trust/transparency literature that currently lean heavily on an undifferentiated “black box” metaphor.

5. **Good command of philosophical literature on causation and explanation.**  
   The paper situates its main move relative to standard topics in philosophy of science and causation (e.g., Salmon, Psillos, discussions of distal vs proximal causation, sneezing example on Page 7). While the coverage is not exhaustive, it is sufficient to show that the author is not reinventing elementary points and is instead making a more specific claim about correlative continuity.

## Weaknesses

1. **Very limited technical or methodological contribution for an ICLR main-track paper.**  
   The paper is essentially a philosophy-of-science essay with no formal definitions, no mathematical development, and no empirical or algorithmic component. There is no attempt to formalize “correlative continuity” within a causal inference or representation-learning framework (e.g., in terms of structural causal models, coarse-graining, or abstraction of internal representations). For instance, when referring to features \(f_j(z_i)\) and \(f_m(z_k)\) as in Section 1.3 and Section 2, the paper never specifies what class of mappings \(f_j\) it is allowing, whether these are measurable functions of internal states, or how they relate to standard notions like causal variables or latent factors. For ICLR’s main track, this lack of formalization means the central claim cannot be evaluated or used as a technical tool by the ML community.

2. **Core metaphysical claim is controversial and under-argued.**  
   The key thesis is strong: that there exist cases of causal continuity without “individuatable” intermediate correlates, even in principle, and that this is an ontological rather than epistemic limitation (Section 2.3, Page 7: “Even an omniscient god could not identify a feature in the still clay at \(t_2\)….”). This conclusion is drawn almost entirely from a single stylized example of wobbling clay plus some verbal commentary. The argument does not engage with alternative readings, such as:  
   - the idea that the microphysical configuration at \(t_2\) encodes a dynamical susceptibility that *is* a correlating property at a finer level of description;  
   - or that the “holistic form of the clay” is itself a high-dimensional feature that can be treated as a causal variable.  
   Instead, the paper largely asserts that treating “the state of the clay” as a causal variable is explanatorily unsatisfying, and from this infers that no intermediate causal correlate exists. That move is philosophically contentious and insufficiently defended, especially given existing work on causal abstraction and multi-scale causal modeling.

3. **Misalignment with how “black box” is used in the ML / XAI community.**  
   The paper’s target is the idea that opacity is due to *nonexistent* intermediate correlates. In practice, most ML and XAI work treats “black box” as shorthand for (i) extremely high-dimensional non-linear mechanisms that are hard to probe and summarize, (ii) difficulty of mapping internal activations to human concepts, or (iii) difficulty of reverse-engineering mechanistic computations. This does *not* require the strong metaphysical assumption that, for every distal cause, there must exist a single, crisp intermediate “owl” variable in the model weights or activations.  
   The quote from Dwivedi et al. (2023) in Section 1 does not clearly commit to such an assumption; it is compatible with a purely epistemic notion of difficulty. As a result, even if the correlative continuity fallacy were real in the sense the author suggests, it is not clear that it underpins the bulk of current interpretability or trust-related discourse. This weakens the claimed impact: the “myth of the box” may be more of a strawman than a live assumption widely held in ML.

4. **No engagement with the technical literature on causal abstraction, mechanistic interpretability, and internal causal structure in neural networks.**  
   The paper heavily cites general XAI/trust/ethics work but omits dense, directly relevant technical literatures that explicitly *formalize* questions about how high-level causal features relate to neural network internals. For example:  
   - Geiger et al., “Causal Abstractions of Neural Networks” (2021) and Geiger et al., “Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability” (2025) formalize when a high-level causal model can be faithfully implemented by a neural network and what it means to align internal structure with causal variables.  
   - Geiger et al., “Inducing Causal Structure for Interpretable Neural Networks” (2022) and Melander et al., “Causal Interpretation of Neural Network Computations with Contribution Decomposition” (2026) propose ways of inducing or extracting internal causal structure.  
   - Biswas & Shlizerman’s work on “Statistical Perspective on Functional and Causal Neural Connectomics” (2022, 2023) and Somvanshi’s survey on mechanistic interpretability (2026) discuss causal structure in functional networks.  
   These papers directly address whether, and how, internal causal features corresponding to high-level behavior can be individuated. The submission’s central thesis would need to be articulated in relation to this body of work, or at least acknowledge where it diverges. Its absence makes the positioning incomplete and undermines claims about widespread conceptual confusion.

5. **The Secret Owls / subliminal learning example is used in a way that may overstate what is actually established.**  
   In Section 1.3 and 3.1, the author treats Cloud et al.’s setting as a clean case where no intermediate correlates exist: the student’s owl preference is caused by the teacher’s owl preference via a dataset of 3-digit numbers that are “semantically uninteresting”. However, the argument here is purely verbal. For example:  
   - There is no attempt to characterize more precise structure in the training data (e.g., in distributional terms) that could, in principle, encode correlations with owl-related internal states, even if unintuitive to humans.  
   - It is simply asserted that “There is no feature of the set that ‘means’ ‘owl’” (Page 8), but that is exactly the point at stake; it is not distinguished from our current failure to *decode* such features.  
   The leap from “we cannot presently identify any owl-related correlate in the dataset” to “no such intermediate correlate exists even in principle” is large, and the paper does not make that step rigorous. For an ML audience, this will read as speculative rather than established.

6. **Lack of formal definitions and ambiguity in mathematical notation.**  
   The paper frequently writes things like “we have two features, \(f_j(z_i)\) and \(f_m(z_k)\), where \(f_m(z_k)\) causes and explains \(f_j(z_i)\) across some intermediary system” (Page 4), but never specifies what class of functions \(f_j\) and \(f_m\) range over, how \(z_i\) vs \(z_k\) are indexed (input vs output vs latent), or how causation over such features is to be understood (counterfactual dependence, structural equations, etc.).  
   Similarly, in Section 1.1 the notation \(f_j(y_i)\) and \(f_j(x_i)\) is used when discussing dependence of output features on input features, but no formal criterion is given for when such a feature “explains” another. The discussion is purely intuitive. For a paper whose central claim is about the relation between causation and correlation *of features*, the lack of a minimal mathematical framework (e.g., SCMs with feature maps; coarse-graining operators; equivalence classes of internal states) is a serious limitation and makes it hard to assess or generalize the thesis.

7. **No concrete implications or testable predictions for interpretability practice.**  
   Section 3 gestures at consequences for how we talk about trust, transparency, and explanation, but remains vague on practical impact. For example:  
   - Should researchers stop trying to find intermediate causes for certain behaviors in LMs because they might be ontologically absent?  
   - Are there criteria to distinguish cases where correlative continuity holds from those where it does not, in actual neural nets?  
   - How would this view reshape the design or evaluation of mechanistic interpretability methods, attribution methods, or concept-activation approaches?  
   At present, the paper largely reframes existing discomfort with opacity as “not a box at all” but does not yield actionable guidance on what to do differently in practice.

8. **Scope and target venue mismatch.**  
   While conceptual clarity around “black box” is valuable, the paper reads more like a philosophy-of-AI article appropriate for a philosophy or STS journal than an ICLR main-track submission. There is no new learning algorithm, no new theoretical result in learning or optimization, no empirical study, and no formal analysis of representation structure. For a top ML conference, this is a significant weakness: the work’s contribution is almost entirely discursive and interpretive rather than technical or empirical.

## Potentially Missing Related Work

Below are directly related works that, as far as I can see, are not cited in the submission but should be:

1. **Alexander Geiger, H. Lu, T. Icard, “Causal Abstractions of Neural Networks” (2021).**  
   This paper directly tackles how high-level causal variables relate to internal neural representations and provides a formal definition of causal abstraction. It is highly relevant to the paper’s claims about the (non)existence of intermediate correlates. It should be discussed in Section 2, where causal continuity and correlative continuity are introduced, and in Section 3.1 when interpreting internal structure in the Secret Owls scenario.

2. **Alexander Geiger, D. Ibeling, A. Zur, “Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability” (2025).**  
   This work formalizes mechanistic interpretability via causal abstraction and is directly aligned with the submission’s topic of causation and interpretable structure in neural networks. It should be discussed in relation to the “black box” characterization in Sections 1.1–1.2 and in Section 3.3 when talking about the language of opacity.

3. **Alexander Geiger, Z. Wu, H. Lu, “Inducing Causal Structure for Interpretable Neural Networks” (2022).**  
   Proposes methods to impose or recover interpretable causal structure in neural networks. Relevant to the claim that in some systems there may be no meaningful intermediate correlates, and should be contrasted in Section 3.1 where the author argues that “the owls you see are the owls you get”.

4. **R. Biswas, E. Shlizerman, “Statistical Perspective on Functional and Causal Neural Connectomics: A Comparative Study” (2022).**  
   Provides a systemic framework for network-level causal analysis that bears directly on how one might individuate internal causal features. This should be included in the background discussion of causation in complex neural systems in Section 2.

5. **R. Biswas, E. Shlizerman, “Statistical Perspective on Functional and Causal Neural Connectomics: The Time-Aware PC Algorithm” (2023).**  
   Introduces an algorithm for causal inference in neural systems. It would be appropriate to mention this in the context of tracing causal structure in neural networks in Section 1.1 or 3.3, especially since the paper is concerned with whether such causal relations are in principle traceable.

6. **Pulvermüller, Tomasello, Henningsen-Schomers, “Biological Constraints on Neural Network Models of Cognitive Function” (2021).**  
   This paper discusses constraints on neural representations and their interpretability. It could enrich Section 2.1, where the author distinguishes between systems like brains, clay, and neural nets, by showing how biological realism affects interpretability and causal explanations.

7. **Cao & Yamins, “Explanatory Models in Neuroscience: Part 2 – Constraint-Based Intelligibility” (2021).**  
   Deals with how computational models become intelligible under structural constraints, which is closely related to the notion of correlative continuity. Should be cited in Sections 2.1 and 2.3 when discussing intelligibility and the “view from above”.

8. **Melander, Alaoui, Liu, “Causal Interpretation of Neural Network Computations with Contribution Decomposition” (2026).**  
   Directly proposes a causal interpretation scheme for neural computations. This is a central piece of related work and should be discussed in Section 3.1 as a contrasting approach to the claim that there may be “no box” to open in some neural behaviors.

9. **Somvanshi, “Bridging the Black Box: A Survey on Mechanistic Interpretability in AI” (2026).**  
   This survey summarizes various approaches to mechanistic interpretability that are directly concerned with the black-box issue. It should be integrated into Section 1.2 or 3.3 to situate the paper’s conceptual critique relative to what mechanistic interpretability actually does.

10. **Park, Choe, Veitch, “The Linear Representation Hypothesis and the Geometry of Large Language Models” (2024).**  
    Investigates how high-level concepts are represented in LLMs and thus directly bears on whether and how “owl preferences” might be encoded in internal geometry. It would be natural to reference this in Section 1.3 and 3.1 when discussing the Secret Owls example and how conceptual preferences might or might not be localized in representation space.

## Questions

1. **Formalization of “correlative continuity.”**  
   Can you provide a minimal formal definition of “correlative continuity” in terms of, for example, a structural causal model over variables \(\{Z_t\}\) and feature maps \(f_j\), \(f_m\)? As written, the thesis is intuitive but hard to pin down. A formal definition (even if high level) would help clarify what exactly your counterexample denies.

2. **Scope of the ontological claim.**  
   Are you claiming that for some systems and some explananda, *no* fine-grained description (e.g., down to microphysics) will yield intermediate correlates, or only that no *useful* or “explanatory” intermediate correlates exist at our preferred level of description? If the latter, then your position seems closer to an epistemic/pragmatic view, and the “omniscient god” claims in Section 2.3 may be too strong.

3. **Relation to causal abstraction frameworks.**  
   How does your claim interact with formal work on causal abstraction in neural networks (e.g., Geiger et al.)? Do you think those frameworks presuppose the correlative continuity you are rejecting, or can they be adapted to handle cases like the wobbling clay? A concrete comparison would significantly strengthen the paper’s relevance to current interpretability research.

4. **Empirical disambiguation in neural networks.**  
   In the LLM/Secret Owls scenario, what would count as empirical evidence *for* or *against* your interpretation that “there is no intermediate owl feature” in the dataset or in the student’s internal representations? For instance, do you think mechanistic probing or representational similarity analysis could, in principle, falsify your claim that “the explanation is complete” at the distal level?

5. **Practical implications for XAI practitioners.**  
   Suppose your thesis is accepted. What, concretely, should practitioners of mechanistic interpretability or attribution methods change in their approaches? Should they view some failures to localize mechanisms as evidence of ontological absence rather than methodological limitation? Some more explicit guidance or criteria for when to suspect “no box” would make the work more actionable.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The argument is coherent and clearly presented at a high conceptual level, but the central ontological claim is strong, controversial, and under-supported, and the absence of any formal framework or empirical analysis makes it difficult to rigorously assess.

## Presentation Rating

3: good.  
The paper is clearly written, well-structured, and readable. Philosophical points are laid out systematically, although the lack of formal definitions for key notions like \(f_j(z_i)\) and “correlative continuity” limits precision.

## Contribution Rating

1: poor.  
The work raises an interesting conceptual point and could stimulate discussion, but it does not offer new algorithms, theory, or empirical findings, and it insufficiently engages with the existing technical literature on causal structure and interpretability in neural networks. The contribution is mainly discursive and not at the level expected for ICLR main track.

## Overall Rating

2: Reject, not good enough.  
While the paper is thought-provoking and nicely written, its core contribution is a philosophical re-interpretation of opacity that is not formalized, not tightly connected to current technical approaches to causal/interpretability analysis in neural networks, and reliant on contentious metaphysical assumptions. For ICLR, the lack of technical depth, empirical evidence, and formal clarity, plus incomplete engagement with key related work, outweigh the conceptual interest.

## Reviewer Confidence

4: confident.  
I am familiar with the interpretability, causal abstraction, and XAI literatures, and the paper’s claims are presented in a way that is relatively easy to assess. My evaluation could be partially revised by a more formal development and deeper engagement with causal abstraction work, but I am reasonably confident in the main judgments.