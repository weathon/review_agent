## Summary
This paper proposes **World modeling through Lie Action (WLA)**, a multi-environment world-modeling framework that combines a slot-based object-centric autoencoder with Lie-structured latent dynamics. The core idea is to encode transitions as continuous latent generators parameterized by scaling/rotation components, then adapt external actions to these latent dynamics through a learned controller interface. Empirically, the paper shows promising results on ProcGen and a robotics video dataset, with strong gains over the reported Genie setup, and qualitative demonstrations of interpolation/composition on PHYRE.

## Strengths
- **The paper has a genuinely distinctive modeling idea:** it combines object-centric slots with a Lie-structured latent transition model, where the latent transition matrices are explicitly parameterized through continuous generators \(A(t)\) and exponentials \(M_{t,\delta}\) (Sec. 3.1, Eq. 4–5). This is a more principled inductive bias for continuity/composition than a generic black-box latent dynamics model.
- **The formalization of the controller interface problem (CIP) is specific and useful to this paper’s contribution.** In particular, the distinction between unstructured and structured CIP (Sec. 2.2) clarifies how the method uses unsupervised latent action discovery first and then learns a task/environment-specific controller adaptation.
- **The multi-environment training setup is meaningful and nontrivial.** The paper trains “a *single common model* across all environments” on ProcGen (Sec. 6.2), rather than one model per environment, which is aligned with its inter-environmental motivation.
- **The reported ProcGen results are strong on the paper’s chosen metrics.** In Table 2, WLA improves substantially over the reported Genie setup across all 8 seen ProcGen environments in PSNR and \(\Delta_t\) PSNR, and usually in LPIPS as well; some gains are large (e.g., coinrun PSNR 22.10 vs 11.30, \(\Delta_t\) PSNR 9.03 vs 0.48).
- **The Android experiment provides at least some evidence that the method is not confined to toy 2D settings.** On the 1X robotics dataset, WLA is slightly worse in framewise PSNR but substantially better in FVD (131.02 vs 393.85) and better in \(\Delta_t\) PSNR (Table 3), which is consistent with the paper’s claim that the latent structure helps temporal/action coherence.
- **The ablation does support that the proposed structure matters, albeit only partially.** Table 1 shows degraded MSE when removing either rotation or least-action slot alignment, suggesting the full design is not just incidental engineering.

## Weaknesses

###: Fatal

### Major:
- **The abstract and framing overclaim adaptation/generalization relative to the actual experiments.** The paper claims WLA can “with minimal or no action labels, quickly adapt to new environments with novel action sets,” but the presented experiments do not establish this. There is no label-efficiency curve, no adaptation-speed analysis, and no explicit experiment on genuinely novel action sets. The ProcGen section mentions seen/unseen and in-play/out-of-play settings, but quantitatively the paper mainly reports seen-environment generative metrics (Table 2) and only ActionACC for the out-play/unseen setting (Table 1 right). This weakens one of the paper’s main stated reasons for significance.
- **The central empirical comparison is too narrow to isolate the value of the proposed Lie structure.** The paper compares primarily to Genie, and does not include a non-Lie multi-environment baseline with comparable object-centric capacity. As written, the experiments do not disentangle whether gains come from the Lie-group inductive bias, from the slot-based/object-centric design, or from other architectural/training differences. Since the paper’s key scientific claim is not merely “a stronger model works,” but that Lie-structured continuous/compositional dynamics help, this missing isolation matters.
- **The Genie comparison is not sufficiently convincing as a definitive superiority claim.** The paper adapts Genie to the structured-CIP-with-labels setting by “incorporat[ing] trainable embeddings of action labels and append[ing] them to the output of the action embeddings in their latent action model” and uses an open-source implementation with largely default settings plus more iterations (Sec. 6.2). That is enough for a reasonable baseline, but not enough to make a very strong claim that WLA is better than the strongest relevant alternative. This is especially important because Genie is the main baseline carrying the empirical case.
- **The broad compositionality claim is materially limited by the paper’s own commutativity assumption.** The conclusion explicitly states: “our method does not account for the possible randomness of the environment” and “we assume a priori that transitions in the environment commute with each other.” For many game and robotics settings, order-sensitive action composition is important. This does not invalidate the reported results, but it does mean the paper should frame its contribution as compositionality within a restricted commuting family, not as a broadly general solution to compositional control.
- **The evaluation of the paper’s core “continuity” and “compositionality” claims is too weak on PHYRE.** Section 6.1 is entirely qualitative: Figure 3 shows interpolation, and Figure 4 shows composed effects, but there are no quantitative metrics, no baselines, and no systematic tests of whether the learned composition actually matches observed composed dynamics. Since continuity and compositionality are central claims—not side observations—this leaves a significant evidential gap.
- **The controllability evidence relies mainly on proxy metrics rather than direct control evaluation.** \(\Delta_t\) PSNR is a sensitivity-style metric, and ActionACC is measured through a post hoc logistic regressor from inferred \((\lambda,\theta)\) to action labels (Sec. 6.2). These are informative, but indirect. For a paper centered on building a better controller interface, stronger action-grounded evaluation would be more convincing.

### Minor
- **The ablation evidence is thinner than the text suggests.** Table 1 only reports MSE for the ablations on one table, with no broader breakdown across the main claimed outcomes such as controllability or unseen-environment generalization. Thus the claim that the components “significantly contribute” is directionally supported but not comprehensively established.
- **Sensitivity to key structural hyperparameters is not analyzed.** The method depends on user-specified numbers of slots \(N\) and Lie action components \(J\) (Sec. 4.4), and the paper explicitly says increasing them generally improves performance at higher cost, but provides no sensitivity study. Given that these choices are central to the proposed representation, some robustness analysis would have helped.
- **The connection between the formal group-equivariant story and the implemented training objective is not fully tight.** The paper motivates the method through Eq. 2–3 and equivariance/linearization, but the practical training in Eq. 7–9 is based on reconstruction and forward/backward prediction, without an explicit equivariance loss. This does not make the method unsound, but it does make the theory-to-implementation bridge less direct than the framing suggests.
- **The robotics claim should be stated more carefully.** The Android experiment is encouraging, but one dataset with video metrics alone is weaker evidence than the conclusion’s broader wording about “modeling real-world robot actions in 3D environments.”

### Trivial
- **The novelty claim in the conclusion is unsupported as written.** The statement that this is “the first of its kind as a generative interactive framework that is based on a state-space model” is not established in the paper body and should be softened or removed.

## Nice-to-Haves
- Add a label-efficiency study for \(\text{Ctrl}_{adapt}\) to support the “minimal labels” claim.
- Report full unseen/out-of-domain ProcGen generative metrics, not just ActionACC.
- Include at least one stronger non-Lie multi-environment baseline to isolate the value of Lie structure.
- Quantify PHYRE interpolation/composition rather than relying only on figures.
- Analyze sensitivity to \(N\), \(J\), and perhaps the sparsity penalty.
- Add failure cases and longer rollouts to clarify the method’s operating regime.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work X/Y/Z.”** Removed per instructions; I cannot verify uncited external works beyond what the paper already includes.
- **Pure reproducibility nitpicks about omitted implementation details or hyperparameters.** The paper already specifies the main architecture and losses, and minor missing details are not core flaws here.
- **Complaints that the paper should compare to many more baselines simply because they exist.** A broader comparison would help, but the core issue is narrower: the current setup does not isolate the proposed Lie structure well enough. The weakness is kept in that calibrated form rather than as a generic “more baselines” demand.
- **Harsh claims that the paper’s mathematics is invalid because the equivariant autoencoder existence theorem may not strictly apply in practice.** The paper uses this as motivation and architecture design, not as a formal proof of exact realization in experiments. This is not a fatal soundness issue on the basis of the submission alone.
- **Formatting/style issues from the PDF extraction.** Ignored as instructed.

## Novel Insights
The most important synthesis across the reviews is that this paper is **better as a structured modeling proposal than as a fully validated generalization paper**. Its strongest contribution is not “we solved multi-environment controllable world modeling,” but rather: *a Lie-structured latent dynamics prior, paired with object-centric slots, appears to materially improve action-conditioned temporal coherence in a shared multi-environment model.* The submission becomes weaker exactly where it claims more than that—especially on rapid adaptation, minimal labels, and broad compositionality beyond commuting transitions. In other words, the method seems promising and the reported gains are real enough to take seriously, but the paper’s narrative currently outruns the evidence.

## Suggestions
- Narrow the headline claims to what is directly shown: strong shared-model action-conditioned generation, not yet fast few-shot adaptation to novel action vocabularies.
- Add an experiment where the simulator is trained on a subset of environments and adapted to a held-out one with varying numbers of labels; this would directly test the paper’s most ambitious claim.
- Add a non-Lie but otherwise comparable multi-environment baseline to isolate the value of the proposed inductive bias.
- Quantify continuity/compositionality on PHYRE with explicit interpolation/composition metrics.
- Reframe compositionality more precisely in light of the commutativity assumption.
- Soften the conclusion’s novelty language and the robotics-generalization wording.

## Score and Decision
**Novelty:** good. The Lie-action/object-centric combination is genuinely distinctive.  
**Technical soundness:** moderate. The method is coherent, but some theoretical framing is stronger than what the implementation/evaluation verifies.  
**Empirical support:** moderate. The seen-setting ProcGen results are strong, but the evidence for the broadest claims is incomplete, and the comparison protocol does not fully isolate the proposed contribution.  
**Significance:** moderate. There is clear promise, but the current paper does not yet establish the general adaptation story it advertises.  
**Clarity:** reasonably good overall, especially in the conceptual setup, though some claims should be stated more carefully.

Relative to calibration examples in adjacent world-model papers, this feels stronger than a weak reject with little evidence, but not strong enough for acceptance at ICLR in its current form because the main claims are only partially demonstrated and the experimental case is too narrow for the breadth of the framing.

**Score: 5.8**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.8</pineapple>
MY FINAL DECISION: <orange>Reject</orange>