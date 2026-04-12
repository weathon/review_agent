## Summary
This paper studies how narrow finetuning changes internal activations in LLMs and argues that these changes leave unusually legible traces even on unrelated inputs. The proposed Activation Difference Lens (ADL)—combining average activation differences, Patchscope/Logit Lens, and activation steering—surfaces tokens and steered generations that reveal the finetuning domain, and an LLM-based auditing agent uses these signals to identify finetuning objectives far better than black-box querying alone. The paper further argues that this phenomenon is strongest for semantically narrow “model organism” finetunes, is much weaker for broader post-training, and can be mitigated by mixing unrelated data into finetuning.

## Strengths
- **The paper isolates a surprisingly strong empirical phenomenon across a diverse set of organism families and model families.** It evaluates 33 organisms spanning SDF, emergent misalignment, subliminal learning, and taboo-word models across Gemma, Llama, and Qwen from 1B to 32B parameters. This breadth matters because the main claim is not about one bespoke setup, but about a recurrent artifact of narrow finetuning.
- **ADL is simple but unusually revealing.** The core procedure uses only mean activation differences on the first few tokens of unrelated text plus standard interpretability tools (Patchscope, Logit Lens, steering), yet it surfaces concrete domain semantics such as “stocks,” “culinary,” or “cat”-related outputs in Figure 1. The simplicity is a strength here: the paper shows that one does not need elaborate mechanistic tooling to detect these traces.
- **The interpretability-agent evaluation is a specific and meaningful validation of informativeness rather than only qualitative cherry-picking.** The agent with ADL access substantially outperforms blackbox agents even when the latter get many more interactions. The paper reports that ADL with \(i=5\) identifies the broad objective for 91% of organisms, while the best blackbox baseline at \(i=50\) reaches 39%. That gap supports the claim that the surfaced traces contain operationally useful information.
- **The paper does useful scoping work rather than overclaiming universality.** Appendix E explicitly checks broader finetuning regimes and finds that standard chat finetuning leaves minimal or no detectable traces, while semi-narrow domain adaptation produces weaker effects. This is important because it turns the contribution into a more precise warning: narrow model-organism finetunes may be atypically easy to audit with diffing-style methods.
- **The mitigation experiments are practically useful even if mechanistically incomplete.** Mixing pretraining data into the narrow finetuning corpus substantially reduces token relevance and steering similarity, and the paper honestly shows the trade-off with reduced false-fact internalization. This gives concrete guidance for researchers building model organisms.
- **The paper does more than present a positive result; it triangulates the phenomenon from several angles.** Position ablations, layer ablations, full-vs-LoRA finetuning, reduced-sample experiments, mixed-data experiments, and grader-ablation analyses collectively make the empirical case more robust than a single headline figure would.

## Weaknesses
### Fatal
None.

### Major:
- **The mechanistic claim that the traces are specifically a form of “overfitting” is not established as strongly as the empirical detection claim.**  
  Section 5 shows that ablating the bias direction harms performance on finetuning data and sometimes helps on pretraining data, which supports that the direction is functionally important and potentially harmful off-distribution. But this does not by itself isolate *overfitting to semantic homogeneity* as the causal mechanism, as opposed to a more generic learned task direction or broader distributional specialization. The paper itself uses appropriately softer language in places (“We suspect that these biases are a form of overfitting”; “likely connect to ideas from catastrophic forgetting”), but some summary statements are stronger than the evidence fully warrants.
- **The causal analysis has an unresolved inconsistency on Gemma3 that weakens the universality of the proposed explanation.**  
  The paper explicitly states: “For Gemma3 1B, the causal effect on \(D_{pt}\) is slightly positive but comparable to baseline effects,” and attributes this to larger representational divergence between base and finetuned models. This is a reasonable possible explanation, but it remains a post hoc explanation rather than a fully validated one. Since the overfitting narrative leans on the sign difference between \(D_{ft}\) and \(D_{pt}\), the mixed result on Gemma should temper the strength of the mechanistic conclusions.
- **The evaluation pipeline depends heavily on LLM graders and rubric-conditioned judgments, which leaves some uncertainty about the exact magnitude of the reported gains.**  
  Token relevance, coherence thresholds, and final hypothesis grades are all LLM-scored. The paper does take this concern seriously and includes grader ablations in Appendix D, but the agreement is only moderate for token relevance (\(\alpha=0.65\)). This does not invalidate the result—the effect sizes are large enough that the conclusion likely survives grader noise—but it does make fine-grained quantitative claims, especially dramatic ratios like “30 times better” on grade \(\ge 4\), less definitive than they first appear.
- **The paper’s practical significance is strongest for narrow model-organism setups and substantially weaker for broader real-world post-training.**  
  This is not a flaw in honesty—the appendix clearly documents it—but it is a real limitation on significance for a general ICLR audience. The most consequential warning is about the realism of current model-organism studies, not about mainstream instruction tuning in deployed systems. That is still valuable, but narrower than the abstract and introduction initially suggest.

### Minor
- **The “discovery” aspect of the main phenomenon is somewhat less conceptually surprising than the presentation suggests.**  
  The method computes the average activation difference between base and finetuned models and then explicitly interprets or steers with that mean shift. It is therefore not shocking that this direction carries semantics related to the finetuning objective. What is genuinely interesting is the *strength and readability* of the effect on unrelated early tokens across many organisms, not the mere existence of some directional information. The paper would benefit from framing the contribution more as a quantitative empirical characterization than as an unexpected conceptual finding.
- **The boundary between “narrow enough to leave readable traces” and “broad enough that traces disappear” is not characterized systematically.**  
  The paper does show that chat finetuning mostly lacks the effect and that adding pretraining data attenuates it, but it does not provide a more quantitative account of semantic narrowness, dataset diversity, or training conditions that govern the transition.
- **The mitigation story is useful but incomplete.**  
  Mixing in unrelated data reduces detectability, but also weakens false-fact alignment, and at 1:1 the agents fail to recover average grade \(\ge 2\). The paper therefore demonstrates a trade-off more than a clean solution. It would help to quantify more directly whether useful target behavior is preserved for non-SDF settings beyond the FFA proxy.
- **The paper does not deeply analyze the representation of the bias direction itself.**  
  The claim that the bias comes from “constant semantic concepts shared across all finetuning samples” would be stronger with decomposition of \(\delta\) into subfeatures or subspaces rather than only using Patchscope, steering, and a 1D causal projection.
- **Some central scoping results are relegated to the appendix despite being crucial to interpretation.**  
  In particular, the fact that chat finetuning leaves “minimal or no detectable traces” is essential to understanding the paper’s scope and should be emphasized earlier in the main text, not mainly in Appendix E.

### Trivial
- **The steering setup is somewhat elaborate and computationally involved.**  
  Searching for steering strengths with coherence graders and aggregating over prompts is sensible, but it adds operational complexity that may make the method less lightweight in practice than the headline concept suggests.

## Nice-to-Haves
- Quantify “narrowness” more directly, e.g., via corpus diversity or semantic self-similarity, and show how ADL effectiveness varies along that axis.
- Add experiments on real-world narrow SFTs (e.g., legal, medical, code-domain finetunes) to bridge the gap between synthetic organisms and standard post-training.
- Decompose the activation-difference vectors with SAEs, PCA, or crosscoders to test whether the effect is driven by a small number of dominant features versus distributed drift.
- Provide one or two full case studies contrasting a successful ADL agent and a failed blackbox agent, showing exactly which surfaced tokens/steered outputs enabled the inference.
- Clarify more prominently in the main text that the strongest claims concern narrow finetuning/model organisms, not general chat-tuning.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The trace discovery is conceptually guaranteed by the mean-difference methodology.”**  
  Overstated. The method indeed computes a mean activation-difference direction, so it is unsurprising that it contains finetuning information. However, the paper’s substantive empirical claim is that this information is *highly readable on unrelated early-token activations across many narrow finetunes* and useful enough to power an auditing agent. That is not vacuous.
- **Concern about data leakage because the 10,000 pretraining samples may overlap with evaluation samples.**  
  The paper states it computes average activation differences on “a pretraining corpus containing 10,000 samples” and uses those differences for interpretation/steering; it is not making a train/test generalization claim over that corpus in the conventional sense. The criticism of “circularity” is too strong from the text provided.
- **Claim that the ADL-vs-blackbox comparison is unfair because ADL gets stronger signals.**  
  This is not a valid weakness here. The point of the paper is precisely to test whether ADL-derived access to internal differences is more informative than black-box interaction. The asymmetry is intentional and appropriate to the question being asked.
- **Requests to release activation tensors/checkpoints/raw intermediates for independent verification.**  
  Removed under the reproducibility rule; the paper already provides code and an appendix on reproducibility, and large artifact release is not required here.
- **Complaints about missing comparisons to unspecified external methods.**  
  Removed because external baselines cannot be verified here and the review instructions explicitly forbid mentioning missing related works.
- **Formatting/parser issues.**  
  Ignored as instructed.

## Novel Insights
The paper’s most important insight is less “activation differences contain finetuning information” than a sharper methodological warning: current narrow model-organism finetunes may be *pathologically legible* to simple diffing analyses in a way that broader post-training is not. This reframes positive results on such organisms: a technique succeeding there may partly be exploiting a narrow-data artifact rather than uncovering mechanisms that matter in realistic alignment or chat-tuning. The appendix evidence that chat finetuning largely lacks the same signal is therefore not peripheral—it is what makes the main result consequential.

## Suggestions
- Reframe the central claim more carefully: emphasize that the key empirical result is the *magnitude and readability* of the traces across narrow finetunes, not merely that the mean difference contains some task information.
- Soften or better qualify the mechanistic “overfitting” claim unless stronger causal isolation is added.
- Strengthen Section 5 with additional controls that distinguish “generic useful learned direction” from “artifact of semantic homogeneity,” and directly address the Gemma inconsistency.
- Move the broader-finetuning results from Appendix E into the main paper or foreground them much earlier, since they are essential for scoping significance.
- Add at least one real-world narrow finetune beyond synthetic organisms to test whether the phenomenon transfers outside model-organism settings.
- If space permits, include a more direct non-LLM or partially human sanity check for one evaluation component to complement the grader-heavy pipeline.

