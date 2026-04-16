## Summary
This paper argues that high cosine similarity between distinct CLIP image embeddings (“erroneous agreements”) is not by itself sufficient to explain downstream VLM failures. The core empirical evidence is striking: LLaVA-1.5-7B, using the same frozen CLIP image encoder, strongly outperforms CLIP-like scoring on What’sUp and also improves over CLIP on MMVP/MMVP-VLM, suggesting that information may still be present in the embeddings but not effectively extracted or utilized by CLIP-style matching. The paper further explores this gap through ablations on evaluation, training data, and text encoding, and shows that decoding changes (M3ID) and pairwise comparative evaluation can further improve LLaVA’s performance.

## Strengths
- **The paper makes a genuinely important empirical point that directly challenges an overly strong interpretation of “erroneous agreements.”** Table 1 is especially compelling: on What’sUp Subset A Left/Right, CLIP-ViT-L/14-336px gets 49.0 individual / 1.9 pair accuracy while LLaVA-1.5-7B reaches 99.0 / 98.1, despite average embedding cosine similarity of **0.995**. This is hard evidence that high cosine similarity does not automatically imply downstream blindness for all models using that encoder.
- **The central phenomenon is shown across more than one benchmark.** Beyond What’sUp, the paper reports strong LLaVA gains over CLIP-like models on COCO-spatial and GQA-spatial (Table 2), and a smaller but still consistent gap on MMVP/MMVP-VLM (Table 3). This makes the observation look real rather than a single-benchmark artifact.
- **The paper does useful work disentangling a narrower, credible practical claim: utilization/decoding matters even with a fixed encoder.** The M3ID result on MMVP (Table 6: pair accuracy from 25.3 to 31.3) is not huge, but it is meaningful because it improves performance without changing the image encoder, directly supporting the claim that post-encoding utilization is part of the bottleneck.
- **The limitations section is candid and materially useful for interpreting the claims.** The authors explicitly acknowledge that they do not probe the fine-grained mechanism and that they do not retrain CLIP/SigLIP from scratch or with larger batch sizes, which helps bound what can and cannot be concluded.

## Weaknesses

###: Fatal
- None.

### Major:
- **The paper’s strongest causal conclusion—namely that the performance gap is “likely caused by the inadequate vision-language alignment of CLIP’s paradigm”—is not established as cleanly as claimed.**  
  The Section 4 argument is largely residual: unify one evaluation on MMVP/MMVP-VLM, then test finetuning CLIP-like models on converted LLaVA data, and then replace the text encoder with an LLM-derived encoder. But these are only partial controls. The paper itself says in Section 4.2 that it converts LLaVA data to image-caption format and, “**By default, we lock the image encoder during finetuning for strict ablation**,” and in Section 6 it concedes, “**we do not train CLIP or SigLIP models from scratch or use larger batch sizes... so the conclusion on the effects of different factors is restricted.**” Those are reasonable limitations, but they also mean the ablations do not fully isolate “paradigm” from objective/training/interface differences. The evidence supports a weaker conclusion: CLIP-style matching is substantially worse than LLaVA-style processing here, and the gap is not explained away by the specific data/text swaps tried.

- **The paper’s comparability story is strongest on MMVP/MMVP-VLM, but less fully unified for What’sUp, where the most dramatic result appears.**  
  Section 4.1’s “unified evaluation” evidence is shown in Table 3 for MMVP/MMVP-VLM via multiple-choice scoring for LLaVA. However, the headline What’sUp result in Table 1 still compares CLIP’s cosine-matching setup against LLaVA’s prompted generation behavior. This does not invalidate the empirical gap, but it weakens the strongest interpretation that the paper has directly shown “more information extracted from the same embeddings” under tightly matched task interfaces on the benchmark where the effect is largest.

- **Section 5.2’s “relaxed constraints” evaluation changes the task substantially and is overinterpreted.**  
  The paper is fairly explicit that this new setup is different: “**we ‘force’ the model to output differently for two images in a pair**,” and the criterion jointly compares both images/captions. This can be a useful diagnostic of pairwise discrimination signal, but it is not the original one-image task and should not be treated as a direct “upper bound” on usable performance for that task. The result in Table 7 is therefore informative as evidence of latent comparative signal, but the claim that it shows how much task-relevant information is preserved or merely underutilized is too strong without stronger validation.

### Minor
- **The paper remains somewhat too LLaVA-centric for the breadth of its conclusions.**  
  The main analysis and mechanistic discussion are centered on LLaVA-1.5-7B. The paper does state that the gap “generalizes to some other MLLMs” in Appendix B.5, so a blanket criticism that only one model is studied would be inaccurate. Still, in the main paper, the evidence and narrative are dominated by a single architecture family, which limits how broadly one should generalize the conclusions about VLMs as a whole.

- **The notion of “paradigm” remains too coarse to yield a sharp mechanistic takeaway.**  
  The paper usefully argues that something beyond encoder quality matters, but “paradigm” currently bundles together many changes at once: contrastive vs. generative objective, dual-encoder cosine scoring vs. connector+LM, tokenization/interface differences, and decoding behavior. As a result, the practical lesson is directionally useful but not yet mechanistically precise.

- **The toy example in Section 3.2 is only intuition, not evidence about real CLIP geometry.**  
  The 3D example with cosine similarity >0.989 and Spearman correlation -1 is fine as a conceptual reminder that cosine similarity is incomplete, but it does not itself show that actual CLIP embeddings preserve exploitable nonlinear structure of this kind. The paper mostly relies on experiments for that anyway, so this is a presentation/interpretation limitation rather than a core flaw.

- **The MMVP improvements from M3ID, while meaningful, are modest and should not be overstated.**  
  Table 6 shows a clear gain (+6 pair accuracy), which supports the narrower decoding/utilization claim. But it remains a partial recovery on a difficult benchmark, not a decisive solution.

### Trivial
- None.

## Nice-to-Haves
- A direct probe on frozen CLIP embeddings for the relevant distinctions (e.g., left/right on What’sUp) would have complemented the paper well and more directly tested whether task-relevant information is linearly or nonlinearly recoverable from the embeddings.
- A more mechanistic analysis of where LLaVA’s advantage arises—e.g., connector outputs, early LM layers, or changes in pairwise separability before/after the MLP—would make the “paradigm” claim much more actionable.
- Extending the main-text evidence beyond spatial reasoning would help establish how general this phenomenon is across failure modes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper only studies LLaVA-1.5 / only one model.”**  
  Removed in its strong form because the paper explicitly states: “**We also find that this performance gap relative to CLIP generalizes to some other MLLMs with different scales and language models in Appendix B.5.**” The fair retained criticism is narrower: the *main-paper evidence* is still mostly LLaVA-centric.

- **“Comparisons to Libra / I-MoF are unfair and should be criticized as asymmetrical comparisons.”**  
  Removed per instruction. The comparison in Section 5.1 is used to argue that decoding changes can be competitive even without modifying the encoder; any asymmetry there does not disadvantage the baselines in a way that undermines the authors’ stronger point.

- **Pure reproducibility complaints about omitted implementation details or omitted appendix specifics.**  
  Removed as standalone weaknesses. While some conversions/evaluation details would be helpful, generic reproducibility nitpicks are not a substantive reason to downgrade the paper here.

## Novel Insights
The most important synthesis is that this paper should be judged as making **two different claims of different strength**. The strong causal claim—“the CLIP paradigm is the main cause of the gap”—is only partially supported. But the weaker claim is both well supported and important: **high cosine similarity in CLIP embedding space is not sufficient evidence of irreversible information loss for downstream multimodal use**. The What’sUp result, together with the M3ID gain and the pairwise relaxed evaluation, jointly suggests a more nuanced view: encoder geometry, extraction strategy, and decoding/utilization all contribute, and these benchmarks are probing the whole pipeline rather than the encoder alone.

## Suggestions
- **Narrow and sharpen the main claim.** Reframe the headline conclusion from “the gap is likely caused by inadequate vision-language alignment of CLIP’s paradigm” to a more defensible version: CLIP-style matching underutilizes information that LLaVA-style processing can sometimes exploit, and the current ablations suggest this gap is not explained solely by the tested data/text factors.
- **Reframe Section 5.2 more carefully.** Present the relaxed evaluation as a diagnostic of pairwise comparative signal rather than an upper bound on the original task.
- **Add a direct embedding probe.** A linear/nonlinear probe on frozen CLIP embeddings for What’sUp/MMVP-style distinctions would directly test the paper’s central information-preservation thesis.
- **Probe the mechanism behind “paradigm.”** Analyze the connector outputs or LM layers, or compare separability before and after the projector, to show where the additional extractability emerges.
- **Clarify the scope of generalization.** If Appendix B.5 contains convincing results on other MLLMs, elevate at least one of them into the main paper.

## Score and Decision
**Evaluation by axis:**  
- **Originality:** Good. The paper offers a meaningful corrective to an increasingly common narrative about erroneous agreements.  
- **Importance:** Good. Understanding whether failures stem from encoder blindness vs. downstream utilization is highly relevant to VLM research.  
- **Claims support:** Mixed. The narrow empirical claim is well supported; the broader causal/mechanistic claim is overstated relative to the ablations.  
- **Experimental soundness:** Reasonable but incomplete. The experiments are informative and honest about limits, but not sufficient to conclusively isolate “paradigm.”  
- **Clarity:** Generally clear and easy to follow.  
- **Value to the community:** Moderate to high, especially as a corrective empirical study, though less so as a mechanistic explanation.

**Calibration against human-reviewed anchors:**  
- Compared with **bb2Cm6Xn6d** (“Intriguing Properties...”, scores 6/6/5/5, reject), this paper is *more focused* and has a sharper core empirical result, but it shares the weakness of drawing broader conclusions than the evidence fully warrants. I place it slightly above that work’s center because the key What’sUp result is unusually strong and directly relevant.  
- Compared with **UndmcWatBN** (“Dissecting Zero-Shot Visual Reasoning...”, scores 3/5/3/3), this paper is clearly stronger: its central observation is more surprising, better evidenced, and more consequential.  
- Compared with **chanJGoa7f** (“Towards Interpreting Visual Information Processing...”, scores 6/8/6/8, accept), this paper is somewhat weaker mechanistically: it has an important empirical finding but less direct insight into *how* the model extracts the information.  
- Compared with **gam5LiMPKT** (“Fading Focus...”, scores 3/6/5/3/6), this paper is stronger overall because its contribution is not just a modest decoding tweak; the decoding result is supplementary to a stronger main empirical claim.

Overall, this lands for me as a **borderline but slightly positive paper**: the central observation is strong and worth the community seeing, but the paper should substantially tone down its causal interpretation.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>