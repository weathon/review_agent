=== CALIBRATION EXAMPLE 88 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Yes. The title clearly conveys the central idea: combining a video generator with a multi-view 3D reconstruction network via stitching. It is specific enough to signal the method’s novelty.

- **Does the abstract clearly state the problem, method, and key results?**  
  Largely yes. It identifies the motivation (pairing a text-to-video generator with a 3D decoder), the two technical components (model stitching and direct reward finetuning), and the main empirical claim (improved text-to-3DGS and pointmap generation). This is aligned with the paper.

- **Are any claims in the abstract unsupported by the paper?**  
  The abstract’s strongest claim is that “all tested pairings markedly improve over prior text-to-3D models that output Gaussian splats.” The results table does show improvements on the listed benchmarks, but the paper does not fully justify how broad “all tested pairings” is in terms of model selection and benchmark coverage. Also, “high-quality text-to-pointmap generation” is supported mostly qualitatively rather than by an established benchmark, so that claim is weaker than the abstract suggests.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes. The introduction makes a convincing case that current text-to-3D methods suffer from either slow per-scene optimization or weakly coupled multi-stage pipelines, and that end-to-end latent approaches still rebuild 3D decoding from scratch. The gap—reusing strong pretrained 3D reconstruction models rather than training bespoke decoders—is clearly stated.

- **Are the contributions clearly stated and accurate?**  
  Mostly. The two contributions are clear: stitching a pretrained 3D model into the latent space of a video VAE, and aligning the generator with the stitched decoder via direct reward finetuning. The paper also claims broader applicability across video generators and 3D backbones, which is consistent with the experiments.

- **Does the introduction over-claim or under-sell?**  
  It somewhat over-claims generality. The paper frames VIST3A as a “general framework,” but in practice the method is validated on a small set of hand-picked, strong pretrained backbones and requires the input views to be arranged sequentially like video. That is an important constraint relative to the broad framing.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The overall idea is clear, but reproducibility is only partial from the main method section. The stitching procedure is understandable: choose a layer \(k^\star\) by minimizing linear MSE between encoder latents and intermediate 3D features, insert a linear stitching layer, then finetune with pseudo-targets. However, some details remain under-specified in the main text, especially how the encoder/video latent dimensionality is matched in practice and how the reward finetuning interacts with the decoder/rendering pipeline for each target representation. The appendix helps, but key implementation choices are spread across several sections.

- **Are key assumptions stated and justified?**  
  The key assumption is that a layer in the 3D backbone has a representation linearly aligned with the video VAE latent space. The paper motivates this empirically, but the justification is not deeply theoretical. The method relies heavily on this empirical compatibility, and the paper does not fully explain why this should hold across very different backbones and data regimes.

- **Are there logical gaps in the derivation or reasoning?**  
  A major conceptual gap is the claim that the stitched decoder can be treated as a “3D VAE” with minimal degradation after inserting a linear map. The evidence suggests this often works, but the paper does not rigorously explain why discarding the early part of the 3D network preserves its reconstruction ability beyond the empirical MSE criterion.  
  Another gap is in the reward formulation: the paper defines rewards using CLIP/HPS on both decoded frames and rendered 3D outputs, plus a consistency term, but it is not fully explained why these terms are sufficient to produce better latent distributions rather than simply encouraging better-looking samples under the specific benchmarks.

- **Are there edge cases or failure modes not discussed?**  
  Yes. The most important one is that the encoder expects video-like ordered sequences. The limitations section mentions this, but it is more serious than a minor caveat: it means the approach is not naturally suited to arbitrary unordered multi-view inputs, which is a common setting in 3D reconstruction and some 3D generation pipelines.  
  Another issue is domain mismatch between video backbones and 3D scene synthesis for prompts that require unusual camera motion or extremely long horizons; the paper mentions prompt-based camera control, but not when this becomes unstable.

- **For theoretical claims: are proofs correct and complete?**  
  The paper does not present formal proofs of its own main claims. It cites an upper bound from Insulla et al. (Eq. 4) to motivate the MSE criterion, which is reasonable, but the paper does not prove that this bound is tight or sufficient for selecting the best stitching layer in its setting. The theoretical support is therefore suggestive rather than conclusive.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Broadly yes. The experiments directly test the two main claims: that stitching pretrained 3D models into the video latent space works, and that reward finetuning improves the generator’s compatibility with the stitched decoder. The ablations in Table 6 and Fig. 5/8 are especially relevant.  
  That said, the strongest claim—general applicability across video generators and 3D backbones—is only partially tested, because the cross-product of models is limited and not exhaustive.

- **Are baselines appropriate and fairly compared?**  
  The 3D generation baselines are relevant and competitive for the cited task family. However, the comparison is somewhat uneven in two ways. First, different methods may rely on different auxiliary information or optimization budgets, and the paper’s discussion does not always make those differences fully explicit. Second, for pointmap generation, the paper states there are no established baselines/benchmarks, which makes those claims less directly comparable than the 3DGS results.

- **Are there missing ablations that would materially change conclusions?**  
  Yes, several. The most important missing ablation is a controlled comparison between stitching and simply finetuning the same 3D decoder end-to-end from the VAE latent space without stitching, at comparable parameter/update budgets. That would isolate the value of preserving pretrained 3D knowledge versus merely re-adapting the decoder.  
  Another important missing ablation is the contribution of each component of the reward more systematically across multiple backbones and datasets. Table 6 is helpful, but it is only for SceneBench and one model pair.  
  The paper also does not clearly isolate whether gains come from the direct reward finetuning itself or from the stronger decoder selection enabled by stitching.

- **Are error bars / statistical significance reported?**  
  No, not in the main tables. Table 4 reports human-study average ranks, but no confidence intervals or variance estimates. Given that some improvements are modest on certain metrics and benchmarks, the lack of statistical significance makes it harder to judge robustness.

- **Do the results support the claims made, or are they cherry-picked?**  
  The main quantitative tables support the paper’s headline claim of strong performance. I do not see obvious cherry-picking in the reported benchmark set, but the evaluation is selective in the sense that it emphasizes settings where the method is especially well matched: pretrained strong video models, strong pretrained 3D backbones, and benchmarks that reward prompt adherence and visual quality.  
  A particularly important caution is that Table 1/2 metrics are all image-based or judge-model-based, not direct 3D geometry metrics for the text-to-3D task. That is acceptable to some extent, but it means the claimed 3D quality gains are mediated through rendering and reward models rather than directly measured geometric accuracy.

- **Are datasets and evaluation metrics appropriate?**  
  Mostly. T3Bench, SceneBench, and DPG-Bench are sensible choices for text-to-3D generation. The use of VBench-style image quality, CLIP, and UnifiedReward is reasonable for prompting and image fidelity, though these are indirect proxies for 3D quality. For the reconstruction/stitching part, RealEstate10K, 7-Scenes, ETH3D, and ScanNet are appropriate.  
  The main concern is that the text-to-3D evaluation heavily depends on a mixture of automated scoring models whose calibration for rendered 3D outputs is not guaranteed. The paper itself acknowledges some limitations of traditional metrics, but the chosen judge models also have their own biases.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  The method is mostly understandable, but the interaction between stitching and reward finetuning could be explained more cleanly. In particular, it is not immediately obvious how the reward is computed through both the video decoder and the stitched 3D decoder without conflating reconstruction quality with generation quality.  
  The experimental section is clear overall, though some claims about “marked improvement” are stronger than the tables alone justify in every metric.

- **Are figures and tables clear and informative?**  
  The figures/tables are informative in content, especially Table 1, Table 3, Table 5, and Table 6, which align well with the method’s claims. Fig. 5 and Fig. 8 are useful for the stitching/robustness arguments.  
  The main clarity issue is that several conclusions rely on figure trends without sufficient quantitative summaries. For example, Fig. 6/10 are used to argue about layer selection, but the text could better summarize how stable the chosen layer is across different model pairs.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  They acknowledge one important limitation: the encoder assumes ordered, video-like inputs, so arbitrary unordered multi-view inputs are not naturally handled. This is a real and relevant limitation.

- **Are there fundamental limitations they missed?**  
  Yes. The method depends on access to strong pretrained video generative models and strong pretrained 3D reconstruction models, which may not always exist for new domains or modalities. It also depends on reward models (CLIP, HPSv2, UnifiedReward) whose preferences may not align with actual 3D fidelity.  
  Another important limitation is that stitching is performed at a specific layer chosen by linear alignment; if the latent structure is not compatible, the method may fail silently or require extensive search/tuning. The paper notes that absolute MSE does not transfer across architectures, which suggests the selection process may still be somewhat brittle.

- **Are there failure modes or negative societal impacts not discussed?**  
  The paper does not discuss broader impacts in depth. For a generative 3D system, likely concerns include misuse for synthetic scene creation, copyright/style imitation, and the potential to generate misleading or immersive content. These are not central to the technical contribution, but ICLR typically expects at least a brief acknowledgment when the system can generate realistic scenes from prompts.

### Overall Assessment
VIST3A is a strong and timely paper with an appealing idea: instead of learning a new 3D decoder from scratch, reuse a pretrained 3D reconstruction backbone via model stitching, then align a video generator to it with reward-based finetuning. The empirical results are promising and the ablations generally support the design. That said, the paper’s strongest claims of generality are somewhat ahead of the evidence, and the evaluation relies heavily on indirect metrics and a limited set of backbone pairings. The most important open questions are how robust stitching is outside the chosen model family, how much the gains depend on the reward models rather than true 3D improvements, and whether the approach can handle genuinely unordered or more diverse multi-view inputs. On balance, the contribution is substantive and likely of interest at ICLR, but the paper would be stronger with broader and more controlled evidence for generality and robustness.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents VIST3A, a framework for text-to-3D generation that combines a pretrained text-to-video model with a pretrained feedforward 3D reconstruction model via model stitching. The core idea is to reuse the latent space of a video VAE as the interface, stitch in a 3D decoder at the most linearly compatible layer, and then further align the generator with the stitched decoder using direct reward finetuning. The method is evaluated across multiple video backbones and 3D backbones, and the paper claims strong gains over prior text-to-3DGS methods, plus the additional ability to generate pointmaps.

### Strengths
1. **Clear and timely high-level idea with good systems integration.**  
   The paper identifies a meaningful bottleneck in recent latent diffusion-based text-to-3D systems: the decoder. Reusing strong pretrained 3D reconstruction models rather than training new decoders from scratch is a practical and appealing direction, especially given the rapid pace of progress in 3D foundation models.

2. **The model-stitching angle is genuinely interesting and potentially useful beyond this paper.**  
   The paper does more than simply “plug models together”: it searches for the most linearly compatible stitching layer, uses a closed-form linear map for initialization, and then finetunes lightly. This is a concrete mechanism that could generalize to other multimodal or latent-space composition problems.

3. **Good breadth of empirical evaluation.**  
   The method is tested with multiple video generators (Wan, CogVideoX, SVD, HunyuanVideo) and multiple 3D backbones (MVDUSt3R, VGGT, AnySplat), which supports the claim that the framework is not tied to one specific pairing. The paper also evaluates on several benchmarks, including T3Bench, SceneBench, and DPG-Bench, as well as pointmap and pose estimation tasks.

4. **Ablations target the two main design choices.**  
   The paper includes studies on stitching-layer selection, direct reward finetuning, and the comparison between integrated vs. sequential generation. These ablations are aligned with the paper’s main claims and help clarify why the approach works.

5. **Potentially strong practical significance.**  
   If the reported results hold up, the approach offers a relatively modular way to build text-to-3D systems on top of fast-moving foundation models. That is appealing for the ICLR audience because it combines representation learning, generative modeling, and transfer/composition of pretrained networks.

### Weaknesses
1. **Novelty is somewhat mixed because the paper combines several existing ideas rather than introducing a clearly new learning principle.**  
   The method is built from model stitching, latent diffusion-style 3D generation, and direct reward finetuning—each of which has prior art. The contribution lies in the combination and adaptation to text-to-3D, but the paper does not fully convince that this is more than a well-engineered synthesis of known ingredients.

2. **The empirical story is strong but not fully convincing on fairness and comparability.**  
   The paper compares against several baselines, but the setups differ significantly across methods in terms of backbone strength, training data, and inference procedures. For ICLR standards, it would be important to show that the gains come from the method rather than from particularly strong chosen backbones, reward models, or evaluation protocol choices.

3. **Some central claims rely on heuristic choices that are under-justified.**  
   For example, the stitching layer is chosen by minimum linear MSE on a held-out set, but the paper only partially explains why that criterion should be expected to generalize across backbones and domains. Similarly, the reward design mixes CLIP, HPSv2, and perceptual consistency losses with several scaling factors; the rationale for these coefficients feels empirical rather than principled.

4. **Reproducibility would be challenging without substantial implementation detail beyond the main paper.**  
   The paper provides many training details in the appendix, which is good, but the overall system is still complex: multiple pretrained models, custom stitching, LoRA finetuning, reward optimization over denoising trajectories, and benchmark-specific rendering/evaluation choices. A reader would likely need code to reproduce the results reliably, especially because many hyperparameters and data-processing choices are model-specific.

5. **The direct reward finetuning component raises stability and optimization questions.**  
   The paper acknowledges gradient explosion issues and uses modified detached-gradient handling plus partial timestep sampling. That is sensible, but it also suggests the optimization is delicate. The paper does not fully characterize failure modes, sensitivity to reward weights, or whether training is robust across random seeds.

6. **Evaluation is mostly centered on rendered-image metrics rather than intrinsic 3D quality.**  
   The benchmarks and metrics are reasonable for the task, but many reported numbers are indirect proxies of 3D quality. There is limited analysis of geometry correctness beyond pointmap and pose benchmarks, and little evidence about long-horizon consistency, multi-object reasoning in 3D, or editability of the generated scenes.

### Novelty & Significance
**Novelty: moderate.** The paper’s strongest novelty is in framing text-to-3D as “stitch a video generator to a pretrained 3D reconstruction network” and showing that this can work across several backbones. The individual components are not new in isolation, but their combination is plausibly useful and reasonably well-motivated.

**Significance: moderate to high if the results are robust.** The approach targets a real bottleneck in text-to-3D systems and could reduce the need for expensive custom 3D decoders. That said, ICLR’s acceptance bar typically expects both conceptual novelty and convincing evidence of generality; here the practical value is clear, but the conceptual leap is more incremental than breakthrough-level.

**Clarity: good overall, though somewhat overloaded.** The paper is readable and logically organized, but it packs a lot of moving parts into one framework, which makes the method harder to parse quickly. Some sections, especially the reward optimization and stitching details, would benefit from tighter exposition and cleaner notation.

**Reproducibility: fair but not ideal.** Enough implementation detail is given to understand the method, but the system’s complexity, reliance on several pretrained models, and many engineering choices make exact reproduction nontrivial.

### Suggestions for Improvement
1. **Add stronger evidence that stitching is the right mechanism, not just a workable one.**  
   Include comparisons to simpler alternatives such as training a small decoder from the same latent space, using CKA-only selection, or learning a nonlinear adapter. This would sharpen the claim that model stitching is uniquely beneficial.

2. **Provide a more rigorous fairness discussion for the baseline comparisons.**  
   Clarify whether all baselines use similarly strong pretrained backbones, comparable training data, and similar inference budgets. If not, add matched-backbone comparisons to isolate the effect of VIST3A.

3. **Report sensitivity analyses for the reward weights and stitching choices.**  
   Since the reward combines several terms with manually chosen coefficients, show how performance varies under reasonable perturbations. The same applies to the choice of \(K\), \(T_1/T_2\), and the layer selected for stitching.

4. **Strengthen the geometry-centric evaluation.**  
   Add more direct 3D evaluations where possible, such as surface consistency, multi-view reprojection error, or user studies focused specifically on geometry. This would better support the claim that the method improves actual 3D scene quality, not just rendered aesthetics.

5. **Clarify failure cases and limitations more concretely.**  
   The paper briefly notes that the encoder expects ordered video-like input, but it would be helpful to show explicit failure examples, especially for unordered multi-view inputs, unusual prompt types, or scenes outside the training distribution.

6. **Streamline the presentation of the alignment method.**  
   The current description of direct reward finetuning is informative but dense. A cleaner algorithmic presentation, with a concise pseudo-code box and a clearer explanation of which gradients flow where, would improve accessibility and reproducibility.

7. **Include compute and training-cost comparisons.**  
   Since one appeal of the method is reusing pretrained 3D models, it would be useful to report training cost, memory use, and inference speed relative to prior text-to-3D systems. This is especially relevant for assessing practical significance at ICLR.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against the strongest recent direct text-to-3D and latent-3D baselines on the same benchmarks, including any methods that also use pretrained video or large reconstruction backbones. Right now the claim that VIST3A “markedly improves over prior text-to-3D models” is not fully convincing without the most competitive contemporaneous systems and a clearly matched evaluation protocol.

2. Add a baseline that swaps in a conventional learned decoder trained from scratch on the same latent space, while keeping the same video generator and training data. Without this, the core claim that stitching a pretrained 3D model is better than training an ad hoc decoder is not isolated from the effects of extra capacity, different inductive bias, or extra supervision.

3. Add an ablation that removes the reward tuning but keeps the stitched decoder, and another that uses reward tuning with a non-stitched decoder. The paper claims alignment is essential to make the latent generator decodable; without these controls, it is unclear whether gains come from stitching, from reward tuning, or from simply fine-tuning the video backbone longer.

4. Add scaling experiments on training data size and prompt/domain transfer, especially for out-of-distribution prompts and scenes. ICLR reviewers will expect evidence that the method is not just fitting DL3DV/ScanNet-style data and that the “small dataset/no labels” claim holds under reduced data and distribution shift.

5. Add compute/memory/runtime comparisons against prior text-to-3D systems and against training a standard decoder from scratch. The paper’s contribution is framed as a more practical way to reuse foundation models, so efficiency claims need quantitative support, not just qualitative statements.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze when and why stitching works by reporting layer-wise transferability across more than one backbone-decoder pair, with the actual downstream performance, not just MSE/CKA. The current evidence is too indirect to justify the claim that the best stitching layer can be selected reliably from linear alignment alone.

2. Quantify how much each reward component changes the latent distribution, decoder-domain match, and 3D consistency over the denoising trajectory. The paper claims trajectory-level alignment is important, but it does not show whether the rewards actually keep samples in-distribution or simply improve output aesthetics.

3. Measure failure modes by prompt type, scene complexity, and camera-motion prompts. The method relies on a video prior and a 3D decoder; without breakdowns, it is unclear whether it fails systematically on long prompts, occlusions, repeated structures, reflective surfaces, or extreme motion instructions.

4. Analyze whether the stitched decoder preserves the original 3D model’s reconstruction behavior beyond the selected benchmarks, including sensitivity to unordered views and view count. The limitation section admits sequence ordering matters, so the paper needs to show how brittle the stitched system is to this assumption.

5. Validate that the reported gains are not driven by the evaluation metrics themselves, especially reward-model-based scores and CLIP/HPS-derived metrics. The paper needs correlation or sanity-check analysis against human judgments and geometry-aware metrics to establish that the improvements reflect real 3D quality.

### Visualizations & Case Studies
1. Show denoising-trajectory visualizations: intermediate latents, decoded frames, and reconstructed 3D outputs across several steps before and after reward tuning. This would reveal whether the method actually improves decodability progressively or only patches the final sample.

2. Show side-by-side failure cases for stitching at the wrong layer versus the selected layer, with corresponding MSE/CKA and output quality. That would directly test the paper’s central claim that linear transferability predicts usable stitched decoders.

3. Show qualitative comparisons on hard prompts where video priors tend to drift: repeated objects, precise counting, unusual geometry, and long scene descriptions. These cases would expose whether VIST3A genuinely solves text-to-3D or mostly works on visually easy prompts.

4. Show examples where the sequential pipeline fails but the stitched model succeeds under latent perturbations, with rendered outputs and quantitative deltas. The robustness claim is important, but the current evidence is too abstract without concrete visual failure illustrations.

5. Show camera-control examples and corresponding 3D trajectories/renderings for multiple prompts, not just one or two curated examples. This would verify that prompt-based viewpoint control is a stable capability rather than a cherry-picked demo.

### Obvious Next Steps
1. Extend the framework to more 3D backbones and at least one non-Gaussian representation, and report whether stitching still works without bespoke engineering. The paper’s main claim is generality, so it should demonstrate portability beyond the three chosen models.

2. Add a true end-to-end evaluation of editability and controllability, not just generation quality. If the method is meant as a text-to-3D foundation, reviewers will expect evidence that the produced scenes can be reliably steered, edited, and rendered under novel camera paths.

3. Include a stronger study of the “small data, no labels” claim by varying the amount of unlabeled stitching data and alignment data separately. This would show whether the method genuinely reduces data dependence or whether it just shifts the burden to a different stage.

4. Provide a direct comparison to a sequential image/video-to-3D pipeline under matched compute and matched backbone quality. The paper argues that integrated latent-space generation is better than sequential generation, but that should be established with a clean apples-to-apples benchmark.

# Final Consolidated Review
## Summary
This paper proposes VIST3A, a framework that turns a text-to-video latent generator into a text-to-3D generator by stitching it to a pretrained feedforward 3D reconstruction model, then aligning the generator with the stitched decoder using direct reward finetuning. The core appeal is pragmatic: reuse strong 3D foundation models instead of training a bespoke decoder from scratch, and use reward-based alignment to keep the generator’s latents decodable into 3D-consistent outputs.

## Strengths
- The central idea is genuinely useful: repurposing pretrained 3D reconstruction backbones as decoders for a video latent generator is a strong systems-level move, and the paper shows it can work across multiple pairings (Wan/CogVideoX/SVD/Hunyuan with AnySplat/VGGT/MVDUSt3R) rather than only one carefully chosen combination.
- The paper gives relevant ablations for its two main components: layer selection for stitching, direct reward finetuning, and integrated vs. sequential generation. These are well aligned with the paper’s claims and make the main mechanism more credible than a pure end-to-end black box.

## Weaknesses
- The generality claim is still ahead of the evidence. The paper validates a limited set of strong, hand-picked pretrained backbones, and the encoder requires video-like ordered inputs; this is a meaningful restriction for a method presented as a “general framework.”
- The evaluation is still too indirect for the strength of the claims. Most text-to-3D results are based on rendered-image metrics and judge models rather than intrinsic 3D geometry metrics, so the paper is partly optimizing for what the evaluators score well, not fully demonstrating improved 3D scene fidelity.
- The key stitching criterion is heuristic and not fully de-risked. Choosing the stitch point by linear MSE is reasonable and empirically supported, but the paper does not convincingly show that this criterion is robust across architectures or that a linear bridge is always sufficient; the cross-architecture MSE discussion even suggests the selection process is brittle and must be redone per pair.

## Nice-to-Haves
- A cleaner apples-to-apples comparison against a learned decoder trained from scratch on the same latent space would help isolate whether the gain truly comes from stitching a pretrained 3D model rather than from extra capacity or engineering.
- More sensitivity analysis on reward weights, gradient-enabled steps, and stitching-layer choice would make the training story more trustworthy, especially given the number of manually tuned coefficients.
- Direct failure cases would strengthen the paper: unordered multi-view inputs, very long prompts, unusual camera-motion instructions, and hard geometry cases.

## Novel Insights
The most interesting insight is that the latent space of a modern video generator can be made to interface surprisingly well with the internal representations of a pretrained 3D reconstruction model, if the stitching point is chosen carefully and the decoder is only lightly adapted. Equally important is the observation that reward tuning should act through the full denoising trajectory, not just on final outputs, because the main problem is not merely visual quality but keeping the generated latents inside the domain that the stitched 3D decoder can interpret. That combination makes the method more than a simple pipeline: it is an attempt to unify generation and reconstruction in shared latent space, and that is a meaningful direction.

## Potentially Missed Related Work
- **Representation Autoencoders / encoder-replacement work** — relevant because the paper explicitly notes related concurrent work on swapping VAE components, and the broader question of compatibility between pretrained latent spaces and downstream decoders is directly related.
- **Any recent work on latent-space 3D generation with pretrained video or image backbones** — relevant as neighboring methods for direct comparison and to better distinguish stitching from other latent-3D integration strategies.
- **Deep model reassembly / stitchable networks** — relevant as the methodological ancestor of the stitching idea, though the paper already cites this line and applies it in a nontrivial 3D setting.

## Suggestions
- Add one controlled baseline that keeps the same video generator and data but trains a conventional decoder from scratch; this is the cleanest way to show stitching a pretrained 3D model is actually the right choice.
- Report a more geometry-centered evaluation where possible, and include explicit failure cases for unordered inputs and hard prompts so the limitations are concrete rather than only acknowledged in prose.
- Provide sensitivity plots for the reward coefficients and stitch-layer selection to show the method is not overly dependent on brittle heuristics.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
