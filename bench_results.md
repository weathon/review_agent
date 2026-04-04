# ICLR Benchmark Results

Date: 2026-04-03 04:44
Critic/Merger: z-ai/glm-5 (OpenRouter)
Neutral: z-ai/glm-5, Related Work: z-ai/glm-5:online (OpenRouter)

## Mq6bGrtktf

- GT: Reject (avg 3.2)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
This paper investigates an underexplored question in LLM citation research: not which documents to cite, but what content merits citations. The authors construct a dataset of 6,000 Wikipedia sentences organized into eight citation-motivation categories, conduct pairwise human annotation to establish preferences across 28 category combinations, and evaluate 11 LLMs against these preferences. They find that models systematically over-select sentences marked "Citation needed" on Wikipedia (up to 27.4% higher than humans) while under-selecting numeric and person-name content, and demonstrate that DPO can improve alignment by an average of 5.76%.

## Strengths
- **Novel research direction**: The paper correctly identifies a gap in the literature—prior work focuses on document retrieval for citation, not cite-worthiness assessment—and this has direct practical implications for deployed LLM systems where models implicitly control citation decisions.
- **Comprehensive evaluation scope**: Testing 11 models spanning open/closed-source and parameter scales enables meaningful conclusions about scale effects, such as Llama 1B performing near random baseline while Llama 70B achieves 61.6% agreement.
- **Concrete, actionable findings**: The identification of systematic misalignment patterns—over-selection of "Citation needed" sentences (+27.4% for Llama 70B) and under-selection of numeric (-22.6%) and person-name (-20.1%) sentences—provides specific targets for remediation in deployed systems.
- **Constructive training comparison**: The contrast between fine-tuning (average performance decrease) and DPO (+5.76% average improvement) offers practical guidance that preference-based training is more effective than supervised fine-tuning for this task.

## Weaknesses
- **Missing inter-annotator agreement statistics**: The paper reports 402 annotators but provides no agreement metrics (e.g., Cohen's kappa), making it impossible to assess annotation reliability—a critical gap for a dataset contribution.
- **Incomplete reproducibility information**: Appendix references appear as "??" throughout the paper, breaking links to essential details about model versions, prompts, and training hyperparameters. The annotation prompt and LLM evaluation prompt are not shown in the main text.
- **No statistical significance testing or confidence intervals**: Tables report agreement rates without error bars or significance tests. Given the dataset size (~2,596 pairs after filtering, ~1,288 test pairs), it is unclear whether reported differences (e.g., 5.76% average DPO improvement) are statistically meaningful.
- **No baseline comparisons for DPO**: The DPO improvement is reported relative to non-fine-tuned models, but simple baselines (random selection, majority class, or prompt-based approaches) would contextualize whether the ~60% agreement rates and improvements represent meaningful learning.
- **Unexplained negative results for larger models**: DPO improves Llama 1B by 11.8% and Llama 3B by 9.1%, but decreases Llama 70B by 1.6%. The paper notes this pattern but offers no analysis or hypothesis for why larger models might not benefit.
- **Pairwise evaluation paradigm gap**: The study evaluates pairwise preferences, but the actual deployment task is binary ("does this sentence need a citation?"). Without validating that improved pairwise agreement translates to better standalone citation decisions, the practical implications remain unclear.

## Nice-to-Haves
- Cross-domain validation on non-Wikipedia text (news, scientific papers) to support claims about generalization beyond Wikipedia's specific editorial culture.
- Example sentences showing where models diverge most from human preferences, which would clarify concrete failure modes.

## Novel Insights
The finding that LLMs systematically over-select sentences explicitly tagged "Citation needed" on Wikipedia—likely due to training data imprint—while under-selecting numeric and person-name content reveals a fundamental misalignment between how models learn cite-worthiness implicitly and what users actually need. The contrast between fine-tuning failure and DPO success demonstrates that citation preferences are better transmitted through preference optimization than supervised learning, suggesting that citation behavior should be explicitly trained rather than assumed to emerge from pretraining. Medical content emerging as consistently prioritized across humans and models indicates domain-specific alignment that could inform targeted deployment strategies.

## Potentially Missed Related Work
- Wright & Augenstein (2021), "CiteWorth: Cite-worthiness detection for improved scientific document understanding" — directly addresses cite-worthiness in scientific papers, relevant for cross-domain comparison with the Wikipedia-focused approach.
- Redi et al. (2019), "Citation needed: A taxonomy and algorithmic assessment of Wikipedia's verifiability" — cited but their taxonomy of citation reasons on Wikipedia could provide additional validation for the 8-category scheme.

## Suggestions
- Add inter-annotator agreement statistics (e.g., Cohen's kappa) to the dataset description.
- Include confidence intervals or significance tests for key results (Tables 3, 5, 6) to establish statistical reliability.
- Provide the annotation prompt and LLM evaluation prompt in an accessible appendix or supplementary material.
- Add simple baselines (random, majority class) to contextualize model agreement rates and DPO improvements.
- Analyze why DPO fails for larger models (Llama 70B) or discuss this as a limitation.

---

## wUzBBsrdB1

- GT: Reject (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper identifies a fundamental problem in Sparse Autoencoder (SAE) training: incorrect L0 settings cause SAEs to learn incorrect features. When L0 is too low, SAEs mix correlated features into single latents; when too high, they produce degenerate solutions. The authors demonstrate this through toy model experiments with ground-truth features, provide theoretical proof that MSE loss incentivizes feature mixing when L0 is constrained below the true L0, and propose a proxy metric (decoder pairwise cosine similarity, c_dec) to detect incorrect L0. They validate this metric on Gemma-2-2b and Llama-3.2-1b, showing correlation with peak sparse probing performance.

## Strengths

- **Paradigm-challenging finding**: The paper convincingly demonstrates that the widely-used sparsity–reconstruction tradeoff evaluation is fundamentally flawed—an SAE with correct features achieves worse reconstruction than one that mixes correlated features when L0 is too low. This has significant implications for how the field evaluates SAE architectures.

- **Rigorous theoretical grounding**: The proof in Appendix A.5 formally shows that MSE loss incentivizes feature mixing when L0 is constrained below the true L0, and Appendix A.6 provides theoretical justification for why c_dec increases with feature mixing.

- **Clear experimental validation**: Toy model experiments (Figures 2-6) cleanly demonstrate the failure mode using ground-truth features, and LLM experiments show that c_dec correlates with sparse probing performance, providing external validation.

- **Practical impact**: The analysis of open-source SAEs on Neuronpedia (Figure 22) shows most have L0 < 100 while the paper suggests optimal L0 around 200, directly informing practitioners.

## Weaknesses

- **No interpretability case studies**: The paper claims low-L0 SAEs produce "incorrect" features but never demonstrates concretely what this means. Showing specific latents from wrong-L0 vs. correct-L0 SAEs—with their top activating examples—would make "feature mixing" tangible rather than abstract.

- **Sparse probing as sole validation metric**: Using sparse probing F1 to validate L0 selection is reasonable but limited—it measures downstream utility, not monosemanticity. Orthogonal validation (e.g., autointerpretability scores, manual interpretability evaluation) would strengthen the claim that c_dec-optimal L0 yields more interpretable features.

- **Limited LLM scope**: Only 2-3 layers of two small models are tested. Different layers may have fundamentally different feature distributions, and generalization to larger models is unverified.

- **Flat metric regions complicate practical use**: c_dec sometimes has shallow plateaus rather than unique minima (Figure 8, Gemma), requiring practitioners to identify "elbows" heuristically. This undermines the claim of a single "correct L0" and needs deeper analysis.

- **No comparison with existing L0 selection methods**: MDL-SAEs and AFA-SAEs are mentioned in related work but never empirically compared—practitioners need to know whether c_dec finds better L0 values than existing alternatives.

## Nice-to-Haves

- Analysis of how c_dec behaves when features have heterogeneous firing frequencies (Section 4.2 hints at this but doesn't fully address whether a single global L0 is appropriate).

- Empirical comparison with MDL-SAEs or AFA-SAEs to situate the proposed approach relative to existing methods.

## Novel Insights

The key insight—that reconstruction loss actively penalizes correct features when L0 is constrained below the true value—is non-obvious and important. The paper correctly identifies that sparsity–reconstruction plots, which implicitly assume better reconstruction at given sparsity means better SAEs, would reject the ground-truth solution in favor of one that mixes correlated features. The decoder pairwise cosine similarity metric elegantly exploits the observation that mixed features create systematic non-orthogonality in the decoder. The finding that JumpReLU SAEs maintain better sparse probing at high L0 than BatchTopK (due to per-latent threshold adjustment) is also an interesting secondary contribution suggesting the L0 problem interacts with architecture choice.

## Potentially Missed Related Work

- None identified in this review cycle.

## Suggestions

- Add 2-3 interpretability case studies comparing specific latents from low-L0 vs. correct-L0 SAEs to demonstrate concretely that "feature mixing" produces less interpretable latents.

- Validate with at least one additional metric beyond sparse probing (e.g., autointerpretability scores or feature absorption tests) to establish that c_dec-optimal L0 corresponds to genuinely more monosemantic features.

- Expand LLM validation to additional layers and at least one larger model to establish generalizability.

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (3.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Thoughtbubbles, a transformer architecture that learns to dynamically allocate parallel computation during pretraining through learned forking and deletion of residual streams. The method uses cumulative scores to determine which tokens receive additional "bubble" computation, trained purely with language modeling loss. Experiments across 150M-772M parameter scales show perplexity improvements over parameter-matched and computation-matched baselines on OpenWebText and peS2o, with gains on LAMBADA and HellaSwag zero-shot evaluations.

## Strengths
- **Novel architectural contribution**: The paper presents a genuinely new mechanism for unsupervised learning of adaptive computation during pretraining, addressing a key limitation of existing pause-token approaches that require manual placement or cannot be applied during pretraining.
- **Principled design**: The score attenuation mechanism (equations 8-10) creates a principled gradient signal—tokens that survive must be attended to and updated, encouraging meaningful score assignments without additional supervision.
- **Strong empirical results**: The method consistently outperforms baselines across scales. Notably, the 319M model achieves lower perplexity than the 772M baseline (Figure 3), demonstrating meaningful computational efficiency gains.
- **Learned adaptive behavior**: Figure 5 provides evidence that the model learns to allocate computation at high-entropy tokens without explicit supervision, aligning with theoretical expectations about where additional compute should be directed.
- **Comprehensive baseline comparisons**: The paper compares against both parameter-matched (standard transformer) and computation-matched (copy-N) baselines, controlling for model capacity and computational budget.

## Weaknesses
- **Missing comparison to adaptive computation baselines**: The paper claims to be the "first-known architecture" for unsupervised dynamic parallel computation, but does not compare against relevant methods like Mixture-of-Depths (Raposo et al., 2024) or Universal Transformers (Dehghani et al., 2019). The copy-N baselines are naive and do not establish competitive positioning against prior adaptive compute methods.

- **Extremely limited pretraining scale**: Models are trained for only 2.5B tokens. Modern language model pretraining involves hundreds of billions of tokens, making it unclear whether the observed improvements would persist at practical scales or represent early-training artifacts.

- **No ablation studies on core mechanisms**: The paper does not ablate the score attenuation mechanism, the modified RoPE position encoding, the number/placement of forking layers, or the top-k selection strategy. Performance gains cannot be confidently attributed to the core forking mechanism versus implementation choices.

- **Top-k gradient bottleneck**: The authors acknowledge (Section 8) that hard top-k decisions prevent gradient flow to early layers when tokens with high cumulative scores are dropped later. This is a fundamental architectural limitation, and the ablation in Appendix B showing that more forking layers hurt performance (Table 4) suggests the mechanism may not be learning as intended.

- **Mixed downstream evaluation results**: The method underperforms computation-matched baselines on BLiMP (syntax understanding) across most scales, and results on PIQA are inconsistent. The paper attributes this to short training but does not provide deeper analysis of when adaptive computation helps versus hurts.

## Nice-to-Haves
- Wall-clock training and inference time comparisons to establish practical efficiency beyond FLOPs-matched baselines.
- Evaluation on at least one reasoning benchmark (e.g., GSM8K variants suitable for small models) given the paper's motivation about "complex, multi-step problems."
- Training dynamics analysis showing how forking behavior evolves during training—does allocation stabilize, and when?

## Novel Insights
The concave relationship between entropy and computation allocation (Figure 5) is an interesting finding: the model allocates more computation at moderate-entropy tokens but reduces allocation at highest-entropy tokens. The authors hypothesize this reflects diminishing returns—highest-entropy regions often correspond to clause boundaries or coreferences where additional computation cannot resolve uncertainty, while moderate-entropy regions (e.g., choosing between plausible next tokens) benefit most from extra compute. This aligns with information-theoretic intuitions but deserves deeper investigation.

## Potentially Missed Related Work
- None identified in the search report. However, the paper would benefit from explicit comparison to Mixture-of-Depths (Raposo et al., 2024) and Universal Transformers (Dehghani et al., 2019), which also enable adaptive computation through different mechanisms.

## Suggestions
- Add comparison to at least one modern adaptive computation method (Mixture-of-Depths or Universal Transformers) on the same pretraining setup to establish relative merits.
- Provide proper compute budget analysis with actual FLOPs calculations and inference-time scaling curves showing performance as κ varies.
- Train for longer duration (minimum 10-20B tokens) to validate that perplexity improvements persist beyond early training.
- Add ablation studies on forking layer placement and score attenuation to isolate which components drive performance gains.
- Investigate the counterintuitive result from Appendix B (more forking layers hurt performance) to understand whether the gradient bottleneck is the root cause.

---

## ymUOPsbxLi

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
The paper introduces Nested Subspace Networks (NSNs), an architectural paradigm for enabling dynamic, granular compute adjustment at inference time. By re-parameterizing linear layers as low-rank factorizations with a nested subspace property—where lower-rank models are strict subspaces of higher-rank ones—the method allows a single model to operate across a continuous spectrum of computational budgets. An uncertainty-aware multi-rank training objective jointly optimizes all sub-models, and the approach can be surgically applied to pre-trained LLMs via SVD initialization.

## Strengths
- **Principled architectural foundation**: The nested subspace property (Definition 2) provides a clean mathematical guarantee that Im(W_r) ⊆ Im(W_{r+1}), enabling theoretical bounds on interpolation error (Proposition 1). This is more rigorous than prior elastic inference methods that rely on empirical regularization.
- **Effective post-hoc adaptation**: The surgical SVD-based initialization allows applying NSNs to pre-trained models (Pythia-2.8B, GPT-Neo-2.7B, Gemma-2B, Qwen2-0.5B) without training from scratch, achieving meaningful efficiency gains—e.g., 50% FLOPs reduction with ~5pp accuracy loss on Pythia.
- **Interpretable training dynamics**: The learned log-variances in the uncertainty-weighted objective serve as emergent proxies for rank expressiveness, with higher ranks converging to lower variances (Figure 3), providing insight into how the optimization balances different sub-models.
- **Comprehensive ablation study**: Table 1 systematically isolates the contribution of the "Two CEs" objective versus alternative regularization schemes, demonstrating that joint optimization across ranks—not explicit regularizers—is the key mechanism.

## Weaknesses
- **No empirical comparison to competing dynamic inference methods**: Despite an extensive related work discussion of MatFormer, Flextron, LLAMAFLEX, and DyLoRA, the paper provides no head-to-head empirical comparison. Table 3 is purely conceptual. Readers cannot assess whether NSNs actually outperform these methods in practice, which is essential for establishing practical significance.
- **Narrow evaluation scope**: LLM evaluation is limited to NLI classification. The paper claims broad "applicability to large language models" but does not evaluate language modeling perplexity, generation tasks, or diverse benchmarks (e.g., MMLU, GSM8K, reading comprehension). This limits generalization claims.
- **Attention layers unmodified**: NSNs replace linear layers only in MLP blocks, but attention projections (Q, K, V, O matrices) account for a substantial portion of transformer FLOPs. The reported "50% FLOPs reduction" applies only to modified components; the actual end-to-end efficiency gains are therefore smaller than presented.
- **No deployment mechanism**: The paper demonstrates that different ranks can be selected but does not address how to choose the rank at inference time. Without a routing or selection policy, the practical utility for dynamic environments is incomplete.

## Nice-to-Haves
- Wall-clock inference time and memory footprint measurements would strengthen the efficiency claims beyond theoretical FLOPs calculations.
- Analysis of hyperparameter sensitivity (anchor rank selection, number/spacing of trainable ranks) would guide practitioners in applying the method.
- Qualitative examples showing model behavior at different ranks (e.g., predictions, confidence patterns) would illuminate how performance degrades gracefully.

## Novel Insights
The paper reveals that the energy decay structure (Assumption 1) emerges naturally from the NSN parameterization and training objective but is absent in standard dense models (Appendix C.4-C.5). This structural property—where early basis vectors carry more functional information than later ones—is not an incidental artifact but a direct consequence of jointly training nested sub-models. The analysis showing that low-rank constraints force layers to learn redundant, globally useful functions while higher ranks enable specialization (Figure 8) provides insight into why the hierarchy works: lower ranks are not "crippled" versions of the full model but rather learn fundamentally different, more robust representations.

## Potentially Missed Related Work
- **DyLoRA** (Valipour et al., 2022): Enables dynamic rank adjustment during fine-tuning but targets fixed ranks at inference. An empirical comparison would clarify NSN's advantages in the post-hoc, continuous-spectrum setting.
- **Flextron** (Cai et al., 2024): Converts pretrained LLMs to elastic networks with learned routers. Direct comparison would establish whether NSN's rank-based parameterization offers advantages over head/neuron-level slicing.

## Suggestions
- Add empirical comparison to at least one prior elastic inference method (DyLoRA, MatFormer, or Flextron) on the same task, reporting accuracy-vs-compute curves for fair assessment.
- Expand LLM evaluation beyond NLI to include language modeling perplexity and at least one generation or reasoning benchmark.
- Clarify actual end-to-end efficiency by reporting FLOPs reduction for the full model (including unmodified attention layers) or extend NSNs to attention projections.
- Develop and evaluate a simple rank selection mechanism (e.g., entropy-based routing or compute-budget-aware scheduling) to complete the deployment story.

---

## 21dxwzKCPO

- GT: Reject (avg 2.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
SELU introduces an energy-based framework for LLM unlearning that combines LoRA adapters with straight-through estimators to enable gradient flow through discrete tokens. The key insight is addressing the calibration gap between conservatively fine-tuned models and retain-only baselines by explicitly elevating energy of forget examples while maintaining low energy for retain examples.

## Strengths
- **Novel formulation for unlearning**: The energy-based objective with straight-through estimators provides a principled way to handle discrete tokens while explicitly separating forget and retain examples via an energy margin—a conceptual improvement over suppression-based methods like NPO.
- **Important practical insight about learning rate mismatch**: The observation that conservative fine-tuning rates (1e-5) leave models underconfident on retain data compared to aggressive retain-only training (1e-4) identifies a real but underappreciated calibration challenge in practical unlearning scenarios.
- **Strong empirical results**: On TOFU forget05/forget10 settings, SELU achieves meaningfully higher Forget Quality (~0.31) than NPO (~0.06-0.10) while maintaining comparable Model Utility, with Pareto frontier analysis clearly demonstrating favorable trade-offs.
- **Comprehensive ablations**: The paper systematically ablates STE variants (STL, Gumbel-Softmax, deterministic) and loss components (calibration, coupling, pairwise margin), providing clear evidence for the importance of calibration loss and Gumbel-Softmax estimator.

## Weaknesses
- **Training instability without practical guidance**: The method exhibits a "two-phase trajectory" where forget quality and utility fluctuate non-monotonically (initial bump, crash around epoch 3-5, recovery). The paper acknowledges this but provides no principled stopping criterion or validation metric—critical for real-world deployment where practitioners cannot hand-select epochs.
- **Missing hyperparameter specifications**: The loss function introduces multiple weights (λ_ce, λ_e, λ_calib, λ_cpl), thresholds (τ_low, τ_high), and margin parameters, but the paper does not specify values used. This significantly hinders reproducibility.
- **No statistical rigor**: Tables report single values without standard deviations or confidence intervals across multiple runs. Given the acknowledged training instability, variance characterization is essential.
- **Limited model scale and benchmark diversity**: Experiments are restricted to LLaMA-2-7B on the synthetic TOFU benchmark. Unlearning dynamics may differ for larger models or real-world scenarios (e.g., copyrighted content removal).
- **No adversarial robustness evaluation**: The paper claims knowledge removal but evaluates only KS-test p-values on forget-set paraphrases. Standard unlearning evaluations—extraction attacks, membership inference, paraphrased prompts, multilingual queries—are absent, leaving open whether knowledge is genuinely removed versus suppressed for specific token sequences.

## Nice-to-Haves
- Analysis of energy score distributions across forget/retain examples across epochs to validate the core mechanism visually
- At least one larger model evaluation to demonstrate scalability

## Novel Insights
The learning rate mismatch insight—that conservatively fine-tuned forget+retain models differ fundamentally from aggressively trained retain-only baselines in likelihood calibration—connects unlearning to a broader optimization challenge that extends beyond the specific energy-based formulation. This framing explains why standard unlearning methods may struggle when expected to match retain-only reference models: the baseline itself operates at a different confidence regime. The ablation showing that removing calibration loss (L_cal) causes forget quality to fail—because "energy cannot serve as an anchor" without proper retain-side calibration—provides mechanistic evidence that the energy objective works in concert with, rather than replaces, standard likelihood alignment.

## Potentially Missed Related Work
- No specific missed works identified in this review cycle.

## Suggestions
- Provide all hyperparameter values (λ weights, thresholds, margin, Gumbel temperature) in a table, along with sensitivity analysis showing which parameters are most critical.
- Propose an automatic stopping criterion—e.g., stop when forget/retain energy gap stabilizes on a validation set—to address the training instability practical deployment challenge.
- Add at least one adversarial robustness test (e.g., paraphrased prompts for forgotten entities or extraction via beam search) to strengthen claims about knowledge removal versus token suppression.

---

## ppXAVexrAM

- GT: Reject (avg 4.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
ARSS introduces a decoder-only autoregressive transformer for novel view synthesis from a single image, conditioned on camera trajectories. The key contributions include: (1) a video tokenizer for temporal consistency across views, (2) a camera autoencoder that encodes Plücker ray coordinates as 3D positional guidance, and (3) a spatial permutation strategy that preserves temporal causality while allowing the model to learn from bidirectional spatial context. To the authors' knowledge, this is the first application of GPT-style autoregressive models to novel view synthesis with explicit camera control.

## Strengths
- **Novel problem formulation**: The paper correctly identifies that diffusion-based NVS methods generate views jointly, making strictly causal generation along camera trajectories difficult. Applying decoder-only AR transformers to this task is genuinely novel and well-motivated through the world modeling lens.
- **Principled camera conditioning**: The Plücker ray coordinate autoencoder with geometry-aware loss terms (Eq. 5) provides a principled way to inject 3D positional guidance, incorporating regularization for unit ray length and orthogonality constraints.
- **Effective permutation strategy**: The hybrid ordering approach—shuffling spatial tokens while preserving temporal order—addresses the fundamental mismatch between bidirectional visual context and unidirectional AR models. Ablations in Table 2 demonstrate clear benefits over both raster ordering and full permutation.
- **Comprehensive evaluation across datasets**: Testing on RealEstate10K, ACID, and zero-shot DL3DV with multiple metrics (PSNR, SSIM, LPIPS, FID, FVD) against strong baselines including SEVA, ViewCrafter, LVSM, and MotionCtrl provides thorough validation.
- **Error accumulation analysis**: Figure 6 demonstrates that ARSS maintains quality better than baselines along camera trajectories, supporting the claimed advantage of causal generation for long-horizon synthesis.

## Weaknesses
- **Overstated claims relative to results**: The paper states in the conclusion that it "outperforms state-of-the-art methods," but Table 1 shows mixed results. On Re10K, ARSS achieves better PSNR (+1.1%) and LPIPS, while SEVA achieves better SSIM (+5.8%) and FID. The abstract's "comparable to state-of-the-art" is more accurate than the conclusion's "outperforms."
- **No inference efficiency analysis**: Autoregressive models require sequential token generation, which is inherently slower than parallel diffusion sampling. The paper provides no wall-clock time, memory usage, or throughput comparisons—critical for evaluating whether the causal generation paradigm offers practical advantages or introduces prohibitive costs.
- **Missing reproducibility details**: The camera autoencoder training procedure (joint vs. separate training, dataset used) is not specified. Additionally, the inference token ordering (deterministic vs. random) is unclear—training uses random spatial permutation, but the sampling strategy is not described.
- **Limited resolution and missing scaling analysis**: All experiments use 256×256 resolution, significantly below contemporary NVS methods that operate at 512×512 or higher. The paper acknowledges training "from scratch using limited public datasets with relatively low resolution" as a limitation, but the scalability implications for AR models at higher resolutions remain unexplored.
- **No statistical significance reporting**: Table 1 reports single metric values without standard deviations or confidence intervals. Given the stochastic nature of both AR sampling and evaluation, statistical significance cannot be assessed.

## Nice-to-Haves
- **Camera autoencoder ablation**: The proposed camera autoencoder with Plücker coordinates and geometry losses is presented as a key contribution, but there is no ablation comparing it to simpler alternatives (e.g., sinusoidal positional encodings of camera parameters) to justify the design complexity.
- **Failure case analysis**: Qualitative results show only successful cases. Including representative failure modes (extreme viewpoint changes, textureless regions, complex geometry) would help readers understand practical limitations.
- **Trajectory length limits**: The paper claims better long-horizon behavior but evaluates only 17-frame sequences. Testing on longer trajectories (e.g., 50+ frames) would validate this claim more robustly.

## Novel Insights
The spatial-permutation-while-preserving-temporal-order strategy is a clever insight that bridges the gap between autoregressive image generation (which typically struggles with bidirectional spatial context) and the sequential nature of novel view synthesis. The error accumulation analysis (Figure 6) provides empirical evidence that causal generation genuinely mitigates quality degradation along camera trajectories—a claim often made theoretically but rarely validated. The finding that video tokenization (with temporal compression) outperforms frame-by-frame VQ tokenization by 62% in FVD (Table 3) suggests that temporal consistency is better preserved in the latent space than through post-hoc constraints.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- Revise the conclusion to accurately reflect mixed results: "achieves competitive performance with state-of-the-art" rather than "outperforms."
- Add inference time and memory comparisons against diffusion baselines to substantiate practical viability.
- Report standard deviations across multiple runs for quantitative metrics.
- Clarify whether the camera autoencoder is trained jointly or separately, and specify the inference token ordering strategy.
- Include at least one higher-resolution experiment (e.g., 512×512) to demonstrate scalability.

---

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
The paper identifies a novel "priming vulnerability" in Masked Diffusion Language Models (MDLMs): affirmative tokens appearing at intermediate denoising steps can steer subsequent generation toward harmful responses, even in safety-aligned models. The authors propose Recovery Alignment (RA), which trains models to recover safe responses from intentionally contaminated intermediate states. Experiments across three MDLMs demonstrate significant robustness improvements against both intervention-based attacks and conventional jailbreaks.

## Strengths
- **Novel vulnerability identification**: The paper identifies a genuinely new safety issue specific to MDLMs' iterative denoising mechanism, distinguishing it from ARM-based prefilling attacks. The insight that intermediate tokens during parallel denoising can suppress refusals is important and timely for an emerging architecture.
- **Comprehensive empirical validation**: The paper validates the vulnerability across three MDLMs (LLaDA Instruct, LLaDA 1.5, MMaDA MixCoT) using multiple attack methods (Anchoring Attack, First-Step GCG, PAD, DiJA, PAIR, ReNeLLM, Crescendo), two datasets (JBB-Behaviors, AdvBench), and three evaluation metrics (GPT-4o, guardrail model, keyword matching).
- **Effective proposed solution with theoretical grounding**: RA consistently outperforms baselines (SFT, DPO, MOSA) across most settings. Theorem 4.1 provides a theoretical lower bound enabling efficient First-Step GCG optimization, and empirical validation (Appendix C.2) supports the monotonicity assumption across tested models.
- **Well-designed evaluation methodology**: The anchoring attack provides a controllable probe for vulnerability strength by varying intervention step, enabling systematic analysis of the relationship between attack timing and success rate.
- **Minimal utility degradation**: Table 4 demonstrates that RA preserves general capability across 11 benchmarks, with average accuracy remaining ~52-53% for LLaDA models, addressing potential utility concerns.

## Weaknesses
- **Incomplete robustness under strong attacks**: RA shows substantial residual vulnerability—ReNeLLM achieves 72-80% ASR even on RA-aligned models (Table 3). The paper provides limited analysis of why RA struggles with certain attack types, leaving readers without understanding of failure patterns.
- **Theoretical assumption lacks rigorous justification**: The monotonicity assumption in Theorem 4.1 is empirically validated but not theoretically guaranteed. While Appendix C.2 shows it holds broadly across models and prompts, the paper acknowledges potential edge cases without quantifying their frequency or characterizing when they occur.
- **Limited mechanistic analysis**: The paper empirically demonstrates RA reduces ASR but doesn't analyze what internal representations or behaviors change. Whether RA learns to detect harmful content at intermediate steps or memorizes specific refusal patterns remains unclear, limiting understanding of generalization.
- **No adaptive attack evaluation**: All evaluated attacks are unaware of the defense mechanism. An adaptive attacker could potentially optimize attacks that specifically evade the reward model used in RA training or craft harmful content that doesn't trigger the recovery mechanism.
- **Narrow architectural scope**: Experiments cover only MDLMs with random masking. Generalizability to continuous DLMs or alternative masking strategies (span masking, entropy-based) is untested, limiting claims about the fundamental nature of the vulnerability.

## Nice-to-Haves
- Qualitative failure case analysis explaining what types of attacks succeed against RA and why, which would clarify remaining challenges and guide future work
- Inference-time computational overhead analysis beyond training time, as practical deployment requires understanding latency trade-offs
- Comparison with ARM baselines under comparable attacks (prefilling for ARMs vs. anchoring for MDLMs) to strengthen claims of MDLM-specificity

## Novel Insights
The priming vulnerability concept is genuinely novel: unlike ARM-based prefilling attacks that exploit causal left-to-right generation, MDLMs' parallel denoising allows intermediate tokens to suppress refusals through bidirectional context. The "recovery trajectory" framing—training models to generate safe responses from contaminated states rather than only from clean initial states—provides a principled approach to DLM-specific safety that could generalize beyond the specific implementation. The theoretical connection between first-step log-likelihood and full denoising probability (Theorem 4.1) enables efficient attack optimization that would otherwise require expensive Monte Carlo sampling, and the curriculum-based intervention scheduling (linear increase of t_max during training) offers a practical method for stabilizing training while achieving robustness against progressively stronger attacks.

## Potentially Missed Related Work
- None identified from the search.

## Suggestions
- Add qualitative analysis of RA failure cases under ReNeLLM and similar attacks, including example responses, to illuminate what attack characteristics circumvent the defense
- Include concrete examples with actual queries, harmful suffixes, and model responses at various intervention steps to ground the methodology
- Evaluate RA under adaptive attacks that account for the defense mechanism (e.g., attacks optimizing harmful content probability while minimizing reward model detection)
- Extend experiments to alternative masking strategies to test whether the vulnerability and defense generalize across masking schedules

---

## L2rfd2Czbj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary
The paper introduces **wd1**, a weighted policy optimization method for fine-tuning diffusion-based large language models (dLLMs). The key insight is reformulating the RL objective as a weighted log-likelihood that avoids computing policy ratios—eliminating the need for multiple likelihood approximations and preventing exponential error amplification. The method derives from reverse-KL regularized policy optimization, includes a negative sample penalty term, and is extended to leverage intermediate denoising steps (wd1++).

## Strengths
- **Novel methodological contribution**: The paper directly addresses a fundamental challenge in RL for dLLMs—likelihood ratio computation amplifies approximation errors exponentially. By reformulating as weighted log-likelihood from reverse-KL constrained optimization, the method reduces likelihood approximations from three policies to one, with theoretical analysis showing linear vs. exponential error propagation (Appendix A.1).
- **Strong theoretical grounding**: The paper provides rigorous derivations including: (i) Proposition 2 and Theorem 2 establishing monotonic improvement guarantees for reverse-KL policy optimization; (ii) Lemma 1 and Theorem 1 connecting the objective to energy-guided discrete diffusion training; (iii) Remark 2 interpreting the negative sample term as unlearning. Proofs are provided in Appendix A.
- **Substantial empirical improvements**: wd1 achieves 76.4% on Sudoku vs. 17.6% for reproduced d1 (58.8 percentage point improvement), 51.2% on Countdown vs. 25.8% (+25.4 pp), while wd1++ achieves 44.2% MATH500 and 84.5% GSM8K—outperforming concurrent methods including MDPO. The method works without supervised fine-tuning, unlike d1 which requires it.
- **Computational efficiency demonstrated**: wd1 reduces training time (81.16 vs 103.5 sec/step), FLOPs (8.89 vs 9.92 × 10^15), and NFEs (μ vs μ+2 per step). wd1++ requires 10× fewer rollouts (1,280 vs 19,200–30,000) to achieve comparable or better performance.

## Weaknesses
- **No statistical significance testing**: The paper reports single-run results without error bars, confidence intervals, or significance tests. Given the dramatic improvements on Sudoku and inherent variance in RL methods, this raises reproducibility concerns.
- **Limited baseline comparison**: The primary comparison is against d1. While wd1++ compares with SDPO, TCR, and MDPO, other concurrent approaches mentioned in related work (SPG, d2, UniGRPO) are not directly evaluated. The wd1++ comparison uses different training data (OpenR1) than some baselines, complicating fair evaluation.
- **Limited evaluation scope**: Experiments are conducted only on LLaDA-8B model and only on reasoning/math tasks (Sudoku, Countdown, GSM8K, MATH500). No evaluation on general language understanding benchmarks or other dLLM architectures (e.g., Dream-7B) is provided.
- **Ambiguous percentage claims in abstract**: The "+59% improvement" and "+58.8% over d1" wording conflates percentage points with relative percentage improvement. While the numbers are correct (76.4 - 17.6 = 58.8 percentage points), the phrasing is misleading.

## Nice-to-Haves
- Empirical validation of the linear vs. exponential approximation error claim (Appendix A.1 provides theory but no experimental verification)
- Ablation comparing geometric mixture sampling (π_old^ref) against simpler sampling from π_old alone
- Testing on additional dLLM architectures to demonstrate generality beyond LLaDA

## Novel Insights
The paper establishes a meaningful connection between weighted log-likelihood policy optimization and energy-guided diffusion models. Specifically, the advantage-weighted objective (wd1) can be interpreted as training an energy-guided discrete diffusion model where the energy function equals the negative advantage. This bridges RL fine-tuning and energy-based modeling: the optimal policy from reverse-KL regularization corresponds to energy-guided sampling, and the training objective matches advantage-weighted denoising cross-entropy. The negative sample penalty term further connects to data unlearning literature. This theoretical unification provides deeper understanding of why the approach works.

## Potentially Missed Related Work
- None identified (related work search was skipped)

## Suggestions
- Run multiple seeds and report mean ± standard deviation for all experiments to establish statistical significance
- Clarify the abstract's percentage claims—state "percentage point improvement" explicitly or use relative improvement percentages consistently
- Add comparison with at least one additional ratio-free baseline (SPG or d2) to strengthen the empirical contribution

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

This paper presents a pilot study on applying active learning to flow matching models for shape design applications. The authors develop a theoretical framework using piecewise-linear neural networks to analyze how individual data points influence model diversity and accuracy, deriving the insight that data with identical labels enhance diversity while data with distinct labels improve accuracy. Based on this analysis, they propose two conflicting query strategies—QD for diversity and QA for accuracy—and a hybrid approach with tunable weights, validating their methods on synthetic and real-world shape design datasets.

## Strengths

- **Novel problem formulation**: The paper addresses a genuine gap by developing active learning strategies specifically for generative models (flow matching), whereas most prior work focuses on "generative models for active learning" rather than "active learning for generative models."

- **Insightful theoretical contribution**: The derivation that diversity and accuracy represent inherently conflicting objectives from a dataset perspective is interesting and partially explains a fundamental trade-off in generative models—QD minimizes label distance while QA maximizes it.

- **Computational efficiency**: The proposed query strategies operate directly on the dataset without requiring intermediate retraining of the flow matching model during the query process, which is practically valuable for expensive labeling scenarios.

- **Comprehensive experimental validation**: The paper evaluates on four datasets (synthetic, airfoil, flying wing, starship) with continuous labels relevant to engineering design, showing consistent improvements over discriminative-model baselines.

## Weaknesses

- **Strong theoretical assumptions without empirical validation**: The piecewise-linear neural network assumption is central to the theoretical framework but is not validated empirically. Modern flow matching models typically use complex architectures (U-Nets, transformers) with smooth activations, and the paper provides no evidence that trained models exhibit the assumed interpolation behavior.

- **Limited baseline comparison**: The paper compares against methods designed for discriminative models (coreset, committee, anchor) but does not compare against active learning methods specifically designed for generative models, such as GALISP (Zhang et al., 2024), which is mentioned but not used as a baseline.

- **No statistical significance testing**: Figures 4-7 present results without error bars, confidence intervals, or statistical significance tests. Given the inherent stochasticity in training and sampling, this weakens the conclusions about comparative performance.

- **Heuristic elements not theoretically justified**: The QD query strategy (Eq. 4) includes multiple terms—distance(y, Y), ∆entropy, and distance(x, X)—with weighting coefficients α, β, γ that are not derived from the theoretical framework. The gap between the theoretical analysis (which prescribes selecting data with identical labels) and the practical implementation (which uses similarity rather than exact matches) requires clearer justification.

- **Narrow application scope**: All experiments use continuous labels in shape design domains. The approach's applicability to more common generative tasks with categorical labels (e.g., class-conditional image generation) is unexplored.

- **Missing implementation details**: Key details are absent—RBF network architecture and training procedure for label prediction, cluster thresholds for entropy computation, and hyperparameter values (α, β, γ, ω) are not clearly specified.

- **Counterintuitive claim without investigation**: The paper states that QD "achieves the highest diversity, even outperforming the model trained on the full dataset," which is counterintuitive and warrants deeper investigation rather than being presented as a simple observation.

## Nice-to-Haves

- Empirical validation that trained flow matching models exhibit the piecewise-linear interpolation behavior assumed in the theoretical framework (e.g., testing if generated samples follow Eq. 3 predictions).

- Timing comparisons demonstrating computational efficiency claims versus model-based active learning methods that require retraining.

- Sensitivity analysis for the weighting coefficients α, β, γ, ω to guide practitioners in hyperparameter selection.

## Novel Insights

The key insight of this paper—that the diversity-accuracy trade-off in flow matching models can be understood from a dataset composition perspective—is genuinely novel. The authors show that data points sharing labels with existing training data enhance model diversity by increasing the number of possible interpolations, while data points with novel labels improve accuracy by reducing interpolation error bounds. This provides a principled explanation for why these objectives conflict: adding diverse labels increases interpolation distances (hurting accuracy), while adding identical labels does nothing to reduce interpolation error (limiting accuracy gains). The theoretical connection between the piecewise-linear structure and the resulting data selection principles, while relying on strong assumptions, offers a new lens for understanding generative model behavior.

## Potentially Missed Related Work

- None identified through the review process. The paper references relevant active learning literature and recent flow matching work. Comparison with GALISP (Zhang et al., 2024) would strengthen the experimental validation.

## Suggestions

- Validate the piecewise-linear assumption empirically by testing whether generated samples at novel conditions are indeed interpolations of training samples as Eq. 3 predicts, or discuss when this assumption holds versus when it breaks down for practical architectures.

- Add comparisons against active learning methods specifically designed for generative models (e.g., GALISP) to establish the contribution relative to the most relevant prior work.

- Report mean and standard deviation across multiple experimental runs with different random seeds to establish statistical significance of the reported improvements.

---

## Awf3ebMpKw

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
The paper introduces Expert Merging, a training-light method that learns layer-wise coefficients by aligning merged model hidden states and logits with corresponding experts using only 5–10 unlabeled calibration samples per task. Expert Merging++ extends this with importance-guided chunk-wise coefficients that allocate more parameters to high-importance layers. Experiments on LLMs (Mistral-7B) and MLLMs (InternVL, Qwen2-VL) show consistent improvements over training-free and training-based baselines.

## Strengths
- **Principled alignment objective**: Unlike AdaMerging's entropy minimization (which can produce confidently wrong predictions), the logit alignment loss (Eq. 4) explicitly matches the merged model's output distribution to expert distributions via KL divergence. Ablations confirm removing logit alignment causes substantial degradation (-3.39 avg on InternVL, -3.81 on Mistral).
- **Strong empirical results with practical efficiency**: Expert Merging++ achieves best average scores across all three backbones (58.45 InternVL, 63.63 Qwen2-VL, 48.71 Mistral), outperforming strong baselines including WUDI v2. On InternVL and Qwen2-VL, it even surpasses supervised Mixture Training. The method requires only 5–10 unlabeled samples per task, demonstrated across Tables 11–13.
- **Comprehensive layer importance analysis**: Section 4.3 provides detailed analysis of coefficient magnitudes across layer types (Q/K/V/O, Gate/Up/Down) and depths, showing late-stage MLP layers are most important. This empirically justifies the chunk-wise design.
- **Controllable task trade-offs**: Explicit task weights β_k provide interpretable control over domain priorities (Tables 23–25), allowing users to prioritize specific tasks without catastrophic forgetting—addressing a gap in prior methods lacking transparent control mechanisms.
- **Thorough ablations and reproducibility**: Extensive ablations cover each component across all backbones. Appendix includes sensitivity analyses for regularization (Tables 8–10), initialization (Tables 14–16), and hidden alignment layers (Tables 17–19). Code is available.

## Weaknesses
- **Calibration data source needs clarification**: The paper states "we sample 5–10 unlabeled examples from the training or test sets" (p.6). Using test-set data for calibration, even unlabeled, would compromise evaluation validity. The paper should clarify whether test data is ever used and justify if so.
- **Missing baseline for LLM comparison**: Tables 1 and 2 include Mixture Training results, but Table 3 (Mistral) omits this baseline. This asymmetry limits comparison to the supervised upper bound in the LLM setting.
- **Statistical rigor insufficient**: The paper reports results averaged over 5 runs but provides no standard deviations, error bars, or significance tests. Given reliance on only 5–10 calibration samples, variance characterization is important for assessing reliability.
- **Hyperparameter sensitivity across backbones**: Optimal regularization γ varies substantially (5.0 for InternVL, 0.8 for Qwen2-VL/Mistral). Similarly, initialization ᾱ differs (0.1 vs. 0.3). This requires per-backbone tuning, reducing out-of-the-box applicability.
- **Scalability claims unvalidated**: Experiments use only 3–5 experts, but the abstract claims the approach is "scalable." Without testing with 10+ experts, this claim is unsubstantiated.
- **Computational cost not reported**: Despite claiming "training-light," the paper provides no training time, memory footprint, or FLOPs comparison. Expert Merging++ requires two full optimization stages, and readers cannot assess efficiency claims.

## Nice-to-Haves
- Same-budget comparison for chunking: "Chunk all layers" uses 2× the coefficient budget; a fairer baseline would match Expert Merging++'s budget to isolate the benefit of importance-guided allocation.
- Qualitative examples showing where merged models preserve expert capabilities versus failure cases would strengthen behavior-level analysis.

## Novel Insights
The layer importance analysis (Section 4.3) provides a genuinely novel empirical finding: late-stage MLP layers dominate importance across all three architectures, while attention layers contribute substantially less. This challenges implicit assumptions in prior merging work that treats all layers uniformly. The finding that chunking all layers uniformly degrades performance (Table 4) while importance-guided chunking improves it validates that layer heterogeneity matters and that the proposed importance metric captures meaningful signal. The task-vector cosine similarity analysis (Figure 6) reveals that Chat and Code experts on Mistral share substantial similarity (≈0.42), while Math is nearly orthogonal—suggesting merging dynamics depend on expert complementarity, which future work could exploit.

## Potentially Missed Related Work
- None identified (search was not conducted).

## Suggestions
- Clarify calibration data source explicitly: state whether training data is used exclusively, and if test data is ever sampled, justify why this doesn't constitute leakage.
- Add Mixture Training results for the Mistral LLM setting, or explain its absence.
- Report standard deviations across runs for key results to enable statistical comparison.
- Provide computational overhead analysis (training time, memory) for both Expert Merging and Expert Merging++.

---

## xxsacQ3tdb

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
The paper introduces Rule Picking Rules (RPRs), a framework for selecting among rank aggregation methods based on expected consistency across data collection repetitions. The authors propose Aggregation by Consistency (AbC), which splits evaluators randomly and selects rules minimizing disagreement between outputs. The work provides axiomatic analysis (satisfying reversal symmetry and plurality-shuffling consistency), computational hardness (NP-completeness), and empirical validation across synthetic and real-world datasets.

## Strengths
- **Novel conceptual framework**: The RPR framework addresses a fundamental gap between axiomatic approaches (which face impossibility results) and statistical approaches (which assume ground truth), providing principled rule selection without committing to either paradigm.

- **Strong theoretical results**: The axiomatic analysis is substantial—AbC satisfies reversal symmetry and plurality-shuffling consistency, while the impossibility result (no RPR satisfies reversal symmetry + PSC + union consistency) establishes fundamental limits. The NP-completeness proof for PERFPOS justifies the practical sampling approach.

- **Broad empirical validation**: Experiments span synthetic data (Mallows, Plackett-Luce), peer review (ALMA proposals), sports (Formula 1, Olympics), political elections (PrefLib), and recommender systems (MovieLens), demonstrating wide applicability.

- **Practical impact demonstrated**: The ALMA analysis showing that proposed modifications (Trimmed Borda) hurt consistency exemplifies actionable insights enabled by the framework.

- **Clear ML relevance**: The framing around RLHF, constitutional AI, and agent evaluation provides strong motivation for the ML community.

## Weaknesses
- **No empirical comparison to alternative RPRs**: The paper introduces RPRs as a novel concept but compares AbC only against fixed voting rules (Borda, plurality, etc.), not against other principled RPRs. Natural baselines include welfare-maximizing RPRs or cross-validation-based rule selection. Without such comparison, it is unclear whether AbC's computational overhead is justified.

- **Limited validation for real-world data without ground truth**: On synthetic data, Figure 1 shows ground truth distance correlates with split distance, and model MLEs achieve lowest disagreement. However, for real-world datasets (F1, Olympics, elections), there is no ground truth—only consistency across splits is measured, not independent validation that AbC's choices produce better outcomes.

- **Potential overfitting in optimized positional scores**: The "Best Positional Scores" found via simulated annealing often outperform standard rules, but there is no analysis showing these optimized rules generalize across different random split seeds or held-out data.

- **Missing theoretical guarantee for MLE recovery**: The empirical finding that AbC selects model MLEs under Mallows and Plackett-Luce distributions is compelling but lacks theoretical backing—under what conditions does AbC provably converge to the correct generating model as sample size grows?

## Nice-to-Haves
- Sample complexity analysis for how many random splits are needed for reliable rule selection.
- Experiments varying the split ratio (not just 50-50) to analyze robustness to this design choice.
- Analysis of what profile properties correlate with which rules being selected, improving interpretability.

## Novel Insights
The plurality-shuffling consistency axiom is particularly insightful—it formalizes when a profile's "signal" is concentrated at certain ranking positions (e.g., only top choices are informative). That welfare-maximizing RPRs fail this axiom while AbC satisfies it reveals a fundamental distinction: welfare maximization applies fixed utilities across all profiles, while AbC identifies where each profile's informational content lies. The connection to MVUEs is elegant—both seek estimators minimizing variance between independent samples, with consistency between splits playing the role of minimum variance between independent estimates.

## Potentially Missed Related Work
- None identified.

## Suggestions
- Compare AbC empirically to at least one alternative RPR baseline (e.g., welfare-maximizing RPR or cross-validation-based selection) to establish whether the consistency principle provides measurable benefits over simpler alternatives.
- Add theoretical analysis for MLE recovery: prove conditions under which AbC converges to the correct model as sample size increases, making the empirical Figure 1 results rigorous.
- Validate optimized positional scores on held-out splits to ensure generalization and characterize when overfitting occurs.

---

## azj53PLJRL

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the novel task of Image Quality Assessment (IQA) for Embodied AI, constructing the Embodied-IQA database with 36.9k reference/distorted image pairs and over 5.5 million annotations from Vision Language Models (VLMs), Vision Language Action models (VLAs), and real-world robot executions. The authors propose a Perception-Cognition-Decision-Execution pipeline grounded in Mertonian systems theory and benchmark 15 existing IQA methods, demonstrating significant performance gaps compared to human-oriented IQA.

## Strengths
- **Novel and timely problem formulation**: The paper correctly identifies that traditional human-oriented IQA methods are inadequate for Embodied AI, which requires task-oriented quality assessment across Cognition, Decision, and Execution stages. The Mertonian framework distinguishing predictable "Newtonian" systems (HVS/MVS) from unpredictable "Mertonian" systems (RVS) provides theoretical grounding for why existing IQA fails for robots.
- **Comprehensive database construction**: The Embodied-IQA database is substantial in scale (30 corruption types at 5 severity levels, 2.77M labels each from 15 VLMs and 15 VLAs) with multi-dimensional scoring (Precision/Recall/Semantic for VLM; Position/Rotation/State for VLA) that provides fine-grained quality signals for downstream analysis.
- **Real-world robotic validation**: Unlike most IQA papers relying solely on simulated data, this work includes actual robot experiments (1,500 executions with UR5 arm) validating that Decision scores correlate with Execution outcomes (SRCC > 0.6), demonstrating the practical relevance of the annotations.
- **Thorough benchmarking and analysis**: The evaluation of 15 IQA methods across multiple dimensions (scoring dimensions, distortion sensitivity levels, perspectives, Sim2Real) provides actionable insights. The JND-based distortion sensitivity analysis (Figure 7) revealing different VLM/VLA sensitivities to corruption types offers valuable guidance for future Embodied AI system design.

## Weaknesses
- **No proposed IQA method**: The paper introduces a new task and benchmark but proposes no new method to address the identified gap. While establishing a benchmark is valuable, the absence of even a simple baseline tailored to Embodied IQA limits immediate practical impact and leaves future researchers without a strong starting point.
- **Limited real-world validation scale relative to annotations**: The 1,500 real-world execution samples constitute only 0.03% of the annotation scale (5.5M+), creating an asymmetry between annotation claims and validation evidence. While acknowledged in Limitation 2, the paper does not provide statistical confidence bounds or uncertainty estimates to assess reliability.
- **Low inter-model agreement raises annotation reliability questions**: The reported SRCC correlations between VLMs (~0.3) and VLAs (~0.25) are remarkably low. While the paper argues that averaging multiple models is necessary, it does not analyze whether this averaging produces meaningful ground truth or amplifies noise. Practitioners cannot determine which annotation sources are trustworthy.
- **VLA scoring methodology lacks deeper justification**: The Decision score combines Position, Rotation, and State after 0-1 normalization with equal weighting, but the paper does not justify whether these dimensions are equally important for task success, nor validate this scoring against ground-truth task completion rates beyond the limited real-world sample.
- **No human subjective comparison**: The paper claims HVS and RVS differ fundamentally but provides no experiment comparing human quality ratings with VLM/VLA scores on the same distorted images. Without this direct comparison, the core premise that Embodied AI requires a separate IQA paradigm lacks direct empirical verification.

## Nice-to-Haves
- Per-distortion-type breakdown of IQA performance would help identify which corruption categories are most problematic for Embodied AI and guide targeted metric development.
- Scatter plots of VLA predictions versus real-world execution (rather than aggregate SRCC values) would reveal whether errors are systematic or random, informing calibration strategies.

## Novel Insights
The paper's most significant insight is the empirical demonstration that the Vision Language Action (VLA) stage exhibits fundamentally different quality sensitivity patterns compared to both human visual systems and machine vision systems. The finding that VLAs are more affected by certain distortions (e.g., multiplicative noise) while VLMs are more affected by others (e.g., lens blur)—despite VLMs being a component of VLAs—suggests that the Decision layer introduces non-trivial quality requirements beyond simple perception. This decomposition of embodied perception into Cognition, Decision, and Execution with empirically distinct quality signatures provides a useful framework for future work. Additionally, the cross-database validation showing that training on Embodied-IQA degrades human-oriented IQA performance (and vice versa) confirms the genuine gap between these paradigms rather than merely differing task definitions.

## Potentially Missed Related Work
None identified.

## Suggestions
- Propose at least one baseline IQA method tailored for Embodied AI (e.g., jointly encoding task semantics with visual distortion features) to demonstrate feasibility and establish a benchmark for future work.
- Expand real-world validation to cover multiple distortion types and severity levels (currently limited to level-1 tasks), and provide statistical confidence intervals for the correlation estimates to strengthen reliability claims.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
The paper investigates the reliability of verifiers used in reinforcement learning with verifiable reward (RLVR) for mathematical reasoning. Through comprehensive analysis across multiple datasets and verifier types, the authors find that rule-based verifiers suffer from significant false negatives (14% of correct answers misclassified), while model-based verifiers—despite higher static accuracy—are vulnerable to reward hacking during RL training, with fine-tuned verifiers becoming paradoxically more susceptible to adversarial manipulation.

## Strengths
- **Comprehensive empirical analysis**: The paper evaluates three rule-based verifiers and multiple model-based verifiers across four mathematical datasets plus a general science dataset (WebInstruct-Verified), providing thorough coverage of the verification landscape with consistent methodology.
- **Important finding about scaling**: The observation that stronger models face higher false negative rates from rule-based verifiers (recall drops from ~0.98 for weaker models to ~0.92 for Long-CoT models in Figure 2) directly impacts future RLVR research with increasingly capable models.
- **Counterintuitive discovery about fine-tuning**: The finding that fine-tuned verifiers achieve better static performance but become MORE vulnerable to reward hacking (Table 3: R1-Distill-Verifier-1.5B has 35% adversarial prefix success rate vs. 21.7% for its base model) is surprising and valuable for practitioners.
- **Systematic adversarial probing methodology**: The construction of 13 hacking patterns and evaluation across multiple verifiers provides a useful framework for future robustness research, with clear examples in Figures 11-12.

## Weaknesses
- **No demonstration that policy models naturally discover hacking patterns during RL**: Section 6 shows verifiers are susceptible to hand-constructed adversarial patterns, but the paper doesn't demonstrate that policy models actually discover and exploit these vulnerabilities during normal RL training. The reward hacking observed in Section 5.2 uses the trained R1-Distill-Verifier-1.5B, but the connection between the artificial probing attacks and emergent hacking behavior is not established—this is a significant gap for claims about "hacking vulnerability."
- **No mechanistic explanation for why fine-tuning increases vulnerability**: The paper documents that R1-Distill-Verifier-1.5B becomes more hackable after training (35% vs 21.7% on adversarial prefixes) but offers no explanation. Without understanding whether this is overfitting, distribution shift, or another factor, the community cannot develop principled fixes.
- **Limited model scale investigation**: Experiments cover policy models only up to 7B and verifiers up to 3B parameters. Given that frontier reasoning models use much larger scales (DeepSeek-R1 is 671B), the generalization of findings remains unclear.
- **Lack of statistical significance testing**: Most experiments report single runs without error bars. Table 2 notes "all benchmarks are reported with a single sample due to computational constraints" (except AIME/AMC with Avg@32). For claims about 2.3-point improvements, the lack of repeated runs undermines confidence.
- **Underexplored architecture-robustness connection**: xVerify models (discriminative) resist attacks (<1% success) while generative models fail. The paper notes this but doesn't explain WHY discriminative architecture confers robustness—missing a key insight for building better verifiers.

## Nice-to-Haves
- Quantify computational overhead of hybrid vs. rule-based verification to inform practical deployment decisions.
- Ablation on hybrid verifier design order (rule-based first vs. model-based first) to validate the proposed architecture.
- Comparison with process-supervised verifiers (PRMs) which represent a major alternative paradigm for learned verification.

## Novel Insights
The paper's most significant insight is the fundamental tension between static verification accuracy and RL training robustness. While prior work assumed improving verifier accuracy would benefit RLVR, this paper demonstrates that fine-tuning for accuracy can paradoxically increase susceptibility to exploitation—a finding that challenges the prevailing approach of simply training better verifiers. The observation that discriminative verifiers (xVerify) resist attacks while generative ones fail suggests architecture choice matters for robustness, not just training data or scale.

## Potentially Missed Related Work
- Process Reward Models (PRMs) represent a major class of learned verifiers in reasoning research that may have different robustness properties than the outcome verifiers studied here.
- Formal verification approaches (e.g., proof assistants for mathematical verification) could provide a fundamentally different paradigm that avoids neural vulnerabilities entirely.

## Suggestions
- Include at least 3 random seeds for main experiments to establish statistical significance of reported improvements.
- Add analysis of what emerges during RL training when reward hacking occurs: log response patterns over training to show whether policy models discover specific hacking strategies (like single symbols or gibberish) that match the adversarial patterns identified in probing.
- Provide a mechanistic hypothesis for why fine-tuning increases vulnerability, even if not fully validated—this would guide future research on mitigation.

---

## i1erFkzMIt

- GT: Reject (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper investigates whether capability-specific neurons exist in Large Language Models and whether they exhibit compositional generalization. The authors propose DCSN (Detecting Capability-Specific Neurons) to identify neurons correlated with specific capabilities (primarily arithmetic operations), demonstrate through enhancement/erasure experiments that these neurons are causally related to capabilities, and discover that they activate compositionally when models solve multi-operation problems. Based on these findings, they propose CNLF (Capability Neuron-Level Fine-tuning) which fine-tunes only identified capability-specific neurons (0.05% of parameters), achieving notable performance improvements on unseen datasets compared to full fine-tuning.

## Strengths

- **Novel empirical finding on compositional generalization**: The discovery that capability-specific neurons exhibit compositional generalization—activating together when solving multi-operation expressions—is genuinely novel and provides mechanistic insight into how LLMs combine capabilities. The forward reasoning experiments showing 89.53% activation rate of capability-specific neurons during multi-operation tasks, combined with the ablation study revealing inverse operation correlations (subtraction neurons activating during addition-only tasks), offer substantive evidence for this phenomenon.

- **Rigorous validation methodology**: The enhancement/erasure experimental framework provides strong causal evidence. Fine-tuning only 0.05% of parameters yields ~19.9% improvement while disabling 10% of capability-specific neurons causes ~21.7% performance drop, compared to minimal effects from random parameter manipulation. The cohesiveness (94.3%) and separability (3.6%) metrics provide quantitative validation that identified neurons are reliably localized and distinct across capabilities.

- **Practical parameter-efficient fine-tuning contribution**: The CNLF method demonstrates practical value—achieving comparable fitting performance to full fine-tuning while significantly outperforming on generalization to unseen datasets (improvements ranging from ~7-16% across tasks in Table 6). The discovery that capability neurons can be fine-tuned without disrupting other capabilities addresses a key limitation of prior knowledge/task neuron approaches.

## Weaknesses

- **Narrow experimental scope for core claims**: The compositional generalization experiments focus exclusively on four arithmetic operations (+, -, *, /). While Section 6 extends to math/programming/language capabilities, the compositional generalization phenomenon itself is only demonstrated on arithmetic. The paper claims "capability neurons exhibit compositional generalization" broadly, but this is only validated for basic arithmetic—whether reasoning neurons and language neurons exhibit similar compositional properties remains untested.

- **Inappropriate citation for threshold selection**: The paper justifies σ=6 by citing "statistical principles (Kumar et al., 2007)" which is a Six Sigma case study about automotive customer loyalty—entirely unrelated to neural network analysis. While Appendix D.2 provides empirical validation that σ=6 performs best, the citation is misleading and undermines methodological credibility. The threshold selection should be justified empirically or theoretically without this inappropriate reference.

- **Missing comparison with relevant baselines**: The fine-tuning experiments compare against FT-Random, FT-All, and O-LoRA, but omit standard LoRA and knowledge editing methods (ROME, MEMIT) that also target specific parameters. Without these comparisons, the advantages of CNLF over established methods remain unclear. Additionally, the NeFT comparison in Appendix C is incomplete for the fine-tuning task.

- **Limited mechanistic analysis of detected neurons**: The paper assumes detected neurons encode target capabilities but provides no analysis ruling out confounds such as number representation, syntax patterns, or memorization. No probing experiments or layer-wise analysis investigates what these neurons actually encode or where they reside in the model architecture beyond the FFN focus stated.

## Nice-to-Haves

- **Layer-wise distribution analysis**: Understanding whether capability neurons concentrate in specific layers or distribute across the network would strengthen interpretability claims and guide future work.

- **Error analysis after neuron manipulation**: Qualitative analysis of model outputs after erasure would reveal whether models produce systematic errors or random outputs, providing insight into what capability is actually lost.

## Novel Insights

The ablation finding that subtraction neurons activate during addition-only tasks (6.3% performance drop when erased vs ~0.1-0.2% for multiplication/division) provides a novel insight into the inverse relationship between operations in neural representations. This suggests LLMs develop functionally related neural circuits for inverse operations, which has implications for understanding how mathematical reasoning is organized in neural networks. The discovery that fine-tuning capability neurons improves generalization while fine-tuning task-specific neurons (prior work) degrades other tasks explains a key failure mode of previous localization approaches.

## Potentially Missed Related Work

None identified (related work search was not performed).

## Suggestions

- **Extend compositional generalization experiments to non-arithmetic domains**: Test whether neurons for reasoning, language understanding, and coding capabilities exhibit similar compositional activation patterns to substantiate the broad claims about "capability neurons."

- **Replace the Kumar et al. (2007) citation with proper justification**: Either remove the citation and acknowledge the threshold is empirically determined, or provide theoretical justification based on the actual statistical properties of the neuron activation distribution.

- **Add standard LoRA and knowledge editing baselines**: Include comparison with widely-used PEFT methods and parameter editing approaches to properly position CNLF's contributions.

- **Add probing experiments**: Validate what capability neurons encode through controlled experiments that distinguish capability representation from confounds like number recognition or syntactic patterns.

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the concept of **random set stability** to derive worst-case generalization bounds for stochastic optimization algorithms that produce parameter trajectories. The key contribution is replacing intractable mutual information terms in existing topological generalization bounds with a stability parameter, yielding the first fully computable bounds in terms of box-counting dimension, α-weighted lifetime sums, and positive magnitude. The framework properly recovers classical algorithmic stability bounds (for single iterates) and Rademacher complexity bounds (for fixed hypothesis sets).

## Strengths
- **Addresses a genuine limitation in prior work**: The paper correctly identifies that existing topological/fractal generalization bounds (Simsekli et al., 2020; Birdal et al., 2021; Andreeva et al., 2024) contain intractable mutual information terms, and provides a principled alternative via stability that yields computable bounds.
- **Strong theoretical foundations**: The random set stability definition (Assumption 3.1) meaningfully extends classical algorithmic stability to trajectory settings. Lemma 3.2 establishes that uniform argument stability implies random set stability, providing a verification pathway through established results (e.g., Corollary 3.3 for projected SGD).
- **Recovery of classical results**: Corollaries 3.5 and 3.6 show the framework recovers standard algorithmic stability bounds (Bousquet & Elisseeff, 2002) and Rademacher complexity bounds for fixed hypothesis sets, validating the approach's generality.
- **Comprehensive empirical evaluation**: Experiments on ViT (CIFAR-100) and GraphSAGE (MNISTSuperpixels) investigate both bound tightness and the interplay between stability and topological complexity measures across varying sample sizes and hyperparameters.

## Weaknesses
- **Significant rate degradation**: The main bounds achieve rate O(β_n^{1/3}) rather than O(β_n). When β_n = O(1/n) (typical for stable algorithms), this yields O(n^{-1/3}) versus the classical O(n^{-1/2}) for fixed hypothesis sets. The trade-off between computability and rate is acknowledged implicitly but not analyzed—readers should understand this is not a free lunch.
- **Intractable supremum in stability definition**: Assumption 3.1 requires sup_{z∈Z} |ℓ(w,z) - ℓ(w',z)|, which cannot be computed exactly. The authors approximate this with 500 held-out samples, yielding necessarily optimistic estimates of β_n. While acknowledged, the magnitude of this bias and its impact on bound validity remain unquantified.
- **Loose bounds in practice**: Table 1 shows bounds consistently ~10-100× larger than actual generalization gaps. While the authors argue bounds below 100% are "meaningful guarantees," this threshold is weak by practical standards.
- **Expected bounds only**: The paper provides only expected generalization bounds, not high-probability guarantees. This limits practical applicability where confidence intervals are needed for deployment decisions.
- **Unverified assumptions**: Assumption 4.1 requires local Lipschitz continuity on each trajectory, but no discussion is provided on verifying this for common architectures or how to estimate L_{S,U}.

## Nice-to-Haves
- Analysis of when the O(β_n^{1/3}) rate is actually preferable to classical bounds, or discussion of whether this rate can be improved under additional assumptions
- Sensitivity analysis for the J parameter choice beyond the theoretical optimum
- Validation on settings with analytically known stability parameters (e.g., SGD for strongly convex objectives) to verify estimation procedures

## Novel Insights
The paper makes a meaningful connection between the stability literature and topological complexity measures. The key insight is that stability-based approaches can eliminate intractable mutual information terms while preserving connections to empirically relevant complexity measures. The empirical finding that topological complexity measures increase with sample size (Figures 8-13) and that their correlation with generalization gap strengthens with n (Figures 2-7) provides empirical support for the theoretical framework, particularly the product structure in Theorem 4.4 where β_n^{1/3} multiplies the complexity term.

## Potentially Missed Related Work
- None identified through the review process. The paper appropriately cites prior work on topological generalization bounds (Birdal et al., 2021; Andreeva et al., 2024), fractal dimensions (Simsekli et al., 2020), and stability-based approaches (Hardt et al., 2016; Foster et al., 2019).

## Suggestions
- The δ_n parameter in Theorem 4.3, which depends on uniformity of box-counting dimension convergence, should be discussed in the main text rather than deferred to the appendix, as it affects bound interpretability.
- Provide upper bounds on the stability parameter β_n that can be verified without taking suprema over Z, or validate the 500-sample approximation against scenarios where theoretical bounds on stability are known.
- A visualization showing how bound values and actual generalization gap co-evolve across training iterations (not just final values) would strengthen the empirical validation of whether the bounds track generalization dynamics.

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary
This paper provides a blueprint for rigorous empirical ML research through a detailed case study of Nguyen et al. (2024), a high-visibility ICLR 2025 Oral paper on min-p sampling. Through comprehensive re-analysis of the original paper's four lines of evidence—human evaluations, NLP benchmarks, LLM-as-Judge evaluations, and community adoption claims—the authors demonstrate that min-p's claimed superiority is not supported by its own data when analyzed correctly. The paper derives methodological recommendations from this case study.

## Strengths
- **Systematic re-analysis across all evidence types**: The paper examines each of the original paper's four lines of evidence with careful attention to methodological correctness, discovering omitted human evaluation data (one-third of collected scores), selective reporting in LLM-as-Judge results, and retracted community adoption claims—each verified through public documentation and author interactions.

- **Novel Best-of-N methodology for fair comparison**: The proposed approach to control for hyperparameter tuning volume across methods is a genuine methodological contribution, visualized across 12 models in Figures 4-5 and 7-8, showing that min-p's apparent advantages disappear when each sampler is given equal hyperparameter search budgets.

- **Correct statistical analysis applied**: Table 1 properly applies Bonferroni correction for 12 comparisons and introduces Intersection-Union Tests to evaluate the "consistently outperforms" claim—only 1 of 12 comparisons remains significant after correction, directly contradicting the original claim.

- **Substantial experimental investment**: The GSM8K sweep required approximately 6000 A100-hours across 9 models, 4 samplers, 31 temperatures, and 6 hyperparameter values per sampler, demonstrating commitment to thorough evaluation.

## Weaknesses
- **Best-of-N methodology lacks formal validation**: While the hyperparameter volume control approach is intuitive and empirically useful, the paper does not validate that this method reliably detects selective reporting or provide theoretical grounding for when it might fail. The approach assumes the chosen hyperparameter grid adequately covers relevant parameter space.

- **GPQA benchmark analysis missing**: The original paper evaluated both GSM8K and GPQA, but this re-analysis covers only GSM8K due to compute constraints. While the GSM8K analysis is thorough, the NLP benchmark conclusions remain incomplete without GPQA analysis.

- **Effect sizes not reported or discussed**: The paper focuses on statistical significance but does not report effect sizes for human evaluation differences. Readers cannot assess whether differences are trivial vs. meaningful in magnitude, only whether they cross significance thresholds.

- **Qualitative annotation methodology underspecified**: Section 2.3 states qualitative responses were "manually annotated" but provides no details about annotation protocol, number of annotators, inter-annotator agreement, or annotation guidelines.

- **Limited exploration of boundary conditions**: The analysis focuses on demonstrating min-p is not superior, but provides little characterization of when different samplers might be preferable (e.g., specific temperature ranges, model families, or task types). The finding that samplers perform comparably deserves more nuanced discussion.

## Nice-to-Haves
- An adequately powered, independently conducted human evaluation study demonstrating what conclusions correct methodology would yield would substantially strengthen the blueprint contribution.
- Statistical power analysis for the original human evaluation (n=53) would help readers understand whether the "no significant difference" findings reflect true equivalence or insufficient power to detect meaningful effect sizes.

## Novel Insights
The paper's most important insight is that methodological advantages can evaporate when hyperparameter tuning budgets are equalized. The Best-of-N framework reveals that min-p appeared superior not because of inherent algorithmic properties, but because it received substantially more hyperparameter tuning than baselines (2× to 10× more per Figure 6). This finding has implications beyond this case study: any empirical comparison of methods should account for differential tuning efforts. The taxonomy of four issues (omitted data, incorrect statistics, mischaracterized qualitative feedback, selective reporting) provides a useful template for reviewers evaluating empirical ML papers.

## Potentially Missed Related Work
- None identified. (Related work search was skipped.)

## Suggestions
- Develop the Best-of-N methodology more formally, including discussion of its assumptions, limitations, and validation beyond this single case study.
- Add a figure showing qualitative text examples comparing outputs from different samplers to help readers assess practical relevance of the null findings.

---

## e3Z60Ri5JP

- GT: Withdrawn (treated as Reject) (avg 2.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
AttackSeqBench introduces a benchmark for evaluating large language models' ability to analyze adversarial attack sequences from Cyber Threat Intelligence (CTI) reports. The authors construct three hierarchical tasks (Tactic, Technique, and Procedure inference) using real-world CTI reports, evaluating 7 LLMs, 5 Large Reasoning Models (LRMs), and 4 post-training strategies across three inference settings (Zero-shot, Context, RAG-empowered).

## Strengths
- **Addresses a genuine gap**: Existing CTI benchmarks focus on entity extraction and attribution; sequential reasoning about adversarial behaviors remains underexplored. This fills an important need for proactive cybersecurity.
- **Principled benchmark design**: The 4-tuple formalization (Tactic Sequence, Technique Mappings, Procedure Mappings, CTI Outline) grounded in MITRE ATT&CK provides a clean mathematical framework. The six-criteria refinement process with both human and automatic evaluation demonstrates quality control.
- **Comprehensive experimental coverage**: Testing 7 LLMs, 5 LRMs, and 4 post-training strategies across 3 tasks and 3 settings provides broad coverage of current approaches.
- **Counterintuitive findings**: The discovery that LRMs often underperform LLMs on attack sequence analysis (contrasting with their advantages in math/coding) is non-trivial and provides actionable insights for domain-specific model development.
- **Detailed analysis**: The reasoning path analysis (Table 4) and RAG error analysis (Figure 6) provide qualitative insights into model behaviors beyond raw metrics.

## Weaknesses
- **Missing statistical rigor**: Tables 2 and 3 report only point estimates without confidence intervals or significance tests. Performance differences between models are often small, and readers cannot assess whether observed differences are meaningful or within sampling variance.
- **Circular evaluation concern**: GPT-4o is used for question generation and refinement while also being evaluated as a model. Human evaluation partially mitigates this, but potential bias from the generation model's knowledge distribution is not systematically analyzed.
- **Ground truth validation insufficient**: The benchmark relies on automated LLM-extraction of TTPs from CTI reports without systematic validation that extracted sequences accurately represent source reports. Human evaluation focuses on question quality, not attack sequence extraction accuracy.
- **Limited failure analysis**: RAG error analysis examines only GPT-4o responses, and reasoning path analysis contrasts only one LLM-LRM pair. Broader analysis across models and tasks would strengthen conclusions.
- **Inter-annotator agreement unreported**: Three cybersecurity experts evaluate questions, but no agreement metrics (Cohen's Kappa, ICC) are provided, making it difficult to assess ground truth reliability.

## Nice-to-Haves
- Include simpler baselines (frequency-based, keyword matching) to establish lower bounds and contextualize LLM performance levels.
- Add analysis of dataset coverage by APT group, time period, and attack type to help readers assess representativeness of real-world threats.

## Novel Insights
The finding that Large Reasoning Models (LRMs) often underperform standard LLMs on attack sequence analysis is counterintuitive and significant. The reasoning path analysis reveals that LRMs can "overthink"—constructing redundant reasoning loops that amplify minor misunderstandings into incorrect conclusions. In Table 4, the LLM correctly identifies temporal constraint misalignment, while the LRM gets distracted by plausibility reasoning. This contrasts sharply with LRM advantages in mathematics where extended reasoning chains reliably improve accuracy. The insight is that domain-specific knowledge integration, not reasoning chain length, may be the bottleneck for cybersecurity tasks.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- Add confidence intervals (e.g., bootstrap or binomial) to all results tables to enable readers to assess statistical significance of performance differences.
- Report inter-annotator agreement metrics (Cohen's Kappa or ICC) for human evaluation to establish ground truth reliability.
- Validate a sample of LLM-extracted attack sequences against source CTI reports to establish extraction accuracy.

---

## qZVgiiWQUt

- GT: Reject (avg 2.5)
- Predicted: N/A (3.5/10)
- Match: N/A

### Final Review

## Summary

ArchPilot introduces a multi-agent neural architecture search framework that reduces computational cost by replacing expensive full training evaluations with adaptive multi-proxy scoring. The system decomposes NAS into three specialized agents—Orchestration (MCTS-based search), Generation (LLM-driven code production), and Evaluation (proxy training and weight optimization)—that collaborate to efficiently explore the solution space. Experiments on MLE-Bench demonstrate improved performance over AIDE and ML-Master baselines across task difficulty levels and compute budgets.

## Strengths

- **Well-motivated problem formulation**: The paper correctly identifies that existing LLM-based ML agents waste computational resources on full training runs for unpromising candidates, and proposes a principled solution through proxy-guided evaluation with selective full-training escalation.

- **Clean architectural decomposition**: The three-agent design (Orchestration, Generation, Evaluation) provides clear separation of concerns, enabling modular upgrades and interpretability. Each agent has well-defined responsibilities: GA handles code generation, EA provides quantitative evaluation, and OA manages search and resource allocation.

- **Principled proxy aggregation mechanism**: The ridge-regularized least-squares weight fitting with simplex projection and hard-zero policy for failed proxies is technically sound. The mechanism adaptively reweights proxies based on accumulated ground-truth observations, improving upon static weighting schemes.

- **Comprehensive benchmark evaluation**: Testing across 75 diverse MLE-Bench tasks (tabular, vision, NLP) with varying difficulty levels provides evidence for generalizability. The budget sensitivity analysis in Figure 2 showing consistent improvements across GPU-hour constraints is informative.

- **Tree restart mechanism**: The insight that proxy weight changes should trigger tree restarts to prevent stale MCTS statistics from misleading search is well-integrated and theoretically justified.

## Weaknesses

- **Complete absence of ablation studies**: Despite claiming contributions from multi-proxy evaluation, adaptive reweighting, and tree restarts, the paper provides no ablations to isolate component contributions. Critical missing comparisons include: single-proxy vs. multi-proxy, fixed weights vs. adaptive reweighting, and search with vs. without restarts. This substantially weakens the ability to attribute improvements to specific design choices.

- **No statistical significance reporting**: Tables 1 and 2 report only point estimates across 75 tasks without confidence intervals, standard errors, or significance tests. Given modest improvements (e.g., valid submission rate 0.893 vs. 0.867; ranking 0.6149 vs. 0.6535), readers cannot assess whether observed differences are meaningful or within noise.

- **Proxy-ground-truth correlation not validated**: The method's central premise—that proxy scores predict true performance—lacks empirical validation. No scatter plot, correlation analysis, or prediction error metrics demonstrate whether the learned weights actually predict full-training outcomes. Without this, the proxy optimization mechanism is unvalidated.

- **Key hyperparameters underspecified**: Critical values are missing: UCT exploration constant C, ridge regularization parameter α, thresholds τ_improve and τ_debug for terminal nodes, weight-change threshold ε for triggering restarts, and the top-k value for reseeding after restarts. This impedes reproducibility.

- **Compute efficiency claims unsubstantiated**: The paper claims efficiency gains but provides no breakdown of actual time/compute spent on proxy evaluations versus full training. The number of proxy evaluations per full-training run, proxy training duration, and total computational savings are not quantified.

- **Abstract-experiment discrepancy**: The abstract references results on "MLE-Bench Lite" with specific percentage claims ("surpasses 66% of teams"), but Section 4 reports results on the full 75-task MLE-Bench using normalized ranking metrics. The relationship between these metrics and the abstract's percentages is unclear.

## Nice-to-Haves

- Comparison against established zero-cost proxy baselines (SNIP, GraSP, SynFlow) to contextualize whether the multi-proxy approach offers advantages over existing efficient NAS methods.

- Analysis of proxy weight evolution over search iterations, showing which proxies receive higher weights and whether the hard-zero policy effectively prunes unreliable proxies.

## Novel Insights

The restart-enabled MCTS with dynamic score recalibration when proxy weights change is a technically interesting contribution. Traditional MCTS assumes stationary reward functions, but ArchPilot addresses the non-stationary setting where evaluation semantics shift as ground-truth observations accumulate. The simplex-projected ridge regression for proxy weight optimization, combined with the hard-zero policy for failed proxies, provides a principled way to combine heterogeneous signals while maintaining interpretability. These mechanisms could generalize beyond NAS to other domains where cheap surrogate evaluations guide expensive search.

## Potentially Missed Related Work

- **Zero-cost NAS proxies**: The paper's three proxies (one-epoch validation, noisy validation, feature-dropout) are standard heuristics. Related work on zero-cost proxies such as SNIP (Lee et al., 2018), GraSP (Wang et al., 2020), and SynFlow (Tanaka et al., 2020)—which compute architecture quality from single forward/backward passes without training—could be relevant for comparison or integration. NAS-Bench-Suite-Zero (Krishnakumar et al., 2022) systematically benchmarks these approaches and finds that combining multiple proxies improves correlation with ground-truth performance, which aligns with ArchPilot's multi-proxy philosophy.

## Suggestions

- **Run ablation experiments** comparing: (a) ArchPilot with single-proxy vs. multi-proxy, (b) fixed uniform weights vs. adaptive reweighting, (c) MCTS with vs. without restarts. These directly test the claimed contributions.

- **Report proxy-ground-truth correlation**: Compute Pearson or Kendall's Tau correlation between aggregated proxy scores and true validation metrics across all evaluated candidates to validate the proxy pipeline.

- **Add uncertainty estimates**: Include bootstrap confidence intervals for ranking metrics or report standard deviations across tasks; clarify whether improvements are statistically significant.

- **Quantify computational overhead**: Report the average proxy evaluation time, number of proxy evaluations per full-training run, and total compute breakdown to substantiate efficiency claims.

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
AdaSVD proposes an adaptive SVD-based compression method for LLMs with two contributions: (1) adaComp, which compensates for truncation errors through alternating updates of singular matrices using Moore-Penrose pseudoinverse optimization, and (2) adaCR, which assigns layer-specific compression ratios based on activation similarity. Experiments across multiple LLM families show consistent improvements over prior SVD-based methods, particularly at higher compression ratios.

## Strengths
- **Novel compensation mechanism**: adaComp addresses truncation error by directly optimizing retained singular matrices via alternating least-squares with pseudoinverse stabilization. The reformulation addresses numerical instability issues in naive matrix inversion, a real problem at high compression ratios.
- **Strong empirical validation**: Consistent improvements across 4 LLM families (LLaMA2-7B, OPT-6.7B, Vicuna-7B, Mistral-7B) and 8 evaluation benchmarks. At 60% compression on WikiText-2, AdaSVD achieves 50.33 perplexity vs 89.90 for SVD-LLM—a 44% relative improvement (Table 2).
- **Effective at aggressive compression**: Improvements are most pronounced at 50-60% compression where prior methods degrade severely, addressing a key practical limitation.
- **Comprehensive ablation structure**: Tables 3a-3d isolate contributions of adaComp, adaCR, iteration counts, and minimum retention ratios, helping readers understand component importance.
- **Orthogonality to quantization demonstrated**: Table 4 shows AdaSVD can combine with GPTQ-INT4, with AdaSVD+INT4 outperforming SVD-LLM+INT4 across compression ratios.
- **Layer importance analysis**: Figure 4 reveals a "bowl-shaped" importance pattern in LLaMA models, with early and late layers showing higher importance—providing empirical rationale for non-uniform compression.

## Weaknesses
- **Missing computational cost analysis**: The paper does not report wall-clock time or peak memory usage for the adaComp compression process. Since alternating updates iterate up to 15 times per layer in ablations, the practical overhead is unknown, which is critical for deployment decisions.
- **Hyperparameters unspecified for main results**: While ablations explore k ∈ {1, 3, 15} and various mrr values, the exact configurations used for Tables 1-2 are not stated. This hinders reproducibility.
- **Limited justification for importance metric**: adaCR uses cosine similarity between input and output activations as the importance metric, justified only by "for simplicity." No empirical comparison to alternatives (gradient-based, Hessian-based) or theoretical motivation is provided.
- **No inference latency measurements**: The paper claims SVD "accelerates model inference" in the introduction but provides no latency or throughput measurements on actual hardware.
- **VLM evaluation lacks quantitative metrics**: Figure 5 shows only qualitative image captioning examples for LLaVA without standard metrics (BLEU, METEOR, CIDEr).
- **No calibration data sensitivity analysis**: Results depend on 256 random WikiText-2 samples, but variance across random seeds or sensitivity to calibration dataset choice is not reported.

## Nice-to-Haves
- Direct comparison with mainstream quantization methods (GPTQ, AWQ) at similar memory footprints, beyond the combined SVD+quantization experiments.
- Convergence analysis (theoretical or empirical) for the alternating update procedure in adaComp.

## Novel Insights
The layer-wise importance analysis reveals that early transformer layers consistently carry highest importance across LLM families, with a characteristic "bowl shape" in LLaMA models where both early and late layers are more important than middle layers. This provides empirical grounding for the adaCR strategy. The reformulation of truncation compensation as an alternating optimization problem—solving for U and V^T sequentially via pseudoinverse rather than closed-form update—is a technically sound approach to avoiding numerical instability.

## Potentially Missed Related Work
- None identified. The related work section adequately covers SVD-based compression methods (FWSVD, ASVD, SVD-LLM) and broader compression techniques.

## Suggestions
- Specify all hyperparameter values (iteration count, bucket size, minimum retention ratio) for main experiments in the paper or appendix.
- Add wall-clock time for compression and inference latency measurements for practical deployment assessment.
- Include quantitative metrics for VLM experiments (BLEU/CIDEr on COCO captioning).
- Report variance across random seeds for calibration data sampling to assess result stability.

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether LLM-generated bibliographies can be distinguished from human-curated reference lists through structural and semantic analysis of induced citation graphs. Using 10,000 focal papers from SciSciNet (~275k references), the authors construct paired citation graphs (ground truth vs. GPT-4o/Claude-generated) and find that structural features alone achieve only ~60% accuracy (near-chance), while semantic embeddings yield 83% (Random Forest) to 93% (GNN) accuracy—demonstrating that LLM bibliographies mimic human topology but leave detectable semantic fingerprints.

## Strengths
- **Large-scale systematic study**: The dataset of 10,000 focal papers with ~275,000 references, paired design (same focal papers for ground truth and LLM conditions), and multiple robustness checks (Claude Sonnet 4.5, SPECTER embeddings, subfield-level and temporally-constrained baselines) provide substantial statistical power and strengthen confidence in the findings.
- **Progressive methodological decomposition**: The stepwise analysis from interpretable graph-level descriptors → aggregated embeddings → GNNs cleanly isolates what signal is captured by each approach, making the conclusion well-supported that topology is indistinguishable while semantics provide detection signal.
- **Meaningful cross-model generalization**: Training on GPT-4o and testing on Claude achieves ~68-80% accuracy, demonstrating that the semantic fingerprint is partially shared across LLMs rather than being model-specific artifacts.
- **Strong control experiments**: The random-vector control showing accuracy drops to chance when embeddings are replaced with i.i.d. vectors, and the PCA-dimension ablation showing accuracy tracks explained variance, convincingly establish that separability stems from semantic structure rather than raw dimensionality.
- **Clear practical implications**: The finding that detection should target content signals rather than graph structure provides actionable guidance for auditing LLM-generated bibliographies.

## Weaknesses
- **Semantic fingerprint remains uninterpreted**: The paper demonstrates that semantic embeddings separate LLM from human references but provides minimal analysis of which semantic dimensions drive separability. Is it recency bias? Venue prestige? Author overlap? Topical drift? Without feature importance analysis or probing of specific semantic properties, the "semantic fingerprint" remains a black box, limiting both scientific understanding and practical utility.
- **Temporal violations may confound detection signal**: The paper notes that ~6% of GPT-generated references cite papers published after the focal paper—a temporal impossibility in genuine bibliographies—but does not control for this in the main experiments. The classifier may partially exploit this trivial artifact rather than meaningful semantic differences. A clean ablation removing temporally-violating references would strengthen the claim that detection operates on genuine semantic content.
- **GNN test results lack uncertainty estimates**: While RF results report standard deviations across 10 runs, Table 3 reports GNN test results without uncertainty despite visible variance in validation distributions (Figure 4). This makes it difficult to assess the reliability of the reported 93% accuracy.
- **Graph construction loses citation direction**: Converting directed citation graphs to undirected (p. 4) discards potentially useful temporal ordering information—the direction of who cites whom—which could contain structural signatures beyond the five centrality measures tested.
- **Selection bias from excluded graphs unanalyzed**: 779 graphs (7.8%) were excluded because GPT-4o generated no valid references. The paper does not analyze whether these excluded papers differ systematically from the retained ones (e.g., in field distribution, publication year, or reference count), leaving open potential selection bias in the conclusions.

## Nice-to-Haves
- Include perplexity-based or simpler detection baselines to contextualize whether graph-based approaches offer advantages over text-based detection methods.
- Explore attention-weighted or learnable graph pooling strategies to validate whether simple embedding summation is optimal.

## Novel Insights
The paper offers a clean decomposition of what LLMs learn about citation behavior: they internalize the *topology* of scholarly networks (who connects to whom, hub structure, local triadic closure) but not the precise *semantics* of which papers belong together for a given research context. This distinction has implications for how we think about LLM knowledge representation—parametric knowledge captures structural regularities but encodes distinctive semantic preferences that likely reflect training data biases (prestige venues, recency, reduced self-citation). The cross-generator generalization result is particularly interesting: GPT-4o and Claude share enough of a "semantic fingerprint" that a classifier trained on one partially generalizes to the other, suggesting these biases may be common across LLM families trained on similar scientific corpora.

## Potentially Missed Related Work
- Work on citation network generation and citation recommendation (e.g., Huang et al., "Citation Reason Classification for Context-Aware Citation Recommendation") could contextualize how LLM citation patterns compare to traditional recommendation systems.
- Research on temporal citation networks and citation order (e.g., Clough et al. on citation temporal ordering) could inform whether the 6% temporal violations are anomalous relative to human patterns.
- Studies on embedding-based detection of AI-generated content (e.g., OpenAI's AI text classifier, DetectGPT, watermarking methods) could provide baselines for comparison even if targeting text rather than bibliographies.

## Suggestions
- Add a temporal-violation ablation: retrain classifiers on graphs where all GPT-generated references are temporally valid (published before or in the same year as the focal paper). If accuracy remains high, this would strengthen claims about semantic fingerprints; if accuracy drops substantially, the contribution needs reframing.
- Report hallucination rates: state what fraction of GPT/Claude reference suggestions were not found in SciSciNet (not just the 779 focal papers with zero valid references), as this characterizes the selection process and potential biases.
- Provide feature importance analysis: RF feature importance or SHAP values on embedding dimensions would reveal which semantic aspects drive detection, making the "semantic fingerprint" interpretable.

---

## XX5EZoe4ec

- GT: Reject (avg 2.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
RetrievalFormer proposes a dual-encoder transformer architecture for sequential recommendation that enables efficient ANN-based retrieval while handling cold-start items through feature-based encoding. The key contribution is reformulating next-item prediction from a softmax classification over all items to a retrieval problem in a learned embedding space, achieving 86–91% of transformer baseline accuracy with 288× speedup at 10M items, while enabling zero-shot recommendation of items unseen during training.

## Strengths
- **Addresses critical deployment challenges**: The paper correctly identifies two fundamental limitations of ID-softmax transformers—O(N) inference cost and inability to score new items—and provides a practical solution that bridges academic advances with production requirements.
- **Rigorous cold-start evaluation**: The Leave-One-Out Cold (LOOC) protocol ensures zero item ID leakage between training and evaluation, providing a methodologically sounder assessment than standard protocols that test on items seen during training. The paper honestly reports the 25–35% performance drop under LOOC.
- **Strong efficiency demonstration**: The latency scaling analysis from 10K to 10M items with IVF-PQ provides concrete empirical evidence of sub-linear scaling, with Figure 2 clearly demonstrating the 288× speedup claim.
- **Well-designed architecture with empirical validation**: The AttentionFusion mechanism shows measurable improvement over mean pooling (+10.1% Recall@20 in ablations), and shared embeddings provide parameter efficiency and semantic consistency.
- **Reproducibility**: Detailed hyperparameters, training schedules, and FAISS configurations are provided, using standard public datasets with preprocessing following prior work.

## Weaknesses
- **Missing uncertainty estimates for RetrievalFormer**: Table 1 reports baseline results "averaged over five runs with std < 0.001 not reported" but RetrievalFormer results lack standard deviations entirely. This omission prevents fair statistical comparison and should be addressed.
- **Limited comparison to two-tower retrieval baselines**: The paper compares primarily against ID-softmax transformers, but dual-encoder retrieval models are the most natural baselines. Without comparison to established two-tower recommenders, it's unclear whether the transformer-based user tower provides meaningful gains over simpler retrieval architectures.
- **Gap with strongest baseline unexplained**: AttrFormer achieves Recall@20 of 0.4128 on MovieLens-1M versus RetrievalFormer's 0.337 (81.6% relative). The paper notes this as a "notable outlier" but does not investigate whether AttrFormer's architectural innovations could inform RetrievalFormer's design.
- **No embedding space quality analysis**: The paper claims InfoNCE produces ANN-suitable embeddings but provides no analysis of embedding space properties—uniformity, alignment, dimensional collapse, or cluster structure. This claim remains empirically unsubstantiated.
- **Cold-start comparison limited to single baseline**: The LOOC evaluation compares against only a content-based KNN approach. Comparing against other feature-based or cold-start methods would strengthen the analysis.

## Nice-to-Haves
- Analysis across multiple retrieval cutoffs (K ∈ {10, 20, 50, 100}) to understand retrieval quality across different funnel stages
- Embedding space visualization (t-SNE/UMAP) showing item clusters and user-item alignment patterns
- Failure case analysis identifying when RetrievalFormer underperforms (sparse history users, tail items, specific categories)
- Exploration of ANN recall-quality trade-offs beyond the single n_probe=32 configuration

## Novel Insights
The key insight is reframing sequential recommendation from vocabulary-limited classification to retrieval in a learned embedding space. This architectural choice is fundamentally different from prior two-stage approaches that retain ID-softmax models for ranking, and different from feature-enhanced transformers that still predict over fixed vocabularies. The LOOC protocol exposes a genuine capability gap: ID-softmax models score zero on held-out items while feature-based encoders maintain meaningful ranking ability. The shared embedding design across user and item towers—where the same categorical value uses identical representation regardless of context—creates semantic consistency that benefits both cold-start generalization and training efficiency. The non-monotonic relationship between history length and performance (the jump at L=25) suggests interesting threshold effects in transformer sequence modeling that warrant further investigation.

## Potentially Missed Related Work
- Related work search was not performed for this review cycle. Authors should ensure comparison with established two-tower retrieval methods (e.g., sampled softmax, YouTube's dual-encoder architecture, Facebook's EBR) and cold-start approaches (e.g., DropoutNet, embedding propagation methods).

## Suggestions
- Add standard deviations for RetrievalFormer results across multiple runs to enable statistical comparison with baselines
- Include at least one two-tower retrieval baseline to contextualize performance relative to established retrieval approaches
- Provide embedding space analysis (uniformity/alignment metrics or visualization) to substantiate claims of ANN-suitability
- Investigate and explain the non-monotonic history length behavior observed in ablations
- Add brief discussion of broader impacts (fairness, filter bubbles, deployment costs) for completeness

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper introduces CANON (Conditional AdvaNtage estimatiON), a novel advantage estimation method for reinforcement learning with verifiable rewards (RLVR) in large reasoning models. CANON regroups sampled responses into two groups based on a target metric (e.g., entropy, response length), computing inter-group advantages to identify which metric trend yields higher accuracy and intra-group advantages to select better responses within groups—thereby avoiding the need for handcrafted directional priors (higher/lower-is-better) used in prior methods. The approach is theoretically grounded and demonstrates consistent improvements over DR.GRPO across three LLMs on mathematical and logical reasoning tasks.

## Strengths

- **Novel and well-motivated methodological contribution**: The conditional regrouping approach provides a principled way to incorporate metric signals without imposing directional biases. The insight that DR.GRPO is a special case with μ=0.5 (equal weighting of inter- and intra-group advantages) is meaningful and correctly derived.

- **Strong theoretical grounding**: Theorems 1 and 2 provide formal guarantees—equal-sized splits maximize the advantage signal amplification, and CANON selectively amplifies only the metric used for grouping. The analysis distinguishing CANON from naive advantage scaling (Table 4) shows the contribution comes from selective amplification, not numerical scaling.

- **Comprehensive empirical evaluation**: Experiments span three models (Qwen2.5-Math-7B/1.5B, Llama3.1-8B), six math benchmarks, and three difficulty levels of logic reasoning. The consistent improvements—1.9 points on math tasks with CANON-Inter and 5.2 points on complex logic with CANON-Intra—are substantial.

- **Clear practical efficiency benefits**: CANON-Eff achieves 2.63× higher performance at low token budgets and reduces token consumption by 45.5% at the same performance level, establishing a superior Pareto frontier for the performance-efficiency trade-off (Figure 4).

- **Good reproducibility practices**: Code is provided, training details (learning rates, batch sizes, temperatures) are specified, and hyperparameter sensitivity is analyzed in Tables 10-11 showing how performance varies with μ and α.

## Weaknesses

- **No statistical significance reporting**: All results report single training runs. RL training has high variance; without confidence intervals across multiple seeds, readers cannot assess whether improvements are reproducible or noise.

- **Scheduling strategies are empirical and somewhat ad-hoc**: The dynamic scheduling of μ uses multiple heuristics (accuracy-based, cosine annealing) and the paper selects the best-performing strategy per model (Section 5.2, Appendix C.6). This introduces a form of hyperparameter tuning that partially contradicts the motivation of avoiding handcrafted priors, and lacks principled guidance for practitioners.

- **Theoretical analysis assumes binary rewards**: The proofs in Appendix E use $a_+$ and $a_-$ to denote probabilities of correct responses, assuming binary verifiable rewards. Real RLVR settings may involve partial credit rewards, and this limitation is not discussed.

- **Computational overhead unanalyzed**: CANON requires computing two advantage terms and regrouping responses per batch. While token efficiency is measured, the wall-clock or FLOPs overhead compared to DR.GRPO is not reported.

- **Asymmetric benefits across task types lack mechanistic explanation**: CANON-Inter excels at math while CANON-Intra excels at complex logic, but the explanation relies on post-hoc analysis of "reflection gains" without deeper investigation into why entropy dynamics differ fundamentally between these task categories.

## Nice-to-Haves

- **Ablation on group split ratio**: Theorem 1 justifies equal-sized groups theoretically, but empirical validation testing unequal splits (e.g., 25/75, 40/60) would strengthen confidence in this design choice.

- **Comparison with entropy-advantage baselines on logic tasks**: Table 1 includes Entropy Adv and Clip-Cov comparisons for math, but not for high-complexity logic tasks. Since CANON-Intra claims superiority on complex reasoning, comparing against entropy-based baselines there would isolate the contribution more clearly.

## Novel Insights

The key insight is that metric direction in RLVR should be learned dynamically rather than prescribed. The paper reveals an elegant decomposition: inter-group advantage identifies which metric trend (e.g., higher or lower entropy) correlates with accuracy, while intra-group advantage rewards correct answers from the disadvantaged group (encouraging exploration). The observation that CANON-Inter favors exploitation (beneficial for math where models have higher baseline capability) while CANON-Intra encourages exploration (beneficial for complex logic where models need to discover new patterns) provides a conceptual framework for understanding when different advantage components should dominate. The theoretical result that equal-sized groups maximize signal amplification is non-obvious and provides practical guidance.

## Potentially Missed Related Work

None identified. The related work section covers relevant advantage estimation methods (GAE, ReMax, RLOO, GRPO, REINFORCE++) and RLVR approaches with reward/advantage shaping. The citations to concurrent work on entropy in RL for reasoning (Cheng et al., Cui et al., Gandhi et al.) are appropriate.

## Suggestions

- Report mean ± standard deviation across at least 3 random seeds for main results (Table 1, Table 2) to establish statistical significance of claimed improvements.

- Provide default recommendations for μ and α values based on task properties (e.g., complexity level, model capability) rather than requiring per-model tuning, or offer a principled automatic scheduling method.

- Add a brief discussion of computational overhead (e.g., wall-clock time per training step for CANON vs. DR.GRPO) to help practitioners assess practical trade-offs.

- Clarify the binary reward assumption in theoretical analysis and discuss whether/how results generalize to partial credit reward settings.

- Include qualitative examples comparing reasoning outputs from CANON vs. baselines to illustrate what specific reasoning behaviors improve (e.g., more efficient chains, better reflection patterns).

---

## oiz0QHejVj

- GT: Reject (avg 4.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

CLIP-Map proposes a mapping-based compression framework for CLIP models that replaces traditional select-based pruning with learnable transformation matrices. The method uses Kronecker factorization to efficiently map large model parameters to smaller dimensions, introduces diagonal inheritance initialization to address optimization challenges, and combines this with knowledge distillation in a retraining stage. Experiments demonstrate consistent improvements over TinyCLIP baselines across multiple compression ratios and benchmarks, with particularly strong results under extreme compression.

## Strengths

- **Novel paradigm for CLIP compression**: The paper introduces mapping-based compression as an alternative to select-based pruning, avoiding hard parameter removal. This conceptual shift is well-motivated, and the adaptation from model growth (LiGO, LeTs) to compression addresses real challenges (Section 1).

- **Strong empirical performance under extreme compression**: Results show substantial gains at 1% compression ratio (TR@1 improves from 12.5% to 15.8% on MSCOCO vs TinyCLIP with progressive training). Performance gains are consistent across 21 downstream classification datasets, demonstrating robust generalization.

- **Technical contribution in initialization**: The diagonal inheritance initialization (Section 3.2.3) addresses a real optimization challenge with theoretical justification (Equations 5-10). Table 5 demonstrates dramatic improvement over random/Xavier/Kaiming initialization (28.9% vs <5% ImageNet accuracy).

- **Efficiency improvements**: Table 3 shows CLIP-Map achieves comparable performance with fewer seen samples (0.45B vs 0.75B for TinyCLIP at 10% compression). Table 11 demonstrates significant training speedup (21h50m vs 49h33m for tiny models).

## Weaknesses

- **Depth compression justification insufficient**: Equation 2 defines depth compression as a linear combination of layer weights: $W^{new}_{l'} = \sum_{l=1}^{L_1} L_{depth}[l', l] \cdot W_l$. Different transformer layers learn different representations, but the paper provides no justification for why linear combination is valid or analysis of when this assumption fails. A depth-only ablation would strengthen this claim.

- **Catastrophic failure with non-diagonal initialization unanalyzed**: Table 5 shows random initialization achieves only 0.1% vs 28.9% with diagonal init. This dramatic gap raises a critical question: is the method learning meaningful mappings, or does diagonal initialization simply provide weight inheritance with minimal contribution from learned off-diagonal elements? An ablation with frozen diagonal matrices would isolate the learnable mapping's contribution.

- **Missing width vs depth ablation**: The method combines both width compression (Kronecker factors) and depth compression (linear layer combination). Without separating these contributions, we cannot verify whether both are necessary or whether one dominates performance gains.

- **Limited baseline comparison breadth**: Primary comparison is against TinyCLIP. Other methods (UPop, EfficientVLM, DynaCLIP) appear only in Table 7 without reproduction under identical settings. The appendix notes "due to time constraints, we don't adapt these methods to our experimental setting."

- **Information preservation claim not directly measured**: The paper claims mapping "preserves more information" than selection, but provides no quantitative measure of information preservation (e.g., embedding similarity between compressed and original models).

- **Architecture comparison fairness unclear**: Table 6 shows CLIP-Maptiny uses different architecture (512 width, 12/6 depth) than TinyCLIPtiny (128 width, 4/2 depth). The relationship between architecture choices and mapping constraints should be clarified.

## Nice-to-Haves

- Statistical significance reporting (error bars across multiple runs) would strengthen reproducibility claims.

- A mechanistic analysis of learned mapping matrices (beyond visualization in Figure 5) would help explain why the method works.

## Novel Insights

The diagonal inheritance initialization reveals an important optimization insight for mapping-based compression: when compressing a model via learned transformations, the optimization landscape is highly sensitive to initialization. The variance analysis (Equations 5-8) shows that independently initialized Kronecker factors lead to multiplicative variance scaling that destabilizes training. By initializing diagonal elements to 1, the method effectively starts from an approximate identity transformation, inheriting subset selection as a special case while allowing learned deviations. This bridges select-based and mapping-based paradigms—the starting point is select-based (taking a subset of weights), but the optimization can learn arbitrary linear combinations.

## Potentially Missed Related Work

None identified in this review cycle.

## Suggestions

- Add an ablation with diagonal initialization but frozen off-diagonal elements to isolate the contribution of learnable mapping beyond simple weight inheritance.

- Add quantitative analysis of information preservation (e.g., embedding cosine similarity between compressed and original models) to directly validate the core claim.

- Include width-only and depth-only compression variants to understand individual contributions of each component.

---

## lqjQs2lVNm

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
HyperPrune introduces a resource-efficient optimization framework for learning n:m semi-structured sparse masks in LLMs using a shared, context-aware hypernetwork conditioned on learnable layer and component embeddings. The method incorporates continual pruning regularization to preserve cross-layer knowledge and feature outlier regularization to retain critical activations, enabling pruning of LLaMA-2 models from 7B to 70B on a single A100 GPU while achieving competitive accuracy-sparsity trade-offs.

## Strengths
- **Memory-efficient hypernetwork design**: By generating masks block-wise (m=4 for 2:4 sparsity) through a shared hypernetwork conditioned on layer/component embeddings, the method achieves dramatic memory savings—15-22 GB for 7B/13B models vs. 339-630 GB for MaskLLM (Table 7), enabling 70B pruning on a single GPU where competing optimization methods cannot run.
- **Strong empirical results with practical efficiency**: HyperPrune achieves competitive or superior accuracy (53.76 mean accuracy on LLaMA-2-7B vs. 52.81 for MaskPro) while requiring only 7-15 GPU hours compared to thousands for MaskLLM (Table 8). The consistent improvements across model scales (7B, 13B, 70B) demonstrate scalability.
- **Well-motivated regularization components**: The continual pruning regularization (Eq. 9) draws a meaningful connection to continual learning for cross-layer knowledge preservation, while feature outlier regularization (Eq. 10-11) recovers Wanda's importance score as a special case with variance-aware extensions, providing principled grounding.
- **Comprehensive ablation studies**: Figures 4-5 and Table 9 demonstrate the contribution of each component, showing that removing layer embeddings increases perplexity from 10.11 to 11.36, and prior-based initialization provides meaningful improvements (10.11 vs. 10.75 without prior).

## Weaknesses
- **No error bars or statistical significance**: All results in Table 1 and throughout the paper report single values without variance across multiple runs. Given the stochastic nature of Gumbel-Softmax sampling and potential initialization variance, this omission significantly undermines confidence in the reported improvements (e.g., 53.76 vs. 52.81 mean accuracy on 7B).
- **Theorem 1 overstates the theoretical contribution**: The theorem claims equivalence ("⇔") between mutual information maximization and MSE minimization, but the proof relies on strong assumptions—linear activation, Gaussian inputs, and small perturbations—that do not hold for Transformers with SwiGLU nonlinearities and aggressive 50% sparsity. While Remark 3 acknowledges these limitations, the theorem statement should use "approximately equivalent" rather than equivalence.
- **Incomplete baseline comparison**: MaskLLM, the direct optimization-based competitor for n:m sparsity, is excluded due to computational cost without citing its published results. Readers cannot assess how HyperPrune's 10.11 Wikitext perplexity compares to MaskLLM's reported metrics on equivalent settings, weakening claims of state-of-the-art performance against optimization-based methods.
- **Limited evaluation scope**: Only 2:4 sparsity pattern is evaluated despite claiming support for arbitrary n:m. Experiments are restricted to LLaMA-2/3 architectures without testing on OPT, Pythia, Mistral, or other model families. No post-pruning fine-tuning experiments are included, leaving practical deployment utility unclear.

## Nice-to-Haves
- Analysis of learned mask patterns: What structures does the hypernetwork discover? Do patterns correlate with magnitude-based scores or show layer-specific characteristics?
- Per-layer accuracy breakdown to verify whether sequential optimization causes error accumulation in deeper layers.
- Hyperparameter sensitivity analysis showing performance variation across the reported λ₁, λ₂ ranges spanning 6 orders of magnitude.

## Novel Insights
HyperPrune's key insight is that n:m structured sparsity fundamentally constrains the mask space (e.g., only 6 valid patterns for 2:4), enabling a compact hypernetwork operating on m-sized blocks rather than full weight matrices. This architectural choice—sharing parameters across all layers while differentiating through context embeddings—transforms an intractable global optimization into a scalable layer-wise procedure. The information-theoretic framing, though approximate, provides the first principled justification for reconstruction-based objectives in structured sparsity, connecting mask learning to mutual information preservation between dense and pruned models.

## Potentially Missed Related Work
None identified in this review cycle.

## Suggestions
- Include MaskLLM's published results from its original paper (e.g., Wikitext perplexity on LLaMA-2-7B at 2:4 sparsity) to contextualize performance, even if reproduction is infeasible.
- Add experiments with at least one additional n:m pattern (e.g., 4:8) and one non-LLaMA architecture to strengthen generalizability claims.
- Report mean and standard deviation across at least 3 random seeds for all quantitative results to establish statistical significance of improvements.
- Consider adding post-pruning fine-tuning experiments (e.g., LoRA) to demonstrate practical deployment utility.

---

## sLkis6UGKk

- GT: Reject (avg 5.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

DeFacto introduces a counterfactual reasoning framework for multimodal language models that enforces both answer correctness and visual grounding faithfulness through a three-way training paradigm (positive, counterfactual, and random-masking) combined with GRPO-based reinforcement learning. The key innovation is teaching models to abstain when evidence is masked, preventing "spurious correctness" where models arrive at correct answers through wrong reasoning paths.

## Strengths

- **Addresses an important and understudied problem**: The paper identifies "spurious correctness"—where models produce correct answers with incorrect reasoning—as a critical issue for trustworthy multimodal AI, formalizing this failure mode and proposing a principled solution.
- **Well-designed three-way training paradigm**: The combination of positive supervision, counterfactual abstention, and random-masking elegantly prevents shortcut learning. The random-masking condition specifically addresses the concern that models might learn to associate mask presence with abstention behavior.
- **Automatic and scalable dataset construction**: The pipeline using MLLM descriptor extraction, open-vocabulary detection (DINO-X), OCR, and RPN enables construction of counterfactual data at scale without manual annotation, making the approach practical and reproducible.
- **Comprehensive empirical evaluation**: The paper evaluates on 15+ benchmarks spanning general VQA, document understanding, and scene text tasks, demonstrating consistent improvements over strong baselines (Tables 1-2, 6).
- **Novel faithfulness evaluation protocol**: The 1k human-annotated validation set with IoU-based grounding metrics (mAP, AP50, AP75) provides direct measurement of reasoning grounding quality—Table 3 shows DeFacto achieves 30.6 mAP compared to near-zero for baselines.
- **Clear ablation studies**: Table 4 systematically isolates contributions from counterfactual data (+5.3% on VQAv2) and GRPO rewards (+9.3% additional improvement), demonstrating each component's value.

## Weaknesses

- **Evidence localization threshold not specified**: The threshold τ for partitioning regions into evidence vs. irrelevant (Eq. 3) is not provided in the paper or appendix, affecting reproducibility of the dataset construction pipeline.
- **No statistical significance reporting**: All results are single-point estimates without standard deviations or confidence intervals. For smaller improvements (e.g., +1.4% on DocVQA, +1.8% on InfoVQA), uncertainty quantification is essential to assess meaningfulness.
- **Unfair backbone comparison with ViCrop**: Table 1 compares DeFacto (Qwen2.5-VL-7B) against ViCrop (LLaVA-1.5 Vicuna-7B)—different base models with different capabilities. The improvements cannot be attributed solely to the proposed method.
- **Near-zero grounding metrics for baselines need explanation**: Table 3 shows GRIT achieving 73.7% accuracy but 0.0 mAP, and DeepEyes with 2.2 AP50 but 44.0% accuracy. While the paper attributes this to "shortcut behaviors," such extreme grounding failures warrant deeper investigation—whether this reflects true limitations or evaluation protocol mismatches.
- **No quality analysis of automatic counterfactual construction**: The pipeline relies on external detectors (Qwen2.5-VL, DINO-X, RPN, OCR) for evidence localization, but provides no quantification of annotation quality or error rates in the constructed dataset.
- **Missing ablation for random-masking component**: The ablation study (Table 4) bundles random-masking with counterfactual alignment. Without isolating this component, the claim that it prevents superficial correlations remains unvalidated.
- **Single backbone evaluation**: All experiments use Qwen2.5-VL-7B. Generalization to other architectures (LLaVA, InternVL) is not demonstrated, limiting claims about method generality.

## Nice-to-Haves

- Sensitivity analysis for the 9+ reward hyperparameters (γ_unk, ρ_unk, γ_guess, γ_corr, β_pos, β_neg, γ_∅, α, λ₁, λ₂) to provide guidance for practitioners.
- Analysis of abstention rates on validation data to understand when and why the model chooses to abstain vs. answer.
- Evaluation of abstention behavior on genuinely ambiguous real-world queries to assess whether the model has learned meaningful uncertainty quantification.

## Novel Insights

The three-way training paradigm represents a clever insight: simply masking evidence regions could teach models to associate masks with abstention (a spurious correlation). The random-masking condition breaks this by masking irrelevant regions, ensuring the model learns to attend to actual evidence rather than superficial mask patterns. This addresses a subtle but critical failure mode that standard counterfactual training would miss. Additionally, the distinction between "Mislocalized Failure" (wrong region → wrong answer) and "Spurious Correctness" (wrong region → correct answer) is an important conceptual contribution—the latter is more insidious because it masks reasoning deficiencies.

## Potentially Missed Related Work

- None identified. The related work section adequately covers both "thinking with images" approaches (GRIT, DeepEyes, Chain-of-Focus, etc.) and counterfactual reasoning in VLMs.

## Suggestions

1. Specify the threshold τ value for evidence localization in the paper or appendix to enable reproducibility.
2. Report results with standard deviations across multiple random seeds to establish statistical significance, particularly for improvements under 5%.
3. Add a comparison with the same backbone (Qwen2.5-VL-7B baseline without counterfactual training) as the primary baseline, treating ViCrop as a secondary reference due to backbone differences.
4. Include an analysis of automatic annotation quality by sampling and manually verifying a subset of counterfactual examples.
5. Add an ablation isolating the random-masking component to validate its contribution to preventing shortcut learning.

---

## jKeOsMdMe5

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
Fisale proposes a data-driven framework for two-way fluid-solid interaction (FSI) that introduces three key innovations: (1) explicitly modeling the coupling interface as a separate component alongside fluid and solid domains, (2) multiscale latent ALE grids that provide unified, geometry-aware embeddings across heterogeneous domains, and (3) a Partitioned Coupling Module (PCM) that decomposes the coupled dynamics into sequential substeps inspired by classical numerical methods. The method is evaluated on three challenging FSI scenarios covering 2D/3D settings and different task types (single-step prediction, autoregressive simulation, steady-state inference).

## Strengths
- **Physically-motivated architecture design**: The framework draws clear inspiration from established numerical techniques (ALE method, partitioned coupling algorithms), with the PCM mirroring how classical FSI solvers decompose coupled problems. This provides strong physical justification for architectural choices.
- **Consistent empirical improvements**: Fisale achieves state-of-the-art performance across all three tasks and physical quantities. On Structure Oscillation, it achieves 0.0066 mean relative L2 error vs. 0.0089-0.0104 for baselines; similar gains appear on Venous Valve and Flexible Wing tasks (Tables 2, 4, 5).
- **Explicit interface modeling validated by ablation**: Removing the interface component causes 28.68% performance degradation (Table 15), demonstrating the importance of treating the coupling interface as a separate entity rather than folding it into fluid or solid domains.
- **Comprehensive baseline comparison**: Over 11 baselines across neural operators (GeoFNO, GINO, CoDA-NO, LSM, LNO), transformers (Galerkin, GNOT, ONO, Transolver), and GNNs (MGN, HOOD, AMG) with matched parameter counts (Table 10).
- **Memory-efficient despite attention-based architecture**: Fisale uses only 3.10 GiB GPU memory on the Flexible Wing task compared to 11-23 GiB for other attention-based models, due to the partitioned design that splits large matrices into smaller submatrices (Table 14).

## Weaknesses
- **Computational overhead for accuracy**: While memory-efficient, Fisale requires 296.30s per epoch on Flexible Wing compared to 50-72s for several baselines (Table 14). The paper claims efficiency but should more explicitly acknowledge this trade-off.
- **Custom datasets limit reproducibility**: Two of three datasets (Venous Valve, Flexible Wing) are newly generated using COMSOL. While parameters are detailed (Tables 6-8), independent verification requires proprietary software access.
- **No conservation property verification**: FSI problems fundamentally require mass and momentum conservation. The paper reports prediction errors but does not verify whether predictions satisfy these physical constraints, which is critical for trust in scientific applications.
- **Limited geometry generalization evidence**: All benchmarks use fixed structural geometries with varying parameters (Reynolds number, material properties). The claim of "geometry-aware embeddings" is not tested on unseen shapes (e.g., different wing profiles or valve morphologies).
- **Missing monolithic baseline comparison**: The ablations test interface modeling and PCM ordering, but do not compare against a monolithic attention architecture with the same capacity. This makes it difficult to isolate the benefit of the partitioned coupling strategy versus simply having more parameters.
- **No classical numerical solver comparison**: The paper compares only against learning-based methods. Without comparison to established FSI solvers (ALE FEM, immersed boundary methods), the practical utility for practitioners remains unclear.

## Nice-to-Haves
- **Hyperparameter selection guidance**: The ablation on pathway number H (Table 19) shows degradation from H=4 to H=5, but provides limited guidance on how to choose H, grid sizes, or channel dimensions for new problems.
- **Latent ALE grid visualization**: Visualizing how the learned ALE grids deform during prediction would illuminate whether they capture physically meaningful motion patterns.
- **Failure mode analysis**: The paper demonstrates success cases but does not discuss scenarios where Fisale might fail (extreme deformation rates, topological changes, near-contact situations).

## Novel Insights
The attention pattern analysis in Appendix I reveals complementary behaviors across resolution scales that illuminate why the multiscale design is effective. On low-resolution grids, attention focuses primarily on self-relations within each component (solid-to-solid, fluid-to-fluid), while high-resolution grids emphasize cross-component interactions. This division of labor—coarse grids capturing global structural patterns and fine grids capturing local interface dynamics—provides a compelling mechanistic explanation for the performance gains. The finding that interface self-attention shows reversed patterns (both resolutions emphasizing cross-component aggregation) further validates the explicit interface modeling choice, suggesting the interface inherently requires integrating information from both domains rather than maintaining independent dynamics.

## Potentially Missed Related Work
None identified. Related work search was not performed, and the paper provides comprehensive coverage of neural operators, GNN-based simulators, and classical FSI literature.

## Suggestions
- **Add conservation verification**: Report mass conservation error and velocity divergence for predictions to verify physical consistency, particularly for the Venous Valve autoregressive simulation.
- **Test geometry generalization**: Include experiments with varying structural shapes within each benchmark to validate the geometry-aware embedding claim.
- **Compare inference speed with classical solvers**: Include timing comparison with traditional ALE or immersed boundary methods to establish practical utility for practitioners.
- **Add monolithic baseline**: Implement a non-partitioned attention baseline with matched capacity to isolate the benefit of PCM architecture from the explicit interface modeling.

---

