# ICLR Benchmark Results

Date: 2026-04-03 22:57
Critic/Merger: z-ai/glm-5 (OpenRouter)
Neutral: z-ai/glm-5, Related Work: z-ai/glm-5:online (OpenRouter)

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Final Review

## Summary
The paper identifies three misconceptions in the emerging online map-based motion prediction protocol and proposes OMMP-Bench with corrections: (1) a new spatially-disjoint data partition to eliminate train-validation distribution gap, (2) refined metrics evaluating all moving non-ego agents instead of only ego vehicle, and (3) a boundary-free baseline using deformable attention on image features to supplement distant agents beyond the online map perception range.

## Strengths
- **Clear identification of train-val gap**: The paper correctly identifies that training the online mapping model on data later used for motion prediction training creates an unrealistic scenario where maps are highly accurate during training but degraded during validation. Table 1 empirically demonstrates this gap (default split: 0.6839 minADE vs. proposed split: 0.6308 minADE with MapTRv2-CL+HiVT).
- **Systematic analysis of evaluation flaws**: The critique that existing protocols evaluate only ego vehicle while motion prediction's purpose is collision avoidance with other agents is well-motivated. Table 6 shows static agents have near-perfect prediction (0.009 minADE), confirming the need to exclude trivial cases.
- **Empirical validation of perception range mismatch**: Table 2 shows MapTR performance drops from 0.124 mAP at 30×60m to 0.014 at 100×100m, demonstrating that current online mapping models cannot scale to motion prediction's required range. Table 6 confirms far agents have significantly worse performance than close agents.
- **Practical baseline solution**: The image feature integration via deformable attention provides consistent improvements across model combinations (e.g., MapTRv2-CL+HiVT improves far-agent minADE from 0.6999 to 0.6274).

## Weaknesses
- **No ground truth map comparison**: The paper lacks experiments with ground truth HD maps to establish the performance upper bound. Without this, readers cannot quantify the true cost of using imperfect online maps or assess how much room for improvement remains.
- **No verification of eliminated train-val gap**: The paper claims the new split eliminates train-val map quality gap but provides no map mAP numbers across splits to substantiate this. Showing map prediction accuracy on motion-train vs. motion-validation sets would directly verify the core claim.
- **Reduced training data without impact analysis**: The new split uses only 367 scenes for map training and 397 for motion training, significantly less than nuScenes' original ~700 training scenes. The paper does not analyze whether this reduction affects model convergence or final performance.
- **Missing statistical significance reporting**: None of the tables include standard deviations or confidence intervals. Given the relatively small number of validation scenes (86), variance across different random seeds or scene samples is important to report.
- **No agent distribution statistics**: The paper does not report how many agents fall into "Moving-Non-Ego-Close" vs. "Moving-Non-Ego-Far" categories. If "far" agents are a small minority, the reported improvements may not translate to practical significance.

## Nice-to-Haves
- **Computational overhead analysis**: The deformable attention mechanism for image features introduces additional computation. For real-time autonomous driving applications, inference time and memory requirements should be quantified.
- **Qualitative trajectory prediction examples**: Visualizations showing actual predicted trajectories with online maps vs. ground truth maps would help readers understand what specific errors the image feature baseline addresses.

## Novel Insights
The finding that existing methods (uncertainty prediction, BEV features) improve ego-vehicle prediction but sometimes degrade close non-ego agent prediction (Table 7) is an important negative result. This reveals that optimizing for the wrong evaluation metric can lead to misleading progress. The distinction between close and far agents, combined with the image feature solution for distant agents, provides a practical direction for future research addressing perception range limitations in online map-based prediction systems.

## Potentially Missed Related Work
- None identified (search was skipped)

## Suggestions
- Add experiments with ground truth HD maps to establish performance upper bounds and quantify the degradation from imperfect online maps.
- Report map mAP scores on motion-train and motion-validation sets to verify the claimed elimination of the train-val distribution gap.
- Report agent distribution statistics (number of scenes, agents per category) and include standard deviations across multiple runs.
- Include computational overhead analysis for the image feature baseline to demonstrate practical feasibility for autonomous driving deployment.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether commonly used sparse autoencoder (SAE) quality metrics and auto-interpretability pipelines can distinguish trained transformers from randomly initialized ones. Across Pythia models (70M to 6.9B parameters) and multiple randomization schemes, the authors find that auto-interpretability scores and reconstruction metrics are surprisingly similar for trained and randomized models, with a true Gaussian control at chance level. The paper proposes token distribution entropy as a complementary metric and shows it reveals qualitative differences in feature "abstractness" between trained and random models.

## Strengths
- **Important empirical finding with practical implications**: The core result that aggregate SAE metrics fail to clearly distinguish trained from random transformers is surprising and consequential for mechanistic interpretability research, challenging assumptions underlying many SAE evaluation practices.
- **Well-designed experimental controls**: The paper uses multiple randomization variants (Step-0, re-randomized with/without embeddings, Gaussian control), providing a gradient of comparisons that strengthens the evidence. The Gaussian control showing near-chance performance validates the experimental setup.
- **Comprehensive evaluation across model scales**: Experiments span 70M to 6.9B parameters with consistent findings, demonstrating the robustness of results across model sizes.
- **Novel metric proposal**: Token distribution entropy (Figure 2, bottom row; Figure 20) successfully distinguishes trained from random models—trained models show increasing entropy across layers while randomized models maintain low entropy, suggesting more abstract features in trained models at later layers.
- **Mechanistic hypotheses provided**: The toy model experiments (Section 4) offer plausible hypotheses for why random networks produce interpretable SAE features—random networks may preserve or amplify superposition present in input data.

## Weaknesses
- **Lack of uncertainty quantification on main results**: Figure 2 presents results without error bars or confidence intervals. While Appendix E shows uncertainty for Pythia-70M with 5 random seeds, the main results lack statistical significance testing, making it unclear whether observed differences (or lack thereof) are meaningful.
- **Only TopK SAE architecture tested**: Given the proliferation of SAE architectures (ReLU, JumpReLU, Gated), testing only TopK SAEs limits the generality of conclusions about SAE evaluation metrics broadly.
- **Entropy metric lacks validation for computational relevance**: While token distribution entropy differentiates trained from random models, the paper does not demonstrate that high-entropy features are more useful for downstream tasks (steering, causal intervention) or more "computationally relevant" as claimed.
- **Limited model family diversity**: All experiments use Pythia models from a single training run family; findings may not generalize to other architectures or training procedures.
- **Reconstruction metrics show partial separation but analysis is incomplete**: Cosine similarity and explained variance (Figure 2, rows 3-4) show some differences between trained and randomized variants, yet the paper's conclusion emphasizes the similarity across all metrics without systematically analyzing which metrics distinguish better and why.

## Nice-to-Haves
- **Causal intervention experiments** to test whether SAE features from trained vs. random transformers have different functional effects (e.g., steering success). This would directly validate claims about "computational relevance."
- **Testing additional SAE architectures** (JumpReLU, Gated SAEs) to establish whether the finding generalizes beyond TopK SAEs.
- **Analysis of scaling trends**: The paper observes smaller gaps between trained and random models at larger scales but does not investigate the underlying mechanism—whether this reflects metric saturation, genuine convergence, or artifacts.

## Novel Insights
The paper provides a critical empirical finding that aggregate auto-interpretability scores are insufficient to guarantee that learned features are computationally relevant. The token distribution entropy analysis is a novel and valuable contribution: trained models show increasing entropy across layers (features becoming less token-specific and more abstract), while randomized models maintain consistently low entropy (features remain tied to specific tokens). This suggests that current SAE evaluation metrics may be capturing statistical properties of the data and architecture rather than evidence of learned computation. The toy model analysis showing that random MLPs can preserve or amplify superposition provides a plausible mechanistic explanation for why random networks yield interpretable SAE features.

## Potentially Missed Related Work
- None identified in the provided review.

## Suggestions
- Add error bars and statistical significance tests to Figure 2 to clarify whether differences between conditions are meaningful.
- Validate token distribution entropy as a measure of computational relevance by testing whether high-entropy features from trained models can be used for downstream tasks (steering, concept erasure) while low-entropy features cannot.
- Test at least one additional SAE architecture (e.g., JumpReLU or Gated SAEs) to strengthen claims about SAE evaluation metrics broadly.
- Acknowledge and analyze more explicitly that some reconstruction metrics do show partial separation between trained and randomized models, providing a more nuanced picture of which evaluation approaches work better.

---

## KsWRLyIAKP

- GT: Withdrawn (treated as Reject) (avg 3.2)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a novel formulation of lead-lag detection in financial markets as a temporal link prediction task on dynamic graphs, where assets are nodes and directed edges capture predictive relationships over time. The authors create a benchmark dataset of 37 financial assets with five years of daily data enriched with financial indicators and sentiment features, then adapt and compare eight deep learning models including LSTMs and state-of-the-art temporal GNNs (JODIE, DySAT, TGAT, TGN, APAN, GraphMixer), finding that the simpler GraphMixer architecture outperforms more sophisticated attention-based models.

## Strengths
- **Novel problem formulation**: Reformulating lead-lag detection as temporal link prediction on dynamic graphs is innovative and enables modeling multi-asset interdependencies simultaneously rather than through pairwise analysis
- **Comprehensive empirical evaluation**: The paper evaluates eight models across multiple metrics (AP, AAUC, R@k, MRR) with statistical significance testing (Friedman + Conover's tests), providing a thorough comparison
- **Dual scenario evaluation**: Explicitly considering both positive-only and positive-and-negative lead-lag relationships addresses ambiguity in existing literature and offers practical insights for different trading strategies
- **Benchmark contribution**: The custom dataset comprising five years of daily pricing data, financial indicators, and sentiment features for 37 entities is a valuable community resource
- **Well-structured ablation**: Systematic evaluation of feature types (embeddings, prices, financial indicators, sentiment) reveals that description embeddings alone achieve best performance for most models

## Weaknesses
- **No simple or statistical baselines**: The paper lacks a persistence baseline (predicting yesterday's edges for today) or adapted statistical methods (e.g., rolling cross-correlation). Without these, it's unclear whether the models are learning non-trivial patterns or the task formulation itself is inherently tractable. The authors acknowledge inability to compare with traditional methods, but this gap limits practical positioning
- **Threshold sensitivity unexplored**: The ground truth labels depend entirely on the ε=5% threshold for defining lead-lag relationships. The paper states this "maintains a balanced approach" but provides no systematic analysis of how results change across different thresholds (e.g., 3%, 7%, 10%). This is critical for assessing robustness
- **Small graph scale limits claims**: With only 37 nodes, the graph is relatively small. This limits conclusions about TGNN effectiveness for larger financial networks and raises questions about whether scalability benefits of simpler models like GraphMizer would hold in more complex settings
- **Unexplained model ranking**: GraphMixer (an MLP-based architecture) consistently outperforms sophisticated attention-based models (TGN, TGAT), but the paper offers no analysis of why. Understanding whether this is due to data sparsity, overfitting in complex models, or favorable inductive bias would provide meaningful methodological insights
- **Ablation findings underinterpreted**: The finding that prices and financial indicators don't help—and sometimes hurt—performance raises fundamental questions. If temporal price features don't improve predictions, what predictive signal are the models actually capturing? Static embeddings cannot encode temporal dynamics alone, suggesting the task may rely heavily on graph topology rather than financial signals

## Nice-to-Haves
- Cross-temporal regime validation (e.g., performance during COVID volatility vs. normal periods) to assess robustness during market stress
- Held-out asset generalization test to evaluate whether learned patterns transfer to unseen assets
- Economic interpretation of top predicted relationships (e.g., "Lucid Group → SunRun") to validate that discovered patterns correspond to plausible market mechanisms

## Novel Insights
The counterintuitive finding that GraphMixer outperforms attention-based temporal GNNs suggests that simpler architectures may be better suited for sparse, dynamic financial graphs where overfitting is a risk. Equally notable is the ablation result showing that static description embeddings outperform temporal financial features for most models—a paradox that hints the predictive signal may stem from structural patterns in the evolving graph topology rather than from the financial features themselves. This observation merits deeper investigation and could inform future work on feature engineering for temporal financial graphs.

## Potentially Missed Related Work
None identified in this review cycle.

## Suggestions
- Add at least one simple baseline (persistence: predict prior day's edges) and one adapted statistical method (e.g., threshold-based cross-correlation as link prediction) to contextualize TGNN performance
- Conduct and report threshold sensitivity analysis across multiple ε values to demonstrate robustness of conclusions
- Provide deeper analysis of why GraphMixer succeeds, potentially through probing experiments or examination of the learned representations
- Consider reporting performance across distinct market regimes (e.g., pre-COVID, COVID volatility, post-COVID) to assess practical robustness

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

ELMUR introduces a transformer architecture with layer-local external memory for long-horizon decision making under partial observability. Each layer maintains persistent memory embeddings that interact with tokens via bidirectional cross-attention (mem2tok for reading, tok2mem for writing), updated through a Least Recently Used (LRU) mechanism with convex blending. The method achieves 100% success on T-Maze corridors up to one million steps (100,000× beyond the attention window), best performance on 21 of 23 MIKASA-Robo tasks with visual observations, and top scores on 24 of 48 POPGym environments.

## Strengths

- **Strong empirical results across diverse benchmarks**: The paper demonstrates consistent improvements on T-Maze (100% success up to 1M steps with L=10 context), MIKASA-Robo robotic manipulation tasks (~70% aggregate improvement over prior best), and POPGym (best on 24/48 tasks). The T-Maze extrapolation result is particularly compelling, showing retention far beyond the training horizon.

- **Theoretical grounding with formal analysis**: Propositions 1-2 provide formal bounds on exponential forgetting, half-life, and memory embedding boundedness. The derivation shows H₀.₅ ≈ M·L·ln(2)/λ, giving practitioners a principled way to reason about memory retention.

- **Comprehensive ablation study with mechanistic evidence**: The paper ablates memory size M, blending factor λ, initialization σ, segment configurations, and individual components (LRU, relative bias, per-layer memory). The memory probing analysis (Appendix A.9) demonstrates that task-relevant information is linearly decodable from memory embeddings, with confusion matrices showing near-perfect color classification from memory vectors—evidence that the mechanism is actually used rather than relying on parameter capacity alone.

- **Clear algorithmic presentation**: Algorithm 1 and 2 provide complete pseudocode, and the relative bias formulation (Equations 6-7) precisely specifies temporal grounding. The reproducibility statement includes code, hyperparameters, and detailed training configurations.

## Weaknesses

- **Missing comparisons with established memory-augmented architectures**: The paper does not compare against Transformer-XL (the canonical segment-level recurrence approach), Compressive Transformer, or Memorizing Transformer—all foundational works on extending transformer context. While RATE and DMamba are included, the absence of these baselines leaves the positioning incomplete relative to the broader memory-augmented transformer literature.

- **Hyperparameter sensitivity without guidance**: Table 7 shows λ ranging from 0.05 (T-Maze) to 0.90 (CartPole) across tasks, and Figure 6 reveals sharp performance drops when M < N or λ ≈ 0.5. The paper provides no principled method for selecting these for new environments, creating practical deployment concerns.

- **Limited computational efficiency analysis**: The paper reports inference time per step (6.8ms for ELMUR vs. 7.2ms for RATE, 10.7ms for DT) but omits training time, GPU memory footprint, and FLOPs comparisons. For a method claiming efficiency through bounded memory, these metrics are essential.

- **Theoretical-empirical gap under-discussed**: The half-life bound predicts retention of ~2,770 steps for the reported hyperparameters, yet empirical results show success at 1M steps. While the paper notes that "effective horizons are often much longer," this 360× discrepancy deserves deeper explanation—is the bound pessimistic, or are other mechanisms extending retention?

- **Segment-level recurrence prevents cross-segment gradients**: Memory is detached between segments during training, preventing backpropagation through time. This design choice limits the ability to learn long-horizon credit assignment via gradients, placing the burden entirely on the LRU mechanism.

## Nice-to-Haves

- A visualization of memory read/write patterns on the long-horizon T-Maze task itself, rather than only on RememberColor3-v0, would directly validate the core claim of 100,000× retention.

- Guidance on hyperparameter selection—either adaptive λ mechanisms, rules of thumb based on task properties, or demonstration that a fixed configuration works reasonably across diverse tasks.

## Novel Insights

The layer-local memory design with explicit LRU management is a clever integration of established ideas. The key insight is that treating memory as a structured, bounded resource with explicit write policies (rather than implicit recurrence) enables both theoretical analysis and predictable scaling. The probing experiments (Figure 9) reveal a striking pattern: ELMUR performs a targeted one-shot write into a dedicated memory slot immediately after observing the cue, then maintains this representation with high stability—evidence of sparse, purposeful memory usage rather than continuous churn. The theoretical bound, while conservative, provides a useful mental model: retention scales linearly with both memory slots M and segment length L, giving practitioners clear levers for tuning.

## Potentially Missed Related Work

- Transformer-XL (Dai et al., 2019) — foundational work on segment-level recurrence with cached activations, directly relevant to the segment-level processing approach.
- Compressive Transformer (Rae et al., 2019) — extends Transformer-XL with compressed memory of past segments, conceptually related to ELMUR's external memory.
- Memorizing Transformer (Wu et al., 2022) — kNN-augmented attention with external key-value cache for long contexts.

## Suggestions

- Add baseline comparisons with Transformer-XL and Compressive Transformer to contextualize against segment-level recurrence approaches.
- Report training time and peak GPU memory consumption alongside inference latency to substantiate efficiency claims.
- Provide empirical or theoretical guidance on selecting M and λ for new environments, or propose adaptive variants that reduce tuning burden.
- Discuss why the empirical retention far exceeds the theoretical half-life bound—is the bound pessimistic, or are other factors (e.g., task structure, attention patterns) contributing?

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces OptMerge, a method for merging multimodal large language models (MLLMs) that combines SVD-based task vector denoising with optimization over task vector interactions. The authors also present the first systematic benchmark for MLLM model merging, covering five capability categories (VQA, Geometry, Chart, OCR, Grounding) and modality merging across vision, audio, and video. Experiments demonstrate that merged models can match or exceed mixture training performance at significantly lower computational cost.

## Strengths

- **First systematic MLLM merging benchmark with clear task categorization**: The paper addresses a genuine gap by providing fine-grained task divisions and publicly releasing checkpoints for both LoRA and full fine-tuning scenarios. This enables standardized comparison of merging methods on meaningful capability categories rather than treating each fine-tuning dataset as a separate task.

- **Strong empirical coverage across architectures and settings**: Experiments span multiple model families (InternVL2.5, Qwen2-VL, Vicuna-7B), fine-tuning approaches (LoRA, full fine-tuning), and evaluation settings (capability merging, modality merging, real Hugging Face checkpoints from independent developers).

- **Demonstrated computational efficiency**: Table 7 shows model merging requires only 3.78h and 21.97GB GPU memory versus 25.38h and 240GB for mixture training—a substantial practical advantage for practitioners.

- **Theoretical insight into fine-tuning intensity**: Theorem 3.1 formalizes why over-trained models merge poorly, showing that cross-task interference grows with ηT while convergence benefits diminish. This provides principled guidance for creating merge-friendly expert models.

- **Practical applicability on real-world checkpoints**: Table 6 demonstrates the method works on actual Hugging Face models from independent developers, validating practical utility beyond controlled experiments.

## Weaknesses

- **Overstated claims in abstract**: The abstract states the merged model "can even outperform... mixture training," but Table 2 shows OptMerge (57.00 avg) underperforms Mixture Training (57.66 avg) on InternVL2.5. The claimed "2.48% average performance gain" is not clearly derived from reported results—Table 4 shows +4.65% and +2.35% for different settings, while Section 5.2 reports +0.44% and +1.9% for others. These inconsistencies should be corrected.

- **Missing statistical significance**: All results report single numbers without standard deviations or repeated runs. Given inherent variability in LLM evaluations, it is unclear whether differences between methods (e.g., OptMerge 57.00 vs. TSV Merging 56.70 on InternVL2.5) are statistically meaningful.

- **No comparison with AdaMMS or UQ-Merge**: The paper identifies these as the most relevant prior work for MLLM merging but does not include them as baselines. Comparing against these methods on the same benchmark settings would enable fair assessment of relative performance.

- **Incomplete ablation for full fine-tuning**: Table 4 provides ablation only for LoRA settings and modality merging. Since full fine-tuning uses different techniques (centering + SVD versus truncated SVD without centering), ablation results for InternVL2.5 are needed to validate those design choices.

- **Limited modality merging evaluation**: The claim about "complementary nature of modal information" rests on only two datasets (MUSIC-AVQA and AVQA). Without testing on additional audio-visual benchmarks, the generalizability of modality merging claims is not well-established.

- **Unclear hyperparameter selection strategy**: The method searches λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5} and uses rank ratio k = rank/task_count without principled justification. For a "data-free" method, it is unclear how optimal values are selected without validation data.

## Nice-to-Haves

- **Qualitative examples of merged model outputs**: For a multimodal paper, showing actual VQA predictions, grounding results, and failure cases would help readers assess real-world utility and understand what capabilities are preserved or lost during merging.

- **Task interference quantification**: The paper claims task vectors contain "significant redundancy and noise" but does not quantify interference between task pairs (e.g., via cosine similarity or subspace overlap). Such analysis would strengthen the empirical motivation for SVD-based denoising.

## Novel Insights

The observation that SGD works better than Adam for LoRA model merging—with initialization playing a critical role—deserves further investigation. Table 4 shows SGD alone degrades Qwen2-VL performance by -9.77%, but combining SGD with mean initialization yields +4.43% improvement. This counterintuitive finding about optimizer choice for low-rank task vectors, attributed to "implicit regularization" and better handling of null spaces, is an interesting direction that merits deeper mechanistic analysis.

## Potentially Missed Related Work

- **AdaMMS (Du et al., 2025)** and **UQ-Merge (Qu et al., 2025)**: These MLLM-specific merging methods are discussed in Related Work but not included in experimental comparisons. Direct comparison on the benchmark would clarify OptMerge's relative strengths.

## Suggestions

1. **Correct abstract claims**: Replace "outperform mixture training" with "approach or match mixture training," and explicitly derive the 2.48% improvement figure from reported results.

2. **Add statistical rigor**: Report means and standard deviations over at least 3 random seeds to establish statistical significance of improvements.

3. **Include AdaMMS comparison**: Implement AdaMMS's unsupervised coefficient selection on the benchmark for fair comparison with prior MLLM merging work.

4. **Explain the data-free λ selection paradox**: Either clarify how optimal λ is selected without validation data, or acknowledge that the current implementation requires some form of validation set—potentially qualifying the "data-free" claim.

5. **Add InternVL2.5 ablation**: Provide ablation results for the full fine-tuning setting to validate the centering + SVD design choices.

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (3.0/10)
- Match: N/A

### Final Review

## Summary
The paper establishes a theoretical connection between activation steering and influence functions, proving that—to first order—these techniques are equivalent: any steering vector can be represented as an influence weighting over training data and vice versa. The authors provide constructive algorithms (IAS), a diagnostic measure ω(x) for feasibility, spectral optimality results, and generalization bounds.

## Strengths
- **Novel unifying theoretical framework**: The paper provides the first formal equivalence between activation steering and influence functions, two previously disconnected areas of interpretability research. Theorems 4.2, 5.1-5.3, and 6.1-6.2 establish this rigorously.
- **Practical diagnostic tool**: The alignment measure ω(x)—the cosine of the smallest principal angle between Jacobian subspaces—offers a concrete pre-check for practitioners. Figure 2 validates that ω increases monotonically with layer depth (0.64 → 0.94), supporting the theoretical prediction that later layers are better suited for steering.
- **Computational efficiency**: The method requires only two Jacobian-vector products per input and a rank-d pseudoinverse, making it tractable for large models.
- **First-order equivalence empirically supported**: Figure 1 shows cosine 0.978 between predicted and actual logit shifts across 5000 prompt-token pairs, validating the core theoretical claim.

## Weaknesses
- **IAS underperforms the CAA baseline**: Table 1 shows IAS achieves worse toxicity (0.0164 vs. 0.0150) and worse perplexity (13701 vs. 13291) compared to Contrastive Activation Addition. This is not discussed despite being a central practical question—why use IAS if CAA is better?
- **No empirical validation of data attribution workflow**: A core claim is that steering vectors can be mapped back to "causal training examples" via ρ_s (Corollary 1), but the paper provides no experiments validating this. No training examples are inspected to verify causal relevance, undermining the practical utility claim.
- **No comparison to established influence methods**: The paper claims IAS provides an "integrated workflow" for data attribution but never compares against standard influence function baselines (Koh & Liang, TracIn, RelatIF). It is unknown whether IAS recovers similar or better training examples.
- **Systematic first-order deviation unexplained**: The slope of 1.50 in Figure 1 indicates predictions systematically overshoot actual logit shifts by 50%. The paper states this is "consistent with linear regime" but provides no analysis of why this bias occurs or what magnitude range is "small enough."
- **Limited experimental scope**: Only GPT-2 Medium is tested for language tasks; only one ImageNet class is tested for spectral optimality. No validation on modern LLMs (Llama, Mistral) where steering is practically relevant.
- **Missing statistical significance**: Table 1 reports point estimates without confidence intervals or significance tests.

## Nice-to-Haves
- Ablation studies on the damping parameter φ and rank k would clarify robustness.
- Experiments varying steering magnitude to characterize where the first-order approximation breaks down.
- At least one case study showing: steering vector → top identified training examples → verification of causal relevance.

## Novel Insights
The primal-dual formulation in Section 3 is particularly elegant: the primal program finds the minimum-norm activation change matching a parameter-space perturbation, while the dual program reveals a Fisher-metric certificate λ_ε quantifying the "effort" required to cover components outside the activation subspace. This provides a principled diagnostic—‖λ_ε‖ large signals that steering is insufficient and weight-space editing is needed—computed with the same cost as steering itself. The No-Free-Lunch theorem (6.2) formalizes this intuition: when ω(x) is small, the geometry itself forbids steering from replicating influence effects.

## Potentially Missed Related Work
No potentially missed related work was identified in this review cycle.

## Suggestions
- Add at least one experiment validating the data attribution workflow: compute ρ_s for a steering vector, retrieve top training examples, and verify causal relevance through manual inspection or a proxy task.
- Discuss why IAS underperforms CAA and under what conditions IAS would be preferred (e.g., when the diagnostic workflow is the primary goal rather than steering efficacy alone).
- Report confidence intervals and statistical tests for Table 1 results.
- Validate the framework on at least one modern LLM (≥1B parameters) to demonstrate scalability.

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper investigates memory optimization strategies for reasoning models, where KV cache can dominate memory consumption due to extended generation. Through systematic experiments spanning 1,700+ configurations across three model families (Qwen3, DeepSeek-R1-Distill, OpenReasoning-Nemotron) and four benchmarks, the authors identify a scale-dependent threshold: models below approximately 8-bit 4B effective size benefit from allocating memory to larger weights rather than longer generations, while larger models benefit from the opposite strategy. The paper also finds that mathematical reasoning tasks favor 8- or 16-bit weights over 4-bit, while knowledge-intensive tasks favor 4-bit quantization.

## Strengths

- **Comprehensive empirical study**: The paper spans model sizes (0.6B–32B), weight precisions (4/8/16-bit), token budgets (2k–30k), parallel scaling group sizes (1–16), and KV cache compression strategies (eviction and quantization), providing a thorough characterization of the memory-accuracy design space.

- **Practical relevance and actionable findings**: The memory-centric framing addresses real-world deployment constraints that FLOPs-based test-time scaling studies overlook. The paper provides concrete guidelines (e.g., "for models effectively smaller than 8-bit 4B, prioritize model capacity over test-time compute") that practitioners can directly apply.

- **Clear identification of task-dependent precision preferences**: The finding that mathematical reasoning and code generation favor higher precision while knowledge-intensive tasks favor 4-bit quantization is both practically useful and theoretically interesting, suggesting different tasks place different demands on model parameters.

- **Validation across model families**: The generalization of findings to DeepSeek-R1-Distill and OpenReasoning-Nemotron strengthens claims that results are not model-specific, with consistent threshold behavior observed across architectures.

## Weaknesses

- **Lack of statistical rigor**: All accuracy measurements are reported as point estimates without standard errors or confidence intervals. For benchmarks like AIME25 with challenging problems, variance can be substantial, and readers cannot assess whether observed differences between configurations are statistically meaningful. This is particularly important when comparing configurations with similar performance near threshold boundaries.

- **Empirical threshold without theoretical grounding**: The central "8-bit 4B" threshold is identified post-hoc from the data without principled justification. The paper does not explain why this specific effective size divides the two regimes, nor does it provide sensitivity analysis around this boundary. Understanding whether this threshold relates to model capacity, reasoning depth requirements, or KV cache growth dynamics would strengthen the contribution.

- **Limited mechanistic explanation for key findings**: The paper observes that 4-bit quantization harms mathematical reasoning more than knowledge-intensive tasks, claiming that math "may rely on numerical precision within the weights," but provides no evidence for this mechanism. Similarly, the finding that eviction outperforms quantization for small models is presented without explanation—whether this stems from reduced redundancy in smaller KV caches, quantization noise propagation, or other factors remains unexplored.

- **Narrow eviction algorithm comparison**: Only R-KV and StreamingLLM are evaluated among many eviction methods in the literature. The claim that "eviction provides better memory-accuracy trade-off than KV cache quantization for small models" may not generalize to other eviction algorithms like H2O, SnapKV, or Scissorhands.

## Nice-to-Haves

- Sensitivity analysis for hyperparameters such as GPTQ group size (fixed at 128) and budget forcing prompt strategies (e.g., alternative continuation prompts beyond "Wait").

- Hyperparameter sensitivity around the KV cache quantization group size (fixed at 64), which could affect the relative performance of quantization vs. eviction.

## Novel Insights

The paper's shift from FLOPs-centric to memory-centric analysis of test-time scaling is a valuable reframing that better reflects real deployment constraints. The observation that reasoning models require fundamentally different memory allocation strategies than non-reasoning models—specifically, that the conventional wisdom of 4-bit quantization as universally memory-optimal fails for mathematical reasoning—challenges existing assumptions in the quantization literature. The identification of a scale-dependent threshold where model capacity and test-time compute trade off differently is conceptually interesting and suggests that optimal deployment strategies depend critically on model scale. The finding that eviction outperforms quantization for small effective models while both strategies are competitive for larger ones provides practical guidance for practitioners choosing between these compression approaches.

## Potentially Missed Related Work

- None identified (related work search was not performed for this review).

## Suggestions

- Add error bars or confidence intervals to accuracy measurements in main figures, particularly for comparisons near threshold boundaries where small differences could change optimal strategy recommendations.

- Provide a principled explanation for the 8-bit 4B threshold, potentially through theoretical analysis relating model capacity to reasoning depth requirements, or through sensitivity analysis showing robustness of conclusions to the exact threshold value.

- Include at least one alternative eviction method beyond R-KV and StreamingLLM to strengthen generalization claims about eviction vs. quantization trade-offs.

- Consider adding qualitative examples showing how 4-bit quantization degrades mathematical reasoning (e.g., error propagation in multi-step problems) to support the precision-sensitivity claim.

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether the complexity of Group Relative Policy Optimization (GRPO) is necessary for training LLMs to reason. Through systematic ablation, the authors find that negative feedback is essential while PPO-style clipping can be removed. They propose RGRA (REINFORCE with Group Relative Advantage), a simplified variant that matches or exceeds GRPO performance on mathematical reasoning benchmarks.

## Strengths
- **Well-motivated research question**: The paper addresses a practically important issue—whether GRPO's complexity is justified—which has implications for understanding and efficiency of RL post-training for reasoning.
- **Systematic ablation methodology**: The paper cleanly decomposes GRPO into components (group-relative advantage estimation, PPO clipping, KL regularization) and tests each variant separately, enabling clear attribution of effects.
- **Multiple model families tested**: Experiments include both Qwen2.5 (0.5B, 1.5B) and Llama3.2 (1B), providing evidence that findings generalize across architectures.
- **Comprehensive benchmark coverage**: Evaluation spans 9 benchmarks across English and Chinese mathematics and STEM domains.
- **Counterintuitive finding with practical value**: The result that PPO-style clipping is unnecessary challenges conventional wisdom and, if validated at scale, offers practitioners a simpler training pipeline.

## Weaknesses
- **No statistical significance reporting**: Tables report single accuracy values without error bars, confidence intervals, or indication of whether results are averaged across multiple seeds. Differences of 1-3 percentage points (e.g., GSM8K: 50.9 vs 53.1 for GRPO vs RGRA) could reflect random variation rather than genuine improvements. This significantly weakens confidence in the claimed performance advantages.
- **Limited model and data scale**: Experiments use only models up to 1.5B parameters and train on just 1,800 GSM8K examples. While acknowledged as a limitation, this substantially limits generalizability claims—findings may not transfer to the larger models where reasoning behaviors are more pronounced.
- **Missing key ablations**: The paper does not ablate the KL regularization term, group size (G=8), or temperature, leaving unclear whether RGRA's stability depends on these specific settings. The edge case where all samples in a group receive identical rewards (causing division by near-zero std) is also not addressed.
- **Underdeveloped theoretical justification**: The paper empirically observes that clipping is unnecessary but offers limited analysis of *why* group-relative advantage estimation provides sufficient stability. The connection to Ahmadian et al.'s observation about strong initial policies is mentioned but not developed into a principled explanation.

## Nice-to-Haves
- Comparison against at least one recent GRPO variant (e.g., DAPO, CPPO) mentioned in related work to position RGRA's relative effectiveness.
- Analysis of training efficiency (steps to convergence, computational cost) since a simpler loss may offer practical advantages beyond accuracy.
- KL divergence tracking over training to clarify whether KL regularization—rather than clipping—is the primary stabilizing factor.

## Novel Insights
The paper provides a valuable practical insight: methods that ignore negative feedback (RAFT, GRPO-pos) exhibit catastrophic collapse, with the 0.5B model converging to degenerate outputs within ~20 steps. This phenomenon, attributed to "reward hacking," offers a concrete warning to practitioners. The systematic demonstration that PPO-style clipping is unnecessary when training from strong pretrained policies—and that a simple REINFORCE variant with group-relative advantages suffices—provides actionable guidance for simplifying RL post-training pipelines.

## Potentially Missed Related Work
- None identified (the related work search was not performed for this submission).

## Suggestions
1. **Run multiple random seeds (minimum 3-5)** and report mean ± standard deviation for all benchmark results. Claims that RGRA outperforms GRPO require statistical substantiation.
2. **Add ablation for KL regularization coefficient** to clarify whether this component is essential for RGRA's stability, and consider analyzing the edge case of uniform group rewards.
3. **Test with at least one larger model (e.g., 3B-7B)** to establish that findings generalize beyond small-scale experiments, even if limited to one configuration.

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary
This paper presents a unified theoretical framework for understanding simplicity bias in neural network learning through "saddle-to-saddle dynamics." The authors show that various architectures—fully-connected linear/ReLU networks, convolutional networks, and self-attention models—learn solutions of increasing complexity by traversing a hierarchy of saddle points connected by invariant manifolds. The key contributions are: (1) characterization of embedded fixed points across architectures (Theorem 1), (2) analysis of invariant manifolds that constrain network complexity during training (Theorem 3), and (3) identification of two distinct timescale separation mechanisms—data-induced (leading to low-rank weights) versus initialization-induced (leading to sparse weights).

## Strengths
- **Unified theoretical framework**: The paper successfully unifies analysis of multiple architectures under Equation (1), providing a common formalism for discussing fixed points and invariant manifolds across fully-connected, convolutional, and attention-based networks. Theorems 1 and 3 are rigorously proven and apply to general deep networks.

- **Novel distinction between mechanisms**: The separation between data-induced timescale separation (low-rank solutions, Figure 1B-C) and initialization-induced timescale separation (sparse solutions, Figure 1F-G) is an insightful contribution that explains previously disparate observations in the literature.

- **Testable predictions with experimental validation**: The paper derives non-trivial predictions validated experimentally: network width minimally affects plateau duration in linear networks but shortens plateaus in quadratic/self-attention models (Figure 2A); equal singular values in data eliminate intermediate plateaus for linear networks but not for quadratic networks (Figure 2B); large low-rank initialization induces saddle-to-saddle dynamics even away from saddles (Figure 2C).

- **Clear architecture-appropriate definition of simplicity**: The paper defines complexity as the number of effective units (neurons, kernels, or attention heads), which naturally connects to each architecture's inductive bias and provides a principled link between stages and simplicity.

- **Honest acknowledgment of limitations**: The paper clearly states the scope (two-layer dynamics analysis) and identifies open questions (exhaustiveness of fixed points, deep network extensions).

## Weaknesses
- **Dynamics analysis limited to two-layer networks**: While fixed points and invariant manifolds are characterized for general deep networks, the rigorous dynamics analysis (Theorem 4, Proposition 5) only covers two-layer networks. Extensions to deep networks remain conjectural, limiting the claimed universality.

- **Heuristic connection between invariant manifolds and dynamics**: The paper proves invariant manifolds exist (Theorem 3) but relies on heuristic arguments to establish that gradient flow trajectories actually follow these manifolds during saddle escapes. Rigorous proofs for these transitions remain open, acknowledged by the authors but representing a gap between structural and dynamical claims.

- **Lack of statistical rigor in experiments**: All experimental results show single training runs without error bars or variance estimates over multiple seeds. Given the stochastic nature of initialization, this limits confidence in the empirical claims.

- **Limited empirical validation scope**: Experiments use synthetic data generated by teacher networks and MNIST binary classification. Validation on more complex or realistic benchmarks would strengthen claims about broad applicability.

- **Missing quantitative characterization of conditions**: Key quantitative questions—such as how close initialization must be to invariant manifolds for trajectories to approach fixed points, or how long plateaus last as a function of singular value gaps—remain qualitative.

## Nice-to-Haves
- Systematic quantification of the "boundary" between saddle-to-saddle and smooth dynamics regimes as a function of initialization scale, beyond the qualitative exploration in Figure 2D.

- Explicit comparison to alternative simplicity bias frameworks (NTK spectral bias, implicit regularization) to clarify the relationship between these theories.

- Full trajectory visualizations projected onto low-dimensional subspaces to show that paths actually track invariant manifolds, rather than just showing endpoint configurations.

## Novel Insights
The distinction between data-induced and initialization-induced saddle-to-saddle dynamics is genuinely novel. Prior work on saddle-to-saddle dynamics did not clearly separate these two mechanisms, which lead to fundamentally different weight structures (low-rank versus sparse). This insight explains why different architectures exhibit qualitatively different learning curves—the mechanism depends on whether the architecture has linear or quadratic/quasi-quadratic dependence on weights. The finding that large low-rank initialization can induce saddle-to-saddle dynamics without starting near a saddle (Figure 2C) is also unexpected and reveals that proximity to invariant manifolds, not proximity to saddles, is the key condition.

## Potentially Missed Related Work
- None identified through the review process. The paper provides substantial related work coverage in Appendix A.

## Suggestions
- Add error bars or confidence intervals to experimental figures by running multiple seeds with different random initializations.

- Provide quantitative bounds (even asymptotic) on the distance to invariant manifolds required for saddle-to-saddle dynamics to occur.

- For the deep network conjecture, either provide one rigorous result for a special case (e.g., three-layer linear network) or explicitly state that the extension is empirical observation pending theoretical analysis.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

This paper addresses the "copy-paste" artifact in identity-consistent image generation, where models undesirably replicate reference images rather than generating identity-preserving outputs with natural variation. The authors contribute: (1) MultiID-2M, a large-scale paired dataset of 500k group photos with reference images; (2) MultiID-Bench, a benchmark with a novel Copy-Paste metric quantifying the trade-off between identity fidelity and variation; and (3) WithAnyone, a FLUX-based model using ground-truth aligned ID loss and contrastive training with extended negatives to mitigate copy-paste while maintaining identity fidelity.

## Strengths

- **Novel problem formulation**: The paper identifies and formalizes the copy-paste artifact as a distinct failure mode, correctly noting that reconstruction-based training inadvertently encourages copying—a meaningful insight that challenges the prevailing "higher similarity is better" paradigm (Section 1, Figure 2).

- **Substantial dataset contribution**: MultiID-2M provides 500k identified group photos with paired references (~100 images per identity across ~3k identities), filling a critical gap in available resources. Table 4 demonstrates significant scale improvements over prior datasets.

- **Thoughtful benchmark design**: MultiID-Bench introduces Sim(GT) as the primary metric and a Copy-Paste score (Equation 2) that captures relative bias toward reference versus ground truth. This properly incentivizes models to generate identity-consistent outputs that follow prompts rather than copying.

- **Strong empirical results**: WithAnyone achieves the highest Sim(GT) with the lowest Copy-Paste score across all tested models (Tables 1–2, Figure 5), demonstrably breaking the observed trade-off curve. User study (Figure 8) confirms perceptual quality preferences.

- **Technically sound training innovations**: The GT-aligned ID loss (Equations 4, 14) enables identity supervision at all noise levels efficiently by using ground-truth landmarks instead of noisy predictions. The contrastive loss with extended negatives (up to 4096) leverages the labeled dataset for stronger discrimination signals.

## Weaknesses

- **Lack of statistical rigor**: Tables 1–3 report point estimates without error bars, standard deviations, or significance tests. The numerical differences between methods (e.g., Sim(GT) of 0.464 vs 0.452) cannot be assessed for statistical meaningfulness without variance estimates. This is particularly important for claims of "state-of-the-art" performance.

- **Dataset quality concerns not analyzed**: The identity assignment uses ArcFace cosine similarity thresholds (0.4–0.5), which may introduce false positive and false negative errors at 2M scale. No analysis is provided quantifying error rates or their propagation through training.

- **Missing ablations for key components**: The attention mask and location control mechanism (Appendix E) is described but never ablated in Table 3. Additionally, the negative pool size is only ablated at two extremes (63 vs 4096) without intermediate values, leaving the cost-benefit trade-off unclear.

- **No failure case analysis**: The paper shows only successful generations. Without examples of where WithAnyone struggles (extreme poses, occlusion, challenging lighting, identity blending), readers cannot assess real-world limitations.

- **Limited non-celebrity evaluation**: Figure 16 shows 3 qualitative examples on non-celebrity identities, but systematic quantitative evaluation is absent. Since training focuses on celebrities (Figure 13), the generalization claim to "WithAnyone" requires stronger validation.

- **Demographic bias unaddressed**: Figure 13 shows heavy skew toward Chinese/US/European celebrities, yet no analysis examines whether identity preservation quality varies across demographic groups or whether underrepresented groups perform worse.

## Nice-to-Haves

- Output diversity demonstration: The paper claims reduced copy-paste enables diversity, but never shows multiple outputs for identical inputs to demonstrate this benefit visually.

- Mechanistic analysis of copy-paste: A deeper investigation into why reconstruction training causes copying (e.g., attention patterns, gradient flow) would strengthen the theoretical contribution.

- Computational cost analysis: The four-phase training pipeline's GPU hours and resource requirements are not discussed, which would aid reproducibility assessment.

## Novel Insights

The Copy-Paste metric formulation is genuinely insightful: by measuring the relative angular distance between the generated image and reference versus ground truth (M_CP = (θ_gt - θ_gr) / max(θ_tr, ε)), the authors capture a previously unquantified failure mode. This metric correctly penalizes trivial copying while rewarding faithful identity preservation with natural variation. The finding that existing methods cluster along a trade-off curve (Figure 5), while WithAnyone operates outside this curve, provides compelling evidence that the proposed training paradigm fundamentally changes the optimization landscape rather than merely trading one metric for another.

## Potentially Missed Related Work

None identified—the paper provides a comprehensive survey of identity customization methods and properly situates its contributions.

## Suggestions

1. **Add confidence intervals**: Report standard deviations across multiple runs or bootstrap samples for all quantitative metrics to enable assessment of statistical significance.

2. **Analyze dataset noise propagation**: Quantify ArcFace matching error rates and discuss how they might affect training quality.

3. **Include failure cases**: Add a figure showing failure modes to provide balanced assessment of limitations.

4. **Expand ablation granularity**: Test intermediate negative pool sizes (256, 512, 1024) to justify the 4096 choice and help practitioners understand compute/quality trade-offs.

5. **Evaluate demographic fairness**: Since the dataset has known demographic skew, analyze whether identity preservation performance differs across demographic groups.

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

This paper investigates output volatility in long-form LLM generation—the phenomenon where models produce inconsistent outputs across multiple generations from the same prompt. The authors introduce VOLTBench, a comprehensive benchmark measuring both length volatility and generation quality across structured and unstructured tasks; conduct mechanistic analysis identifying "Attention Collapse" and "Attention Instability" as failure patterns; and propose SELB, a training-free decoding strategy that enforces structural constraints via logit manipulation.

## Strengths

- **Novel problem formulation**: The paper identifies output volatility as a critical but overlooked dimension of long-form generation. Figure 1 compellingly shows LongWriter-8B with output standard deviation reaching 103% of mean length—a finding with real implications for deployment reliability and cost predictability.

- **Comprehensive benchmark design**: VOLTBench spans structured (code, LaTeX, JSON) and unstructured (story, diary) tasks, two languages, multiple complexity levels, and scales up to 500 sections. The multi-sample volatility metrics (LSD, LVC, MLA) are methodologically sound additions to existing benchmarks.

- **Mechanistic insight**: The attention trace analysis (Section 5, Figures 4 and 9) provides interpretable explanations for failure modes, linking "Attention Collapse" to premature termination and "Attention Instability" to section skipping. The CKA analysis (Appendix H) offers supporting evidence that SELB maintains representational stability (cosine similarity ~0.68 vs. ~0.34 for baseline at t=10,000).

- **Strong empirical results**: SELB achieves substantial improvements—on 100-section tasks, generating 88/100 sections vs. LongWriter's 45, with 100% structured content accuracy vs. LongWriter's 32.6% (Table 31). The training-free nature enables immediate application to existing models.

- **Extensive model coverage**: Experiments evaluate 9 diverse models (GPT-4o-mini, Claude-3.5-Sonnet, DeepSeek-R1/V3, Qwen variants, Llama3.1, LongWriter, Mamba), demonstrating that output volatility is a widespread problem.

## Weaknesses

- **Quantitative claim inconsistency**: The abstract claims the method "improves the mean output length of the base model by 148%." However, examining Table 31, the baseline Qwen2.5-7B produces 350 words while SELB produces 15,651 words—a much larger improvement. The 148% figure requires clarification and may indicate an error or reference to a different baseline.

- **Limited generalization validation**: SELB is applied only to Qwen2.5-7B in the main experiments. While the attention analysis examines Qwen2.5-3B, validating the method across different architectures (Llama, DeepSeek, Mamba) would strengthen claims about general applicability.

- **Missing ablation studies**: The method combines structural enforcement (M_struct), failure token suppression (M_fail), and EOS blocking. No experiments isolate the contribution of each component, leaving unclear which aspects drive the improvements.

- **Insufficient hyperparameter analysis**: Critical parameters (boost magnitude β, section length threshold τmax, banned token set V_banned) are neither specified nor analyzed for sensitivity. Practitioners cannot determine optimal settings or understand how performance varies with these choices.

- **Baseline comparison concerns**: The primary baseline (Qwen2.5-7B) fails catastrophically on long-form tasks (350 words on a 100-section task), making relative improvements appear dramatic. Comparisons against prompt-based length enforcement or simpler decoding interventions would better contextualize the method's contribution.

## Nice-to-Haves

- Explicit specification of V_banned tokens and β values for reproducibility
- Inference time and memory overhead measurements
- Qualitative examples comparing baseline vs. SELB outputs to verify quality maintenance
- Cross-model validation (applying SELB to at least one different architecture)

## Novel Insights

The attention-based mechanistic analysis offers a valuable explanatory framework for long-form generation failures. The observation that "Attention Collapse" (attention dropping to near-zero before premature termination) and "Attention Instability" (abnormal spikes preceding section skipping) predict observable failures provides a diagnostic tool for understanding when and why models abandon tasks. The finding that structured tasks exhibit lower volatility than unstructured ones (Figure 3) is counter-intuitive and suggests that explicit structural guidance helps models maintain coherence—a principle that informs the SELB design. The CKA analysis linking representational drift to content degradation offers a mechanistic explanation for why forced structural enforcement preserves quality.

## Potentially Missed Related Work

None identified.

## Suggestions

1. **Clarify the 148% improvement claim**: Provide the exact calculation and baseline reference supporting this figure, or correct it if there is an error.

2. **Add component ablation**: Report results for M_struct alone, M_fail alone, and combined to quantify each component's contribution.

3. **Specify implementation details**: List the banned tokens in V_banned, the boost magnitude β, and the section length threshold τmax used in experiments.

4. **Validate across architectures**: Apply SELB to at least one additional model family to demonstrate generalization beyond Qwen.

5. **Compare against simpler baselines**: Include prompt-based length enforcement as a baseline to contextualize the value of logits manipulation.

---

## rI2Fa13fUL

- GT: Reject (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Generative Trajectory Policies (GTP), a unified framework for offline RL that views diffusion, flow matching, consistency models, and related approaches as instances of learning a continuous-time ODE solution map Φ(x_t, t, s). The authors address practical challenges through a theoretically grounded score approximation technique and an advantage-weighted variational training objective, achieving state-of-the-art performance on D4RL benchmarks.

## Strengths
- **Principled theoretical unification**: The paper successfully unifies diffusion models, consistency models, consistency trajectory models, shortcut models, and mean flows under a single ODE-based framework (Section 3). The parameterization via Φ and ϕ, along with two complementary training objectives, provides a clean conceptual foundation for understanding generative policies.

- **Strong empirical results**: GTP achieves impressive performance on D4RL benchmarks. On Gym tasks, it achieves 89.0 average score vs. 87.9 for Diffusion-QL; on AntMaze, it achieves 80.6 vs. 78.3 for QGPO. The perfect score (100.0) on antmaze-umaze and state-of-the-art performance on several challenging AntMaze tasks is genuinely meaningful.

- **Theoretical justification for score approximation**: Theorem 1 rigorously establishes that replacing the true vector field with the closed-form surrogate f̃(x_t, t) = (x_t - x)/t changes the objective by only O(h^p), providing formal grounding for efficient training without ODE solver integration.

- **Well-motivated advantage-weighted objective**: The derivation in Theorem 2 and Appendix B.5 shows that exponential advantage weighting emerges naturally from KL-regularized policy optimization, properly grounding the value-guidance mechanism in established theory.

- **Comprehensive baseline comparison**: Tables 1-2 compare against a wide range of methods including BC, IQL, CQL, TD3+BC, Diffusion-QL, QGPO, Consistency-AC, and others across both behavior cloning and offline RL settings.

## Weaknesses
- **Incomplete baseline comparisons**: Several entries in Table 2 are missing (BDM and C-AC show "-" for some AntMaze tasks). The paper should explain whether these methods were not run, failed to produce results, or were excluded for other reasons. This affects the completeness of the empirical comparison.

- **Limited efficiency analysis**: Table 6 reports inference time on only one task (halfcheetah-medium-expert). While the paper emphasizes the expressiveness-efficiency trade-off as a core contribution, a more comprehensive efficiency evaluation across multiple tasks would strengthen this claim. Training time is also only partially addressed.

- **Large variance on some tasks**: The variance on certain AntMaze tasks is substantial (e.g., antmaze-mp: 94.2 ± 8.1). While still outperforming baselines, this suggests some instability that warrants discussion.

- **Ablation design could isolate components better**: Table 3 shows the combined effect of score approximation by comparing GTP to a version without it, but the relative contributions of the two key techniques (score approximation and variational guidance) are not cleanly separated. The comparison to linear Q-terms shows the variational guidance helps, but a more systematic dissection would strengthen the ablation.

## Nice-to-Haves
- Include experiments on additional D4RL domains (Adroit, Kitchen) or other offline RL benchmarks to demonstrate broader applicability beyond locomotion tasks.
- Provide training dynamics curves in the appendix showing convergence behavior and stability across methods.
- Visualize learned action distributions on actual D4RL tasks to validate expressiveness claims beyond the synthetic 2D example.

## Novel Insights
The key insight is that the expressiveness-efficiency trade-off in generative policies can be resolved by learning the *full solution map* Φ(x_t, t, s) rather than committing to either slow iterative sampling (diffusion) or fast single-step approximations (consistency models). The paper shows that the surrogate score approximation (x_t - x)/t provides an analytically tractable training signal that converges to the ideal objective as discretization becomes fine, while the advantage weighting naturally emerges from KL-regularized optimization. This reveals that prior methods like CTMs, Shortcut Models, and Mean Flows are actually learning specific aspects of the same underlying ODE trajectory—a unifying perspective that could inform future policy designs.

## Potentially Missed Related Work
- None identified (related work search was not performed).

## Suggestions
- Add a brief note in the paper explaining the missing baseline values in Table 2, whether due to computational constraints, method incompatibility, or other reasons.
- Report inference time across at least 2-3 diverse tasks (not just halfcheetah-medium-expert) to substantiate the efficiency claims more robustly.
- Consider adding a dedicated paragraph in Section 3.4 that explicitly articulates what aspects of GTP are novel versus adapted from prior methods (CTM's trajectory consistency, consistency training's analytical supervision), to help readers understand the incremental contribution.

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (6.5/10)
- Match: N/A

### Final Review

## Summary

This paper introduces the conditional causal bandit problem, where interventions are conditional (a variable X is set to a value determined by a policy function g that depends on observed context variables Z_X). The authors provide a rigorous graphical characterization of the minimal globally interventionally superior set (mGISS)—the smallest set of nodes guaranteed to contain the optimal intervention target—and prove it equals the LSCA closure of the parents of the target variable Y. A linear-time algorithm (C4) computes this set, and empirical validation demonstrates substantial search space reduction on both synthetic and real-world graphs.

## Strengths

- **Novel problem formulation**: The extension from hard interventions to conditional interventions (do(X = g(Z_X))) meaningfully advances causal bandits toward realistic decision-making settings where intervention values depend on observed context.

- **Strong theoretical foundation**: Proposition 4 elegantly establishes equivalence between conditional intervention superiority and deterministic atomic intervention superiority, enabling simpler analysis. Theorem 13 provides a clean graphical characterization via Λ-structures and LSCA closure.

- **Efficient algorithm**: The C4 algorithm correctly computes mGISS in O(|V| + |E|) time using the connector concept (Definition 14), with rigorous correctness proof (Theorem 16).

- **Empirical validation**: Experiments on bnlearn real-world graphs demonstrate substantial search space reduction—up to 90%+ for larger models (Figure 6). Integration with CondIntUCB shows practical regret improvements (Figure 3).

- **Complete proofs**: All theoretical results are fully proven in the appendices, including non-trivial constructions in Appendix F establishing minimality.

## Weaknesses

- **Restrictive conditioning set assumptions**: The paper assumes An(X) \ {X} ⊆ Z_X and requires the "observable conditioning set" property (Z_W ⊆ Z_X when W ∈ An(X)). These may not hold in practice, and no sensitivity analysis or relaxation is provided for cases where only partial context is available.

- **No latent confounders**: The assumption of no unobserved confounding significantly limits practical applicability, as many real-world causal systems involve latent variables. While acknowledged, the paper offers no analysis of how results degrade under violations.

- **Missing baseline comparisons**: The experiments compare only against brute-force (all ancestors). Comparing against simpler heuristics—such as selecting only Pa(Y), distance-k ancestors, or random subsets—would demonstrate whether the mGISS computation complexity is justified.

- **No theoretical regret analysis**: While empirical regret improvements are shown, the paper lacks theoretical bounds quantifying how search space reduction from |An(Y)| to |mGISS| translates to provable regret or sample complexity improvements.

- **Limited sensitivity analysis on target selection**: Experiments only use Y as the node with most ancestors. Effectiveness likely depends on Y's position in the graph (depth, number of parents), but this is not analyzed.

## Nice-to-Haves

- Comparison to simpler node selection heuristics (e.g., Pa(Y) only, k-nearest ancestors) to justify the algorithm's complexity
- Discussion of how to handle continuous variables or large conditioning sets where policy learning becomes computationally challenging
- Analysis of robustness to graph misspecification (missing or misoriented edges)

## Novel Insights

The key conceptual insight is that conditional intervention superiority coincides with deterministic atomic intervention superiority (Proposition 4), which dramatically simplifies the analysis by allowing reasoning in terms of fixed interventions on deterministic models. The Λ-structure characterization (Theorem 12) provides an elegant alternative to the recursive LSCA definition: a node belongs to the mGISS if and only if it forms a Λ-structure over (Pa(Y), Pa(Y)), capturing the intuition that nodes affecting Y through multiple divergent paths to different parents are worth intervening on. The linear-time C4 algorithm exploits this via the connector concept—tracking whether all paths from a node reach Pa(Y) through the same "bottleneck" or through multiple distinct paths.

## Potentially Missed Related Work

- The paper could discuss connections to recent work on structural causal bandits with latent confounders (Lee & Bareinboim, 2018, 2019) beyond the current distinction, particularly whether any insights transfer to the single-node case.

## Suggestions

- Add experiments comparing against simpler heuristics (Pa(Y) only, distance-k pruning) to contextualize the benefits of mGISS computation
- Provide theoretical regret bounds showing how the search space reduction provably improves sample complexity
- Include sensitivity analysis varying the target Y (not just the highest-ancestor node) and analyze graph properties predicting large vs. minimal search space reduction

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper investigates optimization differences between Transformers and modern recurrent models (SSMs like Mamba) on associative recall and copying tasks. The authors demonstrate that SSMs exhibit extreme learning rate sensitivity—success confined to narrow windows—while Transformers are robust across wide ranges, and show that SSMs favor width scaling while Transformers favor depth, with architectural ablations identifying convolutions as critical for single-layer performance.

## Strengths
- **Substantial empirical effort with systematic methodology**: Over 3,000 experiments across approximately 20,000 GPU hours with systematic LR grid searches and 5 seeds per configuration, exceeding typical rigor for hyperparameter comparison studies.
- **Important corrective to prior work**: The demonstration that Arora et al.'s learning rate grid missed optimal values for SSMs (Figure 1, dashed lines vs. actual working rates) provides concrete evidence that optimization confounds previous expressivity comparisons—this is a genuinely valuable contribution to fair architecture benchmarking.
- **Counterintuitive single-layer findings with strong ablation support**: The finding that 1-layer Mamba can solve MQAR while 1-layer Transformers cannot challenges theoretical assumptions, and Table 2's ablations (adding convolution to Attention: 2%→99%; removing from Mamba: 99%→2%) provide compelling mechanistic evidence for convolution's role.
- **Cross-task generalization**: The learning rate sensitivity finding replicates on both MQAR and copying tasks (Figures 1 and 5), demonstrating the phenomenon is not benchmark-specific.

## Weaknesses
- **Limited scope to synthetic benchmarks without language modeling validation** — The paper draws implications for language modeling based on tasks "highly correlated with" these abilities, but provides no experiments on actual LLM perplexity or downstream NLP benchmarks. Whether the narrow learning rate window persists at practical scales remains untested, limiting immediate practical impact.
- **Mechanistic claims lack supporting analysis** — The paper claims single-layer Transformers exhibit loss bumps "resembling the formation of an induction head circuit" (Section 6) based solely on loss curve inspection, without attention pattern visualization or gradient flow analysis. Similarly, the DeltaNet stability hypothesis (Householder matrices vs. decay rates) is asserted without gradient norm evidence or controlled ablation isolating this mechanism.
- **Missing optimization interventions** — Despite the central thesis about optimization difficulty, only learning rate is varied. The paper does not test whether gradient clipping, learning rate warmup, cosine schedules, or alternative optimizers (Adam vs. AdamW, different betas) could broaden the stable window. This leaves practitioners with "exhaustive search" as the only recommendation.
- **Over-claiming about expressivity vs. learnability** — The abstract claims "fundamental mismatch in the loss landscape" without formal loss landscape analysis, and the introduction frames optimization as *the* differentiator rather than *a* differentiator. However, Figure 3 shows Hyena struggles at low widths even with proper tuning, confirming expressivity constraints still matter—the contribution is about learnability confounds being underappreciated, not expressivity differences being absent.

## Nice-to-Haves
- Initialization scheme specification and ablation (the paper uses unspecified initialization for SSMs that have specialized strategies like S4's HiPPO).
- Explicit accuracy threshold definition for "solves the task" throughout (approaching 100% in figures, but unstated).

## Novel Insights
The finding that convolution is both necessary for Mamba's single-layer MQAR performance and sufficient to enable single-layer Attention solves this task reveals a deeper mechanistic connection: locality priors matter for shallow sequence mixers regardless of architecture. The opposing scaling behaviors (SSMs favor width to expand hidden state capacity; Transformers favor depth for induction head formation) provide actionable guidance for architecture design—the paper correctly identifies that parameter-matched comparisons using deep SSMs vs. shallow Transformers are systematically unfair. The observation that DeltaNet achieves Transformer-like learning rate robustness while Mamba does not points toward architectural modifications (avoiding decay-induced gradient pathologies) as a potential solution direction.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
1. **Validate on actual language modeling**: Add at least one small-scale perplexity experiment (e.g., WikiText-103) to test whether LR sensitivity transfers beyond synthetic tasks—this directly addresses whether MQAR/copying are valid proxies.
2. **Test optimization interventions**: Evaluate whether warmup, cosine decay, or gradient clipping broaden the stable learning rate window—this provides actionable guidance rather than just identifying the problem.
3. **Add gradient norm analysis**: Plot gradient norms across training to empirically validate the hypothesized vanishing/exploding gradient dynamics underlying SSM instability.
4. **Mechanistically verify induction head claims**: Include attention pattern visualizations for single-layer Transformers to substantiate the claimed "attempted" induction head formation.

---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

The paper proposes a learnable three-channel codec architecture inspired by the Gray-Wyner network from information theory, designed to separate common from task-specific information across multiple vision tasks. The authors extend Wyner's and Gács-Körner common information concepts to the lossy case, derive theoretical bounds via interaction information, and propose an optimization objective with a β parameter controlling the transmit-receive rate tradeoff. Experiments on synthetic data, colored MNIST, Cityscapes, and COCO 2017 demonstrate that the approach reduces redundancy and outperforms independent coding baselines.

## Strengths

- **Strong theoretical foundation**: Theorem 1 establishes meaningful bounds relating Gács-Körner and Wyner's lossy common information through interaction information, with a complete proof in Appendix A. Theorem 2 derives a tractable entropy-based optimization objective from the original Gray-Wyner formulation, grounding the architecture in principled information theory.

- **Well-motivated transmit-receive tradeoff**: The distinction between transmit rate (R₀ + R₁ + R₂, relevant when tasks are performed jointly) and receive rate (2R₀ + R₁ + R₂, relevant when tasks are performed on different devices) has genuine practical implications for distributed inference scenarios. The β parameter provides a principled way to navigate this tradeoff.

- **Comprehensive architecture ablations**: The comparison of Shared, Separated, and Combined architectures (Figures 3, 9) provides empirical evidence for the proposed design. The theoretical justification in Appendix C using Rademacher complexity bounds to analyze representation compatibility is a meaningful addition.

- **Edge case validation**: The colored MNIST experiments with Dependent, Independent, and Mixture PMFs (Figure 4) demonstrate that the method appropriately adapts its channel allocation based on the underlying common information structure, validating the theoretical predictions.

- **Consistent empirical improvements**: The proposed method shows BD-rate advantages over independent coding across all tested scenarios, with an average transmit rate improvement of -81.58% reported across three computer vision experiments.

## Weaknesses

- **Missing comparison to existing multi-task compression methods**: The related work cites Chamain et al. (2021), Feng et al. (2022), and Guo et al. (2024) as prior work on multi-task learned codecs, yet experiments only compare against self-designed baselines (Joint, Independent, Separated, Combined). Without comparison to these prior methods, the relative contribution of the Gray-Wyner formulation versus standard multi-task approaches remains unclear.

- **No statistical significance measures**: All rate-distortion results (Tables 2-8, Figures 3-5) report single values without error bars or confidence intervals. For empirical claims about consistent improvements, statistical evidence is needed to establish that observed differences are not due to training variance.

- **Limited evaluation of β on real vision tasks**: β controls the core transmit-receive tradeoff, yet only β = 1 is evaluated on Cityscapes and COCO experiments. The synthetic experiments show β ∈ {1, 1.5, 2} matters, but the practical utility of navigating this tradeoff on real applications is not demonstrated.

- **No verification of actual information separation**: The paper claims the architecture separates common from private information, but provides no quantitative measurement of mutual information between learned representations (e.g., I(Y₀; Y₁), I(Y₀; Y₂), I(Y₁; Y₂|Y₀)). The MNIST reconstructions are qualitative only; Cityscapes and COCO experiments lack channel content visualization entirely.

- **No computational cost analysis**: The three-channel architecture with entropy models introduces additional complexity, but no analysis of parameter count, encoding/decoding time, or FLOPs compared to baselines is provided. This information is essential for practical deployment assessment.

## Nice-to-Haves

- End-to-end training comparison where task models are fine-tuned along with the codec, rather than frozen. This would isolate the contribution of the architecture itself from the limitations of pretrained feature extraction.

- Analysis of the observed gap between theoretical bounds and empirical performance. The paper notes results are "within an order of magnitude of theoretical bounds" but does not investigate what architectural or optimization factors cause this discrepancy.

## Novel Insights

The paper establishes an interesting connection between the separability of common information (Eq. 8's block-diagonal structure) and the practical ability to isolate it across channels. The insight that when mutual information is "separable" from private information, the Wyner and Gács-Körner common information measures coincide—and that this coincidence is rare in practice—provides theoretical grounding for why perfect common information isolation is often unattainable. The interaction between the β parameter and the auxiliary loss weight γ (where γ = 1 can discourage common channel usage, requiring β adjustment) reveals an optimization challenge that could benefit from more principled hyperparameter selection.

## Potentially Missed Related Work

- Chamain et al. (2021), Feng et al. (2022), Guo et al. (2024) are cited in the paper as multi-task codec approaches but are not included as experimental baselines. Comparison to these methods would better position the contribution.

## Suggestions

- Add at least one comparison to an existing multi-task codec method (e.g., from Guo et al. 2024) to establish relative performance against prior work.

- Report standard deviations across multiple training runs or use statistical significance tests for BD-rate comparisons.

- Evaluate multiple β values (e.g., β ∈ {1, 1.5, 2}) on at least one real vision task to demonstrate practical control over the transmit-receive tradeoff.

- Estimate mutual information between learned representations to quantitatively verify that Y₀ captures common information and Y₁, Y₂ capture task-specific information.

- Include parameter count and FLOPs comparison for all methods to enable practical assessment of computational overhead.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

This paper presents a systematic analysis of verifiers used in reinforcement learning with verifiable rewards (RLVR) for mathematical reasoning. The authors find that rule-based verifiers achieve near-perfect precision but suffer from false negatives (~14% recall gap), while model-based verifiers improve recall but are vulnerable to reward hacking during RL training. The key finding is that static classification accuracy does not reliably predict RL training effectiveness.

## Strengths

- **Comprehensive empirical investigation**: The paper evaluates 3 rule-based verifiers and multiple model-based verifiers (off-the-shelf and trained variants) across 5 datasets (MATH, DeepscaleR, ORZ-Math, Skywork-OR1, WebInstruct-Verified), providing strong empirical grounding for its claims.

- **Important practical discovery**: The finding that static accuracy does not correlate with RL effectiveness (Section 5.1, Figure 3) is significant for practitioners—a verifier achieving 92% recall can produce worse RL outcomes than a lower-recall verifier due to reward hacking vulnerability.

- **Systematic adversarial probing methodology**: The construction of 13 distinct hacking patterns (Table 9) and evaluation against multiple verifiers provides actionable diagnostic tools. The finding that discriminative verifiers (xVerify) are more robust than generative ones (Table 3) is practically useful.

- **Clear evidence of reward hacking**: Concrete examples (Figures 11, 12) showing how trained verifiers are fooled by trivial patterns like "{" or gibberish text make the vulnerability tangible and verifiable.

- **Cross-domain validation**: Results extend beyond mathematics to general science (WebInstruct-Verified, Appendix J), demonstrating the generality of findings.

## Weaknesses

- **Lack of statistical significance testing for RL experiments**: The paper reports single training runs without error bars or multiple seeds. This is a significant methodological concern given the known variance in RL training dynamics. The paper acknowledges this limitation but does not address it—reporting results from at least 2-3 seeds with standard deviations would substantially strengthen confidence in the findings.

- **Insufficient mechanistic explanation for fine-tuning vulnerability**: Table 3 shows R1-Distill-Verifier-1.5B has substantially higher attack success rates (35.0% on Adversarial Prefixes) compared to its base model (21.7%). The paper documents this counterintuitive finding but does not explain WHY rejection fine-tuning increases vulnerability—this is perhaps the most important practical finding and deserves deeper investigation.

- **Limited policy model diversity**: All RL experiments use Qwen2.5-7B as the policy model. The finding that "stronger models face harder verification" (Figure 2) is important but only demonstrated across Qwen model scales. Testing different architectures would strengthen generalizability claims.

- **Incomplete analysis of discriminative verifier robustness**: Section 6.2 notes that xVerify (discriminative) is more robust than generative verifiers, but provides limited investigation into what architectural or training differences cause this—a mechanistic explanation would inform future verifier design.

## Nice-to-Haves

- **Computational overhead quantification**: The hybrid verifier design reduces model-based verifier calls (Appendix G), but the actual training throughput impact is not quantified. Practitioners need to know if improved recall justifies the computational cost.

- **Concrete mitigation strategies**: The paper identifies the reward hacking problem but offers limited solutions. Even preliminary experiments with adversarial training or ensemble verification would strengthen practical contributions.

## Novel Insights

The paper makes a counterintuitive discovery: fine-tuning verifiers for higher classification accuracy can *degrade* their robustness during RL training. This reveals a fundamental tension in verifier design—optimizing for static accuracy may inadvertently create exploitable vulnerabilities. The finding that discriminative architectures are inherently more robust than generative chain-of-thought verifiers suggests that reasoning traces, while improving accuracy, may introduce attack surfaces. This insight has implications beyond mathematical reasoning for any RLVR system.

## Potentially Missed Related Work

- **Cobbe et al. (2021), "Training Verifiers to Solve Math Word Problems"**: Early work on training verifiers for mathematical reasoning that could inform the discussion of why fine-tuning affects robustness.

- **Recent concurrent work (Su et al., 2025; Ma et al., 2025; Seed et al., 2025)**: The paper briefly mentions these as concurrent works on model-based verifiers but does not deeply compare approaches or findings.

## Suggestions

1. **Add multi-seed experiments**: Report RL training results from at least 3 random seeds with standard deviations to establish that findings (especially the reward hacking timing) are systematic rather than random.

2. **Investigate why rejection fine-tuning increases vulnerability**: Analyze whether the fine-tuning process narrows decision boundaries, overfits to specific answer formats, or induces distribution shift that adversarial patterns can exploit.

3. **Test additional policy model architectures**: Validate that the "stronger models → harder verification" trend holds across different model families, not just Qwen scales.

4. **Analyze discriminative vs. generative architecture differences**: Investigate whether the robustness advantage comes from the discriminative output format, training objective, or other architectural factors—this would guide future verifier development.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary

EmoSign introduces the first multimodal dataset for emotion recognition in American Sign Language (ASL), containing 200 video clips annotated with 7-point sentiment ratings, 10 emotion categories with intensity levels, and open-ended descriptions of emotion cues. Annotations were collected from 3 Deaf native ASL signers with professional interpretation experience, addressing a key quality issue in prior work. The paper establishes benchmarks across four multimodal LLMs, demonstrating that current models struggle to leverage visual cues for emotion recognition and exhibit bias toward positive emotions.

## Strengths

- **Addresses a genuine gap in the literature**: Emotion recognition in sign languages remains poorly studied compared to spoken languages, with practical consequences for communication access in legal, medical, and other critical settings. The dataset fills a clear need.

- **High-quality annotation methodology**: Recruiting Deaf native ASL signers with professional interpretation experience is a significant strength, as hearing annotators frequently misinterpret grammatical versus emotional facial expressions in ASL (as noted in prior work by Lim et al., 2024).

- **Rich annotation schema with unique qualitative data**: The three-layer annotation (sentiment, emotion categories with intensity, and open-ended cue descriptions) provides valuable granularity. The qualitative descriptions from native signers about specific non-manual markers (facial expressions, signing speed, body movement) offer unique insights not available in other datasets.

- **Strong inter-annotator agreement relative to established benchmarks**: The average Krippendorff's alpha of 0.593 compares favorably to MELD (Fleiss' kappa = 0.43) and IEMOCAP (Fleiss' kappa = 0.48). The finding that positive emotions have higher agreement than negative ones is itself an interesting observation about emotion perception in ASL.

- **Comprehensive ablation study design**: Evaluating models under caption-only, video-only, and video+caption conditions provides clear insights into modality contributions. The key finding—that caption-only performance often matches or exceeds video+caption—demonstrates current MLLMs' limited visual grounding for emotion recognition in sign language.

## Weaknesses

- **Dataset size limits utility**: With only 200 utterances (~16 minutes), the dataset's utility for training models or robust benchmarking is constrained. While the authors cite precedent from other small high-quality datasets, the size reduces statistical power for benchmark comparisons and limits practical impact for downstream applications.

- **VADER-based selection may introduce bias**: Using text-based sentiment (VADER) to select videos creates a mismatch with the paper's stated goal—visual emotion cues in ASL often diverge from text content. The paper notes this discrepancy but does not quantify it or discuss what valuable videos may have been excluded by this filter.

- **Low inter-annotator agreement on specific emotion categories**: Krippendorff's alpha for "surprise (negative)" is 0.119 and for "disgust" is 0.166, substantially below the average. The paper should discuss whether these categories are fundamentally ambiguous in ASL or whether annotation instructions need refinement, as this affects dataset validity for supervised learning on those emotions.

- **No statistical significance testing or cross-validation**: Tables 3 and 4 report single-run results without error bars, confidence intervals, or statistical tests. With 200 samples and class imbalance, variance could substantially affect conclusions about model comparisons.

- **Emotion cue grounding task lacks quantitative evaluation**: The grounding analysis in Section 5.3 is purely qualitative ("manually inspected several randomly selected videos"). Without temporal IoU or similar metrics, claims about model grounding capabilities remain anecdotal.

- **No fine-tuning experiments**: The paper claims the dataset can "inspire new architectures" and serve as a benchmark, but provides no evidence that models can learn from this data. Even a simple linear probe or small-scale fine-tuning experiment would demonstrate the dataset's training utility.

## Nice-to-Haves

- **Multi-label emotion classification results**: The annotation data supports multi-label classification, but the paper only evaluates single-label emotion prediction on a subset of 140 clips. Including multi-label metrics would leverage the full dataset.

- **Analysis of annotator disagreement patterns**: Examining cases where annotators disagreed could illuminate which emotions are inherently ambiguous in ASL expression, providing valuable linguistic insight.

- **Demographic information about annotators**: Given documented regional and demographic variation in ASL, providing annotator demographics (age range, region, etc.) would clarify generalizability.

## Novel Insights

The paper reveals a critical limitation in current multimodal LLMs: they largely fail to integrate visual emotion cues from sign language, instead relying heavily on text captions. The analysis shows that models like GPT-4o fall back to common emotional descriptors (e.g., "happiness," "frustration") without textual context, while AffectGPT exhibits a systematic neutral bias. The qualitative analysis of reasoning outputs demonstrates that models sometimes correctly identify visual cues but interpret them differently depending on whether text context is available—a finding with implications beyond sign language for understanding how MLLMs perform visual grounding. The observation that negative emotions show lower inter-annotator agreement than positive ones in ASL is also noteworthy, potentially reflecting greater subtlety in negative emotional expression or cultural factors in perception.

## Potentially Missed Related Work

The related work search was not performed for this paper, so no specific suggestions are available. The paper's comparison to FePh is appropriate, though a more detailed benchmark comparison would strengthen claims about EmoSign's unique value.

## Suggestions

- **Report cross-validation results**: With 200 samples, use k-fold cross-validation and report mean performance with standard deviations to increase confidence in benchmark comparisons.

- **Add quantitative grounding metrics**: For the emotion cue grounding task, define evaluation criteria (e.g., overlap between model-identified cues and annotator-described cues) to enable systematic comparison.

- **Include at least one fine-tuning experiment**: Demonstrate that the dataset supports learning by fine-tuning a smaller vision-language model or training a linear probe on visual features.

- **Quantify VADER-annotator divergence**: Provide statistics on how often annotator labels differed from VADER predictions, and discuss what types of emotional content may be missed by text-based filtering.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces a benchmark for evaluating text-to-image (T2I) models on their ability to generate images for WordNet taxonomy concepts. The authors propose 9 evaluation metrics—including preference-based ELO scores, reward models, and taxonomy-specific CLIP-based similarity metrics—grounded in probability theory and KL divergence. They evaluate 12 T2I approaches across three test configurations: common-sense concepts, randomly sampled WordNet nodes, and LLM-predicted concepts, finding that Playground-v2 and FLUX consistently outperform others while retrieval-based methods perform poorly.

## Strengths

- **Novel task formulation**: The paper addresses an important gap in understanding how T2I models handle concepts at different levels of abstraction, which has implications for automated taxonomy enrichment and structured knowledge visualization.

- **Principled metric design**: The taxonomy-specific metrics (Lemma, Hypernym, Cohyponym Similarity, and Specificity) are formally derived from probability theory using KL divergence and mutual information (Appendix D), providing theoretical grounding beyond ad-hoc similarity measures.

- **Comprehensive model coverage**: Evaluating 11 generative models plus a retrieval baseline across multiple dataset configurations provides a thorough picture of current T2I capabilities for this task.

- **Human-AI alignment analysis**: The paper provides detailed analysis of GPT-4's correlation with human preferences (Spearman ρ≈0.88-0.92) while transparently identifying and discussing GPT-4's positional bias toward the first option, which is valuable methodological information for future benchmarking work.

- **Useful qualitative error analysis**: Appendix I identifies systematic failure modes (abstract concepts, rare words, functional roles) and visualizes specific artifacts, offering actionable insights for model improvement.

## Weaknesses

- **Core claim about abstraction levels not empirically validated**: The introduction states that the task requires understanding "concepts of different level of abstraction," yet no analysis breaks down performance by depth in the WordNet hierarchy. Without showing how models perform on abstract vs. concrete concepts, this central motivation remains unsubstantiated.

- **Single image per concept limits reliability**: T2I models have high stochastic variance. Generating only one image per concept makes rankings susceptible to seed-level noise and undermines reproducibility of the reported results.

- **No direct comparison to prior metric (ISP)**: The paper claims to generalize In-Subtree Probability from Baryshnikov & Ryabinin (2023) but provides no empirical comparison, making it impossible to assess whether the proposed metrics actually improve upon prior work.

- **GPT-4 positional bias undermines evaluation reliability**: Figure 5 and Appendix G show GPT-4 has strong bias toward the first option (unlike humans). While acknowledged, this is not mitigated (e.g., via position swapping or multiple runs), and GPT-4 rankings are still used as a primary metric.

- **Limited human evaluation scale**: Four annotators evaluating ~3,370 pairs means limited per-annotator data. The paper doesn't report inter-annotator agreement per subset or how ties/"both bad" judgments were distributed.

- **Different metrics produce different "best" models with no resolution**: CLIP-based metrics favor SDXL-turbo while preference metrics favor Playground/FLUX. This discrepancy is noted but not analyzed deeply—it matters whether we're measuring image-concept alignment versus human aesthetic preference, and the paper doesn't clarify which is the appropriate objective.

## Nice-to-Haves

- **Downstream utility demonstration**: A user study showing whether generated images help humans identify correct synset meanings or navigate taxonomies would strengthen the practical significance claims.

- **Closed-source model comparison**: Adding DALL-E 3 or Midjourney as baselines would contextualize how open-source models compare to production systems.

## Novel Insights

The finding that model rankings for taxonomy image generation differ from standard T2I benchmarks is significant—it suggests current T2I evaluation may not capture structured knowledge representation capabilities. The discrepancy between CLIP-based and preference-based rankings (SDXL-turbo vs. Playground/FLUX) reveals an important tension: models that excel at text-image alignment may not produce images that humans prefer. This aligns with findings in other domains where alignment metrics diverge from human judgment, and warrants deeper investigation into what each metric family actually captures for concept visualization tasks.

## Potentially Missed Related Work

- **Liao et al. (2024) "Text-to-Image Generation for Abstract Concepts"** (AAAI 2024) — Already cited by the paper; addresses abstract concept generation from a different angle.

- **ConceptBed (Patel et al., 2024)** — Cited in the paper; focuses on concept learning in diffusion models.

- **General T2I evaluation benchmarks** — The paper appropriately references GenAI Arena and ConceptBed; no major omissions identified.

## Suggestions

1. **Analyze performance by hierarchy depth**: Group concepts by distance from the WordNet root and report metrics per level. This directly validates the claim about abstraction handling.

2. **Generate multiple samples per concept**: Regenerate images with 5-10 different seeds and report mean and standard deviation for metrics to improve reliability.

3. **Implement GPT-4 position correction**: Use alternating positions or majority voting to mitigate the identified positional bias, rather than only acknowledging it.

4. **Clarify the "best model" question**: If different metrics favor different models, discuss what each metric measures and provide guidance on which to prioritize for taxonomy visualization use cases.

---

## QryPmx2MNh

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

This paper introduces the novel task of discovering learning-friendly orderings for chain-of-thought sequences in Transformers for arithmetic tasks. The authors propose a method combining loss profiling (training on mixed-order datasets and identifying orders with fast early-stage loss drops) with a two-stage hierarchical search to handle the factorial search space. Experiments on three newly designed order-sensitive arithmetic tasks and a multiplication task show the method can find optimal orders among billions of candidates and successfully recovers the known reverse-digit order for multiplication.

## Strengths

- **Novel problem formulation**: The paper formalizes an underexplored aspect of autoregressive reasoning—systematically discovering optimal output orderings rather than relying on heuristics or manual design. This opens a new research direction with clear motivation from prior work (e.g., Shen et al., 2023).

- **Strong validation via recovery of known result**: The method successfully recovers the reverse-digit (least-significant-first) order for multiplication previously reported by Shen et al. (2023), providing convincing validation on a task with established ground truth.

- **Creative task design for controlled evaluation**: The three order-sensitive tasks (RELU, SQUARE-19, INDEX) are carefully designed using non-injective recurrence relations to create tasks where forward order is uniquely optimal, enabling clean evaluation of the reordering method.

- **Efficient hierarchical search strategy**: The two-stage global/local approach enables searching over factorial spaces practically (up to 13! ≈ 6×10⁹ permutations with random initialization; up to L=40 with structured initialization), demonstrating computational tractability.

- **Loss profiling insight**: Leveraging the "easy-to-hard" learning dynamics of neural networks to identify learning-friendly orders through early-stage loss is a creative and principled approach.

## Weaknesses

- **Unexplained failure cases on harder task configurations**: Table 2 shows the method fails to recover the forward order for INDEX (d=4 and d=8), finding suboptimal permutations instead. The paper provides no analysis of why these failures occur or what distinguishes tasks where the method succeeds from those where it fails. Understanding failure modes is critical for assessing reliability.

- **No statistical significance testing**: All results appear to be from single runs without standard deviations or multiple random seeds. This is insufficient for assessing result reliability, particularly for Table 1 (success rates) and Table 2 (discovered orders).

- **No out-of-distribution generalization testing despite claims**: The abstract claims the method makes learning "generalizable to out-of-distribution samples," but all experiments test only in-distribution success rates. No experiments validate generalization to longer sequences or unseen values.

- **Limited scope to synthetic arithmetic tasks**: All experiments are on carefully designed arithmetic tasks with known optimal orders. The approach is not validated on any established reasoning benchmarks, algorithmic tasks, or natural language CoT settings, limiting assessment of broader applicability.

- **Weak theoretical justification for loss profiling mechanism**: The key assumption—that early-stage loss drops identify learning-friendly orders—is motivated by "easy-to-hard" learning dynamics observed for dataset samples, not token sequences. The paper provides no analysis of why this property transfers to sequence ordering.

- **Limited baseline comparisons**: The evolutionary strategy baseline in Appendix C achieves comparable results (100% for RELU L=10, 99.9% for SQUARE-19 L=10), yet the proposed method's advantages are not systematically analyzed. The main text under-emphasizes this comparison.

## Nice-to-Haves

- **Ablation studies for key hyperparameters**: The paper does not ablate the number of epochs E in loss profiling, depth K in global stage, or the exploration model size. These choices affect computational efficiency and should be validated.

- **Analysis of order transfer across model sizes**: The paper uses small models for search and large models for final training without verifying that discovered orders generalize across architectures.

## Novel Insights

The key insight—that sequence ordering fundamentally affects autoregressive learning difficulty, and that this difficulty can be quantified through early-stage loss dynamics—opens interesting perspectives on chain-of-thought design. The hierarchical decomposition is clever: recognizing that token-level permutations explode factorially, but block-level structure can be discovered first, mirrors how humans might approach ordering problems. The task design using non-injective functions to create controlled order sensitivity is methodologically clean and provides a useful testbed for future work on CoT optimization. The finding that attention maps are sparser for learning-friendly orders (Appendix A-B) hints at mechanistic explanations worth deeper investigation.

## Potentially Missed Related Work

No specific related work was identified in the search report. The authors may consider connecting to broader literature on curriculum learning, chain-of-thought prompting optimization, and neural architecture search methods that could inform permutation discovery approaches.

## Suggestions

1. **Add out-of-distribution generalization experiments**: Test whether discovered orders improve performance on longer sequences than seen during training, which would validate the "generalization" claim in the abstract.

2. **Analyze and report failure cases**: Provide explicit discussion of why INDEX (d=4, d=8) fails to recover forward orders and what this reveals about method limitations.

3. **Include statistical significance**: Report mean ± std across at least 3 random seeds for all quantitative results.

4. **Strengthen baseline comparison**: Either explain why the evolutionary strategy is computationally inferior or acknowledge it as a competitive alternative.

5. **Broaden task scope**: Include at least one non-arithmetic task (e.g., symbolic reasoning, algorithmic task) to demonstrate broader applicability beyond synthetic arithmetic domains designed for this paper.

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes AdaSVD, an adaptive SVD-based compression method for LLMs with two main contributions: (1) adaComp, which compensates for SVD truncation errors through alternating updates of singular matrices using Moore-Penrose pseudoinverse, and (2) adaCR, which assigns layer-specific compression ratios based on each layer's relative importance. Experiments across multiple LLM families demonstrate consistent perplexity improvements over prior SVD-based methods, particularly at higher compression ratios.

## Strengths
- **Strong empirical improvements**: AdaSVD consistently outperforms SOTA SVD-based methods across multiple LLM families (LLaMA2-7B, OPT-6.7B, Mistral-7B, Vicuna-7B). At 60% compression on LLaMA2-7B, AdaSVD achieves 50.33 perplexity on WikiText-2 versus 89.90 for SVD-LLM—a 44% relative improvement.
- **Comprehensive ablation studies**: Table 3 provides thorough analysis of each component (adaComp and adaCR), iteration counts, and minimum retention ratios, demonstrating the individual contributions of each technique.
- **Orthogonality to other compression methods**: Table 4 shows AdaSVD can be combined with GPTQ weight quantization, demonstrating practical composability with existing techniques.
- **Layer-wise importance insights**: Figure 4 provides useful empirical observations about varying layer importance across architectures, revealing that first and final layers tend to have higher importance.

## Weaknesses
- **No practical deployment metrics reported**: The paper motivates SVD compression for reducing "memory requirements" and "accelerating model inference," yet reports no actual measurements of inference latency, throughput, or memory footprint. For a compression paper, this is a critical omission that undermines claims of practical utility.
- **No statistical significance or error bars**: All results in Tables 1-4 are point estimates without variance measures. Given that calibration data is randomly sampled from WikiText-2, results may vary across runs—this should be quantified.
- **Limited theoretical grounding**: The adaComp alternating update scheme has no convergence analysis or theoretical guarantees. Similarly, adaCR's use of cosine similarity as the importance metric is stated as chosen "for simplicity" without justification or comparison to alternative metrics (gradient-based, Hessian-based, activation magnitude).
- **Calibration data sensitivity unexplored**: The method relies on 256 calibration samples, but the paper provides no analysis of how results vary with calibration data size, source domain, or selection strategy—a key practical concern for post-training compression.
- **Narrow baseline comparison**: Only SVD-based methods are compared. Readers cannot assess how SVD-based compression compares to quantization (GPTQ, AWQ) or structured pruning (SliceGPT) at comparable compression levels.

## Nice-to-Haves
- Quantitative VLM evaluation with standard metrics (BLEU, CIDEr) rather than only qualitative examples in Figure 5.
- Analysis of computational overhead for the adaComp alternating optimization process.
- Guidance on hyperparameter selection (iteration count, minimum retention ratio) based on model architecture or target compression ratio.

## Novel Insights
The observation that alternating updates with Moore-Penrose pseudoinverse provides stable error reduction (Figure 3a) is a meaningful engineering contribution—the comparison with naive matrix inverse updates shows clear numerical stability benefits. The finding that layer importance varies substantially (with first and final layers being most important) provides empirical grounding for non-uniform compression strategies, though the metric choice remains heuristic.

## Potentially Missed Related Work
- None identified from the provided inputs.

## Suggestions
1. **Report actual inference latency and memory measurements**: This is fundamental for any compression paper—report wall-clock time, throughput, and GPU memory footprint for compressed models compared to the original.
2. **Add comparison with quantization methods**: Include GPTQ and/or AWQ baselines at comparable parameter counts to contextualize SVD-based compression within the broader LLM compression landscape.
3. **Include standard deviations across multiple runs**: Run experiments with different random calibration samples and report mean ± std to demonstrate robustness.
4. **Provide convergence analysis or empirical study**: Analyze how many adaComp iterations are sufficient and whether the procedure converges consistently across different layers and compression ratios.

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

This paper investigates whether LLM-generated bibliographies can be distinguished from human-curated ones using citation graph structure and semantic embeddings. Using paired citation graphs for 10,000 focal papers (~275k references) from SciSciNet, the authors find that structural features alone achieve near-chance discrimination (~60% accuracy) between ground truth and GPT-generated references, while semantic embeddings via Random Forest (~83%) or Graph Neural Networks (~93%) achieve substantially higher accuracy—demonstrating that LLM bibliographies closely mimic human citation topology but leave detectable semantic fingerprints.

## Strengths

- **Large-scale, systematic dataset construction with thoughtful controls.** The paired design (10,000 focal papers with ground truth and LLM-generated references) provides a robust empirical foundation, and the field-matched random baseline that preserves out-degree and field distributions while breaking latent structure is a methodologically sound control for isolating what LLMs actually reproduce versus what any topic-matched sampling would produce.

- **Progressive methodology enabling clear attribution.** Moving from interpretable structural features (centrality, clustering) to aggregated embeddings to GNNs allows the clear conclusion that structure alone is insufficient—the structural analysis showing near-identical topology between LLM and human graphs is itself an important negative finding with implications for detection approaches.

- **Comprehensive robustness checks across generators and embeddings.** Testing multiple LLM families (GPT-4o, Claude Sonnet 4.5), multiple embedding models (OpenAI text-embedding-3-large, SPECTER2), and cross-model generalization experiments demonstrates that the semantic fingerprint finding is not artifact-specific to one model or encoder.

- **Dimensionality control experiment.** Replacing semantic embeddings with i.i.d. random vectors of matched dimensionality shows accuracy collapses to chance, ruling out the trivial explanation that higher-dimensional features alone drive separability.

## Weaknesses

- **Selection bias from excluding graphs where LLM produces no valid references.** The paper removes 779 graphs for GPT-4o and 89 for Claude where no valid references were found in SciSciNet. This exclusion may overestimate the structural realism of LLM bibliographies and limit real-world applicability, since in practice users encounter cases where LLMs produce entirely hallucinated references.

- **No analysis of what semantic dimensions drive detection.** While the paper demonstrates that embeddings enable discrimination (83-93% accuracy), it does not investigate which semantic factors explain the fingerprint—recency bias, venue prestige, topic drift, author network effects, or other patterns. Without this analysis, the "semantic fingerprint" remains uninterpreted, limiting actionable insights for mitigation and debiasing.

- **Scope limited to parametric knowledge generation.** The study explicitly excludes retrieval-augmented generation, which is increasingly the standard deployment mode for reference suggestion tools. The findings may not transfer to RAG-based systems that ground references in actual database retrieval.

- **Temporal violations noted but not systematically exploited.** The paper mentions that ~6% of GPT-generated references point to papers published after the focal paper—a clear hallucination signal—but does not analyze or exploit this feature as a baseline detection method, which could be a simpler alternative to GNNs.

## Nice-to-Haves

- Formal statistical tests comparing accuracy distributions across conditions would strengthen claims, though the tight standard deviations (~0.005) suggest differences are meaningful.

- Analysis of failure modes (which papers/fields are misclassified) and confusion matrices would better inform practical deployment considerations.

## Novel Insights

The central finding—that LLM bibliographies achieve near-perfect structural mimicry of human citation networks while retaining detectable semantic signatures—carries significant practical implications. Detection and debiasing efforts should focus on content signals (embedding distributions, topical patterns) rather than graph topology. The cross-model generalization results (training on GPT-4o, testing on Claude achieving 68-80% accuracy) suggest the semantic fingerprint is partially shared across model families, pointing toward systematic biases in how LLMs retrieve scholarly knowledge from parametric representations.

## Potentially Missed Related Work

No potentially missed related work was identified in this review round.

## Suggestions

- **Analyze semantic drivers of separability.** Use RF feature importance, probing classifiers, or attention analysis on the embedding dimensions to identify interpretable factors (recency, prestige, topic clusters) that differentiate LLM from human bibliographies—this would transform the paper from detection-focused to diagnostic.

- **Report hallucination prevalence and discuss selection bias implications.** Quantify what fraction of LLM-generated references were excluded due to non-existence in SciSciNet, and explicitly discuss how this may affect conclusions about structural realism.

- **Add a recency-matched control experiment.** Subsample ground truth references to match the LLM's year distribution to test whether detection persists when temporal patterns are controlled, isolating deeper semantic differences from recency bias.

---

## opU91paIvZ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary

This paper addresses the challenge of making chain-of-thought (CoT) reasoning in language models more monitorable by improving two key properties: faithfulness (reasoning honestly reflects decision factors) and conciseness (reasoning is sufficiently brief). The authors formalize CoT monitorability as a constrained optimization problem, analyze why naive RL approaches fail due to sparse gradient signals, and propose a prior-guided distillation approach: using an instruction-tuned model to transform raw CoT traces into faithful/concise versions, filtering for correctness, and training via supervised fine-tuning.

## Strengths

- **Insightful analysis of why naive RL fails.** The paper provides a compelling theoretical explanation—the gradient term L₁ for monitorability vanishes because f(z) ≈ 0 for most samples under the initial policy. Figure 2 empirically validates this analysis by showing that naive RL training produces no improvement in either faithfulness or conciseness.

- **Well-designed proof-of-concept experiment.** Figure 3 validates the core hypothesis that "monitorable traces are reward-compatible" by showing the base model can produce correct answers when conditioned on transformed traces. This is a thoughtful way to justify the algorithm design before implementation.

- **Meaningful empirical improvements.** The method achieves ~10 percentage point improvement in faithfulness (15%→25% on hint verbalization) and up to 60% reduction in reasoning length while maintaining ≥90% relative task accuracy. The trade-off between monitorability and performance is effectively addressed.

- **Clear problem formulation.** The constrained optimization formulation in Eq. 1 cleanly separates the monitorability objective from task performance constraints, providing a principled foundation for the approach.

## Weaknesses

- **Missing comparisons to prior methods.** The paper compares against naive RL (which fails) and the base model, but does not compare against existing approaches such as Arora & Zanette (2025) for conciseness, prompting-based approaches (e.g., instructing models to verbalize hints), or preference optimization methods. This makes it difficult to assess whether the proposed prior-guided distillation approach is actually superior to simpler alternatives.

- **No statistical significance testing.** Results throughout the paper report single values without error bars, confidence intervals, or significance tests. For example, the faithfulness improvement from ~15% to ~25% lacks variance measures, making it impossible to assess whether improvements are statistically meaningful.

- **Narrow operationalization of faithfulness.** The paper focuses exclusively on hint verbalization as the faithfulness metric. While this captures one important aspect, other forms of unfaithfulness—such as fabricated reasoning steps, omitted crucial reasoning, or post-hoc rationalization—are not addressed. A model could satisfy this metric while remaining unfaithful in other ways.

- **LLM-as-judge evaluation without independent validation.** Faithfulness is evaluated using Qwen 14B Instruct, which may share biases or failure modes with the prior model used to generate training data. Without human validation or an independent verification method, faithfulness claims rest on potentially circular assumptions.

## Nice-to-Haves

- **Ablation on prior model quality.** The method depends on Qwen 2.5-7B Instruct for generating transformed traces. Testing how performance degrades with weaker/smaller priors would help practitioners understand robustness requirements.

- **Per-hint-type breakdown.** The paper defines six hint types (sycophancy, consistency, visual pattern, metadata, grader hacking, unethical information) but aggregates results. A breakdown would reveal whether improvements generalize across all types.

- **Threshold sensitivity analysis for conciseness.** The token thresholds (125 for GSM8K, 950 for MATH500) are set without justification. Understanding how performance varies with different thresholds would aid practical deployment.

## Novel Insights

The paper's core theoretical insight is valuable: standard policy gradient methods fail for CoT monitorability because the monitorability signal f(z) is sparse under the initial policy—most samples have f(z) ≈ 0, causing the gradient term L₁ to vanish. This explains why models struggle to learn monitorable reasoning despite being capable of producing it when conditioned appropriately. The prior-guided transformation approach effectively converts a sparse reward problem into a dense supervised learning task by using an external model to reshape reasoning traces before training. This insight—that capability exists but cannot be accessed via standard RL—has broader implications for training models on properties that are initially rare but achievable.

## Potentially Missed Related Work

No systematic related work search was performed. The paper cites relevant prior work on CoT reasoning, monitorability, and faithfulness, though comparison with Arora & Zanette (2025) for conciseness methods would strengthen the empirical contribution.

## Suggestions

1. **Add comparisons to alternative approaches.** At minimum, compare against a prompting baseline (e.g., instructing the base model to "acknowledge any hints in your reasoning") to establish that the prior-guided distillation approach provides benefits beyond simpler alternatives.

2. **Report variance across multiple runs.** Include error bars or confidence intervals across multiple random seeds for all quantitative results to establish statistical significance.

3. **Validate faithfulness evaluation with human annotations.** Even on a small subset (e.g., 100-200 examples), human validation would strengthen confidence that the LLM-as-judge approach captures true faithfulness.

4. **Test cross-domain generalization.** Evaluate whether models trained for faithfulness on MMLU-Pro hints transfer to other domains or hint types not seen during training.

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

This paper introduces Visuo-Tactile World Models (VT-WM), the first multi-task world model integrating fingertip tactile sensing (Digit 360) with exocentric vision for robot manipulation. By grounding predictions in contact physics, VT-WM addresses hallucinations common in vision-only models—objects disappearing under occlusion or moving without applied forces—and demonstrates improved imagination quality and zero-shot planning on contact-rich real-robot tasks.

## Strengths

- **Clear problem identification with concrete motivation**: The paper correctly identifies that vision-only world models struggle with occlusion and ambiguous contact states, leading to physically implausible predictions. The examples (objects disappearing, teleporting, cloth moving without contact) are compelling and well-visualized in Figures 5 and 7.

- **Novel contribution**: To my knowledge, this is the first multi-task world model explicitly designed to integrate tactile and visual modalities for robot manipulation, filling a genuine gap in prior work on visual dynamics models.

- **Strong empirical validation on real robots**: The experiments demonstrate zero-shot transfer on a Franka Panda with Allegro hand across five contact-rich tasks (pushing, wiping, stacking, reach-and-push, button pressing), with meaningful improvements in success rates. The imagination metrics include proper statistical significance testing with paired t-tests.

- **Intuitive qualitative demonstrations**: Figures 5, 7, and the appendix effectively illustrate how V-WM produces physically inconsistent predictions (object disappearance, motion without contact) while VT-WM maintains coherent hand-object relationships.

- **Data efficiency experiment**: The 77% vs 22% comparison against behavioral cloning (ACT) with only 20 demonstrations of a new task (dish rack insertion) demonstrates practical value for real-world deployment where data is limited.

## Weaknesses

- **Insufficient statistical rigor for planning experiments**: The planning success rates are reported from only n=5 trials per task, which provides weak statistical grounding for claims about improvements. Confidence intervals or variance measures are not reported, making it difficult to assess the reliability of the 10-35 percentage point improvements.

- **Conflated comparison in data efficiency experiment**: VT-WM (multi-task pretrained) is compared against ACT (single-task, trained from scratch). The 77% vs 22% result does not isolate the contribution of world modeling from the benefits of multi-task pre-training. A multi-task BC baseline would enable fair attribution.

- **Missing ablation studies**: No ablations are provided on architectural choices (attention mechanisms, transformer depth), tactile encoder selection, temporal context windows, or multimodal fusion strategies. This makes it unclear which design choices are critical to the method's success.

- **No generalization testing beyond training distribution**: Experiments use held-out trajectories but within the same tasks and objects. Claims about capturing "physics of contact" would be strengthened by testing on novel objects with different shapes, sizes, or materials.

- **VT-WM failure modes unanalyzed**: The paper visualizes V-WM failures extensively but does not provide systematic analysis of where VT-WM fails. Understanding failure modes is essential for practical adoption.

- **Computational overhead unreported**: Adding tactile encoding and prediction increases model complexity, and CEM planning requires many autoregressive rollouts. Inference latency and computational cost are not discussed, leaving an open question about real-time feasibility.

## Nice-to-Haves

- Ablation on the sampling loss contribution versus teacher forcing alone
- Clarification on the tactile decoder used for visualizations (Sparsh-X is an encoder)
- Temporal alignment mechanism between vision (9 frames, 1.5s) and tactile (2 frames, 0.16s) streams
- Confidence intervals or standard errors for planning success rates

## Novel Insights

The key insight—that tactile sensing provides a grounding signal for contact physics that vision alone cannot capture—is convincingly demonstrated through the object permanence and causal compliance metrics. The paper makes a compelling case that the "blindness" of vision-only models to contact states is a fundamental limitation for manipulation, not merely a quantitative gap. The qualitative examples (e.g., Figure 7 showing V-WM hallucinating cloth motion when the hand hovers above it, while VT-WM correctly predicts stasis due to lack of tactile contact) provide memorable illustrations of this principle. The data efficiency results further suggest that contact-grounded world models can transfer learned dynamics to new tasks more efficiently than learning from scratch.

## Potentially Missed Related Work

- **Closed-loop world model planning approaches**: Methods like DayDreamer (Wu et al., 2022) and IRIS use world models for closed-loop control. The paper's open-loop execution could be contextualized against these approaches.

- **Multimodal robot learning with tactile sensing**: Prior work on tactile servoing and visuo-tactile learning for manipulation (e.g., Sutanto et al., 2019; Tian et al., 2019) may inform the multimodal integration approach.

- **Multi-task behavioral cloning baselines**: The data efficiency comparison would benefit from context on how multi-task BC policies perform on new tasks with limited demonstrations.

## Suggestions

- Report confidence intervals or run additional trials for the planning experiments to strengthen statistical claims.

- Add a multi-task BC baseline (trained on the same multi-task dataset as VT-WM) to isolate the contribution of world modeling versus multi-task pre-training in the data efficiency experiment.

- Include at least one ablation (e.g., comparing the proposed attention-based fusion to simple concatenation) to validate architectural choices.

- Test generalization to at least one novel object not seen during training to substantiate claims about learning general contact dynamics.

- Analyze and report VT-WM failure cases alongside V-WM failures to provide a balanced view of limitations.

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

This paper introduces Ambig-SWE, a benchmark for evaluating how LLM agents handle underspecified instructions in software engineering tasks. The authors create synthetically underspecified variants of SWE-Bench Verified issues and evaluate six models across three capacities: detecting missing information, asking targeted clarification questions, and leveraging interaction to improve task performance. Key findings show that models default to non-interactive behavior, struggle to distinguish well-specified from underspecified inputs, but can achieve up to 74% relative improvement in task completion when interaction is enabled.

## Strengths

- **Well-motivated problem with practical significance**: The paper addresses a real challenge in deploying coding agents—users often provide incomplete instructions, leading to misaligned solutions and potential safety risks. The three-capacity framework (detection, clarification, integration) provides a useful decomposition for targeted improvements.

- **Systematic benchmark design**: Creating controlled underspecified variants from SWE-Bench Verified enables causal measurement of interaction impact with ground truth specifications, addressing a gap in prior work that lacked paired complete specifications.

- **Comprehensive empirical coverage**: The evaluation spans six models (Claude Sonnet 4/3.5, Haiku 3.5, Qwen 3 Coder, Deepseek-v2, Llama 3.1 70B) with Wilcoxon signed-rank statistical tests across three experimental settings, providing actionable insights about model behavior.

- **Valuable behavioral insights**: The finding that Qwen 3 Coder exhibits rigid protocol-following (performance worsening with navigational information) while Claude models use exploration-first strategies is a substantive contribution for agent design. The analysis of different question-asking strategies (quantity, exploration efficiency, answerability) provides concrete guidance.

- **Good reproducibility**: Code, data, and full prompts are provided for independent replication.

## Weaknesses

- **Synthetic underspecification lacks human validation**: All underspecified issues are generated by GPT-4o without human verification that critical information is genuinely missing. The distributional analysis (Appendix §2.1) shows synthetic issues differ from natural ones—they target code snippets and error messages more aggressively, while natural underspecified issues contain more conversational fragments and external links. This may limit ecological validity claims about real-world underspecification patterns.

- **Turn allocation inconsistency confounds efficiency comparisons**: Claude Sonnet 4 and Qwen 3 Coder receive up to 100 interaction turns while other models receive 30. While justified for models with "greater reasoning capacity," this asymmetry affects direct comparability of efficiency metrics (steps per solution) and costs across models.

- **No human evaluation of question quality**: Question quality relies entirely on cosine distance metrics and GPT-4o-as-judge scores. Whether questions are genuinely helpful from a real user's perspective remains unvalidated, and GPT-4o serving as both user proxy and quality judge introduces potential systematic bias.

- **Causality unclear in navigational information analysis**: Table 1 shows models that request navigational information achieve higher resolve rates, but the paper acknowledges this may reflect model capability rather than information benefit. The counterintuitive finding that Qwen 3 Coder's performance *decreases* with navigational information (55.43% → 52.38%) suggests the relationship is more complex than presented.

- **Detection evaluation limited to early turns**: Measuring underspecificity detection only in the first 3 turns may underestimate models that recognize missing information after initial exploration. While acknowledged, this limitation affects conclusions about which models are better at detection.

## Nice-to-Haves

- **Cost-efficiency analysis**: Reporting average tokens, API calls, or computational cost per setting would help practitioners weigh tradeoffs between interactive and non-interactive approaches.
- **Analysis of failure cases where interaction doesn't help**: The paper reports aggregate improvements but does not analyze cases where models receive correct information yet still fail—valuable for understanding where information integration breaks down.
- **Simple baseline comparisons**: Comparing against naive strategies like "always ask 3 questions" would establish whether sophisticated detection/questioning matters or if gains come simply from having any interaction opportunity.

## Novel Insights

The paper provides two particularly novel observations. First, the disconnect between information extraction and integration: Claude Sonnet 3.5 and Haiku extract similar information (cosine distance 0.136 vs 0.135) yet achieve vastly different resolve rates (39.6% vs 26.8%), demonstrating that how models integrate information matters as much as what they extract. Second, the finding that higher capability models (Claude Sonnet 4, Qwen 3 Coder) derive smaller relative benefits from navigational information suggests improved code localization abilities reduce dependence on user-provided file paths—a promising trend for reducing user burden as models improve.

## Potentially Missed Related Work

- None identified beyond the paper's existing related work section.

## Suggestions

- Conduct human validation on a sample of synthetically underspecified issues to verify that critical information is genuinely missing and that underspecification patterns resemble real-world cases.
- Include qualitative analysis of failure cases where interaction succeeds (correct information provided) but task completion still fails, to identify integration bottlenecks.
- Consider evaluating detection accuracy beyond the first 3 turns, perhaps at key decision points in the solution trajectory, to capture late-stage recognition of missing information.

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
LoRA-Mixer introduces a mixture-of-experts framework that applies task-specific LoRA experts serially to linear projection layers within attention/SSM modules, rather than using parallel branches or FFN replacement. The authors propose a Routing Specialization Loss (RSL) that combines load balancing with entropy regularization to enable input-aware routing while maintaining balanced expert utilization. Experiments across 15 benchmarks on three base models demonstrate consistent improvements with 48% of the trainable parameters compared to baselines.

## Strengths
- **Novel architectural design**: Applying LoRA experts serially to projection matrices enables direct influence on attention computation without disrupting model architecture, distinguishing it from prior FFN-focused or parallel-branch approaches like MixLoRA and MoLE.
- **Strong theoretical grounding**: The appendix provides convergence analysis (Theorem 1) and generalization bounds (Theorem 2) demonstrating that entropy regularization supplies strong convexity on the product simplex, yielding uniform stability for ERM solutions.
- **Comprehensive empirical evaluation**: Testing across 15 benchmarks covering medical QA, mathematics, coding, and NLP tasks on three architectures (LLaMA3-8B, Mistral-7B, Falcon-Mamba-7B) demonstrates broad applicability. Results show consistent improvements over MoLE, MixLoRA, LoRAHub, and routing-optimized baselines (GMoE, AESL, DsMoE).
- **Parameter and data efficiency**: LoRA-Mixer achieves strong performance using only 48% of MixLoRA's trainable parameters, and the RSL loss enables effective routing training with as little as 2K samples (Table 9).
- **Architecture-agnostic design**: The method works on both Transformer and SSM architectures (Falcon-Mamba), demonstrated through cross-model transfer experiments where routers trained on Mistral-7B transfer to LLaMA3-8B.

## Weaknesses
- **Missing controlled ablations**: The paper compares LoRA-Mixer (new architecture + RSL loss) against baselines using different architectures and loss functions simultaneously. Without isolating the projection-layer placement from the RSL contribution, the individual benefits of each innovation remain unclear. An ablation testing LoRA-Mixer architecture with standard auxiliary loss would substantiate the architectural claims.
- **No statistical significance reporting**: All results report only mean values from three runs without standard deviations or confidence intervals. Given improvements of 1-3% on many tasks, variance measures are essential to establish meaningful differences.
- **RSL performs worse at 4K training data**: Table 9 shows RSL achieving 78.77% vs. auxiliary loss achieving 79.14% at 4K samples—RSL underperforms here. While Appendix A.16 offers an explanation about exploration dynamics, this anomaly undermines confidence in RSL's reliability across different data regimes.
- **Limited expert scalability analysis**: Only 6 experts are tested throughout. Claims about scaling to "large-scale modular language models" lack empirical support for scenarios with more experts.
- **Cross-domain evaluation limited**: The Math-Medical and Math-Coding experiments (Table 10) use only 200 examples each, insufficient for robust cross-domain generalization claims.

## Nice-to-Haves
- Layer-wise routing distribution analysis to verify whether different transformer layers learn meaningful specialization patterns
- Token-level routing heatmaps on representative inputs to demonstrate interpretable expert selection
- Expert orthogonality/diversity quantification to justify that multiple experts learn complementary features

## Novel Insights
The information-theoretic perspective on routing—viewing the router as an information bottleneck where entropy regularization preserves semantic distinctions between tokens—is a conceptually fresh contribution. The theoretical analysis shows that standard auxiliary losses have uniform distributions as unique minimizers (Appendix A.17), formally proving they inherently bias toward equal activation and suppress specialization. The RSL loss addresses this by adding token-level entropy gradients that amplify input-aware variance, with Theorem 1 proving convergence guarantees under this modified objective.

## Potentially Missed Related Work
- **Model merging approaches** (e.g., Task Arithmetic, Ties-Merging) — These offer alternative methods for combining multiple fine-tuned models without routing mechanisms and could serve as additional baselines for multi-task composition efficiency.

## Suggestions
- Add an ablation study comparing LoRA-Mixer architecture with standard auxiliary loss to isolate the architectural contribution from RSL's benefit.
- Report standard deviations or confidence intervals across the three experimental runs.
- Provide clearer explanation for the 4K data regime where RSL underperforms, ideally with additional experiments or analysis beyond the current speculative explanation.
- Expand experiments with more than 6 experts to validate scalability claims, and increase cross-domain evaluation beyond 200-example datasets.

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary

VINCIE proposes learning in-context image editing directly from video data, bypassing the need for task-specific paired image datasets. The authors construct interleaved multimodal sequences from videos using VLM annotations and segmentation models, train a diffusion transformer with three proxy tasks (next-image prediction, current/next segmentation prediction), and introduce MSE-Bench for evaluating multi-turn editing. Results demonstrate competitive performance on MagicBrush and MSE-Bench, with emergent capabilities in multi-concept composition and story generation.

## Strengths

- **Novel problem formulation**: Learning in-context image editing from native video data addresses the scalability bottleneck of curated paired editing datasets. The insight that videos naturally encode sequential visual transformations relevant to editing is compelling and well-motivated.

- **Scalability empirically demonstrated**: Figure 5 shows clear log-linear scaling of 5-turn editing success rate (5%→22%) when scaling training data from 0.25M to 10M sessions, validating the core premise that video data enables scalable training.

- **Comprehensive evaluation**: The paper compares against 15+ baselines including proprietary models (GPT-4o, Gemini), evaluates on two benchmarks (MagicBrush, MSE-Bench), and includes human evaluation with correlation analysis (Table 7: Pearson r=0.4858 between GPT-4o and human scores).

- **Well-designed proxy tasks**: The three training objectives (NIP, CSP, NSP) are thoughtfully motivated. Table 3 shows segmentation prediction tasks improve consistency (DINO scores increase from 0.847 to 0.890) and success rates (Turn-5: 11%→17%), validating the design choices.

- **Emergent capabilities demonstrated**: Figures 19-24 show zero-shot abilities in multi-concept composition, story generation, and chain-of-editing without explicit training—suggesting the learned representations generalize meaningfully.

## Weaknesses

- **Proprietary base model dependency**: The model is initialized from an "in-house MM-DiT (3B and 7B), pre-trained on text-to-video tasks" (Section 4.1). Without releasing these weights, the core method cannot be fully reproduced, limiting community validation and extension. This is a significant reproducibility concern.

- **Substantial gap between automatic and human evaluation**: Table 2 reports 25% Turn-5 success (GPT-4o evaluation) while Table 6 reports 7% Turn-5 success (human evaluation) for the same "Ours*" model. This ~18 percentage point discrepancy warrants explanation—it suggests GPT-4o may be systematically more lenient or that evaluation protocols differ.

- **Missing comparisons to closely related video-based methods**: UES (Chen et al., 2024a) and RealGeneral (Lin et al., 2025) are discussed in related work as methods that also leverage video frames for image generation/editing, but are not included in experimental comparisons. This limits positioning against the most relevant prior work.

- **No statistical significance measures**: Tables 1-2 and Table 6 report single numbers without error bars, confidence intervals, or standard deviations. Given the small MSE-Bench size (100 examples), variance estimates are important for confidence in the reported improvements.

- **VLM annotation noise acknowledged but impact unclear**: Table 8 reports 75.14% accuracy and 69.06% recall for VLM annotations. While the paper notes this is acceptable for large-scale pre-training, the impact of this noise on downstream editing performance is not quantified.

## Nice-to-Haves

- **Single-turn editing benchmark results**: The paper focuses on multi-turn editing but does not report performance on standard single-turn editing benchmarks. Readers would benefit from understanding baseline editing capability before assessing multi-turn improvements.

- **Ablation with text-to-image model initialization**: The model inherits video priors from the video foundation model. Initializing from a text-to-image model instead would help isolate what the video training data contributes versus what comes from the foundation model's pre-training.

## Novel Insights

The paper reveals an interesting finding: video-only pre-training can match or exceed pairwise editing data performance for multi-turn tasks (Table 5: sequence data achieves 22% Turn-5 success vs 1% for pairwise data). This suggests temporal coherence in videos provides stronger supervision for contextual reasoning than isolated before/after pairs. The chain-of-editing strategy—predicting segmentation masks before generating images—mirrors "chain-of-thought" reasoning in language models and provides a mechanism for more precise region-aware editing.

## Potentially Missed Related Work

- **UES (Chen et al., 2024a) and RealGeneral (Lin et al., 2025)**: Both are discussed in related work as methods leveraging video temporal structure for image tasks, but excluded from experiments. Including comparisons would strengthen positioning against similar video-based approaches.

## Suggestions

1. Release the in-house MM-DiT base model weights or provide experiments with open-source video foundation models (e.g., CogVideoX, HunyuanVideo) to enable reproducibility.

2. Explain the large discrepancy between GPT-4o evaluation (Table 2) and human evaluation (Table 6) for the same model—clarify whether evaluation protocols differ or GPT-4o is systematically more lenient.

3. Add confidence intervals or standard deviations to key metrics, especially for MSE-Bench where the sample size is only 100 examples.

4. Include comparisons with UES and RealGeneral in experiments, or justify their exclusion if comparison is not feasible.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
GUI-Spotlight introduces an iterative visual grounding approach for GUI agents that uses three specialized tools (crop, extract, find_color) to progressively narrow focus on target elements. The model is trained via a three-stage pipeline (SFT warmup, RL with modified GSPO, high-resolution refinement) and achieves 52.8% accuracy on ScreenSpot-Pro using only 18.5K training samples, outperforming baselines trained on millions of samples.

## Strengths
- **Strong data efficiency**: Achieving competitive accuracy with 18.5K samples versus millions for baselines (V2P-7B uses 9.6M, GTA-1-7B uses 1.56M) is a compelling result demonstrated across three benchmarks (ScreenSpot-Pro, OSWorld-G, UI-Vision).
- **Novel iterative spotlighting mechanism**: The approach of coordinating multiple tools to progressively refine focus mirrors human visual search and is conceptually well-motivated. The comparison against training-free iterative inference (Figure 5) validates that the learned strategy provides genuine benefit.
- **Comprehensive RL methodology documentation**: The paper documents algorithmic explorations (Figure 3), reward design ablations (Figure 4), and negative results—valuable transparency for future research on agentic visual grounding.
- **Cross-backbone generalization**: Demonstrated improvements from both UI-specialized (UI-TARS-1.5-7B) and general-purpose (Qwen2.5-VL-7B) backbones, showing the method transfers beyond UI-specific models.

## Weaknesses
- **Missing inference cost analysis**: The iterative multi-turn approach requires multiple forward passes per query, but no wall-clock time, iteration count statistics, or computational cost comparison is provided. For real-world GUI agents, this overhead could be prohibitive—understanding the accuracy-latency trade-off is essential.
- **No systematic tool ablation**: The three tools (crop, extract, find_color) are introduced together without experiments isolating individual contributions. Without this analysis, it remains unclear whether all tools are necessary or whether simpler approaches would achieve similar results.
- **Data efficiency claim lacks fair comparison**: The model trains on 18.5K curated high-quality samples, but baselines use larger, potentially noisier datasets. Without training baseline models on the same filtered data, it's unclear whether gains come from the methodological innovation or from more effective use of higher-quality data.
- **Overclaiming on UI-Vision results**: The abstract states GUI-Spotlight "substantially outperform[s] comparable 7B baselines" on UI-Vision with 23.4%, but Table 4 shows UI-Venus-Ground-7B achieves 26.5%. The claim should accurately reflect that GUI-Spotlight improves over its base models but does not surpass all 7B baselines.
- **Missing failure analysis**: The inference pipeline returns `None` after `T_max` iterations, but failure rates, failure modes (which UI elements or layouts cause problems), and iteration statistics are not reported.

## Nice-to-Haves
- **Iteration depth statistics**: Report average number of tool calls per query, distribution of iteration counts, and correlation between iterations and accuracy to substantiate the "iterative refinement" benefit.
- **Tool usage analysis**: Quantify how often each tool is invoked, successful tool sequences, and whether the model learns meaningful tool selection versus defaulting to one tool.

## Novel Insights
The modified GSPO objective with tool-filtered positive cross-entropy loss (J'(θ)) addresses a practical training stability problem in multi-turn tool-using agents. The observation that vanilla GRPO/GSPO collapses around 300 steps due to format degradation, and that this can be prevented by reinforcing format-correct tool calls during training, is a useful empirical contribution that others training agentic models will likely encounter. The finding that moderately increasing Extract reward weight relative to Crop reward improves accuracy (because Extract requires only approximate localization while Crop demands precise coordinates) provides actionable guidance for reward design in similar multi-tool RL settings.

## Potentially Missed Related Work
- None identified in this review cycle.

## Suggestions
- Add inference efficiency analysis: report average tool calls per query, wall-clock time comparison with single-pass baselines, and computational cost (FLOPs) to enable practical deployment assessment.
- Include systematic tool ablations: experiments with tool subsets (crop only, crop+extract without find_color, etc.) to clarify each tool's contribution.
- Clarify results presentation: ensure tables clearly show GUI-Spotlight's performance relative to baselines, and ensure claims accurately reflect comparative performance.

---

## ppXAVexrAM

- GT: Reject (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
ARSS introduces a decoder-only autoregressive transformer for novel view synthesis from a single image, using a video tokenizer for temporal consistency, a camera autoencoder to encode Plücker raymaps as positional guidance tokens, and a spatial permutation strategy that preserves temporal causality while enabling bi-directional spatial context. The method achieves competitive results with diffusion-based baselines on RealEstate10K, ACID, and DL3DV benchmarks.

## Strengths
- **Novel problem formulation**: The paper identifies a genuine gap—existing diffusion-based NVS methods generate views jointly, making strictly causal, incremental generation along camera trajectories difficult. Applying decoder-only autoregressive models to this task is a well-motivated contribution.

- **Thoughtful technical design**: The three-module architecture addresses key challenges: video tokenization preserves temporal consistency (validated by 62% FVD improvement in Table 3), the camera autoencoder provides explicit 3D positional guidance with geometrically-constrained loss (Eq. 5), and spatial-only permutation maintains temporal causality while exploiting bi-directional spatial context.

- **Strong ablation studies**: The permutation strategy ablation (Figure 7, Table 2) clearly demonstrates the benefit of spatial-only permutation over raster order and full permutation. The tokenizer comparison (Table 3) quantifies the importance of video tokenization.

- **Error accumulation analysis**: Figure 6 provides useful analysis showing ARSS maintains quality over longer trajectories, supporting the claim that causal generation reduces accumulated errors compared to baselines.

- **Zero-shot generalization**: Experiments on DL3DV (not seen during training) and AI-generated images (Figure 5) demonstrate robust generalization.

## Weaknesses
- **Missing statistical reporting**: Table 1 reports single values without standard deviations or confidence intervals, making it difficult to assess whether observed differences are statistically significant. This is below standards for empirical work.

- **Incomplete ablations**: The camera autoencoder is presented as a core contribution yet is never ablated—no comparison to simpler alternatives (e.g., direct Plücker embeddings) is provided. Classifier-free guidance (mentioned in Appendix A.2) is also not ablated.

- **Technical clarification needed**: The paper uses VidTok with Finite Scalar Quantization (FSQ) which produces continuous latents, but Eq. 3 assumes discrete tokens for cross-entropy loss. The conversion process between continuous FSQ latents and discrete tokens for AR training is not explained.

- **No efficiency analysis**: A claimed advantage of AR models over diffusion is efficiency for incremental generation, but no training time, inference time, or FLOPs comparisons are provided. Sequential token generation may be slower than parallel diffusion denoising—this tradeoff should be quantified.

- **Resolution limitation**: All experiments use 256×256 resolution while modern diffusion-based NVS methods often operate at higher resolutions. This limits both practical applicability and fair comparison with baselines that may have been trained/evaluated at higher resolutions.

- **Incomplete reproducibility details**: Loss weights λ₁–λ₄ in Eq. 5 are not specified, and the Plücker raymap construction from camera parameters is mentioned but not fully detailed.

## Nice-to-Haves
- Failure case visualization showing limitations with large viewpoint changes or complex occlusions would help readers understand practical boundaries of the approach.
- Experiments on longer trajectories (beyond 17 frames) would better validate the causal generation advantage claimed for world modeling applications.

## Novel Insights
The spatial-permutation-while-preserving-temporal-order strategy is an elegant solution to the tension between autoregressive models' uni-directional nature and images' bi-directional spatial context. By randomly permuting tokens within each frame while maintaining strict temporal ordering across frames, the model learns to handle arbitrary spatial orderings during training while ensuring that distant views are always generated conditioned on closer views. This insight—that the key inductive bias for view sequences is temporal causality, not spatial raster ordering—could influence future work beyond this specific application.

## Potentially Missed Related Work
None identified in this review cycle.

## Suggestions
1. Add standard deviations or confidence intervals to all quantitative results in Table 1.
2. Provide ablations for the camera autoencoder component and classifier-free guidance.
3. Clarify the tokenization pipeline: how are continuous FSQ latents converted to discrete tokens for cross-entropy training?
4. Include timing and computational cost comparisons with diffusion baselines to substantiate efficiency claims.
5. Specify all loss hyperparameters (λ₁–λ₄) in the paper body or implementation details.
6. Temper claims about "out-performing" state-of-the-art methods given mixed quantitative results—acknowledge where SEVA achieves competitive or better metrics alongside ARSS's advantages.

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Final Review

## Summary

SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking. Rather than parameterizing ligands via torsional angles, the method decomposes ligands into rigid-body fragments and learns to reassemble them within the binding pocket. The approach achieves 79.9% Top-1 success rate (RMSD < 2Å & PB-valid) on the PoseBusters benchmark, surpassing prior deep learning methods and matching classical physics-based docking under comparable evaluation conditions.

## Strengths

- **Strong theoretical justification**: Theorem 1 formally demonstrates that torsional parameterizations induce non-product measures in Cartesian space due to geometric coupling, while fragment-based SE(3) parameterizations yield factorized product measures. This provides principled motivation for the approach rather than relying solely on empirical validation.

- **Substantial empirical improvements**: The reported 79.9% Top-1 (RMSD < 2Å & PB-valid) on PoseBusters significantly outperforms prior DL methods (DiffDock at 12.7%, others at 12.7-32.8%) and matches classical docking (Vina at 80.5%) under the intended train-test split. The comparison uses identical training data splits for fairness.

- **Practical efficiency**: The method runs 50× faster than AlphaFold3 at inference, uses only ~19k training complexes, and requires no separate confidence model or post-hoc energy minimization—addressing key practical concerns about DL docking methods.

- **Rigorous ablations**: Table 1 demonstrates meaningful contributions from each component: triangulation conditioning (+8.6% absolute improvement), fragmentation merging (+6.2%), and protein-ligand interactions (+1.3%). The co-factor analysis (Table 2) provides insight into failure modes.

- **Well-designed architecture**: The SO(3)-equivariant prediction head (Theorem 2) correctly handles gauge ambiguity from arbitrary local coordinate frame orientations for fragments—a subtle technical issue properly addressed.

## Weaknesses

- **No confidence intervals or statistical significance tests**: Results are reported as point estimates (79.9%) without uncertainty quantification. Given 308 test complexes, confidence intervals are easily computable and necessary to substantiate claims of "surpassing classical physics-based docking."

- **Missing controlled ablation against torsional diffusion**: The central theoretical claim is that fragment-based SE(3) diffusion avoids torsional entanglement. However, there is no experiment comparing torsional parametrization using the same EquiformerV2 backbone and training pipeline. Comparing to DiffDock (different architecture, training setup) conflates multiple factors.

- **Chirality preservation not quantified**: The paper claims generation of "chemically plausible" poses and acknowledges chirality challenges in limitations, but never reports what fraction of generated poses correctly preserve stereochemistry. This directly impacts the PB-validity claims.

- **Limited evaluation scope—re-docking only**: All experiments use holo protein structures with known binding pockets. Cross-docking and apo-docking scenarios, which are practically important, are deferred to future work. This limits claims about practical drug discovery utility.

- **Incomplete comparison with recent baselines**: Methods like SurfDock, Uni-Mol Docking v2, and PoseX (cited but not compared) should be included for completeness, even if trained on different splits.

- **No detailed failure mode analysis**: Table 2 shows co-factor effects but doesn't characterize what molecular properties drive the remaining ~20% failures: number of rotatable bonds, molecular weight, pocket geometry, or fragment connectivity patterns.

## Nice-to-Haves

- Cross-docking experiments using apo structures would substantially strengthen claims about practical applicability.

- Training dynamics comparison (loss curves, convergence speed) between fragment and torsional parametrizations with controlled architecture would empirically validate the theoretical contribution.

- Breakdown of PB-validity failures by type (chirality inversions, bond geometry violations, steric clashes) would clarify whether inductive biases achieve their intended effect.

## Novel Insights

The key conceptual insight is that torsional diffusion models face fundamental geometric challenges: local torsional changes produce non-local Cartesian displacements, creating entangled implicit dynamics and gauge ambiguity. Fragment-based SE(3) diffusion avoids these issues by operating in a product space where independent noise on fragments remains independent in Cartesian space. The soft triangulation constraints provide geometric priors that reduce degrees of freedom while preserving dihedral freedom—a clean design that yields both theoretical coherence and empirical performance without requiring post-processing corrections.

## Potentially Missed Related Work

No specific missed related work identified beyond those already cited. The paper would benefit from including comparisons with SurfDock (Cao et al., 2025), Uni-Mol Docking v2 (Alcaide et al., 2025), and PoseX (Jiang et al., 2025) on the PoseBusters split if feasible.

## Suggestions

- Report 95% confidence intervals for all success rate metrics to enable statistical comparison with baselines.
- Add a controlled ablation implementing torsional diffusion with the same EquiformerV2 backbone to isolate the contribution of the fragmentation scheme.
- Quantify stereochemistry preservation rate across the test set to substantiate "chemically plausible" claims.
- Include at least one cross-docking experiment to demonstrate practical utility beyond the re-docking setting.

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
The paper proposes a compositional meta-learning framework that frames test-task acquisition as probabilistic inference rather than parameter updates. The model learns reusable computational modules (RNNs) and their transition statistics (via a gating RNN) from training tasks, then uses particle filtering to infer module sequences for new tasks. Experiments on synthetic rule learning and motor learning tasks demonstrate ground-truth component recovery and one-shot inference, even under sparse feedback conditions.

## Strengths
- **Conceptual novelty**: The approach of solving new tasks through probabilistic inference in a learned generative model—explicitly avoiding parameter updates—offers a genuinely different paradigm from gradient-based meta-learning methods like MAML.
- **Principled separation of concerns**: The architecture cleanly separates within-module dynamics ("syllables") from between-module dynamics ("grammar"), enabling the gating network to learn task statistics independently of module content.
- **Strong results on sparse feedback**: The particle filtering approach naturally maintains multiple hypotheses during inference, enabling task acquisition when feedback is only intermittently available—a compelling demonstration (Figures 2e, 2f, 4e).
- **Appropriate baseline comparisons**: Figure 3 provides systematic comparisons to RNN controls, MAML, MLDG, and ablations without gating, showing clear benefits of the full architecture.

## Weaknesses
- **Limited empirical scope**: Both tasks (6D vector shifts, 2D motor trajectories) are synthetic and low-dimensional. While this enables ground-truth verification, the paper provides no evaluation on standard meta-learning benchmarks or realistic domains, making it difficult to assess practical significance.
- **No comparison to in-context learning methods**: The paper emphasizes solving tasks "without parameter updates" but only compares to gradient-based meta-learning. In-context learning methods (transformers, memory-augmented networks) that also avoid parameter updates should be included as baselines.
- **Missing comparison to closely related work**: The paper identifies Alet et al. (2019) and Hummos et al. (2024) as "most similar in spirit" but provides no empirical comparison, leaving the claimed advantages of probabilistic inference over simulated annealing or latent embedding optimization unsubstantiated.
- **Fixed number of modules**: The architecture requires specifying the number of modules N a priori. While Figure A1 shows graceful degradation under mismatch, this remains a practical limitation for open-ended learning that the authors acknowledge but do not resolve.
- **Training instability under-characterized**: The appendix acknowledges that "training the gating and module parameters simultaneously is prone to local minima and instability," but the paper provides no quantitative analysis of failure rates, convergence sensitivity, or the reproducibility of results across seeds.
- **Missing statistical and computational analysis**: Figures show individual seed traces without error bars, confidence intervals, or significance tests. No analysis of computational cost (particle filtering scales with K×T) or scalability to higher dimensions is provided.

## Nice-to-Haves
- **Multi-shot experiments**: The paper focuses exclusively on single-episode inference; demonstrating how inference improves with additional episodes would showcase the framework more completely.
- **Module count sensitivity quantification**: A quantitative analysis of how performance degrades with module count mismatch would help practitioners choose N.
- **Failure case analysis**: Understanding when inference fails (beyond OoD detection) would strengthen practical applicability claims.

## Novel Insights
The key insight—that compositional meta-learning can be framed as probabilistic inference in a learned generative model rather than gradient-based parameter adaptation—reconceptualizes rapid task acquisition as constrained hypothesis testing. The gating network's ability to constrain module sequences during sparse feedback (hypotheses branch only at valid transition points) provides a principled explanation for why structured knowledge enables faster learning: it reduces the effective hypothesis space. This perspective connects meta-learning to classical probabilistic AI methods while leveraging modern neural network expressivity.

## Potentially Missed Related Work
- None identified through automated search. However, the authors should consider empirically comparing to **Alet et al. (2019)** (modular meta-learning via simulated annealing) and **Hummos et al. (2024)** (gradient-based inference of task embeddings), which the paper identifies as closest in spirit.

## Suggestions
- **Add comparison to in-context learning baselines** (e.g., transformer or memory-augmented networks) to validate the claim of efficient task acquisition without parameter updates.
- **Include at least one standard meta-learning benchmark** (e.g., Mini-ImageNet, Omniglot, or Meta-World tasks) to enable comparison with the broader field.
- **Report training statistics**: Include failure rates across seeds, convergence curves, and computational cost (wall-clock time) for both training and inference.

---

## wgGJE6Z1B3

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
This paper proposes "flatness" (cosine similarity between a token's predictive distribution and the uniform distribution) as a metric for identifying valuable training samples for speculative decoding draft models. The key insight is that tokens with flatter target distributions contribute more to acceptance rate improvement per training step. The authors propose SFDD, a data selection method that achieves over 2× training speedup at 50% data retention while maintaining inference speedup within 4% of the full-data baseline.

## Strengths
- **Novel data-centric perspective**: The paper takes a fresh approach to SD training efficiency by focusing on data selection rather than loss function modification. This complements existing work on L1-norm objectives and is genuinely new to the SD literature.
- **Solid theoretical foundation**: Section 3.2 provides rigorous KL-constrained optimization analysis using Gaussian distributions, with proper derivation in Appendix A. The asymptotic extension in Appendix B justifies the cosine similarity proxy for vocabulary-level distributions.
- **Empirical validation aligns with theory**: Figure 2 convincingly shows that high-flatness tokens exhibit larger ΔL1 reductions and greater draft distribution changes during training, supporting the theoretical predictions.
- **Significant practical efficiency gains**: At 50% retain ratio, the method achieves 2.02× training speedup with less than 4% inference speedup degradation (Tables 1-2), representing meaningful practical value.
- **Comprehensive baseline comparisons**: The paper compares against six alternative importance metrics (entropy, top-1 probability, margin, energy score, PPL, random), demonstrating consistent superiority across datasets.

## Weaknesses
- **No downstream task quality evaluation**: The paper reports only inference speedup and acceptance length metrics, without evaluating whether the filtered training affects actual generation quality (e.g., GSM8K accuracy, summarization ROUGE scores). This leaves open whether efficiency gains come at an unmeasured quality cost.
- **Limited SD framework diversity**: All main experiments use only the EAGLE-2 framework. Appendix G.1 adds Vicuna-7B results, but whether flatness-based selection generalizes to other draft architectures (Medusa, LayerSkip, PASS) remains unverified.
- **Lack of statistical significance measures**: Results are reported from single runs without error bars or confidence intervals. Given training variability, the reported differences between methods (e.g., SFDD vs. Top-1 Probability) need statistical validation.
- **No semantic analysis of filtered content**: The paper provides no analysis of what linguistic or semantic properties characterize high-flatness vs. low-flatness tokens/samples, leaving practitioners without insight into what the method actually prioritizes.

## Nice-to-Haves
- **Qualitative examples**: Showing sample sequences with their flatness scores would help practitioners understand what content is being prioritized or filtered.
- **Temperature sensitivity analysis**: The main experiments use temperature 1.0; while Appendix C shows temperature 0 results, analysis of how temperature affects the flatness metric itself would be valuable.

## Novel Insights
The paper's central insight—that acceptance rate improvement per training step correlates with target distribution flatness rather than with standard uncertainty measures—is genuinely novel. The theoretical argument that flat distributions yield larger L1-norm reductions per KL-constrained update (because flat distributions lack sharp peaks that cause large pointwise differences) is elegant and well-supported by the empirical validation in Figure 2. The finding that cosine similarity to uniform outperforms entropy for identifying low-value tokens (Figure 2d) suggests that the shape of distribution matters beyond just uncertainty magnitude—high-entropy tokens with a dominant peak may still contribute less than genuinely uniform tokens.

## Potentially Missed Related Work
- None identified (related work search was not performed for this review).

## Suggestions
- Add evaluation of downstream task quality metrics (e.g., GSM8K accuracy, summarization quality) to verify that efficiency gains don't compromise generation quality.
- Run multiple seeds and report confidence intervals or error bars to establish statistical significance of the claimed improvements over baseline metrics.
- Test on additional SD frameworks beyond EAGLE (e.g., Medusa, LayerSkip) to demonstrate broader applicability.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary
This paper presents a streaming framework for real-time neural stimulation design that targets latent neural dynamics. The authors develop: (1) a novel streaming jPCA algorithm (sjPCA) for identifying rotational dynamics in real time, (2) a kernel regression model for learning stimulus-response mappings that accounts for state-dependence and temporal non-stationarity, and (3) an optimization framework for designing high-dimensional stimulation patterns under sparsity and non-negativity constraints to achieve desired perturbations in the latent space.

## Strengths
- **Addresses an important practical gap**: Real-time stimulation design for neural dynamics is crucial for closed-loop neuroscience experiments and brain-machine interfaces, yet existing methods are limited. The paper tackles a genuine computational challenge—selecting from combinatorially many stimulation patterns while tracking time-varying neural dynamics.

- **Comprehensive and well-integrated framework**: The method integrates streaming latent space construction (proSVD, sjPCA, mmICA), dynamical modeling (KF, VJF, Bubblewrap), stimulus-response learning, and constrained optimization into a coherent pipeline. The comparison across multiple latent space methods and dynamical models provides practical guidance for experimental deployment.

- **Real-time feasibility demonstrated**: The reported <10ms average runtime (typically <100ms maximum) across all components is validated through systematic benchmarking (Figure F.1-F.2), making actual closed-loop experiments feasible.

- **Handles realistic experimental constraints**: The optimization explicitly incorporates non-negativity (appropriate for excitation-only optogenetics) and sparsity constraints (limited simultaneous targets), reflecting actual experimental limitations.

- **Adaptive to non-stationarity**: The kernel regression includes temporal discounting (K₃ in Eq. 7), and Figure 2e demonstrates recovery from sudden changes in the stimulus-response mapping within ~15 seconds.

- **Validation on multiple datasets**: Testing on both calcium imaging (Zong et al., 2022) and electrophysiology (O'Doherty, 2024), plus two additional photostimulation datasets in Appendix C, demonstrates applicability across recording modalities.

## Weaknesses
- **Primary optimization claims lack real-stimulation validation**: All stimulation optimization experiments (Figures 4-5) use simulated stimulation effects on real neural data. The core contribution—designing stimuli to achieve desired latent perturbations—is never tested with actual delivered stimulations in closed-loop. While Appendix C validates the stimulus-response prediction on real photostimulation data, the optimization framework itself is not validated with real neural responses.

- **Missing comparison to existing stimulation design methods**: The paper compares only to random baselines (Single, Multiple, Shuffled in Figure 4a) but not to established adaptive stimulation methods like Bayesian optimization (Minai et al., 2024; Wagenmaker et al., 2024) or input-output dynamical modeling (Yang et al., 2021) cited in related work. This makes it difficult to assess relative contribution.

- **Confusing notation in optimization formulation**: Equation 8 uses ||u||_max alongside ||u||_1, but the variables table defines ||u||_max as "maximum acceptable L_0 norm," while L₁ regularization is used in the objective. The text states this "encourages a solution with the number of non-zero elements close to n," but the mathematical relationship between L₀ and L₁ constraints in this formulation is unclear and should be clarified.

- **Kernel regression scalability unanalyzed**: The stimulus-response model uses kernel regression over high-dimensional stimulus vectors u ∈ R^N. No analysis is provided on how performance scales with neural population size or number of observed stimulations—critical for practical deployment.

## Nice-to-Haves
- **Characterization of reachable perturbation directions**: The paper notes some directions are "infeasible" (e.g., Negative in Figure 4), but systematic analysis of which latent perturbations are achievable under constraints would help experimental planning.

- **Ablation of kernel regression components**: The kernel uses three components (spatial, stimulus, temporal), but no ablation demonstrates whether each is necessary.

## Novel Insights
The key insight enabling this work is the combination of streaming latent space construction with a differentiable stimulus-response model learned via kernel regression. This allows gradient-based optimization of stimuli in the original neural space to achieve desired effects in the low-dimensional latent space, while respecting realistic experimental constraints. The observation that the method can distinguish between latent space hypotheses by measuring prediction accuracy (Figure 1c) is noteworthy—it suggests the framework could be used not just for stimulation design but for identifying which manifold structure best explains ongoing dynamics. The finding that closed-loop stimulus design using a learned S-hat can outperform open-loop design (Figure 5b) in non-trivial stimulus-response mappings validates the adaptive approach.

## Potentially Missed Related Work
- None identified (related work search was not performed for this submission).

## Suggestions
- **Validate stimulation optimization with actual neural responses**: Partner with an experimental lab to test whether designed stimulations produce the intended latent perturbations in closed-loop experiments. Even a proof-of-concept demonstration would substantially strengthen the claims.

- **Add comparison to at least one existing adaptive stimulation method**: Implement Bayesian optimization or active learning baseline to contextualize performance gains over current approaches.

- **Clarify the optimization formulation**: Either correct the notation relating ||u||_max to the constraint on number of targets, or provide explicit justification for why the L₁ surrogate achieves the stated L₀ goal.

- **Provide convergence analysis for stimulus-response learning**: Quantitatively show how prediction error decreases with number of stimulations across conditions, not just qualitatively in Figure 2c-d.

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (2.0/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a Covariance-Adjusted Support Vector Machine (CSVM) that incorporates class-specific covariance information into the SVM optimization problem. The authors argue that traditional SVM assumes Euclidean distance, while the input "statistical space" requires Mahalanobis distance, motivating a class-conditional Cholesky decomposition transformation. An iterative SM Algorithm is proposed to estimate population covariance from training data when test labels are unknown, and experiments across five datasets show improvements over standard SVM kernels and PCA/ZCA whitening approaches.

## Strengths
- **Principled motivation for covariance incorporation**: The paper provides a coherent argument for why class-specific covariance structure should influence margin calculation, grounded in Mahalanobis distance and vector space properties (Section 2).
- **Clear mathematical derivation**: Equations (8)-(14) derive how margins in input space relate to class covariance matrices, formalizing the intuition that higher-variance classes deserve larger margins (Lemma 2.3).
- **Practical algorithm for unknown test labels**: The SM Algorithm (Section 3) addresses the real limitation that population covariance cannot be computed without test labels, providing an iterative self-training approach that refines covariance estimates.
- **Consistent empirical improvements**: CSVM-Cholesky achieves best or second-best accuracy on 4 of 5 datasets, with improvements ranging from ~1-3% over standard SVM kernels (Tables 1-4).

## Weaknesses
- **Imprecise "non-Euclidean" terminology**: The paper repeatedly calls the input space "non-Euclidean," but standard mathematical terminology reserves this for curved geometries (hyperbolic, spherical). The input space is $\mathbb{R}^n$ with a Mahalanobis metric—still a vector space with a different inner product. This terminology may confuse readers familiar with differential geometry and weakens the theoretical framing.
- **No comparison to cited prior variance-adjusted SVM methods**: The introduction cites MCVSVM (Zafeiriou et al.), Mahalanobis kernel SVM (Wang et al.), and twin Mahalanobis SVM (Peng & Xu), claiming these have "gaps in application of appropriate vector spaces and dimensional inconsistencies." However, none of these methods appear in experimental comparisons. Without direct comparison, the claimed advantages over prior work cannot be assessed.
- **Missing statistical rigor in empirical evaluation**: Tables 1-4 report single values without error bars, standard deviations, confidence intervals, or statistical significance tests. Results are reported from a single 80:20 train/test split with no cross-validation. For the claimed improvements (e.g., 0.974 vs 0.956 accuracy on Breast Cancer), we cannot determine if differences are statistically meaningful.
- **No convergence analysis for SM Algorithm**: The iterative algorithm lacks theoretical guarantees or empirical analysis of convergence properties—conditions for convergence, expected iterations, or failure modes when initial labels are poor are not discussed.
- **Missing hyperparameter and implementation details**: The paper does not specify SVM regularization parameter $C$ values used, how RBF kernel parameters were selected, or whether hyperparameter tuning was performed consistently across all methods.

## Nice-to-Haves
- **Ablation study**: Compare CSVM with simple class-conditional whitening (without the SM iteration) to isolate the contribution of each component.
- **Visualization of decision boundaries**: 2D synthetic examples showing how margin splitting by covariance ratios differs from standard SVM would strengthen the theoretical claims.
- **Synthetic experiments with controlled covariance**: Validate the theoretical claim that margins should split proportionally to class covariances.

## Novel Insights
The paper's key insight is that when transforming from a covariance-weighted "statistical space" back to the original input space, the margins are not equal for both classes—they split in proportion to the inverse class covariance matrices (Equation 14). This provides a mathematical justification for why class-conditional whitening should precede SVM classification: it ensures the optimization problem is solved in a space where Euclidean distance is the appropriate metric. While the "non-Euclidean" framing is imprecise, the underlying observation that SVM's margin maximization implicitly assumes isotropic covariance is valid and worth formalizing.

## Potentially Missed Related Work
- **None identified in this review cycle** (related work search was skipped). However, the paper should compare against the variance-adjusted SVM methods already cited in its introduction: MCVSVM (Zafeiriou et al., 2007), weighted Mahalanobis distance kernels (Wang et al., 2007), and twin Mahalanobis SVM (Peng & Xu, 2012).

## Suggestions
1. Replace "non-Euclidean" terminology with more precise language (e.g., "covariance-weighted space" or "Mahalanobis metric space").
2. Add experimental comparisons against at least MCVSVM and Mahalanobis kernel SVM—the specific prior work claimed to have "gaps."
3. Report results with k-fold cross-validation, including means ± standard deviations and statistical significance tests (e.g., paired t-tests or Wilcoxon signed-rank tests).
4. Provide hyperparameter selection details for all methods to ensure fair comparison.
5. Add analysis of SM Algorithm convergence—either theoretical guarantees or empirical demonstration of typical iteration counts and conditions for success.

---

## eETr3lrOQB

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Final Review

## Summary

This paper proposes VQ-Transplant, a framework for efficiently integrating new vector quantization (VQ) modules into pre-trained visual tokenizers without costly end-to-end retraining. The method operates in two stages: (1) VQ module substitution with frozen encoder-decoder, and (2) lightweight decoder adaptation requiring only 5 epochs. The authors also introduce MMD-VQ, which uses Maximum Mean Discrepancy for distributional alignment and is shown to handle non-Gaussian feature distributions better than prior Wasserstein VQ.

## Strengths

- **Strong practical motivation**: Training state-of-the-art tokenizers like VAR requires substantial resources (16× A100 GPUs for 60 hours on OpenImages). VQ-Transplant reduces training to 22 hours on 2× A100 while achieving comparable or better reconstruction quality, genuinely democratizing VQ research.

- **Comprehensive empirical evaluation**: The paper systematically evaluates across five VQ algorithms (Vanilla, EMA, Online, Wasserstein, MMD), multiple codebook sizes (4096–65536), both multi-scale and fixed-scale configurations, and four datasets (ImageNet-1k, FFHQ, CelebA-HQ, LSUN-Churches).

- **Strong reconstruction results**: VQ-Transplant with MMD-VAR achieves r-FID of 0.81 on ImageNet-1k, outperforming the baseline VAR tokenizer (0.92 r-FID). On FFHQ, the method achieves state-of-the-art r-FID of 1.21, significantly outperforming VQGAN-LC's 3.81.

- **Cross-dataset generalization demonstrated**: Decoder adaptation was performed only on ImageNet-1k, yet strong performance transfers to structurally different datasets (FFHQ, CelebA-HQ, LSUN-Churches), demonstrating genuine generalization capability.

- **Theoretical justification for MMD-VQ**: The synthetic experiments (Tables 12-13, Figure 7) empirically validate that MMD-VQ outperforms Wasserstein VQ when feature distributions deviate from Gaussian assumptions, providing principled reasoning for the proposed method.

## Weaknesses

- **No downstream task evaluation**: The paper claims VQ-Transplant enables VQ techniques for visual generation but evaluates only reconstruction quality. There is no evaluation of how transplanted tokenizers perform on actual downstream tasks such as autoregressive image generation or diffusion-based synthesis—the primary use cases for visual tokenizers. Reconstruction fidelity is a proxy but does not guarantee downstream utility.

- **Missing same-budget fine-tuning baseline**: The paper compares against training from scratch but omits a critical baseline: full fine-tuning of encoder+decoder+VQ for the same 22-hour compute budget. This comparison would establish whether decoder-only adaptation sacrifices performance for efficiency, or whether it is genuinely the optimal strategy.

- **Limited tokenizer architecture diversity**: Claims of "plug-and-play integration" are primarily substantiated on VAR tokenizer. The LDM-16 experiments (Table 16) show notably worse adaptation (r-FID 2.58 vs. 0.81), suggesting the framework's effectiveness depends heavily on the base tokenizer quality—an important limitation that deserves main-text discussion.

- **No statistical significance reporting**: All results appear to be single-run evaluations without error bars or confidence intervals. Given the inherent variability of adversarial training, this raises concerns about result reproducibility and robustness.

- **Marginal empirical gains of MMD-VQ**: While MMD-VQ is theoretically superior for non-Gaussian distributions, empirical gains on real datasets are modest. On ImageNet-1k with codebook size 8192, MMD-VAR adaptation achieves r-FID of 0.81 vs. Wasserstein VAR's 0.83—a small practical difference despite the theoretical advantages.

## Nice-to-Haves

- **Ablation on adversarial loss necessity**: The decoder adaptation uses adversarial training with a DINO-S discriminator. An ablation testing whether simpler alignment losses (e.g., L2 + perceptual loss only) would suffice would strengthen claims that the approach is "lightweight."

- **Analysis of real encoder feature distributions**: MMD-VQ's advantage over Wasserstein-VQ is motivated for non-Gaussian distributions, but the paper does not verify whether real encoder features from VAR actually exhibit non-Gaussianity in practice.

- **Computational cost breakdown**: Table 1 shows total hours but lacks Stage I vs. Stage II breakdown and GPU memory requirements, which would help practitioners assess practical efficiency.

## Novel Insights

The key insight is that VQ module quality and encoder-decoder quality can be decoupled: freezing a strong encoder-decoder while training only a new VQ module preserves the learned feature representations, requiring only lightweight decoder adaptation to align with the new quantization space. This challenges the prevailing paradigm that VQ algorithms must be trained jointly with encoder-decoder architectures. The empirical finding that decoder adaptation alone (vs. joint optimization) achieves strong results with lower training cost is practically valuable. The synthetic experiments demonstrating MMD-VQ's robustness to non-Gaussian distributions provide principled motivation, though verification on real feature distributions would strengthen this claim.

## Potentially Missed Related Work

- None identified in the provided materials. The paper adequately covers prior work on discrete visual tokenizers (VQVAE, VQGAN, VAR) and VQ techniques (EMA VQ, Online VQ, Wasserstein VQ).

## Suggestions

- Add downstream task evaluation: Train an autoregressive generator using the transplanted tokenizers and report generation FID/IS to verify practical utility beyond reconstruction.

- Include a same-budget full fine-tuning baseline to strengthen the efficiency claims.

- Report results with error bars from multiple runs to establish statistical significance.

---

## GRufFX1gAy

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Final Review

## Summary
InnoGym proposes a benchmark for evaluating AI agent innovation potential through two complementary metrics: performance gain (improvement over best-known solutions) and novelty (methodological dissimilarity from prior approaches). The benchmark includes 18 "Improvable Tasks" curated from real-world competitions and scientific problems, with a unified execution environment (iGym). Key findings reveal that current agents can achieve novelty but often lack robustness, highlighting a gap between creativity and effective performance.

## Strengths
- **Novel problem formulation**: The paper correctly identifies that existing benchmarks measure only correctness, overlooking methodological diversity. This addresses a genuine gap in AI agent evaluation as agents tackle increasingly complex problems where solution approach matters.
- **Principled theoretical framework**: The formalization of tasks as quadruples (P, S, V, D) with mathematically defined performance gain G and novelty N metrics (Equations 2-3) provides solid theoretical grounding. The three-category taxonomy (Solved, Improvable, Exploratory problems) offers conceptual clarity.
- **Rigorous benchmark construction**: The two-stage filtering (resource availability, evaluator quality) and standardization steps (validator construction, evaluator normalization, data partitioning) demonstrate careful methodology. The 18 tasks span real competitions (NeurIPS, KDD Cup, ROADEF) and classic optimization problems.
- **Validation of novelty metric**: Appendix F validates D_AGENT against human judgments on EquiBench triplets and cross-domain method comparisons, showing reasonable alignment (Pearson r ≈ 0.84-1.0, 75-100% triplet agreement). The triplet-based protocol with algorithmic variants vs. superficial edits is a sensible sanity check.
- **Substantive empirical finding**: The central insight—that agents achieve novelty without corresponding performance gains—represents a meaningful contribution. The identification of robustness as the primary bottleneck for innovation is actionable for future agent development.

## Weaknesses
- **Limited validation of the novelty metric**: D_AGENT relies on LLM-based judgments (Codex for extraction, GPT-5 for comparison). The human validation covers only 11 annotated triplets (8 EquiBench + 3 domain triplets), which is insufficient to establish metric reliability. No ablation studies examine sensitivity to the choice of judge model, the 6 comparison dimensions, or extraction prompts. Without demonstrating that novelty scores are stable across different LLM judges, the metric's validity remains uncertain.
- **No human baseline under comparable conditions**: The paper compares agent performance to leaderboard solutions developed by humans over weeks or months, but restricts agents to 12 hours. This asymmetric comparison makes claims about "performance gaps" difficult to interpret—agents may underperform due to time constraints rather than capability limitations. A fair comparison requires humans working under identical conditions.
- **Incomplete experimental coverage and high failure rates**: Only 10 of 18 tasks are evaluated due to computational constraints, with many "/" entries (failed submissions) in Table 2. Two tasks (CDML, PTTALC) have 0% success rates across all agents. The paper provides no failure mode analysis to explain why agents cannot produce valid submissions—whether due to task setup complexity, agent limitations, or other factors. This limits interpretability of what the benchmark actually measures.
- **Statistical limitations for novelty claims**: Table 5 shows that while performance differences between frameworks (MLAB vs. others) are statistically significant, novelty differences are not (p > 0.05). Claims about framework differentiation should be tempered accordingly.

## Nice-to-Haves
- **G-N correlation analysis**: A scatter plot of performance gain vs. novelty across all solutions would directly visualize the claimed "gap between creativity and effectiveness." Currently, the paper asserts this relationship without systematic quantitative demonstration.
- **Qualitative examples of novelty**: Showing actual code or methodology differences that receive different novelty scores would help readers understand what the metric captures and verify it measures meaningful methodological differences.
- **Simpler metric baselines**: Comparing D_AGENT against alternatives (code embedding similarity, AST edit distance) would contextualize whether the LLM-based approach offers genuine advantages.

## Novel Insights
The conceptual innovation lies in formally distinguishing solution effectiveness from methodological originality—a distinction masked by correctness-only evaluation. The complex-plane visualization (Figure 5b) that jointly encodes performance gain (magnitude) and novelty (angle) provides an intuitive geometric interpretation of innovation trajectories. The empirical finding that robustness, not novelty generation, is the primary bottleneck for current agents reframes the challenge for future agent development: agents can propose diverse approaches but struggle to implement them correctly.

## Potentially Missed Related Work
None identified.

## Suggestions
1. Expand D_AGENT validation to at least 50+ human-annotated triplets with inter-rater reliability statistics (Cohen's κ, ICC) to establish metric robustness.
2. Add ablation studies comparing D_AGENT outputs across different judge models (e.g., GPT-4, Claude, DeepSeek) to assess sensitivity.
3. Include failure mode analysis for tasks where all agents failed to produce valid submissions—determining whether failures stem from task complexity, formatting issues, or fundamental capability gaps.
4. Clarify the availability and reproducibility of experiments involving GPT-5 and Gemini-2.5-Pro, or replace with publicly accessible models.

---

## wUzBBsrdB1

- GT: Reject (avg 5.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Final Review

## Summary
This paper investigates how incorrect L0 (sparsity) settings in Sparse Autoencoders (SAEs) lead to learned features that fail to match underlying true features. Using toy model experiments with ground truth, the authors demonstrate that low L0 causes SAEs to mix correlated features to improve reconstruction, while high L0 produces degenerate solutions. The paper proposes decoder pairwise cosine similarity as a proxy metric to identify correct L0, validated against sparse probing performance on Gemma-2-2b and Llama-3.2-1b.

## Strengths
- **Critical insight about sparsity-reconstruction tradeoff**: The paper convincingly demonstrates that the widely-used "sparsity-reconstruction tradeoff" evaluation is fundamentally flawed—showing that ground-truth SAEs achieve worse reconstruction than incorrect SAEs that mix features (Section 3.4, Figure 4-5). This has immediate implications for how SAEs should be evaluated.
- **Strong theoretical grounding**: Appendix A.5 provides a formal proof that MSE loss incentivizes feature mixing when L0 is constrained below the true L0. The toy model experiments (Figures 2-3) cleanly isolate the correlation-dependent mixing mechanism for both positive and negative correlations.
- **Practical validation**: The c_dec metric correlates with sparse probing F1 peaks across multiple models and layers (Figure 8), and the analysis of open-source SAEs (Appendix A.13) provides concrete evidence that commonly-used L0 values may be too low.
- **Architecture comparison**: Testing both BatchTopK and JumpReLU SAEs with analysis of their different behaviors at high L0 (Appendix A.16) adds useful nuance—the observation that JumpReLU "sticks" near correct L0 across sparsity coefficients is practically relevant.

## Weaknesses
- **Ambiguous metric interpretation in practice**: The c_dec curves sometimes lack clear minima (e.g., Figure 8, Gemma-2-2b layer 5 shows a nearly flat region across a wide L0 range). The paper relies on a visual "elbow" heuristic without quantitative criteria, making practical L0 selection difficult for real-world use.
- **Limited validation scope and quality proxies**: The paper tests only 3-4 layers across 2 small models (Gemma-2-2b, Llama-3.2-1b) and relies solely on sparse probing F1 as a quality proxy. No comparison to alternative L0 selection methods (MDL-SAEs, AFA-SAEs), interpretability evaluations, or downstream task performance is provided.
- **Underexplored high L0 mechanism**: While low L0 feature mixing is well-explained both theoretically and empirically, the mechanism for why high L0 causes "degenerate solutions" is less thoroughly analyzed (Section 3.2, 4.2). The finding that L0 can be simultaneously too high and too low (Section 4.2) deserves deeper explanation.
- **Assumption of "true L0" needs justification**: The paper assumes LLMs have a well-defined "true L0" to match, but this may not hold if features have different firing rates across contexts or if underlying representations are non-linear.

## Nice-to-Haves
- Decoder cosine similarity matrices for real LLM SAEs (matching the clear toy model visualizations in Figures 1-3) would strengthen the claim that the same mixing patterns occur in practice.
- Qualitative examples showing what specific latents encode at different L0 values would make the "polysemanticity" claims more concrete.
- Ablations on how optimal L0 varies with SAE width (dictionary size) and training data distribution would clarify generalizability.

## Novel Insights
The paper provides an important conceptual insight: L0 is not merely a hyperparameter trading off sparsity against reconstruction, but rather must be set correctly to avoid systematic feature corruption. The finding that low L0 causes *all* latents to become polysemantic (not just some) is significant—it means underspecified sparsity damages the entire dictionary. The theoretical connection between correlated features and the MSE reconstruction incentive (Appendix A.5) elegantly explains why this failure mode occurs, and the demonstration that sparsity-reconstruction plots would reject a hypothetically perfect SAE in favor of a corrupted one is a compelling result that should change evaluation practices in the field.

## Potentially Missed Related Work
- None identified in the provided materials. The paper references relevant prior work on SAE limitations (feature splitting, feature hedging, absorption) and alternative L0 selection approaches (MDL-SAEs, AFA-SAEs), though experimental comparison to these methods is lacking.

## Suggestions
- Provide quantitative criteria or an algorithmic method for identifying correct L0 from c_dec curves (e.g., curvature-based methods or threshold-based "elbow" detection) beyond visual inspection.
- Add experimental comparison with MDL-SAE's approach for L0 selection as a baseline to establish whether c_dec provides better guidance.
- Include additional validation metrics beyond sparse probing (e.g., autointerpretability scores, steering experiments) to verify that identified L0 values produce genuinely more interpretable features.

---

## piylyBPSau

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Final Review

## Summary
The paper proposes GenCoGS, a unified approach for few-shot novel view synthesis using 3D Gaussian Splatting. The key innovation lies in two generative completion strategies: (1) GCGI, which generates and filters complementary points to complete sparse point clouds for better Gaussian initialization, and (2) GCGO, which uses an image-to-video diffusion model with a perturbed camera trajectory and generative consistency loss to synthesize complete pseudo views that guide Gaussian optimization in unobserved regions while mitigating hallucination artifacts.

## Strengths
- **Strong quantitative performance**: GenCoGS achieves substantial improvements across three benchmark datasets. On DTU 3-view, it reaches 23.11 PSNR (2.40 dB improvement over the second-best 3DGS method), with consistent gains across PSNR, SSIM, LPIPS, and AVGE metrics on LLFF, DTU, and Shiny datasets.
- **Novel problem formulation**: The paper identifies a clear limitation in existing 3DGS few-shot methods—inadequate scene completion in unobserved regions causing floating artifacts and hollow structures—and proposes a unified solution addressing both initialization and optimization stages.
- **Comprehensive ablation studies**: Tables 4-6 and Appendix B.3 systematically analyze each component's contribution. The analysis of the "see-saw effect" between hallucination suppression and unobserved region exploration (Figure 8) provides useful practical insights.
- **Thoughtful hallucination mitigation**: The generative consistency loss with adaptive confidence masking (Equations 12-16) and point filtering based on kd-Tree distance (Equations 5-8) show careful engineering to address diffusion model hallucination artifacts.
- **Efficiency remains reasonable**: Training time increases to 40 minutes (vs. ~30 min for BinoGS) with 4GB memory—acceptable overhead given the performance gains.

## Weaknesses
- **Critical missing training details for CPG module**: Section 3.1.1 describes the Complementary Point Generation module architecture but provides no information about how it is trained—whether pretrained weights are used, what training data/loss functions are employed, or whether it's trained jointly with the main pipeline. This significantly impacts reproducibility.
- **Chamfer distance evaluation is circular**: Table 8 compares the completed point cloud against the "final optimized 3DGS point cloud"—not ground truth geometry. This shows GCGI produces points closer to what the optimization converges to, but doesn't prove geometric correctness of the completed regions.
- **No statistical variance reported**: Given the stochastic nature of both diffusion model sampling and 3DGS optimization, the absence of error bars or multiple-run statistics limits reproducibility assessment.
- **Missing failure case analysis**: The paper claims to mitigate hallucination but provides no analysis of scenes where generative completion introduces artifacts that persist in final outputs, or where the confidence mask fails to identify distorted regions.
- **I2V model dependency unanalyzed**: The method relies critically on ViewCrafter for pseudo-view synthesis, yet no ablation tests alternative I2V models to assess whether results generalize or are model-specific.
- **No per-scene quantitative breakdown**: Reporting only mean metrics across datasets may hide scene-specific failures, making it impossible to verify consistent performance.
- **Several hyperparameter choices lack justification**: Parameters like perturbation amplitude A=2.0, wave frequency f=1.0, and thresholds δ₁=1.0, δ₂=20, δ₃=8 are tuned empirically without theoretical grounding, raising questions about generalization across diverse scene types.

## Nice-to-Haves
- Include visualizations of the confidence mask M̂r to demonstrate where hallucination is identified versus missed, and point cloud visualizations before/after GCGI to illustrate structural improvements.
- Add per-scene results tables for LLFF and DTU to show consistent improvement across individual scenes rather than relying solely on aggregate metrics.

## Novel Insights
The paper's core insight—that few-shot NVS requires "imagination" beyond observed regions, analogous to human visual completion—is well-motivated and effectively translated into a dual-strategy approach. The perturbed camera trajectory for pseudo-view sampling is a practical contribution that systematically explores unobserved scene regions. The empirical discovery of the "see-saw effect" between hallucination suppression and unobserved region exploration (where larger perturbation amplitudes increase hallucination) represents a valuable finding for future diffusion-guided 3D reconstruction work.

## Potentially Missed Related Work
- **MVSplat and Splatter Image (2024)**: Recent feed-forward 3D reconstruction methods that are directly relevant baselines for few-shot NVS. Comparing against these efficient alternatives would better contextualize GenCoGS's contributions.
- **GaussianPro (2024)**: Uses progressive Gaussian propagation for scene completion—relevant for comparison on initialization strategies.

## Suggestions
- Provide complete training details for the CPG module in the main paper or supplementary material: training data source, loss functions, and whether pretrained weights are used.
- Report standard deviations across multiple runs with different random seeds for main quantitative results.
- Add at least one alternative I2V model ablation to demonstrate robustness of the approach beyond ViewCrafter.
- Include failure case visualizations showing where GenCoGS produces worse results than baselines or where hallucination artifacts persist despite filtering.

---

