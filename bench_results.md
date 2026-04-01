# ICLR Benchmark Results

Date: 2026-04-01 01:51
Critic/Merger: z-ai/glm-5 (OpenRouter)
Neutral: z-ai/glm-5, Related Work: z-ai/glm-5:online (OpenRouter)

## zfeso8ceqr

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (6.6/10)
- Match: Yes

### Final Review

{
  "summary": "This paper presents a substantial empirical study of optimizer behavior for autoregressive language model pretraining, comparing SGD, Adam, Adafactor, Lion, Signum, and an appendix-level Sophia analysis across model scales from 150M to 1.2B parameters, multiple hyperparameters, and two architecture variants. The main empirical conclusion is that, aside from SGD, the tested diagonal/adaptive optimizers are broadly comparable in both best achieved validation loss and robustness to common hyperparameter choices. The more novel contribution is the Adalayer-based dissection suggesting that adaptive preconditioning is especially important for the last layer and LayerNorm parameters, while much of the rest of the network can tolerate much coarser or even frozen scaling. Overall, the paper is well executed and practically relevant, with meaningful scale and careful ablations. The main limitations are the absence of variance estimates across seeds, the reliance on 1D hyperparameter sweeps, and somewhat stronger wording than the evidence fully supports around the “necessity” of last-layer/LayerNorm adaptivity. The paper would be stronger with more statistical rigor, broader validation beyond a single dataset, and deeper mechanistic explanation of the layerwise findings.",
  "strengths": [
    "The experimental scale is strong for an optimizer comparison paper: models up to 1.2B parameters trained for approximately Chinchilla-style token budgets are much more convincing than purely small-scale proxy studies.",
    "The optimizer comparison is systematic, covering learning rate, momentum, weight decay, warmup, beta2, epsilon, architectural variants, and longer training runs; this makes the paper practically useful as a reference for LM training.",
    "A clear and valuable empirical message emerges: among the tested non-SGD diagonal/adaptive optimizers, no method clearly dominates Adam in either peak performance or hyperparameter robustness.",
    "The Adalayer analysis is the most novel part of the paper and provides actionable insight: adaptivity appears disproportionately important for the last layer and LayerNorm parameters.",
    "The Signum/Adam connection is well motivated and empirically supported, helping clarify that much of Adam’s advantage over SGD may be tied to sign-like behavior rather than fine-grained second-moment estimation everywhere.",
    "The paper is generally clear, transparent about limitations, and appropriately acknowledges related and concurrent work such as BAGM and Adam-mini."
  ],
  "weaknesses": [
    "A key empirical weakness is the lack of multiple seeds or variance estimates. For a benchmarking-style paper making comparative claims about small differences among optimizers, the absence of error bars reduces confidence in how robust those conclusions are.",
    "The methodology relies primarily on 1D hyperparameter sweeps around a selected learning rate, and the paper itself acknowledges that this can miss important higher-order interactions, especially between learning rate and momentum or batch size. This limits the strength of claims about overall 'stability.'",
    "All main experiments are on a single pretraining dataset (C4) and one task family (autoregressive LM pretraining), so the generality of the conclusions to other corpora, tokenizers, domains, or downstream quality remains untested.",
    "The claims about adaptivity on the last layer and LayerNorm being 'necessary' are somewhat too strong relative to the evidence. The experiments convincingly show these components are especially important, but some results still show residual gaps to Adam/Adalayer*, so 'largely necessary' or 'particularly important' would be more precise.",
    "The paper identifies an interesting phenomenon for last-layer and LayerNorm adaptivity but provides limited mechanistic explanation for why these components specifically require it.",
    "Sophia is mentioned in the abstract/introduction but is mostly moved to the appendix; although the appendix does provide some justification, the treatment feels less thorough than for the main optimizers and would benefit from clearer tuning/positioning in the main text."
  ],
  "nice_to_haves": [
    "Add 3+ seeds for representative sweeps and report mean/std or confidence intervals, especially for claims of comparability among Adam, Adafactor, Lion, and Signum.",
    "Include at least one additional dataset or domain (e.g., another web corpus or code data) to test whether the conclusions are C4-specific.",
    "Provide training-dynamics plots, not just final validation loss, to show whether the optimizers are truly similar throughout training or only at convergence.",
    "Add wall-clock throughput and memory measurements, since the paper argues practitioners should choose among optimizers based on practical considerations like memory and implementation.",
    "Clarify the relationship between Adalayer* and concurrent Adam-mini more explicitly, especially what is analytically distinct here versus overlapping.",
    "Either temper Sophia’s prominence in the framing or expand its treatment in the main text with clearer tuning details and conclusions."
  ],
  "novel_insights": "The paper’s most interesting insight is not the broad optimizer comparison itself, but the decomposition of where adaptivity matters in autoregressive transformers. The evidence suggests that much of Adam-like behavior can be reproduced without per-parameter adaptivity everywhere: coarse or even frozen scaling appears sufficient for many matrix layers, whereas the last layer and LayerNorm parameters are disproportionately important for retaining performance and learning-rate robustness. This is a useful empirical refinement of the common narrative that full per-parameter adaptivity is uniformly essential across transformer parameters.",
  "missed_related_work": [
    "Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019) would be a useful citation since the paper uses AdamW-style decoupled weight decay.",
    "The Marginal Value of Adaptive Gradient Methods in Deep Learning (Wilson et al., 2017) could provide broader historical context on when adaptive methods help relative to SGD.",
    "On the Convergence of Adam and Beyond (Reddi et al., 2018) may be worth citing for theoretical context around Adam’s stability/convergence behavior."
  ],
  "suggestions": [
    "Temper the wording around 'necessity' of last-layer and LayerNorm adaptivity unless supported by stronger evidence; the current results strongly indicate importance, but not complete sufficiency/exclusivity.",
    "Add statistical support via multiple seeds for at least the main optimizer comparisons and the key Adalayer findings.",
    "Strengthen the mechanistic analysis of why last-layer and LayerNorm parameters need adaptivity, for example via gradient-scale statistics, Hessian/spectrum analysis, or layerwise update diagnostics.",
    "Discuss more explicitly how much the conclusions may depend on the fixed batch size and 1D sweep protocol, especially for the stability claims.",
    "If possible, broaden validation beyond C4 and/or include downstream evaluation to show whether comparable pretraining loss translates to comparable model quality.",
    "Clarify Sophia’s tuning/setup and justify its reduced emphasis more directly in the main paper.",
    "Include practical profiling (memory/throughput) to support the recommendation that optimizer choice can be guided by implementation and memory considerations."
  ],
  "novelty": 7.0,
  "technical_soundness": 7.5,
  "empirical_support": 7.0,
  "significance": 7.5,
  "clarity": 8.0
}

---

## nibeaHUEJx

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.8/10)
- Match: Yes

### Final Review

{
  "summary": "This paper proposes a Fourier-domain transformation for time-series shift invariance: by linearly phase-shifting all harmonics so that a selected low-frequency harmonic attains a target phase, circularly shifted versions of the same signal are mapped to an identical canonicalized signal. The paper provides mathematical arguments for bijectivity and invariance under the stated circular-shift setting, introduces a small guidance network to choose target angles, and evaluates the approach on a broad set of time-series tasks. The empirical study is unusually extensive and generally supports the main claim that the method achieves perfect consistency under the paper’s evaluation protocol while often improving downstream performance. The main caveats are that the guarantees rely on circular shifts, the practical role of the guidance network is mostly empirical rather than theoretically explained, and computational/robustness analyses are limited.",
  "strengths": [
    "Clear and meaningful problem formulation: shift sensitivity in time-series models is important, and the paper convincingly shows that image-motivated anti-aliasing/subsampling approaches do not reliably solve it in this domain.",
    "Technically interesting core idea: using the phase of a harmonic with period at least the signal length to parameterize circular shifts, then applying a linear phase shift to all harmonics to canonicalize the signal, is novel in this context and mathematically elegant.",
    "Strong theoretical component for the stated setting: the paper gives explicit propositions/theorems and appendix proofs supporting the bijection/canonicalization claims under circular shifts.",
    "Architecture-agnostic formulation: the method operates on inputs rather than requiring a specialized backbone, and the paper demonstrates compatibility with multiple architectures and tasks.",
    "Very broad empirical coverage: 9 datasets across 6 tasks, with multiple baselines and ablations, is a substantial experimental effort.",
    "Empirical evidence is largely consistent with the main invariance claim: under the paper’s shift-consistency metric over the full circular shift space, the proposed method attains 100% consistency across reported tasks.",
    "Ablation studies are useful and show that the learned guidance mechanism matters; simply mapping all samples to a fixed angle is generally not as effective as the full method.",
    "The paper is generally honest in reporting mixed outcomes in some combinations/baselines rather than presenting uniformly favorable results."
  ],
  "weaknesses": [
    "The guarantees are fundamentally for circular shifts, and this limitation is important for practice. Although the paper states this assumption, it is not emphasized enough relative to the strength of the headline claims about unrestricted shift invariance. For many real time-series settings, shifts induce boundary effects rather than wrap-around.",
    "The guidance network is only weakly justified theoretically. The empirical ablations suggest it helps, but the paper does not provide a principled explanation of why minimizing angle variance or learning sample-dependent canonical phases should improve task performance beyond invariance alone.",
    "The claim of preserving 'all task-relevant information without loss' is stronger than what is rigorously established. The transformation preserves the signal under circular-shift equivalence, but whether absolute timing relative to the window start is irrelevant is task-dependent and not universally analyzed.",
    "Robustness of the method to low-amplitude/noisy reference harmonics is not analyzed. Since the method extracts phase from a selected low-frequency component, it would be helpful to understand failure modes when that component has weak magnitude or unstable phase.",
    "Computational overhead is not quantified. The method requires FFT-based preprocessing/transformation plus a guidance network, but the paper does not report runtime, memory, or FLOP overhead, which matters for deployment in real-time applications.",
    "Some baselines are unavailable on certain datasets (e.g., APS entries marked missing), and while the paper gives an explanation, the comparison remains incomplete in those cases.",
    "The shift-consistency metric is important and appropriate, but for the proposed method it is satisfied by construction under the stated assumptions; thus the more discriminating empirical question is the accuracy/performance tradeoff, where gains are real but not uniform across every dataset/metric.",
    "Sensitivity to design choices central to the method—especially harmonic selection and dependence on signal length/sampling setup—is not thoroughly explored."
  ],
  "nice_to_haves": [
    "Add non-circular shift experiments (e.g., zero-padded/window-truncating shifts) to delineate where the theoretical guarantee does and does not apply.",
    "Report runtime/memory/FLOP overhead for training and inference relative to baselines.",
    "Include sensitivity analysis for harmonic choice, especially the selected reference harmonic and cases with low magnitude.",
    "Provide a more quantitative analysis of what the guidance network is doing, e.g., intra-class compression/inter-class separation or class-conditional angle distributions.",
    "Add robustness experiments under low-SNR conditions or perturbations affecting phase estimation.",
    "Clarify the practical implications of losing absolute timing information, perhaps by discussing tasks where this is acceptable versus problematic."
  ],
  "novel_insights": "The paper’s most interesting insight is that, for circular shifts, one can use the phase of a sufficiently low-frequency harmonic as a unique coordinate over the shift space, then canonically align all harmonics via a linear phase correction. This reframes shift invariance for time series as canonicalization in signal space rather than anti-aliased architecture design. The experiments also suggest an important practical lesson: in time series, enforcing invariance through low-pass filtering or adaptive subsampling may be much less suitable than direct canonicalization because high-frequency content and cross-frequency interactions often remain task-relevant.",
  "missed_related_work": [
    "Group Equivariant Convolutional Networks (Cohen & Welling, 2016) as foundational equivariant deep learning background.",
    "Steerable CNNs (Cohen & Welling, 2017) as another foundational reference on equivariance/invariance.",
    "Wavelet scattering / scattering networks as a theoretically motivated shift-invariant baseline worth discussing more explicitly if not already covered via Mallat’s scattering reference.",
    "Potentially related recent work on hard-coded invariances for time series (if concurrent/timely for submission), as a suggestion rather than a required citation."
  ],
  "suggestions": [
    "Reframe the main claim more carefully around circular-shift invariance, and discuss applicability to non-circular/window-boundary shifts more explicitly in the main text rather than mainly in footnotes/appendix.",
    "Temper or qualify the 'no loss of task-relevant information' claim unless supported by a stronger task-level argument; alternatively, state clearly that the method preserves information modulo circular shifts.",
    "Add computational cost analysis and deployment-relevant overhead measurements.",
    "Add sensitivity/robustness experiments for weak reference harmonics, noisy signals, and alternative harmonic choices.",
    "Strengthen the explanation of the guidance network, either theoretically or with targeted empirical analyses showing how it changes class geometry.",
    "Where baseline results are missing due to implementation limitations, provide fuller discussion so readers can judge fairness and scope of the comparison.",
    "If space permits, include comparison or discussion with additional theoretically relevant invariant methods such as scattering-based approaches."
  ],
  "novelty": 8.0,
  "technical_soundness": 7.5,
  "empirical_support": 8.5,
  "significance": 7.5,
  "clarity": 7.5
}

---

## oYSsbY3G4o

- GT: Accept (Poster) (avg 6.4)
- Predicted: Accept (6.7/10)
- Match: Yes

### Final Review

{
  "summary": "This paper proposes GQT, a graph tokenizer that decouples tokenizer pretraining from Transformer training by combining a GNN encoder, residual vector quantization, and multi-task graph self-supervised learning (DGI + GraphMAE2 + commitment loss). The resulting discrete tokens are serialized with semantic-edge augmentation and PPR-based neighborhoods, then consumed by a vanilla Transformer with token modulation. The paper is ambitious in scope and empirically strong: it evaluates on 22 benchmarks spanning homophilic, heterophilic, large-scale, and long-range settings, and reports best performance on 20/22 with additional robustness and efficiency evidence. Overall, the work is practically valuable and clearly above the level of a minor engineering tweak. The main caveat is that the technical novelty is more in the integration of known ingredients than in a fundamentally new algorithmic idea, and several important analyses are still missing: fairness of comparisons given extensive per-dataset tuning and reliance on literature baselines, preprocessing/training cost of the two-stage pipeline, sensitivity to quantizer design choices, and stronger evidence for transfer/generalization claims beyond same-dataset training.",
  "strengths": [
    "Strong empirical coverage: the method is evaluated on 22 datasets across multiple graph regimes, including large-scale OGB datasets and LRGB tasks, and performs very competitively throughout.",
    "The memory-efficiency story is compelling. The paper explains the compression mechanism clearly, gives concrete reduction examples, and includes inference memory/runtime evidence (Table 10).",
    "The tokenizer design is sensible and well motivated: GNNs capture local interactions, quantization compresses them into discrete tokens, and the Transformer focuses on longer-range interactions.",
    "The ablations are meaningful and support several core claims: removing the tokenizer hurts performance, SSL objectives matter substantially, and semantic edges / structural gating are especially useful in heterophilic settings.",
    "The paper includes robustness experiments under adversarial attacks and shows gains over an RQ-VAE-style alternative, which broadens the empirical support beyond plain accuracy.",
    "The method is described with enough detail to understand the pipeline, including tokenizer pretraining, quantization, graph serialization, and Transformer modulation."
  ],
  "weaknesses": [
    "The novelty is moderate rather than fundamental. Much of the contribution comes from combining existing components—RVQ, DGI, GraphMAE2, PPR-based serialization, and semantic-neighbor augmentation—into a coherent pipeline, with limited new algorithmic or theoretical insight.",
    "Baseline comparison fairness is a real concern. The paper performs substantial per-dataset tuning for GQT while many baseline numbers are inherited from prior work, so it is unclear whether competing methods benefited from comparable tuning budgets and hardware settings.",
    "The paper emphasizes inference memory savings, but the total cost of the two-stage pipeline is not quantified. Tokenizer pretraining and preprocessing (semantic edges, PPR sequences) may be nontrivial, especially on large graphs, yet training-time cost comparisons are absent.",
    "Sensitivity analyses for core quantization choices are incomplete. Although RVQ is central to the method, the paper does not provide a dedicated study of the number of codebooks, codebook size, or alternatives to RVQ.",
    "The claim of improved generalization is only indirectly supported through downstream performance on the same datasets and comparison to RQ-VAE; there is no cross-dataset or transfer-style evaluation to validate broader tokenizer reusability.",
    "Some design choices are under-justified, especially the semantic-edge construction (PCA + cosine + kNN), the fixed gating temperature, and why the particular SSL objective combination is especially suitable for tokenization.",
    "A comparison against joint end-to-end training or fine-tuning is missing, so the benefits and possible costs of freezing/decoupling the tokenizer are not fully established.",
    "Not all gains are decisive; on some datasets improvements are small and no formal statistical significance tests are reported."
  ],
  "nice_to_haves": [
    "More analysis of the two datasets where GQT is not best (e.g., Chameleon and PCQM-Contact).",
    "Visualizations of codebook usage or token hierarchy across RVQ levels.",
    "Guidance or heuristics to reduce per-dataset hyperparameter tuning burden.",
    "Additional discussion of failure modes, such as weak/no node features or very expensive PPR preprocessing on larger graphs."
  ],
  "novel_insights": "The paper's most interesting insight is not merely that quantization compresses graph inputs, but that a decoupled tokenizer can shift the division of labor between architectures: a GNN absorbs local structural bias during self-supervised pretraining, while a vanilla Transformer can then operate over compact discrete summaries enriched with long-range serialization. This is a useful systems-level perspective on graph Transformers. The ablations also suggest an important nuance: most of the accuracy gains seem to come from the SSL-pretrained tokenizer and sequence construction, while quantization contributes primarily via compactness and some regularization rather than being the dominant source of predictive improvement. That framing makes the work more convincing as a practical graph-tokenization pipeline than as a fundamentally new quantization method.",
  "missed_related_work": [
    "Graph-BERT: Only Attention is Needed for Learning Graph Representations (Zhang et al., 2020), as an early graph-to-sequence / vanilla-Transformer-oriented graph representation work.",
    "SoundStream (Défossez et al., 2021) as an original source of Residual Vector Quantization, in addition to the cited image-generation RVQ work."
  ],
  "suggestions": [
    "Add a clearer accounting of total compute: tokenizer pretraining time, preprocessing cost for semantic edges and PPR, and end-to-end training/inference comparisons against strong baselines.",
    "Strengthen comparison fairness by either rerunning a subset of the strongest baselines under matched tuning/hardware budgets or explicitly discussing this limitation in the paper.",
    "Include a dedicated quantization study varying number of codebooks and codebook sizes, and ideally compare RVQ to simpler quantization alternatives.",
    "Evaluate transfer more directly: pretrain a tokenizer on one graph or graph collection and test reuse on a different dataset/domain to support the decoupling/generalization claim.",
    "Add a comparison to joint end-to-end training or fine-tuning of the tokenizer to show what is gained or lost by freezing the pretrained tokenizer.",
    "Provide stronger justification or ablations for semantic-edge construction choices (PCA, cosine similarity, kNN) and for the gating design.",
    "If space permits, analyze codebook utilization / collapse and whether different RVQ levels capture meaningfully different graph properties.",
    "Clarify in the main text which baselines were unavailable or OOM, and avoid overinterpreting '20/22 SOTA' without that context."
  ],
  "novelty": 6.6,
  "technical_soundness": 7.8,
  "empirical_support": 8.4,
  "significance": 8.0,
  "clarity": 8.2
}

---

## CU8CNDw6Vv

- GT: Reject (avg 5.2)
- Predicted: Accept (6.6/10)
- Match: No

### Final Review

{
  "summary": "This paper tackles an important and underexplored question: whether widely used spurious-correlation benchmarks are actually valid tools for comparing mitigation methods. The core empirical observation—that benchmark rankings disagree substantially—is compelling and practically relevant. The proposed framework of ERM Failure, Discriminative Power, and Convergent Validity is clear and useful, and the model-dependent statistic K is a thoughtful attempt to characterize task difficulty due to spurious correlation beyond label-attribute mutual information. The paper’s strongest contribution is the benchmark meta-analysis itself, especially the evidence that some benchmarks (most convincingly ImageNetBG) are poor evaluation instruments under the paper’s criteria. The practical benchmark-selection recipe is interesting, but the evidence for it is more modest than the paper’s stronger validity-analysis claims. Overall, this is a meaningful and novel benchmark-analysis paper with solid motivation and several useful insights, but some conclusions—especially binary validity judgments and practitioner-facing recommendations—would benefit from stronger statistical grounding, broader validation, and clearer discussion of limitations.",
  "strengths": [
    "Identifies a real and important problem: existing spurious-correlation benchmarks can yield conflicting method rankings, and the disagreement shown in Figure 1 is striking and consequential for method selection.",
    "Moves beyond method benchmarking to benchmark evaluation itself, which is a valuable meta-level contribution and well positioned relative to SubpopBench, WILDS, and related benchmark efforts.",
    "Introduces a clear conceptual framework with three intuitive desiderata—ERM Failure, Discriminative Power, and Convergent Validity—and operationalizes each in a concrete, reproducible way.",
    "The proposed K statistic addresses a genuine limitation of purely data-only measures such as mutual information by incorporating model-dependent learnability of both target and spurious attribute.",
    "Synthetic sanity checks for K are useful and largely convincing: K behaves as expected under manipulated confounder strength and foreground/background corruption, and Citybirds/AvP serve as sensible controlled checks.",
    "The conclusion that some benchmarks are not meaningful under the proposed criteria is supported, with ImageNetBG being the clearest and best-supported example.",
    "The analysis of method robustness across benchmarks is practically informative; the observation that some methods are consistently stronger and less variable than others is useful for the community.",
    "The paper is generally well written, well motivated, and organized around a coherent narrative from disagreement to validity analysis to tentative practitioner guidance.",
    "The authors explicitly discuss important limitations, including dependence on group annotations and model architecture, and raise a thoughtful ethical concern about CelebA’s gender annotation setup."
  ],
  "weaknesses": [
    "The empirical basis for Convergent Validity is somewhat limited. Figure 4/5 trends are suggestive, but several slopes and R^2 values are modest, and the paper does not provide a principled threshold or statistical test for when a benchmark should be considered to have 'passed' or 'failed' convergent validity.",
    "As a result, the binary valid/invalid labeling of benchmarks appears stronger than the evidence supports for some cases. ImageNetBG is convincing, but judgments on benchmarks such as MultiNLI and AvP seem less definitive and would be better framed more continuously or cautiously.",
    "K is explicitly model-dependent and the experiments use only ResNet-50 and BERT-base. The paper acknowledges this, but does not empirically test whether benchmark rankings or validity conclusions are stable across other architectures.",
    "The practical recommendation in Section 5 is promising but only lightly validated. The leave-one-out evaluation is small, CivilComments is excluded, and the gains over simpler averaging strategies are modest in several cases.",
    "The practitioner-facing recipe also inherits the requirement for group annotations and knowledge of the relevant spurious attribute, which limits deployment in realistic settings where such information is unavailable or noisy.",
    "The paper does not report uncertainty estimates or significance analyses for several central claims, including benchmark-agreement trends, method robustness summaries, and the win-rate of the closest-benchmark strategy.",
    "There is some tension in the use of CelebA: the paper relies on it analytically while also arguing it should not be used for benchmarking on ethical grounds. This tension should be handled more explicitly in the framing of conclusions and recommendations.",
    "The analysis explains which benchmarks fail under the proposed tests, but gives limited deeper diagnosis of why particular benchmarks fail and how benchmark creators could repair them."
  ],
  "nice_to_haves": [
    "Report confidence intervals or permutation/bootstrap tests for the convergent-validity slopes and for the leave-one-out recommendation results.",
    "Present validity as a continuous score or profile across the three desiderata, rather than a binary pass/fail label.",
    "Add architecture-sensitivity experiments for K and downstream validity conclusions using alternative backbones.",
    "Include more real-world held-out datasets to validate the benchmark-matching recommendation.",
    "Provide brief failure-case analysis for datasets where the closest-benchmark strategy does not help.",
    "Expand the discussion into actionable guidance for future benchmark construction."
  ],
  "novel_insights": "The paper’s most valuable insight is that benchmark disagreement should not simply be averaged away: contradictory benchmark results may reflect that benchmarks are measuring different constructs, and only some of those constructs correspond to the intended notion of spurious-correlation robustness. Framing this through validity desiderata, rather than raw leaderboard performance, is a strong conceptual contribution. The introduction of a model-dependent task-difficulty measure is also insightful because it operationalizes the idea that a spurious correlation matters only insofar as a model can exploit it relative to the true signal.",
  "missed_related_work": [
    "Consider citing the original JTT paper (Liu et al., 2021), since JTT is evaluated among the methods.",
    "Consider citing the original DFR paper (Kirichenko et al., 2023), since DFR is discussed prominently in the results.",
    "Potentially relevant as context: Wang et al. (2024) on fine-grained analysis of real-world debiasing/spurious correlation, as a complementary perspective on benchmark validity."
  ],
  "suggestions": [
    "Strengthen the convergent-validity analysis with formal uncertainty quantification and a clearer criterion for interpreting slope magnitude or declaring failure.",
    "Temper the binary valid/invalid claims for borderline cases, or replace them with continuous validity assessments across the three desiderata.",
    "Broaden validation of K across architectures and, if possible, additional reference mitigation models beyond the current reweighting and GroupDRO checks.",
    "Scale back or better support the practitioner recipe in Section 5 by adding more held-out datasets, stronger baselines, and confidence intervals for the reported gains.",
    "Discuss more explicitly the practical limitations of requiring group annotations and identified spurious attributes to compute K.",
    "Clarify how to reconcile the ethical critique of CelebA with its inclusion in the analysis—e.g., treat it as an informative case study rather than a recommended benchmark.",
    "Add deeper analysis of why specific benchmarks fail and what concrete design changes could make future benchmarks satisfy the proposed desiderata."
  ],
  "novelty": 8.0,
  "technical_soundness": 7.0,
  "empirical_support": 7.0,
  "significance": 8.0,
  "clarity": 8.0
}

---

## EDJ7cPZk7V

- GT: Reject (avg 6.4)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

{
  "summary": "This paper studies catastrophic forgetting from an example-level perspective and shows a strong empirical relationship between an example’s learning speed on its original task and its later susceptibility to forgetting in continual learning. Building on this observation, it proposes Goldilocks, a simple replay-buffer sampling strategy that removes examples learned too quickly and too slowly, preferentially retaining mid-speed examples. The paper is well motivated and experimentally extensive: it evaluates the correlation itself across datasets/settings and shows that Goldilocks improves a range of replay-based continual learning methods on standard image benchmarks. Overall, the work’s main value is a clear empirical characterization plus a lightweight, practical intervention. The main reservations are that the conceptual novelty is somewhat incremental relative to prior example-level forgetting work, the mechanistic explanation for why mid-speed examples are optimal remains underdeveloped, and the strongest comparative claims (especially “state of the art”) are broader than what the current baseline set fully supports.",
  "strengths": [
    "The paper provides convincing empirical evidence that learning speed is strongly correlated with later forgetting in continual learning, including both quantitative correlations and intuitive epoch-wise classification visualizations.",
    "Goldilocks is simple and computationally lightweight: it requires only a per-example score accumulated during training and can be plugged into existing replay-based methods without changing their optimization logic.",
    "Experimental coverage is broad within image continual learning: multiple datasets (CIFAR-10/100, TinyImageNet), both task- and class-incremental settings, several architectures/optimizers/training schedules, and many replay-based CL methods are tested.",
    "The method appears robust across many settings, and the paper includes useful ablations showing similar qualitative behavior under changes in architecture, optimizer, learning rate, regularization, epochs, and task splits.",
    "Goldilocks is shown to be compatible with a diverse set of replay-based methods (e.g., ER, GEM, A-GEM, DER/DER++, X-DER, RPC, FDR), suggesting the contribution is reasonably orthogonal to existing buffer-utilization strategies.",
    "The paper is commendably explicit about an important limitation: the current learning-speed formulation is not suitable for single-epoch/streaming continual learning."
  ],
  "weaknesses": [
    "The core conceptual novelty is moderate. Prior work on example-level forgetting/forgettability is already relevant, and while this paper makes a useful continual-learning-specific connection via learning speed, the distinction from that literature could be articulated more sharply.",
    "The paper is almost entirely empirical and offers limited mechanistic or theoretical grounding for why mid-speed examples should be the best replay candidates. The simplicity-bias discussion is plausible, but still largely intuitive.",
    "The paper’s strongest performance claims are overstated relative to the comparison set. In particular, the abstract/introduction’s 'state-of-the-art' language is not fully justified without broader comparison to stronger or more recent replay/sample-selection baselines.",
    "Hyperparameter selection for the quick/slow filtering percentages is not fully frictionless in practice. Although the paper argues that a broad range works and suggests defaults, the proposed RotNet-based tuning route adds extra complexity and compute.",
    "The method is currently limited to multi-epoch task training. Since online/streaming continual learning is an important setting, this significantly constrains applicability.",
    "Most evaluations are on standard small-to-medium image classification benchmarks. The paper does not test larger-scale, domain-incremental, or non-vision settings, so external validity remains somewhat unclear.",
    "Although Appendix G compares learning speed with alternative scores visually, there is no direct method-level ablation showing whether Goldilocks built on those alternative scores would perform better or worse than the proposed learning-speed version."
  ],
  "nice_to_haves": [
    "A direct Goldilocks ablation using alternative per-example scores (e.g., uncertainty, loss, c-score) would strengthen the claim that learning speed is the most useful signal for buffer construction, not just for visualization/correlation.",
    "A more systematic study of fixed default hyperparameters versus tuned ones would help clarify real-world usability.",
    "Failure-case analysis would be helpful: identifying settings where Goldilocks offers little benefit, or where the learning-speed/forgetting relationship weakens.",
    "Evaluation on larger-scale or domain-shift benchmarks would strengthen the significance of the findings."
  ],
  "novel_insights": "The most interesting insight is not only that slower-learned examples are more forgettable, but that replay value is non-monotonic: the best examples to retain are often neither the easiest nor the hardest, but those in an intermediate ‘Goldilocks zone’. The paper also suggests an appealing perspective that stronger continual learners (via larger buffers or better architectures) effectively expand retention toward progressively harder/slower examples, which gives a useful way to think about forgetting as structured rather than arbitrary.",
  "missed_related_work": [
    "Online Continual Learning with Maximally Interfered Retrieval (MIR) as a relevant replay-sampling baseline to discuss or compare against.",
    "Experience Replay with Data Resampling for Continual Learning (CLS-ER) as a complementary replay-buffer optimization approach.",
    "A broader forgetting-focused survey could help contextualize the paper’s characterization contribution.",
    "A recent class-incremental learning survey may help position the appendix CIL results."
  ],
  "suggestions": [
    "Temper the 'state-of-the-art' framing unless broader and more up-to-date sampling/replay baselines are included.",
    "Clarify more precisely how the contribution differs from prior example-level forgetting/forgettability work, especially Toneva et al. and Maini et al., beyond simply moving to a continual-learning setting.",
    "Strengthen the explanation for why mid-speed examples are optimal, ideally with either a simple theoretical model or deeper empirical mechanism analysis (e.g., gradient interference, representation drift, or task overlap).",
    "Add direct comparisons to additional replay selection baselines where feasible, especially stronger/more recent methods.",
    "Include a practical ablation showing how well a fixed default choice such as quick=slow=20% works across datasets/methods without any extra tuning, compared with the RotNet-tuned version.",
    "If possible, add a method-level comparison where Goldilocks uses alternative scoring metrics instead of learning speed.",
    "Discuss more explicitly the computational overhead of hyperparameter tuning, even if the per-example storage cost of the final method is small."
  ],
  "novelty": 6.4,
  "technical_soundness": 7.9,
  "empirical_support": 8.4,
  "significance": 7.2,
  "clarity": 8.1
}

---

## 9AtlhmFVDi

- GT: Reject (avg 5.5)
- Predicted: Accept (5.6/10)
- Match: No

### Final Review

{
  "summary": "This paper studies whether standard Transformers, given linearly embedded 3D coordinates, can learn Euclidean-distance-based attention without specialized SE(3)-equivariant architecture. The core contribution is a theoretical argument that LayerNorm combined with simple coordinate embeddings can make attention scores approximate a negative squared distance, yielding Gaussian distance filtering. The paper then validates this in controlled simulated-point experiments and in protein masked-token pretraining, and shows downstream gains on GO function prediction when structure is included. Overall, the paper has a genuine and interesting conceptual contribution: it offers a plausible mechanism for how vanilla attention can encode spatial locality. The experiments support the basic claim that coordinate-augmented Transformers can exploit 3D structure. However, the empirical case is not yet strong enough for the broader comparative claims made in the paper. In particular, comparisons to modern structural baselines are limited, results appear to be from single runs without uncertainty estimates, and some claims about SE(3)-invariance and outperforming custom structural models are stronger than what is fully established by the evidence.",
  "strengths": [
    "The paper provides a novel theoretical insight: LayerNorm can induce approximately quadratic dependence from linear coordinate embeddings, helping standard attention approximate functions of squared Euclidean distance.",
    "The simulated experiments are well aligned with the theory and directly test key predictions, including that Gaussian-like distance filtering is the easiest form for the model to learn.",
    "The empirical finding that roughly n+2 head dimensions suffice to learn distance in R^n is concrete and potentially useful for model design.",
    "Protein experiments show a substantial practical benefit from adding coordinates during masked-token pretraining, including large improvements in training/validation perplexity and downstream GO prediction on the chosen setup.",
    "The attention analysis in the coordinate-aware protein model provides supportive evidence that early layers learn local 3D distance-sensitive attention patterns.",
    "The paper is generally clear in motivation and successfully connects the theory to practical protein modeling."
  ],
  "weaknesses": [
    "The main theoretical argument relies on restrictive assumptions, especially small coordinates and a simplified first-layer setting with effectively identity-like Q/K mappings. While the paper discusses scaling and provides appendix derivations, it does not fully quantify the practical error regime or establish that training reliably reaches this mechanism in realistic models.",
    "The paper sometimes phrases the result as learning SE(3)-invariant structure processing, but the model does not have architectural SE(3) invariance/equivariance. In practice, robustness to rotations/translations is encouraged through recentering/rescaling and random rotation augmentation rather than guaranteed by design.",
    "The empirical comparison is limited for the strength of the claims. The main downstream comparisons are against DeepFRI/DeepCNN-style baselines, which do not adequately support broad statements such as outperforming custom structural models in general.",
    "The protein experiments appear to be single-run results without variance estimates or multiple seeds, making it difficult to assess robustness of the reported gains.",
    "The protein attention-vs-distance analysis is informative but partly relies on controlled interventions (fixing amino acids/positions), and the appendix indicates noisier behavior on unmodified inputs; this weakens the claim that clean Gaussian distance dependence is directly visible in realistic settings.",
    "Experiments are relatively small-scale (~35K proteins, ~33M-parameter pretraining model), so it remains unclear how strongly the conclusions transfer to larger modern protein modeling regimes."
  ],
  "nice_to_haves": [
    "A more explicit quantitative bound or practical rule for the coordinate scaling regime in which the approximation is accurate.",
    "Analysis of whether different heads learn distinct Gaussian widths/spatial resolutions, as suggested by the theory.",
    "Ablations comparing the proposed embedding families (linear, trigonometric, quadratic-inspired) rather than only using the linear version in the main empirical studies.",
    "Runtime/memory comparisons versus structural GNNs or equivariant Transformers to support the claimed efficiency advantage of vanilla Transformers with FlashAttention-style implementations.",
    "Validation on an additional non-protein 3D domain, such as molecules or point clouds, to demonstrate generality beyond proteins."
  ],
  "novel_insights": "The most valuable insight is that vanilla Transformer attention may not need explicit distance edges or equivariant machinery to become geometry-aware: LayerNorm itself can transform simple coordinate embeddings into approximately quadratic interactions, making Gaussian distance filtering emerge naturally. This is a useful mechanistic explanation for why coordinate-fed Transformers can work at all, and it helps bridge empirical successes of coordinate-aware Transformer systems with a more interpretable mathematical picture.",
  "missed_related_work": [
    "Geometric Vector Perceptrons / GVP-GNN for protein structure modeling (Jing et al., 2021) would be a useful related comparison/discussion point.",
    "RoseTTAFold (Baek et al., 2021) would strengthen the protein-structure related work discussion.",
    "More recent structural protein baselines such as GearNet could be discussed and, if feasible, compared against.",
    "Equivariant Transformer baselines such as SE(3)-Transformer / Equiformer-style models would help contextualize the claims."
  ],
  "suggestions": [
    "Temper the comparative claims in the abstract and conclusion: the current results support that standard Transformers can effectively use coordinates and can outperform the specific older baselines tested, but not yet that they generally outperform custom structural models.",
    "Clarify throughout that the method learns approximate geometry sensitivity and augmentation-supported SE(3) robustness, rather than providing architectural SE(3) invariance/equivariance.",
    "Add uncertainty estimates via multiple seeds for the main downstream results and, ideally, for key simulated experiments as well.",
    "Strengthen empirical evaluation with at least one or two modern structural baselines on the same task, or narrow the claims if such comparisons are infeasible.",
    "Provide a clearer quantitative discussion of the small-coordinate assumption and scaling sensitivity, ideally translating the appendix analysis into practical guidance in the main text.",
    "Include a head-level analysis to verify whether different heads indeed learn different spatial variances/resolutions, as predicted by the theory.",
    "If space permits, add an additional benchmark beyond GO function prediction to show broader utility of the approach."
  ],
  "novelty": 7.9,
  "technical_soundness": 6.8,
  "empirical_support": 6.3,
  "significance": 7.0,
  "clarity": 7.8
}

---

## aMBSY2ebPw

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

{
  "summary": "This paper provides a careful empirical re-examination of the claim that long-context LLMs can learn translation of an extremely low-resource language from a grammar book. Its main contribution is a useful ablation that separates grammar-book parallel examples from non-parallel grammatical exposition, showing across Kalamang and two additional low-resource languages that most translation gains come from the parallel examples rather than the explanatory text. The paper further shows that a standard MT fine-tuning setup on the same parallel data is competitive with long-context prompting, and that grammatical information can still help on more linguistically aligned tasks such as grammaticality judgment and interlinear gloss prediction, especially via a typological-feature prompt. Overall, the paper is timely, well motivated, and practically relevant. The core negative result for translation is convincing, but the strength of some conclusions is limited by small Kalamang evaluation data, the heavy reliance on one main long-context model, and the lack of uncertainty estimates for many key comparisons. The typology-based task results are interesting, though more modest and somewhat under-ablated.",
  "strengths": [
    "Strong central ablation: splitting grammar books into parallel examples and non-parallel explanatory text directly addresses the key confound in prior work and is the paper's main empirical contribution.",
    "Findings are supported across more than one language: beyond Kalamang, experiments on Nepali and Guarani with FLORES make the translation conclusion more credible.",
    "Useful practical takeaway: fine-tuning a smaller MT model on the extracted parallel data performs competitively with long-context LLM prompting, reinforcing the paper's recommendation to prioritize parallel data collection for MT.",
    "The regression analysis linking translation quality to test-set type coverage, and not prompt length, adds a valuable quantitative perspective to the main claim.",
    "The paper is appropriately nuanced in showing that grammar is not useless overall: typological prompting helps on grammaticality judgment and gloss prediction, supporting the broader 'task-appropriate data' message.",
    "The work is well situated relative to MTOB and related low-resource MT literature, and the motivation is clear and important for XLR NLP and language documentation."
  ],
  "weaknesses": [
    "The Kalamang translation test set remains very small (100 examples), and several headline differences discussed in the paper are small enough that practical significance is uncertain.",
    "Most long-context translation evidence comes from a single primary API model (Gemini-1.5-Flash); Llama results are only partial because of context limits, so the title-level claim about 'LLMs' is broader than the empirical base.",
    "Tables 2-5 report point estimates without confidence intervals or direct significance testing for key pairwise comparisons, which is especially important given the small observed differences in some settings.",
    "The manual extraction of BOOKpara vs. BOOKnon-para is central, but the extraction criteria are not specified in enough procedural detail to maximize reproducibility.",
    "Some conclusions about grammar books are stated a bit too broadly relative to the evidence, which is mainly about descriptive linguistic grammars of the tested kind and about translation specifically, not all possible grammar resources or formats.",
    "The typological prompting results, while promising on linguistic tasks, are relatively modest and not fully isolated by ablation; the added benefit over BOOKpara is small in some cases."
  ],
  "nice_to_haves": [
    "Evaluate at least one additional long-context model to test whether the negative translation result is model-specific or more general.",
    "Add bootstrap confidence intervals or paired significance tests for CHRF++ and the task metrics in the main tables.",
    "Provide a more explicit, reproducible protocol for extracting parallel versus non-parallel material from grammar books.",
    "Include a learning-curve analysis over number of parallel examples to quantify how much parallel data is needed before it surpasses grammar-book prompting.",
    "Test more structured forms of grammatical knowledge for translation, e.g., concise rule prompts or normalized grammar summaries, to distinguish content limitations from formatting/retrieval limitations."
  ],
  "novel_insights": "The paper's most compelling insight is not merely that grammar books help less than expected, but that their apparent translation benefit can be largely explained by the presence of embedded bilingual examples and resulting vocabulary/type coverage rather than by successful use of grammatical exposition. At the same time, the paper usefully reframes the issue by showing that grammatical knowledge does become helpful when the task is aligned with it and when the information is presented in a more machine-usable form, such as typological features. This task-dependent view is a meaningful contribution.",
  "missed_related_work": [
    "Cotterell et al. (2023) on whether LLMs can learn morphology from in-context examples, which may be relevant to the grammaticality/glossing discussion.",
    "Gu et al. (2018) on meta-learning for low-resource NMT as another adaptation paradigm for scarce-parallel settings.",
    "Arivazhagan et al. (2019) on massively multilingual NMT as foundational context for multilingual transfer and fine-tuning baselines like NLLB."
  ],
  "suggestions": [
    "Temper the broadest claims to specify that the evidence shows descriptive grammar-book explanations are not effective for translation in the tested setups, rather than implying a blanket statement about all grammar resources or all LLMs.",
    "Report uncertainty for the main comparisons, especially BOOKall vs BOOKpara and typology-related gains, since several differences are small.",
    "Clarify the extraction methodology for book subsets in more operational detail so others can reproduce the ablation on new languages.",
    "If feasible, add another long-context model or a stronger Gemini variant to strengthen the generalization of the central claim.",
    "Expand the analysis of why grammatical explanations fail for translation, for example by testing shorter structured rule prompts or retrieval-oriented variants.",
    "Clarify the comparison language when discussing the small '+0.7' and '-0.3' effects so readers can immediately map the text to the relevant rows/settings in Table 2."
  ],
  "novelty": 7.8,
  "technical_soundness": 7.2,
  "empirical_support": 7.0,
  "significance": 7.8,
  "clarity": 8.0
}

---

## LBl7Hez0fF

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

{
  "summary": "This paper studies hallucination in large vision-language models through the lens of vision-feature instability and proposes Visual and Textual Intervention (VTI), a training-free inference-time method that steers hidden representations in both the vision encoder and text decoder. The paper’s core empirical finding is that mild image perturbations induce a long-tailed set of unstable visual features, and that averaging features across perturbations reduces hallucination. VTI approximates this effect by extracting PCA-based steering directions from a small calibration set and applying them at inference. Across POPE, CHAIR, and MMHAL-Bench, and on LLaVA-1.5, InstructBLIP, and Qwen-VL, the method shows consistent reductions in hallucination and generally improves over strong decoding baselines such as VCD and OPERA. Overall, the paper has a useful mechanistic perspective and promising empirical results, but some claims are stronger than the evidence supports, and several methodological details and evaluations remain incomplete.",
  "strengths": [
    "The paper offers a meaningful mechanistic perspective specific to LVLMs: hallucination is connected to instability in visual representations passed from a separately pretrained vision encoder to a text decoder.",
    "The empirical evidence for the stability-hallucination connection is reasonably compelling: Figure 2 shows a long-tailed distribution of unstable features and that averaging over perturbations reduces hallucination, while Figure 5 shows VTI improves feature stability under multiple perturbation types.",
    "VTI is practically attractive as a training-free test-time method and avoids the heavy cost of naive feature averaging over many perturbed inputs.",
    "Results are strong and consistent across several benchmarks: POPE, CHAIR, and MMHAL-Bench all improve, often by clear margins, and gains are shown on three LVLM families.",
    "The separate visual-only and text-only ablations are valuable and support the claim that the two interventions are complementary.",
    "The paper includes some interpretability-oriented analysis, e.g., textual intervention increasing attention to vision tokens and decreasing text-to-text reliance."
  ],
  "weaknesses": [
    "The paper overstates task/dataset agnosticism. While the intervention directions are fixed across benchmarks, the intervention strengths are not: Appendix A uses different α, β settings for different evaluations (e.g., 0.4 on CHAIR and 0.9 on MMHAL-Bench). This does not invalidate the method, but the claim should be softened.",
    "There is an internal inconsistency in the number of calibration examples used to compute directions: the introduction mentions 50 examples, Section 5 states 80 examples, and Appendix C again refers to 50 examples for textual direction construction. This hurts reproducibility and should be clarified.",
    "The causal claim is stronger than the evidence. The paper demonstrates correlation between feature instability and hallucination and provides supportive interventions, but it does not decisively establish that instability is the primary cause rather than one contributing factor.",
    "The justification for using the top PCA direction as the transferable steering vector is heuristic. Table 5 helps, but the paper does not fully explain why one principal direction should capture the relevant robustification effect, nor whether this generalizes beyond the reported setting.",
    "Important ablations are missing, especially which layers matter most for intervention, sensitivity to the number and choice of calibration examples, and sensitivity to the masking ratio used to construct the visual direction.",
    "The textual direction depends on externally generated hallucinated captions (GPT-based generation), but the paper gives only limited detail about this construction and little analysis of how sensitive results are to that data-generation procedure.",
    "There are no confidence intervals, multi-seed results, or statistical significance analyses, which weakens confidence in the robustness of the reported gains.",
    "Evaluation focuses on hallucination-oriented benchmarks; the paper does not directly quantify whether VTI harms standard task performance on broader VQA/captioning benchmarks, even though it argues that recall and generation length are reasonably preserved on CHAIR.",
    "Baseline coverage is good but not exhaustive for training-free hallucination mitigation; some additional recent or adjacent methods could strengthen the empirical positioning."
  ],
  "nice_to_haves": [
    "Human evaluation to confirm that automatic hallucination metrics align with genuinely better outputs.",
    "A wall-clock and memory overhead analysis for both direction precomputation and inference-time intervention.",
    "Failure case analysis showing when VTI does not help or may introduce errors.",
    "A more detailed semantic analysis of the learned steering directions, including per-layer effects and whether certain layers dominate the gains."
  ],
  "novel_insights": "The strongest contribution is not the raw steering mechanism itself, which is an adaptation of representation/activation steering ideas, but the paper’s LVLM-specific framing: hallucinations can arise from instability in visual features passed into a separately pretrained language decoder. The paper makes this concrete by showing that smoothing over perturbations reduces hallucination and that a learned latent-space approximation to this smoothing effect is effective. The complementary behavior of visual and textual interventions across benchmark categories is also a useful observation that may guide future multimodal control methods.",
  "missed_related_work": [
    "Activation Steering in Large Language Models (Turner et al., 2024) as additional methodological context for activation/representation steering.",
    "VaLiD: Mitigating the Hallucination of Large Vision-Language Models by Visual Layer Fusion Contrastive Decoding (Wang et al., 2024), if contemporaneous with the submission timeline."
  ],
  "suggestions": [
    "Clarify the calibration-set size discrepancy (50 vs 80) and provide a precise, unified description of how visual and textual directions are constructed.",
    "Temper the task-agnostic claim or demonstrate a single fixed α, β setting across all benchmarks and models.",
    "Add robustness analyses: multiple random seeds, confidence intervals, and sensitivity to the chosen calibration examples and calibration-set size.",
    "Include ablations on intervention location (which layers, vision vs text subsets) and on the masking ratio used during visual direction construction.",
    "Provide stronger evidence for the proposed mechanism, e.g., controls using random steering directions or alternative non-PCA directions, to show that the specific computed directions matter.",
    "Evaluate the effect of VTI on non-hallucination capabilities using standard VQA/captioning benchmarks to quantify any accuracy-faithfulness trade-off.",
    "Expand reproducibility details for textual direction generation, including prompts, filtering, and exact use of GPT-generated hallucinated captions.",
    "If space permits, compare against a broader set of training-free baselines or explain why certain relevant methods were omitted."
  ],
  "novelty": 6.9,
  "technical_soundness": 7.0,
  "empirical_support": 7.6,
  "significance": 7.5,
  "clarity": 7.2
}

---

## WRKVA3TgSv

- GT: Reject (avg 3.0)
- Predicted: Accept (5.6/10)
- Match: No

### Final Review

{
  "summary": "This paper introduces GraphModQA, a benchmark for evaluating LLMs on dynamic graph modification tasks, and studies whether modern LLMs can maintain and update graph state under sequences of node/edge additions and deletions. The paper makes a useful empirical point: compared with earlier work focused on static graph properties, current frontier models are much stronger on many basic graph-property tasks, but performance drops substantially on modification tasks, especially with adjacency-matrix encodings and longer modification sequences. The proposed MAP prompting method—requiring intermediate graph outputs—consistently improves performance and is a practically useful intervention. Overall, the benchmark and empirical findings are interesting and likely valuable to the community. The main limitations are that the evaluation remains narrow (small synthetic graphs, undirected/unweighted settings, limited prompting/baseline comparisons), some claims are stronger than the evidence supports, and the empirical analysis would benefit from more rigorous contextualization and ablations.",
  "strengths": [
    "Well-motivated problem formulation: graph modification is a natural and interpretable extension beyond static graph-property QA, and it captures dynamic state-tracking abilities that prior graph benchmarks do not test directly.",
    "GraphModQA is systematically constructed, with clear variation across encoding, modification type, number of steps, final question, and prompting strategy; the dataset construction algorithms are described in detail and appear reproducible.",
    "The paper identifies adjacency-matrix encoding as a particularly challenging and underexplored setting for LLM graph reasoning, and the results convincingly show a large gap between this encoding and more textual encodings.",
    "The core empirical finding is meaningful: modern LLMs are much better than older baselines on many static graph tasks, yet still struggle on dynamic graph modification, especially as the number of modifications increases.",
    "MAP prompting is simple, intuitive, and empirically effective; the gains across multiple models and modification types make it a useful practical takeaway.",
    "The appendix provides substantial supporting analysis, including error taxonomies, graph-type experiments, and size/density ablations, which help characterize model failure modes beyond top-line accuracy.",
    "The paper includes interesting model-specific observations, such as the strong o1-mini performance on some harder modification tasks and the fact that CoT prompting can hurt some models."
  ],
  "weaknesses": [
    "The evaluation is restricted to very small synthetic graphs (primarily 7-20 nodes), which substantially limits claims about practical relevance or scalability to realistic graph settings.",
    "All main experiments are on undirected, unweighted, synthetic graphs; the paper motivates applications such as social networks and knowledge graphs, but does not test on any real-world graphs or richer graph types.",
    "The empirical setup lacks contextual baselines beyond direct prompting of black-box LLMs. In particular, there is no comparison to tool-augmented or program-generation approaches, which is important because these tasks are trivially solvable algorithmically once represented symbolically.",
    "Some framing is overstated. In particular, saying modern LLMs 'excel' at basic graph property tasks is too strong given that several tasks/encodings still show middling accuracy, especially edge counting.",
    "The mechanism behind MAP prompting is not cleanly isolated. The paper shows that it helps, but does not disentangle whether gains come from explicit intermediate-state printing, extra computation time, output-format regularization, or some other factor.",
    "Several interesting findings are underexplained, especially why CoT hurts o1-mini while helping some other models.",
    "The paper does not fully probe the source of adjacency-matrix difficulty. Explanations such as dense numerical representation and implicit renumbering are plausible, but not tested with targeted controlled ablations.",
    "For the Print Graph task, the evaluation protocol is only implicitly described; the paper should state explicitly whether scoring is exact-match over the full structured output or computed in another way."
  ],
  "nice_to_haves": [
    "Report token lengths / prompt sizes for different encodings and prompting methods to help interpret whether failures are partly due to sequence length or representational density.",
    "Include confidence intervals or significance testing for key MAP-vs-zero-shot comparisons, even if the overall trends already look qualitatively strong.",
    "Add a controlled ablation separating 'print intermediate matrices' from 'reason step by step without printing' to better explain MAP's gains.",
    "Explore larger graphs and, if API cost prevents broad coverage, at least provide a small-scale scaling study beyond 20 nodes for more than one model.",
    "Include directed and/or weighted graph variants, or discuss them explicitly as future benchmark extensions.",
    "Provide one or two richer post-modification queries (e.g., shortest path or connectivity after edits) to test reasoning on the modified graph beyond state tracking."
  ],
  "novel_insights": "A particularly valuable insight is that dynamic graph modification exposes a qualitatively different weakness from static graph QA: many models can answer local/global graph-property questions reasonably well, yet fail when they must maintain a mutable structured state across sequential edits. The adjacency-matrix results sharpen this further by suggesting that failures are not only about graph reasoning in the abstract, but also about precise manipulation of dense symbolic structure. The error analyses add an interesting nuance: models often make near-miss errors (e.g., changing adjacent indices or preserving a perceived graph template), which hints that their failures are structured rather than random and may be amenable to better intermediate representations or neuro-symbolic/tool-based methods.",
  "missed_related_work": [
    "Austin et al. (2022), 'Show Your Work: Scratchpads for Intermediate Reasoning with Language Models' — relevant grounding for MAP-style intermediate-state outputs.",
    "Chen et al. (2023), 'Program-of-Thoughts Prompting' — conceptually related to using intermediate symbolic representations to improve reasoning.",
    "A broader recent survey on LLMs for graphs could help situate the contribution within the rapidly growing literature.",
    "If applicable at submission time, newer/concurrent work on dynamic graph reasoning or graph editing benchmarks/methods could be discussed to clarify overlap and distinctions."
  ],
  "suggestions": [
    "Temper the claim that modern LLMs 'excel' at static graph property tasks, and frame the result more precisely as substantial improvement over prior PaLM-era results with strong but task-dependent performance.",
    "Clarify the exact evaluation criterion for Print Graph outputs and whether scoring is strict exact match over the full graph representation.",
    "Add at least one nontrivial contextual baseline, ideally a tool-augmented setting where the model can generate or call simple graph-manipulation code, to distinguish reasoning limits from text-only manipulation limits.",
    "Expand the analysis of MAP prompting with a controlled ablation against a scratchpad/step-by-step prompt that does not require printing full intermediate matrices.",
    "Investigate the o1-mini CoT degradation more carefully; even a focused diagnostic experiment would strengthen the paper.",
    "If possible, include at least a modest scaling experiment on larger graphs and/or a real-world graph subset to better support claims of broader applicability.",
    "Test targeted hypotheses about adjacency-matrix difficulty, such as explicit versus implicit node renumbering, or alternative matrix formatting choices.",
    "Discuss more explicitly the scope limitations of the benchmark: small graphs, synthetic data, undirected/unweighted setting, and black-box prompting-only evaluation."
  ],
  "novelty": 7.3,
  "technical_soundness": 7.1,
  "empirical_support": 6.8,
  "significance": 7.4,
  "clarity": 8.0
}

---

## EO8xpnW7aX

- GT: Accept (Oral) (avg 8.0)
- Predicted: Accept (6.8/10)
- Match: Yes

### Final Review

{
  "summary": "This paper introduces SymmetricDiffusers, a discrete diffusion framework tailored to distributions over finite symmetric groups. The core design is well motivated: use card-shuffling random walks as the forward process, with riffle shuffle selected based on classical mixing-time results, and parameterize reverse transitions with either inverse shuffles or a new generalized Plackett-Luce (GPL) family. The paper also derives exact TV-distance formulas for riffle-shuffle marginals and uses them to guide diffusion length and reverse-step merging. Empirically, the method is strong on permutation-heavy tasks, especially longer-sequence sorting and harder jigsaw settings, and is competitive on TSP. Overall, the work is novel and technically interesting, with a meaningful blend of group-theoretic theory and modern generative modeling. The main limitations are the gap between the expressive GPL theory and practical learnability/scalability, sensitivity of the denoising schedule, incomplete uncertainty reporting, and limited runtime/scaling analysis.",
  "strengths": [
    "Novel problem setting and formulation: the paper tackles diffusion modeling directly on symmetric groups rather than on relaxed permutation representations or generic discrete alphabets.",
    "Strong theoretical grounding for the forward process: the use of random-walk theory on finite groups and the choice of riffle shuffle are convincingly motivated by mixing-time/cutoff results.",
    "Meaningful reverse-process contribution: the generalized Plackett-Luce distribution is a genuine extension of PL, and the paper proves PL is limited while GPL is strictly more expressive.",
    "The TV-distance analysis for riffle shuffles is useful and nontrivial; it supports a principled, if partly heuristic, approach to choosing diffusion length and merged reverse steps.",
    "Experimental breadth is good: sorting, jigsaw, and TSP demonstrate applicability across perceptual and combinatorial domains.",
    "Empirical results are strongest where the method is most needed: on longer sorting sequences and harder jigsaw instances, the approach clearly outperforms differentiable sorting baselines.",
    "The paper is generally reproducible: architecture details, schedules, implementation choices, and training hyperparameters are extensively documented in the appendices."
  ],
  "weaknesses": [
    "The main theoretical expressiveness result for GPL is representational rather than algorithmic: it shows existence of a reverse process capable of modeling arbitrary distributions, but does not address whether such distributions are learnable with the practical parameterizations, objectives, and small numbers of reverse steps used in experiments.",
    "There is a clear theory-practice gap around GPL expressiveness. The constructive argument underlying universality is not directly reflective of the practical settings used in the paper, so the implications for finite-step diffusion in realistic regimes remain unclear.",
    "Scalability is still limited. The method has O(n^2) costs from GPL parameterization and transformer attention, and empirical performance deteriorates substantially at larger n (e.g., very low exact accuracy at n=200).",
    "The denoising schedule is important but highly sensitive. The paper provides helpful heuristics based on TV distance, yet the specific thresholds used are still empirical and not theoretically justified; Table 7 shows that poor schedule choices can collapse performance.",
    "The training objective relies on Monte Carlo estimation because the usual low-variance KL form is unavailable in general. The paper explains this constraint well, but does not quantify the resulting variance, optimization difficulty, or training overhead.",
    "Empirical reporting lacks uncertainty estimates such as standard deviations or confidence intervals, despite averaging over multiple runs. This makes it hard to judge the robustness of modest gains, especially on TSP.",
    "For n=200 in sorting, the experiments switch from GPL to PL and random-timestep training for efficiency. This is clearly stated in the paper, but it weakens the direct empirical support for the paper's main GPL-based story at larger scales.",
    "Runtime, memory, and wall-clock comparisons are missing. Given the paper's claims about efficiency from schedule merging and the use of beam search at inference, explicit computational comparisons would materially strengthen the evaluation."
  ],
  "nice_to_haves": [
    "A more systematic ablation over diffusion length T, not just denoising schedules, would better validate the mixing-time-based design rules.",
    "Additional analysis of failure cases and qualitative reverse trajectories would help readers understand what kinds of permutations are hard for the model.",
    "Stronger TSP benchmarking on larger instances and/or against more neural combinatorial optimization baselines would improve positioning.",
    "A practical study of when GPL is preferable to PL as n grows would clarify the real operating regime of the proposed reverse family."
  ],
  "novel_insights": "The most compelling insight is that diffusion over permutations becomes far more natural when cast as a random walk on the group itself rather than as diffusion over a generic discrete encoding. This lets the authors import classical results on card shuffling—especially riffle-shuffle cutoff behavior—to inform modern diffusion design. Equally interesting is the role of GPL: it reframes reverse denoising as staged autoregressive selection over positions, giving a richer family than standard PL while preserving permutation structure. The paper's strongest conceptual contribution is thus the bridge between finite-group random-walk theory and practical generative modeling on permutation spaces.",
  "missed_related_work": [
    "Pointer Networks (Vinyals et al., 2015) as a classic permutation-generating baseline, particularly relevant to sorting/TSP-style tasks.",
    "Broader neural combinatorial optimization references for TSP, such as POMO and related attention-based solvers, to strengthen task positioning.",
    "Set-structured architecture references such as Deep Sets or Set Transformer may be useful as architectural context, though they are not essential omissions for the core contribution.",
    "A broader survey on learning plus combinatorial optimization could help contextualize the TSP application."
  ],
  "suggestions": [
    "Clarify more explicitly that GPL universality is a representational result and discuss what it does and does not imply for practical finite-step learning.",
    "Add uncertainty estimates (std/CI) to the main tables and, where gains are small, comment on run-to-run variability.",
    "Report runtime, memory, and decoding cost, including the effect of denoising schedules and beam search.",
    "Provide a systematic ablation over diffusion length T and a more thorough empirical validation of the TV-based scheduling heuristics.",
    "Analyze larger-scale behavior: identify whether degradation is primarily due to reverse-family limitations, optimization difficulty, schedule choice, or model capacity.",
    "Discuss more explicitly the tradeoff between GPL and PL, especially since the largest sorting experiment uses PL rather than GPL.",
    "If space allows, add stronger permutation-generation baselines, especially autoregressive permutation models and additional TSP neural baselines."
  ],
  "novelty": 8.8,
  "technical_soundness": 7.9,
  "empirical_support": 7.4,
  "significance": 8.0,
  "clarity": 8.0
}

---

## OXIIFZqiiN

- GT: Reject (avg 1.5)
- Predicted: Reject (2.6/10)
- Match: Yes

### Final Review

{
  "summary": "The paper proposes IGCP for patch representation learning and reports improved patch description generation over prior baselines, with an ablation suggesting that its loss components contribute meaningfully. However, the submission has major credibility and substantiation issues. Most importantly, the title and framing emphasize a dual-modal/image-guided/visual-prompt setting, but the paper does not actually define or evaluate any visual modality. In addition, the extensive theoretical development (measure theory, RKHS, quantum information, free probability, SGLD, phase transitions) is not convincingly connected to the implemented model or training procedure, and several claimed capabilities—such as unifying predictive and generative tasks and exhibiting domain generalization—are not empirically demonstrated. The empirical results on patch description generation are promising, but the paper in its current form overclaims relative to what is actually specified and validated.",
  "strengths": [
    "The paper addresses an important software-engineering problem: learning useful representations of code patches and generating patch descriptions.",
    "On the reported benchmark, IGCP improves over listed baselines on standard generation metrics, with notable gains in BLEU, ROUGE-L, and METEOR.",
    "The ablation study is directionally useful: removing PDM and PDG losses leads to clear performance drops, suggesting these objectives matter.",
    "The paper attempts to think beyond a single-task formulation and aims for a unified patch-text representation framework, which is a worthwhile research direction."
  ],
  "weaknesses": [
    "There is a severe title/content mismatch. The paper is framed around 'dual-modal,' 'image-guided,' and 'visual prompts,' but the methodology and experiments do not actually include images, a visual encoder, or any visual prompting mechanism.",
    "The central theoretical apparatus is largely disconnected from the actual system. The paper introduces kernel mean embeddings, spectral analysis, quantum information bottleneck, free probability, SGLD convergence, and phase-transition claims, but does not clearly show how these ideas are instantiated in the implemented model.",
    "The submission lacks essential implementation details for reproducibility: architecture specifics, dimensions, optimization setup, training hyperparameters, whether SGLD is actually used, and how the losses are concretely defined and combined.",
    "Claims about unifying predictive and generative tasks are not supported empirically. The experiments only evaluate patch description generation; no predictive/classification task is reported despite being emphasized in the motivation and abstract.",
    "Claims of domain generalization are unsupported by the evidence presented. Evaluation appears limited to Java patches from one benchmark setup, with no cross-language or cross-domain transfer study.",
    "Several theoretical results appear to be standard background results rather than paper-specific innovations, and the paper currently presents them as if they were core novel contributions without demonstrating new algorithmic consequences.",
    "The empirical section lacks rigor expected for strong validation: no variance across runs, confidence intervals, or significance testing; no exact result tables beyond figures; and limited discussion of baseline implementation/fairness.",
    "Terminology is inconsistent in places (e.g., IGCP vs. IPPMF), which further reduces confidence in the clarity of the contribution."
  ],
  "nice_to_haves": [
    "Include qualitative examples of generated patch descriptions and failure cases.",
    "Provide a clearer architecture figure with layer types and tensor dimensions.",
    "Report computational cost, model size, and training/inference efficiency relative to baselines.",
    "Add human evaluation for commit-message quality/usefulness, since automatic metrics alone are limited."
  ],
  "novel_insights": "The most plausible underlying contribution is not the heavy mathematical formalism, but rather a patch-text learning framework with multiple objectives for matching/contrastive/generation signals. If the paper were rewritten around that concrete contribution, with the theory demoted to optional intuition or removed unless operationalized, it could become substantially more convincing. As written, the strongest signal is that there may be a reasonable empirical model underneath, but the presentation overextends far beyond what is actually evidenced.",
  "missed_related_work": [
    "GraphCodeBERT (Guo et al., 2021) as a structurally informed code-text pretraining baseline/reference.",
    "CoTexT (Phan et al., 2021) for multi-task code-text learning relevant to the paper's stated goal of unifying understanding and generation.",
    "Learning to Represent Programs with Graphs (Allamanis et al., 2018) as foundational graph-based code representation work.",
    "PatchSynth (Tang et al., 2024) as closely related patch-text pretraining work with overlapping goals."
  ],
  "suggestions": [
    "Retitle and reframe the paper to accurately match the actual content, unless a genuine visual modality is added and evaluated.",
    "Either operationalize the theoretical components in the implemented method or substantially trim them. In particular, explain precisely what is optimized during training and how each claimed theoretical concept affects model design.",
    "Provide complete implementation details: encoder/decoder architecture, dimensions, tokenization, losses, hyperparameters, optimizer, training schedule, hardware, and code availability.",
    "Add experiments on at least one predictive task if the claim is that the framework unifies predictive and generative patch analysis.",
    "Evaluate cross-domain or cross-language generalization if domain generalization is a headline claim.",
    "Strengthen empirical validation with multiple runs, confidence intervals or significance tests, exact-value tables, and clearer baseline fairness details.",
    "Compare against stronger and more modern code/text baselines where feasible.",
    "If the phase-transition, SGLD, or quantum bottleneck claims are kept, include direct empirical tests or ablations validating those specific claims."
  ],
  "novelty": 3.2,
  "technical_soundness": 2.8,
  "empirical_support": 4.2,
  "significance": 4.0,
  "clarity": 3.5
}

---

## s3IBHTTDYl

- GT: Accept (Poster) (avg 6.8)
- Predicted: Accept (6.6/10)
- Match: Yes

### Final Review

{
  "summary": "This paper studies whether language models can learn counting in a genuinely inductive way, i.e., generalize to larger cardinalities/longer sequences than seen in training. The core contribution is a carefully designed empirical framework that disentangles OOD-position, OOD-vocabulary, and OOD-cardinality, and uses it to compare multiple Transformer positional encodings as well as classical and modern recurrent architectures. The evidence supports a nuanced but important conclusion: standard Transformers do not reliably acquire inductive counting from in-domain supervision alone; success is highly dependent on positional encoding, task format, and depth, and even the strongest positive results are fragile under harder extrapolation. In contrast, simple RNNs/LSTMs generalize much more robustly, while modern recurrent architectures often underperform them. The paper is strong in experimental design and in its PE-specific mechanistic probing, but its empirical claims are weakened by high seed variance and by presenting best-of-5-seed results in the main tables without equally foregrounding median variability. Overall, this is a meaningful empirical contribution that sharpens the discussion around counting as a primitive operation and around the role of inductive biases in length generalization.",
  "strengths": [
    "Excellent task design: the paper carefully separates OOD-position, OOD-vocabulary, and OOD-cardinality, and the helper-token / shifted-start constructions are thoughtful ways to expose number vocabulary while withholding OOD cardinalities.",
    "Broad and informative architectural comparison: the study spans Transformers with multiple PE schemes, classical RNN/LSTM baselines, and modern recurrent models (S4, Mamba, RWKV), yielding a useful comparative picture.",
    "A substantive empirical finding is that traditional RNNs/LSTMs solve these counting tasks far more robustly than Transformers and often more robustly than modern recurrent alternatives.",
    "The PE analysis is a real strength: the paper does not stop at benchmark outcomes, but proposes concrete mechanisms for modular and selective counting and backs them with targeted probes (e.g., first-token-recognition tasks, PCA/attention analyses).",
    "The paper productively challenges the assumption in some formal/programmatic frameworks that counting is an easy primitive for Transformers; the shallow-model evidence makes that assumption less automatic.",
    "The paper is generally clear, with good motivating discussion and intuitive task illustrations that make the setup understandable despite the synthetic nature of the tasks.",
    "The BOS/first-token analyses in the appendix add credibility to the claim that specific inductive biases and symmetry-breaking mechanisms are central to success or failure."
  ],
  "weaknesses": [
    "The main empirical presentation relies on best-of-5-seed results, while the appendix shows substantial seed variance for several key Transformer+PE settings. This materially weakens claims that certain PE choices reliably enable generalization.",
    "Some positive claims, especially around RoPE on shifted-start counting, need stronger qualification. The paper itself shows that this success is fragile and degrades substantially as extrapolation becomes harder, so it should not be read as robust inductive counting.",
    "The study is restricted to shallow Transformers (1-4 layers). This is justified relative to the paper's critique of 'counting as a primitive,' but it still limits how broadly one can generalize the conclusions to larger modern LLMs.",
    "Some mechanistic explanations remain suggestive rather than definitive. For example, the RoPE recency-bias account for selective-counting failure is plausible and partially supported, but not fully established causally.",
    "The comparison across architectures is insightful but not perfectly controlled: Transformer sizes are fixed in one regime while recurrent architectures use a separate hyperparameter search space, so fairness of absolute performance comparisons should be interpreted with some caution.",
    "The tasks are synthetic and isolate counting well, but the paper does not directly test whether the observed counting abilities transfer to downstream tasks such as formal-language recognition or algorithmic reasoning, despite motivating the work via those connections."
  ],
  "nice_to_haves": [
    "Report median (or mean) performance with dispersion in the main tables, not only in the appendix.",
    "Move the RoPE extrapolation-fragility analysis (currently appendix) into the main paper.",
    "Include at least one larger Transformer depth/scale condition to test whether the qualitative conclusions persist beyond 4 layers.",
    "Add a small failure-mode analysis of OOD predictions (e.g., saturation, drift, periodic errors) to better characterize what 'non-inductive' counting looks like.",
    "Either test or soften the conclusion's suggestion that combining PE schemes may help, since this is currently speculative."
  ],
  "novel_insights": "The most compelling insight is not simply that Transformers struggle with length generalization, but that 'counting' itself decomposes into multiple subproblems with different inductive-bias requirements: symmetry breaking / first-token recognition, position-sensitive modular structure, and the ability to ignore positional information when token-identity-based attention is needed. This PE-dependent decomposition helps explain why different positional encodings succeed on different counting variants, and why there is no single uniformly strong Transformer configuration here. A second notable insight is that modern recurrent architectures do not automatically inherit the counting robustness of classical RNNs/LSTMs; the paper raises the interesting possibility that architectural concessions made for parallelizable training degrade the very recurrent dynamics that support inductive counting.",
  "missed_related_work": [
    "When Can Transformers Count to n? (Yehudai et al., 2024) — potentially relevant complementary theory on counting capacity.",
    "Transformers are RNNs: Fast Autoregressive Models with Linear Attention (Katharopoulos et al., 2020) — potentially relevant for framing links between attention variants and recurrent computation.",
    "From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers (Duan et al., 2024) — potentially relevant as a solution-oriented arithmetic/length-generalization comparison."
  ],
  "suggestions": [
    "Foreground robustness: present median/variance statistics in the main results and temper language about PE-specific 'success' accordingly.",
    "Qualify the shifted-start RoPE result as limited-range extrapolation rather than robust inductive counting, since harder train/test ratios already reveal collapse.",
    "If space permits, add one deeper/larger Transformer experiment to test whether the conclusions are about architecture or merely shallow compute budget.",
    "Strengthen the mechanistic claims where possible with more causal interventions, not only correlational probes.",
    "Clarify that the conclusions are strongest for controlled synthetic counting settings and shallow models, and more speculative for large pretrained LLMs.",
    "As future work, it would be valuable to test whether better counting on these probes predicts better behavior on downstream formal-language or algorithmic tasks."
  ],
  "novelty": 7.8,
  "technical_soundness": 7.2,
  "empirical_support": 7.0,
  "significance": 7.5,
  "clarity": 8.2
}

---

## FCBbh0HCrF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Reject (4.5/10)
- Match: No

### Final Review

{
  "summary": "The paper studies online vertical federated learning under an event-driven setting where only a subset of clients are naturally activated at each round, and adapts dynamic local regret (DLR) to this VFL scenario to handle non-convex and non-stationary streaming settings. The motivation is timely and the paper makes a meaningful step beyond the synchronous online-VFL assumption in prior work. Theoretical analysis is provided for the random partial-activation setting, and experiments show that DLR-based variants are more stable than OGD baselines under synthetic non-stationarity while reducing some communication/computation costs. However, the work has an important conceptual limitation: despite the event-driven framing, passive clients are still queried each round to provide embeddings, so the system is not fully asynchronous and the practical source of passive-client features is not sufficiently clarified. In addition, the theory covers random activation probabilities, whereas one of the main experimental settings uses data-dependent threshold-based event activation, leaving a noticeable theory-practice gap. Empirical validation is promising but still limited by synthetic event mechanisms, synthetic drift, and lack of multi-seed statistical reporting.",
  "strengths": [
    "Addresses a real and underexplored problem in online VFL: prior online-VFL work assumes synchronous feature arrival, while this paper explicitly studies partial/event-driven activation.",
    "Adapts dynamic local regret to the VFL setting with partial client activation, which is a nontrivial extension of prior non-convex online-learning analysis.",
    "Provides theoretical treatment for both the non-convex DLR setting and a convex OGD setting, with regret bounds that explicitly account for partial activation.",
    "Experiments consistently suggest that DLR variants are more stable than OGD under partial activation and synthetic non-stationarity.",
    "Shows an efficiency tradeoff: partial activation reduces some client-side computation and backward communication relative to full activation.",
    "The paper openly acknowledges the incomplete-view limitation and situates it as a central challenge for future work."
  ],
  "weaknesses": [
    "The core method has an unresolved conceptual ambiguity: passive clients are queried and must still provide embeddings every round, yet the paper motivates the problem by saying those clients are not activated by the event. The manuscript does not clearly explain what data passive clients use when queried in this online setting.",
    "Because passive clients still participate each round, the framework is not fully asynchronous/event-driven in the strong sense; the practical gains mainly come from reducing backward/update participation rather than avoiding passive-client involvement altogether.",
    "There is a clear theory-practice mismatch: Theorem 1 assumes random activation with client probabilities p_m, while the main 'Event' mechanism in experiments is data-dependent threshold activation. The current theory does not justify that case.",
    "Empirical validation is limited in rigor: results appear to be single-run curves/tables without variance, confidence intervals, or significance tests.",
    "The non-stationary setting is synthetic and relatively simple (class-probability changes every 50 rounds), so evidence for robustness to realistic concept drift remains limited.",
    "Experimental validation of the event-driven claim is narrow: the event process is synthetic, and the motivating real-world scenarios (sensor/enterprise VFL) are not evaluated with realistic traces or naturally event-driven datasets.",
    "The communication-efficiency claim would be stronger with a breakdown showing exactly which communication components are reduced; since passive clients still send embeddings, current aggregate totals are harder to interpret.",
    "Comparison to SLR is informative algorithmically, but part of the large communication gap is inherited from SLR's design rather than arising solely from the proposed innovation."
  ],
  "nice_to_haves": [
    "Provide a precise communication/computation decomposition separating active-client backward traffic, passive-client embedding uploads, and server-side costs.",
    "Add multi-seed results with mean/std or confidence intervals.",
    "Evaluate on more realistic drift types (gradual drift, recurrence) or real temporal datasets.",
    "Include visualizations of activation patterns over time and their relation to performance.",
    "Discuss privacy implications of data-dependent activation patterns, since activation itself may leak information."
  ],
  "novel_insights": "The strongest insight is not the use of DLR by itself, but the paper's reframing of online VFL around feature-arrival events rather than assuming synchronized streams across parties. This exposes a distinct online-VFL difficulty: even if only some clients observe new event-relevant features, the server still faces an incomplete-view problem and must somehow reconcile active and passive participants. The paper usefully shows that DLR can stabilize training under this partial-activation regime, but it also implicitly reveals a deeper systems bottleneck: without a mechanism to eliminate or approximate passive-client embeddings, event-driven VFL remains only partially asynchronous.",
  "missed_related_work": [
    "Feng et al. (2023), 'Asynchronous Vertical Federated Learning: A New Paradigm with Systematic Learning Algorithms' — potentially relevant for positioning against asynchronous VFL.",
    "Zhang et al. (2023), 'Online Vertical Federated Learning with Incomplete Features' — potentially relevant given the paper's partial-feature / passive-client setting.",
    "Shen et al. (2023), 'Federated Online Learning with Non-Convex Loss Functions' — potentially relevant background on non-convex online federated optimization, even if not VFL-specific."
  ],
  "suggestions": [
    "Clarify exactly what passive clients observe and transmit when queried in Algorithm 1. If they use newly sampled local features, explain why those features exist despite not being event-activated; if they use cached or stale features, state this explicitly and analyze implications.",
    "Reframe the efficiency claim more precisely: emphasize that savings come primarily from reduced backward/update participation, not from eliminating all passive-client communication.",
    "Either extend theory to cover the data-dependent event activation used in experiments, or clearly state that the theorem applies only to the random-activation variant and treat event activation as empirical.",
    "Report results over multiple random seeds with uncertainty estimates and, where possible, significance tests.",
    "Add stronger non-stationarity evaluations using more realistic drift patterns or real sequential datasets.",
    "Compare against more directly related asynchronous/incomplete-view VFL approaches where feasible, or explain why such comparisons are not possible.",
    "Discuss when the method is expected to work best, e.g., regimes where activated clients contain most event-relevant signal and passive features are relatively stable.",
    "Include a brief discussion of privacy leakage through activation patterns in the event-based setting."
  ],
  "novelty": 7.1,
  "technical_soundness": 6.3,
  "empirical_support": 6.0,
  "significance": 6.7,
  "clarity": 6.6
}

---

## N0i0d27RTW

- GT: Reject (avg 4.5)
- Predicted: None (None/10)
- Match: Parse fail

### Final Review

ERROR: Expecting value: line 367 column 1 (char 2013)

---

## LbgIZpSUCe

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (5.6/10)
- Match: Yes

### Final Review

{
  "summary": "This paper introduces MRDS-IR, a probabilistic multi-region state-space model that combines nonlinear within-region latent dynamics with directed inter-region communication channels parameterized as linear impulse-response filters. The formulation is well motivated for neuroscience because it can express local nonlinear computation while preserving an interpretable temporal/frequency-domain description of communication. The paper also develops a variational filtering/learning procedure that accommodates deterministic channel-state transitions without requiring state-noise inversion. Empirically, the method is validated on matched synthetic data, on two simulated RNN tasks designed to probe integration/gating and rhythmic communication, and on V1/V2 recordings, where it shows competitive held-out prediction and yields qualitatively interpretable feedforward/feedback channel structure. Overall, this is a thoughtful and technically interesting contribution with clear scientific motivation. The main limitations are that the empirical study does not fully position the method against the strongest multi-region baselines, provides limited uncertainty/sensitivity analysis, and leaves important practical/theoretical questions—especially identifiability, scalability, and hyperparameter selection—largely open.",
  "strengths": [
    "The modeling idea is compelling: nonlinear region-specific dynamics are paired with linear impulse-response communication channels, yielding a useful trade-off between expressivity for local computation and interpretability for inter-region signaling.",
    "The channel parameterization is scientifically meaningful. By working with impulse responses / transfer functions, the model can naturally represent delays and frequency-selective communication, which many prior multi-region models handle less directly.",
    "The paper is generally well positioned relative to prior multi-region modeling work, and Table S1 helps clarify the design space.",
    "The inference scheme is technically nontrivial and relevant to the model structure; in particular, the state-noise inversion-free treatment of deterministic channel transitions is a meaningful implementation contribution.",
    "The synthetic matched-model experiment shows the method can recover latent dynamics and active/inactive channels when the model assumptions are satisfied.",
    "The RNN case studies go beyond pure recovery and illustrate that the model can extract interpretable computational motifs such as integration, gating, and rhythmic transmission under model mismatch.",
    "On real V1/V2 data, the method achieves strong held-out prediction relative to the reported baselines and produces channel estimates with plausible feedforward/feedback temporal structure.",
    "The paper is scientifically ambitious and aims not just at prediction but at extracting interpretable descriptions of distributed computation, which is a valuable goal for this application area."
  ],
  "weaknesses": [
    "The empirical comparison set is incomplete. In particular, methods discussed as close conceptual competitors, especially MR-SDS and mp-srSLDS, are not included in experiments, making it harder to judge the practical advantage of the proposed design choices.",
    "Some baseline comparisons are not fully convincing. In the integration-task experiment, CURBD is reported as using default hyperparameters, which raises fairness concerns if the proposed method was tuned more carefully.",
    "The paper lacks systematic sensitivity/ablation analyses for key modeling choices such as latent dimensionality, filter order, and connectivity assumptions. As a result, it is difficult to assess robustness or give practitioners guidance on model selection.",
    "There is little discussion of computational complexity and scalability. Because channel state dimensionality grows with the number of region pairs and filter order, it remains unclear how well the method would scale beyond the 2-3 region settings shown here.",
    "The decomposition between local nonlinear dynamics and communicated signals is central to the interpretation claims, but identifiability is not analyzed. It is therefore unclear under what conditions communication effects can be uniquely attributed to channels rather than absorbed into local dynamics.",
    "The evidence on real data is still relatively narrow: the main real-data evaluation is on one V1/V2 dataset, so the breadth of practical impact is not yet fully demonstrated.",
    "Although the real-data section reports multiple partitions and some multiple-seed evaluation, uncertainty quantification is limited overall, and several quantitative plots would be stronger with explicit variability estimates."
  ],
  "nice_to_haves": [
    "More explicit training details in the main paper or a clearly organized appendix summary (optimizer, learning-rate schedule, training length, architecture sizes) would improve reproducibility.",
    "A more integrated presentation of the filtering algorithm would help readability; currently some key pieces are split between the main text and multiple appendix sections.",
    "For the frequency-domain interpretation, fuller visualizations (e.g., phase as well as magnitude) could further showcase the benefits of the transfer-function view."
  ],
  "novel_insights": "The strongest insight is not any one component in isolation, but the particular synthesis: using nonlinear local latent dynamics while constraining inter-region interactions to stable linear filters creates a modeling interface that is unusually well aligned with neuroscience questions about delays, temporal filtering, and frequency-selective communication. This makes it possible to interpret communication channels using classic systems tools without giving up nonlinear computation within regions. The paper also usefully highlights that causal filtering in this structured multi-region setting should respect the observation factorization, so that one region’s observations do not instantaneously update another region’s latent belief; that modeling/inference consistency is easy to overlook and is conceptually important here.",
  "missed_related_work": [
    "LFADS (Pandarinath et al., 2018) could be cited to better situate the variational neural-dynamics lineage in the neural data literature.",
    "The original rSLDS work underlying later multi-population adaptations may be worth citing alongside the mp-srSLDS discussion.",
    "Depending on timing, a brief mention of very recent multi-region LFADS-style work could help contextualize concurrent developments."
  ],
  "suggestions": [
    "Add quantitative comparisons to stronger and more directly related multi-region baselines, especially MR-SDS and, if feasible, mp-srSLDS, on at least one synthetic or benchmark setting.",
    "Include ablations or sensitivity analyses over latent dimensionality, channel order, and possibly connectivity sparsity to show robustness and give practical model-selection guidance.",
    "Report uncertainty more consistently in quantitative figures (e.g., mean ± std or confidence intervals across seeds/splits) and, where appropriate, statistical tests for key comparisons.",
    "Discuss computational complexity explicitly, including how state dimension, memory, and runtime scale with number of regions, latent dimensions, and channel order.",
    "Add a discussion—empirical or theoretical—of identifiability and failure modes: when can communication be reliably separated from local dynamics, and when might that interpretation break down?",
    "Clarify in the paper which parts of the inference/learning machinery are novel versus adapted from prior amortized variational filtering / SVAE-style approaches.",
    "If possible, broaden real-data validation to additional datasets or more regions, or at least temper scalability claims accordingly.",
    "Consider citing LFADS and related neural-dynamics literature to better connect the method to prior sequential variational approaches in neuroscience."
  ],
  "novelty": 7.4,
  "technical_soundness": 7.5,
  "empirical_support": 6.8,
  "significance": 7.6,
  "clarity": 7.8
}

---

## X4Rcxi9588

- GT: Reject (avg 5.5)
- Predicted: Accept (5.6/10)
- Match: No

### Final Review

{
  "summary": "The paper presents a practical approach for adapting pretrained LMMs to longer video inputs without retraining on long video-text data. The key idea is to distinguish a visual context window from the language context window and adapt YaRN-style RoPE scaling accordingly, combined with a progressive pooling strategy to reduce visual token count and memory. Empirically, the method improves over the LLaVA-OneVision-7B baseline on VideoMME, MLVU, and LongVideoBench, and the memory savings from progressive pooling are substantial. Overall, the work is useful and well-motivated, but its main technical contribution is somewhat incremental, the causal justification for the visual/language context-window distinction is not fully established, and several important analyses/ablations are still missing.",
  "strengths": [
    "Addresses an important and practically relevant problem: extending strong pretrained LMMs to long video understanding without expensive retraining on long video-text corpora.",
    "The method is simple, lightweight, and easy to integrate with existing RoPE-based LMMs; this practicality is a genuine strength.",
    "Empirical results are consistently positive on multiple benchmarks. Relative to the reproduced LLaVA-OneVision-7B baseline, the method improves performance on VideoMME, MLVU, and LongVideoBench, especially in longer-video regimes.",
    "The progressive pooling component provides meaningful deployment value: Table 5 shows memory reduction from 73GB to 40GB at 256 frames with no loss relative to the unpooled version used in their ablation, which is a strong practical result.",
    "Ablations do support both components to some extent: Table 4 shows gains over baseline and standard YaRN, while Table 5 explores several pooling settings and demonstrates a useful accuracy/memory tradeoff.",
    "The paper’s central observation—that language-side long-context capability does not automatically transfer to long visual sequences—is interesting and well motivated by the reported performance trends.",
    "The appendix includes an additional visual needle-in-a-haystack style evaluation suggesting extension to substantially longer frame counts without retraining."
  ],
  "weaknesses": [
    "The core technical novelty is limited. The visual context window extension is largely an adaptation of YaRN to visual tokens by redefining the scaling based on visual pretraining length rather than introducing a fundamentally new mechanism.",
    "The paper’s conceptual claim that modality discrepancy implies distinct effective visual and language context windows is suggestive but not rigorously established. Figure 1b is diagnostic, but t-SNE clustering alone does not provide causal evidence for the proposed explanation.",
    "The gain over standard YaRN, while real, is moderate rather than transformative in the main ablation (e.g., 58.5 to 61.0 overall on VideoMME at 256 frames without pooling), so the paper should calibrate the strength of its claims accordingly.",
    "Progressive pooling is heuristic. The assumption that grouped frames correspond to an event, and that the first frame in a group should retain higher spatial resolution, is intuitive but not validated against alternative designs such as middle-frame, keyframe, or content-adaptive selection.",
    "A key component-isolation ablation is missing: the paper does not report progressive pooling without visual context window extension, so the independent effect of pooling is not fully separated from the extension method.",
    "Hyperparameter sensitivity remains underexplored. In particular, there is little analysis of how the method depends on the assumed visual training context window or how robust the pooling parameters are beyond the small grid reported.",
    "The comparison highlighting that the method outperforms GPT-4o on MLVU should be framed more carefully, since the evaluation uses different input sampling strategies; this makes the headline comparison less clean than implied.",
    "The paper notes degradation on shorter-duration subsets when using 512 frames, but the proposed explanation ('attention distraction') is speculative and not backed by deeper analysis.",
    "The study is limited to a single backbone family (LLaVA-OneVision-7B), so it is unclear how broadly the visual-context-window formulation generalizes across LMM architectures."
  ],
  "nice_to_haves": [
    "Report inference latency/throughput in addition to memory, since deployment efficiency is one of the paper’s main motivations.",
    "Add multi-run statistics or variance estimates if feasible, especially for the relatively modest gains over YaRN.",
    "Include more analysis of failure cases and task/video-type breakdowns, especially for rapid scene changes or videos with weak event structure.",
    "Provide clearer guidance for selecting frame count and pooling parameters as a function of video duration."
  ],
  "novel_insights": "The most interesting insight is not the specific formula change itself, but the framing that long-context limitations in LMMs may be modality-dependent: a decoder that can handle long text contexts may still have a much shorter effective range for visual token sequences inherited from short-video pretraining. Even if the evidence is not yet fully causal, this reframing is useful and may motivate more modality-specific context-extension methods in future multimodal systems.",
  "missed_related_work": [
    "Video-LLaMA (original paper), as foundational prior work on video-language modeling with LLMs.",
    "ALiBi as an alternative long-context positional scheme that could be discussed when positioning RoPE-based extension methods.",
    "Potential long-context efficiency works such as Landmark Attention, if the authors want to broaden discussion of long-sequence handling beyond RoPE scaling and pooling."
  ],
  "suggestions": [
    "Strengthen the evidence for the visual-vs-language context-window hypothesis with quantitative analysis beyond t-SNE, e.g., embedding distance statistics across layers or direct attention/position-sensitivity analysis.",
    "Add a 2x2 ablation isolating the two proposed components: with/without visual context extension and with/without progressive pooling.",
    "Validate the design of progressive pooling against simple alternatives, such as preserving the middle frame, random frame, or content-selected keyframe at high resolution.",
    "Study sensitivity to the assumed visual training context window and explain more explicitly how it is determined from the pretrained model setup.",
    "Temper the GPT-4o claim or explicitly acknowledge the mismatch in sampling strategy to avoid overstating fairness of that comparison.",
    "Evaluate on additional backbones or model sizes to show that the method is not specific to LLaVA-OneVision-7B.",
    "Investigate the short-video degradation under dense sampling and provide either analysis or an adaptive frame-selection rule based on video duration.",
    "If possible, report runtime/latency alongside memory to better support the practical-efficiency claims."
  ],
  "novelty": 6.0,
  "technical_soundness": 7.0,
  "empirical_support": 7.5,
  "significance": 7.0,
  "clarity": 7.5
}

---

