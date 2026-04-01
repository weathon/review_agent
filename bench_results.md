# ICLR Benchmark Results

Date: 2026-03-31 20:20
Critic/Merger: claude-sonnet-4-6 (Claude SDK, free)
Neutral: z-ai/glm-5, Related Work: perplexity/sonar-pro (OpenRouter)

## q3EbOXb4y1

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (6.1/10)
- Match: Yes

### Final Review

```json
{
  "summary": "Retri3D is the first end-to-end framework for text-based retrieval of pre-trained 3D Neural Graphics Representations (NGRs) from large data stores. The paper introduces two core technical contributions: Neural Graphics Noise Analysis, which leverages intermediate activations of pre-trained VLMs to distinguish floater/artifact pixels from clean pixels via a fitted multivariate Gaussian in activation space, and a Smart Camera Movement Module (SCMM) that iteratively navigates a camera to artifact-free viewpoints for higher-quality visual embedding extraction. The framework renders RGB images from any pre-trained NGR and uses VLM embeddings for cosine-similarity-based retrieval, evaluated on the LERF (13 scenes) and ScanNet++ (280 scenes) datasets.",
  "strengths": [
    "Genuine novelty in problem formulation: this is the first work to frame cross-scene retrieval of pre-trained NGRs as a standalone problem, and the motivation (proliferation of NGR assets on sharing platforms without retrieval infrastructure) is credible and timely.",
    "Non-trivial empirical insight: the observation that VLM intermediate activations tightly cluster floater/artifact pixels across diverse NGR types and scenes — and are separable from clean-region features — is well-supported by multi-level t-SNE visualizations (Figs. 2, 6, 7, 8, 9) spanning different VLMs (XDecoder, OpenCLIP) and different NGR types (Splatfacto/Nerfacto), and it enables a practical noise estimator without requiring training images or poses.",
    "Format-agnostic design: by operating exclusively on rendered RGB images rather than on learned parameters, the pipeline is compatible with any NGR format that supports rendering, including future formats. This is a principled architectural choice with real long-term value.",
    "Practical efficiency: Table 3 shows Retri3D achieves 17-second scene analysis vs. 53-152 seconds for adapted baselines, and 20.58MB embedding storage vs. 18.78GB/225.4GB, enabling 1ms retrieval at 280 scenes via FAISS IVF1024. These are meaningful engineering advantages.",
    "Meaningful accuracy improvement over random viewpoints: SCMM consistently outperforms random viewpoint sampling by 20%+ in P@1 across both object-label and LLaVA-caption query modes (Table 1), and achieves retrieval accuracy within 2-5% of the oracle training-viewpoint upper bound at 50-100 images per scene.",
    "ScanNet++ results at 280 scenes (Table 4) provide a credible demonstration of practical scalability, with SCMM maintaining a 3-4% gap behind training viewpoints across all frame counts, and a multi-sentence query strategy (Table 13) uniquely enabled by Retri3D's compact embedding structure."
  ],
  "weaknesses": [
    "Primary evaluation dataset is extremely small: the LERF dataset has only 13 scenes. With P@1 reported as the primary metric, differences of 3-5% between conditions — which determine the paper's core claims about SCMM's benefit — involve changes of fewer than 1 scene in a 13-scene database. No confidence intervals, error bars, or statistical significance tests appear anywhere in the paper, making it impossible to determine whether the observed improvements are reliable signals or sampling noise.",
    "Baselines are structurally disadvantaged and not designed for retrieval: LERF and LangSplat were built for interactive scene-level semantic queries within a single scene, not cross-scene retrieval. LangSplat's per-scene autoencoders break cross-scene comparability by design; LERF's relevancy computation requires negative prompts unavailable at retrieval time (Appendix B.9/B.10). The comparison reads as partially a strawman. A fairer baseline — e.g., render training views with known poses and apply CLIP directly — is not included, leaving readers unable to assess how much of Retri3D's advantage comes from noise-aware viewpoint selection vs. simply using a better VLM.",
    "SCMM is compared only against random viewpoints, not against any alternative active/next-best-view selection strategy. Works like FisherRF (Jiang et al., 2023) and ActiveNeRF (Pan et al., 2022) are cited in the paper but not positioned against SCMM in either related work or experiments. Without such a comparison, the specific design contribution of SCMM (linear-decay edge padding, maximum-sum submatrix selection) is unvalidated relative to simpler noise-aware alternatives.",
    "The high-dimensional covariance estimation in the noise analysis module is a latent reliability concern: the Gaussian is fit in the c=512 dimensional activation space of XDecoder's FPN Level 1, using approximately K×50 = 2,500 activation samples from 50 random-viewpoint renderings. Fitting a 512×512 covariance matrix from 2,500 samples is ill-conditioned, and the regularization constant ε is never reported. In low-sample / high-dimension regimes this can produce highly unstable Mahalanobis distances. The paper provides no analysis of how sensitivity to K or ε affects retrieval accuracy.",
    "The claim that Retri3D is compatible with 'any NGR representation' is empirically undersubstantiated: experiments cover only Splatfacto and Nerfacto (both from the nerfstudio ecosystem), which share volumetric rendering characteristics and produce similar artifact profiles. NGR formats with fundamentally different rendering paradigms (differentiable meshes, neural SDFs with surface rendering) may produce artifact types not represented in the trained noise Gaussian, potentially breaking the cross-format transfer assumption.",
    "Key ablations are missing: (1) the contribution of SCMM steps (1 vs. 3 vs. more iterations) on retrieval accuracy; (2) the effect of whole-image-only vs. whole-image + segment embeddings in isolation (which would quantify XDecoder's segmentation advantage); (3) sensitivity to the noise Gaussian training set size K, which is critical given the high-dimensional covariance estimation concern; (4) sensitivity to the scaling factor α and cutoff c in the clean score function (Eq. 3)."
  ],
  "nice_to_haves": [
    "Testing on broader NGR formats (InstantNGP, TensoRF, Zip-NeRF, Scaffold-GS) to more rigorously support the format-agnosticism claim.",
    "A scalability stress test at 1,000-10,000 NGRs to validate the motivation around large online data stores, which the 280-scene ScanNet++ experiment only partially addresses.",
    "Mechanistic explanation for why VLM activations tightly cluster floater/artifact features — connecting this to volumetric rendering geometry (elongated Gaussian blobs, depth discontinuities) would make the technique feel principled rather than empirically discovered.",
    "A deployment-oriented scalability discussion covering FAISS index retraining, noise Gaussian retraining frequency, and GPU-hour costs for datasets at platform scale (millions of NGRs).",
    "Image-query retrieval (image-to-NGR) as a natural low-overhead extension, since the VLM already supports both modalities.",
    "SCMM 3D trajectory visualization superimposed on a reconstructed mesh, which would make the navigation behavior more immediately intuitive than sequential RGB snapshots."
  ],
  "novel_insights": "The most intellectually interesting finding is that VLM intermediate activations act as a universal floater/artifact detector across NGR types and scene content without any task-specific training or fine-tuning. The t-SNE evidence showing that noise feature distributions from Splatfacto and Nerfacto are closer to each other than to clean features from their own scenes (Fig. 8) suggests a rendering-physics basis for this clustering — floaters produce characteristic high-frequency, context-free activation patterns that VLM encoders map to a tight manifold in representation space. This is a potentially generalizable principle for detecting distribution shift in rendered 3D scenes. Separately, the scene coverage analysis (Table 5) quantifying that training trajectories cover only ~10-14% of scene volume (location+orientation) provides a concrete empirical grounding for why random viewpoints are so artifact-prone — undersampled regions lack geometric constraints during training, causing floater formation that random viewpoints inadvertently sample.",
  "missed_related_work": [],
  "suggestions": [
    "Add confidence intervals or bootstrap standard errors to all P@k results, especially for the 13-scene LERF dataset where statistical significance is critical to interpreting 3-5% accuracy differences.",
    "Include a fairer retrieval baseline: render training-pose images from scenes where poses are available and apply CLIP/XDecoder cosine similarity — this represents the simplest approach a practitioner would attempt and would clarify how much benefit comes specifically from SCMM vs. viewpoint quality generally.",
    "Compare SCMM against at least one alternative active-view selection strategy (e.g., entropy-based uncertainty sampling or a simplified FisherRF-style criterion) to validate that the specific SCMM design contributes beyond generic noise-aware selection.",
    "Report the regularization constant ε and conduct a sensitivity analysis over K (noise Gaussian training set size) and the covariance conditioning, to address the high-dimensional estimation concern.",
    "Explicitly test on at least one NGR format outside the nerfstudio/volumetric-rendering family to support the 'any NGR' generality claim."
  ],
  "score": 6.1,
  "score_justification": "Novel, well-motivated systems contribution addressing a genuinely new problem, with a non-trivial technical insight (VLM activation-based noise detection) and practical efficiency gains, but held back by a primary evaluation on only 13 scenes with no statistical testing, structurally disadvantaged baselines, missing ablations on key design choices, and an undersubstantiated generality claim — consistent with a borderline ICLR acceptance.",
  "decision": "Accept"
}
```

---

## UapxTvxB3N

- GT: Accept (Poster) (avg 5.8)
- Predicted: Reject (4.8/10)
- Match: No

### Final Review

```json
{
  "summary": "Trajectory-LLM (TrajLLM) proposes a two-stage 'interaction-behavior-trajectory' translation pipeline that uses an LLM (Llama-7B) to generate vehicle trajectories from brief textual interaction descriptions, with an intermediate behavior-generation step grounded in driving logic. The authors also introduce the L2T dataset of 240K annotated traffic scenarios. Experiments show improvements in trajectory realism and controllability over prior language-based generators, and generated data provides modest gains when combined with real WOMD training data for downstream prediction.",
  "strengths": [
    "Compelling and well-illustrated motivation: Figure 1 concretely demonstrates that purely text-to-trajectory generation produces unsafe behaviors (dangerous speeding, topology-blind lane changes), and the behavior intermediary is shown via ablation (Table 3) to dramatically improve controllability from 36.7% to 81.6%.",
    "Consistent realism improvements across agent- and scene-level kinematic metrics in Table 1, with Traj-LLM outperforming all compared baselines (SNet, TSim, BITS, CTG, LGen, CTG++) on AVG realism at both levels by substantial margins.",
    "L2T dataset contribution: 240K annotated scenarios with six road topologies, four object classes, and human-annotated behavior-logic text sequences fills a genuine gap and is released with code for community use.",
    "Ablation study (Table 3) cleanly decomposes the contribution of driving-logic learning, random jitter, and locality attention, showing each component plays a distinct role.",
    "Practical downstream benefit is demonstrated: combining WOMD training data with Traj-LLM-generated data improves both MTR and HDGT on WOMD validation across all three metrics (mAP, minADE, Miss Rate)."
  ],
  "weaknesses": [
    "Factual error in abstract and introduction: the paper explicitly claims improvements on 'Waymo and Argoverse datasets,' but no Argoverse results appear anywhere in the paper — only WOMD and L2T validation results. This is a straightforward inaccuracy, not a parser artifact.",
    "Trajectory tokenization is entirely absent from the main paper: cross-entropy loss L_T is used, implying discrete coordinate tokens, but the discretization scheme (bin resolution, vocabulary design, precision bounds) is never described. At 20 Hz over 20 seconds with (x, y, heading, speed) per waypoint this is a critical reproducibility gap for the method's core output.",
    "The downstream prediction experiment (Table 4) does not control for total training data size — the combined model trains on 800K scenes vs. 400K for single-source baselines. The paper acknowledges a fixed-size ablation exists in Appendix D but relegates it there; for a paper whose primary applied claim is that generated data benefits prediction models, this ablation must appear in the main body.",
    "The full model trades substantial scene-level diversity for realism: the ablation (Table 3) shows WD drops from 22.64 (Logic+Random only) to 13.81 (full model), a ~39% reduction driven by the Locality component. This trade-off is acknowledged in one sentence but never analyzed or presented as a tunable Pareto curve, despite being highly relevant to practitioners who may prioritize diversity.",
    "Stage 1 accuracy and error propagation are uncharacterized: generated behavior text B from Stage 1 directly conditions Stage 2, but no metrics on Stage 1 accuracy (e.g., behavior-label F1 against ground-truth B̃) or analysis of how Stage 1 errors cascade are provided anywhere.",
    "No error bars or statistical significance are reported in any table, despite Traj-LLM's stochastic components (random locality attention) and the small absolute gains in downstream prediction (MTR mAP: 0.405 → 0.416, a +2.7% improvement).",
    "The random locality attention hyperparameters K (top-K) and the normal distribution parameters for the jitter variables are never specified in the main paper, making the method incompletely specified for reproduction.",
    "LGen in Table 1 and LCTGen in Table 2 appear to be the same system (Tan et al., 2023) but are named inconsistently, creating confusion about whether one comparative result is missing.",
    "The simulator used to construct L2T is described only as 'a new simulator we constructed' with no characterization of physics fidelity, rendering realism, or validation against real-world trajectory statistics, which is foundational to assessing the dataset's utility."
  ],
  "nice_to_haves": [
    "LLM backbone ablation (Llama-7B vs. larger variants) to determine whether gains stem from architecture or model scale.",
    "Per-topology breakdown of realism and diversity metrics to identify where the interaction-behavior framework helps most.",
    "Data scaling curve showing downstream prediction performance as a function of number of generated scenes added to WOMD training.",
    "Wall-clock inference time and throughput comparison across methods, given the paper's motivation of efficient generation.",
    "Failure case gallery illustrating when Traj-LLM produces physically infeasible or interaction-mismatched trajectories.",
    "Out-of-distribution text prompt evaluation using free-form user text not from L2T annotators.",
    "Attention weight visualization showing which polylines each interaction feature attends to, for interpretability."
  ],
  "novel_insights": "The core insight — that inserting natural-language behavior descriptions with explicit driving logic as an intermediate representation between abstract interaction text and numerical trajectories improves both realism and map-topology generalization — is genuine and well-supported by the ablation. The random locality attention, which stochastically focuses each interaction on its most spatially relevant map polylines and trajectory features while adding controlled jitter, is a technically interesting mechanism for simultaneously improving realism, diversity, and map-grounding. The finding that the Locality component constrains diversity (trading WD ~22 for ~13) while the Random component recovers partial diversity is a nuanced result that deserves more prominence as a design principle for controllable generation.",
  "missed_related_work": [
    "InteractTraj (NeurIPS 2024) — directly parallel LLM-based interactive trajectory generation from natural language; potentially concurrent but relevant for positioning.",
    "LC-LLM (Peng et al., 2024) — fine-tunes GPT-2 with chain-of-thought to predict lane changes with textual justifications, a closely related paradigm of LLM-based driving behavior reasoning with generated explanations."
  ],
  "suggestions": [
    "Either include Argoverse results in the paper or remove the Argoverse claim from the abstract and introduction — this factual error should be corrected before any venue publication.",
    "Add a dedicated paragraph or appendix section describing the trajectory tokenization scheme (binning resolution, vocabulary size, coordinate precision) — this is essential for reproducibility and for assessing whether the cross-entropy formulation is appropriate.",
    "Move the fixed-total-dataset-size ablation from Appendix D into the main body of Section 5.3 so that the core downstream benefit claim is rigorously supported in the main paper.",
    "Add a diversity-realism Pareto analysis by sweeping the Locality strength (K values) and report both WD and AVG realism, giving users principled guidance on tuning the trade-off.",
    "Report Stage 1 behavior-prediction accuracy (e.g., behavior-type F1 against ground-truth annotations) to characterize the error propagation risk in the two-stage pipeline.",
    "Add standard deviations across multiple runs to Tables 1–4, particularly for the stochastic diversity metrics and the downstream prediction improvements."
  ],
  "score": 4.8,
  "score_justification": "Solid applied contribution with a genuine insight (behavior intermediary improves realism and map generalization) and a useful dataset, but critically undermined by a factual error in the abstract, missing trajectory tokenization details that prevent reproduction, and an uncontrolled data-size comparison for the primary downstream claim.",
  "decision": "Reject"
}
```

---

## b4b6rERW4G

- GT: Reject (avg 4.8)
- Predicted: Reject (4.2/10)
- Match: Yes

### Final Review

```json
{
  "summary": "DPOT proposes a data-poisoning-only backdoor attack on Federated Learning that dynamically generates a new trigger each FL round by (1) selecting high-gradient pixels as trigger placements and (2) optimizing trigger pixel values via gradient descent, with the goal of minimizing the current global model's loss on backdoored data so that malicious clients' model updates remain indistinguishable from benign ones. The paper provides theoretical justification under a linear regression model and evaluates against 11 defenses across 4 datasets and 4 model architectures, claiming consistent attack success rates above 50% with only 5% malicious clients.",
  "strengths": [
    "Conceptually sound and well-motivated attack principle: if the global model is already near-optimal on the backdoored data, further local training on that data will produce minimal parameter drift, causing malicious updates to blend with benign ones. This is a legitimate insight that challenges the foundational assumption of model-update-analysis defenses.",
    "Impressive evaluation breadth: 4 datasets (FashionMNIST, FEMNIST, CIFAR10, Tiny ImageNet), 4 model architectures (custom CNNs, ResNet18, VGG11), 11+ defense strategies including FLAIR, FLAME, FoolsGold, FLCert, and client-side defenses (Flip, FRL). This is substantially more comprehensive than most FL backdoor papers.",
    "Clever decomposition experiment (Table 1): separating the contribution of 'Trigger Optimization' alone (ASR* via benign-only aggregation) from 'Aggregation of Backdoored Model Updates' (ASR** with malicious clients) provides direct mechanistic evidence for each component's role, which is a well-designed ablation.",
    "Data-poisoning-only threat model is practically motivated and novel in context: by operating exclusively through data manipulation, DPOT remains effective in TEE-secured FL environments where model-poisoning is authenticated away—a security setting that existing attacks fail to address.",
    "A3FL comparison (Table 6, Appendix E.1) shows DPOT dramatically outperforms the closest prior art in the 'dynamic objective + L0 trigger' category (Final ASR 100% vs. 48.9% on FedAvg, CIFAR10), lending credibility to the core algorithmic contribution.",
    "Trigger evolution visualization (Appendix F.4) empirically confirms the Markov chain structure—the trigger changes gradually and coherently as the global model evolves, supporting the design rationale and providing intuitive insight into DPOT's stability."
  ],
  "weaknesses": [
    "Theory-to-practice gap: the theoretical justifications (Appendix C, Propositions C.1–C.3) are derived exclusively under a linear regression model with MSE loss. All experiments use deep neural networks (ResNet18, VGG11, CNNs) with cross-entropy loss. The paper never discusses whether the linear-model guarantees transfer to the DNN setting, even informally. This gap is a core credibility issue for ICLR, which expects theory and experiments to align.",
    "Proof of Proposition C.3 has a mathematical gap: the greedy recursive argument for joint placement-and-value optimization assumes that selecting pixel i based on its gradient magnitude at step k is independent of how the gradient at step k+1 changes after updating pixel i's value. In image data processed by DNNs, pixel gradients are strongly correlated, so the independence assumption underlying the recursive construction is invalid. The fixed-step-size Δv in the proof also does not correspond to the iterative gradient descent in Algorithm 2.",
    "No statistical uncertainty quantification: all results in Tables 1, 2, 3, 6, and 7 are single-run point estimates with no error bars, confidence intervals, or variance across random seeds. FL training with non-IID data and stochastic aggregation has substantial run-to-run variance. For margins that matter (e.g., FLAIR/FLAME where DPOT is near the 50% threshold), the absence of statistical testing makes it impossible to confirm whether reported differences are reliable.",
    "A3FL—the closest and most important baseline—is relegated to Appendix E.1 rather than the main paper. A3FL is the direct predecessor in the 'dynamic objective + L0-norm trigger + data poisoning' category; its comparison is more important than FT and DFT (which use static triggers). Demoting this comparison misrepresents the paper's primary contribution.",
    "Non-standard evaluation metric inflates apparent effectiveness: ASR is tested each round using a trigger freshly optimized for that round's global model. At inference time, any real attacker uses a fixed trigger (at most, the last round's optimized trigger). The Avg ASR metric is especially inflated because it tests each round with a round-specific trigger, not the trigger an attacker would actually deploy. A complementary evaluation using the final round's trigger tested retroactively across all rounds is needed.",
    "Default local data poison rate of 50% is extremely aggressive and not ablated in the main paper. At 50%, roughly half of each malicious client's data is mislabeled—a level detectable by label-distribution checks or server-side statistical validation. The paper acknowledges lower rates induce stealthier updates but provides no analysis of the minimum poison rate needed for DPOT to succeed.",
    "FLAIR and FLAME partial failures are not analyzed. Against FLAIR (MCR=0.05, Table 2), DPOT achieves only 62% Final ASR. Against FLAME, ~60%. These are above the 50% threshold but qualitatively different from near-100% performance against other defenses. The paper does not explain what properties of these defenses partially resist DPOT, leaving the claimed generality of the concealment mechanism in question.",
    "Physical-world threat model inconsistency: the paper argues L0-norm triggers are practical because they can be applied as physical stickers. However, DPOT generates a different trigger each FL round (the trigger is model-dependent and changes as the global model evolves). A physical sticker cannot be updated each round, so the real-world applicability argument is undermined. The paper does not clarify whether the last round's fixed trigger achieves acceptable ASR when deployed as a physical sticker."
  ],
  "nice_to_haves": [
    "Comparison with CerP/Cerberus (Lyu et al., 2023), which occupies the same 'data-poisoning + dynamic objective' quadrant and is cited in related work but excluded from experiments.",
    "Discussion of Neurotoxin (Zhang et al., 2022) in related work, which also targets specific model parameters for durable backdoors and is a related concealment strategy.",
    "Computational overhead analysis: Algorithms 1 and 2 require gradient computation over D every round for each malicious client. For VGG11 on Tiny ImageNet (~35M parameters), this could be substantial; a wall-clock comparison would help practitioners assess feasibility.",
    "Ablation separating the marginal contributions of placement optimization (Algorithm 1) vs. value optimization (Algorithm 2)—i.e., fixed placement + optimized values vs. optimized placement + fixed values vs. both—to quantify each component's independent benefit.",
    "Discussion of adaptive defenses specifically designed to counter gradient-aligned trigger placement, since DPOT's placement algorithm is deterministic given the global model state.",
    "Extension to partial client participation (e.g., 10–30% per round), which is more realistic in large-scale FL deployments.",
    "Brief discussion of DPOT under differential privacy (DP-SGD) settings, since DP noise in the global model could disrupt round-specific trigger optimization."
  ],
  "novel_insights": "The core insight—that minimizing the global model's loss on backdoored data before local training causes model updates to be dominated by benign gradients—is a genuine and underexplored observation. Prior dynamic-objective attacks (A3FL, F3BA) optimized triggers to maximize neuron activations or unlearning resistance, without directly targeting gradient alignment with benign updates. DPOT's Markov chain observation (trigger and global model co-evolve gradually and coherently across rounds) is an interesting structural insight with potential theoretical implications. The decomposition experiment design (Table 1) that measures the independent contribution of trigger optimization vs. backdoored update aggregation is a methodologically sound and informative analysis technique that goes beyond typical FL backdoor evaluations.",
  "missed_related_work": [],
  "suggestions": [
    "Move the A3FL comparison (Table 6) to the main paper and reduce the prominence of FT/DFT, which are weaker and less informative baselines for the paper's core claim.",
    "Add empirical gradient alignment analysis in the DNN setting (e.g., cosine similarity between malicious and benign gradients under DPOT vs. FT, measured across rounds) to bridge the linear-regression theory to DNN experiments.",
    "Run each experimental configuration across at least 3 seeds and report mean ± std across Tables 1–3 and 6; for key claims (FLAIR, FLAME), report statistical significance.",
    "Add an evaluation where the final round's optimized trigger is used as a fixed trigger across all test rounds (or across a held-out test set), to provide a realistic inference-time ASR metric alongside the current round-specific metric.",
    "Provide an ablation over local data poison rates (e.g., 0.1, 0.2, 0.3, 0.5) to establish the minimum effective rate and characterize the stealthiness-effectiveness tradeoff more rigorously.",
    "Analyze why FLAIR and FLAME partially resist DPOT—e.g., by plotting the cosine similarity distributions of malicious vs. benign updates under each defense and identifying what signal these defenses detect—to honestly characterize the attack's limitations.",
    "Correct the duplicate paragraph in Section 4.2 and unify the Wg/Wgg notation inconsistency throughout the paper."
  ],
  "score": 4.2,
  "score_justification": "Real and meaningful empirical contribution with comprehensive evaluation breadth, but undermined by a theory-to-practice gap (all proofs are for linear regression, all experiments use deep networks), a flawed recursive proof (Proposition C.3 independence assumption), no statistical uncertainty quantification across any result, relegation of the most important baseline (A3FL) to an appendix, and a non-standard evaluation metric that inflates apparent effectiveness—collectively insufficient for ICLR's bar for rigor.",
  "decision": "Reject"
}
```

---

## X5rO5VyTgB

- GT: Accept (Poster) (avg 5.6)
- Predicted: Accept (6.1/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper introduces UnKE, a method for editing unstructured (long-form, free-text) knowledge in LLMs, alongside UnKEBench, the first dedicated benchmark for this task. UnKE replaces the 'local layer key-value storage' of ROME/MEMIT with 'non-local block key-value storage' (shallow layers as key generators, deep layers as value generators) and replaces 'term-driven optimization' with 'cause-driven optimization' that anchors edits at the last token of the input, avoiding the need to locate subject terms and preventing context information loss. Experiments on UnKEBench and structured editing benchmarks show strong lexical and semantic similarity improvements over baselines, with competitive structured-editing performance.",
  "strengths": [
    "Novel and well-motivated problem formulation: unstructured knowledge editing is genuinely underexplored, and the paper clearly identifies two concrete failure modes of existing methods (local KV inadequacy for rich content, term localization difficulty) with supporting experiments (Figure 2, Table 1).",
    "UnKEBench is a genuine benchmark contribution with a comprehensive evaluation framework spanning four dimensions (lexical similarity, semantic similarity, factual correctness, general ability), human evaluation with inter-annotator agreement (κ=0.57), and Pearson correlation validation showing >0.95 correlation between automatic and human metrics.",
    "Strong empirical results on semantic/lexical metrics: UnKE achieves Bert-Score 99.61/93.09 vs. the best non-retrieval baseline LoRA at 88.05/84.62, with consistent superiority across BLEU, ROUGE-1/2/L on UnKEBench.",
    "Computational efficiency: Table 12 shows UnKE requires only 10.5 hours vs. ROME (21 hours), MEMIT (27.75 hours), and MEND (38 hours) on the same hardware — a genuine practical advantage, not a disadvantage.",
    "Ablation study confirms the necessity of both losses (Key Preservation, Key Causal) and both module types (MLP + Attention), with the Key Causal Loss ablation showing dramatic performance collapse (Bert-Score 21.19/26.27) that validates the cause-driven design.",
    "Sequential and batch editing evaluations demonstrate genuine robustness advantages of UnKE over ROME, MEMIT, and fine-tuning baselines across multiple metrics."
  ],
  "weaknesses": [
    "Critical FactScore gap undermining the core claim: UnKE achieves only 42.49 FactScore vs. IKE (w/o ICL) at 93.08 and IKE (w/ ICL) at 94.60. The paper dismisses IKE as 'cheating' because it does not modify parameters, but the 2x factual accuracy gap directly contradicts the claim that UnKE has 'truly edited' the model's internal knowledge. The paper's own case analysis (Table 16) confirms that UnKE output is 'almost identical to the original text' — near-perfect BLEU/ROUGE (>98%) combined with only 42.49 FactScore is strong evidence the model is encoding target text as a retrievable string (memorization) rather than internalizing factual knowledge.",
    "ROME/MEMIT baselines are handicapped by fixed-layer constraint: Table 2 footnote forces ROME and MEMIT to target layer 7, overriding ROME's causal tracing mechanism that specifically identifies the optimal layer. Results for ROME/MEMIT with their default configurations are absent, making the performance comparison potentially unfair.",
    "Locality trade-off in structured editing underanalyzed: Table 3b shows UnKE achieves lower Src-Acc (90.40 vs. MEMIT's 97.60) and Tgt-Acc (68.80 vs. MEMIT's 76.40), indicating higher interference with related knowledge — a meaningful trade-off that receives only one sentence of acknowledgment.",
    "Hyperparameter L=7 lacks principled justification: Table 9 shows performance is stable for L=5–10, which means the specific architectural interpretation (shallow=key encoder, deep=value decoder) is less a principled mechanistic claim and more a tunable hyperparameter. The paper asserts 7 is appropriate based on empirical search alone.",
    "Limited model scope: all experiments use only 7B-parameter models (LLaMA-2-7B-Chat and Qwen1.5-7B-Chat). Whether non-local block KV generalizes to 13B/70B models, or whether L=7 is architecture-independent, is entirely untested.",
    "No statistical significance reporting: Tables 2 and 3 report no confidence intervals or standard deviations. Human evaluation uses only 36 randomly selected samples per method — a very small subset for reliable conclusions — with only moderate inter-annotator agreement (κ=0.57).",
    "AKEW (Wu et al., 2024) and LEME (Rosati et al., 2024) — the most directly competing benchmark proposals — are relegated to Appendix B rather than discussed in the main related work section. The conceptual distinction (editing entity concepts using unstructured text vs. editing the unstructured text itself) is substantive and deserves prominence in the main paper.",
    "Benchmark construction quality: UnKEBench is generated via GPT-3.5-turbo from ConflictQA source texts. While automated quality filters and manual checks are applied, the quality of generated sub-questions and the potential for GPT-3.5-introduced hallucinations or artifacts are inadequately characterized."
  ],
  "nice_to_haves": [
    "Experiments on 13B or 70B models to validate scalability of the non-local KV hypothesis.",
    "Analysis of entity count vs. editing performance to empirically validate the claim that high entity density makes editing harder.",
    "Mechanistic probing experiments (e.g., linear probes at each layer) to justify L=7 beyond empirical grid search.",
    "Sequential editing evaluation beyond 64 samples to characterize catastrophic forgetting at realistic deployment scale.",
    "Comparison with RAG-based baselines more explicitly to delineate when parameter editing is preferable.",
    "Attention pattern visualizations before/after editing to provide intuitive evidence for the non-local storage claim.",
    "Cross-lingual unstructured editing experiments to test generalization beyond English."
  ],
  "novel_insights": "The paper's most interesting empirical insight — that MEMIT performance is largely insensitive to which layers are edited (Figure 2) — supports the 'non-local storage' claim but also admits an alternative interpretation: MEMIT may simply be robust to layer choice, not that knowledge is genuinely distributed. The distinction between editing unstructured text itself vs. editing entity concepts using unstructured text (the AKEW/LEME distinction) is a useful taxonomic contribution to the field. The observation that last-token anchoring eliminates the need for subject localization while preserving causal attention context is a clean and practically useful insight, even if the 'cause-driven' naming is somewhat misleading relative to the causal-tracing literature.",
  "missed_related_work": [],
  "suggestions": [
    "Directly confront the memorization-vs-internalization question: report per-instance correlation between BLEU and FactScore, and evaluate performance when paraphrase questions require lexical reformulation not present in the target text — this would determine whether high BLEU reflects genuine knowledge or string copying.",
    "Report ROME and MEMIT with their default layer configurations alongside the layer-7-matched comparison, so readers can separately assess the effect of layer choice vs. method design.",
    "Include at least one experiment on a 13B model to establish whether the L=7 boundary and non-local KV hypothesis transfer across model scales.",
    "Move the AKEW/LEME benchmark comparison into the main related work section, as the conceptual distinction is central to positioning UnKEBench's contribution.",
    "Add standard deviations or confidence intervals to the main results tables; even three runs per method would substantially increase reliability claims."
  ],
  "score": 6.1,
  "score_justification": "Genuine novel problem formulation and benchmark contribution, with strong lexical/semantic editing results, but a critical FactScore gap (42.49 vs. 94.60 for IKE) and confirmed verbatim copying behavior materially undermine the claim of knowledge internalization; additional weaknesses in baseline fairness, locality trade-offs, and limited model scope prevent a clear accept.",
  "decision": "Accept"
}
```

---

## YbURbViE7l

- GT: Accept (Poster) (avg 6.5)
- Predicted: Reject (5.4/10)
- Match: No

### Final Review

```json
{
  "summary": "GOttack is a targeted structural poisoning attack on GNN node classification that leverages graph orbit theory to reduce the adversarial search space. The paper's central empirical finding is that gradient-based attacks disproportionately select nodes occupying orbits 15 and 18 (peripheral positions in 5-node graphlets), and GOttack formalizes this as a topology-guided attack strategy using precomputed Graph Orbit Vectors to constrain candidate edge perturbations. Experiments span 5 datasets, 3 GNN backbones, 4 attack baselines, and 4 defense models.",
  "strengths": [
    "Genuinely novel mechanistic discovery: Table 5 shows that Nettack targets orbit-15/18 nodes in 97.5% of initial attacks on Polblogs despite these nodes comprising only 9.41% of the graph, and similar trends hold across other datasets and gradient-based methods. This empirical observation—that a universal topological pattern underlies apparently diverse gradient-based attacks—is a meaningful contribution to the GNN robustness literature.",
    "Orbit precomputation is a principled way to reduce the O(2^{|V|×|V|}) adversarial search space to roughly 23% of nodes, with concrete timing numbers (e.g., 0.17s on Cora) and the well-established ORCA algorithm as the implementation backbone.",
    "Comprehensive multi-axis evaluation: 5 datasets (including one heterophilic), 3 GNN backbones (GCN, GIN, GraphSAGE), 4 state-of-the-art attack baselines, and 4 defense models. Code is publicly released, hyperparameters are reported, and 5 random seeds with reported standard deviations (appendix) support reproducibility.",
    "The GNNExplainer case study (Figure 5) and Table 48 provide empirical support for Theorem 1: orbit-15/18 attacks reduce distances to differently-labeled nodes (−0.03) more than to similarly-labeled ones (−0.02), consistent with the periphery-orbit hypothesis even if the effect size is modest."
  ],
  "weaknesses": [
    "The abstract's efficiency claim is directly contradicted by Table 4: GOttack's median end-to-end time (165.37s) is 2.4× slower than SGA's median (68.23s). The '55% of the time required by the fastest competing model' actually refers specifically to GOttack vs. Nettack on the single BlogCatalog dataset, making the abstract statement materially misleading.",
    "The '155 tasks' figure in the abstract is never reconciled with the '28 out of 65 tasks' figure reported in Section 5.1. The counting basis for 155 is opaque, and the two figures create contradictory impressions of the method's win rate.",
    "Empirical margins over the strongest baselines are thin and lack main-body statistical significance: the highest overall misclassification rate is 0.58 vs. SGA/PRBCD's 0.57, and the defense-model overall rate difference is 33.07 vs. SGA's 32.5—within plausible noise. Standard deviations are relegated to the appendix, preventing the reader from assessing significance in the main tables.",
    "Theorem 1 makes a universal claim ('for any graph G, any target node v') that orbits 15 and 18 have the longest expected random walk hitting times. This conflates a local orbit property (defined within 5-node induced subgraphs) with a global graph property (hitting time). The proof is in the appendix and cannot be independently assessed, but the claim as stated is logically fragile—random walk hitting times depend on global topology in ways that are not captured by local graphlet membership.",
    "PRBCD is a global non-targeted poisoning attack while all other compared methods are targeted; including it in the same targeted-attack comparison table without flagging this mismatch inflates the appearance of GOttack's competitiveness in several cells where PRBCD performs anomalously poorly.",
    "Table 4 contains two columns labeled 'Global' and 'Local' that are never defined or explained in the main text, making the timing table uninterpretable without appendix access.",
    "The surrogate loss linearization (Â²XW) is specific to GCN. The paper never explains how this surrogate is adapted for GIN or GraphSAGE, which have fundamentally different aggregation mechanisms.",
    "Only one heterophilic dataset is tested (BlogCatalog, h=0.40), and the paper explicitly acknowledges the orbit proxy holds 'in a weaker form' there. Strongly heterophilic benchmarks (Chameleon, Squirrel, Actor with h<0.3) are absent, limiting the generality of the core homophily-based theoretical argument."
  ],
  "nice_to_haves": [
    "OGB-scale experiments (ogbn-arxiv, ogbn-products) to substantiate scalability claims beyond Pubmed's ~20K nodes.",
    "Ablation on graphlet size k ∈ {3,4,5,6} to determine whether orbits 15/18 remain critical across graphlet conventions or are an artifact of the 5-node choice.",
    "Systematic surrogate-victim transferability analysis (e.g., GIN surrogate attacking GCN victim) to sharpen the universality claim.",
    "Adaptive attack evaluation in the main body rather than relegated to the appendix.",
    "Statistical significance tests (paired t-test or chi-squared) on the orbit-targeting frequency discovery to formalize what is currently a compelling but informal observation.",
    "Experiments on attention-based GNNs (GAT, APPNP) to test whether orbit-based targeting transfers to non-message-passing aggregation.",
    "Non-targeted (global accuracy degradation) attack experiments to cover a broader threat model."
  ],
  "novel_insights": "The paper's most important insight is empirical rather than algorithmic: gradient-based adversarial attack methods appear to independently converge on the same topological target set—nodes occupying peripheral positions in 5-node graphlet structures (orbits 15 and 18)—despite using very different optimization objectives. This emergent universality in attacker behavior, supported by Table 5 across multiple datasets and methods, suggests that graph topology encodes a structural vulnerability that transcends specific attack formulations. The connection to random walk hitting times (Theorem 1) provides an intuitive mechanistic story: peripheral nodes serve as bridges to distant, differently-labeled graph regions, and adding edges to them maximally disrupts the homophily-based label propagation that GNNs rely on. The practical implication—that orbit precomputation can shrink the attack candidate set to ~23% of nodes while preserving competitive misclassification rates—is a useful engineering result even if the theoretical grounding requires tightening.",
  "missed_related_work": [
    "Structack (Hussain et al., 2021) — cited in related work as a structure-centrality-based attack with similar philosophy to GOttack but excluded from experimental baselines without justification; direct quantitative comparison would clarify the marginal value of orbit-based vs. centrality-based candidate selection.",
    "Bojchevski & Günnemann (2019) on adversarial attacks on node embeddings via graph poisoning — cited in references but not compared or positioned against in the context of transferable structural attacks."
  ],
  "suggestions": [
    "Correct the abstract's efficiency claim to accurately state that GOttack requires ~55% of Nettack's time on BlogCatalog, and acknowledge that SGA is faster overall (Table 4 median: 68.23s vs. 165.37s).",
    "Reconcile the '155 tasks' and '28 out of 65 tasks' figures, or replace both with a single clearly defined count with explicit methodology.",
    "Move the orbit-comparison ablations (1519, 1819, 1922 vs. 1518) from the appendix to the main body with confidence intervals, since orbit selection is the central design decision.",
    "Define 'Global' and 'Local' columns in Table 4, or remove them if they refer to methods not described in the main paper.",
    "Explicitly describe how the GCN-based surrogate loss is adapted when attacking GIN and GraphSAGE targets.",
    "Add at least two strongly heterophilic datasets (e.g., Chameleon h=0.23, Squirrel h=0.22) to test the orbit proxy under conditions where the homophily assumption fails.",
    "Separate PRBCD into its own row or table with a note that it is a global non-targeted method, to avoid misleading targeted-attack comparisons."
  ],
  "score": 5.4,
  "score_justification": "Genuine empirical discovery about orbit-targeting universality across gradient-based attacks, but significant abstract misrepresentations (efficiency claim contradicted by Table 4, opaque task-count discrepancy), thin empirical margins without main-body significance tests, a theoretically fragile Theorem 1, an unexplained surrogate adaptation for non-GCN architectures, and insufficient heterophilic coverage collectively prevent the contribution from meeting top-venue standards.",
  "decision": "Reject"
}
```

---

## WpZyPk79Fu

- GT: Accept (Poster) (avg 6.5)
- Predicted: Accept (5.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "AnyPrefer proposes an agentic framework for synthesizing preference data by modeling the process as a two-player cooperative Markov Game between a target model and an external judge model augmented with domain-specific tools (e.g., Grounded SAM, Florence-2, medical imaging models). A feedback mechanism iteratively optimizes prompts for both models. The framework generates Anyprefer-V1 (58K preference pairs spanning NLG, VLU, medical imaging, and robotics) and demonstrates improvements across 21 datasets in four application domains.",
  "strengths": [
    "Genuine multi-domain breadth: Anyprefer-V1 is the only preference dataset covering NLG, VLU, medical imaging, AND robotics simultaneously (Table 1), including the novel Img-Ctrl-Seq modality for robot trajectories — a real community contribution.",
    "Tool-augmented judging addresses the self-rewarding bias problem in a concrete, measurable way: Section 3.5 reports a 22.4% human-evaluated improvement in judge accuracy (67.2% → 89.6%) when tools and feedback are enabled, supporting the core claim.",
    "Ablation studies across all four domains (Tables 5, 9, 12, 15) consistently confirm the individual contributions of tools and feedback, providing structured evidence rather than end-to-end attribution only.",
    "Iterative self-improvement is demonstrated across all four domains (Figure 3, Tables 6, 8, 11, 14), with monotonically increasing performance over three rounds — validating the iterative preference learning design.",
    "Strong NLG result on GSM8K (22.44% → 38.14%) and AlpacaEval (5.20 → 19.25 length-controlled win rate) for LLaMA2-7B shows meaningful gains in a well-studied domain relative to self-rewarding and meta-rewarding baselines."
  ],
  "weaknesses": [
    "GPT-4o circularity contradicts the paper's stated motivation: GPT-4o serves as both the judge model (ranking target-model responses) and the reward model (evaluating preference pair quality) across all domains. The paper identifies shared-weight self-rewarding bias as the core problem, but replaces it with a single proprietary model performing both roles — a higher-level version of the same circular evaluation. This is the most direct tension between motivation and implementation.",
    "The medical image analysis domain uses a fundamentally different methodology — ground-truth labels serve as preferred responses while the target model only generates dispreferred responses — providing a supervision signal unavailable to self-rewarding and meta-rewarding baselines. The 30% improvement in this domain is not a fair comparison to the baselines and is presented as equivalent to the other domains without explicit acknowledgment that this is a qualitatively different, more privileged setup.",
    "No statistical significance tests or confidence intervals are reported across any table. For robotics tasks with 50 binary-outcome trials, differences such as 0.38 vs. 0.32 success rate require binomial significance testing. For VLU benchmarks, differences like POPE 86.98 vs. 86.88 are almost certainly within noise. This omission is pervasive and undermines quantitative claims throughout.",
    "The two-player cooperative Markov Game formalism is decorative. The actual implementation is a prompt-optimization loop with tool-augmented scoring. No MG-specific analysis is provided (state/action spaces are never fully specified, no Nash equilibrium analysis, no convergence guarantees). The 'policy gradient' in Eq. 2 is text-based prompt optimization analogous to TextGrad — not standard policy gradient — but this is not stated explicitly.",
    "Internal numerical inconsistencies: robotics improvement is reported as 16.00% in the abstract but 14.50% in the introduction; medical improvement is reported as 31.05%, 31.02%, and 30.05% in different sections. These are not rounding — the discrepancy between 16.00% and 14.50% is not explained.",
    "The 'first automatic framework for preference data synthesis' claim in the abstract is inaccurate — the paper itself cites RLAIF, UltraFeedback, Nectar, self-rewarding, and meta-rewarding as automatic frameworks. The body more accurately says 'first agentic framework,' but the abstract claim is misleading.",
    "AnyPrefer does not consistently outperform all baselines on VLU: MME_P LLaVA-1.5 baseline (1510.7) > AnyPrefer (1510.1); GQA VLFeedback (63.2) > AnyPrefer (62.2); VisWiz RLHF-V (54.2) > AnyPrefer (54.0). The characterization of 'consistent outperformance' in Section 3.2 is inaccurate.",
    "The tool selection mechanism is underspecified: Algorithm 1 states the set S of selected tools is 'decided by the strategy of πj,' but no concrete description of this selection strategy is provided (fixed rule, prompted decision, or learned). The reward threshold τ — a critical hyperparameter governing data acceptance — is never defined, discussed, or ablated."
  ],
  "nice_to_haves": [
    "Ablation on the number of candidates C (fixed at 5 without analysis) and sensitivity analysis on reward threshold τ.",
    "API cost breakdown: GPT-4o is used extensively as judge, reward model, and tool coordinator; total cost to generate Anyprefer-V1 and per-pair cost are not reported.",
    "Open-source judge model experiments (e.g., LLaMA-3-70B-Instruct or Qwen-72B as judge) to demonstrate accessibility and non-dependence on proprietary APIs.",
    "Individual tool contribution ablation (removing each tool one at a time) to identify which external knowledge sources are essential vs. redundant."
  ],
  "novel_insights": "The most substantive insight is that task-specific external tools (segmentation, detection, captioning, domain-expert models) can serve as grounding mechanisms for a judge model, providing verifiable factual context that a self-rewarding model cannot access. The 22.4% improvement in human-evaluated judge accuracy empirically supports this. The extension of preference data synthesis to robotics action sequences via trajectory cost functions derived from pixel-level segmentation is genuinely novel — no prior preference dataset covers this modality. The iterative cooperation between target and judge via prompt optimization (rather than weight updates) is a practical design choice that keeps the synthesis pipeline lightweight.",
  "missed_related_work": [
    "Toolformer (Schick et al., 2023, arXiv) — directly relevant to the tool-augmented judge design; pioneered the paradigm of LLMs invoking external tools for self-improvement and should be cited in the Related Work alongside the tool-augmented reasoning motivation."
  ],
  "suggestions": [
    "Explicitly acknowledge the GPT-4o circularity: either justify using the same model as judge and reward or introduce a held-out evaluator (human scores or a separate model) for at least one domain to demonstrate independence.",
    "Present the medical domain as a separate experimental condition with its own subsection clearly noting the ground-truth-preferred paradigm, and include a fair ablation comparing this setup to one that does not use ground truth as preferred responses.",
    "Add binomial confidence intervals for all robotics success-rate comparisons and report variance across runs for NLG/VLU benchmarks; this is essential for interpreting differences of 0.02–0.5 in success rates.",
    "Narrow the contributions claim to 'first agentic, tool-augmented, multi-domain preference synthesis framework' and reconcile the numerical inconsistencies (16.00% vs. 14.50%) before submission to any future venue."
  ],
  "score": 5.5,
  "score_justification": "Genuine multi-domain breadth and dataset contribution with empirical improvements, but undermined by a GPT-4o circularity that contradicts the stated motivation, a non-comparable medical domain setup inflating the largest headline result, pervasive absence of statistical significance testing, and a decorative theoretical formalism — collectively material flaws that hold the paper at borderline quality.",
  "decision": "Accept"
}
```

---

## CFMdrcK935

- GT: Reject (avg 6.2)
- Predicted: Reject (3.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper extends the sum operation from Reproducing Kernel Hilbert Spaces (RKHSs) to Reproducing Kernel Banach Spaces (RKBSs), defines an infinite sum of RKBSs via the RKBS characterization theorem, proves compatibility between this sum and the direct sum of feature (Banach) spaces (Proposition 4.2), and uses this to decompose the integral RKBS F_σ(X, Ω) — the hypothesis space of one-layer infinite-width neural networks — into an ℓ¹-sum of p-norm (specifically p=1) RKBSs {L¹_σ(μᵢ)}_{i∈I}, where {μᵢ} is a maximal singular family in P(Ω). The work is analogous to the classical decomposition of M(K) into ⊕¹ L¹(μᵢ), lifted to the RKBS function-space level.",
  "strengths": [
    "Mathematically rigorous and internally consistent: definitions (Definitions 3.2, 3.4, 3.6), the RKBS characterization framework, and the main proofs (Propositions 3.7, 4.2, Theorem 4.4) are self-contained and appear correct, with detailed appendix arguments.",
    "The core construction is natural and well-motivated by a clear analogy: just as M(Ω) ≅_B ⊕¹ L¹(μᵢ) via a classical Banach space result, the paper lifts this to F_σ(X, Ω) ≅_B ⊕¹ L¹_σ(μᵢ), providing a structurally satisfying parallel.",
    "Proposition 4.2 (compatibility between RKBS sum and feature-space direct sum) is a genuinely useful abstraction that could serve as a reusable tool in future RKBS research.",
    "Proposition 5.2 (finite sums of p-norm RKBSs with different σᵢ embed into an integral RKBS on an extended parameter space via Tietze extension) is the more original application result and provides non-trivial structural insight into the flexibility of the integral RKBS class."
  ],
  "weaknesses": [
    "Self-undermining primary application: Remark 5.3 explicitly concedes that Proposition 5.1 — the headline application of Section 5 — follows directly from Corollary 13 of Spek et al. (2022) without requiring the infinite-sum machinery developed in this paper. An application that the paper itself admits does not need the new framework cannot demonstrate the framework's power.",
    "Uncountable and non-constructive decomposition: the maximal singular family {μᵢ}_{i∈I} is guaranteed by Zorn's lemma and is generally uncountable and non-unique. Decomposing a hypothesis space into uncountably many non-constructively obtained components has no clear path toward practical algorithm design, undermining the paper's stated motivation of enabling explicit algorithms for the Representer Theorem.",
    "Critical gap between stated motivation and actual contribution: the introduction and conclusion both claim the decomposition will 'contribute to the development of explicit algorithms for solutions guaranteed by the Representer Theorem in neural network settings,' but no concrete steps, even informal, are taken toward this goal. The gap between 'a decomposition exists' and 'here is how to use it algorithmically' is large and entirely unexplored.",
    "Novelty concerns relative to prior work: the main theorem essentially applies the classical M(Ω) ≅ ⊕¹ L¹(μᵢ) decomposition through the RKBS characterization framework. The individual functional-analytic steps are standard. The paper does not clearly explain what is genuinely new in Theorem 4.4 beyond this combination, especially relative to the identity F_σ(X, Ω) = ∪_{π∈P(Ω)} L^p_σ(π) already established in Spek et al. (2022).",
    "No concrete examples: despite ICLR's ML audience and the paper's own motivation via neural networks, no specific activation function (e.g., ReLU, sigmoid) is ever instantiated in the abstract framework. A reader cannot see what a component L¹_σ(μᵢ) looks like for any concrete σ, which severely limits the paper's communicative value at this venue.",
    "Proposition 3.5's logical bridge is incomplete: the paper asserts that if im(A) were closed in C(X) it would be finite-dimensional, and hence F_σ(X, Ω) ⊊ C(X) for infinite X — but the argument for why im(A) cannot generically be finite-dimensional for 'interesting' σ is not provided, leaving this consequence unjustified."
  ],
  "nice_to_haves": [
    "A constructive or approximate method for identifying the maximal singular family for specific activation functions and parameter spaces.",
    "Generalization bounds or Rademacher complexity estimates derived from the decomposition structure.",
    "Discussion of whether the result extends to non-compact or infinite-dimensional X or Ω.",
    "A connection to the mean-field / particle dynamics interpretation of gradient descent on infinite-width networks.",
    "A Representer Theorem for the infinite sum, even as a partial result under sparsity assumptions."
  ],
  "novel_insights": "The paper's genuine insight is that the RKBS characterization theorem (Theorem 3.3) acts as a functorial device: any isometric isomorphism at the feature-space level (e.g., the classical Θ: ⊕¹ L¹(μᵢ) → M(Ω)) automatically lifts to an isometric isomorphism of the induced RKBSs. This 'transport of RKBS structure' principle (captured in Lemma 4.3 and Proposition 4.2) is the paper's cleanest and most reusable contribution. The compatibility diagram (Figure 1) captures this elegantly. The observation that every RKBS induced by a direct-sum feature space is isometrically isomorphic to the sum of the component RKBSs — and conversely — provides a useful structural lens on the RKBS class.",
  "missed_related_work": [],
  "suggestions": [
    "Work out at least one fully concrete example — e.g., σ(x,w) = ReLU(x·θ − b) on a simple domain — showing what a maximal singular family looks like and what functions the component spaces L¹_σ(μᵢ) contain. This alone would transform the paper's communicative impact.",
    "Either develop at least a sketch of how the decomposition enables or informs an algorithm (greedy component selection, MKL-style optimization over components, etc.) or remove the algorithmic motivation from the introduction and conclusion to avoid overclaiming.",
    "Clarify explicitly in the main theorem statement that the decomposition produces L¹_σ components (p=1 in Definition 3.6) rather than general p-norm RKBSs, since this is what is proved.",
    "Discuss the non-uniqueness of the maximal singular family and whether different choices yield isometrically isomorphic RKBS decompositions — this is essential for the decomposition to serve as a basis for any canonical analysis or algorithm.",
    "Consider submitting to a more appropriate venue such as Applied and Computational Harmonic Analysis, Journal of Machine Learning Research (theory track), or a functional analysis journal, where the mathematical contribution can be evaluated on its own terms without requiring practical algorithmic payoff."
  ],
  "score": 3.5,
  "score_justification": "Mathematically sound but the contribution is incremental (lifting a known Banach-space decomposition through a characterization framework), the primary application is self-undermined by the authors themselves (Remark 5.3), the decomposition is generally non-constructive and uncountably infinite with no path to the claimed algorithmic goal, and the ML relevance for ICLR is insufficient — this paper belongs at a specialized mathematical venue.",
  "decision": "Reject"
}
```

---

## 7liN6uHAQZ

- GT: Accept (Poster) (avg 6.5)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper proposes Sketching for Regularized Optimization (SRO) and its iterative variant (Iterative SRO) for large-scale regularized least squares with convex or nonconvex regularization. The core algorithmic insight is that a single fixed projection matrix suffices for the iterative sketching process, unlike Iterative Hessian Sketch (IHS) which requires resampling at each iteration. The paper's main theoretical contributions are minimax optimal estimation rates O(√(s̄ log d/n)) for sparse convex learning (Lasso) via Iterative SRO, and for sparse nonconvex learning (SCAD, MCP) via one-shot SRO, unified under a 'degree of nonconvexity' framework.",

  "strengths": [
    "The single-fixed-projection-matrix insight is the paper's most practical algorithmic contribution and is rigorously justified by Theorems D.2–D.3: because a single oblivious ℓ₂-subspace embedding suffices to bound the approximation error across all iterates, Iterative SRO avoids the O(Nnd log ñ) resampling cost of IHS, yielding a concrete runtime advantage evidenced in Table 1 (4.71s vs 5.11s at matched approximation error).",
    "Establishing minimax optimal rates O(√(s̄ log d/n)) for sparse nonconvex learning via sketching (Theorem 4.5) is, to the best of this reviewer's knowledge, genuinely novel; the nonconvex case was considered difficult and open, and the 'degree of nonconvexity' abstraction (Definition 5.2) provides a clean unified handle for both convex and nonconvex penalties including SCAD and MCP.",
    "Theorem 4.4 provides a meaningful theoretical justification that the complex-looking Assumption 4 holds under standard conditions (RIP, or simply large n with log d/n → 0), which substantially mitigates concerns about circularity in the nonconvex theory.",
    "Experimental geometric convergence of the approximation error (Figures 1–2, logarithm of error drops linearly with iteration number) directly corroborates Theorem 3.1 across both Gaussian and sparse subspace embeddings, for a range of sketch sizes and problem scales (up to n=10000, d=100000).",
    "The application to Generalized Lasso (Section 7.1, Appendix C.2) and to subspace clustering (Appendix C.4) demonstrates that SRO is not restricted to standard Lasso but applies to a broad class of sparsity-inducing regularizers, extending practical reach."
  ],

  "weaknesses": [
    "The nonconvex minimax result (Theorem 4.5) requires sketch size ñ ≥ Cn/C₃², which is a constant fraction of n. For C₃ only moderately larger than 1 (e.g., C₃ = 2, ñ = n/4), most of the computational savings of sketching evaporate relative to solving the original problem directly. This critical limitation—that the nonconvex result may provide only modest practical acceleration—is acknowledged in passing ('at the expense of having a large constant factor C₁') but is not analyzed quantitatively or foregrounded as a scope limitation. The abstract and introduction implicitly claim sketching benefits for both convex and nonconvex settings equally, which is misleading.",
    "The convex minimax result (Theorem 4.1) critically requires rank(X) = r ≪ n. Many high-dimensional sparse regression designs (e.g., random Gaussian X with d > n) are full-rank, falling outside this regime. This scope restriction is not prominently acknowledged in the abstract or introduction, which state the problem with X ∈ ℝⁿˣᵈ without qualification.",
    "Theorem 3.1's nonconvex branch (Lh-smooth Fréchet subdifferential case) requires X to have full column rank with σ²_min(X) · Lh < (1−ε), which implies d ≤ n. This directly contradicts the high-dimensional sparse regime (d ≫ n) that motivates the paper, and the paper does not reconcile this tension.",
    "Theorems D.2 and D.3—the linchpin results that justify Iterative SRO's fixed-projection approach—are stated only in the appendix; the main paper in Section 5 presents only informal paraphrases. This makes the theoretical chain (Theorem 3.1 ← Theorems D.2/D.3 ← Lemma D.1) unverifiable from the main text alone.",
    "Experimental comparisons are entirely internal (SRO vs. Iterative SRO vs. Iterative SRO-IHS). Table 1 omits a baseline of FISTA applied directly to the original (unsketched) GLasso problem, making it impossible to assess the actual end-to-end speedup of the sketching approach. No comparison to other scalable sparse learning algorithms (e.g., coordinate descent, stochastic variance-reduced methods) is provided.",
    "The headline theoretical claim—minimax rate O(√(s̄ log d/n))—is never directly validated empirically. Table 2 shows Iterative SRO achieves error comparable to β*, but no rate experiment (e.g., log-log plot of estimation error vs. n at fixed d, s̄) is presented to verify the claimed slope."
  ],

  "nice_to_haves": [
    "Comparison with SRHT (Subsampled Randomized Hadamard Transform) sketches, which offer O(nd log ñ) sketch computation and are widely used in practice.",
    "Explicit experiments with SCAD and MCP regularizers (as opposed to only capped-ℓ₁) to directly instantiate Theorem 4.5.",
    "Support recovery (sign consistency) plots as a function of sketch size, supplementing the ℓ₂ estimation error in Table 2.",
    "A phase-transition diagram (sparsity s̄ vs. sketch size ñ) showing empirically when Iterative SRO achieves near-minimax error.",
    "Extension to approximately low-rank matrices (effective rank via singular value decay) to cover practical cases where X is not exactly low-rank.",
    "Discussion of adaptive sketch-size selection when r and s̄ are unknown in practice.",
    "The introduction's three-paragraph defense against Yang & Li (2021) is repetitive (the same comparison appears in the abstract, introduction, Section 5, and the discussion of Section 5); consolidating to one clear statement would improve readability."
  ],

  "novel_insights": "The key intellectual contribution—establishing that a single oblivious subspace embedding provides sufficient approximation control for an iterative refinement process, without needing fresh projections per iteration—is elegant and has practical consequences for large-scale sparse learning. The 'degree of nonconvexity' (Definition 5.2) as a unified quantity capturing how far a regularizer deviates from convexity is a potentially reusable theoretical tool, connecting SCAD, MCP, and capped-ℓ₁ under a common analytical umbrella. The observation that sketching for nonconvex problems requires a larger constant fraction of the full data (vs. sub-linear for convex), visible in the ñ = Θ(n/C₃²) requirement, is itself an interesting information-theoretic insight about the additional difficulty of nonconvex optimization under dimensionality reduction.",

  "missed_related_work": [
    "Kim, Yun & Suh (2022, ICML) 'Randomized Iterative Hessian Sketch for Convex and Nonconvex Optimization' — extends iterative Hessian sketching to nonconvex objectives, directly in the same problem space as this paper's convex/nonconvex divide; the authors compare extensively against Pilanci & Wainwright (2016) but do not cite this more recent work."
  ],

  "suggestions": [
    "Add a direct baseline in Table 1: FISTA on the original (full, unsketched) GLasso problem, so readers can assess the actual end-to-end speedup of Iterative SRO relative to not sketching.",
    "Include a rate-validation experiment: for fixed d and s̄, vary n and plot log(‖β̃* − β̄‖₂) vs. log(n), verifying the predicted slope of −1/2 from the O(√(s̄ log d/n)) rate.",
    "Prominently state in the abstract and introduction that the nonconvex minimax result requires ñ = Θ(n/C₃²), and provide a quantitative discussion of the practical operating regime (what C₃ must be for meaningful sketch-size reduction).",
    "State Theorems D.2 and D.3 formally in the main paper (not just as informal previews in Section 5), so the theoretical chain justifying Iterative SRO is self-contained.",
    "Clarify the tension between Theorem 3.1's nonconvex branch (requiring full column rank, d ≤ n) and the high-dimensional setting (d ≫ n) that motivates the paper — either restrict the claim or prove an alternative condition applicable when d > n."
  ],

  "score": 6.5,
  "score_justification": "Genuine theoretical novelty in establishing minimax-optimal sketching rates for nonconvex sparse learning and the fixed-projection iterative scheme, but meaningfully limited by the near-Θ(n) sketch size required for nonconvex guarantees, the low-rank/full-column-rank restrictions, and experiments that validate geometric convergence without directly benchmarking against unsketched baselines or validating the headline minimax rate empirically.",
  "decision": "Accept"
}
```

---

## avSocG0oFA

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (6.2/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper analyzes the failure modes of DARE (Drop And REscale), a random delta-parameter pruning method, and proposes two improvements: DAREx-q, a post-hoc rescaling factor modification, and AdamR, an in-training L2/L1 regularization on delta parameters. Through Theorem 3.1 (tight Berend-Kontorovich concentration bounds), the authors identify the excessive rescaling factor at high pruning rates and high mean/variance of delta parameters as the two root causes of DARE's collapse. Experiments on BERT, RoBERTa, and multiple Llama2-7B fine-tunes show dramatic gains over vanilla DARE at p=0.99, with DAREx-q achieving >40 pp improvements on COLA and comparable gains on decoder models.",
  "strengths": [
    "Theorem 3.1 provides a tight high-probability bound on |h_i^diff| using Berend-Kontorovich inequalities, with the CLT argument in App. E.1 confirming the bound's √(p/(1-p)) scaling is asymptotically exact — a genuine theoretical contribution that correctly pinpoints the rescaling factor and delta statistics as failure causes.",
    "DAREx-q is a purely post-hoc method requiring no retraining, directly applicable to the large ecosystem of pre-existing fine-tuned models on HuggingFace; the four variants (global/per-layer × labeled/unlabeled) provide practical flexibility for different deployment settings.",
    "Empirical gains are large and consistent: vanilla DARE collapses to ~4% on COLA at p=0.99, while DAREx-q recovers to ~53%, and DAREx-L2 reaches ~57%; gains are replicated across BERT, RoBERTa, and four Llama2-7B mathematical reasoning models.",
    "The paper demonstrates orthogonality to LoRA (Table 4) and applicability to structured pruning (Table 9) and computer vision (ViT on CIFAR-100, Appendix G.1), establishing broader scope beyond NLP.",
    "The identification of a reliable unsupervised proxy (last-layer output change minimized by qe) for task performance is practically valuable, enabling the label-free variant to closely match the labeled variant in most settings (Fig. 1)."
  ],
  "weaknesses": [
    "DAREx-q is conspicuously absent from Table 6, which is the primary evidence for the claim that 'importance-based DPP can outperform random-based DPP when DPs are large.' At p≤0.5 on MetaMath-LLaMA-7B, DARE collapses while WANDA/MP survive, but it is unknown whether DAREx-q would also recover here — a material omission that leaves the key comparative claim unsubstantiated.",
    "Standard deviations are not reported for decoder model results in Table 3, making it impossible to assess statistical reliability of striking claims like 'MetaMath-7B: 34.87 vs. 0.00 at p=0.99'; this is especially important because gains are computed relative to a degenerate baseline.",
    "The theoretical disconnect between penalizing ||ΔW||_F via L2 regularization and bounding the actual influence statistics c_{ij} = ΔW_{ij}x_j is not formally closed: outlier and massive activations (analyzed empirically in App. D) mean that small ||ΔW||_F does not guarantee small Σ_j c_{ij}², weakening the formal justification for AdamR-L2.",
    "The primary stated applications — model merging and multi-task serving — are not empirically demonstrated; all experiments involve single-task pruning only, leaving the model merging use case (the headline motivation) unvalidated in this paper.",
    "At p=0.99 for decoder models, even the best DAREx-q variant shows large absolute gaps from the unpruned model (e.g., MetaMath-7B: 34.87 vs. 65.50 no-pruning, a ~47% relative drop); this is not discussed, and the practical acceptability of such degradation for the stated applications is unaddressed."
  ],
  "nice_to_haves": [
    "Comparison with BitDelta (1-bit delta quantization) as an alternative extreme-compression baseline to contextualize DAREx in the broader delta-compression landscape.",
    "A model-merging experiment (e.g., combining two GLUE task-specific BERT models via DARE vs. DAREx-q) to directly validate the primary stated motivation.",
    "Convergence guarantee or regret bound for AdamR; training curves (App. F) establish empirical stability but a formal theorem would strengthen the optimizer contribution.",
    "Layer-wise heatmap of optimal per-layer q values to explain the empirical finding that per-layer rescaling helps encoders but not decoders.",
    "Sensitivity analysis on validation set size for the labeled qv variant and on the grid-search range for N rounds in Algorithm 2.",
    "Decoder model experiments beyond GSM8K (e.g., instruction-following, code generation) to test generality beyond math reasoning."
  ],
  "novel_insights": "The paper's most genuinely novel insight is that DARE's zero-expectation justification is insufficient: what matters is the absolute magnitude of output perturbations, controlled jointly by the rescaling factor and delta parameter statistics. The tight Berend-Kontorovich bound (matching the CLT scaling for p>1/2) is a principled refinement that enables the unsupervised proxy (last-layer output change) to serve as a reliable stand-in for validation performance — a practically elegant result. The AdamR idea of regularizing (θ_t − θ_P) rather than θ_t is conceptually clean and yields a more stable optimizer than standard weight decay under high regularization, as shown in App. F.",
  "missed_related_work": [
    "TaLoS (ICLR 2025) — contemporaneous work on task arithmetic via localized sparse fine-tuning; relevant as a same-problem alternative that combines sparsity with model merging, complementing the paper's App. C.11 comparison with Diff Pruning.",
    "TIES-merging (Yadav et al., 2023) — directly manipulates delta parameters for model merging by resolving sign conflicts; highly relevant to the model merging motivation and could contextualize when DAREx-q delta sparsity interacts with merging procedures."
  ],
  "suggestions": [
    "Add DAREx-q as a baseline in Table 6 (MetaMath-LLaMA-7B) to determine whether the importance-based vs. random-based DPP comparison holds when the proposed method is included; this is the single most important missing experiment.",
    "Report standard deviations for all decoder model results in Table 3 (even 3 runs would suffice) to enable statistical assessment of gains at p=0.99.",
    "Include at least a brief discussion of why the ~47% relative performance drop at p=0.99 for decoder models is or is not acceptable for the stated applications, and consider framing practical recommendations around realistic pruning rates.",
    "Formally bound the link between ||ΔW||_F and Σ_j c_{ij}² under realistic activation statistics, or explicitly state this as an unresolved theoretical gap, to avoid overstating the formal justification for AdamR-L2."
  ],
  "score": 6.2,
  "score_justification": "Solid engineering contribution with genuine theoretical grounding and large empirical gains, but the core comparative claim (importance-based vs. DAREx-q for large DPs) is left unanswered by a key missing baseline, decoder experiments lack variance reporting, and the primary motivating application (model merging) is never demonstrated.",
  "decision": "Accept"
}
```

---

## uWtLOy35WD

- GT: Accept (Poster) (avg 6.5)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "LLaVA-MoD is a framework for building compact Multimodal Large Language Models (s-MLLMs) via knowledge distillation from a large-scale MLLM (l-MLLM). The approach integrates sparse Mixture-of-Experts (MoE) into the student LLM using sparse upcycling, and employs a progressive two-stage distillation: mimic distillation (dense-to-dense general knowledge, then dense-to-sparse specialized knowledge via KL divergence), followed by preference distillation that uses the teacher as the reference model in KTO-based optimization. A 2B student distilled from a 7B Qwen-series teacher achieves competitive comprehension benchmarks and notably surpasses its teacher and several RLHF-based 7B–13B models on hallucination benchmarks, while delivering 2.5× faster decoding, 26% FLOPs, and 38% memory vs. Qwen-VL-Chat-7B.",
  "strengths": [
    "Strong hallucination mitigation: LLaVA-MoD-2B achieves 11.2%/5.9% response/mention hallucination rates on Object HalBench, surpassing its own 7B teacher (29.7%/23.4%) and multiple RLHF-based 7B–13B models (RLHF-V, LLaVA-RLHF, HA-DPO), demonstrating a genuine and quantified benefit of teacher-referenced preference optimization.",
    "Comprehensive and honest ablations: Tables 5–9 and Appendix B systematically isolate contributions of KD vs. SFT (+8.1%), D2D+D2S vs. D2S (+4.0%), sparse vs. dense architecture (+3.7%), teacher capacity, MoE configurations, routing strategies, and preference-optimization loss choice (KTO vs. DPO), providing strong empirical grounding for each design decision.",
    "Concrete efficiency gains: Table 18 documents 2.5× faster decoding, 26% FLOPs, and 38% memory relative to Qwen-VL-Chat-7B on a single A100-80G GPU, with the full pipeline requiring only 960 GPU hours — these are real, hardware-measured numbers rather than theoretical estimates.",
    "Novel use of l-MLLM as reference model in preference optimization: Treating the teacher rather than an SFT checkpoint as the reference in KTO forces the student to simultaneously align with and exceed the teacher's preference distinctions, enabling the 2B student to surpass 13B RLHF-trained models on hallucination benchmarks.",
    "Code and full reproducibility details: GitHub code is released, and Tables 10–11 provide complete hyperparameters and dataset specifications across all training stages."
  ],
  "weaknesses": [
    "Missing load balancing analysis for MoE training: The paper applies sparse upcycling with Top-k routing but never mentions an auxiliary load-balancing loss (standard in Switch Transformer, DeepSeekMoE, etc.). Without it, expert collapse — where a few experts dominate routing — is a known failure mode. The paper reports no expert utilization statistics, and the puzzling result that 8 experts underperform 4 (Table 12) is left unexplained; expert collapse is the most natural explanation. This is a significant methodological gap for a paper whose central architecture novelty is MoE.",
    "Missing critical ablation for preference distillation's core claim: Using l-MLLM as the reference model (instead of the student's own SFT checkpoint, as in standard DPO/KTO) is presented as the key novelty of the preference distillation stage. There is no direct comparison between these two reference choices, leaving the paper's central preference distillation claim without the most important control experiment.",
    "Training data comparison is contextually misleading: The headline 'only 0.3% of the training data' compares 5M instruction-tuning samples to Qwen-VL-Chat's 1,450M samples, which include large-scale visual pre-training. These are qualitatively different data types; the comparison inflates the apparent efficiency gain and is never clarified in the paper.",
    "Incomplete acknowledgment of baselines that outperform LLaVA-MoD on hallucination: RLAIF-V achieves 8.5%/4.3% on Object HalBench vs. LLaVA-MoD's 11.2%/5.9%, and MiniCPM-V-2 achieves MMHal Score 4.09 vs. LLaVA-MoD's 2.91. The paper's claim of surpassing 'recent RLHF-based models' is overstated for these specific comparisons, which are not discussed.",
    "No statistical significance or variance estimates: All reported numbers are from single training runs. Performance differences as small as 0.3–1.0 percentage points (e.g., Table 3, comprehension after preference distillation) are used to draw conclusions about alignment tax, yet no confidence intervals or repeated runs are provided.",
    "KL divergence direction unjustified: The paper applies forward KL divergence for mimic distillation but does not acknowledge or justify this choice relative to MiniLLM's well-argued case for reverse KL to avoid overestimating low-probability regions — a directly relevant prior work that is cited in related work."
  ],
  "nice_to_haves": [
    "Expert routing visualization by task type (e.g., OCR vs. science vs. grounding tokens) to empirically validate the claim that different experts capture different specialized knowledge.",
    "Training loss and routing distribution curves across distillation stages to confirm stability and progressive specialization.",
    "Cascaded distillation experiment (7B → 1.8B → 0.5B) to address the teacher-capacity gap observed for the 0.5B student in Table 9.",
    "Evaluation on harder reasoning benchmarks (MMMU, MathVista, MMStar) to test generalization of distilled specialized knowledge beyond the current benchmark suite.",
    "On-device latency measurement on actual mobile hardware (e.g., Snapdragon) to substantiate the mobile deployment motivation.",
    "Analysis of comprehension benchmark subtask degradation after preference distillation to characterize the alignment tax more precisely."
  ],
  "novel_insights": "The most genuinely novel element is using the teacher MLLM as the reference model in KTO-based preference optimization, rather than the student's own SFT checkpoint. This forces the student to assign higher probability to positive responses than the teacher does and lower probability to negative responses — a deliberate mechanism for the student to surpass the teacher on preference-aligned metrics. The empirical result that this enables a 2B model to beat its 7B teacher and several RLHF-based 13B models on hallucination benchmarks is the paper's most striking finding. The D2D→D2S progressive sparsification — initializing dense general knowledge before converting to sparse specialized knowledge — is a pragmatic and well-ablated design choice that demonstrates cleaner knowledge transfer than joint training. The observation that KD provides a substantially better optimization signal than SFT for training MoE blocks (+8.1%) is a useful practical finding suggesting that soft teacher distributions help distribute gradient signal across multiple experts more stably than hard label supervision.",
  "missed_related_work": [
    "LLaVA-KD (Cai et al., ICCV 2025) — directly addresses MLLM knowledge distillation with Multimodal Distillation and Relational Distillation; likely concurrent with LLaVA-MoD given ICLR 2025 vs. ICCV 2025 timeline, so non-citation is understandable but worth noting for positioning."
  ],
  "suggestions": [
    "Add an ablation comparing teacher-as-reference vs. student-SFT-as-reference in the preference distillation stage; this is the single most important missing experiment for validating the paper's central novel claim.",
    "Report expert utilization statistics (e.g., selection frequency per expert across tasks) and clarify whether any load balancing auxiliary loss is used; if not, ablate its effect and explain why the 8-expert configuration underperforms.",
    "Clarify the training data comparison in the abstract and Table 18 by distinguishing pre-training data from instruction-tuning data, or restrict the comparison to instruction-tuning samples only.",
    "Acknowledge and discuss the cases where RLAIF-V and MiniCPM-V-2 outperform LLaVA-MoD on specific hallucination metrics to give readers an accurate picture of the competitive landscape.",
    "Provide at least a brief justification for using forward KL divergence in the mimic distillation stage, either theoretically or empirically comparing against reverse KL."
  ],
  "score": 6.5,
  "score_justification": "Solid practical contribution with genuine efficiency and hallucination results, well-supported by comprehensive ablations, but hampered by a missing critical ablation (teacher-as-reference vs. standard reference), unaddressed load balancing for MoE, misleading efficiency framing, and single-run statistics — real issues that a thorough review process would flag, consistent with a borderline accept at ICLR.",
  "decision": "Accept"
}
```

---

## TwZBQKgwdW

- GT: Reject (avg 5.2)
- Predicted: Reject (3.8/10)
- Match: Yes

### Final Review

```json
{
  "summary": "NTK-DFL proposes applying neural tangent kernel (NTK)-based weight evolution to decentralized federated learning (DFL) to address statistical heterogeneity. The method combines per-round neighbor weight averaging with NTK-driven updates (via exchanged Jacobians) and a final model aggregation step, claiming 4.6× fewer communication rounds than leading baselines and over 10% accuracy gain of the aggregated model vs. mean local models on highly heterogeneous benchmarks (Fashion-MNIST, FEMNIST, MNIST) with a 300-client, 2-layer MLP setup.",
  "strengths": [
    "First application of NTK-based weight evolution to the decentralized FL setting — the prior centralized NTK-FL work (Yue et al., 2022) is a natural predecessor but the decentralized extension is non-trivial and the contribution is confirmed as novel by all three reviewers.",
    "Consistent empirical gains across multiple datasets, heterogeneity levels (α ∈ {0.05, 0.1, 0.5}, IID, feature-skew FEMNIST), and network topologies — the convergence advantage is robust and not cherry-picked to a single configuration.",
    "Meaningful ablation studies: per-round averaging ablation (Figure 6) demonstrates that the averaging step is essential; topology and initialization ablations (Figures 9, 10) show robustness; variance–accuracy analysis (Figure 7) provides an interesting mechanistic insight.",
    "Practical engineering contributions: Jacobian batching, data subsampling, and compression (top-k sparsification + quantization + random projection) extend the method toward deployment; compression experiments show 3.9× round reduction over DFedAvg under realistic bandwidth constraints.",
    "The inter-model variance finding — that NTK-DFL generates beneficial variance that model averaging exploits — is an intriguing empirical observation that could inform future DFL design.",
    "A preliminary reconstruction-attack privacy analysis (Appendix E) adds honest acknowledgment of the privacy surface introduced by Jacobian exchange."
  ],
  "weaknesses": [
    "No theoretical convergence analysis whatsoever. ICLR consistently expects at least a convergence sketch for a novel optimization method. The centralized predecessor (Yue et al., 2022) provides theoretical analysis; its absence here is a substantive gap, not a presentation issue.",
    "The NTK linearization is justified only in the infinite-width limit (Jacot et al., 2018), yet all experiments use a two-layer MLP with 100 hidden neurons. The paper never acknowledges this gap between theory and experimental regime, nor discusses when the finite-width approximation is expected to hold.",
    "All experiments use a two-layer MLP on Fashion-MNIST, FEMNIST, and MNIST — among the simplest FL benchmarks. No CNN, ResNet, or transformer results are provided. The paper itself flags CNNs as 'future work,' but for ICLR 2025 this represents a critical gap in validating practical relevance.",
    "The headline '4.6× fewer communication rounds' omits that each NTK-DFL round costs approximately 7.5× more bits than DFedAvg (Figure 16, Appendix D.2). With compression the advantage shrinks to roughly 1.9× more total bits than DFedAvg. This trade-off is buried in the appendix and is absent from the abstract, introduction, and main comparison table — a materially misleading framing of the key claim.",
    "The 10%+ accuracy gain of the aggregated model over mean local client accuracy is presented as a primary contribution, but this compares the aggregated global model to individual local models within NTK-DFL, not against baselines. The same phenomenon appears in any reasonable model averaging scheme; it does not demonstrate superiority over other methods.",
    "No error bars or confidence intervals are reported in any figure or table, despite stochastic topology sampling and random data partitioning. Given that claimed margins (2–3% in Figure 3) are small, variance across seeds is material and unreported.",
    "The critical ablation is missing: NTK evolution alone (without per-round averaging) vs. simple gradient-based update with the same per-round averaging. This would isolate the NTK contribution from the averaging contribution, which is the core scientific question.",
    "Wall-clock runtime is never reported. Each round involves up to 800 ODE timesteps, 6 forward+backward passes per client (self + 5 neighbors), and NTK matrix construction. The computational cost per round could dominate and is uncharacterized.",
    "Final model averaging may require a central aggregator ('secure, centralized manner' is explicitly listed as an option in Section 3.1) or O(M) additional communication rounds for fully decentralized alternatives — neither is analyzed, weakening the DFL premise."
  ],
  "nice_to_haves": [
    "Experiments on CIFAR-10 with a CNN would significantly broaden practical relevance but go beyond the paper's current scope.",
    "A convergence plot with the x-axis as cumulative MB communicated (rather than rounds) would allow a fairer visual comparison.",
    "Comparison against more recent heterogeneity-aware DFL baselines (e.g., GT-SARAH, BEER) beyond the 2017–2022 baselines used.",
    "Quantitative privacy metrics (PSNR, SSIM) in the reconstruction attack appendix, rather than visual-only evidence.",
    "Client dropout and fault-tolerance experiments for practical robustness assessment.",
    "NTK matrix structure visualization to build intuition about what 'more expressive' NTK updates look like vs. gradient updates."
  ],
  "novel_insights": "The most genuinely interesting finding is the synergy between NTK-based evolution and decentralized model averaging: NTK updates appear to generate a beneficial level of inter-model variance that model averaging can exploit, whereas SGD-based methods either produce too little variance (consensus) or too much (harmful drift). This is empirically supported by Figure 7 and the per-round averaging ablation (Figure 6), though the mechanism is only conjectured. The observation that cross-client Jacobian exchange allows each client to see how its data responds to neighbor weight configurations — enabling more globally informed updates without a central server — is the conceptual engine of the method and is a legitimate extension of the centralized NTK-FL idea.",
  "missed_related_work": [
    "Koloskova et al. (2021, NeurIPS) 'Gossip-based Peer-to-Peer Federated Learning' — provides convergence analysis for gossip-based decentralized learning under heterogeneous topology, directly relevant to the topology analysis in Figures 9 and 11.",
    "Lian et al. (2021) 'Decentralized Federated Learning via Mutual Knowledge Transfer' — addresses DFL statistical heterogeneity via a different (knowledge distillation) mechanism; situating NTK-DFL against this approach would sharpen the contribution."
  ],
  "suggestions": [
    "Include at least one convergence theorem or proof sketch — even for a simplified setting (e.g., quadratic loss, fixed graph) — to provide theoretical grounding. The centralized NTK-FL analysis (Yue et al., 2022) is the natural starting point.",
    "Add a total communication cost comparison (bits to reach target accuracy) as a primary result in Figure 2, not just round counts — this is essential to fairly characterize the method's efficiency claim.",
    "Explicitly acknowledge the finite-width NTK approximation gap and discuss empirically whether NTK stability (kernel matrix consistency across rounds) holds during training.",
    "Add the missing ablation: per-round averaging added to DFedAvg/DFedAvgM, to isolate the NTK update contribution from the averaging contribution.",
    "Report error bars (at least across 3–5 seeds) for all main accuracy and convergence results.",
    "Even one experiment with a convolutional network (e.g., a simple LeNet on Fashion-MNIST) would substantially strengthen practical relevance claims.",
    "Report wall-clock time per round alongside communication rounds to give practitioners a complete efficiency picture."
  ],
  "score": 3.8,
  "score_justification": "Genuine novelty in applying NTK to DFL with solid empirical results, but undermined by absent convergence theory, experiments limited to tiny MLPs on simple benchmarks, a misleading headline efficiency claim (rounds without per-round bit cost), missing critical ablation, and no error bars — collectively these are substantive scientific gaps, not presentation issues, placing the paper at borderline-to-clear reject.",
  "decision": "Reject"
}
```

---

## 1p6xFLBU4J

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "GenSE reformulates speech enhancement as a conditional language modeling task over discrete tokens, proposing a hierarchical two-stage framework: an N2S front-end that converts noisy semantic tokens to clean ones, and an S2S back-end that generates clean acoustic tokens conditioned on a token chain prompt. The framework introduces SimCodec, a single-quantizer codec with a novel two-stage codebook reorganization process, and a token chain prompting mechanism [noisy semantic | clean semantic | noisy acoustic] to preserve speaker timbre. Evaluated on DNS and CHiME-4 benchmarks against seven baselines, GenSE demonstrates consistent quality improvements and notably strong generalization under domain shift.",
  "strengths": [
    "Hierarchical N2S→S2S decoupling is strongly validated: ablations show OVL drops 5.6% without hierarchical modeling and SECS collapses from 0.66 to 0.43 without the token chain prompting mechanism (Table 4), providing concrete empirical support for both design choices.",
    "Compelling generalization results on the CHiME-4 out-of-domain benchmark: discriminative baselines (FullSubNet, Inter-Subnet) nearly regress to noisy-input levels, while GenSE retains strong performance — 43.8% VQ improvement and a 25.1% WER reduction relative to FullSubNet — demonstrating the generative paradigm's robustness to domain shift.",
    "SimCodec achieves competitive reconstruction quality (PESQ 3.05, STOI 0.954 at 1.3 kbps) using only 100 tokens/s with a single quantizer, substantially reducing downstream LM complexity, validated against strong codec baselines in Table 3 and the codec ablation in Table 8.",
    "Evaluation is unusually comprehensive for the genre: objective (DNSMOS, SECS, VQScore, WER), subjective NMOS/SMOS, ABX perceptual test, and a separate generalization benchmark all point consistently in the same direction.",
    "The codebook reorganization insight — that group-quantized embeddings selected by usage frequency can be concatenated into a larger single codebook while avoiding the collapse observed with naively large codebooks (Figure 2) — is a practically motivated and empirically supported engineering contribution."
  ],
  "weaknesses": [
    "Factual inaccuracies in empirical claims: Section 4.2 asserts GenSE achieves 'best performance in both quality and speaker similarity' on CHiME-4, but DOSE outperforms GenSE on SECS (0.654 vs 0.650). In Table 1's with-reverb condition, DOSE also achieves higher BAK (3.79 vs GenSE's 3.73). These specific over-claims undermine the credibility of the broader results narrative.",
    "WER is reported only on CHiME-4 (Table 2) and is entirely absent from the primary DNS evaluation (Table 1), despite intelligibility being a central motivation for incorporating semantic information. This omission weakens the intelligibility argument in the paper's core evaluation setting.",
    "The temporal alignment between XLSR semantic tokens (50 Hz / 20 ms frames) and SimCodec acoustic tokens (100 tokens/s at 1.3 kbps) is never described. How these streams are aligned or upsampled before concatenation in the S2S module is a critical reproducibility detail that is missing from both the main paper and the appendix.",
    "LM architecture is insufficiently specified: Appendix A.3 covers only training hyperparameters (optimizer, learning rate, steps). The number of transformer layers, attention heads, and total parameter counts for both N2S and S2S modules are not reported, substantially limiting reproducibility.",
    "The token chain prompt ablation (Table 4) tests only full presence vs. full absence. The contribution of each constituent — noisy semantic tokens, clean semantic tokens, and noisy acoustic tokens independently — is never isolated, leaving the mechanism of action unresolved despite being a novel proposed contribution.",
    "Participant count in the listening study (NMOS, SMOS, ABX) is not reported anywhere in the paper or appendix, making it impossible to assess the statistical reliability of these subjective results.",
    "Genhancer (Yang et al., 2024a), the most conceptually similar published work using discrete codec tokens for generative SE, is cited in related work but absent from Tables 1–2. This is the most pertinent missing baseline.",
    "Hallucination risk is unaddressed: Figure 5 (ABX test) shows listeners prefer GenSE over clean speech more often (39.8%) than they prefer clean over GenSE (38.4%). This plausible-but-unfaithful generation artifact — a well-known failure mode of LM-based systems — is neither discussed nor analyzed, yet it is the most fundamental risk of the proposed approach."
  ],
  "nice_to_haves": [
    "SNR-stratified performance breakdown (e.g., across -5, 0, 5, 10, 20 dB bins) to identify where GenSE's advantage is concentrated.",
    "Per-environment results within CHiME-4 (street, cafeteria, bus, pedestrian area) to understand whether gains are uniform or noise-type specific.",
    "Oracle N2S experiment — feeding ground-truth clean semantic tokens to the S2S module — to quantify the error propagation penalty and identify the current bottleneck.",
    "LM parameter scaling study (small/medium/large LM) to situate GenSE within the scaling narrative relevant to the ICLR community.",
    "RTF comparison for all baseline systems, not just GenSE variants, to contextualize the inference cost.",
    "Multilingual benchmark evaluation to empirically validate the XLSR multilingual motivation stated in the paper."
  ],
  "novel_insights": "The codebook reorganization process is the most technically original sub-contribution: by training group quantizers independently (avoiding residual imbalance), selecting top-N and top-K embeddings by usage frequency, and pairwise-concatenating them into a new N×K codebook, the method recycles high-utility discrete representations to populate a larger codebook without suffering the dead-entry collapse observed in direct large-codebook training. This addresses a known failure mode of single-quantizer codecs elegantly. The token chain prompting sequence [noisy semantic | clean semantic | noisy acoustic] → clean acoustic also represents a thoughtful conditioning design: the noisy acoustic tokens provide timbre-level speaker identity cues that transcend what the denoised semantic tokens alone capture, which is why the SECS drop on removal is so severe (0.66 → 0.43). These are pragmatic but genuine contributions to the discrete-token audio generation paradigm.",
  "missed_related_work": [
    "'From Continuous to Discrete: Cross-Domain Collaborative General Speech Enhancement via Hierarchical Language Models' (2025) — substantial conceptual overlap with GenSE's N2S/S2S hierarchical LM design on discrete tokens; if concurrent, differences in hierarchical decomposition and the SimCodec design should be explicitly discussed."
  ],
  "suggestions": [
    "Correct the factual claims in Sections 4.2 and Table 2 discussion to accurately reflect that DOSE outperforms GenSE on SECS in CHiME-4 and on BAK in the with-reverb DNS condition.",
    "Add WER to Table 1 (DNS primary evaluation) to substantiate the intelligibility claim in the setting where all other metrics are reported.",
    "Describe the temporal alignment procedure between 50 Hz semantic and 100 Hz acoustic token streams in the method section (even one sentence and a footnote clarifying interpolation or decimation strategy).",
    "Report LM architecture details (layers, heads, parameter count per module) in Appendix A.3.",
    "Expand Table 4 ablation to include individual token-chain components (noisy semantic only, clean semantic only, noisy acoustic only) to isolate each constituent's role.",
    "Report participant count and confidence intervals for the NMOS, SMOS, and ABX evaluations.",
    "Include Genhancer as a direct baseline in Tables 1–2, or provide an explicit justification for its exclusion (e.g., code unavailability).",
    "Add a short discussion of hallucination risk — particularly in the context of the ABX result where GenSE is preferred over clean speech — and propose metrics (e.g., character error rate against ground-truth transcription on CHiME-4 samples) to bound content faithfulness."
  ],
  "score": 6.5,
  "score_justification": "Solid engineering contribution with strong generalization results and well-validated design choices, but held back by factual claim inaccuracies, missing reproducibility details, an incomplete ablation of the novel prompting mechanism, and an unaddressed hallucination concern fundamental to LM-based SE.",
  "decision": "Accept"
}
```

---

## T26f9z2rEe

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.1/10)
- Match: Yes

### Final Review

```json
{
  "summary": "DYNMOE introduces two components for automatic MoE hyperparameter selection: (1) a 'top-any' gating mechanism that treats expert activation as multi-label classification using cosine similarity and learnable per-expert thresholds, allowing each token to activate a variable number of experts; and (2) an adaptive training process that dynamically adds experts when tokens activate none and removes unused experts. A diversity-plus-simplicity auxiliary loss encourages sparse, orthogonal expert representations. Evaluated across vision (DomainBed/ViT-S), language (GLUE/BERT-large), and vision-language (MoE-LLaVA-style with three backbones) tasks.",
  "strengths": [
    "Addresses a genuine and practically important problem: MoE performance varies by 1–3% across (K, k) configurations (Figure 1a), and searching over these is expensive at scale — a well-documented pain point confirmed by multiple independent works.",
    "Multi-modal empirical coverage is commendably broad: three modalities, four vision datasets, five GLUE tasks, ten VQA-style benchmarks, and three language model backbones. Results are averaged over three seeds with standard deviations for language tasks (Table 11).",
    "Vision-language efficiency gains are real and well-documented: Table 2 shows 15% fewer activated parameters with StableLM-1.6B while matching or exceeding MoE-LLaVA performance; Table 4 confirms improved throughput (30 vs. 27 tokens/sec) and lower latency with actual hardware measurements.",
    "Architectural insights are a genuine secondary contribution: Figures 5–6 reveal that bottom transformer layers require more diverse expert activation than top layers, and one expert per layer naturally develops a lower activation threshold — an observation independently consistent with DeepSeekMoE's shared-expert design choice.",
    "Code is publicly released and experimental settings are sufficiently detailed (Table 5, Algorithm 1) to support reproducibility for the language and VL tasks."
  ],
  "weaknesses": [
    "The core 'auto-tuning' claim is materially overstated. Users must still specify a maximum expert cap (16 for vision/language, 4 for VL), an initial expert count (6 and 2, respectively), an adaptive-process frequency (100–300 iterations), and auxiliary loss weights — all of which affect results and none of which are analyzed for sensitivity. The method replaces (K, k) tuning with a different but still consequential set of hyperparameters.",
    "The primary comparison in Figure 4 is against the average across all MoE configurations, not the best achievable oracle. Inspecting Tables 9 and 10 reveals that on MNLI DYNMOE (86.37) underperforms the best fixed setting (86.73), and on RTE DYNMOE (73.41) substantially underperforms the best (75.33). These cases where DYNMOE is clearly beaten by a tuned fixed configuration are not acknowledged in the main analysis.",
    "In the language task experiments, DYNMOE converges to K=9 experts with avg k ≈ 6.5–8.0 — activating 72–89% of all experts per token, which approaches a dense model. This renders the efficiency narrative inapplicable to the language setting. The paper does not explain this discrepancy or acknowledge that the efficiency claim is task-specific rather than universal.",
    "The gradient bypass of the sign function in the top-any gating (copying gradients from the output directly to σ(s(x)) − σ(G)) is ad hoc and unjustified. Both other reviewers flagged this. There is no discussion of whether this induces biased gradient estimates, training instability, or divergence, even though it governs whether any expert is activated at all.",
    "Algorithm 1, Line 15 contains a notational inconsistency: the condition reads 'if R_S,e = 0 then Add new expert', but the text correctly states experts are added when R_S ≠ 0. This is a genuine pseudocode error that affects reproducibility.",
    "Vision-language results in Table 2 lack standard deviations across seeds, making it impossible to assess statistical significance for the modest per-benchmark differences (e.g., 77.4 vs. 76.7 VQAv2 for StableLM). Vision task averages in Table 1 also omit standard deviations despite being stated in the appendix as averaged over three seeds.",
    "Table 23 shows that using gating scores as aggregation weights catastrophically degrades performance (VQAv2: 77.4→73.9, MMBench: 63.2→52.1), yet the paper offers no theoretical explanation for why independently computed per-expert scores are unreliable for weighting. This suggests the training dynamics of the top-any gating are non-trivial and the explanation in Remark 3.1 is insufficient.",
    "The auxiliary loss coefficients (weights between diversity loss, simplicity loss, and task loss) are not specified anywhere in the main paper or appendix tables. This is a critical reproducibility gap for a loss that is essential to the method's sparsity behavior."
  ],
  "nice_to_haves": [
    "Direct head-to-head comparison with Expert Choice routing (Zhou et al., 2022), SoftMoE (Puigcerver et al., 2023), and XMoE (Yang et al., 2024) on the same benchmarks to sharpen the contribution's positioning — currently only top-p at one setting is compared (Table 17).",
    "Validation on 7B+ models to confirm that the emergent patterns (shared-expert, top-layer convergence) and efficiency gains persist at scale relevant to modern deployments.",
    "Sensitivity sweep over adaptive-process intervals ({10, 50, 100, 300, 1000} iterations) to characterize robustness to this remaining hyperparameter.",
    "Visualization of expert count evolution over training steps (birth/death timeline) to make the adaptive process concrete and show when expert addition/removal stabilizes.",
    "Theoretical connection between the diversity loss (orthogonality constraint on W_g) and established results in representation learning or expert specialization.",
    "A dense-equivalent baseline at matched total parameter count during training to isolate how much benefit comes from parameter count vs. the adaptive routing mechanism."
  ],
  "novel_insights": "The observation that top transformer layers tend to converge to single-expert activation while bottom layers utilize all experts uniformly is a practically actionable insight that independently validates design choices made in production systems like DeepSeekMoE. The emergence of a low-threshold 'shared expert' per layer without being explicitly designed is a meaningful empirical finding. The multi-label classification framing for gating — treating each expert independently with its own learnable threshold rather than competing top-k selection — is a clean conceptual reframing that naturally accommodates variable sparsity per token.",
  "missed_related_work": [
    "'Top-K or Top-Any? Adaptive Number of Activated Experts per Token' (Huang et al., 2024, arXiv) — addresses essentially the same core problem with a different mechanism; the current paper cites Huang et al. for top-p gating ('Harder tasks need more experts') but this appears to be a distinct work that would strengthen the related-work discussion.",
    "GLaM (Du et al., 2022, ICML) — foundational MoE scaling paper that provides important efficiency context for the motivation; notably absent from the references despite Switch Transformers and GShard being cited."
  ],
  "suggestions": [
    "Explicitly compare DYNMOE against the best oracle fixed-(K,k) configuration in the main results, not just the average. Acknowledge and analyze the cases (MNLI, RTE) where DYNMOE underperforms the best tuned setting.",
    "Add standard deviations across seeds to Table 2 (VL results) and Table 1 (vision results summary) to enable significance assessment.",
    "Specify auxiliary loss coefficients in the main paper and provide at least a brief sensitivity analysis. These are required for reproducibility.",
    "Fix the Algorithm 1 Line 15 pseudocode inconsistency (R_S = 0 vs. R_S ≠ 0 as the trigger condition for adding experts).",
    "Dedicate a paragraph to the language-task near-dense activation finding, explaining mechanistically why DYNMOE converges to near-dense activation in that setting and what this implies for the efficiency claim's scope.",
    "Either provide theoretical justification for the straight-through-style gradient bypass of the sign function, or ablate it against Gumbel-softmax or concrete relaxations."
  ],
  "score": 6.1,
  "score_justification": "Genuine practical contribution with broad multi-modal evaluation and real efficiency gains in VL tasks, but the auto-tuning claim is materially overstated, the comparison to the average rather than oracle obscures cases where DYNMOE underperforms, the language-task near-dense activation undermines the universal efficiency narrative, and key reproducibility details (auxiliary loss weights, VL standard deviations) are missing — collectively holding the paper at borderline accept.",
  "decision": "Accept"
}
```

---

## wCNuEA5MSv

- GT: Reject (avg 5.5)
- Predicted: Reject (4.5/10)
- Match: Yes

### Final Review

```json
{
  "summary": "REPST proposes a PLM reprogramming framework for spatio-temporal forecasting combining: (1) a Dynamic Mode Decomposition (DMD)-based decomposer that factorizes spatio-temporal data into evolutionary modes, and (2) a selective Gumbel-Softmax-based vocabulary selection scheme that maps decomposed signals into the frozen GPT-2's token embedding space. Experiments across six datasets demonstrate strong few-shot and zero-shot generalization, while full-training results are more mixed.",

  "strengths": [
    "Strong few-shot generalization: Table 1 shows consistent and substantial improvements over all eleven baselines in the 1-day training setting across six diverse datasets (traffic, solar energy, air quality), which is the paper's most compelling empirical contribution and aligns with a practically important deployment scenario.",
    "Competitive zero-shot cross-domain transfer: Table 4 shows REPST outperforms TimesFM, FPT, and Time-LLM on nearly all cross-domain transfers, and matches or exceeds OpenCity variants despite being trained on far less domain-relevant data.",
    "Technically sound DMD derivation: The Koopman/DMD mathematical framework in Appendix A.5 is correct and well-developed, providing a principled way to decompose spatio-temporal dynamics into modal components with meaningful spectral structure.",
    "Comprehensive evaluation breadth: Six datasets spanning traffic (METR-LA, PEMS-BAY, Beijing Taxi, NYC Bike), energy (Solar Energy), and air quality (Air Quality), evaluated under full-training, few-shot, and zero-shot regimes with 11 baselines, provides a thorough empirical picture.",
    "Meaningful ablation coverage: Table 5 (PCA vs. DMD), Figure 5/9 (decomposer, PLM, vocabulary components), and Appendix D.2 (per-node spatial decomposition) collectively demonstrate that each component contributes to the observed improvements."
  ],

  "weaknesses": [
    "No statistical significance reporting despite marginal full-training gains: Table 2 shows REPST achieving MAE of 3.63 on METR-LA vs. STAEFormer's 3.60 (REPST is worse), and a four-way tie on PEMS-BAY RMSE (4.33). The appendix notes three repeated runs for few-shot but reports no standard deviations anywhere. The headline claim of 'consistently achieving state-of-the-art' is not supported for the full-training regime.",
    "'Physics-aware' framing is a substantive overclaim: DMD is a purely data-driven spectral method that identifies modes correlated with variance in the observations. It does not incorporate domain-specific physical equations, conservation laws, or constraints. Unlike STDEN (Ji et al., 2022) or AirPhyNet (Hettige et al., 2024) — which the paper cites — DMD does not parameterize known physical differential equations. The paper provides no empirical evidence that extracted modes correspond to identifiable physical phenomena (e.g., visualizing modes overlaid on road networks or correlating with known physical cycles). The Discussion section (D.1) acknowledges this through analogy rather than evidence.",
    "'Expanded vocabulary' is a misnomer: The method selects K=1000 tokens from the existing PLM vocabulary using Gumbel-Softmax — this is a restricted subset selection, not an expansion. The vocabulary space becomes smaller, not larger. This terminological error is inconsistent with the mathematical formulation and the actual mechanism.",
    "Spatial computation through GPT-2 is unexplained and likely absent: The paper claims PLMs model 'relationships among tokens in a 3D geometric space,' but the N×P patch embeddings are concatenated and fed as a 1D sequence to frozen GPT-2 self-attention. No mechanism is described or implemented that imposes 3D or graph-structured computation on the PLM. Spatial structure is captured only implicitly via the DMD preprocessing. The '3D geometric space' claim appears in the contributions, abstract, and method sections but is never operationalized.",
    "Key hyperparameters are not ablated: The number of top-k dominant DMD modes (α), the vocabulary size K (fixed at 1000 with no sensitivity analysis), the Gumbel-Softmax temperature τ, and the number of GPT-2 layers used are all fixed without ablation. These choices likely materially affect performance, especially K, which determines the expressivity of the reprogramming step.",
    "Missing MAPE metric: Appendix A.3 defines MAPE as one of three evaluation metrics but no MAPE results appear in any table in the paper. This unexplained discrepancy suggests either a reporting error or that MAPE results were unfavorable.",
    "Abstract naming inconsistency and baseline count error: The abstract uses 'REEPST' while the paper body consistently uses 'REPST'/'RePST.' Additionally, the abstract claims outperformance of 'twelve state-of-the-art baseline methods' but exactly eleven baselines appear in the results tables. Both are factual errors in the submission."
  ],

  "nice_to_haves": [
    "Experiments with larger or alternative PLM backbones (LLaMA, GPT-4) to demonstrate that benefits scale with PLM capacity and are not GPT-2-specific.",
    "Direct comparison with Fourier decomposition as an ablation, since Figure 1 motivates the decomposition approach specifically against Fourier methods but the full ablation only compares DMD vs. 'transformer encoder'.",
    "Comparison with foundation models (OpenCity, UniST) in the few-shot setting (they appear only in zero-shot), since they represent the strongest relevant competition for generalization.",
    "Learning curves across multiple few-shot percentages (5%, 10%, 25%) to quantify how quickly REPST's advantage shrinks as training data grows.",
    "Visualization of DMD modes as spatial heatmaps on the road network to support or refute the physics-aware framing with direct evidence.",
    "Comparison with PEFT approaches (LoRA-GPT2) as a competing modality-bridging baseline.",
    "Computational runtime comparison with key baselines to substantiate practical feasibility given the quadratic complexity of GPT-2 attention over N×P tokens."
  ],

  "novel_insights": "The most genuinely insightful observation in the paper — supported by Figure 1 and the few-shot ablation — is that even simple decomposition of spatio-temporal signals into spectral components substantially improves PLM comprehension. This suggests that the modality gap between time series and text is partially addressed by providing the PLM with smoother, more separable components rather than raw coupled signals, which is a practically actionable finding. The vocabulary visualization (Figure 8) showing domain-coherent word selection ('dusty', 'rain', 'wind' for air quality; 'rise', 'spread' for spatial dynamics) is anecdotally suggestive of meaningful alignment between DMD-decomposed signals and PLM token semantics, though this lacks quantitative validation. The zero-shot cross-domain transfer results — particularly Solar→Air and NYC→CHI — demonstrate that the PLM's pre-trained knowledge provides non-trivial cross-domain generalization that purely task-trained GNN models cannot offer.",

  "missed_related_work": [
    "Koopa (Liu et al., 2024c) — Already cited in the paper but not discussed in related work as a direct competitor; it uses Koopman-based decomposition for time series forecasting, making it the closest methodological predecessor and deserving explicit differentiation.",
    "STGCN (Yu et al., 2018, KDD) — A seminal foundational work combining graph convolutions with temporal modeling for traffic prediction that is conspicuously absent from the related work despite being among the most-cited papers in the field and establishing the paradigm that subsequent baselines (DCRNN, GWNet) build upon."
  ],

  "suggestions": [
    "Replace 'physics-aware' throughout with 'dynamics-aware' or 'Koopman-informed' and provide mode visualization evidence (e.g., DMD mode amplitudes overlaid on sensor maps) to support any remaining physics interpretation claims.",
    "Rename the 'expanded vocabulary' mechanism to 'selective vocabulary' or 'vocabulary subset selection' to accurately reflect that K tokens are selected from, not added to, the existing PLM vocabulary.",
    "Report mean ± standard deviation across runs in all tables, and add statistical significance tests (e.g., Wilcoxon signed-rank) for comparisons where margins are less than 0.1 MAE.",
    "Add an ablation comparing DMD vs. Fourier decomposition (as originally motivated in Figure 1) and DMD vs. raw signals in the full-training setting, not just few-shot.",
    "Clarify exactly how N×P patches are ordered and processed through GPT-2 attention, and acknowledge explicitly that the spatial inductive bias is confined to the DMD preprocessing stage rather than the PLM computation.",
    "Include MAPE results in results tables or explain why it was defined but not reported.",
    "Fix the REPST/REEPST naming inconsistency and the baseline count claim (eleven, not twelve)."
  ],

  "score": 4.5,
  "score_justification": "Genuine few-shot and zero-shot contributions on six datasets, but undermined by uncorrected factual errors, unsubstantiated 'physics-aware' and '3D geometric space' claims, no statistical significance testing on marginal (sometimes negative) full-training results, missing MAPE metrics, and absence of sensitivity analyses on key hyperparameters.",
  "decision": "Reject"
}
```

---

## LqTz13JS2P

- GT: Accept (Spotlight) (avg 7.2)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper studies generalized principal-agent problems (Stackelberg games, contract design, Bayesian persuasion) where the principal lacks commitment power and the agent uses contextual learning algorithms. The authors provide a unified framework reducing the repeated learning interaction to a one-shot approximate best-response problem, yielding tight, quantitative bounds: the principal can guarantee at least U* − O(√(Reg(T)/T)) against no-regret learners (using a fixed strategy), while no-swap-regret learning caps the principal at U* + O(SReg(T)/T) even under adaptive principal strategies. The asymmetry between the √δ lower bound and the linear δ upper bound is proven tight, and mean-based learners are shown to be exploitable beyond U*.",
  "strengths": [
    "Elegant unifying reduction: the mapping from the repeated learning interaction to a static approximate best-response problem (Theorem 3.1, Lemma 3.2) is clean and immediately yields tight bounds across Stackelberg games, contract design, and Bayesian persuasion without problem-specific arguments.",
    "Genuine tightness of asymmetric bounds: the √δ vs. δ asymmetry between worst-case (OBJ_R(δ) ≥ U* − O(√δ)) and best-case (OBJ̄_R(δ) ≤ U* + O(δ)) objectives is non-obvious, well-intuited in the introduction, and proved tight via Example 4.1 (a specific 2×2 Bayesian persuasion instance with a careful constrained optimization).",
    "New and substantive result for Bayesian persuasion: Corollary 5.2 shows the sender cannot exploit a no-swap-regret learning receiver beyond U* + o(1) even with informational advantage over the receiver, and even if the principal observes the agent's strategy before acting — this follows directly from Theorem 3.4 applying to all adaptive principal strategies, including observation-conditioned ones.",
    "Quantitative refinement of prior work: the prior results of Deng et al. (2019) and Guruganesh et al. (2024) gave only o(1) bounds; this paper replaces those with explicit O(√(Reg(T)/T)) and O(SReg(T)/T) terms, which is strictly more informative and enables concrete comparison across problem instances (Corollaries D.1, D.2, D.3).",
    "Upper bound against adaptive strategies (Theorem 3.4): the clever construction treating (s^t, a^t) as a joint signal from a hypothetical principal strategy π' is technically novel and shows the swap-regret cap holds even when the principal can exploit the agent's revealed history — a strong result.",
    "Bayesian persuasion ≡ cheap talk under learning: the observation that the learning model makes Bayesian persuasion strategically equivalent to cheap talk (since the receiver need not know the prior or signaling scheme) is a clean and non-obvious conceptual contribution."
  ],
  "weaknesses": [
    "Proof of Theorem 3.5 (mean-based exploitation, a stated main result) is informal and incomplete: Claims 2 and 3 in Appendix E.4 rely on phrases like 'the receiver will continue to play action M in most times' and 'after ≈ T/(4√γ) rounds the utility of L should decrease to below 0' without formally invoking the γ-mean-based condition (Definition 3.1) to bound the number of rounds in each regime. The accountings of accumulated signal utilities and the threshold crossing time need precise computation; as written, this is a proof sketch rather than a proof. Since this is one of the four main results, it materially weakens the paper's theoretical completeness.",
    "Principal's knowledge of CReg(T) is a strong and unexamined assumption: Theorem 3.1 requires the principal to know the agent's contextual regret bound CReg(T), yet in practice the principal typically knows neither the exact algorithm nor its regret rate. No robustness analysis is provided — e.g., if the principal overestimates or underestimates CReg(T), how does the achieved utility guarantee degrade? This is a meaningful practical gap.",
    "Bounds become vacuous when the inducibility gap G is small: all bounds in Theorems 4.1 and 4.2 involve factors of 1/G or √(1/G). When G → 0 (near-dominated actions exist), the bounds diverge and provide no useful information. The paper does not discuss what happens in this regime or whether a weaker bound can be salvaged.",
    "Restriction to agents without private information substantially limits scope: the main positive results (upper bound in Result 2, unified lower bound in Result 1) apply only when the agent has no private information. The paper acknowledges this but does not attempt any partial results for agents with limited private information, leaving the scope materially narrower than the title 'generalized principal-agent problem' suggests.",
    "No computational tractability analysis: the paper establishes that the optimal fixed strategy π achieving OBJ_R(CReg(T)/T) − ε exists and suffices for Result 1, but computing it requires solving sup_π min_{ρ ∈ R_δ(π)} U(π, ρ), a two-level optimization. The computational complexity of this problem — even for Bayesian persuasion — is not discussed beyond a citation to Yang & Zhang (2024). A paper claiming to provide actionable principal strategies should at minimum state the computational hardness or tractability."
  ],
  "nice_to_haves": [
    "A complete characterization of which no-regret algorithms (beyond mean-based) are exploitable by the principal — the current gap between 'no-swap-regret is not exploitable' and 'mean-based is exploitable' is not fully characterized.",
    "Explicit tightness of the O(δ) upper bound on OBJ̄_R(δ): the paper proves √δ tightness for the lower bound but does not show OBJ̄_R(δ) ≥ U* + Ω(δ) to fully characterize the asymmetry in both directions.",
    "Extension to continuous agent action spaces, relevant for contract design where effort is often modeled as a continuum.",
    "Sensitivity analysis of bounds to the scaling of |S| (contextual regret grows as O(|A|√|S|T)), which becomes large in Bayesian persuasion with many states.",
    "Illustration figures for the mean-based bait-and-switch strategy and the perturbation geometry in Theorems 4.1/4.2 would significantly aid comprehension."
  ],
  "novel_insights": "The most genuinely novel insight is the intrinsic asymmetry between the principal's worst-case and best-case utility against a δ-approximately-best-responding agent: the worst case scales as √δ while the best case scales linearly in δ. The mechanism is illuminating — a randomized δ-best-responding agent can take √δ-suboptimal actions with probability √δ (causing O(√δ) loss), but the best randomized δ-best-response for the principal is equivalent to a deterministic strategy giving at most O(δ) gain. This √δ vs. δ gap arises purely from the geometry of randomized vs. deterministic approximate best response and is not an artifact of proof technique (Example 4.1 proves tightness). The observation that no-swap-regret learning makes Bayesian persuasion equivalent to cheap talk — with the sender unable to exploit informational advantage beyond o(1) — is a conceptually clean and unexpected consequence of the framework. Lemma F.3's approach of connecting OBJ_R(δ) to OBJ_D(Δ) via a redistribution argument is a technically elegant device enabling the √δ lower bound from a linear-δ deterministic bound.",
  "missed_related_work": [
    "Dütting, Gkatzelis & Roughgarden on simplicity and approximation in mechanism design — potentially relevant to the computational aspects of finding principal strategies achieving the derived bounds, though the connection requires verification."
  ],
  "suggestions": [
    "Complete the proof of Theorem 3.5 (Appendix E.4): formally bound the number of rounds in each phase using the precise γ-mean-based condition (Definition 3.1), compute the accumulated utility thresholds rigorously, and convert the 'approximately' and 'with high probability' language into formal probability bounds with explicit constants.",
    "Analyze robustness of Theorem 3.1 to misspecification of CReg(T): show that if the principal uses a strategy calibrated to CReg'(T) when the true bound is CReg(T), the achieved utility is at least OBJ_R(CReg(T)/T) − O(|CReg'(T) − CReg(T)|/T), or provide a counterexample showing this fails.",
    "Add a discussion of the small-G regime: characterize qualitatively (or with a modified theorem) what guarantees survive when the inducibility gap G is small but positive, or identify structural conditions on the game that ensure G is bounded away from zero.",
    "Clarify in the main text (not just the appendix) that Theorem 3.3's agent construction requires the agent to know the principal's overall policy — label it explicitly as a minimax/adversary argument to avoid misleading readers about what natural learning algorithms achieve."
  ],
  "score": 7.0,
  "score_justification": "Solid theoretical contribution with a clean unifying reduction, tight and novel asymmetric bounds, and new results for Bayesian persuasion, marred by an informal proof for one of four stated main results (Theorem 3.5) and a strong unexamined assumption (principal knows CReg(T)), but not enough to overturn acceptance.",
  "decision": "Accept"
}
```

---

## x07rHuChwF

- GT: Reject (avg 5.0)
- Predicted: Reject (5.2/10)
- Match: Yes

### Final Review

```json
{
  "summary": "This paper introduces Geoperception, a benchmark for evaluating MLLMs' ability to accurately perceive 2D geometric information (7 tasks derived from Geometry-3K), an empirical study of MLLM design choices for geometric perception yielding three architectural/training lessons, and Euclid, a family of models (ConvNeXt encoder + 1.5B Qwen2 LLM) trained with an adaptive curriculum on purely synthetic geometric data. Euclid achieves strong results on primitive geometric perception tasks—e.g., 78.94% vs. Gemini-1.5-Pro's 24.42% on PointLiesOnLine—demonstrating that synthetic-only curriculum training generalizes to real textbook diagrams.",
  "strengths": [
    "The Geoperception benchmark is a genuine and needed contribution: it isolates low-level visual perception from reasoning, which prior benchmarks conflate. The construction pipeline (logical-form extraction from Geometry-3K, GPT-4o-mini verification, stratified question generation) is well-documented and yields 98%+ annotation accuracy on random inspection.",
    "The curriculum learning finding (Lesson 3) is the best-supported empirical result: Fig. 6 reports validation loss on a held-out Shape 3 set under multiple strategies, showing that pure Shape 3 training fails entirely while staged curriculum reliably converges. This is a counterintuitive and practically important finding.",
    "Euclid's results on PointLiesOnLine are genuinely impressive—78.94% vs. Gemini-1.5-Pro's 24.42%—achieved with a 1.5B LLM backbone trained exclusively on synthetic data, demonstrating that domain-targeted synthetic curriculum training can dramatically outperform much larger frontier models on narrow geometric perception.",
    "The adaptive (on-the-fly) progressive training methodology (Eqs. 2–3, Section 5) with exponential attenuation and stage-advancement thresholding is an original and practical contribution that avoids static dataset construction.",
    "The paper provides strong reproducibility support: pseudocode for all four data synthesis algorithms, geometry shape generation code, evaluation prompt templates, and Hugging Face links for all encoders."
  ],
  "weaknesses": [
    "Lessons 1 (CNN > ViT) and 2 (tuning > freezing) in the empirical study are based exclusively on training loss convergence speed, without any test accuracy validation for the encoder comparison. The relationship between faster training loss convergence and superior held-out benchmark performance is assumed but never demonstrated. Fig. 18 (Appendix) shows test accuracy only for LLM size ablation, not for the central encoder comparison. This undermines the credibility of the paper's primary architectural conclusions.",
    "The evaluation metric (Eq. 1) assigns zero whenever any prediction is not in the ground truth (P ⊄ G), regardless of how many correct predictions were made. This severely penalizes over-prediction and produces starkly different model rankings compared to the recall-based and binary metrics reported in Appendix Tables 6–7 (e.g., Molmo's POL score changes from 1.75 to 50.25 under recall). No principled justification is provided for preferring Eq. 1 over F1 or precision/recall decomposition, and the main text does not alert readers to these ranking differences.",
    "No human performance baseline is reported anywhere in the paper. For a benchmark introduced as 'straightforward for humans' and designed to measure an absolute shortcoming of MLLMs, the absence of a human ceiling is a critical omission standard for ICLR benchmark papers.",
    "Table 4 (Euclid evaluation) compares only against Pixtral-12B and Gemini-1.5-Pro, omitting 7 of the 9 models evaluated in Table 2 without explanation. Most notably, Qwen2-VL-7B (the natural open-source comparison given Euclid's 1.5B Qwen2 LLM backbone) is excluded despite scoring 40.75 overall in Table 2.",
    "Euclid is trained and evaluated on only 4 of 7 Geoperception tasks, leaving out Parallel, Perpendicular, and Equals. The 'Average' in Table 4 covers a different task subset than Table 2, making the scores non-comparable. This is never flagged explicitly.",
    "The observed multi-task training degradation (POL accuracy drops from 78.94% to 55.22% under joint training) is noted but not investigated. Two speculative hypotheses are offered without supporting ablations, leaving an important negative result unexplained.",
    "The benchmark covers only 2D geometry tasks from high-school textbooks. The paper's stated motivation—robotics, medical imaging, autonomous driving, manufacturing—is not validated with any downstream evaluation. Whether Geoperception performance improvements translate to any of these application domains is entirely untested."
  ],
  "nice_to_haves": [
    "Comparison against specialized computer vision models (line detection, geometric property estimation) to contextualize MLLM performance in an absolute sense.",
    "Cross-dataset generalization experiments on GeoQA, Geomverse, or MathVista geometry subsets beyond Geometry-3K images.",
    "Mechanistic analysis (e.g., GradCAM visualizations) explaining why ConvNeXt outperforms ViTs for geometric perception tasks.",
    "Systematic scaling of the LLM backbone beyond 1.5B in the main paper (not just Appendix).",
    "Extension to the annotated geometry tasks (Parallel, Perpendicular, Equals) using alternative synthetic generation strategies.",
    "Few-shot or chain-of-thought prompting baselines for frontier models to establish stronger upper bounds."
  ],
  "novel_insights": "The most compelling insight is that curriculum learning on progressively complex synthetic geometric shapes enables convergence on tasks where direct training fails even with large datasets—this is well-evidenced by the validation loss curves in Fig. 6 and represents a principled application of curriculum learning theory to the MLLM domain. The finding that ConvNeXt encoders outperform ViTs for low-level geometric perception is intriguing and directionally plausible (CNNs' inductive bias toward local edge structure vs. ViTs' global attention), but remains unvalidated at the test accuracy level. The demonstration that purely synthetic curriculum training generalizes to real textbook geometric diagrams, achieving 3× improvement over Gemini-1.5-Pro on PointLiesOnLine, is the paper's most practically significant finding.",
  "missed_related_work": [],
  "suggestions": [
    "Validate Lessons 1 and 2 with test accuracy on Geoperception tasks (not just training loss), even on a subset of encoder variants—this is essential for the empirical study's conclusions to be credible.",
    "Adopt F1 or precision/recall as the primary evaluation metric with Eq. 1 as a secondary metric, and justify the metric choice against the substantially different rankings in Appendix Tables 6–7.",
    "Report human performance on a representative subset of Geoperception tasks to establish the human ceiling the paper claims models fall far short of.",
    "Include all baselines from Table 2 in Table 4, or explicitly justify each omission; at minimum, include Qwen2-VL-7B as the natural open-source scale comparison.",
    "Conduct a structured error analysis of Euclid's failures (beyond the single example in Fig. 7), categorizing errors by diagram complexity, annotation density, and number of geometric elements."
  ],
  "score": 5.2,
  "score_justification": "Genuine contributions (Geoperception benchmark, curriculum learning insight, strong Euclid results on primitive tasks) are offset by the empirical study's primary lessons being validated only by training loss, the unjustified non-standard evaluation metric that produces different rankings than standard alternatives, the absence of human baselines, and selective baseline comparison in the main results table—collectively these undermine the paper's core empirical claims.",
  "decision": "Reject"
}
```

---

## EpnZEzYDUT

- GT: Accept (Poster) (avg 6.5)
- Predicted: Reject (5.8/10)
- Match: No

### Final Review

```json
{
  "summary": "MADiTS proposes a diffusion-based data augmentation pipeline for offline multi-agent reinforcement learning that addresses both temporal imbalance (suboptimal decision windows) and spatial imbalance (underperforming individual agents). The method generates trajectory segments via a conditioned diffusion model, filters them with a bidirectional dynamics constraint, and uses integrated gradient-based credit assignment to identify and partially re-noise underperforming agents' trajectories. Experiments across MPE, SMAC, SMACv2, and MAMuJoCo on synthetically constructed imbalanced datasets show consistent improvements over baselines, approaching balanced-dataset performance.",
  "strengths": [
    "Clear and meaningful problem decomposition into temporal vs. spatial imbalance in offline MARL datasets — the distinction is novel and well-motivated for cooperative settings where agents share a global reward.",
    "Coherent multi-module design: the bidirectional dynamics constraint (forward + inverse MLP), the IG-based credit assignment, and partial noising complement each other logically, and the ablation in Figure 4(a) empirically confirms each module's contribution.",
    "Algorithm-agnostic pipeline: MADiTS is shown to improve three qualitatively different downstream algorithms (BC, OMIGA, CFCQL), strengthening the claim that the benefit comes from dataset quality, not algorithm-specific tuning.",
    "Comprehensive empirical coverage across four benchmarks (MPE, SMAC, SMACv2, MAMuJoCo), multiple map sizes, and two imbalance levels (exp-m, exp-s), with 5 random seeds and standard errors reported throughout.",
    "Appendix H.2 demonstrates that MADiTS also improves performance on balanced datasets of varying quality (Expert, Medium, Medium-Replay, Random), indicating broader utility beyond the imbalanced-data framing.",
    "Appendix H.5 provides a Kendall correlation analysis validating that the IG-based credit assignment produces rankings correlated with ground-truth agent contributions in CN, lending empirical support to a key mechanistic claim."
  ],
  "weaknesses": [
    "All imbalanced datasets are synthetically constructed by the authors via controlled perturbations (random actions for fixed durations or entire episodes). No evaluation is performed on naturally imbalanced offline MARL datasets such as OG-MARL (Formanek et al., 2023), which the paper cites. This circularity — constructing the problem and solving it — limits generalizability claims and is a real experimental gap.",
    "SIT (Tian et al., 2023), the most directly competing method for spatial imbalance in offline MARL (uses attention-based reward decomposition and trajectory sharing), is acknowledged only in Appendix B and is never compared against experimentally. Given the direct thematic overlap with MADiTS's spatial correction module, its absence makes it impossible to assess whether the IG + partial-noising approach is genuinely superior.",
    "The integrated gradient decomposition proof (Appendix C, Equation 7) silently requires that the reward function be expressible as a potential difference r(x_t) = R̃(x_t) − R̃(x_{t+1}). This is a non-trivial constraint that is neither stated as a theorem condition nor acknowledged as a limitation in the main text. Its validity is validated only on CN (a smooth, distance-based reward); it is unverified for SMAC battle-won-rate signals or MAMuJoCo locomotion rewards.",
    "Improvements on larger, harder environments (12m, SMACv2 terran/zerg) are substantially smaller than on simpler scenarios, and several results fall within standard error of the MADiff baseline (e.g., 12m-OMIGA-exp-m: MADiff 0.65 ± 0.21 vs. MADiTS 0.65 ± 0.12; zerg-CFCQL-exp-m: MADiff 0.46 vs. MADiTS 0.50). This suggests limited scalability to more complex environments.",
    "The δ_rank threshold for identifying underperforming agents is set to 2.33 in most environments regardless of actual performance disparity. In a 3-agent scenario this mechanically flags the bottom-ranked agent every episode, causing unnecessary partial noising even when all agents are performing adequately. No sensitivity analysis for δ_rank is reported in the main paper.",
    "Some results in Table 1 have extremely large standard errors relative to means (e.g., CN-CFCQL exp-m: 23.02 ± 69.69), making the bolded best results statistically unreliable. No formal significance tests are reported despite variance levels that would warrant them."
  ],
  "nice_to_haves": [
    "Ablation of bidirectional vs. unidirectional dynamics constraint to confirm the bidirectional design is necessary rather than just forward-only.",
    "Sensitivity analysis for δ_rank hyperparameter across multiple environments.",
    "Analysis of trajectory discard/utilization rates per environment to understand how often the dynamics constraint truncates generated segments.",
    "Credit assignment accuracy validation (Kendall correlation) beyond CN to SMAC and MAMuJoCo environments.",
    "Training curves (return vs. training steps) to distinguish whether MADiTS accelerates learning or primarily improves asymptotic performance.",
    "Ablation of the circular shift technique in isolation to quantify its independent contribution to performance gains.",
    "Scaling experiments beyond 12 agents to assess tractability of joint-observation diffusion in larger multi-agent systems."
  ],
  "novel_insights": "The decomposition of offline MARL data quality into temporal imbalance (low-quality windows within episodes) and spatial imbalance (underperforming individual agents) is a useful and underexplored framing. The application of integrated gradients to multi-agent credit assignment for the purpose of data augmentation — rather than for policy training — is a creative repurposing of an interpretability tool. The finding that MADiTS sometimes exceeds balanced-dataset performance (e.g., PP and 2mvs1z) suggests that trajectory stitching can synthesize coordination patterns not present in the raw data, rather than merely recovering lost quality.",
  "missed_related_work": [],
  "suggestions": [
    "Add a direct experimental comparison against SIT (Tian et al., 2023) on at least one benchmark where spatial imbalance is the primary challenge, or provide a principled justification for its exclusion.",
    "Evaluate on at least one naturally imbalanced offline MARL dataset (e.g., from OG-MARL) to demonstrate generalizability beyond synthetically perturbed data.",
    "Explicitly state the potential-function assumption (r(x_t) = R̃(x_t) − R̃(x_{t+1})) as a condition in the credit assignment theorem and acknowledge it as a limitation for environments with non-potential reward structures.",
    "Replace or supplement the fixed δ_rank heuristic with a magnitude-aware criterion (e.g., agents are flagged only when their contribution falls below a fraction of the team mean, not just by rank) to avoid unnecessary partial noising when all agents are performing similarly."
  ],
  "score": 5.8,
  "score_justification": "Solid engineering contribution with comprehensive experiments and coherent design, but evaluation is entirely on self-constructed datasets, the closest competing method (SIT) is absent from comparison, a hidden reward assumption undermines the theoretical module, and gains on harder environments are marginal — collectively placing this at borderline.",
  "decision": "Reject"
}
```

---

