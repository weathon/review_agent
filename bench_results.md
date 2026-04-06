# ICLR Benchmark Results

Date: 2026-04-05 23:47
Critic/Merger: deepseek/deepseek-v3.2 (OpenRouter)
Neutral: deepseek/deepseek-v3.2, Related Work: deepseek/deepseek-v3.2:online (OpenRouter)

## P1qlqfRtgo

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (2.3/10)
- Match: N/A

### Final Review

## Summary
This paper presents a comparative study of three neural network architectures (MLP, a custom U-Net-style residual network, and a DeepONet-inspired model) for learning the stiff chemical kinetics of a hydrogen-air thermal explosion. The key finding is that the residual network consistently outperforms the others in mean squared error and prediction stability on a realistic, multi-regime dataset.

## Strengths
- The paper establishes a clear, controlled comparative framework. All models are trained and evaluated on the same substantial dataset (50k training samples) spanning wide thermodynamic ranges, and performance is reported with confidence intervals, showing a statistically significant advantage for the residual network.
- The work addresses a recognized and computationally expensive bottleneck in reactive flow simulation—solving stiff chemical ODEs—providing a concrete, application-driven evaluation of architectural choices.

## Weaknesses
- **The training protocol is ambiguously described, affecting reproducibility.** The loss function (Eq. 4) is defined over a 30-step prediction horizon, but the text does not clarify if this is achieved via an autoregressive rollout during training or another method. This omission makes it difficult to interpret the source of error accumulation and to replicate the study.
- **Evaluation relies solely on aggregate MSE, lacking domain-specific metrics critical for combustion.** The paper does not report errors on physically meaningful quantities like ignition delay time, peak temperature, or species concentrations at key phases. A model could have low aggregate MSE but fail on these critical features, limiting the assessment of its practical utility for the stated application.
- **The claim that neural networks can "significantly speed up" the process is unsupported.** No data on inference time, parameter counts, or computational cost compared to the traditional ODE solver is provided. For a surrogate modeling paper aiming at acceleration, this omission is a significant gap in the evaluation.

## Nice-to-Haves
- A more detailed error analysis, such as breaking down performance by combustion regime (e.g., induction vs. explosion) or key species, would provide deeper insight into when and why the architectures succeed or fail.
- An ablation study on the components of the residual network (e.g., the role of skip connections versus the specific layer dimensions) would help attribute the performance gains more precisely.
- Implementing a more canonical neural operator baseline (e.g., a standard DeepONet) would strengthen the comparison to operator-learning methods, though the paper's focus is a controlled comparison of its specific implementations.

## Novel Insights
The paper provides a clear, empirical demonstration that even relatively simple architectural inductive biases—specifically, residual/skip connections—can yield substantial improvements in predictive accuracy and stability for learning stiff chemical kinetics. This finding is valuable for practitioners in scientific machine learning, underscoring that architecture selection is a critical, non-trivial step when building surrogates for complex physical systems, even before incorporating more elaborate physical constraints.

## Suggestions
- Clarify the multi-step training procedure in the methodology section. Explicitly state whether predictions are fed back autoregressively during training and how the loss is computed across the 30-step horizon.
- Incorporate at least one domain-specific metric, such as the error in predicting ignition delay time or peak temperature, to ground the evaluation in the physics of the problem.
- Report the inference time per step for each model and, if possible, compare it to the baseline ODE solver to substantiate the speed-up claim.

---

## 9WiPZy3Kro

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces GROUNDCUA, a large-scale, human-annotated dataset for grounding natural language instructions to UI elements in desktop applications, comprising 56K screenshots and over 3.56M element annotations from 87 open-source applications. Using this dataset, the authors develop the GROUNDNEXT family of vision-language models (3B and 7B parameters) trained via supervised fine-tuning (SFT) and reinforcement learning (RL). The models achieve state-of-the-art results on multiple GUI grounding benchmarks while using significantly less training data than prior work, demonstrating that high-quality, expert-driven data enables data-efficient training of robust computer-use agents.

## Strengths
- **High-quality, novel dataset:** GROUNDCUA is the first large-scale, densely annotated, human-verified dataset focused on desktop grounding. It features high-resolution screenshots, fine-grained elements (average area 0.13% of image), and broad coverage (87 applications across 12 categories), addressing a clear gap in existing automated or synthetic collections (Table 1, Figure 5).
- **Data-efficient, high-performing models:** GROUNDNEXT models, trained on only 700K SFT samples, outperform prior SFT models trained on millions of samples across five major benchmarks (ScreenSpotPro, OSWorld-G, MMBench-GUI, ScreenSpot-v2, UI-Vision). The 3B model achieves performance comparable to much larger models (e.g., JEDI-7B) in agentic evaluation, demonstrating strong practical utility (Table 2, Table 4).

## Weaknesses
- **Potential bias from open-source application selection:** The dataset exclusively uses open-source applications, which may not fully capture the UI diversity, proprietary layouts, or domain-specific artifacts of widely used commercial software (e.g., Microsoft Office, Adobe Suite). While the authors argue these serve as valid proxies, generalization to closed-source ecosystems is not rigorously tested, leaving a gap in real-world applicability.
- **Modest and incremental RL contribution:** RL post-training provides consistent but small improvements (e.g., +2.0 average points for the 3B model). The analysis correctly attributes this to the strong SFT baseline, but the RL stage uses a simple, handcrafted reward function. The paper does not explore more sophisticated reward designs (e.g., learned rewards) that might yield larger gains, which is a missed opportunity given the dataset's potential for reward modeling.

## Nice-to-Haves
- A side-by-side visual comparison of annotation density between GROUNDCUA and other datasets (e.g., OS-Atlas, JEDI) for a common application would help visually substantiate the claim of "denser annotations."
- A more quantitative breakdown of cross-domain error modes (e.g., failures correlated with aspect ratio, specific UI patterns, or semantic shifts) would strengthen the analysis of generalization limits to mobile and web platforms.

## Novel Insights
The primary novel insight is that high-quality, densely grounded, human-verified supervision can substitute for massive data scale in training GUI grounding models. The paper provides evidence that a carefully curated dataset of 700K samples enables models to outperform those trained on orders-of-magnitude more data, challenging the prevailing scale-up paradigm in the field. Furthermore, the analysis reveals that the marginal gain from RL is inversely correlated with the strength of the SFT baseline, suggesting that in this setting, high-quality SFT establishes a strong ceiling that RL can only modestly refine.

## Suggestions
- To better support the claim that open-source applications are effective proxies for real-world coverage, consider adding a performance analysis on a benchmark that includes both open-source and proprietary software (if available) or a discussion correlating performance on specific application categories with their similarity to training data.
- In the error analysis (Appendix E), provide a quantitative categorization of failures (e.g., percentage of errors that are near-misses vs. semantic confusions, per UI element category) to better pinpoint the model's specific weaknesses and validate claims about strengths in icon recognition.

---

## do4hqhMBiu

- GT: Reject (avg 0.0)
- Predicted: N/A (1.3/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a diffusion-based image-to-image augmentation workflow designed to increase the diversity of a synthetic medical imaging dataset (AcuSim) while preserving annotated anatomical landmarks (acupoints). The method integrates existing control modules (IP-Adapter, IC-Light) with Stable Diffusion and an automated controller to alter environmental factors like lighting. Evaluation involves training a CNN for acupoint localization and measuring facial landmark drift using MediaPipe.

## Strengths
- **Addresses a practical, high-stakes problem**: The work tackles the challenge of generating diverse training data for medical/biometric applications where data is scarce and annotations are costly, focusing on preserving fine-grained anatomical landmarks to avoid re-labeling.
- **Systematic and automated pipeline**: The authors develop a controller program and custom nodes to handle dataset heterogeneity (e.g., gender, hairstyle), demonstrating a practical engineering solution for adapting the diffusion process to different sample types in an automated manner.

## Weaknesses
- **Evaluation does not substantiate the core claim**: The paper's primary claim is the preservation of acupoint landmarks, but the geometric consistency evaluation uses only 8 generic facial landmarks from MediaPipe, not the 174 annotated acupoints. This misalignment leaves the central claim unverified.
- **Missing baseline comparisons undermines claims of utility**: The CNN evaluation reports high performance when training on the augmented set but provides no comparison to training on the original AcuSim dataset. Without this, the claim that augmentation "enriches" the dataset or maintains performance is unsupported. Furthermore, no comparison to other augmentation methods (traditional or modern) is provided.
- **Methodological description is insufficient for reproduction**: Critical details are omitted, including the exact text prompts, the controller program logic, the choice of scene image for lighting guidance, and a clear explanation of key parameters (e.g., splice ratio \(t_0\)). The provided equation is garbled, and the roles of IP-Adapter/IC-Light in preserving acupoints are described only vaguely.
- **Major presentation issues hinder understanding**: The abstract is critically incomplete (cutting off at "99.99"). Section 5.2 contains a substantial duplicate paragraph, confusing the results. Numerous formatting artifacts (strikethroughs, broken references) further reduce clarity.

## Nice-to-Haves
- **Visual examples and landmark drift visualization**: Side-by-side comparisons of original and augmented images, along with visualizations of landmark offsets, would help assess variation and preservation qualitatively.
- **Quantitative diversity assessment**: Metrics like FID or LPIPS could substantiate the claim of increased dataset diversity beyond anecdotal description.
- **Ablation study on control parameters**: An analysis of how key parameters (IP-Adapter weight, CFG scale) affect the trade-off between diversity and landmark preservation would strengthen the methodological rationale.

## Novel Insights
None beyond the paper's own contributions. The work applies an integration of existing diffusion control tools to a specific medical imaging problem but does not introduce a novel algorithmic insight or uncover new phenomena.

## Suggestions
- **Evaluate preservation of the actual acupoints**: Use the available acupoint annotations to compute the displacement of these points between original and augmented images, providing direct evidence for the core claim.
- **Add essential baseline comparisons**: Train and evaluate an identical CNN on the original AcuSim dataset (with and without traditional augmentation) and compare its performance to the model trained on the proposed augmented set.
- **Clarify methodology and release details**: Provide the full set of prompts, controller logic, and cleaned-up technical descriptions. Releasing code and a sample of the augmented dataset would greatly aid reproducibility.

---

## 6Y9NP1qhoM

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces MISINFOTASK, a novel dataset for evaluating misinformation injection attacks in Multi-Agent Systems (MAS), and proposes ARGUS, a training-free defense framework that combines graph-based adaptive localization with goal-aware reasoning for misinformation rectification. The work demonstrates that standard MAS are vulnerable to covert misinformation and that ARGUS significantly mitigates this threat across various models and attack vectors.

## Strengths
- The paper addresses a timely and important problem—covert misinformation in MAS—with two concrete, well-motivated contributions: a new dataset designed for complex, realistic tasks and an innovative defense framework that integrates spatial (graph-topological) and temporal (reasoning-driven) strategies.
- The experimental evaluation is comprehensive, covering multiple LLMs (GPT-4o, DeepSeek-V3, Gemini), three distinct attack methods (Prompt Injection, RAG Poisoning, Tool Injection), five MAS topologies, and includes insightful ablation studies and temporal propagation analysis, strongly supporting the claims.

## Weaknesses
- The core mechanism for inferring the attacker’s misleading goal during rectification is insufficiently described. While Section 4.2 and Algorithm 1 mention goal inference, the exact process (e.g., how the corrective agent derives and records the goal from a message) is vague, making it difficult to assess and reproduce this key component of ARGUS.
- The choice of baseline defenses is not fully justified. G-Safeguard is a trained method requiring pre-collected logs, creating an asymmetric comparison with the training-free, online ARGUS. A comparison with other online, inference-time defenses (e.g., consensus-based debate) would better situate the contribution.
- The reliance on an LLM judge for the primary metrics (MT and TSR) is not validated, and the paper does not report statistical significance for the results (based on only three trials per data point). This undermines the reliability of the quantitative improvements claimed.
- The paper does not test ARGUS on existing MAS benchmarks or against adaptive, multi-round attackers, leaving its generalization and robustness under more realistic, evolving attack scenarios unclear.

## Nice-to-Haves
- A more detailed sensitivity analysis of hyperparameters (e.g., the weights α, β, γ in edge scoring, the number of monitored edges k) would provide insight into the robustness of the design choices.
- A deeper qualitative analysis of failure cases and the impact of different MAS topologies on defense performance would help identify the boundaries of the approach.
- Including a more thorough cost-benefit analysis (e.g., latency per round, scaling of token usage with agent count) would aid in assessing practical deployment trade-offs.

## Novel Insights
The paper’s key insight is that misinformation in MAS can be effectively countered by dynamically monitoring communication channels most relevant to the inferred intent of the attack and using the LLM’s own reasoning capabilities for persuasive correction. This combines spatial (graph-topological) and temporal (reasoning-driven) strategies in a unified, training-free framework, offering a novel approach distinct from prior consensus-based or adversarial training defenses.

## Suggestions
- Clarify the goal inference process in Section 4.2, possibly with an example or by expanding the prompt description for the corrective agent, and ensure Algorithm 1 aligns with the main text descriptions (e.g., define θ and P_goal).
- Add a comparison with at least one additional online, inference-time defense baseline (e.g., a consensus-based method from related work) to strengthen the experimental comparison.
- Report statistical significance (e.g., confidence intervals or p-values) for the main results and consider validating the LLM judge scores with a small human evaluation or multiple LLM judges to bolster metric reliability.

---

## UtFQNwWBaA

- GT: Reject (avg 4.0)
- Predicted: N/A (4.7/10)
- Match: N/A

### Final Review

## Summary
HiT-JEPA proposes a hierarchical self-supervised learning framework for trajectory representation. It builds a three-level JEPA (Joint Embedding Predictive Architecture) that abstracts point-level, segment-level, and trajectory-level semantics, with a novel mechanism to propagate attention weights from higher to lower levels to integrate multi-scale information. The method is evaluated on six real-world trajectory datasets, demonstrating strong performance in similarity search, robustness to noise, and superior zero-shot generalization across domains.

## Strengths
- **Novel and well-motivated hierarchical architecture.** The three-level design explicitly addresses the multi-scale nature of trajectory data, unifying fine-grained details and global semantics within a single self-supervised framework—a clear advance over prior single-scale trajectory representation methods.
- **Strong empirical performance, especially in zero-shot generalization.** Experiments on six diverse datasets (urban GPS, check-ins, vessel tracks) show that HiT-JEPA consistently achieves the best or highly competitive mean ranks in similarity search, with particularly impressive zero-shot transfer results across heterogeneous domains (e.g., from dense taxi data to sparse check-in or maritime trajectories).
- **Interpretable design and thorough evaluation.** The paper provides qualitative visualizations of attention maps and decoded trajectories, showing that the learned hierarchical attention aligns with semantic waypoints (origin, stops, destination). The evaluation is comprehensive, including robustness tests (downsampling, distortion) and downstream fine-tuning to approximate classical similarity measures.

## Weaknesses
- **Insufficient justification for key methodological choices.** The use of bilinear interpolation to upsample attention matrices between hierarchy levels is presented without justification; attention weights are not spatially continuous like images, and the suitability of this operation is not discussed or ablated against alternatives (e.g., nearest-neighbor upsampling or cross‑attention). Similarly, the choice of exactly three levels is not ablated; the contribution of the hierarchy depth remains unverified.
- **Lack of statistical rigor and missing core baselines.** Results are reported as single point estimates without variance measures, confidence intervals, or statistical tests. For ICLR, this undermines claims of superiority, especially where margins are small (e.g., fine‑tuning on Porto). Moreover, while the paper compares against recent deep learning baselines, it omits direct comparison with classical trajectory similarity measures (e.g., DTW, Frechet, Hausdorff) in the similarity search experiments, making it difficult to gauge whether the learned representations actually outperform standard metrics.
- **Limited analysis of what each hierarchical level captures.** While visualizations suggest different granularities, there is no quantitative analysis demonstrating that each level encodes distinct semantic properties (e.g., local curvature vs. global route shape). This leaves the core multi‑scale claim only partially validated.

## Nice-to-Haves
- An ablation study varying the number of hierarchical levels (e.g., 2, 4, or more) to understand the sensitivity and optimal depth of the architecture.
- Inclusion of classical similarity measures (DTW, Frechet, etc.) as baselines in the self‑similarity experiments to better contextualize the performance of learned representations.
- A deeper error analysis of zero‑shot failure cases to clarify the limitations of cross‑domain transfer.

## Novel Insights
The paper’s key novel insight is that propagating attention weights—rather than just embeddings—from coarser to finer levels within a hierarchical JEPA enables the model to focus lower‑level feature extraction on semantically important trajectory segments (e.g., origins, destinations, turns) while maintaining consistency with the global context. This attention‑based “top‑down spotlight” mechanism, validated through interpretable visualizations, provides a principled way to integrate multi‑scale trajectory semantics beyond simple feature concatenation or independent level‑wise training.

## Suggestions
- Justify the choice of bilinear interpolation for attention upsampling with an ablation comparing alternative upsampling schemes (e.g., nearest‑neighbor, learned deconvolution) and discuss why attention matrices can be treated as spatially continuous in this context.
- Report standard deviations or confidence intervals for key results (e.g., mean ranks, fine‑tuning metrics) across multiple runs or via bootstrapping to substantiate claims of statistical significance.
- Add a quantitative analysis correlating features from each hierarchy level with handcrafted trajectory properties (e.g., local speed changes, overall direction) to objectively demonstrate the distinct semantic roles of different levels.

---

## qbDnX2YC6F

- GT: Reject (avg 4.5)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper presents the first systematic study of interaction effects between representation learning (RL) and postprocessing (PP) methods in Open-Set Recognition (OSR). It introduces a modular two-stage framework, discovers a failure mode termed "magnitude collapse" where certain RL methods degrade at scale, and demonstrates that a simple baseline (AddON) combined with magnitude-aware PP can achieve strong performance. A key finding is that small-scale evaluations with auxiliary data are not predictive of large-scale performance.

## Strengths
- **Novel and well-motivated investigation**: The study formally addresses the unexplored modularity and interaction effects between RL and PP components in OSR, a clear gap in the literature. The decomposition of performance into RL and PP contributions (ΔRL, ΔPP) provides a simple but effective analytical tool.
- **Strong empirical foundation and actionable insights**: The experiments are comprehensive, spanning dataset scales (CIFAR+N, ImageNet P1-P3), multiple backbones (ResNet, Swin), and a variety of RL/PP methods. The discovery of "magnitude collapse" is compellingly supported by feature magnitude distributions and class-wise regression analysis. The paper delivers practical guidelines (e.g., use AddON with MA PP, avoid MM methods with similar auxiliary data) grounded in evidence.

## Weaknesses
- **State-of-the-art claim requires broader contextualization**: The paper claims AddON+PostMax achieves "state-of-the-art performance," but the comparison is primarily against the selected set of methods in Table 1. A more direct comparison on established large-scale benchmarks (e.g., Semantic Shift Benchmark) against a wider range of contemporary OSR methods would better substantiate this claim and the paper's practical impact.
- **Theoretical mechanism of magnitude collapse is primarily empirical**: While the feature magnitude analysis and the derivation for AddON's incentive are valuable, the explanation for why magnitude collapse occurs in MM methods (OE, OS) remains largely correlational. A more formal analysis linking the training objective, data similarity, and gradient dynamics would strengthen the theoretical contribution.

## Nice-to-Haves
- A controlled ablation systematically varying semantic similarity between known and auxiliary classes (beyond the fixed P1-P3 protocols) would strengthen the causal claim that high similarity induces collapse.
- Visualizations of feature directions (e.g., t-SNE/UMAP plots) alongside magnitude histograms could provide a more complete view of the representation space.
- A brief discussion on the applicability of findings to other modalities or to the closely related OOD detection task would clarify the scope of generalizability.

## Novel Insights
The paper provides a genuinely novel synthesis: it demonstrates that RL and PP components are largely independent when no auxiliary data is used, enabling additive modular gains, but that significant interaction effects emerge when auxiliary data is introduced. Crucially, it identifies "magnitude collapse" as the mechanism behind the degradation of magnitude-manipulating RL methods at scale, linking it to high semantic similarity between known and auxiliary classes—a scenario not captured by small-scale benchmarks. This insight invalidates the common practice of extrapolating from small-scale evaluations and provides a clear, evidence-based warning to the community.

## Suggestions
- To strengthen the SOTA claim, consider adding a comparison on a standard large-scale benchmark (e.g., SSB) against a broader set of recent OSR methods, even if briefly in the appendix or as a limitation.
- In the discussion or limitations section, explicitly acknowledge that the large-scale ImageNet results are from single runs (common due to cost) and that the reported trends, while clear and consistent across protocols and backbones, lack variance estimates.

---

## zwfpyw345l

- GT: Reject (avg 0.5)
- Predicted: N/A (1.5/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a hierarchical attention model for learning state representations of source code in reinforcement learning (RL) tasks. The model integrates sequential (transformer) and structural (graph attention) mechanisms across token, function, and module levels, and is optimized end-to-end with an RL objective. It is evaluated on three code-related RL tasks—code completion, program repair, and algorithmic problem solving—showing improved performance over several baselines.

## Strengths
- **Well-motivated hierarchical design:** The multi-level attention mechanism aligns with the natural hierarchical structure of code, addressing a clear limitation of flat or purely graph-based representations.
- **Comprehensive experimental framework:** The evaluation spans three distinct and challenging code RL tasks using established datasets, demonstrating the generality of the approach.
- **Rigorous ablation study:** An ablation study clearly shows that each component (token, function, and module-level attention) contributes positively to overall performance.

## Weaknesses
- **Critical lack of methodological clarity and reproducibility:** Key equations are garbled (e.g., Equations 1, 2, 7) and architectural details—such as how the transformer and graph attention pathways are combined, the definition of the final state representation, and the content of Figure 1—are insufficiently described. This makes the method impossible to reproduce and assess properly.
- **Insufficient experimental details for replication:** The paper does not specify the Markov Decision Process formulation (state/action spaces, reward functions, termination conditions) for each task, nor how baseline models are adapted to the RL setting. This hinders reproducibility and fair comparison.
- **Outdated and incomplete baseline comparisons:** The baselines do not include recent state-of-the-art code representation models (e.g., CodeT5, GraphCodeBERT) or direct hierarchical alternatives (e.g., SG-Trans), limiting the claim of advancement over current best methods.
- **Incomplete statistical reporting and analysis:** The main results table lacks measures of variance (standard deviations, confidence intervals) and statistical significance details, despite a claim of significance. Figures referenced (e.g., learning curves, scalability) are not described with information about multiple runs or error bars, which is standard for RL experiments.
- **Superficial analysis of RL-specific benefits:** While results show improved task performance, the paper does not convincingly demonstrate that the learned representations provide specific RL advantages (e.g., better exploration, credit assignment, or generalization) beyond simply being better code embeddings. The discussion of policy entropy and training dynamics is mentioned but not substantiated with data or interpretation.
- **Missing limitations section:** Section 7.1 is essentially empty, failing to acknowledge important limitations such as computational complexity, reliance on specific code representations (AST, CDG), or potential overfitting to the chosen tasks.

## Nice-to-Haves
- Include more recent hierarchical code models (e.g., SG-Trans) as direct baselines to better contextualize the hierarchical attention contribution.
- Evaluate the learned representations on probing tasks (e.g., code understanding benchmarks) to disentangle representation quality from the RL policy's learning capability.
- Provide more quantitative analysis of what each attention level captures (e.g., attention distribution statistics across syntactic vs. semantic features) to validate the hierarchical design.
- Analyze the learned dynamic edge features to understand what semantic dependencies they encode.
- Show visualizations (e.g., attention heatmaps over concrete code examples, t-SNE plots of the state space) to improve interpretability.

## Novel Insights
The core novel insight is the integration of hierarchical, multi-granularity attention—combining sequential and graph-based mechanisms—specifically for end-to-end optimization of RL state representations in code domains. While hierarchical code models exist for tasks like summarization, their joint optimization with an RL policy to form task-adaptive state embeddings is a distinct contribution. However, the reviews do not surface additional novel insights beyond what the paper explicitly claims.

## Suggestions
- **Revise Section 4 (Method) thoroughly:** Provide clear, correct equations and a succinct algorithmic description (e.g., pseudocode) of the forward pass, explicitly defining how the transformer and graph attention outputs are combined at each level and how the final state representation is constructed.
- **Expand Section 5 (Experimental Setup) with essential details:** Explicitly define the MDP for each task (state/action spaces, rewards, termination) and describe exactly how each baseline model is integrated into the RL loop (e.g., how a Tree-LSTM produces a state embedding for the policy network).
- **Update baseline comparisons and strengthen analysis:** Add comparisons to recent state-of-the-art code models and perform an analysis that links the learned representations to RL-specific benefits (e.g., measure the smoothness of the value function over the embedding space or conduct policy transfer experiments).
- **Improve statistical reporting:** Include standard deviations or confidence intervals in Table 1 and ensure all figures clearly indicate variability (e.g., error bars across multiple runs).
- **Complete the limitations section:** Discuss meaningful limitations such as computational cost, dependence on specific code representations, and potential generalization challenges.

---

## FtL9eEmU6v

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
EditBench introduces a benchmark for evaluating LLM capabilities in instructed code editing, built from real-world data collected via a VS Code extension. It features diverse user instructions, code contexts with highlighted sections and cursor positions, and multiple natural and programming languages. Evaluations on 40 models show the task is challenging and performance depends on contextual information.

## Strengths
- **Real-world grounding**: The benchmark is constructed from in-the-wild data collected from nearly 500 developers using a custom extension, capturing authentic user instructions and code contexts, as evidenced by comparison with synthetic benchmarks (Table 1) and greater library diversity (Figure 3).
- **Comprehensive evaluation and analysis**: The paper evaluates 40 diverse LLMs, revealing that only one model exceeds 60% pass@1, and provides valuable insights such as the impact of highlighted code on performance (Table 3) and variation across edit categories (Figure 5).

## Weaknesses
- **Curation bias**: The aggressive filtering from 2672 initial responses to 109 core problems may skew the benchmark towards difficult, well-defined tasks, potentially underrepresenting the full distribution of "messy" real-world edits. While filtering is necessary for quality, it could compromise the claim of capturing authentic user behavior.
- **Scalability and reliability of test creation**: The manual process for creating test harnesses by annotators is not scalable and lacks reported inter-annotator agreement, raising concerns about consistency and potential bias in ground truth definitions. This limits the benchmark's ability to grow and remain objective.
- **Surface-level failure analysis**: While performance variations are reported, the paper does not deeply diagnose why models fail (e.g., misunderstanding intent, API errors, handling ambiguity), limiting the benchmark's diagnostic value for guiding model improvements.
- **Unexplained context effects**: The mixed impact of cursor position on performance (Table 3) is noted but not analyzed, leaving an open question about how models utilize different contextual cues, which is central to the benchmark's design.

## Nice-to-Haves
- Expanding programming language coverage beyond Python and JavaScript to include more ecosystems like Java or Go.
- Conducting a human performance baseline to calibrate the difficulty of EditBench problems.
- Quantifying instruction ambiguity and correlating it with model performance to validate the claim that real-world instructions are challenging.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Report inter-annotator agreement for the test harness creation process to establish benchmark reliability.
- Perform a qualitative error analysis categorizing common failure modes across models and problem categories.
- Expand the context ablation study to include more models to generalize findings about the importance of highlighted code and cursor position.

---

## k60jAxVSv7

- GT: Reject (avg 3.3)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces DenseFace, a post-training method for mitigating demographic bias in face recognition. DenseFace models face embeddings with von Mises-Fisher distributions, estimates local embedding densities via a balanced anchor set, and performs density-aware probabilistic matching. Experiments on RFW and RB-WebFace show consistent bias reduction across multiple pre-trained models without compromising verification accuracy.

## Strengths
- The method is model-agnostic and requires no retraining of the base face recognition model, making it practical for deployment on existing systems.
- Extensive experiments demonstrate significant bias reduction (measured by a NIST-style FPR metric) across multiple state-of-the-art models (AdaFace, CosFace) with different architectures, training datasets, and loss functions, while preserving or slightly improving verification accuracy.
- The paper provides a thoughtful critique of existing bias metrics (e.g., accuracy standard deviation) and adopts a more appropriate evaluation protocol (FPR at a fixed threshold), contributing to better evaluation practices.

## Weaknesses
- The method relies on a curated, demographically balanced anchor set for density estimation. The paper does not analyze the sensitivity of results to the anchor set's size, balance, or quality, which is critical for real-world application where such a set may be unavailable.
- There is no direct comparison with existing post-training bias mitigation methods (e.g., score normalization by Terhörst et al., Linghu et al.), making it difficult to assess the relative improvement over the state of the art.
- Key hyperparameters—the margin \(m\) for handling near-orthogonal embeddings and the number of neighbors \(K\)—are introduced without ablation or justification, leaving their impact on performance and bias reduction unclear.
- The learning-based variant, while promising for efficiency, is not thoroughly validated: the paper does not explicitly confirm that the density regressor was trained and evaluated on disjoint identity sets, raising concerns about overfitting and generalization.

## Nice-to-Haves
- Extending the evaluation to identification benchmarks (e.g., IJB-C) and other sensitive attributes (e.g., age, pose) would demonstrate broader applicability.
- A more detailed theoretical explanation for the observed correlation between inter-class embedding density and demographic bias would strengthen the motivation.
- Visualizing the embedding space before and after adjustment (e.g., with t-SNE/UMAP) could provide intuitive support for the method's effect on inter-class similarities.

## Novel Insights
The paper's key novel insight is the empirical observation that inter-class embedding density (estimated via von Mises-Fisher distributions) correlates with demographic bias, and that adjusting similarity scores based on this density can mitigate bias without retraining. This insight leads to a practical post-hoc method that preserves accuracy, which is a significant advance over prior work that often trades accuracy for fairness.

## Suggestions
- Perform an ablation study on the anchor set (size, balance) and hyperparameters (\(m\), \(K\)) to guide practitioners and clarify the method's sensitivity.
- Compare DenseFace directly with recent post-training bias mitigation methods on the same NIST-style FPR metric to better situate its contribution.
- Clarify the experimental setup for the learning-based variant: ensure and report that the density regressor was trained and tested on disjoint identity sets to rule out data leakage and confirm generalization.

---

