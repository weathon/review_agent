# Direct-Scoring Baseline Results

Model: z-ai/glm-5

## avSocG0oFA

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

# Review of "DARE the Extreme: Revisiting Delta-Parameter Pruning for Fine-Tuned Models"

## Assessment

This paper addresses an important practical problem in delta-parameter pruning (DPP) for fine-tuned language models. The authors identify that the popular DARE method fails at high pruning rates and with large delta parameters, and they propose two main solutions: DAREx-q (rescaling factor modification) and DAREx-L2 (regularization during fine-tuning with AdamR).

**Strengths:**

1. **Clear theoretical motivation**: Theorem 3.1 provides useful bounds on output changes, identifying two failure factors for DARE: excessive rescaling factors and high mean/variance in delta parameters. This theoretical grounding helps explain when DARE struggles.

2. **Comprehensive empirical evaluation**: The paper tests extensively across encoder models (BERT, RoBERTa) and decoder models (Llama-7B variants, Qwen), covering multiple datasets (GLUE tasks, GSM8K). The consistent improvements (>40% absolute on some tasks at 99% pruning) demonstrate practical value.

3. **Multiple practical variants**: The authors provide four DAREx-q variants for different scenarios (labeled/unlabeled data, global/per-layer rescaling), along with guidance on when to use importance-based vs. random-based DPP.

4. **Orthogonal to PEFT**: Demonstrating compatibility with LoRA and structural pruning broadens applicability.

**Weaknesses:**

1. **Limited conceptual novelty**: The core insight—that 1/(1-p) becomes suboptimal at high p and can be tuned—is relatively straightforward. The paper doesn't provide a closed-form solution for optimal q; it requires empirical tuning.

2. **DAREx-L2 requires retraining**: AdamR-L2 modifies fine-tuning, limiting use for the many already fine-tuned models available (a major DPP use case). This should be positioned more clearly as a complementary rather than primary solution.

3. **Missing baselines**: At extreme compression rates (99%), comparison to alternatives like quantization, distillation, or other compression methods would strengthen the contribution. DAREx-q might be achieving similar quality with similar computational overhead as these alternatives.

4. **Limited ablations on q selection**: While per-layer tuning is proposed, benefits over global tuning are modest, and the computational overhead for finding q isn't analyzed.

5. **Validation data requirement**: DAREx-q variants that outperform require either validation data or unlabeled data optimization, adding complexity compared to vanilla DARE's simplicity.

## Score: 7.2

The paper makes solid contributions to an active area of model compression. The theoretical analysis is sound, experiments are thorough, and practical improvements are demonstrated. However, the core idea is incremental rather than breakthrough—the paper extends DARE rather than introducing fundamentally new insights. The requirement for empirical q-tuning and the limited applicability of AdamR-L2 prevent a higher score. The work should interest practitioners working with multiple fine-tuned models and merits publication.

---

## kbjJ9ZOakb

- GT: Accept (Oral) (avg 8.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces a method for learning and aligning continuous invariance manifolds of visual neurons, addressing an important gap in computational neuroscience: systematically comparing invariance properties across neural populations despite differences in receptive field properties like position, size, and orientation.

**Strengths:** The paper presents a well-motivated two-stage approach combining implicit neural representations (INRs) with contrastive learning for template learning, followed by affine transformation learning for alignment. The methodology is innovative, particularly the use of an asymmetric activation matrix to capture functional relationships and the model-agnostic design. Validation on simulated neurons demonstrates accurate recovery of ground truth neuron types, and application to macaque V1 data yields interpretable clusters including canonical simple/complex cells and potentially novel cell types. The cross-model generalization analysis (Figure 3E) provides evidence that results aren't artifacts of a specific model architecture. The comparison with prior methods (Klindt et al., Ustyuzhaninov et al.) clearly shows advantages in handling full affine transformations.

**Weaknesses:** Several limitations temper the contribution. First, the computational complexity scales quadratically with neuron count, which the authors acknowledge but don't substantially address. Second, the method is limited to 1D invariance manifolds, yet higher-dimensional invariances may exist. Third, the biological validation of identified clusters remains speculative—the authors explicitly note they "cannot fully exclude the possibility that these clusters result from recording artifacts." No electrophysiological validation is provided. Fourth, the template learning component heavily builds on Baroni et al. (2023), so novelty is primarily in the alignment step. Finally, while hyperparameters are detailed, systematic sensitivity analysis is limited.

Overall, this is a solid computational neuroscience contribution with clear methodological innovation and thorough empirical validation on simulated data. The biological application produces interpretable results but lacks independent validation. The paper fits well at an ML venue given its use of deep learning techniques and potential applicability beyond neuroscience.

Score: 7.0

---

## ZYd5wJSaMs

- GT: Accept (Poster) (avg 6.4)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces Diff-2-in-1, a framework that integrates diffusion models for both multi-modal data generation and dense visual perception tasks. The key idea is a self-improving learning mechanism where one network (creation parameters) generates synthetic training data while another (exploitation parameters) learns from both real and synthetic samples, with EMA feedback between them.

**Strengths:**
The paper makes a reasonable contribution by proposing a unified framework that preserves both generative and discriminative capabilities of diffusion models—unlike recent work like Marigold that repurposes diffusion models solely for perception. The self-improving mechanism with EMA is well-motivated, drawing from mean teacher approaches in semi-supervised learning. Empirical results demonstrate consistent improvements across multiple tasks (surface normal estimation, semantic segmentation, multi-task learning) and backbones. The partial noise generation strategy (inspired by SDEdit) to create in-distribution samples is sensible. The ablations are thorough, including analysis of timesteps, EMA parameters, and comparison with alternatives like direct finetuning.

**Weaknesses:**
The novelty is somewhat incremental—the individual components (SDEdit-style generation, Mean Teacher EMA, diffusion feature extraction) are all established techniques. While integration is novel, the method's complexity (warm-up stage, two parameter sets, captioning model dependency) may limit practical adoption. The performance gains, while consistent, are modest (e.g., +5.2 points on 11.25° metric for surface normal). The frozen backbone setting limits mutual benefits between generation and perception; the unfrozen experiments are relegated to the appendix. The generation quality evaluation (FID ~40) is not competitive with modern generative models. Additionally, relying on BLIP-2 for text prompts adds external dependency.

**Score: 7.0**

---

## YaeZwhXJ4k

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents BinaryDM, the first weight binarization method specifically designed for diffusion models, addressing the critical challenge of deploying compute-intensive diffusion models on edge devices. The approach introduces two key technical contributions: (1) Evolvable-Basis Binarizer (EBB), which uses multi-basis binarization during early training and evolves to single-basis for efficient inference, and (2) Low-rank Representation Mimicking (LRM), a knowledge distillation technique operating in low-rank space to stabilize optimization.

**Strengths:**
The paper tackles a timely and important problem with strong novelty—no prior work has successfully applied weight binarization to diffusion models. The EBB technique is clever: by using learnable dual-basis during training and regularizing toward single-basis, the method achieves better representation capacity without inference overhead. The LRM component is well-motivated, addressing optimization ambiguity through PCA-based dimensionality reduction. Experimental evaluation is comprehensive across multiple datasets (CIFAR-10, LSUN, FFHQ, ImageNet), metrics (FID, IS, sFID, Precision/Recall), and activation bit-widths. Results show impressive improvements—e.g., FID reduced from 10.87 to 7.74 on LSUN-Bedrooms (W1A4), even outperforming W4A4 EfficientDM. The efficiency analysis includes actual hardware measurements, adding practical credibility.

**Weaknesses:**
Several limitations merit attention. First, the method only binarizes weights, leaving activations at 4-bit minimum—true W1A1 binarization remains unaddressed. Second, the gap from full-precision models remains substantial (FID 3.09→7.74 on LSUN-Bedrooms), limiting practical deployment appeal. Third, multiple hyperparameters (τ, λ, K, transition timing) require careful tuning, and the heuristic selection of EBB application locations lacks adaptive justification. Fourth, the training overhead from QAT with full-precision teacher models is non-trivial, though this is acknowledged. Finally, while comparisons with quantization methods are fair, direct comparison against more general BNN techniques (ReActNet, INSTA-BNN) shows limited analysis of why these fail for diffusion models specifically.

The paper makes a meaningful contribution to efficient diffusion models, with solid technical innovation and thorough validation. However, the significant performance gap from full-precision and partial binarization (weights only) prevent it from being a stronger contribution.

Score: 7.0

---

## iEUZMISIKj

- GT: Reject (avg 4.8)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes SwitchLoRA, a parameter-efficient training method for pre-training large language models that extends LoRA by dynamically switching vectors in the low-rank adapter matrices. The core innovation is maintaining a pool of candidate vectors and periodically replacing portions of the LoRA matrices, enabling accumulation of full-rank information in the frozen weight matrix while maintaining low-rank training's efficiency benefits.

**Strengths:**
The paper addresses an important problem—LoRA's poor performance during pre-training—and proposes a well-motivated solution. The switching mechanism is clever: by incrementally updating different subsets of candidate vectors, the method can approach full-rank training behavior while keeping communication costs low. The theoretical analysis in the appendix provides reasonable justification for the design choices, particularly the initialization scheme and the handling of optimizer states. The empirical results are generally positive, showing perplexity improvements over both full-rank training (15.23 → 15.01 on LLaMA 1.3B) and competitive baselines (ReLoRA, GaLore). The comparison with GLUE fine-tuning demonstrates downstream task transferability.

**Weaknesses:**
Several limitations temper the contribution. First, the model scales tested (130M-350M) are relatively small for claims about "large language models," with only the 1.3B experiment approaching moderate scale. Second, achieving competitive performance requires relatively high LoRA ranks (256-512), meaning the trainable parameter reduction is less dramatic than suggested (Table 4 shows 45-55% reduction, not the "50-60%" range). Third, the paper shows significant hyperparameter sensitivity in ablations (interval_0, ratio, frozen steps N), which creates practical deployment challenges. Fourth, while SwitchLoRA "surpasses" full-rank training on perplexity, the GLUE results show mixed performance (e.g., CoLA regresses from 48.60 to 47.43) with high variance (±15 for some tasks), raising questions about statistical significance. Finally, the communication overhead reduction claim (54%) lacks detailed distributed training experiments to validate.

**Overall Quality:**
The paper makes a clear contribution to the parameter-efficient training literature with a novel mechanism and reasonable experimental validation. The weaknesses are notable but not fatal—this is solid work that advances the field incrementally rather than breakthrough.

Score: 7.5

---

## 6EkWIfvjj9

- GT: Reject (avg 5.2)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces RARe, a method that adapts retrieval models to leverage in-context examples (query-document pairs) during inference. The key observation is that while in-context learning works well for decoder-only LLMs, simply prepending in-context examples to queries hurts performance in existing embedding models. RARe addresses this by fine-tuning models with in-context examples retrieved via BM25 from semantically similar training queries.

**Strengths:** The paper tackles an interesting and underexplored question—whether ICL techniques from generative LLMs can benefit embedding models. The observation that naive prepending fails (Figure 2) is valuable and motivates the approach well. The experimental evaluation is thorough, covering multiple architectures (LLM2Vec, E5-Mistral, Llama variants), benchmarks (BeIR, RAR-b), and providing detailed ablations on example selection, quantity, and format. The analysis on latency tradeoffs and the relationship between example-query similarity and performance gains adds practical insight.

**Weaknesses:** The core idea has strong parallels to query expansion and pseudo-relevance feedback, well-established IR techniques—the paper doesn't sufficiently differentiate itself from this literature or compare against standard query expansion baselines. The practical limitations are significant: inference latency increases substantially (up to 40x on some datasets), and the method requires access to training data at inference time. Results are inconsistent across datasets and models (LLM2Vec shows minimal gains while E5-Mistral benefits more). The comparison with Promptriever is not entirely fair since Promptriever uses synthetic data. The paper lacks deeper mechanistic analysis explaining *why* the improvements occur beyond surface-level ablations.

**Overall:** This is a reasonably executed paper addressing a legitimate research gap, but the contribution is incremental. The technique is simple, practical constraints are non-trivial, and the conceptual novelty relative to existing query expansion methods is limited. The thorough experimentation saves it from being a reject, but weaknesses in novelty and practicality keep it from a clear accept.

Score: 5.5

---

## ajSmXqgS24

- GT: Accept (Poster) (avg 6.8)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents DexTrack, a method for learning a generalizable neural tracking controller for dexterous manipulation from human references. The approach combines reinforcement learning and imitation learning within an iterative data flywheel framework, enhanced by a homotopy optimization scheme for mining high-quality demonstrations.

**Strengths:**
The paper addresses an important and challenging problem—developing a general-purpose tracking controller for dexterous manipulation that can handle diverse objects and complex in-hand manipulations. The proposed homotopy optimization scheme is an interesting idea, drawing inspiration from chain-of-thought reasoning to solve challenging tracking problems through progressively simpler intermediates. The empirical evaluation is comprehensive, spanning two datasets (GRAB and TACO), including both simulation and real-world experiments on a LEAP hand with Franka arm. The >10% improvement in success rates over baselines is meaningful, and the qualitative examples demonstrate handling of challenging scenarios like thin objects and subtle in-hand re-orientations. The ablations appropriately validate the contribution of each component.

**Weaknesses:**
The individual technical components are not particularly novel—combining RL with imitation learning is well-established, curriculum-style optimization has prior art, and the data flywheel concept is borrowed from other domains. The training pipeline is complex and computationally expensive (~4 days with 8 GPUs), raising questions about practical adoptability. The baselines are somewhat limited; comparisons to adapted versions of methods not originally designed for tracking (DGrasp) and re-implemented baselines are weaker than comparison to dedicated whole-body tracking methods like PHC adapted for hands. The homotopy path generator shows limited generalization (Table 9 shows only 28% effectiveness on out-of-distribution tasks), undermining some practical value. Real-world success rates notably lag simulation, and the success criteria definitions are somewhat lenient (Level 1 success only requires "exhibiting potential to lift"). The reward function contains many hand-tuned weights, limiting transferability.

**Overall:**
This is a solid contribution with clear methodology and strong empirical validation. While the technical novelty is somewhat incremental and the training complexity is a practical concern, the work advances dexterous manipulation tracking in a meaningful way. The real-world demonstration and thorough ablations strengthen the paper.

Score: 7.0

---

## Z7FLmWFUFo

- GT: Reject (avg 3.8)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces Litee, a lightweight exploration method for deep reinforcement learning that repurposes the value network's embeddings to drive exploration using linear multi-armed bandit techniques (UCB or Thompson Sampling). The approach requires no additional parameters, reduces computational complexity from O(n³) to O(d³), and provides theoretical sub-linear regret bounds. The authors also propose Litee+, which adds a small auxiliary network for sparse reward settings with minimal parameter overhead.

The paper has several notable strengths. First, the core idea is elegant and practical—leveraging existing learned embeddings for exploration rather than training separate networks is both efficient and principled. The connection to neural bandit literature (Neural-LinUCB/TS) is well-motivated and theoretically grounded. Second, the empirical results are solid: Litee+ outperforms E3B on MiniHack tasks while using far fewer parameters (0.8% increase vs. 65%), and Litee consistently improves SAC, PPO, and TD3 on MuJoCo. Third, the simplicity of implementation (~10 lines of code) makes the method highly accessible. Finally, providing theoretical regret bounds—rare for practical deep RL exploration methods—is a meaningful contribution toward bridging theory and practice.

However, the paper has notable weaknesses. The empirical evaluation is relatively narrow: MiniHack and MuJoCo only, with no Atari experiments which are standard for exploration papers. While the authors mention other baselines (ICM, RND, RIDE, NovelD), detailed comparisons are absent, leaving readers to rely on prior literature. Second, the theoretical analysis relies on strong NTK assumptions that may not hold in practice, and the practical Algorithm 2 differs significantly from the theoretical Algorithm 3. Third, the Litee+ extension, while empirically useful, breaks the theoretical guarantees claimed for the base method. Finally, some key implementation details are relegated to the appendix, making reproduction less straightforward than the "10 lines" claim suggests.

Score: 7.0

---

## JyQYYjtO88

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a systematic study of quantum algorithms for nonconvex optimization with noisy oracles, specifically addressing the problem of finding $\epsilon$-approximate second-order stationary points. The authors characterize query complexity across different noise regimes and provide both upper bounds (algorithms) and lower bounds.

**Strengths:**

1. **Novel and Systematic Contribution:** This is the first comprehensive treatment of quantum algorithms for nonconvex optimization under noisy oracle access. The systematic characterization across tiny, small, intermediate, and large noise regimes is valuable and well-motivated by practical considerations (empirical risk minimization, near-term quantum devices).

2. **Strong Theoretical Results:** The paper establishes exponential quantum speedups in certain noise regimes (e.g., $\tilde{O}(\log d)$ vs $\Omega(d/\log d)$) and polynomial speedups in others. The matching of upper and lower bounds in some regimes strengthens confidence in the results.

3. **Clear Organization:** Tables 1-3 effectively summarize contributions and facilitate comparison with classical bounds. The paper clearly delineates which regimes yield quantum advantages.

4. **Complete Characterization:** The paper identifies four regimes: poly-logarithmic quantum queries possible, polynomial quantum queries possible, exponential queries required, and information-theoretically unsolvable—providing a comprehensive landscape.

**Weaknesses:**

1. **Missing Technical Details:** The main technical contributions are stated informally with proofs relegated to supplementary materials. Key theorem statements lack formal precision in the main text, making verification difficult.

2. **Gap Between Bounds:** Several gaps remain between upper and lower bounds (e.g., $O(\log d/\epsilon^{1.75})$ vs $\Omega(\epsilon^{-12/7})$), and noise thresholds don't perfectly align across different results.

3. **Restricted Practical Applicability:** The exponential speedup regimes require very restrictive noise thresholds ($\nu = O(\epsilon^{10}/d^5)$), which may be difficult to achieve in practice. The paper acknowledges this but doesn't fully discuss practical implications.

4. **Limited Algorithmic Novelty:** The proposed algorithms combine existing techniques (Jordan's gradient estimation, perturbed gradient descent, quantum mean estimation). While novel in combination, the core components are established.

5. **No Quantum Advantage in Some Regimes:** In Table 2's last row, classical and quantum achieve identical $O(\log^4 d/\epsilon^2)$ complexity, showing no advantage.

**Overall Assessment:**
This is a solid theoretical contribution that fills an important gap in understanding quantum optimization robustness. While the systematic characterization and exponential speedup results are valuable, the missing technical details and remaining gaps prevent a stronger recommendation.

Score: 7.0

---

## pcnq7fZs4t

- GT: Reject (avg 3.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes a Common Feature Learning approach for Zero-shot Image Recognition (CF-ZIR) that aims to learn fine-grained visual-semantic relationships at the image level rather than class level. The method uses a two-stage dictionary learning framework: visual-attribute embedding followed by visual-semantic embedding.

**Strengths:**
The paper identifies a reasonable limitation in existing ZSL methods: most approaches establish coarse-grained class-level relationships between visual and semantic spaces, ignoring within-class image variations. The proposed approach of using attributes to guide extraction of common visual features and generating image-level semantic vectors is conceptually sensible. The method achieves competitive results on three standard benchmarks (aPY, AwA1, AwA2), showing improvements over several embedding-based baselines. The ablation studies demonstrate the contribution of the discrimination loss and visual-semantic alignment components.

**Weaknesses:**
The paper has several significant issues. First, the experimental evaluation is incomplete—many comparison cells in Table 2 are marked with "-" and the paper only evaluates the standard ZSL setting without considering the more challenging and widely-used Generalized ZSL (GZSL) setting. The baseline comparisons are outdated; many methods are from 2018-2021, and no comparison with recent CLIP-based or transformer-based ZSL methods is provided despite mentioning them in related work. Second, the method has limited novelty—dictionary learning for ZSL has been explored before (CDL, HCDDL), and the "dual-layer embedding" contribution appears incremental. Third, the paper lacks important experimental rigor: no hyperparameter sensitivity analysis despite having 5+ hyperparameters, no statistical significance tests, and no confidence intervals. Fourth, there are writing quality issues including a typo in the paper title ("IM## AGE") and inconsistent notation in the optimization objectives. Finally, the empirical gains are modest (1-2% improvement on most datasets) with incomplete baselines, making it difficult to assess true state-of-the-art performance.

**Overall Quality:**
The paper addresses a valid problem but offers an incremental solution with incomplete experimental validation. The missing GZSL evaluation, outdated baselines, and lack of comparison with recent methods significantly weaken the contribution claims. For a top venue like ICLR, the paper does not meet the standards for novelty and experimental completeness.

Score: 4.0

---

## XFpb3T5Zc9

- GT: Reject (avg 5.7)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper introduces iFedDR, an inexact federated Douglas-Rachford splitting algorithm that automatically adjusts the number of local client computation steps through a relative error condition, rather than requiring pre-specified local iterations. The key innovation is enabling on-demand refinement when approximate proximal evaluations are insufficient, eliminating the need to tune the number of local steps.

**Strengths:**
The paper makes several noteworthy contributions. First, the relative error condition approach is clever and addresses a genuine practical pain point in federated learning—the difficulty of choosing the right number of local steps a priori. The theoretical analysis is rigorous, extending beyond minimization to handle minimax problems and constrained games via the monotone inclusion framework. The derivation of iFedDR from the more general iPPPA framework with semidefinite preconditioning is technically sound and provides useful theoretical machinery. The error condition can be computed efficiently on the server without additional memory allocation. Experiments on logistic regression, linear probing, and fair classification demonstrate competitiveness with tuned baselines.

**Weaknesses:**
Several issues limit the paper's impact. The core idea of relative error conditions for proximal methods dates back to Solodov & Svaiter (1999), and while extending it to federated learning with semidefinite preconditioning is non-trivial, it's not revolutionary. Practical concerns exist: the method requires clients to send both x̄ᵢₖ and ∇fᵢ(x̄ᵢₖ), doubling communication compared to FedAvg. Refinement triggers additional communication rounds, and while a heuristic is proposed to mitigate this, the total communication cost including refinements isn't analyzed. No comparison with ProxSkip or other acceleration methods is provided. The analysis is limited to convex settings, restricting applicability to neural network training. The experiments, while reasonable, lack large-scale deep learning validation.

**Overall:**
This is a solid contribution addressing a real problem with adequate theoretical grounding and empirical validation. However, the approach builds substantially on existing techniques, has practical overhead concerns, and lacks comparison with recent accelerated methods. The minimax extension and automatic tuning benefit are valuable but don't constitute a major breakthrough.

Score: 5.5

---

## a3g2l4yEys

- GT: Accept (Poster) (avg 6.8)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**

This paper makes a substantial contribution to an increasingly important problem in multimodal AI: the English and Western-centric bias in MLLMs. The work is comprehensive in scope, introducing both a 6M-sample training dataset (PANGEAINS) covering 39 languages and a holistic evaluation suite (PANGEABENCH) with 14 datasets spanning 47 languages. The methodology for constructing culturally diverse instructions—filtering LAION-Multi images by cultural relevance using LLM scoring, then generating multilingual instructions—is thoughtful and novel. The empirical results demonstrate meaningful improvements over existing open-source models (+10.9% on multilingual tasks), and the ablation studies on English data proportion and training sample scaling provide actionable insights for future work. The complete release of data, model, and benchmarks represents a significant resource for the community.

**Weaknesses:**

Several limitations temper the contribution. First, the core multilingual training data relies heavily on machine translation (Gemini 1.5 Pro), which inevitably introduces translation artifacts and potential cultural mismatches—a limitation the paper acknowledges but doesn't quantify empirically. Second, the model architecture itself is standard (LLaVA-Next with Qwen2-7B), so the novelty lies primarily in data curation rather than methodological innovation. Third, while PANGEA narrows the gap with proprietary models, significant performance gaps remain, particularly in complex reasoning tasks. Fourth, some evaluation components (xMMMU) are also machine-translated, which may confound benchmark results. Finally, the paper could benefit from deeper analysis of failure modes, especially for low-resource languages where performance remains challenging.

**Overall Quality:**

This is a well-executed project that addresses a genuinely important problem with practical impact. The comprehensive evaluation across multiple task types and languages, combined with systematic ablations and complete open-sourcing, positions this as a valuable contribution to multilingual multimodal AI. While not revolutionary in methodology, the scale and thoroughness of the work justify acceptance for a top venue.

Score: 7.5

---

## dMj3SDNxn4

- GT: Reject (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces UNICORNN, a novel approach for deep ordinal regression that guarantees unimodal output probabilities while maintaining good calibration properties. The work addresses three important limitations of existing methods: (1) lack of guaranteed unimodality, (2) inadequate handling of ordinal structure via maximum likelihood, and (3) poor calibration from optimal transport losses.

The proposed method uses a truncated normal distribution with fixed bins to guarantee unimodality, learns location (μ) via optimal transport loss, and then calibrates the scale (σ) via Brier score minimization while preserving accuracy. The theoretical guarantees (Lemmas 1 and 2) are well-motivated and correctly proved. Empirically, the method shows consistent improvements across six benchmark datasets, outperforming recent baselines on MAE and calibration metrics while guaranteeing 100% unimodality.

**Strengths:** The paper clearly identifies an important trade-off between OT loss and calibration—a useful insight. The architectural guarantee of unimodality is elegant and avoids the pitfalls of soft-label approaches. The two-phase training procedure is intuitive, and the accuracy-preservation lemma provides theoretical justification. Experiments are thorough with multiple datasets and metrics.

**Weaknesses:** The core idea—using a truncated normal with fixed bins—is relatively straightforward. The comparison to CORN (a strong recent baseline) is relegated to the appendix and covers only two datasets. The paper lacks deeper analysis: why not jointly optimize μ and σ with a combined loss? How does this compare to simpler post-hoc calibration (e.g., temperature scaling) applied to OT-trained models? The calibration analysis relies solely on ECE, which has known limitations. Finally, no discussion addresses when the unimodality assumption might fail (e.g., bimodal label distributions).

Overall, this is a solid, well-executed contribution with clear theoretical grounding and consistent empirical improvements, but the novelty is incremental rather than transformative.

Score: 7.0

---

## pPyJyeLriR

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper presents ScaleGUN, a novel approach for scaling certified graph unlearning to billion-edge graphs by integrating approximate graph propagation techniques. The work addresses a fundamental tension in the field: certified unlearning requires exact embeddings for theoretical guarantees, while scalable methods necessarily introduce approximation errors.

**Strengths:**

1. **Identifies an important and timely problem.** As privacy regulations become stricter, efficient and provably-private graph unlearning becomes critical. The authors correctly identify that existing certified methods don't scale, which limits practical adoption.

2. **Novel theoretical contribution.** The key insight—that approximation errors can be bounded and incorporated into the certified unlearning framework—is non-trivial. The paper develops a lazy local propagation framework for Generalized PageRank propagation and proves that both worst-case and data-dependent bounds remain valid despite approximation errors. The theoretical derivations are rigorous and complete.

3. **Strong empirical results.** The experiments demonstrate remarkable efficiency gains: ScaleGUN achieves certified unlearning on ogbn-papers100M (1.6B edges) in 20 seconds versus 1.91 hours for retraining. The comparisons against CGU and CEU show clear advantages on large graphs.

4. **Comprehensive evaluation across unlearning scenarios.** The paper addresses node feature, edge, and node unlearning with appropriate theoretical treatment for each. The unlearning efficacy evaluations (adversarial edge removal, DDRT, MIA) provide solid validation.

5. **Extends beyond the specific method.** The lazy local propagation framework is general and could be combined with other dynamic propagation techniques.

**Weaknesses:**

1. **Limited to linear models for certified guarantees.** The method only provides certified unlearning for linear models (SGC-style). For deep models, it becomes heuristic without guarantees. While the authors acknowledge this limitation, it constrains practical applicability since many modern GNNs use non-linearities.

2. **Conservative theoretical bounds.** The worst-case bounds contain many constants and may be loose in practice. While data-dependent bounds help, the gap between theoretical bounds and actual gradient norms could be analyzed more deeply.

3. **Limited comparison with non-certified unlearning methods.** The paper compares against CGU and CEU but doesn't discuss how the efficiency-efficacy tradeoff compares to other graph unlearning methods (GIF, GNNDelete, GraphEraser) that prioritize empirical effectiveness over certification.

4. **The propagation scheme is fixed.** The method relies on a specific Generalized PageRank propagation scheme; adapting to other message-passing architectures may require additional work.

5. **Batch unlearning analysis could be stronger.** While sequential unlearning is well-analyzed, the theoretical treatment of batch operations receives less attention despite being important for practical deployment.

**Overall:**

This is a strong contribution to privacy-preserving graph learning. The technical novelty of bridging approximate propagation and certified guarantees is substantial, the theoretical framework is sound, and the empirical demonstration on billion-edge graphs is compelling. The limitations around linear models are honest and inherent to certified unlearning as a research direction. The work meaningfully advances the state of certified graph unlearning.

**Score: 7.5**

---

## 8WtBrv2k2b

- GT: Reject (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses dynamic scheduling for quantum resource state construction using reinforcement learning with a Transformer architecture. The authors formulate the problem as scheduling entanglement links between inhomogeneous qubits with varying fidelities and success rates, developing a "Transformer-on-QuPairs" model that processes qubit pair sequences via self-attention.

**Strengths:** The problem is timely and relevant—optimizing cluster state construction for emerging quantum platforms (color centers, quantum dots) with configurable connectivity is important for scaling quantum systems. The physical modeling via Monte Carlo simulation of the Barrett-Kok protocol is detailed and grounded in quantum optics principles. The experimental methodology is reasonably comprehensive, comparing against multiple baselines (random, MST, greedy) and alternative architectures. The scalability experiments with transfer learning from 40 to 160 qubits show practical consideration for deployment.

**Weaknesses:** The claimed "more than 3× improvement" is misleading—the actual improvement in μ (log quantum volume) is from 13.90 to 15.58, approximately 12%. The "3×" refers to 2^(Δμ), an unconventional interpretation of "improvement" that overstates results. Statistical significance is questionable given overlapping error bars (Greedy: 13.90±0.62 vs. Transformer: 15.58±0.84). The Transformer application is straightforward without architectural innovation—self-attention on qubit pairs is a natural encoding but not novel. The homogeneous case (σ(F)=0) shows all methods performing similarly (~μ=4.6), suggesting the method only helps under specific inhomogeneity conditions. No comparison to graph neural networks or other architectures more natural for this graph-structured problem. The appendix contains excessive basic quantum mechanics background irrelevant to the core contribution.

**Overall:** This is a competent application paper with a relevant problem and reasonable execution, but limited ML novelty and overstated claims. The contribution is incremental rather than field-advancing.

Score: 5.5

---

## zvYJ1qG1Fy

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes ParamReL, a framework for learning latent representations in the parameter space of Bayesian Flow Networks (BFNs), rather than in observation space. The key innovation is a "self-encoder" that progressively encodes intermediate parameters θ_t into low-dimensional latent semantics z_t, combined with a conditional decoder for generation and a reverse-sampling procedure for reconstruction tasks.

**Strengths:**
The paper addresses a legitimate gap in the BFN literature—BFNs can handle mixed-type data but cannot learn low-dimensional latent representations. The parameter-space perspective is genuinely novel, as most representation learning happens in observation space. The theoretical formulation using variational inference is sound, with proper ELBO derivation and mutual information regularization. The reverse-sampling procedure enables reconstruction and interpolation, capabilities missing from standard BFNs. Experiments cover multiple datasets and tasks, with competitive results on disentanglement metrics (DCI) and classification (AUROC).

**Weaknesses:**
The core contribution is incremental—adding an encoder to extract latents has been done extensively for VAEs and diffusion models (DiffAE, InfoDiffusion). The novelty is primarily in applying this idea to BFNs and using parameters rather than observations. The key claim of handling "mixed-type data" is not strongly demonstrated; experiments evaluate discrete and continuous data separately, without showing a compelling mixed-type scenario. Baselines for discrete representation learning are incomplete (missing VQ-VAE variants). Ablation studies are limited—how critical is parameter-space encoding vs. observation-space? The computational overhead compared to baselines is not discussed. The claim that z_t captures "progressive semantics" across timesteps is visually suggested but not quantitatively validated. The self-encoder architecture borrows heavily from existing diffusion feature extraction techniques, raising questions about architectural novelty.

**Overall:**
While the parameter-space perspective is interesting and the work fills a gap in BFNs, the contribution is incremental. The advantages over existing representation learning methods are not dramatic, and the mixed-type data advantage claimed in the abstract is undersold in experiments. The execution is competent but not exceptional.

Score: 5.5

---

## rwqShzb9li

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper investigates whether LLMs possess linear representations of political ideology (specifically the liberal-conservative axis in American politics). The authors probe attention heads across three open-source LLMs, training linear probes to predict DW-NOMINATE scores of U.S. lawmakers, then demonstrate generalization to media slant prediction and successful steering of political perspectives.

**Strengths:** The paper addresses a well-defined question with rigorous methodology. The use of DW-NOMINATE (a validated political science measure) as ground truth is a strength. The authors test multiple models, perform appropriate robustness checks (non-linear probes, randomized labels, transformed labels), and demonstrate practical applications through token-level analysis and steering experiments. The cross-validation approach and comparison of linear vs. non-linear probes provides evidence for the linear representation hypothesis in this domain. The generalization to media slant prediction is a nice validation that probes capture something beyond memorized lawmaker associations.

**Weaknesses:** The study is limited to 7B parameter models, raising questions about scaling. The U.S.-centric focus limits generalizability—the appendix shows weaker cross-national performance (ρ=0.531 vs. 0.870). While the steering experiments work, effect sizes are moderate (best ρ≈0.6), and effectiveness varies by topic. Using GPT-4o for evaluation, despite human validation (ICC=0.91), introduces potential systematic bias. The approach follows prior work on linear representations closely; the main novelty is the domain application. Some alternative interpretations aren't fully ruled out—probes could partially encode name-frequency or demographic correlates rather than ideology per se.

Overall, this is a solid contribution to mechanistic interpretability, extending linear representation analysis to political ideology. The execution is clean, but scope is somewhat limited and findings incremental rather than field-advancing.

Score: 7.5

---

## f6GMwpxXHG

- GT: Reject (avg 2.2)
- Predicted: N/A (3.0/10)
- Match: N/A

### Review

# Assessment

This paper proposes "zephyr loss," a new loss function for GAN training defined as L(a) = α(√(a² + ε) - √ε), and introduces ZGAN built on this loss. While the theoretical analysis and experimental comparisons appear comprehensive at first glance, several serious issues undermine the paper's claims.

**Strengths:**
The paper provides theoretical properties including convexity and Lipschitz continuity proofs, convergence analysis for the optimal discriminator, and extensive experiments across multiple datasets (CIFAR-10, CIFAR-100, STL-10, SVHN). The comparison framework includes WGAN, LSGAN, and Diffusion-GAN baselines.

**Critical Weaknesses:**

**1. Implausible Experimental Results:** The Inception Scores reported are extremely problematic. The paper reports IS of 1.11 for WGAN and 2.1 for LSGAN on CIFAR-10—these values are essentially random noise level. State-of-the-art GANs achieve IS of 8-9+ on CIFAR-10, and even basic DCGAN should achieve ~6-7. This indicates either fundamentally broken implementations, incorrect evaluation methodology, or both. Compounding this concern, the FID scores (1.5-3.5) contradict the terrible IS values—good FID should correlate with good IS. This inconsistency suggests serious evaluation errors.

**2. Limited Novelty:** The proposed "zephyr loss" L(a) = α(√(a² + ε) - √ε) is essentially the well-known Charbonnier/Smooth L1 loss with minor reformulation. This loss has been extensively used in computer vision (optical flow, depth estimation) for decades. The connection to Huber loss is acknowledged superficially, but the paper fails to recognize that this formulation is not novel.

**3. Outdated Experimental Framework:** The use of DCGAN architecture (2015) as the backbone is outdated. Modern GAN papers use StyleGAN variants or ProjectedGAN architectures. Comparing to WGAN/LSGAN without including recent methods makes the evaluation incomplete.

**4. Theoretical Issues:** The mathematical analysis, while detailed, doesn't establish compelling theoretical advantages over existing losses. The convergence proof follows standard GAN analysis patterns without offering new insights into why this specific loss formulation should outperform alternatives.

Given the implausible experimental metrics that undermine the credibility of the empirical claims, combined with the limited novelty of the core contribution, this paper does not meet the acceptance threshold.

Score: 3.0

---

## ZDoN4W5s8d

- GT: Reject (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces a game-theoretic framework (SPECGAME) for modeling ML regulation as a principal-agent problem, capturing incomplete information and misaligned incentives between regulators and companies. The central insight—"Lossgate"—is that measurement uncertainty can either force conservative behavior (costing companies up to 8% utility) or enable strategic gaming (causing 70-96% higher social cost than collaboration).

**Strengths:**
The novel PAP formulation is genuinely valuable—previous trustworthy ML work assumed a single agent optimizing jointly, ignoring the multi-agent reality. The analysis correctly identifies hidden information and hidden action as key friction points. The PARETOPLAY algorithm with its SPCE recovery theorem provides technical grounding, and empirical evaluation across 6 datasets (vision and tabular) demonstrates practical relevance. The first-mover advantage finding—that regulators who specify constraints first achieve better privacy guarantees—offers actionable regulatory guidance. The paper successfully bridges game theory and ML accountability in an important but underexplored intersection.

**Weaknesses:**
The assumption that regulators and companies share the same underlying Pareto frontier (estimated on different datasets) is strong—real companies may have different architectures and training procedures. The experimental setup uses only FairPATE; generalization to other algorithms isn't established. The calibration step training models each round is computationally prohibitive in practice. No formal bounds on Price of Anarchy are provided. The connection to mechanism design is mentioned superficially, and the estimation of penalty scalars (C parameters) in Appendix B.2 has a somewhat circular quality. Real-world deployment questions—how regulators estimate λ parameters without company cooperation—remain open.

**Assessment:**
This is a solid contribution addressing an important societal problem with appropriate theoretical tools. While some assumptions are strong and practical applicability needs further development, the core insight about strategic gaming of regulatory uncertainty is valuable. The work opens a meaningful research direction at the intersection of game theory and trustworthy ML.

Score: 7.0

---

## oRfHv642qD

- GT: Reject (avg 4.4)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a theoretically-grounded contribution to NeuroAI, deriving an iterative Poisson VAE (iP-VAE) from ELBO maximization and showing it naturally implements Bayesian posterior inference through membrane potential dynamics in a spiking neural network. The work addresses genuine limitations of Gaussian-based predictive coding—negative rates and biologically implausible prediction signals—by adopting Poisson distributions, which better model neural spiking statistics.

The derivation is elegant: starting from the ELBO for Poisson-distributed latents and posteriors, the authors show that iterative inference on log-rates yields dynamics resembling membrane potential updates, with the linear decoder case reducing to a spiking version of the Locally Competitive Algorithm. The empirical results demonstrate meaningful improvements: iP-VAE closes the gap with sparse coding on natural images, achieves superior OOD generalization on character datasets, and uses significantly fewer parameters than amortized baselines.

However, several issues limit the impact. First, the experiments are conducted on relatively simple datasets (MNIST, Omniglot, 16×16 patches). Demonstrating scalability on CIFAR-10 or higher-resolution images would significantly strengthen the paper. Second, the training methodology lacks clarity—the paper trains with 4-64 iterations but tests with 1000, without adequately explaining how the model generalizes so far beyond its training regime or analyzing the computational cost. Third, the derivation in Appendix D makes approximations (straight-through estimator, removing proportionality constants) without sufficient justification for why these simplifications preserve the theoretical guarantees. Finally, while the neuroscience connections are compelling, they remain theoretical—empirical validation against neural data would be valuable.

The contribution is incremental relative to Vafaii et al. (2024)'s P-VAE, primarily extending from amortized to iterative inference. Nevertheless, the prescriptive theory angle—showing that ELBO maximization with Poisson assumptions yields spiking dynamics—is significant and fills an important gap in connecting variational methods to biological neural networks.

Score: 7.5

---

## 2orBSi7pvi

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces STDM, a modification to the forward diffusion process for time series data that incorporates spatio-temporal correlations through convolutional kernels and correlated noise. The core idea is intuitive and well-motivated: standard diffusion models add i.i.d. Gaussian noise that ignores the inherent temporal structure of time series data. By smoothing the signal before adding noise and introducing correlation structure in the noise itself, the method aims to preserve useful information during degradation.

The primary strength of this work is its clean mathematical formulation that maintains the Markov property while introducing temporal structure. The approach is modular and can theoretically be integrated into existing diffusion models with minimal modification. The experiments demonstrate consistent improvements on synthetic anomaly detection benchmarks (up to 31.8% relative improvement on the Trend dataset) and moderate improvements on most forecasting datasets.

However, several weaknesses limit the contribution. First, the novelty is incremental—structured noise and non-Gaussian degradation processes have been explored in prior work like Cold Diffusion (cited but not deeply discussed). Second, the evaluation is limited to only two baseline models (DiffusionAE and TimeGrad), omitting comparison with recent strong baselines like CSDI or TimeDiff. Most concerningly, the forecasting results on the Electricity dataset show significant degradation with STDM (CRPS increases from 0.0222 to 0.0453), which is not adequately explained or analyzed. Third, the method introduces hyperparameters (kernel size, scheduler parameters) without systematic sensitivity analysis. Finally, the theoretical grounding for why this approach should improve performance is limited beyond the intuitive motivation.

Score: 5.0

---

## GySIAKEwtZ

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

This paper provides a theoretical analysis of how long-tailed data distributions affect feature representations in supervised contrastive learning, and proposes FeatRecon, a method to rebalance skewed representations by generating synthetic features for tail classes.

**Strengths:**
The theoretical contribution in Theorem 2 is novel and meaningful—it formally characterizes how the optimal representation configuration changes under class imbalance, deriving the relationship between imbalance ratio and angles between class centers, and identifying a critical threshold for tail class collapse. This provides principled insight into why head classes dominate feature space, complementing prior empirical observations. The proposed FeatRecon method is well-motivated by this theory: generating synthetic features within confidence supports to balance sample sizes is a reasonable approach. The head class regularization for estimating tail class statistics is intuitive given limited samples. Experiments across four datasets demonstrate consistent improvements over strong baselines (BCL, TSC, etc.), with particularly notable gains on tail classes.

**Weaknesses:**
The theory is limited to the simplified "one-vs-all" case where one head class differs from K-1 equally-sized tail classes. Real-world long-tailed distributions have varying sizes across many classes, making the theoretical guidance incomplete. While the method is theoretically motivated, several components are heuristic: the quantile-based confidence support estimation, the regularization magnitude γ, the temperature adjustment formula (Eq. 11), and the iterative update scheme lack theoretical derivation. The ablation study (Table 5) is limited—only one setting is tested, and it doesn't isolate the importance of head class regularization or analyze sensitivity to hyperparameters. The paper also lacks analysis of whether generated synthetic features actually achieve the symmetric configuration predicted by theory, which would strengthen the connection between theory and method.

**Assessment:**
This is a solid paper with a genuine theoretical contribution that advances understanding of long-tailed representation learning. While the theoretical scope is limited and some methodological choices are heuristic, the work provides valuable insight and strong empirical results. The writing is clear and proofs are thorough.

Score: 7.2

---

## LNp7KW33Cg

- GT: Reject (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a Hierarchical Domain Adaptation (HDA) framework for stabilizing behavioral decoding in Brain-Computer Interfaces (BCI), addressing the critical challenge of non-stationary neural signals across recording sessions. The work combines three components: raw signal alignment via LSGAN, dynamical latent feature extraction using causal LSTM architectures, and semantic subspace alignment with disentanglement.

**Strengths:**
The paper addresses an important and practical problem in BCI research—maintaining decoding performance over time without frequent recalibration. The proposed hierarchical approach is well-motivated by the observation that direct alignment in latent spaces can lead to feature deviations. The integration of Lyapunov stability theory provides a principled framework for validating feature stability, which goes beyond typical empirical evaluations in domain adaptation papers. The experimental evaluation is comprehensive across three motor cortex datasets, demonstrating consistent improvements over strong baselines including NoMAD, Cycle-GAN, and ERDiff (6-10% R² improvement). The ablation studies effectively validate each component's contribution, and the analysis using Maximum Lyapunov Exponents offers interesting insights into the stability properties of the learned representations.

**Weaknesses:**
The methodological novelty is somewhat incremental—the approach combines existing techniques (GAN-based alignment, VAE disentanglement, adversarial domain adaptation) rather than introducing fundamentally new ideas. The theoretical analysis using Lyapunov theory, while interesting, serves primarily as a validation metric rather than providing deep theoretical insights into why the method works. The framework involves many hyperparameters (λy, λb, λo, latent dimensions, window size) and multiple adversarial training procedures, which raises concerns about practical tuning difficulty and reproducibility. While computational efficiency metrics are provided, real-time feasibility for actual BCI deployment is not thoroughly discussed. The experiments are limited to non-human primate motor cortex data; extending to human subjects and other recording modalities remains unaddressed. Some presentation issues (notation inconsistencies, formatting errors in the PDF) detract from clarity.

**Overall:**
This is a solid contribution addressing a meaningful problem with practical impact. The hierarchical adaptation strategy and stability analysis represent good integration of existing techniques for a challenging application domain. However, the incremental methodological contribution and hyperparameter complexity prevent it from being exceptional.

Score: 6.5

---

## 96jZFqM5E0

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents SiMHand, a contrastive learning framework for pre-training 3D hand pose estimators using mined similar hand pairs from large-scale in-the-wild videos. The key idea is to construct positive pairs from non-identical but similar hand images (rather than augmented versions of the same image), using 2D keypoint similarity for mining.

**Strengths:**
The paper addresses an important practical problem and the proposed approach is well-motivated. The core insight—that diverse positive pairs from different videos provide richer training signal than same-image augmentations—is intuitive and supported by results. The scale of pre-training data (2.0M images from Ego4D and 100DOH) substantially exceeds prior work and demonstrates scalability. Empirical results are convincing, with consistent improvements across three datasets: 15% on FreiHand, 10% on DexYCB, and 4% on AssemblyHands over PeCLR. The ablation studies are thorough, covering pre-training/fine-tuning data sizes, adaptive weighting effects, and Top-K analysis.

**Weaknesses:**
The core technical novelty is incremental. The concept of non-self-positives has been explored in prior work (cited by authors), and adapting it to hand poses with PCA-reduced 2D keypoint matching is straightforward. The adaptive weighting mechanism is a simple linear scaling scheme that provides modest gains (~2-3% when applied to other methods alone). The mining pipeline depends heavily on off-the-shelf pose estimators (MediaPipe), introducing noise and potential domain gaps that are addressed only superficially. The comparison with TempCLR is relegated to the appendix, and the weakly-supervised comparison (Table 6) is methodologically flawed—joint training would be more appropriate than comparing different paradigms. Finally, the paper lacks discussion of failure cases or the computational overhead of mining similar pairs across 2M images.

**Overall:**
This is a solid contribution with clear empirical benefits. The work pushes hand pose pre-training forward meaningfully but relies on relatively straightforward extensions of existing contrastive learning ideas. The large-scale data effort and thorough experimentation are commendable, though the methodological novelty is limited.

Score: 7.5

---

## owEQ0FTfVj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**
The paper introduces GLYCANML, the first comprehensive machine learning benchmark specifically designed for glycan understanding. This fills an important gap in biomolecular ML research, where benchmarks exist for proteins, nucleic acids, and small molecules, but not for glycans. The benchmark is thoughtfully designed with 11 diverse tasks covering taxonomy prediction (8 hierarchical levels), immunogenicity prediction, glycosylation type prediction, and protein-glycan interaction prediction. The authors provide two representation schemes (tokenized sequences and planar graphs) enabling evaluation of both sequence-based and graph-based models. The dataset splitting strategy is particularly well-motivated—using motif-based clustering to ensure test glycans contain novel structural motifs, which simulates real-world generalization scenarios. The inclusion of a multi-task learning testbed (GLYCANML-MTL) adds valuable dimensions for algorithm evaluation. The finding that heterogeneous GNNs (RGCN) consistently outperform other architectures provides actionable insights for future glycan modeling research. The comprehensive baseline evaluation across 18 models and the open-source release of code and data will facilitate community progress.

**Weaknesses:**
The paper has several limitations. First, there is limited methodological novelty—this is primarily an application paper applying existing architectures to a new domain without proposing novel models or training techniques. Second, baseline coverage could be improved: existing domain-specific glycan models (e.g., GlyNet, SweetOrigins mentioned in references) are not directly compared against. Third, some tasks have small datasets (1,320 for immunogenicity, 1,683 for glycosylation), raising concerns about statistical reliability. Fourth, the analysis of why heterogeneous GNNs outperform other methods and why small molecule encoders struggle could be deeper—the paper provides observations but limited mechanistic explanations. Fifth, the MTL evaluation shows mixed results (most MTL methods underperform single-task learning), but this finding isn't deeply investigated. Sixth, hyperparameter tuning procedures are not clearly described, making reproducibility concerns valid. Finally, the protein-glycan interaction task uses ESM-1b for protein encoding without justification or comparison to alternatives.

**Overall Quality:**
This is a solid benchmark paper that will serve as a valuable community resource. While the methodological contribution is limited (applying existing methods to new data), the task design, dataset curation, and comprehensive baseline evaluation represent meaningful contributions. The paper is well-written and organized, with clear scientific motivation. However, the limited analysis depth and missing comparisons to domain-specific baselines prevent a higher score.

Score: 7.5

---

## E0dTlxy1T4

- GT: Reject (avg 5.8)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents MMEvol, a multimodal instruction data evolution framework that iteratively improves the quality of image-text instruction data through three complementary evolution strategies: fine-grained perception evolution, cognitive reasoning evolution, and interactive evolution, followed by instruction elimination to filter failed evolutions.

**Strengths:**
The paper addresses a genuine and important bottleneck in MLLM development—the scarcity of high-quality, diverse instruction data. The three-pronged evolution approach is well-motivated: fine-grained perception evolution targets overlooked visual objects to improve robustness; cognitive reasoning evolution extends reasoning chains for complex tasks; and interactive evolution diversifies instruction formats. The empirical results are strong, achieving SOTA on 9 of 13 benchmarks while using substantially less training data than competitors like Cambrian-1. The ablation study (Table 1) clearly demonstrates each component's contribution, and the qualitative analysis of long-tail distribution improvements and verb-noun diversity provides useful insights into what the evolution process accomplishes.

**Weaknesses:**
The primary concern is the heavy reliance on OpenAI GPT-4o/GPT-4o-mini APIs for the evolution process, which raises reproducibility concerns and introduces dependency on proprietary systems. While the authors mention plans to use Qwen2-VL in future work, this remains a current limitation. The experiments are limited to 8B-scale models; validation on larger models would strengthen claims about scalability. The core idea—iterative data evolution—borrows heavily from text-domain Evol-Instruct, making the contribution somewhat incremental, though the adaptation to multimodal settings requires non-trivial design to maintain visual grounding. Additionally, while instruction elimination filters "failed evolutions," the paper provides limited analysis of whether subtle hallucinations might still slip through, and how the quality scoring mechanism correlates with actual downstream performance.

**Overall:**
This is a solid, well-executed contribution that demonstrates meaningful performance improvements through systematic data quality enhancement. While the approach is incremental and has some reproducibility limitations, the thorough experimental validation and practical utility justify acceptance.

Score: 7.0

---

## 4Kw4KAoVnx

- GT: Reject (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents Sparse-MeZO, a modification to the Memory-Efficient Zeroth-Order (MeZO) optimization method for fine-tuning large language models. The key insight is that ZO gradient estimation noise disproportionately harms optimization when applied to large weights versus small weights. Based on this observation, the authors propose selecting only small-magnitude weights for perturbation and updates via a dynamic sparse mask, enabling higher learning rates and better convergence.

**Strengths:**
- The paper identifies a non-obvious and practically important phenomenon: ZO gradient noise affects large weights more negatively than small weights. The empirical observation that continuing training from divergence with only small-weight updates can recover performance (Figure 2c) is compelling.
- The proposed method achieves substantial improvements over vanilla MeZO (9% accuracy gain on RTE, 3.5x convergence speedup) while maintaining the key benefit of inference-level memory consumption.
- The memory-efficient implementation that computes the sparse mask on-the-fly during the forward pass is a nice engineering contribution that preserves the memory footprint.
- Empirical evaluation is comprehensive across multiple model families (LLaMA-7b/30b, Mistral-7B, OPT-13b) and tasks from SuperGLUE, with appropriate baselines including random masking (R-MeZO), LoRA, and full fine-tuning.
- Theoretical convergence analysis is provided, showing dimension-dependent speedup from operating on a sub-network.

**Weaknesses:**
- The explanation for *why* small weights are more noise-resilient remains superficial. The paper shows that it works but doesn't deeply analyze the mechanism—is it related to loss landscape geometry, the role of different weights in pre-trained models, or something else?
- The sparsity hyperparameter requires per-task tuning (optimal values vary from 0.6-0.8 across tasks), and no adaptive mechanism is proposed. This limits practical applicability compared to methods with fewer hyperparameters.
- While R-MeZO (random mask) is included as a baseline, the comparison is underexplored. A more thorough analysis of why magnitude-based selection outperforms random selection would strengthen the paper.
- The performance gap with first-order methods (FT, LoRA) remains substantial (e.g., ~3-4% average accuracy gap on LLaMA-7b), limiting practical adoption where gradient computation is feasible.
- Some implementation details (e.g., the exact forward-pass mask computation procedure) could be clearer for reproducibility.

Overall, this is a solid contribution to the ZO optimization literature for LLM fine-tuning. The insight is novel, the method is practical, and results are convincing. However, the mechanism is not fully explained and hyperparameter tuning is required.

Score: 7.5

---

## TRWxFUzK9K

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes an innovative approach for solving video inverse problems using only pre-trained image diffusion models, avoiding the need to train computationally expensive video diffusion models. The key technical insight is treating the temporal dimension of video as a batch dimension and introducing batch-consistent sampling to synchronize stochastic noise across frames, combined with conjugate gradient optimization for spatio-temporal data consistency.

**Strengths:**
The paper addresses an important practical problem—video diffusion models are notoriously difficult and expensive to train. The proposed batch-consistent sampling strategy is elegant: by initializing all frames with identical noise and using synchronized re-noising, the method enforces temporal coherence without requiring video-specific priors. The experiments demonstrate strong quantitative improvements (e.g., PSNR gains of 10-15 dB over baselines on temporal degradations) with 10-50× speedup. The extension to blind restoration and the efficiency analysis (running on <11GB VRAM) adds practical value. The writing is clear, and the geometric illustration of sampling paths (Figure 2) effectively conveys the intuition.

**Weaknesses:**
The baseline comparisons are somewhat limited. While DPS and DiffusionMBIR are reasonable baselines, the paper lacks comparison with recent video-specific restoration methods in the main text (DiffIR2VR appears only in the appendix). The reliance on a 256×256 pixel-space diffusion model is a significant limitation; extending to latent diffusion models would strengthen the work. The hyperparameter choices (NFE, η) vary across tasks without systematic analysis of sensitivity. Additionally, while the method works well for the tested degradation types, its effectiveness for more severe temporal degradations (e.g., frame interpolation) remains unclear. The theoretical justification for why batch-consistent sampling should work could be strengthened.

**Overall:**
This is a solid contribution addressing an important gap in diffusion-based inverse problem solving. The method is practical, well-motivated, and demonstrates impressive results. While some aspects could be strengthened, the core idea and execution merit publication.

**Score: 7.5**

---

## FDnZFpHmU4

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper investigates LLM ensembling, proposing both an analysis of factors affecting ensemble performance and a novel method called UNITE (Union Top-k Ensembling) that improves efficiency by operating on the union of top-k tokens rather than full vocabulary alignment.

**Strengths:**
The paper makes a solid practical contribution to LLM ensembling. The systematic analysis of factors affecting ensemble performance—model performance gap, vocabulary size, and response style—provides valuable insights for practitioners. The finding that vocabulary size has marginal impact while performance gap and response style matter significantly challenges some assumptions in prior work. The UNITE method is well-motivated: by observing that next-token candidates typically reside within top-k probabilities, the authors avoid costly full vocabulary alignment. The efficiency gains are substantial (tokens manipulated reduced to <0.04%, latency only ~10ms overhead), making the method practically deployable. Experiments across six benchmarks with multiple model combinations demonstrate consistent improvements over existing methods like DEEPEN and GAC.

**Weaknesses:**
The technical novelty is incremental. The core insight that operating on top-k tokens suffices for next-token prediction is intuitive rather than groundbreaking. The model selection strategy relies on heuristic thresholds (10% performance gap, text length ratios) without theoretical justification. The tokenization alignment handling—taking the first subtoken's probability when a token doesn't exist in a model's vocabulary—seems ad-hoc and lacks rigorous justification. The response style analysis, while interesting, offers only a simple few-shot preprocessing solution. The paper misses comparison with recent routing-based approaches and provides no principled method for selecting the hyperparameter k beyond empirical tuning.

**Overall Quality:**
This is a competent, well-executed paper with practical value for the LLM ensembling community. While the contributions are more engineering-focused than conceptually novel, the systematic analysis and efficiency gains warrant publication. The work would benefit from deeper theoretical grounding and more sophisticated model selection criteria, but adequately addresses real deployment challenges.

Score: 7.0

---

## rAylWUIKtu

- GT: Reject (avg 4.2)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses a critically important problem in LLM evaluation: benchmark inflation due to data contamination. The authors introduce a rigorous methodology for constructing "retro-holdout" datasets—datasets created post-hoc but statistically validated to be indistinguishable from their corresponding public benchmarks. The core insight is clever: by comparing model performance on a public benchmark versus an unpublished but equivalent dataset, one can quantify the "inflation" caused by the benchmark's public availability.

The methodology is a significant strength. The four tests for "sufficient indistinguishability"—difficulty similarity using pre-release models, semantic embedding similarity via permutation tests, prediction accuracy via fine-tuned classifiers, and human distinguishability—provide multiple complementary validation mechanisms. The formal statistical framework in Appendix A adds theoretical rigor, and the iterative tools (attention visualization, embedding analysis) offer practical guidance for future practitioners. The release of Retro-Misconceptions as a usable dataset provides immediate practical value, and the evaluation of 20 models—including both API and open-release models—yields actionable findings: some models show inflation up to 16 percentage points.

However, the paper has notable limitations. The methodology is demonstrated on only one benchmark category (TruthfulQA's non-adversarial Misconceptions), raising questions about generalizability to other benchmark types. The exclusion of adversarial entries due to GPT-3 filtering bias is a meaningful constraint, as adversarial questions are often the most safety-relevant. The reliance on pre-release models for the difficulty test (Babbage-002, Davinci-002, NeoX-20b) is problematic—these models are quite dated, limiting statistical power. The human study sample size (n=23) is modest, and the resource-intensive nature of manual dataset creation without LLM assistance may limit adoption. Additionally, the concurrent GSM1k analysis in Appendix H, while interesting, feels underdeveloped compared to the main contribution.

Despite these limitations, this is a well-executed contribution on a timely topic. The methodological rigor, clear problem formulation, and practical tools distinguish it from the growing literature on benchmark contamination.

Score: 7.5

---

## tl63stKeSC

- GT: Reject (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces LearnQuad, a learnable quadrature framework for solving PDEs using neural networks. The core idea is novel: parameterize weight functions from the modified Gauss-Jacobi family using neural networks, which induce families of orthogonal polynomials whose roots serve as adaptive collocation points. The authors leverage recent asymptotic expansion results to efficiently compute quadrature nodes and weights in parallel.

**Strengths:** The paper presents a creative synthesis of classical quadrature theory with modern machine learning. The theoretical foundation is solid, building on orthogonal polynomials and recent advances in asymptotic expansions. The empirical evaluation is comprehensive, covering multiple PDEs (convection, Burgers', Allen-Cahn, wave, diffusion) and comparing against numerous baselines including PINN variants, RAR-G, RAD, and R3 sampling. The consistent improvements across most benchmarks demonstrate the method's effectiveness. The extension to solving families of PDEs via hyper-networks is a natural and useful application of the framework. The method's versatility across strong, weak, and energy formulations is also noteworthy.

**Weaknesses:** Several issues limit the impact. First, the computational overhead of the learnable quadrature module is not analyzed—while the asymptotic expansions enable parallel computation, the total training cost compared to simpler adaptive methods is unclear. Second, while performance improvements are shown, the paper lacks analysis of what the learned weight functions capture; visualization or interpretation would strengthen the contribution. Third, the hyperparameters (α, β) for the Gauss-Jacobi weight are largely unexplored beyond Table 5. Fourth, the 100D experiment shows no improvement over Monte Carlo, suggesting limited benefits for high-dimensional problems. Finally, some recent adaptive PINN methods are omitted from comparison, and wall-clock time comparisons are absent.

The paper makes a genuine contribution to physics-informed neural networks by introducing principled adaptive quadrature. However, the limited analysis of computational efficiency and lack of insight into what is actually being learned prevent this from being a strong accept. The method appears most beneficial for low-dimensional problems with non-smooth solutions, but this scope is narrower than claimed.

Score: 6.5

---

## UYZRaUCLAg

- GT: Reject (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces ADP-3D, a framework for solving inverse problems in protein structure determination using pretrained diffusion models as priors within a plug-and-play MAP estimation framework. The authors demonstrate applications to structure completion from partial structures, cryo-EM map refinement, and structure prediction from sparse pairwise distances.

**Strengths:**
The paper makes a meaningful contribution by successfully adapting diffusion-based inverse problem techniques from image processing to protein structure determination. The approach is principled—leveraging the plug-and-play framework with half-quadratic splitting—and demonstrates versatility across multiple measurement modalities. The use of existing pretrained models (Chroma, RFdiffusion) as priors is clever, avoiding the need for task-specific training while still achieving good results. The preconditioning strategy for handling ill-conditioned linear measurements shows technical depth, and the handling of nonlinear forward models (cryo-EM) with gradient-based optimization is appropriately implemented. Results show consistent improvements over posterior sampling baselines, and the framework's generality allows future improvements in diffusion models to be incorporated without retraining.

**Weaknesses:**
The experimental evaluation has notable limitations. The primary comparison is against Chroma's SubstructureConditioner, but there's no comparison to established structural biology tools like PHENIX or Rosetta that would demonstrate practical utility. The evaluation scope is narrow: only single-chain proteins are tested (a significant limitation acknowledged but not addressed), and the number of test structures is small. The structure completion experiments are explicitly described as a "toy problem," limiting their real-world impact. While cryo-EM refinement uses experimental maps, the evaluation depends heavily on ModelAngelo's output quality rather than independent ground truth. The paper lacks analysis of computational cost and hyperparameter sensitivity. Finally, the framework focuses on MAP estimation and thus doesn't provide uncertainty quantification, which could be valuable for experimental scientists.

**Overall Assessment:**
This is a solid paper with a clear contribution that bridges diffusion models and structural biology inverse problems. The methodology is sound, results demonstrate promise, and limitations are acknowledged. However, the limited experimental scope and missing comparisons to domain-specific baselines prevent a stronger recommendation. The work would benefit from more extensive validation on diverse structures and comparison to established methods.

Score: 7.0

---

## TwZBQKgwdW

- GT: Reject (avg 5.2)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes NTK-DFL, a decentralized federated learning method that adapts Neural Tangent Kernel-based training from centralized FL to the decentralized setting. The key innovation is combining NTK-based weight evolution with per-round model averaging among neighbors and final model aggregation.

**Strengths:**
The work represents a novel adaptation of NTK-based optimization to decentralized FL, which to my knowledge is indeed a first. The empirical evaluation is thorough across multiple datasets (Fashion-MNIST, FEMNIST, MNIST), heterogeneity levels, and network topologies. The results are compelling: 4.6× faster convergence in heterogeneous settings and maintained accuracy where baselines degrade significantly. The ablation studies—particularly on per-round averaging and the client selection algorithm—provide useful insights. The discussion of practical concerns like Jacobian batching for memory efficiency and reconstruction attacks for privacy adds value.

**Weaknesses:**
However, the paper has significant limitations. First, experiments use only a simple 2-layer MLP with 100 hidden neurons, which is far from the wide-network regime where NTK theory applies best. Testing only on simple image classification tasks with small models undermines claims of practical relevance. Second, while claiming faster convergence, Figure 16 reveals NTK-DFL uses 7.5× more bits per round than DFedAvg—total communication cost may not actually be lower. Third, there's no theoretical convergence analysis, leaving the claimed benefits without rigorous justification. Fourth, the comparison with DisPFL is unfair; it's a personalized FL method compared on global model performance. Fifth, key baselines like D-SCAFFOLD (decentralized SCAFFOLD) that specifically target heterogeneity are missing. Finally, the core NTK evolution mechanism is borrowed directly from prior work (Yue et al.), making the methodological contribution incremental—the novelty is primarily in combining existing techniques with decentralized averaging.

**Overall:**
This is a borderline submission. The adaptation to decentralized FL is interesting and results are promising within the limited experimental scope, but the simple model architectures, incomplete communication cost analysis, lack of theory, missing baselines, and incremental methodological contribution prevent a clear acceptance. The work needs evaluation on modern architectures and fairer communication cost accounting.

Score: 5.5

---

## ERce2rgMQC

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

This paper presents Controllable Safety Alignment (CoSA), a framework for adapting LLM safety behavior at inference time through natural language "safety configs." The work addresses an important gap in current safety alignment—the "one-size-fits-all" paradigm that ignores cultural and contextual diversity in safety requirements.

**Strengths:**
The problem motivation is excellent. The examples (game developers needing violence content, harassment training requiring discriminatory language discussion, cultural differences around alcohol) compellingly illustrate why static safety alignment is inadequate. The framework is comprehensive, providing a conceptual approach, training methodology (CoSAlign), evaluation protocol (CoSA-Score), and a human-authored benchmark (CoSApien). The empirical results are strong—CoSAlign significantly outperforms in-context alignment and cascade methods, even surpassing Cascade-Oracle (with human filtering) on CoSApien. Testing on both seen and unseen configs demonstrates genuine generalization rather than mere memorization. The CoSA-Score metric elegantly combines helpfulness and configured safety into a single interpretable measure.

**Weaknesses:**
The human evaluation is limited—a single annotator on 200 prompts lacks statistical rigor. The method is complex with multiple components (risk taxonomy, config synthesis, error-scoring mechanism) and arbitrary hyperparameters (α=0.1, β=3, γ=1) that lack thorough ablation. Heavy reliance on synthetic GPT-4o data for both training and evaluation raises concerns about circular biases. Only 8B models are tested, leaving scaling properties unclear. The GPT-4 experiments are incomplete (SFT only, no DPO). The error analysis reveals imperfect control, particularly for the "Other Harms" category and partial prompts. Finally, while prompt injection is mentioned, security analysis is limited.

**Overall:**
This is a well-executed contribution to pluralistic alignment with strong empirical results and meaningful benchmark contributions. However, the evaluation limitations, method complexity without sufficient ablation, and incomplete experiments on larger models prevent it from being a strong accept. The work advances an important direction but leaves room for improvement in control precision and evaluation robustness.

Score: 7.2

---

## UKZqSYB2ya

- GT: Reject (avg 2.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a two-stage framework for lung nodule segmentation combining Deformable-DETR for region proposal with a fine-tuned SAM for pixel-wise segmentation, evaluated on the LUNA16 dataset. The work addresses a clinically relevant problem with reasonable reported results (92.4% Dice coefficient).

**Strengths:** The paper addresses the important challenge of class imbalance in lung nodule detection, which is a real barrier to clinical deployment. The use of Maximum Intensity Projection (MIP) is well-motivated and clinically grounded, helping distinguish nodules from vessels. The two-stage paradigm—separating detection from segmentation—is sensible for this task, and the integration of Focal Loss for handling imbalance is appropriate. Results are competitive with prior work, and the discussion of limitations around small nodule detection is honest.

**Weaknesses:** The novelty is limited. Combining Deformable-DETR with SAM is straightforward architectural integration rather than a methodological innovation—both components are existing architectures applied in their standard configurations. More critically, the paper lacks ablation studies to justify design choices: Is the two-stage approach actually better than end-to-end segmentation? Does SAM provide benefits over a simpler U-Net decoder? What is the contribution of MIP versus training on individual slices? Without these analyses, it's unclear what drives the reported performance.

The comparison with prior methods (Table 2) is problematic because these baselines likely use different data splits, preprocessing, and evaluation protocols. Direct numerical comparison without controlled experiments is not rigorous. The paper also omits comparison with important modern baselines like nnU-Net. Additionally, there's an inconsistency between the abstract (93.2% precision) and results table (93.3% specificity)—these are different metrics. The test set having 5% nodule sparsity while training has 12.7% is an unusual design choice that warrants more justification.

Score: 5.5

---

## X4Rcxi9588

- GT: Reject (avg 5.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper addresses an important problem in multimodal learning: extending Large Multimodal Models (LMMs) to handle long videos without requiring expensive retraining on long video-text pairs. The authors propose two main contributions: (1) a visual context window extension method that adapts the YaRN technique from LLMs to visual tokens, and (2) a progressive pooling strategy to reduce memory consumption.

The paper has several strengths. First, the empirical results are impressive—the method outperforms GPT-4o on MLVU with only a 7B model and achieves consistent improvements across VideoMME and LongVideoBench. The training-free nature of the approach is practically valuable, avoiding data collection and computational costs. The progressive pooling mechanism is intuitive and achieves 45% memory reduction without performance degradation.

However, the paper has notable weaknesses. The core technical contribution—adapting YaRN's scaling factor to visual tokens—is relatively straightforward. The theoretical justification for separate "visual" and "language" context windows relies primarily on t-SNE visualizations showing distinct embedding clusters, but this observation (that different modalities have different embeddings) doesn't necessarily prove the existence of separate context windows in the claimed sense. The paper doesn't rigorously define how the "visual context window" size is determined for their baseline model. Additionally, the progressive pooling strategy, while effective, is a relatively simple heuristic without strong theoretical motivation beyond reducing redundancy in consecutive frames. The evaluation focuses primarily on LLaVA-OneVision, leaving questions about generalizability to other architectures. Comparisons against other context window extension methods (beyond YaRN) are absent.

Overall, this is a solid contribution with strong empirical results and practical utility, but the technical novelty is incremental. The method demonstrates clear value for practitioners working with long videos, though the theoretical grounding could be stronger.

Score: 7.0

---

## cXxfVkRCHJ

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes CFDG (Classifier-Free Diffusion Generation), a data augmentation method for offline-to-online reinforcement learning. The core idea is to use conditional diffusion models with classifier-free guidance to separately generate offline-like and online-like synthetic data, recognizing that these two data types serve different roles in O2O RL. The authors argue that offline data provides diversity to prevent suboptimal convergence, while online data improves stability and convergence speed.

The paper's strengths include a reasonable motivation supported by t-SNE visualization of data distributions, a clean technical approach that avoids training separate models for each data type, and empirical improvements over three baseline algorithms (IQL, PEX, APL) on D4RL locomotion and AntMaze tasks. The comparison with existing diffusion-based methods (SynthER, EDIS) is appropriate, and the ablation study shows that generating both offline and online data is beneficial.

However, the paper has several notable weaknesses. First, the technical novelty is incremental—the core contribution is applying classifier-free guidance (a well-established technique from image generation) to O2O RL with minimal adaptation. Second, key ablations are missing: the paper doesn't compare classifier-free guidance against alternative conditioning methods or against training two separate diffusion models. Third, the presentation has issues—Figure 1 appears twice with different captions, and some results show very high variance (e.g., hopper-me-v2 with IQL shows 66±33 vs 93±6), making statistical significance questionable. Fourth, computational overhead is not analyzed, and the hyperparameter sensitivity admitted in the limitations is not addressed experimentally. Finally, the AntMaze experiments for APL were conducted by the authors rather than using published baselines, making comparisons less reliable.

**Score: 5.0**

---

## 5x9kfRXhBd

- GT: Reject (avg 3.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes STGAT, a Spatial-Temporal Graph Attention Network with Hierarchical Transformer for forex rate prediction. The approach combines CNNs for local feature extraction, a transformer for global temporal dependencies, and a graph attention network to capture inter-currency relationships.

**Strengths:** The motivation is well-founded—forex markets exhibit strong interdependencies between currencies that standard time-series methods ignore. The proposed architecture reasonably addresses both spatial and temporal aspects of the problem. The paper includes ablation studies and hyperparameter sensitivity analysis, demonstrating that each component contributes to performance. Comparisons against 10 baselines across regression, transformer, and GNN categories provide a reasonable evaluation scope.

**Weaknesses:** Several significant issues undermine confidence in the results. First, the evaluation methodology is critically underspecified—the paper fails to describe the train/validation/test splits for time-series data, raising serious concerns about data leakage. Forex prediction requires strict temporal splits; without this, results may be invalid. Second, the reported R² values (0.957) are unrealistically high for financial prediction tasks, suggesting potential overfitting or methodological flaws. Third, the "hierarchical" transformer is not truly hierarchical—it appears to be a standard transformer with positional encoding, misrepresenting the contribution. Fourth, the graph construction via k-means clustering at each time step is poorly motivated and creates unstable graph structures. Fifth, Table 2's ablation results are contradictory—"Nonlinear GAT" alone achieves better RMSE/R² than the full STGAT model, which undermines the claimed benefit of the full architecture. Sixth, baselines like XGBRegressor and Lasso show negative R² values, suggesting improper implementation or tuning of comparison methods. Finally, the paper lacks directional accuracy metrics and trading-relevant evaluation, which are essential for forex prediction applications.

The core idea has merit, but the evaluation methodology gaps and unrealistic performance claims are concerning for a top ML venue.

**Score: 3.5**

---

## IjiIPQcLbV

- GT: Reject (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes a new metric called "round-trip score" for evaluating the synthesizability of molecules generated by drug design models. The metric uses a retrosynthetic planner to propose synthetic routes, then validates these routes using a forward reaction prediction model to check whether they actually reproduce the target molecule. The authors argue that existing metrics—SA score and search success rate—are inadequate for this purpose, and they provide a benchmark evaluating seven structure-based drug design (SBDD) models.

**Strengths:**
The paper addresses an important and practical problem in computational drug discovery: the gap between computationally predicted molecules and wet-lab synthesizability. The motivation is clear and well-articulated, with concrete examples illustrating the limitations of SA score and search success rate. The empirical validation demonstrates that the round-trip score achieves better precision (81.5%) than search success rate (71.6%) in distinguishing feasible from infeasible routes. The benchmark results across seven SBDD models provide useful insights, showing that even the best model (Pocket2Mol) only achieves ~20% success rate with the top-5 routes, highlighting significant room for improvement in the field.

**Weaknesses:**
The core methodology is not novel—using forward reaction models to validate retrosynthetic predictions has been explored in prior work (which the paper cites). The contribution is primarily in applying this idea to evaluate generative models rather than proposing a fundamentally new method. The empirical validation is limited: only 100 molecules are used to evaluate metric reliability, and the manual verification of "feasibility" using CAS SciFinder combined with "domain knowledge" introduces subjectivity. Critical hyperparameters like the 0.9 threshold for round-trip score and beam size of 5 lack sufficient justification or ablation. The metric's reliability fundamentally depends on the quality of both the retrosynthesis planner and forward model, a dependency the paper acknowledges but doesn't thoroughly analyze. Additionally, the computational cost of evaluation isn't discussed, which matters for practical adoption. The results show that 24 out of 68 feasible routes were incorrectly flagged as infeasible (FN=24), indicating substantial room for improvement in the metric itself.

**Overall:**
The paper addresses an important problem and provides a useful benchmark for the community. However, the technical contribution is incremental, the validation scope is modest, and several design choices lack proper justification. While the benchmark has value, the methodology doesn't advance the state-of-the-art significantly.

Score: 5.0

---

## bEvI30Hb2W

- GT: Reject (avg 3.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**
The paper addresses an important practical problem in long-form video understanding by proposing LVM-Net, a memory-based architecture that achieves impressive efficiency gains (18-75x speedup) through single-pass video processing with a fixed-size memory representation. The neural sampler mechanism using Gumbel-TopK relaxations is well-motivated, and the online continual learning loss to address sampling bias is a reasonable technical contribution. The ablation studies comparing neural sampling to random sampling and analyzing continual learning effects provide useful insights. The writing is generally clear, and the method is well-positioned against existing literature on token efficiency and long-context video processing.

**Weaknesses:**
The most significant concern is the substantial performance degradation compared to baselines. For activity queries, LVM-Net achieves only 32.4% vs TubeDETR's 45.3% on short queries—a ~29% relative drop. For object queries, medium-length performance drops to 11.9% from 25.4%. The paper repeatedly claims "competitive performance," but these gaps are significant and poorly justified. The evaluation is limited to a single dataset (ReST-ADL), and direct comparisons with recent memory-based approaches like MeMViT are missing despite being discussed in related work. The baseline "Modified TubeDETR" modifications are not clearly described. Important hyperparameter sensitivities (memory size vs. performance) are unexplored. Time query performance (8.6% vs 12.8% for long queries) suggests the memory may not preserve temporal information effectively, which is concerning for a video reasoning method.

**Overall Quality:**
While the efficiency contribution is valuable and the approach is technically sound, the significant accuracy trade-offs are under-discussed, and claims of "competitive performance" appear overstated given the metrics. The limited evaluation scope and missing baselines further weaken confidence in the method's general applicability.

Score: 4.5

---

## 8ZA7lrzw7O

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

**Assessment**

This paper provides a theoretical analysis of data echoing and proposes a communication-efficient variant for data parallelism. The work addresses the practical problem of data loading bottlenecks in large-scale training.

**Strengths:**
The paper makes a meaningful theoretical contribution by providing a sharper convergence analysis for stochastic data echoing. Unlike the prior work by Agarwal et al. (2020), which only showed linear speedup in the curvature term, this paper demonstrates linear speedup in the statistical term as well—a significant improvement. The analysis uses a clever "shifted state" technique to bound gradient estimation bias more tightly than previous Markov chain gradient descent analyses. The proposed communication-efficient variant for data parallelism is well-motivated and addresses a genuine limitation of vanilla data echoing. The cosine diminishing schedule for data loading probability is intuitive and empirically validated across image classification and language modeling tasks.

**Weaknesses:**
Several limitations reduce the paper's impact. First, the theoretical results require a burn-in period of O(p^{-4}) steps before speedup benefits appear—this is substantial and not thoroughly discussed in terms of practical implications. Second, the experiments are conducted on relatively small-scale problems (CIFAR, WikiText-2) with only 4 nodes in distributed settings, leaving questions about scalability. Third, there's no comparison to established communication-efficient methods like gradient compression, nor wall-clock time measurements to demonstrate practical speedup. Finally, while the bounded gradient dissimilarity assumption is weaker than bounded gradients, it remains restrictive for many real-world scenarios.

**Overall Quality:**
The paper provides solid theoretical contributions with clear improvements over prior work. The communication-efficient extension, while fairly straightforward as a combination with local SGD ideas, addresses a practical concern. However, the limited experimental scale and lack of practical efficiency demonstrations prevent this from being a strong accept. The paper sits at the borderline—good theoretical contribution but limited practical validation.

Score: 5.5

---

## q3EbOXb4y1

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Retri3D, the first framework for retrieving 3D neural graphics representations (NeRFs, 3D Gaussian Splatting) from databases using text queries. The authors make a compelling case that as 3DNGRs become more prevalent (Polycam, Luma AI, etc.), efficient retrieval systems are needed but currently absent.

**Strengths:**
- **Novel problem formulation**: Identifying the gap in NGR retrieval is timely and practical. The approach of rendering RGB images first and using VLMs for feature extraction is elegant in its generality—compatible with any NGR that can render images.
- **Innovative noise detection**: The insight that VLM activation features naturally separate clean regions from NGR artifacts/noise is interesting. Using a multivariate Gaussian to model noise features across scenes is a clever zero-shot approach that avoids needing clean training data.
- **Comprehensive evaluation**: Experiments span LERF (13 scenes) and ScanNet++ (280 scenes) with both NeRF and 3DGS models. Comparisons with LERF and LangSplat show substantial efficiency gains (orders of magnitude faster, dramatically less storage).
- **Practical SCMM module**: The smart camera movement strategy is intuitive and empirically effective for finding clean viewpoints, addressing the real problem of floaters in randomly sampled views.

**Weaknesses:**
- **Limited retrieval baselines**: The comparisons focus on semantic understanding methods (LERF, LangSplat) adapted for retrieval rather than established 3D retrieval techniques. A comparison with multi-view retrieval baselines would strengthen claims.
- **Heuristic noise modeling**: While the Gaussian approach works empirically, there's limited theoretical justification for why noise features cluster consistently across diverse scenes and NGR architectures.
- **Evaluation depth**: LERF contains only 13 scenes, limiting statistical conclusions. The object-label retrieval task is relatively straightforward—more complex compositional queries would better stress-test the system.
- **Self-training dependency**: The noise Gaussian requires random renderings to train, creating a chicken-egg problem for completely new scenes, though the paper shows cross-scene transfer helps.

Overall, this is a well-executed paper addressing an important emerging problem. The technical contributions are incremental but thoughtfully combined into a practical system. The efficiency improvements over semantic field methods are substantial, and the cross-NGR compatibility is a significant practical advantage.

Score: 7.5

---

## n2xueVy5ek

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper presents an empirical evaluation of large language models' capabilities to conduct cyberattacks on embedded/IoT devices. The authors perform 32 attacks across five commercial consumer devices (smart locks, vacuum cleaner, car adapter, and garage door), comparing manual attacks against LLM-assisted ones using the DREAD security assessment framework with an added LLM autonomy metric.

**Strengths:** The paper makes a valuable contribution by evaluating LLMs on *real-world* devices rather than synthetic CTF challenges, which better reflects actual adversarial capabilities. The DREAD-based evaluation framework provides structured comparison across attack types, and the findings have clear practical implications: LLMs improve reproducibility (40%), exploitability (35%), and discoverability (66%) but never increase actual damage. The detailed appendix with device-specific attack reports adds credibility and enables replication.

**Weaknesses:** Several methodological issues limit the contribution. First, the paper only tests GPT-3.5 and GPT-4, with inconsistent usage across attacks (attacks 1-12 used GPT-3.5, others GPT-4), making cross-attack comparisons problematic. No justification is given for model selection, and open-source alternatives are ignored. Second, the DREAD scoring appears subjective without inter-rater reliability measures or statistical significance testing. Third, the XSS game results may be contaminated by training data inclusion—a limitation acknowledged but not adequately addressed. Fourth, the title overclaims "conducting" attacks when most required substantial manual intervention (69% required assistance). Finally, the paper lacks deeper analysis of *why* LLMs succeed or fail on specific attack types, limiting insights for future work.

The real-world device testing is commendable, but the inconsistent methodology and lack of statistical rigor prevent stronger conclusions. For a top ML venue, these methodological gaps are significant concerns.

Score: 5.0

---

## eW4yh6HKz4

- GT: Accept (Spotlight) (avg 7.6)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes CBQ, a cross-block reconstruction-based post-training quantization method for large language models. The authors identify inter-layer and intra-layer dependencies as key challenges in low-bit quantization and propose three main components: Cross-Block Dependency (CBD) for maintaining long-range dependencies, LoRA-Rounding for efficient weight compensation, and Coarse-to-Fine Pre-processing (CFP) for outlier handling.

**Strengths:**
The paper provides a solid theoretical motivation using Taylor expansion analysis to show that second-order quantization error terms become more significant at low bit-widths. The empirical results are comprehensive, demonstrating consistent improvements over baselines across multiple bit-width settings (W4A16, W2A16, W4A8, W4A4) on both OPT and LLAMA models. The ablation studies are thorough, validating each component's contribution. The efficiency analysis shows CBQ is faster than OmniQuant while achieving better accuracy, with the 4.3-hour quantization time for LLAMA-65B being practically useful. The unified framework combining multiple techniques (CBD, LoRA-Rounding, CFP) is well-designed and addresses multiple quantization challenges simultaneously.

**Weaknesses:**
Several issues detract from the paper's overall contribution. First, there are notable missing baselines - AWQ (Lin et al. 2023) is a highly-cited weight-only quantization method not included in comparisons, and QuIP/QuIP# are recent methods for 2-bit quantization that should be compared against. Second, the technical contributions are incremental: CBD applies standard sliding window concepts, LoRA-Rounding directly adapts existing LoRA techniques, and CFP uses well-known statistical methods (quartile criterion). Third, the computational overhead is significant - Table 9 shows GPU memory increases from 17.2GB to 39GB for larger windows, which could limit practical deployment. Fourth, model coverage is limited to OPT and LLAMA1, with LLAMA2 relegated to the appendix and no modern architectures (Mistral, Gemma) evaluated. Finally, the title contains a typo ("LAN## GUAGE") suggesting insufficient proofreading.

The paper makes a reasonable contribution but the incremental nature of the techniques and missing baselines weaken the claim of superiority over state-of-the-art methods.

Score: 5.5

---

