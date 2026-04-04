# Direct-Scoring Baseline Results

Model: z-ai/glm-5

## YkfhTzq3hL

- GT: Reject (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces SHALLOW, a benchmark framework for evaluating ASR hallucinations across four dimensions: lexical fabrications, phonetic fabrications, morphological errors, and semantic errors. The work addresses an important limitation of WER, which treats all errors equally and fails to capture the varying severity of different error types—a particularly critical issue in domains like healthcare where transcription errors can have serious consequences.

**Strengths:** The paper makes a compelling case for going beyond aggregate WER metrics. The proposed taxonomy is well-motivated, grounded in linguistic distinctions between acoustic fidelity and language modeling. The empirical evaluation is thorough, covering 12 ASR architectures across 10 diverse datasets spanning standard speech, noisy environments, accented speech, and specialized domains. The synthetic benchmark validation demonstrates that each metric responds to its intended error category, and the medical case study effectively shows how SHALLOW catches critical errors (e.g., "cannot rotate" → "can rotate") that low WER scores would mask. The correlation analysis showing divergence between WER and SHALLOW metrics at higher WER levels is particularly valuable, providing empirical evidence that aggregate metrics become unreliable in challenging conditions.

**Weaknesses:** The weighting schemes (e.g., 0.5/0.3/0.2 for LF components, 0.25/0.75 for local/global semantic errors) are justified as "empirical observations" but lack systematic validation. The framework would benefit from human evaluation to validate that these metrics correlate with perceived error severity. While the individual metrics combine established techniques, this systematic integration is valuable but not fundamentally novel. The English-only focus and computational overhead from running multiple linguistic models (parser, grammar checker, embeddings, NLI) limit practical applicability. Finally, the paper diagnoses issues but provides limited guidance on how practitioners should use the four-dimensional profiles to improve models.

Overall, this is a solid contribution addressing a real gap in ASR evaluation. The empirical work is comprehensive and the framework is well-designed, though the arbitrary weighting choices and lack of human validation are notable limitations.

Score: 7.0

---

## 0aBAAS0rRT

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents SigMap, a multimodal foundation model for wireless localization that introduces cycle-adaptive masking and map-as-prompt mechanisms. The work addresses a practical problem in 5G/6G applications and makes several meaningful contributions.

**Strengths:**
The cycle-adaptive masking strategy is a thoughtful domain-specific adaptation that exploits the inherent periodicity of CSI data. Rather than blindly applying vision-based MAE techniques, the authors correctly identify that generic masking allows models to exploit periodic shortcuts, and propose a principled solution through cross-correlation-based mask generation. The geographic prompt tuning mechanism efficiently integrates 3D map information via GNNs, enabling parameter-efficient adaptation (only ~0.4% parameters updated during fine-tuning) while maintaining strong performance. The empirical results are impressive—34.4% MAE improvement over LWLM in single-BS scenarios and strong transfer to unseen environments. The ablation studies on masking strategies and map modalities provide useful insights.

**Weaknesses:**
Several issues limit the paper's impact. First, while LWM (Large Wireless Model by Alikhani et al. 2024) is mentioned in related work as a relevant foundation model, it is notably absent from experimental comparisons despite being directly relevant to the claim of "state-of-the-art." Second, all experiments use synthetic ray-tracing data (DeepMIMO, WAIR-D) without real-world CSI validation—this is a significant limitation for a practical localization system. Third, the paper claims "zero-shot generalization" but actually performs fine-tuning with ~100 samples; this is few-shot transfer, not zero-shot, which is misleading. Fourth, key algorithmic details are underspecified: the exact method for detecting periodicity shift d_final is not clearly described, and hyperparameter sensitivity analysis is missing. Finally, the comparison tables could be better organized (inconsistent table numbering in Section 4.5).

**Overall:**
The paper makes a solid contribution to wireless foundation models with a principled domain adaptation of masked autoencoding. However, missing baseline comparisons (LWM), lack of real-world validation, and some overclaiming prevent a stronger evaluation. The technical novelty is genuine but incremental.

Score: 7.0

---

## 9qbKOaF8YJ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Distribution-based Knowledge Distillation (DKD) for class-incremental semantic segmentation, addressing two key issues: parameter competition between old and new knowledge, and underutilization of previously acquired knowledge. The approach uses a minimization-maximization strategy with three loss components: L_Min releases low-sensitivity parameters from the old model, L_Esti estimates new knowledge distribution via Laplacian-based projection, and L_Max maximizes shared knowledge through entropy-based optimization.

**Strengths:**
- **Clear motivation**: The paper correctly identifies that standard KD methods constrain parameter fitting for new classes while wasting old knowledge. The minimization-maximization framework provides an elegant solution.
- **Strong empirical results**: The method achieves state-of-the-art performance on Pascal VOC and near-upper-bound results on ADE20K across multiple incremental settings. The 2-2 setting (10 steps) is particularly challenging and demonstrates the method's robustness.
- **Comprehensive evaluation**: The paper includes extensive ablations on each component, hyperparameter sensitivity, error analysis with repeated runs, both overlap and disjoint settings, and cross-architecture validation (ViT and ResNet101).
- **Theoretical grounding**: The appendix provides mathematical derivations for each loss component, showing how L_Min aligns predictions with pruned old knowledge, L_Esti bounds distillation error via Laplacian consistency, and L_Max maximizes mutual information.

**Weaknesses:**
- **Incremental novelty**: The individual components (parameter pruning, entropy-based regularization, knowledge distillation) are not novel. The contribution lies in their specific combination for CISS, which is meaningful but not groundbreaking.
- **Heuristic threshold**: The pruning threshold τ=0.1 is selected empirically without principled justification. Different tasks use different γ values, suggesting manual tuning is required.
- **Limited architectural comparison**: While compatibility with ResNet101 is shown, the primary experiments use ViT. More diverse backbone comparisons would strengthen claims.
- **Formulation clarity**: The position map computation (Eq. 4) appears incomplete—some terms in the second-order gradient computation are not fully specified.

**Overall:**
This is a solid contribution to CISS. The minimization-maximization perspective is insightful, results are strong, and the method approaches joint-training upper bounds. While not field-advancing, it represents a meaningful improvement over existing KD methods for incremental segmentation.

Score: 7.5

---

## xcBV0fK0ZK

- GT: Withdrawn (treated as Reject) (avg 1.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper presents a systematic study of adversarial robustness in LLM-based multi-agent systems (MAS) for engineering problems. The authors investigate how misleading agents affect collaborative reasoning across tasks like pipe pressure loss calculations, beam deflection, and graph traversal, varying prompts, task complexity, agent ordering, and personalization.

**Strengths:**
The paper addresses an important and underexplored domain—engineering applications require numerical accuracy and formal rigor, making adversarial vulnerabilities particularly consequential. The methodology is systematic, testing multiple dimensions (13+ prompt variations, 4 task types, various agent configurations) with proper statistical analysis. Key findings are practically useful: explicit warnings improve rejection rates dramatically (up to 87%), the "first mover effect" shows early speakers have disproportionate influence, and non-concise leader characters improve robustness. The experimental design controls variables carefully, and the authors commit to releasing code.

**Weaknesses:**
The primary concern is the unrealistic threat model. The "misleading agent" is explicitly programmed via system prompt to give wrong answers—an adversary would rarely have such direct control in real deployments. More realistic attack vectors (prompt injection, compromised API communications, jailbreaks) are not considered. The tasks are simple textbook exercises rather than realistic engineering workflows, and all experiments use only OpenAI models (GPT-4o, o3-mini), limiting generalizability. The paper characterizes vulnerabilities but offers no defense mechanisms beyond prompt engineering insights. Additionally, mechanistic explanations for *why* certain factors help are shallow—why does non-concise style improve robustness? The empirical patterns are presented without deeper theoretical understanding.

**Overall:**
While the systematic empirical approach and domain-specific focus are valuable, the artificial threat model limits practical relevance. The paper provides useful characterization but lacks the theoretical depth or defense proposals expected at a top venue. The contribution is incremental rather than field-advancing.

Score: 5.5

---

## ZOV3697bZZ

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**

This paper introduces In-Context Routing (ICR), a novel implicit ICL method that operates at the attention logits level rather than injecting vectors into residual streams. The key insight is that modulating attention geometry through Principal ICL Directions (PIDs) extracted via multi-domain PCA enables better generalization than task-specific shift vectors. The contributions are significant: (1) a theoretically-motivated approach grounded in spiked covariance models and Davis-Kahan bounds, explaining why pooled PCA captures transferable ICL patterns; (2) strong empirical results across 12 datasets (5 ID, 7 OOD) on three LLMs (Llama2-7B, Qwen2.5-7B, Llama3.1-8B), consistently outperforming vector-based baselines (I2CL, LIVE, M²IV) with zero collapse cases; (3) comprehensive analysis including ablations, layer/head importance, and mechanism discussion. The train-once-and-reuse framework is practically valuable, and the method achieves few-shot-level performance at zero-shot cost with strong OOD transfer.

**Weaknesses:**

Several aspects limit the paper's impact: (1) The near/far OOD distinction is somewhat arbitrary—MR (movie reviews for sentiment) arguably overlaps substantially with SST-2 training, weakening generalization claims; (2) Experiments focus on 7-8B models with limited exploration of larger scales (appendix-only for 32B/70B); (3) The mechanism discussion about "internalizing ICL patterns" remains speculative without direct probing of attention patterns; (4) The PCA rank choice (r=8) receives limited justification beyond showing r=4 and r=12 fail; (5) The theoretical analysis, while present, doesn't fully explain why attention routing should capture ICL patterns better than the residual vector approaches it critiques. The method also requires storing PIDs and training a router, which adds complexity compared to training-free alternatives.

**Overall Quality:**

This is a well-executed paper with a clear technical contribution and strong empirical validation. The attention routing paradigm offers genuine improvements over prior implicit ICL methods, particularly in OOD settings. While the theoretical claims exceed what's empirically demonstrated and model scale coverage could be broader, the work makes a solid contribution to the field.

Score: 7.5

---

## OPFE1zPYbU

- GT: Reject (avg 1.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper presents a thought-provoking analysis of diffusion models in high-dimensional spaces, challenging the conventional statistical interpretation of how they work. The authors argue that due to data sparsity in high dimensions, the posterior distribution p(x₀|xₜ) becomes highly concentrated on a single training sample, causing the fitting target to "degrade" from a weighted sum to essentially predicting the original training sample. Based on this observation, they propose a "Natural Inference" framework that reformulates various sampling methods (DDPM, DDIM, DPM-Solver, etc.) without relying on statistical concepts.

The paper has several strengths: the analysis of the weighted sum degradation phenomenon is mathematically sound and supported by empirical measurements on ImageNet. The unified treatment of different sampling methods as linear combinations of predicted clean samples is thorough and well-documented with coefficient matrices.

However, there are significant weaknesses. First, the core argument presents a false dichotomy: predicting x₀ from noisy inputs and learning statistical quantities (score, velocity field) are not mutually exclusive interpretations—they're complementary viewpoints. Even if the posterior concentrates on a single sample during training, the model still learns a meaningful denoising function that generalizes to novel inputs. Second, the observation about posterior concentration in high dimensions is not novel; similar insights appear in prior work (e.g., Karras et al. 2022, appendix B). Third, the practical implications are unclear—the paper doesn't propose new methods or demonstrate improved performance. The "Self Guidance" concept is largely a rebranding of linear combination operations that already exist. Finally, there's no experimental validation showing that this perspective leads to better models or sampling algorithms.

While the paper offers an interesting alternative perspective, it doesn't fundamentally advance our understanding or provide practical value to the community. The reformulation is technically correct but the conclusions are overstated.

Score: 5.5

---

## zuYXSoOzYG

- GT: Reject (avg 2.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a primal-dual learning framework for training monotonic neural networks, formulating monotonicity as a chance constraint in constrained optimization. The approach is novel in this domain—it avoids architectural restrictions and adaptively adjusts constraint penalties through dual variables, addressing real limitations of prior work (specialized architectures requiring manual regularization tuning).

The key strengths include the elegant formulation enabling use of arbitrary architectures, the adaptive regularization via dual variables (eliminating manual tuning), and comprehensive experiments across five datasets plus a control systems application. The method achieves competitive or superior results to state-of-the-art methods like SMNN and LMN while using fewer parameters in some cases. The chance constraint formulation (with α parameter) offers flexibility between strict monotonicity and prediction performance, which is practically useful.

However, there are significant weaknesses. First, Claim 1 provides a *sufficient* condition—the constraint $E[\max(t, 0 - \partial f_\theta/\partial x)] \leq \alpha t$ is stricter than the original chance constraint, making the method potentially over-conservative. This limitation is not discussed. Second, unlike certified approaches (Liu et al. 2020; Sivaraman et al. 2020), this paper provides no convergence analysis and does not verify that trained models actually satisfy monotonicity constraints empirically. The "continuously and adaptively enforce" claim lacks theoretical grounding. Third, while claiming "no pre-processing such as tuning of regularization," the method introduces new hyperparameters ($\alpha$, $t$, $\gamma_\mu$, $N$) without sensitivity analysis. Fourth, sampling from Uni($X$) to enforce constraints across the input space may not scale well to high dimensions. Fifth, some reported standard deviations are suspiciously low (e.g., 65.4% ± 0.0%), raising questions about experimental rigor.

The contribution, while interesting, applies well-known primal-dual optimization to an existing problem without substantial theoretical advances. The lack of monotonicity certification or verification in experiments is a notable gap for a paper focused on constraint satisfaction.

Score: 5.5

---

## DTQIjngDta

- GT: Accept (Poster) (avg 8.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces π[3], a permutation-equivariant architecture for visual geometry reconstruction that eliminates the dependence on a fixed reference view—a common design choice inherited from traditional SfM that the authors identify as introducing problematic inductive bias. The work demonstrates that existing methods suffer significant performance degradation when reference views change, and proposes an elegant solution through truly permutation-equivariant architecture that predicts affine-invariant camera poses and scale-invariant local point maps.

The paper's strengths are substantial. The identification of reference view bias as a critical limitation in feed-forward reconstruction methods is a genuine and valuable insight. The proposed solution is principled: by removing order-dependent components (frame index positional embeddings, reference tokens) and using relative pose supervision, the method achieves true permutation equivariance. The empirical validation is compelling—Table 6 shows near-zero variance across input permutations, compared to orders-of-magnitude higher variance in prior methods. The method achieves SOTA results across multiple benchmarks for pose estimation, depth estimation, and point map reconstruction, while also being faster (57.4 FPS vs. VGGT's 43.2 FPS).

However, there are notable weaknesses. First, the method relies heavily on VGGT pre-trained weights for initialization. When training from scratch, the authors must introduce a "global proxy" auxiliary task that paradoxically uses a reference view, suggesting the core method struggles with cold-start optimization. This raises questions about whether the improvements stem from the architectural changes or from better training initialization. Second, the architectural contribution is incremental—the encoder and alternating attention modules are directly borrowed from VGGT, with the main novelty being component removal and the relative supervision formulation. Third, while the pose distribution analysis (Figure 4, 6) claims improved geometric structure, the theoretical justification for why removing reference views yields this benefit remains somewhat shallow.

Overall, this is a solid contribution with a meaningful insight and thorough empirical validation. The permutation-equivariance demonstration is convincing, and the practical benefits (robustness, accuracy, speed) are clear. While the architectural novelty is limited and initialization dependency is a concern, the work advances the field by successfully challenging a deeply-ingrained design paradigm.

Score: 7.5

---

## 9rvefNQN1C

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces IBMDP (Implicit Bayesian Markov Decision Process), a framework for sequential assay planning in drug discovery where no simulator or transition data exists—only static historical outcomes. The key innovation is constructing an implicit generative model through similarity-weighted sampling from historical cases, combined with ensemble MCTS planning.

**Strengths:**
- **Novel problem formulation:** Addresses a genuinely important gap—sequential experimental planning with only static historical data, a common but underexplored setting in scientific domains.
- **Principled theoretical grounding:** The POMDP correspondence (Appendix A) is well-developed, showing similarity weights implement exact Bayesian belief updates over latent historical prototypes.
- **Practical impact demonstrated:** The CNS drug discovery case study shows impressive 92% resource reduction with maintained decision confidence against heuristics.
- **Synthetic benchmark with oracle:** Creating a controlled environment where the optimal policy is computable enables principled evaluation—IBMDP achieves 47% alignment vs. 36% for the deterministic baseline.
- **Comprehensive appendices:** Include theoretical consistency proofs, detailed algorithm specification, and notation reference.

**Weaknesses:**
- **Limited baseline comparison:** The paper argues comparing with traditional RL/POMDP methods is "unfair" (Appendix C), which has merit but leaves the reader without broader context. Only two baselines: VI-Theo (oracle) and VI-Sim (deterministic variant).
- **Synthetic benchmark is simplified:** Linear model with independent features doesn't capture real assay correlations, limiting how much the consistency theorem (Section D.6) applies to practice.
- **Hyperparameter sensitivity:** Many parameters (λw, λk, Ne, c, ϵ, τ) with limited sensitivity analysis acknowledged as a limitation.
- **Small real-world dataset:** Only N=220 compounds—scalability claims are theoretical.
- **Modest match rate:** 47% alignment with optimal policy, while explained as reflecting policy space exploration, may concern reviewers expecting stronger empirical results.

Overall, this is a solid contribution addressing a real practical problem with principled methodology. The empirical validation, while limited in scope, demonstrates meaningful improvements over heuristic approaches. The theoretical grounding distinguishes it from ad-hoc approaches.

Score: 7.0

---

## ewdqbKskUL

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces "answer-set consistency," a novel formalization for evaluating whether LLM outputs respect set-theoretic relations (equivalence, containment, disjointness) when answering enumeration questions. The authors create a carefully curated benchmark (ASCB) of 600 question quadruples capturing multiple relations simultaneously, evaluate 18 contemporary LLMs, and propose mitigation strategies.

**Strengths:** The problem formulation is genuinely novel—while LLM consistency has been studied for boolean facts and paraphrases, the consistency of set-valued answers across relationally-linked questions is unexplored. The benchmark construction is methodologically sound, combining existing KGQA datasets (LC-QUAD 2.0, QALD, QAWiki) with synthetic generation and manual curation. The empirical analysis is comprehensive, covering models from multiple families (GPT, Gemini, Llama, DeepSeek, Mistral, Grok), and the analysis distinguishes between stochastic variability and semantic misunderstanding as causes of inconsistency. The finding that CtE sometimes outperforms Oracle due to increased "idk" responses is an interesting insight about model uncertainty calibration.

**Weaknesses:** The mitigation strategies are incremental—CtE is essentially chain-of-thought prompting applied to this specific task, without novel technical contributions. The paper lacks comparison to alternative approaches like constrained decoding, post-hoc verification, or neurosymbolic methods. The error analysis (Appendix H) is brief; deeper investigation into why containment relations are particularly challenging would strengthen the work. The dataset is English-only and limited to static factual domains with 2-100 answers, constraining generalizability. Finally, the observation that newer/larger models don't universally outperform older/smaller ones deserves more investigation—it's mentioned but not deeply analyzed.

The paper makes a solid contribution to understanding LLM reliability, with a well-designed benchmark and thorough empirical study. However, the mitigation approaches are straightforward, and some analyses are underdeveloped.

Score: 7.0

---

## oBXfPyi47m

- GT: Accept (Poster) (avg 8.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Review of "Efficient Reinforcement Learning by Guiding World Models with Non-Curated Data"

## Assessment

This paper addresses a practical and important problem: leveraging non-curated offline data (reward-free, mixed-quality, multi-embodiment) to improve sample efficiency in online reinforcement learning. The authors identify that naive fine-tuning of pre-trained world models fails due to distributional shift between offline pre-training data and online fine-tuning data, and propose two techniques to address this: experience rehearsal (retrieving task-relevant trajectories from offline data) and execution guidance (using a BC policy to guide early exploration).

The paper's primary strength lies in its comprehensive empirical evaluation across 72 visuomotor control tasks from DMControl and Meta-World benchmarks. The results are compelling—NCRL achieves nearly double the aggregate score of learning-from-scratch baselines under limited sample budgets and substantially outperforms prior methods that leverage offline data. The analysis of why naive fine-tuning fails, including t-SNE visualizations and Wasserstein distance measurements, provides valuable insight into the distributional shift problem.

However, the technical contributions are somewhat incremental. Experience rehearsal is essentially retrieval-based replay, conceptually similar to prior work like RLPD but adapted for non-curated data. Execution guidance closely resembles JSRL's approach of mixing a prior policy with the RL policy, differing mainly in using BC instead of offline RL and allowing mid-episode switching. The theoretical analysis, while present, relies on assumptions that may not fully hold in practice. Additionally, the offline data, while "non-curated," is still in-domain (same benchmarks), which is less challenging than truly in-the-wild data.

The paper would benefit from deeper ablation analysis regarding computational overhead of retrieval, sensitivity to offline data quantity, and more thorough comparison at equivalent model sizes. Some baselines like iVideoGPT use additional techniques (reward shaping, demo pre-filling) that complicate fair comparison.

Overall, this is a well-executed paper addressing a practical problem with strong empirical results. While the individual components are incremental, the complete pipeline is thoughtfully designed and clearly effective.

Score: 7.5

---

## nHrYBGujps

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces BIRD-INTERACT, a benchmark for evaluating interactive text-to-SQL systems through dynamic multi-turn interactions. The work makes a compelling case that existing benchmarks inadequately capture real-world database interaction complexity, and proposes a well-designed framework to address this gap.

**Strengths:** The benchmark design is thoughtful and comprehensive. The distinction between c-Interact (conversational) and a-Interact (agentic) settings provides flexibility for evaluating different interaction paradigms. The function-driven user simulator is a notable contribution—by mapping clarification requests to constrained symbolic actions before generating responses, it significantly reduces ground-truth leakage compared to baseline LLM-based simulators (67.4% failure rate reduced to 2.7% on unanswerable questions). The ambiguity injection methodology is systematic, covering intent-level, implementation-level, knowledge, and environmental ambiguities. The inclusion of follow-up sub-tasks with state dependency (requiring systems to reason over modified database states) adds meaningful complexity. The memory grafting experiment is clever, demonstrating that GPT-5's poor c-Interact performance stems from communication deficiencies rather than SQL generation capability. The task scale (900 tasks, 11,796 interactions) and annotation rigor (93%+ inter-annotator agreement) are commendable.

**Weaknesses:** There are significant concerns. First, the paper repeatedly references "GPT-5" and "Claude-Sonnet-4"—models that, to my knowledge, do not exist publicly as named. The citation for GPT-5 ("OpenAI, 2025") lacks a proper reference. This raises serious questions about result authenticity and reproducibility. Second, the user simulator, while improved over baselines, still depends on pre-annotated clarifications from ground-truth SQL snippets, limiting its ability to handle novel clarification requests. Third, while the paper mentions prior interactive benchmarks (COSQL, SParC, MINT), it lacks direct comparison to any methods specifically designed for multi-turn text-to-SQL. Fourth, the budget constraint hyperparameters (patience=3, base budget=6) lack justification. Finally, the normalized reward metric uses arbitrary 70/30 weighting without explanation.

**Overall Quality:** Despite the model naming issues, the benchmark contribution is substantial. The work identifies a real gap in current evaluation, proposes a comprehensive solution with rigorous design, and provides meaningful analysis. The function-driven user simulator is a genuine methodological advance. However, the concerns about model references and reproducibility detract from an otherwise strong submission.

Score: 5.5

---

## B5NEdEQH1K

- GT: Reject (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents Jigsaw3D, a method for disentangled 3D style transfer that uses patch shuffling and masking to separate style from content in reference images before feeding them into a multi-view diffusion pipeline. The work addresses an important problem in 3D content creation—transferring artistic style from 2D images to 3D assets while maintaining geometric fidelity and multi-view consistency.

**Strengths:** The core idea of using a jigsaw operation to destroy semantic content while preserving style statistics is well-motivated and grounded in the observation that local patches carry sufficient style information. The training data construction—creating pseudo-paired style-texture data from Objaverse—is clever and enables supervised learning without requiring curated style-3D pairs. The paper provides comprehensive experiments with multiple baselines (StyleTex, MV-Adapter, 3D-style-LRM), thorough ablations on patch size and mask ratio, and demonstrates practical applications including scene styling, partial stylization, and tileable textures. The feed-forward inference (∼40s) offers significant practical advantages over per-asset optimization methods. Qualitative results are compelling, showing better style fidelity and semantic disentanglement than baselines.

**Weaknesses:** The primary concern is incremental novelty—the jigsaw/shuffling strategy for style disentanglement has been explored in 2D style transfer (e.g., StyleAdapter by Wang et al., 2023), and the contribution is mainly applying this idea to the 3D domain. The multi-view attention architecture largely follows prior work (MV-Adapter, MVDream). The evaluation uses only 20 test objects, which is limited, and while the paper shows competitive CLIP scores, the interpretation of CLIP score as a disentanglement metric (lower = better) is unconventional and could be better justified. The asymmetry between training (64×64 patches) and inference (128×128 patches) patch sizes appears ad hoc. Additionally, the method struggles with fine-grained patterns like text/symbols, which the authors attribute to SDXL limitations, but this represents a meaningful failure mode.

**Overall:** This is a solid contribution that makes a practical advance in 3D stylization. The training data construction and jigsaw-based disentanglement are useful ideas, even if borrowing from 2D work. The execution is thorough and the method produces high-quality results efficiently.

Score: 7.0

---

## XJXZXuTj11

- GT: Accept (Poster) (avg 6.8)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents QVGen, a novel quantization-aware training (QAT) framework specifically designed for video diffusion models, addressing an important and understudied problem. The authors make several valuable contributions: (1) a theoretical analysis connecting gradient norm minimization to improved QAT convergence, (2) auxiliary modules that reduce quantization errors during training while stabilizing optimization, and (3) a rank-decay strategy using SVD to progressively eliminate these modules without inference overhead.

The empirical evaluation is comprehensive, spanning four state-of-the-art video DMs (CogVideoX-2B/5B, Wan 1.3B/14B) with parameters up to 14B. The results are impressive: QVGen achieves near full-precision performance at W4A4 quantization, substantially outperforming prior QAT methods (LSQ, Q-DM, EfficientDM) and PTQ baselines. The ablation studies validate each component, and the analysis of singular value dynamics during training provides useful insight into why the rank-decay approach works.

Strengths include the practical motivation (video DMs are computationally demanding), the zero inference overhead design, and solid technical execution. The comparison of different decay strategies (linear, sparse, residual quantization) demonstrates that the proposed SVD-based rank-decay is genuinely better than alternatives.

Weaknesses are relatively minor: the training overhead is modest but non-zero (~2% more GPU-days); hyperparameter selection (initial rank, shrinking ratio) requires tuning; and 3-bit results show notable degradation, particularly on Scene Consistency. The theoretical analysis, while motivated, relies on convexity assumptions that don't hold for deep networks.

Overall, this is a well-executed paper addressing a timely problem with meaningful practical impact. The approach is novel, empirically validated, and comes with released code.

Score: 7.5

---

## yk3QBsB43u

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important and practical problem: out-of-distribution (OOD) detection when training labels are noisy. The authors correctly identify that while both label-noise learning and OOD detection have been extensively studied separately, their intersection remains underexplored. This is a significant contribution since real-world datasets inevitably contain label noise, yet most OOD detection methods assume clean labels.

**Strengths:**
The paper proposes a sensible framework combining loss correction methods with low-rank and sparse decomposition of feature embeddings. The intuition is sound: noisy labels corrupt the feature space, making it harder to distinguish ID from OOD samples. By decomposing features into low-rank (clean) and sparse (noisy/OOD-like) components, the method aims to recover a cleaner representation for OOD detection.

The empirical evaluation is comprehensive, covering synthetic noise settings (10-50%), real-world noisy datasets (CIFAR-10N, CIFAR-100N, Animal-10N), and multiple OOD test datasets. The results show consistent improvements over both standard OOD detection baselines and label noise-robust methods. Particularly impressive are the results at high noise rates (50%), where NOODLE achieves substantially better FPR95 than the best baseline. The ablation studies on hyperparameters provide useful insights into the method's behavior.

**Weaknesses:**
The novelty is somewhat incremental—the individual components (loss correction, low-rank decomposition via power iteration) are established techniques. The paper lacks theoretical justification for why this particular combination should work for OOD detection. There's no formal analysis of the decomposition's effect on OOD detection performance.

The method introduces several hyperparameters (λ, p, K) that require tuning. While ablations are provided, this raises practical concerns about deployment. Additionally, all experiments use DenseNet-101; generalization to other architectures (ResNet, ViT) is not demonstrated.

The synthetic experiments only use symmetric noise, but real-world noise can be instance-dependent or class-conditional. The paper would also benefit from comparing with more recent OOD detection methods and analyzing why certain loss correction methods work better in different scenarios.

**Overall:**
This is a solid paper addressing a relevant problem with reasonable methodology and strong empirical results. While the contribution is somewhat incremental and lacks theoretical depth, the practical significance and consistent empirical improvements merit acceptance.

Score: 7.5

---

## 3iHQ97INBP

- GT: Reject (avg 3.3)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

This paper proposes a method for learning interpretable (small) models by learning an optimal training distribution using uncertainty scores from a probabilistic "oracle" model. The core idea is to project training instances to one dimension using oracle uncertainty scores, model this as a Dirichlet Process-based Infinite Beta Mixture, and optimize the distribution parameters via Bayesian Optimization to maximize accuracy of size-constrained interpretable models.

**Strengths:**
- The approach is well-motivated and addresses a real challenge in interpretable ML: the size-accuracy trade-off.
- Comprehensive experimental evaluation across 13 datasets, multiple model families (DT, LPM, GBM), and different oracles (GBM, RF).
- Demonstrates practical versatility: works with multivariate model sizes and different feature spaces between oracle and target model.
- Shows impressive relative improvements (up to ~100%+) for small models.
- Meaningful applications to cluster explanation and prototype-based classification show competitiveness with specialized techniques (IMM, ExShallow, ProtoNN, SNC).
- Accommodates non-differentiable losses via BayesOpt, which is valuable for interpretable models like decision trees.
- Good empirical rigor with statistical significance tests and comparison to prior work.

**Weaknesses:**
- **Computational cost is a major limitation**: The paper admits runtimes of close to an hour for single configurations. While BoTorch alternatives are mentioned, results are preliminary and not systematically evaluated.
- **The "one hyperparameter" claim is misleading**: T (optimization budget) significantly impacts results and was tuned via "limited search" - this is effectively another hyperparameter.
- **Incremental novelty**: The core idea of learning training distributions comes from the authors' prior work (Ghose & Ravindran, 2020); the contribution is using uncertainty projections and DP-based density modeling.
- **No theoretical grounding**: Why uncertainty-based projections work well lacks formal analysis beyond intuitive justification about boundary proximity.
- **Ad-hoc procedures**: The flattening/smoothing technique in the appendix lacks strong justification.
- **Oracle dependency**: The method requires training a good oracle first, adding computational overhead and potential failure modes not explored.

**Assessment:**
This is a solid, practical contribution to interpretable ML with thorough empirical validation. The method shows meaningful improvements, especially for very constrained models. However, computational cost, limited novelty over prior work, and lack of theoretical grounding prevent it from being a standout contribution. The technique is most valuable when model size constraints are severe—a scenario well-suited to the evaluation but limiting in broader applicability.

Score: 5.0

---

## HAP8useYqu

- GT: Reject (avg 4.0)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper presents TempO (Temporal Operator flow matching), a method for long-horizon PDE forecasting that combines time-conditioned latent flow matching with an FNO-based vector field regressor. The key insight is to leverage FNOs as velocity field regressors within a flow matching framework rather than using them directly for autoregressive prediction.

**Strengths:**
The paper makes several genuine contributions. First, the theoretical analysis (Theorem 3.1, Proposition 3.2) provides meaningful bounds showing FNO regressors can achieve approximation accuracy with asymptotically fewer parameters than sampler-based architectures—a useful theoretical justification for the approach. Second, the architectural innovations (channel folding, sparse conditioning, multiscale autoencoder) are well-motivated by the PDE evolution operator perspective. Third, empirical results are comprehensive across three PDE benchmarks with clear improvements: 16% lower MSE on Navier-Stokes vorticity, stable Pearson correlations above 0.98 over 40 timesteps, and impressive efficiency (7× fewer parameters than ViT, 28× fewer than U-Net). The spectral analysis connecting model performance to physically relevant frequency content is particularly relevant for PDE applications.

**Weaknesses:**
Several issues limit the contribution. The theoretical bounds rely on spectral decay assumptions that may not hold across diverse PDE systems, and the connection to actual performance improvements remains qualitative. The baseline comparisons use architectures with vastly different parameter counts (TempO: 0.49M vs. ViT: 3.39M); while efficiency is part of the contribution, a controlled comparison at matched parameters would strengthen claims. The 40-step horizon, while reasonable, is still relatively short for many scientific applications. Finally, some components (attention-based autoencoder, FNO backbone) are existing techniques—the novelty primarily comes from their integration.

**Overall Quality:**
This is a solid, well-executed contribution that addresses an important problem with principled methodology and strong empirical validation. The theoretical grounding distinguishes it from purely empirical works, though the analysis depth could be improved. The efficiency gains and stable long-horizon predictions are meaningful for the scientific ML community.

Score: 7.2

---

## zmYx32SSOR

- GT: Reject (avg 1.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes four novel metrics for assessing individual fairness in ML models: Proxy Dependency Score (PDS), Counterfactual Stability Rate (CSR), Attribution Independence Score (AIS), and Intra-Cohort Decision Consistency (IDC). While the topic of individual fairness is important and under-explored relative to group fairness, the proposed metrics have significant methodological issues that undermine their validity.

**Major Weaknesses:**

1. **PDS has a fundamental conceptual flaw.** The metric computes `1 - Accuracy(shadow_model)/Accuracy(full_model)` where the shadow model excludes protected attributes. This measures whether protected attributes are *predictively useful*, not whether they create unfair proxy dependencies. A protected attribute that legitimately improves accuracy (e.g., age in healthcare predictions) would yield a high PDS, flagging it as "unfair" by construction, while a protected attribute that the model ignores would yield PDS ≈ 0 regardless of actual fairness. The negative values in Table 1 (-0.0123 for race in COMPAS) further suggest implementation or conceptual issues.

2. **CSR oversimplifies counterfactual fairness.** Simply flipping protected attributes while holding all other features constant ignores the causal structure that counterfactual fairness literature emphasizes. If changing race should also causally affect income or education in a counterfactual world, the proposed approach gives misleading results. This has been discussed extensively in Kusner et al. (2017) and subsequent work.

3. **AIS is underspecified.** The formula `1 - |corr(Attr_f(x), Protected(x))|` is unclear—Attr_f(x) produces feature attributions for *all* features, while Protected(x) is a scalar. How is correlation computed between a vector and scalar? Which attribution method (SHAP, LIME) is used matters significantly, yet this is not specified or analyzed.

4. **IDC uses arbitrary similarity.** K-means clustering does not implement the individual fairness principle properly. The number of clusters k is not justified, and K-means distance is not a meaningful similarity metric for fairness. The clustering is sensitive to initialization and feature scaling.

5. **Weak empirical evaluation.** The paper tests on only two datasets with no comparison to existing individual fairness metrics from the literature (e.g., from Dwork et al., Lahoti et al., or Mukherjee et al.). Critical implementation details are missing: what ML models were used, what hyperparameters, and whether protected attributes were included as features during training.

6. **Overstated claims.** The paper claims to show "models deemed unfair by group metrics may exhibit individual-level fairness" but the empirical evidence is thin—one model per dataset with no baselines or ablations.

**Minor Strengths:**
- Addresses an important problem (individual vs. group fairness gap)
- Provides pseudocode for implementation
- Connections to explainability via AIS are relevant

The paper needs substantial revision: clearer metric definitions grounded in existing fairness theory, comparison with existing individual fairness approaches, and more thorough experiments.

Score: 3.5

---

## txiGUfI4yF

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Latent Stochastic Interpolants (LSI), extending the Stochastic Interpolants (SI) framework to enable joint end-to-end training of an encoder, decoder, and latent generative model. The key contribution is deriving a principled ELBO objective in continuous time that supports simulation-free training, overcoming SI's requirement that both distributions be directly observed.

**Strengths:**
The theoretical contribution is substantial. The paper provides rigorous derivations in the appendix, including a generalized form of the path-space KL divergence via Girsanov's theorem. The core insight—using diffusion bridges to construct a variational posterior that allows simulation-free sampling—is elegant. The ELBO derivation cleanly unifies the reconstruction term with the dynamics mismatch penalty. Empirically, the method demonstrates competitive FID scores on ImageNet (2.62-3.91 across resolutions) with meaningful computational savings during sampling (48.6-73.6% FLOP reduction). The paper also explores practical aspects: the effect of loss weighting β, encoder noise scale, and different parameterization schemes. Supporting diverse priors (Gaussian, Uniform, Laplacian, Gaussian Mixture) preserves a key flexibility of SI.

**Weaknesses:**
The linear SDE assumption (ht zt, σt) for the variational posterior is a significant restriction that limits expressiveness—the paper acknowledges this but doesn't analyze its implications deeply. The FID results, while reasonable, are not state-of-the-art (e.g., SiD2 achieves 1.26 vs. LSI's 3.12 at 128×128). Key baselines are missing: no direct comparison with Latent Diffusion Models (LDM) or LSGM, which are natural competitors for latent-space generative modeling. The observation-space SI comparison is weak since SI itself is a framework rather than a standard baseline. Some implementation choices appear ad-hoc (tanh bounding, 3× dimensionality ratio, 16 transformer blocks) without thorough ablation or motivation.

**Overall:**
This is a solid contribution with sound theory and a meaningful extension of SI to latent spaces. The unified training objective is novel and the framework is well-developed. However, the empirical results are not exceptional, the linear SDE assumption is restrictive, and missing comparisons with LDM/LSGM weaken the evaluation. The paper would benefit from stronger baselines and deeper analysis of the theoretical limitations.

Score: 6.5

---

## Op6Scc62ME

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents DMIS, a unified framework for training conditional diffusion models under imprecise supervision (noisy labels, partial labels, and supplementary-unlabeled data). The work is well-motivated, as real-world datasets inevitably contain imperfect labels.

**Strengths:**
1. **Unified theoretical framework**: The decomposition of likelihood maximization into generative and classification components (Eq. 7-10) is elegant, and Theorem 1's insight—that imprecise-label scores are convex combinations of clean-label scores—is both clean and practically useful.

2. **Comprehensive empirical evaluation**: The paper tests across three distinct tasks (generation, weakly supervised learning, dataset condensation) on three datasets with strong baselines. Results are consistently strong, often achieving SOTA or competitive performance.

3. **Practical efficiency**: The timestep sampling strategy (Section 5) addresses computational concerns with diffusion classifiers, providing meaningful speedups.

4. **Novel application**: Noisy dataset condensation is an interesting and practical new problem setting that the paper pioneer studies.

5. **Clear writing and reproducibility**: The paper is well-structured with detailed algorithm descriptions and comprehensive appendices.

**Weaknesses:**
1. **Incremental technical novelty**: The core methodology combines existing ideas—weighted score matching for noise-robust diffusion (similar to Na et al., 2024) and EMA-stabilized classification objectives borrowed directly from PRODEN, self-training, and ELR. While the unification is valuable, the individual components are not novel.

2. **Limited architectural innovation**: The method uses standard EDM architecture; all contribution is in the loss design.

3. **Missing ablations**: There's insufficient analysis of key hyperparameters (EMA decay rate, timestep interval length Δ) or component contributions.

4. **Limited scale**: Experiments only cover 32×32 and 64×64 images; no large-scale validation on ImageNet or modern text-to-image models.

5. **Computational overhead**: Despite efficiency improvements, diffusion classifiers remain expensive compared to discriminative approaches—a significant practical concern not thoroughly discussed.

6. **Heuristic design**: The classification objectives for each supervision type (Eqs. 13-15) are essentially existing methods transplanted with minimal adaptation or justification for why diffusion specifically benefits from these formulations.

**Overall Assessment:**
This is a solid, well-executed paper addressing an important practical problem. The unified framework is conceptually clean, and empirical results are strong. However, the novelty is more about integration and unification than fundamental innovation. The work advances the field meaningfully but incrementally.

Score: 7.0

---

## 9X2NfyZpR2

- GT: Reject (avg 3.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents TbLTA, the first weakly-supervised approach for long-term action anticipation (LTA) that uses only video transcripts—ordered action lists without timing information—during training, eliminating the need for costly frame-level annotations. The approach combines temporal alignment via pseudo-label generation with cross-modal attention between video features and transcript embeddings.

**Strengths:** The paper addresses an important and timely problem. Dense frame-level annotations are expensive and limit scalability, making weakly-supervised alternatives valuable. The problem formulation is novel for LTA; while weakly-supervised temporal action segmentation has been explored, extending this to dense long-term anticipation is a meaningful contribution. The technical approach is sound: using ATBA for pseudo-label generation, CTC loss for transcript alignment, and CRF for sequence coherence provides a reasonable framework. The results are impressive in several cases—on Breakfast with 30% observation, TbLTA outperforms all supervised baselines, demonstrating that transcript-level semantic structure can indeed capture procedural regularities. The cross-modal attention mechanism for grounding video features is a nice touch that goes beyond using transcripts merely as weak labels.

**Weaknesses:** The technical novelty is largely incremental—most components (ATBA module, transformer decoder architecture, CRF formulation) are adapted from prior work, with the main contribution being their combination for this specific weakly-supervised LTA setting. The quality of the entire pipeline depends critically on pseudo-label accuracy from the ATBA module, creating potential for error propagation. On 50Salads, where videos are longer with denser action distributions, performance gaps with supervised methods are more pronounced (~10-15% in some settings), highlighting the method's sensitivity to temporal regularity. The duration prediction component remains notably weak. Additionally, the comparison to weakly-supervised baselines is limited to WS-DA (Zhang et al., 2021), which is semi-supervised rather than fully transcript-based, making direct comparison difficult. Finally, while transcripts are cheaper than dense annotations, they still require manual ordering effort, and the paper doesn't quantify this annotation cost reduction.

Overall, this is a solid contribution that opens a new direction for LTA research. The execution is competent and results demonstrate feasibility of the transcript-only paradigm, though technical limitations and performance gaps prevent it from being exceptional.

Score: 7.5

---

## azsnOWy9MZ

- GT: Reject (avg 5.3)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents PLORA, a system for efficient LoRA hyperparameter tuning through packing multiple LoRA configurations into concurrent training jobs. The work is motivated by observations that current LoRA fine-tuning significantly underutilizes GPU resources (SM occupancy around 16.7%, memory utilization under 55%) and that optimal LoRA hyperparameters vary substantially across tasks and models.

**Strengths:**

The paper provides a compelling empirical foundation, demonstrating through 1,000+ experiments that LoRA hyperparameters strongly influence model quality (accuracy differences up to 23.4% between configurations) and that optimal settings vary across tasks. The core insight—packing multiple small LoRA configurations to improve hardware utilization—is novel and well-motivated.

The technical contributions are solid: the ILP-based scheduling formulation with provable approximation bounds, and the custom CUDA kernels achieving near-linear speedups (up to 30× for 32 packed adapters). The kernel design for both forward and backward passes shows good technical depth. The evaluation across multiple model families (Qwen, LLaMA), sizes (3B-32B), and hardware (A100, A10) demonstrates broad applicability.

**Weaknesses:**

The baselines are limited to strawman sequential approaches (Min GPU, Max GPU). There's no comparison to existing hyperparameter tuning frameworks (Optuna, Ray Tune) or recent systems like mLoRA that could potentially benefit from similar insights. While the speedups are impressive, showing efficiency relative to established HPO systems would strengthen the contribution.

The paper relies on grid search for hyperparameter tuning, which isn't state-of-the-art. Integrating PLORA with Bayesian optimization or other efficient HPO methods could provide additional benefits not explored here.

Some practical concerns remain unaddressed: checkpoint management for dozens of concurrent adapters, metric logging infrastructure, and how gradient accumulation would interact with packing. The reliance on small batch sizes being optimal for LoRA (while empirically supported) somewhat limits the approach's generalizability.

The theoretical approximation bound (Appendix D) provides limited practical insight—the claimed bound depends on the last job's characteristics and doesn't meaningfully constrain worst-case performance.

Overall, this is a solid systems contribution with clear practical relevance, though with room for more comprehensive comparison to related work and practical considerations.

Score: 7.5

---

## yX1Nn63DwQ

- GT: Reject (avg 0.5)
- Predicted: N/A (3.0/10)
- Match: N/A

### Review

# Review: A New Efficient Method for Combining Gradients of Different Orders

## Assessment

This paper proposes GOC (Gradient Order Combination), a new optimization method for convex quadratic problems. The authors attempt to unify steepest descent (SD) and the CBB method under an "order" framework and develop higher-order variants using products of Hessian matrices.

**Major Weaknesses:**

The paper has significant issues that undermine its contribution. First, the writing quality is poor throughout, with numerous grammatical errors, typos, and unclear sentences (e.g., "Whave developed," "the the unconstrained," missing articles). This makes the paper difficult to follow and suggests insufficient preparation.

Second, the core contribution is poorly defined. The concept of "order" (first-order for SD, second-order for CBB) is introduced without rigorous mathematical definition. The leap from analyzing SD/CBB to proposing GOC lacks proper motivation—the derivation appears to simply combine terms with Hessian-vector products without clear theoretical justification.

Third, the experimental evaluation is inadequate. The method is only tested on a single synthetic convex quadratic problem with specific eigenvalue settings. There is no comparison with standard optimizers (L-BFGS, nonlinear CG, Adam) or testing on non-quadratic problems. The results lack statistical rigor (no multiple runs, confidence intervals, or ablation studies).

Fourth, the computational cost is not discussed. The algorithm requires computing A²g (two Hessian-vector products), which is expensive for large-scale problems. This practical limitation is completely ignored.

Fifth, the theoretical contribution is minimal—no convergence rate theorem is proven, and the claims about "faster descent rate" are not substantiated with rigorous analysis.

**Minor Strengths:**

The geometric interpretation connecting SD and CBB via symmetry is interesting, and the paper correctly notes that Hessian-vector products can be computed without forming the Hessian explicitly.

## Overall

The paper presents an underdeveloped idea with insufficient theoretical grounding, limited experimental validation, and poor presentation. The concept of method "order" needs proper formalization, and the practical utility of the approach remains unproven beyond a contrived example.

Score: 3.0

---

## ZS4fa5FgTD

- GT: Withdrawn (treated as Reject) (avg 2.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper introduces DyCO-GNN, a method for solving dynamic combinatorial optimization problems without requiring training data. The core idea builds on PI-GNN (an instance-specific unsupervised learning approach for static CO) and adapts it to dynamic graph settings using "shrink and perturb" (SP) to warm-start optimization across graph snapshots.

**Strengths:**
The paper addresses an important and underexplored problem—extending ML-based combinatorial optimization to dynamic settings, which is highly relevant for real-world applications. The proposed method is elegantly simple: rather than re-optimizing each snapshot from scratch or naively warm-starting (which leads to poor solutions due to model overconfidence), DyCO-GNN applies SP to balance fast convergence with solution quality. The empirical results are comprehensive across MaxCut, MIS, and TSP, demonstrating consistent improvements over baselines with reported 3-60x speedups. The theoretical analysis using the Goemans-Williamson algorithm provides grounding for the perturbation strategy. The paper is well-written and includes thorough ablations and sensitivity analyses.

**Weaknesses:**
The methodological contribution is relatively incremental—the SP technique is borrowed from supervised learning literature and applied to this new context. While effective, this limits the novelty. The baseline comparisons are restricted to the PI-GNN family; the authors argue this is appropriate but excluding trained GNN methods and classical dynamic/online optimization algorithms weakens the empirical case. For TSP, results are notably weaker than for MaxCut/MIS, requiring beam search for larger instances. The dynamic graph construction is somewhat artificial (simple edge additions/deletions), and more complex dynamics (node changes, edge weight modifications) are not explored. The method requires setting two hyperparameters (λ_shrink, λ_perturb), and while the authors use fixed values, the appendix shows performance can vary with these choices.

**Overall:**
This is a solid, well-executed paper addressing an important problem with practical relevance. The contribution is meaningful but modest in scope, with good empirical validation but limited theoretical depth and baseline breadth. The paper advances an important research direction for the ML-CO community.

Score: 7.0

---

## 9gw03JpKK4

- GT: Accept (Oral) (avg 8.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Gaia2, a benchmark for evaluating LLM agents in asynchronous, event-driven environments. The key insight is that existing benchmarks are synchronous—the environment only changes when the agent acts—whereas real-world agent deployments must handle time-sensitive tasks, adapt to dynamic events, and coordinate with other agents. The paper contributes both the ARE framework (a platform for building such benchmarks) and Gaia2 itself (1,120 scenarios across capability splits including Execution, Search, Ambiguity, Adaptability, Time, Noise robustness, and Agent2Agent collaboration).

The strengths are substantial. First, the asynchronous evaluation paradigm is genuinely novel and addresses a real gap—most benchmarks cannot assess temporal reasoning, responsiveness to environment changes, or multi-agent coordination. Second, the write-action verifier is a thoughtful contribution: by checking state-changing actions against oracle annotations (with 0.98 agreement and 0.99 precision on held-out trajectories), it enables fine-grained credit assignment usable for RLVR training. Third, the empirical analysis is thorough and yields actionable insights: the inverse scaling on Time tasks (reasoning-heavy models are slower and miss deadlines) has practical implications for agent deployment, and the heterogeneous multi-agent collaboration results suggest architecture directions. The cost-performance analysis adds practical relevance.

However, there are weaknesses. The absolute performance ceiling (GPT-5 high at 42%) raises calibration questions—while the benchmark should be challenging, it's unclear whether the difficulty spectrum is well-tuned. The verification relies on an LLM judge for soft checks, which introduces potential noise, though the authors acknowledge this. The ReAct scaffold, while model-agnostic, may not be optimal; the Time-split challenges could partially reflect orchestration limitations rather than model capabilities (though the PTC ablation partially addresses this). Finally, the individual components build on prior work (GAIA, AppWorld, ToolSandbox) without radical novelty in isolation.

Overall, this is a solid contribution that advances agent evaluation infrastructure and provides meaningful empirical insights. The asynchronous design is genuinely new, the framework enables community extension, and the findings have practical value. The limitations are acknowledged and don't undermine the core contributions.

Score: 7.5

---

## Ilnbgf1eeS

- GT: Withdrawn (treated as Reject) (avg 1.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes using Hamiltonian Monte Carlo (HMC) to train Bayesian neural networks with lottery ticket pruning masks, eliminating the need for the specific initialization that generated the mask. The core idea is interesting: since HMC has convergence guarantees to the true posterior, it should theoretically find optimal weights regardless of initialization, thereby "winning the lottery" with any mask.

**Strengths:**
The paper addresses a legitimate research question about the initialization dependence of lottery ticket networks. The conceptual framing—treating the initialization problem in a Bayesian manner—is novel and well-motivated. The authors provide comparisons across multiple architectures (LeNet variants, ResNet-18) and datasets (MNIST, CIFAR-10), and include both HMC and the more practical SVI method. The observation that deterministic pruning masks transfer reasonably well to Bayesian networks is a useful empirical finding.

**Weaknesses:**
The limitations significantly undermine the contribution. First, the experimental scope is very narrow—HMC experiments are restricted to tiny networks (LeNet), with ResNet only tested using SVI. Second, the claimed "theoretical grounding" is thin; citing HMC's known convergence guarantees is not a novel theoretical contribution. Third, the computational cost is prohibitive (days for HMC vs. minutes for standard training), making the approach impractical. Fourth, the absolute performance on CIFAR-10 (53-56%) is quite poor, making relative improvements less meaningful. Fifth, the paper lacks proper statistical analysis—multiple runs are mentioned but confidence intervals are inconsistently reported. Finally, there's insufficient engagement with alternative approaches to the lottery ticket problem (e.g., weight rewinding, learning rate rewinding) and limited comparison to the evolutionary strategies approach mentioned in related work.

The paper presents an interesting proof-of-concept but the practical utility and theoretical depth are insufficient for a top venue acceptance.

Score: 4.5

---

## cyQUZDMpg3

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents HIGS (History-Guided Sampling), a training-free enhancement for diffusion model sampling that uses momentum-based updates from past predictions. The work offers an interesting theoretical connection between Euler sampling and gradient descent, motivating the use of history-based momentum terms inspired by optimization techniques like STORM.

**Strengths**: The paper provides strong theoretical grounding with a formal error analysis showing improvement in local truncation error from O(h²) to O(h³). The empirical evaluation is thorough, spanning multiple models (SDXL, SD3, DiT, SiT), diverse metrics, and various sampling regimes. The SOTA FID of 1.61 on unguided ImageNet generation is impressive, and the training-free nature makes it immediately applicable to existing models. The extensive ablation studies help justify design choices.

**Weaknesses**: The method accumulates significant complexity through multiple components (EMA averaging, weight scheduling, orthogonal projection, DCT filtering) that may not all be essential. The hyperparameter sensitivity is concerning—Tables 10-12 show different parameter settings across models, suggesting per-model tuning is needed. The theoretical analysis only addresses a simplified version without the DCT filtering and projection components. Additionally, comparisons are limited to vanilla CFG; the paper would benefit from comparisons with other sampling enhancement techniques or momentum-based approaches. Recall metrics in Table 1 show mixed results (sometimes decreasing), raising potential diversity concerns.

**Overall**: This is a solid, practical contribution with meaningful empirical improvements, though the methodological complexity and hyperparameter dependence temper the impact somewhat. The work advances practical diffusion sampling but isn't transformative.

Score: 7.5

---

## L2rfd2Czbj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.8/10)
- Match: N/A

### Review

This paper introduces wd1, a weighted policy optimization method for diffusion-based language models (dLLMs) that eliminates the need for computing policy ratios during reinforcement learning. The key insight is that existing approaches like diffusion-based GRPO suffer from exponential amplification of likelihood approximation errors when computing importance sampling ratios—a significant issue given the intractability of exact likelihoods in dLLMs.

**Strengths:**
The paper makes several notable contributions. First, the ratio-free formulation using weighted log-likelihood is elegant and well-motivated, directly addressing a fundamental limitation of current dLLM RL methods. The theoretical analysis connecting the approach to energy-guided discrete diffusion training with negative sample unlearning provides strong justification for the method's effectiveness. Empirically, the results are impressive: wd1 achieves dramatic improvements on Sudoku (76.4% vs 17.6% for d1) and Countdown (51.2% vs 25.8%) without requiring supervised fine-tuning. The wd1++ extension achieves state-of-the-art results on MATH500 (44.2%) and GSM8K (84.5%) with only 20 training steps and 10× fewer rollouts than baselines. The computational efficiency gains—eliminating reference and old policy likelihood approximations—are meaningful for practical applications.

**Weaknesses:**
The paper has some limitations. While the comparison to d1 is thorough, comparisons to other concurrent methods (SDPO, TCR, MDPO) are limited and would benefit from more extensive analysis. The method relies on d1's likelihood approximation, inheriting any associated biases. All experiments use LLaDA-8B, leaving questions about generalization to other dLLM architectures. The ablation on negative sample weights (w⁻) shows importance, but more analysis of failure cases and why math benchmarks show more modest gains than reasoning tasks would strengthen the work.

**Overall Assessment:**
This is a well-executed paper addressing an important problem in an emerging research area. The theoretical grounding, strong empirical results, and computational efficiency gains make it a solid contribution. While there are some gaps in experimental scope, the core method is novel, sound, and demonstrates clear benefits over prior work.

Score: 7.8

---

## FMjeC9Msws

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper presents a systematic empirical study of reinforcement learning compute scaling for LLMs, motivated by the observation that while RL compute budgets have grown dramatically, our understanding of how to scale RL predictably lags behind pre-training. The authors propose a sigmoidal scaling framework relating compute to validation performance, characterized by three interpretable parameters: asymptotic performance (A), efficiency (B), and midpoint (C_mid). Through extensive ablations totaling 400K GPU-hours, they identify key design choices affecting scaling and propose SCALERL, a recipe combining PipelineRL, CISPO loss, FP32 precision, and other components. The key finding is that the 100K GPU-hour performance can be accurately predicted from early training data.

**Strengths:**
- **Scale and rigor**: The experimental scope is impressive, with systematic ablations across multiple design axes (loss type, off-policy algorithm, precision, aggregation methods, curriculum, etc.). The leave-one-out validation at 16K GPU-hours per run is thorough.
- **Practical framework**: The sigmoidal scaling law provides interpretable metrics for comparing RL methods. Unlike power laws, sigmoid curves naturally capture bounded metrics like accuracy, and the parameter decomposition (A for ceiling, B for efficiency) cleanly separates concerns.
- **Important empirical insights**: The paper's key observations—that not all recipes share the same asymptotic ceiling, that methods superior at small compute may be worse at scale, and that many design choices affect efficiency more than asymptote—are valuable for practitioners.
- **Validation across axes**: Testing scaling across batch size, generation length, model scale (MoE), and multi-task settings (math+code) demonstrates generality.
- **Predictability demonstration**: Successfully extrapolating from 50K to 100K GPU-hours validates the framework's predictive capability.

**Weaknesses:**
- **Limited algorithmic novelty**: SCALERL primarily combines existing techniques (CISPO from MiniMax, PipelineRL, FP32 precision fix). The contribution is empirical rather than algorithmic.
- **Narrow task scope**: Experiments focus on verifiable math/code reasoning. Generalization to RLHF, open-ended generation, or agentic tasks is unexplored.
- **Model scale limitations**: The 8B dense and 17B×16 MoE models, while respectable, are smaller than frontier models where RL scaling questions are most pressing.
- **Empirical grounding only**: The sigmoidal fit is justified empirically but lacks theoretical motivation. The comparison to power laws is done, but deeper theoretical analysis is absent.
- **Reproducibility challenges**: 400K GPU-hours of experimentation is beyond most researchers' resources, and some findings (e.g., about truncation-related instabilities) could benefit from deeper diagnosis.

Overall, this is a solid empirical contribution addressing an important practical gap in RL for LLMs. While the technical novelty is modest, the systematic methodology, scale of experimentation, and practical insights merit acceptance.

**Score: 7.5**

---

## xDO0239YOm

- GT: Reject (avg 1.3)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces HypoGeneAgent, an LLM-based framework for selecting clustering resolution in single-cell and Perturb-seq data by measuring annotation consistency. The core idea—using LLM-generated functional annotations to guide resolution selection—is novel and addresses a genuine gap in single-cell analysis where resolution tuning remains subjective.

**Strengths:**
The problem formulation is well-motivated: classical metrics like silhouette score and modularity are geometry-based and ignore biological interpretability. The proposed metrics (intra-cluster agreement and inter-cluster distinctiveness) are intuitive and the Resolution Score provides a principled optimization objective. The two-stage experimental design—first benchmarking on curated GOBP gene sets, then applying to Perturb-seq—is systematic. The ablation studies covering embedding methods, prompt designs, temperature settings, and LLM backbones are thorough.

**Weaknesses:**
Several significant issues undermine the paper's claims:

1. **Validation is weak**: Only one dataset (K562 Perturb-seq) is tested, with no clear ground truth for the "correct" resolution. The claim that selected resolutions "align with known pathway" is asserted but not rigorously demonstrated with quantitative evidence.

2. **GPT-5 references**: The paper reports results using "GPT-5," which does not publicly exist. This raises serious concerns about experimental validity and reproducibility.

3. **Circular evaluation**: The comparison against GO enrichment analysis is somewhat circular when the agent itself generates GO-based hypotheses.

4. **Arbitrary hyperparameters**: The weight w=1/3 for the Resolution Score is justified only as "chosen by a small grid search" with no details provided.

5. **No statistical testing**: Results are presented without confidence intervals or significance testing, making it difficult to assess whether observed differences are meaningful.

6. **Limited novelty in LLM application**: Gene-set annotation with LLMs has been explored in cited prior work; the main contribution (resolution scoring) is relatively straightforward.

The approach is interesting and the methodology is generally sound, but the validation is insufficient for the claims made, and the GPT-5 issue is concerning.

Score: 4.5

---

## ndIEb3Bghf

- GT: Reject (avg 1.3)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a regularization method based on characteristic functions, arguing that neural network outputs can be modeled as sums of Bernoulli random variables that (by Lyapunov's CLT) converge to Gaussian distributions. While the idea of bringing characteristic function theory into ML regularization is novel, the paper suffers from fundamental conceptual flaws that undermine its core claims.

**Major Theoretical Issues**: The application of the Central Limit Theorem here is deeply problematic. The CLT requires independent random variables drawn from *fixed* distributions, but neural network softmax outputs are deterministic functions of the input—they are not random draws from Bernoulli distributions. For any single forward pass, there is no "repeated sampling" happening. The number of classes K (often 10-1000) is fixed and typically too small for asymptotic results to meaningfully apply. The paper never clearly explains how $\phi_D(u)$ is actually computed from network outputs during training—the integral from $-\infty$ to $\infty$ is discretized but the practical implementation remains opaque.

**Empirical Concerns**: While extensive (16 datasets), the results are mixed. ElasticNet often performs comparably or better, particularly on smaller datasets. The claimed trend—that the method works better on high-class-count datasets—could simply be post-hoc rationalization rather than evidence for CLT-based arguments. No error bars or statistical significance tests are provided despite using Optuna for hyperparameter tuning. The custom GES and GenScore metrics are ad-hoc, complex combinations that make the baseline artificially equal to zero and may be cherry-picked.

**Presentation Issues**: The paper has significant notational confusion, unclear definitions (Axiom 1 states $\aleph \sim D$ but then defines $\aleph$ as a specific sum), and never bridges the gap between the theoretical framework and practical implementation.

**Strengths**: The literature review on characteristic functions is thorough, the experimental scope is broad, and the visualization of characteristic function behavior provides useful intuition. The core idea of distribution-aware regularization is worth exploring.

However, the theoretical foundation does not hold up to scrutiny, and the connection between the proposed regularizer and improved generalization remains unestablished. The method may work empirically in some settings, but not for the theoretical reasons claimed.

Score: 3.5

---

## EsumhpzFK9

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes KARMA, a framework that integrates knowledge graphs and causal inference to dynamically adjust reward signals in reinforcement learning. The core idea is to combine domain knowledge (encoded as knowledge graphs) with causal structure learning and counterfactual reasoning to refine potentially spurious or poorly-specified rewards.

**Strengths:**
The paper addresses an important problem—reward misspecification is indeed a central challenge in RL, and the recent RLVR findings about spurious rewards provide timely motivation. The proposed architecture is conceptually sensible, combining knowledge integration, causal discovery, and reward adjustment in a modular fashion. The empirical evaluation across three environments with multiple baselines provides useful comparative data. The ablation studies help isolate component contributions, and the analysis of generalization and robustness extends beyond simple performance metrics.

**Weaknesses:**
However, the paper has several significant issues. First, the causal inference component is underspecified. The paper claims to use "Pearl's do-calculus" and counterfactual reasoning, but provides no concrete details on how structural equations are defined, how counterfactual queries are actually computed from finite RL interaction data, or how confounders are handled. Causal discovery during online RL is notoriously difficult—the PC algorithm assumes i.i.d. observational data from a stationary distribution, but RL agents actively explore and their distribution shifts throughout training.

Second, the theoretical claims are overstated. The paper mentions "formal results on convergence and efficiency," but no actual theorems or proofs appear in the main text—they are relegated to an appendix not included here. The policy invariance claim relies on potential-based shaping assumptions that may not hold for the proposed causal reward formulation.

Third, the novelty is limited. Integrating knowledge graphs with RL and applying causal reasoning to RL are both established areas; the paper combines them without deep technical innovation. The dynamic weighting scheme between knowledge and causal rewards is relatively straightforward.

Fourth, the environments are quite simple (5×5 and 10×10 grids, simulated robotic arm), and the connection to the mentioned RLHF/RLVR problems is tenuous—the method hasn't been demonstrated on language models or complex environments where reward misspecification is most critical.

**Overall:**
The paper presents a reasonable integration of existing ideas with solid empirical results, but the causal inference claims lack technical depth, and the theoretical guarantees are not substantiated in the provided material. The contribution feels incremental rather than field-advancing.

Score: 4.5

---

## N8ntZEb4Ap

- GT: Withdrawn (treated as Reject) (avg 2.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes AutoNFS, a neural network approach for automatic feature selection that uses Gumbel-Sigmoid sampling with a cardinality penalty to determine both *which* features to select and *how many* to retain, without requiring manual specification of the feature count.

**Strengths:**
The paper addresses a genuine practical pain point—existing feature selection methods require users to specify the number of features a priori, which is often unknown. The proposed combination of Gumbel-Sigmoid sampling with a learned embedding and cardinality penalty is elegant and enables end-to-end differentiable training. The empirical evaluation is comprehensive: the method is tested on 11 OpenML benchmark datasets across three corruption scenarios, 24 metagenomic datasets, and includes computational complexity analysis. Results show AutoNFS consistently achieves top or near-top rankings while selecting significantly fewer features than baselines. The near-constant time complexity with respect to input dimensionality is a valuable practical property for high-dimensional data.

**Weaknesses:**
The technical novelty is incremental—Gumbel-Softmax/Sigmoid relaxation, cardinality penalties, and end-to-end neural feature selection are all well-established techniques. The relationship to prior work, particularly Stochastic Gates (STG) by Yamada et al. (2020), which also uses differentiable gates for feature selection with regularization, is not clearly differentiated. While the paper claims "automatic" feature count determination, the λ hyperparameter still implicitly controls sparsity—trading one hyperparameter for another. Several important baselines from related work (e.g., INVASE) are not included in comparisons. Additionally, the paper lacks ablation studies examining the contribution of each component, and there is no theoretical analysis of convergence or selection quality guarantees.

**Overall:**
This is a solid contribution with clear practical utility, thorough empirical validation, and consistent strong results. However, the incremental technical novelty and limited differentiation from existing neural feature selection methods prevent it from being a standout contribution.

Score: 7.0

---

## bbuxDoRD2D

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Review Assessment

This paper presents DISK, a differentiable framework for approximating dense convolution kernels using sequences of sparse kernels. The authors replace heuristic search methods (like simulated annealing) with gradient-based optimization, enabling more efficient convergence and better approximation quality. They also contribute a novel initialization strategy combining radial and sparse sampling for handling non-convex kernels, and propose filter-space interpolation for efficient spatially-varying filtering.

**Strengths:**
The paper addresses a practical problem in graphics and computational photography with a sensible technical approach. The differentiable formulation is cleaner than simulated annealing alternatives, and the empirical results clearly demonstrate superior approximation quality (lower LPIPS, higher PSNR) compared to both low-rank decomposition and parallel tempering baselines. The filter-space interpolation technique is elegant, decoupling kernel synthesis from image resolution—a practical contribution for real-time applications. The experimental evaluation covers diverse kernel types (Gaussian, geometric primitives, non-convex shapes, optical PSFs) and spatially-varying scenarios.

**Weaknesses:**
The core contribution is somewhat incremental—replacing simulated annealing with gradient descent is a natural progression rather than a breakthrough insight. A significant limitation is the requirement that the target dense kernel must be known a priori during optimization; this constrains applicability in scenarios where kernels must be estimated or learned jointly with images. The comparison methodology raises concerns: PST uses 100,000 total iterations versus 1,000 for the proposed method, which, while highlighting efficiency, doesn't isolate whether the quality difference stems from the optimization method or iteration count. The paper lacks theoretical analysis—no bounds on approximation error, convergence guarantees, or formal justification for why gradient descent outperforms simulated annealing in this setting. Finally, the baseline set is limited; comparisons to FFT-based methods or recent neural network approaches would strengthen the evaluation.

Overall, the paper makes a solid contribution to practical image filtering with competent execution, but the novelty is modest and several limitations remain unaddressed.

Score: 7.0

---

## VjGU55hEwV

- GT: Reject (avg 2.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

**Assessment:**

This paper introduces RLIE, a framework combining LLM-based rule generation with logistic regression for probabilistic rule combination. The work addresses a genuine gap: existing LLM-based rule learning methods don't systematically combine rules in a principled probabilistic manner. The authors propose a four-stage pipeline (Rule generation, Logistic regression, Iterative refinement, Evaluation) and systematically compare four inference strategies.

**Strengths:** The paper presents a well-motivated integration of classical probabilistic methods with LLM-based rule generation. The systematic comparison of inference strategies—particularly the finding that "Linear-only" prediction consistently outperforms injecting rules back into the LLM—is an interesting empirical contribution. The experimental evaluation across six datasets with multiple baselines (Zero-shot, IO Refinement, HypoGeniC) and backbone models provides reasonable coverage. The paper is clearly written with reproducible methodology.

**Weaknesses:** The core contribution is incremental. Using logistic regression to combine rule predictions is straightforward, and error-based iterative refinement is standard practice—the novelty lies primarily in applying these well-known techniques to LLM-generated rules. The experimental results show modest improvements; on some datasets, IO Refinement performs comparably or better, and differences between methods are often within standard deviation bounds. The dataset sizes are quite small (200 train, 200 val, 300 test), raising concerns about statistical significance. Additionally, the method relies on LLMs for "local judgment" (ternary rule evaluation), introducing another source of inconsistency that isn't analyzed in depth. No human evaluation of rule quality is provided, and computational costs of many LLM calls are not discussed.

**Overall:** While competently executed and making an interesting empirical observation about LLMs' limitations in probabilistic reasoning, the technical contribution is limited. The paper reads more like a thorough empirical study than a methodological advance.

Score: 5.0

---

## DjxNqXsApM

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

# Review: Enforcing Orderedness in SAEs to Improve Feature Consistency

## Assessment

This paper introduces Ordered Sparse Autoencoders (OSAE), which extends Matryoshka SAEs by enforcing a strict ordering of latent dimensions through nested dropout, aiming to address the permutation non-identifiability problem in SAE feature learning.

**Strengths:** The paper tackles an important and timely problem—SAE feature inconsistency across seeds and hyperparameters—which undermines mechanistic interpretability claims. The theoretical contribution (Theorem 3.1) provides formal guarantees for ordered recovery under spark conditions, with a complete proof in the appendix. The method itself is a sensible extension: treating each feature as its own "group" rather than sampling dictionary sizes adds determinism. The paper introduces useful evaluation metrics (stability and orderedness), and the cross-dataset experiments on real LLMs (Gemma2-2B, Pythia-70M) add practical relevance.

**Weaknesses:** Several issues limit the impact. First, the empirical gains are modest and come with trade-offs: O-SAE achieves higher orderedness but often worse stability for later features (see Figure 2b, where stability drops sharply after ~512 features), and consistently worse reconstruction loss than baselines—a significant cost not adequately analyzed. Second, the theoretical assumptions (spark condition, ordered atom frequencies) are restrictive and unlikely to hold in real neural network activations. Third, unit sweeping—shown to improve toy model results—is not used in main experiments without explanation. Fourth, the paper acknowledges limited hyperparameter sweeps, which confounds comparisons. Fifth, the practical significance of "orderedness" remains unclear: what downstream interpretability benefit does this ordering provide? Finally, critical baselines (orthonormality penalties, post-hoc alignment) mentioned in the introduction are not compared against.

**Overall:** The paper makes a reasonable contribution to SAE methodology with theoretical grounding, but the empirical trade-offs, restrictive assumptions, and lack of clear practical benefit limit its impact. The method improves orderedness at the cost of reconstruction quality and later-feature stability—a trade-off not well-justified.

Score: 5.0

---

## gkTx4sPyAw

- GT: Reject (avg 1.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

This paper proposes template-based generation for LLM tool calling as an alternative to schema-constrained generation (e.g., JSON). The central hypothesis is that template-based outputs more closely resemble natural language, which should better leverage LLMs' pretraining. The authors evaluate across three datasets (API-Bank, ToolACE, When2Call) and four models (GPT-4o, GPT-5, Mistral, DeepSeek-Coder).

**Strengths**: The core intuition is sound and the paper provides empirical support for the hypothesis on several model/dataset combinations. The error analysis categorizing failures (schema violations, incorrect tool names, etc.) offers useful insights into why template-based generation helps—particularly for reducing schema violation errors. Testing multiple datasets and models provides breadth, and statistical significance testing is included.

**Weaknesses**: Several issues undermine this work. First, the contribution is incremental—essentially comparing two prompting strategies without novel architectural or training innovations. Second, the results are mixed: GPT-5 shows performance *degradation* on multiple datasets, which contradicts the central hypothesis and isn't adequately explained. Third, GPT-5 appears to be referenced as a released model with API access, which raises credibility concerns about the experimental setup. Fourth, the template approach relies on regex parsing, which seems brittle and may not scale to complex nested parameters. Fifth, there's no comparison to function-calling APIs (e.g., OpenAI's native function calling) or grammar-constrained decoding methods. The analysis remains superficial—claiming "natural language alignment" without mechanistic evidence.

**Overall**: While the idea has merit, the execution reveals significant limitations. The mixed results on GPT-5, lack of comparison to established alternatives, and incremental nature of the contribution make this unsuitable for acceptance.

Score: 3.5

---

## 4OF7DQA3yW

- GT: Withdrawn (treated as Reject) (avg 1.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper proposes SCNet, a GAN-based neural vocoder that introduces two main innovations: (1) a subband condition network (CondNet) that provides prior frequency-domain knowledge to guide the backbone network, and (2) a magnitude-aware anti-wrapping phase loss that weights phase errors by target magnitude.

**Strengths:**
The dual-branch architecture is well-motivated, addressing the "black box" problem in GAN vocoders where intermediate features lack guidance. Predicting low-frequency subband signals (rather than full-band) is clever, as it avoids phase continuity issues associated with larger frame shifts. The magnitude-aware phase loss is technically sound—using sin²(Δθ/2) naturally handles phase wrapping, and magnitude weighting appropriately prioritizes perceptually important regions. Empirical results are strong: SCNet achieves superior MOS scores (4.21 vs 4.11 for BigVGAN on in-domain) with ~1/8 the parameters, and comprehensive ablations validate each component. The efficiency gains (145.67 vs 41.48 xRT for BigVGAN) are practically significant.

**Weaknesses:**
The contributions are somewhat incremental—subband processing exists in Avocodo, and anti-wrapping phase losses appear in APNet/APNet2. More concerningly, Table 2 reveals that BigVGAN (112M params) actually achieves comparable PESQ (4.027 vs 4.007 for SCNet at 2M steps), tempering claims of "superior performance." The comparisons in Table 1 retrain all baselines on the small train-clean-100 dataset, which may disadvantage methods designed for large-scale training. The paper is limited to 24kHz speech, missing evaluation at higher sampling rates common in modern vocoders. Additionally, the TTS experiment in the appendix (Table 12) uses only 50 utterances with limited details.

**Overall:**
A solid contribution with good empirical results and efficiency benefits, but with incremental novelty and some concerns about evaluation fairness. The sample efficiency argument is valid and valuable for practical deployment.

Score: 7.0

---

## o8kbAXPu7P

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper tackles an important and previously underexplored problem in temporal knowledge graph reasoning: handling emerging entities that have no historical interactions. The authors make several strong contributions.

**Strengths:**
The empirical investigation is thorough and compelling. The paper demonstrates that emerging entities constitute ~25% of entities in standard TKG benchmarks, causing significant performance degradation in existing methods due to representation collapse. This problem formulation is valuable and well-motivated. The proposed TRANSFIR framework leverages semantic similarity to transfer reasoning patterns through a VQ codebook—a sensible approach for the inductive setting. The experimental results are strong, with consistent improvements averaging 28.6% MRR across four datasets, and the ablation studies properly validate each component. The evaluation includes diverse baselines (transductive, path-based, and inductive methods), multiple settings (Emerging vs. Unknown), and efficiency analysis.

**Weaknesses:**
The approach relies critically on textual embeddings for semantic clustering, which limits applicability to TKGs without meaningful entity names. The ablation on GDELT shows this weakness clearly—performance improves when removing text features because entity names contain noisy abbreviations. While the combination of techniques is novel, the individual components (VQ codebook, Transformer encoders, pattern aggregation) are well-established. Many baselines compared are transductive methods not designed for inductive settings; although inductive baselines are included, the gap partly reflects problem mismatch rather than architectural superiority. The specific "Emerging" setting (first appearance only) is restrictive; the smaller gaps in the "Unknown" setting suggest advantages diminish with even minimal history.

**Overall Assessment:**
A solid contribution addressing a real problem with thorough empirical analysis and meaningful improvements. The methodology is reasonable, though reliance on textual features and the specific experimental setup limit broader impact.

Score: 7.0

---

## AjaicwLUmj

- GT: Reject (avg 1.0)
- Predicted: N/A (3.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a visualization system for multivariate time series using Palantir Foundry's AIP Agent platform, combining Vega charts ("Wirbelsäule-Plot") with LLM-powered prompts. While the application domain (athlete health monitoring) has practical value, the paper has significant shortcomings that make it unsuitable for a top ML venue.

**Strengths:** The paper addresses a real visualization challenge—displaying explanatory features at specific time points alongside multivariate time series. The integration of multiple components (ontology objects, LLM-generated tooltips, DTW for similarity comparison, and agent-based prompts) demonstrates a complete system. The security and access control considerations show awareness of practical deployment concerns.

**Weaknesses:** The paper suffers from fundamental issues. First, there is no clear research contribution—it primarily describes configuring existing tools (Vega charts, GPT-4, DTW, Palantir AIP) rather than proposing novel methods. Second, there is no evaluation: no baselines, no quantitative metrics, no user studies, and no comparison to existing visualization approaches. Third, the work is tightly coupled to proprietary commercial software (Palantir Foundry/AIP), severely limiting reproducibility and broader applicability. Fourth, the writing quality is poor: the paper has grammatical errors, inconsistent citation formatting (references [1]-[3] appear to be reused for both citations and footnotes), and reads more like marketing material than academic research ("Palantir Foundry is perfect platform for this task"). Fifth, technical details are sparse—there's no explanation of the actual algorithmic contributions in "modality encoding" or specifics on how the LLM integration works beyond calling GPT-4. Finally, the related work section does not survey relevant academic literature and instead focuses on domain-specific context.

The paper would be better suited as a technical report, white paper, or demo at an industry-focused venue, but lacks the methodological novelty, rigorous evaluation, and reproducibility expected at a top academic ML conference.

**Score: 3.0**

---

## WgMZPsdJmC

- GT: Reject (avg 0.5)
- Predicted: N/A (3.0/10)
- Match: N/A

### Review

**Assessment:**

This paper analyzes the steepest descent method for convex quadratic optimization by introducing a multiplicative coefficient t to the step length, studying how different values affect the dynamics of the parameter r (reciprocal of optimal step length). The authors derive explicit formulas for the 2D case and characterize three behavioral regimes: convergence to fixed values, oscillation between two values, and chaotic behavior.

**Strengths:**
The paper tackles an interesting problem - understanding gradient descent dynamics through a dynamical systems lens. The explicit derivation for the 2D case showing how the function G(r) governs the iteration dynamics of r_k has some merit. The observation that different step length coefficients lead to qualitatively different behaviors (stable, critical, chaotic) could potentially inform adaptive step size strategies.

**Weaknesses:**
The paper suffers from severe presentation issues that significantly impede readability. There are numerous typos ("steplenth," "DIFFER## ENT" in title), formatting problems with equations (strange characters like ~~~~ appear), and grammatical errors throughout. More substantively, the paper's contribution is unclear - the dynamics of steepest descent have been thoroughly analyzed by Akaike (1959), Forsythe (1968), and numerous subsequent works, yet this paper does not clearly articulate what new insights it provides. The extension to N dimensions lacks rigorous analysis beyond empirical observation. The experimental evaluation is extremely weak, consisting only of synthetic quadratic functions with no comparison to modern optimization methods or practical benchmarks. Finally, the claim that the "unstable/chaotic state" could accelerate convergence is counter-intuitive and not supported by theory or convincing experiments - chaos typically degrades convergence guarantees.

**Overall Quality:**
The core idea has potential, but the execution is poor. The paper reads like an incomplete draft rather than a polished submission. The theoretical analysis is limited to diagonal matrices (essentially coordinate-wise separable functions), significantly limiting generality. No practical algorithm or empirical performance gains are demonstrated.

**Score: 3.0**

---

## kI27Niy4xY

- GT: Accept (Oral) (avg 8.0)
- Predicted: N/A (7.8/10)
- Match: N/A

### Review

## Assessment

This paper presents VIST3A, a framework for text-to-3D generation that combines pretrained video generators with feedforward 3D reconstruction models through model stitching and direct reward finetuning. The key insight is elegant: rather than training a custom decoder from scratch—a costly process requiring large labeled datasets—the authors repurpose existing 3D foundation models (DUSt3R, VGGT, AnySplat) by identifying compatible latent representations and stitching them to video VAEs.

**Strengths:**
The paper makes a practical and timely contribution. As video generation and 3D reconstruction models rapidly improve, the stitching approach allows leveraging SOTA components without prohibitive retraining. The empirical results are strong—VIST3A consistently outperforms prior text-to-3DGS methods (Director3D, Prometheus3D, SplatFlow, VideoRFSplat) across T3Bench, SceneBench, and DPG-Bench. The framework is versatile, demonstrated across multiple video generators (Wan, CogVideoX, SVD, HunyuanVideo) and 3D models. The ablations are thorough, confirming the MSE criterion for layer selection and the value of each reward component. The paper also justifies the stitching approach theoretically via bounds on stitching risk.

**Weaknesses:**
The technical components—model stitching and direct reward finetuning—are borrowed from existing literature; novelty lies in their application to this problem. The method inherits limitations from its components, requiring video-coherent input sequences and being constrained by the quality of underlying models. Some design choices (specific LoRA ranks, reward weight schedules) lack extensive justification. The user study is modest (28 participants), though results align with quantitative metrics.

Overall, this is a well-executed paper addressing an important problem with a sensible, resource-efficient approach. The results are convincing, the methodology is sound, and the contribution advances text-to-3D generation meaningfully.

Score: 7.8

---

## vQLUAkl5SG

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes DRAGON, a training-free framework for LLM unlearning that combines a detection module with in-context chain-of-thought intervention. The work addresses practical limitations of existing unlearning methods, which often require retain data access and fine-tuning.

**Strengths:**
The paper tackles a practically important problem—real-world unlearning scenarios where retain data is unavailable and fine-tuning is impractical. The framework's design is sensible: detecting forget-worthy prompts via a trained scoring model combined with similarity matching, then routing these through a guard model that generates CoT instructions. The empirical evaluation is comprehensive, covering multiple datasets (TOFU, WMDP, MUSE), various model families and sizes, and a range of baselines including fine-tuning methods (GA, NPO, DPO) and training-free approaches (ICUL+, Filter-Prompting). The proposed metrics—Refusal Quality, Dynamic Deviation Score, and Dynamic Utility Score—thoughtfully address limitations in existing evaluation, particularly for continual unlearning. Results demonstrate consistent improvements across settings, and the ablation studies provide useful insights into the importance of the CoT component.

**Weaknesses:**
The core novelty is incremental—the combination of detection + in-context intervention for unlearning has been explored in prior work (e.g., ICUL, Guardrail+), and using CoT for safety alignment is established. The evaluation methodology has concerns: ICUL+ operates in an unrealistic "idealized" setting with perfect knowledge of forget data, and Filter-Prompting uses a near-perfect classifier, making fair comparison difficult. The "training-free" claim is somewhat misleading—the scoring model requires training on synthetic data, and the guard model needs fine-tuning. The detection module's near-100% accuracy on TOFU raises questions about how it handles adversarial perturbations beyond the limited attacks tested. The RQ metric weights are arbitrary (1, 1, 0.2) without empirical justification. Importantly, this approach doesn't truly "unlearn"—it's fundamentally a guardrail mechanism that suppresses outputs rather than removing knowledge from model weights. The scalability of the unlearn store for many unlearning requests is not thoroughly examined.

**Overall Quality:**
The paper addresses a meaningful problem with solid empirical results and introduces useful metrics. However, the incremental novelty, evaluation concerns, and the fact that the method is fundamentally output suppression rather than true unlearning limit its contribution. The practical utility for black-box LLMs is valuable, but not sufficient to overcome these issues.

Score: 5.5

---

## bhR00j6Mku

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents the first systematic study of benchmark contamination detection in Large Reasoning Models (LRMs), identifying a critical vulnerability: existing contamination detection methods fail in two key scenarios—when RL training follows SFT contamination (Stage I), and when CoT-based contamination is applied to advanced LRMs (Stage II).

**Strengths:**
The paper addresses a timely and important problem. As LRMs like DeepSeek-R1 gain prominence, ensuring fair evaluation is critical. The key finding—that GRPO-style RL training conceals contamination signals via PPO-style importance sampling and clipping—is non-obvious and well-supported through both empirical experiments and theoretical analysis. The ablation comparing RAFT (no concealment) vs. RAFT++ (con concealment) nicely isolates the clipping mechanism as the root cause. The experimental design is thorough, covering 10 detection methods, 6 benchmarks, and multiple base models, with clear controls demonstrating that concealment is not simply "forgetting" through additional training.

**Weaknesses:**
The primary limitation is that the paper diagnoses the problem without proposing a solution. While Section 5 outlines future directions, there's no new detection method or mitigation strategy offered—this is essentially an exposition of a vulnerability. The Stage II findings (that CoT-based contamination leaves minimal traces) are somewhat intuitive: if models learn reasoning patterns rather than memorize specific sequences, they will naturally generalize to similar distributions. The theoretical analysis, while valuable, could be more accessible to readers unfamiliar with PPO-style objectives. Additionally, the paper focuses primarily on GRPO/RAFT++; broader generalization to other RL algorithms (e.g., DAPO, REINFORCE variants) would strengthen claims about "a broad class of RL methods."

**Overall:**
This is a solid empirical contribution that identifies a genuine vulnerability in LRM evaluation. The systematic methodology, theoretical grounding, and clear presentation make it a valuable addition to the literature. While the lack of a proposed solution limits its immediate practical impact, raising awareness of this vulnerability is itself an important service to the community.

Score: 7.0

---

## 9WiPZy3Kro

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.8/10)
- Match: N/A

### Review

This paper introduces GROUNDCUA, a large-scale human-annotated dataset for desktop GUI grounding, along with GROUNDNEXT models trained on this data. The work addresses a genuine gap in the computer-use agents literature—high-quality desktop grounding data has been limited compared to mobile and web domains.

**Strengths:**
The dataset contribution is substantial: 56K screenshots with 3.56M human-verified annotations across 87 applications is genuinely valuable for the community. The human-verified aspect is crucial, as prior datasets relying on accessibility trees or DOM extraction often suffer from noise and incompleteness. The data efficiency claim is compelling—achieving SOTA results with 700K samples versus JEDI's 9M samples demonstrates that high-quality curated data can substitute for scale. The evaluation is comprehensive across five benchmarks, and the agentic evaluation on OSWorld-Verified shows practical utility. The ablation studies on instruction types and scaling behavior provide useful insights.

**Weaknesses:**
The RL contribution is modest both methodologically (simple reward function, standard RLOO) and empirically (1-2% average improvement). While the paper explains this as a "strong SFT ceiling," it raises questions about whether the RL component meaningfully advances grounding methodology. There's also a concern about dataset overlap: UI-Vision is acknowledged as "in-domain," which could inflate those particular results. The cross-platform generalization is uneven—web performance notably lags on ScreenSpot-v2. The paper focuses on open-source applications for licensing reasons, which may limit applicability to proprietary software commonly used in practice.

**Overall Assessment:**
The dataset is a genuine contribution that will benefit the research community, and the models achieve strong empirical results. The training methodology is competent if not groundbreaking. The paper is well-written and thorough. While the RL gains are limited, the core contribution—the dataset and SFT-based models—stands on its own merits.

Score: 7.8

---

## IqXlvYA7En

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a theoretical and methodological contribution to autoregressive image generation with diffusion loss. The work examines the relationship between conditional diffusion models and autoregressive diffusion models, proposing an optimal transport-based condition refinement approach to address "condition inconsistency" in autoregressive generation.

**Strengths:**
The paper offers substantial theoretical contributions, including formal proofs that patch denoising in autoregressive models mitigates condition errors (Theorem 2 shows exponential decay of gradient norm). The Wasserstein Gradient Flow formulation for condition refinement is novel and well-motivated, with a convergence guarantee (Theorem 3). The mathematical framework is rigorous, with proper lemmas, assumptions, and complete proofs in appendices. Experimental results demonstrate consistent improvements over MAR and other baselines on ImageNet across multiple model sizes (208M-943M parameters) and resolutions (256×256 and 512×512), with FID improving from 2.31→1.96 (208M) to 1.55→1.31 (943M).

**Weaknesses:**
The experimental validation has notable gaps. First, ablation studies are missing—there's no analysis of the impact of key hyperparameters (λ, ε, K iterations) or the OT refinement components in isolation. Second, while the paper acknowledges computational constraints for large-scale validation, the 943M parameter scale is modest compared to current SOTA models. Third, the connection between the theoretical insights and practical algorithm implementation could be clearer—for instance, how do the convergence guarantees translate to practical stopping criteria? Fourth, some implementation details remain underspecified, such as the architecture for autoregressive condition prediction. Finally, while multiple baselines are included, comparison with more recent approaches (e.g., VAR variants) would strengthen the evaluation.

**Overall Quality:**
This is a solid paper with meaningful theoretical contributions to understanding and improving autoregressive diffusion models. The rigorous mathematical framework and consistent empirical improvements justify acceptance, though the experimental gaps (missing ablations, limited scale) temper enthusiasm somewhat.

Score: 7.0

---

## ao7VBbvWIK

- GT: Withdrawn (treated as Reject) (avg 1.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**

The paper addresses a genuinely important problem in LLM-based software engineering: how to provide relevant yet structurally coherent code context within limited context windows. The core insight—combining structure-aware (AST-based) approaches with relevance-focused (hybrid IR) methods—is well-motivated and the tension between these paradigms is clearly articulated. The architectural design is sensible, with components like AST-aware chunking, call graph expansion, and reciprocal rank fusion being technically sound choices.

The writing is generally clear, and the related work section provides a comprehensive positioning of the contribution across four relevant research threads. The paper correctly identifies that neither pure structure-aware nor pure relevance-focused approaches suffice for code retrieval tasks.

**Weaknesses:**

The experimental evaluation has significant deficiencies. The curated dataset contains only 6 files—an extremely small sample that cannot support meaningful statistical claims. The reported Judge Scores of 90-100 across all files are suspiciously high and may indicate problems with the LLM-as-judge methodology, such as judge-model compatibility or lack of calibration. The paper defines "Hallucination Rate" as a metric but never reports it, leaving a key claim unsubsubstantiated.

The correlation analysis claiming r=-0.97 between compression and quality is meaningless with n=6 and the interpretation that compression harms quality contradicts the paper's thesis. No ablation study isolates component contributions, making it unclear whether the AST-guided chunking, hybrid retrieval, or call graph expansion drives any observed benefits.

The baselines are weak—comparing only against naïve truncation and ablated versions of HASTE rather than established code retrieval methods. No evaluation on standard benchmarks like HumanEval, MBPP, or Defects4J is provided. Critical implementation details are missing: how token budgets are enforced, exact prompt templates for the judge, and the Suggestion Generator's design.

**Overall Quality:**

While the problem is important and the approach is conceptually reasonable, the paper lacks the rigorous empirical validation expected at a top venue. The evaluation is too limited to support the claims made.

**Score: 4.5**

---

## f4oAYJxrgH

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Flatness-Aware Regularization (FA-Regularization), which adds a penalty term based on the trace of the squared Hessian to encourage convergence to flat minima. The method uses Hutchinson's stochastic trace estimator to make the curvature computation tractable.

**Strengths:** The paper addresses an important topic in deep learning—the relationship between loss landscape geometry and generalization. The proposed method is mathematically well-motivated, and the use of Hutchinson's estimator is a reasonable approach for efficient approximation. The authors are refreshingly honest about limitations, including computational overhead and inconsistent results across tasks. The writing is clear and the methodology is properly described.

**Weaknesses:** The empirical evaluation has significant issues. First, the model choices are unrepresentative—a 2-layer MLP on CIFAR-100 achieving ~26% accuracy is far from standard practice (modern CNNs achieve >70%); logistic regression on text is similarly simplistic. Second, the reported improvements are marginal: CIFAR-100 shows <1% absolute gain (26.3% → 27.0%), while the text and tabular tasks show *no improvement at all*. Third, there's no comparison to existing flatness-promoting methods like SAM, Entropy-SGD, or even basic data augmentation—this is a critical omission given SAM is the dominant approach in this space. Fourth, statistical rigor is lacking: only 3 runs per configuration with no standard deviations reported for CIFAR-100, making the marginal gains highly suspect. Fifth, the computational cost is prohibitive—40x slower training for <1% improvement is not practical. Finally, the novelty is limited; combining Hutchinson's estimator with Hessian-based regularization is straightforward, and the paper doesn't provide new theoretical insights beyond what's known from prior work on flatness and generalization.

**Overall:** While the paper is clearly written and tackles a relevant problem, the empirical contribution is weak—marginal improvements on an unrepresentative model, no improvements on other tasks, no comparisons to prior work, and prohibitive computational cost. The results do not convincingly demonstrate that the proposed method offers value over existing approaches.

Score: 3.5

---

## knHHCx1prj

- GT: Reject (avg 0.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Recurrent Deep Differentiable Logic Gate Networks (RDDLGNs), extending the prior DDLGN framework to sequential modeling tasks. The idea of applying discrete logic operations to recurrent architectures for potential FPGA acceleration is conceptually interesting, and the paper provides extensive hyperparameter ablation studies. The memorization experiments showing RDDLGN's superior retention over temporal shifts compared to RNN/GRU are a genuine strength.

However, the paper has significant weaknesses. Most critically, the BLEU scores are alarmingly low: 5.00 for RDDLGN and 5.98 for the Transformer baseline on WMT'14 En-De. These scores are an order of magnitude below reasonable expectations (standard baselines achieve 25-30+ BLEU). This suggests either severe undertraining, fundamental issues with the experimental setup, or the 16-token sequence truncation is crippling performance to the point where the task is effectively trivialized.

The efficiency claims lack validation. While the paper argues for lower energy/compute based on logic operation counts, no actual FPGA implementation or energy measurements are provided. The parameter count comparison is misleading—RDDLGN requires 4x larger embeddings (16.4M parameters) than baselines because binary representations are less expressive, yet this is presented as a design choice rather than a limitation.

The collapsed model shows substantial degradation (12% relative BLEU drop), the gradient analysis section lacks clear motivation, and the evaluation uses non-standard accuracy metrics for MT. The memorization experiments on shifted monolingual data don't meaningfully connect to translation quality.

Score: 3.5

---

## pW6rFymZ8F

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents EmbodiedMAE, a unified 3D multi-modal representation learning framework for robot manipulation, along with DROID-3D, a large-scale dataset with high-quality depth maps and point clouds. The work addresses an important gap in embodied AI: the lack of suitable 3D vision foundation models trained on domain-appropriate data.

**Strengths:**

The paper makes a valuable data contribution by creating DROID-3D, processing the full DROID dataset through ZED SDK to obtain temporally consistent depth maps and point clouds. Figure 2 effectively demonstrates the quality advantage over AI-estimated depth, addressing a real pain point for the community. The multi-modal MAE architecture with stochastic masking and cross-modal decoder fusion is well-motivated—the approach of sampling mask ratios from a Dirichlet distribution to avoid modality bias is sensible. Empirically, the paper is thorough: 70 simulation tasks (LIBERO, MetaWorld) and 20 real-world tasks across two platforms (SO100 and xArm) provide convincing validation. The finding that RGBD consistently outperforms point cloud in real-world settings due to sensor noise is practically useful, as is the enhanced preprocessing pipeline that significantly boosts PC performance (77.1% → 82.1%). The ablation studies on masking ratio, feature alignment positions, and encoder architectures are informative.

**Weaknesses:**

The architectural contribution is incremental—combining MAE with multi-modal inputs and cross-attention decoding follows established patterns from MultiMAE and similar works. The paper lacks deeper architectural ablations: why cross-attention for fusion versus concatenation or other strategies? How sensitive is performance to the Dirichlet concentration parameter α? The DINOv2-RGBD baseline appears intentionally weak ("naïvely incorporating depth"), and more competitive 3D baselines beyond PonderV2 would strengthen the comparison. The compute requirements for training ViT-Giant from scratch are substantial and not discussed in terms of accessibility. Finally, the depth quality claims lack quantitative metrics (RMSE, AbsRel) beyond visual comparison.

**Overall:**

This is a solid contribution with practical value for the embodied AI community. The dataset alone will benefit future research, and the model demonstrates consistent improvements with reasonable design choices. While not architecturally novel, the work is well-executed and addresses a genuine need.

Score: 7.5

---

## Dxb9zYD23D

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes ST-Diff, a novel framework for multivariate time series generation that reframes the problem as video generation by using STFT to create spectro-temporal video tensors. The key insight—that existing image-based approaches collapse temporal structure while time-domain models struggle with spectral dynamics—is well-motivated, and treating the time-frequency representation as video is a clever conceptual contribution.

The strengths of this work are substantial. The empirical results are impressive, with ST-Diff achieving state-of-the-art performance across most datasets and metrics. The Context-FID improvements, particularly on complex datasets like Energy and fMRI, demonstrate meaningful advances in capturing distributional properties. The scalability experiments (L=64, 128, 256) show that the approach degrades more gracefully than baselines as sequence length increases. The architectural design—including anisotropic patching and learnable attention biases for covariate and frequency axes—is thoughtful and well-justified by the data structure.

However, the paper has notable weaknesses. Most significantly, it lacks ablation studies to disentangle which components drive performance improvements. Without understanding the contribution of the STFT representation, anisotropic patching, attention biases, or the trend-residual decomposition, it's difficult to assess whether the video diffusion paradigm is essential or if simpler modifications to existing methods would suffice. The computational overhead of video diffusion models compared to time-domain alternatives is acknowledged but not quantified—a practical concern given that efficiency matters for real applications. Additionally, the STFT introduces hyperparameters (window size, hop length) whose sensitivity is not analyzed. Some results appear surprisingly strong (e.g., Discriminative Scores near zero), which raises questions about potential overfitting or evaluation methodology issues. Finally, concurrent work on frequency-domain diffusion (Crabbé et al., 2024) is mentioned but not empirically compared, leaving an important comparison unaddressed.

Overall, this is a solid contribution with a creative framing of the problem and strong empirical demonstration. The time-series-as-video paradigm offers a meaningful new direction that could influence future work, despite the noted limitations.

**Score: 7.0**

---

## 4S5x8yhJ5H

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces VIBEFACE, a facial biometric dataset specifically designed for electronic Know Your Client (eKYC) verification scenarios. The dataset comprises 2,250 images and 1,550 videos from 50 demographically balanced subjects, captured across multiple devices, lighting conditions, and with/without eyeglasses. The authors benchmark the dataset using standard face detection (MTCNN, RetinaFace, MediaPipe) and verification (ArcFace, MagFace) methods.

The paper's primary strength lies in addressing a genuine gap in existing resources—there are no publicly available datasets that combine eKYC-style verification videos with diverse acquisition conditions while maintaining ethical collection practices and demographic balance. The focus on practical authentication workflows (head rotations, blinking, expression changes) is timely given the growing adoption of remote identity verification in banking and other sectors. The explicit attention to GDPR and AI Act compliance is commendable and increasingly important for dataset publications. The demographic balance across gender, race, and age groups addresses a well-documented bias problem in existing face recognition datasets.

However, the paper has significant limitations. First, the scale of 50 subjects is modest for a biometric dataset, limiting statistical power for demographic subgroup analyses. Second, the benchmark experiments are elementary—simply running standard off-the-shelf models and reporting accuracy percentages. There's no algorithmic contribution, no comparison of performance between VIBEFACE and existing datasets to demonstrate its unique value, and limited analysis of failure modes or demographic disparities. Third, the detection results (RetinaFace and MediaPipe achieving near-perfect accuracy) suggest the dataset may not be sufficiently challenging for modern detection methods. Fourth, the verification experiments use an arbitrary fixed threshold of 0.5 without justification or ROC analysis. Finally, while PAD and deepfake detection are mentioned as potential applications, no baseline experiments are provided.

Score: 5.0

---

## 34V0IZytle

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a theoretical analysis of score-based generative models under the manifold hypothesis, identifying a fundamental rate separation between geometric and distributional learning. The key insight is that in the small-σ regime, information about the data manifold appears at order Θ(σ^{-2}), while information about the underlying distribution emerges only at Θ(1). This separation explains why diffusion models can capture data support even with imperfect score estimates, and suggests a paradigm shift toward geometric learning rather than full distributional recovery.

The paper's main strengths include: (1) The rate separation observation is genuinely novel and provides meaningful insight into why diffusion models succeed despite score estimation difficulties. (2) The theoretical framework is rigorous, with proper assumptions and detailed proofs using Laplace's method and WKB approximation. (3) The proposed Tempered Score Langevin dynamics offers a simple modification with theoretical guarantees for uniform manifold sampling. (4) The empirical validation on both synthetic manifolds and Stable Diffusion demonstrates practical relevance.

However, several limitations reduce the paper's impact. First, the practical implications for diffusion models are limited because the analysis doesn't track cumulative error along sampling trajectories—it analyzes a simplified setting assuming access to final distribution error. Second, the L∞ score error assumption is stronger than the L2 errors targeted by practical training objectives. Third, the WKB ansatz (Assumption B.2) is a non-trivial assumption about stationary distribution form that lacks thorough justification. Fourth, the paper analyzes continuous-time dynamics without quantifying discretization error crucial for implementation. Finally, while diversity improvements in experiments are promising, the empirical validation remains preliminary with limited hyperparameter exploration.

The theoretical insight about rate separation is significant and could influence how we understand and improve diffusion models. However, the gap between the theoretical analysis and practical implementation limits immediate practical impact. The uniform sampling result is mathematically elegant but may not always be desirable since data distributions encode useful information beyond geometry.

Score: 7.0

---

## UtFQNwWBaA

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents HiT-JEPA, a hierarchical self-supervised learning framework for trajectory representation that extends the JEPA (Joint Embedding Predictive Architecture) paradigm to capture multi-scale trajectory semantics. The work addresses a genuine limitation in existing trajectory representation methods, which typically operate at a single temporal granularity and cannot simultaneously capture fine-grained point-level details and global trajectory patterns.

**Strengths:**
The proposed three-layer hierarchy is conceptually sound, creating trajectory abstractions at point, segment, and route levels through convolutions and pooling. The attention propagation mechanism for hierarchical interactions is a reasonable approach to cross-level feature integration. The use of hexagonal grids (Uber H3) instead of rectangular grids is a thoughtful design choice that better respects spatial neighborhood relationships. Empirically, the paper demonstrates consistent improvements over strong baselines (TrajCL, CLEAR, T-JEPA) across six diverse datasets. The zero-shot transfer results are particularly impressive, showing substantial margins on TKY, NYC, and AIS(AU) datasets. The ablation studies, visualizations of learned attention patterns, and decoded trajectory representations provide useful interpretability insights.

**Weaknesses:**
The technical contribution is somewhat incremental—the hierarchical construction follows established patterns from vision and NLP (multi-scale representations via pooling) without substantial adaptation to trajectory-specific challenges. The attention upsampling mechanism, while functional, is relatively straightforward. The improvements on Porto and GeoLife are sometimes marginal or mixed compared to T-JEPA, raising questions about whether the added complexity always pays off. The paper would benefit from deeper theoretical justification: why three layers specifically, and how should layer weighting relate to trajectory characteristics? The ablation showing that direct embedding concatenation causes collapse is interesting but suggests the attention mechanism is primarily stabilizing rather than transformative.

**Overall:**
This is a well-executed paper addressing a relevant problem with solid empirical validation. While the architectural novelty is limited, the work demonstrates clear practical value and establishes a useful direction for multi-scale trajectory modeling.

Score: 7.0

---

## PtgQrxQ9Ak

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces AReUReDi, a multi-objective optimization framework for discrete sequence generation that extends Rectified Discrete Flows (ReDi) with annealed Tchebycheff scalarization, locally balanced proposals, and Metropolis-Hastings updates. The work addresses an important problem in therapeutic peptide design, where multiple competing objectives (affinity, solubility, hemolysis, half-life, non-fouling) must be simultaneously optimized.

**Strengths:**
- **Novel contribution**: This is the first extension of rectified discrete flows to multi-objective optimization, combining ideas from MCMC (locally balanced proposals) with generative modeling in a principled way.
- **Solid theoretical foundation**: The paper provides rigorous theorems on invariance, convergence to Pareto fronts, and coverage guarantees. The proofs are correct and the theoretical framework is well-developed.
- **Comprehensive empirical evaluation**: Experiments cover 8 protein targets for wild-type peptides and 5 targets for chemically-modified peptides, comparing against classical MOO methods (NSGA-III, SMS-EMOA, SPEA2, MOPSO) and a state-of-the-art diffusion baseline (PepTune). Ablation studies validate key design choices.
- **Clear algorithm presentation**: Algorithm 1 provides complete pseudocode, and the method is reproducibly described.
- **Practical relevance**: The ability to optimize 5 therapeutic properties simultaneously addresses a real need in drug discovery.

**Weaknesses:**
- **Computational efficiency**: AReUReDi is substantially slower than baselines (55s vs 2-33s for comparable tasks). The best-of-N comparison under matched wall-clock time shows the method produces better results with fewer samples, but the computational burden is significant.
- **Theory-practice gap**: The monotonicity constraint introduced for practical efficiency (accepting only improving moves) may compromise the theoretical Pareto coverage guarantees. This is acknowledged but the implications deserve deeper discussion.
- **Moderate predictor quality**: The score models have F1 scores of 0.58-0.71 for classification and Spearman correlations of 0.64 (affinity) and 0.86 (half-life). These moderate accuracies may limit real-world utility.
- **Missing statistical analysis**: Results report only averages without error bars or statistical significance tests, making it difficult to assess result reliability.
- **No Pareto front visualization**: For a multi-objective optimization paper, visualization of actual Pareto fronts achieved (rather than just average property values) would strengthen the claims.

The paper makes a solid, well-motivated contribution with proper theoretical grounding. While computational cost and predictor quality are limitations, the framework is novel and the results demonstrate meaningful improvements over baselines.

Score: 7.0

---

## K5t8PfzwFR

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

This paper presents VID, a dataset and benchmark for estimating human joint torques directly from real images, along with a baseline network (VID-Network). The work addresses an important practical problem: existing torque estimation methods require specialized equipment (motion capture, EMG, force plates) that limit real-world deployment, while this approach aims to enable torque estimation from monocular images.

**Strengths:**
The problem formulation is novel and practically meaningful—estimating biomechanical quantities from images could enable applications in sports analytics, rehabilitation monitoring, and human-robot interaction outside laboratory settings. The dataset contribution is valuable: 63,369 synchronized frames with kinematic and dynamic annotations, carefully processed and validated. The three-level evaluation protocol (overall, joint-specific, action-specific) provides meaningful structure for future comparisons. The reported 39.81% improvement over baselines appears substantial at first glance.

**Weaknesses:**
However, there are significant concerns. First, the comparison methodology is problematic—Dino and ImDy use motion capture data as input, while VID-Network uses images. This is not an apples-to-apples comparison; the baselines should be adapted to use images, or a pipeline approach (image→pose→torque) should be compared. Second, ground truth quality is under-analyzed: torques are computed via OpenSim inverse dynamics from motion capture, inheriting errors from marker placement, soft tissue artifacts, and numerical differentiation. The "manually validated" claim lacks detail on validation methodology or error quantification. Third, critical evaluations are missing—no cross-subject generalization testing despite having only 9 subjects, and no analysis of failure cases (e.g., why does the method underperform on walkingTS1/walkingTS2?). Fourth, the dataset is derived from OpenCap/AMASS rather than newly collected, making the contribution primarily curation rather than data creation. Fifth, architectural details are sparse—pre-training datasets and hyperparameter sensitivity are insufficiently documented. Finally, class imbalance across actions (squats: 800+ frames vs. walking: ~150 frames) may bias results but is not discussed.

Score: 5.0

---

## SkmkGKEZ1U

- GT: Reject (avg 0.5)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper presents O-Forge, a system combining LLMs with Mathematica's Resolve function to prove asymptotic inequalities. The key idea is sensible: LLMs propose domain decompositions, and the CAS verifies each piece via quantifier elimination. The authors position this as addressing Terry Tao's question about AI tools for research mathematics.

**Strengths:** The motivation is clear and well-articulated. Proving asymptotic inequalities is indeed a time-consuming aspect of research in analysis and number theory. The division of labor between LLM (creative decomposition suggestions) and CAS (rigorous verification) is a reasonable architecture. The authors have created an accessible web interface, which lowers barriers for mathematicians without programming expertise. The case studies, attributed to Terry Tao, demonstrate the approach on non-trivial examples.

**Weaknesses:** The evaluation is severely deficient. The paper mentions testing on "around 40-50 easier problems" but provides no systematic results—no success rates, no comparison to baselines, no failure analysis, no statistics. There is no comparison to simply asking an LLM directly, or using Mathematica alone. The prompt structure section contains only empty XML tags. The claim of addressing "research-level" mathematics is not convincingly demonstrated; the examples appear to be standard exercises solvable with known decomposition techniques. The paper relies entirely on Mathematica's closed-source Resolve function without producing verifiable proof objects, undermining claims of "rigorous verification." Technical details are thin: no information on which specific LLM versions are used, temperature settings, retry strategies, or failure modes. Significant related work in automated theorem proving and mathematical assistants is not adequately addressed.

**Overall:** The core idea has merit, but the paper reads more like a tool demo than a research contribution. The complete absence of quantitative evaluation, baseline comparisons, and failure analysis makes it impossible to assess the actual capabilities or limitations of the system. For a top venue, this level of rigor is insufficient.

Score: 3.5

---

## 2eAGrunxVz

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper proposes Spherical Watermark, a novel lossless watermarking framework for diffusion models that eliminates the need for per-image key storage while maintaining visual fidelity and traceability. The core technical innovation is a mapping strategy that transforms binary watermarks into Gaussian-distributed latent noise using spherical designs.

**Strengths:**
1. **Novel mathematical framework**: The use of spherical t-designs to construct a mapping from binary watermarks to approximately Gaussian noise is elegant and theoretically grounded. The proofs showing that the constructed noise matches the Gaussian distribution up to third-order moments are rigorous.

2. **Practical advantages**: Eliminating per-image key management (unlike Gaussian Shading) and avoiding complex cryptographic operations (unlike PRC Watermark) provides real practical benefits. The ~10,000x speedup in extraction time compared to PRC Watermark is significant.

3. **Comprehensive evaluation**: The paper thoroughly evaluates undetectability via FID, classifier-based detection, and distribution analysis. Robustness is tested across post-processing attacks, adversarial attacks, and image editing operations. Ablation studies cover key design choices.

4. **Theoretical insight on losslessness**: Appendix E provides an interesting analysis connecting losslessness to adversarial robustness via the Fisher divergence bound, explaining why lossless schemes resist detector-aware attacks.

**Weaknesses:**
1. **Limited theoretical guarantee**: The proof only establishes matching up to third-order moments, not full distributional indistinguishability. While empirical results support near-indistinguishability, higher-order statistics could theoretically be exploited.

2. **Security assumption on secret key**: The method relies on keeping the rotation matrix C secret. The paper doesn't analyze what happens if this secret is compromised or leaked, nor discuss key rotation strategies.

3. **Incomplete comparison with PRC**: While computational efficiency is highlighted, the paper could better analyze the error-correction tradeoffs between the repetition code used here versus the error-correcting codes in PRC.

4. **Marginal improvements over baselines**: In some robustness tests (Figure 5), improvements over PRC Watermark are modest. The main advantages are efficiency and simplicity rather than dramatic performance gains.

5. **Hyperparameter sensitivity**: While ablations show robustness to parameter choices, the default settings (N=31 repetitions for 512-bit watermarks) result in significant redundancy. A deeper capacity-robustness analysis would be valuable.

The paper makes a solid, incremental contribution to lossless watermarking for diffusion models. The approach is mathematically principled, well-evaluated, and addresses real practical limitations of prior work, though the theoretical guarantees are incomplete and practical security considerations could be strengthened.

Score: 7.0

---

## s7gSTR2AqA

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

This paper investigates whether large language models (LLMs) exhibit human-like inductive biases toward efficient semantic categorization, using the Information Bottleneck (IB) framework and color naming as a testbed. The authors conduct two experiments: an English color naming task across 39 models, and a novel Iterated In-Context Language Learning (IICLL) paradigm to simulate cultural evolution of category systems.

The paper has several strengths. First, the research question is timely and important—understanding whether LLMs develop human-aligned semantic systems despite not being explicitly trained for this objective has significant implications for AI alignment and cognitive modeling. Second, the theoretical grounding in the IB framework is solid and enables direct quantitative comparison with human behavioral data. Third, the methodology is comprehensive, testing 39 models across multiple dimensions including size, instruction-tuning, and multimodality. The IICLL paradigm, adapted from Zhu & Griffiths' iterated in-context learning, provides a principled way to probe inductive biases beyond training data memorization. The finding that only Gemini 2.0 recapitulates the full range of human IB tradeoffs is interesting and suggests meaningful differences in emergent capabilities across frontier models.

However, there are notable weaknesses. The IICLL experiments only test 4 models (the best performers from naming), which severely limits generalizability of conclusions about inductive bias. The claim that LLMs exhibit "human-like inductive bias toward IB-efficiency" is not fully supported—models may simply have learned statistical patterns from training data that align with IB-efficient solutions. The failure of models when prompted with CIELAB coordinates (Appendix C) is concerning and not well-explained; if models truly understand color categorization principles, they should generalize beyond sRGB. Additionally, the paper provides limited mechanistic insight into why certain models (particularly Gemini) succeed while others fail. The scope limitation to color naming, while well-motivated, constrains broader impact. Finally, statistical analysis is largely visual/qualitative without formal significance tests.

Score: 5.0

---

## jsPQFNmnln

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes Latent Basis Function NPE (LBF-NPE), a novel variational family for neural posterior estimation that parameterizes log densities via basis expansions. The key idea is to represent the log density as a linear combination of basis functions, either fixed (B-splines, wavelets) or adaptively learned. This creates an exponential family representation that enjoys favorable optimization properties, including marginal convexity when basis functions are fixed.

**Strengths:**
The method offers a creative middle ground between simple Gaussian variational families (which are easy to optimize but insufficiently expressive) and normalizing flows/MDNs (which are flexible but suffer from local optima). The theoretical analysis demonstrating marginal convexity under fixed basis functions is meaningful, as are the connections to NTK theory and global convergence results. The empirical evaluation is comprehensive, spanning synthetic benchmarks, astronomical object detection, and cosmological redshift estimation on real survey data. Results consistently show improvements over MDNs and neural spline flows, with particularly strong performance on multimodal posteriors. The stereographic projection reparameterization to address identifiability issues is a nice practical touch.

**Weaknesses:**
The method fundamentally requires computing a log-normalizer integral via Monte Carlo or grid-based methods, which limits scalability to high dimensions. While the paper tests a 50-dimensional experiment, it evaluates only 2D marginal posteriors after integration—the core method remains constrained to low-dimensional latent spaces. Sampling from the fitted variational distribution requires inverse transform sampling or Langevin dynamics, both of which become impractical in higher dimensions. The number of basis functions K is a critical hyperparameter with limited guidance on selection. Some comparisons (e.g., against EigenVI) may favor LBF-NPE since EigenVI requires orthogonal bases while LBF-NPE can learn flexible bases.

**Overall:**
This is a solid contribution that addresses a real gap in the NPE literature. The approach is well-motivated, theoretically grounded, and empirically validated. However, the method's inherent limitation to low-dimensional settings restricts its applicability. The contribution is incremental rather than field-advancing—it provides a useful tool for a specific class of problems rather than a broadly transformative technique.

Score: 7.2

---

## lfsjVdi72l

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents SEED-SET, a Bayesian experimental design framework for ethical evaluation of autonomous systems that combines objective metrics with subjective stakeholder preferences using hierarchical Variational Gaussian Processes. The key contribution is a novel acquisition strategy that jointly considers information gain in both objective and subjective layers.

**Strengths:**
The problem formulation is timely and practically important—ethical evaluation of autonomous systems is understudied, and the paper correctly identifies key challenges: difficulty measuring ethical behavior, value alignment being user-dependent, and expensive real-world evaluation. The hierarchical VGP decomposition (separating objective GP from subjective GP) is well-motivated and interpretable, grounding preferences in observable system outcomes rather than latent parameters. The proposed acquisition function elegantly balances exploration (via mutual information terms) with exploitation of learned preferences. The empirical evaluation spans three domains (power grid, fire rescue, optimal routing) with multiple ablations, and the use of LLMs as proxy evaluators addresses the practical challenge of obtaining human feedback at scale.

**Weaknesses:**
The evaluation methodology has significant limitations. The hand-crafted "preference score" functions used to evaluate alignment are somewhat circular—they're designed to match the LLM's stated preferences rather than validating against real stakeholder judgments. While the TravelMode case study (Appendix H) uses real human data, the comparison is limited to distribution matching rather than preference accuracy. The baselines are relatively narrow; missing comparisons to multi-objective BO methods and more recent preference learning approaches. The claimed scalability to "high-dimensional" spaces (40D in 30-Bus) is modest by modern BO standards. Additionally, the assumption that objective metrics are fully known a priori (A3) may not hold for genuinely novel ethical concerns.

**Overall Quality:**
The paper makes a solid contribution to an important emerging area. The technical approach is sound, the hierarchical decomposition is principled, and the acquisition strategy is well-designed. However, the evaluation relies heavily on proxy metrics and LLM simulations, which limits confidence in real-world applicability. The improvements over baselines are consistent but not dramatic. With stronger validation involving actual human stakeholders or more extensive baseline comparison, this could be a stronger contribution. As presented, it represents a reasonable but not exceptional advance.

Score: 7.0

---

## n1AvXiU2lu

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper introduces "Real-time reasoning" as a new problem formulation for LLM agents and proposes Real-Time Reasoning Gym to evaluate agents in dynamic environments where time doesn't pause during reasoning. The key insight is valuable: real-world agents must balance timely reactions with deliberate planning, and current evaluation paradigms assume static environments that wait for agent computation.

The paper's strengths are substantial. First, the problem formulation is genuinely novel and addresses a real gap—existing benchmarks like WebArena or SWE-bench assume the environment halts during reasoning, which doesn't match real-world deployment. Second, the gym design is thoughtful: three games (Freeway, Snake, Overcooked) test distinct challenges (hazards, opportunities, coordination), and the two-axis control of cognitive load and time pressure enables systematic evaluation. Third, the token-based time abstraction is clever and validated against wall-clock time (R² = 0.998), ensuring reproducibility. Fourth, AgileThinker's dual-thread architecture with partial reasoning trace sharing is conceptually clean and demonstrates consistent improvements across conditions—outperforming both reactive and planning baselines as difficulty and time pressure increase.

However, there are notable limitations. The environments, while carefully chosen, are relatively simple grid-based games with discrete actions—far from real-world complexity. The model scope is restricted primarily to DeepSeek V3/R1 because reasoning traces are required; the Gemini experiments use an approximation that accesses only completed outputs, not partial traces. The paper doesn't directly compare to existing dual-process architectures (e.g., the cited Zhang et al. 2025 or Liu et al. 2024) as baselines. Additionally, AgileThinker requires tuning the reactive thread budget N_TR per environment, though the dynamic adjustment mechanism partially addresses this.

The empirical work is thorough: significance tests confirm advantages, wall-clock experiments validate the token abstraction, and concurrent (resource-sharing) execution still shows benefits. The paper successfully opens a new research direction with clear motivation, solid methodology, and comprehensive evaluation.

**Score: 7.5**

---

## dUgq4bLY4X

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents a rigorous mathematical framework for understanding and mitigating "symmetry increase" in equivariant neural networks (ENNs)—a phenomenon where symmetric inputs lose discriminative information through equivariant mappings.

**Strengths:**
1. **Substantial theoretical contribution**: The paper introduces the concept of "symmetry infimum" and proves it is uniquely determined by the feature space structure (Thm 3.1, 3.3). This provides the first general framework for predicting when symmetry degradation occurs, significantly advancing beyond prior work that documented isolated cases.

2. **Complete mathematical treatment**: The proofs leverage sophisticated tools from equivariant topology (orbit type stratification, Whitney conditions, equivariant transversality) with complete derivations in the appendix. The treatment covers both faithful and non-faithful group actions, essential for practical applications.

3. **Practical algorithms**: The proposed algorithms for computing symmetry infimum (Algo 1, 2) are concrete. The comprehensive tables in Appendix E for SO(3)/O(3) subgroups provide immediate practical utility for molecular and materials applications.

4. **Theoretical depth**: Thm 5.2 establishes that "almost isovariant" maps are generic under appropriate conditions, connecting the theory to practical approximation results for TFN and similar architectures.

5. **Clear organization**: The progression from fundamental definitions through algorithms to empirical validation is logical and well-presented.

**Weaknesses:**
1. **Limited architectural innovation**: While the paper provides analysis tools, it doesn't propose new architectures that automatically satisfy the guidelines. Practitioners must manually select appropriate feature degrees based on the tables.

2. **Computational considerations**: The algorithms require enumerating subgroup lattices, which could be challenging for novel groups not covered in the appendix.

3. **Empirical scope**: The QM9 experiments validate theoretical predictions but show modest practical impact on downstream task performance. Testing on more diverse tasks and datasets would strengthen empirical claims.

4. **Narrow group scope**: While SO(3)/O(3) are highly relevant, extension to other groups (SE(3), permutation groups) would increase impact.

The paper makes a solid theoretical contribution that fills an important gap in understanding ENNs. The mathematical framework is rigorous and complete, with empirical support. The contribution justifies acceptance, though practical impact is somewhat limited.

Score: 7.5

---

## pcaHnwjnsO

- GT: Reject (avg 1.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

# Review of "Graph Adversarial Refinement for Robust Code Fixes"

## Assessment

This paper proposes GARM (Graph Adversarial Refinement Module), a method to improve robustness of policy networks for automated code repair by combining graph structure learning with adversarial contrastive training. While the problem of adversarial robustness in code models is important and timely, the paper has several significant issues that undermine its contribution.

**Strengths:** The paper addresses a relevant problem—adversarial vulnerability in RL-based code repair systems. The modular design combining graph structure learning, adversarial perturbation generation, and contrastive learning is sensible. The experiments cover multiple datasets and include ablation studies. The idea of identifying "less critical" edges for targeted perturbation rather than random attacks has merit.

**Weaknesses:** The paper has serious presentation and methodological issues. First, there's a glaring naming inconsistency—the abstract introduces "GARM" while Section 1 suddenly refers to "GBAR" (Graph-Based Adversarial Refinement), suggesting inadequate proofreading. The writing quality is poor throughout, with phrases like "which mainstreamer the structural weaknesses" that don't make sense. Second, critical sections (6.1-6.3 on limitations, applications, and ethics) are essentially empty placeholders—this is unacceptable for a complete submission. Third, Figure 1 is referenced but only has a caption with no actual visualization content. Fourth, the technical novelty is limited: the edge importance scoring (Equation 3) is standard attention, the perturbation operations are well-known graph modifications, and the contrastive loss formulation is standard. The contribution is primarily in combination rather than innovation.

**Evaluation concerns:** The citation "(Liu et al., 2025)" references a future publication, which is problematic for reproducibility. No statistical significance tests or multiple experimental runs are reported. The perturbation budget of 15% lacks justification. Baseline implementation details are sparse, making fair comparison difficult.

Overall, while the problem is worthwhile, the paper appears to be an incomplete draft with substantial writing issues, missing content, and incremental technical contribution.

## Score: 4.0

---

## qNlTH4kYJZ

- GT: Accept (Oral) (avg 7.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Review

This paper introduces AdAEM, a dynamic and self-extensible framework for evaluating LLMs' value orientations. The authors identify an important "informativeness challenge" in existing value benchmarks: static datasets with generic or contaminated questions yield saturated, indistinguishable results across LLMs. The proposed solution uses an information-theoretic optimization approach to automatically generate value-evoking questions by probing diverse LLMs' boundaries.

**Strengths:**

The paper addresses a genuine and timely problem in LLM evaluation. Static benchmarks indeed face issues of data contamination and overfitting, and the observation that existing value benchmarks produce indistinguishable results across diverse LLMs is compelling and well-motivated. The theoretical formulation using generalized Jensen-Shannon divergence with a disentanglement regularization is principled, and the EM-like optimization with multi-armed bandit exploration provides a sound mechanism for question generation.

The experimental validation is thorough. The authors demonstrate question quality through diversity metrics, validate measurement validity via controlled priming experiments (showing expected value changes when models are primed), analyze regional/temporal patterns in generated questions, and compare against multiple baselines. The creation of a 12,310-question benchmark that can be regenerated addresses the contamination issue directly.

I particularly appreciate the ethical considerations section and the acknowledgment of limitations. The extension to Moral Foundation Theory shows generalizability beyond Schwartz's values.

**Weaknesses:**

The gap between the theoretical formulation and practical implementation is substantial. The authors use multiple approximations (sampling-based estimation of intractable probabilities, BERTScore surrogates for semantic coherence terms) that may undermine the theoretical guarantees. While the empirical results support effectiveness, the connection between the elegant theory and implemented approximations feels tenuous.

The reliance on GPT-4o/Mini as value classifiers introduces potential circularity—these models have their own value biases that may influence what "values" are detected. The paper could benefit from human validation of extracted values or alternative validation approaches.

The complexity of the pipeline raises questions about necessity. Could simpler approaches (e.g., prompting diverse LLMs to generate controversial questions directly) achieve similar results? An ablation study comparing against such baselines would strengthen the contribution.

The appendix-heavy presentation, while standard for venue limits, buries important implementation details. Key approximations and hyperparameter justifications require careful appendix reading to fully evaluate.

**Overall:**

This is a solid contribution that addresses a real problem with a principled (if approximated) solution. The self-extensible nature and demonstrated ability to reveal cultural differences in LLM values are valuable. While the methodological complexity may exceed necessity and some approximations warrant scrutiny, the work provides a useful framework that could influence future value evaluation practices.

Score: 7.5

---

## pfw176o1YJ

- GT: Accept (Oral) (avg 7.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents a systematic investigation into how LLMs develop visual priors from text-only pre-training. The authors decompose visual priors into perception and reasoning components, showing that reasoning-centric data (code, math, academia) progressively scales visual reasoning ability while perception priors emerge more diffusely from broad corpora. Through extensive experiments (~100 controlled experiments, 500K GPU-hours), they derive a practical data mixing recipe (~60% reasoning + 15% visual content) that improves downstream MLLM performance.

**Strengths:** The scale and rigor of experiments are impressive - spanning multiple model sizes (340M to 13B), systematic ablations across 16 data sources, and validation at 1T token scale. The conceptual decomposition of visual priors into separable perception and reasoning components is meaningful and well-supported empirically. The correlation analysis showing these abilities are loosely coupled is insightful. The proposed data mixing recipe is actionable and practically useful for practitioners. The introduction of MLE-Bench for evaluating perception at different object scales is a useful contribution.

**Weaknesses:** The scope is limited to adapter-style architectures, leaving open questions about tokenization-based approaches. Safety implications of text-derived visual priors (potential biases) are not addressed. The absolute performance gains from the balanced recipe are modest (~1.3 points on VQA average). Model scale tops out at 13B, so applicability to frontier models remains unclear. Some sections are verbose with redundancy.

The paper makes a solid contribution through rigorous empirical validation of intuitions about pre-training data composition. While not paradigm-shifting, it provides valuable, actionable insights for building better MLLMs.

Score: 7.5

---

## r12Cz05mUT

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces LAMP, a method for test-time latent adaptation in masked diffusion language models. The core idea is to identify low-confidence tokens, apply policy-gradient updates to their hidden states guided by reward signals, and use "clamp-and-inpaint" decoding to propagate edits while maintaining global coherence.

**Strengths:**
The paper addresses a timely and underexplored problem—test-time reasoning enhancement for diffusion LLMs, which have received far less attention than autoregressive models. The methodology is clearly presented with pseudo-code, and experiments cover multiple benchmarks (GSM8K, MATH-500, AIME) across three model backbones (LLaDA, LLaDA-1.5, Dream). The qualitative analysis and self-reward transition dynamics provide useful insights into when the method helps versus hurts.

**Critical Weaknesses:**
The most serious issue is that the "Perfect Sparse Reward Model" (PSRM) used for the main results **relies on ground truth answers during inference**. PSRM returns reward=1 only if the model's answer matches the ground truth label. This is an oracle setup that cannot be used in any realistic deployment scenario. The impressive headline numbers (+13.3 on GSM8K, +16 on MATH-500) therefore reflect an unrealistic scenario where the model is told whether its answer is correct. When using the practical self-reward (which doesn't require ground truth), improvements drop to only +1-3 points and are sometimes negative. This disparity should be the paper's central focus, not buried in the results.

Additionally, there's no comparison to standard baselines like best-of-N sampling or self-consistency that would also benefit enormously from oracle verification. The qualitative analysis (Table 9) reveals that the method causes TRUE→FALSE regressions where correct answers become incorrect—concerning for a method meant to improve reliability. The AIME results show many zero baselines with minimal or no improvement, suggesting limited applicability to challenging problems.

The core technical contribution—applying policy-gradient to diffusion latent states—is largely adapted from LatentSeek for autoregressive models, with the novel diffusion-specific element being the clamp-and-inpaint mechanism. While non-trivial, this isn't a major conceptual advance.

**Overall:**
The paper's primary experimental claims are based on an unrealistic oracle reward, while the practical self-reward results are modest and inconsistent. The method shows instabilities that could degrade already-correct outputs. Without meaningful baselines or realistic reward settings, the claimed benefits are difficult to evaluate fairly.

Score: 4.0

---

## r99m9ziONQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents IA2 (ICL Activation Alignment), a method that improves supervised fine-tuning (SFT) by first priming the model with activations from in-context learning (ICL). The key insight is that ICL and SFT produce fundamentally different activation patterns despite similar surface-level outputs, and ICL activations contain richer, more generalizable representations. The authors propose a two-step pipeline: first align model activations to match ICL's functional behavior, then perform standard SFT.

**Strengths:** The paper makes several notable contributions. First, the observation that ICL and SFT occupy distinct activation spaces—with SFT weights being nearly orthogonal to IA2-induced weights—is genuinely interesting and provides mechanistic insight into why these methods differ. Second, the empirical evaluation is extensive: 12 benchmarks spanning classification and generation tasks, two model families (Qwen3 and Llama-3.2), and over 13,000 trained models. The consistent improvements in both accuracy and calibration across most settings are convincing. Third, the subspace overlap analysis is a strong piece of evidence that IA2 accesses a training signal fundamentally unavailable to SFT alone. The comparison to knowledge distillation baselines and testing on multiple PEFT methods (LoRA and (IA)³) adds credibility.

**Weaknesses:** Several limitations merit attention. The experiments are restricted to 1B-4B parameter models; whether findings scale to 70B+ models remains unknown, which is significant given deployment trends. The method introduces computational overhead (collecting ICL activations requires a forward pass), though the authors argue this is one-time. More concerning is dependency on ICL response quality—if ICL performs poorly, aligning to its activations could be harmful (partially observed in QASCr experiments). The paper also lacks comparison to recent context distillation methods beyond brief mentions. Finally, while MSE loss on activations is intuitive, the paper doesn't explore alternative alignment objectives or justify this choice theoretically.

The writing is clear and the methodology reproducible. The connection to prior work on ICL-as-gradient-descent provides useful context, though deeper theoretical analysis would strengthen the contribution. Overall, this is a solid empirical contribution with practical implications for the fine-tuning community.

Score: 7.5

---

## 5o0zF03RP9

- GT: Withdrawn (treated as Reject) (avg 0.5)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces IncentRL, a framework that augments RL objectives with a KL-based penalty between predicted and preferred outcome distributions, with the weight β adapted via Bayesian inference. The core idea—adapting the exploration-exploitation trade-off online rather than through manual tuning—is motivated by cognitive science connections to dopamine RPE and the Free Energy Principle.

**Strengths:** The paper addresses a legitimate problem in intrinsic motivation methods: the need to manually tune trade-off parameters. The theoretical grounding connecting KL-based shaping to established frameworks (FEP, RPE) provides useful conceptual framing. Propositions 1 and 2 correctly characterize limit behavior. The MiniGrid experiments demonstrate sample efficiency gains in a sparse-reward setting.

**Weaknesses:** The paper has significant issues that undermine its claims:

1. **Bayesian adaptation is underspecified**: The abstract and introduction prominently claim "Bayesian adaptation of β" as the central novelty, yet the methods section contains *no description* of the prior, likelihood model, or update equations. Figures 3-4 and Table 3 present adaptation results without explaining how they were computed. This is a critical omission for reproducibility.

2. **Missing baseline comparisons**: The paper compares only against vanilla RL (β=0). There are no comparisons to established intrinsic motivation methods like ICM, RND, or DIAYN—baselines that are standard in this literature. The claim that IncentRL "improves over standard RL and fixed-regularization baselines" is weak without these comparisons.

3. **Limited experimental scope**: The three experiments (2-state MDP, MountainCar, MiniGrid DoorKey) are relatively simple. MountainCar shows improvement only at β=0.1 with degradation at higher values, suggesting brittleness. No experiments on harder exploration benchmarks (e.g., Montezuma's Revenge, larger MiniGrid mazes) or continuous control.

4. **Preference specification opacity**: How q(o|s) is constructed in each experiment remains unclear. If these are hand-specified using knowledge of the goal state, the method's practical applicability is limited.

5. **Marginal empirical gains**: The MiniGrid improvement (90.5%→98% success) is modest, and the MountainCar results show negative performance at moderate β values, indicating sensitivity to hyperparameters—ironically, the very problem the paper claims to solve.

Score: 4.5

---

## sSbEEHNEsL

- GT: Accept (Poster) (avg 8.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents USR 2.0, an improvement over the Unified Speech Recognition framework that addresses two key limitations: the computational cost of autoregressive pseudo-labelling and the vulnerability to out-of-distribution errors from decoupled supervision. The core technical contribution is CTC-driven teacher forcing, where greedily decoded CTC outputs condition the decoder to generate attention pseudo-labels in a single forward pass, eliminating the AR bottleneck during training.

The paper's strengths are substantial. The motivation is clear and well-grounded: the authors correctly identify that AR decoding during pseudo-labelling is both slow and error-prone on OOD inputs. The key insight—that sequence-level coherence is unnecessary when teacher and student share the same conditioning—is genuinely clever and non-obvious. The empirical evaluation is comprehensive, demonstrating consistent improvements across in-distribution benchmarks (LRS2, LRS3), out-of-distribution scenarios (noise robustness, length generalization, WildVSR), and computational efficiency (~2× faster training). The ablations are thorough, examining the contribution of each component and the effect of key hyperparameters. The writing is clear and the paper is well-organized.

However, there are notable weaknesses. The contribution is incremental in nature—the individual components (CTC, teacher forcing, scheduled sampling) are well-established, and the novelty lies primarily in their combination for this specific application. The method is specifically designed for iterative self-training; it does not apply to inference-time decoding or offline pseudo-labelling. While the authors discuss this limitation, it constrains the broader applicability of the technique. Some hyperparameters (the 0.5 mixing probability, loss weights) appear somewhat arbitrary, though ablations show reasonable robustness. The comparison to non-autoregressive transformers or other efficient decoding approaches would have strengthened the positioning.

Overall, this is a solid contribution with clear practical value for semi-supervised speech recognition. The efficiency gains and robustness improvements are meaningful, and the work advances the state-of-the-art on established benchmarks with a unified model across ASR, VSR, and AVSR tasks.

Score: 7.5

---

## wkVsKDnl4s

- GT: Reject (avg 1.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper presents HighClass, a metagenomic classification framework that combines variable-length token indexing (from QA-Token), quality-aware scoring, and learned sparsification to achieve 4.2× speedup and 68% memory reduction over MetaTrinity while maintaining competitive accuracy (85.1% F1, within 1.5% of SOTA).

**Strengths:** The paper provides a comprehensive theoretical framework including generalization bounds via Rademacher complexity, concentration inequalities under α-mixing for dependent tokens, and consistency proofs for maximum likelihood classification. The empirical evaluation is thorough with proper ablation studies isolating component contributions (vocabulary: +6.8pp, quality weighting: +1.9pp). The 4.2× speedup and 68% memory reduction are practically meaningful for large-scale metagenomic analysis.

**Weaknesses:** The primary concern is limited novelty—the paper primarily integrates three existing techniques (QA-Token vocabulary from Gollwitzer et al. 2025, MetaTrinity architecture from Gollwitzer et al. 2023, and gradient-based sparsification from Alser et al. 2024), all from the same research group. The core innovation—replacing alignment with hash-based token lookup—is well-established in k-mer methods. The theoretical contributions, while sound, apply standard results (Rademacher complexity bounds, α-mixing concentration) rather than introducing new techniques. The 1.5% accuracy gap versus MetaTrinity is meaningful for clinical applications. Comparisons are limited, primarily against methods from the same group, and some baselines (Kraken2 at 70% F1) appear outdated. The theoretical bounds are loose (excess risk of 0.021) and provide limited practical insight. The α-mixing assumption for genomic tokens may not hold well given complex dependencies from conserved regions and horizontal gene transfer.

Score: 5.0

---

## 78WdKlYSeO

- GT: Reject (avg 1.3)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes LaaC (LLM as a Classifier), a framework that reformulates classification tasks for decoder-based LLMs by introducing atomic special tokens for each class label, enabling single-token output and thus significantly reducing inference latency compared to standard prompting approaches.

**Strengths:**
The paper addresses a practical and important problem—LLMs are inefficient classifiers due to multi-token generation—and proposes an elegant solution. The technique of adding reserved control tokens and randomizing label-token mappings during training to prevent memorization is clever. Empirical results are strong: the fine-tuned Gemma-3-27B achieves 62.7% on MIntRec 2.0, outperforming GPT-4o and purported GPT-5, while running orders of magnitude faster. The latency analysis is thorough, including P50/P95 metrics and batch scaling studies. The framework works across text and multimodal inputs, and the zero-shot generalization experiments demonstrate the model isn't simply memorizing token-to-label mappings.

**Weaknesses:**
The core technical contribution is relatively incremental—the approach of adding special tokens and fine-tuning with LoRA is straightforward engineering rather than a novel methodological advance. Several comparisons are problematic: the GPT-4o/GPT-5 comparisons include network latency for API models and compare against untuned versions, while LaaC models are fine-tuned on in-domain data—this is an apples-to-oranges comparison. The paper mentions GPT-5 results, but GPT-5 has not been publicly released, raising concerns about reproducibility. Encoder baselines (BERT/RoBERTa) are evaluated with only 16-shot fine-tuning, an artificially weak setting. Text benchmark evaluations use only 200 randomly sampled examples rather than full test sets. The approach's scalability to label spaces beyond 500 classes is unaddressed, and there's limited analysis of calibration or out-of-distribution handling.

**Overall:**
This is a solid systems-level contribution with practical value, but the novelty is limited and some experimental comparisons favor the proposed method unfairly. The approach is useful but the paper overreaches on claims of fundamental advancement.

Score: 5.5

---

## wyCnT4BUsT

- GT: Reject (avg 4.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces DeepCritic, a two-stage framework for training LLMs to produce more deliberate and effective critiques of mathematical reasoning. The key insight is that existing LLM critics provide shallow, superficial critiques that merely echo the original reasoning rather than critically evaluating it.

**Strengths:**

The paper makes several valuable contributions. First, the iterative critique generation pipeline—where an initial critique is followed by an in-depth meta-critique that either verifies from alternative perspectives or challenges the initial critique—is innovative and well-motivated. This encourages the model to develop reflection and self-correction behaviors. Second, the empirical results are impressive: DeepCritic-7B significantly outperforms GPT-4o and same-sized DeepSeek-R1-Distill models on error identification benchmarks, and demonstrates strong performance in critique-based refinement where it helps generators correct errors. Third, the paper demonstrates weak-to-strong generalization potential, showing that a 7B critique model can effectively supervise 72B generators. Fourth, the self-improvement experiments (Section 4.4) show the method works without a stronger teacher, though with reduced performance. Fifth, the paper is thorough in its evaluation, including ablation studies, majority voting experiments, and extension to subjective domains.

**Weaknesses:**

Several limitations temper my enthusiasm. First, the seed data generation relies heavily on Qwen2.5-72B-Instruct; while self-improvement is demonstrated, the gap is substantial (38.2 vs 54.1 F1). Second, the best RL results depend on PRM800K human-labeled data; auto-annotated results (Numina) are notably weaker (64.4 vs 69.1), raising scalability concerns. Third, the data pipeline is complex—multiple prompting rounds, filtering based on ground-truth labels, and careful curation—which may limit reproducibility. Fourth, while Qwen2.5-Math-PRM-7B outperforms DeepCritic on error identification (72.9 vs 69.1 average F1), the paper dismisses this based on weaker refinement assistance, but this comparison could be more rigorous. Fifth, Appendix P reveals hacking vulnerabilities when ground-truth labels are provided to the in-depth generator, suggesting robustness issues.

Overall, this is a well-executed contribution addressing an important problem in scalable oversight, with strong empirical support and practical applications. The limitations are acknowledged and represent reasonable trade-offs rather than fundamental flaws.

Score: 7.5

---

## B6HnApgkP3

- GT: Reject (avg 2.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a post-hoc adaptation of the Feature Selection Layer (FSL) for interpreting pre-trained neural networks on tabular data. The core idea is straightforward: freeze the weights of a pre-trained model and train a lightweight layer that learns feature relevance weights by backpropagating only through this new layer.

**Strengths:**
The paper addresses a practical problem—extending embedded feature selection methods to already-deployed models. The evaluation is comprehensive, covering synthetic and real-world datasets with multiple evaluation dimensions: predictive performance, feature selection accuracy (PIFS/PSFI), visual metrics (weighted t-SNE, silhouette scores), and stability metrics. The comparison against established post-hoc methods (Integrated Gradients, DeepLIFT, Gradient SHAP, Feature Ablation) provides useful context. Testing on high-dimensional, low-sample-size (HDLSS) microarray datasets demonstrates practical relevance.

**Weaknesses:**
However, the contribution is limited. The method itself is essentially "train an FSL layer while keeping the rest of the network frozen"—a straightforward adaptation requiring minimal algorithmic innovation. More critically, the empirical results are mixed: post-hoc FSL underperforms the baseline on SynthA (Table 3), shows inferior stability metrics across datasets (Tables 1, 4, 9), and struggles with multi-class classification as acknowledged in the limitations. The visualization-based evaluation via weighted t-SNE and silhouette scores, while creative, is unconventional for evaluating feature attribution quality. The paper lacks comparison to simpler baselines (e.g., mutual information-based selection), and provides no ablation studies on design choices. The results do not demonstrate clear superiority over existing post-hoc interpretability methods—post-hoc FSL is comparable in some metrics but worse in stability.

Score: 4.5

---

## ZBaPU5FL0Z

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces OP-LoRA, a method that overparameterizes LoRA adapters during training by using a small MLP to predict the low-rank matrices, then discarding the MLP at inference time. The key insight is that this provides optimization benefits without inference overhead.

**Strengths:**
The idea is elegant and well-motivated. The theoretical analysis connects to prior work on implicit acceleration through overparameterization, deriving how OP-LoRA creates adaptive learning rates and line-search-like behavior. The empirical results are strong: OP-LoRA substantially outperforms standard LoRA on image generation (CMMD improvements of 10-15 points), and achieves consistent gains on VQA and commonsense reasoning tasks. Critically, the method is efficient—training time is only ~15% longer than LoRA, much faster than competing optimizers like LoRA-Pro (14x slower), with zero inference overhead. The extension to other adapter variants (DoRA, VeRA) demonstrates generality.

**Weaknesses:**
The main practical concern is training memory overhead (69GB vs 44GB for LoRA), which could be prohibitive for larger models or constrained settings. The MLP width introduces a new hyperparameter requiring tuning—the ablation shows inverted U-shape behavior. The scale testing stops at LLaMA 7B; behavior at larger scales is unknown. The theoretical analysis is primarily linear-case with ReLU extension in appendix; the connection between theory and empirical gains could be tighter.

**Overall:**
This is a solid, well-executed contribution to PEFT methods. The train-time overparameterization concept is novel for this setting, results are comprehensive and convincing across domains, and the efficiency benefits are practical. While not field-advancing, it represents a meaningful improvement over existing approaches.

Score: 7.5

---

## 05hNleYOcG

- GT: Accept (Poster) (avg 2.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents PLAGUE, a modular framework for generating multi-turn jailbreak attacks against LLMs. The framework decomposes attacks into three phases—Planner, Primer, and Finisher—and incorporates a lifelong-learning memory mechanism for strategy retrieval.

**Strengths:**
The empirical results are impressive, achieving 81.4% ASR on OpenAI's o3 and 67.3% on Claude Opus 4.1—significant improvements over existing baselines (30-40% relative gains). The three-phase decomposition provides a clean conceptual framework for understanding multi-turn attacks, and the plug-and-play architecture is well-designed, demonstrated through integration of existing methods (GOAT, Crescendo, ActorBreaker) as interchangeable components. The ablation studies are thorough, showing contributions from backtracking, reflection, planning, and strategy retrieval. Efficiency analysis comparing LLM call budgets adds practical value.

**Weaknesses:**
The novelty is limited—most components (reflection, retrieval-based memory, iterative prompting) are borrowed from existing work, with the main contribution being their combination. The "lifelong learning" claim is overstated; this is retrieval-augmented in-context learning, not persistent learning. The paper lacks deeper analysis of *why* the attacks succeed—what specific vulnerabilities are being exploited across different models. Most problematically, while the ethics statement claims defensive utility, no defenses are proposed or analyzed. The evaluation also has concerns: using DeepSeek-R1 as the attacker may not generalize, and some baselines have limited configuration details. The paper would benefit from discussing limitations of the attack and potential mitigations.

**Overall:**
A solid, well-executed attack paper with strong empirical results and a useful modular framework, but limited in conceptual novelty and lacking the defensive analysis that would strengthen its contribution to the safety literature.

Score: 7.0

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper provides a thorough and rigorous re-analysis of the min-p sampling paper (Nguyen et al., 2024), an ICLR 2025 Oral presentation. The authors systematically examine all four lines of evidence from the original paper—human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption claims—and find significant methodological issues that invalidate the original conclusions. The key findings include: (1) one-third of human evaluation data was omitted without justification, and proper statistical analysis shows no consistent advantage for min-p; (2) extensive hyperparameter sweeps on GSM8K demonstrate that min-p's claimed superiority disappears when controlling for hyperparameter tuning volume; (3) the LLM-as-a-Judge evaluations suffered from inconsistent reporting; and (4) community adoption claims were retracted.

The paper's strengths are substantial. It introduces a novel "Best-of-N" methodology for fair comparison of methods requiring different hyperparameter search volumes—a genuinely useful contribution for empirical ML research. The statistical analysis is careful and correct, properly addressing multiple comparison corrections. The empirical work is extensive (~6000 A100-hours), covering 9 models across base and instruct variants with systematic hyperparameter sweeps. The extraction of general lessons for rigorous science—fair hyperparameter comparison, proper statistical testing, data transparency, and scrutinizing qualitative claims—provides actionable guidance for the community.

The primary limitation is that this is fundamentally a critique paper rather than novel method development. However, given the crisis of rigor in ML research that the introduction documents, this type of careful re-examination is precisely what the field needs. The methodology introduced for controlling hyperparameter tuning volume is itself a valuable contribution that extends beyond this case study. Some lessons (e.g., correcting for multiple comparisons) are not novel to statistics but bear repeating given their frequent violation in ML publications.

Score: 7.5

---

## VaFqN9dnCa

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

This paper proposes "topological closure" as a foundational principle for cognitive computation, arguing that the homological identity ∂² = 0 provides a mechanism for memory and reasoning that transcends the limitations of Turing-style enumeration. The authors develop Memory-Amortized Inference (MAI) as an operational framework and connect the theory to biological mechanisms like oscillatory phase coding.

**Strengths:**
The paper presents an original and conceptually ambitious framework. The mathematical machinery (homology, chain complexes, ∂² = 0) is correctly applied, and the biological connections to theta-gamma coupling and STDP show thoughtful integration across disciplines. The time-reversal duality between RL and MAI is a creative insight, and the writing is generally clear despite the abstract subject matter.

**Weaknesses:**
The most significant issue is the complete absence of empirical validation. This is purely theoretical speculation with no simulations, experiments, or computational demonstrations that the proposed framework actually works. MAI is described abstractly but never implemented. The claims that this approach "transcends" Turing computation remain unsubstantiated—showing ∂² = 0 applies to computation doesn't prove it avoids Gödelian limitations.

Additionally, the biological claims are speculative rather than demonstrated; the paper doesn't show that neural systems implement homological operations as described. Key concepts like "context" and "content" variables lack rigorous operationalization. There's no comparison to existing topological data analysis methods, attention mechanisms, or modern deep learning approaches that might implement similar principles.

The paper reads more like an extended position paper or theoretical proposal than a complete research contribution. For acceptance at a top ML venue, substantial computational implementation and empirical demonstration would be essential.

**Score: 4.0**

---

## p15DIvdOV9

- GT: Withdrawn (treated as Reject) (avg 2.8)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper presents a method for image editing using rectified flow models, addressing the challenging problem of multi-object editing in complex scenes. The authors propose three main technical contributions: (1) Starting Point Optimization (SPO) that adaptively determines when to begin editing based on image structural complexity, (2) Trajectory Optimization combining semantic-aware vector orthogonalization and frequency-adaptive scaling, and (3) a selective attention injection strategy leveraging MM-DiT's frequency-aware properties.

The paper has several notable strengths. First, the empirical results are convincing—the method consistently outperforms existing baselines across multiple metrics on PIE-Bench, PIE-Bench++, and OIR benchmarks, with particularly strong performance in structure preservation (lowest Structure Distance, highest SSIM). Second, the user study demonstrates clear preference (50.5% and 54.8% for single and multi-object edits respectively) over competing methods. Third, the phase analysis (Chaos/Layout/Refinement) provides interesting insights into the rectified flow editing process, and the frequency-domain analysis of MM-DiT layers is a valuable contribution. The ablation studies are comprehensive and demonstrate each component's necessity.

However, the paper has notable weaknesses. The frequency-adaptive scaling strategy, while empirically effective, lacks strong theoretical motivation—the energy-based coefficient formulation appears somewhat ad-hoc. The method is complex with multiple interacting components, making it difficult to attribute improvements cleanly. The writing contains several clarity issues and grammatical errors that occasionally obscure the technical presentation. Additionally, the paper acknowledges using "GPT-5" for language polishing, which appears to be an error (GPT-5 does not exist), raising concerns about accuracy. Finally, while comparisons with training-free methods are extensive, there is no comparison with fine-tuning based approaches or recent commercial editing systems.

Score: 7.0

---

## jymuXl8GYi

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Direct Group Preference Optimization (DGPO), a method for post-training diffusion models that combines the group-wise preference learning from GRPO with the direct optimization approach of DPO. The key insight is that GRPO's effectiveness stems primarily from leveraging relative preference information within groups, rather than its policy-gradient formulation. This allows DGPO to use efficient ODE-based sampling and avoid trajectory-level training while still benefiting from fine-grained group comparisons.

**Strengths:**
The paper presents a compelling solution to a real efficiency problem in diffusion model post-training. The claimed 20-30× speedup over Flow-GRPO is substantial and well-supported by experiments. The technical derivation is sound, particularly the advantage-based weighting scheme that elegantly cancels the intractable partition function. Empirically, DGPO achieves impressive results on GenEval (0.97 vs 0.95 for Flow-GRPO and 0.63 baseline) while maintaining strong out-of-domain metrics. The timestep clip strategy is a practical addition that addresses real artifacts from few-step sampling.

**Weaknesses:**
The core idea—combining group-wise preferences from GRPO with direct optimization from DPO—is relatively straightforward once one understands both methods. The primary technical novelty is the weighting scheme to eliminate the partition function, which relies on a simple property of zero-mean advantages. The baseline comparisons could be more comprehensive; for instance, comparing against Diffusion-DPO with enumerated pairwise comparisons within groups would strengthen the analysis. The offline setting comparison against online DPO (rather than offline DPO) is not entirely fair. While hyperparameters like β=100 and group size 24 are specified, systematic ablations on these would be valuable.

**Overall:**
This is a solid contribution with meaningful practical impact. The efficiency gains are substantial, the empirical results are strong, and the method is well-motivated. While the technical novelty is incremental, the paper successfully identifies and addresses an important limitation of prior work. The writing is clear and the experiments are reasonably comprehensive.

Score: 7.5

---

## 5Y4wvlp923

- GT: Reject (avg 0.5)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses an interesting problem in contrastive time-series representation learning: semantic imbalance where dominant components (e.g., trend) suppress weaker ones (e.g., seasonality). The authors propose Semantic Disentanglement Error (SDE) as a diagnostic metric and an asymmetric weighting mechanism to rebalance contrastive losses.

**Strengths:**
The problem identification is valuable—the analysis in Table 1 clearly demonstrates that TS2Vec struggles to balance semantic components, with systematic asymmetry depending on amplitude ratios. The proposed SDE metric provides a principled way to quantify this imbalance. The idea of dynamically reweighting losses based on component recoverability is intuitive and the method plugs into existing frameworks like CoST without architectural changes.

**Weaknesses:**
However, the paper has significant issues. First, the novelty is limited—the core contribution is essentially adding asymmetric weights to CoST's existing disentangled objectives, which is a relatively simple extension. Second, the experimental validation is incomplete: the claimed "consistent gains" are not strongly supported by results. Looking at the actual table, improvements over vanilla CoST are modest and inconsistent—CoST+APW wins on some horizons but loses on others. Third, no ablation studies are provided for the proposed components (MLP fusion layer, sensitivity to hyperparameters γ, γ'). Fourth, the paper has significant presentation issues: SDE is defined twice (Sections 3.2 and 4.3 with inconsistent naming), Table 3 is missing from the main numbering (results appear in an unnumbered table), and the failed SDE regularization experiment (Table 2) consumes space without yielding actionable insights. Fifth, the baseline comparison is limited to TS2Vec, TNC, and CoST—more recent methods should be included. Finally, the MLP fusion layer is introduced without clear justification or analysis of its role.

**Overall Quality:**
While the problem diagnosis is valuable, the proposed solution is incremental, experimental results are mixed, and evaluation lacks rigor (no ablations, no hyperparameter sensitivity). The presentation needs substantial revision.

Score: 4.0

---

## 6Y9NP1qhoM

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper addresses an important and timely problem: misinformation injection in LLM-based Multi-Agent Systems (MAS). The authors make two main contributions: (1) MISINFOTASK, a novel dataset of 108 realistic tasks designed to evaluate MAS robustness against misinformation, and (2) ARGUS, a two-stage training-free defense framework combining adaptive localization (using graph topology and semantic relevance) with goal-aware persuasive rectification.

The paper has notable strengths. First, it identifies a meaningful gap in existing MAS security research—most prior work focuses on overtly malicious content, while misinformation is more subtle and challenging to detect. Second, the proposed framework is well-motivated, leveraging both spatial (topological importance) and temporal (reasoning-based rectification) dimensions. Third, the experimental results demonstrate consistent improvements across multiple LLM backbones (GPT-4o, DeepSeek-V3, Gemini) and attack types (Prompt Injection, RAG Poisoning, Tool Injection), with approximately 28% reduction in misinformation toxicity and 10% improvement in task success rates. The ablation studies and topology experiments provide useful insights into component contributions and generalizability.

However, the paper has several weaknesses. The baseline comparisons are limited to Self-Check and G-Safeguard; including more recent defense methods would strengthen the evaluation. The weighting scheme for combining localization scores (α=0.2, β=0.2, γ=0.6) appears somewhat arbitrary despite the ablation study. The evaluation relies entirely on LLM-based scoring (GPT-4o as judge), which introduces potential biases—human evaluation would significantly strengthen the claims. Additionally, the experiments are limited to 3-6 agent systems, and scalability to larger MAS remains unexplored. The "training-free" claim, while technically accurate, somewhat oversells the practical benefits since the approach still requires substantial LLM inference overhead.

Score: 7.0

---

## VbTLgEUocp

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

**Assessment:**

This paper presents Calgacus, a steganographic protocol for hiding one text within another text of the same length using Large Language Models. The method is straightforward: encode a message by recording the rank of each token in the LLM's predicted distribution, then generate a stegotext by selecting tokens at those ranks from a prompted LLM. Decoding reverses this process.

The work has several strengths. First, the core idea is genuinely novel—achieving "full capacity" steganography where the hidden message and cover text have identical length is a meaningful contribution to generative steganography. Second, the method is elegantly simple and practical, working effectively on consumer hardware with modest 8B-parameter models. Third, the paper thoughtfully explores the philosophical and safety implications of its technique, including the provocative "Shibbolethian Theatre" scenario for covertly deploying unaligned LLMs. The analysis of why stegotexts are less probable than originals (Figure 5's "low entropy token choices" insight) provides real technical understanding.

However, the paper has significant weaknesses. The evaluation is quite limited—comparing log-probabilities to real text distributions provides useful baselines, but lacks human evaluation to assess whether stegotexts are actually indistinguishable from authentic writing. More critically, there's no systematic comparison to existing LLM steganography methods (Meteor, Wu et al., Zamir) on key metrics like detectability, capacity, or quality. The security analysis is also thin; the claim about O(d^|k|) brute-force difficulty is trivial, and the deniability discussion, while interesting, lacks formal treatment. Finally, the philosophical discussions about authorship and hallucination, while engaging, somewhat substitute for deeper technical contributions.

**Score: 5.0**

---

## c5mdo1hWrs

- GT: Accept (Poster) (avg 7.3)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper presents Flash Sparse Attention (FSA), an alternative kernel implementation for Native Sparse Attention (NSA) that addresses inefficiencies when the number of query heads per GQA group is small—a common configuration in modern LLMs like Llama-3. The key insight is to exchange the loop order: FSA iterates over KV blocks in the outer loop and query tokens in the inner loop, eliminating padding overhead that NSA incurs with small GQA group sizes.

**Strengths:**
The paper identifies a real and practical limitation of NSA—its kernel is optimized for large GQA group sizes but many popular LLMs use small groups (GQA=4 in Llama-3, for example). The proposed solution is technically sound: by reversing the loop order, FSA can batch multiple query tokens instead of batching query heads, naturally meeting GPU tensor core requirements without padding. The evaluation is comprehensive, spanning kernel-level benchmarks, end-to-end training/inference measurements, multiple GPU architectures (H20, H200), multiple model families (Llama3, Qwen), and various sequence lengths including ultra-long contexts (128K, 256K). The reported speedups (up to 3.5× kernel, 1.25× end-to-end training) are meaningful for practitioners. The paper also provides correctness validation through training loss comparisons and ablation studies on the kernel optimizations.

**Weaknesses:**
The primary concern is incremental novelty. Loop reordering for GPU efficiency is a well-established optimization technique, and applying it to NSA—while useful—is not fundamentally innovative. The paper compares only against NSA and full attention, omitting other sparse attention implementations (e.g., H2O, StreamingLLM, Quest) that could provide broader context. The memory overhead from additional buffers (up to ~12GB for 256K sequences) is acknowledged but not deeply analyzed in terms of when it becomes problematic. The benefits diminish substantially at GQA=8, where NSA's original approach becomes competitive, limiting the scope of improvement.

**Overall:**
This is a solid systems paper with practical impact, but the contribution is primarily optimization of an existing algorithm rather than novel research. The work would fit better in a systems-focused venue. For ICLR, while well-executed, the limited algorithmic contribution makes it borderline.

Score: 5.0

---

## xURArpl40L

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important and underexplored question in conformal prediction: whether the prediction set size (PSS), which is commonly interpreted as a measure of predictive uncertainty, is actually calibrated with respect to prediction accuracy. The authors propose a formal definition of CP calibration, derive a target calibration function based on theoretical analysis, and develop a CP-aware calibration (CPAC) algorithm to improve calibration.

**Strengths:**
The paper identifies a meaningful gap in the CP literature—while coverage guarantees are well-studied, the alignment between set size and actual prediction reliability has received limited attention. The multinomial sampling framework for connecting PSS to accuracy is sensible, and the proposed calibration target function f(k) = 1/k^τ has some theoretical justification from Dirichlet/logistic-normal distribution analysis. The experimental evaluation is comprehensive across three datasets (CIFAR100, ImageNet, Topic Classification), multiple model architectures (ResNet, ViT, GPT-2), and various perturbation types. Results consistently show that CPAC reduces calibration error (both standard and uniform CP-ECE) in most settings.

**Weaknesses:**
Several concerns arise: (1) The multinomial sampling approach for measuring accuracy feels somewhat arbitrary—why not simply use top-1 accuracy or coverage-based metrics? The temperature parameter adds complexity without clear justification. (2) The theoretical result (Theorem 4.2) relies on strong distributional assumptions (Dirichlet/logistic-normal) that may not hold in practice, limiting its practical guidance. (3) The empirical improvements are modest in many cases, and CPAC sometimes increases PSS (reducing efficiency) while improving calibration. (4) The algorithm introduces several hyperparameters (τ, λ, learning rate, optimization rounds) that require tuning, reducing practical applicability. (5) The paper lacks comparison to alternative calibration approaches for CP beyond basic Platt scaling. (6) The practical significance of PSS calibration remains unclear—how does it improve downstream decision-making compared to standard coverage guarantees?

Overall, this is a solid contribution that identifies and addresses an important problem, though the methodological advances are incremental and the empirical gains are moderate rather than transformative.

Score: 7.0

---

## TjF9WLcu8o

- GT: Reject (avg 0.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Contrastive-Online-Meta (COM), a framework combining contrastive pre-training with online meta-learning for adapting instruction-tuned CodeLLMs. The core idea of separating task-invariant representation learning (via contrastive objectives) from fast adaptation (via online meta-learning) is conceptually reasonable. The authors claim this helps address catastrophic forgetting and enables real-time adaptation during deployment.

**Strengths:**
The problem motivation is relevant—adapting CodeLLMs to dynamic environments while preserving programming knowledge is important for practical deployment. The modular architecture that keeps the base model frozen while learning adaptable components (instruction encoder and meta-learner) is a sensible design choice. The framework attempts to unify multiple learning paradigms (contrastive learning, meta-learning, continual learning) in service of this goal.

**Weaknesses:**
The paper has significant issues that substantially undermine its contribution. First, the writing quality is poor throughout, with numerous grammatical errors, unclear sentences, and nonsensical phrases (e.g., "programming England's instructions," "scope for improvementCivil War"). This severely impacts readability and suggests insufficient care in preparation.

Second, the novelty is limited. Combining contrastive learning with meta-learning is not new—similar ideas appear in prior work for other domains (Qin et al., 2023 is cited). The online meta-learning formulation in Equation 5 is essentially gradient descent with L2 regularization, which is standard practice. The "spectral normalization" and memory buffer additions are minor architectural tweaks.

Third, experimental details are lacking. The actual results tables/figures are not shown in the text, making it impossible to evaluate claims like "12-18% improvement" or "3-5× fewer updates." The StreamCode benchmark appears to be self-constructed without adequate description.

Fourth, the claim of "online meta-learning during inference" is technically suspect. Meta-learning typically requires an inner-loop optimization across multiple tasks during training—not single-step gradient updates during inference, which is simply online fine-tuning with regularization.

Score: 3.5

---

## bhPaXhWVKG

- GT: Reject (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper introduces MermaidFlow, a framework for agentic workflow generation that uses Mermaid (a graph markup language) as a declarative intermediate representation with built-in static verification. The key idea is to constrain the search space through type-safe, structurally valid graph representations, enabling evolutionary programming operators that preserve correctness by construction.

**Strengths:**
1. **Clear motivation and problem framing**: The paper correctly identifies that existing code-based workflow generation methods (like AFlow, ADAS) produce fragile, unexecutable plans due to implicit structure and runtime-only validity. This is a real and important problem.

2. **Principled approach**: The use of declarative graphs with explicit typing and static verification is a clean solution. The formal definitions of the search space and transformation operators, including Lemma 1 on closure properties, provide solid theoretical grounding.

3. **Consistent empirical improvements**: The method outperforms all baselines across four benchmarks (GSM8K, MATH, HumanEval, MBPP), with meaningful gains particularly on MATH (+2.61% over AFlow) where baseline performance leaves more room for improvement.

4. **Strong experimental analysis**: The paper includes useful ablations on optimization efficiency (showing ~50% token savings vs AFlow), success rates (>90% valid code generation vs ~50% for AFlow), and scalability with different optimization LLMs.

**Weaknesses:**
1. **Incremental novelty over prior work**: The core idea of evolutionary workflow search heavily builds on AFlow. The main difference is using Mermaid instead of Python code. While valuable, this is a representation change rather than a fundamentally new approach.

2. **Limited expressiveness**: Mermaid cannot naturally express conditionals or loops, which the authors acknowledge. This restricts the types of workflows representable. The workaround (adding more node types) is mentioned but not fully developed.

3. **Still relies on LLM for translation**: The Mermaid-to-Python translation uses an LLM, which could reintroduce errors that the representation was designed to avoid. A rule-based translator would strengthen the claim of safety guarantees.

4. **Modest absolute improvements**: While consistent, gains are 1-3% on most benchmarks. The practical significance of these improvements, while real, is not transformative.

5. **Verification is relatively simple**: The "safety constraints" enforced are basic structural checks (connectivity, type matching). Semantic correctness of workflows is not verified.

Overall, this is a solid contribution to automated workflow generation. The idea of using structured, verifiable intermediate representations addresses a genuine limitation of prior work, and the execution is thorough. While the novelty over AFlow is incremental, the paper provides sufficient innovation and validation to merit publication.

**Score: 7.0**

---

## 5SQo2E9Ujj

- GT: Withdrawn (treated as Reject) (avg 0.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a conceptual reframing of curriculum learning in goal-conditioned RL as "selective data acquisition," arguing that curricula should be understood as mechanisms that reshape state-goal distributions rather than merely exploration heuristics. The authors support this view with experiments in GridWorld using UVFAs, comparing uniform goal sampling against edge-weighted curriculum sampling.

**Strengths**: The paper offers a potentially useful perspective connecting curriculum learning to broader questions about training data distribution in RL. The motivation linking this work to open-ended learning is clear and timely. The experimental design is clean, isolating distributional effects by keeping architecture and training fixed while varying only the sampling strategy. The authors are commendably honest about limitations, acknowledging the modest effect sizes and restricted experimental scope.

**Weaknesses**: The empirical contribution is severely limited. The experiments are conducted only in simple GridWorld environments with manually specified curricula, lacking comparison to standard GCRL baselines (e.g., HER, automated curriculum methods). The reported improvements are tiny—overall success improving by only +0.02 and edge-goal success by +0.08—with large standard deviations that call into question statistical significance. The hand-designed "edge-weighted" curriculum requires prior knowledge of goal difficulty, undermining the claim of a general principle. Crucially, the paper claims to show "reduced approximation error" but provides no quantitative analysis of this. Only 3 seeds with 1000 episodes each is insufficient for robust conclusions.

The core claim that curriculum is "selective data acquisition" rather than exploration is largely semantic—existing curriculum learning literature already implicitly recognizes that curricula change what data agents see. The paper does not demonstrate substantial new insights or practical advances from this reframing. The connection to open-ended learning remains speculative without experiments in continual or expanding task settings.

**Overall**: This submission reads like preliminary work or a workshop paper rather than a complete ICLR contribution. The conceptual framing has merit but is insufficiently supported by evidence, and the empirical scope is too narrow to draw meaningful conclusions about the proposed perspective.

Score: 4.0

---

## CSrGFB070m

- GT: Withdrawn (treated as Reject) (avg 2.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper presents IndicSuperTokenizer (IST), a tokenizer for Indic multilingual LLMs that combines two-stage subword-superword learning with language-specific pre-tokenization. The work addresses an important practical problem—existing tokenizers are highly inefficient for Indic languages, with fertility scores up to 10x worse than for English—and demonstrates significant efficiency improvements.

**Strengths:**
The evaluation is comprehensive, spanning 22 Indic languages, English, and code, with comparisons against 9 baseline tokenizers including specialized Indic ones (Sutra, Sarvam). The paper introduces multiple intrinsic metrics (fertility, NSL, bytes-per-token, Rényi efficiency) and validates downstream task performance and inference latency. The 39.5% fertility improvement over LLaMA-4 and 44% throughput gain are practically meaningful for deployment. The ablation studies are thorough, examining vocabulary size, training data, transition points, and vocabulary allocation strategies.

**Weaknesses:**
The core methodological contribution is incremental—the two-stage curriculum approach is borrowed directly from SuperBPE (Liu et al., 2025), with the main adaptation being the use of LLaMA-4 regex instead of GPT-2 rules. While this choice is effective, it's not novel. More concerningly, downstream task improvements are marginal (English: 0.279 vs 0.279; Indic: 0.388 vs 0.394), suggesting efficiency gains don't translate to better representations. The vocabulary size confound (IST uses 200K tokens vs. 68K for Sarvam, 128K for LLaMA-3.2) complicates fair comparison on fertility. The continual pretraining experiment (Table 11) shows comparable rather than superior results, with slight drops on some benchmarks. While efficiency matters, tokenizers should ideally improve or at least maintain task performance—the paper doesn't clearly demonstrate this.

**Overall Quality:**
This is a competent systems paper with practical value for Indic NLP practitioners. However, the limited methodological novelty (adapting an existing approach) and marginal downstream improvements make it a borderline contribution. The efficiency gains are valuable, but without clearer evidence of maintained or improved representation quality, the case for acceptance is moderate rather than strong.

Score: 5.5

---

## ACn1rbjlFB

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper investigates the vulnerability of tensor ring (TR) decomposition to adversarial perturbations, proposing a novel asymmetric adversarial attack (AdaTR) and a computationally efficient variant (FAG-AdaTR). The work makes several meaningful contributions: it correctly identifies that the symmetric min-max formulation inherited from adversarial NMF (ATTR) fails to effectively attack tensor decomposition—ironically improving reconstruction under small perturbations—and provides theoretical justification (Theorem 1). The proposed asymmetric max-min objective directly maximizes reconstruction error via a bilevel optimization, which is fundamentally better suited for this attack setting.

The strengths include rigorous theoretical analysis with convergence guarantees for both methods, comprehensive experimental evaluation across decomposition, completion, and recommendation tasks, and practical efficiency improvements through FAG-AdaTR. The extension to other decomposition models (CP, Tucker, TT) and robustness experiments under different TR-ranks and compression formats add useful breadth. The experiments demonstrate substantial degradation in reconstruction quality (RSE increases of 2-4x) even with small perturbation budgets.

However, there are notable limitations. The computational overhead remains significant—Table 4 shows AdaTR requires 8.57GB memory versus 999MB for FAG-AdaTR on videos, and runtime is 1.5x slower. The transferability claims to defense methods (TRPCA-TNN, TRNNM, etc.) could be better substantiated with targeted experiments. While Theorem 1's theoretical condition is elegant, the paper doesn't empirically verify when this condition holds in practice. Finally, the discussion of potential defenses is limited; understanding how to protect against such attacks would strengthen the practical impact.

Overall, this is a solid contribution that opens an important direction in tensor robustness. The asymmetric formulation is well-motivated, the theoretical foundation is sound, and the experimental validation is thorough. While not groundbreaking, it represents clear progress in understanding adversarial vulnerabilities beyond neural networks.

Score: 7.5

---

## CbK7lYbmv8

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper presents Latent Reasoning Tuning (LRT), a framework that replaces explicit chain-of-thought reasoning with learned latent representations generated by an auxiliary network. The work addresses the "overthinking" problem in reasoning LLMs, where models generate unnecessarily long reasoning trajectories.

**Strengths:**
The paper's most compelling contribution is the empirical analysis in Section 2, demonstrating that reasoning trajectories contain substantial redundancy—models maintain high accuracy even when conditioned on trajectories with 50% of tokens/steps randomly removed. This finding motivates the approach well and provides a solid foundation. The proposed framework is modular and non-intrusive: the base LLM parameters remain frozen while a lightweight reasoning network (initialized from Qwen3-Embedding-0.6B) generates latent vectors that condition answer generation. This design enables seamless switching between latent and explicit reasoning modes, offering practical flexibility. The two-stage training (SFT followed by RL with GRPO) is well-motivated, and ablations confirm both stages contribute meaningfully. The experimental evaluation is reasonably comprehensive, covering in-domain (MATH-500, GSM8K, AMC) and out-of-domain (LSAT, GPQA) benchmarks with multiple baselines.

**Weaknesses:**
Several concerns limit my enthusiasm. First, the core idea of latent reasoning is not novel—the paper acknowledges Coconut (Hao et al., 2024) and related work, but the distinctions claimed (parallel generation vs. iterative refinement) are architectural rather than conceptual. The paper lacks direct experimental comparison with these latent reasoning baselines, making it difficult to assess relative merits. Second, the primary experiments focus on 1.5B-4B models, with limited evaluation on larger scales; scalability claims are not strongly substantiated. Third, the budget-forcing evaluation setup artificially constrains baselines, potentially favoring the proposed method. Fourth, the "latent reasoning tokens" remain uninterpretable, and the geometric analysis in Appendix D.4, while interesting, doesn't clarify what reasoning patterns are actually captured. Finally, the hyperparameter sensitivity (the 256-token "sweet spot") requires further investigation—why does performance degrade at 512 tokens?

**Overall:**
This is a solid contribution addressing an important efficiency problem with reasonable methodology and empirical support. However, the incremental nature of the approach over existing latent reasoning work and gaps in experimental comparison prevent it from being a stronger submission.

Score: 5.5

---

## VP204Aa0gH

- GT: Withdrawn (treated as Reject) (avg 2.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

**Strengths:**

OmniCode makes a valuable contribution by expanding the scope of LLM coding benchmarks beyond the narrow focus of existing work. While HumanEval targets single-function synthesis and SWE-Bench focuses on bug fixing, OmniCode introduces a multi-faceted evaluation spanning four distinct software engineering tasks: bug fixing, test generation, code review response, and style fixing. The multi-language coverage (Python, Java, C++) is also a welcome expansion beyond Python-centric benchmarks.

The synthetic task generation methodology is creative and principled. Using failed agent attempts and perturbed correct patches to create bad patches for test generation evaluation is particularly clever—requiring tests to pass on gold patches AND fail on bad patches is more robust than prior approaches. The manual validation of base instances adds quality assurance often missing in automated benchmark construction.

The empirical evaluation is thorough, comparing multiple models (Gemini 2.5 Flash, DeepSeek-V3.1, GPT-5-mini, Qwen3-32B) and two agent frameworks (SWE-Agent vs. Aider). The findings are interesting: the strong correlation between bug-fixing and review-response performance (0.921) versus weak correlation with style-fixing (0.512) provides meaningful insight into agent capabilities. The analysis of patch complexity and failure modes adds depth beyond simple accuracy numbers.

**Weaknesses:**

The dataset is relatively small at 494 base instances, and task-specific subsets are even smaller (77 Java instances and 44 C++ instances for test generation). This limits statistical confidence in some results. More concerningly, the synthetic task generation relies on LLMs (Gemini 2.0 Flash) to create reviews and perturb patches—potential systematic biases or errors in these synthetic annotations are not analyzed.

While combining task types is valuable, individual task types have been explored before (SWT-Bench for test generation, various style linters for style checking). The paper could strengthen its contribution by more directly comparing against these specialized benchmarks. Additionally, the base instances drawn from SWE-Bench and Multi-SWE-Bench may still face data leakage concerns despite claims about synthetic augmentation.

Some presentation issues detract from clarity: the style-fixing metric formula appears incomplete/garbled, and certain tables have formatting inconsistencies. The analysis sections occasionally become dense without clear actionable takeaways.

**Overall Quality:**

This is a solid benchmark paper addressing a genuine gap in the evaluation landscape. The methodology is generally sound, the evaluation comprehensive, and the findings informative. While not groundbreaking, it provides useful infrastructure for future research on more capable software engineering agents.

Score: 7.5

---

## P0GOk5wslg

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper proposes "Speculative Actions," a framework for accelerating AI agents by predicting and pre-executing likely future actions in parallel while waiting for authoritative responses. The approach adapts speculative execution ideas from computer architecture and speculative decoding from LLM inference to the agentic systems domain.

## Assessment

**Strengths:** The paper identifies a timely and important problem—agent latency—as AI systems increasingly operate in interactive environments. The abstraction of treating all agent-environment interactions as API calls provides a clean, general framework applicable across diverse domains. The theoretical analysis (Proposition 1, Theorems 3-6) offers rigorous characterization of speedup bounds and cost-latency tradeoffs, with closed-form expressions for both breadth-focused and depth-focused speculation strategies. The evaluation across four distinct environments—chess, e-commerce, web search, and OS tuning—demonstrates breadth of applicability, with each domain stress-testing different latency bottlenecks. The 19.5-20% speedup and up to 55% prediction accuracy, while not dramatic, validate the core concept. The cost-latency analysis provides principled guidance for tuning speculative breadth.

**Weaknesses:** The empirical results are modest—20% speedup and 40-55% accuracy are useful but far from the theoretical maximum of 50% speedup. More concerning is the lack of empirical comparison to the closest related work (Hua et al. 2024, Guan et al. 2025); the paper discusses these methods but doesn't benchmark against them. The use of "GPT-5" for model names is problematic for reproducibility—this model doesn't exist publicly, and no explanation is given for what this actually refers to. The evaluation focuses exclusively on single-step breadth speculation; multi-step and depth-focused strategies, while theoretically analyzed, lack empirical validation. The e-commerce accuracy metric (predicting API calls) is somewhat weak since parameter matching isn't strictly evaluated. Finally, the OS tuning "lossy" setting, while interesting, diverges significantly from the core lossless speculation framework.

**Overall:** The paper makes a clear contribution with its general framework and theoretical analysis. The problem framing is novel and important. However, the empirical validation has gaps (missing comparison to related work, limited coverage of theoretical extensions), and the reproducibility concern regarding model naming is notable. The contribution is solid but not exceptional, with execution gaps that should ideally be addressed.

Score: 7.0

---

## b8TlYh6PN6

- GT: Accept (Oral) (avg 8.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment**

This paper addresses the fundamental problem of characterizing distributional equivalence in linear non-Gaussian causal models with latent variables and cycles—a gap that has hindered the development of general, structural-assumption-free causal discovery methods. The authors provide a complete theoretical framework including: (1) a graphical criterion for equivalence based on "children bases," (2) a novel "edge rank" tool dual to the classical path rank, and (3) a transformational characterization enabling traversal of equivalence classes via admissible cycle reversals and edge additions/deletions.

The key strength is the conceptual innovation: the edge rank concept and its duality to path ranks (Theorem 1) provides a local, combinatorial alternative to the global path rank conditions that have dominated this space. This enables the decomposition in Theorem 2, where checking equivalence reduces to examining singletons rather than all subsets—crucially improving tractability. The theoretical development is rigorous, building on transversal matroid theory, and the analogies to classical CPDAG results (Theorem 4) are elegant.

The paper's practical significance lies in enabling the first structural-assumption-free discovery algorithm (glvLiNG). The evaluation is comprehensive: equivalence class size quantification, runtime comparisons showing the constraint-based approach outperforms MILP baselines significantly, oracle-input benchmarking revealing existing methods' fragility under misspecification, and simulations demonstrating robustness to graph density. The real-world stock market application shows interpretable results.

**Weaknesses:** The algorithm's practical deployment depends on OICA, which is computationally challenging. The authors acknowledge this limitation but it does constrain immediate applicability. The faithfulness assumption is standard but unavoidable. Sample complexity requirements (10^4-10^5 samples) are significant, though partially attributable to OICA. The comparison with methods operating under different structural assumptions (LaHiCaSl, PO-LiNGAM) raises questions about fairness when graphs violate their assumptions—though the authors' goal precisely is to avoid such assumptions.

Overall, this is a solid theoretical contribution that fills an important gap. The edge rank tool may find applications beyond this setting, and the complete equivalence characterization enables principled discovery without restrictive structural assumptions.

**Score: 7.5**

---

## y3oHMcoItR

- GT: Accept (Oral) (avg 7.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces RealPDEBench, a benchmark for scientific machine learning that pairs real-world experimental measurements with numerical simulations. The work addresses a significant gap in the field—most existing benchmarks rely exclusively on simulated data, limiting the evaluation and development of models for real-world deployment.

The strengths are substantial: (1) The contribution is timely and addresses a genuine need for real-world data in scientific ML. (2) The benchmark is comprehensive, offering 5 datasets spanning fluid dynamics and combustion, 736+ trajectories, 3 well-motivated task settings (simulated training, real-world training, and pretrain-finetune), 9 evaluation metrics, and 10 baselines including foundation models. (3) The experimental methodology is rigorous, with clear descriptions of PIV measurements, the circulating water tunnel setup, and combustion diagnostics. (4) The empirical findings are meaningful—demonstrating the sim-to-real gap quantitatively and showing that simulated pretraining consistently benefits real-world performance. (5) Reproducibility is strong with a modular codebase, HDF5 data format, and thorough documentation.

However, several limitations temper the contribution: (1) The physical domain coverage is narrow—all 5 datasets are fluid-related, missing important PDE domains like elasticity or electromagnetics. (2) Real-world modalities are limited (only velocity fields for fluids, light intensity for combustion), constraining evaluation scope. (3) While the sim-to-real gap is demonstrated, its physical sources aren't deeply analyzed beyond acknowledging numerical errors and measurement noise. (4) The baseline comparisons largely confirm known architectural trade-offs without surprising insights. (5) The surrogate model approach for combustion feels ad-hoc and introduces additional complexity.

Overall, this is a well-executed contribution that fills an important gap and will serve the community. The limitations are notable but don't undermine the core value of providing paired real-world/simulated data for benchmarking.

Score: 7.5

---

## do4hqhMBiu

- GT: Reject (avg 0.0)
- Predicted: N/A (3.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a diffusion-based image-to-image augmentation workflow for human facial images that aims to preserve acupoint landmarks while introducing variations in lighting, background, and other environmental factors. The work builds on the AcuSim dataset and uses Stable Diffusion 1.5 with IP-Adapter and IC-Light modules.

**Strengths:** The paper addresses a relevant problem in medical/biometric image augmentation—preserving anatomical landmarks while increasing dataset diversity. The dual evaluation approach (CNN-based task performance and facial landmark drift analysis) provides useful validation. The automated workflow design with controller program and custom nodes shows practical engineering effort.

**Weaknesses:** The paper has significant issues that undermine its quality. Most critically, the abstract is incomplete, ending abruptly at "99.99" with no conclusion. The Results section contains verbatim duplicate paragraphs—both the CNN evaluation and facial-landmark evaluation descriptions are pasted twice. Figure 2 has a placeholder caption "Enter Caption" that was never filled in. Many citations are incomplete, showing "Author(s) omitted" and "(Add full citation)" placeholders. Beyond these editorial issues, the technical novelty is limited: the work essentially applies existing diffusion tools (SD 1.5, IP-Adapter, IC-Light) with parameter tuning, without introducing new algorithmic contributions. The evaluation lacks proper baselines—no comparison to traditional augmentation methods or other diffusion-based approaches. The CNN evaluation shows the augmented dataset works, but doesn't demonstrate it provides value over the original dataset.

**Overall:** This appears to be an incomplete draft submitted prematurely. While the application domain is interesting, the paper has fundamental presentation issues and lacks sufficient technical contribution for a top venue. The incomplete abstract, duplicate text, placeholder captions, and missing citations suggest the paper was not ready for review.

Score: 3.5

---

## 83F6YF4Hz6

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Review Assessment

## Summary

This paper proposes SPELL, a self-play reinforcement learning framework for long-context reasoning that enables LLMs to evolve without human annotations. A single model cyclically assumes three roles: questioner (generating questions from documents), responder (solving them), and verifier (evaluating semantic equivalence). Key innovations include a history memory mechanism for curriculum learning, Gaussian-shaped rewards calibrated to the model's competence frontier, and role-specific dynamic sampling for balanced training.

## Strengths

**Strong empirical contribution.** The paper demonstrates consistent improvements across 12 diverse models (4B-32B, dense and MoE architectures) on 6 benchmarks at both 16K and 100K contexts. Notably, SPELL-trained base models outperform instruction-tuned counterparts that use extensive human annotations, highlighting impressive data efficiency.

**Addresses an important gap.** Extending RLVR to long-context reasoning is genuinely challenging due to the lack of programmatically verifiable rewards and the unreliability of human annotations for complex long-document tasks. The semantic verification mechanism via self-consistency is a sensible solution.

**Comprehensive analysis.** The paper includes thorough ablations (questioner updates, history memory, verifier components, reward mapping functions), comparison with AZR and R-Zero reward schemes, and analysis of training dynamics. The test-time scaling analysis showing improved pass@k curves is particularly valuable.

**Adaptive curriculum.** The history memory mechanism that progressively increases difficulty by conditioning on previously solved problems is elegant—the analysis showing difficulty stagnating without updates (Figure 4) is convincing.

## Weaknesses

**Incremental novelty.** The core self-play concept with proposer/solver roles is well-established (AZR, R-Zero). The three-role extension is sensible but not fundamentally transformative. The verifier role essentially provides semantic matching for non-verifiable outputs, which is straightforward conceptually.

**Heuristic design choices.** Many hyperparameters (σ=0.5/3, L=3, G=8) are hand-tuned. While ablations justify the final choices, there's limited theoretical grounding for why these specific values work well.

**Limited training context.** Training is restricted to 16K tokens due to computational constraints. Though 100K evaluation shows generalization, the framework's scalability to truly long contexts (128K+) remains untested.

**Missing self-play baselines.** The paper compares against RLVR and alignment methods (LongPO, SoLoPO) but doesn't adapt existing self-play methods (AZR, R-Zero) to long-context settings for fair comparison. This makes it hard to assess whether the gains come from domain adaptation or the specific three-role design.

## Overall

This is a solid, well-executed paper addressing a meaningful problem. While the technical novelty is incremental, the empirical contribution is substantial, the methodology is sound, and the framework design is thoughtful. The work demonstrates that self-play can effectively improve long-context reasoning without human supervision—a valuable finding for the field.

Score: 7.5

---

## Ic65R0Tbc1

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Preference-Based Reward Repair (PBRR), a method for iteratively correcting human-specified proxy reward functions using trajectory preference comparisons. The core insight is pragmatic: practitioners often have an imperfect but informative proxy reward function and would benefit from efficiently repairing it rather than learning a reward function from scratch via expensive preference collection.

**Strengths:**

The paper makes a timely and practical contribution to reward alignment. The key methodological innovations—(1) the exploration strategy comparing proxy-reward policies against a reference policy, and (2) the novel preference-learning objective with regularization terms that prevent unnecessary corrections—are well-motivated and technically sound. The L+ term correctly preserves correct rankings from the proxy, while the L- term focuses corrections on down-weighting undesired behaviors.

Empirical evaluation is thorough across four diverse reward-hacking benchmarks (pandemic mitigation, glucose monitoring, traffic control, AI safety gridworld). PBRR consistently outperforms baselines including standard RLHF and concurrent residual reward methods. Ablations convincingly demonstrate both components are necessary. The theoretical analysis provides regret bounds matching prior strategic RLHF work in tabular settings.

**Weaknesses:**

1. **Theoretical-practical gap**: The regret bounds assume linear rewards in tabular settings, while experiments use neural networks in high-dimensional continuous domains—this disconnect limits the theory's relevance.

2. **Simulated preferences**: All preference labels are generated from ground-truth rewards via Boltzmann distribution, not collected from humans. While standard practice, this doesn't capture real human noise and biases.

3. **Optimism assumption**: The loss function design assumes proxy rewards are "optimistic or aligned," but this assumption is violated in the Glucose environment where PBRR still succeeds. The theoretical justification for the loss is thus somewhat undermined.

4. **Simplified exploration in practice**: The algorithm sets C1=0 in experiments, essentially ignoring the uncertainty-based exploration fallback. This simplification isn't fully analyzed.

5. **Concurrent work**: Cao et al. (2025) proposed similar residual reward learning, though this paper distinguishes itself via the novel objective and exploration strategy.

Overall, this is a solid paper addressing an important practical problem with meaningful empirical contributions. The weaknesses are notable but don't undermine the core contribution.

**Score: 7.5**

---

## n3u7PK2kyd

- GT: Withdrawn (treated as Reject) (avg 2.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper makes two main contributions to topological representation analysis: (1) completing the theoretical framework of Representation Topology Divergence (RTD) by introducing Symmetric RTD (SRTD), and (2) proposing Normalized Topological Similarity (NTS), a normalized similarity measure based on merge-order comparison.

**Strengths:**
The theoretical analysis of SRTD is rigorous and satisfying. The paper proves that RTD, Max-RTD, and SRTD form a coherent mathematical family through an exact sequence relationship (Theorem 3.3), elegantly explaining the empirical complementarity between directional RTD measures. The introduction of NTS is a practical and useful contribution—by using Spearman correlation on merge times from MSTs, it produces a naturally normalized [0,1] score that addresses a genuine limitation of divergence-based measures. The empirical evaluation is comprehensive, covering synthetic experiments, CNN layer analysis, and LLM representation comparison. Particularly compelling is the finding that NTS detects the functional shift at pooling layers that CKA misses, while still providing interpretable graded similarity patterns.

**Weaknesses:**
The SRTD contribution is somewhat incremental—essentially computing divergence between union and intersection structures rather than averaging directional computations. The computational savings (one vs. two MST/PH computations) are modest. More critically, NTS is fundamentally limited to H₀ (connected component) topology, discarding higher-order topological features (cycles, voids) that RTD captures. This is acknowledged but not deeply analyzed. The LLM experiments, while interesting, would benefit from more quantitative rigor—the claims about CKA saturation versus NTS discriminative power rely heavily on visual inspection rather than systematic metrics. Finally, NTS's non-differentiability limits its utility for optimization tasks.

**Overall:**
This is a solid, well-written paper that advances the toolbox for representation analysis. The theoretical contributions are mathematically sound, and NTS addresses a real practical need. However, the limitations around H₀-only topology and some empirical claims being under-substantiated prevent it from being exceptional.

Score: 7.0

---

## QGXVZ0OPLy

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper proposes DualPrompt, a method for training-free zero-shot multi-label classification using CLIP. The key insight is that co-occurrence information can help multi-label recognition but also causes object hallucination. The authors propose using both discriminative prompts (containing only the target label) and correlative prompts (containing target and co-occurring labels), combining their predictions to leverage co-occurrence while mitigating hallucination.

**Strengths:**
1. **Clear motivation**: The paper correctly identifies that CLIP struggles with multi-label tasks due to lack of co-occurrence exploitation, supported by empirical analysis showing gaps between CLIP predictions and true co-occurrence probabilities.
2. **Interesting finding**: The dual nature of co-occurrence—helping recognition but causing hallucination—is a valuable insight, well-illustrated through the causal framework and empirical analysis.
3. **Solid experimental results**: The method achieves state-of-the-art performance on MS-COCO (70.0 mAP with TagCLIP) and VG-256 (40.7 mAP), with consistent improvements across different backbones.
4. **Simple and practical**: The approach is computationally efficient and can work with any CLIP backbone, including both CNN and ViT variants.

**Weaknesses:**
1. **Questionable theoretical derivation**: The transition from Eq. (1) (subtraction form) to Eq. (2) (addition form) is poorly justified. The claim that Eq. (1) "hardly works" lacks empirical evidence, and the derivation in Appendix A makes strong, unverified assumptions. Setting λ=1 without ablation or theoretical justification is concerning.
2. **Misleading "training-free" claim**: The best results require estimating co-occurrence from 1-2% training data. When using ChatGPT for co-occurrence (truly training-free), results are notably worse. The paper should be clearer about this trade-off.
3. **Limited novelty in the method**: The final formula T_k(x) = p(y_k=1|x, P_k^c) + p(y_k=1|x, P_k^d) is simply adding two probability scores. The elaborate causal framework seems disconnected from this simple operation.
4. **Missing comparisons**: No comparison with other zero-shot multi-label approaches beyond CLIP variants. Recent work on vision-language models for multi-label classification is not discussed.
5. **Hyperparameter sensitivity unexplored**: The number of co-occurring labels (l=2) is set arbitrarily without systematic study.

Overall, while the paper presents an interesting observation and achieves good empirical results, the theoretical contribution is weak, and the method's novelty is limited. The disconnect between the causal framework and the simple final implementation undermines the paper's depth.

Score: 5.5

---

