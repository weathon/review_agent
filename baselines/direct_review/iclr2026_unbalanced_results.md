# Direct-Scoring Baseline Results

Model: z-ai/glm-5

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper identifies and addresses a novel safety vulnerability specific to Masked Diffusion Language Models (MDLMs). The authors demonstrate that affirmative tokens appearing at intermediate denoising steps can steer subsequent generation toward harmful responses—a phenomenon they term the "priming vulnerability." They propose Recovery Alignment (RA), a training method that teaches models to recover safe responses from contaminated intermediate states.

**Strengths:**
The paper makes several notable contributions. First, the vulnerability identification is timely and insightful—MDLMs' parallel, non-causal generation mechanism creates fundamentally different safety risks than ARMs, and this work provides the first systematic analysis of this specific vulnerability. Second, the theoretical derivation (Theorem 4.1) provides a tractable lower bound enabling efficient gradient-based attacks without Monte Carlo sampling, demonstrating that even realistic attackers without direct denoising intervention can exploit this vulnerability. Third, the proposed solution (RA) is well-motivated: standard alignment trains models only from fully masked sequences, leaving them vulnerable when harmful tokens appear mid-generation; RA directly addresses this by training on contaminated intermediate states. Fourth, the empirical evaluation is thorough—testing across three MDLMs, seven attack types, two datasets, multiple evaluation metrics, and 11 capability benchmarks. The results convincingly show RA significantly reduces attack success rates (e.g., anchoring attack ASR drops from 88.7% to 8.3% for t_inter=16 on LLaDA Instruct) while preserving general capabilities.

**Weaknesses:**
Several limitations exist. The mitigation is incomplete—at high intervention steps (t_inter=32), ASR remains around 50%, indicating residual vulnerability when many harmful tokens are injected. The theoretical analysis relies on a monotonicity assumption that, while empirically validated, may not hold universally for all responses. The training overhead is non-trivial (~16 hours on 4 GPUs per model), and the paper lacks comparison to ARM defenses adapted for MDLMs beyond MOSA. Additionally, the paper doesn't discuss potential adaptive attacks that could specifically target RA-trained models.

Overall, this is a solid contribution addressing an important and underexplored problem in DLM safety. The analysis is rigorous, the solution is principled, and the empirical validation is comprehensive.

**Score: 7.5**

---

## 1E4Bltg6Xb

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a Dynamics Feature Representation (DFR) framework for reinforcement learning-based dynamic path planning in urban road networks. The core idea is to address the fundamental trade-off between using complete but computationally expensive global dynamics versus efficient but potentially incomplete local dynamics. The framework employs a two-stage hierarchical refinement: (1) policy attention mechanism that identifies top-k shortest paths to extract a task-relevant subgraph, and (2) n-hop neighborhood method that further distills node-specific local features.

**Strengths:**
The paper addresses a genuine and important challenge in RL-based path planning—state representation design. The proposed approach is intuitive and well-motivated, with clear theoretical grounding in Predictive State Representations. The experimental evaluation on realistic urban networks (Nanjing, Beijing Chaoyang, Shanghai Pudong) is comprehensive, demonstrating consistent improvements across multiple RL algorithms (DQN, PPO, GCN+DQN). The ablation study on parameters k and n provides useful insights, and the reported planning time reductions (46-86%) show practical value. The writing is clear and the problem formulation is rigorous.

**Weaknesses:**
The primary concern is incremental novelty. The policy attention mechanism essentially pre-computes shortest paths on a static graph—this is a straightforward application of existing path-finding algorithms rather than a novel contribution. The n-hop neighborhood method is similarly standard. While the integration is sensible, neither component represents a methodological advance. The framework introduces two hyperparameters (k, n) requiring manual tuning, which the authors acknowledge as a limitation but offer no principled solution. The policy attention is distance-based and ignores temporal dynamics during subgraph selection, potentially missing critical dynamic patterns. The theoretical claims about preserving Markov properties lack formal proof. Additionally, comparing DFR against "All Dynamics" is somewhat unfair—comparisons with other state abstraction or dimensionality reduction techniques would strengthen the evaluation. The success rates around 88-92% warrant deeper analysis of failure cases.

Overall, this is a competent application paper with practical contributions but limited methodological novelty. For a venue like ICLR that emphasizes algorithmic innovation, it falls short of the bar for acceptance.

**Score: 5.5**

---

## WwDNiisZQm

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Content-Aware Mamba for Learned Image Compression (CMIC), introducing two key mechanisms: Content-Adaptive Token Permutation (CTP) and Global-Prior Prompting (GPP). The work addresses genuine limitations of applying Mamba-style SSMs to image compression—specifically, that raster-scan ordering fails to group semantically similar tokens, and strict causality limits global context access.

**Strengths:** The paper presents strong empirical results, achieving state-of-the-art BD-rate improvements of -15.91%, -21.34%, and -17.58% over VTM-21.0 on Kodak, Tecnick, and CLIC datasets respectively. The improvements over existing Mamba-based compression methods (MambaVC, MambaIC) are substantial. The core ideas are well-motivated: clustering tokens by content similarity before scanning is intuitive for compression, and the prompting mechanism provides a lightweight alternative to multi-directional scanning (reducing memory by 78% versus MambaIC). The ablation studies are comprehensive, including ERF visualizations that demonstrate content-adaptive receptive fields. The clustering visualizations confirm that tokens with similar visual attributes are grouped together, validating the approach.

**Weaknesses:** The prompting mechanism draws significant inspiration from MambaIRv2, and while the paper provides comparisons in the appendix showing improvements over direct MambaIRv2 adaptation, this reduces novelty. The K-means clustering with 5 iterations per training step adds computational overhead (claimed at 5% of training time). The comparison with MambaIRv2 and detailed analysis of clustering differences should have been in the main text rather than appendix. Additionally, while the approach outperforms Transformers (FTIC) in RD performance, the paper could benefit from deeper analysis of why Mamba-based approaches offer advantages beyond computational efficiency.

Overall, this is a solid contribution that meaningfully advances learned image compression by making Mamba content-adaptive. The technical innovations are well-executed, and the empirical results are convincing.

Score: 7.5

---

## aiM6bRd6bG

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper introduces the problem of PPI candidate ranking—prioritizing novel protein interactions for experimental validation—and proposes an interpretability-guided framework using active embedding regions from D-SCRIPT and Topsy-Turvy models, combined with multi-source re-ranking.

**Strengths:** The problem formulation is practically motivated and distinct from standard PPI classification, addressing a genuine bottleneck in computational biology. The prospective evaluation using STRING v11→v12 is a strong design choice that tests genuine predictive capability. The empirical improvements are substantial, with Recall@10 improving from ~1% to ~26%, which has meaningful practical implications for experimental prioritization. The comprehensive analysis of re-ranking signals (structural plausibility, semantic features, LLM-based scores) provides useful insights about complementary evidence sources.

**Weaknesses:** The methodological novelty is incremental—the individual components (cosine similarity, pDockQ, LLM re-ranking) are not novel; the contribution lies in their assembly. The PiNUI benchmark results raise concerns: D-SCRIPT shows better Average Rank (86.50 vs 924.78), which the authors attribute to coverage differences, but this explanation feels incomplete for a benchmark designed to be less biased. STRING v12 incorporates AlphaFold predictions, creating potential circularity where "novel" interactions may already be computationally inferred. The interpretability framing is overblown—contact maps serve as feature extraction rather than providing genuine interpretability. Key design choices (top-10 re-ranking, max similarity aggregation, active region extraction) lack ablation or justification. Missing are comparisons to simpler baselines like sequence similarity or network proximity that would contextualize the improvements.

**Overall:** A solid contribution addressing a practical problem with meaningful empirical results, but limited in novelty and with evaluation concerns. The assembly of existing techniques is well-executed but doesn't constitute a major methodological advance.

Score: 5.5

---

## Pa6ak2B9jJ

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper presents AUTO-RT, a reinforcement learning framework for automated red-teaming of large language models. The work addresses an important problem in LLM safety: discovering diverse jailbreak strategies beyond fixed templates. The proposed method decomposes attack generation into strategy generation and rephrasing, with two technical contributions: Dynamic Strategy Pruning (DSP) for efficient exploration and Progressive Reward Tracking (PRT) for handling sparse rewards via downgrade models and the First Inverse Rate (FIR) metric.

**Strengths:**
- Clear motivation and problem formulation as a constrained MDP
- Reasonable technical approach with DSP and PRT addressing concrete challenges
- Comprehensive evaluation across 16 white-box and 2 black-box models with multiple metrics (effectiveness, diversity, defense generalization)
- Ablation studies demonstrate contribution of each component
- Practical applicability demonstrated in both white-box (fine-tuning) and black-box (ICL) settings

**Weaknesses:**
- Technical contributions are incremental—DSP is essentially early termination (a well-known technique), and PRT adapts existing reward shaping ideas
- The FIR metric for downgrade model selection lacks strong theoretical justification
- Missing comparison with AutoDAN-turbo and other recent methods; baselines (FS, IL, basic RL) are relatively weak
- Black-box evaluation is limited compared to extensive white-box testing
- Transferability results (Table 6) show strategies are somewhat model-specific, limiting generalization claims

The paper makes a solid contribution to automated red-teaming methodology with thorough experimentation, though the core techniques are more evolutionary than revolutionary. The work is suitable for publication but would benefit from stronger baseline comparisons and theoretical grounding for the FIR heuristic.

Score: 7.2

---

## 1EdAn5gMVv

- GT: Reject (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents SpatialBoost, a framework for enhancing vision encoders with 3D spatial knowledge through language-guided reasoning using LLMs. The approach converts dense 3D spatial information into linguistic expressions via a multi-turn Chain-of-Thought dataset construction pipeline, then fine-tunes vision encoders using dual-channel attention to preserve pre-trained knowledge.

**Strengths:**
The paper addresses an important limitation of current vision encoders—lack of 3D spatial awareness—using a novel training paradigm that doesn't require costly 3D training data. The hierarchical spatial reasoning dataset construction (pixel-level → object-level → scene-level) is well-motivated and the dual-channel attention mechanism provides an elegant solution to catastrophic forgetting. The empirical evaluation is comprehensive, spanning depth estimation, segmentation, 3D scene understanding, robot learning, classification, and retrieval. Consistent improvements across multiple SOTA vision encoders (DINOv2, DINOv3, SigLIPv2, OpenCLIP) demonstrate broad applicability. The ablation studies (Tables 6-8, 15-17) are thorough and provide insight into each component's contribution.

**Weaknesses:**
Several concerns temper enthusiasm. First, the DINOv3 baseline cites "arXiv:2508.10104" which suggests a future-dated paper—this raises reproducibility concerns about a major baseline. Second, the three-stage training pipeline is complex; it's unclear if this complexity is necessary or if simpler alternatives could achieve similar results. Third, the reliance on off-the-shelf depth/segmentation models for training data generation introduces potential bias propagation (Table 19 attempts to address this but with limited analysis). Fourth, the improvements on ImageNet classification (88.4%→90.2%) are surprising for a method focused on spatial understanding—the explanation involving scene captions seems insufficient. Fifth, computational costs are not discussed, which is important given the LLM-based training. Finally, while dual-channel attention is borrowed from CogVideo, the incremental contribution here is limited.

Score: 7.0

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Assessment

This paper presents a unified theoretical framework explaining why neural networks trained with gradient descent learn solutions of increasing complexity—a phenomenon known as simplicity bias. The authors show that this occurs through "saddle-to-saddle dynamics," where networks progress through a hierarchy of fixed points, each corresponding to solutions expressible with fewer units.

**Strengths:**
1. **Broad Architectural Coverage:** The framework genuinely spans multiple architectures—fully-connected linear and ReLU networks, convolutional networks, and linear self-attention. This breadth is impressive and rare in theory papers.

2. **Rigorous Core Results:** Theorems 1 and 3 on embedded fixed points and invariant manifolds are rigorously proven. These provide a solid mathematical foundation.

3. **Meaningful Distinction:** The paper identifies two distinct mechanisms—data-induced timescale separation (producing low-rank weights) versus initialization-induced timescale separation (producing sparse weights). This is a novel and useful categorization.

4. **Predictive and Validated:** The theory makes concrete predictions about network width, data distribution, and initialization effects, which are experimentally validated. The distinction between how width affects linear networks versus self-attention is particularly insightful.

5. **Clear Writing:** The paper is well-organized, with good figures illustrating the conceptual framework and supporting experiments.

**Weaknesses:**
1. **Heuristic Dynamics Arguments:** While the fixed point and invariant manifold results are rigorous, the core argument that dynamics *follows* saddle-to-saddle trajectories relies on timescale separation heuristics. The paper acknowledges this gap but doesn't fully close it.

2. **Limited Depth for Deep Networks:** The detailed dynamical analysis is restricted to two-layer networks. Deep networks are discussed but without rigorous treatment—the extension to depth remains conjectural.

3. **Incremental Aspects:** Cases (i) and (ii) of Theorem 1 were discovered by Fukumizu & Amari (2000). The linear case analysis parallels existing "silent alignment" results. While the synthesis and new cases add value, some components build heavily on prior work.

4. **Synthetic Experiments:** Validations use mostly synthetic or simple settings. Demonstrating the framework on realistic datasets or modern architectures would strengthen the claims.

5. **Practical Implications Underdeveloped:** The paper is strong theoretically but provides limited insight into how this understanding could improve model design or training.

Overall, this is a solid theoretical contribution that successfully synthesizes and extends ideas across multiple architectures. The mathematical framework is coherent and predictive, though the dynamical arguments could be more rigorous. It advances understanding of an important phenomenon in deep learning.

Score: 7.5

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces Thoughtbubbles, a novel transformer architecture that enables dynamic parallel computation allocation during pretraining using only language modeling loss. The core innovation is a "forking" mechanism that can duplicate or prune residual streams based on learned cumulative scores, allowing the model to allocate additional computation to tokens that need it.

**Strengths:**
The paper makes a genuine contribution to adaptive computation research. Unlike prior approaches (CoT, pause tokens) that require explicit tokens or post-hoc training, Thoughtbubbles learns to allocate computation during pretraining with standard LM loss only. The empirical results are compelling: the method consistently outperforms both parameter-matched baselines and computation-matched baselines across scales from 150M to 772M parameters. The result that a 319M model outperforms a 772M baseline on perplexity is particularly striking. The analysis showing that computation is allocated to high-entropy tokens provides useful interpretability, aligning with recent findings about "thinking" tokens. The writing is clear and the method is well-motivated relative to prior work.

**Weaknesses:**
Several limitations prevent a higher score. First, the scale is modest—772M parameters trained on only 2.5B tokens—which limits confidence in scaling behavior. Second, efficiency concerns are acknowledged but not rigorously addressed; the paper admits "raw wall-clock efficiency is relatively low" without quantitative comparison. Third, key ablations are missing: the paper doesn't justify why forking occurs at layers 3, 7, and 11 specifically, and there's no sensitivity analysis on the budget parameter κ. Fourth, the method underperforms on BLiMP compared to computation-matched baselines, raising questions about whether all linguistic phenomena benefit equally. Fifth, there's no evaluation on reasoning benchmarks like GSM8k, which would be most relevant to the stated goal of enabling inference-time compute scaling for multi-step problems. Finally, the autoregression requires a specific "dynamic budget" mitigation to avoid distribution shift, adding deployment complexity.

**Overall:**
This is a solid contribution with a genuinely novel architecture that works as claimed. The weaknesses (limited scale, efficiency, missing ablations) are acknowledged by the authors and represent reasonable avenues for future work rather than fatal flaws. The paper advances the field by demonstrating that adaptive parallel computation can be learned during pretraining without additional supervision.

Score: 7.0

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper investigates whether LLM-generated reference lists can be distinguished from human ones using citation graph structure and semantic embeddings. The authors construct paired citation graphs (ground truth vs. GPT-4o-generated) for 10,000 focal papers from SciSciNet, with well-designed random baselines, and systematically compare structure-only features against semantic embeddings using Random Forests and GNNs.

**Strengths:** The paper addresses a timely and important question about LLM-generated bibliographies. The methodology is rigorous: large-scale dataset (10k papers, ~275k references), multiple random baselines (field-matched, subfield-matched, temporally-constrained), progressive modeling from interpretable features to GNNs, and extensive robustness checks (Claude generator, OpenAI/SPECTER embeddings, cross-model generalization). The finding that structural features barely separate LLM from human (≈60% accuracy) while embeddings achieve ≈83-93% is clear and potentially impactful. The random vector ablation and PCA dimension experiments are thoughtful controls.

**Weaknesses:** The primary contribution is empirical rather than methodological—standard models (RF, GCN, GAT, GIN, GraphSAGE) are used without architectural novelty. More critically, the paper does not deeply analyze *what* semantic dimensions drive separability. The title mentions "semantically biased" but we don't learn whether recency, prestige, author overlap, or topic drift explains the fingerprint. Additionally, practical deployment is not addressed: the classification task assumes paired graphs, which doesn't match real-world detection scenarios where you'd need to classify a single reference list as LLM-generated or not. The 93% GNN accuracy raises the question of whether the task is too easy without understanding the underlying patterns.

The paper is a solid empirical contribution with clear findings and rigorous methodology, but limited by standard methods and insufficient depth in explaining the semantic differences discovered.

Score: 7.5

---

## crKJJ4Ej60

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes Copy-Paste, a generation paradigm for RAG systems that improves contextual faithfulness by directly embedding contextual fragments into responses, instantiated through CopyPasteLLM via two-stage preference training.

**Strengths:**
The paper presents a well-motivated and intuitive approach—the observation that copying degree inversely correlates with hallucinations is compelling and empirically validated on RAGTruth. The empirical results are impressive: CopyPasteLLM achieves 12-24% accuracy improvements on FaithEval counterfactual benchmarks while requiring only 365 training samples (50x more data-efficient than Context-DPO). The two-stage pipeline is thoughtfully designed, with three prompting variants offering different copying-fluency trade-offs. The interpretability analysis using Context-Parameter Copying Capturing provides useful mechanistic insights, showing that the method recalibrates parametric knowledge confidence rather than enhancing contextual representations. Evaluation across multiple datasets (FaithEval, ConFiQA, PubMedQA), models, and both counterfactual and non-counterfactual settings is comprehensive.

**Weaknesses:**
The core concept has limited novelty—copying/extracting from context shares DNA with extractive summarization, citation generation, and CoCoLex (a compared baseline). While the systematic framing and DPO pipeline are valuable, the fundamental idea isn't groundbreaking. The approach has clear limitations: CP-Order and CP-Link sacrifice fluency; the method depends heavily on context completeness and quality; and performance degrades on tasks requiring synthesis beyond explicit context (as shown in the "Negotiation" subset results). The explicit optimization of copying metrics (κ, δ) raises concerns about whether improvements reflect genuine faithfulness or metric gaming. Additionally, the method seems most suited for extractive tasks—the paper could benefit from clearer discussion of when copying helps versus when synthesis is needed.

**Overall:**
This is a solid contribution with practical value and empirical strength, but with notable limitations in scope and novelty that prevent it from being exceptional. The data efficiency and comprehensive evaluation are commendable, but the work is more engineering-focused than conceptual breakthrough.

Score: 7.2

---

## wgGJE6Z1B3

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a novel data-centric perspective on training draft models for speculative decoding (SD). The key insight is that tokens inducing flatter predictive distributions from the target model are disproportionately valuable for improving acceptance rates. The authors provide theoretical analysis using Gaussian distributions, propose a practical "flatness" metric (cosine similarity to uniform distribution), and develop SFDD for sample selection. Experiments on EAGLE-2 show 2× training speedup with 50% data while maintaining inference speedup within 4% of full training.

**Strengths:**
1. **Novel perspective**: Shifting from loss function design to data selection for SD is fresh and meaningful. The insight that not all tokens contribute equally to acceptance rate optimization is valuable.

2. **Solid theoretical motivation**: The Gaussian analysis in Section 3.2 provides intuition for why high-variance (flat) target distributions yield larger L1 reduction per training step. The connection to acceptance rate via L1 norm is properly grounded.

3. **Practical impact**: Achieving 2× training speedup with minimal performance degradation addresses a real pain point for practitioners deploying SD.

4. **Comprehensive empirical validation**: Multiple baselines (entropy, top-1 prob, margin, energy score, PPL), five downstream tasks, and ablations across retain ratios from 5-70% strengthen the claims.

5. **Clear explanation of mechanism**: Figure 2 effectively shows that high-flatness tokens exhibit larger changes in draft statistics and L1 discrepancy during training.

**Weaknesses:**
1. **Limited theoretical depth**: The Gaussian assumption is simplified, and the extension to Exponential/Half-normal distributions in Appendix F.3 shows the conclusion doesn't always hold—Figure 6 reveals non-monotonic behavior depending on draft scale. The gap between continuous theory and discrete distributions is notable.

2. **Metric simplicity**: Flatness is highly correlated with entropy (shown in Appendix F.2). The improvement over entropy is modest (Figure 2d), raising questions about whether the proposed metric captures something fundamentally different beyond uncertainty.

3. **Limited evaluation scope**: Only LLaMA3-8B-Instruct is tested in main experiments (Vicuna-7B-v1.3 in appendix). Testing across model scales (e.g., 7B vs 70B) and different SD frameworks would strengthen generalization claims.

4. **Unexplained phenomenon**: Table 2 shows 70% retention sometimes outperforming full data, suggesting filtered data can be better. The paper attributes this to removing "noisy" data but doesn't investigate this phenomenon deeply.

5. **Sample aggregation is ad-hoc**: Simple averaging of token-level scores may not optimally weigh token importance within a sample.

The work provides a meaningful practical contribution to SD training efficiency with solid experimental support. However, the theoretical contribution is limited by simplifying assumptions, and the proposed metric offers only marginal improvement over standard uncertainty measures.

Score: 7.0

---

## iDki7djO2K

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper proposes a unified theoretical framework for understanding forgetting in machine learning, defining it as a violation of predictive self-consistency rather than through task-specific metrics. The authors formalize learning as an interaction process between a learner and environment, introduce a k-step consistency condition, and derive an operational measure called "propensity to forget" Γ_k(t).

**Strengths:**
The paper addresses a genuine gap in the literature—existing forgetting metrics are either task-specific (e.g., continual learning measures) or conflate forgetting with backward transfer. The self-consistency perspective is conceptually elegant and genuinely novel. The mathematical formalism is rigorous and well-motivated, drawing productively from general RL frameworks. The thought experiments in the appendix (the clock, hash map, binary flipper, etc.) are excellent—they systematically validate that the definition aligns with intuitive notions of forgetting while handling edge cases. The empirical work spans multiple paradigms (regression, classification, generative modeling, RL, continual learning), and the finding that "optimal forgetting is not zero" is an interesting insight that challenges naive intuitions.

**Weaknesses:**
The proposed measure is computationally impractical for realistic neural networks—computing Γ_k(t) requires maintaining N particle copies and running k simulated updates per particle, which is prohibitively expensive. The paper does not seriously address scalability or provide approximations that would enable practical use. While the theory is solid, the practical utility remains unclear: the paper doesn't demonstrate how this measure could improve algorithms or diagnose learning problems. The empirical results mostly use toy problems (sinusoid regression, two-moons, CartPole), raising questions about whether insights transfer to realistic settings. Additionally, there's no comparison between Γ_k(t) and existing continual learning metrics to establish whether the proposed measure correlates with established notions.

**Overall Assessment:**
This is a solid theoretical contribution that identifies a real conceptual gap and proposes a principled solution. The formalism is correct, the thought experiments validate the definition, and the empirical characterization is reasonably comprehensive. However, the limited practical utility—due to computational impracticality and lack of algorithmic contributions—prevents a stronger rating. The work is more foundational than applied, which is acceptable but limits its immediate impact.

Score: 7.0

---

## QryPmx2MNh

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper addresses a novel and interesting problem: discovering learning-friendly orderings for chain-of-thought reasoning in Transformers. The key insight—that early training dynamics can identify orders where loss drops faster—is elegant and well-motivated by prior work on curriculum learning and easy-to-hard learning dynamics.

**Strengths:**
The paper makes several solid contributions. The problem formulation is genuinely novel—while chain-of-thought design has received significant attention, the ordering of reasoning steps has been largely overlooked. The proposed loss profiling method is intuitive and computationally efficient, requiring only short training runs. The hierarchical global-local search is clever for handling factorial search spaces. Empirically, the results are convincing: discovering optimal orders among billions of candidates, improving success rates from ~10% to near 100%, and successfully recovering the known reverse-digit order for multiplication. The designed order-sensitive tasks (RELU, SQUARE-19, INDEX) provide controlled testbeds where ground-truth "optimal" orders are known by construction.

**Weaknesses:**
However, the paper has notable limitations. First, the proposed tasks are artificially designed to have the forward order be optimal—this construction somewhat circularly validates the method, and it's unclear how well the approach would work on real tasks where optimal orders are unknown. Second, the scope is limited to synthetic arithmetic tasks; the paper doesn't demonstrate applicability to natural language reasoning or more complex problems. Third, for the INDEX task (the hardest), the method fails to fully recover the forward order (Table 2, d=4,8), suggesting limitations on harder problems. Fourth, while claiming efficiency, the method still requires multiple training runs (1-7 hours reported), and scaling beyond L=40 requires structured initialization rather than random search. Finally, there's limited theoretical analysis of *why* certain orders are learning-friendly beyond the non-injectivity argument.

The writing is clear and the paper is well-organized, though the contribution is somewhat narrow in scope. The connection to prior work on multiplication order recovery is a nice validation, but the broader impact remains unclear.

Score: 7.2

---

## ahpO7S1Ppi

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents Pctx, a personalized context-aware tokenizer for generative recommendation that addresses an important limitation of existing approaches: static, non-personalized tokenization that enforces universal item similarity. The key insight is that in autoregressive generation, semantic IDs with shared prefixes receive similar probabilities, so fixed item-to-token mappings cannot capture user-specific perspectives on item similarity.

**Strengths:**
- The paper identifies a meaningful and previously underexplored limitation in GR tokenization. The motivation is clear and well-articulated.
- The methodology is well-designed, combining context representation extraction via DuoRec, adaptive clustering, redundant ID merging, and multi-facet inference.
- Experiments are comprehensive across three Amazon datasets with comparisons to many baselines, detailed ablations, and supporting analyses.
- Consistent improvements are demonstrated (up to ~9% NDCG@10), and the case study provides intuition about how different user contexts lead to different tokenizations.

**Weaknesses:**
- The approach is complex, requiring pre-trained DuoRec, clustering per item, multi-ID management, and several merging/augmentation strategies. It's unclear whether simpler alternatives (e.g., directly conditioning the generator on user context) would achieve comparable results.
- Strong dependency on DuoRec as an auxiliary model adds system complexity. The ablation shows SASRec doesn't work as well, but the paper doesn't deeply analyze what properties make DuoRec suitable.
- No comparison with learnable tokenization methods or simpler personalization approaches (e.g., concatenating user embeddings with item features before RQ-VAE).
- Absolute metric values are low (NDCG@10 around 0.03-0.05), raising questions about practical significance.
- Computational overhead during inference (beam search across multiple IDs) is not discussed.

**Overall:**
This is a solid contribution with a novel problem formulation and thorough experimentation, but the approach complexity and lack of simpler baseline comparisons are notable concerns. The contribution is clear but the weaknesses prevent a stronger rating.

Score: 7.0

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents ELMUR, a transformer architecture with layer-local external memory designed for long-horizon decision-making under partial observability. The core innovation lies in combining (1) persistent layer-local memory embeddings, (2) bidirectional cross-attention between tokens and memory, and (3) an LRU-based memory management policy with convex blending for updates.

**Strengths:**
The empirical results are compelling. ELMUR achieves 100% success on T-Maze corridors up to 1 million steps with a context window of only 10, demonstrating effective retention far beyond the native attention span. On MIKASA-Robo, it achieves the best performance on 21 of 23 tasks with a ~70% improvement in aggregate success rate over the previous best baseline. The theoretical analysis provides clean derivations for exponential forgetting, half-life, and memory boundedness. The ablation study is thorough, examining memory size, blending factor, initialization scale, and component contributions. The memory probing analysis in the appendix provides interpretability insights, showing that task-relevant information is actually stored in the memory slots.

**Weaknesses:**
The novelty is somewhat incremental—the individual components (external memory, cross-attention access, LRU-style management) have precedents in prior work, though their combination is novel. The MIKASA-Robo benchmark is authored by the same team, which raises concerns about benchmark selection bias. The hyperparameter sensitivity shown in ablations raises practical deployment questions—how should one set M, λ, and σ for a new task? The comparison set could be stronger; recent state-space models like Mamba are mentioned but DMamba comparisons are limited. Parameter and compute overhead relative to baselines is not thoroughly analyzed.

**Overall:**
This is a solid, well-executed contribution addressing an important problem. The approach is sensible, the empirical results are strong, and the paper includes comprehensive analysis. While not groundbreaking, it represents meaningful progress in memory-augmented RL that should interest the robotics and RL communities.

Score: 7.5

---

## PFhrOUJZ5o

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents LAION-Comp, a large-scale dataset of 540K+ images with detailed scene graph annotations for compositional image generation. The work addresses a fundamental limitation in current T2I models: their struggle with complex multi-object scenes. Rather than proposing architectural fixes, the authors correctly identify the root cause as a data deficiency and provide a substantial resource to the community.

The strengths are significant. First, the dataset construction is thoughtful, using GPT-4o with carefully designed prompts to ensure comprehensive annotation of objects, attributes, and relations. Human verification demonstrates impressive accuracy (98.8% objects, 97.5% attributes, 95.7% relations). Second, the empirical validation is thorough—models trained on LAION-Comp consistently outperform those trained on COCO-Stuff and Visual Genome across multiple metrics and backbones (SDXL, SD3.5, FLUX). The ablation study showing that even 10% of LAION-Comp outperforms full VG training is compelling evidence of annotation quality. Third, the CompSGen benchmark fills a gap by focusing specifically on complex scenes, and the SG-based editing application demonstrates practical utility beyond generation.

However, some weaknesses temper the overall assessment. The reliance on GPT-4o for annotation introduces potential biases and hallucinations, though human verification shows acceptable error rates (~2-5%). The model architecture—a GNN-based scene graph encoder—is functional but not novel; the primary contribution lies in data rather than method. The evaluation metrics (SG-IoU, etc.) also rely on GPT-4o for extraction, creating circular dependencies. Finally, comparisons with recent layout-based or attention-based compositional methods are limited.

Overall, this is a valuable resource contribution with solid experimental validation. The dataset scale, quality, and benchmark design advance the field meaningfully, and the public release will benefit the community.

## Score: 7.5

---

## FlcMckO6x5

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents theoretical analysis and an optimization method for Separable Neural Networks (SepNNs). The authors make three main contributions: (1) proving universal approximation theorems for CP, TT, and Tucker SepNN variants; (2) deriving NTK regimes showing deterministic kernel convergence under infinite width/rank and random kernel under fixed rank; and (3) proposing SepPGD, an efficient preconditioned gradient descent method leveraging the separable structure.

**Strengths:**
- The NTK analysis for SepNNs is technically solid and novel. The derivations of both deterministic and random NTK limits provide genuine insight into training dynamics of these architectures.
- The SepPGD algorithm cleverly exploits the separable structure for O(nD) complexity versus O(n^D) for standard methods—a meaningful efficiency gain for grid-structured problems.
- The paper provides rigorous proofs in the appendix and connects well to established theoretical frameworks.
- Experiments across multiple domains (INRs, PINNs) demonstrate practical benefits of the proposed method.

**Weaknesses:**
- The universal approximation theorem, while correct and comprehensive across SepNN variants, follows somewhat straightforwardly from Stone-Weierstrass combined with standard MLP approximation results. The extension from the bivariate case (Cho et al.) is incremental.
- The fixed-rank regime analysis (Corollary 1) only establishes convergence to a random kernel without providing practical convergence guarantees or generalization bounds.
- The efficiency benefits are limited to grid-structured inputs; the non-grid experiments show SepPGD performs similarly to existing methods, as expected.
- Missing practical guidance on rank/width selection despite theoretical analysis focusing on asymptotic regimes.
- The paper assumes significant background knowledge and notation can be dense.

**Assessment:**
This is a solid contribution to understanding separable neural architectures. The NTK analysis is the strongest element—rigorous, novel, and providing actionable insights for spectral bias mitigation. SepPGD is well-motivated by theory and demonstrates practical gains. However, the approximation theory is somewhat expected, and the fixed-rank analysis doesn't yield strong practical guarantees. For ICLR, this represents a good but not exceptional paper that advances understanding of an increasingly important architecture class.

Score: 7.0

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

This paper proposes a compositional meta-learning framework that learns reusable computational modules and a gating network capturing their transition statistics. New tasks are solved via probabilistic inference (particle filtering) rather than parameter updates—a clean conceptual formulation that differs meaningfully from gradient-based meta-learning approaches.

**Strengths:** The core idea is elegant and well-motivated. By separating within-module dynamics from between-module statistics, the approach enables rapid inference of novel task solutions from single episodes, even under sparse feedback conditions. The probabilistic framing allows principled uncertainty handling through hypothesis tracking. The demonstrations across rule learning and motor learning domains show some generality, and control experiments (comparing to RNNs with/without task identity, ablations without gating) help isolate contributions. The ability to generalize to longer test tasks than seen during training is a nice demonstration of the learned transition statistics.

**Weaknesses:** The primary limitation is evaluation scope. Both task domains are synthetic and low-dimensional: shift operations in 6D space and simple trajectory primitives. While these provide ground truth for validation, they don't establish practical utility. No evaluation on standard benchmarks (Meta-World, mini-ImageNet) or realistic domains is provided. The fixed number of modules is acknowledged but unsolved. Training instability requiring careful initialization hints at optimization challenges. The particle filtering approach (250 particles) raises scalability concerns for higher-dimensional or longer-horizon problems. Architectural modifications needed for motor learning (resetting hidden states, removing input, module-specific readouts) suggest limited domain generality.

**Overall:** This is a conceptually solid contribution with a clean probabilistic formulation. However, the limited empirical validation on toy tasks prevents strong confidence in practical applicability. The approach shows promise but needs broader evaluation to demonstrate real-world relevance.

Score: 5.0

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces a benchmark for evaluating text-to-image models on generating images for taxonomy concepts (WordNet synsets), a novel task with practical implications for automating taxonomy enrichment. The work is timely and addresses an understudied intersection of structured knowledge representation and generative AI.

**Strengths:**
The problem formulation is genuinely novel—while text-to-image evaluation is well-studied, the specific challenge of generating images for taxonomy concepts (which range from concrete to abstract, and may require disambiguating synsets) is original. The benchmark includes multiple thoughtfully constructed datasets: easy concepts, randomly sampled WordNet nodes across different relation types, and LLM-predicted concepts. The 9 proposed metrics are comprehensive, combining preference-based evaluation (ELO scores with human/GPT-4), CLIP-based similarity metrics with theoretical grounding in KL divergence and mutual information, and standard metrics (FID, IS). The evaluation of 12 models reveals interesting findings—notably that rankings differ from standard T2I benchmarks, with Playground and FLUX performing well while retrieval-based approaches fail badly.

**Weaknesses:**
Several concerns limit the contribution. First, the CLIP-based metrics (Lemma, Hypernym, Cohyponym Similarity, Specificity) all rely on the same underlying CLIP embeddings, making them potentially redundant and dependent on CLIP's limitations. The theoretical justification (Appendix D) doesn't convincingly establish that CLIP similarity is a valid proxy for the proposed probabilistic framework. Second, the Specificity metric can reward poor images that match neither the concept nor its neighbors. Third, human evaluation involves only 4 annotators on ~3370 pairs, which limits statistical robustness. Fourth, the paper lacks deeper analysis explaining *why* Playground and FLUX perform better—the findings are primarily empirical without mechanistic insight. Finally, FID computed against retrieved images is a questionable choice since retrieved images themselves may not accurately represent concepts.

The discrepancy between preference-based and CLIP-based rankings (SDXL-turbo dominates CLIP metrics while Playground dominates preferences) remains unexplained, suggesting evaluation may measure different underlying qualities rather than converging on a single notion of quality.

Overall, this is a useful benchmark paper that enables future research, with solid empirical work but limited methodological novelty and some unresolved questions about metric validity.

Score: 7.0

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

This paper introduces Ambig-SWE, a benchmark for evaluating how LLM agents handle underspecified instructions through interactive clarification in software engineering tasks. The authors decompose the problem into three key capabilities: detecting underspecificity, asking targeted questions, and leveraging interaction to improve performance. They evaluate both proprietary (Claude Sonnet 3.5/4, Claude Haiku 3.5) and open-weight models (Llama 3.1 70B, Deepseek-v2, Qwen 3 Coder) across these dimensions.

The paper's primary strength is addressing an important real-world problem—underspecified instructions are common in practice but understudied in agent evaluation. The three-step evaluation framework is well-motivated and enables targeted analysis of where models succeed or fail. The empirical findings are valuable: models default to non-interactive behavior without prompting, struggle to distinguish well-specified from underspecified inputs (except Claude Sonnet models), and interaction can recover up to 74% of performance lost to underspecification. The analysis of questioning strategies (exploration-first vs. immediate asking) provides actionable insights for agent design. The finding that Qwen 3 Coder exhibits rigid non-interactive behavior regardless of prompting (100% FNR) is particularly notable.

However, there are notable limitations. The underspecified issues are synthetically generated via GPT-4o rather than naturally occurring, raising concerns about ecological validity. The user proxy setup assumes users have complete ground-truth information, which is often unrealistic. Some evaluation metrics have limited discrimination (LLM-as-judge scores converge around 4/5 across models). Detection is only measured in the first three turns, missing cases where underspecificity emerges later. The methodology follows prior interactive agent work closely, limiting novelty.

The paper makes a solid contribution to understanding interactive agent behavior under uncertainty, with clear empirical insights. While the synthetic setup and idealized user proxy are limitations, the controlled experimental design enables causal inference about interaction effects. The framework and findings provide a useful foundation for future work on more adaptive agents.

Score: 7.2

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper provides a systematic empirical study of memory-performance trade-offs for reasoning models, which generate substantially longer sequences than standard LLMs and thus face different memory constraints. The authors investigate how to allocate a fixed memory budget across model size, weight precision, generation length, parallel samples, and KV cache compression.

**Strengths:**
The paper addresses a timely and practically important problem. As reasoning models become prevalent, understanding how KV cache memory interacts with model deployment is critical. The empirical study is thorough, spanning 1,700+ configurations across multiple model families (Qwen3, DeepSeek-R1-Distill, OpenReasoning-Nemotron), benchmarks (AIME25, GPQA-Diamond, LiveCodeBench, MATH500), and quantization methods (GPTQ, AWQ, FP8). The key finding that 4-bit quantization is NOT universally optimal—contradicting established findings for non-reasoning models—is significant and counter-intuitive. The task-dependent precision insights (math/code prefer higher precision, knowledge-intensive tasks prefer 4-bit) provide actionable guidance. The analysis of when to prioritize model capacity vs. test-time compute based on effective model size is practically useful.

**Weaknesses:**
The contribution is primarily empirical without novel methods or theoretical framework. The "8-bit 4B" threshold is empirically identified without theoretical justification—is this fundamental or specific to current architectures? The scope of KV cache compression methods is limited (only R-KV, StreamingLLM, HQQ). The paper focuses narrowly on budget forcing for serial scaling, leaving other reasoning approaches (verifiers, search-based methods) largely unexplored. While the verifier analysis with PRM is included, it's quite brief.

**Overall:**
This is a solid empirical contribution with practical value for an emerging area. The systematic study yields useful guidelines, though it lacks methodological novelty or theoretical depth. The findings are important enough to merit publication, but the empirical-only approach limits broader impact.

Score: 7.0

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper provides a rigorous theoretical analysis comparing adaptive optimizers (Adam, AdaGrad, Shampoo) with Normalized Steepest Descent (NSD) methods. The key insight is that while both algorithm families exploit non-Euclidean geometry, they do so through fundamentally different smoothness and variance assumptions.

**Strengths:**
- The paper clearly separates two notions of smoothness: standard smoothness (governing NSD convergence) and adaptive smoothness (governing adaptive optimizer convergence). The key result is that adaptive smoothness enables O(1/T²) acceleration with Nesterov momentum in convex settings, while standard ℓ∞-smoothness cannot achieve better than Ω(1/T)—a meaningful separation result.

- The introduction of adaptive gradient variance and its dimension-free convergence guarantee (versus dimension-dependent bounds under standard variance) is a nice parallel contribution, with matching lower bounds establishing the gap is unavoidable.

- The unified analysis covering AdaGrad, Adam, and one-sided Shampoo via well-structured preconditioner sets is elegant, and the novel matrix inequality (Lemma 3.3/C.1) for handling noncommutativity in the nonconvex setting appears to be a genuine technical contribution with potential independent interest.

- The presentation builds logically from the convex analysis (building on Xie et al. 2025b) to nonconvex settings, and the analogy between smoothness and variance assumptions is well-motivated.

**Weaknesses:**
- The acceleration benefit comes from a strictly stronger assumption—adaptive smoothness implies standard smoothness, so the comparison isn't perfectly apples-to-apples. While the separation result is correct, the practical implication (i.e., when does adaptive smoothness meaningfully hold in practice?) receives limited discussion.

- The log(d) factor for general well-structured preconditioners versus diagonal-only cases (where it disappears) could be better contextualized in terms of practical significance.

- Some proofs are deferred to the appendix with dense technical machinery; additional intuition could improve accessibility.

**Overall:** This is a solid theoretical contribution that meaningfully advances our understanding of why adaptive methods differ from NSD methods beyond algorithmic equivalence. The separation results are significant, and the unified framework is valuable. The paper is well-executed with appropriate references to concurrent and prior work.

Score: 7.5

---

## 0cbUKCyBsH

- GT: Reject (avg 3.5)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces Influence-Aware Time Series Forecasting (IATSF), arguing that the field's stagnation stems from a "self-stimulation" assumption where models predict using only historical values. The authors provide theoretical analysis, a new benchmark, and the FIATS model.

**Strengths:** The theoretical framework (Propositions 2.1, 3.1) formally establishes that ignoring exogenous influences creates an irreducible error floor—a valuable insight grounded in control theory. The Temporal-Synced IATSF benchmark with multiple datasets (Toy, Electricity, Atmospheric Physics, Traffic, GAUD) designed to avoid information leakage is a solid community resource. The FIATS architecture, particularly CASM and CAPS mechanisms, provides principled mechanisms for channel-specific influence modulation. Empirical results show dramatic improvements on synthetic data and meaningful gains on real-world datasets.

**Weaknesses:** Several concerns limit confidence in the claimed contributions: (1) The FM Toy dataset is explicitly designed for the theory to succeed—it's a "gotcha" benchmark where models *must* use influence to predict frequency changes, making the near-zero FIATS error versus total baseline failure unsurprising rather than impressive. (2) Comparisons against baselines that receive *no* influence information are inherently unfair; a proper baseline would be models with access to exogenous variables formatted conventionally. (3) The Atmospheric Physics dataset uses weather *forecasts* as influence, which are themselves predictions of quantities being forecasted—this creates potential circularity. (4) The "hard mathematical barrier" framing is overstated; the bound depends on system structure and influence independence assumptions that may not hold broadly. (5) The theoretical contribution is essentially classical (exogenous inputs matter in dynamic systems), repackaged without sufficient acknowledgment of control theory precedent.

The paper makes a genuine contribution in highlighting the importance of external influences and providing tools to study this systematically. However, the experimental design biases toward confirming the theory rather than rigorously testing it, and comparisons lack appropriate influence-aware baselines.

Score: 5.0

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important problem in multimodal learning—modality imbalance—through a novel lens connecting it to class imbalance. The authors propose CCAT, a two-stage framework that pretrains an unbiased classifier and freezes it during alternating modality training, using LoRA modules for modality-specific adaptation.

**Strengths:**
The paper's key insight—that alternating training reduces encoder interference but fails to address classifier bias toward dominant modalities—is compelling and well-motivated. The empirical analysis in Figure 1 effectively demonstrates this gap. The proposed framework is principled, with clear components that each address specific aspects of the problem: classifier freezing provides stable optimization targets, LoRA modules preserve modality-specific features, and sample-level re-optimization handles extreme imbalance cases. Experimental results demonstrate consistent improvements over strong baselines across three benchmarks (+1.35% to +6.76%), with comprehensive ablation studies validating each component's contribution. The t-SNE visualizations and clustering metrics provide useful qualitative evidence.

**Weaknesses:**
The theoretical analysis in Section 3.1, while conceptually interesting, makes strong simplifying assumptions and doesn't provide rigorous guarantees. The claimed "theoretical isomorphism" between class and modality imbalance is somewhat overstated—the gradient analysis is fairly elementary. The method combines existing techniques (frozen classifiers from long-tail learning, LoRA from LLM adaptation, sample re-weighting) without substantial algorithmic novelty. Several hyperparameters (r, β, λ) require dataset-specific tuning, and the paper lacks analysis of computational overhead from the two-stage training and secondary updates. Additionally, the mutual information estimation in Eq. 5 uses inner products as a proxy rather than actual MI, which is imprecise.

**Overall:**
This is a solid, well-executed contribution with consistent empirical gains. While the technical components are incremental, their combination in service of a clearly-identified problem (classifier bias in alternating training) represents meaningful progress. The work could benefit from deeper theoretical analysis and efficiency discussion but offers clear value to the multimodal learning community.

Score: 7.5

---

## Kw2mvnzCoc

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Assessment

This paper introduces TSPulse, a lightweight (1M parameter) pre-trained model for time-series diagnostic tasks that achieves strong performance through disentangled representation learning across temporal, spectral, and semantic views.

## Strengths

The paper makes several valuable contributions. First, the **disentangled representation learning** approach is well-motivated—different time-series tasks genuinely benefit from different perspectives (temporal precision for imputation, spectral features for periodicity detection, semantic embeddings for similarity search). The empirical evidence that different embeddings exhibit complementary properties (e.g., temporal embeddings being phase-sensitive while semantic embeddings are invariant) supports this design choice.

Second, the **empirical results are impressive**: achieving state-of-the-art on TSB-AD leaderboard for anomaly detection, significant improvements on imputation (+50%), and strong classification accuracy, all while being 10-100× smaller than competing models. The efficiency analysis (Table 3) demonstrates genuine practical value for deployment.

Third, the **hybrid masking strategy** addresses a real limitation in existing approaches—block masking during pre-training creates a distribution mismatch with irregular real-world missingness patterns. The ablation showing 79% performance degradation without hybrid masking (Table 1c) validates this design.

Fourth, the paper provides **comprehensive ablations** and sensitivity analyses demonstrating that learned embeddings genuinely capture disentangled properties.

## Weaknesses

**Incremental novelty**: Individual components—dual-space learning (explored in TF-C, BTSF), register tokens (vision transformers), hybrid masking—are not novel. The contribution is integration rather than fundamental innovation.

**Comparison fairness concerns**: For imputation, the paper compares TSPulse's task-specialized pre-training against MOMENT's zero-shot performance. When MOMENT uses task-specific adaptation (Table 20 shows prompt-tuned UniTS), the gap narrows. Similarly, using Chronos (a forecasting model) for similarity search is a somewhat weak baseline.

**Task-specific pre-training overhead**: The approach requires training separate models per task through loss reweighting, reducing the appeal of a "unified" foundation model. The paper acknowledges this but doesn't fully address whether task-agnostic pre-training could achieve comparable results.

**Limited architectural innovation**: The backbone is essentially TSMixer with standard modifications. The main innovations are training objectives, which while effective, represent a narrower contribution scope.

**Anomaly detection evaluation**: TSB-AD has known methodological concerns (varied ground truth quality across datasets). The paper could more critically discuss these limitations.

## Overall

This is a solid contribution to time-series foundation models with strong empirical results and practical value. The disentanglement idea is thoughtfully designed and validated. However, the novelty is primarily in integration of existing techniques rather than fundamental innovation. The efficiency focus and strong empirical performance across multiple tasks make this a reasonable contribution to ICLR.

Score: 7.5

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes CANON (Conditional AdvaNtage estimatiON), a novel advantage estimation method for RLVR that leverages training metrics (entropy, length) without imposing directional priors. The key insight is clever: instead of hand-crafting penalties that assume "higher-is-better" or "lower-is-better" for a metric, the method regroups responses by metric values and lets the reward signal determine which direction is beneficial. The inter-group advantage identifies which metric trend correlates with higher rewards, while intra-group advantage identifies better responses within each trend.

**Strengths:** The paper has strong theoretical motivation, correctly showing that DR.GRPO is a special case of CANON (μ=0.5) and that equal-sized groups maximize signal amplification. Theorem 2 demonstrates that CANON selectively amplifies only the grouping metric without affecting independent factors. Empirical results are comprehensive across 3 models and 9 benchmarks, with consistent improvements: +1.9 points on math tasks (CANON-Inter), +5.2 points on complex logic (CANON-Intra), and a genuine Pareto frontier improvement for token efficiency. The ablations (random regrouping, numerical scaling, per-token reflection) properly isolate the contribution of metric-based conditional grouping.

**Weaknesses:** The dynamic scheduling introduces complexity—four different strategies are tried with the paper selecting the best per model, which raises concerns about cherry-picking. While the paper claims μ doesn't add tuning burden, Table 10 shows performance varies meaningfully with μ, and selecting the right scheduling strategy requires experimentation. The connection between inter-group favoring exploitation and intra-group favoring exploration is intuitively plausible but not deeply analyzed. The improvements, while consistent, are modest in absolute terms (~2-5 points). Only entropy and length metrics are explored, though this is acknowledged as a limitation.

Overall, this is a well-executed paper with a clear, principled contribution to RLVR methods. The idea of avoiding directional priors while still leveraging metric structure is valuable, and the theoretical grounding is solid.

Score: 7.5

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper provides a comprehensive empirical analysis of verifiers used in reinforcement learning with verifiable rewards (RLVR) for mathematical reasoning. The work is timely and addresses a critical component of modern reasoning models like DeepSeek-R1, making it highly relevant to the current research landscape.

**Strengths:**
The paper makes several important contributions. First, it systematically quantifies the limitations of widely-used rule-based verifiers, showing that they achieve only ~86% recall on average—a significant finding that challenges assumptions in prior RLVR work. The observation that false negative rates increase with stronger policy models is particularly concerning for the field's trajectory. Second, the paper provides valuable insights into model-based verifiers: while they improve static accuracy, fine-tuned verifiers can become vulnerable to reward hacking during RL training—a counterintuitive and practically important finding. Third, the systematic probing study with adversarial patterns reveals that generative (CoT-based) verifiers are more susceptible to simple attacks than discriminative ones, offering actionable guidance for future verifier design. The empirical methodology is thorough, spanning multiple datasets, verifier types, and training configurations, with GPT-4o used as an oracle to detect reward hacking phenomena.

**Weaknesses:**
Several aspects could be strengthened. First, while the paper clearly identifies problems, it offers limited exploration of solutions—proposing defenses against reward hacking would have strengthened the contribution. Second, the reliance on GPT-4o as ground truth for validation, though partially justified with human annotation, may introduce its own biases. Third, the mechanistic explanation for why fine-tuning increases vulnerability remains underdeveloped. Finally, while the paper briefly extends to general science domains, deeper analysis across more diverse reasoning tasks would strengthen generalizability claims.

**Overall:**
This is a solid empirical contribution that surfaces important challenges in RLVR systems. The findings will influence how practitioners design reward systems, and the careful experimental design provides valuable insights for the community.

Score: 7.5

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (8.0/10)
- Match: N/A

### Review

# Assessment

This paper introduces SIGMADOCK, a fragment-based SE(3) diffusion model for molecular docking that achieves state-of-the-art results on the PoseBusters benchmark. The work makes several important contributions: (1) a novel fragmentation scheme (FR3D) that reduces degrees of freedom while preserving chemical validity, (2) rigorous theoretical analysis explaining why fragment-based diffusion avoids the entangled measures that plague torsional models, and (3) an SO(3)-equivariant architecture with soft triangulation constraints.

The empirical results are compelling—the method achieves 79.9% Top-1 success (RMSD < 2Å & PB-valid) on PoseBusters, compared to 12.7-32.8% for previous deep learning methods and ~59% for classical docking. This represents the first deep learning approach to surpass classical physics-based methods under the standard PoseBusters train-test split, marking a meaningful advance in the field. The authors also demonstrate strong generalization across sequence similarity splits and perform thorough ablations validating each component.

The theoretical grounding is a strength. Theorem 1 formally shows that torsional models induce non-product measures in Cartesian space (causing entangled, ill-conditioned dynamics), while fragment-based approaches yield factorized product measures. This provides principled justification for the approach and explains previous difficulties with torsional diffusion models.

Several limitations deserve mention: the evaluation is restricted to re-docking with known holo-structures (no cross-docking or apo-docking), the method requires a user-specified pocket definition, and chirality preservation relies on post-filtering. However, these are honestly acknowledged and represent reasonable trade-offs. The comparison to AlphaFold3 in the appendix shows comparable performance with ~25× less training data, supporting the authors' claim about data efficiency.

The paper is well-written with comprehensive appendices including proofs, implementation details, and extended analyses. The code availability statement enhances reproducibility.

Score: 8.0

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces the conditional causal bandit problem, where arms are single-node conditional interventions on a causal graph. The authors characterize the minimal globally interventionally superior set (mGISS)—the smallest set of nodes guaranteed to contain the optimal intervention target. The key insight is connecting conditional intervention superiority to deterministic atomic intervention superiority (Proposition 4), enabling simpler analysis. The main theoretical result shows that mGISS equals the LSCA closure of the parents of Y, where LSCAs are "lowest strict common ancestors"—a refined notion excluding paths through the other node. The C4 algorithm computes this set in O(|V| + |E|) time using a connector-based bottom-up traversal.

**Strengths:** The problem is well-motivated: conditional interventions generalize both hard and soft interventions and better model real-world decision-making. The graphical characterization via Λ-structures is elegant, and the algorithm is both correct and optimally efficient. The equivalence between conditional and deterministic atomic superiority is a clever simplification. Empirical results demonstrate meaningful search space reduction, particularly for sparse graphs (often retaining <30% of nodes), and improved regret when integrated with a UCB algorithm on real-world bnlearn graphs.

**Weaknesses:** The restrictive assumptions—single-node interventions only, and no latent confounders—limit practical applicability compared to Lee & Bareinboim (2018), which handles multi-node hard interventions with confounders. While the authors argue the single-node setting is more challenging, it remains a limitation. The conditioning set assumptions (An(X) ⊆ ZX) may not hold in all applications. The regret experiments use a simple UCB variant; comparison with more sophisticated causal bandit algorithms would strengthen the evaluation. Finally, the work extends existing ideas rather than introducing an entirely new paradigm.

The contribution is solid and technically sound, but the restricted setting and incremental nature relative to prior work prevent a higher score.

Score: 7.0

---

## 3icvqeC1sA

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents ChaosNexus, a foundation model for universal chaotic system forecasting. The work introduces ScaleFormer, a U-Net-inspired Transformer architecture with multi-scale temporal representations, combined with Mixture-of-Experts layers and wavelet-based frequency fingerprinting.

**Strengths:**
The paper makes a strong case that chaotic systems fundamentally differ from general time series—they exhibit broadband continuous spectra rather than sparse line spectra, and contain multi-scale temporal structures that single-resolution models fail to capture. This insight is valuable and well-motivated. The ScaleFormer architecture with hierarchical patch merging/expansion effectively addresses this limitation. The empirical evaluation is comprehensive: 9,300+ synthetic chaotic systems, real-world weather forecasting on WEATHER-5K, and multiple metrics including attractor statistics (correlation dimension, KL divergence, Lyapunov exponent error). Results are strong—ChaosNexus achieves competitive point-wise accuracy while notably outperforming baselines on attractor fidelity metrics (D_frac: 0.203 vs 0.227, D_stsp: 1.206 vs 2.369 vs Panda). The zero-shot weather forecasting below 1°C MAE without any fine-tuning, outperforming baselines fine-tuned on 473K samples, is impressive. The scaling analysis (system diversity > trajectory count) provides useful guidance for future foundation models. Ablation studies are thorough, and expert activation visualizations provide interpretability.

**Weaknesses:**
The architectural contribution is somewhat incremental—it combines established techniques (U-Net structure, MoE, wavelet transforms, MMD regularization) without introducing fundamentally new mechanisms. While the combination is appropriate for the domain, novelty is limited. The weather evaluation lacks comparison to modern weather forecasting models (GraphCast, FourCastNet, Pangu-Weather); instead, it compares to general time-series models. Training exclusively on synthetic ODE systems raises questions about distribution shift when applied to real chaotic systems. The point-wise accuracy improvement over Panda is modest (68.901 vs 69.567 sMAPE), though attractor statistics show stronger gains. The MMD weight λ2=0.5 appears important (Table 5), raising potential sensitivity concerns.

**Overall:**
This is a solid, well-executed contribution to foundation models for scientific computing. The multi-scale insight for chaotic systems is valuable, and empirical results are convincing. However, the incremental architectural nature and limited comparison to domain-specific weather models prevent a higher score.

Score: 7.5

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a streaming framework for stimulation-response modeling of latent neural dynamics, addressing an important gap in neuroscience methodology: how to design neural stimulations that drive latent dynamics in desired directions under realistic experimental constraints.

**Strengths:**
The paper tackles a timely and significant problem - enabling causal testing of neural manifold hypotheses through targeted stimulation. The framework integrates streaming dimensionality reduction, adaptive stimulus-response modeling via kernel regression, and constrained optimization in a coherent pipeline. The novel sjPCA algorithm for streaming estimation of rotational subspaces is a solid contribution, and the demonstration of <10ms average runtime makes real-world deployment feasible. Testing across two modalities (calcium imaging and electrophysiology) and multiple dynamical models shows careful validation. The optimization formulation thoughtfully incorporates realistic constraints (non-negativity, sparsity, magnitude limits) for excitation-only optogenetics.

**Weaknesses:**
The most significant limitation is that all real-data experiments use *simulated* stimulations rather than actual photostimulation. While understandable as proof-of-concept, this substantially weakens the claims about real-world applicability. The individual components (Kalman filter, kernel regression, proSVD) are existing methods; novelty lies primarily in integration. The closed-loop optimization sometimes underperforms open-loop (Figure 5), which deserves deeper analysis. Baselines are limited—the "blind" comparison simply ignores stimulation effects. The framework assumes only one pending stimulus at a time, limiting experimental throughput.

**Overall:**
This is a solid methods paper addressing an important problem with practical engineering that could enable new experiments. However, the lack of real closed-loop stimulation validation and the incremental nature of the technical contributions keep it from being exceptional.

Score: 7.0

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a Covariance-Adjusted Support Vector Machine (CSVM) that incorporates class-specific covariance information through Cholesky decomposition. The authors argue that traditional SVM is invalid in "non-Euclidean" input space and must be performed in a transformed Euclidean space.

**Strengths:** The paper identifies a legitimate issue—SVM's margin calculation doesn't account for differing class covariances, and class-specific whitening could improve classification. The experimental evaluation on five datasets shows competitive performance against standard SVM kernels and whitening approaches. The iterative SM algorithm attempts to address the practical challenge of unknown test labels when computing population covariance.

**Weaknesses:** The theoretical framework has fundamental flaws. The paper incorrectly characterizes input space as "non-Euclidean"—feature space $\mathbb{R}^n$ with standard inner product is by definition Euclidean. Mahalanobis distance is simply a different metric, not a different geometric space. The claim that "SVM is only valid in Euclidean space" is false; SVM works in any reproducing kernel Hilbert space. Lemma 2.2's assertion that N classes require N classifiers contradicts standard multi-class SVM theory. Additionally, class-specific whitening creates a circular dependency (need class labels to compute transformation, but transformation affects classification), which the SM algorithm only addresses heuristically without convergence guarantees. Experimental gains over baselines are often marginal (e.g., 0.974 vs 0.956 accuracy), lack statistical significance testing, and show no cross-validation results. The approach is essentially class-specific whitening + SVM, which has been explored in prior work, though the theoretical motivation presented here is novel but flawed.

The core insight (class-specific whitening for SVM) has merit, but the theoretical justification is incorrect, and empirical gains don't compensate for the conceptual issues.

Score: 4.5

---

## L2rfd2Czbj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes **wd1**, a novel policy optimization method for diffusion-based large language models (dLLMs) that reformulates the RL objective as a weighted log-likelihood, avoiding the need to compute likelihood ratios for importance sampling. The key insight is that likelihood approximations in dLLMs are computationally expensive and can introduce exponential error amplification when computing policy ratios. By deriving a weighted log-likelihood objective from reverse-KL regularized policy optimization, the authors enable single-likelihood-approximation training while providing both positive and negative sample weighting.

**Strengths:**
- **Well-motivated problem**: The challenges of applying ratio-based RL methods to dLLMs are clearly articulated. The exponential error amplification from likelihood approximation in ratio computation (Appendix A.1) is an important observation.
- **Solid theoretical grounding**: The connection to energy-guided diffusion sampling (Theorem 1, Lemma 1) provides a meaningful interpretation, showing that the objective trains an advantage-guided diffusion model.
- **Strong empirical results**: The improvements over d1 are substantial—+59% on Sudoku, +16% on Countdown—with reduced computational cost (no SFT needed, fewer NFEs per step). The wd1++ extension achieves SOTA results on MATH500 and GSM8K with 10× fewer rollouts.
- **Novel method design**: The introduction of both positive and negative weights (w+, w-) with the "unlearning" interpretation is clever, and the ablations confirm its importance.

**Weaknesses:**
- **Incomplete theoretical justification**: The theoretical analysis (Theorem 1) only justifies the positive weight term; the negative weight term relies on intuitive motivation rather than rigorous derivation. The "unlearning" interpretation is mentioned but not formally connected.
- **Biased likelihood approximation**: The method inherits d1's biased likelihood approximation. While this enables efficiency, the theoretical guarantees assume exact likelihoods, and the interaction between bias and weighted objectives isn't analyzed.
- **Limited scope of comparisons**: The primary comparison is against d1. Comparisons with concurrent methods (SDPO, TCR, MDPO) are only for wd1++ and lack detailed analysis of why wd1's advantages transfer.
- **Hyperparameter sensitivity**: Ablations show sensitivity to ψ and combined weights, but the paper provides limited guidance on tuning these in new domains.

Overall, this is a well-executed contribution to an emerging area with meaningful practical improvements. The core insight is valuable, though the theoretical analysis has gaps.

Score: 7.2

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces ConciseHint, a novel framework for improving the efficiency of large reasoning models by injecting concise hints *during* the reasoning generation process, rather than applying interventions beforehand. This represents a genuinely new intervention paradigm that the authors argue (convincingly) has been overlooked by prior work focused on prompting strategies or fine-tuning approaches.

**Strengths:**
The core insight—intervening during reasoning generation rather than before—is conceptually clean and well-motivated. The complexity-adaptive mechanism that reduces hint intensity for longer reasoning chains (potentially complex problems) while maintaining high intensity for shorter chains (easy problems) is a clever design that empirically works well. The experimental validation is thorough across multiple models (Qwen3 series, DeepSeek-R1) and benchmarks (GSM8K, AIME24, GPQA-Diamond), showing consistent token reductions of 40-65% with minimal accuracy loss. The method's ability to combine with existing baselines (BeConcise, Deer, NoWait) to push efficiency further is a practical strength. The ablation studies on injection interval and position are informative, and the analysis of computational overhead shows negligible latency costs (~0.3%). The trainable variant (ConciseHint-T) adds useful controllability through interpolation.

**Weaknesses:**
The complexity estimation heuristic (using current reasoning length as a proxy) lacks theoretical grounding—while empirically reasonable, the paper doesn't explore why this proxy works or its failure modes. The injection position formula is somewhat arbitrary without clear justification beyond empirical observation. The manual hint design ("make answer concise!") is simple; a more systematic exploration of hint design could strengthen the work. For ConciseHint-T, training on GSM8K-derived data and evaluating on GSM8K (in-domain) raises minor concerns about generalization claims, though out-of-domain results are provided. The paper could benefit from standard deviation reporting and more detailed failure case analysis.

**Overall:**
This is a solid contribution to efficient reasoning. The in-reasoning intervention paradigm is novel, the adaptive mechanism is well-designed, and empirical results are convincing. While some design choices lack deep theoretical justification, the practical effectiveness and thorough evaluation make this a clear accept.

Score: 7.5

---

## sh1hWO9RHo

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces the Agent GPA (Goal-Plan-Action) framework for evaluating LLM agents through seven specialized LLM judges covering goal fulfillment, logical consistency, execution efficiency, plan quality, plan adherence, tool selection, and tool calling. The framework is evaluated on TRAIL/GAIA, an internal dataset, and TRAIL/SWE-bench.

**Strengths:** The paper addresses an important gap in agent evaluation—existing methods focus on final outcomes or require ground-truth references. The GPA framework provides a structured, reference-free approach with strong empirical validation: 95% error detection rate (vs. 55% baseline), 86% error localization agreement with humans, and strong inter-rater reliability (α > 0.7 for most judges). The detailed analysis of judge trade-offs (e.g., TC as "conservative" vs. TS as high-recall) provides actionable guidance for practitioners. The GEPA prompt optimization experiments demonstrate practical utility and generalization potential. The inclusion of orthogonality analysis between metrics is methodologically sound.

**Weaknesses:** The conceptual framework itself is relatively intuitive—decomposing agent evaluation into goal/plan/action components is a natural progression from prior work. The Plan Quality judge shows notably poor reliability but isn't adequately addressed. The internal dataset has only 17 traces, limiting confidence in those results. While the LLM-as-Judge approach achieves good human agreement, it inherits inherent limitations (potential biases, difficulty with nuanced errors). Some metric overlap exists (EE-TC correlation), suggesting redundancy. Baseline comparisons are limited to TRAIL's judge; comparisons with other frameworks like AgentRewardBench would strengthen the contribution. The human annotation process lacks detail on guidelines and disagreement resolution.

**Overall:** This is a well-executed contribution with clear practical value for agent debugging and improvement, though with moderate conceptual novelty and some methodological limitations.

Score: 7.5

---

## 32mrjmaeMP

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes TAK (Task Arithmetic with KFAC regularization), a data-free approach to weight disentanglement in task arithmetic. The key contribution is establishing a theoretical connection between representation drift regularization and the generalized Gauss-Newton matrix, enabling use of KFAC approximations instead of requiring access to external task data.

**Strengths:**
The paper makes a clever theoretical connection by showing that representation drift simplifies to a quadratic form involving the Jacobian Gram matrix under linearization, which is equivalent to the GGN. This enables leveraging well-studied KFAC approximations for practical regularization. The method achieves strong empirical results on both vision (CLIP ViT) and language (T5-base) benchmarks, matching or exceeding prior methods while being fully dataless. Particularly impressive is the task negation/unlearning performance and the robustness to scaling coefficients that eliminates the need for held-out tuning. The aggregation scheme achieving O(1) complexity in the number of tasks is a meaningful efficiency improvement. The writing is clear and the evaluation is thorough across multiple regimes.

**Weaknesses:**
The theoretical justification for the KFAC merging heuristic (Eq. 8) is thin. While Appendix C provides a basic error bound, this doesn't establish that the approximation is suitable for optimization purposes. The performance gap from τJp (which uses task data), though modest, persists. Memory scaling with layer width could be problematic for very large models. The non-linear regime extension via attention-only fine-tuning, while interesting, shows significant performance gaps from the linearized regime. The method requires pre-computing and sharing KFAC factors across tasks, which while more efficient than sharing data, still requires coordination.

**Overall:**
This is a solid contribution with practical value for privacy-preserving and modular model composition. The core insight is elegant, execution is competent, and results are meaningful. However, the theoretical gaps around the aggregation heuristic and residual performance gap from data-dependent methods prevent it from being exceptional.

Score: 7.0

---

## 4Ha2srdhPN

- GT: Reject (avg 4.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper presents GRAID, a framework for generating spatial reasoning VQA data using only 2D bounding boxes, avoiding the error propagation from single-image 3D reconstruction that plagues existing methods like SpatialVLM. The approach is motivated by a compelling human evaluation showing that only 57.6% of questions from SpatialVLM's community implementation are valid, compared to 91.16% for GRAID-generated data.

The key insight is elegant: qualitative spatial relationships (left/right, counting, size comparisons) can be determined reliably from 2D geometry alone, without cascading errors from depth estimation and camera calibration. The authors generate 8.5M VQA pairs across three driving datasets using 22 question templates, and introduce SPARQ for efficient predicate-based filtering that achieves up to 1400× speedup.

The experimental evaluation is comprehensive. The authors demonstrate: (1) cross-dataset generalization where training on GRAID-BDD improves performance on unseen GRAID-NuImages; (2) transfer from 6 question types to held-out types, showing genuine concept learning; and (3) consistent improvements on external benchmarks (BLINK, A-OKVQA, RealWorldQA) across four VLM backbones (Llama, Gemma, Qwen). The +35.66% improvement on BLINK spatial relations for Llama-3.2 is particularly notable.

However, several limitations temper my enthusiasm. First, the question types are relatively simple—mostly binary spatial relations, counting, and size comparisons. More complex spatial reasoning (occlusion handling, 3D relationships, perspective-taking) is not addressed. Second, the driving-domain focus means all experiments involve automotive scenes; the claimed domain-agnostic nature is not empirically demonstrated. Third, the depth-related questions (Closer, Farther) still rely on monocular depth estimation, somewhat undermining the "no 3D reconstruction" claim. Fourth, the comparison baseline (OpenSpaces/SpatialVLM) appears to severely degrade model performance, making GRAID's improvements partly about avoiding harm rather than providing benefit. Finally, while the 91.16% human validation rate is impressive, the evaluation methodology could be more robust—only 317 pairs were evaluated by 4 annotators without inter-annotator agreement metrics.

The technical contribution is straightforward but well-executed. The SPARQ optimization is practical, and the empirical results are solid. The work addresses a real need in the VLM community for higher-quality spatial reasoning training data. However, the approach does not fundamentally advance spatial reasoning capabilities—it provides cleaner data for learning simple, existing question types.

Score: 7.2

---

## oh9ChF7Pv0

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces EGG-SR, a unified framework that leverages equality graphs (e-graphs) to incorporate symbolic equivalence into symbolic regression algorithms. The core insight is that many mathematically equivalent expressions have different syntactic representations (e.g., log(x₁³x₂²) = 3log(x₁) + 2log(x₂)), and treating these as distinct leads to redundant exploration in the search space.

**Strengths:**

1. **Novel and principled approach**: Using e-graphs for symbolic equivalence in SR is innovative. The paper correctly identifies that existing methods waste computation exploring equivalent expressions, and provides an elegant solution with theoretical grounding.

2. **Strong theoretical contributions**: Theorem 3.1 shows EGG-MCTS achieves tighter regret bounds (κ_∞ ≤ κ), and Theorem 3.2 proves EGG-DRL yields an unbiased gradient estimator with strictly lower variance. The proofs are detailed and sound.

3. **Unified framework across paradigms**: The approach generalizes across MCTS, DRL, and LLM-based SR, demonstrating broad applicability. The integration into each paradigm is thoughtful and well-motivated.

4. **Solid empirical results**: Tables 1-2 show consistent improvements across all three paradigms. The trigonometric dataset results are particularly strong for EGG-MCTS. Figure 4 demonstrates substantial memory efficiency, and Figure 5 shows acceptable computational overhead.

5. **Clear exposition**: The paper explains e-graphs well for a ML audience, and the methodology section is detailed enough for reproduction.

**Weaknesses:**

1. **Limited benchmark coverage**: Experiments focus primarily on trigonometric datasets for MCTS/DRL and only 4 scientific benchmarks for LLM. Standard SR benchmarks (Feynman, Nguyen, etc.) would strengthen the evaluation. The paper mentions Feynman in Appendix D but only for visualization, not quantitative comparison.

2. **Rewrite rule dependency**: Effectiveness hinges on rule quality/coverage, yet there's no ablation on rule set size, no discussion of what happens with incomplete rules, and limited analysis of computational cost scaling with rule set size.

3. **Missing hyperparameter analysis**: Key parameters like K (number of sampled equivalents in EGG-DRL/LLM) are not ablated. How sensitive is performance to this choice?

4. **Domain constraint handling**: Appendix B.2 notes that rewrite rules have domain restrictions (e.g., log identities require positive arguments), but the handling seems cursory. Expressions generating NaN on training data are simply discarded—this could bias the search.

5. **Dated baselines**: The DRL baseline (Petersen et al., 2021) and MCTS baseline (Sun et al., 2023) are not the most recent SR methods. Including more modern baselines would strengthen the comparison.

Despite these weaknesses, the core contribution is significant: a theoretically-grounded, unified approach to exploit symbolic equivalence across SR paradigms. The consistent improvements and principled methodology make this a solid contribution.

Score: 7.5

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces EmoSign, a dataset for emotion recognition in American Sign Language (ASL) videos, addressing a genuine gap in the literature. The work has several notable strengths: it provides the first emotion-labeled ASL dataset, employs Deaf native ASL signers as annotators (critical for cultural and linguistic accuracy), includes multi-layered annotations (sentiment, emotion categories, and qualitative cue descriptions), and offers systematic benchmarking of multiple multimodal LLMs across different input conditions.

The methodology is commendable—the authors clearly describe their collaboration with the Deaf community, the annotation pipeline, and provide reasonable inter-annotator agreement metrics (Krippendorff's α = 0.593). The findings that current MLLMs struggle with video-only emotion recognition and exhibit positive bias are useful contributions.

However, the paper has significant limitations. **Most critically, the dataset is very small**—only 200 video clips totaling approximately 16 minutes. While the authors argue similar-sized datasets have been valuable for benchmarking, this scale severely limits practical utility for training or meaningful evaluation. The benchmark experiments only evaluate existing models in zero-shot settings without any fine-tuning or training, which substantially weakens the contribution. We don't learn whether this dataset could actually help models improve.

Additional concerns include: (1) using VADER to filter videos based on English captions may bias the dataset toward cases where text and visual sentiment align; (2) the "emotion cue grounding" task lacks systematic evaluation, offering only qualitative examples; (3) the 10 emotion categories seem arbitrary without strong justification; and (4) FePh, the closest prior dataset, receives insufficient comparison.

The paper is well-written and addresses an important problem. The collaboration with native Deaf signers is a real strength. However, the limited scale and absence of training experiments make this feel more like a pilot study than a substantial contribution suitable for a top venue. A dataset this small provides limited value for the community without accompanying experiments demonstrating its utility for model development.

Score: 4.5

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Visuo-Tactile World Models (VT-WM), which integrates fingertip tactile sensing with vision to address a fundamental limitation of vision-only world models in contact-rich manipulation tasks. The work is well-motivated: vision-only models struggle with occlusion and visual aliasing during contact, leading to physically implausible rollouts where objects disappear, teleport, or move without applied forces.

The key strengths include: (1) a novel contribution as the first multi-task visuo-tactile world model, (2) comprehensive real-robot evaluation across five contact-rich tasks demonstrating up to 35% improvement in zero-shot planning success rates, (3) principled evaluation of physical plausibility through object permanence and causal compliance metrics, and (4) data efficiency experiments showing VT-WM achieves 77% success versus 22% for behavioral cloning with limited demonstrations. The integration of established encoders (Cosmos for vision, Sparsh-X for tactile) with a transformer predictor is straightforward but effective.

However, there are notable limitations. First, the baseline comparison is limited to a vision-only ablation of the same architecture—comparisons to other state-of-the-art world models would strengthen the claims. Second, some improvements (wipe cloth and scribble tasks) don't reach statistical significance. Third, the CEM-based planning is computationally expensive and operates open-loop, limiting practical deployment. Fourth, generalization testing is restricted to held-out trajectories within training tasks, not truly novel objects or scenarios. Fifth, the dataset is modest (124 demonstrations), raising scalability questions. Finally, the BC comparison uses only a single-task policy rather than a multi-task baseline.

Overall, this is a solid contribution addressing an important gap in robotic world models. The empirical validation on real hardware is substantial, and the failure modes of V-WM versus VT-WM are convincingly demonstrated. While not field-advancing, it represents a meaningful advance for contact-rich manipulation.

Score: 7.5

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

The paper investigates an important and timely question: whether the complexity of GRPO's loss function is necessary for training LLMs to reason. Through systematic ablations, the authors find that (1) negative feedback is essential (positive-only training causes collapse), (2) PPO-style clipping is unnecessary, and (3) their proposed RGRA method—essentially REINFORCE with group-relative advantages—achieves slightly better performance than full GRPO.

The work has several strengths. First, the research question is highly relevant given GRPO's widespread adoption after DeepSeek-R1. The systematic ablation approach is methodologically sound, and the finding that clipping can be removed without degradation is practically useful for simplifying implementation. The empirical coverage across multiple benchmarks and model families (Qwen, Llama) provides reasonable evidence for the claims. The observation about training collapse with positive-only advantages highlights an important practical pitfall.

However, the paper has significant limitations. Most critically, experiments are conducted only on small models (0.5B-1.5B parameters), leaving open the question of whether findings generalize to the scales where GRPO has demonstrated its strongest results. The training data is remarkably small (1,800 samples), which may not adequately test the methods' behavior under typical RL training regimes. The improvements of RGRA over GRPO, while consistent, are modest—often within a few percentage points—and it's unclear whether equal hyperparameter tuning effort was applied to both methods. Additionally, some findings (e.g., that RL requires negative examples) are not particularly novel from a theoretical standpoint, and the paper doesn't deeply investigate why clipping is unnecessary beyond citing prior work.

The paper provides useful empirical insights but falls short of being a definitive contribution. The small-scale experiments and incremental improvements limit the impact, while the core insight—that GRPO can be simplified—was somewhat anticipated by prior work on REINFORCE for LLMs.

Score: 5.5

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper presents RoPE++, an extension to Rotary Position Embeddings that reintroduces the imaginary component discarded in standard RoPE implementations. The authors observe that while RoPE operates in the complex plane, only the real part is retained for attention scores—a potential information loss they characterize and address.

**Strengths:**
The paper identifies a genuine and previously overlooked aspect of RoPE's formulation. The mathematical analysis deriving characteristic curves for both real and imaginary attention components is well-executed, providing theoretical grounding for why imaginary attention captures different positional patterns—specifically showing that imaginary heads exhibit stronger global attention while real heads focus locally. The empirical work is thorough, testing multiple model scales (376M, 776M, 1.5B), comparing against multiple baselines (FoPE, ALiBi, Pythia), and evaluating both short and long-context benchmarks. The efficiency variants are practically motivated: RoPE++EH halves KV cache while maintaining performance, and the attention pattern visualizations confirm the claimed behavior differences. The method integrates cleanly with existing long-context techniques like YaRN and PI.

**Weaknesses:**
The improvements, while consistent, are modest—typically 0.5-2 points on average metrics. The paper only tests models up to 1.5B parameters, which limits conclusions about scaling to modern LLM scales. The theoretical connection between the sine integral characteristic curve and long-context performance, while plausible, remains somewhat intuitive rather than rigorously established. The method requires training from scratch, reducing immediate applicability compared to plug-and-play alternatives. Some experimental details could be clearer, particularly around exact computational overhead for RoPE++EC (which doubles output heads). The noise-injection experiment showing imaginary attention's importance is clever but shows only moderate effect differences (~5-8 points at σ=1.0).

**Overall:**
This is a well-motivated, carefully executed contribution that advances understanding of position embeddings. The insight about recovering imaginary information is genuine, the empirical validation is comprehensive, and the efficiency implications are practical. While not transformative, it represents solid incremental progress with clear value for the community.

**Score: 7.5**

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents GUI-Spotlight, a visual grounding model for GUI agents that iteratively refines focus using three specialized tools (crop, extract, find_color) trained via a three-stage pipeline with modified GSPO reinforcement learning. The approach achieves 52.8% accuracy on ScreenSpot-Pro with only 18.5K training samples, surpassing models trained on millions of samples.

**Strengths:**
- **Impressive data efficiency**: Training with 18.5K curated samples while outperforming V2P-7B (9.6M samples) and GTA-1-7B (1.56M samples) is a significant achievement. The rigorous data cleaning pipeline (IQ, BA, CON filtering) contributes meaningfully to this efficiency.
- **Novel training methodology**: The modified GSPO with auxiliary cross-entropy loss addresses RL instability in multi-turn tool-use scenarios—a practical contribution with documented negative results.
- **Well-motivated design**: The three tools serve distinct purposes (coarse quadrant extraction, color-guided focusing, fine-grained cropping), and the iterative "spotlight" metaphor naturally aligns with how humans might visually search interfaces.
- **Comprehensive evaluation**: Testing across ScreenSpot-Pro, OSWorld-G, and UI-Vision with comparisons against strong baselines provides convincing evidence of generalization.

**Weaknesses:**
- **Incremental novelty**: Iterative refinement with visual tools builds on existing ideas (e.g., UniVGR). The core contribution is applying this to GUI grounding with specific tool design rather than fundamentally new methodology.
- **Modest gains**: The 2.2% improvement over V2P-7B (52.8% vs 50.6%) is meaningful but not dramatic. The claim of "substantial" improvement should be tempered.
- **Missing analyses**: The paper lacks detailed analysis of tool usage patterns—how often each tool is invoked, failure modes, and computational overhead from multi-step inference.
- **Reward engineering complexity**: Five reward components with specific weights appears somewhat ad-hoc without strong theoretical justification. The complexity raises questions about reproducibility and generalization to other domains.

Score: 7.0

---

## DZUehXNiBn

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents VISTA, a modular framework for scalable causal structure learning that decomposes the global DAG learning problem into local Markov Blanket subgraphs, aggregates via weighted voting, and enforces acyclicity through Feedback Arc Set post-processing.

**Strengths:**
- **Clear motivation and timely contribution**: Scalable causal discovery remains a significant challenge, and the divide-and-conquer approach is well-motivated. The paper provides a principled alternative to existing ILP-based or heuristic merging schemes.
- **Theoretical grounding**: The coverage guarantee (Proposition 3.1), finite-sample error bounds (Theorems 3.2-3.5), and Bayesian interpretation of the weighted voting scheme provide solid theoretical justification. The analysis of λ's effect on precision-recall trade-offs is useful.
- **Model-agnostic design**: The framework works with arbitrary base learners (NOTEARS, DAG-GNN, GOLEM, SCORE, CAM), making it widely applicable without requiring modifications to underlying methods.
- **Computational efficiency**: The parallel decomposition and lightweight O(n²) aggregation yield substantial runtime improvements, as demonstrated empirically.
- **Comprehensive experiments**: Testing across multiple graph types (ER, SF), sizes (30-300 nodes), and base learners provides convincing evidence of effectiveness. The comparison with DCILP in the appendix is valuable.

**Weaknesses:**
- **Dependency on MB quality**: The framework heavily relies on accurate Markov Blanket identification, but MB estimation errors are not thoroughly analyzed. If the MB estimator fails, errors propagate directly to the final result.
- **Limited real-world validation**: Only the small Sachs dataset (11 nodes, 17 edges) is used for real data evaluation. This doesn't demonstrate scalability on practical benchmarks.
- **Hyperparameter selection**: While λ=0.5 and t=0.7 are fixed for experiments, the sensitivity analysis shows substantial variation in precision-recall. No guidance exists for choosing λ without ground truth knowledge.
- **Idealized independence assumptions**: Theorem 3.2 assumes independent votes across subgraphs, which is acknowledged as idealized. The practical implications of correlated subgraphs are not deeply characterized.
- **Undirected edges treatment**: Discarding undirected edges as "no vote" loses potentially useful information; a more nuanced treatment could improve performance.
- **FAS approximation quality**: The greedy heuristic for acyclicity enforcement lacks analysis of its impact on recovered DAG quality.

**Overall Quality:**
This is a solid contribution to causal discovery methodology. The weighted voting mechanism with theoretical guarantees is novel and well-executed. The framework is practical, computationally efficient, and demonstrates consistent improvements across baselines. However, the limited real-world validation, dependency on MB estimation quality, and practical hyperparameter selection issues prevent a higher score.

Score: 7.5

---

## x6bG2Hoqdf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces CALM (Co-evolution of Algorithms and Language Model), a framework for Automatic Heuristic Design (AHD) that jointly optimizes both the prompt generation process and the underlying LLM through reinforcement learning. The key insight is that existing LLM-based AHD methods rely solely on "verbal gradients" (prompt manipulation) while keeping the model fixed, missing the opportunity to adapt the LLM based on heuristic quality feedback.

**Strengths:**
The paper presents a novel contribution to LLM-based AHD by introducing numerical gradients via GRPO-based fine-tuning. The empirical results are compelling: a quantized 7B model running on a single 24GB GPU outperforms GPT-4o-mini baselines across OBP, TSP, CVRP, and OP tasks. The method is well-motivated, with careful design choices including fine-granularity operators (injection, replacement), diversity-aware crossover, a collapse mechanism for escaping local optima, and a reward function that attributes credit relative to base heuristics. The ablation studies are thorough, demonstrating the contribution of each component. The practical significance is notable—democratizing AHD research for those without access to expensive API-based models.

**Weaknesses:**
The contribution, while solid, is incremental in nature: GRPO is an existing algorithm, and the evolutionary framework builds on prior work like EoH and FunSearch. The paper acknowledges concurrent work (EvoTune) exploring similar directions. The evaluation scope is limited to four combinatorial optimization problems, and deeper analysis of what patterns the fine-tuned LLM learns would strengthen the paper. The 5-7 hour runtime per task, while feasible, still represents significant computational overhead.

**Overall:**
This is a well-executed contribution that advances LLM-based AHD by demonstrating the value of co-evolution. The joint optimization paradigm opens new research directions, and the practical benefit of achieving SOTA results with limited compute resources is meaningful. The paper has clear methodology, strong empirical validation, and thorough ablations.

Score: 7.5

---

## XKLPlnfZzM

- GT: Reject (avg 3.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces TDDM, a diffusion-based trajectory generation model that factors the problem into spatial occupancy priors and temporal dynamics. The key insight—separating "where" people move from "how" they move—is elegant and enables zero-shot generalization to new regions by conditioning only on aggregate spatial distributions rather than sample-specific statistics.

**Strengths:** The paper makes a strong contribution through its hierarchical problem decomposition. Conditioning on spatial priors (marginal distributions) rather than trajectory-level statistics enables cross-region generalization without retraining—a genuinely useful capability for urban mobility applications. The canonicalization via similarity transforms is well-motivated and allows a single model to serve multiple geographic regions. The empirical evaluation is comprehensive: three geographically diverse cities (Beijing, Porto, San Francisco), multiple baseline paradigms (GANs, VAEs, diffusion), and an extensive metric suite covering fidelity, distributional coverage, and usefulness. Results show consistent improvements, with KL divergences reduced by up to 4× compared to baselines. The ablation studies and generalization experiments (both intra-city and city-to-city) provide useful insights—particularly the finding that training on Porto generalizes better than limited local data.

**Weaknesses:** While the conceptual contribution is novel, the diffusion architecture itself is relatively standard—conditioning on additional modalities via tokens in transformers is well-established. The reliance on map matching during preprocessing limits applicability in regions lacking road network data, though the ablation shows the model still outperforms baselines without it. Some improvements are modest: TSTR improves only marginally (0.011 vs 0.013 for DiffTraj). The cross-city length error degradation (0.06–0.11 vs 0.004 in-distribution) reveals that fine-grained temporal statistics don't transfer—a limitation acknowledged but not deeply analyzed. Additionally, zero-shot transfer still requires computing spatial priors from the target region; this isn't truly "unconditional" generation.

Overall, this is a well-executed contribution addressing an important problem with clear practical relevance. The factorization approach and generalization capabilities advance the field meaningfully.

Score: 7.5

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents VINCIE, a novel approach to learning in-context image editing models directly from video data rather than curated image editing pairs. The core insight is that videos naturally contain sequential visual transitions that can serve as supervision for multi-turn editing—eliminating the need for task-specific data curation pipelines.

**Strengths:**

The paper makes a compelling conceptual contribution. The idea that video frames naturally encode editing-like transformations (object appearance/disappearance, pose changes, camera movements) is intuitive and well-motivated. By treating these natural transitions as training data, the authors unlock scalability benefits impossible with hand-curated pairs.

The technical approach is sound. The data construction pipeline (VLM annotation + SAM2 grounding for RoE extraction) and three proxy tasks (NIP, CSP, NSP) are well-designed. The insight that segmentation prediction tasks help the model learn grounding and reduce positional drift is validated through ablations.

Empirically, the results are strong. Training solely on video data, VINCIE achieves SOTA or competitive results on MagicBrush and significantly outperforms academic baselines on MSE-Bench (25% vs <2% success rate at turn-5). The scalability analysis (5% to 22% turn-5 success when scaling from 0.25M to 10M sessions) demonstrates the method's promise.

The new MSE-Bench benchmark addresses real limitations in existing multi-turn editing benchmarks by including 5-turn sessions with realistic editing categories and aesthetic considerations.

**Weaknesses:**

The reliance on an in-house proprietary MM-DiT video foundation model undermines reproducibility. While code is promised, the base model is not accessible, making it difficult for the community to build on this work.

The VLM annotation quality (75% accuracy in human evaluation) introduces noise that the paper doesn't thoroughly analyze. How does annotation error propagate through training? Is there a quality threshold below which data becomes harmful?

The gap with proprietary models (GPT-4o, Nano Banana) remains large (25% vs ~63% turn-5 success), which contextualizes the results but also highlights limitations.

Some comparisons are thin—the story generation comparison only includes In-context LoRA, and multi-concept composition evaluation is primarily qualitative.

**Overall:**

This is a solid contribution that opens a promising new direction for image editing research. The video-as-supervision paradigm could have broader implications beyond editing. The execution is competent, evaluation is reasonably comprehensive, and the writing is clear. The main limitation is reliance on proprietary infrastructure, which limits immediate community impact.

Score: 7.5

---

## W42oLSwI9p

- GT: Reject (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes three one-step diffusion-based solvers (CMILP, SCMILP, MFILP) for integer linear programming, extending neural ILP solvers beyond binary to non-binary integer problems. The key technical contributions include an Iterative Integer Projection (IIP) layer for handling general integer constraints without binary transformation, and a momentum-based objective-guided sampling scheme.

**Strengths:**
The extension to non-binary ILP is a meaningful contribution, as prior neural ILP work largely focused on binary cases. The IIP layer provides a differentiable mechanism for integer projection without the exponential blowup that binary encoding would cause. The experimental evaluation is reasonably comprehensive, covering multiple problem types (set cover, facility location, combinatorial auction, inventory management) and comparing against both traditional solvers and neural baselines. The speed improvements over vanilla diffusion methods (IP Guided DDPM/DDIM) are substantial, reducing inference from hours/minutes to seconds in many cases.

**Weaknesses:**
However, the solution quality remains concerning. The optimality gaps are often 10-15% or higher, which is significant for practical deployment. More problematically, dataset feasibility rates on non-binary problems are frequently below 90%, meaning the solver fails to produce any feasible solution for a non-trivial fraction of instances (e.g., 38% failure on IM-(50,5,10)). The adaptation of consistency models, shortcut models, and meanflow models to ILP is relatively straightforward, with IIP being the main novel technical contribution. The IIP layer lacks theoretical justification for why this particular projection form was chosen. Additionally, the momentum-guided sampling contribution is incremental—viewing guidance as gradient descent and adding momentum is a natural extension. The paper also lacks discussion of fallback strategies when infeasibility occurs, which is critical for practical deployment.

Score: 5.5

---

## U6ROetm5nW

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents novel algorithmic improvements for Kernel Density Estimation (KDE) in high dimensions by applying asymmetric locality-sensitive hashing (LSH). The key technical insight is leveraging the space-time tradeoff inherent in asymmetric LSH constructions (Andoni et al., 2017) to obtain the first known query-time versus space tradeoffs for KDE data structures.

**Strengths:**

The main technical contribution is both conceptually clean and quantitatively meaningful. By reformulating the Charikar et al. (2020) framework in terms of general (c,r)-ANN data structures, the authors can instantiate it with asymmetric LSH to obtain better tradeoffs. The improvement from 1/μ^{0.25} (data-independent LSH baseline) to 1/μ^{0.1865} at linear space is a meaningful improvement, and the tradeoff curve yielding query time 1/μ^{0.05} with polynomial space is impressive. The paper clearly positions its contributions relative to prior work and provides intuition for why the technique works—the maximum query time occurs at a different scale than the maximum space requirement, allowing asymmetric LSH parameters to improve one without proportionally penalizing the other.

The technical analysis is sound. The reduction to optimization over parameters ρ_q and ρ_s under the LSH constraint (c²+1)√ρ_q + (c²-1)√ρ_s ≥ 2c is correctly formulated, and the numerical evaluations appear carefully done. The observation that there's a fundamental plateau (query exponent ≈ 0.05) even with arbitrary space is interesting, though not fully explained.

**Weaknesses:**

The paper would benefit from deeper discussion of several points. First, the "plateau" phenomenon deserves more analysis—why can't we achieve constant query time with polynomial space? The paper mentions this briefly but doesn't prove impossibility. Second, while the improvement over data-independent LSH is clear, the comparison to data-dependent LSH (0.173 vs 0.1865) is somewhat unfavorable; the paper's main defense is "simpler analysis," which is valid but less compelling. Third, the reliance on numerical optimization for parameter settings reduces the elegance somewhat—can any special cases be solved analytically? Finally, the paper focuses exclusively on the Gaussian kernel; generalization results would strengthen the work.

The writing is dense but reasonably clear for a theory paper. The appendix contains necessary technical details. The Python script for reproducing numerical results is a nice touch for reproducibility.

Score: 7.0

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces CaTS-Bench, the first large-scale multimodal benchmark for context-aware time series captioning. The work addresses a genuine gap in the literature: existing TSC benchmarks rely on synthetic data, overly simplistic pattern-based captions, or lack metadata and visual representations.

**Strengths:**
The benchmark is substantial in scope—spanning 11 diverse real-world domains with 570k timesteps—and thoughtfully designed with multiple modalities (numeric series, metadata, plots, captions). The quality validation is thorough: manual verification of ~2.9k captions shows 98.6% factual accuracy; a human detectability study (35 participants, 41.1% accuracy) demonstrates that oracle-generated captions are indistinguishable from human-written ones; and diversity analyses confirm low template reliance. The paraphrasing experiments across multiple LLM architectures provide robustness validation against oracle-specific bias. Beyond captioning, the inclusion of 460 multiple-choice Q&A tasks enables fine-grained capability assessment. The proposed numeric fidelity metrics (Statistical Inference Accuracy, Numeric Score) are valuable contributions that address limitations of standard NLP metrics for this task.

The most interesting empirical finding is that VLMs largely fail to leverage visual inputs for TSC, instead defaulting to textual priors—supported by both modality ablation and attention analysis. This has implications beyond time series, highlighting fundamental limitations in current multimodal architectures.

**Weaknesses:**
The core limitation is reliance on semi-synthetic ground truth. While quality validation is extensive, the majority of captions are LLM-generated, not human-authored. The human-revisited subset (579 samples) is relatively small compared to the full benchmark. Though the paraphrasing experiments show robustness to linguistic style variation, using a single oracle model (Gemini 2.0 Flash) could still introduce subtle biases. The Q&A task filtering methodology (removing questions answered by Qwen 2.5 Omni) risks biasing toward model-specific weaknesses. Additionally, while the findings about visual modality underutilization are important, the paper stops short of proposing solutions—alternative visual encodings (GAF, RP) were tested but underexplored.

**Overall:**
This is a solid benchmark contribution that fills a clear gap and provides meaningful insights about VLM limitations. The validation methodology is rigorous, and the dataset will be valuable for the community. The semi-synthetic ground truth is an acknowledged limitation but does not undermine the work's value.

Score: 7.0

---

## MwuSvrthXq

- GT: Reject (avg 4.0)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes WeCAN, a reinforcement learning framework for heterogeneous DAG scheduling with task-resource compatibility constraints. The main contributions include a weighted cross-attention mechanism to incorporate compatibility coefficients, theoretical analysis of optimality gaps in list scheduling, and a skip-action mechanism for the single-pass inference setting.

**Strengths:**
The weighted cross-attention design thoughtfully integrates compatibility coefficients into the attention mechanism while maintaining adaptability to varying numbers of pools and task types. The theoretical analysis correctly identifies that list scheduling cannot always reach optimal solutions, particularly when heavy tasks require waiting for resources—a counterexample clearly demonstrates this gap. The skip-action mechanism enables the single-pass architecture to overcome this limitation without requiring expensive multi-round inference. Empirical results are strong across TPC-H and Computation Graphs datasets, with consistent improvements of 7-18% over heuristics and 5-9% over neural baselines. The computational efficiency is notable: greedy inference achieves runtimes comparable to heuristics while outperforming prior neural schedulers by substantial margins.

**Weaknesses:**
The skip-action contribution, while theoretically sound, appears to have limited practical impact—Appendix H.3 explicitly states it's disabled for regular TPC-H experiments. Its benefits are demonstrated primarily on artificially constructed "heavy task" scenarios, raising questions about real-world applicability. The non-auto-regressive design sacrifices modeling of dynamic state evolution; the paper argues this has minimal impact, but the empirical comparison in Appendix B shows the AR variant performs comparably or slightly better, suggesting this may not be a principled design choice. The architectural components (LDDGNN, weighted cross-attention) are relatively straightforward modifications of existing attention mechanisms. Some baselines use modified implementations (OneShot uses Graphormer instead of Topoformer), complicating direct comparisons. The evaluation would benefit from naturally occurring heavy-task scenarios rather than synthetic modifications.

**Overall Quality:**
This is a solid, well-executed contribution that advances heterogeneous DAG scheduling. The combination of theoretical insight, efficient architecture, and strong empirical results represents meaningful progress. However, the limitations of the skip action's practical impact and the incremental nature of the architectural innovations prevent a higher rating.

Score: 7.2

---

## 31CznLfRIS

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces VideoJudge, a bootstrapping framework for training multimodal LLMs to evaluate video understanding model outputs. The work addresses an important and timely problem—scalable evaluation of video understanding systems without costly human annotation.

**Strengths:** The paper makes several meaningful contributions. First, the generator-evaluator pipeline for bootstrapping training data is clever and practical, enabling creation of over 100K training examples without human labels. Second, the empirical results are impressive: VideoJudge-7B matches or outperforms models 10× larger (Qwen2.5-VL-32B/72B) on several benchmarks, demonstrating the value of specialized training. Third, the rubric-generation capability is a nice addition that improves interpretability, and the ablation studies on temperature and frame counts provide useful practical insights. The paper is well-written and includes comprehensive experiments across multiple benchmarks.

**Weaknesses:** Several concerns weaken the paper. The most significant is the potential circularity in evaluation: both training data and some meta-evaluation benchmarks (VideoJudgeLLaVA-MetaEval, VideoJudgeVCG-MetaEval) are constructed using the same generator-evaluator pipeline, which could inflate reported performance. While external benchmarks (VATEX, VideoAutoArena) partially address this, they represent a minority of the evaluation suite. Second, the generator and evaluator models used in bootstrapping are not clearly specified—the paper mentions "strong vision-language models" but lacks this critical implementation detail. Third, the error analysis reveals substantial calibration issues: 14.8% of cases have ≥2-point overestimation, and 81.3% of rating-4 responses are incorrectly scored as 5, suggesting the model struggles with fine-grained distinctions. Finally, human validation is limited to only 250 pairwise comparisons, insufficient to fully validate data quality across the diverse training set.

**Overall:** This is a solid contribution addressing a real gap in video understanding evaluation, with practical methodology and encouraging results. However, the evaluation methodology has potential circularity issues, and calibration problems suggest limitations in the current approach. The work would benefit from clearer documentation of the bootstrapping components and more rigorous separation between training and evaluation data.

Score: 6.5

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important and timely problem in LLM post-training: the mismatch between FSDP's fine-grained collective communication (which assumes balanced workloads) and the inherently imbalanced workloads arising from variable sequence lengths. The core insight—revisiting parameter server concepts to relax synchronization from per-layer to per-minibatch—is elegant and well-motivated.

The technical approach is sound. By replacing all-gather/reduce-scatter with point-to-point RDMA-based primitives, ODC decouples device execution while preserving FSDP's memory efficiency. The implementation using CUDA IPC and NVSHMEM is practical, and the integration with FSDP appears straightforward. The enabling of minibatch-level load balancing (versus microbatch-level) is a natural benefit that the paper correctly identifies and exploits.

The evaluation is comprehensive, covering multiple tasks (SFT, RL), model scales (1.5B-32B), datasets with varying skew, and thoughtful ablation studies. The reported speedups (up to 36%) are significant, and the bubble rate analysis helps explain where gains come from. The convergence verification is a welcome inclusion for a systems paper modifying training dynamics.

However, several limitations temper enthusiasm. First, Figure 11 shows that ODC's point-to-point communication significantly underperforms NCCL collectives for inter-node transfers. While the paper argues computation can hide this latency (particularly for long sequences), this is a notable constraint that limits applicability. Second, the scale of experiments (max 32 GPUs) is modest for LLM training; larger-scale validation would strengthen claims. Third, the benefits are highly dependent on workload imbalance—when minibatch size equals 1, gains vanish entirely, indicating the approach has narrow applicability.

The paper would benefit from deeper discussion of failure modes and edge cases. For instance, how does ODC interact with gradient checkpointing, mixed precision training, or pipeline parallelism? The hybrid sharding mitigation for inter-node inefficiency is mentioned but not thoroughly evaluated.

Overall, this is a solid contribution that identifies a real problem and proposes a practical solution. The core idea of adapting parameter server principles to modern sharded DP is valuable, and the results demonstrate meaningful improvements for the target workloads.

**Score: 7.5**

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper makes an important contribution to the emerging field of online map based motion prediction by identifying and correcting several methodological flaws in existing protocols. The authors provide a systematic critique of current practices and propose OMMP-Bench as a solution.

The key strengths include: (1) identifying the train-validation gap problem caused by training online mapping models on the same data used to generate training inputs for motion prediction—this is a legitimate methodological flaw that could lead to over-optimistic results; (2) recognizing the mismatch between the limited perception range of online mapping models and the broader requirements of motion prediction for distant agents; and (3) correctly pointing out that evaluating only ego vehicle trajectories misses the core purpose of motion prediction. The proposed image feature baseline using deformable attention is a reasonable approach to address the out-of-range agent problem, demonstrating meaningful improvements (e.g., 12.7% minADE reduction for distant agents).

However, the paper has notable limitations. The technical novelty is modest—the deformable attention mechanism for feature aggregation is straightforward and not particularly innovative. The primary contribution is essentially dataset reorganization and evaluation protocol correction rather than novel methodology. While benchmark papers are valuable, the methodological advances here are incremental. Additionally, the paper could benefit from exploring alternative solutions to the identified problems rather than focusing primarily on a single baseline approach. The analysis of different map element types (Table 5) is interesting but somewhat shallow.

Overall, this is a solid benchmark paper that addresses real problems in an emerging field. While not groundbreaking in terms of technical innovation, it provides a valuable service to the community by establishing proper evaluation standards. The issues identified are substantive, and the proposed corrections are well-motivated and thoroughly validated.

Score: 7.0

---

## CQ0U1wZYoy

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents PRISM, a conditional diffusion framework for handling compound (multiple overlapping) degradations in scientific imagery. The key innovation is a weighted contrastive learning objective that creates a compositional latent space, enabling both joint restoration of mixed degradations and selective removal of specific distortions through natural language prompts.

**Strengths:** The paper addresses an important and underexplored problem. While most restoration work focuses on single degradations, real-world scientific images suffer from compound effects (underwater scattering + low light, clouds + haze in satellite imagery). The weighted contrastive objective using Jaccard distance to align primitives and their mixtures is well-motivated theoretically and creates genuinely useful compositional structure. The evaluation is comprehensive, spanning microscopy, wildlife monitoring, remote sensing, and urban scenes. Crucially, the paper introduces downstream scientific utility metrics (species classification accuracy, segmentation IoU, fluorescence intensity) rather than relying solely on perceptual quality metrics. The finding that selective restoration outperforms "restore everything" in 3/4 domains is an important contribution—it shows controllability is not merely convenient but necessary for scientific precision.

**Weaknesses:** The method relies entirely on synthetic degradations for training, creating a potential sim-to-real gap that the authors acknowledge but don't fully address. The computational cost of diffusion models remains substantially higher than encoder-decoder alternatives. The architectural contributions are incremental—building on Stable Diffusion v1.5 and fine-tuned CLIP—with novelty primarily in the training objective. The prompt sensitivity analysis (Appendix E) is limited, and real-world prompt variability could affect practical deployment. The downstream evaluation uses off-the-shelf models, which may conflate restoration quality with model-specific biases.

**Overall:** This is a solid contribution that addresses real needs in scientific imaging. The compositional latent space design is principled, empirical results are convincing across diverse domains, and the focus on scientific utility over perceptual quality is refreshing. While not revolutionary, it meaningfully advances compound degradation handling.

Score: 7.5

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a novel theoretical contribution bridging two previously disconnected areas in ML interpretability: activation steering and influence functions. The central claim—that these techniques are equivalent to first order—is both mathematically rigorous and practically interesting. The paper provides proper theorem statements with proofs, including the steering-influence equivalence theorem, alignment bounds, and a no-free-lunch result characterizing when steering cannot replicate influence.

The strengths of this work lie in its clean mathematical formulation. The primal-dual formulation connecting parameter and activation spaces is elegant, and the $\omega(x)$ diagnostic offers a principled way to decide when steering will succeed versus when weight-space editing is necessary. The computational framework relying only on Jacobian-vector products and small pseudoinverses is practical and scalable.

However, the empirical validation has notable weaknesses. The experiments are limited to GPT-2 Medium (a relatively old model) on a single detoxification task, with only 500 evaluation prompts. The comparison with Contrastive Activation Addition is minimal, and there's no comparison with modern influence estimation methods (TracIn, Representer Point selection) or weight-space editing approaches (ROME, MEMIT) despite discussing them. The spectral optimality experiment on ImageNet—testing a single class against random directions—provides thin evidence for Theorem 5.3's practical utility. Additionally, the paper doesn't demonstrate the claimed causal training-example identification from Corollary 1 in practice; a case study showing retrieved training examples would substantially strengthen the practical claims.

The first-order approximation limitation, while acknowledged, deserves deeper empirical characterization—when does it break down in realistic steering scenarios? Despite these limitations, the theoretical contribution is substantial: the equivalence theorem and alignment geometry genuinely advance our understanding of how activation-space and weight-space interventions relate.

Score: 6.5

---

## eETr3lrOQB

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes **VQ-Transplant**, a framework for efficiently integrating new vector quantization modules into pre-trained visual tokenizers without requiring costly end-to-end retraining. The key technical approach is a two-stage process: (1) VQ module substitution with frozen encoder-decoder, and (2) lightweight decoder adaptation (5 epochs). The paper also introduces **MMD-VQ**, a new quantization method using Maximum Mean Discrepancy for distributional alignment.

**Strengths:**
The paper addresses a practically important problem—training state-of-the-art tokenizers like VAR requires substantial computational resources (hundreds of GPU hours), which limits research accessibility. The empirical results are compelling: VQ-Transplant achieves 21.8× speedup while matching or exceeding baseline performance (MMD-VAR achieves 0.81 r-FID vs. VAR's 0.92). The cross-dataset generalization experiments (FFHQ, CelebA-HQ, LSUN-Churches) demonstrate robustness. The comparison between MMD-VQ and Wasserstein-VQ on non-Gaussian synthetic data provides useful theoretical insight.

**Weaknesses:**
The core idea of replacing VQ modules and adapting the decoder is relatively straightforward. The MMD-VQ contribution, while technically sound, applies a well-known technique to VQ without major innovation—on real benchmarks where features are approximately Gaussian, differences from Wasserstein-VQ are minor. More critically, the paper lacks evaluation on downstream generation tasks; reconstruction fidelity alone doesn't guarantee good token quality for image generation. The efficiency comparison omits the initial pre-training cost amortization. A simple fine-tuning baseline (jointly updating all components) would strengthen the comparison, though the appendix partially addresses this.

Overall, this is a solid practical contribution that will help democratize VQ research, but the methodological novelty is incremental and downstream evaluation is missing.

Score: 7.2

---

## vGkXf8nvt9

- GT: Reject (avg 4.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper presents Forget-to-Focus (F2F), a two-stage protocol that first unlearns general knowledge before domain-specific fine-tuning. The central insight—that removing irrelevant pretraining priors can improve specialization—is novel and well-motivated. The paper reframes machine unlearning from a privacy tool to a principled intervention for domain adaptation, which is an interesting conceptual contribution.

The strengths are substantial. First, the empirical evaluation is comprehensive: experiments span three domains (medical, coding, mathematics), multiple model architectures (Qwen, LLaMA, Gemma), and scales from 0.6B to 72B parameters. The reported improvements are significant, e.g., HumanEval pass@1 improving from 19.50 to 42.07 on Qwen-0.6B. Second, the paper provides mechanistic insights through CKA, SVCCA, Fisher information, and PCA-shift analyses, showing that unlearning reshapes representational geometry away from generalist features. Third, the comparison to multiple baselines (SFT, DAPT, LoRA, CurlLoRA) and various unlearning methods (GA, GA+GD, NPO, GA+KL) is thorough.

However, several weaknesses temper my enthusiasm. The selection of BookCorpus as the forget set is somewhat arbitrary—the rationale for why this corpus contains "irrelevant" knowledge for medical/coding/math domains is thin. While BC-Select improves over BC-Mixed, the paper doesn't provide concrete examples of what spurious correlations are being removed. Second, some baseline comparisons appear suboptimal: for instance, Gemma-2B's HumanEval drops from 16.46% to 11.30% with LoRA, suggesting potential hyperparameter issues. Third, the theoretical analysis relies on strong assumptions (linear models, convex losses, orthogonal feature decomposition) that may not transfer to LLMs. Fourth, the method can be unstable—GA-only unlearning causes catastrophic drops (LLaMA 8B HumanEval to 1.20), requiring careful tuning of the retain weight σ. Finally, while the gains are impressive on some settings, others show modest improvements (Qwen 72B HumanEval: 71.12 → 78.50), and the variance across settings isn't well explained.

**Score: 7.0**

---

## VgVeQpagf7

- GT: Reject (avg 4.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents SPS (Summarize-Privatize-Synthesize), a novel approach to differentially private machine learning that generates DP synthetic datasets via dataset distillation. The key insight is to privatize low-dimensional activation statistics from a public pretrained model rather than high-dimensional gradients, enabling more efficient noise allocation.

**Strengths:**

The empirical results are impressive: SPS+ achieves 96.2%/76.6% accuracy on CIFAR-10/100 at ε=1, outperforming DP-SGD baselines (94.8%/70.3%). This appears to be the first generation-based DP method to match or exceed gradient-based approaches on image classification, which is a meaningful contribution. The technical innovations—particularly grouped pseudo-classes (GPC) for handling multi-class scenarios and multistage clipping (MC) for high-privacy regimes—are well-motivated and provide measurable improvements. The approach enables practical benefits that DP-SGD cannot provide: model ensembling, federated learning, and continual learning without additional privacy composition. The writing is clear and the experiments are reasonably comprehensive, including ablations on key design choices.

**Weaknesses:**

The method depends critically on having a suitable public pretrained model, and experiments are limited to relatively small-scale datasets (CIFAR-10/100, CAMELYON17). No evaluation on larger datasets like ImageNet or with modern architectures (ViTs) is provided. The grouped pseudo-classes technique lacks theoretical justification beyond "optimization dynamics," which is unsatisfying. Computational cost is significant (8-21 hours on H100 GPUs), comparable to DP-SGD. The paper has numerous hyperparameters (K_clip, D_G, D_C, λ_C, M, P, N_c/p) that may require tuning, though the authors argue for robustness. Some baseline comparisons are relegated to appendices rather than main text.

**Overall:**

This is a solid contribution to DP ML. Beating DP-SGD on standard benchmarks with a generation-based approach that offers additional flexibility is notable. The work opens promising directions for practical private ML. While the scope is limited, the core ideas are sound and well-executed.

Score: 7.5

---

## NFB4QGGS65

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper establishes a compelling theoretical connection between GPTQ (a widely-used LLM quantization method) and Babai's nearest plane algorithm from lattice theory. The authors prove that GPTQ executed back-to-front is mathematically equivalent to solving a closest vector problem (CVP) on the lattice defined by the Hessian of layer inputs.

**Strengths:**
The theoretical contribution is substantial and non-obvious. The equivalence proof is rigorous, and the geometric interpretation provides intuitive insight into why GPTQ's greedy local updates work well globally—the algorithm is effectively performing hyperplane projections in a carefully constructed lattice space. This connection enables importing error bounds from classical lattice theory, yielding tight layer-wise quantization error guarantees in the no-clipping regime.

The practical applications demonstrate the theory's value: the proposed HPTQ and SSQR methods avoid clipping violations and achieve competitive or superior results compared to GPTQ, particularly at aggressive bitwidths (2-3 bits). The CUDA kernel implementation for SSQR shows practical engineering effort, achieving ~2× speedup over BF16 baselines in the low-batch inference regime. The experimental evaluation is thorough, covering multiple model families (Qwen3, Llama) and bitwidths, with comparisons to recent SOTA methods like AQLM, QuIP#, and QTIP.

**Weaknesses:**
The theoretical guarantees only apply in the no-clipping setting, though the authors correctly note that modern formats (MXFP4, NVFP4) effectively operate in this regime due to small group sizes. The practical improvements over standard GPTQ at 4-bit are modest; the more significant gains appear at 2-3 bits where vanilla GPTQ degrades severely. The min-pivot ordering heuristic derived from theory offers limited practical benefit over the simpler act-order. A concurrent work (Birnick 2025) apparently reached similar conclusions, which slightly diminishes novelty claims.

**Overall:**
This paper makes a genuine and valuable contribution to understanding LLM quantization from a principled theoretical perspective. The lattice-based interpretation opens new avenues for importing classical algorithmic ideas into neural network compression. The work bridges theory and practice effectively—rare in this space.

Score: 7.5

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces a novel framework of "random set stability" for deriving worst-case generalization bounds over data-dependent random sets produced by stochastic optimization algorithms. The work addresses a significant limitation in prior topological generalization bounds (Simsekli et al., Birdal et al., Andreeva et al.), which relied on intractable mutual information terms.

**Strengths:**
1. The random set stability concept meaningfully extends classical algorithmic stability to handle entire trajectories while properly accounting for algorithmic randomness—a genuine improvement over Foster et al.'s hypothesis set stability.

2. The derived bounds successfully eliminate the intractable mutual information terms while still capturing topological complexity. The framework elegantly recovers classical results: J=1 recovers algorithmic stability bounds, while J=n recovers standard Rademacher complexity bounds.

3. Corollary 3.3 provides concrete stability bounds for projected SGD under standard Lipschitz/smooth assumptions, demonstrating practical applicability.

4. The theoretical analysis is rigorous, with clear connections between the proposed framework and existing stability notions via Lemma 3.2.

5. Experiments on ViT and GraphSAGE validate the theory and investigate the interplay between stability and topological complexity, showing that smaller stability parameters correlate with better generalization.

**Weaknesses:**
1. The bounds are quite loose—approximately an order of magnitude larger than actual generalization gaps (Table 1). However, this is typical for theoretical generalization bounds.

2. The paper only provides expected bounds, not high-probability bounds. This limitation is acknowledged but limits practical applicability.

3. The local Lipschitz assumption (Assumption 4.1) may not hold generally for neural networks, restricting the scope of application.

4. The empirical estimation of β_n is necessarily optimistic (using 500 held-out samples rather than the true supremum), though the authors acknowledge this.

5. The O(β_n^{1/3}) convergence rate is slower than classical O(n^{-1/2}) rates—the authors justify this as a trade-off for avoiding infinite IT terms, but it does represent a cost.

Overall, this is a solid theoretical contribution addressing a real gap in the topological generalization bounds literature. The framework is well-motivated, properly developed, and empirically supported. While not groundbreaking, it represents meaningful progress in making topological bounds computable.

Score: 7.0

---

## piylyBPSau

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes GenCoGS, a 3D Gaussian Splatting-based method for few-shot novel view synthesis that incorporates generative completion strategies for both Gaussian initialization and optimization. The work addresses a legitimate and important problem: existing few-shot NVS methods struggle to represent unobserved scene regions due to their reliance solely on observed training views.

**Strengths:**
The paper clearly identifies a key limitation in existing few-shot 3DGS methods—incomplete scene representation leading to floating artifacts and missing details in unobserved regions. The proposed two-pronged approach (GCGI for initialization, GCGO for optimization) is conceptually coherent. The generative point cloud completion strategy, which generates complementary points and filters outliers using kd-Tree proximity, provides a reasonable mechanism for improving initial coverage. The use of an I2V diffusion model with perturbed camera trajectories to generate pseudo views for unobserved regions, combined with a hallucination-mitigating consistency loss, shows careful design consideration. Experimental results demonstrate consistent improvements across LLFF, DTU, and Shiny datasets, with notable gains on the challenging Shiny dataset (1.47 dB PSNR improvement over baselines). The ablation studies are comprehensive, showing the contribution of each component.

**Weaknesses:**
The technical contributions are largely incremental. The point cloud completion module relies on existing architectures (DGCNN, Transformer, FoldingNet) with a relatively simple heuristic filtering scheme based on nearest-neighbor distances. The GCGO strategy builds directly on ViewCrafter's I2V diffusion model, with the main additions being perturbed camera trajectories and a consistency loss—both straightforward extensions. The method introduces significant complexity and computational overhead (40 min training vs. 30 min for BinoGS) while relying on multiple pre-trained models, raising reproducibility concerns. The extensive hyperparameter tuning (six key parameters: δ₁, δ₂, δ₃, A, f, α, β) suggests potential overfitting to the evaluation benchmarks, and the heuristic nature of these design choices (sin-based trajectory perturbation, Gaussian blur for confidence masks) lacks strong theoretical justification. Additionally, while improvements are consistent, they are relatively modest on some datasets (0.55-0.74 dB on LLFF), and the paper provides limited discussion of failure cases or the method's limitations beyond computational efficiency.

Score: 5.5

---

## Iq1fNZus2W

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper proposes PKA (Patch-wise and Keyword-Aware Attention), an efficient attention mechanism for multi-condition control in Diffusion Transformers. The key insight is that standard "concatenate-and-attend" approaches have unnecessary computational overhead, which the authors address through two specialized modules: Position-Aligned Attention (PAA) for spatial conditions (reducing complexity from O(N²) to O(N) via one-to-one alignment) and Keyword-Scoped Attention (KSA) for subject-driven conditions (using attention masking to focus on relevant regions). The paper also introduces an early-timestep sampling strategy for efficient fine-tuning.

**Strengths:**
The paper makes a compelling and well-motivated contribution to an increasingly important problem. The empirical analysis of attention sparsity patterns in multi-condition DiTs is insightful and provides a principled foundation for the proposed approach. The efficiency gains are substantial—up to 10× speedup and 5.12× memory reduction—making multi-condition generation far more practical. The evaluation is reasonably thorough, with quantitative metrics (FID, SSIM, CLIP-I, DINOv2, controllability measures) across multiple tasks and informative ablations on each component. The Condition Cache mechanism leveraging the fact that condition tokens only perform self-attention is a clever practical optimization.

**Weaknesses:**
Several limitations temper my enthusiasm. First, the baseline comparison is limited to only OminiControl2 and UniCombine; including more diverse approaches or discussing why ControlNet-style methods weren't considered for DiTs would strengthen the work. Second, the method is only validated on FLUX.1, leaving generalizability to other DiT architectures unclear. Third, PAA's one-to-one spatial alignment assumes perfect correspondence between conditions and latents, which may not always hold—potential failure modes aren't discussed. Fourth, the KSA threshold ε (default 0.2) introduces a hyperparameter requiring tuning, though the ablation shows reasonable robustness. Finally, some qualitative improvements over baselines appear marginal despite quantitative gains.

**Overall:**
This is a solid, practical contribution addressing a real computational bottleneck in multi-condition DiTs. The insight that different condition types exhibit distinct attention sparsity patterns is valuable, and the proposed solutions are elegant. While not groundbreaking, the work offers meaningful efficiency improvements with maintained quality.

Score: 7.5

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents an important negative finding for the mechanistic interpretability community: commonly used SAE evaluation metrics, particularly auto-interpretability scores, often fail to distinguish between trained transformers and randomly initialized ones. The authors systematically compare SAEs trained on Pythia models (70M to 6.9B parameters) across multiple randomization schemes, finding that aggregate auto-interpretability AUROC scores are remarkably similar between trained and randomized variants.

The key strengths of this work include: (1) the rigorous experimental design with multiple randomization schemes and control conditions, (2) comprehensive evaluation across model scales with appropriate robustness checks (hyperparameters, training data, random seeds), (3) the proposed token distribution entropy metric as a proof-of-concept for measuring feature "abstractness" which successfully distinguishes trained from random models, and (4) clear practical recommendations for the field to include randomized baselines in SAE evaluation.

However, the paper has notable limitations. The toy model section (Section 4) is preliminary and doesn't provide conclusive mechanistic explanations for why random networks produce interpretable features—it shows plausibility rather than proof. The token distribution entropy metric, while promising, is a relatively crude proxy for abstractness and may miss important dimensions of what makes features "computationally relevant." Additionally, testing only Pythia models limits generalizability, and the paper doesn't fully engage with the philosophical question of whether interpreting the structure preserved by random networks is inherently meaningless or simply interpreting a different (simpler) computation.

Nevertheless, the core finding is significant and timely. As SAE research accelerates, this work serves as an important corrective—demonstrating that current evaluation practices are insufficient and prompting the development of more rigorous metrics. The paper is well-written with extensive appendix materials that support reproducibility. While not a breakthrough methodological contribution, it provides essential empirical grounding that the field needs.

Score: 7.0

---

## c7OsKOOZo8

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes an end-to-end framework for multi-view diabetic retinopathy grading that generates lesion proposals internally rather than relying on costly external annotations. The approach consists of two main contributions: GALP (Grade-Activated Lesion Proposals), which uses auxiliary classifiers to derive grade-conditioned evidence maps for selecting lesion-relevant regions, and LGRF (Lesion Expert-Guided Regional Fusion), which employs mixture-of-experts routing and Top-K weighted cross-view attention for selective fusion.

**Strengths:**
The paper addresses a genuine clinical limitation—the dependency of prior methods on expensive vessel or lesion annotations that break end-to-end training and create brittleness. The technical approach is well-motivated and the combination of CAM-based proposal generation with MoE-guided cross-view fusion is sensible. Empirically, the method achieves competitive results without external annotations (83.9% accuracy on MFIDDR, matching or exceeding several externally-informed baselines), and establishes new SOTA when lesion information is incorporated (84.6%). The ablation studies confirm both GALP and LGRF contribute meaningfully, and hyperparameter analysis provides useful insights.

**Weaknesses:**
Several aspects limit the contribution. First, the novelty is incremental—CAM-based attention and MoE routing are well-established techniques; the contribution lies primarily in integration. Second, the paper lacks qualitative validation that the GALP proposals actually correspond to lesions; CAM-based localization is known to be noisy, and without visualization or correlation analysis with ground-truth lesions, the "lesion proposal" characterization remains unverified. Third, the method introduces multiple hyperparameters (retention ratio, expert counts, patch sizes) that differ between datasets, raising generalization concerns. Fourth, computational overhead is not discussed despite adding auxiliary classifiers and MoE routing at multiple stages. Finally, while the "with lesion" variant demonstrates integration capability, it somewhat contradicts the paper's central premise of annotation-free inference.

**Overall:**
This is a solid contribution addressing an important practical problem with reasonable technical solutions and strong empirical results. While not groundbreaking, it represents meaningful progress in reducing annotation dependency for multi-view medical image analysis.

Score: 7.2

---

## rI2Fa13fUL

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Generative Trajectory Policies (GTP), a new policy class for offline reinforcement learning that unifies several modern generative modeling approaches under a continuous-time ODE framework. The authors show that diffusion models, consistency models, flow matching, and related approaches can all be viewed as learning the solution map of an ODE trajectory.

**Strengths:**
The paper makes a genuine conceptual contribution through its unified ODE framework. The insight that diffusion, consistency models, CTMs, Shortcut Models, and Mean Flows all approximate aspects of the same flow map Φ provides clarity to a fragmented literature. The theoretical foundation is solid: Theorem 1 rigorously shows that the proposed score approximation introduces only O(h^p) error, and Theorem 2 correctly derives the advantage-weighted objective from KL-regularized policy optimization.

Empirically, the results are impressive. GTP achieves state-of-the-art performance on D4RL benchmarks, with particularly notable gains on challenging AntMaze tasks (perfect scores on some variants). The BC-only experiments convincingly demonstrate the expressiveness of the policy architecture. The ablations on score approximation and variational guidance provide meaningful validation of the design choices.

**Weaknesses:**
The technical novelty is somewhat incremental. The score approximation technique closely resembles consistency training's use of straight-line paths, and the advantage weighting is standard in offline RL (similar to AWAC). While the synthesis is valuable, the individual components are not novel.

The method shows hyperparameter sensitivity: Table 4 reveals different learning rates, η values, and gradient norms across tasks, suggesting per-task tuning may be needed. Some baseline comparisons are incomplete (missing values for BDM and C-AC on AntMaze). The training time remains substantial (4-6 hours), which the paper acknowledges but doesn't fully address.

The evaluation is limited to D4RL locomotion tasks; testing on robotic manipulation or other domains would strengthen the claims about expressiveness.

**Overall:**
This is a solid, well-executed contribution that advances generative policies for offline RL. The unified framework provides genuine insight, the practical techniques work well, and the empirical results are strong. While individual components draw on existing work, the synthesis is coherent and the total contribution exceeds the sum of parts.

Score: 7.5

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

# Review of "Active Learning for Flow Matching Model in Shape Design: A Perspective from Continuous Condition Dataset"

## Assessment

This paper addresses an important and relatively unexplored problem: applying active learning techniques to generative models (specifically flow matching models) rather than the more commonly studied direction of using generative models to improve active learning for discriminative tasks. The application to shape design with expensive numerical simulation labels is practically relevant and well-motivated.

The paper's main strength lies in its theoretical framework based on piecewise-linear neural networks and closed-form flow matching models. The insight that data with identical labels primarily enhance diversity while data with distinct labels improve accuracy provides a useful conceptual framework for understanding the diversity-accuracy trade-off from a data-centric perspective. The proposed query strategies (QD for diversity, QA for accuracy, and a weighted hybrid) are intuitive and grounded in this analysis. The experiments across synthetic and real-world shape design datasets demonstrate that the proposed strategies outperform classical active learning methods designed for discriminative models.

However, several weaknesses limit the paper's impact. **First, the theoretical analysis relies on very strong assumptions** - specifically that neural networks exhibit piecewise-linear behavior and that closed-form flow matching models accurately represent practical flow matching. The paper provides limited justification for how well these assumptions hold in practice, and the gap between theory and practice is not adequately addressed. **Second, the experimental evaluation has notable limitations**: the comparison focuses on classical AL methods for discriminative models but excludes more recent approaches designed for generative models; the diversity metric (average pairwise Euclidean distance) is quite simplistic; and the domain is restricted to shape design without demonstration of broader applicability. **Third, the use of RBF networks to predict labels for unlabeled data introduces approximation errors that could significantly affect the query strategies' effectiveness, yet this source of error is not analyzed.**

Additionally, while the paper claims to explain the "fundamental diversity-accuracy trade-off," the proposed strategies are essentially coreset-like methods operating on label space, with the main novelty being the adaptation to the generative context rather than fundamental algorithmic innovation. The decoupling of the query process from model training, while computationally efficient, may miss important model-data interactions that could improve selection.

Score: 5.5

---

## Mz98kwANpF

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents a thought-provoking challenge to the prevailing paradigm in multi-task LoRA architectures. The authors make three key empirical observations: (1) M-LoRA, a simplified multi-head variant with high inter-head similarity, outperforms complex diversity-encouraging architectures; (2) simply increasing rank in standard LoRA matches multi-component architectures; and (3) their proposed Align-LoRA, which adds an explicit alignment loss to encourage task-shared representations, achieves superior performance with zero inference overhead.

The paper's primary strength lies in its contrarian insight—demonstrating that architectural complexity for task-specific knowledge isolation may be unnecessary. The empirical evaluation is thorough, spanning multiple model families (LLaMA2, LLaMA3, Qwen2.5) and scales (3B-14B), with comprehensive benchmarks. The theoretical generalization bound derivation, while not groundbreaking, provides principled justification for the alignment approach. Crucially, Align-LoRA retains mergeability, avoiding the inference latency penalty of multi-component methods—a significant practical advantage.

However, the paper has notable weaknesses. First, the core insight that shared representations benefit multi-task learning is well-established in broader ML literature; the contribution is primarily applying this to PEFT, which is valuable but incremental. Second, there's a logical tension: if M-LoRA naturally achieves high similarity and performs well, why is explicit alignment necessary? The connection between observations and the proposed method could be tighter. Third, some comparisons use mismatched configurations (different ranks and parameter budgets), complicating fair assessment. Fourth, the Gaussian distribution assumption for KL divergence is simplistic, and deeper analysis of alternative alignment approaches would strengthen the work. Finally, hyperparameter sensitivity for λ deserves more investigation.

Overall, this is a solid contribution that challenges established assumptions with compelling empirical evidence. The practical benefits of mergeability and simplicity, combined with competitive performance, make this a valuable direction for multi-task PEFT research.

Score: 7.5

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces OptMerge, a novel method for model merging in Multimodal Large Language Models (MLLMs), along with a comprehensive benchmark for evaluating merging approaches. The work addresses an important gap in the literature, as most prior model merging research has focused on visual classification models or text-only LLMs for code/math tasks.

**Strengths:**

The benchmark contribution is valuable and timely. By providing fine-grained task categorization (VQA, Geometry, Chart, OCR, Grounding) with clearly separated training and evaluation data, the paper establishes a standardized evaluation framework that will facilitate future research. The inclusion of both full fine-tuning and LoRA fine-tuning scenarios, plus experiments on actual Hugging Face checkpoints, adds practical relevance.

The theoretical analysis (Theorem 3.1) provides meaningful insight into why fine-tuning intensity affects merging performance—the bound showing how convergence can actually harm merging quality due to cross-task interference and curvature terms is a useful contribution beyond empirical observations.

The method itself is well-motivated: using SVD-based low-rank approximation to denoise task vectors and applying different optimization strategies for full fine-tuning versus LoRA models addresses real challenges in practice. The observation that merge vectors tend to increase magnitude during optimization to achieve orthogonality (Figure 3) is insightful, and the solutions (SGD for LoRA, mean initialization) are practical.

The modality merging experiments demonstrate an interesting direction toward omni-language models, showing that static merging can even outperform dynamic online composition methods while using less storage.

**Weaknesses:**

The methodological contribution is somewhat incremental. OptMerge builds directly on WUDI Merging with relatively straightforward modifications (SVD denoising, initialization strategy, optimizer choice). While effective, these are not fundamentally novel ideas.

The scale of experiments is limited—most results are on 1B/7B models. The single 32B experiment is brief and doesn't fully establish scalability. Given the importance of scale in current ML research, this limits the paper's impact.

Some experimental comparisons raise questions: Iso-C performs catastrophically on Qwen2-VL (Table 3), which the authors explain but may indicate issues with that baseline's implementation. The resource comparison in Table 7 compares InternVL-1B against Qwen2-VL-7B, which isn't a fair apples-to-apples comparison.

The hyperparameter sensitivity analysis is limited. The fixed rank ratio (1/5) and λ search range [0.1-1.5] are somewhat arbitrary. While Table 8 shows robustness to rank ratio changes, more systematic analysis would strengthen the paper.

**Overall Quality:**

This is a solid contribution that advances MLLM model merging. The benchmark alone justifies publication, and the method shows consistent improvements across settings. While the technical novelty is moderate, the comprehensive evaluation and practical insights make this a worthwhile contribution to the community.

Score: 7.5

---

## qSak1Hjfdq

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces the All-day Multi-scenes Lifelong VLN (AML-VLN) problem and proposes Tucker Adaptation (TuKA) to address catastrophic forgetting when VLN agents learn across multiple scenes and diverse environmental conditions (normal, low-light, scattering, overexposure).

**Strengths:**
The problem formulation is well-motivated and practically relevant. Real-world VLN deployment requires agents to operate across diverse scenes and lighting/weather conditions, yet existing approaches suffer from catastrophic forgetting. The use of Tucker decomposition to represent multi-hierarchical knowledge is a creative technical contribution—by decomposing a fourth-order tensor into scene experts, environment experts, and a shared core, the method naturally separates knowledge across these dimensions. The empirical results are strong: AlldayWalker achieves 65% average SR compared to 52% for the best baseline (SD-LoRA), with significantly lower forgetting rates (11% vs 18% F-SR). The paper provides extensive evaluation across 24 tasks with 12 baselines, thorough ablation studies (including third-order vs fourth-order tensor analysis, scaling experiments), and real-world robot deployment. The creation of an extended Habitat benchmark with physics-based degradation models is a valuable contribution for the community.

**Weaknesses:**
Several concerns limit the paper. First, the method introduces substantial complexity with multiple hyperparameters (λ1, λ2, λ3, ω, rank dimensions r1-r4) and loss terms (EWC regularization, expert consistency, orthogonal constraints)—it's unclear if all components are necessary. Second, the inference-time expert selection via CLIP feature matching assumes the agent can reliably identify scene/environment from observations, which may fail when initial observations are ambiguous (e.g., partial scene view). Third, while the paper compares against many LoRA variants, it lacks comparison with recent VLN-specific continual learning approaches beyond FSTTA/FeedTTA. Finally, the expert selection mechanism requires storing CLIP features for all scene-environment combinations, which may not scale efficiently to many scenarios.

**Overall:**
This is a solid contribution with clear novelty in problem formulation and technical approach. The high-order tensor representation is principled, and the results demonstrate meaningful improvements. However, the methodological complexity and practical deployment concerns prevent a higher score.

Score: 7.5

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces LoRA-Mixer, a framework for combining multiple LoRA experts through routing mechanisms applied to projection layers, along with a novel Routing Specialization Loss (RSL). The work addresses an important problem in multi-task LLM adaptation: how to effectively compose pre-trained LoRA modules without requiring full retraining.

**Strengths:**
The paper makes several solid contributions. First, the architectural design of placing LoRA experts on projection layers rather than replacing entire attention/FFN blocks is sensible—it maintains the core model structure while enabling fine-grained specialization, and importantly works across both Transformer and SSM architectures. Second, the RSL formulation addresses a real issue with standard auxiliary losses that force uniform expert distribution regardless of input semantics. The entropy regularization term is theoretically motivated, and the appendix provides convergence analysis and generalization bounds, which strengthens the contribution. Third, the experimental evaluation is comprehensive, covering 15 benchmarks across five domains with three base models, demonstrating consistent improvements over baselines like MixLoRA, MoLE, and LoRAHub. The claim of using only 48% of trainable parameters while achieving better performance is compelling. Fourth, the plug-and-play capability with frozen LoRAs from public repositories has practical value.

**Weaknesses:**
Several issues limit the impact. First, while the RSL loss is a genuine contribution, the core idea of combining LoRA experts via routing isn't novel—MixLoRA, MoLE, and others already do this. The main novelty is the placement strategy and loss function, which is incremental. Second, the semantic interpretability of routing decisions isn't well analyzed—Figure 4 shows load balancing but doesn't demonstrate that experts meaningfully specialize in specific domains. Third, the 4k data anomaly in Table 9 (where auxiliary loss temporarily outperforms RSL) has a post-hoc explanation but raises questions about RSL's stability. Fourth, important baselines like Ties-Merging, DARE, and other recent LoRA merging methods are missing. Fifth, hyperparameter sensitivity (α, λ, β) receives limited analysis. Finally, some presentation issues exist, including a formatting error in the title.

**Overall Quality:**
This is a solid, well-executed paper that makes incremental but meaningful contributions to LoRA-MoE integration. The RSL loss is technically sound with theoretical grounding, and the comprehensive experiments support the claims. However, the work doesn't represent a paradigm shift—it's a competent improvement on existing frameworks rather than a breakthrough contribution.

Score: 7.5

---

## ppXAVexrAM

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents ARSS, a novel framework that applies decoder-only autoregressive transformers to novel view synthesis from a single image. The key contribution lies in being the first to adapt the GPT-style next-token prediction paradigm to multi-view generation with camera control.

**Strengths:** The paper addresses a meaningful gap in the NVS literature by exploring autoregressive models as an alternative to diffusion-based approaches. The technical approach is well-motivated: the video tokenizer (rather than per-frame tokenization) preserves temporal consistency, the camera autoencoder with Plücker coordinates provides appropriate 3D positional guidance, and the hybrid permutation strategy (spatial shuffle with preserved temporal order) is a sensible adaptation for handling bi-directional image context. The quantitative results are impressive—ARSS achieves competitive or superior performance against strong baselines including SEVA and ViewCrafter on RealEstate10K and ACID datasets. The zero-shot generalization results on DL3DV and AI-generated images further demonstrate robustness. The ablation studies on token permutation and tokenization strategies provide useful insights.

**Weaknesses:** Several concerns limit the impact. First, the resolution is limited to 256×256, which is notably lower than recent NVS methods. Second, while the paper motivates causal generation for incremental extension and trajectory adaptation, these claimed benefits are never demonstrated experimentally—experiments only use fixed trajectories. Third, some comparisons on DL3DV exclude key baselines (SEVA, ViewCrafter, RayZer) because it was in their training data, limiting evaluation breadth. Fourth, the approach depends heavily on tokenizer quality, which the authors acknowledge struggles with large view changes. Finally, there's no discussion of computational efficiency compared to diffusion alternatives, which is relevant for the world-modeling applications cited.

**Overall:** This is a solid contribution that opens a promising research direction. The methodology is sound, results are competitive, and ablations are informative. However, the resolution limitation and unvalidated claims about causal generation benefits prevent it from being a clear accept. The weaknesses are notable but don't undermine the core contribution.

Score: 7.0

---

## 7yvz93kBw9

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses the challenging problem of sparse-view 3D reconstruction using Gaussian Splatting. The authors identify a key insight: sparse-view 3DGS exhibits a dual failure pattern with overfitting in near-field regions (excessive Gaussian density) and underfitting in far-field regions (insufficient coverage). This observation is supported by compelling visualization of Gaussian distributions under dense vs. sparse view settings.

The proposed D²GS framework introduces two complementary components: DD-Drop (depth-and-density guided dropout) that adaptively regularizes overfit near-field regions using both local and global strategies, and DAFE (distance-aware fidelity enhancement) that strengthens supervision for distant regions using monocular depth priors. Additionally, the authors propose IMR, a novel robustness metric based on Wasserstein distance between Gaussian mixture distributions.

**Strengths:** The problem analysis is thorough and well-motivated, with clear visual evidence of the failure modes. The methodological design is sensible—combining local density/depth scoring with global depth-layer attenuation creates a principled dropout strategy. The experimental evaluation is comprehensive across LLFF, MipNeRF360, and DTU datasets, showing consistent PSNR improvements of 0.5-0.9 dB over strong baselines like DropGaussian and CoR-GS. The ablation studies systematically validate each component's contribution. The IMR metric offers a novel perspective on 3D representation stability beyond image-space metrics.

**Weaknesses:** The core ideas are somewhat incremental—dropout regularization follows DropGaussian's direction, and depth supervision has been extensively explored in NeRF literature. The method introduces multiple hyperparameters (dropout weights, layer thresholds, loss weights) requiring careful tuning. The DAFE module's reliance on external monocular depth estimation adds dependency and potential error propagation. While IMR is theoretically interesting, its practical utility is limited—it requires training multiple models to compute and doesn't directly optimize reconstruction quality. The paper doesn't deeply analyze failure cases or computational overhead beyond training time.

Overall, this is a solid contribution with clear empirical gains, though the novelty is moderate rather than transformative.

Score: 7.0

---

## d2pUyiXwcm

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces SCaSML, a novel framework for improving pre-trained PDE solvers at inference time via defect correction. The core idea is elegant: by deriving a "Structural-preserving Law of Defect"—a new semi-linear PDE that describes the surrogate's error—the method can leverage well-established Monte Carlo solvers to correct neural network predictions without retraining.

**Strengths:**
- **Original Concept**: The inference-time scaling paradigm for PDEs, inspired by recent LLM advances, is genuinely innovative. The key insight that defect correction preserves semi-linear structure (enabling efficient MLP solvers) is mathematically insightful.
- **Strong Theory**: The paper provides rigorous error bounds (Theorem 2.5), proving the final error is bounded by the product of surrogate and simulation errors. The improved convergence rate (Corollary 2.6) is properly established with detailed proofs in appendices.
- **Comprehensive Experiments**: Testing across four PDE types (LCD, viscous Burgers, HJB, diffusion-reaction) with dimensions up to 160d, using both PINN and GP surrogates, demonstrates broad applicability. Statistical significance tests (p ≪ 0.001) and detailed error analyses strengthen claims.
- **Practical Impact**: The "elastic compute" concept—trading inference time for accuracy on demand—addresses a real need in scientific computing where retraining may be impractical.

**Weaknesses:**
- **Computational Cost**: The MLP correction adds substantial overhead. While comparisons show SCaSML outperforms surrogates, the cost-benefit tradeoff relative to simply training longer isn't fully explored in the main experiments (though Appendix G.7 addresses this partially).
- **Dependence on Surrogate Quality**: The theoretical bounds depend on Assumption 2.4 (bounded surrogate error), and practical success relies on reasonably good initial predictions. Very poor surrogates would yield limited benefit.
- **Limited Surrogate Diversity**: Only PINNs and GPs are tested. Performance with neural operators or other architectures remains unexplored.
- **Hyperparameter Sensitivity**: Parameters like clipping thresholds and MLP levels require tuning, though the paper provides reasonable defaults.

Overall, this is a well-executed contribution that meaningfully advances hybrid ML-numerical methods. The theoretical framework is sound, experiments are thorough, and the results are consistently positive across challenging benchmarks.

**Score: 7.5**

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

**Assessment:**

This paper presents a timely empirical investigation into the optimization dynamics of modern recurrent models (SSMs like Mamba, Hyena) versus Transformers on associative recall and copying tasks. The central contribution is demonstrating that SSMs suffer from critical optimization instability—succeeding only within extremely narrow learning rate windows—unlike Transformers which are robust to hyperparameter choices. This finding has important implications for interpreting prior work comparing expressivity of these architectures.

The experimental work is substantial, spanning over 3,000 runs across multiple architectures and configurations. Key findings include: (1) SSMs require careful learning rate tuning, with performance collapsing outside narrow ranges; (2) SSMs and Transformers exhibit opposite scaling preferences (width vs. depth); (3) 1-layer Transformers show "phantom" induction head dynamics without accuracy gains, while properly-tuned 1-layer SSMs can solve MQAR; (4) convolutions are critical for single-layer performance in both architectures. The architectural ablations are informative, and the inclusion of newer architectures like DeltaNet provides useful context.

However, the paper has notable limitations. First, the evaluation is confined to synthetic benchmarks (MQAR, copying)—while argued to correlate with language modeling, there's no validation on actual downstream tasks. Second, the learning rate sensitivity observation, while important, lacks theoretical grounding; the connection to vanishing/exploding gradients is mentioned but not formally analyzed. Third, some claims appear overstated: the title suggests "revisiting" expressivity conclusions, but the paper primarily shows learnability is *also* important, not that expressivity differences don't exist. Fourth, the 1-layer comparison is somewhat artificial, and the finding that Mamba's convolution is crucial (and that adding convolution to Attention also helps) somewhat undermines the architectural comparison narrative. Finally, the DeltaNet analysis showing better stability is intriguing but underexplored.

Overall, this is a solid empirical contribution addressing an active debate, with actionable findings for practitioners working with SSMs. The limitations are acknowledged but do reduce impact.

**Score: 7.0**

---

## XX5EZoe4ec

- GT: Reject (avg 2.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents RetrievalFormer, a dual-encoder architecture that addresses the scalability limitations of transformer-based sequential recommenders by replacing the O(N) softmax with efficient ANN retrieval. The work combines a transformer-based user tower with a feature-based item tower, enabling both efficient serving and zero-shot cold-item recommendation.

The paper's strengths include its clear motivation and practical relevance—addressing a real deployment bottleneck for transformer recommenders. The architectural design is principled, using shared embedding tables across towers and an attention fusion mechanism for heterogeneous features. The evaluation is comprehensive, covering multiple datasets, baseline comparisons, ablation studies, and a novel Leave-One-Out Cold (LOOC) protocol for evaluating true cold-start capability. The efficiency gains (288× speedup at 10M items) are significant and well-demonstrated.

However, several weaknesses temper my enthusiasm. First, there's a notable accuracy gap: RetrievalFormer achieves 86-91% of strong transformer baselines, and on MovieLens-1M, the gap to AttrFormer (0.337 vs 0.4128 Recall@20) is substantial. The paper attributes this to the softmax vs. dual-encoder formulation, but a deeper analysis would strengthen this claim. Second, the dual-encoder architecture for retrieval is well-established in industry practice (YouTube, Facebook), so novelty lies primarily in the specific combination of techniques rather than fundamental innovation. Third, while the LOOC protocol is methodologically sound, the absolute cold-start performance (8-22% Recall@20) is modest, and the comparison to content-based baselines rather than showing competitive cold-start quality limits the impact of this contribution.

The attention fusion mechanism and shared embedding design are executed well, but individually these are incremental improvements over prior work. The paper's main value proposition is the integration of these components into a deployable system with honest empirical characterization.

Score: 7.0

---

## IdJakw2jta

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), addressing a meaningful gap between existing STVG research (focusing on videos <1 minute) and practical applications requiring processing of longer videos. The proposed ART-STVG framework uses an autoregressive transformer with selective memory banks and a cascaded spatio-temporal decoder design to handle streaming video input efficiently.

**Strengths:**
The paper makes a solid contribution by formalizing the LF-STVG problem. The methodological approach is well-motivated: autoregressive processing naturally handles longer videos without the memory bottleneck of processing all frames simultaneously; the selective memory mechanisms (spatial selection via text similarity, temporal selection via event boundary detection) are intuitively designed for the task; and the cascaded decoder design leveraging spatial predictions for temporal localization is sensible. Empirically, the paper demonstrates significant improvements over existing STVG methods on long-form benchmarks (e.g., +9.1% m_tIoU on 3-min videos), while maintaining competitive performance on short-form STVG. The efficiency analysis showing ~3x lower GPU memory usage (7.9G vs 25.1G) is compelling. Ablation studies are thorough and validate each component's contribution.

**Weaknesses:**
Several concerns limit the paper's impact. First, the benchmark construction methodology is limited—extending HCSTVG-v2 validation videos to create LF-STVG benchmarks may not capture the complexity of truly long-form scenarios where target events are sparsely distributed. Second, training all methods exclusively on short videos (20 seconds) while testing on long videos creates an uneven comparison that favors methods inherently designed for streaming. Third, despite claiming the first LF-STVG work, there's no comparison with memory-augmented video understanding methods (e.g., MovieChat, MA-LMM) adapted for this task. Fourth, the slower inference time (1.09s vs ~0.5-0.7s) and inability to operate in real-time are acknowledged but significant limitations for practical deployment.

**Overall:**
The paper identifies an important problem and proposes a reasonable first solution with solid empirical support. While the evaluation scope is limited and benchmark construction could be more rigorous, the core contributions are meaningful. The work opens a new research direction with clear potential for follow-up improvements.

Score: 7.0

---

## oiz0QHejVj

- GT: Reject (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

# Assessment

## Strengths

The paper presents CLIP-Map, a novel mapping-based compression framework for CLIP that replaces traditional select-based pruning with learnable transformation matrices. This is a well-motivated idea—select-based methods inherently lose information from dropped weights, while mapping-based approaches can potentially preserve more knowledge from the original model.

The technical approach is sound: using Kronecker factorization to reduce the mapping parameter overhead from O(D₁²D₂²) to O(D₁D₂) is mathematically justified, and the diagonal inheritance initialization is a clever solution to optimization instability. The two-stage mapping-retraining pipeline with knowledge distillation is sensible.

Empirically, the results are strong. CLIP-Map consistently outperforms TinyCLIP baselines across compression ratios, with particularly notable gains at extreme compression (1%: 15.8 vs 10.5 TR@1 on MSCOCO). The efficiency benefits are meaningful—25 training epochs vs 75 for progressive TinyCLIP while achieving better performance. The evaluation covers multiple scales (tiny/small/base), datasets (MSCOCO, Flickr30K, ImageNet, 21 downstream tasks), and includes useful ablations on initialization methods and training duration.

## Weaknesses

The baseline comparison is limited. While TinyCLIP is the most relevant baseline, other CLIP compression methods (UPop, EfficientVLM, MobileCLIP) receive minimal attention. The authors cite "time constraints" for not adapting these methods, which is unsatisfying for a top venue.

Key ablations are missing. There's no analysis of Kronecker factorization vs full mapping (does the approximation hurt performance?), no ablation isolating depth vs width compression contributions, and no investigation of what the learned mapping matrices actually encode.

The comparison methodology has issues. TinyCLIP with progressive compression (†) uses 3× more epochs, making the efficiency comparison somewhat unfair. Also, some reproducibility concerns exist—the exact training setup for baseline reproductions isn't fully specified.

The contribution, while solid, is incremental: combining established techniques (Kronecker factorization from prior work, knowledge distillation, diagonal initialization intuition) for a new application. The novelty lies primarily in the combination and adaptation to CLIP compression.

## Overall Quality

This is a solid contribution with good empirical results and a well-motivated approach. The mapping-based paradigm for compression is a meaningful alternative to select-based methods. However, the limited baseline comparison and missing ablations reduce confidence in fully understanding the method's advantages. The paper would benefit from deeper analysis of what the mappings learn and comparisons with a broader set of compression approaches.

Score: 7.0

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper provides a detailed methodological critique of "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), an ICLR 2025 Oral paper. Through systematic re-analysis of the original paper's four lines of evidence, the authors identify multiple issues: (1) omitted human evaluation data (1/3 of collected scores), (2) incorrect statistical analysis (no correction for multiple comparisons, inappropriate pooling), (3) mischaracterized qualitative feedback, (4) unfair hyperparameter comparisons, (5) inconsistent reporting in LLM-as-a-Judge evaluations, and (6) unsubstantiated community adoption claims that were subsequently retracted. The paper concludes with practical recommendations for more rigorous empirical ML research.

The strengths of this work are substantial. First, the methodological contributions—including the "Best-of-N" hyperparameter analysis framework—are novel and address a genuine gap in how methods requiring different hyperparameter tuning are compared. Second, the re-analysis is thorough, including ~6000 A100-hours of experiments across 9 models to test whether min-p's advantages persist under fair comparison. Third, the paper identifies serious issues that undermine the original paper's conclusions, including omitted data and selective reporting. Fourth, the distilled lessons for reviewers and researchers represent practical guidance that could improve empirical standards in the field. The statistical analysis is generally appropriate, particularly the Bonferroni correction for multiple comparisons.

However, there are some limitations. The paper focuses heavily on critiquing one specific paper, which could be perceived as overly adversarial, though the authors frame this as a case study for broader lessons. Some claims slightly overreach—for instance, showing that min-p matches baselines doesn't necessarily mean it offers no value, as algorithmic simplicity or other properties might still matter. The new human evaluation study added by the original authors (Appendix C.2) receives less thorough analysis than it might deserve. Additionally, while GSM8K is an appropriate benchmark, limited compute meant only one benchmark was tested thoroughly.

Overall, this is an important contribution that advances methodological rigor in empirical ML. The issues identified are substantive, the methodological contributions are genuine, and the practical recommendations are valuable for both researchers and reviewers. This type of critical analysis is essential for maintaining scientific integrity in a rapidly growing field.

Score: 7.5

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces DeepScientist, an ambitious system for autonomous scientific discovery that operates over extended timescales using Bayesian optimization to guide exploration. The work demonstrates substantial engineering effort, transparency about success rates, and generates three novel research contributions across different domains.

**Strengths:**
The paper's primary strength is its comprehensive demonstration of end-to-end autonomous research. The Bayesian optimization framework with Upper Confidence Bound acquisition and persistent "Findings Memory" provides a principled approach to balancing exploitation of promising directions with exploration of novel hypotheses. The empirical transparency is commendable—the paper openly reports a 1-5% success rate, with only 21 of 5,000+ ideas leading to scientific progress. The generated papers (A2P for agent failure attribution, T-Detect and TDT for AI text detection, and ACRA for inference acceleration) are technically coherent and demonstrate the system can produce complete research artifacts. The human evaluation by domain experts provides useful external validation.

**Weaknesses:**
Several issues merit concern. First, the novelty of the generated methods is modest—T-Detect applies t-distribution normalization to existing curvature-based detection (a reasonable but not transformative modification), and TDT applies wavelet transforms to token sequences (competent signal processing but not groundbreaking). The improvements over SOTA, while real, are incremental (1.9% on inference acceleration, 7.9% on AI text detection). The 183.7% improvement on agent attribution comes from a low baseline (~17%) and requires contextual interpretation.

Second, reproducibility is severely limited by reliance on proprietary models (Gemini-2.5-Pro, Claude-4-Opus) and prohibitive computational costs ($100,000+). Third, the evaluation methodology has gaps: only 3 human reviewers from the authors' institution, and the DeepReviewer comparison uses their own system. Fourth, the comparison to other AI scientist systems on the same tasks is absent—the evaluation compares to human SOTA rather than other automated systems. Finally, while the Bayesian optimization formulation is sound, the core contribution is more systems engineering than algorithmic innovation.

The work advances the field by demonstrating that automated systems can produce publishable-quality research, but the generated contributions themselves are not field-transforming, and the resource requirements limit practical impact.

Score: 6.5

---

## tswBfpkwHn

- GT: Reject (avg 5.0)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

# Review Assessment

## Strengths

1. **Novel Theoretical Contribution**: This paper presents the first theoretical analysis of Mamba's training dynamics and ICL generalization capabilities. The rigorous treatment of the nonlinear gating mechanism—a key distinction from Transformers—is genuinely novel and provides important mechanistic insights.

2. **Clear Characterization of Mechanisms**: The paper convincingly identifies two complementary mechanisms: (a) the linear attention layer selects context examples sharing relevant patterns with the query, and (b) the nonlinear gating suppresses outlier-containing examples while inducing locality bias through exponential decay based on index distance.

3. **Meaningful Comparison**: The theoretical comparison between Mamba and linear Transformers is well-motivated. The finding that Mamba can tolerate outlier fractions approaching 1 (with appropriate prompt length) while linear Transformers fail when α > 1/2 is an important theoretical insight.

4. **Empirical Validation**: The synthetic experiments support the theoretical predictions, showing Mamba maintains low error even when α approaches 0.8 while linear Transformers break down at α > 0.5.

5. **Well-Organized Proofs**: The appendix provides detailed proofs with appropriate supporting lemmas (Lemmas 3-6), making the theoretical claims verifiable.

## Weaknesses

1. **Limited Theoretical Scope**: The analysis is restricted to one-layer models, binary classification, and synthetic data with orthogonal pattern assumptions. While this is common in theoretical ML papers, it limits the practical implications significantly.

2. **Comparison with Linear Attention Only**: The main theoretical comparison is with linear Transformers, but empirical results in Appendix B show softmax attention performs comparably to Mamba and doesn't suffer from the CQ vulnerability. This somewhat weakens the practical significance of the claimed advantages.

3. **CQ Vulnerability Under-addressed**: When outliers are closest to the query (CQ setting), Mamba's accuracy drops dramatically (82.73% vs ~99% for other placements). This practical limitation deserves more discussion about potential mitigation strategies.

4. **Complex Conditions**: The convergence and generalization conditions (e.g., conditions (i)-(iv) in Theorem 1) involve many interdependent parameters, making practical interpretation challenging.

5. **Missing Empirical Ablations**: No ablation study isolating the gating mechanism's contribution—showing what happens when gating is removed or modified—would strengthen the empirical validation.

6. **Modest Real-World Validation**: While SST-2 experiments are included, they're limited and don't fully validate whether the theoretical insights transfer to practical scenarios.

## Overall Quality

This paper makes a meaningful contribution to understanding Mamba's ICL capabilities through rigorous theoretical analysis. The mechanism identification is valuable for the research community. However, the limited scope and the comparable performance of softmax attention (revealed only in appendix) temper the practical significance. The theoretical contribution is solid but not transformative enough for a higher score.

Score: 7.2

---

## OuMNJoKJBQ

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper investigates why current LLM alignment methods fail under jailbreak attacks and proposes a solution through reasoning-aware post-training. The work makes several contributions: (1) causal intervention experiments suggesting alignment operates independently of reasoning, (2) a new CoT safety dataset, and (3) Alignment-Weighted DPO (AW-DPO) that separately weights reasoning and response components.

**Strengths:**
The causal intervention experiment is clever—deactivating reasoning-critical neurons and showing safety behavior persists provides empirical support for the "superficial alignment" hypothesis. The linear probing analysis across layers is informative, showing near-perfect alignment detection in early layers versus delayed reasoning accuracy. The AW-DPO formulation addresses a real failure mode identified through error analysis (correct reasoning with unsafe responses, and vice versa). The empirical evaluation is comprehensive, covering multiple model families and diverse attack types from SorryBench. The comparison with reasoning-oriented models (Phi-4) usefully demonstrates that general reasoning improvement alone doesn't guarantee safety.

**Weaknesses:**
The contribution is incremental—the core idea of using CoT for safety alignment has been explored in recent concurrent work (cited by authors: Guan et al., Mou et al., Zhang et al.). AW-DPO is a weighted modification of DPO that, while sensible, lacks deep theoretical justification for why the specific weighting scheme should work. More concerning is the heavy reliance on GPT-4o as judge—the reported Pearson correlations (0.66 for full generation, 0.58 for reasoning) indicate only moderate consistency, introducing noise into the preference labels. The causal claims about "superficial alignment" could also be interpreted differently: safety and reasoning may simply be parallel mechanisms rather than safety being "shallow." The utility trade-off is understated—Table 1 shows notable utility drops with standard DPO (Mistral: 48.32%→41.45%), though AW-DPO recovers some. Hyperparameter sensitivity is high (Table 5 shows ASR from ~1% to 14% with different learning rates), suggesting brittleness. Finally, the method requires a complex pipeline (CoT generation, harmfulness scoring, DPO training) that may be difficult to reproduce reliably.

**Overall Quality:**
The paper is competent and addresses an important problem, but the technical novelty is limited. The empirical work is thorough but not exceptional. The methodological concerns around LLM-as-judge reliability and hyperparameter sensitivity, combined with the incremental nature of AW-DPO, make this a borderline contribution for a top venue.

Score: 5.5

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents AdaSVD, an adaptive SVD-based compression method for LLMs that introduces two main components: (1) adaComp, which compensates for SVD truncation errors through alternating Moore-Penrose pseudoinverse updates of singular matrices, and (2) adaCR, which assigns layer-specific compression ratios based on importance metrics.

**Strengths:** The paper addresses an important practical problem with clear motivation. The empirical results are strong and consistent—AdaSVD achieves substantial improvements over SOTA SVD-based methods (SVD-LLM, ASVD, FWSVD) across multiple compression ratios (40-80%) and model families (LLaMA2, OPT, Mistral, Vicuna). For instance, at 60% compression on WikiText-2, AdaSVD reduces perplexity from 89.90 (SVD-LLM) to 50.33, a 44% relative improvement. The ablation studies cleanly separate the contributions of each component, and the extension to VLMs (LLaVA) demonstrates generalizability. The stack-of-batch strategy for handling limited GPU memory is a practical contribution.

**Weaknesses:** The technical novelty is incremental—Moore-Penrose pseudoinverse updates are well-established in matrix approximation literature, and the layer importance metric (cosine similarity between input/output) is relatively simple. The paper lacks comparison with non-SVD compression methods (pruning, quantization) at comparable compression ratios, limiting assessment of competitive positioning. No inference speed or actual memory footprint measurements are provided, only perplexity/accuracy metrics. The convergence properties of alternating updates are not analyzed theoretically. The paper has presentation issues: Figure 2 contains garbled text, and some tables have formatting problems.

**Overall:** The paper offers solid empirical improvements over existing SVD-based methods with reasonable methodology, but the technical contribution is incremental. The consistent gains across models and compression ratios are meaningful for practitioners, though the lack of broader comparisons and efficiency metrics somewhat limits impact assessment.

Score: 7.0

---

## khBHJz2wcV

- GT: Accept (Poster) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a framework for post-training fine-tuning of flow-matching generative models to enforce physical constraints (PDEs) while jointly inferring latent physical parameters. The approach builds on the adjoint matching framework, extending it to jointly evolve both state variables and unknown parameters through a learned latent parameter predictor.

**Strengths:**
1. **Important problem**: Fine-tuning pre-trained generative models for physics-aware inference without requiring paired parameter-solution training data addresses a real need in scientific machine learning.

2. **Solid methodology**: The joint state-parameter evolution with surrogate base flows for α is a principled extension. The scaled memoryless noise schedule (κ parameter) provides useful numerical stabilization.

3. **Comprehensive evaluation**: Testing across four PDE families (Darcy, elasticity, Helmholtz, Stokes) covering elliptic, hyperbolic, and incompressible flow problems, plus natural images, demonstrates versatility.

4. **Good ablations**: Comparisons against Base AM, Base AM+φ, PBFM, and ECI isolates the contribution of each component. The trade-off curves between residual reduction and distributional fidelity are informative.

5. **Practical relevance**: The ability to handle boundary misspecification, model mismatch (e.g., forcing terms), and sparse observations via guidance addresses real scientific settings.

**Weaknesses:**
1. **Incremental methodology**: The core adjoint matching framework is directly borrowed from Domingo-Enrich et al. (2025). The joint evolution extension, while useful, represents a moderate rather than substantial methodological advance.

2. **Hyperparameter sensitivity**: The method requires tuning multiple λ parameters with unclear guidance. The trade-offs between λx, λα, and λf depend on problem specifics.

3. **Baseline fairness**: PBFM is a training-time method requiring complete retraining, making comparison to this post-hoc approach somewhat unfair. The reported PBFM convergence issues may reflect implementation rather than fundamental limitations.

4. **Limited theoretical analysis**: No convergence guarantees, error bounds for the inverse predictor φ, or analysis of how weak-form residual minimization relates to physical accuracy beyond the empirical results.

5. **Practical complexity**: The method requires pre-training an inverse predictor, computing weak-form residuals with test function sampling, and careful scheduling choices. The "lightweight" claim (20 gradient steps for Darcy) may not generalize to harder problems.

6. **Weak-form justification**: While claimed to be more stable, the paper doesn't quantify advantages over strong-form residuals or provide guidance on test function selection.

Overall, this is a solid contribution addressing an important problem with thorough experimentation. However, the methodological novelty is incremental, and practical deployment complexity remains a concern.

Score: 7.0

---

## KsWRLyIAKP

- GT: Withdrawn (treated as Reject) (avg 3.2)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a novel formulation of lead-lag detection in financial markets as a temporal link prediction task on dynamic graphs, evaluating eight deep learning models including several state-of-the-art Temporal Graph Neural Networks (TGNNs). The work introduces a custom dataset of 37 financial assets (stocks and commodities) with five years of daily data enriched with technical indicators and sentiment features.

**Strengths:** The paper makes a valuable contribution by bridging financial econometrics and temporal graph learning, an intersection that has been largely unexplored. The problem reformulation is creative and the benchmark dataset will be useful for the community. The experimental evaluation is reasonably comprehensive, comparing multiple TGNN architectures against a sequential baseline (LSTM), with proper statistical significance testing via Friedman and Conover's tests. The ablation study on feature types provides useful insights—showing that simple embedding features often outperform complex temporal features. The finding that GraphMixer, despite its simplicity, outperforms more complex architectures is noteworthy and aligns with recent literature on temporal networks.

**Weaknesses:** Several significant issues limit the paper's impact. First, there is no comparison with traditional financial methods (e.g., Granger causality, cointegration, lead-lag estimators from econometrics). While the authors argue that direct comparison is difficult due to formulation differences, this limits the paper's relevance to the financial community and leaves open whether the TGNN approach captures relationships that simpler methods miss. Second, the methodological novelty is limited—the only proposed extension (GM-TNF) performs worse than the base model, and the core contribution is applying existing architectures to a new problem. Third, the dataset is small (only 37 nodes), and evaluation is limited to a single dataset with no generalization tests. Fourth, the extremely high R@10 scores (approaching 0.99) raise questions about task difficulty—whether the sparse graph structure makes ranking artificially easy. Finally, the node embeddings from LLM-generated descriptions potentially contain information about company relationships that may constitute data leakage in realistic trading scenarios.

Overall, this is a competent application paper with an interesting problem formulation, but it lacks depth in financial validation and methodological innovation to be a strong contribution.

Score: 5.5

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes QubitCache, a KV-cache compression method that encodes attention patterns into quantum-inspired states while preserving critical tokens classically. The central insight—that attention patterns carry more essential information than individual tokens—is well-motivated through citations showing transformers maintain 95% accuracy with only 10-20% of attention connections preserved.

The empirical evaluation is comprehensive, testing across five models (Llama-8B, Mistral-7B, Phi-4-mini, Qwen2-7B, DeepSeek-Coder-7B) and six benchmarks. Results show 7× memory reduction with 92-97% performance retention, outperforming H2O, ScissorHand, and StreamingLLM. The 15-25% improvement on multi-hop reasoning tasks (HotpotQA) versus token-selection baselines is notable and supports the claim that preserving relational structure matters for complex reasoning.

However, the paper has significant weaknesses. The critical issue is that the "quantum" contribution appears oversold. The ablation study (Table 4) reveals that removing quantum encoding (classical fallback) only causes a 3.9% performance drop (0.491 → 0.472), while removing attention-based critical token selection causes a 20.4% drop. This suggests the real innovation is the attention-based token selection strategy, not the quantum amplitude encoding. The quantum formalism adds substantial complexity for marginal benefit.

Furthermore, all experiments use classical quantum simulation (Qiskit Aer), not actual quantum hardware. Claims about "NISQ compatibility" and "9-qubit circuits" are speculative without hardware validation. The computational overhead of quantum state simulation isn't analyzed—no latency or throughput comparisons with baselines are provided, which is critical for inference optimization research.

The theoretical claims about "bounded reconstruction error" and "logarithmic compression beyond classical information-theoretic limits" require more rigorous treatment. While encoding N tokens into log(N) qubits is theoretically appealing, the classical simulation requires storing complex amplitudes, negating the claimed memory advantage in practice.

Score: 5.5

---

## 2EQPpEZtEK

- GT: Reject (avg 3.3)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Review of DISTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation

## Assessment

**Strengths:**

The paper presents DISTAR, a novel zero-shot TTS framework that combines an autoregressive language model with masked diffusion over discrete RVQ tokens. This is a well-motivated contribution that addresses real limitations of existing approaches: pure AR methods suffer from exposure bias and struggle with long-range consistency, while continuous diffusion approaches face optimization challenges in high-dimensional latent spaces. The key insight—using discrete tokens with masked diffusion for intra-patch parallelism while maintaining AR for cross-patch dependencies—is clever and well-executed.

The empirical results are strong. DISTAR achieves the lowest WER on both LibriSpeech-PC and SeedTTS benchmarks, outperforming baselines including F5TTS, E2TTS, and DiTAR. Notably, it does so with fewer parameters (0.15B-0.3B vs. 0.5B-0.6B for some baselines). The subjective evaluations (CMOS/SMOS) also show favorable results. The ablation studies on decoding strategies, CFG schemes, and patch sizes provide useful practical insights.

Several design choices demonstrate practical engineering maturity: the removal of explicit duration predictors through [EOS] tokens, stochastic layer truncation for variable bitrate control at test time, and the temperature shaping heuristics for stable sampling all add practical value. The formulation of masked diffusion over RVQ tokens is mathematically clean and the paper is generally well-written.

**Weaknesses:**

The core architecture borrows heavily from DiTAR—the patch-wise factorization strategy is directly adapted, with the main novelty being the application to discrete tokens and the use of masked diffusion rather than continuous diffusion. While this adaptation is non-trivial and valuable, the conceptual novelty is incremental rather than groundbreaking.

The comparison methodology has some concerns. DISTAR uses NFE=24 while DiTAR uses NFE=10, making the efficiency comparison incomplete. The paper lacks direct inference latency/RTF comparisons, which is crucial for a method that claims computational advantages. Without this, claims about "inference cost close to its continuous counterpart" remain unsubstantiated.

The evaluation scope is limited—training only on ~50k hours of English data, while baselines like CosyVoice2 are trained on multilingual corpora. The codec dependency (MAGICODEC-based RVQ) is not thoroughly analyzed; codec quality fundamentally limits TTS quality.

The "tail-first bias" explanation for the decoding heuristics is hand-wavy. While the proposed fixes work, a deeper understanding would strengthen the contribution. Additionally, the relationship between patch size, NFE, and efficiency-quality trade-offs could be more comprehensively explored.

**Overall:**

This is a solid, well-executed paper that makes a meaningful contribution to zero-shot TTS. The approach is technically sound, results are strong, and the practical design choices add value. However, the incremental novelty relative to DiTAR and some missing analyses (inference speed, fair NFE comparison) prevent a higher score.

Score: 7.5

---

## wUzBBsrdB1

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper makes an important contribution to the sparse autoencoder (SAE) literature by systematically investigating the role of L0 (average active features per token) in determining whether SAEs learn correct, disentangled features. The authors demonstrate that when L0 is set incorrectly—either too low or too high—SAEs learn corrupted feature representations rather than the ground-truth features of the underlying model.

**Strengths:**
- The paper identifies a fundamental issue in SAE training: the standard "sparsity-reconstruction tradeoff" evaluation is misleading because reconstruction loss can improve even as feature correctness degrades. The toy model experiments with ground-truth features clearly demonstrate that a ground-truth SAE achieves worse reconstruction than a corrupted SAE when L0 is constrained below the true value.
- The theoretical analysis (Appendix A.5) formally proves that MSE loss incentivizes feature mixing when L0 is constrained below the true number of active features. This is a valuable contribution that grounds the empirical observations.
- The proposed decoder pairwise cosine similarity metric (c_dec) provides a practical way to detect incorrect L0 settings without ground-truth labels. The validation against sparse probing performance on real LLMs (Gemma-2-2b, Llama-3.2-1b) is compelling.
- The finding that most open-source SAEs have L0 values that are likely too low (Appendix A.13) has immediate practical relevance for the interpretability community.

**Weaknesses:**
- The c_dec metric sometimes exhibits a flat region near the optimal L0, making it ambiguous where exactly the "correct" L0 lies. The paper acknowledges this limitation but doesn't fully resolve it.
- The validation relies on sparse probing performance as a proxy for feature correctness, but sparse probing itself may not be a perfect ground truth—it could have its own biases and limitations.
- While the paper discusses automatic L0 optimization (Appendix A.11), the proposed method requires substantial hyperparameter tuning and is described as having "limited utility" in its current form.
- The toy model assumptions (orthogonal features) simplify away from the superposition regime that real LLMs operate in, though the paper does include experiments with added superposition noise.

Overall, this is a well-executed investigation of an important practical question in SAE research. The main insight—that L0 must be set correctly and current practices likely err on the side of too-low L0—is valuable. However, the paper provides only a partial solution (a diagnostic metric) rather than a definitive method for finding optimal L0.

Score: 7.5

---

## wSbVv6xaRr

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces FedMPDD, a novel federated learning algorithm that uses multi-projected directional derivatives to simultaneously reduce communication costs and provide inherent privacy protection against gradient inversion attacks. The key idea is elegant: clients encode their gradients by computing directional derivatives along random vectors, transmitting only scalars and seeds rather than full gradient vectors.

**Strengths:**
The paper makes a genuine contribution by introducing projected directional derivatives to the FL literature. The theoretical analysis is thorough, with formal convergence guarantees (O(1/√K) for single projection, O(1/K) for multi-projection) and privacy bounds (Lemmas 1-2). The approach differs fundamentally from existing structured/sketched updates by using *dynamic* random projections rather than fixed subspaces. Empirical results across MNIST, FMNIST, and CIFAR-10 demonstrate effectiveness against both DLG and recent GIA attacks, with SSIM scores significantly lower than baselines. The connection to Johnson-Lindenstrauss theory for convergence is insightful.

**Weaknesses:**
Several concerns limit the contribution: (1) **Computational overhead**: Computing m directional derivatives requires O(dm) operations per client. While the paper mentions JVP efficiency, this still requires m forward passes per round—a significant cost not reflected in the "communication-efficient" framing. (2) **Privacy composition fragility**: The bound T×m < d for multi-round privacy is restrictive. For models with millions of parameters, training beyond hundreds/thousands of rounds risks privacy loss, yet modern FL often requires thousands of rounds. (3) **Seed security**: If seeds are small (32 bits), enumeration attacks are feasible; if large, communication overhead increases. The paper doesn't analyze this trade-off. (4) **LDP comparison**: The comparison uses arbitrary noise levels (var=0.1, 1.0) without proper DP accounting (ε-δ guarantees), making the privacy-utility trade-off comparison incomplete. (5) **Missing baselines**: Recent secure aggregation and compression methods that jointly address privacy and communication (beyond those cited) could strengthen comparisons.

The writing is clear but dense; some proofs could be moved entirely to appendix. The novelty is genuine, but practical deployment concerns (dropout handling, heterogeneous compute, seed synchronization) remain unaddressed.

**Overall:** This is a solid submission with a novel idea and good execution, but concerns about computational overhead, privacy composition limits, and practical deployment prevent it from being a clear accept. The theoretical privacy guarantees are weaker than claimed under realistic multi-round training scenarios.

Score: 5.5

---

## opU91paIvZ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

This paper addresses an important problem in LLM interpretability: making chain-of-thought reasoning more "monitorable" by improving faithfulness (accurately reflecting reasoning processes) and conciseness (reducing unnecessary verbosity). The authors formalize this as a constrained optimization problem and propose a prior-guided transformation method.

**Strengths:**
The paper provides a clear theoretical motivation for why standard RL approaches fail for CoT monitorability optimization. The analysis of vanishing gradients—arising because the monitorability signal f(z) is sparse under the initial policy—is well-reasoned and empirically validated. The proposed solution, using an instruction-tuned prior model to transform reasoning traces into more monitorable versions before supervised fine-tuning, is practical. The empirical results show meaningful improvements: ~67% relative gain in faithfulness (hint verbalization) and significant CoT length reduction (up to 60%) while maintaining approximately 90% of baseline accuracy.

**Weaknesses:**
Several concerns limit the paper's contribution. First, the core method essentially reduces to knowledge distillation from a stronger model (Qwen 2.5-7B Instruct) to a weaker one (DeepSeek R1 Qwen-1.5B)—an established technique lacking novelty. The "constrained optimization" framing doesn't meaningfully inform the algorithm; the method is simply data generation followed by SFT. Second, the faithfulness evaluation has significant limitations: relying on hint "verbalization" doesn't guarantee genuine faithfulness (a model could verbalize hints post-hoc without actually reasoning through them), and using LLM-as-judge introduces known reliability issues. Third, the paper lacks comparison to meaningful baselines beyond naive RL—no comparisons to prompting approaches, other SFT methods, or alternative CoT optimization techniques. Fourth, the ~10% relative accuracy drop is non-trivial for safety-critical applications that the paper targets. Finally, faithfulness is only evaluated on hint injection scenarios; other forms of unfaithfulness (fabricated reasoning, omitted steps) remain unaddressed.

The paper makes incremental progress on an important problem but the limited methodological novelty, evaluation concerns, and missing baseline comparisons are significant weaknesses.

Score: 5.0

---

## 1j0ormf8uI

- GT: Accept (Poster) (avg 5.2)
- Predicted: N/A (7.2/10)
- Match: N/A

### Review

## Assessment

This paper proposes a novel calibration procedure for constructing lower prediction bounds (LPBs) for counterfactual survival time predictions under general right-censored data. The key innovation is providing exact marginal coverage guarantees, rather than the PAC-type guarantees offered by previous methods like Gui et al. (2024) and Davidov et al. (2025).

**Strengths:**
The paper makes a meaningful contribution to an important problem. The theoretical framework is rigorous, with Theorem 4.1 providing distribution-free exact coverage guarantees that explicitly quantify the error from weight estimation, and Theorem 4.2 establishing double robustness properties. The derivation of the reweighting scheme that transforms the counterfactual coverage problem into a weighted conformal inference problem is clever and well-motivated. Empirical validation is thorough, with experiments across six synthetic settings and a real clinical lung cancer dataset demonstrating both validity and improved informativeness of the LPBs compared to baselines. The application to comparing different radiochemotherapy regimens shows practical clinical relevance.

**Weaknesses:**
The method relies on several strong assumptions: SUTVA, strong ignorability (no unmeasured confounders), and critically, the assumption that potential outcomes are independent of censoring time conditional on covariates and treatment. This last assumption is non-trivial and may not hold in many clinical scenarios where censoring is informative. The coverage guarantee in Theorem 4.1 depends on the quality of weight function estimation, with an explicit error term that could be substantial in practice with limited data. While the approach adapts weighted conformal prediction techniques from Lei & Candès (2021), the novelty lies primarily in the specific reweighting scheme for survival counterfactuals rather than fundamentally new methodology. The real data evaluation cannot validate coverage guarantees since counterfactuals are unobserved, limiting the empirical conclusions.

Overall, this is a solid, well-executed contribution that advances uncertainty quantification for causal survival analysis, with appropriate theoretical grounding and empirical support, though practical applicability may be constrained by the required assumptions.

**Score: 7.2**

---

## 7L7kmHHfgf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents PIRN, a method for few-shot multimodal anomaly detection that combines prototype-based reconstruction with cross-modal communication. The work is motivated by the clear limitation that existing MAD methods struggle when training samples are scarce—cross-modal alignment methods cannot learn reliable correspondences, and memory-based methods misclassify unseen normal variations.

The technical approach is well-designed, introducing three complementary components: BPA (Balanced Prototype Assignment) uses optimal transport to prevent codebook collapse, APR (Adaptive Prototype Refinement) uses GRUs to adapt prototypes at test time, and MNC (Multimodal Normality Communication) exchanges prototype information across modalities via graph attention and cross-attention. The intuition behind each component is sound and they work together coherently.

Experimentally, the paper demonstrates consistent improvements across MVTec-3D-AD, Eyecandies, and Real-IAD D3 benchmarks, with particularly strong gains in the 5-shot and 10-shot settings. The computational efficiency analysis shows PIRN achieves 85% fewer FLOPs than FIND while matching or exceeding accuracy. The ablation studies are comprehensive, clearly validating each component's contribution.

However, the paper has some limitations. The technical contributions are incremental—optimal transport for balanced assignment, GRU-based adaptation, and cross-attention for multimodal fusion are all established techniques applied to this specific problem. While the combination is novel and well-executed, no fundamentally new insight or paradigm shift is introduced. The comparison against INP-Former is done via a two-stream adaptation that may not be optimal. Additionally, the paper lacks deeper analysis of failure cases and potential negative societal impacts.

Overall, this is a solid contribution with clear practical value for industrial anomaly detection in data-scarce settings. The execution is thorough, the writing is clear, and the results are convincing. While the novelty is somewhat incremental, the work successfully addresses an important problem with a well-integrated solution.

Score: 7.5

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important practical problem in identity-preserving image generation: the "copy-paste" artifact where models simply replicate reference faces rather than synthesizing identity under different poses/expressions. The work makes three main contributions—a large-scale paired dataset (MultiID-2M), a benchmark with novel copy-paste metrics (MultiID-Bench), and a method (WithAnyone) with a 4-phase training strategy and GT-aligned ID loss.

**Strengths:** The problem formulation is strong—formalizing copy-paste artifacts and proposing metrics to quantify them addresses a real gap in prior evaluation. The dataset contribution is valuable: 500k paired multi-ID images with diverse references per identity enables the paired-training paradigm. The GT-aligned ID loss is clever, avoiding noisy landmark extraction at the cost of requiring GT alignment. Empirical results demonstrate clear improvements: WithAnyone achieves higher Sim(GT) while maintaining lower copy-paste scores, breaking the observed trade-off. The ablation studies systematically validate each component, and the user study shows correlation between proposed metrics and human perception.

**Weaknesses:** The methodological novelty is incremental—contrastive learning and paired training are standard techniques applied to this domain. The copy-paste metric depends on having ground-truth reference images, limiting its applicability. While the paper claims CC-licensed data collection, privacy concerns around celebrity face datasets remain valid. Some baseline comparisons are limited (DynamicID excluded due to unavailable code, GPT-4o as black box). The paper could more thoroughly analyze failure cases and societal impacts.

Overall, this is a solid contribution with meaningful dataset/benchmark artifacts and competent methodology. The paper addresses an important practical problem with reasonable solutions and strong empirical validation.

**Score: 7.5**

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important but underexplored problem in LLM research: output volatility in long-form generation. While most prior work evaluates single-generation quality, this work systematically studies the instability of outputs across multiple generations—a practically important problem for reliable LLM deployment.

**Strengths:**
The VOLTBench contribution is substantial. The benchmark is thoughtfully designed with heterogeneous tasks (structured code/math and unstructured story/diary tasks), multiple languages, varying instruction complexities, and length scales from 5 to 500 sections (up to 100k words). Critically, it introduces multiple sampling to measure stability—a novel and necessary addition to the evaluation paradigm. The mechanistic analysis through attention traces provides meaningful insights, identifying "Attention Collapse" and "Attention Instability" patterns that correlate with generation failures. The proposed SELB method is lightweight and training-free, making it immediately practical. Empirical results show impressive improvements: 148% increase in mean output length and 69% reduction in volatility over LongWriter-8B, while maintaining generation quality.

**Weaknesses:**
The method's applicability is limited. SELB requires knowing the target number of sections (P_total) and relies on structured outputs with identifiable section markers. The free-form adaptation (SELB-Hybrid) appears only in the appendix despite being critical for real-world utility. The method operates by boosting specific tokens (title tokens) and blocking certain phrases ("I hope these..."), which is somewhat heuristic and may not generalize across diverse generation contexts. Comparisons to baselines are limited—only one specialized long-form model (LongWriter-8B) is compared, and the training-free baselines (Repetition Penalty, Entropy-Based Stopping, etc.) are described without full implementation details. The mechanism analysis establishes correlation but not causation between attention patterns and failures. Additionally, using only N=5 samples for volatility estimation may not fully capture output distributions.

**Overall:**
This is a solid contribution with a valuable benchmark and practical method. The limitations around task specificity and the underemphasized free-form evaluation prevent a higher score, but the systematic study of volatility and the comprehensive benchmark represent meaningful progress for the field.

Score: 7.0

---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a learnable Gray-Wyner Network for separating common and task-specific information across multiple vision tasks, bridging classical information theory with modern deep learning-based compression.

**Strengths:**
The theoretical contribution is the paper's strongest aspect. Theorem 1 provides bounds relating Gács-Körner and Wyner's lossy common information through interaction information, extending classical results to the lossy case. Theorem 2 derives a tractable optimization objective from Gray-Wyner theory, enabling end-to-end learning. The proposed "Shared" architecture with a common channel (split and merged from two task-specific branches) and two private channels is principled, and the auxiliary loss encouraging compatibility between the two common representations is well-motivated.

The experimental evaluation is comprehensive: synthetic data validates theoretical predictions about transmit-receive tradeoffs controlled by β; colored MNIST elegantly demonstrates edge cases (fully dependent vs. independent tasks); and real vision tasks (Cityscapes, COCO) show practical applicability. The ablation comparing Shared, Separated, and Combined architectures provides useful insights.

**Weaknesses:**
The empirical improvements on real tasks are modest—the proposed method lies between Joint and Independent baselines but doesn't dramatically outperform simpler alternatives. The restriction to only two tasks limits practical scope, and the extension to multiple tasks (mentioned but not explored) faces exponential channel growth. Using frozen pre-trained task models restricts the ability to learn jointly optimized representations. Additionally, comparing against more recent multi-task compression methods would strengthen the evaluation.

The architectures themselves are standard (ResNet blocks), and the theoretical bounds may not be tight in practice—there's limited analysis of how close learned representations get to these limits.

Overall, this is a solid contribution that meaningfully bridges information theory and learned compression, with clear theoretical grounding and practical applicability. The execution is competent though not exceptional.

Score: 7.0

---

## XIAta0WOJ6

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes F²SA-p, a family of fully first-order methods for stochastic bilevel optimization that achieves improved complexity bounds by leveraging higher-order smoothness in the lower-level variable. The key insight is interpreting the prior F²SA method as a forward difference approximation to the hyper-gradient, which naturally extends to p-th order finite differences with better error guarantees.

**Strengths:**
The paper makes a solid theoretical contribution by establishing an Õ(pϵ^{-4-2/p}) complexity bound for p-th order smooth problems, improving upon the previous best Õ(ϵ^{-6}) result. The connection between bilevel optimization and finite difference methods is insightful and well-motivated. The near-optimality result when p = Ω(log(κ/ϵ)/log log(κ/ϵ)) is meaningful, matching the Ω(ϵ^{-4}) lower bound established via reduction from single-level optimization. The lower bound construction is clean and avoids issues in prior attempts. The fully first-order nature of the method (no Hessian-vector products) has practical relevance for large-scale applications.

**Weaknesses:**
The high-order smoothness assumption (Assumption 2.5) is strong and may not hold for many practical problems. The empirical evaluation is limited—a single logistic regression experiment shows the method works but doesn't demonstrate the theoretical speedup. The condition number dependency (κ^{9+2/p}) is potentially large, and the gap between upper and lower bounds remains open for small p values. The normalized gradient step modification, while simplifying analysis, requires more justification for practical use. Additionally, the paper could better explain when higher-order smoothness naturally arises in ML applications beyond the softmax example.

Overall, this is a well-executed theoretical contribution that advances our understanding of stochastic bilevel optimization complexity. While not groundbreaking, it provides meaningful improvements and opens interesting directions for future work.

**Score: 7.5**

---

## s7oURFZTQD

- GT: Reject (avg 3.0)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes Multi-Grade Deep Learning (MGDL) as an alternative to standard end-to-end training (SGDL), where networks are trained sequentially on residuals from previous stages. The paper provides theoretical analysis (convergence guarantees, convex reformulation for single-layer ReLU, eigenvalue analysis) and empirical evaluation across image regression, denoising, deblurring, CIFAR classification, and time series tasks.

**Strengths:** The paper is well-organized and presents a systematic study. The empirical evaluation covers multiple architectures (fully connected, CNN, Transformer) and tasks. The observation that MGDL exhibits more stable training dynamics (eigenvalues of the iteration matrix staying in (-1,1)) while SGDL often produces eigenvalues outside this range is empirically interesting. The convex reformulation result for single-layer ReLU grades (Theorem 3) is a solid theoretical contribution within its restricted scope.

**Weaknesses:** Several significant issues undermine the paper's contribution:

1. **Limited theoretical novelty**: Theorems 1-2 are standard GD convergence results on nonconvex functions under bounded Hessian assumptions—they don't exploit MGDL's specific structure. The claimed advantage that "α_l << α" is asserted but not rigorously established.

2. **Unfair empirical comparisons**: SGDL and MGDL use different architectures (e.g., architectures 26-27, 28-29), confounding the evaluation. The total capacity differs, making it unclear whether improvements stem from the training methodology or architectural differences.

3. **Missing relevant baselines**: No comparison to layerwise training, progressive stacking, boosting-based neural methods, or positional encoding approaches for spectral bias—all highly relevant prior work.

4. **Questionable experimental choices**: Using MSE loss for classification (rather than cross-entropy) is suboptimal. Inconsistent optimizer usage (Adam vs. full-batch GD) across experiments.

5. **Incomplete theoretical explanation**: The eigenvalue analysis uses a local linear approximation; the connection to actual training dynamics is informal. The paper doesn't rigorously prove why MGDL's Jacobian has more favorable spectral properties.

6. **Overlooked connections**: The sequential residual training approach closely resembles gradient boosting with neural networks, yet this connection is unacknowledged.

**Overall:** The paper has merit in its systematic empirical study and identifies real optimization challenges. However, the theoretical contributions are limited, experimental methodology has fairness issues, and relevant prior work is missing. The weaknesses outweigh the strengths for a top venue.

Score: 5.0

---

## GRufFX1gAy

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces InnoGym, a benchmark for evaluating AI agents' innovation potential through two complementary metrics: performance gain (improvement over best-known solutions) and novelty (methodological difference from prior approaches). The work addresses a genuine gap in existing benchmarks, which focus on correctness but overlook whether agents can discover novel approaches.

The paper's primary strength lies in its principled conceptual framework. The formalization of innovation through the quadruple (P, S, V, D) and the dual metrics G (performance gain) and N (novelty) provides a coherent foundation. The taxonomy of task types—solved, improvable, and exploratory—offers useful conceptual clarity for categorizing different problem regimes. The benchmark construction process is thorough, with a two-stage filtering pipeline from 197 candidate tasks to 18 curated "Improvable Tasks" from real competitions.

A significant contribution is the validation of the Agent-as-judge distance function D_AGENT in Appendix F. The experiments comparing against human judgments on EquiBench code variants and cross-domain method triplets show strong correlation (Pearson r = 0.84-1.0) and reasonable agreement rates (75-100%), lending credibility to the novelty metric.

However, several concerns limit my enthusiasm. First, the novelty metric's reliability depends critically on the completeness and quality of S_known. For several tasks, only 1-3 reference solutions exist (e.g., ROADEF tasks), making novelty scores potentially unreliable. Second, the use of GPT-5 for novelty evaluation raises reproducibility concerns since this model is not publicly available. Third, only 10 of 18 tasks are used in main experiments due to computational constraints, reducing the benchmark's demonstrated scope. Fourth, the experimental finding that agents achieve novelty without performance gain could alternatively indicate that the novelty metric captures superficial variation rather than meaningful innovation—though the validation experiments partially address this concern. Finally, the paper lacks comparison against simpler baselines (e.g., direct LLM prompting), making it unclear whether agent frameworks add value beyond the base models.

The experimental results showing current agents perform substantially below human SOTA while sometimes achieving moderate novelty scores represent an interesting finding, though the interpretation that "robustness is the bottleneck" requires more supporting evidence.

Score: 7.0

---

## 41JeFWdVFa

- GT: Reject (avg 4.7)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes LDP (Lightweight Denoising Plugin), a denoising autoencoder-based module that improves generalization of single-image super-resolution models to unseen degradations. The method models the degradation process within a DAE framework, leveraging the observation from diffusion models that HR and LR features become aligned after noise addition. LDP can operate in two modes: as a training-time loss or as an inference-time post-processing module for diffusion models.

**Strengths:**
- The paper provides comprehensive empirical evaluation across diverse SR architectures (CNN, Transformer, GAN-based, Diffusion-based, Mamba) and multiple benchmarks (synthetic and real-world), showing consistent improvements.
- The method is practical with only 642K parameters and offers flexibility in application (training loss vs. inference post-processing).
- The ablation studies are thorough, examining loss components, patch sizes, frequency bands, and scale factors.
- The comparison against existing degradation models (DRN, DualSR) in Table 1-2 provides useful context.

**Weaknesses:**
- The contribution is incremental. The core ideas borrow heavily from existing work: dual regression (DRN), noise alignment property (DR2), and posterior sampling (DPS). The novelty lies in combination rather than fundamental innovation.
- The diffusion connection is overstated. The actual implementation doesn't use diffusion training—it's a standard DAE with patch-wise noise. The paper claims to "leverage a property of diffusion models" but this is more conceptual than practical.
- Improvements on strong baselines are modest. SwinIR and MambaIR show gains of only 0.05-0.83 dB PSNR. Larger gains appear on weaker baselines (StableSR, FeMaSR), suggesting the method primarily helps models with inherent artifacts.
- Computational overhead during inference is significant. Table 13 shows posterior sampling with LDP increases inference time from 19s to 28-178s per image, contradicting the "lightweight" claim.
- Some metrics degrade with LDP (e.g., CLIPIQA drops in Table 4), raising questions about perceptual quality trade-offs.
- The frequency loss formulation (Eq. 14-15) lacks proper normalization over spatial dimensions.

The paper presents solid empirical work but lacks the novelty and theoretical depth expected for a top venue. The method is a reasonable engineering contribution but doesn't advance fundamental understanding.

Score: 5.5

---

## mDuTDAK6KU

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces KOALA, an adversarial detector that leverages the disagreement between KL-divergence and L0-based prediction metrics. The core insight—that dense and sparse perturbations affect different metrics differently—is elegant and well-motivated. The method requires only clean-image fine-tuning, making it practical as a plug-in defense.

**Strengths:**
The primary strength is the formal theoretical guarantee (Theorem 1), which provides explicit conditions under which adversarial examples are guaranteed to be detected. This is rare in adversarial detection literature, where most methods lack provable correctness. The proof structure is sound, establishing necessary conditions for successful attacks on each metric and showing their mutual exclusivity under the stated assumptions. The experimental validation on theorem-compliant samples (100% precision/recall in Table 1) provides strong empirical support for the theory.

The methodology is clean—replacing the classifier head with a nearest-prototype classifier using two complementary metrics is intuitive. The fine-tuning procedure is lightweight, requiring no adversarial examples. Results on ResNet/CIFAR-10 (0.94 precision, 0.81 recall) demonstrate practical viability.

**Weaknesses:**
Several issues limit the paper's impact. First, the assumptions (A1-A4) are restrictive, particularly A3 (coordinate-wise bound |δi| ≤ (2/3)|p*_i|) which may not hold for many attacks. Second, the "theorem-compliant" subset analysis is somewhat circular—it validates the theory only on samples that satisfy the theorem's conditions, which may not represent real-world distributions. On the full test set (Table 1's non-compliant samples), performance drops significantly (F1 ~0.53-0.72).

Third, there's no comparison with existing adversarial detection methods (feature squeezing, LID, Mahalanobis, etc.) mentioned in the related work. Fourth, attack settings are limited to L∞-bounded perturbations with small ε values; no adaptive attacks, L2 attacks, or larger perturbations are tested.

The CLIP/Tiny-ImageNet results (precision 0.66) indicate a high false positive rate that could limit deployment. The paper also lacks hyperparameter analysis—τ is set to 0.75 without ablation.

**Overall:**
The paper makes a meaningful theoretical contribution to adversarial detection, but empirical validation has notable gaps. The method shows promise but requires more comprehensive evaluation against baselines and diverse attack settings to establish practical utility.

Score: 5.5

---

