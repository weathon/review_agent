# Direct-Scoring Baseline Results

Model: z-ai/glm-5

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper investigates a previously unexplored safety vulnerability in Masked Diffusion Language Models (MDLMs) arising from their iterative denoising process. The authors identify that affirmative tokens appearing at intermediate denoising steps can steer generation toward harmful responses—a "priming vulnerability" distinct from ARM prefilling attacks. They propose Recovery Alignment (RA), which trains models to generate safe responses from contaminated intermediate states, and demonstrate its effectiveness across multiple MDLMs and attack types.

**Strengths:** The paper makes a timely and novel contribution to MDLM safety, an underexplored area as these models gain prominence. The vulnerability characterization is thorough: the "anchoring attack" provides a clean experimental framework for measuring vulnerability strength, while First-Step GCG demonstrates exploitation without denoising intervention. The theoretical derivation (Theorem 4.1) provides grounding for the optimization-based attack. RA is well-motivated—directly addressing the training gap where models never learn to recover from contaminated states. The evaluation is comprehensive: three MDLMs, seven attack methods (both intervention-based and conventional), three evaluators, and 11 capability benchmarks. Ablations on intervention step scheduling and generation length add depth. The curriculum approach for t_max is sensible, and results show substantial ASR reductions for priming attacks (e.g., 58%→11.3% on LLaDA Instruct for First-Step GCG).

**Weaknesses:** Results on conventional jailbreaks are mixed—RA underperforms baseline on ReNeLLM/MMaDA (81.7% vs 79.3%) and achieves only moderate PAIR improvements. The 72.3% ReNeLLM ASR on aligned LLaDA Instruct indicates remaining vulnerability. Training overhead is notable (16 hours/4 GPUs) but not deeply contextualized against alternatives. The monotonicity assumption validation is limited to one dataset. Only three MDLMs are evaluated, all recent models from similar research groups.

Overall, this is a solid contribution addressing an important emergent problem with clear methodology and comprehensive evaluation. The identified vulnerability is genuinely MDLM-specific, and the proposed defense directly targets the root cause.

Score: 7.5

---

## WwDNiisZQm

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Content-Aware Mamba (CAM) for learned image compression, addressing two key limitations of applying Mamba to images: the content-agnostic raster scan order and strict causality constraints. The authors introduce two mechanisms: Content-adaptive Token Permutation (CTP), which clusters and reorders tokens based on feature similarity, and Global-Prior Prompting (GPP), which injects sample-specific global priors to relax causality. The resulting CMIC model achieves state-of-the-art BD-rate performance, surpassing VTM-21.0 by 15.91-21.34% across multiple datasets.

**Strengths:** The paper clearly identifies fundamental limitations of applying sequential SSMs to 2D image compression. The CTP mechanism is well-motivated—clustering content-similar tokens enables more effective redundancy removal, which is validated through ERF visualizations showing content-adaptive receptive fields. The empirical results are strong: CMIC consistently outperforms existing Mamba-based methods (MambaVC, MambaIC) while being more efficient (56% fewer parameters, 78% less memory than MambaIC). The ablations demonstrate that both CTP and GPP contribute meaningfully. The comparison with MambaIRv2 in the appendix is thorough and clarifies technical distinctions.

**Weaknesses:** The GPP mechanism borrows heavily from MambaIRv2's attentive state-space formulation—the main difference is tying prompts to clustering centroids rather than using standalone learnable parameters. While this is sensible for compression, the core prompting concept isn't novel. The codebook-based clustering with EMA updates is standard VQ-VAE technique; the novelty lies primarily in application to Mamba token reordering. The K-means clustering during training adds computational overhead (5 iterations per step accounts for ~5% of training time) that could be analyzed more deeply. Some ablations are limited—why K=64? How sensitive is performance to clustering quality? The entropy model contributes minimally, suggesting gains come primarily from transform networks.

**Overall:** This is a solid contribution with clear motivation, strong empirical results, and good efficiency. While some technical components are adapted from prior work, the integration is thoughtful and the application to compression is novel and well-executed.

Score: 7.5

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper introduces "Thoughtbubbles," a novel transformer architecture that learns to dynamically allocate parallel computation through latent residual stream forking during pretraining. The key innovation is using learned cumulative scores to determine which tokens receive additional computation "bubbles," trained purely via language modeling loss without explicit supervision.

**Strengths:**
The core idea is genuinely novel—unlike pause tokens or filler tokens that must be inserted manually, this approach learns where to allocate extra computation adaptively. The mechanism is elegant: residual streams can fork based on learned importance scores, with attention/residual updates attenuated by these scores. The empirical results are consistent across scales (150M-772M parameters), showing perplexity improvements and better zero-shot performance on LAMBADA and HellaSwag. The analysis showing computation allocation correlates with entropy is interpretable and sensible. The finding that a 319M model can outperform a 772M baseline is noteworthy.

**Weaknesses:**
Several issues limit the paper's impact. First, the experimental scale is very small—2.5B training tokens is orders of magnitude below modern standards, and model sizes are modest. Second, the baselines are weak: copy-3/copy-5 are naive approaches that don't meaningfully leverage extra capacity, and there's no comparison to established adaptive compute methods like Mixture-of-Depths, Universal Transformers, or other recurrent approaches. Third, the hard top-k selection creates gradient bottlenecks the authors acknowledge but don't fully resolve. Fourth, the implementation has poor hardware efficiency due to dynamic scatter/gather operations. Finally, evaluations lack reasoning benchmarks (GSM8k, etc.) where adaptive computation should shine—explained away by compute limitations.

The paper presents an interesting architectural contribution with reasonable execution, but the limited scale, training duration, missing relevant baselines, and efficiency concerns suggest this is more of a proof-of-concept than a mature contribution.

Score: 5.5

---

## 1E4Bltg6Xb

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes a Dynamics Feature Representation (DFR) framework for RL-based dynamic path planning in urban road networks. The core idea is to hierarchically refine high-dimensional global dynamics into compact, decision-relevant features through a "policy attention" mechanism (top-k shortest paths) and n-hop neighborhood extraction.

**Strengths:** The paper addresses a meaningful problem—the trade-off between global dynamics (complete but computationally expensive) and local dynamics (efficient but potentially incomplete). The hierarchical refinement approach is intuitive, and experiments on three real urban networks demonstrate practical applicability. The performance improvements in planning time (85.59%, 46.08%, 79.32% reduction) while maintaining solution quality are noteworthy. The ablation study on hyperparameters k and n provides useful insights.

**Weaknesses:** Several issues limit the contribution. First, the "policy attention" mechanism is not learned—it uses pre-computed static shortest paths via distance-based rewards. This is problematic because in dynamic scenarios, optimal paths may not correspond to static shortest paths, undermining the core premise of capturing "decision-relevant" dynamics. Second, the theoretical justification for Markov property preservation is weak—Equation 8 merely states an approximation without proof. Third, baselines are limited: comparing against "All Dynamics" is a strawman; modern approaches like attention-based GNNs or learned state representations are not included. Fourth, dynamics are simulated with simple congestion factors rather than real traffic data. Finally, the two new hyperparameters (k, n) require manual tuning with no principled guidance.

**Overall:** The paper tackles a relevant problem with a reasonable empirical demonstration, but the proposed method has fundamental limitations—the attention mechanism doesn't actually learn from dynamics, theoretical claims lack rigor, and baseline comparisons are inadequate for ICLR standards.

Score: 4.5

---

## aiM6bRd6bG

- GT: Reject (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

This paper introduces PPI candidate ranking as a novel problem formulation and proposes an interpretability-guided framework for prioritizing protein-protein interaction candidates for experimental validation. The core idea is clever: rather than treating PPI prediction as binary classification, the authors leverage known interaction partners to identify "active" embedding regions in D-SCRIPT/Topsy-Turvy models, using these to rank novel candidates. They further refine rankings using complementary signals including structural plausibility (SpeedPPI) and semantic/functional annotations via LLMs.

**Strengths:**

The problem formulation is genuinely novel and addresses a practical bottleneck—experimentalists need prioritized candidate lists, not just binary predictions. The prospective evaluation design using STRING v11→v12 transitions is methodologically sound and tests true predictive capability rather than retrospective fitting. The empirical improvements are substantial: Recall@10 increases from ~1.2% to ~26% for D-SCRIPT, representing meaningful practical gains. The systematic comparison of re-ranking strategies (Table 2) provides actionable insights—PubMedBERT achieves 75.5% maintain-or-improve rates while being computationally tractable. Including the PiNUI benchmark demonstrates generalization beyond STRING-specific biases.

**Weaknesses:**

The computational cost is prohibitive: retrieval requires hundreds of hours (Figure 2), limiting practical adoption. The re-ranking analysis operates on only 2,280 pairs (top-10 per protein), which is a tiny fraction of the search space—the scalability of the full pipeline remains unclear. The approach fundamentally assumes novel interactions follow patterns of known partners, which limits applicability to poorly characterized proteins. While the title emphasizes "interpretability," the final rankings remain non-interpretable; the method uses interpretability as a computational device, not for biological explanation. Some claims warrant scrutiny: the "two orders of magnitude" improvement refers to relative gains, while absolute recall remains modest. Additionally, the individual components (D-SCRIPT embeddings, SpeedPPI, LLM re-ranking) are borrowed from existing work; the novelty lies primarily in integration and problem formulation.

**Overall Assessment:**

This is a solid, publishable contribution. The problem formulation is novel and practically motivated, the empirical improvements are substantial and rigorously evaluated, and the systematic analysis of complementary signals provides useful insights. However, the computational inefficiency, limited scope of re-ranking analysis, and dependency on known partners prevent a higher score.

Score: 6.5

---

## Pa6ak2B9jJ

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes AUTO-RT, a reinforcement learning framework for automatic jailbreak strategy exploration in LLM red-teaming. The work addresses an important gap: existing automated red-teaming methods rely on fixed templates and focus on severity over exploitability. The paper introduces two technical contributions—Dynamic Strategy Pruning (DSP) for early termination of unpromising branches, and Progressive Reward Tracking (PRT) using downgraded models to address sparse rewards—both motivated by real challenges in RL-based exploration.

**Strengths:** The paper makes a solid contribution by reformulating jailbreak generation as a sequential decision process with a strategy-level decomposition. This allows discovery of diverse, transferable attack patterns rather than template-matching. The two technical innovations are well-motivated: DSP addresses the overwhelming safe-signal problem, while PRT provides a clever mechanism for reward shaping. The empirical evaluation is comprehensive—covering 16 white-box and 2 black-box models across three dimensions (effectiveness, efficiency, diversity), with ablation studies validating both components. The results show consistent improvements (up to 16.63% ASR gain) over strong baselines.

**Weaknesses:** Several concerns temper the contribution. First, the black-box results (Table 4, Table 10) show substantially lower ASRs, suggesting limited practical applicability. Second, computational efficiency is not discussed—the method requires 8×A100 GPUs for 9,000 episodes, raising questions about accessibility. Third, while the FIR metric for downgrade model selection is interesting, its theoretical justification remains thin. Fourth, recent baselines like TAP and GPTFuzzer could have enriched the comparison. Finally, the DeD metric (defense generalization diversity) is clever but relies on a specific defense construction procedure that may not generalize.

Overall, this is a well-executed contribution with clear motivation, solid methodology, and comprehensive experiments. The framework generalizes beyond fixed templates, and the ablation confirms component contributions. Minor limitations in black-box settings and efficiency discussion prevent a higher score, but the work meaningfully advances automated red-teaming methodology.

Score: 7.0

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a unified theoretical framework for understanding saddle-to-saddle dynamics and simplicity bias across multiple neural network architectures, including fully-connected, convolutional, and attention-based models.

**Strengths:**
1. **Unified architectural framework:** The paper provides an elegant general formulation (Equation 1) that encompasses fully-connected, convolutional, and self-attention architectures. Theorems 1 and 3 on embedded fixed points and invariant manifolds apply generally across these architectures, which is a meaningful unification.

2. **Two distinct mechanisms:** The paper makes an important distinction between data-induced timescale separation (in linear networks, leading to low-rank weights) and initialization-induced timescale separation (in quadratic/self-attention networks, leading to sparse weights). This decomposition is novel and provides insight into why different architectures exhibit qualitatively different learning dynamics.

3. **Testable predictions:** The paper derives clear, falsifiable predictions about how network width, data distribution (singular value spectrum), and initialization affect plateau duration and dynamics. The experiments in Figure 2 validate these predictions.

4. **Connection to simplicity bias:** The paper provides a principled definition of "simplicity" in terms of effective width (number of functional units), and links this to the saddle hierarchy through invariant manifolds. This is more architectural and interpretable than generic complexity measures.

**Weaknesses:**
1. **Heuristic dynamics analysis:** The core dynamics analysis (Theorem 4, Proposition 5) relies on linear/quadratic approximations near initialization and does not rigorously prove that trajectories *follow* invariant manifolds. The claim that dynamics "evolve near an invariant manifold, approaching a saddle, and switching" is presented as mechanism rather than a proved theorem.

2. **Limited experimental scope:** Experiments are confined to small networks on synthetic data. Validation on realistic-scale models or standard datasets would substantially strengthen claims about practical relevance.

3. **Partial overlap with prior work:** While the unified framework is novel, saddle-to-saddle dynamics in linear networks has been extensively studied (Saxe et al., Jacot et al., Berthier, Pesme & Flammarion). The paper's main novelty lies in the architectural unification and the two-mechanism distinction rather than fundamentally new phenomena.

4. **Open questions remain:** The paper acknowledges but doesn't resolve key technical questions: quantifying proximity to invariant manifolds, Markovian nature of saddle sequences, and exhaustiveness of fixed point classifications.

**Overall:**
This is a solid theoretical contribution that advances our understanding of training dynamics. The architectural unification and two-mechanism insight are valuable. However, the dynamics analysis remains heuristic, and experimental validation is limited. The paper would benefit from either more rigorous dynamics proofs or broader experimental validation.

Score: 7.0

---

## crKJJ4Ej60

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes Copy-Paste, a generation paradigm for improving contextual faithfulness in RAG systems by encouraging direct lexical reuse from retrieved context. The work is motivated by an observed inverse correlation between copying degree and hallucination rates on RAGTruth. The authors propose a two-stage approach: (1) three prompting methods to generate high-copying responses, and (2) CopyPasteLLM trained via DPO on automatically constructed preference data. The paper also introduces Context-Parameter Copying Capturing for mechanistic interpretation.

**Strengths:**
- **Impressive data efficiency**: Training on only 365 samples while outperforming baselines requiring 18,000+ samples is a notable practical contribution. The preference data construction pipeline is well-designed.
- **Strong empirical results**: The 12.2%-24.5% improvements on FaithEval counterfactual benchmarks are substantial, and evaluations across multiple datasets and model architectures demonstrate robustness.
- **Comprehensive analysis**: The ablation studies, training dynamics analysis, and mechanistic interpretation via logits/hidden states provide useful insights into why the method works.
- **Clear methodology**: The two-stage pipeline is clearly explained, and the paper is generally well-written and well-organized.

**Weaknesses:**
- **Limited conceptual novelty**: The core insight—that copying verbatim reduces hallucinations—is relatively intuitive. The contribution lies more in operationalization than in the underlying insight.
- **Counterfactual-heavy evaluation**: The most impressive results are on counterfactual benchmarks where context deliberately contradicts parametric knowledge. Real-world RAG systems typically retrieve accurate context, so this evaluation setup may not reflect practical deployment. Results on PubMedQA and original ConFiQA settings show more modest gains.
- **Potential for harmful copying**: The paper acknowledges limitations with incomplete context but doesn't thoroughly analyze when copying is detrimental. Blind copying of irrelevant or misleading context fragments could harm answer quality.
- **Prompt engineering overhead**: CP-Refine (the best performing prompting method) requires iterative refinement with a reviewer LLM, adding inference cost and complexity that isn't fully analyzed.
- **Mechanistic analysis limitations**: The logits and hidden state visualizations suggest differences in knowledge source reliance but don't conclusively prove the claimed mechanism of "parametric suppression" vs. contextual enhancement.
- **Limited baselines for Stage 1**: The comparison against only Attributed and Citations baselines for Copy-Paste-Prompting is narrow.

Overall, this is a solid contribution with practical value in data-efficient fine-tuning for RAG faithfulness. The work is well-executed and addresses an important problem, but the conceptual contribution is incremental and the evaluation emphasizes adversarial scenarios that may overstate practical impact.

Score: 7.0

---

## 1EdAn5gMVv

- GT: Reject (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

# Assessment

This paper presents SpatialBoost, a framework for enhancing pre-trained vision encoders with 3D spatial awareness through language-guided reasoning. The core idea is to convert dense spatial information into linguistic expressions (via a multi-turn CoT process from pixel to scene level) and inject this knowledge using LLM-based fine-tuning with a dual-channel attention mechanism.

## Strengths

The paper addresses an important limitation: pre-trained vision encoders lack 3D spatial understanding despite excelling at semantic tasks. The proposed approach of using language as an intermediate representation for spatial knowledge transfer is creative and well-motivated. The multi-turn hierarchical reasoning framework (pixel → object → scene level) provides a structured way to inject dense spatial information.

The evaluation is comprehensive, spanning depth estimation, segmentation, 3D scene understanding, robot learning, and image classification/retrieval. Consistent improvements across multiple vision encoders (OpenCLIP, SigLIPv2, DINOv2, DINOv3) demonstrate the generality of the approach. The ablations on reasoning order, data composition, and fine-tuning strategies are informative, particularly the comparison showing LLM supervision outperforms pixel-level alternatives.

## Weaknesses

The pipeline complexity is a concern. The method relies on multiple pre-existing models (depth estimation, segmentation, 3D reconstruction) to generate training data, raising questions about noise/bias propagation and reproducibility. While the paper includes an ablation on VFM-based vs GT-based data, it only scratches the surface of this issue.

The dual-channel attention mechanism is directly borrowed from prior work (Hong et al., 2023a), so the technical novelty lies primarily in pipeline design. Additionally, some results appear surprisingly large (e.g., 3D semantic understanding mIoU jumping from 6.9 to 54.9 for OpenCLIP), warranting careful scrutiny of the evaluation setup.

The paper lacks discussion of failure cases or limitations of the approach. There's also no analysis of computational overhead, which is relevant given the multi-stage training process involving an LLM.

Score: 7.0

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper investigates whether LLM-generated bibliographies can be distinguished from human ones using citation graph structure versus semantic embeddings. The authors construct paired citation graphs from 10,000 focal papers—ground truth and GPT-4o-generated—and systematically compare structural features (centrality, clustering) against title/abstract embeddings using Random Forests and GNNs.

The key finding is clear and interesting: structural features alone achieve only ~60% accuracy (near chance) distinguishing LLM from human reference lists, while cleanly separating both from random baselines (~89-93%). However, semantic embeddings achieve ~83% with RF and ~93% with GNNs. This suggests LLM bibliographies closely mimic human citation topology but retain detectable semantic fingerprints.

**Strengths:**
- The research question is timely and important for understanding how LLMs reproduce scholarly practices
- Methodologically sound progressive approach from interpretable features to deep learning
- Comprehensive experiments with multiple LLMs (GPT-4o, Claude), embedding models (OpenAI, SPECTER), and GNN architectures
- Appropriate baselines including field-matched random graphs preserving out-degree distributions
- Robustness checks including cross-model generalization, subfield/temporal baselines, and random-vector controls
- Clear, actionable finding: detection should target semantic content rather than graph topology

**Weaknesses:**
- Limited ML methodological novelty—standard RF and GNN architectures applied to graph classification
- The semantic differences driving separability are not deeply analyzed—what *specifically* in the embeddings distinguishes LLM selections?
- Dataset reused from prior work (Algaba et al., 2025)
- GNN improvement over RF on embeddings (~10%) is meaningful but not thoroughly interrogated
- Missing comparison to prior LLM-generated text detection methods
- The 6% temporal ordering violations in GPT suggestions could be more deeply explored as a confound

The paper makes a solid empirical contribution with clear implications for detecting LLM-generated bibliographies. While the ML methodology is standard, the domain application is novel and the experimental rigor is commendable.

Score: 7.0

---

## iDki7djO2K

- GT: Reject (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper proposes a general, algorithm-agnostic theoretical framework for defining and measuring "forgetting" in machine learning. The key insight is elegant: forgetting can be defined as a violation of predictive self-consistency—when a learner's predictions change after being updated on data consistent with its own expectations, this represents loss of information rather than acquisition. The paper derives an operational measure called "propensity to forget" and validates it empirically across regression, classification, generative modeling, continual learning, and reinforcement learning.

**Strengths:**
The paper makes a genuinely novel contribution by providing what appears to be the first unified definition of forgetting applicable across learning paradigms. The theoretical framework is mathematically sound, with clear desiderata and thoughtful thought experiments that stress-test the definition against edge cases (e.g., showing why parameter drift doesn't necessarily imply forgetting). The empirical validation across multiple domains demonstrates the measure's generality, and the finding that optimal learning often requires non-zero forgetting is counterintuitive and interesting.

**Weaknesses:**
The practical utility remains underdeveloped. While the measure correlates with training efficiency, the paper doesn't demonstrate how it could be used to design better algorithms or guide hyperparameter selection in practice. The computational cost of computing the measure via Monte Carlo particle approximation limits scalability, and no clear guidance is provided for choosing the horizon parameter k. Experiments focus on small models and simple tasks, leaving scalability to modern architectures unexplored. Additionally, while existing metrics are critiqued, there's limited empirical comparison showing the proposed measure offers superior predictive value.

**Overall:**
This is a solid theoretical contribution that addresses a genuine gap in the literature. The framework provides conceptual clarity that could inform future algorithm design, even if immediate practical applications aren't fully realized. The work is well-executed within its scope, though the practical impact would be strengthened by demonstrations of algorithmic improvements.

Score: 7.5

---

## ahpO7S1Ppi

- GT: Reject (avg 3.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Pctx, a personalized context-aware tokenizer for generative recommendation that conditions item tokenization on user interaction history, enabling the same item to receive different semantic IDs under different user contexts.

**Strengths:**
1. **Novel and well-motivated insight**: The observation that static tokenization implicitly enforces a universal similarity standard—where items with similar semantic ID prefixes receive similar probabilities regardless of user context—is compelling. The motivation using examples like watches purchased as gifts vs. investments effectively illustrates the problem.

2. **Solid technical approach**: The method addresses the sparsity-generalization trade-off through multiple practical strategies (adaptive clustering, redundant ID merging, data augmentation). The design shows awareness of practical challenges.

3. **Comprehensive experimental evaluation**: The paper includes thorough ablation studies, parameter sensitivity analysis, model ensemble comparisons, and an explainability experiment. Consistent improvements across three datasets strengthen the findings.

4. **Clear exposition**: The paper is well-organized with good figures that effectively communicate the key ideas.

**Weaknesses:**
1. **Modest absolute improvements**: While percentage improvements (up to 8.9% NDCG@10) sound substantial, the absolute metric values are quite low (0.02-0.05 range). The improvement over ActionPiece, the strongest baseline, is approximately 3-12% relatively, but the actual gains are small in absolute terms.

2. **Added complexity with multiple dependencies**: The method requires: training an auxiliary DuoRec model, clustering context representations, hyperparameter tuning for clustering (T, K, C_start, δ), fusion weight α, augmentation probability γ, and frequency threshold τ. This complexity raises practical deployment concerns.

3. **Hyperparameter sensitivity across datasets**: The optimal configurations differ notably across datasets (Table 6), suggesting that tuning is essential and the method may not generalize without careful calibration.

4. **Dependency on auxiliary model quality**: The entire approach relies on DuoRec for context representations. While the paper shows DuoRec outperforms SASRec for this purpose, there's limited analysis of what happens when the auxiliary model is suboptimal.

5. **Missing baseline comparisons**: The paper doesn't compare against simpler alternatives—e.g., using user context as additional input features alongside static semantic IDs—making it unclear whether the personalized tokenization approach is truly necessary vs. being an overengineered solution.

Overall, this is a solid contribution with a novel perspective on tokenization for generative recommendation. The idea is interesting and well-executed, but the moderate improvements relative to the added complexity and hyperparameter sensitivity prevent a higher score.

**Score: 6.5**

---

## wgGJE6Z1B3

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses draft model training efficiency for speculative decoding through a data-centric lens, proposing that tokens with flatter target distributions are more valuable for training. The work combines theoretical analysis using Gaussian distributions with practical experimentation on the EAGLE framework.

**Strengths:** The paper offers a fresh perspective on SD training efficiency—rather than modifying loss functions or architectures, it focuses on identifying which training samples provide the most value. The theoretical derivation in Appendix A is rigorous, showing that under a KL-constrained update model, higher-variance target distributions yield larger L₁ reductions. The proposed "flatness" metric (cosine similarity to uniform) is simple and computable offline without draft model warm-up. Empirically, the SFDD method achieves near-full performance with only 50% of training data (within 4% speedup gap), demonstrating practical utility. The paper comprehensively compares against multiple baselines (entropy, perplexity, margin, energy score) and provides useful analysis of training dynamics (gradient norms, loss curves).

**Weaknesses:** The Gaussian theoretical model, while analytically tractable, is a significant simplification of discrete token distributions—the paper bridges this gap using cosine similarity as a proxy, but the connection is imperfect. Additionally, the insight that uncertainty correlates with training value connects to established active learning literature, though the application to SD is novel. The empirical gains over entropy-based filtering (the closest baseline conceptually) are meaningful but not transformative. The evaluation is somewhat limited in scope—main experiments use only LLaMA3-8B-Instruct with EAGLE-2, though Appendix G provides additional model validation. Finally, the claimed "2× training speedup" primarily comes from data reduction rather than the method improving per-sample convergence, making the framing slightly misleading.

**Overall:** This is a solid contribution addressing an important practical problem in making speculative decoding more deployable. The work is well-executed with clear motivation, reasonable theory, and thorough empirical validation. While not groundbreaking, it provides actionable insights for practitioners.

Score: 7.0

---

## QryPmx2MNh

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

# Assessment

This paper introduces a novel task—discovering learning-friendly orderings of decoder tokens for arithmetic tasks—and proposes a method based on "loss profiling" that exploits the easy-to-hard learning dynamics of neural networks. The approach trains a Transformer briefly on a mixture of candidate orderings, then identifies favorable orders by their faster loss decrease. A hierarchical global-local search strategy handles the factorial search space efficiently.

**Strengths:** The problem formulation is original and practically relevant—output order matters for autoregressive learning, yet has been underexplored. The method is intuitive and builds on established phenomena (easy-to-hard learning dynamics). The hierarchical search is clever and enables exploration of up to billions of permutations. The paper successfully recovers the known reverse-digit order for multiplication from prior work, demonstrating that the method can discover non-trivial orders. The three proposed order-sensitive tasks (RELU, SQUARE-19, INDEX) provide useful testbeds, and the writing is clear.

**Weaknesses:** The evaluation has a significant circularity problem—all three novel tasks (RELU, SQUARE-19, INDEX) are *designed* so that forward order is optimal. Demonstrating that the method recovers forward order on tasks designed to have forward order as optimal provides limited validation. The PROD task is the only non-artificial benchmark, and while recovering the known result is encouraging, it's a single data point. The method also has practical limitations: it requires multiple training runs, is restricted to fixed-length sequences, and provides no strong baseline comparisons (the evolutionary strategy baseline and failed soft-permutation approach are relegated to appendices). The INDEX task shows poor success rates even with forward order in some configurations, raising questions about task design.

**Overall Quality:** The paper makes a genuine contribution in formalizing an interesting problem and proposing a reasonable solution. However, the evaluation methodology limits confidence in how broadly the approach applies beyond specifically-designed tasks. The novelty of the problem pushes this above a clear reject, but the circular evaluation prevents it from reaching accept territory.

Score: 5.5

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces ELMUR, a transformer architecture with layer-local external memory for long-horizon RL under partial observability. The method equips each transformer layer with persistent memory embeddings that interact with tokens via bidirectional cross-attention, managed by an LRU update policy.

**Strengths:**

The paper addresses an important problem in RL/robotics: retaining information across long horizons when key cues may appear thousands of steps before they're needed. The proposed approach is conceptually clean—layer-local memory with LRU management provides bounded capacity while allowing information persistence. The empirical results are genuinely strong: 100% success on T-Maze with corridors up to 1 million steps (trained on context length 10), best performance on 24/48 POPGym tasks, and best results on 21/23 MIKASA-Robo tasks with ~70% improvement over prior methods. The theoretical analysis (exponential forgetting bounds, half-life derivation) and memory probing experiments provide useful insight into the mechanism. The ablation study is thorough, examining memory size, blending factor, initialization, and component contributions.

**Weaknesses:**

The core contribution is somewhat incremental. External memory with learned read/write mechanisms dates back to Neural Turing Machines and DNC; LRU-style cache management is standard in systems; segment-level recurrence exists in Transformer-XL. The novelty lies in combining these for RL with layer-local memory. Comparisons to related memory-augmented transformers (Memorizing Transformer, Block-Recurrent Transformer) are limited—while RATE and DT are compared, a broader comparison would strengthen positioning. The fixed λ blending parameter could benefit from adaptive mechanisms. The "100,000× beyond attention window" claim is extrapolation rather than training at those scales. The MoE FFN component seems tangential to the main contribution—ablations show MLP FFN achieves equivalent performance.

**Overall:**

This is a solid contribution with strong empirical validation across multiple benchmarks and useful theoretical analysis. While the technical novelty is incremental (combining existing ideas), the application domain and thorough experimental validation establish meaningful value. The paper is well-written, reproducible, and addresses a practical problem in robotic decision-making.

Score: 7.0

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Review

This paper introduces a benchmark for evaluating text-to-image (T2I) models on generating images for WordNet taxonomy concepts—a novel and underexplored task. The work is comprehensive, proposing 9 evaluation metrics (including taxonomy-aware similarity metrics grounded in KL divergence and mutual information), testing 12 models across multiple dataset splits, and conducting both human and GPT-4 preference evaluations.

The key strengths are: (1) a genuinely novel task connecting T2I generation to structured knowledge resources; (2) well-designed taxonomy-specific metrics that leverage WordNet's hierarchical structure; (3) comprehensive model comparison including recent architectures (FLUX, SD3, PixArt); and (4) empirical findings that model rankings differ significantly from standard T2I benchmarks, validating the task's distinctness.

However, several weaknesses temper the contribution. First, this is purely an evaluation paper—no novel methods or improvements to T2I models are proposed. Second, the CLIP-based metrics inherit CLIP's known biases and limitations. Third, while human evaluation is conducted, it involves only 4 annotators without detailed inter-annotator agreement analysis. Fourth, GPT-4 preference evaluation shows positional bias (Figure 5, 12), and raw preference correlations are near zero despite ranking correlations being high—a concerning inconsistency that warrants deeper investigation. Fifth, the dataset sizes are relatively small (483-1700 concepts per split). Finally, the theoretical grounding of metrics relies on approximations that may not fully capture intended semantics.

The paper would benefit from deeper failure analysis (current examples in Appendix I lack quantification), investigation of why retrieval performs poorly beyond example images, and discussion of practical applications beyond benchmarking.

Score: 6.0

---

## FlcMckO6x5

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents a theoretical and algorithmic study of Separable Neural Networks (SepNNs), making three contributions: (1) a universal approximation theorem for SepNNs using Stone-Weierstrass arguments; (2) NTK analysis showing convergence to deterministic kernels (infinite width and rank) or random kernels (infinite width, fixed rank); and (3) a separable preconditioned gradient descent (SepPGD) method achieving O(nD) complexity for n^D grid samples.

**Strengths:**
- The theoretical analysis is rigorous and comprehensive, covering multiple SepNN variants (CP, TT, Tucker).
- SepPGD exploits the separable structure effectively, reducing complexity from O(n^D) to O(nD) compared to standard NTK-based preconditioning—a significant practical benefit.
- The experimental evaluation spans multiple applications (KRR, INRs, PINNs) with consistent improvements.
- Lemma 2 establishes a clean connection between SepPGD and classical PGD for the bivariate case via Kronecker product properties.

**Weaknesses:**
- The theoretical contributions are somewhat incremental. The approximation theorem follows from direct application of Stone-Weierstrass combined with standard universal approximation theory. The NTK analysis adapts existing MLP techniques to SepNNs without major conceptual novelty.
- The NTK regime requiring both infinite width AND infinite rank is restrictive; the fixed-rank result yields a random kernel, limiting practical applicability of the theory.
- SepPGD lacks formal convergence guarantees beyond the structural connection in Lemma 2. The relationship between spectral modification and convergence speed remains empirical.
- Comparisons with recent spectral-bias mitigation methods (e.g., WIRE) are missing. For surface representation, MSK could not be run, limiting comparison completeness.
- The approximation theorem provides no explicit error rates in terms of rank or width, offering limited practical guidance.

Overall, this is a solid contribution that advances understanding of SepNNs with practical algorithmic benefits. The efficiency gains are clearly demonstrated, but the theoretical and methodological contributions are evolutionary rather than field-advancing.

Score: 7.0

---

## PFhrOUJZ5o

- GT: Reject (avg 5.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper presents LAION-Comp, a large-scale dataset of 540K+ images with structured scene graph annotations designed to improve compositional image generation. The authors argue that the fundamental limitation in current T2I models is not architectural but rather a data deficiency—the lack of explicit structural annotations for complex scenes.

**Strengths:**
The paper addresses a genuinely important problem in generative AI. Compositional generation remains a persistent failure mode for diffusion models, and the authors' focus on structured annotations as a solution is well-motivated. The dataset contribution is substantial: creating 540K+ scene graph annotations with objects, attributes, and relations at scale represents significant effort. The automated pipeline using GPT-4o with partial human verification (showing 98.8%, 97.5%, and 95.7% accuracy for objects, attributes, and relations) provides a practical path forward. The introduction of CompSGen Bench (20,838 test samples) fills a gap in the evaluation landscape for compositional generation. The baseline models trained on both diffusion (SDXL) and flow-matching (SD3.5, FLUX) backbones demonstrate consistent improvements, and the user study (63% preference for SG-generated images) supports the practical utility of structured annotations.

**Weaknesses:**
The technical novelty of the scene graph encoder is modest—a standard GNN approach with learnable scaling. The paper's primary contribution is the dataset, with models serving as validation rather than innovation. Using GPT-4o for both annotation creation and evaluation (extracting scene graphs from generated images) introduces potential systematic bias that weakens the evaluation validity. The vocabulary diversity is notably reduced compared to the original LAION-Aesthetics (1,429 vs 5,811 object types after excluding proper nouns), limiting coverage. The comparison with T2I models on SG-aligned metrics may unfairly favor SG-based approaches. Additionally, the paper lacks comparison with recent layout-based compositional generation methods that use bounding boxes or spatial constraints as intermediate representations.

**Overall:**
This is a solid dataset paper with clear contributions to an important problem. While the methodological novelty is limited, the comprehensive experiments across multiple backbones and the new benchmark provide genuine value to the community. The weaknesses are largely acknowledged and do not undermine the core contribution.

Score: 7.0

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

# Assessment

This paper presents a compositional meta-learning framework that learns reusable computational modules and their transition statistics, then solves new tasks through probabilistic inference rather than parameter updates. The key technical contribution is a probabilistic generative model combining RNN-based gating (learning "task grammar") with RNN-based modules (learning "task syllables"), trained via particle filtering and marginal likelihood maximization.

**Strengths:**
The conceptual framework is elegant and well-motivated. The separation of within-module and between-module dynamics provides a strong inductive bias for compositional generalization. The probabilistic formulation naturally handles sparse feedback through hypothesis tracking—when target outputs are unavailable, the gating RNN constrains which module sequences are plausible. This is a genuine advantage over gradient-based meta-learning methods like MAML, which require parameter updates and struggle with sparse supervision. The experiments convincingly demonstrate that the model recovers ground truth structure in controlled synthetic domains, and the one-shot inference without parameter updates is impressive compared to the hundreds of episodes required for gradient descent.

**Weaknesses:**
The main limitation is the narrow empirical scope. The tasks are carefully designed synthetic domains that, while having ground truth structure to verify learning, are far simpler than real-world meta-learning benchmarks. More critically, the comparison to related work is incomplete. There's no comparison to modern in-context learning approaches (transformer-based), amortized inference methods, neural program synthesis, or recent modular meta-learning techniques. The paper compares to MAML/MLDG, but these are gradient-based methods—comparing to other inference-based or compositional approaches would strengthen the contribution. Additionally, the number of modules must be pre-specified, the scalability to higher dimensions/longer sequences is unclear (particle filtering with 250 particles can be expensive), and the "chicken-and-egg" training instability is acknowledged but not resolved.

**Overall:**
This is a conceptually solid paper with a clean probabilistic formulation for compositional meta-learning. The one-shot inference capability is valuable. However, the limited empirical validation on synthetic tasks and incomplete baseline comparisons make it difficult to assess whether the approach scales to problems the community cares about. The contribution feels more like a proof-of-concept than a complete solution.

Score: 5.5

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

This paper investigates memory optimization strategies for reasoning models, which have fundamentally different memory profiles than non-reasoning models due to their long chain-of-thought generations. The central insight—that the KV cache can dominate memory for reasoning models—is timely and important given the recent surge of interest in reasoning models like o1 and DeepSeek-R1.

The empirical contribution is substantial: over 1,700 experimental configurations across three model families (Qwen3, DeepSeek-R1-Distill, OpenReasoning-Nemotron) and four benchmarks, examining weight quantization, KV cache compression, serial and parallel test-time scaling. The finding that the standard "4-bit is optimal" prescription fails for mathematical reasoning tasks is significant, as is the discovery of a scale-dependent threshold (~8-bit 4B effective size) below which model capacity should be prioritized over longer generations. The analysis of when KV cache eviction outperforms quantization adds practical value.

However, the paper has notable limitations. First, the 8-bit 4B threshold is empirically derived without theoretical justification—why this specific threshold exists remains unexplained, and it may not generalize across architectures. Second, the paper focuses heavily on Qwen3 (other families appear in limited capacity), raising questions about broader applicability. Third, while comprehensive in breadth, the analysis is primarily observational; there's no deeper investigation into *why* mathematical reasoning is more precision-sensitive than knowledge-intensive tasks. Fourth, the KV cache compression comparison is limited to three methods when many others exist. Finally, the PRM analysis is cursory despite being relevant to the parallel scaling discussion.

Despite these weaknesses, the work provides valuable practical guidance for practitioners deploying reasoning models under memory constraints. The systematic exploration and clear articulation of memory-allocation principles represent a solid empirical contribution to an emerging problem space.

Score: 6.5

---

## 0cbUKCyBsH

- GT: Reject (avg 3.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Influence-Aware Time Series Forecasting (IATSF), proposing that traditional forecasting models are fundamentally limited by a "self-stimulation" assumption that ignores external influences. The authors provide theoretical analysis from a control-theoretic perspective, a new benchmark with textual influences, and the FIATS model to operationalize their framework.

**Strengths:**

The theoretical contribution is the paper's strongest element. Proposition 2.1 formally derives an error lower bound for self-stimulated models (Cov(ε) ⪰ BΣB^⊤ for linear systems), showing that ignoring unobserved influences creates an irreducible error floor. This provides a principled explanation for why billion-parameter foundation models struggle to outperform linear baselines—the bottleneck isn't model capacity but missing information. The FM Toy experiments compellingly validate this theory: all baseline models collapse while FIATS achieves near-zero error, demonstrating that the limitation is truly architectural.

The benchmark contribution is valuable for the community. The datasets are carefully designed to be "leak-free" with properly synchronized textual influences, addressing real issues in existing multimodal forecasting datasets. The CASM mechanism, which learns channel-specific sensitivity to influences via cross-attention, is well-motivated by the theoretical analysis and the ablation studies confirm its importance.

**Weaknesses:**

The core insight—that external information improves forecasting—is well-established in classical time series literature (ARIMAX, VARMAX). While the theoretical formalization is novel, practitioners have long incorporated exogenous variables. The paper frames this as a paradigm shift, but it's more accurately an extension to textual influences.

The FM Toy dataset, while useful for validation, is designed explicitly to prove the authors' point—it's a system where the only signal comes from textual influences. This synthetic setup doesn't establish that such structure exists broadly in real-world systems. The Atmospheric Physics experiments rely on weather forecasts as influences, but weather-atmospheric physics relationships are well-known physical correlations, not patterns the model "discovers" from text.

Several assumptions in the theory (U ⊥ X_h, independence of influences) may not hold in practice. Many real-world influences are autocorrelated or historically dependent. The paper also assumes "perfect influence forecaster" for fair evaluation, which is unrealistic—practical systems must forecast influences too, and errors would compound.

The GAUD results are mixed: while average improvement is 12.6%, many games show marginal or no improvement, suggesting the approach works better for certain domains than others.

**Overall Quality:**

This is a solid, well-executed paper that advances the field by providing theoretical grounding for incorporating textual influences in forecasting. The theoretical analysis and benchmark are valuable contributions. However, the practical novelty is limited (exogenous variables are not new), and some experimental setups are biased toward validating the approach. The paper represents clear progress but isn't a paradigm-shifting contribution.

Score: 6.5

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents Ambig-SWE, a benchmark for evaluating how LLM agents handle underspecified instructions in software engineering tasks. The work addresses an important practical problem: real-world task descriptions often lack crucial information, yet agents rarely ask clarifying questions and instead make unwarranted assumptions.

**Strengths:**
The paper's primary contribution is a well-structured evaluation framework that decomposes the underspecificity problem into three measurable capabilities: detection, clarification, and leveraging interaction. This decomposition enables targeted diagnosis of model weaknesses. The empirical analysis is comprehensive, covering 6 models (proprietary and open-weight) across multiple settings with statistical significance testing. Key findings are insightful: models default to non-interactive behavior without explicit prompting; Qwen 3 Coder completely fails to interact even with strong encouragement (100% FNR); Claude Sonnet 4 achieves 89% detection accuracy while other models struggle; and question quality—not just quantity—determines performance recovery. The exploration-first strategy observation (Claude models exploring before asking) versus immediate questioning (Deepseek, Qwen) provides actionable design guidance.

**Weaknesses:**
Several limitations affect the contribution's impact. First, the underspecified issues are synthetically generated by GPT-4o rather than naturally occurring, which may not reflect real underspecification patterns. While the authors compare their generations to naturally underspecified SWE-Bench issues, the synthetic approach limits ecological validity. Second, using GPT-4o as a user proxy, while practical, may not reflect realistic user responses—users may be uncooperative, confused, or provide incorrect information. Third, the methodology is relatively straightforward without novel algorithmic contributions; the value lies primarily in empirical analysis. Fourth, some findings are unsurprising (models need prompting to interact, stronger models perform better), though the structured quantification is useful. Finally, Claude Sonnet 4's Hidden evaluation uses only 100/500 instances due to cost, affecting comparability.

**Overall Quality:**
This is a solid empirical contribution that systematically documents current LLM limitations in handling underspecified instructions. While not groundbreaking, it provides a useful diagnostic framework and empirical baseline for future work on interactive agents. The findings about detection failures and question quality differences offer concrete directions for improvement.

Score: 7.0

---

## Kw2mvnzCoc

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents TSPulse, a compact (1M parameter) pre-trained time-series model that achieves strong performance across multiple diagnostic tasks through disentangled representation learning.

## Strengths

The core contribution—a multi-space disentangled representation framework that explicitly separates temporal, spectral, and semantic embeddings—is well-motivated and executed. The empirical results are impressive: achieving +20% on anomaly detection (TSB-AD), +50% on imputation, and +5-16% on classification while being 10-100× smaller than competitors. The efficiency argument is compelling, with genuine GPU-free deployment potential. The hybrid masking strategy thoughtfully addresses pre-training bias in existing approaches, and the ablation studies provide meaningful insight into each component's contribution. The code release commitment enhances reproducibility.

## Weaknesses

The novelty is somewhat incremental—time-frequency fusion has been explored before (TF-C, BTSF), and the disentanglement mechanism (separate heads on different embedding segments) is conceptually straightforward. The anomaly detection triangulation method requires labeled validation data for head selection, limiting its true zero-shot applicability—a practical concern not adequately discussed. The "semantic" embedding characterization focuses on robustness properties but lacks deeper semantic analysis. Some evaluation comparisons could be fairer; for instance, comparing against models not specifically designed for classification tasks. The backbone choice (TSMixer) is borrowed from prior work rather than novel.

## Overall Assessment

This is a solid engineering contribution with strong practical impact. The efficiency-quality tradeoff achieved is genuinely valuable for the community. While not revolutionary in conceptual novelty, the execution is thorough and the results speak for themselves. The paper makes a clear case for small, specialized pre-trained models in time-series analysis.

Score: 7.5

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces CANON (Conditional AdvaNtage estimatiON), a novel advantage estimation method for reinforcement learning in large reasoning models. The key insight is to regroup sampled responses based on training metrics (like entropy or response length) and compute advantages through inter-group and intra-group comparisons, thereby amplifying the metric's impact without presupposing its optimal direction. The approach is well-motivated: prior methods using handcrafted directional priors (e.g., entropy penalties) can be overly biased without careful tuning.

The strengths of this work are substantial. First, the conceptual framework is elegant—by splitting responses into high/low groups for a given metric, the inter-group advantage naturally identifies which trend yields higher accuracy, while intra-group advantage selects better responses within each trend. This avoids imposing arbitrary directional biases. Second, the theoretical analysis (Theorems 1 and 2) provides formal justification that CANON selectively amplifies the grouping metric without affecting independent factors. Third, the empirical results are comprehensive and convincing: CANON-Inter improves math reasoning by 1.9 points over DR.GRPO on average, CANON-Intra excels on complex logic tasks (+5.2 points on the hardest subset), and CANON-Eff achieves superior Pareto frontiers for efficiency-performance trade-offs. The experiments span three model families (Qwen2.5-Math-7B/1.5B, Llama3.1-8B) and multiple benchmarks.

However, several weaknesses temper my enthusiasm. The method introduces new hyperparameters (μ for inter/intra balance, α for group weighting) that require tuning, and the scheduling strategies (accuracy-based, cosine annealing) feel somewhat ad-hoc and model-specific. While the authors argue μ has clear physical meaning, the need for different schedules per model suggests practical complexity. Additionally, only entropy and length are explored in depth—other mentioned metrics remain untested. The improvements, while consistent, are moderate rather than transformative. Finally, the dynamic scheduling adds implementation complexity compared to simpler baseline methods.

Overall, this is a solid contribution with a fresh perspective on incorporating training metrics into RLVR. The work advances beyond naive reward/advantage shaping and demonstrates practical benefits. Minor issues don't fundamentally undermine the contribution.

Score: 7.5

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Paper Review

## Assessment

This paper provides a rigorous theoretical analysis comparing adaptive optimizers (Adam, Shampoo) and normalized steepest descent methods (NSD, e.g., Lion, Muon) through the lens of non-Euclidean geometry. The central insight is that while both algorithm families exploit similar geometry, they rely on fundamentally different smoothness notions: adaptive smoothness (Λ_H(f)) versus standard smoothness (L_{∥·∥_H}(f)).

**Strengths:**

1. **Clear and timely motivation:** The comparison between adaptive methods and NSD is highly relevant given recent interest in Muon/Lion as alternatives to Adam. The paper frames precise research questions about whether these families exploit geometry differently.

2. **Unified framework:** The "well-structured preconditioner set" abstraction elegantly unifies analysis across AdaGrad, Adam, and Shampoo variants. This provides clarity beyond case-by-case analyses.

3. **Novel technical contributions:** Lemma 3.3 introduces a non-trivial matrix inequality handling noncommutativity when extending from diagonal to general preconditioner sets. This is a genuine technical advancement that resolves a key barrier in the analysis.

4. **Meaningful separation results:** The paper establishes clear theoretical separations:
   - Adaptive smoothness enables O(T^{-2}) acceleration (Theorem 4.3) while standard ℓ_∞ smoothness cannot do better than Ω(T^{-1})
   - Adaptive variance enables dimension-free rates (Theorem 4.5) while standard variance leads to dimension-dependent lower bounds (Theorem 4.7)

5. **Comprehensive analysis:** The paper covers both convex and nonconvex settings, deterministic and stochastic cases, providing a fairly complete picture.

**Weaknesses:**

1. **Gap between theory and practice:** The results depend on unknown quantities (Λ_H(f), L_{∥·∥_H}(f)) that are hard to estimate in practice. The paper doesn't discuss how these quantities behave on real neural networks or whether the theoretical differences manifest empirically.

2. **Loose bounds:** Proposition 2.5 shows adaptive smoothness differs from standard smoothness by at most factor d, but this bound may be pessimistic. Without examples showing meaningful separation, the practical significance remains unclear.

3. **Log d dependence in general case:** The O(log d) factor for general preconditioner sets (vs. constant for diagonal) is somewhat disappointing. While the paper acknowledges this, it represents an incomplete resolution.

4. **Concurrent work:** The paper acknowledges Kovalev & Borodich (2025) independently proposed adaptive variance, but claims superiority based on using standard smoothness. This comparison could be more substantiated.

5. **Missing practical guidance:** The paper would benefit from discussion on when adaptive smoothness offers meaningful advantages and how practitioners might leverage these insights.

**Overall Quality:**

This is a solid theory paper with clear contributions to optimization. The technical results appear correct, the framework is elegant, and the separation results are meaningful. However, the lack of empirical validation and practical implications prevents it from being exceptional. The work advances theoretical understanding but leaves practical translation as future work.

## Score: 7.5

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper provides a timely and important empirical analysis of verifiers used in reinforcement learning with verifiable rewards (RLVR) for mathematical reasoning. With the rise of reasoning models like DeepSeek-R1 and OpenAI o1, understanding the reliability of reward signals is crucial for the field.

**Strengths:**
The paper makes several novel contributions. First, it systematically quantifies a known but poorly understood problem: rule-based verifiers have high precision but poor recall (~86%), and this degradation worsens with stronger policy models. Second, and more surprisingly, it demonstrates that model-based verifiers with higher static accuracy can perform worse in RL training due to susceptibility to "reward hacking" — the policy model learns to exploit patterns that fool the verifier. The finding that fine-tuning verifiers for verification can *increase* vulnerability to hacking (from 21.7% to 35% attack success rate for adversarial prefixes) is counterintuitive and valuable. The distinction between generative and discriminative verifier robustness (discriminative verifiers like xVerify are much more robust) provides practical guidance. The methodology is sound, using GPT-4o as an oracle to detect reward hacking and validating this approach against human annotations. The cross-domain experiments (math and general science) demonstrate generalizability.

**Weaknesses:**
The paper is primarily diagnostic rather than prescriptive — while it identifies problems with both verifier types, solutions beyond the hybrid approach remain limited. Using GPT-4o as an oracle introduces potential biases, though the human validation helps. The adversarial patterns explored are relatively simple (empty symbols, gibberish); more sophisticated attacks might reveal additional vulnerabilities. The compute requirements (4 nodes × 8 H100s × 3 days) limit reproducibility. Additionally, the findings about rule-based verifier brittleness are somewhat expected, though the systematic quantification and RL impact analysis are valuable contributions.

Overall, this is a solid empirical study that uncovers important issues in current RLVR practices and provides actionable insights for researchers building reasoning systems. The discovery of the static-RL performance mismatch in verifiers is a significant finding that will influence future work on robust reward systems.

Score: 7.0

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents SIGMADOCK, a fragment-based SE(3) diffusion model for molecular docking that achieves impressive empirical results and provides a novel methodological contribution to the field.

**Strengths:**

The core innovation—replacing torsional diffusion with fragment-based SE(3) diffusion—is well-motivated and theoretically grounded. Theorem 1 rigorously demonstrates why torsional models suffer from entangled induced measures while fragment-based approaches yield clean product distributions, leading to better-conditioned learning dynamics. This insight alone is a valuable contribution.

The empirical results are compelling. SIGMADOCK achieves 79.9% Top-1 success (RMSD < 2Å & PB-valid) on PoseBusters, representing the first deep learning method to surpass classical docking (59-67%) and prior DL approaches (12.7-32.8%) on proper train-test splits. The comparison is fair—the authors deliberately restrict training to PDBBind(v2020) rather than exploiting larger datasets with potential leakage.

The FR3D fragmentation scheme and triangulation constraints are clever inductive biases that reduce degrees of freedom while preserving chemical validity. The architecture modifications (removing bias terms in MLPs for smooth cutoff behavior) are sensible. The method doesn't require a separate confidence model or energy minimization, reducing computational overhead.

**Weaknesses:**

The evaluation is limited to re-docking only—not cross-docking or apo-docking, which are more relevant for drug discovery practice. While the paper acknowledges this, it remains a significant scope limitation.

Some recent methods (Boltz-1, Chai-1) are not directly compared on identical benchmarks. The co-factor handling limitation is acknowledged but affects real-world applicability for many targets.

The chirality preservation during fragmentation requires post-hoc filtering, which is not fully elegant. The dependence on user-specified pocket centers, while common, limits the method's practicality for blind docking scenarios.

The writing is dense in places, and the paper assumes substantial background in both diffusion models and structural chemistry. The main text's mathematical formalism, while rigorous, could benefit from more intuitive exposition.

**Overall:**

This is a solid contribution that advances deep learning for molecular docking. The fragment-based SE(3) approach is novel and well-justified, the empirical results establish a new benchmark, and the methodology is executed rigorously. While not field-defining, it represents clear progress on an important problem.

Score: 7.5

---

## 3icvqeC1sA

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces ChaosNexus, a foundation model for chaotic system forecasting built on a U-Net-inspired Transformer architecture with Mixture-of-Experts layers and wavelet-based frequency fingerprinting. The work is motivated by the observation that chaotic systems exhibit intrinsic multi-scale temporal structures that single-resolution architectures may fail to capture.

**Strengths:**
The paper presents a well-motivated architecture for an important problem. The ScaleFormer design—hierarchical patch merging in the encoder and expansion in the decoder with skip connections—is a natural fit for multi-scale dynamics, analogous to how CNN architectures handle spatial scales. The combination of MoE layers for system/region specialization and wavelet scattering for frequency-domain conditioning is sensible and grounded in dynamical systems theory.

The empirical evaluation is comprehensive, covering 9,000+ synthetic chaotic systems and real-world weather forecasting. The key finding that system diversity in pretraining matters more than trajectory volume is valuable for future foundation model design. The zero-shot weather forecasting results (<1°C MAE) are impressive, demonstrating practical transfer from synthetic to real chaotic systems. The multi-metric evaluation—including correlation dimension, attractor KL divergence, and Lyapunov exponent errors—appropriately captures both point-wise accuracy and long-term dynamical fidelity.

**Weaknesses:**
The architectural innovations are incremental rather than fundamental. The ScaleFormer is essentially applying U-Net to temporal patches, and MoE/wavelet components are well-established techniques. While the combination is effective, the novelty is engineering-focused rather than conceptually groundbreaking.

The weather forecasting comparison has fairness issues: ChaosNexus is pretrained on chaotic systems while baseline models (FEDformer, PatchTST, etc.) are trained from scratch. A more equitable comparison would involve baselines also pretrained on the synthetic corpus. Additionally, comparison with domain-specific weather models (GraphCast, FourCastNet) is absent despite claiming weather forecasting as an application.

The synthetic training data, while enabling scale, introduces potential distribution shift concerns. The paper lacks analysis of how well the synthetic systems cover the space of real-world chaotic dynamics. Some claims about "state-of-the-art" should be tempered given the limited architectural novelty and domain-specific baselines.

**Overall Quality:**
This is a solid contribution to the emerging area of foundation models for scientific computing. The paper demonstrates clear improvements over prior work (Panda, DynaMix) on chaotic system benchmarks and provides useful scaling insights. While the architectural novelty is modest, the systematic evaluation and real-world demonstrations make this a publishable contribution.

Score: 7.5

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces the problem of conditional causal bandits, where arms are conditional interventions (interventions that depend on observed variables) rather than hard interventions. The main contribution is a complete graphical characterization of the minimal set of nodes guaranteed to contain the optimal intervention target, along with a linear-time algorithm to compute this set.

**Strengths:**
- The problem formulation is novel and well-motivated. Conditional interventions are practically relevant, modeling realistic decision-making scenarios like medical treatment selection based on observed symptoms.
- The theoretical contributions are solid: the equivalence between conditional and deterministic atomic intervention superiority (Proposition 4) is a useful reduction, and the characterization of the mGISS as the LSCA closure of Y's parents (Theorem 13) is elegant and rigorously proven.
- The C4 algorithm is simple, runs in O(|V| + |E|) time, and its correctness proof relies on the insightful "connector" characterization (Lemma 15).
- Experiments on both random graphs and real-world Bayesian networks demonstrate meaningful search space reduction, with up to 90% pruning in larger models. The integration with a UCB-based bandit algorithm shows improved regret curves.

**Weaknesses:**
- The assumption of no latent confounders is a significant limitation. While the authors acknowledge this, it restricts applicability to settings with fully observed causal graphs.
- Only single-node interventions are considered. This is justified as making the problem more challenging, but multi-node interventions could yield better rewards in some scenarios.
- The conditioning sets Z_X are assumed predetermined rather than optimized—future work on this would strengthen practical applicability.
- The bandit integration experiments use a straightforward CondIntUCB; more sophisticated causal bandit algorithms might yield additional insights.

**Overall:**
This is a clear contribution to the causal bandits literature. The paper makes a novel theoretical contribution (characterizing the minimal search space for conditional interventions), provides a simple and efficient algorithm, and validates its practical utility. While the scope has limitations, they are appropriately acknowledged as directions for future work.

Score: 7.5

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a novel streaming framework for designing neural stimulation patterns that drive latent neural dynamics in desired directions. The work addresses an important gap in neuroscience: how to adaptively design high-dimensional stimulation patterns to achieve targeted effects in low-dimensional neural state spaces, with realistic experimental constraints like limited number of stimulation targets and non-negative stimulation magnitudes.

The main contributions include: (1) a new streaming jPCA algorithm (sjPCA) for real-time identification of rotational structure, (2) a kernel regression-based adaptive model for learning stimulus-response mappings that can handle state-dependent and time-varying response characteristics, and (3) an optimization framework that respects experimental constraints (sparsity, non-negativity). The evaluation is thorough across synthetic data and two real neural datasets (calcium imaging and electrophysiology), demonstrating real-time capability (<100ms).

**Strengths:**
- Important and timely problem for causal neuroscience experiments
- Comprehensive approach integrating streaming latent space construction, dynamics modeling, and stimulation optimization
- Reasonable computational efficiency with demonstrated streaming capability
- Comparison across multiple latent space methods and dynamical models
- Handles realistic experimental constraints (excitation-only, sparse targets)
- Appendix includes validation on actual photostimulation datasets (Daie et al., Draelos et al.)

**Weaknesses:**
- Main results rely on simulated stimulations applied to recorded data rather than actual closed-loop experiments
- Kernel regression requires storing all past observations, raising scalability concerns for very long experiments
- The optimization approach is relatively straightforward—alignment maximization under constraints
- sjPCA contribution is incremental (streaming adaptation of existing jPCA)
- Limited theoretical analysis of failure modes or convergence guarantees

The paper makes a solid practical contribution to an emerging area of neuroscience methodology. While the novelty in individual components is somewhat incremental, the integrated system addresses a real experimental need with appropriate engineering rigor. The main limitation—lack of actual closed-loop stimulation validation—is somewhat expected given the specialized equipment required, and the authors do validate on existing stimulation datasets.

Score: 7.0

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses modality imbalance in multimodal learning through Classifier-Constrained Alternating Training (CCAT). The authors identify an important gap: alternating training methods reduce encoder-level interference but fail to prevent classifier bias toward faster-converging modalities. The proposed solution involves pre-training a classifier with contribution-regularized cross-attention, freezing it during alternating training while using modality-specific LoRA adapters, and performing secondary updates on severely imbalanced samples.

**Strengths:** The paper makes a valid observation that encoder-level solutions alone cannot address entrenched classifier bias, supported by empirical analysis showing persistent imbalance during alternating training. The theoretical connection drawn between class imbalance and modality imbalance through gradient dynamics, while not deeply novel, provides useful intuition. The two-stage framework is well-motivated, and the empirical results demonstrate consistent improvements across three benchmarks (+1.35% on CREMA-D, +6.76% on Kinetic-Sound, +1.92% on MVSA). The ablation study is thorough, confirming each component's contribution. The t-SNE visualization and clustering metrics provide additional validation of improved feature discriminability.

**Weaknesses:** The method is notably complex, combining multiple techniques (pre-training, frozen classifier, LoRA, alternating training, secondary updates) without clear justification for why this particular combination is necessary versus simpler alternatives. The claim of achieving an "unbiased" classifier through regularization is questionable—regularization reduces but doesn't eliminate bias from training data. The theoretical analysis in Section 3.1 is superficial; noting that both class and modality imbalance involve gradient suppression isn't a novel insight. Improvements on two datasets are modest (1-2%), raising questions about whether the complexity justifies the gains. The computational overhead of two-stage training and secondary updates isn't discussed. Finally, the hyperparameter sensitivity (three dataset-specific parameters: r, β, λ) could limit practical applicability.

Score: 5.5

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

# Review of "An Algorithm to Perform Covariance-Adjusted Support Vector Classification in Non-Euclidean Spaces"

## Assessment

This paper proposes a "Covariance-Adjusted SVM" (CSVM) that accounts for class-specific covariance structure when computing margins. The authors argue that traditional SVM is valid only in Euclidean space, while input data resides in a "non-Euclidean statistical space" where Mahalanobis distance is appropriate. They propose transforming data using Cholesky decomposition of class covariance matrices and provide an iterative algorithm (SM Algorithm) to estimate population covariance from training data.

**Strengths**: The paper identifies a legitimate concern—that SVM's margin computation treats all directions equally, while data distributions may have different covariance structures across classes. The motivation to incorporate class-specific variance information is reasonable, and the empirical results show improvements over basic SVM kernels on several datasets.

**Weaknesses**: Unfortunately, the paper suffers from fundamental conceptual errors that undermine its theoretical contributions. First, the central claim that "input/statistical space is non-Euclidean" is mathematically incorrect—R^n with the standard inner product IS Euclidean; using Mahalanobis distance means choosing a different metric on the same space, not working in a "different space." Second, the proposed method is essentially **class-conditional whitening** followed by standard SVM, which is well-established and not novel. Third, Lemma 2.2's claim that an N-class problem requires N classifiers contradicts the very purpose of classification—one decision boundary separates two classes regardless of metric choice. The experimental evaluation has significant gaps: no comparison with the Mahalanobis kernel SVMs or MCVSVM mentioned in related work, no statistical significance tests, and marginal improvements (e.g., 97.4% vs 95.6% accuracy on Breast Cancer). The SM Algorithm lacks convergence analysis and essentially performs semi-supervised learning without proper theoretical grounding.

**Overall Quality**: While the empirical results show some improvement, the theoretical framework is built on incorrect premises, the claimed novelty overlaps heavily with existing whitening approaches, and critical baselines are missing from evaluation.

Score: 4.0

---

## sh1hWO9RHo

- GT: Reject (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces the Agent GPA (Goal-Plan-Action) framework, a principled approach to evaluating LLM agents by decomposing evaluation into six dimensions: Goal Fulfillment, Plan Quality, Plan Adherence, Logical Consistency, Execution Efficiency, Tool Selection, and Tool Calling. Each dimension is assessed by a specialized LLM judge, enabling systematic error identification and localization.

The key strength of this work lies in its comprehensive empirical validation. The authors demonstrate that their multi-judge approach captures 95% of TRAIL-annotated errors compared to 55% for the baseline, with strong human-LLM agreement (80-95%) and impressive error localization (86%). The cross-metric orthogonality analysis thoughtfully demonstrates that the six dimensions capture distinct failure modes, justifying the framework's decomposition. The consistency analysis (Krippendorff's α > 0.7 for most metrics) addresses concerns about LLM judge reliability. Additionally, the paper provides meaningful practical utility: by localizing errors to specific dimensions, it enables targeted debugging rather than just outcome-based pass/fail assessment.

However, several weaknesses merit attention. First, the Plan Quality judge underperforms significantly—the paper acknowledges poor precision and limited sample sizes (n≤2 for low-impact errors in test), undermining one of the framework's core components. Second, the evaluation is heavily reliant on Claude-4-Sonnet; results may not generalize across models. Third, generalizability claims are incomplete—the SWE-bench case study excludes PQ, PA, and TS due to architectural constraints, yet these exclusions are not adequately discussed as limitations. Fourth, the internal production dataset ("ANON-Data-Agent") cannot be independently verified, limiting reproducibility of those results. Finally, while the individual metrics are sensible, none are conceptually novel—the contribution is primarily in the unified framework and empirical demonstration of its utility.

Overall, this is a timely and well-executed contribution addressing an important gap in agent evaluation. The framework's principled decomposition, strong empirical results, and actionable debugging capabilities make it valuable for both researchers and practitioners. While some components need refinement and broader validation, the core contribution is solid.

Score: 7.0

---

## L2rfd2Czbj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper proposes wd1, a weighted policy optimization method for diffusion-based large language models that avoids computing policy ratios for importance sampling. The key insight is that likelihood approximations in dLLMs can introduce significant bias and variance when computing policy ratios, and the proposed method reformulates the RL objective as a weighted log-likelihood requiring only a single likelihood approximation.

**Strengths:**
1. **Clear motivation**: The paper identifies a real problem—exponential amplification of likelihood approximation errors in policy ratios—and provides a principled solution.
2. **Theoretical contribution**: The analysis connecting the method to energy-guided discrete diffusion training and negative sample unlearning is interesting and adds depth.
3. **Strong empirical results**: The improvements over d1 are substantial, particularly on Sudoku (76.4% vs 17.6%) and Countdown (51.2% vs 25.8%). The wd1++ extension achieves competitive results on math benchmarks.
4. **Efficiency gains**: Removing the need for reference and old policy likelihood approximations reduces computational overhead, and the method works without SFT.

**Weaknesses:**
1. **Limited baseline comparisons**: The main wd1 evaluation primarily compares against d1 (a concurrent work). While wd1++ comparisons include concurrent methods like MDPO, more comprehensive comparisons for the core method would strengthen the paper.
2. **Biased approximation still used**: The method still relies on d1's approximate likelihood at t=1, which introduces bias. The paper acknowledges this but doesn't explore alternatives.
3. **Remarkably large gains on some tasks**: The Sudoku improvement (+59%) is unusually large, raising questions about task specificity or reward design.
4. **Sampling overhead**: Sampling from the geometric mixture policy π_old^ref requires multiple forward passes, which partially offsets efficiency gains.
5. **Hyperparameter sensitivity**: The weighting scheme (ψ, λ) requires tuning; while ablations are provided, the sensitivity analysis could be more comprehensive.

The paper makes a solid contribution to RL for dLLMs with good theoretical grounding and strong empirical results. The ratio-free formulation is elegant and well-motivated. However, the limited baseline comparisons and some questions about the magnitude of gains temper enthusiasm slightly.

Score: 7.5

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

This paper proposes ConciseHint, a method to improve the efficiency of large reasoning models (LRMs) by injecting hints during the reasoning generation process, rather than before it begins. The approach adaptively controls hint intensity based on reasoning length (as a proxy for complexity) and dynamically adjusts injection position.

**Strengths:**
The paper addresses an important practical problem—verbose reasoning in models like DeepSeek-R1—and proposes a clean, intuitive intervention paradigm. The adaptive injection mechanism is well-motivated: easy queries receive more aggressive compression while complex queries are treated more conservatively. The training-free variant offers immediate applicability, and the learned embedding variant (ConciseHint-T) provides additional efficiency gains. Empirical results are solid across multiple models (Qwen3, DeepSeek-R1) and benchmarks (GSM8K, AIME24, GPQA-Diamond), with substantial token reductions (e.g., ~48% on GSM8K) and minimal accuracy loss. The compatibility with existing methods like Deer and NoWait demonstrates flexibility, and the end-to-end latency measurements add practical relevance. Ablation studies reasonably justify the design choices.

**Weaknesses:**
The technical novelty is limited—the core idea is essentially injecting "be concise" prompts during generation, with an adaptive schedule. The complexity proxy (current length) is simplistic and doesn't capture problem difficulty until significant generation has occurred. There's insufficient analysis of *what* reasoning content is being compressed—beyond transition word counts, we don't understand whether critical reasoning steps are being preserved. The ConciseHint-T experiments are limited to a small model (Qwen3-1.7B), weakening generalization claims. Hyperparameter sensitivity to α deserves more scrutiny despite claims of robustness. Additionally, the KV cache re-computation costs from injection could become non-trivial at scale, and the method's effectiveness appears to decrease on harder benchmarks (AIME24 shows more accuracy variability).

**Overall:**
A useful contribution addressing a timely problem, with solid empirical validation. However, the approach is technically incremental, and deeper mechanistic understanding of why during-generation intervention works better than pre-generation prompting would strengthen the paper considerably.

Score: 6.5

---

## 32mrjmaeMP

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important practical problem in Task Arithmetic: enabling weight disentanglement without requiring access to external task data. The authors propose TAK (Task Arithmetic with KFAC regularization), which cleverly reformulates representation drift regularization as a curvature approximation problem solvable via KFAC.

**Strengths:**
- The theoretical connection between representation drift and the GGN matrix is elegant and well-motivated. Starting from the objective of minimizing interference and showing it reduces to a quadratic form under linearization provides solid grounding.
- The dataless property is genuinely valuable—enabling privacy-preserving scenarios where sharing training data is prohibited.
- Empirical results are strong: TAK achieves state-of-the-art on task addition (86.0% on ViT-B/32 vs 85.6% for τJp) and task negation, while being dataless.
- The proposed factor aggregation scheme provides O(1) complexity in the number of tasks, addressing scalability concerns.
- The analysis of task localization (Fig. 5) and robustness to scaling coefficients adds practical value.
- Comprehensive experiments across vision and language domains, multiple model scales, and both linearized/non-linear regimes.

**Weaknesses:**
- The core technique (KFAC) is not novel; the contribution is primarily its application to this specific problem. The aggregation heuristic (Eq. 8) is empirically validated but theoretically thin.
- The method is theoretically grounded in linearized fine-tuning. While non-linear results via attention-only training are promising, they're not as strong as the linearized regime.
- KFAC storage scales quadratically with layer width, creating memory challenges for very large models (acknowledged in limitations).
- The regularization weight λ varies substantially across model scales (100 to 2000), requiring some tuning.
- Language task gains are more modest than vision gains.

Overall, this is a solid, well-executed contribution with practical impact. While the core technique leverages existing machinery, the application is novel, the problem formulation is insightful, and the empirical results demonstrate clear value.

Score: 7.5

---

## 4Ha2srdhPN

- GT: Reject (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents GRAID, a framework for generating spatial reasoning VQA data using 2D bounding boxes rather than 3D reconstruction or caption-based synthesis. The key insight—that qualitative spatial relationships can be reliably determined from 2D geometry—is simple but effective, and the empirical results are compelling: 91.16% human-validated accuracy vs 57.6% for existing methods.

**Strengths:**
The paper addresses a genuine and important problem. The finding that only 57.6% of SpatialVLM-generated questions are valid is striking and validates the authors' critique of existing approaches. The solution is elegant—avoiding cascading errors from depth estimation and hallucinations from LLM generation by operating entirely in 2D. The scale is impressive (8.5M VQA pairs), and the human evaluation provides strong evidence of quality improvement. The multi-model, multi-benchmark evaluation demonstrates convincing transfer: models trained on GRAID data improve not just on held-out question types but on external benchmarks like BLINK (with notable gains on spatial relations and relative depth tasks). The SPARQ optimization yielding up to 1400× speedups is a nice engineering contribution that enables practical large-scale generation.

**Weaknesses:**
The scope is limited to 2D spatial reasoning—while this avoids 3D errors, it also means important spatial concepts (occlusion reasoning, 3D position estimation, complex depth relationships) remain unaddressed. The training data is exclusively from autonomous driving datasets, and while the authors claim domain agnosticism, this isn't convincingly demonstrated; generalization to BLINK helps but doesn't fully prove the point. The human evaluation sample size (317 questions evaluated by 4 people) is relatively small given the scale of generated data. The baseline comparison uses a "community implementation" of SpatialVLM rather than the official release, and SpatialRGPT couldn't be properly evaluated. Including depth-related questions (Closer, Farther) using depth estimation models partially contradicts the stated goal of avoiding unreliable depth estimation. Finally, the paper lacks discussion of what spatial reasoning capabilities remain beyond GRAID's reach.

**Overall Quality:**
This is a solid contribution with clear practical value. The approach is well-motivated, the execution is thorough, and the results demonstrate meaningful improvements. However, the limited scope of spatial reasoning covered and the narrow domain of training data prevent this from being exceptional. The technical contribution is incremental rather than transformative—applying established object detection to generate template-based questions—but the execution and validation are rigorous enough to merit acceptance.

Score: 7.0

---

## oh9ChF7Pv0

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper introduces EGG-SR, a unified framework that exploits symbolic equivalence to accelerate symbolic regression across three algorithm families (MCTS, DRL, and LLM-based methods). The core insight is compelling: equivalent expressions like log(x₁³x₂²) and 3log(x₁)+2log(x₂) are treated as distinct by existing SR methods, causing redundant exploration.

The key strength lies in the principled theoretical foundation. Theorem 3.1 establishes that EGG-MCTS achieves a tighter regret bound through reduced effective branching factor (κ_∞ ≤ κ), while Theorem 3.2 proves EGG-DRL produces an unbiased gradient estimator with strictly lower variance. The proof techniques leveraging transposition tables and Rao-Blackwellization are elegant and appropriate.

Empirically, the results show consistent improvements across baselines. EGG-MCTS achieves substantially lower NMSE on trigonometric datasets (e.g., 0.006 vs 0.033 on (4,4,6)), and EGG-DRL shows meaningful gains in both noiseless and noisy settings. The LLM integration, while straightforward (enriching feedback prompts with equivalent expressions), also demonstrates improvements.

However, there are notable limitations:
1. The effectiveness depends heavily on the manually-curated rewrite rules—expressions without algebraic equivalences see no benefit.
2. The LLM integration is relatively basic compared to MCTS/DRL adaptations.
3. Datasets in Table 1 are selected for trigonometric identities, potentially favoring EGG.
4. Computational overhead analysis could be more thorough, particularly for MCTS.

The paper is well-written with clear exposition and helpful visualizations. The approach represents a genuine methodological contribution that is orthogonal to existing knowledge-guided SR techniques and could be combined with them.

**Score: 7.0**

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

**Assessment:**

This paper introduces EmoSign, a novel dataset for emotion recognition in American Sign Language, addressing an important and underexplored gap at the intersection of sign language processing and multimodal emotion recognition. The authors make a compelling case for the difficulty of the problem: facial expressions in sign language serve dual grammatical and emotional functions, requiring cultural and linguistic expertise to disentangle. The annotation methodology is strong, with three Deaf native ASL signers providing multi-layered annotations including sentiment ratings, emotion intensity scores, and open-ended cue descriptions. The inter-annotator agreement (average Krippendorff's α = 0.593) is reasonable and well-contextualized against established emotion datasets like MELD and IEMOCAP. The benchmarking analysis across four MLLMs with ablation conditions provides useful insights about current models' limitations in visual grounding for emotion recognition—they rely heavily on text captions and exhibit positive sentiment bias, with AffectGPT defaulting to neutral predictions in video-only conditions.

However, the paper has notable limitations. The dataset size is quite small: only 200 utterances (~16 minutes of video). While the authors cite precedents for smaller high-quality datasets, this limits the dataset's utility for training and robust benchmarking. The source material from ASLLRP is laboratory-recorded, which may not capture naturalistic emotional expressions. Additionally, using VADER—a lexicon tool designed for English text—to filter ASL videos for emotional salience is methodologically questionable, given the paper's own observation that text sentiment and visual emotional cues often diverge in sign language. The benchmark experiments, while thorough, offer limited technical novelty—the findings about text bias in MLLMs align with prior work, and the paper does not propose novel architectures or methods to address identified limitations. Finally, the absence of sign language-specific baselines (e.g., pose-based methods) leaves questions about whether specialized approaches might perform better.

**Score: 5.5**

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces Visuo-Tactile World Models (VT-WM), a multi-task world model that combines vision with tactile sensing to improve physical reasoning in contact-rich robot manipulation. The key insight—that vision-only world models fail under occlusion and cannot distinguish contact states—is both well-motivated and practically important.

**Strengths:**
The paper makes a compelling case for multimodal world models in manipulation. The methodology is sound: using Cosmos and Sparsh-X pretrained encoders with a transformer predictor employing factorized spatio-temporal attention and cross-attention for action conditioning. The training objective combining teacher forcing with autoregressive sampling is appropriate.

The empirical evaluation is thorough. The object permanence evaluation (using CoTracker and Fréchet distance) provides quantitative evidence that VT-WM reduces hallucinated motion by ~33%. The causal compliance experiments demonstrate that VT-WM better respects physical laws when objects should remain stationary. Most importantly, the real-robot planning experiments across five tasks show consistent improvements, with up to 35% higher success rates on contact-rich tasks. The data efficiency experiment comparing against behavioral cloning (77% vs 22% success with 20 demonstrations) is a nice addition.

**Weaknesses:**
The baseline comparison is limited to V-WM only. Comparing against other world model architectures or classical baselines would strengthen the contribution. The CEM-based planning requires expensive autoregressive rollouts; the paper doesn't address computational feasibility for real-time control. Generalization experiments are limited to held-out trajectories within training tasks—the model is not tested on novel objects or scenes. Several standard ablations are missing (impact of tactile encoder choice, number of sensors, temporal context length). The tactile modality is limited to Digit 360 sensors, though the authors acknowledge this limitation.

**Overall:**
This is a solid contribution addressing an important problem in robot manipulation. The core insight is valuable, the execution is competent, and the real-world validation is substantial. While not groundbreaking in methodology, the systematic demonstration that tactile grounding improves world model fidelity and planning performance is an important result for the field.

**Score: 7.0**

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (6.0/10)
- Match: N/A

### Review

This paper provides a systematic ablation study of GRPO, investigating whether its complexity is necessary for training LLMs to reason mathematically. The authors identify two key findings: (1) negative feedback is essential—positive-only training leads to collapse; (2) PPO-style clipping is unnecessary. They propose RGRA, a simplified REINFORCE variant with group-relative advantages, which performs competitively or better than GRPO across mathematical benchmarks.

The paper has several strengths. First, the research question is timely and practically important given GRPO's prominence in recent reasoning models like DeepSeek-R1. Second, the methodology is systematic—the ablation study isolates components clearly, examining positive-only advantages, direct rewards, and the removal of clipping. Third, the training dynamics analysis (Figure 1) provides valuable insight into why certain approaches collapse. Fourth, the empirical evaluation covers multiple benchmarks and model sizes, and the code is provided for reproducibility.

However, there are notable weaknesses. **Scale limitations**: experiments are conducted only on small models (0.5B, 1.5B, 1B parameters), while GRPO was demonstrated on much larger scales. It remains unclear whether findings generalize to production-scale models. **Limited training data**: using only 1,800 GSM8K examples is quite restrictive compared to typical RL training budgets. **Incremental contribution**: the insight that clipping is unnecessary aligns closely with Ahmadian et al. (2024)'s observation that pretrained LLMs have favorable properties for simpler policy gradient methods. **Missing comparisons**: no comparison to other GRPO variants mentioned in related work (CPPO, DAPO, S-GRPO). **Narrow scope**: experiments are limited to mathematical reasoning, though the title addresses "reasoning" broadly.

The performance improvements of RGRA over GRPO, while consistent, are often modest (e.g., GSM8K: 50.9%→53.1% for 0.5B). The paper would be stronger with analysis of why clipping removal helps and theoretical grounding beyond citing prior work. Overall, this is a solid but not exceptional contribution—useful for practitioners but limited in scope and scale.

Score: 6.0

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces GUI-Spotlight, a visual grounding model for GUI agents that employs iterative tool-based focus refinement. The core idea is elegant: instead of predicting coordinates directly, the model invokes tools (crop, extract, find_color) to progressively narrow down to the target element, similar to how humans might visually search a complex interface. The method is trained through a three-stage pipeline combining SFT warmup and modified GSPO reinforcement learning.

**Strengths:**
The most compelling contribution is data efficiency—achieving 52.8% accuracy on ScreenSpot-Pro with only 18.5K samples, surpassing V2P-7B (50.6%, 9.6M samples) and GTA-1-7B (50.1%, 1.56M samples). The iterative tool-use framework is conceptually sound, and the ablation studies on RL algorithm variants and reward design are thorough and valuable for future research. The three-stage training methodology with the auxiliary cross-entropy loss term appears well-motivated for stabilizing multi-turn tool-use RL.

**Weaknesses:**
However, several concerns limit my enthusiasm. First, the absolute improvement over baselines is modest (~2-3 percentage points), and the paper lacks a fair comparison against training-free iterative inference with the same base model and inference budget. Second, the complexity of the approach—custom tools, three-stage training, modified GSPO, five-component reward function—seems disproportionate to the gains achieved. Third, there's no analysis of inference cost: iterative tool calls require multiple forward passes, which could be prohibitive in real-time applications. Fourth, the paper doesn't deeply analyze failure modes—when does iterative refinement fail versus direct prediction? Finally, the comparison in Section 5.4 uses somewhat artificial baselines rather than established iterative grounding approaches from prior work.

Overall, this is a solid incremental contribution with practical value in data efficiency, but the modest absolute improvements and complexity of the approach suggest it falls short of a clear accept.

Score: 6.5

---

## XKLPlnfZzM

- GT: Reject (avg 3.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces TDDM (Temporal Deaggregation Diffusion Model), a hierarchical framework for trajectory generation that separates spatial priors (marginal distributions over geographical occupancy) from temporal dynamics. The key insight—conditioning on aggregate spatial statistics rather than sample-specific attributes—enables both controllability and cross-region generalization, addressing a genuine gap in the trajectory generation literature.

**Strengths.** The approach is well-motivated and elegant: by canonicalizing regions via similarity transforms and conditioning on spatial priors, the model learns location-invariant dynamics. This enables zero-shot transfer to unseen cities, a compelling capability demonstrated through experiments across three datasets from different continents. The empirical results are strong—TDDM substantially outperforms baselines on distributional metrics (KLsym reduced from ~1.2 to ~0.28) while maintaining fidelity. The evaluation framework is comprehensive, spanning multiple metrics for fidelity, diversity, proportionality, and usefulness, and the ablation studies meaningfully probe design choices. The generalization experiments, both intra-city (training on 25% and generating for the rest) and cross-city, convincingly demonstrate the practical value of the spatial-temporal factorization.

**Weaknesses.** Several limitations deserve attention. First, the method relies heavily on map-matched preprocessing; the ablation without map matching shows notable performance degradation, raising concerns about generalization to non-road-constrained trajectories (pedestrians, ferries). Second, the architecture itself is relatively standard—conditioning on discretized heatmaps via transformers is straightforward. Third, the "zero-shot" claim is somewhat overstated: generating for new cities requires computing spatial priors from target-city trajectory data, which is still using target-domain information even if not individual trajectories. Fourth, the paper doesn't address how trajectories spanning multiple regions are handled, limiting scalability for truly city-wide trajectory generation. Finally, while the comparison baselines are reasonable, some relevant recent work (ControlTraj, COLA) are only theoretically discussed due to lack of code.

**Overall.** This is a clear, solid contribution with practical implications for urban mobility research. The spatial-temporal factorization is elegant and the empirical validation thorough. However, the architectural novelty is incremental, and some claims around zero-shot generalization merit qualification. The work advances the field but doesn't represent a breakthrough.

Score: 7.5

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes RoPE++, an extension to Rotary Position Embeddings that reincorporates the imaginary component typically discarded in standard RoPE implementations. The authors provide both theoretical analysis and empirical validation across multiple model scales.

**Strengths:** The central insight—that standard RoPE discards imaginary information from complex-valued attention—is genuinely interesting and previously overlooked. The theoretical derivation showing imaginary attention follows a sine integral (enabling slower decay at distance) versus the cosine integral for real attention provides useful intuition for why this might benefit long-context modeling. The paper offers practical efficiency benefits: RoPE++EH achieves comparable performance with half the KV cache, which addresses a real bottleneck in long-context inference. The experimental coverage is thorough, spanning three model sizes (376M, 776M, 1.5B), multiple baselines (FoPE, ALiBi, Pythia), and both short and long-context benchmarks.

**Weaknesses:** The magnitude of improvements is modest. At 776M, RoPE++EC achieves ~2 points better average score than vanilla RoPE on short-context tasks—a meaningful but not transformative gain. More concerning, the improvements don't scale consistently: the 1.5B results in Table 5 show RoPE++EH marginally beating RoPE on some metrics while losing on others. The long-context results in Table 6 for 1.5B are inconsistent, with RoPE++EH underperforming on RULER while outperforming on BABILong. The efficiency argument for RoPE++EH comes with a caveat—halving QKV parameters alongside cache—which deserves more discussion. While theoretical analysis is solid, the claim that imaginary attention "dominates" long-context modeling relies on noise-perturbation experiments rather than more direct mechanistic interpretability. Finally, validation stops at 1.5B parameters, limiting confidence in scaling to production-relevant model sizes.

**Overall:** This is a conceptually clean contribution with solid theoretical grounding, but the empirical improvements are modest and don't clearly scale favorably. The paper sits at the borderline—interesting enough to merit consideration, but the incremental benefits and scaling uncertainty prevent a stronger recommendation.

Score: 5.5

---

## DZUehXNiBn

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper presents VISTA, a modular framework for causal structure learning that decomposes the global DAG learning problem into local subgraphs via Markov Blankets, then aggregates them through weighted voting. The approach is model-agnostic, computationally efficient through parallelization, and comes with theoretical guarantees.

**Strengths:**
- The paper addresses a meaningful scalability challenge in causal discovery. The divide-and-conquer strategy based on Markov Blankets is principled.
- The model-agnostic design is valuable—VISTA can wrap around arbitrary base learners.
- Theoretical analysis includes finite-sample error bounds and asymptotic consistency.
- Comprehensive experiments across multiple baselines and graph types.

**Weaknesses:**
- The core methodological novelty is limited. Markov Blanket-based decomposition has been explored in prior work (e.g., DCILP), and the main contributions—weighted voting and FAS-based acyclicity enforcement—are relatively straightforward extensions.
- The empirical results are mixed and somewhat misleading. VISTA-NV often *dramatically degrades* performance (e.g., NOTEARS ER5: FDR jumps from 0.21→0.87, SHD from ~209→3172, F1 from 0.76→0.23). VISTA-WV then applies aggressive filtering to recover. This framing obscures that the weighted voting is primarily compensating for noise introduced by naive aggregation.
- Improvement claims are selective. For several baselines (particularly GOLEM, GraN-DAG), VISTA-WV shows modest F1 gains only by substantially reducing TPR. A fairer comparison would involve tuning baseline thresholds directly.
- Theoretical assumptions (independent votes, known probabilities p/q) are unrealistic, limiting practical applicability.
- Real-data evaluation is minimal (only Sachs with 11 nodes), failing to demonstrate scalability claims.
- The handling of latent confounders introduced by subgraph restrictions is acknowledged but not rigorously analyzed.

While the paper has merit in formalizing the aggregation problem and providing theoretical grounding, the incremental methodological contribution and the mixed empirical picture—where the proposed method often requires aggressive post-hoc filtering to avoid substantial performance degradation—raise concerns about its practical value.

**Score: 4.5**

---

## x6bG2Hoqdf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces CALM, a framework that co-evolves heuristics and the underlying language model for automatic heuristic design. The key innovation is combining traditional prompt manipulation ("verbal gradients") with reinforcement learning fine-tuning ("numerical gradients") via GRPO, allowing the LLM to improve during the evolutionary search process. Prior LLM-based AHD methods kept the LLM frozen.

**Strengths:**
- The core contribution is genuinely novel. Fine-tuning the LLM during heuristic evolution addresses a clear limitation in prior work (EoH, ReEvo, FunSearch, MCTS-AHD all use frozen models).
- Strong empirical results across four optimization tasks (OBP, TSP, CVRP, OP), consistently outperforming SOTA baselines including those using GPT-4o-mini, while running on a single 24GB GPU with a 7B INT4 model.
- Comprehensive ablations isolating the contribution of GRPO, reward design, collapse mechanism, and individual operators. The RL component shows the most significant impact.
- Technical innovations like fine-granularity operators (injection/replacement) are well-motivated by the observation that GRPO's token-level advantage scores can be unreliable when small code changes have large performance effects.
- Practical accessibility: democratizes AHD by running on consumer hardware without requiring expensive API calls.

**Weaknesses:**
- The verbal gradient components (operators, diversity-aware crossover, collapse mechanism) are somewhat incremental over prior evolutionary AHD methods.
- Limited model variety in main experiments (primarily Qwen2.5-7B-INT4). Additional experiments with Llama and o4-mini appear in appendices but are limited.
- Each run requires 5-7 hours; comparison with API-based methods doesn't fully account for the different compute/latency tradeoffs.
- The reward function has multiple hyperparameters, and while ablations show robustness, the complexity adds implementation burden.
- Limited theoretical analysis of why GRPO should specifically improve heuristic generation.

Overall, this is a clear contribution that advances LLM-based automatic heuristic design. The co-evolution concept is original, execution is solid, evaluation is comprehensive, and the practical implications are meaningful.

Score: 7.5

---

## W42oLSwI9p

- GT: Reject (avg 5.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes three one-step diffusion-based solvers (CMILP, SCMILP, MFILP) for integer linear programming, extending prior work on binary ILP to non-binary cases through a novel Iterative Integer Projection (IIP) layer. The work addresses an important gap in the literature, as most neural ILP solvers have been limited to binary variables, and transforming non-binary problems to binary form incurs exponential blowup.

**Strengths:**
- The extension to non-binary ILP via the IIP layer is a meaningful and practical contribution. The projection function $f_{proj}(x) = x - \frac{\sin(2\pi x)}{2\pi}$ provides a differentiable approximation to integer rounding, avoiding the need for costly binarization.
- Empirical results demonstrate substantial speedups over both traditional solvers (Gurobi, SCIP) and prior diffusion-based methods (IP Guided DDPM/DDIM). The one-step models achieve solutions in seconds versus minutes or hours.
- The paper provides comprehensive comparisons across multiple datasets (binary and non-binary) and baselines. The momentum-based guidance for sampling is a reasonable addition that improves feasibility.

**Weaknesses:**
- **Large optimality gaps**: The reported gaps (10-120% depending on dataset) are quite substantial. While the paper claims practical relevance through speed, the solution quality is significantly inferior to traditional solvers, limiting real-world applicability.
- **Incremental technical contributions**: Applying consistency models, shortcut models, and mean flow models to ILP is relatively straightforward. The momentum-based guidance is essentially standard optimization technique applied to diffusion sampling.
- **Limited ablation studies**: The analysis of the IIP layer is minimal. There's no comparison with simpler approaches like direct rounding at inference time, which would help justify the proposed differentiable projection.
- **Synthetic non-binary datasets**: The inventory management and random ILP datasets are not particularly challenging or representative of real-world ILP applications. Evaluation on more realistic benchmarks (e.g., MIPLIB) would strengthen the claims.
- **Inconsistent baseline performance**: Some baselines (Neural Diving) show 0% feasibility, which appears to be an implementation issue rather than a fair comparison, raising concerns about reproducibility of baseline implementations.

The paper represents a reasonable contribution but falls short of a clear accept due to the quality-speed tradeoff heavily favoring speed, and the incremental nature of the diffusion adaptations.

**Score: 5.5**

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents VINCIE, a method for learning in-context image editing directly from video data. The key insight is that videos naturally contain temporal transitions (object appearance/disappearance, pose changes, camera movements) that can serve as implicit supervision for image editing operations, without requiring manually crafted paired editing data.

**Strengths:**
1. **Novel problem framing**: The idea of leveraging native video data for in-context image editing is genuinely innovative. Previous approaches rely on synthetic data from expert models (segmentation, inpainting), while this work demonstrates that videos provide a scalable, natural alternative.

2. **Strong empirical results**: The model achieves competitive performance on MagicBrush and shows particular strength in multi-turn editing, with success rates improving from 5% to 22% at 5-turn editing when scaling from 0.25M to 10M sessions. The comparison with GPT Image 1 and other baselines is comprehensive.

3. **Useful contributions**: The proposed MSE-Bench benchmark addresses a real gap in evaluating multi-turn editing, and the three proxy tasks (NIP, CSP, NSP) provide a principled framework for learning from interleaved sequences.

4. **Emergent capabilities**: The demonstrations of story generation, multi-concept composition, and chain-of-editing emerging from video-only training are interesting findings that suggest the approach captures meaningful representations.

**Weaknesses:**
1. **Reproducibility concerns**: The method heavily relies on proprietary components—an in-house VLM for annotation, an in-house MM-DiT video foundation model (3B/7B), and GroundingDINO+SAM2. While code is promised, the foundation models are not publicly available, making full reproduction challenging.

2. **Limited architectural novelty**: The model architecture is essentially applying existing diffusion transformers (MM-DiT) with standard attention mechanisms. The primary contribution is the data pipeline, not architectural innovation.

3. **Data quality concerns**: The VLM annotation achieves only 75% accuracy and 69% recall. While the authors argue this is acceptable for large-scale pretraining, the noise level could impact editing quality for edge cases.

4. **Incomplete ablations**: The paper lacks a direct comparison between learning from video data versus equivalent synthetic data pipelines, which would better validate the claimed advantage of native video data.

5. **Evaluation limitations**: While GPT-4o evaluation correlates with human judgment, the 100-example benchmark is relatively small, and some editing categories are underrepresented.

Overall, this is a well-executed paper with a novel perspective on a practical problem. The core insight of using video as a scalable source for editing supervision is valuable, and the results support the claims. However, the reliance on proprietary models and limited architectural contribution prevent it from being an exceptional contribution.

Score: 7.5

---

## U6ROetm5nW

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents improved algorithms for Kernel Density Estimation (KDE) in high dimensions by leveraging asymmetric locality-sensitive hashing. The main results are: (1) a query time of (1/μ)^0.05 with space (1/μ)^4.15, significantly improving upon the previous best query exponent of 0.173; (2) the first time-space tradeoffs for KDE, allowing query exponent ξ(δ) with space exponent 1+δ; and (3) a query exponent of 0.1865 in the linear space regime, improving the non-adaptive bound of 0.25.

**Strengths:** The primary strength is the novel contribution of time-space tradeoffs for KDE—a genuine addition to a well-studied problem that has seen sustained interest in the theory community. The technical approach of replacing symmetric LSH with asymmetric LSH from Andoni et al. (2017) within the Charikar et al. (2020) framework is sound and executed carefully. The linear-space result achieving exponent 0.1865 is notably close to the data-dependent result (0.173) with a simpler analysis, which has independent value. The presentation is clear with well-structured proofs and appropriate use of numerical optimization for parameter settings.

**Weaknesses:** The strongest result (exponent 0.05) requires polynomial space, making it not directly comparable to prior linear-space results and limiting practical relevance. The linear-space improvement (0.1865 vs 0.173) is modest (~8% reduction in exponent). The core technique borrows heavily from existing asymmetric LSH constructions—the innovation lies in the application rather than the technique itself. The results rely on numerical optimization rather than closed-form expressions, which limits intuition. Additionally, the plateau phenomenon (query exponent cannot go below ~0.05 even with arbitrarily large space) is an interesting negative finding but somewhat limits the utility of the tradeoff.

**Overall Quality:** This is a solid contribution to the theory of high-dimensional KDE. The time-space tradeoffs are genuinely new, the analysis is correct, and the improvements are meaningful. However, the polynomial space requirement and modest linear-space gains prevent this from being an exceptional paper. The work fits comfortably within the ICLR theory track as a clear accept.

Score: 7.0

---

## MwuSvrthXq

- GT: Reject (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents WeCAN, a reinforcement learning framework for heterogeneous DAG scheduling that addresses task-resource compatibility constraints. The authors propose a weighted cross-attention mechanism to handle varying numbers of resource pools and task types, a longest directed distance GNN (LDDGNN) for encoding DAG dependencies, and a skip action mechanism for single-pass networks to address optimality gaps inherent in list scheduling approaches.

**Strengths:**
The paper makes several solid contributions. First, the theoretical analysis of optimality gaps in list scheduling is valuable—the formal characterization of when and why list scheduling fails to find optimal solutions (through the concepts of reduced space, feasible reduced space, and generation maps) provides genuine insight into the limitations of widely-used neural scheduling approaches. Second, the skip action mechanism for single-pass networks is novel and addresses a real limitation: prior skip-action designs required multi-round inference, negating computational benefits. Third, the weighted cross-attention layer is well-motivated—it maintains adaptability to varying environment sizes while incorporating compatibility coefficients, which is preferable to fixed-dimension embeddings. Fourth, the experimental evaluation is thorough, covering multiple datasets, ablations, generalization tests, and comparisons against both heuristics and neural baselines. The gains are consistent (7-19% over neural baselines across settings) and the runtime analysis demonstrates practical efficiency.

**Weaknesses:**
Several issues moderate enthusiasm. First, while the components are well-designed, they are relatively incremental: weighted cross-attention extends standard cross-attention; LDDGNN builds directly on Graphormer/Topoformer architectures; the skip action idea itself predates this work. Second, the neural baselines are limited—PPO-BiHyb (2021) and One-Shot (2023) are the primary comparisons, and more recent specialized scheduling RL methods could strengthen evaluation. Third, the skip action's effectiveness is demonstrated primarily on synthetically modified "heavy task" datasets; on standard TPC-H instances, skip is disabled, raising questions about practical necessity. Fourth, the computational efficiency claims should be tempered—Table 1 shows WeCAN-S(256) has comparable runtime to One-Shot-S(256), so the single-pass advantage seems marginal in sampling mode. Finally, the TPC-H modifications (adding heterogeneous features) lack real-world validation on actual heterogeneous systems.

Overall, this is a solid, well-executed paper with meaningful theoretical analysis and comprehensive experiments. The contributions advance understanding of neural scheduling limitations and provide practical improvements, though individual components are incremental rather than breakthrough innovations.

Score: 7.5

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces CaTS-Bench, a large-scale benchmark for context-aware time series captioning. The work addresses a genuine gap in the evaluation of foundation models for time series understanding, where existing benchmarks rely on synthetic data, simplistic captions, or lack multimodal components.

**Strengths:**
- **Comprehensive benchmark design**: The paper integrates numeric time series, rich metadata, visual plots, and validated captions across 11 diverse real-world domains. The inclusion of both captioning and Q&A tasks provides multiple evaluation angles.
- **Rigorous quality validation**: The authors conduct extensive validation including manual verification (>98.6% accuracy on statistical claims), human detectability studies (41.1% accuracy, near random), diversity analysis, and paraphrasing robustness experiments.
- **Novel evaluation metrics**: The proposed numeric fidelity metrics go beyond standard N-gram overlap to specifically measure numeric accuracy and statistical inference—critical for time series applications.
- **Significant empirical findings**: The discovery that VLMs largely fail to leverage visual inputs (with performance sometimes improving without images) is a consequential finding supported by ablation studies and attention visualization.
- **Strong reproducibility commitment**: Detailed prompts, extensive appendices, and commitment to release data and code.

**Weaknesses:**
- **Heavy reliance on semi-synthetic captions**: The ground truth relies primarily on Gemini 2.0 Flash outputs. While validated, the human-revisited subset (579 captions) is relatively small compared to the full test set (~4k), and the oracle model may introduce subtle biases.
- **Narrow caption scope**: Captions focus primarily on descriptive statistics (mean, min, max, trends) without capturing more complex patterns like seasonality, anomalies, or causal relationships.
- **Limited human baseline for captioning**: Human evaluation covers Q&A tasks but not caption generation itself, making it difficult to contextualize model performance.
- **Somewhat incremental finding on visual modality**: While documenting VLM failures to use visual inputs is valuable, the paper could offer more insight into potential solutions rather than merely identifying the problem.

Overall, this is a well-executed benchmark paper that makes solid contributions to time series understanding and reveals important limitations in current VLMs. While it lacks methodological novelty, it provides valuable infrastructure for future research.

Score: 7.0

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

**Assessment:**

This paper identifies important methodological issues in the emerging field of online map-based motion prediction and proposes OMMP-Bench as a corrective benchmark. The work is timely, as this research area is relatively new (the cited protocol from CVPR 2024) and establishing proper evaluation practices early is valuable.

The key strength lies in systematically identifying three critical flaws in existing protocols: (1) the train-validation gap caused by using the online mapping model's predictions on its own training data, (2) the misaligned perception ranges between online mapping (±15×30m) and motion prediction (agents >100m away), and (3) non-discriminative metrics that evaluate only ego trajectories. The proposed spatially disjoint data split is a clean solution to the leakage problem, and the insight about evaluating all moving agents rather than just ego is fundamentally correct for the motion prediction task. The proposed image feature baseline, while technically straightforward, effectively addresses the out-of-range issue and demonstrates practical improvement (12.7% minADE reduction for far agents).

However, the paper has notable limitations. The technical novelty of the baseline method is minimal—the deformable attention mechanism is a standard technique applied in an obvious way. The core contributions are primarily about fixing experimental setups rather than advancing modeling capabilities. The experiments, while thorough for the tested models, cover only two motion prediction architectures (HiVT and DenseTNT) and two mapping models, limiting the scope of conclusions. Some findings (e.g., "stronger online mapping model benefits motion prediction") are intuitive rather than surprising. Additionally, while the benchmark improvements are valuable, the paper would benefit from more analysis on why certain map element types matter more and deeper investigation into failure modes.

Overall, this is a solid benchmark paper that corrects important methodological issues. It will serve the community by establishing proper evaluation practices, but the technical depth is limited.

**Score: 6.5**

---

## 31CznLfRIS

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces VideoJudge, a bootstrapping framework for training small MLLM-based evaluators (3B/7B parameters) for video understanding tasks. The core idea is elegant: a generator-evaluator pipeline iteratively produces and refines candidate responses at different quality ratings, filtering those that meet acceptance criteria to create training data without human annotation.

**Strengths:**
The paper tackles an important problem—scalable evaluation of video understanding models—where traditional metrics fail to capture semantic nuances and human evaluation is prohibitively expensive. The bootstrapping approach is creative and yields substantial training data (~104K examples). The empirical results are compelling: VideoJudge-7B matches or exceeds Qwen2.5-VL-32B/72B on multiple benchmarks, demonstrating that carefully-curated supervision can substitute for scale. The instance-specific rubric generation feature adds interpretability, and human/LLM evaluations show the generated rubrics are preferred over those from much larger models. The ablation studies on temperature robustness and frame counts are thorough, and the error analysis honestly acknowledges overestimation bias.

**Weaknesses:**
The primary concern is the partial circularity in evaluation: both training data and some meta-evaluation benchmarks (VideoJudgeLLaVA-MetaEval, VideoJudgeVCG-MetaEval) derive from the same generator-evaluator pipeline. While external benchmarks (VideoAutoArena, VatexEval) provide some validation, the reliance on the pipeline for ground-truth labels limits how much we can trust the reported gains. The calibration issues are significant—14.8% overestimation by ≥2 points versus 1.5% underestimation—suggesting the model may be learning evaluator preferences rather than genuine quality assessment. The human evaluation is limited (250 pairs in the hardest 2-vs-3 rating case), and the dependency on Qwen2.5-VL-32B/72B for bootstrapping raises questions about whether this is effectively distillation rather than truly learning evaluation capabilities.

**Overall:**
This is a solid, well-executed contribution to an emerging area. The bootstrapping approach addresses a real bottleneck, and the results demonstrate practical value. However, the circular evaluation concern and calibration issues are non-trivial limitations that prevent this from being a strong accept. The work advances the field incrementally rather than transformationally.

Score: 6.5

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important systems challenge in LLM post-training: the inefficiency caused by FSDP's per-layer collective communication under imbalanced workloads. The key insight—that collective communication creates unnecessary synchronization barriers when sequence lengths vary across devices—is well-motivated and timely, given the prevalence of variable-length sequences in modern LLM training.

**Strengths:**
The core observation is clean and impactful: FSDP's all-gather and reduce-scatter operations force all devices to synchronize at every layer, which creates straggler effects when computational loads differ. The proposed ODC method elegantly reframes FSDP as a decentralized parameter server, replacing collectives with point-to-point RDMA operations. This allows devices to proceed independently within a minibatch, reducing synchronization overhead. The technical execution is solid—the use of CUDA IPC and NVSHMEM for low-overhead communication demonstrates engineering competence. The evaluation is comprehensive across SFT and RL tasks, multiple model scales (1.5B to 32B), and includes useful ablation studies. The 36% speedup on long-sequence SFT tasks is practically meaningful.

**Weaknesses:**
The most significant limitation is inter-node performance. Figure 11 shows ODC lags substantially behind NCCL collectives for cross-node communication, which is a serious concern for large-scale training. While the authors propose hybrid sharding as mitigation, this limits applicability to smaller scales or requires additional complexity. The evaluation at only 32 GPUs doesn't sufficiently address multi-node scaling. Additionally, RL gains (10%) are notably smaller than SFT gains (36%), suggesting benefits may not generalize uniformly. The paper could benefit from deeper comparison with alternative approaches (asynchronous SGD variants, other fault-tolerant systems) and analysis of memory overhead from additional buffers.

**Overall Quality:**
This is a well-executed systems paper that makes a clear, practical contribution to LLM training infrastructure. The insight is valuable, the implementation is open-sourced, and the gains are meaningful for practitioners. However, the inter-node efficiency limitation and constrained experimental scale prevent it from reaching exceptional impact.

**Score: 7.0**

---

## eETr3lrOQB

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

**Strengths:**

This paper addresses an important practical challenge in visual tokenization research: the prohibitive computational cost of training state-of-the-art VQ tokenizers from scratch. The proposed VQ-Transplant framework is simple yet effective—replacing the VQ module in a frozen tokenizer and performing lightweight decoder adaptation. The empirical results are strong: achieving comparable or better reconstruction fidelity than the original VAR tokenizer while reducing training cost by 95% (22 hours on 2×H100 vs. 60 hours on 16×A100). The comprehensive evaluation across multiple VQ methods (Vanilla, EMA, Online, Wasserstein, MMD), multiple datasets (ImageNet, FFHQ, CelebA-HQ, LSUN-Churches), and detailed ablations on adaptation epochs strengthens the work considerably. The proposed MMD-VQ method offers theoretical advantages over Wasserstein-VQ by matching all distribution moments rather than just first two moments, which is empirically validated on synthetic non-Gaussian data.

**Weaknesses:**

The core conceptual contribution is incremental—the two-stage approach (substitution + adaptation) is fairly straightforward, and MMD for distribution matching is not novel. The practical gap between MMD-VQ and Wasserstein-VQ on real datasets is marginal (r-FID differences of ~0.01-0.05), with the main advantage only appearing in synthetic experiments. The paper lacks downstream task evaluation (e.g., image generation quality using the tokenizers), which is crucial for VQ tokenizers whose ultimate purpose is to serve as front-ends for generative models. The comparison with from-scratch training uses only 5-7 epochs—an unrealistic baseline since tokenizers typically require hundreds of epochs. Results on LDM-16 tokenizer are notably weaker (r-FID ~2.58 vs. 0.81 on VAR), suggesting limited generalization across tokenizer architectures.

**Overall Quality:**

This is a solid practical contribution that could genuinely democratize VQ research by enabling efficient experimentation without massive compute resources. The execution is thorough with comprehensive experiments and clear writing. However, the conceptual novelty is moderate, and the lack of downstream evaluation is a notable gap for a method targeting tokenizer improvement.

Score: 7.0

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a theoretical framework unifying activation steering and influence functions, showing they are dual perspectives on the same underlying sensitivity tensor. The main contribution is establishing that, to first order, any steering vector can be represented as an influence weighting over training data and vice versa.

**Strengths:**
The theoretical contribution is genuinely novel. Connecting these two previously separate areas through a principled primal-dual optimization framework is elegant and potentially impactful. The alignment diagnostic ω(x)—the cosine of the smallest principal angle between Jacobian subspaces—provides a meaningful characterization of when steering can faithfully approximate influence effects. The "no-free-lunch" theorem (Theorem 6.2) and spectral optimality result (Theorem 5.3) add theoretical depth. The generalization bounds for low-rank steering interventions via Rademacher complexity analysis strengthen the contribution. The empirical validation showing 0.978 cosine between predicted and actual logit shifts supports the first-order approximation claims.

**Weaknesses:**
The empirical evaluation is limited. Experiments use GPT-2 Medium (quite outdated) with only one primary task (detoxification) and minimal baselines (only CAA). No comparison with modern steering methods like representation engineering or recent influence estimation techniques. The practical workflow—steer first, check ω, then decide on weight editing—is proposed but not systematically validated against alternatives. First-order theory has clear limitations for practical steering magnitudes, yet the paper doesn't empirically characterize where this breaks down. Computational scalability concerns (pseudoinverses, SVDs) are acknowledged but not thoroughly addressed.

**Overall Quality:**
This is a solid theoretical contribution with clean mathematics and meaningful insights, but the empirical demonstration is too narrow to fully substantiate practical claims. The framework deserves publication, but stronger experimental validation across multiple models, tasks, and baselines would significantly improve the paper.

Score: 5.5

---

## CQ0U1wZYoy

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

**Assessment:**

This paper presents PRISM, a conditional diffusion framework for restoring scientific images affected by compound degradations. The work addresses an important practical problem: scientific imaging (microscopy, wildlife monitoring, remote sensing) often suffers from multiple overlapping distortions, and indiscriminate restoration can erase meaningful signals. The authors make three contributions: (1) a weighted contrastive learning approach that creates compositional structure in the latent space, (2) compound-aware supervision training on mixed degradations, and (3) a benchmark for scientific utility evaluation.

The strengths are notable. The problem formulation is well-motivated—standard restoration metrics (PSNR, LPIPS) don't capture whether restored images remain scientifically useful, and the downstream task evaluation across four domains (remote sensing, ecology, microscopy, urban monitoring) is a meaningful addition to standard benchmarks. The key insight that selective restoration outperforms full restoration in 3/4 domains validates the core premise. Methodologically, the Jaccard-weighted contrastive loss is a principled way to enforce compositional relationships between primitive degradations and their mixtures. The zero-shot results on unseen degradation types (underwater, under-display cameras, fluid lensing) demonstrate genuine generalization.

However, there are weaknesses. The training pipeline relies entirely on synthetic degradations, which may not capture the complexity of real-world distortions. The core architecture builds directly on Stable Diffusion v1.5 with adapted conditioning, so the novelty lies primarily in the CLIP fine-tuning strategy and training methodology rather than architectural innovation. Computational costs of diffusion models remain a barrier for widespread scientific adoption, and the paper's runtime comparison is deferred to the appendix. The comparison with OneRestore (the most directly competing composite method) could be more extensive.

Overall, this is a solid contribution with clear practical relevance and strong empirical results. The compositional latent space design and compound-aware supervision are meaningful technical contributions that advance beyond existing "all-in-one" approaches.

**Score: 7.5**

---

## VgVeQpagf7

- GT: Reject (avg 4.7)
- Predicted: N/A (8.0/10)
- Match: N/A

### Review

# Assessment

This paper presents SPS (Summarize-Privatize-Synthesize) and its enhanced variant SPS+, novel algorithms for generating differentially private synthetic datasets using dataset distillation techniques. The key claim is that this is the **first generation-based DP method to match or exceed DP-SGD accuracy** on image classification benchmarks.

## Strengths

**Significant Novel Contribution**: The paper achieves a milestone that has eluded prior work—private synthetic data generation that performs competitively with direct DP-SGD training. On CIFAR-10/100 at ε=1, SPS+ achieves 96.2%/76.6% vs DP-SGD's 94.8%/70.3%, convincingly demonstrating this advancement.

**Well-Motivated Technical Innovations**: The two key innovations—multistage clipping (iteratively refining clipping centers) and grouped pseudo-classes (reducing the O(C/N) noise rate for per-class statistics)—are principled adaptations from DP mean estimation literature. The ablations in Tables 5 and 8 validate their importance.

**Comprehensive Evaluation**: Beyond main benchmarks, the paper includes out-of-domain testing (CAMELYON17), federated learning, continual learning, ablations, and comparisons to multiple baselines (DP-Diffusion, Private Evolution, DP-KIP, etc.).

**Practical Flexibility Advantages**: The approach genuinely enables applications that are impractical with DP-SGD—ensembling without composition, federated learning without synchronization constraints, and continual learning without revisiting past data—all demonstrated empirically.

## Weaknesses

**Computational Overhead**: Generating a 50k dataset takes 8-21 hours on H100 GPUs. While the authors argue this is comparable to DP-SGD with high augmentation multiplicity, it remains a substantial cost that limits practical deployment.

**Dependence on Public Pretrained Models**: The method requires a model pretrained on public data, which is standard in recent DP work but still a limitation. The CAMELYON17 experiment helps but more extensive analysis of domain shift scenarios would strengthen the work.

**Complexity with Limited Theoretical Insight**: The algorithm combines many components (multistage clipping, pseudo-classes, class rescaling, eigenvalue clipping, SAM optimization). While ablations show each helps, there's limited theoretical analysis of why the method works as well as it does.

**FID Scores Lag Behind Generative Methods**: FID scores (22-25) are notably worse than diffusion-based approaches (Private Evolution: <7.9), suggesting the synthetic images may not capture fine-grained distributional aspects despite good downstream accuracy.

## Overall Quality

This is a strong paper making a genuine contribution to differentially private ML. Being the first generation-based method to match DP-SGD on standard image classification benchmarks is a meaningful milestone. The technical approach is sound, experiments are thorough, and the flexibility advantages are real. The limitations (compute cost, public model dependence) are acknowledged and don't fundamentally undermine the contribution.

Score: 8.0

---

## vGkXf8nvt9

- GT: Reject (avg 4.7)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes Forget-to-Focus (F2F), a two-stage protocol that first unlearns general-domain knowledge before fine-tuning for domain specialization. The core idea is that removing potentially interfering pretraining priors can improve downstream adaptation, repurposing machine unlearning from a privacy tool to a domain adaptation technique.

**Strengths:**
The paper presents a novel and creative application of unlearning. The empirical evaluation is comprehensive, spanning three domains (medical, mathematics, coding), multiple model families (Qwen, LLaMA, Gemma), and model scales from 0.6B to 72B parameters. The reported improvements are substantial—e.g., HumanEval pass@1 gains of 32.5% on Qwen-0.6B and 11.95% on Qwen-72B. The representational analysis via CKA, SVCCA, Fisher information, and PCA-shift provides useful mechanistic insight into how unlearning reshapes model geometry. The calibration analysis showing improved reliability on medical QA is a practically relevant finding. The theoretical proposition, while making strong assumptions, provides useful intuition.

**Weaknesses:**
Several concerns limit my enthusiasm. First, the forget set selection is ad-hoc—the paper uses BookCorpus but never establishes that this data contains knowledge actively *interfering* with medical/math/coding domains. The core assumption that there exists an orthogonal "irrelevant subspace" is not validated empirically. Second, the retain set is described as "a subset of D" (the target domain data), which means during "unlearning," the model is already seeing domain-specific data—this confounds whether improvements come from knowledge removal or early domain exposure. Third, the paper doesn't compare against simple alternatives like longer fine-tuning or data mixing, which could achieve similar effects more straightforwardly. Fourth, results for Gemma-2B are inconsistent, with some configurations showing collapsed performance (0.00 on HumanEval), suggesting the method may not be universally applicable. Fifth, while the calibration claim is interesting, it's demonstrated on only one dataset; broader evaluation would strengthen this. Finally, alternative mechanisms (regularization, better optimization landscape) aren't ruled out—the GA-only (σ=0) variant performs poorly, suggesting the retain set (domain data) is doing most of the work rather than the unlearning itself.

**Overall:**
The paper makes an interesting contribution with strong empirical results, but the mechanism is not fully validated, and some experimental choices raise questions about whether "unlearning" per se is responsible for the gains. The idea merits publication, but the execution has gaps.

Score: 6.5

---

## NFB4QGGS65

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper establishes a novel theoretical connection between GPTQ (a widely-used LLM quantization method) and Babai's nearest plane algorithm for the closest vector problem on lattices. The authors prove that GPTQ executed back-to-front is mathematically equivalent to Babai's algorithm on a lattice defined by the Hessian matrix, deriving tight error bounds and proposing new practical methods.

**Strengths:**
1. **Novel theoretical insight:** The connection between post-training quantization and lattice algorithms is genuinely original. This bridges two previously disconnected fields and provides the first principled explanation for why GPTQ's greedy error propagation works well globally.

2. **Rigorous proofs:** The equivalence theorem (Theorem 4) is carefully proved through multiple algebraic steps. The geometric interpretation (Section 4.2) provides intuitive understanding via hyperplane projections, making the connection accessible.

3. **Practical implications:** The theory yields concrete benefits: (a) tight error bounds for GPTQ (Theorem 5), (b) two new methods (SSQR and HPTQ) that avoid weight clipping, and (c) efficient CUDA kernels achieving ~2× inference speedup.

4. **Comprehensive experiments:** Results span multiple model families (Qwen, Llama), perplexity and zero-shot benchmarks, and comparisons with state-of-the-art methods (AQLM, QuIP#, QTIP). HPTQ achieves competitive 3-bit results.

5. **Research direction:** The paper opens avenues for importing decades of lattice algorithm research into quantization—a valuable cross-disciplinary contribution.

**Weaknesses:**
1. **No-clipping assumption:** The main theoretical results assume no weight clipping, while most practical quantization uses clipping. The authors acknowledge this but it limits direct applicability of the bounds.

2. **Concurrent work:** Birnick (2025) independently discovered similar connections. While the authors' preprint appeared first, differentiation should be clearer.

3. **Modest practical gains:** The min-pivot order, though theoretically motivated, yields marginal improvements over act-order. SSQR/HPTQ are relatively straightforward applications of the theory.

4. **Back-to-front requirement:** The equivalence only holds when GPTQ runs back-to-front, differing from standard implementations. This could cause confusion without careful reading.

5. **Limited SOTA comparisons:** While Table 16 compares to AQLM/QuIP#/QTIP, more comprehensive baselines (e.g., recent 2-4 bit methods) would strengthen practical claims.

**Overall:** This is a solid theoretical contribution that advances our understanding of GPTQ. The mathematical results are novel and well-proven, with practical methods demonstrating the theory's value. While the practical gains are incremental, the theoretical foundation and new research direction are significant enough for acceptance.

Score: 7.5

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents a careful and important empirical study testing whether standard sparse autoencoder (SAE) evaluation metrics can distinguish trained transformers from randomly initialized ones. The authors systematically compare SAEs trained on Pythia models across multiple scales (70M to 6.9B parameters) under various randomization schemes, finding that aggregate auto-interpretability scores and reconstruction metrics are surprisingly similar between trained and randomized models.

**Strengths:**
The paper addresses a fundamental and timely question for mechanistic interpretability: are current SAE evaluation metrics actually measuring what we think they measure? The experimental design is thorough, with appropriate controls (randomized-at-inference embeddings that perform at chance), multiple randomization schemes, and comprehensive metrics including fuzzing/detection AUROC, reconstruction error, explained variance, and a novel token distribution entropy measure. The finding that token entropy distinguishes trained from randomized models while aggregate interpretability scores do not is particularly valuable—it suggests that "abstractness" is an important but overlooked quality of meaningful features. The toy model analysis attempting to explain why random networks might preserve or amplify superposition adds theoretical grounding.

**Weaknesses:**
The scope is limited to Pythia models and TopK SAEs; testing other architectures (JumpReLU, gated SAEs) and model families would strengthen generalizability. The entropy metric, while promising, is preliminary—it may conflate token-specificity with feature complexity. The paper notes but doesn't deeply explore the puzzling finding that Step-0 models sometimes outperform trained models on auto-interpretability. Additionally, a causal intervention (steering) experiment demonstrating that trained-model features have functional relevance would strengthen the argument about "learned computation."

**Overall:**
This is a solid, well-executed contribution with clear implications for interpretability research. The recommendation to use randomized baselines and develop metrics capturing feature abstractness is actionable and important. While not field-defining, it's exactly the kind of careful empirical work that improves methodological rigor.

Score: 7.5

---

## piylyBPSau

- GT: Reject (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes GenCoGS, a unified approach for few-shot novel view synthesis that addresses scene completion limitations in existing 3DGS-based methods through two generative completion strategies: point cloud completion for Gaussian initialization and pseudo view completion for Gaussian optimization.

**Strengths:**
The paper clearly identifies a meaningful limitation in few-shot NVS—existing methods struggle with unobserved regions due to over-reliance on sparse training views. The proposed dual strategy is logically motivated: initializing Gaussians with a completed point cloud and optimizing with diffusion-generated pseudo views. The quantitative results are strong, with consistent improvements across LLFF, DTU, and Shiny datasets. On the challenging Shiny dataset (3-view), GenCoGS achieves a substantial 3.33 dB PSNR improvement over ReconX. The ablation studies are comprehensive, clearly demonstrating the contribution of each component. The generative consistency loss design to mitigate hallucination is a sensible approach to a real problem in diffusion-based NVS.

**Weaknesses:**
The technical novelty is limited—most components are adapted from existing work. The complementary point generation module uses standard building blocks (DGCNN, Transformer, FoldingNet), and the filtering strategy is a simple kd-tree-based thresholding. The use of diffusion models for pseudo view generation follows ReconFusion, ViewCrafter, and CAT3D closely. The design contains numerous heuristics: the sinusoidal perturbation (A=2.0, f=1.0), multiple thresholds (δ₁, δ₂, δ₃), and loss weights (α=10, β=0.1) without theoretical justification beyond empirical tuning. The computational overhead is notable (40 min vs 30 min for BinoGS), and the dependency on SfM initialization limits robustness. A minor concern: Table 2 and Table 3 show identical PSNR values (23.11) for GenCoGS on different datasets, which may warrant clarification.

**Overall Quality:**
This is a solid incremental contribution to few-shot NVS. The integration of point cloud completion with 3DGS initialization is a reasonable insight, and the hallucination mitigation strategy is practical. However, the individual components lack novelty, and the gains over prior diffusion-based methods are meaningful but not transformative. The paper would benefit from more analysis of failure cases and theoretical grounding for the design choices.

Score: 6.5

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces **random set stability**, a new framework for deriving worst-case generalization bounds over data-dependent random sets (like optimization trajectories). The main innovation is replacing intractable mutual information terms from prior topological/fractal bounds with a stability parameter, making the bounds fully computable for the first time in this literature.

**Strengths:**
The paper identifies a genuine limitation in existing work: topological generalization bounds by Simsekli et al., Birdal et al., and Andreeva et al. contain mutual information terms that are computationally intractable and can be infinite. The proposed solution—leveraging algorithmic stability—is conceptually sound. The random set stability framework appropriately extends classical stability notions, and Lemma 3.2 shows it's implied by uniform argument stability (satisfied by SGD under standard assumptions). The recovery of classical results (Corollaries 3.5-3.6) validates the framework's generality. The experiments on ViT and GraphSage demonstrate meaningful correlations between stability, topological complexity, and generalization.

**Weaknesses:**
The trade-off for tractability is steep: the convergence rate is O(β_n^{1/3}), which for SGD yields O((T²/n)^{1/3})—significantly slower than classical O(n^{-1/2}). The bounds are expected-value only, not high-probability, limiting practical utility. The empirical bounds are quite loose (Table 1 shows bounds of 68-105% vs actual gaps of 4-13%), and the stability parameter estimation is necessarily optimistic since the supremum over Z cannot be evaluated. Assumption 3.1 requires verification over all "data-dependent selections," and while Lemma 3.2 provides a sufficient condition, the local Lipschitz assumption (Assumption 4.1) may be hard to verify for practical neural networks. The contribution, while useful, is somewhat incremental—combining existing stability techniques with topological complexity measures in a natural way.

**Overall:**
The paper makes a legitimate methodological contribution by providing the first computable bounds in the topological/fractal generalization literature. However, the rate degradation, looser bounds, and incremental nature of combining existing ideas temper the impact. The work advances the field modestly but faces significant limitations.

Score: 5.5

---

## Iq1fNZus2W

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important and timely problem: the computational inefficiency of multi-condition control in Diffusion Transformers (DiTs). The authors propose PKA, combining Position-Aligned Attention (PAA) for spatial conditions and Keyword-Scoped Attention (KSA) for subject-driven conditions, along with an early-timestep sampling strategy for training. The work demonstrates substantial efficiency gains (up to 10× speedup, 5.12× memory reduction) while maintaining generation quality.

**Strengths:** The paper provides a clear empirical analysis of attention redundancy in multi-condition DiTs, showing that spatial conditions exhibit diagonal-localized attention while subject conditions are keyword-sparse. This insight is valuable and well-motivated. The efficiency results are impressive and practically meaningful—enabling complex multi-condition generation on more accessible hardware. The ablation studies are reasonably comprehensive, examining PAA against sliding window attention variants and analyzing KSA's threshold sensitivity. The early-timestep sampling strategy for training acceleration is a nice additional contribution that shows understanding of the diffusion process.

**Weaknesses:** The technical novelty is somewhat limited. PAA essentially applies masked/local attention—a well-established technique—to this specific context. While effective, it's not conceptually new. KSA has more originality but introduces complexity; the claim that the threshold is "not sensitive" seems overstated given Figure 10 shows visible quality differences with different ε values. The baseline comparison is reasonable but limited to two methods (OminiControl2, UniCombine), and the paper would benefit from comparing against more general efficient attention mechanisms. The evaluation relies heavily on automated metrics (FID, CLIP scores) without human evaluation to substantiate quality claims. Finally, testing on only FLUX.1 and limited condition types restricts generalizability assessment.

**Overall:** This is a solid, practical contribution that addresses a real bottleneck in multi-condition DiTs with meaningful efficiency gains. The approach is well-motivated and validated. However, the techniques are incremental applications of established ideas, and the quality evaluation could be stronger. The paper represents a clear engineering advance rather than a conceptual breakthrough.

Score: 7.0

---

## rI2Fa13fUL

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper introduces Generative Trajectory Policies (GTP), a new policy class for offline RL that unifies various generative models under a continuous-time ODE framework. The key insight is that diffusion models, consistency models, flow matching, and related approaches can all be viewed as learning different aspects of the same solution map Φ(x_t, t, s). The authors propose learning this full trajectory map and develop two practical adaptations: a score approximation technique that avoids costly ODE solving during training, and a variational advantage-weighted objective for policy improvement.

The strengths of this work are substantial. First, the unified ODE perspective provides genuine conceptual clarity—the paper convincingly shows how prior methods (consistency models, CTMs, shortcut models, mean flows) emerge as special cases, which is valuable for the field. Second, the theoretical grounding is solid: Theorem 1 provides a proper bound on the approximation error, and the derivation of the advantage-weighted objective from KL-regularized optimization follows established principles. Third, the empirical results are impressive, particularly on the challenging AntMaze tasks where GTP achieves dramatic improvements over prior methods (66.3 vs 44.1 for BC, 80.6 vs 69.6 for offline RL). The comprehensive ablations—covering score approximation, variational guidance, sampling horizons, and efficiency tradeoffs—strengthen confidence in the method.

However, there are notable weaknesses. The technical contributions, while sound, are somewhat incremental—the score approximation uses the same closed-form path construction as consistency training, and the advantage weighting parallels AWAC. The AntMaze results, while striking, warrant verification given how substantially they exceed prior work. Training efficiency (4.26h baseline) is not dramatically better than alternatives despite claims about computational benefits. The implementation requires careful orchestration of multiple components (EMA networks, dynamic timestep scheduling, weighted losses), raising concerns about sensitivity.

Nevertheless, the paper makes a clear contribution: it successfully adapts modern generative modeling advances to offline RL with proper theoretical justification and demonstrates compelling empirical improvements. The unified framework provides lasting conceptual value, and the practical algorithm advances the state-of-the-art.

Score: 7.5

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper presents OptMerge, a method for merging Multimodal LLMs, alongside a comprehensive benchmark for evaluating MLLM merging across capabilities and modalities.

**Strengths:**
The paper makes a valuable contribution by introducing the first systematic benchmark for MLLM merging with clear task categorization (VQA, Geometry, Chart, OCR, Grounding). The resource contribution is significant—providing both LoRA and full fine-tuning checkpoints for two model families (InternVL2.5 and Qwen2-VL) addresses a gap in the literature. The experimental evaluation is thorough, covering capability merging, modality merging, and real-world Hugging Face checkpoints. The efficiency argument is compelling: model merging requires ~21GB GPU memory compared to ~256GB for mixture training, a meaningful practical advantage. The theoretical analysis attempting to explain why fine-tuning intensity affects merging performance adds intellectual depth.

**Weaknesses:**
The methodological novelty is incremental. OptMerge builds directly on WUDI Merging with three relatively simple modifications: SVD-based low-rank approximation, SGD substitution for LoRA optimization, and mean initialization. While these components improve performance, they are not groundbreaking innovations. The theoretical result (Theorem 3.1) provides bounds but doesn't directly inform algorithm design—the assumptions are standard PL-inequality conditions that don't yield actionable insights for practitioners.

Several evaluation concerns arise. The comparison to mixture training uses Qwen2-VL-Instruct as an upper bound, which already has extensive prior SFT—an imperfect comparison. Some methods exhibit dramatic failures (e.g., Iso-C collapses on Qwen2-VL LoRA), suggesting potential instability issues. The method requires hyperparameter tuning over multiple values, and the ablation study reveals concerning interdependencies: adding SGD alone degrades performance by 9.77%, only recovering with additional components. Prior work like AdaMMS and UQ-Merge, discussed in related work, are absent from experimental comparisons. The modality merging experiments are limited to audio-visual QA without comprehensive evaluation.

**Overall Quality:**
Despite incremental methodological contributions, the benchmark resource and solid empirical work merit publication. The paper successfully demonstrates that model merging offers a viable alternative to mixture training for MLLMs, with meaningful computational savings. However, the gap between the claimed contribution and actual methodological novelty prevents a higher score.

Score: 6.5

---

## c7OsKOOZo8

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a well-motivated approach to multi-view diabetic retinopathy grading that eliminates the need for costly external lesion annotations while maintaining competitive performance. The proposed method introduces two modules: GALP (Grade-Activated Lesion Proposal), which generates self-derived lesion proposals via auxiliary classifiers and CAM-based region selection, and LGRF (Lesion Expert-Guided Regional Fusion), which uses a mixture-of-experts approach for cross-view fusion guided by lesion proposals.

**Strengths:**
The paper addresses an important practical limitation of existing multi-view DR methods—their reliance on expensive expert annotations. The motivation is clear and clinically relevant. The method is technically sound: using stage-wise auxiliary classifiers to enhance discriminability while simultaneously generating lesion proposals is an elegant dual-purpose design. Experimental results are strong, achieving 83.9% accuracy on MFIDDR without external annotations—competitive with or superior to methods requiring lesion/vessel masks. The ablation study validates each component's contribution. The comparison against both end-to-end and externally-informed baselines is comprehensive.

**Weaknesses:**
The technical novelty is somewhat incremental—CAM-based region selection and MoE fusion are established techniques combined in a sensible but not groundbreaking way. The paper lacks qualitative visualization of generated lesion proposals to support interpretability claims. Several hyperparameters (K₁, K₂, M, α) require tuning with limited guidance for new datasets. Computational overhead of the MoE routing is not analyzed. The evaluation is limited to two datasets without cross-dataset generalization analysis.

**Overall Quality:**
A solid contribution with practical impact for clinical deployment. The execution is competent and results are convincing, though the novelty ceiling prevents a higher rating. The elimination of annotation dependency while matching SOTA is valuable.

Score: 7.0

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper investigates active learning for flow matching models in shape design applications with continuous condition labels. The authors develop a theoretical framework based on piecewise-linear neural networks and closed-form flow matching models to analyze how individual data points affect model diversity and accuracy. From this analysis, they derive two competing query strategies: Q_D to enhance diversity by selecting samples with labels similar to existing data, and Q_A to improve accuracy by selecting samples with labels far from existing ones. They also propose a weighted hybrid strategy to balance this trade-off. Experiments on synthetic and real-world shape design datasets (airfoil, flying wing, starship) demonstrate that the proposed strategies outperform active learning methods designed for discriminative models.

The primary strength of this work lies in its novel problem formulation—active learning specifically designed for generative models rather than using generative models to aid discriminative active learning. This is a meaningful and underexplored direction. The theoretical framework provides interpretable insights about the inherent diversity-accuracy trade-off in dataset composition for flow matching models. The derivation of competing query strategies from first principles is elegant, and the practical relevance to domains requiring expensive numerical simulations is clear.

However, several weaknesses limit the paper's impact. First, the theoretical analysis relies on strong assumptions—piecewise-linear neural networks, closed-form flow matching models, and interpolation behavior—that may not hold well in practice for modern flow matching implementations. The gap between these assumptions and real neural networks weakens the claimed theoretical grounding. Second, while the core insight about diversity-accuracy trade-off is valuable, the proposed query strategies themselves are relatively simple combinations of existing concepts (distance-based sampling, entropy, coreset methods), offering limited methodological novelty beyond the application context. Third, the experimental evaluation has notable gaps: no comparison with random baselines, no statistical significance testing across multiple runs, no ablation on the 6% selection ratio, and limited discussion of computational overhead. The diversity and accuracy metrics, while reasonable, are not comprehensive—standard generative metrics like FID are dismissed without adequate justification. Finally, the reliance on RBF networks for label prediction introduces another source of potential error not sufficiently analyzed.

Score: 5.5

---

## Mz98kwANpF

- GT: Reject (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Assessment

This paper presents a thoughtful re-examination of multi-task LoRA architectures, challenging the prevailing paradigm that multi-component designs with task-specific isolation are necessary for effective multi-task adaptation. The work makes several important empirical observations: (1) a simplified multi-head LoRA (M-LoRA) without routing outperforms complex variants like R-LoRA despite having higher head similarity; (2) simply increasing the rank of standard LoRA can match multi-component architectures; and (3) an alignment-based approach (Align-LoRA) that explicitly encourages task-shared representations achieves superior performance while maintaining zero inference overhead.

The paper's strengths lie in its systematic challenge to conventional wisdom and strong empirical validation across multiple model families (Qwen2.5, LLaMA2, LLaMA3) and scales. The finding that task-shared knowledge may be more valuable than task-specific isolation is significant for the field. The proposed Align-LoRA method is simple, mergeable (preserving LoRA's key practical advantage), and demonstrates consistent improvements across benchmarks. The theoretical generalization bound analysis provides additional justification for the approach.

However, the paper has notable limitations. First, the core technique—using KL divergence or MMD for representation alignment—is borrowed from established domain adaptation literature, so the technical novelty is primarily in application rather than methodology. Second, while the paper convincingly shows that task-shared learning works well, it overstates the case against task-specific isolation without analyzing scenarios where diversity might actually be beneficial. Third, some comparisons are incomplete: the paper doesn't compare against recent PEFT methods like DoRA or PiSSA, and lacks comparison with full fine-tuning to contextualize the performance gains. Finally, the analysis of when Align-LoRA might fail or when the alignment loss could be harmful is limited.

Overall, this is a solid contribution with important empirical findings that question established assumptions in multi-task PEFT, though the technical innovation is moderate.

Score: 7.5

---

## 7yvz93kBw9

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses sparse-view 3D Gaussian Splatting by identifying two failure modes—overfitting in near-field regions (high Gaussian density) and underfitting in far-field regions (insufficient coverage)—and proposing complementary solutions: a depth-and-density guided dropout mechanism (DD-Drop) and a distance-aware fidelity enhancement module (DAFE). Additionally, the authors introduce IMR, a novel metric based on Wasserstein distance to measure inter-model robustness of learned Gaussian distributions.

**Strengths:** The paper provides a clear empirical analysis of failure modes in sparse-view 3DGS, with good visualizations comparing Gaussian distributions between dense and sparse settings. The proposed solutions are well-motivated and directly address the identified problems. Experimental results demonstrate consistent improvements over baselines across LLFF, MipNeRF360, and DTU datasets (0.5-0.9 dB PSNR gains). The ablation studies are thorough, and the IMR metric is a principled approach to evaluating representation stability using optimal transport theory.

**Weaknesses:** The technical contributions are somewhat incremental. DD-Drop essentially combines depth and density into a weighted dropout score with heuristic layer-based attenuation—a natural extension of DropGaussian's uniform dropout. DAFE applies depth-based loss reweighting, which has been explored in prior NeRF/3DGS works. The IMR metric measures consistency across training runs, but it's unclear whether consistency correlates with quality—a model could be consistently poor. The method introduces several hand-tuned hyperparameters (ω_depth, ω_density, λ_middle, λ_far, depth thresholds), and DAFE depends on external monocular depth estimation (DepthAnything V2), adding complexity without analyzing robustness to depth estimation errors. The computational overhead (density computation via k-NN) increases training time by ~46% (56s to 82s).

Overall, this is a solid contribution with clear problem identification and reasonable solutions, but the novelty is limited to sensible combinations of existing ideas rather than fundamental advances.

Score: 6.5

---

## ppXAVexrAM

- GT: Reject (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents ARSS, a novel approach that applies GPT-style decoder-only autoregressive models to novel view synthesis from a single image. The work is timely and addresses an interesting gap: while autoregressive models have shown promise in image generation, their application to novel view synthesis with explicit camera control remains unexplored.

**Strengths:**

The paper makes several solid contributions. First, the proposed framework cleverly adapts autoregressive generation to the multi-view setting through three key components: a video tokenizer for temporal consistency, a camera autoencoder that encodes Plücker ray coordinates as positional guidance tokens, and a spatial permutation strategy that preserves temporal causality while allowing bidirectional spatial context. The technical approach is well-motivated and coherent. Second, the experimental results are competitive with state-of-the-art diffusion-based methods, achieving comparable or better metrics on RealEstate10K, ACID, and DL3DV benchmarks. The ablation studies on token permutation and tokenizer choices provide meaningful insights into the design decisions. Third, the error accumulation analysis demonstrates that the autoregressive approach maintains better long-horizon consistency compared to baselines. Finally, the zero-shot generalization experiments on AI-generated images and DL3DV benchmark are convincing.

**Weaknesses:**

However, several issues temper my enthusiasm. First, the technical novelty is somewhat limited—the video tokenizer is adopted from VidTok, the spatial permutation idea is adapted from RandAR and related works, and Plücker coordinate encoding is standard in NVS literature. The main novelty lies in combining these elements for the NVS task. Second, the 256×256 resolution is notably low by current standards, and the paper doesn't discuss how the approach would scale to higher resolutions. Third, while results are competitive, they are not substantially better than existing diffusion methods, raising questions about the practical advantages of the AR paradigm beyond its causal structure. Fourth, inference efficiency—a key concern for autoregressive models—is not analyzed or compared against diffusion baselines. Fifth, some relevant recent baselines may be missing, and the comparison could be more comprehensive.

**Overall:**

This is a solid contribution that opens a promising research direction. The work demonstrates that autoregressive models can be effectively applied to novel view synthesis with camera control—a non-trivial achievement. While components are adapted from existing work and results are competitive rather than superior, the systematic integration and comprehensive evaluation make this a meaningful contribution to the field.

Score: 7.0

---

## qSak1Hjfdq

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

# Review Summary

## Strengths

This paper proposes a novel approach to lifelong Vision-and-Language Navigation (VLN) using Tucker decomposition for parameter-efficient adaptation. The key strengths include:

1. **Well-motivated problem formulation**: The AML-VLN setting—requiring agents to navigate across multiple scenes AND diverse environmental conditions (low-light, scattering, overexposure)—addresses a real gap in VLN research and is practically relevant for embodied AI deployment.

2. **Creative methodological approach**: The TuKA method applies Tucker decomposition to represent multi-hierarchical knowledge (scene × environment × shared) as a 4th-order tensor, which naturally decouples cross-task knowledge. This is more expressive than 2D matrix-based LoRA variants and provides a principled way to handle multi-factor continual learning.

3. **Comprehensive experimental evaluation**: The paper compares against 12 baselines including recent MoE-LoRA variants, conducts extensive ablations (tensor order, shared components, scaling), and includes real-world deployment experiments. The consistent improvements—65% SR vs 44% for the next best baseline with substantially lower forgetting (11% vs 18% F-SR)—demonstrate effectiveness.

4. **Benchmark contribution**: Extending Habitat with physics-based degradation models creates a useful testbed for the community.

## Weaknesses

1. **Incremental technical novelty**: While applying Tucker decomposition to LoRA is creative, the decomposition technique itself is well-established. The contribution is primarily in adaptation, not fundamental innovation.

2. **Inference assumptions**: The expert selection mechanism requires matching CLIP features to stored scene/environment features. This presumes some form of scenario identification is available or can be inferred, which may not hold in truly open-world deployment.

3. **Hyperparameter complexity**: The method introduces multiple hyperparameters (λ₁, λ₂, λ₃, ω, rank dimensions r₁-r₄) with limited sensitivity analysis beyond scaling experiments.

4. **Limited analysis of computational cost**: The paper does not report training/inference time overhead, which is important for practical embodied AI systems.

5. **Negative forgetting metric values**: The F-SR metric shows negative values in some cases (e.g., -3, -4), suggesting "backward transfer" improvement. While interesting, this phenomenon is not well-explained in the text.

## Overall Assessment

This is a solid contribution to lifelong VLN research. The problem is timely, the tensor-based approach is well-justified for multi-hierarchical knowledge decoupling, and the empirical results are convincing. While the core technique builds on established mathematical tools, the novel application to LoRA-style continual learning in embodied AI is valuable. The paper would benefit from deeper analysis of inference-time behavior and hyperparameter sensitivity, but the experimental coverage is already thorough.

Score: 6.5

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces LoRA-Mixer, a modular mixture-of-experts framework that routes task-specific LoRA experts through the linear projection layers of attention modules, along with a novel Routing Specialization Loss (RSL) for training robust routers.

**Strengths:**
The architectural insight of placing LoRA experts serially on projection matrices rather than replacing entire FFN/attention blocks or adding parallel branches is well-motivated—this allows experts to directly influence the core representation pathway. The proposed RSL loss is theoretically grounded with convergence analysis and generalization bounds provided; the entropy regularization to prevent over-uniform routing is a solid contribution. The empirical evaluation is comprehensive across 15 benchmarks with three different base model architectures (Transformer and SSM), demonstrating cross-architecture compatibility. The parameter efficiency claims (48% of trainable parameters vs. MixLoRA) with competitive or superior performance are compelling. The plug-and-play capability for frozen LoRAs sourced from public repositories adds practical value.

**Weaknesses:**
The core contribution is incremental—serial LoRA placement is a reasonable design choice but conceptually similar to existing hybrid MoE-LoRA approaches. The parameter efficiency comparison is somewhat misleading since LoRAHub requires zero trainable parameters. The RSL's data efficiency claim is undermined by Table 9 showing RSL underperforms the baseline at 4K training data; the explanation in A.16 appears to be post-hoc rationalization. Performance gains are consistent but not dramatic—some improvements (e.g., SST-2: 95.30→95.41) are marginal. The ablation study on RSL components could be more thorough. Cross-model transfer results (Table 5) use confusing notation that doesn't clearly separate baseline improvements from transfer contributions. The theoretical analysis assumes smooth surrogates while actual routing uses top-k, creating a disconnect between theory and practice.

**Overall:**
A solid contribution with meaningful theoretical grounding and extensive experiments, but hampered by incremental novelty, some methodological concerns, and inconsistent empirical claims. The work would benefit from fairer baseline comparisons and clearer justification for anomalous results.

Score: 5.5

---

## IdJakw2jta

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), extending the existing STVG task to handle videos of 1-5 minutes rather than typical 20-second clips. The proposed ART-STVG framework uses an autoregressive transformer with memory-augmented spatial and temporal decoders, processing frames sequentially to handle computational challenges of long videos.

**Strengths:** The problem formulation is timely and practically relevant—real-world applications require grounding in longer videos, and existing methods struggle due to GPU memory constraints and difficulty capturing long-range dependencies. The autoregressive design with selective memory banks is well-motivated, and the cascaded spatio-temporal decoder that uses spatial predictions to guide temporal localization is a sensible contribution. Empirical results are strong: ART-STVG substantially outperforms existing methods across all LF-STVG benchmarks (6-9% absolute improvements in m_tIoU), and maintains competitive performance on short-form STVG (only 1.2% behind TA-STVG). The ablation studies clearly validate the importance of memory selection and the cascaded design.

**Weaknesses:** The dataset extension methodology has limitations. Extending HCSTVG-v2 validation videos to 1-5 minutes by including more context frames before/after the original clips doesn't create "true" long-form grounding scenarios—the target event duration remains unchanged, only the surrounding irrelevant content increases. This makes the task somewhat artificial compared to genuinely long events. Additionally, all comparison methods are trained on the original 20-second training set, which may unfairly disadvantage methods designed for short videos. The memory selection strategies (text-similarity for spatial, TextTiling-inspired for temporal) are relatively simple heuristics without learnable components. While efficient GPU memory usage is claimed (7.9G vs 25G+ for others), the autoregressive inference is slower (1.09s vs ~0.5-0.7s for 64 frames), and latency will compound for genuinely long videos.

**Overall:** This is a solid contribution that addresses a meaningful problem with reasonable methodology and strong empirical results. However, the somewhat artificial dataset construction and incremental architectural novelty prevent a higher rating.

Score: 6.5

---

## XX5EZoe4ec

- GT: Reject (avg 2.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes RetrievalFormer, a dual-encoder transformer architecture for sequential recommendation that addresses two key limitations of traditional ID-softmax transformer models: O(N) inference cost and inability to score cold-start items. The approach is well-motivated—replacing exhaustive softmax scoring with ANN retrieval over learned embeddings while using feature-based item representations to enable zero-shot recommendation of new items.

**Strengths:** The paper makes a practical contribution by bridging the gap between powerful transformer-based sequential models and production deployment requirements. The dual-encoder design with feature-based item encoding is intuitive and the efficiency gains (288× speedup at 10M items) are impressive. The proposed LOOC (Leave-One-Out Cold) evaluation protocol is a meaningful methodological contribution that rigorously tests cold-start capability. The experimental evaluation is comprehensive, comparing against 12 baselines across 3 public datasets plus a production case study. The technical details on attention fusion, shared embeddings, and InfoNCE training are thorough and the appendices provide substantial implementation guidance.

**Weaknesses:** The main concern is the accuracy trade-off—achieving 86-91% of baseline Recall@20 represents a meaningful performance gap that may limit adoption. Additionally, AttrFormer's substantially higher performance on MovieLens (0.4128 vs 0.337) is not deeply investigated. The core architecture novelty is incremental; dual-encoder retrieval models are well-established, and applying transformers to the user tower with ANN serving is a logical extension rather than a breakthrough. The model's performance also degrades significantly on datasets with sparse features (33% drop in LOOC on Amazon Beauty).

The paper is well-written and addresses an important practical problem, but sits in a competitive space where the accuracy-efficiency trade-off is non-trivial and the architectural innovation is moderate.

Score: 6.5

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper provides a thorough empirical investigation into the optimization dynamics of modern recurrent models (SSMs like Mamba) compared to Transformers on associative recall and copying tasks. The central finding—that SSMs exhibit critical optimization instability confined to a narrow learning rate window—has important practical implications for how architectural comparisons should be conducted. The extensive experiments (~3,000 runs, 20,000 GPU hours) demonstrate commendable rigor.

The paper makes several solid contributions: (1) identifying learning rate sensitivity as a confounder in prior expressivity claims, (2) documenting contrasting scaling behaviors (width vs. depth), and (3) providing mechanistic insights through architectural ablations showing that convolutions enable 1-layer solutions for both architectures. The observation that single-layer Transformers show induction head-like loss dynamics despite failing the task is an interesting negative result.

However, the work has notable limitations. First, all experiments use synthetic benchmarks—while well-motivated and correlated with downstream tasks, the absence of validation on real language modeling tasks limits practical impact. Second, while the paper hypothesizes connections to vanishing gradients, there's limited theoretical depth explaining WHY the optimization landscape differs so dramatically. Third, some findings (SSMs benefit from width) largely confirm known intuitions from prior work.

The writing is clear and the empirical methodology is sound. The paper's main value lies in raising an important practical caution for researchers comparing architectures: hyperparameter tuning can dramatically change conclusions about model capabilities. This is a worthwhile contribution, though not field-advancing.

Score: 7.0

---

## d2pUyiXwcm

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

# Assessment of SCaSML Paper

## Summary

This paper introduces SCaSML (Simulation-Calibrated Scientific Machine Learning), a framework for improving pre-trained PDE solvers at inference time through defect correction. The key insight is that the error (defect) of a surrogate model satisfies a semi-linear PDE that inherits the structure of the original problem, enabling efficient Monte Carlo solution via Multilevel Picard iteration. The authors provide theoretical bounds showing the final error is the product of surrogate and simulation errors, and empirically demonstrate 20-80% error reduction across PDEs up to 160 dimensions.

## Strengths

**Theoretical Contribution**: The derivation of the Structural-preserving Law of Defect (Fact 2.3) is mathematically sound and non-trivial. The key theorem (2.5) establishing error bounds proportional to e(û)·E(M,N) and the improved scaling law (Corollary 2.6) provide rigorous justification for the method. The proofs in Appendices E and F are thorough.

**Conceptual Innovation**: The analogy to LLM inference-time scaling is timely and the defect-correction approach elegantly bridges neural network surrogates with classical numerical methods. The insight that Monte Carlo methods are well-suited for correcting high-frequency residuals (where neural networks struggle due to spectral bias) is well-motivated.

**Experimental Breadth**: Testing across multiple PDE types (linear convection-diffusion, viscous Burgers, HJB, diffusion-reaction) with dimensions up to 160d, using both PINN and GP surrogates, demonstrates broad applicability. The statistical significance tests (p ≪ 0.001) and violin plots provide robust validation.

## Weaknesses

**Practical Overhead**: While the paper argues for computational efficiency, Tables 1 shows inference times 10-87× longer than the surrogate alone. The "elastic compute" argument is somewhat undermined without clearer guidance on when the correction cost is justified.

**Hyperparameter Sensitivity**: Clipping thresholds vary from 0.01 to 10 across experiments without systematic justification. The MLP levels (N=2) and sample base (M=10) are fixed with limited ablation.

**Limited Baselines**: Comparisons are limited to naive MLP and uncorrected surrogates. Missing are comparisons to multi-fidelity Monte Carlo, variational methods, or iterative refinement approaches that also combine ML with numerical methods.

**Assumption Dependence**: The theoretical guarantees depend on Assumption 2.4/E.2 (bounded W^{1,∞} error), which may not hold for poorly trained surrogates. The paper offers no guidance on detecting when assumptions are violated.

## Overall Assessment

This is a well-executed contribution to physics-informed ML. The core idea—leveraging the structural preservation of defect PDEs—is novel and rigorously developed. While practical considerations could be addressed more thoroughly, the theoretical and empirical contributions are substantial.

Score: 7.5

---

## oiz0QHejVj

- GT: Reject (avg 4.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes CLIP-Map, a mapping-based compression framework for CLIP models that uses learnable matrices to transform pretrained weights into smaller dimensions, contrasting with traditional select-based pruning methods. The key technical contributions include: (1) Full-Mapping with Kronecker Factorization to reduce the parameter overhead of learnable mapping matrices from O(D₁²D₂²) to O(D₁D₂), (2) Diagonal Inheritance Initialization to address optimization instability by initializing mapping matrices as identity-like transformations, and (3) a unified mapping-retraining pipeline incorporating knowledge distillation.

The strengths of this work include a novel and well-motivated approach to model compression. The mapping-based paradigm offers a fundamentally different perspective from selection-based pruning, and the Kronecker factorization is a clever way to make the approach tractable. The diagonal initialization analysis (showing variance explosion with random initialization) provides good theoretical justification. Empirically, the method shows strong performance, particularly at extreme compression ratios (1%), where CLIP-Maptiny substantially outperforms TinyCLIP (e.g., MSCOCO TR@1: 15.8 vs 10.5). The paper also demonstrates faster training convergence compared to progressive compression baselines.

However, there are several weaknesses. The baseline comparison is narrow—while TinyCLIP is the primary baseline, other methods (UPop, EfficientVLM, DynaCLIP) are mentioned but not meaningfully compared, with the authors citing time constraints. This limits our understanding of how CLIP-Map compares to the broader landscape of VLM compression methods. The depth mapping via linear layer combination is relatively simplistic compared to more sophisticated depth pruning approaches. While the method shows strong results at 1% compression, the gains diminish at 50% compression (Table 1), suggesting the advantage is most pronounced in extreme scenarios. The evaluation is limited to YFCC-15M; validation on larger datasets like LAION-2B would strengthen the conclusions. Additionally, the diagonal initialization essentially performs weight inheritance at initialization—this connection to traditional methods could be made more explicit.

Overall, this is a solid contribution with clear novelty and demonstrated effectiveness in extreme compression scenarios. The mapping-based paradigm is well-reasoned and the technical execution is competent, though the evaluation scope could be broader.

Score: 6.5

---

## c2ozZYoZFd

- GT: Reject (avg 2.7)
- Predicted: N/A (8.0/10)
- Match: N/A

### Review

This paper presents a thorough re-analysis of Nguyen et al. (2024), a high-visibility ICLR 2025 Oral paper that introduced min-p sampling. The authors systematically examine four evidence streams and find serious methodological issues that invalidate the original claims.

**Strengths:**
The paper demonstrates exceptional rigor in its re-analysis. The authors identify multiple critical issues: (1) human evaluations omitted one-third of collected data without justification, (2) statistical tests were applied incorrectly (no correction for multiple comparisons, inappropriate pooling), (3) qualitative feedback was mischaracterized, (4) NLP benchmarks gave min-p unfair hyperparameter tuning advantages, and (5) community adoption claims were unsubstantiated and retracted. The "Best-of-N" methodology introduced for controlling hyperparameter tuning volume is a genuine methodological contribution with applications beyond this critique. The extensive compute budget (~6000 A100-hours) for fair benchmark sweeps demonstrates serious commitment to getting the analysis right. The paper derives actionable lessons—controlling for hyperparameter volume, proper statistical testing, data transparency—that address widespread issues in empirical ML research.

**Weaknesses:**
As a single-paper case study, the generalizability of findings is inherently limited. The GSM8K-only re-examination (due to compute constraints) leaves other benchmarks from the original paper unaddressed. While the tone is generally professional, some passages feel adversarial, though this is arguably appropriate given the severity of issues found.

**Overall Quality:**
This paper serves an essential function for scientific integrity. It exposes serious methodological flaws in a prominent paper through rigorous, transparent analysis while contributing novel methodology for fair hyperparameter comparisons. The lessons derived are valuable for authors and reviewers alike. The execution is thorough, the findings are significant, and the recommendations are actionable. While not proposing a new model or algorithm, it advances the field by establishing higher standards for empirical research.

Score: 8.0

---

## khBHJz2wcV

- GT: Accept (Poster) (avg 3.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents a framework for fine-tuning flow-matching generative models to enforce physical constraints (PDEs) while jointly inferring latent physical parameters. The core idea is to reformulate fine-tuning as a stochastic optimal control problem via Adjoint Matching, using weak-form PDE residuals as a reward signal, while augmenting the generative process with learnable latent parameter evolution.

**Strengths:**
The paper tackles an important and underexplored problem: how to enforce parameter-dependent PDE constraints in generative models when paired solution-parameter training data is unavailable. This has clear relevance to scientific machine learning and inverse problems. The methodological approach is clever—the combination of weak-form residuals (more numerically stable than strong-form) with the adjoint-matching framework provides principled fine-tuning with theoretical grounding. The extension of the memoryless noise schedule with a scaling parameter κ is a nice technical contribution. The experimental evaluation is comprehensive, spanning four PDE families (Darcy, elasticity, Helmholtz, Stokes) with different types of model misspecification, plus a natural image demonstration. The ablations (Base AM, Base AM+φ) properly isolate the contribution of the joint parameter flow. Computational efficiency is practical (20 gradient steps for Darcy).

**Weaknesses:**
The approach has significant hyperparameter sensitivity (λx, λα, λf, κ) that practitioners would need to tune carefully for each new problem—Figure 3 shows this explicitly but the paper offers limited guidance. The PBFM comparison is acknowledged to be disadvantaged since it requires training-time physics; additional comparisons to other inference-time methods would strengthen the paper. The Stokes experiments show PBFM failing to converge entirely, which warrants more investigation. The method's dependence on a pre-trained inverse predictor φ is a potential failure mode not deeply analyzed. Scaling to higher-dimensional problems or larger grids is unexplored. The implementation complexity (architectural modifications, PDE-specific weak residual formulations) may limit adoption.

**Overall:**
This is a solid contribution at the intersection of generative modeling and physics-informed learning. The approach is novel, well-grounded, and thoroughly evaluated. While practical limitations exist around hyperparameter tuning and implementation complexity, these don't undermine the core contribution.

Score: 7.5

---

## OuMNJoKJBQ

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

**Assessment:**

This paper proposes Alignment-Weighted DPO (AW-DPO) to improve LLM safety alignment by separately weighting reasoning and response components during preference optimization. The work includes a causal intervention study claiming that current alignment is superficial, a new CoT safety dataset, and the AW-DPO method.

**Strengths:** The paper presents an interesting hypothesis about why alignment fails—namely, that models use shallow pattern-matching rather than genuine reasoning. The linear probing experiments showing near-perfect accuracy on safety tasks in early layers versus delayed accuracy on reasoning tasks is compelling evidence. The AW-DPO method is intuitive: by separately scoring and weighting the reasoning trace versus final response, the method can target fine-grained failure modes (e.g., correct reasoning but unsafe response). The experimental evaluation is comprehensive across multiple models (Llama, Mistral) and attack types (writing styles, persuasion, encoding, multilingual). The paper also releases a new CoT safety dataset.

**Weaknesses:** The causal intervention argument has significant methodological issues. Deactivating neurons with high probing accuracy on CommonsenseQA and finding unchanged safety performance does not establish that safety is "superficial"—it could simply indicate that different neural circuits subserve different tasks. The claim that reasoning and safety are "decoupled" doesn't necessarily support the proposed solution. Additionally, the AW-DPO weighting scheme lacks theoretical justification; the formulation appears somewhat arbitrary. The evaluation relies heavily on GPT-4o as a judge, introducing model-specific biases, and many results show high standard deviations (e.g., 41.32% ± 28.29%), raising concerns about result stability. While utility is claimed to be preserved, notable drops appear in some configurations. The comparison with reasoning models like Phi-4 is poorly motivated—these models aren't safety-aligned, so comparing safety performance is not informative. Finally, key ablations are missing: What if only reasoning or only response is weighted? How sensitive is performance to the weight formulation?

**Overall:** This is a reasonable contribution with interesting empirical observations, but the causal claims are overstated and the method lacks theoretical grounding. The high variance in results and evaluation concerns further limit confidence in the conclusions.

**Score: 5.5**

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

**Assessment of DeepScientist**

This paper presents DeepScientist, an AI system for autonomous scientific discovery that employs Bayesian optimization over a "Findings Memory" to iteratively improve upon state-of-the-art methods. The system is evaluated on three frontier AI tasks, claiming to surpass human SOTA by substantial margins.

**Strengths:**
The work demonstrates impressive scale and ambition. Running for month-long timelines across 20,000 GPU hours, the system generated approximately 5,000 ideas and produced 21 validated scientific innovations. The Bayesian optimization framing—treating discovery as a search problem balancing exploration and exploitation—is conceptually sound, and the "Findings Memory" mechanism for accumulating knowledge is a valuable contribution. The empirical evaluation spans three diverse, meaningful tasks with legitimate baselines from ICML, ACL, and ICLR. The discovered methods (A2P, T-Detect, TDT) contain genuine technical content; the appended papers are coherent and pass automated review standards. The scaling analysis showing near-linear relationship between compute and discoveries is an interesting finding. The paper is commendably honest about failure rates (~3% success), and the analysis of why implementations fail (60% from code errors) provides useful insights.

**Weaknesses:**
Several issues limit the strength of this contribution. First, claims of "fully autonomous" discovery are undermined by the statement that "three human experts supervise the process to verify outputs and filter out hallucinations"—this represents substantial human involvement throughout the pipeline. Second, the efficiency is questionable: $100,000 for approximately 5 publishable papers raises concerns about whether this approach is practical. Third, while the discovered methods are publishable, they appear incremental rather than paradigm-shifting (e.g., applying t-distribution normalization to existing detectors, or using wavelet transforms)—valuable but not revolutionary. Fourth, the 183.7% improvement claim is less impressive given the low baseline (12-17% accuracy), where large relative gains are easier to achieve. Fifth, human reviewers noted the generated papers lack "comprehensive validation plans" and "analytical experiments," suggesting quality limitations. Finally, there's no direct comparison of research output quality against other AI Scientist systems on the same tasks—only automated paper review comparisons.

**Overall:**
This is a solid contribution demonstrating that AI systems can produce publishable research through systematic exploration. The system architecture and empirical findings advance the field. However, the substantial human supervision, low efficiency, and incremental nature of discoveries prevent this from being a transformative result. The honest analysis of failures and realistic framing of limitations are refreshing.

Score: 6.5

---

## tswBfpkwHn

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper presents the first rigorous theoretical analysis of Mamba models for in-context learning (ICL) under outlier contamination, a timely and important contribution given Mamba's growing prominence as an alternative to Transformers. The work makes several notable contributions: (1) characterizing the convergence and sample complexity of one-layer Mamba for ICL tasks with outliers, (2) proving that Mamba can tolerate outlier fractions approaching 1 while linear Transformers break down beyond 1/2, and (3) providing mechanistic insights into how the nonlinear gating suppresses outliers and biases toward local context.

The technical approach is sound, with careful proofs tracking gradient dynamics for attention parameters (W_B, W_C) and gating parameters (w). The analysis successfully decomposes into meaningful lemmas showing how relevant patterns dominate attention while gating learns to suppress outlier-containing examples. The empirical validation on synthetic data supports theoretical predictions, and the additional comparison with softmax attention in the appendix provides useful context.

However, the paper has notable limitations. The theoretical analysis is confined to one-layer single-head Mamba on binary classification with orthogonal patterns—a significant simplification. The comparison with linear attention (rather than softmax attention) understates the practical gap, and the authors acknowledge softmax Transformers achieve comparable robustness in their experiments. The data model assumptions (orthogonal patterns, linear combinations of training outliers for test outliers) are restrictive. Additionally, Table 1 reveals Mamba's vulnerability when outliers appear near the query (CQ setting), suggesting positional limitations not fully addressed theoretically.

Despite these limitations, the paper makes a valuable contribution to understanding state-space models' ICL mechanisms. The trade-off between training complexity and outlier robustness is theoretically grounded, and the insights about gating's role in local bias and outlier suppression advance understanding of why Mamba performs well in certain regimes. The work opens avenues for extending theoretical analysis to multi-layer architectures and more realistic data models.

Score: 7.5

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes AdaSVD, an adaptive SVD-based compression method for LLMs with two main components: **adaComp** (adaptive compensation for SVD truncation errors via alternating updates of U and V matrices using Moore-Penrose pseudoinverse) and **adaCR** (adaptive compression ratio assignment based on layer importance).

**Strengths:**
The paper addresses genuine limitations in existing SVD-based compression methods—namely, the lack of error compensation after truncation and uniform compression ratios across layers. The adaComp technique is technically sound, reformulating the compensation as an alternating least-squares problem with pseudoinverse for numerical stability. The empirical evaluation is comprehensive, covering multiple model families (OPT, LLaMA2, Mistral, Vicuna, LLaVA), compression ratios (40-80%), and both language modeling perplexity and downstream task accuracy. The consistent improvements over SOTA methods (SVD-LLM, ASVD, FWSVD) are notable, especially at high compression ratios where AdaSVD shows 40-60% PPL reduction. The ablation studies and combination with GPTQ quantization add practical value.

**Weaknesses:**
The contributions are somewhat incremental—alternating least-squares optimization is a known technique, and layer importance based on activation similarity is straightforward. There is limited theoretical depth: no convergence guarantees for adaComp, no analysis of why the importance metric should correlate with optimal compression ratio, and the bowl-shaped importance curve for LLaMA is observed but unexplained. Practical considerations are under-reported: inference speedup and actual memory savings are not measured, and computational cost of adaComp iterations is not quantified. The paper also lacks comparison with non-SVD compression methods (pruning, pure quantization) that would contextualize SVD's practical utility. Calibration data sensitivity is not analyzed, and some hyperparameters (bucket size, mrr selection) receive limited exploration.

**Overall Quality:**
This is a competent contribution with solid empirical results but limited theoretical novelty. The methods are sensible and well-executed, but the core ideas are straightforward adaptations of known optimization techniques. The paper would benefit from deeper analysis of why these methods work and practical efficiency measurements.

Score: 6.5

---

## 2EQPpEZtEK

- GT: Reject (avg 3.3)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper introduces DISTAR, a zero-shot TTS framework that operates in discrete RVQ code space by combining an autoregressive language model with a masked diffusion transformer. The approach uses patch-level AR drafting followed by parallel masked diffusion infilling within each patch, eliminating the need for explicit duration predictors or forced alignment.

**Strengths:**
The paper presents a technically sound and well-motivated approach that addresses real limitations in both continuous-space diffusion methods (fragility under distribution shift) and single-codebook AR models (exposure bias, limited expressivity). The results are strong: DISTAR achieves state-of-the-art WER on both benchmarks (1.66% and 1.32%) while maintaining competitive speaker similarity and quality metrics. The framework offers practical advantages including natural termination via EOS tokens, variable bitrate through RVQ layer pruning at inference time, and robust greedy decoding. The writing is clear and the ablations on decoding strategies, CFG, and patch size provide useful insights.

**Weaknesses:**
The core novelty is somewhat incremental—the combination of AR + diffusion follows DiTAR's paradigm, masked diffusion borrows from LLaDA, and RVQ representations are standard in codec literature. More concerning is the reliance on multiple hand-tuned heuristics (layer-wise and position-wise temperature shaping, hybrid sampling, repetition penalty) that appear necessary to address a "tail-first bias" in decoding, suggesting potential issues with the underlying formulation. The efficiency comparison with DiTAR is incomplete: DISTAR uses 24 NFE versus DiTAR's 10, which significantly impacts inference speed but isn't adequately analyzed. Comparisons to discrete AR baselines like VALL-E variants are missing, and subjective evaluations cover only a subset of systems on one benchmark. The approach also depends heavily on the chosen RVQ codec, with no sensitivity analysis.

**Overall:**
This is a solid contribution with good empirical results and practical benefits, but the incremental novelty and reliance on decoding heuristics prevent it from being a standout paper. The work represents a meaningful step forward in discrete TTS but stops short of being field-advancing.

Score: 6.5

---

## KsWRLyIAKP

- GT: Withdrawn (treated as Reject) (avg 3.2)
- Predicted: N/A (5.0/10)
- Match: N/A

### Review

## Assessment

This paper proposes framing lead-lag detection in financial markets as a temporal link prediction task on dynamic graphs, evaluates several TGNN architectures, and introduces a custom financial dataset. While the problem formulation is sensible and the empirical comparison is thorough, the paper has notable limitations.

**Strengths:** The reformulation of lead-lag detection as temporal link prediction is natural and well-motivated. The comprehensive evaluation of six TGNN architectures (JODIE, DySAT, TGAT, TGN, APAN, GraphMixer) plus a novel GM-TNF variant, along with a sequential LSTM baseline, provides useful comparative insights. The inclusion of statistical significance tests (Friedman test with Conover post-hoc) adds rigor. The finding that GraphMixer outperforms more complex architectures aligns with recent evidence that simpler temporal models can be highly effective.

**Weaknesses:** Several issues limit the contribution. First, there's no comparison to traditional statistical methods (Granger causality, cointegration, correlation-based approaches) or simpler ML baselines, which makes it difficult to assess whether TGNNs provide meaningful value over established approaches. Second, the graph is extremely small (37 nodes), raising questions about whether GNN methods are necessary compared to simpler approaches. Third, the threshold choices (ε=5%, τ=1) are not well-justified, and robustness analysis is missing. Fourth, the methodological contribution is limited—GM-TNF doesn't improve over base GraphMixer, so the proposed extension adds nothing. Fifth, there's no analysis of economic interpretability or practical trading utility of the detected relationships. Finally, the dataset construction relies on an anonymized API, limiting reproducibility.

The paper reads more as an application paper than a methods contribution. The core insight (temporal graphs for lead-lag) is reasonable, but executing existing TGNN models on a small proprietary dataset with limited baseline comparisons doesn't constitute a substantial advancement.

**Score: 5.0**

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.5/10)
- Match: N/A

### Review

## Assessment

This paper proposes QubitCache, a KV-cache compression method that uses quantum-inspired amplitude encoding to preserve attention patterns rather than discarding tokens. The central insight—that attention relationships between tokens carry more essential information than the tokens themselves—is well-motivated and grounded in existing literature on attention head specialization and graph-theoretic properties of attention matrices.

The empirical evaluation is comprehensive, spanning five models and six benchmarks, with consistent improvements over baselines. The 7× compression ratio with 92-97% performance retention is impressive, and the ablation studies provide useful insights. The method does show genuine improvement over prior work, particularly on multi-hop reasoning tasks where relational preservation matters most.

However, the paper has significant issues that undermine its core claims. **First**, the "quantum-inspired" framing appears to be largely decorative. The paper explicitly states it uses classical simulation (Qiskit statevector simulator), which requires O(2^n) operations for n qubits—the claimed "logarithmic compression beyond classical information-theoretic limits" is impossible to achieve on classical hardware. The memory savings demonstrably come from keeping only 15% of tokens, not from any quantum encoding advantage.

**Second**, the ablation study undermines the quantum contribution: removing quantum encoding causes only a 3.9% performance drop (0.491 → 0.472). This suggests the benefits come primarily from the hybrid token selection strategy rather than the quantum amplitude encoding. The paper never compares against a simple classical attention-weighted interpolation baseline that would isolate whether quantum formalism provides real advantages.

**Third**, Table 3 shows QubitCache at 0.55GB vs GEAR at 0.59GB—hardly the revolutionary compression claim. The theoretical claims about "beyond classical information-theoretic limits" are fundamentally misleading for a classical implementation.

The core insight about preserving relational structure is valuable, but dressing it in quantum formalism without demonstrating genuine quantum advantage weakens the contribution. The method works, but not for the reasons claimed.

Score: 4.5

---

## opU91paIvZ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important problem in AI safety and interpretability: making chain-of-thought (CoT) reasoning in language models more "monitorable" by improving faithfulness and conciseness. The authors identify a key insight—that naive RL fails because monitorability signals are too sparse under the initial policy, causing the relevant gradient term to vanish. This analysis of why standard policy gradient methods fail is a genuine contribution.

The proposed solution uses a stronger instruction-tuned model as a "prior" to transform raw CoT traces into more monitorable versions, then filters and uses these for supervised fine-tuning. Results on MMLU-Pro, GSM8K, and MATH500 show meaningful improvements: ~10% relative gain in faithfulness and up to 60% reduction in reasoning length while maintaining roughly 90-96% of base accuracy.

However, the paper has several notable weaknesses. First, the core methodology—using a capable model to generate training data for a weaker model—is not novel; it's essentially knowledge distillation applied to monitorability. Second, the absolute improvements are modest: faithfulness increases from ~15% to ~25%, meaning the model still fails to verbalize hints 75% of the time. Third, the faithfulness evaluation relies on "LLM as Judge," which introduces circularity concerns. Fourth, key baselines are missing: no comparison to alternative RL methods that might handle sparse rewards (e.g., curiosity-driven exploration), simple prompting strategies, or other distillation approaches. Fifth, the method's success depends heavily on prior quality, which limits generalizability.

The paper is well-written and addresses a timely topic, and the theoretical analysis of gradient vanishing has value. However, the incremental methodological contribution, modest empirical gains, missing baselines, and reliance on a potentially unreliable evaluation metric weaken the overall contribution.

Score: 5.5

---

## 1j0ormf8uI

- GT: Accept (Poster) (avg 5.2)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

This paper proposes a conformalized survival counterfactual prediction method for general right-censored data, providing exact marginal coverage guarantees for lower prediction bounds (LPBs) under different treatment regimes.

**Strengths:**
- **Novel theoretical contribution**: The key advance is providing exact marginal coverage guarantees, improving upon prior PAC-type guarantees in Gui et al. (2024) and Davidov et al. (2025). This is technically meaningful for high-stakes clinical applications.
- **Sound methodology**: The approach cleverly transforms the counterfactual prediction problem into a weighted conformal inference problem via importance weighting under the strong ignorability assumption. Theorems 4.1 and 4.2 provide useful theoretical guarantees, including a doubly robustness property.
- **Comprehensive evaluation**: The paper includes experiments on six synthetic settings and a real lung cancer dataset (541 patients), comparing against multiple baselines. The method achieves less conservative LPBs while maintaining coverage validity.
- **Practical relevance**: Survival counterfactual prediction with uncertainty quantification addresses an important clinical need for personalized treatment selection.

**Weaknesses:**
- **Strong assumptions**: The method requires SUTVA, strong ignorability, overlap, and independence between censoring and potential outcomes. These are standard but unverifiable, especially in observational data. The paper lacks sensitivity analysis for assumption violations.
- **Limited methodological novelty**: The core technique is applying weighted conformal prediction (Lei & Candes, 2021) to this specific problem. While the adaptation is non-trivial, the fundamental approach borrows heavily from existing work.
- **Weight estimation challenges**: Performance depends on accurate estimation of the propensity and censoring weights. With 124-dimensional covariates in the real data, weight estimation becomes challenging, though the paper doesn't thoroughly address this.
- **Sample size sensitivity**: Appendix E.1 shows degraded performance below 1000 samples, limiting applicability to smaller clinical datasets.

Overall, this is a solid contribution addressing an important gap in conformalized survival analysis. The exact coverage guarantee is a meaningful theoretical improvement, though the method's practical value depends on the validity of strong causal assumptions that are difficult to verify in observational settings.

Score: 6.5

---

## wUzBBsrdB1

- GT: Reject (avg 5.0)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

This paper makes an important contribution to the mechanistic interpretability literature by identifying and characterizing a fundamental problem in Sparse Autoencoder (SAE) training: incorrect L0 hyperparameter selection leads to feature mixing, destroying monosemanticity. The key insight—that low L0 causes SAEs to "cheat" by mixing correlated features to achieve better reconstruction—is compelling and well-supported through toy model experiments, theoretical analysis, and validation on real LLMs.

**Strengths:**
- The core problem is important and underappreciated in the literature. The critique of "sparsity-reconstruction tradeoff" plots is valuable—showing that ground-truth SAEs would score worse than incorrect SAEs on this metric is a strong argument.
- The toy model experiments clearly demonstrate the mechanism of feature mixing, and the visualization of decoder cosine similarities with true features is illuminating.
- The proposed metric (c_dec) is simple, theoretically motivated, and validated against sparse probing performance on real LLMs.
- The theoretical proof in Appendix A.5 that MSE loss incentivizes mixing when L0 is constrained below the true L0 strengthens the claims.
- The analysis of JumpReLU vs BatchTopK behavior at high L0 provides useful practical insights.

**Weaknesses:**
- The assumption that there exists a "true L0" for LLM representations is philosophically debatable. The Linear Representation Hypothesis is stated as fact rather than hypothesis, and recent work shows some representations are non-linear.
- The empirical validation is limited to Gemma-2-2b and Llama-3.2-1b at specific layers. Validation on more models and layers would strengthen the claims.
- The c_dec metric sometimes shows a flat region or lacks a clear global minimum (acknowledged by authors), making the "elbow" heuristic somewhat subjective.
- No comparison to alternative L0 selection methods (MDL-SAEs, AFA-SAEs) on the same benchmarks.
- The automatic L0 optimization in Appendix A.11 requires significant hyperparameter tuning, limiting practical applicability.
- Sparse probing is an imperfect proxy for feature correctness—it measures recoverability for specific tasks, not monosemanticity.

The paper is well-written, the experiments are thorough, and the contribution is meaningful for practitioners. While not field-transforming, it provides concrete methodology that can improve SAE research immediately.

Score: 7.5

---

## 7L7kmHHfgf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.5/10)
- Match: N/A

### Review

## Assessment

This paper presents PIRN, a prototype-based reconstruction framework for few-shot multimodal anomaly detection that addresses real limitations in existing approaches. The work is well-motivated: cross-modal alignment methods struggle with limited training data, and memory-based approaches misclassify unseen normal variations. The proposed solution uses three technically sound components—BPA for preventing codebook collapse via optimal transport, APR for test-time prototype adaptation via GRU gates, and MNC for cross-modal knowledge exchange at the prototype level.

The empirical results are consistently strong across MVTec-3D-AD, Eyecandies, and Real-IAD D3 datasets, with meaningful improvements over baselines in few-shot settings (+3.9% AUROC_I at 5-shot on MVTec, +4.0% on Eyecandies). The method is also computationally efficient, requiring 85% fewer FLOPs and being 4.35× faster than recent SOTA. The ablation studies are comprehensive, covering each component, codebook size, decoder depth, and aggregation methods.

However, the novelty is somewhat incremental. Each component—optimal transport for balanced assignment, GRU-based prototype updates, and cross-attention for multimodal fusion—builds on established ideas in the literature. While the integration is thoughtful and well-executed, none of the individual contributions are surprising. The paper would also benefit from deeper analysis of failure cases and sensitivity to hyperparameters like the Sinkhorn regularization strength.

The dependency on frozen DINOv2 features means the representation quality is largely inherited, and the Real-IAD comparison with D3M is slightly unfair given D3M uses three modalities. Nevertheless, the paper addresses an important problem with a practical solution that demonstrably works. The visualization of prototype displacement and codebook utilization provides useful insights into why BPA helps.

Overall, this is a solid contribution to multimodal anomaly detection with clear practical value. The approach is technically sound, experiments are thorough, and writing is clear—but the contribution is evolutionary rather than revolutionary.

Score: 7.5

---

## wSbVv6xaRr

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces FedMPDD, a novel federated learning algorithm that simultaneously addresses communication efficiency and privacy preservation against gradient inversion attacks. The core idea—encoding gradients via multi-projected directional derivatives—is elegant and well-motivated.

**Strengths:**
The paper makes a genuine contribution by recognizing that the rank deficiency of random projections creates inherent privacy protection while also enabling communication reduction. The theoretical analysis is rigorous: Lemma 1 establishes the expected gradient reconstruction error, Lemma 2 provides data reconstruction lower bounds, and Theorem 2 shows O(1/√K) convergence matching FedSGD. The connection to Johnson-Lindenstrauss for dimension-independent convergence is a clever theoretical insight. The empirical evaluation is comprehensive across multiple datasets (MNIST, FMNIST, CIFAR-10), architectures, and baselines including QSGD, Top-k, LDP variants, and sketching methods.

**Weaknesses:**
Several concerns remain. First, the multi-round privacy bound (T < d/m) is a critical limitation—modern FL often runs thousands of rounds, yet this constraint would force m close to d, eliminating communication savings. The paper mentions this but doesn't adequately address practical implications. Second, the computational cost on clients (O(dm) per round) could be substantial for large models; while JVP optimization is discussed, the experiments don't appear to use it. Third, the comparison to LDP is not entirely fair—SSIM measures reconstruction quality, not formal privacy guarantees, so comparing "LDP with variance 10" to FedMPDD without proper DP accounting misses important trade-offs. Fourth, the experimental m values (e.g., m=600 for d≈300k parameters) fall below the theoretical requirement m = O(ln(d)/ε²), raising questions about whether convergence guarantees hold in practice.

**Overall:**
This is a solid contribution with a novel core idea that elegantly addresses two practical FL challenges simultaneously. The theory is sound and experiments support the claims. However, the practical privacy guarantees under realistic training scenarios and computational overhead require more careful treatment. The paper represents a meaningful advance but with notable limitations that should be acknowledged.

Score: 7.0

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses an important and underexplored problem in LLM generation: the volatility of output length across multiple generations. While prior work has noted that models struggle to generate long text, this work highlights that the inconsistency itself—rather than just the failure to meet length requirements—is a significant practical concern for reliability and cost predictability.

**Strengths:**
The paper makes three solid contributions. First, VOLTBench is a well-designed benchmark that systematically evaluates both structured and unstructured generation tasks across languages, complexity levels, and extreme length scales (up to 500 sections). The introduction of volatility metrics (LSD, LVC) alongside quality metrics provides a more complete picture of model capabilities. Second, the mechanistic analysis through attention traces is a meaningful addition—identifying "Attention Collapse" and "Attention Instability" as failure patterns adds depth beyond purely empirical observation. Third, the empirical results are strong: SELB achieves dramatic improvements in both length accuracy and stability while maintaining quality.

**Weaknesses:**
Several concerns limit my enthusiasm. First, the proposed method is relatively simple—essentially logit manipulation to force section transitions and suppress stop tokens. While effective, this is closer to careful engineering than a fundamental algorithmic contribution. Second, the method heavily depends on explicit section structure being present in the prompt; the free-form extension (SELB-Hybrid) appears less principled. Third, the benchmark tasks feel artificial—who needs 500 sections or 20,000-word novels? More realistic tasks would strengthen relevance. Fourth, key hyperparameters (β, τmax) receive no sensitivity analysis. Fifth, the method is primarily evaluated on Qwen2.5-7B, with limited demonstration of cross-architecture generalization. Finally, comparison against iterative prompting or plan-and-write approaches would strengthen the baseline comparison.

Overall, this is a solid contribution with a well-designed benchmark and interesting mechanistic insights, though the method itself is relatively straightforward. The problem importance and empirical thoroughness outweigh the methodological simplicity.

Score: 6.5

---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

## Assessment

This paper presents a learnable three-channel codec inspired by the Gray-Wyner network from information theory, aiming to separate common information from task-specific details across multiple vision tasks. The work bridges classical information theory concepts with modern neural compression.

**Strengths:**
The paper's primary strength is its solid theoretical foundation. The extension of lossless common information concepts (Wyner's and Gács-Körner common information) to the lossy case via Theorem 1 is a meaningful contribution. The bounds relating interaction information to these common information measures provide insight into the fundamental tradeoffs between transmit and receive rates. The derivation of a practical optimization objective from Gray & Wyner's formulation (Theorem 2) is elegant and directly connects theory to implementation.

The experimental evaluation is thorough, covering synthetic data, controlled MNIST experiments with varying mutual information levels, and real vision tasks (Cityscapes, COCO). The inclusion of multiple baseline architectures (Joint, Independent, Separated, Combined) allows clear understanding of design choices. The BD-rate improvements over Independent coding are substantial.

**Weaknesses:**
The paper is limited to two-task scenarios, with only a brief mention that extensions to more tasks face exponential channel growth. This significantly limits practical applicability, as many real-world scenarios involve more than two tasks.

There is a notable gap between theoretical bounds and empirical performance—the rates achieved are substantially higher than theoretical predictions. While acknowledged, this raises questions about how close the method gets to optimal in practice.

The comparison landscape is incomplete. The related work mentions multi-task learnable codecs and disentanglement methods in VAEs, but no empirical comparison to these alternatives is provided.

The empirical results on real vision tasks show that the method produces relatively high absolute rates (as noted in the paper), and the task performance gains over the Joint baseline are not extensively discussed.

**Overall Quality:**
This is a solid contribution that successfully bridges classical information theory with practical neural codec design. The theoretical contributions are novel and sound, and the experimental validation is comprehensive. The main limitations are the restriction to two-task scenarios and the gap between theory and practice. The paper would benefit from more discussion of scaling to multiple tasks and comparisons with alternative disentanglement methods.

Score: 7.0

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.5/10)
- Match: N/A

### Review

## Assessment

This paper addresses the important problem of "copy-paste" artifacts in identity-preserving image generation, where models tend to replicate reference images rather than synthesizing diverse outputs with consistent identity. The authors make three contributions: (1) MultiID-2M, a large-scale paired multi-identity dataset; (2) MultiID-Bench, a benchmark with metrics quantifying copy-paste artifacts; and (3) WithAnyone, a FLUX-based model using contrastive training to balance identity fidelity with generation diversity.

**Strengths:** The identification of copy-paste artifacts as a fundamental failure mode is insightful and underexplored in prior work. The MultiID-2M dataset (500k paired group photos with reference images) is a substantial resource that addresses a genuine data bottleneck—the field has relied on reconstruction-based training due to scarcity of paired data. The benchmark introduces thoughtful metrics: SimGT (similarity to ground-truth rather than reference) and a copy-paste metric that captures the relative bias toward reference versus target. The GT-aligned ID loss is clever, avoiding noisy landmark extraction during training by using ground-truth landmarks. The ID contrastive loss with extended negatives effectively leverages the dataset structure. Experimental results are strong—with state-of-the-art SimGT while achieving lower copy-paste scores than competing methods. The user study validates metric correlation with human judgment, and full open-sourcing enhances community value.

**Weaknesses:** The copy-paste metric normalization by θtr could be unstable when reference and ground-truth embeddings are similar. The dataset construction relies on celebrity web-scraping with clustering (threshold 0.4-0.5), but quality control for mislabeled identities isn't thoroughly discussed. The nationality distribution is heavily China/USA biased. Computational cost is significant—8 H100s for four-phase training—yet efficiency analysis is absent. Architectural novelty is limited; the core contribution is the training methodology rather than novel architecture. Some baseline comparisons mix closed APIs (GPT-4o) with open models, creating fairness concerns. The SigLIP integration for controllable attribute retention is mentioned but lacks thorough ablation.

Overall, this is a solid contribution addressing an important problem with valuable dataset/benchmark resources. However, metric stability concerns, computational overhead, and limited architectural novelty prevent a higher assessment.

Score: 6.5

---

## s7oURFZTQD

- GT: Reject (avg 3.0)
- Predicted: N/A (4.0/10)
- Match: N/A

### Review

This paper investigates why Multi-Grade Deep Learning (MGDL) outperforms standard end-to-end training (SGDL), providing both theoretical analysis and empirical evaluation. The authors present convergence theorems for gradient descent, show that single-layer ReLU MGDL grades reduce to convex subproblems, analyze eigenvalue distributions of iteration matrices, and benchmark MGDL against SGDL on image tasks and classification.

**Strengths:**
The empirical work is extensive, covering image regression, denoising, deblurring, and classification. The eigenvalue analysis provides interesting insight into why MGDL may exhibit more stable training—the observation that eigenvalues stay within (-1, 1) for MGDL while SGDL's often fall below -1 is a useful empirical finding. The paper is well-organized and clearly written.

**Weaknesses:**
The theoretical contributions are limited. Theorems 1, 2, and 5 present standard GD convergence results with no novelty—they require bounded iterates and appropriate learning rates, which are classical assumptions. The convexification result (Theorem 3) is a direct application of Pilanci & Ergen (2020) to individual grades, not a fundamentally new insight. Crucially, the theory applies only to full-batch GD, while all experiments use Adam optimizer—this disconnect undermines the theoretical claims.

The experimental comparison raises fairness concerns. SGDL and MGDL use different architectures and hyperparameter regimes, and MGDL has access to per-grade learning rates, giving it inherent tuning advantages. The paper lacks comparison to relevant baselines: layer-wise training (Bengio et al., 2006), progressive training methods, and simple stabilization techniques like gradient clipping. For CIFAR-100, using MSE loss instead of cross-entropy makes results incomparable to standard benchmarks. The connection between MGDL and established layer-wise pretraining approaches is not clearly distinguished, raising questions about novelty.

**Overall:**
The paper offers empirical insights but limited theoretical contribution. The experimental methodology has fairness issues, key baselines are missing, and the theory-practice disconnect is significant. While the eigenvalue observations are interesting, the causal claims about why MGDL outperforms SGDL are not rigorously established.

Score: 4.0

---

## 41JeFWdVFa

- GT: Reject (avg 4.7)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper proposes LDP, a lightweight denoising autoencoder plugin for single-image super-resolution that improves generalization to unseen degradations. The core idea is to model the degradation process within a DAE framework, using LR high-frequency components as conditioning to distinguish different LR images corresponding to the same HR image. LDP can operate as either a training-time loss or an inference-time post-processing module.

**Strengths:** The paper presents a reasonably novel approach by connecting diffusion model principles with degradation modeling for SR. The lightweight design (642K parameters) is practical, and the flexibility to work in two modes adds value. The empirical evaluation is comprehensive—testing across four SR architectures (FeMaSR, StableSR, SwinIR, MambaIR) and four diffusion models on both synthetic and real-world benchmarks. The consistent improvements across most settings demonstrate the method's effectiveness. The ablation studies cover key design choices including patch size, frequency band selection, and scale factors.

**Weaknesses:** The core contribution builds on existing degradation modeling work (DRN, DualSR, Lway) without sufficient theoretical novelty. The conditioning mechanism using LR high-frequency components lacks deep justification beyond empirical validation. While improvements are consistent, they are modest (typically 0.3-0.8 PSNR). Some real-world results show metric regressions (e.g., FeMaSR's CLIPIQA drops significantly), which the paper attributes to metrics favoring artifacts—but this feels like post-hoc reasoning. The inference overhead for posterior sampling is substantial (Table 13 shows 9x slowdown in the full version). The comparison with Lway relies on re-implementation since official code is unavailable, raising fairness concerns. Finally, the patch-dependent noise addition mechanism, while interesting, lacks thorough empirical justification beyond basic ablation.

**Overall:** This is a competent paper with practical contributions and thorough experimentation, but the methodological novelty is incremental and the empirical gains are modest. The work is publishable quality but not field-advancing.

Score: 5.5

---

## XIAta0WOJ6

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.0/10)
- Match: N/A

### Review

This paper proposes F²SA-p, a family of fully first-order methods for stochastic bilevel optimization that leverages p-th order finite difference approximations to achieve improved complexity bounds under higher-order smoothness assumptions. The key insight is reformulating the prior F²SA method as a forward difference approximation to the hyper-gradient, which naturally generalizes to higher-order finite difference schemes.

**Strengths:**
The paper makes a meaningful theoretical contribution by improving the SFO complexity from Õ(ϵ^{-6}) to Õ(pϵ^{-4-2/p}) for p-th order smooth problems. The finite difference perspective is elegant and provides a principled framework for algorithm design. The paper is comprehensive, including both upper bounds and a matching Ω(ϵ^{-4}) lower bound via reduction from single-level optimization, establishing near-optimality for large p. The presentation is clear, with good motivation from practical problems like data hyper-cleaning and learn-to-regularize.

**Weaknesses:**
The dramatic improvements require high-order smoothness (Assumption 2.5), which may be restrictive in practice. For p=1 (the standard first-order smooth setting), the improvement is only a factor of κ in condition number, which is modest. The per-iteration cost increases with p (solving p lower-level problems), and this trade-off isn't rigorously analyzed. The condition number dependency (κ^{9+2/p}) remains significantly worse than HVP-based methods. Experimental validation on logistic regression is limited—more challenging tasks would better demonstrate practical benefits. The finite difference connection was recently noted by Chayti & Jaggi (2024) for meta-learning, though extending it to general bilevel optimization is novel.

**Overall:**
This is a solid theoretical contribution with a nice conceptual framework. While the practical applicability for standard first-order smooth problems is limited, the work advances our understanding of achievable complexity under additional structure and properly addresses the near-optimality question.

Score: 7.0

---

## GRufFX1gAy

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.0/10)
- Match: N/A

### Review

## Assessment

This paper introduces InnoGym, a benchmark for evaluating AI agent innovation beyond mere correctness, proposing a two-dimensional framework measuring both performance gain over known solutions and methodological novelty. The work addresses an important gap: existing benchmarks conflate solution correctness with innovation, failing to distinguish between merely tuning conventional methods versus discovering genuinely novel approaches.

The paper's strengths include its principled formal framework (T = (P, S, V, D)), thoughtful task curation from real competitions and classical problems, and the iGym unified execution environment. The validation of the distance function D_AGENT against human judgments is methodologically sound, and the empirical finding—that agents achieve novelty without robustness—is insightful. The observation that "the primary bottleneck for agents on complex tasks is not a deficit of novel ideas, but rather the inability to translate them into correct and robust implementations" is a meaningful contribution to understanding current agent limitations.

However, significant concerns limit my enthusiasm. First, the benchmark scale is modest—18 tasks with experiments on only 10—and the paper does not convincingly argue why only "improvable" tasks merit inclusion. Second, the novelty metric relies on an LLM-as-judge approach (Codex/GPT-5), which introduces potential bias and opacity; the validation experiments, while encouraging, cover limited ground. Third, all tested agents show negative performance gains (G < 0), preventing analysis of the high-G, high-N regime where true "breakthrough innovation" would occur. This means the key claim—about the gap between novelty and effectiveness—rests on an incomplete picture. Fourth, the reference to "GPT-5" with a 2025 citation raises reproducibility concerns and questions about experimental authenticity. Finally, the novelty metric captures only methodological distance; factors like computational efficiency, theoretical elegance, or long-term impact remain unaddressed.

The paper is clearly written and makes a genuine contribution to agent evaluation methodology, but the limitations in scope, metric validity, and experimental completeness prevent it from being a truly compelling advance.

Score: 6.0

---

## mDuTDAK6KU

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.5/10)
- Match: N/A

### Review

This paper presents KOALA, an adversarial detection method that leverages disagreement between KL-divergence and L0-based similarity metrics to flag adversarial inputs. The key contribution is a formal proof of correctness guaranteeing detection when certain prototype separation conditions are satisfied.

**Strengths:**
- **Novel theoretical foundation**: The paper provides a rigorous mathematical guarantee for adversarial detection—a rarity in this empirically-driven field. The proof establishes that under norm-bounded perturbations and sufficient inter-class separation, adversarial examples cannot simultaneously fool both metrics.
- **Well-motivated metric combination**: The intuition that KL divergence captures dense perturbations while L0 captures sparse, high-impact changes is sound and the mutual exclusivity argument is theoretically compelling.
- **Clean experimental validation of theory**: Table 1 confirms perfect recall (1.0) for theorem-compliant samples, supporting the theoretical claims.
- **Practical advantages**: The method requires only clean-image fine-tuning without adversarial training or architectural modifications.

**Weaknesses:**
- **Severely limited scope of guarantee**: The theoretical guarantee only applies when specific conditions hold. Crucially, only ~10% of CLIP/Tiny-ImageNet samples are theorem-compliant, limiting practical applicability of the guarantee.
- **Poor performance outside theoretical regime**: Non-compliant samples show substantial degradation (e.g., 0.42 recall on ResNet/CIFAR-10), meaning the detector fails when its theoretical conditions don't hold.
- **No adaptive attack evaluation**: The paper evaluates only standard attacks (PGD, CW, AutoAttack) without testing against adaptive attacks designed to evade this specific detector—a critical oversight for a detection method.
- **Missing comparison to prior work**: Despite extensive related work discussion, there's no direct experimental comparison to existing detectors like MagNet, feature squeezing, or LID-based methods.
- **Strong assumptions**: Assumption A4 (both metrics agree on clean inputs) and A1 (normalized features) require specific training and may not generalize across architectures.
- **Incomplete practical discussion**: When detection triggers, the system outputs ⊥ (abstention), but there's no discussion of fallback strategies or deployment implications.

The theoretical contribution is interesting and the proof is technically correct, but the narrow applicability conditions and incomplete experimental evaluation (missing adaptive attacks and baseline comparisons) are significant limitations. The paper demonstrates that formal guarantees are possible but only under restrictive conditions that often don't hold in practice.

Score: 5.5

---

