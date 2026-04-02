# ICLR Benchmark Results

Date: 2026-04-01 22:28
Critic/Merger: minimax/minimax-m2.7 (OpenRouter)
Neutral: minimax/minimax-m2.7, Related Work: minimax/minimax-m2.7:online (OpenRouter)

## P7f55HQtV8

- GT: Accept (Poster) (avg 6.5)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
QuaDiM introduces a novel non-autoregressive conditional diffusion model for quantum state property estimation (QPE), addressing the fundamental limitation that autoregressive models impose sequential ordering on qubits that lacks physical justification. The method iteratively denoises Gaussian noise conditioned on Hamiltonian parameters to generate quantum measurement samples, then uses classical shadow for post-processing. Evaluated on the 1D anti-ferromagnetic Heisenberg model with up to 100 qubits, QuaDiM demonstrates superior performance over autoregressive baselines (RNN, LLM4QPE) and classical methods (classical shadow, kernel methods) in predicting correlation and entanglement entropy, particularly under limited measurement data conditions.

## Strengths
- **Novel contribution**: First non-autoregressive conditional generative model for QPE using diffusion models; the core idea that treating qubits equally avoids sequential bias is physically motivated and technically sound
- **Comprehensive evaluation**: Extensive experiments across system sizes (L ∈ {10, 40, 70, 100}), two tasks (correlation, entanglement entropy), multiple baselines, and thorough ablations (Tables 3-10 in appendices)
- **Consistent empirical advantage**: QuaDiM outperforms all baselines across nearly all configurations, with improvements validated under different sample complexities, training data sizes, positional encoding schemes, and POVM choices
- **Practical relevance**: Demonstrates superior sample efficiency under resource-constrained conditions (limited training measurements Min, reduced inference samples Mout), which is valuable for real quantum computing applications
- **Scalability to 100 qubits**: Successfully handles large quantum systems with competitive performance, making it relevant for near-term quantum applications

## Weaknesses
- **Missing non-autoregressive baseline comparisons**: The paper claims superiority of non-autoregressive approaches but doesn't compare against VAEs, normalizing flows, or other non-autoregressive generative methods mentioned in related work. This limits the ability to assess whether the diffusion-based approach specifically provides advantages over other non-autoregressive alternatives
- **Limited physical system diversity**: All experiments use 1D Heisenberg and XY models; no 2D lattice experiments or different Hamiltonian families (e.g., transverse-field Ising, Fermi-Hubbard) are shown. This undermines claims about scalability to "real-world quantum computations" like IBM's 2D qubit arrays
- **Ground truth approximation unquantified for L > 10**: For large systems, ground truth labels are approximated via classical shadow with M=320,000 samples. The paper never quantifies the error in this approximation. If the "ground truth" itself contains non-trivial error, the reported RMSE improvements may be overstated
- **OOD generalization gap**: Table 7 shows RMSE nearly 4x worse for OOD vs. in-distribution (e.g., 0.0117→0.0417 for L=70). The paper markets the method as generalizing to unseen systems, but substantial OOD degradation suggests potential overfitting to the training distribution
- **Incomplete positional embedding discussion**: The paper claims "equal, unbiased treatment of all qubits" yet Appendix F.5 shows positional embeddings are necessary—removing them significantly degrades performance. The paper should clarify whether positional information reintroduces ordering bias and what "equal treatment" precisely means

## Nice-to-Haves
- **Statistical significance reporting**: Standard deviations are buried in supplementary tables; at minimum, prominent reporting of mean ± std for main results with experiments across multiple random seeds would strengthen confidence
- **Memorization analysis**: With N_tr=100 training Hamiltonians and 1000 measurements each, experiments testing whether the model memorizes specific Hamiltonians vs. learning generalizable representations would be informative (e.g., testing with fewer training samples)
- **Generated distribution visualization**: Show histograms or KL divergence between generated and true measurement statistics—not just downstream properties—to reveal whether the model learns correct distributions or matches only low-order statistics
- **Failure mode analysis**: Understanding when QuaDiM fails would be more informative than cherry-picked successes

## Novel Insights
The paper's key insight—that autoregressive models impose physically unjustified sequential ordering on qubits, and that non-autoregressive diffusion models can treat qubits equally—is both novel and compelling for the quantum ML community. The empirical demonstration that this architectural choice leads to consistent improvements across system sizes and metrics suggests the sequential bias in autoregressive approaches is a genuine limitation rather than a minor detail. Additionally, the demonstration that diffusion models can bridge discrete quantum measurements to continuous representations while maintaining quantum physical properties opens a promising research direction for quantum machine learning.

## Potentially Missed Related Work
- Normalizing flows for quantum state representation/tomography (mentioned in related work discussion but not used as baselines)
- VAE-based quantum state learning approaches (Rocchetto et al., 2018) as alternative non-autoregressive baselines
- Consistency models or other diffusion acceleration techniques that could address the computational cost concern

## Suggestions
1. Add VAE and normalizing flow baselines to substantiate the claim that non-autoregressive approaches are superior, not just diffusion models specifically
2. Quantify the error in classical shadow ground truth approximations (L > 10) to ensure RMSE comparisons are meaningful
3. Clarify the positional embedding discussion—either provide theoretical justification for why positional information doesn't reintroduce ordering bias, or acknowledge this as a limitation
4. Include statistical significance testing with multiple random seeds prominently reported in main results

---

## gInIbukM0R

- GT: Reject (avg 2.5)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes a quantitative framework to measure "Emergence" in neural networks, defined as the number of paths from inactive to active nodes, and studies how this measure correlates with training dynamics and pruning behavior. The authors claim higher Emergence correlates with improved final performance while higher "relative Emergence" (normalized by parameter count) explains faster convergence in pruned networks. The framework is evaluated on MNIST, Fashion-MNIST (MLP), and CIFAR-10 (VGG19) with magnitude-based pruning experiments.

## Strengths
- **Novel theoretical connection**: The paper attempts to formalize emergence in neural networks using a categorical/quiver representation framework, providing a mathematically grounded approach to quantifying a phenomenon often discussed only informally.
- **Insightful observation on training dynamics**: The finding that Emergence systematically decreases during training as networks specialize (with zero Emergence coinciding with convergence) is an interesting empirical observation that could have practical value for monitoring training progress.
- **Relative Emergence insight**: The distinction between absolute Emergence (scale-dependent) and relative Emergence (efficiency metric) provides a useful lens for understanding why pruned networks can converge faster despite having fewer parameters—a non-obvious insight.

## Weaknesses
- **Arbitrary threshold without justification**: Active nodes are defined by activation > 0.05 with no theoretical or empirical justification. No sensitivity analysis is provided, making it impossible to assess whether results are robust to threshold choice or if findings are artifacts of a particular threshold.
- **Insufficient statistical rigor**: All experiments appear to be single runs with no error bars, variance metrics, or statistical tests. Performance differences are negligible (e.g., MNIST final accuracies: 95.7%, 95.7%, 95.6%, 95.1% across pruning levels). The claim that "higher Emergence correlates with improved training performance" is not supported by quantitative correlation analysis.
- **Loss landscape claims unsupported**: The paper asserts that higher Emergence indicates "greater concentration of local minima and a more rugged loss landscape" but provides no empirical validation. Section 3.3.1 contains only a hand-wavy visualization (Figure 2) and qualitative speculation.
- **Missing Lottery Ticket Hypothesis comparison**: The pruning discussion would benefit significantly from connecting to Frankle & Carbin (2019), which is the seminal work on pruning and its relationship to initialization and trainability. This omission weakens the paper's positioning in the pruning literature.
- **Incomplete VGG19/CIFAR-10 results**: The CIFAR-10 experiments are described qualitatively ("validates our hypothesis") without quantitative results, tables, or figures showing actual numbers, despite claiming these experiments "further validate" the framework.
- **Redundant content**: Key claims (e.g., "Emergence increases with scale" and "relative Emergence correlates with training performance") are repeated verbatim in Sections 1, 3.3, and 4, suggesting insufficient editing.

## Nice-to-Haves
- Ablation study varying the activation threshold (0.01, 0.05, 0.1, 0.2) to validate robustness
- Full quantitative results for CIFAR-10 experiments rather than qualitative descriptions
- Comparison against existing complexity/trainability measures (e.g., sharpness, mutual information, PAC-Bayes bounds)
- Experiments on larger-scale architectures (ResNet, transformers) to support general claims about "designing and optimizing architectures"
- Concrete design guidelines or prescriptions derived from the Emergence framework

## Novel Insights
The paper offers a potentially valuable perspective by defining Emergence as structural nonlinearity measurable through path-counting between active and inactive nodes. The key insight—that relative Emergence (normalized by parameters) explains the pruning-convergence speedup—is genuinely interesting and non-obvious: pruned networks have fewer parameters but achieve higher relative Emergence, suggesting more efficient use of remaining capacity for emergent trait development. The observation that Emergence converges to zero as training completes could have practical value as a training progress indicator. However, these insights are diminished by weak empirical support and the lack of clear causal mechanisms explaining *why* path-counting should predict emergent behavior.

## Potentially Missed Related Work
- **Lottery Ticket Hypothesis (Frankle & Carbin, 2019)**: Directly relevant to the pruning-initializer interplay claims; the paper would benefit from discussing whether their Emergence framework provides a different lens on why certain subnetworks ("winning tickets") train better.
- **Loss landscape visualization methods (Li et al., 2018)**: Referenced in the paper but not used for empirical validation of local minima claims.

## Suggestions
The paper should provide explicit statistical analysis: compute Pearson/Spearman correlation coefficients between Emergence values and training accuracy at each epoch, with p-values. Run experiments with multiple random seeds and report means ± standard deviations. This would transform the qualitative observations into defensible quantitative claims.

---

## CTC7CmirNr

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (7.5/10)
- Match: Yes

### Final Review

## Summary
This paper reveals that masked diffusion models (MDMs) are theoretically equivalent to masked models (both in training and sampling) and identifies a critical numerical precision issue in prior evaluations. The authors propose a First-Hitting Sampler achieving up to 20× speedup through analytical sampling, while demonstrating that previous claims of MDM superiority over ARMs were artifacts of truncated Gumbel sampling in 32-bit precision, not genuine advantages.

## Strengths
- **Rigorous theoretical contribution**: The mathematical derivations connecting continuous-time ELBO to discrete masked-token formulation (Proposition 3.1), proving optimal MDMs are time-independent (Proposition 3.2), and deriving analytical first-hitting times (Proposition 4.1) are well-structured and provide genuine insight into the relationship between MDMs and masked models.

- **Significant practical finding**: The identification of 32-bit floating-point truncation in Gumbel-based categorical sampling as a source of artificially low perplexity is an important reproducibility contribution. The closed-form analysis in Proposition 5.2 explaining the temperature-lowering effect and prioritized unmasking provides both theoretical understanding and empirical verification.

- **Effective sampling optimization with theoretical guarantee**: The First-Hitting Sampler provides genuine efficiency gains while being theoretically equivalent to the original reverse Markov process, with practical extensions to parallel decoding and high-order variants.

- **Clear organization and compelling narrative**: The paper effectively structures three major findings (training equivalence, sampling efficiency, numerical issue) with explicit contribution statements and appropriate citations to related work.

## Weaknesses
- **Context-dependent speedup claim**: The headline 20× speedup applies specifically to their text setting (L=1024, |V|=50,257, ~600M params). The paper itself acknowledges in Appendix J.3 that for DiffSound (|V|=256), the speedup drops to ~1.07×. This variability should be more prominently discussed rather than leading with the most favorable number.

- **Limited experimental scale**: Evaluations use only 64 samples, a single dataset (OpenWebText), and relatively small models (~170M parameters). The claim that "MDMs lack a clear prospect" to replace ARMs is based on experiments far below LLM scale, where diffusion models have shown strong scaling behavior in other domains.

- **Incomplete domain validation**: The paper focuses exclusively on text generation. MDMs and masked models are widely used for image generation (MaskGIT), where bidirectional attention provides clearer advantages and ARMs are not the dominant paradigm. Without validation in these domains, the general claim that "simpler masked models are sufficient" may not generalize.

- **Missing MaskGIT-style baseline**: The paper argues MDMs are equivalent to masked models but doesn't directly compare against actual masked model implementations (e.g., MaskGIT, BERT-style) using identical architectures. The only comparison is against ARMs, which inherently favors ARMs on sequential text data.

- **Theoretical gap in high-order variants**: The paper admits "higher-order methods tend to degrade performance" yet proposes both extrapolation and predictor-corrector variants without clear theoretical guidance on which to use under different conditions (N≤128 vs N≥256).

## Nice-to-Haves
- Experimental validation on image generation tasks where masked models excel, to support the general claim about masked model sufficiency
- Quantified experimental comparison of KV caching incompatibility with concrete wall-clock inference benchmarks, rather than only theoretical discussion
- Multi-seed experiments with confidence intervals for training comparisons, given the variance visible in training curves
- Systematic temperature study across a wider range (e.g., 0.5-1.5) with human evaluation or metrics like MAUVE to better characterize the quality-diversity trade-off

## Novel Insights
The paper's most significant insight is the reformulation of continuous-time MDM training through the lens of masked token counts, revealing that the time variable is essentially a continuous relaxation of the masked ratio (Proposition 3.1), and that the optimal model is time-independent (Proposition 3.2). The discovery of numerical precision issues in Gumbel sampling represents a genuine contribution to reproducibility—it demonstrates that the apparent superiority of MDMs over ARMs (Gen PPL ~15 vs ~40) was an artifact of 32-bit truncation artificially lowering effective temperature, rather than genuine capability. The first-hitting sampler provides an elegant solution that simultaneously addresses sampling inefficiency and avoids the numerical issue by changing the sampling paradigm from simultaneous multi-position categorical sampling to sequential token-by-token decoding.

## Potentially Missed Related Work
- **InSERtive Diffusion** (Grave et al.) and related insertion-based non-autoregressive methods — for comparison on efficiency claims
- **MaskGIT** (Chang et al., 2022) — could serve as a stronger baseline for the equivalence claim with identical architecture
- **Ou et al. (2024)** — concurrent work also noting time-agnostic properties, which the paper acknowledges in passing but could be discussed more thoroughly regarding priority and complementary findings

## Suggestions
The paper should clarify the conditions under which the 20× speedup applies by providing a theoretical formula for expected speedup as a function of vocabulary size, sequence length, and model size, as this would help practitioners assess applicability to their settings. Additionally, the comparison section should include MaskGIT-style baselines at identical architecture to directly validate the claimed equivalence rather than relying solely on comparison to ARMs.

---

## LbgIZpSUCe

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
This paper introduces MRDS-IR, a multi-region neural dynamics model combining nonlinear within-region dynamics (parameterized by deep neural networks) with linear communication channels between regions characterized by their impulse response. The key innovation is enabling interpretable characterization of inter-area communication while retaining expressive local dynamics. The authors develop a state-noise inversion-free variational filtering algorithm and demonstrate the approach on synthetic data, RNN-based computational tasks, and real V1/V2 neural recordings.

## Strengths
- **Strong theoretical foundation**: The paper elegantly connects linear systems theory (impulse response, transfer functions, realizations) to state-space models for neural data, providing principled interpretability for communication channels. This bridges a gap between circuit-level theories and statistical methods.
- **Novel model combination**: The integration of nonlinear local dynamics with linear impulse-response communication channels fills a gap in existing methods (see Table S1), offering a better trade-off between expressiveness and interpretability than prior approaches.
- **Technically elegant inference**: The state-noise inversion-free variational filtering algorithm handles hybrid stochastic/deterministic transitions in a principled way, and the block-structured updates enable computational efficiency.
- **Comprehensive empirical validation**: Ground truth experiments (Sections 3.1-3.3) validate recovery of dynamics, channel structure, and pruning of inactive channels. Real V1/V2 data experiments (Section 3.4) show competitive or superior performance against established baselines (MRM-GP, DLAG, LN, NL).
- **Demonstrated interpretability**: The approach successfully recovers frequency selectivity, delays, and gating structure in inter-area communication—features directly relevant to neuroscientific questions (e.g., feedforward vs. feedback timing in V1/V2).

## Weaknesses
- **Missing ablation for key innovation**: The paper never directly tests whether the impulse-response parameterization actually helps versus simpler instantaneous linear coupling. An ablation comparing MRDS-IR to an otherwise identical model with instantaneous connectivity (as in Glaser et al. 2020) would strengthen the paper by demonstrating the value of the core contribution.
- **Limited real-data validation scope**: The V1/V2 validation uses only one session with simple orientation coding. Testing on datasets with more cognitively demanding tasks (decision-making, working memory) or demonstrating reproducibility across multiple sessions would better support claims about understanding distributed computation.
- **Ad-hoc stimulus encoding**: The V1/V2 experiments use "amplitude proportional to stimulus ID" as input, acknowledged as "a first pass." The paper should discuss sensitivity to this choice or explore more principled stimulus conditioning, as this affects the interpretability of results.
- **Hyperparameter selection lacks guidance**: Choices of latent dimensions (L_k) and filter order (M) are critical but not systematically analyzed for sensitivity or selection methodology. The paper does not provide guidance on how practitioners should make these choices.
- **Identifiability not addressed**: The model has many degrees of freedom (channel filters, nonlinear dynamics, latent dimensions). The paper never discusses whether learned solutions are unique or how posterior uncertainty affects interpretation.
- **Training stability underexplored**: The combination of nonlinear dynamics + linear channel parameters + variational inference could suffer from optimization challenges. The paper does not analyze failure modes, convergence reliability, or show ELBO trajectories across random seeds.

## Nice-to-Haves
- Quantitative channel recovery metrics (e.g., correlation or MSE between recovered and ground truth impulse responses) in Section 3.1, rather than purely qualitative assessment.
- Single-trial V1/V2 trajectories showing oscillatory structure consistency across trials.
- Computational complexity analysis (training time, scalability) for ICLR audience.
- Systematic sensitivity analysis showing how results change across L_k and M choices.
- More sessions/animals for V1/V2 validation (the paper mentions session 107l003p143 in some figures but doesn't fully leverage multiple sessions).

## Novel Insights
The paper's most valuable insight is that interpretable communication channels parameterized by their impulse response can be seamlessly integrated with expressive nonlinear within-region dynamics while maintaining tractable inference. The demonstration that learned channel frequency responses reveal passband structure aligned with task-relevant frequencies (Figure 3G) provides a concrete example of how the linear systems framework can yield neuroscientifically interpretable results. Similarly, the V1/V2 finding that feedforward (V1→V2) signals are prominent early while feedback (V2→V1) ramps up later aligns with known neuroscience principles, suggesting the model captures genuine physiological phenomena rather than artifacts.

## Potentially Missed Related Work
- **Nonlinear state-space models with structured inference**: While the paper discusses linear LDS variants and switching nonlinear models (mr-srLDS), it could benefit from deeper engagement with the broader SSM literature on nonlinear dynamics estimation and amortized inference (e.g., Klushyn et al. 2021 on deep state-space models).
- **Alternative stable parameterizations**: The paper mentions Orvieto et al. 2023 as an alternative for stability enforcement but does not compare approaches or discuss trade-offs.

## Suggestions
- Add an ablation comparing MRDS-IR to an otherwise identical model with instantaneous linear coupling to isolate the contribution of the impulse-response channels.
- Include a sensitivity analysis for hyperparameter choices (L_k, M) and provide guidance on model selection using ELBO or held-out predictive likelihood.
- Demonstrate reproducibility on V1/V2 across multiple sessions (at minimum, sessions 106r001p26 and 107l003p143) to strengthen the real-data validation claims.
- Address training stability concerns by showing ELBO convergence curves across random seeds and discussing failure modes.

---

## nwDRD4AMoN

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces Artificial Kuramoto Oscillatory Neurons (AKOrN), a novel neural network architecture where neurons are represented as N-dimensional unit vectors on a hypersphere that evolve via generalized Kuramoto dynamics. The method demonstrates competitive performance on unsupervised object discovery (outperforming SSL baselines on PascalVOC), strong reasoning capability on Sudoku puzzles (89.5% OOD with energy-based voting), and notable adversarial robustness with well-calibrated uncertainty estimates on CIFAR10. The work is notable as the first synchrony-based model to scale competitively to natural images.

## Strengths
- **Innovative architectural design bridging dynamical systems and deep learning**: Successfully incorporates the Kuramoto model (from statistical physics) as a fundamental neural processing unit, providing a principled alternative to conventional residual updates. The finding that asymmetric (non-reciprocal) connections outperform symmetric alternatives is both biologically plausible and empirically important.

- **Strong empirical contributions across diverse domains**: Demonstrates AKOrN's effectiveness on unsupervised object discovery (competitive with slot-based models on CLEVRTex, FG-ARI of 88.5%; outperforms DINO/MoCoV3/MAE on PascalVOC with MBO_i of 52.0), Sudoku solving (100% ID, 89.5% OOD), and adversarial robustness (58.91% against AutoAttack on CIFAR10 with ECE of 1.3). The ablation studies comprehensively validate each component's contribution.

- **Comprehensive ablation and architectural analysis**: Thorough ablation studies examining the projection operator, Kuramoto vs. residual updates, number of rotating dimensions (N), and the roles of C and m terms. The analysis showing that N=2 underfits object discovery tasks while N=4 underperforms on robustness provides important guidance.

- **Theoretical grounding with practical insights**: Includes Lyapunov stability proofs and connections to physics models (Heisenberg model, active matter). The appendix provides rigorous mathematical justification while acknowledging limitations of the asymmetric case.

- **Reproducibility**: Code and project page provided; experimental settings are detailed in appendices.

## Weaknesses

- **Theoretical foundation undermined by empirical approach**: The paper proves in Appendix F that Eq. (3) is a Lyapunov function only under symmetric constraints (J_ij = J_ji, Ω_i = Ω, Ωc_i = 0), yet explicitly uses asymmetric connections because they "perform better across all tasks." The paper states "it is unclear whether the energy defined in Eq. (3) is proper" for attentive connectivity, yet relies on this energy for the energy-based voting mechanism. This disconnect between theory and practice should be more prominently acknowledged rather than buried in the appendix.

- **N=2/N=4 trade-off creates confusion and undermines robustness claims**: Table 20 reveals that AKOrN_mix with N=4 achieves 93.51% clean accuracy but 0% adversarial accuracy, while N=2 achieves 58.91% adversarial accuracy but only 91.23% clean. The main text prominently showcases N=2 robustness results without adequately explaining that the robustness property disappears when N is increased to match object discovery performance. The paper does not provide guidance on when to use which variant, nor does it discuss why this trade-off exists.

- **Energy-based voting scalability concern**: The 89.5% OOD Sudoku result requires T_eval=512 and K=1000 samples. This represents extensive test-time compute (essentially ensembling over 1000 initializations) that is not compared fairly against baselines using equivalent test-time compute. The paper should contextualize this as a computationally expensive approach.

- **Adversarial robustness evaluation limited**: Only CIFAR10 is tested for robustness; no ImageNet-scale results are provided, limiting generalizability claims. Additionally, the AutoAttack evaluation uses ε=8/255, which is relatively mild compared to stronger attacks tested in the robustness literature.

- **Up-tiling not systematically evaluated on baselines**: While Figure 21 shows up-tiling visually helps DINO features, full metrics with up-tiling applied to baseline models are not reported in the main tables, making it unclear whether the up-tiling contribution is unique to AKOrN.

- **OOD results on COCO underemphasized**: While AKOrN outperforms SSL baselines on PascalVOC, it is outperformed by SPOT on COCO2017 (35.0 vs 31.3 MBO_i). The paper should acknowledge this limitation more explicitly rather than emphasizing the PascalVOC results.

## Nice-to-Haves
- **Analysis of energy-predictability relationship across tasks**: The paper shows energy correlates with Sudoku correctness but provides no analysis of whether energy tracks anything meaningful for object discovery or robustness. Investigating this would strengthen the "energy-based model" framing.

- **Deeper analysis of N>4 performance degradation**: Figure 16-17 shows sharp performance drops when N exceeds 4-16 depending on task. The paper acknowledges "further experimental and mathematical analysis is needed" but this fundamental limitation directly impacts practical applicability.

- **Ablation isolating sphere constraint**: The ablation comparing Kuramoto vs. residual updates (Table 6) also removes the sphere constraint entirely, confounding what drives the benefit. An ablation keeping unit-norm but using standard attention would isolate the Kuramoto dynamics' contribution.

- **Failure case analysis on Sudoku OOD**: The 10.5% failure rate on OOD Sudoku deserves analysis—what types of puzzles does AKOrN systematically fail on?

## Novel Insights
This paper makes several genuinely novel observations beyond the architectural contribution. First, it demonstrates that asymmetric (non-reciprocal) connections—biologically more plausible than symmetric ones—actually improve performance over symmetric variants, overturning assumptions from the energy-based model literature (Energy Transformer). Second, it shows that the energy function arising from Kuramoto dynamics naturally serves as a solution quality indicator, enabling energy-based voting that substantially improves OOD generalization without task-specific heuristics. Third, it reveals that test-time extension of the iterative computation (more Kuramoto steps) improves accuracy on harder problems while hurting easier ones—an "adaptive computation time" property emerging from the dynamics themselves rather than being designed in. These observations suggest that the Kuramoto dynamics provide more than just a different architectural primitive; they induce qualitatively different learning and inference behaviors.

## Potentially Missed Related Work
- **Complex-valued neural networks with binding mechanisms** (Reichert & Serre, 2013; Lösche et al., 2022; Stanić et al., 2023): The paper discusses these but could more thoroughly compare against complex-valued approaches that also leverage phase for binding, particularly given the connection between the N-dimensional unit vectors and complex-valued representations.

- **Neural Oscillators for Computation** (van Gerven & Jensen, 2024): Recent work on oscillatory neural networks for computation that appeared around the same time and addresses similar questions about dynamical representations.

- **Traveling wave computations in biological networks** (Keller et al., 2024): The paper cites neuroscience evidence for traveling waves but could connect more explicitly to computational models of such waves.

## Suggestions
1. **Reconcile N=2 and N=4 variants**: Provide clear guidance on when to use each variant, or investigate hybrid approaches. The current presentation leaves readers uncertain about which variant to trust.

2. **Contextualize adversarial robustness claims**: Discuss explicitly that robustness comes at a clean accuracy cost and that the N=2 variant sacrifices task performance for robustness. Consider whether the contribution is "robustness by design" or "robustness with accuracy trade-offs."

3. **Compare fairly on test-time compute**: For Sudoku, compare against baselines (RRN, R-Transformer) using equivalent test-time ensembling to determine whether the improvement is from the method or from the ensemble.

4. **Strengthen energy-based reasoning claims**: Investigate whether the energy-predictability relationship extends beyond Sudoku to other constraint satisfaction or reasoning tasks.

---

## fBSc0c1IXJ

- GT: Reject (avg 3.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces Remote Reinforcement Learning (RRL) with communication constraints, where a controller with reward access guides an actor (without reward access) through a rate-limited channel. The proposed GRASP method combines channel simulation (importance sampling) to efficiently communicate action samples, with behavioral cloning to establish a common reference distribution that minimizes communication cost. The empirical evaluation spans diverse environments and RL algorithms, demonstrating 12-13x average communication savings while maintaining equivalent policy performance.

## Strengths
- **Well-motivated practical problem**: The RRL setting models realistic scenarios like wireless control and edge computing where reward evaluation is expensive or centralized but action execution must happen remotely. The four limitations of direct reward transmission are clearly articulated.
- **Solid information-theoretic grounding**: The method builds on established channel simulation literature (Cuff 2008, Li & El Gamal 2018), with proper discussion of bounds (Equations 1-2) showing why D_KL[P||Q] bits suffice versus H(P) + D_KL[P||Q] bits for naive transmission.
- **Comprehensive empirical evaluation**: Experiments span diverse environments (CartPole, LunarLander, Breakout, BipedalWalker, HalfCheetah, multi-agent settings), multiple RL algorithms (PPO, DQN, SQ, DDPG), and 8-20 random seeds for statistical reliability.
- **Algorithmic clarity and reproducibility**: Algorithms 1-3 provide clear pseudocode for both controller and actor components, with specific channel simulation implementation details.
- **Meaningful communication savings**: 12-258x reduction in communication bits depending on environment, with substantial savings in continuous action spaces where naive approaches require 32-bit floats per dimension.

## Weaknesses
- **Reward transmission baseline not empirically evaluated**: The paper theoretically argues that GRASP achieves "41x less communication than sending the reward" (assuming 32 bits/step), but never runs these experiments. This is a central comparison point that would validate the practical claim. The paper states "sending the reward is functionally equivalent to action source coding" but this equivalence holds only for communication cost, not for the intelligence/computation tradeoff the paper claims to make.
- **Continuous action space claim is imprecise**: The abstract states "50-fold reduction for environments with continuous action spaces" but experimental results show wide variance: from 10x (Pendulum-DDPG) to 258x (HalfCheetah-PPOcont). The 50-fold figure appears to be a representative rounded number not matching any specific statistic in the paper. This should be clarified or the claim revised.
- **Minor numerical inconsistency**: Abstract claims "12-fold" average reduction while Tables 2 and 4 report "geometric average of 13 times reduction" (or "×258.20" in one case). While minor, this should be resolved for precision.
- **Limited failure mode analysis**: The paper does not discuss conditions where behavioral cloning degrades (e.g., Breakout shows 15% return gap, the largest in experiments). The KL-divergence between policies could grow unbounded in some scenarios, and the paper lacks analysis of how communication efficiency degrades when the actor's learned policy diverges from the controller's.

## Nice-to-Haves
- **Ablation studies**: Systematically analyze the contribution of channel simulation versus behavioral cloning to communication savings. Currently, GRASP combines both, making it difficult to isolate each component's impact.
- **Extension to noisy/delayed channels**: The motivating application mentions "delay constraints" for wireless channels, but all experiments assume perfect synchronous communication. Testing robustness to packet loss, delays, and bit errors would strengthen practical applicability.
- **Theoretical analysis**: While not required, regret bounds or sample complexity results for GRASP in the RRL setting would strengthen the contribution beyond empirical validation.
- **KL-divergence evolution plot**: Show how KL divergence between controller and actor policies evolves during training and correlates with communication rate.

## Novel Insights
The paper makes a valuable contribution by formalizing the RRL problem, which sits at the intersection of reinforcement learning and communication constraints—an underexplored setting where the agent taking actions lacks reward access. The insight that channel simulation (specifically, importance sampling over candidate samples) can reduce communication from H(P) + D_KL[P||Q] to approximately D_KL[P||Q] bits is both theoretically grounded and practically significant. The clever combination of channel simulation with behavioral cloning allows the actor to simultaneously learn a useful policy while serving as a shared reference distribution, creating a virtuous cycle where better actor policies enable more efficient communication. This framework is particularly well-suited for multi-agent and parallel settings, where a centralized controller coordinates distributed actors.

## Potentially Missed Related Work
- **Pujol Roig & Gündüz (2020)** "Remote Reinforcement Learning over a Noisy Channel" — Directly addresses RL with communication constraints over noisy channels; the paper mentions noisy channels in related work but does not compare with or build upon this specific approach.
- **Pase et al. (2022)** "Rate-constrained remote contextual bandits" — The paper cites this work but a more detailed comparison of GRASP's approach versus their rate-constrained bandit formulation would strengthen positioning.
- **Active RL literature** (Krueger et al., 2020; Eberhard et al., 2024) — The paper mentions active learning briefly but doesn't fully explore connections to cost-sensitive reward acquisition, where rewards are costly or delayed.

## Suggestions
- **Resolve numerical claim precision**: Clarify whether the "50-fold reduction for continuous action spaces" is a maximum, minimum, or average, and reconcile the 12-fold vs 13-fold discrepancy between abstract and body.
- **Add empirical reward-transmission comparison**: Run at least one experiment with actual 32-bit reward transmission to validate the theoretical 41x savings claim.
- **Characterize failure modes**: Analyze why Breakout shows the largest return gap (15%) and discuss conditions under which behavioral cloning fails to capture the controller's policy adequately.

---

## rPup1cWk4d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.5/10)
- Match: Yes

### Final Review

## Summary
This paper proposes a data augmentation framework that embeds structured data (e.g., tensors) into a statistical manifold via log-linear models on posets, then uses a novel "backward projection" algorithm to reverse dimension reduction and generate augmented samples. The method aims to provide interpretable, white-box data augmentation as an alternative to black-box generative models like autoencoders, leveraging information geometry concepts including dually-flat structures and Bregman divergences.

## Strengths
- **Novel backward projection algorithm**: Algorithm 4.1 presents a genuinely original approach to the inverse problem of dimension reduction using nearest neighbors and local sub-manifold construction, with theoretical grounding in divergence minimization. This concept may have applications beyond data augmentation.

- **Strong theoretical foundation**: The paper correctly leverages established information geometry concepts—dually-flat manifolds, Bregman divergences, m-projections—applied in a novel "meta" perspective by treating multiple data points as probability distributions on a statistical manifold.

- **Principled interpretability through many-body approximation**: The construction of sub-manifolds using ℓ-body approximations provides a semantically meaningful parameterization (capturing mode interactions at different orders), offering more principled control over dimensionality than black-box latent spaces where dimensions lack clear meaning.

- **Clean algorithmic formulation**: Algorithm 4.2 presents a complete, well-structured three-phase pipeline (encoding, generating, decoding) with clear mathematical notation and theoretical guarantees.

## Weaknesses
- **Uncompetitive and poorly-documented autoencoder baseline**: The paper admits the autoencoder "appears to overfit" but provides zero details on architecture, training procedure, or hyperparameters beyond "2+2 layers with latent dimension 17." A simple architecture trained on 1000 samples without regularization is not a representative baseline for autoencoder-based augmentation. This undermines the paper's central comparative claim.

- **"Ours" alone underperforms original data**: Table 1 shows "Ours" (75.37%) substantially underperforms "Original" (81.79%) as a training set. Only "Original + Ours" (83.40%) marginally beats "Original" (81.79%), and this improvement is within error bars (±3.22% vs ±4.57%). The "competitive performance" claim is weakened by this fact.

- **No statistical significance testing**: All comparisons between methods use error bars ranging from ±3-15%, yet no statistical tests (e.g., paired t-tests, McNemar's) establish whether differences are meaningful. This is particularly problematic for UCI experiments where error bars are enormous relative to accuracy differences.

- **Algorithm 4.1 lacks full specification**: The "Sub-Manifold" and "Projection" operations are black boxes. While Remark 4.1 provides one concrete example (fixing θ-coordinates), this is insufficient for reproducibility. The paper states "D is created by some linear constraints" without specifying what constraints or how they're chosen.

- **Missing standard data augmentation baselines**: For image data augmentation at ICLR, comparison with classical methods (rotation, flipping, Mixup, CutMix) is standard. These baselines are entirely absent, making it impossible to assess whether the proposed approach adds value over simpler alternatives.

- **Limited quantitative metrics for generated samples**: The paper relies solely on downstream classification accuracy and visual inspection. Standard generative model metrics (FID, Inception Score, precision/recall curves) are absent, leaving the quality of augmented samples inadequately assessed.

## Nice-to-Haves
- **Scale experiments to CIFAR-10 or ImageNet**: MNIST (1000 samples) is insufficient to demonstrate practical value. The method's scalability to realistic image datasets is unproven.

- **Ablation studies**: Systematically ablate key components (poset structure choices, alternative sub-manifold construction methods, KDE vs. other generation methods) to isolate each contribution's impact.

- **Concrete interpretability examples**: Demonstrate operational interpretability—what does a dimension in B represent? How can a practitioner control augmentation behavior? Currently, interpretability claims are asserted but not demonstrated.

- **Runtime and memory analysis**: Report computational complexity and compare efficiency against baselines, as this matters for practical deployment.

## Novel Insights
The paper's most valuable contribution is the "backward projection" concept: given a point in the low-dimensional latent space, identify its k-nearest neighbors among projected original data, construct a local sub-manifold from their pre-images, and project backward onto it. This provides a geometrically intuitive, data-centric solution to the inverse problem of dimension reduction that avoids black-box decoders. The insight that the interplay between linearity (flat sub-manifolds with linear constraints) and non-linearity (curved statistical manifold) yields "pseudo-non-linear" transformations is conceptually interesting. The meta-perspective of treating datasets as probability distributions on a statistical manifold also offers a fresh viewpoint for information geometry applications.

## Potentially Missed Related Work
- **Out-of-sample extension methods**: Classical techniques like Nyström approximation (Williams & Seeger, 2000) and Geometric Harmonics (Coifman & Lafon, 2006) address similar inverse problems for kernel methods. The paper's relation to these established approaches is not discussed.

- **Mixup and geometric data augmentation**: Related interpretable augmentation strategies (Zhang et al., 2018; Verma et al., 2022) that operate in latent/projection spaces are absent from the comparison.

- **Energy-based models for augmentation**: While the paper claims an energy-based approach, connections to established EBM literature for data augmentation (e.g., Du & Mordatch, 2019) are not explored.

## Suggestions
**Strengthen the experimental methodology**: (1) Use a properly regularized autoencoder (VAE with KL divergence, or denoising AE with dropout) with documented hyperparameters and training procedures. (2) Include classical augmentation baselines (rotation, flipping, Mixup) as controls. (3) Add FID/Inception Score metrics for quantitative generative quality assessment. (4) Conduct statistical significance tests for all comparative claims. These changes would substantially strengthen the paper's central narrative about interpretability advantages without sacrificing theoretical rigor.

---

## 5IkDAfabuo

- GT: Accept (Oral) (avg 7.5)
- Predicted: Accept (6.5/10)
- Match: Yes

### Final Review

## Summary
The paper proposes Prioritized Generative Replay (PGR), a framework that models an RL agent's replay buffer as a conditional diffusion model conditioned on "relevance functions" that guide synthetic data generation toward more learning-informative transitions. The key insight is that successful generative replay depends on generating the right *kinds* of transitions (not just high-quality ones), and the paper identifies curiosity (ICM-based prediction error) as an effective default relevance function that promotes diversity and reduces overfitting.

## Strengths
- **Strong empirical breadth**: Experiments span state-based DMC, pixel-based DMC, OpenAI Gym, and DMLab with diverse baselines including PER, exploration bonuses, model-based methods, and the close prior work SYNTHER. The evaluation protocol following Lu et al. (2024) enables direct comparison.
- **Excellent mechanism analysis**: The paper convincingly demonstrates *why* PGR works through dormant ratio analysis (showing reduced overfitting), curiosity score distributions (showing shifted novelty toward higher-relevance regions), and controlled experiments showing generation quality is similar between PGR and SYNTHER, isolating that the *target* of conditioning (not quality) drives gains.
- **Conceptually novel framing**: Connecting prioritized experience replay to conditional generation provides a fresh unifying perspective, and identifying curiosity as a principled default relevance function is a valuable contribution.
- **Minimal overhead with clear benefits**: <5% additional training time with substantial and consistent gains across settings.
- **Informative ablations**: Systematic comparisons to PER, exploration bonuses, different relevance functions, and scaling experiments demonstrate the framework's flexibility and the importance of conditioning.

## Weaknesses
- **Underspecified conditional generation methodology**: The "prompting strategy" from Peebles et al. (2022) referenced in Section 4.3 is not explained in sufficient detail for full reproducibility. How exactly are top-k F-values used as conditioning signals during diffusion sampling? The percentile-k sampling is described but the mechanism connecting F-values to CFG is opaque.
- **tSNE analysis lacks quantification**: Section 4.1 claims PGR generates a "distinct sub-portion" of the data space with "red and blue dots largely separate" in panel B, but this is asserted without quantitative support (e.g., average minimum distance, KL divergence between distributions). This weakens the mechanistic argument.
- **Reward-PGR underperformance underanalyzed**: The observation that conditioning on reward actually hurts performance compared to unconditional SYNTHER is a significant empirical finding that deserves deeper analysis. The paper attributes this to lack of "how to navigate to" information, but this post-hoc reasoning could be strengthened with additional analysis of what specifically fails in Reward-PGR.
- **Limited environment diversity**: All main experiments are on continuous control locomotion tasks. Generalization to more complex domains (e.g., Atari, robotic manipulation, multi-task settings) is undemonstrated in the main paper.
- **Guidance scale ω not systematically studied**: The CFG guidance scale is a critical hyperparameter but is neither ablated nor discussed in sensitivity across experiments.

## Nice-to-Haves
- Systematic guidance scale sensitivity analysis to provide practical recommendations
- Comparison to additional overfitting mitigation methods (e.g., Munchausen DQN, data augmentation techniques) to justify the added complexity of the generative model
- Extended analysis of what makes a good relevance function F beyond the empirical comparison
- Additional validation that dormant ratio correlates with overfitting in actor-critic (SAC/REDQ) settings, given it was introduced for value-based methods

## Novel Insights
The paper's central insight—that synthetic data quality (measured by dynamics MSE) is nearly identical between conditional and unconditional generation, yet conditional generation substantially outperforms—represents a genuinely novel contribution to understanding generative RL. The insight that the *target* of conditioning (relevance function choice) matters more than generation quality itself is both counterintuitive (reward-based conditioning fails) and well-supported by analysis of curiosity-PGR's effect on diversity and overfitting. The framing elegantly unifies experience replay prioritization and conditional generation, providing a new design space for generative RL research.

## Potentially Missed Related Work
- **DreamerV3 world models with planning**: While DREAMER-V3 is cited, a more detailed comparison discussing the distinction between PGR's weaker coupling to dynamics (generating transitions independently) versus world model approaches that plan through learned dynamics could strengthen positioning.
- **Guided offline RL with diffusion**: Works like "Diffusion Model for RL: A Survey" (Zhu et al., 2023) and related guided-synthesis methods may provide additional context for the conditional generation framing.

## Suggestions
- Add a supplementary experiment isolating the conditional generation benefit with a "conditional uniform random" baseline alongside the existing conditional VAE comparison to definitively show that *any* relevant conditioning helps, not just curiosity specifically.
- Quantify the tSNE separation in Figure 2 (e.g., using earth mover's distance or average minimum cross-domain distance) to provide empirical backing for the mechanistic claims.

---

## y2ch7iQSJu

- GT: Reject (avg 2.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces **BBSurv**, the first method for budgeted active learning with censored survival data where queries provide only partial (incremental) information through an I-oracle. The authors adapt BatchBALD's mutual information framework to handle censored data, formulate batch selection as a maximum coverage problem, and provide a greedy algorithm. Empirical evaluation on three medical datasets (SUPPORT, MIMIC-IV, NACD) shows modest improvements over baselines in the MAE-PO metric.

## Strengths
- **Novel and important problem formulation**: This is the first work addressing budgeted learning with survival data where queries yield partial (not full) label information, directly modeling realistic clinical trial scenarios where follow-up may not fully uncensor patients.
- **Sound theoretical foundation**: The authors rigorously connect batch selection to the well-studied maximum coverage and budgeted maximum coverage problems (Khuller et al., 1999), establishing NP-hardness and leveraging established approximation algorithms.
- **Comprehensive experimental evaluation**: Experiments span three real-world medical datasets with varying sizes (2,402 to 38,520 patients), censorship rates (32% to 67%), and feature dimensions (42 to 93 features), testing both uniform and non-uniform cost settings.
- **Practical I-oracle formulation**: The model of incremental information gain through an I-oracle (e.g., extending follow-up by I years) directly corresponds to real clinical workflows and provides a generalizable framework that includes traditional AL as the special case I=∞.

## Weaknesses
- **Theoretical guarantee attribution is misleading**: The paper claims a (1−1/e) approximation guarantee, but this guarantee applies to Khuller et al.'s Algorithm 2 (the full budgeted maximum coverage algorithm), not the simplified Algorithm 1 that is actually implemented. The authors acknowledge the simplifications "are not guaranteed or proven," but this disconnect between theory and implementation should be more prominently addressed with either a proof or explicit disclaimer that Algorithm 1 is a heuristic.
- **Complexity claim is imprecise**: The paper states the approach "provides bounds and time complexity asymptotically equivalent to BatchBALD," but Algorithm 1 runs O(budget × n × k) iterations (where n is pool size and k is samples), which differs from BatchBALD's complexity for batch candidate evaluation.
- **Modest and questionably significant improvements**: The MAE-PO improvements over baselines are typically 0.01–0.10 on values around 2–4.5 (approximately 1–2% improvement). The paper claims "significantly better" without performing formal statistical tests; many reported differences fall within overlapping error bounds.
- **Missing ablation studies**: The paper doesn't isolate contributions of key components: the censored probability adjustment (pcens), the I-oracle handling (p_final), and the coverage-based greedy selection. Without ablation, it's unclear which innovation drives the gains.
- **Limited model evaluation**: All experiments use Bayesian Linear MTLR exclusively. The generality of BBSurv to other Bayesian survival models is not demonstrated, limiting understanding of model-dependent performance.

## Nice-to-Haves
- Statistical significance testing (paired t-tests or Wilcoxon signed-rank tests) across multiple random seeds for all comparisons
- Ablation isolating pcens normalization to show it's not the sole driver of improvement
- Direct comparison to Hüttel et al. (2024), which extends BALD to censored data but uses a different (non-batch) approach
- Investigation of why the simple "Mean Closest to Middle" (MCtH) heuristic occasionally ties with BBsurv—this unexplained phenomenon warrants analysis as it may undermine the justification for the method's complexity
- Analysis of failure modes (e.g., when the oracle increment I is very small, or when censorship rates are extremely high)

## Novel Insights
This paper's core insight is recognizing that survival analysis with censoring is a natural setting for budgeted learning because the "label" (time-to-event) may only be partially observable even after querying—an oracle can provide additional follow-up time but not necessarily complete uncensoring. The I-oracle formulation elegantly captures this: rather than the traditional AL assumption that querying reveals the true label, here querying extends observation by I units, which may or may not capture the event. This insight enables principled application of mutual information-based acquisition to survival data by appropriately adjusting probability distributions to only consider time bins within the oracle's increment and aggregating all bins beyond as a single "unknown after increment" class.

## Potentially Missed Related Work
- **Hüttel et al. (2024)**: Extends the BALD framework to right-censored data. While cited, a direct experimental comparison would clarify whether the batch-based approach or the survival-specific acquisition function provides additional value over single-instance BALD.

## Suggestions
- **Clarify theoretical claims**: Either prove that Algorithm 1 retains the (1−1/e) bound under stated assumptions, or explicitly reframe Algorithm 1 as a heuristic approximation with empirical validation and no formal guarantee beyond BatchBALD's submodular property.
- **Add an ablation table** systematically comparing: (1) baseline AL + pcens normalization, (2) baseline AL + p_final, (3) BBsurv (full method). This isolates which component is responsible for observed gains.

---

## LJULZNlW5d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces FGL (Fast Gradient Leakage), a gradient inversion attack for federated learning that combines StyleGAN priors with a joint gradient matching loss (combining L₁, L₂, and cosine distance) to achieve faster and higher-quality image reconstruction from shared gradients. The authors claim their method enables attack on high-resolution (224×224) face images with batch sizes up to 60, representing an improvement in attack efficiency and scalability over prior gradient inversion work.

## Strengths
- **Modular ablation design**: Table 1 presents a clear, progressive ablation study demonstrating incremental contributions from each proposed component (S_init, S_final, T_trans, M_grad, N_grad, M_seed), with quantitative evidence showing meaningful improvements at each step.
- **Practical efficiency gains**: The reported time costs (2.58 minutes for batch=1 vs. 23-24 minutes for baseline methods) represent a substantial practical improvement in attack feasibility.
- **Scalability demonstration**: Achieving gradient inversion attacks at batch size 60 on 224×224 CelebA images, compared to prior work's limitations with high-resolution data, addresses a meaningful capability gap.
- **Multi-faceted optimization approach**: The combination of joint gradient matching loss, selection strategy, gradient normalization, and multi-seed optimization provides multiple complementary signals for improving attack quality.

## Weaknesses
- **Factual error regarding CelebA dataset**: The paper states CelebA has "1000 classes." This is incorrect—CelebA contains 202,599 face images with 40 binary attribute labels per image, not 1000 classes. This fundamental misunderstanding of the dataset undermines credibility.
- **Inflated "first time" claim**: The paper claims the first optimization-based high-resolution CNN reconstruction, but Geiping et al. (2020) demonstrated gradient inversion on 128×128 ImageNet images with ResNet-152, and Yin et al. (2021) showed 224×224 reconstruction on ImageNet-scale data. The contribution is better framed as combining StyleGAN priors with GIAs for improved performance, not as enabling capability that was previously impossible.
- **Unjustified evaluation metrics**: The paper dismisses standard metrics (SSIM, PSNR, MSE) claiming FGL aims for "images with akin features" rather than reconstruction. Yet Table 2 shows perfect Top-1=1.0 accuracy, which directly contradicts this framing—if reconstructions achieve 100% classifier accuracy, they are effectively indistinguishable from originals. The metric interpretation requires clarification or alternative metrics should be provided.
- **Unfair baseline comparisons**: DLG is evaluated on CIFAR-10 (batch size 4) while all other baselines use CelebA (batch size 1). This methodological inconsistency undermines the comparative claims. All methods should be evaluated under identical conditions.
- **Missing reproducibility details**: Critical hyperparameters α₁, α₂, α₃ for the joint loss function are not specified in the main text. The number of seeds in multi-seed optimization is also unspecified. Without these values, the method cannot be independently reproduced.

## Nice-to-Haves
- **Defense evaluation**: The paper claims FGL can "advance the development of privacy defense techniques" but presents no experiments against gradient obfuscation defenses (compression, noise injection, differential privacy). Including such experiments would directly support this claimed contribution.
- **Ablation isolating StyleGAN's contribution**: It remains unclear how much of the improvement comes from the StyleGAN prior (which dramatically constrains the optimization space) versus the proposed joint loss function. An ablation without StyleGAN would clarify this.
- **Statistical rigor**: Reporting confidence intervals and standard deviations across multiple random seeds would strengthen the experimental claims, which currently report only single best-case results.
- **Generalization beyond faces**: Experiments on non-face datasets (CIFAR-10/100, ImageNet subsets) would validate claims about attacking "complex datasets."

## Novel Insights
The paper's primary insight—adapting techniques from model inversion attacks (MIAs), specifically the use of pretrained GANs as image priors, into gradient inversion attacks—is sensible and timely. The joint gradient matching loss combining L₁, L₂, and cosine distance represents a reasonable engineering contribution, though the theoretical motivation for why multiple correlated losses avoid local optima better than a single loss remains underexplained. The gradient normalization technique and multi-seed selection strategy are practical refinements that appear to yield meaningful improvements in the ablation studies.

## Potentially Missed Related Work
- **PPA (Plug & Play Attacks)** by Struppek et al. — referenced in related work as inspiration but not directly compared in experiments. PPA achieves large-batch, high-accuracy attacks on face datasets using a similar GAN-based approach.
- **Inverting Gradients on Vision Transformers (GradViT)** by Hatamizadeh et al. — demonstrates GIAs on ViT architectures; experiments could be strengthened by including transformer-based target models.
- **Auditing Privacy Defenses** by Huang et al. (2021) — identifies BatchNorm statistics as a strong assumption in GIAs. The paper does not address whether FGL relaxes this assumption, which would be important for practical threat assessment.

## Suggestions
1. **Correct the CelebA dataset characterization** and clarify whether experiments use class labels or identity labels.
2. **Rephrase "first time" claims** to accurately position relative to prior work (Geiping et al., Yin et al.)—the contribution should be framed as combining StyleGAN priors with improved loss functions, not as enabling previously impossible attacks.
3. **Provide standard reconstruction metrics (SSIM, PSNR/MSE)** alongside proposed metrics, or explicitly redefine metrics to measure "face recognition accuracy of reconstructions" rather than "reconstruction quality."
4. **Evaluate all baselines under identical conditions** (same dataset, same batch sizes) for fair comparison.
5. **Specify all hyperparameters** (α values, number of seeds, learning rates) to enable reproducibility.
6. **Add defense experiments** against gradient compression, top-k sparsification, or differential privacy to support the paper's stated goal of advancing defense research.

---

