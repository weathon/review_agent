# Inferring brain plasticity rule under long-term stimulation with structured recurrent dynamics

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Understanding how long-term stimulation reshapes neural circuits requires uncovering the rules of brain plasticity. While short-term synaptic modifications have been extensively characterized, the principles that drive circuit-level reorganization across hours to weeks remain unknown. Here, we formalize these principles as a latent dynamical law that governs how recurrent connectivity evolves under repeated interventions. To capture this law, we introduce the Stimulus-Evoked Evolution Recurrent dynamics (STEER) framework, a dual-timescale model that disentangles fast neural activity from slow plastic changes. STEER represents plasticity as low-dimensional latent coefficients evolving under a learnable recurrence, enabling testable inference of plasticity rules rather than absorbing them into black-box parameters. 
We validate STEER with four benchmarks: synthetic Lorenz systems with controlled parameter shifts, BCM-based networks with biologically grounded plasticity, a task learning setting with adaptively optimized external stimulation and longitudinal recordings from Parkinsonian rats receiving closed-loop DBS. Our results demonstrate that STEER recovers interpretable update equations, predicts network adaptation under unseen stimulation schedules, and supports the design of improved intervention protocols. By elevating long-term plasticity from a hidden confound to an identifiable dynamical object, STEER provides a data-driven foundation for both mechanistic insight and principled optimization of brain stimulation. The source code of this study is available at https://github.com/ncclab-sustech/STEER.git.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The original idea and intuition behind this paper are excellent. 

STEER represents recurrent connectivity as a low-rank CP tensor decomposition and learns a stimulus-conditioned dynamical law $z_{k+1}=g_{\theta}(z_k,\bar{u}_k)$  governing how low-dimensional motif coefficients evolve across sessions. Identifiability is promoted through unit-norm and orthogonality constraints and an optional sign mask enforcing Dale’s principle. The approach is validated on increasingly realistic tasks: synthetic Lorenz systems, a Bienenstock–Cooper–Munro (BCM) plasticity model, and a longitudinal Parkinson’s disease DBS dataset.

Several essential details are relegated to the appendix and the presentation undermine the overall quality of the work. Some of those choices make me dubious on the validity of the model.

I am giving the paper an overall 4 "marginally below the acceptance threshold" but will consider increasing that grade if the weaknesses are adresses.

### Strengths
This work is well motivated and conceptually strong and addresses a fundamental challenge in neuroscience of understanding how long-term stimulation reshapes neural circuits. The proposed framework introduces a dual-timescale formulation that separates fast within-session dynamics from slow plasticity adaptation.

The paper is innovative. By employing a low-rank decomposition of the recurrent connectivity tensor, STEER enforces structure and identifiability, leading to interpretable motif-level representations of plasticity. 

The experimental design is particularly strong. The authors carefully structure their validation in increasing order of biological realism: starting from the synthetic Lorenz system, progressing to a controlled Bienenstock–Cooper–Munro (BCM) plasticity model, and culminating with real longitudinal deep-brain stimulation (DBS) data in Parkinsonian rats.

### Weaknesses
The presentation needs a brush

Clarity:
-"session" is never really well defined in the beginning and is such a vague term. Does itrefers to a single day, trial block, animal or experiment ? 
- section 4.1. Lorentz equation should be in main text, not hidden in the appendix
- l299-204.  the are no motivation behind equation 10
- Figure 4 is too small 

Motivation 
-  In the Lorentz experiment, it's not clear why this specific synthetic plasticity is chosen; they appear arbitrary.  A brief justification would be okay and would help readers evaluate whether the benchmark reflects plausible plasticity.  (you already justify parameters range, does the same justification work for the function form ?)

Conceptually:
- you are presenting the performances on the states but the evaluation of plasticity (evaluation of the parameters) changes from benchmark to benchmark
- similarly you don't provide systematic comparison with other methods

### Questions
* Identifiability and factor uniqueness: The authors reduce CP scaling/sign indeterminacy by constraining motif factors to unit‑norm and penalising orthogonality.  Do these constraints guarantee a unique solution, or could different factor orderings or scalings still yield the same connectivity?  How do they interpret the motifs biologically when multiple equivalent factorizations exist?

* Report performance on the inferred plasticity, not just on state prediction. The paper emphasises within‑session predictive accuracy ($R^2$) and explained variance on held‑out trajectories, but these metrics speak only to how well the model predicts neural activity.  For a method whose main goal is to infer the slow plasticity rule, one would also expect quantitative assessments of how accurately the latent plasticity dynamics $z_{k+1}=g_\theta(z_k,\bar{u}_k)$ and the motif coefficients $c_k$ recover the ground truth.  In the Lorenz and BCM benchmarks, the authors report a dynamical‑similarity score (DSA = 0.63) and correlations with BCM thresholds, but there is no systematic evaluation of plasticity inference across rank choices or hyperparameters.  Likewise, in the DBS experiment they assess alignment between $\Delta c$ and functional connectivity rather than the fidelity of the inferred slow law.  Expanding these analyses to provide error metrics on the inferred plasticity would strengthen the claim that STEER recovers the underlying rule.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper developed a framework, named Stimulus-Evoked Evolution Recurrent dynamics (STEER), a structured model that separates fast within-session activity from slow across-session adaptation. The method explored how recurrent connectivity evolves under repeated interventions.

### Strengths
1. The paper treats long-horizon plasticity as a latent dynamical law, rather than unstructured parameter drift.
2. The model separates fast within-session dynamics and slow across-session evolution.

### Weaknesses
1. The dynamical systems have input weights and readouts, which can also encode some information of connectivity. Therefore, it is still unsure if the connectivity recovered by the model is true or believable. (This may have been claimed by the author in the limitation part, but it remains a substantive concern.)
2. The authors stated that the proposed method enforces an identifiable separation between fast within-session responses and slow network reconfiguration. But there’s no theorem or ablation demonstrating uniqueness of learned dynamics.

### Questions
1. If you learn fast dynamics at first, then learn slow dynamics of these fast ones, will the results be different? Learning jointly requires hyperparameter search such as lambda slow, are the results sensitive to these parameters? 
2. Could the author define delta W? (Figure 4D) Since there are many Ws in the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces STEER, a dual-timescale recurrent framework for inferring long-term neural plasticity rules from longitudinal stimulation data. STEER models fast neural activity within sessions using a low-rank recurrent network, while slow structural changes across sessions are captured by a learnable latent dynamical system driven by stimulation input. Experiments on synthetic benchmarks and Parkinson’s DBS data show that STEER allows disentangle short- and long-term dynamics, recovers biologically interpretable motifs, and generalizes to unseen stimulation protocols.

### Strengths
1. **Motivation:** The paper addresses an underexplored area by extending short-term plasticity modeling toward the longer timescales of circuit reorganization. The proposed framework offers a structured, data-driven way to describe how stimulation may gradually reshape network connectivity.
2. **References:** The related research is carefully reviewed, linking established neuroscience findings on Hebbian and homeostatic mechanisms with recent machine learning approaches for recurrent dynamics and meta-learning. This grounding strengthens the biological and methodological motivation.
3. **Evaluation:** The method is tested on two synthetic datasets (Lorenz and BCM) and one real Parkinson’s DBS dataset. The experiments provide supporting evidence that the model can capture slow network adaptations and generate interpretable patterns, though further validation would be beneficial.

### Weaknesses
1. **Evaluation:** Results on the BCM and Parkinson’s DBS datasets appear modest, and there is a visible mismatch between the ground truth in Fig. 4(a) and the model output in Fig. 4(b), suggesting partial recovery of the connection change. 

2. **Baselines:** Only one baseline (MD-SSM) is evaluated on the BCM simulation. Including other relevant machine learning approaches discussed in the related work would strengthen the empirical comparison.

3. **Benchmark:** The evaluation spans two synthetic and one real dataset. Additional simulations or real neural datasets would better demonstrate the method’s generalization and robustness across experimental settings.

### Questions
1. Could the authors discuss how choices such as rank, learning rate, or regularization impact performance? What might explain the prediction gap between the ground truth in Fig. 4(a) and the inferred results in Fig. 4(b)?

2. How does the method scale to larger networks or longer time series? Please comment on computational cost and potential limitations for larger-scale simulations.

3. Since the model infers latent dynamics and connectivity from observed activity, how does it avoid learning spurious correlations due to partial observations and noises?

4. More implementation details (e.g., initialization, optimizer settings, runtime) would be helpful for reproducibility.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces STEER, a method to infer the dynamical rules underlying long-term evolution of recurrent connectivity over sessions/hours/days from neural data. The framework learns low-dimensional latent coefficients for the neural dynamics underlying a session, which themselves evolve over slower timescales – thus allowing the inference of plasticity rules explicitly. The framework is trained by jointly optimising for reconstruction fidelity within a session, regularisation for consistency of inferred plasticity rule across sessions, and also regularisation for smoothness. The authors validate their framework on three tasks – two synthetic tasks involving learning a Lorenz system with varying coefficients and synthetic data generated with a specific, known plasticity rule; and one real dataset of neural recordings from rats undergoing DBS treatment for Parkinson's disease. The main claim is that STEER disentangles effectively the within-session dynamics from cross-session dynamics, allowing better extraction and interpretability of the rules/changes underlying cross-session variability.

### Strengths
* The framework seems principled in its design and the approach overcomes disadvantages of certain prior meta-learning approaches by not assuming a specific functional form of the underlying learning rule (e.g., not restricted to just variants of Oja's rule).
* The presentation and figures are mostly clear.
* The experimental evaluations span both synthetic tasks and real data, which is important for such works.

### Weaknesses
* The use of DSA is good but you only report a single DSA value of 0.63 with no baseline or control/chance value. DSA is a relative metric, as emphasised in the original paper. Thus, it is not possible to know whether a DSA of 0.63 is good without a baseline comparison. Could the authors provide a baseline on shuffled values of the implicit factor, for example, and show that the actual inferred dynamics have a higher DSA score with the true dynamics?
* In Fig. 4f I am not sure you can claim that you perform comparably to MD-SSM without a statistical significance test. The avg. score for STEER seems lower even when considering error bars. What are the error bars over – sessions or seeds? If the result of a stat. test is not significant then that validates the claim of comparable performance. I would also ask that the plot y axis range be restricted so it's clearer what the difference in performance is.
* There is no quantification for Fig. 4d where it is claimed that STEER better recovers the $\Delta \mathbf{W}$ compared to MD-SSM. As far as I'm concerned, the block structure of both the MD-SSM nor STEER weight changes do not particularly resemble the ground truth weight changes. Could the authors comment on this? The value scales are also quite different and while I understand that exact values/scale may not matter, maybe they need to be normalised so the plots can be compared better?
* There is no comparison against the meta-learning approaches mentioned in the introduction. While I can understand the difference between these methods, I think it is important to see clear evidence to back up the claim that they cannot model the effects of DBS, for example, and also that STEER can recover plasticity rules as those methods do in short-term settings.
* Minor nit: in lines 130-131 you cite Bredenberg et al. and Kepple et al. as fitting synaptic plasticity rules through gradient-based optimisation on observational data. While Bredenberg et al. does do this, to my knowledge, they do not do it on real data. Meanwhile, Kepple et al. do not fit parameters for learning rules – they mainly evaluate different learning rules and curricula in an in silico setting on the basis of learning speeds (and classifying between these on the basis of such observations), again not working with real data to my knowledge.

### Questions
* Minor typo in Fig. 1a: "Brian" -> "Brain".
* Why is the evolution of coefficients jagged and not smooth in the Lorenz system plots?
* What is the difference between Fig. 4 and App. Fig. 1? Why are the results so different qualitatively?
* Could the authors provide additional information/motivation on the consistency and smoothness terms? How are the coefficient $\lambda$s tuned?

### Soundness
2

### Presentation
3

### Contribution
2
