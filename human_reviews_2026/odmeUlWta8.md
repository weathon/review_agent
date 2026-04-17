# In-Context Multi-Objective Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Balancing competing objectives is omnipresent across disciplines, from drug design to autonomous systems. Multi-objective Bayesian optimization is a promising solution for such expensive, black-box problems: it fits probabilistic surrogates and selects new designs via an acquisition function that balances exploration and exploitation. In practice, it requires tailored choices of surrogate and acquisition that rarely transfer to the next problem, is myopic when multi-step planning is often required, and adds refitting overhead, particularly in parallel or time-sensitive loops. We present TAMO, a fully amortized, universal policy for multi-objective black-box optimization. TAMO uses a transformer architecture that operates across varying input and objective dimensions, enabling pretraining on diverse corpora and transfer to new problems without retraining: at test time, the pretrained model proposes the next design with a single forward pass. We pretrain the policy with reinforcement learning to maximize cumulative hypervolume improvement over full trajectories, conditioning on the entire query history to approximate the Pareto frontier. Across synthetic benchmarks and real tasks, TAMO produces fast proposals, reducing proposal time by 50–1000× versus alternatives while matching or improving Pareto quality under tight evaluation budgets. These results show that transformers can perform multi-objective optimization entirely in-context, eliminating per-task surrogate fitting and acquisition engineering, and open a path to foundation-style, plug-and-play optimizers for scientific discovery workflows.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TAMO, a Transformer-based algorithm for black-box multi-objective optimization using a pre-trained policy. It is trained on synthetic Gaussian Process tasks with varying input and output dimensions, combining reinforcement learning and supervised losses for model-agnostic optimization. Experiments show that TAMO matches or surpasses state-of-the-art methods while being 50–1000 times faster at inference time.

### Strengths
1. This paper proposes a _Model-amortized Multi-objective Optimization_ algorithm that achieves significantly lower inference time compared to other MOBO methods. It could be one of the earliest works to leverage the scalability and in-context learning capability of Transformer Neural Process models to improve inference efficiency in MOBO context.
    
2. The paper introduces a  Dimension-agnostic embedder to the Transformer that enables dimension-agnostic learning across various MOBO tasks, making it highly generalizable to unseen tasks with varying dimensions, using a single pre-trained model. By adapting this architecture to the MOO setting, the method goes beyond a naive extension of similar approaches in single-objective Meta-BO [2] and BOFormer [1], which also implements a Transformer policy, but without dimension-agnostic learning.

### Weaknesses
Besides the limitations mentioned in the paper, there are some non-negligible issues:
- **Lack of statistical significance in task generalization performance** . The confidence bars of all the baselines, including TAMO, in the out-of-distribution evaluations of problems GP-DX3-DY2, GP-DX3-DY3 in Figure 4 (a) are not well-separated. Thus, Figure 4. only demonstrate limited evidence for the statistical significance of TAMO's superiority in task generalization.
- **Lack of benchmarks.** Although both synthetic and real-world benchmarks were evaluated, the input and output dimensions are relatively small (both ≤ 4), which limits the results to small-scale MOO problems. Note that the most similar and recent work BOFormer [1] includes benchmark performance with **input dimension up to 30**. Also, there is only a single real-world benchmark compared in the experiments, which is not sufficient to demonstrate the general performance of TAMO in real-world settings.
    
- **Lack of baseline comparisons.**
    
    1. The information-theoretic approaches discussed in the related work section were not included in the experimental comparisons. Other Transformer-based policies applicable to MOBO settings , such as **Optformer** [4], should also be considered for comparison, which is also compared in the paper of BOFormer [1].
        
    2. The performance gap between GP-based and Transformer-based models is not well studied experimentally. Given that GP and Transformer models differ substantially in scale, direct comparison is difficult. To better evaluate the usefulness of the Transformer Neural Process in terms of regret performance, variants of TAMO should be included in the ablation study, e.g., versions with RL policies using other policy parametrizations should be considered, e.g., smaller neural networks (as in [3]). 
- **Necessity of the RL policy.** Although the use of Transformers is well-motivated, which possess strong in-context learning capabilities, the paper provides limited evidence supporting the need for a Transformer-based RL policy. Future experiments could include a baseline that performs GP-based MOBO (e.g., qNEHVI, qParEGO) but replaces the GP surrogates with Transformer Neural Process surrogates (pre-trained on GP priors). Comparing TAMO against such a baseline would help clarify whether using RL to learn a policy is necessary, compared with directly employing a Transformer surrogate. Moreover, this proposed baseline should also benefit from reduced inference time through forward-pass inference, thus, this comparison would also help evaluate the effectiveness of TAMO’s RL policy under similar low-inference-cost conditions

### Questions
See Weakness,

and 

- Why is a **mixed-normal regression head** (equation 1) used for the prediction head? Would it be possible to apply a **Bar distribution (Riemann distribution) regression head** instead? [5] has shown that normal regression heads may underperform bar distribution regression heads in terms of supervised-learning efficiency in a similar setting of fitting GP priors.
    
- Is there a specific reason for using the **normalized hypervolume level** as the reward, rather than the **normalized hypervolume improvement** [1]? Using net improvement as reward signal is also common in RL-based BO policies [2,3] for single-objective problems. Conceptually, would using the normalized hypervolume level allow repeatedly rewarding the same query, potentially slowing down TAMO’s exploration?
    
- **Potentially high training time?** Although TAMO achieves significantly lower inference time than other methods, Transformer-based BO policies have been shown to require substantial training resources (as noted in [2]), even for single-objective BO. There may be a potential issue of high training cost for complex real-world problems with high input/output dimensionality. It would be helpful for the authors to discuss TAMO’s training time.

- In Section 3.2, the definition of positional tokens is confusing for me. In Line 178,
	 "These positional tokens are randomly sampled for each batch from fixed pools of learned embeddings"
	What's is the definition and the practical implementation of this "fixed pools of learned embeddings?"
- The Section 3.2 is difficult to read, I think it could be improved by providing a full model architecture illustration in Figure 2., explicitly drawing the $B_1, B_2$ blocks of layers and use arrows or attention maps to indicate which tokens can attend to which tokens in each phase.

**Minor issues**
- Figure 4. is confusing to me before I found (a), (b) in the subfigures. They look very small and are in the top of the subfigures. 
- Some Figure references in the appendix are not well-compiled.


**Citations**

[1] Hung, Y. H., Lin, K.-J., Lin, Y.-H., Wang, C.-Y., Sun, C., & Hsieh, P.-C. (2025). BOFormer: Learning to solve multi-objective Bayesian optimization via non-Markovian RL. The Thirteenth International Conference on Learning Representations. [https://openreview.net/forum?id=UnCKU8pZVe](https://openreview.net/forum?id=UnCKU8pZVe)

[2] Alexandre Max Maraval, Matthieu Zimmer, Antoine Grosnit, and Haitham Bou Ammar. End-to-end
meta-bayesian optimisation with transformer neural processes. Advances in Neural Information
Processing Systems, 34, 2023.

[3] Michael Volpp, Lukas P. Fr¨ohlich, Kirsten Fischer, Andreas Doerr, Stefan Falkner, Frank Hutter,
and Christian Daniel. Meta-learning acquisition functions for transfer learning in bayesian opti-
mization. In International Conference on Learning Representations, 2020.

[4] Yutian Chen, Xiaoxi Song, Chung-Ching Lee, Zihang Wang, Ruoxi Zhang, David Dohan, Kenji
Kawakami, Greg Kochanski, Arnaud Doucet, Marc’Aurelio Ranzato, et al. Towards learning
universal hyperparameter optimizers with transformers. In Advances in Neural Information Pro-
cessing Systems, volume 35, pp. 32053–32068, 2022.

[5] Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., & Hutter, F. (2022). Transformers can do Bayesian inference. International Conference on Learning Representations. [https://openreview.net/forum?id=KSugKcbNf9](https://openreview.net/forum?id=KSugKcbNf9)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TAMO, a fully amortized, task-agnostic, dimension-agnostic policy for multi-objective black-box optimization. A single transformer backbone, trained offline with (i) an in-context prediction warm-up and (ii) a policy-level RL objective that maximizes trajectory hypervolume (HV) progress, maps histories and a candidate set to the next query in one forward pass—eliminating per-task surrogate fitting and acquisition optimization. Key architectural elements include a dimension-agnostic embedder, encoder–decoder with task-specific tokens, and dual prediction/policy heads; the policy is trained with REINFORCE on HV-normalized rewards. Empirically, TAMO achieves **50×–1000×** lower proposal time than GP-based MOBO and BOFormer while delivering competitive regret on synthetic and real tasks; it also shows transfer to unseen input/output dimensionalities and decoupled observations. Code is promised upon acceptance; hyperparameters, algorithms, and synthetic pretraining data generation are documented.

### Strengths
* **End-to-end amortization:** Eliminates per-task surrogate fitting and acquisition optimization; single forward-pass proposals reduce decision latency by **50×–1000×**. 
* **Dimension-agnostic architecture:** Embedder + task-tokens operate across varying dx, dy; supports heterogeneous pretraining and cross-dimensional transfer. 
* **Non-myopic training signal:** RL objective directly optimizes trajectory HV, aligning learning with Pareto-front discovery over horizons. 
* **Generalization studies:** Evidence of transfer to unseen dimensionalities and decoupled observations under fixed budget/cost accounting. 
* **Methodological transparency:** Clear preliminaries, training/inference algorithms, hyperparameters; baselines (qNEHVI, qNParEGO, qHVKG, BOFormer) implemented with standard toolchains. 
* **Performance profile:** Competitive or better regret across several synthetic and real tasks, with consistent runtime advantage.

### Weaknesses
* **OOD gaps & sensitivity:** Underperforms classic MOBO on Branin–Currin and LaserPlasma; authors attribute mismatch to pretraining length-scales—this warrants a systematic analysis. 
* **Discrete candidate pool assumption:** Inference relies on a fixed candidate set; implications for high-dimensional continuous or combinatorial design spaces are acknowledged but unresolved.

### Questions
1. **Timing fairness & ablations:** Please report per-candidate acquisition latency (µs/ms) and GPU-to-GPU comparisons against GPyTorch/BoTorch with matched MC budgets; include breakdown of surrogate refit vs. acquisition time. 
2. **Pretraining-prior sensitivity:** Provide a controlled study varying kernel families, length-scales, ARD, and output correlations to quantify transfer sensitivity and explain gaps on Branin–Currin/LaserPlasma. 
3. **Continuous action spaces:** Can TAMO couple to a continuous proposal mechanism (e.g., gradient-guided refinement, learned sampler) to move beyond pool-based scoring? Any preliminary results? 
4. **RL stability:** What variance-reduction techniques were used for REINFORCE (e.g., learned baseline, advantage normalization)? Any ablation on γ and λ_{rl}? 
5. **Decoupled observations policy:** How are costs integrated at training time? Could explicit cost-aware reward shaping further improve decoupled performance (e.g., on Ackley–Rosenbrock)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper targets three flaws with traditional multi-objective Bayesian optimization methods; that they require re-training from scratch for each use, the reliance on a large number of hyperparameters, and the focus on single-step gains. The authors introduce TAMO, which leverages a transformer-based optimizer trained on diverse synthetic tasks using reinforcement learning to learn multi-step strategies. Once trained, users use iterative forward passes of TAMO and a tokenized history of the steps inputted into TAMO to run the optimization process. The authors claim to match or improve the quality of state of the art methods while reducing proposal time.

### Strengths
The authors present a novel architecture which functions as both the acquisition function and surrogate model. The dimension-agnostic embedder is an interesting solution to generalizing over spaces with varying dimensions. TAMO is demonstrated to have equivalent or superior metrics to many other methods, and is demonstrated to run in a shorter time. The paper is overall pretty clear.

### Weaknesses
1. In which cases do multi-step strategies actually matter? When is myopia actually a limiting factor for Bayesian optimization? Can the authors demonstrate an ablation where multi-step strategies provide a clear advantage over traditional BO?

2. There is only one real-world benchmark. Can the authors demonstrate applicability to other real-world scenarios, such as for Gaussian splatting as done in the Boformer paper (Yu-Heng Hung, et al. 2025) or neural network hyperparameter selection as in OptFormer (Yutian Chen, et al. 2022).

3. Can the authors demonstrate clear advantages in practice to other transformer-based MOO methods, such as Optformer?

4. The authors mention that the pretraining data composition is important for generation. The authors should provide a study to experimentally demonstrate how the diversity of the pretraining data affects optimization performance. For the out-of-distribution experiments, authors should demonstrate

### Questions
Is the prediction task fitting to the raw data? Does the policy part of the model not converge without the prediction task? Why is this? Is there any practical application to the prediction mechanism post-training? This seems to be a central part of the architecture and is not thoroughly justified. Can the authors demonstrate an ablation for training with and without the prediction task?
Can the authors include a figure for the architecture for 3.2 (II) and (III)? The exact model architecture (where self vs cross attention is applied, where task-specific tokens are inputted) is unclear simply due to wordiness and a simple diagram could very clearly explain it.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces TAMO, a novel amortized policy for multi-objective black-box
optimization, aiming to address the high computational cost and task-specific nature of GP-based
methods. The core idea is to pre-train a single, dimension-agnostic Transformer on a diverse
corpus of synthetic tasks. This pre-trained model can then be deployed on new, unseen problems,
replacing the slow, iterative refitting of a surrogate model with a single, fast forward pass to
propose the next query. The model is trained using a combination of a prediction loss and a
non-myopic RL objective based on cumulative hypervolume improvement .

### Strengths
● Paper is very well written and easy to understand
● By replacing the iterative GP refitting and acquisition optimization process with a single
neural network forward pass, the method reduces inference latency by 50-1000x.
● The proposed dimension-agnostic embedder is a clever architectural contribution . It
allows a single Transformer backbone to be pre-trained on and deployed to problems of
varying input and output dimensions

### Weaknesses
● The framework relies on a pre-defined discrete candidate pool from which the policy
head selects the next query. This is a significant limitation as it makes the approach
unusable for true continuous-domain optimization or generative tasks (like de novo drug
design).
● The "task-agnostic" claim is weakened by the model's sensitivity to the pre-training data.
The authors hypothesize that the poor performance on BraninCurrin stems from not
seeing those objective properties in the synthetic pre-training corpus. The model isn't
truly "agnostic" but is rather "multi-task" for a specific family of synthetic GP-based
tasks.

### Questions
● The model has two heads (Prediction and Policy) and is trained with a joint loss.
However, the contribution of the auxiliary prediction task L(p) is not ablated. How much
does the "warm-up" and joint training contribute to the final policy versus simply training
the policy head alone?

### Soundness
3

### Presentation
3

### Contribution
3
