# Interpretable time series analysis with Gumbel dynamics

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 2, 6

## Abstract
Switching dynamical systems can model complicated time series data while maintaining interpretability by inferring a finite set of dynamics primitives and explaining different portions of the observed time series with one of these primitives. However, due to the discrete nature of this set, such models struggle to capture smooth, variable-speed transitions, as well as stochastic mixtures of overlapping states (e.g., non-instantaneous state transitions), and the inferred dynamics often display spurious rapid switching on real-world datasets. Here, we propose the Gumbel Dynamical Model (GDM). First, by introducing a continuous relaxation of discrete states and a different noise model defined on the relaxed-discrete state space via the Gumbel distribution, GDM expands the set of available state dynamics, allowing the model to approximate smoother and non-stationary ground-truth dynamics more faithfully. Breaking from established literature, this new class directly links states to observations and does not blur latent dynamics with Gaussian noise. Second, the relaxation makes the model fully differentiable, enabling fast and scalable training with standard gradient descent methods. We validate our approach on standard simulation datasets and highlight its ability to model soft, sticky states and transitions in a stochastic setting. Furthermore, we apply our model to two real-world datasets, demonstrating its ability to infer interpretable states in stochastic time series with multiple dynamics, a setting where traditional methods often fail.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Gumbel Dynamical Model (GDM), a switching dynamical system in which the discrete state is replaced by a Gumbel–Softmax continuous relaxation, enabling gradient-based training and “soft, sticky” state transitions that better capture variable-speed and overlapping regimes than classical methods. The generative model mixes per-state linear dynamics in a projected space to produce observations and uses amortized variational inference with a differentiable posterior over states. The paper shows an equivalence to a three-level mixture SLDS, discusses identifiability caveats, and argues that keeping noise only at the observation level avoids competition between state and latent-trajectory noise. Empirically, on synthetic NASCAR the method attains test “inferred state accuracy” of 0.88±0.10 in the standard setting and 0.70±0.03 in a harder “soft-sticky” setting where all baselines degrade to ≤0.43±0.09.

### Strengths
1. In terms of originality, relaxing the discrete state with a Gumbel-Softmax and designing the generative and inference maps to remain interpretable yields an expressive yet still human-legible model. The equivalence to a mixture SLDS clarifies what is novel versus inherited.
2. In terms of clarity, the model equations, training objective, and forecasting procedure are explained coherently with concrete choices, and the evaluation metric for state quality is operationalized via a k-NN trained on soft states to ground-truth labels.
3. In terms of significance, the combination of interpretable state mixtures, fewer rapid switches, and amortized inference that generalizes across trials makes GDM a practical tool for scientific time series with overlapping regimes where classic SLDS methods often fail.

### Weaknesses
1. The empirical study under-reports statistical variability, omits computational accounting, and leaves several core hyperparameters unexplored, such as the temperature, stickiness, softness, and projection dimension. These matters because the relaxation can blur state boundaries and the stickiness term changes dwell-time statistics.
2. Although the paper notes identifiability issues in the three-level formulation, it does not probe identifiability between the state-mixture dynamics and observation noise in the two-level GDM, nor does it quantify how increasing $\tau$ affects spurious blending versus accuracy.

### Questions
1. Please provide confidence intervals and seed counts for the results presented in tables and figures.
2. Please consider adding ablations regarding the terms mentioned in Weakness - 1.
3. How F is chosen for each dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces the Gumbel Dynamical Model (GDM), a differentiable and interpretable time-series model that extends dynamical systems. The key idea is to replace discrete latent states with continuous Gumbel-Softmax relaxations, enabling end-to-end gradient-based learning while preserving the interpretability of local dynamics. An amortized inference network predicts per-step logits for the Gumbel-Softmax states, forming a fully differentiable variational inference scheme.

### Strengths
1. The paper bridges discrete switching dynamical systems and differentiable deep models through a novel use of Gumbel-Softmax relaxation.
2. Derivations of Gumbel dynamics and the corresponding amortized variational inference are rigorous and internally consistent. 
3. Extensive experiments are conducted on several datasets with different levels of complexity.

### Weaknesses
1. The model’s behavior seems to critically depend on the Gumbel-Softmax temperature, yet no sensitivity analysis is provided. 
2. The “interpretable” property originates from the switching dynamical system itself, GDM mainly preserves this structure while making inference differentiable. A clearer formal definition or quantitative measure of interpretability would strengthen the claim.

### Questions
1. How sensitive are results to the temperature value?
2. Could GDM scale to high-dimensional observations?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes GDM, an estimation method for recurrent SLDSs and AR-HMMs. The methodology is based on introducing Gumble-softmax relaxations on the discrete variables. The latent variables are inferred via variational inference with MC sampling from the discrete posterior, and the model enables backpropagation thanks to the proposed relaxation. The paper demonstrates performance of GDM via synthetic NASCAR trajectories, F1 data, and mouse trajectories.

Unfortunately, the paper ignores critical related works which have specifically studies similar continuous relaxations of the discrete states for Switching Dynamical Systems.

### Strengths
- The presentation of the paper is adequate.
- The paper presents applications to real-world scenarios which are very promising for SDSs.

### Weaknesses
- The main concern is the novelty front of this paper. Unfortunately, GDM does not seem a novel methodology. See below a comparison of related works which might be relevant.
- The paper does not cite important related works that address inference in recurrent switching dynamical systems after Linderman's 2017 paper (sticky priors). There is a citation on a 2021 Neurips paper, but it involves group dynamics in SDSs. See below a list of critical related works:
  - Fraccaro et al. (2017) [1]: Kalman VAE. A Kalman filter in latent space, with a soft-switching mechanism. In their paper, they propose using a gumble-softmax sampling approach to estimate switches.
  - Becker-Ehmck et al. (2019) [2]: I believe this paper proposes exactly GDM (see Section 2.3 in their paper), for exactly the same type of SDSs (recurrent and linear).
  - Dong et al. (2020) [3]: This paper uses recurrent nonlinear SDS, it proposes exact inference on the switches "collapsed", but in the experiments they also propose an additional method as baselines which uses Gumble-softmax relaxation in a similar way as GDM. See Appendix A.2 in their paper. Importantly, their amortized inference network uses a bidirectional RNN to encode the sequence and a forward RNN to produce per-time-step switch posteriors, which matches the inference architecture proposed in this paper for GDM.
  - Ansari et al. (2021) [4]: Same as Dong et al. (2020), but the switches are explicit duration models.
- For acceptance, the paper should:
  - compare empirically to [1]–[4] on the proposed benchmarks, and
  - clearly articulate what is substantively new beyond those works. At present, I am not able to identify a principled difference between Becker-Ehmck et al. (2019) [2] and the proposed GDM; the submission presents GDM as novel, but it looks like a restatement of [2] (and of Dong et al.’s amortized inference design).
- The presented methodology lacks theoretical analysis on the presented methodology. Analysis on whether GDM is identifiable would be an important problem (i.e. the ground-truth representation is unique up to certain equivalence, e.g. permutaion of the switches). Without at least a statement of the assumptions under which GDM recovers meaningful modes, it is difficult to argue that the discovered “states” are interpretable rather than arbitrary mixtures.


[1] Fraccaro, Marco, et al. "A disentangled recognition and nonlinear dynamics model for unsupervised learning." Advances in neural information processing systems 30 (2017).

[2] Becker-Ehmck, Philip, Jan Peters, and Patrick Van Der Smagt. "Switching linear dynamics for variational bayes filtering." International conference on machine learning. PMLR, 2019.

[3] Dong, Zhe, et al. "Collapsed amortized variational inference for switching nonlinear dynamical systems." International Conference on Machine Learning. PMLR, 2020.

[4] Ansari, Abdul Fatir, et al. "Deep explicit duration switching models for time series." Advances in Neural Information Processing Systems 34 (2021): 29949-29961.

### Questions
- Are there any estimation guarantees of the proposed approach?
- Is it possible to determine the number of switching mechanisms, are the switches identifiable?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Gumbel Dynamical Model (GDM), a new interpretable framework for modeling complex, non-stationary time series. By introducing a Gumbel-Softmax relaxation of discrete latent states, GDM bridges the gap between interpretable switching dynamical systems and smooth, continuous state transitions. The model remains fully differentiable, allowing efficient optimization via standard gradient descent and enabling amortized variational inference that generalizes to unseen sequences without re-optimization. Empirical evaluations on synthetic (NASCAR), real-world (Formula 1 telemetry), and behavioral (CalMS21 mouse social interaction) datasets show that GDM achieves higher state interpretability and comparable or better predictive accuracy than strong baselines such as SLDS, rSLDS, and p-dLDS, particularly under stochastic and soft transition regimes.

### Strengths
The paper presents a technically sound and conceptually elegant contribution that enhances both interpretability and scalability of state-space models. The Gumbel-based relaxation is well-motivated and effectively mitigates spurious rapid switching common in discrete-state models. The proposed amortized inference framework is a practical strength, enabling efficient reuse across datasets and improving generalization without retraining. The experiments are extensive, diverse, and clearly demonstrate GDM’s advantages in recovering meaningful latent states and handling uncertainty. The writing is clear and thorough, with strong empirical grounding and transparent comparisons against relevant baselines.

### Weaknesses
While the paper is thorough and technically solid, its scope remains focused primarily on the SLDS family of models. It does not explore broader connections to other modern dynamical modeling frameworks—such as continuous-time approaches like Neural ODEs or more expressive deep sequence models—which could help position GDM within the wider landscape of time-series modeling. The discussion of the Gumbel relaxation, while mathematically correct, could be made more intuitive to clarify why interpretability is preserved under soft transitions. Additionally, the impact of the temperature parameter is discussed but not systematically explored or adapted in experiments.

### Questions
1) Have the authors considered learning or annealing the Gumbel temperature τ instead of fixing it, to better balance interpretability and flexibility?
2) Can the authors provide or suggest a quantitative metric to evaluate interpretability beyond qualitative visualizations?
3) How does GDM relate to or differ from continuous-time latent models (e.g., Neural ODEs), and could the approach extend to that setting?

### Soundness
3

### Presentation
3

### Contribution
3
