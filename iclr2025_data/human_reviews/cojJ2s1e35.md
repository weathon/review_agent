## Human Reviewer 1

### Summary
The paper proposes a method akin to state-space models for simulating environments by applying Neural Fourier Transform. It uses Lie group theory to obtain a continuous representation of the environment dynamics.

### Strengths
- The paper is clearly structured and well written
- The use of the Lie group theory and NFT to model the dynamics of interactive environments seems to be both novel and interesting
- The method is well motivated and theoretically grounded
- Good reproducibility (code provided)

### Weaknesses
- No runtime analysis
- No discussion of hyperparameter selection
- No significance tests
- No error bars
- No scaling laws
- No quantitative results on Phyre

Unfortunately, the empirical evaluation does not meet scientific standards and, therefore, I do not deem the paper ready for publication yet. Once these problems are solved, I think this paper will constitute a nice contribution to the field.

### Questions
- In Eq. (1), does the function Ctrl return an observation or a transition operator?
- In Eq. (3), shouldn't one take the limit of delta to 0 instead of t to infinity?
- Are there any limitations arising from the restriction of transitions to Lie groups?
- Since the latent space is divided into different slots for different objects and the slots forward dynamics are independent of each other, how can the model represent interactions between objects? 
- It seems that the size of the latent space is a crucial hyperparameter; how was it determined (both N and J)?

### Soundness
1

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a method for enforcing a Lie group structure on the latent space of a multi-environment world model.  This structure imposes compositionality and continuity in the latent space, and allows the dynamics in the latent space to be linear.  The latent space is further structured by imposing object-specific dynamics (through the use of slot attention and assignment of "close" slots in nearby time steps).  This method allows to learn a common dynamic model from a sequence of observations sampled from a set of environments, and the paper shows that this model can generate observations that better reconstruct the ground truth observation compared to a strong baseline.

### Strengths
- The paper presents an interesting idea, and a clever method for exploiting similarities between several environments at once.
- Learning reliable world models from observation trajectories without paired actions would be a dramatic step forward, and this paper presents a novel angle on that problem.
- Evaluation shows improvements over a strong baseline.

### Weaknesses
- In places the presentation can be a bit unclear.  This is mostly in the description of the method.  I would have benefitted from a diagram of the steps required to train WLA and then use that to solve the CIP.  The current Figure 1 may not be necessary; the inter-environment aspect of WLA is probably the most straightforward step in the process.  
- Although not direct alternatives to the latent space presented here, there is some previous work in structured latent spaces that it may be interesting to compare against in the related work section.  eg.
    - Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.  Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, Martin Riedmiller
    - Object Files and Schemata: Factorizing Declarative and Procedural Knowledge in Dynamical Systems.  Anirudh Goyal, Alex Lamb, Phanideep Gampa, Philippe Beaudoin, Sergey Levine, Charles Blundell, Yoshua Bengio, Michael Mozer
- Not a weakness, but Line 111 contains a reference to OCCAM, which I presume is a previous name for WLA?

### Questions
- How are the parameters $N$ and $J$ chosen?  Do they need to scale with the complexity of the environment?  If so, how do we expect computation costs to scale? 
- Line 192 suggests an assumption about infinitesimal changes to observations between timesteps.  This assumption seems very strong, how would the method handle larger changes?
- How similar do the environments need to be for the inter-environment simulator to model them jointly?  For example, could you mix the Phyre and ProcGen environments?
- Does it matter how the dataset is sampled from each environment?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
In this paper, the authors present World modeling through Lie Action (WLA), a generative state-space-modeling framework that can be trained in an unsupervised fashion on interactive environments, and that allows controllable generation of future frames of a given environment. One main goal of this framework is learning continuous and compositional action representations, similar to what humans are able to do. It is a step towards an interactive world model that generalizes across environments, given these environments have a common basic rules of composition and continuity.
The paper presents a model built using this framework that is benchmarked on two datasets (ProcGen and Phyre). In general, the model can be trained using no or only few action labels, making it more versatile and faster to adapt to new settings with different action labels.

### Strengths
1. The paper is clearly structured, and the individual components of the model with accompanying formulas, as well as the setting is well introduced. The authors carefully set the scope of the work and how it compares to similar approaches.
2. Benchmarks with different test settings (e.g. different FPS compared to train) shows the robustness of the model on this benchmark and the ability to infer continuous dynamics.
3. Given a main focus of the framework is the Lie group theory, I appreciate the ablation study that reveals the relevance of it (Rotation) to the predictive performance of the model

### Weaknesses
My main concern is the limited amount of comparability to previous works and approaches. The model is only benchmarked on two datasets, and one is mainly used as a (successful) sanity check. The second dataset (with 16 quite diverse environments) is benchmarked against only one other model, though using several metrics. Even if it is a ‘first of its kind’ framework as stated by the authors, more benchmarks to confirm its proper function would increase trust in the method/framework. It can be acknowledged though that the number of available datasets/benchmarks for this specific scope is low, as also stated in a recent survey [1].
Other things that are not clear to me at the moment are in the question section.


[1] McCarthy, Robert, et al. "Towards Generalist Robot Learning from Internet Video: A Survey." arXiv preprint arXiv:2404.19664 (2024).

### Questions
1. Was the encoder and decoder trained from scratch?
2. Did you do multiple runs with different random initializations, or are the results from just one experimental run?

### Soundness
2

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
The paper introduces World Modeling through Lie Action (WLA), a generative framework that learns continuous, compositional actions for controlling agents across diverse environments using Lie group theory. Claimed contributions include:

Unified Simulator: A shared, environment-agnostic simulator leverages continuous actions, unlike models with discrete, environment-specific setups.
Controller Interface Problem (CIP): WLA solves CIP with adaptable controllers, even under minimal labeling.
Object-centric Modeling: Slot-based object-centric modeling supports interactions across multiple objects and actions.
Performance: WLA seems to outperform on benchmarks (Phyre and ProcGen), proving its generalizability and adaptability.
This approach enhances state-space modeling with flexible, multi-environment simulations

### Strengths
The originality seems novel. The universal interface for control is an important problem to study. The emprical evidence seems to be strong, it can outperform the baseline significantly. I think there could be some contributions in the paper, but the presentation needs to be improved a lot.

### Weaknesses
The paper’s ambitious framework for inter-environmental modeling has notable weaknesses, especially in mathematical clarity and formulation, which hinder its accessibility and rigor.
 Key Issues:
1. **Ambiguity in Transition Operator (\( g_{t, \delta} \))**: \( g_{t, \delta} \) maps specific observations \( x(t) \) to \( x(t+\delta) \), making it appear as a trivial one point to one point mapping. This definition does not capture the desired dynamic evolution. It does not make any sense in the form written in the paper. The paper acknowledges that \( g_{t, \delta} \) depends on individual trajectories but simplifies it as a generic operator, risking confusion. This omission detracts from the model’s mathematical precision, especially for multi-environment dynamics.

3. **Underdefined Action Space \( A \)**: The action space \( A \) and action \( a(t, \delta) \) lack clarity on whether actions are fixed or variable over intervals. This ambiguity in structure reduces the framework's comprehensibility in continuous control scenarios.

4. **Non-Compositional Transition Operators**: Due to the triviality of \( g_{t, \delta} \), composing transitions over time is not feasible, so why does it form a Lie group? The paper introduces Lie groups without a clear justification for their relevance to the specific environments. Additional reasoning would strengthen its argument for using Lie groups to model compositional and continuous dynamics.

To improve clarity, the paper would benefit from revisiting the definitions and dependencies in its transition operators, better defining the action space, and providing justification for its mathematical choices. More precise formulation and notation would significantly enhance its accessibility and application.

### Questions
Revise the math definitions and formulations significantly and I will reconsider my score.

See weakness for all the math confusions.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
5