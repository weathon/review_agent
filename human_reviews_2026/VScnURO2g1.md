# APILaNet: Adaptive Physics-Informed Latent Network for Single-Sensor Forecasting

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Forecasting conservation-governed dynamics is often constrained by sparse sensing: in practice, we may have only a single boundary sensor and noisy exogenous variables. In this work we design an Adaptive Physics-Informed Latent Network (APILaNet) that learns a latent field and enforces 1D-conservation of physics law in the weak form using a learned, normalized space--time measure. Normalization makes physics enforcement insensitive to quadrature resolution and concentrates it on transient violations. A monotone, Lipschitz measurement layer maps latent variables to observed targets, improving identifiability from a single sensor. An adaptive, bounded scheduler scales the physics and smoothness loss terms with meaningful representations, emphasizing conservation of physics laws during events while preserving training stability. Learning a space-time measure for weak-form enforcement, combined with a monotone mapping and adaptive scheduling, enables accurate, data-efficient single-sensor forecasting in physics-governed systems. We evaluate APILaNet through a synthetic and hydrological case study, APILaNet outperforms strong sequence baselines and reduces MSE during extreme events, while improving Nash--Sutcliffe efficiency. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces APILaNet, a deep learning framework for single-sensor forecasting in systems governed by conservation laws. It learns a latent spatiotemporal field and enforces physical constraints through a learned, normalized space–time weighting measure, allowing the model to prioritize physics enforcement where violations are largest. Applied to five UK hydrological catchments, APILaNet outperforms SoTA baselines in mean squared error (MSE), Nash–Sutcliffe efficiency (NSE).

### Strengths
- The use of a measure-weighted weak form and learned space–time weighting map is innovative and mathematically principled. It avoids the dense collocation points needed in traditional PINNs and adapts well to single-sensor setups, a major gap in current PINN and PDE-learning research.

- The paper presents comprehensive experiments across multiple catchments, with consistent improvements in MSE/NSE and stability. Ablation studies clearly show the contribution of each component (monotone link, PDE term, adaptive scheduler), suggesting a well-engineered and reproducible design.

### Weaknesses
- The individual elements: weak-form enforcement, monotonic constraints, and adaptive loss scaling, are each known techniques. The contribution is largely an integration and reparameterization of these under one framework, not a fundamentally new paradigm.

- The adaptive scheduler introduces several hyperparameters that appear tuned by hand, with no sensitivity analysis. This undermines claims of robustness and could limit reproducibility.

- Ultimately, with several theorems and formalisms, all experiments are hydrological and one-dimensional. The claims of generality  are unsupported by cross-domain tests such as fluid or thermodynamic systems, making the contribution more applied than general.

### Questions
See my above on weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a novel method for PDE-constrained ML, especially targeting 1D conservation laws. The paper vows to improve upon PINNS, especially in situations in which observations are sparse or even only available at one sensor location. The proposed methodology, "APILANET" accomplishes this by constructing a latent spatio-temporal domain anchored at observations. Overall, I think this is impressive work that may very well have a high impact on the community and addresses an important gap: sparse observation of PDE-governed data and the need for the associated constrained learning of entire space-time fields.

### Strengths
- Impact and Relevance: This is an important area of research; PDE-informed ML is a fantastic contribution. 
- A good motivation involving the weaknesses of PINNs.
- Product: This study produces a software product for general users.
- An ablation study is included.
- A good number of tested competitors.
- Provided statistics of performance measures.
- Strong results: The proposed method performs admirably compared with the chosen competitors in the present test. 
- No obvious mistakes in the methodology were apparent to me.

### Weaknesses
- One experiment only; The experimental validation needs more examples. I would suggest a synthetic example and one more real-data example. This would also give the authors the chance to explain inputs and outputs more clearly. 
- Figure 1: This figure should be simplified or made larger.  It is very difficult to parse. Maybe certain steps in the pipeline can be consolidated into fewer boxes. 
- Language. It is tough, in places, to follow the manuscript because of insufficient writing quality. For example, "and shown equivalent to" should be  "shown to be equivalent to", or "(2) Theory—conditions
for single-gauge identifiability under a monotone, Lipschitz observation and mild driver excitation, reparameterization invariance of the weak objective on the latent reach, and an equivalence
between learned-density and learned test-function formulations;" which is not a sentence because of the missing verb. I suggest going through the manuscript with a fine-tooth comb to avoid mistakes like this because they make it much harder for the reader to follow the author's thought process. 
- The manuscript states, "Although motivated by hydrology, the framework applies to 1-D conservation laws under sparse spatial supervision." Is there an "all" or a "many" missing? This sentence is really just an example of a broader issue: it is difficult to decipher the exact target application area. It seems to be all 1D conservation laws, but the theory would suggest that higher-dimensional situations are considered. 
- I believe the paper used language from Hydrology, which should probably be avoided in the technical parts (example: "reach", "rainfall" in 3.3, "catchments"). The language could stem in part from other areas I am not familiar with.

### Questions
What is the exact application area of the proposed method? I got confused about the dimensionality of the considered domains. Is it all 1D conservation laws? If so, that should be stated clearly.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes APILaNet (Adaptive Physics-Informed Latent Network) for forecasting conservation-law dynamicswhen you only have one downstream sensor plus exogenous drivers like rainfall. The core idea is: instead of trying to enforce the PDE at many physical points you don’t actually observe, the model creates a latent 1-D reach, broadcasts the predicted temporal derivatives across it, and then enforces the PDE in a weak form under a learned, normalized space–time measure. That learned measure tells the model where in latent space–time to care most about conservation, so it can focus on transients (flood peaks, sharp inflows) and not waste physics budget on flat periods.

### Strengths
- Improved experimental results on various tasks
- Interesting, principled latent weak-form idea

### Weaknesses
- All in one domain

- Missing direct comparison to existing adaptive PINNs on the same data

- Some implementation details of the scheduler are underspecified in the main paper

### Questions
- Exact construction of the “event likelihood” / regime signals s
Are these pure data-driven (from residuals)?

- The driver projection Rκ(t, x) = ¯r(t) e −κx  
Is κ global for the whole dataset, per-catchment, or learned per-sequence/minibatch?

- How the latent grid size X is chosen. esp for different settings

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces APILaNet, a neural framework for forecasting physical systems when only one sensor is available. It learns a hidden space–time field where conservation laws are enforced in a weak, measure-weighted form, making the model robust to sparse spatial data. A dual LSTM captures slow and fast flow components, a monotone neural mapping ensures physically consistent observations, and an adaptive scheduler adjusts physics constraints based on signal difficulty. Applied to hydrological forecasting, APILaNet consistently outperforms leading deep sequence models in accuracy and stability, especially during extreme events.

### Strengths
The paper’s strength lies in tackling the single-sensor forecasting problem, which is a genuinely tricky and underexplored setup. The idea of using a weak-form physics constraint with a learned space–time weighting is a thoughtful technical twist: it avoids the brittle collocation sampling that usually plagues PINNs. The monotone mapping between discharge and stage is a sensible touch: it mirrors how water levels in rivers actually rise with flow instead of letting the network invent unrealistic relationships. The adaptive physics scheduling, while somewhat heuristic, shows awareness of practical training instability in physics-informed models and attempts to handle it dynamically. These pieces together make the framework conceptually interesting for sparse, physically constrained domains.

### Weaknesses
The overall novelty appears incremental, and the architecture feels somewhat overengineered relative to its contribution. The combination of multiple components (dual LSTMs, adaptive schedulers, weak-form latent mesh, and monotone mapping) adds complexity without a clear demonstration of which elements are essential or theoretically justified. The problem setup: "forecasting from a single downstream sensor with exogenous rainfall", is well-motivated but rather narrow, which may limit its broader relevance to general physics-informed or sequence modeling audiences. The theoretical component, particularly the “learned measure” weak form, seems to build on established weak PINN concepts with limited conceptual advancement. Empirical results show consistent but modest improvements, and could be strengthened by more robust statistical analysis and tests beyond hydrology. The paper would benefit from simplifying the model to highlight the key idea more clearly and from expanding the scope or validation to illustrate broader applicability.

### Questions
How sensitive is APILaNet’s performance to the specific choice of latent spatial discretization and the learned weighting measure?

### Soundness
2

### Presentation
2

### Contribution
2
