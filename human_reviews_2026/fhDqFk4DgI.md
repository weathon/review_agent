# Micro-Macro Coupled Koopman Modeling on Graph for Traffic Flow Prediction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Traffic systems are inherently multi-scale: microscopic vehicle interactions and macroscopic flow co-evolve nonlinearly. Microscopic models capture local interactions but miss flow evolution; macroscopic models enforce aggregated consistency yet overlook stochastic vehicle-level dynamics. We propose Micro–Macro Coupled Koopman Modeling (MMCKM), which lifts the coupled dynamics to a high-dimensional linear observation space for a unified linear-operator representation. Unlike grid-based discretizations, MMCKM adopts a vehicle-centric dynamic graph that preserves microscopic perturbations while respecting macroscopic conservation laws by discretizing   PDEs onto this graph. At the micro scale, scenario-adaptive Koopman evolvers selected by an Intent Discriminator are designed to model vehicle dynamics. A Koopman control module explicitly formulate how flow state influences individual vehicles, yielding bidirectional couplings. To our knowledge, this is the first work to jointly model vehicle trajectories and traffic flow density using a unified Koopman framework without requiring   historical trajectories. The proposed MMCKM is validated for trajectory prediction on NGSIM and HighD. While MMCKM uses only real-time measurement, it achieves comparable or even higher accuracy than history-dependent baselines.  We further analyze the effect of the operator interval and provide ablations to show the improvement by intent inference, macro-to-micro control, and diffusion. Code and implementation details are included to facilitate reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes **MMCKM (Micro–Macro Coupled Koopman Modeling)**, a unified framework for traffic flow prediction that links microscopic vehicle trajectories and macroscopic flow dynamics. Both scales are lifted into **Koopman spaces**, where their evolution is governed by linear operators aligned through **spectral and physical constraints**.

At the macro level, advection–diffusion PDEs are discretized on a **vehicle-centric dynamic graph**, preserving physical properties such as antisymmetric advection and positive semi-definite diffusion. At the micro level, multiple Koopman operators represent different driving intents, chosen via an **Intent Discriminator**, and macro information is injected through a **Koopman control mechanism**.

The model predicts from a single snapshot (history-free) and achieves strong results on **NGSIM** and **HighD** datasets, with ablation studies validating the importance of macro–micro coupling, intent modeling, and diffusion components.

### Strengths
1. **Innovative modeling perspective** — integrates micro-level trajectories and macro-level PDE dynamics within a unified Koopman framework.
2. **Physically consistent design** — antisymmetric advection, PSD diffusion, and spectral constraints ensure stability and interpretability.
3. **Modular and efficient structure** — multiple small Koopman operators (for different intents) improve computational efficiency and interpretability.
4. **Strong practical motivation** — “history-free” prediction is appealing for real-world deployment with intermittent data.
5. **Clear ablation and physical insight** — experiments reveal the distinct contributions of each model component.

### Weaknesses
1. **KDE-based macro labels** are not ground-truth densities; performance depends on kernel parameters. Sensitivity analysis is missing.
2. **Computational scalability** not analyzed — unclear runtime behavior as vehicle count grows. And there is a lack of specific runtime comparison with other baseline models
3. **Intent Discriminator details** (labeling rules, accuracy, and noise sensitivity) are insufficiently described.

### Questions
1. How sensitive are the macro-level results to KDE bandwidth? Please provide results under different bandwidths (e.g., 10 m, 25 m, 50 m).
2. Could you describe the process of intent labeling in more detail? What thresholds or criteria were used, and how accurate is the Intent Discriminator?
3. What is the computational complexity and runtime per prediction step? How does it scale with the number of vehicles?
4. Could the model generalize to more complex network structures (urban intersections), and what modifications might be required?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a Micro–Macro Coupled Koopman Model (MM-Koop) for unified modeling and prediction of multi-agent traffic flow dynamics.
By embedding both micro-level nonlinear interactions and macro-level fluid statistics into a shared Koopman space, the method enables a cross-scale linearized representation of complex traffic systems.

The framework consists of:

1. A micro-level encoder mapping individual agents’ states to Koopman-observable embeddings;

2. A macro-level aggregator capturing density, velocity, and energy fields;

3. A cross-scale coupling operator that models bidirectional feedback between individual and collective dynamics.

Experiments on the NGSIM (highway) and ETH/UCY (pedestrian) datasets demonstrate superior multi-step prediction accuracy and interpretability compared to LSTM, GraphKoopman, and other baselines.

### Strengths
- Presents a \textbf{novel cross-scale Koopman operator framework} that linearly couples micro-level agent dynamics and macro-level flow statistics.  
- Provides a \textbf{mathematically rigorous formulation} with well-defined block operator structure and spectral regularization for stability.  
- Offers strong \textbf{interpretability and physical insight}, explaining emergent traffic phenomena (e.g., congestion waves, flow bifurcation) via Koopman spectral modes.  
- Demonstrates \textbf{solid predictive performance} compared to LSTM and GraphKoopman baselines.  
- The theoretical structure is \textbf{generalizable} to broader multi-agent dynamical systems such as robotic swarms or fluid-based networks.

### Weaknesses
- Experiments rely primarily on controlled simulations; no real-world deployment or sensor-based evaluation is shown.  
- The paper’s \textbf{learning component is limited}: Koopman matrices are estimated rather than learned via gradient-based training, reducing its alignment with mainstream ML innovation.  
- Lacks \textbf{comprehensive ablation studies} to assess the contribution of each coupling term ($K_m, K_M, K_{mh}, K_{hm}$).  
- The related work section omits recent Koopman-based learning frameworks (e.g., NeuralEDMD, DeepKoopman++, Contrastive Koopman Learning).  
- While theoretically strong, the overall \textbf{ML novelty and scalability} remain somewhat constrained.

### Questions
1. Is the Koopman operator learned using neural networks or estimated through regression? How is numerical stability ensured?  
2. Does the cross-scale coupling introduce instability into the operator spectrum?  
3. Could the model benefit from graph-based message passing to represent heterogeneous agent interactions?  
4. Would the framework generalize to domains without explicit macro-aggregators, e.g., aerial swarms?  
5. How do the authors interpret the sparsity of Koopman spectra under high-density conditions? Is it an artifact of strong linearization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Micro–Macro Coupled Koopman Modeling (MMCKM), a unified Koopman-based framework that jointly models microscopic vehicle interactions and macroscopic traffic flow evolution. 

Unlike conventional microscopic or macroscopic approaches that treat these scales separately, MMCKM lifts the coupled dynamics into a shared high-dimensional linear observation space. On the macroscopic side, the method discretizes advection–diffusion traffic flow PDEs onto a vehicle-centric dynamic graph, preserving physical flow consistency without grid constraints. On the microscopic side, scenario-adaptive Koopman evolvers, guided by an Intent Discriminator, model diverse vehicle behaviors, while a Koopman control module captures the bidirectional influence between flow states and vehicle dynamics. MMCKM adopts a vehicle-centric dynamic graph that preserves microscopic perturbations while respecting macroscopic conservation laws by discretizing PDEs onto this graph. Evaluated on NGSIM and HighD datasets, MMCKM achieves comparable trajectory prediction accuracy to state-of-the-art history-dependent models despite using only real-time inputs. Ablation studies confirm the contributions of intent inference, macro-to-micro control, and operator interval design. The authors also outline future work on interpretable interaction analysis and extending the framework to urban traffic scenarios.

### Strengths
- The introduced method seems to be very original;
- The article is well-written; it has a good structure, and I like the fact that it includes an ablation study and a reproducibility statement, and the use of LLMs statement
- The results of the experiments are good
- The method has the potential to be significant, taking into account that it does not require historical trajectories

### Weaknesses
- The method still does not outperform the best methods utilizing historical data, but the fact that the results are comparable is noticeable
- There are some typos and minor writing issues, e.g., due to the lack of space,: 
  - l. 49: "frameworksHuang" -> "frameworks Huang" 
  - l. 50: "limitsCristiani" -> "limits Cristiani"
  - l. 128: "applicationsBrunton" -> "applications Brunton"
  - l. 133: "R^nLusch" -> "R^n Lusch"
  - l. 144: "spaceProctor" -> "space Proctor"
  - l. 370: "3 8 s" -> "3-8 seconds"

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Micro-Macro Coupled Koopman Modeling (MMCKM), a framework for unified traffic flow prediction that combines microscopic vehicle interactions and macroscopic flow dynamics by using Koopman operator theory. This combines microscopic dynamics, where every single vehicle is simulated, to macroscopic dynamics where the overall traffic flow is evolved. The method predicts the future state from the current time step, without relying on historical trajectories, by leveraging a high-dimensional linear observation space, and performs better than history-dependent baselines on experiments on NGSIM and HighD datasets.

### Strengths
The paper tackles an important problem, to bridge the gap between microscopic and macroscopic dynamics in traffic modeling, both of which have advantages and drawbacks. For instance the microscopic scale allows for simulating individual vehicle dynamics while the macroscopic scale enforces high-level flow dynamics such as conservation laws. The authors use an innovative Koopman-based unified model to jointly learn both scales, and leveraging a graph discretization of the PDE, as opposed to standard grid discretization.  Empirical validation on two real-world datasets (NGSIM & HighD) support the approach, achieving competitive or superior results without relying on historical trajectories. The ablation studies validate the impact of each module. The paper is well-written overall and well-motivated.

### Weaknesses
- The paper is quite technical and math-heavy, which can make it hard for readers not familiar with the area to follow the core concepts. Some sections such as 4.1/4.2 could benefit from more intuitive explanations or visual illustrations.
- Experiments are limited to highway datasets with estimated (not sensor) density labels, so it’s unclear how well the method would generalize.
- There are no direct comparisons with other hybrid or Koopman-based approaches, so it makes it harder to tell how much of the improvement comes from the coupling design itself.

### Questions
- Is the model applicable to urban or more complex scenarios than structured highways?
- Can the authors comment on the runtime efficiency of the method? Is it able to run at real-time?

### Soundness
3

### Presentation
3

### Contribution
4
