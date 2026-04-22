# Controllable Sequence Editing for Biological and Clinical Trajectories

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Conditional generation models for longitudinal sequences can produce new or modified trajectories given a conditioning input. However, they often lack control over when the condition should take effect (timing) and which variables it should influence (scope). Most methods either operate only on univariate sequences or assume that the condition alters all variables and time steps. In scientific and clinical settings, interventions instead begin at a specific moment, such as the time of drug administration or surgery, and influence only a subset of measurements while the rest of the trajectory remains unchanged. CLEF learns temporal concepts that encode how and when a condition alters future sequence evolution. These concepts allow CLEF to apply targeted edits to the affected time steps and variables while preserving the rest of the sequence. We evaluate CLEF on 8 datasets spanning cellular reprogramming, patient health, and sales, comparing against 9 state-of-the-art baselines. CLEF improves immediate sequence editing accuracy by 16.28% (MAE) on average against their non-CLEF counterparts. Unlike prior models, CLEF enables one-step conditional generation at arbitrary future times, outperforming their non-CLEF counterparts in delayed sequence editing by 26.73% (MAE) on average. We test CLEF under counterfactual inference assumptions and show up to 62.84% (MAE) improvement on zero-shot conditional generation of counterfactual trajectories. In a case study of patients with type 1 diabetes mellitus, CLEF identifies clinical interventions that generate realistic counterfactual trajectories shifted toward healthier outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies controllable time‑series editing and proposes CLEF, a framework for conditional generation of time‑series given an intervention time and value. The model encodes past history, time difference, and the intervention into a temporal context embedding that modulates the extent of output change. The method is evaluated on conditional and counterfactual generation tasks across eight datasets from multiple domains, including cellular dynamics, healthcare, and sales.

### Strengths
* The work addresses counterfactual outcome estimation, an important problem in computational biology and healthcare.
* The introduction motivates the problem well, and Fig. 1 is informative. The writing is generally clear. Figures are descriptive and polished, although they are currently too small to read without zooming, so they should be enlarged.
* The model design is simple and clean.
* The method is evaluated on a substantial number of datasets from three domains; however, the experimental setup is difficult to follow from the main text at times (see Weaknesses).

### Weaknesses
The paper exhibits several weaknesses: the proposed method shows unexpectedly poor performance relative to a simple linear baseline, the experimental setup lacks clarity in several places, and the results as presented in the Experiments section are not fully aligned with the abstract. I am willing to reconsider my score if these concerns are addressed convincingly.

**(i) Performance relative to simple baselines**

This is the main weakness. In R1 and R2, performance is unexpectedly poor relative to a simple linear baseline: the linear baseline appears better on average than non‑CLEF models and comparable to CLEF models. This raises concerns about applicability and practicality. If these models do not outperform simply using the previous time point, the value of training and deploying them is unclear.

**(ii) Ambiguity in experimental setup and model variants**

It is unclear how CLEF and non‑CLEF models differ. Lines 297–298 state that “CLEF and non‑CLEF differ only in the components needed to learn temporal concepts,” which is ambiguous, and Appendix D is difficult to follow. Please clarify whether non‑CLEF models (e.g., Transformer, xLSTM) are implemented exactly as in their original papers or adapted in some way. It is surprising that these models underperform the simple baseline on average in Fig. 4.

**(iii) Abstract–experiment misalignment and selective reporting**

The abstract’s presentation of results does not accurately reflect the evidence shown in the Experiments section. It is unclear where the abstract’s claims are reported in the main results. Reporting “accuracy gain up to” an arbitrary or worst baseline is not a fair summary. In Fig. 4, the simple baseline appears on par with the proposed complex models. Examples:
* Lines 21–22: “… immediate sequence editing accuracy by up to 36.74% (MAE).”
* Lines 22–23: “… delayed sequence editing by up to 65.71% (MAE).”

**(iv) Task definition: R5 vs. R4**

The definition of R5 (zero‑shot conditional generation of counterfactual trajectories) is not clear. The distinction from counterfactual outcome estimation in R4 should be specified.

**(v) Intervention mechanics in R6**

For R6, it is unclear how interventions on temporal concepts are performed to decrease or increase glucose levels. As a sanity check, can authors construct a simple baseline also for this task?

### Questions
* Question: Have the authors considered/experimented with other concept encoder and concept decoder architectures?
* Suggestion: Lack of qualitative trajectory examples. The paper (and related counterfactual outcome estimation work [Bica+20; Seedat+22; Melnychuk+22]) does not provide example estimation trajectories. Including sample time‑series from each dataset with baseline and proposed model estimates would illustrate data characteristics and model behavior.
* Suggestion: The paragraph about "controllable text generation" in the Introduction (starting ln. 59) does not seem directly relevant to the problem at hand. The authors could reconsider removing it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of sequence editing and forecasting. The paper proposes a new method to achieve this goal. The proposed method is based on encoder decoder solution. Given a condition, the proposed method changes the sequence to meet the given condition. The paper splits the problem into two subproblems: immediate sequence editing and delayed sequence editing. Experiments are conducted on variety of biomedical and financial datasets.

### Strengths
- The paper addresses an interesting problem with significant practical applications.
- The paper is well-written and the proposed method is clearly motivated.
- Experimental section is thorough and convincing.

### Weaknesses
- The related work section could be expanded. Certain areas, such as reinforcement learning, are closely related to this problem but are not discussed in the current version.
- The paper could also benefit from including more baselines by adapting other existing and closely related methods to the experimental setup. For example, some methods proposed for biological, protein, and DNA sequence editing could be applied to the problem studied in this paper, even though they were originally designed to address slightly different tasks in other biological domains. However, this is not necessary for this paper whereas it can improve its strengths'.

### Questions
- After reading the paper, it is not entirely clear how the proposed solution and results differ between immediate sequence editing and delayed sequence editing. Could you elaborate on this? My understanding is that immediate sequence editing can be viewed as a special case of delayed sequence editing.
- There are existing approaches originally proposed for biological sequence design that could be applied to this problem. For example, Bayesian optimization such as the one introduced in “Accelerating Bayesian optimization for biological sequence design with denoising
autoencoders” (Stanton et al., 2022) could be considered. Another possible direction is to model the problem using reinforcement learning, where $x$ represents the state and $s$ denotes the value of that state. Note that the term state in this paper may differ from its usage in reinforcement learning. Furthermore, GFlowNets have been widely applied in biological sequence design and specifically in sequence editing as introduced by “GFlowNet-Assisted Biological Sequence Editing” (Ghari et al., 2024) and multi-objective generation as introduced by " Multi-objective GflowNets" (Jain et al., 2023). Although direct comparison with these methods is not necessary for this paper, I recommend including a discussion of such approaches to motivate future research.

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
2

### Summary
The authors propose CLEF, a new framework for Controllable Sequence Editing, designed to modify longitudinal sequences (e.g., patient health trajectories, cellular development) based on a specified condition. [cite_start]The key contributions are twofold: 1) a new problem formulation, "delayed sequence editing," which involves applying a condition at a future time and generating the output in a single step, and 2) the CLEF model, which learns "temporal concepts" to represent the change from the last observed time to the future time.

The core mechanism of CLEF is to predict a future state as an element-wise product of the last observed state and the learned temporal concept. This concept $c$ is generated by a "concept encoder" that takes the encoded history ($h_x$), the encoded condition ($h_s$), and the time delta ($\Delta_{t_i, t_j}$) as input.

### Strengths
- The formalization of "delayed sequence editing"  as a one-step generation task is interesting contribution. This distinguishes the problem from standard auto-regressive forecasting
- Good empirical results CLEF-based models demonstrate consistent and often large improvements over their non-CLEF counterparts across all primary tasks: immediate editing , delayed editing , and zero-shot counterfactual generation

### Weaknesses
- Does the model assume that the entire, complex dynamic evolution of each specific variable (e.g., a single lab test) over an arbitrary time $\Delta t$ can be modeled as a single multiplicative scaling factor $c$ for that variable? This still seems dynamically and biologically implausible, as it ignores the coupled, differential nature of these systems.
- Btw if that is the case, then if any single variable in the last observed state, $x_{k, t_i}$, is 0, then the predicted value for that variable, $\hat{x}_{k, t_j}^s$, must also be 0 (since $c_k \odot 0 = 0$). This is a severe limitation for any data that is sparse or has variables that can cross zero.

### Questions
- See weaknesses
- Also I'd argue that c is more of a ratio than a rate of change

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
This paper proposes CLEF, a novel framework for controllable sequence editing, designed to address the lack of granular control over timing and scope in the conditional generation of longitudinal data. The core idea is to learn "temporal concepts," which represent the rate of change for each variable, and apply them via a simple multiplicative decoder to generate future states in a single, non-autoregressive step. This approach is designed to handle both immediate and delayed edits—modifying a sequence at a distant future time point without generating intermediate steps. The authors extend this framework to counterfactual outcome estimation, arguing that the architecture provides implicit balancing that outperforms state-of-the-art methods, even without explicit balancing losses. The method's effectiveness is demonstrated through an extensive empirical evaluation on 8 datasets spanning biology, clinical medicine, and finance, where CLEF-augmented models show significant improvements over numerous baselines, particularly in the novel delayed editing task and in zero-shot counterfactual generation. A case study on generating "healthier" trajectories for diabetic patients further highlights the model's practical utility and interpretability.

### Strengths
*   **Clear, useful problem framing for controllable sequence editing.** In my opinion, the split between immediate vs. delayed editing—and the choice to define delayed editing as a single-step, non-autoregressive jump—cleanly targets a real pain point (compounding error) in scientific/clinical forecasting. This makes the task definition itself a contribution.

*   **Broad, careful empirical evaluation with new benchmarks.** I think the scope (8 datasets, 4 contributed benchmarks, 9 baselines, plus generalization and zero-shot counterfactual tests) is a major strength. It shows the idea isn’t brittle and gives the community reusable testbeds. The writing is also notably clear about how this fits vs. related work.

*   **Simple, modular architecture that travels across encoders.** In my view, the “temporal concept” (multiplicative edit) is easy to implement, computationally light, and integrates with varied sequence encoders. That portability lowers the barrier to adoption and facilitates ablations/reuse.

*   **Actionable interpretability via direct concept edits.** I like that domain experts can intervene on the learned concept vector and see resulting counterfactuals; the Type 1 Diabetes case study makes this tangible for in-silico hypothesis exploration. It’s not interpretability in the semantic-concept sense, but it is a practical, controllable handle on predictions.

*   **Good engineering trade-offs for stability vs. flexibility.** In my opinion, the “one-step to $t_j$” design is a pragmatic choice: it sacrifices full trajectory modeling to reduce error accumulation and make edits predictable. Given many operational needs are “endpoint-centric,” that’s an appealing trade-off.

### Weaknesses
*   **“Implicit balancing” feels asserted more than demonstrated.** In my opinion, the paper leans on architectural intuition and empirical accuracy to suggest that the learned representation is balanced, but it doesn’t directly test balance. It might read more cautiously if the claim were framed as “consistent with improved balance,” and, if feasible, paired with light diagnostics (e.g., predicting treatment from the learned representation and reporting an IPM-style distance such as MMD/HSIC between treated vs. control in the rep space).

*   **Decoder family may be too restrictive for coupled or constrained systems.** I read the diagonal, multiplicative generator $\hat{x} = c \odot x$ as elegant but narrow. In my view, this per-variable scaling risks missing cross-variable couplings, conservation/compositional constraints, or saturation effects. A short limitations note—and, if bandwidth allows, a small ablation with a lightly coupled decoder $(I+W)(c \odot x)$ where $W$ is sparse or low-rank—could clarify where the current choice shines and where it struggles.

*   **One-step “jump” to $t_j$ trades off path information for stability.** Personally, I see the single-step design as a smart way to avoid compounding autoregressive error, but it does mean path-dependent phenomena (transients, intermediate interventions, accumulation) aren’t modeled explicitly. A brief scope statement that CLEF targets endpoint editing—and a short-horizon chaining stress test (3–5 steps) to show how rollouts behave—would, in my opinion, set expectations more cleanly.

*   **Evaluation blends predictive strength with causal validity.** My sense is that strong MAE/RMSE/AUC is being read as support for the causal story, but predictive accuracy alone doesn’t confirm deconfounding. It may help to report a couple of lightweight causal diagnostics alongside accuracy—(i) treatment predictability from the learned reps, (ii) a balance/divergence metric, and (iii) a small sensitivity analysis—to separate “good predictions” from “good balancing.”

### Questions
* On the "Implicit Balancing" Claim: The paper's causal claims rely on the idea of "implicit balancing," which is currently supported by downstream predictive accuracy. To substantiate this claim more directly, could you provide a more targeted diagnostic for balance? For instance, could you report on either (a) the predictability of treatment assignment from the learned representations (where a lower AUC would indicate better balance) or (b) an IPM-style distance like MMD between the representation distributions for treated versus control groups?  

* On the Limitations of the Multiplicative Decoder: The diagonal multiplicative decoder, x^=c⊙x, is elegant but seems to impose a strong linearity assumption. Could you please add a discussion on its potential limitations in systems with known non-linear couplings, compositional constraints (e.g., variables that must sum to a constant), or saturation effects? Clarifying the boundaries of this design choice would help readers understand its ideal application scope.  

* On the Scope of Endpoint vs. Trajectory Generation: The one-step forecast is a key design choice to avoid compounding error. Could you clarify the model's intended scope (i.e., is it primarily for endpoint editing rather than full trajectory simulation)? To help illustrate this, could you include a short-horizon chaining test (e.g., 3–5 steps) to show how error accumulates when the model is applied autoregressively compared to the one-step approach?

### Soundness
3

### Presentation
3

### Contribution
3
