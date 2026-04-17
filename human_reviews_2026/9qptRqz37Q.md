# Flow Matching for Robust Simulation-Based Inference under Model Misspecification

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Simulation-based inference (SBI) is transforming experimental sciences by enabling parameter estimation in complex non-linear models from simulated data. A persistent challenge, however, is model misspecification: simulators are only approximations of reality, and mismatches between simulated and real data can yield biased or overconfident posteriors. We address this issue by introducing Flow Matching Corrected Posterior Estimation (FMCPE), a framework that leverages the flow matching paradigm to refine simulation-trained posterior estimators using a small set of real calibration samples. Our approach proceeds in two stages: first, a posterior approximator is trained on abundant simulated data; second, flow matching transports its predictions toward the true posterior supported by real observations, without requiring explicit knowledge of the misspecification. This design enables FMCPE to combine the scalability of SBI with robustness to distributional shift. Across synthetic benchmarks and real-world datasets, we show that our proposal consistently mitigates the effects of misspecification, delivering improved inference accuracy and uncertainty calibration compared to standard SBI baselines, while remaining computationally efficient.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to solve the model misspecification problem under simulation-based inference (SBI) with flow matching. It proposes a two-stage flow-matching scheme: (i) an observation-space flow $T_X$ that transports Gaussian perturbations around a real observation y toward a proxy distribution q(x∣y), and (ii) a parameter-space flow $T_\\theta$ that maps samples from a source distribution built from an amortized posterior into an approximation of the true posterior. Across several toy and real-world tasks, the method appears robust in low-calibration-data regimes and improves downstream metrics vs. amortized baselines.

### Strengths
* The proposed method is quite interesting. The construction of $\\pi(\\theta∣y)$ via $T_X$ reduces the transport gap and stabilizes training in small-calibration settings.
* To my understanding, it works as a post-hoc calibration layer on top of any amortized posterior estimator and does not rely on restrictive conditional-independence assumptions like some of the previous methods.
* The empirical result looks solid, which includes multiple tasks, and shows consistent gains as calibration size grows.
* The presentation is clear, also the source code is provided.

### Weaknesses
* Lack of robust SBI baselines: Although the paper argues RoPE is “not directly comparable,” I believe it is still a meaningful baseline because the setups are closely related: both methods correct amortized posteriors under simulator–real mismatch using a small calibration set and a learned transport/alignment step. The key difference is protocol (RoPE is transductive—using the test set as a whole—whereas your method is inductive—per-sample), but that does not preclude comparison; it simply requires transparent labeling. A practical compromise is to report RoPE under its native transductive protocol (clearly marked as such) alongside your inductive results. This lets readers gauge the attainable performance when the full test set is accessible, while still appreciating your method’s per-sample advantages. In addition, adding other previous robust SBI baselines like what RoPE did would further strengthen the paper—e.g., NNPE, NPE-RS, or J-NPE that jointly trains on simulated and calibration pairs. 
* Missing ablations: The paper motivates each module but does not quantify its contribution. It would be great if the authors could add some ablation studies (some may not make sense)
  * X flow only without theta flow, then you could feed $\\tilde{x}$ into a standard NPE.
  * theta flow only. You could feed the pre-trained NPE directly with y, obtain $\\hat{\\theta}$, and use your theta flow to map them to the target.
  * Gaussian start. sample theta_0 from a Gaussian, learn only theta flow with few calibration pairs.
  * Sequential vs joint training. Is joint training necessary? What if we train them separately?
* It would be great to add a few more metrics, such as MMD, and also coverage/ECP for posterior calibration.

### Questions
* jC2ST scores below 0.5 in wind tunnel example is a bit unusual to me given N_test = 5000. Could you give more explanation about it?
* A Gaussian $N(y, \\sigma^2I)$ is used to sample $x_0$, I think it needs some clearer justification, and how sensitive are results to $\\sigma$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies multifidelity simulator-based inference, where the goal is to infer the posterior distribution of model parameters $\theta$ given the low-fidelity data $(x,\theta) \sim p(x,\theta)$ and the limited high-fidelity data $(y,\theta) \sim p(y,\theta)$ (termed as calibration data). The paper proposes training two flows (i) a flow from the base distribution to $q(x|y)$ using the calibration data, and (ii) a flow from the misspecified posterior of $\theta$ to the correctly specified posterios of $\theta$, using joint flow-matching. The method shows good performance compared to some exististing baselines  in synthetic and real-world tasks,  in terms of quantative and qualitive metrics.

### Strengths
The paper is well-written, easy to follow, and the proposed method is cleary articuled in Section 3, where I could not find any tehcnical flaws or ad-hoc justifications. While there are some limitations in the experimental section (see discussion in Weaknesses), the paper demontstrate good and relatively robust emprical performance in synthetic and real-world tasks. When evaluated in the context of multifidelity SBI, the proposed method constitutes a contribution in terms of novel ideas and empirical performance. 

It is great that the experimental section contains visualizations of the learned posteriors for both the baselines and the proposed method. This makes it easier to qualitatively judge performance and shows that the proposed method produces more accurate and robust posteriors than the baselines.

### Weaknesses
The main weakness is the insufficiently clear problem formulation, which results in a somewhat exaggerated scope. I am quite convinced that the paper does not tackle SBI under model misspecification but rather multifidelity SBI, where in the latter there is some information about the true model as we have access to the samples $(y,\theta)$. I try to elaborate this weaknesses below:

Lines 78-82: “…high-fidelity data—accurate representations of the phenomenon obtained either from costly high-quality simulations or from ground-truth observations.”
Isn’t the application scope quite limited in the setting where the ground-truth observations are samples from the joint $p(\theta,y)$ rather than from the marginal $p(y)$? Typically, $\theta$ represent latent variables that only exists within the model. For that reason, the considered problem in the paper is not a classical SBI problem but rather a multi-fidelity SBI problem, the term that already appears in the title of (Krouglova et al., 2025). Of course, this same critique applies to (Wehenkel et al., 2025), but I think it is important to be explicit about this to avoid misunderstanding that the proposed method is a general SBI method (as e.g. the title currently hints). This fact is also reflected in the experiments where only real-world tasks are from (Gamella et al., 2025).

Given this, I suggest:
1.	The title should replace the term “SIMULATION-BASED INFERENCE” to “MULTIFIDELITY SIMULATION-BASED INFERENCE” to honestly capture the scope of the paper. 
2.	The accommodiate the first sentences in Section 3: “We consider the general problem of sampling from a posterior distribution…” to something where the paper early states that ‘”…we have access to samples $(y,\theta)$”.

Next, I discuss some minor weaknesses.

Evaluation metrics (Section 4.2)

This section lacks justification and discussion for the chosen evaluation metrics. For example, if the task is to infer the posterior of $\theta$, why the paper does not consider the distance (say Wasserstein) between the inferred  posterior and the ground-truth $p(\theta)$. This should be feasible at least in the task “Gaussian”, where the ground-truth $p(\theta)$ is known. Further, why the paper considers MSE of $\theta$ samples, while we are dealing with distributions, isn’t some metric in the space of probabilitdy distributions (Wasserstein etc.) more appropriate?

Baselines

“….we do not include RoPE (Wehenkel et al., 2025) because it is a method that requires access to the full test set at inference time and is not directly comparable to our approach, as explained in Section 2.” The conditional independence assumption imposed by (Wehenkel et al., 2025) it maybe a limitation of their work but should not be a reason to exclude RoPE from the experiments. Further, can you elaborate why requiring access to the full test set at the inference time is a critical limitation? Concernign the sequential setup, I think that all the experiments in the paper are non-sequential, right?

Experimental results

In real-world tasks (Wind tunnel, Light tunnel), and when considering the metric that focuses on the posterior samples of $\theta$ (i.e. MSE), the all the methods looks achieving comparable performance when “Calibration size” is not very small (Figure 4). It would be good to discuss this in the main paper

### Questions
Can you list some real-world problems that fits to the problem considered in the paper? I tried to quickly check (Wehenkel et al, 2025), but so far could only confirm that it matches with (Gamella et al., 2025).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method to address model misspecification in SBI via flow matching trained on a calibration dataset with known ground truths. I am short on time due to the semester start. Apologies if my reviews are a bit short. I am happy to engage in reviewer discussion should be concerns not be clear. And I am happy to consider increasing my score should the authors provide convincing responses to my concerns.

### Strengths
- The paper addresses the important problem of model misspecification in SBI/ABI.
- The method appears theoretical sound and overall sensible to me. 
- I think the paper is well written. But this assessment may be different for readers less familiar with the field of SBI/ABI.

### Weaknesses
- Assuming the existance of a calibration dataset with known ground truths is a strong assumption. I know several papers make this assumption. But still, for many application areas we just never have the ground truth parameters available ruling out the proposed method for their analyses.
- The tested benchmarks are all very low dimensional in parameter space. How does the size of the required calibration dataset scale with the parameter dimensionaliry of the problem? I would have expected this scalability to be analyzed at the very least in toy examples (Gaussian).
- Some relevant related work was not cited or considered as benchmarks (at least https://arxiv.org/abs/2501.13483 and https://arxiv.org/abs/2502.04949). Perhaps the references therein provide even more relevant research?
- A comparison of both training and inference speeds between the approaches is lacking to my reading. At least I haven't found it. Providing information in that regard would be important too I think.

### Questions
- Just to double check: The approach is fully amortized right? In the sense that, at inference time, only forward passes through the networks are required without any (re-)training?
- What is the target of inference? The true posterior under the misspecified model or the true posterior under the correctly specified model? See also https://arxiv.org/abs/2502.04949 for a taxonomy of interence targets in the face of misspecification.
- How strongly is the idea of the method rooted in flow matching? I.e., how easy would it be to replace the flow matching networks with standard diffusion models or even coupling flows?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes **FMCPE**, a two-stage flow-matching correction for SBI under misspecification: train an NPE on simulations, then learn two conditional flows—one in data space $T_X$​ and one in parameter space $T_\Theta$—to transport toward a “corrected” posterior using scarce calibration pairs. The authors consider a *joint training* objective of corrected posterior and source distribution. Practical (Algorithm 2) each training tuple requires a **simulator call** ​ and an **ODE solve** to estimate $T_x$ jointly with a correction map $T_\Theta$ the adjust a pretrained NPE (on low-fidelity or misspecified data) (Algs. 1–2). The paper then empirically validates the proposed approach on several misspecification-tasks from previous work (Wehenkel et.al. 2025) and demonstate empirical advantages against a naive NPE baseline (only trained on calibration set) and a MF-NPE (pretrained on simulations, then fine tune on calibration).

### Strengths
- **Novelty**: To my knowledge this learning-based approach is novel and address limitations of previous approaches  which relied on pre-specified form of misspecification or conditional independence assumptions.

### Weaknesses
1. **Conceptual positioning (misspecification vs. multifidelity/multilevel):**  
    The method and experiments read sometimesmore like a **multifidelity/multilevel** correction pipeline (low-fidelity simulator + scarce high-fidelity pairs + learned transport) than a principled treatment of misspecification. Additionally the comparisons target **NPE** and **MFNPE**; there is **no direct baseline targeted at misspecification** (e.g., methods that explicitly correct simulator bias), and RoPE is excluded from evaluation. This creates an **identity gap** between the stated aim (misspecification) and what is empirically validated (multifidelity fine-tuning). The authors consider the same tasks as in Wehenkel et al. 2025, but claim that its not "comparable" because it requires the full test set at inference time. While I agree that this is a limitation that the proposed method addresses, this **does not** prevent a comparison! 

2. **Computational cost & performance:**  
    Every training tuple triggers **both** a **simulation** **and** an **ODE integration**. This pipeline seems **extremely costly** compared to single-flow or single-stage amortized approaches and scales poorly with calibration size. Analysis or discussion of this is completely missing in the manuscript. Furthermore it should come with strong performance gains, specifically it **must** be better than methods just using e.g. the calibration dataset (similar to what the authors demonstrate with NPE). Recent work [1] has show that Tabular Foundation models can do SBI with improved robustness and simulation-efficiency. A good way to demonstrate the efficacy would be to include a comparison against such methods that e.g. only "train" (evaluate) on the calibration datasets (or maybe mixed with cheap simulations similar to MF-NPE). 

[1] Vetter, Julius, et al. "Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models." _arXiv preprint arXiv:2504.17660_ (2025).

3. **Weak metrics and limited empirical coverage:**  
        The paper evaluates via **joint** discrepancy metrics (jC2ST, Wasserstein on the joint), rather than **posterior-level** diagnostics that directly assess the **misspecified posterior** when that posterior is tractable or obtainable via MCMC (which is true for at least the Gaussian task). This makes the evidence **indirect** and weaker than necessary. For example this are effectively very similar to *global coverage* metrics which are only *necessary* but **not** sufficient properties that can be fooled by conservative posterior approximations (i.e. any convex combination of the posterior and prior). The empirical evidence in general is additionally limited to 4 tasks. While the authors claim that the method does not require the conditional independence assumption required for rope (and I agree conceptually), its unclear if the presented tasks demonstrate this.

### Questions
- **Multi-fidelity vs. misspecification.** These are related but to a some degree also quite different concepts, and the distinction should be communicated clearly by explicitly defining both terms in the text. In misspecification (e.g., sim-to-real gaps), calibration datasets are typically rare or impossible to obtain; by contrast, in multi-fidelity/multilevel settings they exist by definition. Moreover, simulators are generally misspecified to some degree and need not be “cheap” to run, whereas in multi-fidelity setups usually include lower-cost simulators that would justify the additional simulation burden of the proposed approach (but this is not necessarily given in misspecified settings). This should be clearly communicated in the manuscript.
- Can you add **model-misspecification baselines** i.e. ROPE or others (not just multi-fidelity ones) and report **posterior-level** metrics where the ground-truth misspecified posterior is tractable? 
- Another recent multi-fidelity approach needs to be discussed [1].
- What is the **wall-clock/NFE** budget per calibration size, broken down into simulator calls vs. **odeint** solves​ and flow training steps?


[1] Hikida, Yuga, et al. "Multilevel neural simulation-based inference." _arXiv preprint arXiv:2506.06087_ (2025).

### Soundness
3

### Presentation
1

### Contribution
2
