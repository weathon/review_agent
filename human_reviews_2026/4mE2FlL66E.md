# Sharpness-Aware Minimization in Logit Space Efficiently Enhances Direct Preference Optimization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Direct Preference Optimization (DPO) has emerged as a popular algorithm for aligning pretrained large language models with human preferences, owing to its simplicity and training stability. However, DPO suffers from the recently identified *squeezing effect* (also known as *likelihood displacement*), where the probability of preferred responses decreases unintentionally during training. To understand and mitigate this phenomenon, we develop a theoretical framework that models the coordinate-wise dynamics in logit space. Our analysis reveals that negative-gradient updates cause residuals to expand rapidly along high-curvature directions, which underlies the squeezing effect, whereas Sharpness-Aware Minimization (SAM) can suppress this behavior through its curvature-regularization effect. Building on this insight, we investigate *logits-SAM*, a computationally efficient variant that perturbs only the output layer with negligible overhead. Extensive experiments on Pythia-2.8B, Mistral-7B, and Gemma-2B-IT across multiple datasets and benchmarks demonstrate that logits-SAM consistently improves the effectiveness of DPO and integrates seamlessly with other DPO variants. Code is available at <https://github.com/RitianLuo/logits-sam-dpo>.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the "squeezing effect" in DPO, where preferred response probabilities unintentionally decrease. Through a theoretical analysis of logit space dynamics, the authors identify rapid expansion along high-curvature directions under standard optimization as the cause. They propose that Sharpness-Aware Minimization (SAM) mitigates this via curvature regularization.

To apply this efficiently, they advocate for logits-SAM, perturbing only the final layer. Empirical results on Pythia-2.8B and Mistral-7B across various alignment tasks show that logits-SAM consistently improves DPO performance with negligible computational overhead.

### Strengths
1. The paper tackles the important and recently identified "squeezing effect" issue in DPO, which is a practical limitation of the algorithm.
2. It provides a novel theoretical framework connecting parameter-space and logit-space dynamics, specifically analyzing the role of curvature in the squeezing effect under GD and how SAM counteracts it. The insight about matching the sign of the perturbation radius $\rho$ with the effective learning rate $\eta$ is particularly interesting.
3. Based on their theoretical insights into curvature and the squeezing effect, the paper identifies the computationally efficient logits-SAM variant as a suitable mitigation strategy and successfully applies it to the DPO context, demonstrating its practical value for improving alignment.

### Weaknesses
1. The core theoretical analysis simplifies DPO by modeling the dynamics associated with preferred ($y^+$, using positive $\eta$) and dispreferred ($y^-$, using negative $\eta$) responses separately within a multiclass classification framework. However, the actual DPO loss function couples the gradients for $y^+$ and $y^-$ through the sigmoid of their log-likelihood ratio difference. This theoretical separation doesn't fully capture the coupled nature of the true DPO gradient dynamics, potentially limiting the direct applicability of conclusions drawn from the simplified, independent analyses (e.g., regarding negative $\eta/\rho$).
2. A key aspect of the squeezing effect is the unintended probability increase of the most confident incorrect prediction ($y^\star$). While the theory (Corollary 3.6) and toy example (Fig 1b) suggest SAM suppresses this increase, the main real-world experiment (Fig 1c) only tracks the chosen ($y^+$) and rejected ($y^-$) probabilities. It lacks direct empirical evidence showing SAM also prevents the rise of $y^*$ probabilities during actual DPO training.

### Questions
1. The theoretical framework links the mitigation of the squeezing effect (when modeling the $y^-$ dynamics with a conceptual $\eta < 0$) to using a negative perturbation radius ($\rho < 0$). Since $\rho$ in standard SAM represents the magnitude of perturbation within a neighborhood, could the authors provide further intuition or clarification on the interpretation and role of a negative $\rho$ in this theoretical context, especially regarding how it achieves the desired curvature regularization?
2. Could the authors provide empirical results from the real-world DPO experiments (similar to Figure 1c) that explicitly track the probability dynamics of the most confident incorrect prediction ($y^*$)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper seeks to address the "squeezing effect" (or "likelihood displacement") issue in DPO fine-tuning. This is a potentially harmful phenomenon, where the model performance degrades during DPO when it should be improving. The authors aim to theoretically analyze this behavior, and propose the root cause is due to interaction between the DPO objective and gradient descent. It is stated that rapid model updates expanding along high-curvature directions in the logit space explains the "squeezing effect". To remedy this, the authors propose using Sharpness-Aware Minimization (SAM), which high level seeks to minimize to "flat" surfaces. Since full SAM is too slow, the authors introduce a more practical variant, logits-SAM, which perturbs the outer-layer parameters only.

### Strengths
- The authors present rigorous theoretical diagnosis for a known DPO failure mode. Correlating the "squeezing effect" to "high-curvature directions" in the logit space  is a specific, actionable, and clear. The theoretical contributions and discussions are also clearly written, and presented as an elegant explanation for an existing harmful phenomenon. The clear theory to practical pipeline is also well motivated. Section 3.2 predicts how SAM should behave, Fig1-a,b confirm this prediction on a toy problem, and Fig1-c shows it on a real GPT-2 model.

- The message being delivered is also succinct: choose the SAM hyperparameter with the same sign as the learning rate to alleviate the squeezing effect. 

- The paper's primary practical contribution is the logits-SAM algorithm. This identifies that only the last layer needs perturbation, and the authors propose a simple practical "fix". Beyond just merely stating the instability of DPO and analyzing its failure modes, this additional practical solution is a strong addition.

### Weaknesses
- This method introduces a new highly sensitive hyperparameter ($\rho$). Table 3 demonstrates that while $\rho=10^{-4}$ might yield good results, the slightly larger $\rho=10^{-3}$ actually performs worse than the baseline. The optimal range seems extremely small. The majority of preference fine-tuning methods already encompass highly sensitive hyperparameters, and discussions about reference-based versus reference-free preference fine-tuning, in order to shift towards more robust solutions that are cheaper to tune. 

- Logits-SAM is not a new method. There are several works in this area recently, making this paper's contribution mildly incremental. The novelty of this work is in its application of logits-SAM in this particular setting. [1], [2], [3]

- Efficiency is highly model-dependent. The claim that overhead is negligible requires the final output later to be a tiny fraction of the total model parameters. This is true for many decoder-only models, but it is not a universal guarantee. Hence the experiments conducted are all on small models like GPT-2 with toy datasets, and are largely outdated. 

[1] Wen, Kaiyue, Tengyu Ma, and Zhiyuan Li. "How does sharpness-aware minimization minimize sharpness?." arXiv preprint arXiv:2211.05729 (2022).

[2] Li, Dongqi, et al. "Sharpness-aware minimization for out-of-distribution generalization." International Conference on Neural Information Processing. Singapore: Springer Nature Singapore, 2023.

[3] Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020).

### Questions
- Section 3.2 explicitly argues that for the negative learning rate in DPO, "one should choose a negative $\rho$". Section 3.3 says that the implementation "consistently use a positive $\rho$". The claim that this is equivalent hold for first-order methods, but is there a leap to assume it holds for curvature-aware methods like SAM? 

- Have the authors conducted experiments on larger more currently relevant models beside GPT-2?

- DPO (Eq. 2) already has an implicit KL regularization term. AdamW has weight decay. How does SAM's curvature regularization interact with these forms of regularization? 

- Why is perturbing only the last layer sufficient? Proposition 3.1 links the full parameter Hessian to the logit Hessian via a features map. Yet this is the output of all other layers. Can the authors give any intuition on why this is sufficient to regularize the entire system? Or is GPT-2 the limitation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes the ``squeezing'' effect in DPO—where preferred responses lose likelihood—by deriving a unified logit-space dynamic that links GD and SAM and reveals high-curvature modes as the culprit. It proves that SAM, applied with a coefficient sharing the sign of the effective learning rate, damps these unstable modes. Leveraging this, **logits-SAM** perturbs only the output layer, preserving curvature-aware regularization with ~2–3% overhead. Experiments on Pythia-2.8B and Mistral-7B across HH, TL;DR, UltraFeedback, AlpacaEval 2, Arena-Hard, and MT-Bench show consistent gains and reduced sharpness, with robust performance for small ρ (≈1e-5–1e-4).

### Strengths
1. **Clear Theoretical Contribution:** Provides a unified logit-space dynamical analysis that links GD and SAM, pinpointing high-curvature mode amplification as the mechanism behind DPO ``squeezing,'' and proving sign-aligned SAM mitigates it.

2. **Practical, Efficient Method:** Introduces logits-SAM, an output-layer perturbation that retains curvature-aware regularization with minimal overhead (~2–3%) and seamless integration into existing DPO/SLiC-HF/CPO pipelines.

### Weaknesses
1. **Theory–practice gap:** Core analysis relies on first/second-order approximations in logit space (fixed features, softmax CE), which may not fully capture nonlinearity and parameter coupling in deep, decoder-only LMs.

2. **Final-layer perturbation bias:** Restricting SAM to the output layer improves efficiency but may miss sharp directions arising in earlier blocks/attention layers, potentially undercutting robustness on harder distributions.

3. **DPO-specific framing:** The mitigation is analyzed through the ``negative learning rate / squeezing'' view of DPO; it is unclear how strongly the guarantees transfer to alternative preference-learning objectives or online RLHF settings.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to address the squeezing effect of DPO and proposes a new DPO method with leveraging sharpness-aware minimization (SAM). Comprehensive theoretical and experimental analyses have been conducted to demonstrate the effectiveness of the SAM. As such, while this work has some limitations. I still recommend an acceptance.

### Strengths
This work has the following strengths:

1.	This work studies on an important problem.
2.	This work proposes to leverage SAM to mitigate the squeezing effect of DPO, with providing comprehensive theoretical evidences.
3.	Extensive experiments on real-world datasets have been conducted to verify the efficacy of the proposed method.

### Weaknesses
I also have some concerns: 

1.	On the description of “gradient descent with a negative learning rate”: The phrase “gradient descent with a negative learning rate” is unconventional, as learning rates in DPO are typically positive. This may cause unnecessary confusion for readers. From the theoretical analysis, I understand that DPO may, in certain cases, apply a reversed gradient direction. However, it would be more precise to present this as theoretically equivalent to using a negative learning rate, rather than stating that the learning rate is negative outright. This clarification is important to prevent misinterpretation.

2.	Gap between theoretical setting and practical application: For ease of theoretical derivation, the authors formulate DPO as a multi-class logistic classification problem. While this abstraction facilitates analysis, it may not fully reflect practical usage scenarios of DPO. Additional explanation should be provided to justify the relevance and applicability of the theoretical setting to real-world DPO implementations.

3.	Many techniques in the theoretical section appear to follow the framework introduced by Ren & Sutherland. This is not necessarily a critical limitation, as research naturally builds upon prior work. However, the following questions should be addressed: (1) What are the specific theoretical challenges in integrating SAM into DPO? (2) What are the novel theoretical contributions of this work compared with Ren & Sutherland’s approach?



4.	I also some concerns on the experiments: 1) The “squeezing effect” is not a new concept and has been studied in recent work (e.g., [a1], [a2]). How does the proposed method perform compared with these recent baselines? 2) It would be better to include more experiments on diverse backbones and benchmark. 

[a1] C2-DPO: Constrained Controlled Direct Preference Optimization (arxiv'25)
[a2] Unintentional unalignment: Likelihood displacement in direct preference optimization (ICLR’25)

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
