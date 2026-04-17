# RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Standard human preference-based alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), are a cornerstone technology for aligning Large Language Models (LLMs) with human values. However, these methods are all underpinned by a strong assumption that the collected preference data is clean and that all observed labels are equally reliable. In reality, large-scale preference datasets contain substantial label noise due to annotator errors, inconsistent instructions, varying expertise, and even adversarial or low-effort feedback. This creates a discrepancy between the recorded data and the ground-truth preferences, which can misguide the model and degrade its performance. To address this challenge, we introduce Robust Enhanced Policy Optimization (RE-PO). RE-PO employs an Expectation-Maximization algorithm to infer the posterior probability of each label’s correctness, which is used to adaptively re-weigh each data point in the training loss to mitigate noise. We further generalize this approach by linking a broad class of preference losses to induced probabilistic models. This enables systematic robustification of existing alignment algorithms while preserving exact probabilistic equivalence for likelihood-style losses. Theoretically, under perfect calibration and a population/full-batch setting, we show that RE-PO recovers the true annotator reliability. Our experiments demonstrate RE-PO’s effectiveness as a general framework, generally enhancing four state-of-the-art alignment algorithms (DPO, IPO, SimPO, and CPO) against their corresponding standard versions. When applied to Mistral and Llama 3 models, the RE-PO-enhanced methods improve AlpacaEval 2 win rates by up to 7.0% over their respective baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Robust Preference Optimization (RPO), a novel method designed to address the challenge of label noise in Large Language Model (LLM) alignment. The authors correctly identify that existing alignment methods like DPO, IPO, and others operate under the strong and often flawed assumption that all human preference labels are correct. RPO reframes the problem by treating the true preference as a latent variable and employs an Expectation-Maximization (EM) framework to jointly infer the correctness of labels (E-step) and update the model policy along with annotator reliability parameters (M-step). A key strength is its design as a meta-framework, making it compatible with various existing preference loss functions. Comprehensive experiments on models like Mistral-7B and Llama-3-8B across benchmarks like AlpacaEval 2 and ArenaHard show consistent and significant performance gains when RPO is applied to several base algorithms (DPO, IPO, SimPO, CPO). The paper also provides theoretical guarantees, proving that under idealized conditions, RPO can recover the true reliability of annotators.

### Strengths
1. The core idea of applying an EM-based, soft-labeling approach to LLM preference alignment is highly innovative. Moving from hard, assumed-correct labels to a probabilistic model that explicitly accounts for noise is a significant conceptual and practical advancement.

2. The formulation of RPO as a wrapper that can enhance a wide range of existing alignment methods (demonstrated with DPO, IPO, SimPO, CPO) is a major strength. This greatly increases the potential impact and applicability of the work.

3. The paper is not merely empirical; it provides a solid theoretical foundation. The derivation of the EM algorithm for this context and, crucially, the proof of convergence to the true annotator reliability (Theorem 4.2) lend significant credibility to the method.

4. The experimental section is thorough, validating the method's effectiveness across two different base models, two challenging benchmarks, and four distinct alignment algorithms. The consistent improvements are compelling evidence for RPO's utility. The authors thoughtfully address the computational challenges of a full-batch EM algorithm by proposing a practical mini-batch training scheme with an Exponential Moving Average (EMA) for updating annotator reliabilities, making the method feasible for large-scale training.

### Weaknesses
1.  A notable weakness is the experimental setup where all preference data is modeled as coming from a single, virtual annotator (K=1). While the justification in a footnote is reasonable, it undermines the validation of one of RPO's core capabilities: distinguishing between multiple annotators of varying reliability. The empirical results primarily demonstrate noise robustness in general, but not necessarily the multi-annotator aspect.

2. The ablation study reveals that performance is sensitive to the initial annotator reliability (η₀) and the EMA momentum (α). This sensitivity could pose a non-trivial tuning burden in practice, especially in real-world scenarios where the optimal values are not known a priori.

3. The experiments rely on existing datasets and synthetic noise injection. While valuable, this does not fully capture the complexity and structure of noise found in real-world, multi-annotator preference data. Validation on a dataset with genuine, traceable multi-annotator labels would be stronger.

4. The EM procedure, with its additional forward passes to compute confidences (w_i) and updates for η, inevitably introduces computational overhead compared to the base algorithms. While likely acceptable given the performance gains, a brief discussion of the training time/ cost increase would be helpful for practitioners.

### Questions
1. How would RPO perform in a setting with a dataset containing multiple annotators with widely different and unknown reliabilities? Are there plans to validate the method on a dataset with genuine, traceable multi-annotator labels (e.g., from a crowdsourcing platform)?

2. Theorem 4.2 assumes full-batch updates and a perfectly calibrated model. In the practical mini-batch setting with stochastic gradient descent, have you observed any issues with the convergence or stability of the EM procedure, particularly for the annotator reliability estimates?

3. The experiments are conducted on 7B and 8B parameter models. Do you anticipate any challenges (e.g., memory, convergence) in scaling RPO to much larger models (e.g., 70B+ parameters), and have you conducted any preliminary experiments in this direction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RPO, a new framework for aligning LLMs that is designed to be resilient to noisy preference feedback. RPO addresses this by reformulating the alignment problem using an Expectation-Maximization. The core idea is to treat the "true" underlying preference as a latent variable. In the E-step, the framework uses the current model parameters to infer the posterior probability (a "confidence score" $w_i$) that each observed preference label is correct. In the M-step, these confidence scores are used as adaptive weights in the loss function to update both the LLM policy and a set of learned annotator reliability parameters $\eta_k$. The authors present RPO as a "meta-framework" capable of making existing alignment algorithms (like DPO, IPO, SimPO, and CPO) more robust. They provide a theoretical proof that RPO can converge to the true annotator reliability under the assumption of a perfectly calibrated model.

### Strengths
- I like the use of EM to jointly infer label correctness and optimize the model. I find it an elegant and principled approach.

- The authors provide a generalization of RPO into a meta-framework. The formulation (Eq. 2) that connects arbitrary preference loss functions $\mathcal{L}\_{pref}$ to a unified probabilistic model allows RPO to be easily applied to a wide range of existing methods

- The experiments show that RPO consistently improves performance across different PO algorithms and two different base models. The reported gains are substantial. 

- The paper provides theoretical analysis and empirically show that the RPO framework can accurately estimate the true reliability of different datapoints

### Weaknesses
- The paper positions RPO as a solution for robust alignment but doesn't compare it against other methods designed for the exact same problem. The related work section explicitly mentions rDPO and Selective DPO, but they are absent from the main results in Table 2. The paper also misses simpler baselines. For instance, how does RPO compare to standard DPO (or IPO, etc.) trained with label smoothing? Without these comparisons, it is impossible to conclude that the added complexity of the EM framework is justified. 

- The paper claims to address both simple annotator error and the "inherent pluralism of human preferences". However, it also states that these disagreements are "functionally equivalent to noise" and proceeds to learn a single "latent collective preference". This is a big oversimplification. The method does not truly deal with preference pluralism (e.g., by learning a distribution over rewards) but instead treats all non-majority opinions as incorrect data to be down-weighted. The related work on diverse preferences is not adequately addressed (e.g. [1]).

- The central theoretical convergence guarantee (Theorem 4.2) relies on the assumption of a "perfectly calibrated model". The assumption that the model already matches the ground-truth preference distribution is at odds with the goal of training, which is to find such a model. This weakens the applicability of the theoretical claims to the practical training process, which generally starts from a poorly calibrated state. I understand that, in practice and in most cases, the starting model is already good enough to provide decent information, but this needs to be addressed and discussed somewhere in the paper. The EM algorithm's E-step depends on the model's current (and potentially poor) estimates of preference probabilities $p(y_w >^* y_l | x, \theta^{(t)})$. If the initial model is badly aligned, its confidence estimates $w_i$ will be unreliable, potentially leading the M-step to down-weight correct labels and up-weight noise, hindering or preventing convergence. The paper does not investigate or discuss this. 

- The paper is missing a qualitative analysis of what its model identifies as noise. What kind of preference pairs receive a low confidence score $w_i$? Are they obvious annotator mistakes (e.g., typos), subjective disagreements (i.e., pluralism) or something else? 

- The writing could be improved for clarity. For instance, $K$ is used in Algorithm 1 but is never explicitly defined in the main text, only appearing in a footnote on page 7

[1] A Minimaximalist Approach to Reinforcement Learning from Human Feedback, https://arxiv.org/abs/2401.04056

### Questions
- Could the authors please provide experimental comparisons against the highly relevant baselines mentioned in their own related work, specifically rDPO and Selective DPO? Furthermore, how does RPO compare against simpler robustness-enhancing techniques like applying label smoothing to the standard DPO/IPO/SimPO/CPO losses? 

- Can the authors provide 2-3 qualitative examples of preference pairs from the UltraFeedback dataset that were assigned very low confidence scores $w_i$ by the RPO-DPO model?

- I think the authors should add discussions around poor alignment of the initial model

### Soundness
3

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
4

### Summary
This paper introduces Robust Preference Optimization (RPO), a general framework designed to address preference noise commonly encountered in large language model (LLM) alignment. Existing preference-learning methods such as DPO and IPO typically assume clean and consistent preference labels, which rarely hold in real-world settings where human feedback is diverse and occasionally erroneous. RPO treats preference labels as latent variables and employs an Expectation–Maximization (EM) procedure to jointly estimate the “soft label” confidence and annotator reliability during training. This enables adaptive reweighting of noisy data. Furthermore, the authors propose a unified probabilistic formulation that generalizes multiple preference objectives under a single likelihood-based framework, thereby enhancing methodological coherence and extensibility.

### Strengths
1. The paper tackles a realistic and important issue—label noise in preference alignment—that directly affects the reliability of RLHF-style training.
2. Modeling annotation noise via latent variables and integrating EM into preference optimization is a well-grounded and elegant idea.
3. The probabilistic unification of different preference losses provides conceptual clarity and could potentially bridge diverse alignment approaches under one framework.
4. The experiments demonstrate consistent robustness improvements across several noisy-preference setups, validating the framework’s effectiveness.
5. The paper is generally well written, with clear mathematical derivations and a reasonable balance between theory and empirical validation.

### Weaknesses
1. All experiments are conducted under a single-annotator noise setting. Incorporating multi-annotator scenarios would make the robustness claims more realistic.
2. Methods such as ORPO and rDPO share similar robustness motivations but are not included in the empirical evaluation. Adding these baselines would strengthen the comparison.
3. Using GPT-4o preferences as ground truth could introduce systemic bias. It would be better to include multiple evaluators or human assessments for verification.

### Questions
see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Robust Preference Optimization (RPO), a general framework for aligning large language models (LLMs) with noisy human preference data. Unlike standard RLHF or DPO methods that assume all feedback is correct, RPO explicitly models the correctness of each annotation as a latent variable. Using an Expectation–Maximization (EM) approach, RPO jointly estimates annotator reliability and model parameters. In the E-step, it computes sample-wise confidence weights reflecting annotation trustworthiness; in the M-step, it updates model parameters and annotator reliabilities via weighted loss optimization.

### Strengths
1. RPO provides an unification of robust preference optimization via probabilistic modeling and EM inference. 

2. The paper includes convergence proofs for the EM updates and demonstrates that annotator reliability can be recovered under realistic assumptions. 

3. Experiments on multiple LLMs (Mistral, Llama-3) and datasets (AlpacaEval 2, Arena-Hard) show RPO outperforming baselines.

### Weaknesses
1. Although RPO models annotator reliability, the main experiments assume a single virtual annotator (K=1). The paper would be more convincing if it demonstrated effectiveness on datasets with genuine multi-annotator disagreement.

2. The EM-based procedure may introduce extra computational costs and slower convergence. While the paper mentions using EMA for efficiency, it does not quantify runtime overhead or training stability compared to standard DPO.

3. While related work such as rDPO and Hölder-DPO is cited, these methods are not included in the empirical comparison. Adding such baselines would clarify the relative robustness advantages of RPO.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2
