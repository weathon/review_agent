# UniARM: Towards a Unified Autoregressive Reward Model for Multi-Objective Test-Time Alignment

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 4, 8

## Abstract
Multi-objective alignment aims to align LLM responses with multiple human preference objectives. Among existing methods, guiding the generation of frozen LLMs through autoregressive reward models (ARMs) to accomplish multi-objective test-time alignment is a low-cost solution.
However, these methods typically rely on independent parameters for each preference objective, either by training ARMs independently across preference dimensions, which neglects interactions among preference features, or by training a single ARM with separate feature extraction modules for each preference, which can cause feature entanglement. Both strategies can result in misalignment between generated outputs and user preferences.
To address this limitation, we propose Preference-Modulated \& Shared Low-Rank Adaptation (MoSLoRA) for ARM training, which  first extracts shared features via a preference-agnostic module and then  applies affine transformations to shared features via a preference modulation module conditioned on mixed preference vectors.  This design mitigates feature entanglement and enables precise control over preference trade-offs during inference. Building on this, we introduce the Unified Autoregressive Reward Model (UniARM), a novel framework for multi-objective test-time alignment. UniARM jointly models all preference dimensions in a single parameter space, eliminating the need for independent parameters for each preference objective. Experimental results show that UniARM improves HV and MIP by 18.5\% and 30.2\% in the safety alignment task. It also enables weak-to-strong guidance, where a smaller UniARM guides a larger frozen LLM, yielding HV and MIP improvements of 9.1\% and 6.8\% in the safety alignment task, and 5.4\% and 10.7\% in the assistant task. Notably, these gains are achieved without introducing additional parameters or increasing inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets multi-objective alignment, and proposes Preference-Modulated & Shared Low-Rank Adaptation (MoSLoRA) for autoregressive reward modelling. Specifically, MoSLoRA extracts features shared across preferences and then adopts affine transformation to shared features. Building upon MoSLoRA, this paper proposes the UniARM, a autoregressive reward modelling framework for multi-objective test-time alignment. Experiments demonstrate the effectiveness of UniARM in safety and helpfulness alignment tasks, outperforming existing baselines.

### Strengths
- This paper addresses an interesting direction of multi-objective test-time alignment.
- This paper proposes a new MoSLoRA architecture.
- Experiments show the effectiveness of the proposed MoSLoRA and UniARM.
- There is a clear structure in related work review.
- The case study presentation is very intuitive.

### Weaknesses
- It seems that MoSLoRA relies on predefined semantic representation $o$, whereas PBLoRA does not. Therefore, it is unclear whether the performance improvement of MoSLoRA over PBLoRA stems from its architecture that integrates more information.
- It seems that this work is an incremental research study of previous PARM, and MoSLoRA is an improved version of PBLoRA. While many readers are familiar with standard LoRA, they may be less familiar with PARM and its proposed PBLoRA. I think that additional explanation or background from standard LoRA will enhance the readability of the paper.
- The evaluation metrics used in this paper are mainly HV and MIP. However, WinRate is a more widely used metric in the LLM preference alignment studies, but it is not involved into this paper to assess the generation quality of the model.
- It seems that the experiments are primarily conducted on Alpaca-7B, which is a relatively outdated model. Using more recent models, such as Qwen3 series, will be better convincing.
- There is no anonymous code link to present the reproducibility of this work.

### Questions
- Have you considered comparing UniARM with vanilla RLHF methods, such as PPO with a pretrained reward model, under test-time alignment scenario?

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
4

### Summary
This paper proposes UniARM, a unified framework for multi-objective alignment of LLMs that balances multiple human preference goals at test time. It achieves this by introducing MoSLoRA, which learns shared features across preferences and modulates them through preference-conditioned transformations, reducing feature entanglement and enabling fine-grained trade-offs. The approach improves alignment quality and efficiency without slowing inference and without retraining the base LLMs over and over again.

### Strengths
1. This paper deals with a very relevant area of using test-time alignment method to achieve multi-objective goals and achieve good experimental result. 

2. Unlike the prior art PARM, this method does not require linearly combining different core tensors based on the preference vector during test time, which is more reasonable. 

3. The experiments are extensive, including helpfulness and harmlessness evaluation, as well as weak-to-strong extension. It is especially good to see the weak-to-strong experiments as it is very expensive to retrain larger LLMs for multi-objective goals using training-based method, and the proposed method seems like an efficient alternative.

4. The paper is well-written and clearly explain the difference between the prior work.

### Weaknesses
1. While the experiments setting follows prior work, the LLM used here seems not very up to date. It would be nice if the author can evaluate on more recent LLMs. For example, tulu-3 instead of tulu-2. 

Other minor issues
1. Typo in equation (10)
2. In Figure 2, the results of RS and MOD are set to zero. Although it is understandable that they are very expensive to run, I am not sure if it is a good idea to set them to be zero.

### Questions
If the model has been trained for, for example, objective A and B, and then there is a different objective C, what is the quickest way to adapt to the new objective? It seems that in GenARM, the reward model of objective C can be immediately used. I wonder what we should do for your proposed UniARM.

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
3

### Summary
This paper introduces UniARM, a framework for multi-objective test-time alignment. The proposed method employs a parameter-efficient architecture called MoSLoRA, which consists of a preference-agnostic module and a preference-modulation module. Experimental results demonstrate the effectiveness of UniARM compared to existing test-time alignment approaches.

### Strengths
- The related work section is comprehensive.

- The method eliminates the need to train multiple separate ARMs or preference-aware modules. Instead, it requires training only one preference-agnostic module and one preference-modulation module for alignment multiple objectives.

- The experimental results show superior performance compared to previous test-time alignment methods.

### Weaknesses
Overall, I find the methodological design of the paper reasonable, though the experimental section requires further strengthening. If the following concerns can be adequately addressed, I would consider raising my score.

- I'm unfamiliar with test-time multi-objective alignment methods based on ARM. Therefore, I'm puzzled about the motivation behind this approach. Since we can fine-tune the ARM, why do we just fine-tune the LLM itself?

- The backbone model used is relatively outdated. Employing a more recent backbone model (e.g., LLaMA-3, Qwen-3) and reward model (e.g., ARMO-RM, Skywork-LLaMA-V2) would strengthen the validity of the results.

- Since the ARM also involves fine-tuning the base model, it is essential to compare UniARM with other training-based multi-objective methods, such as MODPO and other state-of-the-art baselines.

- Incorporating LLM-as-a-judge evaluation on benchmarks like AlpacaEval (for helpfulness) would provide a more realistic and convincing assessment.

- Table 2: It would be beneficial to include results without any ARM, using only Alpaca-65B. Additionally, the use of an older model raises concerns about whether similar performance can be achieved with newer backbone models.

- An analysis of the sensitivity to hyperparameters such as λ and β is missing. Also, the impact of varying the text descriptions of objectives on the results should be investigated.

- Details regarding training and inference computational costs are not provided.

Typos:

- Line 32: The meaning of "PB" should be explicitly explained to enhance clarity.

- Eq. (3):The notation "i" is not defined, which may cause confusion.

- Eq. (10): A closing bracket ")" is missing.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the problem of multi-objective test-time alignment of LLMs. The authors propose UniARM based on the better parameterization method named MoSLoRA. The method is parameter-efficient because it uses the semantic embeddings of the model itself to encode the multiple preferences instead of additional weights. At the same time, the empirical results also demonstrate the superiority of the method in comparison to previous state-of-the-art methods. Although there is some weakness, I believe they can be improved with minor revisions after the rebuttal. Therefore, I recommend "accept".

### Strengths
1. The paper is well-written and easy to understand.
2. The proposed method performs better while being more parameter-efficient than baselines. I believe this gain outweighs the slightly weak originality of the proposed method (weak originality as it consists in a minor change to the low-rank adapter used in PARM plus a regularizer).
3. I think the experiments are very well designed. Enough meaningful baselines are included. The ablation experiments provide clear clues as to how each component of the method affects the performance. The choice of the quantitative metric of HV and MIP makes the performance gap between the proposed method and baselines more convincing.
4. The method itself is simple to reimplement, which implies a strong potential to be an impactful contribution to the community.

### Weaknesses
1. Although the proposed method is empirically effective, there is a lack of intuitive or theoretical explanation of where the effectiveness (i.e., the better Pareto front) comes from. Can the authors explicitly provide such explanations? For example, why can a different parameterization of the token-level reward models alone (according to the ablation experiment when $\lambda=0$) lead to a better Pareto front? 
2. (Partially related to the first weakness) The generality of the method is unknown. Is the better Pareto front only limited to the Alpaca family? Can the authors provide results on other families (Llama, Qwen, etc.) to make the generality of the method more convincing?
3. I find the term "Pareto-optimal" too strong and vague to describe an ARM as there is still room for improvement. I suggest a softer term "more/less Pareto-efficient".

---------------
Note: I identified several typos (not a weakness but a nontrivial issue to fix): 
 * (Less serious) Line#292-#293:  "the reward of UniARM is computed as::" --> "the reward of UniARM is computed as:"
 * (Less serious) Line#429: "GenARM (Xu et al., 2025));" --> "GenARM (Xu et al., 2025);"
 * (More serious) Equation (5): There is a missing right parenthesis for the difference of the two log probabilities.
 * (More serious) also Equation (5): If I am not mistaken, either $(-1)^{z_i}$ needs to be negated as $-(-1)^{z_i}$ or the meaning of $z_i$ needs be flipped.

### Questions
see **Weakness** where I have included the questions.

### Soundness
3

### Presentation
4

### Contribution
3
