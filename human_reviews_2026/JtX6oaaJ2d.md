# Improving LLM Unlearning Robustness via Random Perturbations

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Here, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the *unlearning process as backdoor attacks and defenses*: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods *themselves poison the model*, make it more vulnerable to forget-tokens, and *hide rather than erase* target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of models.  Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanisms of unlearning that can shed light on future research directions for improving unlearning robustness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies why current large language model unlearning methods often weaken model robustness, leading to errors when retain queries accidentally include forget tokens. The authors show that the unlearning process behaves like a backdoor attack in which forget tokens act as triggers that reintroduce forgotten behavior. To mitigate this issue, the paper reframes the retaining process as a backdoor defense and proposes Random Noise Augmentation, a simple and model agnostic technique that injects small Gaussian noise into retain representations during fine tuning. Theoretical and empirical results demonstrate that this method improves the robustness of unlearned models while maintaining their forgetting and retaining performance.

### Strengths
1. The paper introduces a fresh angle by connecting LLM unlearning with backdoor attack and defense mechanisms, offering a clear framework for understanding existing weaknesses. 

2. Theoretical analysis provides partial support for the proposed method, though it relies on many assumptions that weaken its rigor. 

3. The writing is clear, and the figures are well-presented.

### Weaknesses
1. The experiments lack sufficient baselines. The authors only study a few non-robust LLM unlearning methods, while many recent approaches have been specifically proposed to enhance robustness in LLM unlearning [1-2].
2. The experimental evaluation is narrow, focusing only on the WMDP benchmark. How does the proposed method perform on other benchmarks such as TOFU [3] and MUSE[4]?
3. Using only MMLU to evaluate model utility is limited. It would be more convincing to include additional utility metrics[5].
4. The paper does not discuss how the proposed method behaves under other types of attacks, such as relearning attacks or adversarial prompts [6].
5. Many proofs rely on overly strong assumptions. For example, Eq. (9) asserts that adding Gaussian perturbations increases the expected loss, which depends on a strong assumption of local convexity or a positive-definite Hessian. In deep LLM latent spaces, this condition often fails, so the inequality $\mathbb{E}[\ell(y|z{+}v)]>\ell(y|z)$ is not generally guaranteed.

> [1] Tamirisa R, Bharathi B, Phan L, et al. Tamper-resistant safeguards for open-weight llms[J]. arXiv preprint arXiv:2408.00761, 2024.
> 
> [2] Fan, Chongyu, et al. "Towards llm unlearning resilient to relearning attacks: A sharpness-aware minimization perspective and beyond." arXiv preprint arXiv:2502.05374 (2025).
> 
> [3] Maini, Pratyush, et al. "Tofu: A task of fictitious unlearning for llms." arXiv preprint arXiv:2401.06121 (2024).
>
> [4] Shi, Weijia, et al. "Muse: Machine unlearning six-way evaluation for language models." arXiv preprint arXiv:2407.06460 (2024).
>
> [5] Che Z, Casper S, Kirk R, et al. Model tampering attacks enable more rigorous evaluations of llm capabilities[J]. arXiv preprint arXiv:2502.05209, 2025.
>
> [6] Łucki, Jakub, et al. "An adversarial perspective on machine unlearning for ai safety." arXiv preprint arXiv:2409.18025 (2024).

### Questions
See weakness.

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
4

### Summary
This paper propose Random Noise Augmentation, a method to avoid the misbehave when forget-tokens are in the retain-queries. The author frames current unlearning methods as inadvertently introducing a backdoor attack and proposes RNA as a backdoor defense to enhance the robustness of unlearned models against non-adversarial forget-tokens in retain-queries. By adding random perturbations in the data retaining stage, RNA significantly improves the model's robustness. The theoretical analysis of this paper focuses on how the presence of forget-tokens introduces randomness in latent representations, and proves a bounded probability for RNA to reject the misbehavior caused by forget-tokens.

### Strengths
1. The paper establishes a unified view of the two primary classes of LLM unlearning methods, Representation Misdirection and Preference Optimization, by analyzing both through the lens of the generative latent variable model.

2. The paper provides theoretical guarantees for RNA's effectiveness in improving the robustness of models by rejecting the detrimental effects caused by forget-tokens.

3. Comprehensive experiments are conducted. This extensive evaluation rigorously validates the effectiveness and generalizability of the proposed RNA method by including diverse unlearning methods and demonstrating generalization across three different LLMs. The rigorous testing further includes detailed ablations on the RNA mechanism itself, varying the crucial noise scale and the inner model layers for injection.

### Weaknesses
1. The paper's theoretical guarantees are oversimplified approximations because they rely heavily on Assumption 4.1, treating the perturbation $\epsilon$ as simple, independent Gaussian noise. This ignores the reality that the actual perturbation is deterministic, complex, and highly dependent on the model's parameters and context, a complexity not fully addressed by the mere "Gaussian-like" appearance of the empirical activation differences in Figure 9.  The oversimplification is further evidenced by the experimental results in Table 6. If the influence of unlearning could indeed be modeled as Gaussian noise, then by the same logic, adversarial perturbations would exhibit similar characteristics, and the RNA should yield consistent improvements in adversarial robustness. However, the results in Table 6 do not support this expectation.

2. The proposed method requires manipulation of the model's inner layers, which may be impractical for real-world scenarios where users cannot modify these layers to inject random noise. This is an unavoidable weakness, although it is discussed in the appendix.

3. Since the paper does not explicitly report running multiple independent trials using different random seeds or provide measures of variance (e.g., standard deviation), the reported performance figures are susceptible to random initialization effects and sampling randomness.

### Questions
1. See weakness 1.

2. How to determine which perturbation is added to the retain-query. Since there can be many key-words in the forget set. 

3. It is interesting that the choice of layer significantly affects the effectiveness of the proposed method. In practical settings, should the layer be selected by some principles or determined through grid search?

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
4

### Summary
This paper proposes reframing LLM unlearning as a backdoor attack-defense process, showing that current unlearning methods make models fragile when forget-tokens appear in retain queries. To improve robustness, the authors introduce Random Noise Augmentation (RNA), which injects small Gaussian noise into retain representations during fine-tuning. RNA is lightweight, model-agnostic, and theoretically grounded. Experiments on several 7B–8B LLMs demonstrate significant improvements in robustness without sacrificing forget or retain performance.

### Strengths
- This paper recasts unlearning as a backdoor attack/defense mechanism is original and intuitive. It clarifies why unlearned models “misbehave” when seeing forgotten tokens, they’re activating an unintended backdoor.

- This paper also has a clear theoretical grounding, for example, analytical results (Eqns 9 and 13) give interpretable relationships between robustness, noise variance, and token perturbation.

- Experiments also show substantial accuracy recovery on perturbed MMLU queries.

### Weaknesses
- Limited experimental diversity. The experiments focus mainly on the WMDP benchmark and a single model scale. It would strengthen the paper to include additional benchmarks such as MUSE [1] and to evaluate across both larger and smaller model sizes to assess generality.

- Synthetic perturbation design. The way retain sets are perturbed (e.g., replacing one token with “SARS-CoV-2”) feels somewhat artificial compared to real-world mixed-context prompts. Including datasets like MUSE could help verify whether the proposed method remains effective under more natural conditions.

- Terminology clarity. The paper’s use of “unlearning robustness” could be better defined. Traditionally, robustness in unlearning refers to resistance against relearning or recovery of forgotten knowledge, whereas this work mainly studies how retain-set performance degrades when forget-tokens appear. Clearer differentiation would help readers understand the scope.

- Missing robustness evaluations. It would be valuable to test the method against relearning and jailbreaking attacks to more comprehensively assess robustness.

> [1] Shi, Weijia, et al. "Muse: Machine unlearning six-way evaluation for language models." arXiv preprint arXiv:2407.06460 (2024).

### Questions
Please refer to the weaknesses section. I would be willing to raise my score if these issues are adequately addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focuses on understanding and improving robustness of llm unlearning. Different from prior efforts that mostly look at robustness to unlearned content, this work focus on robustness to retain content. This work propose a theoretical framework that frames the unlearning problem as a backdoor attack and defense problem. Based on the understanding, the authors propose random perturbations on top of the latent representation to preserve retain performance when the retain set contains forget tokens.

### Strengths
- This paper focus on understanding robustness of unlearning and specifically look at robustness on the retain set, which is largely under explored in prior literature.
- The RNA method is a very simple heuristic and seems to be effective on perturbed retain set.

### Weaknesses
- I did not get Section 5.1. The author said we train model with the "poisoned" forget set and the retain set, but equation 10 is still sampling data from $\mathcal{Z}$, which is the non-"poisoned" forget set and retain set. Then I don't understand the role of $T$ and $\Omega$ here. Now given equation 11 and equation 12 is correct and make sense, why does the conclusion: "current state-of-the-art LLM unlearning methods themselves “poison” the model and make it more vulnerable to forget-tokens" follows? There is no analysis of how current LLM unlearning methods map to this framework. Detailed explanation in this part is largely missing, making this section very confusing.
- In the experiment section, the evaluation is quite limited. The authors only evaluate on wmdp unlearning and only on one forget token "SARS-CoV-2". The authors should evaluate on more tasks (tofu, rwku, etc) and a variety of different forget tokens for each task. Otherwise, it's hard to tell whether the proposed method works for just this dataset or for general unlearning task.
- Figure 1 is very hard to interpret. For each figure, I suggest using different symbols for different methods and one color for baseline unlearning method, one color for the RNA augmented method. Or use a table to present the results. Otherwise it is very hard to quantify the advantage of RNA.

### Questions
See weaknesses above. My main question is the how the framework converts to the conclusion that "current state-of-the-art LLM unlearning methods themselves “poison” the model and make it more vulnerable to forget-tokens".

### Soundness
2

### Presentation
2

### Contribution
3
