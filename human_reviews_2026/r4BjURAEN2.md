# Towards Causal Fine-Tuning under Latent-Confounded Shift

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 0, 2

## Abstract
Adapting to latent-confounded shift remains a core challenge in modern AI. Such shift is driven by hidden variables that induce spurious, non-transportable correlations between inputs and outputs. A practical failure mode arises when fine-tuning pre-trained foundation models on confounded data (e.g., where certain text tokens or image backgrounds spuriously correlate with the label), leaving models vulnerable at deployment. We introduce *causal fine-tuning, which frame model adaptation as an identification problem* and pose an explicit causal model that decomposes inputs into low-level spurious features and high‐level causal representations. Under this family of models, we formalize the assumptions required for identification. Using pre-trained language models as a case study, we show how identifying and adjusting these components during causal fine-tuning enables automatic adaptation to such shift at test time. Experiments on real-world stress-test benchmarks demonstrate that our method outperforms black-box domain generalization baselines, highlighting the benefits of explicitly modeling causal structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of "latent-confounded shift," where foundation models, when fine-tuned, learn spurious correlations induced by unobserved confounders (e.g., data source) that do not generalize to out-of-distribution (OOD) settings. The authors propose "Causal Fine-Tuning" (CFT), a novel framework that recasts model adaptation as a causal identification problem. CFT is based on an explicit causal graph that decomposes the input into a high-level causal representation C, a high-level spurious representation S, and a low-level, environment-invariant representation Φ. The key technical contribution lies in a two-part identification strategy: 1) identifying the invariant causal feature C by contrasting representations from a frozen pre-trained model (R_0) and a fine-tuned model (R_1), and 2) using a front-door adjustment via the low-level features Φ to compute an interventional prediction p(y|do(x)). Experiments on sentiment analysis benchmarks with injected, controlled spurious correlations demonstrate that CFT significantly outperforms standard fine-tuning and other domain generalization baselines.

### Strengths
1. The paper provides a clear and compelling formulation of "latent-confounded shift" as a critical failure mode for fine-tuned foundation models. By moving beyond standard OOD settings and explicitly modeling the role of unobserved confounders, the work addresses a problem of significant practical and theoretical importance.

2. The proposed CFT framework is a principled and elegant application of causal inference principles to model fine-tuning.

3. The paper is well-organized and well-written.

### Weaknesses
1. The entire framework is built upon a detailed and complex set of causal assumptions (Assumptions 4.1-4.4 and the graph in Fig. 2). While these assumptions enable the identification strategy, their validity in real-world scenarios is difficult. For example, the assumption that low-level features Φ (e.g., the embedding layer) are strictly environment-invariant (no arrow from σ) is a strong claim. Environmental shifts could subtly influence token distributions and their embeddings. Also, the front-door path assumption (Assumption 4.4) is also very strong, requiring that Φ contains all necessary information to identify C and that C fully mediates the effect of Φ on Y. The paper could be strengthened by discussing the sensitivity of the method to violations of these assumptions.

2. The experimental validation is limited. It is confined to text classification with artificially injected spurious correlations (e.g., modified stop words), which may not reflect more natural and subtle forms of confounding found in real-world. Furthermore, the lack of experiments on other modalities, such as vision, leaves the method's generalizability across different tasks an open question.

3. The authors argue against using GenAI tools like ChatGPT for creating benchmarks due to cost and reproducibility. This is a valid point. However, could GenAI be used to generate more diverse and subtle confounders than the simple suffix-based injection used in the paper, providing a more challenging testbed?

4. The graph includes confounders U_S and U_Φ. The role of U_S (confounding R_1 and Φ) and U_Φ (confounding Φ and Y) is crucial for justifying the front-door adjustment. However, the intuition for what these confounders represent in a real-world NLP task is not fully developed. Providing concrete examples would greatly improve clarity.

5. Loss Function L_C (Eq. 3): The first term enforces invariance by minimizing the L2 distance between two distributions. Was this choice of metric (L2) crucial? Have other divergence measures (e.g., KL, Wasserstein) been considered?

6. The font size in Figures 4 and 5 is too small, making them difficult to read. I would strongly recommend increasing the font size to improve readability.

### Questions
See Weaknesses

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
This paper proposes causal fine-tuning for robust generalization under confounded shifts, validated via experiments on sentiment analysis using language models.

### Strengths
- Propose a front-door adjustment fine-tuning method, offering a principled approach to handling latent confounding.
- Provides a theoretical analysis with clear problem formulation, ensuring the method's derivation is well-motivated and grounded.

### Weaknesses
- There are some unclear assumptions and designs:
  - Why does the causal variable $C$ remain the same across pre-training and fine-tuning?
  - How to select $k$ for low-level features in different applications, and there are no ablation studies.

- Experiments are confined to BERT on sentiment analysis, leaving broader NLP tasks and models unverified.

- Writing issues, for example:
   - Line 170, missing "an" before "assumption"
   - Line 250, ambiguous phrasing "the pre-trained and training fine-tuning data". It may be "the pre-trained model"
   - Line 354, likely typo "WSA" should be "SWA"

### Questions
- How do the computational cost and training time of causal fine-tuning compare to baseline methods?
- Given that SWA outperforms CFT in some scenarios (Table 2), how can we determine whether to select CFT over other methods in real-world applications, particularly when the level of spurious correlations is unknown?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a causally inspired framework for improving model robustness during fine-tuning. It introduces latent variables representing causal and spurious factors and assumes that spurious correlations in pretraining and fine-tuning arise from distinct sources. Under several structural assumptions, the authors derive a formulation for estimating p(y∣do(x)) and propose an algorithm that approximates this intervention. Experiments on synthetic and text classification benchmarks suggest improved out-of-distribution generalization compared to standard fine-tuning baselines.

### Strengths
1. The paper introduces an interesting and ambitious causal framing for fine-tuning;
2. The confounded shift scenario in Figure 1b is under-explored;
3. The proposed algorithm yields consistent OOD improvements on sentiment and synthetic datasets.

### Weaknesses
1. Non-standard (and confusing) use of the regime variable $\sigma$. Writing $\sigma=do(x)$ makes $\sigma$ both a random variable and an intervention command.
2. (R_0,R_1,\Phi)=f(X)$ are introduced non-constructively, then used in proofs as if measurable variables. The paper later says it will "explicitly" construct them, but the construction is only sketched and heavily heuristic. It is unclear how the chosen construction satisfies the causal graph in Figure 2.
3. Assumption 4.2 requires $\{S_0,S_1,C\}$ to be mutually independent, with $S_0 \to R_0$ only, $S_1 \to R_1$ only, and $\sigma$ can only affect $S_1$. It also postulates that any dependence between $S_0$ and $S_1$ is solely via $C$. These restrictions drive the identification story, but seem implausible in many real settings. It is also not validated empirically if these assumptions hold in real data.
4. Assumption 4.3 states $\sigma$ affects the system only via S, but Assumption 4.2 already restricts $\sigma$’s effect to $S_1$ (not $S_0$).
5. Front-door–style Assumption 4.4 is very strong and untested. It requires no direct edge $\Phi\to Y$, while allowing confounding between $\Phi$ and $Y$. This is a demanding structural claim, and the paper gives no empirical test for it; yet it is pivotal for identification.
6. The estimation for P(Y|do(X)) is computed over all possible $\Phi$ and $x$, but in practice is only estimated within a mini-batch. This surrogate for interventional averaging lacks a causal justification and may bias the estimate.
7. Experiments are run on synthetic datasets where spurious signals are injected with controlled strengths. This is useful for controlled tests, but it doesn’t demonstrate that the assumptions (4.2–4.4) hold in naturally occurring OOD shifts.
8. Core choices (e.g., choice of $\Phi$, dimensionality of C, shuffle scheme) are not thoroughly ablationed.
9. The language is very informal (e.g. "we may be thrown away too far...and the safer bet...") and the paper appears to be unpolished.

### Questions
See above weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes "Causal Fine tuning" as a new fine-tuning method that is robust against spurious correlations. This involves first fine-tuning the model in a standard way. Next, they learn the hidden causal invariant feature using the method from Kugelgen et al. Then, using C together with low-level features, they estimate P(y|do(x)). Experiments show that CFT is more robust to spurious correlations or change in the (hidden) confounder compared to SFT.

### Strengths
1. The paper solves an important problem of distribution shift in fine-tuning
2. The paper clearly states its assumptions in the main paper

### Weaknesses
**Theory**
1. The main theoretical result (identifiability of C) is from Kugelgen et al. (2021) and the paper is cited. However, I could not find any discussion on how 1)this paper's notation corresponds to the Kugelgen paper. For example, what is the C, R0 and R1 in the Kugelgen paper? 2)whether the assumptions of this paper satisfy the assumptions required for Kugelgen's theorem 4.4. Omitting this places a burden on the reader to go through and understand the complete Kugelgen paper (including their notation, assumptions etc.). I would also expect a proof sketch or even a complete proof in the appendix even if it largely follows Kugelgen's proof. 

2. Several assumptions are not clearly mentioned. For example Line 248 mentions "Let the mapping between {S0, S1, C} and {R0, R1, Φ} obey the invertibility conditions of Von Kugelgen et al. (2021)". However, the paper should explicitly mention what these are in this context. As far as I understand, it is required that the function from C,S0,S1 to R0,R1 to be invertible. Why is this a reasonable assumption?

3. Component 2: Line 315 says the loss function is constructed from Theorem 4.5. It is not clear to me, however, how it is constructed from Theorem 4.5. Please add discussion and proof in the appendix.

4. The new variables introduced in Section 4 should be explained with an example. It is hard to understand why these variables were introduced and what they intuitively mean. For example, in the Kugelgen paper, C is the content and S is the style. However, I am not sure what they mean in this paper's setting.

**Experiments**

5. The causal graph in Section 4 is quite complex and I would have liked the experimental section to show that the framing of the problem as this causal graph is really necessary. For example, is it necessary to learn C? One good experiment might be to replace C by R1 in Step 8 of Algorithm 1. That is, you still do the shuffling of $\phi$ but use R1 for the final prediction.

6. While CFT outperforms SFT, the performance of CFT too decreases considerably as the degree of shift increases. Could the authors explain why it is so? Is it because real data may not satisfy all the assumptions? If there is indeed a setting where all the assumptions are satisfied, would we still expect the performance to decrease?

**Minor**

7. Some missing references on hidden confounder shift papers: [1] [2] 

8. I would like to see results on fine-tuning bigger models like Qwen-2.5-3b, Llama-3-8b, Olmo-2-7b etc I am interested in seeing if they too degrade with the shifts in the current experimental section. I understand the authors may face compute constraints and the review period may not be enough time. Therefore, this is just a minor weakness for me. But experiments on one of these big models would be great.


[1]Tsai, Katherine, et al. "Proxy methods for domain adaptation." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.
[2] Prashant, Parjanya Prajakta, et al. "Scalable Out-of-Distribution Robustness in the Presence of Unobserved Confounders." International Conference on Artificial Intelligence and Statistics, PMLR, 2025

**Note:** While my current rating is quite low, I am open to raising it based on the rebuttal. In particular, I would like the author to add discussion that clearly shows that they satisfy all the conditions required for Kugelgen's theorems. Also, I would like to see solid evidence that the current framing of the problem in terms of the causal graph in Figure 2 is required and something simpler will not work.

### Questions
Some questions are mentioned in the weaknesses.

1. The authors mention that each model was finetuned upto 10 epochs. I am curious what happens if you finetune only for 1 or 3 epoch? What about if you use LoRA? Does this sort of early stopping or LoRA training stop SFT from fitting so much to the spurious features?

### Soundness
2

### Presentation
2

### Contribution
2
