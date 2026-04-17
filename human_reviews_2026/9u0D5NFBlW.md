# LO-ARMs++: Improving Learning-Order Autoregressive Models for Molecular String Generation

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Autoregressive models (ARMs) have become the workhorse for sequence generation tasks, because of their simplicity and ability to exactly evaluate  their log-likelihood. Classical Fixed-Order (FO) ARMs factorize high-dimensional data according to a fixed canonical ordering, framing the task as next-token prediction. While a natural ordering exists for text (left-to-right), canonical orderings are less obvious for many data modalities, such as molecular graphs and sequences. Learning-Order (LO) ARMs address this limitation, but their training relies on the optimization of an Evidence Lower Bound (ELBO), rather than on their exact log-likelihood. Therefore, FO-ARMs tend to remain advantageous. In this paper, we introduce LO-ARMs++, an improved version of LO-ARMs, to address this issue through incorporating several technical improvements. We introduce an improved training method called $\alpha$-$\beta$-ELBO, together with network architectural improvements. We demonstrate the general applicability of $\alpha$-$\beta$-ELBO, which yields improvement on the distribution learning metrics on both molecular graph and string generation. Moreover, on the challenging domain of molecular sequence generation, LO-ARMs++ match or surpass state-of-the-art results of Fixed-Order(FO) ARMs on the GuacaMol and MOSES SMILES benchmarks in terms of key metrics for distribution similarity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents LO-ARM++, an improvement over Learning-Order Autoregressive Models (LO-ARMs) for SMILES generation. The main contributions of the paper are:
- a tighter $\alpha$-$\beta$-ELBO training objective that adds maximum-entropy regularization on the variational order policy $q_\theta$.  $\alpha$ and $\beta$ are progressively decayed with a scheduler to implement an exploration-exploitation strategy to learn an optimal order policy.
- a stability fix that applies dropout to attention outputs instead of attention scores (which is sub-optimal for padded sequences).
- a prefix tokenization strategy that combines matched parenthesis pairs into single tokens.

LO-ARMS++ is tested on the GuacaMol benchmark, where it improves NLL and FCD over LO-ARM and performs comparably or better than fixed-order (FO) ARMs on distributional metrics (validity, uniqueness, novelty).

[1] Wang et al. Learning-order autoregressive models with application to molecular graph generation. ICML 2025

### Strengths
- The improvements over LO-ARM are tightly focused to resolve its shortcomings. Their choice is adequately motivated by ablations.
- The learned orderings are easy to interpret (although this was true already for LO-ARM).
- Performances are convincing (although not entirely transparent, see below).
- The paper is well written, easy to follow, with impacting visualizations.

### Weaknesses
- The paper reads like a well-engineered variant of LO-ARM rather than a new design or generative framework. In general terms, I'd positively value the contributions, but framed in ICLR's context, it has poor novelty value. 
- The issue above is paired with the scope of the contributions, which is confined to one benchmark (GuacaMol), a single task (unconditional generation) and a single molecule tokenization (SMILES). While prefix tokenization is justifiable only for SMILES, it is not clear whether the rest of the methodology (in particular $\alpha$-$\beta$-ELBO and the dropout attention fix) is a genuine improvement that transcends the specific representation to which it is coupled. For example, would it work with SELFIES tokenization? Again, without wider validation, all of these sound like ad-hoc hacks tied to the specific data, task, and representation.
- Framing LO-ARM++ as a "discrete diffusion-style model" is a stretch (there is no diffusion in this model, only denoising). I suggest to either rephrase the claim, or to provide an adequate justification as to why the authors believe that's the case.
- If I understood correctly, low-frequency tokens (and the SMILES that contain them) are filtered out only for LO-ARM++ but not for the competitors, meaning that they are not trained on the same data, nor on the same representation (e.g. the vocabularies are different). This appers to be acknowledged in the paper, however the unfair setup makes the actual improvement brought forth by LO-ARM++ hard to judge. 
- Although the impact of scheduling of $\alpha$ and $\beta$ is assessed, their default values are fixed withouth much justification. Could the authors provide a sensitivity analysis on these important parameters?

### Questions
See above.

### Soundness
3

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
This paper proposes LO-ARMs++, an improved training strategy for Learning-Order Autoregressive Models in SMILES-based molecular generation. The core idea is an α–β–ELBO objective that jointly enforces (1) maximum-entropy exploration on the variational order policy and (2) KL distillation from the variational policy into the model policy — with annealing schedules to gradually shift from exploration to exploitation. Two additional engineering contributions — prefix-style bracket tokenization and attention-output dropout — stabilize training. On GuacaMol distribution matching, LO-ARMs++ closes most of the gap to strong fixed-order models and even surpasses them on FCD, while producing interpretable “structure-first” generation orders.

### Strengths
- Clear motivation addressing a known pathology of LO-ARM — early collapse of the order distribution.

- Conceptually simple yet effective α–β–ELBO, with solid grounding in max-entropy variational inference / diffusion-like frameworks.

- Strong empirical gains, especially over vanilla LO-ARM, matching or exceeding fixed-order baselines on GuacaMol.

- Thorough ablations, showing the necessity of both α (entropy) and β (distillation) plus the improved dropout.

- Interpretability results are concrete and quantified — the model reliably learns “plan → refine” generation order.

- Engineering contributions are practical and reusable beyond chemistry (tokenization + attention dropout fix).

### Weaknesses
1. Single benchmark focus — experiments are limited to GuacaMol distribution-learning only (no goal-directed tasks or additional datasets).

2. Likelihood claims are partially qualified — FO vs LO NLLs are not fully comparable due to different length-conditioning setups.

3. No variance/CI reporting — results are averaged over 5 seeds but lack dispersion or statistical uncertainty.

4. Length prior is assumed rather than studied — impact on generation quality is unclear.

5. Some claims (e.g., “first diffusion-style model to reach this level”) may be slightly overstated without broader comparisons.

### Questions
1. Can you report mean ± std or confidence intervals for the main table, to quantify robustness?

2. How is the length prior estimated / smoothed, and what is its effect on FCD / uniqueness / novelty?

3. Have you tested LO-ARMs++ on goal-directed GuacaMol tasks, or any dataset beyond SMILES distribution modeling?

4. Could an asynchronous schedule for α and β be even better (e.g., entropy decays slower than distillation)?

5. Can you release code, especially for the bracket-prefix tokenizer and attention-output dropout patch?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a set of modifications to an existing discrete diffusion-based model for generating molecules.
Concretely, changes to tokenization, loss function and regularisation are the main focus points;
Experiments on the GuacaMol benchmark suggest favourable comparison compared to the base model (without modifications).

### Strengths
- Molecule generation is a relevant task, given the number of papers written on the topic.
 - The proposed modifications have not been published elsewhere.
 - Overall, the paper is clearly structured and well-written.

### Weaknesses
- The results in table&nbsp;1 indicate that the method is not competitve with existing methods.
   First of all, the way novelty is measured is not very meaningful and can be trivially increased (Renz et al., 2019).
   Better metrics for evaluating space exploration have been provided (e.g. Xie et al., 2023).
   On the other metrics, the model does not manage to match results from an LSTM model from 2019.
   Finally, the comparison ignores relevant modern molecule generation models that present even better results (e.g. Özçelik et al., 2024)
 - The proposed modifications are marginal at best.
   The prefix tokenization is a clean solution to the challenge with parentheses, but is limited to adding one special token.
   Similarly, the dropout is a nice workaround for a practical problem, but comes down to changing the order of layers slightly.
   Finally, the proposed $\alpha$-$\beta$-ELBO comes down to adding tunable scaling factors in the loss function.
   None of these modifications provide significant(ly new) insights.
   Furthermore, these modifications are very much tailored to LO-ARMs,
   making their general applicability questionable.
   Because these changes do not suffice to make LO-ARMs competitive with modern baselines,
   I fail to find much value for the general ML community.
 - It is not entirely clear why it would be necessary or beneficial to allow different orderings.
   After all, this work seems to report results that rely on canonicalized SMILES, which should have a fixed ordering anyway.
   On the other hand, even if the model is able to produce non-canonical SMILES, it does not seem to provide any advantage.
 - It is unclear whether the SMILES are canonicalised during training.
   On line 331 the authors suggest that the model is able to learn custom orderings,
   suggesting that the model has seen non-canonical SMILES during training.
   However, this does not seem to be mentioned explicitly.

### Minor Issues
 - AO-ARM acronym is used before the acronym (or the corresponding model) is introduced (line 219).

### Additional References
 - Özçelik et al. (2024). [Chemical language modeling with structured state space sequence models](https://www.nature.com/articles/s41467-024-50469-9). Nature Communications, 15(1), 6176.
 - Renz et al. (2019). [On failure modes in molecule generation and optimization](https://doi.org/10.1016/j.ddtec.2020.09.003). Drug Discovery Today: Technologies, 32, 55-63.
 - Xie et al. (2023) [How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules](https://openreview.net/forum?id=Yo06F8kfMa1). International Conference on Learning Representations.

### Questions
1. How does this method perform on the #circles metric (Xie et al., 2023) in comparison to other models?
 2. How does the method perform compared to state-of-the-art molecule generation methods?
 3. Are there any (new) insights in this work that could be useful for the more general ML community?
 4. Were all models trained on canonicalized SMILES? If not, which models were trained with arbitrary SMILES?
 5. What is the value of a model that is not sensitive to the ordering of SMILES, given canonical SMILES?
 6. How does the prefix tokenization affect other baseline models (e.g. LSTM)?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes improvements to learned-order autoregressive models which explore more generation orders and stabilize training, yielding better test-set log-likelihood and better distribution learning. They derive a tighter ELBO that unifies fixed-order and any-order autoregression, and add an entropy regularization term, with new terms weighted by hyperparameters $\alpha$ and $\beta$. Annealing large $\alpha$ and $\beta$ during training allows the model to initially explore different orders until eventually exploiting to optimize ELBO. The paper verifies that these changes do result in a less-greedy order policy. The authors also find that applying dropout to attention scores disrupts training, and instead apply dropout to the output of the attention layer. In experiments on distribution learning GuacaMol, LO-ARM++ outperforms LO-ARM in FCD and NLL, and is competitive with FO-ARMs. LO-ARM++ also finds interpretable generation orders.

### Strengths
The paper well-characterizes the learned generation order with informative figures.

Extensive ablations show how the proposed changes affect the resulting model's entropy and training stability.

The paper shows how its contributions generalize other works such as FO-ARMs and AO-ARMs.

### Weaknesses
Multiple tricks are employed to improve performance specifically on generating SMILES, which casts doubt on the claim of generalizable improvements over LO-ARMs. A new tokenization is required for LO-ARM to compete with fixed-order autoregressive models, as well as filtering out uncommon tokens. The experimental setup is also restricted to just canonical SMILES, but it is known that random SMILES orders boosts performance.

A pessimistic criticism is that the new $\alpha$-$\beta$-ELBO simply provides extra hyperparameters for achieving better performance than previous models.

### Questions
1. In Table 1, why does Transformer FO-ARM underperform LSTM ARM?
2. Why is test NLL reported as an upper bound, when FO-ARM provides exact NLL?
3. Why is FO-ARM NLL on padded SMILES not comparable to LO-ARM NLL? Can't you exclude the contribution of log-likelihood of pad tokens?
4. Is the annealing schedule expected to always provide a benefit, or is it dependent on the domain?
5. To support the claim that LO-ARMs are generally improved by the new regularization terms, there should be at least one other set of experiments on a different domain such as text or Sudoku, or even toy data on traversing graphs.
6. A body of work studying token-ordering in autoregressive / discrete diffusion models is not cited. These works mention experiments such as Sudoku which may be able to better demonstrate the utility of learned-order autoregression.

[1] Kim, J., Shah, K., Kontonis, V., Kakade, S., & Chen, S. (2025). Train for the worst, plan for the best: Understanding token ordering in masked diffusions. arXiv preprint arXiv:2502.06768.

[2] Bachmann, G., & Nagarajan, V. (2024). The pitfalls of next-token prediction. arXiv preprint arXiv:2403.06963.

nit-picking
1. There is a missing comma in the $D_{KL}$ in Equation 3.
2. Table 2: "SMILES strsings"
3. line 640: "LO-AMRs+"
4. line 647: "highlighy"
5. line 809: repeated sentence

### Soundness
2

### Presentation
2

### Contribution
2
