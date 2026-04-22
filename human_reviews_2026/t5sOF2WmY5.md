# Towards a Comprehensive Scaling Law of Mixture-of-Experts

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6

## Abstract
Mixture-of-Experts (MoE) models have become the consensus approach for enabling parameter-efficient scaling and cost-effective deployment in large language models. However, existing scaling laws for dense models are inapplicable to MoE models, which stems from three critical challenges: the multiplicity of influencing factors, their intricate coupling relationships and the non-monotonic nature of their performance impacts. They collectively necessitate a fine-grained investigation into MoE-specific scaling laws. In this work, we perform a systematic decomposition of MoE settings, identifying five key factors that influence model performance from both size and structural perspectives (data size ($D$), total model size ($N$), activated model size ($N_a$), number of active experts ($G$) and the ratio of shared experts ($S$)). Specifically, we design $450$ controlled experiments to characterize their marginal effects, ultimately constructing a comprehensive and precise joint MoE scaling law that considers all essential factors. Furthermore, we derive the theoretically optimal and practically efficiency-aware optimal configurations for $G$, $S$ and $N_a/N$ with detailed analyses. Our results demonstrate that the optimal settings for $G$ and $S$ are independent of both the model architecture and data size. With the scaling of $N$, the optimal activation parameter ratio of $N_a/N$ becomes sparser. Our proposed MoE scaling law could function as an accurate and insightful guidance to facilitate future MoE model design and training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The study extends existing Mixture-of-Experts (MoE) scaling law research to encompass the architectural choices present in current frontier open models. It decomposes MoE configurations into five variables influencing performance: data size $(D)$, total model size $(N)$, activated model size $(N_a)$, number of activated experts $(G)$, and ratio of shared experts $(S)$ in the total activated experts.  
Using $446$ controlled experiments, the authors quantify the marginal effects of these factors and construct a joint scaling law incorporating all five.  
From the fitted scaling laws, analytical expressions are derived for the theoretically optimal and efficiency-aware values of $G$, $S$, and $N_a/N$, showing that the optimal $G$ and $S$ remain invariant with respect to architecture and data scale, while the optimal activation ratio $N_a/N$ decreases as total model size increases.  
These optimal configurations are compared against existing frontier open MoE language models.  
The resulting formulation provides a quantitative framework for predicting MoE performance and configuring model sparsity.

This work is more directly applicable for analysis and design of modern MoE LLMs, seeing as it directly deals with the number of activated experts or the number of shared experts. It also compares directly to existing open MoEs, and shows partial agreement between their predicted optimal configurations and choices made in open models - and where there is disagreement, elaborates on why that might be.

The hypothesis space considered in this work is larger than in the existing work, and as such can directly compare the quality of fit with other existing works by reproducing them on the same data - although I have reservations as to whether the comparisons are useful and what the takeaways are (more in Weaknesses/Questions).

### Strengths
1. **Wide scope** — compared to existing MoE scaling law studies, the formulation captures nearly all architectural and scaling choices used in frontier MoE LLMs.

2. **Progressive hypothesis construction** — development proceeds systematically from dense scaling laws (Chinchilla; Hoffmann et al., 2022) to a joint law over all five variables $(N, D, N_a, G, S)$, with validation at each stage.

3. **Comprehensive experimental design** — 446 controlled experiments are conducted to isolate and quantify the marginal effects of each factor.

4. **Analytical interpretation** — theoretical expressions for the optimal $G$, $S$, and $N_a/N$ are derived from the fitted law and compared with contemporary large MoE models.

5. **Transparent reporting** — all experimental configurations are listed, though the absence of of final loss values, as well as the fitting procedure, greatly hinders reproducibility.

### Weaknesses
I have several significant concerns that currently prevent me from recommending acceptance. I would welcome the authors' responses to these points, as they could substantially affect my evaluation.

### 1. Clarity needed on the fitting procedure

The description of the fitting methodology in the paper requires clarification. The current text states:

>To determine the specific value of hyperparameters in Eq. 11, we implement all 446 experiments that encompass diverse configurations of N, D, Na, G and S. The corresponding hyperparameter values are provided in Table 2 in Appendix C.1. Next, we evaluate the fitting performance of our MoE scaling law in Figure 5, where the satisfactory fitting performance demonstrates the advantage of our MoE scaling law compared to others. Notably, for the sake of fairness, during the comparison, the baseline MoE scaling laws were first re-fitted with hyperparameters using the same data points, followed by prediction.

This paragraph is difficult to comprehend. I would appreciate more detailed information on:
- The specific fitting algorithm or optimization method used
- How the validation set was utilized in the fitting process
- The meaning of "baseline MoE scaling laws were first re-fitted with hyperparameters using the same data points, followed by prediction"

My main concern is whether the validation set was used for final model selection. Given that the proposed model has 12 parameters (compared to 7 in Ludziejewski et al. (2024), 10 in Abnar et al. (2025), and 9 in Ludziejewski et al. (2025)), if the training set was used to generate candidate fits and the validation set was used for selection, models with more parameters may have an inherent advantage due to increased flexibility.

Additionally, while mean RMSE and visual comparisons are provided, I suggest including additional evaluation metrics, as recommended in recent work such as Besiroglu et al. (2024).

### 2. Clarification needed on validation set characteristics

I would appreciate explicit information about whether the validation set represents interpolation or extrapolation relative to the training datapoints. This distinction is important for interpreting the results. While this is possible to do based just on the appendix, it requires actual data analysis.

### 3. Suggestion for distributional comparison with prior work

The current comparison evaluates prior scaling laws on data that may be out-of-distribution for those functional forms. To strengthen the paper, I suggest including comparisons using data closer to the original experimental distributions of each prior work (Ludziejewski et al., 2024; Abnar et al., 2025; Ludziejewski et al., 2025). This would provide a fairer assessment of whether the proposed hypothesis space offers genuine improvements.

### 4. Minor concern regarding experimental controls

Following Porian et al. (2024), I note that the experimental design does not appear to adjust for batch size or learning rate across different scales, which could introduce bias, particularly at smaller scales. While this is a minor point, addressing it would strengthen the experimental foundation.

---

### References

- Kaplan, J., et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).

- Hoffmann, J., et al. (2022). [Training compute-optimal large language models](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf). NeurIPS 2022.

- Ludziejewski, J., et al. (2024). [Scaling Laws for Fine-Grained Mixture of Experts](https://proceedings.mlr.press/v235/ludziejewski24a.html). ICML 2024.

- Porian, T., et al. (2024). [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/b6341525cd84f3be0ef203e4d7cd8556-Paper-Conference.pdf). NeurIPS 2024.

- Besiroglu, T., et al. (2024). [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).

- Abnar, S., et al. (2025). [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370). ICML 2025.

- Ludziejewski, J., et al. (2025). [Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient](https://proceedings.mlr.press/v267/ludziejewski25a.html). ICML 2025.

### Questions
### 1. Please describe the fitting procedure in detail.

The strongest description here would be: release the code + your data in an anonymous repo, together with fitting of other existing scaling laws. It would mean that you provide a script that starts with raw data, define the search grid (which I assume you did, following e.g. [Hoffmann et al. (2022), Section D.2 in appendix](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)), choose the optimiser, define the logic of choosing the best fit, similar to [Wortsman et al. (2024)](https://arxiv.org/abs/2405.20204). It is clearly a useful thing for the research community, allowing extended, independent analysis, exemplified by [Besiroglu et al. (2024)'s Chinchilla replication scaling](https://arxiv.org/abs/2404.10102). This is especially important for the ML community going forward, as there are more and more scaling laws works, and comparing them directly will get more and more difficult without the actual data and fitting procedures used.

### 2. Please make the comparison against existing scaling laws fair.

It is my impression that this paper uses comparisons between [Ludziejewski et al. (2024)](https://proceedings.mlr.press/v235/ludziejewski24a.html) and [Abnar et al. (2025)](https://arxiv.org/abs/2501.12370) to convince the reader that functional form proposed by these respective works is inferior to one proposed in this work. I think that's misleading. Seeing as the full training and validation sets in this work have significant variations of $S$ and $N_a / N$, limiting them to ranges similar to the works you compared to would be a more direct comparison between the functional forms. Right now, the experiment only implies that the previous scaling laws do not work well for high $S$ and/or $N_a / N$, which is not surprising, nor does it shed new light on the existing work - they were not designed for these conditions. This can be partially achieved by limiting the train and validation datasets to regions to which the respective scaling laws were designed for - but an issue might come up regardless, if there are not enough datapoints after such filtering.

---

## References

- Kaplan, J., et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).

- Hoffmann, J., et al. (2022). [Training compute-optimal large language models](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf). NeurIPS 2022.

- Ludziejewski, J., et al. (2024). [Scaling Laws for Fine-Grained Mixture of Experts](https://proceedings.mlr.press/v235/ludziejewski24a.html). ICML 2024.

- Wortsman, M., et al. (2024). [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2405.20204).

- Porian, T., et al. (2024). [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/b6341525cd84f3be0ef203e4d7cd8556-Paper-Conference.pdf). NeurIPS 2024.

- Besiroglu, T., et al. (2024). [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).

- Abnar, S., et al. (2025). [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370). ICML 2025.

- Ludziejewski, J., et al. (2025). [Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient](https://proceedings.mlr.press/v267/ludziejewski25a.html). ICML 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes a scaling law for Mixture of Experts decoder-only Transformers, which unifies the following factors:
- number of training tokens ($D$),
- total number of model parameters ($N$),
- number of active parameters per token ($N_a$),
- number of active experts per token ($G$),
- fraction of shared experts among all active experts ($S$).
The paper walks through the derivation of the scaling law, starting with the Chinchilla-like Scaling Law (dependent on $N$, $D$) and expanding it by considering the impact of: $N_a$, $G$, and finally, $S$. The quality of the fit is compared to two earlier works ([3, 4]). Based on the final law, some key implications are derived:
- The optimal number of activated experts $G$ (which is constant)
- The optimal ratio of shared experts $S$ (which is constant)
- The optimal sparsity (which increases as models grow).

### Strengths
- Large number and variety of performed experiments
- Principled, step-by-step description of the process of deriving the scaling law, including fitting the scaling laws for intermediate formulas
- Clear presentation of the key implications (section 5)

### Weaknesses
- The paper does not specify whether embedding or non-embedding active / total parameters are used 
- There is no bootstrapping involved, i.e. no measure of uncertainty for the fitted coefficients in the Table 2
- All experiments use the same learning rate, which is not uncommon in scaling laws research, however, other approaches have been proposed [2, 3]
- Lack of evaluation on downstream tasks
- Omission of some technical details (RMSNorm or LayerNorm, vocab size, positional encodings used, etc.), leading to reduced reproducibility. Supplying the code would contribute to improving this aspect.
- Limited information on the specifics of the fitting procedure (what algorithm was used to optimize the coefficients, what was the grid of initial values in fitting, etc.)
- It is not clear how many experts per layer there are in the models: line 146 says "up to 256 experts", however table 1 claims 32 experts for all models.
- The notion of "Efficiency awareness" is not explained clearly (see questions)
- If I understand correctly, the presented scaling law does not consider scaling of $E$ (expansion rate as per [3]). This is a clear limitation which constrains the applicability of the results (see [2]) and __needs__ to be mentioned in the limitations section (and perhaps in the table 5 as well).

Other minor remarks:
- C.1 title should read "Numerical" not "Numberical"
- Line 492 incorrectly claims the appendix F contains information about the implementation details

### Questions
- OLMoE (Muennighoff et al., https://arxiv.org/pdf/2409.02060v1) fail to find convincing evidence for employing shared experts. Do the authors have an idea how to reconcile that result with the results in this work?
- Can the authors clarify the usage of embedding / non-embedding parameters?
- It seems a standard practice (Scaling Laws for Neural Language Models, Kaplan et al., https://arxiv.org/pdf/2001.08361; Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient, Ludziejewski et al., https://arxiv.org/pdf/2502.05172v2) is to keep constant depth-to-width (dm / l) ratio for all models, however here the ratio changes (Table 1). Is there a principled reason for that?
- What are the "associated costs" mentioned in line 418?
- What is the applicability of the results considering the lack of scaling w.r.t. $E$ (expansion rate, see [3])?

References:
1. Scaling Laws for Neural Language Models, Kaplan et al., https://arxiv.org/pdf/2001.08361
2. Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient, Ludziejewski et al., https://arxiv.org/pdf/2502.05172v2
3. Scaling Laws for Fine-Grained Mixture of Experts, Krajewski et al., https://arxiv.org/abs/2402.07871
4. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models, Abnar et al., https://arxiv.org/abs/2501.12370

### Soundness
2

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
4

### Summary
This paper proposes a scaling law for MoE language models, extending scaling relations to include five interacting factors: total model size N, data size D, activated model size Na, number of active experts G, and shared-expert ratio S. The authors empirically derive a joint loss function that accurately predicts performance across a wide range of MoE configurations and yields optima such as  Gopt ⁣≈ ⁣7, Sopt ⁣≈ ⁣0.3, and decreasing optimal activation ratio.

### Strengths
The paper presents one of the most extensive empirical studies of MoE scaling , systematically analyzing five interrelated factors—
D,N,Na,G,S

The authors methodically vary one factor at a time, isolate marginal effects, and then assemble a joint scaling law.

The implications are interesting, leveraging the quadratic equation model to obtain optimal G and S and that the N_a/N decreases with total model size. The derived optima correspond well to empirical settings in industrial MoEs

The model fit is very good.

### Weaknesses
The main findings are about S and G which have relatively little impact for the loss. Due to small impact on S, it is questionable if they should be included in the formula, making it seem a little over-engineered.

The paper lacks the validation of the scaling law on a very large scale (e.g. much larger model that was not used to get the scaling law)

No analysis of variance across random seeds. Given the many fitted coefficients, it’s unclear whether the final joint law is over-parameterized or robust to noise.

A little confusing notation, G originally described granularity

### Questions
You include the parameters of the number of acivated experts but not the number of experts and no granularity. How did you make choices of parameters to appear in your scaling law?

How do the fitted exponents parameters (say α,β) compare with those in other works, Krajewski, Ludziejewski, Chinchilla?

### Soundness
3

### Presentation
3

### Contribution
3
