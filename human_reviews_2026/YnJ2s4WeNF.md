# Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
While scaling laws for Large Language Models (LLMs) traditionally focus on proxy metrics like pretraining loss, predicting downstream task performance has been considered unreliable. This paper challenges that view by proposing a direct framework to model the scaling of downstream accuracy from the training budget. We demonstrate that for a fixed token-to-parameter ratio, a simple two-parameter scaling law accurately describes this relationship. Our findings are validated by experiments on models with up to 17B parameters trained on up to 350B tokens, showing that downstream capabilities scaling can be described using a scaling law. Furthermore, we extend this framework to extrapolate and predict accuracy of target model with up to 6.7x larger training budget based on a set of smaller experiments. We release a complete list of model losses and downstream evaluation results at various different scales to support reproducibility and encourage future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper investigates the power of scaling laws in language models to directly predict downstream performance from models' parameter count and dataset size.  Using a functional form without irreducible error $-log(Q)=A/N^\alpha + B/D^\beta$ the authors fit scaling laws for 48 distinct training budgets and 5 token-to-parameter rations. Then they validate the predictive power of fitted scaling laws using error estimation on the held-out points. The results show that the direct approach of predicting downstream performance is a promising direction in exploring new ways to accurately and reliably predict language models' performance at large scale.

### Strengths
- One-stage approach to to directly predict downstream performance of LLMs from N and D.
- Two different approaches to downstream task modeling - BSNL and a simple power-law relationship.
- Authors validated their predictions using error on held-out points.
- Works for different downstream metrics: pass@k, accuracy etc.
- Baselines like two-stage approaches that are commonly used (FLOPs-to-NLL and then NLL-to-accuracy).
- Direct downstream prediction is more reliable and accurate than two-stage approaches.

### Weaknesses
- No code or data currently available which makes the reproduction and independent verification of the authors' claims impossible. The authors though promise to release the model losses and downstream evaluation results.
- Better performance of one-stage approach compared to two-stage one on individual benchmarks can be due to the fact that individual task performance may not be correlated to the specific downstream task performance (which also depends on the the validation set) and not only compounding errors from different fits (please see the questions section for more info).

### Questions
### Questions

- Are the checkpoints well tuned? The authors mentioned that they used optimizer setup from [1] as well as  $\mu P$ for hyperparameter transfer but they don't mention any hyperparameter tuning or at least their motivation to use specific hyperparameters.
- In [2] authors note that it's hard to predict single benchmark performance with their two-stage approach reliably while the average across 17 benchmarks can be predicted quite well. Using the one-stage approach described here, can the average of multiple downstream task accuracies be reliably predicted?
- In two-stage approaches (Fig. 7) what specific proxy metric was used?
- In table 9, we see high $R^2$ and RMSE for different proxies and downstream tasks, which should prove that all proxy metrics are strong predictors for downstream performance, while in the Fig. 7, fits with higher RMSE and $R^2$ do not necessarily have the best predictive power (e.g. in Fig. 7a BNSL has higher $R^2$ and RMSE than TwoStage-Logistic and yet TwoStage-Logistic has lower MAE and MRE on held-out points).  So does the statement *"most proxy metrics demonstrate strong predictive power for downstream task performance"* still hold?
- Why modelling downstream performance without irreducible error? In principle, yes it is possible to have 100% accuracy on a downstream task, however, in practice it is known that the real benchmark data can have ill-posed, mislabeled or ambiguous questions [3].


### References

1. Gunter, Tom, et al. "Apple intelligence foundation language models." _arXiv preprint arXiv:2407.21075_ (2024).
2. Gadre, Samir Yitzhak, et al. "Language models scale reliably with over-training and on downstream tasks." _arXiv preprint arXiv:2403.08540_ (2024).
3. Vendrow, Joshua, et al. "Do large language model benchmarks test reliability?." _arXiv preprint arXiv:2502.03461_ (2025).

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
This paper proposes to directly predict downstream task performance from training budget. The authors run experiments across a variety of benchmarks and compute budgets to show that a simple scaling law equation can model this relationship and outperforms prior work.

### Strengths
- wide coverage of benchmarks and types of tasks, various compute budgets (both scale and TPR)
- simplicity of approach, taking into account the nature of these benchmarks (e.g. S-shaped)
- results demonstrate some extrapolation to larger compute budgets
- compares directly with prior works (two stage approach)

### Weaknesses
- lack of motivation for practical usage of this compared to standard scaling laws
- scaling laws are often used to justify design decisions (e.g. architectural or dataset choices) - there is a lack of these alternatives and showing that the scaling laws preserve the ordering of the "better" design decision

### Questions
Irreducible term may be theoretically unnecessary, but benchmarks often have incorrect labels (e.g. MMLU). How would you account for this?

Could you better motivate why direct scaling laws to evaluation are practically more useful than perplexity alone?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes to directly model task accuracy as a function of compute budget, rather than relying on the usual two-stage approach where downstream performance is predicted through proxy metrics like pretraining loss, which often makes such predictions unreliable. The authors introduce a simple two-parameter scaling model under a fixed token-to-parameter ratio and show that it fits downstream performance well. They validate the framework on models up to 17B parameters trained on 350B tokens, and further demonstrate its ability to extrapolate and predict accuracy for models trained with up to 6.7× larger compute budgets.

### Strengths
The paper is clearly written, and I appreciate the measured and non-sensational tone throughout. The proposed accuracy scaling law fits the evaluated metrics well, showing strong consistency across scales. I also like that the paper carefully compares the one-stage approach to the traditional two-stage setup, showing lower MAE, MRE, and higher R² for the proposed method.

### Weaknesses
Please see Questions section.

### Questions
1. What do you think about applying the scaling law fits to predict a single aggregated metric, such as the average downstream accuracy across a broad set of tasks? Scaling laws are often used to understand and guide the training of generalist models that need to perform well across many tasks. Would your proposed fits extend to that setting, and how would you think about aggregating or weighting the different task metrics?
2. The scaling law fits for all benchmarks in the paper look very good, and I appreciate the mention of cases where non-monotonic behavior makes direct fitting difficult. I’m curious to know in what other situations this direct fitting approach might fail beyond the ones you’ve already discussed. Are there additional examples or patterns you noticed where the relationship breaks down?
3. Knits: L182 & L187 “BSNL” should be corrected to “BNSL.”
4. References that are relevant and also can be discussed in the paper:
   - Li et al. (2025) - “(Mis)Fitting: A Survey of Scaling Laws”
   - Mayilvahanan et al. (2025) - “LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws”

### Soundness
4

### Presentation
4

### Contribution
3
