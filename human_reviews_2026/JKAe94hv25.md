# PRISM: Pareto-Responsive Iterative Sampling with DPO for Multi-objective Planning

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Many planning-style applications of large language models are inherently multi-objective. Beyond correctness, users care about efficiency and the avoidance of irrelevant or unsafe actions. Yet most alignment pipelines optimize a single scalar reward, which hides trade-offs and offers little control when secondary objectives have uncertain or deployment-specific weights. We present PRISM, a Pareto responsive framework that integrates Direct Preference Optimization. PRISM adds three components designed for offline, several convergence toward balanced solutions. First, it uses golden comparisons that isolate per-objective preferences. Second, it computes attention-style weights from deficiency diagnostics that combine loss and gradient information. Third, it applies Pareto guided sampling that orients preference pairs by cosine alignment with the current weight direction.This loop performs common-descent updates for a vector of objective deficiencies and stops at a certificate of first-order Pareto stationarity. It removes the need for online reinforcement learning, reward sweeps, or families of specialist models. On six benchmarks in question answering, coding, and mathematical reasoning, PRISM improves accuracy over strong baselines while simultaneously reducing latency and step count and driving off-domain actions to near zero. PRISM provides a principled and compute efficient recipe for robust multi-objective alignment of LLM-based planners.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose PRISM, an offline preference-based fine-tuning framework that finds a balanced policy accounting for several objectives simultaneously, without requiring multiple specialized models or explicit weight tuning by the user. PRISM builds on Direct Preference Optimization (DPO) but extends it to the multi-objective setting in a principled way. 

(1) It generates golden comparison pairs that isolate individual objectives – specifically, they collect pairs of model outputs where one is clearly better on one objective but roughly equal on others. These serve as clean training signals for each criterion (e.g. a pair where solution A is correct but longer vs solution B is incorrect but shorter isolates the accuracy objective). 

(2) The framework computes a dynamic weighting over objectives using deficiency diagnostics. For each objective, it measures how often the model prefers the worse outcome in those golden comparisons and also how large the gradient updates are for that objective. Objectives in which the model is performing poorly and finds hard to improve (high loss and high gradient norm) are assigned higher weights. This is done via an attention-like softmax weighting of objectives, which continuously updates during training – essentially telling the model “focus more on objectives you’re currently deficient in.” 

(3) PRISM employs Pareto-guided sampling of preference pairs: when sampling training comparisons from a large pool, it biases selection toward those that align well with the current weight vector direction in objective space. It uses a cosine similarity criterion to pick or orient each preference pair so that the chosen “better” answer is the one that moves the model’s policy in the direction that improves the weighted combination of objectives. This ensures each update step is pushing the model toward a Pareto-optimal trade-off rather than oscillating or favoring one objective at the extreme. The training loop iteratively fine-tunes the model with these weighted, oriented preferences and stops when no further reweighting can simultaneously improve all objectives (reaching a first-order Pareto stationary point).

### Strengths
Empirical results on six benchmarks (spanning question answering, coding tasks, and mathematical reasoning) show that PRISM can significantly improve the primary metric (accuracy or success rate) while also reducing secondary costs like the number of steps, execution latency, and the incidence of off-domain or unsafe actions. For example, compared to strong baselines (including single-reward optimized models), PRISM achieves higher solution accuracy and drives undesirable behaviors (such as irrelevant tool calls or hallucinations) nearly to zero, all in a single fine-tuned model. 

It thereby produces a set of policies along the effective frontier of the trade-off curve without needing multiple models or online RL. 

The framework is novel in that it provides a general, data-driven way to balance objectives: the deficiency-based weighting is an innovative idea to adapt training focus, and the use of golden pairs plus vectorized updates steers the model towards a well-balanced solution. A noteworthy aspect is that PRISM avoids the inefficiency of previous multi-objective approaches that required sweeping through reward weightings or training conditional policies – instead, it finds one policy that is inherently balanced.

### Weaknesses
One potential limitation is the complexity of generating and evaluating the comparison data: the approach assumes access to a reward model or evaluators for each objective and a procedure to produce diverse candidate outputs, which could be non-trivial for new domains. I hope to hear a extensive discussions about this matter.

as with any multi-objective tuning, the final trade-off point might reflect the particular choice of deficiency weights and stopping criteria, which might need calibration for different applications.

Also, please provide more detailed environment on how GPU hours has been logged.

I am willing to increase the score if questions above are well treated.

### Questions
Discussed in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
As human preference should be managed considering the multiple aspect, multi objective DPO has been studied as one of prominent researches recently. This paper propose pareto responsive iterative sampling with DPO, or PRISM, to align multi objective preference for LLM finetuning. PRISM suggests (1) plan generation and golden comparisons by considering the differences between plans, (2) deficiency-based adaptive weighting, which is a normalized softmax value from loss and gradient and (3) Pareto-guided training and adaptive sampling to jointly optimize multiple objectives.
The method avoids reinforcement learning and multiple specialized models, instead converging to an approximate Pareto-stationary solution through iterative preference optimization.

### Strengths
- The deficiency-based adaptive weighting and sampling may offer a principled mechanism for balancing competing objectives.
- consistent gains in accuracy.

### Weaknesses
- The method is too complicated to achieve the objective, meaning not knowing which treatment did what. There is also a possibility of conflict between objectives. I checked table3 or ablation study, but only performance does not fully explain why each treatment contributed to model performance in that way.
- So many hyperparameters are additionally required, - \gamma, \epsilon, and \tau.
- It reguires the gradient norm, which cause another computation complexity.
- The paper does not test whether adjusting objective weights \w leads to predictable trade-offs among objectives.

### Questions
- It seems golden comparison (or pair selection for each objective) is so important to provide clean feedback. However, how can authors be sure the selected samples provide exactly true learning signal? 
- Pareto optimization for multi objective DPO is not a novel. What is the contribution of authors in the viewpoint of the optimization?
- One of the problems of multi objective DPO is that each objective is not fully independent, although each objective is modeled as independent. I suppose this gap would surely affect sample selection per each objective and weighting process. How the authors think about it? or can it be solved?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a Pareto-responsive framework for multi-objective planning in direct preference optimization (DPO) of large language models. The method identifies golden pairs, sample pairs where one objective improves with remaining others, and uses their DPO loss values and gradient norms to derive preference signals represented as weights on a simplex. These weights are then used to adjust the sampling probabilities of training examples, enabling adaptive multi-objective DPO updates.

### Strengths
* Most LLM alignment methods rely on a single scalar reward, so addressing multi-objective optimization is an important and meaningful direction.
* Employing a Pareto-based approach to handle multi-objective trade-offs is a reasonable and well-motivated choice.

### Weaknesses
* The paper assumes that, among multiple rewards, $O_1$ serves as the primary objective and $O_n$ measures a hard constraint. It is unclear whether such an assumption is necessary. In particular, hard constraints may not always be representable as a single scalar, which could limit the generality of the framework.

* When defining golden pairs, the paper requires that non-target objectives remain approximately unchanged. A deeper discussion would be helpful on whether this condition is important. For example, why are pairs that improve multiple objectives simultaneously excluded from being considered golden pairs?

* It is unclear whether golden pairs are expected to be evenly distributed across all objectives. If so, obtaining them for each objective would require separate sampling efforts. Does the method repeatedly sample until a certain balance or target ratio is reached?

* The DPO loss in Section 3.2 differs from the original formulation in Rafailov et al. (2023), for example, by omitting the reference policy term. It is unclear whether this is a notational simplification or a methodological change. If it is the latter, the paper should explicitly justify and discuss the implications of this modification.

* In Section 3.2, the method derives preference signals using the weighted sum of the DPO loss value and gradient norm. The rationale for this choice should be elaborated, both theoretically and empirically. In particular, a sensitivity analysis on the parameter $\gamma$ would be valuable to assess robustness.

* The paper introduces several hyperparameters, e.g., $\Delta_{min}^i, \delta_j, \gamma, \lambda, \tau, \beta$, but does not specify their values and provide any analysis regrading their effects on performance.

* The writing could be improved for clarity. For example, Section 3.3 presents a sequence of equations without clear statements highlighting the key claims. It would be helpful to include concise statements (e.g., Theorem, Proposition, ...), and algorithmic description outlining the overall PRISM method.

* Although multiple models and datasets are used in the experiments, their sources and configurations are not properly referenced and described.

* Figure 2 in Section 4 is difficult to interpret, and the explanation in Section 4.4 does not clearly convey the intended meaning. A more detailed and accessible description is needed to help readers understand the figure's significance.

### Questions
Please provide the response on the points in Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a framework named PRISM, a preference fine-tuning framework that jointly improves accuracy, efficiency, and error avoidance. It also introduces deficiency-aware weighting and Pareto Pareto-guided sampling mechanism.

### Strengths
The proposed idea seems to be novel and addresses important issues in multi-objective planning.

### Weaknesses
The paper is readable but not reader-friendly. Some questions remain to be answered. Please see the questions.

### Questions
**Questions**

Q1. Important related work Panacea [1] needs to be introduced and compared with the proposed method in terms of multi-objective optimization.
[1] Zhong, Yifan, et al. "Panacea: Pareto alignment via preference adaptation for llms." Advances in Neural Information Processing Systems 37 (2024): 75522-75558.

Q2. Are composite score, efficiency and error-avoidance only aspects in multi-objective planning? If not, what other aspects need to be considered besides them?

Q3. What happens when there is no pair of plans satisfying the golden comparisons among the initially generated plans?

Q4. It is better to use the same notation or the same representation in Section 3.1 and Figure 1. The value r is suddenly introduced in Figure 1, while Section 3.1 explains the reward function $O$. $r$ in r represents the same $r$ in Section 3.2?

Q5. Scalability with respect to $n$ is questionable. 

Q6. How to select $\gamma$ and $\beta$? Is the proposed method robust to such hyperparameters? 

Q7. The proposed method includes several additional components, including additional gradient and loss computations. Therefore, the proper computational cost analysis should be discussed. 

Q8. The paper does not use Equation numbers and repeatedly defines the same ones, such as $\bf{a}$ and $\bf{w}$, in Sections 3.2 and 3.3. Why not use an equation number and refer to it to avoid confusion?

Q9. In Section 3.3, explicit expression $\nabla _{\theta}$ instead of $\nabla$ would be helpful.

Q10. Do the authors intend to present their code? 

Q11. Why not elaborate on the last equation in Section 3.3 for the sake of readers at least in the Appendix?

**Minor Comments**

C1. In Figure 2, a text description overlapped with a dashed line and reducing the readability.

C2. Learning rate $\eta$ is not explicitly defined.

### Soundness
3

### Presentation
2

### Contribution
2
