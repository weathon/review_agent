# Meta-Router: Bridging Gold-standard and Preference-based Evaluations in LLM Routing

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
In language tasks requiring extensive  human-model interaction, the inference cost of large language models (LLMs) can be substantial. To reduce expenses while preserving the quality of the responses, an LLM router selects among candidate models to balance between the expected response quality and the inference cost. A central challenge in router training is the accuracy and accessibility of reliable supervision. Gold-standard data, obtained from domain experts or benchmark labels, provide accurate quality evaluations of LLM responses but are costly and difficult to scale. In contrast, preference-based data, collected via crowdsourcing or LLM-as-a-judge systems, are cheaper and more scalable, yet often biased in reflecting the true quality of responses. We cast the problem of LLM router training with combined Gold-standard and preference-based data into a causal inference framework by viewing the response evaluation mechanism as the treatment assignment. This perspective further reveals that the bias in preference-based data corresponds to the well-known causal estimand: the conditional average treatment effect (CATE). Based on this new perspective, we develop an integrative causal router training framework that corrects preference-data bias, addresses imbalances between two data sources, and improves routing robustness and efficiency.  Numerical experiments demonstrate that our approach delivers more accurate routing and improves the trade-off between cost and quality. Illustrative code to reproduce our main experiment is available at https://github.com/yichistat/Meta-router.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of training LLM routers to balance response quality and inference cost. It proposes a novel causal inference framework, viewing gold-standard and preference-based evaluation data as a treatment assignment problem. The key insight is that bias in preference-based data corresponds to the conditional average treatment effect (CATE). The proposed Meta-Router framework corrects this bias via meta-learners (S-learner, T-learner, X-learner), incorporates propensity score weighting to address data source imbalances, and applies distributionally robust optimization (DRO) for routing robustness. Experiments on benchmarks (AlpacaEval, MT-Bench, MMLU, GSM8K) show improved cost-quality trade-offs compared to standard baselines.

### Strengths
**Originality**: The framing of LLM router training as a causal inference problem is novel and insightful; the connection between preference evaluation bias and CATE is a principled advancement. Creative application of meta-learners to this setting broadens the impact of causal inference in LLM routing.

**Quality**: The methodology is sound, with solid theoretical framing and systematic derivations. Experiments are thorough, covering benchmarks, model combinations, and ablations.

**Clarity**: The paper is well-written, with clear motivation, notation, and visualizations.

**Significance**: Addresses a highly relevant and practical problem. The framework is generalizable to different model pools and routing architectures.

### Weaknesses
**Limited Baseline Scope**: Compares mostly to classical methods (IPW, DR); recent LLM routing methods and stronger supervised baselines are needed.

**Scalability**: Only small model pools tested, unclear if methods scale to larger pools or what challenges emerge.

**Missing Error Analysis**: No detailed analysis of failure cases, e.g., when Meta-Router misroutes queries.

**Computational Overhead**: No empirical or theoretical analysis of the increased training cost for meta-learners and propensity score modeling, particularly for the X-learner.

### Questions
1. What is the computational overhead compared to baselines (training and inference)? Can you quantify the trade-off?

2. Can you empirically measure and report the actual bias in preference data vs. gold-standard benchmarks used?

3. How does performance scale with model pool size—do you have results for routing among 8–10 models?

4. Could you compare to modern LLM routing baselines, such as confidence-based or learned reward model routing?

5. Have you performed error analysis outlining types of queries or inputs where Meta-Router fails?

6. Why does T-learner outperform X-learner in Table 2 occasionally—are simpler meta-learners sometimes preferable?

7. How is out-of-distribution or universally poor query detection handled by the method?

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
4

### Summary
This paper presents a novel framework for LLM routing, specifically addressing the challenge of cost-effective routing between LLMs while ensuring optimal response quality. The core idea is to balance the expected quality of responses against the computational cost of using different LLMs. The paper introduces an integrative causal router training framework that combines gold-standard data with preference-based data, with the goal of reducing bias and improving routing efficiency and robustness. The framework is empirically validated using a healthcare benchmark, HealthBench, demonstrating improved efficiency in routing decisions.

### Strengths
1. The proposed method models the challenge as a causal inference problem, framing the task as correcting biases in preference-based data using causal meta-learners.

2. These meta-learners, specifically the R-learner and DR-learner, are used to estimate the Conditional Average Treatment Effect, which quantifies the discrepancy between gold-standard and preference-based evaluations.

### Weaknesses
The oracle integrative router in Section 3.1 assumes that the shift function (∆(q)) between the gold-standard and preference-based evaluations is known a priori. This assumption, while ideal in theoretical settings, is unrealistic in practical scenarios where the true shift function is unknown and must be estimated from data. Exploring semi-supervised learning or incorporating active learning strategies to iteratively estimate the shift function could make the approach more applicable.

### Questions
1. How does the method handle situations where the preference-based data distribution does not match the gold-standard data distribution, particularly in more complex or subjective queries?
2. The use of meta-learners such as R-learner and DR-learner is central to the paper. While the paper highlights their benefits, how does the method perform when faced with data heterogeneity or imbalances across large-scale datasets? Could the integration of more flexible or adaptive meta-learners improve performance further in such cases?

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
3

### Summary
This paper proposes Meta-Router, an LLM router training method that leverages both expensive, scarce gold data (GS) with cheap, scalable preference-based (PB) supervision. Recognizing that PB can be biased relative to GS, the authors frame the discrepancy between GS and PB as a causal shift, and use meta-learners like R-learner or DR learner to estimate it. The estimated shift is then incorporated into router training to leverage both data sources in a consistent manner. Experiments on HealthBench, using random forest and XGBoost, show higher efficiency gains than the baselines trained on pooled GS+PB or GS-only data across primary model usage ratios.

### Strengths
1. The paper explicitly identifies the bias in preference-based (PB) supervision, and provides clear motivation of framing the discrepancy between PB and GS data as a causal shift. Then it leverages meta-learning to estimate the bias, thereby aligning PB with GS evaluation.
2. The framework is derived through clear, step-by-step equations. The equations are coherent and easy to follow.
3. The experiments include multiple ablated variants, which effectively demonstrates the strengths of Meta-Router.

### Weaknesses
1. Both GS and PB scores are obtained from GPT5-mini. This would potentially reduce the real discrepancy between them and add confounding factors. Using a less capable LLM for PB may better reflect the bias . The paper also lacks validation of GPT5-mini GS scores against human judgements, especially in an expert-dependent domain.
2. PB scores are normalized to have same variance as GS, and training is restricted to overlapping supports. Normalization may be unstable when overlap is small. The paper mentions GS sample size, but the overlap size is unclear.
3. Experiments are done on a single domain, Health. Mixed-domain tests might be needed to assess whether the meta-learner generalizes well across domains.

### Questions
Please see weaknesses.

1. How would this method extend to more than two LLMs?

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
3

### Summary
This paper tackles a key challenge in LLM routing: balancing response quality and inference cost amid the constraints of two data types—scarce but accurate gold-standard (GS) data (from experts/benchmark labels) and scalable but biased preference-based (PB) data (from crowdsourcing/LLM-as-a-judge).
To address this, the work frames LLM router training within a
causal inference framework
: it treats GS and PB evaluations as two "treatment assignments" for query quality assessment, identifying PB data biasas the
Conditional Average Treatment Effect (CATE)
—the expected quality gain difference between GS and PB evaluations for a given query.

Experiments on the
HealthBench benchmark
(5,000 healthcare dialogues) use M_p (Gemini 2.5 Pro, high-cost/high-quality) and (Gemma 3 12B, low-cost/low-quality), with GPT-5-mini generating GS (rubric-based) and PB (preference) evaluations. Results show Meta-Router outperforms baselines (pooled GS- PB, GS-only, random routing) across GS sample sizes (n=100, 500, 1000), achieving higher efficiency gain (EG) by cutting M_p usage while preserving quality. R-learner-based Meta-Router is most robust, especially with scarce GS data.

The framework provides a principled way to integrate biased scalable data with accurate scarce data, advancing reliable LLM routing in high-expertise domains (e.g., healthcare). Future work will explore truncation-based extensions for non-overlapping GS-PB query distributions.

### Strengths
Novel Causal Framing for a Critical Practical Gap:
The paper addresses a longstanding tension in LLM routing—scarce but accurate gold-standard (GS) data vs. scalable but biased preference-based (PB) data—by innovatively framing the problem as causal inference. By treating GS/PB evaluations as "treatment assignments" and PB bias as the Conditional Average Treatment Effect (CATE), it links router training to rigorous semiparametric causal estimation (Imbens & Rubin, 2015), providing a principled solution to debias PB data rather than relying on ad-hoc fusion.

Robust and Flexible Meta-Router Framework:
The proposed Meta-Router is both theoretically grounded and practically adaptable. Its two-step design—CATE estimation via R/DR-learners (Nie & Wager, 2021; Kennedy, 2023) and debiased data fusion—enjoys the oracle property and supports off-the-shelf ML models (random forests, XGBoost) for nonparametric regression , . This flexibility allows it to handle heterogeneous query spaces (e.g., healthcare dialogues) and avoids over-reliance on specific model architectures.

Targeted Validation on High-Stakes, Expert-Driven Benchmark:
Experiments are conducted on HealthBench (Arora et al., 2025)—a 5,000-sample healthcare benchmark with expert-designed rubrics and physician consensus—addressing a critical limitation of prior routing work (e.g., using MMLU with objective answers). By testing on open-ended, professional tasks (healthcare dialogues) and using realistic model pairs (Gemini 2.5 Pro as M_p, Gemma 3 12B as M_p), the study validates Meta-Router’s relevance to high-stakes domains where quality-cost trade-offs matter most .

### Weaknesses
Lack of Practical Implementation Guidance:
The Meta-Router framework is presented up to its theoretical formulation but provides no actionable guidance for real-world deployment. For instance, it does not outline end-to-end workflows. This gap undermines its utility, as the paper addresses a problem with clear real-world relevance but fails to bridge the gap between theory and application.


Inadequate Experimental Scope:
The experimental design is overly limited. The main figures (Figure 1, Figure 2) only compare variants of the proposed Meta-Router against basic baselines (pooled GS- PB data, GS-only data) , with no comparison to state-of-the-art (SOTA) routing methods. This omission prevents readers from assessing Meta-Router’s competitive advantage. Furthermore, the absence of a "PB-only predictive router" baseline (a logical counterpoint to GS-only routing) leaves unaddressed how much value GS data adds to debiased PB data.

Missing Ablation Studies:
The paper lacks ablation experiments to isolate the impact of critical Meta-Router components. For example, it does not test: (1) whether CATE estimation (via R/DR-learners) is indispensable (e.g., comparing to simpler bias correction methods like linear scaling); (2) the necessity of data normalization (per Remark 1) by omitting this step and measuring performance degradation; or (3) how different nonparametric regressors (random forests vs. XGBoost) influence accuracy. Without these ablations, the contribution of individual components to Meta-Router’s performance remains unsubstantiated.

### Questions
1. Is "LMM judge" in Section 3.1 a typo for "LLM judge," and why was no alternative notation used to avoid symbol "m" overloading?

2. Why is the function h(q) in Equation (5) not explicitly defined, and how does this affect reproducibility?

3. Can the "GS–PB joint Data Generation Process" be simplified to reduce comprehension costs?

4. Does the paper provide any end-to-end workflow or hyperparameter guidance for Meta-Router’s real-world deployment?

5. Were empirical checks conducted for the propensity score positivity assumption, and how is performance affected by its violation?

6. Why are distinct PB bias sources not disentangled, and is there experimental proof that blurring them is acceptable?

7. Has Meta-Router been tested on non-healthcare domains or multi-model routing scenarios to verify generalization?

8. Why is there no comparison to SOTA routing methods or a "PB-only predictive router" baseline in experiments?

9. What ablation experiments are missing to validate the necessity of Meta-Router’s core components (e.g., CATE estimation)?

### Soundness
1

### Presentation
3

### Contribution
2
