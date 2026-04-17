# Human-in-the-Loop Adaptive Optimization for Improved Time Series Forecasting

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Time-series forecasting models often produce systematic and predictable errors, even in critical domains such as energy, finance, and healthcare. We introduce a novel post-training adaptive optimization framework that improves forecast accuracy without retraining or architectural changes. Our approach adds a lightweight model-agnostic correction layer that automatically finds expressive output transformations optimized by reinforcement learning, contextual bandits, or genetic algorithms. Theoretically, we prove the benefit of an affine correction and quantify the expected performance gain together with its computational cost. The framework also supports an optional human-in-the-loop component: domain experts can guide corrections using natural language, which is parsed into actions by a language model. Across multiple benchmarks (e.g., electricity, weather, traffic), we observe consistent accuracy gains with minimal computational overhead. Our interactive demo (link) showcases the usability of the framework in real time. By combining automated post-hoc refinement with domain-expert corrections to the base forecasting model, our approach offers a lightweight yet powerful direction for practical forecasting systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a novel post-training adaptive correction framework designed to enhance the accuracy of time series forecasting models without retraining or architectural modification. A key innovation is the integration of a Human-in-the-Loop component, enabling domain experts to guide corrections through natural language feedback. These human instructions are parsed by a large language model into executable transformation actions that are then validated and optimized within the same adaptive pipeline. The paper provides both theoretical foundations (affine correction guarantees lower MSE, Theorem 1) and empirical analyses across multiple benchmarks, demonstrating consistent accuracy improvements with minimal overhead.

### Strengths
1. The framework is fully model-agnostic, compatible with any forecasting architecture without altering parameters or requiring retraining. This plug-and-play property makes it directly usable in production systems.

2. HITL component represents a novel integration of human expertise via natural language, supported by a safe validation mechanism that prevents code injection and ensures feedback quality (Algorithm 2, Fig. 5). This bridges the gap between machine optimization and domain expert intuition, offering genuine practical utility.

### Weaknesses
1. The paper claims low computational overhead, yet the runtime analysis (Table 3) only compares optimization time to base model training cost for a few horizons on a single dataset (ETTh1). There is no evaluation on larger models or streaming scenarios where real-time adaptation is claimed to be possible. Moreover, the runtime excludes the LLM inference and validation costs required for parsing human feedback, which could be nontrivial in practice.

2. Safety and robustness of LLM-based feedback translation are insufficiently validated. Algorithm 2 specifies a validation routine, but the paper never clarifies the criteria for safety or correctness. There is no discussion of how the system handles ambiguous, adversarial, or syntactically valid but semantically incorrect feedback. This omission raises concerns about reliability in real deployment.

3. The role and quantifiable contribution of the Human-in-the-Loop component are inadequately analyzed. The authors present qualitative examples but provide no quantitative ablation isolating HITL performance relative to the automated optimization baseline. It is therefore unclear how much of the reported gains are due to algorithmic improvements versus human intervention. Without such evidence, the practical value of the human feedback module remains speculative

### Questions
1. Could the authors include an ablation comparing (i) automated optimization only, (ii) HITL without optimization (manual corrections only), and (iii) full system performance, to quantify human feedback benefits?

2. What is the end-to-end computational cost, including LLM feedback parsing?

3. Is there a theoretical guarantee that RL- or GA-based transformation policies converge or do not degrade performance under noisy validation data?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a model-agnostic framework with two components: 
- A post-training affine transformation to refine predictions that employs existing search strategies: random search, bandits, genetic algorithm, and reinforcement learning to optimizes parameters of the transformation and covers four actions: Scale Amplitude, Piecewise Scaling, Linear Trend, and Min/Max Adjustment.  
- A human in the loop mechanism that provides natural language instructions such as “Increase values above the 80th percentile” that translate this to an action that can be added to the action pool

The utility of the proposed method was studied on standard benchmarks and authors showed performance improvement in terms of MSE percentage change across various time series forecasting models.

### Strengths
- The paper provides some theoretical insights which can be interesting for the community. 
- I am happy to see that authors conducted experiments beyond the usual time series forecasting benchmarks.

### Weaknesses
1- I am not sure if I fully agree with the role of the human in the loop here. For example, in figure 5(a) increasing the amplitude of predictions by x% it could and should be something that the objective function does as to me it is not really a domain knowledge. Is it really necessary that a human provide this type of feedback? I don’t believe so. In addition, code generation for a new action via prompting is also not something novel. Therefore, overall I am struggling to understand the contribution here of Human-In-The-Loop Feedback part. 

2- Although authors have provided time analysis based on different actions however, performance analysis for different number of actions is missing. 

3- Overall, I had a hard time following the paper as some details which may seem trivial for authors are missing and I think presentation can be significantly improved. For example:  
- Figure 3 is not cited anywhere in the paper, what is the significance of it? also, is the regularization in this figure for the ridge regression? 
- In line 213, what’s the definition of the loss? Is this the same loss as in line 1588?
- What are the rows in the table provided in section A.3.2?
- In line 242, what does “and consistency is checked on the training set to ensure generalizable gains” mean? What’s the definition of consistency here? gap between train and validation loss? 
- I think the paper will benefit from more in-depth elaboration around post-training and inference. For example, what exactly happens at post-training meaning are the parameters being updated sequentially or batch-wise? 

I’d be happy to revisit my score if authors address these weaknesses.

### Questions
What is the applicability domain of this framework as a whole? for example, looking at table 6 it seems most improvements are marginal with the exception of two transformer-based model. Does this mean the presented framework is more suitable for transformers? If so, why? These discussions are missing in the paper specially based on the model-agnostic claim of the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an adaptive optimization framework for time series forecasting, applied after training. Rather than retraining base predictors, the method uses a set of interpretable output transformations to correct systematic prediction errors. The approach can incorporate Human-In-The-Loop (HITL) guidance, in which domain experts provide natural-language suggestions that are converted into executable actions by an LLM. The authors provide a theoretical analysis demonstrating that affine corrections can reduce MSE under matched distributions, and offer a budget-dependent performance bound for SH-HPO. Experiments conducted across multiple datasets and base predictors demonstrate consistent improvements in MSE and cross-metric performance, with relatively low computational overhead.

### Strengths
1. The paper addresses the highly practical and impactful problem of improving time-series forecasts without having to retrain base models.
2. The authors present a transparent and comprehensible design for the transformation space.
3. The paper offers theoretical support for the proposed approach, including proof that affine corrections can reduce mean squared error (MSE) under aligned distributions, and performance bounds for Successive-Halving-based optimization.

### Weaknesses
1. The method relies heavily on the assumption that the validation and test distributions are aligned. However, the paper does not sufficiently analyze how the approach behaves under distribution drift, which is common in real-world time series environments.
2. The Human-in-the-Loop pipeline lacks detailed safeguards to prevent information leakage, bias, or iterative overfitting to the validation set.
3. The experimental analysis does not adequately address failure cases, particularly instances where the proposed corrections result in a decline in performance, as observed in PatchTST.

### Questions
1. The paper claims that LLMs are used only for language editing, yet the role of the LLM in generating executable code contradicts this. Furthermore, the authors do not provide ablation studies or prompt details to clarify how LLM behavior affects the system’s stability or reproducibility.
2. Could the authors please explain their strategy for handling multivariate target series? This should include details of whether transformations are applied independently or jointly, and how cross-dimensional constraints or correlations are preserved.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a lightweight, model-agnostic post-training adaptive optimization layer that refines forecasts by selecting and tuning a small set of interpretable output transformations via various algorithms, and optionally accepts natural-language expert guidance translated to executable candidate actions by an LLM. The authors theoretically justify affine corrections, describe a discrete-action + continuous-parameter action space, and show empirical improvements across many forecasting benchmarks, reporting consistent MSE reductions and a web demo.

### Strengths
1. The paper presents a simple yet practical idea, post-training output transformations, that can be applied to any base forecasting model without retraining and therefore has immediate engineering value.
2. The theoretical statement that an optimal affine correction cannot increase MSE is clean and appropriately motivates richer post-training actions, and the action design is interpretable, compact, and easy for practitioners to reason about and extend. Comparing several search strategies (random search, SH-HPO bandits, PPO, GA) and reporting their relative merits gives the reader actionable guidance about tradeoffs between simplicity, runtime, and performance.
3. Integrating human feedback via natural language, with an LLM translating prompts into candidate actions which are then validated before use, is a useful and modern design that can make forecasting pipelines more usable to domain experts.
4. The paper includes reproducibility and ethics statements and points to appendices and an interactive demo, which increases transparency and practical verifiability.

### Weaknesses
1. The claimed average improvements are modest and inconsistent across model/dataset pairs, and several cases show small or negative gains, so it is not yet clear when practitioners should expect reliable improvements versus when the method risks degrading performance.
2. The method depends critically on a representative validation set and the choice of validation budget. When validation and test distributions mismatch, such as concept drift, the optimization could overfit to validation artifacts and harm generalization. The paper states this issue, but provides limited empirical stress tests under realistic distribution shifts.
3. he human-in-the-loop path converts natural language to executable code. This opens safety and security questions (sandboxing, injection, resource use) that are only briefly sketched. The paper asserts validation and that the LLM never accesses raw data, but concrete safeguards, adversarial scenarios, and human-study evidence are missing.

### Questions
1. Please explain how validation sets were constructed and whether the optimization budget (number of evaluations) was the same for all datasets and horizons; how sensitive are results to this budget?
2. Please explain why some model/dataset combinations (e.g., PatchTST on some sets) show degraded performance after optimization, and whether action constraints or regularization can avoid such negative cases.

### Soundness
3

### Presentation
3

### Contribution
3
