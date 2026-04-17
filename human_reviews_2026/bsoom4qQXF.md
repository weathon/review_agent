# Bootstrapped Exploration with Causal Reasoning: A Training Paradigm for Adaptive Forecasting Agent

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Time series forecasting is critical in domains such as finance, energy, and healthcare, yet real-world datasets often exhibit non-stationarity, noise, missing values, and distribution shifts, posing severe challenges for generalization. In practice, industry solutions typically rely on customized forecasting frameworks that combine imputation, decomposition, and specialized models. However, such frameworks incur high labor costs. Moreover, we observe that many frameworks suffer from the impacts of distribution shifts, which degrade their respective performance. Thus, it is critical to establish a new paradigm that retains high transferability across diverse datasets while accumulating reusable strategy knowledge. This is fundamental for large-scale and dynamic environments. While large language model-based agents have recently demonstrated strong reasoning and tool-use capabilities, no forecasting agent can automatically adapt to diverse time-series datasets. This gap arises from two core obstacles: the scarcity of labeled supervision and the inherent complexity of mapping dataset-specific meta-features to effective forecasting strategies. To address these challenges, we propose BECRA, a novel agent training paradigm that learns forecasting intelligence through contrast-aware exploration and causal lesson extraction, without any human-annotated supervision. BECRA distills symbolic strategy lessons that enable in-context planning on unseen datasets, achieving zero training adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training paradigm (BECRA) that autonomously learns how different forecasting toolchains perform under varying dataset meta-features. It explores combinations of preprocessing and modeling tools, evaluates their performance, and uses a large language model to extract interpretable causal “lessons” describing when and why each toolchain succeeds or fails. These lessons are stored in textual form, allowing the agent to reuse and compose them for zero-shot forecasting on unseen datasets.

### Strengths
1. The method description is mostly clear, and Figure 1 effectively illustrates the core workflow of the BECRA framework.

2. The mathematical notation is standard and consistent, making the formulation easy to follow.

3. The paper integrates exploration, causal reasoning, and symbolic knowledge extraction in a unified pipeline, which is novel for time-series forecasting.

### Weaknesses
1. Unclear formulation (Page 3, Line 159):
The description of the UCB-based exploration step is ambiguous. It states that empirical performance is updated for each toolchain, but the equation provided defines $a^*$, which instead selects the next toolchain. 

2. Reliance on LLM reasoning without validation:
The method assumes that the LLM can accurately extract causal “lessons,” but there is no quantitative or qualitative analysis demonstrating how reliable these extracted lessons are. The paper should include either human evaluation or consistency checks to assess hallucination or reasoning errors.

3. Limited expressiveness of meta-features:
The agent operates on hand-crafted, scalar meta-features that summarize dataset characteristics. These descriptors may fail to capture the full complexity of high-dimensional or non-stationary time series, limiting the generality of the causal lessons.

4. Toolchain selection criteria unclear:
The paper does not describe how individual tools (imputation, decomposition, forecasting models, etc.) were chosen. It remains unclear whether the library is comprehensive, balanced, or biased toward specific architectures.

5. Experimental setup lacks transparency:
The description of datasets, forecasting tasks, and evaluation protocol is too brief. The paper should specify dataset sizes, forecast horizons, and input/output configurations. The baselines are listed but not fully explained or cited in the main text.

6. Missing statistical rigor:
Reported results lack error bars or significance tests. Variability across runs should be reported to support claims of robustness and superiority.

7. Figure quality issues:
Figure 2 suffers from small font size and low resolution, making it difficult to read. The shaded red region below the curve is unexplained and should be clarified.

8. Overly strong or subjective language (minor):
The writing repeatedly uses subjective qualifiers such as "clearly", "strong results".  Such language should be replaced by factual statements and quantitative evidence, allowing readers to draw their own conclusions.

### Questions
Please address my each point in the weakness section.

### Soundness
2

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
3

### Summary
The paper addresses the challenges of time series forecasting, highlighting the limitations of current industry frameworks, which often require manual trial-and-error, fail to adapt efficiently, and do not accumulate reusable forecasting knowledge. The authors propose BECRA, an agent-based training paradigm designed to develop "forecasting intelligence" through contrast-aware exploration and causal lesson extraction, all without human-annotated supervision. BECRA aims to replace traditional frameworks with a LLM agent.

Authors summarize as main contributions: 1) identify limitations of existing models and frameworks, 2) BECRA - LLM agent training for forecasting intelligence with contract-aware exploration and causal lesson extraction without human supervision, 3) library. The BECRA framework operates in a four-stage cycle: sampling strategies, causal lesson learning, lesson verification, and lesson-guided forecasting, aiming to replace manual framework design with automated.

### Strengths
- Clear Problem Identification: The paper provides an analysis of the limitations in current time series forecasting frameworks;
- Introduces an LLM agent for time series forecast, moving beyond traditional manual and trial-and-error methods.
- The framework enables automated, structured exploration and in-context planning, allowing the agent to adapt to new datasets without human supervision.
- Not dataset (or domain) dependent, being able to be adopted in a range of different time-series applications.

### Weaknesses
(There are questions related to most of these weaknessses)
- The 'causal reasoning' component: I'm convinced what we see is actually 'causal' lessons, and not just correlations. It would be nice to have more information about it.
- Meta-features seems very important to the whole process, but there limited information about how their quality might impact on overall performance; 
- The experiments lack confidence intervals, which makes it hard to make comparisons between different methods.

### Questions
- Impact of Meta-Features: How does the quality of meta-features affect the BECRA framework? What happens if meta-features quality is poor? How much can one over-engineer meta-features? 
- RL Agents vs. LLM Agents: Could reinforcement learning (RL) agents be used instead of LLM agents, given that RL also balances exploration and exploitation? Did you make tests with RL agents?
- Causal Lesson Validity: How are counterfactuals for causal lessons obtained, and what guarantees exist that these lessons reflect true causality rather than spurious correlations?
- Computational Cost: While fine-tuning is computationally expensive, LLMs also incur significant costs (albeit monetary rather than time). How does this trade-off compare?
- Reporting of Results: In Table 1, what are the confidence intervals for the reported metrics? Without error bars, it is difficult to interpret the results.

### Soundness
2

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
2

### Summary
This paper introduces BECRA, a novel paradigm for training adaptive forecasting agents by distilling reusable, symbolic knowledge without human supervision. The approach demonstrates strong empirical performance in rigorous zero-shot settings, outperforming specialized models and foundation models across various benchmarks utilizing a leave-one-out protocol.

### Strengths
1. The paper introduces a novel approach to address the scarcity of labeled supervision for time-series strategy selection, shifting the focus from repetitive dataset optimization (like AutoML) to reusable, symbolic knowledge. 

2. The use of contrast-aware UCB sampling explicitly preserves both high and low performing examples, which is important for causal reasoning. The performance drop of the greedy variant in the ablation study validates this choice.

3. The paradigm is model-agnostic, the distilled symbolic lessons are transferrable and can be successfully utilized by various other LLMs with only graceful performance degradation.

4. The manuscript is clearly structured and easy to follow.

### Weaknesses
**1. Ambiguous Formulation:**
The UCB objective formulation in Sec. 3.1 is defined as $a^{*}=arg~max_{a\in\mathcal{A}}\mu(a)+\lambda\cdot\sigma(a)$, where $\mu(a)$ is the average historical performance (MSE). The formulation should imply minimization (e.g., Lower Confidence Bound) or maximization of the inverted MSE.

**2. Negative Lessons:**
The paper says they “verify negative causal lessons” and “check if performance improves when the lesson is contradicted or removed.”, but the Algorithm 3 seems only removes $\phi_k$ but never contradicts it (checking if performance improves when the lesson is contradicted is not clearly integrated).

**3. Verification Procedure:**
The verification procedure in Algorithm 3 involves the agent planning strategies guided by the lesson library with and without the specific lesson $\phi_{k}$. This means the measured effect $\Delta_{\phi_{k}}$ can be potentially confounded by the LLM agent's subsequent planning choices, rather than isolating the direct impact of the lesson itself.

**4. (Minor) Threshold not Defined:**
The definition of the causal effect $\Delta_{\phi_{k}}$ relies on $P(y=1|...)$, where $y=1$ signifies a successful outcome based on a threshold $\tau$. However, the criteria for determining this threshold are not defined.

### Questions
1. Planning retrieves all lessons whose conditions match the new dataset, then passes them together to the LLM to pick a strategy. Is there a explicit rule for resolving contradictions among those retrieved lessons?

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
4

### Summary
In this paper, the authors present BECRA, an agent training framework for time-series forecasting. It first explores a diverse set of toolchains including time-series analysis and processing tools. Then it generates causal lessons (strategies of choosing toolchains) based on the data metadata and toolchains that achieve high-reward and low-reward samples, via upper confidence bound. It verify the derived causal lessons by measuring the causal effect of these lessons' successful outcome under corresponding patterns. The verified lessons are stored in knowledge base and then retrieved and employed for language agent to do in-context forecasting. Experiments on long-term,  short-term forecasting, and noisy data (missing values, contaminated values) show the efficiency of the proposed approach, that help the language agent choose appropriate toolchains considering different dataset characteristic. Ablation studies confirm the effectiveness of proposed modules such as contrast UCB sampling, causal lesson learning and verification.

### Strengths
* This paper targets at an impactful and practical question about domain gaps in time-series forecasting. It presents its motivation and methodology very clear.

* The knowledge base of causal lessons are valuable for the community and applications to efficiently leverage language agents to choose proper analysis techniques and forecasting models, reducing the requirement of human prior. 

* Experiments show that BECRA works as an accurate orchestration layer that smartly choose pipelines referring data properties, The  robustness experiments like missing and contaminated data are valuable. The ablations probe the roles of exploration strategy, causal lessons  and verification, and demonstrate the flexibility of in-context learning compared to SFT models.

### Weaknesses
* This work is highly relevant to the recent line of combining language agents and time-series models, e.g. [1][2][3][4]. The author should expand Related Work to review this line of work and compare BECRA’s lesson induction/verification to other time-series agent studies. 

* Time-series foundation models are general forecaster which are expected to handle diverse domains' data, which is another way of handling domain gaps. The author should consider benchmarking the proposed method (agent aspect) with time-series foundation models (model aspect) that don't have data leakage, e.g., using benchmark GIFT-eval [5].

* The author should run more runs and add the uncertainty/confidence metrics to the experiments. Means without error bars/tests obscure whether improvements are statistically reliable.

[1] Yeh, Chin-Chia Michael, et al. "Empowering Time Series Forecasting with LLM-Agents." arXiv preprint arXiv:2508.04231 (2025).

[2] Garza, Azul, and Reneé Rosillo. "TimeCopilot." arXiv preprint arXiv:2509.00616 (2025).

[3] Zhao, Haokun, et al. "Timeseriesscientist: A general-purpose ai agent for time series analysis." arXiv preprint arXiv:2510.01538 (2025).

[4] Wang, Xinlei, et al. "From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection." Advances in Neural Information Processing Systems 37 (2024): 58118-58153.

[5] Aksu, Taha, et al. "Gift-eval: A benchmark for general time series forecasting model evaluation." arXiv preprint arXiv:2410.10393 (2024).

### Questions
I have listed most of my concerns and suggestions in the weakness section. Here are my additional questions:

* How is the set of candidate toolchains sampled? Are they just combinations of all possible available tool sequences? Wouldn't this be intractable for following computation? 

* What is the training data for collecting causal lessons? Does it have overlapping with the test data?

* How does the generalizability of the compiled knowledge base? Could it be used to completely unseen datasets? How does it behave and could authors provide some failure/suboptimal cases and analysis? Demonstrations of the agent behavior and reasoning would also be interesting and more interpretable to have.

### Soundness
3

### Presentation
3

### Contribution
2
