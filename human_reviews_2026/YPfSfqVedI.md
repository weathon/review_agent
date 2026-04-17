# PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware nextprompt that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource—complete with code, rules, and generation logs—to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PersonaLedger, a system designed to generate realistic financial transaction data while preserving privacy. It combines persona-conditioned large language models (LLMs) with a programmatic rule-driven engine to create transaction streams that are both diverse and compliant with real-world financial constraints. This framework allows for the generation of 30 million transactions from 23,000 users, providing a synthetic dataset that can be used for financial forecasting and anomaly detection tasks such as illiquidity classification and identity theft segmentation. The system also ensures that the generated transactions respect hard financial rules, such as income-to-spending ratios and timely payments.

### Strengths
Originality: The paper introduces a new approach for generating synthetic financial data by combining LLMs with a rule-based system, which ensures both diversity and logical consistency. This is an innovative step forward in financial AI research, where previous methods have struggled to balance these two aspects.

Quality: The proposed system is well-designed and effectively ensures that financial transactions respect real-world constraints. The dataset is large and diverse, providing a valuable resource for future research in financial anomaly detection and forecasting.

Clarity: The paper is generally well-written, and the figures and tables are effective in illustrating the system’s design and the experimental results. Some sections could benefit from more detailed explanations, but overall, the clarity of the writing is good.

Significance: The contribution of the paper is significant, especially in the context of financial AI research, which often suffers from a lack of public, high-quality datasets. PersonaLedger addresses this gap by offering a publicly available dataset that is both diverse and realistic. Moreover, the introduction of the rule-grounded generation engine is a novel approach to ensuring financial consistency in synthetic data generation. However, the benchmarking tasks and their evaluation could be more rigorously compared against existing benchmarks to strengthen the paper’s claims of originality and impact.

### Weaknesses
1 Lack of Comparative Validation: The benchmark tasks are only tested on the PersonaLedger dataset, without validation against real-world financial data or other synthetic datasets, making it unclear if the models would perform similarly in practical scenarios.

2 Insufficient Quantitative Support for Claims: Claims about the dataset's "socioeconomically realistic" patterns lack statistical validation, such as p-values or confidence intervals, weakening their credibility.

3 Unclear Merchant Name Selection: The paper does not explain how the LLM selects specific merchants (e.g., Walmart vs. Kroger), which could impact the dataset’s realism and utility for merchant-specific analysis or fraud detection.

4 Static User Personas: The Personas used in the dataset are static, without temporal evolution or integration of life events (e.g., marriage, retirement), which limits the dataset’s ability to capture long-term or evolving user behaviors.

### Questions
1 LLM Degradation Over Time: The paper mentions that LLMs may struggle with rule consistency as transaction sequences grow longer. Could you provide more detailed empirical results on how the system maintains financial accuracy in long sequences, and what steps can be taken to mitigate any degradation in performance over time?

2 Handling Rare Events: How does PersonaLedger handle rare or unusual user behaviors that significantly deviate from typical patterns (e.g., sudden large expenses or life events)? Are these edge cases adequately captured, and how does the system adapt to them?

3 Benchmarking Comparison: How does PersonaLedger compare to other synthetic financial datasets in terms of task difficulty and the realism of generated transactions? Could you provide a more direct comparison with other widely-used datasets for tasks like fraud detection or financial forecasting to validate the uniqueness and value of your approach?

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
3

### Summary
This paper proposes PersonaLedger: using persona-conditioned LLMs to generate candidate transactions, then a programmable rule engine performs state updates and constraint verification, forming a "generate→verify→correct→regenerate" closed loop. Based on this engine, the authors synthesize and promise to release a dataset of approximately 30 million transactions from 23,000 users, providing two evaluation tasks: liquidity stress prediction (user-level) and identity theft segmentation (event-level), reporting Precision/Recall/F1/AUC for various baseline models under a unified protocol. The paper emphasizes reproducibility and auditability (code, rules, prompts, seeds, and generation logs).

### Strengths
Excellent work, great creativity and idea, congratulations on realizing it!
Clear methodology, closed loop: LLM ensures diversity, rule engine strictly controls accounting and calendar constraints, errors can be corrected through structured prompts.
Complete resources: Large data scale with comprehensive fields, plus two tasks closely related to risk control/anti-fraud with a unified protocol.
Broad baseline coverage: Compares Transformer, PatchTST, Autoformer, iTransformer, etc. under the same settings, providing a reproducible starting point.

### Weaknesses
While the paper provides statistical analysis (Section 2.2) and benchmark tasks (Section 3), there is no systematic evaluation of whether the generated transactions are actually realistic compared to real financial data.

### Questions
(1) You state that you will open-source the code and dataset, but I haven't seen any related links yet. If this can be confirmed, I will revise my score. "we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks" - I can only evaluate you based on existing materials, not just promises. I can consider revising the score when the dataset and codes are realsed.

(2) Can you release some code to facilitate reproduction for reviewers and readers?

### Soundness
2

### Presentation
2

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
This paper proposes PersonaLedger, a synthetic data generation engine that combines a persona-conditioned LLM with a rule-based programmatic controller to simulate realistic financial transaction sequences. The closed-loop design ensures that each generated transaction satisfies accounting constraints (e.g., credit limits, cash balances, due dates), while the LLM provides behavioral diversity conditioned on rich user personas derived from Nemotron-Personas. Using this system, the authors generate 30 million transactions for 23,000 users, and release two benchmark tasks: illiquidity classification and identity theft segmentation.

### Strengths
1. Timely contribution: The lack of publicly available transaction data due to privacy restrictions is a real bottleneck in financial AI research. The proposed approach represents an ambitious and creative attempt to overcome this constraint while maintaining logical consistency.

2. Interesting idea: The LLM + rule engine closed loop is very interesting. It directly addresses the brittleness of rule-based simulators and the constraint violations of purely generative models (e.g., GANs or VAEs).

3. Transparency and reproducibility: The release plan, including prompts, rules, seeds, and logs, is commendable.

### Weaknesses
1. Insufficient validation of realism: The main limitation lies in the lack of quantitative or external validation demonstrating that the generated data are truly realistic or useful proxies for real-world ledgers. Statistical diversity and rule adherence are necessary but not sufficient. Without comparison to real transaction datasets (even if at an aggregated or stylized level), it is hard to judge whether the synthetic data exhibit realistic interdependencies or temporal dynamics.

2. Shallow persona grounding: The process by which financial profiles are inferred from personas (using a Llama-3.3-70B model with only seven expert-crafted examples) appears subjective and fragile. If the inferred income or credit limit deviates from plausible values, the downstream simulation could produce misleading liquidity patterns. The framework assumes personas are rich and realistic, but there is no ablation or validation showing that persona quality drives realistic behavior.

3. Task design and representativeness: The two benchmark tasks, while well-defined, feel somewhat artificial and detached from genuine financial modeling challenges. For instance, “identity theft segmentation” is simulated simply by inserting another user’s transactions into a sequence — a simplistic proxy that may not reflect actual fraud dynamics. Similarly, “illiquidity classification” is driven by the system’s own liquidity rules, creating a self-referential task that may not generalize beyond this synthetic world.

### Questions
Can practitioners or researchers genuinely trust and adopt this dataset as a benchmark? Without external calibration or interpretability analysis (e.g., correlation structures, spending autocorrelation, realistic merchant co-occurrence), does the dataset risk becoming a closed synthetic ecosystem—internally consistent but not empirically grounded?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors construct an entirely synthetic financial transaction dataset consisting of 30 million transactions from 23k users. They leverage the world knowledge inside the LLM to generate realistic spending rollouts for a diverse set of profiles which are seeded from nemotron personas.

### Strengths
- The framework developed to generate the data looks useful as an artifact. There is thoughtful design on the abstractions in the engine orchestrating the LLM calls. In particular, the interface for adding rules is well thought out.

- The writing of the paper is very good. The authors thoroughly motivate the problem at hand, discuss issues with naive solutions, and give a very clear exposition of the structure and characteristics of the dataset release.

- The data resource is of high quality. A significant amount of manual effort was used to make the data cleaner and more consistent.

### Weaknesses
- There is a lack of concrete evaluation criteria proposed to assess the fidelity of the proposed dataset with respect to a real financial transaction dataset. Arguments of realism are mostly qualitative, or pertain to an arbitrarily picked attribute.

- More evidence of the usefulness of the dataset would be appreciated. For the downstream task benchmarks, it would be good to assess whether the synthetic-benchmark-induced rankings of methods align with the ranking of methods on real tasks; alternatively if the synthetic benchmarks can identify systematic gaps in the performance of a model trained on real data; and also if the synthetic data is useful training data for the real task.

### Questions
As the data is entirely synthetically generated, it is important to understand to what extent the resource is useful for developing models for real data. Although this appears to be a weak point of the paper, I believe overall the dataset and generation framework are valuable artifacts for the research community and recommend acceptance.

### Soundness
3

### Presentation
4

### Contribution
3
