# Control Tax: The Price of Keeping AI in Check

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The rapid integration of agentic AI into high-stakes real-world applications requires robust oversight mechanisms. The emerging field of AI Control (AIC) aims to provide such an oversight mechanism, but practical adoption depends heavily on implementation overhead. To study this problem better, we introduce the notion of Control tax---the operational and financial cost of integrating control measures into AI pipelines. Our work makes three key contributions to the field of AIC: (1) we introduce a theoretical framework that quantifies the Control Tax and maps classifier performance to safety assurances; (2) we conduct comprehensive evaluations of state-of-the-art language models in adversarial settings, where attacker models insert subtle backdoors into code while monitoring models attempt to detect these vulnerabilities; and (3) we provide empirical financial cost estimates for control protocols and develop optimized monitoring strategies that balance safety and cost-effectiveness while accounting for practical constraints like auditing budgets. Our framework enables practitioners to make informed decisions by systematically connecting safety guarantees with their costs, advancing AIC through principled economic feasibility assessment across different deployment contexts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the concept of Control Tax—the operational and financial cost of implementing oversight mechanisms in AI systems, especially under the framework of AI Control (AIC). It develops a quantitative framework linking model monitoring performance with long-term system safety and cost-efficiency. The authors simulate adversarial conditions using code backdooring tasks in the APPS dataset, where attacker LMs introduce subtle flaws, and monitor LMs attempt to detect them. Through empirical ROC analyses and game-theoretic modeling, the study quantifies how safety scales with auditing budgets, model performance, and financial cost.

The paper’s main contribution is establishing a principled economic-safety trade-off in AI oversight. It demonstrates that stronger monitors do not always yield proportionally higher safety—cost-effective combinations of models and auditing strategies can outperform expensive monitors under limited budgets. The theoretical results extend to batch and sequential deployment scenarios, offering practical guidelines: conceal deployment horizons, audit heavily early on, and balance cost with monitoring quality. Ultimately, the work provides a foundational step in formalizing the economics of AI safety, giving researchers and practitioners a concrete basis for optimizing control strategies in real-world deployments.

### Strengths
The paper presents an original and timely concept—Control Tax—that quantifies the trade-off between AI oversight cost and safety, linking economic modeling with AI control in a novel way. The methodology is solid, combining theoretical analysis with well-designed experiments on code-based adversarial tasks. The writing is clear and effectively supported by figures illustrating key trade-offs. Overall, the paper makes a significant contribution by formalizing the economics of AI safety and offering practical insights for cost-efficient oversight in real-world AI systems.

### Weaknesses
The paper’s evaluation is limited to synthetic and code-based scenarios, which may not fully capture the complexity of real-world AI oversight challenges. Some assumptions in the theoretical modeling—such as simplified attacker–monitor dynamics and fixed audit budgetsreduce the generality of its conclusions. The framework’s practical applicability could be better demonstrated through broader experiments or deployment case studies. Additionally, parts of the mathematical exposition are dense, which may hinder accessibility for readers less familiar with economic modeling in AI safety.

### Questions
How sensitive are the results to the choice of monitoring and attack models? Would performance trends hold under different model architectures or non-code tasks?

Could the authors provide more empirical validation on real-world or human-in-the-loop oversight settings to strengthen practical relevance?

Theoretical results assume fixed budgets and known audit horizons—how would the Control Tax framework adapt to dynamic or uncertain resource environments?

Have the authors explored nonlinear or diminishing returns effects in monitoring performance as spending increases, and how might that influence policy implications?

Can the authors elaborate on how Control Tax could integrate with governance mechanisms such as red-teaming or model certification frameworks to enhance real-world applicability?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The manuscript introduces the concept of a Control Tax, defined as the operational and financial cost required to maintain effective oversight in AI deployment pipelines. In the presented setting, a potentially misaligned attacker model inserts subtle backdoors into code solutions from the APPS dataset, while a monitor model attempts to detect these vulnerabilities. The manuscript develops a two-part contribution: first, it empirically evaluates a wide range of modern language models as both attackers and monitors to estimate their detection and evasion capabilities; second, it provides a game-theoretic framework that translates a monitor’s ROC curve and auditing budget into a quantifiable probability of avoiding compromise during long-term deployment. The resulting Control Tax curves illustrate the trade-off between safety and auditing cost and show how optimal monitoring strategies can differ depending on attacker capability, resource constraints, and whether the deployment is batch or sequential.

### Strengths
The manuscript’s core contribution is the introduction of the Control Tax as a variable linking monitoring system performance to operational and financial deployment costs. This formulation is original in that it treats oversight not only as a technical alignment problem but as a resource allocation problem. Prior work in AI control has largely focused on proving that monitoring protocols can succeed in principle; this manuscript instead addresses the conditions under which they are practically adoptable. The resulting framework allows different safety targets to be expressed in terms of budget and auditing strategy, which provides a concrete decision structure for practitioners considering real deployment contexts. 

The empirical component is constructed with clear methodological discipline. The attacker and monitor models are evaluated using a standardized APPS coding task environment, which ensures controlled variation across experiments. Backdoored solutions are generated using an agentic multi-step pipeline that enforces correctness on public test cases and explicitly validates malicious triggers, ensuring that the resulting dataset meaningfully distinguishes intentional vulnerabilities from ordinary programming errors. The monitoring task is operationalized through a calibrated suspiciousness scoring procedure, rather than binary detection, which makes the measurements directly compatible with downstream auditing-budget analyses. The breadth of models evaluated, combined with balanced datasets and formatting normalization (e.g., comment stripping), helps reduce confounds arising from stylistic artifacts. 

The theoretical modeling is presented with clarity. The manuscript explains how ROC curves derived from the monitoring evaluations map into dynamic auditing decisions in both batch and sequential settings. The derivation of safety probabilities under optimal attacker and defender strategies is shown with reference to established stochastic process frameworks, and intermediate assumptions are made explicit. The paper consistently connects the modeling back to operational interpretations, explaining not just the mathematical structure but also how the defender’s auditing schedule should be shaped across time. 

The significance of the work lies in providing a tractable bridge between monitor accuracy metrics and deployment-level safety guarantees. Organizations deciding whether to deploy model oversight systems require not only performance comparisons but also cost and scheduling implications. This manuscript offers a structured method for inferring how much auditing is needed to maintain a desired safety level and how that cost changes when adopting a stronger or weaker monitor. These results directly inform strategy discussions in AI governance, red-teaming design, and scalable oversight implementation.

### Weaknesses
The primary limitation of the work lies in the assumption of a stateless, non-adaptive attacker. In the empirical pipeline, the attacker generates backdoors independently and does not update strategy based on audit outcomes or monitor behavior. In practice, attackers (human or model-based) may iteratively probe the monitoring boundary, learn suspiciousness patterns, and reduce detection rates over time. The theoretical analysis also assumes that the red team maintains a fixed insertion probability optimized only with respect to the task horizon. Extending the framework to adaptive attackers—for example, those modeling the monitor as a learnable discriminator—would materially strengthen claims about real-world robustness. The manuscript acknowledges this scenario but does not examine parameter regimes in which adaptivity meaningfully alters the Control Tax. 

The evaluation domain is restricted to code backdooring in APPS. This setting has several desirable properties (e.g., clear correctness signals, deterministic vulnerability definitions), but it differs from domains where oversight is often considered most critical, such as multi-step planning, agentic task execution, or socially mediated harm. The specific structure of programming tasks—particularly the existence of public test cases that constrain solution space—may limit the diversity of viable backdoors and thus the challenge imposed on monitors. It remains uncertain whether the same monitoring strategies and cost-safety trade-offs apply when failures are less crisply defined or when correctness requires implicit world modeling. A discussion of transferability, or a small-scale extension to one additional domain with less explicit ground truth, would clarify the scope of applicability. 

The Control Tax curves depend on model cost estimates treated as fixed. In real deployment settings, inference cost varies substantially across hardware configurations, batching strategies, and licensing models. While this does not undermine the conceptual framework, it means the curves are best interpreted as relative guidance rather than directly actionable cost predictions. The manuscript could strengthen its practical significance by characterizing sensitivity to cost assumptions (e.g., showing how safety frontiers change when monitor cost decreases by an order of magnitude, or when attackers and monitors use locally deployed models instead of hosted APIs).

### Questions
How does the framework change if the attacker adapts to the monitor’s observed actions over time?

How sensitive are the cost-benefit conclusions to different pricing assumptions for inference compute?

### Soundness
3

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
3

### Summary
This paper introduces Control Tax, a framework for quantifying the operational and financial costs (“tax”) associated with adding oversight mechanisms—monitors—to AI systems. Using a code backdoor detection setup on the APPS dataset, the authors simulate attacker–monitor dynamics between language models (e.g., GPT-4o, Gemini, o3-mini) and quantify trade-offs between safety, auditing budgets, and cost. They propose a game-theoretic analysis translating ROC curves of monitor performance into long-term safety probabilities, comparing batch and sequential auditing strategies.

### Strengths
The introduction of “Control Tax” formalizes an important but underexplored cost-safety trade-off in AI control, bridging empirical LM monitoring with theoretical economics of oversight.

### Weaknesses
1. Limited scope of evaluation: Experiments rely solely on the APPS dataset and code tasks, which may not generalize to broader or more agentic AI behaviors.
2. Transferability not discussed. For example, if adaptive attackers can learn the moderator's auditing policy, does this affect safety guarantees?

### Questions
1. How sensitive are the conclusions (e.g., optimal auditing strategies) to the specific backdoor definition or dataset used—would agentic or multimodal tasks change the control tax curves?
2. Does the framework extend to adaptive or deceptive attackers that learn the monitor’s auditing policy over time, and how might this affect safety guarantees?

### Soundness
3

### Presentation
3

### Contribution
3
