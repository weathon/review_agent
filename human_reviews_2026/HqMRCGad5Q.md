# ProSocialAlign: Preference-Conditioned Test-Time Alignment in Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Current language model safety paradigms often fall short in emotionally charged or high-stakes settings, where refusal-only approaches may alienate users and naive compliance can amplify risk. We propose **ProSocialAlign**, a test-time, parameter-efficient framework that steers generation toward safe, empathetic, and value-aligned responses without retraining the base model. We formalize five human-centered objectives and cast safety as lexicographic constrained generation: first, applying hard constraints to eliminate harmful continuations; then optimizing for prosocial quality within the safe set. Our method combines (i) *directional regulation*, a harm-mitigation mechanism that subtracts a learned "harm vector" in parameter space, and (ii) *preference-aware autoregressive reward modeling* trained jointly across attributes with *gradient conflict resolution*, enabling fine-grained, user-controllable decoding. Empirical evaluations across five safety benchmarks demonstrate state-of-the-art performance, reducing unsafe leakage and boosting alignment to human values, with strong gains across multiple evaluation metrics. **ProSocialAlign** offers a robust and modular foundation for generating context-sensitive, safe, and human-aligned responses at inference time. To facilitate reproducibility, we will publicly release the full source code and dataset upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper aims to guide LLMs to generate prosocial responses in emotional or high-stakes settings, i.e., to provide emotional support while avoiding harm. Specifically, the proposed method incorporates a soft subtraction operation in parameter space and a decoding-time technique to steer a white-box LLM, while it requires multiple models and datasets given a pre-defined attribute set.

### Strengths
1. The authors explore real-world requirements for LLMs and incorporate insights from psychology and clinical science to steer the models from an interdisciplinary perspective. The research is well-motivated and well-grounded.
2. Multiple steering techniques are thoughtfully incorporated into ProSocialAlign to achieve responses that are both socially-aware and safe.
3. The experiments are conducted using various benchmarks and metrics, demonstrating that the proposed method effectively solve the formulated problem.

### Weaknesses
1. Although the authors claim that ProSocialAlign doesn't require retraining the base model, the harm direction used in DiReg is actually derived from a fine-tuned base model, $M_h$, as depicted in L157. Additionally, I couldn't find the definition of $H$ prior to L160.
2. The data and computational requirements for ProSocialAlign remain non-trivial, which compromises its flexibility and efficiency.

    i) Since human values in the real world are always conflicting and evolving, if any new values need to be incorporated into the prosocial attribute sets, do we have to repeat the procedures in Sec. 4.1 and re-train the reward & harmful model? 

    ii) If so, the authors are suggested to justify that the five criteria are sufficiently representative of human values.

    iii) Additionally, if we need to align to a new user, what would the associated annotation and computational costs be?
3. As the objective is to support the users (by maximizing the five attributes), using GPT-4o as a judge may introduce potential bias. It is better to justify the reliability of GPT-4o for this task, or provide evaluations from other LLM or human judges.

### Questions
1. The equation for $v_{pf}$ in L146 should be adjusted to make the subscripts consistent.
2. The practice of keeping the top-k dimensions for harm vectors is interesting. Have you tried other methods for imposing this constraint? Why didn’t you apply the constraints in the embedding space instead?
3. Are there any specific requirements for the backbone of the reward model?
4. Is it possible to align a black-box LLM using ProSocialAlign?

### Soundness
4

### Presentation
4

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
The paper proposes ProSocialAlign, a test-time alignment framework that steers a frozen base LM toward safe responses without retraining the base model. Experiments on multiple safety benchmarks report reduced unsafe and improved value alignment against strong instruction-tuned baselines.

### Strengths
1. $\textbf{Comprehensive experiments.}$ The evaluation is broad and generally supports the paper’s central claim that the proposed method improves safety at inference time.

### Weaknesses
1. $\textbf{Method vs. prior work organization.}$ Section 3.3.1 introduces PBLoRA, which is not proposed in this paper but used as a component of the reward model. Moving such background to Related Work / Preliminaries would clarify the paper’s original contributions and streamline the Method section.
2. $\textbf{Conflict-mitigation objective.}$ The gradient-conflict mitigation strategy across different attributes is an interesting idea. However, when attribute conflicts are inevitable, could this mitigation hurt performance on specific values or tasks? An ablation comparing with vs. without conflict mitigation would make the benefits and trade-offs concrete.
3. $\textbf{Computation and practicality as a test-time method.}$ The approach requires (a) a harm-tuned model to estimate a harm direction and (b) a preference-conditioned reward model whose scores are combined with the base model at each decoding step. Please quantify the computational overhead relative to other test-time alignment methods, and discuss memory/throughput implications for long-context decoding.

### Questions
Please see my weakness. I would consider raising my scores if the above concerns are solved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ProSocialAlign, a test-time framework for doing better than simply refusing to respond based on safety concerns, encouraging responses aligned to five key prosocial attributes. This

### Strengths
The paper addresses a significant limitation in LLM alignment in terms of trying to move beyond refusal base paradigms. he proposed ProSocialAlign framework appears practical and well detailed along with beyond parameter-efficient and running at test time. A key contribution here is the preference-conditioned autoregressive reward model that allows for fine tuning on the key Paretro tradeoffs they consider at inference time. Consideration of potentially conflicting gradients on the key attributes is directly considered. The empirical evaluation is strong and test's the framework's performance across five distinct safety benchmarks and against a number of baselines, and generally outperforms them on Pareto tradeoffs.

### Weaknesses
There seems to be a fundamental reliance on the assumption that subtracting a "harm vector" is a reliable means of pushing responses in the intended direction, while even if supported by the experiments, feels brittle to me. I am not sure how likely this is to really hold for crucial failure modes, and don't feel strongly convinced by the discussion within the paper. 

I feel like the approach while it can work well for the k=5 prosocial attributes that are evaluated, would struggle to scale to larger k. Not sure that training and conflict resolution will be able to work as well empirically as k grows. Generally the gradient conflict resolution seems to be largely a heuristic that lacks much theoretical guarantees or even empirical understanding. Reducing conflicts upon these vectors seems like a really nontrivial matter, especially if we consider conflicts arising from higher order interactions between vectors (like 3 conflicting all together). 

In general I feel that the paper doesn't give much to understand the reason why we might expect it to be the outer frontier on the Pareto frontier between helpfulness/truthfulness and the prosocial attributes it considers and how generalizable or fundamental this methodology is beyond these. Also the one shot evaluation on benchmarks may overstate its efficacy since a big challenge of practical safety scenarios such as these lies in multi-turn conversations.

### Questions
How do you expect the performance of ProSocialAlign to scale to larger k?

How could ProSocialAlign extend to multi turn conversations?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents ProSocialAlign, a test-time alignment framework for language models that addresses safety challenges in emotionally charged or high-stakes scenarios. The method formulates safety as lexicographic constrained generation, first applying hard constraints to eliminate harmful continuations, then optimizing for prosocial quality within the safe set. The approach combines directional regulation (subtracting a learned "harm vector" from model parameters) with a preference-aware autoregressive reward model trained jointly across five human-centered attributes: empathy, sensitivity, non-judgmental stance, truthfulness, and helpfulness. The authors evaluate their method across multiple safety benchmarks and demonstrate improvements in unsafe content reduction and human value alignment while maintaining task utility.

### Strengths
1. The authors demonstrate notable innovation by proposing five fine-grained dimensions for model evaluation criteria, providing a more nuanced framework for assessing prosocial alignment.
2. The proposed method demonstrates commendable simplicity in its design, making it readily reproducible.

### Weaknesses
1. I believe the research problem addressed in this paper lacks innovative value, as several of the proposed viewpoints and methods have already been extensively studied in the multi-objective alignment domain. It is unclear why the authors did not directly frame this as a multi-objective alignment problem, and at minimum, the experiments should include comparisons with established baselines from this field.

   [1] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model
   [2] Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment
   [3] Controllable Preference Optimization: Toward Controllable Multi-Objective Alignment

2. The proposed method introduces excessive complexity to the pipeline. While the base model requires no retraining, the approach necessitates training two additional models: a harm detection model and a multi-objective reward model. Furthermore, during inference, the method requires computing parameter differences, which I believe introduces substantial computational overhead that significantly impacts practical deployment efficiency.

3. Why subtract the harm model to select vectors? The harm model may not exclusively contain harmful semantic information but could also encode other benign information, such as grammatical or syntactic features. Directly subtracting the harm model's representations may inadvertently remove useful information that is unrelated to harm detection, potentially degrading the overall model performance.

### Questions
see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
