# TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Agentic systems often implement routing, shortlisting, gating, and verification with repeated frontier-LLM calls, which accumulates token/latency costs over a run. We introduce TabAgent, a framework that reframes such closed-set decision heads as textual–tabular classification trained on signals extracted from execution traces. TabAgent comprises (i) TabSchema, which distills schema, state, and dependency features from trajectories; (ii) TabSynth, which adds schema-aligned synthetic supervision to improve coverage of rare but decision-critical patterns; and (iii) TabHead, a compact classifier that outputs calibrated probabilities for each candidate in a single forward pass. Evaluated as a drop-in replacement for the GPT-based shortlister in IBM CUGA on AppWorld, TabAgent maintains shortlist quality—e.g., Recall@7 ≥ 0.88 and Recall@9 ≥ 0.92 across five applications—and, with TabSynth, improves macro P@R by +0.14 on average. Critically, TabAgent eliminates shortlist-time LLM calls, achieving a ~95% latency reduction and an 85–91% cost reduction relative to CUGA’s GPT-4.1 shortlister in our setup, while also generalizing to other heads such as application selection and task-complexity gating. These results suggest execution traces expose sufficiently rich, tabular-representable signals to replace generative components with efficient discriminative heads in production agentic architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes TabAgent, a framework that replaces expensive generative components in agent systems (like tool selection) with efficient tabular-textual classifiers trained from execution traces. By extracting structured features and using synthetic data to boost coverage, TabAgent achieves similar decision quality to LLM-based shortlisters while reducing inference cost by up to 90% and latency by 95%, making it a practical solution for scalable, low-cost agent deployment.

### Strengths
1. **Practical Focus on Reducing Inference Cost in Agent Systems**  
   The paper tackles a real bottleneck in deploying LLM-based agents at scale—high latency and cost from repeated generation. By proposing a discriminative alternative, it offers a concrete path toward more efficient agent execution.

2. **Modular and Reusable Framework Design**  
   The TabAgent architecture is presented in a modular manner (TabSchema, TabSynth, TabHead), making it relatively straightforward to apply the same approach to other agent components beyond tool selection.

3. **Significant Efficiency Gains Demonstrated in Experiments**  
   Results show that the framework can reduce inference cost by over 85% and latency by 95% while maintaining comparable performance, giving strong empirical support for the idea of replacing generation with classification in suitable cases.

### Weaknesses
See questions.

### Questions
1. **Limited Scope of Application**  
   The proposed framework is only evaluated on a single tool-selection scenario. How can TabAgent be generalized to more complex agentic components (e.g., planning, dialogue control) where structured tabular data may be insufficient?

2. **Dependence on Execution Traces**  
   TabAgent relies heavily on historical agent execution traces for feature extraction and training. How does the method handle cold-start settings, or domains where such trajectories are unavailable or incomplete?

3. **Synthetic Data Justification**  
   The TabSynth module generates synthetic samples to augment training data, but the methodology seems heuristic. Are there any theoretical guarantees or empirical analyses showing that these synthetic examples preserve distributional fidelity and do not induce bias?

4. **Limited Evaluation Metrics**  
   The metrics focus mainly on latency and cost. Can the authors provide more comprehensive evaluations, such as error analysis on misclassified tools, or effects on downstream task success under diverse environments?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work propose a new framework to reduce the overhead for agentic systems by replacing the long context to tabular and reformulate this problem to a classification problem. The results show the proposed framework TabAgent can reduce 95% latency but maintatin similar performance.

### Strengths
1. The motivation is very clear to me.
2. The evaluation are comprehensive and the results are convincing.

This work tries to solve an important problem and they find an interesting perspective, no matter the results or the rationale, I think the novelty is already there.
There are various results provided in the evaluation section and in the appedix, I appreciate these efforts.

### Weaknesses
1. The organization and clarity can be improved.
2.  More insights are required to make it better to understand.

My major concern about this work is the organization and clarity, but I believe these can be improved before submitting the camera-ready version. My detailed comments can be found below:
1. The description of TabSynth is very limited, there is no visulization for it and I'm not sure whether I understand this part, either. I wonder howw do you validate the quality of the data generated by TabSynth? Maybe it is better clarify it and also include a subfigure of TabSynth in Fig 1.
2. The TabSchema seems to be the core of this work. However, it is unclear to me why it works. I think this is one of the key difference between an empirical report and a scientific paper. Maybe it is better to include more insights in Section 3 and explain which part is TbSchema is original, and which part is inspried by previous works. Also, please include rationales behind the choices and show it in evaluations.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes the TabAgent framework, which aims to replace the LLM-based generative decision modules with a lightweight discriminative classifier. This avoids the slow rollout of the LLM models when repetitively generating tokens to decide the final action, rather, the classifier can make the decision in one forward pass.

TabAgent first converts agent execution traces into a structured form named TabSchema using a LLM. Specifically, the key details about the task’s schema, current state, and dependencies are converted into a structural format. This converts the raw trajectory data into a features that describes each decision point and the available candidate options. Then, TabSynth synthesizes extra training data by generating synthetic examples that follow the same schema.

The TabHead is trained on both real and synthesized data, and achieve better performance than DSR and LLM-based methods on AppWorld task with IBM CUGA agent framework.

### Strengths
- Strong motivation -- attempting the tackle the slowness problem in the current agent framework due to the LLM autoregressive token generation.
- Clear design of the TabAgent framework, which is consist of constructing TabSchema from traces, synthesizing more training data with TabSynth, and training the TabHead classifier to replace expensive generative decision components.
- Strong improvement in the efficiency of the agent framework, which achieves good performance while reducing the inference cost and time.

### Weaknesses
- What is the generalizability of to other tasks than AppWorld under the IBM CUGA agentic framework?
- How flexible can TabAgent framework be adapted to another agent framework compared to the baselines?
- What is the performance comparisons LLMs designed specifically for agentic usage?
- What would be the benefit of TabAgent over fine-tuning small language models regarding to cost-effectiveness (https://arxiv.org/abs/2506.02153)?
- To what extend the TabAgent is reliant on GPT-4.1 for TabSchema extraction? What would the performance impacts be if weaker or stronger models are adopted?

### Questions
The main questions are already reflected in the weakness listed above.

### Soundness
3

### Presentation
2

### Contribution
2
