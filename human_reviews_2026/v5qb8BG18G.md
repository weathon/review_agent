# Rewiring Experts on the Fly: Continuous Rerouting for Better Online Adaptation in Mixture-of-Expert models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Mixture-of-Experts (MoE) models achieve efficient scaling through sparse expert activation, but often suffer from suboptimal routing decisions due to distribution shifts in deployment. While existing test-time adaptation methods could potentially address these issues, they primarily focus on dense models and require access to external data, limiting their practical applicability to MoE architectures. However, we find that, instead of relying on reference data, we can optimize MoE expert selection on-the-fly based only on input context. As such, we propose a data-free, online test-time framework that continuously adapts MoE routing decisions during text generation without external supervision or data. Our method cycles between two phases: During the prefill stage, and later in regular intervals, we optimize the routing decisions of the model using self-supervision based on the already generated sequence. Then, we generate text as normal, maintaining the
modified router until the next adaption. We implement this through lightweight additive vectors that only update router logits in selected layers, maintaining computational efficiency while preventing over-adaptation. The experimental results show consistent performance gains on challenging reasoning tasks while maintaining robustness to context shifts. For example, our method achieves a 5.5\% improvement on HumanEval with OLMoE. Furthermore, owing to its plug-and-play property, our method naturally complements existing test-time scaling techniques, e.g., achieving 6\% average gains when incorporated with self-consistency on DeepSeek-V2-Lite.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a data-free, online test-time adaptation framework for MoE LLMs. It continuously optimizes expert routing decisions during text generation by introducing lightweight additive parameters to router logits. To further improve efficiency and stability, the framework updates only a subset of MoE layers determined by confidence-based layer selection, which identifies task-relevant layers.

### Strengths
The paper tackles an important problem of test-time adaptation for MoE models and presents an effective solution. The proposed method is data-free, computationally efficient, and easily applicable to existing models. Experiments show consistent performance gains across diverse reasoning and generation tasks.

### Weaknesses
(1) The paper lacks theoretical analysis explaining why optimizing routing decisions on the current context leads to better performance on future generation. What guarantees that reducing cross-entropy loss on already-generated tokens improves routing for subsequent reasoning steps? This is particularly concerning since the optimization objective (past context) differs from the actual goal (future generation quality).

(2) The evaluation focuses primarily on relatively short reasoning tasks (coding problems, math word problems). The method's effectiveness on longer-context scenarios (multi-turn dialogues, document summarization, extended reasoning chains) remains unclear. Given that the optimization interval is set to 128 tokens, how does performance scale with generation length? Does the method maintain benefits after generating more tokens, or do cumulative routing adjustments lead to drift?

(3) While the ablation study (Table 2) shows that confidence-based selection outperforms alternatives, the underlying assumption (high-confidence layers are more "important" for adaptation) lacks empirical validation. High confidence could equally indicate that these layers have already converged to good routing decisions and need less adjustment. Do high-confidence layers exhibit more task-relevant expert specialization? Are low-confidence layers fundamentally noisy or processing difficult parts of the task?

### Questions
Do the reported hyperparameters (η=0.05, n=5, m=128) transfer across different models and tasks, or does each setting require task-specific tuning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses a fundamental challenge in Mixture-of-Experts (MoE) models: **suboptimal routing decisions during inference**. MoE models achieve efficiency by selectively activating only a subset of experts for each token, but the quality of these routing decisions directly impacts performance. The authors draw an analogy to neuroplasticity in the human brain, arguing that MoEs need similar adaptive mechanisms during deployment.

Challenges addresseed in this work in regards to routing:

- Routers are simple linear functions that must approximate the anticipated utility of activating each expert
- Distribution shifts between training and test data lead to poor routing choices
- During standard inference, there's no mechanism to reinforce successful routing or reduce routing to unhelpful experts
- Existing test-time adaptation methods require external reference data, limiting practicality

### Key Contributions and Methodology

The paper introduces a **data-free, online test-time rerouting framework** with three main components:

1. Router Logits Modification (§ 3.2)
    - Introduces lightweight additive parameter vectors $δ^{(l)}$ for each MoE layer
    - Modifies router logits: $z̃^{(l)} = z^{(l)} + δ^{(l)}$
    - These vectors **steer expert selection** without modifying the underlying model weights
2. Dynamic Layer Selection (§ 3.3)
    
    The framework selectively updates only high-confidence layers rather than all layers:
    
    - **Confidence metric**: $C_i = -\frac{1}{k} \sum_{j=1}^{k} \log p_{i,j}$ (confidence is calculated as the negative log probability of selected experts; higher values indicate more decisive routing)
    - **Two strategies**:
        - Hard selection: Choose top-r proportion ($S_t = \text{TopK}(\{C_i\}_{i=1}^L, r)$) of highest confidence layers
        - Soft weighting: Apply confidence-based weights to gradient updates
3. Two-Phase Optimization Procedure (§ 3.4)
    
    The method alternates between:
    
    - **Phase 1 - In-Context Routing Optimization**: Uses the current context as self-supervised training data, optimizing  to minimize cross-entropy loss over n=5 iterations
    - **Phase 2 - Steered Generation**: Generates m=128 tokens with optimized routing, then returns to Phase 1 with extended context

### Strengths
- **Data-free self-supervised adaptation**: The method treats each input as its own training sample, computing cross-entropy loss on the current context (prompt + generated text) to optimize routing parameters $\delta^{(l)}$ without any external data or retrieval overhead. This achieves better performance than baselines that use external data
- **Self-Supervised Routing Refinement**: The framework treats each input prompt as a self-supervised learning opportunity for dynamic expert selection. During inference, the method alternates between two phases: In-Context Routing Optimization, where the current context itself becomes the training data for computing gradient updates to router logits, and Steered Generation, where text is generated using these optimized routing decisions. This creates a dynamic feedback loop - as the model generates text, that very text provides the supervision signal for improving subsequent routing choices. The model continuously refines its understanding of what the task requires based on its own generation progress, leading to increasingly informed expert routing decisions as generation proceeds.
- **Dynamic continuous optimization during generation**: Unlike static approaches, the method alternates between optimization phases (updating $\delta^{(l)}$) and generation phases at 128-token intervals, allowing routing to adapt as the model's understanding of the task evolves. This creates feedback loop where generation quality informs subsequent routing.
- Mechanistic insights that validate the approach:
    - Edit distance analysis shows task-specific pathway modifications concentrated in deeper layers
    - Expert utilization heatmaps reveal strategic redistribution rather than uniform changes
    - Confidence-based layer selection outperforms other form of updates
- The paper follows a natural progression from problem identification through methodology, experiments, and analysis, making it easy to follow the authors' reasoning and working.

### Weaknesses
- The paper optimizes routing by adding bias vectors to router logits: $\tilde{z}^{(l)} = z^{(l)} + \delta^{(l)}$ where $z^{(l)} = W_r^{(l)} h^{(l)}$. However, this approach doesn't address the inherent constraint that MoE routing remains a linear function of the hidden state. The modification merely shifts decision boundaries in the existing linear space.
    
    The relatively small improvements despite optimizing on the exact test context suggest that the linear routing architecture itself **may** be the limiting factor.
    
- This paper fundamentally assumes that different experts have learned distinct, task-relevant capabilities worth optimizing routing for. However, it never validates this critical assumption. Recent research has shown that MoE experts often exhibit significant redundancy and lack clear specialization, which raises questions about what this routing optimization actually achieves.
    
    There is no clear anaylsis on the following: 
    
    - Whether experts are meaningfully specialized
    - What the "preferred" experts actually compute differently
    - If routing changes actually correlate with expert capabilities
- The paper's central approach has a fundamental flaw: it optimizes routing to minimize cross-entropy loss on **already-generated text**, then uses those parameters to generate **new text**. This could create problems like:
    - The routing that best explains existing tokens may be entirely different from routing that generates good future tokens
    - The model is essentially being trained to "retroactively justify" its past decisions rather than improve future ones
    - No evidence is provided that minimizing reconstruction loss on context correlates with better generation quality

### Questions
How much performance gain can realistically be achieved by steering the router which is fundamentally a linear layer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a data-free and online test-time framework that continuously adapts MoE routing decisions during text generation without external supervision or data.

### Strengths
- The method demonstrates consistent and significant performance gains across multiple, distinct MoE architectures and a diverse set of challenging reasoning benchmarks.
- Excellent Analysis and Ablations. The ablation studies in Section 5.2 thoroughly validate the key design choices: the superiority of confidence-based layer selection over alternatives and the benefit of continuous refinement.

### Weaknesses
- Concerns regarding Latency. The paper's claim of a "modest"  computational overhead is a significant understatement. According to Table 3, the proposed method nearly doubles the inference time on HumanEval (20.12s) compared to the baseline (10.71s). Also, the FLOPs for proposed method is 1.96e+12, which is **larger** than 1.93e+12 of ICL(3-shot). 

- Lack of sensitivity analysis on optimization steps T.

### Questions
In the abstract, this paper claim that existing test-time adaptation methods could potentially address these issues, but they primarily focus on dense models and require access to external data. Also, in the related work, this paper claim that these methods assume accessible training data during deployment and introduce significant retrieval overhead, limiting real-world practicality. It would be better to showcase detailed performance of these methods.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a test-time adaptation method for MoE models, which optimizes the router logits based on the cross-entropy loss of the input sequence and current generation itself. This allows the MoE routing to be adjusted based on the characteristics of the actual input data. Furthermore, by adapting continuously, the model can acquire more task-specific routing. The experimental results show modest performance gains across various tasks, with a notable improvement confirmed on HumanEval in particular, compared to the baseline.

### Strengths
This is a very simple and lightweight method, and it could be a promising option in scenarios where test-time adaptation proves effective. Since significant performance gains are achieved on highly specialized tasks like HumanEval, it suggests that this lightweight optimization alone may be sufficient even in situations requiring significant domain adaptation. Furthermore, because it is data-free and can be used even when reference data is not available, it offers many practical advantages.

### Weaknesses
- There are some points that seem unclear regarding the fairness of the experimental setup. Please refer to the Questions.
- It appears there are two errors in the optimization strategy. First, the paper states a policy of updating layers that performed 'high-confidence' routing. It seems this should be the opposite; layers with 'low' confidence should be updated. Second, regarding Equation (2), the paper claims, "higher confidence values indicate more decisive routing decisions." This formula, however, is equivalent to entropy, where a higher value actually represents greater uncertainty. It looks as though these two errors might be acting complementarily, coincidentally leading to the correct optimization strategy (i.e., updating the uncertain layers). However, this makes the paper's reasoning unreliable. The authors should probably re-verify that the results shown in the paper align with their intended methodology.

### Questions
- When applying this method, are the adaptation parameters (the $\delta$ vectors) re-initialized to zero for each test example? If it's not the case, it suggests that for all examples after the first one, a form of task-related leakage might be occurring. This could violate the fundamental assumption of evaluating samples independently and would also imply that performance becomes dependent on the evaluation order of the examples.
- If each example is completely independent, the proposed method should perform better in the latter half of long sequences. Are there any experimental results that have been verified for different lengths of examples?

### Soundness
3

### Presentation
2

### Contribution
3
