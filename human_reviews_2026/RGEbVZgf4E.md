# Task Vectors, Learned Not Extracted: Performance Gains and Mechanistic Insights

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Large Language Models (LLMs) can perform new tasks from in-context demonstrations, a phenomenon known as in-context learning (ICL). Recent work suggests that these demonstrations are compressed into task vectors (TVs), compact task representations that LLMs exploit for predictions. However, prior studies typically extract TVs from model outputs or hidden states using cumbersome and opaque methods, and they rarely elucidate the mechanisms by which TVs influence computation. In this work, we address both limitations. First, we propose directly training Learned Task Vectors (LTVs), which surpass extracted TVs in accuracy and exhibit superior flexibility—acting effectively at arbitrary layers, positions, and even with ICL prompts. Second, through systematic analysis, we investigate the mechanistic role of TVs, showing that at the low level they steer predictions primarily through attention-head OV circuits, with a small subset of “key heads” most decisive. At a higher level, we find that despite Transformer nonlinearities, TV propagation is largely linear: early TVs are rotated toward task-relevant subspaces to improve logits of relevant labels, while later TVs are predominantly scaled in magnitude. Taken together, LTVs not only provide a practical approach for obtaining effective TVs but also offer a principled lens into the mechanistic foundations of ICL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces learned task vectors (LTVs), a method of directly training task vectors via gradient descent, rather than existing approaches of extracting the vectors from model activations. Additionally, the paper provides a comprehensive mechanistic interpretation of how TVs affect the model's computation. The investigation shows that, at a low level, TVs influence model computation mainly through attention-head OV circuits, and at a high level, the TVs are linearly propagated through the layers of the network.

### Strengths
- The idea of training TVs instead of extracting them from model activations is interesting and intuitive. The formulation is easy to understand and effective
- There are extensive experiments offering insights into the mechanistic understanding of TVs.
- I like the part about separating the interpretation of TVs into low-level and high-level, with different mechanisms.

### Weaknesses
A major limitation in my opinion, is that the proposed LTV is more like a PEFT method, rather than an ICL method, since extra parameters need to be finetuned on a training dataset. While it is interesting to find a "middle ground" between them, I hope to see an experimental comparison with other PEFT methods to adequately demonstrate the effectiveness of the proposed method.

### Questions
- According to Eq.3, there seems to be no need to involve ICL demonstrations for training LTVs. Is it true that LTVs are "demonstration-agnostic"?
- What is the biggest advantage of LTV compared to existing PEFT methods, say LoRA? Is it because of the placce that the extra parameters are injected? Is there an analysis about that?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes learning a task vector that is injected into a model’s hidden states to induce task behavior. Instead of extracting a vector from demonstrations, the vector is optimized directly on labeled training examples and can be placed at chosen layers and positions, with or without demonstrations. Empirically, the learned vector outperforms extraction-based baselines and often matches few-shot prompting while touching only a tiny number of parameters. The analysis traces how the vector acts through attention head output paths and shows that its effect propagates in a largely linear way, behaving like rotation toward a task subspace early in the network and magnitude scaling in later layers.

### Strengths
- The method is simple and effective. Optimize a single hidden-state vector per task and inject it where needed. This yields strong accuracy with minimal trainable parameters and integrates cleanly with or without demonstrations. 

- The mechanistic account is a real contribution. The work shows that task vectors influence predictions primarily via attention-head OV circuits, reconstructs their effect by aggregating OV transforms across layers, and validates the role of a small set of key heads through ablations. 

- The high-level propagation story is clear and useful. Layer-wise diagnostics with logit lens, logit difference, and task alignment reveal that early injections mainly rotate representations toward label directions, while later injections mostly amplify task-aligned components. This gives actionable intuition for where to inject. 

- Results are consistent across multiple datasets and models, and the method remains flexible across layers and positions. The narrative that a single learned vector can rival few-shot prompting is well supported by the reported experiments.

### Weaknesses
- Writing clarity needs a pass. There are mismatches between figure and table information and the main text. Aligning the Figure 2 caption with the plotted trend and clarifying the layer sets in Table 1 would improve trust and readability. 

- Please quantify efficiency and position it with matched baselines. It would help to report FLOPs, latency, and peak memory for training and inference, and include matched comparisons to common PEFT methods like LoRA, prefix tuning, or adapters at similar parameter budgets. This would make the practical efficiency gap clear.

- The OV-only reconstruction under-recovers the trained vector’s effect. The gap is visible but not analyzed component-wise, leaving open which pathways beyond OV account for the missing performance.

- The interpretation in section 4.2–4.3 is interesting, but it would become a more complete account with explicit theoretical grounding for when the linear approximation holds.

### Questions
- Please state the exact protocol for Figure 2. is it a single-layer injection sweep where the vector is added at one layer at a time, or a simultaneous multi-layer injection?
- In Table 1 item four the layer set differs from the main text L={0,4,.. } in main vs L={0,2,..} in table. Which schedule was actually used, and how sensitive are results to the stride of the layer set.
- Can you report training and inference FLOPs, latency, and peak memory for LTV and matched PEFT baselines like LoRA or prefix tuning at equal parameter budgets?
- What accounts for the gap between the trained vector and the OV reconstruction (Figure 5)?
- Could you give a theoretical account of when the linear propagation approximation should hold?
If several of these weaknesses are resolved, I would consider increasing my rating.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method for learning task vectors (LTVs) via direct gradient-based optimization, providing an alternative to extraction-based approaches in in-context learning. The proposed LTVs demonstrate improved performance and greater flexibility across model layers, positions, and prompt formats, as shown through evaluations on multiple language models and benchmark tasks. The authors further conduct a mechanistic analysis, identifying specific attention heads as the primary components through which task vectors affect model predictions. The study offers both a practical technique and insight into the underlying mechanisms of task representation in large language models.

### Strengths
1. The paper offers a structured mechanistic analysis of in-context learning by distinguishing between low-level effects and high-level behaviors. This decomposition is supported by empirical results and contributes to a clearer understanding of how injected task vectors influence model predictions.

2. The proposed method is extensively evaluated across eight backbone language models and seven benchmark datasets covering classification and reasoning tasks. Ablation studies further examine the impact of injection layer, token position, and multi-vector configurations, providing a comprehensive empirical characterization of LTV’s behavior.

3. The use of a directly optimized task vector in LTVs via gradient descent simplifies the construction process compared to prior extraction-based methods. Despite its simplicity, the approach demonstrates consistent improvements over baseline techniques across injection layers.

### Weaknesses
1. The evaluation primarily compares LTV against baseline task-vector methods such as Vanilla TV and Function Vector. However, several recent approaches have explored improved task-vector formulations or parameter-efficient tuning methods (e.g., LoRA), which are not included in the comparison and could provide a more comprehensive baseline context.

2. Since each task is associated with a single learned vector, the approach may face limitations in generalizing across related tasks or domains. The paper provides limited analysis regarding the transferability and compositionality of learned task vectors.

### Questions
1. The paper proposes a gradient-based approach for learning task vectors but does not discuss its computational cost. Could the authors clarify the training cost of LTVs compared to extraction-based methods or parameter-efficient tuning approaches such as LoRA?

2. The evaluation focuses on Vanilla TV and FV, without comparisons to more recent task vector methods or PEFT baselines. Have the authors considered evaluating LTV against improved task-vector variants or methods like LoRA?

3. As one vector is learned per task, it is unclear whether LTVs can generalize beyond the training distribution. Do the authors have any analysis on the generalization of LTVs across tasks or under distribution shifts?

### Soundness
3

### Presentation
3

### Contribution
3
