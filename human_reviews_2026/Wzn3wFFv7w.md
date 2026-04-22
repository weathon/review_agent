# MuEdit: A Lightweight yet Effective Multi-task Model Editing Method

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Large language models (LLMs) are prone to misinterpreting instructions and generating incorrect responses, stimulating the development of model editing methods. Existing model editing methods, however, face limitations in handling multi-task knowledge updates due to interference between different tasks. This paper firstly by analyzing the shortcomings of traditional editing methods and Fang's null-space projection method, which fails to generalize to multi-task scenarios because the null-space projection matrix of the current task may not lie within that of previous tasks, causing conflicts. To tackle this, we propose a new concept, the \textbf{Conflict Index}, to measure conflicts between two tasks' editing objectives. We then design two methods: \textbf{finding the optimal editing path} to minimize the total conflict index and \textbf{using a low-rank matrix expression method based on the conflict index to expand the null-space dimension} when conflicts remain high. Experimental results show that our proposed Mu-Edit method effectively alleviates multi-task editing conflicts, outperforming existing baseline methods in various metrics across multiple tasks and able to preserve abilities in general domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper points out that existing knowledge editing methods cannot effectively handle multi-task editing, and conflicts exist between different tasks, which affects editing performance. This paper proposes a multi-task editing framework that uses two complementary strategies to resolve multi-task conflicts.

### Strengths
1. This paper is well-written, with clear logic and concise readability.

2. It conducts extensive experiments, including comparative experiments on models of different types and scales.

3. The authors provide a wide range of comprehensive evaluation metrics.

### Weaknesses
1. First, the paper points out that Mu-Edit relies on low-rank decomposition to achieve editing across multiple tasks, but lacks comparative results with full-model fine-tuning and LoRA.

2. Second, the motivation is insufficient. Knowledge editing is often used for low-cost knowledge updates. It remains unclear whether the setting of performing knowledge updates across multiple tasks is reasonable, and what advantages this setting offers over full-model fine-tuning or LoRA.

3. Finally, previous works (such as D4S [1] and AlphaEdit [2]) have addressed the issue of model performance degradation after editing multiple samples. Based on the existing experimental results, Mu-Edit fails to demonstrate this capability, which casts doubts on its practical application.

**References**

[1] Reasons and Solutions for the Decline in Model Performance after Editing

[2] Alphaedit: Null-space constrained knowledge editing for language models

### Questions
See Weaknesses.

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
This paper proposes MuEdit, a lightweight and effective method for multi-task model editing. The authors argue that existing model editing approaches suffer from strong interference when updating multiple tasks simultaneously. To address this, they introduce a novel metric called the Conflict Index to quantify conflicts between task-specific editing objectives.
Based on this metric, they design two strategies Optimal Editing Order Selection, and Conflict-Guided Low-Rank Matrix Approximation to solve this problem. 
Extensive experiments on multiple benchmarks and two model (Llama3-8B and GPT2-XL) demonstrate that MuEdit outperforms state-of-the-art methods such as ROME, MEMIT, AlphaEdit, and AnyEdit, while maintaining strong general-domain capabilities.

### Strengths
1.	Novel problem formulation – The paper is the first to explicitly define and analyze multi-task model editing from a null-space conflict perspective.
2.	The theoretical foundation based on linear algebra (null-space and rank analysis) is sound and logically consistent, which is an interpretable approach.
3.	This paper Covers five heterogeneous tasks, Includeing ablation studies, sensitivity analysis, and significance testing (p < 0.05). MuEdit achieves substantial improvements in multi-task editing and maintains general-domain abilities better than all baselines.

### Weaknesses
1.	Although the Conflict Index is an interesting idea, it is heuristic and lacks a rigorous theoretical connection to optimization conflicts (e.g., gradient interference or Fisher information).
2.	The “optimal editing order” involves a factorial search over tasks (O(N!)); the paper does not clarify how this is handled in practice.
3.	The method assumes all tasks are known beforehand; it is unclear how Mu-Edit performs when new tasks arrive incrementally.
4.	Results are shown on GPT2-XL and Llama3-8B; it remains uncertain whether the conclusions hold for larger models like Llama3-70B.

### Questions
1. How scalable is the Conflict Index computation and order search when the number of tasks exceeds 10?

2. Does low-rank approximation reduce the model’s knowledge capacity, potentially leading to long-term forgetting?

### Soundness
3

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
The author proposed a novel concept termed the Conflict Index, which quantifies the degree of conflict between the editing objectives of two tasks. Building on this idea, the author introduced a method that integrates two key strategies: 1) optimal edit path identification; 2) a low-rank matrix approximation method based on the conflict index to expand the null-space dimension.

### Strengths
- The author provides a clear formulation of the multi-task editing problem and introduces the Conflict Index, an insightful and valuable concept. 
- The idea of leveraging the common null space and employing low-rank matrix decomposition to mitigate task conflicts is both inspiring and technically interesting.

### Weaknesses
- The paper strongly lacks analysis and experiment to support its idea. 

(1) No further experiment to support the key observation of this paper, which is that during sequential multitask editing, the new knowledge matrix Kn compresses the null space of Kn−1 (in Sec. 3.2) after the teaser figure. 

(2) In the Sec. 4.1 and the appendix, the main experiment was still conducted on the Llama3-8B and GPT2-XL, which is a pretty old combination. The author should add more experiments on SOTA LLMs like Qwen2.5.

- The proposed method mainly addresses the sequential editing scenario, which corresponds to lifelong model editing in practice. However, the Best Order concept introduced in Sec. 3.3.1 is not realistic in real-world applications, as the future knowledge to be edited is inherently unpredictable. If multiple pieces of knowledge are already available as a batch, conventional fine-tuning would be a more appropriate choice. This, however, contradicts the core motivation of knowledge editing, which is to enable efficient and localized updates for small pieces of knowledge at a time.

### Questions
See above

### Soundness
2

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
3

### Summary
This paper zeroes in on a pretty practical problem in model editing how to update a model for multiple tasks at once without everything falling apart. The authors argue, pretty convincingly, that the interference comes from conflicting editing objectives. Their big idea is a "Conflict Index," a new metric to quantify how much two tasks' null-spaces clash. Based on this, they propose Mu-Edit, which is a two-part strategy. First, it figures out the best sequence to apply the edits to minimize total conflict. Second, if the clash is still too severe, it actively expands the common null-space by running a low-rank approximation (SVD) on the knowledge matrix of the most "conflicting" task. The experiments on a few multi-task benchmarks seem to back this up, showing it preserves performance better than existing methods.

### Strengths
1. The paper addresses an important and under-explored problem of multi-task model editing, which is more realistic than sequential single-task editing.
2. The introduction of the Conflict Index provides a quantitative way to measure and analyze conflicts between different editing tasks.
3. The proposed optimization strategies (optimal editing path and low-rank approximation) are well-motivated and appear to effectively address the multi-task conflict problem.
4. The method demonstrates strong empirical performance across multiple tasks while maintaining general model capabilities.

### Weaknesses
1. The $O(N!)$ complexity for finding the best edit order is a major scalability problem. The practical greedy solution is hidden in the appendix.
    
2. SVD is a blunt tool. The long-term, cumulative impact of repeatedly cutting rank on multiple tasks isn't really explored.
    
3. The method seems fragile. The worst-case example (43.7% reduction) is dangerously close to the 45% failure point, suggesting it could easily break.
    
4. The reliance on a large, static K matrix for each task feels brittle and may not handle evolving tasks or unseen knowledge well.

### Questions
1. The $O(N!)$ order search is impractical. Is the greedy algorithm from the appendix the intended method? What about other ordering heuristics?

2. Regarding the SVD, your worst-case (43.7% reduction) is right at the 45% performance cliff. What happens when a task pair requires a 50% reduction? Does the method just fail?

3. Also, why did performance get worse in Table 9 when optimizing over 4 or 5 tasks instead of 3? This seems counter-intuitive and suggests a potential unaddressed issue.

### Soundness
3

### Presentation
3

### Contribution
3
