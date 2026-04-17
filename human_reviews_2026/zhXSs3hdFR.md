# Orthogonal Calibration for Asynchronous Federated Learning

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Asynchronous federated learning mitigates the inefficiency of conventional synchronous protocols by integrating updates as they arrive. Due to asynchrony and data heterogeneity, learning objectives at the global and local levels are inherently inconsistent—global optimization trajectories may conflict with ongoing local updates. Existing asynchronous methods simply distribute the latest global weights to clients, which can overwrite local progress and cause model drift. In this paper, we propose OrthoFL, an orthogonal calibration framework that decouples global and local learning progress to reduce interference. In OrthoFL, clients and the server maintain separate model weights. Upon receiving an update, the server aggregates it into the global weights via a staleness-aware moving average. For client weights, OrthoFL computes the global weight shift during the client's delay and removes its projection onto the direction of the received update. The resulting parameters lie in a subspace orthogonal to the client update and preserve the maximal information from the global progress within the orthogonal hyperplane. The calibrated shift is then merged into the client weights for further training. Extensive experiments demonstrate OrthoFL improves accuracy by 9.6% and achieves a speed-up of 12× compared to synchronous methods. Moreover, it consistently outperforms state-of-the-art asynchronous baselines under various delay patterns and heterogeneity scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper targets the challenges of potential gradient conflict in AFL due to the asynchronous updates between the server and clients. The authors propose to decouple global and local learning progress by orthogonalizing their weights (per client) before applying it to client weights. Experiments show higher accuracy and substantial speed-ups compared to synchronous and consistently outperform several asynchronous baselines across various datasets.

### Strengths
The authors propose an interesting perspective of looking at the client drift in AFL as from "spatial drift" and "temporal staleness", though the concept has always been there.
The paper is well written with clear motivation and algorithmic description.
The results show solid empirical improvements on various datasets against both synchronized FL and AFL.

### Weaknesses
The major ideas, staleness-aware and orthogonal learning have both been quite well explored, relevant works can be easily found in papers published in recent years. Therefore, the novelty and originality become quite limited. 

The convergence rate has only been shown with empirical results without math analysis.

For larger models, the operation of orthogonalization may raise costs worth considering. This is overlooked in the draft. 

Removing the projection may not help reduce bias, it may sometimes even favor the frequent clients resulting in larger bias.

### Questions
1. Please clarify the distinction between this work and related works that focus on staleness-awareness and orthogonal learning in AFL and in FL, because some orthogonal learning approaches in synchronized setup can be applied to AFL as well. 

2. Please analyze the global convergence rate. 

3. Could the authors discuss the orthognalization operation's cost?

4. What do the authors think about the potential bias problem? Could orthognoal help it? How?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Can the authors provide any theoretical justification for LEAN's convergence and stability?
How sensitive is the method to the choice of library size (n) and checkout size (m), and how should they be set?
What is the inference cost of the final summed-pair adapter compared to a standard LoRA adapter?
Have the authors considered more intelligent strategies for sampling basis pairs from the library (e.g., based on staleness)?

### Strengths
The use of orthogonal projection to decouple conflicting updates is an elegant, geometrically intuitive, and novel solution. The evaluation uses strong baselines, diverse tasks, and realistic delay simulations to convincingly demonstrate the method's robustness and superiority. The evaluation is a major strength, using strong baselines, diverse tasks, and realistic delay simulations to convincingly demonstrate the method's robustness and superiority.

### Weaknesses
1. The theoretical analysis is limited. The analysis in Appendix I is a one-step analysis, showing that under certain conditions, a single step of ORTHOFL is better than a baseline. This is insightful but does not constitute a full convergence proof over the entire training process. For a top-tier conference like ICLR, a more complete theoretical treatment guaranteeing convergence would significantly strengthen the paper.
2. The ORTHOFL algorithm, particularly the server-side implementation (Algorithm 1), requires the server to store the historical state W(τ+) for each client currently performing local training. In a large-scale system with thousands or millions of clients, where many could be "in-flight" at any given time, the memory overhead of storing a full model copy for each active client could become a significant scalability bottleneck. While the client-side version (Algorithm 2) mitigates this, it places more burden on the client, and the paper could discuss this trade-off more explicitly.
3. The method projects the global shift ∆ onto the client's update direction ∆m. ∆m is defined as W(t)m - W(τ+)m, which represents the total change in the client model over E local epochs. This vector is an aggregation of many local gradient steps. It's not immediately obvious that this single aggregate direction is the most representative one to project against.

### Questions
1. What are the main challenges in extending the one-step analysis to a full convergence proof?
2. How do you address the potential server-side memory bottleneck from storing historical states for many concurrent clients?
3. Why does layer-wise projection outperform full-model projection? Does this imply drift is a layer-specific issue?
4. Have you experimented with projecting against other directions besides the total client update vector ∆m?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the ORTHOFL method to decouple global and local learning progress, thereby reducing model drift. Additionally, this approach optimizes weights on both client and server sides to enhance accuracy. Based on these methods, the authors present some new result on asynchronous federated learning.

### Strengths
1. The experiments in this paper are conducted comprehensively, though some minor flaws exist in certain details.

2. Most methods mentioned in the paper are elaborated upon in detail within the appendix.

3. By integrating the experiments in the main text with the experimental results in the appendix, it is evident that the proposed method in this paper demonstrates certain generalization capabilities and cutting-edge potential.

### Weaknesses
1. The author mentions improving model performance by optimizing weights on both client and server sides. It is suggested to add 1-2 adaptive weighting methods to the comparative experiments to demonstrate the model's advantages.

2. Table 2 presents the performance of the proposed model and comparison models across five datasets. However, the subsequent ablation experiments only include three datasets. It is recommended to include the remaining two datasets here.

3. The experimental sections in Figures 4 and 5 also utilize two or three datasets. It is recommended to explain why only these datasets were selected. Alternatively, all datasets should be included as in the previous benchmark experiments.

4. Figures and tables are not consistently referenced in the text. For example, Figure 3 is not referenced within the body.

5. The formatting of references appears inconsistent. It is recommended to thoroughly proofread them according to the conference's formatting requirements.

6. The text contains numerous long, complex sentences separated by commas, along with some fundamental grammatical issues. It is recommended that the author carefully proofread the entire document to enhance the reader's experience.

7. It is recommended to include an overall model architecture diagram to help readers understand the model's structure.

### Questions
1. Will the orthogonal direction gradually deviate from the global optimum? Is there bounded deviation?

2.Does orthogonal calibration guarantee long-term convergence?

3.Beyond the advantages outlined in the paper, has consideration been given to risks such as privacy breaches? Have the authors implemented corresponding measures to address these risks?

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
4

### Summary
This paper introduces ORTHOFL, an asynchronous federated learning framework designed to mitigate interference between global model progress and local client training. The core idea is to decouple the global and local model weights. When a client's update arrives, the server first aggregates it into the global model using a staleness-aware moving average. Then, to update the client's model for its next training round, it calculates the global model's shift during the client's delay period and removes the component of this shift that is parallel to the client's own update direction. This is done via a layer-wise orthogonal projection. The resulting "calibrated" global shift is then sent back to the client. Experiments on diverse datasets and under various delay simulations show that ORTHOFL improves accuracy and convergence speed compared to synchronous and other asynchronous baselines.

### Strengths
* The core idea of using orthogonal calibration to decouple and manage the interference between global and local updates is novel and intuitive.
* The paper is well-written and clearly presented, making the motivation and methodology easy to follow.
* The experiments are comprehensive, covering multiple datasets, a strong set of synchronous and asynchronous baselines, and various simulated delay distributions.

### Weaknesses
* The paper claims layer-wise projection is "memory-efficient" in Section 4.1. However, storing the model weights for projection, whether layer-by-layer or as a single flattened vector, seems to require the same amount of storage space. Could the authors clarify this claim?
* The method excludes the component of the global shift that is projected onto the local update. Should this information always be discarded? Since local models can be biased due to non-IID data, this projected component (whether aligned or opposing) might contain useful information to either reinforce or correct the local update direction.
* The ablation study shows that removing the global moving average (ORTHOFL w/o MA) hurts performance more than removing calibration (ORTHOFL w/o Calib.). This suggests that modifying the global aggregation contributes more to the performance gain than the orthogonal calibration itself. Does this finding support the concern that excluding the projected component might be discarding useful information?
* Figure 1 effectively visualizes the oscillating global weight changes to motivate the problem. It would be very helpful to include a similar plot for ORTHOFL. A direct comparison would make it easier to understand the effect of the proposed calibration method.

### Questions
* How does the magnitude of the calibrated global shift, compare to the magnitude of the original global shift? Does this relationship change as training progresses? As these indirectly influence the local step size for updating the local parameter in Eq.(7). 

* Following the above, the global aggregation weight βt in Eq. (3) currently only depends on staleness. The corresponding adjustment of global aggregation weights of different clients also remains a question. Have the authors considered adjusting βt based on information from the projection?

### Soundness
3

### Presentation
3

### Contribution
3
