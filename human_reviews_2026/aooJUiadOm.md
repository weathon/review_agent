# ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 4, 6, 4

## Abstract
Mixture-of-Experts (MoE) architectures have emerged as a promising approach to scale Large Language Models (LLMs). MoE boosts the efficiency by activating a subset of experts per token. Recent works show that *fine-grained experts* substantially enriches the combinatorial flexibility of active experts and enhances model expressiveness. However, such a design is fundamentally limited by the *layer-local* routing mechanism: each layer is restricted to its own expert pool. This requires a careful trade-off between expert dimensionality and routing diversity given fixed parameter budgets. We describe ReXMoE, a novel MoE architecture that improves routing beyond the existing *layer-local* approaches by allowing routers to **re**use e**x**perts across adjacent layers. ReXMoE decouples expert dimensionality from per-layer budgets, enabling richer expert combinations without sacrificing individual expert capacity or inflating overall parameters. To this end, we propose a new *progressive scaling routing* (PSR) strategy to gradually increase the candidate expert pool during training. As a result, ReXMoE improves both language modeling and downstream task performance. Extensive experiments on models ranging from 0.5B to 7B parameters across different architectures demonstrate that ReXMoE consistently improves performance under fixed architectural dimensions, confirming ReXMoE as new design paradigm for parameter-efficient and scalable MoE-based LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes REXMoE, a novel Mixture-of-Experts (MoE) paradigm that enables the reuse of experts across groups of adjacent layers. REXMoE extends routing beyond layer-local boundaries, allowing routers to leverage experts from other layers within the same group. Moreever, it introduces a curriculum learning–based strategy, Progressive Scaling Routing, to facilitate more scalable training. The authors pre-train REXMoE models at different scales and conduct comprehensive experiments to demonstrate their strong performance across various tasks.

### Strengths
- The idea of reusing adjacent layers of experts is novel and provides a straightforward method to enable a sparser MoE with more flexible routing mechanism while keeping the number of parameters fixed.

- The authors adapt REXMoE to vLLMs and provide a detailed analysis of inference speed, which enhances the practical applicability of REXMoE.

- The authors pre-train language models of different scales from scratch and conduct comprehensive experiments to demonstrate the effectiveness of REXMoE.

### Weaknesses
- Some parts of the paper’s presentation need further improvement (see Q1 & Q2).

- Some experimental results are not convincing enough to support the authors’ claims (see Q3, Q4, and Q5).

### Questions
---
Q1: What does "mask" means in Figure 1?

The term “mask” in Figure 1 is unclear. Could the authors provide a more detailed explanation of the mask operation? 
Even after reading the full text, I find it difficult to fully understand how this operation works.

---
Q2: Is there a typo in Equation 4 and line 88?

From my understanding, the adjacent layers are partitioned into different groups. 
Accordingly, Equation 4 should be reformulated as:

$\mathcal{G} = \{\lfloor l/r \rfloor \cdot r + k \mid 1 \le k \le r\}$，

The current form of Equation 4 is somewhat unclear and may cause confusion.
Hope the authors can clarify that.

---
Q3: Does REXMoE achieve its superior model performance by accelerating the convergence of pre-training?

Could the authors pre-train a smaller REXMoE model with more tokens (e.g., 350B or 500B) and report results similar to those in Table 2?
Conducting more extensive pre-training would provide stronger evidence to convincingly demonstrate REXMoE’s effeciveness.

---
Q4: Concerns regarding the results presented in Table 3

(1) It's unclear why certain results are missing, given that these open-sourced models can be readily evaluated using tools like LM Eval.
Completing the table will provide a more comprehensive and convincing comparison.

(2) The claim of competitiveness between REXMoE and Llama 2 (as stated in the caption) is currently unconvincing to me.

(3) Can authors include results for other competitive open-source models, such as OLMoE [1].
Additionally, to better support claims of superiority in reasoning and knowledge-intensive tasks, the authors can conduct further evaluations on benchmarks like GSM8K and MMLU.

[1] OLMoE: Open Mixture-of-Experts Language Models

--- 
Q5: Can the the authors provide an inference speed comparison for the 7B REXMoE?

On one hand, I am curious about how the inference efficiency of larger-scale REXMoE compares to that of standard MoE models.

On the other hand, when r is larger, it becomes difficult to include r layers of experts within a single GPU (worker), and could further complicate the  pipeline parallelism.
I hope the authors can clarify how this challenge can be addressed.

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
3

### Summary
Recent studies have shown that fine-grained experts can significantly enhance the combinatorial flexibility of activated experts and improve the expressiveness of models. Traditional Mixture-of-Experts (MoE) models are constrained by layer-local routing mechanisms, where each layer is limited to its own expert pool. This paper proposes REXMOE, a novel MoE architecture that allows routers to reuse experts across adjacent layers, thereby breaking through the routing limitations of existing layer-local methods. Specifically, a Progressive Scaling Routing strategy is proposed in REXMOE: this strategy gradually expands the candidate expert pool during training to reduce language modeling loss and improve downstream task accuracy. Experiments demonstrate that REXMOE consistently enhances models' language modeling capabilities and downstream task performance across different model scales and architectures.

### Strengths
This paper proposes REXMOE, a novel MoE architecture that allows routers to reuse experts across adjacent layers, thereby breaking through the routing limitations of existing layer-local methods.

Specifically, a Progressive Scaling Routing strategy is proposed in REXMOE: this strategy gradually expands the candidate expert pool during training to reduce language modeling loss and improve downstream task accuracy.

### Weaknesses
The authors note that REXMOE can serve as a new MoE paradigm. However, due to its cross-layer routing mechanism, it remains unclear whether parameter scaling may affect pipeline parallelism (pp) or expert parallelism (ep) strategies, thereby hindering efficient scalability.

The authors only conducted comparative experiments on MoE models with 0.5B and 2.3B total parameters, and the results show that the performance improvement brought by the proposed REXMOE is not significant. It is plausible that the improvement is insignificant when the parameter size is small; if feasible, supplementary comparative experiments between 7B REXMOE and vanilla MoE are recommended.

The authors' ablation experiments indicate that Expert reuse itself yields almost no improvement, whereas the PSR strategy contributes substantially. This raises questions about consistency with the motivation of combinatorial numbers mentioned in the introduction. It is further worth exploring whether cross-layer MoE reuse in REXMOE is necessary, and whether PSR can be applied to gradually expand the number of experts within the same layer. If possible, supplementary experiments or theoretical explanations are advised.

The cross-layer MoE reuse mechanism and PSR strategy in the proposed REXMOE are highly likely to cause expert load imbalance. The key question is: how to alleviate this load imbalance issue within this new paradigm?

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ReXMoE, a parameter-efficient mixture of experts (MoE) architecture. By sharing the expert pool across adjacent layers, the authors increase the diversity of experts. As a training methodology, they introduce progressive scaling routing (PSR). Furthermore, through an experimental analysis of expert activation frequency with respect to reuse frequency, the paper investigates cases where the MoE architecture may fail.

### Strengths
- Compared to existing models, the proposed approach improves model performance by increasing the expert pool for each layer while maintaining the same number of parameters.
- To address the limitation that simply expanding the expert pool leads to only marginal performance improvements, the authors propose a novel training methodology. Through an ablation study, they demonstrate not only the effectiveness of the new architecture but also its practical applicability.
- They observed that task-specific experts are activated more frequently compared to the vanilla MoE.

### Weaknesses
- As the size of the expert pool increases, there is a significant slowdown in the prefill stage. While it is acknowledged that decoding speed plays a more critical role in inference, the slowdown during the prefill phase becomes a weakness of this methodology, especially considering that token sequences can be quite long in recent large language models (LLMs).
- The authors conducted experiments by varying the reuse frequency $r$, and according to their claims, performance improves as $r$ increases. However, this trend cannot be clearly explained for the cases of $r = 16, 32$. Although they observed a performance degradation phenomenon due to routing collapsing into a small subset of experts, the paper lacks sufficient theoretical analysis and explanation of the underlying causes. Furthermore, the proposed PSR strategy may introduce additional difficulty in hyperparameter tuning. It also remains unclear whether this approach can be generalized to other models.
- There is a lack of experiments involving an upper-bound model. Considering the reuse frequency, it would be beneficial to compare the performance improvements against an upper bound obtained by increasing the number of experts per layer.
- Despite the use of PSR strategy, the load imbalance problem becomes more severe as the reuse frequency $r$ increases.

### Questions
Q.1. The training procedure differs somewhat from that of the conventional MoE. What would happen if the PSR strategy were also applied when training a vanilla MoE?

Q.2. How is the active ratio in Section 4.4 computed numerically? It would be better to also report the deviation values.

Q.3. While load imbalance is generally considered a problem, is task-specific specialization necessarily desirable?

Q.4. In the LogiQA, SIQA, and WinoGrande benchmarks, the trend with respect to reuse frequency is not clearly observed. Do the authors have any hypotheses or explanations for this?

Starting from the next question, I’ll be asking about parts I’m not sure about — whether they are typos or actual errors.
1. The equation in Eq. (4) does not seem to align with the description in the text. Shouldn’t the element term in group $G$ be $r*\lfloor l/r \rfloor + k$? Also, the indexing starting points of $l$ are unclear. It seems that $l$ uses 0-based indexing, but this is not explicitly stated, which makes it confusing.
3. In the description of Figure 4, the text refers to Figure 3 — is this a typo?

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
5

### Summary
To address the limitations of routing mechanisms in Mixture-of-Experts (MoE) models, this paper proposes a novel MoE architecture named REXMOE, which enables routers to reuse experts across adjacent layers, thereby overcoming the routing constraints of existing layer-local methods. Corresponding to this improvement, a new progressive scaling routing (PSR) strategy is put forward to gradually expand the candidate expert pool during training.

### Strengths
This paper designs REXMOE, a method that breaks the limitation of layer-local routing in MoE architectures and proposes a Progressive Scaling Routing strategy in REXMOE, which gradually enlarges the candidate expert pool during training, thereby reducing language modeling loss and improving downstream task accuracy

### Weaknesses
1. There is a lack of theoretical analysis on the effectiveness of REXMOE, particularly regarding the expert combination numbers and PSR mentioned by the authors.
2. Ablation experiments indicate that it is PSR rather than cross-layer expert reuse that yields substantial improvements. Thus, one may question the necessity of cross-layer reuse—given that such reuse would affect pipeline parallelism (pp) and expert parallelism (ep) strategies when the model scales. This is particularly critical for determining whether REXMOE can qualify as a new MoE paradigm.
3. Comparative experiments are only conducted on models with 0.5B and 2.3B total parameters, with no comparisons on larger models. From the experimental results, the improvements brought by REXMOE are not significant.

### Questions
No further questions, see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes REXMOE that enables cross-layer expert reuse to improve routing flexibility without inflating model parameters. Unlike conventional MoE designs where each layer maintains its own isolated expert pool, REXMOE allows routers to select from experts across $r$ adjacent layers, effectively expanding the candidate pool from $N$ to $rN$ experts while adding only negligible router parameters. To stabilize training with enlarged expert pools, the authors introduce a Progressive Scaling Routing (PSR) strategy that gradually increases the number of available experts during training. Experiments on models ranging from 0.5B to 7B parameters demonstrate consistent improvements in both language modeling perplexity and downstream task accuracy compared to vanilla MoE baselines, with particularly notable gains on reasoning-intensive benchmarks. The work provides a new architectural degree of freedom for designing parameter-efficient MoE-based language models, though practical adoption may be constrained by observed prefill latency degradation during inference.

### Strengths
- The paper introduces a conceptually elegant approach to MoE design by enabling cross-layer expert reuse, representing a meaningful departure from conventional layer-local routing. While parameter sharing exists in prior work, applying it specifically to MoE blocks with Progressive Scaling Routing offers a fresh perspective on balancing expert capacity and routing diversity. The minimal overhead (only router parameters) and consistent improvements across multiple model scales (0.5B to 7B) demonstrate practical viability and reasonable generalizability across different architectural configurations.

- The empirical study is thorough, spanning different model sizes, architectural variants, and evaluation metrics. Results show consistent improvements across most benchmarks, particularly on reasoning tasks. Ablation studies examining reuse frequency, PSR variants, and component contributions provide useful insights, while qualitative analysis of expert activation patterns offers preliminary evidence of task-specific specialization, though deeper mechanistic understanding would strengthen these observations.

- The work tackles a relevant challenge in scaling MoE architectures: enriching expert combinations without inflating parameters or reducing expert capacity. Given industry trends toward fine-grained MoE designs (Qwen3, DeepSeek-V3), this research direction has clear practical significance.

### Weaknesses
- The results in Figure 2(a) reveal substantial prefill speed degradation (up to 77% slowdown for short sequences with R8 configuration), which significantly limits the practical applicability of REXMOE in latency-sensitive applications. While the authors acknowledge this issue stems from increased I/O operations due to the larger expert pool, they do not explore potential mitigation strategies or provide detailed profiling to identify the exact bottlenecks. The paper would benefit from: (1) a breakdown of where time is spent during prefill (expert loading, routing computation, actual computation), (2) analysis of whether expert caching or other optimization techniques could reduce this overhead, and (3) discussion of use cases where the prefill penalty is acceptable versus problematic. Without addressing these concerns, practitioners may hesitate to adopt REXMOE despite its modeling advantages, especially for interactive applications where prefill latency directly impacts user experience.

- The experimental evaluation focuses primarily on TopK routing with specific architectural choices (GQA, shared experts in some configurations), but does not adequately explore how REXMOE interacts with other important MoE design decisions. For example, the paper does not investigate: (1) compatibility with different routing mechanisms beyond TopK (e.g., expert choice routing, soft routing), (2) interaction with different expert capacity factors and load balancing auxiliary losses, (3) performance under different expert granularities when controlling for total parameters. Table 3 compares against external baselines with different training data and configurations, making it difficult to isolate the contribution of expert reuse. More controlled comparisons—such as taking an existing MoE architecture and applying REXMOE versus carefully matched baselines—would better demonstrate the method's generalizability and help identify where it provides the most value.

- The paper claims that REXMOE "decouples expert dimensionality from per-layer budgets," but the experimental validation of this claim is limited. All experiments use fixed architectural configurations (Table 1), and there is no systematic study exploring how different combinations of expert size and reuse frequency affect performance under constant parameter budgets. For instance, the paper does not compare: (1) a model with 64 larger experts using R4 reuse versus one with 256 smaller experts using R1 (no reuse) at the same total parameter count, (2) whether maintaining expert capacity while increasing reuse outperforms decreasing expert capacity to accommodate more local experts, or (3) optimal trade-offs between expert dimensionality, number of experts, and reuse frequency across different model scales. Without these comparisons, it remains unclear whether the performance gains primarily come from the increased combinatorial flexibility of expert routing or from other factors, and practitioners lack guidance on how to configure these architectural choices for their specific constraints.

### Questions
- Can you provide a more detailed analysis of the prefill latency degradation and potential mitigation strategies?

- How does REXMOE compare against alternative approaches for increasing routing flexibility under controlled conditions?

- What strategies can address expert collapse at higher reuse factors, and what are the fundamental scaling limits?

### Soundness
3

### Presentation
2

### Contribution
3
