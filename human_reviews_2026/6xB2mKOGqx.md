# Routing-Deconstructed LoRA in Federated Fine-Tuning

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
The integration of Large Language Models (LLMs) with Federated Learning (FL) offers a promising approach to privacy-preserving Parameter-Efficient Fine-Tuning (PEFT). However, resource and data heterogeneity in FL cause differences in local knowledge distribution across clients. As a representative PEFT approach, LoRA still faces three key challenges in such settings: aggregation noise, knowledge contamination, and aggregation distortion. To address these issues, we propose Routing-Deconstructed LoRA (RD-LoRA). Building on an alternating freezing strategy to mitigate aggregation noise and concurrently reduces communication cost, RD-LoRA further introduces two novel components. For knowledge contamination, we design a Server-Client Routing Deconstructor (SCRD) that separates shared semantics from local biases, retaining fine-grained knowledge with semantic consistency. To address aggregation distortion, we propose a Poly-Consensus Aggregation (PCA) mechanism that uses adaptive weighted averaging to align global LoRA parameters with heterogeneous client distributions, thus correcting the global update direction. Extensive experiments demonstrate that RD-LoRA is effective and robust in both homogeneous and heterogeneous settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes RD-LoRA for federated LoRA fine-tuning under client/data heterogeneity. It builds on alternating freezing (only A or B is updated per round) to lower communication and aggregation noise, and introduces a routing matrix split to decouple shared semantics from client-specific signals and reduce knowledge contamination. On the server, a PCA applies position-wise adaptive weights with historical regularization to mitigate aggregation distortion when fusing A/B/R. The method handles heterogeneous ranks via zero-padding/truncation before aggregation and redistribution. Experiments on Llama-2 and TinyLlama report consistent gains over baselines in both homogeneous and heterogeneous settings, with communication similar to alternating-freeze methods.

### Strengths
The paper clearly pinpoints three key challenges in federated LoRA fine-tuning: aggregation noise, knowledge contamination, and aggregation distortion. The solution proposed focus on the specific challenges and has clear motivation.

Empirical results are strong and consistent across Llama-2 and TinyLlama in both homogeneous and heterogeneous rank settings, with low communication comparable to alternating-freeze methods and ablations showing that removing SCRD or PCA hurts performance.

The paper clearly states the strength and weakness of previous works, exploring the accuracy, communication, and heterogeneity.

### Weaknesses
The pipeline is overly complex for practical LLM fine-tuning, hinging on alternating A/B rounds, an extra routing matrix, and server-side PCA; moreover, the experiments use very heavy schedules (e.g., 200 communication rounds with 10 local epochs per round), which may not be necessary in many real deployments and can confound the contribution of the routing/PCA design.

The treatment of heterogeneous ranks lacks a stronger theoretical underpinning: cross-rank alignment ultimately relies on zero-padding or truncation, which the paper itself acknowledges as a limitation and potential source of degradation, this seems partly at odds with the broader claim of mitigating aggregation noise.

Some figures and expressions are not sufficiently precise; for instance, the RD-LoRA overview (Figure 2) is visually crowded with many modules/arrows, making the flow difficult to understand quickly.

### Questions
Could you report the test accuracy trajectory over all 200 communication rounds, including variance across seeds, and clarify at which round RD-LoRA first surpasses baselines?

What is the base model’s zero-shot (or supervised) accuracy on each benchmark before any federated fine-tuning, so we can quantify absolute gains?

How does the method handle client selection or partial participation per round (e.g., random subset of clients)? Please specify any changes to routing updates, PCA aggregation, and convergence behavior under varying participation rates.

### Soundness
3

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
4

### Summary
This paper proposes Routing-Deconstructed LoRA (RD-LoRA), a federated fine-tuning framework for large language models that introduces two main components: the Server–Client Routing Deconstructor (SCRD) to separate global and local knowledge, and the Poly-Consensus Aggregation (PCA) mechanism, which employs adaptive weighted averaging for aggregation under data heterogeneity. The work aims to alleviate aggregation noise, knowledge contamination, and aggregation distortion in federated LoRA fine-tuning.

### Strengths
- The proposed method demonstrates clear empirical effectiveness, especially under non-IID settings, and achieves superior performance over several strong baselines.
- The experimental section is comprehensive in terms of datasets, models, and metrics, providing a convincing empirical comparison.

### Weaknesses
- Equation (2) is imprecise and potentially misleading. $\frac{1}{k}\sum \Delta W_i$ is not directly equal to $\frac{1}{k}\sum \Delta B_i \times \frac{1}{k}\sum \Delta A_i$ but somehow approximated by it.
- The discussion of bias from averaging B and A (Equation 3) lacks depth. The paper shows that the left-hand side differs from the right-hand side but does not analyze whether this necessarily constitutes a biased or sub-optimal estimate. More intuition or references to prior analyses of LoRA aggregation bias would strengthen the argument.
- The reasoning of theorem 3.1 of “ effectiveness of SCRD arising from reduced gradient scale” is unclear. The theoretical analysis focuses solely on local updates, while in federated settings $R_{\text{global}}$ is produced through server-side aggregation. It is not evident how the local analysis guarantees separation between local and global knowledge in practice.
- A key baseline, FedIT (FedAvg + LoRA), is missing. Given that RD-LoRA claims novelty on aggregation scheme, direct comparison with FedIT is essential as a comparison with the most naive method is necessary.
- Only two clients out of ten are sampled per communication round, which may not adequately represent real-world FL participation patterns and can bias results.
- No indication is given that experiments were repeated with multiple random seeds. Reporting variance would improve reliability.

### Questions
- **OSIM implementation:** The paper does not describe how the per-branch logits $s_i[u,v]$ in the Omni-Scale Integration Module are computed.
- **PCA specificity:** The proposed Poly-Consensus Aggregation seems generally applicable to any matrix aggregation, not specifically tailored to LoRA. Can author share more insights on how LoRA can be beneficial on PCA?
- **Assumption in Theorem 3.1:** Is it realistic to assume the symmetric part of $R_{\text{global}}^\top R_{\text{client}}$ is positive definite?

### Soundness
3

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
3

### Summary
In this paper, the authors propose RD-LoRA, a communication-efficient federated fine-tuning framework for large language models using parameter-efficient LoRA adapters, designed for realistic non-IID settings where clients have heterogeneous data distributions and even different LoRA ranks.  It introduces SCRD—a routing mechanism separating global and client-specific adaptation to prevent knowledge contamination—and PCA, an adaptive aggregation method that selects dominant client updates per parameter and stabilizes training with historical regularization.  The authors conduct experiments on Llama2-7B and TinyLlama show RD-LoRA outperforms existing methods on MMLU and MT-Bench.

### Strengths
1. The authors provide a practical problem formulation, and identify three concrete bottlenecks in scalable federated LoRA (aggregation noise, knowledge contamination, distortion) .

2.The authors introduce a routing matrix split into global (frozen) and client-specific (trainable) components, cleanly separating shared knowledge from local bias, with theoretical support for improved update stability.

3. The authors propose fine-grained aggregation via pca, which learns position-wise reliability weights and uses historical alignment to stabilize updates.

4. The paper measures and reports per round upload and download cost for each baseline.

### Weaknesses
1. Regarding experiments:

 a. the authors only adopt Llama2 7B and TinyLlama 1.1B as the backbone model, how about other models such as Qwen series model?

b. The training loop alternates B round and A round. But we only see two clients per round in experiments. How does that scale to a larger pool of clients across two hundred rounds.

c. Limited ablations on PCA internals:
   PCA has several moving parts. Poly Fusion Gate to create a contextual prior. Omni Scale Integration Module to learn per position weights. Historical steady alignment with a lambda term. The paper provides a single ablation that swaps PCA with FedAvg style averaging, and then discusses cosine similarity and singular value retention. That shows PCA helps, but it is still hard to tease apart which sub part of PCA is doing the heavy lifting. For example, is the historical regularizer alone already enough to stabilize training, or is the per position attention absolutely necessary. More granular ablations would make PCA easier to adopt by others.

### Questions
see weakness 1.a, 1.b, 1.c

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method named RD-LoRA, which aims to improve federated LLM fine-tuning using LoRA-based methods. RD-LoRA seeks to address the limitations and challenges of existing PEFT methods for federated tuning of LLMs, namely aggregation noise, knowledge contamination, and aggregation distortion. The authors claim that RD-LoRA can address all the aforementioned challenges simultaneously. The key aggregation method used by RD-LoRA is the Poly-Consensus Aggregation approach, which adaptively aligns global model weights with local model weights. Experimental results have been presented to demonstrate the effectiveness of RD-LoRA.

### Strengths
- The paper is well written and well motivated.  
- Improving the performance and efficiency of federated PEFT over LLMs seems to be a reasonable research direction to pursue.  
- The proposed RD-LoRA method is easy to follow and intuitive.  
- The experimental results of the paper look promising.

### Weaknesses
- The authors proposed a relatively complex method, but the accuracy improvement appears to be marginal (if not very marginal) based on the main results shown in Table 1 and Table 2, especially compared to FlexLoRA.  
- The experiments were conducted on somewhat outdated models, for example, Llama2. We already have Llama4 and even stronger open source models, so why still fine-tuning on Llama2?  
- The fine-tuning tasks also seem a bit outdated; it would be more interesting to show results on more challenging benchmarks such as AIME 2025.  
- It is not clear what the implementation and hyperparameter tuning overheads are when using RD-LoRA compared to other simpler methods.

### Questions
Please focus on addressing the concerns in Weaknesses.

- If the performance gain is marginal, why using RD-LoRA?
- What does the performance of RD-LoRA look like on say Qwen3 finetuning and for harder tasks, e.g., some reasoning benchmarks.
- What are the overheads introduced by RD-LoRA ?

### Soundness
3

### Presentation
3

### Contribution
2
