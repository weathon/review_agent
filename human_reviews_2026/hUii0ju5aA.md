# Decoupling Safety into Orthogonal Subspace: Cost-Efficient and Performance-Preserving Alignment for Large Language Models

- Decision: Reject
- Scores: 2, 4, 2, 8, 6

## Abstract
Safety alignment is essential for building trustworthy artificial intelligence, yet it remains challenging to enhance model safety without degrading general performance. Current approaches require computationally expensive searches for the optimal proportion of safety-critical and general-purpose data to balance safety and general performance, incurring high costs with limited gains. In this work, we show that LoRA-based Refusal-training enables performance-preserving safety alignment even when trained solely on safety data, demonstrating that LoRA serves as \textbf{cost-efficient}, \textbf{performance-preserving}, and \textbf{plug-and-play} safety patches. Beyond empirical findings, we provide both theoretical and experimental evidence that LoRA effectively decouples safety into a low-rank subspace largely orthogonal to the model’s intrinsic transformation space, ensuring that safety enhancements do not interfere with inherent capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates LoRA-based refusal training as a cost-efficient approach for safety alignment of large language models (LLMs). The authors argue that LoRA fine-tuning constructs a “safety subspace” that is largely orthogonal to the model’s intrinsic weight space, thereby enhancing safety (i.e., reducing jailbreak success rate) while degrading general capabilities less than full fine-tuning. The paper provides both a theoretical intuition based on SVD analysis of LoRA update matrices and empirical evaluations across several instruction-tuned models, including Qwen2.5-7B-IT, LLaMA3.1-8B-IT, and Mistral-7B-IT.

### Strengths
* The figures and tables are well-organized, making it easy to follow
* Experiments cover multiple instruction-tuned backbones (Qwen2.5-7B-IT, LLaMA3.1-8B-IT, Mistral-7B-IT)

### Weaknesses
* **Limited conceptual novelty.** Most of the paper’s conclusions are already well established in prior literature. Earlier works such as [1] have systematically shown that LoRA updates will forget less and better keep general ability. [2] has shown that safety subspace is usally low rank. This paper mainly repackages these known observations without introducing a new mechanism or deeper insight.

* **The “safety gap between lora & finetuning” result is weak .** The only somewhat new finding, that the difference between LoRA and full fine-tuning is smaller in the safety domain than in other domains (code, finance) is not  convincing enough. The training datasets are limited, and it is unclear whether this effect is general or simply dataset-specific.

* **Missing robustness analysis.** The paper lack investigation about how well the learned “safety subspace” persists under further finetuning.

* **theoretical analysis is a bit trival**. The theoretical part provides a simple SVD-based explanation of “orthogonality” between LoRA updates and the base model weights.  the analysis is largely descriptive and does not yield new insight into how a safety subspace is actually formed, what governs its geometry, or how it differs from other task-specific subspaces. 

[1] LoRA Learns Less and Forgets Less

[2] Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications

### Questions
In section 5.3, the paper compares the right singular vectors $V_0$ of the original weight matrix $W_0$ and $V_{\Delta}$ of the LoRA update $\Delta V$, , both truncated to $r$ dimensions. However,  I think $W_0$  is typically higher rank, truncating  $V_0$  to be the same as $V_{\Delta}$ seems not that reasonable?

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
The paper proposes using LoRA-based finetuning as a "safety patch" for LLMs. The central claims are (1) efficacy: LoRA-based fine-tuning can achieve strong safety alignment while preserving general performance; (2) efficiency: lora is more cost-efficient than full-parameter fine-tuning; (3) mechanism: the paper argues this works because LoRA decouples the safety update into a low-rank subspace that is "largely orthogonal" to the original model's "intrinsic transformation".

### Strengths
- Cost-efficiency and simplicity: The method is presented as highly cost-efficient because it successfully achieves safety alignment by training solely on safety-critical data and also utilize lora technique. 
- The paper demonstrates that LoRA-based SFT can substantially enhance model safety.

### Weaknesses
- The authors claim the theoretical insight as a part of the contributions. However, the concept of using orthogonal subspaces to prevent catastrophic forgetting in adapter-based tuning is not new. The paper itself cites O-LoRA, a method explicitly designed for "Orthogonal Subspace Learning" in continual learning. The core idea of this paper that a new task can be learned in a subspace orthogonal to the original model's to prevent interference is exactly the central thesis of O-LoRA. The paper fails to adequately differentiate its core mechanism from this prior work. 
- One experiment is missing from the paper, that is LoRA with safety-general data. It is possible that LoRA-based SFT on the safety-general would perform even better, which would invalidate the paper's claim that mixing general data compromises safety. 
- Regarding the life-long learning experiment, the authors seem to choose a rather weak baseline as DPO + full parameters seems to lead to low performance in a variety of tasks. 
- There are also some related works that the authors could cite, for example [1,2]

[1] Refusal in Language Models Is Mediated by a Single Direction
[2] LoX: Low-RankExtrapolation Robustifies LLMSafetyAgainst Fine-tuning

### Questions
Please refer to the the weakness section.

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
3

### Summary
The paper conducts extensive experiments to investigate and show that Low Rank Adaptation (LoRA) only using a refusal dataset is a better and more reliable approach for weight-space safety alignment/tuning compared to full finetuning (SFT) with a mixed dataset of refusal and general-purpose data. Experiments are mostly focused on supporting this claim by showing that LoRA-based safety-tuning induces parameter updates that influences safety-related input more than general-purpose tasks (hence, less degradation of general utility).

### Strengths
1. **Writing:** Paper is well-organized with plenty details while maintaining coherent logical flow.
2. **Motivation:** The problem being addressed is well-motivated and Section 2 sets up the stage by quantifying how uncareful safety-tuning degrades general performance.
3. **Extensive empirical analysis:** Convincing empirical analysis is provided that LoRA is more reliable for the task by comparing and visualizing $\Delta W$ and $W_0$ using various matrix alignment metrics.

### Weaknesses
1. Although a plenty of prior works from safety alignment literature are brought up and discussed, some works that are directly related to LoRA-based safety alignment seem to be overlooked. Particularly, [1] also shows that LoRA on refusal dataset can bypass the "safety tax" (negative impact on general utility) in the case of reasoning abilities. Also, I think [2] takes it further by implementing an extra projection for LoRA updates to ensure it lies in the subspace present in the aligned model parameters.
2. The theoretical explanation also remains quite generic. To be more precise, the approximate orthogonality of $\Delta V$ and $V_0$ is a necessary condition for the observed experimental results. It would be more meaningful contribution if, for example, Figure 8(b) could be explained through this argument. *"Why does safety tuning induce more orthogonal parameter updates compared to finance and code?"*, which I would guess it is because the pre-training dataset had finance and code-related data already.

### Questions
Stemming from the weaknesses section:

1. Could authors compare and pinpoint portions of their contributions complementary/orthogonal to [1] and [2]? I admit that [1] is only recently publicly available (first arXiv version on July 22), while [2] is old enough and peer-reviewed.
2. Could authors comment on the pretrained models and validate/invalidate my guess in Weakness 2? Easiest way to test it would be to reproduce Figure 8(b) with a safety-aligned model as a baseline, and compare safety, finance, and code cosine similarities.

___

Overall, I find the investigations in this paper thorough and useful in most aspects. However, lack of comparison with directly related works makes it hard for me to place the paper in the map of current safety research state. I am open to change my score if both of my questions are addressed in a satisfactory manner.

___

### References

1. Yihao Xue, Baharan Mirzasoleiman. LoRA is All You Need for Safety Alignment of Reasoning LLMs. arXiv:2507.17075
2. Chia-Yi Hsu, Yu-Lin Tsai, Chih-Hsun Lin, Pin-Yu Chen, Chia-Mu Yu, Chun-Ying Huang. Safe LoRA: The Silver Lining of Reducing Safety Risks when Finetuning Large Language Models. NeurIPS 2024

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the use of LoRA-based Refusal-training for safety alignment in Large Language Models (LLMs), proposing it as a cost-efficient, performance-preserving, and plug-and-play alternative to traditional full-parameter fine-tuning methods. A key finding is that LoRA-based alignment, even when trained solely on safety-critical data, significantly enhances safety (near-zero Attack Success Rate or ASR) while incurring minimal degradation to general capabilities (performance-preserving).
The authors provide a theoretical explanation based on transformation subspace orthogonality. They propose that the LoRA-induced safety update (ΔW) lies in a low-rank subspace that is largely orthogonal to the original model's intrinsic transformation space (W_0 ). This orthogonality, quantified using Sim(V_Δ,V_0)=V_Δ^⊤ V_0 ≈0 , minimizes the interference (or "catastrophic forgetting") between the safety adjustments and the model's inherent knowledge and abilities.

Empirical analyses support this claim through comparisons of:
1.Parameter Update Magnitude: Counter-intuitively, LoRA produces larger parameter updates than full-parameter training in most layers, yet it better preserves general performance.
2. Layer-wise Hidden State Shifts: LoRA-based Refusal-SFT induces smaller hidden state shifts on benign inputs but larger shifts on jailbreak attacks compared to full-parameter methods.
3. Orthogonality: LoRA-based alignment yields the lowest similarity (highest orthogonality) between the safety subspace and the model's intrinsic transformation space.
Furthermore, the paper demonstrates LoRA’s utility for lifelong safety alignment in multi-round red-teaming and shows that the safety subspace is more orthogonal and less intrusive than subspaces induced by domain-specific fine-tuning (e.g., code and finance).

### Strengths
*Originality & Significance: The paper presents a novel and highly effective method for safety alignment using LoRA-based Refusal-SFT and introduces subspace orthogonality as a powerful theoretical lens to explain the mechanism. This theoretical perspective is a significant conceptual contribution to understanding parameter-efficient fine-tuning (PEFT) in the context of safety and catastrophic forgetting.
*Quality & Clarity: The work is supported by rigorous and extensive empirical validation across multiple LLM architectures (Qwen, LLAMA, Mistral) and alignment paradigms (SFT, DPO). The results consistently and strongly support the core hypothesis, particularly the superiority of LoRA's safety-utility trade-off shown in Figure 1(a). The writing is excellent, making the complex technical concepts (SVD, orthogonality) accessible.
*Practical Value: The demonstration that LoRA-based Refusal-SFT performs best using only safety data is a crucial finding for practical deployment, as it eliminates the costly and difficult task of searching for the optimal proportion of safety-critical and general-purpose data. The plug-and-play feature for lifelong alignment is highly relevant for continuous model maintenance.

### Weaknesses
*Orthogonality Measurement Robustness: The current measure of orthogonality, Sim(V_Δ,V_0)=V_Δ^⊤ V_0 , relies on the ΔW from SVD, which itself is an approximation for LoRA weights (which are already low-rank AB matrices). While the results in Appendix H show a clear link between LoRA rank and the number of non-negligible singular values, a clearer theoretical or empirical connection between the inherent LoRA structure (ΔW=AB) and the guarantee of orthogonality to W_0 would strengthen the claim. The current finding is empirical; a theoretical bound or analysis on the orthogonality of span(V_Δ) for LoRA updates is missing.
*Generalizability of Safety Domain: The cross-domain analysis in Section 5.4 only compares safety with code and finance domains. To fully support the claim that the safety subspace is uniquely orthogonal and less intrusive, it would be beneficial to compare against other, potentially less "entangled" domains such as factual knowledge updates or style transfer.
*LoRA Rank Selection and Scaling: Although Appendix F and G discuss the effect of LoRA rank, the best rank for larger models (rank 16 for Qwen2.5-14B-IT) suggests the optimal rank is model- and size-dependent. A systematic study or guideline for selecting an appropriate LoRA rank, perhaps based on the initial W_0  properties (e.g., singular value decay of W_0 ), would improve the methodology's completeness and practical application.

### Questions
1. Theoretical Link to LoRA Structure: Can the authors provide a more formal theoretical justification, beyond empirical SVD, that the specific structure of LoRA updates (ΔW=AB) inherently biases the resulting subspace span(V_Δ) towards orthogonality with the initial weight matrix subspace span(V_0)? This would significantly elevate the theoretical contribution from an observation to a mechanism.

2. Orthogonality vs. Interference in Practice: The derivation in Appendix I (Non-Orthogonal Case) shows that the interference terms are 
W_0x_Δ​+ΔWx_0 . The paper claims that LoRA-based SFT produces larger weight updates (∣ΔW∣) but smaller hidden state shifts (Δh^(l) ) on benign inputs. Since the hidden state shift is directly related to the interference, can the authors specifically quantify and compare the magnitude of the interference terms (∣∣W_0x_Δ+ΔWx_0∣∣) between LoRA-based and full-parameter models on benign inputs? This would directly connect the theoretical explanation to the key empirical findings.

3. Impact of Initial Model Alignment: The paper focuses on aligning instruction-tuned models (e.g., LLaMA3.1-8B-IT). How does the initial degree of safety alignment of the base model (e.g., a highly safe Llama-Guard-like model vs. a non-aligned base LLM) affect the final orthogonality of the LoRA safety patch? Does an already safe base model force the LoRA update to be more orthogonal, or does it make the effect negligible?

4. Beyond Refusal-SFT: The primary success is observed with LoRA-based Refusal-SFT. Given that DPO also exhibits better parameter and hidden state stability than SFT (Sections 5.1, 5.2) , why does LoRA-based DPO not achieve a similarly dominant trade-off as LoRA-based Refusal-SFT (e.g., lower safety gains for LoRA-based DPO in Table 1)? A deeper analysis into the mechanism difference between SFT and DPO loss functions in the orthogonal subspace framework would be insightful.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper discusses how to implement safety finetuning for LLMs using LoRA weight updates. The authors argue that full-parameter safety finetuning on safety datasets alone still causes unnecessary changes in weight directions that are indicative of performances; fine-tuning on both safety and general data compromises safety guarantees. Thus, they hypothesize that performance and safety can be decomposed into (almost) orthogonal directions and using LoRA allows for safety finetuning to focus on the 'safety' directions only.

### Strengths
The paper addresses an important and timely topic. Already in this conference, there is another submission that also attempts to use LoRA for safety finetuning. 

The experiments performed in the paper is comprehensive and well-demonstrates the points being made, which is that LoRA directions are generally almost orthogonal to base weight directions. 

The proposed methodology is simple enough to be tested in practice with ultra-large LLMs, making the research valuable not just to academia but also to the industry.

### Weaknesses
The theoretical justifications of the paper is rather lacking in several ways:
- No orthogonal decomposition guarantees. The authors assumed that performance and safety can be orthogonalized (at least approximately) without giving arguments, even heuristical arguments, on why that should be the case. While experimental data shows that LoRA update has small (matrix) inner product with the base weights, that fact can be induced by implicit or explicit regularizations (since LoRA is constraint to have small rank, it must packs all gradient update information in a few dimensions; and this is done most efficiently with dimensions that are orthogonal to the base weights). In the above scenario, the orthogonality behavior may be at odd with 'true' safety dimensions (which was shown to exist, for instances, in Wei et al., "Assessing the brittleness of safety alignment via pruning and low-rank modifications"), and by forcing the update to be orthogonal via LoRA, safety guarantees are weakened.
- Orthogonality of weights does not imply independence of performance. While it is a point usually made in the literature that orthogonal updates to the weights preserve original performance, it is still a heuristical argument, since the final model is nonlinear in the transformer weights. It is fine to use this idea as a motivation and test empirically again (which the authors did a decent job empirically), the final results are inherently experimental in nature and the theoretical contribution would be too weak to be counted as a major contribution of the paper. This is also relevant to the next point, since the authors distinguish themselves from literature mostly by this 'orthogonalization' concept. 
- Missing direct comparison to Wei et al  "Assessing the brittleness of safety alignment via pruning and low-rank modifications", which identifies 'safety directions' and 'safety neurons' quantitatively. Why, or under which condition, is the methodology in the current paper more correct than that proposed in Wei et al (pruning least safety-relevant neurons or removing safe-relevant directions (via LoRA) to improve safety)? While the reviewer think that there are enough differences between the two papers, a direct comparison should be done to previous LoRA methods for LLM safety. 
- Minor: the authors have pointed out that different finetuning methods (DPO vs SFT-safety, etc.) induce different norms/magnitude in weight changes, yet the (dis)similarity score (matrix inner product of base and update weights) are left unnormalized. This may result in the small values observed simply coming from the matrix weights values being small, rather than orthogonality. Some normalization should be done in this comparison. 

The weaknesses pointed out above are mainly conceptual and theoretical. Since the paper's impact and focus is purely experimental, I still recommend weak acceptance but would consider increasing my score if my concerns are addressed.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
