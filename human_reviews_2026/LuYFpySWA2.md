# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation to balance performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where underloaded experts complete computations early but must wait for overloaded experts, leading to global delays. We define this phenomenon as the \textbf{\textit{Straggler Effect}}, as the most burdened experts dictate the overall inference latency. To address this, we first propose \textit{\textbf{Capacity-Aware Token Drop}}, which enforces expert capacity limits by discarding excess tokens from overloaded experts, effectively reducing load imbalance with minimal performance impact (e.g., $30\%$ speedup with only $0.9\%$ degradation on OLMoE).  
Next, given the presence of low-load experts remaining well below the capacity threshold, we introduce \textit{\textbf{Capacity-Aware Expanded Drop}}, which allows tokens to include additional local experts in their candidate set before enforcing strict local capacity constraints, thereby improving load balance and enhancing the utilization of underused experts. 
Extensive experiments on both language and multimodal MoE models demonstrate the effectiveness of our approach, yielding substantial gains in expert utilization, model performance, and inference efficiency, e.g., applying Expanded Drop to Mixtral-8$\times$7B-Instruct yields a {0.2\%} average performance improvement and a {1.85$\times$} inference speedup. The code is released at: https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The core problem addressed in this paper is the Straggler Effect in the inference phase of MoE systems: due to the imbalanced allocation between tokens and experts, overloaded experts become the bottleneck for inference latency, causing underutilized experts to remain idle while waiting, thereby reducing overall efficiency.
The key solution proposed is a Capacity-Aware Inference framework, consisting of Capacity-Aware Token Dropping and Capacity-Aware Expanded Dropping. The core concept involves discarding low-importance tokens for overloaded experts while increasing token allocation for underutilized experts. It achieves an average 1.85x inference speedup when applied to Mixtral-8×7B-Instruct.

### Strengths
1. The significance of the addressed problem lies in being the first to explicitly identify the "Straggler Effect" during MoE inference, and through quantitative analysis (e.g., revealing how some experts bear 7× the expected load N), it demonstrates its critical impact on inference latency.
2. The methodology is designed to be efficient: both strategies involve lightweight modifications during the inference phase, requiring neither adjustments to the model training process nor additional computational resources (such as the expert redundancy deployment in DeepSeek-V3). Token Dropping rapidly alleviates overload by discarding low-importance tokens, while Expanded Drop utilizes local expert expansion of candidate sets to avoid cross-device communication overhead.

### Weaknesses
1. The long-term impact of token dropping remains unvalidated: existing experiments only evaluate performance changes on static tasks (e.g., MMLU, OBQA), without investigating its effects on long-context generation (such as dialogue and text creation). For instance, whether persistent dropping of low-importance tokens may lead to degraded contextual coherence and generation quality remains unexamined, lacking verification in dynamic scenarios.
2. The method's generalizability has limitations: experiments show the approach achieves optimal results with models deploying "single expert per GPU" (e.g., Mixtral-8×7B-Instruct), but exhibits diminished acceleration effects for models with "multiple experts per GPU" (e.g., OLMoE-Instruct with 8 experts/GPU). The aggregated workload across multiple experts reduces the benefits of constraining straggler experts. The authors suggest "increasing GPU count to reduce experts per GPU" without proposing adapted solutions for multi-expert single-GPU scenarios, thus limiting universal applicability.
3. The study lacks comparison with existing inference optimization methods: it fails to benchmark the proposed approach against other MoE inference optimization solutions (such as PowerInfer's expert offloading or MoE-Infinity's activation-aware scheduling).

### Questions
1. Regarding the multiple experts per GPU scenario, the authors mention that aggregated workload diminishes the method's effectiveness but provide no specific improvement strategies. Have you considered dynamically adjusting the capacity limits for experts within a single GPU?
2. In the token dropping strategy, the gating score G(x) is used as the importance metric, but its reliability during inference has not been sufficiently validated. What about other important metrics (such as token's contextual contribution or semantic relevance), and how robust is the gating score across different tasks (e.g., logical reasoning, sentiment analysis)?
3. The experimental results show model inference speed improvements rely on reduced expert computation time, but fail to analyze changes in communication overhead. Does Expanded Drop's expansion of candidate expert sets increase token scheduling overhead within local devices?
4. Are there no related works on MoE inference acceleration? For example, https://github.com/LINs-lab/DynMoE

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
5

### Summary
This paper identifies the **Straggler Effect** in Mixture-of-Experts (MoE) models during inference: overloaded experts cause global latency bottlenecks due to imbalanced token-to-expert assignments. To mitigate this, the authors propose **Capacity-Aware Inference**, introducing two lightweight strategies:

1. **Token Drop**: enforces capacity limits on overloaded experts by discarding low-importance tokens, reducing latency with minimal performance loss.
2. **Expanded Drop**: allows tokens to select additional local experts, improving utilization of underloaded experts and enhancing performance.

Experiments on both language and multimodal MoE models show significant improvements in inference efficiency (up to **1.85× speedup**) with comparable or even slightly better accuracy. The methods are training-free, easy to implement, and broadly applicable.

### Strengths
1. This paper is well written and organized, with clearly stated motivation and method, together with relatively comprehensive experiments.
2. This paper investigated both LLM and MLLM and demonstrated efficiency gains, demonstrating its effectiveness and universality.

### Weaknesses
1. The novelty of this paper is insufficient, since there has been comprehensive research on load balancing in MoE model training, such as Switch Transformer (Google) and AuxFree load balancing (DeepSeek). This paper does not have fundamental distinctions from these well-known papers, where the only difference is taming the load balancing issue during the inference stage in a training-free manner.
2. In Lines 322-323, you should clarify that DP is applied to attention modules while EP is applied to MoE modules. 
3. You should specify the prefilling stage or decoding stage in your efficiency-related experiments with the listed workload, such as batch size and context length, which include Figures 4, 5, and 6.
4. Also, you should provide more comprehensive experimental results on efficiency benefits under different workloads. For example, you should scale the context length with a fixed batch size, and scale the batch size with fixed context length, to compare and visualize the speedup or latency, and therefore provide more insights with a scaled workload.

### Questions
1. What is the specific model you used in Figures 1 and 2? You should provide the detailed information in the caption texts.

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
This paper identifies that Mixture-of-Experts (MoE) models exhibit highly imbalanced token-to-expert assignments during inference, leading to inefficient inference computation. To address this, the authors propose Capacity-Aware Token Drop, where overloaded experts drop excess tokens beyond the compputation budget. To further compensate for potential performance loss and utilize idle capacity on underloaded experts, they introduce Capacity-Aware Expanded Drop, allowing tokens to leverage additional underutilized experts on the same device. Experiments across multiple models demonstrate that the proposed methods achieve notable inference speedups with minimal or no performance degradation.

### Strengths
* The experimental section is rich and goes beyond reporting results—it includes several insightful analyses that provide independent value.

* The proposed approach is simple  and easy to implement.

### Weaknesses
* When the number of experts per device exceeds one, the observed speedup becomes limited.

* The writing could be clearer in several places (see “Questions” below).

### Questions
## Writing clarify

* Figure 1: The definition of "normalized load" should be explained so that the figure is self-contained; it’s not clear until Section 3.

* Which dataset and model are used in Figure 1?

* Line 204 mentions that the capacity constraint is applied per device rather than per expert. However, later discussion (e.g., Line 217: "each overflowed expert selectively discards those with lower scores") seems to describe per-expert constraints. This discrepancy makes the algorithm unclear.

* Token Drop: If I understand correctly, a token could potentially end up with no assigned expert after dropping, if its scores assigned to each expert are ranked below the threshold. How does the algorithm ensure each token still has a valid output?

* Expanded Drop: It seems that each token additionally selects local experts (beyond the normal top-k experts), and these experts then drop tokens following the same rule as in "Token Drop Regulates the Latency of High-Load Experts" section. In that case, could an expert retain some tokens assigned as local experts while dropping others from the top-k assignment? Please clarify.

* Does Table 1 include the Expanded Drop technique?

* In Figure 5, how many experts per GPU are used for models other than Mixtral?

* In Figure 6, do the x-axis labels "1.0," "1.5," and "2.0" correspond to the \gamma values?

* In Figure 9, which model and dataset are used?

* Line 448 states that "we also experiment with a strategy that prioritizes dropping image tokens before selectively removing text tokens (“Image First”)." Could the authors explain the implementation details of this strategy?

## Experiments

* Eq. 4 resents the latency of an MoE layer. The intuition is sound, but latency in practice depends on many factors. It would strengthen the paper to:
    * (1) Measure actual layer latency and compare it with this theoretical bound; and 
    * (2) Measure end-to-end model latency to verify whether max({N_i}i=1^n) remains the dominant factor.

* Line 340: The claim that "As illustrated in Figure 5, for Mixtral-8×7BInstruct, deploying a single expert per GPU maximizes the effectiveness of capacity-aware inference." requires experiments with multiple experts per GPU for the Mixtral model. Currently, Figure 5 only reports results for a single expert per GPU for the Mixtral model.

* Line 352 states that "Notably, the duration of permutation and communication increases when tokens are expanded across a range of global experts." Could the authors clarify what it means and how to see it from Figure 6.

* Line 373: The paper claims "regulating the maximum capacity has a limited impact on the overall number of accommodated tokens," yet Figure 7 shows that varying \gamma dramatically changes the percentage of dropped tokens from nearly 100% to 0%. Please clarify.

* Figure 3: Why does “Expanded Drop + Image First” significantly outperform the baseline? Additional explanation would be helpful.

## Other minior issues

* The citation format is incorrect in many places. For example, \citep should be used in:
    * "In recent years, the rapid evolution of Large Language Models (LLMs) OpenAI (2024); Team (2024a); DeepSeek-AI et al. (2024b) ..."
    * "Specifically, MoE Shazeer et al. (2017b); Fedus et al. (2022) enhances scalability ..."
    * "We conduct experiments on OLMoE Muennighoff et al. (2024), Qwen1.5-MoE Team (2024b), DeepSeek-V2-Lite DeepSeek-AI et al. (2024a) and Mixtral Jiang et al. (2024)"
    * "The number of shots for each task is detailed in Table 4, which includes multiple tasks: ARC-C Clark et al. (2018), BoolQ Clark et al. (2019), HellaSwag Zellers et al. (2019), MMLU Hendrycks et al. (2021), OBQA Mihaylov et al. (2018), PIQA Bisk et al. (2019), RTE Wang et al. (2019), WinoGrande ai2 (2019) and GSM8K Cobbe et al. (2021). The evaluation code is based on EleutherAI’s LM Harness framework Gao et al. (2023)."

* Section 1 claims the identification of "the Straggler Effect caused by token imbalance at inference time in Mixture of Experts" as a contribution. However, as discussed in the paper, this phenomenon is already mentioned in prior work (e.g., DeepSeek-V3). It should be framed as a motivation rather than a contribution.

* Line 706: Add a comma before "and"

* Line 713: Add a comma before "and"

### Soundness
2

### Presentation
2

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
The authors propose a method for reducing the inference time of mixture-of-experts (MoE) models by load balancing during inference. In MoE models, tokens are typically routed to top-k most relevant experts and uneven loads across experts significantly slows down inference since sync operations are performed after the slowest expert has completed (termed ‘straggler effect’). The work proposes dropping tokens associated with overloaded experts and potentially re-routing them to underloaded ones to reduce the peak load on any single expert. Experiments on several MoE models and benchmark datasets suggest that the proposed method results in huge speedups with minor drop in performance.

### Strengths
The paper is mostly well written and easy to understand. The authors identify a specific issue with MoE models and propose a solution to address it. The idea is simple, interesting and well motivated. The experiments are thorough, with results on multiple models, benchmark datasets and ablations and analysis to understand the different components of the proposed method. The authors show that existing models suffer from the ‘straggler effect’ and that the proposed solution can alleviate it, achieving 7-18% speedup with up to 0.4% points drop in average performance. The solution is also extended to multi-modal models, where huge gains can be achieved by dropping the redundant tokens in the image domain.

### Weaknesses
The work does not have any major weaknesses. 

1. There is not much discussion/comparison with other approaches to address the ‘straggler effect’. For instance, the authors mention the use of additional GPUs for overloaded experts. For commercial use-cases where there are thousands/millions of concurrent requests to a service, would just having extra servers for overloaded experts be efficient or would the proposed solution still be necessary given that it comes with a performance drop? The proposed method is also shown to provide the biggest improvements when the number of experts per GPU is reduced by increasing the number of GPUs. 
2. Discussion on the effect of different sequence lengths within a batch and dataset is missing. Since sequences in a batch can be of different lengths and the proposed method prunes tokens at the batch level, how does the variance in length affect efficiency and performance? Also, how would a high variance in sequence lengths within a dataset affect the speed and performance?
3. The expanded token drop method (in the Methodology section) is a bit unclear. How exactly are the additional ‘m’ experts chosen and how are they ensured to be ‘local’ experts? 
4. Since some experts specialize towards particular domains, the proposed method might have a bigger drop in performance on specific tasks/datasets. For instance, in table 2, the maximum drop in performance over the 8 datasets on DeepSeek-V2-Lite-Chat model is 2.4% points while the average drop is 0.4% points (although this seems to be a one-off scenario in the table). How can this be predicted / handled?

### Questions
Please refer to the weakness section.

### Soundness
4

### Presentation
3

### Contribution
3
