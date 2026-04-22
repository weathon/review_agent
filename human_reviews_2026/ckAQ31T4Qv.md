# Sample, Don't Search: Rethinking Test-Time Alignment for Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Increasing test-time computation has emerged as a promising direction for improving language model performance, particularly in scenarios where model finetuning is impractical or impossible due to computational constraints or private model weights. However, existing test-time search methods using a reward model (RM) often degrade in quality as compute scales, due to the over-optimization of what are inherently imperfect reward proxies. We introduce QAlign, a new test-time alignment approach.  As we scale test-time compute, QAlign converges to sampling from the optimal aligned distribution for each prompt. 
  By adopting recent advances in Markov chain Monte Carlo for text generation, our method enables better-aligned outputs without modifying the underlying model or even requiring logit access. We demonstrate the effectiveness of QAlign on mathematical reasoning benchmarks (GSM8K and GSM-Symbolic) using a task-specific RM, showing consistent improvements over existing test-time compute methods like best-of-$n$ and majority voting. When applied with more realistic RMs trained on the Tulu 3 preference dataset, QAlign outperforms direct preference optimization (DPO), best-of-$n$, majority voting, and weighted majority voting on a diverse range of datasets (GSM8K, MATH500, IFEval, MMLU-Redux,  and TruthfulQA).
  A practical solution to aligning language models at test time using additional computation without degradation, our approach expands the limits of the capability that can be obtained from off-the-shelf language models without further training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces QALIGN, a test-time alignment method for language models that leverages Markov Chain Monte Carlo (MCMC) sampling to approximate the “optimal aligned distribution” without finetuning model parameters. The method adapts the QUEST algorithm to use a reward model (RM) for preference-based acceptance–rejection sampling, allowing aligned text generation with closed or open-weight models. Experiments on mathematical reasoning (GSM8K, GSM-Symbolic) and general instruction-following (MMLU-Redux, TruthfulQA, IFEval, MATH500) demonstrate consistent gains over best-of-n (BoN), majority voting (MV), weighted MV (WMV), and even Direct Preference Optimization (DPO) under equivalent compute budgets.

### Strengths
- The paper identifies an important limitation in current test-time search methods, over-optimization of imperfect RMs when scaling compute, and proposes a principled sampling-based alternative.

- Results are shown across both task-specific and general alignment settings, demonstrating robustness to RM imperfections and outperforming multiple baselines.

### Weaknesses
While the paper is clearly written and technically sound, several aspects could be clarified or strengthened to better support its claims:

- The method can be viewed as a length-normalized, RM-guided extension of BoN using MCMC proposals from QUEST. While the MH-based acceptance criterion is well-formulated, it is not entirely clear that this constitutes a fundamentally new paradigm rather than a reparameterization of BoN/MBR with additional tuning. In addition, since QALIGN generates samples sequentially, it forgoes the parallel efficiency of BoN, which raises questions about its overall computational practicality.

- The evaluation focuses on a single policy model (LLAMA-3.1-8B-Instruct) and one reward model (TÜLU3-8B-RM), leaving it uncertain whether the observed improvements would generalize across different architectures or RM designs.

- The main comparison in Table 1 uses N = 1024 samples for both BoN and QALIGN, which is considerably higher than typical inference budgets (e.g., N = 16–32). Results under more practical compute settings would strengthen the study. Furthermore, the FLOPs-adjusted comparison with DPO is not fully equitable, as training FLOPs are a one-time cost whereas inference FLOPs recur for each query.

### Questions
- A natural baseline, sampling multiple random indices and running BoN over them, is not tested, making it unclear whether MCMC brings tangible benefit.

- Can the authors offer more details regarding FLOPs computation for inference-time and training-time methods and how do they align?

- The related work section is relatively comprehensive but misses several works on inference-time alignment, such as:

1. Fast Best-of-N Decoding via Speculative Rejection.

2. Test-Time Preference Optimization: On-the-Fly Alignment via Iterative Textual Feedback.

3. TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling. 

4. Args: Alignment as reward-guided search.

5. Inference-time language model alignment via integrated value guidance.

### Soundness
2

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
4

### Summary
This paper introduces a test-time alignment method for LLMs that improves performance without additional training or access to model logits. It does so by leveraging Markov chain Monte Carlo sampling to better explore and sample from the optimal aligned output distribution as computation increases, avoiding over-optimization issues seen in prior reward-model-based methods. The result is more accurate and well-aligned responses across reasoning and preference benchmarks, outperforming existing test-time compute and alignment baselines.

### Strengths
1. The paper analyzes the language model sampling from a MCMC prospective, and the algorithm is novel.
2. The algorithm is surprisingly simple and does not require accessing the models' logits, making it suitable for any LLM (open-sourced or commercial).
3. The evaluation seems comprehensive.

### Weaknesses
1. There seems error in derivation. First, in equation (7), when computing the ratio, the right hand side's numerator does not seem to be consistent with the definition of the proposal distribution. Also, it is not very clear how equation (8) is obtained, especially how everything is reduced to the length ratio. 

2. Unclear experiment setting: It is unclear how finetuning based method such as DPO and SFT is scaled during inference time. Also, can the author provide the number n for best-of-n? The Inference FLOPs is helpful, but number n is also commonly used. 

3. Lack of experimenting closed-source model. One claimed advantage of the paper is that it does not require accessing the logic. It would be very helpful if some results on closed-source model can be provided. 

4. This method requires selecting an index and complete the full response, for multiple times, which can be very expensive. Recently there are token-level reward model [1] that have shown to be efficient at guide a frozen LLM to generate aligned outputs. It would be helpful to also compare with such inference time alignment approach, which also seems to give the optimal distribution under the RL problem. 

[1] GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-Time Alignment, ICLR 2025.

### Questions
1. Is there a reason to choose the specific proposal distribution by QUEST? Is this choice optimal in any sense?

### Soundness
2

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
2

### Summary
The paper *“SAMPLE, DON’T SEARCH: Rethinking Test-Time Alignment for Language Models”* proposes **QAlign**, a new test-time alignment method that leverages Markov chain Monte Carlo methods for text generation, improving efficiency–alignment trade-offs at high compute budgets, while maintaining the flexibility of inference-time alignment without modifying model parameters.

### Strengths
- **Strong empirical performance in high-compute regimes:** QAlign consistently outperforms other alignment methods when sufficient inference-time compute is available.   
- **Solid experimental validation:** The experiments include diverse prompts and datasets, providing a fair comparison across multiple baselines and compute settings.

### Weaknesses
- **Compute intensity:** QAlign requires substantial inference-time compute to surpass baselines, limiting its practicality in typical deployment settings.  
- **Limited theoretical justification:** Although the paper claims convergence to the “optimal aligned distribution,” this claim is not clearly seen to be proven. The paper would benefit from a clearer mathematical explanation or proof sketch illustrating how sampling asymptotically approximates the optimal aligned distribution.  
- **Narrow advantage region:** In most compute-constrained regimes, traditional search-based or weighted-logit methods still outperform QAlign.

### Questions
1. QAlign appears to require high compute budgets to outperform existing baselines. In practice, are such inference-time budgets realistic or allowed?  
2. Under what specific conditions (model size, compute budget, or sampling strategy) would you recommend using QAlign over search-based alternatives?  
3. Can you provide a formal or empirical justification for the claimed convergence of QAlign to the optimal aligned distribution for each prompt?

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
3

### Summary
This paper introduces QALIGN, a novel test-time alignment method designed to address the over-optimization problem where existing methods, like Best-of-n (BoN), degrade as compute scales due to imperfect reward models (RMs). Instead of maximizing the RM, QALIGN uses a Markov chain Monte Carlo (MCMC) sampling approach, adapted from QUEST, to draw samples from the optimal aligned posterior distribution for a given prompt. This method notably requires no model finetuning or logit access, only the ability to sample from the base LM and query an RM. The final answer is selected from the resulting samples using Minimum Bayes Risk (MBR), which amounts to majority voting for tasks with discrete answers. The authors empirically demonstrate that QALIGN consistently improves with increased computation, outperforming BoN, majority voting (MV), and weighted majority voting (WMV) , and even surpasses the performance of the finetuned DPO model on a diverse suite of benchmarks when given a comparable inference budget.

### Strengths
The paper's primary strength lies in its proposal of QALIGN, a novel and practical test-time alignment method that directly addresses the critical over-optimization problem where methods like Best-of-n (BoN) see performance degrade with increased compute. The method is well-motivated, and its technical approach—using MCMC sampling to approximate the optimal aligned posterior distribution—is elegant. The key advantage, which is well-supported by experiments, is that QALIGN's performance consistently improves as the compute budget scales, allowing it to avoid the performance degradation that plagues other methods. The empirical evaluation is strong, demonstrating that QALIGN not only outperforms other test-time methods (BoN, MV, WMV) but also surpasses the performance of a fully finetuned DPO model across a diverse suite of benchmarks when given a comparable inference budget.

### Weaknesses
**Scope of Empirical Evaluation is Limited to "Matched" Model Pairs**: The paper does not test the robustness of QALIGN when the base Language Model (LM) and the Reward Model (RM) are "mismatched." In the Task-Specific Tuning experiments (Sec 4.1), the base LM is LLAMA-3.1-8B-INSTRUCT, and the RM used is a custom model finetuned from that same LLAMA-3.1-8B-INSTRUCT model. In the General Alignment experiments (Sec 4.2), the base LM is TÜLU3-8B-SFT, which is paired with the TÜLU3-8B-RM. These models are from the same family and were explicitly chosen for their close relationship to allow for a fair comparison with the TÜLU3-8B-DPO model.

### Questions
1. **Robustness to Mismatch**: Could the authors please comment on the robustness of QALIGN to "mismatched" model pairs? For instance, how would the method perform if the TÜLU3-8B-RM were used to align a different base model, such as a Llama 3.1 or Qwen 2.5 model?

2. **RM Dependence**: How dependent is QALIGN's success on the RM being trained (or finetuned) on the specific output distribution of the base LM? Is it possible that the MCMC sampling process becomes inefficient or unstable if the RM's preferred distribution is too far from the base LM's proposals?

### Soundness
2

### Presentation
3

### Contribution
3
