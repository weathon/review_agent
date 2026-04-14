# ROLoRA: Rank Optimization  for Low-Rank Adaptation under Memory Constraints

- Decision: Reject
- Scores: 5, 6, 5, 5

## Abstract
Low-Rank Adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) with limited computational resources. However, by injecting low-rank adapters with a rank identical across all layers, standard LoRA overlooks the varying importance of the weight matrices, often leading to suboptimal performance. Therefore, discovering an optimal rank configuration that efficiently utilizes limited training resources remains an open question. Existing solutions typically compromises computational constraints for performance gains, limiting their practical usage in resource-constrained scenarios. To address these issues, in this paper, we propose a novel method named ROLoRA to efficiently discover an effective rank configuration for low-rank adaptation, while strictly adhering to a constrained computational budget during training. In particular, our method iteratively prunes saturated adapters and expands under-fitted ones to increase their capacity until they converge to a highly optimized configuration. Our approach is delicately designed within the Frank-Wolfe algorithmic framework, which offers potential theoretical guarantees. Experimentally, we demonstrate that ROLoRA outperforms standard LoRA on common natural language processing tasks, including the GLUE and SQuAD benchmarks. Additionally, we provide a comprehensive analysis to explain why ROLoRA surpasses competing state-of-the-arts.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a method called ROLoRA. ROLoRA is a novel method that efficiently discovers an effective rank configuration for low-rank adaptation while strictly adhering to a constrained computational budget during training. It outperforms standard LoRA on natural language processing tasks and is practical for resource-constrained scenarios.

### Strengths
>The writing is good and the paper is easy to follow.

>The method can efficiently discover an effective rank configuration for low-rank adaptation. 


>ROLoRA outperforms standard LoRA on some benchmarks under some model configurations.

### Weaknesses
> The experiments are conducted on models like RoBERTa and DeBERTa. Have any experiments been conducted on modern large language models like Llama or OPT? Including these experiments will make the method much more valuable in modern settings.

> The experimental results sometimes seem to fall behind baselines and the improvements are not that significant. For e.g., LoRA⋆ in Table 1 and AdaLoRA⋆ in Table 2.


> Any comparisons in running time and convergence speed with baselines?

### Questions
See weanesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ROLoRA, a method for optimizing adapter ranks in Low-Rank Adaptation (LoRA) to enable efficient fine-tuning of large language models within memory constraints. Unlike standard LoRA, which applies a fixed rank across all layers, ROLoRA iteratively adjusts ranks by pruning overfitted adapters and expanding under fitted ones, all within a specified memory budget. Experimental results demonstrate that ROLoRA outperforms existing methods like AdaLoRA on benchmarks such as GLUE and SQuAD, particularly by optimizing ranks for crucial weight matrices in transformer layers, such as the value matrices in attention mechanisms.

### Strengths
Overall, this is a good work. The algorithm is well-motivated and reasonably well-explained. The method is backed by adequate empirical evidence, with results that support the effectiveness of ROLoRA over existing approaches. In particular, the ablation study is a valuable addition, demonstrating that the rank assignments are not only adaptive but also focus on key components, such as the value matrices in attention layers, which are shown to benefit from higher ranks.

### Weaknesses
Weaknesses

1. Insufficient References in Key Claims (Lines 139-141): Certain claims in the paper are presented without adequate referencing, reducing their credibility, for instance “can often lead to a more favorable optimization landscape”.
   
2. Lack of Justification for Assumption 1: The authors assume that the SPARSIFY operator maintains memory constraints and can remove redundancy without sacrificing model performance. This assumption is pivotal to the algorithm, yet it lacks theoretical backing. Providing additional rationale here would reinforce the assumption’s validity.

3. Ambiguity in Proof of Proposition 1: The proof for Proposition 1, which posits that ROLoRA iteratively improves model performance, is not entirely convincing. The algorithm currently appears heuristic, without formal assurance that each iteration yields a performance improvement similar to the EM algorithm. A clearer proof structure or additional evidence supporting iterative improvement would strengthen this point.

4. Limited Explanation of Frank-Wolfe Connection and Convergence: While the authors mention a connection to the Frank-Wolfe algorithm, it needs further explanation. It is unclear how the discrete-to-continuous transition (which seems more heuristic) impacts convergence guarantees. Further elaboration on how the theoretical aspects would still hold after this discrete to continuous transition would enhance clarity.

5. Unclear Baseline Iteration Details: The paper lacks a detailed comparison of iteration counts between ROLoRA and baseline methods like LoRA* and AdaLoRA*. It is therefore uncertain whether these baselines received an equivalent level of optimization. Including these details would facilitate a more accurate assessment of relative performance.

6. Average Rank in Table 4: Table 4 indicates that ROLoRA achieves a lower average rank than sparsification-only methods like AdaLoRA, despite ROLoRA’s additional expansion steps. The reasoning behind this outcome is unclear. A more detailed explanation of how the rank pruning and expansion operations jointly lead to this effect would clarify the results.

7. Absence of Average Rank on SQuAD Datasets: The paper presents average rank results for GLUE but omits similar data for the SQuAD datasets. Providing this information would complete the evaluation, illustrating ROLoRA’s impact on question-answering tasks.

8. Scalability Testing on Larger Models: The paper’s evaluation on smaller models leaves open questions regarding scalability. Testing on larger models, such as those with 1B or 7B parameters, would confirm if ROLoRA’s efficiency extends to more substantial architectures, making the findings more broadly applicable.

### Questions
Please see above.

I am curious to see how ROLoRA performs using the sparsification method from AutoLoRA (https://aclanthology.org/2024.naacl-long.282.pdf), which is built upon a similar motivation as AdaLoRA. It would be interesting to explore how this approach compares or complements the existing sparsification techniques used in this work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes ROLoRA - a new PEFT (Parameter-Efficient Fine-Tuning) method that adjusts adapter ranks for different modules and layers under a constrained memory budget. Unlike LoRA, which applies the same rank for all adapters, but similar to AdaLoRA, ROLoRA suggests that different ranks for different modules are a better approach (confirmed by experiments). Compared to AdaLoRA, ROLoRA stays within the constrained budget during training but may increase training time (though by how much is unclear from the paper). ROLoRA is an iterative method involving pruning and then expanding ranks. The approach is tested on RoBERTa-base and DeBERTa-v3-base. Experiments show improvements over LoRA and competitive performance to AdaLoRA on the GLUE and SQuAD benchmarks.

### Strengths
Strengths

-Clear motivation and introduction

-Relevant problem

-New iterative framework that operates within a constrained budget throughout LLM finetuning

-Interesting analysis and insights on the importance of value matrices in finetuning

### Weaknesses
In general, I think this is a relevant problem and interesting approach. However, the biggest issue is that I would like to understand how this approach is better than AdaLoRA. Specifically, what is the increase in finetuning time introduced by ROLoRA, and how much does AdaLoRA exceed the computational budget during fine-tuning? What if we set the computational budget to be max N in AdaLoRA and exactly N in ROLoRA? How do the two algorithms compare? I would like to understand when someone would prefer to use ROLoRA, as it is a more complex algorithm (with longer fine-tuning time). I would also like to see the method’s behavior on larger models (decoder-only) and more recent tasks. Also, there is no mention of releasing code for this framework, which raises concerns about usability.

Weaknesses in points:

-Tested only on two encoder-only models and only GLUE and SQUAD benchmarks. I think that more models, possibly a larger decoder-only model, and some more recent benchmarks would be beneficial.

-The insights about value matrices are interesting, but they seem more observational than a motivating factor for ROLoRA design. In the conclusions, the value insight is highlighted as a main takeaway, but it’s only mentioned briefly in the ablation study at the end.

-The paper mentions balancing iterations and training time, but it would be helpful to see a clear analysis showing how much training time increases.

-Figure 2 could be improved to present a better side-by-side comparison (currently, it’s difficult to read).

-Table 4 shows average ranks, but a summary of parameter counts for LoRA, AdaLoRA, and ROLoRA would clarify the overall memory savings.

-The final sections of the paper are not as comprehensive as the earlier sections. The value matrix insight is introduced very quickly, and further analysis would be useful.

### Questions
Questions (a few questions also in the Weaknesses section)

-How does training time increase per iteration in ROLoRA? Is it 3x normal training for 3 iterations, or is it faster?

-Given ROLoRA complexity, do you plan to release the code?

-L062-L064: Could you add citations here?

-I think this paper could benefit from a visualization of rank changes. This might offer interesting insights for future work. Is it possible to generate such a plot?

-About Frank-Wolfe framework: Could you please add more explanation of why the Frank-Wolfe framework was chosen? Can you clarify the term "delicately designed" (L023) in describing the Frank-Wolfe framework? Why “potential” in “potential theoretical guarantees”? Could you explain? (L024)

-How were the hyperparameters for the experiments chosen? This is not clear from the paper and may have an impact on the results.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes ROLoRA, an iterative algorithm for optimizing rank configurations in LoRA-based fine-tuning of large language models. The key insight is adaptively adjusting ranks across different weight matrices while strictly adhering to memory constraints. The method iteratively sparsifies saturated adapters and grows underfitted ones within a Frank-Wolfe optimization framework. The authors evaluate ROLoRA on GLUE and SQUAD benchmarks using RoBERTa-base and DeBERTa-v3-base models, demonstrating improved performance over standard LoRA and AdaLoRA baselines while maintaining lower memory usage. A notable finding is that value matrices in transformer architectures require higher adaptation capacity compared to key/query matrices.

### Strengths
- The problem formulation effectively addresses a practical limitation of LoRA - the need to determine optimal rank configurations under strict memory constraints. However, although the problem is clearly defined, I'm not sure if it is important. Given that LoRA already works well and is simple and effective, do we need to further constrain the budget to achieve trivial improvements?
- The iterative optimization approach is theoretically grounded in the Frank-Wolfe framework, making the method more principled than heuristic alternatives.
- The empirical analysis is comprehensive, with clear ablation studies that reveal insights about the varying importance of different weight matrices.

### Weaknesses
- If I remember correctly, I have run AdaLoRA and even AdaLoRA is very time-consuming. Thus my major concern is the computational efficiency of the method. Does RoLoRA also face issues with computational overheads? Will there be detailed analysis? Given that RoLoRA's improvements over LoRA are not particularly significant, it's difficult to assess the merits of this method without detailed computational analysis. 
- I know that many LoRA works use the same experimental settings as this one, and it is convenient to make comparisons. However, I think using these models and benchmarks in 2024 might be somewhat outdated, as their behaviors may change as model capabilities continue to improve.

### Questions
- Figure 1 can be further improved. For example, add legends to clarify which kinds of grids represent which kinds of matrices.

### Soundness
3

### Presentation
2

### Contribution
2
