# When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 2, 4

## Abstract
Layer pruning has emerged as a widely adopted technique for improving the efficiency of large language models (LLMs). Although existing methods demonstrate strong performance retention on general knowledge tasks, their effect on long-chain reasoning, a more brittle yet crucial capability, remains largely unexplored.  In this work, we study the impact of layer pruning on long-chain reasoning through the lens of test-time scaling, a key mechanism in modern LLMs that enables strong reasoning capacity by allocating more computation at inference time. With extensive experiments, we demonstrate that pruning even one or two layers can severely impair test-time scaling, with performance collapsing drastically on long reasoning benchmarks even when performance on knowledge-intensive and shallow reasoning tasks remains stable. Furthermore, we find that standard supervised fine-tuning remedies fail to recover lost test-time scaling once it has deteriorated. Through in-depth analyses, we identify the mechanisms underlying this fragility of test-time scaling and highlight the fundamental risks of applying layer pruning to reasoning-intensive LLMs. These findings call for a rethinking of layer pruning strategies and provide insights for developing methods that preserve the robustness of reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper experimentally demonstrates that Layer Pruning, a mainstream technique for improving model efficiency, causes LLMs' performance to collapse on complex reasoning tasks reliant on long sequential Chains-of-Thought. Crucially, standard supervised fine-tuning (such as LoRA or full-parameter fine-tuning) cannot effectively recover this lost test-time scaling capability. Through mechanistic analysis, the authors attribute the performance degradation to structural damage in the model, resulting in an increase in redundant loops, a decrease in reasoning trajectory diversity, and a weakening of self-reflection capability within the reasoning paths. The work calls for future efforts to explore hybrid strategies that balance efficiency and robustness, ensuring that pruning preserves both performance and reasoning depth.

### Strengths
1. The paper demonstrates a degree of originality by proposing a novel and critical problem formulation—investigating the impact of pruning on "Test-Time Scaling" capability. This is a question that integrates efficiency research with reasoning research, establishing a necessary new evaluation criterion for model compression.

2. The authors provide both quantitative and qualitative analyses to explain the cause of the performance collapse, with ample experimentation and clear, compelling evidence to support their findings.


3. The paper proves that standard recovery techniques, such as LoRA and full-parameter fine-tuning, are largely ineffective in recovering the lost test-time scaling capability caused by pruning. These results are significant for both academic research and practical engineering.

### Weaknesses
1. The paper only validates training-free pruning techniques and does not incorporate mainstream training-based pruning methods. This makes it impossible to verify whether such methods can alleviate the fragility of test-time scaling. While the paper makes certain contributions, the generalizability of its conclusions is limited.

2. The paper explicitly states that it studies both sequential and parallel test-time scaling. However, most of the content and core analysis focus on sequential scaling. Regarding the performance collapse mechanism of parallel scaling methods, the paper's analysis is relatively weak and fails to provide in-depth mechanistic insights like those for sequential scaling.

3. The s1K-1.1 dataset used for fine-tuning is not introduced, and only this single dataset is employed in the fine-tuning experiments. Consequently, the conclusion that 'fine-tuning has limited effect' is questionable.

### Questions
1. As mentioned in the Weaknesses, if more complex methods such as training-based pruning methods are used, can the damage to test-time scaling ability be effectively alleviated?

2. Since the effect of standard fine-tuning is limited, does there exist or is it considered to design a customized fine-tuning scheme targeting reasoning trajectory loss? Can it repair the structural damage caused by pruning from a mechanistic perspective?

3. It is suggested that the authors supplement some comparison charts of the model evaluation indicators before and after fine-tuning in Section 4, so as to intuitively demonstrate the limitations of supervised fine-tuning methods in restoring the reasoning ability of models after layer pruning.

4. Is there a mistake in the introduction of ShortGPT in Appendix B? As far as I know, a lower Block Influence (BI) score indicates a higher cosine similarity between two layers, which means that the layer has minimal transformation on the hidden state and low importance, so it can be pruned with limited performance loss [1]. If so, please correct it; if not, please ignore this comment.

[1] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. ACL Findings, 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the impact of layer pruning on the test-time scaling capability of Large Language Models for long-chain reasoning tasks. Through experiments, the authors demonstrate that pruning even 1-2 layers severely impairs sequential test-time scaling, despite stability on knowledge-intensive tasks. Parallel scaling is also harmed by direct pruning methods but preserved by merging-based pruning. Additionally, supervised fine-tuning fails to recover the degraded test-time scaling. These findings provide fresh insights into building lightweight reasoning models.

### Strengths
1) The multiple conclusions identified offer clear reference value for lightweighting reasoning LLMs and for on-device deployment.  
2) The experiment covers a diverse spectrum of lightweighting techniques.
3) The work also supplies explicit qualitative and quantitative case analyses for the discovered phenomena.

### Weaknesses
1) Experiments have only been conducted on models with fewer than 10B parameters; results would be more convincing if larger-scale models were also included.  
2) When exploring supervised fine-tuning as a recovery remedy, incorporating the dominant RL recipes used in current reasoning-model training would further complete the findings of this work.

### Questions
Please refer to the weakness part.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper shows a finding that layer pruning methods that work well for non-reasoning models might not work as effectively for reasoning models. Also, the authors show that SFT training is not sufficient to recover the performance after pruning. The authors further try SFT training but this also does not recover the performance drop occuring from layer pruning.

### Strengths
The observation that layer pruning does not work effectively for reasoning models (which has become the de-facto model we adopt in the community for experiments) is timely and important.

### Weaknesses
1. This paper presents negative results but the explanations or experimental setting to analyze the negative results are limited. For example, the observation that "most layers play a non-trivial role in enabling test-time scaling" is very interesting, but the underlying explanation for whether that is not the case for "non-reasoning models" or what is the reason behind that is very limited.

2. As a follow-up of 1, I think there should be trends of non-reasoning models on the same experimental setting for Figure 2,3,4. A very simple way to do this would be to turn off the reasoning mode on Qwen3-8B and check if the trends differ after applying the pruning methods or Qwen2.5-7B-Instruct (which is the base model for s1.1).

3. For the qualitative example in Section 5.1 (Figure 5), could setting a higher temperature or applying repetition penalty mitigate this issue? Related to 1, there is insufficient explanation of why this repetition is happening to reasoning models versus non-reasoning models and what is the mechanistical or other reason behind this.

### Questions
See weaknesses above

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates on the effect of layer pruning in LLM long chain reasoning, showing that pruning even 1 or 2 layers significantly impair the performance in test-time scaling. The experiments cover different models, datasets, pruning methods and evaluation metrics, and the results are basically consistent. Furthermore, the authors show that SFT after pruning cannot recover the original performance.

### Strengths
The experiments contain sufficient ablations, covering different models, datasets, pruning methods and evaluation metrics. The overall conclusions are consistent and can well support the main claim that layer pruning hurts the long chain reasoning performance.

### Weaknesses
The main conclusion of the paper is simple and rather intuitive. The layer pruning method would surely decrease the performance since the model loses part of its parameters and face OOD problems compared with training. One could naturally expect these results without experiments, and the conclusions are known. The paper does not provide new interesting results, nor the solution to address the problem. 

Moreover, layer pruning itself is not a practically meaningful method from my perspective. Large scale pretraining / post-training aims to improve the reasoning performance, while minimum pruning would severely hurt the performance, which deviates from the original target. Why would people need layer pruning anyways? It is not a principled way in any sense. Even in terms of efficiency, pruning 1 or 2 layers would only bring marginal acceleration, while other methods such as distillation or quantization would significantly improve the efficiency without sacrificing much performance. Unfortunately, the main result of the paper lies in the natural consequence of layer pruning, which is deemed to hurt the performance, while the authors fail to provide any theoretical results, nor any successful methodologies to avoid the performance drop.

### Questions
* Since the performance degrade is natural, can you theoretically characterize the phenomenon? Note that contents in Section 5 are mostly case studies and heuristics, not theoretically grounded.
* Can you provide any practical methods to mitigate the issue? Did you try out other SFT configurations? The current setting seems not convincing (only SFT on s1K seems insufficient).
* Did you try out more models? Do you have intuitions on the slight difference in performance under various settings?
* Can you provide any convincing reasons why layer pruning is worth studying? Since all methods hurt the performance, what would be the next step?

### Soundness
3

### Presentation
3

### Contribution
1
