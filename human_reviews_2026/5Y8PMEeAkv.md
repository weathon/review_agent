# ZSPAPrune: Zero-Shot Prompt-Aware Token Pruning for Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
As the capabilities of Vision-Language Models (VLMs) advance, they can process increasingly large inputs, which, unlike in LLMs, generates significant visual token redundancy and leads to prohibitive inference costs. While many methods aim to reduce these costs by pruning visual tokens, existing approaches, whether based on attention or diversity, typically neglect the guidance of the text prompt and thus fail to prioritize task relevance. In this work, we propose a novel, zero-shot method that reframes the problem by introducing a prompt-aware perspective, explicitly modeling visual token pruning as a balance between task relevance and information diversity. Our hierarchical approach first selects a core set of task-relevant visual tokens and then supplements them with diversity tokens to preserve broader context. Experiments across multiple models and benchmarks show that our method achieves performance that matches or surpasses the state-of-the-art with only minimal accuracy loss, even when pruning up to 90\% of the tokens. Furthermore, these gains are accompanied by significant reductions in GPU memory footprint and inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ZSPAPrune, a zero-shot, plug-and-play visual token pruning method that accelerates inference in vision-language models without any fine-tuning. The approach selects a small budget of visual tokens in two stages: first choosing the tokens most relevant to the text prompt (task-relevant core set), then adding tokens that are maximally diverse to preserve global context. Evaluated on LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct under aggressive 90% pruning, ZSPAPrune matches or improves accuracy on benchmarks such as MMMU, GQA, AI2D, POPE, TextVQA, and ChartQA compared to strong baselines like DivPrune. The paper also reports modest latency and memory reductions at inference time and emphasizes that the method is model-agnostic and easy to integrate.

### Strengths
The paper presents a clear, zero-shot pruning method that balances prompt relevance and visual diversity, which prior work did not.

Experiments across strong VLMs and multiple benchmarks show it maintains or improves accuracy under extreme pruning while reducing cost.

The method is practically significant because it can be dropped into existing VLMs without any retraining or architectural changes.

### Weaknesses
The paper does not report direct quantitative comparisons against strong prompt-aware pruning baselines (e.g., GlimpsePrune), so it is hard to verify that the proposed approach is actually better than the closest prior work.


The efficiency claims are based on a single model/setting and only at an extreme 90% pruning ratio, with limited analysis of where latency and memory savings come from or how they scale with pruning level.

The method is essentially heuristic and lacks a clear formal objective or robustness analysis (e.g., failure cases when relevance vs. diversity is misbalanced).

The evaluation is limited to ~7B-scale VLMs, and there is no evidence that the proposed pruning strategy remains effective or stable for larger vision-language models, where attention structure and token redundancy may differ.

### Questions
How stable is ZSPAPrune across different prompt styles (e.g., long multi-step reasoning questions vs. short factual queries), and does the same relevance/diversity ratio work across them without retuning?

Have you investigated automatically selecting the relevance–diversity ratio at inference time (e.g., predicting it from the prompt or task type), rather than setting it manually per dataset?

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
4

### Summary
The paper proposes ZSPAPrune, a zero-shot, prompt-aware token pruning framework for Vision-Language Models (VLMs). Existing pruning methods are often prompt-agnostic, ignoring text guidance and thus failing to prioritize task-relevant visual information. ZSPAPrune addresses this by reframing pruning as a balance between task relevance and information diversity, achieved through a hierarchical process: Prompt Simplification, Prompt-Aware Selection, and Diversity Balance. The method selects core visual tokens most relevant to the prompt and augments them with diverse tokens to retain global context. Experiments on multiple benchmarks and models show that ZSPAPrune achieves state-of-the-art or comparable performance with minimal accuracy loss even when pruning up to 90% of tokens, while significantly reducing GPU memory usage and inference latency.

### Strengths
1. From a perspective of prompt-aware token selection to balance task relevance and information diversity in visual representations.
2. Introducing a hierarchical pruning mechanism composed of Prompt Simplification, Prompt-Aware Selection, and Diversity Balance to achieve controllable token reduction.
3. Achieving significant inference efficiency improvements with minimal accuracy loss under zero-shot settings across multiple Vision-Language Models and benchmarks.

### Weaknesses
1. The paper lacks comparison with other methods that explicitly address the trade-off between task relevance and information diversity. Without such comparison, it remains unclear whether the proposed balance strategy is superior or merely heuristic.
2. As a plug-and-play method, ZSPAPrune should be validated on more models with different parameter scales to confirm its general applicability. The current experiments are limited to a narrow range of architectures, reducing the evidence of scalability.
3. The comparison with task-relevance-based approaches appears potentially unfair. Some baselines are reimplemented without clear alignment in training setup or hyperparameter tuning, which may bias the reported results.
4. The proposed method is overly simple and lacks crucial theoretical analysis. No formal justification or complexity discussion is provided to explain why the hierarchical prompt-aware pruning mechanism should work effectively.
5. The framework figure (i.e., Figure 2) is overly general and resembles a process diagram rather than an architectural framework. It fails to visually highlight the innovation and importance of the proposed components, and a more informative figure is recommended.

### Questions
null

### Soundness
3

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
4

### Summary
This paper studies token pruning issue in vision large language models. Specifically, it takes token pruning in vLLMs as  a tunable balance between task relevance and information diversity. In implementation, the prompt-aware score by calculating the relevance between prompts and token embeddings, while the diversity balance is calculated by selecting the token most disimilar to all previously selected tokens. Experiments are done on several benchmarks.

### Strengths
The strengths are as follows:  
1.The paper is easy to read and the method is easy to follow.  
2.Evaluated datasets and vLLMs are diverse.

### Weaknesses
The weakness are as follows:  
1.There are many existing works on task relevance of token pruning for vLLMs. This work additionally considers the information diversity, which seems incremental novelty. Meanwhile, in Figure 1, it is not easy to understand why the information diversity is useful for token pruning task.   
2.Missing related works. Recently, there are many other token pruning methods[1,2,3,4], which are not analyzed and discussed in this work. These works should also be added for comparison.  
3.In the method design, I have some concerns:  
(1) In Eq.4, averge pooling is applied on the prompt token embeddings. It is not quite reasonable since the prompt text may involve many not task-releted tokens.  
(2) The diversity balance is performed by selecting some tokens dissimilar to previously selected ones. Probably, this could select some useless tokens and background tokens. I am not sure this motivation is correct.  


[1] Boosting multimodal large language models with visual tokens withdrawal for rapid inference  
[2] Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification.  
[3] Visionzip: Longer is better but not necessary in vision language models  
[4] Folder: Accelerating multi-modal large language models with enhanced performance

### Questions
See above

### Soundness
1

### Presentation
2

### Contribution
1
