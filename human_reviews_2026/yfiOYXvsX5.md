# Causal-Steer: Disentangled Continuous Style Control without Parallel Corpora

- Decision: Accept (Poster)
- Scores: 6, 4, 2

## Abstract
Controlling stylistic attributes of Large Language Models (LLMs), such as formality or conceptual complexity, is crucial for effective human-AI interaction. However, current methods often suffer from discreteness, reliance on expensive parallel corpora, and instability, limiting their practical utility. This paper introduces a novel framework for robust activation steering that eliminates the need for parallel corpora, enabling continuous, fine-grained, and linear control over LLM outputs. Our key insight is to reframe Low-Rank Adaptation (LoRA) as a causal intervention tool. By contrasting activations on identical inputs with and without a LoRA perturbation trained via a contrastive objective, we separate the influence of content. To enhance reliability, we introduce a robust aggregation pipeline that uses Principal Component Analysis (PCA) for denoising and the geometric median for centrality estimation, yielding a stable and disentangled style vector. At inference, this vector allows for precise bidirectional control via activation steering with negligible computational overhead. We demonstrate state-of-the-art performance on controlling conceptual complexity, text detoxification, and formality control. Our method not only provides superior control but also generalizes across different models and tasks, and enables simultaneous multi-attribute control.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a framework for continuous and bidirectional style control in LLMs without requiring parallel corpora. The key idea is to reinterpret LoRA as a causal intervention. Style-specific LoRA modules are trained in a contrastive learning manner to capture style features. Then, their influence is isolated by comparing activations with the frozen base model. The resulting style vectors after postprocessing are applied at inference time through activation steering. Experiments on multiple benchmarks show that their method outperforms prior methods and achieves controllable style transfer while preserving fluency and relevance.

### Strengths
- I like the reinterpretation of LoRA as a causal intervention, which is conceptually novel and technically lightweight.
- The empirical results are strong, demonstrating the effectiveness of the proposed method.

### Weaknesses
- The method requires two LORA models, which limits the style to be binary.
- Although I understand that many early style transfer works relied on parallel corpora, I do not consider this to be a novel contribution for LLMs. Using non-parallel corpora can provide the model with richer data for training, increasing the chance that LLMs encounter more diverse lexical distributions.
- The recently proposed Persona Vectors[1] method from Anthropic is conceptually similar to this work, as both control generation through directions in the latter activations. Several of the other baselines mentioned in the paper are also training-free, in contrast to the proposed approach, which involves additional training and thus carries greater complexity.

[1] https://arxiv.org/pdf/2507.21509

### Questions
- How can you extend the model to support more general, multi-directional controllable style transfer?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel framework to improve activation steering for stylistic attributes, by reframing LoRA as a causal intervention tool. The authors claim that by using LoRA, they are able to disentangle stylistic and semantic variation, thereby allowing for better steering on tasks that no longer rely on parallel data. Additionally, the authors add PCA based denoising to identify vectors that are transferable across different task settings. Concisely:
1. The authors use LoRA to introduce a low rank perturbation that can extract style-specific activation differences.
2. Extracted style vectors are denoised using PCA
3. Style vector is normalized and used for activation steering.
Authors test for generalization across multiple task settings and models.

### Strengths
- Strong empirical work, with extensive experimentation across multiple models and tasks, thus showing evidence for generalization.
- Comparisons with multiple baselines.
- Causal-Steer is useful at higher steering factors alpha
- Fun multi-lingual example testing, though in appendix.

### Weaknesses
I am really confused by the core claim this paper is making. My understanding of the claim is that Causal Steer
- Eliminates the need for parallel corpora, but needs contrastive corpora
- Separates stylistic differences while keeping content the same

1. What is the difference between parallel vs contrastive corpora? For eg. I suggest the authors look at [1] where the authors steer using two datasets that are not parallel (as in are not required to differ at 1 token position etc) and instead just exhibit contrastive meta properties, i.e. one dataset contains harmful questions and the other contains helpful questions. They use difference in means activation and it works very well, even though there isn't a high consistency between the contrastive inputs.

> However, this method assumes that content-related information can be eliminated via vector subtraction, thereby isolating the pure style signal. This assumption holds only under the condition of extremely high content consistency between texts, which in turn necessitates a meticulously crafted parallel corpus. Causal-Steer, in contrast, captures the LoRA-induced perturbation on the very same input di, elegantly circumventing this reliance on parallel corpora.

then becomes unsubstantiated.

2. The paper should also look at ReFT ([2]), which uses LoRA adaptors to fine-tune the representations model directly on a set of inputs and outputs, therefore truly no parallel corpora and is highly performant and efficient. This should definitely be a baseline in your paper

3. It's unclear how the layers where the steering vectors were applied were selected. The authors simply mention "Specifically, for a selected set of layers $L_{steer}$, we intervene in the output of the MLP submodule during the generation of each token $t$". How many layers are being steered on? How are these layers determined? Is the intervention applied on the entire MLP output or on a partial subset -- it's unclear. 

4. The authors utilize the same activation difference approach used in earlier methods, but with LoRA, and by doing so they negate some of their own claims such as 

>However, this method [difference-in-means] assumes that content-related information can be eliminated via vector subtraction, thereby isolating the pure style signal.

Don't the authors make this same assumption but for the LoRA fine-tuned activations? Why would the former not hold while their difference holds?

5. The use of causal-intervention is very shaky here. The intervention is LoRA training + PCA over _all_ layers + geometric mean. Any of these could be causing the behaviors produced by the authors. The ablation studies show that all factors are important.

6. Unclear what metrics are being reported in each table. The information in the appendix is valuable. Could the authors update the table captions as well as the main text to reflect that the scores are from 1 - 10?

7. Missing citations. A few hand-picked examples:

> We attribute this failure to its use of PCA for analyzing activations, a technique that requires a much larger dataset to identify a meaningful style direction. This hypothesis is supported by our generalization experiments, where its performance improves on the larger Formality dataset (16,000 examples). 

> Intervening in the initial layers (1-10) proves far less effective and can even be detrimental to the output. This empirical result aligns with the prevailing hypothesis that later transformer layers encode more abstract semantic and stylistic information.

> Flesch Grade Level

8. The text in the figures is extremely tiny in many cases. Please fix this. 

9. Many sentences are unclear:

> Removing the contrastive learning objective (“- w/o Contrast”) causes a catastrophic failure in
bidirectional control, with the model unable to generate ”Easy” content.

What is easy content? Why is this "catastrophic" failure? Please keep terminology and wording consistent between paragraphs.

10. The main figure is very hard to read, unclear what the different suffixes are given the content at this stage. The formality/informality task comes much later.

[1] Arditi, Andy, et al. "Refusal in language models is mediated by a single direction." Advances in Neural Information Processing Systems 37 (2024): 136037-136083.
[2] - Wu, Zhengxuan, et al. "Reft: Representation finetuning for language models." Advances in Neural Information Processing Systems 37 (2024): 63908-63962.

### Questions
1. What is the difference between parallel vs contrastive corpora? 
2. Why is this a "causal" steer? The ablation studies show that all factors are important., so which is the causal factor here? Why is LoRA training alone attributed causality? 
3. 
>However, this method [difference-in-means] assumes that content-related information can be eliminated via vector subtraction, thereby isolating the pure style signal.

Don't the authors make this same assumption but for the LoRA fine-tuned activations? Why would the former not hold while their difference holds?
4. It's unclear how the layers where the steering vectors were applied were selected, or how many layers the intervention is applied on. The authors simply mention "Specifically, for a selected set of layers $L_{steer}$, we intervene in the output of the MLP submodule during the generation of each token $t$". How many layers are being steered on? How are these layers determined? Is the intervention applied on the entire MLP output or on a partial subset -- it's unclear.

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
3

### Summary
In this paper the authors present Casual-Steer, an approach to control LLM output with a single linear scale to move between two reference styles. The approach makes use of LoRA and a specialized loss to allow for controllable weights between the two styles. The authors investigate a number of approaches for extracting a controllable vector and primarily make use of a PCA-based approach. The authors compare to a number of text stylization baselines, finding similar performance across many text quality metrics, all of which arise from prompting ChatGPT-4.1.

### Strengths
The originality of the paper is quite high. While LoRA controllability approaches exist, none make use of this specific setup. If successful and generalizable this approach could be very useful for stylized text generation without expensive additional processing. This is the greatest strength of the paper. 

The quality of the work in terms of the system overview is high. The experiments are more mixed. The use of two different models (even if they are somewhat small) and three different pairs of datasets is excellent. However, the reliance on ChatGPT-4.1 for all metrics is not ideal. Further, the authors do not present standard deviation or other analyses of the distribution of scores. 

The clarity of the paper is also high. The authors' approach is well-explained and outside a few grammar issues the paper is well-written. The figures are also clearly presented. 

The significance of the work is hampered primarily due to the issues presented with the results.

### Weaknesses
I think this is overall a strong paper hampered by weak evaluations. The authors primarily make use of ChatGPT-4.1 for evaluating the output text and primarily make use of text quality metrics. Table 1 demonstrates very little difference in ChatGPT-4.1 scores. The authors do not present any standard deviation values, making it unclear if they ran each experiment a single time (not ideal) or if the std dev values would make clear that all outputs fit within the same distribution. The Flesch grade level is the most useful metric in the initial table, but even this shows very minor variation in most cases. There is no standard deviation reported for this either. 

Similarly, the Toxicity metric results in Table 2 are somewhat confusing. Assumably the Toxic style should be more toxic, not less toxic, but the authors' approach achieves the lowest toxicity in both cases. Unfortunately no standard deviation score is reported.

The human evaluation, while it only appears in the appendices, is too small to draw any generalized takeaways from, with only three annotators. The annotators also only evaluate the quality of the text. 

Given the authors claims around controllability rather than text quality I would have preferred to see evaluations around the controllability of the text. 

More broadly, only a single text example is given in the paper, including in the appendices. 

Given all of this, the results are simply not sufficient to determine the extent to which the authors' claims are supported.

There are also a few grammar issues in the paper. For example in 4.0 it should likely be "Evaluation Metrics" not "Evaluation Metric" and "in the Appendix E" should just be "in Appendix E".

### Questions
1. What were the standard deviation values for the metrics in the paper?
2. Why did the authors focus on text quality?
3. In Table 4, why does Casual-Steer present the lowest toxicity in both cases?

### Soundness
1

### Presentation
3

### Contribution
2
