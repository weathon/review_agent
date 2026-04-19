# The Fair Language Model Paradox

- Decision: Reject
- Scores: 5, 3, 6, 5

## Abstract
Large Language Models (LLMs) are widely deployed in real-world applications, yet little is known about their training dynamics at the token level. Evaluation typically relies on aggregated training loss, measured at the batch level, which overlooks subtle per-token biases arising from (i) varying token-level dynamics and (ii) structural biases introduced by hyperparameters. While weight decay is commonly used to stabilize training, we reveal that it silently introduces performance biases detectable only at the token level. In fact, we empirically show across different dataset sizes, model architectures and sizes ranging from 270M to 3B parameters that as weight decay increases, low-frequency tokens are disproportionately depreciated. This is particularly concerning, as these neglected low-frequency tokens represent the vast majority of the token distribution in most languages, calling for novel regularization techniques that ensure fairness across all available tokens.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Inspired by a discovery in the computer vision field regarding unbalanced sample classification tasks, where regularization methods are more effective for larger classes, the authors investigated the impact of weight decay—a common regularization technique—on token-level learning dynamics in large language models. The experimental results demonstrated that as weight decay increases, the performance of low-frequency tokens is disproportionately affected, while high-frequency tokens are learned faster than their low-frequency counterparts. This is a novel finding, as previous work typically relies on aggregated training loss measured at the batch level, overlooking token-specific dynamics.

### Strengths
1.This represents a novel finding, as prior work has typically focused on aggregated training loss measured at the batch level, neglecting the detailed dynamics of individual tokens.
2.The experiments demonstrated that the models' performance on low-frequency tokens significantly deteriorates as weight decay increases.

### Weaknesses
1.The paper does not offer specific insights on the implementation of regularization techniques that ensure fairness across all tokens. While the impact of weight decay on low-frequency tokens is highlighted, there is no detailed discussion on how to address this imbalance or propose alternative regularization methods that might mitigate the disproportionate effect on low-frequency tokens, ensuring more equitable performance across the entire vocabulary.
2.There is a lack of experiments or other evidence demonstrating the necessity of treating low-frequency tokens with the same level of importance as high-frequency tokens. Furthermore, it remains unclear whether this approach could lead to other issues, such as affecting the stability of model training or overall model performance. Addressing these concerns would be essential to understanding the broader implications of implementing equal importance for all tokens in language models.

### Questions
My main point of confusion revolves around whether low-frequency tokens and high-frequency tokens should be treated equally in large language models. Could you provide some practical examples to illustrate this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors analyze the relationship between weight decay and token-level loss in large language models. As a regularization technique that stabilizes model training, weight decay is widely applied in the training of large language models. Through the training of models ranging from 270M to 3B,  the authors have discovered that as weight decay increases, the ability of the model to learn low-frequency tokens deteriorates, which is reflected by an increasing loss for these low-frequency tokens. Additionally, the gap between the losses for low-frequency and high-frequency tokens also grows larger. This phenomenon suggests that there is a need to develop new regularization techniques to avoid this issue.

### Strengths
1. The paper presents a novel perspective for analyzing the performance of large language models. The author observed the difference in the learning high-frequency and low-frequency tokens, and identifies the cause of the differences, namely, the weight decay regularization technique. The experimental results demonstrate a significant correlation between weight decay and the loss of low-frequency tokens.

2. In addition to empirical conclusions, the authors also provide a theoretical disscussion on the impact of weight decay on per-token loss for different token frequencies.

### Weaknesses
1. The experiments in this paper use the IMDB corpus for model training. However, this corpus is biased and differs significantly from mainstream pre-training corpora. Consequently, it may not adequately reflect potential issues in mainstream large language model training.

2.  The experiments in this paper are based on training sequences of lengths 128 and 64, which are somewhat too short for large language model (LLM) training. For instance, in Figure 2, the tokenized tokens using the llama3 tokenizer already consists of 92 tokens, which appears to be relatively short text even in common pre-training corpora. A widely recognized viewpoint is that the context window has a significant impact on the per-token loss of LLMs, and longer context windows can help the model learn better. Mainstream models typically use a length of around 8192, and there is a considerable gap between this window length and the lengths used in the authors' experiments. Consequently, whether these conclusions can be generalized to mainstream large language models remains to be further validated.

3. This experiment compared the impact of weight decay \(\lambda\) ranging from 0.0 to 2.0 on the model. From Figure 1, Figure 4, and Table 1, it can be observed that starting from \(\lambda = 0.3\), there is a noticeable change in the per-token loss for low-frequency tokens. However, most current LLMs set the weight decay \(\lambda\) to 0.1, which, as shown in Table 1, has a negligible impact on the model. Therefore, whether the issue raised by the authors is universal remains to be further examined.

### Questions
Page 4, line 221 mentions that the experiments were conducted on an A100 32GB GPU, but Nvidia A100 does not have a 32GB version. It is suspected that this should be Nvidia V100 instead.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The article proposes that the increased weight decay of large language models leads to model underperformance on low-frequency tokens and significantly better performance on high-frequency tokens, which can lead to model bias and unfairness. It triggers further thinking in the field of NLP on the contradiction between model generalization performance and model bias under long-tailed data, and focuses the attention of large language models on token-level performance.

### Strengths
1. Innovative thinking on model weight decay for unbalanced class distribution data: the article proposes that the increased weight decay of large language models leads to model underperformance on low-frequency tokens and significantly better performance on high-frequency tokens, which can lead to model bias and unfairness. It triggers further thinking in the field of NLP on the contradiction between model generalization performance and model bias under long-tailed data, and focuses the attention of large language models on token-level performance. 
2. informative textual analysis and experimental validation: the article experimentally validates the average model performance, the impact of token-level performance under weight decay and own word frequency, and theoretically analyzes why the loss function of high-frequency tokens monotonically decreases when trained with weight decay, rigorously arguing the point of view from the theory and experiments
3. challenges to existing practices: the paper challenges the weight decay technique commonly used in current LLM training practices, and emphasizes the need to develop new regularization techniques to ensure the fairness of all tokens.
4. concise language and clear logic: the paper is concise and logical, and the experimental results are well organized to help readers clearly understand its research contributions.

### Weaknesses
1.dataset limitation: although the paper uses the IMDB dataset for experiments, the dataset is limited in types and domains, and may not be able to fully represent the model's performance in diverse tasks and domains.
2.Lack of different regularization comparison experiments: the paper lacks comparison experiments for the effects of different regularization techniques, for example, comparison with other types of regularization methods (e.g., dropout, data augmentation, etc.), which can make the experimental results more convincing.
3.There's still room to explore: although the article puts forward the contradiction between the fairness at the token level and model generalization under the existing regularization techniques, it does not put forward a proven solution, which is regrettable.

### Questions
I don't have questions.

### Soundness
3

### Presentation
3

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
The paper titled The Fair Language Model Paradox presents an investigation into token-level biases in Large Language Models (LLMs) induced by weight decay, a common regularization method. The authors explore how weight decay affects low-frequency tokens disproportionately, leading to performance degradation in these tokens even as aggregated training loss metrics remain stable. This study reveals the hidden biases against low-frequency tokens, calling for more equitable regularization techniques to ensure fairness across the token distribution.

### Strengths
The paper brings forward a nuanced perspective on weight decay, highlighting an often-overlooked effect on low-frequency tokens in LLMs. This is particularly timely given the widespread use of weight decay without token-level monitoring.


The study uses multiple models with varying architectures and sizes across different datasets, demonstrating the robustness of the findings.

### Weaknesses
The use of only the IMDB dataset (including an extended version) raises concerns about the generalizability of the results across other types of text data. Testing on a more varied set of corpora (e.g., diverse languages or topics) would strengthen the claims about low-frequency token bias.

The paper’s theoretical discussion on the link between token frequency, regularization, and loss functions feels dense and somewhat disjointed from the empirical findings. A clearer integration of these theoretical insights into the experimental results would enhance the readability and cohesion.

The broader impact section is sparse, particularly given the potential implications of token biases in LLMs on marginalized dialects or low-resource languages. The authors could deepen their exploration of societal impacts to underscore the relevance of their findings.

### Questions
Did the authors consider alternative regularization methods beyond weight decay during their experiments?

How would the findings differ if tested on corpora with varied linguistic features, such as highly inflected languages or low-resource languages?

Would the proposed metrics, such as per-token learning speed, generalize effectively to larger and more diverse datasets?

### Soundness
3

### Presentation
3

### Contribution
3
