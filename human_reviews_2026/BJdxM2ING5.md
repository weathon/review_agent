# Residual Connections Relay Generalization but Not Memorization in Transformers

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Residual connections are one of the main components in transformers, helping stabilize training and improve optimization, yet it remains unclear how they influence memorization, a behavior that transformers are known to exhibit, especially in overparameterized regimes. Therefore, in this work, we investigate the impact of residual connections on memorization in transformers. Our analysis shows that residual connections do not influence memorization; instead, their removal primarily impairs learning, which is a novel finding. Furthermore, we find that residual connections in early layers are significantly more important for performance than those in later layers. To explain these findings, we perform a gradient flow and output margin analysis, demonstrating how residual connections support learning dynamics without propagating memorization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
It studies the role of residual connection on memorization and generalization in transformers, finding that removing one residual connection does not affect memorization, but affect generalization when the removal happens in early layers.

### Strengths
It does detailed ablation experiments, and train models on multiple datasets.

### Weaknesses
It's not surprising that deeper layers are less effective e.g. see [1]. It's expected that removing residual connection at deeper layers has less effects on the test accuracy, and early residuals are critical for learning is not something new.

If we can stably train LLMs with residual connection removed, it's expected that it can memorize the training data since it's overparameterized.

The noise label rate is fixed to 1% for all experiments, the claims may no longer hold when noise label rate becomes 10%.

[1] The Curse of Depth in Large Language Models

### Questions
1. What are the gradient norm dynamics during training for the models with residual connection removed? It would be better if they are provided along with the theoretical analysis of gradient norms
2. In each decoder layer, there are two residual connections, do you remove both of them or just one of them? After removal, is the training still stable?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an empirical investigation exploring the impact of residual connections on generalization and memorization in transformer architectures. The authors perform a series of experiments using various architectures (GPT2, Small-LM, Qwen2, ViT-B, etc.) and datasets (Emotions, 20Newsgroup, CIFAR, etc.). The study's main findings are:

- Removing residual connections has a negative effect on generalization, but not on memorization.
- Early residual connections help the most with generalization capabilities.
- The norm of the gradient with respect to the residual input is higher for natural examples than for noise examples.

### Strengths
- The paper proposes a novel empirical study examining the effect of residual connections on memorization in transformer architectures.
- The authors perform a broad set of experiments, considering 8 architecture variants and 7 datasets.
- Empirical results clearly show the impact of residual connections on generalization (Fig. 1).
- The authors propose an analysis to explain why residual connections help with generalization, investigating the effect on the gradient norm.

### Weaknesses
- The residual connection is one of the most standard components of modern deep learning architectures. While this study explores a novel aspect of residual connections, it is unclear how one could use these insights to improve existing architectures. So, while the contribution showcases a new effect of residual connections—which is valuable—it is unclear how one could build on this observation.
- Most of the experiments use 1% label noise as the setup. How robust are the paper's findings for several levels of noise (1%, 5%, 10%, 50%, etc.)?
- Similarly, it seems that most of the gradient analyses are done after model training. How do the different metrics (gradient norm, output margin, etc.) evolve during training? Would the conclusions be different at different times during training?
- In line 300, the authors make the assumption that x^C ≈ x^NL as they come from the same class. This seems quite strong and unnecessary, as the authors verify empirically that the standard deviations are similar for the two examples.

### Questions
See Weaknesses.

### Soundness
3

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
This paper demonstrates, both experimentally and theoretically, that residual connections affect generalization performance but do not contribute to memorization on training data with added noise. It compares test error with and without residual connections, and memorization measured on training data with 1% label noise. As a result, it experimentally shows that the presence or absence of residual connections influences test error but does not affect memorization. The paper also provides a theoretical interpretation via upper bounds on gradient norms to explain these experimental results. In particular, it shows that the impact of residual connections is larger in the shallow layers.

### Strengths
- By comparing gradient norms in residual connections, this paper shows that the learning gradient norm is larger than the memorization gradient norm, thereby demonstrating, both experimentally and theoretically, that residual connections tend to transmit learning-relevant information through gradients.

- It was experimentally shown that, particularly in the shallow layers, residual connections have a large impact on test performance.

### Weaknesses
- Since residual connections have no parameters and do not memorize data, it seems natural that, if the training loss can be driven to zero, removing residual connections would not change memorization performance. In that respect, there appears to be little novelty. Moreover, performance improvements from adding residual connections were demonstrated in the original ResNet paper, so the novelty also seems limited in that sense.

- Parameterized layers memorize data. Given that, it seems that removing residual connections would not change memorization. The argument based on upper bounds of gradient norms is indirect, and an inequality between the learning gradient norm and the memorization gradient norm only states an ordering, which is weak to support the high memorization reported in the experiments.

- Because residual connections pass information forward, removing them in the early layers impedes the flow to the deeper layers of the network, so it is to be expected that learning in those deep layers will be affected. Prior work has described this as learning a coarse-grained view, so this seems to be a restatement. In addition, the inequality presented in Section 5.1 is not actually proved.

### Questions
- Rather than removing residual connections entirely, one could multiply them by a suitable coefficient and study their effect continuously. This would make it possible to show whether the phenomena reported in the paper occur only when residual connections are completely removed, or whether they can also arise when the residual pathway is merely weakened. Eliminating residual connections is a large intervention on the model, and there is a possibility that the reported drop in generalization performance simply reflects continuing to use hyperparameters tuned for a model with residual connections.

- Since the parameter count does not change with or without residual connections, might the memorization metric be essentially determined by model size? Note that from a training standpoint, if residual connections are entirely absent, increasing depth can cause vanishing gradients, so the difficulty of memorizing the data may simply increase.

- The current experiments use only 1% noise, which is a restrictive setting. Experiments with other noise levels, such as 5% or 10%, are natural to consider, so what is the reason for using only 1%?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper asks whether residual (skip) connections in transformers propagate memorization or merely aid learning/generalization. The authors inject 1% label noise into several benchmarks, train models to 100% training accuracy, and then study (i) accuracy on clean test sets (a proxy for generalization) and (ii) accuracy on the mislabeled training points (their memorization metric). Across 8 transformer variants spanning vision and text (e.g., ViT‑Base, TinyViT, BEiT, DeiT, GPT‑2 Small/Medium, Qwen2‑0.5B, Smol‑LM) and 7 datasets, they report that removing residual connections hurts test accuracy but leaves "memorization" essentially unchanged. Layerwise ablations further suggest early residuals matter most for generalization. They support these findings with analyses of gradient norms w.r.t. residual-stream inputs, an output‑margin study, and two theorems giving upper bounds that attribute the memorization/learning gap primarily to the MSE betweenn y_hat and y and to multiplicative depth factor.
Overall, this is an interesting paper but I think requires some revisions before being ready for publication

### Strengths
Measuring gradients w.r.t. residual‑stream inputs and relating them to output margins and accuracy (Figs. 3, 6, 7) gives a coherent picture of where learning signal flows

The cross‑modal, multi‑model layerwise study is thorough and consistent. The effect shows that early residuals are far more consequential for test accuracy than later ones, while the chosen memorization metric barely moves

### Weaknesses
The memorization story is, in its current form, undermined by a narrow metric and by an experimental setup that can make "100% memorization regardless of residuals" nearly tautological. The theoretical analysis is intuitive but rests on sigma-based bounds whose applicability to trained, normalized transformers is not fully substantiated.


Measuring only label‑noise fitting on small‑/mid‑scale classification fine‑tunes misses prominent memorization phenomena in transformers (e.g., verbatim recall/PII leakage in generative LMs). Claims like “residuals do not relay memorization” are too broad given the limited metric and tasks

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
