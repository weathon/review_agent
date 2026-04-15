# Core Tokensets for Data-efficient Sequential Training of Transformers

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Deep networks are frequently tuned to novel tasks and continue learning from ongoing data streams. Such sequential training requires consolidation of new and past information, a challenge predominantly addressed by retaining the most important data points - formally known as coresets. Traditionally, these coresets consist of entire samples, such as images or sentences. However, recent transformer architectures operate on tokens, leading to the famous assertion that an image is worth 16x16 words. Intuitively, not all of these tokens are equally informative or memorable. Going beyond coresets, we thus propose to construct a deeper-level data summary on the level of tokens. Ours, respectively named core tokensets, both select the most informative data points and leverage feature attribution to store only their most relevant features. We demonstrate that core tokensets yield significant performance retention in incremental image classification, open-ended visual question answering, and continual image captioning with significantly reduced memory. In fact, we empirically find that a core tokenset of 1\% of the data performs comparably to at least a twice as large and up to 10 times larger coreset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel concept of core tokensets, which selects only the most informative / important tokens from training samples as replay data for continual learning.
Previous works also study how to select more important samples for memory efficiency, but they do not operate at the token level.
This work further enhances this idea with a two-step selection process. First, they select important training samples with coreset selection methods (like GradMatch). Then, they identify and retain only the most relevant tokens from those selected samples using methods like Grad-Cam, Grad-LRP, and Atman.
To make transformers amenable to such partial inputs, this paper also employs random token dropping during training.
In summary, core tokensets provide an advanced way to sub-select training data at the token level, leveraging the properties of transformer architectures to enable highly efficient data replay for continual learning.

### Strengths
- The paper introduces the concept of core tokensets, that selects the most informative tokens from important data samples.
- The authors extensively evaluate their approach across multiple sequential learning tasks, including image classification, visual question answering, and image captioning.
- The paper explores various strategies for selecting important tokens, including Grad-Cam, Rollout, GradLRP, and Atman, providing a thorough comparison.
- I like this paper's straightforward writing style, as it provides detailed explanations of existing methods and comprehensive results, rather than over-emphasizing a supposedly novel method.

### Weaknesses
- The proposed token-based core tokensets are specifically designed for transformer models, which are widely used today. But this idea may have limited applicability to inspire other fields.
- Although this paper presents detailed results, I think its main drawback is the lack of theoretical findings, which makes the work silghtly looks like an experimental report. As shown in Table 1  and Table 2, the best results for Core Tokenset are obtained by combining the top Coreset selection and Core Token selection methods. However, the performance differences between all these approaches are small. This gives the impression that the paper simply stating, "A is good, B is good, so A+B is better." However, how to know which one is better? To improve the paper, I suggest conducting more theoretical studies to explain the selection process.

### Questions
What is the theoretical bounds or guarantees on the performance of core tokensets?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new way to extract core tokenset. Overall, the paper is well written. My comments are as below:
1. I can see that the tasks used in the paper are generally easy task with 90%+ accuracies. I'm wondering the performance of the proposed method on more difficult tasks.
2. I want to see more visualized results. Will that be possible to show the extracted token, and corresponding image patches?
3. I think the work can be related to model explainability, will the authors cite some related works and explain the model?
4. How to define the relevancy among tokens? I still think the proposed method lack theoretical basis; also, I want to see more model training details.

### Strengths
The proposed method is with practical value, which could be applied to lots of applications.

### Weaknesses
As can be found in the summary.

### Questions
I want to see how the authors define the "discriminative tokens". More mathematical reasoning should be provided.

### Soundness
3

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
3

### Summary
The paper introduces the concept of core tokensets for data-efficient sequential training of transformers, demonstrating improved performance and reduced memory usage in various vision and language tasks, including image classification, visual question answering, and image captioning.

### Strengths
- **Novel Concept**: The concept of core tokensets, which select the most informative tokens within informative data points, represents a novel and efficient way to summarize datasets for sequential training of transformers.

- **Solid Performance**: Core tokensets demonstrate comparable effectiveness to other data selection methods (e.g., Coreset, Core Token) while substantially reducing memory usage.

- **Versatility**: Core tokensets was shown effective on multiple different vision and language tasks inlcuding image classification, visual question answering, and image captioning

- **Presentation**: The paper is clearly written and easy to follow.

### Weaknesses
While core tokensets significantly reduce memory usage, their two-step approach—first selecting informative data points and then identifying key tokens within those points—may introduce more latency in dataset summarization compared to single-step methods (e.g., Coreset, Core Token).

### Questions
- Have you compared the latency of core tokensets with other data selection methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores the incremental learning task using transformers.  The authors focus on improving the efficacy of replay methods, where they argue for storing key tokens rather than entire images.  They evaluate performance on image classification, VQA, and captioning tasks, where they improve performance over other coreset methods.

### Strengths
1. I do not know of another paper that explores the effect of storing a subset of tokens for replay in incremental learning
2. The diversity of tasks in the experiments is good- many incremental learning papers just evaluate on one task
3. Using a subset of tokens could also benefit efficiency of replay methods.

### Weaknesses
1. Identifying a subset of tokens that does as well as the whole sample has been explored in prior work.  In particular, there are token pruning/merging methods (e.g., [A,B]) as well as methods used to for faster training (e.g., [C]).  The authors don't discuss or compare to these methods (I only gave examples, but there are many papers on this topic that should be included in the discussion).  As such, one could see this paper as simply a new application of a known technique.

2. The authors do not compare to the state-of-the-art, but rather restrict their comparisons to closely related papers.  As such, it isn't clear how the proposed approach compares.  The authors should consider more recent methods as well as those from the broader literature for their comparisons (e.g., [D,E]).

3. It would be interesting to see how the proposed method works on a more diverse set of datasets.  While the diversity of tasks was appreciated, they are typically limited to a single dataset (and ImageNet is also restricted to a subset of categories.  A more thorough evaluation like those in the related work would improve the paper and provide additional insight into how well the proposed approach works (e..g,, [F]).

[A] DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. NeurIPS 2021

[B] Beyond Attentive Tokens: Incorporating Token Importance and Diversity for Efficient Vision Transformers. CVPR 2023

[C] Masked Autoencoders Are Scalable Vision Learners. CVPR 2022

[D] Adaptformer: Adapting vision transformers for scalable visual recognition. NeurIPS 2022

[E] InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning. CVPR 2024

[F] CLR: Channel-wise Lightweight Reprogramming for Continual Learning. ICCV 2023

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
