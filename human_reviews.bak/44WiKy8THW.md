# Integrating Geodesic Interpolation and Flow Matching for Non-Autoregressive Text Generation in Logit Space

- Decision: Reject
- Scores: 1, 5, 3

## Abstract
Non-autoregressive language models are emerging as effective alternatives to autoregressive models in the field of natural language processing, facilitating simultaneous token generation. This study introduces a novel flow matching approach that employs Kullback-Leibler (KL) divergence geodesics to interpolate between initial and target distributions for discrete sequences. We formulate a loss function designed to maximize the conditional likelihood of discrete tokens and demonstrate that its maximizer corresponds to the flow matching velocity during logit interpolation. Although preliminary experiments conducted on the TinyStories dataset yielded suboptimal results, we propose an empirical sampling scheme based on a pretrained denoiser that significantly enhances performance. Additionally, we present a more general hybrid approach that achieves strong performance on more complex datasets, such as Fine Web and Lamini Instruction.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
2

### Summary
This work presents a flow matching approach for generating discrete sequences. This approach treats discrete tokens as one-hot vectors and constructs a flow by interpolation on the logit space. Randomized top-k sampling is proposed for inference.

### Strengths
N/A

### Weaknesses
This paper is only half-baked and needs substantial refinements before resubmission. For example, the presentation is poor (many variables are not explained, Figure 1/2 have the same caption, and some references are placeholders), experiments are only conducted on toy datasets (Tiny Stories, MNIST), and evaluation metrics are not sound (only use generative perplexity for language modeling).

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a novel method for non-autoregressive text generation using KL-divergence geodesics and flow matching in logit space. The authors propose a conditional flow matching approach to address the challenges of discrete sequence modeling, demonstrating theoretical alignment between the loss function and flow matching velocity. To enhance performance, they implement an empirical sampling scheme based on a pretrained denoiser. Experiments on both text and image datasets show that the method outperforms traditional autoregressive models. Despite promising results, the sampling technique lacks full theoretical justification.

### Strengths
- The paper introduces a novel application of KL-divergence geodesics for text generation, addressing limitations in linear interpolation commonly encountered in discrete sequence modeling.

- The use of a pretrained denoiser-based empirical sampling scheme demonstrates ingenuity, compensating for initial performance shortcomings and achieving improved generation results.

### Weaknesses
- The paper seems to have been written in a hurry and lacks proper polish, with numerous missing references that make it difficult for me to follow. For example, references are missing at lines 32, 33, 39, 53, and 90, which disrupts the flow of the paper.
- I find the experimental section quite limited, as it only includes a single experiment for both text and image generation. A detailed ablation study is missing, making it hard to understand the impact of different components.
- I believe the evaluation metric for text generation is too restricted, relying almost exclusively on perplexity. While perplexity is useful for understanding how well the generated text fits the probable distribution, it can fail to capture semantic richness. I would recommend adding metrics like BLEU, ROUGE, or exploring newer evaluation methods for a more comprehensive assessment.
- After reading the introduction, I still do not fully understand why flow matching is necessary for generation models. The motivation for choosing this specific approach remains unclear to me.

### Questions
See the Weaknesses part

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a novel approach for non-autoregressive text generation in logit space. It uses Kullback-Leibler (KL) divergence geodesics for flow matching between initial and target distributions of discrete sequences. A loss function is defined to maximize the conditional likelihood of discrete tokens, and its theoretical properties are explored. Despite initial poor results on the TinyStories dataset, an empirical sampling scheme based on a pretrained denoiser is proposed, which significantly improves performance. The method is also applied to image generation tasks for comparison.

### Strengths
1) Novel theoretical approach: The use of KL-divergence geodesics for flow matching in discrete sequence modeling is a novel concept. The theoretical justification provided for the likelihood function and its relation to the flow matching velocity adds to the rigor of the method.

### Weaknesses
1) Extremely low writing quality: The writing and presentation of this article are extremely poor and unreasonable.

2) Limited dataset evaluation: The evaluation is conducted on two uncommon datasets.

### Questions
Given the low quality of presentation, I have no further questions. I hope that the authors can make full preparations and improvements before the next submission.

### Soundness
3

### Presentation
1

### Contribution
2
