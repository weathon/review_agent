## Human Reviewer 1

### Summary
This paper provides a comprehensive study to analyze the answers supplied by the VLMs. Specifically, It compares the performances of CLIP and LlaVa-1.5-7B in the What’s Up and MMVP benchmarks. These benchmarks ask questions about a pair of images that contain the same objects and background but in different positions. This paper shows that the LlaVa-1.5-7B can perform better than CLIP in these benchmarks even when LlaVa uses CLIP as a visual encoder, and the average cosine similarity of the CLIP embedding of the image pair is greater than 0.95. Moreover, it provides ablation studies to explain this behavior.

### Strengths
This paper provides some interesting insights to show that the metric commonly used to measure the embedding similarity (Cosine Similarity) does not depict all aspects of vector pairs. Therefore, it suggested a complementary metric, Spearman’s rank correlation coefficient. However, table 1 only provides the average Cosine Similarity.

### Weaknesses
The paper is challenging to follow, primarily due to the absence of a clear statement of its main contributions in the Introduction. Its content closely parallels the CVPR24 paper, "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs," raising concerns about the originality of this work. The CVPR24 paper highlights that Visual Language Models (VLMs), often relying on CLIP as the visual encoder, struggle with recognizing fine-grained details, such as object locations. It introduces the MMVP benchmark to evaluate these limitations comprehensively. I encourage the authors to clarify how their contributions provide novel insights beyond this existing research.

### Questions
I recommend including Spearman's rank correlation coefficient in Table 1 to enhance the analysis. Additionally, a more comprehensive study would be valuable. For example, could the authors provide Spearman's rank correlation coefficient and cosine similarity for the questions with the highest- and lowest-accurate answers?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper challenges the prevailing belief that Vision-Language Models' (VLMs) failures in visual reasoning are primarily due to CLIP image encoder's "erroneous agreements" (where distinct images have high cosine similarity). Using LLaVA-1.5-7B as an example, they demonstrate that MLLMs can successfully extract distinct information from similar image embeddings, achieving high accuracy on tasks where CLIP performs poorly. This suggests that the limitation lies not in the image embeddings themselves, but in how effectively models extract and utilize the encoded information.

### Strengths
Provides compelling empirical evidence through controlled experiments across multiple benchmarks.
Challenges and refines an important assumption in the field about VLM limitations.
Demonstrates that existing architectures might be more capable than previously thought, just requiring better utilization strategies.

### Weaknesses
The paper's scope might be too focused on LLaVA-1.5 as the primary example, potentially limiting the generalizability of findings
While the paper shows that information can be extracted from similar embeddings, it doesn't fully tackle why LLaVA-1.5 is able to do this.

### Questions
How do these findings generalize to other MLLMs beyond LLaVA-1.5?
What specific mechanisms allow MLLMs to extract distinct information from seemingly similar embeddings?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper examines the performance of LLaVA-1.5-7B on visual reasoning tasks, specifically WhatsUp and MMVP, and concludes that its suboptimal performance is not due to CLIP's visual features. While CLIP visual features effectively capture semantic similarities, they occasionally misinterpret spatial differences in object placement (e.g., "mug on the left" vs. "mug on the right"), which results in high cosine similarity (over 0.95) despite subtle image differences—referred to as "erroneous agreements." The authors show that CLIP’s visual features are accurate; instead, they attribute the performance issues to LLaVA not making effective use of these features. They further demonstrate that poor alignment between visual and textual inputs, not the visual features themselves, explains the bad performance in CLIP models for these tasks and datasets. Unlike CLIP, LLaVA does not exhibit this alignment problem, and this is shown quantitatively. Finally, the authors try better decoding strategies in Llava like M3ID such that the decoding better makes use of the visual features. They also show that  multiple image inputs works better to highlight the difference in images. They also explore performance gaps related to evaluation methods, training data, and the text encoder.

### Strengths
- This paper delivers a valuable message to the community by advocating for enhancing Multimodal LLMs and keeping the image encoder fixed. Previous research suggested that the image encoder introduced issues by producing "erroneous agreements" (similar embeddings for semantically similar but visually distinct images). However, this paper counters that claim, attributing the problem instead to the the model not utilizing these visual features effectively. 

- Interesting observation of better decoding algorithms and methods for evaluating specific tasks.

### Weaknesses
- There is an incoherent story. The abstract initially suggests that LLaVA performs well on reasoning tasks and achieves high accuracy, yet later the paper claims LLaVA performs poorly on MMVP, contradicting the initial statement. They also mention that LLava is able to extract the correct information from the visual features, and that it does not face issues (L186, and demo image). Only later is it clarified that LLaVA performs well on WhatsUp but not on MMVP. In general, I feel there is an unclear and confusing story. 

- WhatUp, MMVP, COCO-spatial and GQA-spatial are not really well-known datasets and publicly-agreed on to measure reasoning. I actually came to know them after reading this paper. Measuring reasoning on MMLMs are usually not done on these datasets. These datasets are not enough to reflect model reasoning and to come up with general conclusions about LLava or MMLMs in general. The authors don’t show ablation and analysis results using their ablation strategies, on important reasoning tasks such as VQA, GQA, OK-VQA, VCR and others (specifically, those that LLava reports on). I feel the scope, task and datasets are not enough to reach the standard required for ICLR.

### Questions
In general the second weakness is the biggest to me. I would like to hear what the authors say on this?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 4

### Summary
Previous works have argued that the poor performance of VLMs on simple visual reasoning tasks is due to their dependence on CLIP encoder. They show that CLIP can encode two visually different images with high cosine similarity (called erroneous agreement) and argue that many VLMs fail due because they use CLIP as their vision encoder.

In this paper the authors show that with better extraction and utilization methods, clip encoder can still be used for downstream tasks of visual reasoning. They show experiments with LLaVA-1.5 and show that it performs good on benchmarks despite using CLIP as its vision encoder.

### Strengths
1. Important analysis shown in section 4 (Investigating the performance gap)- this section answers the questions related to training data, language model and evaluation method. This analysis is important to make the claim that visual information extraction is the key factor in determining the performance gap on downstream tasks. And these other factors (eval method, language encoder, training data) are not contributing much to the improved performance.

2. Detailed benchmarking of the models on different datasets and good ablation studies.

3. They show, using a different decoding method, that even with a fixed pre-trained image encoder if we try to 'force' VLMs to attend to visual features while decoding (and not just relying on language priors), we can perform good on downstream visual reasoning tasks.  Although they used a previously proposed decoding strategy M3ID  (Favero et al., 2024).

### Weaknesses
1. The authors show that the visual feature extraction technique in LLaVA (a two layer MLP) is an important step in distinguishing between two erroneous images. But they do not provide an convincing argument on why is it an important step. An analysis on "why just adding a 2-layer MLP on top of pre-trained CLIP makes it so much better?" would have been an amazing addition to the paper. 

2. On Spearman's rank correlation (also asked in the questions): Since CLIP is trained using loss based on cosine similarity, I think using Spearman's rank correlation to show that two embeddings are "fully opposed" is not correct. For example, consider the example given on LN 232-233. Although the ranks of the dims are reversed giving  ρ = −1, their absolute values are pretty close. And if we assume (in an ideal world) them to be separable features, for example the embeddings could be of dog images and the features are 'ear-length' , 'fur color', 'nose-shape', both the embeddings will still show two very similar looking dogs (and not 'fully opposite') even though the embedding might have ρ = −1.

### Questions
Would a high negative Spearman's rank correlation show that the embeddings are quite different? 

LN 232-236 says: "While SC (fv(v1), fv(v2)) > 0.989, Spearman’s rank correlation coefficient can tell their sharp difference: ρ = −1, showing that they are fully opposed in this sense. Therefore, the difference in visual inputs might still be extracted through other means when erroneous agreements occur"

How does ρ = −1 show that the embeddings are 'fully opposed'? If the authors could show this or cite a paper that shows this, that would be great.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4