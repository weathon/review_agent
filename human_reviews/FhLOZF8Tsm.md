# Towards Efficient and Scalable Implementation of Differentially Private Deep Learning

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Differentially private stochastic gradient descent (DP-SGD) is the standard algorithm for training machine learning models under differential privacy (DP). The most common DP-SGD privacy accountants rely on Poisson subsampling for ensuring the theoretical DP guarantees. Implementing computationally efficient DP-SGD with Poisson subsampling is not trivial, which leads to many implementations ignoring this requirement. We conduct a comprehensive empirical study to quantify the computational cost of training deep learning models under DP given the requirement of Poisson subsampling, by re-implementing efficient methods using Poisson subsampling and benchmarking them. We find that using the naive implementation DP-SGD with Opacus in PyTorch has between 2.6 and 8 times lower throughput of processed training examples per second than SGD. However, efficient gradient clipping implementations with e.g. Ghost Clipping can roughly halve this cost. We propose alternative computationally efficient ways of implementing DP-SGD with JAX that are using Poisson subsampling and achieve only around 1.2 times lower throughput than SGD based on PyTorch. We highlight important implementation considerations with JAX. Finally, we study the scaling behaviour using up to 80 GPUs and find that DP-SGD scales better than SGD.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper investigates the computational cost and efficiency of DP-SGD training, emphasizing that many current implementations forego proper Poisson subsampling, which could result in weaker privacy guarantees than intended. The authors propose an alternative, more efficient DP-SGD implementation in JAX with Poisson subsampling, achieving throughput only 1.2 times lower than standard PyTorch-based SGD. Besides this, the author also attempt to reveal the reason behind the high training cost and the scaling behavior using a larger clusters of hardware resources.

### Strengths
1. The authors re-implement DP-SGD in a more efficient way and the source code is shared in the supplementary materials.
2. The authors discuss various types of gradient clippings.
3. The authors investigated the reason of high training cost by the NVIDIA Nsight System.

### Weaknesses
1. The paper is written unprofessionally, unstructured and very hard to follow.
2. The motivation is unclear. Based on the abstract and introduction, the paper starts from the issue of weaken privacy guarantees in current improper implementations of Poisson subsampling. However, the rest of the paper discuss the computation cost of DP-SGD, which is unrelated. After reading the whole manuscript, the reviewer still does not know how, whether the authors address this problem or not.
3. The background of implementations of DP-SGD is poor written. In Sec. 2.1, the paper quickly jumps into Virtual batching and discuss some detailed batch sizes (25000, 300, etc), which seems to be more appropriate in Sec. 3. How are paragraphs of Sec. 2.1 relate to each others? Are they parallel or progressive? The reviewer find them unstructured. Virtual Batching, Poisson subsampling are high-level algorithm idea, and Opacus is a software package. The reviewer does not think they are appropriate to be listed together. The rest of the paper has the same issue, where the authors list a bunch of unstructured items in a section, which are very hard to follow.
4. The contributions are poorly summarized in the introduction section. The reviewer understands that the authors have investigated on various aspects of the cost, but still finds unclear what the main points are here.
5. The novelty remains a major drawback of this paper. First, the author acknowledged that their current implementation is incomplete and does not support all network types yet. Second, using JAX or lower precision for speed up training is very common and too naive, even in other generic non-private training. Third, the paper raises some new questions from their empirical findings, while the authors do not address them. Based on these 3 points, the reviewer thinks the paper is incomplete.

### Questions
1. The authors need to present their proposed implementation and explain how, whether it insures correct privacy guarantee of Poisson subsampling,
2. How algorithms are optimized in Sec. 5.2 to decrease the cost? From the writing, it seems that the way to "optimize" is just using a better GPU (A100 instead of V100).

### Soundness
3

### Presentation
1

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
The authors present an extensive empirical study of how efficient current proper (i.e., with poisson sampling) DP implementations are, and then provide an algorithm that leverages JAX better to improve the throughput of DP training. I believe the empirical study and modifications are interesting and impactful for the field; scalable DP training is an important technical challenge for private learning, and they provide detailed understanding for current computational issues with DP implementations and ways of bypassing some of these. Furthermore, I found the paper to be generally well written.

### Strengths
1) Extensive empirical study
2) Proposed algorithm improves previous approaches
3) Well-written: I expect this will help with future work on the topic

### Weaknesses
1) One could argue the algorithmic contributions of the paper are limited, being just modifications to improve sampling. However, I believe it required a detailed study to realize that Jax could help if one tuned away the reocurring compilation times, paying a modest increase in gradient evaluations. In the questions I also ask the authors to surface more details on the algorithm they propose to the main text, which may help convey the contributions more.

### Questions
1) In the caption for figure 2, is there a typo when saying lower throughput is better?
2) In lines 399 to 400, could you elaborate on why larger models are too expensive (to have benefit from lower precision)?
3) In the JAX paragraph of section 2.2, could you elaborate on how masking achieves poisson sampling? I found the algorithm in the appendix, and I think describing it briefly and pointing to the algorithm in the appendix would help with clarity on this contribution: e.g.,  something along the lines "Specifically, we first sample a batch size from the distribution of batch sizes, and with this sample a number of random physical batches to have more samples: one can then mask out some datapoints to have effectively sampled with poisson."

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper provide a implementation of DP-SGD algorithms with Poisson sampling, and benchmarked a few clipping algorithms in Opacus and jax. The benchmark shows jax is typically faster and ghost clipping algorithms have comparable (in the sense that they are not 10x slower) throughput with non-dp conterparts.

### Strengths
The paper provided efficient implementations of DP-SGD with Poisson sampling which would be useful to the community. Also the paper benchmarked the implementation with Opacus and jax with informative results on what clipping algorithms one should choose.

### Weaknesses
My major concern is the contribution seems not significant enough, the benchmarks are nice but are not comprehensive enough if viewing it as a benchmark paper. For example, torch compile is not considered in the paper which is one of the most useful emerging acceleration technique in pytorch newer versions and . Also when comparing with Opacus, the backend used by Opacus is not mentioned, Opacus have a few backend methods for efficient clipping, like functorch based implementation and backward hook used approach. More details at the link: https://github.com/pytorch/opacus/tree/main/opacus/grad_sample. 

The Poisson sampling implementation is useful I am not sure it is significant enough given the existing tools like Opacus and jax already have pretty good support to non-Poisson sampling, the Poisson sampling should be just changing the data batching parts while other parts are utilizing the existing tools. In addition, the paper did not benchmark the performance difference between non-Poisson sampling and Poisson sampling which is nice to have since Poisson sampling is the main focus of the paper.

### Questions
1. Could the authors provide some explanations on the novelty or difficulty in implementing the Poisson sampling?
2. Opacus backend does have optimized implementation of gradient clipping and ghost clipping is now supported as well, and torch compile should be able to improve speed and memory consumption in some cases. I wonder why authors did not compare with the optimizations that can be done in pytorch when comparing with jax.

### Soundness
2

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
3

### Summary
The paper discusses which aspects influence efficiency and scalability when implementing differentially private stochastic gradient descent (DP-SGD).
The authors perform an empirical study to identify which operations hinder computational efficiency and propose an efficient JAX implementation. 
By measuring throughput (i.e., how many samples are processed per unit of time during training), the authors compare their implementation with existing library implementations of DP-SGD, as well as with non-DP SGD implementations.
With their experiments, the authors find that they can almost match the throughput of a PyTorch implementation of SGD.

### Strengths
The paper tackles an important problem in privacy-preserving ML, as DP-SGD is often regarded as the most commonly used approach to adopt DP in practice. 
The authors identify the main sources of inefficiency, namely Poisson subsampling, gradient clipping, and the per-sample gradient computation that characterize DP-SGD.
With respect to Poisson subsampling, the authors note that DP-SGD libraries which do not implement it, may have weaker privacy guarantees than claimed.
With respect to gradient clipping, the authors empirically compare standard per-example clipping with more recent optimizations, i.e., ghost clipping and book keeping.
The empirical evaluation is extensive and focuses on vision classifiers (ViT Base and ResNet).
While the authors do not introduce novel techniques, their analysis of existing strategies (and their re-implementation) promises to be very useful in practical settings.

### Weaknesses
**Summary of "lessons learnt"**

While the empirical evaluation proposed by the authors can be very useful to identify which strategies are more effective when implementing DP-SGD efficiently, the paper is (in this sense) difficult to parse.
To expand on this, I find that absence of a clear "lessons learnt" section/table/summary/discussion can negatively affect the impact of the contribution.
By the end of the paper it is not easy to remember which techniques have the biggest impact on efficiency.
The conclusions section, for instance, simply reminds the reader that feasible speed-ups include "efficient implementations of DP-SGD, JAX, efficient implementation of Poisson sampling, lower precision using TF32".
From a practitioner's perspective, an informative summary is therefore missing and would significantly increase the value of the contribution, in my opinion.

**Structure**

With reference to my previous comment, the structure of the paper does not help readability.
The main result (figure 1) is presented in the introduction, where most of the information necessary to parse the visualization has not been presented yet.
The Experiment Overview in Section 3 does not effectively present an overview of the experiments, as it is not clear which libraries will be compared, with which clipping strategies, etc.


**Throughput as the only quality measure**

While I do agree that throughput is an intuitive measure to quantify efficiency, I think that some more extensive discussion on how the various strategies to increase efficiency affect model utility is missing. 
As there are many moving parts contributing to the final result (i.e., a model trained with DP-SGD), it would in my opinion be important to discuss how to improve throughput while maintaining similar model utility.
To be more specific, it is mentioned that Poisson subsampling is sometimes not implemented, as efficient implementations are difficult.
I can expect this to impact not only efficiency, but also model utility (in terms of, e.g., classification accuracy) and privacy.
In this sense, a comparison of the throughput between your implementation which uses Poisson sampling and an implementation which foregoes Poisson sampling is, I think, unfair.
On the one hand, your implementation may results to be "less efficient" in this case, as it may process instance more slowly.
On the other hand it is unclear what the impact on privacy and utility is.
While you do specify in section 2.1 that all the experiments are "seeded and have the same logical batch size", it is not clear whether all experiments obtain models which have comparable utility and privacy.
Incidentally, I would expect this detail on how seeding is used to be part of the experiments section, rather than the background.

On a similar note, your claim that (see, e.g., contribution 2 in section 1) DP-SGD "costs between 2.6 and 3.2 times more that non-private training" may be misunderstood.
It may be unclear to the reather whether this suggest that the overall cost of the training procedure has a factor ~3 difference (e.g., until model convergence), or whether this refers to, e.g., a single epoch.
In any case, it is necessary to explicitly discuss how the performance of a model trained with SGD and with the various DP-SGD approaches compare.


**Importance of Poisson subsampling and of lower precision**

You mention that foregoing a proper implementation of Poisson subsampling, as well as using lower precision, can both improve efficiency at the possible cost of privacy (and utility?).
I think that more insights on this could be within the scope of your investigation.
You mention that TF32 can increase throughput, but if this happens at a huge privacy cost, a practitioner would likely not adopt this strategy.
While you mention that this may be a topic for further research, I do think that some more insights on this would be important to actually understand the impact of TF32 on private model training, on a similar line of thought as discussed in the previous section.
I have similar questions with respect to Poisson subsampling, where the impact on privacy is not clearly quantified.

### Questions
In addition to the points discussed in the weaknesses section, I have these further questions.

* In your experiments you use a privacy budget of $\epsilon=8$, which is arguably on the upper end of what is usually understood to be a "private model". How does this choice impact the outcome of your experiments?
* In Table A3, you compare non-private models with optimal DP hyperparameters against DP models. Why? How is this a useful or fair comparison?
* What makes it difficult to implement Poisson subsampling? 
* While you say that Poisson subsampling is often not properly implemented, I understand that all the methods you compare against do in fact implement it. Is this the case? And if yes, why do you not compare against methods which do not implement Poisson subsampling?

### Soundness
3

### Presentation
1

### Contribution
3
