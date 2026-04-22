# Benchmarking Self-Supervised Vision Transformers in Astronomy

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
This work does not describe a novel method. Instead, it aims to extend the success of self-supervised pre-training on natural images to astronomical data.
To address the lack of comprehensive benchmarks in astronomy, we first curate an unlabeled pre-training dataset and multiple datasets for typical astronomical tasks. 
Through extensive experiments, we demonstrate that our pre-training scheme has the following advantages.
Representation transferability: pre-training followed by fine-tuning can not only significantly boost performance but also reduce training epochs compared to from-scratch training on downstream tasks (e.g., improve 12\% accuracy and reduce 83\% epochs in galaxy classification), mirroring trends in natural image domains.
Cross-instrument generalization: our pre-trained model generalizes across telescope instruments and outperforms domain models.
Domain-specific pre-training data value: In-domain pre-training data further improves model performance and surpasses the results trained on general datasets such as ImageNet and other domain datasets. 
Furthermore, we explore Vision Transformers (ViTs) scaling in astronomy via parameter and data variation to offer insights and experiences for  vision foundation model development in astronomy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the effectiveness of self-supervised training, in particular, the Masked Auto-Encoder (MAE) framework, on astronomical data.

The paper points out several limitations of existing works on the same topics such as lack of diversity in training data, lack of astronomy-specific optimization in the training pipeline, and ad-hoc evaluations and metrics. The paper makes several contributions to address these issues:
- Building a more diverse dataset consisting of a 1.5M subset of the existing Astro-76M dataset and a 0.5M set of more complex content.
- Optimize the training pipeline with optimized designs for astronomical data.
- Standardizing benchmarks by introducing `galaxy-desi` dataset for galaxy morphology classification and `neuralens-desi` dataset for lensing detection.

The paper shows that self-supervised pre-training is effective for astronomical data. increasing significantly the performance on benchmarks compared to the baselines that are trained from scratch. It also shows that in-domain pre-training brings more benefits than  pre-training on natural images, and the resulted model, C-MAE, generalizes well to cross-telescope data.

### Strengths
Although self-supervised learning (SSL) has been shown effective on many cases, its benefits are not clear on visual domains that are far from natural images. The paper's effort to confirm the benefits of SSL on astronomical data is significant and could potentially encourage more research in this area.

The paper is in general well written and organized, despite some inclarities that will be specified in "Weaknesses". Many experimental results are provided to support different claims in the paper. In the supplemental, the paper also lists some design choices that are ineffective which, I think, as as important as showing choices that work.

I appreciate the paper's focus on data curation and the importance of pre-training data to the model's performance. Indeed, the pre-training data introduced in the paper seems to bring substantial gains compared to the existing Astro-76M dataset. Finally, the paper also highlights astronomy-specific design choices in the model which bring further performance gain.

### Weaknesses
1. Given that the introduction of the new pre-training and evaluation data is probably the most important contributiion of the paper, I think that  the space dedicated to describing the different datasets is somewhat limited. In particular:
- The paper mentions that the 1.5M dataset is a subset of Astro-76M but did not discuss whether it is sampled uniformly from Astro-76M or there is a change in distribution of the images compared to Astro-76M. In their paper, Stein et al. provided subtaintial details about the data pipeline. I expect the paper to include a description with a similar level of details.
- The paper adds a 0.5M dataset of "background" images  but does not specify they are obtained from which galaxies and whether galaxies figured in this 0.5M set overlap with those depicted in the 1.5M set.
- Similarly, I wonder if there is any overlap between the galaxy-desi dataset and the two datasets mentioned above.
- The paper introduces 2 new benchmarks `galaxy-desi` and `neuralens-desi` but little information about their quality is provided. In particular, is there any control on the correctness of the labels. On `neuralens-desi`, obtaining bounding boxes with opencv and labelme 
is quite adhoc.

2. Though interesting, the observations in the paper are hardly surprising.  The effectiveness of SSL on non-natural images have been observed in medical images, satellite images, etc. The fact that SSL pre-training helps reducing the number of labelled data  is well-known.

3. There are some missing ablation studies such as comparing DESI-2M to a dataset consisting of only `background` images, a breakdown on the contribution of the pre-training data and the astronomy-specific optimization.

### Questions
1. Could the authors provide more information about the datasets as specified in `Weaknesses`?
2. Could the authors provide additional ablation studies mentioned in `Weaknesses`?
3. Could the authors provide more insight as to why introducing 0.5M `background` images improves the results? The explanation about data diversity is a bit vague. I wonder if the 0.5M set correlates well with the benchmarks? Could adding even more `backgroung` images further help? Is the 1.5M part necessary at all?
4. I also like to know the authors' opinions on how to further pushing the use of SSL pre-training in astronomical data.

### Soundness
3

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
5

### Summary
This paper aims to adapt and benchmark self-supervised ViTs for astronomical imagery and establish the first large-scale foundation model and evaluation framework for this domain. They curate an unlabeled dataset of 2M images for pre-training, called DESI-2M, that is filtered from the Astro-76M dataset. The authors leverage this data to develop a pre-training strategy for masked auto-encoders, referred to as C-MAE. They perform fine-tuning and evaluation of C-MAE on three tasks: strong gravitational lensing detection (a localization task), galaxy classification, and redshift estimation (a regression task). The proposed C-MAE consistently improves performance on all of these tasks compared to from scratch fine-tuning and fine-tuning from ImageNet pre-trained models.

### Strengths
The problem is interesting and under explored. To the reviewer, who is not an expert in astronomy, a foundation model in this domain could be impactful given the large amount of objects in space that need to be studied. The massive amounts of unlabeled data coming from space instruments, such as telescopes, motivates the use of self-supervised pre-training

The paper proposes a comprehensive benchmark with standard datasets and evaluation protocols for pre-training and transfer in astronomy

The authors perform thorough ablations on their proposed pre-training strategy and demonstrate its effectiveness across three tasks

The paper is well written and the ideas are conveyed clearly

### Weaknesses
The primary weakness lies in the limited novelty of the work for a representation learning conference. Most of the experiments are identical to ones performed in the MAE paper [1], and the primary insight from this work is that domain-specific self-supervised pre-training is more effective than natural image pre-training in the astronomy domain. This insight is consistent with many past works [2, 3, 4] exploring self-supervised pre-training for novel domains
* The novelty does not have to come from the method, it can even come from the data filtering (e.g., [5]). For example, the authors mention limitations of existing large-scale astronomy datasets on Lines 162-171. Is it possible to quantify these limitations and show that they negatively affect the pre-training? Or that they are not necessary for pre-training?

As mentioned above, the authors mention limitations of existing large-scale astronomy datasets but do not quantify these limitations or explain them in detail. It is unclear how the proposed data filtering strategy mitigates these limitations in Astro-76M
* It is also unclear if the proposed data filtering is helping, as no experiment is done with C-MAE trained on Astro-76M (or a random subset to match the size of DESI-2M)

The paper mentions astronomy-specific optimizations of the MAE model, but it is unclear why the proposed optimizations are astronomy specific and not dataset specific as they are only evaluated empirically using the proposed pre-training data and benchmarks. The exploration of why these optimizations help in the astronomy domain is shallow and should be expanded

Some technical details were unclear to the reviewer. On Lines 48-52, it is mentioned that models are trained on three-channel inputs and fine-tuned on five-channel inputs, but how is the architecture adjusted to process these additional channels? The standard MAE uses a convolutional layer to “patchize” each image into tokens, where the convolution kernel weights are fixed for three-channel inputs. The kernel dimensions would no longer match in the fine-tuning setting where there are five-channel inputs.

Minor formatting comments:
* The headers in the tables should be bolded
* Figure 2 contains both a Figure and a Table

[1] He et al., Masked Autoencoders Are Scalable Vision Learners, CVPR 2022

[2] Azizi et al., Big Self-Supervised Models Advance Medical Image Classification, ICCV 2021

[3] Reed et al., Self-Supervised Pretraining Improves Self-Supervised Pretraining, WACV 2022

[4] Kang et al., Benchmarking Self-Supervised Learning on Diverse Pathology Datasets, CVPR 2023

[5] Suorong Yang et al., A CLIP-Powered Framework for Robust and Generalizable Data Selection. ICLR 2025

### Questions
A (potentially naive) question: what is the practical impact of a foundational astronomy model in the astronomy community? Highlighting this in the introduction would be helpful to convey the potential impact of this work

(The questions below correspond to the weaknesses listed above)
What new insight or contribution does this work provide beyond demonstrating that domain-specific pre-training outperforms natural-image pre-training?

Can the authors quantitatively demonstrate how the limitations of existing astronomy datasets (e.g., Astro-76M) affect pre-training performance?

Can the authors show direct experimental evidence that their data filtering strategy improves representation quality or downstream results?

What makes the proposed model modifications astronomy-specific rather than general dataset tuning choices?

How is the MAE patch-embedding layer modified to handle five-channel inputs during fine-tuning?

### Soundness
3

### Presentation
3

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
Presents a dataset and self-supervised pre-training strategy for learning strong representations for typical astronomical data tasks. The proposed data and approach is shown to lead to better performance, faster convergence, cross-instrument generalization. The paper also conducts comprehensive experiments on around scaling and in-domain pretraining for astronomy tasks.

### Strengths
– The paper studies a challenging, interesting, and  under-studied problem – of learning representations for astronomical data – and does so in a principled way. Its experiments are well-designed and likely to be of value to researchers in the community. The paper also includes code to reproduce its findings.

– The paper is extremely well-written, and clearly lays out each claim and its supporting evidence.

– The paper includes a comprehensive set of experiments, across model backbones, tasks (morphology classification, lensing detection, cross-telescope transfer), settings (few-shot v/s full finetuning), and data/parameter scaling

### Weaknesses
– The paper’s study would be more complete with the inclusion of pretraining strategies other than C-MAE to ascertain which SSL strategies work better/worse for this domain. Have the authors attempted benchmarking more lightweight contrastive learning approaches like SimSiam [A]? 

[A] Chen et al., Exploring Simple Siamese Representation Learning, CVPR 2021

### Questions
Kindle address the weakness listed above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper considers training models for astronomical data. It does not claim to introduce a new method. Instead, there is a mix of claimed contributions:
- a new pre-train dataset that sample a subset from the existing Astro-76M  dataset
- some benchmark
- some pretraining scheme called C-MAE (mostly inspired by MAE)

### Strengths
The paper aims at addressing a need for standardization and comparison in deep learning astronomy by curating novel and diverse benchmark datasets.

### Weaknesses
In the first place, I was confused by the positioning of the paper. What is claimed to be to contribution is not very clear overall? The curation of the dataset? The benchmarking itself? The MAE-style variant for training? Depending on when we look in the paper, we have different. claims. Overall, it sounds a bit like a mix of all, with most of the paper dedicated to claiming superior performance for C-MAE over baselines that are questionable (See below). 

The paper is largely focused on what works rather than why it works. There is a lack of depth in analyzing why certain SSP methods (e.g., specific masking strategies in MAE or contrastive loss variants like SimCLR) succeed or fail in the astronomical context compared to natural images.

In multiple recent papers, it was shown that Dino-v2 (And now v3) is the best model for transferring to new kind of data (biological, satellite, etc)., if the model is large enough. The paper only shows a small table in Appendix with the Vit-B model, which is clearly not where Dino is best at. I have multiple concerns:
* I see no good reason for choosing Vit-B for Dino-v2, especially since the paper considers Vit-L and Vit-H. Dino would be MUCH better. 
* One can still fine-tuning a dino pretrained model, which in my humble opinion would evidence that the proposed pretraining scheme has no interest.

### Questions
please address the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1
