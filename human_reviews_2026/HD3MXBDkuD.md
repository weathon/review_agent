# Concept-Aware Batch Sampling Improves Language-Image Pretraining

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
What data should a CLIP model see? Many data curation efforts aiming to answer
this question center on the quality of a dataset. However, recent work has shown that
while admitting impressive performance benefits, none of these curation methods
are concept-centric, leading to them inheriting the biased properties of web-scale
data distributions. In this work, we go beyond such concept-agnostic methods and
advocate a more flexible online concept-based curation approach. To enable this,
our first contribution is DATACONCEPT, a collection of 128M web-crawled image-
text pairs annotated with fine-grained details about their concept composition.
Building on DATACONCEPT, we fill another critical gap in the literature: the lack of
a competitive, open-source alternative to highly performant batch sampling methods
for Language-Image Pretraining. Specifically, we introduce Concept-Aware Batch
Sampling (CABS), a simple yet effective batch-sampling algorithm that distills
batches with the broadest set of available concepts. Through rigorous evaluation on
a broad suite of 28 benchmarks, we demonstrate that CABS significantly benefits
Language-Image Pretraining (LIP) and yields highly performant models on long-
tailed evaluations (up to +2.4 p.p. on Let-it-Wag!), while enabling practitioners to
define custom concept distributions that optimize for specific downstream tasks.
Importantly, with only one hyperparameter tuned for a single (backbone, eval)
combination only, CABS shows full compatibility with both CLIP and SigLIP
models. Both DATACONCEPT and the source code for CABS will be released

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes: (1) a pipeline that automatically generates fine-grained, concept-focused image captions using state-of-the-art tagging, grounding, and VLM models; and (2) evidence that pretraining image-language model with batches whose concepts are uniformly distributed outperforms pretraining the same model using concept-biased batches. Combining these contributions yields considerable accuracy gains across diverse image–language pretraining settings, including different loss functions and base models.

### Strengths
The paper shows that (1) relabeling captions to emphasize image concepts (DataConcept) and (2) pretraining with concept-balanced batches (CAPS-DM) consistently improve zero-shot classification across diverse settings. Zero-shot retrieval accuracy also increases with the optimal batching strategy (CAPS-FM). These conclusions are supported by extensive experiments.

### Weaknesses
- The method performs well on zero-shot classification but degrades zero-shot retrieval accuracy under the default CAPS-DM. Although the CAPS-FM variant improves retrieval, relying on different setups for different applications weakens the contribution, as a single pretrained model is generally expected to work across tasks. If the model underperforms on either classification or retrieval, it may also struggle on downstream tasks such as detection and segmentation, which undermines the promise of foundation models.

- Developing a data relabeling pipeline for pretraining are already well studied (e.g., https://arxiv.org/pdf/2311.06242 and references therein). To establish novelty, more comprehensive qualitative and quantitative comparisons are needed; comparisons limited to naive baselines (IID) and a hard-negative mining variants are insufficient.

### Questions
- If CAPS-FM is applied for zero-shot classification, is that better than the baseline IID? I.E. Can the CAPS-FM be more general than CAPS-DM so that CAPS-FM works in both the applications (zero-shot classification and retrieval)?

- Have author tried CAPS with the existing dataset without using DataConcept? This is important to understand what is the major contribution of the improvement. Is it because of DataConcept or CAPS?

- What is the baseline performance on the benchmark even without DataConcept?

- The ImageNet zeroshot accuracy in Figure 1 is generally too low. If author is using variant of ImageNet zeroshot benchmark, please clearly specify them in detail in the Figure 1.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the role of incorporating concept-level information during large-scale language–image pretraining, which is underexplored in the literature. To this end, the paper introduces DATACONCEPT, a large-scale, fully annotated pretraining dataset, which augments samples with fine-grained concept annotations and concept-driven synthetic captions. Besides, this paper proposes a flexible framework, CABS, for online, concept-aware batch sampling for LIP. Empirical experiments demonstrate the benefits of CABS over IID and other two batch-sampling baselines.

### Strengths
1.	The constructed concept-aware pretraining dataset with rewritten context-aware captions seems promising and useful.

2.	The experiment in table1 shows a clear advantage of concept-aware re-captions in LIP.

### Weaknesses
1.	While the curated concept-aware pretraining dataset is meaningful to the community, it seems the proposed CABS doesn’t work well with the dataset. Different from other sampling strategies that may achieve consistent improvement across classification and retrieval tasks, CABS may perform well on classification but worse on retrieval tasks. Though the authors proposed an alternative CABS-FM to improve performance on retrieval tasks, it makes the total design complicated since a hard choice needs to be made and the trained model can't perform different tasks effectively, which means that it loses the advantage of pretrained models to generalize well across tasks..

### Questions
1.	Although the authors claim that Evans et al. (2024a) and Udandarao et al. (2025) didn’t release their code, is it possible to reproduce them since their methods show strong performance and the paper only has two baselines?

2.	It’s better to include an algorithm (pseudo code) to show how CABS-DM processes samples during training. Only natural language description makes it difficult to follow and understand the procedure clearly.

3.	What are the guidelines for choosing the hyperparameter f? Is it sensitive?

4.	Please explain more about $f_c$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper annotates 128M image-text pairs from DataComp XLarge, and proposes Concept-Aware Batch Sampling (CABS), a diversity-driven strategy that e.g. samples mini-batches based on the diversity of their constituent concepts instead of random sampling. By balancing concept coherence and intra-batch diversity, CABS yields improved performance across vision and language tasks.

### Strengths
- Achieves better zero-shot performance than random sampling. Another heuristic variant that samples examples with more concepts yields a better performance for retrieval.

- Provides empirical validation on benchmarks.

### Weaknesses
I don't see the novelty or new insights provided by this paper. The idea that balanced mini-batches improve the convergence and performance on smaller groups of data is well-known. This idea has been used before in many different domains, including federated learning, data selection, etc. From optimization perspective, the reason is that balanced mini-batches have smaller gradient variance which yield faster convergence, which is theoretically analyzed and shown by several existing papers in the literature (this is a relatively old concept). Besides, upsampling underrepresented groups is obviously beneficial. The main idea of the paper is to annotate the concepts in training example and use them to sample balanced mini-batches. While this is a good usage in a production pipeline, I don't see any new "scientific" insight. If the main contribution is the annotations, the paper is more suitable for the dataset and benchmark track.

### Questions
What's the new scientific insight (finding, method, analysis, etc) from this paper, in authors' opinion?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper advocates for concept-aware data curation for contrastive language-image pre-training (CLIP) and first proposes a dataset DataConcept based on a subset of DataComp, and consists of 128M image-text pairs, where each image annotated with fine-grained concepts, bounding boxes, and concept-aware synthetic captions; Then, the method propose a Concept-Aware Batch Sampling strategy to improve the training effectiveness. Different from MetaCLIP, it seems that this method do not need to build a balanced dataset offline but adaptively adjust the sampling throughout the training to maintain a balanced distribution seen by the model. The experimental evaluation shows untrivial gain on classification and long-tailed benchmarks.

### Strengths
- A new data curation mechanism with online method to adjust the data distribution seen by the model to improve the training effectiveness.
- The motivation to change existing random sampling is appreciated.
- The evaluation is comprehensive.

### Weaknesses
- The definition of concept is critical for the method development. Then, the discussion on why the current concept bank definition is optimal is needed. As the author mentioned MetaCLIP many times, I am curious how the concept bank different from the metadata used in MetaCLIP (in MetaCLIP, the balanced distribution according to metadata is one critical standard for MetaCLIP dataset construction. 
- Following, when the concepts bank contains erroneous or missed concepts, how your method can robustly expand or update it in an online manner.
- Performance trade-off: By comparing performance in Table 1 and Fig. 5, the CABS-DM helps classification but hurts retrieval, while the CABS-FM favors retrieval only. Then, I am curious whether these two mechanism variants can be combined, or whether it is always conflicting for optimizing the performane for classification & retrieval. 
- For your method, I wanna check whether the image encoder during training must be frozen or can also be updated.

### Questions
For multi-modal pre-training, the current method primarily uses text to determine the context in batch sampling, which is similar to [2]. However, whether the visual information can also be utilized (e.g., the team of MetaCLIP also propose CIT for visual pre-training).

[1 ]CIT: Curation in training for effective vision-language data (ICCV)

[2] In-context pretraining: Language modeling beyond document boundaries (ICLR)

Please see my comments in the weakness. Happy to increase the score if all of my questions & weakness can be properly addressed.

### Soundness
3

### Presentation
4

### Contribution
3
