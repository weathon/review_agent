# Universal Algorithm-Implicit Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Current meta-learning methods are constrained to narrow task distributions with fixed feature and label spaces, limiting applicability. We present TAIL, a novel algorithm-implicit meta-learner that functions across tasks with varying domains, modalities, and label configurations.
Out approach reformulates the few-shot learning problem as a sequence modeling problem. We train a non-causal transformer on sequences of data-label-pairs and and unlabeled query sample, to directly predict the label of the query sample. This causes the transformer to learn an implicit learning algorithm, which enables it to learn new concepts at test time without fine-tuning. Empirically, TAIL achieves state-of-the-art performance on standard benchmarks while generalizing to unseen domains and modalities. Unlike other meta-learning methods, it sustains strong performance on tasks with up to 20 times more classes than in training while providing orders of magnitude computational savings. Moreover, we introduce a theoretical framework for meta-learning, which allows us to formally describe important properties of meta-learning paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents TAIL (Transformer-based Algorithm-Implicit Learner), a meta-learner designed for practical universality. The key idea is to reframe few-shot learning as a sequence modeling problem, where a non-causal transformer learns an implicit learning algorithm. The authors introduce two main technical contributions to handle diverse tasks: a universal feature encoding scheme using random projections into a common latent space, and a universal label handling mechanism using a randomized global dictionary of learnable embeddings. The paper also provides a theoretical framework that distinguishes between algorithm-explicit and algorithm-implicit learning. The experimental results demonstrate that TAIL achieves state-of-the-art performance on a variety of benchmarks,

### Strengths
This paper gives a clear and  general definition and discussion about learning problems and learning to learn, which helps readers understand.

The paper presents a comprehensive set of experiments that demonstrate the effectiveness of TAIL. The method is shown to outperform strong baselines across a wide range of settings, including in-domain, cross-domain, cross-modal, and label-space extrapolation scenarios.

### Weaknesses
1. Lack of novelty and contribution. The framework, and the claimed technical contribution of a universal feature encoding scheme and a universal label handling mechanism, and most of the theoretical proofs, are almost identical with [1] [2].

2. Lack of key references. A line of works, which discuss what learning algorithms does ICL model learn, are highly related [3][4]. Especially [5] has exactly pointed out that ICL models are meta-learners with minimal bias, and learning implicit learning algorithms.

3. The reviewer could not agree with the definition of algorithm-explicit/implicit learning in section 2.3. The paper defines the distinction as if its training procedure is explicitly specified, and specify GD as explicitly specified while attention-based meta-learners are not. However, what is "explicitly specified" is not clear. For example, one could specify the training process as the forward-propagation process in transformer with certain training set and query as sequence input, in which case attention-based meta-learners are algorithm-explicit learning.

4.Lack of in-depth analysis of results and ablation study. Considering the similarity among the TAIL and [1][2], the reviewer could not understand why TAIL outperforms the others.

5. Obvious typos, even at the beginning of abstract.

[1] General-purpose in-context learning by meta-learning transformers
[2] Context-aware meta-learning, ICLR 2024
[3] Transformers Learn In-Context by Gradient Descent, ICML 2023
[4] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection, NIPS 2023
[5] Why In-Context Learning Models are Good Few-Shot Learners?, ICLR2025

### Questions
Please refer to weakness

### Soundness
3

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
5

### Summary
The paper proposes TAIL, a non-causal transformer trained on sequences of support examples and unlabeled query samples so that the transformer learns an implicit learning algorithm and predicts query labels in a single forward pass. TAIL extends previous approaches to generalize to unseen domains and modalities using an arbitrary number of classes.

### Strengths
- The extension of previous approaches such as CAML and GPICL, to multiple modalities and large number of classes is interesting and in line with the current research directions in foundation models.
- The distinction between algorithm-explicit and algorithm-implicit learning and the formal formulation of universality is useful and clearly described.
- Experimental results show the strength of the proposed approach.

### Weaknesses
- The formulation of the meta-learning problem and particularly the task definition could be improved by also providing references to survey papers (e.g., [1], [2], etc).
- The motivation behind the choice of the vision and text encoder should be clarified. An ablation experiment on different encoders can strengthen the paper, similarly to what has been done in [3].
- The claim about the computational efficiency at scale is not well supported. Additional information about the memory usage for training and inference and the training time for the different baselines is needed to support this claim.
- The claim of cross-domain generalization is not fully evaluated. It is unclear how data leakage between datasets (e.g., MetaAlbum or ImageNet used for training vs. Birds or Airplanes used for testing) is prevented, given overlapping semantic classes. A stronger assessment of cross-domain capabilities should consider completely different domains (e.g., training on ImageNet + MetaAlbum and testing on MedIMeta).
- Some experimental details are missing, such as how the training tasks are sampled (e.g., within or cross dataset), how the training dataset are shown to the model (e.g., randomized or in a sequence), and what are the hyperparameters used for training.

### Questions
- Linear Probe and ProtoHead perform similarly to TAIL (considering the standard deviation) on cross-modal generalization to unseen modalities (Tab. 4). A discussion on this is needed. Moreover, what is the benefit of using TAIL considering the much larger memory and computation requirements?
- What is the rationale behind the selection of the training dataset? Would training only on MetaAlbum be sufficient to achieve strong generalization, and what additional benefit does including ImageNet or other datasets provide?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TAIL, transformer-based algorithm-implicit learner, which adopts the model-based meta-learning methodology, and propose structure designs for unifying potentially distinct input domains and target class cardinalities.

### Strengths
The overall architecture design makes the model flexible and generalizable to various input modalities and cardinalities.

Specifically, the authors have devised random permutation mappings on top of the encoded features from the inputs, achieving the benefits of preventing the model overfitting over fixed-structured features, as well as implicitly realizing input augmentations which encourage the model robustness. Furthermore, a global learnable dictionary is employed to enable the model to adapt to test-time tasks with classes of different cardinalities.

The experiments cover both adaptation to tasks cross-domain and cross-data-modality, with the proposed method achieving the best results most of the time.

### Weaknesses
The proposed method is fundamentally the same as the model-based meta-learning methods that can be dated back in 2016, which the authors have identified under the Related Work section (with LSTMs or transformers): the few-shot training samples with labels are provided in together with the query sample as a sequence to the model, which directly predicts the label for the query. While the authors have identified short comings from prior works (e.g. not being invariant in sample ordering, generalizing to different modalities or class cardinalities), the novelties are strictly within the detailed architectural designs (which the reviewer has recognized as the strength of the paper). Therefore, the novelty of the paper is severely limited.

Within Section 2, the paper sets up theoretical foundations for meta-learning and so called algorithm-implicit and algorithm-explicit learning systems. However, while it is nice to formulate the notions such as practical universality, for the proposed method, there is lack of connection between these theoretical discussions in Section 2 and the designed approaches. The algorithm-implicit nature of the approach is easily recognized for the model-based approaches as the authors have surveyed. The reviewer questions the necessity of the theoretical discussions in Section 2, as well as the overall theoretical contribution of the paper.

Some parts of the write-ups need to be fixed or improved:
1. For the last sentence in the introduction section, "We demonstrate that algorithm-implicit approaches outperform algorithm, explicit ones for small support sets and varied tasks.", there is a punctuation typo: "algorithm, explicit" -> "algorithm-explicit".
2. For the first sentence under Section 2.2, the symbols shouldn't be the same for both the learning algorithm space and one algorithm within it. 
3. In the last equation at the bottom of pg2 (definition of the general meta-learning problem), the notion "n" is not defined until later in Definition 3.
4. The introduction of the notion "z" at the beginning of section 3 is a bit abrupt, as the definition of the input sequence encoded based on the inputs was not made clear until later in the context.

In the experiment section, there lacks ablation studies justifying the various architectural design choices proposed in the paper.

### Questions
While it is perceivable that algorithm-implicit learning system has better learning capacity, as the authors have demonstrated, the reviewer wonders if the algorithm-explicit learning system has better sample efficiency in learning (where samples here mean the tests in meta-training). Have the authors conducted experiments where the meta-training tasks are limited in quantities, where the inductive bias in the algorithm-explicit learning could be the advantage?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces TAIL (Transformer-based Algorithm-Implicit Learner), a novel meta-learning framework designed to achieve practical universality (the ability to learn across tasks with differing feature domains, modalities, and label spaces). 

The authors first develop a theoretical framework distinguishing algorithm-explicit and algorithm-implicit meta-learners, formalizing concepts such as valid learning algorithms and practical universality.

TAIL implements an algorithm-implicit learner using a non-causal transformer that directly predicts query labels from sequences of support examples and an unlabeled query. It also contains some other techs, such as universal feature encoding, etc.

Extensive experiments validate the effectiveness of TAIL from in-domain, cross-domain, cross-modality settups. It demonstrates robustness to tasks with up to 20× more classes than in training.

### Strengths
1. The motivation is clear and strong.

2. The paper’s claims are well-supported by both strong theoretical grounding and comprehensive empirical validation.
- The theoretical categories of algorithm-implicit vs. explicit learning are novel.
- Experiments cover a broad evaluations: in-domain, cross-domain, cross-modality, and label extrapolation, with ablation studies validating the design choices. Also, the computation is efficient.

### Weaknesses
1. Pretrained encoder dependency: Although random projections help, the reliance on large pretrained encoders complicates the claim of “from-scratch” universality, as part of the final performance may stem from the backbone’s prior knowledge. The importance of this encoder component should be explored more thoroughly. For example, by using the same pretrained encoder for both TAIL and the baselines to isolate its contribution.

2. Computation cost in experiments: While the paper reports efficiency results, it lacks a detailed analysis of the training cost at scale. It would also be helpful to quantify how much of the total computational cost arises from the encoder.

3. Insufficient ablation analysis: The impact of the randomized extended permutation 𝜋 could be analyzed more systematically. Table 5 in the appendix only compares results with and without 𝜋. A sensitivity analysis would clarify how strongly TAIL depends on this component.

### Questions
In addition to the above weakness, some additional questions:

1. Could you provide more insight into how the non-causal transformer architecture outperforms the causal transformer arch?
2. Have you compared TAIL’s performance to recent foundation-model-based few-shot learners (I just searched and found one paper attached below. To answer this question, the authors could use any other related papers or models, not necessarily the following one)
- Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners, CVPR 2023
3. How would the model perform if meta-trained on mixed-modality/distribution tasks rather than purely modality/distribution data?

### Soundness
3

### Presentation
3

### Contribution
4
