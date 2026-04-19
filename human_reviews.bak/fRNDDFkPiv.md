# Controlling Forgetting with Test-Time Data in Continual Learning

- Decision: Reject
- Scores: 5, 6, 8, 8

## Abstract
Foundational vision-language models have shown impressive performance on various downstream tasks. Yet, there is still a pressing need to update these models later as new tasks or domains become available. Ongoing Continual Learning (CL) research provides techniques to overcome catastrophic forgetting of previous information when new knowledge is acquired. To date, CL techniques focus only on the supervised training sessions. This results in significant forgetting yielding inferior performance to even the prior model zero shot performance. In this work, we argue that test-time data hold great information that can be leveraged in a self supervised manner to refresh the model's memory of previous learned tasks and hence greatly reduce forgetting at no extra labelling cost. We study how unsupervised data can be employed online to improve models' performance on prior tasks upon encountering representative samples. We propose a simple yet effective student-teacher model with gradient based sparse parameters updates and show significant performance improvements and reduction in forgetting, which could alleviate the role of an offline episodic memory/experience replay buffer.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose to leverage test time adaptation to improve the performance of foundational models when fine-tuned continually. They show significant improvements on many benchmarks when using their proposed TTA method.

### Strengths
- Explores the intersection of test-time adaptation and continual learning which is an interesting direction
- Performs an ablation study
- Comparison with many existing methods

### Weaknesses
- **W1**: More explanation around the exact setup used for test time adaptation needs to be made. For instance, it is not clear if Test time adaptation is done image-wise, in which case it would be fair to compare a test time adaptation method to another method that does not use test time adaptation, because both methods operate in the same setting. However, if this is not the case (for instance if test time adaptation is performed on batches that come from a single task, then evaluated on that task), this is not fair at all for other methods and any comparison with non-TTA method is doomed.
- **W2:** **Lack of baselines** Related to the previous weakness, in some TTA setting (that make sense in the wild), a lot of these CL methods you compare to could benefit from a simple change that would make them able to do TTA in some way (way less complicated than the method you propose). For instance, simply by allowing them to gather batch statistics in the BN layers if any, or by inferring the current task through retaining task prototypes and then only looking at the output of this tasks head.

I cannot review the paper further without having a clear answer and explanation inside the paper on how this TTA is performed, so I invite the authors to consider explaining this part very carefully. In particular, it should be made clear **in what order** the samples come to the model during the TTA phase, and when is the evaluation perfomed ? (after all samples have been processed or immediately after processing one sample)

### Questions
- **Q1**: Is test time adaptation performed image-wise or batch-wise ?
- **Q2**: If it is performed batch wise, is it performed using the shuffled test set comprising all tasks at the same time, or is it performed task-by-task ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel approach to Continual Learning that enables the learner to benefit from unsupervised data available during the test phase, addressing a challenge known as Test-Time Adaptation (TTA). The approach can be divided into two main contributions: the first focuses on mitigating catastrophic forgetting during training on sequential tasks, while the second introduces test-time adaptation (applied during the evaluation phase). Both processes leverage the well-established teacher-student paradigm, incorporating a dual mechanism to control the rate at which each weight of the teacher moves towards the corresponding weight of the student (dual momentum). Experiments demonstrate that the proposed architecture achieves good performance.

### Strengths
- The setting explored by the authors is noteworthy and realistic.
- The approach is technically solid.
- The concept of dual momentum is promising, though I have a few questions for the authors (see below).

### Weaknesses
My primary concerns pertain to the **experimental setup**. Specifically, I found the comparison in Table 1 somewhat misleading. While DoSAPP leverages additional (albeit unsupervised) data from all tasks, the other approaches are standard continual learning methods that do not utilize test-time data. To their credit, the authors discuss this and include an additional comparison in Table 5 with a simple baseline combining SPU and pseudo-labeling. The authors justify this baseline by stating that SPU is the best-performing continual learning (CL) method; however, this claim does not hold up when examining the results in Table 2. For instance, SparseCL outperforms SPU, as does ZSCL (on average). Consequently, I feel the comparison is currently weak and should be expanded.
Additionally, while pseudo-labeling is a simple and effective approach, it only scratches the surface of Test-Time Adaptation (TTA), with many more advanced and effective methods proposed in recent years. Confidence in the technical contributions of this work (specifically, those outlined in Phase 2) would be strengthened if the authors extended their comparative analysis to include more recent TTA approaches. For example, the authors of CoTTA (Continual TTA) compared their approach not only with pseudo-labeling but also with BatchNorm adaptation and TENT (the latter outperforms pseudo-labeling).

I understand that these suggestions might sound like the typical “please, provide more experiments.” However, when a novel experimental setting bridges two or more fields, it is essential to design experiments that thoroughly consider the current state-of-the-art in each field.

Finally, I recommend that the authors correct the use of bold font in Table 2, where it should highlight top-performing methods rather than the proposed method. For instance, L2P achieves an accuracy of 75.45 on GTSRB, which is 3 points higher than DoSAPP (currently in bold). The same issue occurs with CIFAR-100 (where ZSCL performs better) and with the forgetting metric on Aircraft, CIFAR-100, CUB, and GTSRB.

**Clarity**
The concept of dual momentum is interesting, yet I believe there is a clarity issue in the paper. The authors state that a subset of parameters is frozen during training, following Zhang et al. (2023b). However, if certain student parameters remain unchanged during training, I would assume that their corresponding teacher parameters (updated via EMA) would also remain unchanged. This raises a question about how EMA could affect parameters that are kept frozen (line 311). I’m uncertain if this confusion arises from a lack of clarity in the paper or my own interpretation, but even after multiple readings, I could not resolve it.

**Presentation**

The paper is generally clear and well-written; however:
- In the section titled "Teacher Student Framework," the symbol indicating the parameter set is missing (see Eq. 3). Based on Fig. 2, it appears the symbol was \theta. I recommend the authors add it, as its absence complicates reading.
- Table 4 is presented inline within the text. This is acceptable, but it would be helpful if it appeared closer to its reference in Sec. 4.3.
- I suggest the authors refine punctuation and correct spacing between consecutive words, parentheses, citations, etc. For example, there are multiple instances where a blank space is missing between a word and a parenthesis.
- In Section A.3, the last line seems to be leftover text from a previous submission.

### Questions
I do not have questions for the authors.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes to use the test time data to alleviate the forgetting problem in the class incremental learning problem. The proposed method includes two training phases: a supervised learning phase using labeled training data, and an unsupervised learning phase using test time data. With the proposed DoSAPP, the forgetting problem can be greatly alleviated, and thus achieve a promising final performance.

### Strengths
1. The paper is well-motivated. 
2. The idea of using test-time data to alleviate forgetting is interesting, and the performance is attractive. 
3. It is surprising to have a positive backward transfer, thanks to test time data.

### Weaknesses
**Major:**
1. It can be beneficial to have an ablation study about the size of test-time data $D^u$, with respect to the final performance.

2. It is suggested to include more information about the experiments. For example, the error bars of major experimental results in Table 2. Also, it is suggested to include detailed training time information (comparisons with other methods, training time for supervised and unsupervised phases)

3. The presentation of the paper is a concern. There are significant formatting issues/typos/non-completed equations/informal telegrapheses in the writing. A careful overhaul or English proofreading is suggested.

**Minor:**
1. It can be interesting to expand the experiments from pretrained CLIP to conventional classification backbones, like ImageNet-21k pretrained encoders and FC classifiers.

2. The equations should also be properly punctuated. 

**Summary:**

The paper presents an attractive setting and a novel view of alleviating forgetting in the CIL problem. However, there are significant issues, which limit the readability and the reproducibility, and thus I lean towards rejection. I'm happy to update my score if the concerns are properly addressed.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed to leverage the test-time data in CL to mitigate forgetting, which is novel, interesting and realistic. The authors further proposed a student-teacher learning framework to effectively learn and retain knowledge in this scenario. Specifically, the proposed DoSAPP(Double Smoothing via affine projected parameters) method consists of two stages: a supervised learning stage as in conventional CL and an unsupervised adaptation stage for test-time adaptation. In the former, the student would selectively update its parameters based on the gradient magnitude, and the teacher is a weighted exponential smoothing of the student model with dual momentum. In the latter, they first pseudo-label the test-time data based on the confidence of teacher and student models and then apply a similar learning paradigm as in the supervised learning stage. The authors validate the effectiveness of the proposed method in various datasets.

### Strengths
1. The proposed test-time adaptation to mitigate forgetting is novel, interesting and realistic.

2. The paper is easy-to-follow, well presented

3. The proposed method is simple but effective

### Weaknesses
Some experimental comparisons and ablation studies are missing and some additional details need to be added. Please refer to the Questions section.

### Questions
1. Are the comparisons with existing works fair?

Existing work such as L2P, DualPrompt, SLCA utilized a pretrained ViT model, i.e. ViT-B/16, which is trained on classification tasks, whereas the proposed method leveraged CLIP-ViT-B/16, which is trained on image and text pairs. Did the authors repeat experiments of existing methods with the same backbone? If the backbones are different, it is not convincing to conclude the superiority in performance in Table 2. In addition, the learning strategy of existing methods is also quite diverse: L2P and DualPrompt are prompt-based CL methods, whose backbone is frozen, and SLCA fine-tunes the backbone, ZSCL leverages external unlabeled datasets for finetuning. It seems unreasonable to put all these methods together and directly compare them with DoSAPP. 


2. Why is the ablation study of single momentum + mask union missing in Table 3?

Although it was illustrated in lines 462-464 in the text, the scores are not presented in the table.

3. How is the self-labeling (line 422) done?

It seems that the authors used the fine-tuned model (line Finetune in Tab 2) to do pseudo-labeling, is that correct? Even though this seems a valid choice, an intuitive adaptation of existing methods to the test-time adaptation scenario is to use existing methods, e.g. SPU or DualPrompt, and use their trained model after each task to perform pseudo labeling on test data and further adapt these models as test-time adaptation (TTA). That might be a more comparable baseline in this novel scenario.

4. Some small typos need to be corrected

line 215 'were' -> 'where'

table 2, Acc. of CIFAR100, ZSCL should be bold instead of DoSAPP, similarly, Acc of GTSRB should be L2P, F. of GTSRB should be SLCA.

### Soundness
4

### Presentation
4

### Contribution
3
