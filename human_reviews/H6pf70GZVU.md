# YoooP: You Only Optimize One Prototype per Class for Non-Exemplar Incremental Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 3, 5, 6

## Abstract
Incremental learning (IL) usually addresses catastrophic forgetting of old tasks when learning new tasks by replaying old tasks' data stored in a memory, which can be limited by its size and the risk of privacy leakage. Recent non-exemplar IL methods only store class centroids as prototypes and perturb them with Gaussian noise to create synthetic data for replay. However, the class prototypes learned in different tasks might be close to each other, leading to the intersection of their synthetic data and forgetting. Moreover, the Gaussian perturbation does not preserve the real data distribution and thus can be detrimental. In this paper, we propose YoooP, a novel exemplar-free IL approach that can greatly outperform previous methods by only storing and replaying one prototype per class even without synthetic data replay. Instead of storing class centroids, YoooP optimizes each class prototype by (1) moving it to the high-density region within every class using an attentional mean-shift algorithm; and (2) minimizing its similarity to other classes' samples and meanwhile maximizing its similarity to samples from its class, resulting in compact classes distant from each other in the representation space. Moreover, we extend YoooP to YoooP+ by synthesizing replay data preserving the angular distribution between each class prototype and the class's real data in history, which cannot be obtained by Gaussian perturbation. YoooP+ effectively stabilizes and further improves YoooP without storing any real data. Extensive experiments demonstrate the superiority of YoooP/YoooP+ over non-exemplar baselines in terms of accuracy and anti-forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles the non-exemplar class-incremental learning problem, an important problem in the machine learning field. The authors propose the extension of the prototypical network as corresponding loss terms to tackle this problem. The proposed method is evaluated on several benchmark datasets against other competitors.

### Strengths
1. This paper tackles the non-exemplar class-incremental learning problem, an important problem in the machine learning field.
2. The proposed method is evaluated on several benchmark datasets against other competitors.
3. The ablation study and visualizations are clear and intuitive.

### Weaknesses
1. Overall, I find the title may be unsuitable for the current manuscript. The proposed method utilizes prototypes to construct the training loss, while the embedding and classifiers are always optimizable throughout the learning process. Hence, not only “prototypes” are optimized, but also the backbone and classifiers are optimized for the current task.
2. Some illustrations are unclear. For example, there is no X coordinate in Figure 1, making it hard to figure out the memory usage of different methods. Corresponding explanations are also needed to show why other methods like PASS and FeTrIL require a much larger memory size. These methods also save the class prototype to generate instances, and the memory gap between them should be illustrated.
3. The experimental results should be reorganized. The current results in the main paper only focus on the Base-0 setting, while I also see the results in supplementary that some Base-50 results are also available. It would be better to reorganize these results to contain both settings since Base 50 is also common in today’s CIL.

### Questions
1. Please explain the significance of sampling in Section 3.2.1 and its advantage over PASS/FeTrIL. Besides, clarifying the memory budget of different methods in Figure 1 and Section D is also essential. Showing a table with the extra memory budget could be a good solution.
2. It requires further experimental analysis, e.g., the influence of hyper-parameter gamma in Eq. 11.

In summary, this paper tackles an interesting problem with novel techniques. The proposed method shows competitive results in the benchmark comparison. Generally, I am positive about this submission, while addressing the concerns above is also essential for my decision.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a prototype-based method that can concentrate most samples to their class prototypes and keep each class a compact cluster in the feature space, which can mitigate inter-class interference. When learning a new task, the saved prototypes are used for replay to deal with forgetting.

### Strengths
** The proposed method is reasonable. 
** The paper is easy to read.

### Weaknesses
** The method is incremental as some prototype-based methods already exist. The paper made some improvements. But the improvements can also be obtained by contrastive learning as the proposed idea is very similar to that of contrastive learning. 
** The paper didn’t say where the forgetting occurs. My understanding is that the technique simply learns and saves the prototype for each class. Each task is learned independently. Then there is no forgetting. 
** The inference procedure is not discussed. If the algorithm saves only the class prototype for each class, how does it get the feature representation for each test sample for cosine similarity computation in prediction? 
** The average accuracy in (Rebuffi et al. 2017) is average incremental accuracy (AIC). Based on AIC, your reported accuracy results are low. Please also report the last accuracy, i.e., after learning the last task and compare with the above methods. 
** The experimental datasets and baselines are too few. More baselines with or without using exemplars should be compared as saving some data is not an issue. [a] is a non-exemplar based method. 
(1) Kim et al. (2022). A theoretical study on solving continual learning. NeurIPS.
(2) Wang et al. (2022). Beef: Bi-compatible class-incremental learning via energy-based expansion and fusion. ICLR. 
(3) Wu et al. (2019) Large scale incremental learning. CVPR.
(4) Buzzega et al. (2020) Dark experience for general continual learning: a strong, simple baseline. NeurIPS.
** Please give the efficiency results. The method seems to be quite slow.

### Questions
see the weaknesses

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed new non-exemplar prototype-based method for class-incremental learning (CIL) setting. Prototype are optimized with attentional mean-shit method, while training the new tasks, they are replayed for old classes. To mitigate forgetting the autors as well use distillation at the feature level (l2). During the training, model interpolation and partial freezing of the classifier is used. This method is called YoooP. Additionally, in this work authors proposed extension to it, when some improvement to prototype replay based on data augmentation is proposed. The authors compared their method on Cifar-100 and TinyImageNet in compariosn to few other exemplar-free CIL methods.

### Strengths
S1. CIL exemplar-free setting is a challenging setting, where most of the recent work put more interest to fight only with forgetting - starting with already good backbone (base-50% setting) and only fighting with forgetting (FeTrIL, SSRE). This work seems to be competitive in harder "base0" setting.

S2. Visualization part of the paper is good, well support written text. But still, not enough to provide all the info - seem W1.

S3. Interesting founding that PASS w/o Augh works better for some settings.

### Weaknesses
W1. The main weakness of this work is that after the reading the main paper the reader won't be able to reproduce and know exactly how the method works. The crucial part is in the appendix, i.e. Algorithm 1, where you see that we have S steps inside each epoch to calculate L_t, R iterations to calculate prototypes p_k, main loss - Arcface is mentioned in the implementation details and it hyper-parameter sigma etc. 

W2. Some equations give even a wrong intuition how the loss is computed, e.g Eq.4. and the first component is over the whole dataset, while in practice is withing S minibatches. 

W3. Some crucial hyper-parameters are not provided in main paper, or not provided at all - number of S steps, R iterations.

W4. Multiple things are combinend in the Yooop method  - three losses components, where feature distillation is quite strong regularization preventing forgetting but at the same time lowering plasticity. Additionally, mixing this with model interpolation with beta = 0.6 and freezing the classifier. The ablation is not clear about every single participant. Seems like the model interpolation is crucial here, and bug gain is for YoooP+. 

W5. Why in the first figure we have missing units for the memory size? It's the first figure that supposed to give some motivation, but currently it raises more questions. Additionally, SSRE to my knowledge has the growing part and then compression. How it's possible that it's in line with FeTrIL, PASS, IL2A? 

W6. Why the method is not evaluated with ImageNet-100? This dataset would shade more light how the method performs on the bigger images and how to compare with the other baselines.

W7. Logic in the reasoning. Page 7, before saying: "Therefore, we can draw a conclusion that the proposed methods can outperform the non-exemplar baselines." :

    While SSRE and FeTrIL have lower average forgetting than our methods, their prediction accuracy drops rapidly in the initial tasks, resulting in low accuracy in the final task,  as shown in Figure 4.  In reality,  the lower forgetting in SSRE and FeTrIL is attributed to the sharp drop in accuracy in the early tasks and the limited learning progress in the subsequent tasks

That is true about SSRE and FeTrIL, I agree. They mostly fight with forgetting. That's why they are good in base-half setting (see Tiny and FeTrIL in your appendix, Yooop is lower there). But how you can conclude this after this two sentences. Overall, in base0 setting your methods presents better accuracy. But to conclude this, the behavior of the SSRE FeTrIL dosen't matter here.

W8. Sec 4.3 "This is because YoooP+ can form a compact cluster for the samples in each class via prototype optimization and create high-quality synthetic data from the original distribution of cosine similarity to constrain **the boundary of old tasks.**"   - maybe the experiment to show that? 

Overall, it is quite interesting work, but Yooop combines so many things, and it's not clear how each of it contributes. After reading the main paper you cannot re-produce it, you can have the idea how it works, appendix is necessary, but still not enough. The paper should be rewritten paying attention in all the hyper-params, mentioning arcface loss with sigma, and good ablation. Maybe some insight, comparison with PASS can be moved to the appendix (to have space for more crucial information/ablations).

### Questions
Q1. Out of the curiosity, why in the intro when you introduce class-incremental learning setting you ref. to three quite recent works of Zhu and Zhou (2021-2022)? This scenario of CL training is with us longer than that.

Q2. Why in the ablation of beta (Fig.8 (a)) we see different starting points for the first task? Beta is the interpolation that will take place after the first task.

Q3. Why starting points in Fig.4 for Yooop and Yooop+ are not the same? (easy to see for Tiny)

Q4. Why not providing source code in the supplementary? Without it I think the results will be hard to reproduce. I guess you use an existing framework for your work, looking at the results it's PyCIL? Would be good to give credits to authors and point your starting impoementations.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
(Motivation)
The prototype-based (class-mean) method, which is one of the exemplar-free Class Incremental Learning (CIL) methods, stores class-representative prototypes (=class mean prototype). However, the class mean prototype does not represent the centroids of each class. Utilizing inappropriate prototypes leads to confusion between old classes learned in different stages. Moreover, prototype augmentation, which adds Gaussian noise to the prototype, leads to more serious catastrophic forgetting of previously observed classes.

(Method)
1. They propose the prototype-based CIL method, called YoooP, that optimizes the prototype, considering the weighted average distance of all samples in the class. The paper claims this class-wise prototype is more representative than utilizing the simple class mean one.

2. They propose a prototype augmentation, called YoooP+, that synthesizes the prototypes from the angular distribution between each class’s real data and prototype stored in memory. Synthetic prototypes preserve the distribution of the real data rather than simply adding the Gaussian noise to the prototype (augmentation technique of PASS).

### Strengths
This paper proposes a novel approach to leave the class representative prototype considering the similarity with all real samples. The proposed method aims to be an efficient technique that can be easily applied to the existing prototype-based method.

This paper also proposes a data augmentation performed with the help of a rotation matrix. This strategy approximates real distribution better than conventional prototype augmentation techniques.

In base-0 experiments on CIFAR-100 and Tiny-ImageNet, the proposed methods achieve the best performance.

### Weaknesses
-	The scale of dataset (CIFAR-100, TinyImageNet) is small. Whether the proposed method works with bigger datasets is an important issue that needs to be addressed. 
-	The paper lacks the result with 20 phases which is a quite common configuration. 
-	The effect of an optimized prototype based on the attentional mean-shift method seems limited in many situations.

### Questions
1) For verifying the strength of proposed augmentation, why don’t you apply the new strategy to PASS instead of adding Gaussian noise to prototype? Would you show the performance of it?

2) Could you show the classification confusion matrix result of your approach?

3) In section 3, when define the probabilities over all classes C_(1:t ), isn’t softmax(wG(F(x;θ))) more accurate notation than softmax(wF(x;θ))?

4) The function c(,) in equation 2, equation 4 seems to be different. Is the c(,) in equation 2 cosine similarity function and one in equation 4 classifier (=G(F(p_k;θ)))?

5) How can we define the initial class-wise prototype p_k? Is it the same with the prototype used in PASS? (k-th class mean prototype)?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
