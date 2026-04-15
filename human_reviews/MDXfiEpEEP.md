# LNL+K: Enhancing Learning with Noisy Labels Through Noise Source Knowledge Integration

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 5, 3, 5, 3

## Abstract
Learning with noisy labels (LNL) aims to train a high-performing model using a noisy dataset. We observe that noise for a given class often comes from a limited set of categories, yet many LNL methods overlook this. For example, an image mislabeled as a cheetah is more likely a leopard than a hippopotamus due to its visual similarity.  In fact, we find that many datasets have meta-data information that directly provides potential noise sources. Thus, in this paper, we explore a task we refer to as Learning with Noisy Labels with noise source Knowledge integration (LNL+K), which assumes we have some knowledge about likely source(s) of label noise that we can take advantage of. We find that integrating noise source knowledge boosts performance, even supporting settings where LNL methods typically fail.  For example, LNL+K methods are effective on datasets where noise represents the majority of samples, which breaks a critical premise of most methods developed for the LNL task.  We also find that LNL+K methods can boost performance even when the noise sources are estimated rather than provided in the meta-data. Our experiments provide several baseline LNL+K methods that integrate noise source knowledge into state-of-the-art LNL models across five diverse datasets and three types of noise, where we report gains of up to 15% compared to the unadapted methods. Critically, we show that LNL methods fail to generalize on some real-world datasets, even when adapted to integrate noise source knowledge, highlighting the importance of directly exploring our LNL+K task.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a solution to enhance learning with noisy labels using the knowledge of noise source. It presents a few extensions from current solutions to add the noise source knowledge.  Basically, if the sample is more likely to belong to a noise source, then the label can be considered a noisy label. Experimental results show the method works to some extent.

### Strengths
The paper presents a simple and straightforward method to detect noisy label under a very strong and unrealistic assumption. Multiple datasets are evaluated to conduct the experiments to show the idea work.

### Weaknesses
There are several major limitations of the paper.

(1) The novelty of the paper is very limited. The idea is established under a very strong assumption that the noise source is known. Moreover, mathematically, the method is also unjustified.

(2) The depth of the paper is not strong enough. There is no theory to support the claims. The method also looks ad-hoc. The level of depth is too far away from ICLR paper.

(3) The paper is also not well-written in the sense that the motivation of the paper is not clear.

### Questions
In addition to very limited novelty and not enough technical depth, it is also unclear that how is the probability that labels is clean computed and how the noise source is being identified.

For instance, the noise label created by different people with different levels of experience can be mixed without the knowledge of source. It is a more general case. It is important to solve a more general more instead of an edge case.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper explores an overlooked task of learning with noisy labels utilizing noise source knowledge. The paper proposes a simple wrapper method that can be used on top of existing methods for noise labels. Overall, the paper studies an interesting setup and the proposed method is practical. However, there are some concerns on the applicability of the method and also the experiment evaluation.

### Strengths
1. Intuitively, using noise source knowledge can indeed be helpful.
2. The paper proposes a simple method that can be plugged into existing base models.
3. Experiments shows improvement from the proposed method.

### Weaknesses
1. Writing can be improved. There are simple errors that should have been avoided by proof reading. For example. “when considering the presence of the noise source yellow class, it becomes evident that these noisy samples are closer to their true label class.” Where is “yellow” class in the figure? This causes confusion as “yellow” class is also referred in Equation 2. 
2. The applicability of the method should be made more clear and more intuition should be provided. 
3. Experiment setup can be improved to better justify the claims.

### Questions
1. It seems the method works in the scenaio where there are confusing class pairs on which examples are miss labeled. However, in real-world, there can be many more noise label patterns in a same dataset, for example random white noise. How does the method work with the coexistence of other noise patterns?
2. It is mentioned that the proposed method "are effective on datasets where noise represents the majority of samples". Given a new dataset at hand with only noisy ground-truth labels, how should one decide whether to use the proposed method? and how would one know whether noise represents the majority of samples?
3. How often and also when is equation 2 different from selecting the highest probability class? Can you provide some numbers and examples from experiments? This can be helpful to understand the method better.
4. How is the model trained after detecting the noise examples? Do you just drop the noise examples?
5. It would be the best to also consider baselines that uses robust loss functions/regularization/etc

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The work focuses on a very practical problem - learning with label noise. It introduces the concept of LNL+K, which incorporates additional 'noise source knowledge' into existing sample selection methods. Specifically, this work utilizes additional 'noise source knowledge' to identify potentially confusing noise classes $D_{c-ns}$. Instead of considering the probability of the annotated class being clean~($P_{c}$), the proposed approach compares the probability of a given label being clean with that of the confusing classes. The proposed strategy is combined with various existing sample selection methods and evaluated on several datasets.

### Strengths
1. The presentation is clear.
2. Multiple existing sample selection methods are considered for evaluation.

### Weaknesses
1. The novelty is somewhat limited. As quoted in the paper - "The selected sample’s probability may not be the highest", indeed current sample selection methods, such as the widely-applied GMM style, possibly lead to false positives (hard negatives), as they only consider whether the annotated label is 'clean enough or not' while ignoring its relative 'cleanness' versus other classes. Upon this, though introduced as 'noise source knowledge', the core idea of this work is to compare the 'cleanness' of the annotated class versus other classes, which is straightforward and trivial in existing sample selection heuristics, and been applied already ('consistency measure in [1], probablity difference in [2], there should be more work in major venues). This may sounds a bit stringent, but I expect more insights rather than rephrasement, especially for venues like ICLR.

2. I expect more 'real noise source knowledge' to be considered, rather than current ones (transition matrix, etc.), which still rely on the noisy labels and current in-training models, leading to self-confirmation again. 

3. The confusion class set \(D_{c-ns}\) induced by 'noise source knowledge' involves new hyperparameters. More ablations are necessary. 

4. For a sample selection strategy, there always exists a dilemma of precision and recall, especially when extra hyperparameters are involved. This requires more detailed analysis.

5. The considered real-world noisy datasets lack persuasiveness. Experiments should be conducted on at least Clothing1m and WebVision. 

[1] SSR: An Efficient and Robust Framework for Learning with Unknown Label Noise, BMVC2022
[2] P-DIFF: Learning Classifier with Noisy Labels based on Probability Difference Distributions, ICPR2020

### Questions
See weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed incorporating noise source knowledge into some sample selection methods by comparing the confidence of noisy labels and noise source label. The proposed method is simple and easy to be integrated into multiple existing LNL methods. Experiments confirm the effectiveness of the proposed method in certain cases.

### Strengths
1. The studied problem that how to exploit noise source knowledge in noisy label learning is novel and interesting.
2. The proposed method is very simple but reasonable, which can be integrated into many methods.
3. The datasets include cell datasets which shows the potential of the proposed method in scientific research.

### Weaknesses
1. As shown in the experiments, in some cases, the proposed method will lead to a decrease in performance. The authors should offer a deeper analysis about the reason of the decrease in performance and when could the performance gain be ensured.
2. The real-world datasets are small-scale and special. It would be better to test the performance of the proposed methods with estimated noise source knowledge in more large and general benchmarks, e.g., Clothing1M and WebVision.
3. Some writing need further clarification. First, how to generate dominant noise is still unclear.  Are there multiple noise sources for each recessive label. And why is the dataset still balanced after mislabeling in these cases? It seems that the number of examples labeled by dominant classes is less than recessive classes. Second, how to use DualT to estimate noise source knowledge is not clear.
4. (Minor) The related works can include more recent classifier-consistent methods, e.g. [1,2,3].
5. (Minor) What does the "noise supervision" mean?

[1] Estimating Noise Transition Matrix with Label Correlations for Noisy Multi-Label Learning. NeurIPS 2022

[2] A holistic view of noise transition matrix in deep learning and beyond. ICLR 2023

[3] Identifiability of label noise transition matrix. ICML 2023

### Questions
See above weaknesses.  I am happy to increase my score if my concerns are addressed.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a new task and method for the task of Noise Source Knowledge Integration. They assume that a set of possibly confusing classes for a given other class is made available e.g. the knowledge that trucks and automobiles are easily confused with each other. They integrate this knowledge in the learning process and demonstrate improved results on synthetic and real world datasets when their method is combined with various selected methods from the noisy labels literature which currently do not integrate noise source knowledge.

### Strengths
- The authors report broadly positive results with improvements on most datasets in most settings.

- The authors demonstrate that their method can be successfully combined with a range noisy labels learning methods.

- The use of the CIFAR-10 and CIFAR-100 synthetic noise datasets is relatively standard in the noisy labels literature, and the authors introduce some non-standard evaluation datasets with interesting scientific applications (BBBC036, CHAMMI-CP).

### Weaknesses
- I found the presentation extremely hard to follow, with many terms' definitions unclear to me and/or errors in the notation and presentation that made following the paper very difficult. I have listed what I regard as errors below and have deferred to the questions section various unclear terms for clarification. I am open to substantially improving my score if it is clear I have misunderstood the paper and it is clarified to me, but at present given my careful reading of the presentation I find the entire method definition and hence its evaluation unconvincing as I cannot understand it.
    - There are repeated references to a yellow class in Figure 1 e.g. "the noise source yellow class", "p(yellow | x_i) > p(red ? x_i)", there is no yellow class as far as I can see.
    - The output of algorithm 1 is denoted P(X) = {p(\tilde{y}_i | x_i)}, this neither seems to be the correct definition of P(X) or the algorithm output.


Small points:
- $\tilde{Y}$ is used in the literature and in section 2 to signify noisy labels, yet in paragraph 2 of section 3, this notation is flipped and now $\tilde{y}$ are the "true labels".

### Questions
- The definition of noise sources is unclear to me. Can you please confirm if my understanding based on the definition at the end of the second paragraph in section 3 is correct: noise sources for a given class c, is the set of all other classes such that there is a non-zero probability of a data point that is truly in class c being mislabeled as that class?
    - If my understanding is correct, then I find it hard to understand how this information is useful to the extent demonstrated in the experiments. Firstly, strictly based on this definition, all classes should be in the noise source set as there is a non-zero probability that a class c is mislabelled to any other class. However it seems to be the case that the definition of this set is in fact more loosely applied in the experimental section of the paper where the noise sources set for class c are really classes that have a relatively high probability of being confused for class c. Nonetheless I struggle to see how this information would be as useful as some of the experimental results seem to show, as essentially this additional information would merely say on a dataset level which classes are reasonably likely to be confused for each other. This is not per sample/input dependent nor does it convey as much information as the true transition matrix which would contain transition probabilities and not merely binary values for each class pair. Could the authors please comment?

- Figure 1 is very unclear to me, based on my understanding of noise sources above, then noise sources are not data points but classes, yet Figure 1 presents new data points as noise sources? In addition could the authors please clarify the meaning of red, blue, circles and triangles in the figure. For example, what is the meaning of a blue triangle?

- I do not understand the arguments leading to equation 2 in section 3.1. In particular:
    - "Fig. 1 has a high probability of belonging to the red class, i.e., p(red|xi) > δ, then it is detected as a clean sample in LNL. However, compared to the probability of belonging to the noise source yellow class, p(yellow|xi) > p(red|xi), so the red triangle is detected as a noisy sample in LNL+K." As in this paper, noisy labels methods are usually evaluated on multi-class classification tasks. Hence I do not understand how a standard LNL method would fail in this case when p(yellow|xi) > p(red|xi), then while p(red|xi) > δ it must be the case that p(yellow|xi) >> δ and hence the sample would be labelled as yellow by the standard LNL method. Fundamentally I fail to see how standard LNL methods would not satisfy equation 2, please explain?

- How the method is incorporated into the various LNL methods is unclear to me. I can understand how the various methods currently identify clean labels. But a very short description is given of the integration in each of the 5 cases. I do not think the level of detail would be sufficient to replicate your results or to combine the method with another LNL method. For example: "To estimate the likelihood of a sample label being clean in CRUST+k, we mix this sample with all other noise source class samples and apply CRUST to the combined set. If the sample is selected as part of the noise source class cluster, we assume its label is noisy." What does it mean to mix the "samples"? What are the samples in this case? The noise sources as defined above are simply a set of classes, they are not input dependent nor can I see how they can be mixed with training examples?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
