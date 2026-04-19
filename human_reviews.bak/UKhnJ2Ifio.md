# What Images are More Memorable to Machines?

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 6, 3

## Abstract
This paper studies the problem of measuring and predicting how memorable an image is to pattern recognition machines, as a path to explore machine intelligence. Firstly, we propose a self-supervised machine memory quantification pipeline, dubbed ''MachineMem measurer'', to collect machine memorability scores of images. Similar to humans, machines also tend to memorize certain kinds of images, whereas the types of images that machines and humans memorize are different. Through in-depth analysis and comprehensive visualizations, we gradually unveil that ''complex''  images are usually more memorable to machines. We further conduct extensive experiments across 11 different machines and 9 pre-training methods to analyze and understand machine memory. This work proposes the concept of machine memorability and opens a new research direction at the interface between machine memory and visual data.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a concept called the machine memorability that somehow involves how well a machine can memorize a data sample. For computing the memorability score of an image for a model, it first leans from a rotation prediction task. Then the model is trained to predict seen and unseen images, where seen images are those used in the rotation prediction task. The score of an image perhaps is the score of the seen label given an image. The paper also presents a model to regress the score. The experimental results shows that the memorability score presented in the paper is correlated with pixel statistics.

### Strengths
The idea of machine memorability might be interesting.

### Weaknesses
(1) The paper does not provide a clear definition of machine memorability. Is it solely associated with the model’s structure (the number of parameters) or anything else, like how the model is trained? The current method for measuring the memorability seems to involve not only the structure but also how it is trained. If this is the case, the paper should provide some experiments that show how different pretraining strategies affect the memorability. Because of the missing definition, it’s hard to see if the method provided in the paper makes sense or not.

(2) Related to the previous point, I don’t see what the memorability means and what it can tell about the model (and the training strategy). Is it supposed to give some insights about the performance of the model? I can guess the memorability involves the inductive bias that the model structure has, the number of parameters, and how and over what dataset it is trained. Because of these multiple factors involved, it’s hard to see what it actually measures. 

(3) I don’t see why the memorability should be measured in real time. Does it have any specific application that requires real-time measurement? If the purpose of the memorability measurement is to see the capability of some aspects of the model, the regression model seems to solely deteriorate the precision of the memorability measurement. 

(4) The current implementation uses a fixed number of training epochs, but I think the memorability can be highly affected by overfitting, which may allow the model to predict the seen images easily. This point is missing in the paper.

### Questions
I would like to see the discussion on all points in the weakness section.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work aims to understand which images are "memorable" to machine learning models, paralleling prior work on human memory. The notion of "memorability" here comes from comes from a multistep pipeline. Fix a neural network architecture and 4 disjoint groups of images $A, B,$ and $C$. First, train the network on $A$ and $B$ using self-supervision. Then, replacing the network's final layer with a linear classifier, train the network to discriminate between sets $B$ (labeled "seen") and $C$ (labeled "unseen"). Finally, run this classifier on images in $A$. The "machine memory score" of an image in $A$ is the probability of this classifier labeling it as "seen." Formally, it's the observed probability over 100 runs, where the sets $B$ and $C$ are drawn randomly each run from a larger set of images. This process, with its 100 iterations, is somewhat computationally demanding.

After introducing this notion, the paper works to understand which images have high scores and compare these results to existing work on human memory. They do this in a number of ways, including visual inspection of images and using "GANalyze" to modify images while increasing or decreasing memory scores.

The paper aims to understand memory in "machines" broadly. To this end, they repeat their experiments on several different neural network architectures and show how memory scores are correlated across these experiments.

Finally, to circumvent the computational cost of computing memory scores, they train a resnet to predict the memory scores. Prior work used machine learning to predict the human memorability of images.

One main conclusion is that machines tend to memorize more complex images. This stands in contrast to humans, whom prior work has shown tend to remember simpler images.

### Strengths
This paper is thoughtfully constructed and easy to follow. Despite missing some important related work (see below), I would still describe the work as "scholarly." It seems well-grounded in a line of research on human memory. The authors do an excellent job comparing the results to those on humans.

I enjoyed seeing the results from GANalyze and found those images quite instructive.

Existing work on "what machines remember" contains many takeaways, some of which parallel those found here. However, I suspect that this paper will, after revision, be a solid contribution with a distinct flavor.

### Weaknesses
This paper does not reference the existing literature on memorization in machine learning [important works include 0, 1, 2]. Recent work explores these questions in the context of SSL [3]. Particularly relevant is [4], which find that "which images are memorized" depends on the training data as a whole: if highly memorized examples are removed from the data, then other examples may be subject to higher memorization. In addition to work directly on memorization, there is relevant work on "membership inference," relating memorization to privacy violations, and "example hardness," which often investigates questions about which examples neural networks "remember" and forget." There are many papers on both these topics, see e.g. [5] and [6] for recent work with good related work sections. 

To the best of my knowledge, the experiments in this paper have not been performed before. However, incorporating existing knowledge from this large body of prior work would require a major revision. Thus, I vote to reject.

The work is interesting and I encourage the authors to continue this research. In addition to comparing with prior work, I see two main areas for improvement. First, I hope the paper might more directly address why *this* notion of machine memory is particularly insightful. Second, I would rework the section on predicting memory scores. The regression model performs nontrivial prediction (as we can see by the correlation), but without more validation the reader cannot draw strong conclusions. Perhaps the resnet performs poorly on the most memorable images? This seems possible (given the evidence presented) and would be fatal to some of the results.

[0] Zhang, Chiyuan, et al. "Understanding deep learning requires rethinking generalization." arXiv preprint arXiv:1611.03530 (2016).

[1] Feldman, Vitaly. "Does learning require memorization? a short tale about a long tail." Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing. 2020.

[2] Feldman, Vitaly, and Chiyuan Zhang. "What neural networks memorize and why: Discovering the long tail via influence estimation." Advances in Neural Information Processing Systems 33 (2020): 2881-2891.

[3] Guo, Chuan, et al. "Do SSL Models Have D\'ej\a Vu? A Case of Unintended Memorization in Self-supervised Learning." arXiv preprint arXiv:2304.13850 (2023).

[4] Carlini, Nicholas, et al. "The privacy onion effect: Memorization is relative." Advances in Neural Information Processing Systems 35 (2022): 13263-13276.

[5] Carlini, Nicholas, et al. "Membership inference attacks from first principles." 2022 IEEE Symposium on Security and Privacy (SP). IEEE, 2022.

[6] Maini, Pratyush, et al. "Characterizing datapoints via second-split forgetting." Advances in Neural Information Processing Systems 35 (2022): 30044-30057.

### Questions
Why is the proposed method the correct way to implement the visual memory game of Isola et al.? Are there other options that you feel would be equally adequate?

Is there a task or question that this pipeline is uniquely suited to addressing?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper describes a novel “MachineMem measure" pipeline, which measures if the image can be remembered by a machine network, like Resnet, Vit, etc. The pipeline for measuring the machine memorability score is described, and detailed analysis is conducted on the output scores. Overall machine memorability scores correlate weakly with human memorability score, and more strongly with lower image cues. The scores across two methods (Resnet, Vit) also may not correlate well. Overall paper demonstrates the concept of machine memorability and an analysis of the output scores.

### Strengths
The strengths of the paper are as follows:
1. The concept of machine memorability is novel and different from human memorability score. The concept can be potentially useful in understanding the network, though its not fully established in the paper how. The paper has described the pipeline for setting up the measurement well, using pre-training through rotation, repeat game (similar to human memorability), with ample references.
2. The paper has provided detailed analysis of correlations of the predicted scores with various existing image features and semantics, and tested it across different networks. The emerging pattern show that the machine memorability is quite different from human memorability and that there is variation among networks as well, which established that different networks encode differently. As contrast to low variability among humans (in memorability tasks), machines show much higher variability.

### Weaknesses
The weaknesses are as follows:
1. The paper does not establish a downstream usecase for the scores. A simple experiment would be to analyze the correlation of the machine memorability scores with task output of the network. Does the network makes more mistakes on lower memorability score images? This could establish the usability of the scores better,
2. Since the pipeline is novel, the paper does not present any comparisons with a competing method. If the best usecase for the method is augmenting the network or interpreting the network, authors should present the results on a task after augmenting the dataset with low memorability scores or competing methods for interpreting the network like GradCam.

### Questions
Questions mentioned in the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies an interesting probelm --- what makes an image are more memorable to a machine. To answer this question, the author  conducts extensive experiments across 11 different machines and 9 pre-training methods. To measure the memorability, the author proposes an approch composed of two training stages, (1) recognizing the rotation degree, and (2) recognizing unseen images.

### Strengths
1. The problem is interesting and should be inspire some other researches.
2. Some findings in this paper shoule be inspiring.

### Weaknesses
1. The motivation of this paper should be more clear. After reading this paper, it is difficult to see the significance of the memorability, though it sounds interesting. For example, if we can exactly measure memorability, what can we use it to do in the fields of computer vision and machine learning?

2. I do not think the approach to measure memorability is reasonable. In section 3, the authors argue that they use ResNet-50 as the basic model, but there are two problems. (1) if ResNet-50 is pre-trained on a large-scale dataset like ImageNet-1K, how do you guarantee the data samples in stage (a), (b) and (c) are not in the pre-training dataset? I think the samples in the pre-training dataset are seen images. (2) if ResNet-50 is randomly initialized, after training by using stage (a) and (b), how to guarantee the generalizibility?

### Questions
See the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
