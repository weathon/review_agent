# REVISITING LARS FOR LARGE BATCH TRAINING GENERALIZATION OF NEURAL NETWORKS

- Decision: Reject
- Scores: 6, 6, 5

## Abstract
LARS and LAMB have emerged as prominent techniques in Large Batch Learn-
ing (LBL), ensuring the stability of AI training. One of the primary challenges
in LBL is convergence stability, where the AI agent usually gets trapped into the
sharp minimizer. Addressing this challenge, a relatively recent technique, known
as warm-up, has been employed. However, warm-up lacks a strong theoretical
foundation, leaving the door open for further exploration of more efficacious al-
gorithms. In light of this situation, we conduct empirical experiments to analyze
the behaviors of the two most popular optimizers in the LARS family: LARS
and LAMB, with and without a warm-up strategy. Our analyses give a compre-
hensive insight into the behavior of LARS, LAMB, and the necessity of a warm-
up technique in LBL, including an explanation of their failure in many cases.
Building upon these insights, we propose a novel algorithm called Time Varying
LARS (TVLARS), which facilitates robust training in the initial phase without the
need for warm-up. We run extensive experimental evaluations to demonstrate that
TVLARS achieves competitive results with LARS and LAMB when warm-up is
utilized while surpassing their performance without the warm-up technique.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
LARS proportionally scale the learning rate at each layer to improve stability in large batch learning. However, it is well known that LARS suffer from sluggish convergence at early stages of training, hence requiring a warmup procedure. This paper hypothesizes and empirically verifies that LARS tend to fall into sharp minimizers at the initial stages of training, leading to an explosion of the adaptively scaled gradient. As a result, this paper proposes TVLARS, and verify on experiments that TVLARS outperform LARS with warmup.

### Strengths
This paper proposes TVLARS, and demonstrates empirically that TVLARS outperform LARS under large batch settings for the CIFAR10 and TinyImageNet datasets. This paper is well-written, easy to follow, and the experiment results are clear.

### Weaknesses
The main weakness of this paper is that I am unsure whether TVLARS could be applied in big datasets that might benefit more from large batch training. After all, CIFAR and TinyImageNet are becoming antique from the modern ML perspective. See Questions for more details.

### Questions
1. Could the authors comment on whether TVLARS would be useful in datasets that are large in modern conception---ImageNet, large NLP datasets, etc?
2. I didn't understand what Figure 2 was trying to say. Is the point of Figure 2 to show that warmup is better than no-warmup in some way?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors conduct a very thorough empirical study of existing SOTA per layer learning rate adaptation rules, LARS and LAMB. They then identify key patterns in what makes or breaks a good training trajectory and relate that back to shortcomings in the per layer adaptation rules. They then propose key changes that should reduce these shortcomings, and they package them in TVLARS. 

Their proposed method is then empirically evaluated and ablated over a variety of datasets, learning rates, batch sizes and tasks. TVLARS appears to have significantly better generalization performance than its competitors, and improved training stability.

### Strengths
- Very thorough empircal evaluation of existing methods is already very valuable
- Empirical evaluation of existing methods from new angles adds further value
- The authors identify key shortcomings in their empirical observations that align with theoretical understanding.
- The authors then combine theoretical understanding and their empirical insights to craft a new method, TVLARS, resulting in a well rounded motivation for the method.
- The new method is thoroughly evaluated
- The new method appears to have a significant gain margin over the baselines.

### Weaknesses
- The writing throughout suffers from clumsiness, flow problems and structure problems. 
- The key method description and presentation is hidden in a number of paragraphs, and the key algorithm box is hard to parse. 
- The authors often assume that their audience should know certain things that are not likely to be known by the average deep learning research scientist.

### Questions
Suggestions:

Please take the time to rewrite key parts of the paper, for example, in the abstract:

For example, some constructive comments for the abstract:

Your abstract on LBL optimization introduces pivotal concepts and promising methodologies. I offer succinct feedback for refinement:

- Clarify "sharp minimizer" to aid reader comprehension.
- Detail the theoretical gaps in warm-up techniques, emphasizing the contribution of your work.
- Articulate the distinguishing features of TVLARS to highlight its novelty.
- Quantify "competitive results" to underscore the empirical strength of TVLARS.

These focused enhancements will sharpen the abstract's precision and academic rigor.

Then, in your methodology section:

- Clearly introduce the ingredients of your method.
- Provide intuitive explanations for each
- Cross reference these explanations in your algorithm diagram. 
- Try to reduce the parsing complexity of your algorithm -- or even better use a nice figure that showcases how your method compares to existing methods functionally and mathematically.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focused on the convergence stability of large-batch training. More specifically, the authors noticed that recent methods, such as warmup, lack theoretical foundation and therefore they try to conduct empirical experiments on LARS and LAMB. Based on the above analysis, they propose a  novel algorithm Time Varying LARS (TVLARS) to make the initial training phrase stable without the need of warm-up. The experimental results on CIFAR-10, CIFAR-100 and Tiny-ImageNet also illustrates that the proposed method can further improve the performance of LARS and LAMB for large-batch training.

### Strengths
1. This paper tries to drop warmup and make large-batch training stable is very interesting. Since we usually use a large learning rate when batch size is very large and warm-up is important to make the training process stable.
2. The proposed method is very easy to follow.

### Weaknesses
1. Although this problem is very interesting, but I still not very clear why we should drop warmup. In my experience with large-batch training, I think warmup is a very simple and important method. 
2. For your proposed method, I think we need to tune more hyper-parameters to get a great performance, which may make the proposed method less convenient to use. 
3. Although the proposed method can improve the performance of LARS and LAMB when we don't use warm-up, I noticed that the accuracy is still too low compared with LRAS/LAMB + warm-up.

### Questions
1. I think layer-wise optimization methods (LARS/LAMB) are very sensitive to the initialization methods of weights. For eq. (3), the layer-wise learning rate depends on the weight norm of each layer since the gradient is normalized. Therefore, I think the reason why the initial training process is unstable is related to the initialization method. So my question is whether you try to use different initialization methods and analyze their results.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
