# One-shot Active Learning Based on Lewis Weight Sampling for Multiple Deep Models

- Decision: Accept (poster)
- Scores: 5, 5, 6, 5

## Abstract
Active learning (AL) for multiple target models aims to reduce labeled data querying while effectively training multiple models concurrently. Existing AL algorithms often rely on iterative model training, which can be computationally expensive, particularly for deep models. In this paper, we propose a one-shot AL method to address this challenge, which performs all label queries without repeated model training. Specifically, we extract different representations of the same dataset using distinct network backbones, and actively learn the linear prediction layer on each representation via an $\ell_p$-regression formulation. The regression problems are solved approximately by 
sampling and reweighting the unlabeled instances based on their maximum Lewis weights across the representations. An upper bound on the number of samples needed is provided with a rigorous analysis for $p\in [1, +\infty)$. Experimental results on 11 benchmarks show that our one-shot approach achieves competitive performances with the state-of-the-art AL methods for multiple target models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This study introduces a one-shot AL approach aimed at reducing the computational overhead caused by iterative model training. The strategy for sampling unlabeled instances leverages the maximum Lewis weights across various representations. The authors validate their method on eight classification and regression benchmarks, employing 50 deep learning models to demonstrate its efficiency.

### Strengths
- This work aims to present a more comprehensive framework, applicable to multiple models and multiple error norms. 
- The paper is well-structured and easy to follow.
- The motivation behind the sampling strategy is easy to comprehend.

### Weaknesses
- If the results from the empirical study suggest that as the number of models increases, the sum of the Lewis weights grows very slowly, this may indeed imply that the information between different models might not be so "distinct". If I understand correctly, isn't there a certain gap between this and the motivation mentioned in the introduction?
- In the experiments, why are the baseline settings different for the classification task and the fine-tuning regression task, with the former being iterative and the latter one-shot? 
- In the experiments, the running time of data querying and model training for different methods in classification benchmarks was presented. What about in the fine-tuning scenario?
- Why isn't there a one-shot method for multiple models in the baseline?

### Questions
- Can this method be generalized to situations where different models have differen norms in their training objective functions?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper discusses the use of a one-shot method for active learning to save on the running time of retraining models. The base of the algorithmic development is a solution to the p-th regression problem involving the sampling matrix . The authors also show improvement on prior work for p=2 from O(d^2/e\(\epsilon^4) to O(d/(\epsilon^4)). Empirical results are presented on comparison with other AL algorithms, showing comparable results and reduced running time.

### Strengths
1.	Claims improvement on prior work in terms of sample complexity
2.	Reduced running time for the one-shot approach vs AL.
3.	An approach that tackles real problem of having multiple models

### Weaknesses
1.	Presentation is poorly motivated in many cases
2.	The empirical results cover a rather limited scope of data sets, most are MNIST types data sets and some curves don’t reflect SOTA (see questions below). The experimental setting is also counterintuitive for active learning scenarios.
3.	I am missing the connection between the theoretical & algorithmic construction and the empirical validation.
4.	Clarity: 
a.	It isn’t clear what is the relation between the A matrix, the models and the data. Sometimes A is referred to as data and sometimes it is referred to as a model
b.	Why is the reweighting done by sqrt(mp_q)^-1? What is the motivation\intuition for that?

### Questions
1.	What is p in the in the experimental settings?
2.	Why are you starting with 3000 points for initialization? Isn’t that too much (e.g. for MNIST)? Does actual active learning start of with a random 3000 points? isn’t there too much redundancy in querying 3000 point on each query step?
3.	How come random performs better than other AL methods? E,g, Coreset in cifar 10? This does not correspond to the results reported in Senar et. al.
4.	What is L^2 in (1)? Is it the Lipschitz constant?
5.	Can you please show me where your claim about a reduced sample complexity for p=2 is proven? I’ve looked at the supplemental material and still haven’t seen it. If this is indeed a result it should also be addressed in your theorems, just as Gajjar et. al. do….

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the domain of one-shot active learning for multiple target models. The authors introduce a novel active learning approach that leverages Lewis weights across representations from various target models. The authors provide both sample complexity analysis and empirical results. The empirical results indicate that the proposed method achieves performance comparable to those of iterative active learning methods.

### Strengths
a. This paper delves into a compelling and practical setting, one-shot active learning for multiple models, which holds significance in real-world scenarios.

b. The proposed approach, employing maximum Lewis weights, is both innovative and well-grounded. The sample complexity analysis presented in the paper is sound, with the derived bound improving upon prior results by a factor of d.

c. The paper's empirical results are promising. The authors compare their proposed approach with several iterative baselines, including the state-of-the-art method DIAM, demonstrating a substantial improvement in efficiency without sacrificing performance.

### Weaknesses
a. While the empirical results are promising, the datasets used in the experiments are relatively small for deep learning models. It would be valuable to assess the method's performance on larger datasets, such as CelebA, to evaluate its scalability and generalizability.

b. A more comprehensive evaluation could include an assessment of how the proposed approach compares with state-of-the-art general active learning methods in the multi-model setting, e.g., BADGE[1]. 

c. The paper's writing, particularly in Section 3.1, may benefit from further refinement. The section lists several definitions but offers limited explanations and intuitions, which may challenge readers who are unfamiliar with these terms and concepts, hindering their comprehension of the paper.

[1] Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. arXiv preprint arXiv:1906.03671, 2019.

### Questions
a. What's the method's performance on larger datasets such as CelebA?

b. How does the proposed method compare with state-of-the-art general active learning methods in the multi-model setting?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a one-shot active learning algorithm which only queries the unlabelled target data once, which is in contrast to the existing AL algorithms that often rely on iterative model training. This one-shot query can reduce computation costs from repeated training, especially for deep models. Specifically, the authors extract representations of the same dataset using distinct network backbones and learn the linear prediction layer on each representation via an $\ell_p$-regression formulation.

### Strengths
This paper tackles the one-shot active learning, which is an interesting idea since the iterative training procedure of active learning can indeed be computationally expensive. The proposed method is theoretically rooted and technically sound. Besides, the source code showed that the proposed algorithm seems to be reproducible.

### Weaknesses
My primary concern is about the problem setting. As a motivation, the authors claim that the iterative training of traditional active learning can be computationally expensive, and the same instance can have different representations through distinct backbones. However, in the proposed method, the dataset was pre-processed with 50 different models to get the representation, which, in my view, is still computationally expensive. Thus, I found the motivation for one-shot active learning is somehow weak.

My second concern was about the empirical evaluations. The baselines are compared in the empirical evaluations. Only DIAM was published in the past two years. Some recent Deep AL method is missing in the empirical evaluation. 

Furthermore, the benchmark datasets evaluated in the paper are somehow simple. More complex and challenging dataset, e.g. CIFAR100, which was widely compared in the AL literature, should also be tested.

### Questions
I have some questions and comments, and I hope the authors can try to clarify.

1. My first question is about the one-shot query. I do admit that it will reduce the computational burden; however, I'm wondering whether the performance will be comparable with the traditional iterative training procedure.
2. In the empirical setting part, I'm confused about the setting of *50 distinct architectures*. Since it's noted in the footnote that all the experiments were conducted with the same GPU and CPU, I'm wondering whether it's still necessary to employ such method?
3. As I mentioned in the weakness part, more recent baselines should be compared to demonstrate the effectiveness of the proposed method.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
