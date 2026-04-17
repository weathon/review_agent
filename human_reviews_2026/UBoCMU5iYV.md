# Memorizing Long-tail Data Can Help Generalization Through Composition

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Deep learning has led researchers to rethink the relationship between memorization and generalization. In many settings, memorization does not hurt generalization due to implicit regularization and may help by memorizing long-tailed examples. In this paper, we consider the synergy between memorization and simple composition \--- the ability to make correct prediction on a combination of long-tailed features. Theoretically, we show that for a linear setting, memorization together with composition can help the model make correct predictions on rare test examples that require a combination of long-tailed features, even if such combinations were never observed in the training data. Experiments on neural network architecture on simple data show that the theoretical insight extends beyond the linear setting, and we further observe that the composition capability of the model depends on its architecture.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work considers an interesting an interesting angle of how memorizing long-tail samples can be helpful for generalization. Theoretically, it proves that for linear classifier---whether the underlying distribution is noiseless or noisy---memorization can help generalization both in and out of distribution under some assumptions on the distributions. Empirically, it shows that memorization can help on an interesting construction of task: computing the sum of MNIST digits. With proper choice of model architecture, memorization can help improve the results when a certain digit is significantly under represented in the data set.

### Strengths
The presentation of this work is outstanding: theoretical claims are clearly defined and the underlying intuition of the proof well explained. I also appreciate the authors' effort of motivating the problem as well as presenting the related work with precise and succinct language. The theoretical claims are sound. I don't find any apparent problem in the proof, either. The design of the three-digit-sum problem is new to me. Despite some limitation of the design, which I will come to later, the idea is intriguing. Overall, this work is solid technically.

### Weaknesses
Compared to the outstanding presentation and rigorous theoretical formulation, this paper is slightly weaker on the potential impact. Specifically:

1) The linear case is slightly simplistic as composition is natural. If two features both contributes positively to the prediction score, i.e., have positive corresponding field in $\beta$, then observing one of them at a time in training example should suffice for good test results. In nonlinear case, e.g., an XOR, observing one feature at a time may not be sufficient for the telling the outcome when both features are present (nonzero). In fact, the sum-of-digit task is somewhat linear with respect to the digits. I wonder if the phenomenon of composition can be observed in more general tasks.

2) The notion of memorization here is slightly different from the literature I'm familiar with. I'm more used to influence score based criteria for memorization, e.g., removing a training example will significantly impact the prediction of another example. I believe this work assumes that an overparametrized model with unregularized training will memorize. Is this assumption common in literature? 

3) Following 2), MNIST is a fairly 'simple' dataset for which a small size of sample can lead to good model already with or without memorization. For stronger impact, the authors may want to consider some more complex tasks.

### Questions
My questions are mainly on the potential impact of the work.

1) Could you provide some more real world examples of tasks where composition is natural?

2) If time allows, could you quickly check the influence score a training example on itself or on the test samples? Either a simple leave-one-out test (retrain the model with a training set differing by one entry) or the estimation in Feldman,Vitaly and Zhang.

3) What could be the future extension of the result in this work? 


Feldman, Vitaly, and Chiyuan Zhang. "What neural networks memorize and why: Discovering the long tail via influence estimation." Advances in Neural Information Processing Systems 33 (2020): 2881-2891.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores how memorization of rare, long-tail examples can improve generalization when combined with a model’s ability to compose known features in new ways. Through theoretical analysis in linear settings and small-scale experiments on compositional MNIST tasks, the authors show that memorization enables correct predictions on unseen combinations of rare features.

### Strengths
The paper provides a clear theoretical formulation connecting memorization and compositional generalization, an underexplored relationship in deep learning theory. Its synthetic and modified MNIST experiments effectively illustrate how architectural structure influences compositional ability. Finally, it contributes a valuable conceptual shift, framing memorization not purely as overfitting, but as a potentially beneficial mechanism for learning from long-tail data.

### Weaknesses
**Oversimplified Definition of Memorization**

The paper treats memorization as a binary property i.e., models either memorize or do not memorize. This definition ignores the nuanced ways sample level memorization actually behaves. For example, memorization scores can vary from 0 (i.e., perfect generalization) to 1 (perfect memorization). By treating the property as binary, the authors are ignoring the entire range of values between 0 and 1.


**No Empirical Verification of Memorization**

Despite repeatedly claiming that rare examples (like the digit “9”) were memorized, the authors never tested this directly. They inferred memorization from improved performance on rare-digit test cases but did not apply any established measurement techniques (e.g., Feldman et al's self influence), to verify that the model had indeed memorized those samples. Without this validation, the central claim that memorization enables composition remains speculative.

**Reliance on Indirect Behavioral Evidence**

The experimental support for memorization is limited to behavioral trends: test loss decreases as the frequency of the rare digit increases and increases when weight decay is applied. While suggestive, these results can also be explained by better statistical coverage or regularization effects rather than genuine memorization. The lack of causal evidence weakens the argument of this work.

### Questions
See above

### Soundness
2

### Presentation
2

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
This work builds on to the line of work by Feldman and Zhang that has studied how long tail memorization can help with generalization in deep learning.

There is a key conceptual shift that this paper makes on top of Feldman. Feldman argued that memorization helps because test examples are similar to memorized training examples, which allows the model to recall them directly. This paper adds a new dimension to this discourse: Memorization helps not only because it reproduces similar examples, but also because it enables composition. This means that combining multiple memorized rare examples can lead to generalization into new configurations.

To demonstrate this idea of compositionality, the authors move away from the singleton tasks in past work to new tasks such as (i) a “sum of three MNIST digits” setup and (ii) an MNIST–Omniglot mixture testing one-shot memorization. 

The authors develop a theoretical model in which different data features follow a power-law frequency distribution.
They prove that the min norm solution which memorizes training data can correctly predict on test examples composed of multiple long-tail features that never co-occurred during training. The theoretical argument is supported via the experimental results on the newly created synthetic datasets. 

Results show that networks capable of processing input components modularly (e.g., per-digit ResNets with additive aggregation) generalize compositionally, whereas architectures that entangle inputs early (“cross-channel” ResNets) fail.

The paper also shows that an attempt to mitigate memorization (such as weight decay penalty) leads to a loss in model generalizartion on such compositional tasks.

Disclosure: I have not reviewed the theory carefully.

### Strengths
1. Conceptual Extension: I quite like the extension this paper attempts to make over the singleton memorization argument made in Feldman et. al. The bridge is quite intellectually appealing and can connect various ideas like one-shot generalization and memorization in overparametrized models.
2. The power law based feature setup seems quite simple yet expressive. I believe this is sufficient to motivate the emprical underpinnings of the work
3. The paper has a good mix of toy tasks: from linear regression to a controlled mnist and omniglot task. I like how they are able to connect the architectural dependence here as we visualize the transition from memorizatioon to composition.

### Weaknesses
1. The main weakness of the work is in its experimental scope. I admit that this in general will stay a hard task but I would like to challenge the authors to find meaningful ways to extend these setups to those of more practical relevance. 
     i. this requires identifying where in the real world do one-shot composition of memorized instances naturally happens
     ii. run controlled experiments to actually experiment by ablating away that capability
     iii. if the memorized composition was indeed a mechanism by which models generalize, i actually thing it is quite a useful exercise to show that this happens in real tasks. if not, why is this phenomenon of interest? i am writing this as motivation rather than actually questioning the value of this line of work, which i quite like.
2. I believe this paper also needs a discussion on when memorization hurts composition. This is especially true for scenarios such as spurious correlations. How would the theory and/or experiments inetrsect with this.

### Questions
1. The task of single example memorization in big models is hard. I wonder if some efforts around experimentation with PEFT, or in context examples can somehow connect here. In context learning is an example of single example generalization with high information recall (which is what you intend to use the word memorization for, anyways). This is just one thought to aid experimentation.

### Soundness
3

### Presentation
3

### Contribution
4
