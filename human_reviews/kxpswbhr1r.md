# In-context Convergence of Transformers

- Avg Score: 5.75
- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
Transformers have recently revolutionized many domains in modern machine learning and one salient discovery is their remarkable in-context learning capability, where models can solve an unseen task by utilizing task-specific prompts without further parameters fine-tuning.  This also inspired recent theoretical studies aiming to understand the in-context learning mechanism of transformers, which however focused only on  $\textit{linear}$ transformers.  In this work, we take the first step toward studying the learning dynamics of a one-layer transformer with $\textit{softmax}$ attention trained via gradient descent in order to in-context learn linear function classes. We consider a structured data model, where each token is randomly sampled from a set of feature vectors in either balanced or imbalanced fashion. For data with balanced features, we establish the finite-time convergence guarantee with near-zero prediction error by navigating our analysis over two phases of the training dynamics of the attention map. More notably, for data with imbalanced features, we show that the learning dynamics take a stage-wise convergence process, where the transformer first converges to a near-zero prediction error for the query tokens of dominant features, and then converges later to a near-zero prediction error for the query tokens of under-represented features, respectively via one and four training phases. Our proof features new techniques for analyzing the competing strengths of two types of attention weights, the change of which determines different training phases.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This is a theoretical paper centers around addressing “How do softmax-based transformers trained via gradient descent learn in-context? “. In particular, the author(s) investigated the training dynamics of a one-layer transformer with softmax attention trained by GD for in-context learning. Previous studies only focused on linear transformers i.e. without softmax function.  The authors give convergence results regarding in context learning with balanced and nonbalanced features.

### Strengths
Originality: This is a very novel paper as the authors provide a theoretical aspect of the in context learning dynamics of nonlinear transformer signified by the softmax activation in the self attention layer/module. This has not been examined before. 

Quality: the quality is high. The authors define the problem mathematically and details around the proofs are included. They also provide an interesting aspect of the learning dynamics via the relevant attention scores during the 4 phases of convergence. 

Clarity: It is fairly clear as the mathematical notations are given prior to introduction of their main development. The presentation can be easily followed via definition of problems, proposed approach/theoretical establishment, and training phase analysis for two specific problems of linear regression where sampled features are balanced and nonbalanced. 

Significance: the paper is important for ml theory as it discussed in context learning with “nonlinear” (softmax) transformers.

### Weaknesses
The major weakness is lack of empirical experiments. The linear regression setting according to task distribution and data distribution should not be too challenging to do synthetic experiments for balanced and nonbalanced features similar to Garg et al. or Zhang et al., 2023a. It will be interesting to compared with linear transformers such as Zhang et al. 2023 where in certain task scenarios, linear transformer fails. 
Another concern is the nonlinearity of transformers are examined on features from linear functions. Would be possible to develop some insights on features from nonlinear functions?

### Questions
1.	The paper does not introduce/define Θ(K) in the notations. 
2.	“set of distinct features {vk ∈Rd,k=1,...,K},where all features are orthonormal vectors”, does that mean the maximum K is d, since an orthonormal basis is d dimensional?  Is it too limiting? Correct me if I am wrong, does that also mean we can only have d distinct features/tokens as in part of Definition 3.1 part 2? 
3.	The paper adopts W^V and W^{KQ} from linear SA by Zhang et al.  for example by setting v =1. Although the authors justify the use of v=1 with two reasons.  However, the objective from Zhang et al.  involves a residual connection term. In this paper, there is no residual connection term involved.  Could the authors justify either the use of v =1 as a scaling factor in their objective as in Equation (1) or not including the residual term in Equation (1) and subsequently in equation (3) and (4)? 
4.	The four training phases of under-represented features differ. Could the order of phase 2 and 3 be swapped? 
5.	Did author investigate other tasks with softmax transformer as discussed by the Garg et al. paper?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the convergence of one-layer encoder-based transformer with softmax attention only. The authors show that training this model with gradient descent with pairs $(x_i,y_i)$, where $x_i$ can be one of $K$ possible vectors $u_k$, and $y_i = wx_i$ for some $w$ that is drawn from some distribution. Two settings are considered: in the first one each of the possible $u_k$ vectors has  probability $\propto 1/K$ to be drawn (balanced data), while in the second one some vector has some vector being drawn with constant probability and the rest with probability $\propto 1/K$. In both settings they show that the transformer converges to an approximate minimum.

### Strengths
The convergence of transformer models is of interest to the community. The paper also studies sequence to sequence models, which is a setting unexplored in terms of convergence. The two-phase transition is intuitive, since the model is first trying to maximize the inner product of the token to be predicted with the ones that are the same vectors and then use them to extract their corresponding labels.

### Weaknesses
The main weakness of this paper is the setting. Specifically, even though the authors study sequence to sequence models, they also consider:
1. That all training points are one of k possible vectors. This is not compatible with the training process for linear regression. In general, in which setting we only have $K$ vectors that we sample them in that way.
2. The weight matrices are very constrained and sparsified to match the ones proposed in the construction of [2]. This construction also refers to linear transformers. The authors are multiply the matrices $W_K^\top W_Q$ and train them as one matrix (this also impacts the training dynamics). 
3. The paper also consider only the attention mechanism and no residual. 

I understand that the achieved loss is not very different, however this simply indicates that for the simple setting of linear regression transformers are more expressive than necessary. I do not think that these results are indicative of how the training is evolved.

### Questions
1. Could the authors clarify the motivation of this setting? Fixed data points and constraint sparse matrices while considering the task of linear regression. 
2. Could the authors comment on how the two-phase transition observed in this setting, could be indicative of what is happening during actual training of these models?

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into the behavior of transformer architectures, particularly focusing on their convergence properties in various contexts. Transformers, known for their self-attention mechanism, have gained immense popularity in NLP and other domains. The paper's primary objective is to investigate the factors affecting the convergence speed and stability of transformers when trained in different contexts.

Key contributions and discussions of the paper include:

1. An exploration of how the context (e.g., input data distribution, task complexity) impacts the convergence properties of transformers.
2. The introduction of a novel metric to quantify and measure the convergence speed and stability of transformers.
3. A series of experiments that highlight the varying convergence behaviors of transformers across different tasks and datasets.

### Strengths
1. Given the widespread use of transformers in various domains, understanding their convergence properties is of paramount importance. This paper addresses this gap by exploring the role of context in transformer convergence.
2. The paper provides a series of well-designed experiments that shed light on how transformers behave across tasks, datasets, and other contextual factors. This empirical evidence strengthens the paper's claims and findings.
3. The paper is well-structured, with clear explanations and visualizations, making it accessible to both experts and those less familiar with transformer architectures.

### Weaknesses
1. While the paper does a good job of examining transformers across various tasks and datasets, there's a limitation in the breadth of architectures studied. Delving into different variants of transformers or comparing with other architectures could have provided a more comprehensive picture.

2. While the introduced metric is innovative, there could be more rigorous validation or comparison against other potential metrics. This would strengthen the metric's claim as a standard measure for convergence properties.

3. The paper could benefit from a discussion on the practical implications of the findings. For instance, how can the insights on convergence be used to improve training methodologies or model selection in real-world applications?

4. A deeper exploration into the individual factors affecting convergence (e.g., model size, training data size, task complexity) could provide a granular understanding and more actionable insights.

5. This paper contains several typographical errors, which detract from the reading experience. For example, there is a missing space before "We" in the 9th line of the summary.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
3 good

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
This study presents training dynamics of a one-layer transformer with softmax attention, focusing on in-context learning via gradient descent (GD). Two scenarios, one with balanced features and the other with imbalanced features, are analyzed. The authors show convergence towards a diminishing in-context prediction error by exploring the evolution of attention dynamics in both settings. 

In the case of imbalanced features, a multi-phase behavior is characterized, shedding light on the interplay of attention dynamics between dominant and under-represented target features during training. Overall this is a nice analysis of softmax attention and gradient descent dynamics in the context of in-context learning.

### Strengths
1. The paper studies training dynamics and convergence of gradient descent for training a single layer transformer model with softmax attention to do in-context learning. 

2. It provides convergence guarantees for both balanced and imbalanced feature settings, showcasing the effectiveness of transformers and gradient descent for in-context learning. The results break down training into two phases for balanced features and four phases for imbalanced features. This provides valuable insights into how transformers adapt during training. The paper shows that transformers quickly achieve near-zero prediction error for dominant features and eventually converge to near-zero prediction error for under-represented features, regardless of their infrequent occurrence.

3. The paper introduces a novel proof technique that characterizes softmax attention dynamics by considering the interplay between two types of bilinear attention weights: 'weight of query token and its target feature' and 'weight of query token and off-target features.' The dynamic shift in dominance between these weights throughout the learning process leads to different training phases. This could be applied to other problems involving transformer architectures.

### Weaknesses
Weaknesses and questions,

1. Are the phases artifacts of the analysis or is it really the case? Could you provide some simulations that demonstrate this multi-phase convergence of gradient descent on in-context learning with transformers? 

2. The results and analysis seem to abstract away the task diversity (different $w$ vectors). In-context learning is about generalizing to unseen tasks and it looks like the analysis is based on doing gradient descent on $L(\theta)$ which is the expectation over the task vectors $w$ and the features vectors $x$. The setup for in-context learning in Garg et al. 2022 (Figure 1) is more pragmatic and I was expecting an analysis on this. This analysis studies the convergence of gradient descent in minimizing $L(\theta)$ with transformer/self-attention as an underlying model. It's a nice study, given the additional complexity introduced due to softmax, but I think it is not exactly studying in-context learning instead it's a study of convergence of gradient descent on regression with one-layer transformers. The title also seems confusing, it suggests that the paper is about studying the convergence of transformers to the right output during inference time. 


3. Could you simplify the presentation and expand on the main intuition behind the multi-phase convergence? Especially after page 7, instead of the lemmas some simulations/illustrations could be very helpful in understanding the key ideas behind the multi-phase convergence in balanced and imbalanced settings.

### Questions
See above and I have one more question.
1. Why do you need the features to be orthonormal vectors and be drawn from a finite set? Does the analysis depend heavily on this assumption? 
2. Could you make the dimensions of the main terms clear (e.g. in Definition 3.1) o.w. the readers have to work them out.
3. Minor, the upper case $K$ is used for $W^{K}$ and also for the number of feature vectors. 
4. The bounds on $T^*$ have quadratic dependence on $K$ (the number of distinct feature vectors). Is this order of dependence necessary? What would happen if the features are drawn from a continuous distribution say multi-variate gaussian?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
