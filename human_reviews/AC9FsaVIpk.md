# Gating is Weighting: Understanding Gated Linear Attention through In-context Learning

- Decision: Reject
- Scores: 6, 6, 6, 5, 6

## Abstract
Linear attention methods provide a strong alternative to softmax attention as they allow for efficient recurrent decoding. Recent research has focused on enhancing standard linear attention by incorporating gating while retaining its computational benefits. Such Gated Linear Attention (GLA) architectures include highly competitive models such as Mamba and RWKV. In this work, we examine the in-context learning capabilities of the GLA model and make the following contributions. We show that a multilayer GLA can implement a general class of Weighted Preconditioned Gradient Descent (WPGD) algorithms with data-dependent weights. These weights are induced by the gating and allows the model to control the contribution of individual tokens to prediction. To further understand the mechanics of weighting, we introduce a novel data model with multitask prompts and characterize the optimization landscape of the problem of learning a WPGD algorithm. We identify mild conditions under which there is a unique (global) minimum up to scaling invariance, and the associated WPGD algorithm is unique as well. Finally, we translate these findings to explore the optimization landscape of GLA and shed light on how gating facilitates context-aware learning and when it is provably better than vanilla linear attention.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work establishes a connection between Gated Linear Attention (GLA) architectures and Weighted Projected Gradient Descent (WPGD) algorithms with data-dependent weights in the context of in-context learning (ICL). It characterizes the optimization landscape of learning a WPGD algorithm and then studies the optimization landscape of GLA.

### Strengths
The paper is well-written and easy to follow.

It addresses a scenario of ICL in which training data are derived from multiple tasks, offering a more realistic framework than those considered in prior works.

### Weaknesses
The architecture considered in this paper appears to include only the GLA layer and no MLP layer. Consequently, a multi-layer GLA in section 3.2 would not fully align with a Transformer model. A discussion on the effects of the MLP layer would provide valuable insights.

In Section 5, the multitask setup appears to be simplified through the introduction of vectors $d$ as task boundaries and vectors $c$ as contextual features. Could you clarify the effect of each of $c$ and $d$ on optimal loss in this setting? Additionally, it would be insightful to evaluate the performance impact of including versus excluding each of $c$ and $d$ when training on real data.

Minor Comments
- Typo on line 179: "weight" should be "weigh".
- The label "n" of the x-axis in Figure 1 should be clarified for better understanding.

### Questions
Please see above.

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
3

### Summary
This paper tries to formulate and explore Gated Linear Attention (GLA) models (e.g. Mamba, RWKV) through in-context learning.
The main contributions are:
* Demonstrating that GLA models can implement data-dependent Weighted Projected Gradient Descent (WPGD) algorithms, where weights are induced by the gating function.
* Investigating the problem of learning an optimal WPGD, and by characterizing the loss landscape under a multitask data setting, showing the conditions under which there exists a unique global minimum.
* Characterizing the loss landscape of a 1-layer GLA and showing the constraints on the optimal weights.
*  Showing the differences between linear attention and GLA (with scaler and vector gating) and show scalar gating can has limited expressivity which can be enhanced by vector gating.

### Strengths
* I commend the authors' effort to establish a theoretical foundation for understanding GLA models.

* As the authors cited, the closest work to this work is Li et al. (2024). Using in-context learning, they also showed that H3 architectures (which are also GLA) can implement Weighted Projected Gradient Descent. So, this new paper has incremental contributions such as generalizing the formulation to more complicated GLA models by introducing a novel data model with non-IID examples and multitask prompts.

* Comparing the effects of scalar and vector gating mechanisms on performance provides valuable insights for crafting models.

### Weaknesses
* The paper has strong theoretical contributions. But, more empirical studies and comparisons could strengthen the practical applicability of the theoretical contributions.

* There exists another line of work (which is not based on in-context learning) such as papers cited below. They propose an implicit self-attention formulation for GLA models (including Mamba and RWKV). Do you think there is a connection between your work and this line of work, and is it possible to apply in-context learning for model explainablity and empirical studies?

Zong C, Shao J, Lu W, Zhuang Y. Stock Movement Prediction with Multimodal Stable Fusion via Gated Cross-Attention Mechanism. arXiv preprint arXiv:2406.06594. 2024 Jun 6.

Zimerman I, Ali A, Wolf L. A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models. arXiv preprint arXiv:2405.16504. 2024 May 26.

### Questions
The same as above.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the In-context learning ability of gated linear attention (GLA) and the weighted projected gradient descent (WPGD). It first shows that the single layer of GLA can implement one step of WPGD, and multiple layer of GLA can implement multiple steps of WPGD. Then, it delves into the optimization landscape of WPGD and GLA. It first shows that under some conditions, there exists a stationary point of one-step WPGD when the tasks within a prompt are correlated. It also shows that this stationary point is the unique global minimum under some other conditions. Finally, it shows that under some (strong) assumptions, the optimal ICL risk of GLA matches the optimal ICL risk induced by WPGD. in general, I think this paper delves into a very interesting problem: how does GLA behave on ICL tasks (with correlated tasks), but I think the paper needs a good revision and I will lean towards a rejection.

### Strengths
1. This paper delves into a very interesting problem: how does GLA behave on ICl tasks with correlated tasks? Can it implement something other than one-step of gradient descent (like what linear self-attention does)?

2. The theory looks solid and the proof looks correct to my understanding. They also have experimental evidence to show the performance gap among scaler-gated, vector-gated, and vanilla linear self-attention.

### Weaknesses
1. They claim that they establish the 'equivalence' between WPGD and GLA layer (line 198-199), but their construction result only shows that the GLA can implement a special subset of the WPGD. This is not an 'equivalence' to my understanding. Also, in terms of the optimization results, you have proven that the global minimum of the ICL risk induced by GLA and WPGD matches under some very strong assumptions, which is far away from your claimed equivalence. 

2. In the preliminary section, you introduce a token embedding matrix like (4), but this was not utilized in your submission. In the construction results in section 3, you show that GLA can implement WPGD using this token matrix, but for the optimization landscape result in section 5, you show the global minimum of the ICL risk of GLA estimator using a more complex token embedding matrix with delimiters (in equation 20). The setups in two sections are inconsistent and you did not show me why it is necessary to use the more complex token embedding matrix in terms of the optimization landscape. Is there any intrinsic drawback to use the original token matrix? Moreover, is there any practical case in real applications where people use a token embedding matrix like equation 20? I think the adoption of a more involved token embedding matrix in Equation 20 needs more verification.

3. You claim that you prove the advantage of using the gating scheme in linear self-attention, but you do not present any rigorous results showing that the ICL risks induced by the optimal linear self-attention (LSA) and the optimal GLA have large gaps and how large the gap is. You show the optimal ICL risks for LSA and GLA separately. Do these two optimal ICL risks hold under the exact same assumptions? If so, I will suggest to write a separate corollary to conclude how large the risk gap is and what magnitude it enjoys.

4. The assumptions in the submission needs more justification. For example, why do you need to assume  the delimiters are linearly independent? What is this activation function and what role  does it perform? In assumption C, why do you assume the correlation between tasks are zero for \beta_i and \beta_j (This is a very special case under definition 1, right?)

5. Theorem 6 seems confusing. In prior section, you are considering a scaler-gated linear self-attention and show that it can implement a PWGD estimator like equation (10) or equaion (3). The optimal ICL risk of PWGD is also established in this scaler-gated linear attention setup (like in Theorem 3 and 4). But in there 6, you claim that the optimal ICL risks induced by the vector-gated linear self-attention matches the optimal ICL risk of WPGD. I am wondering whether the L_{WPGD}^* in the theorem 6 means the optimal risk over the function class in equation 10? If so, there is a mismatch between these two setups, since you are saying that the optimal ICL risk induced by a wider class (vector-gated linear attention) can match the optimal ICL risk of WPGD class in equation 10, which a simpler class can induce. I think you should put more efforting in clearly stating the relationship among the scaler-gated, vector-gated linear attention, and a restricted subclass of WPGD represented by equation 10, as well as the optimal ICL risks induced by them.

### Questions
/

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors study In-Context-Learning (ICL) abilities of a broad and general class of recent sequence-to-sequence architectures, such as Mamba, RWKV, and Gated Linear Attention, which they collectively name GLA. They show that the models of this algorithm family with L layers can perform L steps of Weighted Projected Gradient Descent (WPGD) for the task of linear regression during forward pass when presented with several examples. Moreover, the authors examine the setting when multi-task ICL examples are drawn from different distributions and correlated with the target sample. They find out, both empirically and theoretically, that for several weighting mechanisms,  it is possible to reach the optimal loss in this task.

### Strengths
1. Authors encompass different and seemingly distinct model families such as Mamba and Linear Transformer into one analytical framework and study their ICL properties jointly which is a very convenient and refreshing paradigm.
2. If the results stated in the paper are correct, it would be exciting and reassuring to know that a very broad class of models, beside Transformers, is provably capable of In-Context Learning despite some findings in the literature (e. g. [1])

[1] Jelassi, Samy, et al. "Repeat after me: Transformers are better than state space models at copying."

### Weaknesses
Unfortunately, I was not able to fully verify the preconditions and mathematical derivations for the results in the paper. Below there are several examples of what I found confusing: 

1. Presentation is not self-contained, and it’s difficult to follow the arguments without prior reading of several referenced papers, notably Von Oswald et al. (2023), Ahn et al. (2024), Li et al. (2024), as well as  attempting to independently draw missing analogies and reductions between them and the reviewed paper.
2. In the papers (Ahn et al. (2024), Li et al. (2024)) mentioned above, their authors research Preconditioned Gradient Descent. In the reviewed paper, the authors discuss (Weighted) Projected Gradient Descent, although it’s ostensibly implied in line 68 to be the same algorithm. It’s unclear whether it’s indeed the same algorithm, and what are the differences between them if it’s not. It’s worth noting that the terms “projected GD” and “preconditioned GD” refer to different algorithms, in case the authors use them interchangeably.
3. If I understand correctly, the equation 7 states that parameter **$\hat{\beta}$** is the resulting predictor of data-dependent WPGD algorithm. **$\hat{\beta}$**  is constructed as a function of parameters $P_1, P_2$ and $\Omega$. However, formal definition of the algorithm, its connection to the parameters, and derivation on how it optimizes linear regression task and arrives at solution **$\hat{\beta}$** conditioned on priors $P_1, P_2$ and $\Omega$,  are not provided. 
4. Moreover, The theorem 1 states that  the output of single-layer GLA model with a specific construction of weights matrices parametrized by $P_1, P_2$ and $\Omega$  matches the prediction of **$\hat{\beta}$** , also specifically constructed and parametrized by $P_1, P_2$ and $\Omega$.  This joint construction does not shed light on whether the forward pass of 1-layer GLA performs one step of gradient descent or WGPD. It would be helpful to explicitly demonstrate and prove it as in e.g. Lemma 1 of Ahn et al. (2024).

References:

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. Advances in Neural Information Processing Systems, 36, 2024.

Yingcong Li, Ankit Singh Rawat, and Samet Oymak. Fine-grained analysis of in-context linear estimation: Data, architecture, and beyond. arXiv preprint arXiv:2407.10005, 2024.

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023.

### Questions
See weaknesses.

Also, there might be some typos or minor mistakes that I would consider clarifying or correcting:

Line 158: $Z^\top Z$ instead of $Z Z^\top$ in formula for $\hat{y}$.

Line 161: It seems “predictor B” is associated with linear regression, not with linear attention. 

Line 162: Perhaps, the authors meant one step **of** gradient descent?

Lines 202, 771, 775 and other throughout the paper: the authors interchangeably use bold **0** both as vector and a scalar, occasionally in the same line, which is confusing.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows that Gated Linear Attention (GLA) can implement Weighted Projected Gradient Descent (WPGD) algorithms. Furthermore, the gating mechanism in GLA allows the in-context samples to come from distinct tasks. This paper also characterizes the loss landscape of WPGD and one-layer GLA.

### Strengths
1. This paper establish the equivalence between GLA and WPGD. The authors show that GLA can weight the context window through gating, so GLA can learn from non-IID in-context samples, while linear attention can't.
2. This paper characterizes the loss landscape of GLA and WPGD and shows that the WPGD minimizer is unique.

### Weaknesses
1. The attention matrices are restricted in certain form (eq (8), line 201-203). In actual training setting of GLA, the learned attention matrices may not have such forms. It's not sure whether GLA implement WPGD when attention matrices don't have these restricted forms.

2. The token embeddings also have restricted forms (line 426). It is not clear whether GLA can learn the contextual features if the token embeddings are learnable parameters.

### Questions
1. Line 486 says that Assumption B ensures that any $\omega$ in W can be achieved by an appropriate choice of gating parameters. i'm not sure this statement is correct. If the number of tasks $K$ is larger than the dimension of $w_g$ (the trainable parameter of the gating function), the above statement seems to be wrong.

2. Theorem 2 states that an L-layer GLA implements L steps of WPGD. The question is when L is large enough, whether the L-layer GLA found the better predictor than the one-layer GLA. Can Theorem 2 show the advantage of the deeper models?

3. Are there more experiments of multi-layer GLA?

### Soundness
3

### Presentation
2

### Contribution
2
