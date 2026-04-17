# A Random Matrix Analysis of In-context Memorization for Nonlinear Attention

- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Attention mechanisms have revolutionized machine learning (ML) by enabling efficient modeling of global dependencies across inputs.
Their inherently parallelizable structures allow for efficient scaling with the exponentially increasing size of both pretrained data and model parameters. 
Yet, despite their central role as the computational backbone of modern large language models (LLMs), the theoretical understanding of Attentions, especially in the nonlinear setting, remains limited.

In this paper, we provide a precise characterization of the *in-context memorization error* for *nonlinear Attention*, in the high-dimensional regime where the number of input tokens $n$ and their embedding dimension $p$ are both large and comparable. 
Leveraging recent advances in the theory of large kernel random matrices, we show that nonlinear Attention typically incurs higher memorization error than linear regression on random inputs. 
However, this gap vanishes, and can even be reversed, when the input exhibits statistical structure, particularly when the Attention weights align with the input signal direction.
Our theoretical insights are supported by numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper gives a high-dimensional characterization of the in-context 
memorization error (training MSE of the best ridge linear probe fitted on 
the same context) when features come from a single-head nonlinear 
attention block with fixed query/key/value weights. Under a Gaussian 
signal+noise model for tokens $X=\mu y^\top+Z$ and a full-plus-low-
rank structure for $W_K^\top W_Q=I+w_Kw_Q^\top$, the authors 
obtain a closed deterministic $9\times 9$ system whose solution yields 
the limiting in-context training error $\bar E_A$ (Theorem 1). This 
enables clean comparisons against linear regression (LR) on raw $X$ 
(Proposition 2).
Empirically/theoretically they find: with random inputs ($\mu=0$), 
nonlinear attention often underperforms LR; but with structured inputs 
and alignment between attention weights and $\mu$, the gap vanishes 
or reverses. They also show memorization critically requires a linear 
component in $f$ (nonzero $a_1$).

### Strengths
The authors analyze a random feature model with a related regression task, where the feature map has structural properties shared with attention layers. They also compare linear and non-linear attention, which is a topic that may have impact on practical debates on the usefulness of non-linearities in attention.

### Weaknesses
In my understanding, the paper does not really study attention, as the crucial non-linearity (usually a row-wise softmax) is applied entry-wise (while Remark 5 in Appendix B argues that this is not important, it will be as soon as the attention weights are learned - see next comment - but I am happy to be disproven). Additionally, not learning the attention weights (keys and queries in particular) makes the model a random feature model (albeit structured) in disguise (as discussed in lines 139-142). 

The phenomenology observed in Sections 4.2 and 4.3 is standard of random features regression: adding a non-linearity introduces additional noise. See for example Mei and Montanari '21. It is not clear to me which observations here are specific to attention layers, and which are just random feature ones.

### Questions
line 55: I find it a bit unfair to call "stylized models" the works Tiberi '24, Troiani '25 and Rende '24. The phrasing of this line and the next paragraph seems to suggest that this paper goes beyond such stylized models, while model line 173 is stylized as well.

line 200: why does it make sense to regularize the learned weights as || W^T_V w ||^2 instead of ||w||^2? I understand that this simplifies the expressions by virtually removing the value weights from the solution eq (5), but it is still quite a peculiar choice. Does it have some intuitive meaning?

Def 3: it is not clear to me why the error is called "in-context memorization". For me, in-context means that each input sample to the attention specifies the task that the attention needs to solve. Here I only see a classic regression task in which, independently for each sample {x,y}, one just minimizes the L2 error between y and a random feature predictor applied to x. I really fail to see the interpretation as in-context: could you try to explain this better? For comparison, Lu et al '25 truly provides an in-context learning task in my opinion.

Proposition 1: is this a direct application of (Pennington & Worah, 2017)?

Regarding the element-wise non-linearity, I wanted to mention the recent paper https://arxiv.org/pdf/2510.06685 (appeared after the submission deadline, so I am not asking for a comparison, just mentioning it for completeness), where a treatment of the true softmax seems possible.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies one layer of attention with element-wise non-linearity. Specifically, the authors consider the problem of learning a mixture of two isotropic gaussians using either a linear model or with one layer of attention by training only the values and keeping the key-query product equal to identity plus (untrained) rank one spike. The memorization error is studied through a rigorous deterministic equivalence in the proportional regime of large number of samples and large embedding dimension, which are then validated by numerical experiments.

### Strengths
The paper is well written and easy to follow, has a clear exposition and shows the implications of its assumptions clearly. The theoretical study of attention networks is of current interest, and results in this direction are valuable for bridging the gap between empirical performance and principled understanding. I believe the theorems to be correct and their derivations clearly written.

### Weaknesses
The specific setting under analysis is not particularly relevant: the authors study a single layer of attention without even training the key and query weights and without row-wise softmax activation. Additionally, I am not sure the task at hand is in fact in-context learning and I would like this aspect to be clarified by the authors. I don't think paper has enough technical novelty to compensate for the shortcomings in its settings. In particular I think that the element-wise instead of row-wise activation, as well as the strong assumption on the query-key weight matrix make the technical contributions somewhat modest. At its current stage, this manuscript does not offer much more than a random matrix theory exercise, and I would like to see the dependence on all the relevant parameters in the model fully explored.

Minor issue:
Figure 2 contains the lines already presented in Figure 1, they could be combined.

### Questions
1. Could you clarify in which sense is your paper studying an in-context learning task?
2. Would you give precise insights over what is the technical novelty of your work with respect to existing literature on the high-dimensional analysis of attention models?
3. Would you compare with [Cui, Behrens, Krzakala, Zdeborova 2024]?
4. Do you know how would your results change if you consider row-wise softmax attention (even just experimentally)?
5. In your analysis you never find a setting where attention outperforms a linear model. Would you comment on this, in particular in relation with not training the key-query matrix?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies the training error of a linear ridge regression problem that uses pointwise nonlinear attention as a feature extractor for a sequence of tokens $X$. The analysis assumes the nonlinear attention matrices follow a full-rank plus low-rank structure. The work characterizes the regression error in the high-dimensional asymptotic limit, where the number of tokens $n$, their embedding dimension $p$, and the inner dimension of attention product matrices $d$ all blow up. The data is modeled using a signal-plus-noise model where each token $x_i$ consists of a signal component $y_i\mu$ and isotropic noise $z_i$. This error is then compared to that of a standard linear regression model.

### Strengths
The work's primary strength is its technical contribution. It builds on classical random matrix tools, often used in benign overfitting studies of linear regression, by successfully deriving a deterministic equivalent for the Gram matrix using the non-linear attention kernel in a high-dimensional asymptotic regime.

### Weaknesses
**Missing key related work**: The motivation to study this specific model is not well-introduced. The introduction states that a key challenge is the nonlinearity of attention, which is a fair point. However, it then proceeds to cite prior work that largely “reduces attention to generalized linear models”, while failing to mention or engage with several recent works that directly study nonlinear softmax attention [1, 2, 3, 4, 5, 6] (see [2] for a more comprehensive list of related work).


**Issues with framing the problem**: The introduction also introduces the term "in-context memorization" without sufficiently defining the problem. This points to a larger, more fundamental problem with the paper's framing. The problem setup, in its current form, does not appear to be an "in-context" task. In Def 2, each token $x_i$ has a signal component $y_i\mu$ plus noise, and the goal is to isolate signal from noise and map this to $y_i$. The signal direction $\mu$ is fixed across all contexts, hence the task remains the same across contexts. In-context learning, by contrast, typically implies that the underlying task or function changes with the context (e.g., learning different linear regressions or Markov chains from the given context). Since the task here remains consistent, the framing as "in-context" is inappropriate.


The setup is more of the flavour of classical regression task (which the authors correctly identify as a baseline in (12)) where $n$ (x, y) iid pairs are drawn from a distribution. The work is studying the training error of ridge regression on a signal-plus-noise model using attention as a feature extractor, which is more similar to the line of work in [7] than to the ICL literature.


Furthermore, the term "in-context memorization" has a specific meaning in the ICL literature, where it typically refers to a model memorizing a finite set of tasks from its pretraining data rather than generalizing to unseen ones [8, 9, 10, 11]. The paper defines it (Def 3) as the standard training error of the linear probe. Calling this "in-context memorization" is a misrepresentation of the problem being studied.

Finally, this signal-plus-noise token learning framework has been studied in other theoretical analyses of attention [1, 3] (also not discussed when introducing the data model).

**References**

[1] Deora et al. (2023) On the Optimization and Generalization of Multi-head Attention.

[2] He et al. (2025) In-Context Linear Regression Demystified: Training Dynamics and
Mechanistic Interpretability of Multi-Head Softmax Attention

[3] Vasudeva et al. (2024) Implicit Bias and Fast Convergence Rates for Self-Attention

[4] Chen et al. (2024) Unveiling Induction Heads: Provable Training Dynamics and Feature
Learning in Transformers

[5] Tarzanagh et al. (2023) Transformers as SVMs. 

[6] Huang et al. (2023) In-context convergence of Transformers

[7] Boncoraglio et al. (2025) Bayes optimal learning of attention-indexed models. 

[8] Singh et al. (2023) Transient nature of in-context learning

[9] Park et al. (2024) Algorithmic phases of icl 

[10] Lin et al. (2024) Dual operating modes of icl

[11] Raventos et al. (2023) Preptraining task diversity and the emergence of non-bayesian icl for regression

### Questions
Please see weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a theoretical characterization of in-context memorization error for (nonlinear)attention mechanisms using the tools from random matrix theory (RMT). The authors analyze attention in the high-dimensional regime where the number of tokens ($n$) and embedding dimension ($p$) are both large and comparable. 

Under a signal-plus-noise input model, authors have derived the deterministic equivalents for the memorization error and compare nonlinear attention with linear regression. The key finding is that while nonlinear attention generally underperforms linear regression on random inputs, this gap vanishes or reverses when inputs has statistical structure and attention weights align with the signal direction.

### Strengths
1. This work addresses a crucial gap by providing the precise RMT analysis of nonlinear attention on structured inputs. The extension of deterministic equivalents to handle the generalized sample covariance matrix (Proposition 1) could have broader applications.

2. This paper tries to bridges theory and practice by explaining why certain nonlinearities  (those with strong linear components) work better in transformers.  The finding that the first Hermite coefficient $a_1$ is crucial for memorization performance  
can provides actionable design guidance.

3. The comparison between nonlinear attention and linear regression across different regimes  (varying SNR, embedding dimensions, regularization) is thorough.

### Weaknesses
1. The analysis performed in this paper is restricted to single-head attention with rank-one perturbations (Assumption 1)  and binary classification under a Gaussian signal-plus-noise model (Definition 2).  However, modern transformers use multi-head attention with more complex weight structures, and real-world data rarely follows such clean statistical models.  These disparities between the analyzed settings and practical transformers is quite significant .

2. Assumption 2 requires centered nonlinearities with $\mathbb{E}[f(\xi)] = 0$  and specific Hermite coefficient constraints.  
While the authors claim this can be achieved by subtracting constants,  such modification could affect the attention mechanism’s behavior in ways not captured by the analysis.  

3. While the paper includes numerical experiments confirming theoretical predictions,  these are all on synthetic data following the exact assumed model.  There is no validation on real data or even on more realistic synthetic settings.  The mention of *pretrained GPT-2 weights* experiments appears only briefly,  without sufficient details.


##### **Missing Prior Work** #####


The role of attention and FFN nonlinearity for in-context learning task [1, 2], and the impact of LayerNorms and GELU in the model's representational dynamics [3,4] 


[1]  Wang et al., How do nonlinear transformers learn and generalize in in-context learning? ICML 2024

[2] Cheng et al., Transformers implement functional gradient descent to learn non-linear functions in context, ICML 2024

[3] Brody et al., On the expressivity role of layernorm in transformers’ attention, ACL 2023

[4] Jha et al., AERO: Entropy-Guided Framework for Private LLM Inference, 2025

### Questions
1. How fundamental are the technical barriers to extending the analysis to multi-head attention and realistic softmax settings?  
Could authors provide a discussion on whether the key qualitative insights (e.g., importance of linear components, structured input advantage)  still hold when moving beyond hardmax and single-head formulations?

2. How sensitive are the conclusions made in this paper to deviations such as sub-Gaussian or heavy-tailed distributions,  which are common in NLP taks? Would these affect the memorization dynamics or theoretical rank behavior?

3. The analysis suggests that $\cos(t)$ performs poorly when $a_1 \approx 0$,  but periodic activations sometimes succeed in practice.  
Could authors clarify this discrepancy and discuss regimes where low-$a_1$ activations might help?  Additionally, can authors  provide finite-sample bounds or insights on how quickly your asymptotic results converge  for realistic transformer sizes?

4. The results are framed as  ``in-context memorization,"  but the connection to broader in-context learning phenomena in LLMs remains unclear.  How does the analyzed memorization behavior relate to few-shot learning performance observed in practice?

### Soundness
3

### Presentation
3

### Contribution
3
