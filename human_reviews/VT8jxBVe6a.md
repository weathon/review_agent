# $\mathrm{BP}(\lambda)$: bias-free online learning via synthetic gradients

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 3

## Abstract
Training recurrent neural networks typically relies on backpropagation through time (BPTT). BPTT depends on forward and backward passes to be completed, rendering the network locked to these computations before loss gradients are available. Recently, Jaderberg et al. proposed synthetic gradients to alleviate the need for full BPTT. In their implementation synthetic gradients are learned through a mixture of backpropagated gradients and bootstrapped synthetic gradients, analogous to the temporal difference (TD) algorithm in Reinforcement Learning (RL). However, as in TD learning, heavy use of bootstrapping can result in bias which leads to poor synthetic gradient estimates. Inspired by the accumulate $\mathrm{TD}(\lambda)$ in RL, we propose a fully online method for learning synthetic gradients which avoids the use of BPTT altogether: *accumulate* $BP(\lambda)$. As in accumulate $\mathrm{TD}(\lambda)$, we show analytically that accumulate $\mathrm{BP}(\lambda)$ can control the level of bias by using a mixture of temporal difference errors and recursively defined eligibility traces. We next demonstrate empirically that our model outperforms the original implementation for learning synthetic gradients in a variety of tasks, and is particularly suited for capturing longer timescales. Finally, building on recent work we reflect on accumulate $\mathrm{BP}(\lambda)$ as a principle for learning in biological circuits. In summary, inspired by RL principles we introduce an algorithm capable of bias-free online learning via synthetic gradients.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes BP(λ) which combines the works of Jaderberg et. al. and Van Seijen et al. A derivation and architecture to perform this combination is presented with validation.

### Strengths
The authors have presented a well written and understandable argument for their paper to be accepted.

### Weaknesses
This paper's related work overly focuses on building upon the work of Jaderberg et. al., which is mentioned both in the abstract and several times in the main text. Is there a reason why Jaderberg et. al. needs to be mentioned in the abstract?

The paper does not provide a related work section in the main text. It is difficult to evaluate this work without a related work section placing it in the context of other work in the field, and what is the new and novel contribution of this work.

The paper is incremental, and of limited novelty.

The paper is missing supplementary materials which would be a nice place to give significant amount of additional information regarding the work which would greatly help in evaluating and making a confident decision on the paper.

Given the purpose of the introduction is to craft the story, the scaffolding, and also the related work in this paper, it is difficult to trust the related work presentation in the introduction as it is comingled with the story.

What is the purpose of the sentence, "Recently, these properties have led neuroscientists to speculate that synthetic gradients are computed at the systems-level in the brain, explaining a range of experimental observations (Marschall et al., 2019; Pemberton et al.,
2021; Boven et al., 2023)?" I see no connection or relevance at all to the rest of the paper. In addition, are the authors aware that neurons in actual brains of living organisms do not behave much at all like neurons in artificial neural networks?

There is a significant, unmistakable, and serious gap between the authors pointing and hinting towards some connection between their work and biological neural networks. For example, the architecture proposed in, "Evaluating biological plausibility of learning algorithms the lazy way," are the authors aware of the large differences between that architecture and architectures built upon artificial neural networks?

This is the first time I am hearing of the assertion, "As the original authors note, this is highly reminiscent of temporal difference (TD) algorithms used in Reinforcement Learning (RL) which use bootstrapping for estimating the future return (Sutton & Barto, 2018). Indeed, in their supplementary material Jaderberg et al. extend this analogy and introduce the notion of the λ-weighted synthetic gradient, which is analogous to the λ-return in RL. However, λ-weighted synthetic gradients were only presented conceptually and it remained unclear whether they would be of practical benefits as they still require BPTT." Without reading the related work, nor the citation of Sutton & Barto, I hazard a guess what is meant is as follows: the significant problem in reinforcement learning is learning from state-action pairs or full trajectories, TD learning is some sort of optimal bias-variance tradeoff associated with the hyperparameter $\lambda$. Now all of this is stuffed into some neural network, as neural networks can be used to approximate functions. So the neural network becomes a fast computation mechanism for doing the bias-variance tradeoff. Am I correct in this understanding?

Could the authors explain clearly and exactly what is meant in the paragraph:
"In this study, inspired by established RL theory, we make conceptual and experimental advancements on λ-weighted synthetic gradients. In particular, we propose an algorithm for learning synthetic gradients, accumulate BP(λ), which mirrors the accumulate TD(λ) algorithm in RL (Van Seijen et al., 2016). Just as how accumulate TD(λ) provides an online solution to learning the λreturn in RL, we show that accumulate BP(λ) provides an online solution to learning λ-weighted synthetic gradients. The algorithm uses forward-propagating eligibility traces and has the advantage of not requiring (even truncated) BPTT at all. Moreover, we demonstrate that accumulate BP(λ) can alleviate the bias involved in directly learning bootstrapped estimations as suffered in the original implementation (Jaderberg et al., 2017)."

Could the authors explain clearly and exactly, for practitioners in the field of Machine Learning which is a subfield of AI, what is the relevance of, "Next, we touch upon accumulate BP(λ) as a mechanism for learning in biological circuits."

Do the authors believe that ICLR is the correct venue for this work?

The reviewer understands that the contribution of this work is providing synthetic gradients along the lines of Jaderberg et. al. with the BP(λ) approach which is done by mixing the work of Jaderberg et. al. and Van Seijen et. al. Do the authors have further technical contributions in this work?

### Questions
I have no questions for the authors.

### Soundness
3 good

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
This paper provides a new algorithm for computing synthetic gradients for training recurrent networks. Specifically, if one use BPTT to compute the gradient, it will be very time consuming to do backpropagation, especially if the sequence is very long. Existing work considers synthetic gradient, which predicts the future gradients based on the current state. However, this method may contain bias in the prediction. Therefore, this paper improves the existing method, by borrowing the idea of $TD(\lambda)$, which learn $\lambda$-weighted synthetic gradients that can alleviate the bias problem. This new method is empirically verified on synthetic datasets like sequential MNIST and copy-repeat tasks.

### Strengths
Originality: I think the idea of using $TD(\lambda)$ method from RL to do better synthetic gradient estimation is a very interesting idea, and also makes sense. Therefore, the originality is good. 

Quality: I think the quality of the theoretical analysis in this paper is good. I guess most of the theorems in the appendix is a minor modification of the original accumulate TD(\lambda) analysis of Van Seijen et al. 2016, but it is good to know that the method proposed in this paper has nice theoretical guarantees. However, the quality of empirical study is limited, as I will mention below. 

Clarity: I think the paper is easy to follow. 

Significance: I think the paper has limited significance for the reason below.

### Weaknesses
Overall I like the idea of this paper, but the main limitation of this paper is the empirical results are very limited. The authors only consider tasks like Sequential MNIST and copy-repeat, which are extremely preliminary. If the authors cannot demonstrate that their new methods can be used in more realistic settings, it seems to me that the paper is not ready to be published in a top conference like ICLR. 

Moreover, the paper works on RNN, but right now foundation models become the dominant topic in AI. I am not sure whether this technique can be used for foundation models -- if so, it would greatly improve the quality and significance of the paper. Moreover, as the author mentioned in Section 6, this method essentially assumes that the future gradient is almost completely dependent on the current state, and is not heavily related to the future inputs. Otherwise, the synthetic gradient will not be useful. However, in practice, it is hard to imagine a good use case of this kind. It seems that this underlying assumption has limited applications.

### Questions
I do not have additional questions, but I will be happy to adjust my score, if I misunderstand anything in this paper.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The present paper tackles online gradient computation in recurrent neural networks (RNNs). Building upon synthetic gradients [1] and the accumulate $TD(\lambda)$ algorithm [2], the authors propose an improved version of synthetic gradients which totally obviates (beyond a single step)  the use of backpropagation through time (BPTT) within the truncation window to compute the gradient synthesiser targets as well as the RNN weight updates which they call $BP(\lambda)$. The core ingredient of the $BP(\lambda)$ algorithm is the combined use of temporal difference (TD) errors (reprospective / backward error signals) and of eligibility traces (prospective / forward error signals) to train the gradient synthesizer, where $\lambda$ quantifies the amount of forward error flow, while the weight update used for the RNN remains (given the gradient synthesiser prediction) the same as that of [1] (Alg. 1). The results of the paper subsumes theoretical and experimental contributions:

- On the one hand, it is theoretically shown (Theorem A.1.) that the $BP(\lambda)$ synthesiser weight updates (Eq. 8) approximate those of the $\lambda$-weighted synthetic online gradient algorithm (the "online $\lambda$-SG algorithm", per Section A.5.2). The online $\lambda$-SG algorithm, the baseline of the $BP(\lambda)$ algorithm, is a online synthetic gradient-based algorithm which uses interim $\lambda$-weighted synthetic gradient as a synthesizer target. By comparison with the offline counterpart of the $\lambda$-SG algorithm (Section A.5.1), the $\lambda$ parameter of the $BP(\lambda)$ algorithm can be construed as the relative weight of the observed gradients versus the bootstrapped / self-predicted gradients for the target computation (Eq. 9).

- On the other hand, the benefits of the $BP(\lambda)$ is demonstrated across three different setups. First, it is shown on a toy task with a linear RNN that when the weights of the RNN are frozen with only the weights of the gradient synthesiser being learned, synthetic gradients end up being perfectly aligned with BPTT gradients, and this alignment improves with increasing values of $\lambda$ (Fig. 2), which is expected as it implies lesser reliance on bootstrapping. These observations persist in the setting where both the RNN weights and the gradient synthesiser are learned (Fig. 3, whilst being weaker as the gradient synthesiser weights have to keep track of the moving RNN weights) up until  $60$ timesteps. Then two more complex tasks are considered (sequential MNIST and copy-repeat) along with an LSTM model where it is shown that  $BP(\lambda)$ with $\lambda=1$ outperforms truncated BPTT and SG along with truncated BPTT altogether on both tasks (Table 2, Fig. 4, Fig. 5).




*[1] Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver, and Koray Kavukcuoglu. Decoupled neural interfaces using synthetic gradients. In Pro- ceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1627–1635. JMLR. org, 2017*

*[2] Harm Van Seijen, A Rupam Mahmood, Patrick M Pilarski, Marlos C Machado, and Richard S Sutton. True online temporal-difference learning. The Journal of Machine Learning Research, 17 (1):5057–5096, 2016*

### Strengths
- The paper is very clearly written and can be followed end-to-end down to the appendix.
- Building upon the parallel between credit assignment in RL ($TD(\lambda)$) and credit assignment in RNNs ($BP(\lambda)$) is extremely interesting.
- The mathematical proofs are neat.

### Weaknesses
- Getting rid of BPTT for the target computation comes at the cost of using RTRL which is cubic in the dimension of the hidden layers.
- $BP(\lambda)$ does not work on long temporal window ($T=60$ is the maximal time horizon explored if I am not mistaken).
- $BP(\lambda)$ was not tested on more complex tasks (e.g. ListOps, sCIFAR).

### Questions
- We clearly understand from the paper that $BP(\lambda)$ intuitions are strongly grounded in $TD(\lambda)$, with salient properties of the latter be transferred to the former by design. Yet, the resulting algorithm trades the use of BPTT for RTRL-based target computation with a cubic cost. Here is a probably naive question: assuming I am given a sufficient memory / compute budget, why would I use $BP(\lambda)$ over vanilla RTRL to train an RNN in an online fashion?

- To alleviate this cost, you mention an interesting direction in the discussion which would consist to learn gradients for low dimensional embedding of the activities rather than the activities themselves. Is it a direction you already explored? Do you already have some experimental results on this? Also, wouldn't it call for the use of "spatial" backprop across the embedding layer? 

- You justify why you did not test $BP(\lambda)$ on ListOps. Could you please elaborate on the reasons why you did not test it beyond sequential MNIST (e.g. sCIFAR)? Also why not with more than $60$ timesteps? Is it due to the algorithm itself or the models under investigation? For instance in a recent paper [1], it was shown RTRL could be used to train a class of models up until 1000 timesteps.

- Probably another candid question: would there be any practical advantage in using $BP(\lambda)$ to train feedforward nets? 



[1] Zucchet, N., Meier, R., Schug, S., Mujika, A., & Sacramento, J. (2023). Online learning of long range dependencies. arXiv preprint arXiv:2305.15947.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on increasing the efficiency of recurrent networks in training amenable to learning with long-term memory. Classically, one result uses traction of gradients to resolve this issue. Another method proposed is using synthetic gradients i.e. learning them using another network through backpropagated gradients and bootstrapping. The authors propose another method for computing synthetic gradients by drawing a parallel in reinforcement learning which relies on an auxiliary network, updated online which takes the RNN's hidden state and generates an estimate for the future gradients.

### Strengths
The idea is a natural problem to consider and the approach is reasonable. The idea is also well explained.

### Weaknesses
1. The best performance highlighted in Table 2 is by BP(1) which requires the computation of full gradients. Additionally. BP(0.5) which in my interpretation is more reflective of the proposed algorithm since it relies on bootstrapping, whereas BP(1) does not. However. BP(0.5) does not clearly offer the best performance in Table 2.  For example, on the toy task BPTT + SG for n = 2,3 or on MNIST for BPTT n= 3 and onwards. 

2. There is no run-time comparison even though the main focus of the paper is the tradeoff between accuracy and computational complexity. 

3. The authors provide no code and there are different points when one would not expect a drop in average performance. However, one would expect that if too few initial conditions are used, the results in Table 2 cannot be interpreted confidently.  For example, BPTT + SG performance drops for n=4 and n=5.

### Questions
1. Can you please Include a run-time comparison for Table 2?

2. Can you please include the code for the numerics?

3. From equation (18) and  (8), from my interpretation the difference between $\theta^{BP}_{\lambda}$ and $\theta^{\lambda}_t$ converges to $0$ only because both sequences converge to the same **constant** sequence.  Am I missing something?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
