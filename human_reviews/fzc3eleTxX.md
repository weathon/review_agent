# $\sigma$-PCA: a unified neural model for linear and nonlinear principal component analysis

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Linear principal component analysis (PCA) is marked by orthogonality and ordering of variances -- its conventional nonlinear counterpart is marked by neither.
    To yield a useful output, conventional nonlinear PCA requires, as a preprocessing step, whitening. This makes the overall transformation akin to independent component analysis (ICA). But, as a side effect of whitening, the overall transformation becomes non-orthogonal and the variances become non-estimable. Conventional nonlinear PCA thus lacks the two distinctive characteristics of PCA.
    To bridge the disparity, we propose $\sigma$-PCA, a unified neural model for linear and nonlinear PCA as single-layer autoencoders. The key is modelling the variances as separate parameters.  With our model, we show that whitening is not required, and nonlinear PCA can retain both orthogonality and ordering of variances, becoming a special case of ICA for when the overall transformation is assumed orthogonal.  And so where linear PCA fails to separate components into orthogonal directions when their variances are similar, nonlinear PCA can. And when the overall transformation is non-orthogonal, two isolated layers of nonlinear PCA can perform conventional ICA. In the middle between linear PCA and ICA, we carve out a place for nonlinear PCA as a method in its own right.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
A new framework for understanding both linear PCA and nonlinear PCA as single-layer autoencoders is introduced. This allows one to reconcile the fact that linear PCA decomposes data into an orthogonal bases with ordered eigenvalues representing coefficients in that basis, whereas non-linear PCA decomposes data into non-orthogonal bases with coefficients that cannot be estimated. In this framework, both PCA and nonlinear PCA retain orthogonal bases and ordered variances. Independent component analysis is obtained as a special case where the variances have magnitude 1. Nonlinear PCA is discussed as a middle ground in the context of PCA and ICA. Nonlinear PCA is applied to CIFAR10 and some synthetic timeseries.

### Strengths
- Understanding connections between the various classical unsupervised methods such as PCA, ICA, autoencoders, nonlinear PCA, etc. is an interesting and important research direction, especially if it helps us design better variants of known existing algorithms. 
- The paper cites a large amount of classical literature (I only have two minor additions below from 2001 and 2008), which I think is good because it mitigates against the increasing risk that old known results are forgotten.

### Weaknesses
I am excited by this work and think it has potential. I state my concerns below bluntly and directly and do not mean to discourage the authors. If the authors can provide a sound rebuttal or if I have overlooked something, I am happy to raise my score accordingly.

- The biggest weakness, in my opinion, is the imprecision/informality of the presentation. I am not sure whether the derivation is correct/rigorous/precise, because at each step and equation, it is not clear whether the original model is modified to accommodate the step in question, whether an approximation is applied, and whether the approximation or modification is valid. There is no discussion or attempt to quantify the effect of these various simplifications, or even formally state what these simplifications are. 
    - As far as I can tell, my confusion begins at equation (5), where the weights of the encoder and decoder are untied. I am not sure why the authors didn't just start with a model that unties the encoder and decoder (which seems to be the model which is analysed) in place of equation (4). (A minor aside, it is not clear why we "drop the expectation").  
    - In equation (10), we obtain a loss, which is ostensibly an object that should be minimised. What then does the stop-gradient operator (sg) mean, in terms of a loss that should be minimised? It is clear that procedurally that when applying gradient descent to a loss (10), the gradient step should involve an appropriate modification of the gradient of a modified equation (10). But what does this mean mathematically in terms of the loss?
    - Equation (12). For some reason the authors drop a certain Hadarmard product to obtain (12) from (11). It is not clear what "it does also work without it" (What is the first "it" referring to, what does "work" mean, and what is the second "it"?). Appendix F.4 is not helpful in this respect. I am not sure what the quality of the approximation is here, even intuitively or qualitatively.
    - Equation (13). I suspect the authors mean to say  that $W^\top W \approx I$ (because they mention using a regulariser or penalty rather than a hard constraint). There is no discussion around the approximation error in equation (13).
    - Equation (14). Suddenly the previously mentioned dropped Hadarmard product (or something similar? it is not clear) is reintroduced. I am not sure why, if this is an approximation, or if this modifies the original learning objective. Even more confusing, equation (14) is stated as an equality.
    - Given all the above points, I am not sure whether the paper lives up to the title of a "uniformed neural model for linear/nonlinear PCA", because it seems as though the authors change terms/factors one at a time, here and there until they arrive at the result they want, without justification of the steps involved.

### Questions
Questions:
- Top of page 4. " we will untie the weights of the encoder We and the decoder Wd, drop the expectation, and multiply by 1/2 for convenience," Why drop the expectation? This appears to be purely to remove a symbol from the notation, because under extremely mild conditions the gradient will pass through the expectation anyway. 
- Also relating to the above, what is the reason for untying the weights of the encoder and decoder?
- Under equation (10) there is a discussion around $\Sigma$, which I do not understand. In particular, it is mentioned that $\Sigma$ need not be differentiable, however $\Sigma$ is a *parameter*, not a *function*. I am not sure what differentiability of $\Sigma$ means (differentiability with respect to what?). Is it possible that the authors meant to say that $\mathcal{L}$ need not be differentiable with respect to $\Sigma$?
- What should be the key takeaway message from the two experiments conducted in section 4? I see that you tried PCA, nonlinear PCA, and ICA, but how should one assemble these qualitative figures into a concrete message for the reader?

Related work: Could the authors comment on the relevance of (probabilistic) exponential family PCA?
- A Generalization of Principal Component Analysis to the Exponential Family, NeurIPS 2001
- Bayesian Exponential Family PCA, NeurIPS 2008

### Soundness
1 poor

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
This work proposes $\sigma$-PCA, which is a variant of non-linear PCA that can distinguish the principle components with the same variances, and maintain orthogonality without whitening process. The author claims that $\sigma$-PCA can maximise both variance and statistical independence.

### Strengths
1. The paper is clearly written and well-structured.

2. The paper proposes a new variant of non-linear PCA, which emphasizes on latent reconstruction, and can seperate the components with the same variances, which cannot be achived with linear PCA. 

3. The experiments shows that $\sigma$-PCA can lead to more recognizable filters and can also recover the signals with the same variances.

### Weaknesses
1. I feel some claims are not well supported, for instance, the author claims that $\sigma$-PCA can maximise the statiscal independence, which is the purpose of ICA. Does it mean we can also use it to replace FastICA? I also can't find why $\sigma$-PCA can maximise statiscal independence in the paper, although it is claimed to be.

2. For Eq. (9), why the stop-gradient operator is introduced, I found it is a little bit arbitrary, my understanding is that is is aimed to solve the issue that the gradient of encoder is zero when $W_{e}=W_{d}$. But why this is rational to introduce stop-gradient operator here? please clarify it.

3. The experiment only contains qualitative results, adding some quantitative results would be appreciated.

### Questions
1. How is $\sigma$-PCA connect with kernel PCA. What is the advantages and disadvantages?

2. In the time-signal experiment, why non-linear PCA can't recover the signals but 2-layer non-linear PCA can? What is the advantages and disadvantages of $\sigma$-PCA compared with fastICA?

3. How the statistical independece is maximized in $\sigma$-PCA, does that mean it also guarantees identifiability like ICA?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tries to unify the linear and nonlinear PCA as single-layer autoencoders, in order to keep the orthogonality and ordering of variances that Linear principal component analysis (PCA) maintains.

### Strengths
Trying to use single-layer autoencoders to unify the linear and nonlinear PCA may be a good attempt, so that even the nonlinear PCA can keep the orthogonality and ordering of variances that Linear principal component analysis (PCA) maintains.

### Weaknesses
1. The paper consistently makes incorrect assumptions, particularly in the context of non-linear PCA. In traditional non-linear PCA, the specific non-linear mapping functions, such as tanh, are typically unknown and considered part of the modeling process. The paper, however, specifies these mappings explicitly, and deviates their new method based on this.


2. While linear PCA maintains orthogonality and orders variances to derive principal components, the insistence on preserving these characteristics in non-linear PCA might seem counterintuitive. Understanding the advantages of maintaining orthogonality and ordered variances in the non-linear context is crucial. Clarification on why this approach is beneficial by  keeping the orthogonality and ordering of variances?

3. The paper appears to leverage complex neural networks, such as single-layer autoencoders, to unveil linear and non-linear patterns in observed data. This approach raises concerns about the departure from the traditionally interpretable nature of principal component analysis (PCA). Even with the effectiveness of neural networks in capturing underlying patterns, explaining the significance of each component becomes challenging, potentially compromising the simplicity and interpretability associated with PCA.

### Questions
1.  In the non-linear PCA, the author specifies the non-linear mapping, such as tanh. Isn't the non-linear mapping some function that we should not know in the non-linear PCA? Is the non-linear mapping some function that we should know in the non-linear PCA? 

2. We all know that the Linear principal component analysis (PCA) maintains the orthogonality and ordering of variances, and this is how we get the PCs. But for non-linear PCA, why do the authors insist on keeping the orthogonality and ordering of variances? What advantages will it bring to keep the orthogonality and ordering of variances?

3. It seems that the authors use complex neuron networks, even just single-layer autoencoders, to discover the linear/non-linear patterns of the observed data, if I understand correctly. Isn't a bit far away from the attractive easily explanatory property of of the principal component analysis (PCA)? 
even single-layer autoencoders can perfectly discover the underlying patterns of the observed data, is it easy to explain which component is more important by using neural networks?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presented $\sigma$-PCA, which is a unified neural model for both linear PCA and nonlinear PCA.

$\sigma$-PCA dropped the requirement of whitening and both the orthogonality and the variance order are preserved in the nonlinear version.

The nonlinear version can handle the orthogonal case with similar variance which linear PCA cannot handle. It can also perform ICA to deal with the non-orthogonal case.

Experiments on image patches and time signals show some advantage compared with linear and nonlinear PCA.

### Strengths
1. One of the key contribution for this work is that it draw connection among a wide range of literature for nonlinear PCA. The proposed objective can be easily adapted to those cases.
2. In particular for linear PCA, nonlinear PCA, and ICA, comprehensive discussion was provided on which case which models will coincide.
3. Compared with ICA, since $\sigma$-PCA dosen't require whitening, then the orghogonality is preserved.
4. Compared with conventional PCA, $\sigma$-PCA unified the linear and nonlinear case which can handle the case when components have similar variance.

### Weaknesses
1. The key idea that this work claim is to explicitly model the varience, which conventional nonlinear PCA methods utilize whitening to bypass. However, the idea of modeling th variance is already explored in the literature (the author also mentioned this in the relative work section).
2. The necessity for introducing $\mathbf{\Sigma}$ is unclear.
2. The nonlinearity is another major concern. The nonlinear function discussed in this literature is rather simple, especially compared with nonlinear ICA literature. Also in the experiments together with the additional experiment results in the appendix, all those $h$ are fairly simple and may not suitable for more complex data.
3. Some experiment result is not very straightforward for showing the advantage of the proposed method. Especially for the image patches dataset, it's quite hard to tell the quality for the method purely from the figures showed in the paper. Also detailed analysis and discussion on the experiment result is missing.
4. Anothe concern is that the ICA mentioned in this work is linear ICA, however, for the nonlinearity, there are nonlinear ICA methods, which also emphasis the latent reconstruction.

### Questions
1. For the derivation of eq 5 to 8, it's a little bit confusing if the optimizating process doesn't distinguish encoder and decoder, it use the tied weight, why the analysis which seperates the $\mathbf{W}_e$ and $\mathbf{W}_d$ still holds valid?
2. As mentioned in Weaknesses, what is the necessity for introducing $\mathbf{\Sigma}$?
3. For the nonlinearity, if $h$ is not an element wise function like tanh, but a complex nonlinear function (which is usually the case in nonlinear ICA literature), what will happen? Seems that the analysis for dropping $h'$ may not hold anymore.
4. How those filters picture in Figure 2 are connected with the quality for the method. The time series experiment is intuitive but it is too simple to judge the effectiveness. Some quantitive result is expected to show $\sigma$-PCA is a better solution. Some experiments can be synthetic with assumptions like varient scales for data varience.
5. Figure 2 seems have a lot of information by comparing the performance among those methods. A detailed discussion about this result is expected.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
