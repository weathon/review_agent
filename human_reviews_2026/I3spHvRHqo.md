# A Non-vacuous Test Error Guarantee for Deep Learning without Altering the Model

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 0, 4, 6, 6

## Abstract
Deep neural networks (NN) with millions or billions of parameters can perform really well on unseen data, after being trained from a finite training set. Various prior theories have been developed to explain such excellent ability of NNs, but do not provide a meaningful bound on the test error. Some recent theories are non-vacuous under some stringent assumptions and extensive modification (e.g. compression, quantization) to the trained model of interest. Therefore, those prior theories provide a guarantee for the modified models only. In this paper, we present two novel bounds on the true error of a model. One of our bounds can be exactly computable from the training set only, without altering the model, and hence provides a theoretical guarantee for a trained model. Our approach is to decompose the data space into different local areas to approximate the local errors by using training samples in a controlled way, then use those local errors to approximate the true error of a model. Our bounds are verified on 32 modern NNs, which were trained by Pytorch on the ImageNet dataset. The exactly computable bound is found to be non-vacuous. To the best of our knowledge, this is the first non-vacuous  bound at this large scale (NNs with more than 600M parameters, ImageNet), without altering those 32 trained models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides two bounds on the test error of a deep neural network. In comparison to prior work that depends on impractical assumptions such as norms of weight matrices, stability and robustness of algorithm, infinite dimensions of neural networks for NTK-based bounds, this work focuses on calculating bounds that be directly computed without several assumptions, e.g., one such bound can be computed directly from the training data. Further the proposed bounds are claimed to be non-vacuous, i.e., they are relevant in the given context. Experimental results for one of the bounds validate

### Strengths
- The paper proposes a non-vacuous bound on the test error for deep neural networks that can be directly applied even using only the training data. This is and useful and practical bound compared to prior works based on several assumptions that are not relevant in deep learning
- Experimental results showing that the test errors are close to the computed bound are provided.

### Weaknesses
- The paper provides 2 bounds, however, only the second bound is non-vacuous and practical. The first bound has several terms that are not practically computable. The first bound is used for deriving the second bound, but it is not clear to me if the first bound has any use otherwise. 
- The error bound still depends on the optimal partitioning of the data, it is not clear how tight the bounds are. Is it possible to provide bounds obtained from any other previous method that is also non-vacuous for a few cases to show how they compare.
- The experiments are limited to image classification/image models. Is it possible to provide experiments with language models, maybe even some small model to see how the bound compare to empirical loss?

### Questions
Please see the questions in the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This submission fits within the line of work addressing generalisation ability of machine learning models, particularly deep learning, via statistical learning theory and generalisation bounds, also called risk bounds and error bounds in supervised classification. The claimed contributions are: bounds on the expected error of a hypothesis and experiments with large neural networks on ImageNet, presumably showing non-vacuous values.

### Strengths
The extensive experiments comprise results on 32 deep learning models spanning across various kinds of architectures, as evidenced by the results reported in Table 2 on page 8. Arguably, the extent of these experiments is the main strength of this submission.

### Weaknesses
A significant weakness is execution of the writing and narrative, which have problems affecting the quality and clarity of this submission. There is a need to improve clarity on connection to the cited works, and exactly how the submission positions itself with respect to such literature. There are discussions using technical terms that appear before the technical terms are formally defined, which will affect the ability of readers to understand the discussions. There are also some undefined terms, such as meaning of hypothesis and loss function, perhaps they were assumed to be widely known, though at least a comment explaining the meaning would be expected, if not fully formal definitions. Without clarity on these things, the rest of the content will be obscure to readers. My general impression is that this manuscript needs a major makeover to achieve a paper that might be acceptable in terms of clarity and readability.

Other weaknesses I would like to flag are regarding the technical content per se, this could be considered to fall within criteria such as originality and significance. I am inclined to believe that the error bounds presented here are not novel, but perhaps the authors could offer clarifications on why/how the believe the bounds in Theorems 3.1 and 3.2 to be novel as claimed. Another important point to consider is regarding the *kind* or *nature* of these bounds, plus their connection/difference to PAC-Bayes bounds mentioned in many places. The bounds in Theorems 3.1 and 3.2 are bounds on the expected error of an individual hypothesis $h$, whereas PAC-Bayes bounds consider distributions over hypotheses. This distinction has not been addressed clearly, and consequently the discussions and comparisons would appear obscure and perhaps misleading. 

One more I'd like to single out, already mentioned in connection with the previous two, but for the sake of emphasis so that the authors can consider this point by itself. The cited literature is not all clearly connected with the work reported in this submission. Or at least the connection is not presented clearly. I would suggest that cited items should have a good reason for being cited, and that reason should be apparent in the manuscript. On the other hand, there are important works that have been missed, such as Perez-Ortiz et al. (2021) building on the groundbreaking work of Dziugaite & Roy (2017) and closely connected to generalisation and the PAC-Bayes literature this submission has tried to connect to. On the general PAC-Bayes literature, there are important works that weren't cited such as Alquier (2024) and works of Germain and collaborators, whereas the connection of this submission to the PAC-Bayes literature items that have effectively been cited is not completely clear. Perhaps the authors would want to have a think on the arguments for connecting to PAC-Bayes and the chosen literature items to cite.

### Questions
Could the abstract be reframed to focus from the start on the question(s) or problem(s) being addressed here, what's the approach used to tackle the question(s) or problem(s), and highlights of the important contributions and results.

A suggested reorganisation: Could the section on related works be moved to the end of the paper, perhaps before the conclusion section, so that introduction section is followed by sections on error bounds and empirical evaluations.

The introduction didn't do a very successful job in setting the context for this work. Could the authors clarify what they tried to achieve with the narrative in this introduction and have a think about how to streamline it to achieve a better framing for their work.

The technical section on error bounds, which presumably intends to set the theory background, was missing rigour in definitions and theorem statements. Could the authors make sure to offer formal definitions for all notions used, or at the very least intuitive descriptions and explanations, for the sake of clarity on the intended meaning of these things and to help readers understand it. I mentioned before the missing meaning of *hypothesis* and *loss* but they are not the only. Another case: There is a set $\mathcal{Z}$, called an instance set here, but no specification of what kind of set this might be; and in the theorem there is mention of an area measure related to this set, or subsets, and all of this is mysterious in the absence of definitions.

Could the authors offer clarity on their reasoning for the chosen works and bounds to compare to, from theory point of view.

Are the authors aware of works on sample-dependent bounds, such as Satyen Kale and collaborators, and there are others. I would argue that such line of work is more relevant to this submission than the cited.

As mentioned, it would look like the extent of empirical evaluations is the main strength of this submission. Could the authors consider reframing the presentation and organisation, perhaps the whole narrative, to bring this to the forefront.

The conclusion section was rather vague and inconclusive. Perhaps a consequence of some other problems with this submission that generally affect the perception of clarity on what's really being done here.

Regarding the title, could the authors reconsider "test error guarantee" which can lead to confusions, on accounts that "test error" is sometimes conflated with "test set error" and what is really meant is the expected error, also called risk in statistical learning.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provide theoretical nonvacuous bound for generalization error. The bound can be approached through access to data without modification of models. The author also propose the concep of data complexity which is represented through concentration of data at different region of data space. Lastly, they provide experiments results on different models with Imagenet dataset and compare those results with other generalization error bounds to show advantages of their method.

### Strengths
1. The theory is solid and non-vacuous for experiments presented.
2. The method for obtaining the bound is relatively easy compared to prior works.
3. The method provide method that did not require modification of the models.

### Weaknesses
1. Despite the fact that the bound is non-vacuous, it did not provide more insight about why certain models generalize better or how to improve training as it is not depedent on generic properties of models.
2. For bound in theorem 1 and corollary 1, those results require access to a_i and optimal partition. If I understand it correctly, a_i is the generalization error on the specific region assigned. Under the case, this makes the result more of a decomposition than a predictive bound, and the insight may be limited.
3. For bound in theorem 2, the bound require no knowledge of models but also it is a fixed value once those hyper parameters are fixed. One direct evidence is that if we substract error bound by training error in Table 2, we will get almost constant value around 36%-37% or 33% for all models in terms of mild and optimized results. It cannot distinguish between model's generalization ability.
4. Following previous problems, if it is only dependent on data, it will be better to show results on different datasets to test the tightness the bound can offer.
5. The writing and stressing on "model dependent" and "no change of model" can be confusing. The dependency of models is through its generalization error in specific region of data space and that can be circulative conceptually. I suggest removal or change of related statement and directly posit the works as setting fundamental limit of generalization error gap for different datasets which better align with the experiments and its theory contribution.

### Questions
1. Is there a better way to select optimized hyperparameters instead of the grid search used in the main context?
2. Is there a more intuitive way to interpret the data complexity?
3. Is it also fit for smaller or simplier models like linear regression or SVM? Based on the experiment results, my observation is that the bound can be vacuous as the training error is larger or when the models are not able to fit well.
4. For smaller models like LeNet, prior works [1] can obtain tight bound within 10% off in CIFAR10 or MINST. Is this method also accurate in that region? What are the limits of the data dependent bound without exploiting the model properties?

[1] Lotfi, S., Finzi, M., Kapoor, S., Potapczynski, A., Goldblum, M., and Wilson, A. G. Pac-bayes compression bounds so tight that they can explain generalization. Advances in Neural Information Processing Systems, 2022.

To summarize, the paper make theory progress in terms of its technics and verify it through experiments and ablation studies of different components inside the theory. However, there are also limitation about its ability to interpret or offer insight about a training algorithm. There exist overstated connection to deep learning or model dependent properties and should be adjusted. I hope that the author can realign its structure or its presentation to better fit what the theory really reflect on to help reader understand the core of the theory.

### Soundness
3

### Presentation
3

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
This paper introduces novel generalization bounds which depend on the regularity of the sampling distribution rather than that of the model. By recombining ideas (local partitioning of the input space plus concentration for multinomial/binomial counts) into a new model‑ and data‑dependent bound that you can evaluate from the training set alone, without compression, and works on Imagenet scale. That said, the bound is still loose (often ~2–3× the true test error) and depends on a chosen partition, so it is more a pragmatic workaround than a final theory of generalization. There is a lot of hard work that goes into getting the numerical bound on the actual NN tighter, without bounding a compressed NN, as in https://arxiv.org/abs/2211.13609  which is a different object

The suggested convergence rate O(n−1/4) leaves plenty room to improve, and the bound is also sensitive to the partition K

### Strengths
The paper brings an original contribution to what is a huge literature on PAC-Bayes bounds.  There is a good literature review and an extensive test on empirical data.

### Weaknesses
The tight bounds generate less insight than looser bounds can.
 The major draw back is that, just like  compression bounds of  https://arxiv.org/abs/2211.13609, the bound is mostly collapsed to training error. And it's much worse than the compression bound in terms of tightness. Presumably that's the price you need to pay for developing a theory that bounds the actual network.

### Questions
Q1 - how does this work relate to prior non-vacuous bounds such as the marginal-likelihood one in https://arxiv.org/abs/2012.04115 

Q2 - in your main bound - can you give an intuitive description of each term, and explain where it may break down, e.g. under parameter re-scalings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits the classical generalization problem in deep learning and proposes non-vacuous generalization bounds for models that are trained without being compressed or optimized to make the bounds tighter. The authors derive new bounds that depend on the complexity of the data distribution and the local behavior of the hypothesis in different areas of the data space. They also propose a version of this bound that can be computed in practice using the training data and without any changes to the pretrained model. The authors demonstrate that these bounds are non-vacuous for models pretrained on ImageNet, proving the first non-vacuous bounds for non-altered models at the scale of ImageNet.

### Strengths
- The paper addresses a long-standing criticism that generalization bounds for deep networks are constructed for altered versions of the models and do not apply to the original models, or models used in practice. 
- The derivation of the bound and its empirical computation are both sound. 
- The bounds are non-vacuous in practice and the gap between the bounds and the test error is not too big compared to even sota bounds.

### Weaknesses
- The transition between the bounds is very poorly explained, or not explained at all such as the case for 3.2. While including a proof sketch is not required, it would be at least important to keep the flow of the work by explaining why we needed the first two bounds to get to the third bounds and how they connect and differ.
- Overall, the presentation of the notation and theorems is super hard to follow and took a very long time to understand qualitatively what the different quantities measure. More clarity is definitely required. 
- Last point on the presentation: overall the quality of the writing can be improved to be more concise as it takes until the end of page 4 to get to the meat of the paper. The first paragraph of the introduction for instance can be easily removed. 
- While the gap between the test error and the bound is not as large as some of the classical bounds, it is still not totally tight and therefore it would have been interesting to see if the bounds track behavior we see in practice, e.g., CNNs being more compressible and generalizing better than MLPs given the same number of parameters. 
- It is unclear how the optimization over the hyperparameters is accounted for in the bounds to keep them valid. 
- While the lack of baselines is understandable given that other approaches do require changing the model, it is worth noting that PAC-Bayes bounds can be computed with a collapsed posterior around the solution found and do not need to be computed for a stochastic model. 
- The evaluation is limited to ImageNet only; and while this is indeed a very challenging dataset, it would be interesting to see how tight the bounds are for other datasets.

### Questions
- If you optimize your bounds over hyperparameters, do you pay a union bound penalty to make the bound simultaneously hold over all hyperparameters like the authors do in "PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization ", https://arxiv.org/abs/2211.13609?
-  Can you compute baseline bounds proposed in previous works without changing the model at all? If the bounds are vacuous, that would serve as a confirmation that your work indeed improves upon existing works. E.g., for compression bounds that use linear subspace projection, you could try to inject the KL without any subspace projection, ie in the original space, and see if the bounds are indeed vacuous. 
- Can you compute the correlation coefficient between the bounds and the test error? It would be interesting to see to which extent the bounds do track the test error. 
- Can you use the bounds to provide better understanding of generalization in practice. eg, why SwinTransformers perform better than ResNets?

### Soundness
2

### Presentation
1

### Contribution
3
