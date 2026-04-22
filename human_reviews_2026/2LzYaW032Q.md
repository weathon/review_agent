# ProReGen: Progressive Residual Generation under Attribute Correlations

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Attribute correlations in the training data will compromise the ability of a deep generative model (DGM) to synthesize images with under-represented attribute combinations ($\textit{i.e.,}$ minority samples). Existing approaches mitigate this by data re-sampling to remove attribute correlations seen by the DGM, using a classifier to provide $\textit{pseudo-supervision}$ on generated counterfactual samples, or incorporating inductive bias to explicitly decompose the generation into independent sub-mechanisms. We present ProReGen, a $\textit{progressive residual generation}$ approach inspired by the classical Robinson's transformation, to partial out from an image attribute $\mathbf{x}_2$ its component $m(\mathbf{x}_1)$ that is predictable by other image attributes $\mathbf{x}_1$, and the residual $\gamma = \mathbf{x}_2 - m(\mathbf{x}_1)$ that is not. This simplifies the problem of learning a DGM $g(\mathbf{x}_1, \mathbf{x}_2)$ conditioned on correlated inputs, to learning $\tilde{g}(\mathbf{x}_1, \gamma)$ conditioned on orthogonal inputs. It further allows us to progressively learn $\tilde{g}$ by first shifting the burden to abundant majority samples to learn $\tilde{g}(\mathbf{x}_1, \gamma = 0)$, and then expanding it with additional layers $g\_{\text{res}}$ to resolve its difference to $\tilde{g}(\mathbf{x}_1, \gamma)$ using residual attribute $\gamma$ on limited minority samples. On three benchmark datasets with curated varying strengths of attribute correlation and one dataset with natural attribute correlation, we demonstrate that ProReGen---with input orthogonalization and progressive residual learning---improved the correctness of minority generations compared to existing strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study the problem of training a generative model under attribute correlation. To this end, they propose a novel approach that separates training into two stages. First, a part of the model is trained on the majority data, while in the second stage the network learns only the residual information for the correlated attribute. The efficacy of the method is evaluated on variations of MNIST and CIFAR.

### Strengths
- The main idea is novel and clearly presented.

- The studied problem is relevant and the proposed solution appropriately motivated.

- The experimental results highlight the efficacy of the method in variations of MNIST and CIFAR10.

- I appreciate the ablation of the inverse causal direction for MNIST.

### Weaknesses
- The main weakness of the paper lies in the insufficient experimental support. To showcase the effectiveness of the method, as well as the importance of the task, I would expect evaluation on more complex and natural scenarios. For example, a multi-attribute dataset e.g., [1], where this kind of attribute imbalances are naturally occurring would be appropriate.

- I am missing a discussion on earlier works that study generative models under imbalanced attribute distributions. For example,  [2],  [3] to name a few.

- It would be interesting to discuss how (or whether) the presented method can be applied to SOTA image generation models e.g., diffusion-based models.

- On a similar note, I am missing a discussion on the form of the studied problem in scenarios, e.g., text-to image,  where the conditioning variable lives in a combinatorially large space. How would imbalances affect the performance in such models?

- It would be valuable to discuss the presented method under the light of disentanglement and potentially add relevant comparisons.

[1]. Deep Learning Face Attributes in the Wild

[2]. Bias and generalization in deep generative models: An empirical study.

[3]. Multilinear Latent Conditioning for Generating Unseen Attribute Combinations

### Questions
I would appreciate if the authors address the main points raised in the weaknesses section. In particular, I would encourage further experimentation on multi-attribute images (e.g., [1]). Further discussion/comparison to earlier works on disentanglement and generative modelling under imbalanced attribute distributions would also improve the paper.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
ProReGen is an approach for generative modeling that addresses the challenge of generating underrepresented minority samples in the presence of attribute correlations. The paper proposes a two-stage progressive learning framework inspired by Robinson's partialling-out transformation, which orthogonalizes correlated input attributes and decomposes generation into majority and residual components. The method has been evaluated on Colored-MNIST, MNIST-Correlation, and Corrupted-CIFAR10, showing improvements in minority sample generation correctness compared to naive baselines and existing mitigation strategies

### Strengths
- The paper demonstrates both VAEs and GANs, which is diverse in nature
- The application of partialling out transformation for generative modelling is well motivated
- The paper evaluates across multiple metrics and provides multiple ablation studies 
- The existing methods rely on signals from an external classifier to provide pseudo-supervision to the conditional generative models. Interestingly, they partially out an image from its components

### Weaknesses
- The paper demonstrates both VAEs and GANs, which is diverse in nature
- The application of partialling out transformation for generative modelling is well motivated
- The paper evaluates across multiple metrics and provides multiple ablation studies 
- The existing methods rely on signals from an external classifier to provide pseudo-supervision to the conditional generative models. Interestingly, they partially out an image from its components

### Questions
How sensitive is the performance to eros in estmaiting m(x1)?
Why does cGAN not perform well compared to the original paper
What happens when the oracle classifiers are not so accurate
Can the framework also incorporate continuos attributes or only discere ones?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new approach for training deep generative models (DGMs) that are robust to under-represented attribute combinations. Given attributes X_1 and X_2, the model learns g(X_1, $\gamma$) rather than g(X_1, X_2), where $\gamma$ is the residual after predicting X_2 given X_1. This, the generative model is conditioned on orthogonal variables, rather than potentially very correlated variables. This makes the model more robust when generating rare attribute combinations.

### Strengths
- The proposed approach of orthogonalizing attributes and the two-stage residual training approach is novel. The proposed solution is intuitive and seems very reasonable. 

- The fact that ProReGen is agnostic to the choice of underlying DGM, rather than being designed with only VAEs or the like, is a clear strength of the proposed approach 

- The writing is clear and easy to follow

### Weaknesses
- The experiments are all on toy, synthetic or partially-synthetic data. The evaluation is performed on colored-mnist, mnist correlation, and Corrupted-CIFAR10 (wherein synthetic noise was added to CIFAR images). This is not a very convincing evaluation; at least, it does not provide evidence that the proposed ProReGen approach would correctly model under-represented attribute combinations in realistic data with natural correlations. 

- The authors claim that the proposed approach is agnostic to the choice of deep generative models, but only discuss and implement it on VAEs and GANs. I am surprised to not see results or discussion on the case where Diffusion models (DDPM, DDIM, latent diffusion, etc) are used as the base DGM.  I don't see a technical reason why diffusion wouldn't work; the authors' claim that the choice of DGM is arbitrary seems reasonable to me.  For that reason, I think the authors should either empirically evaluate PreReGen on Diffusion models, or else clarify why this is not possible and then amended their claim about the DGM being model agnostic. 

- This is a minor point, but I believe the paper would benefit if it included a discussion of property-controllable VAEs [1]. PCVAE is definitely different, as it tries to learn a disentangled latent representation rather than making the model robust to correlated attributes and minority samples. But the underlying goals are similar enough that I believe comparing and contrasting the proposed approach with existing property-controllable works would be interesting. 

[1] Guo, Xiaojie, Yuanqi Du, and Liang Zhao. "Property controllable variational autoencoder via invertible mutual dependence." ICLR. 2020.

### Questions
Can a Diffusion model be used as the underlying DGM? Does ProReGen perform well on larger, more complicated, realistic datasets? 

If these points are well addressed, then I am happy to raise my score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes ProReGen that tackles conditional generation modeling for synthesizing under-represented minority images. ProReGen first learns the predictable part of one attribute from another and then treats what’s left as a residual signal that highlights the minority cases. Training is performed in two stages: Stage I trains a standard VAE/GAN on the common cases; Stage II freezes this backbone and adds a small residual-conditioned module trained only on minority examples to correct the missing variations. Therefore, this separates the attributes so the base learns general realism while the add-on learns rarity (minority attributes). Both VAE and GAN applicability is shown on Colored-MNIST, MNIST-Correlation, and Corrupted-CIFAR-10. The method improves correctness on rare attribute combinations and achieves competitive performance (FID, Coverage, Density)

### Strengths
- The proposed approach is explained in detail and could be easy to incorporate ie (progressive training scheme and plug and play for VAE/GAN architecture).
- Proper baselines are discussed in the related section and used to compare the performance.

### Weaknesses
- It would be great to show the results on some real and large scale vision dataset to help the readers evaluate the efficacy of the proposed approach. It’s unclear how the gains would translate to higher-resolution, richly annotated datasets (faces, scenes) or to modern diffusion/flow models, which are the latest common practice. A larger-scale study would strengthen the empirical case.
- Stage-II adds networks and trains only on minority samples while freezing the backbone. It would help the readers to show discussions around added compute cost, convergence stability, or sensitivity to the size of the residual sub-network.

### Questions
Please refer to the weakness section

### Soundness
2

### Presentation
2

### Contribution
2
