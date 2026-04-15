# Residual Connections Harm Generative Representation Learning

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
We show that introducing a weighting factor to reduce the influence of identity shortcuts in residual networks significantly enhances semantic feature learning in generative representation learning frameworks, such as masked autoencoders (MAEs) and diffusion models.  Our modification improves linear probing accuracy for both, notably increasing ImageNet accuracy from 67.8\% to 72.7\% for MAEs with a VIT-B/16 backbone, while also boosting generation quality for diffusion models.  This significant gap suggests that, while residual connection structure serves an essential role in facilitating gradient propagation, it may have a harmful side effect of reducing capacity for abstract learning by virtue of injecting an echo of shallower representations into deeper layers.  We ameliorate this downside via a fixed formula for monotonically decreasing the contribution of identity connections as layer depth increases.  Our design promotes the gradual development of feature abstractions, without impacting network trainability.  Analyzing the representations learned by our modified residual networks, we find correlation between low effective feature rank and downstream task performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes adjusting the skip connection for generative models by introducing a weight decay. In particular, skip connections at later layers of the network are multiplied by a $\alpha \leq 1$ that decreases monotonically as a function of the layer index. By doing so, the experiments in this paper shows that the learnt representation in generative models improve.

### Strengths
The main strengths of this work are:

1- Simplicity of the proposed method: a simple adjustment to the network architecture.

2- Experiments are carried out on different generative models and different settings.

3- The paper is generally well-written and easy to follow.

### Weaknesses
Despite the strengths of this work, there are a few concerns that need to be resolved before accepting this work. The order presents the the importance of the concern based on which I rated this paper.

1- Despite the simplicity of the approach, there is no clear methodology on how to choose $\alpha_{\text{min}}$. For example,  in section 4.2, experiments are carried out with $\alpha_{\text{min}}=0.7$, while Table 2 shows that $\alpha_{\text{min}}=0.6$ works the best with ViT-B. This is a key concern as one can choose a suboptimal value of $\alpha_{\text{min}}$ that results in Even a worse performance than just setting $\alpha_{\text{min}}=1$ (i.e. naive residual connection). A discussion on how to properly set $\alpha_{\text{min}}$ should be a core part of the methodology section. Preferably, I suggest the authors to provide systematic analysis of how $\alpha_{\text{min}}$ affects performance across different model sizes and tasks.

2- While the experiments are carried out on both MAE and Diffusion models, the results in this work need to show experiment on  training a generative model with and without the proposed method. For example, a good experiment to show this is to revisit Table and show for each method the impact of the proposed method.

3- The third or of Table 3 (b) raise a critical question: Since weighing both the terms with a simple $\sqrt{0.5}$ results in a significant performance gain, does this mean that the baseline is trained sub-optimally? 

4- In section 4.2, the experiments shows a better separation in the learnt feature space when employing the proposed method. It would also make the argument stronger if we can quantify the improvement in that regard (e.g. reporting silhouette score).

### Questions
After going through the rebuttal, I commend the authors on the efforts put in addressing my comments. Thus, I raise my score from 3 to 6.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors proposed decayed identity shortcuts in residual networks, promoting the gradual development of feature abstractions. The author hypothesized that injecting shallower representations into deep layers with residual connections can be harmful by reducing the capacity for abstract learning. Then the authors conducted experiments to show this improves MAE by 4% and also benefits generation with diffusion models.

### Strengths
S1. The method is straightforward and easy to follow.

S2. The intuition is reasonable, aligns with some hierarchical representation learning and generation work.

S3. Dive into both representation learning and generative modelling.

### Weaknesses
W1. The experiments on representation learning are only on MAEs. In the literature, MAE is not best performing with linear probing and showed to work better in a finetuned manner, more experiments on different models would be greatly beneficial for the paper. No evidence this works universally across models. And the finetuned MAE with decayed identity shortcuts is no better than finetune MAE.
 
W2. Probing the inner state of different layers could enhance explainability and align with the intuition that underpins the work, providing deeper insights into the model’s internal processing, e.g., using representations from previous layers rather than the final layer, etc.

W3. Introducing an extra hyperparameter that needs to be justified/selected when using.

### Questions
Q1. What about a cosine decay schedule? Compared to linear.

Q2. How is this fixed hyperparameter alpha compared to a learnable parameter with regularization and constraints?

Q3. How to compare the idea of the hierarchical representation learning with Matryoshka representation learning? Instead of doing it hierarchically, matryoshka rep learning embeds different virtual concepts into different chunks of embeddings.


[C1] Matryoshka representation learning., Kusupati, Aditya, et al., NeurIPS 2022

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper shows that linearly decaying (over layers) the residual term in ViT-like models improves the training of generative models. With a few details, e.g., adding U-Net like skip connections between encoder and decoder, and initializing the final encoder output layer to be zero, the proposed method could successfully stabilize the training despite of the gradual removal of residual connections. Experiments are mainly done in two scenarios: training (a) Masked Autoencoders on ImageNet-1K, and (b) Diffusion models on CIFAR-100 and ImageNet-100. The paper reports that the proposed method improves the linear probing accuracy of both MAE and Diffusion training, as well as FID for Diffusion models at certain choice of decaying hyperparameter.

### Strengths
- The paper is well-motivated and easy-to-follow.
- The proposed method is notably simple and easy-to-adapt, which can be a strength in large-scale training regimes where recent generative models are indeed at.
- Rethinking the design space of neural architecture, e.g., ViTs, for generative modeling is a timely research direction.
- Figure 4 effectively highlight the quality of the learned representations with quantitative results.

### Weaknesses
- It is still questionable to me whether the proposed method is scalable for deeper networks; in my experience weakening the residual terms does introduce instability of overall training, whereas the paper did not much discuss about when the method fails. For example, Table 2 shows that the optimal hyperparameter of the method varies depending on the depth of the backbone networks, i.e., ViT-S vs. ViT-B. Also, Table 4 seems to show that the method is sensitive to hyperparameter choice depending on datasets. The paper may extend such analysis for wider depths, data, etc., possibly suggesting any guideline in using the method in practice.
- Table 1: The proposed method does not improve MAE on fine-tuning setup, while the performance of MAE was originally focused on after the fine-tuning. Although I agree with the paper’s comment that the fine-tuning performance is harder to compare with, I feel the current results in Table 1 does not fully confirm the main claim, i.e., residual connection harms generative representation learning. The paper may strengthen this point by showing that the method is effective for other generative modeling approaches, e.g., I-JEPA, BEiT, MaskGiT, etc., if applicable.
- The effectiveness of method seems to be dependent to an architectural modification of ViT, i.e., only after adding encoder-decoder skip connections as like U-Net (Table 3a). But such a skip connection may already improve MAE, although I don’t think this point is clearly discussed in the current manuscript.
- (minor) Section 4.2: “Embedding analysis. → Embedding analysis”

### Questions
- One alternative of decaying the residual term can be stochastic depth; i.e., by randomly dropping the identity term during training. I am curious that the authors have tested its performance, given that stochastic depth is also well-known to improve generalization of residual network training.
- Table 3b: The comparison shows that an ablation $\sqrt{0.5}(x_l + f(x_l))$ could achieve a quite competitive result to the proposed method. If so, given that the form has been already used in previous works and that it is simpler (e.g., the form does not have hyperparameter), what would be the key benefits of the proposed method over that ablated form?
- Can this method also improve models other than ViTs?

### Soundness
3

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
4

### Summary
The paper proposes two architectural changes to ViT that can greatly improve MAE's linear probing performance: (1) decreasing shortcut branch weight in deeper layers and (2) skip connection from encoder layers to decoder layers. With both changes, the authors observe better representations, evaluated with qualitative visualizations and linear probing accuracy. The authors showed that similar modifications can also improve diffusion model performance, and used effective rank as a lens to experimentally analyze & intuitively explain the performance gain.

----

I have read the authors' response and decide to keep my original evaluation.

### Strengths
+ Simple modification with effective performance gains.
+ Clear writing of the method & motivation.
+ Diverse experiments for MAE, diffusion models, and effective rank analysis.

### Weaknesses
+ The usual way of using MAE is not linear probing, but fine-tuning (or maybe nonlinear probing). The proposed method, unfortunately, reduces fine-tuning performance (Table 1).

### Questions
1. The proposed method reduces fine-tuning performance of MAE (which is its strongest suit) and doesn't improve linear probing to be on-par with SOTA (e.g., DINOv2). Given this, have the authors explored applying the changes to DINO? If it yields benefits, it could significantly improve the paper's value.

2. If the residual connection fundamentally limits effective rank, and thus hurts representation quality, why do objectives like DINO not have such issues?

### Soundness
3

### Presentation
3

### Contribution
3
