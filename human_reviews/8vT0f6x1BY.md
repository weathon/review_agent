# Going Further: Flatness at the Rescue of Early Stopping for Adversarial Example Transferability

- Decision: Reject
- Scores: 3, 6, 3, 5

## Abstract
Transferability is the property of adversarial examples to be misclassified by other models than the surrogate model for which they were crafted. Previous research has shown that early stopping the training of the surrogate model substantially increases transferability. A common hypothesis to explain this is that deep neural networks (DNNs) first learn robust features, which are more generic, thus a better surrogate. Then, at later epochs, DNNs learn non-robust features, which are more brittle, hence worst surrogate. We demonstrate that the reasons why early stopping improves transferability lie in the side effects it has on the learning dynamics of the model. We first show that early stopping benefits the transferability of non-robust features. Then, we establish links between transferability and the exploration of the loss landscape in the parameter space, on which early stopping has an inherent effect. More precisely, we observe that transferability peaks when the learning rate decays, which is also the time at which the sharpness of the loss significantly drops. 
This leads us to evaluate the training of surrogate models with seven minimizers that minimize both loss value and loss sharpness. One of such optimizers, SAM always improves over early stopping (by up to 28.8 percentage points). We also uncover that the strong regularization induced by SAM with large flat neighborhoods is tightly linked to transferability. Finally, the best sharpness-aware minimizers are competitive with other training techniques, and complementary to other types of transferability techniques.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to increase the transferability of adversarial attacks, by focusing on optimizing the surrogate model rather than the attack objective.
To enhance attack transferability, the authors employ SAM (sharpness-aware minimization) to train a surrogate model with a flatter loss landscape w.r.t. parameters so that adversarial attack optimization does not fall into local minima.

The authors first examine the relationship between non-robust features (Ilyas et al.) and the improvement in transferability resulting from early stopping in surrogate model training. Previous research suggests that "slightly robust features" transfer more effectively than non-robust features. However, this paper demonstrates that "early stopped non-robust features" transfer better than "fully-trained non-robust features," challenging the existing hypothesis.

Next, they explore the connection between the sharpness of the loss surface with respect to the parameters of a surrogate model and adversarial transferability. Firstly, they demonstrate that the transferability decreases as the learning rate decays during surrogate model training due to an increase in the sharpness of the loss surface. Secondly, they illustrate that using SAM (sharpness-aware minimization), particularly with a larger penalty than the default (denoted as l-SAM in the paper), can enhance transferability. Finally, they show that their method complements existing approaches and further enhances transferability.

### Strengths
S1. Their method of employing SAM with a large penalty (i.e., l-SAM) to train a surrogate model improves the transferability.

S2. l-SAM complements existing approaches and further enhances transferability.

### Weaknesses
W1. Throughout the paper, there is no mathematical equation, and it lacks the necessary description to understand the paper. 
- (W1-1) The existing hypothesis and the authors' claim in Sec. 3 regarding the relation between the non-robust features hypothesis and early stopping are vague and difficult to understand since there is no mathematical formulation.
- (W1-2) SAM should be described with a mathematical equation. SAM hyperparameter $\rho$ is not described in the paper (requires referring to the original paper). 
- (W1-3) There are no descriptions of SAM variants (GSAM [Zhuang et al. 2022], ASAM [Kwon et al. 2022], and LookSAM [Liu et al. 2022]) used in their experiments, making their experiments difficult to interpret.

W2. 
I do not see any contradiction between the existing hypothesis and the experimental results in  Section 3 regarding the relation between the non-robust features hypothesis and early stopping. 
As far as I understand, [Benz et al. 2021] claim robust features transfer better than non-robust features. The authors' experiments show that "early-stopped non-robust features" are more transferable than "fully-trained non-robust features," which is aligned with [Benz et al. 2021] since "early-stopped non-robust features" should be more robust than "fully-trained non-robust features." (Anyway, this discussion requires mathematical formulation to discuss precisely.)

W3. The fact that the loss-surface sharpness of a surrogate model affects the transferability has already been shown by [Gubri et al. 2022], hence reducing the scientific contribution of this paper. 
[Gubri et al. 2022] proposed a method called LGV that aims to enhance the loss-surface flatness of the surrogate model, which is the same motivation as this paper.
In fact, in Table 2, the difference between LGV and LGV+l-SAM is minor, making the contribution weak.
The authors should make clear what's the difference between LGV and their approach and their contribution.

-------------------
References

[Benz et al. 2021] Batch Normalization Increases Adversarial Vulnerability and Decreases Adversarial Transferability: A Non-Robust Feature Perspective. ICCV 2021

[Gubri et al. 2022] LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity. ECCV2022

[Springer et al. 2021] A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks. NeurIPS 2021

### Questions
Q1. In Figure 5, the success rate of the transfer attack decreases even when the sharpness of the surrogate model decreases after the 100th epoch. How can it be explained?

Q2. It lacks discussion of attacking adversarially trained models, as discussed by [Wu et al. 2020], [Springer et al. 2021]. Is l-SAM useful for attacking adversarially trained models as well?

Q3. What is the loss-surface sharpness value for other methods, such as SAT [Springer et al. 2021]? I would expect SAT to be sharper than l-SAM based on the paper's claim. If it is, it can strengthen the contribution. If it's not, I would like to know the explanation for why the SAT is less transferable. In other words, does the sharpness metric correlate well with the transferability?

-------------------
References

[Wu et al 2020] SKIP CONNECTIONS MATTER: ON THE TRANSFERABILITY OF ADVERSARIAL EXAMPLES GENERATED
WITH RESNETS. ICLR 2020

[Springer et al. 2021] A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks. NeurIPS 2021

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to enhance the transferability of adversarial examples. Firstly, it provides empirical evidence that challenges the claim made by previous studies, which attribute the effectiveness of early-stopping on transferability to robust or non-robust features. Additionally, this paper establishes a correlation between the peak of transferability during training and both the decay of learning rate and the sharpness of the loss landscape. Based on this observation, the paper proposes the utilization of sharpness-aware minimization (SAM) as the optimizer for training a surrogate attack model, which can effectively improve the success rate of transfer attacks.

### Strengths
1. This paper is clearly written and well-organized.
2. The empirical evidence is sufficient to support the claims related to learning rate decay and early stopping. The experiments were conducted using various datasets and model architectures.
3. The proposed method is clear and can be easily incorporated into other existing transfer attack methods.
4. The experiment settings are clearly described, and all hyper-parameters are provided.

### Weaknesses
1. Though supported by experimental evidence, there is a lack of theoretical justification for why sharpness has a strong connection to transferability.
2. The evaluation experiments are only conducted on standardly trained models. I wonder how l-SAM works when attacking adversarially trained robust models.
3. As acknowledged in Section 2, the underlying mechanism of SAM is still a popular research topic and remains controversial [1, 2]. In particular, it is still uncertain whether the effectiveness of SAM is caused by sharpness. Therefore, I suggest toning down some claims on the proposed method, such as:
    
    > the effect of early stopping on transferability is closely related to the dynamics of the exploration of the loss surface
    > 
4. In Section 2, I suggest adding more details (e.g. training objective and algorithm) to improve readability, particularly for readers who are not familiar with SAM.
5. In the context of adversarial robustness, leveraging SAM is not a completely new idea. While I'm not critiquing the novelty of this paper, I think the following papers [3,4] should be mentioned in the related work.

    a. [3] proposes leveraging SAM to craft adversarial examples to find the common weaknesses of different models, which can boost the transferability of adversarial attacks.

    b. [4] shows that using SAM with a larger $\rho$ (exactly l-SAM in this paper) in standard training can improve adversarial robustness. Additionally, as shown in [5], a little robustness can improve adversarial transferability, so the effectiveness of using SAM in surrogate training is somewhat expected. Please discuss this viewpoint in your revision.

[1] A modern look at the relationship between sharpness and generalization. ICML 

[2] Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization. NeurIPS 

[3] Rethinking Model Ensemble in Transfer-based Adversarial Attacks. arxiv:2303.09105

[4] Sharpness-Aware Minimization Alone can Improve Adversarial Robustness. ICML Workshop

[5] A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks. NeurIPS

### Questions
Please see the weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper examines the impact of early stopping on the transferability of adversarial examples in deep neural networks, revealing that the benefits are due to the effect on the learning dynamics, particularly the exploration of the loss landscape, rather than the learning of robust features, as what the community always believe. It demonstrates that transferability is linked to times when the learning rate decays and loss sharpness decreases. The sharpness-aware optimizer SAM is used, which enhances transferability beyond early stopping. The study also finds a strong correlation between the regularization effects of SAM and increased transferability, positioning sharpness-aware optimization as an effective approach for creating transferable adversarial examples.

### Strengths
The strengths of this paper can be summarized as follows:
1. The paper empirically contests the widely accepted notion that early stopping in training DNNs leads to more robust feature learning and thus better transferability of adversarial examples. This is good to the community.
2. This paper introduces an evaluation of various flat-minima optimizers, showing how these can significantly enhance the transferability of adversarial examples by minimizing loss sharpness.
3. The study's focus on the SAM, particularly its large-neighborhood variant (l-SAM), provides evidence of its effectiveness in avoiding overly specific representations, thereby improving transferability.
4. The paper conducts a comparative analysis of the proposed methods against other training procedures, illustrating their effectiveness and complementarity in improving transferability.

### Weaknesses
The weaknesses of this work is listed below:

1. First of all, the teaser figure (Fig. 1) is very confusing. I think the illustration is not clear enough. I cannot extract any information of the "transferrability" other than from the texts. I would recommend you to make the training curve in a color system that brighter colors represent high transferability, right now it seems SGD has the best transferability and the early stopping one has the worst. Also, why is this  statistic-based or just sketch map. If it is the latter case, I would recommend the authors to use the former to make this more convincing.
2. I think both the notion used as well as the literature review of the sharpness for neural network in this work is very limited. First of all, the concept of sharpness has different meanings or evaluation method. Below are a lot of literature that this work missed:

    > [1] Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape, AISTATS 2022

    > [2] A modern look at the relationship between sharpness and generalization, ICML 2023

    > [3] On large-batch training for deep learning: Generalization gap and sharp minima, ICLR 2017

    > [4]  Fantastic generalization measures and where to find them, ICLR 2020

    > [5] Rethinking parameter counting in deep models: Effective dimensionality revisited (The Hessian-based metrics).
In fact, sharpness even has its own meaning in the context of adversarial training:

    > [6] Evaluating and understanding the robustness of adversarial logit pairing.
To be frank, I do not believe the SAM-based sharpness is a very good metric for sharpness, and the authors need to prove it is meaningful in the context of adversarial training.

3. If early stopping improves adversarial example's transferability, I think Figure 2 can be removed or moved to Appendix. Also, please improve the presentation by referring to the figure when making arguments. For example, on page 4 paragraph "Early stopping indeed increases transferability.", there is no figure referred and it is not friendly to readers.

4. For section 4, I have a general question: what about there is no abrupt learning rate decay used during training, for example the cosine learning rate schedule is very popular in different training settings. Based on this work, does the sharpness always drops during training? This seems a bit odd and contradicts to existing finds saying that better robustness prefers better flatness, see [7] below.

    > [7] Relating Adversarially Robust Generalization to Flat Minima/

5. I think the most significant problem of the evaluation is that, it lacks a serious evaluation on the flatness. For example, the literature [1] provides a list of methods to evaluate the flatness following different metrics. The authors claim better flatness improves tranferability, but forget to measure the flatness. This is not convincing.

### Questions
I do not have additional questions. Please refer to my comments in the "Weaknesses" column.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a hypothesis about the properties of the surrogate models used to produce adversarial samples in black-box attacks for improved success rate of the attack. The existing view is that there exist two types of features that model learns sequentially. So the early stopping improves transferability of attacks because the features learned first allow for more transferable attacks. The empirical evidence in the paper shows that both types of features show similar behavior with respect to attack success and the early stopping. The proposed conjecture is that transferability success depends on the learning schedule - it is shown empirically that around the LR decay epochs transferability spikes. This is further empirically connected with flatness wrt model parameters and then an approach to train flatter models (with SAM) as surrogate models is evaluated.

### Strengths
The paper proposes an interesting view on the reasons for transferability success. Large amount of experiments on different architectures is provided.
The paper is easy to follow and read.

I find the result that transferability improves with very strong flatness regularization, which hurts natural performance, one of the most interesting in the paper.

### Weaknesses
The main issue of the paper is the conceptual mismatch: empirical evidence shows that early stopping helps transferability of the attacks, also that transferability spikes when learning rate decays. Nevertheless, the statement "when LR decays, the sharpness of the parameter space drops" (page 5) is not valid overall. Learning rate does not take part in computation of the sharpness and does not affect it. Smaller learning rate is conjectured to drive models to "sharper" places, but up till now it is a conjecture. And therefore connection from LR experiments to sharpness experiments are not justified. Moreover it is stated that the success of transferability is related to the exploration of the loss surface (page 5), which is also not a precise statement - even with a large learning rate the trajectory of training can be very limited.

Several times the concept of basins of attraction is mentioned (section 5), but it is not precisely defined and it is not explained why this is even useful for the transferability of the attacks. Analogously, several times a very vague term "initial convergence" of the training is used. Please define it properly if it is critical for understanding the explanations, or do not use it.

In the works on adversarial attacks it is very important to describe precisely the attack mode assumed. As I understood, the knowledge of the attacker in the paper is assumed to be almost absolute - so they can train a surrogate model with exactly same architecture, dataset and even training setup. This significantly weakens the challenge of attacking. Moreover, only one type of attack is checked, which might not be enough to make a general conclusion - it might be that exactly BIM attack is affected by the flatness, but PGD for example is not. I would rather suggest to reduce the amount of target architectures checked and the ways to induce the flatness, but add at least one more attack type, that is significantly different from BIM.

The discussion about robust features and non-robust ones is very convoluted. According to the definition provided it should be very hard to create adversarial attacks on the RF, but if they are learned first then how early stopping can improve transferability? Or it makes it harder to create attacks, but they are more universal? The conclusion on page 4, from the robust and non-robust surrogate models training seems to be inverted: early learned NRFs would have low transferability (since it grows with training), but early learned RFs might be this way. Moreover, the experiment with robust and non-robust surrogate model does not seem to prove that the hypothesis about the sequence of learning process (model first learns RFs and then NRFs) is wrong. We still see that there is a significant difference in the success rate between robust and non-robust features. The conclusion that can be made is rather that early stopping effect on transferability is not connected with robustness of features, but not that the models do not learn some features earlier than others and this does not affect transferability.

The conclusion made is that flatness of loss surface with respect to the parameters is defining in the transferability of adversarial attacks. While empirical evidence demonstrates validity of such conclusion, I would suggest to be very careful with distinguishing flatness with respect to parameters and optimization with respect to the input that is performed to generate adversarial attacks. First flatness does not necessarily connect with the second, therefore such conclusion may sound misleading.

Finally, measuring only Hessian eigenvalues and trace is not the most precise way to measure sharpness (see for example Petzka, Henning, et al. "Relative flatness and generalization." Advances in neural information processing systems 34 (2021)).

### Questions
1 - What is the precise statement that is made by the paper? If it is that transferability of attacks requires very flat in the parameters models, then how learning rate and exploration of the loss surface connects to this statement? What exactly shows that better transferability in the beginning is not affected by the difference in features learned by the network?

2 - What is the attack model that is assumed in the paper?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
