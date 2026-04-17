# Learning to Weight Parameters for Training Data Attribution

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
We study gradient-based data attribution, aiming to identify which training examples most influence a given output. Existing methods for this task either treat network parameters uniformly or rely on implicit weighting derived from Hessian approximations, which do not fully model functional heterogeneity of network parameters.
To address this, we propose a method to explicitly learn parameter importance weights directly from data, without requiring annotated labels.
Our approach improves attribution accuracy across diverse tasks, including image classification, language modeling, and diffusion, and enables fine-grained attribution for concepts like subject and style.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Many methods for data attribution consider model parameters uniformly, as if all parameters contribute equally. While influence functions do consider how some directions in parameter-space matter more than others, the true Hessian is intractable in practice and only subsets of parameters are used. The authors show that attribution quality is not uniform across parameters, and propose a method that learns parameter group weights from data.

### Strengths
To my knowledge, this work's exploration of how attribution is impacted by the heterogeneous nature of parameters (that some parameters are more important than others for particular tasks) is the first of its kind. It brings up some new findings, which concur with pruning work:
1. up blocks exhibit higher LDS attribution scores than others. In pruning, deeper computation is also more high-level/hierarchical
2. there is significant variation in terms of how much each parameter (and even each parameter group) affect attribution. Without variation in parameter importance in the context of pruning, we wouldn't be able to prune large fractions without affecting accuracy proportionally.

Separately, using weights from section 4.2, their method improves attribution (LDS score) of TracIn and TRAK for CNN and ViT. On text data (WikiText-103 w/ GPT-2-small) LDS score also improves for all methods tested (TracIn, TRAK, LoGRA, EKFAC). 

Appreciate the extra detail in appendix motivating Eq. 6, which describes how minimizing their loss equates to maximizing the SNR of the attribution score of the existing base attribution method.

### Weaknesses
Main issue: The authors do not specify (or at least do not make clear) what data is used to learn the weights from section 4.2. In section 5.1 they state "weights are learned using samples that are not in the training or test data", but what are that data? In the first imagenet section 5.1.1, there is no specification as to the data that is used, and that continues for the following sections on other data modalities. More specifically, in section 5.1.1, lines 372-376 are on how they train the models and how they evaluate LDS. Lines 388-396 are about evaluating mislabeled data detection. This section misses the most important part: what data is used for $w$ (the column that brings the improvement)? Without information on the above, the improvement could be due to a bit of metric hacking of LDS score. 

Second issue: The section on Finegrained data attribution focuses solely on Recall@10. Under this task, I feel simple baselines like using feature embeddings from pretrained models [1] should be tried. Prior work like Datamodels also do this and call it "representation distance" (See Section F.3.1 Estimation using baselines of [2])

[1] Singla et al., 2023. "A Simple and Efficient Baseline for Data Attribution on Images" (https://arxiv.org/abs/2311.03386)

[2] Ilyas et al., 2022. "Datamodels: Predicting Predictions from Training Data" (https://arxiv.org/abs/2202.00622)

Willing to raise score if weaknesses are corrected and questions addressed.

### Questions
Is maximizing the SNR of an attribution score a form of "metric hacking" of the LDS score? Why or why not? As a simple example, suppose we were measuring RMSE for some task. If one wanted to get a lower one could optimize to reduce variance and this would produce lower RMSE but not necessarily be better at the task (this is what I mean by metric hacking)

### Soundness
2

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
The paper is motivated by the observation that network parameters are not uniformly contribute to the model behavior, and thus the data attribution process. The paper propose a novel way to choose the weight of each parameter. The new method shows good effectiveness on traditional evaluation methods as well as provide more fine-grained attribution of concepts.

### Strengths
- The paper propose an interesting question (and opportunity) to improve the effectiveness of data attribution. This direction is discussed but not checked in depth in previous literature.
- The self-supervised mechanism to learn parameter weight is practical and easy to use.
- The improvement over standard data attribution methods are obvious.

### Weaknesses
- The main weakness lies in the self-supervised weight learning loss design. The analysis use a signal-to-noise ratio model to see the attribution score and try to optimize the parameter weight to get highest signal-to-noise.
  - Problem is that the definition of signal is related to the top-k attribution score. The decision is not justified well in Section 4.2 as well as in Appendix A.
  - Intuitively, top-k attribution score is very important (to the counterfactual prediction), while the least-k attribution score is also important and affect the counterfactual prediction. Why only choose top-k could be stated more carefully in the paper.

- The fine-grained attribution of concepts is somehow weak in both motivation, method design and experiment results
  - In Section 3.3, the motivation of revising parameter weight for concept attribution use similarity as the groundtruth, which is not a good practice for data attribution. Since how does the model learn a data may be different from the intuition that "similar data helps most".
  - The method design relies on the query data's distribution, could this also work for uniform/hessian based parameter reweighing?
  - Figure 4 shows some "imperfect" attribution results: e.g., style No.5 (not in the watercolor style), background No.2&5 (not with the same background)...

### Questions
- A core question could be: what is the essential reason that makes the hessian based parameter reweighing does not work well?

### Soundness
2

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
This paper addresses the problem of data attribution, which aims to identify the training samples most responsible for a model’s prediction. While existing gradient-based methods (e.g., TracIn, TRAK, Influence Functions) treat all parameters equally, this paper observes that attribution quality varies significantly across different parameter groups.   To tackle this, the authors propose a learnable parameter-weighted attribution framework, introducing explicit, non-negative group weights to scale gradient contributions. Extensive experiments on image classification (ResNet-18, ViT-B/16), language modeling (GPT-2), and diffusion models (Stable Diffusion, D-TRAK, DAS) show consistent improvements across metrics like LDS, mislabeled-data AUC, and tail-patch score.

### Strengths
1. Identifying parameter heterogeneity as a fundamental but overlooked factor in data attribution.
2. Generalizes across TracIn, TRAK, DAS, and others with a single weighting formulation.
3. Strong, consistent results across image, text.
4. Learned weights provide semantic insights into layer-level specialization

### Weaknesses
1. Technically, TRAK and similar Hessian-based methods already introduce an implicit parameter weighting through the approximation of $H^{-1}$, which scales gradients by local curvature. The proposed explicit weighting can thus be viewed as learning functional heterogeneity on top of curvature-based scaling.

2. The self-supervised loss bootstraps from existing methods (e.g., TRAK), so its ultimate accuracy may inherit their biases.

3.  Equation (6) defines the weighted attribution as  
  $\tilde{\tau}(x_q, x_i; w) = g(x_q)^\top \mathrm{Diag}(w)K g(x_i),$ but shouldn't it be $\tilde{\tau}(x_q, x_i; w) = g(x_q)^\top \mathrm{Diag}(w^2) Kg(x_i),$? which corresponds to reweighting both query and training gradients equally.

### Questions
1. How sensitive is the learned weighting to the quality of the initial pseudo-ranking? Would a noisy baseline still lead to meaningful weights?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the task of training data attribution using influence functions. Specifically, they analyze the varying effects of attribution across the parameter groups in a neural network. The first part of the paper shows this variance empirically, and the second part focuses on learning weights for individual parameter groups. Authors propose a self-supervised methodology to learn these weights. They use a baseline attribution method to provide pseudo ground-truth training examples.

Experiments span ImageNet classification, image generation and language modeling. Results show that the proposed weighting scheme helps improve attribution across a range of attribution methods (TracIn, TRAK, LoGRA, EKFAC). Additionally, the authors also curate a synthetic Pokemon dataset with images by varying subject, style and background. Results show strong results with attributing style, with moderate (/poor) results with subject and background.

Overall, the idea that different parameter groups have different effects on attribution is intuitive. Prior work makes ad-hoc assumptions, and this paper studies this behavior in a more principled fashion. The main concerns are that the paper mostly explored small-sized models and doesn't include any qualitative analysis on where they find gains over the baseline. For instance, does weighting parameter groups instead of pre-selection help improve attribution in LM generated text?

### Strengths
- This work presents a methodology to automatically learn weight model parameter groups on their attribution abilities. Prior work in attribution using influence functions focuses on a predefined set of parameter groups. This work presents a more principled solution for this.
- Experiments include comparison against multiple strong attribution baselines.
- The analysis of involving subject, style and background variations in synthetic dataset shows strong results.

### Weaknesses
- The language modeling experiments use GPT-2 small and its unclear to me if the gains will necessarily transfer to larger LMs. As the authors highlight in section 2, recent methods (LoGRA, TrackStar) have improved efficiency. Any reason for not experimenting with larger models in this work?
- The paper could benefit from analyzing its relative performance gains across different baseline attribution methods. For instance, they show strong results with TracIn but moderate results with recent methods.
- The paper could also benefit from qualitative analysis of their method for language modeling experiments; this would be an extension to their analysis on the synthetic Pokemon dataset.

### Questions
A few additional questions and comments,

- I would update the lines 218-219 to briefly mention the analysis from C.1. This could strengthen your argument in the main text.
- Why does the proposed method show better gains with some attribution methods? For instance, the gains are higher with TRAK and EKFAC and smaller with LoGRA in Table 2. Do you have a hypothesis for these differences?
- Any reason for not including TrackStar (line 127) in your experiments?

### Soundness
4

### Presentation
4

### Contribution
3
