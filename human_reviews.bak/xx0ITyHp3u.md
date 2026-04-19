# Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging

- Decision: Accept (poster)
- Scores: 6, 6, 6, 5

## Abstract
Neural networks can be significantly compressed by pruning, yielding sparse models with reduced storage and computational demands while preserving predictive performance. Model soups (Wortsman et al., 2022) enhance generalization and out-of-distribution (OOD) performance by averaging the parameters of multiple models into a single one, without increasing inference time. However, achieving both sparsity and parameter averaging is challenging as averaging arbitrary sparse models reduces the overall sparsity due to differing sparse connectivities. This work addresses these challenges by demonstrating that exploring a single retraining phase of Iterative Magnitude Pruning (IMP) with varied hyperparameter configurations such as batch ordering or weight decay yields models suitable for averaging, sharing identical sparse connectivity by design. Averaging these models significantly enhances generalization and OOD performance over their individual counterparts. Building on this, we introduce Sparse Model Soups (SMS), a novel method for merging sparse models by initiating each prune-retrain cycle with the averaged model from the previous phase. SMS preserves sparsity, exploits sparse network benefits, is modular and fully parallelizable, and substantially improves IMP's performance. We further demonstrate that SMS can be adapted to enhance state-of-the-art pruning-during-training approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new pruning technique named Sparse Model Soups, which combines the weight averaging methods from [1] with the pruning method described in [2]. They provide empirical evidence to support the idea that this straightforward aggregation enhances the performance of pruned models in image classification tasks.

References

[1] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, 2022.

[2] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (eds.), Advances in Neural Information Processing Systems, 2015.

### Strengths
Clarity
- The paper is well written, ensuring it is understandable for readers.
- The suggested method is straightforward and effectively explained for easy understanding.

Originality and Significance
- This paper introduces a novel pruning method that incorporates model averaging techniques based on the Model Soup methods.
- They provide empirical evidence demonstrating that the proposed method offers improved performance when compared to baseline approaches.

### Weaknesses
Method
- Although they can parallelize the training process, performing $m\times k$ training epochs still imposes a substantial computational burden. And if the training cost becomes small, the overall performance gain significantly drops for the extreme sparsity cases.

Experiments
- It would be valuable to conduct empirical analyses to investigate why performance degradation occurs in regions of extreme sparsity.
- Similarly, it would be beneficial to empirically analyze why SMS performs well in situations of early sparsity, where batch randomness can lead to divergence between averaged weights [3].
- It would be advantageous to include an ablation study exploring different combinations of averaging coefficients where the $\lambda_i$ values differ from each other.

Recommend
- I suggest that the authors consider adding an Ethics Statement and a Reproducibility Statement immediately following the main paper.

References

[3] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? Advances in neural information processing systems, 2020.

### Questions
See the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents the Sparse Model Soups (SMS) framework, which applies the model soup algorithm to the neural network pruning procedure. Experimental results show that the model soup algorithm, a one of the most well-known weight-averaging methodologies that had demonstrated notable success in training dense neural networks, is also effective in iterative pruning procedures for training sparse neural networks.

### Strengths
Originality and significance: The proposed SMS framework does not bring a substantial level of novelty, as one can consider it as an application of the existing model soup algorithm to an arbitrary iterative pruning process. However, its originality comes from the actual implementation in the domain of neural network pruning, even though the individual elements may already exist separately. Considering the recent success of weight averaging methods for dense neural networks, it is valuable to explore the extension of these techniques to sparse neural networks.

Quality and clarity: Based on observations in the field of transfer learning, where fine-tuned models from the same pre-trained model tend to stay in the nearby region, the hypothesis that a similar phenomenon will occur when re-training the same pruned model is well-founded. The effectiveness of the proposed SMS framework is confirmed through a range of experiments conducted in the domains of image classification, semantic segmentation, and neural machine translation.

### Weaknesses
While the proposed SMS framework incorporates the model soup algorithm in the context of neural network pruning, it does not provide specific insights into the unique factors that are especially pertinent to sparse network training. The questions remain: What attributes of the model soup algorithm contribute to its effectiveness in the neural network pruning regime? Is it the same reason why weight-averaging methods have succeeded in conventional dense network training?

### Questions
1. Alongside model soups, another prominent weight-averaging strategy is Stochastic Weight Averaging (SWA). The fact that SMS requires m training runs at each cycle makes SWA, which performs weight averaging within a single SGD trajectory, somewhat appealing. Are there any baseline results using SWA instead of model soups?

2. I understand that the authors have opted to show exclusively the UniformSoup results for CityScapes and WMT16 since the GreedySoup algorithm utilizes validation data, which are the test split here. However, despite the potential fair comparison issue, presenting additional GreedySoup results might offer valuable insights into the benefits of selectively using soup ingredients at high sparsity levels.

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
The idea is to combine model soups from transfer learning with pruning. The proposal is : in prune-retrain paradigm (be it from scratch or pretrained), the training portion is replaced by model-soup. It improves the overall quality of the final pruned model as is validated extensively in experiments.

[Update]
In light of the responses by authors, I am increasing my score to 6.

### Strengths
This being an empirical paper with simple idea proposal, the authors do an excellent job of evaluating their idea in terms of 
(1) extensive experiments on different domains 
(2) covering baselines that are natural competitors to the proposal.

### Weaknesses
1) I am not sure if this paper contributes new ideas or analysis. The proposal is to replace the training portion of prune-retrain with model soups. I do not have background on transfer learning, but as a general machine learning person, it is not surprising that it improves the accuracy of the model given the backdrop of model soups paper . Since in both the cases it holds that m copies of model start from the same initialization.

### Questions
1) How is IMP $m \times$ implemented? Is the pruning rate for each prune step reduced? or is the training portions increased m$\times$. The latter, I suspect, will not be very useful. 
2) Are there any challenges that are specific to using model soups for training portion of prune-retrain algorithm which differentiate it from applying model soups to finetuning of pretrained models? I felt that there were no new challenges here.

### Soundness
2 fair

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
The paper proposes merging sparse models by initiating each prune-retrain cycle with the averaged model from the previous phase. They show that averaging these models significantly enhances generalization and OOD performance over their individual counterparts. Overall, in summary, it is an extension of model soups for sparse models.

### Strengths
1. The experimental section of the paper + supplementary is rich illustrating noticeable gain.
2. OOD experiments are new for sparse model soup showing SMS consistently improves over the baselines.

### Weaknesses
I have significant novelty concerns with the draft. 

1. The idea of sparse model averaging has been widely explored including the model soups (eg. https://arxiv.org/abs/2205.15322 https://arxiv.org/abs/2208.10842 https://arxiv.org/abs/2306.10460  etc). 
2. The authors have failed to detail how their method contrasts with existing sparse model soup papers in their related work section. I feel it is just an incremental work over the existing literature. The benefits of averaging the sparse masks is already known.
3. Although I appreciate the extensive experiments by authors, I still feel the experiments are limited to small-scale datasets and models (maybe ViT scale or OPT models-based experiments will add value).
4. I feel auxiliary benefits of model soups like OOD robustness, fairness, etc are good directions to explore.

### Questions
See above.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor
