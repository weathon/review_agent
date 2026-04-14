# Metalic: Meta-Learning In-Context with Protein Language Models

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8

## Abstract
Predicting the biophysical and functional properties of proteins is essential for in silico protein design. Machine learning has emerged as a promising technique for such prediction tasks. However, the relative scarcity of in vitro annotations means that these models often have little, or no, specific data on the desired fitness prediction task. As a result of limited data, protein language models (PLMs) are typically trained on general protein sequence modeling tasks, and then fine-tuned, or applied zero-shot, to protein fitness prediction. When no task data is available, the models make strong assumptions about the correlation between the protein sequence likelihood and fitness scores. In contrast, we propose meta-learning over a distribution of standard fitness prediction tasks, and demonstrate positive transfer to unseen fitness prediction tasks. Our method, called Metalic (Meta-Learning In-Context), uses in-context learning and fine-tuning, when data is available, to adapt to new tasks. Crucially, fine-tuning enables considerable generalization, even though it is not accounted for during meta-training. Our fine-tuned models achieve strong results with 18 times fewer parameters than state-of-the-art models. Moreover, our method sets a new state-of-the-art in low-data settings on ProteinGym, an established fitness-prediction benchmark. Due to data scarcity, we believe meta-learning will play a pivotal role in advancing protein engineering.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present Metalic, a new approach to predicting properties of protein sequences. Metalic involves two phases: a "meta-learning" phase, during which a PLM backbone is fine-tuned to learn from data from multiple related fitness prediction tasks presented in-context, and then a per-task "fine-tuning" phase, where models are fine-tuned at inference time on a small amount of data specific to the task at hand. The authors show that Metalic outperforms baselines in the zero-shot setting for single mutations and is competitive for multiple mutations.

### Strengths
The paper is clearly written, includes a range of baselines and ablations, and also attacks an important problem. In some sense, protein fitness prediction is an excellent application for meta-learning, in that it's one of the few settings I can think of where there is an almost arbitrarily broad distribution of tasks that are nevertheless closely related to one another.

### Weaknesses
I'm a bit conflicted about this paper. On the one hand, the method seems to work, and, as I mentioned, the authors already include most of the experiments that I'd think to ask for. On the other: Table 5 shows that the lion's share of the improvement from Metalic comes from the in-context meta-learning; the per-task fine-tuning phase only helps in the many-shot case. But Metalic is only the best method in the zero-shot case (albeit with fewer parameters than baselines), where the only possible improvement comes from meta-learning. If the authors provided parameter-matched versions of Metalic that showed a much clearer benefit to using the method even in the many-shot case, that might change my thinking, but that also obviously changes the viability of performing the fine-tuning step at inference time. As it stands, the paper seems to show that in-context meta-learning works well  for zero-shot protein fitness prediction. In-context meta-learning isn't new, and I just don't know if a simple application of a method that already exists to a task that---for ICLR readers---is ultimately fairly niche warrants acceptance. I'm obviously a bit conflicted and can likely be convinced, but at the moment I'm leaning reject.

### Questions
While the paper is fairly clearly written, there are a few pieces of information I haven't been able to find:

1. Why did you stop at 8M? How much more computationally intensive would it be to run experiments with the 35M model?
2. How many tasks are presented in-context to generate the figures in e.g. Table 1? What data is used in-context at evaluation time? Is there overlap with the data used during the meta-learning phase?
3. "Moreover, our method still conditions on an unlabeled query set, and the protein embeddings in that query set, which allow for meta-learning a form of in-context unsupervised adaptation." -> Is there any evidence in the paper that this occurs? Would be cool to see an ablation of the data from other tasks to isolate this.

### Soundness
4

### Presentation
4

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
The paper titled introduces Metalic, a meta-learning approach for predicting protein fitness, which is a measure of protein properties like stability and binding affinity. Predicting these properties accurately is challenging due to the scarcity of labeled data. Metalic improves on this by applying meta-learning to a distribution of related protein fitness tasks, enabling positive transfer to unseen tasks. The approach combines in-context meta-learning and fine-tuning without accounting for fine-tuning during meta-training, which helps to generalize well even with limited data. Experiments show that Metalic achieves state-of-the-art performance in zero-shot and few-shot settings on the ProteinGym benchmark, using 18 times fewer parameters than current models.

### Strengths
1-By integrating in-context meta-learning with subsequent fine-tuning, Metalic can generalize well to new tasks, even though fine-tuning is not accounted for during meta-training. Moreover, the model is more computationally efficient than state-of-the-art approaches.

2-Metalic achieves strong performance in zero-shot and few-shot protein fitness prediction tasks, which is particularly valuable given the limited availability of labeled data for many protein-related tasks.

### Weaknesses
1-I anticipate a more in-depth analysis of the correlation between PLM probability and fitness values in the context of meta-learning, especially when the authors assume that this correlation may not be reliable.

2-Although in-context learning is a strength, its effectiveness diminishes for tasks that deviate significantly from the distribution of tasks used during meta-training, potentially limiting generalization to out-of-distribution tasks. Experimental results on some OOD tasks can help readers better understand the paper.

3-The paper primarily compares Metalic to a few gradient-based meta-learning methods (e.g., Reptile) but does not extensively benchmark against other recent meta-learning approaches that incorporate different strategies for handling fine-tuning or leveraging unsupervised data.

4-Metalic shows strong performance on single-mutant tasks but falls short of the state-of-the-art on multi-mutant tasks. The authors attribute this to the limited availability of multi-mutant training data, emphasizing the need for more diverse datasets to fully realize the method's potential. A more in-depth analysis of this limitation would be valuable.

5-While the ProteinGym benchmark is a standard for protein fitness prediction, additional evaluations on other benchmarks would provide a more comprehensive understanding of Metalic's strengths and weaknesses across a broader range of protein-related tasks.

### Questions
1-The authors assume that the correlation between PLM likelihood and protein fitness may not be reliable. How does Metalic perform when the assumed correlation is weaker? Would additional analysis on this correlation help refine the model's approach?

2-In-context learning relies heavily on the distribution of tasks used during meta-training. How does Metalic handle tasks that are significantly different from the meta-training distribution, and are there strategies that could improve its generalization to such out-of-distribution tasks?

3-Given that Metalic does not account for fine-tuning during meta-training, how does this omission affect the model's ability to adapt to tasks with extensive fine-tuning requirements? Would incorporating some level of fine-tuning awareness during meta-training improve results?

### Soundness
3

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
This paper introduces Metallic, which performs few-shot (N=16 or N=128) or zero-shot fitness prediction using in-context meta-learning, using the ProteinGym benchmark. The model is meta-trained across many protein prediction tasks, to enable in-context learning, then finetuned on the support set, where the support set defines the number of "shots" for the few-shot task. Evaluations uses pairwise ranks (i.e. if model correctly prefers one mutant to the other). The meta-training procedure yields models that fare better in the finetuned few-shot setting than baselines (ESM1, etc.)

### Strengths
* The idea is relatively under-explored in protein ML despite ICL having shown strong performance in other fields of ML, and the importance of few-shot learning.
* Paper is fairly clearly written
* Attention map analysis is a great addition to the work! Would love to see follow up on this.

### Weaknesses
* In Table 1, 2, and 3a, only ESM2-8M is used for comparison. What happens if we compare to the larger ESM2 models? Can we get better zero-shot or few-shot performance via scaling pretraining parameters?
* Results in Figure 3b and 3a don't seem too much better than baselines.
* More rigorous examination of the effects of varying query set size, etc. might be quite insightful
* Some papers have pointed out that protein likelihoods don't always indicate protein fitness, and might be due to the training data biases [1,2].

Minor / meta comment:
I think that though thinking about ICL for proteins is a good idea, and authors demonstrate thoughtfulness in how this should be done. However, there might be more interesting tasks to examine than likelihood & protein fitness. There has also been working showing that "evotuning" (finetuning on evolutionary neighbors) can improve performance [2,3]. In a real-world setting, if I have a protein for which my pretrained model is underperforming on, I'm not sure if I would prefer to use a meta-trained model or a evo-tuned model.

From a writing perspective, authors might be able to increase impact by outlining real-world drug discovery scenarios where this is relevant.

[1] Protein language models are biased by unequal sequence
sampling across the tree of life. Ding et al., 2024
[2] Protein Language Model Fitness Is a Matter of Preference. Gordon et al., 2024
[3] Unified rational protein engineering with sequence-based deep representation learning. Alley et al., 2024

### Questions
* Are there some tasks that are easier to learn vs. harder to learn? It would be interesting to see a breakdown analysis.
* Do authors have hypotheses for why the performance gap between Metalic and baselines is so much more pronounced for zero-shot vs. few shot tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work introduces Metalic, a framework for performing protein sequence fitness prediction in both in-context and few-shot regimes. It proposes to use in-context-learning as a replacement for gradient updates used in the traditional inner loop of meta-learning. While using the smallest ESM-2 model, it achieves strong performance on the chosen fitness tasks.

### Strengths
Metalic achieves success in both providing a compute efficient alternative to meta-learning and also provides strong performance on downstream tasks. It’s also noteworthy the parameter efficiency that Metalic achieves in the process.

The work is also highly reproducible. Each experiment could be reproduced by a reader from the text alone (barring which datasets were and were not used). In the appendix, readers can find a comprehensive set of hyperparameters and training details used in each experiment.

When comparing metalic, it uses relevant choices in baseline models and also analyzes where and how performance degrades (e.g. the case of multi-mutants). In addition, the ablations show the utility of the different components of the method, even displaying its ability to outperform traditional meta-learning methods like Reptile. The experiments are very thorough in justifying all choices made during modeling.

### Weaknesses
The only major weakness that appears in this work is the limited number of datasets used in evaluating the proposed methods. ProteinGym performance fluctuates a lot amongst tasks, hence it would be nice to decouple the comparatively small number of datasets, 13 in total, for evaluation from performance. A simple experiment selecting different splits and training replicates in a similar manner to a partial k-fold cross validation would be convincing.

On a more minor note, the exact implementation of the algorithm would benefit from a slightly more detailed treatment. Though it can be inferred from Figure 2a, it would benefit from some support either pictorially or mathematically in the exact computations that must be performed.

### Questions
* How does performance change when training and evaluating on a separate set of proteins?
* How do inference and training costs of Metalic compare to the Reptile ablations?
* To clarify the architecture, would it be possible to have a Vaswani et al. 2017 style figure in the appendix of the exact computations that occur?

Overall the work is quite strong, and would benefit from a simple experiment to further prove its robustness. 

Vaswani et al. 2017, "Attention Is All You Need"

### Soundness
3

### Presentation
3

### Contribution
3
