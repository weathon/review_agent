# Expert Divergence Learning for MoE-based Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
The Mixture-of-Experts (MoE) architecture is a powerful technique for scaling language models, yet it often suffers from expert homogenization, where experts learn redundant functionalities, thereby limiting MoE's full potential. To address this, we introduce Expert Divergence Learning, a novel pre-training strategy that explicitly encourages functional specialization among experts. Our method incorporates a label-driven auxiliary loss that leverages domain labels inherent in pre-training corpora to maximize the Jensen-Shannon Divergence between the expert routing distributions of different data domains. This optimization objective guides the model to develop diverged routing policies for varied domains and closer routing policies for the same domain, which leads to emergent and organized expert specialization. We validate our approach by pre-training MoE models of up to 15 billion parameters from scratch. Experimental results demonstrate that models trained with Expert Divergence Learning not only achieve a lower language modeling loss but also exhibit significant performance improvements across a diverse range of downstream benchmarks. Further analysis confirms that our method effectively mitigates expert homogenization and brings greater functional specialization, all with negligible computational overhead during training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles expert homogenization by maximizing routing divergence between domains using their inherent labels as supervision. Results show improvements over baseline MoE across different scales, with larger models and finer domain granularity yielding better gains.

### Strengths
The core idea is well-motivated. Using domain labels already present in pretraining data to explicitly guide expert specialization addresses a real problem in MoE training. The theoretical framing through divergence decomposition is clean, and the experiments are thorough across multiple model scales with consistent improvements.

### Weaknesses
(1)  While the authors demonstrate improvements across three model sizes, the largest model evaluated is only 15B parameters. Given that many SOTA MoE models operate at larger scales and authors have the enough GPU resource (64*80GB), it remains unclear whether the observed benefits would translate to larger MoE LLMs. 

(2) The method fundamentally relies on having meaningful domain labels for the training data. However, the paper provides limited analysis on how sensitive the approach is to labeling noise or granularity mismatches. For example, what happens when documents contain mixed topics, or when the domain classifier makes systematic errors? The Chinese classifier was trained on only 500K examples from DeepSeek-r1, which may introduce biases that propagate through training. 

(3) The paper positions itself against standard MoE training but doesn't compare with other recently proposed solutions to expert homogenization. Shared expert architectures (referenced in the related work) represent a different paradigm for handling commonalities across domains. Similarly, the concurrent ERNIE 4.5 work on router weight orthogonality is mentioned but not empirically compared.

(4) The improvement from 3-class to 49-class domain schemes is relatively modest (36.34 to 36.65 average score for 15B). This raises questions about the cost-benefit tradeoff of investing in high-quality, fine-grained topic classifiers. The paper would benefit from an analysis of the relationship between domain granularity and performance gains, perhaps including intermediate granularities (e.g., 10-class, 20-class) to identify diminishing returns.

(5) All evaluation benchmarks fall roughly within the scope of the three training domains (English, Chinese, Math). The paper doesn't test how the model performs on domains significantly different from the training distribution. For example, code generation or scientific reasoning. If specialized routing patterns become too rigid, there's a risk of reduced flexibility when encountering truly novel inputs.

### Questions
Table 3 shows minimal throughput reduction, but this only captures the direct computational cost. The paper doesn't account for the infrastructure needed to maintain domain labels, the cost of training and running domain classifiers at scale, or the additional memory overhead of tracking domain-specific statistics during training.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
When training mixture-of-experts, the router learns correlated experts, i.e. groups of experts that are often active at the same time. This routing configuration is not optimal, i.e. lower training loss can be achieved with routing that has less correlated experts. This paper 
propose to promote the JS divergence between routing patterns across different domains. Positive impact on the training loss and on end-task performance is shown with the proposed strategy.

### Strengths
1. The paper gives a good overview of related work in routing strategies. 
2. The experiments are performed over a variety of model sizes and multiple end-tasks are considered for evaluation. Performance is reported both in terms of loss/perplexity and end task performance.
3. The method is simple and clearly presented.

### Weaknesses
1. The authors do not show whether better results could be achieved with a different routing strategy. So far JSD is only tried on top of z-loss with alpha=1e-3. Is it the best alpha? Would JSD be impactful on top of a different alpha? Is JSD impactful when the routing model has also always-active FFNs? Is JSD impactful when one uses another way to promote uniform expert assignments, e.g. DeepSeek bias based balancing https://arxiv.org/pdf/2408.15664? 


2. It is not clear whether the domain classifier is necessary. The best results are with 49 classes. From your batch size it seems that a batch contains only a couple documents per domain. Would it be simpler to define each document as its own domain and just push the JSD across all pairs of examples.


3. It would be good to quantify how expert assignment changes with the proposed regularizer. When one takes a pair of documents, what is the typical number of experts per document? What is the number of experts shared by the two documents? Are these numbers the same across depths?

### Questions
1. Are you working to compare your JSD regularizer with the orthogonalization method of Baidu ERNIE 4.5?
2. Are classes necessary? See weaknesses 2.
3. Does the JSD regularizer have a negative impact on the balancing regularizer L_LB? When training with JSD, are consecutive routing decisions for the same document more likely to overlap (promoting inter-example JSD might negatively impact intra-example JSD)?
4. How did you pick the L_LB regularizer parameter alpha? 
5. Could JSD be useful only at the beginning of learning? I.e. once a good initial configuration is found, training can be pursued without JSD regularization. It seems the gap between standard and JSD training is not growing (Fig 2) what happens if JSD is turned off after ~20k steps?
6. Since the largest number of classes give the best results, could better results be achieved with more domains? If the batch size is small compared to the number of classes, rarely the training algorithm will see two documents of the same class in the same batch. The method is equivalent to one class per document. Have you tried one class per document? Does your method work with smaller batch sizes?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an additional loss for the training of transformer MoE models that promotes expert specialization per domain. In more detail, the loss maximizes the pairwise JS divergence between the distributions over experts (averaged per token for each domain). The paper shows that training language models using this loss in addition to the language modeling loss and the expert balancing loss results in better perplexity and better downstream performance. Finally, the authors analyze the behavior of the routers in such models and show some evidence that there is more per-domain specialization.

### Strengths
The most important (albeit possibly weak as discussed in weaknesses) strength of the method proposed in this paper is the experimental results of section 4.2, namely the fact that adding this loss does indeed improve consistently over training the same model without using this loss.

In addition,

- The paper is very well written and presented.
- The idea is both simple and intuitive. It can probably be implemented in a few lines of code in any framework.
- The authors have covered all possible aspects of the impact of their loss to the model and its training. In particular, I appreciate the analysis on training and inference throughput.

### Weaknesses
Given that the idea is simple (which I consider a major strength) the weight falls to the experimental results to convey consistent improvement and at the very least that the method has the intended impact on the model.

However, the downstream evaluations have several relatively worrying qualities. Firstly, the average scores are compared and the conclusion that 49 domains is better and bigger models are more amenable to per-domain specialization. Looking closely at the scores though, there are many cases where the 3 domains are better and where the baseline is better than one of the two contenders. Including the fact that the differences are relatively small for evaluations that have quite high-variance from checkpoint to checkpoint it makes it very hard to draw the conclusion that the loss leads to consistent improvement.

A further minor point for the downstream evaluations is that the scores do seem a bit low. For instance 100B tokens for a 15B/1.5B MoE should probably achieve more than 56.6 on Arc easy. Having said that, the dataset could be the reason for this discrepancy so I do not consider this a major issue. Another minor concern is the fact that the largest model performs worse on 2 out of 7 benchmarks compared to the model with half the parameters.

The plot of the log-likelihood also raises a few concerns. The differences are relatively small and even though the loss seems to improve the log-likelihood, we only see from ~35B to ~65B tokens. What are the final losses at 100B and how does the training compare in the other stages of training (that would only be the middle).

Regarding section 4.3 that analyses the effects of the loss on the model. Permuting the distribution of the router and measuring ΔPPL measures how sensitive the model is to the particular expert choices but it is not clear how it measures per-domain specialization. Simply put a model that had collapsed to always select a single expert would get the worst score on that test and its experts would be anything but specialized.

The heat-maps more directly measure the intended effect of the loss to the model. However, I would propose to the authors to find both a better way to visually show whether 49-class Div vs the Baseline are per-domain specialized. To more clearly point to the problem, specialization can be seen for layer-0 and layer 14 but not for layer 4 which according to the previous test is the most-specialized. I suggest that the authors try to measure the portion of experts that are mostly selected on each domain. For instance if there are experts specialized for a particular domain it should be clearly visible in the inverse distribution eg how often is an expert selected for a token from a particular domain.

### Questions
I have mostly laid out my questions in the weaknesses section but maybe a small recap:

- What is the per-checkpoint variance in these evaluations? How can we increase our confidence that the improvement is real and consistent given that the scores are so close?
- Can we see the final loss and the evolution from 65B tokens and afterwards?
- Is ΔPPL is the correct metric? How does it measure per-domain specialization?
- Can we see different visualizations that clearly show that the experts are specialized per domain? ie some experts should be way more probable for math than for English and this should be clearer or stronger when using the loss.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a pre-training strategy to increase expert divergence in MoEs, hence, specialization through MoE experts. They use domain labels from the subsets of pretraining data and maximize the pairwise Jensen-Shannon divergence between expert routing distributions of data domains. They show that their method leads to better language modeling loss and also higher downstream performance.

### Strengths
1. Specialization in MoEs is often underexplored, but it is highly important for modularity and effectiveness. 

2. The idea is quite novel and very intuitive. 

3. A clear trend in language modeling loss suggests that the method proposed carries a strong potential.

### Weaknesses
1. I think the main weakness of the paper is the results, where there is no clear significant improvement in the majority of downstream tasks. More concretely, there are some items leading me to suspect the results: 
a. Although the model size increases significantly (both total and active params), there is no monotonic increase in evals with large model size (except ARC_e). 
b. Improvement claimed by the proposed method is quite uneven, mainly concentrated on ARC_e
c. Results are too close to draw a conclusion. There needs to be a significance test or multiple runs with different seeds to validate the overall improvements. 

2. The routing is token-wise; however, the proposed auxiliary loss is sequence-wise. It is not clear whether this could lead a suboptimal routing for MoE, which needs to be examined and showcased.  

3. For the additional question, I refer to the "Questions" section.

### Questions
1. How sensitive is the Expert Divergence Learning to differences in data volume in the domain dataset? It is common that some domains might be much smaller than others in an actual pretraining run.

2. How to ensure balancing across data labels in a single batch? Is the proposed method sensitive to such balance?

3. It is very hard to understand how the specialization is realized from the current experiments (For example, 3-class div still has>60 experts to specialize). Would it not be good to decrease the expert number to show specialization by removing the specialized expert on one domain in the test time (such as Chinese)?

### Soundness
3

### Presentation
3

### Contribution
2
