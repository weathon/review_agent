# Scaling Laws for Online Advertisement Retrieval

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
The scaling law is a notable property of neural network models and has significantly propelled the development of large language models. Scaling laws hold great promise in guiding model design and resource allocation. Recent research increasingly shows that scaling laws are not limited to NLP tasks or Transformer architectures; they also apply to domains such as recommendation. However, there is still a lack of literature on scaling law research in online advertisement retrieval systems. This may be because 1) identifying the scaling law for resource cost and online revenue is often expensive in both time and training resources for industrial applications, and 2) varying settings for different systems prevent the scaling law from being applied across various scenarios. To address these issues, we propose a lightweight paradigm to identify online scaling laws of retrieval models, incorporating a novel offline metric $R/R^\*$ and an offline simulation algorithm.  We prove that under mild assumptions, the correlation between $R/R^\*$ and online revenue asymptotically approaches 1 and empirically validates its effectiveness. The simulation algorithm can estimate the machine cost offline. Based on the lightweight paradigm, we can identify online Scaling Laws for retrieval models almost exclusively through offline experiments, and quickly estimate machine costs and revenues for given model configurations.  We further validate the existence of scaling laws across mainstream model architectures—including Transformer-based models, MLPs, and DSSM—in our real-world advertising system. With the identified scaling laws, we demonstrate practical applications for ROI-constrained model designing and multi-scenario resource allocation in the online advertising system. To the best of our knowledge, this is the first work to study the identification and application of online scaling laws for online advertisement retrieval, showing great potential for scaling laws in advertising system optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to establish laws based on empirical observations for predicting how would revenue change given a retrieval model's architectural parameters. The paper devises a metric called $R/R*$ that is claimed to correlate with revenue, propose fitting a BNSL-type scaling law to $f(\mathrm{FLOPs) \approx R/R*$, and then use the model's architecture to compute costs in order to estimate the cost-effectiveness of a model.

### Strengths
1. The problem is important. I am also not aware for good scaling laws for ad retrieval models, where the final ranking is **not** done by the model's predictions, but via the expected revenue.
2. The initial presentation of an ad ranking system is good and puts the paper into context.
3. Benchmarks do show some predictive power of the proposed laws.

### Weaknesses
1. The paper lacks soundness and convincing arguments. Here are a few examples:
   1. the training data talks about a vector $v$ of "ground-truth order", but how is this order determined? I guess it's by the eCPM, since I personally come from the ad industry and understand that the job of pre-ranking is selecting the candidates that have the highest probability of (I come from the advertising business myself), but it's not obvious what is the ground-truth order of a pre-ranking system. 
   2. The FLOPs scaling laws may be convincing for language models, where we have established practices for many architectural free variables, such as the fact that MLP layers have intermediate layer of 4 x embedding dimension, and others. But for a generic MLP, I do not see FLOPs as a reasonable proxy for scaling, unless you add architectural assumptions as well. And who said these assumptions are reasonable in general for ad ranking models?
  3. How exactly can A/B tests establish linear correlation? In A/B tests you're testing two candidates against each other. Here you just need to test many models, not against each other. If you **are** testing them against each other, it's the rank correlation that matters. If you're just collecting revenue samples, these aren't A/B tests. An explanation is in place here.
2. The training data is biased. You know the ground-truth order only for those that actually were selected by pre-ranking, but you don't know the order of those that weren't. Or are you constructing the training data synthetically not from real ad auctions? An explanation is missing. Either how you account for the bias, or how is the training data constructed.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether scaling laws also apply to industrial online advertising systems. The authors propose a lightweight offline paradigm to identify scaling laws between machine cost and online revenue for ad retrieval models. They introduce a new offline metric, R/R*, to be linearly correlated with online revenue. Using FLOPs as the scaling factor, the study confirms broken neural scaling laws across Transformer, MLP, and DSSM architectures in real-world ad retrieval systems. They further develop a machine cost simulation tool and apply the framework to ROI-constrained model design and multi-scenario resource allocation.

### Strengths
As real-world advertising systems become increasingly complex and large-scale, it is important to make changes that comply with ROI constraints. This paper proposes a heuristic scaling law to study the trade-off between cost and return. Overall, the paper is interesting and useful.

### Weaknesses
My main concern is that the writing could be substantially improved, as the current version of the paper is not accessible to the general ICLR audience.

1. The key concept $R / R^*$ is mentioned several times in the Introduction and Abstract, but it is never explained, even heuristically.
2. The definition of $R / R^*$ in Equation (2) is difficult to understand. What is the “hard permutation matrix”? It should be clearly defined, and a heuristic explanation would be helpful.
3. How is the ground-truth rank $v_i^j$ obtained? Is it derived from the final ranking stage?
4. The statement “If the following assumptions are met, we can prove that $R / R^*$ and online revenue have a linear relationship” is never formally presented as a theorem.
5. Typos: “Assumption Theorems” on Line 256; “Section Sections” on Line 36.

Moreover, I have additional comments on the paper:  
6. I do not think the linear relationship between $R / R^*$ and online revenue qualifies as a scaling law. In my view, Equation (4) represents a scaling law, but it is only heuristic.

7. In the case studies and applications, only the revenue gains are reported. However, given that the paper focuses on ROI, it would be important to also report the corresponding cost comparisons.

### Questions
See above

### Soundness
3

### Presentation
1

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
This paper investigates scaling laws in online advertisement retrieval systems, proposing an offline paradigm to predict machine costs and revenue from model configurations. The authors argue that traditional methods for finding scaling laws are prohibitively expensive in this domain, as they require numerous costly and time-consuming online A/B tests to measure the true relationship between machine cost and online revenue.

Key contributions include a offline metric R/R* that correlates with online revenue (theoretically asymptotic correlation of 1 under mild assumptions, empirically R²=0.902), fitting broken neural scaling laws (BNSL) across MLP, DSSM, and Transformer architectures in matching/pre-ranking stages, and a simulation algorithm for cost estimation. 

The authors demonstrate the existence of "broken neural scaling laws" (BNSL) for various architectures (MLP, DSSM, Transformer) in their production system. They then apply this framework to two practical applications: ROI-constrained model design and multi-scenario resource allocation, claiming a substantial 5.1% combined improvement in online revenue.

### Strengths
High Practical Impact and Significance: The paper tackles a critical and expensive problem for any large-scale industrial ML Retrieval system: how to perform cost-aware model development and resource allocation. The ability to accurately estimate the ROI of a model configuration offline is extremely valuable. The reported +5.1% online revenue gain from applying this framework is a very strong testament to its practical utility.

Holistic and Well-Designed Framework: The authors present a complete, end-to-end paradigm. The insight to decouple the problem into a revenue-surrogate and a cost-surrogate (MCET) is clever. The MCET tool has an interesting finding that FLOPs are an unstable proxy for real-world machine cost in highly-optimized, custom serving environments.

### Weaknesses
Limited Conceptual Novelty: The R/R* metric is functionally a "revenue-weighted recall." Can the authors comment on the novelty of this metric in the context of prior work on utility-based or business-value-weighted metrics in recommender systems and information retrieval? The paper's strength seems to be its empirical validation rather than the novelty of the metric's formulation. The contribution is more of an engineering one—successfully applying and validating this known concept in a new domain—rather than a fundamental research one.

Theoretical guarantees of R/R* rely on "mild" assumptions (e.g., proportional pathway improvements, invariant normalized contributions) that may not hold perfectly in heterogeneous real-world ad systems, potentially limiting generalizability; more sensitivity analysis on these assumptions would strengthen claims.

Experiments are rigorous but confined to one proprietary system and standard architectures (MLP, DSSM, Transformer), lacking comparisons to diverse ad platforms or emerging recsys scaling works.

### Questions
For the R/R* metric, Appendix B.3 shows a sensitivity analysis for m (the cutoff) from 1 to 6, and Table 1 uses m=2. How was it chosen? Is it tuned on a validation set to maximize correlation, or does it correspond to a physical property of the system (e.g., the number of ads the Pre-ranking stage actually sends to the Ranking stage)?

The paper motivates MCET by arguing that FLOPs are a poor proxy for machine cost. However, the revenue side of the paradigm relies on FLOPs as a reliable intermediate variable (Model Config -> FLOPs -> R/R*). Why are FLOPs a stable predictor for R/R* (model performance) but not for machine cost (model speed)?

### Soundness
2

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
2

### Summary
This paper establishes scaling trends between cost of serving an ad retrieval model vs expected online revenue for a online ad platform. More specifically, the paper makes 3 contributions - (1) it establishes R/R* (a normalized @k metric which measures the expected revenue of a model's top m predictions vs production system's expected revenue) as a robust offline proxy for online revenue as compared to standard nDCG & recall metrics; (2) it plots relationship between flops of a model vs the R/R* metric on offline data for multiple model sizes and configurations; (3) it presents practical methodology for translating a configuration of a model to its actual deployed cost which is helpful in computing ROI of a model setup. Through extensive experiments on internal closed data, the paper shows improved revenue gains when using optimal models suggested using the scaling trends established in the paper.

### Strengths
1. The paper is well structured and the writing is clean and easy to follow
2. The proposed methodology for establishing scaling laws is well principled
3. The paper presents proper justifications for key design choices taken

### Weaknesses
1. Closed evaluation - I acknowledge that the nature of the problem tackled in the paper makes it difficult to present these results in an open setting but nonetheless because of all experiments being closed source/data makes it impossible to replicate or reproduce.
2. Scope of the results - I am unsure if the results presented in the paper hold in other settings or is of interest to the ICLR community which as per my understanding focuses more on learning algorithms, architecture or a better understanding of ML models.
3. Theoretical justification assumptions are not mild - the theoretical arguments made in the paper in my opinion take a significant leap of faith (such as predicted eCPM being same as actual revenue numbers, only the top-m ads (where m~1-6) from the retrieval stages will be selected by the post-stages for exposure)

### Questions
1. Can the authors elaborate why MCET module is so expensive and what is a typical relation between model size and the associated machine cost predicted by the MCET?
2. Why do the authors restrict there analysis for matching and pre-ranking stage?

### Soundness
3

### Presentation
4

### Contribution
2
