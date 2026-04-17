# PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to $40\%$ in NDCG and fairness by $30\%$ in entropy score over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by $7$ times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes PET, a framework for user preference estimation via inferring a probability distribution over a fixed size of clusters that represent item classes. The proposed approach relies on unsupervised pre-training followed by supervised fine-tuning which approximates attractiveness scores for each of the clusters. During inference, the authors investigate likelihood-based probing compared to generative classification as well as hierarchical probing in case of very large cluster sets. Furthermore the authors provide a theoretical result based on optimality of the ranking via likelihood-probing based on certain assumptions. The experimental results showcase that likelihood-probing usually performs best, hierarchical probing performs well for large cluster sets and PET outperforms a sota embedding model for ranking. Finally, the authors provide improvements over a system used in production and show qualitative examples of evolving user preference clusters over time.

### Strengths
- The performance of PET seem strong compared to the baseline and the production system.
- The paper is easy to understand

### Weaknesses
**Conceptual framing**

The paper relies on the assumption that user preferences can be accurately captured merely by item categories and user history. I am not convinced that this information is sufficient. For example, recent work has shown the benefit of adding user reviews to approximate user preferences in a textual manner [1,2,3].  Furthermore, datasets like MovieLens actually provide additional information on user profiles that are neglected in its entirety in this work. Finally, the authors claim that PET constructs a user profile, but actually it is simply a sequence based classification based on clustering.

[1] Review-driven personalized preference reasoning with large language models for recommendation, Kim et al., SIGIR 2025

[2] Preference Discerning with LLM-Enhanced Generative Retrieval, Paischer et al., TMLR 2025

[3] USER-LLM: Efficient LLM Contextualization with User Embeddings, Ning et al., WWW 2025

**Theoretical results**

The theoretic optimality of PET ranking relies on a way too strong and unrealistic assumption, namely that the transformation from attractiveness score q to average logit score S is monotonic. While this assumption holds for the softmax function, there is an LLM in between that produces the logits for which such an assumption is not reasonable. This is corroborated by results showing that PET does not achieve the optimal ranking, which would be the case if this assumption was true. 
I recommend to either (a) entirely remove the theoretical results as they are inconsistent and rely on a strong and unrealistic assumption, or (b), try to relax the assumption to an approximate or probabilistic monotonicity condition and re-derive the proof.

**Presentation**

The authors make plenty of claims around interpretability, unbiasedness, and fairness. However, the authors only show performance measures in terms of NDCG/Recall and no empirical evidence that supports improved interpretability/fairness. Especially claims on interpretability are heavily overclaimed as preference clusters merely consist of dataset-specific item catagories. The time evolution thereof is something that any other recommendation system can provide by visualizing item category distributions of predictions. Therefore I strongly recommend to caveat the claims on interpretability/fairness. 

Another strong line of claims is that PET was designed for recommending niche/long-tail interests, but it is unclear to me what design choices in particular facilitate recommendation of long-tail interests? In particular, many other recommendation systems, especially generative retrieval based ones [1], support that by simply increasing the temperature during decoding, why is this special in PET? 

[1] Recommender Systems with Generative Retrieval, Rajput et al., NeurIPS 2023

**Baselines**

The authors only compare to different decoding/prompting schemes of the same model and to one other model, but do not consider state-of-the-art generative recommendation systems in the literature. In this regard semantic-ID based methods [1,2,3,4] would be particularly interesting as they promise to provide a similar hierarchical clustering of items as handcrafted for Table 1. Furthermore, other logit-based ranking methods already exist which could also be included in the comparison [5]. Finally, comparing to a production system (Table 2) without disclosing any details about its implementation does have minimal scientific value. Therefore I recommend including the suggested baselines.

[1] Recommender Systems with Generative Retrieval, Rajput et al., NeurIPS 2023

[2] Preference Discerning with LLM-Enhanced Generative Retrieval, Paischer et al., TMLR 2025

[3] Unifying Generative and Dense Retrieval for Sequential Recommendation, Yang et al., TMLR 2025

[4] Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation, Zheng et al., ICDE 2024

[5] Generative Sequential Recommendation with GPTRec, Petrov et al., SIGIR 2023

**Experiment Design**

The authors claim that Table 2 evaluates for long-tail interests, but the experimental design of this scenario does not necessarily require modelling long-tail interests. In fact, this experiment is about long-term predictions (7-14 days), but there is no supporting analysis of this evaluation scenario that ground-truth items in that duration are actually long-tailed ones and not also popular ones. Supporting this claim requires some evidence, like showing the distribution of items of test splits compared to training splits.

**Minor Remarks**

- Table 1: Logit Probing should be replaced with Hierarchical probing, otherwise it makes the wrong impression that vanilla logit probing is used here.

### Questions
- What is meant by the task of user segmentation?
 - Line 127: What is meant by a "transparent distribution"?
 - Line 283: What is the average lifetime for a user for the different datasets? How much longer are those timelines compared to next-item prediction?
 - Line 318: Does the max of 8 sessions mean that the context window at most comprises the last 8 interactions of the user? If so, why was this threshold chosen? Wouldn't it be more beneficial to allow longer context windows especially for long-term predictions?
- Table 1: are the scores only based on predicting the correct L1/L2 cluster? or also for predicting the correct item in L2 cluster? if it is only on abstract cluster level, the item-level scores should be added, prediction on cluster level is a lot easier than predicting items within the cluster.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Preference Evolution Tracking (PET), a new framework that reframes LLM-based personalization by moving beyond direct item ranking. By inferring a dynamic probability distribution over a stable lattice of interpretable preference clusters, PET is designed to achieve a more holistic understanding of user interests. This distributional profile is inferred using scalable techniques like Likelihood-based Probing and Generative Classification, while Hierarchical Probing ensures the framework can handle thousands of clusters in real-world applications. A crucial outcome of the model is a transparent and portable user profile, which serves as an auditable asset that mitigates popularity bias by effectively capturing and ranking users' niche, long-tail preferences. Comprehensive evaluation on public benchmarks like MovieLens and Yelp demonstrates NDCG improvements of up to 40% over direct generation, and validation on an industrial-scale dataset reveals a 7-fold improvement in ranking long-tail content over a state-of-the-art production model. Further analysis confirms PET's ability to generate interpretable, evolving user profiles, adeptly capturing the dynamics of both stable, long-term interests and volatile, context-dependent preferences over time.

### Strengths
1. The paper is well-structured with clear writing and is supported by rich and lucid illustrations that effectively communicate the framework and results.  

2. The proposed paradigm shift from direct item ranking to distributional preference inference is a significant conceptual contribution that thoughtfully addresses well-known and critical challenges in modern recommender systems, such as popularity bias and lack of transparency.  

3. The framework's practical value is substantiated by exceptional empirical results, particularly the substantial performance improvement on long-tail content in a large-scale industrial setting, which provides compelling evidence of its real-world impact beyond academic benchmarks.  

4. The methodology is practical and scalable, offering a thoughtful suite of inference strategies (Likelihood-based Probing, Generative Classification) and a pragmatic solution for high-dimensional spaces (Hierarchical Probing) that shows clear consideration for deployment challenges.

### Weaknesses
1. **Ambiguity in the Mechanism for Countering Popularity Bias:** A central claim of the paper is that the PET framework effectively counters popularity bias and improves the discovery of long-tail user interests. However, there is a conceptual tension in the methodology that requires further clarification. The ground-truth "proxy preference distribution," as defined in Equation (3), is calculated from the normalized frequency of a user's historical 
interactions within each cluster. Since interaction frequency is inherently a measure of popularity (albeit at a personal level), the model is being trained to replicate a signal that is itself influenced by popularity. The paper would be substantially strengthened if the authors explicitly elucidated the mechanism by which a model trained on this frequency-based target can successfully counter global popularity bias.

2. **Ambiguity in the Validation of Long-Tail Preference Capture:** While the results of proposed methods in Table 2 are impressive, the manuscript does not provide a clear and explicit definition of how the "long-tail segment" of TikTok dataset was constructed for the evaluation.  For the reader to fully assess the significance of this core claim, it is crucial to understand what constitutes the "long-tail" in this experiment

### Questions
1. **On the Rationale for Hierarchical Probing's Top-Down Approach:** Could the authors elaborate on the rationale for Hierarchical Probing's top-down (coarse-to-fine) strategy, as a bottom-up approach might seem more intuitive? What are its specific advantages for computational tractability and scalability in extreme multi-label classification scenarios?

2. **On the Mechanistic Difference Between Generative Classification and Direct Generation:** Given the similar prompts, could the authors provide a simple example to illustrate the output difference between Generative Classification and Direct Generation

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a framework to model evolving user preference of LLMs in an interpretable manner. By applying logit probing and generative classification techniques, it models user preference as a probability distribution over interpretable clusters instead of generating ranked lists of items. Experiments are performed on MovieLens, Yelp, and a proprietary TikTok dataset to validate the method.

### Strengths
1. The idea of interpretable and distributional personalization with LLMs sounds to be interesting. Authors reframes LLM-based personalization from direct ranking to distributional inference. This conceptual shift allows interpretability. 
2. Empirically, the performance gain in terms of NDCG is appealing.

### Weaknesses
Overall, the paper’s core value lies in its interpretability and long-tail performance, not in fundamental algorithmic contribution.

1. The core methods introduced in section 3, i.e. logit probing and generative classification, are directly borrowed from existing interpretability work (Petroni et al., 2019; Schick & Schütze, 2020). The main contribution is contextual repurposing to user modeling, and the technical novelty is limited in terms of methodology.

2. The claim of fairer personalization is qualitative. Evaluations on diversity/fairness, which are central to the paper’s motivation, are not quantitatively analyzed (no metrics such as coverage, entropy, or calibration).

3. Authors attempted to provide theoretical arguments for the proposed methodology, but the theoretical substance is limited. 
- The proof in Appendix D is rather heuristics-based and not rigorous in any sense. 
- The “optimality” theorem (Lemma 1) is straightforward and expected under the isotonic assumption, which essentially assumes what needs to be proved (i.e., correct ordering of logits). 
- In terms of the theoretical contributions, it did not provide insights in terms of learning or generalization guarantee of the methodology. 
As such, to me, section 3.3 is insufficiently grounded and would benefit from a more rigorous justification of its claims.

4. The computational overhead of Likelihood-based Probing appears large (O(K) forward passes), even with KV reuse. Author may want to include runtime comparisons to further demonstrate the practical applicability of the method.

5. While the TikTok dataset is proprietary and anonymized, current details on feature engineering, taxonomy creation, and production baselines are not clear from the manuscript. Authors may want to include more detailed info of the dataset.

I am willing to defend my decision if authors can sufficiently address the main weaknesses.

### Questions
1. PET assumes predefined cluster taxonomies. How sensitive is performance to the choice or granularity of these clusters? Could automatic or learned clustering be incorporated?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper argues that while traditional large language model (LLM)-based recommender systems can predict users' next actions, they lack interpretability and tend to amplify popular preferences while neglecting long-tail interests. To address this, the authors propose the PET framework, transforming the user modeling task from "generating recommendation lists" to "inferring the probability distribution of user preferences." Through logit probing and generative classification methods, the model can infer the probability distribution of user preferences across different preference categories from their history, thus achieving interpretable, traceable, and fair personalized recommendations. Theoretically, the method is proven to achieve optimal ranking results and solves the problems of multi-label and large-scale categories through a hierarchical inference mechanism. Overall, PET provides a transparent, scalable, and interpretable new paradigm for user modeling.

### Strengths
Highly interpretable: PET does not directly predict specific recommendations, but instead outputs the probability distribution of users across different interest categories. This allows the system to clearly demonstrate "why users like certain types of content," enabling transparent and traceable recommendation decisions.

Capturing Preference Changes: Through time-sliding windows and distribution evolution modeling, PET can dynamically track changes in user interests over time, distinguishing between short-term fluctuations and long-term trends, thus better understanding the evolution of user behavior.

Reducing Popularity Bias: Because it models the preference distribution at the category level rather than the popular items themselves, PET can balance mainstream and long-tail content in recommendations, preventing the system from overly favoring popular items and improving fairness and diversity.

### Weaknesses
Lack of comparability with traditional recommendation models:
PET outputs a preference distribution rather than a specific recommendation list, making it difficult to directly compare with traditional recommendation systems (such as MF, SASRec, and BERT4Rec) on metrics like click-through rate (CTR) or NDCG, thus hindering the quantifiable evaluation of its practical effectiveness.

High computational and training costs:
PET is based on a Large Language Model (LLM), requiring training and inference on large-scale corpora and long-sequence behavioral data. This results in significantly higher computational costs than traditional deep recommendation models, making it unsuitable for deployment in resource-constrained environments.

Strong dependence on reward signals and category segmentation:
The model's accuracy depends on the quality of preference cluster segmentation and the reliability of implicit feedback (such as clicks and views) in the training data. Inappropriate category segmentation or noisy feedback can distort the preference distribution, affecting tracking performance and interpretability.

The methods lack uniformity and do not discuss other possible paradigms for solving recommendation problems. For example, interpretable recommendation systems can also be achieved through the R1 reasoning paradigm. Related works, such as R1-ranker and Rec-R1, need to be discussed and compared.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
