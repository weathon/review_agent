# A Decoupled Learning Framework for Neural Marked Temporal Point Process

- Decision: Reject
- Scores: 5, 5, 3, 6

## Abstract
The standard neural marked temporal point process employs the EmbeddingEncoder-History vector-Decoder (EEHD) architecture, wherein the history vector encapsulates the cumulative effects of past events. However, due to the inherent imbalance in event categories in real-world scenarios, the history vector tends to favor more frequent events, inadvertently overlooking less common yet potentially significant ones, thereby compromising the model’s overall performance. To tackle this issue, we introduce a novel decoupled learning framework for neural marked temporal point process, where each event type is modeled independently to capture its unique characteristics, allowing for a more nuanced and equitable treatment of all event types. Each event type boasts its own complete EEHD architecture, featuring scaled-down parameters due to the decoupling of temporal dynamics. This decoupled design enables asynchronous parallel training, and the embeddings can reflect the dependencies between event types. Our versatile framework, accommodating various encoder and decoder architectures, demonstrates state-of-the-art performance across diverse datasets, outperforming benchmarks by a significant margin and increasing training speed by up to 12 times. Additionally, it offers interpretability, revealing which event types have similar influences on a particular event type, fostering a deeper understanding of temporal dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a neural modeling approach that independently handles event-specific intensity functions by decoupling the training of event-specific encoder-decoder architectures. This method is designed for time series data, focusing on event times and types, and allows for parallel training. Experimental results on four real-world datasets demonstrate that applying this approach to two previously proposed neural marked temporal process models improves their performance. These models originally relied on a joint intensity function with shared encoder-decoder architectures. The proposed decoupling approach enhances the accuracy of event time and type predictions for these models, speeds up training, and reduces the number of parameters needed.

### Strengths
- The paper focuses on the important and impactful problem of modeling event time/type prediction given irregularly sampled time-series data.
- The paper is relatively well-written and easy to follow.
- The proposed decoupling approach seems easy to implement and yields performance improvements over two previously proposed neural marked temporal point process models in terms of event type/time prediction, training speed, and reduced model parameters.

### Weaknesses
- The proposed approach seems quite limited in novelty, as it essentially boils down to decoupling the encoder-decoder of previously proposed neural marked temporal process modeling approaches.
- The paper's central claim that models focusing on joint-likelihood estimation of marked temporal point process data fail to account for rare event times compared to the proposed decoupled independent likelihood estimation does not seem substantiated, both in terms of theory and experimental results. Experimental results show general performance improvements (Table 2), but predictions for less frequent events do not seem to improve in general (Table 3). Furthermore, it is unclear if the proposed approach results in statistically significant improvements, as no error bars are provided and only two previously proposed methods are considered for decoupling.
- The proposed independent likelihood approach makes a strong conditional independence assumption, which could be violated in practice; that is, given the past event times and types and the parameters of each specific intensity function, the occurrence of an event is assumed to be independent of other events. This approach may fail to capture the complex dependencies and interactions between different event types that a joint likelihood model could naturally represent.
- The paper does not benchmark against approaches that explicitly model event-specific triggering kernels (unlike RNN or transformer-based approaches) within a joint likelihood, including [1, 2]. Without such comparisons, it is difficult to ascertain the significance of this work.

**Minor**
- Table 1: Add definitions of the hyperparameters to the caption. What is $d_h$?
- Lines 281–292: Make $z$ bold symbol since these are vectors, for consistency.
- Lines 122–133: Define $m$.


**References**
-  [1] Yamac et al. (2023), "Hawkes Process with Flexible Triggering Kernels", MLHC.
-  [2] Pan et al. (2021),  "Self-adaptable point processes with nonparametric time decays", NeurIPS.

### Questions
- Table 4: Could you clarify why decoupled encoder-decoder models result in fewer parameters compared to their counterparts? Could you provide the experimental results when the encoder-decoder parameters of the decoupled networks are equivalent to those of the shared networks?
- Figure 3: It's difficult to compare the provided matrices. Given that the data is generated from a known Hawkes process, could you provide plots for the predicted triggering kernels $\phi_k(k_i, t_i)$ versus the ground truth for all the models?
- Table 2: Could you provide results of the decoupling approach applied to all the baselines instead of just to IFL and THP, along with the corresponding error bars?
- Could you benchmark against alternative flexible approaches for modeling triggering kernels, including [1, 2]?
- Theorem 2: I don't think the joint likelihood approach is equivalent to the independent likelihood unless the conditional independence assumption holds; i.e., given the past event times and types and the parameters of each specific intensity function, the occurrence of an event is independent of other events. Could you clarify the equivalence of the decoupled versus shared learning approaches? Also, if the theoretical equivalence holds, why do the empirical results indicate improved performance of the decoupling approach?

### Soundness
2

### Presentation
2

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
This paper proposed a novel decoupled learning framework for neural marked temporal point processes that effectively mitigates the issue of frequency bias.  Each of the types has a vector representation in each latent space. These are used to model the influence of one event type on the same or another event type. Each event type has its own encoder and decoder. By modeling each event type separately within a complete EEHD architecture, this approprach also enables asynchronous parallel training, improving the training speed, but also allows the embeddings to capture the intricate dependencies between event types.

### Strengths
(1) The proposed framework disentangles traditional monolithic modeling with event-type-specific individual modeling, which mitigates the issue of frequency bias while also providing more interpretability than neural TPPs.

(2) The experiment results demonstrate a significant enhancement in training speed and better interpretability on both real-world and synthetic datasets.

(3) The authors show that the proposed framework is general enough by proving that the proposed decoupled learning framework is theoretically equivalent to the standard learning framework.

### Weaknesses
(1) In the proof of the equivalence of the proposed decoupled learning framework and the standard learning framework, the statement could be more clear: when proving the proposed framework is no weaker in expressive power than the standard learning framework, verify "let NNk in Equation 6 equal to that in Equation 5" more specifically; when proving the standard learning framework is no weaker in expressive power than the proposed framework, verify "let NNk in Equation 5 equal to that in Equation 6" more specifically.

(2) Based on the Theorem 2 of the paper, the expressive power of the proposed framework is equivalent to the standard learning framework, then it would be equally useful when using these two frameworks with enough parameters and appropriate training in principle. The authors should have a clearer discussion about which model is more advantageous to use in which specific situations. One of the motivations of the proposed framework is to mitigate the issue of frequency bias, it would be better to also compare the performance of this issue within these two frameworks under different settings.

(3) When adopting the decoupled learning framework, the required effort for hyperparameter selection is nearly k-fold as for the standard learning framework, especially for highly unbalanced type case and personalized EEHD architecture tailored to each type. Though the hyperparameter can be set equally, the effects of this issue on the model performance should be discussed carefully and also verified by experiments.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a decoupled learning framework for neural marked temporal point process, where separate EEHD architecture is used for each event type, including learning multiple  event embeddings corresponding to modeling different event types. The proposed approach is reported to have improvements in terms of performance, training time and interpretability.

### Strengths
1. The paper proposes to address the problem of applying the standard neural marked temporal point process to real-world data sets where event types are unbalanced. This seems to be an important and novel problem to address. 
2. The paper is well written and easy to follow, though some modification in presentation would improve the readability. The experimental results are neatly presented. The Figures clearly demonstrate the proposed approach and the ingenuity of the result presentation (for instance ICLR in Figure 3) is appreciable.

### Weaknesses
1. The fundamental idea of decoupling and learning independently across the event types itself seems to be a major weakness. 
2. The approach results in linear growth in number of EEHD architecture parameters with respect to event types. This may not be scalable. 
3. The approach itself can be a drawback in addressing the performance issue in event types with limited data. Due to decoupling, as separate model parameters are used for each event type,  the proposed approach may not be able to learn well from an event type with limited data. The standard neural marked point process architecture may not suffer much from this as it shares parameters across event types, consequently allowing knowledge across the event types. 
4. The approach might be useful for a specific application or data set, but proposing it as a general approach to solve an unbalanced data set case might be a bit far-fetched. 
5. The experimental results do not convincingly demonstrate the improved performance of the proposed approach, and perform poorly in some data sets and cases.  
6. Not having ablation studies with some variants of the proposed model (event specific embedding but common encoder-decoder, common embedding but event specific encoder-decoder etc. ) makes it difficult to understand what aspect of the decoupling helped in improving the performance. 
7. Experimental results does not follow a consistent pattern. The results with baslines are better at mimic 2, and in several places in Table 3, the proposed method didnt bring any improvements on event types with small number of events. The same is teh case with Figure 3 comparing learnt influence matrix with ground truth. There are mismatches in expected outcomes at several places.

### Questions
1. Discuss applications where the event types are unablanced, and provide motivation  from the application perspective
2. Any reference to support the statement that standard EEHD methods fail in the data with unbalanced event type
3. How is the performance when variants of the proposed model are used, event specific embedding but common encoder-decoder, common embedding but event specific encoder-decoder, common embedding and encoder but event specific decoder  etc.
4. Kindly provide sampling procedure as an  algorithm, to make it more comprehensible. 
5. In the discussion section, connection between standard Hawkes process and the proposed approach is established. But note that, standard Hawkess process has  global shared components while the proposed architecture has no shared components. Sharing of components helps in knowledge transfer and leads to an improved performance. 
6. Kindly provide more details on data set creation, is the 60/20/20 split followed per event or across all events, and how exactly each split is obtained from the event sequence. 
7. Why is small sequence length of MIMIC 2 a problem specifically for the proposed method ? 
8. For experimental comparison, a baseline considering only events of the same type can be considered to see if the proposed method is able to capture the inter-event influences and improve the predictions. 
9. Kindly provide an explanation (or equation) detailing computation of values in the learnt influence matrix.
10. Kindly report in Table 4, the total parameter counts of DEC-IFL and DEC-THP across all the event types. 
11. On paper presentation, I think it would be better to provide a separate background section discussing previous works and then discuss the proposed methodology.

### Soundness
2

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
4

### Summary
The authors propose to model event types in a marked point process individually to alleviate issues associated with (heavy) event sitribution imbalance. Specifically, each event is modeled via a embedding-encoder-history vector-decoder (EEHD), thus models for each event type can be learned in parallel. Experiments on real-world and synthetic datasets demonstrate that the proposed approach achieves state-of-the-art performance in prediction tasks while significantly increasing training speed by a factor of 12 relative to the monolitic EEHD architecture.

### Strengths
The proposed approach has two advantages: i) it is very simple by essentially treating the modeling of intensity functions completely separately; and ii) requiring model specification with less parameters thus making the training process faster.

### Weaknesses
The authors show in Theorem 2 that the proposed and standard learning frameworks are equivalent, which then implies that there is not a reason for the proposed model to perform better (in general) than the standard model. One may argue that the proposed model is easier to train, but the theorem basically guarantees that one can always find a parameterization of the standard model that matches that of the proposed model and vice versa.

The experiments presented by the authors have several issues:
- The choice of the hyperparameters for the models compared in Table 2 is not clear, especially for IFL, Dec-IFL, THP and Dec-THP. For instance, embedding sizes are too different between the standard and the proposed approach, and though there are ablation results in Tables A3 and A4, they consider a very limited range for d. Also, there is no language about learning parameters and regularization strategies.
- The results in Table 4 also need additional explanation. For instance, the text reads that for the proposed approach training speed is "the lowest for K decoupled models", however, how were models trained, serially or in parallel, and how it is guaranteed that training speeds across models are comparable?
- The experiments in Table 2 (and presumably Table 3) are averages over 10 runs but their variation is not reported, which will be important to understand the significance of the performance differences.
- The authors emphasize that the proposed model is advantageous in situation where event incidence is imbalanced, however comparisons with approaches that address imbalance (some of which are mentioned in the related work) are not considered.

### Questions
It will be useful to further explain Figure 3 because as is, the point the authors are trying to make is not very clear. Also, how do these compare with the standard model?

How is the NLL calculated in the experiments? Further, one wonders why NLLs do not seem to be consistent in Table 2, Figure 4 and Table A3?

### Soundness
2

### Presentation
3

### Contribution
2
