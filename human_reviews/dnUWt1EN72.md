# Dynamic Interference Modeling For Estimating Treatment Effects From Dynamic Graphs

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Estimating treatment effects can assist decision-making in various areas, such as commerce and medicine. One application of the treatment effect estimation is to predict the effect of an advertisement on the purchase result of a customer, known as individual treatment effect (ITE).  In online websites, the outcome of an individual can be affected by treatments of other individuals, as people often propagate information with their friends,  a phenomenon referred to as interference. Prior studies have attempted to model interference for accurate ITE estimation under a static network among individuals. However, the network usually changes over time in real-world applications due to complex social activities among individuals. For instance,  an individual can follow another individual on one day and unfollow this individual afterward on an online social website. In this case, the outcomes of individuals can be interfered with not only by treatments for current neighbors but also by past information and treatments for past neighbors, which we refer to as \emph{dynamic interference}. In this work, we model dynamic interference for the first time by developing an architecture to aggregate both the past information of individuals and their neighbors. Specifically, our proposed method contains a mechanism that summarizes historical information of individuals from previous time stamps,  graph neural networks that propagate information about individuals within every time stamp, and a weighting mechanism that estimates the importance of different time stamps. Moreover, the model parameters should gradually change rather than drastically because information of every individual gradually changes over time. To take it into account, we also propose a variant of our method to evolve the model parameters over time with long short-term memory. In our experiments on multiple datasets with dynamic interference, our methods outperform existing methods for ITE estimation because they are unable to capture dynamic interference. This result corroborates the importance of dynamic interference modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the problem of estimating treatment effects from observational graph data in dynamic environments. It proposes the DITE method and its variants to address challenges posed by dynamic interference. Experiments on multiple datasets validate the advantage of the proposed method over existing methods in ITE estimation.
The main contributions of this paper include:

Formalizing the problem of ITE estimation with dynamic interference for the first time, clarifying how individual interference in dynamic environments impacts treatment effect estimation, and providing a clear framework for future research.

Proposing a Dynamic Interference Modeling (DIM) layer that effectively captures dynamic interference through a Historical Aggregation mechanism (HA) and a Within-time Aggregation mechanism (WA). The HA mechanism utilizes a temporal attention mechanism (T-Attention) to estimate the importance of different timestamps and aggregates historical information with weighted importance. The WA mechanism aggregates information within each timestamp using a single-layer GNN. By stacking DIM layers, the model can capture multi-hop dynamic interference among neighbors.

Specifically, to address confounding factors, the approach reduces confounding bias by minimizing the Maximum Mean Discrepancy (MMD) between the covariate representation distributions of the control and treatment groups.

Additionally, the authors design an outcome prediction module based on interference representation, covariate representation, and a treatment training predictor, optimized with a specific loss function. They also propose a variant of DITE that leverages LSTM to evolve model weights, allowing adaptation to dynamic data changes.

### Strengths
1、This paper systematically investigates the problem of treatment effect estimation under dynamic interference for the first time, proposing the Dynamic Interference Modeling (DIM) layer to capture dynamic interference between individuals. This is a significant extension of traditional treatment effect estimation research. Notably, the introduction of a temporal attention mechanism (T-Attention) and stacking of single-layer GNNs to handle dynamically changing information offers significant innovation in addressing interference issues in dynamic graph data, providing new directions and insights for this field.

2、The use of Long Short-Term Memory (LSTM) networks to evolve model parameters is unique, enabling the model to better adapt to the dynamic changes in data over time. Compared to models with traditional static parameter settings, this approach performs better when handling data with temporal dynamics.

3、In the problem definition section, relevant concepts and assumptions, such as interference representation, the definition of ITE, and the identifiability proof of ITE under certain assumptions, are clearly articulated, ensuring a rigorous theoretical framework for the study.

4、The design logic of the proposed DITE method and its variants is clear, with each component (such as the DIM layer, confounding factor handling module, and outcome prediction module) having a distinct function and working collaboratively. This setup effectively addresses dynamic interference and estimates treatment effects. For instance, the HA and WA mechanisms within the DIM layer address the key challenges of dynamic interference modeling from the perspectives of historical information aggregation and within-time information aggregation, respectively.

5、In presenting technical methods, experimental setup, and result analysis, the paper uses concise language supported by numerous diagrams (such as architecture diagrams and experiment result graphs). For example, when explaining the working principle of the DIM layer, each step is clearly described through a combination of formulas and text.

### Weaknesses
1、To simplify the problem and focus on dynamic interference modeling, the paper adopts a strong no-confounding assumption (Assumption 3), which assumes that all confounders can be observed from covariates. However, in real-world applications, hidden confounders may not be accounted for, potentially leading to bias in ITE/ATE estimation. For example, in complex social or medical scenarios, there may be unobserved variables that simultaneously affect treatment assignment and outcomes.

2、Although the paper acknowledges that hidden confounders are an important issue, it does not delve into whether the proposed methods (e.g., the DIM layer) can effectively capture hidden confounders or address estimation bias that may arise from them. This represents a notable limitation in the study.

3、The paper does not thoroughly explore how to reasonably estimate the ITE at future timestamps in the absence of future information and does not propose a solution to address this scenario.

### Questions
1、The paper adopts a strong no-confounding assumption; however, hidden confounders may exist in practice, potentially affecting the accuracy of ITE/ATE estimation. Although the text mentions that existing work could leverage GNNs to recover hidden confounders from the graph and covariates, what potential does the DIM layer have for capturing hidden confounders? Are there plans to conduct experiments or theoretical analyses to evaluate the DIM layer’s effectiveness in handling hidden confounders? Adding experimental studies to address hidden confounders could enhance the evaluation, such as simulating data scenarios with hidden confounders, comparing DITE performance with and without consideration of hidden confounders, and analyzing the DIM layer’s sensitivity and handling capability for hidden confounders, thereby providing a more comprehensive assessment of the method's effectiveness in complex real-world situations.

2、The DITE method and its variants rely on future information to estimate ITE at future timestamps, which may not be available in real-world applications, thus limiting the practicality of the approach. What are the authors’ thoughts on estimating future timestamp ITE without future information? Are there plans to explore new methods or improvements to adapt to this scenario? Investigating how to use historical data and the current model state to predict future timestamp covariates, treatments, and graph structure could support future timestamp ITE estimation. For example, exploring time-series prediction techniques or generative models to create possible future input data, combined with the DITE method, may enable ITE estimation.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a method for estimating Individual Treatment Effects (ITE) in dynamic social networks, where user connections and influences evolve over time. The proposed Dynamic Interference Modeling (DIM) approach captures both within-time and cross-time influences. The architecture, DITE, combines graph neural networks and a time-weighting mechanism to aggregate historical interactions. A variant, DITEev, further adapts parameters over time using LSTM. Experimental results on synthetic and real-world datasets, such as Flickr and BlogCatalog, demonstrate that DITE and DITEev outperform conventional methods.

### Strengths
1. The paper studied an interesting and important problem by generalizing the ITE estimation with network interference from static to dynamic network.
2. The description of the proposed model is clear and the model structure of combining time-attention and GNN  is natural and well justified for the proposed problem. Moreover, the introduction of DITEev introduced a natural way to reduce the number of trainable parameters.
3. The authors carry out experiments on three synthetic dataset with comparison to several causal inference baselines.

### Weaknesses
1. Assumption 1 implies dependency on the treatment of other users at the same time stamp, but the proposed model does not incorporate this information as input, potentially limiting its alignment with the stated assumptions. Moreover, if the model follows the assumption with the treatment as input, it is not clear how the model can be used to in uplift decisions.
2. The model, even with help of DITEev, introduces a large number of trainbale parameters. The authors should clarify the size of the training dataset used and provide experimental results on how model accuracy depends on the amount of training data.
3. While the proposed algorithm is well-suited for the problem, its real-world applicability is limited. Accessing complete network dynamics and covariates for all individuals is often infeasible, and the algorithm’s runtime on large networks may also pose practical challenges.
4. The experiments are conducted on synthetic datasets, where the ITE aligns with the model assumptions. To validate the model's utility, evaluation on real-world data would be beneficial.
5. Although the authors use Maximum Mean Discrepancy (MMD) to address covariate balance, it remains unclear how the method significantly differs from a non-causal, purely supervised learning setup. Further clarification on this distinction is needed.

### Questions
Please see weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the estimation of individual treatment effects (ITE) in the presence of dynamic interference, where individual outcomes are influenced not only by current treatments but also by past interactions and treatments from previous neighbors within a dynamic network context. The authors propose a novel architecture that aggregates historical information from individuals and their neighbors using graph neural networks, incorporating a weighting mechanism to assess the importance of different timestamps. Additionally, they introduce a variant of their method that adapts model parameters over time using long short-term memory (LSTM) networks. Experiments on multiple datasets demonstrate the superiority of their approach in capturing dynamic interference compared to existing methods. The paper is well-written and thoroughly investigates the estimation of individual treatment effects (ITE) in dynamic networks. The authors empirically validate their model, emphasizing its significance in understanding the effects of social dynamics on treatment outcomes.

### Strengths
1. The paper fills a significant gap in the literature by formalizing the problem of ITE estimation with dynamic interference. The introduction of the DIM layer represents a novel contribution that distinguishes this study from existing methods.
2. The DITE framework is well-structured to address challenges related to historical information aggregation, within-time interference, and cross-time interference. This comprehensive approach is crucial for advancing the field of treatment effect estimation in dynamic contexts.
3. The authors provide extensive experimental results that demonstrate the superiority of their proposed methods over existing techniques. This empirical evidence enhances the credibility of their claims and underscores the practical relevance of their work.

### Weaknesses
1. The background and introduction sections do not sufficiently explain ITE, particularly in the context of dynamic networks. A more detailed introduction would help readers appreciate the research value in this area.
2. The overall framework diagram for the experiments lacks clarity. While the mechanism diagrams for DITE and DIM are present, a comprehensive experimental framework diagram is missing. This oversight makes it difficult to quickly grasp the main contributions and optimization mechanisms of the article. I suggest including a clearer framework diagram to facilitate understanding for readers.
3. In the results section, the main experimental results include only two comparison metrics. I recommend adding more widely used metrics to better demonstrate the effectiveness of the experimental results. Additionally, conducting further experiments to validate the model's performance—beyond simple comparisons of standard deviation and mean—would be beneficial. Presenting these comparisons in more intuitive charts would also help to better showcase the experimental outcomes.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work is designed to learn a dynamic interference modeling for estimating treatment effects. Genearlly speaking, the proposed problem is interesting. Prior research has focused on static interference in fixed networks, but real-world networks often change over time, leading to dynamic interference. To tackle the issues in this field, this study introduces a model for dynamic interference, aggregating both past and current information of individuals. Specifically, the proposed model uses a mechanism for summarizing historical data, graph neural networks for time-stamped interactions, and an attention-based weighting system to learn the contribution of each timestamp. Additionally, it adjusts parameters gradually to reflect evolving information dynamics, leveraging long short-term memory (LSTM) to update them. Experiments show that this approach outperforms existing models, highlighting the significance of accounting for dynamic interference in ITE estimation.

### Strengths
1. The authors evaluate various degraded version of the proposed model and tasks in real-world datasets to show the effectiveness of the main components, clearly detailing which components from previous research they incorporate and build upon.
2. The proposed problem is interesting. 
3. The paper is well-presented and easy to follow. 
4. The details of experimental settings are introduced such as the simulation methods.

### Weaknesses
1. The technical contribution of this work is limited. Several works focus on solving the dynamic interference in ITE estimation, so the proposed problem is not new. For example, "Treatment Effect Estimation Amidst Dynamic Network Interference in Online Gaming Experiments" also learns the dynamic interference. The model design seems combine many existing approaches. Stacking dnn layer to capture dynamic information and using attention mechanism to automatically learn the contribution of each timestamp are quite general methods.
2. More advanced gnn-based models should be considered such as lightgcn, gat. 
3. Some important references are missing such as Treatment Effect Estimation Amidst Dynamic Network Interference in Online Gaming Experiments.
4. In a closer look at dynamic interference part, a larger t_max should be considered. It can be better to describe the effect of the historical information. More historical information, better performance. Also, showing the trade-off between the running time and the value of t_max may make the results more convincing.

### Questions
1. It is better to introduce more technical contributions of the proposed model.
2. more gnn-based models should be compared.

### Soundness
3

### Presentation
3

### Contribution
3
