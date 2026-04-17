# GMTRouter: Personalized LLM Router over Multi-turn User Interactions

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Large Language Model (LLM) routing has demonstrated strong capability in balancing response quality with computational cost. As users exhibit diverse preferences, personalization has attracted increasing attention in LLM routing, since even identical queries may require different models to generate responses tailored to individual needs. However, existing approaches are not fully personalized and often fail to faithfully capture the complex interactions between specific users and LLMs. Moreover, user preference data is typically scarce, noisy, and inconsistent in format, which limits the effectiveness of methods that rely solely on user-specific data. To address these challenges, we propose GMTRouter, which represents multi-turn user–LLM interactions as a heterogeneous graph with four node types: user, LLM, query, and response, thereby maximally preserving the rich relational structure of the interaction. Through a tailored message-passing mechanism, GMTRouter learns to capture user preferences from few-shot data within a lightweight inductive graph learning framework, enabling effective personalization. Extensive experiments demonstrate that GMTRouter consistently outperforms the strongest baselines, achieving 0.9%–21.6% higher accuracy and 0.006–0.309 higher AUC across multiple datasets. More importantly, we further demonstrate that GMTRouter can adapt to new users and evolving preferences using only few-shot data, without extensive fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces GMTRouter, a personalized LLM router for LLMs that extracts structured preference information from multi-turn user-LLM interactions. The methodology is to model user-LLM interactions as a heterogeneous graph with different node types (users, queries, responses, LLMs, and virtual "turn" nodes) to capture the relational structure among entities. Then, a heterogeneous graph transformer (HGT) is used to predict which LLM best suits a given user-query pair. Experiments on Chatbot Arena, MT-Bench, GSM8K, and MMLU show consistent improvements over baselines such as GraphRouter and FrugalGPT. The authors also show that GMTRouter generalizes well to new users and remains computationally efficient.

### Strengths
1. Modeling multi-turn user-LLM interactions as a heterogeneous graph with explicit node types and virtual "turn" nodes is a novel approach to personalized routing.
2. The inductive training strategy allows adaptation to new users with minimal interaction data, addressing the common cold-start problem.
3. The framework is lightweight and practical, requiring modest computational resources, which enhances its potential for real-world deployment in LLM routing systems.

### Weaknesses
1. The design of GMTRouter is largely empirical and heuristic. There is no theoretical justification for why the proposed graph structure and the HGT-based model can better extract and generalize user preferences compared to other models.
2. The main novelty lies in the data modeling design (heterogeneous graph construction), while the learning components (e.g., HGT backbone, inductive training) largely follow existing graph learning literature.
3. There are many duplicated references, e.g., (Chen et al., 2023a) and (Chen et al., 2023b), (Chen et al., 2023a) and (Chen et al., 2023b), (Feng et al., 2024a) and (Feng et al., 2024b), (Hu et al., 2020a) and (Hu et al., 2020b), (Ong et al., 2024a) and (Ong et al., 2024b).

Minor issues:
1. The abbreviation "AUC" is not defined in the paper.

### Questions
1. As the heterogeneous graph is constructed directly from the interaction history, it should not contain more information than the history itself. Could the authors elaborate on why such a graph-based representation improves preference elicitation compared to directly learning from the raw interaction table?
2. The graph construction includes multiple users within the same graph (e.g., Figure 3(b)). However, user preferences are typically independent of one another. Would constructing individual user-specific graphs (and learning over them) be more intuitive? How does the shared multi-user graph influence personalization performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GMTRouter, a personalized LLM routing framework that models multi-turn user–LLM interactions as a heterogeneous graph. By utilizing a lightweight inductive graph learning framework, GMTRouter captures user preferences from few-shot data and adapts to new users.

### Strengths
1. The proposed GMTRouter models multi-turn user–LLM interactions using a heterogeneous graph, capturing complex relational dependencies for personalization.
2. The inductive graph learning framework allows GMTRouter to adapt to new users with minimal data.
3. Experimental results show that GMTRouter consistently outperforms baselines across multiple datasets.

### Weaknesses
1. The paper heavily relies on LLM routing for personalization, but this essentially requires the model itself to have inherent personalization capabilities, making the routing process somewhat secondary.
2. Memory construction is a promising approach for personalization, but the paper does not provide a comparison with existing memory-based methods. 
3. The graph construction process is time-intensive; however, the paper lacks a detailed analysis of the cost involved in updating the graph post-construction.

### Questions
The related work section does not sufficiently cover the wide range of recent LLM personalization efforts. More attention should be given to the existing literature on personalized LLMs. 
The paper does not provide a comparison with existing personalized LLM methods.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
GMTRouter tackles personalized LLM routing by modeling multi-turn user–LLM interactions as a heterogeneous graph whose nodes are users, LLMs, queries, and responses, with user feedback embedded as a preference feature. A lightweight inductive Heterogeneous Graph Transformer aggregates information over sampled conversation “turn” nodes, and a cross-attention prediction head ranks candidate LLMs for a new (user, query) pair using only a few past interactions. GMTRouter consistently outperforms baselines and generalizes well to unseen users with few-shot data. The model is compact and trainable on a single GPU.

### Strengths
1. The paper tests on one real-world and three synthetic benchmarks, builds multi-user labels that combine quality, cost, length, and rare-word signals, reports clear metrics, and assess generalization.
2. Practical, resource-efficient design suitable for deployment. GMTRouter is small, with modest storage and max GPU usage (~4.3 GB), and the experiments run on a single RTX A6000.

### Weaknesses
1. Missing baselines. It seems that the paper is missing some baselines achieving the personalization through graph. For example, Knowledge Graph Tuning: Real-time Large Language Model Personalization based on Human Feedback. 
2. This paper is claiming it is a LLM routing based model, however, it does not compare to baselines of routed LLMs. For example, RouteLLM: Learning to Route LLMs with Preference Data.
3. What is the key challenge that this paper is trying to solve, all the modules seems pretty standard to the reviewer to form a GNN. 
4. The writing is a little bit confusing. All previous parts are emphasizing the routing and selection, but the majority of the methods is on constructing the GNN, until the very last of the section, the reviewer can see how you really make the selection.

### Questions
See weaknesses.

### Soundness
4

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
This paper aims to tackle the task of personalized LLM routing: aligning model selection with individual user preferences based on their multi-turn interaction histories (to select the best LLM for the new query), which differs from existing works that are not personalized and often ignore multi-turn conversations. To tackle this, the authors proposed the GMTRouter, which represents users, queries, responses, and LLMs as nodes in a heterogeneous graph, allowing it to capture the relational structure across them. After that, through the message passing over the constructed graph, it enables predicting the suitable LLM for the new query. The authors validate the proposed approach on one realistic and three synthetic datasets, showing that it outperforms strong router baselines.

### Strengths
* The construction of the heterogeneous graph with four types of nodes (users, queries, responses, and LLMs) for personalized LLM routing is convincing.
* The proposed approach outperforms existing router baselines, even under the challenging scenarios (e.g., handling new users or with few data samples). 
* The proposed router design is very efficient, requiring only minimal computing.

### Weaknesses
* The way that the authors construct the synthetic datasets for benchmarking and their quality is questionable. The datasets, such as MT-Bench, GSM8K, and MMLU, are not designed for personalization, and some of them are also not for multi-turn scenarios. How do the authors convert them to the multi-turn personalization settings? Additionally, how do you define users in those settings? More clarifications on them are needed. 
* On the other hand, the scale of the realistic dataset (Chatbot Arena) is small, consisting of only 11 users. It would be great if the authors could scale this up. 
* The novelty of the proposed method over the existing GraphRouter seems marginal, as the GraphRouter similarly constructs the heterogeneous graph and propagates messages between nodes to predict the LLM. It would be great if the authors could clarify whether the proposed approach is the extension of the GraphRouter, additionally considering the multi-turn interactions and users for personalization, or if there are any other aspects that the authors would like to emphasize. 
* While the authors explicitly mention that the preference data by the single user is scarce, noisy, and inconsistent in format, it is questionable whether the authors validate the proposed approach on those challenging settings, and more importantly, whether the proposed approach has a certain design choice to tackle those challenges. 
* It seems the authors reduce the margin for figures and tables at the top a lot, which should be fixed.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
