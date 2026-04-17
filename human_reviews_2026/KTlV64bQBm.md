# Flexible Participation for Differentially Private Synthetic Text Generation in Cross-Silo Federated Learning

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
In cross-silo federated learning (FL), sensitive text datasets remain confined to local organizations due to privacy regulations, making repeated training for each downstream task both communication-intensive and privacy-demanding. A promising alternative is to generate differentially private (DP) synthetic datasets that approximate the global distribution and can be reused across tasks. However, pretrained large language models (LLMs) often fail under domain shift, and federated finetuning is hindered by computational heterogeneity: only resource-rich clients can update the model, while weaker clients are excluded, amplifying data skew and the adverse effects of DP noise. We propose a flexible participation framework that adapts to client capacities. Strong clients perform DP federated finetuning, while weak clients contribute through a lightweight DP voting mechanism that refines synthetic text. To ensure the synthetic data mirrors the global dataset, we apply control codes (e.g., labels, topics, metadata) that represent each client’s data proportions and constrain voting to semantically coherent subsets. This two-phase approach requires only a single round of communication for weak clients and integrates contributions from all participants. Experiments show that our framework improves distribution alignment and downstream robustness under DP and heterogeneity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses generation of differentially private (DP) synthetic text in a cross-silo federated learning setting where clients have heterogeneous compute and data. The proposed two-stage framework lets strong (well-resourced) clients perform DP-SGD federated finetuning of a conditional generator (using control codes), while weak clients contribute via a lightweight DP voting / profiling mechanism that refines generated candidates without backpropagation. Control codes encode semantic partitions (labels/topics) and guide both generation proportions and localized voting to ensure semantically coherent refinement. Experiments on Yelp and PubMed (IID and non-IID partitions) show that (i) partial finetuning by a small fraction of clients improves over zero-shot generation, (ii) the DP voting refinement recovers much of the utility lost to DP noise and client heterogeneity, and (iii) gains hold across several downstream tasks and MAUVE / NER metrics.

### Strengths
- Clear problem motivation & practical relevance. Tackles a realistic cross-silo scenario: many organizations holding sizable private text, with widely varying compute budgets — a problem of immediate practical interest. 
- Simple, interpretable two-phase design. Splitting work between DP federated finetuning and DP voting is elegant: it lets expensive updates be restricted to capable nodes while still incorporating the remaining clients’ distributions. The use of control codes to structure generation is sensible and easy to implement. 
- Concrete algorithmic description and reproducibility effort. The paper includes pseudocode (Algorithm 1, A.2, A.3) and details on datasets, models, and hyperparameters (GPT-2, GPT-2-large, embedding model, MAUVE / downstream classifiers). Authors claim anonymized code in the supplement. These details aid reproducibility.
- Empirical evidence across IID and non-IID settings. The experiments test several participation rates (1–40% Cs), different privacy settings (ε=∞ vs ε=8), and show consistent improvements from the refinement stage across tasks and metrics (classification, MAUVE, NER F1). Tables and plots are informative.

### Weaknesses
- Baselines and ablations are limited. The baselines are (i) zero-shot pretrained generation and (ii) non-DP finetuning. Missing but important comparisons include: (a) PEFT / LoRA / adapter-style federated finetuning (compute-efficient finetuning that still requires backprop but is deployable), (b) other synthetic data generation / refinement approaches (e.g., preference-optimized or prompt-based DP synthesis), and (c) stronger ablations: effect of K (votes per example), sampling rate r in resampling, the sentence embedder choice, and synthetic dataset size s. These would better isolate where gains come from.
- Robustness & adversarial behavior not studied. The voting stage aggregates noisy votes from Cr clients. How robust is the procedure to malicious or biased voting (e.g., a client submitting anomalous profiles/votes)? Is there an attack model (and mitigation) — e.g., outlier detection, clipping of votes, or robust aggregation? Without such analysis, a deployment risk remains.
- Reliance on control codes and their privacy assumption. The method assumes control codes are public / non-private and that partitioning by code yields semantically coherent subsets. In practice, choosing/defining control codes can be nontrivial and may leak information if control codes correspond to sensitive labels. Please discuss sensitivity to mis-specified codes and privacy implications of broadcasting DP profiles over codes.

### Questions
- Baselines — why not PEFT/LoRA or prompt-based refinement baselines? Can you add comparisons to parameter-efficient federated finetuning (LoRA/adapter) or to refinement approaches that do not require voting (e.g., preference optimization, prompt-tuning with public seeds)?
- End-to-end privacy guarantee. What is the final (ϵ,δ) for the published synthetic dataset after composing DP-SGD training, profile perturbation, and vote perturbation? Show composition math or use advanced composition / moments accountant.
- Scaling to larger LLMs / real cross-silo deployments. Do you expect the refinement gains to persist for much larger generators (e.g., modern LLMs) or when evaluating tasks beyond classification/NER? Any deployment notes (latency, single-round comm overhead for Cr)?

### Soundness
3

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
3

### Summary
This paper focus on private synthetic data in FL. This research problem is meaningful when there are several downstream tasks in a federated learning setting. Authors propose a flexible participation framework that adapts to client capacities. Strong clients perform DP federated finetuning, while weak clients contribute through a lightweight DP voting mechanism that refines synthetic text. They apply control codes (e.g., labels, topics, metadata) that represent each client’s data proportions and constrain voting to semantically coherent subsets. Experiments show that the framework improves distribution alignment and downstream robustness under DP and heterogeneity.
I like the motivation of this paper, and I think this technology can be used more widely. It's just that the current experimental results and adaptation framework (synthetic data and federated learning) make me feel that they could be better.

### Strengths
This paper proposes a flexible participation framework for differentially private (DP) synthetic text generation in cross-silo federated learning (FL), addressing a key limitation of prior work — computational heterogeneity. 

The method elegantly combines DP federated finetuning on strong clients with DP voting-based refinement from weak clients, ensuring that all participants can contribute without heavy computation.

Experiments on Yelp and PubMed datasets under both IID and non-IID settings demonstrate strong results: the approach significantly improves synthetic data quality and downstream task performance while maintaining DP guarantees. The results are systematically presented and ablated, showing consistent gains from the refinement step.

This paper is well written.

### Weaknesses
1.	For experiments, the privacy parameter epsilon is set as 8. Why choose this number, 8 is quite large in DP.

2.	The computational cost of the voting step and privacy accounting (ε allocation across finetuning, profiling, and voting) could be more clearly analyzed.

3.	Evaluation primarily uses GPT-2 and GPT-2-large; extending to modern instruction-tuned or open-weight LLMs could strengthen claims of scalability and generality.

4.	Figure 1 is not easy to understand and needs to be explained in the legend.

5.	The entire method relies heavily on predefined control codes (such as tags, topics, and metadata).

### Questions
1.	How sensitive is performance to the choice of DP budgets among the three components ((ε_train, ε_prof, ε_vote))?
2.	How much communication cost is saved compared to full FL training when the number of weak clients is large?
3.	Are there plans to evaluate with larger models (e.g., Llama 3 or Gemma) to test scalability?
4.	In Table 1 and 2, epsilon=8 is following by a downward arrow, and epsilon=8 with refinement is following by a upward arrow, why?
5.	The ultimate goal of FL is "data remains stationary, model moves", to train a high-quality global model without concentrating on the original data. If the final output is synthetic data, then once these data are generated, we can directly use the synthetic data without federated learning. Then they will face the problem of re-centralization. Although this technically avoids the direct sharing of raw data, it seems to be a "step backward" in concept as it creates a new centralized dataset. Therefore, the application value of synthetic data in federated learning is questionable.

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
This paper proposes a two-phase framework for **Differentially Private (DP)** synthetic text generation in **cross-silo federated learning (FL)**, designed to handle **heterogeneous client resources**. 

The key idea is to allow **flexible participation**: 
- **Strong clients** (with sufficient compute) perform **DP-SGD finetuning** of a pretrained LLM to adapt it to domain-specific data. 
- **Weak clients** (unable to train locally) contribute via a **lightweight DP voting mechanism** that refines the synthetic data distribution, ensuring representation of all clients. 

Control codes (labels, topics, metadata) guide both finetuning and refinement, partitioning data into coherent subsets. 

Experiments on **Yelp Reviews** and **PubMed abstracts** demonstrate that this approach improves **distributional alignment and downstream performance** of generated synthetic data, particularly under DP constraints and heterogeneous participation. Refinement with weak-client voting mitigates the utility loss typically induced by DP noise and biased finetuning.

### Strengths
- **Novel integration of heterogeneous participation and DP**: The flexible two-phase structure is a natural and elegant way to involve all clients under compute constraints. 
- **Sound motivation**: Addresses a practical gap between FL for large models and realistic cross-silo deployment, where client capacity varies widely. 
- **Technical clarity**: The description of Algorithm 1 and control-code-based conditioning is detailed and well-structured. 
- **Empirical evaluation**: 
  - Compares baselines with and without DP, and with/without refinement. 
  - Uses multiple datasets and metrics (classification accuracy/F1, MAUVE, NER). 
  - Results consistently show refinement improves performance under DP, especially with few strong clients.

### Weaknesses
1. **Privacy accounting and budgets** 
   - The paper applies separate ε = 8 budgets for training and refinement, but it’s unclear whether these compose into a total DP guarantee or are treated independently. 
   - The choice ε = 8 is relatively high; discussion of lower-ε performance or practical implications would strengthen credibility. 
   - Clarify the full \((\varepsilon_\text{total}, \delta_\text{total})\) budget.

2. **Evaluation of privacy–utility trade-off** 
   - All experiments use ε = 8; it would be valuable to show at least one lower ε (e.g., 4 or 2) to demonstrate robustness to stricter privacy. 
   - Plotting performance as a function of ε would better illustrate the trade-off curve.

3. **Refinement mechanism interpretability** 
   - The “DP voting” phase uses noisy aggregated similarity scores. It would help to explain how KNN-based votes interact with control codes and whether Gaussian noise biases toward majority groups. 
   - Are weak clients’ votes weighted equally regardless of dataset size?

4. **Baselines and ablations** 
   - The “voting” mechanism could be compared against a simpler aggregation (e.g., uniform resampling or non-DP voting) to isolate its effect. 
   - Clarify whether “pretrained + voting” (without any finetuning) was tested.

5. **Conceptual framing** 
   - While the “control code” abstraction is central, it’s borrowed from prior controllable generation work; the novelty lies in its federated adaptation. 
   - The term *Flexible Participation* might overstate generality: the method still assumes clients can be cleanly partitioned into strong/weak and have known control-code profiles.

6. **Relation to prior work** Could better contrast with recent LoRA-based federated adaptation (e.g., FLoRA 2024) and DP synthetic text methods that do global aggregation rather than federated (e.g. private evolution (voting), DP fine tuning with LoRA, PATE based methods)

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a flexible participation framework in the cross silo setting, where some clients may be imbalanced in terms of resources. In particular, some clients may have enough infrastructure and GPUs to run DPSGD on their own data, and some clients may not have enough compute power. So the method proposes to do local DPSGD on clients with compute budget, and use a method similar to Private Evolution to condition synthetic data generations towards the data on the clients which do not have much compute budget. In this way they are able to balance the benefits of training while also leveraging the data of clients that cannot train. They support their algorithm with a set of experimental evaluations on the GPT2 model.

### Strengths
- The algorithm does make sense. While it is not the most elegant thing, it resembles production-ready algorithms in industry which may be a combination of several different algorithms to handle different regimes (the algorithm proposed is essentially FedAvg + Private Evolution).
- The participation model helps include a wide variety of clients, which allows synthetic data to represent a wide diversity of data
- Experimental ablations are decent, with evaluations across different epsilon values, datasets, and strong client participation. They demonstrate that the refinement step is important in getting good performance

### Weaknesses
- Could be more explicit about the privacy composition, there are multiple steps and it is unclear how they are composed
- The algorithm is mostly a simple combination of two existing ones (FedAvg, Private Evolution), there isn't a major methodological breakthrough. However this is not something I would hold against the paper.
- The evaluation could be more comprehensive. For example, they could evaluate more models outside of GPT2. Second, they should compare against prior work better. The real baseline to compare against is not just the zero-shot pretrained model. It also includes the other methods mentioned in section 5, such as Private Evolution and its variants. The papers in the related work have their own evaluations against standard datasets, so the paper should run its method on those datasets, get the numbers, and compare against the results reported in those papers. The other baseline to compare against is pure FedAvg, which should be better than the proposed method but again this evaluation is needed for comparison.

Overall I think this is a promising paper, but feels somewhat incomplete at the moment. For it to be published at a venue like ICLR, I would want to see solid performance gains vs Private Evolution and its follow ups.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
