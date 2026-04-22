# G-Merging: Graph Models Merging for Parameter-Efficient Multi-Task Knowledge Consolidation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The pretrain-finetuning paradigm has achieved notable success in graph learning. Moreover, merging models fine-tuned on different tasks to enable a parameter-efficient model with multi-task capabilities is gaining increasing attention for its practicality. However, existing model merging methods, such as weight averaging and task arithmetic, struggle to generalize well to graph structures and Graph Neural Network (GNN) models due to the unique structural heterogeneity of graph data. In this paper, we propose an innovative graph model merging framework called G-Merging for merging multiple task-specific fine-tuned GNN models. G-Merging first employs task arithmetic to coarsely merge graph models, capturing shared cross-task knowledge. Second, it introduces a Topology-aware Wasserstein Distance (TWD) loss to train lightweight task adapters, preserving domain-specific graph patterns via aligning the embeddings of merged and fine-tuned models. Third, G-Merging integrates the adapters into a training-free, topology-aware router within a mixture-of-experts (MoE) architecture, dynamically routing input graphs to task-specific adapters based on structural similarity, thereby mitigating conflicts and enhancing knowledge sharing. Extensive experiments on 8 graph downstream datasets demonstrate the effectiveness of G-Merging, showing impressive performance close to or exceeding individual finetuned models while improving parameters and training efficiency. Our code is available at https://github.com/cjcj46262/G-Merging.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces G-Merging, a novel framework for consolidating multiple task-specific fine-tuned Graph Neural Network (GNN) models into a unified multi-task model with parameter efficiency. The proposed approach consists of three key components: (1) Task Arithmetic for coarse parameter merging, (2) Topology-aware Wasserstein Distance (TWD) for aligning embeddings and training lightweight task-specific adapters, and (3) a training-free MoE router that dynamically selects task-specific adapters based on structural similarity. The authors position this work as the first systematic approach to address the model merging problem specifically for graph neural networks.

### Strengths
1. This is the first work to systematically address the model merging problem specifically for graph neural networks. The proposed three-stage framework (Task Arithmetic, TWD-based alignment, MoE router) is logically sound with each component complementing the others.
2. By only fine-tuning a small number of adapter parameters rather than the entire model, G-Merging significantly reduces computational and storage costs for multi-task deployment, aligning with current trends in parameter-efficient fine-tuning.
3. The evaluation across multiple molecular property prediction tasks demonstrates G-Merging's effectiveness in consolidating knowledge from multiple task-specific models. Ablation studies validate the contribution of each component.
4. The paper is well-structured with clear methodology descriptions and intuitive visualizations that make complex technical content accessible.

### Weaknesses
1. The current approach assumes all tasks share the same output dimension, preventing it from handling tasks with different output dimensions (e.g., binary vs. multi-class classification). This represents a significant limitation since real-world applications often involve diverse task types with varying output structures.
2. The method involves multiple hyperparameters (task vector scaling factor, loss balancing weights, adapter rank, etc.). While the paper provides some sensitivity analysis, there's insufficient guidance on how to select these hyperparameters based on task characteristics.
3. Current experiments focus exclusively on molecular property prediction tasks (all graph-level binary classification), lacking validation on other graph task types (e.g., node classification, link prediction).
4. The paper primarily compares against model merging approaches but lacks comprehensive comparison with parameter-efficient fine-tuning (PEFT) methods for multi-task learning (e.g., LoRA-based approaches). 
5. While the topology-aware Wasserstein distance is proposed, there's minimal theoretical analysis of its properties, relationship to graph structural characteristics, or convergence guarantees. Additional theoretical insights would strengthen the methodological contribution.

### Questions
1. The paper assumes all tasks use the same pre-trained GNN backbone. How would G-Merging handle cases where different tasks employ different GNN architectures? If not currently supported, what modifications would be needed to extend the framework?
2. In the MoE router, how is "structural similarity" formally defined and computed? Is there theoretical justification for this similarity metric's effectiveness? Were different similarity measures experimentally compared?
3. How does G-Merging perform when merging models trained on graph datasets with significantly different structural distributions (e.g., social networks vs. molecular graphs)? Are there specific mechanisms to handle such task heterogeneity?
4. The paper mentions that TWD leverages graph topological information, but the ablation study only shows results with different adjacency matrix powers. Could you provide more detailed analysis on how different topological aspects (e.g., degree distribution, community structure) affect the merging performance?

If the authors provide satisfactory responses to these concerns during the rebuttal phase, I would be prepared to revise my recommendation to a stronger accept.

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
4

### Summary
The paper introduces a strategy for model merging in graph neural networks (GNNs). In particular, the considered setting includes a pre-trained GNN, trained on some general task, which then gets adapted into multiple specialized versions through fine-tuning for specific tasks. The goal is then to merge the fine-tuned versions together to obtain a single model that can perform all tasks with comparable performance. The proposed method involves first merging parameters with task vectors, then adding adapter modules in between graph convolutions (and training them), and finally a training-free mixture of expert approach to mix the contributions from different adapters at inference time. A novel loss based on Wasserstein distance is used to train the adapters with the goal of reducing the difference in the distribution of the representations between the unified and fine-tuned models.
The experiments consider 8 binary graph classification datasets regarding molecule property prediction, and compare against existing model averaging techniques (Weight Averaging, Task Arithmetic, Ties-Merging, and EMR-Merging, AdaMerging, Twin-merging). Additional baselines are given by multi-tasks learning, individual fine-tuning, and the pre-trained model. Results show that the merged model has similar performance to the individual fine-tuned ones.
An ablation study shows the importance of all components in the proposed method.

### Strengths
- Extensive experimental section with many baselines.
- Informative ablation experiments to showcase the inner workings of the method (router heatmaps and top-k expert selection plots)

### Weaknesses
- The graphs considered in the experimental evaluation mostly have a very small number of nodes. 
- The proposed Wasserstein loss can be computationally prohibitive for large graphs.

### Questions
-  Oversmoothing is a well known problem of GNNs. The Wasserstein loss introduced in this paper seems to actually encourage oversmoothing (it would be minimized if all representations were the same). Could the authors please comment on this?
- The proposed Wasserstein loss has a complexity which is quadratic in the number of nodes (as shown in the appendix). How does this behave for large graphs? Would it be possible to add some runtime values for some of the experiments?
- One of the properties of the proposed Wasserstein loss is that it is "aware of graph topology" (i.e., constrained by the graph adjacency matrix). Have the authors tried removing this constraint and seeing how performance would change? I agree that this seems like a principled choice, but at the same time giving more freedom to the transport plan may provide benefits?
- Could the authors expand on the practicality of the proposed method? It requires a pre-trained model, and several fine-tuned versions of it. While this is a somehow realistic scenario for large language models, is this something that happens when deploying GNNs in real-world applications?
- If possible, as the experiments have been run over 5 random seeds, could you add numbers for the variance/standard deviation? I think this is an important factor to validate the effectiveness of the method.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work primarily proposes a graph model fusion framework that first performs coarse fusion of graph models using task-specific operations, and then trains task adapters via TWD loss, leveraging an MoE-based routing mechanism to reduce conflicts after merging and enhance knowledge sharing.

### Strengths
1. Originality.The authors' work is original.

2. Quality. The quality of the authors' work is excellent; the assumptions and experiments mutually support each other.

3. Clarity. The authors' work is clear. In particular, the figures fully explain what they are doing.

4. Significance. Given the well-known characteristics of graph data, obtaining a unified cross-modal model in the graph domain is an important problem.

### Weaknesses
1. Data Selection: The data selection in this work is confusing to me—why do the experiments only use graph-level datasets and not discuss more widely adopted node-level benchmarks (e.g., Cora)?

2. Lack of Discussion on Related Works: The paper lacks sufficient discussion of related work, which makes it hard to follow. To my understanding, the authors’ method can be summarized as parameter merging combined with knowledge distillation (please correct me if I’m wrong). Even within the graph learning community, this type of approach has been extensively explored—for instance, ParetoGNN [1] (during training) and WAS [2] (during fine-tuning). Could you please clarify how your design differs from or builds upon these prior works?

---

[1] ICLR '23, https://arxiv.org/abs/2210.02016  
[2] ICLR '24, https://arxiv.org/abs/2403.01400

### Questions
please refer to weaknesses.

Overall, I think this is a good paper, but there are some unclear aspects that prevent me from fully evaluating its contribution. If the authors could clarify my concerns, I would be happy to raise my score.

### Soundness
3

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
The paper introduces G-Merging, a three-phase framework to merge multiple task-specific GNNs into a single multi-task model. Phase I performs task arithmetic to coarsely combine fine-tuned models into a unified encoder. Phase II trains lightweight adapters (node- and graph-level) on each task to align the unified model’s representations to those of the original fine-tuned models via a Topology-aware Wasserstein Distance (TWD) plus an L1 graph-level loss. Phase III composes all task adapters into a training-free MoE with a topology-aware router that computes per-input soft weights from representation similarities, enabling test-time expert selection. Across eight molecular property prediction datasets with GIN/GCN backbones and two pretraining strategies, G-Merging matches or exceeds many weight-merging baselines and approaches per-task fine-tuned upper bounds while using far fewer trainable parameters during adapter training.

### Strengths
1. The TWD loss constrains optimal transport by the input graph’s adjacency, encouraging local smoothness in node embeddings and addressing structural heterogeneity that undermines naïve merging. This is a principled graph-specific twist on WD and is well-motivated by prior observations of task/domain clustering.

2. The three phases, task arithmetic (shared knowledge), adapter training (exclusive knowledge), and train-free MoE routing, are clearly specified with equations and an algorithmic sketch, making the method easy to reimplement or ablate.

3. While being, to my knowledge, the first study of model merging in the graph domain is noteworthy, the paper should better justify why such a method is necessary. Given how long model merging has been around, its absence in the graph domain likely reflects factors beyond the task difficulty, for example a lack of clear necessity.

### Weaknesses
1. This method still appears computationally intensive, because a task-specific adapter must be trained for each model involved in the merging. If the computational cost of this training is comparable to, or not much lower than, training a GNN from scratch, it seems to run counter to the original purpose of merging. This concern rests on two points: first, GNN training is itself quite lightweight; second, merging is intended to reduce the training burden when facing new tasks. In addition, how transferable are the trained adapters? Do we need to retrain a new set of adapters for every configuration (number/combination of models)?

2. Compared with the baselines, this method appears to require more GPU memory at inference. The baselines’ memory footprint is essentially that of a single model, whereas this approach involves an MoE component, which requires a router to decide whether and how each adapter is applied during inference. Moreover, the results in Tables 1 and 2 suggest that introducing the MoE architecture has little to no positive effect on overall performance and seems largely dispensable.

3. Figure 3(b) is only shown in the main text but is barely discussed in the experimental description. The paper mentions on lines 473–473 that “as the set of selected experts becomes larger, indicating access to more diverse knowledge, the model performance improves accordingly.” However, this statement does not match the results in Figure 3(b): as the set of selected experts grows, performance first drops and then rises, rather than monotonically increasing. In the appendix, Figure 7 presents more fine-grained results but still lacks detailed analysis. I recommend that the authors provide a thorough analysis of this experiment. Based on Figure 7, the Top-K strategy appears to have markedly different effects across datasets. How do the authors balance these differences in practice, and what explains these divergent impacts?

4. It seems that all tasks are molecular property prediction. It is unclear whether TWD and routing generalize to non-molecule domains (social, knowledge, large sparse graphs). Please add some non-molecular datasets or discuss constraints.

### Questions
See weakness. I'll adjust my rating according to the authors' rebuttal.

### Soundness
2

### Presentation
3

### Contribution
2
