# Understanding Medical Time Series Event Piece by Piece: A Fine-Grained Event Detection Network

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Event detection in medical time series is fundamental to supporting health monitoring and clinical decision-making. However, most existing methods divide time series into fixed-length segments and perform coarse-grained, segment-level detection, which fails to precisely localize the start and end times of events. This limitation can mislead clinical assessment and obscure the true severity of conditions. To address this, we propose EventCompreNet——a universal network for fine-grained event detection leveraging auxiliary tasks. Inspired by the cognitive processes that human detect events, we introduce four human comprehension tasks to enhance the model’s understanding of each piece of events. Moreover, to overcome the limited knowledge transfer in existing multi-task learning structures, we develop a task-deep-coupling framework that facilitates deeper interaction among tasks. Through these designs, EventCompreNet achieves a comprehensive understanding of the entire event life cycle. Experimental results on four benchmark datasets demonstrate that our model significantly outperforms existing state-of-the-art time series models in fine-grained event detection and exhibits strong event comprehension capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes EventCompareNet, a convolutional network for identifying events in medical time series.

### Strengths
- The extensive hyperparameter tuning of the baselines is well done
- Good selection of baselines
- Cross validation throughout the entire experimental setup helps resolve some of the concerns about the tiny dataset size

### Weaknesses
- If I understand the experimental setup correctly, data from a single patient might show up in both the training and test set? If so, this is a severe soundness violation.
- Reporting only F1 score is problematic. It's best to do evaluations on a couple of metrics. One particularly important missing metric is AUROC, which is threshold independent.
- The sizes of the datasets used are relatively small.
- The novelty of the task decoupling network is unclear. Task embeddings have been explored many times in prior work. See https://aclanthology.org/C18-1251.pdf for a good example
- This is a relatively minor point, but to me it's very unclear why the additional "human comprehension tasks" would help, since they are all a subset of the fine-grained event detection task.

### Questions
See above

### Soundness
1

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
The paper proposes EventCompreNet, a multi-task learning model for fine-grained event detection in medical time series. The authors evaluate the method on four public datasets (FOG, DREAMS, SHHS, QT) and report state-of-the-art results compared with both task-specific and general-purpose time-series models.

### Strengths
1. The proposed TDC framework elegantly addresses a real issue in multi-task setups: limited information sharing and redundant parameters. The use of a shared decoder with task embeddings is clean and efficient.
2. The ablation study nicely isolates the effects of HC tasks and the TDC framework. Visualizations help interpret what each auxiliary task contributes.

### Weaknesses
1. The core idea—sharing parameters across multiple auxiliary tasks with task identifiers—has been explored in multi-task learning for years. The paper does not clearly explain what is fundamentally new beyond reusing an existing framework under a new name. To make the contribution more convincing, the authors could formalize the “task-deep-coupling” mechanism, quantify the degree of knowledge transfer, or demonstrate a measurable improvement in training dynamics over conventional shared-encoder designs.
2. The proposed auxiliary tasks are described as mimicking human understanding of events, but this analogy remains superficial. There is no cognitive evidence or ablation showing that these particular four tasks are necessary or complementary. The paper would be stronger if it replaced this speculative framing with a more principled rationale—e.g., decomposing event detection into sub-tasks based on temporal reasoning or uncertainty estimation rather than on loosely defined human cognition.
3. The “class-balanced filtering” step appears to use label information to preselect sequences containing events, which risks data leakage. In addition, some datasets are extremely small, and there is no mention of subject-wise splits or statistical significance testing

### Questions
1. Could the authors clarify how the proposed “task-as-sample” training scheme differs from standard multi-task batching? Does it actually expand the dataset size or simply replicate samples with different task IDs? How does it affect convergence speed and sample balance?
2. In the “class-balanced filtering” step, how does the model decide which sequences contain events without using ground-truth labels? If this decision relies on supervision, wouldn’t it lead to leakage during training or biased evaluation?

### Soundness
2

### Presentation
3

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
This paper addresses the detection of medical events from time series data, such as EEG data, where events are detected in a fine-grained manner, i.e., at the time point level. The authors propose training the main fine-grained detection model, accompanied by multi-perspective human comprehension task models, to improve detection performance. The multi-perspective human comprehension tasks consist of the center localization, boundary localization, lifetime analysis, and coarse-grained event perception tasks, where each task boosts the main fine-grained detection model, and the coarse-grained event perception acts as a class-balanced filter. They utilize a single model, EventCompreNet, which contains a task-deep-coupling framework, for all the above tasks, where the output is switched based on the task ID embedding. The proposed method was evaluated on four fine-grained event detection datasets and consistently outperformed baselines.

### Strengths
-- The multi-perspective human comprehension tasks and task-deep-coupling framework are intuitively reasonable to improve the fine-grained detection model. It may be related to curricuram learning, such as https://arxiv.org/abs/2212.03597 and https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.sciencedirect.com/science/article/pii/S0020025523003183&ved=2ahUKEwi28eKdgM6QAxV2mq8BHcZrHcYQFnoECAsQAQ&usg=AOvVaw2WbyBn9PZOoAGHBkjx9kQd.

-- Experimental results on multiple datasets demonstrated the effectiveness of the proposed method.

### Weaknesses
-- The proposed method may have novel points and have some practical impact, but the clarity issues are severe in understanding the method:
* The important part of the method description, "Task Synchronous Training," is hard to follow and misses a lot of details. How can we handle the differences in output variable types, including binary, multi-class, and real-valued, without modifying the network architecture, using only the task ID embeddings?
* Section "4.3 CLASS-BALANCED FILTERING" requires more details. If we naively do "The sample size for each task is repeatedly sampled to align with the largest sample size task", all the samples are classified as "background" to minimize loss.
* The descriptions for "Center Localization Task" and "Boundary Localization Task" from l.197 are hard to follow. The ideas are simple, so I can understand what is conducted in the tasks briefly. However, it is difficult to understand precisely what the labels are for the tasks.
* Eq.10 and the following description would contain a typo or be confusing.
* Eq.11 can be improved by making the second low "larger than zero" for generalizing the number of classes is larger than 1.
* In. l.126 and l.298, k is used in different meanings.
* It is better to put the tables and figures on top.

### Questions
-- The authors mentioned "it is the first universal network in this research area". Are there any related works in other domains?

-- Curriculum learning can be related, such as https://arxiv.org/abs/2212.03597 and https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.sciencedirect.com/science/article/pii/S0020025523003183&ved=2ahUKEwi28eKdgM6QAxV2mq8BHcZrHcYQFnoECAsQAQ&usg=AOvVaw2WbyBn9PZOoAGHBkjx9kQd. Is it possible to discuss with the literature?

### Soundness
3

### Presentation
2

### Contribution
3
