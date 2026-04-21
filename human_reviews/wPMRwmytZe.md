# Progressive distillation induces an implicit curriculum

- Avg Score: 7.60
- Decision: Accept (Oral)
- Scores: 8, 8, 8, 6, 8

## Abstract
Knowledge distillation leverages a teacher model to improve the training of a student model. A persistent challenge is that a better teacher does not always yield a better student, to which a common mitigation is to use additional supervision from several “intermediate” teachers. One empirically validated variant of this principle is progressive distillation, where the student learns from successive intermediate checkpoints of the teacher. Using sparse parity as a sandbox, we identify an implicit curriculum as one mechanism through which progressive distillation accelerates the student’s learning. This curriculum is available only through the intermediate checkpoints but not the final converged one, and imparts both empirical acceleration and a provable sample complexity benefit to the student. We then extend our investigation to Transformers trained on probabilistic context-free grammars (PCFGs) and real-world pre-training datasets (Wikipedia and Books). Through probing the teacher model, we identify an analogous implicit curriculum where the model progressively learns features that capture longer context. Our theoretical and empirical findings on sparse parity, complemented by empirical observations on more complex tasks, highlight the benefit of progressive distillation via implicit curriculum across setups.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Traditional knowledge distillation relies on a single, powerful teacher model to train a smaller student model. However, it has been observed that stronger teachers do not always yield better-performing students. To address this issue, the authors explore progressive distillation, where the student model is incrementally guided by a sequence of increasingly capable teacher checkpoints. Through theoretical analysis and extensive experiments on tasks such as sparse parity and probabilistic context-free grammars (PCFGs), the paper demonstrates that progressive distillation accelerates the optimization process, achieving improved performance with fewer training steps compared to one-shot distillation and direct learning from data.

### Strengths
Strengths
* **Intuitive Motivation and Theoretical Foundation:** The paper is well-motivated, addressing a significant challenge in the effective distillation from large to small models. The theoretical underpinnings are well grounded, with rigorous proofs using the sparse parity example demonstrating why progressive distillation is effective.
* **Empirical Validation Across Diverse Tasks:** The authors conduct comprehensive experiments on both synthetic tasks (sparse parity and PCFGs) and realistic settings (training BERT on Wikipedia and Books datasets). This breadth of evaluation underscores the generalizability of their findings.

### Weaknesses
**Weaknesses**

- **Scalability to Larger Models and Diverse Architectures**: While the experiments include models like BERT, the paper does not thoroughly investigate how progressive distillation scales with increasingly large models (in the main paper).

- **Effectiveness in GPT Models**: In Appendix E, the paper examines autoregressive training with GPT-2. Although progressive distillation accelerates learning speed, the performance plateaus around 8,000 training steps. It remains unclear why this convergence occurs and what the outcomes would be if training continued beyond 8,000 steps (e.g., up to 20,000 steps).

- **Potential for Degenerate Features**: While progressive distillation leverages an implicit curriculum to identify key patterns, there is a concern that this curriculum might lead to degenerate features that could hinder long-term generalization. This issue could be further investigated by extending the training duration in GPT-2 to ensure that no negative consequences arise from prolonged training.

### Questions
* **Teacher Checkpoint Selection Across Domains**: What guidelines or heuristics can assist in selecting intermediate teacher checkpoints effectively for diverse tasks such as visual classification, image generation, or reinforcement learning?
* **Joint Training of Student Models:** What are the effects of training small student models jointly with large teacher models, instead of using progressive distillation with intermediate checkpoints? Does this approach provide a similar implicit curriculum that benefits the student’s learning?

I am willing to improve my score if the authors demonstrate that progressive distillation remains beneficial compared to one-shot distillation in the long run for large models, such as GPT-2.

### Soundness
4

### Presentation
4

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
The paper investigates how progressive distillation can accelerate the training of neural networks by leveraging an implicit curriculum provided in the form of intermediate teacher checkpoints. The authors conduct experiments on tasks like sparse parity, probabilistic context-free grammars (PCFGs), and real-world datasets ("Wikipedia and Books") using models like MLPs, Transformers, and BERT. The main claimed findings are:

1. Progressive distillation accelerates student learning compared to one-shot distillation.
2. An implicit curriculum emerges from intermediate teacher checkpoints, guiding the student through increasingly complex subtasks.
3. This implicit curriculum results in an observable empirical acceleration, and stems from provable benefits.

### Strengths
I liked the spirit of this paper, which I found complete, well-written, carefully designed and executed. Overall, I enjoyed the following strengths:

- **Experimental completeness:** The paper is generous in providing extensive empirical evidence across synthetic tasks (sparse parity, PCFGs) and real-world datasets (Wikipedia, Books).

- **Theoretical Analysis:** Offers mathematical proofs demonstrating sample complexity benefits in specific cases like sparse parity learning. 

- **Memorable Observations:** Beyond the idea that checkpointed learning leads to faster optimization, it identifies phase transition in the teacher's training where intermediate checkpoints can be most beneficial to the student. These correspond to acquring new skills of increasing complexity.

- **Impact and Practical and actionable implications**

### Weaknesses
- **Alternative intepretations, e.g. winning subnetworks and the Lottery Ticket Hypothesis:**

The first thing I thought about, as a possible intepretation of the empirical (and theoretical) findings of the paper is the lottery ticket hypothesis (LTH), which could alternatively explain the benefits observed in progressive distillation. The hypothesis posits that within a large neural network, there exist smaller subnetworks (winning tickets) that can learn the tasks efficiently. Searching for these subnetworks takes long, but once they are identified, training only on them makes the learning faster. 

So while the paper frames the intermediate checkpoints as providing an implicit curriculum of increasing task complexity, this could be reinterpreted as the teacher progressively revealing parts of the winning subnetwork, narrowing the student's search space at each step. Hence a crucial question is: could the novelty in using intermediate checkpoints be just an operationalization of the lottery ticket hypothesis in a distillation context (i.e. Guided Search) rather than a fundamentally new concept (implicit curriculum)? 
There, the student model is being guided to explore progressively smaller regions of the solution space, as per LTH. These correspond to learning features of increasing complexity as the paper points out, but that's because that's how networks progress in their learning of the best parameters to solve a problem. By emphasizing the curriculum aspect, the paper might divert the "reader" from other factors contributing to accelerated learning, such as the inherent properties of the optimization landscape or the effects of network pruning (implicit or explicit).

And both interpretations would be aligned with the empirical and theoretical findings. 

First, the fact that not all checkpoints provide good performance could be  due to the model learning more and more complex skills as it discovers the winning tickets, and not because the tasks are more and more complex (curriculum). Matter of fact, the training data is not going from simple tasks to complex ones, what goes from simple to complex is what the model has learned, not what is was trained on. Said differently, the discovery of progressively complex features could be the result/consequence of a guided search to efficient parameter configurations.  

Second, looking at the sample complexity improvement that the authors prove:
"the total sample complexity needed for the student to reach ϵ-loss using progressive distillation with 2 checkpoints is Õ(2^k d^2 ϵ^−2 + k^3). However, one-shot distillation requires at least Ω(d^k−1, ϵ^−2) samples"
could perhaps be reinterpreted as: The student benefits from getting direct signal about which features are important (winning tickets), rather than having to discover them from scratch like in one-shot distillation.
Same goes for the theorem about intermediate checkpoints having stronger correlations with degree-1 monomials: could be read as a curriculum of increasing task complexity, or about how the teacher naturally learns (simple correlations first) and why intermediate checkpoints help (they provide clearer signal about important features).

Seen this way, calling this an "implicit curriculum" might be misleading because the task complexity is constant, and what is changing is the model's internal learning progression. 

-  **Other minors**

The specific temperature values, as well as the choice of dramatically different temperatures (10^-4 vs 10^-20) based on vocabulary, could benefit from more rigorous exploration. Ok, the authors still acknowledge this as a limitation they defer to future work.

### Questions
The only perhaps unanswered question in this paper is: are simpler features learned first because they're easier, or because they're sufficient for initial loss reduction? Does the progression represent increasing task difficulty or natural optimization paths as the student discovers incrementally winning subnetworks?

As discussed above, it would be interesting to at least provide an analysis of how progressive distillation differs from or extends the Lottery Ticket Hypothesis (and the variants of progressive stacking/learning subnetworks etc). Would be (may be) interesting to discuss whether there could be experiments to show if the observed acceleration is due to narrowing the search space via subnetworks and how this impacts the interpretation of an implicit curriculum.

**Future experiments to isolate effects/figure out the right interpretations**

Below are just some ideas, probably mostly for future work. 

It would be interesting to evaluate (in the experiments of Figure2), the corresponding sizes of the winning subnetworks as per the lottery ticket hypothesis. 

One could also analyze the sparsity patterns of successful networks at different checkpoints, compare with explicit lottery ticket pruning approaches, then try to isolate whether the benefits come from parameter space guidance or feature complexity. 

**Quick check of the literature**   
The following is (slightly) related from a quick search, mainly for your interest:
https://arxiv.org/pdf/2402.05913 too recent of course. 
Older work on progressive stacking: https://proceedings.mlr.press/v202/j-reddi23a.html  also only slightly related
But more generally, it might be useful to reprobe the related work, with the lens of using winning subnetworks to accelearate learning, perhaps checking how they form and how they help accelerating learning (whether there's an increase in task complexity ...)

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper demonstrates how progressive distillation helps in better training of the student models by inducing an implicit curriculum. The experimental results demonstrate that progressive distillation results in a higher sample efficiency in all cases as well as a higher performance in some cases, as compared to vanilla knowledge distillation (referred to as "one-shot distillation" in the paper). The results also show that progressive distillation induces an implicit curriculum wherein the intermediate checkpoints provide a stronger learning signal and act as "stepping stones" for the student models during the learning process. These results are validated by experimenting across two different model architectures and 3 different tasks. In the case of the sparse parity task, authors also provide a theoretical proof of how progressive distillation with 2 checkpoints (one intermediate and one final) leads to a better sample complexity as compared to one-shot distillation.

### Strengths
* The paper is well-written and easy to read.
* The paper includes results on tasks across different complexity levels - going from a toy setting of sparse parity to PCFGs and then to a non-synthetic task of natural language modeling.
* Authors also run experiments across multiple model architectures, name MLPs and transformers of different sizes.
* The induced curriculum is discussed from a human interpretability point of view (i.e. showing the correlation between degree 1 monomials and the logits of the intermediate teacher checkpoint in the sparse parity task, and drawing similar analogy in the PCFG task).
* The paper (more specifically; the appendix) includes further extensive experimentation on the settings discussed in the main paper.

### Weaknesses
* There is a typo in Definition 4.3: I believe it should be "boundary of span(n^{(i)})" instead of boundary of n^{(i)}
* Discussion about how the relative sizes of teacher and student models were decided is missing. It would be interesting to see a study of how the performance is affected w.r.t the size of the student models
* Empirical analysis on tasks in the vision domain and with other model architectures such as CNNs and recurrent networks would strengthen the paper significantly.

### Questions
* How were the sizes of student models decided? Can the authors show some results on one task (preferably the natural language modeling OR the PCFG tasks) for what happens when the sizes of student models are varied while the size of the teacher size is kept constant?
* Can the authors run similar experiments on a simple multiclass classification task in the visual domain? Even results on something simple such as multiclass classification in CIFAR-100 with an ImageNet pretrained ResNet-18 finetuned on CIFAR-100 as the teacher, and a smaller CNN as the student would be interesting to see.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides theoretical and empirical analysis on how progressive distillation can be helpful to speed up knowledge distillation. Specifically, the author studies a toy use case in sparse parity with synthetic data to show both mathematically and empirically how using intermediate teacher checkpoints can assist student models to learn faster. Then they run progressive distillation experiments in PCFGs and BERT masked token prediction using Wikipedia and Books data to further verify their findings.

### Strengths
1. The paper is well written, and despite the complexity of the narrative, it is generally easy to follow, and I enjoyed the reading.
2. Though only applied to a simple use case, the mathematical analysis does provide useful insight about sample efficiency of progressive distillation.
3. The metrics selected in the analysis such as $\mathcal{M}_{robust}$ is quite useful to understand the feature learning aspect of the method.
4. The authors run experiments in various settings including three datasets, MLPs, BERT and GPT-2 (in the appendix) to show the gains can be generalized.

### Weaknesses
Despite the strength, I think the paper can be improved.
1. I understand the necessity to use a toy use case (sparse parity) to show rigorous mathematical analysis, but the following experiments can be more practical in order to provide stronger empirical evidence of the effectiveness of progressive distillation.
- Instead of masked token prediction, can run experiments in challenging NLP tasks such as QA, summarization and long-form generation.
- Can also experiment with more recent LLMs - GPT-2 in the appendix is a good start, but it is still a very outdated model.
2. Some writing (particularly Theorem 3.2) can be made even clearer. For example, L291-292, I didn't quite follow where higher degree odd polynomials come from. L296-297 "This gap allows .. to grow quickly with only ... samples." This statement isn't clear if I don't read the entire proof in the appendix. Please consider writing it in a more intuitive way in the main text.
3. Probably unfair as this is more of an analysis paper, but the overall contribution appears to be limited considering the scope of its experiments.

### Questions
1. For experiments in Figure 2, how do you explain the results that the best correlation (Middle) with in-support variable does not lead best accuracy (Left)?
2. L406, what's the formula for total variation?
3. How do we pick the most appropriate teacher checkpoints in other more complicated tasks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Knowledge distillation is a widely used technique for training smaller "student" models by leveraging the knowledge captured by larger, pre-trained "teacher" models. This paper focuses on progressive distillation, where a student model is trained on a series of intermediate checkpoints from the teacher's training process, as opposed to one-shot distillation, which relies solely on the final teacher checkpoint.

The authors provide both empirical and theoretical evidence that progressive distillation accelerates student learning and leads to better generalization compared to one-shot distillation. They attribute this improvement to an "implicit curriculum" embedded within the intermediate teacher checkpoints. This curriculum emphasizes easier-to-learn aspects of the task, facilitating the student's learning process.
Two main distillation strategies are compared: one-shot distillation and progressive distillation.  One-shot distillation involves training the student with a fixed, converged teacher model.  In contrast, progressive distillation utilizes multiple checkpoints from the teacher's training trajectory.  The authors also explore a variant of progressive distillation where only a single, carefully selected intermediate checkpoint is used.    

The authors delve deeper into the mechanisms behind progressive distillation by utilizing the sparse parity task, a well-established benchmark for studying feature learning in neural networks.  They employ teacher and student models with identical architectures but varying sizes.  For MLPs, the model width is adjusted, while for Transformers, the number of attention heads is modified.  Increasing either parameter effectively increases the number of "parallel search queries" the model can perform during training.    

A key finding is that the intermediate teacher checkpoints implicitly provide a "degree curriculum."  This means that the checkpoints guide the student model to learn features in a progressive manner, starting with simpler, lower-degree features. Remarkably, progressive distillation using just a single, well-chosen intermediate checkpoint can surpass the performance of one-shot distillation.  Furthermore, the authors demonstrate that progressive distillation reduces the sample complexity of learning the sparse parity task.    


The authors conclude by highlighting the effectiveness of progressive distillation in improving feature learning through the implicit curriculum present in intermediate teacher checkpoints.  They discuss the importance of teacher temperature as a hyperparameter in knowledge distillation and acknowledge limitations in their exploration of this aspect.  Further investigation into the precise role of temperature, particularly its impact on optimization, is proposed as a promising direction for future research.    

Another avenue for future work is extending progressive distillation to the setting where student models are trained on synthetic data generated by teacher models.  This approach has shown significant potential in recent studies. The authors identify key differences between their current work and generation-based methods, primarily in the type and quantity of supervision provided.  Bridging this gap and developing a unified framework for progressive distillation in various training scenarios is an important challenge for the field.

### Strengths
Originality - The paper presents a novel perspective on progressive distillation by identifying and formalizing the concept of an "implicit curriculum." While prior work has explored progressive distillation empirically, this study delves deeper into the underlying mechanisms and provides theoretical grounding for its efficacy. The connection between intermediate teacher checkpoints and an implicit curriculum is a fresh insight that contributes to a better understanding of knowledge distillation.    

Quality - The research is technically sound, employing rigorous methodology and comprehensive experiments. The authors combine theoretical analysis with empirical validation, drawing on diverse tasks like sparse parity, PCFGs, and natural language modeling. The use of multiple progress measures to quantify the implicit curriculum further strengthens the quality of their analysis. The study is well-designed, and the results are convincing.    

Clarity -The paper is clearly written and well-organized. The authors present their ideas in a logical progression, starting with a clear motivation and gradually building up their analysis. The concepts are well-explained, and the figures effectively illustrate key findings. The paper is accessible to readers with a background in knowledge distillation and deep learning.


Significance - The findings have significant implications for the field of knowledge distillation. By elucidating the role of an implicit curriculum in progressive distillation, the study provides valuable insights for designing more effective distillation strategies. The theoretical results on sample complexity offer a deeper understanding of the optimization benefits of progressive distillation. The practical implications of this work are substantial, particularly for training efficient and capable models in resource-constrained settings.

### Weaknesses
Some possible improvements I can see are the following :-

a) more investigation of impact of temeprature on knowledge distillation. this seems to be a bit missing in the main sections of the paper
b) analysis of how implicit curriculum learning varies across model layers, across datasets, trainign objecives etc

c) exploring more tasks and architectures

d ) explore interaction of optimization algorithms, batch size etc with curriculum learning

### Questions
No questions beyond suggestions above:

### Soundness
3

### Presentation
3

### Contribution
3
