# Intelligence at the Edge of Chaos

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We explore the emergence of intelligent behavior in artificial systems by investigating how the complexity of rule-based systems influences the capabilities of models trained to predict these rules. Our study focuses on elementary cellular automata (ECA), simple yet powerful one-dimensional systems that generate behaviors ranging from trivial to highly complex. By training distinct Large Language Models (LLMs) on different ECAs, we evaluated the relationship between the complexity of the rules' behavior and the intelligence exhibited by the LLMs, as reflected in their performance on downstream tasks. Our findings reveal that rules with higher complexity lead to models exhibiting greater intelligence, as demonstrated by their performance on reasoning and chess move prediction tasks. Both uniform and periodic systems, and often also highly chaotic systems, resulted in poorer downstream performance, highlighting a sweet spot of complexity conducive to intelligence. We conjecture that intelligence arises from the ability to predict complexity and that creating intelligence may require only exposure to complexity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores a potential connection between complexity and intelligence using data from elementary cellular automata (ECA) — a class of discrete dynamical systems that exhibit simple, complex, and chaotic dynamics. The authors find that transformers trained on complex and chaotic dynamics perform better on downstream tasks. They first train transformers on these dynamical systems, then adapt the pre-trained transformer to other tasks while freezing the transformer weights and only training the encoder/decoder layers.

### Strengths
This paper proposes a novel approach to studying connections between complexity and intelligence. By pre-training transformers on complex patterns, it aligns with large language model (LLM) studies, which have demonstrated that high-quality training data leads to better performance on testing tasks. This study conducts thorough experiments across various tasks, examining the topic from different perspectives. It reveals significant performance differences in transformers trained on different cellular automata.

### Weaknesses
The most critical issue in this paper is the method of generating training data. As mentioned in lines 146-152, you first generate a large spatiotemporal evolution, then extract only a small spatiotemporal window from it. This approach inadvertently introduces randomness into the evolution because information outside the spatial window is lost, yet its effects continue to diffuse into the observed space. The authors appear to overlook this randomness, claiming the system is deterministic. They then use a linear encoder/decoder to map states into hidden representations and predict the next state from these hidden states.

This mismatch may lead to several problems, potentially explaining some of the results observed in your paper:

1. Your deterministic model may not be strong enough to learn highly random patterns;
2. The attention pattern discovered in Figure 4 might be caused by this mismatch: For simple tasks, external unknown information may not diffuse into the space, so the cut windows contain truly deterministic dynamics, hence the transformer doesn't need to have attention on previous states. However, for complex patterns, information from states outside the window is necessary for prediction. Consequently, the transformers use previous states for better prediction, resulting in the observed attention patterns;
3. This could lead to a trivial, yet unfalsified explanation of your discovery: the reason transformers trained on complex patterns perform well on downstream tasks might be because they have more attention. And the reason they have more attention might be caused by the unknown information outside the spatial window.

To address this issue, you could predict only part of the next state. For example, given state indices 1-60 as input, you could predict only states 2-59, thereby mitigating the problem. I’m wondering if you still observing the results when removing such randomness.

Some other minor issues:

1. The models should not been named as large language model (LLM) — they are not large, and not natural language. I would suggest using transformer instead;
2. Around line 158, please define “binary vector”. Is it a vector of spatial pattern at given time?
3. In line 232, why there isn’t rule 110 in ARC Easy / Chess Move? It is Turning complete, may need more attention on it.
4. It will be good to compare to a randomly initialized transformer. Would complex ECAs beat random transformer?

### Questions
1. How does this relate to Reservoir Computing (RC)? A randomly initialized RC can also perform well on nonlinear and chaotic tasks. How do you distinguish between the sources of good performance—complexity versus reservoir effects?
2. Beyond the trivial explanation mentioned above, what other reasons might explain why the transformer learns to use attention? Have you experimented with varying model sizes? It's possible that more powerful models might not require attention, while smaller models (or those trained with high L2 regularization) might need attention to allocate more computational resources.
3. Does this relate to the principle of computational equivalence?

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
3

### Summary
The authors of this paper utilize a large language model (GPT-2) to demonstrate "Intelligence at the Edge of Chaos," showing that models achieve optimal performance in moderately complex, but not overly chaotic, systems. They train on tasks based on elementary cellular automata (ECA) and evaluate on various downstream tasks to explore the link between system complexity and emergent intelligence. The results align well with their hypothesis, though additional experiments could further strengthen these findings.

### Strengths
The paper’s topic is innovative, as I believe the emergence of intelligence is fostered by increasingly large architectures capable of handling complex data. The authors devote significant attention to explaining the concept of the "Edge of Chaos," ensuring a thorough understanding of this idea. Their explanation of the results is also convincing, and the paper’s visual illustrations are highly effective.

### Weaknesses
The experiments are somewhat limited. Adding more downstream tasks, such as nim games or toroidal chess, could enhance the findings. Additionally, testing the hypothesis on smaller models could provide further support, rather than relying solely on results from larger models.

### Questions
1. Could you include more downstream tasks, like nim games or cylinder chess, to expand the evaluation?

2. Could you test on smaller models, such as ResNet-50, BERT, or Mamba-170m, to validate the results?

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
The authors explore how the complexity of the rule of the data generative system influences the model capabilities on the downstream tasks. They use elementary cellular automata (ECA) as a rule-based dynamic system to control complexity of the training data trained GPT-2 derived transformer architecture and evaluated the trained model on reasoning/chess move prediction tasks. They found that higher but not too chaotic complex system leads to higher downstream task performance. Also, they found that the model trained with more complex system tends to obtain more complex solution rather than a simple rule, which the authors hypothesized as a reason for better downstream task performance.

### Strengths
- I really enjoyed reading the paper. The authors explore the impact of rule complexity of the training data on the model's downstream task performance. I believe it is important and timely subject as there are already many literatures suggesting the importance of data property on acquired intelligence of the neural network models. 
- The paper is well motivated and the authors made good connection to the related previous works on complexity and computational capabilities.
- Particularly, I liked the ECA data model that the authors use, which provides a good control of complexity of the rule-based dynamical system. Also, the authors used multiple measures of complexity rather than single definition. 
- The paper is well-written, it was easy to follow and understand the key claims that the authors make.

### Weaknesses
- While I am fairly convinced with the motivation and general results, but the presented quantitative result is rather weak in the results and figures they provide. Is the difference between accuracy across complexity, for example, in figure 2, shows fairly low r value.  
For wolfram's complexity class categorization, does the p-test shows significance? The absolute accuracy difference between the categories look fairly low. And the baseline performance without the ECA pretraining is not provided. 

- Did the authors run experiments with multiple seeds?

- The claimed 'emergent' intelligence is only evaluated on the downstream tasks after post-tuning, and in limited kinds. Also, the baseline performance without the pre-training on ECA is not given. 

- While authors found that the model trained with complex system tends to obtain more complex solution than the actual rule and made a interesting conjecture that this might be the reason for better downstream performance, no rigorous experiment to further investigate this point is provided. 

Further questions and comments are given in the next section.

### Questions
- Could you comment on the differences between "complexity" and "diversity" in ECA model? Is any of the complexity measures you incorporate captures diversity of the data as well? Are complex data always more diverse and vice-versa? 

-  In figure 2 violin plot, it seems like it's not only mean values but also variances that are correlated to the complexity measure and downstream tasks which I found interesting.  Could the authors provide comment?

- Related to the weaknesses I wrote above, I am curious about baseline performance of the models before pre-trained on ECA data. 

- I see different tendency on Krylov complexity compare to the other on fig 3 i.e. high variance on both very low and very high complexity measure. Is there any reason that you could think of?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper explores whether LLMs can develop generalisable reasoning abilities by pretraining on structured but simple datasets, specifically sequences generated by cellular automata. The authors hypothesise that exposing models to complex patterns, even if the underlying system is deterministic and rule-based, could improve the models’ performance on reasoning tasks. By pretraining LLMs on cellular automata generated data, they test whether this training improves performance on abstract reasoning benchmarks, such as tasks from the Abstraction and Reasoning Corpus and chess move prediction (after finetuning).

The results show that models exposed to complex patterns, those generated by "edge-of-chaos" traces, perform better on these reasoning tasks compared to models pretrained on simpler or more chaotic traces. This suggests that pretraining on richer, more complex patterns helps LLMs develop broader generalisation abilities, even when the training data lacks semantic content.

### Strengths
I like about this paper that it uses a relatively simple approach to create data with varying (and quantifyiable) degrees of complexity. The approach could help in exposing LLMs to large amounts of data, potentially leading to smaller requireemtns of corpora during pretraining. The "edge-of-chaos" idea has been investigated in neural networks before, but not as a result of pretraining with data from CAs.

The approach of pretraining on CAs and finetuning the outputs later seems like a great fit to learning representations / ICLR - an interesting way to derive good features for training models.

### Weaknesses
While I like the idea and the approach of the paper, the choice of words could be improved and language be more precise. The choice of "Intelligence" in the title and throughout the paper is awkward, the work doesn’t necessarily show "intelligence" in the sense of true reasoning or understanding: it demonstrates however that structured complexity in data can lead to the development of generalisable features or representations, that is a better choice would be to describe the approach as learning useful features or representations for predictions or other complex tasks.

The language around the CA rules is also imprecise - all CA rules are similarly simple, but some of the rules lead to complex (or chaotic, or simple...) outputs / traces. It is the generated data that exhibits the complex patterns in case of the class iv CAs. 

As an ablation, it might also be interesting to explore if it is the *structured* complexity that leads to better features, or if complex patterns are sufficient. This could be achieved by using surrogate outputs from CAs (ie scramble the temporal order), to allow the distinction between the complex (spatial) pattern or the structured, sequential nature of the data.

### Questions
In my view the paper is quite interesting and a good fit for ICLR, and would like see it accepted if some of its weaknesses are addressed. Some questions:

- (see above) What role does the temporal structure of CA sequences play in learning generalisable representations?
- Is the amount of data required from CA pretraining comparable to the amount of data required using natural langage (or similar complex data), to reach similar generalisation effects?
- what's the trade-off between complexity and volume of data

### Soundness
3

### Presentation
2

### Contribution
3
