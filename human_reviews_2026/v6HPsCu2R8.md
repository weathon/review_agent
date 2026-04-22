# Hedonic Neurons: A Mechanistic Mapping of Latent Coalitions in Transformer MLPs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Fine-tuned Large Language Models (LLMs) encode rich task-specific features, but the form of these representations—especially within MLP layers—remains unclear. Empirical inspection of LoRA updates shows that new features concentrate in mid-layer MLPs, yet the scale of these layers obscures meaningful structure. Prior probing suggests that statistical priors may strengthen, split, or vanish across depth, motivating the need to study how neurons work together rather than in isolation.  

We introduce a mechanistic interpretability framework based on coalitional game theory, where neurons mimic agents in a hedonic game whose preferences capture their synergistic contributions to layer-local computations. Using top-responsive utilities and the PAC-Top-Cover algorithm, we extract stable coalitions of neurons—groups whose joint ablation has non-additive effects—and track their transitions across layers as persistence, splitting, merging, or disappearance.  

Applied to LLaMA, Mistral, and Pythia rerankers fine-tuned on scalar output tasks, our method finds coalitions with consistently higher synergy than clustering baselines. By revealing how neurons cooperate to encode features, hedonic coalitions uncover higher-order structure beyond disentanglement and yield computational units that are functionally important, interpretable, and predictive across domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a game-theoretic interpretability framework that models neurons in fine-tuned LLM MLP layers as agents in a hedonic cooperative game, capturing their synergistic interactions. Using PAC-Top-Cover to identify stable neuron coalitions, the method reveals functionally indispensable groups that co-adapt during fine-tuning and correspond to interpretable, task-relevant features. Applied to LLaMA, Mistral, and Pythia rerankers, the approach achieves higher synergy and stronger alignment with IR heuristics than clustering or SAE baselines, offering a principled path toward mechanistic understanding of mid-layer representations.

### Strengths
- The paper introduces a novel game-theoretic formulation of mechanistic interpretability, modeling neurons as agents in a hedonic game to capture synergistic interactions. Understanding the MLP layers is always important and challenging for MI field, and this paper provides a very interesting method. 

- The paper is well-organized and clearly motivated, effectively connecting theory, method, and empirical findings. Minor improvements could include more visual intuition for coalition dynamics, but overall readability is high.

- The method is sound in theory with a good amount of theoreitcal justifications, and good amount of empirical supports.

### Weaknesses
- The results and analysis are still exploratory in nature as acknowledged in the paper. 
- The task evaluated in this paper is limited in diversity. This narrow task scope may limit generalizability of the proposed framework to other domains (e.g., reasoning, summarization, or generation)
- The limited diversity may come from another concerns on the scalability of the method and the computational cost. 
- The analysis focuses on LoRA updates in mid MLP layers (7–14) based on prior work’s observation of task activity. This may introduce selection bias: the results may not generalize (in more of a scalability sense) to other layers, architectures, or non-LoRA fine-tuning setups. An ablation showing whether the method would still detect meaningful coalitions if applied to other layer ranges might be nice. (not required especially considering the length of rebuttal and discussion period.)

### Questions
please refer to weakness.

### Soundness
3

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
2

### Summary
This paper uses coalition game theory to group neurons within LoRA fine-tuned transformers into coalitions, treating neurons as agents in a hedonic game. Neurons are grouped together based on which other neurons they strongly associate with, either by weight/activation metrics (orthogonal co-activation) or functional importance (pairwise ablation synergy). Such coalitions, when ablated, cause the largest decreases in downstream performance and track the closest with known IR heuristics. These coalitions are compared across layers, with the authors finding that coalitions tend to predominantly vanish across successive layers, with some persisting and splitting, but very few merging.

### Strengths
1. This formulation of neurons as agents in a hedonic game seems very novel and is an interesting lens through which to interpret neural networks.
2. The Hedonic-MFC algorithm, using either OCA or PAS, finds coalitions that are more important than baseline clustering methods, both in terms of function via ablation and in terms of correlation with known IR heuristics.
3. The study into the dynamics from layers 7 to 14 across the three models are interesting and support the claims about how cooperative units are predominantly pruned in successive layers, with some of them persisting or splitting, but with very few, if any, fusing.
4. The paper is written well and handles its two often disjoint subject matters in an easy to follow manner.

### Weaknesses
1. The LoRA setting seems somewhat restrictive, why was this chosen? What advantages does this have over using a pretrained language model, with ablations being the zeroing out of neuron weights instead of the reset to pre-LoRA values? 
2. The OOD performance drop for Hedonic-PAS seems partially circular: because these coalitions were selected to jointly have an outsized effect under ablation, it follows that performance loss would also be strongest here when these groups are ablated (though this is mitigated somewhat by the IR heuristic correlation).
3. The baselines presented in the paper do not seem completely fair from an information access perspective. The k-means and hierarchical clusterings both receive either weight or activation information. Hedonic-OCA receives both, and Hedonic-PAS receives an even stronger signal in outsized joint ablation degradation between two neurons. A stronger baseline should be considered for both k-means and hierarchical clustering, where they have access to the increased information that the Hedonic-MFC algorithms have access to. Do you see the same performance of Hedonic-MFC compared to, for example, standard graph community detection algorithms on the same information metrics?
4. Despite the tracking of the coalition activations with known IR heuristics, the coalitions themselves are not further explored in terms of their function. Is there any analysis that the authors have done that can reveal interpretable contexts under which the coalitions fire, such as context or topic?

Despite these weaknesses, the novel framing of the paper, the performance of Hedonic-MFC in finding important coalitions, and the study of across-layer dynamics motivate my score of 6. As a caveat, I am not very familiar with game theory, which motivates my low confidence score, but the authors explained the necessary background information well.

### Questions
See weaknesses for full explanation, but summarized:
1. Why LoRA?
2. How can the partially circular performance of Hedonic-PAS be justified?
3. Are the Hedonic-MFC algorithms competitive against stronger clustering baselines that access the same information?
4. Do the coalitions represent any interpretable features?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method for finding groups of neurons in transformer MLP layers that work together in a non-additive way, meaning their combined effect is greater than the sum of their individual contributions. The authors use tools from game theory, specifically hedonic coalition formation, to identify these “neuron coalitions” and show that they are functionally important by measuring performance drops after removing these neurons. They apply their method to language models fine-tuned with LoRA for information retrieval tasks and find that the identified coalitions are more functionally important and i"nterpretable" (based on the previously mentioned ablation and alignment with known heuristics) than baseline clustering methods.

### Strengths
- Across models, ablating hedonic neurons clearly harms results more than ablating coalitions identified by clustering methods based purely on neuron similarity or correlation. This indicates empirically that the method is identifying groups of neurons which are more structurally essential in performing the task.
- The application of hedonic game theory to neuron analysis is original within the interpretability literature from what I can tell and this work develops a connection between a new area of cooperative game theory and neural network analysis.
- The work has extensive reproductibility documentation provided as part of the supplementary material.
- The work proposes a computationally tractable way to measure this theoretical connection to game theory in NNs, which is often challenging for game theoretical given the relatively small number of players in game theory experiments/problems relative to the large number of players/layers in a neural network.

### Weaknesses
This work is relatively far from my area of research (see the low confidence score) and therefore I cannot comment too heavily on the exact technical novelty. I have done a literature review of the space to try and be an informed reviewer, but all following comments admittedly come with that asterisk.

The biggest concern I have with regards to this work is to what degree it delivers on the "interpretability" aspect of mechanistic interpretability. The authors propose a novel framework for identifying synergistic neuron coalitions and present a number of quantitative evaluations suggesting that these coalitions capture important internal structure. However, it is less clear whether this structure yields human-understandable insights. For example, while the coalitions are shown to affect model performance and align with some IR heuristics, there is little evidence that they correspond to semantically meaningful concepts or behaviors. 

Unlike other work in mechanistic interpretability, such as SAEs, the paper does not provide labeled interpretations or vizualizations that would help validate the value of these coalitions. Similarly, unlike core mechanistic discoveries, such as inductive heads, the coalitions cannot be mapped to a particular mechanism the model performs. Further work tying the ablation of these coalitions to concrete changes model behaviors or concepts would strengthen the interpretability claim.

As an aside, from the model pruning/ model compression angle, it is somewhat surprising the work doesn't try to connect to established game theoretical works applied to neural networks such as Neuron Shapley values. Removing the top-k neurons would be a stronger baseline than the clustering based approaches evaluated I believe.

### Questions
- What is the compute cost and/or runtime of computing Hedonic coalitions v.s. the strongest clustering based approach? It seems like given the large number of samples required and the long wall-clock runtime that it might be far more expensive than a clustering based approach, but this cost is never compared to the baselines. Is it O(n^2) in the number of neutrons (which seems intractible, but is stated on line 467)?

- The Hedonic sampling procedure has a relatively large number of hyperparameters. How were these swept and or otherwise decided upon?

- Why was LoRA selected as the Finetuning method to study? It seems that Hedonic coalitions could be learned even for pretrained models before Finetuning (if applied to a task with reasonable zero-shot performance) or for models fully fintetuned. If the reason is computational cost, it would be worth noting it as such.

- Why is the method only studied for LLM backbones? The method does not seem LLM/Transformer specific and could be further validated (at relatively lower cost) on smaller networks such as those for simple tasks like CIFAR10 or ImageNet.

Why was the method only evaluated for re-ranking? As far as I can tell, the method is not specific to this domain so it would be nice to validate it in others since it is being branded as a general purpose method for this type of analysis.

### Soundness
2

### Presentation
2

### Contribution
2
