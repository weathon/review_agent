# Expand Neurons, Not Parameters

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
This work demonstrates how increasing the number of neurons in a network without increasing its number of non-zero parameters improves performance. We show that this gain corresponds with a decrease in interference between multiple features that would otherwise share the same neurons. On symbolic tasks, specifically Boolean code problems, splitting each neuron into sparser sub-neurons with knowledge of the clauses systematically reduces polysemanticity metrics and yields higher task accuracy. Notably, even random splits of neuron weights approximate these gains, indicating that reduced collisions, not precise assignment, are a primary driver. Consistent with the superposition hypothesis, the benefits of this framework grow with increasing interference: when polysemantic load is high, accuracy improvements are the largest. Transferring these insights to real models—classifiers over CLIP embeddings, CNNs, and deeper multilayer networks—we find that widening networks while maintaining a constant non-zero parameter count consistently increases accuracy. These results identify an interpretability-grounded mechanism to leverage width against superposition, improving performance without increasing the number of non-zero parameters. Such a direction is well matched to modern accelerators, where memory movement of non-zero parameters, rather than raw compute, is often the dominant bottleneck.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Fixed-Parameter Expansion (FPE), a method to improve neural network performance without increasing the total count of non-zero parameters. The core hypothesis is that smaller, dense networks suffer from "polysemanticity," where individual neurons represent multiple unrelated features, causing interference that degrades performance. FPE addresses this by creating a wider, sparser network with the same parameter budget. The authors show that FPE improve accuracy on both symbolic Boolean tasks and real-world image classification.

### Strengths
* **S1:** From my knowledge, the concept of decoupling the number of neurons from the number of non-zero parameters appears to be a novel contribution.
* **S2:** The authors provide direct evidence that FPE leads to more disentangled representations, connecting the architectural change to a concrete improvement in the network's internal feature geometry.

### Weaknesses
* **W1. Limited Experimental Setting:** The empirical validation on real-world tasks is exclusively performed on MLP classifiers that operate on pre-computed, frozen CLIP embeddings. This fails to demonstrate if the FPE principle holds for the other models. Without such evidence, the claim that FPE can can guide in improving architecture design, is not yet fully substantiated.

* **W2. Lack of Absolute Performance Metrics:** The paper exclusively reports relative improvement in accuracy. While scientifically valid for isolating the effect of FPE, this makes the results difficult to interpret from a practical standpoint. Without knowing the baseline accuracy, it is hard to understand the significance of the improvements. Including a simple table with absolute accuracies for at least one key experiment would greatly enhance the paper's transparency and impact. 

* **W3. Clarity:** The paper’s overall clarity is limited, and parts of the presentation are at times confusing. Additionally, the paper's central claim is that FPE is a "post-training" procedure. However, in the caption of Figure 5, the authors claim to pre-train the model for 25 epochs before FPE. My understanding is that the baseline model is trained for a short period, then expanded via FPE, and then training is continued for a substantial number of epochs (e.g., 1000 in the symbolic task). This lengthy "fine-tuning" phase makes it unclear whether FPE is a technique for improving a converged model or, rather, a structured re-initialization method used early in a much longer, combined training process. 

* **W4. Unclear Practical Integration:** While successfully establishing a proof-of-concept, the paper lacks a discussion on how FPE could be integrated or how it might interact with other optimization techniques like quantization or knowledge distillation. This omission, combined with the lack of absolute performance data and the limited experiments, makes it difficult for a reader to assess the effort versus reward of adopting this method in an applied setting.

* **W5. Reproducibility:** Providing the implementation would be a valuable contribution to the community.

### Questions
* **Q1. Choice of CLIP and Additional Results:** Could the authors elaborate on the decision to use only frozen CLIP embeddings? Additionally, what's the authors hypothesis on how FPE would perform in an e2e training scenario where the feature extractor's weights are also being updated. Would the benefits of FPE be amplified or diminished? Could the authors provide results with other embeddings?

* **Q2. Absolute Accuracy:** Would the authors be willing to provide the final absolute test accuracies for both the dense baseline and the FPE models for the image classification experiments presented in Figure 5?

* **Q3. Clarification on the Training Methodology and the Role of FPE:** I would appreciate further clarification on the multi-stage training methodology, which I found to be a key source of ambiguity in the paper. A more detailed explanation would help in understanding the precise nature and contribution of the FPE intervention. Specifically:
    *   **Q3.1 Rationale for Intervention Timing:** Could the authors elaborate on the rationale for the timing of the FPE intervention (e.g., applying it after 25 epochs)? How sensitive is the method's effectiveness to this choice? For instance, does applying it much earlier in training or only after the baseline model has fully converged significantly alter the results?
    * **Q3.2 Rationale for Training Duration:** Similarly, what was the rationale for the extensive subsequent training period (e.g., 1000 epochs for the symbolic task)? Does FPE require this lengthy fine-tuning to realize its benefits, or could comparable gains be achieved with a much shorter training schedule?
    * **Q3.3 Conceptual Interpretation of the Method:** Should FPE be interpreted as a method for structured re-initialization that provides a better starting point for a long training run, or as a method to apply after the training that does not require further fine-tuning?
    * **Q3.4 Suggestion on Terminology:** I found the overlapping terminology to be a primary source of confusion. To improve the paper's clarity, the authors could consider explicitly distinguishing in the text between the use of the "pre-trained" CLIP model for feature extraction and the "initial training phase" of their own classifier.

* **Q4. Ablation on the Classifier:** For the vision tasks, the paper mentions using a 1-layer network for all the datasets except for Imagenet1K where a 5-layer network was used. Was there any ablation study on how the architecture of the classifier itself interacts with the FPE procedure?

### Soundness
2

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
4

### Summary
This paper introduces Fixed Parameter Expansion (FPE), a method for improving neural network performance by increasing the number of neurons while keeping the total number of non-zero parameters constant. The core idea is to reduce "polysemanticity," where a single neuron represents multiple features and causes interference. FPE handles this by partitioning the incoming connections of each neurons into non-overlapping components. This disentangles feature representations without increasing the model's non-zero parameter count. The authors show that on both symbolic reasoning tasks and real-world image classification datasets (like CIFAR-100 and ImageNet), FPE improves accuracy. The performance gains are most significant in small networks where feature interference is a major bottleneck.

### Strengths
1.	The paper is well-written and easy to follow.

2.	The idea of increasing the number of neurons by constructing sparse input weight matrix is interesting. This approach keeps the non-zero parameter count constant.

3.	The paper conducts abundant ablation studies, which help readers understand the effect of each quantity and the limitation of the proposed method.

4.	The examples and illustrations are intuitive.

### Weaknesses
1.	The paper is motivated by previous research on interpretability, particularly studies focused on the semantic interpretability of neurons, such as the superposition hypothesis and the polysemanticity of neurons. However, the paper does not include experiments that directly address semantic interpretability. Most of the experiments focus on performance improvements, and there are no analyses or visualizations that interpret the semantics of the “disentangled” features resulting from neuron expansion. As a result, it remains unclear whether neuron expansion actually enhances interpretability.

2.	The widths of the models evaluated in this study are significantly smaller than those used in practical applications. For instance, the paper sets the hidden layer width to a maximum of 16 for models trained on ImageNet, whereas typical image recognition models commonly have widths exceeding 512. This discrepancy limits the practical relevance of the experimental findings. Although the authors state that “the models are intentionally underparameterized […] to study the emergence of superposition and interference,” it is important to note that real-world models are usually overparameterized, and the superposition phenomenon is still observed in such settings.

3.	There is no Related Work section. Although the authors mention a number of related papers in the Introduction, I still recommend adding a Related Work section to systematically survey previous studies and compare them with this paper.

### Questions
1.	This question is a continuation of Weaknesses 1. Is there a way to interpret the semantics of the “disentangled” features after neuron expansion? 

2.	Question about the training procedure. In Line 198, the paper states that models on the Boolean DNF task is trained for three stages: (1) warm-up training for 1000 epochs, (2) neuron expansion, (3) continued training for 1000 epochs. Are the models on image recognition tasks also trained with the same three-stage scheme? It seems that the paper does not mention the continued training stage for these image models. In addition, I wonder whether the weight masks are fixed during the continued training stage.

3.	Could the authors provide a comparison between FPE and Sparse Autoencoders (SAEs)? Since SAEs are also based on the superposition hypothesis, it would be valuable to see a comprehensive comparison between these two approaches.

4.	Minor. In Algorithm A3, Line 8, what does the variable $r$ mean? Is it used in later part of the algorithm?

5.	Minor. In Line 187, the sentence “The full justification can be found in Section 2.5” seems to have a typo. I guess the authors are referring to some section in the Appendix.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Fixed-Parameter Expansion, which splits each neuron into several sub-neurons and divides its connections. This means the total number of weights stays the same, while we have more width. This is motivated by the view that polysemanticity can cause feature intereference which degrades performance. On synthetic Boolean tasks and real computer vision tasks, this improves accuracy by reducing collisions.

### Strengths
1. Very simple and practical idea motivated by clear theory in understanding feature interference
2. Works beyond simple toy data to real CV tasks
3. Validation of theoretical prediction regarding collisions

### Weaknesses
1. Could be extended to modern architectural choices to see if gains persist
2. Success of random splits casts doubt on the idea of structure-awareness 
3. Interference could be measured by mechanistic methods instead of just proxies

### Questions
1. Have you tried to implement this ablation on more realistic model architectures?
2. Is there evidence of less entanglement by doing circuit analysis?
3. Where do returns diminish?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors conduct an empirical study to support the so-called superposition hypothesis, which states that neurons in small neural networks represent different features simultaneously. Unfortunately, due to this sharing arrangement, there is interference between the features represented by the same neurons, which in turn handicaps model performance. 

Therefore, the authors argue that, assuming the hypothesis is true, we should be able to split these "polysemantic" neurons into multiple neurons, each representing a single feature. Furthermore, the authors propose a method for splitting polysemantic neurons that doesn't increase the model's non-zero parameter count.

The authors demonstrate this effect on small neural networks, where they indeed observe an improvement in performance after splitting up to a point. However, if neurons are split too much, performance decreases; hence, in practice, one must find the "sweet spot" for how much to expand neurons.

### Strengths
Prima facie, the authors' work is fascinating. The fact that even the randomised masking approach can improve model performance significantly was quite surprising.

The writing is also quite clear.

### Weaknesses
My main concerns with the authors' work are regarding impact.

The experiments in the paper are very small-scale. For their pilot study, they use single hidden layer MLPs with 8-16 neurons. This is fine to verify the initial effect, but even for their "larger scale" experiments in Section 3.4, they study single hidden layer MLPs with up to 128 neurons. As such, the author's claim that "The results demonstrate that random splitting provides substantial interference reduction even at scale" is not borne out by their experiments. To support this statement, the authors would need to conduct experiments on much larger models.

I also found the starting sentence of Section 3.4 quite strange: "While FPE is motivated primarily by theory ..." - what theory are the authors referring to here? My understanding up to that point is that FPE is based on the superposition hypothesis, which is essentially a collection of purely empirical observations.

Finally, the paper lacks an explicit "Related Works" section, so it is not entirely clear where the authors' work fits into the literature.

Miscellaneous:
 - we duplicate $w_i$ across $\alpha$ sub-neurons: shouldn't it be $n_{\alpha * i: \alpha * (i + 1)}$?
 - "The full justification can be found in Section 2.5." - erroneous reference

### Questions
The random masking variant of FPE reminds me of (indeed, it essentially is) weight dropout, where we insist that exactly a fixed proportion of the weights for each activation are dropped. Firstly, I believe this connection should be noted as a related work, and second, this also suggests several other baselines to compare:
 - How does FPE compare with the suggested weight duplication scheme after which standard weight dropout is applied?
 - How does the following extended, "bootstrapped" model perform: imagine we perform FPE on the same model, but with several different seeds, after which we average the different model predictions. Could this be an even cheaper way to perform MC Dropout?

### Soundness
2

### Presentation
3

### Contribution
2
