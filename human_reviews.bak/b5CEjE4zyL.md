# Neural Networks Decoded: Targeted and Robust Analysis of Neural Network Decisions via Causal Explanations and Reasoning

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Despite their success and widespread adoption, the opaque nature of deep neural networks (DNNs) continues to hinder trust, especially in critical applications. Current interpretability solutions often yield inconsistent or oversimplified explanations, or require model changes that compromise performance. In this work, we introduce TRACER, a novel method grounded in causal inference theory designed to estimate the causal dynamics underpinning DNN decisions without altering their architecture or compromising their performance. Our approach systematically intervenes on input features to observe how specific changes propagate through the network, affecting internal activations and final outputs. Based on this analysis, we determine the importance of individual features, and construct a high-level causal map by grouping functionally similar layers into cohesive causal nodes, providing a structured and interpretable view of how different parts of the network influence the decisions. TRACER further enhances explainability by generating counterfactuals that reveal possible model biases and offer contrastive explanations for misclassifications. Through comprehensive evaluations across diverse datasets, we demonstrate TRACER's effectiveness over existing methods and show its potential for creating highly compressed yet accurate models, illustrating its dual versatility in both understanding and optimizing DNNs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper present method grounded in causal theory to estimate the causal link between the various layers of neural networks. This technique is used for both the explanation of the model and its optimization.

### Strengths
The use of the causal paradigm for building an explanability tool ensures it has interesting groundings. 
The related work section is thorough.
A various number of considerably unique experiments are considered.

### Weaknesses
1. The first major weakness concerns the writing of the paper.

1.1 I see the « proofs » in section A of the appendices as explanations as to why the propositions and theorems presented in the main paper make sense and are coherent, and not formal nor rigorous proofs (especially A.1).

1.2 In section 4.1, no explanation is given about what P is in practice; the proportion p that is used. The layer grouping is explained in Section 3, but no details regarding the inputs themselves are provided. How are the most important features found? Many parts of the approach and the experiments lack clear explanations of the various steps, leading to many unanswered questions (see the Questions section) and questions found in the present section. Figure 1 does not give much information about the presented approach. How are the counterfactuals generated in practice, and how are they used? How is the feature contribution made?

1.3 There are many inconsistencies or errors in the notation and the writing itself, see ** Typos and such**.

2. « Causality » is central in the article, yet I see no causality in the proposed approach.

2.1 For example, let’s consider the way the various layers in the network are compared via Centered Kernel Alignments. This « distance measure », or similarity measure, does not tells us anything about causality, simply about resemblance. 

2.2 The relationship is considered or examined between the various components of a single level (e.g. feature level), which truly undermine the scope of the analysis.

3.1 The approach is based on the fact that the considered networks are straightforward in their nature; the output of a layer is considered to be the input of the next layer, but what if more sophisticated architecture is considered, such as U-nets? How would the approach work in the case of recurrent networks? Or networks with parallel components, such as attention heads?

3.2  I am conceptually unsure of the approach, in that it only considers huge perturbations made by a single layer. What about if the whole network is a composition of many small perturbations, such as in Variational autoencoders?

3.3 It is said that if a group of layers is similar, they can be regrouped in a single transformation $g_{i,j} = fj ◦ fj−1 ◦ . . . ◦ fi = fi$, but how seeing this part of the network as a single transformation helps to reduce the number of parameters? Simply replacing $ fj ◦ fj−1 ◦ . . . ◦ fi$ by $fi$ simply cannot be done when the dimensions of the various layers aren’t the same.

4. The last concern is about the contributions of the paper. Line 532 : « Through our foundational principles and findings, we have ascertained that by producing intuitive, human-interpretable explanations, TRACER offers outstanding transparency to neural networks [...] » The paper proposes a way to compute how big of a transformation is applied at each step of the network, and to generate counterfactual, so I would not say so.

4.1 Concerning the originality of the approach: The use of counterfactual is not a contribution, for the same procedure could be used with a different model



**Typos and such**

-Line 89 : rior → Prior

-Line 96 : explainabibilty → explainability

-Line 161 : What does do() mean?
-Line 172 : Based on the notation only, it seems like it is the masked input that is in $\{0,1\}^d$, where, I guess, it is implied that it is the mask that is in $\{0,1\}^d$.

-Line 175 : Please define the « |= » operator.

-Definition 2 : In both C2 and C3, the notation is wrong, for you consider applying masks of dimension different than d to set of examples of dimension d. For example, in C3, instead of « for all M’ $\subset$ M », you probably meant « For all M’ such that |M’|₁ < |M|₁ : … ».

-Line 231 : Please provide citations for «  … prevalent approach for quantifying the
similarity between high-dimensional embeddings. »

-Line 231 : Notations are introduced for talking about the inner components of neural networks at too many various places in the article; there should be a section where this notation is properly introduced. Please state explicitly the dimensions of $f_i$ and $f_j$. It is not obvious only looking at the notation whether $K_i$ is a matrix of size $nxn$ or else.

-Line 252 : The composition increases the layer number at certain places (line 252, 259) but decreases at other places (line 259, 266). Please be coherent with the mathematical notation.

-Line 197 : The usage of « nodes » in « causal nodes » refers to the concept of layer in this work, but « node » typically refer to a single neuron of a layer. It wold be less ubiquitous to use another term for talking about « causal nodes ».

-Line 359 : when citing MNIST, please consider the following indication: http://citebay.com/how-to-cite/mnist/. 

-Line 378 : The hyperparameters that were used should be found in the appendices.

-Line 408 : This letter P (with this exact calligraphy) is already used for Probability.

-Table 1 : C1, C2 and C3 already refers to concept in Definition 2. Please use distinct notation.

-Line 512 : « In this study, we focused our evaluations of TRACER on white-box neural network. » I consider neither AlexNet nor ResNet-50 to be white-box models.

### Questions
1. Section 3.2.1 : Why consider a single unique value to replace a subset of $x$, and why would it be different than say 0?

2. Line 241 : « Flexibility: It accommodates various kernel functions, such as linear or Gaussian, enabling flexibility based on specific requirements of the analysis ». Please explain in which situation it would be preferable to have a Gaussian kernel instead of a linear one. Why this current choice of kernel?

3. Line 287 : Why consider the many neurons of a layer as realizations of a random variable? To which probability are associated?

4. Line 292 : How does $P (g′i(x) | do(X = x′))$ and $P (g′i(x) | do(X = x))$ are computed in practice?

5. Line 408 : How does P is found in practice and what does P actually outputs? A mask?

6. Table 1 : How is the simpler model created?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes TRACER, a method for generating "explanations" of neural network behavior with methods from causal inference. TRACER groups neural network layers into "groups" based on their similarity, and then computes a causal effect score for each input dimension in terms of how that feature affects the intermediate network layers and the final output. This is done by making a series of interventions to the input and observing how that affects downstream layer groups and the network output. The authors claim that the method generates more sensible saliency maps than existing methods, as measured by a "reliability score" that they define. They also claim that their method can be used to compress an MNIST classifier by over 99% without harming accuracy.

### Strengths
* The paper is well-presented with high-quality figures
* The research direction, of using causal inference for explainability, is an interesting and important one.

### Weaknesses
* The paper uses a lot of vague and flowery language that makes me worry that it may be substantially LLM-written.
* The authors claim that they can reduce the size of an MNIST classifier by 99.42% without substantially harming model performance. This seems too good to be true, and their method for doing this not very clearly described.

Overall, I am having a hard time assessing the method and claims of the paper, so will give a score of 5 for now with low confidence.

### Questions
* You say that you use a pre-trained AlexNet model for MNIST classification, however AlexNet was an ImageNet model. Did you fine-tune it with a new head to classify MNIST digits? If so, why isn't this described anywhere in the paper?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a novel method, called TRACER, to investigate the causal dynamics inside DNNs by intervening on input features. They further use TRACER to generate counterfactuals to improve explainability.

### Strengths
- The paper is well organized and well written.
    
- The proposed method is technically sound.
    
- The problem is of great importance and of interest to the community

### Weaknesses
- As we know that DNNs are strong correlation learners, the learned links/parameters between neurons/layers might contain lots of strong spurious connections, rather than causal connections. In this case, we cannot distinguish them even though intervening on input features and monitoring the changes on intermediate outputs. Thus, I guess that it might be difficult to learn a true causal graph.
    
- Many details in the experimental part are missing, e.g., what interventions are performed in each specific experiment? What specific regularization metric is chosen for the counterfactual analysis? etc.

### Questions
- It seems not to show a counterfactual generation for a misclassified ImageNet sample? I want to see the quality of such a bit complicated counterfactual.
    
- It is well known GANs have the mode collapse problem. Does it occur in your experiments? If so, how to deal with it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper (TRACER) aims to analyze the causal dynamics of deep neural network (DNN) decisions without altering their architecture. By intervening on input features, TRACER tries to identify feature importance and constructs a causal map.

### Strengths
- aims to  estimate the causal mechanisms underpinning DNN decisions

- aims to provide explanations for correct and misclassified samples

### Weaknesses
-  The paper is poorly structured and difficult to follow, with several instances of sloppiness. For instance, in lines 157-160, it says the first level of PCH is Association; However, it is typically Abduction, which involves inferring exogenous noises.


>Association: We extract dependency structures from the DNN activations and outputs $P(Y^{(i)}| X)$
where $X$ and $Y^{(i)}$ represent the input and the $i$-th layer’s output variables, respectively;

Is $Y^{(i)}$ exogenous noise?  **Please clarify their interpretation of the Association level and how it relates to abduction**

- A clear description of causal variables is missing. 
  >line (204-213) : Assuming b to be causally independent (e.g., binary mask), all input
features, before and after interventions, can be considered exogenous variables in the causal map.

How? **Please justify.**

- Many definitions and theorems lack adequate motivation. For instance, the reasoning behind the Average Causal Effect presented in that specific form is unclear. **Please provide more context or motivation for key definitions and theorems, particularly the Average Causal Effect.**

- >*Upon obtaining the similarity measures, we establish causality by grouping layers based on their CKA values* line (246-251)...causal explanation depends on the grouping based on predetermined threshold $\epsilon$. 

Different thresholds will give different causal structures or explanations for the DNN. The consistency of the explanations is not discussed.
**How does the choice of threshold affect the stability and reliability of causal explanations?**

- There is no clear explanation for selecting $\epsilon$, which is a critical hyperparameter. **How do you select or tune the 
 parameter? Please  include any empirical studies or theoretical justifications for your choice**

- Figure 1 does not adequately illustrate the methodology or framework.

-  misses relevant literature, such as:
> - *Neural network attributions: A causal perspective*, Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar, and Vineeth N Balasubramanian. 
> - *Towards learning and explaining indirect causal effects in neural networks*,  Abbaavaram Gowtham Reddy, Saketh Bachu, Harsh Nilesh Pathak, Ben Godfrey, Vineeth N. Balasubramanian, V Varshaneya, and Satya Narayanan Kar. 

**A comparison with these works is necessary..**

### Questions
Please see the weaknesses...

### Soundness
2

### Presentation
2

### Contribution
2
