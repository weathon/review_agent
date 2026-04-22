# On Why Form Shapes Reasoning: Structuring Latent Program Networks with Category-Theoretic Constraints

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Human reasoning is inherently structured: we perceive, compose, and abstract patterns to make sense of the world. Following Kant’s view that cognition imposes structure on experience, we ask how neural networks can acquire structured, compositional reasoning. We present a category-theoretic formulation of Latent Program Networks (LPNs), neural architectures that represent programs as continuous latent vectors inferred from input–output examples. We treat latent transformations as categorical morphisms and introduce differentiable constraints enforcing associativity, identity, and closure, thereby shaping the latent space into a compositional system without explicit symbolic rules. On structured grid-transformation tasks, these constraints significantly improve compositional generalization, latent alignment, and interpretability. Our results demonstrate that category-theoretic structure can be imposed on latent representations to induce compositional reasoning in neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes adding several inductive biases from cathegory theory in form of additional losses in the training phase of LPNs to improve their overall compositional abilities. The authors perform evaluations on grid transformation tasks.

### Strengths
I very much like the idea of this work and find it valuable for the community. The paper is refreshingly well structured and written making it easy to follow what is being introduced and why. The experimental evaluations also show strong evidence for the author's claims.

### Weaknesses
Overall, there are mainly a few clarifications. The main weakness, and I hate to be that reviewer, is that the experimental evidence though strong is only from one specific task/domain. It would be great if the authors could provide more experimental evidence of the overall findings. E.g. scale up the data in terms of grid size, look at more x-step compositions, or different kind of transformations. Or indeed use the the ARC challenge as in the original LPN paper.

It would be important to specify what the model is in ll 194.

I think an important baseline would be, instead of training via the losses, create a training set that explicitly represents the targeted categorical constraints.

ll 290: "Structured latent composition encourages semantic regularity: traversals in latent space yield coherent transformations" --> this is really interesting, but is there a way to specifically see this in terms of results? Right now I am missing evidence for this claim.

Minor: Providing some name or pseudonym for the proposed approach might make it more intuitive rather than "Full" in Tab. 2 or "Method/Model" as section header for section 3. 

Also an overview figure to visuallize the intuition behind the training setup would be good.

Also please fix the table overflow of Tab. 2.


Overall, if the authors can provide justifications or additional material regarding these issues I would definitely consider raising my score as I find the paper valuable enough.

### Questions
Tab. 1: why do the authors provide this table if only three fo these metrics are actually used in the evaluations?

Tab 2.: Why were these particular metrics chosen from the set in Tab. 1? What does Closure tell us exactly? Maybe a formula would be good for this.

ll 260-264: Interesting that identity is so important. What is the intuition behnd this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors attempt to impose structure on the latent vectors learned in an autoencoder. They do this as a means of improving compositional reasoning. They use category theory to implement the structure.

### Strengths
- the authors are addressing an interesting issue of symbolic structure within neural latent representations
- the authors took inspiration from human reasoning
- what was presented in the paper was well written, clear, easy to follow

### Weaknesses
My main concerns for this paper are that it acts as though modern LLMs and video models don't exist, the specifics of the presented model/approach lacked detail/clarity, and there were multiple claims that lacked supporting evidence (see below).

- lines 035-036 need a citation to defend the specific claim that NN's lack explicit structure. A few citations that potentially dispute that claim. Geiger et al. Finding alignments between interpretable causal variables and distributed neural representations, 2023. and Griffiths et al. Whither symbols in the era of advanced neural networks?, 2025.
- lines 076-077 seem to be ignoring the fact that modern LLMs are able to compositionally generalize in many tasks
- for the first paragraph in section 3.1, it might help to give a concrete example of why we might care about latent program networks. Provide a concrete problem that they are capable of solving or used to address. Do you maybe mean that they're optimized at "training" time, instead of "test" time? More elaboration would be helpful. You have the space for it.
- in the Tasks section (lines 187-191) how does the model know what type of transformation to perform? Is it just supposed to match the statistics of the dataset?
- what is a composed latent (line 211)? how is it constructed?
- lines 256-257 need supporting evidence. why can't the model memorize the solution?
- what does the model consist of? is it a multi-layer perceptron? a convolutional neural network?
- difficult to interpret results with so much ambiguity surrounding the model

### Questions
See weaknesses section.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper drops the test-time latent optimization of Latent Program Network and adds a few regularization terms as losses to the latent "program" variables. The additional regularization terms are to train a composition operator of two "latent programs" following associativity and identity constraints. The model is evaluated on a few customized synthetic simple grid transformation programs.

### Strengths
The paper studies models with latent programs.

### Weaknesses
The model is only evaluated on simple synthetic grid transformation programs, not even ARC-1, and the model does not perform perfectly on those synthetic tasks.

The paper lacks details about models (like architectures of encoders and decoders) and datasets (e.g., the synthetic training & val datasets).

### Questions
* How does the model perform on ARC-1 and ARC-2?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces a modification Variational Autoencoders (VAEs) that include differentiable constraints on the latent based on category theory. The authors claim this method to be related to Latent Program Networks (LPNs). The proposed method is tested on the ARC challenge. Overall many details are missing.

### Strengths
1. The idea or regularizing latent variables with general category constraints is interesting.

### Weaknesses
Some aspects of the presented method are unclear and details seem missing.

1 .the methods are referred as "Latent Program Networks (LPNs)" but it also states "latent programs are not directly predicted; instead, they are optimized at test time via gradient descent to minimize reconstruction error" and "In this work, we do not adopt test-time latent optimization.". This seems like a big departure from LPNs which would be otherwise just a regular VAE with additional constraints.

2. what architectures and sizes are used to parametrize the networks (both theta and psi ones)

also the experimental setup lacks needed hyperparameter sweeps (also, see questions below). 

1. No weights for the different constraints are explored, a and a single value of regularization weight beta is used. A reasonable setup would at least compare the full method and the baseline under a sweep of beta values on a held out set different from the ARC test set.

2. results seem to be relatively fragile, with performance collapsing with small changes over the "Full" method. In this setting the parameter sweep is even more relevant.

### Questions
the authors state 

> We selected the KL target empirically based on preliminary runs that balanced latent capacity and regularization

1. I understand this as the value of beta=0.003 was determined based on initial runs. Was this on some held out set different from the ARC test? is this value set based on the "Full" run? optimal beta may be different for the different experiments particularly the "Sequential VAE (free-run)" which has less constraints and therefore less overall regularization. 

2. What is the behavior of the methods in table 2 under a sweep of the beta parameter.

### Soundness
2

### Presentation
1

### Contribution
2
