# Multi-step Predictive Coding Leads To Simplicity Bias

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Predictive learning is a framework for understanding the formation of low-dimensional internal representations mirroring the environment's latent structure. The conditions under which such representations emerge remain unclear. In this work, we investigate how the prediction horizon and network depth shape the solutions of predictive learning tasks. Using a minimal abstract setting inspired by prior work, we show empirically and theoretically that sufficiently deep networks trained with multi-step prediction horizons consistently recover the underlying latent structure, a phenomenon explained through the Ordinary Least Squares estimator structure and biases in learning dynamics. We then extend these insights to nonlinear networks and complex datasets, including piecewise linear functions, MNIST, multiple latent states and higher dimensional state geometries. Our results provide a principled understanding of when and why predictive learning induces structured representations, bridging the gap between empirical observations and theoretical foundations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how *multi-step prediction horizons* influence learned representations. Using a series of *toy environments*, the authors show that when models are trained to predict several steps into the future, the resulting internal representations tend to collapse onto low-dimensional manifolds that reflect the latent structure of the environment.

They analyze this phenomenon through:

- Experiments on a *deep linear model* trained on synthetic one-hot predictive tasks.
- A proposed connection between the model’s learned solutions and the Ordinary least squares estimator structure.
- Extensions to simple nonlinear settings, a piecewise-linear environment, and a toy MNIST variant where the model predicts digit transformations.

The main empirical observation is that *longer prediction horizons* induce smoother, lower-rank latent representations, while short-horizon prediction does not.

### Strengths
- **Clarity and organization:** The paper is well-written and easy to follow. The toy setups and visualizations (e.g., Figures 2–6) effectively illustrate the claimed phenomena.
- **Empirical insight:** It clearly shows that increasing prediction horizon can lead to emergent structure in latent representations, providing an interesting demonstration of implicit regularization in overparameterized systems.
- **Bridging empirical and analytical intuition:** The link drawn between OLS structure, gradient descent bias, and representation collapse is conceptually interesting, even if informal.

### Weaknesses
1. **Not actual predictive coding:** 
Despite the framing, the models used are standard feedforward or GAN-like architectures trained with backpropagation on predictive tasks. There is no connection to biologically inspired predictive coding mechanisms or frameworks involving iterative inference or prediction error minimization. The title, abstract, and introduction should be rephrased to make this distinction clear and remove link to neuroscience.
2. **Toy nature of experiments:**
    - All the strong results occur in heavily idealized environments (one-hot states, small state spaces, or piecewise linear mappings).
    - The conclusions fail to generalize when within-class variability increases or when data complexity approaches real-world levels (as noted by the authors themselves for MNIST).
3. **Lack of theoretical depth:**
    - Theoretical analysis is *post hoc* and heuristic. The OLS and implicit bias discussions provide qualitative rather than rigorous proofs.
    - No new theorems or formal results are presented, only intuitive arguments.
4. **Limited novelty and impact:**
    - There is no compelling connection to either neuroscience (predictive coding) or impactful machine learning insights beyond toy-level illustration.

### Questions
1. How does your setup relate to predictive coding networks (e.g., Rao & Ballard 1999, Friston 2010, or modern hierarchical predictive coding frameworks)? Could your approach be reframed simply as multi-step supervised prediction?
2. Have you tested whether your results hold when using recurrent architectures trained on temporally structured data, rather than static feedforward mappings?
3. Could you provide *formal theoretical results* rather than qualitative explanations, e.g., a proof of when multi-step prediction induces rank compression or alignment with the latent manifold?
4. Given the limited generalization to realistic data (e.g., MNIST variability), what do you believe is the practical implication of your findings for representation learning or predictive modeling?

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
3

### Summary
This paper explores the conditions under which neural networks learn simple representations of environments, capturing their structure.
The paper identifies a factor that contributes to this, namely the prediction horizon. In the paper this is formalized as the magnitude of the largest permitted action. The authors show that, increasing the magnitude of the largest permitted action helps the neural network learn simple representations of the training environment that capture its structure. This is shown in simple toy settings and in a more challenging task with MNIST images.

### Strengths
* The results for the GAN model are cool and creative. These also seem robust in the sense that several regularization methods were tried.
* The paper contains quite a few results (although some seem to be buried in the appendix?)

### Weaknesses
* It's not entirely clear whether action magnitude really captures multi-step prediction. Increasing the permissable action magnitude also offers more actions that can help the model disentangle the geometry of the environment. Is this really related to multi-step prediction?
* The line and grid environments are quite toyish, and other works have shown that other representation learning methods can disentangle environment geometry without multi-step prediction (see [1], [2] and [3]). These additional works should be discussed in the related works section.
* Some important details are not included in the main paper and discussed in the appendix. This is a bit strange since the paper is pretty short and comfortably below the page limit. In particular, the exposition of the line environment is very dense and difficult to understand. The same goes for the GAN experiment.

In my opinion, this is a good paper, but the exposition is a bit lacking. The experiments with the RNNs in the appendix can also be discussed more, as they offer an alternative setup to multi-step prediction than the one studied in section 3. The related works section also lacks references to past representation learning approaches that are able to learn good environment representations without multi-step prediction (see [1], [2] and [3]). Lastly, it has recently been shown that LLMs can learn representations of graphs in-context. These models have only been trained to perform next-token prediction, but nevertheless adapt to represent graph environments in a manner that captures their underlying structure (see [4] and [5]).
If the authors extend the related works section and make the exposition clearer I would be happy to increase my score to 6.

References:

[1] Watter et al. "Embed to control: A locally linear latent dynamics model for control from raw images." NeurIPS 2015

[2] Saanum et al. "Simplifying latent dynamics with softly state-invariant world models." NeurIPS 2024

[3] Kipf et al. "Contrastive learning of structured world models." ICLR 2020

[4] Park et al. "Iclr: In-context learning of representations." ICLR 2025

[5] Demircan et al. "Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models" ICLR 2025

### Questions
* Is there a reason for using a GAN in this experiment? Do similar representations emerge if you are using an Auto-encoder with a bottleneck to decode the target image label?
* When training the GAN with multi-step and single step actions, do you see differences in training dynamics and performance on the task it was trained to do? From what I can tell, only the representations were evaluated.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper looks at how multi-step predictive coding and using deeper neural networks can help models figure out the low-dimensional structure behind the data they’re observing. The authors start with a simple, theory-friendly setting and then move on to more complex and nonlinear tasks, such as experiments with MNIST digits. They show that when networks are deep enough and asked to predict farther into the future, their internal representations tend to organize themselves in ways that mirror the true structure of the environment. As a disclaimer, I'm definitely not an expert on the topic, so I will place low confidence.

### Strengths
The paper does a good job connecting theory and practical experiments. Starting simple, the authors gradually ramp up to more realistic datasets. Furthermore, the explanations around why deep networks and longer prediction horizons matter are clear and give new insights that go beyond what’s in earlier work, at least according to the author's claim. 

The experiments extend to nonlinear, more natural examples, showing the main ideas hold up in several contexts. The theoretical claims of the authors seem well supported, and correct.

### Weaknesses
It seems to me that most of the results are for simple, linear cases. While results on more complex tasks are shown, it’s not totally clear how robust the main claims are in messier, real-world settings. I would be interested in seeing experiments on slightly more complex tasks.

### Questions
Overall, the paper is well-written and the results are intriguing, but maybe there’s still room for more evidence about what happens in tougher settings or with newer architectures. How come the authors did not test on this setup?

### Soundness
3

### Presentation
2

### Contribution
2
