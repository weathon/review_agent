# Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escape, and Network Embedding

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
In this paper, we study the loss landscape of one-hidden-layer neural networks with ReLU-like activation functions trained with the empirical squared loss using gradient descent (GD). We identify the stationary points of such networks, which significantly slow down loss decrease during training. To capture such points while accounting for the non-differentiability of the loss, the stationary points that we study are directional stationary points, rather than other notions like Clarke stationary points. We show that, if a stationary point does not contain "escape neurons", which are defined with first-order conditions, it must be a local minimum. Moreover, for the scalar-output case, the presence of an escape neuron guarantees that the stationary point is not a local minimum. Our results refine the description of the *saddle-to-saddle* training process starting from infinitesimally small (vanishing) initialization for shallow ReLU-like networks: By precluding the saddle escape types that previous works did not rule out, we advance one step closer to a complete picture of the entire dynamics. Moreover, we are also able to fully discuss how network embedding, which is to instantiate a narrower network with a wider network, reshapes the stationary points.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper characterizes the (possibly non-differentiable) stationary points of a one-hidden layer ReLU-like neural networks. By ReLU-like, I mean a piecewise linear activation with a single kink at x = 0. The authors specifically define the notion of a stationary point in their problem of interest, using the notion of directional derivatives. With the definition they are able to show that if a stationary point does not contain "escape neurons" - which is a specific type of neuron, the stationary point is a local minimum. This insight enables the authors to understand saddle-to-saddle dynamics, and network embedding principles with their own perspective, and extend existing results to non-differentiable settings.

### Strengths
The idea of considering a different coordinate system to exploit the fact that the nonlinearity is only non-differentiable at x = 0 is interesting.

### Weaknesses
1. I am not convinced on the definition of stationary points for this paper (Definition 3.3). Here, the authors define stationary points as points that have zero derivative with respect to second-layer weights and radial direction and nonnegative directional derivative with respect to tangent directions. The point I am not convinced is "why are the radial terms and tangential terms considered differently?" - is it because the radial term is differentiable but the tangential term is non-differentiable? 

The definition of stationary points also felt more like a "local minimum" to me (this could possibly be wrong - but my understanding is that the defined saddle point has zero differential at differentiable parameters and local minimum for non-differentiable parameters).

Overall, the definition of stationary points in Definition 3.3 seems to be quite restricted. 

2. Also, it would be good to explicitly mention that the notion of stationary points this paper discusses may be different from the notion from other papers in the literature, at the introduction and in the abstract. For example, there exists papers that completely characterize Clarke stationary points of a one-hidden layer ReLU network [1], so to make the contributions more concise, describing the notion of stationarity could help. 

3. It is unclear to me how their approach is different from using one-sided directional derivatives (e.g. [2]). I think this paper also uses the notion of one-sided differential to mitigate non-differentiability (from the definition of radial and tangential derivative in pg 4)? Probably another major difference is radial/tangential derivatives - but I don't see the necessity of such concepts except when the authors define stationarity.

Overall, it is rather unclear how non-differentiability mitigates an analysis of saddle points, and how the authors actually overcame the difficulty with their idea.

4. Most part of main results are discussions rather than concise theorems or statements. Furthermore, while mentioning saddle points, it is confusing whether the authors are referring to "their notion of saddle points" or the notion of saddle points in the literature.

[1] Wang, Yifei, Jonathan Lacotte, and Mert Pilanci. "The hidden convex optimization landscape of two-layer relu neural networks: an exact characterization of the optimal solutions." arXiv preprint arXiv:2006.05900 (2020).

[2] Cheridito, Patrick, Arnulf Jentzen, and Florian Rossmannek. "Landscape analysis for shallow neural networks: complete classification of critical points for affine target functions." Journal of Nonlinear Science 32.5 (2022): 64.

### Questions
1. Addressing the weaknesses would be good. If I can see that the author's definition of saddle points makes sense, and how they are related to saddle points in other papers, I would raise my score.

2. Some clearer wording is needed: in Lemma 3.2, we have \delta(s_i) > 0 sufficiently small. Is there a clearer way to state this? In pg. 8, could you be more specific what larger initialization scale 8.75 x 10^-4 means? In pg.20, I am not sure if we can justify the existence of higher order directional derivatives just with continuity, because in general, continuity does not imply the existence of directional derivatives.

3. I got curious about the link between existing work and this paper.
 - It is proven that in [3], first-order methods avoids strict saddle points. The intuition in this paper is that strict saddle points have a descent direction where gradient descent falls down. I think this notion is similar to the notion of escape neuron, where we have a certain direction that the loss decreases. Do the two concepts have something in common, or are they intrinsically different?
 - In pg 6, you mention the notion of dead neurons. [2] also uses the term dead neurons - are they related?

[3] Lee, Jason D., et al. "First-order methods almost always avoid strict saddle points." Mathematical programming 176 (2019): 311-337.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors study the dynamics of shallow ReLU-like neural networks. The analysis starts with the definition of stationary points, saddle points, and two types of local minima for ReLU-like networks, whose loss landscape is characterized by non-differentiability confined in hyper-planes. Then, the concept of an escape neuron is defined, whose existence indicates the possibility of escape. Numerical experiment with synthesized data and vanishing initialization motivates the theory that saddle escaping is driven by small living neurons gaining amplitude following alignment. Further experiments show the impact of small but finite initialization. Last but not least, the condition under which embedding preserves stationarity is given in the non-smooth setting.

### Strengths
The authors developed their own notation based on direction derivatives to tame the non-differentiability in ReLU-like networks, allowing them to study the saddle-to-saddle dynamics in the non-differentiable landscape. With theoretical argument and numerical experiment, the authors manage to explain the whole trajectory when training a shallow ReLU-like network with vanishing initialization. Although not every argument in the paper is original, I am happy to see a paper that collects a lot of existing results and presents them in an integrated manner. The authors put a lot of effort into making their result understandable.

### Weaknesses
There are constraints for the theoretical analysis, and the experiments are performed on synthesized data instead of real data. Also, I find the definition of stationary point a little counter-intuitive. I have the impression that the definition of stationary point in this work is not exactly the extension of stationary point of smooth function, which is where $f’(x)=0$ regardless of trapping GD or not. It appears to me that Clarke subdifferential the authors mentioned could be better in this regard, in the sense that the origin of both $|x|$ and $-|x|$ are considered to be a stationary point. With the authors’ definition of stationary point, I can imagine that local maxima are usually not stationary points.

### Questions
Although I appreciate the experiment in section 4.1.1 very much and agree that it supports the authors’ theory of saddle escaping, I don’t know if the evidence is strong enough to rule out all alternative theories (discovered or not).

Is there any specific reason for choosing $x_k$ and $y_k$ as given in the paper in section 4.1.1?

Line 453/454 should the “does get” be “doesn’t get”?

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
This paper presents a new definition of stationary point for non-differentiable neural nets caused by ReLU-type activations. It characterizes the relationship between the local minimum and the "escape neuron", giving a sufficient and necessary condition for local minima under scalar output. Based on this notion, the authors provide a new perspective on saddle-to-saddle dynamics and show that escape from a saddle must perturb the escape neuron. Furthermore, the authors investigate the effect of unit replication on the stationary point.

### Strengths
The proposed definition is novel and interesting, and it provides a more complete characterization of saddle points in two-layer ReLU neural networks. This new definition can help to deepen our understanding of the loss landscape and training dynamics.

### Weaknesses
1. The current studies focus on the two-layer ReLU neural network and it is unclear how the result can be generalized to deep networks, especially since the characteristic of escape neurons can be complicated when the network is deep, it is unclear if we can draw meaningful conclusions there using the proposed notion. 

2. The current study focuses on a non-differentiable activation function only, and the conclusion will be trivial if using a differentiable activation function. As more and more smooth activations are used such as GELU and ELU, the result will have less meaningful impact on practice. 

3. Fact 4.6 is an important conclusion to discuss the dynamics, but it was provided as an empirical observation. As this paper mainly focuses on theoretical characterization, it would increase the rigor of the analysis by providing a theoretical justification.

### Questions
1. The escape neuron condition requires that the output weight of a specific neuron be zero. Does that imply the landscape of scalar output  2-layer neural network is rather trivial? It seems to be sufficient to just perturb the weight of this specific neuron to escape the saddle. 

2. Although the authors argue that the sub-differentiable definition is unsuitable for a non-convex function, does it also fail specifically in two-layer neural networks? In particular, it would be helpful to provide an example in the ReLU network to show that the sub-gradient can not properly capture the saddle point. Since the proposed analysis is restricted to the two-layer neural nets, it would be helpful to understand the difference specifically in this setting.

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
3

### Summary
The paper studies the loss landscape of one hidden layer with ReLU-like activations trained on squared loss. It provides conditions for stationary points that work for both differentiable and non-differentiable parts of the loss landscape. The authors introduce "escape neurons" as the indicator of whether stationary points are local minimums. This conclusion is necessary and sufficient for scalar-output nets. With this characterization, they take a step further to understand the saddle-to-saddle dynamics of two-layer networks. They also discuss how network embedding helps reshape the stationary points.

### Strengths
- The ReLU networks' stationary points analysis is novel and interesting. The authors are able to characterize the "escape neurons" as the indicator of two-layer ReLU networks defined with first-order conditions, which can help to characterize whether a stationary point is a strict saddle to escape. This rigorous analysis can help us further understand the saddle-to-saddle dynamics from a landscape perspective. 

- The toy experiments verify the theoretical results above with vanishing initialization, both displaying examples for stationary points without escape direction and saddle-escaping phenomena when "escape neurons" exist. The explanation of Figure 4 is intuitive and insightful.

- The escape neuron analysis helps to understand how the interesting "network embedding" technique reshapes the landscape.

### Weaknesses
- The analysis for the landscape seems to be a bit limited to the local stationary point analysis as the supplement of [Kumar and Haupt, 2024], precluding other types of saddles. To understand the saddle escaping or saddle-to-saddle dynamics, it would be better to more analyses of gradient dynamics in the neighborhood of saddle points. For example, can we have the escape direction in Corollary 4.4 related to the gradient, implying that GF/GD can help escape the saddle points?

- The discussion of the saddle-to-saddle dynamics with vanishing/finitely small initialization (sec 4.2) looks too heuristic to me. It is better to present those implications or corollaries of the stationary point characterization more formally instead of descriptions like Consequence 4.7.  

- As a minor weakness, the toy examples only have one-dimensional inputs (with bias), which is not comprehensive. More empirical results need to be shown to corroborate the theoretical results' implications.

### Questions
Is it possible to come up with a more rigorous example (with gradient dynamics analysis) for Figure 4 to understand the saddle-to-saddle dynamics?

### Soundness
3

### Presentation
3

### Contribution
3
