# Who Said Neural Networks Aren't Linear?

- Avg Score: 3.50
- Decision: Reject
- Scores: 0, 6, 4, 4

## Abstract
Neural networks are famously nonlinear. However, linearity is defined relative to a pair of vector spaces, $f$$:$$\mathcal{X}$$\to$$\mathcal{Y}$. Is it possible to identify a pair of non-standard vector spaces for which a conventionally nonlinear function is, in fact, linear? This paper introduces a method that makes such vector spaces explicit by construction. We find that if we sandwich a linear operator $A$ between two invertible neural networks, $f(x)=g_y^{-1}(A g_x(x))$, then the corresponding vector spaces $\mathcal{X}$ and $\mathcal{Y}$ are induced by newly defined addition and scaling actions derived from $g_x$ and $g_y$. We term this kind of architecture a Linearizer. This framework makes the entire arsenal of linear algebra, including SVD, pseudo-inverse, orthogonal projection and more, applicable to nonlinear mappings. Furthermore, we show that the composition of two Linearizers that share a neural network is also a Linearizer. We leverage this property and demonstrate that training diffusion models using our architecture makes the hundreds of sampling steps collapse into a single step. We further utilize our framework to enforce 
idempotency (i.e. $f(f(x))=f(x)$) on networks leading to a globally projective generative model and to demonstrate modular style transfer.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors reinvent a classical mathematical technique which I know under the name transport of structure: https://en.wikipedia.org/wiki/Transport_of_structure. This method typically an exercise when vector spaces are introduced.

### Strengths
There are experiments that seem to show some practical value of the approach. The paper is also easy to read.

### Weaknesses
Theoretically, the whole approach is beyond trivial. Saying that this method is well known would be an understatement of epic proportions. 
This idea is quite literally in every introduction course to linear algebra and functional analysis. It is central in topology,  and geometry. This induced vector space is the whole reason tangent spaces are a thing, hence all of differential geometry is based on it. It is absolutely inconceivable to me that the authors have never come across transport of structure before.

### Questions
I am sorry, I do not have any questions.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies maps that are the composition $g_y^{-1}\circ A\circ g_x:X\to Y$ of an invertible neural network $g_x$, a linear map $A$ and an inverse neural network $g^{-1}_y,$ where the neural networks depend only on the spaces $X$ and $Y$.  In other words, it studies maps from $X$ to $Y$ that can be represented as linear map after the spaces are transformed in a suitable non-linear  way. This makes it possible to bring several tools of linear algebra in a new way to machine learning.  For example  projections, semigroups, singular value decomposition are then formulated for potentially non-linear operators. This is an interesting idea. The paper can be consider also a technically simple way (not in the sense that it is trivial, but easily implementable) to do manifold learning and learn maps between manifolds.

### Strengths
The paper has a clear idea that is well developed. The question when maps or families of maps can be coded or represented as linear maps is simple but the idea is developed in creative and quite deep way. The paper is very well written.

### Weaknesses
The families of the maps which can be represented in the form  $g_y^{-1}\circ A\circ g_x$ may be quite small. In fact, the authors discuss this well and give examples of maps for which their method does not work. However, the authors give also several interesting examples of problems to which they can apply their architecture and this shows that their results are widely applicable.

### Questions
1. What you mean by  "space", can it be it a finite or infinite dimensional vector space, an affine space, or a topological space. If can can be an infinite dimensional vector space, what is its topology? 

2. Are $g_x$ and $g_y$ and their inverse maps continuous?

3. When $g_x$ and $g_y$ are represented as neural networks or neural operators, what kind of architecture you can use to ensure that  these maps are bijections?

4. Do you need uniform continuity of the inverses of the maps $g_x$ and $g_y$ to make the $\oplus$ and $\odot$ operations continuous?

5. On Lemma 3: Is the function $g_y$  fixed in the space $Y$ when you consider families of functions $X\to Y$?

6. In formula (14), is the norm the $g$-norm?

7. Do we have any universal approximation results (or approximation results in some restricted class of functions or families of functions) for the linearizers?

8. In general, please explain in more detail what functions in the linearizer are trained. For example if you consider one function from $X$ to $Y$, do you find all the maps $g_x$, $g_y$ and the linear operator
$A$ by training? How training is done when you have a family of functions between the same spaces $X$ and $Y$?

9. In formula (15), what are the properties of  the function $f(x,t)$?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a methodology for linearizing nonlinear neural networks by applying linear space theory. First, it introduces an algebra (vector operations and a metric) that endows curved spaces with the structure of inner product spaces using invertible neural transformations. Using this, it treats neural networks as linear mappings on curved vector spaces transformed by these invertible networks. As applications, this paper proposes one-step inference via flow-matching, interpolation for style transfer, and the construction of idempotent networks.

### Strengths
Several studies have explored deep learning models that perform linear operations in latent spaces, such as Mixup and state-space models. However, few have provided a theoretical framework for such operations. This paper offers a theoretical framework that reframes neural networks as linear mappings in curved spaces. To my knowledge, no other paper has framed neural networks in this manner.

### Weaknesses
1. As the authors point out, the proposed linearizable neural networks have limitations in their expressive power. Therefore, I question whether they are theoretically useful architectures.
2. The experiments are limited to generation tasks on relatively small image datasets, such as MNIST and CelebA. Therefore, their significance is limited from the perspective of practical usefulness.
3. I would like more clarification on the benefit of reframing deep learning models as linear mappings in curved spaces. If all neural networks could be interpreted this way, it would be convincing. However, as this paper itself points out, this is not the case. Therefore, it seems more reasonable to view this paper as proposing a specific type of architecture that combines invertible networks with linear mappings.

### Questions
See the Weaknesses section, particularly the third point.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes to use invertible neural networks as non linear coordinate transformations of vector spaces R^n and to consider the linear structure within one such coordinate representation, leading to a non classical linear structure in the other and to a non-linear structure w.r.t. the standard operations on R^n. The authors thereafter consider the linear operations with respect to the deformed addition and scalar multiiplication. When seeing the results from the angle of non linear coordinates of a vector space, the results of the given Lemma are pretty straight forward, but valuable for less mathematically oriented readers. 
Interestingly, the authors can then employ their linearization approach in learning image synthesis style transfer. The FM generation paths that usually require the solution of a non linear ODE can, in the linearized representation be represented by a learned linear map that is obtained as the time integral of a linear vector field A_t v which is represented by a neural network. When the solution operator to A_t is computed once, this can be used for a essentially lossless one step generation.
The authors demonstrate the viability of their approach for MNIST and Celebrity Faces image generation tasks and neural style transfer. The results are decent and show the potential of the author's point of view.

### Strengths
* the paper provides a really original idea to consider invertible neural networks in vector spaces and then build machine learning tasks on linear methods in the constructed latent space
* The authors show that this approach can be combined with generative learning tasks in the image domain, in particular that the time integration of the FM vector field can be effectively represented by a linear solution operator obtained as the time integration of (time dependent) linear vector fields.
* The theoretical documentation is rather complete and easy to follow

### Weaknesses
- The core idea of the paper - looking at vector spaces in non linear coordinates - could be presented somewhat clearer. Many results given in the lemma would then just be clear and in parts superfluous (like Lemma 4). Also all the linear algebra checked in detail would follow from one structural argument. 
- Better visualizations would be really helpful for a better representation of the main idea. 
- For me the missing part is a detailed investigation of the interaction of the chosen 'coordinate maps' g with the learning task. Obviously not any g would do (e.g. not the identity or linear g's).
- Related to this, I would appreciate a much more extensive analysis of the training details, especially with regard to the previous point.
- Also the interaction between the learning task and the expressivity of the linearizers would be of interest. Is the reproduction of the FM time integration with a linear flow compatible with other approaches that do not have as 'stiff' vector fields as FM - take e.g. a likelihood based trained neuralODE. Similar questions arise for the style transfer task.
- The title, due to its generality, somehow promises more than is kept by the rather specific application domain.
- The authors address flow matching as diffusion. As an ODE based method it is not a diffusion process, unlike SDE based DDPM. In a mathematically based paper, such inaccuracies should be avoided.

### Questions
- Can you give a detailed and reproducible account on how the coordinate maps g are obtained in the respective application context?

### Soundness
3

### Presentation
2

### Contribution
4
