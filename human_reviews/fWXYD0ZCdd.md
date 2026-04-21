# A New Look at Low-Rank Recurrent Neural Networks

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
Low-rank recurrent neural networks (RNNs) have recently gained prominence as a framework for understanding how neural systems solve complex cognitive tasks. However, fitting and interpreting these networks remains an important open problem.
Here we address this challenge using a perspective from the ``neural engineering framework'', which shows how to embed an arbitrary ordinary differential equation (ODE) into a low-rank RNN using least-squares regression. Under this perspective, individual neurons in a low-rank RNN provide nonlinear basis functions for representing an ODE of interest. This clarifies limits on the expressivity of low-rank RNNs, such as the fact that with a $\tanh$ non-linearity they can only capture odd-symmetric functions in the absence of per neuron inputs or biases. Building on this framework, we propose a method for finding the smallest low-rank RNN to implement a given dynamical system using a variant of orthogonal matching pursuit. We also show how to use regression-based fitting to obtain low-rank RNNs with time-varying dynamics. This allows for the rapid training of vastly different dynamical systems that nevertheless produce a given time-varying trajectory. Finally, we highlight the usefulness of our framework by comparing to RNNs trained using backprop-through-time on neuroscience-inspired tasks, showing that our method achieves faster and more accurate learning with smaller networks than gradient-based training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a novel perspective by viewing low-rank RNNs as parameterized ordinary differential equations (ODEs) with nonlinear basis functions. Concretely, given a generic RNN $\dot{x} =  -x +J\phi(x) + I u(t)$ and $z = W\phi(x)$ and assuming that $rank(J) = dim(z)$ (i.e. a low-rank RNN for low dim. output $J=MN^T$).  These dynamics can be equivalently represented as a neural ODE with a single hidden layer (of $dim(x)$) and output dynamics $\dot{z} = -z + n^T \phi(Mz_t + Iv_t)$. The authors then consider the problem to fit $\dot{z} = g(z)$ to a *known* output dynamics $g(z)$. This view allows low-rank RNNs to be fitted using generalized linear regression on $n$ given a fixed set of nonlinear basis functions defined through $M$ and inputs $I,v$. The ability to fit certain dynamics is then limited by the expressivity of the basis functions. The authors propose a greedy method to select a suitable basis based on an orthogonal matching pursuit framework (from a predefined set of random basis functions obtained through randomly sampling M,I).

### Strengths
The paper is well-written, sound, and easy to follow. The authors provide examples of how this interpretation/method can be used to, e.g., analyze the expressivity of low-rank RNNs and other properties. The proposed method is efficient and easy to apply to given dynamics, and the paper effectively demonstrates the utility of generating new insights on the expressivity of low-rank RNNs.

### Weaknesses
- **Training RNNS**: The manuscript claims to "address the issues of RNN training," but the method effectively only "maps" known low dimensional dynamics to high dimensional recurrent neural activity. These are different problems, aren't they? (If I know the low dimensional dynamics, why would I ever use BPTT to embed it into higher dimensional space?). I find the comparison to Backpropagation Through Time (BPTT) somewhat strange. RNNs trained with BPTT effectively address a different problem: learning dynamics from data points. (at least, I assume this is how the RNNs in Fig 5 are trained, as no details are provided). Isn't it expected that they perform worse in your examples because they have to figure out the right dynamics from the data, while your method just fits to the known "true" dynamics? 

- **Novelty**: Both the neural ODE interpretation and fitting method using linear regression are not new (e.g., see Beiran et al. 2021, which also uses least squares regression to fit RNNs to known dynamics). While the authors do cite this related literature, these methodological similarities should be more clearly stated in the manuscript.

- **Limitations**: The method does require a known dynamical system to fit i.e. $g(z)$. This is a strong assumption and limits the applicability of the method.  I am unfamiliar with the neuroscience literature, but this is rarely the case, no? The authors should clearly discuss this limitation in more detail (earlier in the manuscript) and add some motivation on relevance (add some more details about the NFE?).

### Questions
- Can the authors provide motivation or introduction to the NFE framework?
- Eq. 11 is missing the residual, no?
- Regarding the identified weaknesses, I am particularly concerned about how RNN training is framed in the current manuscript, given that the approach appears to have limited applicability for training and is mainly suited for analysis (see weaknesses).

Overall, I hence tend to reject this paper. Given that I am not an expert on the relevance of this contribution to the neuroscience topic, I will only vote for a weak rejection.

### Soundness
3

### Presentation
3

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
Rank-$R$ RNNs are universal approximations of arbitrary $R$-dimensional dynamical systems that are (typically) trained with back propagation. The NEF conversely approximates dynamical systems with known equations, using RNNs with low-rank structure, by solving a least-squares minimisation problem. In this contribution low-rank RNNs are fitted to known flow-fields by solving a NEF style optimisation problem, with some focus on obtaining networks with small amount of units.

### Strengths
- The work aims to connect some existing lines of work. While I am of the opinion that much of this is was introduced in past work (see weaknesses below), writing out connections explicitly can be useful.
- A new method is introduced for fitting small (in number of units) low-rank RNNs to neural data. 
- The paper reads well and the figures do a good job of illustrating the points made.

### Weaknesses
My main concern with this paper is that it presents multiple results as new, even though they might not be — sometimes with statements that appear to be incorrect. Given that the paper has “a new look” in the title, I expected more novel insights. I think some rewriting and rephrasing is important, after which this can paper could turn in a useful and nice overview of recent lines of work with some new ideas.

--------

1. Non-linear recurrent dynamics are commonly used in the NEF

> 
(Line 073) while the NEF has been widely explored, most studies have focused on feed forward-networks, comparatively few studies focused on non-linear recurrent dynamics

This statement seems false, the NEF is used to embed (non-linear) low-D functions in high-D networks, which can be (and are often) straightforwardly made recurrent, for instance when used for integration / memory, or pattern generation [1]. Most of the online tutorials using the python implementation of NEF (nengo) also deal with time-varying dynamical systems, e.g., non-linear oscillators or the chaotic Lorenz attractor: https://www.nengo.ai/nengo/examples.html.

--------

2. On fitting low-rank RNNs to flow-fields with regression

  - 2.1 The NEF can (and is) also straightforwardly used with rate neurons [1], at which point it becomes at least very close to your framework (it solves the same regression problem). While the NEF crowd typically doesn’t explicitly call their models low-rank RNNs, is there any significant difference with your framework besides the naming?

  - 2.2 In Ref [2] low-rank RNNs were also fit with linear regression to known flow-fields (Section 6). While the parameterisation is slightly different, the general idea is very much the same (and there also the connection to NEF was pointed out). 
--------

3. On new insights into universality

Ref [2] also shows that low-rank RNNs with inputs / biases are universal approximations (including for exactly the same system as is used in this paper). It also includes many results relating to symmetry when $\tanh$ is used. 

--------

4. On new insight into input driven dynamics

Ref [3] Fig. 1, shows very similar example flow-fields to Fig. 4 in this paper, and used them to make the almost the same point about identifiability. What is exactly the new insight here compared to Ref [3]?

--------

5.  On Low-rank RNNs as neural ODEs

The fact that Eq. 7 is a neural ODE with one hidden layer was explicitly pointed out in previous work (e.g., in the discussion of [4]). The additional insight of the connection to neural ODEs, the literature of which largely focuses on adjoint methods, is in any case also not completely clear to me (some insights by using the adjoint method in relation to low-rank RNNs were obtained in Ref. [5]). I think the main (known) observation to be made here is that Eq. 7 is a universal approximation with one hidden layer.

--------

6. On non-linear oscillators

> (
Line 274) Note that this 2D system is highly nonlinear—unlike recent work focused on oscillations in linear dynamical systems

This statement is misleading. A quick search should give one many studies that investigated oscillations / limit cycles in non-linear RNNs (including low-rank ones!). Ref [2] also derived a limit cycle oscillator in low-rank RNNs.


--------

Refs

[1] Stewart, 2012. A Technical Overview of the Neural Engineering Framework

[2] Beiran et al., 2021. Shaping Dynamics With Multiple Populations in Low-Rank Recurrent Networks

[3] Galgali et al., 2023. Residual dynamics resolves recurrent contributions to neural computation

[4] Pals et al., 2024. Inferring stochastic low-rank recurrent neural networks from neural data

[5] Pellegrino et al., 2023. Low Tensor Rank Learning of Neural Dynamics

### Questions
1. Is the introduced method for finding small RNNs guaranteed to converge to the smallest RNN? Or can it converge to a local optimum?

--------

2. 
> (Line 277) This target ODE is not radially odd-symmetric, so once again (although not shown here), embedding the system in a low-rank RNN fails if we do not include inputs.


It is unclear to me how to reconcile the statement with ref [2], where a rank-2 RNN with $\tanh$ units (without biases) was derived such that it implements a similar limit cycle. 
One answer could it be that (unlike stated in the text) the system actually is odd-symmetric? 
From a quick try, converting your system to back to cartesian coordinates using $z_1 = r\cos(\theta)$, $z_2 = r\sin(\theta)$, I get: $\dot{z}_1 = az_1-z_2$, $\dot{z}_2 = az_2+z_1$, with $a=\frac{1-r^2}{r}$, which satisfies $F(-z_1,-z_2) = - F(z_1,z_2)$.

### Soundness
3

### Presentation
2

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
The authors make use of Neural Engineering Framework to present a different perspective on how to embed a given dynamical system into a low-rank RNN. They put emphasis on neuron-specific inputs or bias to make the low-rank RNNs universal approximators. The paper claims concerns on the fixed-point analyse framework of RNNs which is one of the mainstream type of reverse-engineering method in computational neuroscience.

### Strengths
The provided perspective of latent space view is beautiful and shows connections between neural ODEs and low-rank RNNs.

The paper demonstrates clear examples of indistinguishable dynamical systems.

Proposed learning algorithm is fast and efficient.

### Weaknesses
First, the problem that is concerned is to embed a _given_ dynamical system to a low-rank RNN. However, this approach, as also acknowledged by the authors, limits the applications. I believe this restriction is big and should be discussed by authors more. Currently, I do not think it is addressed enough since it is one of the *main* issues of the paper. I think it would be better to discuss this earlier. It is clear that this method cannot be used to find the dynamics required for a novel task --which is arguably, one of the most important use-cases of RNNs. So it can be only used for the tasks that we already know (or at least have hypotheses) its dynamical system. I would expect from authors to have more (and more importantly, distinct) use-cases of their algorithm for science purposes.

Second, I do not believe comparing your algorithm with BPTT is fair, because you do not provide what BPTT provides (see above). Plus, I couldn't see the details of how you train the two tasks using BPTT. Is there a clear way of embedding a given dynamical system into the RNN using BPTT? If so, I think if you do this to compare your algorithm with BPTT, it would be better. If not, I think this can be put into supplementary results but I don't think it is important.

Third, you state "A low-rank RNN is therefore not a universal approximator unless it has inputs, or equivalently, different biases or offsets to each neuron." I think this is an odd point to make. Because,  1-) An RNN is only a universal approximator when it has inputs or different biases. 2-) A low-rank RNN is strictly a subset of RNN. And so therefore, if an RNN does not have a property under some conditions, and so low-rank RNNs are. So I do not understand the point made by authors. You emphasize this point in your abstract as well, saying, "our perspective clarifies limits on the expressivity of low-rank RNNs", which I think is misleading, because it is not a problem of low-rank RNNs but RNNs in general. 

Fourth, I liked the Figure 4, but I don't think this analyses could only done using your perspective. I think the field already knows the point you make, but you make your point nicely so it is still important. So you say "Our discussion on non-autonomous dynamics suggests, how similar trajectories can be observed in the presence of dynamical fixed points/ no fixed points at all." but does this analysis allowed by your algorithm or is this a distinct point made in your paper besides the provided algorithm? Moreover, you can transform non-autonomous RNN into an autonomous one. This is a very important point which should introduce philosophical questions on your point from the neuroscience point of view. 

I think before introducing the Eq 9, you can talk (very briefly) about least-squares regression. 

You put an introduction to low-rank RNNs but not NEF. I think the paper would benefit a lot if you also put emphasis on NEF more (mathematically) to make your connection more clear.

Typos (did not affect my score):
Line 097, ReLu -> ReLU
Line 242 phi(x)
Line 715, "we thus use present a general"
Line 777, "(A) Bi-Stable Attractor ODE and (B) Line Attractor ODE." unusual capital usage
Line 796, M is m1, m2, m2, and better to put space between them.
Line 801, "Following our discussion o the"
Line 812, "Data Simulated from trajectories", didn't understand the choice of capitals
When you write Lorenz, you should use capital L.

### Questions
What would be a way of finding the smallest RNN using BPTT? Does your algorithm makes it faster, or it provides a theoretical way of doing it as well, can you put more discussions on this?

You wrote an algorithm section for finding the smallest RNN, can you also do it for your main algorithm as well? I think this would clarify a lot.

Under which conditions low-rank RNNs are equivalent to neural-ODEs? If the latent space has noise, are they still equivalent? I would expect a more rigorous way of stating this. 

You say "(although not shown here) ... fails", I wondered how does it look like when it fails.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper advances the understanding of low-rank RNNs by introducing a new theoretical framework, an efficient training approach, and empirical validation. These contributions improve both the interpretability and representational capacity of low-rank RNNs, demonstrating their ability to embed complex nonlinear dynamical systems through a randomized basis. The authors emphasize the importance of inputs in capturing odd-symmetric functions, expanding on prior research regarding the universal function approximation properties of these models. Also, their method enables closed-form parameter learning through regression, which greatly reduces the number of parameters compared to traditional methods. Altogether, the paper positions low-rank RNNs as an efficient and interpretable alternative to conventional training techniques.

### Strengths
**Originality**: The paper introduces a new perspective on low-rank RNNs by modeling them as low-dimensional ODEs using nonlinear basis functions. This approach links low-rank RNNs to the NEF, which has mostly been used for feed-forward networks. The authors present a more interpretable and efficient training method, avoiding the limitations of traditional gradient-based training.

**Significance**: The significance of the paper lies, particularly, in its potential to address the challenges of training and interpreting RNNs.

### Weaknesses
1) The proposed method is not clear enough. For instance:

- **Rank r**. It is unclear how one can systematically determine the optimal rank r for low-rank RNNs in general. It might be easier achievable for benchmark systems with specific trajectories or dynamics, but it is not clear enough for more complicated cases or real-world datasets with higher dimensions. Could you provide guidelines or heuristics for selecting **the rank r** for different types of problems or datasets?

- **Role of different activation functions**. Given the impact of different activation functions ϕ on network performance, what criteria should be used to select the most appropriate activation function for a given task? Are there specific tasks where certain functions consistently outperform others? Could the authors include a comparative analysis of different activation functions on a set of benchmark tasks, showing how performance varies across functions?

- **Probability distribution of the parameters**. Choosing an appropriate probability distribution for the parameters that define the random basis is crucial, as it may directly impact the model's expressivity, learning efficiency, and the stability/convergence of the training process.  Could the authors conduct an ablation study comparing different probability distributions for parameter sampling? Or, is there a data-driven sampling scheme to generate the random basis (i.e., a method for adapting the sampling distribution based on the characteristics of the dataset or task, e.g., similar to https://arxiv.org/abs/2410.23467)?  

2) The paper claims that the proposed method converges faster than traditional BPTT methods, but it lacks a detailed analysis of computational efficiency, including **training time**. Could the authors provide a more detailed analysis of the actual training times across different models and tasks?

3) Regarding the Lorenz attractor, in *Figure SI-3*, the plots do not allow for a clear comparison between the true and fitted (reconstructed test) trajectories. Could you include additional plots for comparing the true and fitted **chaotic trajectories**, as well as the associated **time series** of the x, y, and z components?

4) The paper mentions the **interpretability** of low-rank RNNs but does not delve deeply into it. Could the authors provide more explanations about the interpretability of their method compared to other methods?

### Questions
What additional experiments could be conducted to further validate these findings, particularly in real-world applications?

Please see also "Weaknesses".

### Soundness
2

### Presentation
2

### Contribution
3
