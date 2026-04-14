# Scalable Mechanistic Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We propose Scalable Mechanistic Neural Network (S-MNN), an enhanced neural network framework designed for scientific machine learning applications involving long temporal sequences. By reformulating the original Mechanistic Neural Network (MNN) (Pervez et al., 2024), we reduce the computational time and space complexities from cubic and quadratic with respect to the sequence length, respectively, to linear. This significant improvement enables efficient modeling of long-term dynamics without sacrificing accuracy or interpretability. Extensive experiments demonstrate that S-MNN matches the original MNN in precision while substantially reducing computational resources. Consequently, S-MNN can drop-in replace the original MNN in applications, providing a practical and efficient tool for integrating mechanistic bottlenecks into neural network models of complex dynamical systems. Source code is available at https://github.com/IST-DASLab/ScalableMNN.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Building on mechanistic neural networks (MNN), this work proposes a strategy to efficiently solve long sequences using S-MNN. It addresses the scalability issues and reduces complexities to linear with 3 key improvements: elimination of slack variables, higher-order smoothness constraints, and least square regression. The result is a framework that can tackle long sequences with up to 4.9x speedup over MNN, and 50% GPU memory reduction, something the original MNN could not do.

### Strengths
Overall, the exposition is crisp, much clearer than the work it builds upon, of which the authors have given a great overview. The more direct formulation allows efficient forward pass, and a backward pass now directly supported by automatic differentiation. That said, the authors propose an analytical form that enables quicker numerical integration. The range of experiments is broad and the results are compelling, showing significant improvements over ordinary dense and sparse MNN.

### Weaknesses
1) The work does not address the trade-offs from the 3 main changes it brings, notably the consequence of dropping the slack variables. For instance, MNN notably allows **nonlinear** ODEs to be integrated with relatively ease. Is there any reason S-MNN doesn't do the same ?
2) Limited evaluation. The original MNN showed improved performance over ResNets and FNOs, but the current method has only been compared to MNN. Particularly, in this same setting, it would be interesting to know how well models outside the MNN family perform on the KdV and long-term sea surface temperature forecasting experiments, at the very least to gauge how difficult the task is.
3) Training and testing. During training, a number of quantities are learned. But during testing, it is not clear whether only the $y$s are solved for, and whether other matrix quantities from Algorithm 1-4 are reused. Appropriate training and testing algorithms should help clarify this.

### Questions
1) The paper mentions the batch size of 512 as a default setting. For the Lorentz system with initial conditions $x=y=z=1$, what makes up an instance in the "batch", especially since small batch and block sizes seems to be a limitation of S-MNN (line 536).
3) The Cholesky decomposition in Eq. 12 hinges on the matrix $M$ being positive definite, which is claimed in line 201. But how is that guaranteed ? The same remark goes for Eq. 28 where the symmetric nature of $M^{-1}$ is implied but not (re)stated.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors provide an alternative method to implement ODE solving in mechanistic neural networks (MNNs) to achieve better time and space complexity without sacrificing performance.

### Strengths
- straightforward and good idea, fuelled by several computational tricks
- strong complexity reduction!
- clear writing
- clear motivation
- experiments on toys to illustrate the point and on real world data to show feasibility

### Weaknesses
- while the description of MNNs is nice and concise, i believe a reader could benefit from a small additional description of the aspects of the network that are improved upon, like the slack variables.
- I am missing some ablations studies, for instance on the importance weights. do they do anything in practice?

### Questions
- I don’t fully understand what determines the sparsity level of $A$. You say 95% in typical applications, which kinds of applications yield denser $A$? IIUC, the sparsity comes from the fact that none of the Eqs 5,6,7,8 entangle $y$ that are more than a timestep apart.

### Nits:
- $V$ is undefined when introducing Eq 1, only defined later
- maybe you can write out the definition of A, y and b once for clarity


### To the other reviewers
I am not an expert in this area, so take my review with a grain of salt.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In their paper, the authors propose an alteration to the recently published Mechanistic Neural Networks (MNN), making the architecture less computationally demanding, while maintaining its predictive power. Their method, called Scalable MNNs (S-MNN), decreases computational cost primarily by removing certain slack variables present in the original method, as well as replacing a quadratic programing problem with a simple least squares regression, which has known analytical solution and can be more efficiently computed.

The authors begin by contextualizing their work among the scientific machine learning literature, then providing a brief overview of MNNs, an architecture aimed at modeling time-series data by learning an ODE in latent space, where the inputs are first encoded to represent the dynamics of the desired ODE. The solution of this ODE is then optionally passed through a decoder to map the solution to the desired output space. The authors then go on to describe their method, S-MNN, which simplifies certain aspects of the original MNN implementation. The main difference between the two methods is the linearization of the underlying ODE, which makes it solvable using a traditional least squares regression, with known analytical solution. The authors also propose a sparse method of computation for obtaining this solution, which is easily parallelizable using GPUs. Overall, their alterations reduce the time and memory complexity of the architecture so that it scales linearly with the sequence length $T$.

Finally, the authors briefly describe other related work in the scientific machine learning literature, and carry out experiments to empirically validate their method. The authors employ S-MNNs on 1) a simple set of ODEs, as a sanity check for the architecture; 2) the discovery of underlying parameters of a Lorenz system; 3) solving the Korteweg-De Vries (KdV) PDE from data; and 4) forecasting sea surface temperatures from the SST-V2 dataset. Overall, the experiments indicate that S-MNNs perform comparable or slightly better than traditional MNNs, but at considerably lower computational cost.

The authors conclude by summarizing their work and outlining limitations and possible future directions of work, including parallelizing their implementation over the time dimension, which currently need to be solved sequentially.

### Strengths
### Originality
This work innovates by proposing a more computationally efficient way of implementing Mechanistic Neural Networks (MNNs), making it more scalable for longer roll-outs and numerically stable. Although the paper largely basis itself on the recent work by Pervez et all., 2024, they improve upon this method in an original way, and present practical ways of making the MNN strategy less computationally expensive.


### Quality
The submission appears to be well thought-out, making precise contributions to the previous work of MNNs. The simplification of the underlying ODE solver implied by the architecture appears to yield similar performance at a computational cost 2-5 times lower, with greater grains when the sequence length $T$ is bigger. The author's computational improvements are substantial, and the theoretical arguments used to formulate and motivate S-MNNs are well crafted.


### Clarity
I was not originally familiar with MNNs, so there are parts of the paper that seemed unclear to me, but I think the authors overall do a decent job of introducing the important concepts and explaining how their alterations lead to computational efficiency.

### Weaknesses
### Contribution Is Specific To A Niche Method
Although the ideas in the paper seem interesting, my main criticism of this submission is that their contribution is very narrow, in the sense that it improves on a method that is generally not very employed in practice. This makes their work less relevant to the scientific machine learning community. While the authors carry out several well-designed experiments comparing MNNs and S-MNNs, it is unclear to me what the advantages of S-MNNs are over the other methods described in section 4 of the paper. Potentially including some of these other methods in the experiments carried out, and comparing their accuracy/computational cost would help contextualize their work and make a stronger case for their approach. I detail some suggestions for comparisons in the "Questions" section below.


### Errors Reported Using Absolute Error Instead of Relative Error (minor weakness)
Errors should ideally be reported as relative L2 error, instead of absolute error squared. Using relative L2 error helps contextualize the performance of the method for target functions of different magnitudes. For Example, in section 5.1 the harmonic damping problem has a solution that oscillates from $\approx -0.1$ to $\approx. 0.1$, while the population problem oscillates from 0 to $\approx 40$. Using relative L2 errors would also make their results more easily comparable against what is reported in other papers for comparable problems.


### Other comments that did not affect my score:
- [line 333] While it is great to see very close agreement between true and learned solutions, it would still be valuable to report exact errors for each benchmark, likely in the appendix.
- [figure 4] Since all predictions look very close to the ground truth solution when eyeballing the plots, it would be great to plot absolute error as well, which could help differentiate the performance of the two methods.

### Questions
I list below my two main suggestions. As previously mentioned, addressing the first point is what is most likely to change my current rating of the paper.

- [**Overall Relevance To Scientific Computing**] Although the authors do list related methods in scientific machine learning in section 4, it is not clear to me why improving on traditional MNNs is desirable over utilizing these other methods. Why choose to focus on MNNs? In order to answer this question, for example, the authors could run their experiments using some of the other methods listed in section 4, listing their accuracy and computational cost when compared to MNNs and S-MNNs. For example, some suggestions for comparisons are: 1) in section 5.1 and 5.2, they could run comparisons against Neural-ODEs and perhaps a traditional neural operator such as FNO or DeepONet; 2) in section 5.3 they could compare against a traditional MLP using both data and a Physics Informed loss (as detailed in line 300); 3) in section 5.4 the could compare against other autoregressive models including a traditional transformer or a neural operator. Doing so would help contextualize their work in a quantifiable way, and could help increase my score of their submission, even if S-MNNs do not beat other all other methods.

- [**Use of Relative L2 Errors**] As mentioned in the previous section, reporting errors using relative L2 error instead of MSE/RMSE provides a more reliable way of comparing methods across different benchmarks, since solutions can have drastically different magnitudes. Is there a particular reason why reporting absolute MSE/RMSE makes more sense for your benchmarks?

### Soundness
4

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
2

### Summary
The paper presents Scalable Mechanistic Neural Network (S-MNN) designed for long temporal sequences.
S-MNN is more efficient compared to a previously proposed solution MNN as it reduces the computational time and space to linear in the sequnce length.
The paper describes the algorithmic improvements to reduce the computational cost and provide experimental results

### Strengths
- The problem is well-motvated 
- S-MNN seems to achieve good performance while improving the computational cost 
- Detailed experimental results are presented

### Weaknesses
As discussed by the authors in the limitation section, parallelism along the time dimension can be challenging.

### Questions
- Are there any cases where intuitively MNN should outperform S-MNN?

### Soundness
3

### Presentation
2

### Contribution
3
