# A Diffusion Model Induced by MSE Training

- Decision: Reject
- Scores: 4, 0, 2

## Abstract
In the DDPM paper, Ho, Jain, and Abbeel introduced two reversible diffusion processes parameterized by a noise schedule—a generator and an oracle process that the generator learns from—and derived a formula for the Kullback-Leibler divergence (KL) in the form of a time-weighted Mean Squared Error (MSE). However, they empirically found that omitting the weights improved performance on image-synthesis benchmarks, a result later corroborated by many studies. More recently, removing the stochastic component at generation time has proved effective. (1) In this work, we provide a theoretical justification for these practices. We consider a broader class of diffusion processes (not necessarily reversible) parameterized by a noise schedule and a diffusion size b that share the same marginals. Since the weight associated with the MSE depends on b, omitting the weight is equivalent to solving the equation weight(b)=1, which yields a unique "MSE-diffusion". For SOTA models, we checked that b is close to zero; that is, the learned MSE-diffusion is nearly a flow, and we confirm this observation by comparing generators on ImageNet 512×512. 
Therefore, flows beat reversible diffusions because training of SOTA models is an implementation of KL minimization for MSE-diffusions, which are nearly  flows. The models that succeed are the ones that are really trained. (2) Moreover, by generalizing the diffusion process to both discrete and continuous time, we obtained a novel representation of the diffusion state as the sum of an explicit linear component, an unweighted pathwise integral of the denoiser, and a noise term. This representation offers the advantages of DPM-solvers while enabling the use of classical numerical methods for ODEs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a common discrepancy in diffusion models where the training process (defined by an MSE objective) differs from the generation process. 

The authors propose a "principle of coherence" to align them, deriving a closed-form expression for the generative diffusion schedule based on the specific noise schedule and network parameterization used during training. This "MSE-induced diffusion" is intended to be the process the denoiser actually learned , and the paper also provides a new state representation to simplify numerical integration.

### Strengths
1. The paper has a clear motivation, it highlights a common mismatch in diffusion models where the denoiser is trained using one implicit process defined by an MSE objective, noise schedule, and network parameterization but is then used for generation with a different process, e.g., one with an ML-optimal diffusion schedule or a deterministic flow

2. The work derives a closed-form expression for the diffusion schedule. This means the generative process isn't an arbitrary choice but is analytically determined by the noise schedule and network parameterization used during training. This is called "MSE-induced diffusion"

### Weaknesses
The most important issue is that the article lacks practical evidence

1. The paper is entirely theoretical. It proposes the MSE-induced diffusion process based on a coherence principle , but it presents no experiments to demonstrate that this new process is stable, effective, or produces better results than the "incoherent" methods it critiques

2. The paper claims its new state representation e.g. "makes it straightforward to apply classical numerical integration methods" and clarifies the relation to DPM-solvers. It even alludes to the potential for new solvers like a 4th-order Runge-Kutta. However, it does not actually implement or test any such solver, so these practical benefits remain hypothetical.

3. The primary evidence for the theory's validity is Figure 1. This figure merely shows a visual similarity between the shape of derived schedule and schedules found empirically in other recent work, e.g., empPML, Discount. This correlation is interesting but does not prove that the coherence principle is the reason for that shape or that the resulting process is effective.

4. The paper critiques widely used deterministic flows as being incoherent. It then proposes its stochastic coherent process as a better alternative. However, it doesn't sufficiently justify why coherence is inherently superior to the fast and high-performing deterministic flows, largely assuming that aligning the training and sampling processes is axiomatically beneficial

### Questions
I hope the author can provide enough evidence to prove the effectiveness of the methods and theories. Especially the followings:

1. The core claim is that **it is beneficial to generate samples using the very process that is actually learned**. This is a compelling, testable hypothesis. However, the paper is entirely theoretical, and its primary evidence (Figure 1) is correlational .
Could the authors provide any preliminary empirical results (e.g., FID scores, sample quality on a standard benchmark like CIFAR-10) to support this central hypothesis?

2. The paper makes strong, practical claims about its new state representation (Eq. 12 ), suggesting it **makes it straightforward to apply classical numerical integration methods** and potentially enables new solvers, like an RK4 analogue, which are currently difficult for DPM-solvers. Have the authors performed any proof-of-concept implementations of a numerical solver using this new unweighted pathwise integral formulation?

3. The paper critiques widely used deterministic flows where $\lambda_t' \rightarrow 0$ and explicitly contrasts them with its proposed MSE-induced process, which is stochastic. Given the demonstrated speed and high performance of incoherent deterministic samplers, what is the specific benefit of adhering to the coherence principle?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a principle of coherence between the training and generation process of the diffusion model: if a denoiser is trained with time-weighted MSE under a given noise schedule and parameterization, then the generation process should use a matching diffusion schedule derived in closed form in the paper. The authors analyze both discrete- and continuous-time settings via simple autoregressive arguments and introduce a state representation that makes connection to classic ODE solvers straightforward.

### Strengths
The paper derives an analytical formula for the diffusion schedule given the training noise schedule and network parameterization.

### Weaknesses
The paper is predominantly analytical and offers only extremely light empirical glimpse—mainly a figure comparing shapes of diffusion schedules under various noise/parameterization choices. The idea behind analysis is not deep enough, the analysis itself is not mathematically challenging. Although the work argues its state representation makes classic ODE solvers like RK4 straightforward to use, it does not demonstrate numerical benefits versus modern diffusion solvers. Overall the paper lacks a clear result and falls well below the standard of ICLR.

### Questions
None

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework called “MSE-Induced Coherent Diffusion.” Starting from weighted MSE training (given a noise schedule and parameterization), the authors introduce an “MSE–ML Coherence Principle,” which provides a closed-form expression for the generative diffusion rate (Proposition 3). This leads to a generative process theoretically consistent with the training objective. They also reformulate the diffusion state as a combination of a linear term, an unweighted path integral, and a noise term, enabling direct use of standard ODE solvers (e.g., RK4). The paper argues that in practice, researchers often “train one schedule but sample with another” (e.g., ML-optimal or deterministic zero-diffusion flows), and advocates using a generative process coherent with the training objective, providing closed-form and discrete formulations consistent with recent empirical practices.

### Strengths
**Theoretical Novelty – The Coherence Principle.**

A major ambiguity in diffusion models lies in the disconnect between training and inference processes. The authors attempt to resolve this by introducing the Coherence Principle, which asserts that the empirical MSE loss and the theoretical ML objective should be proportional over any time interval $[t_0, t]$. This is a novel and well-defined (strong) theoretical assumption.

### Weaknesses
1. Unconvincing Motivation.

The authors claim to eliminate inconsistency between training and inference noise schedules, but this is unnecessary. The core idea of diffusion models (e.g., DDPM, VP-SDE, Rectified Flow) is distribution matching between the forward and reverse processes. As long as the marginal distributions match, the generative model is valid. There are infinitely many possible paths that share the same marginal distributions—for example, infinitely many SDEs corresponding to the same VP-SDE marginals, or an ODE with equivalent marginals. Once the reverse parameter (score, noise, or velocity) is learned, there theoretically exist infinitely many valid sampling schemes. Therefore, enforcing the Coherence Principle is not inherently necessary.

2. Lack of Experiments / Related Work.

The authors argue that the noise schedule determined by the Coherence Principle is beneficial, citing a few works (Cui et al., 2025; Ma et al., 2024) that allegedly conform to it. However, this reasoning is flawed. The examples are too limited, as many SOTA methods do not satisfy the Coherence Principle, yet were selectively omitted. Moreover, no experiments are presented. If deterministic sampling is said to violate the principle, the authors must show that all schedules coherent with it outperform deterministic ODE sampling; otherwise, the claimed advantage of the Coherence Principle remains unsubstantiated.

### Questions
1. Can the authors justify the necessity of the Coherence Principle? It is a very strong assumption, yet the paper devotes too little space to explaining or motivating it.
2. Can the validity of the Coherence Principle be verified through extensive ablation and reasoning experiments, rather than being limited to the few cited works?

### Soundness
1

### Presentation
2

### Contribution
2
