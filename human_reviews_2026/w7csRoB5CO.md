# ConvT3: Structured State Kernels for Convolutional State Space Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Modeling long spatiotemporal sequences requires capturing both complex spatial correlations and temporal dependencies.
Convolutional State Space Models (ConvSSMs) have been proposed to incorporate spatial modeling in State Space Models (SSMs) using the convolution of tensor-valued states and kernels. 
Yet, existing implementations remain limited to $1\times 1$ state kernels for computational feasibility, which limits the modeling capacity of ConvSSMs.
We introduce a novel spatiotemporal model, ConvT3 (ConvSSM using Tridiagonal Toeplitz Tensors), designed to equivalently realize ConvSSMs with extended $3\times 3$ state kernels.
ConvT3 structures a state kernel for its corresponding tensor to be composed as a structured SSM matrix on hidden state dimensions and a constrained tridiagonal Toeplitz tensor on spatial dimensions.
We show that the structured tensor can be diagonalized, which enables efficient parallel training while leveraging $3\times 3$ state convolutions.
We demonstrate that ConvT3 effectively embeds rich spatial and temporal information into the dynamics of tensor-valued states, achieving state-of-the-art performance on most metrics in long-range video generation and physical system modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an extension of the ConvSSM by enabling a larger state-space kernel size, which increases from the original value of $1$ to $3$. The model is parameterized via a tridiagonal Toeplitz tensor and asymptotic stability is guaranteed. The model is empirically validated on long-range video generation (Moving-MNIST) and physical system modeling (PDEBench) tasks.

### Strengths
* The derivation of the ConvT3 block is rigorous and well-grounded in tridiagonal Toeplitz matrix theory.
* The empirical results look promising, and the model seems to gain performance at a small cost of computing.
* The ablation is thorough and insightful. It demonstrates the importance of four design choices separately.

### Weaknesses
* It is not thoroughly analyzed in the manuscript why increasing the kernel size from $1$ to $3$ in the latent space improves the model. While a complete theory may not be very accessible without an extensive amount of work, the paper would benefit from a dedicated discussion (theoretical or empirical, but on a more insightful toy example where further analysis can be done) that illustrates the benefits of a larger convolution kernel.
* The mathematical derivation in section 3.3 is dry and purely technical. It is hard to tell what the final parameterization method is with a first glance. I would recommend that the author(s) start with a clear and accessible method and then use a theorem to show its stability property. (The proof can be dumped into an appendix.)
* While ConvT3 is as efficient as ConvS5 in the $\mathcal{O}$-notation, the constant inside is also important. This governs how the ConvT3 model is less computationally efficient in practice. It would be good to show some analysis of this kind or show some scaling plots.

### Questions
1. Is $P = 3$ a special number? That said, since the theory is heavily grounded in the tridiaognal Toeplitz matrix theory, I imagine that it could be hard to generalize this framework to a larger $P$ per Galois theory; is this true? How does $P = 2$ compare to $3$?
2. How does the memory usage of ConvT3 compare to that of ConvS5?
3. In Table 1, you seem to show that ConvT3 works much better than ConvS5 in the long-horizon regime (see 800 frames versus 1200 frames). Do you have a good intuition in that?

### Soundness
3

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
2

### Summary
This paper proposes a ConvSSM method using Tridiagonal Toeplitz Tensors, which equivalently implements a ConvSSM with 3 $\times$ 3 state convolution. Compared with traditional ConvSSM, this method give a constrained tridiagonal Toeplitz tensor. This method can avoid exploding computation in parallel scans with larger kernel and learn state dynamics from effectively capturing spatiotemporal context.

### Strengths
1. This paper proposes a new method which are used in statistical theory and leverage this method into state space model.
2. This paper gives the proofs in detail and gives some theorems and definitions.
3. The experiments prove that this method is good enough compared traditional methods on video generation tasks and complex physical system modeling.

### Weaknesses
1. This paper introduces an excessive number of notations, with some lacking sufficient detail—for instance, the formula in line 196 is not adequately elaborated.
2. The paper fails to theoretically validate the effectiveness of ConvT3 in comparison to ConvSSM. It is suggested that the authors supplement a dedicated theoretical analysis section, rather than relying solely on experimental results.
3. Regarding the experimental validation, only two experiments are presented. However, classification and detection tasks are standard benchmarks in computer vision. Additionally, traditional SSM-based methods (e.g., Vmamba) have been applied to CV tasks. It is recommended to include more comparative experiments covering these typical tasks.
4. Certain theoretical details require clarification. Specifically, the calculation process of Equation 9 and its underlying rationale are not sufficiently explained.
5. It would be valuable to provide a comparison of time complexity/cost between the proposed method and transformer-based approaches.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes ConvT3, a convolutional state-space model (ConvSSM) that realizes a 3×3 state kernel while keeping linear-time parallel scans for long sequences. The key idea is to structure the state kernel as (i) a diagonalizable SSM matrix over hidden channels and (ii) a tridiagonal Toeplitz (TT) tensor over spatial axes. This structure enables efficient training while capturing richer spatial dynamics than prior 1×1 state kernels (e.g., ConvS5). On Moving-MNIST, ConvT3 achieves state-of-the-art results across most metrics; on PDEBench, it attains the best accuracy with efficiency close to ConvS5.

### Strengths
+ The paper shows how to move from 1×1 to 3×3 state kernels in ConvSSMs without losing linear-time scans, via a structured (diagonalizable) state tensor and tridiagonal Toeplitz (TT) formulation. 

+ Training stability: The paper uses a Hurwitz condition-based parameterization (constraining eigenvalues to have negative real parts) and a positive eigen-tensor construction to keep dynamics stable during training. This is validated by the training loss curve (Figure 3).

+ Ablations on kernel size and minimal parameterization: MiniT3 outperformed ConvS5 despite the minimal increase in parameters, and the kernel-size ablations show the 3×3 state kernel A is the main driver of gains, not B/C alone.

### Weaknesses
- Limited benchmarks: Moving-MNIST and PDEBench tasks show the benefits, but they are relatively synthetic. Adding natural video datasets or harder physics targets (e.g., Navier–Stokes) would strengthen the paper. 

- Compute and throughput details: The paper states linear-time scans and that efficiency is close to ConvS5, but the actual comparisons are limited. Per-component speed and throughput are missing. Also, efficiency measurement across sequence lengths and resolutions would make the efficiency claims stronger.

- Ablation: The kernel-size studies of A, B, and C are helpful, but additional ablations such as 3×3 vs 5×5 state kernels would be beneficial.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
