# Interpreting neural networks depends on the level of abstraction: Revisiting modular addition

- Decision: Reject
- Scores: 6, 5, 5, 3, 5

## Abstract
Prior work in mechanistic interpretability has analyzed how neural networks solve modular arithmetic tasks, but conflicting interpretations have emerged, questioning the universality hypothesis—that similar tasks lead to similar learned circuits. Revisiting modular addition, we identify that these discrepancies stem from overly granular analyses, which obscure the higher-level patterns that unify seemingly disparate solutions. Using a multi-scale approach—microscopic (neurons), mesoscopic (clusters of neurons), and macroscopic (entire network)—we show that all scales align on (approximate) cosets and implement an abstract algorithm resembling the approximate Chinese Remainder Theorem. Additionally, we propose a model where networks aims for a constant logit margin, predicting $\mathcal{O}(\log(n))$ frequencies—more consistent with empirical results in networks with biases, which are more expressive and commonly used in practice, than the $\frac{n-1}{2}$ frequencies derived from bias-free networks. By uncovering shared structures across setups, our work provides a unified framework for understanding modular arithmetic in neural networks and generalizes existing insights to broader, more realistic scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits the modular addition and aims to find a unifying answer in light of the divergent narratives in the literature. Their main discoveries are: (1) they study not only prime but also composite p for modulo addition. By scanning a large number of random seeds, they discovered the learned frequencies are uniform (which is expected), but the interaction between frequencies is non-trivial (e.g., if frequency n is learned, then frequency n/2 and 2*n are less likely to be learned). (2) There are simple neurons (pure frequency) and fine-tuning neurons (a mixture of a few frequencies). (3) They proposed a mathematical model which posits that certain neurons represent approximate cosets, unifying clock/pizza algorithms. (4) They derive that O(log n) clusters should be needed, matching with experimental results.

### Strengths
1. The paper is well-written and sound, nicely balancing theoretical and empirical results
2. Unifying previous diverging narratives is one contribution of this work
3. They have several very interesting observations that may be worth separate papers: frequency interaction, the distribution of the number of clusters, etc.

### Weaknesses
1. Some parts are hard to read. For example, I find myself a bit lost in Section 5.4. It'd be nice to highlight some important takeaways since not all readers can follow through the detailed mathematical derivations.
2. While this paper digs deep into the modular addition task, it is unclear whether/how the discoveries can be extended to general tasks.

### Questions
1. I'm not sure what the "level of abstraction" in the title refers to.
2. The result that O(log n) clusters are needed is nice, but since only n = 89 and 91 are studied, it could be a fluke. Is there evidence for smaller n or larger n to have fewer/more number of clusters?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper focuses on how neural networks learn modular addition with prime and composite modules. The authors show that different previously observed algorithms are unified manifestations of the same underlying mechanism, which is related to implementing coset structures of subgroups under addition. They identify two types of neurons (simple and fine-tuning) and demonstrate that networks implement an approximate version of the Chinese Remainder Theorem by clustering neurons with different frequencies.

### Strengths
1. Clear and straightforward setup
2. The authors offered simple mathematical explanations of the phenomenon they observed
3. The authors tried to build a unified view on different algorithms observed related to the ``grokking'' phenomena

### Weaknesses
1. Several references were overlooked or inadequately credited, which affects this paper's novelty:
  
  - D. Stander et al. (arXiv:2312.06581) demonstrated that their model learns coset structures during training on permutation groups $S_5$ and $S_6$, where they also identified circuits using Group Fourier Transform. Their work performed a very similar analysis to this paper's examination of modular addition.
  - The weight structure discussed in Section 5.1 is identical to that proposed by A. Gromov (arXiv:2301.02679). The authors neither properly credit this previous work nor rigorously explain how such weight matrices would correctly generate predictions with ReLU activation. Section 5.2 merely presents minor generalizations to the case of composite numbers.

2. The authors draw conclusions based solely on their specific experimental settings. While they attempt to explain and unify various phenomena, these were originally observed in different settings. The paper would be more convincing if it provided experimental evidence from the original settings used by others and demonstrated that the coset explanation holds consistently across different settings for the same task.

### Questions
Could the authors address my concerns mentioned in weakness section?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors revisit the problem of reverse engineering neural networks trained to learn modular addition, attempting to unify seemingly different algorithms learned by neural networks in the literature. In particular, the authors identify “simple” neurons and “fine-tuning” neurons, and build a mathematical model for them. First, the authors explain that simple neurons implement approximate coset. The authors also demonstrate that the constructive interference of multiple frequency components leads to maximum signal (logits), and identify its relation to the Chinese Remainder Theorem (CRT) when the modulus is composite.

### Strengths
- The paper explores reverse engineering neural networks trained to learn composite modulus, which has rarely been studied in the literature.
- The paper develops an interesting mathematical model that can account for seemingly different learned algorithms in the literature.
- The paper effectively links the mathematical model with experimental results.

### Weaknesses
While this paper presents a lot of interesting ideas and results, I believe that further analysis on (a) the identified frequency clusters and (b) networks trained to learn composite modular addition would significantly strengthen its contributions.

The formalization of constructive interference between different frequency components, which was briefly introduced by Pearce et al. (2023), is a valuable aspect of this work. However, I feel that certain analyses are missing. For example:

- According to authors’ analysis, the number of frequency clusters depends on $O(log n)$. It would be valuable to experimentally validate this for a broader range of $n$, say from 3 to 997, not just around 90.
- Do all neurons in the same cluster play an equally important role in computing the answer?
- Does the model performance significantly drop if we ablate any of the $O(log n)$ clusters? What happens if we ablate only a part of the cluster? Models may develop unimportant frequency clusters that are not crucial for computation.
- What exactly are the chosen frequencies? Are they random, or do they satisfy certain constraints? For instance, in Figure 4(b), it would be interesting to include a plot that shows which frequencies tend to co-occur given a random seed.

For learning composite modulo addition, the authors explain that the constructive interference is very much analogous to CRT; Providing additional evidence that neural networks indeed implement CRT - such as by localizing intermediate computation results - would further bolster the paper’s contribution.

### Questions
- Typo in Line 330 ($n\rightarrow n'$ ). I think it may be clearer to reorganize section 5.2 in terms of theorems, i.e., first provide a formal definition of “approximate coset,” and then provide a proof.
- At the beginning of 5.3, I think neurons in the same cluster having different phase does not necessarily disprove Nanda et al. (2023a)’s algorithm - It is possible that there are multiple circuits doing the same operation (starting by translating one-hot $a,b$ into Fourier basis) within the model, each corresponding to neurons of same frequency but different phase.
- It is unclear if the approximation in Line 412 is valid. $cos(2\pi k) \approx 1-(2\pi k)^2/2$ is a valid approximation when $k\ll1$, not when $k$ is close to integers.
- Do you have any explanations for the conditional distribution of frequencies, on why it is less likely to learn $2f$ or $f/2$ when $f$ is learned?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The premise of this paper is to understand how modular representations appear in networks and introduces language to characterize them (i.e. coset structures/circuits, "simple" neurons, and "fine-tuning" neurons). This paper also offers theory for identifying these structures in trained networks.

### Strengths
- In general, this paper is interesting in the way that it conveys notions of model superposition and modularity with group-theoretic ideas.  

- Fig. 3b is nice in that it offers some intuition relating "simple" neurons and coset structures. 

- "Re-opened conjecture 2" is certainly interesting.

### Weaknesses
$\underline{\text{Nits}}$

L298: modelling $\to$ modeling

Figure 5: Why are "Contributions" and "Clusters" capitalized? 

Frequent use of the LaTeX \citet function instead of \citep.

Figure axis labels and titles are extremely difficult to parse/understand. Figures themselves are poorly designed and constructed. 

$\underline{\text{General}}$

While the ideas in this paper are interesting and offer some new perspectives on well-studied topics in ML literature, I believe this paper is somewhat poorly written, making it difficult to properly assess. 

- What is Fig. 1 doing on page 2? I had a difficult time understanding in how this figure lends to the story being described in the introduction, only to find it finally referenced a few pages later. Even then, the figure captions and titles did not help me understand the figures easily. 

- Many of the analyses are not well motivated. For instance, the first section of the Experimental Findings immediately delves into understanding/characterizing the difference in networks trained with batch sizes = n and with batch sizes = size of the full training dataset. Why? What motivates this analysis? Although I found much later in the paper that this was something studied in one of the reference papers, this paper needs a major revision/reworking in order to convey the story better. Fig. 2 can be a lot clearer in explaining why this analysis is necessary. Finally, what do you expect to happen for weights when trained on SGD (batch sizes = n) vs GD (batch sizes = size of dataset)? Yes, you mention that smaller batches find sparser embeddings. But what does this information provide the reader? 

- If the goal is to show that across 100k trained networks (of varying seed) that the coset structures generally emerge, wouldn't one also do strict ablations / variations of learning rate, model architectures (larger/smaller parameters, varying depth), choice of optimizer, and with and without the L1 regularization? 

- Although I like Fig. 3b and the accompanying text that describes these findings, how significant is this result? Would $\textit{any}$ network trained on $\textit{any}$ inherently modular task learn redundant functions/circuits/representations? Additionally, this network is a simple 1-hidden layer MLP. How do your conclusions hold for MLPs of varying depth? 

- L344-352: I am unsure if this text is proposing this information as a novel contribution of this paper. Model superposition papers exist in plethora which show that orthogonality of learned features/functions are not optimal in many cases, and that smaller networks replicate the circuitry of larger, more sparse networks in specific cases.

- Do you really need L1 regularization in Conjecture 1? Additionally, wouldn't Conjecture 1 only hold for MLPs of 1 hidden layer?
 
- A general comment I have is that some intuitions for how this group-theoretic perspective can aid in understanding modularity and model superposition in networks trained on non-mathematical tasks would be especially helpful, considering since these topics are well studied in standard ML / mech. interp. literature.

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies modular addition (prime and composite) with a 2-layer ReLU-MLP. They interpret the learned algorithm in the language of cosets (and the so-called approximate cosets) in conjunction with the known Fourier features. They present empirical evidence for their suggested algorithm. Using their findings, they attempt to unify the findings of prior works on the topic.

### Strengths
- Studying composite modulus for modular addition and highlighting the subtlety with cosets is new.

- I find the connection to non-circular algorithms found in [1] (with Lissajous curves) to be quite interesting (Lines 477-481).

- In networks exhibiting frequency clusters (such as [2]), discussing the phase shifts within a frequency-cluster is new, to my knowledge. 

- I appreciate the authors averaging their experiments of 100k random seeds to eliminate fluctuations.

### Weaknesses
My main concern about this work lies in its novelty/contribution.

- For prime moduli, the current work offers very little more insight than the union of prior works [2, 3]. Specifically, the superposition of multiple frequencies resulting in high output at the correct logit has been extensively studied in [3]. The current paper studies a slightly different setup, which results in frequency clusters (like in [2]) rather than individual frequency neurons (like in [3]). However, the paper effectively distills down to a superposition of these works. While the result itself is new in this setting, the insight gained is limited.

- For prime moduli, the coset picture does not add any more insight, since all cosets are of order 1 ($gcd(f,n)=1 \\;\\forall f$). For composite moduli, one would expect that having multiple elements in coset would affect the prediction accuracy of individual neurons. However, Fig. 4 shows no difference between $n=89$ (prime) and $n=91$ (composite). Furthermore, the "approximate coset" interpretation is just a different way to describe the algorithm found in [3], to my understanding.

Additionally, some of the claims made in the paper are not presented with sufficient empirical evidence (Elaborated in the Questions).

### Questions
- How do the magnitude histograms shown in Fig. 2 prove that rotation matrices are not learnt? I think this claim requires further justification.

- Lines 208-211: How is this finding different from the results from [3]?

- Line 232: Isn't it more reasonable to look at relative magnitude of highest frequency (relative to other frequencies within the neuron) rather than absolute magnitude?

- Section 5.3: It is unclear how the suggested representations are drastically different from the rotation matrices. Aren't they just shifted+rescaled rotation matrices? 

- Lines 450-456: The result that the number of clusters $\\approx log_e n$ requires further verification. One needs to empirically verify this using multiple different moduli -- and check if it scales correctly. For one/two moduli, the values of $\\delta$ and $\\rho$ can be cherry-picked to match the observed number of clusters.

- Lines 477-481: This is an interesting hypothesis. But it needs to be made more precise and shown empirically.

- It is unclear to me how the presented algorithm unifies clock and pizza algorithms from [1]. Could the authors clarify this?

Suggestions:

- Line 241: "Approximate coset" term is used here, but defined much later -- this hurts the clarity.

- Line 330: potential typo: "multiple of $n'$" instead of "multiple of $n$"

- Lines 327-336: The explanation for approximate coset might benefit from a simple figure.

[1] Zhong et al., "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks" (2023)

[2] Nanda et al., "Progress measures of grokking via mechanistic interpretability" (2023)

[3] Gromov, "Grokking Modular Arithmetic" (2023)

### Soundness
3

### Presentation
2

### Contribution
2
