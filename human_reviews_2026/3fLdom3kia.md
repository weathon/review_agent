# Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We study the capabilities of small-scale transformer models in symbolic reasoning, focusing on the NP-hard algebraic task of multivariate polynomial decomposition, with widespread applications in science and engineering. Our approach includes a fine-grained synthetic data generation pipeline, supervised pretraining, beam search, evaluations for scaling behavior and generalizability, and a novel rank-aware reinforcement learning method called Beam Grouped Relative Policy Optimization (BGRPO), which improves accuracy while reducing inference compute by up to 75%. Additionally, our model demonstrates competitive performance in polynomial simplification, outperforming Mathematica in various cases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The polynomial decomposition problem is defined as follows:
- Input: A multivariate polynomial $f(x_1, ..., x_n)$.
- Outputs: $m$ inner polynomials $h_1(x_1, ..., x_n), ..., h_m(x_1, ..., x_n)$, and an outer polynomial $g(y_1, ..., y_m)$, such that $f(x_1, ..., x_n) = g(h_1(x_1, ..., x_n), ..., h_m(x_1, ..., x_n))$.

This is a fundamental problem in mathematics. The paper studies whether transformers can be trained to solve this problem, i.e., to map $f$ to $h_1,...,h_m,g$ as above.

The main methodological innovation is the introduction of Beam GRPO (BGRPO), which adds a ranking (as in beam search) to GRPO. The evaluation suite studies the effects of the problem setup (degrees / num variables of inner and output polynomials), the architecture (embedding dim, num layers, num heads). In addition, the ability of a model trained on one distribution to adapt to another via fine-tuning is explored.

### Strengths
- The problem of decomposing multivariate polynomials is an important and difficult (e.g., NP-hard) problem in mathematics. Many papers training transformers from scratch on arithmetic/algebraic capabilities focus on "toy problems", but this problem is already very interesting to study in itself. This paper would be a significant contribution to the literature, in this regard.
- An original method that dramatically increases inference accuracy. That said, it is not clear whether BGRPO is useful in any other domain (a preliminary experiment here would strengthen the paper).
- Besides minor (yet consistent) writing errors (see Weaknesses), the paper is overall well-written. The problem setup and methodolgy are clear. Most experiments are clear (see again Weaknesses).

### Weaknesses
- BLEU is a notoriously difficult metric to interpret; a point-increase in BLEU could mean entirely different things depending on reference dataset quality/amount, tokenization scheme, and even then many authors call the metric into question [e.g. 1,2]. It's acceptable to rely on BLEU when comparing different models (translators) on the same data setup, but otherwise one should argue carefully why BLEU is used. Therefore, the comparison in Section 4.4 is problematic, namely, comparing a "6.3x improvement" with BGRPO to beam search that "yields BLEU score improvements of only 2--4 points". The authors should evaluate BGRPO vs. beam search either apples-to-apples (on a polynomial decomposition task), or oranges-to-oranges (machine translation on a fixed data setup), but comparing apples-to-oranges is invalid.
- The comparison to Mathematica was not clear to me. That is, I was not able to conclude that transformers with BGRPO are actually more effective than Mathematica. For one, as a reader unfamiliar with Mathematica, Leaf Count as a metric was not clear to me (also, should standard errors be reported in addition to the mean?). Importantly, because Mathematica's FullSimplify is not neural but a heuristic crafted by humans, I expect it to generalize better (in fact, it has no "training dataset"); therefore, if the authors want to claim that their method is superior to FullSimplify, a more comprehensive evaluation (e.g., length generalization) should be conducted. It's also important to report how the time and memory requirements of each method compare. With heuristic-based search algorithm such as FullSimplify, it is usually the case that output quality scales with the time and memory allotted to the algorithm.
- Polynomial decomposition, as a fundamental problem in mathematics, has a rich literature behind it. However, as the paper is currently written, none of the insights given to us by mathematicians over decades are used; instead polynomial factorization is treated as a "black-box" problem, with the literature only cited as motivation. One particularly begging direction is to use the literature to gain interpretability insights for error analysis. As an example, [3] explores the problem of learning to compute the GCD of integers and their error analysis is able to determine with near-certainty where the model will err depending on algebraic properties of the inputs.
- Minor note: Many citations should be made parenthetical. That is, use the "citep" command instead of "citet".

[1] Tangled up in BLEU. Nitika Mathur, Timothy Baldwin, Trevor Cohn. ACL 2020.
[2] A Call for Clarity in Reporting BLEU Scores. Matt Post. WMT 2018.
[3] Learning the greatest common divisor. François Charton. ICLR 2024.

### Questions
- Following my comment on Section 4.4, how does BGRPO fare on NLP tasks?
- How much time and memory (RAM and GPU) does your model take as compared to FullSimplify? If FullSimplify's time and memory usage depends on the input, then reporting per-input score (or e.g. dividing by FullSimplify's runtime) is interesting.
- Have you compared to any other method besides FullSimplify?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explores the capabilities of transformers on the NP-hard problem of multivariate functional decomposition. For data generation, the method utilizes the fact that the reverse direction is much easier, i.e., going from composed functions to the flat expression. Furthermore, the authors propose rank-aware BGRPO, a variant of GRPO where the reward groups are sampled using beam search instead of randomly and the rewards are weighted based on the corresponding output's position in the beam (ranked by the sum of log probabilities). The experiments support the merit of incoporating the rank information and the general benefit of BGRPO alignment. The authors extensive ablation studies on their method's hyperparameters. They also compare their method to Mathematica's FullSimplify function on the task of simplifiying polynomial expressions where they show that their method outperforms Mathematica's (i.e., produces simpler expressions) in some cases.

### Strengths
1. The paper tackles an interesting application of deep learning, where it appears (to my knowledge) to be the first work exploring the potential of transformers on functional decomposition. 
2. The ablation studies are fairly extensive. 
3. The method outperforms Mathematica on the task of simplification in 2 out of 5 attempted complexity configurations.

### Weaknesses
1. All evaluations were performed on synthetic data, lacking evaluations on real-world instances. 
2. The lack of baseline evaluations makes it hard to contextualize the overall performance. While many existing algorithms tackle different constraints of the problem, the authors could still evaluate their method on these special cases. More importantly, Faugère & Perret (2009) [1] present a heuristic algorithm that handles the single-polynomial multi-multivariate decomposition case (when u=1) on which the current method operates, so a comparison between the two is very much needed. 
3. The paper has a claimed contribution of extending GRPO by sampling the output group using beam search instead of random sampling, but the experiments do not compare the impact of this extension (BGRPO) to the baseline GRPO. Section 4.5 shows that BGRPO leads to overall benefit to the supervised pretrained model, which is good, but it also needs to show the value of the extension by comparing it to vanilla GRPO. 
4. (Minor) At the beginning of page 7, there is an extra indentation to the first two lines. It looks like the padding of Figure 4 carried over to the next page. 
5. (Minor) In the keywords of the submission, "Functional decomposition" is missing an 'F'.

### Questions
1. Can you evaluate your method on vanilla GRPO to highlight the benefit of beam sampling (ideally, in Figure 8)? 
2. Do you have any famous case studies or datasets from cryptography (or any real-world problems) to support evaluations? 
3. Can you compare with pretrained LLMs? They could offer an interesting off-the-shelf baseline to demonstrate the value of the training pipeline. 
4. Have you considered incorporating rank information into the pretraining (with GT labels) objective as well?

I'd be willing to increase my score if literature baseline comparisons (on general or special cases) are provided, because otherwise, it's really hard to put the performance numbers in perspective.

**References:**  
[1] Faugère, J. C., & Perret, L. (2009). An efficient algorithm for decomposing multivariate polynomials and its applications to cryptography. Journal of Symbolic Computation, 44(12), 1676-1689.

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
4

### Summary
This work investigates whether transformer models can be trained to solve multivariate polynomial decomposition (with the main aim being non-linear pattern discovery). Authors primarily follow these four steps 1) create a backward synthetic data generation pipeline. 2) train a lightweight transformer model using SFT and analyze performance, 3) observe that beam search decoding strategy works much better than multi-sampling and greedy strategies and finally 4) To improve further, rank-aware GRPO is proposed to decrease the computational intensity of beam search.
Authors make various interesting observations: 1) many polynomial complexity parameters asymmetrically affect performance, scaling with more data helps in better accuracy, and impressive transfer accuracy is obtained with small amount of data, 2) BGRPO improves decomposition accuracy and reduces beam width requirements by up to 50%, leading to around 75% computational savings, 3) demonstrates competitive results in polynomial simplification compared to Mathematica.

### Strengths
1. The problem formulation is definitely insightful. I have seen various different avatars of polynomial handling. But, this also has practical implications.
2. Experiments are comprehensive, with in-depth analysis of both vanilla models and improved with BGRPO.
3. Rank-aware BGRPO seems to be an innovative contribution (modulo the fact that improvements seem to decrease with dimension size).
4. Various insights are produced which are useful.

### Weaknesses
1. The repercussions of using beam search instead of sampling from the distribution is not discussed. Maybe this is why the effect decreases with more model capacity.
2. Some ablations across varying representation and effect numeracy is missed.
3. While the Lample-Charton era work has discussed polynomial handling ability of vanilla transformers, it would have been great to discuss how pretrained Language models (pre-LLM is fine as well) can handle such tasks.

### Questions
L145: Using beam search during "sampling", as the authors themselves say alters the distribution. I wonder what is the theoretical implication of this.

L172: While I understand, there have been a lot of work during the Lample-Charton era of looking at the effect of representations, I believe looking at that ablation would have been better here as well.
Similarly, it is well-known that Transformers do not handle numeric multiplications well [1]. One can easily convert this into symbols such as c1, c2 and do the computation as suggested by [1] (may have disentangled this effect).

[1] Agarwal et al 2021, ICLR MathAI

L269: Same as above. Is this possibly also because of complexity increase with numeric addition at the exponent space?


Minor:
1. L591: I am guessing there are some formatting issues. Its unclear whether its written as an algo or paragraph?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigates the ability of transformer models to discover hidden algebraic structures by addressing the challenging multivariate polynomial decomposition problem. The authors propose a synthetic data generation pipeline, a systematic evaluation framework, and a novel BGRPO method to improve beam search efficiency. Experimental results show that their approach enhances model accuracy, reduces inference cost, and outperforms Mathematica in certain tasks.

### Strengths
1. The paper claims to be the first to systematically explore transformers’ ability to uncover hidden nonlinear algebraic structures.
2. The experiments span multiple dimensions—including problem complexity, model architecture, distribution adaptation, and search strategies—offering a comprehensive and systematic evaluation.

### Weaknesses
1. The work is limited in scope, as it focuses solely on polynomial decomposition rather than broader symbolic reasoning or algebraic tasks.
2. The paper does not provide a clear motivation for using a Transformer architecture.
3. The experimental evaluation of BGRPO is somewhat incomplete. The method is introduced as an improvement over GRPO and PPO, yet no quantitative baseline results are provided for these methods. This makes the advantage of BGRPO unclear.
4. Although the paper states that the polynomial decomposition problem has broad real-world applications, the experimental evaluation is conducted solely on synthetic data, with no real-world datasets or case studies to demonstrate practical effectiveness.
5. The paper does not report the complete computational overhead of the method, such as the training cost of BGRPO or runtime comparisons against Mathematica.
6. The paper contains several typographical errors. For example:
   - Line 56: multi-multivariate
   - Line 615: ouput

### Questions
1. Since polynomial decomposition is generally not unique, is there any notion of quality or preference among different valid decompositions?

2. In BGRPO, the authors incorporate rank information by scaling the reward with an exponential decay function $e^{−rank/w}$. Could the authors clarify why this specific form was chosen? Have other ranking functions (e.g., linear or logarithmic decay) been considered or tested?

### Soundness
2

### Presentation
3

### Contribution
3
