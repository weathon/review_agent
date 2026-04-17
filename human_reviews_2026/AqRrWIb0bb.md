# Towards Scalable and Robust Filtration Learning for Point Clouds via Principal Persistence Measure

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Topological features in persistent homology extracted via a filtration process have been shown to enhance the performance of machine learning tasks on point clouds. The performance is highly related to the choice of filtration, thereby underscoring the critical significance of filtration learning. However, current supervised filtration learning method for point clouds can not scale well. We identify that this shortcoming stems from the utilization of Persistence Diagrams (PD) for encoding topological features, such as connected component, ring or void, etc.
To address this issue, we propose to use Principal Persistence Measure (PPM), a statistical approximation of PD, as an alternative representation and adapt existing network for PPM-based filtration learning.  Experimental results on point cloud classification task demonstrate the effectiveness, scalability and robustness of our PPM-based framework.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an extension of [Nishikawa et al, 2023] to allow  learning weighted Rips
filtration, using Principal Persistence Measures (PPM). This measure can be
approximated by computing an average of persistence diagrams on $2q+2$ points,
which can be fully performed on the GPU.

The authors then motivate this construction with a theoretical analysis, which
guarantees that the PPM is robust to a small fraction of arbitrary noise. Then,
the experimental section showcase the statistical and computational performance
of this construction, as well as its practical robustness to outliers.

### Strengths
- This approach seems sound to me. 
 PPM are well-defined objects, with satisfying properties, and the adaptation
 of [Nishikawa et al, 2023] to this special case totally make sense in this case.
 - Computational complexity.
 The degree $q$ persistence of a point cloud with $2q+2$ has a closed form
 formula, which makes the empirical measure very efficient to compute on the
 GPU.

### Weaknesses
- Theoretical Analysis. I'm not convinced by the results given in the paper,
 and I feel that some other results could be stated.
    - Section 5. The assumptions are not formally stated, so I had to infer some
    assumptions. 
    For instance, 
      - The error bound on Lemma 5.1 does not depend on the noise $U$, which I
      is surprising at first glance.
      By looking at the proof, we read that it requires (to apply a result from
      [Chazal, Divol, 2018]) that the sampling measure (noise included) has to have a density
      w.r.t. the Hausdorff measure on the underlying manifold.
      In particular, the noise $U$ has to be supported on the maniforld $M$.
      This is a bit counterintuitive to me as the point of the manifold
      assumption is usually to assume that the dataset has some density w.r.t.
      the uniform measure of this manifold.
    - Some results (with the proofs when needed) are worth mentioning:
       - stability of PPM (c.f. [Gómez et al, 2024])
       - convergence rates of the empirical measure to the PPM
       - error bounds
       - ... 
    - Some definitions / assessments are false or imprecise.
       - l33. Deathtimes can be infinite.
       - Section 2.1. Please detail the manifold assumption.
       - Section 2.2. The Rips filtration doesn't have the same topology as a union of balls.
       - l129. "rarely developed". 
       - l159. This statement is not true for the degree 0
       homology. Two points generate two $0$-cycles in homology. 
       Also, "topological feature" is very vague.
       - l254-270. This is not clear. Is $P$ a measure absolutely continuous w.r.t. the Lebesgue measure on $M$? Is $X$ a sample of $P$? or of $P_o$? What are the assumptions on $P$, $U$?  
       - l260. Are the Assumptions K1-K5 linked to the assumptions made l221? 
       - l741. "where under the assumption that each xi is in the dense area
       (support with positive density) of P". What about the other points?
 - I think that the fact that this computation can be done on the GPU is a
 very good selling point, and should be a bit more emphased. 
 For instance, by mentionning Theorem 4.4 from [Gómez et al, 2024], and detailing the
 procedure.
 - Theorem 5.2. I feel that this result is a bit weak in practice.
 The volume of the manifold can be arbitrarily large, and at a large power
 (unless $q=0$), so the fraction of noise has to be very small to control the
 measure errors.
 - Experiments. 
   - No competitors. A motivating factor is that this method is scalable, so I
   agree that most of TDA method will not scale up with code that can be ran on
   GPU only.
   Nevertheless, even if it is not necessarily competitive (which make sense
   since the main goal is to scale up) I think this is important to compare
   this method with others when possible, e.g., the previous method [Nishikawa et al, 2023],
   Perslay, ATOL, or non-DL based methods (e.g. Landscapes, Sliced Wasserstein
   Kernel).
   - Figure 4. There are not enough $\varepsilon$s, no
   error bars, not enough datasets, and I don't see a plateau around
   $\varepsilon=0$ suggesting that this method "ignores" the first outliers. 
 - Formatting issues. there are plenty of times where a mathrm or a `\` is
 missing in the latex. ($exp$, $diam$, $inf$, etc.)

### Questions
See weaknesses.

### Soundness
1

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
5

### Summary
This paper studies the use of persistent homology (PH) for machine learning on point-cloud data. By parameterizing the filtration with a neural network and training it jointly with the downstream network, task performance can sometimes be improved. However, existing approaches for point clouds are not scalable due to the high computational cost of PH. This work leverages PPM, a fast approximation of PH based on subsampling from point clouds, to accelerate filtration learning. Experiments on protein data and object-surface data show that the proposed method achieves performance comparable to (or higher than) existing methods while being computationally faster.

### Strengths
- The paper addresses an important challenge in applying PH to machine learning: the computational cost of PH often becomes a bottleneck in large-scale ML. To tackle this, the paper shows that the recently developed PPM can also be used for filtration learning, thereby making PH-based filtration learning more practical.
- The advantages of the proposed method are comprehensively validated. On the protein dataset and ModelNet10, the proposed PPM-FL attains performance comparable to or better than PD-FL while requiring less training time.
- Beyond task performance, the paper investigates scalability in detail (Sec. 6.3). It analyzes how the training time changes with the point count $n$ and the number of subset samplings $M$.

### Weaknesses
1. Apart from using PPM and kernel-based weighting, the method appears to have limited novelty. Many ideas (such as using DeepSets and distance matrices) seem to come from prior work. Since the method is faster, one might expect it to accommodate a more expressive neural network for the weighting function; additional exploration of architectural improvements could strengthen the contribution.
2. The experimental diversity is limited. Given the reduced computational cost compared to prior methods, experiments on larger-scale datasets should now be feasible. Rather than stopping at speedups on small datasets, it would be desirable to clarify the effectiveness of filtration learning on large-scale point-cloud classification tasks.
3. The paper only compares point-cloud backbones with DeepSets and PointNet. It would be preferable to include comparisons (and combinations) with more modern architectures such as DGCNN or PointMLP.

### Questions
1. In the proposed method, how does performance vary with respect to the bandwidth $\sigma$ of the kernel $K$? Can the authors comment on a principled selection method or on robustness to this choice?
2. PPM may struggle to capture topology in sparse regions of a point cloud. Could the learned weighting in filtration learning mitigate this issue?
3. Is it possible to transfer the learned filtration from the proposed method to other tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces PPM-FL, a neural network architecture that learns the best weighted Rips filtration for persistent homology (PH), that heavily builds on Principal Persistence measure (PPM), which calculates PH of small random subsets of point cloud data. The scalability and robustness of the proposed approach are discussed theoretically, and supported by experimental results on protein and ModelNet10 datasets.

### Strengths
The paper deals with, in my opinion, two main issues when working with PH:  
(1) How to choose the best filtration, the input of PH? This choice is crucial as it determines the type of information captured by PH.  
(2) How to reduce the large computational cost of PH?

These issues are dealt in the paper with:  
(1) FL = filtration learning  
(2) using PPM rather than the standard PD (persistence diagram), an alternative and scalable output of PH.

### Weaknesses
However, even though I find the problem very relevant, one could argue that the proposed approach contains only incremental contribution that might not be substantial enough for ICLR. More precisely, the authors use:
- PD-FL: existing filtration learning approach with PDs from Nishikawa (2025), and
- PPM: existing PH signature from Gomez & Memoli (2024),  

to design their proposed PPM-FL approach. 

Other important weaknesses are addressed in the questions in the separate part of the review below.

Less important, but very annoying: Although the writing and organization are in general clear and easy to follow, there is an abundance of typos, strange formulations and formatting (some of which are listed below), that leave a poor impression of the paper not yet being ready for review. I suggest the authors to carefully re-read the complete text.

Minor comments:

-	Place the references in brackets whenever needed (most of the time).
-	Remove the blank space between the reference word and corresponding footnote number (many occasions).
-	$\cup \{X_s^i\}$ -> $\cup X_s^i$, throughout the paper?
-	measure of -> measure on, throughout the paper?
-	References: Check capitalization, e.g., vietoris-rips, Perslay, Pllay, 3d, Gpu, Gefl, …

-	p1, line 106: can not -> cannot
-	p1, line 018: connected component, ring or void -> connected components, rings or voids
-	p1, line 030: The topological features -> Topological features
-	p2, line 069: This can be equivalently expressed that for balls centered at each point with radius of η/2, if two balls touches [-> touch], a connection [-> an edge] between two balls’ centers is added. This sentence is rather convoluted, consider rephrasing.
-	p3, line 119: center -> centered
-	p3, line 145: network. PD -> network, PD
-	p3, footnote 3: the Appendix A.3 -> Appendix A.3
-	p5, line 216: $f(x_j, X_s^i)$ -> $f(X_s^i, x_j)$ (and same for $K$ on lines 233 and 234)
-	p7, line 363: the Appendix D.1 -> Appendix D.1
-	p8, line 382: as that -> to that?
-	p8, line 383: $(q=0&1 )$ with homology dimensions $q=0&1$ -> with homology dimensions $q=0&1$
-	p8, line 390: noted -> noting
-	p8, line 425: The settings in each phase here is -> The settings in each phase here are
-	p8, line 426: all the homology dimensions 0 and 1 -> both homology dimensions 0 and 1
-	p8, line 431: fixed, The -> fixed, the
-	p9, line 468: demonstrates -> demonstrate
-	p14, line 723: $D(K(X_s)$ -> $D(K(X_s))$
-	p14, line 734: $H_{kn}$ -> $H_k$?
-	p24, line 754: $M$ -> $\mathcal{M}$
-	p15, line 779: size being -> size
-	p15, line 785: of The -> of the
-	p15, line 789: . we -> . We
-	p15, line 290: was -> were
-	p16, line 838: The first -> In the first

### Questions
(Q1!!!) Crucial ingredients: In order to show that your PPM-FL approach is meaningful, and to justify your title – i.e., to show that it is scalable and robust, the following things should be demonstrated:  

(1) PPM is meaningful: (1.1) EPD is meaningful, (1.2) PPM as special case of EPD is meaningful  
(2) FL is meaningful   
(3) PPM(-FL) is scalable  
(4) PPM(-FL) is robust  

To do this, you either rely on some existing theoretical results from the literature, or demonstrate it experimentally, but I think (1) is missing in these discussions, leading to my main question. 

(1.1)	When defining EPD, you cite Chazal & Divol (2018). Is there a result in this paper that ensures that EPD is meaningful?

You rather seem to refer to Cao & Monod (2022) to support this [d(EPD(X), PD(X)) is upper bounded by a function of $n$ and $M$], but you do not frame it in this way? Also, this only comes in Appendix A, so you should make sure to at least provide one sentence explanation and a reference in the main text (with the theorem formulation and more detailed discussion in the appendix, as it is now). Finally, make sure to explicitly reference the particular Theorem 15 [Rate estimates for the approximation error for mean persistence measures].

(1.2)	Coming to my main question, why is PPM meaningful, when one considers only tiny subsets $X_s^i$ with a very few points (maximum 2q+2=4 when q=1). How does this makes sense? What is this invariant capturing, what does it tell us about the point cloud? I think the paper lacks this discussion (and in particular, since you mention a few times that this representation is “vague”).

At least some intuition could be beneficial. For instance, is the idea that if there is a loop in the point cloud $X$, it should be registered by a lot of random samples of 4 points? What if $X$ has two loops, how will PPM be different?

And indeed, you do show experimentally that PPM is as effective as PD, yielding a similar accuracy, but we know nothing about to what extent is the information captured similar? On p7 line 340, you write that PPM is effective in extracting meaningful topological features (and you conclude the same in the Conclusion), where is this coming from? Is there any result in Gomez & Memoli (2024), that you cite for your definition of PPM, which can aid this discussion? You only briefly mention their Theorem 3.2 in the appendix, but what does this result guarantee, is this not a crucial component for the paper (that should thus at least briefly be mentioned in the main text as well)?

If so, the question then remains whether the assumptions of the existing theorems are satisfied in your context (this is true for all the results that you use, throughout the paper, you should comment on this more extensively!!!). For instance, before Theorem 3.20 in Gomez & Memoli (2024), the authors mention that “that 10^6 configurations of 4 points are sampled uniformly at random”, so does this imply that $M$ needs to be really large (so that your rule of thumb of $M > n/(2q+2)$ might not be sufficient)?

Furthermore, in my opinion, Theorem 15 in Cao & Monod (2022) does not seem to be sufficient here, as it assumes that $|X_s^i|=n$, whereas in your approach these subsets are tiny, e.g. $|X_s^i|=2q+2=4$ for q=1?

Finally, if no theoretical guarantees can be obtained in this direction, you can frame your discussion about the experimental results in Tung (2025) [small $d(PPM(X), PPM(Y))$ yields small $W_p(PD(X), PD(Y))$, or is this small $d(PPM(K_1(X)), PPM(K_2(X))$ yields small $W_p(PD(K_1(X)), PD(K_2(X))$, be more precise] to support this argument (that PPM is related to PD).  

(2) FL being beneficial is shown experimentally in Section 6.1 (PPM-FL is better than PPM-Rips), so it’s fine. Note that you should thus rename this section as “Comparison with PPM-Rips and PD-FL”, see related (Q14) below.  

(3) PPM being scalable is briefly discussed, but this could be improved. Indeed, while you have a theoretical Section 5: “Robustness of PPM against outliers”, there is no section on the more important scalability (that is used to motivate your proposed approach). Would it be better to summarize the discussion about the theoretical computational complexity that is now partially lost in the footnotes in an additional Section 5: “Scalability of PPM”? In this section, one could start with a precise formulation of the theoretical complexity for the calculation of PD on the point cloud with $2q+2$ points, see also the first table in (Q2). Finally, some argumentation or intuition for the derived computational complexity of PPM-FL could be provided, at least by pointing to Figure 2.
 
PPM-FL being scalable is shown experimentally in Section 6.3, so this part is also fine. I would however change the order of Section 6.2 and 6.3 – indeed, your title says “scalable and robust”, and the former is more important, as you use it as motivation for your approach.

(4) PPM being robust [d(PPM(X with outliers), PPM(X)) is upper bounded by a function of the noise level $\epsilon$] is shown theoretically in Section 5, and PPM-FL is shown to be robust experimentally in Section 6.2, so this is great. As already mentioned above though, it’s probably better to reverse the order of Section 6.2 and Section 6.3.

In conclusion, I think you should provide a similar overview as above, e.g., when discussing the list of contributions in the introduction / the overview of the (structure of the) paper. Once again, I must stress the need to explain why your setup satisfies the assumptions in the theoretical results from the literature that you heavily use to justify your approach (e.g., in Chazal & Divol (2018),  “a real analytic compact connected submanifold” is assumed), primarily when it comes to moving to the consideration of only tiny subsets of $X$.


The remaining questions are listed in chronological order as they appear in the text, so that (Q!!!) is used to indicate more fundamental concerns.


(Q2) Notation: Most of the time a notation table (in an appendix) can be very helpful for the reader, and here it could also help to clarify your pipeline better, e.g., with something like:

Pipeline and scalability:

| Point cloud | Number of point cloud points | Weight | PH | Measure |  Computational complexity |
| ------------- | ---------------------------------  | -------- | ---  | ---------- |  ------------------------------ |
| $X$ | $n$ | $f$ | $D(X)$ | $\mu$ | $\mathcal{O}(n^{3(q+2)})$ algebraic algorithm via matrix reduction |
| $X_s^i \subset X$ | $2q+2$ | $f$ | $D(X_s^i)$ | $\mu_i$ | $\mathcal{O}((2q+2)^2)$ simple geometric algorithm (no matrix reduction needed) |
| $\mathbb{X}_s = \cup_{i=1}^M X_s^i$ | $(2q+2)M$ | $\overline{f}( \mathbb{X}_s, \cdot) =  \sum K(X_s^i, \cdot) f(X_s^i, \cdot)$| $PPM = \cup_{i=1}^M D(X_s^i)$ | $\overline{\mu}$ |  $\mathcal{O}((2q+2)^2M)$ |

Additional columns could also summarize the robustness properties. Rather than a table, a diagram or flowchart could be used to summarize the notation, give an understanding of the approach and could be used to summarize the contributions and structure of the paper.



|Notation | Description |
|-------------|-----------------|
| $r=(b,d)$ | persistence diagram point, topological feature that is born at $b$ and dies at $d$ |
|…|…|
| $P$ | true, outlier-free distribution |
| $P_o$ | noisy, outlier-contaminated distribution |
| $U$ | noise, outlier distribution |
| $\epsilon$ | noise level = percentage of outliers, since $P_o = (1-\epsilon)P + \epsilon U$ |



(Q3) Abstract (and elsewhere), FL: You write “current filtration learning method cannot scale well” where you assume the FL approach introduced in Nishikawa (2023), can you also comment on the scalability of other FL approaches in the introduction to Related work?

(Q4) Abstract, PPM: From the abstract (“To address this issue, we propose to use PPM…”) it seems that this paper is the first to introduce PPM, rephrase to make it clearer that this is also adopted from earlier work.

(Q5) Footnote 1: You write that PH pipeline can also be used to graph data, but the same is true for image data or e.g., constructible sets?

(Q6) Figure 1: What do the many PD points in (e) represent? After reading through the paper, I assume that each point corresponds to a feature in subset $X_s^i \subset X$, but this is not clear to the reader at this stage, so mention it explicitly in the caption.

(Q7) Section 2.1 on PD: 
- You write that $\mathcal{X}$ is finite, but this cannot be true?
- Should $\mathcal{X}$ be a topological space?
- You write that a cycle dies at $d$ when it ceases to exist in $\mathcal{X}_p^g$ for any $p>d$, but this formulation is imprecise, as it equally holds for any $d’ > d$.

(Q8!!!) Section 2.2 on weighted filtration: Firstly, you write that the sublevel set of the Rips filtration is the union of balls, but maybe you can first say that $g$ is here the distance from point cloud?

Secondly, and more importantly, the idea and motivation behind weighted Rips filtration should be clarified. In my current understanding, I have come across two ways this has be done:
- All point cloud points (0-simplices) have zero filtration function value and thus appear immediately in the filtration, as in Rips. However, the balls around points are constructed with different radii, influencing when the edges and other higher-dimensional simplices appear in the filtration (and is thus not only based on distances between the points, as in Rips) [2]. One reason to do this is to construct smaller balls around points in higher-density regions to avoid connecting them too early; this can be helpful to e.g. capture only one and not too loops in the Cassini curve (eyeglasses), see Figure 5 and Figure 6 in [1].
- In another approach, all point cloud points still have zero filtration function value and appear immediately in the filtration, but the inclusion of other higher-dimensional simplices is based on the Fermat rather than Euclidean distance [2]. Nothing is however weighted here, so that the approach can be seen as the standard Rips but in a non-Euclidean setting, or weighted Rips where balls radii are defined by Fermat distance.
- Point clouds points can have non-zero filtration function values (so that some do not appear immediately in the filtration, as in Rips), and the balls all have the same radius, such as in Anai (2020). One reason to do this is to smooth out outliers, by letting these outlying points have a rather high filtration function value and thus appearing very late in the filtration.

[1] Abigail Hickok, "A family of density-scaled filtered complexes",  arXiv:2112.03334 (2021).  
[2] Ximena Fernández, Eugenio Borghini, Gabriel Mindlin, and Pablo Groisman, "Intrinsic persistent homology via density-based metric learning", Journal of Machine Learning Research24, no. 75 (2023).  
[I have not read these papers in detail, and I am thus unsure whether I summarized the proposed approaches properly, but in any case such approaches can of course be considered.]. 

When reading Section 2.1, I thought that the weighted filtration in this paper falls in the former setting, but from Section 3 I then realized that it seems to generalize the two approaches, since at filtration scale $\eta = 0$ we can have $B(x, -\infty) = \emptyset$ so that the point $x$ does not appear immediately in the filtration. I think it is better to include the precise definition of the ball radius (the two cases) immediately in Section 2.1.

Therefore, at least a short discussion about the different definitions of the weighted Rips filtration (with appropriate references) should be mentioned, and the current approach better positioned and motivated. 

Finally, it should be acknowledged that Section 2.2 and the proposed FL approach in the paper are limited to the weighted Rips filtration.


(Q9) Section 3: While on the one hand I do like the title Section 3: “Related work” --- as it makes it clear that the two crucial components of your approach (FL and PPM) are adopted from previous work, it makes it difficult for the reader to identify that it is here where you define your important notion of PPM. One way to improve this is to organize the text into two subsections:
- Section 3.1: Neural network architecture for filtration learning (FL). 
- Section 3.2: Principal persistence measure (PPM).

Some other questions:

- I suggest to replace “network” with “neural network” throughout the paper.
- Is PPM reasonable, see (Q1) (1.2)?

- Is the motivation for the introduction of EPD in Chazal & Divol (2018) to “reduce the time cost of computing PD” or is it something else? In the latter case, rephrase the beginning of Section 3.2. to 

“In order [something else, e.g. give a better approximation of PH of noisy data], Chazal & Divol (2018) introduce EPD, where ….

By considering these random sets to be subsets of $X$ (small part of the point cloud), one can reduce the cost time of computing PD.”

This will give the reader some understanding on why PPM is robust to outliers, see related (Q11).

- It took me a while to understand what $r_1$ was, so maybe write before “the birth and death value of the single topological feature can be easily calculated:”.

(Q10!!!) Section 5, theoretical results (about robustness): The statements of your theoretical results, Lemma 5.1 and Theorem 5.2, should be more precise and complete, including all assumptions. For instance, when I was first reading them, I kept wondering what $\epsilon$ was, and it took me a while to find it in the text to realize that this is the percentage of outliers; hence you should explicitly state that $P_o^n = (1-\epsilon) P^n + \epsilon U^n$ in the statement of the theorems (and indeed, it is otherwise not clear in the proof in Appendix B where U comes from). Some other questions:
- What is $k$?
- Why write $p_{n-1}(\epsilon) \epsilon$, and not $p_n(\epsilon)$?
- Why is $\eta$ mentioned, how does the choice of measure influence the bound? Note also that, while you explicitly mention the measure in Lemma 5.1, you fail to do so in Theorem 5.2.
- Are all the assumptions from the existing results satisfied here? For instance, in Chazal, “a real analytic compact connected submanifold” is assumed. 
- Consider naming the results respectively as “Robustness of EPD to outliers” and “Robustness of PPM to outliers”?
- Footnote 8: You write that Hausdorff measure is a generalization of area and volume to non-integer dimensions, and then immediately assume $k$ to be an integer? Note that “and” in the definition of $H_k^\delta(A)” should be changed from math to text mode.

(Q11!!!) Motivation for robustness: In the abstract and introduction you talk about the:
- importance of the choice of filtration which motivates filtration learning   
- computational issues of PDs which motivate the use of PPM, a scalable alternative. 
but you never discuss noise, so that Section 5: “Robustness of PPM against outliers” seems to come out of nowhere for the reader, keeping us wonder whether this robustness was a desired property (that was incorporated in the design of the approach), or a fortunate by-product, a bonus. See also next question.

Is this robustness of PPM-FL achieved mainly by the robustness of:
- FL, since the weighted filtration allows to, among other things, smooth out the outliers (which is currently not at all mentioned in Section 2.2 nor Section 3, anywhere), see also related (Q8), 
- PPM (proven in Section 5), since multiple random subsets of the given point cloud are sampled, see also related (Q9).

Indeed, when looking at the results in Figure 3, I wondered for a moment why PPM-FL is more robust to outliers than PD-FL, as both of these approaches can learn (e.g., DTM-like) weight that smooths out the outliers, which is not discussed at all here. Adding the discussion above could aid the explanation of these results.


(Q12) Noise types: PH with respect to different filtrations is robust or sensitive to different types of noise. For instance, while PH on Rips filtration is robust under Gaussian noise (points close to the point cloud), it is extremely sensitive to outliers (points far from the point cloud) – even a single outlier can be “deadly”, as pointed out in the literature. This should be explicitly mentioned to motivate the focus on robustness to outliers (rather than some other type of noise).

On a related note, how heavily do your theoretical results about the robustness of PPM in Section 5 depend on the assumption that the noise is given by the outliers, can Theorem 7.1 in Chazal & Divol (2018) provide similar guarantees for other types of noise? Can this be any type of noise (including the underlying noise in the given classification problem, since the weight is learned)? This could yield a more general result; at least provide a very brief discussion?

(Q13) Figure 2 caption: Write rather “$v_j$ is the output of the weight function $\overline{f}(\mathbb{X_s}, x_j)$ on point cloud point $x_j$”. Remind the reader also here what are $g_1$, $g_2$ and $h$.

(Q14) Section 6.1, experimental section: Here it makes sense to first demonstrate that:
- FL is meaningful, i.e., PPM-FL is better than PPM-Rips, and then. 
- PPM is as effective as PD, i.e., PPM-FL achieves similar performance to PD-FL,  
and indeed this is how you discuss the results in the text; what also matches the steps in the PH pipeline: one first chooses the input of PH – the filtration, and then the output of PH – the representation, with e.g., PD or PPM. Therefore, I suggest to also switch the order of rows in Table 1 and Table 2 to PPM-Rips, PD-FL, PPM-FL (and to do the same for Table 2), to ease the navigation for the reader.

Some other questions:
- In Table 1 row 1, why is accuracy for $q=0&1$ lower than for $q=1$?
- In Table 1, do you have any intuition why is PPM-FL much better than PD-FL for $q=0$?
- Table 2 (and also Table 6) caption: Replace “Results” with “Accuracy”.
- Why is two-phase training needed for ModelNet10, but not for protein data?

(Q15) Section 6.2: As I write in the related question (Q11) above, here you could add a brief explanation why PPM-FL is more robust to outliers than PD-FL. Also, why is there no standard deviation in the results here (confidence bands around the curves)? I suggest to add these, and detailed experimental results (table with precise values of the accuracy and standard deviation) in an appendix.

(Q16) Section 6.3: 
- Similarly to Section 6.2, I think it would be better to replace the table results (which should be moved to an appendix) with plots, to make it clearer how the computational complexity grows with respect to point cloud size (e.g., quadratic vs. linear growth).
- When discussing Table 3, you can refer back to the new proposed theoretical section on the scalability of PPM, see (Q1)(3), i.e., how the theoretical computational complexity depends on $n$, to explain why the computational cost of PPM-FL increases in each column: for a fixed $M$, when $n$ increases (since, for instance, for $M=100$ and $q=1$, you are always sampling 100 subset point clouds with 4 points). The same is true for explaining why the runtime of PPM-FL increases in each row: for a fixed $n$, when $M$ increases (which can otherwise seem surprising since the algorithm is parallelized across $M$)?
- Table 5 is related to scalability (“scalability pf PPM-FL is achieved without sacrificing accuracy”), but is the same true for Table 4? In my opinion, Table 4 rather deals with some kind of stability of your approach across different values of $M$, that, indeed as you write, can give us some guidelines on how to choose this important parameter $M$ for your proposed PPM-FL approach. Should this discussion then not be placed elsewhere, where you are rather demonstrating that your approach is meaningful?
- Related to the last item above, you conclude that if $M$ is not large enough, the PPM-FL performance degrades, but we see this in Table 4 also when $M$ increases, why is this the case?

(Q17) The important last sentence in Section 6 does not only refer to Section 6.3, but rather summarizes the experimental findings in the complete section, so it should be rather placed in the introduction of this section, with specific pointers: “In summary, while both PPM-FL and PD-FL produce comparable results in point cloud classification task (Section 6.1), PPM-FL has better scalability (new Section 6.2) and is more robust to outliers (new Section 6.3) than PD-FL”.

(Q18) Appendix A:
- What is $(a, b, r_0)$-standard assumption?
- Why is a constant weight assumed in PersLay, can a better PH representation not be learned if one allows different PD points to be weighed differently? Also, a different notation for the weight should probably be introduced here, to differentiate it from the weight in your FL approach?
- What are $p$ and $m$?
- You conclude section A.2 with “the scalability issue of Expected Persistence Diagram (EPD), a more general form of PPM, has been discussed in detail in Bubenik et al. (2015) and Section 4.3.1 in Gomez & M´ emoli (2024).” Could you briefly summarize the main insights of these results, the scalability of EPD is crucial for your work?


(Q19) Appendix B:
- There is no Figure 16 in Gomez & Memoli (2024)? Consider rephrasing “In the caption …, it claims” to something like “The caption… states”.
- Should Appendix D.4 rather not become Appendix B.3 (with B.1 Proof and B.2 On the assumption…, and B renamed), as it relates to the theoretical robustness to outliers? 

(Q20) Appendix C: 
- Does it not make more sense to order the paragraphs chronologically, i.e., Datasets, Networks, Optimization and PPM? 
- What are the “parameters in PPM-FL”, e.g., the number of subsets $M$ is also a parameter of $PPM$? Be more precise.
- Consider replacing “$m$” to “the length of vectorization $m$”, and adding at least some brief info about the ModelNet10 data, e.g., at least the number of samples, features and classes. 

(Q21) Appendix D.3: What do the two considered approaches intuitively entail, how do they differ?

### Soundness
2

### Presentation
3

### Contribution
3
