# Toward Practical Learning-based Frequency Estimation without Ground Truth

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5, 3

## Abstract
Estimating the frequency of items on the high-volume, fast data stream has been extensively studied in many areas, such as database and network measurement. Traditional sketch algorithms only allow to give very rough estimates with limited memory cost, whereas some learning-augmented algorithms have been proposed recently, their offline framework requires actual frequencies that are challenging to access in general for training, and speed is too slow for real-time processing, despite the still coarse-grained accuracy. To this end, we propose a more practical learning-based estimation framework namely UCL-sketch, by following the line of equation-based sketch to estimate per-key frequencies. In a nutshell, there are two key techniques: online training via equivalent learning without ground truth, and highly scalable architecture with logical estimation buckets. We implemented experiments on both real-world and synthetic datasets. The results demonstrate that our method greatly outperforms existing state-of-the-art sketches regarding per-key accuracy and distribution, while preserving resource efficiency. Our code is attached in the supplementary material, and will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of learning-augmented frequency estimation on a stream. The goal is to maintain a sketch that can approximately report the frequency of the elements seen on the stream so far. A more recent line of work improves the accuracy/memory trade-off by augmenting the sketch with an ML component, trained on past frequency data.

The sketch proposed in this paper trains the ML component online as the stream is observed (as opposed to prior approaches that assumed access to an already trained ML component at the start of the stream). In order to avoid having to rely on unknown ground truth frequencies, the ML component is trained on approximate frequencies obtained by compressed sensing.

### Strengths
Training on the fly and without access to ground truth is indeed more realistic and useful than training on ground truth data prior to the beginning of the stream. The experimental results show that the proposed method performs favorably compared to baselines.

### Weaknesses
The analysis seems to yield weaker results than in prior works. In Hsu et al, Aamand et al, for example, the error guarantee applies directly to the implemented algorithm. Here, in theorem 2, the guarantee is given under the idealized assumption that the ML component learns a "perfect" solution $f^*$. This disregards the specific learning dynamics (particularly in an online setting) and the fact that such ideal $f^*$ might not be representable by the ML component at all. This doesn't render the analysis wrong or meaningless, albeit weaker in nature and less directly related to the method than in prior works on the problem.

### Questions
If the premise of theorem 2 is that $Af=Af^*$ (you use exact equality in some steps of the proof), why do you have $\|Al\|\approx0$ (where $l=f^*-f$) and not just $\|Al\|=0$ in other parts, and what does "$\approx$" precisely mean here anyway?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposed a new frequency estimation streaming algorithm within the framework of learning-augmented algorithms. Past work on learning-augmented frequency estimation (e.g., Hsu et al, 2019) uses a machine learned predictor of heavy hitters trained on past data to improve the performance of classic sketches such as CountMin and CountSketch. The approach in this paper is quite different. The starting point is equation-based sketches for frequency estimation which recover an estimate of the frequency vector of the stream by attempting to invert the sketching process (this is an underdetermined problem but one can hope to find sparse solutions). The authors posit that existing methods are too cumbersome and propose learning a function to produce a frequency estimate from the sketch. They propose to learn this method online as the stream arrives and choose a loss function based on a heavy-tailed assumption of the frequencies. They provide experimental evidence that their proposed sketch performs well in terms of error and computation compared to baselines.

### Strengths
The experimental performance of the proposed sketch is impressive. It is shown to have high throughput and achieve low error. Recovery time is good relative to other equation-based solutions without learning (though recovery using non-equation-based sketches is much simpler and likely much faster).

The idea of using learning to speed up the inversion process of equation-based skeches is interesting and fits nicely in the learning-augmented algorithms framework.

### Weaknesses
The paper is difficult to read. The goals of the algorithm are not made clear from the introduction and problems/solutions of the design process are introduced alongside each other in an ad hoc manner in reading the paper. There are many moving parts described in the paper, but the reader is not adequately guided through them. The algorithmic choices are generally not clearly motivated and  explanations for why the proposed solutions are the right ones are missing. I would suggest simplifying the paper into clear sections which each address a specific and well-motivated question about the design of the algorithm. For instance, an interesting question is how to incorporate learning to speed up the inversion procedure of equation-based sketches. Clearly and formally stating this problem and its solution could be a self-contained section. Another question is how to do this learning efficiently over the course of a stream. Separately defining these challenges, possible baselines, and your solution could constitute a second focus on the paper. Currently, all these components are muddled together in the text.

The experiments are missing a comparison to the improved learning augmented sketching algorithms presented in Aamand et al. (2023) That work demonstrates that simple tricks (such as truncating low-frequency esimates to zero) yields significant improvements over LCS. Both that paper and the original by Hsu et al. (2019) measure error in terms of expected error weighted by the frequency of keys, which is missing from the experimental results. This corresponds to a query distribution matching the distribution of the stream. Furthermore, in the Aamand et al. paper, they discuss parsimonious learning-augmented sketches which get the same guarantees while infrequently querying the machine learning oracle. This would have a significant affect on throughput as it is currently measured in this paper by running inference on an RNN at every timestep.

### Questions
- What are the sizes in bytes of the datasets?

 - What fraction of space is used for the different components of your algorithm for a fixed space budget?

 - Does the Bloom filter component of the algorithm limit the algorithm to have a minimum space requirement growing linearly with the number of unique keys?

 - Why is the throughput of CM and CS less than UCL? If appropriately implemented, I would suspect that CM and CS have very fast throughput. Is the training component of UCL counted in the throughput?

### Soundness
2

### Presentation
1

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
This paper presents an unsupervised learning-based frequency estimation algorithm UCL-sketch which is ground truth free, thus it can be learned in online scenarios and quickly response to streaming distribution drift. And UCL-sketch realizes highly scalable architecture with logical buckets sharing parameters across them. Experiments on real datasets demonstrate the UCL-sketch having better accuracy compared to baselines.

### Strengths
1. The paper is well written and organization is clear
2. Unsupervised online learning of estimating the frequency of items seems interesting and practical.
3. The experimental results highlight some benefits of the proposed solution.

### Weaknesses
1. As an unsupervised learned sketch, this paper seems to overlook some important related works, such as ICML19 [1], ICLR23[2], and TPAMI24 [3].  These studies also utilize unsupervised learning to construct learned sketches for summarizing data streams, which appears to be quite similar to the approach in this paper. It would be better if technical comparison on these works can be made to highlight the specific contributions of this work.

2. Although this paper demonstrates the effectiveness of UCL-sketch on certain error metrics, the error values appear relatively high. For example, in terms of ARE, several sketches cited in this paper generally achieve values below 1 with suitable memory settings. This expectation aligns with the definition of ARE, where an overall ARE of 1 would result if all predictions were zero. Consequently, an ARE exceeding 1 may lack practical significance. The authors should consider adjusting the memory settings to bring the error into a more reasonable range for evaluation.

[1]Rae et al. "Meta-learning neural bloom filters. " International Conference on Machine Learning. PMLR, 2019.

[2]Feng, et al. "Mayfly: a Neural Data Structure for Graph Stream Summarization." The Twelfth International Conference on Learning Representations. 2023.

[3]Cao, et al. "Learning to Sketch: A Neural Approach to Item Frequency Estimation in Streaming Data." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

### Questions
Please address weakness points 1-2.

What are the specific values of AAE, ARE, and WMRD errors under different data streams when all predictions are set to 0 or to the mean frequency?

Others:
There is a typo in the statements of contributions: 'UCL-sketch' is mistakenly written as 'ULC-sketch.'

### Soundness
2

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
5

### Summary
The paper studies learning-augmented algorithms for frequency estimation in the streaming model. As far as I can tell (see more below), the main idea seems to be training a heavy-hitter prediction as the streaming algorithm progresses. This is an interesting idea and departs from the methodology used in prior works, which assume a heavy-hitter prediction oracle that has been already trained on past data (and whose space does not need to be accounted for as it will be `ammortized' across many data streams).

### Strengths
See below.

### Weaknesses
Unfortunately I believe there are some fundamental misunderstandings in the paper, which I hope to clarify below.

1) Understanding of prior works: It is claimed that all sketching based frequency estimation algorithms can all be captured by solving a linear system. This is simply incorrect, even for the two most famous examples, CountMin and CountSketch. While it is true that they are both linear sketches in the sense that their updates are just linear equations, the estimation part of the algorithms are not linear! Take CountMin for example. CountMin hashes the universe into a small number of buckets and repeats this independently (this part is all linear). But when it comes to the estimation phase (the authors call it recovery), it is crucial for the theoretical guarantees that one takes the *minimum* estimated frequency across all independent trials. This is not a linear operation. Similarly for CountSketch, we need to take the median across independent trials. 

2) Space complexity: I am not sure if the theoretical guarantees of the proposed algorithm are meaningful. The space useage seems to be *linear* in the domain size. E.g. on line 349 K seems to be the size of the universe and the algorithm uses linear in $K$ space. Thus, theoreticaly, this offers no advantage over just keeping all the exact frequencies in memory. 

2) Error metric: The authors do not seem to realize that prior learning-augmented works for frequency estimation are not using the standard point-wise error (they are not bounding the worst error across all frequency estimates). Rather, they are bounding a notion of weighted error (see Eq. 3.1 in Hsu et al https://openreview.net/pdf?id=r1lohoCqY7). This is also true for the follow up works, such as Aamand et al (NeurIPS 23). This does not seem to be the case here. That is totally fine, as the total sum of error used in this work is also natural. But, this means that directly comparing to prior work as done in Table 1 does not make sense. Furthermore, the error bounds derived for prior work, as in Aamand et al., are specialized to the Zipfian frequency case and are also normalized (so the frequencies add to 1). Neither of those are true for this paper so I am confused why the authors are directly comparing to such bounds in Table 1. 

3) Empirical evaluations: The paper fails to compare against the state of the art learning-augmented algorithms of Aamand et al. (NeurIPS 23). 

4) Writing: Finally, I don't really understand what the exact proposed algorithm is. I've tried to read the submission several times and while the high level idea is intriguing, there are no clear details given in the submission. A formal algorithm block in the main body with all parts self-contained would be helpful towards this. In the theoretical section, there are many parameters used with no explanations. For example, what are $\epsilon_c, \delta_c$ in line 346? 

5) The authors do not seem to realize that countmin and countsketch also have guarantees based on the `tail' contribution of the frequency vectors. E.g. see Lemma 4.1.5 here (https://www.sketchingbigdata.org/fall20/lec/notes.pdf). This is similar to what Theorem 2 is claiming (I think), but using sublinear space (whereas the submission seems to use linear in the universe size space ...). 

Given these points, while the high level idea seems promising, it is hard for me to judge the contribution of this paper and recommend acceptance.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
