## Human Reviewer 1

### Summary
This paper proposes a computationally efficient privacy auditing procedure by leveraging the f-DP curve, and shows that the resulting lower bounds are tighter than those of previous work.

### Strengths
The paper is well-motivated and, for the most part, clearly written. It provides a notable improvement over prior privacy auditing techniques.

### Weaknesses
The paper contains some ambiguities and cosmetic errors that should be addressed to improve clarity and overall presentation.
1) clarify that by "one run" you mean a training run (rather than an inference run)
2) explicitly state the limitation of Steinke et al. (2023) that you are addressing (in Line 80-82)
3) change the references to algorithm 3.1 to algorithm 3 (given that that is what the algorithm is called)
4) remove double mentions of Steinke et al. by just using the reference instead (e.g., in Line 420)
5) fix the reading flow in Definition 6 (second bullet point is not a full sentence)
6) correct typos (e.g., Line 194/195, 307, 466, 505/506) and wrong capitalizations in the middle of sentences (e.g., Line 100)

### Questions
1) What do you mean by "gubernatorial analysis"? (Line 95)
2) Do you have an intuition why the bound in Steinke et al. (2023) degrades with higher numbers of canaries while your bounds continue to improve?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper presents a novel algorithm designed to audit $f$-DP guarantees within a single execution of a mechanism.
This area of research has become increasingly significant within the privacy community, particularly due to the limitations of existing auditing mechanisms.
Existing empirical auditing methods are either computationally expensive (requiring multiple runs of the machine learning algorithm) or fall short in providing a tight empirical privacy guarantee.
The need to run the mechanism multiple times has hindered practical applications.
Steinke et al. (2023) introduced a pioneering approach that balances the number of runs with the tightness of the audit.
This present work enhances this trade-off further by auditing $f$-DP guarantees, which provide a more precise representation of a mechanism's privacy compared to traditional approximate DP parameters.

### Strengths
- Valuable Contribution to Existing Research: There has been extensive work on auditing differential privacy guarantees. This paper distinguishes itself by offering a solution that enhances both computational efficiency and the precision of empirical privacy guarantees. The reliance on multiple runs of the mechanism has been a major obstacle to the widespread application of auditing methods. Their approach, requiring only a single run, makes auditing significantly more practical, especially for complex machine-learning algorithms involving extensive model training.
- Using the $f$-DP framework is a particularly strong aspect of this work. $f$-DP offers a more general and accurate representation of a mechanism's privacy compared to traditional approximate differential privacy. This choice allows for a more fine-grained and robust analysis of privacy. The authors convincingly demonstrate that auditing $f$-DP leads to tighter empirical privacy assessments. By performing the analysis in a single training run, the paper achieves a more comprehensive understanding of the privacy implications within a practical computational framework.

### Weaknesses
- The main weakness of this paper is its presentation. The write-up seems very rushed which at times hinders the flow of the reader. Many references are broken e.g. reference to Algorithm B. Lines 300-312 contain many typos and incomplete sentences. These are issues that can be addressed quickly but in the current state I would argue that the presentations limits the value of this work to the community.
- The authors have not provided a code artifact. While the contributions of this work are mostly theoretical, the implementation of the algorithm requires care and it would help reproducibility if a code artifact were supplied.

### Questions
- On the section “Empirical Privacy” line no 307, why do the trade off curves need to be ordered? If you have a set of trade off curves $f_i$ that pass couldn’t you build a new trade off curve $f(x) = \min_i f_i(x)$ 
- In what sense are the empirical results tight in Fig 7 and why is that not also evident in Fig 1?
- Can you explain why abstentions are important in this algorithm?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper presents a novel algorithm for auditing differential privacy (DP) mechanisms in a single run, building upon and extending the work of Steinke et al. (2023). By leveraging the theoretical framework of f-Differential Privacy (f-DP), the authors provide tighter lower bounds on privacy leakage, thereby enhancing the existing toolbox for f-DP analysis. Notably, their auditing algorithm can be applied to various adversaries beyond the standard membership inference, such as reconstruction attacks.

### Strengths
* **Advancement of f-DP Tools**: The paper contributes to the understanding and practical application of f-DP, which could be of independent interest.
* **Interesting Problem**: Auditing DP mechanisms in a one-run scenario is interesting for practical implementations (particularly in the black-box scenario, see the weakness section), and the paper makes significant progress in this area.
* **Experimental Validation**: The experimental results are compelling and demonstrate the effectiveness of the proposed approach.
* **Versatility in Adversarial Models**: Extending the auditing algorithm to handle different adversaries, such as reconstruction attacks.

### Weaknesses
The authors investigate exciting problems and provide interesting results. I encourage the authors to continue working on these results, as they are sound and exciting to the DP community. However, I don’t think the work is ready to be published in its current form, as it is somewhat rushed. I sketch my main concerns below.
* **Writing and Presentation Quality**: The manuscript contains several errors and unclear explanations. The authors should revise it before publication, as there are plenty of writing errors and bad citing style.
* **Unreferenced Figures and Results**: Some results, particularly those in Figure 7, need to be adequately referenced or explained within the text, leading to confusion about their significance.
* **Incomplete Explanation of Gaps**: The paper needs to explain the gaps between theoretical and lower bounds. Possible reasons for these gaps should be analysed, such as limitations of the f-DP framework, assumptions made in the analysis, or practical considerations in implementation.
* **Insufficient Experimental Details**: There are no experiments in the black-box setting for which we are compelled to use one-shot auditing. The white-box setting enjoys a tight and efficient auditing algorithm (Nasr et al., 2023), while the black-box algorithms are rather expensive.

### Questions
Questions:
* You claim that your approach achieves tighter results as the number of canaries increases, outperforming the empirical privacy results from Steinke et al. (2023), suggesting that the results can be tight as we increase the number of canaries. Could you elaborate on why your bounds continue to improve with more canaries while the bounds in previous work degrade? What underlying mechanisms in your algorithm contribute to this improvement? Citing the authors: ” Figure 1 demonstrates that our approach outperforms the empirical privacy results from Steinke et al. Interestingly, while the bound in Steinke et al. (2023) degrades as the number of canaries increases, our bounds continue to improve.”
* What potential sources contribute to any lack of tightness in your lower bounds? Are there specific aspects of the f-DP framework or your implementation that introduce looseness? How might these be addressed in future work to enhance the tightness of the bounds?
* How does your algorithm perform in the black-box setting compared to the white-box setting? Can you provide detailed experimental results illustrating this performance?

### Soundness
3

### Presentation
1

### Contribution
3

### Rating
3

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper proposes an approach for auditing the guarantees of a differentially-private algorithm, which in contrast to other existing auditing schemes, does not require re-training of the model. In addition, the approach provides tighter bounds that the related work by Steinke et al.

### Strengths
-The existing literature on privacy auditing is clearly reviewed as well as the main limitations of existing approaches. 

-The paper is well-written and easy to read. The authors have also provide a clear introduction of the notions necessary to the understanding of their work such as the f-DP curve.

-The proposed approach has the benefit of only requiring a single run of the mechanism. The relationship with the method of Steinke at al. is also clearly explained. One of the main novelty of the approach is also to connect the privacy auditing procedure with previous works bounding the accuracy of reconstruction attacks.

-The experiments conducted demonstrate that the approach proposed performed better in terms of the tightness of the bound estimated when the noise added is a a low to high regime.

### Weaknesses
-Some notions such as the concept of empirical privacy could have been more formally defined. Other important details are missing such as the relationship between the way canaries are designed and the quality of their possible reconstruction as well as the design of the function f that should be considered for the auditing. 

-The number of canaries needed for the experiments is very high and is likely to have a significant impact on the utility of the classifier learnt. While an experiment has been conducted with CIFAR-10 to measure the impact of the introduction of 5000 canaries more experiments should be conducted by varying the number of canaries to observe in a more fine-grained manner the impact of the introduction of canaries.

-The approach has been validated empirically only on one dataset. Additional experiments at least two other datasets should be conducted to validate how well the approach generalizes to other settings. 

-There are some minor typos that could be corrected in a future revised version. For instance, "(e.g., [1])." seems to refer to a different bibliography style. Other typos : "an attack algorthm A" should be "an attack algorithm A", "in Essene" should be "in essence" and "augumented multiplicity" should be "augmented multiplicity"

### Questions
-Can you discuss more how the design of the set of canaries impact the reconstruction games?

-Apart from the classical example of differential privacy, can you provide a few other examples of function f that could be audited using your framework?

-Can the approach be used on classifiers for other types of data such as for example tabular data?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4