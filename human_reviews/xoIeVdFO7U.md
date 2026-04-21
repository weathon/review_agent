# Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning

- Avg Score: 7.50
- Decision: Accept (Oral)
- Scores: 8, 6, 8, 8

## Abstract
Self-supervised learning has the potential of lifting several of the key challenges in reinforcement learning today, such as exploration, representation learning, and reward design. Recent work (METRA) has effectively argued that moving away from mutual information and instead optimizing a certain Wasserstein distance is important for good performance. In this paper, we argue that the benefits seen in that paper can largely be explained within the existing framework of mutual information skill learning (MISL).
Our analysis suggests a new MISL method (contrastive successor features) that retains the excellent performance of METRA with fewer moving parts, and highlights connections between skill learning, contrastive representation learning, and successor features. Finally, through careful ablation studies, we provide further insight into some of the key ingredients for both our method and METRA.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a novel self-supervised skill learning algorithm in the RL setting. Their work is motivated by recent work (METRA) that suggests moving away from the typical MI setting, which they analyse, and show that it could be reinterpreted in the familiar MI setting. In doing so, they create a simplified version of METRA, CSF, which achieves the same performance as METRA. The authors combine both theoretical and empirical evidence to support their claims.

### Strengths
This paper is incredibly well-written. The message/purpose is clear. There is an extensive literature survey, and the authors' discussion of the relevant material and methodology is informative. Unlike some papers that resort to mathematics unnecessarily, all components seem necessary, and are meaningfully explained in text. The authors perform a relatively large set of experiments to both show the performance of their algorithm, but also to back up other claims (such as the properties of representations learned). I also commend the authors for the extended information in the appendices, and also for providing code (I checked these briefly but not extensively).

### Weaknesses
I do not believe there are any substantive weaknesses of this work, but there are some questions the authors could address. As acknowledged by the authors in their own Limitations section, it is unclear how well these algorithms scale beyond the relatively simple MuJoCo benchmarks, but the authors have performed a significant amount of experiments on these domains.

### Questions
- Where did the "fixed coefficient ξ = 5" come from/does it have any interpretation? How sensitive is CSR to this hyperparameter?
- There is a note that there are many possible parameterisations for the optimisation of a lower bound on I(S, S′; Z), but CSR (and METRA) use (ϕ(s′)−ϕ(s))⊤z; an ablation study shows that this is a crucial choice. Can the authors provide a further comment on the importance of the temporal distance metric?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a new approach for learning different locomotor skills without supervision. The paper explains relation between SOTA method (METRA) with mutual information maximization between skills and state transitions, an objective shared by most related works. Following the analysis, it proposes modifications to METRA to make it explicitly maximize such a mutual information. Experiments show that the new approach matches METRA.

Overall, the paper is easy to follow and well-written, the analysis is sound and the experiments are relevant. The main limitation is the novelty of the final approach, which is, in the end, very close to METRA and CIC. It also does not show strong improvements.

I'm balanced but a lean towards borderline reject, I did not find sufficiently strong arguments justifying the proposed approach, compared to METRA (see Questions).

### Strengths
The paper is easy to follow and well-written, the analysis is sound and the experiments are relevant.

### Weaknesses
The final model is very close to previous work and do not present substantial improvements on the different environment compared to METRA. This is overall acknowledged by the authors.

### Questions
The statement line 278 does not hold for all cited methods (like DIAYN), as the "omitted" is sometimes a useless constant. This is actually illustrated by Eq 2 and 3.

I also find the argument that the method removes 5 hyper-parameters compared to METRA to be bad faith: "(1) the ϵ slack variable, (2) the norm constraint value in Eq. 4, (3) the dual gradient descent learning rate, (4) the dual gradient descent optimizer, and (5) the choice of discrete or continuous skills z." The choice between discrete/continuous skills is actually a positive thing, the norm constraint value is not a hyper-parameter, and the Lagrange multiplier seems (as far as I understand) to use the same optimizer/learning rate as the rest of the parameters in the original paper. Could you clarify which hyper-parameter actually require tuning and how many hyper-parameters you introduce ?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work makes two major contributions. First, it establishes an approximate equivalence that bridges the representation objectives of the state-of-the-art method METRA with contrastive loss, specifically similar to InfoNCE. It shows that the actor objective in METRA is equivalent to the information bottleneck of $I(S, S'; Z) - I(S, S'; \phi(S') - \phi(S))$ (lower bounded). Essentially, it uses mutual-information-based skill discovery to elucidate METRA. Building on the analytical framework, the authors propose contrastive successor features to simplify METRA, employing contrastive objectives for representation learning and successor features for policy learning. Results indicate that the proposed method is empirically competitive with METRA. 

Overall, I like the analytical framework that unifies METRA with the mutual-information-based skill discovery method, and the theoretical foundation appears solid. Additionally, the results support the hypotheses and propositions, suggesting that the proposed method is even more flexible than METRA. Given these strengths, I would recommend an accept in this initial review.

### Strengths
- **[Technical soundness and novelty]** The technical soundness is robust; this work provides a thorough in-depth analysis of the METRA method, finding approximate equivalences with contrastive objectives and the information bottleneck. The analysis leads to a novel method that simplifies METRA, and I found no technical flaws; the method is both novel and solid.


- **[Evaluation]** The empirical evaluation effectively validates the hypotheses and theoretical analysis, enhancing the overall persuasiveness of the work.

- **[Presentation]** The presentation is clear and easy to follow.

### Weaknesses
- **[About performances]** A significant question arises especially in the Quadruped experiments, where performance still shows room for improvement compared to METRA. Given that the proposed framework has a similar objective function, fewer hyperparameters, and avoids complex min-max optimization, why does the empirical performance (or at least the rate of convergence) not exceed that of METRA? Any discussion on this would be beneficial.
- **[About demonstrations]** It would be advantageous to include demonstrations or other forms of visualization for the skills learned, as I did not find this in the appendix code.

### Questions
- Are there any theoretical insights supporting the assertion that parameterization is key for contrastive successor features? Did you experiment with different kernel functions?

- Any insights or rough ideas on how to further scale the framework to more complicated cases, perhaps through large-scale pre-training or using foundation models to replace the representation learning component, would be beneficial to include in the main paper

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper critiques a recent method (METRA), which optimizes a Wasserstein distance for skill learning, and argues that its benefits can be explained within the existing framework of mutual information skill learning (MISL). The authors propose a new MISL method called Contrastive Successor Features (CSF), which retains METRA's performance with fewer complexities (namely fewer hyperparameters but same performance).
The paper highlights connections between skill learning, contrastive representation learning, and successor features, and provides insights through ablation studies.

### Strengths
The paper provides a thorough theoretical analysis of METRA, reinterpreting it within the mutual information skill learning (MISL) framework. This helps demystify the method and connects it to well-established concepts like contrastive learning and information bottlenecks.

The presentation is clear and to the point. The writing is excellent, I did not find typos or mistakes.

The paper includes extensive empirical evaluations, comparing CSF with existing methods across various tasks. This robust experimental setup strengthens the validity of the proposed method.

### Weaknesses
While I appreciate that the paper is presented as an improvement on METRA, I'd have enjoyed more a reading that was presenting a new method that is then shown to be equivalent to METRA under certain conditions.

Given that the presented method performs are par with METRA, it would also be nice to show where (if anywhere) one fails when the other succeeds. Perhaps partially observed MPDs, more interactive objects or discrete actions spaces would be key in identifying where exactly both methods stand.

### Questions
- Can the authors elaborate on the assumptions made in the theoretical analysis, and how they might affect the generalizability of the results? It is not clear from the main text if these assumptions are sound and/or restrictive in any significant way.

### Soundness
4

### Presentation
4

### Contribution
4
