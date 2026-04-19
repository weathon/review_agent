# PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
It is widely known that state-of-the-art machine learning models — including vision and language ones — can be seriously compromised by adversarial perturbations, so it is also increasingly relevant to develop capability to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks, with population level risk guarantees. In particular, given a specific attack, we introduce the notion of a $(\alpha,\zeta)$ machine learning model safety guarantee: this guarantee, which is supported by a testing procedure based on the availability of a calibration set, entails one will only declare that a machine learning model adversarial (population) risk is less than $\alpha$ (i.e. the model is safe) given that the model adversarial (population) risk is higher than $\alpha$ (i.e. the model is in fact unsafe), with probability less than $\zeta$. We  also propose Bayesian optimization algorithms to determine very efficiently whether or not a machine learning model is  $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with their statistical guarantees. We apply our framework to a range of machine learning models — including various sizes of vision Transformer (ViT) and ResNet models — impaired by a variety of adversarial attacks such as AutoAttack, SquareAttack and  natural evolution strategy attack, in order to illustrate the merit of our approach. Of particular relevance, we show that ViT's are generally more robust to adversarial attacks than ResNets and ViT-large is more robust than smaller models. Overall, our approach goes beyond existing empirical adversarial risk based certification guarantees, paving  the way to more effective AI regulation based on rigorous (and provable) performance guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors both propose a new Bayesian mechanism for certifications, and attempt to demonstrate the relative performance of different architectures.

### Strengths
The presented framework is interesting, and the introduction as presented takes a very unique perspective on the reasons why there is a critical need for research in the field of AI security.

### Weaknesses
While the idea within this work is interesting, I do not believe it has suitable rigorous experimentation (especially in terms of dataset diversity), or experimentation (does not follow standard expectations regarding the trade off between certification proportion and size that are common in other certification papers). While there is validity in a paper that demonstrates that a new approach has the potential to extend the ability of certifications to new frontiers - however, part of doing this kind of validation would require a comprehensive set of experiments demonstrating scaling and performance, all of which are missing. 

Also one of the stated contributions of this work is to extend Randomised Smoothing from $\ell_2$ to $\ell_p$. However, this is missing a wide range of literature on $\ell_p$ certifications in randomised smoothing, see Yang et. al "Randomised Smoothing of All Shapes and Sizes", 2020 as an example of this. 

I also worry that this paper is attempting to cover quite a few bases - it's trying to both introduce a Bayesian optimisation mechanism and to demonstrate that VIT's are more robust than other architectures. But in attempting to cover both of these points I believe that neither contribution is sufficiently addressed - the Bayesian mechanism is insufficiently detailed, implementation details are sparse, and the range of experiments (including datasets and metrics) are sparse relative to the level of experimental evidence to truly make these points. 

As a few other notes:
- Figure 1 talks about "adversarial risk certification for various models under AutoAttack" - the involvement of AutoAttack is only tangentially referred to within the document. Figures 1 & 2 don't even seem to be referenced in text? So there's no 
- Just as a note on page 3 there's no space between "functions.Lipschitz" in the sentence relating to Wong & Kolter.
- Citation capitalisation is inconsistent - especially when it comes to the names of journals / conferences / venues. 
- The paper could do with algorithms and implementation details, even just in the appendices.

### Questions
What's the difference between the $(\alpha, \zeta)$ safety framework relative to something like Differential Privacy (as considered by Lecuyer)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces PROSAC, a method to certify a machine learning model's robustness against an adversarial attack type, regardless of the hyperparameters chosen for that adversarial attack. The claims are substantiated by experiments attacking a few vision models with benchmark attacks. This work also applies Guassian Process Upper Confidence Bound (GP-UCB) to hyperparameter selection during the certification process.

### Strengths
* (Moderate) The paper is well-motivated in showing the need for robustness certifications as ML models are used for increasingly critical areas and will likely be subject to more government regulations.

### Weaknesses
* (Major) The work's presentation overall is difficult for me to understand. This includes the use of undefined variables and terms in the writing and algorithms as well as hard to find experimental details. Details in questions.

* (Major) In certifying the space of attack hyperparameters a model is robust to, it is unclear what hyperparameters are being varied in each attack and what hyperparameter values the work is certifying are safe. Also, this work seems to omit relevant attack hyperparameters from the certification, including attack budget and constraint norm. 

* (Major) The experimental results in section 5 seem to suggest the certification is unreliable. For example, Figures 1 and 2 show very different p-values for only slightly different hyperparameter values where I would expect p-values to be similar for similar hyperparameter values. An example of this is Fig 2c showing epsilon 0.0011 having a p-value above 0.8, epsilon of 0.0012 having a p-value of 0.0, then epsilon of 0.0013 having a p-value of 1.

* (Moderate) Section 2 contains some confusing statements about prior work. Details in questions.

* (Moderate) Theorem 4 seems to say that multiple rounds of approximation are needed to certify a machine learning model, but it does not quantify or attempt to make a statement about that number of rounds of approximation. Later in the text, there seems to be a sentence saying "See Supplementary Material" for this information. However, as it is core to this work's claim, at least a summary of the math required to estimate how many rounds should be executed is required.

### Questions
* What is the relation between this work and PAC (Probably Approximately Correct) learning / adversarial PAC learning? It seems the guarantees are very similar to those proposed in this work.


* Regarding the results shown in Figures 1 and 2, I would have expected similar values of epsilon to have similar p-values, with a monotonic increase in p-value as epsilon increases. Why is this not the case and why is there so much variability in p-values? These large differences indicates an unreliable certification since a very small decrease or increase of epsilon can change the p-value from 0 to 1 (as between epsilon = 0.0012 and 0.0013 in Fig 2c).


* Section 2 states "... RS (randomized smoothing) is limited  to certifying empirical risk of a machine learning model on pre-defined test datasets under $l_2$-norm bounded adversarial perturbations." What is meant by a pre-defined test dataset and how is randomized smoothing restricted to it? How is PROSAC not restricted to it?

* Section 2 states "randomized smoothing (RS) represents a versatile certification methodology free from model architectural constraints or model parameters access" but then later states "Our certification framework shares RS’s versatility but a) it also exhibits the ability to accommodate a diverse range of lp norm-based adversarial perturbations; b) it is not restricted to particular model architectures...". These statements seem to first state that RS is free from model architectural constraints but then states it is restricted to particular model architectures. Do I misunderstand what is being said?

* In Table 1, why is the hyperparameter field "N.A." for AutoAttack? There are several hyperparameters that can be set (e.g., number of gradient steps, number of expecation over transformation estimates, the type of attack to execute, etc.).


* Equation 10 is difficult for me to understand. What is $h_1$? Could a narrative be given for what this equation is saying?

* In section 4.1, a footnote says that the attack budget and norm are not considered hyper-parameters because it would not be possible to control the risk if the adversary can choose any attack budget. This justification does not explain why the norm is not considered a hyper-parameter. Is there a reason the norm is not considered a hyperparameter?

* In equation 7, shouldn't the risk no longer have a $\lambda$ subscript?

* In the paragraph below equation 5, I don't understand what is meant by "We will be assuming in the sequel, where appropriate,..." and later "We will be representing in the sequel..." What is the meaning of the word sequel here?... 

* In algorithm 1, what is $\beta$ and what is $k$? Is $\mu$ and $\sigma$ the mean and variance of $\lambda$?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper derives provable statistical guarantees on the adversarial population risk given an attack algorithm, by computing p-values. This paper also uses a Gaussian Process Upper Confidence Bound (GP-UCB) algorithm for certification against attacks with set of hyperparameter configurations.

### Strengths
* This paper derives statistical guarantees on the adversarial *population* risk, which is different from many previous methods for machine learning certification. 
* This paper considers that the attack algorithm is known, but it allows the hyperparameters of the attack algorithm to vary within a set of configurations, which is different from previous works on the population risk.
* The proposed method is independent from model architectures, and the experiments applied the proposed method on models including ViT and ResNet.

### Weaknesses
* There are lots of existing works on machine learning's robustness certification. Those works have been mentioned in Section 2, but that is probably too late. The first section does not mention the robustness certification works which are not about population risks. It is unclear from the beginning of the paper how this work differs from the previous works, and title is also not sufficiently informative. 
* Compared to the existing machine learning certification algorithms that are independent from attack algorithms, the one proposed in this paper requires a specific attack algorithm, which is not applicable when an attacker uses a different attack algorithm but still follows the same threat model. Thus, the importance of such a certification scheme is unclear.
* It is unclear what the computational cost is, and how the number of data samples may affect the results.

### Questions
* See the last weakness point.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This submission proposes a new approach to certify the performance of machine learning models against adversarial attacks, in the sense of asserting the model's population risk is lower than some threshold with high probability for a range of hyperparameters of a given attack. Experiments on a few image-based models and both white and black attacks demonstrate the effectiveness of the proposed approach.

### Strengths
- The rigorous risk control is a critical problem in trustworthy machine learning, given the legal requirements. The submission tackles this problem with an effective approach.

- The proposed approach is scalable in terms of certifying large ViT models, and experiments cover a wide spectrum of attack methods.

### Weaknesses
- The submission may not be rigorous enough. Especially, Theorem 4 only states that "we can do sth by relying on Alg 1". But how is the Algorithm 1's result used to derive the final guarantee in Eqn. (12)? As a certification approach, this process needs to be made more clear. Furthermore, GP-UCB provides maximized $p$ value under some latent assumptions if I understand Appendix D correctly. If this is the case, such assumptions should be inherited in the main theorem under which the certification holds.

- The experimental evaluation is not quite clear and may lack some baselines. For inference, on page 8, the submission has the text "We use $\alpha = 0.10$ and $\zeta = 0.05%$ in the safety certification". However, most results in the paper are presented in terms of $p$ value. How are $p$ values connected with these fixed certification parameters?

- The certification may be a bit limited compared to other $L_p$-norm-based certification, where this work can only guarantee the population risk for a certain type of attack but the existing literature can guarantee the risk for any attack within the perturbation budget. These constraints may need to be made clear.

Minor typos:
1. On Page 3, "relies on ReLU activation"
2. On Page 6, "Fix the machine learning model M, and fix ..."
3. On Page 9, "to the default one in Croce & Hein ..."

### Questions
See the questions and suggestions above.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
3 good
