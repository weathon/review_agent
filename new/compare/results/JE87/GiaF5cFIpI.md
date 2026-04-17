# Review

## Summary
The authors propose a real-time algorithm for modeling latent neural dynamics and designing neural stimulation to perturb along desired directions in the latent space. The proposed algorithm is demonstrated on simulated and real neural data.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
The paper is well motivated and organized. The authors address an important problem in computational neuroscience, i.e., adaptive closed-loop stimulation to perturb neural dynamics along desired directions.

## Weaknesses
1. The authors propose several methods for latent space identification and neural dynamic modeling, and they can be used in a plug-and-play manner. However, the paper lacks a systematic comparison between these methods, and it is unclear how the choice of latent space or dynamic modeling affects the downstream stimulation design. 
2. The authors propose several methods for latent space identification and neural dynamic modeling, and they can be used in a plug-and-play manner. However, the paper lacks a systematic comparison between these methods, and it is unclear how the choice of latent space or dynamic modeling affects the downstream stimulation design. 
3. The proposed method is demonstrated on a toy model and two real datasets, but the analysis is rather limited. For example, in Figure 3, it is unclear how the model performs across different trials. Moreover, the authors simulate the effect of stimulation, but it is unclear how the proposed method would perform in the absence of such simulation. 
4. The authors consider several infeasible stimulation targets and demonstrate that the proposed method can identify the infeasible solutions. However, it is unclear how often the proposed method would identify infeasible solutions in practice when the true underlying ground-truth is unknown. 
5. The proposed method is motivated by the need for designing stimulation in real-time, but the real-time performance is not demonstrated in the paper.

## Questions
1. How do different latent space identification and neural dynamic modeling methods affect the downstream stimulation design?
2. How does the proposed method perform across different trials in the presence and absence of simulated stimulation?
3. How often would the proposed method identify infeasible solutions in practice when the true underlying ground-truth is unknown?
4. How does the proposed method perform in real-time?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4