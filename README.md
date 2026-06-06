# Hidden in the Vectors

## This repository contains:
| Documents: | Description |
|---|---|
| `Purpose Demo.py` | Loads GloVe, demonstrate cosine similarity, runs the bias demonstrations. |
| `glove.6B.100d.txt` | developed at Stanford University  |
| `README.md` | Contains: purpose, context, and sources so far. |

Please **Note that `glove.6B.100d.txt`:** is the 100-dimensional file from Stanford's glove.6B set 
(Pennington et al., 2014), released under the Public Domain Dedication and Licence (PDDL v1.0). 
For this purpose demonstration I used the **100d**. To run this python file, please download it 
directly from <https://nlp.stanford.edu/projects/glove/> and place the extracted file next to the
Purpose Demo.py.

## I. How embeddings work in AI and the "Echo-Chamber effect" is the spark that motivate me to pursue this project.

Embedding is a load-bearing infrastructure in tools that we use daily. It converts words
into vectors so that machines can understands, and it empowers search engines, resume-screening tools,
translation tools as well as other powerful algorithm used in content recommendations. Because it is
pre-trained once and then used everywhere, the flaws inside embedding spreads widely onto
any system that imports it. 

My biggest fear is the echo-chamber effect that comes with it. Imagine if a model learns associations
from human text (men <-> tech, women <-> caregiving), and the model is deployed into a tool that 
shpaes what people see, and who will get shortlisted in algorithms recommending applicant to tech
employers. Those outputs has real power to influence decision making that could impact lives.
Bolukbasi et al. (2016) name exactly this: biased embeddings don't just reflect stereotypes, their 
widespread use "tends to amplify these biases.

**Why it matters: it is a wellbeing and equity issue, not just a technical one.** 
If a biased embedding feeds into an automated hiring filter, it can quietly down-rank a certain group
of people for technical roles; when it feeds a search or translation system, it can reinforce who is
assumed to be a doctor and who is assumed to be a nurse. These are *representational harms*: harms 
that arise simply from a system encoding a group in a demeaning or stereotyped way (Blodgett et al., 2020).
It affects people's access to opportunity and their dignity.

**What this program contributes.** I aim to raise awareness in these tool. And explore how embeddings 
can reveal patterns about culture and history through text. Surfacing the bias because it is the first 
step toward mitigating these type of problems from happening in the first place.

## A worked breakdown of GloVe 

### What a word vector is

GloVe (Global Vectors for Word Representation) is an **unsupervised** algorithm that learns a dense numeric 
vector for every word from how often words co-occur in a large corpus (Pennington et al., 2014). In the 100d file, 
every word becomes a point in 100-dimensional space for example:

```
king   →  [ 0.0033, -0.2407,  0.3984,... ]   (100 numbers)
queen  →  [ 0.1376, -0.2459,  0.4001,... ]   (100 numbers)
```

Words used in similar contexts end up close together; while unrelated words end up far apart. 
The direction in this space carries meaning.

### Finding closeness with cosine similarity

Formula:

$$
\cos(A, B) = \frac{A \cdot B}{\lVert A \rVert \, \lVert B \rVert} = \frac{\sum_i A_i B_i}{\sqrt{\sum_i A_i^2} \, \sqrt{\sum_i B_i^2}}
$$

- It ranges from {-1 to 1}.
- It cares about *direction*, not length, so it captures meaning rather than the frequency of the word.

Step-by-step example with 3D vectors say A = (1, 2, 3) and B = (2, 0, 4):

```
dot product: 1·2 + 2·0 + 3·4 = 14
||A|| : sqrt(1 + 4 + 9) = sqrt(14) ≈ 3.742
||B|| : sqrt(4 + 0 + 16) = sqrt(20) ≈ 4.472
cosine : 14 / (3.742 × 4.472) ≈ 0.837
```

On the real 100-D vectors as shown in the program does exactly this, just over 100 components.
Here are the results from `glove.6B.100d`:

| Pair | Cosine similarity | Reading |
|---|---|---|
| king <-> queen | **≈ 0.751** | high, signifying both royalty, similar contexts |
| king <-> potato | **≈ 0.184** | low, unrelated |
| doctor <-> physician | **≈ 0.767** | high, almost synonyms |

### Vector arithmetic

GloVe's headline property is that relationships are directional, you can add and subtract (Pennington et al., 2014). The classic analogy:

```
woman - man + king   ≈   queen
```
This can be itnerpret as *"man is to king as woman is to ___?"*. Geometrically, woman - man isolates a gender-shift direction; 
adding it to `king` lands near `queen`.

### The same maths, now surfacing bias

Nothing about the method changes for the next two queries — only the words do:

| Vector operation | Result | What it reveals |
|---|---|---|
|`doctor - male + female` | **nurse** | gendered occupation stereotype |
|`female - male + programmer` | **animator** | technical role shifted away from "female" |

A fair model would ideally return *doctor* and *programmer* the same professions. It returns *nurse* and *animator* because 
the training text statistically tied those words to gender that way. The maths is neutral; the bias lives in the data the model
was fed. That is precisely the point the program makes visible.

## References 

Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: 
  A critical survey of "bias" in NLP. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* 
  (pp. 5454–5476). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.485

Bolukbasi, T., Chang, K.-W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). Man is to computer programmer as woman is to homemaker?
  Debiasing word embeddings. In *Advances in Neural Information Processing Systems* (Vol. 29, pp. 4349–4357). Curran Associates.
  https://arxiv.org/abs/1607.06520

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In *Proceedings of the 2014 
  Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543). Association for Computational Linguistics.
  https://doi.org/10.3115/v1/D14-1162

## Resources:
Chollet, F., & Allaire, J. J. (2018). *Deep learning with R* (Chapter 6). Manning Publications. 

The GloVe data file is from Pennington et al. (2014), distributed under the Open Data Commons Public Domain Dedication and Licence (PDDL) v1.0.
