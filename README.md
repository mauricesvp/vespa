# vespa 🛵

Document Relevancy Ranking and Similarity Scoring using **Ve**ctor **Spa**ce Model.

Supporting all modes described [here](#modes).


## Installation

To install directly from github, run:

```sh
pip install git+ssh://git@github.com/mauricesvp/vespa.git
# or
pip install git+https://git@github.com/mauricesvp/vespa.git
```

To install from source:

```sh
git clone git@github.com:mauricesvp/vespa.git
# or
git clone https://github.com/mauricesvp/vespa.git

cd vespa
pip install .
```

## Usage

```python3
from vespa import Vespa

corpus = ["Example document."]  # corpus: list of documents (strings)
vsm = Vespa(corpus)

results = vsm.score("Example query")
# > (0.7071067811865475, 'Example document.')

results = vsm.k_score("Example query", k=1)
# > [(0.7071067811865475, 'Example document.')]
```

The default mode is `lnc.ltc`, which means `lnc` is applied to each corpus document, and `ltc` to each query document.
You can either supply a different mode when initializing, or to `k_score` or `score` directly (this *will* change the mode for subsequent calls).

If you want to get the score of a specific document, you can use the additional `document` argument for `score`:

```python3
results = vsm.score(query="Your query", document="Some document in corpus")
```

Documents can be added to the corpus:
```python3
vsm.add("some new document")  # str or list of str
```

or the corpus can be rebuilt, removing all previous entries:
```python3
vsm.corpus(new_corpus)  # str or list of str
```

## Modes

All available modes are noted below ([more details](https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System)).

|       | Term frequency  ![equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/9a90ffe8f10aedb2063d5ff8b3537f11d2757731) |       | Document frequency    ![equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/3aa1ed0c1f56acf127ab59dbe7917174e1632370) |       | Document length normalization   ![equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/67a1817f9610a2e039a2259be96b0739915c5ed0) | 
| ----- | ----------------------------------------------------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **b** | Binary weight                                                                                                                 | **n** | Disregards the collection frequency                                                                                                 | **n** | No document length normalization                                                                                                              |
| **n** | Raw term frequency                                                                                                            | **f** | Inverse collection frequency                                                                                                        | **c** | Cosine normalization                                                                                                                          |
| **a** | Augmented normalized frequency                                                                                                | **t** | Inverse collection frequency                                                                                                        | **u** | Pivoted unique normalization                                                                                                                  |
| **l** | Logarithm                                                                                                                     | **p** | Probabilistic inverse collection frequency                                                                                          | **b** | Pivoted characted length normalization                                                                                                        |
| **L** | Average-term-frequency-based normalization                                                                                    |       |                                                                                                                                     |       |                                                                                                                                               |
| **d** | Double logarithm                                                                                                              |       |                                                                                                                                     |       |                                                                                                                                            
## Limitations

Vespa does _not_ feature:

* Lemmatization and Stemming
* Stopword filtering
* Spelling correction
* Any kind of machine learning


## Background

For further reading, please reference:

* [Wikipedia Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model)
* [Information Retrieval Book](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html)
