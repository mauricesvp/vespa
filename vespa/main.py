"""
vespa - Vector Space Model
---

Background:
    https://en.wikipedia.org/wiki/Vector_space_model
    https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html

vespa supports all weighting options described here:
    https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class Vespa:
    TF_MODES = ["b", "n", "a", "l", "L", "d"]
    DF_MODES = ["n", "f", "t", "p"]
    NO_MODES = ["n", "c", "u", "b"]

    # Used for pivoted normalization,
    NO_SLOPE = 0.25

    def __init__(
        self,
        corpus: list,
        modes: str = "lnc.ltc",
    ) -> None:
        if type(corpus) is not list:
            raise ValueError("Corpus needs to be list of strings.")

        self._set_modes(modes)

        self.corpus(corpus)

    def corpus(self, *args: Union[List[str], str]) -> None:
        """Rebuild corpus."""
        # preserve original case sensitivity for returning documents
        self.documents: List[str] = []
        # lower case version used for scoring
        self.documents_lower: List[str] = []

        self.DF: Dict[str, int] = {}
        self.VD: Dict[str, Dict] = {}

        self.add(*args)

    def add(self, *args: Union[List[str], str]):  # type: ignore
        """Add document(s) to corpus."""
        for arg in args:
            if type(arg) is str:
                self.documents.append(arg)
            elif type(arg) is list:
                if all(isinstance(x, str) for x in arg):
                    self.documents += arg
                else:
                    raise ValueError("All documents need to be strings.")
            else:
                raise ValueError("All documents need to be strings.")

        self.documents_lower = [x.lower() for x in self.documents]

        self._build_df_vd()

    def k_score(self, query: str, k: int, modes: str = None) -> List[Tuple[float, str]]:
        """Return k best scoring documents for query.

        Does _not_ raise an error if k is larger than number of documents in corpus.
        """

        if k < 1:
            raise ValueError("k needs to be a positive integer!")

        if modes is not None:
            self._set_modes(modes)

        scores = []
        for document in self.documents:
            scores.append((self._score(query, document), document))

        # sort from best score to worst
        # sorted() will use first tuple value by default,
        # but we'll make it explicit just to be sure
        scores = sorted(scores, key=lambda x: x[0], reverse=True)

        return scores[:k]

    def score(
        self, query: str, document: str = None, modes: str = None
    ) -> Tuple[float, str]:
        """Return single best score, document for query."""
        if modes is not None:
            self._set_modes(modes)

        if document is not None:
            score = self._score(query=query, document=document)
            return (score, document)

        return self.k_score(query=query, k=1)[0]

    def _score(self, query: str, document: str) -> float:
        """Returns similarity score of query and document."""

        document = document.lower()

        if document not in self.documents_lower:
            raise ValueError("Document is not in corpus!")

        terms = set(self._q2t(query))

        # Optimization: Check if any query term is in document
        if not any(x in document for x in terms):
            return 0

        # Query Vector
        vec_query: Dict[str, float] = {}

        for t in terms:
            vec_query[t] = self._tfidf(t, query, mode_tf=self.q_tf, mode_df=self.q_df)

        # Document Vector
        vec_doc = None
        _d_mode_no_norm = self._d_mode[:-1]
        if _d_mode_no_norm in self.VD and document in self.VD[_d_mode_no_norm]:
            vec_doc = self.VD[_d_mode_no_norm][document]
        else:
            self._build_vd()
            vec_doc = self.VD[_d_mode_no_norm][document]

        # Vector dot product
        score = 0
        for k in vec_query:
            if k in vec_doc:
                score += vec_query[k] * vec_doc[k]

        if score == 0:
            return 0

        def get_num(vector: dict, mode: str) -> Optional[int]:
            if mode == "u":
                num = len(set(vector.keys()))
            elif mode == "b":
                num = len("".join(vector.keys()))
            else:
                return None
            return num

        # Normalize

        avg_query = None
        num_query = get_num(vec_query, mode=self.q_no)
        if self.q_no == "u":
            avg_query = self.avg_unique
        if self.q_no == "b":
            avg_query = self.avg_charlength

        avg_doc = None
        num_doc = get_num(vec_doc, mode=self.d_no)
        if self.d_no == "u":
            avg_doc = self.avg_unique
        if self.d_no == "b":
            avg_doc = self.avg_charlength

        norm_query = self._no(
            weights=list(vec_query.values()),
            mode=self.q_no,
            num=num_query,
            avg=avg_query,
        )
        norm_doc = self._no(vec_doc.values(), self.d_no, avg=avg_doc, num=num_doc)

        norm = norm_query * norm_doc

        if norm == 0:
            return 0

        return score / norm

    def _set_modes(self, modes: str) -> None:
        """Expects triplets in SMART Notation in the form 'ddd.qqq'.

        Example:
          'lnc.ltn'

        The first triplet denotes the document modes, the second the query modes.

        For more information see:
          https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
        """

        if not isinstance(modes, str):
            raise ValueError("Modes triplet needs to be a string !")

        if len(modes) != 7:
            raise ValueError(
                "Modes triplet needs to be exactly 7 characters long!"
                "(Example: 'lnc.ltn')"
            )

        self.d_tf = modes[0]
        self.d_df = modes[1]
        self.d_no = modes[2]

        self.q_tf = modes[4]
        self.q_df = modes[5]
        self.q_no = modes[6]

        def check_support(tf: str, df: str, no: str) -> None:
            if tf not in self.TF_MODES:
                raise ValueError(
                    "TF mode is not supported! Choose one of {n, l, a, b, L, d}."
                )
            if df not in self.DF_MODES:
                raise ValueError(
                    "DF mode is not supported! Choose one of {n, f, t, p}."
                )
            if no not in self.NO_MODES:
                raise ValueError(
                    "Norm. mode is not supported! Choose one of {n, c, u, b}"
                )

        check_support(self.d_tf, self.d_df, self.d_no)

        check_support(self.q_tf, self.q_df, self.q_no)

        tmp = list(modes)
        tmp[3] = "."
        modes = "".join(tmp)

        self.modes = modes

    @property
    def _q_mode(self) -> str:
        return self.modes[-3:]

    @property
    def _d_mode(self) -> str:
        return self.modes[:3]

    def _q2t(self, query: str) -> List[str]:
        """Return terms in query."""

        query = self._preprocess(query)

        return query.split(" ")

    def _preprocess(self, string: str) -> str:
        """Preprocess string."""

        string = string.lower()

        # Filter punctuation etc.
        string = re.sub(r"[^a-zA-Z0-9_ ]", "", string)

        return string

    def _build_df_vd(self) -> None:
        """Populate df and vd matrix."""

        _d_mode_no_norm = self._d_mode[:-1]
        if _d_mode_no_norm not in self.VD:
            self.VD[_d_mode_no_norm] = {}

        avg_unique = 0
        avg_charlength = 0

        for i, document in enumerate(self.documents_lower):
            vd = {}
            terms = set(self._q2t(document))

            avg_unique += len(terms)
            avg_charlength += len("".join(terms))

            for term in terms:
                if term in self.DF:
                    self.DF[term] += 1
                else:
                    self.DF[term] = 1

                vd[term] = self._tfidf(
                    term,
                    document,
                    mode_tf=self.d_tf,
                    mode_df=self.d_df,
                )

            self.VD[_d_mode_no_norm][document] = vd

        self.avg_unique = avg_unique / self.N
        self.avg_charlength = avg_charlength / self.N

    def _build_vd(self) -> None:
        if not hasattr(self, "VD"):
            self.VD = {}
        _d_mode_no_norm = self._d_mode[:-1]
        if _d_mode_no_norm not in self.VD:
            self.VD[_d_mode_no_norm] = {}

        for i, document in enumerate(self.documents_lower):
            vd = {}
            terms = set(self._q2t(document))

            for term in terms:
                vd[term] = self._tfidf(
                    term,
                    document,
                    mode_tf=self.d_tf,
                    mode_df=self.d_df,
                )

            self.VD[_d_mode_no_norm][document] = vd

    def _tfidf(self, term: str, document: str, mode_tf: str, mode_df: str) -> float:
        """Return tfidf weights."""

        max_t = None
        if mode_tf == "a":
            counter = Counter(self._q2t(document))
            max_t = counter.most_common(1)[0][1]

        ave_t = None
        if mode_tf == "L":
            counter = Counter(self._q2t(document))
            ave_t = sum([x[1] for x in counter.most_common()]) / len(counter)

        return self._tf(term, document, mode_tf, max_t, ave_t) * self._df(term, mode_df)

    def _df(self, term: str, mode: str) -> float:
        """Document Frequency.
        Returns number of occurances of term in corpus.
        """

        if mode not in self.DF_MODES:
            raise ValueError("Unsupported df mode!")

        # no
        if mode == "n":
            return 1

        if term not in self.DF or self.DF[term] == 0:
            return 0

        # idf
        if mode == "f":
            return math.log2(self.N / self.DF[term])
        # idf (zero-corrected)
        if mode == "t":
            return math.log2((self.N + 1) / self.DF[term])
        # probabilistic idf
        if mode == "p":
            df = self.DF[term]
            try:
                return max(0, math.log2((self.N - df) / df))
            except ValueError:
                if self.N == 1 and df == 1:
                    return 1
                return 0

        # Should never be reached
        raise ValueError("Unexpected Error!")

    def _tf(
        self,
        term: str,
        document: str,
        mode: str,
        max_t: int = None,
        ave_t: float = None,
    ) -> float:
        """Term Frequency.
        Returns number of occurances of word in corpus.
        """

        if mode not in self.TF_MODES:
            raise ValueError("Unsupported tf mode!")
        if mode == "a" and max_t is None:
            raise ValueError("Augmented mode needs max_t argument!")
        if mode == "L" and ave_t is None:
            raise ValueError("Log average mode needs ave_t argument!")

        tf = document.count(term)

        # boolean
        if mode == "b":
            return 1 if tf > 0 else 0
        # natural
        if mode == "n":
            return tf
        # logarithm
        if mode == "l":
            if tf == 0:
                return 1
            return 1 + math.log2(tf)
        # augmented
        if mode == "a" and max_t:
            return 0.5 + (0.5 * max_t)
        # log average
        if mode == "L" and tf is not None and ave_t is not None:
            if tf == 0:
                return 0
            return (1 + math.log2(tf)) / (1 + math.log2(ave_t))
        # double logarithm
        if mode == "d":
            return 1 + math.log2(1 + math.log2(tf))

        # Should never be reached
        raise ValueError("Unexpected Error!")

    def _no(
        self, weights: List[float], mode: str, num: float = None, avg: float = None
    ) -> float:
        """Normalization."""

        if mode not in self.NO_MODES:
            raise ValueError("Unsupported norm mode!")
        if mode in ["u", "b"] and (not num or not avg):
            raise ValueError(
                "Pivoted norm modes need num + average (unique/char-length) argument!"
            )

        # none
        if mode == "n":
            return 1

        # cosine
        if mode == "c":
            return math.sqrt(sum([weight**2 for weight in weights]))

        # pivoted {unique, char length}
        if (mode == "u" or mode == "b") and num and avg:
            assert avg != 0
            assert 0 <= self.NO_SLOPE <= 1
            return 1 - self.NO_SLOPE + self.NO_SLOPE * (num / avg)

        # Should never be reached
        raise ValueError("Unexpected Error!")

    @property
    def N(self) -> int:
        """Number of documents in corpus."""
        assert len(self.documents) == len(self.documents_lower)
        return len(self.documents)
