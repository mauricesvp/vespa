import itertools

import pytest

from vespa import Vespa


def _is_close(a: float, b: float) -> bool:
    """Return true if a and b are closer than eps."""
    eps = 1e-9

    return abs(a - b) <= eps * max(abs(a), abs(b))


def test_main_dummy() -> None:
    corpus = [
        "Example document with loads of random words not making any sense whatsoever.",
        "Some other document i guess.",
        "More vocab foo",
        "dummy",
    ]
    corpus += ["stress test document"] * 5000
    vsm = Vespa(corpus)
    k = 4
    result = vsm.k_score("example query document foo", k)

    assert len(result) == k

    for r in result:
        assert 0 <= r[0] <= 1
        assert r[1] in corpus

    assert sorted(result, reverse=True) == result


def test_main_zero() -> None:
    corpus = [
        "Example document with loads of random words not making any sense whatsoever.",
        "Some other document i guess.",
        "Some other document.",
        "More vocab foo",
        "dummy",
    ]
    corpus += ["stress test document"] * 5000
    vsm = Vespa(corpus)
    k = 7
    result = vsm.k_score("no results query", k)
    assert len(result) == k
    for r in result:
        assert r[0] == 0.0
        assert r[1] in corpus


def test_score_case() -> None:
    corpus = ["Example document.", "Some other document."]
    vsm = Vespa(corpus)

    score, doc = vsm.score("lower case example")

    assert score > 0.1
    assert doc == corpus[0]


def test_main_q2t() -> None:
    corpus = ["Example document.", "Some other document."]
    vsm = Vespa(corpus)

    assert vsm._q2t("Just a query test. (test)") == [
        "just",
        "a",
        "query",
        "test",
        "test",
    ]


def test_main_init_modes() -> None:
    corpus = ["Example document.", "Some other document."]
    vsm = Vespa(corpus, modes="Ltb.dfc")

    assert vsm._q2t("Just a query test. (test)") == [
        "just",
        "a",
        "query",
        "test",
        "test",
    ]


def test_score_same() -> None:
    document = "some sample document with many words"
    corpus = [document]

    vsm = Vespa(corpus)

    score, result_doc = vsm.score(document)

    assert _is_close(score, 1)
    assert result_doc == document


def test_high_k() -> None:
    document = "some sample document with many words"
    corpus = [document]

    vsm = Vespa(corpus)

    score, result_doc = vsm.score(document)
    k_score, k_result_doc = vsm.k_score(query=document, k=3)[0]
    internal_score = vsm._score(query=document, document=document)

    assert score == k_score
    assert internal_score == k_score
    assert result_doc == k_result_doc


def test_score_k() -> None:
    document = "some sample document with many words"
    corpus = [document]

    vsm = Vespa(corpus)

    score, result_doc = vsm.score(document)
    k_score, k_result_doc = vsm.k_score(query=document, k=1)[0]
    internal_score = vsm._score(query=document, document=document)

    assert score == k_score
    assert internal_score == k_score
    assert result_doc == k_result_doc


def test_faulty_modes() -> None:
    vsm = Vespa(["sample"])

    with pytest.raises(ValueError):
        vsm._set_modes("toolongstring")
    with pytest.raises(ValueError):
        vsm._set_modes("short")
    with pytest.raises(ValueError):
        vsm._set_modes("xxx.xxx")
    with pytest.raises(ValueError):
        vsm._set_modes("LLL.LLL")
    with pytest.raises(ValueError):
        vsm._set_modes("lnc.lnx")
    with pytest.raises(ValueError):
        vsm._set_modes(123)  # type: ignore


def test_all_modes() -> None:
    """Test all valid mode combinations."""
    vsm = Vespa(["sample document or something"])

    a = ["n", "l", "a", "b", "L", "d"]
    b = ["n", "f", "t", "p"]
    c = ["n", "c", "u", "b"]

    perms = list(itertools.product(a, b, c))
    single_mode = ["".join(x) for x in perms]

    second_perms = list(itertools.product(single_mode, single_mode))
    modes = [".".join(x) for x in second_perms]

    for mode in modes:
        vsm._set_modes(mode)
        vsm.k_score(query="sample query", k=1)
        assert mode == vsm.modes


def test_mode() -> None:
    vsm = Vespa(["sample"])

    vsm._set_modes("lnc.ltc")

    assert vsm._q_mode == "ltc"
    assert vsm._d_mode == "lnc"

    assert vsm.d_tf == "l"
    assert vsm.d_df == "n"
    assert vsm.d_no == "c"

    assert vsm.q_tf == "l"
    assert vsm.q_df == "t"
    assert vsm.q_no == "c"


def test_wrong_init() -> None:
    with pytest.raises(ValueError):
        Vespa(corpus="not a list")  # type: ignore


def test_wrong_k() -> None:
    with pytest.raises(ValueError):
        vsm = Vespa(corpus=["sample"])
        vsm.k_score("query", k=0)


def test_set_mode() -> None:
    vsm = Vespa(corpus=["sample document"])
    vsm.k_score("query", k=1, modes="lnc.ltc")
    vsm.score(query="sample query", modes="Ltu.apb")
    vsm.score(query="sample query", modes="dnb.Lpu")

    vsm = Vespa(corpus=["sample document or something with many words"])
    vsm.k_score(query="sample query", k=1, modes="dnb.Lpu")


def test_wrong_document() -> None:
    vsm = Vespa(corpus=["sample"])
    with pytest.raises(ValueError):
        vsm._score(query="query", document="wrong document")
    with pytest.raises(ValueError):
        vsm.score(query="query", document="wrong document")


def test_add_documents() -> None:
    documents = ["sample"]
    vsm = Vespa(corpus=documents)

    assert vsm.documents == documents
    documents = documents.copy()

    another_document = "another document"
    vsm.add(another_document)
    documents.append(another_document)

    assert vsm.documents == documents

    another_documents = ["even more", "documents"]
    vsm.add(another_documents)
    documents += another_documents

    assert vsm.documents == documents

    arg0 = ["foo", "bar"]
    arg1 = "foo"
    vsm.add(arg0, arg1)
    documents += arg0
    documents.append(arg1)

    assert vsm.documents == documents

    new = ["new", "and", "fresh"]
    vsm.corpus(new)

    assert vsm.documents == new


def test_add_wrong() -> None:
    documents = ["sample"]
    vsm = Vespa(corpus=documents)

    with pytest.raises(ValueError):
        vsm.add(123)  # type: ignore
    with pytest.raises(ValueError):
        vsm.add(["asd", 123])  # type: ignore
