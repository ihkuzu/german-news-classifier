from newsclf.report import Result, format_classes, format_report, load_all, safe_name, save

GOLD = ["Sport", "Sport", "Web", "Web", "Kultur", "Kultur"]


def full(name, predictions, **kwargs):
    return Result(name=name, method=name, gold=GOLD, predictions=predictions, **kwargs)


def test_safe_name_removes_characters_windows_rejects():
    assert safe_name("zero-shot-llama3.2:3b") == "zero-shot-llama3.2-3b"
    assert safe_name("deepset/gbert-base") == "deepset-gbert-base"


def test_save_and_load_round_trip(tmp_path):
    result = full("a:b", GOLD, train_seconds=1.5, details={"epochs": 3})
    path = save(result, tmp_path / "results")
    assert path.name == "a-b.json"
    assert load_all(tmp_path / "results") == [result]


def test_report_without_results():
    assert format_report([]) == "no results yet"


def test_report_shows_accuracy_and_speed():
    result = full("base", ["Sport", "Sport", "Web", "Web", "Kultur", "Web"], predict_seconds=0.6)
    table = format_report([result])
    assert "Full test set" in table
    assert "| base | 6 | 0.833 |" in table
    assert "100.0 ms" in table


def test_report_scores_all_methods_on_the_llm_sample():
    base = full("base", ["Sport", "Web", "Web", "Web", "Kultur", "Kultur"])
    llm = Result(
        name="llm",
        method="llm",
        gold=["Sport", "Kultur"],
        predictions=["Sport", None],
        indices=[1, 5],
    )
    table = format_report([base, llm])
    section = table.split("Same 2 articles as llm")[1]
    # base is wrong on article 1 and right on article 5
    assert "| base | 2 | 0.500 |" in section
    assert "| llm | 2 | 0.500 |" in section
    assert "| llm |" not in table.split("Same 2 articles as llm")[0]


def test_partial_result_is_not_compared_on_articles_it_lacks():
    first = Result(name="a", method="a", gold=["Sport"], predictions=["Sport"], indices=[0])
    second = Result(name="b", method="b", gold=["Web"], predictions=["Web"], indices=[2])
    table = format_report([first, second])
    assert "| b |" not in table.split("Same 1 articles as b")[0]


def test_classes_table_lists_mistakes_and_missing_labels():
    result = full("x", ["Sport", "Web", "Web", "Web", None, "Kultur"])
    text = format_classes(result)
    assert "| Sport | 1.00 | 0.50 | 0.67 | 2 |" in text
    assert "Sport -> Web: 1" in text
    assert "no valid label returned: 1" in text
