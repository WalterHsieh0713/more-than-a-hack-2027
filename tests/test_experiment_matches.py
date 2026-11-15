from src import experiment


def test_the_licensed_set_is_german_and_the_full_set_only_adds_metrica(monkeypatch):
    class Fake:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(experiment.idsse, "MATCHES", ("A", "B"))
    monkeypatch.setattr(experiment.idsse, "load", lambda i: Fake("IDSSE-" + i))
    monkeypatch.setattr(experiment.metrica, "load", lambda game: Fake("metrica"))
    monkeypatch.setattr(experiment.model, "fill_blanks", lambda m: m)
    assert sorted(experiment.german()) == ["A", "B"]
    assert sorted(experiment.matches()) == ["A", "B"] + list(experiment.METRICA)
