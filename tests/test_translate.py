"""Which descriptions are taken for French (twelfth round)."""

from whenframe.translate import is_french


def test_french_descriptions_are_taken_for_french():
    for text in ("le chien saute dans la piscine", "une personne ouvre la porte", "feux d'artifice sur la ville",
                 "Ajouter le sel et le poivre", "bébé qui pleure", "l'enfant mange du gâteau",
                 "femme s'assoit", "homme en bleu", "deux poissons nagent", "plage", "anniversaire"):  # fmt: skip
        assert is_french(text), text


def test_english_descriptions_are_not():
    for text in ("a dog jumps into the pool", "a person opens the door", "fireworks over the city",
                 "add salt and pepper", "a baby is crying", "a man in a café", "dog", "the man's hat",
                 "it's raining", "don't stop"):  # fmt: skip
        assert not is_french(text), text


def test_a_tie_or_no_mark_is_english():
    assert not is_french("orchestre")  # no mark
    assert not is_french("la piscine at the hotel")  # two French marks, two English
