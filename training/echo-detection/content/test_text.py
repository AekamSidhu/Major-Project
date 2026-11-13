from content.text import first_words, to_transcript


def test_hc3_tokenisation_and_links_disappear():
    raw = "If you 're hearing about it , it 's because [ buy a few books ] ( URL_0 ) . You do n't ."
    assert to_transcript(raw) == "if youre hearing about it its because buy a few books you dont"


def test_markdown_and_line_breaks_disappear():
    raw = "## Answer\n\n1. **First**, use a hash map.\n- Second: it's O(1).\n"
    assert to_transcript(raw) == "answer first use a hash map second its o 1"


def test_same_words_give_the_same_transcript():
    assert to_transcript("Don’t   PANIC!") == to_transcript("dont panic")


def test_first_words():
    assert first_words("a b c d", 2) == "a b"
