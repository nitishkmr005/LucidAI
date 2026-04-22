from app.utils.emotion import strip_emotion_tags


def test_strips_laugh():
    assert strip_emotion_tags("[laugh] That is funny.") == "That is funny."


def test_strips_chuckle():
    assert strip_emotion_tags("Sure thing. [chuckle] Let me check that.") == "Sure thing. Let me check that."


def test_strips_sigh():
    assert strip_emotion_tags("[sigh] I understand your frustration.") == "I understand your frustration."


def test_strips_gasp():
    assert strip_emotion_tags("Oh! [gasp] I see the issue now.") == "Oh! I see the issue now."


def test_strips_clears_throat():
    assert strip_emotion_tags("[clears throat] Right, so the account shows...") == "Right, so the account shows..."


def test_strips_multiple_tags():
    assert strip_emotion_tags("[chuckle] Happy to help. [sigh] It can be tricky.") == "Happy to help. It can be tricky."


def test_no_tags_unchanged():
    assert strip_emotion_tags("Hello, how can I help you today?") == "Hello, how can I help you today?"


def test_empty_string():
    assert strip_emotion_tags("") == ""


def test_collapses_extra_spaces():
    assert strip_emotion_tags("Hello  [laugh]  world") == "Hello world"


def test_unknown_tag_stripped():
    assert strip_emotion_tags("[hesitates] Well, let me think.") == "Well, let me think."
