from build_index import detect_failed_embedding_response


def test_detect_failed_embedding_response_detects_html():
    error = RuntimeError(
        "<!--# blocked by corporate proxy -->\n<!DOCTYPE html><html><body>Access denied</body></html>"
    )
    assert detect_failed_embedding_response(error) is True


def test_detect_failed_embedding_response_rejects_plain_text():
    error = RuntimeError("Error code 429: Rate limited by provider")
    assert detect_failed_embedding_response(error) is False
