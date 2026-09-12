"""Fixtures dùng chung. Mini-dataset inline — KHÔNG đọc file thật, để vòng lặp
TDD chạy trong vài chục millisecond và test không phụ thuộc mạng."""
import pytest


@pytest.fixture
def mini_squad():
    """Dataset tối thiểu bao đủ các trường hợp biên:
    - câu answerable 1 đáp án
    - câu impossible (có plausible_answers — KHÔNG được dùng làm gold)
    - câu answerable nhiều đáp án
    - 2 article, mỗi article 1 context
    """
    return {
        "version": 2.0,
        "data": [
            {
                "title": "Hà Nội",
                "paragraphs": [
                    {
                        "context": "Hà Nội là thủ đô của Việt Nam. Dân số khoảng 8 triệu người.",
                        "qas": [
                            {
                                "id": "q1",
                                "question": "Thủ đô của Việt Nam là gì?",
                                "is_impossible": False,
                                "answers": {"text": ["Hà Nội"], "answer_start": [0]},
                            },
                            {
                                "id": "q2",
                                "question": "GDP của Hà Nội là bao nhiêu?",
                                "is_impossible": True,
                                "answers": {"text": [], "answer_start": []},
                                "plausible_answers": {
                                    "text": ["8 triệu"],
                                    "answer_start": [44],
                                },
                            },
                        ],
                    }
                ],
            },
            {
                "title": "Huế",
                "paragraphs": [
                    {
                        "context": "Huế là thành phố ở miền Trung Việt Nam.",
                        "qas": [
                            {
                                "id": "q3",
                                "question": "Huế ở đâu?",
                                "is_impossible": False,
                                "answers": {
                                    "text": ["miền Trung Việt Nam", "miền Trung"],
                                    "answer_start": [19, 19],
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def duplicated_context_squad():
    """Cùng một context xuất hiện ở HAI article khác nhau — trường hợp dedup phải bắt."""
    ctx = "Văn bản trùng lặp xuất hiện hai lần."
    return {
        "version": 2.0,
        "data": [
            {"title": "A", "paragraphs": [{"context": ctx, "qas": [
                {"id": "a1", "question": "Câu hỏi A?", "is_impossible": False,
                 "answers": {"text": ["trùng lặp"], "answer_start": [8]}}]}]},
            {"title": "B", "paragraphs": [{"context": ctx, "qas": [
                {"id": "b1", "question": "Câu hỏi B?", "is_impossible": False,
                 "answers": {"text": ["trùng lặp"], "answer_start": [8]}}]}]},
        ],
    }
