from pydantic import BaseModel, Field, field_validator


class UserQuestionRequest(BaseModel):
    """Validated request body for the question endpoint."""

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="Natural-language question for the multi-agent system.",
        examples=["What is Bitcoin?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty or whitespace.")
        return value
