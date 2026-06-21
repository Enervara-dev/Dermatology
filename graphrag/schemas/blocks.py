from __future__ import annotations

from typing import List, Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field

class SummaryData(BaseModel):
    model_config = {"extra": "forbid"}
    text: str = Field(..., min_length=1)

class KeyPointsData(BaseModel):
    model_config = {"extra": "forbid"}
    points: List[str] = Field(..., min_length=1)

class BulletListData(BaseModel):
    model_config = {"extra": "forbid"}
    title: Optional[str] = None
    items: List[str] = Field(..., min_length=1)

class FollowUpQuestionsData(BaseModel):
    model_config = {"extra": "forbid"}
    questions: List[str] = Field(..., min_length=1)

class WarningData(BaseModel):
    model_config = {"extra": "forbid"}
    text: str = Field(..., min_length=1)
    severity: Literal["info", "caution", "critical"]

class NextStepsData(BaseModel):
    model_config = {"extra": "forbid"}
    steps: List[str] = Field(..., min_length=1)

class Condition(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(..., min_length=1)
    likelihood: Optional[str] = None
    description: Optional[str] = None

class ConditionListData(BaseModel):
    model_config = {"extra": "forbid"}
    conditions: List[Condition] = Field(..., min_length=1)

class SummaryBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["summary"]
    data: SummaryData

class KeyPointsBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["key_points"]
    data: KeyPointsData

class BulletListBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["bullet_list"]
    data: BulletListData

class FollowUpQuestionsBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["follow_up_questions"]
    data: FollowUpQuestionsData

class WarningBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["warning"]
    data: WarningData

class NextStepsBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["next_steps"]
    data: NextStepsData

class ConditionListBlock(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["condition_list"]
    data: ConditionListData

Block = Annotated[
    Union[
        SummaryBlock,
        KeyPointsBlock,
        BulletListBlock,
        FollowUpQuestionsBlock,
        WarningBlock,
        NextStepsBlock,
        ConditionListBlock,
    ],
    Field(discriminator="type")
]

class AnswerResponse(BaseModel):
    model_config = {"extra": "forbid"}
    blocks: List[Block]

BLOCK_TYPES = (
    "summary",
    "key_points",
    "bullet_list",
    "follow_up_questions",
    "warning",
    "next_steps",
    "condition_list",
)
