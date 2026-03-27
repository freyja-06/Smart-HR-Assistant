from typing import List, Optional
from pydantic import BaseModel, Field

class CandidateProfile(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    summary: Optional[str] = None

    # đơn giản hóa skill
    skills: List[str] = Field(default_factory=list)

    # experience flatten
    experiences: List[str] = Field(
        default_factory=list,
        description="Each item: 'Job title at Company (time): description'"
    )

    # education đơn giản
    education: List[str] = Field(
        default_factory=list,
        description="Each item: 'Degree - Major - School'"
    )