from typing import List
from pydantic import BaseModel, Field

class CandidateProfile(BaseModel):
    full_name: str
    email: str
    phone: str | None = None

    summary: str | None = None

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

    cv_file_name: str = Field(description = "Tên file cv của ứng viên, chỉ cần trả về định dạng <tên file>.pdf, ví dụ 'abc.pdf' ") 