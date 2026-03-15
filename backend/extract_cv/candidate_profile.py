from typing import List
from pydantic import BaseModel, Field
from enum import Enum

class Education(BaseModel):
    degree: str
    major: str
    institution: str
    start_year: int | None
    end_year: int | None

class Experience(BaseModel):
    job_title: str
    company: str
    start_date: str
    end_date: str | None
    description: str

class Level(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

class Skill(BaseModel):
    name: str
    level: Level = Field(description="Beginner | Intermediate | Advanced | Expert")

class CandidateProfile(BaseModel):
    full_name: str
    email: str
    phone: str | None
    address: str | None

    summary: str

    skills: List[Skill]
    education: List[Education]
    experience: List[Experience]

    languages: List[str] | None


