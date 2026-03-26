"""
Chứa business logic để gộp các schema/dữ liệu lại với nhau.
"""

from backend.data_ingestion.schemas import CandidateProfile

def merge_profiles(results: list[CandidateProfile]) -> CandidateProfile:
    """Gộp nhiều CandidateProfile (từ các chunk khác nhau) thành một profile hoàn chỉnh."""
    final = {}

    for res in results:
        if not res:
            continue

        for key, value in res.model_dump().items():
            if not value:
                continue

            # Nếu field là list -> nối (extend)
            if isinstance(value, list):
                if key not in final:
                    final[key] = []
                final[key].extend(value)
            # Nếu field là string/primitive -> giữ giá trị xuất hiện đầu tiên
            else:
                if key not in final:
                    final[key] = value

    return CandidateProfile(**final)
