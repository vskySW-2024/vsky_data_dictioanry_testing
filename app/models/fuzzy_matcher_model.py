from pydantic import BaseModel
from typing import List, Tuple, Any

class MatchRequest(BaseModel):
    source: List[Tuple[str, str]]
    target: List[Tuple[str, str]]

class MatchResponse(BaseModel):
    matches: List[Tuple[str, str, str, str, float]]

class MatchResponseForDescMatcher(BaseModel):
    matches: List[Tuple[str, str, str, str, float, str]]

#with Relation Name
class MatchRequestWithRelation(BaseModel):
    source: List[Tuple[str, str, str, str]]
    target: List[Tuple[str, str, str, str]]


