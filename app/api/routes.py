from fastapi import APIRouter, HTTPException
from app.models.fuzzy_matcher_model import MatchRequest, MatchResponse, MatchRequestWithRelation,MatchResponseForDescMatcher
from app.services.fuzzy_matcher import perform_fuzzy_match,  perform_fuzzy_match_with_relation_excluded_full_matches #perform_fuzzy_match_with_relation,

router = APIRouter()
#routes
@router.post("/vsky_fuzzy_match", response_model=MatchResponse)
def fuzzy_match(data: MatchRequest):
    if not data.source or not data.target:
        raise HTTPException(status_code=400, detail="Source and target lists cannot be empty.")
    
    matches = perform_fuzzy_match(data.source, data.target)
    return MatchResponse(matches=matches)

# #routes
# @router.post("/vsky_fuzzy_match_with_relation", response_model=MatchResponse)
# def fuzzy_match_with_relation(data: MatchRequestWithRelation):
#     if not data.source or not data.target:
#         raise HTTPException(status_code=400, detail="Source and target lists cannot be empty.")
    
#     matches = perform_fuzzy_match_with_relation(data.source, data.target)
#     return MatchResponse(matches=matches)

#routes
@router.post("/vsky_fuzzy_match_with_relation_unique_full_matches", response_model=MatchResponseForDescMatcher)
def fuzzy_match_with_relation_unique_full_matches(data: MatchRequestWithRelation):
    if not data.source or not data.target:
        raise HTTPException(status_code=400, detail="Source and target lists cannot be empty.")
    
    matches = perform_fuzzy_match_with_relation_excluded_full_matches(data.source, data.target)
    return MatchResponseForDescMatcher(matches=matches)

