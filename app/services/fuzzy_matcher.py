from typing import List, Tuple, Any
from rapidfuzz import process, fuzz
import re

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import torch

def init_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer

def perform_fuzzy_match(source: List[Tuple[str, str]], target: List[Tuple[str, str]]) -> List[Tuple[str, str, str, str, float]]:
    target_dict = {name: id_ for id_, name in target}

    matches = []
    for src_id, src_name in source:
        best_match, score,_ = process.extractOne(
            src_name, 
            target_dict.keys()
        )
        if score >= 75:  # Set a threshold for the match percentage
            target_id = target_dict[best_match]
            matches.append((src_id, src_name, target_id, best_match, score))
        else:
            matches.append((src_id, src_name, "", "", 0.00))  # No match

    return matches

def perform_fuzzy_matching_for_relations_and_fields(source, target,threshold):
    target_dict = {name: id_ for id_,rel, name, *_ in target}

    matches = []
    for src_id,rel, src_name,*_ in source:
        best_match, score,_ = process.extractOne(
            src_name, 
            target_dict.keys()
        )
        if score >= threshold:  # Set a threshold for the match percentage
            target_id = target_dict[best_match]
            matches.append((src_id, src_name, target_id, best_match, score,"By Field Name"))
        else:
            matches.append((src_id, src_name, "", "", 0.00,""))

    return matches

def description_checker(source_desc, embeddings2,target,model, tokenizer):#-->targetId, TargetFieldname,score
    if source_desc:
        source = [html_cleaner(source_desc)]
        embeddings1 = encode_sentences(source,model, tokenizer)
        best_matcher_index = None
        score=0.00
        for index,(tid_,tRname_, tFname_,tDesc_) in enumerate(target):
            if tDesc_:
                sim_cos = cosine_similarity(embeddings1,[embeddings2[index]])*100
                if score < sim_cos[0]:
                    best_matcher_index = index
                    score = sim_cos[0][0]
                    
        target_id = target[best_matcher_index][0]
        target_field_name = target[best_matcher_index][2]
        score = score 
        return target_id,target_field_name,score
    return "","",0.00

def encode_sentences(sentences, model, tokenizer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

def html_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text().strip()
    cleaned_text = cleaned_text.replace('  ',' ').replace('\xa0','')
    return cleaned_text

def check_and_add_record(hash_set, record):
    if record in hash_set:
        return False
    else:
        hash_set.add(record)
        return True

def perform_fuzzy_match_with_relation_excluded_full_matches(source, target,threshold_for_fields = 75,higher_threshold_for_relation = 75,lower_threshold_for_relation = 40):
    response_matche = []
    
    #set source and target relations 
    source_relations = list(set([source[1] for source in source]))
    target_relations = list(set([target[1] for target in target]))
    
    #get relation matches
    relation_matches = get_relation_matches_full_matches(source_relations, target_relations,higher_threshold_for_relation,lower_threshold_for_relation)
    #set source and target for field matching
    for relation_match in relation_matches: 
        source_list = [source_fields for source_fields in source if source_fields[1] == relation_match["source_relation"]]        
        target_list = []
        for target_fields in relation_match["target_relations"]:
            target_list.extend([t for t in target if t[1]==target_fields["target_relation"]])
        matches_for_current_relation = perform_fuzzy_matching_for_relations_and_fields(source_list, target_list,threshold_for_fields)
        response_matche.extend(matches_for_current_relation)
    
    full_matches = set([match[3] for match in response_matche if match[4] == 100.00])
    excluded_matches = [match for match in response_matche if match[4] == 100.00]

    # Sort the non-100% matches by x[4] in descending order
    non_exact_matches = sorted(
        [x for x in response_matche if x[4] != 100.00], 
        key=lambda x: x[4], 
        reverse=True
    )

    # Iterate through the sorted non-exact matches
    model, tokenizer = init_model()
    embeddings_target = encode_sentences([t[3] for t in target], model= model, tokenizer=tokenizer)
    for i in non_exact_matches:
        check_exist = check_and_add_record(full_matches, i[3])        
        if check_exist and i[4] !=0.00:
            excluded_matches.append(i)
        else:
            source_desc =[j for j in source if j[0] == i[0]]
            if source_desc[0][3]:
                tid, tname, matching_score = description_checker(source_desc[0][3],embeddings_target,target,model=model, tokenizer=tokenizer)                
                excluded_matches.append([
                    i[0],i[1],tid, tname, matching_score,"By Description"
                ])

    return excluded_matches

def get_relation_matches_full_matches(source_relations, target_relations, higher_threshold=75, lower_threshold=40):
    return_matches = []
    for source_relation in source_relations:
        is_spec_category = False        
        higher_matches = []
        lower_matches = []        
        if re.match(r'^Spec.*\d$', source_relation):
            is_spec_category_exist = False  
            partial_spec_matches = []
            for target_relation in target_relations:            
                source_number = re.search(r'\d+$', source_relation).group()
                is_spec_category = True 
                if re.match(r'^Spec*.*{number}$'.format(number=source_number), target_relation): 
                    return_matches.append({
                        "source_relation":source_relation,
                        "target_relations": [{
                            "target_relation":target_relation,
                            "score":100.00
                        }]
                    }) 
                    is_spec_category_exist = True
                    break
                if fuzz.partial_token_set_ratio(source_relation,target_relation) == 100.00:
                    partial_spec_matches.append({
                        "target_relation":target_relation,
                        "score":100
                    }) 
            if partial_spec_matches and not is_spec_category_exist:
                return_matches.append({
                    "source_relation":source_relation,
                    "target_relations": partial_spec_matches
                })  
                    
        if not is_spec_category:
            for target_relation in target_relations:            
                # Calculate fuzzy ratio
                fuzzy_ratio = fuzz.ratio(source_relation, target_relation)
                if fuzzy_ratio >= higher_threshold:
                    higher_matches.append({
                        "target_relation":target_relation,
                        "score": fuzzy_ratio
                    })
                elif lower_threshold < fuzzy_ratio < higher_threshold:
                    lower_matches.append({
                        "target_relation":target_relation,
                        "score": fuzzy_ratio
                })        
        # Handle higher matches
        if len(higher_matches)!=0:
            if len(higher_matches) == 1:  # Only one higher match
                return_matches.append({
                    "source_relation":source_relation,
                    "target_relations": higher_matches
                })
            else:
                top_higher_match = max(
                    return_matches, 
                    key=lambda x: x["score"]
                )
                if any(keyword in top_higher_match["target_relation"].lower() for keyword in ['header', 'line', 'lines']):
                    ordered_higher_matches = sorted(
                        [
                            x for x in lower_matches                            
                            if x["target_relation"].lower().replace('header', 'line') == top_higher_match["target_relation"].lower()
                            or x["target_relation"].lower().replace('lines', 'line') == top_higher_match["target_relation"].lower()
                        ],
                        key=lambda x: x["score"],
                        reverse=True
                    )
                    if len(ordered_higher_matches) != 0:                    
                        ordered_higher_matches = ordered_higher_matches[:min(len(ordered_higher_matches),2)]
                    return_matches.append({
                        "source_relation":source_relation,
                        "target_relations": ordered_higher_matches
                    })                    
                else:                    
                    return_matches.append({
                        "source_relation":source_relation,
                        "target_relations": [top_higher_match]
                    }) 
        # Handle lower matches if no higher matches
        elif len(lower_matches) !=0:            
            #adding subset matching
            matches_based_on_subset_condition = []
            for lower_match in lower_matches:
                subset_ratio = fuzz.token_set_ratio(source_relation,lower_match["target_relation"])
                if subset_ratio == 100.00:
                    matches_based_on_subset_condition.append({
                        "target_relation":lower_match["target_relation"],
                        "score": lower_match["score"]
                    })
            if matches_based_on_subset_condition:
                return_matches.append({
                    "source_relation":source_relation,
                    "target_relations": [max(matches_based_on_subset_condition, key= lambda x:x["score"])]
                })
            else: #not any subset conditon satsfied then
                if len(lower_matches) == 1:  # Only one lower match
                    return_matches.append({
                        "source_relation":source_relation,
                        "target_relations": lower_matches
                    })
                else:
                    top_lower_match = max(
                        lower_matches, 
                        key=lambda x: x["score"]
                    )
                    if any(keyword in top_lower_match["target_relation"].lower() for keyword in ['header', 'line', 'lines']):
                        ordered_lower_matches = sorted(
                            [
                                x for x in lower_matches                            
                                if x["target_relation"].lower().replace('header', 'line') == top_lower_match["target_relation"].lower()
                                or x["target_relation"].lower().replace('lines', 'line') == top_lower_match["target_relation"].lower()
                            ],
                            key=lambda x: x["score"],
                            reverse=True
                        )
                        if len(ordered_lower_matches) != 0:                    
                            ordered_lower_matches = ordered_lower_matches[:min(len(ordered_lower_matches),2)]
                        return_matches.append({
                            "source_relation":source_relation,
                            "target_relations": ordered_lower_matches
                        })                    
                    else:                    
                        return_matches.append({
                            "source_relation":source_relation,
                            "target_relations": [top_lower_match]
                        }) 

    return return_matches
#--------------------------------------------------------------------------------------------
# def perform_fuzzy_match_with_relation(source: List[Tuple[str, str]], target: List[Tuple[str, str]],threshold_for_fields = 75,higher_threshold_for_relation = 75,lower_threshold_for_relation = 40) -> List[Tuple[str, str, str, str, float]]:
#     response_matche = []
    
#     #set source and target relations 
#     source_relations = list(set([source[1] for source in source]))
#     target_relations = list(set([target[1] for target in target]))
#     #get relation matches
#     relation_matches = get_relation_matches(source_relations, target_relations,higher_threshold_for_relation,lower_threshold_for_relation)
#     #set source and target for field matching
#     for relation_match in relation_matches: 
#         source_list = [source_fields for source_fields in source if source_fields[1] == relation_match["source_relation"]]        
#         target_list = []
#         for target_fields in relation_match["target_relations"]:
#             target_list.extend([t for t in target if t[1]==target_fields["target_relation"]])
#         matches_for_current_relation = perform_fuzzy_matching_for_relations_and_fields(source_list, target_list,threshold_for_fields)
#         response_matche.extend(matches_for_current_relation)
    
#     full_matches = set([match[3] for match in response_matche if match[4] == 100.00])
#     excluded_matches = [match for match in response_matche if match[4] == 100.00]

#     # Sort the non-100% matches by x[4] in descending order
#     non_exact_matches = sorted(
#         [x for x in response_matche if x[4] != 100.00 and x[4] != 0.00], 
#         key=lambda x: x[4], 
#         reverse=True
#     )

#     # Iterate through the sorted non-exact matches
#     for i in non_exact_matches:
#         check_exist = check_and_add_record(full_matches, i[3])
#         if check_exist:
#             excluded_matches.append(i)

#     return excluded_matches
#     #return response_matche

# def get_relation_matches(source_relations, target_relations, higher_threshold=75, lower_threshold=40):
#     return_matches = []
    
#     for source_relation in source_relations:
#         higher_matches = []
#         lower_matches = []
        
#         for target_relation in target_relations:
#             # Calculate fuzzy ratio
#             fuzzy_ratio = fuzz.ratio(source_relation, target_relation)
#             if fuzzy_ratio >= higher_threshold:
#                 higher_matches.append({
#                     "target_relation":target_relation,
#                     "score": fuzzy_ratio
#                 })
#             elif lower_threshold < fuzzy_ratio < higher_threshold:
#                 lower_matches.append({
#                     "target_relation":target_relation,
#                     "score": fuzzy_ratio
#                 })
        
#         # Handle higher matches
#         if higher_matches:
#             if len(higher_matches) == 1:  # Only one higher match
#                 return_matches.append({
#                     "source_relation":source_relation,
#                     "target_relations": higher_matches
#                 })
#             else:
#                 top_higher_match = max(
#                     higher_matches, 
#                     key=lambda x: x["score"]
#                 )
#                 if any(keyword in top_higher_match["target_relation"].lower() for keyword in ['header', 'line', 'lines']):
#                     ordered_higher_matches = sorted(
#                         [
#                             x for x in lower_matches                            
#                             if x["target_relation"].lower().replace('header', 'line') == top_higher_match["target_relation"].lower()
#                             or x["target_relation"].lower().replace('lines', 'line') == top_higher_match["target_relation"].lower()
#                         ],
#                         key=lambda x: x["score"],
#                         reverse=True
#                     )
#                     if len(ordered_higher_matches) != 0:                    
#                         ordered_higher_matches = ordered_higher_matches[:min(len(ordered_higher_matches),2)]
#                     return_matches.append({
#                         "source_relation":source_relation,
#                         "target_relations": ordered_higher_matches
#                     })                    
#                 else:                    
#                     return_matches.append({
#                         "source_relation":source_relation,
#                         "target_relations": [top_higher_match]
#                     }) 
                
                
        
#         # Handle lower matches if no higher matches
#         elif lower_matches:
#             if len(lower_matches) == 1:  # Only one lower match
#                 return_matches.append({
#                     "source_relation":source_relation,
#                     "target_relations": lower_matches
#                 })
#             else:
#                 top_lower_match = max(
#                     lower_matches, 
#                     key=lambda x: x["score"]
#                 )
#                 if any(keyword in top_lower_match["target_relation"].lower() for keyword in ['header', 'line', 'lines']):
#                     ordered_lower_matches = sorted(
#                         [
#                             x for x in lower_matches                            
#                             if x["target_relation"].lower().replace('header', 'line') == top_lower_match["target_relation"].lower()
#                             or x["target_relation"].lower().replace('lines', 'line') == top_lower_match["target_relation"].lower()
#                         ],
#                         key=lambda x: x["score"],
#                         reverse=True
#                     )
#                     if len(ordered_lower_matches) != 0:                    
#                         ordered_lower_matches = ordered_lower_matches[:min(len(ordered_lower_matches),2)]
#                     return_matches.append({
#                         "source_relation":source_relation,
#                         "target_relations": ordered_lower_matches
#                     })                    
#                 else:                    
#                     return_matches.append({
#                         "source_relation":source_relation,
#                         "target_relations": [top_lower_match]
#                     }) 
#     return return_matches
    
#--------------------------------------------------------------------------------------------
