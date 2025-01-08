"""
USE THIS TO COLLECT ALL TESTING ROUTINES
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter
from fractions import Fraction
from typing import List, Dict, Union
import ast
import json
from tqdm.autonotebook import tqdm, trange
from EnhancedDataCleanser import DataCleanser
from GlossaryHandler import GlossaryHandler
from QdrantSearchHandler import QdrantSearchHandler

from dotenv import load_dotenv
load_dotenv()

ENCODER = os.getenv("ENCODER")

# qdrant
import qdrant_client
from qdrant_client.http.models import Filter, FieldCondition, MatchValue,  MatchAny, PointStruct, VectorParams, Distance
from qdrant_client import QdrantClient, models

ENDLESSFORMS_QDRANT_URL = os.getenv("ENDLESSFORMS_QDRANT_URL")
ENDLESSFORMS_TEST_CLUSTER_KEY = os.getenv("ENDLESSFORMS_TEST_CLUSTER_KEY")


def activate_qdrant_client(qdrant_url,api_key):
    """use this to access vector database"""
    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key,
        timeout=3000
    )
    return client


#openai
from openai import OpenAI
OPENAIKEY = os.getenv("OPEN_AI_KEY")
openaiclient = OpenAI(api_key=OPENAIKEY)

import voyageai
VOYAGE_AI = os.getenv("VOYAGE_AI")
vo = voyageai.Client(api_key=VOYAGE_AI)


pd.set_option('display.max_colwidth', None)

#encoder stuff

def voyage_vectorizer(voyageclient,encoder,text):
    """use voyageai client to encode text"""
    result = voyageclient.embed(text, model=encoder, input_type='query', output_dimension=2048)
    return result.embeddings[0]

def get_encoded_embedding(text, encoder):
    if text is not None:
        if encoder == "text-embedding-3-large":
            return open_ai_vectorizer(openaiclient, encoder, text)
        elif encoder == "text-embedding-3-small":
            return open_ai_vectorizer(openaiclient, encoder, text)
        elif encoder == "voyage-3-large":
            return voyage_vectorizer(vo,encoder,text)
        else:
            return encoder(text).astype(float)

def open_ai_vectorizer(openaiclient, model, text):
    """input text to vectorize, output vector"""
    response = openaiclient.embeddings.create(
    input=text,
    model=model)

    return response.data[0].embedding

def activate_multilingual_e5_encoder():
    """activate current encoder"""
    """
    https://huggingface.co/intfloat/multilingual-e5-base
    https://huggingface.co/spaces/mteb/leaderboard
    https://pytorch.org/get-started/locally/
    """
    with open(ENCODER, "rb") as f:
        encoder = joblib.load(f)
    return encoder

#helper funcitons to parse results into dataframes
def convert_scored_points_to_df(data: Union[str, List]) -> pd.DataFrame:
    """
    Converts ScoredPoint data into a pandas DataFrame.
    
    Parameters:
    data (Union[str, List]): Either a string containing ScoredPoint data or a list of ScoredPoint data
    
    Returns:
    pandas.DataFrame: DataFrame containing the parsed data
    """
    try:
        # Initialize list to store the records
        records = []
        
        # If the input is a string, evaluate it to get the list
        if isinstance(data, str):
            try:
                data = ast.literal_eval(data)
            except:
                raise ValueError("Could not parse input string into list")
        
        # Process each entry in the list
        for entry in data:
            try:
                # Extract payload and score
                payload = entry['payload'] if isinstance(entry, dict) else entry.payload
                score = entry['score'] if isinstance(entry, dict) else entry.score
                
                # Create a copy of the payload
                record = payload.copy()
                
                # Add score to the record
                record['score'] = score
                
                records.append(record)
            except (AttributeError, KeyError, TypeError) as e:
                print(f"Skipping malformed entry: {e}")
                continue
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        
        # Convert size to float if it exists
        if 'PRIMARY SIZE' in df.columns:
            df['PRIMARY SIZE'] = df['PRIMARY SIZE'].astype(float)
        
        # Sort columns alphabetically for consistency
        df = df.reindex(sorted(df.columns), axis=1)
            
        return df
    
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# Helper function to parse the string format if needed
def parse_string_to_list(text_data: str) -> List:
    """
    Parses a string containing ScoredPoint data into a list.
    Only used if the main function fails with string input.
    
    Parameters:
    text_data (str): String containing ScoredPoint data
    
    Returns:
    List: List of dictionaries containing the parsed data
    """
    try:
        # Extract the list content between square brackets
        start_idx = text_data.find('[')
        end_idx = text_data.rfind(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not find list boundaries in the input text")
            
        list_content = text_data[start_idx:end_idx + 1]
        
        # Parse the string into a Python object
        data = ast.literal_eval(list_content)
        return data
        
    except Exception as e:
        print(f"Error parsing string: {str(e)}")
        return []

def convert_to_mixed_number(decimal):
    whole_number = int(decimal)
    fractional_part = Fraction(decimal - whole_number).limit_denominator()

    if fractional_part == 0:
        return str(whole_number)
    else:
        return f"{whole_number} {fractional_part}"


## ---- STANDARDIZE THE DATASET HERE

def make_gt_data():
    """load GT data for study"""
    #df_gt_raw = pd.read_csv(r'data/ferg_GT_100_parsed_merged.csv')
    df_gt_raw = pd.read_csv(r'data\MATSCO_GT_100_PARSED_RFQS.CSV')

    #df_gt_raw['PRIMARY_SIZE'] = df_gt_raw['primary_size']

    print('size of df: ', len(df_gt_raw))

    c = DataCleanser(
        default_uppercase=True,
        measurement_standardization=True,
        fraction_conversion=True
    )

    df_gt = c.clean_data(df_gt_raw)

    df_gt['RANK'] = 0
    df_gt['QUALITY'] = 'EXCELLENT'

    df_gt['PRIMARY_SIZE'] = df_gt['PRIMARY_SIZE_RFQ']
    df_gt['MATERIAL_NAME'] = df_gt['MATERIAL_NAME_RFQ']
    df_gt['TYPE'] = df_gt['TYPE_RFQ']
    df_gt['PRIMARY_SCHEDULE'] = df_gt['PRIMARY_SCHEDULE_RFQ']

    print(df_gt.columns)

    return df_gt


## --------------TESTS GO HERE --------------


### TEST 4 -----------
def run_vector_pandas_filters(qdrantclient, df_gt, COLLECTION_NAME,encoder, output_csv_filename, log_filename,studyheader):
    """
    VECTOR + QDRANT SIZE. PANDAS TYPE, MATERIAL_NAME, SCHEDULE FILTERS
    """
    study_list = []
    query_limit = 10000

    # Create a function to write to both console and file
    def log_print(*args, **kwargs):
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert all arguments to strings and join them
        output = ' '.join(str(arg) for arg in args)
        
        # Add timestamp to the output
        timestamped_output = f"[{timestamp}] {output}"
        
        # Print to console
        print(timestamped_output)
        
        # Write to file
        with open(log_filename, 'a') as f:
            f.write(timestamped_output + '\n')

    log_print(studyheader)
    log_print('unraveling dynamic filtering query process...')

    glossary_handler = GlossaryHandler()

    for i, row in df_gt.iterrows():
        start_time = perf_counter()

        gt_desc = row['GT_DESC']
        inc_rfq = row['INC_RFQ']
        prior_rank = row['RANK']
        quality = row['QUALITY']
        size_raw = row['PRIMARY_SIZE']

        log_print('-----------------------------------------GT_NUMBER---> i: ', i, 'previous rank: ', prior_rank)
        log_print('gt_desc: ', gt_desc)
        log_print('inc_rfq: ', inc_rfq)
        log_print('quality: ', quality)
        log_print('size raw: ', size_raw)
        log_print('--------------------------------------------------------------------------------------------------')

        ### ---- does gt object actually exist in database?
        gt_response = None

        gt_response = qdrantclient.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="DESCRIPTION",
                        match=models.MatchValue(value=gt_desc))
                        ]),
                limit=5,
            )
        
        log_print('did we find it gt_response? ', gt_response)

        bolGT = False
        if gt_response != None:
            bolGT = True

        v = get_encoded_embedding(inc_rfq, encoder=encoder)
        size_raw = str(df_gt['PRIMARY_SIZE'].iloc[i])
        material_raw = str(df_gt['MATERIAL_NAME'].iloc[i]).replace('_',' ').upper()
        type_raw = str(df_gt['TYPE'].iloc[i]).replace('_',' ').upper()
        schedule = str(df_gt['PRIMARY_SCHEDULE'].iloc[i]).replace('_',' ').upper()
        
        log_print('what is size_raw: ', size_raw)
        log_print('what is material: ', material_raw)
        log_print('what is type: ', type_raw)
        log_print('what is schedule: ', schedule)

        v = get_encoded_embedding(inc_rfq, encoder=encoder)

        size_perms = glossary_handler.get_permutations(size_raw, 'size')

        should_conditions = [
        models.FieldCondition(
            key="PRIMARY_SIZE",  # Replace with your actual field name
            match=models.MatchValue(value=string)
        ) for string in size_perms]

        r = qdrantclient.search(
            collection_name=COLLECTION_NAME,
                query_vector=v,
                query_filter=models.Filter(
            should=should_conditions),
            limit=query_limit)

        df_results = convert_scored_points_to_df(r)

        log_print('len of vect + qdrant size results:', len(df_results))

        # now do other perms
        
        type_perms = glossary_handler.get_permutations(type_raw, 'type')
        materials_perms = glossary_handler.get_permutations(material_raw, 'material')
        schedule_perms = glossary_handler.get_permutations(schedule, 'schedule')
        return_limit = 5

        log_print('**** TYPE *****')
        log_print("Incoming dataframe size: ", len(df_results))

        # TYPE
        if isinstance(type_raw, str):

            if type_raw != "NAN":

                log_print('what is type: ', type_raw)
                log_print('type perms: ', type_perms)
                df_type = df_results[df_results["TYPE"].isin(type_perms)]
                log_print('len of type filter results: ', len(df_type))

                if len(df_type) < return_limit:
                    #keep remainder
                    log_print('no filtering taking place for type...')
                    df_type = df_results

            else:
                log_print('type is NAN, no filtering taking place...')
                df_type = df_results

        else:
            log_print('no type filtering..., no perms for filtering')
            df_type = df_results

        log_print('**** MATERIAL *****')
        log_print("Incoming dataframe size: ", len(df_type))

        # MATERIAL
        if isinstance(material_raw, str):

            if material_raw != "NAN":

                log_print('what is material: ', material_raw)
                log_print('material perms: ', materials_perms)
                df_mat = df_type[df_type["MATERIAL_NAME"].isin(materials_perms)]
                log_print('len of material filter results: ', len(df_mat))

                if len(df_mat) < return_limit:
                    #keep remainder
                    df_mat = df_type

            else:
                log_print('material is NAN, no filtering taking place')
                df_mat = df_type

        else:
            log_print('no material filtering..., no perms for material')
            df_mat = df_type
        
        # SCHEDULE

        log_print('**** SCHEDULE *****')
        log_print("Incoming dataframe size: ", len(df_mat))

        if isinstance(schedule, str):

            if schedule != "NAN":

                log_print('what is sched: ', schedule)
                log_print('sched perms: ', schedule_perms)
                df_sched = df_mat[df_mat["PRIMARY_SCHEDULE"].isin(schedule_perms)]
                log_print('len of schedule filter results: ', len(df_sched))

                if len(df_sched) < return_limit:
                    #keep remainder
                    df_sched = df_mat

            else:
                log_print('schedule is NAN, no filtering...')
                df_sched = df_mat

        else:
            log_print('no schedule filtering...')
            df_sched = df_mat

        # back to results...
        df_r = df_sched

        gt_desc = gt_desc.upper()
        df_r = df_r.applymap(lambda x: x.upper() if isinstance(x, str) else x)

        df_top25 = df_r.head(25)

        log_print('---------------------------top 25 for item: ',i)
        log_print(str(df_top25['DESCRIPTION']))

        n_results = len(df_r)

        if n_results == 0:
            log_print('unable to find matches.. ')
            index_out = query_limit
        else:
            idx = df_r.index[df_r['DESCRIPTION'] == gt_desc]

            log_print('ranking is: ', idx)
            log_print('lens of idx: ', len(idx))

            if len(idx) == 0:
                index_out = query_limit
            elif len(idx) == 1:
                index_out = idx[0]
            elif len(idx) >1:
                index_out = np.nanmin(idx)
            else:
                index_out = idx    

        log_print('------------------------------------------------------------GT_NUMBER---> i: ', i, 'RANK: ', index_out)
        total_time = perf_counter() - start_time
        study_list.append([inc_rfq,gt_desc,bolGT,i,index_out,total_time,df_r.head(25), prior_rank, quality])

    study_cols = ['inc_rfq','gt_desc','GT_exist','i','rank','totaltime','top_25', 'prior_rank', 'quality']
    df_outputlist = pd.DataFrame(data=study_list,columns=study_cols)
    df_outputlist.to_csv(output_csv_filename)


### TEST 3 ------------

def run_vector_qdrant_four_filters_test(qdrantclient, df_gt, COLLECTION_NAME,encoder, output_csv_filename, log_filename,studyheader):
    """
    VECTOR + QDRANT SIZE, TYPE, MATERIAL_NAME, SCHEDULE FILTERS
    """
    study_list = []
    query_limit = 10000
    
    # Create a function to write to both console and file
    def log_print(*args, **kwargs):
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert all arguments to strings and join them
        output = ' '.join(str(arg) for arg in args)
        
        # Add timestamp to the output
        timestamped_output = f"[{timestamp}] {output}"
        
        # Print to console
        print(timestamped_output)
        
        # Write to file
        with open(log_filename, 'a') as f:
            f.write(timestamped_output + '\n')

    log_print(studyheader)

    # grab inc rfq and gt val
    for i, row in df_gt.iterrows():
        start_time = perf_counter()

        gt_desc = row['GT_DESC']
        inc_rfq = row['INC_RFQ']
        prior_rank = row['RANK']
        quality = row['QUALITY']
        size_raw = row['PRIMARY_SIZE']

        log_print('-----------------------------------------GT_NUMBER---> i: ', i, 'previous rank: ', prior_rank)
        log_print('gt_desc: ', gt_desc)
        log_print('inc_rfq: ', inc_rfq)
        log_print('quality: ', quality)
        log_print('size raw: ', size_raw)
        log_print('--------------------------------------------------------------------------------------------------')

        ### ---- does gt object actually exist in database?
        gt_response = None

        gt_response = qdrantclient.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="DESCRIPTION",
                        match=models.MatchValue(value=gt_desc))
                        ]),
                limit=5,
            )
        
        log_print('did we find it gt_response? ', gt_response)

        bolGT = False
        if gt_response != None:
            bolGT = True

        v = get_encoded_embedding(inc_rfq, encoder=encoder)

        # Initialize handlers
        glossary_handler = GlossaryHandler()
        search_handler = QdrantSearchHandler(glossary_handler)

        size_raw = str(df_gt['PRIMARY_SIZE'].iloc[i])
        material_raw = str(df_gt['MATERIAL_NAME'].iloc[i]).replace('_',' ').upper()
        type_raw = str(df_gt['TYPE'].iloc[i]).replace('_',' ').upper()
        schedule = str(df_gt['PRIMARY_SCHEDULE'].iloc[i]).replace('_',' ').upper()

        log_print('what is size_raw: ', size_raw)
        log_print('what is material: ', material_raw)
        log_print('what is type: ', type_raw)
        log_print('what is schedule: ', schedule)
        
        # Example search parameters
        test_params = {
            'size': size_raw,
            'type_val': type_raw,
            'material': material_raw,
            'schedule': schedule
        }
        
        # Create search filter
        search_filter = search_handler.create_search_filter(**test_params)
        
        if search_filter:
            log_print("Search filter created successfully:")
            log_print(f"Number of conditions: {len(search_filter.should)}")
            for condition in search_filter.should:
                log_print(f"Field: {condition.key}")
                log_print(f"Values: {condition.match.any}")
                log_print("---")

            df_r = search_handler.search_qdrant(
                qdrantclient,
                collection_name=COLLECTION_NAME,
                query_vector=v,
                search_filter=search_filter)

            gt_desc = gt_desc.upper()

            # store as index, rank, timing
            n_results = len(df_r)
            df_top25 = df_r.head(25)

            log_print('---------------------------top 25 for item: ',i)
            log_print(str(df_top25['DESCRIPTION']))

            if n_results == 0:
                log_print('unable to find matches.. ')
                index_out = query_limit

            log_print('--> are we in last else:')

            idx = df_r.index[df_r['DESCRIPTION'] == gt_desc]

            log_print('ranking is: ', idx)
            log_print('lens of idx: ', len(idx))

            if len(idx) == 0:
                log_print('nothing found:')
                index_out = query_limit

            elif len(idx) == 1:
                log_print('what is this1', idx)
                index_out = idx[0]
                log_print('idx out for 1: ', index_out)

            elif len(idx) > 1:
                log_print('you have multiple hits, get smallest val')
                index_out = np.nanmin(idx)
            else:
                index_out = idx    

            log_print('------------------------------------------------------------GT_NUMBER---> i: ', i, 'RANK: ', index_out)
            total_time = perf_counter() - start_time
            study_list.append([inc_rfq,gt_desc,bolGT,i,index_out,total_time,df_r.head(25), prior_rank, quality])

    study_cols = ['inc_rfq','gt_desc','GT_exist','i','rank','totaltime','top_25', 'prior_rank', 'quality']
    df_outputlist = pd.DataFrame(data=study_list,columns=study_cols)
    df_outputlist.to_csv(output_csv_filename)





### TEST 2 ------------

def run_vector_qdrant_size_test(qdrantclient, df_gt, COLLECTION_NAME,encoder, output_csv_filename, log_filename,studyheader):
    """
    VECTOR + QDRANT SIZE FILTER
    """

    study_list = []
    query_limit = 10000

    size_handler = GlossaryHandler()

    # Create a function to write to both console and file
    def log_print(*args, **kwargs):
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert all arguments to strings and join them
        output = ' '.join(str(arg) for arg in args)
        
        # Add timestamp to the output
        timestamped_output = f"[{timestamp}] {output}"
        
        # Print to console
        print(timestamped_output)
        
        # Write to file
        with open(log_filename, 'a') as f:
            f.write(timestamped_output + '\n')

    log_print(studyheader)
    log_print('size of df: ', len(df_gt))

    for i, row in df_gt.iterrows():
        start_time = perf_counter()

        gt_desc = row['GT_DESC']
        inc_rfq = row['INC_RFQ']
        prior_rank = row['RANK']
        quality = row['QUALITY']
        size_raw = row['PRIMARY_SIZE']

        log_print('-----------------------------------------GT_NUMBER---> i: ', i, 'previous rank: ', prior_rank)
        log_print('gt_desc: ', gt_desc)
        log_print('inc_rfq: ', inc_rfq)
        log_print('quality: ', quality)
        log_print('size raw: ', size_raw)
        log_print('--------------------------------------------------------------------------------------------------')

        size_perms = size_handler.get_permutations(size_raw, 'size')

        if size_perms != None:
            log_print(f"Permutations: {size_perms}")
            log_print(f"Primary term: {size_handler.get_primary_term(size_raw, 'size')}")

            ### ---- does gt object actually exist in database?
            gt_response = None

            gt_response = qdrantclient.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="DESCRIPTION",
                            match=models.MatchValue(value=gt_desc))
                            ]),
                    limit=5,
                )
            
            log_print('did we find gt in collection? ', gt_response)

            bolGT = False
            if gt_response != None:
                bolGT = True
            
            # next, we perform a vector search, gather results, and see if gt exists in the results
            v = get_encoded_embedding(inc_rfq, encoder=encoder)

            should_conditions = [
            models.FieldCondition(
                key="PRIMARY_SIZE",  # Replace with your actual field name
                match=models.MatchValue(value=string)
            ) for string in size_perms]

            r = qdrantclient.search(
                collection_name=COLLECTION_NAME,
                    query_vector=v,
                    query_filter=models.Filter(
                should=should_conditions),
                limit=query_limit)
                    
            # rank the results and convert to dataframe
            df_r = convert_scored_points_to_df(r)
            df_top25 = df_r.head(25)
            
            log_print('---------------------------top 25 for item: ',i)
            log_print(str(df_top25['DESCRIPTION']))

            n_results = len(df_r)
            log_print('num of results found: ', n_results)

            if len(df_r) == 0:
                log_print('unable to find matches.. ')
                index_out = query_limit
            else:
                idx = df_r.index[df_r['DESCRIPTION'] == gt_desc]
                log_print('did we find a match? ', idx)

                if len(idx) == 0:
                    index_out = query_limit
                elif len(idx) == 1:
                    index_out = idx[0]
                elif len(idx) >1:
                    index_out = np.nanmin(idx)
                else:
                    index_out = idx    

                log_print('------------------------------------------------------------GT_NUMBER---> i: ', i, 'RANK: ', index_out)

                total_time = perf_counter() - start_time

                study_list.append([inc_rfq,gt_desc,bolGT,i,index_out,total_time,df_r.head(25), prior_rank, quality])

    study_cols = ['inc_rfq','gt_desc','GT_exist','i','rank','totaltime','top_25', 'prior_rank', 'quality']
    df_out = pd.DataFrame(data=study_list,columns=study_cols)
    df_out.to_csv(output_csv_filename)
    log_print('study complete. Find data here: ', output_csv_filename)

### TEST 1 --------------------
def run_vector_only_test(qdrantclient, df_gt, COLLECTION_NAME,encoder, output_csv_filename, log_filename,studyheader):
    """
    VECTOR ONLY
    """
    query_limit = 10000
    study_list = []
    
    # Create a log file with timestamp

    def log_print(*args, **kwargs):
        # @TODO -- MAKE CLASS
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert all arguments to strings and join them
        output = ' '.join(str(arg) for arg in args)
        
        # Add timestamp to the output
        timestamped_output = f"[{timestamp}] {output}"
        
        # Print to console
        print(timestamped_output)
        
        # Write to file
        with open(log_filename, 'a') as f:
            f.write(timestamped_output + '\n')

    log_print(studyheader)

    # grab inc rfq and gt val

    for i, row in df_gt.iterrows():
        start_time = perf_counter()
        gt_desc = row['GT_DESC']
        inc_rfq = row['INC_RFQ']
        prior_rank = row['RANK']
        quality = row['QUALITY']
        
        log_print('-----------------------------------------GT_NUMBER---> i: ', i, 'previous rank: ', prior_rank)
        log_print('gt_desc: ', gt_desc)
        log_print('inc_rfq: ', inc_rfq)
        log_print('quality: ', quality)
        log_print('--------------------------------------------------------------------------------------------------')

        ### ---- does gt object actually exist in database?
        gt_response = None
        gt_response = qdrantclient.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="DESCRIPTION",
                        match=models.MatchValue(value=gt_desc))
                        ]),
                limit=5,
            )
        
        log_print('did we find gt in collection? ', gt_response)

        bolGT = False
        if gt_response != None:
            bolGT = True
        
        # next, we perform a vector search, gather results, and see if gt exists in the results
        v = get_encoded_embedding(inc_rfq, encoder=encoder)
        
        r = qdrantclient.search(
            collection_name=COLLECTION_NAME,
            query_vector=v,
            limit=query_limit)
        
        # rank the results and convert to dataframe
        df_r = convert_scored_points_to_df(r)

        df_top25 = df_r.head(25)
        log_print('---------------------------top 25 for item: ',i)
        log_print(str(df_top25['DESCRIPTION']))

        # store as index, rank, timing
        idx = df_r.index[df_r['DESCRIPTION'] == gt_desc]

        log_print('did we find a match? ', idx)

        if len(idx) == 0:
            index_out = query_limit
        elif len(idx) == 1:
            index_out = idx[0]
        elif len(idx) >1:
            index_out = np.nanmin(idx)
        else:
            index_out = idx

        log_print('------------------------------------------------------------GT_NUMBER---> i: ', i, 'RANK: ', index_out)

        total_time = perf_counter() - start_time

        study_list.append([inc_rfq,gt_desc,bolGT,i,index_out,total_time,df_r.head(25), prior_rank, quality])

    study_cols = ['inc_rfq','gt_desc','GT_exist','i','rank','totaltime','top_25', 'prior_rank', 'quality']
    df_outputlist = pd.DataFrame(data=study_list,columns=study_cols)
    df_outputlist.to_csv(output_csv_filename)
    log_print('study complete. Find data here: ', output_csv_filename)