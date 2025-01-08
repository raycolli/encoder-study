import pandas as pd
import numpy as np
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from typing import List, Optional, Dict, Any

class QdrantSearchHandler:
    def __init__(self, glossary_handler):
        self.glossary_handler = glossary_handler

    def get_field_permutations(self, value: Any, field_type: str) -> Optional[List[str]]:
        """
        Get permutations for a field value, handling NaN/None cases
        
        Args:
            value: The value to get permutations for
            field_type: The type of glossary to use ('size', 'type', 'material', 'schedule')
            
        Returns:
            List of permutations if found, None if value is invalid or no permutations exist
        """
        if pd.isna(value) or value is None:
            return None
            
        try:
            perms = self.glossary_handler.get_permutations(str(value), field_type)
            return perms if perms else None
        except (ValueError, TypeError):
            return None

    def create_field_conditions(self, field_name: str, values: List[str]) -> List[FieldCondition]:
        """
        Create Qdrant field conditions for a given field and its values
        
        Args:
            field_name: Name of the field to create conditions for
            values: List of possible values for the field
            
        Returns:
            List of FieldCondition objects
        """
        return [FieldCondition(key=field_name, match=MatchAny(any=values))]

    def create_search_filter(self, 
                           size: Any = None,
                           type_val: Any = None,
                           material: Any = None,
                           schedule: Any = None) -> Optional[Filter]:
        """
        Create a Qdrant search filter with permutations for all fields
        
        Args:
            size: Size value
            type_val: Type value
            material: Material value
            schedule: Schedule value
            
        Returns:
            Qdrant Filter object or None if no valid conditions exist
        """
        field_mappings = {
            'PRIMARY_SIZE': ('size', size),
            'TYPE': ('type', type_val),
            'MATERIAL_NAME': ('material', material),
            'PRIMARY_SCHEDULE': ('schedule', schedule)
        }

        all_conditions = []
        
        for field_name, (glossary_type, value) in field_mappings.items():
            if perms := self.get_field_permutations(value, glossary_type):
                field_conditions = self.create_field_conditions(field_name, perms)
                all_conditions.extend(field_conditions)
        
        if not all_conditions:
            return None
            
        return Filter(should=all_conditions)

    def search_qdrant(self, 
                     client,
                     collection_name: str,
                     query_vector: List[float],
                     search_filter: Optional[Filter],
                     limit: int = 10000) -> pd.DataFrame:
        """
        Perform Qdrant search and convert results to DataFrame
        
        Args:
            client: Qdrant client instance
            collection_name: Name of the collection to search
            query_vector: Vector to search with
            search_filter: Filter to apply to search
            limit: Maximum number of results to return
            
        Returns:
            DataFrame containing search results
        """
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )
        
        return self.convert_scored_points_to_df(results)
    
    def convert_scored_points_to_df(self, scored_points: List) -> pd.DataFrame:
        """
        Convert Qdrant scored points to pandas DataFrame
        
        Args:
            scored_points: List of scored points from Qdrant search
            
        Returns:
            DataFrame containing the search results
        """
        records = []
        for point in scored_points:
            record = point.payload.copy()
            record['id'] = point.id
            record['score'] = point.score
            records.append(record)
            
        return pd.DataFrame.from_records(records)

# Usage example
def main():
    # Initialize handlers
    glossary_handler = GlossaryHandler()
    search_handler = QdrantSearchHandler(glossary_handler)
    
    # Example search parameters
    test_params = {
        'size': '1.0',
        'type_val': 'ELBOW',
        'material': 'CS',
        'schedule': '40'
    }
    
    # Create search filter
    search_filter = search_handler.create_search_filter(**test_params)
    
    if search_filter:
        print("Search filter created successfully:")
        print(f"Number of conditions: {len(search_filter.should)}")
        for condition in search_filter.should:
            print(f"Field: {condition.key}")
            print(f"Values: {condition.match.any}")
            print("---")
    else:
        print("No valid search conditions could be created from the input parameters")

if __name__ == "__main__":
    main()