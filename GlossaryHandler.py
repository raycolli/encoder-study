import pandas as pd
from typing import List, Dict, Optional

class GlossaryHandler:
    def __init__(self):
        self.glossaries: Dict[str, Dict[str, List[str]]] = {
            'category': self._load_glossary('glossarys\Category Glossary.csv'),
            'grade': self._load_glossary('glossarys\Grade Glossary.csv'),
            'material': self._load_glossary('glossarys\Material_Glossary_Dec31.csv'),
            'schedule': self._load_glossary('glossarys\Schedule_Glossary_Dec31.csv'),
            'type': self._load_glossary('glossarys\Type_Glossary_Dec31.csv'),
            'size': self._load_glossary('glossarys\Size Glossary_Dec22.csv')
        }

    def _load_glossary(self, filename: str) -> Dict[str, List[str]]:
        df = pd.read_csv(filename)
        glossary = {}
        
        # Get the first column name (should be 'Primary Term' or similar)
        first_col = df.columns[0]
        
        for _, row in df.iterrows():
            # Filter out empty/NaN values
            variants = [str(val).strip() for val in row if pd.notna(val) and str(val).strip()]
            if variants:
                primary_term = str(row[first_col]).strip()
                glossary[primary_term] = variants
                
                # Add each variant as a key pointing to the same list
                for variant in variants[1:]:
                    variant = str(variant).strip()
                    if variant:  # Only add non-empty variants
                        glossary[variant] = variants
                    
        return glossary

    def get_permutations(self, term: str, glossary_type: str) -> Optional[List[str]]:
        """
        Get all permutations for a given term from specified glossary type.
        
        Args:
            term: The term to look up
            glossary_type: One of 'category', 'grade', 'material', 'schedule', 'type', 'size'
            
        Returns:
            List of permutations if found, None if not found
        """
        if glossary_type not in self.glossaries:
            raise ValueError(f"Invalid glossary type. Must be one of: {list(self.glossaries.keys())}")
            
        # Try to find the term in the specified glossary
        glossary = self.glossaries[glossary_type]
        term = str(term).strip().upper()
        
        # Special handling for size glossary
        if glossary_type == 'size':
            # Try to find an exact match first
            if term in glossary:
                return glossary[term]
            
            # If no exact match, try to match numeric values
            try:
                # Convert term to float for numeric comparison
                numeric_term = str(float(term))
                # Look for matching numeric values in the glossary
                for key, values in glossary.items():
                    try:
                        if float(key) == float(numeric_term):
                            return values
                    except ValueError:
                        continue
            except ValueError:
                pass
                
        return glossary.get(term)

    def get_primary_term(self, term: str, glossary_type: str) -> Optional[str]:
        """
        Get the primary (standardized) term for a given variant.
        
        Args:
            term: The variant term to look up
            glossary_type: One of 'category', 'grade', 'material', 'schedule', 'type', 'size'
            
        Returns:
            Primary term if found, None if not found
        """
        permutations = self.get_permutations(term, glossary_type)
        if permutations:
            return permutations[0]
        return None
    
def main():
    handler = GlossaryHandler()
    
    # Test size permutations
    size_tests = ["1", "1.0", "1.0IN", "1 IN"]
    for test in size_tests:
        perms = handler.get_permutations(test, "size")
        print(f"\nPermutations for '{test}': {perms}")
        primary = handler.get_primary_term(test, "size")
        print(f"Primary term for '{test}': {primary}")

    # Test material permutations
    material_tests = ["Carbon Steel", "CS", "C-STL"]
    for test in material_tests:
        perms = handler.get_permutations(test, "material")
        print(f"\nPermutations for '{test}': {perms}")
        primary = handler.get_primary_term(test, "material")
        print(f"Primary term for '{test}': {primary}")

if __name__ == "__main__":
    main()