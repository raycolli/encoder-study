import re
import unicodedata
import pandas as pd
import json
import numpy as np
from typing import Union, Dict, List, Any, Optional
from pathlib import Path
from fractions import Fraction

class DataCleanser:
    """
    Enhanced class for cleaning and standardizing data from various input sources.
    Handles strings, DataFrames, Excel files, and JSON data with additional 
    normalization features.
    """
    
    def __init__(self, 
                 default_uppercase: bool = True,
                 measurement_standardization: bool = True,
                 fraction_conversion: bool = True):
        """Initialize DataCleanser with configurable settings."""
        self.default_uppercase = default_uppercase
        self.measurement_standardization = measurement_standardization
        self.fraction_conversion = fraction_conversion
        
        # Common measurement mappings
        self.measurement_map = {
            '"': ' INCH',
            '″': ' INCH',
            '"': ' INCH',
            'IN.': ' INCH',
            'IN ': ' INCH ',
            'FT.': ' FT',
            'FT ': ' FT ',
            'LB.': ' LB',
            'LB ': ' LB ',
            'OZ.': ' OZ',
            'OZ ': ' OZ ',
            'MM.': ' MM',
            'MM ': ' MM ',
            'CLASS': 'CL',
            'SDR': ' SDR ',
            'WOG': ' WOG ',
            'OD': ' OD ',
            'ID': ' ID ',
            'PSI': ' PSI ',
            'PSIG': ' PSI ',
            'BAR': ' BAR ',
            'KPA': ' KPA ',
            'MPA': ' MPA ',
            'NPS': ' NPS ',
            'DN': ' DN '
        }
        
        # Pressure conversion factors (to PSI)
        self.pressure_conversions = {
            'BAR': 14.5038,
            'KPA': 0.145038,
            'MPA': 145.038
        }
        
        # Common nominal pipe sizes with fractions
        self.nps_fractions = {
            '1/8': '0.125',
            '1/4': '0.25',
            '3/8': '0.375',
            '1/2': '0.5',
            '3/4': '0.75'
        }

    def _convert_fraction(self, fraction_str: str) -> str:
        """
        Convert fraction string to decimal string.
        Handles simple fractions like '1/2' but not mixed numbers.
        
        Args:
            fraction_str (str): String containing a fraction (e.g., '1/2')
            
        Returns:
            str: Decimal representation of the fraction
        """
        try:
            if '/' in fraction_str:
                num, denom = map(int, fraction_str.split('/'))
                if denom != 0:
                    return str(round(num / denom, 3))
            return fraction_str
        except (ValueError, ZeroDivisionError):
            return fraction_str

    def _convert_mixed_number(self, match: re.Match) -> str:
        """
        Convert mixed number string to decimal string.
        Handles mixed numbers like '1 1/2' or simple fractions.
        
        Args:
            match (re.Match): Regex match object containing the number
            
        Returns:
            str: Decimal representation of the number
        """
        try:
            text = match.group()
            parts = text.split()
            
            if len(parts) == 1 and '/' in parts[0]:
                return self._convert_fraction(parts[0])
            elif len(parts) == 2 and '/' in parts[1]:
                whole = int(parts[0])
                frac_value = float(self._convert_fraction(parts[1]))
                return str(round(whole + frac_value, 3))
            return text
        except (ValueError, ZeroDivisionError):
            return match.group()

    def normalize_unicode(self, text: str, uppercase: Optional[bool] = None) -> str:
        """Normalize unicode characters in text."""
        if uppercase is None:
            uppercase = self.default_uppercase
            
        text = str(text)
        text = unicodedata.normalize("NFKD", text).encode('ASCII', 'ignore').decode('utf-8')
        
        # Standard character replacements
        unicode_map = {
            "\u2019": "'",    # Right single quotation mark
            "\u201d": '"',    # Right double quotation mark
            "\u2044": "/",    # Fraction slash
            "\u2013": "-",    # En dash
            "\u2022": "-",    # Bullet
            "\u00ae": " ",    # Registered sign
        }
        
        for old, new in unicode_map.items():
            text = text.replace(old, new)

        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text.upper() if uppercase else text

    def clean_text(self, text: str,
                  alpha_num: bool = False,
                  impute_none: bool = False,
                  uppercase: Optional[bool] = None,
                  max_tokens: Optional[int] = None) -> str:
        """Clean and standardize text with enhanced handling."""
        if uppercase is None:
            uppercase = self.default_uppercase

        # Initial normalization
        text = self.normalize_unicode(text, uppercase=False)
        
        if impute_none and (text is None or text == ""):
            return "NONE" if uppercase else "None"

        # Convert fractions if enabled
        if self.fraction_conversion:
            # Handle mixed numbers first (e.g., "1 1/2")
            text = re.sub(r'\d+\s+\d+/\d+|\d+/\d+', self._convert_mixed_number, text)
        
        # Standardize measurements if enabled
        if self.measurement_standardization:
            for old, new in self.measurement_map.items():
                text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)
        
        # Handle parentheses spacing
        text = re.sub(r"\(\s", "(", text)
        text = re.sub(r"\s\)", ")", text)
        
        # Remove non-alphanumeric characters if requested
        if alpha_num:
            text = re.sub(r"\W+", " ", text)
        
        # Clean up special characters and whitespace
        text = re.sub(r"#+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        # Handle max tokens
        if max_tokens and len(text.split()) > max_tokens:
            text = " ".join(text.split()[:max_tokens])
        
        return text.upper() if uppercase else text

    def clean_size(self, text: str) -> str:
        """Clean and standardize size measurements with NPS support."""
        text = self.clean_text(text, uppercase=True)
        
        # Handle NPS notation
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:"|INCH|NPS)', r'\1 NPS', text)
        text = re.sub(r'(?i)\bDN\s*(\d+)\b', r'DN \1', text)
        
        # Convert standard inch notation if not already NPS
        if 'NPS' not in text:
            text = re.sub(r'(\d+(?:\.\d+)?)"', r"\1 INCH", text)
        
        text = re.sub(r"(\d+)'", r"\1 FT", text)
        
        # Handle common nominal pipe sizes with fractions
        for fraction, decimal in self.nps_fractions.items():
            text = re.sub(fr'\b{fraction}\s*(?:NPS|INCH)\b', f'{decimal} NPS', text)
        
        # Standardize size-related terms
        matches = ["X", "INCH", "FT", "OD", "ID", "MM", "NPS", "DN"]
        for mt in matches:
            text = re.sub(rf"\b{mt}\b", f" {mt} ", text)
            
        return re.sub(r"\s+", " ", text).strip()

    def clean_schedule(self, text: str) -> str:
        """Clean and standardize schedule information."""
        text = self.clean_text(text, uppercase=True)
        matches = ["-", "SDR"]
        for mt in matches:
            text = re.sub(rf"\b{mt}\b", f" {mt} ", text)
        return re.sub(r"\s+", " ", text).strip()

    def clean_pressure(self, text: str, convert_to_psi: bool = True) -> str:
        """Clean and standardize pressure specifications with PSI support."""
        text = self.clean_text(text, uppercase=True)
        
        # Standardize class notation
        text = re.sub(r"(\d+)\s*CLASS", r"CL \1", text)
        text = re.sub(r"(\d+)\s*LBS", r"\1 LB", text)
        text = re.sub(r"\bCLASS\b", "CL", text)
        
        # Handle PSI variations
        text = re.sub(r"(\d+)\s*PSIG?", r"\1 PSI", text)
        
        # Convert other pressure units to PSI if requested
        if convert_to_psi:
            for unit, factor in self.pressure_conversions.items():
                pattern = rf"(\d+(?:\.\d+)?)\s*{unit}"
                text = re.sub(pattern, lambda m: f"{float(m.group(1)) * factor:.1f} PSI", text)
        
        # Clean up common pressure ratings
        pressure_classes = {
            '150': '285 PSI',
            '300': '740 PSI',
            '600': '1480 PSI',
            '900': '2220 PSI',
            '1500': '3705 PSI',
            '2500': '6170 PSI'
        }
        
        for class_rating, psi in pressure_classes.items():
            text = re.sub(rf"\bCL\s*{class_rating}\b", f"CL {class_rating} ({psi})", text)
        
        # Standardize pressure-related terms
        matches = ["CL", "WOG", "PSI", "BAR", "KPA", "MPA", "-"]
        for mt in matches:
            text = re.sub(rf"\b{mt}\b", f" {mt} ", text)
            
        return re.sub(r"\s+", " ", text).strip()

    def clean_dataframe(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       drop_duplicates: bool = True,
                       **kwargs) -> pd.DataFrame:
        """Clean DataFrame with improved handling."""
        df = df.copy()
        
        # Clean column names
        df.columns = [self.clean_text(col, alpha_num=True, uppercase=True) 
                     for col in df.columns]
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        # Clean specified columns
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.clean_text(x, **kwargs))
        
        # Handle missing values and duplicates
        df = df.fillna("")

        # remove underscores 
        df = df.replace('_', ' ', regex=True)

        if drop_duplicates:
            original_len = len(df)
            df = df.drop_duplicates()
            if original_len != len(df):
                print(f'Removed {original_len - len(df)} duplicate rows')
        
        return df

    def clean_excel(self, file_path: Union[str, Path],
                   sheet_name: Union[str, int] = 0,
                   **kwargs) -> pd.DataFrame:
        """Read and clean Excel file data."""
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return self.clean_dataframe(df, **kwargs)

    def clean_json(self, data: Union[str, Dict, List],
                  text_keys: Optional[List[str]] = None,
                  **kwargs) -> Union[Dict, List]:
        """Clean text values in JSON data."""
        if isinstance(data, str):
            data = json.loads(data)
            
        def clean_value(value: Any) -> Any:
            if isinstance(value, str):
                if text_keys is None or any(key in str(value) for key in text_keys):
                    return self.clean_text(value, **kwargs)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            return value
            
        return clean_value(data)

    def clean_data(self, data: Any, **kwargs) -> Any:
        """Generic method to clean data based on its type."""
        if isinstance(data, pd.DataFrame):
            return self.clean_dataframe(data, **kwargs)
        elif isinstance(data, (dict, list)):
            return self.clean_json(data, **kwargs)
        elif isinstance(data, str):
            if data.endswith('.xlsx'):
                return self.clean_excel(data, **kwargs)
            elif data.endswith('.json'):
                with open(data, 'r') as f:
                    return self.clean_json(json.load(f), **kwargs)
            else:
                return self.clean_text(data, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


# Example usage
if __name__ == "__main__":
    # Initialize cleanser with custom settings
    cleanser = DataCleanser(
        default_uppercase=True,
        measurement_standardization=True,
        fraction_conversion=True
    )
    
    # Test string cleaning
    test_strings = [
        'STEEL_BOLT_1/2_INCH',
        'Copper Wire 1/4"',
        'aluminum PLATE 3/8 in.',
        'Bronze_rod 5/16″',
        '150 CLASS pressure valve (285 PSI)',
        '2" Schedule 40 pipe',
        '4" NPS pipe',
        'DN 100 pipe',
        '1500 PSI valve',
        '10 BAR pressure gauge',
        '1/2 NPS fitting',
        '1 1/2 inch pipe'  # Mixed number test
    ]
    
    print("\nTesting string cleaning:")
    for string in test_strings:
        print(f"\nOriginal: {string}")
        print(f"Cleaned:  {cleanser.clean_text(string)}")
        print(f"Size:     {cleanser.clean_size(string)}")
        print(f"Pressure: {cleanser.clean_pressure(string)}")