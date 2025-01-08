class PromptManager:
    """
    A class to manage loading and handling text prompts from files.
    
    This class provides functionality to:
    - Load prompts from text files
    - Cache loaded prompts to avoid repeated file operations
    - Handle file-related errors gracefully
    """
    
    _prompt_cache = {}  # Class-level cache to store loaded prompts
    
    @classmethod
    def get_prompt(cls, file_path: str) -> str:
        """
        Load a prompt from a file path. Uses caching to avoid repeated file reads.
        
        Args:
            file_path (str): Path to the text file containing the prompt
            
        Returns:
            str: The contents of the prompt file
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            PermissionError: If the program lacks permission to read the file
            IOError: For other file-related errors
        """
        # Check if prompt is already cached
        if file_path in cls._prompt_cache:
            return cls._prompt_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt = file.read()
                cls._prompt_cache[file_path] = prompt  # Cache the prompt
                return prompt
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied when trying to read: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading prompt file: {file_path}. Error: {str(e)}")
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the prompt cache to free up memory."""
        cls._prompt_cache.clear()
    
    @classmethod
    def remove_from_cache(cls, file_path: str) -> None:
        """
        Remove a specific prompt from the cache.
        
        Args:
            file_path (str): Path to the prompt file to remove from cache
        """
        cls._prompt_cache.pop(file_path, None)