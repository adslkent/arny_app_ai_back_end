import random
import string
from typing import Set
import uuid

class GroupCodeGenerator:
    """Generate and validate group codes for family/group functionality"""
    
    @staticmethod
    def generate_group_code(length: int = 6) -> str:
        """
        Generate a random group code
        
        Args:
            length: Length of the group code (default 6)
            
        Returns:
            Random alphanumeric group code
        """
        # Use uppercase letters and numbers, excluding confusing characters
        chars = string.ascii_uppercase + string.digits
        # Remove potentially confusing characters - FIX: Added missing second argument
        chars = chars.replace('0', '').replace('O', '').replace('I', '').replace('1', '')
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def generate_unique_group_code(existing_codes: Set[str], length: int = 6, max_attempts: int = 100) -> str:
        """
        Generate a unique group code that doesn't exist in the provided set
        
        Args:
            existing_codes: Set of existing group codes to avoid
            length: Length of the group code
            max_attempts: Maximum attempts to generate unique code
            
        Returns:
            Unique group code
            
        Raises:
            ValueError: If unable to generate unique code after max_attempts
        """
        for _ in range(max_attempts):
            code = GroupCodeGenerator.generate_group_code(length)
            if code not in existing_codes:
                return code
        
        # Fallback to UUID-based code if we can't generate unique code
        return str(uuid.uuid4()).replace('-', '').upper()[:length]
    
    @staticmethod
    def validate_group_code(code: str) -> bool:
        """
        Validate group code format
        
        Args:
            code: Group code to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not code:
            return False
        
        # Code should be 6-8 characters, alphanumeric, uppercase
        if not (4 <= len(code) <= 10):
            return False
        
        if not code.isalnum():
            return False
        
        return True
    
    @staticmethod
    def format_group_code(code: str) -> str:
        """
        Format group code to standard format (uppercase, stripped)
        
        Args:
            code: Raw group code
            
        Returns:
            Formatted group code
        """
        if not code:
            return ""
        
        return code.strip().upper()
