"""
Utilities module for Arny AI Backend

This module provides utility functions and configurations including:
- Configuration management for environment variables
- Group code generation and validation for family/group travel
- Common utility functions for data validation and formatting
- Helper functions for date/time operations and string processing

Usage:
    from utils import config, GroupCodeGenerator
    from utils import validate_email, format_date, generate_session_id
    
    # Configuration
    api_key = config.OPENAI_API_KEY
    
    # Group codes
    generator = GroupCodeGenerator()
    code = generator.generate_group_code()
    
    # Utilities
    is_valid = validate_email("user@example.com")
    session_id = generate_session_id()
"""

import re
import uuid
import hashlib
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union

# FIXED: Handle email_validator import with fallback
try:
    from email_validator import validate_email as email_validate, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    # Fallback if email_validator is not available
    EMAIL_VALIDATOR_AVAILABLE = False
    EmailNotValidError = ValueError

from .config import Config, config
from .group_codes import GroupCodeGenerator

# Export main classes and functions
__all__ = [
    # Configuration
    'Config',
    'config',
    'get_config',
    'validate_config',
    
    # Group codes
    'GroupCodeGenerator',
    'generate_group_code',
    'validate_group_code',
    'format_group_code',
    
    # Validation utilities
    'validate_email',
    'validate_phone',
    'validate_date',
    'validate_uuid',
    'validate_session_id',
    
    # Formatting utilities
    'format_date',
    'format_currency',
    'format_phone',
    'sanitize_string',
    'truncate_string',
    
    # Generation utilities
    'generate_session_id',
    'generate_unique_id',
    'generate_hash',
    'generate_timestamp',
    
    # Data utilities
    'safe_get',
    'merge_dicts',
    'flatten_dict',
    'clean_dict',
    'normalize_text',
    
    # Constants
    'DEFAULT_DATE_FORMAT',
    'DEFAULT_DATETIME_FORMAT',
    'SUPPORTED_CURRENCIES',
    'PHONE_REGEX_PATTERNS'
]

# Version information
__version__ = '1.0.0'

# ==================== CONSTANTS ====================

DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEZONE = "UTC"

SUPPORTED_CURRENCIES = [
    'USD', 'EUR', 'GBP', 'AUD', 'CAD', 'JPY', 'CNY', 'INR', 'SGD', 'HKD'
]

PHONE_REGEX_PATTERNS = {
    'international': r'^\+[1-9]\d{1,14}$',
    'us': r'^(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$',
    'au': r'^(\+61[-.\s]?)?(\(0\)|0)?[2-9]\d{8}$',
    'uk': r'^(\+44[-.\s]?)?(\(0\)|0)?[1-9]\d{8,9}$'
}

# ==================== CONFIGURATION UTILITIES ====================

def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Config: Global configuration object
    """
    return config

def validate_config() -> Dict[str, Any]:
    """
    Validate the current configuration
    
    Returns:
        dict: Configuration validation results
    """
    try:
        config.validate()
        return {
            'valid': True,
            'message': 'Configuration is valid',
            'missing_fields': []
        }
    except ValueError as e:
        missing_fields = str(e).replace('Required environment variable ', '').replace(' is not set', '').split(', ')
        return {
            'valid': False,
            'message': str(e),
            'missing_fields': missing_fields
        }

# ==================== GROUP CODE UTILITIES ====================

def generate_group_code(length: int = 6) -> str:
    """
    Generate a random group code
    
    Args:
        length: Length of the group code
        
    Returns:
        Random group code
    """
    return GroupCodeGenerator.generate_group_code(length)

def validate_group_code(code: str) -> bool:
    """
    Validate group code format
    
    Args:
        code: Group code to validate
        
    Returns:
        True if valid, False otherwise
    """
    return GroupCodeGenerator.validate_group_code(code)

def format_group_code(code: str) -> str:
    """
    Format group code to standard format
    
    Args:
        code: Raw group code
        
    Returns:
        Formatted group code
    """
    return GroupCodeGenerator.format_group_code(code)

# ==================== VALIDATION UTILITIES ====================

def validate_email(email: str) -> bool:
    """
    FIXED: Validate email address format with fallback
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # First try with email_validator if available
    if EMAIL_VALIDATOR_AVAILABLE:
        try:
            email_validate(email)
            return True
        except EmailNotValidError:
            return False
    
    # Fallback to regex validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email.strip()))

def validate_phone(phone: str, country_code: str = 'international') -> bool:
    """
    Validate phone number format
    
    Args:
        phone: Phone number to validate
        country_code: Country code pattern to use ('international', 'us', 'au', 'uk')
        
    Returns:
        True if valid, False otherwise
    """
    if not phone or not isinstance(phone, str):
        return False
        
    if country_code not in PHONE_REGEX_PATTERNS:
        country_code = 'international'
    
    pattern = PHONE_REGEX_PATTERNS[country_code]
    return bool(re.match(pattern, phone.strip()))

def validate_date(date_string: str, date_format: str = DEFAULT_DATE_FORMAT) -> bool:
    """
    Validate date string format
    
    Args:
        date_string: Date string to validate
        date_format: Expected date format
        
    Returns:
        True if valid, False otherwise
    """
    if not date_string or not isinstance(date_string, str):
        return False
        
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False

def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID string format
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not uuid_string or not isinstance(uuid_string, str):
        return False
        
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format (should be a UUID)
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return validate_uuid(session_id)

# ==================== FORMATTING UTILITIES ====================

def format_date(date_obj: Union[datetime, date, str], output_format: str = DEFAULT_DATE_FORMAT) -> str:
    """
    Format date object to string
    
    Args:
        date_obj: Date object, datetime object, or date string
        output_format: Output format string
        
    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        # Try to parse string date
        try:
            date_obj = datetime.strptime(date_obj, DEFAULT_DATE_FORMAT).date()
        except ValueError:
            try:
                date_obj = datetime.strptime(date_obj, DEFAULT_DATETIME_FORMAT)
            except ValueError:
                return date_obj  # Return as-is if can't parse
    
    if isinstance(date_obj, datetime):
        return date_obj.strftime(output_format)
    elif isinstance(date_obj, date):
        return date_obj.strftime(output_format)
    else:
        return str(date_obj)

def format_currency(amount: Union[int, float], currency: str = 'USD', include_symbol: bool = True) -> str:
    """
    Format currency amount
    
    Args:
        amount: Amount to format
        currency: Currency code
        include_symbol: Whether to include currency symbol
        
    Returns:
        Formatted currency string
    """
    if currency not in SUPPORTED_CURRENCIES:
        currency = 'USD'
    
    # Basic formatting
    formatted = f"{amount:,.2f}"
    
    if include_symbol:
        currency_symbols = {
            'USD': '$', 'EUR': '€', 'GBP': '£', 'AUD': 'A$',
            'CAD': 'C$', 'JPY': '¥', 'CNY': '¥', 'INR': '₹',
            'SGD': 'S$', 'HKD': 'HK$'
        }
        symbol = currency_symbols.get(currency, currency)
        formatted = f"{symbol}{formatted}"
    
    return formatted

def format_phone(phone: str, country_code: str = 'international') -> str:
    """
    Format phone number
    
    Args:
        phone: Phone number to format
        country_code: Country code for formatting
        
    Returns:
        Formatted phone number
    """
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    # Basic formatting based on country
    if country_code == 'us' and len(cleaned) == 10:
        return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
    elif country_code == 'us' and len(cleaned) == 11 and cleaned.startswith('1'):
        return f"+1 ({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
    else:
        return cleaned

def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input
    
    Args:
        text: Text to sanitize
        max_length: Maximum length (optional)
        
    Returns:
        Sanitized string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove potential dangerous characters
    text = re.sub(r'[<>"\';\\]', '', text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].strip()
    
    return text

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].strip() + suffix

# ==================== GENERATION UTILITIES ====================

def generate_session_id() -> str:
    """
    Generate a unique session ID
    
    Returns:
        UUID string for session
    """
    return str(uuid.uuid4())

def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id

def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """
    Generate hash of data
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of hash
    """
    if algorithm == 'md5':
        return hashlib.md5(data.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data.encode()).hexdigest()
    else:  # default sha256
        return hashlib.sha256(data.encode()).hexdigest()

def generate_timestamp(include_microseconds: bool = False) -> str:
    """
    Generate current timestamp string
    
    Args:
        include_microseconds: Whether to include microseconds
        
    Returns:
        Timestamp string
    """
    now = datetime.utcnow()
    if include_microseconds:
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return now.strftime(DEFAULT_DATETIME_FORMAT)

# ==================== DATA UTILITIES ====================

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary
    
    Args:
        data: Dictionary to get value from
        key: Key to look for
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    return data.get(key, default) if isinstance(data, dict) else default

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result

def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        data: Dictionary to flatten
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    def _flatten(obj, parent_key=''):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                items.extend(_flatten(v, new_key).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data)

def clean_dict(data: Dict[str, Any], remove_none: bool = True, remove_empty: bool = True) -> Dict[str, Any]:
    """
    Clean dictionary by removing None/empty values
    
    Args:
        data: Dictionary to clean
        remove_none: Whether to remove None values
        remove_empty: Whether to remove empty strings/lists/dicts
        
    Returns:
        Cleaned dictionary
    """
    if not isinstance(data, dict):
        return data
    
    cleaned = {}
    for key, value in data.items():
        # Skip None values if requested
        if remove_none and value is None:
            continue
        
        # Skip empty values if requested
        if remove_empty and value in ('', [], {}):
            continue
        
        # Recursively clean nested dictionaries
        if isinstance(value, dict):
            cleaned_value = clean_dict(value, remove_none, remove_empty)
            if cleaned_value or not remove_empty:  # Only add if not empty or if we're not removing empty
                cleaned[key] = cleaned_value
        else:
            cleaned[key] = value
    
    return cleaned

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# ==================== FACTORY FUNCTIONS ====================

def create_group_code_generator() -> GroupCodeGenerator:
    """
    Factory function to create a GroupCodeGenerator instance
    
    Returns:
        GroupCodeGenerator: Configured group code generator
    """
    return GroupCodeGenerator()

# ==================== HEALTH CHECK ====================

def health_check() -> Dict[str, Any]:
    """
    Perform health check on utilities module
    
    Returns:
        dict: Health status of utilities
    """
    status = {
        'module': 'utils',
        'version': __version__,
        'status': 'healthy',
        'email_validator_available': EMAIL_VALIDATOR_AVAILABLE,
        'components': {}
    }
    
    try:
        # Test configuration
        config_status = validate_config()
        status['components']['config'] = {
            'status': 'healthy' if config_status['valid'] else 'warning',
            'details': config_status
        }
        
        # Test group code generator
        generator = GroupCodeGenerator()
        test_code = generator.generate_group_code()
        is_valid = generator.validate_group_code(test_code)
        
        status['components']['group_codes'] = {
            'status': 'healthy' if is_valid else 'error',
            'test_code_generated': test_code,
            'test_validation_passed': is_valid
        }
        
        # Test utilities
        test_email = validate_email('test@example.com')
        test_uuid = validate_uuid(str(uuid.uuid4()))
        
        status['components']['validators'] = {
            'status': 'healthy' if test_email and test_uuid else 'error',
            'email_validator': test_email,
            'uuid_validator': test_uuid
        }
        
    except Exception as e:
        status['status'] = 'error'
        status['error'] = str(e)
    
    return status

# ==================== MODULE INITIALIZATION ====================

# Validate configuration on import (optional - can be disabled for testing)
try:
    config.validate()
except ValueError as e:
    # Don't fail on import, just log warning
    import warnings
    warnings.warn(f"Configuration validation failed: {e}", UserWarning)

# Export commonly used instances
group_code_generator = GroupCodeGenerator()