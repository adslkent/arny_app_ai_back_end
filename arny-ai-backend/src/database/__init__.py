"""
Database module for Arny AI

This module provides database models and operations for the Arny AI travel assistant.
It uses Supabase (PostgreSQL) as the backend database with Row Level Security (RLS).

Main Components:
- Models: Pydantic models for data validation and serialization
- Operations: Database operations class for CRUD operations
- Enums: Database-related enumerations

Database Tables:
- user_profiles: User profile information
- onboarding_progress: User onboarding state tracking
- group_members: Group/family travel coordination
- chat_messages: Conversation history
- flight_searches: Flight search results and history
- hotel_searches: Hotel search results and history
- user_preferences: User travel preferences

Usage:
    from src.database import DatabaseOperations, UserProfile, OnboardingStep
    
    db = DatabaseOperations()
    profile = UserProfile(user_id="123", email="user@example.com")
    await db.create_user_profile(profile)
"""

# Import models
from .models import (
    OnboardingStep,
    UserProfile,
    OnboardingProgress,
    GroupMember,
    ChatMessage,
    FlightSearch,
    HotelSearch,
    UserPreferences
)

# Import operations
from .operations import DatabaseOperations

# Export main classes and functions
__all__ = [
    # Enums
    'OnboardingStep',
    
    # Models
    'UserProfile',
    'OnboardingProgress', 
    'GroupMember',
    'ChatMessage',
    'FlightSearch',
    'HotelSearch',
    'UserPreferences',
    
    # Operations
    'DatabaseOperations',
    
    # Exceptions
    'DatabaseError',
    'UserNotFoundError',
    'GroupNotFoundError',
    'DuplicateUserError',
    'InvalidDataError',
    'DatabaseConnectionError',
    'OnboardingError',
    'SearchError',
    
    # Utility functions
    'create_database_connection',
    'validate_user_id',
    'validate_email',
    'validate_group_code'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Arny AI Team'
__description__ = 'Database models and operations for Arny AI travel assistant'

# Database-specific exception classes
class DatabaseError(Exception):
    """Base exception for database-related errors"""
    pass

class UserNotFoundError(DatabaseError):
    """Raised when a user is not found in the database"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User not found: {user_id}")

class GroupNotFoundError(DatabaseError):
    """Raised when a group is not found in the database"""
    def __init__(self, group_code: str):
        self.group_code = group_code
        super().__init__(f"Group not found: {group_code}")

class DuplicateUserError(DatabaseError):
    """Raised when attempting to create a user that already exists"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User already exists: {user_id}")

class InvalidDataError(DatabaseError):
    """Raised when data validation fails"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass

class OnboardingError(DatabaseError):
    """Raised when onboarding-related operations fail"""
    pass

class SearchError(DatabaseError):
    """Raised when search operations fail"""
    pass

# Module-level utility functions
def create_database_connection() -> DatabaseOperations:
    """
    Factory function to create a new DatabaseOperations instance
    
    Returns:
        DatabaseOperations: Configured database operations instance
        
    Raises:
        DatabaseConnectionError: If database connection fails
    """
    try:
        return DatabaseOperations()
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to create database connection: {str(e)}")

def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format (UUID)
    
    Args:
        user_id: User ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str):
        return False
    
    # Check if it's a valid UUID format
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, user_id, re.IGNORECASE))

def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email validation
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def validate_group_code(group_code: str) -> bool:
    """
    Validate group code format
    
    Args:
        group_code: Group code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from ..utils.group_codes import GroupCodeGenerator
        return GroupCodeGenerator.validate_group_code(group_code)
    except ImportError:
        # Fallback validation if import fails
        if not group_code or not isinstance(group_code, str):
            return False
        # Basic validation: 4-10 alphanumeric characters, uppercase
        import re
        return bool(re.match(r'^[A-Z0-9]{4,10}$', group_code))

# Database schema information for reference
DATABASE_TABLES = {
    'user_profiles': {
        'description': 'User profile information including personal details',
        'primary_key': 'user_id',
        'indexes': ['email', 'group_code']
    },
    'onboarding_progress': {
        'description': 'User onboarding progress tracking',
        'primary_key': 'user_id',
        'indexes': ['current_step']
    },
    'group_members': {
        'description': 'Group membership for family/group travel',
        'primary_key': 'id',
        'indexes': ['group_code', 'user_id']
    },
    'chat_messages': {
        'description': 'Chat conversation history',
        'primary_key': 'message_id',
        'indexes': ['user_id', 'session_id', 'created_at']
    },
    'flight_searches': {
        'description': 'Flight search results and history',
        'primary_key': 'search_id',
        'indexes': ['user_id', 'created_at']
    },
    'hotel_searches': {
        'description': 'Hotel search results and history',
        'primary_key': 'search_id',
        'indexes': ['user_id', 'created_at']
    },
    'user_preferences': {
        'description': 'User travel preferences and settings',
        'primary_key': 'user_id',
        'indexes': []
    }
}

# Model validation helpers
def validate_user_profile_data(data: dict) -> bool:
    """
    Validate user profile data before creating UserProfile instance
    
    Args:
        data: Dictionary of user profile data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    required_fields = ['user_id', 'email']
    
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    # Validate user_id format
    if not validate_user_id(data['user_id']):
        return False
    
    # Validate email format
    if not validate_email(data['email']):
        return False
    
    return True

def validate_chat_message_data(data: dict) -> bool:
    """
    Validate chat message data before creating ChatMessage instance
    
    Args:
        data: Dictionary of chat message data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    required_fields = ['user_id', 'session_id', 'message_type', 'content']
    
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    # Validate message_type
    valid_types = ['user', 'assistant']
    if data['message_type'] not in valid_types:
        return False
    
    # Validate user_id format
    if not validate_user_id(data['user_id']):
        return False
    
    return True

# Convenience functions for common operations
async def get_user_profile_safe(user_id: str) -> UserProfile | None:
    """
    Safely get user profile without raising exceptions
    
    Args:
        user_id: User ID to look up
        
    Returns:
        UserProfile or None if not found
    """
    try:
        if not validate_user_id(user_id):
            return None
        
        db = create_database_connection()
        return await db.get_user_profile(user_id)
    except Exception:
        return None

async def check_user_exists(user_id: str) -> bool:
    """
    Check if user exists in database
    
    Args:
        user_id: User ID to check
        
    Returns:
        bool: True if user exists, False otherwise
    """
    profile = await get_user_profile_safe(user_id)
    return profile is not None

async def check_group_exists(group_code: str) -> bool:
    """
    Check if group exists in database
    
    Args:
        group_code: Group code to check
        
    Returns:
        bool: True if group exists, False otherwise
    """
    try:
        if not validate_group_code(group_code):
            return False
        
        db = create_database_connection()
        return await db.check_group_exists(group_code)
    except Exception:
        return False

# Health check function
async def test_database_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        db = create_database_connection()
        # Test with a simple operation
        # Just creating the connection and checking if client exists
        return hasattr(db, 'client') and db.client is not None
    except Exception:
        return False

# Add convenience functions to exports
__all__.extend([
    'get_user_profile_safe',
    'check_user_exists',
    'check_group_exists',
    'test_database_connection',
    'validate_user_profile_data',
    'validate_chat_message_data',
    'DATABASE_TABLES'
])

# Module initialization with logging
def _initialize_module():
    """Initialize the database module"""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Database module initialized successfully")
    except Exception:
        # Fail silently if logging setup fails
        pass

# Call initialization
_initialize_module()
