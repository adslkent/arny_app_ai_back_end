"""
Authentication module for Arny AI

This module provides authentication services using Supabase Auth.
It handles user registration, login, session management, and token verification.

Main Components:
- SupabaseAuth: Main authentication service class

Usage:
    from src.auth import SupabaseAuth
    
    auth = SupabaseAuth()
    result = await auth.sign_up("user@example.com", "password")
"""

from .supabase_auth import SupabaseAuth

# Export main classes and functions
__all__ = [
    'SupabaseAuth',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Arny AI Team'
__description__ = 'Authentication services for Arny AI travel assistant'

# Module-level convenience functions (optional)
async def create_auth_service() -> SupabaseAuth:
    """
    Factory function to create a new SupabaseAuth instance
    
    Returns:
        SupabaseAuth: Configured authentication service instance
    """
    return SupabaseAuth()

# Authentication error classes (if you want to define custom exceptions)
class AuthenticationError(Exception):
    """Base exception for authentication errors"""
    pass

class InvalidCredentialsError(AuthenticationError):
    """Raised when user provides invalid credentials"""
    pass

class TokenExpiredError(AuthenticationError):
    """Raised when an access token has expired"""
    pass

class UserNotFoundError(AuthenticationError):
    """Raised when a user is not found"""
    pass

# Export exceptions as well
__all__.extend([
    'create_auth_service',
    'AuthenticationError',
    'InvalidCredentialsError', 
    'TokenExpiredError',
    'UserNotFoundError'
])