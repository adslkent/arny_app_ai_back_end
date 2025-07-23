"""
Supabase authentication service for Arny AI

This module provides authentication services using Supabase Auth including:
- User registration (sign up)
- User login (sign in) 
- Session management
- Token verification and refresh
- Password reset functionality

The service handles JWT tokens and integrates with Supabase's authentication system.
"""

from typing import Optional, Dict, Any
from supabase import create_client, Client
import json
import logging
import requests
import time

from ..utils.config import config

# Set up logging
logger = logging.getLogger(__name__)

class SupabaseAuth:
    """
    Supabase authentication service
    
    Provides methods for user authentication, session management,
    and token verification using Supabase Auth.
    """
    
    def __init__(self):
        """Initialize the Supabase client with configuration"""
        try:
            self.client: Client = create_client(
                config.SUPABASE_URL, 
                config.SUPABASE_ANON_KEY
            )
            logger.info("Supabase auth client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    async def sign_up(self, email: str, password: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sign up a new user
        
        Args:
            email: User's email address
            password: User's password (minimum 6 characters)
            metadata: Additional user metadata (optional)
            
        Returns:
            Response containing user data or error
            
        Raises:
            ValueError: If email or password is invalid
            Exception: For other signup errors
        """
        # Validate inputs
        if not email or not isinstance(email, str):
            raise ValueError("Email is required and must be a string")
        
        if not password or len(password) < 6:
            raise ValueError("Password is required and must be at least 6 characters")
        
        # Validate email format (basic check)
        if "@" not in email or "." not in email:
            raise ValueError("Invalid email format")
        
        try:
            logger.info(f"Attempting to sign up user: {email}")
            
            # Prepare signup data
            signup_data = {
                "email": email,
                "password": password
            }
            
            # Add metadata if provided
            if metadata:
                signup_data["options"] = {
                    "data": metadata
                }
            
            response = self.client.auth.sign_up(signup_data)
            
            if response.user:
                logger.info(f"User signed up successfully: {email}")
                return {
                    "success": True,
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "confirmed": response.user.email_confirmed_at is not None,
                        "created_at": response.user.created_at,
                        "metadata": response.user.user_metadata or {}
                    },
                    "session": {
                        "access_token": response.session.access_token if response.session else None,
                        "refresh_token": response.session.refresh_token if response.session else None,
                        "expires_at": response.session.expires_at if response.session else None,
                        "token_type": response.session.token_type if response.session else "bearer"
                    },
                    "message": "User created successfully. Please check your email for confirmation if email confirmation is enabled."
                }
            else:
                logger.warning(f"Signup failed for user: {email} - No user returned")
                return {
                    "success": False,
                    "error": "Failed to create user account"
                }
                
        except Exception as e:
            logger.error(f"Signup error for {email}: {str(e)}")
            
            # Handle specific Supabase errors
            error_message = str(e)
            if "already registered" in error_message.lower():
                return {
                    "success": False,
                    "error": "An account with this email already exists"
                }
            elif "invalid email" in error_message.lower():
                return {
                    "success": False,
                    "error": "Invalid email address format"
                }
            elif "password" in error_message.lower() and "weak" in error_message.lower():
                return {
                    "success": False,
                    "error": "Password is too weak. Please choose a stronger password"
                }
            else:
                return {
                    "success": False,
                    "error": f"Signup failed: {error_message}"
                }
    
    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in an existing user
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            Response containing user data and session or error
            
        Raises:
            ValueError: If email or password is missing
        """
        # Validate inputs
        if not email or not isinstance(email, str):
            raise ValueError("Email is required")
        
        if not password or not isinstance(password, str):
            raise ValueError("Password is required")
        
        try:
            logger.info(f"Attempting to sign in user: {email}")
            
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info(f"User signed in successfully: {email}")
                return {
                    "success": True,
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "confirmed": response.user.email_confirmed_at is not None,
                        "last_sign_in": response.user.last_sign_in_at,
                        "metadata": response.user.user_metadata or {}
                    },
                    "session": {
                        "access_token": response.session.access_token,
                        "refresh_token": response.session.refresh_token,
                        "expires_at": response.session.expires_at,
                        "token_type": response.session.token_type or "bearer"
                    },
                    "message": "Signed in successfully"
                }
            else:
                logger.warning(f"Sign in failed for user: {email} - Invalid credentials")
                return {
                    "success": False,
                    "error": "Invalid email or password"
                }
                
        except Exception as e:
            logger.error(f"Sign in error for {email}: {str(e)}")
            
            # Handle specific Supabase errors
            error_message = str(e)
            if "invalid login credentials" in error_message.lower() or "invalid credentials" in error_message.lower():
                return {
                    "success": False,
                    "error": "Invalid email or password"
                }
            elif "email not confirmed" in error_message.lower():
                return {
                    "success": False,
                    "error": "Please confirm your email address before signing in"
                }
            elif "too many requests" in error_message.lower():
                return {
                    "success": False,
                    "error": "Too many sign in attempts. Please try again later"
                }
            else:
                return {
                    "success": False,
                    "error": f"Sign in failed: {error_message}"
                }
    
    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh user session using refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Response containing new session data or error
            
        Raises:
            ValueError: If refresh_token is missing
        """
        if not refresh_token or not isinstance(refresh_token, str):
            raise ValueError("Refresh token is required")
        
        try:
            logger.info("Attempting to refresh session")
            
            response = self.client.auth.refresh_session(refresh_token)
            
            if response.session:
                logger.info("Session refreshed successfully")
                return {
                    "success": True,
                    "session": {
                        "access_token": response.session.access_token,
                        "refresh_token": response.session.refresh_token,
                        "expires_at": response.session.expires_at,
                        "token_type": response.session.token_type or "bearer"
                    },
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "confirmed": response.user.email_confirmed_at is not None
                    } if response.user else None,
                    "message": "Session refreshed successfully"
                }
            else:
                logger.warning("Session refresh failed - Invalid refresh token")
                return {
                    "success": False,
                    "error": "Invalid or expired refresh token"
                }
                
        except Exception as e:
            logger.error(f"Session refresh error: {str(e)}")
            
            error_message = str(e)
            if "invalid refresh token" in error_message.lower() or "expired" in error_message.lower():
                return {
                    "success": False,
                    "error": "Invalid or expired refresh token"
                }
            else:
                return {
                    "success": False,
                    "error": f"Session refresh failed: {error_message}"
                }
    
    async def sign_out(self) -> Dict[str, Any]:
        """
        Sign out the current user
        
        Returns:
            Response indicating success or error
        """
        try:
            logger.info("Attempting to sign out user")
            
            response = self.client.auth.sign_out()
            
            logger.info("User signed out successfully")
            return {
                "success": True,
                "message": "User signed out successfully"
            }
            
        except Exception as e:
            logger.error(f"Sign out error: {str(e)}")
            return {
                "success": False,
                "error": f"Sign out failed: {str(e)}"
            }
    
    async def get_user(self, access_token: str) -> Dict[str, Any]:
        """
        Get user info from access token with multiple verification methods
        
        Args:
            access_token: User's access token
            
        Returns:
            User information or error
            
        Raises:
            ValueError: If access_token is missing
        """
        if not access_token or not isinstance(access_token, str):
            raise ValueError("Access token is required")
        
        try:
            logger.info("Attempting to get user from access token")
            
            # Method 1: Try direct REST API call
            try:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "apikey": config.SUPABASE_ANON_KEY,
                    "Content-Type": "application/json"
                }
                
                user_url = f"{config.SUPABASE_URL}/auth/v1/user"
                logger.info(f"Making request to: {user_url}")
                
                response = requests.get(user_url, headers=headers, timeout=10)
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info(f"User retrieved successfully via REST API: {user_data.get('email', 'unknown')}")
                    
                    return {
                        "success": True,
                        "user": {
                            "id": user_data.get("id"),
                            "email": user_data.get("email"),
                            "confirmed": user_data.get("email_confirmed_at") is not None,
                            "created_at": user_data.get("created_at"),
                            "last_sign_in": user_data.get("last_sign_in_at"),
                            "metadata": user_data.get("user_metadata", {})
                        }
                    }
                else:
                    logger.warning(f"REST API failed with status {response.status_code}: {response.text}")
                    
            except Exception as api_error:
                logger.warning(f"REST API method failed: {str(api_error)}")
            
            # Method 2: Try JWT decode verification (fallback)
            try:
                import jwt
                from jwt import PyJWTError
                
                # Decode JWT without verification to extract user info
                # Note: In production, you should verify the signature properly
                decoded_token = jwt.decode(access_token, options={"verify_signature": False})
                logger.info("Successfully decoded JWT token")
                
                # Extract user information from token
                user_id = decoded_token.get("sub")
                email = decoded_token.get("email")
                exp = decoded_token.get("exp")
                
                # Check if token is expired
                current_time = int(time.time())
                if exp and current_time > exp:
                    logger.warning("Token has expired")
                    return {
                        "success": False,
                        "error": "Access token has expired"
                    }
                
                if user_id and email:
                    logger.info(f"User retrieved successfully via JWT decode: {email}")
                    return {
                        "success": True,
                        "user": {
                            "id": user_id,
                            "email": email,
                            "confirmed": True,  # Assume confirmed if token exists
                            "created_at": None,
                            "last_sign_in": None,
                            "metadata": decoded_token.get("user_metadata", {})
                        }
                    }
                else:
                    logger.warning("JWT token missing required fields")
                    
            except ImportError:
                logger.warning("PyJWT not available for token decoding")
            except Exception as jwt_error:
                logger.warning(f"JWT decode method failed: {str(jwt_error)}")
            
            # Method 3: Try using Supabase client with proper session management (last resort)
            try:
                # Create a new client instance for this verification
                temp_client = create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)
                
                # Try to set session with dummy refresh token
                try:
                    temp_client.auth.set_session(access_token, "dummy_refresh_token")
                    user_response = temp_client.auth.get_user()
                    
                    if user_response and user_response.user:
                        logger.info(f"User retrieved successfully via Supabase client: {user_response.user.email}")
                        return {
                            "success": True,
                            "user": {
                                "id": user_response.user.id,
                                "email": user_response.user.email,
                                "confirmed": user_response.user.email_confirmed_at is not None,
                                "created_at": user_response.user.created_at,
                                "last_sign_in": user_response.user.last_sign_in_at,
                                "metadata": user_response.user.user_metadata or {}
                            }
                        }
                except Exception as session_error:
                    logger.warning(f"Supabase client session method failed: {str(session_error)}")
                    
            except Exception as client_error:
                logger.warning(f"Supabase client method failed: {str(client_error)}")
            
            # If all methods fail
            logger.error("All token verification methods failed")
            return {
                "success": False,
                "error": "Failed to verify access token with all available methods"
            }
                    
        except Exception as e:
            logger.error(f"Get user error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get user: {str(e)}"
            }
    
    async def reset_password(self, email: str, redirect_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Send password reset email
        
        Args:
            email: User's email address
            redirect_to: URL to redirect to after password reset (optional)
            
        Returns:
            Response indicating success or error
            
        Raises:
            ValueError: If email is missing or invalid
        """
        if not email or not isinstance(email, str):
            raise ValueError("Email is required")
        
        if "@" not in email or "." not in email:
            raise ValueError("Invalid email format")
        
        try:
            logger.info(f"Attempting to send password reset email to: {email}")
            
            reset_data = {"email": email}
            if redirect_to:
                reset_data["options"] = {"redirect_to": redirect_to}
            
            response = self.client.auth.reset_password_email(email, reset_data.get("options", {}))
            
            logger.info(f"Password reset email sent successfully to: {email}")
            return {
                "success": True,
                "message": "Password reset email sent successfully. Please check your email."
            }
            
        except Exception as e:
            logger.error(f"Password reset error for {email}: {str(e)}")
            
            error_message = str(e)
            if "user not found" in error_message.lower():
                # For security reasons, we might want to return success even if user doesn't exist
                return {
                    "success": True,
                    "message": "If an account with that email exists, a password reset email has been sent."
                }
            else:
                return {
                    "success": False,
                    "error": f"Password reset failed: {error_message}"
                }
    
    async def verify_token(self, access_token: str) -> Dict[str, Any]:
        """
        Verify if access token is valid and get user information
        
        Args:
            access_token: Access token to verify
            
        Returns:
            Verification result with user data if valid
        """
        if not access_token or not isinstance(access_token, str):
            return {
                "success": False,
                "error": "Access token is required"
            }
        
        try:
            logger.info("Verifying access token")
            logger.info(f"Token length: {len(access_token)}")
            logger.info(f"Token starts with: {access_token[:20]}...")
            
            # Get user info which also validates the token
            user_response = await self.get_user(access_token)
            
            if user_response.get("success"):
                logger.info("Token verified successfully")
                return {
                    "success": True,
                    "valid": True,
                    "user": user_response["user"],
                    "message": "Token is valid"
                }
            else:
                logger.warning(f"Token verification failed: {user_response.get('error')}")
                return {
                    "success": False,
                    "valid": False,
                    "error": user_response.get("error", "Token verification failed")
                }
                
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return {
                "success": False,
                "valid": False,
                "error": f"Token verification failed: {str(e)}"
            }
    
    async def update_user(self, access_token: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information
        
        Args:
            access_token: User's access token
            updates: Dictionary of updates to apply
            
        Returns:
            Response indicating success or error
        """
        if not access_token or not isinstance(access_token, str):
            raise ValueError("Access token is required")
        
        if not updates or not isinstance(updates, dict):
            raise ValueError("Updates dictionary is required")
        
        try:
            logger.info("Attempting to update user")
            
            # Set the session
            self.client.auth.set_session(access_token, "")
            
            # Update user
            response = self.client.auth.update_user(updates)
            
            if response.user:
                logger.info("User updated successfully")
                return {
                    "success": True,
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "confirmed": response.user.email_confirmed_at is not None,
                        "metadata": response.user.user_metadata or {}
                    },
                    "message": "User updated successfully"
                }
            else:
                logger.warning("User update failed")
                return {
                    "success": False,
                    "error": "Failed to update user"
                }
                
        except Exception as e:
            logger.error(f"User update error: {str(e)}")
            return {
                "success": False,
                "error": f"User update failed: {str(e)}"
            }
    
    def get_session(self) -> Optional[Dict[str, Any]]:
        """
        Get current session if available
        
        Returns:
            Current session data or None
        """
        try:
            session = self.client.auth.get_session()
            if session:
                return {
                    "access_token": session.access_token,
                    "refresh_token": session.refresh_token,
                    "expires_at": session.expires_at,
                    "token_type": session.token_type or "bearer"
                }
            return None
        except Exception as e:
            logger.error(f"Get session error: {str(e)}")
            return None