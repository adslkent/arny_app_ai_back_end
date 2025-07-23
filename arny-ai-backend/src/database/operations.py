"""
Database operations for Arny AI - ENHANCED VERSION WITH TENACITY RETRY STRATEGIES

Enhanced with comprehensive retry strategies for all Supabase database operations
using Tenacity library for robust error handling and automatic retries.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, date, timedelta
import json
import uuid
import logging
import asyncio
from supabase import create_client, Client
from postgrest.exceptions import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    retry_any,
    retry_if_result,
    before_sleep_log
)
from pydantic import BaseModel, ValidationError

from ..utils.config import config
from .models import (
    UserProfile, OnboardingProgress, GroupMember, ChatMessage,
    FlightSearch, HotelSearch, UserPreferences, BookingRequest,
    TravelItinerary, OnboardingStep, UserRole, MessageType,
    BookingStatus, PaginationInfo
)

# Set up logging
logger = logging.getLogger(__name__)

# Custom retry conditions for database operations
def retry_on_db_error(result):
    """Retry if database operation result contains errors or warnings"""
    if hasattr(result, 'data') and hasattr(result, 'error'):
        # Supabase response object
        return result.error is not None
    elif isinstance(result, dict):
        return (
            result.get("error") is not None or
            "warning" in result or
            result.get("success") is False
        )
    return False

def retry_on_empty_required_result(result):
    """Retry if result is unexpectedly empty for operations that should return data"""
    if hasattr(result, 'data'):
        return result.data is None or (isinstance(result.data, list) and len(result.data) == 0)
    return False

def retry_on_validation_error(result):
    """Retry if result fails Pydantic model validation"""
    if hasattr(result, 'data') and result.data:
        try:
            # This will be customized per method based on expected model
            return False
        except ValidationError:
            return True
    return False

# Combined retry strategy for database operations
database_retry = retry(
    retry=retry_any(
        retry_if_exception_type((APIError, ConnectionError, TimeoutError)),
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|502|503|504).*"),
        retry_if_result(retry_on_db_error)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Specific retry for operations that should return data
database_retry_with_data = retry(
    retry=retry_any(
        retry_if_exception_type((APIError, ConnectionError, TimeoutError)),
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|502|503|504).*"),
        retry_if_result(retry_on_db_error),
        retry_if_result(retry_on_empty_required_result)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class DatabaseOperations:
    """
    Database operations using Supabase with comprehensive retry strategies
    """
    
    def __init__(self):
        """Initialize the database connection"""
        try:
            self.client: Client = create_client(
                config.SUPABASE_URL, 
                config.SUPABASE_SERVICE_ROLE_KEY
            )
            logger.info("Database client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database client: {e}")
            raise ConnectionError(f"Database connection failed: {e}")
    
    def _validate_and_format_uuid(self, user_id: str, field_name: str = "UUID") -> tuple[bool, str]:
        """
        Enhanced UUID validation and formatting across all methods with better error messages
        """
        if not user_id:
            error_msg = f"{field_name} cannot be empty"
            logger.warning(error_msg)
            return False, error_msg
        
        try:
            # Convert to string and strip whitespace
            user_id_str = str(user_id).strip()
            
            # Remove any quotes that might be present
            user_id_str = user_id_str.strip('"\'')
            
            # Validate UUID format
            uuid_obj = uuid.UUID(user_id_str)
            
            # Return the string representation to ensure consistency
            validated_uuid = str(uuid_obj)
            logger.debug(f"✅ {field_name} validated: {validated_uuid}")
            return True, validated_uuid
            
        except ValueError as e:
            error_msg = f"Invalid {field_name} format: {e}"
            logger.warning(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected {field_name} validation error: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    # ==================== USER PROFILE OPERATIONS ====================
        
    @database_retry
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile by user_id with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_profile: {validated_user_id}")
                return None
            
            logger.info(f"Getting user profile for user_id: {validated_user_id}")
            
            response = self.client.table("user_profiles").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                profile_data = response.data[0]
                logger.info(f"User profile found for user_id: {validated_user_id}")
                
                # PARSE JSON FIELDS BEFORE VALIDATION
                # Parse holiday_preferences from JSON string to list
                if profile_data.get("holiday_preferences") and isinstance(profile_data["holiday_preferences"], str):
                    try:
                        profile_data["holiday_preferences"] = json.loads(profile_data["holiday_preferences"])
                    except json.JSONDecodeError as je:
                        logger.warning(f"Failed to parse holiday_preferences JSON for user {validated_user_id}: {je}")
                        profile_data["holiday_preferences"] = []
                
                # Validate with Pydantic model
                try:
                    return UserProfile(**profile_data)
                except ValidationError as ve:
                    logger.error(f"Profile validation error for user {validated_user_id}: {ve}")
                    return None
            
            logger.info(f"No user profile found for user_id: {validated_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            return None
    
    @database_retry
    async def create_user_profile(self, profile: UserProfile) -> bool:
        """
        Create a new user profile with retry strategies
        """
        try:
            # Validate the user_id in the profile
            is_valid, validated_user_id = self._validate_and_format_uuid(profile.user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id in profile: {validated_user_id}")
                return False
            
            logger.info(f"Creating user profile for user_id: {validated_user_id}")
            
            # Check if profile already exists before creating
            existing_profile = await self.get_user_profile(validated_user_id)
            if existing_profile:
                logger.info(f"User profile already exists for user_id: {validated_user_id}")
                return True  # Consider it successful if already exists
            
            # Convert profile to dict and handle datetime fields
            profile_dict = profile.dict()
            profile_dict["user_id"] = validated_user_id  # Use validated user_id
            profile_dict["created_at"] = datetime.utcnow().isoformat()
            profile_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to string if present
            if profile_dict.get("birthdate"):
                if isinstance(profile_dict["birthdate"], date):
                    profile_dict["birthdate"] = profile_dict["birthdate"].isoformat()
            
            # Convert list fields to JSON if present
            if "holiday_preferences" in profile_dict and profile_dict["holiday_preferences"]:
                if isinstance(profile_dict["holiday_preferences"], list):
                    profile_dict["holiday_preferences"] = json.dumps(profile_dict["holiday_preferences"])
            
            response = self.client.table("user_profiles").insert(profile_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User profile created successfully for user_id: {validated_user_id}")
            else:
                logger.warning(f"Failed to create user profile for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating user profile for {profile.user_id}: {e}")
            return False
    
    @database_retry
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update user profile with enhanced safety, error handling, and retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in update_user_profile: {validated_user_id}")
                return False
            
            logger.info(f"Updating user profile for user_id: {validated_user_id}")
            logger.info(f"Update data keys: {list(updates.keys())}")
            
            # Get original profile first and backup
            original_profile = await self.get_user_profile(validated_user_id)
            if not original_profile:
                logger.error(f"Cannot update: No user profile found for user_id: {validated_user_id}")
                return False
            
            # Enhanced data validation with better error handling
            cleaned_updates = {}
            validation_errors = []
            
            # Add updated_at timestamp
            cleaned_updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Handle each field with enhanced safety
            for key, value in updates.items():
                try:
                    if value is None or value == "":
                        continue  # Skip None/empty values
                    
                    if key == "birthdate":
                        cleaned_value = self._safely_parse_birthdate(value)
                        if cleaned_value:
                            cleaned_updates[key] = cleaned_value
                    
                    elif key == "holiday_preferences":
                        cleaned_value = self._safely_parse_holiday_preferences(value)
                        if cleaned_value:
                            cleaned_updates[key] = cleaned_value
                    
                    elif key in ["name", "email", "gender", "city", "employer", "working_schedule", 
                               "holiday_frequency", "annual_income", "monthly_spending", "travel_style", "group_code"]:
                        # Safe string fields
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_updates[key] = str(value).strip() if isinstance(value, str) else value
                        else:
                            validation_errors.append(f"Invalid type for {key}: {type(value)}")
                    
                    elif key == "onboarding_completed":
                        # Ensure boolean
                        cleaned_updates[key] = bool(value)
                    
                    elif key == "is_active":
                        # Ensure boolean
                        cleaned_updates[key] = bool(value)
                    
                    else:
                        # For other fields, be cautious
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_updates[key] = value
                        else:
                            validation_errors.append(f"Unsupported field or type for {key}: {type(value)}")
                            
                except Exception as field_error:
                    validation_errors.append(f"Error processing field {key}: {str(field_error)}")
                    logger.warning(f"Error processing field {key}: {field_error}")
            
            # Check for validation errors
            if validation_errors:
                logger.warning(f"Data validation errors for user {validated_user_id}: {validation_errors}")
                # Continue with valid fields only, don't fail the entire update
            
            if not cleaned_updates or len(cleaned_updates) <= 1:  # Only updated_at
                logger.warning(f"No valid updates after cleaning for user_id: {validated_user_id}")
                return True  # Consider it successful if no changes needed
            
            logger.info(f"Cleaned update data: {cleaned_updates}")
            
            # Perform update
            try:
                response = self.client.table("user_profiles").update(cleaned_updates).eq("user_id", validated_user_id).execute()
                
                success = len(response.data) > 0
                
                if success:
                    logger.info(f"User profile updated successfully for user_id: {validated_user_id}")
                    
                    # Verify the update worked
                    await asyncio.sleep(0.1)  # Small delay for consistency
                    verification_profile = await self.get_user_profile(validated_user_id)
                    
                    if verification_profile:
                        logger.info(f"✅ Update verification successful for user_id: {validated_user_id}")
                        return True
                    else:
                        logger.error(f"❌ Update verification failed - profile disappeared for user_id: {validated_user_id}")
                        return False
                else:
                    logger.warning(f"No rows updated for user_id: {validated_user_id}")
                    return False
                    
            except Exception as update_error:
                logger.error(f"Database update error for user_id {validated_user_id}: {update_error}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating user profile for {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _safely_parse_birthdate(self, value) -> Optional[str]:
        """Safely parse birthdate with multiple format support"""
        try:
            if isinstance(value, str) and value.strip():
                value = value.strip()
                # Try different date formats
                for date_fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]:
                    try:
                        parsed_date = datetime.strptime(value, date_fmt).date()
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
                
                # If no format worked, check if it's already in YYYY-MM-DD format
                if len(value) == 10 and value.count('-') == 2:
                    try:
                        # Validate it's a real date
                        datetime.strptime(value, "%Y-%m-%d")
                        return value
                    except ValueError:
                        pass
                        
            elif isinstance(value, date):
                return value.isoformat()
                
        except Exception as e:
            logger.warning(f"Could not parse birthdate '{value}': {e}")
        
        return None
    
    def _safely_parse_holiday_preferences(self, value) -> Optional[str]:
        """Safely parse holiday preferences"""
        try:
            if isinstance(value, list):
                return json.dumps(value)
            elif isinstance(value, str) and value.strip():
                value = value.strip()
                # Check if it's already JSON
                try:
                    json.loads(value)  # Test if it's valid JSON
                    return value
                except json.JSONDecodeError:
                    # If not JSON, treat as comma-separated or single preference
                    if "," in value:
                        preferences = [p.strip() for p in value.split(",") if p.strip()]
                    else:
                        preferences = [value]
                    return json.dumps(preferences)
        except Exception as e:
            logger.warning(f"Could not parse holiday_preferences '{value}': {e}")
        
        return None
    
    @database_retry
    async def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete user profile (soft delete by setting is_active to False) with retry strategies
        """
        try:
            logger.info(f"Soft deleting user profile for user_id: {user_id}")
            
            updates = {
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return await self.update_user_profile(user_id, updates)
            
        except Exception as e:
            logger.error(f"Error deleting user profile for {user_id}: {e}")
            return False
    
    @database_retry
    async def complete_onboarding(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """
        Mark onboarding as complete with enhanced filtering and retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in complete_onboarding: {validated_user_id}")
                return False
            
            logger.info(f"Completing onboarding for user_id: {validated_user_id}")
            print(f"🎉 Starting ENHANCED onboarding completion for user: {validated_user_id}")
            print(f"📋 Input profile data keys: {list(profile_data.keys())}")
            
            # STEP 1: CRITICAL - Ensure onboarding completion flag is set FIRST
            print(f"🔥 CRITICAL STEP: Setting onboarding_completed = True")
            completion_success = await self._force_onboarding_completion(validated_user_id)
            
            if not completion_success:
                logger.error(f"CRITICAL: Failed to set onboarding completion flag for user_id: {validated_user_id}")
                print(f"❌ CRITICAL FAILURE: Could not set onboarding completion flag")
                return False
            
            print(f"✅ CRITICAL SUCCESS: Onboarding completion flag set successfully")
            
            # STEP 2: OPTIONAL - Update profile data with enhanced filtering
            try:
                print(f"📝 OPTIONAL STEP: Updating profile data with enhanced filtering")
                
                # Enhanced filtering for profile data
                user_profile_fields = {
                    'name', 'gender', 'birthdate', 'city', 'employer',
                    'working_schedule', 'holiday_frequency', 'annual_income',
                    'monthly_spending', 'holiday_preferences', 'travel_style',
                    'group_code', 'email', 'group_skipped'
                }
                
                # Enhanced: Explicitly exclude problematic fields
                group_invite_exclude_fields = {
                    'group_invites_sent', 'group_invites_declined', 'group_code_shared',
                    'invited_emails', 'conversation_history', 'current_step',
                    'completion_timestamp', 'completed'
                }
                
                # Better filtering and data cleaning
                filtered_profile_data = {}
                for key, value in profile_data.items():
                    if (key in user_profile_fields and 
                        key not in group_invite_exclude_fields and 
                        value is not None and value != ""):
                        filtered_profile_data[key] = value
                    elif key in group_invite_exclude_fields:
                        print(f"🚫 Excluding problematic field: {key} = {value}")
                    elif key not in user_profile_fields:
                        print(f"⚠️ Excluding non-profile field: {key} = {value}")
                    else:
                        print(f"⚠️ Excluding empty/null field: {key} = {value}")

                print(f"📝 Filtered profile data: {list(filtered_profile_data.keys())}")
                print(f"📝 Profile values: {filtered_profile_data}")
                
                if filtered_profile_data:
                    # Update profile data (but don't fail onboarding if this fails)
                    profile_update_success = await self.update_user_profile(validated_user_id, filtered_profile_data)
                    print(f"📄 Profile data update success: {profile_update_success}")
                    
                    if not profile_update_success:
                        logger.warning(f"Profile data update failed, but onboarding completion is still successful")
                        print(f"⚠️ Profile data update failed, but onboarding is still complete")
                    else:
                        print(f"✅ Profile data updated successfully")
                else:
                    print(f"ℹ️ No additional profile data to update after filtering")
                    
            except Exception as profile_error:
                logger.warning(f"Profile data update failed: {profile_error}, but onboarding completion is still successful")
                print(f"⚠️ Profile data update error: {profile_error}, but onboarding is still complete")
            
            # STEP 3: Update onboarding progress to completed
            try:
                # Include completion status for both group invite and skip cases
                progress_completion_data = {
                    "completed": True, 
                    "completion_timestamp": datetime.utcnow().isoformat(),
                    "group_invites_handled": profile_data.get("group_invites_sent", False),
                    "group_setup_skipped": profile_data.get("group_skipped", False)
                }
                
                progress_success = await self.update_onboarding_progress(
                    validated_user_id, 
                    OnboardingStep.COMPLETED, 
                    progress_completion_data
                )
                print(f"📈 Progress update success: {progress_success}")
            except Exception as progress_error:
                logger.warning(f"Failed to update onboarding progress: {progress_error}")
                print(f"⚠️ Progress update failed: {progress_error}")
                # Don't fail the entire process if progress update fails
            
            # STEP 4: Enhanced final verification with multiple attempts
            print(f"🔍 ENHANCED FINAL VERIFICATION: Checking onboarding completion status")
            verification_attempts = 5
            
            for attempt in range(verification_attempts):
                try:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    verification_profile = await self.get_user_profile(validated_user_id)
                    
                    if verification_profile and verification_profile.onboarding_completed:
                        print(f"✅ ENHANCED VERIFICATION PASSED: Onboarding completion confirmed!")
                        logger.info(f"Onboarding completed successfully for user_id: {validated_user_id}")
                        
                        # Additional verification
                        if verification_profile.email and verification_profile.name:
                            print(f"✅ Profile data verification: email={verification_profile.email}, name={verification_profile.name}")
                            return True
                        else:
                            print(f"⚠️ Profile missing critical data: email={verification_profile.email}, name={verification_profile.name}")
                    else:
                        print(f"⚠️ Verification attempt {attempt + 1}: onboarding_completed = {verification_profile.onboarding_completed if verification_profile else 'No profile'}")
                        
                        if attempt == verification_attempts - 1:
                            # Last attempt - try force completion one more time
                            print(f"🔧 Last attempt: Forcing completion again")
                            final_force = await self._force_onboarding_completion(validated_user_id)
                            if final_force:
                                print(f"✅ Final force completion successful")
                                # One more verification after final force
                                await asyncio.sleep(0.5)
                                final_verification = await self.get_user_profile(validated_user_id)
                                if final_verification and final_verification.onboarding_completed:
                                    print(f"✅ Final verification after force: SUCCESS")
                                    return True
                                else:
                                    print(f"❌ Final verification after force: FAILED")
                            
                except Exception as verify_error:
                    print(f"⚠️ Verification attempt {attempt + 1} error: {verify_error}")
                    continue
            
            print(f"❌ ENHANCED VERIFICATION FAILED after {verification_attempts} attempts")
            logger.error(f"Enhanced verification failed for user_id: {validated_user_id}")
            return False

        except Exception as e:
            logger.error(f"Error completing onboarding for {user_id}: {e}")
            print(f"❌ Exception in complete_onboarding: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @database_retry
    async def _force_onboarding_completion(self, user_id: str) -> bool:
        """Force onboarding completion with enhanced reliability and retry strategies"""
        try:
            # Validate user_id first
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in _force_onboarding_completion: {validated_user_id}")
                return False
            
            logger.info(f"Force completing onboarding for user_id: {validated_user_id}")
            print(f"🔧 Force completing onboarding for user: {validated_user_id}")
            
            # Check if profile exists first
            existing_profile = await self.get_user_profile(validated_user_id)
            if not existing_profile:
                print(f"❌ No profile exists, cannot force complete onboarding")
                return False
            
            print(f"📋 Existing profile found: email={existing_profile.email}")
            
            # Multiple force completion attempts with different strategies
            
            # Strategy 1: Direct update with minimal data
            try:
                print(f"🔧 Strategy 1: Direct minimal update")
                response = self.client.table("user_profiles").update({
                    "onboarding_completed": True,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("user_id", validated_user_id).execute()
                
                if len(response.data) > 0:
                    print(f"✅ Strategy 1 successful")
                    
                    # Immediate verification
                    await asyncio.sleep(0.1)
                    check_profile = await self.get_user_profile(validated_user_id)
                    if check_profile and check_profile.onboarding_completed:
                        print(f"✅ Strategy 1 verified successfully")
                        return True
                    else:
                        print(f"⚠️ Strategy 1 update succeeded but verification failed")
                else:
                    print(f"❌ Strategy 1 failed - no rows updated")
                    
            except Exception as strategy1_error:
                print(f"❌ Strategy 1 error: {strategy1_error}")
            
            # Strategy 2: Update with existing profile data + onboarding flag
            try:
                print(f"🔧 Strategy 2: Update with preserved data")
                update_data = {
                    "onboarding_completed": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "email": existing_profile.email,  # Preserve critical data
                    "is_active": True
                }
                
                # Preserve other critical fields if they exist
                if existing_profile.name:
                    update_data["name"] = existing_profile.name
                if existing_profile.group_code:
                    update_data["group_code"] = existing_profile.group_code
                
                response = self.client.table("user_profiles").update(update_data).eq("user_id", validated_user_id).execute()
                
                if len(response.data) > 0:
                    print(f"✅ Strategy 2 successful")
                    
                    # Immediate verification
                    await asyncio.sleep(0.1)
                    check_profile = await self.get_user_profile(validated_user_id)
                    if check_profile and check_profile.onboarding_completed:
                        print(f"✅ Strategy 2 verified successfully")
                        return True
                    else:
                        print(f"⚠️ Strategy 2 update succeeded but verification failed")
                else:
                    print(f"❌ Strategy 2 failed - no rows updated")
                    
            except Exception as strategy2_error:
                print(f"❌ Strategy 2 error: {strategy2_error}")
            
            print(f"❌ All force completion strategies failed")
            logger.error(f"All force completion strategies failed for user_id: {validated_user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error in force onboarding completion for {user_id}: {e}")
            print(f"❌ Error in force completion: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== ONBOARDING OPERATIONS ====================
    
    @database_retry
    async def get_onboarding_progress(self, user_id: str) -> Optional[OnboardingProgress]:
        """
        Get onboarding progress for user with retry strategies and better JSON handling
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_onboarding_progress: {validated_user_id}")
                return None
            
            logger.info(f"Getting onboarding progress for user_id: {validated_user_id}")
            
            response = self.client.table("onboarding_progress").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                progress_data = response.data[0]
                
                # Better handling of collected_data JSON parsing
                collected_data = progress_data.get("collected_data")
                if collected_data:
                    if isinstance(collected_data, str):
                        try:
                            # Try to parse the JSON string
                            parsed_data = json.loads(collected_data)
                            progress_data["collected_data"] = parsed_data
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse collected_data JSON for user {validated_user_id}: {e}")
                            logger.error(f"Corrupted data: {collected_data[:200]}...")
                            # Set to empty dict if JSON is corrupted
                            progress_data["collected_data"] = {}
                    elif not isinstance(collected_data, dict):
                        logger.warning(f"collected_data is neither string nor dict for user {validated_user_id}, setting to empty dict")
                        progress_data["collected_data"] = {}
                else:
                    progress_data["collected_data"] = {}
                
                logger.info(f"Onboarding progress found for user_id: {validated_user_id}")
                
                # Validate with Pydantic model
                try:
                    return OnboardingProgress(**progress_data)
                except ValidationError as ve:
                    logger.error(f"Onboarding progress validation error for user {validated_user_id}: {ve}")
                    return None
            
            logger.info(f"No onboarding progress found for user_id: {validated_user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting onboarding progress for {user_id}: {e}")
            return None
    
    @database_retry
    async def update_onboarding_progress(self, user_id: str, step: OnboardingStep, data: Dict[str, Any]) -> bool:
        """
        Update onboarding progress with retry strategies and better JSON handling
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in update_onboarding_progress: {validated_user_id}")
                return False
            
            logger.info(f"Updating onboarding progress for user_id: {validated_user_id}, step: {step.value}")
            
            # Ensure data is properly formatted for JSON storage
            collected_data_json = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            
            progress_data = {
                "user_id": validated_user_id,
                "current_step": step.value,
                "collected_data": collected_data_json,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Try to update first, if no rows affected, insert
            response = self.client.table("onboarding_progress").update(progress_data).eq("user_id", validated_user_id).execute()
            
            if not response.data:
                # No existing record, create new one
                progress_data["created_at"] = datetime.utcnow().isoformat()
                response = self.client.table("onboarding_progress").insert(progress_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Onboarding progress updated successfully for user_id: {validated_user_id}")
            else:
                logger.error(f"Failed to update onboarding progress for user_id: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating onboarding progress for {user_id}: {e}")
            return False
    
    # ==================== GROUP OPERATIONS ====================
    
    @database_retry
    async def get_group_members(self, group_code: str) -> List[GroupMember]:
        """
        Get all members of a group with retry strategies
        """
        try:
            logger.info(f"Getting group members for group_code: {group_code}")
            
            response = self.client.table("group_members").select("*").eq("group_code", group_code).eq("is_active", True).execute()
            
            members = []
            for member_data in response.data:
                try:
                    members.append(GroupMember(**member_data))
                except ValidationError as ve:
                    logger.error(f"Group member validation error for group {group_code}: {ve}")
                    continue
                    
            logger.info(f"Found {len(members)} members for group_code: {group_code}")
            return members
            
        except Exception as e:
            logger.error(f"Error getting group members for {group_code}: {e}")
            return []
    
    @database_retry
    async def add_group_member(self, group_code: str, user_id: str, role: str = "member") -> bool:
        """
        Add user to a group with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in add_group_member: {validated_user_id}")
                return False
            
            logger.info(f"Adding user {validated_user_id} to group {group_code} with role {role}")
            
            member_data = {
                "id": str(uuid.uuid4()),
                "group_code": group_code.upper(),
                "user_id": validated_user_id,
                "role": role,
                "joined_at": datetime.utcnow().isoformat(),
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").insert(member_data).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {validated_user_id} added to group {group_code} successfully")
            else:
                logger.warning(f"Failed to add user {validated_user_id} to group {group_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding group member {user_id} to {group_code}: {e}")
            # Check if it's a duplicate entry error
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                logger.info(f"User {user_id} is already a member of group {group_code}")
                return True  # Consider it successful if already a member
            return False
    
    @database_retry
    async def remove_group_member(self, group_code: str, user_id: str) -> bool:
        """
        Remove user from a group (soft delete) with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.error(f"Invalid user_id format in remove_group_member: {validated_user_id}")
                return False
            
            logger.info(f"Removing user {validated_user_id} from group {group_code}")
            
            updates = {
                "is_active": False,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table("group_members").update(updates).eq("group_code", group_code).eq("user_id", validated_user_id).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User {validated_user_id} removed from group {group_code} successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing group member {user_id} from {group_code}: {e}")
            return False
    
    @database_retry
    async def check_group_exists(self, group_code: str) -> bool:
        """
        Check if group code exists with retry strategies
        """
        try:
            logger.info(f"Checking if group exists: {group_code}")
            
            response = self.client.table("group_members").select("group_code").eq("group_code", group_code.upper()).eq("is_active", True).limit(1).execute()
            
            exists = len(response.data) > 0
            logger.info(f"Group {group_code} exists: {exists}")
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking group existence for {group_code}: {e}")
            return False
    
    @database_retry
    async def get_existing_group_codes(self) -> set:
        """
        Get all existing group codes with retry strategies
        """
        try:
            logger.info("Getting all existing group codes")
            
            response = self.client.table("group_members").select("group_code").eq("is_active", True).execute()
            
            codes = {member["group_code"] for member in response.data}
            logger.info(f"Found {len(codes)} existing group codes")
            
            return codes
            
        except Exception as e:
            logger.error(f"Error getting existing group codes: {e}")
            return set()
    
    @database_retry
    async def get_user_groups(self, user_id: str) -> List[str]:
        """
        Get all groups a user belongs to with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_groups: {validated_user_id}")
                return []
            
            logger.info(f"Getting groups for user: {validated_user_id}")
            
            response = self.client.table("group_members").select("group_code").eq("user_id", validated_user_id).eq("is_active", True).execute()
            
            groups = [member["group_code"] for member in response.data]
            logger.info(f"User {validated_user_id} belongs to {len(groups)} groups")
            
            return groups
            
        except Exception as e:
            logger.error(f"Error getting user groups for {user_id}: {e}")
            return []
    
    # ==================== USER STATUS CHECK ====================
    
    @database_retry
    async def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user status including onboarding completion with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_status: {validated_user_id}")
                return None
            
            logger.info(f"Getting user status for user: {validated_user_id}")
            
            profile = await self.get_user_profile(validated_user_id)
            if not profile:
                logger.warning(f"No profile found for user: {validated_user_id}")
                return None
            
            # Get onboarding progress
            onboarding = await self.get_onboarding_progress(validated_user_id)
            
            # Get user groups
            groups = await self.get_user_groups(validated_user_id)
            
            status = {
                "user_id": validated_user_id,
                "email": profile.email,
                "onboarding_completed": profile.onboarding_completed,
                "is_active": profile.is_active,
                "current_step": onboarding.current_step.value if onboarding else "group_code",
                "completion_percentage": onboarding.completion_percentage if onboarding else 0.0,
                "groups": groups,
                "profile": profile.dict()
            }
            
            logger.info(f"User status retrieved for user: {validated_user_id}")
            return status
            
        except Exception as e:
            logger.error(f"Error getting user status for {user_id}: {e}")
            return None
    
    # ==================== CHAT OPERATIONS ====================
    
    @database_retry
    async def save_chat_message(self, message: ChatMessage) -> bool:
        """
        Save a chat message with retry strategies and better validation
        """
        try:
            # Enhanced validation before saving
            if not message.user_id:
                logger.error("Cannot save chat message: user_id is missing")
                return False
            
            if not message.session_id:
                logger.error("Cannot save chat message: session_id is missing")
                return False
            
            # Validate UUIDs
            user_valid, _ = self._validate_and_format_uuid(message.user_id, "user_id")
            session_valid, _ = self._validate_and_format_uuid(message.session_id, "session_id")
            
            if not user_valid or not session_valid:
                logger.error(f"Cannot save chat message: invalid UUID format")
                return False
            
            logger.info(f"Saving chat message for user: {message.user_id}, session: {message.session_id}")
            
            message_dict = message.dict()
            message_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert metadata to JSON if present
            if message_dict.get("metadata"):
                message_dict["metadata"] = json.dumps(message_dict["metadata"])
            
            response = self.client.table("chat_messages").insert(message_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Chat message saved successfully for user: {message.user_id}")
            else:
                logger.error(f"Failed to save chat message for user: {message.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving chat message for {message.user_id}: {e}")
            return False
    
    @database_retry
    async def get_conversation_history(self, user_id: str, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """
        Get conversation history for a session with retry strategies and better error handling
        """
        try:
            # Better UUID validation with specific field names
            is_valid_user, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            is_valid_session, validated_session_id = self._validate_and_format_uuid(session_id, "session_id")
            
            if not is_valid_user:
                logger.warning(f"Invalid user_id format in get_conversation_history: {user_id}")
                return []
                
            if not is_valid_session:
                logger.warning(f"Invalid session_id format in get_conversation_history: {session_id}")
                return []
            
            logger.info(f"Getting conversation history for user: {validated_user_id}, session: {validated_session_id}")
            
            response = (self.client.table("chat_messages")
                       .select("*")
                       .eq("user_id", validated_user_id)
                       .eq("session_id", validated_session_id)
                       .order("created_at", desc=False)
                       .limit(limit)
                       .execute())
            
            messages = []
            for msg_data in response.data:
                # Parse metadata JSON if present
                if msg_data.get("metadata") and isinstance(msg_data["metadata"], str):
                    try:
                        msg_data["metadata"] = json.loads(msg_data["metadata"])
                    except json.JSONDecodeError:
                        msg_data["metadata"] = {}
                
                try:
                    messages.append(ChatMessage(**msg_data))
                except ValidationError as ve:
                    logger.error(f"Chat message validation error: {ve}")
                    continue
            
            logger.info(f"Retrieved {len(messages)} messages for session: {validated_session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history for {user_id}: {e}")
            return []
    
    @database_retry
    async def save_conversation_turn(self, user_id: str, session_id: str, user_message: str, 
                                   assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save a complete conversation turn (user message + assistant response) with retry strategies
        """
        try:
            # Enhanced validation before creating ChatMessage objects
            if not user_id or not session_id:
                logger.error(f"Cannot save conversation: user_id or session_id is missing")
                return False
            
            # Validate UUIDs
            user_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            session_valid, validated_session_id = self._validate_and_format_uuid(session_id, "session_id")
            
            if not user_valid or not session_valid:
                logger.error(f"Cannot save conversation: invalid UUID format")
                return False
            
            logger.info(f"Saving conversation turn for user: {validated_user_id}, session: {validated_session_id}")
            
            # Save user message
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=validated_user_id,
                session_id=validated_session_id,
                message_type=MessageType.USER,
                content=user_message,
                metadata=metadata or {}
            )
            
            # Save assistant message
            assistant_msg = ChatMessage(
                id=str(uuid.uuid4()),
                user_id=validated_user_id,
                session_id=validated_session_id,
                message_type=MessageType.ASSISTANT,
                content=assistant_response,
                metadata=metadata or {}
            )
            
            # Save both messages
            user_saved = await self.save_chat_message(user_msg)
            assistant_saved = await self.save_chat_message(assistant_msg)
            
            success = user_saved and assistant_saved
            if success:
                logger.info(f"Conversation turn saved successfully for user: {validated_user_id}")
            else:
                logger.error(f"Failed to save conversation turn for user: {validated_user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving conversation turn for {user_id}: {e}")
            return False
    
    # ==================== SEARCH OPERATIONS ====================
    
    @database_retry
    async def save_flight_search(self, search: FlightSearch) -> bool:
        """
        Save flight search results with retry strategies
        """
        try:
            logger.info(f"Saving flight search for user: {search.user_id}")
            
            search_dict = search.dict()
            search_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to strings
            if search_dict.get("departure_date"):
                search_dict["departure_date"] = search_dict["departure_date"].isoformat()
            if search_dict.get("return_date"):
                search_dict["return_date"] = search_dict["return_date"].isoformat()
            
            # Convert search results to JSON
            search_dict["search_results"] = json.dumps(search_dict["search_results"])
            
            response = self.client.table("flight_searches").insert(search_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Flight search saved successfully for user: {search.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving flight search for {search.user_id}: {e}")
            return False
    
    @database_retry
    async def save_hotel_search(self, search: HotelSearch) -> bool:
        """
        Save hotel search results with retry strategies
        """
        try:
            logger.info(f"Saving hotel search for user: {search.user_id}")
            
            search_dict = search.dict()
            search_dict["created_at"] = datetime.utcnow().isoformat()
            
            # Convert date fields to strings
            if search_dict.get("check_in_date"):
                search_dict["check_in_date"] = search_dict["check_in_date"].isoformat()
            if search_dict.get("check_out_date"):
                search_dict["check_out_date"] = search_dict["check_out_date"].isoformat()
            
            # Convert search results to JSON
            search_dict["search_results"] = json.dumps(search_dict["search_results"])
            
            response = self.client.table("hotel_searches").insert(search_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"Hotel search saved successfully for user: {search.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving hotel search for {search.user_id}: {e}")
            return False
    
    # ==================== USER PREFERENCES OPERATIONS ====================
    
    @database_retry
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences with retry strategies
        """
        try:
            # Use consistent UUID validation
            is_valid, validated_user_id = self._validate_and_format_uuid(user_id, "user_id")
            if not is_valid:
                logger.warning(f"Invalid user_id format in get_user_preferences: {validated_user_id}")
                return None
            
            logger.info(f"Getting user preferences for user: {validated_user_id}")
            
            response = self.client.table("user_preferences").select("*").eq("user_id", validated_user_id).execute()
            
            if response.data and len(response.data) > 0:
                prefs_data = response.data[0]
                
                # Parse JSON fields
                for field in ["preferred_airlines", "preferred_hotels", "dietary_restrictions", "accessibility_needs", "trip_types"]:
                    if prefs_data.get(field) and isinstance(prefs_data[field], str):
                        try:
                            prefs_data[field] = json.loads(prefs_data[field])
                        except json.JSONDecodeError:
                            prefs_data[field] = []
                
                if prefs_data.get("budget_range") and isinstance(prefs_data["budget_range"], str):
                    try:
                        prefs_data["budget_range"] = json.loads(prefs_data["budget_range"])
                    except json.JSONDecodeError:
                        prefs_data["budget_range"] = {}
                
                try:
                    return UserPreferences(**prefs_data)
                except ValidationError as ve:
                    logger.error(f"User preferences validation error for user {validated_user_id}: {ve}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            return None
    
    @database_retry
    async def save_user_preferences(self, preferences: UserPreferences) -> bool:
        """
        Save or update user preferences with retry strategies
        """
        try:
            logger.info(f"Saving user preferences for user: {preferences.user_id}")
            
            prefs_dict = preferences.dict()
            prefs_dict["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert list and dict fields to JSON
            for field in ["preferred_airlines", "preferred_hotels", "dietary_restrictions", "accessibility_needs", "trip_types"]:
                if prefs_dict.get(field):
                    prefs_dict[field] = json.dumps(prefs_dict[field])
            
            if prefs_dict.get("budget_range"):
                prefs_dict["budget_range"] = json.dumps(prefs_dict["budget_range"])
            
            # Try to update first, if no rows affected, insert
            response = self.client.table("user_preferences").update(prefs_dict).eq("user_id", preferences.user_id).execute()
            
            if not response.data:
                # No existing record, create new one
                prefs_dict["created_at"] = datetime.utcnow().isoformat()
                response = self.client.table("user_preferences").insert(prefs_dict).execute()
            
            success = len(response.data) > 0
            if success:
                logger.info(f"User preferences saved successfully for user: {preferences.user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving user preferences for {preferences.user_id}: {e}")
            return False
    
    # ==================== DATABASE HEALTH AND MAINTENANCE ====================
    
    @database_retry
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection with retry strategies
        """
        try:
            logger.info("Performing database health check")
            
            # Test basic connectivity
            response = self.client.table("user_profiles").select("user_id").limit(1).execute()
            
            # Get table counts (approximate)
            tables_status = {}
            table_configs = {
                "user_profiles": "user_id",
                "chat_messages": "id",
                "group_members": "id",
                "flight_searches": "search_id",
                "hotel_searches": "id",
                "onboarding_progress": "user_id",
                "user_preferences": "user_id"
            }

            for table, primary_key in table_configs.items():
                try:
                    count_response = self.client.table(table).select(primary_key, count="exact").limit(1).execute()
                    tables_status[table] = "healthy"
                except Exception as e:
                    tables_status[table] = f"error: {str(e)}"
            
            health = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "connected",
                "tables": tables_status
            }
            
            logger.info("Database health check completed successfully")
            return health
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "disconnected",
                "error": str(e)
            }

# ==================== MODULE EXPORTS ====================

__all__ = [
    'DatabaseOperations'
]