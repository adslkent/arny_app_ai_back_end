"""
Enhanced User Profile Agent with Group Support and AI Filtering - ENHANCED VERSION

This module provides user profile management and intelligent filtering for travel search results.
Enhanced with support for larger datasets and always applies AI filtering for both individuals and groups.

Key Features:
- Support for up to 50 flight/hotel results
- Always applies AI filtering for both single travelers and groups  
- Enhanced group profile processing with no member limits
- Optimized caching for better performance
- Comprehensive retry strategies for API reliability

Usage example:
```python
from user_profile_agent import UserProfileAgent

# Initialize agent
profile_agent = UserProfileAgent()

# Filter flight results
result = await profile_agent.filter_flight_results(user_id, flight_results, search_params)
```
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    retry_any,
    retry_if_result,
    before_sleep_log,
    retry_if_exception
)
from pydantic import ValidationError

from ..utils.config import config
from ..database.operations import DatabaseOperations

# Configure logging
logger = logging.getLogger(__name__)

# ==================== OPENAI API RETRY CONDITIONS ====================

def retry_on_openai_api_error_result(result):
    """Condition 2: Retry if OpenAI API result contains error/warning fields"""
    if hasattr(result, 'error') and result.error:
        return True
    if isinstance(result, dict):
        # FIXED: Don't retry on flight/hotel filtering results
        if 'filtered_results' in result:
            return False  # This is a valid filtering result, not an API error
            
        # Only check for top-level error/warning fields, not deep in nested data
        return (
            result.get("error") is not None or
            result.get("warning") is not None or  # FIXED: Check for warning field, not word in result
            result.get("success") is False
        )
    return False

def retry_on_openai_http_status_error(result):
    """Condition 1: Retry on HTTP status errors for OpenAI API calls"""
    if hasattr(result, 'status_code'):
        return result.status_code >= 400
    if hasattr(result, 'response') and hasattr(result.response, 'status_code'):
        return result.response.status_code >= 400
    return False

def retry_on_openai_validation_failure(result):
    """Condition 5: Retry if validation fails on OpenAI result"""
    try:
        # FIXED: Check for flight/hotel filtering result structure
        if isinstance(result, dict):
            # If it's a filtering result (has filtered_results key), it's valid
            if 'filtered_results' in result:
                return False  # Valid filtering result
            # If it has an output attribute (OpenAI response), it's valid
            if hasattr(result, 'output'):
                return False  # Valid OpenAI response
        
        # If result exists and has expected structure, don't retry
        if result and hasattr(result, 'output'):
            return False  # Has expected output structure
        
        # Only retry if result is None, empty, or clearly invalid
        return result is None or (isinstance(result, str) and len(result.strip()) == 0)
        
    except Exception:
        return True  # Any validation error

def retry_on_openai_api_exception(exception):
    """Condition 4: Custom exception checker for OpenAI API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'rate limit', 'api error',
        'connection', 'network', 'server error'
    ])

# ==================== RETRY DECORATORS ====================

openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_openai_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_openai_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class UserProfileAgent:
    """
    ENHANCED: User Profile Agent with support for larger result sets - NO TIMEOUTS OR MEMBER LIMITS
    ALWAYS APPLIES AI FILTERING for both single travelers and groups
    """
    
    def __init__(self):
        """Initialize with enhanced settings for larger datasets"""
        try:
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.db = DatabaseOperations()
            
            # Initialize result cache for instant responses
            self._filter_cache = {}
            
            logger.info("UserProfileAgent initialized with enhanced support for larger datasets - ALWAYS APPLIES AI FILTERING")
        except Exception as e:
            logger.error(f"Failed to initialize UserProfileAgent: {e}")
            raise Exception(f"UserProfileAgent initialization failed: {e}")
    
    async def filter_flight_results(self, user_id: str, flight_results: List[Dict[str, Any]], 
                                  search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Filter flight results with support for up to 50 flights, returning up to 10 - ALWAYS APPLIES AI FILTERING
        """
        try:
            logger.info(f"ENHANCED flight filtering for user {user_id} - ALWAYS APPLIES AI FILTERING")
            
            print(f"🚀 ENHANCED: Processing {len(flight_results)} flight results - ALWAYS APPLIES AI FILTERING")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"flight_{user_id}_{len(flight_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached flight filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles - NO TIMEOUT LIMIT, ALL MEMBERS
            try:
                group_profiles = await self._get_group_profiles_enhanced(user_id)
                print(f"📊 Retrieved {len(group_profiles)} group profiles - NO MEMBER LIMITS")
            except Exception as e:
                logger.warning(f"Failed to get group profiles: {e}")
                group_profiles = []
            
            # OPTIMIZATION 3: Direct return if no flights
            if not flight_results:
                result = {
                    "filtered_results": [],
                    "total_results": 0,
                    "filtering_applied": False,
                    "reasoning": "No flight results to filter"
                }
                self._filter_cache[cache_key] = result
                return result

            # REMOVED: Single traveler bypass condition - ALWAYS APPLY AI FILTERING NOW
            # OLD CODE WAS: if len(group_profiles) <= 1: return top_flights without filtering

            # OPTIMIZATION 4: Enhanced data preparation for all users (single and groups)
            enhanced_flights = self._extract_enhanced_flight_data(flight_results)
            
            if not enhanced_flights:
                result = {
                    "filtered_results": [],
                    "total_results": len(flight_results),
                    "filtering_applied": False,
                    "reasoning": "Could not process flight data"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 5: Create profile summary (works for both single travelers and groups)
            profile_summary = self._create_profile_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_flights)} flights with profile: {profile_summary}")

            # OPTIMIZATION 6: AI filtering applied for ALL users with NO TIMEOUT LIMITS
            filtered_result = await self._filter_flights_with_ai_enhanced(
                enhanced_flights, profile_summary, len(group_profiles) or 1, flight_results
            )
            
            # Cache and return result
            self._filter_cache[cache_key] = filtered_result
            self._cleanup_cache()
            
            print(f"✅ ENHANCED flight filtering complete - returned {len(filtered_result.get('filtered_results', []))} flights")
            return filtered_result

        except Exception as e:
            logger.error(f"Error in enhanced flight filtering: {e}")
            import traceback
            traceback.print_exc()
            
            # Return graceful fallback
            return {
                "filtered_results": flight_results[:10] if flight_results else [],
                "total_results": len(flight_results) if flight_results else 0,
                "filtering_applied": False,
                "reasoning": f"Filtering failed: {str(e)}"
            }

    async def filter_hotel_results(self, user_id: str, hotel_results: List[Dict[str, Any]], 
                                 search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Filter hotel results with support for up to 50 hotels, returning up to 10 - ALWAYS APPLIES AI FILTERING
        """
        try:
            logger.info(f"ENHANCED hotel filtering for user {user_id} - ALWAYS APPLIES AI FILTERING")
            
            print(f"🚀 ENHANCED: Processing {len(hotel_results)} hotel results - ALWAYS APPLIES AI FILTERING")

            # OPTIMIZATION 1: Check cache first
            cache_key = f"hotel_{user_id}_{len(hotel_results)}"
            if cache_key in self._filter_cache:
                print(f"⚡ CACHE HIT: Returning cached hotel filtering")
                return self._filter_cache[cache_key]

            # OPTIMIZATION 2: Get group profiles - NO TIMEOUT LIMIT, ALL MEMBERS
            try:
                group_profiles = await self._get_group_profiles_enhanced(user_id)
                print(f"📊 Retrieved {len(group_profiles)} group profiles - NO MEMBER LIMITS")
            except Exception as e:
                logger.warning(f"Failed to get group profiles: {e}")
                group_profiles = []
            
            # OPTIMIZATION 3: Direct return if no hotels
            if not hotel_results:
                result = {
                    "filtered_results": [],
                    "total_results": 0,
                    "filtering_applied": False,
                    "reasoning": "No hotel results to filter"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 4: Enhanced data preparation for all users (single and groups)
            enhanced_hotels = self._extract_enhanced_hotel_data(hotel_results)
            
            if not enhanced_hotels:
                result = {
                    "filtered_results": [],
                    "total_results": len(hotel_results),
                    "filtering_applied": False,
                    "reasoning": "Could not process hotel data"
                }
                self._filter_cache[cache_key] = result
                return result

            # OPTIMIZATION 5: Create profile summary (works for both single travelers and groups)
            profile_summary = self._create_profile_summary(group_profiles)
            
            print(f"🧠 Starting AI filtering for {len(enhanced_hotels)} hotels with profile: {profile_summary}")

            # OPTIMIZATION 6: AI filtering applied for ALL users with NO TIMEOUT LIMITS
            filtered_result = await self._filter_hotels_with_ai_enhanced(
                enhanced_hotels, profile_summary, len(group_profiles) or 1, hotel_results
            )
            
            # Cache and return result
            self._filter_cache[cache_key] = filtered_result
            self._cleanup_cache()
            
            print(f"✅ ENHANCED hotel filtering complete - returned {len(filtered_result.get('filtered_results', []))} hotels")
            return filtered_result

        except Exception as e:
            logger.error(f"Error in enhanced hotel filtering: {e}")
            import traceback
            traceback.print_exc()
            
            # Return graceful fallback
            return {
                "filtered_results": hotel_results[:10] if hotel_results else [],
                "total_results": len(hotel_results) if hotel_results else 0,
                "filtering_applied": False,
                "reasoning": f"Filtering failed: {str(e)}"
            }

    async def _get_group_profiles_enhanced(self, user_id: str) -> List[Dict[str, Any]]:
        """Get group profiles with optimized processing - NO TIMEOUTS OR MEMBER LIMITS"""
        try:
            print(f"🔍 Getting enhanced group profiles for user: {user_id}")
            
            # Get user's individual profile first
            user_profile = await self.db.get_user_profile(user_id)
            print(f"🔍 Raw user profile type: {type(user_profile)}")
            
            if not user_profile:
                print(f"🔍 No user profile found for user: {user_id}")
                return []
            
            # ENHANCED: Debug the profile object and safely extract data
            profile_dict = None
            try:
                print(f"🔍 Profile object attributes: {dir(user_profile)}")
                
                if hasattr(user_profile, 'dict') and callable(user_profile.dict):
                    # Pydantic model
                    profile_dict = user_profile.dict()
                    print(f"🔍 Pydantic profile dict created: {list(profile_dict.keys())}")
                elif isinstance(user_profile, dict):
                    # Already a dict
                    profile_dict = user_profile.copy()
                    print(f"🔍 Profile was already dict: {list(profile_dict.keys())}")
                else:
                    # Try to convert object to dict manually
                    profile_dict = {}
                    for attr in ['user_id', 'name', 'email', 'group_code', 'city', 'annual_income', 'holiday_preferences']:
                        if hasattr(user_profile, attr):
                            profile_dict[attr] = getattr(user_profile, attr)
                    print(f"🔍 Manual profile dict created: {list(profile_dict.keys())}")
            except Exception as e:
                print(f"[ERROR] Profile conversion failed: {e}")
                # Create minimal profile dict
                profile_dict = {
                    "user_id": user_id,
                    "name": "User",
                    "email": getattr(user_profile, 'email', ''),
                }
            
            # Extract group_code safely
            group_code = profile_dict.get('group_code') if profile_dict else None
            print(f"🔍 Extracted group_code: {group_code}")
            
            # Get user's groups
            try:
                user_groups = await self.db.get_user_groups(user_id)
                print(f"🔍 User groups from DB: {len(user_groups) if user_groups else 0}")
            except Exception as e:
                print(f"[ERROR] Failed to get user groups: {e}")
                user_groups = []
            
            if not user_groups and not group_code:
                # Return individual user profile
                print(f"🔍 User {user_id} not in any group - returning individual profile")
                
                if profile_dict:
                    profile_dict["group_role"] = "individual"
                    profile_dict["group_code"] = None
                    print(f"✅ Returning individual profile: {profile_dict.get('name', 'Unknown')} from {profile_dict.get('city', 'Unknown')}")
                    return [profile_dict]
                else:
                    # Last resort fallback
                    fallback_profile = {
                        "user_id": user_id,
                        "name": "User",
                        "group_role": "individual",
                        "group_code": None
                    }
                    print(f"⚠️ Using fallback profile for {user_id}")
                    return [fallback_profile]
            
            # Process group members if user has group
            all_profiles = []
            for group_code in user_groups:  # FIXED: user_groups contains strings, not objects
                try:
                    # FIXED: group_code is already a string, don't access .group_code
                    group_members = await self.db.get_group_members(group_code)
                    print(f"📊 Processing group {group_code} with {len(group_members)} members")
                    
                    for member in group_members:
                        try:
                            member_profile = await self.db.get_user_profile(member.user_id)
                            if member_profile:
                                # Convert member profile to dict
                                if hasattr(member_profile, 'dict') and callable(member_profile.dict):
                                    member_dict = member_profile.dict()
                                elif isinstance(member_profile, dict):
                                    member_dict = member_profile.copy()
                                else:
                                    member_dict = {
                                        "user_id": member.user_id,
                                        "name": getattr(member_profile, 'name', 'Member'),
                                        "email": getattr(member_profile, 'email', ''),
                                    }
                                
                                member_dict["group_role"] = member.role
                                member_dict["group_code"] = group_code  # FIXED: Use group_code string directly
                                all_profiles.append(member_dict)
                        except Exception as e:
                            print(f"[WARNING] Failed to get profile for member {member.user_id}: {e}")
                            continue
                except Exception as e:
                    print(f"[WARNING] Failed to get members for group {group_code}: {e}")  # FIXED: Use group_code string directly
                    continue
            
            if all_profiles:
                print(f"✅ Returning {len(all_profiles)} group profiles")
                return all_profiles
            else:
                # Fallback to individual profile even if groups were found but failed to process
                if profile_dict:
                    profile_dict["group_role"] = "individual"
                    profile_dict["group_code"] = None
                    print(f"⚠️ Group processing failed, falling back to individual profile")
                    return [profile_dict]
                else:
                    print(f"⚠️ Everything failed, using minimal fallback")
                    return [{
                        "user_id": user_id,
                        "name": "User",
                        "group_role": "individual",
                        "group_code": None
                    }]
            
        except Exception as e:
            print(f"[ERROR] Major error in _get_group_profiles_enhanced: {e}")
            import traceback
            traceback.print_exc()
            
            # Ultimate fallback
            return [{
                "user_id": user_id,
                "name": "User", 
                "group_role": "individual",
                "group_code": None,
                "error": str(e)
            }]

    async def _get_group_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all group profiles for filtering - NO MEMBER LIMITS - FIXED VERSION"""
        try:
            print(f"🔍 Getting group profiles for user: {user_id}")
            
            # Get user's primary profile first
            user_profile = await self.db.get_user_profile(user_id)
            
            if not user_profile:
                logger.warning(f"No user profile found for user {user_id}")
                return []
            
            # Handle group_code access properly - FIXED
            group_code = None
            if hasattr(user_profile, 'group_code'):
                group_code = user_profile.group_code
            elif isinstance(user_profile, dict):
                group_code = user_profile.get('group_code')
            
            if not group_code:
                print(f"🔍 User {user_id} not in any group - returning individual profile")
                # Return individual profile as list for consistent processing
                profile_dict = user_profile.__dict__ if hasattr(user_profile, '__dict__') else user_profile
                return [profile_dict] if profile_dict else []
            
            # Get all group members - NO LIMITS
            group_profiles = await self.db.get_group_profiles(group_code)
            print(f"📊 Retrieved {len(group_profiles)} group profiles for group {group_code}")
            
            return group_profiles
            
        except Exception as e:
            logger.error(f"Error getting group profiles: {e}")
            print(f"[ERROR] Error getting group profiles: {e}")
            return []

    def _extract_enhanced_flight_data(self, flights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract flight data for up to 50 flights efficiently"""
        enhanced_flights = []
        
        # Process all flights but optimize data extraction
        for i, flight in enumerate(flights[:50]):  # CHANGED: Process up to 50 flights
            try:
                price = flight.get("price", {})
                itineraries = flight.get("itineraries", [])
                
                enhanced_flight = {
                    "id": i + 1,
                    "price": price.get("total", "N/A"),
                    "currency": price.get("currency", ""),
                    "duration": "",
                    "stops": 0,
                    "departure_time": "",
                    "arrival_time": "",
                    "airline": "",
                }
                
                if itineraries and len(itineraries) > 0:
                    first_itinerary = itineraries[0]
                    segments = first_itinerary.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        enhanced_flight.update({
                            "departure_time": first_segment.get("departure", {}).get("at", ""),
                            "arrival_time": last_segment.get("arrival", {}).get("at", ""),
                            "stops": max(0, len(segments) - 1),
                            "duration": first_itinerary.get("duration", ""),
                            "airline": first_segment.get("carrierCode", "")
                        })
                
                enhanced_flights.append(enhanced_flight)
                
            except Exception as e:
                logger.warning(f"Error processing flight {i}: {e}")
                continue
        
        return enhanced_flights

    def _extract_enhanced_hotel_data(self, hotels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract hotel data for up to 50 hotels efficiently"""
        enhanced_hotels = []
        
        # Process all hotels but optimize data extraction
        for i, hotel in enumerate(hotels[:50]):  # CHANGED: Process up to 50 hotels
            try:
                offers = hotel.get("offers", [])
                first_offer = offers[0] if offers else {}
                
                enhanced_hotel = {
                    "id": i + 1,
                    "name": hotel.get("name", "Unknown Hotel"),
                    "price": first_offer.get("price", {}).get("total", "N/A"),
                    "currency": first_offer.get("price", {}).get("currency", ""),
                    "rating": hotel.get("rating", "N/A"),
                    "address": hotel.get("address", {}).get("lines", [""])[0] if hotel.get("address", {}).get("lines") else "",
                    "amenities": hotel.get("amenities", [])[:5],  # First 5 amenities
                    "distance": hotel.get("distance", {}).get("value", "N/A")
                }
                
                enhanced_hotels.append(enhanced_hotel)
                
            except Exception as e:
                logger.warning(f"Error processing hotel {i}: {e}")
                continue
        
        return enhanced_hotels

    @openai_api_retry
    async def _filter_flights_with_ai_enhanced(self, flights: List[Dict[str, Any]], 
                                             profile_summary: str, member_count: int,
                                             original_flights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: AI-based flight filtering with support for larger datasets - NO TIMEOUT LIMITS"""
        try:
            print(f"🧠 AI filtering {len(flights)} flights for profile: {profile_summary}")
            
            # Create filtering prompt for larger datasets
            prompt = f"""You are a travel expert helping to filter flight options for travelers.

TRAVELER PROFILE: {profile_summary}

FLIGHT OPTIONS ({len(flights)} total):
{json.dumps(flights[:30], indent=2)}

TASK: Select the best 10 flights from the provided options based on the traveler profile.

FILTERING CRITERIA:
1. Price appropriateness for the traveler's budget level
2. Flight timing preferences and convenience
3. Number of stops vs direct flights
4. Airlines and travel style match
5. Overall value for money

RESPONSE FORMAT:
Return ONLY a JSON object with:
{{
    "selected_flight_ids": [1, 3, 5, 7, 9, 12, 15, 18, 20, 25],
    "reasoning": "Brief explanation of selection criteria used"
}}

Select exactly 10 flight IDs from the provided list."""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                ai_response = content_item.text.strip()
                                
                                try:
                                    result = json.loads(ai_response)
                                    selected_ids = result.get("selected_flight_ids", [])
                                    reasoning = result.get("reasoning", "AI filtering applied")
                                    
                                    # Map selected IDs back to original flights
                                    filtered_flights = []
                                    for flight_id in selected_ids[:10]:  # Ensure max 10
                                        if 1 <= flight_id <= len(original_flights):
                                            filtered_flights.append(original_flights[flight_id - 1])
                                    
                                    print(f"✅ AI filtered to {len(filtered_flights)} flights")
                                    
                                    return {
                                        "filtered_results": filtered_flights,
                                        "total_results": len(original_flights),
                                        "filtering_applied": True,
                                        "reasoning": reasoning
                                    }
                                    
                                except json.JSONDecodeError:
                                    logger.warning("AI response was not valid JSON")
            
            # Fallback if AI filtering fails
            print("❌ AI filtering failed, returning top 10 flights")
            return {
                "filtered_results": original_flights[:10],
                "total_results": len(original_flights),
                "filtering_applied": False,
                "reasoning": "AI filtering failed, returned top results"
            }
            
        except Exception as e:
            logger.error(f"Error in AI flight filtering: {e}")
            return {
                "filtered_results": original_flights[:10],
                "total_results": len(original_flights),
                "filtering_applied": False,
                "reasoning": f"AI filtering error: {str(e)}"
            }

    @openai_api_retry
    async def _filter_hotels_with_ai_enhanced(self, hotels: List[Dict[str, Any]], 
                                            profile_summary: str, member_count: int,
                                            original_hotels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ENHANCED: AI-based hotel filtering with support for larger datasets - NO TIMEOUT LIMITS"""
        try:
            print(f"🧠 AI filtering {len(hotels)} hotels for profile: {profile_summary}")
            
            # Create filtering prompt for larger datasets
            prompt = f"""You are a travel expert helping to filter hotel options for travelers.

TRAVELER PROFILE: {profile_summary}

HOTEL OPTIONS ({len(hotels)} total):
{json.dumps(hotels[:30], indent=2)}

TASK: Select the best 10 hotels from the provided options based on the traveler profile.

FILTERING CRITERIA:
1. Price appropriateness for the traveler's budget level
2. Hotel amenities matching traveler preferences
3. Location and accessibility
4. Rating and quality standards
5. Overall value for money

RESPONSE FORMAT:
Return ONLY a JSON object with:
{{
    "selected_hotel_ids": [1, 3, 5, 7, 9, 12, 15, 18, 20, 25],
    "reasoning": "Brief explanation of selection criteria used"
}}

Select exactly 10 hotel IDs from the provided list."""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=prompt
            )
            
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                ai_response = content_item.text.strip()
                                
                                try:
                                    result = json.loads(ai_response)
                                    selected_ids = result.get("selected_hotel_ids", [])
                                    reasoning = result.get("reasoning", "AI filtering applied")
                                    
                                    # Map selected IDs back to original hotels
                                    filtered_hotels = []
                                    for hotel_id in selected_ids[:10]:  # Ensure max 10
                                        if 1 <= hotel_id <= len(original_hotels):
                                            filtered_hotels.append(original_hotels[hotel_id - 1])
                                    
                                    print(f"✅ AI filtered to {len(filtered_hotels)} hotels")
                                    
                                    return {
                                        "filtered_results": filtered_hotels,
                                        "total_results": len(original_hotels),
                                        "filtering_applied": True,
                                        "reasoning": reasoning
                                    }
                                    
                                except json.JSONDecodeError:
                                    logger.warning("AI response was not valid JSON")
            
            # Fallback if AI filtering fails
            print("❌ AI filtering failed, returning top 10 hotels")
            return {
                "filtered_results": original_hotels[:10],
                "total_results": len(original_hotels),
                "filtering_applied": False,
                "reasoning": "AI filtering failed, returned top results"
            }
            
        except Exception as e:
            logger.error(f"Error in AI hotel filtering: {e}")
            return {
                "filtered_results": original_hotels[:10],
                "total_results": len(original_hotels),
                "filtering_applied": False,
                "reasoning": f"AI filtering error: {str(e)}"
            }
        
    def _create_profile_summary(self, group_profiles: List[Dict[str, Any]]) -> str:
        """Create comprehensive profile summary for both individual travelers and groups"""
        try:
            print(f"🔍 _create_profile_summary called with {len(group_profiles)} profiles")
            
            if not group_profiles:
                print(f"🔍 No group profiles provided")
                return "unknown traveler profile"
            
            # Check if this is an individual traveler
            if len(group_profiles) == 1:
                profile = group_profiles[0]
                print(f"🔍 Processing individual profile with keys: {list(profile.keys())}")
                
                # DEBUG: Print actual values for all relevant fields
                print(f"🔍 travel_style: {profile.get('travel_style')} (type: {type(profile.get('travel_style'))})")
                print(f"🔍 city: {profile.get('city')} (type: {type(profile.get('city'))})")
                print(f"🔍 annual_income: {profile.get('annual_income')} (type: {type(profile.get('annual_income'))})")
                print(f"🔍 monthly_spending: {profile.get('monthly_spending')} (type: {type(profile.get('monthly_spending'))})")
                print(f"🔍 holiday_preferences: {profile.get('holiday_preferences')} (type: {type(profile.get('holiday_preferences'))})")
                print(f"🔍 birthdate: {profile.get('birthdate')} (type: {type(profile.get('birthdate'))})")
                print(f"🔍 gender: {profile.get('gender')} (type: {type(profile.get('gender'))})")
                print(f"🔍 employer: {profile.get('employer')} (type: {type(profile.get('employer'))})")
                print(f"🔍 working_schedule: {profile.get('working_schedule')} (type: {type(profile.get('working_schedule'))})")
                print(f"🔍 holiday_frequency: {profile.get('holiday_frequency')} (type: {type(profile.get('holiday_frequency'))})")
                print(f"🔍 name: {profile.get('name')} (type: {type(profile.get('name'))})")
                
                summary_parts = ["1 individual traveler"]
                
                # Add name if available
                if profile.get("name"):
                    summary_parts.append(f"named {profile['name']}")
                    print(f"✅ Added name: {profile['name']}")
                else:
                    print(f"❌ name is empty/None")
                
                # Add travel style
                if profile.get("travel_style"):
                    summary_parts.append(f"style: {profile['travel_style']}")
                    print(f"✅ Added travel_style: {profile['travel_style']}")
                else:
                    print(f"❌ travel_style is empty/None")
                
                # Add location
                if profile.get("city"):
                    summary_parts.append(f"from: {profile['city']}")
                    print(f"✅ Added city: {profile['city']}")
                else:
                    print(f"❌ city is empty/None")
                
                # Calculate age if available
                if profile.get("birthdate"):
                    try:
                        from datetime import date
                        birthdate_value = profile["birthdate"]
                        
                        # Handle both date object and string cases
                        if isinstance(birthdate_value, date):
                            birth_year = birthdate_value.year
                        elif isinstance(birthdate_value, str):
                            birth_year = int(birthdate_value[:4])
                        else:
                            raise ValueError(f"Unexpected birthdate type: {type(birthdate_value)}")
                            
                        current_year = date.today().year
                        age = current_year - birth_year
                        if 0 < age < 120:
                            summary_parts.append(f"age: {age}")
                            print(f"✅ Added age: {age}")
                    except:
                        print(f"❌ Failed to parse birthdate: {profile.get('birthdate')}")
                        pass
                else:
                    print(f"❌ birthdate is empty/None")
                
                # Add gender
                if profile.get("gender"):
                    summary_parts.append(f"gender: {profile['gender']}")
                    print(f"✅ Added gender: {profile['gender']}")
                else:
                    print(f"❌ gender is empty/None")
                
                # Add employment info
                if profile.get("employer"):
                    summary_parts.append(f"works at: {profile['employer']}")
                    print(f"✅ Added employer: {profile['employer']}")
                else:
                    print(f"❌ employer is empty/None")
                
                # Add working schedule
                if profile.get("working_schedule"):
                    summary_parts.append(f"schedule: {profile['working_schedule']}")
                    print(f"✅ Added working_schedule: {profile['working_schedule']}")
                else:
                    print(f"❌ working_schedule is empty/None")
                
                # Add holiday frequency
                if profile.get("holiday_frequency"):
                    summary_parts.append(f"travels: {profile['holiday_frequency']}")
                    print(f"✅ Added holiday_frequency: {profile['holiday_frequency']}")
                else:
                    print(f"❌ holiday_frequency is empty/None")
                
                # Add financial information
                if profile.get("annual_income"):
                    summary_parts.append(f"income: {profile['annual_income']}")
                    print(f"✅ Added income: {profile['annual_income']}")
                else:
                    print(f"❌ annual_income is empty/None")
                
                # Add monthly spending budget
                if profile.get("monthly_spending"):
                    summary_parts.append(f"monthly budget: {profile['monthly_spending']}")
                    print(f"✅ Added monthly_spending: {profile['monthly_spending']}")
                else:
                    print(f"❌ monthly_spending is empty/None")
                
                # Add preferences
                if profile.get("holiday_preferences"):
                    try:
                        if isinstance(profile["holiday_preferences"], list):
                            preferences = profile["holiday_preferences"][:2]  # First 2
                        else:
                            preferences = str(profile["holiday_preferences"])[:30]  # First 30 chars
                        summary_parts.append(f"prefers: {preferences}")
                        print(f"✅ Added preferences: {preferences}")
                    except:
                        print(f"❌ Failed to parse holiday_preferences: {profile.get('holiday_preferences')}")
                        pass
                else:
                    print(f"❌ holiday_preferences is empty/None")
                
                result = ", ".join(summary_parts)
                print(f"🔍 Final enhanced profile summary: {result}")
                return result
            
            else:
                # Group travel summary (enhanced version)
                summary_parts = [f"{len(group_profiles)} group travelers"]
                
                # Collect group data
                styles = [p.get("travel_style") for p in group_profiles if p.get("travel_style")]
                cities = [p.get("city") for p in group_profiles if p.get("city")]
                employers = [p.get("employer") for p in group_profiles if p.get("employer")]
                frequencies = [p.get("holiday_frequency") for p in group_profiles if p.get("holiday_frequency")]
                
                if styles:
                    unique_styles = list(set(styles[:3]))  # Up to 3 unique styles
                    summary_parts.append(f"styles: {unique_styles}")
                
                if cities:
                    unique_cities = list(set(cities[:2]))  # Up to 2 unique cities
                    summary_parts.append(f"from: {unique_cities}")
                
                if frequencies:
                    unique_frequencies = list(set(frequencies[:2]))  # Up to 2 unique frequencies
                    summary_parts.append(f"travel frequency: {unique_frequencies}")
                
                # Group age range
                ages = []
                for profile in group_profiles:
                    if profile.get("birthdate"):
                        try:
                            from datetime import date
                            birthdate_value = profile["birthdate"]
                            
                            # Handle both date object and string cases
                            if isinstance(birthdate_value, date):
                                birth_year = birthdate_value.year
                            elif isinstance(birthdate_value, str):
                                birth_year = int(birthdate_value[:4])
                            else:
                                continue  # Skip invalid birthdate types
                                
                            current_year = date.today().year
                            age = current_year - birth_year
                            if 0 < age < 120:
                                ages.append(age)
                        except:
                            pass
                
                if ages:
                    summary_parts.append(f"ages: {min(ages)}-{max(ages)}")
                
                return ", ".join(summary_parts)
                
        except Exception as e:
            print(f"[ERROR] _create_profile_summary failed: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"Error creating profile summary: {e}")
            return f"{len(group_profiles)} travelers" if group_profiles else "unknown travelers"
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory bloat"""
        try:
            if len(self._filter_cache) > 100:  # Keep cache under 100 entries
                # Remove oldest entries (simple FIFO cleanup)
                keys_to_remove = list(self._filter_cache.keys())[:-50]  # Keep last 50
                for key in keys_to_remove:
                    del self._filter_cache[key]
                print(f"🧹 Cache cleanup: removed {len(keys_to_remove)} old entries")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    # ==================== LEGACY METHODS FOR BACKWARD COMPATIBILITY ====================

    async def filter_flights_enhanced(self, flights: List[Dict[str, Any]], 
                                    user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - redirects to new enhanced filtering"""
        user_id = user_profile.get("user_id", "unknown")
        return await self.filter_flight_results(user_id, flights, {})

    async def filter_hotels_enhanced(self, hotels: List[Dict[str, Any]], 
                                   user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - redirects to new enhanced filtering"""
        user_id = user_profile.get("user_id", "unknown")
        return await self.filter_hotel_results(user_id, hotels, {})
