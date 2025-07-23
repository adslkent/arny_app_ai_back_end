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
        return (
            result.get("error") is not None or
            "warning" in result or
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
        if result and hasattr(result, 'output'):
            return False  # Has expected output structure
        return True  # Missing expected structure
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
            # Get user's groups
            user_groups = await self.db.get_user_groups(user_id)
            
            if not user_groups:
                # If no groups, get individual user profile
                user_profile = await self.db.get_user_profile(user_id)
                if user_profile:
                    profile_dict = user_profile.dict()
                    profile_dict["group_role"] = "individual"
                    return [profile_dict]
                return []
            
            # Process all group members - NO LIMITS
            all_profiles = []
            for group in user_groups:
                try:
                    group_members = await self.db.get_group_members(group.group_code)
                    print(f"📊 Processing group {group.group_code} with {len(group_members)} members - NO MEMBER LIMITS")
                    
                    for member in group_members:
                        try:
                            member_profile = await self.db.get_user_profile(member.user_id)
                            if member_profile:
                                profile_dict = member_profile.dict()
                                profile_dict["group_role"] = member.role
                                profile_dict["group_code"] = group.group_code
                                all_profiles.append(profile_dict)
                        except Exception as e:
                            logger.warning(f"Failed to get profile for member {member.user_id}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to get members for group {group.group_code}: {e}")
                    continue
            
            return all_profiles
            
        except Exception as e:
            logger.error(f"Error getting group profiles: {e}")
            return []

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
        """Create concise profile summary for both individual travelers and groups"""
        try:
            if not group_profiles:
                return "unknown traveler profile"
            
            # Check if this is an individual traveler
            if len(group_profiles) == 1:
                profile = group_profiles[0]
                summary_parts = ["1 individual traveler"]
                
                # Add individual preferences
                if profile.get("travel_style"):
                    summary_parts.append(f"style: {profile['travel_style']}")
                
                if profile.get("city"):
                    summary_parts.append(f"from: {profile['city']}")
                
                # Calculate age if available
                if profile.get("birthdate"):
                    try:
                        from datetime import date
                        birth_year = int(profile["birthdate"][:4])
                        current_year = date.today().year
                        age = current_year - birth_year
                        if 0 < age < 120:
                            summary_parts.append(f"age: {age}")
                    except:
                        pass
                
                if profile.get("annual_income"):
                    summary_parts.append(f"income: {profile['annual_income']}")
                
                if profile.get("holiday_preferences"):
                    try:
                        if isinstance(profile["holiday_preferences"], list):
                            preferences = profile["holiday_preferences"][:2]  # First 2
                        else:
                            preferences = str(profile["holiday_preferences"])[:30]  # First 30 chars
                        summary_parts.append(f"prefers: {preferences}")
                    except:
                        pass
                
                return ", ".join(summary_parts)
            
            else:
                # Group travel summary
                summary_parts = [f"{len(group_profiles)} group travelers"]
                
                # Collect group preferences
                styles = [p.get("travel_style") for p in group_profiles if p.get("travel_style")]
                cities = [p.get("city") for p in group_profiles if p.get("city")]
                
                if styles:
                    unique_styles = list(set(styles[:3]))  # Up to 3 unique styles
                    summary_parts.append(f"styles: {unique_styles}")
                
                if cities:
                    unique_cities = list(set(cities[:2]))  # Up to 2 unique cities
                    summary_parts.append(f"from: {unique_cities}")
                
                # Group age range
                ages = []
                for profile in group_profiles:
                    if profile.get("birthdate"):
                        try:
                            from datetime import date
                            birth_year = int(profile["birthdate"][:4])
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