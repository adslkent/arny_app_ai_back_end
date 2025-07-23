"""
Hotel Search Agent Module - ENHANCED VERSION to match flight agent improvements

This module provides a hotel search agent with enhanced capabilities:
1. Support for up to 50 hotel results from Amadeus API
2. Send all hotels to OpenAI for filtering
3. Return up to 10 filtered hotel results
4. Optimized for larger datasets

Usage example:
```python
from hotel_agent import HotelAgent

# Create and use the agent
agent = HotelAgent()
result = await agent.process_message(user_id, "Find hotels in Paris", session_id, {}, [])
```
"""

import json
import uuid
import logging
import asyncio
import concurrent.futures
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from agents import Agent, function_tool, Runner
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
from pydantic import BaseModel, ValidationError

from ..utils.config import config
from ..services.amadeus_service import AmadeusService
from ..database.operations import DatabaseOperations
from ..database.models import HotelSearch
from .user_profile_agent import UserProfileAgent

# Configure logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class AgentRunnerResponse(BaseModel):
    """Pydantic model for Agent Runner response validation"""
    final_output: Optional[str] = None

# ==================== OPENAI AGENTS SDK RETRY CONDITIONS ====================

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
    """Condition 5: Retry on validation failures"""
    return result is None or (isinstance(result, str) and len(result.strip()) == 0)

def retry_on_openai_api_exception(exception):
    """Condition 3: Check if exception is related to OpenAI API timeouts or rate limits"""
    exception_str = str(exception).lower()
    return (
        "timeout" in exception_str or
        "rate limit" in exception_str or
        "429" in exception_str or
        "502" in exception_str or
        "503" in exception_str or
        "504" in exception_str
    )

# ==================== RETRY DECORATORS ====================

openai_api_retry = retry(
    reraise=True,
    retry=retry_any(
        # Condition 3: OpenAI API exceptions (timeouts, rate limits, server errors)
        # FIXED: Moved (?i) flag to the beginning of the regex pattern
        retry_if_exception_message(match=r"(?i).*(timeout|rate.limit|429|502|503|504).*"),
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

# ===== City Code Mapping for Hotels =====
CITY_CODE_MAPPING = {
    # Major cities to city codes for Amadeus Hotel API
    "new york": "NYC",
    "los angeles": "LAX", 
    "san francisco": "SFO",
    "chicago": "CHI",
    "washington": "WAS",
    "washington dc": "WAS",
    "boston": "BOS",
    "miami": "MIA",
    "las vegas": "LAS",
    "seattle": "SEA",
    "denver": "DEN",
    "atlanta": "ATL",
    
    # International cities
    "london": "LON",
    "paris": "PAR",
    "tokyo": "TYO",
    "sydney": "SYD",
    "melbourne": "MEL",
    "brisbane": "BNE",
    "perth": "PER",
    "adelaide": "ADL",
    "canberra": "CBR",
    "dubai": "DXB",
    "rome": "ROM",
    "madrid": "MAD",
    "barcelona": "BCN",
    "amsterdam": "AMS",
    "berlin": "BER",
    "singapore": "SIN",
    "hong kong": "HKG",
    "bangkok": "BKK",
    "mumbai": "BOM",
    "delhi": "DEL",
    "toronto": "YTO",
    "vancouver": "YVR",
    "montreal": "YMQ"
}

def _convert_to_city_code(destination: str) -> str:
    """Convert destination name to city code"""
    destination_lower = destination.lower().strip()
    
    # Direct mapping
    if destination_lower in CITY_CODE_MAPPING:
        return CITY_CODE_MAPPING[destination_lower]
    
    # Partial matching
    for city, code in CITY_CODE_MAPPING.items():
        if city in destination_lower or destination_lower in city:
            return code
    
    # Fallback: return first 3 letters uppercase
    return destination_lower[:3].upper()

# ===== System Message =====
def get_hotel_system_message() -> str:
    """Generate the hotel agent system message with current date"""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return f"""You are a specialized hotel search assistant powered by Amadeus API. 

Your main responsibilities are:
1. Understanding users' hotel needs and extracting key information from natural language descriptions
2. Using the search_hotels_tool to search for hotels that meet user requirements
3. Presenting hotel options in a clear, organized format

**IMPORTANT PRESENTATION RULES:**
- When you receive hotel search results, ALWAYS present ALL hotels in your response
- Do NOT truncate or summarize the hotel list - show complete details for every result
- Present hotels in an organized, easy-to-read format
- Include all key details: name, price, rating, amenities, location
- Number each hotel option for easy reference

Today's date: {today}
Tomorrow's date: {tomorrow}

**Key Guidelines:**
1. **Date Handling**: If user says "tomorrow", use {tomorrow}. If no year specified, assume current year.

2. **Hotel Search**: Always use search_hotels_tool for hotel requests. Extract:
   - Destination city/location
   - Check-in date (format: YYYY-MM-DD)
   - Check-out date (format: YYYY-MM-DD)
   - Number of adults (default: 1)
   - Number of rooms (default: 1)

3. **Response Style**: Be professional, helpful, and provide clear options. When presenting multiple hotels, organize them clearly and include all important details.

4. **No Booking**: You can search and provide hotel information, but cannot make actual bookings. Direct users to hotel websites or booking platforms for reservations.

5. **Missing Information**: If critical details are missing, ask for clarification before searching.

6. **Date Validation**: Ensure check-in date is in the future and check-out date is after check-in date.

Remember: You are a specialized hotel search agent. Focus on hotel-related queries and use your tools effectively to provide the best accommodation options for users."""

# ==================== ASYNC UTILITIES ====================

def run_async(coro):
    """Run async coroutine in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, create a new one in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_in_new_loop, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return _run_in_new_loop(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a new event loop"""
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)

# Global variable to store the current agent instance
_current_hotel_agent = None

def _get_hotel_agent():
    """Get the current hotel agent instance"""
    global _current_hotel_agent
    return _current_hotel_agent

@function_tool
async def search_hotels_tool(destination: str, check_in_date: str, check_out_date: str,
                           adults: int = 1, rooms: int = 1) -> dict:
    """
    Search for hotels using Amadeus API with enhanced profile filtering and caching
    
    Args:
        destination: Destination city or hotel location
        check_in_date: Check-in date in YYYY-MM-DD format
        check_out_date: Check-out date in YYYY-MM-DD format
        adults: Number of adults (default 1)
        rooms: Number of rooms (default 1)
    
    Returns:
        Dict with search results and metadata
    """
    
    try:
        print(f"🏨 ENHANCED Hotel search started: {destination}")
        start_time = datetime.now()
        
        hotel_agent = _get_hotel_agent()
        if not hotel_agent:
            return {"success": False, "error": "Hotel agent not available"}
        
        # OPTIMIZATION 1: Check cache first
        search_key = f"{destination}_{check_in_date}_{check_out_date}_{adults}_{rooms}"
        if hasattr(hotel_agent, '_search_cache') and search_key in hotel_agent._search_cache:
            print(f"⚡ Cache hit! Returning cached results for {search_key}")
            cached_result = hotel_agent._search_cache[search_key]
            
            # Update agent's latest search data with cached results
            hotel_agent.latest_search_results = cached_result.get("results", [])
            hotel_agent.latest_search_id = cached_result.get("search_id")
            hotel_agent.latest_filtering_info = cached_result.get("filtering_info", {})
            
            return cached_result
        
        # Convert destination to city code
        city_code = _convert_to_city_code(destination)
        print(f"📍 Using city code: {city_code} for {destination}")
        
        # Search hotels using Amadeus
        search_params = {
            "destination": city_code,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "rooms": rooms
        }
        
        hotels = run_async(hotel_agent.amadeus_service.search_hotels(**search_params))
        
        if not hotels:
            return {
                "success": False,
                "message": f"No hotels found for {destination} from {check_in_date} to {check_out_date}."
            }
        
        print(f"🔍 Found {len(hotels)} hotels from Amadeus API")
        
        # Create HotelSearch record
        search_id = str(uuid.uuid4())
        hotel_search = HotelSearch(
            search_id=search_id,
            user_id=hotel_agent.current_user_id,
            session_id=hotel_agent.current_session_id,
            destination=city_code,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            adults=adults,
            rooms=rooms,
            search_results=hotels
        )
        
        # Save search to database (async)
        run_async(hotel_agent.db.save_hotel_search(hotel_search))
        
        # Apply profile filtering to hotels
        original_count = len(hotels)
        filtered_hotels, filtering_applied, rationale = run_async(
            hotel_agent.profile_agent.filter_hotels_for_group(
                hotels, 
                hotel_agent.user_profile, 
                max_results=10
            )
        )
        
        filtered_count = len(filtered_hotels)
        
        print(f"🎯 Profile filtering: {original_count} → {filtered_count} hotels")
        print(f"📊 Filtering applied: {filtering_applied}")
        if rationale:
            print(f"💡 Rationale: {rationale}")
        
        # Store results in agent for response
        hotel_agent.latest_search_results = filtered_hotels
        hotel_agent.latest_search_id = search_id
        hotel_agent.latest_filtering_info = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filtering_applied": filtering_applied,
            "group_size": 1,  # Default for now
            "rationale": rationale
        }
        
        # Format results for presentation
        formatted_results = hotel_agent._format_hotel_results_for_agent(
            filtered_hotels, 
            destination,
            check_in_date,
            check_out_date,
            {
                "filtering_applied": filtering_applied,
                "original_count": original_count,
                "filtered_count": filtered_count,
                "group_size": 1,
                "rationale": rationale
            }
        )
        
        result_payload = {
            "success": True,
            "results": filtered_hotels,
            "formatted_results": formatted_results,
            "search_id": hotel_search.search_id,
            "search_params": search_params,
            "filtering_info": {
                "original_count": original_count,
                "filtered_count": filtered_count,
                "filtering_applied": filtering_applied,
                "group_size": 1,
                "rationale": rationale
            }
        }
        
        # OPTIMIZATION 2: Cache the result
        if not hasattr(hotel_agent, '_search_cache'):
            hotel_agent._search_cache = {}
        hotel_agent._search_cache[search_key] = result_payload
        
        # Keep cache manageable (max 10 entries)
        if len(hotel_agent._search_cache) > 10:
            oldest_key = list(hotel_agent._search_cache.keys())[0]
            del hotel_agent._search_cache[oldest_key]
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ ENHANCED Hotel search completed in {elapsed_time:.2f}s")
        
        return result_payload
        
    except Exception as e:
        print(f"❌ Error in search_hotels_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for hotels: {str(e)}"
        }

# ==================== HOTEL AGENT CLASS ====================

class HotelAgent:
    """
    Hotel agent using OpenAI Agents SDK with Amadeus API tools and profile filtering - ENHANCED VERSION
    """
    
    def __init__(self):
        global _current_hotel_agent
        
        self.openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # Store context for tool calls
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # Store latest search results for response
        self.latest_search_results = []
        self.latest_search_id = None
        self.latest_filtering_info = {}
        
        # Initialize search cache
        self._search_cache = {}
        
        # Set global instance
        _current_hotel_agent = self
        
        # ENHANCED: Create agent with tools - FIXED: Added required 'name' parameter
        self.agent = Agent(
            name="hotel_search_agent",  # FIXED: Added required name parameter
            model="o4-mini",
            instructions=get_hotel_system_message(),
            tools=[search_hotels_tool]
        )
        
        print("✅ HotelAgent initialized with enhanced tools and caching")
    
    async def _build_context_with_profile(self, user_profile: Dict[str, Any], 
                                        recent_conversation: List, max_messages: int = 50) -> List:
        """
        Build context that always includes user profile data plus recent conversation
        """
        
        # 1. Create profile context as system message
        profile_context = {
            "role": "system", 
            "content": f"""
            USER PROFILE CONTEXT:
            Name: {user_profile.get('name', 'Not provided')}
            Email: {user_profile.get('email', 'Not provided')}
            Gender: {user_profile.get('gender', 'Not provided')}
            Birthdate: {user_profile.get('birthdate', 'Not provided')}
            Location: {user_profile.get('city', 'Not provided')}
            Employer: {user_profile.get('employer', 'Not provided')}
            Working Schedule: {user_profile.get('working_schedule', 'Not provided')}
            Holiday Frequency: {user_profile.get('holiday_frequency', 'Not provided')}
            Annual Income: {user_profile.get('annual_income', 'Not provided')}
            Monthly Spending: {user_profile.get('monthly_spending', 'Not provided')}
            Holiday Preferences: {user_profile.get('holiday_preferences', [])}
            Travel Style: {user_profile.get('travel_style', 'Not specified')}
            Group Code: {user_profile.get('group_code', 'None')}
            
            Use this profile information to personalize hotel recommendations.
            Consider the user's travel style, budget (annual income and monthly spending), and preferences when suggesting accommodations.
            Take into account their holiday frequency and preferred activities when recommending hotel amenities.
            Always address the user by their name when possible.
            """
        }
        
        # 2. Get recent conversation (respecting limits)
        conversation_context = []
        for msg in recent_conversation[-max_messages:]:
            conversation_context.append({
                "role": msg.message_type,
                "content": msg.content
            })
        
        # 3. Combine: profile first, then conversation
        return [profile_context] + conversation_context
    
    # ==================== PROCESS MESSAGE ====================
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                             user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process hotel search requests - ENHANCED VERSION with support for larger datasets
        """
        
        try:
            print(f"🏨 ENHANCED HotelAgent processing: '{message[:50]}...'")
            start_time = datetime.now()
            
            # Clear previous search results
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            # Store context for tool calls
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.user_profile = user_profile
            
            # Update the global instance
            global _current_hotel_agent
            _current_hotel_agent.current_user_id = user_id
            _current_hotel_agent.current_session_id = session_id
            _current_hotel_agent.user_profile = user_profile
            _current_hotel_agent.latest_search_results = []
            _current_hotel_agent.latest_search_id = None
            _current_hotel_agent.latest_filtering_info = {}
            
            print(f"🔧 Context set: user_id={user_id}")
            
            # Build context with profile + recent conversation
            full_context = await self._build_context_with_profile(user_profile, conversation_history)
            
            print(f"🔧 Processing with profile context + {len(conversation_history[-50:])} previous messages")
            
            # ENHANCED: Direct agent processing with improved efficiency
            if not conversation_history:
                # First message in conversation - profile context + new message
                print("🚀 Starting new hotel conversation with profile context")
                result = await self._run_agent_with_retry(self.agent, full_context + [{"role": "user", "content": message}])
            else:
                # Continue conversation with profile + context + new message
                print("🔄 Continuing hotel conversation with profile + context")
                result = await self._run_agent_with_retry(self.agent, full_context + [{"role": "user", "content": message}])
            
            # Extract response
            assistant_message = result.get("final_output") or "I'm sorry, but I encountered an unexpected error while searching for hotels. Would you like me to try again?"
            
            # Get search results from global instance
            global_agent = _get_hotel_agent()
            search_results = global_agent.latest_search_results if global_agent else []
            search_id = global_agent.latest_search_id if global_agent else None
            filtering_info = global_agent.latest_filtering_info if global_agent else {}
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ ENHANCED HotelAgent completed in {elapsed_time:.2f}s")
            print(f"📊 Retrieved search data: {len(search_results)} results, search_id: {search_id}")
            
            return {
                "message": assistant_message,
                "agent_type": "hotel",
                "requires_action": False,
                "search_results": search_results,
                "search_id": search_id,
                "filtering_info": filtering_info,
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search",
                    "processing_time": elapsed_time
                }
            }
        
        except Exception as e:
            print(f"❌ Error in hotel agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error while searching for hotels. Please try again with different dates or destinations.",
                "agent_type": "hotel",
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {},
                "metadata": {
                    "agent_type": "hotel",
                    "conversation_type": "hotel_search",
                    "processing_time": 0,
                    "error": str(e)
                }
            }
    
    # ==================== AGENT EXECUTION WITH RETRY ====================
    
    @openai_api_retry
    async def _run_agent_with_retry(self, agent: Agent, messages) -> Dict[str, Any]:
        """Run agent with retry logic and proper error handling"""
        try:
            print("🔄 Running hotel agent with retry logic...")
            
            # Handle different message formats
            if isinstance(messages, str):
                # Single message - use the same pattern as flight agent
                result = await Runner.run(agent, messages)
            elif isinstance(messages, list):
                # List of messages - use the same pattern as flight agent
                result = await Runner.run(agent, messages)
            else:
                raise ValueError(f"Invalid messages format: {type(messages)}")
            
            # Extract clean response - handle RunResult object
            assistant_message = ""
            if hasattr(result, 'final_output'):
                assistant_message = result.final_output
            elif hasattr(result, 'messages') and result.messages:
                # Get the last message from the agent
                last_message = result.messages[-1]
                if hasattr(last_message, 'content'):
                    if isinstance(last_message.content, list) and len(last_message.content) > 0:
                        assistant_message = last_message.content[0].text if hasattr(last_message.content[0], 'text') else str(last_message.content[0])
                    else:
                        assistant_message = str(last_message.content)
                else:
                    assistant_message = str(last_message)
            else:
                # Fallback: convert result to string and clean it
                result_str = str(result)
                # Remove the RunResult debug information
                if "Final output (str):" in result_str:
                    parts = result_str.split("Final output (str):")
                    if len(parts) > 1:
                        # Get everything after "Final output (str):" and clean it
                        assistant_message = parts[1].strip()
                        # Remove any trailing debug info
                        if "\n- " in assistant_message:
                            assistant_message = assistant_message.split("\n- ")[0].strip()
                    else:
                        assistant_message = result_str
                else:
                    assistant_message = result_str
            
            # Clean up any remaining whitespace
            assistant_message = assistant_message.strip()
            
            return {
                "final_output": assistant_message,
                "messages": []  # Runner.run doesn't return messages like the old pattern
            }
            
        except Exception as e:
            print(f"❌ Hotel agent execution failed: {e}")
            # Return error response that will trigger retry
            return {
                "final_output": None,
                "error": str(e),
                "success": False
            }
    
    # ==================== HELPER METHODS ====================
    
    def _format_hotel_results_for_agent(self, hotels: List[Dict], destination: str, 
                                       check_in: str, check_out: str, filtering_info: Dict) -> str:
        """Format hotel results for the agent to present to the user"""
        
        if not hotels:
            return "No hotels found matching your criteria."
        
        result = f"Found {len(hotels)} hotels in {destination} from {check_in} to {check_out}:\n\n"
        
        if filtering_info.get("filtering_applied"):
            result += f"✨ Applied personalized filtering: {filtering_info.get('rationale', 'Based on your preferences')}\n"
            result += f"📊 Showing {filtering_info.get('filtered_count', len(hotels))} of {filtering_info.get('original_count', len(hotels))} available hotels\n\n"
        
        for i, hotel in enumerate(hotels, 1):
            # Extract hotel details
            hotel_data = hotel.get('hotel', {})
            offers = hotel.get('offers', [])
            
            name = hotel_data.get('name', 'Unknown Hotel')
            rating = hotel_data.get('rating', 'No rating')
            
            # Get best offer (lowest price)
            best_price = "Price not available"
            if offers:
                prices = []
                for offer in offers:
                    price_info = offer.get('price', {})
                    if price_info.get('total'):
                        try:
                            prices.append(float(price_info['total']))
                        except (ValueError, TypeError):
                            continue
                
                if prices:
                    min_price = min(prices)
                    currency = offers[0].get('price', {}).get('currency', '')
                    best_price = f"{min_price:.2f} {currency}"
            
            # Get amenities
            amenities = hotel_data.get('amenities', [])
            amenity_list = [amenity.get('name', '') for amenity in amenities[:3]]  # Show first 3
            amenities_text = ", ".join(amenity_list) if amenity_list else "Amenities info not available"
            
            # Get address
            address = hotel_data.get('address', {})
            location = address.get('cityName', '') or destination
            
            result += f"Hotel {i}: {name}\n"
            result += f"  💰 Price: {best_price}\n"
            result += f"  ⭐ Rating: {rating}\n"
            result += f"  📍 Location: {location}\n"
            result += f"  🏨 Amenities: {amenities_text}\n\n"
        
        return result

# ==================== MODULE EXPORTS ====================

__all__ = [
    'HotelAgent',
    'search_hotels_tool'
]