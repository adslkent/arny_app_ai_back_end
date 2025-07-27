"""
Supervisor Agent Module - ENHANCED VERSION with Unified Conversation Context

This module provides a supervisor agent that routes requests to specialized agents
and handles general conversation with unified context management.

Key Features:
1. Ultra-fast keyword-based routing (no LLM calls for routing decisions)
2. Direct routing to specialized agents (flight/hotel)
3. Unified conversation context approach (50 messages)
4. Enhanced general conversation handling with conversation history
5. NO TIMEOUT LIMITS for better reliability

Usage example:
```python
from supervisor_agent import SupervisorAgent

# Create and use the agent
agent = SupervisorAgent()
result = await agent.process_message(user_id, "Find flights", session_id, {}, [])
```
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

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
import requests

from ..utils.config import config
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent

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

class SupervisorAgent:
    """
    ENHANCED: Supervisor agent with fast routing, NO TIMEOUT LIMITS, and unified conversation context
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize the specialized agents
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
    
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
            Group Code: {user_profile.get('group_code', 'None')}
            
            Use this profile information to personalize travel recommendations and responses.
            Always refer to the user by their name when possible and tailor suggestions to their preferences.
            Consider their budget, holiday frequency, and preferred activities.
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
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Process user message with ultra-fast routing and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"🤖 SUPERVISOR: Processing message: '{message[:50]}...'")
            
            # OPTIMIZATION 1: Ultra-fast keyword-based routing (no LLM call)
            routing_decision = self._ultra_fast_routing(message)
            
            print(f"⚡ INSTANT routing decision: {routing_decision}")
            
            # OPTIMIZATION 2: Direct routing without delays
            if routing_decision == "flight_search":
                print("✈️ Routing to FlightAgent")
                return await self.flight_agent.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            elif routing_decision == "hotel_search":
                print("🏨 Routing to HotelAgent")
                return await self.hotel_agent.process_message(
                    user_id, message, session_id, user_profile, conversation_history
                )
            else:
                print("💬 Handling as general conversation")
                return await self._handle_general_conversation_no_timeout(
                    user_id, message, session_id, user_profile, conversation_history
                )
        
        except Exception as e:
            print(f"❌ Error in supervisor agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error processing your request. Please try again.",
                "agent_type": "supervisor",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    def _ultra_fast_routing(self, message: str) -> str:
        """
        OPTIMIZATION: Ultra-fast keyword-based routing without LLM calls
        
        Returns:
            - "flight_search": For flight-related requests
            - "hotel_search": For hotel-related requests  
            - "general": For general conversation
        """
        
        message_lower = message.lower()
        
        # FLIGHT KEYWORDS (comprehensive list)
        flight_keywords = [
            "flight", "flights", "fly", "flying", "plane", "airplane", "aircraft",
            "departure", "arrive", "arrival", "takeoff", "landing", "airline", "airways",
            "airport", "boarding", "check-in", "gate", "terminal", "runway",
            "ticket", "booking", "seat", "aisle", "window", "economy", "business", "first class",
            "layover", "stopover", "direct", "nonstop", "connecting", "transfer",
            "baggage", "luggage", "carry-on", "checked bag",
            "itinerary", "travel time", "flight time", "duration",
            "roundtrip", "round trip", "one way", "return",
            # Specific airlines
            "qantas", "jetstar", "virgin", "american airlines", "united", "delta",
            "british airways", "lufthansa", "air france", "singapore airlines", "emirates"
        ]
        
        # HOTEL KEYWORDS (comprehensive list)
        hotel_keywords = [
            "hotel", "hotels", "stay", "staying", "accommodation", "accommodations",
            "room", "rooms", "suite", "bed", "bedroom", "bathroom",
            "check-in", "check-out", "checkin", "checkout",
            "reservation", "booking", "book", "reserve",
            "night", "nights", "overnight", "sleep",
            "lodge", "inn", "resort", "motel", "hostel", "b&b", "bed and breakfast",
            "apartment", "villa", "house", "rental",
            "amenities", "pool", "gym", "spa", "restaurant", "breakfast",
            "wifi", "parking", "concierge", "housekeeping",
            "star", "rating", "luxury", "budget", "cheap", "expensive",
            # Hotel chains
            "hilton", "marriott", "hyatt", "sheraton", "holiday inn", "best western",
            "intercontinental", "radisson", "westin", "renaissance"
        ]
        
        # Count keyword matches
        flight_score = sum(1 for keyword in flight_keywords if keyword in message_lower)
        hotel_score = sum(1 for keyword in hotel_keywords if keyword in message_lower)
        
        print(f"🎯 Routing scores - Flight: {flight_score}, Hotel: {hotel_score}")
        
        # Routing logic with clear priority
        if flight_score > hotel_score and flight_score > 0:
            return "flight_search"
        elif hotel_score > flight_score and hotel_score > 0:
            return "hotel_search"
        elif flight_score == hotel_score and flight_score > 0:
            # Tie-breaker: Look for more specific patterns
            if any(word in message_lower for word in ["from", "to", "departure", "arrival", "fly"]):
                return "flight_search"
            elif any(word in message_lower for word in ["stay", "night", "check-in", "room"]):
                return "hotel_search"
        
        # Default to general conversation
        return "general"
    
    @openai_api_retry
    async def _handle_general_conversation_no_timeout(self, user_id: str, message: str, session_id: str,
                                                     user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Handle general conversation with conversation history and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"💬 Handling general conversation for: '{message[:50]}...'")
            
            # Build context with profile + recent conversation
            full_context = await self._build_context_with_profile(user_profile, conversation_history)
            
            # Add current user message
            full_context.append({"role": "user", "content": message})
            
            print(f"🔧 Built context with profile + {len(conversation_history[-50:])} conversation messages")
            
            # Call OpenAI with retry logic
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use efficient model for general conversation
                messages=full_context,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            print(f"✅ General conversation response generated: '{assistant_message[:50]}...'")
            
            return {
                "message": assistant_message,
                "agent_type": "supervisor",
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {},
                "metadata": {
                    "agent_type": "supervisor",
                    "conversation_type": "general",
                    "model_used": "gpt-4o-mini"
                }
            }
            
        except Exception as e:
            print(f"❌ Error in general conversation: {e}")
            return {
                "message": "I'm here to help you with travel planning. You can ask me about flights, hotels, or general travel questions!",
                "agent_type": "supervisor",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    def _get_general_conversation_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """
        Generate system prompt for general conversation based on user profile
        """
        
        user_name = user_profile.get('name', 'there')
        
        return f"""You are Arny, a friendly and professional AI travel assistant. You're chatting with {user_name}.

Your personality:
- Warm, helpful, and knowledgeable about travel
- Professional but conversational
- Proactive in offering travel-related assistance
- Knowledgeable about destinations, travel tips, and planning

Your capabilities:
- Flight search and recommendations
- Hotel search and recommendations  
- General travel advice and planning
- Destination information
- Travel tips and insights

Guidelines:
1. Be conversational and friendly
2. If the user asks about flights or hotels, let them know you can help search for those
3. Offer helpful travel tips and advice
4. Ask clarifying questions when needed
5. Keep responses concise but informative
6. Show enthusiasm for travel and helping with their plans

Remember: You can search for flights and hotels when users are ready, but for general conversation, just be helpful and engaging."""

# ==================== MODULE EXPORTS ====================

__all__ = [
    'SupervisorAgent'
]
