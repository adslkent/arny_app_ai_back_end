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
from agents import Agent, Runner, WebSearchTool
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
    
    @openai_api_retry
    async def _run_agent_with_retry(self, agent, input_data):
        """Run agent with retry logic applied"""
        return await Runner.run(agent, input_data)
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Process user message with ultra-fast routing and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"🤖 SUPERVISOR: Processing message: '{message[:50]}...'")
            
            # Context-aware routing using LLM with conversation history
            routing_decision = await self._context_aware_routing(message, conversation_history)
            
            print(f"🎯 Routing decision: {routing_decision}")
            
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
                print("💬 Handling as general conversation with web search capability")
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

    async def _context_aware_routing(self, message: str, conversation_history: list) -> str:
        """
        Context-aware routing using LLM with conversation history
        
        Args:
            message: Current user message
            conversation_history: List of recent conversation messages
        
        Returns:
            - "flight_search": For flight-related requests
            - "hotel_search": For hotel-related requests  
            - "general": For general conversation
        """
        
        try:
            # Build conversation context from history (last 50 messages)
            context_messages = []
            recent_messages = conversation_history[-50:]
            
            for msg in recent_messages:
                role = "user" if msg.message_type == "user" else "assistant"
                context_messages.append(f"{role}: {msg.content}")
            
            # Add current message
            context_messages.append(f"user: {message}")
            
            # Create full context string
            conversation_context = "\n".join(context_messages)
            
            routing_agent = Agent(
                name="Routing Assistant",
                instructions="""You are a routing assistant for a travel AI system. Your ONLY job is to analyze the conversation and determine the correct routing.

                Based on the conversation context and the most recent user message, respond with ONLY one of these three exact words:
                - "flight_search" if the user is asking about flights, airlines, airports, or air travel
                - "hotel_search" if the user is asking about hotels, accommodations, stays, or lodging  
                - "general" for all other conversations, questions, or greetings

                IMPORTANT CONTEXT RULES:
                1. The current message likely references the MOST RECENT previous messages, not older context
                2. If the user asks a follow-up question (e.g., "what about tomorrow?" or "how much does it cost?"), determine the routing based on what they were JUST discussing
                3. Pay special attention to the last 2-3 exchanges to understand what the user is referring to
                4. Common follow-up patterns:
                - "What about [different date]?" - refers to the most recent flight or hotel search
                - "Show me more options" - refers to the most recent flight or hotel search
                - "How much?" or "What's the price?" - refers to the most recent flight or hotel discussed
                - "Book it" or "I'll take it" - refers to the most recent flight or hotel option shown

                Example routing decisions:
                - User previously asked about flights to Paris, now asks "what about next week?" → route to "flight_search"
                - User previously asked about hotels in Tokyo, now asks "any cheaper options?" → route to "hotel_search"
                - User asks "how's the weather there?" after discussing flights → route to "general"

                Your response must be ONLY one of these three words, nothing else.""",
                model="o4-mini"
            )
            
            # Run the routing agent
            result = await self._run_agent_with_retry(routing_agent, conversation_context)
            
            # Extract routing decision
            routing_decision = result.final_output if hasattr(result, 'final_output') else str(result)
            routing_decision = routing_decision.strip().lower()
            
            # Validate the response
            if routing_decision in ["flight_search", "hotel_search", "general"]:
                print(f"🎯 Context-aware routing decision: {routing_decision}")
                return routing_decision
            else:
                print(f"⚠️ Invalid routing response: {routing_decision}, defaulting to general")
                return "general"
                
        except Exception as e:
            print(f"⚠️ Error in context-aware routing: {e}, defaulting to general")
            return "general"
    
    @openai_api_retry
    async def _handle_general_conversation_no_timeout(self, user_id: str, message: str, session_id: str,
                                                     user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        ENHANCED: Handle general conversation with conversation history and NO TIMEOUT LIMITS
        """
        
        try:
            print(f"💬 Handling general conversation with web search capability for: '{message[:50]}...'")

            # Create agent with WebSearchTool capability
            general_agent = Agent(
                name="Arny General Assistant",
                instructions=self._get_general_conversation_system_prompt(user_profile),
                model="o4-mini",
                tools=[WebSearchTool()]
            )

            # Build context with profile + recent conversation
            full_context = await self._build_context_with_profile(user_profile, conversation_history)

            # Convert context to string format for Agents SDK
            conversation_context = []
            for ctx in full_context:
                conversation_context.append(f"{ctx['role']}: {ctx['content']}")

            # Add current user message
            conversation_context.append(f"user: {message}")

            # Create input for the agent (use last 50 messages for context)
            agent_input = "\n".join(conversation_context[-50:])

            print(f"🔧 Built context with profile + {len(conversation_history[-50:])} conversation messages")

            # Run agent with retry logic using Agents SDK
            result = await self._run_agent_with_retry(general_agent, agent_input)

            # Extract the response
            assistant_message = result.final_output if hasattr(result, 'final_output') else str(result)

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
                    "conversation_type": "general_with_web_search",
                    "model_used": "o4-mini",
                    "web_search_available": True
                }
            }
        
        except Exception as e:
            print(f"❌ Error in general conversation: {e}")
            return {
                "message": "I'm here to help you with travel planning. You can ask me about flights, hotels, or general questions!",
                "agent_type": "supervisor",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    def _get_general_conversation_system_prompt(self, user_profile: Dict[str, Any]) -> str:
        """
        Generate enhanced system prompt for general conversation with web search capability
        """
        
        user_name = user_profile.get('name', 'there')
        
        return f"""You are Arny, a friendly and professional AI travel assistant. You're chatting with {user_name}.

Your personality:
- Warm, helpful, and knowledgeable about travel
- Professional but conversational
- Proactive in offering travel-related assistance
- Knowledgeable about destinations, travel tips, and planning
- Use WebSearchTool whenever you need current information to provide accurate answers

Your capabilities:
- Flight search and recommendations
- Hotel search and recommendations  
- General travel advice and planning
- Destination information
- Travel tips and insights
- For questions requiring current information (weather, news, travel advisories, current events, 
  recent policy changes, etc.), use WebSearchTool to get accurate, up-to-date information

AMBIGUOUS CITY HANDLING:
- If a user requests flights or hotels to a city without specifying the country, ask the user to provide the country name first before proceeding with any search
- Once the country is specified, proceed with the appropriate flight or hotel search  

CONVERSATION CONTEXT MEMORY:
- When users ask follow-up questions about flights or hotels without specifying a destination, always refer to their most recent flight or hotel search context
- Keep track of the most recent travel query (destination, dates, etc.) throughout the conversation and use it for context when users ask related follow-up questions

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
