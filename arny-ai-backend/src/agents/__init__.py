"""
AI Agents module for Arny AI

This module contains all AI agents used in the Arny AI travel assistant application.
Each agent has a specific responsibility and uses OpenAI's APIs for natural language
processing and conversation management.

Main Agents:
- OnboardingAgent: Handles user onboarding and profile setup
- SupervisorAgent: Routes requests and handles general conversation
- FlightAgent: Manages flight searches using Amadeus API
- HotelAgent: Manages hotel searches using Amadeus API
- UserProfileAgent: Filters search results based on group preferences

Agent Architecture:
The agents follow a hierarchical structure where the SupervisorAgent acts as a 
coordinator that routes specific requests to specialized sub-agents. Each agent
is stateless and relies on database operations for persistence.

Usage:
    from src.agents import OnboardingAgent, SupervisorAgent, FlightAgent, HotelAgent, UserProfileAgent
    
    # Create agents
    onboarding = OnboardingAgent()
    supervisor = SupervisorAgent()
    flight = FlightAgent()
    hotel = HotelAgent()
    profile = UserProfileAgent()
    
    # Process messages
    response = await supervisor.process_message(user_id, message, session_id, profile, history)
"""

# Import all agent classes
from .onboarding_agent import OnboardingAgent
from .supervisor_agent import SupervisorAgent
from .flight_agent import FlightAgent
from .hotel_agent import HotelAgent
from .user_profile_agent import UserProfileAgent

# Export main classes and functions
__all__ = [
    # Agent Classes
    'OnboardingAgent',
    'SupervisorAgent', 
    'FlightAgent',
    'HotelAgent',
    'UserProfileAgent',
    
    # Agent Management
    'AgentManager',
    'AgentRegistry',
    
    # Exceptions
    'AgentError',
    'OnboardingAgentError',
    'FlightAgentError',
    'HotelAgentError',
    'SupervisorAgentError',
    'UserProfileAgentError',
    'AgentTimeoutError',
    'AgentConfigurationError',
    
    # Utility Functions
    'create_agent_manager',
    'get_available_agents',
    'validate_agent_response',
    'format_agent_error'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Arny AI Team'
__description__ = 'AI agents for travel planning and conversation management'

# ==================== AGENT-SPECIFIC EXCEPTIONS ====================

class AgentError(Exception):
    """Base exception for all agent-related errors"""
    pass

class OnboardingAgentError(AgentError):
    """Raised when onboarding agent encounters an error"""
    pass

class FlightAgentError(AgentError):
    """Raised when flight agent encounters an error"""
    pass

class HotelAgentError(AgentError):
    """Raised when hotel agent encounters an error"""
    pass

class SupervisorAgentError(AgentError):
    """Raised when supervisor agent encounters an error"""
    pass

class UserProfileAgentError(AgentError):
    """Raised when user profile agent encounters an error"""
    pass

class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out"""
    pass

class AgentConfigurationError(AgentError):
    """Raised when an agent is not properly configured"""
    pass

class AgentResponseError(AgentError):
    """Raised when an agent returns an invalid response"""
    pass

# ==================== AGENT REGISTRY ====================

class AgentRegistry:
    """
    Registry for managing and tracking available agents
    """
    
    def __init__(self):
        self._agents = {}
        self._agent_capabilities = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default agents with their capabilities"""
        self._agent_capabilities = {
            'onboarding': {
                'class': OnboardingAgent,
                'description': 'Handles user onboarding and profile setup',
                'capabilities': ['profile_setup', 'group_management', 'email_scanning'],
                'required_config': ['openai_api_key', 'database_connection']
            },
            'supervisor': {
                'class': SupervisorAgent,
                'description': 'Routes requests and handles general conversation',
                'capabilities': ['routing', 'general_chat', 'agent_coordination'],
                'required_config': ['openai_api_key']
            },
            'flight': {
                'class': FlightAgent,
                'description': 'Manages flight searches and information',
                'capabilities': ['flight_search', 'amadeus_integration'],
                'required_config': ['openai_api_key', 'amadeus_api_key', 'database_connection']
            },
            'hotel': {
                'class': HotelAgent,
                'description': 'Manages hotel searches and information',
                'capabilities': ['hotel_search', 'hotel_offers', 'amadeus_integration'],
                'required_config': ['openai_api_key', 'amadeus_api_key', 'database_connection']
            },
            'user_profile': {
                'class': UserProfileAgent,
                'description': 'Filters search results based on group preferences',
                'capabilities': ['profile_filtering', 'group_analysis', 'preference_matching'],
                'required_config': ['openai_api_key', 'database_connection']
            }
        }
    
    def register_agent(self, name: str, agent_class, capabilities: list, description: str = ""):
        """
        Register a new agent type
        
        Args:
            name: Agent name/identifier
            agent_class: Agent class
            capabilities: List of agent capabilities
            description: Agent description
        """
        self._agent_capabilities[name] = {
            'class': agent_class,
            'description': description,
            'capabilities': capabilities,
            'required_config': []
        }
    
    def get_agent_info(self, name: str) -> dict:
        """Get information about a specific agent"""
        return self._agent_capabilities.get(name, {})
    
    def get_all_agents(self) -> dict:
        """Get information about all registered agents"""
        return self._agent_capabilities.copy()
    
    def create_agent(self, name: str):
        """Create an instance of the specified agent"""
        if name not in self._agent_capabilities:
            raise AgentConfigurationError(f"Unknown agent type: {name}")
        
        agent_class = self._agent_capabilities[name]['class']
        try:
            return agent_class()
        except Exception as e:
            raise AgentConfigurationError(f"Failed to create {name} agent: {e}")

# Global agent registry instance
agent_registry = AgentRegistry()

# ==================== AGENT MANAGER ====================

class AgentManager:
    """
    Manager class for coordinating multiple agents
    
    This class provides a unified interface for managing and coordinating
    different types of agents in the Arny AI system.
    """
    
    def __init__(self):
        self.registry = agent_registry
        self._agent_instances = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all default agents"""
        try:
            self._agent_instances = {
                'onboarding': OnboardingAgent(),
                'supervisor': SupervisorAgent(), 
                'flight': FlightAgent(),
                'hotel': HotelAgent(),
                'user_profile': UserProfileAgent()
            }
        except Exception as e:
            raise AgentConfigurationError(f"Failed to initialize agents: {e}")
    
    def get_agent(self, agent_type: str):
        """
        Get an agent instance by type
        
        Args:
            agent_type: Type of agent ('onboarding', 'supervisor', 'flight', 'hotel', 'user_profile')
            
        Returns:
            Agent instance
            
        Raises:
            AgentConfigurationError: If agent type is unknown
        """
        if agent_type not in self._agent_instances:
            raise AgentConfigurationError(f"Unknown agent type: {agent_type}")
        
        return self._agent_instances[agent_type]
    
    def route_message(self, agent_type: str, *args, **kwargs):
        """
        Route a message to the appropriate agent
        
        Args:
            agent_type: Type of agent to route to
            *args: Positional arguments for the agent
            **kwargs: Keyword arguments for the agent
            
        Returns:
            Agent response
        """
        agent = self.get_agent(agent_type)
        
        try:
            return agent.process_message(*args, **kwargs)
        except Exception as e:
            raise AgentError(f"Error processing message with {agent_type} agent: {e}")
    
    def get_agent_capabilities(self, agent_type: str) -> list:
        """Get capabilities of a specific agent type"""
        info = self.registry.get_agent_info(agent_type)
        return info.get('capabilities', [])
    
    def health_check(self) -> dict:
        """
        Perform health check on all agents
        
        Returns:
            Dictionary with health status of each agent
        """
        health_status = {}
        
        for agent_type, agent in self._agent_instances.items():
            try:
                # Basic check - try to access the agent
                agent_class = agent.__class__.__name__
                health_status[agent_type] = {
                    'status': 'healthy',
                    'class': agent_class,
                    'capabilities': self.get_agent_capabilities(agent_type)
                }
            except Exception as e:
                health_status[agent_type] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_status
    
    def reload_agent(self, agent_type: str):
        """
        Reload a specific agent
        
        Args:
            agent_type: Type of agent to reload
        """
        try:
            self._agent_instances[agent_type] = self.registry.create_agent(agent_type)
        except Exception as e:
            raise AgentConfigurationError(f"Failed to reload {agent_type} agent: {e}")

# ==================== UTILITY FUNCTIONS ====================

async def create_agent_manager() -> AgentManager:
    """
    Factory function to create a new AgentManager instance
    
    Returns:
        AgentManager: Configured agent manager instance
        
    Raises:
        AgentConfigurationError: If agent manager creation fails
    """
    try:
        return AgentManager()
    except Exception as e:
        raise AgentConfigurationError(f"Failed to create agent manager: {e}")

def get_available_agents() -> dict:
    """
    Get information about all available agents
    
    Returns:
        Dictionary with agent information
    """
    return agent_registry.get_all_agents()

def validate_agent_response(response: dict) -> bool:
    """
    Validate that an agent response has the required structure
    
    Args:
        response: Agent response dictionary
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check for required fields
    required_fields = ['message']
    optional_fields = ['agent_type', 'requires_action', 'metadata', 'error']
    
    # Must have at least the message field
    if 'message' not in response:
        return False
    
    # All fields should be expected ones
    for field in response.keys():
        if field not in required_fields + optional_fields:
            return False
    
    return True

def format_agent_error(agent_type: str, error: Exception) -> dict:
    """
    Format an agent error into a standard response
    
    Args:
        agent_type: Type of agent that encountered the error
        error: The exception that occurred
        
    Returns:
        Formatted error response
    """
    return {
        'message': f"I encountered an error while processing your request. Please try again.",
        'agent_type': agent_type,
        'requires_action': False,
        'error': str(error),
        'error_type': type(error).__name__
    }

def get_agent_by_capability(capability: str) -> list:
    """
    Get all agents that have a specific capability
    
    Args:
        capability: Capability to search for
        
    Returns:
        List of agent names that have the capability
    """
    agents_with_capability = []
    
    for agent_name, agent_info in agent_registry.get_all_agents().items():
        if capability in agent_info.get('capabilities', []):
            agents_with_capability.append(agent_name)
    
    return agents_with_capability

def create_agent_response(message: str, agent_type: str, **kwargs) -> dict:
    """
    Create a standardized agent response
    
    Args:
        message: Response message
        agent_type: Type of agent generating the response
        **kwargs: Additional response fields
        
    Returns:
        Standardized response dictionary
    """
    response = {
        'message': message,
        'agent_type': agent_type,
        'requires_action': kwargs.get('requires_action', False)
    }
    
    # Add optional fields if provided
    optional_fields = ['metadata', 'search_results', 'search_id', 'onboarding_complete', 
                      'next_step', 'collected_data', 'recommendations', 'booking_options',
                      'filtering_info', 'group_preferences']
    
    for field in optional_fields:
        if field in kwargs:
            response[field] = kwargs[field]
    
    return response

# ==================== AGENT ROUTING HELPERS ====================

def determine_agent_for_message(message: str, context: dict = None) -> str:
    """
    Determine which agent should handle a message based on content and context
    
    Args:
        message: User message
        context: Additional context (user state, etc.)
        
    Returns:
        Agent type that should handle the message
    """
    message_lower = message.lower()
    
    # Check for onboarding context
    if context and not context.get('onboarding_completed', True):
        return 'onboarding'
    
    # Check for specific search requests
    flight_keywords = ['flight', 'fly', 'plane', 'airline', 'airport', 'departure', 'arrival']
    hotel_keywords = ['hotel', 'accommodation', 'stay', 'room', 'check-in', 'check-out']
    
    if any(keyword in message_lower for keyword in flight_keywords):
        return 'flight'
    elif any(keyword in message_lower for keyword in hotel_keywords):
        return 'hotel'
    else:
        return 'supervisor'

def validate_agent_context(agent_type: str, context: dict) -> bool:
    """
    Validate that the provided context is sufficient for the agent
    
    Args:
        agent_type: Type of agent
        context: Context dictionary
        
    Returns:
        True if context is valid, False otherwise
    """
    required_context = {
        'onboarding': ['user_id'],
        'supervisor': ['user_id', 'user_profile'],
        'flight': ['user_id', 'user_profile'],
        'hotel': ['user_id', 'user_profile'],
        'user_profile': ['user_id']
    }
    
    required_fields = required_context.get(agent_type, [])
    
    for field in required_fields:
        if field not in context or context[field] is None:
            return False
    
    return True

# ==================== MODULE INITIALIZATION ====================

def _initialize_module():
    """Initialize the agents module"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Agents module initialized")
    
    # Validate agent configurations
    try:
        manager = AgentManager()
        health = manager.health_check()
        healthy_agents = [name for name, status in health.items() if status.get('status') == 'healthy']
        logger.info(f"Initialized {len(healthy_agents)} healthy agents: {', '.join(healthy_agents)}")
    except Exception as e:
        logger.error(f"Failed to initialize agent manager: {e}")

# Call initialization
_initialize_module()

# ==================== CONSTANTS ====================

# Agent types
AGENT_TYPES = {
    'ONBOARDING': 'onboarding',
    'SUPERVISOR': 'supervisor', 
    'FLIGHT': 'flight',
    'HOTEL': 'hotel',
    'USER_PROFILE': 'user_profile'
}

# Agent capabilities
CAPABILITIES = {
    'PROFILE_SETUP': 'profile_setup',
    'GROUP_MANAGEMENT': 'group_management',
    'EMAIL_SCANNING': 'email_scanning',
    'ROUTING': 'routing',
    'GENERAL_CHAT': 'general_chat',
    'FLIGHT_SEARCH': 'flight_search',
    'HOTEL_SEARCH': 'hotel_search',
    'AMADEUS_INTEGRATION': 'amadeus_integration',
    'PROFILE_FILTERING': 'profile_filtering',
    'GROUP_ANALYSIS': 'group_analysis',
    'PREFERENCE_MATCHING': 'preference_matching'
}

# Add constants to exports
__all__.extend([
    'AGENT_TYPES',
    'CAPABILITIES',
    'agent_registry',
    'determine_agent_for_message',
    'validate_agent_context',
    'create_agent_response',
    'get_agent_by_capability'
])
