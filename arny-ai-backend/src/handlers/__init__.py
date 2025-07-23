"""
Handlers module for Arny AI Backend

This module provides AWS Lambda request handlers for different application flows:
- MainHandler: Handles main travel conversation flow (flight/hotel search, general chat)
- OnboardingHandler: Handles user onboarding flow (profile setup, group management)

Usage:
    from handlers import MainHandler, OnboardingHandler
    
    main_handler = MainHandler()
    onboarding_handler = OnboardingHandler()
    
    # Process requests
    response = await main_handler.handle_request(event, context)
    onboarding_response = await onboarding_handler.handle_request(event, context)
"""

from .main_handler import MainHandler
from .onboarding_handler import OnboardingHandler

# Export main handler classes
__all__ = [
    'MainHandler',
    'OnboardingHandler',
    'create_main_handler',
    'create_onboarding_handler',
    'get_handler',
    'route_request'
]

# Version information
__version__ = '1.0.0'

# Handler factory functions for easy instantiation
def create_main_handler():
    """
    Factory function to create a MainHandler instance
    
    Returns:
        MainHandler: Configured main handler instance for travel conversations
    """
    return MainHandler()

def create_onboarding_handler():
    """
    Factory function to create an OnboardingHandler instance
    
    Returns:
        OnboardingHandler: Configured onboarding handler instance
    """
    return OnboardingHandler()

# Handler registry for dependency injection and routing
HANDLER_REGISTRY = {
    'main': MainHandler,
    'onboarding': OnboardingHandler,
    'travel': MainHandler,  # Alias for main handler
    'chat': MainHandler,    # Alias for main handler
    'auth': MainHandler,    # Auth requests go to main handler
    'user': MainHandler,    # User status requests go to main handler
}

def get_handler(handler_name: str):
    """
    Get a handler instance by name
    
    Args:
        handler_name: Name of the handler ('main', 'onboarding', 'travel', 'chat', 'auth', 'user')
        
    Returns:
        Handler instance
        
    Raises:
        ValueError: If handler name is not recognized
    """
    if handler_name not in HANDLER_REGISTRY:
        raise ValueError(f"Unknown handler: {handler_name}. Available handlers: {list(HANDLER_REGISTRY.keys())}")
    
    handler_class = HANDLER_REGISTRY[handler_name]
    return handler_class()

def route_request(path: str):
    """
    Route request to appropriate handler based on path
    
    Args:
        path: Request path (e.g., '/onboarding/chat', '/travel/chat', '/auth/signin')
        
    Returns:
        tuple: (handler_instance, handler_type)
        
    Examples:
        handler, handler_type = route_request('/onboarding/chat')
        # Returns: (OnboardingHandler(), 'onboarding')
        
        handler, handler_type = route_request('/travel/chat')
        # Returns: (MainHandler(), 'main')
    """
    path = path.strip('/')
    
    # Route based on path prefix
    if path.startswith('onboarding'):
        return create_onboarding_handler(), 'onboarding'
    elif path.startswith(('chat', 'travel')):
        return create_main_handler(), 'main'
    elif path.startswith(('auth', 'user')):
        return create_main_handler(), 'main'
    else:
        # Default to main handler for unknown paths
        return create_main_handler(), 'main'

async def handle_lambda_request(event, context):
    """
    Universal Lambda request handler that routes to appropriate handler
    
    Args:
        event: AWS Lambda event
        context: AWS Lambda context
        
    Returns:
        Response from appropriate handler
        
    Usage:
        # In your Lambda function
        from handlers import handle_lambda_request
        
        def lambda_handler(event, context):
            return asyncio.run(handle_lambda_request(event, context))
    """
    try:
        # Extract path from event
        path = event.get('path', '')
        
        # Route to appropriate handler
        handler, handler_type = route_request(path)
        
        # Handle the request
        if handler_type == 'onboarding':
            if path.endswith('/group/check'):
                return await handler.handle_group_code_check(event, context)
            elif path.endswith('/group/create'):
                return await handler.handle_create_group(event, context)
            elif path.endswith('/group/join'):
                return await handler.handle_join_group(event, context)
            else:
                return await handler.handle_request(event, context)
        else:  # main handler
            if path.startswith('/auth'):
                return await handler.handle_auth_request(event, context)
            elif path.startswith('/user'):
                return await handler.handle_user_status(event, context)
            else:
                return await handler.handle_request(event, context)
                
    except Exception as e:
        # Return error response
        import json
        from datetime import datetime
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': f'Handler routing error: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            })
        }

# Handler type constants
class HandlerTypes:
    """Constants for handler types"""
    MAIN = 'main'
    ONBOARDING = 'onboarding'
    TRAVEL = 'travel'
    CHAT = 'chat'
    AUTH = 'auth'
    USER = 'user'

# Path routing patterns
ROUTE_PATTERNS = {
    'onboarding': ['onboarding'],
    'main': ['chat', 'travel', 'auth', 'user'],
}

def get_handler_type_for_path(path: str) -> str:
    """
    Get handler type for a given path
    
    Args:
        path: Request path
        
    Returns:
        Handler type string
    """
    path = path.strip('/')
    
    for handler_type, patterns in ROUTE_PATTERNS.items():
        if any(path.startswith(pattern) for pattern in patterns):
            return handler_type
    
    return 'main'  # Default

# Module-level constants
SUPPORTED_HANDLERS = list(HANDLER_REGISTRY.keys())
SUPPORTED_PATHS = [
    '/auth/signup', '/auth/signin', '/auth/refresh', '/auth/signout',
    '/onboarding/chat', '/onboarding/group/check', '/onboarding/group/create', '/onboarding/group/join',
    '/chat', '/travel/chat',
    '/user/status'
]

# Convenience imports for backward compatibility
main_handler = MainHandler
onboarding_handler = OnboardingHandler

# Handler configuration
DEFAULT_HANDLER_CONFIG = {
    'timeout': 30,
    'memory_size': 512,
    'environment': 'development'
}

def configure_handlers(**config):
    """
    Configure handler settings (for future extensibility)
    
    Args:
        **config: Configuration parameters
    """
    DEFAULT_HANDLER_CONFIG.update(config)

# Health check function for handlers
def health_check():
    """
    Perform health check on all handlers
    
    Returns:
        dict: Health status of all handlers
    """
    status = {
        'handlers': {},
        'overall': 'healthy'
    }
    
    try:
        # Test handler instantiation
        main = create_main_handler()
        onboarding = create_onboarding_handler()
        
        status['handlers']['main'] = 'healthy'
        status['handlers']['onboarding'] = 'healthy'
        
    except Exception as e:
        status['overall'] = 'unhealthy'
        status['error'] = str(e)
    
    return status