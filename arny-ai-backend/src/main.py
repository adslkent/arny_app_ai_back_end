import json
import os
import asyncio
from typing import Dict, Any

# Import handlers
from .handlers.main_handler import MainHandler
from .handlers.onboarding_handler import OnboardingHandler
from .utils.config import config

# Initialize handlers
main_handler = MainHandler()
onboarding_handler = OnboardingHandler()

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main AWS Lambda handler - routes requests to appropriate handlers
    ENHANCED: Now supports both API Gateway and Function URL events
    
    Args:
        event: Lambda event containing request data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    
    # Run the async handler using asyncio.run()
    return asyncio.run(_async_lambda_handler(event, context))

async def _async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Internal async handler that contains all the original logic
    ENHANCED: Detects and handles both API Gateway and Function URL events
    
    Args:
        event: Lambda event containing request data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    
    try:
        # Validate configuration
        config.validate()
        
        # ENHANCED: Detect event type (API Gateway vs Function URL)
        is_function_url = _is_function_url_event(event)
        
        if is_function_url:
            print(f"🔗 Processing Function URL request")
            path, http_method = _extract_function_url_details(event)
        else:
            print(f"🔍 Processing API Gateway request")
            path = event.get('path', '')
            http_method = event.get('httpMethod', 'POST')
        
        print(f"🔍 Processing request: {http_method} {path}")
        
        # Handle CORS preflight requests for both event types
        if http_method == 'OPTIONS':
            return _cors_response(is_function_url)
        
        # Route based on path
        if path.startswith('/auth'):
            return await _handle_auth_routes(event, context, is_function_url)
        elif path.startswith('/onboarding'):
            return await _handle_onboarding_routes(event, context, is_function_url)
        elif path.startswith('/chat') or path.startswith('/travel') or path == '/':
            # Function URL ONLY for all travel chat (15-minute timeout)
            return await _handle_main_routes(event, context, is_function_url)
        elif path.startswith('/user'):
            return await _handle_user_routes(event, context, is_function_url)
        elif path == '/health':
            return await _handle_health_route(event, context, is_function_url)
        else:
            return _error_response(404, f"Route not found: {path}", is_function_url)
            
    except ValueError as e:
        # Configuration error
        print(f"❌ Configuration error: {str(e)}")
        return _error_response(500, f"Configuration error: {str(e)}", False)
    except Exception as e:
        print(f"❌ Unexpected error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return _error_response(500, "Internal server error", False)

def _is_function_url_event(event: Dict[str, Any]) -> bool:
    """
    Detect if this is a Function URL event vs API Gateway event
    
    Args:
        event: Lambda event
        
    Returns:
        True if Function URL event, False if API Gateway event
    """
    # Function URL events have different structure than API Gateway events
    # Key indicators:
    # - Function URL: has 'requestContext.domainName' without 'requestContext.stage'
    # - Function URL: has 'rawPath' instead of 'path'
    # - Function URL: has 'version' field as '2.0'
    
    request_context = event.get('requestContext', {})
    
    # Check for Function URL specific fields
    if event.get('version') == '2.0' and 'rawPath' in event:
        return True
    
    # Check for API Gateway specific fields
    if 'stage' in request_context and 'path' in event:
        return False
    
    # Additional check: Function URLs don't have stage in requestContext
    if 'domainName' in request_context and 'stage' not in request_context:
        return True
    
    # Default to API Gateway if uncertain
    return False

def _extract_function_url_details(event: Dict[str, Any]) -> tuple[str, str]:
    """
    Extract path and method from Function URL event
    
    Args:
        event: Function URL event
        
    Returns:
        Tuple of (path, http_method)
    """
    # Function URL events use different field names
    path = event.get('rawPath', '/')
    http_method = event.get('requestContext', {}).get('http', {}).get('method', 'POST')
    
    # For Function URL, we treat root path as /chat since this is our travel function
    if path == '/':
        path = '/chat'
    
    return path, http_method

async def _handle_auth_routes(event: Dict[str, Any], context: Any, is_function_url: bool = False) -> Dict[str, Any]:
    """Handle authentication routes"""
    
    if is_function_url:
        # Function URL is only for travel chat, not auth
        return _error_response(404, "Auth routes not available on Function URL", is_function_url)
    
    path = event.get('path', '')
    
    if path == '/auth/signup' or path == '/auth/signin' or path == '/auth/refresh' or path == '/auth/signout':
        return await main_handler.handle_auth_request(event, context)
    else:
        return _error_response(404, f"Auth route not found: {path}", is_function_url)

async def _handle_onboarding_routes(event: Dict[str, Any], context: Any, is_function_url: bool = False) -> Dict[str, Any]:
    """Handle onboarding routes"""
    
    if is_function_url:
        # Function URL is only for travel chat, not onboarding
        return _error_response(404, "Onboarding routes not available on Function URL", is_function_url)
    
    path = event.get('path', '')
    
    if path == '/onboarding/chat':
        return await onboarding_handler.handle_request(event, context)
    elif path == '/onboarding/group/check':
        return await onboarding_handler.handle_group_code_check(event, context)
    elif path == '/onboarding/group/create':
        return await onboarding_handler.handle_create_group(event, context)
    elif path == '/onboarding/group/join':
        return await onboarding_handler.handle_join_group(event, context)
    else:
        return _error_response(404, f"Onboarding route not found: {path}", is_function_url)

async def _handle_main_routes(event: Dict[str, Any], context: Any, is_function_url: bool = False) -> Dict[str, Any]:
    """
    Handle main travel conversation routes
    SIMPLIFIED: Function URL ONLY (no API Gateway for travel chat)
    """
    
    # Extract path based on event type
    if is_function_url:
        path = event.get('rawPath', '/')
        if path == '/':
            path = '/chat'  # Treat Function URL root as chat
        print(f"🚀 Function URL travel request: {path}")
        print(f"⏱️ Using Function URL with extended timeout (up to 15 minutes)")
    else:
        # This shouldn't happen anymore since we removed API Gateway events for travel
        print(f"⚠️ API Gateway request received for travel route - this is unexpected")
        return _error_response(404, "Travel chat only available via Function URL", is_function_url)
    
    return await main_handler.handle_request(event, context)

async def _handle_user_routes(event: Dict[str, Any], context: Any, is_function_url: bool = False) -> Dict[str, Any]:
    """Handle user-related routes"""
    
    if is_function_url:
        # Function URL is only for travel chat, not user routes
        return _error_response(404, "User routes not available on Function URL", is_function_url)
    
    path = event.get('path', '')
    
    if path == '/user/status':
        return await main_handler.handle_user_status(event, context)
    else:
        return _error_response(404, f"User route not found: {path}", is_function_url)

async def _handle_health_route(event: Dict[str, Any], context: Any, is_function_url: bool = False) -> Dict[str, Any]:
    """Handle health check route"""
    
    try:
        # Basic health check
        from .database.operations import DatabaseOperations
        db = DatabaseOperations()
        health_status = await db.health_check()
        
        health_data = {
            'status': 'healthy',
            'service': 'arny-ai-backend',
            'database': health_status,
            'version': '1.0.0',
            'event_type': 'function_url' if is_function_url else 'api_gateway'
        }
        
        return _success_response(health_data, is_function_url)
        
    except Exception as e:
        return _error_response(500, f"Health check failed: {str(e)}", is_function_url)

def _cors_response(is_function_url: bool = False) -> Dict[str, Any]:
    """
    Return CORS preflight response
    ENHANCED: Different handling for Function URL vs API Gateway
    """
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET,PUT,DELETE'
    }
    
    if is_function_url:
        # Function URL response format
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    else:
        # API Gateway response format
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }

def _success_response(data: Dict[str, Any], is_function_url: bool = False) -> Dict[str, Any]:
    """
    Return successful response
    ENHANCED: Handles both API Gateway and Function URL formats
    """
    
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }
    
    body_data = {
        'success': True,
        'data': data,
        'timestamp': asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
    }
    
    response = {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps(body_data, default=str)
    }
    
    # Function URL and API Gateway use same response format
    return response

def _error_response(status_code: int, error_message: str, is_function_url: bool = False) -> Dict[str, Any]:
    """
    Return error response
    ENHANCED: Handles both API Gateway and Function URL formats
    """
    
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }
    
    body_data = {
        'success': False,
        'error': error_message,
        'timestamp': asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else None
    }
    
    response = {
        'statusCode': status_code,
        'headers': headers,
        'body': json.dumps(body_data, default=str)
    }
    
    # Function URL and API Gateway use same response format
    return response

# For local testing
if __name__ == "__main__":
    import asyncio
    
    # Example test event for API Gateway
    api_gateway_event = {
        'path': '/health',
        'httpMethod': 'GET',
        'body': None,
        'requestContext': {
            'stage': 'dev'
        }
    }
    
    # Example test event for Function URL
    function_url_event = {
        'version': '2.0',
        'rawPath': '/',
        'requestContext': {
            'domainName': 'lambda-url.us-east-1.on.aws',
            'http': {
                'method': 'POST'
            }
        },
        'body': json.dumps({
            'user_id': 'test-user',
            'message': 'Hello',
            'access_token': 'test-token'
        })
    }
    
    class MockContext:
        def __init__(self):
            self.function_name = "arny-ai-backend"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:arny-ai-backend"
            self.memory_limit_in_mb = "1536"
            self.remaining_time_in_millis = lambda: 30000
    
    context = MockContext()
    
    print("Testing API Gateway event:")
    response = lambda_handler(api_gateway_event, context)
    print(json.dumps(response, indent=2))
    
    print("\nTesting Function URL event:")
    response = lambda_handler(function_url_event, context)
    print(json.dumps(response, indent=2))